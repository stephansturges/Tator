"""Encrypted recovery archives and frozen canonical dataset exports.

Full recovery archives always use external age. Tests may inject an explicit
transport double; the production default never falls back to plaintext.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterable, Mapping

from .artifacts import (
    CHUNK_SIZE, ArtifactError, ArtifactRef, ArtifactStore, fsync_directory,
    hash_stream, open_regular,
)

FORMAT = "tator-hosted-recovery"
FORMAT_VERSION = 1


class BackupError(ValueError):
    pass


class _RecoveryTarInfo(tarfile.TarInfo):
    def _proc_member(self, archive):
        # Reject extension/sparse headers BEFORE tarfile reads their claimed
        # bodies. Otherwise a malicious PAX/longname header can force an
        # unbounded allocation before member-level validation is reached.
        if self.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
            raise BackupError("Recovery archives permit ordinary regular members only; extended headers are forbidden.")
        safe_archive_path(self.name)
        return super()._proc_member(archive)


@dataclass(frozen=True)
class RestoreLimits:
    max_manifest_bytes: int = 32 * 1024 * 1024
    max_files: int = 1_000_000
    max_file_bytes: int = 1024 ** 4
    max_total_bytes: int = 8 * 1024 ** 4


@dataclass(frozen=True)
class BackupResult:
    path: Path
    sha256: str
    size_bytes: int
    artifact_count: int
    database_sha256: str


@dataclass(frozen=True)
class RestoreResult:
    root: Path
    database_path: Path
    artifacts_root: Path
    artifact_count: int
    source_database_sha256: str
    metadata: dict


@dataclass(frozen=True)
class ExportEntry:
    path: str
    artifact: ArtifactRef | None = None
    data: bytes | None = None


def safe_archive_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise BackupError("Invalid archive member path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise BackupError("Archive paths must be normalized relative paths.")
    if ":" in path.parts[0] or len(value.encode("utf-8")) > 4096:
        raise BackupError("Invalid archive member path.")
    return value


class AgeTool:
    """Invoke age without shell expansion; keys never enter archive metadata."""

    def __init__(self, executable: str = "age"):
        self.executable = executable

    def check_available(self) -> str:
        resolved = shutil.which(self.executable)
        if not resolved:
            raise BackupError("age is required for encrypted recovery backups; no plaintext fallback is available.")
        try:
            result = subprocess.run([resolved, "--version"], capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            raise BackupError("age could not be started.") from error
        return result.stdout.strip() or result.stderr.strip()

    def encrypt_stream(self, recipient: str, output: BinaryIO,
                       write_archive: Callable[[BinaryIO], None]) -> None:
        self.check_available()
        if not isinstance(recipient, str) or not recipient.strip() or recipient.startswith("-") or "\n" in recipient or "\r" in recipient:
            raise BackupError("A valid age recipient is required.")
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                [self.executable, "--encrypt", "--recipient", recipient],
                stdin=subprocess.PIPE, stdout=output, stderr=errors,
            )
            try:
                assert process.stdin is not None
                write_archive(process.stdin)
                process.stdin.close()
                code = process.wait()
                if code:
                    raise BackupError(self._failure(errors, "age encryption failed"))
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
            finally:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()

    def decrypt_stream(self, identity: str | Path, archive: Path,
                       read_archive: Callable[[BinaryIO], None]) -> None:
        self.check_available()
        identity = Path(identity).absolute()
        with open_regular(archive) as encrypted, tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                [self.executable, "--decrypt", "--identity", str(identity)],
                stdin=encrypted, stdout=subprocess.PIPE, stderr=errors,
            )
            try:
                assert process.stdout is not None
                read_archive(process.stdout)
                # Consume to EOF before accepting: age authenticates the final
                # chunk even when tar has already reached its end marker.
                while process.stdout.read(CHUNK_SIZE):
                    pass
                code = process.wait()
                if code:
                    raise BackupError(self._failure(errors, "age decryption failed"))
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
            finally:
                if process.stdout:
                    process.stdout.close()

    @staticmethod
    def _failure(errors: BinaryIO, message: str) -> str:
        errors.seek(0)
        detail = errors.read(16 * 1024).decode("utf-8", errors="replace").strip()
        return f"{message}: {detail}" if detail else message


def _reference(value: object) -> ArtifactRef | str:
    if isinstance(value, (ArtifactRef, str)):
        return value
    # sqlite3.Row supports key lookup but is not a Mapping subclass.
    try:
        return ArtifactRef(value["sha256"], value["size_bytes"])  # type: ignore[index]
    except (TypeError, KeyError, IndexError) as error:
        raise BackupError("Artifact enumeration must return references, digests or sha256/size_bytes rows.") from error


def _integrity(connection: sqlite3.Connection) -> None:
    rows = [tuple(row) for row in connection.execute("PRAGMA integrity_check")]
    if rows != [("ok",)]:
        raise BackupError("SQLite integrity check failed.")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise BackupError("SQLite foreign-key check failed.")


def _file_ref(path: Path) -> ArtifactRef:
    with open_regular(path) as stream:
        digest, size = hash_stream(stream)
    return ArtifactRef(digest, size)


def _publish_file(stage: Path, destination: Path) -> None:
    # Do not replace a prior recovery archive or follow a destination symlink.
    os.link(stage, destination, follow_symlinks=False)
    fsync_directory(destination.parent)


def _output_stage(destination: str | Path) -> tuple[Path, Path, int]:
    destination = Path(destination).absolute()
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise BackupError("Output parent must be an existing real directory.")
    if destination.exists() or destination.is_symlink():
        raise BackupError("Output already exists; choose a new filename.")
    fd, name = tempfile.mkstemp(prefix=".tator-output-", dir=destination.parent)
    return destination, Path(name), fd


def _tar_file(archive: tarfile.TarFile, name: str, source: Path, size: int) -> None:
    info = tarfile.TarInfo(safe_archive_path(name))
    info.size = size
    info.mode = 0o600
    info.mtime = 0
    with open_regular(source) as stream:
        archive.addfile(info, stream)


def create_encrypted_backup(
    database_path: str | Path,
    artifacts: ArtifactStore,
    destination: str | Path,
    *,
    recipient: str,
    references: Callable[[sqlite3.Connection], Iterable[object]],
    age: AgeTool | None = None,
    metadata: Mapping | None = None,
) -> BackupResult:
    """Capture one consistent DB plus its immutable references, then encrypt.

    ``references`` MUST enumerate from the supplied captured connection, never
    the live DB. Include registered custom weights/configuration as well as media.
    Writers that introduce/remove references and GC must use artifacts.exclusive.
    """
    age = age or AgeTool()
    age.check_available()  # Fail before creating any plaintext capture.
    destination, encrypted_stage, fd = _output_stage(destination)
    capture_root: Path | None = None
    try:
        capture_root = Path(tempfile.mkdtemp(prefix="backup-", dir=artifacts.staging))
        pins = capture_root / "pins"
        pins.mkdir(mode=0o700)
        snapshot = capture_root / "database.sqlite3"
        with artifacts.exclusive():
            source_path = Path(database_path).absolute()
            if source_path.is_symlink() or not source_path.is_file():
                raise BackupError("Backup source must be an existing regular SQLite database.")
            with contextlib.closing(sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)) as source:
                with contextlib.closing(sqlite3.connect(snapshot)) as captured:
                    source.backup(captured, pages=256)
                    captured.commit()
                    _integrity(captured)
                    captured.row_factory = sqlite3.Row
                    pinned: dict[str, ArtifactRef] = {}
                    for value in references(captured):
                        reference = _reference(value)
                        ref = artifacts.pin(reference, pins)
                        previous = pinned.get(ref.sha256)
                        if previous is not None and previous != ref:
                            raise BackupError("Conflicting artifact reference sizes.")
                        pinned[ref.sha256] = ref
            with open_regular(snapshot) as handle:
                os.fsync(handle.fileno())
            fsync_directory(capture_root)
        database = _file_ref(snapshot)
        manifest = {
            "format": FORMAT, "version": FORMAT_VERSION, "created_at": time.time(),
            "database": database.as_dict(),
            "artifacts": [ref.as_dict() for ref in sorted(pinned.values(), key=lambda ref: ref.sha256)],
            "metadata": dict(metadata or {}),
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        _manifest(manifest, RestoreLimits())
        if len(manifest_bytes) > RestoreLimits().max_manifest_bytes:
            raise BackupError("Backup manifest exceeds the supported restore limit.")

        def write_archive(stream: BinaryIO) -> None:
            # Fixed short member names need no extended headers. GNU's base-256
            # size field also supports custom weights larger than 8 GiB.
            with tarfile.open(fileobj=stream, mode="w|", format=tarfile.GNU_FORMAT) as archive:
                info = tarfile.TarInfo("manifest.json")
                info.size = len(manifest_bytes)
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(manifest_bytes))
                _tar_file(archive, "database.sqlite3", snapshot, database.size_bytes)
                for ref in pinned.values():
                    _tar_file(archive, f"artifacts/{ref.sha256}", pins / ref.sha256, ref.size_bytes)

        with os.fdopen(fd, "wb") as output:
            fd = -1
            age.encrypt_stream(recipient, output, write_archive)
            output.flush()
            os.fsync(output.fileno())
        encrypted = _file_ref(encrypted_stage)
        _publish_file(encrypted_stage, destination)
        return BackupResult(destination, encrypted.sha256, encrypted.size_bytes, len(pinned), database.sha256)
    finally:
        if fd >= 0:
            os.close(fd)
        encrypted_stage.unlink(missing_ok=True)
        fsync_directory(destination.parent)
        if capture_root is not None:
            shutil.rmtree(capture_root)
            fsync_directory(artifacts.staging)


def _manifest(value: object, limits: RestoreLimits) -> tuple[ArtifactRef, dict[str, ArtifactRef], dict]:
    if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("version") != FORMAT_VERSION:
        raise BackupError("Unsupported recovery archive format.")
    try:
        database = ArtifactRef(**value["database"])
        rows = value["artifacts"]
        if not isinstance(rows, list) or len(rows) + 1 > limits.max_files:
            raise BackupError("Recovery archive file limit exceeded.")
        refs = {}
        for row in rows:
            ref = ArtifactRef(**row)
            if ref.sha256 in refs:
                raise BackupError("Duplicate artifact in recovery manifest.")
            refs[ref.sha256] = ref
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise BackupError("Recovery metadata must be an object.")
    except (KeyError, TypeError, ArtifactError) as error:
        raise BackupError("Malformed recovery manifest.") from error
    values = [database, *refs.values()]
    if any(ref.size_bytes > limits.max_file_bytes for ref in values) or sum(ref.size_bytes for ref in values) > limits.max_total_bytes:
        raise BackupError("Recovery archive size limit exceeded.")
    return database, refs, metadata


def restore_encrypted_backup(
    archive_path: str | Path,
    destination_root: str | Path,
    *,
    identity: str | Path,
    restore_hook: Callable[[sqlite3.Connection, dict], None],
    age: AgeTool | None = None,
    database_name: str = "hosted.sqlite3",
    limits: RestoreLimits | None = None,
) -> RestoreResult:
    """Verify/decrypt in isolation, revoke authority via hook, publish empty root.

    The mandatory hook must revoke sessions, auth links, editor leases; rotate
    installation identity; and reconcile interrupted jobs. Assignments/history
    remain. It runs before the new root becomes visible and inside a transaction.
    """
    if not callable(restore_hook):
        raise BackupError("A restore authority-revocation hook is required.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", database_name) or database_name == "artifacts":
        raise BackupError("Database name must be a simple filename.")
    age = age or AgeTool()
    age.check_available()
    limits = limits or RestoreLimits()
    destination = Path(destination_root).absolute()
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise BackupError("Restore parent must be an existing real directory.")
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise BackupError("Restore destination must be absent or empty.")
    stage = Path(tempfile.mkdtemp(prefix=".tator-restore-", dir=destination.parent))
    manifest: dict = {}
    database_ref: ArtifactRef | None = None
    refs: dict[str, ArtifactRef] = {}
    metadata: dict = {}
    published = False
    try:
        restored_store = ArtifactStore(stage / "artifacts")

        def read_archive(stream: BinaryIO) -> None:
            nonlocal manifest, database_ref, refs, metadata
            with tarfile.open(fileobj=stream, mode="r|", tarinfo=_RecoveryTarInfo) as archive:
                first = archive.next()
                if first is None or first.name != "manifest.json" or not first.isreg() or first.size > limits.max_manifest_bytes:
                    raise BackupError("Recovery manifest must be the first bounded regular member.")
                manifest_stream = archive.extractfile(first)
                assert manifest_stream is not None
                try:
                    manifest = json.loads(manifest_stream.read(limits.max_manifest_bytes + 1))
                except (ValueError, UnicodeError) as error:
                    raise BackupError("Recovery manifest is not valid JSON.") from error
                database_ref, refs, metadata = _manifest(manifest, limits)
                expected = {"database.sqlite3": database_ref, **{f"artifacts/{digest}": ref for digest, ref in refs.items()}}
                seen: set[str] = set()
                for member in archive:
                    # TarFile iteration includes its already-returned first member.
                    if member is first:
                        continue
                    name = safe_archive_path(member.name)
                    if not member.isreg() or name not in expected or name in seen:
                        raise BackupError("Unexpected, duplicated or nonregular recovery member.")
                    ref = expected[name]
                    if member.size != ref.size_bytes:
                        raise BackupError("Recovery member size disagrees with its manifest.")
                    source = archive.extractfile(member)
                    if source is None:
                        raise BackupError("Recovery member has no file content.")
                    if name.startswith("artifacts/"):
                        restored_store.put_stream(source, expected_sha256=ref.sha256, expected_size=ref.size_bytes, max_bytes=limits.max_file_bytes)
                    else:
                        target = stage / database_name
                        digest = hashlib.sha256()
                        copied = 0
                        with target.open("xb") as output:
                            os.chmod(target, 0o600)
                            while chunk := source.read(CHUNK_SIZE):
                                copied += len(chunk)
                                digest.update(chunk)
                                output.write(chunk)
                            output.flush()
                            os.fsync(output.fileno())
                        if copied != ref.size_bytes or digest.hexdigest() != ref.sha256:
                            raise BackupError("Recovery database hash or size mismatch.")
                    seen.add(name)
                if seen != set(expected):
                    raise BackupError("Recovery archive is missing referenced files.")

        age.decrypt_stream(identity, Path(archive_path).absolute(), read_archive)
        database_path = stage / database_name
        with contextlib.closing(sqlite3.connect(database_path)) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            _integrity(connection)
            with connection:
                restore_hook(connection, manifest)
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            _integrity(connection)
        with open_regular(database_path) as database_handle:
            os.fsync(database_handle.fileno())
        fsync_directory(stage)
        # POSIX rename replaces an empty directory only, never a populated root.
        os.rename(stage, destination)
        published = True
        fsync_directory(destination.parent)
        assert database_ref is not None
        return RestoreResult(destination, destination / database_name, destination / "artifacts", len(refs), database_ref.sha256, metadata)
    except (tarfile.TarError, EOFError, ArtifactError) as error:
        raise BackupError(f"Recovery verification failed: {error}") from error
    finally:
        if not published:
            shutil.rmtree(stage)


def create_canonical_zip(destination: str | Path, entries: Iterable[ExportEntry], *,
                         artifacts: ArtifactStore) -> Path:
    """Export an already-frozen canonical snapshot; never enumerate live data.

    The caller supplies version-bound annotation bytes/labelmap and artifact
    references from that snapshot, and enforces administrator authorization.
    """
    destination, stage, fd = _output_stage(destination)
    names: set[str] = set()
    try:
        with os.fdopen(fd, "wb") as output:
            with artifacts.exclusive(), zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for entry in entries:
                    name = safe_archive_path(entry.path)
                    if name in names or (entry.artifact is None) == (entry.data is None):
                        raise BackupError("Each export member needs a unique path and exactly one content source.")
                    names.add(name)
                    if entry.data is not None:
                        if not isinstance(entry.data, bytes):
                            raise BackupError("Frozen export content must be bytes.")
                        archive.writestr(name, entry.data)
                    else:
                        assert entry.artifact is not None
                        with artifacts.open(entry.artifact) as source, archive.open(name, "w", force_zip64=True) as target:
                            shutil.copyfileobj(source, target, length=CHUNK_SIZE)
            output.flush()
            os.fsync(output.fileno())
        _publish_file(stage, destination)
        return destination
    finally:
        stage.unlink(missing_ok=True)
        fsync_directory(destination.parent)
