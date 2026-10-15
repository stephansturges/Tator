"""Durable immutable objects for hosted Tator.

Callers authorize a dataset/image before opening an object. A digest is an
identity, never an authorization token. Reference commits and garbage collection
must acquire ``exclusive()`` before their SQLite transaction.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import os
import re
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

CHUNK_SIZE = 1024 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_LOCKS: dict[str, tuple[threading.RLock, threading.local]] = {}
_LOCKS_GUARD = threading.Lock()


def _after_fork() -> None:
    global _LOCKS, _LOCKS_GUARD
    _LOCKS = {}
    _LOCKS_GUARD = threading.Lock()


os.register_at_fork(after_in_child=_after_fork)


class ArtifactError(ValueError):
    """An object identity, file or integrity check is invalid."""


def validate_digest(value: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ArtifactError("Expected a lowercase SHA-256 digest.")
    return value


@dataclass(frozen=True)
class ArtifactRef:
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        validate_digest(self.sha256)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ArtifactError("Artifact size must be a nonnegative integer.")

    def as_dict(self) -> dict[str, str | int]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _directory(path: Path) -> None:
    """Create one owned directory; refuse a pre-existing symlink."""
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ArtifactError(f"Storage directory is not a real directory: {path.name}")


def open_regular(path: Path) -> BinaryIO:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ArtifactError("Expected a regular artifact file.")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def hash_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


class ArtifactStore:
    def __init__(self, root: str | Path):
        # The configured root is trusted; all object identities below it are
        # generated/validated here and can never supply filesystem paths.
        self.root = Path(root).absolute()
        self.root.parent.mkdir(parents=True, exist_ok=True)
        _directory(self.root)
        self.root = self.root.resolve()
        self.blobs = self.root / "blobs"
        _directory(self.blobs)
        self.sha256_root = self.blobs / "sha256"
        _directory(self.sha256_root)
        self.staging = self.root / ".staging"
        _directory(self.staging)
        fsync_directory(self.root.parent)
        fsync_directory(self.root)
        fsync_directory(self.blobs)
        self._lock_pid = 0
        self._process_lock()

    def _process_lock(self) -> None:
        if self._lock_pid != os.getpid():
            with _LOCKS_GUARD:
                self._lock, self._local = _LOCKS.setdefault(
                    str(self.root), (threading.RLock(), threading.local())
                )
            self._lock_pid = os.getpid()

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        """Lock order: artifact store, then SQLite transaction.

        A backup holds this lock while capturing SQLite and pinning its exact
        references. Never acquire it while already holding a write transaction.
        Reentrancy is supported across instances for the same root/thread.
        """
        self._process_lock()
        with self._lock:
            depth = getattr(self._local, "depth", 0)
            if depth:
                self._local.depth += 1
                try:
                    yield
                finally:
                    self._local.depth -= 1
                return
            fd = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            owner_pid = os.getpid()
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ArtifactError("Artifact lock is not a regular file.")
                fcntl.flock(fd, fcntl.LOCK_EX)
                self._local.depth = 1
                try:
                    yield
                finally:
                    if owner_pid == os.getpid():
                        self._local.depth = 0
                        fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _path(self, digest: str, *, create: bool = False) -> Path:
        validate_digest(digest)
        for directory in (self.blobs, self.sha256_root):
            if not stat.S_ISDIR(directory.lstat().st_mode):
                raise ArtifactError("Artifact directory was replaced.")
        shard = self.sha256_root / digest[:2]
        if create:
            _directory(shard)
            fsync_directory(self.sha256_root)
        if not stat.S_ISDIR(shard.lstat().st_mode):
            raise ArtifactError("Artifact shard is not a real directory.")
        return shard / digest

    @contextlib.contextmanager
    def open(self, reference: ArtifactRef | str, *, verify: bool = True) -> Iterator[BinaryIO]:
        digest = reference.sha256 if isinstance(reference, ArtifactRef) else validate_digest(reference)
        with open_regular(self._path(digest)) as stream:
            if verify:
                actual_digest, size = hash_stream(stream)
                if actual_digest != digest or (isinstance(reference, ArtifactRef) and size != reference.size_bytes):
                    raise ArtifactError("Artifact hash or size mismatch.")
                stream.seek(0)
            yield stream

    def verify(self, reference: ArtifactRef | str) -> ArtifactRef:
        digest = reference.sha256 if isinstance(reference, ArtifactRef) else validate_digest(reference)
        with self.open(reference) as stream:
            return ArtifactRef(digest, os.fstat(stream.fileno()).st_size)

    def put_stream(self, stream: BinaryIO, *, expected_sha256: str | None = None,
                   expected_size: int | None = None, max_bytes: int | None = None) -> ArtifactRef:
        if expected_sha256 is not None:
            validate_digest(expected_sha256)
        for value in (expected_size, max_bytes):
            if value is not None and (type(value) is not int or value < 0):
                raise ArtifactError("Byte limits must be nonnegative integers.")
        if not stat.S_ISDIR(self.staging.lstat().st_mode):
            raise ArtifactError("Artifact staging is not a real directory.")
        fd, temporary = tempfile.mkstemp(prefix="object-", dir=self.staging)
        stage = Path(temporary)
        try:
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(fd, "wb") as output:
                while chunk := stream.read(CHUNK_SIZE):
                    if not isinstance(chunk, bytes):
                        raise ArtifactError("Artifact input must be a binary stream.")
                    size += len(chunk)
                    if max_bytes is not None and size > max_bytes:
                        raise ArtifactError("Artifact exceeds the permitted size.")
                    output.write(chunk)
                    digest.update(chunk)
                if expected_size is not None and size != expected_size:
                    raise ArtifactError("Artifact input size mismatch.")
                checksum = digest.hexdigest()
                if expected_sha256 is not None and checksum != expected_sha256:
                    raise ArtifactError("Artifact input hash mismatch.")
                output.flush()
                os.fchmod(output.fileno(), 0o400)
                os.fsync(output.fileno())
            reference = ArtifactRef(checksum, size)
            with self.exclusive():
                target = self._path(checksum, create=True)
                try:
                    # Linking is atomic and cannot replace an existing object.
                    os.link(stage, target, follow_symlinks=False)
                except FileExistsError:
                    self.verify(reference)
                fsync_directory(target.parent)
            return reference
        finally:
            stage.unlink(missing_ok=True)
            fsync_directory(self.staging)

    def put_bytes(self, data: bytes, **kwargs: object) -> ArtifactRef:
        return self.put_stream(io.BytesIO(data), **kwargs)

    def put_file(self, source: str | Path, **kwargs: object) -> ArtifactRef:
        # This is an administrative import interface, not a public path API.
        # It reads the original through a no-follow descriptor and never edits it.
        with open_regular(Path(source)) as stream:
            return self.put_stream(stream, **kwargs)

    def pin(self, reference: ArtifactRef | str, directory: str | Path) -> ArtifactRef:
        """Create a hard-link pin in a private, caller-owned staging directory."""
        directory = Path(directory)
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise ArtifactError("Pin destination is not a real directory.")
        with self.exclusive():
            checked = self.verify(reference)
            target = directory / checked.sha256
            try:
                os.link(self._path(checked.sha256), target, follow_symlinks=False)
            except FileExistsError:
                with open_regular(target) as existing:
                    digest, size = hash_stream(existing)
                if (digest, size) != (checked.sha256, checked.size_bytes):
                    raise ArtifactError("Existing pin does not match its artifact.")
            fsync_directory(directory)
            return checked

    def delete_unreferenced(self, digest: str, *, is_referenced: Callable[[str], bool]) -> bool:
        """Low-level GC primitive; caller must also enforce purge/retention policy.

        The reference check executes under the shared backup/reference lock.
        Existing hard-link backup pins survive removal of the store entry.
        """
        validate_digest(digest)
        with self.exclusive():
            if is_referenced(digest):
                return False
            try:
                target = self._path(digest)
                with open_regular(target):
                    pass
                target.unlink()
            except FileNotFoundError:
                return False
            fsync_directory(target.parent)
            return True
