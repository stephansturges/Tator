"""Composition uses real SQLite/files; TarTransport is NOT encryption evidence."""

import hashlib
import errno
import io
import json
import shutil
import sqlite3
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from services.hosted.artifacts import ArtifactRef, ArtifactStore
from services.hosted.backup import (
    AgeTool, BackupError, ExportEntry, RestoreLimits, create_canonical_zip,
    create_encrypted_backup, restore_encrypted_backup,
)


class TarTransport:
    """Explicit test transport only. Production default always invokes age."""

    def check_available(self):
        return "TEST ONLY: noncryptographic archive transport"

    def encrypt_stream(self, recipient, output, write_archive):
        assert recipient
        write_archive(output)

    def decrypt_stream(self, identity, archive, read_archive):
        with open(archive, "rb") as stream:
            read_archive(stream)


def references(connection):
    return connection.execute("SELECT sha256,size_bytes FROM media")


def revoke(connection, manifest):
    assert manifest["format"] == "tator-hosted-recovery"
    connection.execute("DELETE FROM sessions")
    connection.execute("DELETE FROM editor_leases")
    connection.execute("UPDATE metadata SET value='restored-installation' WHERE key='installation_id'")


@pytest.fixture
def state(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    store = ArtifactStore(root / "artifacts")
    ref = store.put_bytes(b"immutable-image-pixels")
    database = root / "hosted.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript("""
            CREATE TABLE media(sha256 TEXT PRIMARY KEY,size_bytes INTEGER NOT NULL);
            CREATE TABLE versions(id TEXT PRIMARY KEY,content TEXT NOT NULL);
            CREATE TABLE assignments(image TEXT PRIMARY KEY,owner TEXT,state TEXT);
            CREATE TABLE sessions(id TEXT PRIMARY KEY);
            CREATE TABLE editor_leases(image TEXT PRIMARY KEY,tab TEXT);
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE receipts(id TEXT PRIMARY KEY,version TEXT);
        """)
        connection.execute("INSERT INTO media VALUES(?,?)", (ref.sha256, ref.size_bytes))
        connection.execute("INSERT INTO versions VALUES('v1',?)", (json.dumps({"boxes": [1], "caption": "draft", "alternates": ["alternate"]}),))
        connection.execute("INSERT INTO assignments VALUES('image1','annotator1','returned')")
        connection.execute("INSERT INTO sessions VALUES('old-session')")
        connection.execute("INSERT INTO editor_leases VALUES('image1','old-tab')")
        connection.execute("INSERT INTO metadata VALUES('installation_id','original-installation')")
        connection.execute("INSERT INTO receipts VALUES('operation1','v1')")
    # sqlite's context manager commits but does not close. Close explicitly so
    # the byte baseline below includes the WAL checkpoint before backup starts.
    connection.close()
    return database, store, ref


def make_backup(state, tmp_path, **kwargs):
    database, store, _ = state
    return create_encrypted_backup(database, store, tmp_path / "composed.testarchive", recipient="test-recipient", references=references, age=kwargs.pop("age", TarTransport()), **kwargs)


@pytest.mark.parametrize("reuse", [False, True])
def test_composed_restore_preserves_work_revokes_authority_and_continues(state, tmp_path, reuse):
    database, store, ref = state
    original = database.read_bytes()
    result = make_backup(state, tmp_path, metadata={"dataset": "fixture"})
    restored = restore_encrypted_backup(result.path, tmp_path / "restored", identity="test-only", restore_hook=revoke, age=TarTransport(), artifact_cache=store if reuse else None)
    assert restored.artifact_count == 1
    assert restored.metadata == {"dataset": "fixture"}
    assert ArtifactStore(restored.artifacts_root).verify(ref) == ref
    recovered_blob = ArtifactStore(restored.artifacts_root)._path(ref.sha256)
    assert (recovered_blob.stat().st_ino == store._path(ref.sha256).stat().st_ino) is reuse
    with sqlite3.connect(restored.database_path) as connection:
        assert connection.execute("SELECT * FROM sessions").fetchall() == []
        assert connection.execute("SELECT * FROM editor_leases").fetchall() == []
        assert connection.execute("SELECT owner,state FROM assignments").fetchone() == ("annotator1", "returned")
        assert connection.execute("SELECT version FROM receipts").fetchone() == ("v1",)
        assert connection.execute("SELECT value FROM metadata").fetchone() == ("restored-installation",)
        prior = json.loads(connection.execute("SELECT content FROM versions WHERE id='v1'").fetchone()[0])
        assert prior["caption"] == "draft" and prior["alternates"] == ["alternate"]
        connection.execute("INSERT INTO versions VALUES('v2',?)", (json.dumps({**prior, "caption": "continued after restore"}),))
    with sqlite3.connect(restored.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM versions").fetchone()[0] == 2
    assert database.read_bytes() == original
    assert store.verify(ref) == ref
    assert not list(store.staging.iterdir())
    assert not list(tmp_path.glob(".tator-restore-*"))


def test_capture_uses_snapshot_references_and_pins_survive_later_gc(state, tmp_path):
    database, store, old = state
    newer = store.put_bytes(b"new-admission")

    class MutationAfterCapture(TarTransport):
        def encrypt_stream(self, recipient, output, write_archive):
            with store.exclusive(), sqlite3.connect(database) as connection:
                connection.execute("DELETE FROM media")
                connection.execute("INSERT INTO media VALUES(?,?)", (newer.sha256, newer.size_bytes))
                connection.execute("INSERT INTO versions VALUES('v2','new live version')")
            assert store.delete_unreferenced(old.sha256, is_referenced=lambda _: False)
            super().encrypt_stream(recipient, output, write_archive)

    backup = make_backup(state, tmp_path, age=MutationAfterCapture())
    restored = restore_encrypted_backup(backup.path, tmp_path / "restored", identity="test-only", restore_hook=revoke, age=TarTransport())
    recovered = ArtifactStore(restored.artifacts_root)
    assert recovered.verify(old) == old
    with pytest.raises(FileNotFoundError):
        recovered.verify(newer)
    with sqlite3.connect(restored.database_path) as connection:
        assert connection.execute("SELECT sha256 FROM media").fetchall() == [(old.sha256,)]
        assert connection.execute("SELECT id FROM versions").fetchall() == [("v1",)]


def rewrite_archive(original, output, mutate):
    with tarfile.open(original, "r:") as source:
        members = [(member, source.extractfile(member).read()) for member in source.getmembers()]
    members = mutate(members)
    with tarfile.open(output, "w") as target:
        for info, content in members:
            target.addfile(info, io.BytesIO(content) if info.isreg() else None)


@pytest.mark.parametrize("case", ["traversal", "symlink", "duplicate", "missing", "corrupt", "extra"])
@pytest.mark.parametrize("reuse", [False, True])
def test_invalid_archives_never_publish_or_run_restore_hook(state, tmp_path, case, reuse):
    good = make_backup(state, tmp_path)
    bad = tmp_path / "bad.testarchive"

    def mutate(members):
        if case == "missing":
            return members[:-1]
        if case == "corrupt":
            info, content = members[-1]
            members[-1] = (info, b"X" * len(content))
            return members
        if case == "duplicate":
            return members + [members[-1]]
        extra = tarfile.TarInfo("../escaped" if case == "traversal" else "extra")
        if case == "symlink":
            extra.type = tarfile.SYMTYPE
            extra.linkname = "/etc/passwd"
        else:
            extra.size = 1
        return members + [(extra, b"x")]

    rewrite_archive(good.path, bad, mutate)
    calls = []
    with pytest.raises(BackupError):
        restore_encrypted_backup(bad, tmp_path / "restored", identity="test-only", restore_hook=lambda *args: calls.append(args), age=TarTransport(), artifact_cache=state[1] if reuse else None)
    assert calls == []
    assert not (tmp_path / "restored").exists()
    assert not (tmp_path / "escaped").exists()
    assert not list(tmp_path.glob(".tator-restore-*"))


def test_restore_rejects_limits_nonempty_root_and_failed_hook(state, tmp_path):
    backup = make_backup(state, tmp_path)
    destination = tmp_path / "restored"
    with pytest.raises(BackupError, match="limit"):
        restore_encrypted_backup(backup.path, destination, identity="test-only", restore_hook=revoke, age=TarTransport(), limits=RestoreLimits(max_total_bytes=1))
    destination.mkdir()
    protected = destination / "keep.txt"
    protected.write_text("keep")
    with pytest.raises(BackupError, match="empty"):
        restore_encrypted_backup(backup.path, destination, identity="test-only", restore_hook=revoke, age=TarTransport())
    assert protected.read_text() == "keep"
    protected.unlink()

    def fail_hook(connection, manifest):
        revoke(connection, manifest)
        raise RuntimeError("hook failed")

    with pytest.raises(RuntimeError, match="hook failed"):
        restore_encrypted_backup(backup.path, destination, identity="test-only", restore_hook=fail_hook, age=TarTransport())
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("reuse", [False, True])
def test_final_decryption_failure_is_not_published(state, tmp_path, reuse):
    backup = make_backup(state, tmp_path)

    class AuthenticationFailure(TarTransport):
        def decrypt_stream(self, identity, archive, read_archive):
            super().decrypt_stream(identity, archive, read_archive)
            raise BackupError("final authentication failed")

    calls = []
    with pytest.raises(BackupError, match="authentication"):
        restore_encrypted_backup(backup.path, tmp_path / "restored", identity="test-only", restore_hook=lambda *args: calls.append(args), age=AuthenticationFailure(), artifact_cache=state[1] if reuse else None)
    assert calls == []
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".tator-restore-*"))
    assert state[1].verify(state[2]) == state[2]


def test_restore_cache_mixes_verified_links_with_archive_only_members(state, tmp_path):
    database, source, first = state
    second = source.put_bytes(b"new artifact missing from the recovery cache")
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO media VALUES(?,?)", (second.sha256, second.size_bytes))
    backup = make_backup(state, tmp_path)
    cache = ArtifactStore(tmp_path / "cache")
    cache.put_bytes(b"immutable-image-pixels")
    restored = restore_encrypted_backup(backup.path, tmp_path / "restored", identity="test-only", restore_hook=revoke,
                                        age=TarTransport(), artifact_cache=cache)
    recovered = ArtifactStore(restored.artifacts_root)
    assert recovered.verify(first) == first and recovered.verify(second) == second
    assert recovered._path(first.sha256).stat().st_ino == cache._path(first.sha256).stat().st_ino
    with pytest.raises(FileNotFoundError):
        cache.verify(second)
    # Removing a cache entry through normal GC cannot remove the restored pin.
    assert cache.delete_unreferenced(first.sha256, is_referenced=lambda _: False)
    assert recovered.verify(first) == first
    assert source.verify(first) == first and source.verify(second) == second


@pytest.mark.parametrize("case", ["corrupt", "symlink", "cross_device"])
def test_restore_cache_fault_is_not_a_cache_miss(state, tmp_path, monkeypatch, case):
    backup = make_backup(state, tmp_path)
    ref = state[2]
    cache = ArtifactStore(tmp_path / "cache")
    cache.put_bytes(b"immutable-image-pixels")
    cached = cache._path(ref.sha256)
    if case == "corrupt":
        cached.chmod(0o600)
        cached.write_bytes(b"X" * ref.size_bytes)
    elif case == "symlink":
        cached.unlink()
        cached.symlink_to(state[1]._path(ref.sha256))
    else:
        import os
        original_link = os.link
        def cross_device(source, target, **kwargs):
            if Path(source) == cached:
                raise OSError(errno.EXDEV, "cross-device link")
            return original_link(source, target, **kwargs)
        monkeypatch.setattr(os, "link", cross_device)
    calls = []
    with pytest.raises((BackupError, OSError)):
        restore_encrypted_backup(backup.path, tmp_path / "restored", identity="test-only", restore_hook=lambda *args: calls.append(args),
                                 age=TarTransport(), artifact_cache=cache)
    assert calls == [] and not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".tator-restore-*"))
    assert state[1].verify(ref) == ref
    if case == "corrupt":
        assert cached.read_bytes() == b"X" * ref.size_bytes
    if case == "symlink":
        assert cached.is_symlink()


def test_extended_tar_header_is_rejected_before_reading_oversized_body(tmp_path):
    header = tarfile.TarInfo("pax-header")
    header.type = tarfile.XHDTYPE
    header.size = 2 ** 40

    class HeaderOnly:
        def __init__(self):
            self.first = True

        def read(self, size):
            if self.first:
                self.first = False
                return header.tobuf(format=tarfile.GNU_FORMAT)
            raise AssertionError("Parser attempted to read the oversized extension body")

    class MaliciousTransport(TarTransport):
        def decrypt_stream(self, identity, archive, read_archive):
            read_archive(HeaderOnly())

    with pytest.raises(BackupError, match="extended headers"):
        restore_encrypted_backup(tmp_path / "unused", tmp_path / "restored", identity="test-only", restore_hook=revoke, age=MaliciousTransport())
    assert not (tmp_path / "restored").exists()


def test_encryption_failure_cleans_pins_and_never_publishes_plaintext(state, tmp_path):
    class FailedEncryption(TarTransport):
        def encrypt_stream(self, recipient, output, write_archive):
            output.write(b"partial")
            raise BackupError("encrypt failed")

    with pytest.raises(BackupError, match="encrypt failed"):
        make_backup(state, tmp_path, age=FailedEncryption())
    assert not (tmp_path / "composed.testarchive").exists()
    assert not list(state[1].staging.iterdir())
    assert not list(tmp_path.glob(".tator-output-*"))


def test_age_is_required_without_plaintext_fallback(state, tmp_path):
    with pytest.raises(BackupError, match="age is required"):
        make_backup(state, tmp_path, age=AgeTool(str(tmp_path / "missing-age")))
    assert not (tmp_path / "composed.testarchive").exists()
    assert not list(state[1].staging.iterdir())


@pytest.mark.parametrize("symlink", [False, True])
def test_recovery_cli_rejects_missing_or_symlink_cache_before_initializing_it(state, tmp_path, capsys, symlink):
    from tools.hosted_recovery import main
    cache = tmp_path / "invalid-cache"
    if symlink:
        cache.symlink_to(state[1].root, target_is_directory=True)
    assert main(["restore", "--archive", str(tmp_path / "unused"), "--destination", str(tmp_path / "restored"),
                 "--identity", str(tmp_path / "unused-key"), "--reuse-artifacts", str(cache)]) == 1
    assert "existing real artifact-store root" in capsys.readouterr().err
    assert cache.is_symlink() if symlink else not cache.exists()
    assert not (tmp_path / "restored").exists()


def test_canonical_zip_uses_frozen_members_and_annotation_bytes(state, tmp_path):
    _, store, ref = state
    frozen = [
        ExportEntry("images/one.png", artifact=ref),
        ExportEntry("labels/one.txt", data=b"0 0.5 0.5 1 1\n"),
        ExportEntry("labelmap.txt", data=b"vehicle\n"),
    ]
    unrelated = store.put_bytes(b"incoming-not-canonical")
    output = create_canonical_zip(tmp_path / "canonical.zip", frozen, artifacts=store)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {entry.path for entry in frozen}
        assert archive.read("images/one.png") == b"immutable-image-pixels"
        assert archive.read("labels/one.txt") == frozen[1].data
        assert unrelated.sha256 not in archive.namelist()
    with pytest.raises(BackupError, match="already exists"):
        create_canonical_zip(output, frozen, artifacts=store)


@pytest.mark.parametrize("entries", [
    [ExportEntry("../escape", data=b"x")],
    [ExportEntry("/absolute", data=b"x")],
    [ExportEntry("x\\escape", data=b"x")],
    [ExportEntry("same", data=b"a"), ExportEntry("same", data=b"b")],
])
def test_canonical_export_rejects_unsafe_or_duplicate_members(state, tmp_path, entries):
    with pytest.raises(BackupError):
        create_canonical_zip(tmp_path / "bad.zip", entries, artifacts=state[1])
    assert not (tmp_path / "bad.zip").exists()


@pytest.mark.skipif(not shutil.which("age") or not shutil.which("age-keygen"), reason="Real age/age-keygen are required for cryptographic integration evidence")
@pytest.mark.parametrize("reuse", [False, True])
def test_real_age_encryption_roundtrip_and_wrong_identity(state, tmp_path, reuse):
    identity = tmp_path / "identity.txt"
    subprocess.run(["age-keygen", "-o", str(identity)], check=True, capture_output=True)
    recipient = subprocess.run(["age-keygen", "-y", str(identity)], check=True, text=True, capture_output=True).stdout.strip()
    output = tmp_path / "encrypted.age"
    result = create_encrypted_backup(state[0], state[1], output, recipient=recipient, references=references)
    assert output.read_bytes().startswith(b"age-encryption.org/v1")
    assert b"immutable-image-pixels" not in output.read_bytes()
    restored = restore_encrypted_backup(output, tmp_path / "restored", identity=identity, restore_hook=revoke, artifact_cache=state[1] if reuse else None)
    assert ArtifactStore(restored.artifacts_root).verify(state[2]) == state[2]
    other = tmp_path / "other.txt"
    subprocess.run(["age-keygen", "-o", str(other)], check=True, capture_output=True)
    with pytest.raises((BackupError, tarfile.TarError)):
        restore_encrypted_backup(result.path, tmp_path / "wrong-key", identity=other, restore_hook=revoke, artifact_cache=state[1] if reuse else None)
    assert not (tmp_path / "wrong-key").exists()
    damaged = tmp_path / "damaged.age"
    ciphertext = bytearray(output.read_bytes())
    ciphertext[-1] ^= 1
    damaged.write_bytes(ciphertext)
    with pytest.raises(BackupError):
        restore_encrypted_backup(damaged, tmp_path / "damaged-root", identity=identity, restore_hook=revoke, artifact_cache=state[1] if reuse else None)
    assert not (tmp_path / "damaged-root").exists()
