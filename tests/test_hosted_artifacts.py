import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from services.hosted.artifacts import CHUNK_SIZE, ArtifactError, ArtifactRef, ArtifactStore


def test_streamed_import_is_durable_deduplicated_and_preserves_source(tmp_path):
    source = tmp_path / "original.bin"
    payload = b"pixels" * (CHUNK_SIZE // 2)
    source.write_bytes(payload)
    original_stat = source.stat()
    store = ArtifactStore(tmp_path / "objects")

    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= CHUNK_SIZE
            return super().read(size)

    reference = store.put_stream(BoundedReader(payload))
    assert reference == ArtifactRef(hashlib.sha256(payload).hexdigest(), len(payload))
    with store.exclusive():
        # Reentrant across distinct store instances sharing the root.
        assert ArtifactStore(store.root).put_file(source) == reference
    with ThreadPoolExecutor(max_workers=3) as executor:
        assert list(executor.map(lambda _: store.put_bytes(payload), range(3))) == [reference] * 3
    with store.open(reference) as stream:
        assert stream.read() == payload
    assert source.read_bytes() == payload
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert source.stat().st_mode == original_stat.st_mode
    assert not list(store.staging.iterdir())
    assert len(list(store.sha256_root.glob("*/*"))) == 1


@pytest.mark.parametrize("kwargs", [
    {"expected_sha256": "0" * 64}, {"expected_size": 100}, {"max_bytes": 2},
])
def test_bad_input_never_publishes_and_cleans_stage(tmp_path, kwargs):
    store = ArtifactStore(tmp_path / "objects")
    with pytest.raises(ArtifactError):
        store.put_bytes(b"actual", **kwargs)
    assert not list(store.staging.iterdir())
    assert not list(store.sha256_root.glob("*/*"))


def test_publish_failure_keeps_existing_objects_and_can_retry(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "objects")
    existing = store.put_bytes(b"existing")
    actual_link = os.link

    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "link", disk_full)
    with pytest.raises(OSError, match="No space"):
        store.put_bytes(b"new")
    assert store.verify(existing) == existing
    assert not list(store.staging.iterdir())
    monkeypatch.setattr(os, "link", actual_link)
    assert store.put_bytes(b"new").size_bytes == 3


@pytest.mark.parametrize("digest", ["../escape", "/etc/passwd", "A" * 64, "0" * 63, "0" * 64 + "/more"])
def test_object_identity_never_accepts_paths(tmp_path, digest):
    store = ArtifactStore(tmp_path / "objects")
    with pytest.raises(ArtifactError):
        with store.open(digest):
            pass


def test_store_rejects_symlinks_and_detects_corruption(tmp_path):
    store = ArtifactStore(tmp_path / "objects")
    reference = store.put_bytes(b"original")
    target = store.sha256_root / reference.sha256[:2] / reference.sha256
    os.chmod(target, 0o600)
    target.write_bytes(b"modified")
    with pytest.raises(ArtifactError, match="mismatch"):
        store.verify(reference)
    with pytest.raises(ArtifactError, match="mismatch"):
        store.put_bytes(b"original")
    target.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"original")
    target.symlink_to(outside)
    with pytest.raises(OSError):
        store.verify(reference)
    assert outside.read_bytes() == b"original"


def test_reference_check_and_backup_pin_protect_bytes(tmp_path):
    store = ArtifactStore(tmp_path / "objects")
    ref = store.put_bytes(b"retained-history")
    pins = tmp_path / "pins"
    pins.mkdir()
    store.pin(ref, pins)
    assert not store.delete_unreferenced(ref.sha256, is_referenced=lambda _: True)
    assert store.delete_unreferenced(ref.sha256, is_referenced=lambda _: False)
    assert (pins / ref.sha256).read_bytes() == b"retained-history"
    with pytest.raises(FileNotFoundError):
        store.verify(ref)


def test_existing_symlink_directory_is_never_followed(tmp_path):
    root = tmp_path / "objects"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "blobs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactError):
        ArtifactStore(root)
    assert not list(outside.iterdir())
