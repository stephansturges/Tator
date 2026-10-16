import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from services.hosted.artifacts import CHUNK_SIZE, ArtifactError, ArtifactRef, ArtifactStore


@pytest.mark.parametrize("encoding", ["JPEG", "PNG", "WEBP", "TIFF"])
@pytest.mark.parametrize("orientation", range(1, 9))
def test_encoded_working_media_preserves_all_orientations_and_native_pixels(tmp_path, encoding, orientation):
    from PIL import Image
    from services.hosted.media import decode_encoded, working_media_bytes
    from services.hosted.adapters import _image
    picture = Image.new("RGB", (64, 48), "red")
    picture.paste("blue", (0, 0, 24, 48))
    exif = Image.Exif(); exif[274] = orientation
    stream = io.BytesIO(); picture.save(stream, format=encoding, exif=exif)
    original = stream.getvalue()
    pixels, info = decode_encoded(original)
    working, view = working_media_bytes(original)
    assert info.width == 64 and info.height == 48 and info.orientation == orientation
    assert view["source_sha256"] == hashlib.sha256(original).hexdigest()
    with Image.open(io.BytesIO(working)) as neutral:
        assert neutral.size == (64, 48) and neutral.getexif().get(274, 1) == 1
        assert neutral.convert("RGB").tobytes() == pixels.tobytes()
    path = tmp_path / "working"; path.write_bytes(working)
    native, digest = _image({"media_path": str(path), "media_sha256": view["sha256"]}, {"media_root": str(tmp_path)})
    assert native.tobytes() == pixels.tobytes() and digest == view["sha256"]
    assert pixels.getpixel((8, 20))[2] > 240 and pixels.getpixel((48, 20))[0] > 240
    assert (working != original) == (orientation != 1 or encoding == "TIFF")


def test_full_media_decode_rejects_truncation_and_shared_limits_before_retention(tmp_path, monkeypatch):
    from PIL import Image
    from services.hosted import media
    from services.hosted.errors import DomainError
    from services.hosted.store import HostedStore
    stream = io.BytesIO(); Image.new("RGB", (64, 48), "orange").save(stream, format="JPEG", quality=95)
    data = stream.getvalue(); truncated = data[:-40]
    with Image.open(io.BytesIO(truncated)) as header:
        header.verify()  # The original regression: header validation succeeds.
    store = HostedStore(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    service = media.HostedMedia(store, artifacts)
    with pytest.raises(DomainError, match="invalid_image_import"):
        service.ingest(truncated)
    assert not list(artifacts.sha256_root.glob("*/*"))
    monkeypatch.setattr(media, "MAX_MEDIA_BYTES", len(data) - 1)
    with pytest.raises(DomainError, match="hosted_image_bytes_exceeded"):
        service.ingest(data)
    monkeypatch.setattr(media, "MAX_MEDIA_BYTES", len(data))
    monkeypatch.setattr(media, "MAX_MEDIA_PIXELS", 64 * 48 - 1)
    with pytest.raises(DomainError, match="hosted_image_pixels_exceeded"):
        service.ingest(data)
    monkeypatch.setattr(media, "MAX_MEDIA_PIXELS", 64 * 48)
    assert service.ingest(data)["source_sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("orientation", range(1, 9))
def test_explicit_display_label_conversion_matches_encoded_boxes_and_polygons(orientation):
    from services.hosted.media import MediaInfo, prepare_annotations
    from services.hosted.errors import DomainError
    width, height = 64, 48
    forward = {1: lambda x, y: (x, y), 2: lambda x, y: (width-x, y),
        3: lambda x, y: (width-x, height-y), 4: lambda x, y: (x, height-y),
        5: lambda x, y: (y, x), 6: lambda x, y: (height-y, x),
        7: lambda x, y: (height-y, width-x), 8: lambda x, y: (y, width-x)}[orientation]
    encoded = [[4, 8], [20, 8], [20, 32], [4, 32]]
    displayed = [forward(*point) for point in encoded]
    xs, ys = zip(*displayed)
    annotations = {"objects": [{"bbox": [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)], "polygon": displayed}]}
    info = MediaInfo("a" * 64, 100, width, height, "JPEG", orientation, orientation != 1)
    result = prepare_annotations(annotations, info, "exif_display")
    assert result["objects"][0] == {"bbox": [4, 8, 16, 24], "polygon": encoded}
    assert result["media_coordinate_provenance"]["source_coordinate_space"] == "exif_display"
    assert prepare_annotations(result, info, "encoded")["objects"] == result["objects"]
    if orientation != 1:
        with pytest.raises(DomainError, match="label_coordinate_space_required"):
            prepare_annotations(annotations, info)
    with pytest.raises(DomainError, match="invalid_bbox"):
        prepare_annotations({"objects": [{"bbox": [0, 0, -2, 2]}]}, info, "exif_display")


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
