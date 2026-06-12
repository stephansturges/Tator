from __future__ import annotations

import json
import stat
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

import localinferenceapi as api
from localinferenceapi import _extract_zip_safely_impl


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def test_extract_zip_safely_rejects_path_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    zip_path.write_bytes(_zip_bytes({"../escape.txt": b"nope"}))

    with zipfile.ZipFile(zip_path, "r") as zf, pytest.raises(HTTPException) as exc_info:
        _extract_zip_safely_impl(
            zf,
            tmp_path / "extract",
            max_entry_bytes=1024,
            max_total_uncompressed_bytes=2048,
            traversal_detail="dataset_zip_path_traversal",
            symlink_detail="dataset_zip_symlink_unsupported",
            entry_too_large_detail="dataset_zip_entry_too_large",
            total_too_large_detail="dataset_zip_uncompressed_too_large",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_zip_path_traversal"


@pytest.mark.parametrize("member_name", ["C:/escape.txt", "\\\\server\\share\\escape.txt"])
def test_extract_zip_safely_rejects_windows_absolute_members(
    tmp_path: Path, member_name: str
) -> None:
    zip_path = tmp_path / "bad_windows.zip"
    zip_path.write_bytes(_zip_bytes({member_name: b"nope"}))

    with zipfile.ZipFile(zip_path, "r") as zf, pytest.raises(HTTPException) as exc_info:
        _extract_zip_safely_impl(
            zf,
            tmp_path / "extract",
            max_entry_bytes=1024,
            max_total_uncompressed_bytes=2048,
            traversal_detail="dataset_zip_path_traversal",
            symlink_detail="dataset_zip_symlink_unsupported",
            entry_too_large_detail="dataset_zip_entry_too_large",
            total_too_large_detail="dataset_zip_uncompressed_too_large",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_zip_path_traversal"


def test_extract_zip_safely_rejects_symlink_members(tmp_path: Path) -> None:
    zip_path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("dataset/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, b"../../outside")

    with zipfile.ZipFile(zip_path, "r") as zf, pytest.raises(HTTPException) as exc_info:
        _extract_zip_safely_impl(
            zf,
            tmp_path / "extract",
            max_entry_bytes=1024,
            max_total_uncompressed_bytes=2048,
            traversal_detail="dataset_zip_path_traversal",
            symlink_detail="dataset_zip_symlink_unsupported",
            entry_too_large_detail="dataset_zip_entry_too_large",
            total_too_large_detail="dataset_zip_uncompressed_too_large",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_zip_symlink_unsupported"


def test_extract_zip_safely_rejects_oversize_entry(tmp_path: Path) -> None:
    zip_path = tmp_path / "big_entry.zip"
    zip_path.write_bytes(_zip_bytes({"dataset/file.bin": b"x" * 2048}))

    with zipfile.ZipFile(zip_path, "r") as zf, pytest.raises(HTTPException) as exc_info:
        _extract_zip_safely_impl(
            zf,
            tmp_path / "extract",
            max_entry_bytes=512,
            max_total_uncompressed_bytes=4096,
            traversal_detail="dataset_zip_path_traversal",
            symlink_detail="dataset_zip_symlink_unsupported",
            entry_too_large_detail="dataset_zip_entry_too_large",
            total_too_large_detail="dataset_zip_uncompressed_too_large",
        )

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "dataset_zip_entry_too_large"


def test_extract_zip_safely_rejects_oversize_uncompressed_total(tmp_path: Path) -> None:
    zip_path = tmp_path / "big_total.zip"
    zip_path.write_bytes(
        _zip_bytes(
            {
                "dataset/a.bin": b"x" * 600,
                "dataset/b.bin": b"y" * 600,
            }
        )
    )

    with zipfile.ZipFile(zip_path, "r") as zf, pytest.raises(HTTPException) as exc_info:
        _extract_zip_safely_impl(
            zf,
            tmp_path / "extract",
            max_entry_bytes=1024,
            max_total_uncompressed_bytes=1000,
            traversal_detail="dataset_zip_path_traversal",
            symlink_detail="dataset_zip_symlink_unsupported",
            entry_too_large_detail="dataset_zip_entry_too_large",
            total_too_large_detail="dataset_zip_uncompressed_too_large",
        )

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "dataset_zip_uncompressed_too_large"


def test_upload_dataset_zip_rejects_invalid_zip() -> None:
    upload = UploadFile(filename="broken.zip", file=BytesIO(b"not a zip"))

    with pytest.raises(HTTPException) as exc_info:
        api.upload_dataset_zip(
            file=upload,
            dataset_id=None,
            dataset_type=None,
            context=None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_zip_invalid"
    assert upload.file.closed


def test_upload_dataset_zip_rejects_symlinked_registry_root_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside_registry"
    outside.mkdir()
    registry_root = tmp_path / "registry"
    try:
        registry_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    upload = UploadFile(
        filename="demo.zip",
        file=BytesIO(_zip_bytes({"labelmap.txt": b"building\n", "train/images/a.jpg": b"img"})),
    )

    with pytest.raises(HTTPException) as exc_info:
        api.upload_dataset_zip(
            file=upload,
            dataset_id="demo",
            dataset_type=None,
            context=None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_upload_target_invalid"
    assert upload.file.closed
    assert list(outside.iterdir()) == []


def test_upload_dataset_zip_rejects_symlinked_registry_root_parent_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside_registry"
    outside.mkdir()
    link_parent = tmp_path / "linked_parent"
    try:
        link_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", link_parent / "registry")
    upload = UploadFile(
        filename="demo.zip",
        file=BytesIO(_zip_bytes({"labelmap.txt": b"building\n", "train/images/a.jpg": b"img"})),
    )

    with pytest.raises(HTTPException) as exc_info:
        api.upload_dataset_zip(
            file=upload,
            dataset_id="demo",
            dataset_type=None,
            context=None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_upload_target_invalid"
    assert upload.file.closed
    assert list(outside.iterdir()) == []


def test_upload_dataset_zip_rejects_target_symlink_without_target_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    outside = tmp_path / "outside_target"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    try:
        (registry_root / "demo").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "_unique_dataset_name", lambda base, *, root: "demo")
    upload = UploadFile(
        filename="demo.zip",
        file=BytesIO(_zip_bytes({"labelmap.txt": b"building\n", "train/images/a.jpg": b"img"})),
    )

    with pytest.raises(HTTPException) as exc_info:
        api.upload_dataset_zip(
            file=upload,
            dataset_id="demo",
            dataset_type=None,
            context=None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_upload_target_invalid"
    assert upload.file.closed
    assert marker.read_text(encoding="utf-8") == "keep"


def test_upload_dataset_zip_rolls_back_when_metadata_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_root = tmp_path / "registry"
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)

    def fail_metadata_write(*_args, **_kwargs):
        raise HTTPException(status_code=400, detail="metadata_write_failed")

    monkeypatch.setattr(api, "_write_dataset_metadata_json", fail_metadata_write)
    upload = UploadFile(
        filename="demo.zip",
        file=BytesIO(_zip_bytes({"labelmap.txt": b"building\n", "images/a.jpg": b"img"})),
    )

    with pytest.raises(HTTPException) as exc_info:
        api.upload_dataset_zip(
            file=upload,
            dataset_id="demo",
            dataset_type=None,
            context=None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "metadata_write_failed"
    assert upload.file.closed
    assert not (registry_root / "demo").exists()


def test_dataset_upload_session_chunks_finalize_yolo_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_root = tmp_path / "registry"
    session_root = tmp_path / "upload_sessions"
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "YOLO_DATASET_UPLOAD_SESSION_ROOT", session_root)
    api.DATASET_UPLOAD_SESSIONS.clear()

    started = api.init_dataset_upload_session(
        {
            "dataset_id": "active_reference",
            "dataset_type": "bbox",
            "context": "Data Ingestion active reference: demo",
            "classes": ["building"],
            "total_images": 1,
        }
    )
    session_id = started["session_id"]
    upload = UploadFile(filename="a.jpg", file=BytesIO(b"image-bytes"))

    try:
        batch = api.upload_dataset_session_batch(
            session_id,
            json.dumps(
                {
                    "rows": [
                        {
                            "filename": "a.jpg",
                            "split": "train",
                            "label_text": "0 0.5 0.5 0.25 0.25",
                        }
                    ]
                }
            ),
            [upload],
        )
        assert batch["train_count"] == 1
        assert upload.file.closed

        meta = api.finalize_dataset_upload_session(session_id)
    finally:
        api.DATASET_UPLOAD_SESSIONS.clear()

    assert meta["id"] == "active_reference"
    assert meta["image_count"] == 1
    assert meta["classes"] == ["building"]
    dataset_root = registry_root / "active_reference"
    assert (dataset_root / "train" / "images" / "a.jpg").read_bytes() == b"image-bytes"
    assert (
        dataset_root / "train" / "labels" / "a.txt"
    ).read_text(encoding="utf-8") == "0 0.5 0.5 0.25 0.25"
    assert (dataset_root / "labelmap.txt").read_text(encoding="utf-8") == "building\n"
    assert (dataset_root / api.DATASET_META_NAME).exists()


def test_dataset_upload_session_start_rejects_empty_payload_without_creating_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_root = tmp_path / "upload_sessions"
    monkeypatch.setattr(api, "YOLO_DATASET_UPLOAD_SESSION_ROOT", session_root)
    api.DATASET_UPLOAD_SESSIONS.clear()

    with pytest.raises(HTTPException) as exc_info:
        api.init_dataset_upload_session({})

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_upload_session_dataset_id_required"
    assert not session_root.exists()
    assert not api.DATASET_UPLOAD_SESSIONS


def test_dataset_upload_session_finalize_recovers_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_root = tmp_path / "registry"
    session_root = tmp_path / "upload_sessions"
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "YOLO_DATASET_UPLOAD_SESSION_ROOT", session_root)
    api.DATASET_UPLOAD_SESSIONS.clear()

    started = api.init_dataset_upload_session(
        {"dataset_id": "recover_me", "classes": ["building"], "total_images": 1}
    )
    session_id = started["session_id"]
    api.upload_dataset_session_batch(
        session_id,
        json.dumps({"rows": [{"filename": "a.jpg", "split": "train", "label_text": ""}]}),
        [UploadFile(filename="a.jpg", file=BytesIO(b"img"))],
    )

    api.DATASET_UPLOAD_SESSIONS.clear()
    listed = api.list_dataset_upload_sessions()
    assert any(row["session_id"] == session_id and row["source"] == "disk" for row in listed)

    meta = api.finalize_dataset_upload_session(session_id)

    assert meta["id"] == "recover_me"
    assert (registry_root / "recover_me" / "train" / "images" / "a.jpg").exists()
    assert not (session_root / session_id).exists()


def test_dataset_upload_session_rejects_incomplete_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", tmp_path / "registry")
    monkeypatch.setattr(api, "YOLO_DATASET_UPLOAD_SESSION_ROOT", tmp_path / "upload_sessions")
    api.DATASET_UPLOAD_SESSIONS.clear()
    session_id = api.init_dataset_upload_session(
        {"dataset_id": "partial", "classes": ["building"], "total_images": 2}
    )["session_id"]
    api.upload_dataset_session_batch(
        session_id,
        json.dumps({"rows": [{"filename": "a.jpg", "split": "train", "label_text": ""}]}),
        [UploadFile(filename="a.jpg", file=BytesIO(b"img"))],
    )

    try:
        with pytest.raises(HTTPException) as exc_info:
            api.finalize_dataset_upload_session(session_id)
    finally:
        api.cancel_dataset_upload_session(session_id)
        api.DATASET_UPLOAD_SESSIONS.clear()

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_upload_incomplete"


def test_dataset_upload_session_cancel_removes_disk_session_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", tmp_path / "registry")
    monkeypatch.setattr(api, "YOLO_DATASET_UPLOAD_SESSION_ROOT", tmp_path / "upload_sessions")
    api.DATASET_UPLOAD_SESSIONS.clear()
    session_id = api.init_dataset_upload_session(
        {"dataset_id": "cancel_me", "classes": ["building"], "total_images": 1}
    )["session_id"]
    api.upload_dataset_session_batch(
        session_id,
        json.dumps({"rows": [{"filename": "a.jpg", "split": "train", "label_text": ""}]}),
        [UploadFile(filename="a.jpg", file=BytesIO(b"img"))],
    )
    api.DATASET_UPLOAD_SESSIONS.clear()

    result = api.cancel_dataset_upload_session(session_id)

    assert result["status"] == "cancelled"
    assert not ((tmp_path / "upload_sessions") / session_id).exists()


def test_import_prepass_recipe_closes_upload_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    upload = UploadFile(filename="recipe.zip", file=BytesIO(b"zip-bytes"))
    monkeypatch.setattr(api, "_import_prepass_recipe_from_zip", lambda _path: {"status": "ok"})

    assert api.import_prepass_recipe(upload) == {"status": "ok"}
    assert upload.file.closed


def test_import_edr_package_closes_upload_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    upload = UploadFile(filename="package.edr.zip", file=BytesIO(b"zip-bytes"))
    monkeypatch.setattr(
        api,
        "_import_edr_package_bundle",
        lambda _path: ({"id": "pkg"}, {"name": "recipe"}),
    )

    response = api.import_edr_package(upload)

    assert response["package"] == {"id": "pkg"}
    assert response["saved_prepass_recipe"] == {"name": "recipe"}
    assert upload.file.closed
