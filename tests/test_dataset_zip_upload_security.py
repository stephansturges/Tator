from __future__ import annotations

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
