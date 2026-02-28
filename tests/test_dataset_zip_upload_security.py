from __future__ import annotations

import stat
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException

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
