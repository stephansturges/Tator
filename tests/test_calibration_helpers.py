from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.calibration_helpers import (
    _calibration_list_images,
    _calibration_safe_link,
    _calibration_write_record_atomic,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")


def test_calibration_list_images_dedupes_duplicate_names_across_splits(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "val" / "a.jpg")
    _touch(dataset_root / "train" / "a.jpg")
    _touch(dataset_root / "train" / "b.jpg")

    images = _calibration_list_images(
        "dummy",
        resolve_dataset_fn=lambda _: dataset_root,
    )

    assert images == ["a.jpg", "b.jpg"]


def test_calibration_list_images_filters_non_image_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "val" / "a.png")
    _touch(dataset_root / "val" / "notes.txt")
    _touch(dataset_root / "train" / "labels.json")
    _touch(dataset_root / "train" / "b.tif")

    images = _calibration_list_images(
        "dummy",
        resolve_dataset_fn=lambda _: dataset_root,
    )

    assert images == ["a.png", "b.tif"]


def test_calibration_write_record_atomic_replaces_tmp_symlink_without_target_write(
    tmp_path: Path,
) -> None:
    record_path = tmp_path / "cache" / "sample.json"
    record_path.parent.mkdir()
    outside = tmp_path / "outside_tmp.json"
    outside.write_text("external", encoding="utf-8")
    tmp_link = record_path.with_suffix(record_path.suffix + ".tmp")
    try:
        tmp_link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _calibration_write_record_atomic(record_path, {"status": "ok"})

    assert not tmp_link.exists()
    assert json.loads(record_path.read_text(encoding="utf-8"))["status"] == "ok"
    assert outside.read_text(encoding="utf-8") == "external"


def test_calibration_write_record_atomic_replaces_final_symlink_without_target_write(
    tmp_path: Path,
) -> None:
    record_path = tmp_path / "cache" / "sample.json"
    record_path.parent.mkdir()
    outside = tmp_path / "outside_record.json"
    outside.write_text("external", encoding="utf-8")
    try:
        record_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _calibration_write_record_atomic(record_path, {"status": "ok"})

    assert not record_path.is_symlink()
    assert json.loads(record_path.read_text(encoding="utf-8"))["status"] == "ok"
    assert outside.read_text(encoding="utf-8") == "external"


def test_calibration_write_record_atomic_rejects_nested_symlinked_parent_before_mkdir(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="calibration_record_parent_symlink"):
        _calibration_write_record_atomic(
            linked_parent / "nested" / "cache" / "sample.json",
            {"status": "ok"},
        )

    assert list(outside.iterdir()) == []


def test_calibration_safe_link_skips_nested_symlinked_parent_before_mkdir(
    tmp_path: Path,
) -> None:
    src = tmp_path / "source.jpg"
    src.write_bytes(b"image")
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _calibration_safe_link(src, linked_parent / "nested" / "cache" / "source.jpg")

    assert list(outside.iterdir()) == []
