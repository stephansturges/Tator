from __future__ import annotations

from pathlib import Path

from services.calibration_helpers import _calibration_list_images


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
