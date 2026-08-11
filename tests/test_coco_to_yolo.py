from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image
import pytest

import localinferenceapi as api
from utils.image import _label_relpath_for_image_impl, _resolve_coco_image_path_impl


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(128, 128, 128))
    img.save(path)


def _write_coco(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_convert_coco_to_yolo_bbox(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "train" / "images"
    img_path = images_dir / "img1.jpg"
    _write_image(img_path, (100, 50))
    coco = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 50}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10]}],
        "categories": [{"id": 1, "name": "car"}],
    }
    _write_coco(dataset_root / "train" / "_annotations.coco.json", coco)

    meta = api._convert_coco_dataset_to_yolo(dataset_root)
    assert meta["type"] == "bbox"

    labelmap = (dataset_root / "labelmap.txt").read_text(encoding="utf-8").strip().splitlines()
    assert labelmap == ["car"]

    label_path = dataset_root / "train" / "labels" / "img1.txt"
    assert label_path.exists()
    parts = label_path.read_text(encoding="utf-8").strip().split()
    assert parts[0] == "0"
    cx, cy, bw, bh = map(float, parts[1:5])
    assert math.isclose(cx, 0.2, rel_tol=1e-3)
    assert math.isclose(cy, 0.2, rel_tol=1e-3)
    assert math.isclose(bw, 0.2, rel_tol=1e-3)
    assert math.isclose(bh, 0.2, rel_tol=1e-3)


def test_convert_coco_to_yolo_replaces_symlinked_labelmap_without_target_write(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "train" / "images"
    _write_image(images_dir / "img1.jpg", (100, 50))
    _write_coco(
        dataset_root / "train" / "_annotations.coco.json",
        {
            "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 50}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10]}],
            "categories": [{"id": 1, "name": "car"}],
        },
    )
    outside = tmp_path / "outside_labelmap.txt"
    outside.write_text("external\n", encoding="utf-8")
    try:
        (dataset_root / "labelmap.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    api._convert_coco_dataset_to_yolo(dataset_root)

    assert not (dataset_root / "labelmap.txt").is_symlink()
    assert (dataset_root / "labelmap.txt").read_text(encoding="utf-8") == "car\n"
    assert outside.read_text(encoding="utf-8") == "external\n"


def test_convert_coco_to_yolo_sanitizes_parent_traversal_label_path(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "train" / "images"
    _write_image(images_dir / "escape.jpg", (100, 50))
    _write_coco(
        dataset_root / "train" / "_annotations.coco.json",
        {
            "images": [{"id": 1, "file_name": "../escape.jpg", "width": 100, "height": 50}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10]}],
            "categories": [{"id": 1, "name": "car"}],
        },
    )

    api._convert_coco_dataset_to_yolo(dataset_root)

    assert (dataset_root / "train" / "labels" / "escape.txt").exists()
    assert not (dataset_root / "train" / "escape.txt").exists()
    assert _label_relpath_for_image_impl("../escape.jpg") == Path("escape.txt")


def test_convert_coco_to_yolo_segmentation(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "train" / "images"
    img_path = images_dir / "img1.jpg"
    _write_image(img_path, (100, 100))
    coco = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "bbox": [10, 10, 20, 20],
                "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
            }
        ],
        "categories": [{"id": 3, "name": "box"}],
    }
    _write_coco(dataset_root / "train" / "_annotations.coco.json", coco)

    meta = api._convert_coco_dataset_to_yolo(dataset_root)
    assert meta["type"] == "seg"

    label_path = dataset_root / "train" / "labels" / "img1.txt"
    parts = label_path.read_text(encoding="utf-8").strip().split()
    assert parts[0] == "0"
    coords = list(map(float, parts[1:]))
    assert len(coords) >= 6
    assert all(0.0 <= v <= 1.0 for v in coords)


def test_resolve_coco_image_path_ignores_absolute_escape(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "train" / "images"
    images_dir.mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    _write_image(outside, (10, 10))

    resolved = _resolve_coco_image_path_impl(
        str(outside.resolve()),
        images_dir,
        "train",
        dataset_root,
    )

    assert resolved is None


def test_resolve_coco_image_path_uses_windows_absolute_basename(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "train" / "images"
    image_path = images_dir / "escape.jpg"
    _write_image(image_path, (10, 10))

    for raw_name in (
        "C:/outside/escape.jpg",
        r"C:\outside\escape.jpg",
        r"\\server\share\escape.jpg",
    ):
        resolved = _resolve_coco_image_path_impl(
            raw_name,
            images_dir,
            "train",
            dataset_root,
        )

        assert resolved == image_path.resolve()
        assert _label_relpath_for_image_impl(raw_name) == Path("escape.txt")
