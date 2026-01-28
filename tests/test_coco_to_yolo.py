from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

import localinferenceapi as api


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
