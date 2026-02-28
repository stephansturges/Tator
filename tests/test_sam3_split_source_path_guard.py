from __future__ import annotations

import json
from pathlib import Path

import localinferenceapi


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_sam3_training_split_ignores_absolute_image_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    train_images_dir = dataset_root / "train" / "images"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 5):
        (train_images_dir / f"img{idx}.jpg").write_bytes(b"img")

    outside_image = tmp_path / "outside.jpg"
    outside_image.write_bytes(b"outside")

    coco_train = {
        "images": [
            {"id": 1, "file_name": "img1.jpg"},
            {"id": 2, "file_name": "img2.jpg"},
            {"id": 3, "file_name": "img3.jpg"},
            {"id": 4, "file_name": "img4.jpg"},
            {"id": 5, "file_name": str(outside_image.resolve())},
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "obj"}],
    }
    coco_val = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "obj"}]}
    train_json = dataset_root / "train" / "_annotations.coco.json"
    val_json = dataset_root / "val" / "_annotations.coco.json"
    _write_json(train_json, coco_train)
    _write_json(val_json, coco_val)

    monkeypatch.setattr(localinferenceapi, "SAM3_JOB_ROOT", tmp_path / "sam3_jobs")
    out = localinferenceapi._prepare_sam3_training_split(
        dataset_root,
        {
            "id": "ds",
            "classes": ["obj"],
            "coco_train_json": str(train_json),
            "coco_val_json": str(val_json),
        },
        "job_guard_abs",
        random_split=True,
        val_percent=0.4,
        split_seed=42,
    )

    train_out = json.loads(Path(out["coco_train_json"]).read_text(encoding="utf-8"))
    val_out = json.loads(Path(out["coco_val_json"]).read_text(encoding="utf-8"))
    all_names = [img["file_name"] for img in train_out["images"]] + [img["file_name"] for img in val_out["images"]]
    assert set(all_names).issubset({"img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"})
    assert len(all_names) == 4


def test_prepare_sam3_training_split_ignores_parent_traversal_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "dataset2"
    train_images_dir = dataset_root / "train" / "images"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 5):
        (train_images_dir / f"img{idx}.jpg").write_bytes(b"img")

    coco_train = {
        "images": [
            {"id": 1, "file_name": "img1.jpg"},
            {"id": 2, "file_name": "img2.jpg"},
            {"id": 3, "file_name": "img3.jpg"},
            {"id": 4, "file_name": "img4.jpg"},
            {"id": 5, "file_name": "../escape.jpg"},
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "obj"}],
    }
    coco_val = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "obj"}]}
    train_json = dataset_root / "train" / "_annotations.coco.json"
    val_json = dataset_root / "val" / "_annotations.coco.json"
    _write_json(train_json, coco_train)
    _write_json(val_json, coco_val)

    monkeypatch.setattr(localinferenceapi, "SAM3_JOB_ROOT", tmp_path / "sam3_jobs")
    out = localinferenceapi._prepare_sam3_training_split(
        dataset_root,
        {
            "id": "ds2",
            "classes": ["obj"],
            "coco_train_json": str(train_json),
            "coco_val_json": str(val_json),
        },
        "job_guard_parent",
        random_split=True,
        val_percent=0.4,
        split_seed=42,
    )

    train_out = json.loads(Path(out["coco_train_json"]).read_text(encoding="utf-8"))
    val_out = json.loads(Path(out["coco_val_json"]).read_text(encoding="utf-8"))
    all_names = [img["file_name"] for img in train_out["images"]] + [img["file_name"] for img in val_out["images"]]
    assert set(all_names).issubset({"img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"})
    assert len(all_names) == 4
