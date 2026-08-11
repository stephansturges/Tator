from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image

from services.datasets import (
    _convert_coco_dataset_to_yolo_impl,
    _convert_yolo_dataset_to_coco_impl,
    _load_qwen_dataset_metadata_impl,
    _persist_dataset_metadata_impl,
    _persist_qwen_dataset_metadata_impl,
    _persist_sam3_dataset_metadata_impl,
    _prepare_output_file,
    _write_text_file,
)
from utils.io import _load_json_metadata


def test_persist_dataset_metadata_replaces_symlink_targets_without_target_write(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    meta_path = dataset_dir / "dataset_meta.json"
    outside_tmp = tmp_path / "outside_tmp.json"
    outside_final = tmp_path / "outside_final.json"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    tmp_link = meta_path.with_suffix(meta_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_link.symlink_to(outside_tmp)
        meta_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _persist_dataset_metadata_impl(dataset_dir, {"id": "dataset"}, meta_name="dataset_meta.json")

    assert not tmp_link.exists()
    assert not meta_path.is_symlink()
    assert json.loads(meta_path.read_text(encoding="utf-8"))["id"] == "dataset"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_dataset_text_write_is_atomic_and_replaces_symlink_targets_without_target_write(
    tmp_path: Path,
) -> None:
    labelmap_path = tmp_path / "dataset" / "labelmap.txt"
    labelmap_path.parent.mkdir()
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final = tmp_path / "outside_final.txt"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    tmp_link = labelmap_path.with_suffix(labelmap_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_link.symlink_to(outside_tmp)
        labelmap_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _write_text_file(labelmap_path, "person\ncar\n")

    assert not tmp_link.exists()
    assert not labelmap_path.is_symlink()
    assert labelmap_path.read_text(encoding="utf-8") == "person\ncar\n"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_dataset_prepare_output_file_rejects_nested_symlinked_parent_before_mkdir(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(RuntimeError, match="dataset_path_invalid"):
        _prepare_output_file(linked_parent / "nested" / "dataset" / "metadata.json")

    assert list(outside.iterdir()) == []


def test_convert_coco_to_yolo_rejects_nested_symlinked_label_root_before_mkdir(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    outside_train = tmp_path / "outside_train"
    (outside_train / "images").mkdir(parents=True)
    (outside_train / "images" / "img.jpg").write_bytes(b"image")
    (outside_train / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "img.jpg", "width": 10, "height": 10}],
                "categories": [{"id": 1, "name": "object"}],
                "annotations": [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]}],
            }
        ),
        encoding="utf-8",
    )
    dataset_root.mkdir()
    try:
        (dataset_root / "train").symlink_to(outside_train, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _convert_coco_dataset_to_yolo_impl(dataset_root)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "coco_label_path_invalid"
    assert not (outside_train / "labels").exists()


def test_persist_sam3_dataset_metadata_replaces_symlink_without_target_write(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    meta_path = dataset_dir / "sam3_dataset.json"
    outside = tmp_path / "outside_sam3.json"
    outside.write_text(json.dumps({"id": "outside"}), encoding="utf-8")
    try:
        meta_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _persist_sam3_dataset_metadata_impl(dataset_dir, {"id": "sam3"})

    assert not meta_path.is_symlink()
    assert json.loads(meta_path.read_text(encoding="utf-8"))["id"] == "sam3"
    assert outside.read_text(encoding="utf-8") == json.dumps({"id": "outside"})


def test_persist_dataset_metadata_raises_when_final_path_is_unwritable(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_meta.json").mkdir()

    with pytest.raises(HTTPException) as exc_info:
        _persist_dataset_metadata_impl(
            dataset_dir,
            {"id": "dataset"},
            meta_name="dataset_meta.json",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "dataset_metadata_write_failed"


def test_persist_dataset_metadata_can_suppress_read_time_write_failure(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_meta.json").mkdir()

    _persist_dataset_metadata_impl(
        dataset_dir,
        {"id": "dataset"},
        meta_name="dataset_meta.json",
        suppress_errors=True,
    )


def test_persist_sam3_dataset_metadata_raises_when_final_path_is_unwritable(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "sam3_dataset.json").mkdir()

    with pytest.raises(HTTPException) as exc_info:
        _persist_sam3_dataset_metadata_impl(dataset_dir, {"id": "sam3"})

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "sam3_dataset_metadata_write_failed"


def test_persist_qwen_dataset_metadata_raises_qwen_detail_on_write_failure(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.json").mkdir()

    with pytest.raises(HTTPException) as exc_info:
        _persist_qwen_dataset_metadata_impl(dataset_dir, {"id": "qwen"})

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "qwen_dataset_metadata_write_failed"


def test_convert_yolo_to_coco_fails_when_sam3_metadata_cannot_be_written(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    Image.new("RGB", (16, 16), color="white").save(train_images / "img.jpg")
    (train_labels / "img.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (dataset_root / "labelmap.txt").write_text("object\n", encoding="utf-8")
    (dataset_root / "sam3_dataset.json").mkdir()

    with pytest.raises(HTTPException) as exc_info:
        _convert_yolo_dataset_to_coco_impl(dataset_root)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "sam3_dataset_metadata_write_failed"


def test_load_json_metadata_skips_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"id": "outside"}), encoding="utf-8")
    meta_path = tmp_path / "metadata.json"
    try:
        meta_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert _load_json_metadata(meta_path) is None
    assert _load_qwen_dataset_metadata_impl(tmp_path) is None
