from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from services.datasets import (
    _convert_coco_dataset_to_yolo_impl,
    _load_qwen_dataset_metadata_impl,
    _persist_dataset_metadata_impl,
    _persist_sam3_dataset_metadata_impl,
    _prepare_output_file,
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
