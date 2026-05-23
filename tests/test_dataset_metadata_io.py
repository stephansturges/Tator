from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from services.datasets import (
    _load_qwen_dataset_metadata_impl,
    _persist_dataset_metadata_impl,
    _persist_sam3_dataset_metadata_impl,
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
