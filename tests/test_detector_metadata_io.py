from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from services.detectors import (
    _copy2_if_different,
    _load_detector_default_impl,
    _load_rfdetr_active_impl,
    _load_yolo_active_impl,
    _rfdetr_remap_coco_ids_impl,
    _rfdetr_write_run_meta_impl,
    _save_detector_default_impl,
    _save_rfdetr_active_impl,
    _save_yolo_active_impl,
    _strip_checkpoint_optimizer_impl,
    _yolo_write_run_meta_impl,
)


def test_save_yolo_active_replaces_symlink_targets_without_target_write(tmp_path: Path) -> None:
    active_path = tmp_path / "detectors" / "yolo_active.json"
    active_path.parent.mkdir()
    outside_tmp = tmp_path / "outside_tmp.json"
    outside_final = tmp_path / "outside_final.json"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    tmp_link = active_path.with_suffix(active_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_link.symlink_to(outside_tmp)
        active_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    saved = _save_yolo_active_impl({"run_id": "r1"}, active_path)

    assert saved["run_id"] == "r1"
    assert not tmp_link.exists()
    assert not active_path.is_symlink()
    assert json.loads(active_path.read_text(encoding="utf-8"))["run_id"] == "r1"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_load_yolo_active_skips_symlink_escape(tmp_path: Path) -> None:
    active_path = tmp_path / "yolo_active.json"
    outside = tmp_path / "outside.json"
    best = tmp_path / "best.pt"
    best.write_text("weights", encoding="utf-8")
    outside.write_text(json.dumps({"run_id": "escaped", "best_path": str(best)}), encoding="utf-8")
    try:
        active_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert _load_yolo_active_impl(active_path) == {}


def test_load_yolo_active_skips_symlinked_labelmap_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    best = run_dir / "best.pt"
    best.write_text("weights", encoding="utf-8")
    outside_labelmap = tmp_path / "outside_labelmap.txt"
    outside_labelmap.write_text("escaped\n", encoding="utf-8")
    labelmap = run_dir / "labelmap.txt"
    try:
        labelmap.symlink_to(outside_labelmap)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    active_path = tmp_path / "yolo_active.json"
    active_path.write_text(
        json.dumps({"run_id": "run", "best_path": str(best), "labelmap_path": str(labelmap)}),
        encoding="utf-8",
    )

    assert _load_yolo_active_impl(active_path) == {}


def test_load_rfdetr_active_skips_symlinked_best_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.pth"
    outside.write_text("secret", encoding="utf-8")
    best = run_dir / "checkpoint_best_total.pth"
    try:
        best.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    active_path = tmp_path / "rfdetr_active.json"
    active_path.write_text(
        json.dumps({"run_id": "run", "best_path": str(best)}),
        encoding="utf-8",
    )

    assert (
        _load_rfdetr_active_impl(
            active_path,
            tmp_path / "rfdetr_runs",
            save_active_fn=lambda payload: _save_rfdetr_active_impl(payload, active_path),
        )
        == {}
    )


def test_save_rfdetr_active_replaces_symlink_targets_without_target_write(tmp_path: Path) -> None:
    active_path = tmp_path / "detectors" / "rfdetr_active.json"
    active_path.parent.mkdir()
    outside_tmp = tmp_path / "rfdetr_outside_tmp.json"
    outside_final = tmp_path / "rfdetr_outside_final.json"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    tmp_link = active_path.with_suffix(active_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_link.symlink_to(outside_tmp)
        active_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    saved = _save_rfdetr_active_impl({"run_id": "r1"}, active_path)

    assert saved["run_id"] == "r1"
    assert not tmp_link.exists()
    assert not active_path.is_symlink()
    assert json.loads(active_path.read_text(encoding="utf-8"))["run_id"] == "r1"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_detector_default_replaces_symlink_without_target_write(tmp_path: Path) -> None:
    default_path = tmp_path / "detector_default.json"
    outside = tmp_path / "outside_default.json"
    outside.write_text(json.dumps({"mode": "yolo"}), encoding="utf-8")
    try:
        default_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert _load_detector_default_impl(default_path) == {"mode": "rfdetr"}
    saved = _save_detector_default_impl({"mode": "yolo"}, default_path, HTTPException)

    assert saved["mode"] == "yolo"
    assert not default_path.is_symlink()
    assert outside.read_text(encoding="utf-8") == json.dumps({"mode": "yolo"})


def test_yolo_write_run_meta_replaces_symlink_targets_without_target_write(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    meta_path = run_dir / "run.json"
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

    _yolo_write_run_meta_impl(run_dir, {"status": "ok"}, meta_name="run.json", time_fn=lambda: 42.0)

    assert not tmp_link.exists()
    assert not meta_path.is_symlink()
    assert json.loads(meta_path.read_text(encoding="utf-8"))["status"] == "ok"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_rfdetr_write_run_meta_replaces_symlink_targets_without_target_write(tmp_path: Path) -> None:
    run_dir = tmp_path / "rfdetr" / "r1"
    run_dir.mkdir(parents=True)
    meta_path = run_dir / "run.json"
    outside_tmp = tmp_path / "rfdetr_outside_tmp.json"
    outside_final = tmp_path / "rfdetr_outside_final.json"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    tmp_link = meta_path.with_suffix(meta_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_link.symlink_to(outside_tmp)
        meta_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _rfdetr_write_run_meta_impl(run_dir, {"status": "ok"}, meta_name="run.json", time_fn=lambda: 42.0)

    assert not tmp_link.exists()
    assert not meta_path.is_symlink()
    assert json.loads(meta_path.read_text(encoding="utf-8"))["status"] == "ok"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_detector_copy2_if_different_replaces_symlink_to_source(tmp_path: Path) -> None:
    src = tmp_path / "source.bin"
    src.write_text("source", encoding="utf-8")
    dest = tmp_path / "dest.bin"
    try:
        dest.symlink_to(src)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _copy2_if_different(src, dest)

    assert not dest.is_symlink()
    assert dest.read_text(encoding="utf-8") == "source"


def test_rfdetr_remap_replaces_symlink_dest_without_target_write(tmp_path: Path) -> None:
    src = tmp_path / "src.coco.json"
    src.write_text(
        json.dumps(
            {
                "categories": [{"id": 5, "name": "person"}],
                "annotations": [{"id": 1, "category_id": 5}],
            }
        ),
        encoding="utf-8",
    )
    outside = tmp_path / "outside.coco.json"
    outside.write_text("external", encoding="utf-8")
    dest = tmp_path / "dest.coco.json"
    try:
        dest.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _rfdetr_remap_coco_ids_impl(src, dest)

    assert not dest.is_symlink()
    assert json.loads(dest.read_text(encoding="utf-8"))["categories"][0]["id"] == 0
    assert outside.read_text(encoding="utf-8") == "external"


def test_strip_checkpoint_optimizer_skips_symlinked_checkpoint(tmp_path: Path) -> None:
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"checkpoint")
    ckpt = tmp_path / "checkpoint.pt"
    try:
        ckpt.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    class TorchStub:
        @staticmethod
        def load(*_args, **_kwargs):
            raise AssertionError("must not load symlink target")

        @staticmethod
        def save(*_args, **_kwargs):
            raise AssertionError("must not write symlink target")

    stripped, before, after = _strip_checkpoint_optimizer_impl(ckpt, torch_module=TorchStub)

    assert stripped is False
    assert before == 0
    assert after == 0
    assert outside.read_bytes() == b"checkpoint"
