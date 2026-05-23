from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from services.classifier import _validate_clip_dataset_impl
from services.datasets import _resolve_dataset_legacy_impl
from services.detectors import _rfdetr_run_dir_impl, _yolo_run_dir_impl
from services.sam3_runs import _run_dir_for_request_impl
from utils.datasets import _iter_yolo_images
from utils.io import _compute_dir_signature, _dir_size_bytes, _sanitize_yolo_run_id


def test_legacy_dataset_resolution_rejects_sibling_prefix_escape(tmp_path: Path) -> None:
    qwen_root = tmp_path / "qwen"
    sam3_root = tmp_path / "sam3"
    registry_root = tmp_path / "registry"
    escaped = tmp_path / "qwen_evil" / "dataset"
    escaped.mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        _resolve_dataset_legacy_impl(
            "../qwen_evil/dataset",
            qwen_root=qwen_root,
            sam3_root=sam3_root,
            registry_root=registry_root,
            http_exception_cls=HTTPException,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "sam3_dataset_not_found"


def test_sam3_run_lookup_rejects_sibling_prefix_escape(tmp_path: Path) -> None:
    job_root = tmp_path / "sam3_runs"
    escaped = tmp_path / "sam3_runs_evil" / "run1"
    escaped.mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        _run_dir_for_request_impl(
            run_id="../sam3_runs_evil/run1",
            variant="sam3",
            job_root=job_root,
            http_exception_cls=HTTPException,
            http_400=400,
            http_404=404,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"


def test_yolo_run_lookup_rejects_normalized_alias(tmp_path: Path) -> None:
    job_root = tmp_path / "yolo_runs"
    (job_root / "run1").mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        _yolo_run_dir_impl(
            "../run1",
            create=False,
            job_root=job_root,
            sanitize_fn=_sanitize_yolo_run_id,
            http_exception_cls=HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"


def test_yolo_run_lookup_rejects_blank_id(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _yolo_run_dir_impl(
            "",
            create=False,
            job_root=tmp_path / "yolo_runs",
            sanitize_fn=_sanitize_yolo_run_id,
            http_exception_cls=HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"


def test_rfdetr_run_lookup_rejects_normalized_alias(tmp_path: Path) -> None:
    job_root = tmp_path / "rfdetr_runs"
    (job_root / "run1").mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        _rfdetr_run_dir_impl(
            "../run1",
            create=False,
            job_root=job_root,
            sanitize_fn=_sanitize_yolo_run_id,
            http_exception_cls=HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"


def test_compute_dir_signature_skips_symlink_escape(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "local.txt").write_text("local", encoding="utf-8")
    before = _compute_dir_signature(dataset_root)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (dataset_root / "escape.txt").symlink_to(outside)

    assert _compute_dir_signature(dataset_root) == before


def test_dir_size_bytes_skips_file_symlinks(tmp_path: Path) -> None:
    quota_root = tmp_path / "quota"
    quota_root.mkdir()
    (quota_root / "local.bin").write_bytes(b"1234")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x" * 100)
    (quota_root / "escape.bin").symlink_to(outside)

    assert _dir_size_bytes(quota_root) == 4


def test_iter_yolo_images_skips_symlink_escape(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    images_root.mkdir()
    local = images_root / "local.jpg"
    local.write_bytes(b"local")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"secret")
    (images_root / "escape.jpg").symlink_to(outside)

    assert _iter_yolo_images(images_root) == [local]


def test_clip_dataset_validation_skips_symlinked_images(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    labels_root = tmp_path / "labels"
    images_root.mkdir()
    labels_root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"not an image")
    (images_root / "escape.jpg").symlink_to(outside)
    (labels_root / "local.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        _validate_clip_dataset_impl(
            {
                "images_dir": str(images_root),
                "labels_dir": str(labels_root),
            },
            http_exception_cls=HTTPException,
            load_labelmap_simple_fn=lambda _path: ["car"],
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "clip_images_missing"


def test_clip_dataset_validation_skips_symlinked_labels(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    labels_root = tmp_path / "labels"
    images_root.mkdir()
    labels_root.mkdir()
    (images_root / "local.jpg").write_bytes(b"not an image")
    outside = tmp_path / "outside.txt"
    outside.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels_root / "escape.txt").symlink_to(outside)

    with pytest.raises(HTTPException) as exc_info:
        _validate_clip_dataset_impl(
            {
                "images_dir": str(images_root),
                "labels_dir": str(labels_root),
            },
            http_exception_cls=HTTPException,
            load_labelmap_simple_fn=lambda _path: ["car"],
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "clip_labels_missing"
