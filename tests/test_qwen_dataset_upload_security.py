from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

import localinferenceapi


def _register_upload_job(tmp_path: Path, job_id: str = "job1") -> localinferenceapi.QwenDatasetUploadJob:
    root_dir = tmp_path / f"qwen_upload_{job_id}"
    (root_dir / "train").mkdir(parents=True, exist_ok=True)
    (root_dir / "val").mkdir(parents=True, exist_ok=True)
    job = localinferenceapi.QwenDatasetUploadJob(job_id=job_id, root_dir=root_dir)
    with localinferenceapi.QWEN_DATASET_UPLOADS_LOCK:
        localinferenceapi.QWEN_DATASET_UPLOADS[job_id] = job
    return job


def _cleanup_upload_job(job_id: str) -> None:
    with localinferenceapi.QWEN_DATASET_UPLOADS_LOCK:
        localinferenceapi.QWEN_DATASET_UPLOADS.pop(job_id, None)


def test_qwen_chunk_rejects_multiline_annotation(tmp_path: Path) -> None:
    _register_upload_job(tmp_path, "job_ann")
    try:
        upload = UploadFile(filename="a.jpg", file=BytesIO(b"img"))
        with pytest.raises(HTTPException) as exc_info:
            localinferenceapi.upload_qwen_dataset_chunk(
                "job_ann",
                "train",
                "a.jpg",
                '{"a":1}\n{"b":2}',
                upload,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "qwen_dataset_annotation_invalid"
        assert not (tmp_path / "qwen_upload_job_ann" / "train" / "a.jpg").exists()
    finally:
        _cleanup_upload_job("job_ann")


def test_qwen_chunk_limits_size_and_cleans_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _register_upload_job(tmp_path, "job_size")
    monkeypatch.setattr(localinferenceapi, "QWEN_DATASET_CHUNK_MAX_BYTES", 4)
    try:
        upload = UploadFile(filename="big.jpg", file=BytesIO(b"0123456789"))
        with pytest.raises(HTTPException) as exc_info:
            localinferenceapi.upload_qwen_dataset_chunk(
                "job_size",
                "train",
                "big.jpg",
                '{"id":"x"}',
                upload,
            )
        assert exc_info.value.status_code == 413
        assert exc_info.value.detail == "qwen_dataset_chunk_too_large"
        assert not (job.root_dir / "train" / "big.jpg").exists()
        assert job.train_count == 0
    finally:
        _cleanup_upload_job("job_size")


def test_qwen_chunk_sanitizes_image_name_to_split_root(tmp_path: Path) -> None:
    job = _register_upload_job(tmp_path, "job_path")
    try:
        upload = UploadFile(filename="a.jpg", file=BytesIO(b"img"))
        out = localinferenceapi.upload_qwen_dataset_chunk(
            "job_path",
            "train",
            "../escape.jpg",
            '{"id":"x"}',
            upload,
        )
        assert out["status"] == "ok"
        assert (job.root_dir / "train" / "escape.jpg").exists()
        assert not (tmp_path / "escape.jpg").exists()
    finally:
        _cleanup_upload_job("job_path")


def test_qwen_chunk_rejects_duplicate_image_name(tmp_path: Path) -> None:
    job = _register_upload_job(tmp_path, "job_dup")
    try:
        upload1 = UploadFile(filename="a.jpg", file=BytesIO(b"img1"))
        out = localinferenceapi.upload_qwen_dataset_chunk(
            "job_dup",
            "train",
            "dup.jpg",
            '{"id":"x"}',
            upload1,
        )
        assert out["status"] == "ok"
        upload2 = UploadFile(filename="a.jpg", file=BytesIO(b"img2"))
        with pytest.raises(HTTPException) as exc_info:
            localinferenceapi.upload_qwen_dataset_chunk(
                "job_dup",
                "train",
                "dup.jpg",
                '{"id":"x2"}',
                upload2,
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == "qwen_dataset_image_exists"
        assert (job.root_dir / "train" / "dup.jpg").read_bytes() == b"img1"
    finally:
        _cleanup_upload_job("job_dup")
