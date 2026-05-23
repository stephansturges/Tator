from __future__ import annotations

import json
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
        assert upload.file.closed
    finally:
        _cleanup_upload_job("job_ann")


def test_qwen_chunk_rejects_invalid_json_annotation(tmp_path: Path) -> None:
    _register_upload_job(tmp_path, "job_bad_json")
    try:
        upload = UploadFile(filename="a.jpg", file=BytesIO(b"img"))
        with pytest.raises(HTTPException) as exc_info:
            localinferenceapi.upload_qwen_dataset_chunk(
                "job_bad_json",
                "train",
                "a.jpg",
                "not json",
                upload,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "qwen_dataset_annotation_invalid"
        assert not (tmp_path / "qwen_upload_job_bad_json" / "train" / "a.jpg").exists()
        assert upload.file.closed
    finally:
        _cleanup_upload_job("job_bad_json")


def test_qwen_chunk_rejects_empty_image(tmp_path: Path) -> None:
    job = _register_upload_job(tmp_path, "job_empty")
    try:
        upload = UploadFile(filename="empty.jpg", file=BytesIO(b""))
        with pytest.raises(HTTPException) as exc_info:
            localinferenceapi.upload_qwen_dataset_chunk(
                "job_empty",
                "train",
                "empty.jpg",
                '{"id":"x"}',
                upload,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "qwen_dataset_empty_image"
        assert not (job.root_dir / "train" / "empty.jpg").exists()
        assert not (job.root_dir / "train" / "annotations.jsonl").exists()
        assert job.train_count == 0
        assert upload.file.closed
    finally:
        _cleanup_upload_job("job_empty")


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
        assert upload.file.closed
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
        annotation = json.loads((job.root_dir / "train" / "annotations.jsonl").read_text(encoding="utf-8"))
        assert annotation["image"] == "escape.jpg"
        assert upload.file.closed
    finally:
        _cleanup_upload_job("job_path")


def test_qwen_chunk_rejects_reserved_annotation_filename(tmp_path: Path) -> None:
    job = _register_upload_job(tmp_path, "job_reserved")
    try:
        upload = UploadFile(filename="annotations.jsonl", file=BytesIO(b"img"))
        with pytest.raises(HTTPException) as exc_info:
            localinferenceapi.upload_qwen_dataset_chunk(
                "job_reserved",
                "train",
                "annotations.jsonl",
                '{"id":"x"}',
                upload,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "qwen_dataset_image_name_reserved"
        assert not (job.root_dir / "train" / "annotations.jsonl").exists()
        assert upload.file.closed
    finally:
        _cleanup_upload_job("job_reserved")


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
        assert upload1.file.closed
        assert upload2.file.closed
    finally:
        _cleanup_upload_job("job_dup")


def test_qwen_chunk_rejects_image_symlink_without_target_write(tmp_path: Path) -> None:
    job = _register_upload_job(tmp_path, "job_image_link")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"external")
    try:
        (job.root_dir / "train" / "link.jpg").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    try:
        upload = UploadFile(filename="link.jpg", file=BytesIO(b"img"))
        with pytest.raises(HTTPException) as exc_info:
            localinferenceapi.upload_qwen_dataset_chunk(
                "job_image_link",
                "train",
                "link.jpg",
                '{"id":"x"}',
                upload,
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == "qwen_dataset_image_exists"
        assert outside.read_bytes() == b"external"
        assert upload.file.closed
    finally:
        _cleanup_upload_job("job_image_link")


def test_qwen_chunk_rejects_annotation_symlink_without_target_write(tmp_path: Path) -> None:
    job = _register_upload_job(tmp_path, "job_ann_link")
    outside = tmp_path / "outside_annotations.jsonl"
    outside.write_text("external\n", encoding="utf-8")
    try:
        (job.root_dir / "train" / "annotations.jsonl").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    try:
        upload = UploadFile(filename="a.jpg", file=BytesIO(b"img"))
        with pytest.raises(HTTPException) as exc_info:
            localinferenceapi.upload_qwen_dataset_chunk(
                "job_ann_link",
                "train",
                "a.jpg",
                '{"id":"x"}',
                upload,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "qwen_dataset_annotation_path_invalid"
        assert outside.read_text(encoding="utf-8") == "external\n"
        assert not (job.root_dir / "train" / "a.jpg").exists()
        assert upload.file.closed
    finally:
        _cleanup_upload_job("job_ann_link")


def test_qwen_chunk_rejects_split_symlink_without_target_write(tmp_path: Path) -> None:
    job = _register_upload_job(tmp_path, "job_split_link")
    outside = tmp_path / "outside_split"
    outside.mkdir()
    (job.root_dir / "train").rmdir()
    try:
        (job.root_dir / "train").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    try:
        upload = UploadFile(filename="a.jpg", file=BytesIO(b"img"))
        with pytest.raises(HTTPException) as exc_info:
            localinferenceapi.upload_qwen_dataset_chunk(
                "job_split_link",
                "train",
                "a.jpg",
                '{"id":"x"}',
                upload,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "qwen_dataset_split_path_invalid"
        assert not (outside / "a.jpg").exists()
        assert upload.file.closed
    finally:
        _cleanup_upload_job("job_split_link")


def test_qwen_chunk_rejects_symlinked_job_root_without_target_write(tmp_path: Path) -> None:
    outside = tmp_path / "outside_upload_job"
    outside.mkdir()
    root_link = tmp_path / "qwen_upload_job_root_link"
    try:
        root_link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    job = localinferenceapi.QwenDatasetUploadJob(job_id="job_root_link", root_dir=root_link)
    with localinferenceapi.QWEN_DATASET_UPLOADS_LOCK:
        localinferenceapi.QWEN_DATASET_UPLOADS[job.job_id] = job
    try:
        upload = UploadFile(filename="a.jpg", file=BytesIO(b"img"))
        with pytest.raises(HTTPException) as exc_info:
            localinferenceapi.upload_qwen_dataset_chunk(
                job.job_id,
                "train",
                "a.jpg",
                '{"id":"x"}',
                upload,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "qwen_dataset_source_path_invalid"
        assert list(outside.iterdir()) == []
        assert upload.file.closed
    finally:
        _cleanup_upload_job(job.job_id)
