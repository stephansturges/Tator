from pathlib import Path

import pytest

import localinferenceapi as api


def _register_non_empty_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, job_id: str = "job_ready"
) -> api.QwenDatasetUploadJob:
    upload_parent = tmp_path / "dataset_uploads"
    upload_root = upload_parent / f"qwen_upload_{job_id}"
    train_root = upload_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    (upload_root / "val").mkdir(parents=True, exist_ok=True)
    (train_root / "a.jpg").write_bytes(b"image")
    (train_root / "annotations.jsonl").write_text('{"image":"a.jpg"}\n', encoding="utf-8")
    qwen_root = tmp_path / "qwen_datasets"
    qwen_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "DATASET_UPLOAD_ROOT", upload_parent)
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    job = api.QwenDatasetUploadJob(
        job_id=job_id,
        root_dir=upload_root,
        run_name="ready_ds",
        train_count=1,
    )
    with api.QWEN_DATASET_UPLOADS_LOCK:
        api.QWEN_DATASET_UPLOADS.clear()
        api.QWEN_DATASET_UPLOADS[job.job_id] = job
    return job


def test_finalize_qwen_dataset_upload_rejects_empty_before_pop_or_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_root = tmp_path / "upload_job"
    upload_root.mkdir(parents=True, exist_ok=True)
    qwen_root = tmp_path / "qwen_datasets"
    qwen_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)

    job = api.QwenDatasetUploadJob(job_id="job_empty", root_dir=upload_root, run_name="empty_ds")
    with api.QWEN_DATASET_UPLOADS_LOCK:
        api.QWEN_DATASET_UPLOADS.clear()
        api.QWEN_DATASET_UPLOADS[job.job_id] = job

    with pytest.raises(api.HTTPException) as exc_info:
        api.finalize_qwen_dataset_upload(job.job_id, {}, None)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_empty"
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert api.QWEN_DATASET_UPLOADS[job.job_id] is job
        api.QWEN_DATASET_UPLOADS.clear()
    assert upload_root.exists()
    assert list(qwen_root.iterdir()) == []


def test_finalize_qwen_dataset_upload_moves_only_after_metadata_is_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _register_non_empty_upload(tmp_path, monkeypatch)

    meta = api.finalize_qwen_dataset_upload(
        job.job_id,
        {"classes": ["building"], "context": "Urban imagery"},
        "demo",
    )

    target_root = api.QWEN_DATASET_ROOT / "demo"
    assert meta["id"] == "demo"
    assert meta["classes"] == ["building"]
    assert meta["context"] == "Urban imagery"
    assert meta["image_count"] == 1
    assert not job.root_dir.exists()
    assert (target_root / "train" / "a.jpg").read_bytes() == b"image"
    saved_meta = api._load_json_metadata(target_root / "metadata.json")
    assert saved_meta["id"] == "demo"
    assert saved_meta["signature"] == meta["signature"]
    dataset_meta = api._load_json_metadata(target_root / "dataset_meta.json")
    assert dataset_meta["classes"] == ["building"]
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert job.job_id not in api.QWEN_DATASET_UPLOADS


def test_finalize_qwen_dataset_upload_keeps_job_when_move_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _register_non_empty_upload(tmp_path, monkeypatch, "job_move")
    original_move = api.shutil.move

    def fail_move(src: str, dst: Path) -> None:
        raise OSError("simulated move failure")

    monkeypatch.setattr(api.shutil, "move", fail_move)

    with pytest.raises(api.HTTPException) as exc_info:
        api.finalize_qwen_dataset_upload(job.job_id, {}, "demo")

    assert exc_info.value.status_code == 500
    assert str(exc_info.value.detail).startswith("qwen_dataset_finalize_failed:")
    assert job.root_dir.exists()
    assert not (api.QWEN_DATASET_ROOT / "demo").exists()
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert api.QWEN_DATASET_UPLOADS[job.job_id] is job

    monkeypatch.setattr(api.shutil, "move", original_move)
    meta = api.finalize_qwen_dataset_upload(job.job_id, {"classes": ["building"]}, "demo")

    assert meta["id"] == "demo"
    assert (api.QWEN_DATASET_ROOT / "demo" / "train" / "a.jpg").exists()
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert job.job_id not in api.QWEN_DATASET_UPLOADS


def test_finalize_qwen_dataset_upload_does_not_delete_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _register_non_empty_upload(tmp_path, monkeypatch, "job_collision")
    target_root = api.QWEN_DATASET_ROOT / "existing"
    target_root.mkdir(parents=True, exist_ok=True)
    marker = target_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(api, "_unique_dataset_name", lambda base, *, root: "existing")

    with pytest.raises(api.HTTPException) as exc_info:
        api.finalize_qwen_dataset_upload(job.job_id, {}, "demo")

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "qwen_dataset_target_exists"
    assert marker.read_text(encoding="utf-8") == "keep"
    assert job.root_dir.exists()
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert api.QWEN_DATASET_UPLOADS[job.job_id] is job
        api.QWEN_DATASET_UPLOADS.clear()


def test_finalize_qwen_dataset_upload_rejects_broken_target_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _register_non_empty_upload(tmp_path, monkeypatch, "job_target_link")
    target_root = api.QWEN_DATASET_ROOT / "linked"
    try:
        target_root.symlink_to(tmp_path / "missing_target", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "_unique_dataset_name", lambda base, *, root: "linked")

    with pytest.raises(api.HTTPException) as exc_info:
        api.finalize_qwen_dataset_upload(job.job_id, {}, "demo")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_target_invalid"
    assert target_root.is_symlink()
    assert job.root_dir.exists()
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert api.QWEN_DATASET_UPLOADS[job.job_id] is job
        api.QWEN_DATASET_UPLOADS.clear()


def test_finalize_qwen_dataset_upload_replaces_labelmap_symlink_without_target_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _register_non_empty_upload(tmp_path, monkeypatch, "job_labelmap_link")
    outside = tmp_path / "outside_labelmap.txt"
    outside.write_text("external\n", encoding="utf-8")
    try:
        (job.root_dir / "labelmap.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    api.finalize_qwen_dataset_upload(job.job_id, {"classes": ["building"]}, "demo")

    target_labelmap = api.QWEN_DATASET_ROOT / "demo" / "labelmap.txt"
    assert target_labelmap.read_text(encoding="utf-8") == "building\n"
    assert not target_labelmap.is_symlink()
    assert outside.read_text(encoding="utf-8") == "external\n"
