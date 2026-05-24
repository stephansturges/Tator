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


def test_init_qwen_dataset_upload_rejects_symlinked_staging_root_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside_uploads"
    outside.mkdir()
    upload_root = tmp_path / "dataset_uploads"
    try:
        upload_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_UPLOAD_ROOT", upload_root)
    with api.QWEN_DATASET_UPLOADS_LOCK:
        api.QWEN_DATASET_UPLOADS.clear()

    with pytest.raises(api.HTTPException) as exc_info:
        api.init_qwen_dataset_upload("demo")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_upload_path_invalid"
    assert list(outside.iterdir()) == []
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert api.QWEN_DATASET_UPLOADS == {}


def test_init_qwen_dataset_upload_rejects_symlinked_staging_root_parent_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside_uploads"
    outside.mkdir()
    link_parent = tmp_path / "linked_parent"
    try:
        link_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_UPLOAD_ROOT", link_parent / "dataset_uploads")
    with api.QWEN_DATASET_UPLOADS_LOCK:
        api.QWEN_DATASET_UPLOADS.clear()

    with pytest.raises(api.HTTPException) as exc_info:
        api.init_qwen_dataset_upload("demo")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_upload_path_invalid"
    assert list(outside.iterdir()) == []
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert api.QWEN_DATASET_UPLOADS == {}


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


def test_finalize_qwen_dataset_upload_rejects_symlinked_dataset_root_before_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_parent = tmp_path / "dataset_uploads"
    upload_root = upload_parent / "qwen_upload_job_root_link"
    train_root = upload_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    (upload_root / "val").mkdir(parents=True, exist_ok=True)
    (train_root / "a.jpg").write_bytes(b"image")
    (train_root / "annotations.jsonl").write_text('{"image":"a.jpg"}\n', encoding="utf-8")
    outside = tmp_path / "outside_qwen_datasets"
    outside.mkdir()
    qwen_root = tmp_path / "qwen_datasets"
    try:
        qwen_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_UPLOAD_ROOT", upload_parent)
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    job = api.QwenDatasetUploadJob(
        job_id="job_root_link",
        root_dir=upload_root,
        run_name="demo",
        train_count=1,
    )
    with api.QWEN_DATASET_UPLOADS_LOCK:
        api.QWEN_DATASET_UPLOADS.clear()
        api.QWEN_DATASET_UPLOADS[job.job_id] = job

    with pytest.raises(api.HTTPException) as exc_info:
        api.finalize_qwen_dataset_upload(job.job_id, {"classes": ["building"]}, "demo")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_target_invalid"
    assert job.root_dir.exists()
    assert list(outside.iterdir()) == []
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert api.QWEN_DATASET_UPLOADS[job.job_id] is job
        api.QWEN_DATASET_UPLOADS.clear()


def test_finalize_qwen_dataset_upload_rejects_symlinked_dataset_root_parent_before_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_parent = tmp_path / "dataset_uploads"
    upload_root = upload_parent / "qwen_upload_job_root_parent_link"
    train_root = upload_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    (upload_root / "val").mkdir(parents=True, exist_ok=True)
    (train_root / "a.jpg").write_bytes(b"image")
    (train_root / "annotations.jsonl").write_text('{"image":"a.jpg"}\n', encoding="utf-8")
    outside = tmp_path / "outside_qwen_datasets"
    outside.mkdir()
    link_parent = tmp_path / "linked_parent"
    try:
        link_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_UPLOAD_ROOT", upload_parent)
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", link_parent / "qwen_datasets")
    job = api.QwenDatasetUploadJob(
        job_id="job_root_parent_link",
        root_dir=upload_root,
        run_name="demo",
        train_count=1,
    )
    with api.QWEN_DATASET_UPLOADS_LOCK:
        api.QWEN_DATASET_UPLOADS.clear()
        api.QWEN_DATASET_UPLOADS[job.job_id] = job

    with pytest.raises(api.HTTPException) as exc_info:
        api.finalize_qwen_dataset_upload(job.job_id, {"classes": ["building"]}, "demo")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_target_invalid"
    assert job.root_dir.exists()
    assert list(outside.iterdir()) == []
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert api.QWEN_DATASET_UPLOADS[job.job_id] is job
        api.QWEN_DATASET_UPLOADS.clear()


def test_finalize_qwen_dataset_upload_rejects_source_outside_staging_root_before_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_parent = tmp_path / "dataset_uploads"
    upload_parent.mkdir()
    outside_upload = tmp_path / "outside_upload"
    train_root = outside_upload / "train"
    train_root.mkdir(parents=True)
    (outside_upload / "val").mkdir()
    (train_root / "a.jpg").write_bytes(b"image")
    (train_root / "annotations.jsonl").write_text('{"image":"a.jpg"}\n', encoding="utf-8")
    qwen_root = tmp_path / "qwen_datasets"
    qwen_root.mkdir()
    monkeypatch.setattr(api, "DATASET_UPLOAD_ROOT", upload_parent)
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    job = api.QwenDatasetUploadJob(
        job_id="job_outside_upload",
        root_dir=outside_upload,
        run_name="demo",
        train_count=1,
    )
    with api.QWEN_DATASET_UPLOADS_LOCK:
        api.QWEN_DATASET_UPLOADS.clear()
        api.QWEN_DATASET_UPLOADS[job.job_id] = job

    with pytest.raises(api.HTTPException) as exc_info:
        api.finalize_qwen_dataset_upload(job.job_id, {"classes": ["building"]}, "demo")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_source_path_invalid"
    assert outside_upload.exists()
    assert not (outside_upload / api.QWEN_METADATA_FILENAME).exists()
    assert list(qwen_root.iterdir()) == []
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert api.QWEN_DATASET_UPLOADS[job.job_id] is job
        api.QWEN_DATASET_UPLOADS.clear()


def test_finalize_qwen_dataset_upload_rejects_split_symlink_before_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _register_non_empty_upload(tmp_path, monkeypatch, "job_split_link_finalize")
    outside = tmp_path / "outside_split_finalize"
    outside.mkdir()
    shutil_target = job.root_dir / "train"
    for path in shutil_target.iterdir():
        path.unlink()
    shutil_target.rmdir()
    try:
        shutil_target.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        api.finalize_qwen_dataset_upload(job.job_id, {"classes": ["building"]}, "demo")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_source_path_invalid"
    assert shutil_target.is_symlink()
    assert list(outside.iterdir()) == []
    assert list(api.QWEN_DATASET_ROOT.iterdir()) == []
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


def test_cancel_qwen_dataset_upload_unlinks_symlinked_staging_job_without_target_delete(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_upload_job"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    upload_link = tmp_path / "qwen_upload_link"
    try:
        upload_link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    job = api.QwenDatasetUploadJob(job_id="job_cancel_link", root_dir=upload_link)
    with api.QWEN_DATASET_UPLOADS_LOCK:
        api.QWEN_DATASET_UPLOADS.clear()
        api.QWEN_DATASET_UPLOADS[job.job_id] = job

    out = api.cancel_qwen_dataset_upload(job.job_id)

    assert out == {"status": "cancelled", "job_id": job.job_id}
    assert not upload_link.exists()
    assert marker.read_text(encoding="utf-8") == "keep"
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert job.job_id not in api.QWEN_DATASET_UPLOADS


def test_cancel_qwen_dataset_upload_rejects_symlinked_staging_parent_without_target_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside_uploads"
    outside.mkdir()
    target_job = outside / "qwen_upload_job_parent_link"
    target_job.mkdir()
    marker = target_job / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    upload_parent_link = tmp_path / "dataset_uploads"
    try:
        upload_parent_link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_UPLOAD_ROOT", upload_parent_link)
    job = api.QwenDatasetUploadJob(
        job_id="job_parent_link",
        root_dir=upload_parent_link / "qwen_upload_job_parent_link",
    )
    with api.QWEN_DATASET_UPLOADS_LOCK:
        api.QWEN_DATASET_UPLOADS.clear()
        api.QWEN_DATASET_UPLOADS[job.job_id] = job

    out = api.cancel_qwen_dataset_upload(job.job_id)

    assert out == {"status": "cancelled", "job_id": job.job_id}
    assert marker.read_text(encoding="utf-8") == "keep"
    with api.QWEN_DATASET_UPLOADS_LOCK:
        assert job.job_id not in api.QWEN_DATASET_UPLOADS
