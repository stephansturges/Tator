from __future__ import annotations

from pathlib import Path

import pytest

import localinferenceapi as api


def test_delete_sam3_run_resets_active_checkpoint_when_deleted(tmp_path: Path, monkeypatch) -> None:
    run_id = "active_sam3_run"
    run_dir = tmp_path / run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "last.ckpt"
    checkpoint.write_text("weights", encoding="utf-8")

    monkeypatch.setattr(api, "SAM3_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api, "SAM3_CHECKPOINT_PATH", None)
    monkeypatch.setattr(api, "active_sam3_checkpoint", str(checkpoint))
    monkeypatch.setattr(api, "active_sam3_model_id", "active_sam3_run")
    monkeypatch.setattr(api, "active_sam3_enable_segmentation", False)
    monkeypatch.setattr(
        api,
        "active_sam3_metadata",
        {
            "id": "active_sam3_run",
            "label": "Active SAM3 Run",
            "checkpoint": str(checkpoint),
            "source": "custom",
            "enable_segmentation": False,
        },
    )

    out = api.delete_sam3_run(run_id, scope="all")

    assert out["deleted"] == [str(run_dir)]
    assert not run_dir.exists()
    assert api.active_sam3_checkpoint is None
    assert api.active_sam3_model_id == "default"
    assert api.active_sam3_enable_segmentation is True
    assert api.active_sam3_metadata["id"] == "default"


def test_list_sam3_models_clears_missing_custom_active_checkpoint(tmp_path: Path, monkeypatch) -> None:
    missing_checkpoint = tmp_path / "missing" / "last.ckpt"

    monkeypatch.setattr(api, "SAM3_JOB_ROOT", tmp_path / "sam3_runs")
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3_runs" / "datasets")
    monkeypatch.setattr(api, "SAM3_CHECKPOINT_PATH", None)
    monkeypatch.setattr(api, "active_sam3_checkpoint", str(missing_checkpoint))
    monkeypatch.setattr(api, "active_sam3_model_id", "missing_custom")
    monkeypatch.setattr(api, "active_sam3_enable_segmentation", False)
    monkeypatch.setattr(
        api,
        "active_sam3_metadata",
        {
            "id": "missing_custom",
            "label": "Missing Custom",
            "checkpoint": str(missing_checkpoint),
            "source": "custom",
            "enable_segmentation": False,
        },
    )

    models = api.list_sam3_available_models()

    assert models[0]["active"] is True
    assert models[0]["key"] == "base"
    assert api.active_sam3_checkpoint is None
    assert api.active_sam3_model_id == "default"


def test_list_sam3_models_skips_checkpoint_symlink_escape(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "runs" / "run_symlink"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    outside = tmp_path / "outside.ckpt"
    outside.write_text("weights", encoding="utf-8")
    try:
        (checkpoint_dir / "last.ckpt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "SAM3_JOB_ROOT", tmp_path / "runs")
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "runs" / "datasets")
    monkeypatch.setattr(api, "SAM3_CHECKPOINT_PATH", None)
    monkeypatch.setattr(api, "active_sam3_checkpoint", None)

    models = api.list_sam3_available_models()

    assert all(model.get("id") != "run_symlink" for model in models)


def test_promote_sam3_run_rejects_checkpoint_symlink_escape(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "run_symlink"
    run_dir = tmp_path / run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    outside = tmp_path / "outside.ckpt"
    outside.write_text("weights", encoding="utf-8")
    try:
        (checkpoint_dir / "last.ckpt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "SAM3_JOB_ROOT", tmp_path)

    with pytest.raises(api.HTTPException) as excinfo:
        api.promote_sam3_run(run_id)

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "sam3_checkpoints_missing"
    assert outside.read_text(encoding="utf-8") == "weights"


def test_sam3_train_cache_purge_blocks_active_split_job(tmp_path: Path, monkeypatch) -> None:
    sam3_root = tmp_path / "sam3_training"
    split_root = sam3_root / "splits" / "job-active"
    ann_path = split_root / "train" / "_annotations.coco.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path.write_text("{}", encoding="utf-8")
    job = api.Sam3TrainingJob(
        job_id="job-active",
        status="queued",
        config={
            "paths": {
                "train_img_folder": str(split_root / "train"),
                "train_ann_file": str(ann_path),
            }
        },
    )
    monkeypatch.setattr(api, "SAM3_JOB_ROOT", sam3_root)
    with api.SAM3_TRAINING_JOBS_LOCK:
        api.SAM3_TRAINING_JOBS.clear()
        api.SAM3_TRAINING_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as excinfo:
            api.sam3_train_cache_purge()
        assert excinfo.value.status_code == 409
        assert excinfo.value.detail == "sam3_cache_purge_blocked_active_jobs"
        assert split_root.exists()
    finally:
        with api.SAM3_TRAINING_JOBS_LOCK:
            api.SAM3_TRAINING_JOBS.clear()


def test_sam3_train_cache_purge_reports_bytes_and_entries(tmp_path: Path, monkeypatch) -> None:
    sam3_root = tmp_path / "sam3_training"
    split_root = sam3_root / "splits" / "job-done"
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "payload.bin").write_bytes(b"abcdef")
    monkeypatch.setattr(api, "SAM3_JOB_ROOT", sam3_root)
    with api.SAM3_TRAINING_JOBS_LOCK:
        api.SAM3_TRAINING_JOBS.clear()

    out = api.sam3_train_cache_purge()

    assert out == {"status": "ok", "deleted_bytes": 6, "deleted_entries": 1}
    assert not split_root.exists()


def test_sam3_train_cache_purge_unlinks_symlink_directory_without_target_delete(
    tmp_path: Path, monkeypatch
) -> None:
    sam3_root = tmp_path / "sam3_training"
    split_parent = sam3_root / "splits"
    split_parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_split"
    outside.mkdir()
    (outside / "payload.bin").write_bytes(b"external")
    try:
        (split_parent / "linked").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "SAM3_JOB_ROOT", sam3_root)
    with api.SAM3_TRAINING_JOBS_LOCK:
        api.SAM3_TRAINING_JOBS.clear()

    out = api.sam3_train_cache_purge()

    assert out == {"status": "ok", "deleted_bytes": 0, "deleted_entries": 1}
    assert not (split_parent / "linked").exists()
    assert (outside / "payload.bin").read_bytes() == b"external"


def test_sam3_training_job_cleans_split_when_config_build_fails(
    tmp_path: Path, monkeypatch
) -> None:
    sam3_root = tmp_path / "sam3_training"
    monkeypatch.setattr(api, "SAM3_JOB_ROOT", sam3_root)
    monkeypatch.setattr(api, "_resolve_sam3_dataset_meta", lambda _dataset_id: {"id": "demo"})

    def failing_build(payload, meta, job_id, prep_logs):
        split_root = api._sam3_training_split_root(job_id)
        split_root.mkdir(parents=True, exist_ok=True)
        (split_root / "payload.bin").write_bytes(b"split")
        raise api.HTTPException(status_code=409, detail="run_name_exists")

    monkeypatch.setattr(api, "_build_sam3_config", failing_build)
    with api.SAM3_TRAINING_JOBS_LOCK:
        api.SAM3_TRAINING_JOBS.clear()

    with pytest.raises(api.HTTPException) as excinfo:
        api.create_sam3_training_job(api.Sam3TrainRequest(dataset_id="demo"))

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail == "run_name_exists"
    split_parent = sam3_root / "splits"
    assert not split_parent.exists() or list(split_parent.iterdir()) == []
    assert api.SAM3_TRAINING_JOBS == {}


def test_sam3_training_job_cleans_split_when_worker_start_fails(
    tmp_path: Path, monkeypatch
) -> None:
    sam3_root = tmp_path / "sam3_training"
    monkeypatch.setattr(api, "SAM3_JOB_ROOT", sam3_root)
    monkeypatch.setattr(api, "_resolve_sam3_dataset_meta", lambda _dataset_id: {"id": "demo"})

    def build_config(payload, meta, job_id, prep_logs):
        split_root = api._sam3_training_split_root(job_id)
        split_root.mkdir(parents=True, exist_ok=True)
        (split_root / "payload.bin").write_bytes(b"split")
        cfg = api.OmegaConf.create(
            {
                "paths": {"experiment_log_dir": str(sam3_root / "demo-run")},
                "launcher": {"gpus_per_node": 1},
                "trainer": {},
                "scratch": {},
            }
        )
        return cfg, 1

    def fail_start(*args, **kwargs):
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(api, "_build_sam3_config", build_config)
    monkeypatch.setattr(api, "_start_sam3_training_worker", fail_start)
    with api.SAM3_TRAINING_JOBS_LOCK:
        api.SAM3_TRAINING_JOBS.clear()

    with pytest.raises(RuntimeError, match="thread start failed"):
        api.create_sam3_training_job(api.Sam3TrainRequest(dataset_id="demo"))

    split_parent = sam3_root / "splits"
    assert not split_parent.exists() or list(split_parent.iterdir()) == []
    assert api.SAM3_TRAINING_JOBS == {}
