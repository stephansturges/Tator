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
