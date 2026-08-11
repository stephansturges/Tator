from __future__ import annotations

import json
from pathlib import Path

import pytest

import localinferenceapi as api


def _set_missing_custom_qwen_active(tmp_path: Path, monkeypatch) -> Path:
    missing = tmp_path / "deleted_qwen_run" / "latest"
    monkeypatch.setattr(api, "active_qwen_model_id", "deleted_custom")
    monkeypatch.setattr(api, "active_qwen_model_path", missing)
    monkeypatch.setattr(
        api,
        "active_qwen_metadata",
        {
            "id": "deleted_custom",
            "label": "Deleted Custom",
            "model_id": "Qwen/Qwen3-VL-4B-Instruct",
            "runtime_platform": api.QWEN_PLATFORM_TRANSFORMERS,
        },
    )
    monkeypatch.setattr(api, "qwen_model", object())
    monkeypatch.setattr(api, "qwen_processor", object())
    monkeypatch.setattr(api, "loaded_qwen_model_id", "deleted_custom")
    return missing


def test_list_qwen_models_resets_missing_active_custom_adapter(tmp_path: Path, monkeypatch) -> None:
    missing = _set_missing_custom_qwen_active(tmp_path, monkeypatch)

    out = api.list_qwen_models()

    assert not missing.exists()
    assert out["active"] == "default"
    default = next(entry for entry in out["models"] if entry.get("id") == "default")
    assert default["active"] is True
    assert api.active_qwen_model_id == "default"
    assert api.active_qwen_model_path is None
    assert api.qwen_model is None
    assert api.qwen_processor is None
    assert api.loaded_qwen_model_id is None


def test_qwen_status_resets_missing_active_custom_adapter(tmp_path: Path, monkeypatch) -> None:
    _set_missing_custom_qwen_active(tmp_path, monkeypatch)

    out = api.qwen_status()

    assert out["active_model"] == "default"
    assert api.active_qwen_model_id == "default"
    assert api.active_qwen_model_path is None
    assert api.qwen_model is None


def test_qwen_status_resets_corrupt_active_custom_adapter(tmp_path: Path, monkeypatch) -> None:
    corrupt = tmp_path / "corrupt_qwen_run" / "latest"
    corrupt.mkdir(parents=True, exist_ok=True)
    _set_missing_custom_qwen_active(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "active_qwen_model_path", corrupt)

    out = api.qwen_status()

    assert out["active_model"] == "default"
    assert api.active_qwen_model_path is None
    assert api.qwen_model is None


def test_qwen_train_cache_purge_blocks_active_split_job(tmp_path: Path, monkeypatch) -> None:
    qwen_root = tmp_path / "qwen_training"
    split_root = qwen_root / "splits" / "job-active"
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "annotations.jsonl").write_text("{}", encoding="utf-8")
    job = api.QwenTrainingJob(
        job_id="job-active",
        status="running",
        config={"dataset_root": str(split_root)},
    )
    monkeypatch.setattr(api, "QWEN_JOB_ROOT", qwen_root)
    with api.QWEN_TRAINING_JOBS_LOCK:
        api.QWEN_TRAINING_JOBS.clear()
        api.QWEN_TRAINING_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as excinfo:
            api.qwen_train_cache_purge()
        assert excinfo.value.status_code == 409
        assert excinfo.value.detail == "qwen_cache_purge_blocked_active_jobs"
        assert split_root.exists()
    finally:
        with api.QWEN_TRAINING_JOBS_LOCK:
            api.QWEN_TRAINING_JOBS.clear()


def test_qwen_train_cache_purge_reports_bytes_and_entries(tmp_path: Path, monkeypatch) -> None:
    qwen_root = tmp_path / "qwen_training"
    split_root = qwen_root / "splits" / "job-done"
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "payload.bin").write_bytes(b"abcd")
    monkeypatch.setattr(api, "QWEN_JOB_ROOT", qwen_root)
    with api.QWEN_TRAINING_JOBS_LOCK:
        api.QWEN_TRAINING_JOBS.clear()

    out = api.qwen_train_cache_purge()

    assert out == {"status": "ok", "deleted_bytes": 4, "deleted_entries": 1}
    assert not split_root.exists()


def test_qwen_train_cache_purge_unlinks_symlink_directory_without_target_delete(
    tmp_path: Path, monkeypatch
) -> None:
    qwen_root = tmp_path / "qwen_training"
    split_parent = qwen_root / "splits"
    split_parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_split"
    outside.mkdir()
    (outside / "payload.bin").write_bytes(b"external")
    try:
        (split_parent / "linked").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "QWEN_JOB_ROOT", qwen_root)
    with api.QWEN_TRAINING_JOBS_LOCK:
        api.QWEN_TRAINING_JOBS.clear()

    out = api.qwen_train_cache_purge()

    assert out == {"status": "ok", "deleted_bytes": 0, "deleted_entries": 1}
    assert not (split_parent / "linked").exists()
    assert (outside / "payload.bin").read_bytes() == b"external"


def test_qwen_train_cache_purge_reports_cleanup_failure(
    tmp_path: Path, monkeypatch
) -> None:
    qwen_root = tmp_path / "qwen_training"
    split_root = qwen_root / "splits" / "job-cleanup-fail"
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "payload.bin").write_bytes(b"abcd")
    monkeypatch.setattr(api, "QWEN_JOB_ROOT", qwen_root)
    with api.QWEN_TRAINING_JOBS_LOCK:
        api.QWEN_TRAINING_JOBS.clear()

    def fail_rmtree(path, *args, **kwargs):
        if Path(path) == split_root:
            raise OSError("forced purge failure")
        raise AssertionError(f"unexpected rmtree path: {path}")

    monkeypatch.setattr(api.shutil, "rmtree", fail_rmtree)

    with pytest.raises(api.HTTPException) as excinfo:
        api.qwen_train_cache_purge()

    assert excinfo.value.status_code == 500
    assert str(excinfo.value.detail).startswith("qwen_cache_purge_failed:")
    assert split_root.exists()


def test_qwen_train_cache_size_ignores_symlinked_split_root(
    tmp_path: Path, monkeypatch
) -> None:
    qwen_root = tmp_path / "qwen_training"
    qwen_root.mkdir()
    outside = tmp_path / "outside_split_root"
    outside.mkdir()
    (outside / "payload.bin").write_bytes(b"external")
    try:
        (qwen_root / "splits").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "QWEN_JOB_ROOT", qwen_root)

    assert api.qwen_train_cache_size() == {"bytes": 0}


def test_qwen_training_split_root_rejects_symlinked_job_root_parent(
    tmp_path: Path, monkeypatch
) -> None:
    outside = tmp_path / "outside_jobs"
    outside.mkdir()
    link_parent = tmp_path / "linked_parent"
    try:
        link_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "QWEN_JOB_ROOT", link_parent / "qwen_training")

    with pytest.raises(api.HTTPException) as excinfo:
        api._qwen_training_split_root("job-parent-link")

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen_split_path_invalid"
    assert list(outside.iterdir()) == []


def test_qwen_train_cache_purge_unlinks_symlinked_split_root_without_target_delete(
    tmp_path: Path, monkeypatch
) -> None:
    qwen_root = tmp_path / "qwen_training"
    qwen_root.mkdir()
    outside = tmp_path / "outside_split_root"
    outside.mkdir()
    (outside / "payload.bin").write_bytes(b"external")
    split_root = qwen_root / "splits"
    try:
        split_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "QWEN_JOB_ROOT", qwen_root)
    with api.QWEN_TRAINING_JOBS_LOCK:
        api.QWEN_TRAINING_JOBS.clear()

    out = api.qwen_train_cache_purge()

    assert out == {"status": "ok", "deleted_bytes": 0, "deleted_entries": 1}
    assert not split_root.exists()
    assert (outside / "payload.bin").read_bytes() == b"external"


def test_qwen_training_split_cleanup_unlinks_symlink_without_target_delete(
    tmp_path: Path, monkeypatch
) -> None:
    qwen_root = tmp_path / "qwen_training"
    split_parent = qwen_root / "splits"
    split_parent.mkdir(parents=True, exist_ok=True)
    target = split_parent / "target"
    target.mkdir()
    (target / "payload.bin").write_bytes(b"target")
    link = split_parent / "job-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "QWEN_JOB_ROOT", qwen_root)

    api._cleanup_qwen_training_split("job-link")

    assert not link.exists()
    assert not link.is_symlink()
    assert (target / "payload.bin").read_bytes() == b"target"


def test_delete_qwen_dataset_blocks_active_training_reference(
    tmp_path: Path, monkeypatch
) -> None:
    qwen_root = tmp_path / "qwen_datasets"
    dataset_root = qwen_root / "demo"
    dataset_root.mkdir(parents=True, exist_ok=True)
    job = api.QwenTrainingJob(
        job_id="job-active",
        status="running",
        config={"dataset_root": str(dataset_root)},
    )
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    with api.QWEN_TRAINING_JOBS_LOCK:
        api.QWEN_TRAINING_JOBS.clear()
        api.QWEN_TRAINING_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as excinfo:
            api.delete_qwen_dataset("demo")
        assert excinfo.value.status_code == 409
        assert excinfo.value.detail == "qwen_dataset_delete_blocked_active_jobs:qwen_training"
        assert dataset_root.exists()
    finally:
        with api.QWEN_TRAINING_JOBS_LOCK:
            api.QWEN_TRAINING_JOBS.clear()


def test_delete_qwen_dataset_rejects_symlinked_dataset_id_without_target_delete(
    tmp_path: Path, monkeypatch
) -> None:
    qwen_root = tmp_path / "qwen_datasets"
    qwen_root.mkdir()
    target_root = qwen_root / "target"
    target_root.mkdir()
    (target_root / "payload.bin").write_bytes(b"target")
    try:
        (qwen_root / "linked").symlink_to(target_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)

    with pytest.raises(api.HTTPException) as excinfo:
        api.delete_qwen_dataset("linked")

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen_dataset_delete_forbidden"
    assert (target_root / "payload.bin").read_bytes() == b"target"


def test_delete_qwen_dataset_moves_to_managed_trash_and_restores(
    tmp_path: Path, monkeypatch
) -> None:
    qwen_root = tmp_path / "qwen_datasets"
    dataset_root = qwen_root / "demo"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "payload.bin").write_bytes(b"qwen")
    (dataset_root / api.QWEN_METADATA_FILENAME).write_text(
        json.dumps({"id": "demo", "label": "Demo Qwen"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", tmp_path / "registry")
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3_datasets")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)

    out = api.delete_qwen_dataset("demo")

    assert out["status"] == "trashed"
    assert out["storage_mode"] == "managed"
    assert out["restore_available"] is True
    assert not dataset_root.exists()
    trash_entries = api.list_dataset_trash_entries()
    assert [entry["trash_id"] for entry in trash_entries] == [out["trash_id"]]
    assert trash_entries[0]["source"] == "qwen"
    assert Path(trash_entries[0]["dataset_path"], "payload.bin").read_bytes() == b"qwen"

    restored = api.restore_dataset_trash_entry(out["trash_id"])

    assert restored["status"] == "restored"
    assert restored["id"] == "demo"
    assert dataset_root.exists()
    assert (dataset_root / "payload.bin").read_bytes() == b"qwen"
    restored_meta = json.loads(
        (dataset_root / api.QWEN_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert restored_meta["id"] == "demo"
    assert restored_meta["restored_from_trash_id"] == out["trash_id"]
    assert api.list_dataset_trash_entries() == []


def test_delete_dataset_entry_blocks_active_training_reference(
    tmp_path: Path, monkeypatch
) -> None:
    registry_root = tmp_path / "datasets"
    dataset_root = registry_root / "managed-demo"
    dataset_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3_datasets")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen_datasets")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda dataset_id: {
            "id": dataset_id,
            "storage_mode": "managed",
            "registry_root": str(dataset_root),
            "dataset_root": str(dataset_root),
        },
    )
    job = api.QwenTrainingJob(
        job_id="job-active",
        status="queued",
        config={"dataset_root": str(dataset_root / "train")},
    )
    with api.QWEN_TRAINING_JOBS_LOCK:
        api.QWEN_TRAINING_JOBS.clear()
        api.QWEN_TRAINING_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as excinfo:
            api.delete_dataset_entry("managed-demo")
        assert excinfo.value.status_code == 409
        assert excinfo.value.detail == "dataset_delete_blocked_active_jobs:qwen_training"
        assert dataset_root.exists()
    finally:
        with api.QWEN_TRAINING_JOBS_LOCK:
            api.QWEN_TRAINING_JOBS.clear()
