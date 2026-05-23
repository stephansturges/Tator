from pathlib import Path

import pytest

import localinferenceapi as api


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
