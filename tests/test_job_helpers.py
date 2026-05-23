import json
from types import SimpleNamespace

import numpy as np
import pytest

from services.classifier_jobs import _clip_job_update_impl, _serialize_clip_job_impl
from services.calibration import _serialize_calibration_job
from services.detector_jobs import (
    _rfdetr_job_update_impl,
    _serialize_yolo_job_impl,
    _yolo_head_graft_audit_impl,
    _yolo_head_graft_job_update_impl,
    _yolo_job_update_impl,
)
from services.job_payloads import clamp_progress, json_sanitize
from services.prompt_helper import _serialize_prompt_helper_job_impl
from services.qwen_jobs import _qwen_job_update_impl, _serialize_qwen_job_impl
from services.sam3_jobs import _sam3_job_update_impl, _serialize_sam3_job_impl
from services.segmentation import _seg_job_update_impl, _serialize_seg_job_impl


def _job(**overrides):
    base = {
        "job_id": "job123456",
        "status": "queued",
        "progress": 0.0,
        "message": "Queued",
        "config": {},
        "logs": [],
        "metrics": [],
        "artifacts": None,
        "result": None,
        "error": None,
        "created_at": 1.0,
        "updated_at": 1.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_clip_job_update_refreshes_timestamp_without_message_change():
    job = _job(message="same")

    _clip_job_update_impl(job, progress=0.5, message="same", max_logs=10)

    assert job.progress == 0.5
    assert job.updated_at > 1.0
    assert job.logs == []


def test_job_serializers_replace_nonfinite_payload_values():
    metric = {"loss": float("nan"), "nested": [float("inf"), 0.25]}

    assert _serialize_clip_job_impl(_job(metrics=[metric]))["metrics"] == [
        {"loss": None, "nested": [None, 0.25]}
    ]
    assert _serialize_qwen_job_impl(_job(config={"lr": float("-inf")}))["config"] == {
        "lr": None
    }
    assert _serialize_sam3_job_impl(_job(result={"ap": float("nan")}))["result"] == {
        "ap": None
    }
    assert _serialize_yolo_job_impl(_job(metrics=[metric]))["metrics"] == [
        {"loss": None, "nested": [None, 0.25]}
    ]


def test_shared_json_sanitize_handles_numpy_arrays_and_scalars():
    assert json_sanitize({"values": np.array([1.0, float("nan")]), "count": np.int64(3)}) == {
        "values": [1.0, None],
        "count": 3,
    }


def test_clamp_progress_does_not_promote_nonfinite_values_to_complete():
    assert clamp_progress(float("nan"), fallback=0.4) == 0.4
    assert clamp_progress(float("inf"), fallback=0.4) == 0.4
    assert clamp_progress(1.5) == 1.0
    assert clamp_progress(-0.5) == 0.0


def test_job_updates_ignore_nonfinite_progress():
    update_calls = [
        lambda job: _clip_job_update_impl(job, progress=float("nan"), max_logs=10),
        lambda job: _yolo_job_update_impl(job, progress=float("nan")),
        lambda job: _rfdetr_job_update_impl(job, progress=float("nan")),
        lambda job: _yolo_head_graft_job_update_impl(job, progress=float("nan")),
        lambda job: _qwen_job_update_impl(job, progress=float("nan"), max_logs=10),
        lambda job: _sam3_job_update_impl(job, progress=float("nan"), max_logs=10),
        lambda job: _seg_job_update_impl(job, progress=float("nan"), max_logs=10),
    ]
    for call in update_calls:
        job = _job(progress=0.4)
        call(job)
        assert job.progress == 0.4


def test_remaining_service_job_serializers_sanitize_payloads():
    job = _job(
        progress=float("nan"),
        request={"temperature": float("inf")},
        result={"score": float("-inf")},
        logs=[{"value": float("nan")}],
        config={"threshold": float("inf")},
    )
    job.total_steps = 1
    job.completed_steps = 0
    job.phase = "running"
    job.processed = 0
    job.total = 1

    assert _serialize_seg_job_impl(job)["config"] == {"threshold": None}
    assert _serialize_seg_job_impl(job)["progress"] is None
    assert _serialize_prompt_helper_job_impl(job)["request"] == {"temperature": None}
    assert _serialize_calibration_job(job)["result"] == {"score": None}


def test_local_job_serializers_sanitize_nested_status_payloads():
    import localinferenceapi as api

    job = _job(
        progress=float("nan"),
        request={"temperature": float("inf")},
        result={"summary": {"series": np.array([0.1, float("nan")])}},
        logs=[{"value": float("-inf")}],
    )
    job.kind = "analysis"

    assert api._serialize_class_analysis_job(job)["summary"] == {"series": [0.1, None]}
    assert api._serialize_class_analysis_job(job)["request"] == {"temperature": None}
    assert api._serialize_data_ingestion_job(job)["summary"] == {"series": [0.1, None]}
    assert api._serialize_agent_mining_job(job)["logs"] == [{"value": None}]
    assert api._serialize_auto_label_job(job)["progress"] is None


def test_local_job_updates_ignore_nonfinite_progress():
    import localinferenceapi as api

    class_job = _job(progress=0.35)
    data_job = _job(progress=0.35)
    api._class_analysis_update(class_job, progress=float("nan"))
    api._data_ingestion_update(data_job, progress=float("inf"))

    assert class_job.progress == 0.35
    assert data_job.progress == 0.35


def test_yolo_head_graft_audit_replaces_symlinked_log(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("external\n", encoding="utf-8")
    audit_path = run_dir / "head_graft_audit.jsonl"
    audit_path.symlink_to(outside)
    job = _job(config={"paths": {"run_dir": str(run_dir)}})

    _yolo_head_graft_audit_impl(job, "hello", time_fn=lambda: 123.0)

    assert outside.read_text(encoding="utf-8") == "external\n"
    assert not audit_path.is_symlink()
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"event": "log", "level": "info", "message": "hello", "timestamp": 123.0}
    ]


def test_yolo_head_graft_audit_rejects_symlinked_run_parent_without_target_write(tmp_path):
    outside = tmp_path / "outside_parent"
    outside_run = outside / "run"
    outside_run.mkdir(parents=True)
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    job = _job(config={"paths": {"run_dir": str(linked_parent / "run")}})

    _yolo_head_graft_audit_impl(job, "hello", time_fn=lambda: 123.0)

    assert not (outside_run / "head_graft_audit.jsonl").exists()
