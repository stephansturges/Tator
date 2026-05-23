import json
from types import SimpleNamespace

from services.classifier_jobs import _clip_job_update_impl, _serialize_clip_job_impl
from services.detector_jobs import _serialize_yolo_job_impl, _yolo_head_graft_audit_impl
from services.qwen_jobs import _serialize_qwen_job_impl
from services.sam3_jobs import _serialize_sam3_job_impl


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
