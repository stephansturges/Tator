from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import localinferenceapi as api
from services.detectors import _yolo_write_head_graft_yaml_impl


def test_yolo_head_graft_yaml_builder_emits_concat_head(tmp_path: Path) -> None:
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        yaml.safe_dump(
            {
                "nc": 2,
                "backbone": [],
                "head": [[-1, 1, "Detect", [2, [64, 128, 256]]]],
            },
            sort_keys=False,
        )
    )

    out = _yolo_write_head_graft_yaml_impl(
        tmp_path,
        "yolov8n",
        2,
        3,
        variant_base_yaml_fn=lambda _variant, _task, run_dir=None: base_yaml,
        yaml_load_fn=yaml.safe_load,
        yaml_dump_fn=lambda payload: yaml.safe_dump(payload, sort_keys=False),
        http_exception_cls=RuntimeError,
    )

    payload = yaml.safe_load(out.read_text())
    assert payload["nc"] == 5
    assert payload["head"][-2][2] == "Detect"
    assert payload["head"][-2][3][0] == 3
    assert payload["head"][-1] == [[-2, -1], 1, "ConcatHead", [2, 3]]


def test_yolo_head_graft_dry_run_reports_missing_best(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path / "base_run"
    base_dir.mkdir(parents=True, exist_ok=True)
    new_labelmap = tmp_path / "new_labelmap.txt"
    new_labelmap.write_text("new_class\n", encoding="utf-8")

    monkeypatch.setattr(api, "_yolo_run_dir_impl", lambda *_args, **_kwargs: base_dir)
    monkeypatch.setattr(
        api,
        "_yolo_load_run_meta_impl",
        lambda *_args, **_kwargs: {"config": {"task": "detect", "variant": "yolov8n"}},
    )
    monkeypatch.setattr(api, "_yolo_load_run_labelmap_impl", lambda *_args, **_kwargs: ["base_class"])
    monkeypatch.setattr(
        api,
        "_resolve_yolo_training_dataset",
        lambda _payload: {
            "yolo_ready": True,
            "task": "detect",
            "yolo_labelmap_path": str(new_labelmap),
            "id": "dataset_1",
        },
    )

    payload = api.YoloHeadGraftDryRunRequest(base_run_id="base_run", dataset_root=str(tmp_path))
    out = api.yolo_head_graft_dry_run(payload)
    assert out["ok"] is False
    assert out["error"] == "yolo_base_missing_best"
    assert out["base_best_exists"] is False


def test_yolo_head_graft_dry_run_reports_missing_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "base_run"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "best.pt").write_text("weights", encoding="utf-8")

    monkeypatch.setattr(api, "_yolo_run_dir_impl", lambda *_args, **_kwargs: base_dir)
    monkeypatch.setattr(
        api,
        "_yolo_load_run_meta_impl",
        lambda *_args, **_kwargs: {"config": {"task": "detect"}},
    )

    payload = api.YoloHeadGraftDryRunRequest(base_run_id="base_run", dataset_root=str(tmp_path))
    out = api.yolo_head_graft_dry_run(payload)
    assert out["ok"] is False
    assert out["error"] == "yolo_base_variant_missing"
    assert out["base_variant_present"] is False


def test_yolo_head_graft_dry_run_reports_missing_base_labelmap_before_dataset_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "base_run"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "best.pt").write_text("weights", encoding="utf-8")

    monkeypatch.setattr(api, "_yolo_run_dir_impl", lambda *_args, **_kwargs: base_dir)
    monkeypatch.setattr(
        api,
        "_yolo_load_run_meta_impl",
        lambda *_args, **_kwargs: {"config": {"task": "detect", "variant": "yolov8n"}},
    )
    monkeypatch.setattr(api, "_yolo_load_run_labelmap_impl", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        api,
        "_resolve_yolo_training_dataset",
        lambda _payload: (_ for _ in ()).throw(AssertionError("dataset resolution should not run")),
    )

    payload = api.YoloHeadGraftDryRunRequest(base_run_id="base_run", dataset_root=str(tmp_path))
    out = api.yolo_head_graft_dry_run(payload)
    assert out["ok"] is False
    assert out["error"] == "yolo_base_labelmap_missing"


def test_yolo_head_graft_bundle_requires_required_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "head_graft_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(api, "_yolo_run_dir_impl", lambda *_args, **_kwargs: run_dir)
    monkeypatch.setattr(
        api,
        "_yolo_load_run_meta_impl",
        lambda *_args, **_kwargs: {
            "head_graft": {"base_run_id": "base_run"},
            "config": {"run_name": "head_graft_demo"},
            "job_id": "job_1",
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.download_yolo_head_graft_bundle("job_1")
    assert exc.value.status_code == 412
    assert isinstance(exc.value.detail, dict)
    assert exc.value.detail.get("error") == "yolo_head_graft_bundle_incomplete"
    missing = exc.value.detail.get("missing") or []
    assert "best.pt" in missing
    assert "labelmap.txt" in missing


def test_cancel_yolo_head_graft_job_sets_cancelling_state(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = {}
    monkeypatch.setattr(api, "YOLO_HEAD_GRAFT_JOBS", jobs)
    monkeypatch.setattr(api, "_yolo_head_graft_audit", lambda *_args, **_kwargs: None)

    job = api.YoloHeadGraftJob(job_id="job_cancel", status="running", message="Running")
    with api.YOLO_HEAD_GRAFT_JOBS_LOCK:
        jobs[job.job_id] = job

    out = api.cancel_yolo_head_graft_job(job.job_id)
    assert out["status"] == "cancelling"
    assert job.cancel_event.is_set()
    assert job.status == "cancelling"


def test_yolo_active_requires_head_graft_patch_is_false_for_standard_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_standard"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_path.write_text("weights", encoding="utf-8")

    monkeypatch.setattr(api, "_yolo_load_run_meta_impl", lambda *_args, **_kwargs: {"config": {}})
    assert api._yolo_active_requires_head_graft_patch({"best_path": str(best_path)}) is False


def test_yolo_active_requires_head_graft_patch_is_true_for_head_graft_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_graft"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_path.write_text("weights", encoding="utf-8")

    monkeypatch.setattr(
        api,
        "_yolo_load_run_meta_impl",
        lambda *_args, **_kwargs: {"head_graft": {"base_run_id": "base_run"}},
    )
    assert api._yolo_active_requires_head_graft_patch({"best_path": str(best_path)}) is True
