import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import localinferenceapi as api
from services.calibration import (
    CALIBRATION_JOB_STATE_FILENAME,
    CalibrationJob,
    _ensure_prepass_jsonl,
    _prepare_calibration_storage_dir,
    _resolve_calibration_storage_root,
    _write_json_atomic,
    _write_text_atomic,
)


def test_get_calibration_report_bundle_reads_completed_job_artifact(tmp_path: Path):
    report_root = tmp_path / "jobs"
    report_path = report_root / "job-report" / "report_bundle.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"report_kind": "calibration_job", "overall_metrics": {"f1": 0.84}}
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    job = CalibrationJob(job_id="job-report")
    job.result = {"report_bundle_json": str(report_path)}
    original_root = api.CALIBRATION_ROOT
    api.CALIBRATION_ROOT = report_root
    with api.CALIBRATION_JOBS_LOCK:
        api.CALIBRATION_JOBS[job.job_id] = job
    try:
        result = api.get_calibration_report_bundle(job.job_id)
        assert result["report_kind"] == "calibration_job"
        assert result["overall_metrics"]["f1"] == pytest.approx(0.84)
    finally:
        api.CALIBRATION_ROOT = original_root
        with api.CALIBRATION_JOBS_LOCK:
            api.CALIBRATION_JOBS.pop(job.job_id, None)


def test_get_calibration_report_bundle_reads_persisted_artifact_without_live_job(tmp_path: Path):
    job_id = "cal_reportdisk"
    job_dir = api.CALIBRATION_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    report_path = job_dir / "report_bundle.json"
    payload = {"report_kind": "calibration_job", "overall_metrics": {"f1": 0.91}}
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        result = api.get_calibration_report_bundle(job_id)
        assert result["overall_metrics"]["f1"] == pytest.approx(0.91)
    finally:
        report_path.unlink(missing_ok=True)
        job_dir.rmdir()


def test_get_calibration_report_bundle_rejects_symlink_to_sibling_job(
    tmp_path: Path, monkeypatch
) -> None:
    calibration_root = tmp_path / "jobs"
    job_dir = calibration_root / "cal_report"
    sibling_dir = calibration_root / "cal_other"
    job_dir.mkdir(parents=True)
    sibling_dir.mkdir(parents=True)
    sibling_report = sibling_dir / "report_bundle.json"
    sibling_report.write_text('{"escaped": true}', encoding="utf-8")
    try:
        (job_dir / "report_bundle.json").symlink_to(sibling_report)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CALIBRATION_ROOT", calibration_root)
    with api.CALIBRATION_JOBS_LOCK:
        api.CALIBRATION_JOBS.clear()

    with pytest.raises(api.HTTPException) as excinfo:
        api.get_calibration_report_bundle("cal_report")

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "calibration_job_not_found"


def test_get_calibration_report_bundle_rejects_live_job_sibling_result_path(
    tmp_path: Path, monkeypatch
) -> None:
    calibration_root = tmp_path / "jobs"
    job_dir = calibration_root / "cal_live"
    sibling_dir = calibration_root / "cal_other"
    job_dir.mkdir(parents=True)
    sibling_dir.mkdir(parents=True)
    sibling_report = sibling_dir / "report_bundle.json"
    sibling_report.write_text('{"escaped": true}', encoding="utf-8")
    job = CalibrationJob(job_id="cal_live")
    job.result = {"report_bundle_json": str(sibling_report)}
    monkeypatch.setattr(api, "CALIBRATION_ROOT", calibration_root)
    with api.CALIBRATION_JOBS_LOCK:
        api.CALIBRATION_JOBS.clear()
        api.CALIBRATION_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as excinfo:
            api.get_calibration_report_bundle(job.job_id)

        assert excinfo.value.status_code == 404
        assert excinfo.value.detail == "calibration_job_not_found"
    finally:
        with api.CALIBRATION_JOBS_LOCK:
            api.CALIBRATION_JOBS.clear()


def test_get_calibration_report_bundle_404_for_missing_job():
    with pytest.raises(api.HTTPException) as excinfo:
        api.get_calibration_report_bundle("missing-job")
    assert excinfo.value.status_code == 404


def test_get_calibration_job_reads_persisted_state_without_live_job(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(api, "CALIBRATION_ROOT", tmp_path)
    with api.CALIBRATION_JOBS_LOCK:
        api.CALIBRATION_JOBS.clear()
    job_dir = tmp_path / "cal_persisted"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / CALIBRATION_JOB_STATE_FILENAME).write_text(
        json.dumps(
            {
                "job_id": "cal_persisted",
                "status": "completed",
                "message": "Done",
                "phase": "completed",
                "progress": 1.0,
                "processed": 300,
                "total": 300,
                "step_current": 10,
                "step_total": 10,
                "step_label": "Completed",
                "substep_current": 0,
                "substep_total": 0,
                "substep_label": "",
                "created_at": 1.0,
                "updated_at": 2.0,
                "request": {"dataset_id": "demo"},
                "result": {"report_bundle_json": None},
                "error": None,
                "state_schema_version": 1,
            }
        ),
        encoding="utf-8",
    )

    payload = api.get_calibration_job("cal_persisted")

    assert payload["job_id"] == "cal_persisted"
    assert payload["status"] == "completed"
    assert payload["step_total"] == 10


def test_get_calibration_job_rejects_symlinked_state_escape(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(api, "CALIBRATION_ROOT", tmp_path)
    with api.CALIBRATION_JOBS_LOCK:
        api.CALIBRATION_JOBS.clear()
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir(parents=True, exist_ok=True)
    (outside / CALIBRATION_JOB_STATE_FILENAME).write_text(
        json.dumps(
            {
                "job_id": "cal_escape",
                "status": "completed",
                "message": "outside",
                "phase": "completed",
                "progress": 1.0,
                "processed": 1,
                "total": 1,
                "step_current": 1,
                "step_total": 1,
                "step_label": "Done",
                "substep_current": 0,
                "substep_total": 0,
                "substep_label": "",
                "created_at": 1.0,
                "updated_at": 2.0,
                "request": {},
                "result": None,
                "error": None,
                "state_schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    try:
        (tmp_path / "cal_escape").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as excinfo:
        api.get_calibration_job("cal_escape")
    assert excinfo.value.status_code == 404


def test_list_calibration_jobs_marks_persisted_running_jobs_interrupted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(api, "CALIBRATION_ROOT", tmp_path)
    with api.CALIBRATION_JOBS_LOCK:
        api.CALIBRATION_JOBS.clear()
    job_dir = tmp_path / "cal_interrupted"
    job_dir.mkdir(parents=True, exist_ok=True)
    state_path = job_dir / CALIBRATION_JOB_STATE_FILENAME
    state_path.write_text(
        json.dumps(
            {
                "job_id": "cal_interrupted",
                "status": "running",
                "message": "Building features…",
                "phase": "features",
                "progress": 0.4,
                "processed": 120,
                "total": 300,
                "step_current": 5,
                "step_total": 10,
                "step_label": "Build features",
                "substep_current": 0,
                "substep_total": 0,
                "substep_label": "",
                "created_at": 1.0,
                "updated_at": 2.0,
                "request": {"dataset_id": "demo"},
                "result": None,
                "error": None,
                "state_schema_version": 1,
            }
        ),
        encoding="utf-8",
    )

    payload = api.list_calibration_jobs()

    assert payload[0]["job_id"] == "cal_interrupted"
    assert payload[0]["status"] == "failed"
    assert payload[0]["message"] == "Interrupted by backend restart"
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"


def test_list_calibration_jobs_skips_symlinked_state_escape(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(api, "CALIBRATION_ROOT", tmp_path)
    with api.CALIBRATION_JOBS_LOCK:
        api.CALIBRATION_JOBS.clear()
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir(parents=True, exist_ok=True)
    (outside / CALIBRATION_JOB_STATE_FILENAME).write_text(
        json.dumps(
            {
                "job_id": "cal_escape",
                "status": "completed",
                "message": "outside",
                "phase": "completed",
                "progress": 1.0,
                "processed": 1,
                "total": 1,
                "step_current": 1,
                "step_total": 1,
                "step_label": "Done",
                "substep_current": 0,
                "substep_total": 0,
                "substep_label": "",
                "created_at": 1.0,
                "updated_at": 2.0,
                "request": {},
                "result": None,
                "error": None,
                "state_schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    try:
        (tmp_path / "cal_escape").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    payload = api.list_calibration_jobs()

    assert all(item.get("job_id") != "cal_escape" for item in payload)


def test_calibration_write_json_atomic_replaces_tmp_symlink_without_target_write(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "job" / CALIBRATION_JOB_STATE_FILENAME
    state_path.parent.mkdir()
    outside = tmp_path / "outside_tmp.json"
    outside.write_text("external", encoding="utf-8")
    tmp_path_link = state_path.with_suffix(state_path.suffix + ".tmp")
    try:
        tmp_path_link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _write_json_atomic(state_path, {"job_id": "cal_safe"})

    assert not tmp_path_link.exists()
    assert not state_path.is_symlink()
    assert json.loads(state_path.read_text(encoding="utf-8"))["job_id"] == "cal_safe"
    assert outside.read_text(encoding="utf-8") == "external"


def test_calibration_write_json_atomic_replaces_final_symlink_without_target_write(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "job" / CALIBRATION_JOB_STATE_FILENAME
    state_path.parent.mkdir()
    outside = tmp_path / "outside_state.json"
    outside.write_text("external", encoding="utf-8")
    try:
        state_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _write_json_atomic(state_path, {"job_id": "cal_safe"})

    assert not state_path.is_symlink()
    assert json.loads(state_path.read_text(encoding="utf-8"))["job_id"] == "cal_safe"
    assert outside.read_text(encoding="utf-8") == "external"


def test_calibration_write_text_atomic_replaces_symlink_targets_without_target_write(
    tmp_path: Path,
) -> None:
    eval_path = tmp_path / "job" / "ensemble_xgb.eval.json"
    eval_path.parent.mkdir()
    outside_tmp = tmp_path / "outside_tmp.json"
    outside_final = tmp_path / "outside_final.json"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    tmp_link = eval_path.with_suffix(eval_path.suffix + ".tmp")
    try:
        tmp_link.symlink_to(outside_tmp)
        eval_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _write_text_atomic(eval_path, '{"f1": 1.0}')

    assert not tmp_link.exists()
    assert not eval_path.is_symlink()
    assert eval_path.read_text(encoding="utf-8") == '{"f1": 1.0}'
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_calibration_prepass_jsonl_keeps_existing_output_when_cache_incomplete(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "job" / "prepass.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("old final\n", encoding="utf-8")
    job = SimpleNamespace(cancel_event=SimpleNamespace(is_set=lambda: False))

    with pytest.raises(RuntimeError, match="prepass_cache_incomplete"):
        _ensure_prepass_jsonl(
            job=job,
            update_fn=lambda *_args, **_kwargs: None,
            selected=["missing.jpg"],
            total=0,
            dataset_id="dataset",
            labelmap=["object"],
            glossary="",
            prepass_payload=SimpleNamespace(sam_variant=None),
            prepass_config={},
            prepass_config_for_hash={},
            output_path=output_path,
            calibration_cache_root=tmp_path / "calibration_cache",
            write_record_fn=_write_json_atomic,
            hash_payload_fn=lambda payload: "cachekey",
            prepass_worker_fn=lambda *_args, **_kwargs: None,
            unload_inference_runtimes_fn=lambda: None,
            resolve_dataset_fn=lambda _dataset_id: tmp_path / "dataset",
            cache_image_fn=lambda *_args, **_kwargs: "token",
            run_prepass_fn=lambda *_args, **_kwargs: {"detections": []},
        )

    assert output_path.read_text(encoding="utf-8") == "old final\n"
    assert not output_path.with_suffix(output_path.suffix + ".tmp").exists()


def test_resolve_calibration_storage_root_rejects_nested_symlinked_parent_without_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="calibration_root_symlink"):
        _resolve_calibration_storage_root(linked_parent / "nested" / "calibration_jobs", create=True)

    assert list(outside.iterdir()) == []


def test_calibration_write_json_atomic_rejects_nested_symlinked_parent_without_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="calibration_json_parent_symlink"):
        _write_json_atomic(
            linked_parent / "nested" / "calibration_jobs" / "cal_safe" / CALIBRATION_JOB_STATE_FILENAME,
            {"job_id": "cal_safe"},
        )

    assert list(outside.iterdir()) == []


def test_prepare_calibration_storage_dir_rejects_nested_symlinked_parent_without_write(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "calibration_cache"
    cache_root.mkdir()
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = cache_root / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="calibration_cache_dir_symlink"):
        _prepare_calibration_storage_dir(
            linked_parent / "nested" / "images",
            root=cache_root,
            error_prefix="calibration_cache_dir",
        )

    assert list(outside.iterdir()) == []


def test_ensemble_filter_rejects_traversal_job_id_before_artifact_resolution(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(api, "CALIBRATION_ROOT", tmp_path / "calibration_jobs")
    monkeypatch.setattr(
        api,
        "_resolve_agent_clip_classifier_path_impl",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid job_id should stop before classifier resolution")
        ),
    )
    detections = [{"bbox": [0, 0, 1, 1], "label": "car"}]
    warnings = []

    result = api._agent_apply_ensemble_filter(
        detections,
        dataset_id="dataset",
        image_name="image.jpg",
        classifier_id="head.pkl",
        job_id="../outside",
        warnings=warnings,
    )

    assert result is detections
    assert warnings == ["ensemble_filter_job_invalid"]


def test_ensemble_filter_rejects_symlinked_job_dir_escape(tmp_path: Path, monkeypatch):
    calibration_root = tmp_path / "calibration_jobs"
    outside = tmp_path / "outside_job"
    calibration_root.mkdir()
    outside.mkdir()
    (outside / "ensemble_xgb.json").write_text("{}", encoding="utf-8")
    (outside / "ensemble_xgb.meta.json").write_text("{}", encoding="utf-8")
    try:
        (calibration_root / "cal_link").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CALIBRATION_ROOT", calibration_root)
    monkeypatch.setattr(
        api,
        "_resolve_agent_clip_classifier_path_impl",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("escaped job_dir should stop before classifier resolution")
        ),
    )
    detections = [{"bbox": [0, 0, 1, 1], "label": "car"}]
    warnings = []

    result = api._agent_apply_ensemble_filter(
        detections,
        dataset_id="dataset",
        image_name="image.jpg",
        classifier_id="head.pkl",
        job_id="cal_link",
        warnings=warnings,
    )

    assert result is detections
    assert warnings == ["ensemble_filter_job_invalid"]


def test_ensemble_filter_rejects_symlinked_tmp_dir_before_write(
    tmp_path: Path, monkeypatch
):
    calibration_root = tmp_path / "calibration_jobs"
    job_dir = calibration_root / "cal_ok"
    job_dir.mkdir(parents=True)
    (job_dir / "ensemble_xgb.json").write_text("{}", encoding="utf-8")
    (job_dir / "ensemble_xgb.meta.json").write_text("{}", encoding="utf-8")
    upload_root = tmp_path / "uploads"
    outside_tmp = tmp_path / "outside_tmp_ensemble"
    upload_root.mkdir()
    outside_tmp.mkdir()
    try:
        (upload_root / "tmp_ensemble").symlink_to(outside_tmp, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CALIBRATION_ROOT", calibration_root)
    monkeypatch.setattr(api, "UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(
        api,
        "_resolve_agent_clip_classifier_path_impl",
        lambda *args, **kwargs: upload_root / "classifiers" / "head.pkl",
    )
    monkeypatch.setattr(
        api.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid temp dir should stop before scoring subprocesses")
        ),
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api._agent_apply_ensemble_filter(
            [{"bbox": [0, 0, 1, 1], "label": "car"}],
            dataset_id="dataset",
            image_name="image.jpg",
            classifier_id="head.pkl",
            job_id="cal_ok",
            warnings=[],
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "ensemble_filter_tmp_path_invalid"
    assert list(outside_tmp.iterdir()) == []
