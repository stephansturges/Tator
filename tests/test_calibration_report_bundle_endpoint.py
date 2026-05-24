import json
from pathlib import Path

import pytest

import localinferenceapi as api
from services.calibration import (
    CALIBRATION_JOB_STATE_FILENAME,
    CalibrationJob,
    _prepare_calibration_storage_dir,
    _resolve_calibration_storage_root,
    _write_json_atomic,
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
