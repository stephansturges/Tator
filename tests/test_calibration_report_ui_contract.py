from pathlib import Path


def test_calibration_report_ui_controls_and_fetch_hook_exist():
    repo_root = Path(__file__).resolve().parents[1]
    html_text = (repo_root / "ybat-master" / "ybat.html").read_text(encoding="utf-8")
    js_text = (repo_root / "ybat-master" / "ybat.js").read_text(encoding="utf-8")

    for control_id in (
        "qwenCalibrationReportWrap",
        "qwenCalibrationReportSummary",
        "qwenCalibrationReportStatus",
        "qwenCalibrationReportOverview",
        "qwenCalibrationReportPerClass",
        "qwenCalibrationReportPerSource",
        "qwenCalibrationReportBoundary",
        "qwenCalibrationReportUncertainty",
        "qwenCalibrationReportDiagnostics",
    ):
        assert f'id="{control_id}"' in html_text
        assert control_id in js_text

    assert "/calibration/jobs/${encodeURIComponent(jobId)}/artifacts/report_bundle" in js_text
    assert "step_current" in js_text
    assert "substep_current" in js_text
