from pathlib import Path

import pytest


pytestmark = [pytest.mark.ui]


# CASE_ID: CALIBRATION_REPORT_BUNDLE_RENDER


def test_calibration_report_bundle_controls_exist():
    repo_root = Path(__file__).resolve().parents[3]
    html_text = (repo_root / "ybat-master" / "tator.html").read_text(encoding="utf-8")
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
