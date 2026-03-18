import subprocess

import pytest


pytestmark = [pytest.mark.ui, pytest.mark.ui_full]


# CASE_ID: CALIBRATION_SMART_RECIPE_RENDER
def test_calibration_smart_recipe_controls_exist():
    proc = subprocess.run(
        [
            "python",
            "tools/check_playwright_control_coverage.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}".strip()
