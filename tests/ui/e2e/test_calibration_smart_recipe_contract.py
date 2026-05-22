import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.ui, pytest.mark.ui_full]


# CASE_ID: CALIBRATION_SMART_RECIPE_RENDER
def test_calibration_smart_recipe_controls_exist():
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            "tools/check_playwright_control_coverage.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}".strip()
