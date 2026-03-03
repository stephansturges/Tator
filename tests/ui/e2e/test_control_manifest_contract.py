import subprocess

import pytest


pytestmark = [pytest.mark.ui, pytest.mark.ui_full]


# CASE_ID: CONTROL_MANIFEST_CONSISTENCY

def test_control_manifest_coverage_contract():
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
