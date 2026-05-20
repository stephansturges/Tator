import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.ui, pytest.mark.ui_full]


# CASE_ID: CONTROL_MANIFEST_CONSISTENCY

def test_control_manifest_coverage_contract():
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
