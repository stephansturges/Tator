from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_macos_backend_launcher_restarts_crashes_but_not_operator_stop() -> None:
    script = (ROOT / "tools" / "run_macos_backend.sh").read_text()

    assert 'RESTART_ON_CRASH="${TATOR_BACKEND_RESTART_ON_CRASH:-1}"' in script
    assert 'RESTART_MAX="${TATOR_BACKEND_RESTART_MAX:-0}"' in script
    assert 'export TATOR_BACKEND_LAUNCHER="tools/run_macos_backend.sh"' in script
    assert 'export TATOR_BACKEND_LAUNCHER_RESTARTS_CRASHES="${RESTART_ON_CRASH}"' in script
    assert 'export TATOR_BACKEND_LAUNCHER_RESTART_MAX="${RESTART_MAX}"' in script
    assert 'if [[ "${status}" == "0" ]]' in script
    assert 'if [[ "${status}" == "130" || "${status}" == "143" ]]' in script
    assert 'if [[ "${RESTART_ON_CRASH}" == "1"' in script
    assert "restart limit ${RESTART_MAX} reached" in script
    assert "Backend exited with status ${status}; restarting" in script


def _write_fake_python(tmp_path: Path, body: str) -> Path:
    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    python.chmod(0o755)
    return venv


def test_macos_backend_launcher_restarts_crash_until_limit(tmp_path: Path) -> None:
    count_path = tmp_path / "count"
    quoted_count_path = shlex.quote(str(count_path))
    venv = _write_fake_python(
        tmp_path,
        f"""
count=0
if [[ -f {quoted_count_path} ]]; then
  count="$(cat {quoted_count_path})"
fi
count=$((count + 1))
echo "${{count}}" > {quoted_count_path}
exit 134
""",
    )

    env = {
        **os.environ,
        "ENV_FILE": str(tmp_path / "missing.env"),
        "VENV_DIR": str(venv),
        "TATOR_BACKEND_RESTART_ON_CRASH": "1",
        "TATOR_BACKEND_RESTART_MAX": "1",
        "TATOR_BACKEND_RESTART_DELAY": "0",
        "TATOR_BACKEND_RESTART_MAX_DELAY": "0",
    }

    result = subprocess.run(
        ["bash", str(ROOT / "tools" / "run_macos_backend.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 134
    assert count_path.read_text().strip() == "2"
    assert "restarting in 0.0s" in result.stderr
    assert "restart limit 1 reached" in result.stderr


def test_macos_backend_launcher_does_not_restart_operator_stop(tmp_path: Path) -> None:
    count_path = tmp_path / "count"
    quoted_count_path = shlex.quote(str(count_path))
    venv = _write_fake_python(
        tmp_path,
        f"""
echo 1 > {quoted_count_path}
exit 130
""",
    )

    env = {
        **os.environ,
        "ENV_FILE": str(tmp_path / "missing.env"),
        "VENV_DIR": str(venv),
        "TATOR_BACKEND_RESTART_ON_CRASH": "1",
        "TATOR_BACKEND_RESTART_MAX": "5",
        "TATOR_BACKEND_RESTART_DELAY": "0",
    }

    result = subprocess.run(
        ["bash", str(ROOT / "tools" / "run_macos_backend.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 130
    assert count_path.read_text().strip() == "1"
    assert "restarting" not in result.stderr


def test_macos_backend_launcher_advertises_crash_restart_policy_to_backend(tmp_path: Path) -> None:
    env_path = tmp_path / "env.txt"
    quoted_env_path = shlex.quote(str(env_path))
    venv = _write_fake_python(
        tmp_path,
        f"""
printf '%s\\n' "$TATOR_BACKEND_LAUNCHER" > {quoted_env_path}
printf '%s\\n' "$TATOR_BACKEND_LAUNCHER_RESTARTS_CRASHES" >> {quoted_env_path}
printf '%s\\n' "$TATOR_BACKEND_LAUNCHER_RESTART_MAX" >> {quoted_env_path}
printf '%s\\n' "$TATOR_BACKEND_LAUNCHER_RESTART_DELAY" >> {quoted_env_path}
printf '%s\\n' "$TATOR_BACKEND_LAUNCHER_RESTART_MAX_DELAY" >> {quoted_env_path}
exit 0
""",
    )

    env = {
        **os.environ,
        "ENV_FILE": str(tmp_path / "missing.env"),
        "VENV_DIR": str(venv),
        "TATOR_BACKEND_RESTART_ON_CRASH": "1",
        "TATOR_BACKEND_RESTART_MAX": "3",
        "TATOR_BACKEND_RESTART_DELAY": "2",
        "TATOR_BACKEND_RESTART_MAX_DELAY": "9",
    }

    result = subprocess.run(
        ["bash", str(ROOT / "tools" / "run_macos_backend.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert env_path.read_text().splitlines() == [
        "tools/run_macos_backend.sh",
        "1",
        "3",
        "2",
        "9",
    ]
