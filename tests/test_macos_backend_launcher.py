from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_macos_backend_launcher_refuses_busy_port_before_backend_import(tmp_path: Path) -> None:
    script = REPO_ROOT / "tools" / "run_macos_backend.sh"
    venv_python = REPO_ROOT / ".venv-macos" / "bin" / "python"
    if not venv_python.exists():
        pytest.skip(".venv-macos is not available")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        env = os.environ.copy()
        env.update(
            {
                "ENV_FILE": str(tmp_path / "missing.env"),
                "HOST": "127.0.0.1",
                "PORT": str(port),
                "TATOR_BACKEND_RESTART_ON_CRASH": "1",
            }
        )
        result = subprocess.run(
            ["bash", str(script)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    finally:
        listener.close()

    assert result.returncode == 98
    assert "already in use; not starting another backend" in result.stderr
    assert "Use the existing backend" in result.stderr
    assert "Backend exited with status" not in result.stderr
    assert "Loading CLIP model" not in result.stderr
    assert "uvicorn" not in result.stderr.lower()
