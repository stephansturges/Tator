from __future__ import annotations

import os
from pathlib import Path

from services.calibration import _repo_tool_subprocess_env


def test_repo_tool_subprocess_env_prepends_repo_root(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    env = _repo_tool_subprocess_env(root_dir, {"PYTHONPATH": "/tmp/one:/tmp/two"})

    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(root_dir.resolve()),
        "/tmp/one",
        "/tmp/two",
    ]


def test_repo_tool_subprocess_env_does_not_duplicate_repo_root(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    repo_root = str(root_dir.resolve())
    env = _repo_tool_subprocess_env(
        root_dir,
        {"PYTHONPATH": f"{repo_root}{os.pathsep}/tmp/one"},
    )

    assert env["PYTHONPATH"].split(os.pathsep).count(repo_root) == 1
