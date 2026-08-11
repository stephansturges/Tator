from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_backend_runtime_paths_do_not_use_assert_statements() -> None:
    tree = ast.parse((REPO_ROOT / "localinferenceapi.py").read_text(encoding="utf-8"))
    assert_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
    ]

    assert assert_lines == []
