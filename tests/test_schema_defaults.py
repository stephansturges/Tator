from __future__ import annotations

import ast
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def test_pydantic_models_do_not_use_mutable_literal_defaults() -> None:
    paths = [REPO_ROOT / "localinferenceapi.py"]
    paths.extend(
        REPO_ROOT / raw
        for raw in subprocess.check_output(
            ["git", "ls-files", "models/*.py", "api/*.py"],
            cwd=REPO_ROOT,
            text=True,
        ).splitlines()
    )
    mutable_literals = (ast.List, ast.Dict, ast.Set)
    issues: list[str] = []

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if "BaseModel" not in {_base_name(base) for base in node.bases}:
                continue
            for stmt in node.body:
                target: ast.expr | None = None
                value: ast.expr | None = None
                if isinstance(stmt, ast.AnnAssign):
                    target = stmt.target
                    value = stmt.value
                elif isinstance(stmt, ast.Assign) and stmt.targets:
                    target = stmt.targets[0]
                    value = stmt.value
                if isinstance(target, ast.Name) and isinstance(value, mutable_literals):
                    issues.append(f"{path.relative_to(REPO_ROOT)}:{stmt.lineno}:{node.name}.{target.id}")

    assert issues == []
