#!/usr/bin/env python3
"""Heuristic scan for likely-unused function defs."""

from __future__ import annotations

import ast
import pathlib
import re
import sys
from typing import Iterable


EXCLUDE_DIRS = {
    ".venv",
    "__pycache__",
    "node_modules",
    "Qwen-Agent",
    "Qwen3-VL",
    "rf-detr",
    "sam3",
    "sahi",
    "SAM3-UNet",
}


def _iter_py_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def _collect_defs(path: pathlib.Path) -> list[tuple[str, int]]:
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    defs: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append((node.name, node.lineno))
    return defs


def _count_occurrences(name: str, corpus: str) -> int:
    pattern = re.compile(rf"\b{name}\b")
    return len(pattern.findall(corpus))


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
    files = list(_iter_py_files(root))
    corpus = "\n".join(path.read_text() for path in files)
    candidates: list[tuple[str, pathlib.Path, int, int]] = []
    for path in files:
        for name, lineno in _collect_defs(path):
            if len(name) < 4:
                continue
            if name.startswith("_"):
                continue
            count = _count_occurrences(name, corpus)
            if count <= 1:
                candidates.append((name, path, lineno, count))

    candidates.sort(key=lambda item: (item[3], str(item[1]), item[2]))
    print("Likely-unused defs (count<=1):")
    try:
        for name, path, lineno, count in candidates:
            print(f"- {name} ({path}:{lineno}) count={count}")
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
