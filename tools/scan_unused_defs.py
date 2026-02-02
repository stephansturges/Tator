#!/usr/bin/env python3
"""Heuristic scan for potentially unused defs in the repo."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "Qwen-Agent",
    "Qwen3-VL",
    "rf-detr",
    "SAM3-UNet",
    "sam3",
    "sam3_local",
    "sahi",
    "uploads",
    "logs",
    "models",
    "lightning_logs",
    "tests",
    "tools",
}


def iter_py_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


ROUTE_DECORATOR_NAMES = {
    "api_route",
    "delete",
    "get",
    "head",
    "options",
    "patch",
    "post",
    "put",
    "trace",
    "websocket",
}
INDIRECT_DECORATOR_NAMES = {
    "_register_agent_tool",
}


class DefUseCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.defs: List[Tuple[str, int, str, bool]] = []
        self.uses: List[str] = []
        self._class_depth = 0
        self._func_depth = 0
        self._alias_map: Dict[str, str] = {}

    def _is_route_decorator(self, decorator: ast.expr) -> bool:
        target = decorator
        if isinstance(target, ast.Call):
            target = target.func
        if isinstance(target, ast.Attribute):
            return target.attr in ROUTE_DECORATOR_NAMES or target.attr in INDIRECT_DECORATOR_NAMES
        if isinstance(target, ast.Name):
            return target.id in ROUTE_DECORATOR_NAMES or target.id in INDIRECT_DECORATOR_NAMES
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._class_depth > 0 or self._func_depth > 0:
            self.generic_visit(node)
            return
        is_route = any(self._is_route_decorator(dec) for dec in node.decorator_list)
        self.defs.append((node.name, node.lineno, "function", is_route))
        self._func_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._func_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._class_depth > 0 or self._func_depth > 0:
            self.generic_visit(node)
            return
        is_route = any(self._is_route_decorator(dec) for dec in node.decorator_list)
        self.defs.append((node.name, node.lineno, "async_function", is_route))
        self._func_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._func_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defs.append((node.name, node.lineno, "class", False))
        self._class_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._class_depth -= 1

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            name = node.id
            self.uses.append(name)
            original = self._alias_map.get(name)
            if original:
                self.uses.append(original)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            attr = node.attr
            if attr:
                self.uses.append(attr)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.asname and alias.name:
                self._alias_map[alias.asname] = alias.name
        self.generic_visit(node)


def scan(root: Path) -> Dict[str, List[Tuple[str, int, str, bool]]]:
    defs_by_file: Dict[str, List[Tuple[str, int, str, bool]]] = {}
    for path in iter_py_files(root):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except Exception:
            continue
        visitor = DefUseCollector()
        visitor.visit(tree)
        if visitor.defs:
            defs_by_file[str(path)] = visitor.defs
    return defs_by_file


def collect_uses(root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path in iter_py_files(root):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except Exception:
            continue
        visitor = DefUseCollector()
        visitor.visit(tree)
        for name in visitor.uses:
            counts[name] = counts.get(name, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Heuristic scan for unused defs.")
    parser.add_argument("--root", default=".", help="Repo root to scan.")
    parser.add_argument(
        "--max-uses",
        type=int,
        default=1,
        help="Report defs with use-count <= max-uses.",
    )
    parser.add_argument(
        "--include-underscore",
        action="store_true",
        help="Include defs starting with underscore.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    defs_by_file = scan(root)
    uses = collect_uses(root)

    rows: List[Tuple[str, int, str, str, int]] = []
    for file_path, defs in defs_by_file.items():
        for name, lineno, kind, is_route in defs:
            if name.startswith("__") and name.endswith("__"):
                continue
            if "/api/" in file_path and name.startswith("build_") and name.endswith("_router"):
                continue
            if not args.include_underscore and name.startswith("_"):
                continue
            if is_route:
                continue
            count = uses.get(name, 0)
            if count <= args.max_uses:
                rows.append((file_path, lineno, name, kind, count))

    rows.sort(key=lambda r: (r[0], r[1]))
    for file_path, lineno, name, kind, count in rows:
        print(f"{file_path}:{lineno} {kind} {name} uses={count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
