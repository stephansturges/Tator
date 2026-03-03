#!/usr/bin/env python3
"""Validate Playwright control manifest claims against test case ids."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


CASE_RE = re.compile(r"CASE_ID:\s*([A-Z0-9_]+)")
USABILITY_SURFACE_TYPES = {"modal", "tooltip", "banner", "toast", "panel"}


def _collect_case_ids(paths: Iterable[Path]) -> set[str]:
    found: set[str] = set()
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        found.update(CASE_RE.findall(text))
    return found


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _id_exists_in_ui(control_id: str, html_text: str, js_text: str) -> bool:
    if f'id="{control_id}"' in html_text:
        return True
    if f"getElementById(\"{control_id}\")" in js_text or f"getElementById('{control_id}')" in js_text:
        return True
    # Dynamic controls can be referenced by literal id checks in JS.
    if control_id in js_text:
        return True
    return False


def _validate_claimed_by(
    *,
    kind: str,
    obj_id: str,
    claimed_by: object,
    case_ids: set[str],
    errors: list[str],
) -> None:
    if not isinstance(claimed_by, list) or not claimed_by:
        errors.append(f"{kind} {obj_id} must have a non-empty claimed_by list")
        return
    for case_id in claimed_by:
        case_name = str(case_id or "").strip()
        if not case_name:
            errors.append(f"{kind} {obj_id} has blank case id claim")
            continue
        if case_name not in case_ids:
            errors.append(f"{kind} {obj_id} claims missing case id: {case_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="tests/ui/e2e/control_manifest.json",
        help="Path to the control manifest JSON",
    )
    parser.add_argument(
        "--tests-root",
        default="tests/ui/e2e",
        help="Directory containing Playwright tests",
    )
    parser.add_argument(
        "--html",
        default="ybat-master/ybat.html",
        help="Path to UI html",
    )
    parser.add_argument(
        "--js",
        default="ybat-master/ybat.js",
        help="Path to UI js",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    tests_root = Path(args.tests_root)
    html_path = Path(args.html)
    js_path = Path(args.js)

    manifest = _load_manifest(manifest_path)
    controls = manifest.get("controls") if isinstance(manifest, dict) else None
    usability_surfaces = (
        manifest.get("usability_surfaces", []) if isinstance(manifest, dict) else []
    )
    if not isinstance(controls, list):
        print("Manifest must contain a list field named 'controls'.", file=sys.stderr)
        return 2
    if not isinstance(usability_surfaces, list):
        print("Manifest field 'usability_surfaces' must be a list when present.", file=sys.stderr)
        return 2

    case_ids = _collect_case_ids(tests_root.rglob("test_*.py"))
    html_text = html_path.read_text(encoding="utf-8", errors="ignore")
    js_text = js_path.read_text(encoding="utf-8", errors="ignore")

    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_surface_ids: set[str] = set()

    for idx, raw in enumerate(controls):
        if not isinstance(raw, dict):
            errors.append(f"controls[{idx}] must be an object")
            continue
        control_id = str(raw.get("id") or "").strip()
        claimed_by = raw.get("claimed_by")

        if not control_id:
            errors.append(f"controls[{idx}] has empty id")
            continue
        if control_id in seen_ids:
            errors.append(f"duplicate control id: {control_id}")
        seen_ids.add(control_id)

        if not _id_exists_in_ui(control_id, html_text, js_text):
            errors.append(f"control id not found in UI code: {control_id}")
        _validate_claimed_by(
            kind="control",
            obj_id=control_id,
            claimed_by=claimed_by,
            case_ids=case_ids,
            errors=errors,
        )

    for idx, raw in enumerate(usability_surfaces):
        if not isinstance(raw, dict):
            errors.append(f"usability_surfaces[{idx}] must be an object")
            continue
        surface_id = str(raw.get("surface_id") or "").strip()
        surface_type = str(raw.get("type") or "").strip().lower()
        claimed_by = raw.get("claimed_by")

        if not surface_id:
            errors.append(f"usability_surfaces[{idx}] has empty surface_id")
            continue
        if surface_id in seen_surface_ids:
            errors.append(f"duplicate usability surface id: {surface_id}")
        seen_surface_ids.add(surface_id)

        if surface_type not in USABILITY_SURFACE_TYPES:
            errors.append(
                f"usability surface {surface_id} has invalid type '{surface_type}' "
                f"(allowed: {sorted(USABILITY_SURFACE_TYPES)})"
            )
        if not _id_exists_in_ui(surface_id, html_text, js_text):
            errors.append(f"usability surface id not found in UI code: {surface_id}")

        _validate_claimed_by(
            kind="usability surface",
            obj_id=surface_id,
            claimed_by=claimed_by,
            case_ids=case_ids,
            errors=errors,
        )

    if errors:
        print("Playwright control coverage check failed:", file=sys.stderr)
        for line in errors:
            print(f"- {line}", file=sys.stderr)
        return 1

    print(
        "Playwright control coverage check passed: "
        f"{len(controls)} controls, {len(usability_surfaces)} usability surfaces, "
        f"{len(case_ids)} case ids"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
