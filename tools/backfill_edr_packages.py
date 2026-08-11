#!/usr/bin/env python3
"""Backfill hermetic EDR packages for existing canonical recipes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.calibration_recipe_registry import load_registry
from services.canonical_edr_completion import (
    _build_minimal_calibration_request,
    _load_fingerprint_payload,
    _load_saved_canonical_recipe_payload,
    _sanitize_reconstructed_calibration_request,
    build_canonical_completion_context,
    persist_canonical_edr_completion,
    repair_persisted_canonical_completion,
)


def _build_context_from_registry(
    *,
    calibration_cache_root: Path,
    registry_entry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    dataset_id = str(registry_entry.get("dataset_id") or "").strip()
    fingerprint = str(registry_entry.get("fingerprint") or "").strip()
    canonical_recipe_json = Path(str(registry_entry.get("canonical_recipe_json") or "")).resolve()
    if not dataset_id or not fingerprint or not canonical_recipe_json.exists():
        return None
    fingerprint_payload = _load_fingerprint_payload(registry_entry)
    if not fingerprint_payload:
        return None
    saved_recipe = _load_saved_canonical_recipe_payload(
        recipes_root=calibration_cache_root.parent / "prepass_recipes",
        dataset_id=dataset_id,
        recipe_fingerprint=fingerprint,
        canonical_recipe_json=canonical_recipe_json,
    )
    saved_config = (
        dict(saved_recipe.get("config") or {})
        if isinstance(saved_recipe, dict) and isinstance(saved_recipe.get("config"), dict)
        else {}
    )
    calibration_request = (
        _sanitize_reconstructed_calibration_request(saved_config)
        if saved_config
        else _build_minimal_calibration_request(
            dataset_id=dataset_id,
            registry_entry=registry_entry,
            fingerprint_payload=fingerprint_payload,
        )
    )
    if not calibration_request:
        return None
    calibration_request["dataset_id"] = dataset_id
    calibration_request.setdefault(
        "lane_selection",
        str(registry_entry.get("lane_selection") or fingerprint_payload.get("lane_selection") or "window"),
    )
    report_bundle_json = str(registry_entry.get("report_bundle_json") or "").strip()
    report_bundle_path = Path(report_bundle_json).resolve() if report_bundle_json else None
    if report_bundle_path is not None and not report_bundle_path.exists():
        report_bundle_path = None
    return build_canonical_completion_context(
        dataset_id=dataset_id,
        recipe_fingerprint=fingerprint,
        recipe_fingerprint_payload=fingerprint_payload,
        calibration_request=calibration_request,
        resolved_classifier_id=(
            str(saved_config.get("resolved_classifier_id") or registry_entry.get("classifier_id") or "").strip()
            or None
        ),
        glossary_text=str(saved_recipe.get("glossary") or "") if isinstance(saved_recipe, dict) else "",
        labelmap=list(saved_config.get("labelmap") or []),
        report_bundle_json=report_bundle_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-cache-root",
        default="uploads/calibration_cache",
        help="Path to calibration cache root.",
    )
    parser.add_argument(
        "--fingerprint",
        action="append",
        default=[],
        help="Specific recipe fingerprint(s) to repair. Defaults to all registry entries.",
    )
    args = parser.parse_args()

    calibration_cache_root = Path(args.calibration_cache_root).expanduser().resolve()
    registry = load_registry(calibration_cache_root)
    entries = dict(registry.get("entries") or {})
    requested = {str(value).strip() for value in (args.fingerprint or []) if str(value).strip()}
    summary = {"updated": [], "skipped": [], "errors": []}

    for fingerprint, registry_entry in sorted(entries.items()):
        if requested and fingerprint not in requested:
            continue
        entry = dict(registry_entry or {})
        package_id = str(entry.get("edr_package_id") or "").strip()
        package_zip = Path(str(entry.get("edr_package_zip") or "")).expanduser() if package_id else None
        if package_id and package_zip and package_zip.exists():
            summary["skipped"].append({"fingerprint": fingerprint, "reason": "package_present"})
            continue

        run_root_text = str(entry.get("discovery_run_root") or "").strip()
        try:
            if run_root_text:
                run_root = Path(run_root_text).expanduser().resolve()
                if run_root.exists():
                    result = repair_persisted_canonical_completion(
                        calibration_cache_root=calibration_cache_root,
                        run_root=run_root,
                    )
                    summary["updated"].append(
                        {
                            "fingerprint": fingerprint,
                            "mode": "repair_run_root",
                            "edr_package_id": result.get("edr_package_id"),
                        }
                    )
                    continue

            context = _build_context_from_registry(
                calibration_cache_root=calibration_cache_root,
                registry_entry=entry,
            )
            if context is None:
                summary["errors"].append(
                    {"fingerprint": fingerprint, "error": "completion_context_unavailable"}
                )
                continue
            canonical_recipe_json = Path(str(entry.get("canonical_recipe_json") or "")).expanduser().resolve()
            canonical_recipe_md = Path(str(entry.get("canonical_recipe_md") or "")).expanduser().resolve()
            report_bundle_json = Path(str(entry.get("report_bundle_json") or "")).expanduser().resolve()
            result = persist_canonical_edr_completion(
                calibration_cache_root=calibration_cache_root,
                run_root=None,
                canonical_recipe_json=canonical_recipe_json,
                canonical_recipe_md=canonical_recipe_md if canonical_recipe_md.exists() else None,
                canonical_recipe_payload=None,
                completion_context=context,
                existing_registry_entry=entry,
                report_bundle_json=report_bundle_json if report_bundle_json.exists() else None,
                write_summary=False,
            )
            summary["updated"].append(
                {
                    "fingerprint": fingerprint,
                    "mode": "repair_registry_only",
                    "edr_package_id": result.get("edr_package_id"),
                }
            )
        except Exception as exc:  # pragma: no cover - CLI safety
            summary["errors"].append({"fingerprint": fingerprint, "error": str(exc)})

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
