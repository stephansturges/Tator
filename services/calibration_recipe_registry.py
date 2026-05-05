"""Helpers for promoted Ensemble Detection Recipe (EDR) fingerprinting and registry storage."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


CALIBRATION_RECIPE_REGISTRY_VERSION = 1
CALIBRATION_DISCOVERY_PIPELINE_VERSION = 1
CANONICAL_EDR_SCHEMA_VERSION = 1
CANONICAL_PREPASS_RECIPE_SCHEMA_VERSION = CANONICAL_EDR_SCHEMA_VERSION
CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED = "discovery_backed"
CANONICAL_EDR_ORIGIN_IMPORTED_PORTABLE = "imported_portable"


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def stable_hash(payload: Any) -> str:
    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()


def build_recipe_fingerprint_payload(
    *,
    dataset_id: str,
    labelmap_hash: str,
    glossary_hash: str,
    classifier_id: str,
    lane_selection: str,
    prepass_config: Dict[str, Any],
    selected_hash: str,
    selected_count: int,
    selection_seed: int,
    requested_max_images: int,
    support_iou: float,
    context_radius: float,
    label_iou: float,
    eval_iou: float,
    feature_version: int,
    recipe_defaults_version: int = 1,
) -> Dict[str, Any]:
    return {
        "registry_version": CALIBRATION_RECIPE_REGISTRY_VERSION,
        "discovery_pipeline_version": CALIBRATION_DISCOVERY_PIPELINE_VERSION,
        "canonical_recipe_schema_version": CANONICAL_EDR_SCHEMA_VERSION,
        "recipe_defaults_version": int(recipe_defaults_version),
        "dataset_id": str(dataset_id),
        "labelmap_hash": str(labelmap_hash),
        "glossary_hash": str(glossary_hash),
        "classifier_id": str(classifier_id),
        "lane_selection": str(lane_selection),
        "prepass_config": prepass_config,
        "selected_hash": str(selected_hash),
        "selected_count": int(selected_count),
        "selection_seed": int(selection_seed),
        "requested_max_images": int(requested_max_images),
        "support_iou": float(support_iou),
        "context_radius": float(context_radius),
        "label_iou": float(label_iou),
        "eval_iou": float(eval_iou),
        "feature_version": int(feature_version),
    }


def build_recipe_fingerprint(payload: Dict[str, Any]) -> str:
    return stable_hash(payload)


def registry_root(cache_root: Path) -> Path:
    return cache_root / "recipe_registry"


def discovery_runs_root(cache_root: Path) -> Path:
    return cache_root / "discovery_runs"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def discovery_lock(cache_root: Path, fingerprint: str) -> Iterator[None]:
    lock_path = discovery_runs_root(cache_root) / f"{str(fingerprint)}.lock"
    with _exclusive_lock(lock_path):
        yield


def load_registry(cache_root: Path) -> Dict[str, Any]:
    root = registry_root(cache_root)
    index_path = root / "index.json"
    if not index_path.exists():
        return {"version": CALIBRATION_RECIPE_REGISTRY_VERSION, "entries": {}}
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": CALIBRATION_RECIPE_REGISTRY_VERSION, "entries": {}}
    if not isinstance(payload, dict):
        return {"version": CALIBRATION_RECIPE_REGISTRY_VERSION, "entries": {}}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        payload["entries"] = {}
    payload.setdefault("version", CALIBRATION_RECIPE_REGISTRY_VERSION)
    return payload


def save_registry(cache_root: Path, payload: Dict[str, Any]) -> Path:
    root = registry_root(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    return _atomic_write_json(index_path, payload)


def find_matching_recipe(cache_root: Path, fingerprint: str) -> Optional[Dict[str, Any]]:
    registry = load_registry(cache_root)
    entries = registry.get("entries")
    if not isinstance(entries, dict):
        return None
    entry = entries.get(str(fingerprint))
    if not isinstance(entry, dict):
        return None
    if str(entry.get("status") or "promoted") != "promoted":
        return None
    return entry


def register_promoted_recipe(
    cache_root: Path,
    *,
    fingerprint: str,
    fingerprint_payload: Dict[str, Any],
    dataset_id: str,
    canonical_recipe_json: Path,
    canonical_recipe_md: Optional[Path],
    report_bundle_json: Optional[Path],
    discovery_run_root: Optional[Path],
    origin_kind: str = CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED,
    canonical_deployment: Optional[Dict[str, Any]] = None,
    edr_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = registry_root(cache_root)
    recipe_dir = root / str(fingerprint)
    recipe_dir.mkdir(parents=True, exist_ok=True)

    canonical_dst = recipe_dir / "canonical_edr.json"
    canonical_dst.write_text(canonical_recipe_json.read_text(encoding="utf-8"), encoding="utf-8")
    legacy_canonical_dst = recipe_dir / "canonical_prepass_recipe.json"
    legacy_canonical_dst.write_text(canonical_recipe_json.read_text(encoding="utf-8"), encoding="utf-8")

    canonical_md_dst: Optional[Path] = None
    if canonical_recipe_md is not None and canonical_recipe_md.exists():
        canonical_md_dst = recipe_dir / "canonical_edr.md"
        canonical_md_dst.write_text(canonical_recipe_md.read_text(encoding="utf-8"), encoding="utf-8")
        (recipe_dir / "canonical_prepass_recipe.md").write_text(
            canonical_recipe_md.read_text(encoding="utf-8"), encoding="utf-8"
        )

    report_dst: Optional[Path] = None
    if report_bundle_json is not None and report_bundle_json.exists():
        report_dst = recipe_dir / "report_bundle.json"
        report_dst.write_text(report_bundle_json.read_text(encoding="utf-8"), encoding="utf-8")

    fingerprint_payload_path = recipe_dir / "fingerprint.json"
    fingerprint_payload_path.write_text(json.dumps(fingerprint_payload, indent=2), encoding="utf-8")
    normalized_origin = str(origin_kind or CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED).strip().lower()
    if normalized_origin not in {
        CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED,
        CANONICAL_EDR_ORIGIN_IMPORTED_PORTABLE,
    }:
        normalized_origin = CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED
    canonical_deployment = (
        dict(canonical_deployment)
        if isinstance(canonical_deployment, dict)
        else {}
    )
    edr_package = dict(edr_package) if isinstance(edr_package, dict) else {}

    entry = {
        "fingerprint": str(fingerprint),
        "dataset_id": str(dataset_id),
        "origin_kind": normalized_origin,
        "labelmap_hash": str(fingerprint_payload.get("labelmap_hash") or ""),
        "glossary_hash": str(fingerprint_payload.get("glossary_hash") or ""),
        "classifier_id": str(fingerprint_payload.get("classifier_id") or ""),
        "lane_selection": str(fingerprint_payload.get("lane_selection") or "window"),
        "selected_hash": str(fingerprint_payload.get("selected_hash") or ""),
        "selected_count": int(fingerprint_payload.get("selected_count") or 0),
        "selection_seed": int(fingerprint_payload.get("selection_seed") or 0),
        "requested_max_images": int(fingerprint_payload.get("requested_max_images") or 0),
        "detector_signature": stable_hash(fingerprint_payload.get("prepass_config") or {}),
        "windowing_signature": stable_hash(
            {
                "window": (fingerprint_payload.get("prepass_config") or {}).get("window"),
                "nonwindow": (fingerprint_payload.get("prepass_config") or {}).get("nonwindow"),
            }
        ),
        "feature_schema_version": int(fingerprint_payload.get("feature_version") or 0),
        "calibration_request_defaults_version": int(fingerprint_payload.get("recipe_defaults_version") or 1),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recipe_root": str(recipe_dir.resolve()),
        "canonical_recipe_json": str(canonical_dst.resolve()),
        "canonical_recipe_md": str(canonical_md_dst.resolve()) if canonical_md_dst else None,
        "canonical_edr_json": str(canonical_dst.resolve()),
        "canonical_edr_md": str(canonical_md_dst.resolve()) if canonical_md_dst else None,
        "legacy_canonical_recipe_json": str(legacy_canonical_dst.resolve()),
        "legacy_canonical_recipe_md": str((recipe_dir / "canonical_prepass_recipe.md").resolve()) if canonical_md_dst else None,
        "report_bundle_json": str(report_dst.resolve()) if report_dst else None,
        "discovery_run_root": (
            str(discovery_run_root.resolve())
            if discovery_run_root is not None
            else None
        ),
        "fingerprint_json": str(fingerprint_payload_path.resolve()),
        "canonical_deployment_job_id": (
            str(canonical_deployment.get("job_id") or "").strip() or None
        ),
        "canonical_deployment_job_dir": (
            str(canonical_deployment.get("job_dir") or "").strip() or None
        ),
        "canonical_deployment_source_stage": (
            str(canonical_deployment.get("source_stage") or "").strip() or None
        ),
        "canonical_deployment_source_seed": canonical_deployment.get("source_seed"),
        "edr_package_id": str(edr_package.get("id") or "").strip() or None,
        "edr_package_root": str(edr_package.get("package_root") or "").strip() or None,
        "edr_package_zip": str(edr_package.get("package_zip") or "").strip() or None,
        "edr_package_sha256": str(edr_package.get("package_sha256") or "").strip() or None,
        "status": "promoted",
    }
    entry_path = recipe_dir / "registry_entry.json"
    entry["registry_entry_json"] = str(entry_path.resolve())
    _atomic_write_json(entry_path, entry)

    with _exclusive_lock(registry_root(cache_root) / ".index.lock"):
        registry = load_registry(cache_root)
        entries = registry.setdefault("entries", {})
        entries[str(fingerprint)] = entry
        save_registry(cache_root, registry)
    return entry
