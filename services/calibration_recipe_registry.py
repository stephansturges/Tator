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
    if path.parent.is_symlink():
        raise ValueError("recipe_registry_json_parent_symlink")
    parent_resolved = path.parent.resolve(strict=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    for candidate in (tmp_path, path):
        if candidate.is_symlink():
            candidate.unlink(missing_ok=True)
        elif candidate.exists() and candidate.is_dir():
            raise ValueError("recipe_registry_json_target_is_directory")
        try:
            candidate.resolve(strict=False).relative_to(parent_resolved)
        except Exception as exc:
            raise ValueError("recipe_registry_json_path_not_allowed") from exc
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_child_name(value: str, detail: str) -> str:
    name = str(value or "").strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(detail)
    return name


def _write_text_within_parent(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ValueError("recipe_registry_text_parent_symlink")
    parent_resolved = path.parent.resolve(strict=True)
    if path.is_symlink():
        path.unlink(missing_ok=True)
    elif path.exists() and path.is_dir():
        raise ValueError("recipe_registry_text_target_is_directory")
    try:
        path.resolve(strict=False).relative_to(parent_resolved)
    except Exception as exc:
        raise ValueError("recipe_registry_text_path_not_allowed") from exc
    path.write_text(text, encoding="utf-8")
    return path


def _prepare_recipe_registry_root(cache_root: Path) -> Path:
    if cache_root.is_symlink():
        raise ValueError("recipe_registry_cache_root_symlink")
    cache_root.mkdir(parents=True, exist_ok=True)
    if cache_root.is_symlink():
        raise ValueError("recipe_registry_cache_root_symlink")
    cache_resolved = cache_root.resolve(strict=True)
    root = registry_root(cache_root)
    if root.is_symlink():
        raise ValueError("recipe_registry_root_symlink")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("recipe_registry_root_symlink")
    try:
        root_resolved = root.resolve(strict=True)
    except Exception as exc:
        raise ValueError("recipe_registry_root_not_allowed") from exc
    if not _path_within(root_resolved, cache_resolved):
        raise ValueError("recipe_registry_root_not_allowed")
    return root


def _prepare_discovery_runs_root(cache_root: Path) -> Path:
    if cache_root.is_symlink():
        raise ValueError("recipe_discovery_cache_root_symlink")
    cache_root.mkdir(parents=True, exist_ok=True)
    if cache_root.is_symlink():
        raise ValueError("recipe_discovery_cache_root_symlink")
    cache_resolved = cache_root.resolve(strict=True)
    root = discovery_runs_root(cache_root)
    if root.is_symlink():
        raise ValueError("recipe_discovery_root_symlink")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("recipe_discovery_root_symlink")
    try:
        root_resolved = root.resolve(strict=True)
    except Exception as exc:
        raise ValueError("recipe_discovery_root_not_allowed") from exc
    if not _path_within(root_resolved, cache_resolved):
        raise ValueError("recipe_discovery_root_not_allowed")
    return root


def _prepare_recipe_dir(root: Path, fingerprint: str) -> Path:
    safe_fingerprint = _safe_child_name(
        fingerprint,
        "recipe_registry_entry_name_not_allowed",
    )
    recipe_dir = root / safe_fingerprint
    if recipe_dir.is_symlink():
        recipe_dir.unlink(missing_ok=True)
    elif recipe_dir.exists() and not recipe_dir.is_dir():
        raise ValueError("recipe_registry_entry_path_not_directory")
    recipe_dir.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve(strict=True)
    try:
        recipe_dir.resolve(strict=True).relative_to(root_resolved)
    except Exception as exc:
        raise ValueError("recipe_registry_entry_path_not_allowed") from exc
    return recipe_dir


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.parent.is_symlink():
        raise ValueError("recipe_registry_lock_parent_symlink")
    parent_resolved = lock_path.parent.resolve(strict=True)
    if lock_path.is_symlink():
        lock_path.unlink(missing_ok=True)
    elif lock_path.exists() and lock_path.is_dir():
        raise ValueError("recipe_registry_lock_target_is_directory")
    try:
        lock_path.resolve(strict=False).relative_to(parent_resolved)
    except Exception as exc:
        raise ValueError("recipe_registry_lock_path_not_allowed") from exc
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def discovery_lock(cache_root: Path, fingerprint: str) -> Iterator[None]:
    root = _prepare_discovery_runs_root(cache_root)
    safe_fingerprint = _safe_child_name(
        fingerprint,
        "recipe_discovery_lock_name_not_allowed",
    )
    lock_path = root / f"{safe_fingerprint}.lock"
    with _exclusive_lock(lock_path):
        yield


def load_registry(cache_root: Path) -> Dict[str, Any]:
    if cache_root.is_symlink():
        return {"version": CALIBRATION_RECIPE_REGISTRY_VERSION, "entries": {}}
    root = registry_root(cache_root)
    if root.is_symlink():
        return {"version": CALIBRATION_RECIPE_REGISTRY_VERSION, "entries": {}}
    try:
        cache_resolved = cache_root.resolve(strict=False)
        root_resolved = root.resolve(strict=False)
    except Exception:
        return {"version": CALIBRATION_RECIPE_REGISTRY_VERSION, "entries": {}}
    if not _path_within(root_resolved, cache_resolved):
        return {"version": CALIBRATION_RECIPE_REGISTRY_VERSION, "entries": {}}
    index_path = root / "index.json"
    if index_path.is_symlink() or not index_path.exists():
        return {"version": CALIBRATION_RECIPE_REGISTRY_VERSION, "entries": {}}
    try:
        index_resolved = index_path.resolve(strict=True)
        if not _path_within(index_resolved, root_resolved):
            return {"version": CALIBRATION_RECIPE_REGISTRY_VERSION, "entries": {}}
        payload = json.loads(index_resolved.read_text(encoding="utf-8"))
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
    root = _prepare_recipe_registry_root(cache_root)
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
    root = _prepare_recipe_registry_root(cache_root)
    recipe_dir = _prepare_recipe_dir(root, str(fingerprint))

    canonical_dst = recipe_dir / "canonical_edr.json"
    _write_text_within_parent(canonical_dst, canonical_recipe_json.read_text(encoding="utf-8"))
    legacy_canonical_dst = recipe_dir / "canonical_prepass_recipe.json"
    _write_text_within_parent(
        legacy_canonical_dst, canonical_recipe_json.read_text(encoding="utf-8")
    )

    canonical_md_dst: Optional[Path] = None
    if canonical_recipe_md is not None and canonical_recipe_md.exists():
        canonical_md_dst = recipe_dir / "canonical_edr.md"
        md_text = canonical_recipe_md.read_text(encoding="utf-8")
        _write_text_within_parent(canonical_md_dst, md_text)
        _write_text_within_parent(
            recipe_dir / "canonical_prepass_recipe.md",
            md_text,
        )

    report_dst: Optional[Path] = None
    if report_bundle_json is not None and report_bundle_json.exists():
        report_dst = recipe_dir / "report_bundle.json"
        _write_text_within_parent(report_dst, report_bundle_json.read_text(encoding="utf-8"))

    fingerprint_payload_path = recipe_dir / "fingerprint.json"
    _write_text_within_parent(
        fingerprint_payload_path,
        json.dumps(fingerprint_payload, indent=2),
    )
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

    with _exclusive_lock(root / ".index.lock"):
        registry = load_registry(cache_root)
        entries = registry.setdefault("entries", {})
        entries[str(fingerprint)] = entry
        save_registry(cache_root, registry)
    return entry
