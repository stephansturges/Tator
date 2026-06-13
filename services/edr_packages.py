"""Hermetic EDR package helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
import numpy as np
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, List, Optional, Sequence


EDR_PACKAGE_FORMAT_VERSION = 1
EDR_PACKAGE_SCHEMA_VERSION = 1
EDR_PACKAGE_ZIP_NAME = "package.edr.zip"
EDR_PACKAGE_META_NAME = "package.meta.json"
EDR_PACKAGE_MANIFEST_NAME = "edr_manifest.json"
EDR_PACKAGE_PAYLOAD_DIRNAME = "payload"
EDR_PACKAGE_STAGE_META_NAME = ".edr_package_stage.json"
EDR_PACKAGE_KIND_CANONICAL = "canonical_edr"
EDR_PACKAGE_KIND_SAVED = "saved_edr"
EDR_PACKAGE_RUNTIME_MODE = "package"
EDR_PACKAGE_FEATURE_SCHEMA_VERSION = 1
EDR_PACKAGE_REQUIRED_FILES = (EDR_PACKAGE_MANIFEST_NAME, "saved_recipe.json")
_EDR_PACKAGE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_regular_file_within_root(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.resolve(strict=True)
        resolved_root = root.resolve(strict=False)
    except Exception:
        return False
    try:
        resolved_path.relative_to(resolved_root)
    except Exception:
        return False
    return resolved_path.is_file()


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except Exception:
        return False
    return True


def _path_has_symlink_component(path: Path) -> bool:
    candidate = path if path.is_absolute() else path.absolute()
    checks = [candidate]
    checks.extend(candidate.parents)
    for component in checks:
        if component == component.parent:
            continue
        if component.is_symlink():
            return True
    return False


def _prepare_output_file(path: Path) -> None:
    if _path_has_symlink_component(path.parent):
        raise RuntimeError("edr_package_path_invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink_component(path.parent):
        raise RuntimeError("edr_package_path_invalid")
    parent_resolved = path.parent.resolve(strict=True)
    if path.is_symlink():
        path.unlink(missing_ok=True)
    elif path.exists() and path.is_dir():
        raise RuntimeError("edr_package_path_invalid")
    try:
        path.resolve(strict=False).relative_to(parent_resolved)
    except Exception as exc:
        raise RuntimeError("edr_package_path_invalid") from exc


def _prepare_atomic_output_file(path: Path) -> Path:
    _prepare_output_file(path)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    _prepare_output_file(tmp_path)
    return tmp_path


def _safe_child_dir_within_root(path: Path, root: Path) -> bool:
    if path.is_symlink():
        return False
    try:
        resolved_path = path.resolve(strict=True)
        resolved_root = root.resolve(strict=False)
    except Exception:
        return False
    try:
        resolved_path.relative_to(resolved_root)
    except Exception:
        return False
    return resolved_path.is_dir()


def _safe_existing_child_dir(root: Path, child_id: str, *, kind: str) -> Path:
    try:
        safe_id = _validate_edr_package_id(child_id)
    except ValueError as exc:
        raise RuntimeError(f"edr_package_invalid_{kind}_id") from exc
    if _path_has_symlink_component(root):
        raise RuntimeError("edr_package_path_invalid")
    try:
        root_resolved = root.resolve(strict=True)
    except Exception as exc:
        raise RuntimeError("edr_package_path_invalid") from exc
    raw_path = root / safe_id
    if raw_path.is_symlink():
        raise RuntimeError("edr_package_path_invalid")
    try:
        resolved = raw_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"edr_package_missing_detector_bundle:{kind}:{safe_id}") from exc
    except Exception as exc:
        raise RuntimeError("edr_package_path_invalid") from exc
    if not _path_within_root(resolved, root_resolved) or not resolved.is_dir():
        raise RuntimeError("edr_package_path_invalid")
    return resolved


def _copy_tree_filtered(src: Path, dest: Path) -> List[Path]:
    if not src.exists():
        raise FileNotFoundError(str(src))
    copied: List[Path] = []
    src_root = src.resolve(strict=False)
    dest_root = dest.resolve(strict=False)
    for item in sorted(src.rglob("*")):
        if not _safe_regular_file_within_root(item, src_root):
            continue
        rel = item.relative_to(src)
        target = dest / rel
        try:
            target_resolved = target.resolve(strict=False)
            target_resolved.relative_to(dest_root)
        except Exception:
            continue
        if _path_has_symlink_component(target.parent):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy2_if_different(item.resolve(strict=True), target)
        copied.append(target)
    return copied


def _slug(value: Any, *, fallback: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "").strip())
    text = text.strip("_")
    return text or fallback


def _validate_edr_package_id(package_id: Any) -> str:
    cleaned = str(package_id or "").strip()
    if not cleaned or cleaned in {".", ".."} or not _EDR_PACKAGE_ID_RE.fullmatch(cleaned):
        raise ValueError("edr_package_id_invalid")
    return cleaned


def _normalize_string_list(values: Any) -> List[str]:
    if not isinstance(values, (list, tuple)):
        return []
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            out.append(text)
    return out


def _scalar_or_none(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _normalize_float(value: Any) -> Optional[float]:
    value = _scalar_or_none(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_int(value: Any) -> Optional[int]:
    value = _scalar_or_none(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def canonical_edr_package_id(dataset_id: str, recipe_fingerprint: str) -> str:
    dataset_slug = _slug(dataset_id, fallback="dataset")
    fingerprint_slug = "".join(ch for ch in str(recipe_fingerprint or "").strip() if ch.isalnum())[:12] or "unknown"
    return f"canonical_edr_pkg_{dataset_slug}_{fingerprint_slug}"


def _edr_packages_storage_root(packages_root: Path, *, create: bool = False) -> Path:
    if _path_has_symlink_component(packages_root):
        raise ValueError("edr_package_path_invalid")
    if create:
        packages_root.mkdir(parents=True, exist_ok=True)
        if _path_has_symlink_component(packages_root):
            raise ValueError("edr_package_path_invalid")
    if packages_root.exists() and not packages_root.is_dir():
        raise ValueError("edr_package_path_invalid")
    if _path_has_symlink_component(packages_root):
        raise ValueError("edr_package_path_invalid")
    return packages_root.resolve(strict=False)


def edr_package_dir(packages_root: Path, package_id: str, *, create: bool = False) -> Path:
    safe_id = _validate_edr_package_id(package_id)
    root = _edr_packages_storage_root(packages_root, create=create)
    raw_path = packages_root / safe_id
    if raw_path.is_symlink():
        raise ValueError("edr_package_path_invalid")
    path = raw_path.resolve(strict=False)
    if not _path_within_root(path, root):
        raise ValueError("edr_package_path_invalid")
    if create:
        if raw_path.is_symlink():
            raise ValueError("edr_package_path_invalid")
        raw_path.mkdir(parents=True, exist_ok=True)
        if raw_path.is_symlink():
            raise ValueError("edr_package_path_invalid")
        path = raw_path.resolve(strict=True)
        if not _path_within_root(path, root):
            raise ValueError("edr_package_path_invalid")
    return path


def edr_package_zip_path(packages_root: Path, package_id: str) -> Path:
    return edr_package_dir(packages_root, package_id, create=False) / EDR_PACKAGE_ZIP_NAME


def edr_package_meta_path(packages_root: Path, package_id: str) -> Path:
    return edr_package_dir(packages_root, package_id, create=False) / EDR_PACKAGE_META_NAME


def _write_text_no_follow(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    tmp_path = _prepare_atomic_output_file(path)
    try:
        _write_text_no_follow(tmp_path, json.dumps(payload, indent=2))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists() or tmp_path.is_symlink():
            tmp_path.unlink(missing_ok=True)
    return path


def _write_text(path: Path, text: str) -> Path:
    tmp_path = _prepare_atomic_output_file(path)
    try:
        _write_text_no_follow(tmp_path, text)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists() or tmp_path.is_symlink():
            tmp_path.unlink(missing_ok=True)
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    if path.is_symlink():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _path_identity(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except RuntimeError:
        return path.absolute()


def _unlink_self_referential_symlink(path: Path) -> bool:
    if not path.is_symlink():
        return False
    try:
        target = Path(os.readlink(path))
    except OSError:
        return False
    if not target.is_absolute():
        target = path.parent / target
    if _path_identity(target) != _path_identity(path):
        return False
    path.unlink(missing_ok=True)
    return True


def _copy2_if_different(src: Path, dest: Path) -> None:
    src_resolved = src.resolve()
    _prepare_output_file(dest)
    if dest.exists() and src_resolved == _path_identity(dest):
        return
    tmp_path = _prepare_atomic_output_file(dest)
    try:
        shutil.copy2(src_resolved, tmp_path)
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists() or tmp_path.is_symlink():
            tmp_path.unlink(missing_ok=True)


def _remove_existing_child_path(path: Path, root: Path) -> None:
    if path.is_symlink():
        path.unlink(missing_ok=True)
        return
    if not path.exists():
        return
    root_resolved = root.resolve(strict=False)
    try:
        resolved = path.resolve(strict=True)
    except Exception as exc:
        raise RuntimeError("edr_package_path_invalid") from exc
    if not _path_within_root(resolved, root_resolved):
        raise RuntimeError("edr_package_path_invalid")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        path.unlink(missing_ok=True)


def _copy_file(src: Path, dest: Path, *, assets: List[Dict[str, Any]], kind: str) -> None:
    _copy2_if_different(src, dest)
    assets.append(
        {
            "kind": kind,
            "path": str(dest.as_posix()),
            "size": int(dest.stat().st_size),
            "sha256": _sha256_path(dest),
        }
    )


def _copy_tree(src: Path, dest: Path, *, assets: List[Dict[str, Any]], kind: str) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))
    src_root = src.resolve(strict=False)
    for item in sorted(src.rglob("*")):
        if not _safe_regular_file_within_root(item, src_root):
            continue
        rel = item.relative_to(src)
        target = dest / rel
        _copy_file(item, target, assets=assets, kind=kind)


def _sanitize_registry_entry_for_package(entry: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(entry or {})
    for key in (
        "recipe_root",
        "canonical_recipe_json",
        "canonical_recipe_md",
        "canonical_edr_json",
        "canonical_edr_md",
        "legacy_canonical_recipe_json",
        "legacy_canonical_recipe_md",
        "report_bundle_json",
        "discovery_run_root",
        "fingerprint_json",
        "canonical_deployment_job_dir",
    ):
        sanitized.pop(key, None)
    return sanitized


def _build_runtime_contract(saved_recipe: Dict[str, Any]) -> Dict[str, Any]:
    config = saved_recipe.get("config") if isinstance(saved_recipe.get("config"), dict) else {}
    runtime_config = dict(config)
    runtime_config["prepass_caption"] = False
    runtime_config["sam3_text_synonym_budget"] = 0
    runtime_config["resolved_classifier_id"] = (
        str(config.get("resolved_classifier_id") or config.get("classifier_id") or "").strip() or None
    )
    if runtime_config.get("resolved_classifier_id"):
        runtime_config["classifier_id"] = runtime_config["resolved_classifier_id"]
    runtime_config["edr_runtime_mode"] = EDR_PACKAGE_RUNTIME_MODE
    runtime_config["recipe_source_dataset_id"] = str(config.get("dataset_id") or "").strip() or None
    return {
        "schema_version": EDR_PACKAGE_SCHEMA_VERSION,
        "runtime_mode": EDR_PACKAGE_RUNTIME_MODE,
        "fail_mode": "closed",
        "cross_dataset_policy": "exact_labelmap_match",
        "config": runtime_config,
    }


def _build_feature_contract(bundle_meta: Dict[str, Any], runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    labelmap = _normalize_string_list(bundle_meta.get("labelmap") or runtime_config.get("labelmap"))
    classifier_classes = _normalize_string_list(bundle_meta.get("classifier_classes"))
    feature_names = _normalize_string_list(bundle_meta.get("feature_names"))
    return {
        "schema_version": EDR_PACKAGE_FEATURE_SCHEMA_VERSION,
        "feature_names": feature_names,
        "feature_schema_hash": str(_scalar_or_none(bundle_meta.get("feature_schema_hash")) or ""),
        "feature_schema_version": int(
            _normalize_int(bundle_meta.get("feature_schema_version"))
            or EDR_PACKAGE_FEATURE_SCHEMA_VERSION
        ),
        "support_iou": _normalize_float(bundle_meta.get("support_iou"))
        or _normalize_float(runtime_config.get("support_iou"))
        or 0.5,
        "context_radius": _normalize_float(bundle_meta.get("context_radius")) or 0.075,
        "label_iou": _normalize_float(bundle_meta.get("label_iou"))
        or _normalize_float(runtime_config.get("label_iou"))
        or 0.5,
        "eval_iou": _normalize_float(bundle_meta.get("eval_iou"))
        or _normalize_float(runtime_config.get("eval_iou"))
        or 0.5,
        "min_crop_size": _normalize_int(bundle_meta.get("min_crop_size")) or 4,
        "embed_proj_dim": _normalize_int(bundle_meta.get("embed_proj_dim")) or 0,
        "embed_proj_seed": _normalize_int(bundle_meta.get("embed_proj_seed")) or 42,
        "image_embed_proj_dim": _normalize_int(bundle_meta.get("image_embed_proj_dim")) or 0,
        "image_embed_proj_seed": _normalize_int(bundle_meta.get("image_embed_proj_seed")) or 4242,
        "embed_l2_normalize": bool(
            True if bundle_meta.get("embed_l2_normalize") is None else bundle_meta.get("embed_l2_normalize")
        ),
        "classifier_classes": classifier_classes,
        "labelmap": labelmap,
        "labelmap_hash": _sha256_bytes(("\n".join(labelmap)).encode("utf-8")) if labelmap else "",
        "context_variant_id": str(_scalar_or_none(bundle_meta.get("context_variant_id")) or ""),
        "variant_config_json": str(_scalar_or_none(bundle_meta.get("variant_config_json")) or "{}"),
    }


def _feature_contract_incomplete(feature_contract: Dict[str, Any], runtime_config: Dict[str, Any]) -> bool:
    if not isinstance(feature_contract, dict):
        return True
    feature_names = _normalize_string_list(feature_contract.get("feature_names"))
    labelmap = _normalize_string_list(feature_contract.get("labelmap"))
    schema_hash = str(_scalar_or_none(feature_contract.get("feature_schema_hash")) or "").strip()
    classifier_classes = _normalize_string_list(feature_contract.get("classifier_classes"))
    embed_proj_dim = _normalize_int(feature_contract.get("embed_proj_dim")) or 0
    classifier_key = str(
        runtime_config.get("resolved_classifier_id") or runtime_config.get("classifier_id") or ""
    ).strip()
    if not feature_names or not labelmap or not schema_hash:
        return True
    if classifier_key and (not classifier_classes or embed_proj_dim <= 0):
        return True
    return False


def _feature_contract_from_npz(npz_path: Path, runtime_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            def _npz_scalar(name: str) -> Any:
                if name not in data:
                    return None
                value = data[name]
                if hasattr(value, "item"):
                    try:
                        return value.item()
                    except Exception:
                        pass
                return value

            def _npz_strings(name: str) -> List[str]:
                if name not in data:
                    return []
                value = data[name]
                try:
                    seq = value.tolist()
                except Exception:
                    seq = list(value)
                return _normalize_string_list(seq)

            labelmap = _npz_strings("labelmap")
            classifier_classes = _npz_strings("classifier_classes")
            feature_names = _npz_strings("feature_names")
            return {
                "schema_version": EDR_PACKAGE_FEATURE_SCHEMA_VERSION,
                "feature_names": feature_names,
                "feature_schema_hash": str(_npz_scalar("feature_schema_hash") or ""),
                "feature_schema_version": int(
                    _normalize_int(_npz_scalar("feature_schema_version"))
                    or EDR_PACKAGE_FEATURE_SCHEMA_VERSION
                ),
                "support_iou": _normalize_float(_npz_scalar("support_iou"))
                or _normalize_float(runtime_config.get("support_iou"))
                or 0.5,
                "context_radius": _normalize_float(_npz_scalar("context_radius")) or 0.075,
                "label_iou": _normalize_float(_npz_scalar("label_iou"))
                or _normalize_float(runtime_config.get("label_iou"))
                or 0.5,
                "eval_iou": _normalize_float(_npz_scalar("eval_iou"))
                or _normalize_float(runtime_config.get("eval_iou"))
                or 0.5,
                "min_crop_size": _normalize_int(_npz_scalar("min_crop_size")) or 4,
                "embed_proj_dim": _normalize_int(_npz_scalar("embed_proj_dim")) or 0,
                "embed_proj_seed": _normalize_int(_npz_scalar("embed_proj_seed")) or 42,
                "image_embed_proj_dim": _normalize_int(_npz_scalar("image_embed_proj_dim")) or 0,
                "image_embed_proj_seed": _normalize_int(_npz_scalar("image_embed_proj_seed")) or 4242,
                "embed_l2_normalize": bool(
                    True
                    if _npz_scalar("embed_l2_normalize") is None
                    else _npz_scalar("embed_l2_normalize")
                ),
                "classifier_classes": classifier_classes,
                "labelmap": labelmap,
                "labelmap_hash": _sha256_bytes(("\n".join(labelmap)).encode("utf-8")) if labelmap else "",
                "context_variant_id": str(_npz_scalar("context_variant_id") or ""),
                "variant_config_json": str(_npz_scalar("variant_config_json") or "{}"),
            }
    except Exception:
        return None


def _candidate_source_feature_paths(payload_root: Path, runtime_config: Dict[str, Any]) -> List[Path]:
    candidates: List[Path] = []
    lane = str(runtime_config.get("lane_selection") or runtime_config.get("canonical_winner_lane") or "").strip()
    canonical_recipe_json = str(
        runtime_config.get("canonical_edr_json")
        or runtime_config.get("canonical_recipe_json")
        or ""
    ).strip()
    if canonical_recipe_json:
        discovery_root = Path(canonical_recipe_json).expanduser().resolve().parent
        if lane:
            candidates.append(discovery_root / "lanes" / lane / "features.npz")
    deployment_meta = _load_json(payload_root / "models" / "calibration_bundle" / "canonical_deployment.json")
    source_dir = str(deployment_meta.get("source_dir") or "").strip()
    winner_lane = str(deployment_meta.get("winner_lane") or lane or "").strip()
    if source_dir:
        try:
            source_path = Path(source_dir).expanduser().resolve()
            discovery_root = source_path.parents[2]
            if winner_lane:
                candidates.append(discovery_root / "lanes" / winner_lane / "features.npz")
        except Exception:
            pass
    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _repair_feature_contract_if_needed(
    *,
    payload_root: Path,
    manifest: Dict[str, Any],
    runtime_config: Dict[str, Any],
    feature_contract: Dict[str, Any],
    allow_external_source_features: bool = False,
) -> Dict[str, Any]:
    if not _feature_contract_incomplete(feature_contract, runtime_config):
        return feature_contract
    repaired = dict(feature_contract or {})
    if allow_external_source_features:
        for candidate in _candidate_source_feature_paths(payload_root, runtime_config):
            fallback = _feature_contract_from_npz(candidate, runtime_config)
            if not fallback:
                continue
            repaired = fallback
            break
    if not _feature_contract_incomplete(repaired, runtime_config):
        manifest["feature_contract"] = repaired
        _write_json(payload_root / "feature_contract.json", repaired)
        _write_json(payload_root / EDR_PACKAGE_MANIFEST_NAME, manifest)
        labelmap = _normalize_string_list(repaired.get("labelmap"))
        if labelmap:
            try:
                _write_text(payload_root / "labelmap.txt", "\n".join(labelmap) + "\n")
            except Exception:
                pass
    return repaired


def _build_package_summary(
    *,
    package_id: str,
    dataset_id: str,
    manifest: Dict[str, Any],
    saved_recipe: Dict[str, Any],
    package_root: Path,
    package_zip: Path,
) -> Dict[str, Any]:
    config = saved_recipe.get("config") if isinstance(saved_recipe.get("config"), dict) else {}
    runtime_contract = manifest.get("runtime_contract") if isinstance(manifest.get("runtime_contract"), dict) else {}
    return {
        "id": package_id,
        "name": saved_recipe.get("name") or package_id,
        "description": saved_recipe.get("description") or "",
        "dataset_id": dataset_id,
        "recipe_kind": str(config.get("recipe_kind") or EDR_PACKAGE_KIND_CANONICAL),
        "lane_selection": config.get("lane_selection"),
        "recipe_fingerprint": manifest.get("recipe_fingerprint"),
        "origin_kind": manifest.get("origin_kind"),
        "package_format_version": EDR_PACKAGE_FORMAT_VERSION,
        "package_root": str(package_root.resolve()),
        "package_zip": str(package_zip.resolve()),
        "package_sha256": _sha256_path(package_zip),
        "created_at": saved_recipe.get("created_at"),
        "updated_at": saved_recipe.get("updated_at"),
        "expected_mean_f1": config.get("expected_mean_f1"),
        "saved_recipe": saved_recipe,
        "runtime_contract": runtime_contract,
        "manifest": manifest,
    }


def _zip_payload(payload_dir: Path, zip_path: Path) -> None:
    tmp_path = _prepare_atomic_output_file(zip_path)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(payload_dir.rglob("*")):
                if not _safe_regular_file_within_root(item, payload_dir):
                    continue
                zf.write(item.resolve(strict=True), arcname=str(item.relative_to(payload_dir)))
        os.replace(tmp_path, zip_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _validate_edr_package_zip(zip_path: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise RuntimeError("edr_package_zip_corrupt")
            names = set(zf.namelist())
            for required_name in EDR_PACKAGE_REQUIRED_FILES:
                if required_name not in names:
                    if required_name == EDR_PACKAGE_MANIFEST_NAME:
                        raise RuntimeError("edr_package_manifest_missing")
                    raise RuntimeError(f"edr_package_required_file_missing:{required_name}")
    except zipfile.BadZipFile as exc:
        raise RuntimeError("edr_package_invalid_zip") from exc


def _validate_edr_package_payload(payload_root: Path) -> None:
    for required_name in EDR_PACKAGE_REQUIRED_FILES:
        required_path = payload_root / required_name
        if _safe_regular_file_within_root(required_path, payload_root):
            continue
        if required_name == EDR_PACKAGE_MANIFEST_NAME:
            raise RuntimeError("edr_package_manifest_missing")
        raise RuntimeError(f"edr_package_required_file_missing:{required_name}")


def _extract_zip_safely(
    zip_path: Path,
    dest_dir: Path,
    *,
    max_total_uncompressed_bytes: Optional[int] = None,
) -> None:
    root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        total_uncompressed = 0
        for info in zf.infolist():
            member_name = info.filename or ""
            member_win = PureWindowsPath(member_name)
            target = (dest_dir / member_name).resolve()
            if (
                member_name.startswith("/")
                or member_name.startswith("\\")
                or member_win.is_absolute()
                or bool(member_win.drive)
                or (target != root and root not in target.parents)
            ):
                raise RuntimeError("edr_package_path_traversal")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise RuntimeError("edr_package_symlink_unsupported")
            if max_total_uncompressed_bytes and max_total_uncompressed_bytes > 0:
                total_uncompressed += max(int(info.file_size or 0), 0)
                if total_uncompressed > max_total_uncompressed_bytes:
                    raise RuntimeError("edr_package_uncompressed_too_large")
        zf.extractall(dest_dir)


def materialize_canonical_edr_package(
    *,
    packages_root: Path,
    saved_recipe: Dict[str, Any],
    registry_entry: Dict[str, Any],
    fingerprint_payload: Dict[str, Any],
    yolo_job_root: Path,
    rfdetr_job_root: Path,
    calibration_root: Path,
    classifiers_root: Path,
    load_labelmap_fn: Callable[[str], Sequence[str]],
    resolve_classifier_path_fn: Callable[[str], Optional[Path]],
    sam3_checkpoint_path: Optional[str],
) -> Dict[str, Any]:
    config = saved_recipe.get("config") if isinstance(saved_recipe.get("config"), dict) else {}
    dataset_id = str(config.get("dataset_id") or registry_entry.get("dataset_id") or "").strip()
    recipe_fingerprint = str(
        config.get("recipe_fingerprint")
        or config.get("recipe_registry_fingerprint")
        or registry_entry.get("fingerprint")
        or ""
    ).strip()
    if not dataset_id or not recipe_fingerprint:
        raise RuntimeError("edr_package_missing_identity")
    package_id = canonical_edr_package_id(dataset_id, recipe_fingerprint)
    package_root = edr_package_dir(packages_root, package_id, create=True)
    payload_root = package_root / EDR_PACKAGE_PAYLOAD_DIRNAME
    _remove_existing_child_path(payload_root, package_root)
    payload_root.mkdir(parents=True, exist_ok=True)

    assets: List[Dict[str, Any]] = []
    # Canonical artifacts and metadata.
    manifest_registry = _sanitize_registry_entry_for_package(dict(registry_entry or {}))
    manifest_fingerprint = dict(fingerprint_payload or {})
    saved_recipe_copy = json.loads(json.dumps(saved_recipe))
    runtime_contract = _build_runtime_contract(saved_recipe_copy)
    saved_recipe_copy["config"] = dict(runtime_contract.get("config") or {})
    saved_recipe_copy["config"]["edr_package_id"] = package_id
    saved_recipe_copy["config"]["edr_runtime_mode"] = EDR_PACKAGE_RUNTIME_MODE
    saved_recipe_copy["config"]["edr_package_zip"] = str((package_root / EDR_PACKAGE_ZIP_NAME).resolve())
    saved_recipe_copy["config"]["edr_package_root"] = str(package_root.resolve())

    saved_recipe_path = payload_root / "saved_recipe.json"
    _write_json(saved_recipe_path, saved_recipe_copy)
    assets.append({"kind": "saved_recipe", "path": "saved_recipe.json", "size": int(saved_recipe_path.stat().st_size), "sha256": _sha256_path(saved_recipe_path)})

    config_path = payload_root / "deploy_config.json"
    _write_json(config_path, saved_recipe_copy["config"])
    assets.append({"kind": "deploy_config", "path": "deploy_config.json", "size": int(config_path.stat().st_size), "sha256": _sha256_path(config_path)})

    registry_path = payload_root / "registry_entry.json"
    _write_json(registry_path, manifest_registry)
    assets.append({"kind": "registry_entry", "path": "registry_entry.json", "size": int(registry_path.stat().st_size), "sha256": _sha256_path(registry_path)})

    fingerprint_path = payload_root / "fingerprint.json"
    _write_json(fingerprint_path, manifest_fingerprint)
    assets.append({"kind": "fingerprint", "path": "fingerprint.json", "size": int(fingerprint_path.stat().st_size), "sha256": _sha256_path(fingerprint_path)})

    # Labelmap + glossary.
    labelmap_lines = list(config.get("labelmap") or [])
    if not labelmap_lines and dataset_id:
        labelmap_lines = [str(x).strip() for x in load_labelmap_fn(dataset_id) if str(x).strip()]
    labelmap_path = payload_root / "labelmap.txt"
    _write_text(labelmap_path, "\n".join(labelmap_lines) + ("\n" if labelmap_lines else ""))
    assets.append({"kind": "labelmap", "path": "labelmap.txt", "size": int(labelmap_path.stat().st_size), "sha256": _sha256_path(labelmap_path)})

    glossary_text = str(saved_recipe.get("glossary") or "").strip()
    glossary_path = payload_root / "glossary.json"
    _write_json(glossary_path, {"glossary": glossary_text})
    assets.append({"kind": "glossary", "path": "glossary.json", "size": int(glossary_path.stat().st_size), "sha256": _sha256_path(glossary_path)})

    # Canonical artifacts.
    for rel_name, config_key in (
        ("canonical/canonical_edr.json", "canonical_edr_json"),
        ("canonical/canonical_edr.md", "canonical_edr_md"),
        ("canonical/report_bundle.json", "canonical_report_bundle_json"),
    ):
        raw_path = str(config.get(config_key) or "").strip()
        if not raw_path:
            continue
        src = Path(raw_path).expanduser().resolve()
        if src.exists() and src.is_file():
            _copy_file(src, payload_root / rel_name, assets=assets, kind=config_key)

    # Calibration bundle.
    calibration_job_id = str(config.get("canonical_deployment_job_id") or config.get("ensemble_job_id") or "").strip()
    if not calibration_job_id:
        raise RuntimeError("edr_package_missing_calibration_bundle")
    calibration_src = calibration_root / calibration_job_id
    if not calibration_src.exists():
        raise RuntimeError(f"edr_package_missing_calibration_bundle:{calibration_src}")
    calibration_dest = payload_root / "models" / "calibration_bundle"
    _copy_tree(calibration_src, calibration_dest, assets=assets, kind="calibration_bundle")
    bundle_meta = _load_json(calibration_src / "ensemble_xgb.meta.json")
    feature_contract = _build_feature_contract(bundle_meta, saved_recipe_copy["config"])
    feature_path = payload_root / "feature_contract.json"
    _write_json(feature_path, feature_contract)
    assets.append({"kind": "feature_contract", "path": "feature_contract.json", "size": int(feature_path.stat().st_size), "sha256": _sha256_path(feature_path)})

    runtime_path = payload_root / "runtime_contract.json"
    _write_json(runtime_path, runtime_contract)
    assets.append({"kind": "runtime_contract", "path": "runtime_contract.json", "size": int(runtime_path.stat().st_size), "sha256": _sha256_path(runtime_path)})

    # Detector bundles.
    for key, root, kind in (
        ("yolo_id", yolo_job_root, "yolo_run"),
        ("rfdetr_id", rfdetr_job_root, "rfdetr_run"),
    ):
        run_id = str(config.get(key) or "").strip()
        if not run_id:
            continue
        src = _safe_existing_child_dir(root, run_id, kind=kind)
        _copy_tree(src, payload_root / "models" / kind, assets=assets, kind=kind)

    # Classifier.
    classifier_key = str(config.get("resolved_classifier_id") or config.get("classifier_id") or "").strip()
    if classifier_key:
        classifier_path = resolve_classifier_path_fn(classifier_key)
        if classifier_path is None or not classifier_path.exists():
            raise RuntimeError(f"edr_package_missing_classifier:{classifier_key}")
        _copy_file(classifier_path, payload_root / "models" / "classifier" / classifier_path.name, assets=assets, kind="classifier")
        meta_path = classifier_path.with_suffix(".meta.pkl")
        if meta_path.exists():
            _copy_file(meta_path, payload_root / "models" / "classifier" / meta_path.name, assets=assets, kind="classifier_meta")

    # Base SAM3 checkpoint for offline portability.
    if sam3_checkpoint_path:
        sam3_src = Path(str(sam3_checkpoint_path)).expanduser().resolve()
        if sam3_src.exists() and sam3_src.is_file():
            _copy_file(sam3_src, payload_root / "models" / "sam3" / sam3_src.name, assets=assets, kind="sam3_checkpoint")

    manifest = {
        "schema_version": EDR_PACKAGE_SCHEMA_VERSION,
        "package_format_version": EDR_PACKAGE_FORMAT_VERSION,
        "package_id": package_id,
        "package_kind": str(config.get("recipe_kind") or EDR_PACKAGE_KIND_CANONICAL),
        "dataset_id": dataset_id,
        "recipe_fingerprint": recipe_fingerprint,
        "origin_kind": str(registry_entry.get("origin_kind") or "discovery_backed"),
        "runtime_mode": EDR_PACKAGE_RUNTIME_MODE,
        "runtime_contract": runtime_contract,
        "feature_contract": feature_contract,
        "assets": assets,
        "registry_entry": manifest_registry,
        "fingerprint_payload": manifest_fingerprint,
        "saved_recipe_id": saved_recipe_copy.get("id"),
        "canonical_deployment_job_id": calibration_job_id,
        "canonical_deployment_source_stage": config.get("canonical_deployment_source_stage"),
        "canonical_deployment_source_seed": config.get("canonical_deployment_source_seed"),
        "expected_mean_f1": config.get("expected_mean_f1"),
        "labelmap_hash": feature_contract.get("labelmap_hash"),
    }
    feature_contract = _repair_feature_contract_if_needed(
        payload_root=payload_root,
        manifest=manifest,
        runtime_config=runtime_contract.get("config") if isinstance(runtime_contract.get("config"), dict) else {},
        feature_contract=feature_contract,
        allow_external_source_features=True,
    )
    manifest["feature_contract"] = feature_contract
    manifest["labelmap_hash"] = feature_contract.get("labelmap_hash")
    manifest_path = payload_root / EDR_PACKAGE_MANIFEST_NAME
    _write_json(manifest_path, manifest)

    zip_path = package_root / EDR_PACKAGE_ZIP_NAME
    _zip_payload(payload_root, zip_path)
    summary = _build_package_summary(
        package_id=package_id,
        dataset_id=dataset_id,
        manifest=manifest,
        saved_recipe=saved_recipe_copy,
        package_root=package_root,
        package_zip=zip_path,
    )
    _write_json(package_root / EDR_PACKAGE_META_NAME, summary)
    return summary


def _ensure_payload(packages_root: Path, package_id: str) -> Path:
    package_root = edr_package_dir(packages_root, package_id, create=False)
    payload_root = package_root / EDR_PACKAGE_PAYLOAD_DIRNAME
    if (
        _safe_child_dir_within_root(payload_root, package_root)
        and _safe_regular_file_within_root(payload_root / EDR_PACKAGE_MANIFEST_NAME, payload_root)
    ):
        return payload_root
    zip_path = package_root / EDR_PACKAGE_ZIP_NAME
    if not _safe_regular_file_within_root(zip_path, package_root):
        raise FileNotFoundError(str(zip_path))
    _remove_existing_child_path(payload_root, package_root)
    payload_root.mkdir(parents=True, exist_ok=True)
    _extract_zip_safely(zip_path, payload_root)
    return payload_root


def _find_single_child_dir(parent: Path) -> Optional[Path]:
    if not parent.exists():
        return None
    for item in sorted(parent.iterdir()):
        if _safe_child_dir_within_root(item, parent):
            return item
    return None


def _stage_meta_matches(dest: Path, *, package_id: str, package_sha256: str, kind: str) -> bool:
    meta = _load_json(dest / EDR_PACKAGE_STAGE_META_NAME)
    return (
        str(meta.get("package_id") or "").strip() == str(package_id).strip()
        and str(meta.get("package_sha256") or "").strip() == str(package_sha256).strip()
        and str(meta.get("kind") or "").strip() == str(kind).strip()
    )


def _write_stage_meta(dest: Path, *, package_id: str, package_sha256: str, kind: str) -> None:
    _write_json(
        dest / EDR_PACKAGE_STAGE_META_NAME,
        {
            "package_id": str(package_id).strip(),
            "package_sha256": str(package_sha256).strip(),
            "kind": str(kind).strip(),
        },
    )


def _prepare_runtime_stage_root(stage_root: Path) -> Path:
    if _path_has_symlink_component(stage_root):
        raise RuntimeError("edr_package_path_invalid")
    stage_root.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink_component(stage_root) or not stage_root.is_dir():
        raise RuntimeError("edr_package_path_invalid")
    return stage_root.resolve(strict=True)


def _stage_tree_if_needed(
    src: Path,
    dest: Path,
    *,
    package_id: str,
    package_sha256: str,
    kind: str,
) -> None:
    stage_root = _prepare_runtime_stage_root(dest.parent)
    dest = stage_root / dest.name
    if dest.is_symlink():
        dest.unlink(missing_ok=True)
    if dest.exists() and _stage_meta_matches(
        dest,
        package_id=package_id,
        package_sha256=package_sha256,
        kind=kind,
    ):
        return
    _remove_existing_child_path(dest, dest.parent)
    dest.mkdir(parents=True, exist_ok=True)
    _copy_tree_filtered(src, dest)
    _write_stage_meta(
        dest,
        package_id=package_id,
        package_sha256=package_sha256,
        kind=kind,
    )


def resolve_edr_package_runtime(
    *,
    packages_root: Path,
    package_id: str,
    upload_root: Path,
    yolo_job_root: Path,
    rfdetr_job_root: Path,
    calibration_root: Path,
    classifiers_root: Path,
) -> Dict[str, Any]:
    payload_root = _ensure_payload(packages_root, package_id)
    meta = _load_json(edr_package_meta_path(packages_root, package_id))
    manifest = _load_json(payload_root / EDR_PACKAGE_MANIFEST_NAME)
    runtime_contract = manifest.get("runtime_contract") if isinstance(manifest.get("runtime_contract"), dict) else {}
    runtime_config = dict(runtime_contract.get("config") or {})
    feature_contract = manifest.get("feature_contract") if isinstance(manifest.get("feature_contract"), dict) else {}
    feature_contract = _repair_feature_contract_if_needed(
        payload_root=payload_root,
        manifest=manifest,
        runtime_config=runtime_config,
        feature_contract=feature_contract,
    )
    package_sha256 = str(meta.get("package_sha256") or _sha256_path(edr_package_zip_path(packages_root, package_id))).strip()
    labelmap_path = payload_root / "labelmap.txt"
    glossary_path = payload_root / "glossary.json"

    staged_classifier_id = None
    classifier_dir = payload_root / "models" / "classifier"
    if classifier_dir.exists():
        classifiers_root = _prepare_runtime_stage_root(classifiers_root)
        classifier_dir_resolved = classifier_dir.resolve(strict=False)
        for item in sorted(classifier_dir.iterdir()):
            if not _safe_regular_file_within_root(item, classifier_dir_resolved):
                continue
            if item.name.endswith(".meta.pkl"):
                continue
            if item.suffix.lower() in {".pkl", ".joblib"}:
                staged_name = f"{package_id}__{item.name}"
                dest = classifiers_root / staged_name
                if not dest.exists() or _sha256_path(dest) != _sha256_path(item):
                    _copy2_if_different(item, dest)
                meta_src = item.with_suffix(".meta.pkl")
                meta_dest = dest.with_suffix(".meta.pkl")
                if _safe_regular_file_within_root(meta_src, classifier_dir_resolved):
                    if not meta_dest.exists() or _sha256_path(meta_dest) != _sha256_path(meta_src):
                        _copy2_if_different(meta_src, meta_dest)
                staged_classifier_id = staged_name
                break
    staged_yolo_id = None
    yolo_dir = _find_single_child_dir(payload_root / "models" / "yolo_run")
    if yolo_dir is not None:
        staged_yolo_id = f"{package_id}__yolo"
        _stage_tree_if_needed(
            yolo_dir,
            yolo_job_root / staged_yolo_id,
            package_id=package_id,
            package_sha256=package_sha256,
            kind="yolo_run",
        )
    staged_rfdetr_id = None
    rfdetr_dir = _find_single_child_dir(payload_root / "models" / "rfdetr_run")
    if rfdetr_dir is not None:
        staged_rfdetr_id = f"{package_id}__rfdetr"
        _stage_tree_if_needed(
            rfdetr_dir,
            rfdetr_job_root / staged_rfdetr_id,
            package_id=package_id,
            package_sha256=package_sha256,
            kind="rfdetr_run",
        )
    staged_job_id = None
    calib_dir = payload_root / "models" / "calibration_bundle"
    if calib_dir.exists():
        staged_job_id = f"{package_id}__bundle"
        _stage_tree_if_needed(
            calib_dir,
            calibration_root / staged_job_id,
            package_id=package_id,
            package_sha256=package_sha256,
            kind="calibration_bundle",
        )
    sam3_checkpoint_path = None
    sam3_dir = payload_root / "models" / "sam3"
    if sam3_dir.exists():
        sam3_dir_resolved = sam3_dir.resolve(strict=False)
        for item in sorted(sam3_dir.iterdir()):
            if _safe_regular_file_within_root(item, sam3_dir_resolved):
                sam3_checkpoint_path = str(item.resolve())
                break

    labelmap_lines: List[str] = []
    if _safe_regular_file_within_root(labelmap_path, payload_root):
        labelmap_lines = [line.strip() for line in labelmap_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labelmap_lines:
        labelmap_lines = _normalize_string_list(
            runtime_config.get("labelmap") or feature_contract.get("labelmap")
        )
        if labelmap_lines:
            try:
                _write_text(labelmap_path, "\n".join(labelmap_lines) + "\n")
            except Exception:
                pass
    glossary_text = ""
    if _safe_regular_file_within_root(glossary_path, payload_root):
        glossary_payload = _load_json(glossary_path)
        glossary_text = str(glossary_payload.get("glossary") or "").strip()

    return {
        "package_id": package_id,
        "manifest": manifest,
        "runtime_contract": runtime_contract,
        "runtime_config": runtime_config,
        "feature_contract": feature_contract,
        "labelmap": labelmap_lines,
        "glossary_text": glossary_text,
        "staged_classifier_id": staged_classifier_id,
        "staged_yolo_id": staged_yolo_id,
        "staged_rfdetr_id": staged_rfdetr_id,
        "staged_ensemble_job_id": staged_job_id,
        "sam3_checkpoint_path": sam3_checkpoint_path,
        "payload_root": str(payload_root.resolve()),
    }


def load_edr_package_payload(packages_root: Path, package_id: str) -> Dict[str, Any]:
    payload_root = _ensure_payload(packages_root, package_id)
    manifest = _load_json(payload_root / EDR_PACKAGE_MANIFEST_NAME)
    saved_recipe = _load_json(payload_root / "saved_recipe.json")
    registry_entry = _load_json(payload_root / "registry_entry.json")
    fingerprint_payload = _load_json(payload_root / "fingerprint.json")
    return {
        "payload_root": payload_root,
        "manifest": manifest,
        "saved_recipe": saved_recipe,
        "registry_entry": registry_entry,
        "fingerprint_payload": fingerprint_payload,
    }


def list_edr_packages(packages_root: Path) -> List[Dict[str, Any]]:
    packages: List[Dict[str, Any]] = []
    try:
        root = _edr_packages_storage_root(packages_root)
    except ValueError:
        return packages
    if not root.exists():
        return packages
    for entry in sorted(root.iterdir()):
        if entry.is_symlink():
            continue
        try:
            entry_resolved = entry.resolve(strict=True)
        except Exception:
            continue
        if not _path_within_root(entry_resolved, root) or not entry_resolved.is_dir():
            continue
        meta_path = entry_resolved / EDR_PACKAGE_META_NAME
        if not _safe_regular_file_within_root(meta_path, entry_resolved):
            continue
        meta = _load_json(meta_path)
        if meta:
            packages.append(meta)
    packages.sort(key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0.0), reverse=True)
    return packages


def get_edr_package(packages_root: Path, package_id: str) -> Dict[str, Any]:
    package_root = edr_package_dir(packages_root, package_id, create=False)
    meta_path = package_root / EDR_PACKAGE_META_NAME
    if not _safe_regular_file_within_root(meta_path, package_root):
        raise FileNotFoundError(package_id)
    meta = _load_json(meta_path)
    if meta:
        return meta
    raise FileNotFoundError(package_id)


def export_edr_package(packages_root: Path, package_id: str) -> Path:
    package_root = edr_package_dir(packages_root, package_id, create=False)
    zip_path = package_root / EDR_PACKAGE_ZIP_NAME
    if not _safe_regular_file_within_root(zip_path, package_root):
        raise FileNotFoundError(str(zip_path))
    _validate_edr_package_zip(zip_path)
    return zip_path.resolve(strict=True)


def import_edr_package_from_zip(
    *,
    zip_path: Path,
    packages_root: Path,
    max_extract_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    temp_dir = Path(tempfile.mkdtemp(prefix="edr_package_import_"))
    try:
        payload_root = temp_dir / "payload"
        payload_root.mkdir(parents=True, exist_ok=True)
        try:
            _extract_zip_safely(
                zip_path,
                payload_root,
                max_total_uncompressed_bytes=max_extract_bytes,
            )
        except zipfile.BadZipFile as exc:
            raise RuntimeError("edr_package_invalid_zip") from exc
        manifest = _load_json(payload_root / EDR_PACKAGE_MANIFEST_NAME)
        if not manifest:
            raise RuntimeError("edr_package_manifest_missing")
        package_id = str(manifest.get("package_id") or "").strip()
        if not package_id:
            raise RuntimeError("edr_package_id_missing")
        _validate_edr_package_id(package_id)
        _validate_edr_package_payload(payload_root)
        package_root = edr_package_dir(packages_root, package_id, create=True)
        final_payload = package_root / EDR_PACKAGE_PAYLOAD_DIRNAME
        _remove_existing_child_path(final_payload, package_root)
        shutil.copytree(payload_root, final_payload)
        zip_dest = package_root / EDR_PACKAGE_ZIP_NAME
        _copy2_if_different(zip_path, zip_dest)
        saved_recipe = _load_json(final_payload / "saved_recipe.json")
        summary = _build_package_summary(
            package_id=package_id,
            dataset_id=str(manifest.get("dataset_id") or ""),
            manifest=manifest,
            saved_recipe=saved_recipe,
            package_root=package_root,
            package_zip=zip_dest,
        )
        _write_json(package_root / EDR_PACKAGE_META_NAME, summary)
        return summary
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
