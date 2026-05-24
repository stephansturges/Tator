"""Calibration job orchestration and serialization."""

from __future__ import annotations

import hashlib
import math
import copy
from dataclasses import dataclass, field
import json
import multiprocessing
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from services.canonical_edr_completion import (
    build_canonical_completion_context,
    canonical_completion_context_path,
    canonical_completion_summary_path,
    persist_canonical_edr_completion,
    write_canonical_completion_context,
)
from services.calibration_recipe_registry import (
    build_recipe_fingerprint,
    build_recipe_fingerprint_payload,
    discovery_lock,
    discovery_runs_root,
    find_matching_recipe,
)
from services.calibration_helpers import _calibration_resolve_image_path
from services.job_payloads import json_sanitize
from utils.io import _path_is_within_root_impl
from utils.pydantic_compat import model_copy_update, model_dump_compat

DEFAULT_EMBED_PROJ_DIM = 1024
DEFAULT_IMAGE_EMBED_PROJ_SEED = 4242
DEFAULT_CALIBRATION_LANE_SELECTION = "window"

DEFAULT_WINDOWED_SPLIT_HEAD_BY_SUPPORT = False
DEFAULT_NONWINDOWED_SPLIT_HEAD_BY_SUPPORT = True
DEFAULT_WINDOWED_SAM3_TEXT_QUALITY_ALPHA = 0.8
DEFAULT_NONWINDOWED_SAM3_TEXT_QUALITY_ALPHA = 0.5
DEFAULT_WINDOWED_TRAIN_SAM3_SIMILARITY_QUALITY = True
DEFAULT_NONWINDOWED_TRAIN_SAM3_SIMILARITY_QUALITY = False
DEFAULT_WINDOWED_SAM3_SIMILARITY_QUALITY_ALPHA = 0.5
DEFAULT_NONWINDOWED_SAM3_SIMILARITY_QUALITY_ALPHA = 0.5

DEFAULT_ENSEMBLE_POLICY_REFINED: Dict[str, Any] = {
    "sam_bias_scope": "sam_only",
    "logit_bias_by_source_class": {
        "sam3_text": {
            "__default__": -1.4,
        },
        "sam3_similarity": {
            "__default__": -1.2,
        },
    },
    "sam_only_min_prob_default": 0.15,
    "consensus_iou_default": 0.7,
    "consensus_iou_by_source_class": {
        "sam3_text": {"__default__": 0.7},
        "sam3_similarity": {"__default__": 0.7},
    },
    "consensus_class_aware": True,
}

DEFAULT_ENSEMBLE_POLICY_NONWINDOW: Dict[str, Any] = {
    **copy.deepcopy(DEFAULT_ENSEMBLE_POLICY_REFINED),
}

DEFAULT_ENSEMBLE_POLICY_WINDOW: Dict[str, Any] = {
    **copy.deepcopy(DEFAULT_ENSEMBLE_POLICY_REFINED),
}

DEFAULT_ENSEMBLE_POLICY_NONWINDOW_LEGACY: Dict[str, Any] = {
    "logit_bias_by_source_class": {
        "sam3_text": {
            "__default__": -0.8,
            "bike": -1.0,
            "boat": -0.6,
            "building": -1.2,
            "container": -1.2,
            "digger": -1.2,
            "light_vehicle": -0.6,
            "person": -1.2,
            "utility_pole": -1.2,
        },
        "sam3_similarity": {
            "__default__": -0.4,
            "bike": -0.8,
            "building": -0.8,
            "bus": -0.8,
            "container": -0.2,
            "utility_pole": -0.2,
        },
    },
    "sam_only_min_prob_default": 0.0,
    "consensus_iou_default": 0.0,
    "consensus_iou_by_class": {
        "building": 0.7,
        "container": 0.7,
        "person": 0.7,
    },
    "consensus_class_aware": True,
}

CALIBRATION_RECIPE_DEFAULTS_VERSION = 1
CANONICAL_EDR_JSON_NAME = "canonical_edr.json"
CANONICAL_EDR_MD_NAME = "canonical_edr.md"
LEGACY_CANONICAL_RECIPE_JSON_NAME = "canonical_prepass_recipe.json"
LEGACY_CANONICAL_RECIPE_MD_NAME = "canonical_prepass_recipe.md"
_CALIBRATION_METRIC_SUMMARY_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1")
CALIBRATION_JOB_STATE_FILENAME = "job_state.json"
CALIBRATION_JOB_STATE_VERSION = 1
CALIBRATION_JOB_INTERRUPTED_ERROR = "calibration_job_interrupted_after_backend_restart"


def _resolve_calibration_storage_root(
    root_path: Path,
    *,
    create: bool = False,
    error_prefix: str = "calibration_root",
) -> Path:
    raw_root = Path(root_path)
    if _path_has_symlink_component(raw_root):
        raise ValueError(f"{error_prefix}_symlink")
    if create:
        raw_root.mkdir(parents=True, exist_ok=True)
        if _path_has_symlink_component(raw_root):
            raise ValueError(f"{error_prefix}_symlink")
        return raw_root.resolve(strict=True)
    return raw_root.resolve(strict=False)


def _prepare_calibration_storage_dir(
    path: Path,
    *,
    root: Optional[Path] = None,
    error_prefix: str = "calibration_path",
) -> Path:
    raw_path = Path(path)
    if _path_has_symlink_component(raw_path):
        raise ValueError(f"{error_prefix}_symlink")
    raw_path.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink_component(raw_path):
        raise ValueError(f"{error_prefix}_symlink")
    resolved = raw_path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{error_prefix}_not_directory")
    if root is not None:
        root_resolved = Path(root).resolve(strict=True)
        if not _path_is_within_root_impl(resolved, root_resolved):
            raise ValueError(f"{error_prefix}_escape")
    return resolved


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


def _repo_tool_subprocess_env(root_dir: Path, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Ensure repo-root tool scripts can import local packages when spawned as subprocesses."""
    env = dict(base_env or os.environ)
    repo_root = str(root_dir.resolve())
    existing_py = env.get("PYTHONPATH", "")
    paths = [part for part in existing_py.split(os.pathsep) if part]
    if repo_root not in paths:
        env["PYTHONPATH"] = (
            os.pathsep.join([repo_root, *paths]) if paths else repo_root
        )
    return env


def _write_json_atomic(path: Path, payload: Any) -> None:
    if _path_has_symlink_component(path.parent):
        raise ValueError("calibration_json_parent_symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink_component(path.parent):
        raise ValueError("calibration_json_parent_symlink")
    parent_resolved = path.parent.resolve(strict=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    for candidate in (tmp_path, path):
        if candidate.is_symlink():
            candidate.unlink(missing_ok=True)
        elif candidate.exists() and candidate.is_dir():
            raise ValueError("calibration_json_target_is_directory")
        try:
            candidate.resolve(strict=False).relative_to(parent_resolved)
        except Exception as exc:
            raise ValueError("calibration_json_path_not_allowed") from exc
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists() or tmp_path.is_symlink():
            tmp_path.unlink(missing_ok=True)


def _write_text_atomic(path: Path, text: str) -> None:
    if _path_has_symlink_component(path.parent):
        raise ValueError("calibration_text_parent_symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink_component(path.parent):
        raise ValueError("calibration_text_parent_symlink")
    parent_resolved = path.parent.resolve(strict=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    for candidate in (tmp_path, path):
        if candidate.is_symlink():
            candidate.unlink(missing_ok=True)
        elif candidate.exists() and candidate.is_dir():
            raise ValueError("calibration_text_target_is_directory")
        try:
            candidate.resolve(strict=False).relative_to(parent_resolved)
        except Exception as exc:
            raise ValueError("calibration_text_path_not_allowed") from exc
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists() or tmp_path.is_symlink():
            tmp_path.unlink(missing_ok=True)


def _calibration_job_state_path(calibration_root: Path, job_id: str) -> Path:
    root = _resolve_calibration_storage_root(calibration_root)
    raw_job_dir = root / str(job_id)
    if raw_job_dir.is_symlink():
        raise ValueError("calibration_job_path_not_allowed")
    job_dir = raw_job_dir.resolve(strict=False)
    if not _path_is_within_root_impl(job_dir, root):
        raise ValueError("calibration_job_path_not_allowed")
    return job_dir / CALIBRATION_JOB_STATE_FILENAME


def _persist_calibration_job_state(job: Any, calibration_root: Path) -> Dict[str, Any]:
    payload = dict(_serialize_calibration_job(job))
    payload["state_schema_version"] = CALIBRATION_JOB_STATE_VERSION
    root = _resolve_calibration_storage_root(calibration_root, create=True)
    _write_json_atomic(_calibration_job_state_path(root, str(job.job_id)), payload)
    return payload


def _mark_persisted_job_interrupted(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    normalized["status"] = "failed"
    normalized["phase"] = "failed"
    normalized["message"] = "Interrupted by backend restart"
    normalized["error"] = str(normalized.get("error") or CALIBRATION_JOB_INTERRUPTED_ERROR)
    normalized["updated_at"] = time.time()
    normalized["step_label"] = str(normalized.get("step_label") or "Interrupted")
    normalized["substep_current"] = 0
    normalized["substep_total"] = 0
    normalized["substep_label"] = ""
    normalized["state_schema_version"] = CALIBRATION_JOB_STATE_VERSION
    return normalized


def _load_persisted_calibration_job_payload(
    calibration_root: Path,
    job_id: str,
    *,
    mark_interrupted: bool = False,
) -> Optional[Dict[str, Any]]:
    try:
        path = _calibration_job_state_path(calibration_root, job_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if mark_interrupted and str(payload.get("status") or "") in {"queued", "running"}:
        payload = _mark_persisted_job_interrupted(payload)
        try:
            _write_json_atomic(path, payload)
        except Exception:
            pass
    return payload


def _list_persisted_calibration_job_payloads(
    calibration_root: Path,
    *,
    mark_interrupted: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        root = _resolve_calibration_storage_root(calibration_root)
    except ValueError:
        return out
    if not root.exists():
        return out
    for entry in sorted(root.iterdir()):
        if entry.is_symlink():
            continue
        try:
            resolved_entry = entry.resolve()
        except Exception:
            continue
        if not _path_is_within_root_impl(resolved_entry, root):
            continue
        if not resolved_entry.is_dir():
            continue
        payload = _load_persisted_calibration_job_payload(
            root,
            entry.name,
            mark_interrupted=mark_interrupted,
        )
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _canonical_recipe_json_path(run_root: Path) -> Path:
    edr_path = run_root / CANONICAL_EDR_JSON_NAME
    if edr_path.exists():
        return edr_path
    return run_root / LEGACY_CANONICAL_RECIPE_JSON_NAME


def _canonical_recipe_md_path(run_root: Path) -> Path:
    edr_path = run_root / CANONICAL_EDR_MD_NAME
    if edr_path.exists():
        return edr_path
    return run_root / LEGACY_CANONICAL_RECIPE_MD_NAME


def _normalize_eval_metrics_for_api(eval_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Expose the requested reference operating point as the top-level API metrics.

    The XGB evaluator keeps the best IoU/dedupe sweep row at the top level and the
    caller-requested operating point under ``reference_iou.xgb_ensemble``. The job
    API should report the requested operating point first so users do not have to
    reverse-engineer which row the run was actually configured to target.
    """

    if not isinstance(eval_metrics, dict):
        return eval_metrics
    reference_iou = eval_metrics.get("reference_iou")
    if not isinstance(reference_iou, dict):
        return eval_metrics
    reference_metrics = reference_iou.get("xgb_ensemble")
    if not isinstance(reference_metrics, dict):
        return eval_metrics

    normalized = dict(eval_metrics)
    best_sweep_metrics = {
        key: eval_metrics.get(key)
        for key in _CALIBRATION_METRIC_SUMMARY_KEYS
        if key in eval_metrics
    }
    best_differs_from_reference = False
    for key in _CALIBRATION_METRIC_SUMMARY_KEYS:
        if key not in reference_metrics:
            continue
        if normalized.get(key) != reference_metrics[key]:
            best_differs_from_reference = True
        normalized[key] = reference_metrics[key]
    normalized["reported_operating_point"] = {
        "kind": "reference_iou",
        "dedupe_iou": reference_iou.get("dedupe_iou"),
        "eval_iou": reference_iou.get("eval_iou"),
    }
    if best_differs_from_reference and best_sweep_metrics:
        normalized["best_sweep_metrics"] = best_sweep_metrics
    return normalized


def _serialize_calibration_job(job: Any) -> Dict[str, Any]:
    serialized_result = job.result
    if isinstance(serialized_result, dict):
        eval_path = serialized_result.get("eval")
        if eval_path:
            try:
                eval_metrics = json.loads(Path(eval_path).read_text(encoding="utf-8"))
                if isinstance(eval_metrics, dict):
                    serialized_result = dict(serialized_result)
                    serialized_result["metrics"] = _normalize_eval_metrics_for_api(eval_metrics)
            except Exception:
                pass
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "phase": job.phase,
        "progress": json_sanitize(job.progress),
        "processed": job.processed,
        "total": job.total,
        "step_current": getattr(job, "step_current", 0),
        "step_total": getattr(job, "step_total", 0),
        "step_label": getattr(job, "step_label", ""),
        "substep_current": getattr(job, "substep_current", 0),
        "substep_total": getattr(job, "substep_total", 0),
        "substep_label": getattr(job, "substep_label", ""),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "request": json_sanitize(job.request),
        "result": json_sanitize(serialized_result),
        "error": json_sanitize(job.error),
    }


@dataclass
class CalibrationJob:
    job_id: str
    status: str = "queued"
    message: str = "Queued"
    phase: str = "queued"
    progress: float = 0.0
    processed: int = 0
    total: int = 0
    step_current: int = 0
    step_total: int = 0
    step_label: str = ""
    substep_current: int = 0
    substep_total: int = 0
    substep_label: str = ""
    request: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: Any = field(default_factory=threading.Event)


def _remove_calibration_job_state(calibration_root: Path, job_id: str) -> None:
    try:
        state_path = _calibration_job_state_path(calibration_root, job_id)
    except ValueError:
        return
    try:
        state_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        state_path.parent.rmdir()
    except Exception:
        pass


def _build_calibration_step_plan(
    *,
    recipe_mode: str,
    calibration_model: str,
    policy_layer_variant: str,
) -> List[Dict[str, str]]:
    plan: List[Dict[str, str]] = [{"phase": "select_images", "label": "Select images"}]
    plan.append({"phase": "fingerprint", "label": "Fingerprint EDR"})
    if recipe_mode != "force_rediscover":
        plan.append({"phase": "recipe_lookup", "label": "Lookup promoted EDR"})
    if recipe_mode != "reuse_only":
        plan.append({"phase": "recipe_discovery", "label": "Discover canonical EDR"})
    plan.extend(
        [
            {"phase": "prepass", "label": "Build prepass"},
            {"phase": "features", "label": "Build features"},
            {"phase": "labeling", "label": "Label candidates"},
            {"phase": "train", "label": "Train scorer"},
        ]
    )
    if calibration_model == "xgb":
        plan.append({"phase": "relax", "label": "Relax thresholds"})
        plan.append({"phase": "objective", "label": "Tune thresholds"})
        if policy_layer_variant != "none":
            plan.append({"phase": "policy", "label": "Train policy layer"})
    else:
        plan.append({"phase": "calibrate", "label": "Calibrate thresholds"})
        plan.append({"phase": "relax", "label": "Relax thresholds"})
    plan.append({"phase": "eval", "label": "Evaluate EDR"})
    plan.append({"phase": "report", "label": "Build EDR report"})
    return plan


def _calibration_step_payload(
    step_plan: List[Dict[str, str]],
    phase: str,
    *,
    substep_current: int = 0,
    substep_total: int = 0,
    substep_label: str = "",
) -> Dict[str, Any]:
    phase_key = str(phase or "").strip()
    total = len(step_plan)
    for idx, step in enumerate(step_plan, start=1):
        if str(step.get("phase") or "") == phase_key:
            return {
                "step_current": idx,
                "step_total": total,
                "step_label": str(step.get("label") or phase_key.replace("_", " ")),
                "substep_current": max(0, int(substep_current or 0)),
                "substep_total": max(0, int(substep_total or 0)),
                "substep_label": str(substep_label or "").strip(),
            }
    return {}


def _update_calibration_phase(
    *,
    job: Any,
    update_fn: Callable[..., None],
    step_plan: List[Dict[str, str]],
    phase: str,
    message: Optional[str] = None,
    substep_current: int = 0,
    substep_total: int = 0,
    substep_label: str = "",
    **kwargs: Any,
) -> None:
    payload = dict(kwargs)
    payload["phase"] = phase
    if message is not None:
        payload["message"] = message
    payload.update(
        _calibration_step_payload(
            step_plan,
            phase,
            substep_current=substep_current,
            substep_total=substep_total,
            substep_label=substep_label,
        )
    )
    update_fn(job, **payload)


def _normalize_classifier_id_for_fingerprint(raw_value: Any, *, root_dir: Path) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return ""
    raw_path = Path(raw)
    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                raw_path,
                root_dir / raw_path,
                root_dir / "uploads" / "classifiers" / raw_path,
                root_dir / "uploads" / "classifiers" / raw_path.name,
            ]
        )
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists() and resolved.is_file():
            return key
    if raw_path.suffix:
        return raw_path.name
    return raw


def _normalize_similarity_strategy(value: Any) -> str:
    strategy = str(value or "top").strip().lower()
    if strategy not in {"top", "random", "diverse"}:
        strategy = "top"
    return strategy


def _coerce_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _float_with_default(value: Any, default: float) -> float:
    parsed = _coerce_float(value)
    if parsed is None:
        return float(default)
    return parsed


def _first_float_with_default(*values: Any, default: float) -> float:
    for value in values:
        if value is None:
            continue
        parsed = _coerce_float(value)
        if parsed is not None:
            return parsed
    return float(default)


def _validate_classifier_feature_matrix(path: Path) -> None:
    with np.load(path, allow_pickle=True) as data:
        feature_names = [str(name) for name in data.get("feature_names", [])]
        classifier_classes = [str(name) for name in data.get("classifier_classes", [])]
        try:
            embed_proj_dim = int(data.get("embed_proj_dim", 0))
        except Exception:
            embed_proj_dim = 0
        X = np.asarray(data.get("X"), dtype=np.float32)

    if not classifier_classes:
        raise RuntimeError("classifier_classes_empty")
    if embed_proj_dim <= 0:
        raise RuntimeError("embed_proj_dim_zero")

    embed_idx = [
        idx
        for idx, name in enumerate(feature_names)
        if name.startswith("clf_emb_rp::") or name.startswith("embed_proj_")
    ]
    if not embed_idx:
        raise RuntimeError("embed_features_missing")
    clf_prob_idx = [idx for idx, name in enumerate(feature_names) if name.startswith("clf_prob::")]
    if not clf_prob_idx:
        raise RuntimeError("classifier_prob_features_missing")
    if X.ndim != 2:
        raise RuntimeError("feature_matrix_invalid_shape")
    if X.shape[1] != len(feature_names):
        raise RuntimeError("feature_dim_mismatch")
    if X.shape[0] > 0:
        if np.allclose(X[:, embed_idx], 0.0):
            raise RuntimeError("embed_features_all_zero")
        if np.allclose(X[:, clf_prob_idx], 0.0):
            raise RuntimeError("classifier_prob_features_all_zero")


def _canonical_similarity_settings(payload: Any) -> Dict[str, Any]:
    strategy = _normalize_similarity_strategy(getattr(payload, "similarity_exemplar_strategy", None))
    raw_count = getattr(payload, "similarity_exemplar_count", None)
    count = int(raw_count or 3)
    if count < 1:
        count = 1
    raw_seed = getattr(payload, "similarity_exemplar_seed", None)
    if strategy in {"random", "diverse"}:
        seed = int(raw_seed or 0)
    else:
        seed = None
    raw_fraction = getattr(payload, "similarity_exemplar_fraction", None)
    try:
        fraction = float(raw_fraction if raw_fraction is not None else 0.2)
    except (TypeError, ValueError):
        fraction = 0.2
    if not math.isfinite(fraction) or fraction <= 0:
        fraction = 0.2
    raw_min = getattr(payload, "similarity_exemplar_min", None)
    min_count = int(raw_min or 3)
    if min_count < 1:
        min_count = 1
    raw_max = getattr(payload, "similarity_exemplar_max", None)
    max_count = int(raw_max or 12)
    if max_count < min_count:
        max_count = min_count
    raw_quota = getattr(payload, "similarity_exemplar_source_quota", None)
    try:
        source_quota = int(raw_quota) if raw_quota is not None else 1
    except (TypeError, ValueError):
        source_quota = 1
    if source_quota < 0:
        source_quota = 0
    return {
        "strategy": strategy,
        "count": count if strategy in {"top", "random"} else None,
        "seed": seed,
        "fraction": fraction if strategy == "diverse" else None,
        "min": min_count if strategy == "diverse" else None,
        "max": max_count if strategy == "diverse" else None,
        "source_quota": source_quota if strategy == "diverse" else None,
    }


def _canonical_cross_class_dedupe_settings(payload: Any) -> Dict[str, Any]:
    enabled = bool(getattr(payload, "cross_class_dedupe_enabled", False))
    if not enabled:
        return {"enabled": False, "iou": None}
    raw_iou = getattr(payload, "cross_class_dedupe_iou", None)
    try:
        iou = float(raw_iou if raw_iou is not None else 0.8)
    except (TypeError, ValueError):
        iou = 0.8
    iou = max(0.0, min(1.0, iou))
    if iou <= 0.0:
        return {"enabled": False, "iou": None}
    return {"enabled": True, "iou": iou}


def _normalize_window_mode(value: Any) -> str:
    mode = str(value or "grid").strip().lower()
    if mode not in {"grid", "sahi"}:
        mode = "grid"
    return mode


def _canonical_sam3_text_window_settings(payload: Any) -> Dict[str, Any]:
    enabled = bool(getattr(payload, "sam3_text_window_extension", False))
    if not enabled:
        return {"enabled": False, "mode": None, "size": None, "overlap": None}
    mode = _normalize_window_mode(getattr(payload, "sam3_text_window_mode", None))
    if mode != "sahi":
        return {"enabled": True, "mode": mode, "size": None, "overlap": None}
    try:
        size = int(getattr(payload, "sam3_text_window_size", None) or 640)
    except (TypeError, ValueError):
        size = 640
    if size <= 0:
        size = 640
    overlap = _float_with_default(getattr(payload, "sam3_text_window_overlap", None), 0.2)
    if overlap <= 0.0 or overlap >= 1.0:
        overlap = 0.2
    return {
        "enabled": True,
        "mode": mode,
        "size": size,
        "overlap": overlap,
    }


def _canonical_similarity_window_settings(payload: Any) -> Dict[str, Any]:
    enabled = bool(getattr(payload, "similarity_window_extension", False))
    if not enabled:
        return {"enabled": False, "mode": None, "size": None, "overlap": None}
    mode = _normalize_window_mode(getattr(payload, "similarity_window_mode", None))
    if mode != "sahi":
        return {"enabled": True, "mode": mode, "size": None, "overlap": None}
    try:
        size = int(getattr(payload, "similarity_window_size", None) or 640)
    except (TypeError, ValueError):
        size = 640
    if size <= 0:
        size = 640
    overlap = _float_with_default(getattr(payload, "similarity_window_overlap", None), 0.2)
    if overlap <= 0.0 or overlap >= 1.0:
        overlap = 0.2
    return {
        "enabled": True,
        "mode": mode,
        "size": size,
        "overlap": overlap,
    }


def _canonical_sahi_settings(payload: Any) -> Dict[str, float | int]:
    raw_size = getattr(payload, "sahi_window_size", None)
    try:
        size = int(raw_size) if raw_size is not None else 640
    except (TypeError, ValueError):
        size = 640
    if size <= 0:
        size = 640

    raw_overlap = getattr(payload, "sahi_overlap_ratio", None)
    overlap = _float_with_default(raw_overlap, 0.2)
    if overlap <= 0.0 or overlap >= 1.0:
        overlap = 0.2
    return {"size": size, "overlap": overlap}


def _resolve_default_ensemble_policy_json(
    payload: Any,
    *,
    sam3_text_window_cfg: Dict[str, Any],
    similarity_window_cfg: Dict[str, Any],
) -> Optional[str]:
    override = str(getattr(payload, "ensemble_policy_json", "") or "").strip()
    if override:
        return override
    if getattr(payload, "apply_default_ensemble_policy", True) is False:
        return None
    use_window_policy = _use_windowed_calibration_defaults(
        sam3_text_window_cfg=sam3_text_window_cfg,
        similarity_window_cfg=similarity_window_cfg,
    )
    policy = copy.deepcopy(
        DEFAULT_ENSEMBLE_POLICY_WINDOW if use_window_policy else DEFAULT_ENSEMBLE_POLICY_NONWINDOW
    )
    return json.dumps(policy, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _use_windowed_calibration_defaults(
    *,
    sam3_text_window_cfg: Dict[str, Any],
    similarity_window_cfg: Dict[str, Any],
) -> bool:
    return bool(sam3_text_window_cfg.get("enabled")) or bool(similarity_window_cfg.get("enabled"))


def _resolve_default_split_head_by_support(
    payload: Any,
    *,
    sam3_text_window_cfg: Dict[str, Any],
    similarity_window_cfg: Dict[str, Any],
) -> bool:
    raw = getattr(payload, "split_head_by_support", None)
    if raw is not None:
        return bool(raw)
    if _use_windowed_calibration_defaults(
        sam3_text_window_cfg=sam3_text_window_cfg,
        similarity_window_cfg=similarity_window_cfg,
    ):
        return DEFAULT_WINDOWED_SPLIT_HEAD_BY_SUPPORT
    return DEFAULT_NONWINDOWED_SPLIT_HEAD_BY_SUPPORT


def _resolve_default_sam3_text_quality_alpha(
    payload: Any,
    *,
    sam3_text_window_cfg: Dict[str, Any],
    similarity_window_cfg: Dict[str, Any],
) -> float:
    raw = getattr(payload, "sam3_text_quality_alpha", None)
    if raw is not None:
        return _float_with_default(raw, DEFAULT_WINDOWED_SAM3_TEXT_QUALITY_ALPHA)
    if _use_windowed_calibration_defaults(
        sam3_text_window_cfg=sam3_text_window_cfg,
        similarity_window_cfg=similarity_window_cfg,
    ):
        return DEFAULT_WINDOWED_SAM3_TEXT_QUALITY_ALPHA
    return DEFAULT_NONWINDOWED_SAM3_TEXT_QUALITY_ALPHA


def _resolve_default_train_sam3_similarity_quality(
    payload: Any,
    *,
    sam3_text_window_cfg: Dict[str, Any],
    similarity_window_cfg: Dict[str, Any],
) -> bool:
    raw = getattr(payload, "train_sam3_similarity_quality", None)
    if raw is not None:
        return bool(raw)
    if _use_windowed_calibration_defaults(
        sam3_text_window_cfg=sam3_text_window_cfg,
        similarity_window_cfg=similarity_window_cfg,
    ):
        return DEFAULT_WINDOWED_TRAIN_SAM3_SIMILARITY_QUALITY
    return DEFAULT_NONWINDOWED_TRAIN_SAM3_SIMILARITY_QUALITY


def _resolve_default_sam3_similarity_quality_alpha(
    payload: Any,
    *,
    sam3_text_window_cfg: Dict[str, Any],
    similarity_window_cfg: Dict[str, Any],
) -> float:
    raw = getattr(payload, "sam3_similarity_quality_alpha", None)
    if raw is not None:
        return _float_with_default(raw, DEFAULT_WINDOWED_SAM3_SIMILARITY_QUALITY_ALPHA)
    if _use_windowed_calibration_defaults(
        sam3_text_window_cfg=sam3_text_window_cfg,
        similarity_window_cfg=similarity_window_cfg,
    ):
        return DEFAULT_WINDOWED_SAM3_SIMILARITY_QUALITY_ALPHA
    return DEFAULT_NONWINDOWED_SAM3_SIMILARITY_QUALITY_ALPHA


def _resolve_policy_layer_variant(payload: Any) -> str:
    variant = str(getattr(payload, "policy_layer_variant", "none") or "none").strip().lower()
    if variant not in {"none", "bakeoff", "xgb", "lreg"}:
        variant = "none"
    return variant


def _resolve_recipe_mode(payload: Any) -> str:
    mode = str(getattr(payload, "recipe_mode", "auto") or "auto").strip().lower()
    if mode not in {"auto", "reuse_only", "force_rediscover"}:
        mode = "auto"
    return mode


def _resolve_lane_selection(payload: Any) -> str:
    selection = str(
        getattr(payload, "lane_selection", DEFAULT_CALIBRATION_LANE_SELECTION)
        or DEFAULT_CALIBRATION_LANE_SELECTION
    ).strip().lower()
    if selection not in {"window", "nonwindow", "compare_both"}:
        selection = DEFAULT_CALIBRATION_LANE_SELECTION
    return selection


def _detector_fingerprint(active: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(active.get("run_id") or "").strip() or None
    best_path = str(active.get("best_path") or "").strip() or None
    labelmap_path = str(active.get("labelmap_path") or "").strip() or None
    stat_size = None
    stat_mtime_ns = None
    if best_path:
        try:
            stat = Path(best_path).stat()
            stat_size = int(stat.st_size)
            stat_mtime_ns = int(stat.st_mtime_ns)
        except Exception:
            stat_size = None
            stat_mtime_ns = None
    return {
        "run_id": run_id,
        "best_path": best_path,
        "best_size": stat_size,
        "best_mtime_ns": stat_mtime_ns,
        "labelmap_path": labelmap_path,
    }


def _apply_canonical_recipe_to_payload(payload: Any, canonical_recipe: Dict[str, Any]) -> Any:
    scenario = canonical_recipe.get("scenario") if isinstance(canonical_recipe.get("scenario"), dict) else {}
    policy = canonical_recipe.get("policy") if isinstance(canonical_recipe.get("policy"), dict) else {}
    # Normal promoted EDRs no longer publish the learned second-stage block.
    # Keep reading the legacy field so older EDRs and explicit research-only
    # API calls remain backward compatible.
    second_stage = (
        canonical_recipe.get("second_stage_policy_layer")
        if isinstance(canonical_recipe.get("second_stage_policy_layer"), dict)
        else {}
    )
    hp = canonical_recipe.get("xgb_hparams") if isinstance(canonical_recipe.get("xgb_hparams"), dict) else {}
    winner_lane = str(canonical_recipe.get("winner_lane") or "")
    requested_policy_layer_variant = _resolve_policy_layer_variant(payload)

    updates: Dict[str, Any] = {
        "apply_default_ensemble_policy": False,
        "ensemble_policy_json": json.dumps(policy, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        if policy
        else None,
        "split_head_by_support": bool(scenario.get("split_head")) if "split_head" in scenario else None,
        "train_sam3_text_quality": bool(scenario.get("train_sam3_text_quality", True)),
        "sam3_text_quality_alpha": scenario.get("sam3_text_quality_alpha"),
        "train_sam3_similarity_quality": bool(scenario.get("train_sam3_similarity_quality", False)),
        "sam3_similarity_quality_alpha": scenario.get("sam3_similarity_quality_alpha"),
        "policy_layer_variant": (
            requested_policy_layer_variant
            if requested_policy_layer_variant != "none"
            else (
                str(second_stage.get("variant") or "none")
                if bool(second_stage.get("enabled", False))
                else "none"
            )
        ),
        "xgb_max_depth": hp.get("max_depth"),
        "xgb_n_estimators": hp.get("n_estimators"),
        "xgb_learning_rate": hp.get("learning_rate"),
        "xgb_subsample": hp.get("subsample"),
        "xgb_colsample_bytree": hp.get("colsample_bytree"),
        "xgb_min_child_weight": hp.get("min_child_weight"),
        "xgb_gamma": hp.get("gamma"),
        "xgb_reg_lambda": hp.get("reg_lambda"),
        "xgb_reg_alpha": hp.get("reg_alpha"),
        "image_embed_proj_dim": 0,
        "image_embed_proj_seed": DEFAULT_IMAGE_EMBED_PROJ_SEED,
    }
    if winner_lane == "nonwindow":
        updates["sam3_text_window_extension"] = False
        updates["similarity_window_extension"] = False
    elif winner_lane == "window":
        # Preserve the caller's explicit window sizing/overlap settings but ensure
        # the winner lane family remains window-enabled.
        updates["sam3_text_window_extension"] = True
        updates["similarity_window_extension"] = True

    try:
        return model_copy_update(payload, updates, deep=True)
    except Exception:
        pass
    if hasattr(payload, "dict"):
        base = model_dump_compat(payload)
    elif hasattr(payload, "__dict__"):
        base = dict(vars(payload))
    else:
        base = dict(payload)
    base.update(updates)
    return SimpleNamespace(**base)


def _select_effective_canonical_recipe(
    canonical_recipe_payload: Dict[str, Any],
    *,
    lane_selection: str,
    fallback_windowed: bool,
) -> Tuple[str, Dict[str, Any]]:
    if lane_selection == "window":
        lane_family = "windowed"
    elif lane_selection == "nonwindow":
        lane_family = "nonwindowed"
    else:
        winner_lane = str(canonical_recipe_payload.get("discovered_winner_lane") or "").strip().lower()
        if winner_lane == "nonwindow":
            lane_family = "nonwindowed"
        elif winner_lane == "window":
            lane_family = "windowed"
        else:
            lane_family = "windowed" if fallback_windowed else "nonwindowed"
    recipe = (
        canonical_recipe_payload.get("canonical_windowed_recipe")
        if lane_family == "windowed"
        else canonical_recipe_payload.get("canonical_nonwindowed_recipe")
    )
    if not isinstance(recipe, dict) or not recipe:
        raise RuntimeError(f"canonical_recipe_branch_missing:{lane_family}")
    return lane_family, recipe


def _prepass_hash_config_legacy_compatible(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize hash inputs so default/no-op knobs don't churn cache keys.

    This preserves compatibility with older cache keys where certain knobs were not
    serialized when they had default/no-effect values.
    """
    out = dict(config)

    strategy = _normalize_similarity_strategy(out.get("similarity_exemplar_strategy"))
    if strategy != "diverse":
        out.pop("similarity_exemplar_fraction", None)
        out.pop("similarity_exemplar_min", None)
        out.pop("similarity_exemplar_max", None)
        out.pop("similarity_exemplar_source_quota", None)
    if strategy == "top" and out.get("similarity_exemplar_seed") is None:
        # Legacy top strategy persisted seed=0; keep compatibility for cache reuse.
        out["similarity_exemplar_seed"] = 0

    if not bool(out.get("cross_class_dedupe_enabled", False)):
        out.pop("cross_class_dedupe_enabled", None)
        out.pop("cross_class_dedupe_iou", None)

    fusion_mode = str(out.get("fusion_mode") or "").strip().lower()
    if fusion_mode in {"", "primary"}:
        out.pop("fusion_mode", None)

    # Legacy non-windowed payloads kept "grid" mode even when extension was off.
    if out.get("sam3_text_window_extension") is False and out.get("sam3_text_window_mode") is None:
        out["sam3_text_window_mode"] = "grid"

    return out


def _start_calibration_job(
    payload: Any,
    *,
    job_cls: Any,
    jobs: Dict[str, Any],
    jobs_lock: Any,
    run_job_fn: Callable[[Any, Any], None],
    calibration_root: Path,
) -> Any:
    job_id = f"cal_{uuid.uuid4().hex[:8]}"
    job = job_cls(job_id=job_id)
    try:
        request_payload = model_dump_compat(payload)
    except Exception:
        request_payload = {}
    if isinstance(request_payload, dict):
        job.request = request_payload
    with jobs_lock:
        jobs[job.job_id] = job
    try:
        _persist_calibration_job_state(job, calibration_root)
    except Exception:
        with jobs_lock:
            if jobs.get(job.job_id) is job:
                jobs.pop(job.job_id, None)
        raise
    try:
        thread = threading.Thread(target=run_job_fn, args=(job, payload), daemon=True)
        thread.start()
    except Exception:
        with jobs_lock:
            if jobs.get(job.job_id) is job:
                jobs.pop(job.job_id, None)
        _remove_calibration_job_state(calibration_root, job.job_id)
        raise
    return job


def _cancel_calibration_job(
    job_id: str,
    *,
    jobs: Dict[str, Any],
    jobs_lock: Any,
    http_exception_cls: Any,
    time_fn: Callable[[], float],
) -> Any:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise http_exception_cls(status_code=404, detail="calibration_job_not_found")
    if job.status in {"completed", "failed", "cancelled"}:
        return job
    job.cancel_event.set()
    job.status = "cancelled"
    job.message = "Cancelled"
    job.phase = "cancelled"
    job.updated_at = time_fn()
    return job


def _ensure_prepass_jsonl(
    *,
    job: Any,
    update_fn: Callable[..., None],
    selected: List[str],
    total: int,
    dataset_id: str,
    labelmap: List[str],
    glossary: str,
    prepass_payload: Any,
    prepass_config: Dict[str, Any],
    prepass_config_for_hash: Dict[str, Any],
    output_path: Path,
    calibration_cache_root: Path,
    write_record_fn: Callable[[Path, Dict[str, Any]], None],
    hash_payload_fn: Callable[[Dict[str, Any]], str],
    prepass_worker_fn: Callable[..., None],
    unload_inference_runtimes_fn: Callable[[], None],
    resolve_dataset_fn: Callable[[str], Path],
    cache_image_fn: Callable[[Image.Image, Optional[str]], str],
    run_prepass_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    _prepare_calibration_storage_dir(
        output_path.parent,
        error_prefix="calibration_prepass_output_parent",
    )
    labelmap_hash = hashlib.sha1(",".join(labelmap).encode("utf-8")).hexdigest()
    prepass_glossary_text = glossary or ""
    glossary_hash = hashlib.sha1(prepass_glossary_text.encode("utf-8")).hexdigest()
    selected_hash = hashlib.sha1(json.dumps(selected, sort_keys=True).encode("utf-8")).hexdigest()

    prepass_config_key = hash_payload_fn(
        {
            "dataset_id": dataset_id,
            "labelmap_hash": labelmap_hash,
            "glossary_hash": glossary_hash,
            "glossary_text": prepass_glossary_text,
            "prepass": prepass_config_for_hash,
        }
    )
    prepass_key = hash_payload_fn(
        {
            "prepass_config_key": prepass_config_key,
            "selected_hash": selected_hash,
        }
    )
    prepass_cache_dir = calibration_cache_root / "prepass" / prepass_config_key
    image_cache_dir = prepass_cache_dir / "images"
    _prepare_calibration_storage_dir(
        prepass_cache_dir,
        root=calibration_cache_root,
        error_prefix="calibration_prepass_cache_dir",
    )
    _prepare_calibration_storage_dir(
        image_cache_dir,
        root=calibration_cache_root,
        error_prefix="calibration_prepass_image_cache_dir",
    )
    prepass_cache_meta = prepass_cache_dir / "prepass.meta.json"
    glossary_path = prepass_cache_dir / "glossary.json"

    def _normalize_glossary_payload(text: str) -> Any:
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            return text

    if not prepass_cache_meta.exists():
        write_record_fn(
            prepass_cache_meta,
            {
                "dataset_id": dataset_id,
                "labelmap": labelmap,
                "labelmap_hash": labelmap_hash,
                "glossary_text": prepass_glossary_text,
                "glossary_hash": glossary_hash,
                "prepass_config": prepass_config,
                "prepass_config_key": prepass_config_key,
                "created_at": time.time(),
            },
        )
        write_record_fn(
            glossary_path,
            {
                "glossary": _normalize_glossary_payload(prepass_glossary_text),
                "glossary_hash": glossary_hash,
            },
        )

    def _safe_image_cache_name(image_name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_name)
        if safe != image_name:
            suffix = hashlib.sha1(image_name.encode("utf-8")).hexdigest()[:8]
            safe = f"{safe}_{suffix}"
        return safe

    def _cache_path_for_image(image_name: str) -> Path:
        return image_cache_dir / f"{_safe_image_cache_name(image_name)}.json"

    def _load_cached_record(image_name: str) -> Optional[Dict[str, Any]]:
        path = _cache_path_for_image(image_name)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def _write_cached_record(image_name: str, record: Dict[str, Any]) -> None:
        path = _cache_path_for_image(image_name)
        write_record_fn(path, record)

    cached_records: Dict[str, Dict[str, Any]] = {}
    for image_name in selected:
        cached = _load_cached_record(image_name)
        if cached:
            cached_records[image_name] = cached

    processed = len(cached_records)
    if processed:
        update_fn(
            job,
            message="Using cached prepass (partial)…",
            phase="prepass",
            processed=processed,
            progress=processed / total if total else 1.0,
        )

    if processed < total:
        update_fn(job, message="Running deep prepass…", phase="prepass")
        remaining = [image_name for image_name in selected if image_name not in cached_records]
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and remaining:
            unload_inference_runtimes_fn()
            devices = list(range(torch.cuda.device_count()))
            worker_count = min(len(devices), len(remaining))
            tasks = [(image_name, str(_cache_path_for_image(image_name))) for image_name in remaining]
            buckets: List[List[Tuple[str, str]]] = [[] for _ in range(worker_count)]
            for idx, task in enumerate(tasks):
                buckets[idx % worker_count].append(task)
            ctx = multiprocessing.get_context("spawn")
            mp_cancel = ctx.Event()
            progress_queue = ctx.Queue()
            workers = []
            prepass_payload_dict = model_dump_compat(prepass_payload)
            for worker_idx in range(worker_count):
                device_index = devices[worker_idx]
                bucket = buckets[worker_idx]
                if not bucket:
                    continue
                proc = ctx.Process(
                    target=prepass_worker_fn,
                    args=(
                        device_index,
                        bucket,
                        dataset_id,
                        labelmap,
                        glossary,
                        prepass_payload_dict,
                        mp_cancel,
                        progress_queue,
                    ),
                    daemon=True,
                )
                proc.start()
                workers.append(proc)
            processed_local = 0
            while any(proc.is_alive() for proc in workers):
                if job.cancel_event.is_set():
                    mp_cancel.set()
                try:
                    inc = progress_queue.get(timeout=1.0)
                    if isinstance(inc, int):
                        processed_local += inc
                        processed = len(cached_records) + processed_local
                        progress = processed / total if total else 1.0
                        update_fn(job, processed=processed, progress=progress)
                except queue.Empty:
                    pass
                if mp_cancel.is_set():
                    break
            for proc in workers:
                proc.join(timeout=5)
            stuck_workers = [proc for proc in workers if proc.is_alive()]
            if stuck_workers:
                for proc in stuck_workers:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                raise RuntimeError(
                    "deep_prepass_worker_hung:" + ",".join(str(proc.pid or "?") for proc in stuck_workers)
                )
            if mp_cancel.is_set():
                for proc in workers:
                    if proc.is_alive():
                        proc.terminate()
                raise RuntimeError("cancelled")
            failed_workers = [proc for proc in workers if proc.exitcode not in (0, None)]
            if failed_workers:
                raise RuntimeError(
                    "deep_prepass_worker_exit:"
                    + ",".join(f"{proc.pid or '?'}:{proc.exitcode}" for proc in failed_workers)
                )
        else:
            dataset_root = resolve_dataset_fn(dataset_id)
            for image_name in remaining:
                if job.cancel_event.is_set():
                    raise RuntimeError("cancelled")
                img_path = _calibration_resolve_image_path(dataset_root, image_name)
                if img_path is None:
                    record = {
                        "image": image_name,
                        "dataset_id": dataset_id,
                        "detections": [],
                        "warnings": ["deep_prepass_image_missing"],
                    }
                    _write_cached_record(image_name, record)
                    cached_records[image_name] = record
                    processed += 1
                    progress = processed / total if total else 1.0
                    update_fn(job, processed=processed, progress=progress)
                    continue
                try:
                    with Image.open(img_path) as img:
                        pil_img = img.convert("RGB")
                except Exception as exc:
                    record = {
                        "image": image_name,
                        "dataset_id": dataset_id,
                        "detections": [],
                        "warnings": [f"deep_prepass_image_open_failed:{exc}"],
                    }
                    _write_cached_record(image_name, record)
                    cached_records[image_name] = record
                    processed += 1
                    progress = processed / total if total else 1.0
                    update_fn(job, processed=processed, progress=progress)
                    continue
                try:
                    image_token = cache_image_fn(pil_img, prepass_payload.sam_variant)
                    result = run_prepass_fn(
                        prepass_payload,
                        pil_img=pil_img,
                        image_token=image_token,
                        labelmap=labelmap,
                        glossary=glossary,
                        trace_writer=None,
                        trace_full_writer=None,
                        trace_readable=None,
                    )
                    detections = list(result.get("detections") or [])
                    warnings = list(result.get("warnings") or [])
                    provenance = result.get("provenance")
                except Exception as exc:
                    detections = []
                    warnings = [f"deep_prepass_worker_failed:{exc}"]
                    provenance = None
                record = {
                    "image": image_name,
                    "dataset_id": dataset_id,
                    "detections": detections,
                    "warnings": warnings,
                }
                if isinstance(provenance, dict):
                    record["provenance"] = provenance
                _write_cached_record(image_name, record)
                cached_records[image_name] = record
                processed += 1
                progress = processed / total if total else 1.0
                update_fn(job, processed=processed, progress=progress)

    missing_records: List[str] = []
    output_records: List[Dict[str, Any]] = []
    for image_name in selected:
        record = cached_records.get(image_name) or _load_cached_record(image_name)
        if not record:
            missing_records.append(image_name)
            continue
        output_records.append(record)
    if missing_records:
        sample = ",".join(missing_records[:5])
        raise RuntimeError(f"prepass_cache_incomplete:{len(missing_records)}:{sample}")
    output_text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output_records)
    _write_text_atomic(output_path, output_text)

    _write_json_atomic(
        prepass_cache_meta,
        {
            "dataset_id": dataset_id,
            "labelmap_hash": labelmap_hash,
            "glossary_hash": glossary_hash,
            "glossary_text": prepass_glossary_text,
            "prepass_config": prepass_config,
            "cached_images": len(list(image_cache_dir.glob("*.json"))),
            "updated_at": time.time(),
            "config_key": prepass_config_key,
        },
    )
    return {
        "prepass_path": output_path,
        "prepass_config_key": prepass_config_key,
        "prepass_key": prepass_key,
        "labelmap_hash": labelmap_hash,
        "glossary_hash": glossary_hash,
        "selected_hash": selected_hash,
    }


def _build_prepass_request_and_config(
    *,
    payload: Any,
    dataset_id: str,
    classifier_id_resolved: str,
    use_yolo: bool,
    use_rfdetr: bool,
    yolo_run_id: Optional[str],
    rfdetr_run_id: Optional[str],
    similarity_cfg: Dict[str, Any],
    cross_class_cfg: Dict[str, Any],
    sam3_text_window_cfg: Dict[str, Any],
    similarity_window_cfg: Dict[str, Any],
    sahi_cfg: Dict[str, Any],
    prepass_request_cls: Any,
    detector_conf: float,
    sam3_score_thr: float,
    sam3_mask_threshold: float,
    prepass_sam3_text_thr: float,
    prepass_similarity_score: float,
    similarity_min_exemplar_score: float,
    scoreless_iou: float,
    dedupe_iou: float,
    yolo_fingerprint: Optional[Dict[str, Any]],
    rfdetr_fingerprint: Optional[Dict[str, Any]],
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    prepass_payload = prepass_request_cls(
        dataset_id=dataset_id,
        enable_yolo=use_yolo,
        enable_rfdetr=use_rfdetr,
        yolo_id=yolo_run_id,
        rfdetr_id=rfdetr_run_id,
        sam_variant="sam3",
        enable_sam3_text=True,
        enable_sam3_similarity=True,
        prepass_caption=False,
        prepass_only=True,
        prepass_finalize=False,
        prepass_keep_all=True,
        sam3_text_synonym_budget=None
        if payload.sam3_text_synonym_budget is None
        else int(payload.sam3_text_synonym_budget),
        sam3_text_window_extension=sam3_text_window_cfg["enabled"],
        sam3_text_window_mode=sam3_text_window_cfg["mode"],
        sam3_text_window_size=sam3_text_window_cfg["size"],
        sam3_text_window_overlap=sam3_text_window_cfg["overlap"],
        prepass_sam3_text_thr=prepass_sam3_text_thr,
        prepass_similarity_score=prepass_similarity_score,
        similarity_min_exemplar_score=similarity_min_exemplar_score,
        similarity_exemplar_count=similarity_cfg["count"],
        similarity_exemplar_strategy=similarity_cfg["strategy"],
        similarity_exemplar_seed=similarity_cfg["seed"],
        similarity_exemplar_fraction=similarity_cfg["fraction"],
        similarity_exemplar_min=similarity_cfg["min"],
        similarity_exemplar_max=similarity_cfg["max"],
        similarity_exemplar_source_quota=similarity_cfg["source_quota"],
        similarity_window_extension=similarity_window_cfg["enabled"],
        similarity_window_mode=similarity_window_cfg["mode"],
        similarity_window_size=similarity_window_cfg["size"],
        similarity_window_overlap=similarity_window_cfg["overlap"],
        sam3_score_thr=sam3_score_thr,
        sam3_mask_threshold=sam3_mask_threshold,
        detector_conf=detector_conf,
        sahi_window_size=sahi_cfg["size"],
        sahi_overlap_ratio=sahi_cfg["overlap"],
        classifier_id=classifier_id_resolved,
        scoreless_iou=scoreless_iou,
        iou=dedupe_iou,
        fusion_mode=(payload.fusion_mode or "primary"),
        cross_class_dedupe_enabled=cross_class_cfg["enabled"],
        cross_class_dedupe_iou=cross_class_cfg["iou"],
    )

    prepass_config = {
        "sam3_text_synonym_budget": None
        if payload.sam3_text_synonym_budget is None
        else int(payload.sam3_text_synonym_budget),
        "sam3_text_window_extension": sam3_text_window_cfg["enabled"],
        "sam3_text_window_mode": sam3_text_window_cfg["mode"],
        "sam3_text_window_size": sam3_text_window_cfg["size"],
        "sam3_text_window_overlap": sam3_text_window_cfg["overlap"],
        "prepass_sam3_text_thr": prepass_sam3_text_thr,
        "prepass_similarity_score": prepass_similarity_score,
        "similarity_min_exemplar_score": similarity_min_exemplar_score,
        "similarity_exemplar_count": similarity_cfg["count"],
        "similarity_exemplar_strategy": similarity_cfg["strategy"],
        "similarity_exemplar_seed": similarity_cfg["seed"],
        "similarity_exemplar_fraction": similarity_cfg["fraction"],
        "similarity_exemplar_min": similarity_cfg["min"],
        "similarity_exemplar_max": similarity_cfg["max"],
        "similarity_exemplar_source_quota": similarity_cfg["source_quota"],
        "similarity_window_extension": similarity_window_cfg["enabled"],
        "similarity_window_mode": similarity_window_cfg["mode"],
        "similarity_window_size": similarity_window_cfg["size"],
        "similarity_window_overlap": similarity_window_cfg["overlap"],
        "sam3_score_thr": sam3_score_thr,
        "sam3_mask_threshold": sam3_mask_threshold,
        "detector_conf": detector_conf,
        "enable_yolo": use_yolo,
        "enable_rfdetr": use_rfdetr,
        "sahi_window_size": sahi_cfg["size"],
        "sahi_overlap_ratio": sahi_cfg["overlap"],
        "scoreless_iou": scoreless_iou,
        "dedupe_iou": dedupe_iou,
        "fusion_mode": str(payload.fusion_mode or "primary"),
        "cross_class_dedupe_enabled": cross_class_cfg["enabled"],
        "cross_class_dedupe_iou": cross_class_cfg["iou"],
        "yolo_run_id": yolo_run_id,
        "rfdetr_run_id": rfdetr_run_id,
        "yolo_fingerprint": yolo_fingerprint,
        "rfdetr_fingerprint": rfdetr_fingerprint,
    }
    prepass_config_for_hash = _prepass_hash_config_legacy_compatible(prepass_config)
    return prepass_payload, prepass_config, prepass_config_for_hash


def _run_calibration_job(
    job: Any,
    payload: Any,
    *,
    jobs: Dict[str, Any],
    jobs_lock: Any,
    update_fn: Callable[..., None],
    require_sam3_fn: Callable[[bool, bool], None],
    prepare_for_training_fn: Callable[[], None],
    load_yolo_active_fn: Callable[[], Dict[str, Any]],
    load_rfdetr_active_fn: Callable[[], Dict[str, Any]],
    load_labelmap_meta_fn: Callable[[str], Tuple[List[str], str]],
    list_images_fn: Callable[[str], List[str]],
    sample_images_fn: Callable[[List[str]], List[str]],
    calibration_root: Path,
    calibration_cache_root: Path,
    prepass_request_cls: Any,
    active_classifier_head: Any,
    active_classifier_path: Optional[str],
    default_classifier_for_dataset_fn: Optional[Callable[[Optional[str]], Optional[str]]],
    calibration_features_version: int,
    write_record_fn: Callable[[Path, Dict[str, Any]], None],
    hash_payload_fn: Callable[[Dict[str, Any]], str],
    safe_link_fn: Callable[[Path, Path], None],
    prepass_worker_fn: Callable[..., None],
    unload_inference_runtimes_fn: Callable[[], None],
    resolve_dataset_fn: Callable[[str], Path],
    cache_image_fn: Callable[[Image.Image, Optional[str]], str],
    run_prepass_fn: Callable[..., Dict[str, Any]],
    logger: Any,
    http_exception_cls: Any,
    root_dir: Path,
) -> None:
    with jobs_lock:
        jobs[job.job_id] = job
    calibration_root_resolved = _resolve_calibration_storage_root(calibration_root, create=True)
    output_dir = _prepare_calibration_storage_dir(
        calibration_root_resolved / job.job_id,
        root=calibration_root_resolved,
        error_prefix="calibration_job_dir",
    )

    def _persisting_update_fn(inner_job: Any, **kwargs: Any) -> None:
        update_fn(inner_job, **kwargs)
        try:
            _persist_calibration_job_state(inner_job, calibration_root_resolved)
        except Exception:
            pass

    recipe_mode = _resolve_recipe_mode(payload)
    lane_selection = _resolve_lane_selection(payload)
    initial_calibration_model = str(getattr(payload, "calibration_model", "xgb") or "xgb").strip().lower()
    if initial_calibration_model not in {"mlp", "xgb"}:
        initial_calibration_model = "xgb"
    initial_policy_layer_variant = (
        _resolve_policy_layer_variant(payload) if initial_calibration_model == "xgb" else "none"
    )
    step_plan = _build_calibration_step_plan(
        recipe_mode=recipe_mode,
        calibration_model=initial_calibration_model,
        policy_layer_variant=initial_policy_layer_variant,
    )
    _update_calibration_phase(
        job=job,
        update_fn=_persisting_update_fn,
        step_plan=step_plan,
        phase="select_images",
        message="Selecting images…",
        status="running",
        request=model_dump_compat(payload),
    )
    try:
        def _terminate_process(proc: subprocess.Popen[Any]) -> None:
            if proc.poll() is not None:
                return
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass

        def _run_subprocess_cancellable(
            args: List[str],
            *,
            capture_output: bool = False,
            text: bool = False,
            poll_callback: Optional[Callable[[], None]] = None,
        ) -> subprocess.CompletedProcess[Any]:
            if job.cancel_event.is_set():
                raise RuntimeError("cancelled")
            popen_kwargs: Dict[str, Any] = {
                "cwd": str(root_dir),
                "env": _repo_tool_subprocess_env(root_dir),
            }
            if capture_output:
                popen_kwargs.update(
                    {
                        "stdout": subprocess.PIPE,
                        "stderr": subprocess.PIPE,
                        "text": text,
                    }
                )
            proc = subprocess.Popen(args, **popen_kwargs)
            if not capture_output:
                while True:
                    if poll_callback is not None:
                        try:
                            poll_callback()
                        except Exception:
                            pass
                    if job.cancel_event.is_set():
                        _terminate_process(proc)
                        raise RuntimeError("cancelled")
                    rc = proc.poll()
                    if rc is not None:
                        if rc != 0:
                            raise subprocess.CalledProcessError(rc, args)
                        return subprocess.CompletedProcess(args=args, returncode=rc)
                    time.sleep(0.25)
            stdout: Any = None
            stderr: Any = None
            while True:
                if poll_callback is not None:
                    try:
                        poll_callback()
                    except Exception:
                        pass
                if job.cancel_event.is_set():
                    _terminate_process(proc)
                    raise RuntimeError("cancelled")
                try:
                    stdout, stderr = proc.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if proc.returncode:
                raise subprocess.CalledProcessError(
                    proc.returncode,
                    args,
                    output=stdout,
                    stderr=stderr,
                )
            return subprocess.CompletedProcess(
                args=args,
                returncode=int(proc.returncode or 0),
                stdout=stdout,
                stderr=stderr,
            )

        def _run_step(
            phase: str,
            message: str,
            args: List[str],
            *,
            poll_callback: Optional[Callable[[], None]] = None,
            substep_current: int = 0,
            substep_total: int = 0,
            substep_label: str = "",
        ) -> None:
            _update_calibration_phase(
                job=job,
                update_fn=_persisting_update_fn,
                step_plan=step_plan,
                phase=phase,
                message=message,
                substep_current=substep_current,
                substep_total=substep_total,
                substep_label=substep_label,
            )
            _run_subprocess_cancellable(args, poll_callback=poll_callback)

        def _phase_bound_update(
            phase_name: str,
            *,
            substep_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        ) -> Callable[..., None]:
            def _wrapped(inner_job: Any, **kwargs: Any) -> None:
                current_phase = str(phase_name or kwargs.pop("phase", "") or phase_name)
                message = kwargs.pop("message", None)
                kwargs.pop("phase", None)
                substep_payload = {}
                if substep_provider is not None:
                    try:
                        candidate = substep_provider()
                    except Exception:
                        candidate = {}
                    if isinstance(candidate, dict):
                        substep_payload = candidate
                _update_calibration_phase(
                    job=inner_job,
                    update_fn=_persisting_update_fn,
                    step_plan=step_plan,
                    phase=current_phase,
                    message=message,
                    substep_current=int(substep_payload.get("substep_current") or 0),
                    substep_total=int(substep_payload.get("substep_total") or 0),
                    substep_label=str(substep_payload.get("substep_label") or ""),
                    **kwargs,
                )

            return _wrapped

        require_sam3_fn(True, True)
        prepare_for_training_fn()
        yolo_active: Dict[str, Any] = {}
        rfdetr_active: Dict[str, Any] = {}
        if payload.enable_yolo is not False:
            yolo_active = load_yolo_active_fn()
            yolo_best = str(yolo_active.get("best_path") or "")
            if not yolo_best or not Path(yolo_best).exists():
                raise http_exception_cls(status_code=412, detail="yolo_active_missing_weights")
        if payload.enable_rfdetr is not False:
            rfdetr_active = load_rfdetr_active_fn()
            rfdetr_best = str(rfdetr_active.get("best_path") or "")
            if not rfdetr_best or not Path(rfdetr_best).exists():
                raise http_exception_cls(status_code=412, detail="rfdetr_active_missing_weights")
        labelmap, glossary = load_labelmap_meta_fn(payload.dataset_id)
        if not labelmap:
            raise http_exception_cls(status_code=400, detail="calibration_labelmap_missing")
        images = list_images_fn(payload.dataset_id)
        if not images:
            raise http_exception_cls(status_code=404, detail="calibration_images_missing")
        max_images = int(payload.max_images or 0)
        seed = int(payload.seed or 0)
        selected = sample_images_fn(images, max_images=max_images, seed=seed)
        selected_hash = hashlib.sha1(json.dumps(selected).encode("utf-8")).hexdigest()
        total = len(selected)
        _persisting_update_fn(job, total=total, processed=0, progress=0.0)

        prepass_path = output_dir / "prepass.jsonl"
        calibration_cache_root = _resolve_calibration_storage_root(
            calibration_cache_root,
            create=True,
            error_prefix="calibration_cache_root",
        )
        classifier_id_resolved = str(payload.classifier_id or "").strip()
        if not classifier_id_resolved and callable(default_classifier_for_dataset_fn):
            try:
                classifier_id_resolved = str(default_classifier_for_dataset_fn(payload.dataset_id) or "").strip()
            except Exception:
                classifier_id_resolved = ""
        if not classifier_id_resolved:
            classifier_id_resolved = str(active_classifier_path or "").strip()
        if not classifier_id_resolved and isinstance(active_classifier_head, dict):
            raise http_exception_cls(status_code=412, detail="calibration_classifier_id_required")
        if not classifier_id_resolved:
            raise http_exception_cls(status_code=412, detail="calibration_classifier_required")
        classifier_id_resolved = _normalize_classifier_id_for_fingerprint(
            classifier_id_resolved,
            root_dir=root_dir,
        )
        if not (payload.classifier_id or "").strip():
            try:
                req = dict(job.request or {})
                req["classifier_id_resolved"] = classifier_id_resolved
                job.request = req
                _persist_calibration_job_state(job, calibration_root_resolved)
            except Exception:
                pass
        use_yolo = payload.enable_yolo is not False
        use_rfdetr = payload.enable_rfdetr is not False
        yolo_fingerprint = _detector_fingerprint(yolo_active) if use_yolo else None
        rfdetr_fingerprint = _detector_fingerprint(rfdetr_active) if use_rfdetr else None
        yolo_run_id = (yolo_fingerprint or {}).get("run_id")
        rfdetr_run_id = (rfdetr_fingerprint or {}).get("run_id")

        label_iou = _float_with_default(payload.label_iou, 0.5)
        support_iou = _float_with_default(payload.support_iou, 0.5)
        context_radius = _float_with_default(payload.context_radius, 0.075)
        eval_iou = _float_with_default(payload.eval_iou, 0.5)

        similarity_cfg = _canonical_similarity_settings(payload)
        cross_class_cfg = _canonical_cross_class_dedupe_settings(payload)
        sam3_text_window_cfg = _canonical_sam3_text_window_settings(payload)
        similarity_window_cfg = _canonical_similarity_window_settings(payload)
        sahi_cfg = _canonical_sahi_settings(payload)
        prepass_sam3_text_thr = _float_with_default(payload.prepass_sam3_text_thr, 0.2)
        prepass_similarity_score = _float_with_default(payload.prepass_similarity_score, 0.3)
        similarity_min_exemplar_score = _float_with_default(payload.similarity_min_exemplar_score, 0.6)
        sam3_score_thr = _float_with_default(payload.sam3_score_thr, 0.2)
        sam3_mask_threshold = _float_with_default(payload.sam3_mask_threshold, 0.2)
        detector_conf = _float_with_default(payload.detector_conf, 0.45)
        scoreless_iou = _float_with_default(payload.scoreless_iou, 0.0)
        dedupe_iou = _float_with_default(payload.dedupe_iou, 0.75)

        default_ensemble_policy_json = _resolve_default_ensemble_policy_json(
            payload,
            sam3_text_window_cfg=sam3_text_window_cfg,
            similarity_window_cfg=similarity_window_cfg,
        )
        window_discovery_payload = payload.copy(
            update={
                "sam3_text_window_extension": True,
                "similarity_window_extension": True,
            },
            deep=True,
        )
        window_discovery_similarity_cfg = _canonical_similarity_settings(window_discovery_payload)
        window_discovery_cross_class_cfg = _canonical_cross_class_dedupe_settings(window_discovery_payload)
        window_discovery_text_window_cfg = _canonical_sam3_text_window_settings(window_discovery_payload)
        window_discovery_similarity_window_cfg = _canonical_similarity_window_settings(window_discovery_payload)
        window_discovery_sahi_cfg = _canonical_sahi_settings(window_discovery_payload)
        (
            window_discovery_prepass_payload,
            window_discovery_prepass_config,
            window_discovery_prepass_hash_cfg,
        ) = _build_prepass_request_and_config(
            payload=window_discovery_payload,
            dataset_id=payload.dataset_id,
            classifier_id_resolved=classifier_id_resolved,
            use_yolo=use_yolo,
            use_rfdetr=use_rfdetr,
            yolo_run_id=yolo_run_id,
            rfdetr_run_id=rfdetr_run_id,
            similarity_cfg=window_discovery_similarity_cfg,
            cross_class_cfg=window_discovery_cross_class_cfg,
            sam3_text_window_cfg=window_discovery_text_window_cfg,
            similarity_window_cfg=window_discovery_similarity_window_cfg,
            sahi_cfg=window_discovery_sahi_cfg,
            prepass_request_cls=prepass_request_cls,
            detector_conf=detector_conf,
            sam3_score_thr=sam3_score_thr,
            sam3_mask_threshold=sam3_mask_threshold,
            prepass_sam3_text_thr=prepass_sam3_text_thr,
            prepass_similarity_score=prepass_similarity_score,
            similarity_min_exemplar_score=similarity_min_exemplar_score,
            scoreless_iou=scoreless_iou,
            dedupe_iou=dedupe_iou,
            yolo_fingerprint=yolo_fingerprint,
            rfdetr_fingerprint=rfdetr_fingerprint,
        )
        nonwindow_discovery_payload = payload.copy(
            update={
                "sam3_text_window_extension": False,
                "similarity_window_extension": False,
            },
            deep=True,
        )
        nonwindow_discovery_similarity_cfg = _canonical_similarity_settings(nonwindow_discovery_payload)
        nonwindow_discovery_cross_class_cfg = _canonical_cross_class_dedupe_settings(nonwindow_discovery_payload)
        nonwindow_discovery_text_window_cfg = _canonical_sam3_text_window_settings(nonwindow_discovery_payload)
        nonwindow_discovery_similarity_window_cfg = _canonical_similarity_window_settings(nonwindow_discovery_payload)
        nonwindow_discovery_sahi_cfg = _canonical_sahi_settings(nonwindow_discovery_payload)
        (
            nonwindow_discovery_prepass_payload,
            nonwindow_discovery_prepass_config,
            nonwindow_discovery_prepass_hash_cfg,
        ) = _build_prepass_request_and_config(
            payload=nonwindow_discovery_payload,
            dataset_id=payload.dataset_id,
            classifier_id_resolved=classifier_id_resolved,
            use_yolo=use_yolo,
            use_rfdetr=use_rfdetr,
            yolo_run_id=yolo_run_id,
            rfdetr_run_id=rfdetr_run_id,
            similarity_cfg=nonwindow_discovery_similarity_cfg,
            cross_class_cfg=nonwindow_discovery_cross_class_cfg,
            sam3_text_window_cfg=nonwindow_discovery_text_window_cfg,
            similarity_window_cfg=nonwindow_discovery_similarity_window_cfg,
            sahi_cfg=nonwindow_discovery_sahi_cfg,
            prepass_request_cls=prepass_request_cls,
            detector_conf=detector_conf,
            sam3_score_thr=sam3_score_thr,
            sam3_mask_threshold=sam3_mask_threshold,
            prepass_sam3_text_thr=prepass_sam3_text_thr,
            prepass_similarity_score=prepass_similarity_score,
            similarity_min_exemplar_score=similarity_min_exemplar_score,
            scoreless_iou=scoreless_iou,
            dedupe_iou=dedupe_iou,
            yolo_fingerprint=yolo_fingerprint,
            rfdetr_fingerprint=rfdetr_fingerprint,
        )

        labelmap_hash = hashlib.sha1(",".join(labelmap).encode("utf-8")).hexdigest()
        glossary_hash = hashlib.sha1((glossary or "").encode("utf-8")).hexdigest()
        recipe_fingerprint_payload = build_recipe_fingerprint_payload(
            dataset_id=payload.dataset_id,
            labelmap_hash=labelmap_hash,
            glossary_hash=glossary_hash,
            classifier_id=classifier_id_resolved,
            lane_selection=lane_selection,
            prepass_config=(
                {"window": window_discovery_prepass_hash_cfg, "nonwindow": nonwindow_discovery_prepass_hash_cfg}
                if lane_selection == "compare_both"
                else (
                    {"window": window_discovery_prepass_hash_cfg}
                    if lane_selection == "window"
                    else {"nonwindow": nonwindow_discovery_prepass_hash_cfg}
                )
            ),
            selected_hash=selected_hash,
            selected_count=total,
            selection_seed=seed,
            requested_max_images=max_images,
            support_iou=support_iou,
            context_radius=context_radius,
            label_iou=label_iou,
            eval_iou=eval_iou,
            feature_version=calibration_features_version,
            recipe_defaults_version=CALIBRATION_RECIPE_DEFAULTS_VERSION,
        )
        recipe_fingerprint = build_recipe_fingerprint(recipe_fingerprint_payload)
        try:
            req = dict(job.request or {})
            req["recipe_mode"] = recipe_mode
            req["lane_selection"] = lane_selection
            req["recipe_fingerprint"] = recipe_fingerprint
            job.request = req
        except Exception:
            pass
        completion_context_payload = build_canonical_completion_context(
            dataset_id=payload.dataset_id,
            recipe_fingerprint=recipe_fingerprint,
            recipe_fingerprint_payload=recipe_fingerprint_payload,
            calibration_request=dict(job.request or {}),
            resolved_classifier_id=classifier_id_resolved,
            glossary_text=glossary or "",
            labelmap=list(labelmap or []),
        )

        canonical_recipe_payload: Optional[Dict[str, Any]] = None
        recipe_registry_entry: Optional[Dict[str, Any]] = None
        saved_prepass_recipe_entry: Optional[Dict[str, Any]] = None
        canonical_deployment_entry: Optional[Dict[str, Any]] = None
        edr_package_entry: Optional[Dict[str, Any]] = None
        recipe_reused = False
        recipe_discovered = False
        discovery_run_root: Optional[Path] = None
        last_discovery_progress_signature = ""

        def _read_recipe_discovery_substep() -> Dict[str, Any]:
            nonlocal last_discovery_progress_signature
            if discovery_run_root is None:
                return {}
            progress_path = discovery_run_root / "canonical_discovery_progress.json"
            if not progress_path.exists():
                return {}
            try:
                payload = json.loads(progress_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
            if not isinstance(payload, dict):
                return {}
            stage_index = max(0, int(payload.get("stage_index") or 0))
            stage_total = max(0, int(payload.get("stage_total") or 0))
            stage_label = str(payload.get("stage_label") or payload.get("stage_key") or "").strip()
            stage_message = str(payload.get("message") or "").strip()
            signature = json.dumps(
                {
                    "stage_index": stage_index,
                    "stage_total": stage_total,
                    "stage_label": stage_label,
                    "stage_message": stage_message,
                },
                sort_keys=True,
            )
            if signature == last_discovery_progress_signature:
                return {
                    "substep_current": stage_index,
                    "substep_total": stage_total,
                    "substep_label": stage_label,
                }
            last_discovery_progress_signature = signature
            return {
                "substep_current": stage_index,
                "substep_total": stage_total,
                "substep_label": stage_label,
            }

        _update_calibration_phase(
            job=job,
            update_fn=_persisting_update_fn,
            step_plan=step_plan,
            phase="fingerprint",
            message="Computing EDR fingerprint…",
        )
        if recipe_mode != "force_rediscover":
            _update_calibration_phase(
                job=job,
                update_fn=_persisting_update_fn,
                step_plan=step_plan,
                phase="recipe_lookup",
                message="Looking up promoted EDR…",
            )
            recipe_registry_entry = find_matching_recipe(calibration_cache_root, recipe_fingerprint)
            if recipe_registry_entry:
                recipe_path = Path(str(recipe_registry_entry.get("canonical_recipe_json") or ""))
                if recipe_path.exists():
                    canonical_recipe_payload = json.loads(recipe_path.read_text(encoding="utf-8"))
                    recipe_reused = True
                    step_plan = _build_calibration_step_plan(
                        recipe_mode="reuse_only",
                        calibration_model=initial_calibration_model,
                        policy_layer_variant=initial_policy_layer_variant,
                    )
                    _update_calibration_phase(
                        job=job,
                        update_fn=_persisting_update_fn,
                        step_plan=step_plan,
                        phase="recipe_lookup",
                        message="Promoted EDR reused.",
                    )
                elif recipe_mode == "reuse_only":
                    raise http_exception_cls(status_code=412, detail="calibration_recipe_artifact_missing")
                else:
                    recipe_registry_entry = None
        if canonical_recipe_payload is None and recipe_mode == "reuse_only":
            raise http_exception_cls(status_code=412, detail="calibration_recipe_missing_for_fingerprint")

        if canonical_recipe_payload is None:
            discovery_run_root = discovery_runs_root(calibration_cache_root) / recipe_fingerprint
            discovery_inputs_dir = discovery_run_root / "inputs"
            write_canonical_completion_context(discovery_run_root, completion_context_payload)
            _update_calibration_phase(
                job=job,
                update_fn=_persisting_update_fn,
                step_plan=step_plan,
                phase="recipe_discovery",
                message="Waiting for EDR discovery lock…",
            )
            with discovery_lock(calibration_cache_root, recipe_fingerprint):
                if recipe_mode != "force_rediscover":
                    recipe_registry_entry = find_matching_recipe(calibration_cache_root, recipe_fingerprint)
                    if recipe_registry_entry:
                        recipe_path = Path(str(recipe_registry_entry.get("canonical_recipe_json") or ""))
                        if recipe_path.exists():
                            canonical_recipe_payload = json.loads(recipe_path.read_text(encoding="utf-8"))
                            recipe_reused = True
                canonical_recipe_path = _canonical_recipe_json_path(discovery_run_root)
                if canonical_recipe_payload is None and recipe_mode != "force_rediscover" and canonical_recipe_path.exists():
                    canonical_recipe_payload = json.loads(canonical_recipe_path.read_text(encoding="utf-8"))
                    completion_summary = persist_canonical_edr_completion(
                        calibration_cache_root=calibration_cache_root,
                        run_root=discovery_run_root,
                        canonical_recipe_json=canonical_recipe_path,
                        canonical_recipe_md=_canonical_recipe_md_path(discovery_run_root),
                        canonical_recipe_payload=canonical_recipe_payload,
                        completion_context=completion_context_payload,
                        report_bundle_json=discovery_run_root / "report_bundle.json",
                    )
                    recipe_registry_entry = (
                        completion_summary.get("recipe_registry_entry")
                        if isinstance(completion_summary.get("recipe_registry_entry"), dict)
                        else None
                    )
                    saved_prepass_recipe_entry = (
                        completion_summary.get("saved_prepass_recipe")
                        if isinstance(completion_summary.get("saved_prepass_recipe"), dict)
                        else None
                    )
                    canonical_deployment_entry = (
                        completion_summary.get("canonical_deployment_job")
                        if isinstance(completion_summary.get("canonical_deployment_job"), dict)
                        else None
                    )
                    edr_package_entry = (
                        completion_summary.get("edr_package")
                        if isinstance(completion_summary.get("edr_package"), dict)
                        else None
                    )
                    recipe_reused = True

                if canonical_recipe_payload is None:
                    discovery_update_fn = _phase_bound_update("recipe_discovery")
                    _update_calibration_phase(
                        job=job,
                        update_fn=_persisting_update_fn,
                        step_plan=step_plan,
                        phase="recipe_discovery",
                        message="Preparing discovery inputs…",
                    )
                    window_prepass: Optional[Dict[str, Any]] = None
                    if lane_selection in {"window", "compare_both"}:
                        window_prepass = _ensure_prepass_jsonl(
                            job=job,
                            update_fn=discovery_update_fn,
                            selected=selected,
                            total=total,
                            dataset_id=payload.dataset_id,
                            labelmap=labelmap,
                            glossary=glossary,
                            prepass_payload=window_discovery_prepass_payload,
                            prepass_config=window_discovery_prepass_config,
                            prepass_config_for_hash=window_discovery_prepass_hash_cfg,
                            output_path=discovery_inputs_dir / "window_prepass.jsonl",
                            calibration_cache_root=calibration_cache_root,
                            write_record_fn=write_record_fn,
                            hash_payload_fn=hash_payload_fn,
                            prepass_worker_fn=prepass_worker_fn,
                            unload_inference_runtimes_fn=unload_inference_runtimes_fn,
                            resolve_dataset_fn=resolve_dataset_fn,
                            cache_image_fn=cache_image_fn,
                            run_prepass_fn=run_prepass_fn,
                        )
                    nonwindow_prepass: Optional[Dict[str, Any]] = None
                    if lane_selection in {"nonwindow", "compare_both"}:
                        nonwindow_prepass = _ensure_prepass_jsonl(
                            job=job,
                            update_fn=discovery_update_fn,
                            selected=selected,
                            total=total,
                            dataset_id=payload.dataset_id,
                            labelmap=labelmap,
                            glossary=glossary,
                            prepass_payload=nonwindow_discovery_prepass_payload,
                            prepass_config=nonwindow_discovery_prepass_config,
                            prepass_config_for_hash=nonwindow_discovery_prepass_hash_cfg,
                            output_path=discovery_inputs_dir / "nonwindow_prepass.jsonl",
                            calibration_cache_root=calibration_cache_root,
                            write_record_fn=write_record_fn,
                            hash_payload_fn=hash_payload_fn,
                            prepass_worker_fn=prepass_worker_fn,
                            unload_inference_runtimes_fn=unload_inference_runtimes_fn,
                            resolve_dataset_fn=resolve_dataset_fn,
                            cache_image_fn=cache_image_fn,
                            run_prepass_fn=run_prepass_fn,
                        )
                    discovery_cmd = [
                        sys.executable,
                        str(root_dir / "tools" / "run_canonical_prepass_discovery.py"),
                        "--run-root",
                        str(discovery_run_root),
                        "--dataset",
                        payload.dataset_id,
                        "--classifier-id",
                        classifier_id_resolved,
                        "--lane-selection",
                        lane_selection,
                    ]
                    if window_prepass is not None:
                        discovery_cmd += ["--window-prepass-jsonl", str(window_prepass["prepass_path"])]
                    if nonwindow_prepass is not None:
                        discovery_cmd += ["--nonwindow-prepass-jsonl", str(nonwindow_prepass["prepass_path"])]
                    if recipe_mode == "force_rediscover":
                        discovery_cmd.append("--force")
                    _run_step(
                        "recipe_discovery",
                        "Discovering canonical EDR…",
                        discovery_cmd,
                        poll_callback=lambda: _update_calibration_phase(
                            job=job,
                            update_fn=_persisting_update_fn,
                            step_plan=step_plan,
                            phase="recipe_discovery",
                            message="Discovering canonical EDR…",
                            **_read_recipe_discovery_substep(),
                        ),
                    )
                    if not canonical_recipe_path.exists():
                        raise RuntimeError("canonical_recipe_missing_after_discovery")
                    canonical_recipe_payload = json.loads(canonical_recipe_path.read_text(encoding="utf-8"))
                    completion_summary = persist_canonical_edr_completion(
                        calibration_cache_root=calibration_cache_root,
                        run_root=discovery_run_root,
                        canonical_recipe_json=canonical_recipe_path,
                        canonical_recipe_md=_canonical_recipe_md_path(discovery_run_root),
                        canonical_recipe_payload=canonical_recipe_payload,
                        completion_context=completion_context_payload,
                        report_bundle_json=discovery_run_root / "report_bundle.json",
                    )
                    recipe_registry_entry = (
                        completion_summary.get("recipe_registry_entry")
                        if isinstance(completion_summary.get("recipe_registry_entry"), dict)
                        else None
                    )
                    saved_prepass_recipe_entry = (
                        completion_summary.get("saved_prepass_recipe")
                        if isinstance(completion_summary.get("saved_prepass_recipe"), dict)
                        else None
                    )
                    canonical_deployment_entry = (
                        completion_summary.get("canonical_deployment_job")
                        if isinstance(completion_summary.get("canonical_deployment_job"), dict)
                        else None
                    )
                    edr_package_entry = (
                        completion_summary.get("edr_package")
                        if isinstance(completion_summary.get("edr_package"), dict)
                        else None
                    )
                    recipe_discovered = True

        if isinstance(canonical_recipe_payload, dict) and saved_prepass_recipe_entry is None:
            candidate_recipe_json = None
            candidate_recipe_md = None
            candidate_report_bundle = None
            if isinstance(recipe_registry_entry, dict):
                raw_json = recipe_registry_entry.get("canonical_recipe_json")
                raw_md = recipe_registry_entry.get("canonical_recipe_md")
                raw_report = recipe_registry_entry.get("report_bundle_json")
                candidate_recipe_json = Path(str(raw_json)).resolve() if raw_json else None
                candidate_recipe_md = Path(str(raw_md)).resolve() if raw_md else None
                candidate_report_bundle = Path(str(raw_report)).resolve() if raw_report else None
            if candidate_recipe_json is None and discovery_run_root is not None:
                candidate_recipe_json = _canonical_recipe_json_path(discovery_run_root)
                candidate_recipe_md = _canonical_recipe_md_path(discovery_run_root)
                report_path = discovery_run_root / "report_bundle.json"
                candidate_report_bundle = report_path if report_path.exists() else None
            if candidate_recipe_json is not None and candidate_recipe_json.exists():
                completion_summary = persist_canonical_edr_completion(
                    calibration_cache_root=calibration_cache_root,
                    run_root=discovery_run_root if discovery_run_root is not None else None,
                    canonical_recipe_json=candidate_recipe_json,
                    canonical_recipe_md=candidate_recipe_md if candidate_recipe_md and candidate_recipe_md.exists() else None,
                    canonical_recipe_payload=canonical_recipe_payload,
                    completion_context=completion_context_payload,
                    existing_registry_entry=recipe_registry_entry,
                    report_bundle_json=(
                        candidate_report_bundle
                        if candidate_report_bundle is not None and candidate_report_bundle.exists()
                        else None
                    ),
                )
                recipe_registry_entry = (
                    completion_summary.get("recipe_registry_entry")
                    if isinstance(completion_summary.get("recipe_registry_entry"), dict)
                    else recipe_registry_entry
                )
                saved_prepass_recipe_entry = (
                    completion_summary.get("saved_prepass_recipe")
                    if isinstance(completion_summary.get("saved_prepass_recipe"), dict)
                    else None
                )
                canonical_deployment_entry = (
                    completion_summary.get("canonical_deployment_job")
                    if isinstance(completion_summary.get("canonical_deployment_job"), dict)
                    else canonical_deployment_entry
                )
                edr_package_entry = (
                    completion_summary.get("edr_package")
                    if isinstance(completion_summary.get("edr_package"), dict)
                    else edr_package_entry
                )

        if not isinstance(canonical_recipe_payload, dict):
            raise RuntimeError("canonical_recipe_invalid")
        lane_family, effective_recipe = _select_effective_canonical_recipe(
            canonical_recipe_payload,
            lane_selection=lane_selection,
            fallback_windowed=_use_windowed_calibration_defaults(
                sam3_text_window_cfg=sam3_text_window_cfg,
                similarity_window_cfg=similarity_window_cfg,
            ),
        )
        payload = _apply_canonical_recipe_to_payload(payload, effective_recipe)
        calibration_model = str(getattr(payload, "calibration_model", initial_calibration_model) or initial_calibration_model).strip().lower()
        if calibration_model not in {"mlp", "xgb"}:
            calibration_model = initial_calibration_model
        policy_layer_variant = _resolve_policy_layer_variant(payload) if calibration_model == "xgb" else "none"
        step_plan = _build_calibration_step_plan(
            recipe_mode=("reuse_only" if recipe_reused and not recipe_discovered else recipe_mode),
            calibration_model=calibration_model,
            policy_layer_variant=policy_layer_variant,
        )
        default_ensemble_policy_json = _resolve_default_ensemble_policy_json(
            payload,
            sam3_text_window_cfg=_canonical_sam3_text_window_settings(payload),
            similarity_window_cfg=_canonical_similarity_window_settings(payload),
        )
        similarity_cfg = _canonical_similarity_settings(payload)
        cross_class_cfg = _canonical_cross_class_dedupe_settings(payload)
        sam3_text_window_cfg = _canonical_sam3_text_window_settings(payload)
        similarity_window_cfg = _canonical_similarity_window_settings(payload)
        sahi_cfg = _canonical_sahi_settings(payload)
        prepass_sam3_text_thr = _float_with_default(payload.prepass_sam3_text_thr, 0.2)
        prepass_similarity_score = _float_with_default(payload.prepass_similarity_score, 0.3)
        similarity_min_exemplar_score = _float_with_default(payload.similarity_min_exemplar_score, 0.6)
        sam3_score_thr = _float_with_default(payload.sam3_score_thr, 0.2)
        sam3_mask_threshold = _float_with_default(payload.sam3_mask_threshold, 0.2)
        detector_conf = _float_with_default(payload.detector_conf, 0.45)
        scoreless_iou = _float_with_default(payload.scoreless_iou, 0.0)
        dedupe_iou = _float_with_default(payload.dedupe_iou, 0.75)
        prepass_payload, prepass_config, prepass_config_for_hash = _build_prepass_request_and_config(
            payload=payload,
            dataset_id=payload.dataset_id,
            classifier_id_resolved=classifier_id_resolved,
            use_yolo=use_yolo,
            use_rfdetr=use_rfdetr,
            yolo_run_id=yolo_run_id,
            rfdetr_run_id=rfdetr_run_id,
            similarity_cfg=similarity_cfg,
            cross_class_cfg=cross_class_cfg,
            sam3_text_window_cfg=sam3_text_window_cfg,
            similarity_window_cfg=similarity_window_cfg,
            sahi_cfg=sahi_cfg,
            prepass_request_cls=prepass_request_cls,
            detector_conf=detector_conf,
            sam3_score_thr=sam3_score_thr,
            sam3_mask_threshold=sam3_mask_threshold,
            prepass_sam3_text_thr=prepass_sam3_text_thr,
            prepass_similarity_score=prepass_similarity_score,
            similarity_min_exemplar_score=similarity_min_exemplar_score,
            scoreless_iou=scoreless_iou,
            dedupe_iou=dedupe_iou,
            yolo_fingerprint=yolo_fingerprint,
            rfdetr_fingerprint=rfdetr_fingerprint,
        )
        prepass_result = _ensure_prepass_jsonl(
            job=job,
            update_fn=_phase_bound_update("prepass"),
            selected=selected,
            total=total,
            dataset_id=payload.dataset_id,
            labelmap=labelmap,
            glossary=glossary,
            prepass_payload=prepass_payload,
            prepass_config=prepass_config,
            prepass_config_for_hash=prepass_config_for_hash,
            output_path=prepass_path,
            calibration_cache_root=calibration_cache_root,
            write_record_fn=write_record_fn,
            hash_payload_fn=hash_payload_fn,
            prepass_worker_fn=prepass_worker_fn,
            unload_inference_runtimes_fn=unload_inference_runtimes_fn,
            resolve_dataset_fn=resolve_dataset_fn,
            cache_image_fn=cache_image_fn,
            run_prepass_fn=run_prepass_fn,
        )
        prepass_key = str(prepass_result["prepass_key"])
        if job.cancel_event.is_set():
            raise RuntimeError("cancelled")

        label_iou = _float_with_default(payload.label_iou, 0.5)
        support_iou = _float_with_default(payload.support_iou, 0.5)
        context_radius = _float_with_default(payload.context_radius, 0.075)
        embed_proj_dim = DEFAULT_EMBED_PROJ_DIM
        embed_proj_seed = 42
        image_embed_proj_dim = max(0, int(getattr(payload, "image_embed_proj_dim", 0) or 0))
        image_embed_proj_seed = int(
            getattr(payload, "image_embed_proj_seed", DEFAULT_IMAGE_EMBED_PROJ_SEED)
            or DEFAULT_IMAGE_EMBED_PROJ_SEED
        )
        embed_l2_normalize = True
        base_fp_ratio = _float_with_default(payload.base_fp_ratio, 0.1)
        relax_fp_ratio = _float_with_default(payload.relax_fp_ratio, 0.2)
        target_fp_ratio = _first_float_with_default(
            payload.relax_fp_ratio,
            payload.base_fp_ratio,
            default=0.2,
        )
        recall_floor = _float_with_default(payload.recall_floor, 0.6)
        eval_iou = _float_with_default(payload.eval_iou, 0.5)

        features_path = output_dir / "ensemble_features.npz"
        labeled_path = output_dir / f"ensemble_features_iou{label_iou:.2f}.npz"
        calibration_model = (payload.calibration_model or "xgb").strip().lower()
        if calibration_model not in {"mlp", "xgb"}:
            calibration_model = "xgb"
        model_prefix = output_dir / ("ensemble_xgb" if calibration_model == "xgb" else "ensemble_mlp")
        meta_path = Path(str(model_prefix) + ".meta.json")
        eval_path = output_dir / (f"{model_prefix.name}.eval.json")

        classifier_id = classifier_id_resolved
        features_key = hash_payload_fn(
            {
                "prepass_key": prepass_key,
                "classifier_id": classifier_id or "",
                "support_iou": support_iou,
                "min_crop_size": 4,
                "context_radius": context_radius,
                "embed_proj_dim": embed_proj_dim,
                "embed_proj_seed": embed_proj_seed,
                "image_embed_proj_dim": image_embed_proj_dim,
                "image_embed_proj_seed": image_embed_proj_seed,
                "embed_l2_normalize": embed_l2_normalize,
                "features_version": calibration_features_version,
            }
        )
        features_cache_dir = calibration_cache_root / "features" / features_key
        _prepare_calibration_storage_dir(
            features_cache_dir,
            root=calibration_cache_root,
            error_prefix="calibration_features_cache_dir",
        )
        features_cache_path = features_cache_dir / "ensemble_features.npz"
        features_cache_meta = features_cache_dir / "features.meta.json"
        cached_features = features_cache_path.exists()
        if cached_features:
            try:
                _validate_classifier_feature_matrix(features_cache_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Invalid cached classifier features for key=%s (%s); rebuilding.",
                    features_key,
                    exc,
                )
                try:
                    features_cache_path.unlink(missing_ok=True)
                except Exception:
                    pass
                cached_features = False

        if cached_features:
            _update_calibration_phase(
                job=job,
                update_fn=update_fn,
                step_plan=step_plan,
                phase="features",
                message="Using cached features…",
                progress=1.0,
            )
            safe_link_fn(features_cache_path, features_path)
        else:
            _run_step(
                "features",
                "Building features…",
                [
                    sys.executable,
                    str(root_dir / "tools" / "build_ensemble_features.py"),
                    "--input",
                    str(prepass_path),
                    "--dataset",
                    payload.dataset_id,
                    "--output",
                    str(features_cache_path),
                    "--support-iou",
                    str(support_iou),
                    "--min-crop-size",
                    "4",
                    "--context-radius",
                    str(context_radius),
                    "--embed-proj-dim",
                    str(int(embed_proj_dim)),
                    "--embed-proj-seed",
                    str(int(embed_proj_seed)),
                    "--image-embed-proj-dim",
                    str(int(image_embed_proj_dim)),
                    "--image-embed-proj-seed",
                    str(int(image_embed_proj_seed)),
                    "--device",
                    "cuda",
                    "--require-classifier",
                ]
                + ([] if embed_l2_normalize else ["--embed-no-l2-normalize"])
                + (["--classifier-id", classifier_id] if classifier_id else []),
            )
            try:
                _validate_classifier_feature_matrix(features_cache_path)
            except Exception as exc:  # noqa: BLE001
                raise http_exception_cls(
                    status_code=412,
                    detail=f"calibration_classifier_features_invalid:{exc}",
                ) from exc
            _write_json_atomic(
                features_cache_meta,
                {
                    "prepass_key": prepass_key,
                    "features_key": features_key,
                    "features_version": calibration_features_version,
                    "embed_proj_dim": int(embed_proj_dim),
                    "embed_proj_seed": int(embed_proj_seed),
                    "image_embed_proj_dim": int(image_embed_proj_dim),
                    "image_embed_proj_seed": int(image_embed_proj_seed),
                    "embed_l2_normalize": bool(embed_l2_normalize),
                },
            )
            safe_link_fn(features_cache_path, features_path)

        labeled_key = hash_payload_fn(
            {
                "features_key": features_key,
                "label_iou": label_iou,
            }
        )
        labeled_cache_dir = calibration_cache_root / "labeled" / labeled_key
        _prepare_calibration_storage_dir(
            labeled_cache_dir,
            root=calibration_cache_root,
            error_prefix="calibration_labeled_cache_dir",
        )
        labeled_cache_path = labeled_cache_dir / f"ensemble_features_iou{label_iou:.2f}.npz"
        labeled_cache_meta = labeled_cache_dir / "labeled.meta.json"
        cached_labeled = labeled_cache_path.exists()

        if cached_labeled:
            _update_calibration_phase(
                job=job,
                update_fn=update_fn,
                step_plan=step_plan,
                phase="labeling",
                message="Using cached labels…",
                progress=1.0,
            )
            safe_link_fn(labeled_cache_path, labeled_path)
        else:
            _run_step(
                "labeling",
                "Labeling candidates…",
                [
                    sys.executable,
                    str(root_dir / "tools" / "label_candidates_iou90.py"),
                    "--input",
                    str(features_path),
                    "--dataset",
                    payload.dataset_id,
                    "--output",
                    str(labeled_cache_path),
                    "--iou",
                    str(label_iou),
                ],
            )
            _write_json_atomic(
                labeled_cache_meta,
                {
                    "features_key": features_key,
                    "labeled_key": labeled_key,
                    "label_iou": label_iou,
                },
            )
            safe_link_fn(labeled_cache_path, labeled_path)
        if calibration_model == "xgb":
            optimize_metric = (payload.optimize_metric or "f1").strip().lower()
            if optimize_metric not in {"f1", "recall", "tp"}:
                optimize_metric = "f1"
            steps_val = int(payload.threshold_steps or 200)
            steps_val = max(20, min(400, steps_val))
            split_head_by_support = _resolve_default_split_head_by_support(
                payload,
                sam3_text_window_cfg=sam3_text_window_cfg,
                similarity_window_cfg=similarity_window_cfg,
            )
            train_sam3_text_quality = bool(getattr(payload, "train_sam3_text_quality", True))
            sam3_text_quality_alpha = _resolve_default_sam3_text_quality_alpha(
                payload,
                sam3_text_window_cfg=sam3_text_window_cfg,
                similarity_window_cfg=similarity_window_cfg,
            )
            train_sam3_similarity_quality = _resolve_default_train_sam3_similarity_quality(
                payload,
                sam3_text_window_cfg=sam3_text_window_cfg,
                similarity_window_cfg=similarity_window_cfg,
            )
            sam3_similarity_quality_alpha = _resolve_default_sam3_similarity_quality_alpha(
                payload,
                sam3_text_window_cfg=sam3_text_window_cfg,
                similarity_window_cfg=similarity_window_cfg,
            )
            target_fp_ratio_by_label_json = str(
                getattr(payload, "target_fp_ratio_by_label_json", "") or ""
            ).strip()
            min_recall_by_label_json = str(
                getattr(payload, "min_recall_by_label_json", "") or ""
            ).strip()
            train_cmd = [
                sys.executable,
                str(root_dir / "tools" / "train_ensemble_xgb.py"),
                "--input",
                str(labeled_path),
                "--output",
                str(model_prefix),
                "--seed",
                str(int(payload.model_seed or 42)),
                "--target-fp-ratio",
                str(base_fp_ratio),
                "--min-recall",
                str(recall_floor),
                "--threshold-steps",
                str(steps_val),
                "--optimize",
                optimize_metric,
            ]
            if payload.per_class_thresholds is not False:
                train_cmd.append("--per-class")
            if split_head_by_support:
                train_cmd.append("--split-head-by-support")
            if train_sam3_text_quality:
                train_cmd += [
                    "--train-sam3-text-quality",
                    "--sam3-text-quality-alpha",
                    str(sam3_text_quality_alpha),
                ]
            if train_sam3_similarity_quality:
                train_cmd += [
                    "--train-sam3-similarity-quality",
                    "--sam3-similarity-quality-alpha",
                    str(sam3_similarity_quality_alpha),
                ]
            if default_ensemble_policy_json:
                train_cmd += [
                    "--policy-json",
                    default_ensemble_policy_json,
                ]
            if payload.xgb_max_depth is not None:
                train_cmd += ["--max-depth", str(int(payload.xgb_max_depth))]
            if payload.xgb_n_estimators is not None:
                train_cmd += ["--n-estimators", str(int(payload.xgb_n_estimators))]
            if payload.xgb_learning_rate is not None:
                train_cmd += ["--learning-rate", str(float(payload.xgb_learning_rate))]
            if payload.xgb_subsample is not None:
                train_cmd += ["--subsample", str(float(payload.xgb_subsample))]
            if payload.xgb_colsample_bytree is not None:
                train_cmd += ["--colsample-bytree", str(float(payload.xgb_colsample_bytree))]
            if payload.xgb_min_child_weight is not None:
                train_cmd += ["--min-child-weight", str(float(payload.xgb_min_child_weight))]
            if payload.xgb_gamma is not None:
                train_cmd += ["--gamma", str(float(payload.xgb_gamma))]
            if payload.xgb_reg_lambda is not None:
                train_cmd += ["--reg-lambda", str(float(payload.xgb_reg_lambda))]
            if payload.xgb_reg_alpha is not None:
                train_cmd += ["--reg-alpha", str(float(payload.xgb_reg_alpha))]
            if payload.xgb_scale_pos_weight is not None:
                train_cmd += ["--scale-pos-weight", str(float(payload.xgb_scale_pos_weight))]
            if payload.xgb_tree_method:
                train_cmd += ["--tree-method", str(payload.xgb_tree_method)]
            if payload.xgb_max_bin is not None:
                train_cmd += ["--max-bin", str(int(payload.xgb_max_bin))]
            if payload.xgb_early_stopping_rounds is not None:
                train_cmd += ["--early-stopping-rounds", str(int(payload.xgb_early_stopping_rounds))]
            if payload.xgb_log1p_counts:
                train_cmd.append("--log1p-counts")
            if payload.xgb_standardize:
                train_cmd.append("--standardize")
            _run_step("train", "Training XGBoost…", train_cmd)
            _run_step(
                "relax",
                "Relaxing thresholds…",
                [
                    sys.executable,
                    str(root_dir / "tools" / "relax_ensemble_thresholds_xgb.py"),
                    "--model",
                    str(Path(str(model_prefix) + ".json")),
                    "--data",
                    str(labeled_path),
                    "--meta",
                    str(meta_path),
                    "--fp-ratio-cap",
                    str(relax_fp_ratio),
                ],
            )
            tune_cmd = [
                sys.executable,
                str(root_dir / "tools" / "tune_ensemble_thresholds_xgb.py"),
                "--model",
                str(Path(str(model_prefix) + ".json")),
                "--meta",
                str(meta_path),
                "--data",
                str(labeled_path),
                "--dataset",
                payload.dataset_id,
                "--optimize",
                optimize_metric,
                "--target-fp-ratio",
                str(target_fp_ratio),
                "--min-recall",
                str(recall_floor),
                "--steps",
                str(steps_val),
                "--eval-iou",
                str(eval_iou),
                "--dedupe-iou",
                str(dedupe_iou),
                "--scoreless-iou",
                str(scoreless_iou),
                "--use-val-split",
            ]
            if target_fp_ratio_by_label_json:
                tune_cmd += [
                    "--target-fp-ratio-by-label-json",
                    target_fp_ratio_by_label_json,
                ]
            if min_recall_by_label_json:
                tune_cmd += [
                    "--min-recall-by-label-json",
                    min_recall_by_label_json,
                ]
            _run_step("objective", "Tuning object-level thresholds…", tune_cmd)
            if policy_layer_variant != "none":
                policy_layer_dir = output_dir / "policy_layer"
                _run_step(
                    "policy",
                    "Training learned policy layer…",
                    [
                        sys.executable,
                        str(root_dir / "tools" / "train_policy_layer.py"),
                        "--input",
                        str(labeled_path),
                        "--base-model",
                        str(Path(str(model_prefix) + ".json")),
                        "--base-meta",
                        str(meta_path),
                        "--output-dir",
                        str(policy_layer_dir),
                        "--variant",
                        policy_layer_variant,
                        "--seed",
                        str(int(payload.model_seed or 42)),
                        "--nested-folds",
                        "5",
                        "--target-fp-ratio",
                        str(target_fp_ratio),
                        "--min-recall",
                        str(recall_floor),
                        "--threshold-steps",
                        str(steps_val),
                        "--optimize",
                        optimize_metric,
                    ],
                )
        else:
            _run_step(
                "train",
                "Training MLP…",
                [
                    sys.executable,
                    str(root_dir / "tools" / "train_ensemble_mlp.py"),
                    "--input",
                    str(labeled_path),
                    "--output",
                    str(model_prefix),
                    "--hidden",
                    str(payload.model_hidden or "256,128"),
                    "--dropout",
                    str(_float_with_default(payload.model_dropout, 0.1)),
                    "--epochs",
                    str(int(payload.model_epochs or 20)),
                    "--lr",
                    str(_float_with_default(payload.model_lr, 1e-3)),
                    "--weight-decay",
                    str(_float_with_default(payload.model_weight_decay, 1e-4)),
                    "--seed",
                    str(int(payload.model_seed or 42)),
                    "--device",
                    "cuda",
                ],
            )
            optimize_metric = (payload.optimize_metric or "f1").strip().lower()
            if optimize_metric not in {"f1", "recall"}:
                optimize_metric = "f1"
            steps_val = int(payload.threshold_steps or 200)
            steps_val = max(20, min(1000, steps_val))
            calibrate_cmd = [
                sys.executable,
                str(root_dir / "tools" / "calibrate_ensemble_threshold.py"),
                "--model",
                str(Path(str(model_prefix) + ".pt")),
                "--data",
                str(labeled_path),
                "--meta",
                str(meta_path),
                "--target-fp-ratio",
                str(base_fp_ratio),
                "--min-recall",
                str(recall_floor),
                "--steps",
                str(steps_val),
                "--optimize",
                optimize_metric,
            ]
            if payload.per_class_thresholds is not False:
                calibrate_cmd.append("--per-class")
            _run_step("calibrate", "Calibrating thresholds…", calibrate_cmd)
            _run_step(
                "relax",
                "Relaxing thresholds…",
                [
                    sys.executable,
                    str(root_dir / "tools" / "relax_ensemble_thresholds.py"),
                    "--model",
                    str(Path(str(model_prefix) + ".pt")),
                    "--data",
                    str(labeled_path),
                    "--meta",
                    str(meta_path),
                    "--fp-ratio-cap",
                    str(relax_fp_ratio),
                ],
            )
        _update_calibration_phase(
            job=job,
            update_fn=_persisting_update_fn,
            step_plan=step_plan,
            phase="eval",
            message="Evaluating model…",
        )
        if job.cancel_event.is_set():
            raise RuntimeError("cancelled")
        iou_grid = payload.eval_iou_grid or "0.5,0.6,0.7,0.75,0.8,0.85,0.9"
        dedupe_grid = payload.dedupe_iou_grid or iou_grid
        eval_cmd = [
            sys.executable,
            str(
                root_dir
                / "tools"
                / ("eval_ensemble_xgb_dedupe.py" if calibration_model == "xgb" else "eval_ensemble_mlp_dedupe.py")
            ),
            "--model",
            str(Path(str(model_prefix) + (".json" if calibration_model == "xgb" else ".pt"))),
            "--meta",
            str(meta_path),
            "--data",
            str(labeled_path),
            "--dataset",
            payload.dataset_id,
            "--eval-iou",
            str(eval_iou),
            "--eval-iou-grid",
            iou_grid,
            "--dedupe-iou",
            str(dedupe_iou),
            "--dedupe-iou-grid",
            dedupe_grid,
            "--scoreless-iou",
            str(scoreless_iou),
            "--use-val-split",
        ]
        analysis_path = output_dir / f"ensemble_{calibration_model}.analysis.json"
        if calibration_model == "xgb":
            eval_cmd += ["--prepass-jsonl", str(prepass_path)]
        eval_cmd += ["--analysis-json", str(analysis_path)]
        eval_run = _run_subprocess_cancellable(eval_cmd, capture_output=True, text=True)
        eval_text = eval_run.stdout.strip()
        if eval_text:
            _write_text_atomic(eval_path, eval_text)

        metrics = {}
        try:
            metrics = json.loads(eval_path.read_text())
        except Exception:
            metrics = {}
        if isinstance(metrics, dict):
            metrics = _normalize_eval_metrics_for_api(metrics)
        policy_selection = {}
        policy_selection_path = output_dir / "policy_layer" / "policy_layer_selection.json"
        if calibration_model == "xgb" and policy_selection_path.exists():
            try:
                policy_selection = json.loads(policy_selection_path.read_text())
            except Exception:
                policy_selection = {}
        meta_payload = {}
        try:
            meta_payload = json.loads(meta_path.read_text())
        except Exception:
            meta_payload = {}
        report_bundle_json = output_dir / "report_bundle.json"
        report_bundle_md = output_dir / "report_bundle.md"
        report_bundle_warning: Optional[str] = None
        try:
            report_cmd = [
                sys.executable,
                str(root_dir / "tools" / "build_calibration_report_bundle.py"),
                "--eval-json",
                str(eval_path),
                "--meta-json",
                str(meta_path),
                "--model-family",
                calibration_model,
                "--output-json",
                str(report_bundle_json),
                "--output-md",
                str(report_bundle_md),
            ]
            if analysis_path.exists():
                report_cmd += ["--analysis-json", str(analysis_path)]
            if policy_selection_path.exists():
                report_cmd += ["--policy-selection-json", str(policy_selection_path)]
            _run_step("report", "Building calibration report…", report_cmd)
        except Exception as exc:  # noqa: BLE001
            report_bundle_warning = str(exc)

        job.result = {
            "output_dir": str(output_dir),
            "prepass_jsonl": str(prepass_path),
            "features": str(features_path),
            "labeled": str(labeled_path),
            "model": str(Path(str(model_prefix) + (".json" if calibration_model == "xgb" else ".pt"))),
            "meta": str(meta_path),
            "eval": str(eval_path),
            "metrics": metrics,
            "calibration_model": calibration_model,
            "policy_layer_selection": policy_selection,
            "policy_layer_summary": meta_payload.get("policy_layer_summary") if isinstance(meta_payload, dict) else None,
            "recipe_mode": recipe_mode,
            "lane_selection": lane_selection,
            "recipe_fingerprint": recipe_fingerprint,
            "recipe_reused": recipe_reused,
            "recipe_discovered": recipe_discovered,
            "recipe_registry_entry": recipe_registry_entry,
            "saved_prepass_recipe": saved_prepass_recipe_entry,
            "saved_prepass_recipe_id": (
                str(saved_prepass_recipe_entry.get("id"))
                if isinstance(saved_prepass_recipe_entry, dict) and saved_prepass_recipe_entry.get("id")
                else None
            ),
            "canonical_deployment_job": canonical_deployment_entry,
            "canonical_deployment_job_id": (
                str(canonical_deployment_entry.get("job_id"))
                if isinstance(canonical_deployment_entry, dict) and canonical_deployment_entry.get("job_id")
                else None
            ),
            "edr_package": edr_package_entry,
            "edr_package_id": (
                str(edr_package_entry.get("id"))
                if isinstance(edr_package_entry, dict) and edr_package_entry.get("id")
                else None
            ),
            "canonical_recipe_json": (
                str((Path(str(recipe_registry_entry.get("canonical_recipe_json")))).resolve())
                if isinstance(recipe_registry_entry, dict) and recipe_registry_entry.get("canonical_recipe_json")
                else (
                    str(_canonical_recipe_json_path(discovery_run_root).resolve())
                    if discovery_run_root is not None
                    else None
                )
            ),
            "canonical_recipe_md": (
                str((Path(str(recipe_registry_entry.get("canonical_recipe_md")))).resolve())
                if isinstance(recipe_registry_entry, dict) and recipe_registry_entry.get("canonical_recipe_md")
                else (
                    str(_canonical_recipe_md_path(discovery_run_root).resolve())
                    if discovery_run_root is not None and _canonical_recipe_md_path(discovery_run_root).exists()
                    else None
                )
            ),
            "canonical_recipe_branch": lane_family,
            "canonical_completion_context_json": (
                str(canonical_completion_context_path(discovery_run_root).resolve())
                if discovery_run_root is not None and canonical_completion_context_path(discovery_run_root).exists()
                else None
            ),
            "canonical_completion_summary_json": (
                str(canonical_completion_summary_path(discovery_run_root).resolve())
                if discovery_run_root is not None and canonical_completion_summary_path(discovery_run_root).exists()
                else None
            ),
            "discovery_run_root": str(discovery_run_root) if discovery_run_root is not None else None,
            "report_bundle_json": str(report_bundle_json) if report_bundle_json.exists() else None,
            "report_bundle_md": str(report_bundle_md) if report_bundle_md.exists() else None,
            "report_bundle_warning": report_bundle_warning,
        }
        _persisting_update_fn(
            job,
            status="completed",
            message="Done",
            phase="completed",
            progress=1.0,
            step_current=len(step_plan),
            step_total=len(step_plan),
            step_label="Completed",
            substep_current=0,
            substep_total=0,
            substep_label="",
        )
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, RuntimeError) and str(exc) == "cancelled":
            _persisting_update_fn(
                job,
                status="cancelled",
                message="Cancelled",
                phase="cancelled",
                substep_current=0,
                substep_total=0,
                substep_label="",
            )
        else:
            logger.exception("Calibration job %s failed", job.job_id)
            _persisting_update_fn(
                job,
                status="failed",
                message="Failed",
                phase="failed",
                error=str(exc),
                substep_current=0,
                substep_total=0,
                substep_label="",
            )
    finally:
        job.updated_at = time.time()
