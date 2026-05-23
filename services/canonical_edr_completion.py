"""Shared helpers for canonical EDR completion persistence."""

from __future__ import annotations

import json
import os
import re
import shutil
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models.prepass_recipes import PREPASS_RECIPE_SCHEMA_VERSION
from services.calibration_recipe_registry import (
    CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED,
    CANONICAL_EDR_ORIGIN_IMPORTED_PORTABLE,
    load_registry,
    register_promoted_recipe,
)
from services.edr_packages import get_edr_package, materialize_canonical_edr_package
from services.prepass_recipes import upsert_canonical_edr_saved_recipe_impl


CANONICAL_COMPLETION_CONTEXT_VERSION = 1
CANONICAL_COMPLETION_CONTEXT_JSON_NAME = "canonical_completion_context.json"
CANONICAL_COMPLETION_SUMMARY_JSON_NAME = "canonical_completion_summary.json"
CANONICAL_DEPLOYMENT_METADATA_JSON_NAME = "canonical_deployment.json"
CANONICAL_DEPLOYMENT_DEFAULT_SEED = 42
CANONICAL_DEPLOYMENT_FALLBACK_SEEDS = (42, 1337, 2025)
RECONSTRUCTED_REQUEST_DROP_KEYS = {
    "ensemble_enabled",
    "ensemble_job_id",
    "canonical_edr_json",
    "canonical_edr_md",
    "canonical_report_bundle_json",
    "canonical_deployment_job_id",
    "canonical_deployment_job_dir",
    "canonical_deployment_source_stage",
    "canonical_deployment_source_seed",
    "recipe_fingerprint",
    "recipe_registry_fingerprint",
    "recipe_registry_root",
    "expected_mean_f1",
    "edr_saved_source",
}


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
    if src_resolved == _path_identity(dest):
        return
    if dest.is_symlink():
        dest.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_resolved, dest)


def _canonical_slug(value: Any, *, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip()).strip("_")
    return slug or fallback


def canonical_deployment_job_id(dataset_id: str, recipe_fingerprint: str) -> str:
    dataset_slug = _canonical_slug(dataset_id, fallback="dataset")
    fingerprint_slug = re.sub(r"[^a-zA-Z0-9]+", "", str(recipe_fingerprint or "").strip())[:12] or "unknown"
    return f"canonical_edr_{dataset_slug}_{fingerprint_slug}"


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def canonical_completion_context_path(run_root: Path) -> Path:
    return run_root / CANONICAL_COMPLETION_CONTEXT_JSON_NAME


def canonical_completion_summary_path(run_root: Path) -> Path:
    return run_root / CANONICAL_COMPLETION_SUMMARY_JSON_NAME


def _load_completion_summary(run_root: Path) -> Dict[str, Any]:
    return _load_json_dict(canonical_completion_summary_path(run_root))


def _resolve_local_sam3_checkpoint() -> Optional[str]:
    env_value = str(os.environ.get("SAM3_CHECKPOINT_PATH") or "").strip()
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    pattern = str(Path("~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt").expanduser())
    for raw_path in glob(pattern):
        candidate = Path(raw_path).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    return None


def build_canonical_completion_context(
    *,
    dataset_id: str,
    recipe_fingerprint: str,
    recipe_fingerprint_payload: Dict[str, Any],
    calibration_request: Dict[str, Any],
    resolved_classifier_id: Optional[str],
    glossary_text: str,
    labelmap: Optional[List[str]] = None,
    report_bundle_json: Optional[Path] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": CANONICAL_COMPLETION_CONTEXT_VERSION,
        "dataset_id": str(dataset_id),
        "recipe_fingerprint": str(recipe_fingerprint),
        "recipe_fingerprint_payload": dict(recipe_fingerprint_payload or {}),
        "calibration_request": dict(calibration_request or {}),
        "resolved_classifier_id": (
            str(resolved_classifier_id).strip() if resolved_classifier_id is not None else None
        ),
        "glossary_text": str(glossary_text or ""),
        "labelmap": [str(item).strip() for item in (labelmap or []) if str(item).strip()],
        "report_bundle_json": (
            str(report_bundle_json.resolve())
            if report_bundle_json is not None and report_bundle_json.exists()
            else None
        ),
    }


def write_canonical_completion_context(run_root: Path, payload: Dict[str, Any]) -> Path:
    return _write_json_atomic(canonical_completion_context_path(run_root), payload)


def load_canonical_completion_context(
    *,
    run_root: Path,
    context_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    path = context_path or canonical_completion_context_path(run_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"canonical_completion_context_invalid:{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("canonical_completion_context_invalid")
    if int(payload.get("schema_version") or 0) != CANONICAL_COMPLETION_CONTEXT_VERSION:
        raise RuntimeError("canonical_completion_context_invalid")
    dataset_id = str(payload.get("dataset_id") or "").strip()
    recipe_fingerprint = str(payload.get("recipe_fingerprint") or "").strip()
    fingerprint_payload = payload.get("recipe_fingerprint_payload")
    calibration_request = payload.get("calibration_request")
    if (
        not dataset_id
        or not recipe_fingerprint
        or not isinstance(fingerprint_payload, dict)
        or not isinstance(calibration_request, dict)
    ):
        raise RuntimeError("canonical_completion_context_invalid")
    normalized = dict(payload)
    normalized["dataset_id"] = dataset_id
    normalized["recipe_fingerprint"] = recipe_fingerprint
    normalized["recipe_fingerprint_payload"] = dict(fingerprint_payload)
    normalized["calibration_request"] = dict(calibration_request)
    normalized["glossary_text"] = str(normalized.get("glossary_text") or "")
    classifier_id = normalized.get("resolved_classifier_id")
    normalized["resolved_classifier_id"] = (
        str(classifier_id).strip() if classifier_id is not None else None
    )
    report_path = str(normalized.get("report_bundle_json") or "").strip()
    normalized["report_bundle_json"] = report_path or None
    return normalized


def derive_calibration_cache_root_from_run_root(run_root: Path) -> Path:
    parent = run_root.parent
    if parent.name != "discovery_runs":
        raise RuntimeError("canonical_completion_run_root_invalid")
    return parent.parent.resolve()


def _canonical_lane_info(canonical_recipe_payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    preferred_lane = str(
        canonical_recipe_payload.get("discovered_winner_lane")
        or canonical_recipe_payload.get("lane_selection")
        or "window"
    ).strip().lower()
    if preferred_lane not in {"window", "nonwindow"}:
        preferred_lane = "window"
    lane_key = "canonical_windowed_recipe" if preferred_lane == "window" else "canonical_nonwindowed_recipe"
    lane_recipe = (
        dict(canonical_recipe_payload.get(lane_key) or {})
        if isinstance(canonical_recipe_payload.get(lane_key), dict)
        else {}
    )
    if lane_recipe:
        return preferred_lane, lane_recipe
    fallback_window = (
        dict(canonical_recipe_payload.get("canonical_windowed_recipe") or {})
        if isinstance(canonical_recipe_payload.get("canonical_windowed_recipe"), dict)
        else {}
    )
    if fallback_window:
        return "window", fallback_window
    fallback_nonwindow = (
        dict(canonical_recipe_payload.get("canonical_nonwindowed_recipe") or {})
        if isinstance(canonical_recipe_payload.get("canonical_nonwindowed_recipe"), dict)
        else {}
    )
    if fallback_nonwindow:
        return "nonwindow", fallback_nonwindow
    return preferred_lane, {}


def _resolve_existing_path(raw_path: Any) -> Optional[Path]:
    if raw_path is None:
        return None
    candidate = Path(str(raw_path)).expanduser()
    try:
        candidate = candidate.resolve()
    except Exception:
        candidate = Path(str(raw_path))
    return candidate if candidate.exists() else None


def _resolve_model_pair(source_dir: Path) -> Tuple[Path, Path]:
    direct_model = source_dir / "model.json"
    direct_meta = source_dir / "model.meta.json"
    if direct_model.exists() and direct_meta.exists():
        return direct_model, direct_meta
    for meta_path in sorted(source_dir.glob("model*.meta.json")):
        model_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".json"))
        if model_path.exists():
            return model_path, meta_path
    raise RuntimeError(f"canonical_deployment_model_missing:{source_dir}")


def _load_json_dict(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_metric_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def _load_eval_metrics(eval_json: Path) -> Dict[str, float]:
    payload = _load_json_dict(eval_json)
    metrics: Dict[str, float] = {}
    for key in ("f1", "fp", "precision", "recall", "tp", "fn"):
        parsed = _coerce_metric_float(payload.get(key))
        if parsed is not None:
            metrics[key] = parsed
    return metrics


def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, float, float, int]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    f1 = _coerce_metric_float(metrics.get("f1"))
    fp = _coerce_metric_float(metrics.get("fp"))
    precision = _coerce_metric_float(metrics.get("precision"))
    seed = int(candidate.get("seed") or CANONICAL_DEPLOYMENT_DEFAULT_SEED)
    return (
        -(f1 if f1 is not None else -1.0),
        fp if fp is not None else float("inf"),
        -(precision if precision is not None else -1.0),
        seed,
    )


def _deployment_candidate(
    *,
    stage: str,
    lane: str,
    seed: int,
    source_dir: Path,
) -> Optional[Dict[str, Any]]:
    if not source_dir.exists():
        return None
    model_path, meta_path = _resolve_model_pair(source_dir)
    metrics = _load_eval_metrics(source_dir / "eval.json")
    return {
        "stage": stage,
        "lane": lane,
        "seed": seed,
        "source_dir": source_dir,
        "model_path": model_path,
        "meta_path": meta_path,
        "metrics": metrics,
    }


def _choose_best_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return sorted(candidates, key=_candidate_sort_key)[0]


def _resolve_canonical_deployment_source(
    *,
    run_root: Path,
    canonical_recipe_payload: Dict[str, Any],
) -> Dict[str, Any]:
    winner_lane, lane_recipe = _canonical_lane_info(canonical_recipe_payload)
    source_decisions = (
        dict(lane_recipe.get("source_decisions") or {})
        if isinstance(lane_recipe.get("source_decisions"), dict)
        else {}
    )
    similarity_summary = _resolve_existing_path(source_decisions.get("similarity_quality"))
    magnitude_summary = _resolve_existing_path(source_decisions.get("sam_bias_magnitude"))

    if winner_lane == "window" and similarity_summary is not None:
        payload = _load_json_dict(similarity_summary)
        if str(payload.get("status") or "").strip().lower() == "promoted":
            tag = str(payload.get("winner_tag") or "").strip()
            if tag:
                candidates = [
                    candidate
                    for seed in CANONICAL_DEPLOYMENT_FALLBACK_SEEDS
                    for candidate in [
                        _deployment_candidate(
                            stage="postrun_similarity_quality_full_window_eval",
                            lane=winner_lane,
                            seed=seed,
                            source_dir=run_root
                            / "postrun_similarity_quality_full_window_eval"
                            / tag
                            / f"seed_{seed}",
                        )
                    ]
                    if candidate is not None
                ]
                chosen = _choose_best_candidate(candidates)
                if chosen is not None:
                    return chosen

    if magnitude_summary is not None:
        payload = _load_json_dict(magnitude_summary)
        full_winner = payload.get("full_winner") if isinstance(payload.get("full_winner"), dict) else {}
        tag = str(full_winner.get("tag") or "").strip()
        if tag:
            candidates = [
                candidate
                for seed in CANONICAL_DEPLOYMENT_FALLBACK_SEEDS
                for candidate in [
                    _deployment_candidate(
                        stage="postrun_sam_bias_magnitude_sweep",
                        lane=winner_lane,
                        seed=seed,
                        source_dir=run_root
                        / "postrun_sam_bias_magnitude_sweep"
                        / "full"
                        / tag
                        / "full"
                        / f"seed_{seed}",
                    )
                ]
                if candidate is not None
            ]
            chosen = _choose_best_candidate(candidates)
            if chosen is not None:
                return chosen

    candidates = [
        candidate
        for seed in CANONICAL_DEPLOYMENT_FALLBACK_SEEDS
        for candidate in [
            _deployment_candidate(
                stage="final_matrix",
                lane=winner_lane,
                seed=seed,
                source_dir=run_root / "final_matrix" / winner_lane / "full" / f"seed_{seed}",
            )
        ]
        if candidate is not None
    ]
    chosen = _choose_best_candidate(candidates)
    if chosen is not None:
        return chosen
    raise RuntimeError(f"canonical_deployment_source_missing:{run_root}")


def _rewrite_aux_model_path(
    *,
    raw_path: Any,
    source_dir: Path,
    target_dir: Path,
    target_name: str,
) -> Optional[str]:
    if raw_path is None:
        return None
    src_path = Path(str(raw_path))
    if not src_path.is_absolute():
        src_path = (source_dir / src_path).resolve()
    if not src_path.exists():
        return None
    dest_path = target_dir / target_name
    _copy2_if_different(src_path, dest_path)
    return dest_path.name


def rewrite_canonical_deployment_bundle_metadata(
    bundle_dir: Path,
    *,
    job_id: str,
    dataset_id: Optional[str] = None,
    canonical_recipe_json: Optional[Path] = None,
    canonical_recipe_md: Optional[Path] = None,
    canonical_report_bundle_json: Optional[Path] = None,
    source_stage: Optional[str] = None,
    source_seed: Optional[int] = None,
    source_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    meta_path = bundle_dir / "ensemble_xgb.meta.json"
    deployment_path = bundle_dir / CANONICAL_DEPLOYMENT_METADATA_JSON_NAME

    meta_payload = _load_json_dict(meta_path) if meta_path.exists() else {}
    deployment_payload = _load_json_dict(deployment_path) if deployment_path.exists() else {}

    def _remember_original(payload: Dict[str, Any], key: str, new_value: Optional[str]) -> None:
        if new_value is None:
            return
        existing = payload.get(key)
        if existing is None:
            return
        existing_text = str(existing)
        if existing_text == new_value:
            return
        original_key = f"original_{key}"
        if original_key not in payload:
            payload[original_key] = existing_text

    recipe_json_text = str(canonical_recipe_json.resolve()) if canonical_recipe_json is not None else None
    recipe_md_text = str(canonical_recipe_md.resolve()) if canonical_recipe_md is not None else None
    report_text = (
        str(canonical_report_bundle_json.resolve())
        if canonical_report_bundle_json is not None
        else None
    )
    source_dir_text = str(source_dir.resolve()) if source_dir is not None else None
    bundle_dir_text = str(bundle_dir)

    if meta_payload:
        _remember_original(meta_payload, "canonical_recipe_json", recipe_json_text)
        if recipe_json_text is not None:
            meta_payload["canonical_recipe_json"] = recipe_json_text
        if recipe_md_text is not None:
            meta_payload["canonical_recipe_md"] = recipe_md_text
        if report_text is not None:
            meta_payload["canonical_report_bundle_json"] = report_text
        meta_payload["canonical_deployment_job_id"] = str(job_id)
        meta_payload["canonical_deployment_job_dir"] = bundle_dir_text
        if source_stage is not None:
            meta_payload["canonical_deployment_source_stage"] = str(source_stage)
        if source_seed is not None:
            meta_payload["canonical_deployment_seed"] = int(source_seed)
        if source_dir_text is not None:
            _remember_original(meta_payload, "canonical_deployment_source_dir", source_dir_text)
            meta_payload["canonical_deployment_source_dir"] = source_dir_text
        _write_json_atomic(meta_path, meta_payload)

    if deployment_payload:
        _remember_original(deployment_payload, "source_dir", source_dir_text)
        _remember_original(deployment_payload, "canonical_recipe_json", recipe_json_text)
        deployment_payload["job_id"] = str(job_id)
        deployment_payload["job_dir"] = bundle_dir_text
        if dataset_id is not None:
            deployment_payload["dataset_id"] = str(dataset_id)
        if source_stage is not None:
            deployment_payload["source_stage"] = str(source_stage)
        if source_seed is not None:
            deployment_payload["source_seed"] = int(source_seed)
        if source_dir_text is not None:
            deployment_payload["source_dir"] = source_dir_text
        if recipe_json_text is not None:
            deployment_payload["canonical_recipe_json"] = recipe_json_text
        if recipe_md_text is not None:
            deployment_payload["canonical_recipe_md"] = recipe_md_text
        if report_text is not None:
            deployment_payload["canonical_report_bundle_json"] = report_text
        _write_json_atomic(deployment_path, deployment_payload)

    return {
        "meta_json": str(meta_path.resolve()) if meta_path.exists() else None,
        "canonical_deployment_json": (
            str(deployment_path.resolve()) if deployment_path.exists() else None
        ),
    }


def materialize_canonical_deployment_bundle(
    *,
    calibration_jobs_root: Path,
    run_root: Path,
    dataset_id: str,
    recipe_fingerprint: str,
    canonical_recipe_payload: Dict[str, Any],
    canonical_recipe_json: Path,
    report_bundle_json: Optional[Path] = None,
) -> Dict[str, Any]:
    source = _resolve_canonical_deployment_source(
        run_root=run_root,
        canonical_recipe_payload=canonical_recipe_payload,
    )
    winner_lane, lane_recipe = _canonical_lane_info(canonical_recipe_payload)
    scenario = dict(lane_recipe.get("scenario") or {}) if isinstance(lane_recipe.get("scenario"), dict) else {}
    policy = dict(lane_recipe.get("policy") or {}) if isinstance(lane_recipe.get("policy"), dict) else {}

    source_dir = Path(source["source_dir"]).resolve()
    model_path = Path(source["model_path"]).resolve()
    meta_path = Path(source["meta_path"]).resolve()
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    job_id = canonical_deployment_job_id(dataset_id, recipe_fingerprint)
    jobs_root = calibration_jobs_root.resolve()
    jobs_root.mkdir(parents=True, exist_ok=True)
    temp_dir = jobs_root / f".{job_id}.tmp.{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    base_model_name = "ensemble_xgb.json" if model_path.suffix.lower() == ".json" else model_path.name
    _copy2_if_different(model_path, temp_dir / base_model_name)
    meta_copy = dict(meta_payload)
    meta_copy["model_path"] = base_model_name
    meta_copy["ensemble_policy"] = dict(policy or {})
    meta_copy["canonical_recipe_json"] = str(canonical_recipe_json.resolve())
    meta_copy["canonical_recipe_fingerprint"] = str(recipe_fingerprint)
    meta_copy["canonical_winner_lane"] = winner_lane
    meta_copy["canonical_deployment_source_stage"] = str(source.get("stage") or "")
    meta_copy["canonical_deployment_seed"] = int(source.get("seed") or CANONICAL_DEPLOYMENT_DEFAULT_SEED)

    split_cfg = dict(meta_copy.get("split_head") or {}) if isinstance(meta_copy.get("split_head"), dict) else {}
    if "split_head" in scenario and split_cfg:
        split_cfg["enabled"] = bool(scenario.get("split_head"))
    if isinstance(split_cfg.get("models"), dict):
        new_models: Dict[str, str] = {}
        for key, raw_path in split_cfg.get("models", {}).items():
            copied_name = _rewrite_aux_model_path(
                raw_path=raw_path,
                source_dir=source_dir,
                target_dir=temp_dir,
                target_name=f"ensemble_xgb.split_head.{key}.json",
            )
            if copied_name:
                new_models[str(key)] = copied_name
        split_cfg["models"] = new_models
    if split_cfg:
        meta_copy["split_head"] = split_cfg

    def _rewrite_quality_block(block_key: str, alpha_key: str, target_suffix: str) -> None:
        block = dict(meta_copy.get(block_key) or {}) if isinstance(meta_copy.get(block_key), dict) else {}
        if not block:
            return
        if alpha_key in scenario and block.get("enabled"):
            block["alpha"] = scenario.get(alpha_key)
        if bool(block.get("enabled")):
            copied_name = _rewrite_aux_model_path(
                raw_path=block.get("model_path"),
                source_dir=source_dir,
                target_dir=temp_dir,
                target_name=target_suffix,
            )
            if copied_name:
                block["model_path"] = copied_name
        meta_copy[block_key] = block

    _rewrite_quality_block(
        "sam3_text_quality",
        "sam3_text_quality_alpha",
        "ensemble_xgb.sam3_text_quality.json",
    )
    _rewrite_quality_block(
        "sam3_similarity_quality",
        "sam3_similarity_quality_alpha",
        "ensemble_xgb.sam3_similarity_quality.json",
    )

    meta_out = temp_dir / "ensemble_xgb.meta.json"
    _write_json_atomic(meta_out, meta_copy)

    for src_name, dest_name in (
        ("eval.json", "ensemble_xgb.eval.json"),
        ("analysis.json", "ensemble_xgb.analysis.json"),
    ):
        src_path = source_dir / src_name
        if src_path.exists():
            _copy2_if_different(src_path, temp_dir / dest_name)
    if report_bundle_json is not None and report_bundle_json.exists():
        _copy2_if_different(report_bundle_json, temp_dir / "report_bundle.json")

    bundle_meta = {
        "job_id": job_id,
        "dataset_id": str(dataset_id),
        "recipe_fingerprint": str(recipe_fingerprint),
        "winner_lane": winner_lane,
        "source_stage": str(source.get("stage") or ""),
        "source_dir": str(source_dir),
        "source_seed": int(source.get("seed") or CANONICAL_DEPLOYMENT_DEFAULT_SEED),
        "canonical_recipe_json": str(canonical_recipe_json.resolve()),
    }
    _write_json_atomic(temp_dir / CANONICAL_DEPLOYMENT_METADATA_JSON_NAME, bundle_meta)

    final_dir = jobs_root / job_id
    if final_dir.exists():
        shutil.rmtree(final_dir, ignore_errors=True)
    os.replace(temp_dir, final_dir)
    rewrite_canonical_deployment_bundle_metadata(
        final_dir,
        job_id=job_id,
        dataset_id=str(dataset_id),
        canonical_recipe_json=canonical_recipe_json.resolve(),
        canonical_report_bundle_json=(
            (final_dir / "report_bundle.json").resolve()
            if (final_dir / "report_bundle.json").exists()
            else None
        ),
        source_stage=str(source.get("stage") or ""),
        source_seed=int(source.get("seed") or CANONICAL_DEPLOYMENT_DEFAULT_SEED),
        source_dir=source_dir,
    )
    return {
        "job_id": job_id,
        "job_dir": str(final_dir.resolve()),
        "model_family": "xgb",
        "winner_lane": winner_lane,
        "source_stage": str(source.get("stage") or ""),
        "source_seed": int(source.get("seed") or CANONICAL_DEPLOYMENT_DEFAULT_SEED),
        "source_dir": str(source_dir),
        "metrics": dict(source.get("metrics") or {}),
        "meta_json": str((final_dir / "ensemble_xgb.meta.json").resolve()),
        "model_json": str((final_dir / "ensemble_xgb.json").resolve()),
    }


def _serialize_canonical_deployment_bundle(bundle_dir: Path) -> Optional[Dict[str, Any]]:
    bundle_dir = bundle_dir.resolve()
    model_json = bundle_dir / "ensemble_xgb.json"
    meta_json = bundle_dir / "ensemble_xgb.meta.json"
    deployment_json = bundle_dir / CANONICAL_DEPLOYMENT_METADATA_JSON_NAME
    if not (model_json.exists() and meta_json.exists() and deployment_json.exists()):
        return None
    deployment = _load_json_dict(deployment_json)
    if not deployment:
        return None
    created_at = max(
        model_json.stat().st_mtime if model_json.exists() else 0.0,
        meta_json.stat().st_mtime if meta_json.exists() else 0.0,
        deployment_json.stat().st_mtime if deployment_json.exists() else 0.0,
    )
    job_id = str(deployment.get("job_id") or bundle_dir.name).strip() or bundle_dir.name
    dataset_id = str(deployment.get("dataset_id") or "").strip() or None
    canonical_recipe_json = str(deployment.get("canonical_recipe_json") or "").strip() or None
    canonical_recipe_md = str(deployment.get("canonical_recipe_md") or "").strip() or None
    report_bundle_json = str(deployment.get("canonical_report_bundle_json") or "").strip() or None
    request_payload: Dict[str, Any] = {}
    if dataset_id:
        request_payload["dataset_id"] = dataset_id
    serialized = {
        "job_id": job_id,
        "status": "completed",
        "message": "Canonical EDR deployment bundle",
        "phase": "completed",
        "progress": 1.0,
        "processed": 0,
        "total": 0,
        "step_current": 0,
        "step_total": 0,
        "step_label": "Canonical bundle",
        "substep_current": 0,
        "substep_total": 0,
        "substep_label": "",
        "created_at": created_at,
        "updated_at": created_at,
        "request": request_payload,
        "result": {
            "output_dir": str(bundle_dir),
            "model": str(model_json.resolve()),
            "meta": str(meta_json.resolve()),
            "calibration_model": "xgb",
            "canonical_recipe_json": canonical_recipe_json,
            "canonical_recipe_md": canonical_recipe_md,
            "report_bundle_json": report_bundle_json,
            "canonical_deployment_job": {
                "job_id": job_id,
                "job_dir": str(bundle_dir),
                "winner_lane": deployment.get("winner_lane"),
                "source_stage": deployment.get("source_stage"),
                "source_seed": deployment.get("source_seed"),
                "source_dir": deployment.get("source_dir"),
                "meta_json": str(meta_json.resolve()),
                "model_json": str(model_json.resolve()),
            },
            "canonical_deployment_job_id": job_id,
        },
        "error": None,
        "job_kind": "canonical_bundle",
        "persistent_bundle": True,
    }
    return serialized


def list_canonical_deployment_jobs(calibration_jobs_root: Path) -> List[Dict[str, Any]]:
    if not calibration_jobs_root.exists():
        return []
    bundles: List[Dict[str, Any]] = []
    for entry in sorted(calibration_jobs_root.iterdir()):
        if not entry.is_dir():
            continue
        serialized = _serialize_canonical_deployment_bundle(entry)
        if serialized is not None:
            bundles.append(serialized)
    return bundles


def get_canonical_deployment_job(calibration_jobs_root: Path, job_id: str) -> Optional[Dict[str, Any]]:
    candidate = calibration_jobs_root / str(job_id)
    if not candidate.exists() or not candidate.is_dir():
        return None
    return _serialize_canonical_deployment_bundle(candidate)


def _find_registry_entry_for_run(
    *,
    calibration_cache_root: Path,
    run_root: Path,
    summary_payload: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    summary = summary_payload if isinstance(summary_payload, dict) else {}
    summary_entry = summary.get("recipe_registry_entry")
    if isinstance(summary_entry, dict) and str(summary_entry.get("fingerprint") or "").strip():
        return dict(summary_entry)

    canonical_recipe_json = str((run_root / "canonical_edr.json").resolve())
    registry = load_registry(calibration_cache_root)
    entries = registry.get("entries")
    if not isinstance(entries, dict):
        return None
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        entry_run_root = str(entry.get("discovery_run_root") or "").strip()
        if entry_run_root and entry_run_root == str(run_root.resolve()):
            return dict(entry)
        entry_recipe_json = str(entry.get("canonical_recipe_json") or "").strip()
        if entry_recipe_json and entry_recipe_json == canonical_recipe_json:
            return dict(entry)
    return None


def _load_fingerprint_payload(registry_entry: Dict[str, Any]) -> Dict[str, Any]:
    fingerprint_json = _resolve_existing_path(registry_entry.get("fingerprint_json"))
    if fingerprint_json is None:
        return {}
    return _load_json_dict(fingerprint_json)


def _canonical_origin_kind(registry_entry: Optional[Dict[str, Any]]) -> str:
    if not isinstance(registry_entry, dict):
        return CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED
    raw_origin = str(
        registry_entry.get("origin_kind") or CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED
    ).strip().lower()
    if raw_origin in {
        CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED,
        CANONICAL_EDR_ORIGIN_IMPORTED_PORTABLE,
    }:
        return raw_origin
    return CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED


def _resolve_existing_canonical_deployment(
    *,
    calibration_jobs_root: Path,
    registry_entry: Optional[Dict[str, Any]],
    saved_prepass_recipe: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    candidate_ids: List[str] = []
    candidate_dirs: List[Path] = []

    def _add_job_id(raw_value: Any) -> None:
        value = str(raw_value or "").strip()
        if value and value not in candidate_ids:
            candidate_ids.append(value)

    def _add_job_dir(raw_value: Any) -> None:
        if raw_value is None:
            return
        candidate = Path(str(raw_value)).expanduser()
        try:
            candidate = candidate.resolve()
        except Exception:
            candidate = Path(str(raw_value))
        if candidate.exists() and candidate not in candidate_dirs:
            candidate_dirs.append(candidate)

    if isinstance(registry_entry, dict):
        _add_job_id(registry_entry.get("canonical_deployment_job_id"))
        _add_job_dir(registry_entry.get("canonical_deployment_job_dir"))
    if isinstance(saved_prepass_recipe, dict):
        config = (
            dict(saved_prepass_recipe.get("config") or {})
            if isinstance(saved_prepass_recipe.get("config"), dict)
            else {}
        )
        _add_job_id(config.get("canonical_deployment_job_id") or config.get("ensemble_job_id"))
        _add_job_dir(config.get("canonical_deployment_job_dir"))

    for job_id in candidate_ids:
        serialized = get_canonical_deployment_job(calibration_jobs_root, job_id)
        if not isinstance(serialized, dict):
            continue
        result = serialized.get("result")
        if isinstance(result, dict) and isinstance(result.get("canonical_deployment_job"), dict):
            return dict(result["canonical_deployment_job"])
    for job_dir in candidate_dirs:
        serialized = _serialize_canonical_deployment_bundle(job_dir)
        if not isinstance(serialized, dict):
            continue
        result = serialized.get("result")
        if isinstance(result, dict) and isinstance(result.get("canonical_deployment_job"), dict):
            return dict(result["canonical_deployment_job"])
    return None


def _resolve_existing_edr_package(
    *,
    packages_root: Path,
    registry_entry: Optional[Dict[str, Any]],
    saved_prepass_recipe: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    candidate_ids: List[str] = []

    def _add_package_id(raw_value: Any) -> None:
        value = str(raw_value or "").strip()
        if value and value not in candidate_ids:
            candidate_ids.append(value)

    if isinstance(registry_entry, dict):
        _add_package_id(registry_entry.get("edr_package_id"))
    if isinstance(saved_prepass_recipe, dict) and isinstance(saved_prepass_recipe.get("config"), dict):
        config = dict(saved_prepass_recipe.get("config") or {})
        _add_package_id(config.get("edr_package_id"))

    for package_id in candidate_ids:
        try:
            return get_edr_package(packages_root, package_id)
        except Exception:
            continue
    return None


def _load_saved_canonical_recipe_payload(
    *,
    recipes_root: Path,
    dataset_id: str,
    recipe_fingerprint: str,
    canonical_recipe_json: Path,
    summary_payload: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    summary = summary_payload if isinstance(summary_payload, dict) else {}
    summary_recipe = summary.get("saved_prepass_recipe")
    if isinstance(summary_recipe, dict):
        return dict(summary_recipe)

    deterministic_id = canonical_deployment_job_id(dataset_id, recipe_fingerprint)
    direct_meta = recipes_root / deterministic_id / "prepass.meta.json"
    if direct_meta.exists():
        payload = _load_json_dict(direct_meta)
        if payload:
            return payload

    canonical_recipe_json_text = str(canonical_recipe_json.resolve())
    for meta_path in sorted(recipes_root.glob("*/prepass.meta.json")):
        payload = _load_json_dict(meta_path)
        if not payload:
            continue
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        if str(config.get("recipe_kind") or "").strip().lower() != "canonical_edr":
            continue
        if str(config.get("recipe_registry_fingerprint") or "").strip() == str(recipe_fingerprint):
            return payload
        if str(config.get("canonical_edr_json") or "").strip() == canonical_recipe_json_text:
            return payload
    return None


def _build_minimal_calibration_request(
    *,
    dataset_id: str,
    registry_entry: Dict[str, Any],
    fingerprint_payload: Dict[str, Any],
) -> Dict[str, Any]:
    lane_selection = str(
        registry_entry.get("lane_selection")
        or fingerprint_payload.get("lane_selection")
        or "window"
    ).strip() or "window"
    prepass_config = (
        dict(fingerprint_payload.get("prepass_config") or {})
        if isinstance(fingerprint_payload.get("prepass_config"), dict)
        else {}
    )
    lane_prepass = (
        dict(prepass_config.get(lane_selection) or {})
        if isinstance(prepass_config.get(lane_selection), dict)
        else {}
    )
    request: Dict[str, Any] = {
        "dataset_id": str(dataset_id),
        "lane_selection": lane_selection,
        "classifier_id": registry_entry.get("classifier_id") or fingerprint_payload.get("classifier_id"),
        "calibration_max_images": registry_entry.get("requested_max_images")
        or fingerprint_payload.get("requested_max_images"),
        "support_iou": fingerprint_payload.get("support_iou"),
        "label_iou": fingerprint_payload.get("label_iou"),
        "eval_iou": fingerprint_payload.get("eval_iou"),
    }
    for key in (
        "enable_yolo",
        "enable_rfdetr",
        "sam3_text_synonym_budget",
        "sam3_text_window_extension",
        "sam3_text_window_mode",
        "sam3_text_window_size",
        "sam3_text_window_overlap",
        "prepass_sam3_text_thr",
        "prepass_similarity_score",
        "similarity_min_exemplar_score",
        "similarity_exemplar_count",
        "similarity_exemplar_strategy",
        "similarity_exemplar_seed",
        "similarity_exemplar_fraction",
        "similarity_exemplar_min",
        "similarity_exemplar_max",
        "similarity_exemplar_source_quota",
        "similarity_window_extension",
        "similarity_window_mode",
        "similarity_window_size",
        "similarity_window_overlap",
        "fusion_mode",
        "dedupe_iou",
        "cross_class_dedupe_enabled",
        "cross_class_dedupe_iou",
    ):
        if key in lane_prepass:
            request[key] = lane_prepass.get(key)
    return {key: value for key, value in request.items() if value is not None}


def _sanitize_reconstructed_calibration_request(saved_config: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in saved_config.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        if key_text in RECONSTRUCTED_REQUEST_DROP_KEYS:
            continue
        if key_text.startswith("canonical_"):
            continue
        if key_text.startswith("recipe_registry_"):
            continue
        sanitized[key_text] = value
    return sanitized


def reconstruct_canonical_completion_context(
    *,
    calibration_cache_root: Path,
    run_root: Path,
) -> Optional[Dict[str, Any]]:
    existing = load_canonical_completion_context(run_root=run_root)
    if existing is not None:
        return existing

    canonical_recipe_json = run_root / "canonical_edr.json"
    if not canonical_recipe_json.exists():
        return None

    summary_payload = _load_completion_summary(run_root)
    registry_entry = _find_registry_entry_for_run(
        calibration_cache_root=calibration_cache_root,
        run_root=run_root,
        summary_payload=summary_payload,
    )
    if not isinstance(registry_entry, dict):
        return None

    fingerprint = str(registry_entry.get("fingerprint") or "").strip()
    dataset_id = str(registry_entry.get("dataset_id") or "").strip()
    if not fingerprint or not dataset_id:
        return None

    fingerprint_payload = _load_fingerprint_payload(registry_entry)
    if not fingerprint_payload:
        return None

    saved_recipe = _load_saved_canonical_recipe_payload(
        recipes_root=calibration_cache_root.parent / "prepass_recipes",
        dataset_id=dataset_id,
        recipe_fingerprint=fingerprint,
        canonical_recipe_json=canonical_recipe_json,
        summary_payload=summary_payload,
    )
    saved_config = saved_recipe.get("config") if isinstance(saved_recipe, dict) and isinstance(saved_recipe.get("config"), dict) else {}
    glossary_text = (
        str(saved_recipe.get("glossary") or "")
        if isinstance(saved_recipe, dict)
        else ""
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
    resolved_classifier_id = (
        str(saved_config.get("resolved_classifier_id") or "").strip()
        or str(registry_entry.get("classifier_id") or "").strip()
        or None
    )
    report_bundle_path = _resolve_existing_path(
        summary_payload.get("report_bundle_json")
        or registry_entry.get("report_bundle_json")
        or (run_root / "report_bundle.json")
    )
    context = build_canonical_completion_context(
        dataset_id=dataset_id,
        recipe_fingerprint=fingerprint,
        recipe_fingerprint_payload=fingerprint_payload,
        calibration_request=calibration_request,
        resolved_classifier_id=resolved_classifier_id,
        glossary_text=glossary_text,
        labelmap=list(saved_config.get("labelmap") or []),
        report_bundle_json=report_bundle_path,
    )
    write_canonical_completion_context(run_root, context)
    return context


def repair_persisted_canonical_completion(
    *,
    calibration_cache_root: Path,
    run_root: Path,
) -> Dict[str, Any]:
    completion_context = reconstruct_canonical_completion_context(
        calibration_cache_root=calibration_cache_root,
        run_root=run_root,
    )
    if completion_context is None:
        raise RuntimeError("canonical_completion_context_missing")
    canonical_recipe_json = run_root / "canonical_edr.json"
    if not canonical_recipe_json.exists():
        raise RuntimeError("canonical_recipe_missing")
    canonical_recipe_md = run_root / "canonical_edr.md"
    report_bundle_json = run_root / "report_bundle.json"
    return persist_canonical_edr_completion(
        calibration_cache_root=calibration_cache_root,
        run_root=run_root,
        canonical_recipe_json=canonical_recipe_json,
        canonical_recipe_md=canonical_recipe_md if canonical_recipe_md.exists() else None,
        canonical_recipe_payload=None,
        completion_context=completion_context,
        existing_registry_entry=None,
        report_bundle_json=report_bundle_json if report_bundle_json.exists() else None,
        write_summary=True,
    )


def persist_canonical_edr_completion(
    *,
    calibration_cache_root: Path,
    run_root: Optional[Path],
    canonical_recipe_json: Path,
    canonical_recipe_md: Optional[Path],
    canonical_recipe_payload: Optional[Dict[str, Any]],
    completion_context: Optional[Dict[str, Any]],
    existing_registry_entry: Optional[Dict[str, Any]] = None,
    report_bundle_json: Optional[Path] = None,
    write_summary: bool = True,
) -> Dict[str, Any]:
    packages_root = calibration_cache_root.parent / "edr_packages"
    packages_root.mkdir(parents=True, exist_ok=True)
    uploads_root = calibration_cache_root.parent
    discovery_run_root: Optional[Path] = run_root.resolve() if run_root is not None else None
    payload = (
        dict(canonical_recipe_payload)
        if isinstance(canonical_recipe_payload, dict)
        else json.loads(canonical_recipe_json.read_text(encoding="utf-8"))
    )
    canonical_recipe_md = (
        canonical_recipe_md.resolve() if canonical_recipe_md is not None and canonical_recipe_md.exists() else None
    )
    effective_report = (
        report_bundle_json.resolve()
        if report_bundle_json is not None and report_bundle_json.exists()
        else None
    )
    if effective_report is None and completion_context is not None:
        raw_report = str(completion_context.get("report_bundle_json") or "").strip()
        candidate_report = Path(raw_report).resolve() if raw_report else None
        if candidate_report is not None and candidate_report.exists():
            effective_report = candidate_report

    registry_entry = existing_registry_entry if isinstance(existing_registry_entry, dict) else None
    saved_prepass_recipe = None
    canonical_deployment = None
    edr_package = None
    persistence_status = "artifact_only_missing_context"

    def _completion_labelmap() -> List[str]:
        if isinstance(completion_context, dict):
            values = completion_context.get("labelmap")
            if isinstance(values, list):
                return [str(item).strip() for item in values if str(item).strip()]
        if isinstance(saved_prepass_recipe, dict) and isinstance(saved_prepass_recipe.get("config"), dict):
            values = saved_prepass_recipe.get("config", {}).get("labelmap")
            if isinstance(values, list):
                return [str(item).strip() for item in values if str(item).strip()]
        return []

    def _resolve_classifier_path(raw_value: str) -> Optional[Path]:
        text = str(raw_value or "").strip()
        if not text:
            return None
        candidate = Path(text).expanduser()
        if candidate.is_absolute():
            return candidate.resolve() if candidate.exists() else None
        uploads_candidate = (uploads_root / candidate).resolve()
        if uploads_candidate.exists():
            return uploads_candidate
        classifiers_root = uploads_root / "classifiers"
        direct = (classifiers_root / candidate).resolve()
        if direct.exists():
            return direct
        alt = candidate.resolve()
        if alt.exists():
            return alt
        return None

    def _ensure_edr_package(*, allow_materialize: bool) -> Optional[Dict[str, Any]]:
        nonlocal registry_entry, saved_prepass_recipe, edr_package
        def _package_sync_calibration_request() -> Dict[str, Any]:
            if isinstance(saved_prepass_recipe, dict) and isinstance(saved_prepass_recipe.get("config"), dict):
                preserved = _sanitize_reconstructed_calibration_request(
                    dict(saved_prepass_recipe.get("config") or {})
                )
                if preserved:
                    return preserved
            if completion_context is not None:
                return dict(completion_context.get("calibration_request") or {})
            return {}

        def _saved_recipe_missing_package_fields(recipe: Optional[Dict[str, Any]], package: Dict[str, Any]) -> bool:
            if not isinstance(recipe, dict):
                return True
            config = recipe.get("config") if isinstance(recipe.get("config"), dict) else {}
            return str(config.get("edr_package_id") or "").strip() != str(package.get("id") or "").strip()

        def _registry_missing_package_fields(entry: Optional[Dict[str, Any]], package: Dict[str, Any]) -> bool:
            if not isinstance(entry, dict):
                return True
            return str(entry.get("edr_package_id") or "").strip() != str(package.get("id") or "").strip()

        edr_package = _resolve_existing_edr_package(
            packages_root=packages_root,
            registry_entry=registry_entry,
            saved_prepass_recipe=saved_prepass_recipe,
        )
        if edr_package is not None:
            if (
                completion_context is not None
                and isinstance(registry_entry, dict)
                and isinstance(saved_prepass_recipe, dict)
                and (
                    _saved_recipe_missing_package_fields(saved_prepass_recipe, edr_package)
                    or _registry_missing_package_fields(registry_entry, edr_package)
                )
            ):
                registry_entry = register_promoted_recipe(
                    calibration_cache_root,
                    fingerprint=str(completion_context["recipe_fingerprint"]),
                    fingerprint_payload=dict(completion_context["recipe_fingerprint_payload"]),
                    dataset_id=str(completion_context["dataset_id"]),
                    canonical_recipe_json=canonical_recipe_json.resolve(),
                    canonical_recipe_md=canonical_recipe_md,
                    report_bundle_json=effective_report,
                    discovery_run_root=discovery_run_root,
                    origin_kind=_canonical_origin_kind(registry_entry),
                    canonical_deployment=canonical_deployment,
                    edr_package=edr_package,
                )
                saved_prepass_recipe = upsert_canonical_edr_saved_recipe_impl(
                    recipes_root=calibration_cache_root.parent / "prepass_recipes",
                    dataset_id=str(completion_context["dataset_id"]),
                    calibration_request=_package_sync_calibration_request(),
                    classifier_id=completion_context.get("resolved_classifier_id"),
                    recipe_fingerprint=str(completion_context["recipe_fingerprint"]),
                    canonical_recipe=payload,
                    canonical_recipe_json=canonical_recipe_json.resolve(),
                    canonical_recipe_md=canonical_recipe_md,
                    report_bundle_json=effective_report,
                    recipe_registry_entry=registry_entry,
                    glossary_text=str(completion_context.get("glossary_text") or ""),
                    prepass_schema_version=PREPASS_RECIPE_SCHEMA_VERSION,
                    canonical_deployment=canonical_deployment,
                    labelmap=_completion_labelmap(),
                    edr_package=edr_package,
                )
            return edr_package
        if not allow_materialize or completion_context is None:
            return edr_package
        if (
            not isinstance(saved_prepass_recipe, dict)
            or not isinstance(registry_entry, dict)
            or not isinstance(canonical_deployment, dict)
        ):
            return None
        edr_package = materialize_canonical_edr_package(
            packages_root=packages_root,
            saved_recipe=saved_prepass_recipe,
            registry_entry=registry_entry,
            fingerprint_payload=dict(completion_context.get("recipe_fingerprint_payload") or {}),
            yolo_job_root=uploads_root / "yolo_runs",
            rfdetr_job_root=uploads_root / "rfdetr_runs",
            calibration_root=uploads_root / "calibration_jobs",
            classifiers_root=uploads_root / "classifiers",
            load_labelmap_fn=lambda _dataset_id: _completion_labelmap(),
            resolve_classifier_path_fn=_resolve_classifier_path,
            sam3_checkpoint_path=_resolve_local_sam3_checkpoint(),
        )
        registry_entry = register_promoted_recipe(
            calibration_cache_root,
            fingerprint=str(completion_context["recipe_fingerprint"]),
            fingerprint_payload=dict(completion_context["recipe_fingerprint_payload"]),
            dataset_id=str(completion_context["dataset_id"]),
            canonical_recipe_json=canonical_recipe_json.resolve(),
            canonical_recipe_md=canonical_recipe_md,
            report_bundle_json=effective_report,
            discovery_run_root=discovery_run_root,
            origin_kind=_canonical_origin_kind(registry_entry),
            canonical_deployment=canonical_deployment,
            edr_package=edr_package,
        )
        saved_prepass_recipe = upsert_canonical_edr_saved_recipe_impl(
            recipes_root=calibration_cache_root.parent / "prepass_recipes",
            dataset_id=str(completion_context["dataset_id"]),
            calibration_request=_package_sync_calibration_request(),
            classifier_id=completion_context.get("resolved_classifier_id"),
            recipe_fingerprint=str(completion_context["recipe_fingerprint"]),
            canonical_recipe=payload,
            canonical_recipe_json=canonical_recipe_json.resolve(),
            canonical_recipe_md=canonical_recipe_md,
            report_bundle_json=effective_report,
            recipe_registry_entry=registry_entry,
            glossary_text=str(completion_context.get("glossary_text") or ""),
            prepass_schema_version=PREPASS_RECIPE_SCHEMA_VERSION,
            canonical_deployment=canonical_deployment,
            labelmap=_completion_labelmap(),
            edr_package=edr_package,
        )
        return edr_package

    if registry_entry is not None and run_root is None:
        saved_prepass_recipe = _load_saved_canonical_recipe_payload(
            recipes_root=calibration_cache_root.parent / "prepass_recipes",
            dataset_id=str(registry_entry.get("dataset_id") or ""),
            recipe_fingerprint=str(registry_entry.get("fingerprint") or ""),
            canonical_recipe_json=canonical_recipe_json.resolve(),
        )
        canonical_deployment = _resolve_existing_canonical_deployment(
            calibration_jobs_root=calibration_cache_root.parent / "calibration_jobs",
            registry_entry=registry_entry,
            saved_prepass_recipe=saved_prepass_recipe,
        )
        edr_package = _resolve_existing_edr_package(
            packages_root=packages_root,
            registry_entry=registry_entry,
            saved_prepass_recipe=saved_prepass_recipe,
        )
        if (
            edr_package is None
            and completion_context is not None
        ):
            raw_run_root = str(registry_entry.get("discovery_run_root") or "").strip()
            if raw_run_root:
                candidate_run_root = Path(raw_run_root).resolve()
                if candidate_run_root.exists():
                    discovery_run_root = candidate_run_root
            _ensure_edr_package(allow_materialize=True)
        persistence_status = "reused_existing"
    elif completion_context is not None:
        if (
            discovery_run_root is None
            and registry_entry is not None
            and _canonical_origin_kind(registry_entry) == CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED
        ):
            raw_run_root = str(registry_entry.get("discovery_run_root") or "").strip()
            discovery_run_root = Path(raw_run_root).resolve() if raw_run_root else None
        if discovery_run_root is None:
            saved_prepass_recipe = _load_saved_canonical_recipe_payload(
                recipes_root=calibration_cache_root.parent / "prepass_recipes",
                dataset_id=str(completion_context.get("dataset_id") or ""),
                recipe_fingerprint=str(completion_context.get("recipe_fingerprint") or ""),
                canonical_recipe_json=canonical_recipe_json.resolve(),
            )
            canonical_deployment = _resolve_existing_canonical_deployment(
                calibration_jobs_root=calibration_cache_root.parent / "calibration_jobs",
                registry_entry=registry_entry,
                saved_prepass_recipe=saved_prepass_recipe,
            )
            edr_package = _resolve_existing_edr_package(
                packages_root=packages_root,
                registry_entry=registry_entry,
                saved_prepass_recipe=saved_prepass_recipe,
            )
            if registry_entry is None:
                raise RuntimeError("canonical_completion_context_missing_run_root")
            _ensure_edr_package(allow_materialize=True)
            persistence_status = "reused_existing"
        else:
            canonical_deployment = materialize_canonical_deployment_bundle(
                calibration_jobs_root=calibration_cache_root.parent / "calibration_jobs",
                run_root=discovery_run_root,
                dataset_id=str(completion_context["dataset_id"]),
                recipe_fingerprint=str(completion_context["recipe_fingerprint"]),
                canonical_recipe_payload=payload,
                canonical_recipe_json=canonical_recipe_json.resolve(),
                report_bundle_json=effective_report,
            )
        registry_entry = register_promoted_recipe(
            calibration_cache_root,
            fingerprint=str(completion_context["recipe_fingerprint"]),
            fingerprint_payload=dict(completion_context["recipe_fingerprint_payload"]),
            dataset_id=str(completion_context["dataset_id"]),
            canonical_recipe_json=canonical_recipe_json.resolve(),
            canonical_recipe_md=canonical_recipe_md,
            report_bundle_json=effective_report,
            discovery_run_root=discovery_run_root,
            origin_kind=(
                CANONICAL_EDR_ORIGIN_DISCOVERY_BACKED
                if discovery_run_root is not None
                else _canonical_origin_kind(existing_registry_entry)
            ),
            canonical_deployment=canonical_deployment,
            edr_package=edr_package,
        )
        if discovery_run_root is not None or saved_prepass_recipe is None:
            saved_prepass_recipe = upsert_canonical_edr_saved_recipe_impl(
                recipes_root=calibration_cache_root.parent / "prepass_recipes",
                dataset_id=str(completion_context["dataset_id"]),
                calibration_request=dict(completion_context["calibration_request"]),
                classifier_id=completion_context.get("resolved_classifier_id"),
                recipe_fingerprint=str(completion_context["recipe_fingerprint"]),
                canonical_recipe=payload,
                canonical_recipe_json=canonical_recipe_json.resolve(),
                canonical_recipe_md=canonical_recipe_md,
                report_bundle_json=effective_report,
                recipe_registry_entry=registry_entry,
                glossary_text=str(completion_context.get("glossary_text") or ""),
                prepass_schema_version=PREPASS_RECIPE_SCHEMA_VERSION,
                canonical_deployment=canonical_deployment,
                labelmap=_completion_labelmap(),
                edr_package=edr_package,
            )
        _ensure_edr_package(allow_materialize=discovery_run_root is not None)
        if persistence_status != "reused_existing":
            persistence_status = "registered_and_saved"

    summary = {
        "status": "completed",
        "persistence_status": persistence_status,
        "canonical_recipe_json": str(canonical_recipe_json.resolve()),
        "canonical_recipe_md": str(canonical_recipe_md.resolve()) if canonical_recipe_md else None,
        "report_bundle_json": str(effective_report.resolve()) if effective_report is not None else None,
        "recipe_registry_entry": registry_entry,
        "saved_prepass_recipe": saved_prepass_recipe,
        "saved_prepass_recipe_id": (
            str(saved_prepass_recipe.get("id"))
            if isinstance(saved_prepass_recipe, dict) and saved_prepass_recipe.get("id")
            else None
        ),
        "canonical_deployment_job": canonical_deployment,
        "canonical_deployment_job_id": (
            str(canonical_deployment.get("job_id"))
            if isinstance(canonical_deployment, dict) and canonical_deployment.get("job_id")
            else None
        ),
        "edr_package": edr_package,
        "edr_package_id": (
            str(edr_package.get("id"))
            if isinstance(edr_package, dict) and edr_package.get("id")
            else None
        ),
    }
    if write_summary and run_root is not None:
        _write_json_atomic(canonical_completion_summary_path(run_root), summary)
        summary["canonical_completion_summary_json"] = str(
            canonical_completion_summary_path(run_root).resolve()
        )
    else:
        summary["canonical_completion_summary_json"] = None
    return summary
