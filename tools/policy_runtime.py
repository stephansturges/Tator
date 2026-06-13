#!/usr/bin/env python3
"""Shared runtime helpers for base XGB + learned policy-layer scoring."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.policy_layer_features import build_policy_feature_matrix


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _apply_log1p_counts(X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    if not feature_names:
        return X
    count_tokens = ("count", "support_count", "sam3_text_count", "sam3_sim_count")
    out = np.asarray(X, dtype=np.float32).copy()
    for idx, name in enumerate(feature_names):
        if any(token in name for token in count_tokens):
            out[:, idx] = np.log1p(np.maximum(out[:, idx], 0.0))
    return out


def _standardize(X: np.ndarray, mean: Optional[List[float]], std: Optional[List[float]]) -> np.ndarray:
    if mean is None or std is None:
        return X
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    if mean_arr.shape[0] != X.shape[1] or std_arr.shape[0] != X.shape[1]:
        return X
    std_arr = np.where(std_arr < 1e-6, 1.0, std_arr)
    return (X - mean_arr) / std_arr


def _normalize_source_fields(row: Dict[str, Any]) -> Tuple[str, Set[str]]:
    primary = str(row.get("score_source") or row.get("source") or "unknown").strip().lower() or "unknown"
    source_set: Set[str] = set()
    raw = row.get("source_list")
    if isinstance(raw, (list, tuple, set)):
        for src in raw:
            name = str(src or "").strip().lower()
            if name:
                source_set.add(name)
    elif isinstance(raw, str):
        name = raw.strip().lower()
        if name:
            source_set.add(name)
    raw_scores = row.get("score_by_source")
    if isinstance(raw_scores, dict):
        for src in raw_scores.keys():
            name = str(src or "").strip().lower()
            if name:
                source_set.add(name)
    source_set.add(primary)
    return primary, source_set


def _has_detector_support(row: Dict[str, Any]) -> bool:
    _, sources = _normalize_source_fields(row)
    return ("yolo" in sources) or ("rfdetr" in sources)


def _is_sam3_text_only(row: Dict[str, Any]) -> bool:
    primary, sources = _normalize_source_fields(row)
    return primary == "sam3_text" and ("yolo" not in sources) and ("rfdetr" not in sources)


def _is_sam3_similarity_only(row: Dict[str, Any]) -> bool:
    primary, sources = _normalize_source_fields(row)
    return primary == "sam3_similarity" and ("yolo" not in sources) and ("rfdetr" not in sources)


def _blend_probabilities(base: np.ndarray, aux: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(max(0.0, min(1.0, alpha)))
    if alpha <= 0.0:
        return np.asarray(base, dtype=np.float32)
    if alpha >= 1.0:
        return np.asarray(aux, dtype=np.float32)
    return np.asarray((1.0 - alpha) * base + alpha * aux, dtype=np.float32)


def _resolve_model_path(raw_path: Optional[str], *, base_dir: Path) -> Optional[Path]:
    if not raw_path:
        return None
    candidate = Path(str(raw_path))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    alt = (base_dir / candidate).resolve()
    if alt.exists():
        return alt
    return None


def _load_optional_booster(path: Optional[Path]) -> Optional[xgb.Booster]:
    if path is None or not path.exists() or not path.is_file():
        return None
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def transform_base_features(X: np.ndarray, feature_names: Sequence[str], meta: Dict[str, Any]) -> np.ndarray:
    out = np.asarray(X, dtype=np.float32)
    if meta.get("log1p_counts"):
        out = _apply_log1p_counts(out, [str(name) for name in feature_names])
    return _standardize(out, meta.get("feature_mean"), meta.get("feature_std"))


def predict_base_probabilities(
    X_base: np.ndarray,
    *,
    meta_rows: Sequence[Dict[str, Any]],
    model_path: Path,
    meta: Dict[str, Any],
) -> np.ndarray:
    base_dir = model_path.parent
    base_booster = xgb.Booster()
    base_booster.load_model(str(model_path))
    probs = np.asarray(base_booster.predict(xgb.DMatrix(X_base)), dtype=np.float32)

    split_cfg = meta.get("split_head") if isinstance(meta.get("split_head"), dict) else {}
    split_enabled = bool(split_cfg.get("enabled")) and str(split_cfg.get("route") or "detector_support") == "detector_support"
    if split_enabled:
        models = split_cfg.get("models") if isinstance(split_cfg.get("models"), dict) else {}
        det_booster = _load_optional_booster(_resolve_model_path(models.get("detector_supported"), base_dir=base_dir))
        sam_booster = _load_optional_booster(_resolve_model_path(models.get("sam_only"), base_dir=base_dir))
        det_mask = np.asarray([_has_detector_support(row) for row in meta_rows], dtype=bool)
        if det_booster is not None and det_mask.any():
            probs[det_mask] = det_booster.predict(xgb.DMatrix(X_base[det_mask]))
        if sam_booster is not None and (~det_mask).any():
            probs[~det_mask] = sam_booster.predict(xgb.DMatrix(X_base[~det_mask]))

    quality_cfg = meta.get("sam3_text_quality") if isinstance(meta.get("sam3_text_quality"), dict) else {}
    if bool(quality_cfg.get("enabled")):
        quality_booster = _load_optional_booster(_resolve_model_path(quality_cfg.get("model_path"), base_dir=base_dir))
        if quality_booster is not None:
            alpha = _safe_float(quality_cfg.get("alpha"), 0.35)
            alpha = max(0.0, min(1.0, alpha))
            if alpha > 0.0:
                text_mask = np.asarray([_is_sam3_text_only(row) for row in meta_rows], dtype=bool)
                if text_mask.any():
                    q_probs = np.asarray(quality_booster.predict(xgb.DMatrix(X_base[text_mask])), dtype=np.float32)
                    probs[text_mask] = _blend_probabilities(probs[text_mask], q_probs, alpha)
    similarity_quality_cfg = meta.get("sam3_similarity_quality") if isinstance(meta.get("sam3_similarity_quality"), dict) else {}
    if bool(similarity_quality_cfg.get("enabled")):
        similarity_booster = _load_optional_booster(
            _resolve_model_path(similarity_quality_cfg.get("model_path"), base_dir=base_dir)
        )
        if similarity_booster is not None:
            alpha = _safe_float(similarity_quality_cfg.get("alpha"), 0.35)
            alpha = max(0.0, min(1.0, alpha))
            if alpha > 0.0:
                sim_mask = np.asarray([_is_sam3_similarity_only(row) for row in meta_rows], dtype=bool)
                if sim_mask.any():
                    q_probs = np.asarray(similarity_booster.predict(xgb.DMatrix(X_base[sim_mask])), dtype=np.float32)
                    probs[sim_mask] = _blend_probabilities(probs[sim_mask], q_probs, alpha)
    return probs


def _policy_value_by_class(policy_map: Any, *, label: str, default: float) -> float:
    if not isinstance(policy_map, dict):
        return float(default)
    key = str(label or "").strip().lower()
    if key in policy_map:
        return _safe_float(policy_map.get(key), default)
    if "__default__" in policy_map:
        return _safe_float(policy_map.get("__default__"), default)
    if "*" in policy_map:
        return _safe_float(policy_map.get("*"), default)
    return float(default)


def _policy_optional_value_by_class(policy_map: Any, *, label: str) -> Optional[float]:
    if not isinstance(policy_map, dict):
        return None
    key = str(label or "").strip().lower()
    if key in policy_map:
        return _safe_float(policy_map.get(key), 0.0)
    if "__default__" in policy_map:
        return _safe_float(policy_map.get("__default__"), 0.0)
    if "*" in policy_map:
        return _safe_float(policy_map.get("*"), 0.0)
    return None


def _resolve_consensus_iou(
    *,
    label: str,
    primary_source: str,
    consensus_default: float,
    consensus_by_class: Any,
    consensus_by_source_class: Any,
) -> float:
    source_key = str(primary_source or "").strip().lower()
    if isinstance(consensus_by_source_class, dict):
        source_map = consensus_by_source_class.get(source_key)
        source_val = _policy_optional_value_by_class(source_map, label=label)
        if source_val is not None:
            return max(0.0, min(1.0, float(source_val)))
        fallback_map = consensus_by_source_class.get("__default__")
        fallback_val = _policy_optional_value_by_class(fallback_map, label=label)
        if fallback_val is not None:
            return max(0.0, min(1.0, float(fallback_val)))
    class_val = _policy_optional_value_by_class(consensus_by_class, label=label)
    if class_val is not None:
        return max(0.0, min(1.0, float(class_val)))
    return max(0.0, min(1.0, float(consensus_default)))


def _policy_bias_by_source_label(policy: Dict[str, Any], *, source: str, label: str) -> float:
    mapping = policy.get("logit_bias_by_source_class")
    if not isinstance(mapping, dict):
        return 0.0
    source_map = mapping.get(str(source or "").strip().lower())
    if not isinstance(source_map, dict):
        return 0.0
    return _policy_value_by_class(source_map, label=label, default=0.0)


def _should_apply_source_bias(policy: Dict[str, Any], *, primary_source: str, has_detector_support: bool) -> bool:
    scope = str(policy.get("sam_bias_scope") or "primary_source").strip().lower()
    if scope == "sam_only":
        return str(primary_source or "").strip().lower() in {"sam3_text", "sam3_similarity"} and (not bool(has_detector_support))
    return True


def _apply_logit_shift(prob: float, bias: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    logit = math.log(p / (1.0 - p))
    shifted = logit + float(bias)
    return float(1.0 / (1.0 + math.exp(-shifted)))


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _build_detector_support_index(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    support: Dict[str, List[Dict[str, Any]]] = {}
    for payload in rows:
        image = str(payload.get("image") or "")
        label = str(payload.get("label") or "").strip().lower()
        bbox = payload.get("bbox_xyxy_px")
        if not image or not label or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        primary, source_list = _normalize_source_fields(payload)
        if primary == "unknown":
            source_list.discard("unknown")
        if ("yolo" not in source_list) and ("rfdetr" not in source_list):
            continue
        support.setdefault(image, []).append({"label": label, "bbox_xyxy_px": [float(v) for v in bbox[:4]]})
    return support


def _has_detector_consensus(
    support_index: Dict[str, List[Dict[str, Any]]],
    *,
    image: str,
    label: str,
    bbox: Sequence[float],
    iou_thr: float,
    class_aware: bool,
) -> bool:
    if iou_thr <= 0.0:
        return True
    for det in support_index.get(image, []):
        det_bbox = det.get("bbox_xyxy_px")
        if not isinstance(det_bbox, (list, tuple)) or len(det_bbox) < 4:
            continue
        if class_aware:
            det_label = str(det.get("label") or "").strip().lower()
            if det_label and det_label != label:
                continue
        if _iou(bbox, det_bbox) >= iou_thr:
            return True
    return False


def resolve_thresholds(meta: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    thresholds_by_label = meta.get("calibrated_thresholds_objective")
    if not isinstance(thresholds_by_label, dict):
        thresholds_by_label = meta.get("calibrated_thresholds_relaxed_smoothed")
    if not isinstance(thresholds_by_label, dict):
        thresholds_by_label = meta.get("calibrated_thresholds_relaxed") if isinstance(meta.get("calibrated_thresholds_relaxed"), dict) else {}
    if not isinstance(thresholds_by_label, dict):
        thresholds_by_label = meta.get("calibrated_thresholds") if isinstance(meta.get("calibrated_thresholds"), dict) else {}
    default_threshold = float(meta.get("calibrated_threshold") or 0.5)
    return default_threshold, {str(k): float(v) for k, v in thresholds_by_label.items()}


def apply_hand_policy(
    *,
    probs: Sequence[float],
    meta_rows: Sequence[Dict[str, Any]],
    policy: Dict[str, Any],
    default_threshold: float,
    thresholds_by_label: Dict[str, float],
    threshold_override: Optional[float] = None,
) -> List[Dict[str, Any]]:
    sam_floor_default = _safe_float(policy.get("sam_only_min_prob_default"), 0.0)
    sam_floor_map = policy.get("sam_only_min_prob_by_class")
    consensus_default = _safe_float(policy.get("consensus_iou_default"), 0.0)
    consensus_map = policy.get("consensus_iou_by_class")
    consensus_source_map = policy.get("consensus_iou_by_source_class")
    consensus_class_aware = bool(policy.get("consensus_class_aware", True))
    threshold_override_map = policy.get("threshold_by_class_override")
    detector_support = _build_detector_support_index(meta_rows)

    results: List[Dict[str, Any]] = []
    for idx, entry in enumerate(meta_rows):
        label = str(entry.get("label") or "").strip().lower()
        image = str(entry.get("image") or "")
        primary_source, source_list = _normalize_source_fields(entry)
        bbox = entry.get("bbox_xyxy_px")
        has_detector_support = ("yolo" in source_list) or ("rfdetr" in source_list)
        is_sam_primary = primary_source in {"sam3_text", "sam3_similarity"}
        is_sam_only = is_sam_primary and not has_detector_support
        prob_raw = _safe_float(probs[idx], 0.0)
        prob_adj = float(prob_raw)
        if _should_apply_source_bias(policy, primary_source=primary_source, has_detector_support=has_detector_support):
            bias = _policy_bias_by_source_label(policy, source=primary_source, label=label)
            if abs(bias) > 1e-12:
                prob_adj = _apply_logit_shift(prob_adj, bias)
        if threshold_override is not None:
            thr = float(threshold_override)
        elif label and label in thresholds_by_label:
            thr = float(thresholds_by_label[label])
        else:
            thr = float(default_threshold)
        if threshold_override is None:
            thr = _policy_value_by_class(threshold_override_map, label=label, default=thr)
        blocked_reason: Optional[str] = None
        if is_sam_only:
            sam_floor = _policy_value_by_class(sam_floor_map, label=label, default=sam_floor_default)
            if sam_floor > 0.0 and prob_adj < sam_floor:
                blocked_reason = "sam_only_floor"
        if blocked_reason is None and is_sam_only:
            consensus_iou = _resolve_consensus_iou(
                label=label,
                primary_source=primary_source,
                consensus_default=consensus_default,
                consensus_by_class=consensus_map,
                consensus_by_source_class=consensus_source_map,
            )
            if consensus_iou > 0.0 and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                if not _has_detector_consensus(
                    detector_support,
                    image=image,
                    label=label,
                    bbox=[float(v) for v in bbox[:4]],
                    iou_thr=consensus_iou,
                    class_aware=consensus_class_aware,
                ):
                    blocked_reason = "consensus"
        ensemble_accept = (blocked_reason is None) and (prob_adj >= float(thr))
        if blocked_reason is None and not ensemble_accept:
            blocked_reason = "threshold"
        results.append(
            {
                "prob_raw": float(prob_raw),
                "prob": float(prob_adj),
                "accept": bool(ensemble_accept),
                "threshold": float(thr),
                "blocked_reason": blocked_reason,
            }
        )
    return results


def load_selected_policy(
    meta: Dict[str, Any],
    *,
    base_dir: Path,
    meta_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    selected = meta.get("selected_policy_layer")
    if not isinstance(selected, dict):
        return None
    variant = str(selected.get("variant") or "").strip().lower()
    model_path = None
    meta_path = None
    if meta_dir is not None:
        model_path = _resolve_model_path(selected.get("model_path"), base_dir=meta_dir)
        meta_path = _resolve_model_path(selected.get("meta_path"), base_dir=meta_dir)
    if model_path is None:
        model_path = _resolve_model_path(selected.get("model_path"), base_dir=base_dir)
    if meta_path is None:
        meta_path = _resolve_model_path(selected.get("meta_path"), base_dir=base_dir)
    if not variant or model_path is None or meta_path is None:
        return None
    try:
        policy_meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    return {
        "variant": variant,
        "model_path": model_path,
        "meta_path": meta_path,
        "meta": policy_meta,
    }


def apply_selected_policy(
    *,
    X_full: np.ndarray,
    feature_names_full: Sequence[str],
    meta_rows: Sequence[Dict[str, Any]],
    base_probs: Sequence[float],
    selected_policy: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    policy_meta = selected_policy["meta"] if isinstance(selected_policy.get("meta"), dict) else {}
    bundle = build_policy_feature_matrix(
        X_full,
        feature_names_full,
        meta_rows,
        base_probs,
        anchor_similarity=policy_meta.get("anchor_similarity"),
    )
    X_policy = np.asarray(bundle["X"], dtype=np.float32)
    feature_names = list(bundle["feature_names"])
    schema_hash = str(bundle["feature_schema_hash"])
    meta = policy_meta
    expected_hash = str(meta.get("feature_schema_hash") or "")
    if expected_hash and expected_hash != schema_hash:
        raise RuntimeError(f"policy_feature_schema_mismatch:{expected_hash}!={schema_hash}")
    variant = str(selected_policy.get("variant") or "").strip().lower()
    if variant == "lreg":
        payload = joblib.load(selected_policy["model_path"])
        model = payload["model"]
        mean = np.asarray(payload.get("feature_mean"), dtype=np.float32)
        std = np.asarray(payload.get("feature_std"), dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        X_scaled = (X_policy - mean) / std
        delta = np.asarray(model.decision_function(X_scaled), dtype=np.float32)
        base_logits = np.asarray([math.log(max(min(float(p), 1.0 - 1e-6), 1e-6) / (1.0 - max(min(float(p), 1.0 - 1e-6), 1e-6))) for p in base_probs], dtype=np.float32)
        final_logits = base_logits + delta
        final_probs = 1.0 / (1.0 + np.exp(-final_logits))
    elif variant == "xgb":
        booster = xgb.Booster()
        booster.load_model(str(selected_policy["model_path"]))
        final_probs = np.asarray(booster.predict(xgb.DMatrix(X_policy)), dtype=np.float32)
    else:
        raise RuntimeError(f"unsupported_policy_variant:{variant}")
    return final_probs.astype(np.float32), {"feature_names": feature_names, "feature_schema_hash": schema_hash, "bundle": bundle}
