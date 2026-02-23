#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import xgboost as xgb


def _apply_log1p_counts(X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    if not feature_names:
        return X
    count_tokens = ("count", "support_count", "sam3_text_count", "sam3_sim_count")
    for idx, name in enumerate(feature_names):
        if any(token in name for token in count_tokens):
            X[:, idx] = np.log1p(np.maximum(X[:, idx], 0.0))
    return X


def _standardize(X: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    if mean is None or std is None:
        return X
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape[0] != X.shape[1] or std.shape[0] != X.shape[1]:
        return X
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _load_policy(path_or_json: Optional[str], meta: Dict[str, Any]) -> Dict[str, Any]:
    if path_or_json:
        raw = str(path_or_json).strip()
        if not raw:
            return {}
        maybe_path = Path(raw)
        payload = maybe_path.read_text(encoding="utf-8") if maybe_path.exists() and maybe_path.is_file() else raw
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --policy-json payload: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit("Invalid --policy-json payload: expected JSON object.")
        return parsed
    stored = meta.get("ensemble_policy")
    if isinstance(stored, dict):
        return stored
    return {}


def _load_optional_booster(path: Optional[Path]) -> Optional[xgb.Booster]:
    if path is None or not path.exists() or not path.is_file():
        return None
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def _resolve_model_path(raw_path: Optional[str], *, model_path: Path) -> Optional[Path]:
    if not raw_path:
        return None
    candidate = Path(str(raw_path))
    if candidate.exists():
        return candidate
    alt = model_path.parent / candidate
    if alt.exists():
        return alt
    return None


def _normalize_source_fields(payload: Dict[str, Any]) -> tuple[str, Set[str]]:
    primary = str(payload.get("score_source") or payload.get("source") or "unknown").strip().lower() or "unknown"
    source_set: Set[str] = set()
    raw = payload.get("source_list")
    if isinstance(raw, (list, tuple, set)):
        for src in raw:
            name = str(src or "").strip().lower()
            if name:
                source_set.add(name)
    elif isinstance(raw, str):
        name = raw.strip().lower()
        if name:
            source_set.add(name)
    source_set.add(primary)
    return primary, source_set


def _is_sam3_text_only(payload: Dict[str, Any]) -> bool:
    primary, source_set = _normalize_source_fields(payload)
    if primary != "sam3_text":
        return False
    return ("yolo" not in source_set) and ("rfdetr" not in source_set)


def _predict_probabilities(
    X: np.ndarray,
    *,
    parsed_meta: List[Dict[str, Any]],
    model_path: Path,
    meta: Dict[str, Any],
) -> np.ndarray:
    base_booster = xgb.Booster()
    base_booster.load_model(str(model_path))
    probs = np.asarray(base_booster.predict(xgb.DMatrix(X)), dtype=np.float32)

    split_cfg = meta.get("split_head") if isinstance(meta.get("split_head"), dict) else {}
    split_enabled = bool(split_cfg.get("enabled")) and str(split_cfg.get("route") or "detector_support") == "detector_support"
    if split_enabled:
        models = split_cfg.get("models") if isinstance(split_cfg.get("models"), dict) else {}
        det_booster = _load_optional_booster(
            _resolve_model_path(models.get("detector_supported"), model_path=model_path)
        )
        sam_booster = _load_optional_booster(_resolve_model_path(models.get("sam_only"), model_path=model_path))
        if det_booster is not None or sam_booster is not None:
            det_mask = np.asarray(
                [("yolo" in _normalize_source_fields(row)[1]) or ("rfdetr" in _normalize_source_fields(row)[1]) for row in parsed_meta],
                dtype=bool,
            )
            if det_booster is not None and det_mask.any():
                probs[det_mask] = det_booster.predict(xgb.DMatrix(X[det_mask]))
            if sam_booster is not None and (~det_mask).any():
                probs[~det_mask] = sam_booster.predict(xgb.DMatrix(X[~det_mask]))

    quality_cfg = meta.get("sam3_text_quality") if isinstance(meta.get("sam3_text_quality"), dict) else {}
    if bool(quality_cfg.get("enabled")):
        quality_booster = _load_optional_booster(
            _resolve_model_path(quality_cfg.get("model_path"), model_path=model_path)
        )
        if quality_booster is not None:
            alpha = _safe_float(quality_cfg.get("alpha"), 0.35)
            alpha = max(0.0, min(1.0, alpha))
            if alpha > 0.0:
                text_mask = np.asarray([_is_sam3_text_only(row) for row in parsed_meta], dtype=bool)
                if text_mask.any():
                    q_probs = np.asarray(quality_booster.predict(xgb.DMatrix(X[text_mask])), dtype=np.float32)
                    probs[text_mask] = np.asarray(
                        (1.0 - alpha) * probs[text_mask] + alpha * q_probs,
                        dtype=np.float32,
                    )

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


def _policy_bias_by_source_label(policy: Dict[str, Any], *, source: str, label: str) -> float:
    mapping = policy.get("logit_bias_by_source_class")
    if not isinstance(mapping, dict):
        return 0.0
    source_map = mapping.get(str(source or "").strip().lower())
    if not isinstance(source_map, dict):
        return 0.0
    key = str(label or "").strip().lower()
    if key in source_map:
        return _safe_float(source_map.get(key), 0.0)
    if "__default__" in source_map:
        return _safe_float(source_map.get("__default__"), 0.0)
    if "*" in source_map:
        return _safe_float(source_map.get("*"), 0.0)
    return 0.0


def _apply_logit_shift(prob: float, bias: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    logit = math.log(p / (1.0 - p))
    shifted = logit + float(bias)
    return 1.0 / (1.0 + math.exp(-shifted))


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
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _build_detector_support_index(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
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
        support.setdefault(image, []).append(
            {
                "label": label,
                "bbox_xyxy_px": [float(v) for v in bbox[:4]],
            }
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates with ensemble XGBoost.")
    parser.add_argument("--model", required=True, help="Model .json path.")
    parser.add_argument("--meta", required=True, help="Model meta json.")
    parser.add_argument("--data", required=True, help="Input .npz data.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold.")
    parser.add_argument(
        "--policy-json",
        type=str,
        default=None,
        help="Optional policy JSON file/string (defaults to meta['ensemble_policy'] when absent).",
    )
    args = parser.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    thresholds_by_label = meta.get("calibrated_thresholds_objective")
    if not isinstance(thresholds_by_label, dict):
        thresholds_by_label = meta.get("calibrated_thresholds_relaxed_smoothed")
    if not isinstance(thresholds_by_label, dict):
        thresholds_by_label = (
            meta.get("calibrated_thresholds_relaxed")
            if isinstance(meta.get("calibrated_thresholds_relaxed"), dict)
            else {}
        )
    if not isinstance(thresholds_by_label, dict):
        thresholds_by_label = (
            meta.get("calibrated_thresholds") if isinstance(meta.get("calibrated_thresholds"), dict) else {}
        )
    default_threshold = float(meta.get("calibrated_threshold") or 0.5)
    threshold_override = float(args.threshold) if args.threshold is not None else None
    policy = _load_policy(args.policy_json, meta)

    sam_floor_default = _safe_float(policy.get("sam_only_min_prob_default"), 0.0)
    sam_floor_map = policy.get("sam_only_min_prob_by_class")
    consensus_default = _safe_float(policy.get("consensus_iou_default"), 0.0)
    consensus_map = policy.get("consensus_iou_by_class")
    consensus_class_aware = bool(policy.get("consensus_class_aware", True))
    threshold_override_map = policy.get("threshold_by_class_override")

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    meta_rows = list(data["meta"])
    feature_names = [str(name) for name in data.get("feature_names", [])]

    if meta.get("log1p_counts"):
        X = _apply_log1p_counts(X, feature_names)
    X = _standardize(X, meta.get("feature_mean"), meta.get("feature_std"))

    parsed_meta = []
    for row in meta_rows:
        try:
            parsed_meta.append(json.loads(str(row)))
        except json.JSONDecodeError:
            parsed_meta.append({})
    probs = _predict_probabilities(X, parsed_meta=parsed_meta, model_path=Path(args.model), meta=meta)

    detector_support = _build_detector_support_index(parsed_meta)

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, entry in enumerate(parsed_meta):
            label = str(entry.get("label") or "").strip().lower()
            image = str(entry.get("image") or "")
            primary_source, source_list = _normalize_source_fields(entry)
            bbox = entry.get("bbox_xyxy_px")

            prob_raw = _safe_float(probs[idx], 0.0)
            prob_adj = float(prob_raw)
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

            has_detector_support = ("yolo" in source_list) or ("rfdetr" in source_list)
            is_sam_primary = primary_source in {"sam3_text", "sam3_similarity"}
            is_sam_only = is_sam_primary and not has_detector_support

            blocked_reason: Optional[str] = None
            if is_sam_only:
                sam_floor = _policy_value_by_class(sam_floor_map, label=label, default=sam_floor_default)
                if sam_floor > 0.0 and prob_adj < sam_floor:
                    blocked_reason = "sam_only_floor"
            if blocked_reason is None and is_sam_only:
                consensus_iou = _policy_value_by_class(consensus_map, label=label, default=consensus_default)
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

            scored = dict(entry)
            scored["ensemble_prob_raw"] = float(prob_raw)
            scored["ensemble_prob"] = float(prob_adj)
            scored["ensemble_accept"] = bool(ensemble_accept)
            scored["ensemble_threshold"] = float(thr)
            scored["ensemble_policy_blocked"] = bool(blocked_reason and blocked_reason != "threshold")
            scored["ensemble_policy_block_reason"] = str(blocked_reason) if blocked_reason else None
            f.write(json.dumps(scored, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
