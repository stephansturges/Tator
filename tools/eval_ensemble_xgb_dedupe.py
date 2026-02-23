#!/usr/bin/env python
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import xgboost as xgb


def _yolo_to_xyxy(bbox: Sequence[float], w: int, h: int) -> List[float]:
    cx, cy, bw, bh = bbox
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return [x1, y1, x2, y2]


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


def _match_class(gt_boxes: List[List[float]], pred_boxes: List[List[float]], iou_thr: float) -> Tuple[int, int, int]:
    matched_gt = set()
    tp = 0
    fp = 0
    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = None
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            cur = _iou(pred, gt)
            if cur > best_iou:
                best_iou = cur
                best_idx = idx
        if best_idx is not None and best_iou >= iou_thr:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = max(0, len(gt_boxes) - len(matched_gt))
    return tp, fp, fn


def _merge_prepass_detections(
    detections: List[Dict[str, Any]],
    *,
    iou_thr: float,
    fusion_mode: str = "primary",
    source_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    if not detections or iou_thr <= 0:
        return detections, 0

    fusion_mode = str(fusion_mode or "primary").strip().lower()
    source_weights = {
        str(src or "").strip().lower(): float(weight)
        for src, weight in (source_weights or {}).items()
        if str(src or "").strip()
    }

    def _det_score_raw(det: Dict[str, Any]) -> float:
        try:
            return float(det.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _score_by_source(det: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        raw_map = det.get("score_by_source")
        if isinstance(raw_map, dict):
            for raw_src, raw_score in raw_map.items():
                src = str(raw_src or "").strip().lower()
                if not src:
                    continue
                try:
                    score_val = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if src not in out or score_val > out[src]:
                    out[src] = score_val
        primary = str(det.get("score_source") or det.get("source") or "").strip().lower()
        if primary:
            primary_score = _det_score_raw(det)
            if primary not in out or primary_score > out[primary]:
                out[primary] = primary_score
        return out

    def _det_score(det: Dict[str, Any]) -> float:
        if fusion_mode != "source_weighted":
            return _det_score_raw(det)
        score_map = _score_by_source(det)
        if not score_map:
            return _det_score_raw(det)
        weighted_sum = 0.0
        weight_sum = 0.0
        src_set: Set[str] = set()
        for src, score in score_map.items():
            src_name = str(src or "").strip().lower()
            if not src_name:
                continue
            src_set.add(src_name)
            w = float(source_weights.get(src_name, 1.0))
            if not math.isfinite(w) or w <= 0.0:
                continue
            weighted_sum += w * float(score)
            weight_sum += w
        if weight_sum <= 0:
            weighted = _det_score_raw(det)
        else:
            weighted = weighted_sum / weight_sum
        detector_supported = ("yolo" in src_set) or ("rfdetr" in src_set)
        support_bonus = 0.01 * max(0, len(src_set) - 1)
        detector_bonus = 0.02 if detector_supported else 0.0
        return float(weighted + support_bonus + detector_bonus)

    merged: List[Dict[str, Any]] = []
    removed = 0
    ordered = sorted(detections, key=_det_score, reverse=True)
    for det in ordered:
        label = str(det.get("label") or det.get("class_name") or "").strip()
        box = det.get("bbox_xyxy_px")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            merged.append(det)
            continue
        matched_idx = None
        for idx, kept in enumerate(merged):
            kept_label = str(kept.get("label") or kept.get("class_name") or "").strip()
            if label and kept_label and label != kept_label:
                continue
            kept_box = kept.get("bbox_xyxy_px")
            if not isinstance(kept_box, (list, tuple)) or len(kept_box) < 4:
                continue
            if _iou(box, kept_box) >= iou_thr:
                matched_idx = idx
                break
        if matched_idx is None:
            merged.append(dict(det))
        else:
            kept = merged[matched_idx]
            keep_det = kept
            if _det_score(det) > _det_score(kept):
                keep_det = dict(det)
            merged[matched_idx] = keep_det
            removed += 1
    return merged, removed


def _filter_scoreless_detections(
    detections: List[Dict[str, Any]],
    *,
    iou_thr: float,
) -> Tuple[List[Dict[str, Any]], int]:
    if not detections or iou_thr <= 0:
        return detections, 0
    anchors = [
        det
        for det in detections
        if det.get("score") is not None and (det.get("score_source") or det.get("source")) != "unknown"
    ]
    if not anchors:
        return detections, 0
    filtered: List[Dict[str, Any]] = []
    removed = 0
    for det in detections:
        score = det.get("score")
        score_source = det.get("score_source") or det.get("source") or "unknown"
        if score is None or score_source == "unknown":
            bbox = det.get("bbox_xyxy_px") or []
            has_overlap = False
            for anchor in anchors:
                anchor_bbox = anchor.get("bbox_xyxy_px") or []
                if _iou(bbox, anchor_bbox) >= iou_thr:
                    has_overlap = True
                    break
            if not has_overlap:
                removed += 1
                continue
        filtered.append(det)
    return filtered, removed


def _apply_log1p_counts(X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    if not feature_names:
        return X
    count_tokens = ("count", "support_count", "sam3_text_count", "sam3_sim_count")
    for idx, name in enumerate(feature_names):
        if any(token in name for token in count_tokens):
            X[:, idx] = np.log1p(np.maximum(X[:, idx], 0.0))
    return X


def _standardize(X: np.ndarray, mean: Optional[List[float]], std: Optional[List[float]]) -> np.ndarray:
    if mean is None or std is None:
        return X
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape[0] != X.shape[1] or std.shape[0] != X.shape[1]:
        return X
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std


def _compute_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _parse_grid(raw: Optional[str], fallback: float) -> List[float]:
    if not raw:
        return [float(fallback)]
    vals: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            vals.append(float(token))
        except ValueError:
            continue
    return vals or [float(fallback)]


def _load_policy(path_or_json: Optional[str]) -> Dict[str, Any]:
    if not path_or_json:
        return {}
    raw = str(path_or_json).strip()
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _clamp_prob(prob: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, float(prob)))


def _apply_logit_shift(prob: float, bias: float) -> float:
    p = _clamp_prob(prob)
    logit = math.log(p / (1.0 - p))
    shifted = logit + float(bias)
    if shifted >= 0:
        z = math.exp(-shifted)
        return 1.0 / (1.0 + z)
    z = math.exp(shifted)
    return z / (1.0 + z)


def _policy_value_by_class(
    policy_map: Any,
    *,
    label: str,
    default: float,
) -> float:
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
    src = str(source or "").strip().lower()
    lbl = str(label or "").strip().lower()
    src_map = mapping.get(src)
    if isinstance(src_map, dict):
        if lbl in src_map:
            return _safe_float(src_map.get(lbl), 0.0)
        if "__default__" in src_map:
            return _safe_float(src_map.get("__default__"), 0.0)
        if "*" in src_map:
            return _safe_float(src_map.get("*"), 0.0)
    global_map = mapping.get("__default__")
    if isinstance(global_map, dict):
        if lbl in global_map:
            return _safe_float(global_map.get(lbl), 0.0)
        if "__default__" in global_map:
            return _safe_float(global_map.get("__default__"), 0.0)
        if "*" in global_map:
            return _safe_float(global_map.get("*"), 0.0)
    return 0.0


def _load_source_weights(raw: Optional[str], policy: Dict[str, Any]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    policy_weights = policy.get("source_weights")
    if isinstance(policy_weights, dict):
        for src, weight in policy_weights.items():
            src_name = str(src or "").strip().lower()
            if not src_name:
                continue
            merged[src_name] = _safe_float(weight, 1.0)
    if raw:
        parsed: Dict[str, Any] = {}
        path = Path(str(raw).strip())
        if path.exists():
            try:
                parsed = json.loads(path.read_text())
            except Exception:
                parsed = {}
        else:
            try:
                parsed = json.loads(str(raw))
            except Exception:
                parsed = {}
        if isinstance(parsed, dict):
            for src, weight in parsed.items():
                src_name = str(src or "").strip().lower()
                if not src_name:
                    continue
                merged[src_name] = _safe_float(weight, 1.0)
    return merged


def _build_detector_support_index(
    raw_detector_by_source: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    union_map = raw_detector_by_source.get("yolo_rfdetr_union") or {}
    for image, dets in union_map.items():
        by_label: Dict[str, List[List[float]]] = defaultdict(list)
        any_label: List[List[float]] = []
        for det in dets or []:
            if not isinstance(det, dict):
                continue
            label = str(det.get("label") or "").strip().lower()
            bbox = det.get("bbox_xyxy_px")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            box = [float(v) for v in bbox[:4]]
            any_label.append(box)
            if label:
                by_label[label].append(box)
        out[str(image)] = {"any": any_label, "by_label": dict(by_label)}
    return out


def _has_detector_consensus(
    detector_index: Dict[str, Dict[str, Any]],
    *,
    image: str,
    label: str,
    bbox: Sequence[float],
    iou_thr: float,
    class_aware: bool,
) -> bool:
    image_index = detector_index.get(str(image)) or {}
    if class_aware:
        boxes = (image_index.get("by_label") or {}).get(str(label or "").strip().lower(), [])
    else:
        boxes = image_index.get("any") or []
    if not boxes:
        return False
    for det_box in boxes:
        if _iou(bbox, det_box) >= iou_thr:
            return True
    return False


def _compute_geometry_stats(
    gt_by_image: Dict[str, Dict[int, List[List[float]]]],
    image_sizes: Dict[str, Tuple[int, int]],
    *,
    labelmap: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    values: Dict[str, Dict[str, List[float]]] = {
        str(lbl).strip().lower(): {"log_area": [], "log_aspect": []}
        for lbl in labelmap
    }
    for image, by_class in gt_by_image.items():
        size = image_sizes.get(image)
        if not size:
            continue
        img_w, img_h = size
        if img_w <= 0 or img_h <= 0:
            continue
        denom = float(img_w) * float(img_h)
        for cat_id, gt_boxes in by_class.items():
            if cat_id < 0 or cat_id >= len(labelmap):
                continue
            label = str(labelmap[cat_id]).strip().lower()
            for box in gt_boxes:
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0.0 or h <= 0.0:
                    continue
                area_norm = max((w * h) / denom, 1e-12)
                aspect = max(w / h, 1e-12)
                values[label]["log_area"].append(math.log(area_norm))
                values[label]["log_aspect"].append(math.log(aspect))
    stats: Dict[str, Dict[str, float]] = {}
    for label, payload in values.items():
        area_vals = payload["log_area"]
        aspect_vals = payload["log_aspect"]
        if not area_vals or not aspect_vals:
            continue
        area_arr = np.asarray(area_vals, dtype=np.float64)
        aspect_arr = np.asarray(aspect_vals, dtype=np.float64)
        stats[label] = {
            "area_mu": float(area_arr.mean()),
            "area_sigma": float(max(area_arr.std(), 1e-6)),
            "aspect_mu": float(aspect_arr.mean()),
            "aspect_sigma": float(max(aspect_arr.std(), 1e-6)),
        }
    return stats


def _normalize_source_fields(payload: Dict[str, Any]) -> Tuple[str, List[str]]:
    primary = str(payload.get("score_source") or payload.get("source") or "unknown").strip().lower()
    if not primary:
        primary = "unknown"
    src_set: Set[str] = set()
    raw_sources = payload.get("source_list")
    if isinstance(raw_sources, (list, tuple)):
        for src in raw_sources:
            src_name = str(src or "").strip().lower()
            if src_name:
                src_set.add(src_name)
    src_set.add(primary)
    return primary, sorted(src_set)


def _has_detector_support(payload: Dict[str, Any]) -> bool:
    _, source_list = _normalize_source_fields(payload)
    source_set = {str(src or "").strip().lower() for src in source_list}
    return ("yolo" in source_set) or ("rfdetr" in source_set)


def _is_sam3_text_only(payload: Dict[str, Any]) -> bool:
    primary, source_list = _normalize_source_fields(payload)
    if primary != "sam3_text":
        return False
    source_set = {str(src or "").strip().lower() for src in source_list}
    return ("yolo" not in source_set) and ("rfdetr" not in source_set)


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


def _load_optional_booster(path: Optional[Path]) -> Optional[xgb.Booster]:
    if path is None or not path.exists() or not path.is_file():
        return None
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


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
            det_mask = np.asarray([_has_detector_support(row) for row in parsed_meta], dtype=bool)
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


def _evaluate_predictions(
    pred_by_image: Dict[str, List[Dict[str, Any]]],
    gt_by_image: Dict[str, Dict[int, List[List[float]]]],
    *,
    name_to_cat: Dict[str, int],
    dedupe_iou: float,
    eval_iou: float,
    scoreless_iou: float,
    fusion_mode: str = "primary",
    source_weights: Optional[Dict[str, float]] = None,
    apply_dedupe: bool = True,
) -> Dict[str, Any]:
    tp = fp = fn = 0
    for image, gt_by_class in gt_by_image.items():
        image_preds = list(pred_by_image.get(image, []))
        if apply_dedupe:
            image_preds, _ = _merge_prepass_detections(
                image_preds,
                iou_thr=dedupe_iou,
                fusion_mode=fusion_mode,
                source_weights=source_weights,
            )
            if scoreless_iou > 0:
                image_preds, _ = _filter_scoreless_detections(image_preds, iou_thr=scoreless_iou)
        pred_by_class: Dict[int, List[List[float]]] = {}
        unknown = 0
        for det in image_preds:
            label = str(det.get("label") or "").strip().lower()
            cat_id = name_to_cat.get(label)
            if cat_id is None:
                unknown += 1
                continue
            pred_by_class.setdefault(cat_id, []).append(det["bbox_xyxy_px"])
        for cat_id, gt_boxes in gt_by_class.items():
            preds = pred_by_class.get(cat_id, [])
            ctp, cfp, cfn = _match_class(gt_boxes, preds, eval_iou)
            tp += ctp
            fp += cfp
            fn += cfn
        fp += unknown
    return _compute_metrics(tp, fp, fn)


def _filter_by_sources(
    pred_by_image: Dict[str, List[Dict[str, Any]]],
    *,
    include_sources: Set[str],
    attributed: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for image, dets in pred_by_image.items():
        keep: List[Dict[str, Any]] = []
        for det in dets:
            primary = str(det.get("score_source") or det.get("source") or "unknown").strip().lower()
            source_list_raw = det.get("source_list")
            source_list = set()
            if isinstance(source_list_raw, (list, tuple)):
                source_list = {str(src or "").strip().lower() for src in source_list_raw if str(src or "").strip()}
            source_list.add(primary)
            if attributed:
                if source_list & include_sources:
                    keep.append(det)
            else:
                if primary in include_sources:
                    keep.append(det)
        if keep:
            out[image] = keep
    return out


def _count_by_sources(pred_by_image: Dict[str, List[Dict[str, Any]]], *, attributed: bool) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for dets in pred_by_image.values():
        for det in dets:
            primary = str(det.get("score_source") or det.get("source") or "unknown").strip().lower() or "unknown"
            if attributed:
                source_list_raw = det.get("source_list")
                source_list = set()
                if isinstance(source_list_raw, (list, tuple)):
                    source_list = {
                        str(src or "").strip().lower() for src in source_list_raw if str(src or "").strip()
                    }
                source_list.add(primary)
                for src in source_list:
                    counts[src] = counts.get(src, 0) + 1
            else:
                counts[primary] = counts.get(primary, 0) + 1
    return counts


def _candidate_recall_upper_bound(
    pred_by_image: Dict[str, List[Dict[str, Any]]],
    gt_by_image: Dict[str, Dict[int, List[List[float]]]],
    *,
    name_to_cat: Dict[str, int],
    eval_iou: float,
) -> Dict[str, Any]:
    total_gt = 0
    covered = 0
    for image, gt_by_class in gt_by_image.items():
        pred_by_class: Dict[int, List[List[float]]] = {}
        for det in pred_by_image.get(image, []):
            label = str(det.get("label") or "").strip().lower()
            cat_id = name_to_cat.get(label)
            if cat_id is None:
                continue
            pred_by_class.setdefault(cat_id, []).append(det["bbox_xyxy_px"])
        for cat_id, gt_boxes in gt_by_class.items():
            preds = pred_by_class.get(cat_id, [])
            for gt in gt_boxes:
                total_gt += 1
                found = False
                for pred in preds:
                    if _iou(pred, gt) >= eval_iou:
                        found = True
                        break
                if found:
                    covered += 1
    recall = covered / total_gt if total_gt else 0.0
    return {"covered": covered, "total_gt": total_gt, "recall_upper_bound": recall}


def _load_prepass_baseline(
    prepass_jsonl: Path,
    *,
    eval_images: Set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not prepass_jsonl.exists():
        return by_image
    with prepass_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            image = str(record.get("image") or "")
            if not image or (eval_images and image not in eval_images):
                continue
            for det in record.get("detections") or []:
                if not isinstance(det, dict):
                    continue
                label = str(det.get("label") or det.get("class_name") or "").strip().lower()
                bbox = det.get("bbox_xyxy_px")
                if not label or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                primary, source_list = _normalize_source_fields(det)
                try:
                    score = float(det.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                by_image[image].append(
                    {
                        "label": label,
                        "bbox_xyxy_px": [float(v) for v in bbox[:4]],
                        "score": score,
                        "score_source": primary,
                        "source": primary,
                        "source_list": source_list,
                    }
                )
    return by_image


def _load_prepass_raw_detector_baselines(
    prepass_jsonl: Path,
    *,
    eval_images: Set[str],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    by_source: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "yolo": defaultdict(list),
        "rfdetr": defaultdict(list),
        "yolo_rfdetr_union": defaultdict(list),
    }
    if not prepass_jsonl.exists():
        return {key: {} for key in by_source}
    with prepass_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            image = str(record.get("image") or "")
            if not image or (eval_images and image not in eval_images):
                continue
            provenance = record.get("provenance")
            if not isinstance(provenance, dict):
                continue
            atoms = provenance.get("atoms")
            if not isinstance(atoms, list):
                continue
            for atom in atoms:
                if not isinstance(atom, dict):
                    continue
                if str(atom.get("stage") or "").strip().lower() != "detector":
                    continue
                source = str(
                    atom.get("source_primary") or atom.get("source") or atom.get("score_source") or ""
                ).strip().lower()
                if source not in {"yolo", "rfdetr"}:
                    continue
                label = str(atom.get("label") or "").strip().lower()
                bbox = atom.get("bbox_xyxy_px")
                if not label or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                try:
                    score = float(atom.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                entry = {
                    "label": label,
                    "bbox_xyxy_px": [float(v) for v in bbox[:4]],
                    "score": score,
                    "score_source": source,
                    "source": source,
                    "source_list": [source],
                }
                by_source[source][image].append(entry)
                by_source["yolo_rfdetr_union"][image].append(entry)
    return {
        key: {image: list(dets) for image, dets in image_map.items()}
        for key, image_map in by_source.items()
    }


def _summarize_prepass_warnings(
    prepass_jsonl: Path,
    *,
    eval_images: Set[str],
) -> Dict[str, Any]:
    images_total = 0
    images_with_warnings = 0
    warning_counts: Dict[str, int] = {}
    detector_fail_counts: Dict[str, int] = {}
    if not prepass_jsonl.exists():
        return {
            "images_total": 0,
            "images_with_warnings": 0,
            "warning_counts": {},
            "detector_fail_counts": {},
        }
    with prepass_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            image = str(record.get("image") or "")
            if not image or (eval_images and image not in eval_images):
                continue
            images_total += 1
            warnings = [str(w) for w in (record.get("warnings") or []) if str(w)]
            if warnings:
                images_with_warnings += 1
            for warn in warnings:
                warning_counts[warn] = warning_counts.get(warn, 0) + 1
                if warn.startswith("deep_prepass_detector_failed:"):
                    # Format: deep_prepass_detector_failed:{mode}:{run}:{error}
                    parts = warn.split(":", 3)
                    if len(parts) >= 3:
                        mode = parts[1].strip().lower()
                        detector_fail_counts[mode] = detector_fail_counts.get(mode, 0) + 1
    return {
        "images_total": int(images_total),
        "images_with_warnings": int(images_with_warnings),
        "warning_counts": warning_counts,
        "detector_fail_counts": detector_fail_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XGBoost ensemble with dedupe and IoU matching.")
    parser.add_argument("--model", required=True, help="Model .json path.")
    parser.add_argument("--meta", required=True, help="Model meta json.")
    parser.add_argument("--data", required=True, help="Input labeled .npz data.")
    parser.add_argument("--dataset", required=True, help="Dataset id.")
    parser.add_argument("--yolo-root", default=None, help="YOLO root directory.")
    parser.add_argument("--prepass-jsonl", default=None, help="Optional prepass JSONL for source baselines.")
    parser.add_argument("--eval-iou", type=float, default=0.5, help="Eval IoU threshold.")
    parser.add_argument("--eval-iou-grid", type=str, default=None, help="Comma-separated eval IoUs.")
    parser.add_argument("--dedupe-iou", type=float, default=0.75, help="Dedupe IoU threshold.")
    parser.add_argument("--dedupe-iou-grid", type=str, default=None, help="Comma-separated dedupe IoUs.")
    parser.add_argument("--scoreless-iou", type=float, default=0.0, help="Scoreless overlap threshold.")
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="primary",
        choices=["primary", "wbf", "source_weighted"],
        help="Cluster representative mode during dedupe.",
    )
    parser.add_argument(
        "--source-weights-json",
        type=str,
        default=None,
        help="JSON file/path for source weights used by source_weighted fusion.",
    )
    parser.add_argument(
        "--policy-json",
        type=str,
        default=None,
        help="Optional policy JSON file/string for post-score acceptance controls.",
    )
    parser.add_argument("--use-val-split", action="store_true", help="Evaluate only validation split.")
    parser.add_argument("--output-jsonl", default=None, help="Optional JSONL output of deduped detections.")
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
    policy = _load_policy(args.policy_json)
    if not policy and isinstance(meta.get("ensemble_policy"), dict):
        policy = dict(meta.get("ensemble_policy"))
    source_weights = _load_source_weights(args.source_weights_json, policy)
    fusion_mode = str(args.fusion_mode or "primary").strip().lower()
    if fusion_mode == "primary" and isinstance(policy.get("fusion_mode"), str):
        policy_fusion_mode = str(policy.get("fusion_mode") or "").strip().lower()
        if policy_fusion_mode in {"primary", "wbf", "source_weighted"}:
            fusion_mode = policy_fusion_mode

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    feature_names = [str(name) for name in data.get("feature_names", [])]
    parsed_meta = [json.loads(str(row)) for row in data["meta"]]

    if meta.get("log1p_counts"):
        X = _apply_log1p_counts(X, feature_names)
    X = _standardize(X, meta.get("feature_mean"), meta.get("feature_std"))

    probs = _predict_probabilities(
        X,
        parsed_meta=parsed_meta,
        model_path=Path(args.model),
        meta=meta,
    )

    if args.use_val_split:
        val_images = set(meta.get("split_val_images") or [])
        if val_images:
            keep_mask = [str(row.get("image") or "") in val_images for row in parsed_meta]
            mask = np.asarray(keep_mask, dtype=bool)
            if mask.any():
                probs = probs[mask]
                parsed_meta = [row for idx, row in enumerate(parsed_meta) if mask[idx]]

    detections_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    candidates_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    candidate_rows: List[Tuple[str, Dict[str, Any]]] = []
    for idx, payload in enumerate(parsed_meta):
        label = str(payload.get("label") or "").strip().lower()
        image = str(payload.get("image") or "")
        if not image or not label:
            continue
        bbox = payload.get("bbox_xyxy_px")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            bbox = payload.get("bbox") or payload.get("bbox_yolo")
            if bbox and payload.get("image_w") and payload.get("image_h"):
                bbox = _yolo_to_xyxy(bbox, int(payload["image_w"]), int(payload["image_h"]))
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        primary_source, source_list = _normalize_source_fields(payload)
        try:
            raw_score = float(payload.get("score") or 0.0)
        except (TypeError, ValueError):
            raw_score = 0.0
        score_by_source = {}
        raw_score_by_source = payload.get("score_by_source")
        if isinstance(raw_score_by_source, dict):
            for raw_src, raw_src_score in raw_score_by_source.items():
                src_name = str(raw_src or "").strip().lower()
                if not src_name:
                    continue
                try:
                    score_val = float(raw_src_score)
                except (TypeError, ValueError):
                    continue
                prev = score_by_source.get(src_name)
                if prev is None or score_val > prev:
                    score_by_source[src_name] = score_val
        if primary_source and (primary_source not in score_by_source or raw_score > score_by_source[primary_source]):
            score_by_source[primary_source] = raw_score
        candidate_det = {
            "label": label,
            "bbox_xyxy_px": [float(v) for v in bbox[:4]],
            "score": raw_score,
            "score_source": primary_source,
            "source": primary_source,
            "source_list": source_list,
            "score_by_source": score_by_source,
            "ensemble_prob_raw": float(probs[idx]),
        }
        candidates_by_image[image].append(candidate_det)
        candidate_rows.append((image, candidate_det))

    yolo_root = Path(args.yolo_root or f"uploads/clip_dataset_uploads/{args.dataset}_yolo")
    labelmap_path = yolo_root / "labelmap.txt"
    labelmap = [line.strip().lower() for line in labelmap_path.read_text().splitlines() if line.strip()]
    name_to_cat = {name: idx for idx, name in enumerate(labelmap)}

    eval_images = set(str(row.get("image") or "") for row in parsed_meta if str(row.get("image") or ""))
    dataset_root = Path("uploads/qwen_runs/datasets") / args.dataset
    gt_by_image: Dict[str, Dict[int, List[List[float]]]] = {}
    image_sizes: Dict[str, Tuple[int, int]] = {}
    for image in eval_images:
        label_path = yolo_root / "labels" / "val" / f"{Path(image).stem}.txt"
        if not label_path.exists():
            label_path = yolo_root / "labels" / "train" / f"{Path(image).stem}.txt"
        if not label_path.exists():
            continue
        img_path = None
        for split in ("val", "train"):
            candidate = dataset_root / split / image
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        try:
            from PIL import Image

            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            continue
        image_sizes[image] = (int(img_w), int(img_h))
        gt_by_class: Dict[int, List[List[float]]] = {}
        for raw in label_path.read_text().splitlines():
            parts = [p for p in raw.strip().split() if p]
            if len(parts) < 5:
                continue
            try:
                cat_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                continue
            gt_by_class.setdefault(cat_id, []).append(_yolo_to_xyxy([cx, cy, bw, bh], img_w, img_h))
        if gt_by_class:
            gt_by_image[image] = gt_by_class

    baseline_by_image = candidates_by_image
    prepass_health = None
    raw_detector_by_source: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "yolo": {},
        "rfdetr": {},
        "yolo_rfdetr_union": {},
    }
    if args.prepass_jsonl:
        prepass_jsonl = Path(args.prepass_jsonl)
        baseline_from_prepass = _load_prepass_baseline(prepass_jsonl, eval_images=eval_images)
        if baseline_from_prepass:
            baseline_by_image = baseline_from_prepass
        raw_detector_by_source = _load_prepass_raw_detector_baselines(
            prepass_jsonl, eval_images=eval_images
        )
        prepass_health = _summarize_prepass_warnings(prepass_jsonl, eval_images=eval_images)
    detector_support_index = _build_detector_support_index(raw_detector_by_source)
    geom_stats = _compute_geometry_stats(gt_by_image, image_sizes, labelmap=labelmap)

    policy_stats: Dict[str, Any] = {
        "enabled": bool(policy),
        "total_candidates": 0,
        "accepted": 0,
        "rejected_threshold": 0,
        "rejected_sam_only_floor": 0,
        "rejected_consensus": 0,
        "sam_only_candidates": 0,
        "sam_only_accepted": 0,
        "accepted_by_source_primary": {},
        "accepted_by_label": {},
        "applied_fusion_mode": fusion_mode,
    }

    sam_floor_default = _safe_float(policy.get("sam_only_min_prob_default"), 0.0)
    sam_floor_map = policy.get("sam_only_min_prob_by_class")
    consensus_default = _safe_float(policy.get("consensus_iou_default"), 0.0)
    consensus_map = policy.get("consensus_iou_by_class")
    consensus_class_aware = bool(policy.get("consensus_class_aware", True))
    geom_cfg = policy.get("geom_prior")
    if not isinstance(geom_cfg, dict):
        geom_cfg = {}
    geom_enabled = bool(geom_cfg.get("enabled", False))
    geom_min_sigma = max(1e-6, _safe_float(geom_cfg.get("min_sigma"), 0.1))
    geom_area_soft = _safe_float(geom_cfg.get("area_z_soft"), 2.5)
    geom_area_hard = _safe_float(geom_cfg.get("area_z_hard"), 4.0)
    geom_aspect_soft = _safe_float(geom_cfg.get("aspect_z_soft"), 2.5)
    geom_aspect_hard = _safe_float(geom_cfg.get("aspect_z_hard"), 4.0)
    geom_penalty_soft = _safe_float(geom_cfg.get("penalty_soft"), 0.35)
    geom_penalty_hard = _safe_float(geom_cfg.get("penalty_hard"), 0.9)
    geom_penalty_by_class = geom_cfg.get("penalty_by_class") if isinstance(geom_cfg.get("penalty_by_class"), dict) else {}
    threshold_override_map = policy.get("threshold_by_class_override")

    for image, candidate_det in candidate_rows:
        label = str(candidate_det.get("label") or "").strip().lower()
        primary_source = str(candidate_det.get("score_source") or candidate_det.get("source") or "unknown").strip().lower()
        source_list = {
            str(src or "").strip().lower()
            for src in (candidate_det.get("source_list") or [])
            if str(src or "").strip()
        }
        source_list.add(primary_source)
        has_detector_support = ("yolo" in source_list) or ("rfdetr" in source_list)
        is_sam_primary = primary_source in {"sam3_text", "sam3_similarity"}
        is_sam_only = is_sam_primary and not has_detector_support
        policy_stats["total_candidates"] += 1
        if is_sam_only:
            policy_stats["sam_only_candidates"] += 1

        prob_raw = _safe_float(candidate_det.get("ensemble_prob_raw"), 0.0)
        prob_adj = float(prob_raw)
        bias = _policy_bias_by_source_label(policy, source=primary_source, label=label)
        if abs(bias) > 1e-12:
            prob_adj = _apply_logit_shift(prob_adj, bias)

        # Geometry prior penalty (logit-domain subtraction).
        if geom_enabled:
            stats = geom_stats.get(label)
            size = image_sizes.get(image)
            bbox = candidate_det.get("bbox_xyxy_px")
            if stats and size and isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and size[0] > 0 and size[1] > 0:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                if bw > 0.0 and bh > 0.0:
                    denom = float(size[0]) * float(size[1])
                    area_log = math.log(max((bw * bh) / max(denom, 1.0), 1e-12))
                    aspect_log = math.log(max(bw / bh, 1e-12))
                    area_sigma = max(geom_min_sigma, _safe_float(stats.get("area_sigma"), geom_min_sigma))
                    aspect_sigma = max(geom_min_sigma, _safe_float(stats.get("aspect_sigma"), geom_min_sigma))
                    area_z = abs((area_log - _safe_float(stats.get("area_mu"), area_log)) / area_sigma)
                    aspect_z = abs((aspect_log - _safe_float(stats.get("aspect_mu"), aspect_log)) / aspect_sigma)
                    penalty = 0.0
                    if area_z >= geom_area_hard or aspect_z >= geom_aspect_hard:
                        penalty = geom_penalty_hard
                    elif area_z >= geom_area_soft or aspect_z >= geom_aspect_soft:
                        penalty = geom_penalty_soft
                    class_scale = _policy_value_by_class(geom_penalty_by_class, label=label, default=1.0)
                    penalty *= max(0.0, class_scale)
                    if penalty > 0.0:
                        prob_adj = _apply_logit_shift(prob_adj, -penalty)

        # Threshold selection with optional per-class override.
        thr = float(thresholds_by_label.get(label, default_threshold)) if thresholds_by_label else default_threshold
        thr = _policy_value_by_class(threshold_override_map, label=label, default=thr)

        # SAM-only soft floor.
        if is_sam_only:
            sam_floor = _policy_value_by_class(sam_floor_map, label=label, default=sam_floor_default)
            if sam_floor > 0.0 and prob_adj < sam_floor:
                policy_stats["rejected_sam_only_floor"] += 1
                continue

        # SAM-only detector consensus.
        if is_sam_only:
            consensus_iou = _policy_value_by_class(consensus_map, label=label, default=consensus_default)
            if consensus_iou > 0.0:
                bbox = candidate_det.get("bbox_xyxy_px") or []
                if not _has_detector_consensus(
                    detector_support_index,
                    image=image,
                    label=label,
                    bbox=bbox,
                    iou_thr=consensus_iou,
                    class_aware=consensus_class_aware,
                ):
                    policy_stats["rejected_consensus"] += 1
                    continue

        if prob_adj < thr:
            policy_stats["rejected_threshold"] += 1
            continue

        accepted = dict(candidate_det)
        accepted["score"] = float(prob_adj)
        accepted["ensemble_prob"] = float(prob_adj)
        accepted["ensemble_prob_raw"] = float(prob_raw)
        detections_by_image[image].append(accepted)
        policy_stats["accepted"] += 1
        if is_sam_only:
            policy_stats["sam_only_accepted"] += 1
        source_counts = policy_stats["accepted_by_source_primary"]
        source_counts[primary_source] = int(source_counts.get(primary_source, 0)) + 1
        label_counts = policy_stats["accepted_by_label"]
        label_counts[label] = int(label_counts.get(label, 0)) + 1

    eval_ious = _parse_grid(args.eval_iou_grid, float(args.eval_iou))
    dedupe_ious = _parse_grid(args.dedupe_iou_grid, float(args.dedupe_iou))

    metrics_grid: List[dict] = []
    for dedupe_iou in dedupe_ious:
        for eval_iou in eval_ious:
            entry = _evaluate_predictions(
                detections_by_image,
                gt_by_image,
                name_to_cat=name_to_cat,
                dedupe_iou=float(dedupe_iou),
                eval_iou=float(eval_iou),
                scoreless_iou=float(args.scoreless_iou),
                fusion_mode=fusion_mode,
                source_weights=source_weights,
                apply_dedupe=True,
            )
            entry["dedupe_iou"] = float(dedupe_iou)
            entry["eval_iou"] = float(eval_iou)
            metrics_grid.append(entry)

    best = None
    for entry in metrics_grid:
        if best is None or entry["f1"] > best["f1"]:
            best = entry

    ref_dedupe = float(args.dedupe_iou)
    ref_eval = float(args.eval_iou)
    xgb_reference = _evaluate_predictions(
        detections_by_image,
        gt_by_image,
        name_to_cat=name_to_cat,
        dedupe_iou=ref_dedupe,
        eval_iou=ref_eval,
        scoreless_iou=float(args.scoreless_iou),
        fusion_mode=fusion_mode,
        source_weights=source_weights,
        apply_dedupe=True,
    )
    xgb_post_prepass = _evaluate_predictions(
        detections_by_image,
        gt_by_image,
        name_to_cat=name_to_cat,
        dedupe_iou=ref_dedupe,
        eval_iou=ref_eval,
        scoreless_iou=0.0,
        fusion_mode=fusion_mode,
        source_weights=source_weights,
        apply_dedupe=False,
    )

    source_sets: Dict[str, Set[str]] = {
        "yolo": {"yolo"},
        "rfdetr": {"rfdetr"},
        "sam3_text": {"sam3_text"},
        "sam3_similarity": {"sam3_similarity"},
        "yolo_rfdetr_union": {"yolo", "rfdetr"},
        "all_sources": {"yolo", "rfdetr", "sam3_text", "sam3_similarity"},
    }

    post_prepass_primary: Dict[str, Dict[str, Any]] = {}
    post_prepass_attributed: Dict[str, Dict[str, Any]] = {}
    post_cluster_primary: Dict[str, Dict[str, Any]] = {}
    post_cluster_attributed: Dict[str, Dict[str, Any]] = {}
    for name, src_set in source_sets.items():
        primary_preds = _filter_by_sources(baseline_by_image, include_sources=src_set, attributed=False)
        attr_preds = _filter_by_sources(baseline_by_image, include_sources=src_set, attributed=True)
        post_prepass_primary[name] = _evaluate_predictions(
            primary_preds,
            gt_by_image,
            name_to_cat=name_to_cat,
            dedupe_iou=ref_dedupe,
            eval_iou=ref_eval,
            scoreless_iou=0.0,
            fusion_mode=fusion_mode,
            source_weights=source_weights,
            apply_dedupe=False,
        )
        post_prepass_attributed[name] = _evaluate_predictions(
            attr_preds,
            gt_by_image,
            name_to_cat=name_to_cat,
            dedupe_iou=ref_dedupe,
            eval_iou=ref_eval,
            scoreless_iou=0.0,
            fusion_mode=fusion_mode,
            source_weights=source_weights,
            apply_dedupe=False,
        )
        post_cluster_primary[name] = _evaluate_predictions(
            primary_preds,
            gt_by_image,
            name_to_cat=name_to_cat,
            dedupe_iou=ref_dedupe,
            eval_iou=ref_eval,
            scoreless_iou=float(args.scoreless_iou),
            fusion_mode=fusion_mode,
            source_weights=source_weights,
            apply_dedupe=True,
        )
        post_cluster_attributed[name] = _evaluate_predictions(
            attr_preds,
            gt_by_image,
            name_to_cat=name_to_cat,
            dedupe_iou=ref_dedupe,
            eval_iou=ref_eval,
            scoreless_iou=float(args.scoreless_iou),
            fusion_mode=fusion_mode,
            source_weights=source_weights,
            apply_dedupe=True,
        )

    post_prepass_all_metrics = _evaluate_predictions(
        baseline_by_image,
        gt_by_image,
        name_to_cat=name_to_cat,
        dedupe_iou=ref_dedupe,
        eval_iou=ref_eval,
        scoreless_iou=0.0,
        fusion_mode=fusion_mode,
        source_weights=source_weights,
        apply_dedupe=False,
    )
    post_cluster_all_metrics = _evaluate_predictions(
        baseline_by_image,
        gt_by_image,
        name_to_cat=name_to_cat,
        dedupe_iou=ref_dedupe,
        eval_iou=ref_eval,
        scoreless_iou=float(args.scoreless_iou),
        fusion_mode=fusion_mode,
        source_weights=source_weights,
        apply_dedupe=True,
    )

    coverage_post_prepass_all = _candidate_recall_upper_bound(
        baseline_by_image,
        gt_by_image,
        name_to_cat=name_to_cat,
        eval_iou=ref_eval,
    )
    coverage_post_prepass_primary_union = _candidate_recall_upper_bound(
        _filter_by_sources(baseline_by_image, include_sources=source_sets["yolo_rfdetr_union"], attributed=False),
        gt_by_image,
        name_to_cat=name_to_cat,
        eval_iou=ref_eval,
    )
    coverage_post_prepass_attr_union = _candidate_recall_upper_bound(
        _filter_by_sources(baseline_by_image, include_sources=source_sets["yolo_rfdetr_union"], attributed=True),
        gt_by_image,
        name_to_cat=name_to_cat,
        eval_iou=ref_eval,
    )

    raw_detector_tiers: Dict[str, Dict[str, Any]] = {}
    for key in ("yolo", "rfdetr", "yolo_rfdetr_union"):
        preds = raw_detector_by_source.get(key) or {}
        raw_detector_tiers[key] = {
            "candidate_total": int(sum(len(v) for v in preds.values())),
            "post_prepass": _evaluate_predictions(
                preds,
                gt_by_image,
                name_to_cat=name_to_cat,
                dedupe_iou=ref_dedupe,
                eval_iou=ref_eval,
                scoreless_iou=0.0,
                fusion_mode=fusion_mode,
                source_weights=source_weights,
                apply_dedupe=False,
            ),
            "post_cluster": _evaluate_predictions(
                preds,
                gt_by_image,
                name_to_cat=name_to_cat,
                dedupe_iou=ref_dedupe,
                eval_iou=ref_eval,
                scoreless_iou=0.0,
                fusion_mode=fusion_mode,
                source_weights=source_weights,
                apply_dedupe=True,
            ),
            "coverage_upper_bound": _candidate_recall_upper_bound(
                preds,
                gt_by_image,
                name_to_cat=name_to_cat,
                eval_iou=ref_eval,
            ),
        }

    out = {
        "tp": best["tp"] if best else 0,
        "fp": best["fp"] if best else 0,
        "fn": best["fn"] if best else 0,
        "precision": best["precision"] if best else 0.0,
        "recall": best["recall"] if best else 0.0,
        "f1": best["f1"] if best else 0.0,
        "iou_sweep": metrics_grid,
        "reference_iou": {
            "dedupe_iou": ref_dedupe,
            "eval_iou": ref_eval,
            "xgb_ensemble": xgb_reference,
        },
        "counts": {
            "candidate_total": int(sum(len(v) for v in baseline_by_image.values())),
            "accepted_total": int(sum(len(v) for v in detections_by_image.values())),
            "candidate_source_counts_primary": _count_by_sources(baseline_by_image, attributed=False),
            "candidate_source_counts_attributed": _count_by_sources(baseline_by_image, attributed=True),
            "accepted_source_counts_primary": _count_by_sources(detections_by_image, attributed=False),
            "accepted_source_counts_attributed": _count_by_sources(detections_by_image, attributed=True),
        },
        "policy": {
            "config": policy,
            "stats": policy_stats,
            "source_weights": source_weights,
        },
        "metric_tiers": {
            "raw_detector": raw_detector_tiers,
            "post_prepass": {
                "candidate_all": post_prepass_all_metrics,
                "source_primary": post_prepass_primary,
                "source_attributed": post_prepass_attributed,
                "coverage_upper_bound": {
                    "candidate_all": coverage_post_prepass_all,
                    "yolo_rfdetr_primary_union": coverage_post_prepass_primary_union,
                    "yolo_rfdetr_attributed_union": coverage_post_prepass_attr_union,
                },
            },
            "post_cluster": {
                "candidate_all": post_cluster_all_metrics,
                "source_primary": post_cluster_primary,
                "source_attributed": post_cluster_attributed,
            },
            "post_xgb": {
                "accepted_all": xgb_reference,
                "accepted_post_prepass": xgb_post_prepass,
            },
        },
        "baselines": {
            "candidate_all": post_cluster_all_metrics,
            "source_primary": post_cluster_primary,
            "source_attributed": post_cluster_attributed,
            "method_note": (
                "Baselines are tiered in metric_tiers. "
                "raw_detector uses provenance atoms (detector stage replay), "
                "post_prepass is pre-dedupe candidate scoring, post_cluster is deduped."
            ),
        },
        "coverage_upper_bound": {
            "candidate_all": coverage_post_prepass_all,
            "yolo_rfdetr_primary_union": coverage_post_prepass_primary_union,
            "yolo_rfdetr_attributed_union": coverage_post_prepass_attr_union,
        },
    }
    if prepass_health is not None:
        out["prepass_health"] = prepass_health

    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        with out_path.open("w", encoding="utf-8") as handle:
            for image, dets in detections_by_image.items():
                deduped, _ = _merge_prepass_detections(
                    dets,
                    iou_thr=float(args.dedupe_iou),
                    fusion_mode=fusion_mode,
                    source_weights=source_weights,
                )
                if args.scoreless_iou:
                    deduped, _ = _filter_scoreless_detections(deduped, iou_thr=float(args.scoreless_iou))
                handle.write(json.dumps({"image": image, "detections": deduped}, ensure_ascii=True) + "\n")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
