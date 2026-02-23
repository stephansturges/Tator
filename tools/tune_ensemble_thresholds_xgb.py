#!/usr/bin/env python
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
) -> Tuple[List[Dict[str, Any]], int]:
    if not detections or iou_thr <= 0:
        return detections, 0

    def det_score(det: Dict[str, Any]) -> float:
        try:
            return float(det.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    merged: List[Dict[str, Any]] = []
    removed = 0
    ordered = sorted(detections, key=det_score, reverse=True)
    for det in ordered:
        label = str(det.get("label") or det.get("class_name") or "").strip().lower()
        box = det.get("bbox_xyxy_px")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            merged.append(det)
            continue
        matched_idx = None
        for idx, kept in enumerate(merged):
            kept_label = str(kept.get("label") or kept.get("class_name") or "").strip().lower()
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
            if det_score(det) > det_score(kept):
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


def _normalize_source_fields(row: Dict[str, Any]) -> tuple[str, Set[str]]:
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
    if primary != "sam3_text":
        return False
    return ("yolo" not in sources) and ("rfdetr" not in sources)


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
    meta_rows: List[Dict[str, Any]],
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
            det_mask = np.asarray([_has_detector_support(row) for row in meta_rows], dtype=bool)
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
            alpha = float(quality_cfg.get("alpha") or 0.35)
            alpha = max(0.0, min(1.0, alpha))
            if alpha > 0.0:
                text_mask = np.asarray([_is_sam3_text_only(row) for row in meta_rows], dtype=bool)
                if text_mask.any():
                    q_probs = np.asarray(quality_booster.predict(xgb.DMatrix(X[text_mask])), dtype=np.float32)
                    probs[text_mask] = np.asarray(
                        (1.0 - alpha) * probs[text_mask] + alpha * q_probs,
                        dtype=np.float32,
                    )
    return probs


def _load_float_map(path_or_json: Optional[str]) -> Dict[str, float]:
    if not path_or_json:
        return {}
    raw = str(path_or_json).strip()
    if not raw:
        return {}
    candidate = Path(raw)
    payload = candidate.read_text() if candidate.exists() and candidate.is_file() else raw
    try:
        parsed = json.loads(payload)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in parsed.items():
        lbl = str(k or "").strip().lower()
        if not lbl:
            continue
        try:
            out[lbl] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fp_ratio = (fp / tp) if tp else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fp_ratio": float(fp_ratio),
    }


def _threshold_source(meta: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    for key in (
        "calibrated_thresholds_objective",
        "calibrated_thresholds_relaxed_smoothed",
        "calibrated_thresholds_relaxed",
        "calibrated_thresholds",
    ):
        raw = meta.get(key)
        if not isinstance(raw, dict):
            continue
        cleaned: Dict[str, float] = {}
        for k, v in raw.items():
            lbl = str(k or "").strip().lower()
            if not lbl:
                continue
            try:
                cleaned[lbl] = float(v)
            except (TypeError, ValueError):
                continue
        if cleaned:
            return key, cleaned
    return "default", {}


def _score_key(metrics: Dict[str, Any], optimize: str, threshold: float) -> Tuple[float, float, float, float]:
    if optimize == "recall":
        return (
            float(metrics.get("recall") or 0.0),
            float(metrics.get("f1") or 0.0),
            float(metrics.get("precision") or 0.0),
            -float(threshold),
        )
    if optimize == "tp":
        return (
            float(metrics.get("tp") or 0.0),
            float(metrics.get("f1") or 0.0),
            float(metrics.get("precision") or 0.0),
            -float(threshold),
        )
    return (
        float(metrics.get("f1") or 0.0),
        float(metrics.get("recall") or 0.0),
        float(metrics.get("precision") or 0.0),
        -float(threshold),
    )


def _pick_best(
    scored: List[Dict[str, Any]],
    *,
    optimize: str,
    require_fp_ratio: Optional[float],
    require_recall: Optional[float],
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for entry in scored:
        metrics = entry["metrics"]
        if metrics["tp"] <= 0:
            continue
        if require_fp_ratio is not None and require_fp_ratio >= 0:
            if metrics["fp_ratio"] > require_fp_ratio:
                continue
        if require_recall is not None and require_recall > 0:
            if metrics["recall"] < require_recall:
                continue
        if best is None:
            best = entry
            continue
        if _score_key(metrics, optimize, float(entry["threshold"])) > _score_key(
            best["metrics"], optimize, float(best["threshold"])
        ):
            best = entry
    return best


def _build_threshold_grid(values: List[float], *, base_threshold: float, steps: int) -> List[float]:
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        return [float(max(0.0, min(1.0, base_threshold)))]
    steps = max(5, min(401, int(steps)))
    qs = np.linspace(0.0, 1.0, steps)
    grid = np.quantile(vals, qs)
    merged = np.concatenate([grid, np.asarray([base_threshold, 0.0, 1.0], dtype=np.float32)])
    merged = np.clip(merged, 0.0, 1.0)
    merged = np.unique(merged)
    return [float(x) for x in merged.tolist()]


def _evaluate_label_threshold(
    rows_by_image: Dict[str, List[Dict[str, Any]]],
    gt_by_image: Dict[str, Dict[int, List[List[float]]]],
    *,
    cat_id: int,
    threshold: float,
    dedupe_iou: float,
    eval_iou: float,
    scoreless_iou: float,
) -> Dict[str, Any]:
    tp = fp = fn = 0
    images = set(gt_by_image.keys()) | set(rows_by_image.keys())
    for image in images:
        preds = []
        for row in rows_by_image.get(image, []):
            if float(row.get("prob") or 0.0) < threshold:
                continue
            preds.append(
                {
                    "label": row.get("label"),
                    "bbox_xyxy_px": row.get("bbox_xyxy_px"),
                    "score": float(row.get("prob") or 0.0),
                    "score_source": row.get("score_source") or "unknown",
                    "source": row.get("score_source") or "unknown",
                }
            )
        deduped, _ = _merge_prepass_detections(preds, iou_thr=dedupe_iou)
        if scoreless_iou > 0:
            deduped, _ = _filter_scoreless_detections(deduped, iou_thr=scoreless_iou)
        pred_boxes = [det.get("bbox_xyxy_px") for det in deduped if isinstance(det.get("bbox_xyxy_px"), list)]
        gt_boxes = gt_by_image.get(image, {}).get(int(cat_id), [])
        ctp, cfp, cfn = _match_class(gt_boxes, pred_boxes, eval_iou)
        tp += ctp
        fp += cfp
        fn += cfn
    return _compute_metrics(tp, fp, fn)


def _evaluate_global_thresholds(
    label_rows: Dict[str, Dict[str, List[Dict[str, Any]]]],
    gt_by_image: Dict[str, Dict[int, List[List[float]]]],
    *,
    name_to_cat: Dict[str, int],
    threshold_by_label: Dict[str, float],
    default_threshold: float,
    dedupe_iou: float,
    eval_iou: float,
    scoreless_iou: float,
) -> Dict[str, Any]:
    tp = fp = fn = 0
    by_label: Dict[str, Dict[str, Any]] = {}
    for label, cat_id in name_to_cat.items():
        rows_by_image = label_rows.get(label, {})
        thr = float(threshold_by_label.get(label, default_threshold))
        metrics = _evaluate_label_threshold(
            rows_by_image,
            gt_by_image,
            cat_id=int(cat_id),
            threshold=thr,
            dedupe_iou=dedupe_iou,
            eval_iou=eval_iou,
            scoreless_iou=scoreless_iou,
        )
        metrics["threshold"] = float(thr)
        by_label[label] = metrics
        tp += int(metrics["tp"])
        fp += int(metrics["fp"])
        fn += int(metrics["fn"])
    out = _compute_metrics(tp, fp, fn)
    out["by_label"] = by_label
    return out


def _load_gt_boxes(dataset: str, eval_images: List[str]) -> Tuple[Dict[str, int], Dict[str, Dict[int, List[List[float]]]]]:
    yolo_root = Path(f"uploads/clip_dataset_uploads/{dataset}_yolo")
    labelmap_path = yolo_root / "labelmap.txt"
    labelmap = [line.strip().lower() for line in labelmap_path.read_text().splitlines() if line.strip()]
    name_to_cat = {name: idx for idx, name in enumerate(labelmap)}
    dataset_root = Path("uploads/qwen_runs/datasets") / dataset
    gt_by_image: Dict[str, Dict[int, List[List[float]]]] = {}
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
    return name_to_cat, gt_by_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune XGBoost thresholds using deduped object-level metrics.")
    parser.add_argument("--model", required=True, help="Model .json path.")
    parser.add_argument("--meta", required=True, help="Model meta json to update.")
    parser.add_argument("--data", required=True, help="Input labeled .npz data.")
    parser.add_argument("--dataset", required=True, help="Dataset id.")
    parser.add_argument("--optimize", default="f1", choices=["f1", "recall", "tp"], help="Objective metric.")
    parser.add_argument("--target-fp-ratio", type=float, default=0.2, help="Max FP/TP ratio target.")
    parser.add_argument(
        "--target-fp-ratio-by-label-json",
        default=None,
        help="Optional JSON dict/file of per-label FP ratio caps (overrides --target-fp-ratio for listed labels).",
    )
    parser.add_argument("--min-recall", type=float, default=0.0, help="Minimum recall floor.")
    parser.add_argument(
        "--min-recall-by-label-json",
        default=None,
        help="Optional JSON dict/file of per-label recall floors (overrides --min-recall for listed labels).",
    )
    parser.add_argument("--steps", type=int, default=61, help="Threshold grid steps per label.")
    parser.add_argument("--eval-iou", type=float, default=0.5, help="Eval IoU threshold.")
    parser.add_argument("--dedupe-iou", type=float, default=0.75, help="Dedupe IoU threshold.")
    parser.add_argument("--scoreless-iou", type=float, default=0.0, help="Scoreless overlap filter IoU.")
    parser.add_argument("--use-val-split", action="store_true", help="Use validation split from meta.")
    args = parser.parse_args()

    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text())
    threshold_source, base_thresholds = _threshold_source(meta)
    default_threshold = float(meta.get("calibrated_threshold") or 0.5)

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    meta_rows = [json.loads(str(row)) for row in data["meta"]]
    feature_names = [str(name) for name in data.get("feature_names", [])]

    if meta.get("log1p_counts"):
        X = _apply_log1p_counts(X, feature_names)
    X = _standardize(X, meta.get("feature_mean"), meta.get("feature_std"))

    probs = _predict_probabilities(
        X,
        meta_rows=meta_rows,
        model_path=Path(args.model),
        meta=meta,
    )

    if args.use_val_split:
        val_images = set(meta.get("split_val_images") or [])
        if val_images:
            keep_mask = np.asarray([str(row.get("image") or "") in val_images for row in meta_rows], dtype=bool)
            if keep_mask.any():
                probs = probs[keep_mask]
                meta_rows = [row for idx, row in enumerate(meta_rows) if keep_mask[idx]]

    eval_images = sorted({str(row.get("image") or "") for row in meta_rows if str(row.get("image") or "")})
    name_to_cat, gt_by_image = _load_gt_boxes(args.dataset, eval_images)

    label_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    label_probs: Dict[str, List[float]] = defaultdict(list)
    for idx, row in enumerate(meta_rows):
        image = str(row.get("image") or "")
        label = str(row.get("label") or "").strip().lower()
        bbox = row.get("bbox_xyxy_px")
        if not image or not label or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        if label not in name_to_cat:
            continue
        prob = float(probs[idx])
        entry = {
            "label": label,
            "bbox_xyxy_px": [float(v) for v in bbox[:4]],
            "prob": prob,
            "score_source": str(row.get("score_source") or row.get("source") or "unknown").strip().lower(),
        }
        label_rows[label][image].append(entry)
        label_probs[label].append(prob)

    tuned_thresholds: Dict[str, float] = {}
    label_reports: Dict[str, Dict[str, Any]] = {}
    fp_target = float(args.target_fp_ratio)
    recall_floor = float(args.min_recall)
    fp_target_by_label = _load_float_map(args.target_fp_ratio_by_label_json)
    recall_floor_by_label = _load_float_map(args.min_recall_by_label_json)
    for label, cat_id in sorted(name_to_cat.items()):
        rows_by_image = label_rows.get(label, {})
        base_thr = float(base_thresholds.get(label, default_threshold))
        grid = _build_threshold_grid(label_probs.get(label, []), base_threshold=base_thr, steps=int(args.steps))
        scored: List[Dict[str, Any]] = []
        for thr in grid:
            metrics = _evaluate_label_threshold(
                rows_by_image,
                gt_by_image,
                cat_id=int(cat_id),
                threshold=float(thr),
                dedupe_iou=float(args.dedupe_iou),
                eval_iou=float(args.eval_iou),
                scoreless_iou=float(args.scoreless_iou),
            )
            scored.append({"threshold": float(thr), "metrics": metrics})

        label_fp_target = float(fp_target_by_label.get(label, fp_target))
        label_recall_floor = float(recall_floor_by_label.get(label, recall_floor))

        strict = _pick_best(
            scored,
            optimize=str(args.optimize),
            require_fp_ratio=label_fp_target,
            require_recall=label_recall_floor,
        )
        no_recall = _pick_best(
            scored,
            optimize=str(args.optimize),
            require_fp_ratio=label_fp_target,
            require_recall=None,
        )
        unconstrained = _pick_best(
            scored,
            optimize=str(args.optimize),
            require_fp_ratio=None,
            require_recall=None,
        )
        chosen = strict or no_recall or unconstrained
        mode = "strict" if strict else ("fp_only" if no_recall else "unconstrained")
        if chosen is None:
            chosen = {
                "threshold": base_thr,
                "metrics": _evaluate_label_threshold(
                    rows_by_image,
                    gt_by_image,
                    cat_id=int(cat_id),
                    threshold=base_thr,
                    dedupe_iou=float(args.dedupe_iou),
                    eval_iou=float(args.eval_iou),
                    scoreless_iou=float(args.scoreless_iou),
                ),
            }
            mode = "base"
        tuned_thresholds[label] = float(chosen["threshold"])
        label_reports[label] = {
            "mode": mode,
            "base_threshold": base_thr,
            "tuned_threshold": float(chosen["threshold"]),
            "fp_ratio_target": label_fp_target,
            "recall_floor": label_recall_floor,
            "metrics": chosen["metrics"],
        }

    tuned_global = _evaluate_global_thresholds(
        label_rows,
        gt_by_image,
        name_to_cat=name_to_cat,
        threshold_by_label=tuned_thresholds,
        default_threshold=default_threshold,
        dedupe_iou=float(args.dedupe_iou),
        eval_iou=float(args.eval_iou),
        scoreless_iou=float(args.scoreless_iou),
    )
    base_global = _evaluate_global_thresholds(
        label_rows,
        gt_by_image,
        name_to_cat=name_to_cat,
        threshold_by_label=base_thresholds,
        default_threshold=default_threshold,
        dedupe_iou=float(args.dedupe_iou),
        eval_iou=float(args.eval_iou),
        scoreless_iou=float(args.scoreless_iou),
    )

    final_thresholds = dict(tuned_thresholds)
    blend_alpha = 0.0
    final_global = tuned_global
    if fp_target >= 0 and tuned_global["tp"] > 0 and tuned_global["fp_ratio"] > fp_target:
        best_candidate = None
        for alpha in np.linspace(0.05, 1.0, 20):
            blended = {}
            for label, tuned_thr in tuned_thresholds.items():
                base_thr = float(base_thresholds.get(label, default_threshold))
                blended[label] = float(alpha * base_thr + (1.0 - alpha) * tuned_thr)
            metrics = _evaluate_global_thresholds(
                label_rows,
                gt_by_image,
                name_to_cat=name_to_cat,
                threshold_by_label=blended,
                default_threshold=default_threshold,
                dedupe_iou=float(args.dedupe_iou),
                eval_iou=float(args.eval_iou),
                scoreless_iou=float(args.scoreless_iou),
            )
            if metrics["tp"] <= 0:
                continue
            key = _score_key(metrics, str(args.optimize), float(alpha))
            if metrics["fp_ratio"] <= fp_target:
                if best_candidate is None or key > best_candidate["key"]:
                    best_candidate = {"alpha": float(alpha), "thresholds": blended, "metrics": metrics, "key": key}
        if best_candidate is not None:
            blend_alpha = float(best_candidate["alpha"])
            final_thresholds = best_candidate["thresholds"]
            final_global = best_candidate["metrics"]

    meta["calibrated_thresholds_objective_base_source"] = threshold_source
    meta["calibrated_thresholds_objective_base"] = base_thresholds
    meta["calibrated_thresholds_objective"] = final_thresholds
    meta["calibration_objective_by_label"] = label_reports
    meta["calibration_objective_metrics_base"] = base_global
    meta["calibration_objective_metrics_tuned"] = tuned_global
    meta["calibration_objective_metrics"] = final_global
    meta["calibration_objective_params"] = {
        "optimize": str(args.optimize),
        "target_fp_ratio": fp_target,
        "target_fp_ratio_by_label": fp_target_by_label,
        "min_recall": recall_floor,
        "min_recall_by_label": recall_floor_by_label,
        "steps": int(args.steps),
        "eval_iou": float(args.eval_iou),
        "dedupe_iou": float(args.dedupe_iou),
        "scoreless_iou": float(args.scoreless_iou),
        "use_val_split": bool(args.use_val_split),
        "blend_alpha": float(blend_alpha),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(
        json.dumps(
            {
                "base_threshold_source": threshold_source,
                "labels_tuned": len(final_thresholds),
                "base_metrics": base_global,
                "tuned_metrics": tuned_global,
                "final_metrics": final_global,
                "blend_alpha": blend_alpha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
