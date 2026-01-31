#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _compute_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _build_predictions(
    probs: np.ndarray,
    meta_rows: List[dict],
    thresholds_by_label: Dict[str, float],
    default_threshold: float,
) -> List[bool]:
    preds: List[bool] = []
    for prob, row in zip(probs, meta_rows):
        label = str(row.get("label") or "").strip()
        thr = thresholds_by_label.get(label, default_threshold)
        preds.append(bool(prob >= thr))
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XGBoost ensemble with dedupe.")
    parser.add_argument("--model", required=True, help="Model .json path.")
    parser.add_argument("--meta", required=True, help="Model meta json.")
    parser.add_argument("--data", required=True, help="Input labeled .npz data.")
    parser.add_argument("--dataset", required=True, help="Dataset id.")
    parser.add_argument("--eval-iou", type=float, default=0.5, help="Eval IoU threshold.")
    parser.add_argument("--eval-iou-grid", type=str, default=None, help="Comma-separated eval IoUs.")
    parser.add_argument("--dedupe-iou", type=float, default=0.75, help="Dedupe IoU threshold.")
    parser.add_argument("--dedupe-iou-grid", type=str, default=None, help="Comma-separated dedupe IoUs.")
    parser.add_argument("--scoreless-iou", type=float, default=0.0, help="Scoreless overlap threshold.")
    parser.add_argument("--use-val-split", action="store_true", help="Evaluate only validation split.")
    args = parser.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    thresholds_by_label = meta.get("calibrated_thresholds_relaxed_smoothed")
    if not isinstance(thresholds_by_label, dict):
        thresholds_by_label = meta.get("calibrated_thresholds_relaxed") if isinstance(meta.get("calibrated_thresholds_relaxed"), dict) else {}
    if not isinstance(thresholds_by_label, dict):
        thresholds_by_label = meta.get("calibrated_thresholds") if isinstance(meta.get("calibrated_thresholds"), dict) else {}
    default_threshold = float(meta.get("calibrated_threshold") or 0.5)

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feature_names = [str(name) for name in data.get("feature_names", [])]
    meta_rows = [json.loads(str(row)) for row in data["meta"]]

    if meta.get("log1p_counts"):
        X = _apply_log1p_counts(X, feature_names)
    X = _standardize(X, meta.get("feature_mean"), meta.get("feature_std"))

    booster = xgb.Booster()
    booster.load_model(str(Path(args.model)))
    probs = booster.predict(xgb.DMatrix(X))
    preds = _build_predictions(probs, meta_rows, thresholds_by_label, default_threshold)

    # optionally filter to validation split
    if args.use_val_split and meta.get("split_val_images"):
        val_set = set(meta.get("split_val_images") or [])
        mask = [row.get("image") in val_set for row in meta_rows]
        probs = probs[mask]
        preds = [p for p, keep in zip(preds, mask) if keep]
        y = y[mask]
        meta_rows = [row for row, keep in zip(meta_rows, mask) if keep]

    eval_ious = [float(x) for x in (args.eval_iou_grid or str(args.eval_iou)).split(",")]
    dedupe_ious = [float(x) for x in (args.dedupe_iou_grid or str(args.dedupe_iou)).split(",")]

    # group by image
    by_image: Dict[str, List[int]] = {}
    for idx, row in enumerate(meta_rows):
        img = row.get("image") or ""
        by_image.setdefault(img, []).append(idx)

    metrics_grid: List[dict] = []
    for dedupe_iou in dedupe_ious:
        for eval_iou in eval_ious:
            tp = fp = fn = 0
            for img_name, indices in by_image.items():
                img_preds = []
                gt_boxes_by_label: Dict[str, List[List[float]]] = {}
                pred_boxes_by_label: Dict[str, List[List[float]]] = {}
                for idx in indices:
                    row = meta_rows[idx]
                    if not preds[idx]:
                        continue
                    bbox = row.get("bbox_xyxy_px")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                        bbox = row.get("bbox") or row.get("bbox_yolo")
                        if bbox and row.get("image_w") and row.get("image_h"):
                            bbox = _yolo_to_xyxy(bbox, int(row["image_w"]), int(row["image_h"]))
                    label = str(row.get("label") or "").strip()
                    if bbox is None or not label:
                        continue
                    pred_boxes_by_label.setdefault(label, []).append(list(bbox))
                # build gt from meta rows
                for idx in indices:
                    row = meta_rows[idx]
                    gt_box = row.get("gt_bbox_xyxy_px")
                    if not isinstance(gt_box, (list, tuple)) or len(gt_box) < 4:
                        continue
                    label = str(row.get("label") or "").strip()
                    if not label:
                        continue
                    gt_boxes_by_label.setdefault(label, []).append(list(gt_box))

                # dedupe predictions per label
                final_preds: Dict[str, List[List[float]]] = {}
                for label, boxes in pred_boxes_by_label.items():
                    dets = [{"bbox_xyxy_px": b, "label": label, "score": 1.0} for b in boxes]
                    merged, _ = _merge_prepass_detections(dets, iou_thr=dedupe_iou)
                    merged, _ = _filter_scoreless_detections(merged, iou_thr=float(args.scoreless_iou))
                    final_preds[label] = [d["bbox_xyxy_px"] for d in merged]

                for label, gt_boxes in gt_boxes_by_label.items():
                    preds_label = final_preds.get(label, [])
                    tpp, fpp, fnn = _match_class(gt_boxes, preds_label, eval_iou)
                    tp += tpp
                    fp += fpp
                    fn += fnn
            metrics = _compute_metrics(tp, fp, fn)
            metrics_grid.append({**metrics, "dedupe_iou": dedupe_iou, "eval_iou": eval_iou})

    best = None
    for entry in metrics_grid:
        if best is None or entry["f1"] > best["f1"]:
            best = entry
    out = {
        "tp": best["tp"] if best else 0,
        "fp": best["fp"] if best else 0,
        "fn": best["fn"] if best else 0,
        "precision": best["precision"] if best else 0.0,
        "recall": best["recall"] if best else 0.0,
        "f1": best["f1"] if best else 0.0,
        "iou_sweep": metrics_grid,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
