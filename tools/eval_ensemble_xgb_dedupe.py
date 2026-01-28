#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xgboost as xgb


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _apply_log1p_counts(X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    if not feature_names:
        return X
    count_tokens = ("count", "support_count", "sam3_text_count", "sam3_sim_count")
    for idx, name in enumerate(feature_names):
        if any(token in name for token in count_tokens):
            X[:, idx] = np.log1p(np.maximum(X[:, idx], 0.0))
    return X


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XGBoost ensemble with dedupe.")
    parser.add_argument("--model", required=True, help="Model .json path.")
    parser.add_argument("--meta", required=True, help="Model meta json.")
    parser.add_argument("--data", required=True, help="Input .npz data.")
    parser.add_argument("--dataset", default="qwen_dataset", help="Dataset id.")
    parser.add_argument("--yolo-root", default=None, help="YOLO root directory.")
    parser.add_argument("--eval-iou", type=float, default=0.5, help="IoU threshold for evaluation.")
    parser.add_argument("--dedupe-iou", type=float, default=0.5, help="IoU threshold for dedupe.")
    parser.add_argument("--scoreless-iou", type=float, default=0.0, help="IoU threshold for scoreless filter.")
    parser.add_argument("--threshold", type=float, default=None, help="Override global threshold.")
    parser.add_argument("--use-val-split", action="store_true", help="Restrict evaluation to validation split.")
    args = parser.parse_args()

    meta = json.loads(Path(args.meta).read_text()) if Path(args.meta).exists() else {}
    thresholds_by_label = meta.get("calibrated_thresholds") if isinstance(meta.get("calibrated_thresholds"), dict) else {}
    default_threshold = float(meta.get("calibrated_threshold") or 0.5)
    if args.threshold is not None:
        default_threshold = float(args.threshold)

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    meta_rows = list(data["meta"])
    feature_names = [str(name) for name in data.get("feature_names", [])]

    if meta.get("log1p_counts"):
        X = _apply_log1p_counts(X, feature_names)
    mean = meta.get("feature_mean")
    std = meta.get("feature_std")
    if mean is not None and std is not None:
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.shape[0] == X.shape[1] and std.shape[0] == X.shape[1]:
            X = _standardize(X, mean, std)

    parsed_meta: List[Dict[str, Any]] = []
    for row in meta_rows:
        try:
            parsed_meta.append(json.loads(str(row)))
        except json.JSONDecodeError:
            parsed_meta.append({})

    if args.use_val_split:
        val_images = set(meta.get("split_val_images") or [])
        if val_images:
            keep_mask = [str(row.get("image") or "") in val_images for row in parsed_meta]
            mask = np.asarray(keep_mask, dtype=bool)
            if mask.any():
                X = X[mask]
                parsed_meta = [row for idx, row in enumerate(parsed_meta) if mask[idx]]

    model = xgb.Booster()
    model.load_model(str(args.model))
    probs = model.predict(xgb.DMatrix(X))

    detections_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for idx, payload in enumerate(parsed_meta):
        label = str(payload.get("label") or "").strip().lower()
        if thresholds_by_label and label in thresholds_by_label and args.threshold is None:
            thr = float(thresholds_by_label[label])
        else:
            thr = default_threshold
        if probs[idx] < thr:
            continue
        image = str(payload.get("image") or "")
        if not image:
            continue
        bbox = payload.get("bbox_xyxy_px")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        det = {"label": label, "bbox_xyxy_px": [float(v) for v in bbox[:4]], "score": float(probs[idx])}
        detections_by_image.setdefault(image, []).append(det)

    yolo_root = Path(args.yolo_root or f"uploads/clip_dataset_uploads/{args.dataset}_yolo")
    labelmap_path = yolo_root / "labelmap.txt"
    labelmap = [line.strip() for line in labelmap_path.read_text().splitlines() if line.strip()]
    name_to_cat = {name.strip().lower(): idx for idx, name in enumerate(labelmap)}

    dataset_root = Path("uploads/qwen_runs/datasets") / args.dataset
    gt_by_image: Dict[str, Dict[int, List[List[float]]]] = {}
    for image in detections_by_image:
        label_path = (yolo_root / "labels" / "val" / f"{Path(image).stem}.txt")
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

    total_tp = total_fp = total_fn = 0
    for image, dets in detections_by_image.items():
        gt_by_class = gt_by_image.get(image)
        if not gt_by_class:
            continue
        deduped, _ = _merge_prepass_detections(dets, iou_thr=float(args.dedupe_iou))
        if args.scoreless_iou:
            deduped, _ = _filter_scoreless_detections(deduped, iou_thr=float(args.scoreless_iou))
        pred_by_class: Dict[int, List[List[float]]] = {}
        unknown = 0
        for det in deduped:
            lbl = str(det.get("label") or "").strip().lower()
            cat_id = name_to_cat.get(lbl)
            if cat_id is None:
                unknown += 1
                continue
            pred_by_class.setdefault(cat_id, []).append(det["bbox_xyxy_px"])
        tp = fp = fn = 0
        for cat_id, gt_boxes in gt_by_class.items():
            preds = pred_by_class.get(cat_id, [])
            ctp, cfp, cfn = _match_class(gt_boxes, preds, float(args.eval_iou))
            tp += ctp
            fp += cfp
            fn += cfn
        fp += unknown
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    output = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
