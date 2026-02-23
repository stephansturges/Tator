#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def _load_labelmap(dataset_id: str) -> List[str]:
    path = Path(f"uploads/clip_dataset_uploads/{dataset_id}_yolo/labelmap.txt")
    if not path.exists():
        raise SystemExit(f"Missing labelmap at {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Label candidates using IoU>=0.5.")
    parser.add_argument("--input", required=True, help="Input .npz with X and meta.")
    parser.add_argument("--dataset", required=True, help="Dataset id.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for accept label.")
    args = parser.parse_args()

    labelmap = _load_labelmap(args.dataset)
    name_to_cat = {name.strip().lower(): idx for idx, name in enumerate(labelmap)}
    yolo_root = Path(f"uploads/clip_dataset_uploads/{args.dataset}_yolo")

    data = np.load(args.input, allow_pickle=True)
    X = data["X"]
    meta_raw = list(data["meta"])
    feature_names = list(data["feature_names"])
    classifier_classes = list(data.get("classifier_classes", []))
    sam3_iou = float(data.get("sam3_iou", 0.5))
    support_iou = float(data.get("support_iou", 0.5))
    sam3_term_list = list(data.get("sam3_term_list", [])) if "sam3_term_list" in data else []
    sam3_term_hash_dim = int(data.get("sam3_term_hash_dim", 0)) if "sam3_term_hash_dim" in data else 0

    meta = [json.loads(str(row)) for row in meta_raw]
    by_image: Dict[str, List[int]] = {}
    for idx, row in enumerate(meta):
        by_image.setdefault(row["image"], []).append(idx)

    y = np.zeros((len(meta),), dtype=np.int64)
    y_iou = np.zeros((len(meta),), dtype=np.float32)
    best_iou_any = np.zeros((len(meta),), dtype=np.float32)
    best_label_any = np.full((len(meta),), -1, dtype=np.int64)
    for image_name, indices in by_image.items():
        label_path = yolo_root / "labels" / "val" / f"{Path(image_name).stem}.txt"
        if not label_path.exists():
            label_path = yolo_root / "labels" / "train" / f"{Path(image_name).stem}.txt"
        if not label_path.exists():
            continue
        dataset_root = Path("uploads/qwen_runs/datasets") / args.dataset
        img_path = None
        for split in ("val", "train"):
            candidate = dataset_root / split / image_name
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

        gt_all: List[tuple[int, List[float]]] = []
        for cat_id, boxes in gt_by_class.items():
            for box in boxes:
                gt_all.append((cat_id, box))

        for idx in indices:
            row = meta[idx]
            label = str(row.get("label") or "").strip().lower()
            cat_id = name_to_cat.get(label)
            if cat_id is None:
                continue
            bbox = row.get("bbox_xyxy_px") or []
            if not bbox or len(bbox) < 4:
                continue
            best_iou = 0.0
            for gt_box in gt_by_class.get(cat_id, []):
                best_iou = max(best_iou, _iou(bbox, gt_box))
                if best_iou >= args.iou:
                    break
            y_iou[idx] = float(best_iou)
            best_any = 0.0
            best_any_label = -1
            for gt_cat, gt_box in gt_all:
                cur = _iou(bbox, gt_box)
                if cur > best_any:
                    best_any = cur
                    best_any_label = gt_cat
            best_iou_any[idx] = float(best_any)
            best_label_any[idx] = int(best_any_label)
            if best_iou >= args.iou:
                y[idx] = 1

    np.savez(
        args.output,
        X=X,
        y=y,
        y_iou=y_iou,
        best_iou_any=best_iou_any,
        best_label_any=best_label_any,
        meta=np.asarray(meta_raw, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
        labelmap=np.asarray(labelmap, dtype=object),
        classifier_classes=np.asarray(classifier_classes, dtype=object),
        sam3_iou=float(sam3_iou),
        support_iou=float(support_iou),
        sam3_term_list=np.asarray(sam3_term_list, dtype=object),
        sam3_term_hash_dim=int(sam3_term_hash_dim),
        label_iou=float(args.iou),
    )


if __name__ == "__main__":
    main()
