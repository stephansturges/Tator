#!/usr/bin/env bash
set -euo pipefail

python - "$@" <<'PY'
import argparse
import base64
import json
import math
import random
import time
from pathlib import Path

import requests


parser = argparse.ArgumentParser(description="Run Qwen agentic benchmark and compare to COCO labels.")
parser.add_argument("--count", type=int, default=10, help="Number of images to sample.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
parser.add_argument("--dataset", default="qwen_dataset", help="Dataset id under uploads/qwen_runs/datasets.")
parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Thinking", help="Qwen model id.")
parser.add_argument("--variant", default="Thinking", choices=["auto", "Instruct", "Thinking"], help="Model variant.")
parser.add_argument("--output", default=None, help="Output JSONL path.")
parser.add_argument("--api-root", default="http://127.0.0.1:8000", help="API root.")
parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching.")
parser.add_argument("--max-steps", type=int, default=12, help="Agent max steps.")
parser.add_argument("--max-tool-calls", type=int, default=20, help="Agent max tool calls.")
parser.add_argument("--max-detections", type=int, default=800, help="Agent max detections.")
parser.add_argument("--max-new-tokens", type=int, default=1200, help="Agent max new tokens.")
parser.add_argument("--unload-first", action="store_true", help="Call /runtime/unload before starting.")
parser.add_argument("--unload-each", action="store_true", help="Call /runtime/unload before each image.")
args = parser.parse_args()

random.seed(args.seed)

dataset_root = Path("uploads/qwen_runs/datasets") / args.dataset
coco_path = dataset_root / "val" / "_annotations.coco.json"
if not coco_path.exists():
    coco_path = dataset_root / "train" / "_annotations.coco.json"
if not coco_path.exists():
    raise SystemExit(f"Missing COCO annotations at {coco_path}")

coco = json.loads(coco_path.read_text())
images = list(coco.get("images") or [])
if len(images) < args.count:
    raise SystemExit(f"Not enough images ({len(images)}) for {args.count}-image benchmark.")

ann_by_image = {}
for ann in coco.get("annotations") or []:
    img_id = ann.get("image_id")
    if img_id is None:
        continue
    ann_by_image.setdefault(img_id, []).append(ann)

categories = list(coco.get("categories") or [])
name_to_cat = {str(cat.get("name") or "").strip().lower(): int(cat.get("id")) for cat in categories}

selected = random.sample(images, args.count)
out_path = Path(args.output or f"qwen_agentic_benchmark_{args.count}img_seed{args.seed}.jsonl")
summary_path = out_path.with_suffix(".summary.json")

api_url = f"{args.api_root.rstrip('/')}/qwen/agentic"
unload_url = f"{args.api_root.rstrip('/')}/runtime/unload"

if args.unload_first:
    try:
        requests.post(unload_url, timeout=None)
    except Exception:
        pass


def emit(record: dict) -> None:
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def yolo_to_xyxy(bbox, w, h):
    cx, cy, bw, bh = bbox
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return [x1, y1, x2, y2]


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
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
    if denom <= 0:
        return 0.0
    return inter / denom


def match_class(gt_boxes, pred_boxes, iou_thr):
    matched_gt = set()
    tp = 0
    fp = 0
    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = None
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            cur = iou(pred, gt)
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


overall = {"tp": 0, "fp": 0, "fn": 0, "by_class": {}}

for idx, info in enumerate(selected, 1):
    img_name = info.get("file_name")
    img_id = info.get("id")
    width = info.get("width") or 0
    height = info.get("height") or 0
    if not img_name:
        continue
    img_path = dataset_root / ("val" if (dataset_root / "val" / img_name).exists() else "train") / img_name
    img_bytes = img_path.read_bytes()
    image_base64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "dataset_id": args.dataset,
        "image_base64": image_base64,
        "image_name": img_name,
        "model_id": args.model_id,
        "model_variant": args.variant,
        "max_steps": args.max_steps,
        "max_tool_calls": args.max_tool_calls,
        "max_detections": args.max_detections,
        "max_new_tokens": args.max_new_tokens,
    }
    record = {
        "ts": time.time(),
        "image": img_name,
        "model_id": args.model_id,
        "variant": args.variant,
        "payload_bytes": len(json.dumps(payload)),
    }
    if args.unload_each:
        try:
            requests.post(unload_url, timeout=None)
        except Exception:
            pass
    print(f"[{idx}/{args.count}] {img_name}")
    try:
        resp = requests.post(api_url, json=payload, timeout=None)
        record["status"] = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            detections = data.get("detections", [])
            trace = data.get("trace", [])
            record["warnings"] = data.get("warnings", [])
            record["detections"] = detections
            record["trace"] = trace
        else:
            record["error"] = resp.text[:300]
            emit(record)
            continue
    except Exception as exc:  # noqa: BLE001
        record["status"] = "exception"
        record["error"] = str(exc)[:300]
        emit(record)
        continue

    gt_anns = ann_by_image.get(img_id, [])
    gt_by_class = {}
    for ann in gt_anns:
        cat_id = ann.get("category_id")
        bbox = ann.get("bbox")
        if cat_id is None or not bbox:
            continue
        gt_by_class.setdefault(int(cat_id), []).append(xywh_to_xyxy(bbox))

    pred_by_class = {}
    unknown = 0
    for det in detections:
        label = str(det.get("label") or "").strip().lower()
        cat_id = name_to_cat.get(label)
        if cat_id is None:
            unknown += 1
            continue
        bbox_xyxy = det.get("bbox_xyxy_px")
        if not bbox_xyxy and det.get("bbox_yolo") and width and height:
            bbox_xyxy = yolo_to_xyxy(det.get("bbox_yolo"), width, height)
        if not bbox_xyxy or len(bbox_xyxy) < 4:
            continue
        pred_by_class.setdefault(cat_id, []).append([float(v) for v in bbox_xyxy[:4]])

    per_class = {}
    tp = fp = fn = 0
    for cat_id, gt_boxes in gt_by_class.items():
        preds = pred_by_class.get(cat_id, [])
        ctp, cfp, cfn = match_class(gt_boxes, preds, args.iou)
        per_class[str(cat_id)] = {"tp": ctp, "fp": cfp, "fn": cfn}
        tp += ctp
        fp += cfp
        fn += cfn
        overall["by_class"].setdefault(str(cat_id), {"tp": 0, "fp": 0, "fn": 0})
        overall["by_class"][str(cat_id)]["tp"] += ctp
        overall["by_class"][str(cat_id)]["fp"] += cfp
        overall["by_class"][str(cat_id)]["fn"] += cfn
    unknown_fp = unknown
    fp += unknown_fp
    overall["tp"] += tp
    overall["fp"] += fp
    overall["fn"] += fn

    tool_calls = [t for t in (record.get("trace") or []) if t.get("phase") == "tool_call"]
    tool_usage = {}
    for entry in tool_calls:
        tool = entry.get("tool_name") or "unknown"
        tool_usage[tool] = tool_usage.get(tool, 0) + 1

    record["metrics"] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "unknown_fp": unknown_fp,
        "per_class": per_class,
    }
    record["tool_usage"] = tool_usage
    emit(record)

summary = {
    "count": args.count,
    "seed": args.seed,
    "model_id": args.model_id,
    "variant": args.variant,
    "iou": args.iou,
    "tp": overall["tp"],
    "fp": overall["fp"],
    "fn": overall["fn"],
    "by_class": overall["by_class"],
    "output": str(out_path),
}
summary_path.write_text(json.dumps(summary, indent=2))
print("Benchmark complete.")
print("Output:", out_path)
print("Summary:", summary_path)
PY
