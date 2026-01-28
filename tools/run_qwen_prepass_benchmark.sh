#!/usr/bin/env bash
set -euo pipefail

python -u - "$@" <<'PY'
import argparse
import base64
import json
import math
import random
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


parser = argparse.ArgumentParser(description="Run Qwen prepass benchmark and compare to COCO labels.")
parser.add_argument("--count", type=int, default=10, help="Number of images to sample.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
parser.add_argument("--dataset", default="qwen_dataset", help="Dataset id under uploads/qwen_runs/datasets.")
parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Thinking", help="Qwen model id.")
parser.add_argument("--variant", default="Thinking", choices=["auto", "Instruct", "Thinking"], help="Model variant.")
parser.add_argument("--output", default=None, help="Output JSONL path.")
parser.add_argument("--api-root", default="http://127.0.0.1:8000", help="API root.")
parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching.")
parser.add_argument("--prepass-iou", type=float, default=None, help="Override prepass dedupe IoU.")
parser.add_argument("--max-detections", type=int, default=2000, help="Agent max detections.")
parser.add_argument("--max-new-tokens", type=int, default=8192, help="Agent max new tokens.")
parser.add_argument("--thinking-effort", type=float, default=None, help="Thinking effort scaling (lower = less thinking).")
parser.add_argument("--thinking-scale-factor", type=float, default=None, help="Scale factor for thinking-effort logits.")
parser.add_argument("--immediate-action-bias", action="store_true", help="Enable immediate action bias on 'wait'.")
parser.add_argument("--no-immediate-action-bias", action="store_true", help="Disable immediate action bias.")
parser.add_argument("--immediate-action-min-chars", type=int, default=None, help="Min think chars before wait bias.")
parser.add_argument("--immediate-action-min-seconds", type=float, default=None, help="Min think seconds before wait bias.")
parser.add_argument("--immediate-action-logit-bias", type=float, default=None, help="Logit bias for end-think.")
parser.add_argument("--min-items", type=int, default=0, help="Min GT items per image (filter for sampling).")
parser.add_argument("--min-classes", type=int, default=0, help="Min distinct classes per image (filter for sampling).")
parser.add_argument("--use-overlay", action="store_true", help="Enable detection overlay for the agent.")
parser.add_argument("--no-overlay", action="store_true", help="Disable detection overlay for the agent.")
parser.add_argument("--tighten-fp", action="store_true", help="Enable precision tightening profile.")
parser.add_argument("--no-tighten-fp", action="store_true", help="Disable precision tightening profile.")
parser.add_argument("--detector-conf", type=float, default=None, help="Override detector confidence.")
parser.add_argument("--sam3-score-thr", type=float, default=None, help="Override SAM3 score threshold.")
parser.add_argument("--sam3-mask-thr", type=float, default=None, help="Override SAM3 mask threshold.")
parser.add_argument("--prepass-sam3-text-thr", type=float, default=None, help="Override prepass SAM3 text score threshold.")
parser.add_argument("--prepass-similarity-score", type=float, default=None, help="Override prepass SAM3 similarity score threshold.")
parser.add_argument("--sam3-syn-budget", type=int, default=None, help="Override SAM3 text synonym budget.")
parser.add_argument("--similarity-min-exemplar-score", type=float, default=None, help="Override similarity exemplar min score.")
parser.add_argument("--similarity-mid-conf-low", type=float, default=None, help="Override similarity mid-conf low bound.")
parser.add_argument("--similarity-mid-conf-high", type=float, default=None, help="Override similarity mid-conf high bound.")
parser.add_argument("--similarity-mid-conf-class-count", type=int, default=None, help="Override similarity mid-conf class count.")
parser.add_argument("--prepass-similarity-per-class", type=int, default=None, help="Override prepass similarity per-class cap.")
parser.add_argument("--classifier-min-prob", type=float, default=None, help="Override classifier min prob.")
parser.add_argument("--classifier-margin", type=float, default=None, help="Override classifier margin.")
parser.add_argument("--classifier-bg-margin", type=float, default=None, help="Override classifier bg margin.")
parser.add_argument("--classifier-id", default=None, help="Override classifier id/path.")
parser.add_argument("--scoreless-iou", type=float, default=None, help="Override scoreless IoU gate.")
parser.add_argument("--prepass-caption-profile", default=None, help="Prepass caption profile (light/deep).")
parser.add_argument("--prepass-caption-model-id", default=None, help="Override caption model id.")
parser.add_argument("--prepass-caption-variant", default=None, help="Override caption variant.")
parser.add_argument("--prepass-caption-max-tokens", type=int, default=None, help="Override caption max tokens.")
parser.add_argument("--no-prepass-caption", action="store_true", help="Disable prepass captioning.")
parser.add_argument("--prepass-only", action="store_true", help="Return prepass-only detections (skip agent loop).")
parser.add_argument("--prepass-finalize", action="store_true", help="Finalize prepass with classifier/dedupe.")
parser.add_argument("--skip-yolo-baseline", action="store_true", help="Skip YOLO baseline run.")
parser.add_argument("--grid-cols", type=int, default=None, help="Agent grid columns override.")
parser.add_argument("--grid-rows", type=int, default=None, help="Agent grid rows override.")
parser.add_argument("--grid-overlap", type=float, default=None, help="Agent grid overlap ratio.")
parser.add_argument(
    "--label-source",
    default="coco",
    choices=["coco", "yolo"],
    help="Ground-truth source: COCO json or YOLO labels.",
)
parser.add_argument(
    "--yolo-root",
    default=None,
    help="YOLO label root (default: uploads/clip_dataset_uploads/<dataset>_yolo).",
)
parser.add_argument(
    "--yolo-split",
    default="val",
    choices=["val", "train", "all"],
    help="YOLO label split to sample (val/train/all).",
)
parser.add_argument("--selection-file", default=None, help="Optional JSON selection file.")
parser.add_argument("--workers", type=int, default=None, help="Worker count for sharded runs.")
parser.add_argument("--worker-index", type=int, default=None, help="Worker index for sharded runs.")
parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids for sharded runs.")
parser.add_argument("--unload-first", action="store_true", help="Call /runtime/unload before starting.")
parser.add_argument("--unload-each", action="store_true", help="Call /runtime/unload before each image.")
args = parser.parse_args()

random.seed(args.seed)

dataset_root = Path("uploads/qwen_runs/datasets") / args.dataset
label_source = (args.label_source or "coco").strip().lower()
min_items = max(0, int(args.min_items or 0))
min_classes = max(0, int(args.min_classes or 0))
name_to_cat = {}
ann_by_image = {}
selected = []
label_files = []
selection_file = Path(args.selection_file) if args.selection_file else None
if label_source == "coco":
    coco_path = dataset_root / "val" / "_annotations.coco.json"
    if not coco_path.exists():
        coco_path = dataset_root / "train" / "_annotations.coco.json"
    if not coco_path.exists():
        raise SystemExit(f"Missing COCO annotations at {coco_path}")

    coco = json.loads(coco_path.read_text())
    images = list(coco.get("images") or [])
    if len(images) < args.count:
        raise SystemExit(f"Not enough images ({len(images)}) for {args.count}-image benchmark.")

    for ann in coco.get("annotations") or []:
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        ann_by_image.setdefault(img_id, []).append(ann)

    categories = list(coco.get("categories") or [])
    categories.sort(key=lambda item: int(item.get("id") or 0))
    labelmap = [str(cat.get("name") or "").strip() for cat in categories]
    name_to_cat = {str(cat.get("name") or "").strip().lower(): int(cat.get("id")) for cat in categories}
    candidates = []
    for img in images:
        img_id = img.get("id")
        anns = ann_by_image.get(img_id, [])
        if min_items and len(anns) < min_items:
            continue
        if min_classes:
            classes = {ann.get("category_id") for ann in anns if ann.get("category_id") is not None}
            if len(classes) < min_classes:
                continue
        candidates.append(img)
    if len(candidates) < args.count and not selection_file:
        raise SystemExit(
            f"Not enough images ({len(candidates)}) for {args.count} with min_items={min_items}, min_classes={min_classes}."
        )
    if selection_file and selection_file.exists():
        selection_payload = json.loads(selection_file.read_text())
        if selection_payload.get("label_source") != "coco":
            raise SystemExit("Selection file label_source mismatch.")
        image_ids = set(selection_payload.get("image_ids") or [])
        selected = [img for img in candidates if img.get("id") in image_ids]
    else:
        selected = random.sample(candidates, args.count)
else:
    yolo_root = Path(args.yolo_root or f"uploads/clip_dataset_uploads/{args.dataset}_yolo")
    labelmap_path = yolo_root / "labelmap.txt"
    if not labelmap_path.exists():
        raise SystemExit(f"Missing YOLO labelmap at {labelmap_path}")
    labelmap = [line.strip() for line in labelmap_path.read_text().splitlines() if line.strip()]
    name_to_cat = {name.strip().lower(): idx for idx, name in enumerate(labelmap)}

    yolo_split = (args.yolo_split or "val").strip().lower()
    label_dirs = []
    if yolo_split == "all":
        label_dirs = [yolo_root / "labels" / "val", yolo_root / "labels" / "train"]
    else:
        label_dirs = [yolo_root / "labels" / yolo_split]
        if not label_dirs[0].exists():
            fallback = "train" if yolo_split == "val" else "val"
            label_dirs = [yolo_root / "labels" / fallback]
    label_dirs = [p for p in label_dirs if p.exists()]
    if not label_dirs:
        raise SystemExit(f"Missing YOLO labels under {yolo_root / 'labels'}")

    image_paths = []
    for split in ("val", "train"):
        split_dir = dataset_root / split
        if split_dir.exists():
            image_paths.extend([p for p in split_dir.iterdir() if p.is_file()])
    image_by_stem = {p.stem: p for p in image_paths}

    label_files = []
    for label_dir in label_dirs:
        label_files.extend([p for p in label_dir.iterdir() if p.suffix == ".txt"])
    def _label_stats(path: Path) -> tuple[int, int]:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        classes = set()
        for ln in lines:
            parts = ln.split()
            if not parts:
                continue
            try:
                classes.add(int(float(parts[0])))
            except ValueError:
                continue
        return len(lines), len(classes)

    candidates = []
    for p in label_files:
        if p.stem not in image_by_stem:
            continue
        total, uniq = _label_stats(p)
        if min_items and total < min_items:
            continue
        if min_classes and uniq < min_classes:
            continue
        candidates.append(p)
    if len(candidates) < args.count and not selection_file:
        raise SystemExit(
            f"Not enough YOLO label files ({len(candidates)}) for {args.count} with min_items={min_items}, min_classes={min_classes}."
        )
    if selection_file and selection_file.exists():
        selection_payload = json.loads(selection_file.read_text())
        if selection_payload.get("label_source") != "yolo":
            raise SystemExit("Selection file label_source mismatch.")
        stems = set(selection_payload.get("label_stems") or [])
        selected = [p for p in candidates if p.stem in stems]
    else:
        selected = random.sample(candidates, args.count)
out_path = Path(args.output or f"qwen_prepass_benchmark_{args.count}img_seed{args.seed}.jsonl")
summary_path = out_path.with_suffix(".summary.json")
yolo_summary_path = out_path.with_suffix(".yolo.summary.json")

api_url = f"{args.api_root.rstrip('/')}/qwen/prepass"
unload_url = f"{args.api_root.rstrip('/')}/runtime/unload"


def _strip_worker_args(argv):
    value_flags = {"--gpus", "--worker-index", "--workers", "--selection-file", "--output"}
    toggle_flags = {"--unload-first", "--unload-each"}
    out = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in value_flags:
            skip_next = True
            continue
        if arg in toggle_flags:
            continue
        out.append(arg)
    return out


def _write_selection(path: Path, items, source: str) -> None:
    if source == "coco":
        payload = {"label_source": "coco", "image_ids": [img.get("id") for img in items if img.get("id") is not None]}
    else:
        payload = {"label_source": "yolo", "label_stems": [Path(p).stem for p in items]}
    path.write_text(json.dumps(payload, indent=2))


def _spawn_workers(gpu_list, worker_count, selection_path: Path) -> None:
    base_args = _strip_worker_args(sys.argv[1:])
    if "--skip-yolo-baseline" not in base_args:
        base_args.append("--skip-yolo-baseline")
    worker_outputs = []
    procs = []
    for idx in range(worker_count):
        worker_out = out_path.parent / f"{out_path.stem}.worker{idx}{out_path.suffix or '.jsonl'}"
        worker_outputs.append(worker_out)
        cmd = [
            "bash",
            "tools/run_qwen_prepass_benchmark.sh",
            *base_args,
            "--output",
            str(worker_out),
            "--selection-file",
            str(selection_path),
            "--workers",
            str(worker_count),
            "--worker-index",
            str(idx),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_list[idx])
        procs.append(subprocess.Popen(cmd, env=env))
    failures = []
    for proc in procs:
        if proc.wait() != 0:
            failures.append(proc.pid)
    if failures:
        raise SystemExit(f"Worker failures: {failures}")
    with out_path.open("w", encoding="utf-8") as out_f:
        for worker_out in worker_outputs:
            if worker_out.exists():
                out_f.write(worker_out.read_text())
    return worker_outputs


def _aggregate_worker_summaries(worker_outputs):
    combined = {"tp": 0, "fp": 0, "fn": 0, "by_class": {}}
    count = 0
    for worker_out in worker_outputs:
        worker_summary = worker_out.with_suffix(".summary.json")
        if not worker_summary.exists():
            continue
        data = json.loads(worker_summary.read_text())
        count += int(data.get("count") or 0)
        combined["tp"] += int(data.get("tp") or 0)
        combined["fp"] += int(data.get("fp") or 0)
        combined["fn"] += int(data.get("fn") or 0)
        for key, stats in (data.get("by_class") or {}).items():
            combined["by_class"].setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
            combined["by_class"][key]["tp"] += int(stats.get("tp") or 0)
            combined["by_class"][key]["fp"] += int(stats.get("fp") or 0)
            combined["by_class"][key]["fn"] += int(stats.get("fn") or 0)
    return count, combined


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


def _build_baseline_items(selected_items):
    items = []
    for info in selected_items:
        img_name = None
        img_id = None
        width = 0
        height = 0
        label_path = None
        if label_source == "coco":
            img_name = info.get("file_name")
            img_id = info.get("id")
            width = info.get("width") or 0
            height = info.get("height") or 0
            if not img_name:
                continue
            img_path = dataset_root / ("val" if (dataset_root / "val" / img_name).exists() else "train") / img_name
        else:
            label_path = info
            stem = label_path.stem
            img_path = image_by_stem.get(stem)
            if img_path is None:
                continue
            img_name = img_path.name
            try:
                from PIL import Image

                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception:
                width = height = 0
        if width and height:
            entry = {
                "image": img_name,
                "path": img_path,
                "label_path": label_path,
                "width": width,
                "height": height,
            }
            if label_source == "coco" and img_id is not None:
                entry["image_id"] = img_id
            items.append(entry)
    return items


gpu_list = [g.strip() for g in str(args.gpus or "").split(",") if g.strip()]
if gpu_list and args.worker_index is None:
    worker_count = int(args.workers or len(gpu_list))
    if worker_count > len(gpu_list):
        raise SystemExit("workers exceeds available gpus")
    selection_path = selection_file or out_path.with_suffix(".selection.json")
    _write_selection(selection_path, selected, label_source)
    worker_outputs = _spawn_workers(gpu_list[:worker_count], worker_count, selection_path)
    total_count = len(selected)
    summary = {
        "count": total_count,
        "seed": args.seed,
        "model_id": args.model_id,
        "variant": args.variant,
        "iou": args.iou,
        "label_source": label_source,
        "yolo_root": str(args.yolo_root or ""),
        "selection_filter": {"min_items": min_items, "min_classes": min_classes},
        "use_detection_overlay": True if args.use_overlay else (False if args.no_overlay else None),
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "by_class": {},
        "output": str(out_path),
        "worker_outputs": [str(p) for p in worker_outputs],
    }
    count, combined = _aggregate_worker_summaries(worker_outputs)
    summary["count"] = count or total_count
    summary["tp"] = combined["tp"]
    summary["fp"] = combined["fp"]
    summary["fn"] = combined["fn"]
    summary["by_class"] = combined["by_class"]
    if not args.skip_yolo_baseline:
        baseline_items = _build_baseline_items(selected)
        yolo_overall = {"tp": 0, "fp": 0, "fn": 0, "by_class": {}}
        for entry in baseline_items:
            img_path = entry["path"]
            width = entry["width"]
            height = entry["height"]
            img_bytes = img_path.read_bytes()
            image_base64 = base64.b64encode(img_bytes).decode("utf-8")
            resp = requests.post(
                f"{args.api_root.rstrip('/')}/yolo/predict_full",
                json={"image_base64": image_base64, "expected_labelmap": labelmap},
                timeout=None,
            )
            if resp.status_code != 200:
                print(f"[yolo-baseline] error {resp.status_code}: {resp.text[:200]}", flush=True)
                continue
            detections = (resp.json() or {}).get("detections", [])
            gt_by_class = {}
            if label_source == "coco":
                img_id = entry.get("image_id")
                for ann in ann_by_image.get(img_id, []):
                    cat_id = ann.get("category_id")
                    bbox = ann.get("bbox")
                    if cat_id is None or not bbox:
                        continue
                    gt_by_class.setdefault(int(cat_id), []).append(xywh_to_xyxy(bbox))
            else:
                label_path = entry["label_path"]
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
                    gt_by_class.setdefault(cat_id, []).append(yolo_to_xyxy([cx, cy, bw, bh], width, height))

            pred_by_class = {}
            unknown = 0
            for det in detections:
                label = str(det.get("class_name") or "").strip().lower()
                cat_id = name_to_cat.get(label)
                if cat_id is None:
                    unknown += 1
                    continue
                bbox = det.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                x, y, w, h = [float(v) for v in bbox[:4]]
                pred_by_class.setdefault(cat_id, []).append([x, y, x + w, y + h])

            tp = fp = fn = 0
            for cat_id, gt_boxes in gt_by_class.items():
                preds = pred_by_class.get(cat_id, [])
                ctp, cfp, cfn = match_class(gt_boxes, preds, args.iou)
                tp += ctp
                fp += cfp
                fn += cfn
                yolo_overall["by_class"].setdefault(str(cat_id), {"tp": 0, "fp": 0, "fn": 0})
                yolo_overall["by_class"][str(cat_id)]["tp"] += ctp
                yolo_overall["by_class"][str(cat_id)]["fp"] += cfp
                yolo_overall["by_class"][str(cat_id)]["fn"] += cfn
            fp += unknown
            yolo_overall["tp"] += tp
            yolo_overall["fp"] += fp
            yolo_overall["fn"] += fn

        yolo_summary = {
            "count": len(baseline_items),
            "seed": args.seed,
            "iou": args.iou,
            "label_source": label_source,
            "yolo_root": str(args.yolo_root or ""),
            "tp": yolo_overall["tp"],
            "fp": yolo_overall["fp"],
            "fn": yolo_overall["fn"],
            "by_class": yolo_overall["by_class"],
            "output": str(yolo_summary_path),
        }
        yolo_summary_path.write_text(json.dumps(yolo_summary, indent=2))
        summary["yolo_baseline"] = {
            "tp": yolo_overall["tp"],
            "fp": yolo_overall["fp"],
            "fn": yolo_overall["fn"],
            "summary_path": str(yolo_summary_path),
        }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("Benchmark complete.", flush=True)
    print("Output:", out_path, flush=True)
    print("Summary:", summary_path, flush=True)
    if summary.get("yolo_baseline"):
        print("YOLO baseline:", summary["yolo_baseline"]["summary_path"], flush=True)
    sys.exit(0)

if args.workers and args.worker_index is not None:
    selected = [item for idx, item in enumerate(selected) if idx % args.workers == args.worker_index]

if args.unload_first:
    try:
        requests.post(unload_url, timeout=None)
    except Exception:
        pass


overall = {"tp": 0, "fp": 0, "fn": 0, "by_class": {}}
yolo_overall = {"tp": 0, "fp": 0, "fn": 0, "by_class": {}}
baseline_items = []
total_count = len(selected)

for idx, info in enumerate(selected, 1):
    img_name = None
    img_id = None
    width = 0
    height = 0
    label_path = None
    if label_source == "coco":
        img_name = info.get("file_name")
        img_id = info.get("id")
        width = info.get("width") or 0
        height = info.get("height") or 0
        if not img_name:
            continue
        img_path = dataset_root / ("val" if (dataset_root / "val" / img_name).exists() else "train") / img_name
    else:
        label_path = info
        stem = label_path.stem
        img_path = image_by_stem.get(stem)
        if img_path is None:
            continue
        img_name = img_path.name
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception:
            width = height = 0
    if width and height:
        entry = {
            "image": img_name,
            "path": img_path,
            "label_path": label_path,
            "width": width,
            "height": height,
        }
        if label_source == "coco" and img_id is not None:
            entry["image_id"] = img_id
        baseline_items.append(entry)
    img_bytes = img_path.read_bytes()
    image_base64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "dataset_id": args.dataset,
        "image_base64": image_base64,
        "image_name": img_name,
        "model_id": args.model_id,
        "model_variant": args.variant,
        "max_detections": args.max_detections,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.prepass_only or args.prepass_finalize:
        payload["prepass_only"] = True
    if args.prepass_finalize:
        payload["prepass_finalize"] = True
    if args.prepass_iou is not None:
        payload["iou"] = args.prepass_iou
    if args.thinking_effort is not None:
        payload["thinking_effort"] = args.thinking_effort
    if args.thinking_scale_factor is not None:
        payload["thinking_scale_factor"] = args.thinking_scale_factor
    if args.immediate_action_bias:
        payload["immediate_action_bias"] = True
    elif args.no_immediate_action_bias:
        payload["immediate_action_bias"] = False
    if args.immediate_action_min_chars is not None:
        payload["immediate_action_min_chars"] = args.immediate_action_min_chars
    if args.immediate_action_min_seconds is not None:
        payload["immediate_action_min_seconds"] = args.immediate_action_min_seconds
    if args.immediate_action_logit_bias is not None:
        payload["immediate_action_logit_bias"] = args.immediate_action_logit_bias
    if args.grid_cols is not None:
        payload["grid_cols"] = args.grid_cols
    if args.grid_rows is not None:
        payload["grid_rows"] = args.grid_rows
    if args.grid_overlap is not None:
        payload["grid_overlap_ratio"] = args.grid_overlap
    overlay_flag = None
    if args.use_overlay:
        overlay_flag = True
    elif args.no_overlay:
        overlay_flag = False
    if overlay_flag is not None:
        payload["use_detection_overlay"] = overlay_flag
    tighten_fp = None
    if args.tighten_fp:
        tighten_fp = True
    elif args.no_tighten_fp:
        tighten_fp = False
    if tighten_fp is not None:
        payload["tighten_fp"] = tighten_fp
    if args.detector_conf is not None:
        payload["detector_conf"] = args.detector_conf
    if args.sam3_score_thr is not None:
        payload["sam3_score_thr"] = args.sam3_score_thr
    if args.sam3_mask_thr is not None:
        payload["sam3_mask_threshold"] = args.sam3_mask_thr
    if args.prepass_sam3_text_thr is not None:
        payload["prepass_sam3_text_thr"] = args.prepass_sam3_text_thr
    if args.prepass_similarity_score is not None:
        payload["prepass_similarity_score"] = args.prepass_similarity_score
    if args.sam3_syn_budget is not None:
        payload["sam3_text_synonym_budget"] = args.sam3_syn_budget
    if args.similarity_min_exemplar_score is not None:
        payload["similarity_min_exemplar_score"] = args.similarity_min_exemplar_score
    if args.similarity_mid_conf_low is not None:
        payload["similarity_mid_conf_low"] = args.similarity_mid_conf_low
    if args.similarity_mid_conf_high is not None:
        payload["similarity_mid_conf_high"] = args.similarity_mid_conf_high
    if args.similarity_mid_conf_class_count is not None:
        payload["similarity_mid_conf_class_count"] = args.similarity_mid_conf_class_count
    if args.prepass_similarity_per_class is not None:
        payload["prepass_similarity_per_class"] = args.prepass_similarity_per_class
    if args.classifier_min_prob is not None:
        payload["classifier_min_prob"] = args.classifier_min_prob
    if args.classifier_margin is not None:
        payload["classifier_margin"] = args.classifier_margin
    if args.classifier_bg_margin is not None:
        payload["classifier_bg_margin"] = args.classifier_bg_margin
    if args.classifier_id is not None:
        payload["classifier_id"] = args.classifier_id
    if args.scoreless_iou is not None:
        payload["scoreless_iou"] = args.scoreless_iou
    if args.prepass_caption_profile is not None:
        payload["prepass_caption_profile"] = args.prepass_caption_profile
    if args.prepass_caption_model_id is not None:
        payload["prepass_caption_model_id"] = args.prepass_caption_model_id
    if args.prepass_caption_variant is not None:
        payload["prepass_caption_variant"] = args.prepass_caption_variant
    if args.prepass_caption_max_tokens is not None:
        payload["prepass_caption_max_tokens"] = args.prepass_caption_max_tokens
    if args.no_prepass_caption:
        payload["prepass_caption"] = False
    if args.prepass_only:
        payload["prepass_only"] = True
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
    print(f"[{idx}/{total_count}] {img_name}", flush=True)
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
            record["trace_path"] = data.get("trace_path")
            record["trace_full_path"] = data.get("trace_full_path")
            if record["trace_full_path"]:
                print(f"trace_full_path: {record['trace_full_path']}", flush=True)
        else:
            record["error"] = resp.text[:300]
            emit(record)
            continue
    except Exception as exc:  # noqa: BLE001
        record["status"] = "exception"
        record["error"] = str(exc)[:300]
        emit(record)
        continue

    gt_by_class = {}
    if label_source == "coco":
        gt_anns = ann_by_image.get(img_id, [])
        for ann in gt_anns:
            cat_id = ann.get("category_id")
            bbox = ann.get("bbox")
            if cat_id is None or not bbox:
                continue
            gt_by_class.setdefault(int(cat_id), []).append(xywh_to_xyxy(bbox))
    else:
        if not width or not height or label_path is None:
            continue
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
            gt_by_class.setdefault(cat_id, []).append(yolo_to_xyxy([cx, cy, bw, bh], width, height))

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
    "count": total_count,
    "seed": args.seed,
    "model_id": args.model_id,
    "variant": args.variant,
    "iou": args.iou,
    "label_source": label_source,
    "yolo_root": str(args.yolo_root or ""),
    "selection_filter": {"min_items": min_items, "min_classes": min_classes},
    "use_detection_overlay": True if args.use_overlay else (False if args.no_overlay else None),
    "tp": overall["tp"],
    "fp": overall["fp"],
    "fn": overall["fn"],
    "by_class": overall["by_class"],
    "output": str(out_path),
}
if not args.skip_yolo_baseline and baseline_items:
    for entry in baseline_items:
        img_path = entry["path"]
        width = entry["width"]
        height = entry["height"]
        img_bytes = img_path.read_bytes()
        image_base64 = base64.b64encode(img_bytes).decode("utf-8")
        resp = requests.post(
            f"{args.api_root.rstrip('/')}/yolo/predict_full",
            json={"image_base64": image_base64, "expected_labelmap": labelmap},
            timeout=None,
        )
        if resp.status_code != 200:
            print(f"[yolo-baseline] error {resp.status_code}: {resp.text[:200]}", flush=True)
            continue
        detections = (resp.json() or {}).get("detections", [])
        gt_by_class = {}
        if label_source == "coco":
            img_id = entry.get("image_id")
            for ann in ann_by_image.get(img_id, []):
                cat_id = ann.get("category_id")
                bbox = ann.get("bbox")
                if cat_id is None or not bbox:
                    continue
                gt_by_class.setdefault(int(cat_id), []).append(xywh_to_xyxy(bbox))
        else:
            label_path = entry["label_path"]
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
                gt_by_class.setdefault(cat_id, []).append(yolo_to_xyxy([cx, cy, bw, bh], width, height))

        pred_by_class = {}
        unknown = 0
        for det in detections:
            label = str(det.get("class_name") or "").strip().lower()
            cat_id = name_to_cat.get(label)
            if cat_id is None:
                unknown += 1
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x, y, w, h = [float(v) for v in bbox[:4]]
            pred_by_class.setdefault(cat_id, []).append([x, y, x + w, y + h])

        tp = fp = fn = 0
        for cat_id, gt_boxes in gt_by_class.items():
            preds = pred_by_class.get(cat_id, [])
            ctp, cfp, cfn = match_class(gt_boxes, preds, args.iou)
            tp += ctp
            fp += cfp
            fn += cfn
            yolo_overall["by_class"].setdefault(str(cat_id), {"tp": 0, "fp": 0, "fn": 0})
            yolo_overall["by_class"][str(cat_id)]["tp"] += ctp
            yolo_overall["by_class"][str(cat_id)]["fp"] += cfp
            yolo_overall["by_class"][str(cat_id)]["fn"] += cfn
        fp += unknown
        yolo_overall["tp"] += tp
        yolo_overall["fp"] += fp
        yolo_overall["fn"] += fn

    yolo_summary = {
        "count": len(baseline_items),
        "seed": args.seed,
        "iou": args.iou,
        "label_source": label_source,
        "yolo_root": str(args.yolo_root or ""),
        "tp": yolo_overall["tp"],
        "fp": yolo_overall["fp"],
        "fn": yolo_overall["fn"],
        "by_class": yolo_overall["by_class"],
        "output": str(yolo_summary_path),
    }
    yolo_summary_path.write_text(json.dumps(yolo_summary, indent=2))
    summary["yolo_baseline"] = {
        "tp": yolo_overall["tp"],
        "fp": yolo_overall["fp"],
        "fn": yolo_overall["fn"],
        "summary_path": str(yolo_summary_path),
    }
summary_path.write_text(json.dumps(summary, indent=2))
print("Benchmark complete.", flush=True)
print("Output:", out_path, flush=True)
print("Summary:", summary_path, flush=True)
if summary.get("yolo_baseline"):
    print("YOLO baseline:", summary["yolo_baseline"]["summary_path"], flush=True)
PY
