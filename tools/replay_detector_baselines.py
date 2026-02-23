#!/usr/bin/env python3
"""Compute true detector-only baselines by replaying detector inference.

This avoids attribution artifacts from post-dedupe prepass clusters.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import localinferenceapi as api
from tools.eval_ensemble_xgb_dedupe import _evaluate_predictions, _yolo_to_xyxy


def _resolve_image_path(dataset_id: str, image_name: str) -> Optional[Path]:
    dataset_root = Path("uploads/qwen_runs/datasets") / dataset_id
    for split in ("val", "train"):
        candidate = dataset_root / split / image_name
        if candidate.exists():
            return candidate
    return None


def _load_labelmap(dataset_id: str) -> List[str]:
    labelmap_path = Path(f"uploads/clip_dataset_uploads/{dataset_id}_yolo/labelmap.txt")
    if not labelmap_path.exists():
        raise SystemExit(f"Missing labelmap: {labelmap_path}")
    return [line.strip().lower() for line in labelmap_path.read_text().splitlines() if line.strip()]


def _load_job_val_images(job_id: str) -> List[str]:
    meta_path = Path("uploads/calibration_jobs") / job_id / "ensemble_xgb.meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing XGB meta for job {job_id}: {meta_path}")
    meta = json.loads(meta_path.read_text())
    images = [str(x) for x in (meta.get("split_val_images") or []) if str(x).strip()]
    if not images:
        raise SystemExit(f"No validation images found in {meta_path}")
    return sorted(set(images))


def _encode_image_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _detector_input_kwargs(pil_img: Image.Image) -> Dict[str, Any]:
    token = api._calibration_cache_image(pil_img, None)
    if token:
        return {"image_token": token}
    return {"image_base64": _encode_image_base64(pil_img)}


def _normalize_detection(det: Dict[str, Any], default_source: str) -> Optional[Dict[str, Any]]:
    if not isinstance(det, dict):
        return None
    label = str(det.get("label") or "").strip().lower()
    bbox = det.get("bbox_xyxy_px")
    if not label or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        score = float(det.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    source = str(det.get("score_source") or det.get("source") or default_source).strip().lower() or default_source
    source_list_raw = det.get("source_list")
    source_list: List[str] = []
    if isinstance(source_list_raw, (list, tuple)):
        source_list = [str(s).strip().lower() for s in source_list_raw if str(s).strip()]
    if source not in source_list:
        source_list.append(source)
    return {
        "label": label,
        "bbox_xyxy_px": [float(v) for v in bbox[:4]],
        "score": score,
        "score_source": source,
        "source": source,
        "source_list": sorted(set(source_list)),
    }


def _run_detector(
    *,
    pil_img: Image.Image,
    labelmap: Sequence[str],
    mode: str,
    conf: float,
    sahi_enabled: bool,
    sahi_size: int,
    sahi_overlap: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    kwargs = _detector_input_kwargs(pil_img)
    sahi_cfg: Dict[str, Any] = {"enabled": bool(sahi_enabled)}
    if sahi_enabled:
        sahi_cfg["slice_size"] = int(sahi_size)
        sahi_cfg["overlap"] = float(sahi_overlap)
    result = api._agent_tool_run_detector(
        mode=mode,
        conf=float(conf),
        sahi=sahi_cfg,
        expected_labelmap=list(labelmap),
        register=False,
        **kwargs,
    )
    warnings = list(result.get("warnings") or [])
    dets: List[Dict[str, Any]] = []
    for item in result.get("detections") or []:
        norm = _normalize_detection(item, default_source=mode)
        if norm is not None:
            dets.append(norm)
    return dets, warnings


def _load_gt_by_image(dataset_id: str, images: Sequence[str], labelmap: Sequence[str]) -> Dict[str, Dict[int, List[List[float]]]]:
    yolo_root = Path(f"uploads/clip_dataset_uploads/{dataset_id}_yolo")
    out: Dict[str, Dict[int, List[List[float]]]] = {}
    for image in images:
        img_path = _resolve_image_path(dataset_id, image)
        if img_path is None:
            continue
        label_path = yolo_root / "labels" / "val" / f"{Path(image).stem}.txt"
        if not label_path.exists():
            label_path = yolo_root / "labels" / "train" / f"{Path(image).stem}.txt"
        if not label_path.exists():
            continue
        try:
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
            out[image] = gt_by_class
    return out


def _build_preds(
    det_cache: Dict[str, Dict[str, Any]],
    images: Sequence[str],
    key: str,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for image in images:
        row = det_cache.get(image) or {}
        dets = list(row.get(key) or [])
        if dets:
            out[image] = dets
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay detector-only baselines for calibration jobs.")
    parser.add_argument("--dataset", default="qwen_dataset", help="Dataset id.")
    parser.add_argument("--jobs", required=True, help="Comma-separated calibration job ids.")
    parser.add_argument("--conf", type=float, default=0.45, help="Detector confidence.")
    parser.add_argument("--sahi-size", type=int, default=640, help="SAHI tile size.")
    parser.add_argument("--sahi-overlap", type=float, default=0.2, help="SAHI overlap.")
    parser.add_argument("--eval-iou", type=float, default=0.5, help="Evaluation IoU.")
    parser.add_argument("--dedupe-iou", type=float, default=0.75, help="Dedupe IoU.")
    parser.add_argument("--scoreless-iou", type=float, default=0.0, help="Scoreless overlap pruning.")
    parser.add_argument(
        "--cache-json",
        default="uploads/calibration_cache/inventory/detector_replay_cache_qwen_dataset.json",
        help="Detector replay cache path.",
    )
    parser.add_argument(
        "--output-json",
        default="uploads/calibration_cache/inventory/detector_replay_baselines_iou0.50.json",
        help="Summary output path.",
    )
    args = parser.parse_args()

    job_ids = [j.strip() for j in str(args.jobs).split(",") if j.strip()]
    if not job_ids:
        raise SystemExit("No jobs provided.")

    labelmap = _load_labelmap(args.dataset)
    name_to_cat = {name: idx for idx, name in enumerate(labelmap)}

    job_images: Dict[str, List[str]] = {job_id: _load_job_val_images(job_id) for job_id in job_ids}
    all_images = sorted({image for images in job_images.values() for image in images})

    cache_path = Path(args.cache_json)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    det_cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            det_cache = payload.get("images") or {}
        except Exception:
            det_cache = {}

    print(
        f"[replay] images total={len(all_images)} cached={sum(1 for img in all_images if img in det_cache)} "
        f"dataset={args.dataset} conf={args.conf} sahi={args.sahi_size}@{args.sahi_overlap}",
        flush=True,
    )

    start = time.time()
    for idx, image_name in enumerate(all_images, start=1):
        row = det_cache.get(image_name)
        if isinstance(row, dict) and "yolo" in row and "rfdetr" in row:
            if idx % 25 == 0 or idx == len(all_images):
                print(f"[replay] {idx}/{len(all_images)} (cached)", flush=True)
            continue
        img_path = _resolve_image_path(args.dataset, image_name)
        if img_path is None:
            det_cache[image_name] = {"yolo": [], "rfdetr": [], "warnings": ["image_not_found"]}
            continue
        with Image.open(img_path) as img:
            pil_img = img.convert("RGB")
        yolo_full, warn_yolo_full = _run_detector(
            pil_img=pil_img,
            labelmap=labelmap,
            mode="yolo",
            conf=float(args.conf),
            sahi_enabled=False,
            sahi_size=int(args.sahi_size),
            sahi_overlap=float(args.sahi_overlap),
        )
        yolo_sahi, warn_yolo_sahi = _run_detector(
            pil_img=pil_img,
            labelmap=labelmap,
            mode="yolo",
            conf=float(args.conf),
            sahi_enabled=True,
            sahi_size=int(args.sahi_size),
            sahi_overlap=float(args.sahi_overlap),
        )
        rfdetr_full, warn_rf_full = _run_detector(
            pil_img=pil_img,
            labelmap=labelmap,
            mode="rfdetr",
            conf=float(args.conf),
            sahi_enabled=False,
            sahi_size=int(args.sahi_size),
            sahi_overlap=float(args.sahi_overlap),
        )
        rfdetr_sahi, warn_rf_sahi = _run_detector(
            pil_img=pil_img,
            labelmap=labelmap,
            mode="rfdetr",
            conf=float(args.conf),
            sahi_enabled=True,
            sahi_size=int(args.sahi_size),
            sahi_overlap=float(args.sahi_overlap),
        )
        det_cache[image_name] = {
            "yolo": yolo_full + yolo_sahi,
            "rfdetr": rfdetr_full + rfdetr_sahi,
            "warnings": warn_yolo_full + warn_yolo_sahi + warn_rf_full + warn_rf_sahi,
        }
        if idx % 10 == 0 or idx == len(all_images):
            elapsed = max(1e-6, time.time() - start)
            rate = idx / elapsed
            print(f"[replay] {idx}/{len(all_images)} rate={rate:.2f} img/s", flush=True)
            cache_payload = {
                "dataset_id": args.dataset,
                "conf": float(args.conf),
                "sahi_size": int(args.sahi_size),
                "sahi_overlap": float(args.sahi_overlap),
                "images": det_cache,
            }
            cache_path.write_text(json.dumps(cache_payload))

    # Final cache write.
    cache_path.write_text(
        json.dumps(
            {
                "dataset_id": args.dataset,
                "conf": float(args.conf),
                "sahi_size": int(args.sahi_size),
                "sahi_overlap": float(args.sahi_overlap),
                "images": det_cache,
            }
        )
    )

    gt_by_image = _load_gt_by_image(args.dataset, all_images, labelmap)
    out: Dict[str, Any] = {
        "dataset_id": args.dataset,
        "generated_at": int(time.time()),
        "eval_iou": float(args.eval_iou),
        "dedupe_iou": float(args.dedupe_iou),
        "scoreless_iou": float(args.scoreless_iou),
        "jobs": {},
    }

    for job_id in job_ids:
        images = job_images[job_id]
        yolo_preds = _build_preds(det_cache, images, "yolo")
        rfdetr_preds = _build_preds(det_cache, images, "rfdetr")
        union_preds: Dict[str, List[Dict[str, Any]]] = {}
        for image in images:
            y = list((det_cache.get(image) or {}).get("yolo") or [])
            r = list((det_cache.get(image) or {}).get("rfdetr") or [])
            if y or r:
                union_preds[image] = y + r

        gt_subset = {image: gt_by_image.get(image, {}) for image in images if image in gt_by_image}

        yolo_metrics = _evaluate_predictions(
            yolo_preds,
            gt_subset,
            name_to_cat=name_to_cat,
            dedupe_iou=float(args.dedupe_iou),
            eval_iou=float(args.eval_iou),
            scoreless_iou=float(args.scoreless_iou),
        )
        rfdetr_metrics = _evaluate_predictions(
            rfdetr_preds,
            gt_subset,
            name_to_cat=name_to_cat,
            dedupe_iou=float(args.dedupe_iou),
            eval_iou=float(args.eval_iou),
            scoreless_iou=float(args.scoreless_iou),
        )
        union_metrics = _evaluate_predictions(
            union_preds,
            gt_subset,
            name_to_cat=name_to_cat,
            dedupe_iou=float(args.dedupe_iou),
            eval_iou=float(args.eval_iou),
            scoreless_iou=float(args.scoreless_iou),
        )

        eval_path = Path("uploads/calibration_jobs") / job_id / "ensemble_xgb.eval.json"
        xgb_metrics: Dict[str, Any] = {}
        if eval_path.exists():
            try:
                eval_payload = json.loads(eval_path.read_text())
                xgb_metrics = ((eval_payload.get("reference_iou") or {}).get("xgb_ensemble") or {})
            except Exception:
                xgb_metrics = {}

        out["jobs"][job_id] = {
            "images": len(images),
            "gt_images": len(gt_subset),
            "baselines_detector_replay": {
                "yolo": yolo_metrics,
                "rfdetr": rfdetr_metrics,
                "yolo_rfdetr_union": union_metrics,
            },
            "xgb_ensemble": xgb_metrics,
        }
        print(
            f"[metrics] {job_id} "
            f"yolo_f1={yolo_metrics.get('f1', 0.0):.3f} "
            f"rfdetr_f1={rfdetr_metrics.get('f1', 0.0):.3f} "
            f"union_f1={union_metrics.get('f1', 0.0):.3f} "
            f"xgb_f1={float(xgb_metrics.get('f1') or 0.0):.3f}",
            flush=True,
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
