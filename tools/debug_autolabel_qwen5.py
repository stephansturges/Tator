#!/usr/bin/env python3
"""Debug harness for qwen auto-label bbox coverage on a fixed 5-image sample."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localinferenceapi import (  # noqa: E402
    AutoLabelJob,
    _annotation_effective_label_lines,
    _normalize_class_name_for_match,
    _resolve_dataset_entry,
    _run_auto_label_job,
    register_dataset_path,
)
from models.schemas import AutoLabelRequest  # noqa: E402
from services.auto_labeling import bbox_iou_xyxy, parse_yolo_label_line  # noqa: E402


QWEN_ROOT = Path("uploads/qwen_runs/datasets/qwen_dataset")
UPLOADS_ROOT = Path("uploads")
DEFAULT_SAMPLE_SEED = 20260410
DEFAULT_SAMPLE_SIZE = 5
CANONICAL_EDR_PACKAGE = "canonical_edr_pkg_qwen_dataset_8a922d9945b1"


def _load_labelmap() -> List[str]:
    return [ln.strip() for ln in (QWEN_ROOT / "labelmap.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]


def _sample_images(sample_size: int, seed: int) -> List[str]:
    labels_root = QWEN_ROOT / "train" / "labels"
    candidates: List[str] = []
    for label_path in sorted(labels_root.glob("*.txt")):
        lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue
        image_name = label_path.with_suffix(".jpg").name
        png_name = label_path.with_suffix(".png").name
        if (QWEN_ROOT / "train" / image_name).exists():
            candidates.append(image_name)
        elif (QWEN_ROOT / "train" / png_name).exists():
            candidates.append(png_name)
    rng = random.Random(seed)
    return rng.sample(candidates, sample_size)


def _stage_dataset(images: Sequence[str], *, dataset_root: Path) -> Path:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    (dataset_root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "val" / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "val" / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(QWEN_ROOT / "labelmap.txt", dataset_root / "labelmap.txt")
    for image_name in images:
        src = QWEN_ROOT / "train" / image_name
        if not src.exists():
            raise FileNotFoundError(f"missing source image: {src}")
        dst = dataset_root / "train" / "images" / image_name
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)
        (dataset_root / "train" / "labels" / Path(image_name).with_suffix(".txt")).write_text(
            "", encoding="utf-8"
        )
    return dataset_root


def _gt_lines(image_name: str) -> List[str]:
    path = QWEN_ROOT / "train" / "labels" / Path(image_name).with_suffix(".txt")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _image_size(image_name: str) -> tuple[int, int]:
    from PIL import Image

    path = QWEN_ROOT / "train" / image_name
    with Image.open(path) as img:
        return img.size


def _parse_lines(lines: Sequence[str], *, width: int, height: int, labelmap: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in lines:
        parsed = parse_yolo_label_line(line, width=width, height=height)
        if not parsed:
            continue
        class_id = int(parsed["class_id"])
        class_name = labelmap[class_id] if 0 <= class_id < len(labelmap) else str(class_id)
        out.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "bbox_xyxy": tuple(float(v) for v in parsed["bbox_xyxy"]),
                "label_line": str(line),
            }
        )
    return out


def _coverage_report(
    *,
    gt: Sequence[Dict[str, Any]],
    pred: Sequence[Dict[str, Any]],
    iou_threshold: float,
) -> Dict[str, Any]:
    matched = 0
    per_gt: List[Dict[str, Any]] = []
    for gt_item in gt:
        gt_cls = _normalize_class_name_for_match(gt_item["class_name"])
        best_iou = 0.0
        best_pred = None
        for pred_item in pred:
            if _normalize_class_name_for_match(pred_item["class_name"]) != gt_cls:
                continue
            iou = bbox_iou_xyxy(gt_item["bbox_xyxy"], pred_item["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_item
        hit = best_iou >= iou_threshold
        if hit:
            matched += 1
        per_gt.append(
            {
                "class_name": gt_item["class_name"],
                "gt_bbox_xyxy": gt_item["bbox_xyxy"],
                "best_iou": best_iou,
                "hit": hit,
                "best_pred_bbox_xyxy": None if best_pred is None else best_pred["bbox_xyxy"],
                "best_pred_class": None if best_pred is None else best_pred["class_name"],
            }
        )
    total = len(gt)
    return {
        "gt_total": total,
        "matched": matched,
        "coverage": (float(matched) / float(total)) if total else 1.0,
        "per_gt": per_gt,
    }


def _class_names_for_image(image_name: str, labelmap: Sequence[str]) -> List[str]:
    classes: List[str] = []
    seen = set()
    for line in _gt_lines(image_name):
        class_id = int(float(line.split()[0]))
        class_name = labelmap[class_id]
        if class_name in seen:
            continue
        seen.add(class_name)
        classes.append(class_name)
    return classes


def _run_image(
    *,
    dataset_id: str,
    image_name: str,
    class_names: Sequence[str],
    labelmap: Sequence[str],
    package_id: str,
    iou_threshold: float,
    falcon_backend: str,
    falcon_detection_strategy: str,
    falcon_component_mode: str,
    falcon_min_dimension: int,
    falcon_max_dimension: int,
    falcon_coord_dedup_threshold: float,
    falcon_hr_upsample_ratio: int,
    falcon_segmentation_threshold: float,
    enable_falcon: bool,
) -> Dict[str, Any]:
    payload = AutoLabelRequest(
        dataset_id=dataset_id,
        max_images=1,
        split="train",
        unlabeled_only=True,
        image_relpaths=[image_name],
        target_mode="detection",
        falcon_window_mode="quadrants",
        enable_falcon=bool(enable_falcon),
        falcon_overlap_ratio=0.1,
        dedupe_existing_same_class_iou=0.5,
        class_names=list(class_names),
        edr_package_id=package_id,
        falcon_backend=str(falcon_backend),
        falcon_detection_strategy=str(falcon_detection_strategy),
        falcon_component_mode=str(falcon_component_mode),
        falcon_min_dimension=int(falcon_min_dimension),
        falcon_max_dimension=int(falcon_max_dimension),
        falcon_coord_dedup_threshold=float(falcon_coord_dedup_threshold),
        falcon_hr_upsample_ratio=int(falcon_hr_upsample_ratio),
        falcon_segmentation_threshold=float(falcon_segmentation_threshold),
        enable_yolo=True,
        enable_rfdetr=True,
        use_planner_caption=False,
    )
    job = AutoLabelJob(job_id=f"dbg_{uuid.uuid4().hex[:8]}")
    started = time.time()
    _run_auto_label_job(job, payload)
    elapsed = time.time() - started
    width, height = _image_size(image_name)
    entry = _resolve_dataset_entry(dataset_id)
    pred_lines = _annotation_effective_label_lines(entry, "train", Path(image_name))
    gt_lines = _gt_lines(image_name)
    pred = _parse_lines(pred_lines, width=width, height=height, labelmap=labelmap)
    gt = _parse_lines(gt_lines, width=width, height=height, labelmap=labelmap)
    coverage = _coverage_report(gt=gt, pred=pred, iou_threshold=iou_threshold)
    return {
        "image_name": image_name,
        "class_names": list(class_names),
        "job_status": job.status,
        "job_error": job.error,
        "job_message": job.message,
        "elapsed_sec": elapsed,
        "job_result": job.result,
        "pred_count": len(pred),
        "gt_count": len(gt),
        "pred_lines": pred_lines,
        "coverage": coverage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-id", default=CANONICAL_EDR_PACKAGE)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--falcon-backend", default="embedded", choices=["embedded", "server"])
    parser.add_argument(
        "--falcon-detection-strategy",
        default="segmentation_boxes",
        choices=["native_detection", "segmentation_boxes"],
    )
    parser.add_argument(
        "--falcon-component-mode",
        default="component_split",
        choices=["largest_component", "component_split", "component_cluster"],
    )
    parser.add_argument("--falcon-min-dimension", type=int, default=256)
    parser.add_argument("--falcon-max-dimension", type=int, default=1024)
    parser.add_argument("--falcon-coord-dedup-threshold", type=float, default=0.01)
    parser.add_argument("--falcon-hr-upsample-ratio", type=int, default=8)
    parser.add_argument("--falcon-segmentation-threshold", type=float, default=0.3)
    parser.add_argument("--disable-falcon", action="store_true")
    args = parser.parse_args()

    sample_size = max(1, int(args.sample_size))
    sample_seed = int(args.sample_seed)
    images = _sample_images(sample_size, sample_seed)
    dataset_root = UPLOADS_ROOT / "datasets" / f"qwen_autolabel_debug{sample_size}_seed{sample_seed}"
    report_root = UPLOADS_ROOT / "auto_label_debug" / f"qwen_random{sample_size}_seed{sample_seed}_integrated"
    labelmap = _load_labelmap()
    report_root.mkdir(parents=True, exist_ok=True)
    _stage_dataset(images, dataset_root=dataset_root)
    entry = register_dataset_path(
        path=str(dataset_root.resolve()),
        dataset_id=f"qwen_autolabel_debug{sample_size}_seed{sample_seed}",
        label=f"qwen_autolabel_debug{sample_size}_seed{sample_seed}",
        context="auto_label_debug",
        notes=f"{sample_size}-image qwen bbox coverage debug harness",
        force_new=True,
        strict=True,
    )
    dataset_id = str(entry["id"])

    results: List[Dict[str, Any]] = []
    for image_name in images:
        result = _run_image(
            dataset_id=dataset_id,
            image_name=image_name,
            class_names=_class_names_for_image(image_name, labelmap),
            labelmap=labelmap,
            package_id=str(args.package_id),
            iou_threshold=float(args.iou_threshold),
            falcon_backend=str(args.falcon_backend),
            falcon_detection_strategy=str(args.falcon_detection_strategy),
            falcon_component_mode=str(args.falcon_component_mode),
            falcon_min_dimension=int(args.falcon_min_dimension),
            falcon_max_dimension=int(args.falcon_max_dimension),
            falcon_coord_dedup_threshold=float(args.falcon_coord_dedup_threshold),
            falcon_hr_upsample_ratio=int(args.falcon_hr_upsample_ratio),
            falcon_segmentation_threshold=float(args.falcon_segmentation_threshold),
            enable_falcon=not bool(args.disable_falcon),
        )
        results.append(result)

    overall_gt = sum(int(item["coverage"]["gt_total"]) for item in results)
    overall_matched = sum(int(item["coverage"]["matched"]) for item in results)
    summary = {
        "dataset_id": dataset_id,
        "package_id": str(args.package_id),
        "sample_seed": sample_seed,
        "sample_size": sample_size,
        "falcon_backend": str(args.falcon_backend),
        "falcon_detection_strategy": str(args.falcon_detection_strategy),
        "falcon_component_mode": str(args.falcon_component_mode),
        "falcon_min_dimension": int(args.falcon_min_dimension),
        "falcon_max_dimension": int(args.falcon_max_dimension),
        "falcon_coord_dedup_threshold": float(args.falcon_coord_dedup_threshold),
        "falcon_hr_upsample_ratio": int(args.falcon_hr_upsample_ratio),
        "falcon_segmentation_threshold": float(args.falcon_segmentation_threshold),
        "enable_falcon": not bool(args.disable_falcon),
        "images": images,
        "overall_gt_total": overall_gt,
        "overall_matched": overall_matched,
        "overall_coverage": (float(overall_matched) / float(overall_gt)) if overall_gt else 1.0,
        "results": results,
    }
    out_path = report_root / "coverage_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if overall_matched != overall_gt:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
