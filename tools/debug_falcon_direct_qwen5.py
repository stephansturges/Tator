#!/usr/bin/env python3
"""Direct Falcon-only qwen diagnostic with prompt/output logging."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import localinferenceapi as _lia  # noqa: E402
from localinferenceapi import (  # noqa: E402
    AutoLabelRequest,
    _auto_label_build_quadrant_windows,
    _auto_label_dedupe_candidates,
    _auto_label_falcon_candidates_for_window,
    _normalize_class_name_for_match,
)
from services.auto_labeling import (  # noqa: E402
    AUTO_LABEL_TARGET_MODE_DETECTION,
    bbox_iou_xyxy,
    build_falcon_query_tiers,
    parse_yolo_label_line,
)
from services.falcon_perception import (  # noqa: E402
    unload_official_falcon_runtime,
)


QWEN_ROOT = Path("uploads/qwen_runs/datasets/qwen_dataset")
REPORT_BASE = Path("uploads/auto_label_debug")


def _to_jsonable(value: Any) -> Any:
    try:
        import numpy as np  # local import
    except Exception:  # pragma: no cover
        np = None  # type: ignore[assignment]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return repr(value)


def _load_labelmap() -> List[str]:
    return [
        ln.strip()
        for ln in (QWEN_ROOT / "labelmap.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def _load_glossary() -> str:
    try:
        payload = json.loads((QWEN_ROOT / "metadata.json").read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(payload.get("labelmap_glossary") or "")


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


def _resolve_images(
    *,
    sample_size: int,
    seed: int,
    image_names: Sequence[str],
) -> List[str]:
    cleaned = [str(name or "").strip() for name in (image_names or []) if str(name or "").strip()]
    if cleaned:
        return cleaned
    return _sample_images(sample_size, seed)


def _gt_lines(image_name: str) -> List[str]:
    path = QWEN_ROOT / "train" / "labels" / Path(image_name).with_suffix(".txt")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _image_size(image_name: str) -> tuple[int, int]:
    with Image.open(QWEN_ROOT / "train" / image_name) as img:
        return img.size


def _parse_lines(
    lines: Sequence[str],
    *,
    width: int,
    height: int,
    labelmap: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in lines:
        parsed = parse_yolo_label_line(line, width=width, height=height)
        if not parsed:
            continue
        class_id = int(parsed["class_id"])
        out.append(
            {
                "class_id": class_id,
                "class_name": labelmap[class_id] if 0 <= class_id < len(labelmap) else str(class_id),
                "bbox_xyxy": tuple(float(v) for v in parsed["bbox_xyxy"]),
                "label_line": str(line),
            }
        )
    return out


def _class_names_for_image(image_name: str, labelmap: Sequence[str]) -> List[str]:
    names: List[str] = []
    seen = set()
    width, height = _image_size(image_name)
    for item in _parse_lines(_gt_lines(image_name), width=width, height=height, labelmap=labelmap):
        name = str(item["class_name"])
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _windows_for_mode(
    *,
    width: int,
    height: int,
    window_mode: str,
) -> List[Dict[str, Any]]:
    mode = str(window_mode or "quadrants").strip().lower()
    if mode == "full_image":
        return [{"id": "FULL", "xyxy": (0.0, 0.0, float(width), float(height))}]
    return _auto_label_build_quadrant_windows(width, height, overlap_ratio=0.1)


@contextmanager
def _patched_query_tiers(*, prompt_style: str, query_frame: str):
    original = _lia._auto_label_build_falcon_query_tiers
    _lia._auto_label_build_falcon_query_tiers = lambda class_names, labelmap, glossary="": build_falcon_query_tiers(  # type: ignore[assignment]
        class_names,
        labelmap=labelmap,
        glossary=glossary or "",
        prompt_style=prompt_style,
        query_frame=query_frame,
    )
    try:
        yield
    finally:
        _lia._auto_label_build_falcon_query_tiers = original


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
                "best_pred_query": None if best_pred is None else best_pred.get("query"),
                "best_pred_term": None if best_pred is None else best_pred.get("term"),
                "best_pred_window_id": None if best_pred is None else best_pred.get("window_id"),
            }
        )
    total = len(gt)
    return {
        "gt_total": total,
        "matched": matched,
        "coverage": (float(matched) / float(total)) if total else 1.0,
        "mean_best_iou": (
            float(sum(float(item["best_iou"]) for item in per_gt)) / float(total)
        ) if total else 1.0,
        "per_gt": per_gt,
    }


def _whole_window_fraction(candidates: Sequence[Dict[str, Any]], *, image_width: int, image_height: int) -> float:
    if not candidates:
        return 0.0
    full_area = float(max(1, image_width * image_height))
    huge = 0
    for item in candidates:
        x1, y1, x2, y2 = [float(v) for v in (item.get("bbox_xyxy") or (0, 0, 0, 0))]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / full_area
        if area_ratio >= 0.9:
            huge += 1
    return float(huge) / float(len(candidates))


def _degenerate_fraction(candidates: Sequence[Dict[str, Any]]) -> float:
    if not candidates:
        return 0.0
    bad = 0
    for item in candidates:
        area_fraction = float(item.get("bbox_area_fraction_crop") or 0.0)
        border_touch = int(item.get("border_touch_count") or 0)
        if area_fraction >= 0.85 or (border_touch >= 2 and area_fraction >= 0.05):
            bad += 1
    return float(bad) / float(len(candidates))


def _run_image(
    *,
    image_name: str,
    labelmap: Sequence[str],
    glossary: str,
    falcon_model_id: str,
    falcon_device: str,
    iou_threshold: float,
    falcon_backend: str,
    falcon_detection_strategy: str,
    falcon_component_mode: str,
    falcon_min_dimension: int,
    falcon_max_dimension: int,
    falcon_coord_dedup_threshold: float,
    falcon_hr_upsample_ratio: int,
    falcon_segmentation_threshold: float,
    window_mode: str,
    prompt_style: str,
    query_frame: str,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    image_path = QWEN_ROOT / "train" / image_name
    pil_img = Image.open(image_path).convert("RGB")
    img_w, img_h = pil_img.size
    gt = _parse_lines(_gt_lines(image_name), width=img_w, height=img_h, labelmap=labelmap)
    class_names = _class_names_for_image(image_name, labelmap)
    class_id_map = {str(name): idx for idx, name in enumerate(labelmap)}
    windows = _windows_for_mode(width=img_w, height=img_h, window_mode=window_mode)
    payload = AutoLabelRequest(
        dataset_id="qwen_dataset",
        target_mode="detection",
        class_names=list(class_names),
        falcon_window_mode="quadrants",
        falcon_model_id=str(falcon_model_id),
        falcon_device=str(falcon_device),
        falcon_backend=str(falcon_backend),
        falcon_detection_strategy=str(falcon_detection_strategy),
        falcon_component_mode=str(falcon_component_mode),
        falcon_min_dimension=int(falcon_min_dimension),
        falcon_max_dimension=int(falcon_max_dimension),
        falcon_coord_dedup_threshold=float(falcon_coord_dedup_threshold),
        falcon_hr_upsample_ratio=int(falcon_hr_upsample_ratio),
        falcon_segmentation_threshold=float(falcon_segmentation_threshold),
        enable_yolo=False,
        enable_rfdetr=False,
    )

    raw_candidates: List[Dict[str, Any]] = []
    window_logs: List[Dict[str, Any]] = []
    total_query_count = 0
    with _patched_query_tiers(prompt_style=prompt_style, query_frame=query_frame):
        for window in windows:
            window_result = _auto_label_falcon_candidates_for_window(
                pil_img=pil_img,
                crop_window=window,
                class_names=class_names,
                class_id_map=class_id_map,
                labelmap=labelmap,
                glossary=glossary or "",
                payload=payload,
                target_mode=AUTO_LABEL_TARGET_MODE_DETECTION,
            )
            raw_candidates.extend(list(window_result.get("candidates") or []))
            total_query_count += int(window_result.get("query_count") or 0)
            window_logs.append(
                {
                    "window_id": str(window.get("id") or ""),
                    "xyxy": window.get("xyxy"),
                    "diagnostics": window_result.get("diagnostics") or [],
                }
            )

    kept, dropped = _auto_label_dedupe_candidates(
        raw_candidates,
        existing=[],
        target_mode=AUTO_LABEL_TARGET_MODE_DETECTION,
        iou_threshold=0.5,
    )
    raw_coverage = _coverage_report(gt=gt, pred=raw_candidates, iou_threshold=iou_threshold)
    kept_coverage = _coverage_report(gt=gt, pred=kept, iou_threshold=iou_threshold)
    query_counter = Counter(item["query"] for item in raw_candidates if str(item.get("query") or "").strip())
    class_counter = Counter(item["class_name"] for item in kept if str(item.get("class_name") or "").strip())

    return {
        "image_name": image_name,
        "class_names": class_names,
        "gt_count": len(gt),
        "elapsed_sec": float(time.perf_counter() - started_at),
        "window_mode": str(window_mode or ""),
        "prompt_style": str(prompt_style or ""),
        "query_frame": str(query_frame or ""),
        "query_count": int(total_query_count),
        "window_logs": window_logs,
        "raw_candidate_count": len(raw_candidates),
        "dedup_candidate_count": len(kept),
        "duplicates_dropped": int(dropped),
        "raw_whole_window_fraction": _whole_window_fraction(
            raw_candidates,
            image_width=img_w,
            image_height=img_h,
        ),
        "raw_degenerate_fraction": _degenerate_fraction(raw_candidates),
        "dedup_degenerate_fraction": _degenerate_fraction(kept),
        "raw_coverage": raw_coverage,
        "dedup_coverage": kept_coverage,
        "query_hit_counts": dict(query_counter),
        "kept_class_counts": dict(class_counter),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=20260410)
    parser.add_argument("--image-name", action="append", default=[])
    parser.add_argument("--window-mode", default="quadrants", choices=["full_image", "quadrants"])
    parser.add_argument(
        "--prompt-style",
        default="default",
        choices=["default", "canonical_only", "priority_only", "priority_top1", "glossary_only"],
    )
    parser.add_argument(
        "--query-frame",
        default="term",
        choices=["term", "all_instances", "the"],
    )
    parser.add_argument("--falcon-model-id", default="tiiuae/Falcon-Perception")
    parser.add_argument("--falcon-device", default="cuda:1")
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
    parser.add_argument("--falcon-server-url", default="")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--report-root", default="")
    args = parser.parse_args()
    if str(args.falcon_server_url or "").strip():
        os.environ["FALCON_SERVER_URL"] = str(args.falcon_server_url).strip()

    labelmap = _load_labelmap()
    glossary = _load_glossary()
    images = _resolve_images(
        sample_size=int(args.sample_size),
        seed=int(args.sample_seed),
        image_names=args.image_name or [],
    )
    if args.image_name:
        run_id = "falcon_direct_qwen_image_" + "_".join(Path(name).stem for name in images[:4])
    else:
        run_id = f"falcon_direct_qwen_random{args.sample_size}_seed{args.sample_seed}"
    report_root = Path(str(args.report_root).strip()) if str(args.report_root or "").strip() else (REPORT_BASE / run_id)
    report_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    try:
        for image_name in images:
            result = _run_image(
                image_name=image_name,
                labelmap=labelmap,
                glossary=glossary,
                falcon_model_id=str(args.falcon_model_id),
                falcon_device=str(args.falcon_device),
                iou_threshold=float(args.iou_threshold),
                falcon_backend=str(args.falcon_backend),
                falcon_detection_strategy=str(args.falcon_detection_strategy),
                falcon_component_mode=str(args.falcon_component_mode),
                falcon_min_dimension=int(args.falcon_min_dimension),
                falcon_max_dimension=int(args.falcon_max_dimension),
                falcon_coord_dedup_threshold=float(args.falcon_coord_dedup_threshold),
                falcon_hr_upsample_ratio=int(args.falcon_hr_upsample_ratio),
                falcon_segmentation_threshold=float(args.falcon_segmentation_threshold),
                window_mode=str(args.window_mode),
                prompt_style=str(args.prompt_style),
                query_frame=str(args.query_frame),
            )
            results.append(result)
            partial = {
                "sample_seed": int(args.sample_seed),
                "sample_size": int(args.sample_size),
                "images": images,
                "results_completed": len(results),
                "results": results,
            }
            (report_root / "partial_report.json").write_text(
                json.dumps(_to_jsonable(partial), indent=2),
                encoding="utf-8",
            )
    finally:
        unload_official_falcon_runtime()

    raw_hits = sum(int(item["raw_coverage"]["matched"]) for item in results)
    dedup_hits = sum(int(item["dedup_coverage"]["matched"]) for item in results)
    gt_total = sum(int(item["gt_count"]) for item in results)
    summary = {
        "sample_seed": int(args.sample_seed),
        "sample_size": int(args.sample_size),
        "falcon_backend": str(args.falcon_backend),
        "falcon_detection_strategy": str(args.falcon_detection_strategy),
        "falcon_component_mode": str(args.falcon_component_mode),
        "window_mode": str(args.window_mode),
        "prompt_style": str(args.prompt_style),
        "query_frame": str(args.query_frame),
        "falcon_min_dimension": int(args.falcon_min_dimension),
        "falcon_max_dimension": int(args.falcon_max_dimension),
        "falcon_coord_dedup_threshold": float(args.falcon_coord_dedup_threshold),
        "falcon_hr_upsample_ratio": int(args.falcon_hr_upsample_ratio),
        "falcon_segmentation_threshold": float(args.falcon_segmentation_threshold),
        "falcon_server_url": str(args.falcon_server_url or ""),
        "images": images,
        "overall": {
            "gt_total": gt_total,
            "raw_hits": raw_hits,
            "dedup_hits": dedup_hits,
            "raw_coverage": (float(raw_hits) / float(gt_total)) if gt_total else 1.0,
            "dedup_coverage": (float(dedup_hits) / float(gt_total)) if gt_total else 1.0,
            "mean_best_iou": (
                float(
                    sum(
                        float(item["dedup_coverage"].get("mean_best_iou") or 0.0)
                        * float(item["gt_count"] or 0)
                        for item in results
                    )
                )
                / float(gt_total)
            ) if gt_total else 1.0,
            "whole_window_fraction_mean": (
                float(sum(float(item["raw_whole_window_fraction"]) for item in results)) / float(len(results))
            ) if results else 0.0,
            "degenerate_fraction_mean": (
                float(sum(float(item["raw_degenerate_fraction"]) for item in results)) / float(len(results))
            ) if results else 0.0,
            "avg_elapsed_sec": (
                float(sum(float(item["elapsed_sec"]) for item in results)) / float(len(results))
            ) if results else 0.0,
        },
        "results": results,
    }
    out_path = report_root / "report.json"
    out_path.write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(_to_jsonable(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
