#!/usr/bin/env python3
"""Run a random 5-image Falcon auto-label diagnostic with prompt/output logging."""

from __future__ import annotations

import argparse
import base64
import json
import random
import shutil
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import services.falcon_perception as falcon_runtime  # noqa: E402
from localinferenceapi import (  # noqa: E402
    _annotation_effective_label_lines,
    _auto_label_build_existing_annotations,
    _auto_label_build_quadrant_windows,
    _auto_label_dedupe_candidates,
    _auto_label_falcon_candidates_for_window,
    _normalize_class_name_for_match,
    _resolve_dataset_entry,
    _run_prepass_annotation_qwen,
    register_dataset_path,
)
from models.schemas import AutoLabelRequest, QwenPrepassRequest  # noqa: E402
from services.auto_labeling import bbox_iou_xyxy, parse_yolo_label_line  # noqa: E402


QWEN_ROOT = Path("uploads/qwen_runs/datasets/qwen_dataset")
REPORT_BASE = Path("uploads/auto_label_debug")
CANONICAL_EDR_PACKAGE = "canonical_edr_pkg_qwen_dataset_8a922d9945b1"
PREVIOUS_FIXED_SAMPLE = {
    "18639622-4ca4-4d15-8a91-b0014406888f.jpg",
    "ddedfdc9-06da-4464-bac3-4c56db3b5613.jpg",
    "d96bd945-6f75-4881-9c01-552f5c4739b5.jpg",
    "b6eacc8f-b0cf-455e-a3ae-c5a5022c54b7.png",
    "901f0387-6553-4c2e-9cb5-05baa0e42ed7.png",
}


def _to_jsonable(value: Any) -> Any:
    try:
        import numpy as np  # local import
        import torch  # local import
    except Exception:  # pragma: no cover
        np = None  # type: ignore[assignment]
        torch = None  # type: ignore[assignment]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "tensor_shape": list(value.shape),
            "tensor_dtype": str(value.dtype),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return repr(value)


def _load_labelmap() -> List[str]:
    return [ln.strip() for ln in (QWEN_ROOT / "labelmap.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]


def _sample_images(sample_size: int, seed: int) -> List[str]:
    labels_root = QWEN_ROOT / "train" / "labels"
    candidates: List[str] = []
    for label_path in sorted(labels_root.glob("*.txt")):
        image_name = label_path.with_suffix(".jpg").name
        png_name = label_path.with_suffix(".png").name
        if (QWEN_ROOT / "train" / image_name).exists():
            chosen = image_name
        elif (QWEN_ROOT / "train" / png_name).exists():
            chosen = png_name
        else:
            continue
        if chosen in PREVIOUS_FIXED_SAMPLE:
            continue
        lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue
        candidates.append(chosen)
    rng = random.Random(seed)
    return rng.sample(candidates, sample_size)


def _stage_dataset(images: Sequence[str], *, dataset_root: Path) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    (dataset_root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "val" / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "val" / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(QWEN_ROOT / "labelmap.txt", dataset_root / "labelmap.txt")
    for image_name in images:
        src = QWEN_ROOT / "train" / image_name
        dst = dataset_root / "train" / "images" / image_name
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)
        (dataset_root / "train" / "labels" / Path(image_name).with_suffix(".txt")).write_text(
            "", encoding="utf-8"
        )


def _image_size(image_name: str) -> Tuple[int, int]:
    with Image.open(QWEN_ROOT / "train" / image_name) as img:
        return img.size


def _gt_lines(image_name: str) -> List[str]:
    path = QWEN_ROOT / "train" / "labels" / Path(image_name).with_suffix(".txt")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _parse_lines(lines: Sequence[str], *, width: int, height: int, labelmap: Sequence[str]) -> List[Dict[str, Any]]:
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
    for item in _parse_lines(_gt_lines(image_name), width=_image_size(image_name)[0], height=_image_size(image_name)[1], labelmap=labelmap):
        name = str(item["class_name"])
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


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
                "best_pred_source_primary": None if best_pred is None else best_pred.get("source_primary"),
                "best_pred_source_list": None if best_pred is None else best_pred.get("source_list"),
            }
        )
    total = len(gt)
    return {
        "gt_total": total,
        "matched": matched,
        "coverage": (float(matched) / float(total)) if total else 1.0,
        "per_gt": per_gt,
    }


def _candidate_from_det(det: Dict[str, Any], *, img_w: int, img_h: int, class_names: Sequence[str], class_id_map: Dict[str, int]) -> Optional[Dict[str, Any]]:
    class_name = str(det.get("label") or "").strip()
    if class_name not in class_names:
        return None
    bbox = det.get("bbox_yolo")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x, y, w, h = bbox[:4]
    x1 = (float(x) - float(w) / 2.0) * img_w
    y1 = (float(y) - float(h) / 2.0) * img_h
    x2 = (float(x) + float(w) / 2.0) * img_w
    y2 = (float(y) + float(h) / 2.0) * img_h
    source_list = [str(item).strip() for item in (det.get("source_list") or []) if str(item).strip()]
    source_primary = str(det.get("source_primary") or det.get("source") or "").strip() or "baseline"
    return {
        "class_id": int(class_id_map.get(class_name, -1)),
        "class_name": class_name,
        "bbox_xyxy": (x1, y1, x2, y2),
        "mask": None,
        "score": float(det.get("score") or 0.5),
        "source": "baseline",
        "source_primary": source_primary,
        "source_list": source_list or [source_primary],
        "origin": det.get("origin"),
        "score_source": det.get("score_source"),
        "window_id": det.get("owner_cell") or det.get("grid_cell"),
    }


def _summarize_prediction_batch(batch: Any) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    if not isinstance(batch, list):
        return summary
    for idx, item in enumerate(batch):
        row_summary: Dict[str, Any] = {
            "query_index": idx,
            "item_type": type(item).__name__,
            "prediction_count": len(item) if isinstance(item, list) else 0,
        }
        if isinstance(item, list) and item:
            first = item[0]
            row_summary["sample_keys"] = sorted(first.keys()) if isinstance(first, dict) else []
            if isinstance(first, dict):
                row_summary["sample_pred"] = {
                    "xy": first.get("xy"),
                    "hw": first.get("hw"),
                    "has_mask_rle": isinstance(first.get("mask_rle"), dict),
                    "mask_size": (first.get("mask_rle") or {}).get("size") if isinstance(first.get("mask_rle"), dict) else None,
                }
        summary.append(row_summary)
    return summary


def _build_falcon_logging_wrapper(log_sink: List[Dict[str, Any]]):
    original = falcon_runtime.run_falcon_queries

    def _wrapped(
        *,
        pil_image: Any,
        queries: Sequence[str],
        model_id: str,
        device: str,
        local_files_only: bool = True,
        compile_model: bool = False,
        min_dimension: int = 256,
        max_dimension: int = 1024,
        max_new_tokens: int = 1024,
        task: str = "segmentation",
    ) -> List[List[Dict[str, Any]]]:
        width, height = pil_image.size
        dims = falcon_runtime._falcon_retry_dimensions(int(max_dimension), int(min_dimension))
        raw_batch = None
        normalized = None
        used_dim = None
        last_exc: Optional[str] = None
        try:
            if str(task or "segmentation").strip().lower() == "detection":
                normalized = original(
                    pil_image=pil_image,
                    queries=queries,
                    model_id=model_id,
                    device=device,
                    local_files_only=local_files_only,
                    compile_model=compile_model,
                    min_dimension=min_dimension,
                    max_dimension=max_dimension,
                    max_new_tokens=max_new_tokens,
                    task=task,
                )
                return normalized
            falcon_runtime._configure_falcon_torch_runtime(falcon_runtime.torch)
            for dim in dims:
                try:
                    model = falcon_runtime.ensure_falcon_runtime(
                        model_id=model_id,
                        device=device,
                        local_files_only=local_files_only,
                    )
                    raw_batch = model.generate(
                        [pil_image] * len(queries),
                        list(queries),
                        max_new_tokens=int(max_new_tokens),
                        min_dimension=int(min_dimension),
                        max_dimension=int(dim),
                        compile=bool(compile_model),
                    )
                    normalized = falcon_runtime._normalize_prediction_batch(raw_batch, width=width, height=height)
                    used_dim = int(dim)
                    return normalized
                except Exception as exc:  # noqa: BLE001
                    last_exc = str(exc)
                    if falcon_runtime._is_cuda_oom_error(exc) and dim != dims[-1]:
                        falcon_runtime.unload_falcon_runtime()
                        continue
                    raise
        finally:
            log_sink.append(
                {
                    "queries": list(queries),
                    "image_size": [int(width), int(height)],
                    "params": {
                        "model_id": str(model_id),
                        "device": str(device),
                        "local_files_only": bool(local_files_only),
                        "compile_model": bool(compile_model),
                        "min_dimension": int(min_dimension),
                        "max_dimension": int(max_dimension),
                        "max_new_tokens": int(max_new_tokens),
                        "task": str(task),
                        "retry_dimensions": dims,
                        "used_dimension": used_dim,
                    },
                    "raw_output_summary": _summarize_prediction_batch(raw_batch),
                    "normalized_output_summary": _summarize_prediction_batch(normalized),
                    "error": last_exc,
                }
            )

        return original(
            pil_image=pil_image,
            queries=queries,
            model_id=model_id,
            device=device,
            local_files_only=local_files_only,
            compile_model=compile_model,
            min_dimension=min_dimension,
            max_dimension=max_dimension,
            max_new_tokens=max_new_tokens,
            task=task,
        )

    return original, _wrapped


def _attribute_hits(gt: Sequence[Dict[str, Any]], baseline_candidates: Sequence[Dict[str, Any]], falcon_candidates: Sequence[Dict[str, Any]], kept: Sequence[Dict[str, Any]], *, threshold: float) -> Dict[str, Any]:
    raw_source_cover = Counter()
    final_winner_primary = Counter()
    final_source_cover = Counter()
    falcon_only_recovered = 0
    baseline_hit = 0
    falcon_hit = 0
    merged_hit = 0
    per_gt: List[Dict[str, Any]] = []
    for gt_item in gt:
        gt_cls = gt_item["class_name"]
        gt_box = gt_item["bbox_xyxy"]
        best_baseline = 0.0
        best_falcon = 0.0
        source_best: Dict[str, float] = {}
        best_final = None
        best_final_iou = 0.0
        for cand in [*baseline_candidates, *falcon_candidates]:
            if cand["class_name"] != gt_cls:
                continue
            iou = bbox_iou_xyxy(gt_box, cand["bbox_xyxy"])
            if cand.get("source_primary") == "falcon":
                best_falcon = max(best_falcon, iou)
            else:
                best_baseline = max(best_baseline, iou)
            for src in cand.get("source_list") or []:
                source_best[src] = max(source_best.get(src, 0.0), iou)
        for src, iou in source_best.items():
            if iou >= threshold:
                raw_source_cover[src] += 1
        for cand in kept:
            if cand["class_name"] != gt_cls:
                continue
            iou = bbox_iou_xyxy(gt_box, cand["bbox_xyxy"])
            if iou > best_final_iou:
                best_final_iou = iou
                best_final = cand
        if best_baseline >= threshold:
            baseline_hit += 1
        if best_falcon >= threshold:
            falcon_hit += 1
        if best_final_iou >= threshold:
            merged_hit += 1
            primary = str(best_final.get("source_primary") or best_final.get("source") or "unknown")
            final_winner_primary[primary] += 1
            for src in best_final.get("source_list") or [primary]:
                final_source_cover[src] += 1
        if best_baseline < threshold <= best_falcon:
            falcon_only_recovered += 1
        per_gt.append(
            {
                "class_name": gt_cls,
                "baseline_best_iou": best_baseline,
                "falcon_best_iou": best_falcon,
                "final_best_iou": best_final_iou,
                "final_winner_source_primary": None if best_final is None else best_final.get("source_primary"),
                "final_winner_source_list": None if best_final is None else best_final.get("source_list"),
                "falcon_only_recovered": bool(best_baseline < threshold <= best_falcon),
            }
        )
    return {
        "threshold": float(threshold),
        "gt_total": len(gt),
        "baseline_hit_count": baseline_hit,
        "falcon_hit_count": falcon_hit,
        "merged_hit_count": merged_hit,
        "falcon_only_recovered_count": falcon_only_recovered,
        "raw_source_cover": dict(raw_source_cover),
        "final_winner_primary": dict(final_winner_primary),
        "final_source_cover": dict(final_source_cover),
        "per_gt": per_gt,
    }


def _run_image(
    *,
    dataset_id: str,
    image_name: str,
    labelmap: Sequence[str],
    package_id: str,
    iou_threshold: float,
    falcon_device: str,
) -> Dict[str, Any]:
    img_path = QWEN_ROOT / "train" / image_name
    pil_img = Image.open(img_path).convert("RGB")
    img_w, img_h = pil_img.size
    class_names = _class_names_for_image(image_name, labelmap)
    class_id_map = {str(name): idx for idx, name in enumerate(labelmap)}
    image_base64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

    baseline_request = QwenPrepassRequest(
        dataset_id=dataset_id,
        edr_package_id=package_id,
        edr_package_apply_ensemble=False,
        image_base64=image_base64,
        image_name=image_name,
        labelmap=list(labelmap),
        labelmap_glossary="",
        enable_yolo=True,
        enable_rfdetr=True,
        enable_sam3_text=True,
        enable_sam3_similarity=True,
        prepass_caption=False,
        ensemble_enabled=False,
        prepass_only=True,
        prepass_keep_all=True,
        prepass_finalize=False,
    )
    baseline_started = time.perf_counter()
    baseline_response = _run_prepass_annotation_qwen(baseline_request)
    baseline_elapsed = time.perf_counter() - baseline_started

    baseline_candidates = []
    for det in baseline_response.detections or []:
        candidate = _candidate_from_det(det, img_w=img_w, img_h=img_h, class_names=class_names, class_id_map=class_id_map)
        if candidate is not None:
            baseline_candidates.append(candidate)

    payload = AutoLabelRequest(
        dataset_id=dataset_id,
        max_images=1,
        split="train",
        unlabeled_only=True,
        image_relpaths=[image_name],
        target_mode="detection",
        falcon_window_mode="quadrants",
        falcon_device=str(falcon_device),
        falcon_overlap_ratio=0.1,
        dedupe_existing_same_class_iou=0.5,
        class_names=list(class_names),
        edr_package_id=package_id,
        enable_yolo=True,
        enable_rfdetr=True,
        use_planner_caption=False,
    )

    falcon_logs: List[Dict[str, Any]] = []
    original, wrapped = _build_falcon_logging_wrapper(falcon_logs)
    falcon_runtime.run_falcon_queries = wrapped
    import localinferenceapi as li
    original_alias = li._run_falcon_queries_impl
    li._run_falcon_queries_impl = wrapped
    try:
        falcon_candidates = []
        falcon_started = time.perf_counter()
        for window in _auto_label_build_quadrant_windows(img_w, img_h, overlap_ratio=float(payload.falcon_overlap_ratio)):
            crop_window = {**window, "classes": list(class_names)}
            before = len(falcon_logs)
            candidates = _auto_label_falcon_candidates_for_window(
                pil_img=pil_img,
                crop_window=crop_window,
                class_names=class_names,
                class_id_map=class_id_map,
                labelmap=labelmap,
                glossary="",
                payload=payload,
                target_mode="detection",
            )
            for cand in candidates:
                cand["source_primary"] = "falcon"
                cand["source_list"] = ["falcon"]
            falcon_candidates.extend(candidates)
            if len(falcon_logs) > before:
                falcon_logs[-1]["window"] = {
                    "id": str(window.get("id") or ""),
                    "xyxy": [float(v) for v in (window.get("xyxy") or ())[:4]],
                    "crop_size": [int(round(float((window.get("xyxy") or (0, 0, 0, 0))[2] - (window.get("xyxy") or (0, 0, 0, 0))[0]))), int(round(float((window.get("xyxy") or (0, 0, 0, 0))[3] - (window.get("xyxy") or (0, 0, 0, 0))[1])))],
                    "image_name": image_name,
                }
        falcon_elapsed = time.perf_counter() - falcon_started
    finally:
        falcon_runtime.run_falcon_queries = original
        li._run_falcon_queries_impl = original_alias

    kept, dropped = _auto_label_dedupe_candidates(
        [*baseline_candidates, *falcon_candidates],
        existing=[],
        target_mode="detection",
        iou_threshold=float(payload.dedupe_existing_same_class_iou),
    )

    gt = _parse_lines(_gt_lines(image_name), width=img_w, height=img_h, labelmap=labelmap)
    final_coverage = _coverage_report(gt=gt, pred=kept, iou_threshold=iou_threshold)
    baseline_coverage = _coverage_report(gt=gt, pred=baseline_candidates, iou_threshold=iou_threshold)
    falcon_coverage = _coverage_report(gt=gt, pred=falcon_candidates, iou_threshold=iou_threshold)
    attribution = _attribute_hits(gt, baseline_candidates, falcon_candidates, kept, threshold=iou_threshold)

    return {
        "image_name": image_name,
        "class_names": list(class_names),
        "image_size": [img_w, img_h],
        "timings_sec": {
            "baseline": baseline_elapsed,
            "falcon": falcon_elapsed,
        },
        "counts": {
            "baseline_candidates": len(baseline_candidates),
            "falcon_candidates": len(falcon_candidates),
            "kept_candidates": len(kept),
            "duplicates_dropped": int(dropped),
        },
        "baseline_coverage": baseline_coverage,
        "falcon_coverage": falcon_coverage,
        "final_coverage": final_coverage,
        "attribution": attribution,
        "falcon_logs": falcon_logs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=20260407)
    parser.add_argument("--package-id", default=CANONICAL_EDR_PACKAGE)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--falcon-device", default="cuda:1")
    args = parser.parse_args()

    labelmap = _load_labelmap()
    images = _sample_images(args.sample_size, args.sample_seed)
    run_id = f"qwen_random{args.sample_size}_seed{args.sample_seed}"
    dataset_root = Path(f"uploads/datasets/{run_id}")
    report_root = REPORT_BASE / run_id
    if report_root.exists():
        shutil.rmtree(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    _stage_dataset(images, dataset_root=dataset_root)
    entry = register_dataset_path(
        path=str(dataset_root.resolve()),
        dataset_id=run_id,
        label=run_id,
        context="falcon_auto_label_debug",
        notes="Random qwen Falcon debug sample with prompt/output logging",
        force_new=True,
        strict=True,
    )
    dataset_id = str(entry["id"])

    results = []
    for image_name in images:
        results.append(
            _run_image(
                dataset_id=dataset_id,
                image_name=image_name,
                labelmap=labelmap,
                package_id=str(args.package_id),
                iou_threshold=float(args.iou_threshold),
                falcon_device=str(args.falcon_device),
            )
        )
        partial_path = report_root / "partial_report.json"
        partial_payload = {
            "dataset_id": dataset_id,
            "package_id": str(args.package_id),
            "sample_seed": int(args.sample_seed),
            "sample_size": int(args.sample_size),
            "images": images,
            "results_completed": len(results),
            "results": results,
        }
        partial_path.write_text(json.dumps(_to_jsonable(partial_payload), indent=2), encoding="utf-8")

    gt_total = sum(int(item["final_coverage"]["gt_total"]) for item in results)
    baseline_hits = sum(int(item["baseline_coverage"]["matched"]) for item in results)
    falcon_hits = sum(int(item["falcon_coverage"]["matched"]) for item in results)
    final_hits = sum(int(item["final_coverage"]["matched"]) for item in results)
    falcon_only_recovered = sum(int(item["attribution"]["falcon_only_recovered_count"]) for item in results)
    aggregate_winners = Counter()
    aggregate_raw = Counter()
    for item in results:
        aggregate_winners.update(item["attribution"]["final_winner_primary"])
        aggregate_raw.update(item["attribution"]["raw_source_cover"])
    summary = {
        "dataset_id": dataset_id,
        "package_id": str(args.package_id),
        "sample_seed": int(args.sample_seed),
        "sample_size": int(args.sample_size),
        "images": images,
        "overall": {
            "gt_total": gt_total,
            "baseline_hits": baseline_hits,
            "falcon_hits": falcon_hits,
            "final_hits": final_hits,
            "baseline_coverage": (baseline_hits / gt_total) if gt_total else 1.0,
            "falcon_coverage": (falcon_hits / gt_total) if gt_total else 1.0,
            "final_coverage": (final_hits / gt_total) if gt_total else 1.0,
            "falcon_only_recovered_count": falcon_only_recovered,
            "final_winner_primary": dict(aggregate_winners),
            "raw_source_cover": dict(aggregate_raw),
        },
        "results": results,
    }
    out_path = report_root / "report.json"
    json_text = json.dumps(_to_jsonable(summary), indent=2)
    out_path.write_text(json_text, encoding="utf-8")
    print(json_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
