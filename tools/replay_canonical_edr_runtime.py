#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import localinferenceapi as api
from models.schemas import QwenPrepassRequest
from tools.eval_ensemble_xgb_dedupe import _evaluate_predictions
from tools.replay_detector_baselines import _load_gt_by_image, _load_labelmap, _resolve_image_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved canonical EDR on a fixed image list.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-images", required=True)
    parser.add_argument("--recipe-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--yolo-id", required=True)
    parser.add_argument("--rfdetr-id", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def _normalize_det(det: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(det, dict):
        return None
    label = str(det.get("label") or "").strip().lower()
    bbox = det.get("bbox_xyxy_px")
    if not label or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        score = float(det.get("score") or 0.0)
    except Exception:
        score = 0.0
    source = str(det.get("score_source") or det.get("source") or "edr").strip().lower() or "edr"
    source_list_raw = det.get("source_list")
    source_list: List[str] = []
    if isinstance(source_list_raw, (list, tuple)):
        source_list = [str(item).strip().lower() for item in source_list_raw if str(item).strip()]
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


def main() -> int:
    args = _parse_args()
    dataset_id = str(args.dataset)
    recipe_meta_path = Path(args.recipe_meta)
    sample_images_path = Path(args.sample_images)
    output_path = Path(args.output)
    cache_path = Path(args.cache)
    log_path = Path(args.log)

    recipe_meta = json.loads(recipe_meta_path.read_text())
    recipe_config = dict(recipe_meta.get("config") or {})
    images = [line.strip() for line in sample_images_path.read_text().splitlines() if line.strip()]
    if int(args.limit or 0) > 0:
        images = images[: int(args.limit)]

    labelmap = _load_labelmap(dataset_id)
    name_to_cat = {name: idx for idx, name in enumerate(labelmap)}
    gt_by_image = _load_gt_by_image(dataset_id, images, labelmap)
    ensemble_job_id = str(recipe_config.get("ensemble_job_id") or "")
    recipe_id = str(recipe_meta.get("id") or "")

    cache: Dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache_payload = json.loads(cache_path.read_text())
            cache = dict(cache_payload.get("images") or {})
        except Exception:
            cache = {}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = message.rstrip() + "\n"
        print(message, flush=True)
        log_handle.write(line)
        log_handle.flush()

    def save_cache() -> None:
        cache_path.write_text(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "recipe_id": recipe_id,
                    "ensemble_job_id": ensemble_job_id,
                    "yolo_id": args.yolo_id,
                    "rfdetr_id": args.rfdetr_id,
                    "images": cache,
                    "updated_at": time.time(),
                },
                indent=2,
            )
        )

    processed = 0
    start = time.time()
    cached_ok = sum(1 for row in cache.values() if isinstance(row, dict) and row.get("status") == "ok")
    log(
        f"[edr-replay] images total={len(images)} cached={cached_ok} "
        f"dataset={dataset_id} recipe={recipe_id} ensemble={ensemble_job_id}"
    )

    for image_name in images:
        row = cache.get(image_name) or {}
        if isinstance(row, dict) and row.get("status") == "ok":
            processed += 1
            continue

        image_start = time.time()
        img_path = _resolve_image_path(dataset_id, image_name)
        if img_path is None or not img_path.exists():
            cache[image_name] = {"status": "missing_image"}
            save_cache()
            continue

        try:
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                token = api._calibration_cache_image(pil_img, None)
            request = QwenPrepassRequest(
                dataset_id=dataset_id,
                recipe_source_dataset_id=str(recipe_config.get("dataset_id") or dataset_id),
                edr_package_id=str(recipe_config.get("edr_package_id") or "") or None,
                image_token=token,
                image_name=image_name,
                labelmap=labelmap,
                labelmap_glossary=recipe_config.get("labelmap_glossary"),
                enable_yolo=bool(recipe_config.get("enable_yolo", True)),
                enable_rfdetr=bool(recipe_config.get("enable_rfdetr", True)),
                yolo_id=args.yolo_id,
                rfdetr_id=args.rfdetr_id,
                classifier_id=recipe_config.get("resolved_classifier_id")
                or recipe_config.get("classifier_id"),
                enable_sam3_text=bool(recipe_config.get("enable_sam3_text", True)),
                sam3_text_synonym_budget=int(recipe_config.get("sam3_text_synonym_budget", 0) or 0),
                sam3_text_window_extension=bool(
                    recipe_config.get("sam3_text_window_extension", True)
                ),
                sam3_text_window_mode=str(recipe_config.get("sam3_text_window_mode") or "grid"),
                enable_sam3_similarity=bool(recipe_config.get("enable_sam3_similarity", True)),
                similarity_min_exemplar_score=float(
                    recipe_config.get("similarity_min_exemplar_score", 0.6) or 0.6
                ),
                similarity_exemplar_count=int(recipe_config.get("similarity_exemplar_count", 3) or 3),
                similarity_exemplar_strategy=str(
                    recipe_config.get("similarity_exemplar_strategy") or "top"
                ),
                similarity_exemplar_fraction=float(
                    recipe_config.get("similarity_exemplar_fraction", 0.2) or 0.2
                ),
                similarity_exemplar_min=int(recipe_config.get("similarity_exemplar_min", 3) or 3),
                similarity_exemplar_max=int(recipe_config.get("similarity_exemplar_max", 12) or 12),
                similarity_exemplar_source_quota=int(
                    recipe_config.get("similarity_exemplar_source_quota", 1) or 1
                ),
                similarity_window_extension=bool(
                    recipe_config.get("similarity_window_extension", True)
                ),
                similarity_window_mode=str(recipe_config.get("similarity_window_mode") or "grid"),
                prepass_only=True,
                prepass_finalize=True,
                prepass_keep_all=False,
                prepass_caption=False,
                use_detection_overlay=False,
                tighten_fp=True,
                detector_conf=float(recipe_config.get("detector_conf", 0.45) or 0.45),
                sam3_score_thr=float(recipe_config.get("sam3_score_thr", 0.2) or 0.2),
                sam3_mask_threshold=float(recipe_config.get("sam3_mask_threshold", 0.2) or 0.2),
                prepass_sam3_text_thr=float(recipe_config.get("prepass_sam3_text_thr", 0.2) or 0.2),
                prepass_similarity_score=float(
                    recipe_config.get("prepass_similarity_score", 0.3) or 0.3
                ),
                classifier_min_prob=float(recipe_config.get("classifier_min_prob", 0.35) or 0.35),
                classifier_margin=float(recipe_config.get("classifier_margin", 0.05) or 0.05),
                classifier_bg_margin=float(
                    recipe_config.get("classifier_bg_margin", 0.05) or 0.05
                ),
                scoreless_iou=float(recipe_config.get("scoreless_iou", 0.0) or 0.0),
                ensemble_enabled=True,
                ensemble_job_id=ensemble_job_id,
                iou=float(recipe_config.get("iou", 0.75) or 0.75),
                fusion_mode=str(recipe_config.get("fusion_mode") or "primary"),
                cross_class_dedupe_enabled=bool(
                    recipe_config.get("cross_class_dedupe_enabled", False)
                ),
                cross_class_dedupe_iou=float(
                    recipe_config.get("cross_class_dedupe_iou", 0.8) or 0.8
                ),
            )
            response = api._run_prepass_annotation_qwen(request)
            detections: List[Dict[str, Any]] = []
            for det in list(getattr(response, "detections", []) or []):
                normalized = _normalize_det(det)
                if normalized is not None:
                    detections.append(normalized)
            cache[image_name] = {
                "status": "ok",
                "detections": detections,
                "warnings": list(getattr(response, "warnings", []) or []),
                "count": len(detections),
                "elapsed_sec": round(time.time() - image_start, 3),
            }
        except Exception as exc:  # noqa: BLE001
            cache[image_name] = {
                "status": "error",
                "error": str(exc),
                "elapsed_sec": round(time.time() - image_start, 3),
            }
        processed += 1
        save_cache()
        if processed % max(1, int(args.progress_every)) == 0:
            elapsed = max(time.time() - start, 1e-6)
            row = cache.get(image_name) or {}
            log(
                f"[edr-replay] {processed}/{len(images)} image={image_name} "
                f"status={row.get('status')} count={row.get('count', 0)} "
                f"image_sec={row.get('elapsed_sec', 0)} rate={processed/elapsed:.3f} img/s"
            )

    save_cache()
    pred_by_image = {
        image_name: list((row or {}).get("detections") or [])
        for image_name, row in cache.items()
        if isinstance(row, dict) and row.get("status") == "ok"
    }
    metrics = _evaluate_predictions(
        pred_by_image,
        gt_by_image,
        name_to_cat=name_to_cat,
        dedupe_iou=float(recipe_config.get("dedupe_iou", 0.75) or 0.75),
        eval_iou=float(recipe_config.get("eval_iou", 0.5) or 0.5),
        scoreless_iou=float(recipe_config.get("scoreless_iou", 0.0) or 0.0),
        fusion_mode=str(recipe_config.get("fusion_mode") or "primary"),
        apply_dedupe=False,
    )
    status_counts: Dict[str, int] = {}
    for row in cache.values():
        status = str(row.get("status") if isinstance(row, dict) else "invalid")
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "dataset_id": dataset_id,
        "sample_size": len(images),
        "recipe_id": recipe_id,
        "recipe_name": recipe_meta.get("name"),
        "ensemble_job_id": ensemble_job_id,
        "recipe_fingerprint": recipe_config.get("recipe_fingerprint"),
        "yolo_id": args.yolo_id,
        "rfdetr_id": args.rfdetr_id,
        "eval_iou": float(recipe_config.get("eval_iou", 0.5) or 0.5),
        "surface": "fresh canonical EDR runtime replay",
        "metrics": metrics,
        "image_status_counts": status_counts,
        "cache_json": str(cache_path),
        "completed_at": time.time(),
    }
    output_path.write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
