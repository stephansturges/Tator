"""Calibration job orchestration and serialization."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
import json
import multiprocessing
import queue
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

DEFAULT_EMBED_PROJ_DIM = 1024


def _serialize_calibration_job(job: Any) -> Dict[str, Any]:
    serialized_result = job.result
    if isinstance(serialized_result, dict):
        eval_path = serialized_result.get("eval")
        if eval_path:
            try:
                eval_metrics = json.loads(Path(eval_path).read_text())
                if isinstance(eval_metrics, dict):
                    serialized_result = dict(serialized_result)
                    serialized_result["metrics"] = eval_metrics
            except Exception:
                pass
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "phase": job.phase,
        "progress": job.progress,
        "processed": job.processed,
        "total": job.total,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "request": job.request,
        "result": serialized_result,
        "error": job.error,
    }


@dataclass
class CalibrationJob:
    job_id: str
    status: str = "queued"
    message: str = "Queued"
    phase: str = "queued"
    progress: float = 0.0
    processed: int = 0
    total: int = 0
    request: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: Any = field(default_factory=__import__("threading").Event)


def _normalize_similarity_strategy(value: Any) -> str:
    strategy = str(value or "top").strip().lower()
    if strategy not in {"top", "random", "diverse"}:
        strategy = "top"
    return strategy


def _coerce_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _float_with_default(value: Any, default: float) -> float:
    parsed = _coerce_float(value)
    if parsed is None:
        return float(default)
    return parsed


def _first_float_with_default(*values: Any, default: float) -> float:
    for value in values:
        if value is None:
            continue
        parsed = _coerce_float(value)
        if parsed is not None:
            return parsed
    return float(default)


def _validate_classifier_feature_matrix(path: Path) -> None:
    with np.load(path, allow_pickle=True) as data:
        feature_names = [str(name) for name in data.get("feature_names", [])]
        classifier_classes = [str(name) for name in data.get("classifier_classes", [])]
        try:
            embed_proj_dim = int(data.get("embed_proj_dim", 0))
        except Exception:
            embed_proj_dim = 0
        X = np.asarray(data.get("X"), dtype=np.float32)

    if not classifier_classes:
        raise RuntimeError("classifier_classes_empty")
    if embed_proj_dim <= 0:
        raise RuntimeError("embed_proj_dim_zero")

    embed_idx = [
        idx
        for idx, name in enumerate(feature_names)
        if name.startswith("clf_emb_rp::") or name.startswith("embed_proj_")
    ]
    if not embed_idx:
        raise RuntimeError("embed_features_missing")
    clf_prob_idx = [idx for idx, name in enumerate(feature_names) if name.startswith("clf_prob::")]
    if not clf_prob_idx:
        raise RuntimeError("classifier_prob_features_missing")
    if X.ndim != 2:
        raise RuntimeError("feature_matrix_invalid_shape")
    if X.shape[1] != len(feature_names):
        raise RuntimeError("feature_dim_mismatch")
    if X.shape[0] > 0:
        if np.allclose(X[:, embed_idx], 0.0):
            raise RuntimeError("embed_features_all_zero")
        if np.allclose(X[:, clf_prob_idx], 0.0):
            raise RuntimeError("classifier_prob_features_all_zero")


def _canonical_similarity_settings(payload: Any) -> Dict[str, Any]:
    strategy = _normalize_similarity_strategy(getattr(payload, "similarity_exemplar_strategy", None))
    raw_count = getattr(payload, "similarity_exemplar_count", None)
    count = int(raw_count or 3)
    if count < 1:
        count = 1
    raw_seed = getattr(payload, "similarity_exemplar_seed", None)
    if strategy in {"random", "diverse"}:
        seed = int(raw_seed or 0)
    else:
        seed = None
    raw_fraction = getattr(payload, "similarity_exemplar_fraction", None)
    try:
        fraction = float(raw_fraction if raw_fraction is not None else 0.2)
    except (TypeError, ValueError):
        fraction = 0.2
    if not math.isfinite(fraction) or fraction <= 0:
        fraction = 0.2
    raw_min = getattr(payload, "similarity_exemplar_min", None)
    min_count = int(raw_min or 3)
    if min_count < 1:
        min_count = 1
    raw_max = getattr(payload, "similarity_exemplar_max", None)
    max_count = int(raw_max or 12)
    if max_count < min_count:
        max_count = min_count
    raw_quota = getattr(payload, "similarity_exemplar_source_quota", None)
    try:
        source_quota = int(raw_quota) if raw_quota is not None else 1
    except (TypeError, ValueError):
        source_quota = 1
    if source_quota < 0:
        source_quota = 0
    return {
        "strategy": strategy,
        "count": count if strategy in {"top", "random"} else None,
        "seed": seed,
        "fraction": fraction if strategy == "diverse" else None,
        "min": min_count if strategy == "diverse" else None,
        "max": max_count if strategy == "diverse" else None,
        "source_quota": source_quota if strategy == "diverse" else None,
    }


def _canonical_cross_class_dedupe_settings(payload: Any) -> Dict[str, Any]:
    enabled = bool(getattr(payload, "cross_class_dedupe_enabled", False))
    if not enabled:
        return {"enabled": False, "iou": None}
    raw_iou = getattr(payload, "cross_class_dedupe_iou", None)
    try:
        iou = float(raw_iou if raw_iou is not None else 0.8)
    except (TypeError, ValueError):
        iou = 0.8
    iou = max(0.0, min(1.0, iou))
    if iou <= 0.0:
        return {"enabled": False, "iou": None}
    return {"enabled": True, "iou": iou}


def _normalize_window_mode(value: Any) -> str:
    mode = str(value or "grid").strip().lower()
    if mode not in {"grid", "sahi"}:
        mode = "grid"
    return mode


def _canonical_sam3_text_window_settings(payload: Any) -> Dict[str, Any]:
    enabled = getattr(payload, "sam3_text_window_extension", None) is not False
    if not enabled:
        return {"enabled": False, "mode": None, "size": None, "overlap": None}
    mode = _normalize_window_mode(getattr(payload, "sam3_text_window_mode", None))
    if mode != "sahi":
        return {"enabled": True, "mode": mode, "size": None, "overlap": None}
    try:
        size = int(getattr(payload, "sam3_text_window_size", None) or 640)
    except (TypeError, ValueError):
        size = 640
    if size <= 0:
        size = 640
    overlap = _float_with_default(getattr(payload, "sam3_text_window_overlap", None), 0.2)
    if overlap <= 0.0 or overlap >= 1.0:
        overlap = 0.2
    return {
        "enabled": True,
        "mode": mode,
        "size": size,
        "overlap": overlap,
    }


def _canonical_similarity_window_settings(payload: Any) -> Dict[str, Any]:
    enabled = bool(getattr(payload, "similarity_window_extension", False))
    if not enabled:
        return {"enabled": False, "mode": None, "size": None, "overlap": None}
    mode = _normalize_window_mode(getattr(payload, "similarity_window_mode", None))
    if mode != "sahi":
        return {"enabled": True, "mode": mode, "size": None, "overlap": None}
    try:
        size = int(getattr(payload, "similarity_window_size", None) or 640)
    except (TypeError, ValueError):
        size = 640
    if size <= 0:
        size = 640
    overlap = _float_with_default(getattr(payload, "similarity_window_overlap", None), 0.2)
    if overlap <= 0.0 or overlap >= 1.0:
        overlap = 0.2
    return {
        "enabled": True,
        "mode": mode,
        "size": size,
        "overlap": overlap,
    }


def _canonical_sahi_settings(payload: Any) -> Dict[str, float | int]:
    raw_size = getattr(payload, "sahi_window_size", None)
    try:
        size = int(raw_size) if raw_size is not None else 640
    except (TypeError, ValueError):
        size = 640
    if size <= 0:
        size = 640

    raw_overlap = getattr(payload, "sahi_overlap_ratio", None)
    overlap = _float_with_default(raw_overlap, 0.2)
    if overlap <= 0.0 or overlap >= 1.0:
        overlap = 0.2
    return {"size": size, "overlap": overlap}


def _start_calibration_job(
    payload: Any,
    *,
    job_cls: Any,
    jobs: Dict[str, Any],
    jobs_lock: Any,
    run_job_fn: Callable[[Any, Any], None],
) -> Any:
    job_id = f"cal_{uuid.uuid4().hex[:8]}"
    job = job_cls(job_id=job_id)
    with jobs_lock:
        jobs[job.job_id] = job
    thread = __import__("threading").Thread(target=run_job_fn, args=(job, payload), daemon=True)
    thread.start()
    return job


def _cancel_calibration_job(
    job_id: str,
    *,
    jobs: Dict[str, Any],
    jobs_lock: Any,
    http_exception_cls: Any,
    time_fn: Callable[[], float],
) -> Any:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise http_exception_cls(status_code=404, detail="calibration_job_not_found")
    if job.status in {"completed", "failed", "cancelled"}:
        return job
    job.cancel_event.set()
    job.status = "cancelled"
    job.message = "Cancelled"
    job.phase = "cancelled"
    job.updated_at = time_fn()
    return job


def _run_calibration_job(
    job: Any,
    payload: Any,
    *,
    jobs: Dict[str, Any],
    jobs_lock: Any,
    update_fn: Callable[..., None],
    require_sam3_fn: Callable[[bool, bool], None],
    prepare_for_training_fn: Callable[[], None],
    load_yolo_active_fn: Callable[[], Dict[str, Any]],
    load_rfdetr_active_fn: Callable[[], Dict[str, Any]],
    load_labelmap_meta_fn: Callable[[str], Tuple[List[str], str]],
    list_images_fn: Callable[[str], List[str]],
    sample_images_fn: Callable[[List[str]], List[str]],
    calibration_root: Path,
    calibration_cache_root: Path,
    prepass_request_cls: Any,
    active_classifier_head: Any,
    active_classifier_path: Optional[str],
    default_classifier_for_dataset_fn: Optional[Callable[[Optional[str]], Optional[str]]],
    calibration_features_version: int,
    write_record_fn: Callable[[Path, Dict[str, Any]], None],
    hash_payload_fn: Callable[[Dict[str, Any]], str],
    safe_link_fn: Callable[[Path, Path], None],
    prepass_worker_fn: Callable[..., None],
    unload_inference_runtimes_fn: Callable[[], None],
    resolve_dataset_fn: Callable[[str], Path],
    cache_image_fn: Callable[[Image.Image, Optional[str]], str],
    run_prepass_fn: Callable[..., Dict[str, Any]],
    logger: Any,
    http_exception_cls: Any,
    root_dir: Path,
) -> None:
    with jobs_lock:
        jobs[job.job_id] = job
    update_fn(job, status="running", message="Selecting images…", phase="select_images", request=payload.dict())
    output_dir = calibration_root / job.job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        require_sam3_fn(True, True)
        prepare_for_training_fn()
        yolo_active: Dict[str, Any] = {}
        rfdetr_active: Dict[str, Any] = {}
        if payload.enable_yolo is not False:
            yolo_active = load_yolo_active_fn()
            yolo_best = str(yolo_active.get("best_path") or "")
            if not yolo_best or not Path(yolo_best).exists():
                raise http_exception_cls(status_code=412, detail="yolo_active_missing_weights")
        if payload.enable_rfdetr is not False:
            rfdetr_active = load_rfdetr_active_fn()
            rfdetr_best = str(rfdetr_active.get("best_path") or "")
            if not rfdetr_best or not Path(rfdetr_best).exists():
                raise http_exception_cls(status_code=412, detail="rfdetr_active_missing_weights")
        labelmap, glossary = load_labelmap_meta_fn(payload.dataset_id)
        if not labelmap:
            raise http_exception_cls(status_code=400, detail="calibration_labelmap_missing")
        images = list_images_fn(payload.dataset_id)
        if not images:
            raise http_exception_cls(status_code=404, detail="calibration_images_missing")
        max_images = int(payload.max_images or 0)
        seed = int(payload.seed or 0)
        selected = sample_images_fn(images, max_images=max_images, seed=seed)
        total = len(selected)
        update_fn(job, total=total, processed=0, progress=0.0)

        prepass_path = output_dir / "prepass.jsonl"
        calibration_cache_root.mkdir(parents=True, exist_ok=True)
        classifier_id_resolved = str(payload.classifier_id or "").strip()
        if not classifier_id_resolved and callable(default_classifier_for_dataset_fn):
            try:
                classifier_id_resolved = str(default_classifier_for_dataset_fn(payload.dataset_id) or "").strip()
            except Exception:
                classifier_id_resolved = ""
        if not classifier_id_resolved:
            classifier_id_resolved = str(active_classifier_path or "").strip()
        if not classifier_id_resolved and isinstance(active_classifier_head, dict):
            raise http_exception_cls(status_code=412, detail="calibration_classifier_id_required")
        if not classifier_id_resolved:
            raise http_exception_cls(status_code=412, detail="calibration_classifier_required")
        if not (payload.classifier_id or "").strip():
            try:
                req = dict(job.request or {})
                req["classifier_id_resolved"] = classifier_id_resolved
                job.request = req
            except Exception:
                pass
        use_yolo = payload.enable_yolo is not False
        use_rfdetr = payload.enable_rfdetr is not False
        similarity_cfg = _canonical_similarity_settings(payload)
        cross_class_cfg = _canonical_cross_class_dedupe_settings(payload)
        sam3_text_window_cfg = _canonical_sam3_text_window_settings(payload)
        similarity_window_cfg = _canonical_similarity_window_settings(payload)
        sahi_cfg = _canonical_sahi_settings(payload)
        prepass_sam3_text_thr = _float_with_default(payload.prepass_sam3_text_thr, 0.2)
        prepass_similarity_score = _float_with_default(payload.prepass_similarity_score, 0.3)
        similarity_min_exemplar_score = _float_with_default(payload.similarity_min_exemplar_score, 0.6)
        sam3_score_thr = _float_with_default(payload.sam3_score_thr, 0.2)
        sam3_mask_threshold = _float_with_default(payload.sam3_mask_threshold, 0.2)
        detector_conf = _float_with_default(payload.detector_conf, 0.45)
        scoreless_iou = _float_with_default(payload.scoreless_iou, 0.0)
        dedupe_iou = _float_with_default(payload.dedupe_iou, 0.75)
        yolo_run_id = str(yolo_active.get("run_id") or "").strip() or None if use_yolo else None
        rfdetr_run_id = str(rfdetr_active.get("run_id") or "").strip() or None if use_rfdetr else None
        prepass_payload = prepass_request_cls(
            dataset_id=payload.dataset_id,
            enable_yolo=use_yolo,
            enable_rfdetr=use_rfdetr,
            yolo_id=yolo_run_id,
            rfdetr_id=rfdetr_run_id,
            sam_variant="sam3",
            enable_sam3_text=True,
            enable_sam3_similarity=True,
            prepass_caption=False,
            prepass_only=True,
            prepass_finalize=False,
            prepass_keep_all=True,
            sam3_text_synonym_budget=None
            if payload.sam3_text_synonym_budget is None
            else int(payload.sam3_text_synonym_budget),
            sam3_text_window_extension=sam3_text_window_cfg["enabled"],
            sam3_text_window_mode=sam3_text_window_cfg["mode"],
            sam3_text_window_size=sam3_text_window_cfg["size"],
            sam3_text_window_overlap=sam3_text_window_cfg["overlap"],
            prepass_sam3_text_thr=prepass_sam3_text_thr,
            prepass_similarity_score=prepass_similarity_score,
            similarity_min_exemplar_score=similarity_min_exemplar_score,
            similarity_exemplar_count=similarity_cfg["count"],
            similarity_exemplar_strategy=similarity_cfg["strategy"],
            similarity_exemplar_seed=similarity_cfg["seed"],
            similarity_exemplar_fraction=similarity_cfg["fraction"],
            similarity_exemplar_min=similarity_cfg["min"],
            similarity_exemplar_max=similarity_cfg["max"],
            similarity_exemplar_source_quota=similarity_cfg["source_quota"],
            similarity_window_extension=similarity_window_cfg["enabled"],
            similarity_window_mode=similarity_window_cfg["mode"],
            similarity_window_size=similarity_window_cfg["size"],
            similarity_window_overlap=similarity_window_cfg["overlap"],
            sam3_score_thr=sam3_score_thr,
            sam3_mask_threshold=sam3_mask_threshold,
            detector_conf=detector_conf,
            sahi_window_size=sahi_cfg["size"],
            sahi_overlap_ratio=sahi_cfg["overlap"],
            classifier_id=classifier_id_resolved,
            scoreless_iou=scoreless_iou,
            iou=dedupe_iou,
            fusion_mode=(payload.fusion_mode or "primary"),
            cross_class_dedupe_enabled=cross_class_cfg["enabled"],
            cross_class_dedupe_iou=cross_class_cfg["iou"],
        )

        labelmap_hash = hashlib.sha1(",".join(labelmap).encode("utf-8")).hexdigest()
        prepass_glossary_text = glossary or ""
        glossary_hash = hashlib.sha1(prepass_glossary_text.encode("utf-8")).hexdigest()
        selected_hash = hashlib.sha1(json.dumps(selected, sort_keys=True).encode("utf-8")).hexdigest()

        def _detector_fingerprint(active: Dict[str, Any]) -> Dict[str, Any]:
            run_id = str(active.get("run_id") or "").strip() or None
            best_path = str(active.get("best_path") or "").strip() or None
            labelmap_path = str(active.get("labelmap_path") or "").strip() or None
            stat_size = None
            stat_mtime_ns = None
            if best_path:
                try:
                    stat = Path(best_path).stat()
                    stat_size = int(stat.st_size)
                    stat_mtime_ns = int(stat.st_mtime_ns)
                except Exception:
                    stat_size = None
                    stat_mtime_ns = None
            return {
                "run_id": run_id,
                "best_path": best_path,
                "best_size": stat_size,
                "best_mtime_ns": stat_mtime_ns,
                "labelmap_path": labelmap_path,
            }

        yolo_fingerprint = _detector_fingerprint(yolo_active) if payload.enable_yolo is not False else None
        rfdetr_fingerprint = _detector_fingerprint(rfdetr_active) if payload.enable_rfdetr is not False else None
        yolo_run_id = (yolo_fingerprint or {}).get("run_id")
        rfdetr_run_id = (rfdetr_fingerprint or {}).get("run_id")

        prepass_config = {
            "sam3_text_synonym_budget": None
            if payload.sam3_text_synonym_budget is None
            else int(payload.sam3_text_synonym_budget),
            "sam3_text_window_extension": sam3_text_window_cfg["enabled"],
            "sam3_text_window_mode": sam3_text_window_cfg["mode"],
            "sam3_text_window_size": sam3_text_window_cfg["size"],
            "sam3_text_window_overlap": sam3_text_window_cfg["overlap"],
            "prepass_sam3_text_thr": prepass_sam3_text_thr,
            "prepass_similarity_score": prepass_similarity_score,
            "similarity_min_exemplar_score": similarity_min_exemplar_score,
            "similarity_exemplar_count": similarity_cfg["count"],
            "similarity_exemplar_strategy": similarity_cfg["strategy"],
            "similarity_exemplar_seed": similarity_cfg["seed"],
            "similarity_exemplar_fraction": similarity_cfg["fraction"],
            "similarity_exemplar_min": similarity_cfg["min"],
            "similarity_exemplar_max": similarity_cfg["max"],
            "similarity_exemplar_source_quota": similarity_cfg["source_quota"],
            "similarity_window_extension": similarity_window_cfg["enabled"],
            "similarity_window_mode": similarity_window_cfg["mode"],
            "similarity_window_size": similarity_window_cfg["size"],
            "similarity_window_overlap": similarity_window_cfg["overlap"],
            "sam3_score_thr": sam3_score_thr,
            "sam3_mask_threshold": sam3_mask_threshold,
            "detector_conf": detector_conf,
            "enable_yolo": use_yolo,
            "enable_rfdetr": use_rfdetr,
            "sahi_window_size": sahi_cfg["size"],
            "sahi_overlap_ratio": sahi_cfg["overlap"],
            "scoreless_iou": scoreless_iou,
            "dedupe_iou": dedupe_iou,
            "fusion_mode": str(payload.fusion_mode or "primary"),
            "cross_class_dedupe_enabled": cross_class_cfg["enabled"],
            "cross_class_dedupe_iou": cross_class_cfg["iou"],
            "yolo_run_id": yolo_run_id,
            "rfdetr_run_id": rfdetr_run_id,
            "yolo_fingerprint": yolo_fingerprint,
            "rfdetr_fingerprint": rfdetr_fingerprint,
        }
        prepass_config_key = hash_payload_fn(
            {
                "dataset_id": payload.dataset_id,
                "labelmap_hash": labelmap_hash,
                "glossary_hash": glossary_hash,
                "glossary_text": prepass_glossary_text,
                "prepass": prepass_config,
            }
        )
        prepass_key = hash_payload_fn(
            {
                "prepass_config_key": prepass_config_key,
                "selected_hash": selected_hash,
            }
        )
        prepass_cache_dir = calibration_cache_root / "prepass" / prepass_config_key
        image_cache_dir = prepass_cache_dir / "images"
        image_cache_dir.mkdir(parents=True, exist_ok=True)

        prepass_cache_meta = prepass_cache_dir / "prepass.meta.json"
        glossary_path = prepass_cache_dir / "glossary.json"

        def _normalize_glossary_payload(text: str) -> Any:
            if not text:
                return {}
            try:
                return json.loads(text)
            except Exception:
                return text

        if not prepass_cache_meta.exists():
            write_record_fn(
                prepass_cache_meta,
                {
                    "dataset_id": payload.dataset_id,
                    "labelmap": labelmap,
                    "labelmap_hash": labelmap_hash,
                    "glossary_text": prepass_glossary_text,
                    "glossary_hash": glossary_hash,
                    "prepass_config": prepass_config,
                    "prepass_config_key": prepass_config_key,
                    "created_at": time.time(),
                },
            )
            write_record_fn(
                glossary_path,
                {
                    "glossary": _normalize_glossary_payload(prepass_glossary_text),
                    "glossary_hash": glossary_hash,
                },
            )
        else:
            try:
                meta = json.loads(prepass_cache_meta.read_text())
            except Exception:
                meta = {}
            updated = False
            if meta.get("labelmap_hash") != labelmap_hash:
                meta["labelmap_hash"] = labelmap_hash
                meta["labelmap"] = labelmap
                updated = True
            if meta.get("glossary_hash") != glossary_hash:
                meta["glossary_hash"] = glossary_hash
                meta["glossary_text"] = prepass_glossary_text
                updated = True
            if "glossary_text" not in meta:
                meta["glossary_text"] = prepass_glossary_text
                updated = True
            if updated:
                meta["updated_at"] = time.time()
                write_record_fn(prepass_cache_meta, meta)
            if not glossary_path.exists():
                write_record_fn(
                    glossary_path,
                    {
                        "glossary": _normalize_glossary_payload(prepass_glossary_text),
                        "glossary_hash": glossary_hash,
                    },
                )

        def _safe_image_cache_name(image_name: str) -> str:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_name)
            if safe != image_name:
                suffix = hashlib.sha1(image_name.encode("utf-8")).hexdigest()[:8]
                safe = f"{safe}_{suffix}"
            return safe

        def _cache_path_for_image(image_name: str) -> Path:
            return image_cache_dir / f"{_safe_image_cache_name(image_name)}.json"

        def _load_cached_record(image_name: str) -> Optional[Dict[str, Any]]:
            path = _cache_path_for_image(image_name)
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text())
            except Exception:
                return None

        def _write_cached_record(image_name: str, record: Dict[str, Any]) -> None:
            path = _cache_path_for_image(image_name)
            write_record_fn(path, record)

        cached_records: Dict[str, Dict[str, Any]] = {}
        for image_name in selected:
            cached = _load_cached_record(image_name)
            if cached:
                cached_records[image_name] = cached

        processed = len(cached_records)
        if processed:
            update_fn(
                job,
                message="Using cached prepass (partial)…",
                phase="prepass",
                processed=processed,
                progress=processed / total if total else 1.0,
            )

        if processed < total:
            update_fn(job, message="Running deep prepass…", phase="prepass")
            remaining = [image_name for image_name in selected if image_name not in cached_records]
            if torch.cuda.is_available() and torch.cuda.device_count() > 1 and remaining:
                unload_inference_runtimes_fn()
                devices = list(range(torch.cuda.device_count()))
                worker_count = min(len(devices), len(remaining))
                tasks = [
                    (image_name, str(_cache_path_for_image(image_name)))
                    for image_name in remaining
                ]
                buckets: List[List[Tuple[str, str]]] = [[] for _ in range(worker_count)]
                for idx, task in enumerate(tasks):
                    buckets[idx % worker_count].append(task)
                ctx = multiprocessing.get_context("spawn")
                mp_cancel = ctx.Event()
                progress_queue = ctx.Queue()
                workers = []
                prepass_payload_dict = prepass_payload.dict()
                for worker_idx in range(worker_count):
                    device_index = devices[worker_idx]
                    bucket = buckets[worker_idx]
                    if not bucket:
                        continue
                    proc = ctx.Process(
                        target=prepass_worker_fn,
                        args=(
                            device_index,
                            bucket,
                            payload.dataset_id,
                            labelmap,
                            glossary,
                            prepass_payload_dict,
                            mp_cancel,
                            progress_queue,
                        ),
                        daemon=True,
                    )
                    proc.start()
                    workers.append(proc)
                processed_local = 0
                while any(proc.is_alive() for proc in workers):
                    if job.cancel_event.is_set():
                        mp_cancel.set()
                    try:
                        inc = progress_queue.get(timeout=1.0)
                        if isinstance(inc, int):
                            processed_local += inc
                            processed = len(cached_records) + processed_local
                            progress = processed / total if total else 1.0
                            update_fn(job, processed=processed, progress=progress)
                    except queue.Empty:
                        pass
                    if mp_cancel.is_set():
                        break
                for proc in workers:
                    proc.join(timeout=5)
                if mp_cancel.is_set():
                    for proc in workers:
                        if proc.is_alive():
                            proc.terminate()
                    raise RuntimeError("cancelled")
            else:
                for image_name in remaining:
                    if job.cancel_event.is_set():
                        raise RuntimeError("cancelled")
                    img_path = None
                    dataset_root = resolve_dataset_fn(payload.dataset_id)
                    for split in ("val", "train"):
                        candidate = dataset_root / split / image_name
                        if candidate.exists():
                            img_path = candidate
                            break
                    if img_path is None:
                        continue
                    with Image.open(img_path) as img:
                        pil_img = img.convert("RGB")
                    image_token = cache_image_fn(pil_img, prepass_payload.sam_variant)
                    result = run_prepass_fn(
                        prepass_payload,
                        pil_img=pil_img,
                        image_token=image_token,
                        labelmap=labelmap,
                        glossary=glossary,
                        trace_writer=None,
                        trace_full_writer=None,
                        trace_readable=None,
                    )
                    detections = list(result.get("detections") or [])
                    warnings = list(result.get("warnings") or [])
                    provenance = result.get("provenance")
                    record = {
                        "image": image_name,
                        "dataset_id": payload.dataset_id,
                        "detections": detections,
                        "warnings": warnings,
                    }
                    if isinstance(provenance, dict):
                        record["provenance"] = provenance
                    _write_cached_record(image_name, record)
                    cached_records[image_name] = record
                    processed += 1
                    progress = processed / total if total else 1.0
                    update_fn(job, processed=processed, progress=progress)

        missing_records: List[str] = []
        with prepass_path.open("w", encoding="utf-8") as handle:
            for image_name in selected:
                record = cached_records.get(image_name) or _load_cached_record(image_name)
                if not record:
                    missing_records.append(image_name)
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if missing_records:
            sample = ",".join(missing_records[:5])
            raise RuntimeError(f"prepass_cache_incomplete:{len(missing_records)}:{sample}")

        prepass_cache_meta.write_text(
            json.dumps(
                {
                    "dataset_id": payload.dataset_id,
                    "labelmap_hash": labelmap_hash,
                    "glossary_hash": glossary_hash,
                    "glossary_text": prepass_glossary_text,
                    "prepass_config": prepass_config,
                    "cached_images": len(list(image_cache_dir.glob("*.json"))),
                    "updated_at": time.time(),
                    "config_key": prepass_config_key,
                },
                indent=2,
            )
        )

        if job.cancel_event.is_set():
            raise RuntimeError("cancelled")

        label_iou = _float_with_default(payload.label_iou, 0.5)
        support_iou = _float_with_default(payload.support_iou, 0.5)
        context_radius = _float_with_default(payload.context_radius, 0.075)
        embed_proj_dim = DEFAULT_EMBED_PROJ_DIM
        embed_proj_seed = 42
        embed_l2_normalize = True
        base_fp_ratio = _float_with_default(payload.base_fp_ratio, 0.1)
        relax_fp_ratio = _float_with_default(payload.relax_fp_ratio, 0.2)
        target_fp_ratio = _first_float_with_default(
            payload.relax_fp_ratio,
            payload.base_fp_ratio,
            default=0.2,
        )
        recall_floor = _float_with_default(payload.recall_floor, 0.6)
        eval_iou = _float_with_default(payload.eval_iou, 0.5)

        features_path = output_dir / "ensemble_features.npz"
        labeled_path = output_dir / f"ensemble_features_iou{label_iou:.2f}.npz"
        calibration_model = (payload.calibration_model or "xgb").strip().lower()
        if calibration_model not in {"mlp", "xgb"}:
            calibration_model = "xgb"
        model_prefix = output_dir / ("ensemble_xgb" if calibration_model == "xgb" else "ensemble_mlp")
        meta_path = Path(str(model_prefix) + ".meta.json")
        eval_path = output_dir / (f"{model_prefix.name}.eval.json")

        def _run_step(phase: str, message: str, args: List[str]) -> None:
            update_fn(job, phase=phase, message=message)
            if job.cancel_event.is_set():
                raise RuntimeError("cancelled")
            subprocess.run(args, check=True)

        classifier_id = classifier_id_resolved
        features_key = hash_payload_fn(
            {
                "prepass_key": prepass_key,
                "classifier_id": classifier_id or "",
                "support_iou": support_iou,
                "min_crop_size": 4,
                "context_radius": context_radius,
                "embed_proj_dim": embed_proj_dim,
                "embed_proj_seed": embed_proj_seed,
                "embed_l2_normalize": embed_l2_normalize,
                "features_version": calibration_features_version,
            }
        )
        features_cache_dir = calibration_cache_root / "features" / features_key
        features_cache_dir.mkdir(parents=True, exist_ok=True)
        features_cache_path = features_cache_dir / "ensemble_features.npz"
        features_cache_meta = features_cache_dir / "features.meta.json"
        cached_features = features_cache_path.exists()
        if cached_features:
            try:
                _validate_classifier_feature_matrix(features_cache_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Invalid cached classifier features for key=%s (%s); rebuilding.",
                    features_key,
                    exc,
                )
                try:
                    features_cache_path.unlink(missing_ok=True)
                except Exception:
                    pass
                cached_features = False

        if cached_features:
            update_fn(job, phase="features", message="Using cached features…", progress=1.0)
            safe_link_fn(features_cache_path, features_path)
        else:
            _run_step(
                "features",
                "Building features…",
                [
                    sys.executable,
                    str(root_dir / "tools" / "build_ensemble_features.py"),
                    "--input",
                    str(prepass_path),
                    "--dataset",
                    payload.dataset_id,
                    "--output",
                    str(features_cache_path),
                    "--support-iou",
                    str(support_iou),
                    "--min-crop-size",
                    "4",
                    "--context-radius",
                    str(context_radius),
                    "--embed-proj-dim",
                    str(int(embed_proj_dim)),
                    "--embed-proj-seed",
                    str(int(embed_proj_seed)),
                    "--device",
                    "cuda",
                    "--require-classifier",
                ]
                + ([] if embed_l2_normalize else ["--embed-no-l2-normalize"])
                + (["--classifier-id", classifier_id] if classifier_id else []),
            )
            try:
                _validate_classifier_feature_matrix(features_cache_path)
            except Exception as exc:  # noqa: BLE001
                raise http_exception_cls(
                    status_code=412,
                    detail=f"calibration_classifier_features_invalid:{exc}",
                ) from exc
            features_cache_meta.write_text(
                json.dumps(
                    {
                        "prepass_key": prepass_key,
                        "features_key": features_key,
                        "features_version": calibration_features_version,
                        "embed_proj_dim": int(embed_proj_dim),
                        "embed_proj_seed": int(embed_proj_seed),
                        "embed_l2_normalize": bool(embed_l2_normalize),
                    },
                    indent=2,
                )
            )
            safe_link_fn(features_cache_path, features_path)

        labeled_key = hash_payload_fn(
            {
                "features_key": features_key,
                "label_iou": label_iou,
            }
        )
        labeled_cache_dir = calibration_cache_root / "labeled" / labeled_key
        labeled_cache_dir.mkdir(parents=True, exist_ok=True)
        labeled_cache_path = labeled_cache_dir / f"ensemble_features_iou{label_iou:.2f}.npz"
        labeled_cache_meta = labeled_cache_dir / "labeled.meta.json"
        cached_labeled = labeled_cache_path.exists()

        if cached_labeled:
            update_fn(job, phase="labeling", message="Using cached labels…", progress=1.0)
            safe_link_fn(labeled_cache_path, labeled_path)
        else:
            _run_step(
                "labeling",
                "Labeling candidates…",
                [
                    sys.executable,
                    str(root_dir / "tools" / "label_candidates_iou90.py"),
                    "--input",
                    str(features_path),
                    "--dataset",
                    payload.dataset_id,
                    "--output",
                    str(labeled_cache_path),
                    "--iou",
                    str(label_iou),
                ],
            )
            labeled_cache_meta.write_text(
                json.dumps(
                    {
                        "features_key": features_key,
                        "labeled_key": labeled_key,
                        "label_iou": label_iou,
                    },
                    indent=2,
                )
            )
            safe_link_fn(labeled_cache_path, labeled_path)
        if calibration_model == "xgb":
            optimize_metric = (payload.optimize_metric or "f1").strip().lower()
            if optimize_metric not in {"f1", "recall", "tp"}:
                optimize_metric = "f1"
            steps_val = int(payload.threshold_steps or 200)
            steps_val = max(20, min(400, steps_val))
            split_head_by_support = bool(getattr(payload, "split_head_by_support", True))
            train_sam3_text_quality = bool(getattr(payload, "train_sam3_text_quality", True))
            sam3_text_quality_alpha = _float_with_default(
                getattr(payload, "sam3_text_quality_alpha", None), 0.35
            )
            target_fp_ratio_by_label_json = str(
                getattr(payload, "target_fp_ratio_by_label_json", "") or ""
            ).strip()
            min_recall_by_label_json = str(
                getattr(payload, "min_recall_by_label_json", "") or ""
            ).strip()
            train_cmd = [
                sys.executable,
                str(root_dir / "tools" / "train_ensemble_xgb.py"),
                "--input",
                str(labeled_path),
                "--output",
                str(model_prefix),
                "--seed",
                str(int(payload.model_seed or 42)),
                "--target-fp-ratio",
                str(base_fp_ratio),
                "--min-recall",
                str(recall_floor),
                "--threshold-steps",
                str(steps_val),
                "--optimize",
                optimize_metric,
            ]
            if payload.per_class_thresholds is not False:
                train_cmd.append("--per-class")
            if split_head_by_support:
                train_cmd.append("--split-head-by-support")
            if train_sam3_text_quality:
                train_cmd += [
                    "--train-sam3-text-quality",
                    "--sam3-text-quality-alpha",
                    str(sam3_text_quality_alpha),
                ]
            if payload.xgb_max_depth is not None:
                train_cmd += ["--max-depth", str(int(payload.xgb_max_depth))]
            if payload.xgb_n_estimators is not None:
                train_cmd += ["--n-estimators", str(int(payload.xgb_n_estimators))]
            if payload.xgb_learning_rate is not None:
                train_cmd += ["--learning-rate", str(float(payload.xgb_learning_rate))]
            if payload.xgb_subsample is not None:
                train_cmd += ["--subsample", str(float(payload.xgb_subsample))]
            if payload.xgb_colsample_bytree is not None:
                train_cmd += ["--colsample-bytree", str(float(payload.xgb_colsample_bytree))]
            if payload.xgb_min_child_weight is not None:
                train_cmd += ["--min-child-weight", str(float(payload.xgb_min_child_weight))]
            if payload.xgb_gamma is not None:
                train_cmd += ["--gamma", str(float(payload.xgb_gamma))]
            if payload.xgb_reg_lambda is not None:
                train_cmd += ["--reg-lambda", str(float(payload.xgb_reg_lambda))]
            if payload.xgb_reg_alpha is not None:
                train_cmd += ["--reg-alpha", str(float(payload.xgb_reg_alpha))]
            if payload.xgb_scale_pos_weight is not None:
                train_cmd += ["--scale-pos-weight", str(float(payload.xgb_scale_pos_weight))]
            if payload.xgb_tree_method:
                train_cmd += ["--tree-method", str(payload.xgb_tree_method)]
            if payload.xgb_max_bin is not None:
                train_cmd += ["--max-bin", str(int(payload.xgb_max_bin))]
            if payload.xgb_early_stopping_rounds is not None:
                train_cmd += ["--early-stopping-rounds", str(int(payload.xgb_early_stopping_rounds))]
            if payload.xgb_log1p_counts:
                train_cmd.append("--log1p-counts")
            if payload.xgb_standardize:
                train_cmd.append("--standardize")
            _run_step("train", "Training XGBoost…", train_cmd)
            _run_step(
                "relax",
                "Relaxing thresholds…",
                [
                    sys.executable,
                    str(root_dir / "tools" / "relax_ensemble_thresholds_xgb.py"),
                    "--model",
                    str(Path(str(model_prefix) + ".json")),
                    "--data",
                    str(labeled_path),
                    "--meta",
                    str(meta_path),
                    "--fp-ratio-cap",
                    str(relax_fp_ratio),
                ],
            )
            tune_cmd = [
                sys.executable,
                str(root_dir / "tools" / "tune_ensemble_thresholds_xgb.py"),
                "--model",
                str(Path(str(model_prefix) + ".json")),
                "--meta",
                str(meta_path),
                "--data",
                str(labeled_path),
                "--dataset",
                payload.dataset_id,
                "--optimize",
                optimize_metric,
                "--target-fp-ratio",
                str(target_fp_ratio),
                "--min-recall",
                str(recall_floor),
                "--steps",
                str(steps_val),
                "--eval-iou",
                str(eval_iou),
                "--dedupe-iou",
                str(dedupe_iou),
                "--scoreless-iou",
                str(scoreless_iou),
                "--use-val-split",
            ]
            if target_fp_ratio_by_label_json:
                tune_cmd += [
                    "--target-fp-ratio-by-label-json",
                    target_fp_ratio_by_label_json,
                ]
            if min_recall_by_label_json:
                tune_cmd += [
                    "--min-recall-by-label-json",
                    min_recall_by_label_json,
                ]
            _run_step("objective", "Tuning object-level thresholds…", tune_cmd)
        else:
            _run_step(
                "train",
                "Training MLP…",
                [
                    sys.executable,
                    str(root_dir / "tools" / "train_ensemble_mlp.py"),
                    "--input",
                    str(labeled_path),
                    "--output",
                    str(model_prefix),
                    "--hidden",
                    str(payload.model_hidden or "256,128"),
                    "--dropout",
                    str(_float_with_default(payload.model_dropout, 0.1)),
                    "--epochs",
                    str(int(payload.model_epochs or 20)),
                    "--lr",
                    str(_float_with_default(payload.model_lr, 1e-3)),
                    "--weight-decay",
                    str(_float_with_default(payload.model_weight_decay, 1e-4)),
                    "--seed",
                    str(int(payload.model_seed or 42)),
                    "--device",
                    "cuda",
                ],
            )
            optimize_metric = (payload.optimize_metric or "f1").strip().lower()
            if optimize_metric not in {"f1", "recall"}:
                optimize_metric = "f1"
            steps_val = int(payload.threshold_steps or 200)
            steps_val = max(20, min(1000, steps_val))
            calibrate_cmd = [
                sys.executable,
                str(root_dir / "tools" / "calibrate_ensemble_threshold.py"),
                "--model",
                str(Path(str(model_prefix) + ".pt")),
                "--data",
                str(labeled_path),
                "--meta",
                str(meta_path),
                "--target-fp-ratio",
                str(base_fp_ratio),
                "--min-recall",
                str(recall_floor),
                "--steps",
                str(steps_val),
                "--optimize",
                optimize_metric,
            ]
            if payload.per_class_thresholds is not False:
                calibrate_cmd.append("--per-class")
            _run_step("calibrate", "Calibrating thresholds…", calibrate_cmd)
            _run_step(
                "relax",
                "Relaxing thresholds…",
                [
                    sys.executable,
                    str(root_dir / "tools" / "relax_ensemble_thresholds.py"),
                    "--model",
                    str(Path(str(model_prefix) + ".pt")),
                    "--data",
                    str(labeled_path),
                    "--meta",
                    str(meta_path),
                    "--fp-ratio-cap",
                    str(relax_fp_ratio),
                ],
            )
        update_fn(job, phase="eval", message="Evaluating model…")
        if job.cancel_event.is_set():
            raise RuntimeError("cancelled")
        iou_grid = payload.eval_iou_grid or "0.5,0.6,0.7,0.75,0.8,0.85,0.9"
        dedupe_grid = payload.dedupe_iou_grid or iou_grid
        eval_cmd = [
            sys.executable,
            str(
                root_dir
                / "tools"
                / ("eval_ensemble_xgb_dedupe.py" if calibration_model == "xgb" else "eval_ensemble_mlp_dedupe.py")
            ),
            "--model",
            str(Path(str(model_prefix) + (".json" if calibration_model == "xgb" else ".pt"))),
            "--meta",
            str(meta_path),
            "--data",
            str(labeled_path),
            "--dataset",
            payload.dataset_id,
            "--eval-iou",
            str(eval_iou),
            "--eval-iou-grid",
            iou_grid,
            "--dedupe-iou",
            str(dedupe_iou),
            "--dedupe-iou-grid",
            dedupe_grid,
            "--scoreless-iou",
            str(scoreless_iou),
            "--use-val-split",
        ]
        if calibration_model == "xgb":
            eval_cmd += ["--prepass-jsonl", str(prepass_path)]
        eval_run = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
        eval_text = eval_run.stdout.strip()
        if eval_text:
            eval_path.write_text(eval_text)

        metrics = {}
        try:
            metrics = json.loads(eval_path.read_text())
        except Exception:
            metrics = {}

        job.result = {
            "output_dir": str(output_dir),
            "prepass_jsonl": str(prepass_path),
            "features": str(features_path),
            "labeled": str(labeled_path),
            "model": str(Path(str(model_prefix) + (".json" if calibration_model == "xgb" else ".pt"))),
            "meta": str(meta_path),
            "eval": str(eval_path),
            "metrics": metrics,
            "calibration_model": calibration_model,
        }
        update_fn(job, status="completed", message="Done", phase="completed", progress=1.0)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, RuntimeError) and str(exc) == "cancelled":
            update_fn(job, status="cancelled", message="Cancelled", phase="cancelled")
        else:
            logger.exception("Calibration job %s failed", job.job_id)
            update_fn(job, status="failed", message="Failed", phase="failed", error=str(exc))
    finally:
        job.updated_at = time.time()
