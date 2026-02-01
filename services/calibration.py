"""Calibration job orchestration and serialization."""

from __future__ import annotations

import hashlib
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

import torch
from PIL import Image


def _serialize_calibration_job(job: Any) -> Dict[str, Any]:
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
        "result": job.result,
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
        if not payload.classifier_id and not isinstance(active_classifier_head, dict):
            raise http_exception_cls(status_code=412, detail="calibration_classifier_required")
        prepass_payload = prepass_request_cls(
            dataset_id=payload.dataset_id,
            enable_yolo=True,
            enable_rfdetr=True,
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
            sam3_text_window_extension=bool(payload.sam3_text_window_extension)
            if payload.sam3_text_window_extension is not None
            else True,
            sam3_text_window_mode=payload.sam3_text_window_mode or "grid",
            sam3_text_window_size=payload.sam3_text_window_size,
            sam3_text_window_overlap=payload.sam3_text_window_overlap,
            prepass_sam3_text_thr=float(payload.prepass_sam3_text_thr or 0.2),
            prepass_similarity_score=float(payload.prepass_similarity_score or 0.3),
            similarity_min_exemplar_score=float(payload.similarity_min_exemplar_score or 0.6),
            similarity_window_extension=bool(payload.similarity_window_extension),
            sam3_score_thr=float(payload.sam3_score_thr or 0.2),
            sam3_mask_threshold=float(payload.sam3_mask_threshold or 0.2),
            detector_conf=float(payload.detector_conf or 0.45),
            sahi_window_size=payload.sahi_window_size,
            sahi_overlap_ratio=payload.sahi_overlap_ratio,
            classifier_id=payload.classifier_id,
            scoreless_iou=float(payload.scoreless_iou or 0.0),
            iou=float(payload.dedupe_iou or 0.75),
        )

        labelmap_hash = hashlib.sha1(",".join(labelmap).encode("utf-8")).hexdigest()
        prepass_glossary_text = glossary or ""
        glossary_hash = hashlib.sha1(prepass_glossary_text.encode("utf-8")).hexdigest()
        selected_hash = hashlib.sha1(json.dumps(selected, sort_keys=True).encode("utf-8")).hexdigest()
        prepass_config = {
            "sam3_text_synonym_budget": None
            if payload.sam3_text_synonym_budget is None
            else int(payload.sam3_text_synonym_budget),
            "sam3_text_window_extension": payload.sam3_text_window_extension,
            "sam3_text_window_mode": payload.sam3_text_window_mode,
            "sam3_text_window_size": payload.sam3_text_window_size,
            "sam3_text_window_overlap": payload.sam3_text_window_overlap,
            "prepass_sam3_text_thr": float(payload.prepass_sam3_text_thr or 0.2),
            "prepass_similarity_score": float(payload.prepass_similarity_score or 0.3),
            "similarity_min_exemplar_score": float(payload.similarity_min_exemplar_score or 0.6),
            "similarity_window_extension": bool(payload.similarity_window_extension),
            "sam3_score_thr": float(payload.sam3_score_thr or 0.2),
            "sam3_mask_threshold": float(payload.sam3_mask_threshold or 0.2),
            "detector_conf": float(payload.detector_conf or 0.45),
            "sahi_window_size": payload.sahi_window_size,
            "sahi_overlap_ratio": payload.sahi_overlap_ratio,
            "scoreless_iou": float(payload.scoreless_iou or 0.0),
            "dedupe_iou": float(payload.dedupe_iou or 0.75),
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
                    record = {
                        "image": image_name,
                        "dataset_id": payload.dataset_id,
                        "detections": detections,
                        "warnings": warnings,
                    }
                    _write_cached_record(image_name, record)
                    cached_records[image_name] = record
                    processed += 1
                    progress = processed / total if total else 1.0
                    update_fn(job, processed=processed, progress=progress)

        with prepass_path.open("w", encoding="utf-8") as handle:
            for image_name in selected:
                record = cached_records.get(image_name) or _load_cached_record(image_name)
                if not record:
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

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

        features_path = output_dir / "ensemble_features.npz"
        labeled_path = output_dir / f"ensemble_features_iou{float(payload.label_iou or 0.9):.2f}.npz"
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

        classifier_id = payload.classifier_id or ""
        features_key = hash_payload_fn(
            {
                "prepass_key": prepass_key,
                "classifier_id": classifier_id or "",
                "support_iou": float(payload.support_iou or 0.5),
                "min_crop_size": 4,
                "context_radius": float(payload.context_radius or 0.075),
                "features_version": calibration_features_version,
            }
        )
        features_cache_dir = calibration_cache_root / "features" / features_key
        features_cache_dir.mkdir(parents=True, exist_ok=True)
        features_cache_path = features_cache_dir / "ensemble_features.npz"
        features_cache_meta = features_cache_dir / "features.meta.json"
        cached_features = features_cache_path.exists()

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
                    str(float(payload.support_iou or 0.5)),
                    "--min-crop-size",
                    "4",
                    "--context-radius",
                    str(float(payload.context_radius or 0.075)),
                    "--device",
                    "cuda",
                ]
                + (["--classifier-id", classifier_id] if classifier_id else []),
            )
            features_cache_meta.write_text(
                json.dumps(
                    {
                        "prepass_key": prepass_key,
                        "features_key": features_key,
                        "features_version": calibration_features_version,
                    },
                    indent=2,
                )
            )
            safe_link_fn(features_cache_path, features_path)

        labeled_key = hash_payload_fn(
            {
                "features_key": features_key,
                "label_iou": float(payload.label_iou or 0.9),
            }
        )
        labeled_cache_dir = calibration_cache_root / "labeled" / labeled_key
        labeled_cache_dir.mkdir(parents=True, exist_ok=True)
        labeled_cache_path = labeled_cache_dir / f"ensemble_features_iou{float(payload.label_iou or 0.9):.2f}.npz"
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
                    str(float(payload.label_iou or 0.9)),
                ],
            )
            labeled_cache_meta.write_text(
                json.dumps(
                    {
                        "features_key": features_key,
                        "labeled_key": labeled_key,
                        "label_iou": float(payload.label_iou or 0.9),
                    },
                    indent=2,
                )
            )
            safe_link_fn(labeled_cache_path, labeled_path)
        if calibration_model == "xgb":
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
                str(float(payload.base_fp_ratio or 0.1)),
                "--min-recall",
                str(float(payload.recall_floor or 0.6)),
                "--threshold-steps",
                str(int(payload.threshold_steps or 200)),
            ]
            if payload.per_class_thresholds is not False:
                train_cmd.append("--per-class")
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
                    str(float(payload.relax_fp_ratio or 0.2)),
                ],
            )
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
                    str(float(payload.model_dropout or 0.1)),
                    "--epochs",
                    str(int(payload.model_epochs or 20)),
                    "--lr",
                    str(float(payload.model_lr or 1e-3)),
                    "--weight-decay",
                    str(float(payload.model_weight_decay or 1e-4)),
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
                str(float(payload.base_fp_ratio or 0.1)),
                "--min-recall",
                str(float(payload.recall_floor or 0.6)),
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
                    str(float(payload.relax_fp_ratio or 0.2)),
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
            str(float(payload.eval_iou or 0.5)),
            "--eval-iou-grid",
            iou_grid,
            "--dedupe-iou",
            str(float(payload.dedupe_iou or 0.1)),
            "--dedupe-iou-grid",
            dedupe_grid,
            "--scoreless-iou",
            str(float(payload.scoreless_iou or 0.0)),
            "--use-val-split",
        ]
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
