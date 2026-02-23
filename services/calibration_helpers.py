"""Calibration helpers for caching + prepass worker orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch


def _calibration_sample_images(images: List[str], *, max_images: int, seed: int) -> List[str]:
    if max_images <= 0 or len(images) <= max_images:
        return list(images)
    rng = random.Random(seed)
    picks = list(images)
    rng.shuffle(picks)
    return picks[:max_images]


def _calibration_list_images(
    dataset_id: str,
    *,
    resolve_dataset_fn: Callable[[str], Path],
) -> List[str]:
    dataset_root = resolve_dataset_fn(dataset_id)
    images: List[str] = []
    for split in ("val", "train"):
        split_root = dataset_root / split
        if not split_root.exists():
            continue
        for entry in sorted(split_root.iterdir()):
            if entry.is_file() and entry.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
            }:
                images.append(entry.name)
    return images


def _calibration_hash_payload(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _calibration_safe_link(src: Path, dest: Path) -> None:
    try:
        if dest.is_symlink() and not dest.exists():
            dest.unlink()
        if dest.exists() or dest.is_symlink():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(src.resolve()), dest)
    except Exception:
        try:
            if dest.exists():
                return
            shutil.copy2(src, dest)
        except Exception:
            pass


def _calibration_write_record_atomic(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(record, ensure_ascii=False))
    tmp_path.replace(path)


def _calibration_update(job: Any, **kwargs: Any) -> None:
    for key, value in kwargs.items():
        setattr(job, key, value)
    job.updated_at = time.time()


def _calibration_cache_image(
    pil_img: Image.Image,
    sam_variant: Any,
    *,
    store_preloaded_fn: Callable[[str, Any, Any], None],
    default_variant_fn: Callable[[Any], Any],
) -> str:
    np_img = np.ascontiguousarray(np.array(pil_img.convert("RGB")))
    token = hashlib.md5(np_img.tobytes()).hexdigest()
    store_preloaded_fn(token, np_img, default_variant_fn(sam_variant))
    return token


def _calibration_prepass_worker(
    device_index: int,
    tasks: List[Tuple[str, str]],
    dataset_id: str,
    labelmap: List[str],
    glossary: str,
    prepass_payload_dict: Dict[str, Any],
    *,
    cancel_event: Optional[Any],
    progress_queue: Optional[Any],
    resolve_dataset_fn: Callable[[str], Path],
    prepass_request_cls: Any,
    cache_image_fn: Callable[[Image.Image, Optional[str]], str],
    run_prepass_fn: Callable[..., Dict[str, Any]],
    write_record_fn: Callable[[Path, Dict[str, Any]], None],
    set_device_pref_fn: Optional[Callable[[int], None]] = None,
) -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_index)
    except Exception:
        pass
    if set_device_pref_fn is not None:
        try:
            set_device_pref_fn(device_index)
        except Exception:
            pass
    try:
        prepass_payload = prepass_request_cls(**prepass_payload_dict)
    except Exception:
        return
    dataset_root = resolve_dataset_fn(dataset_id)
    for image_name, cache_path in tasks:
        if cancel_event is not None and cancel_event.is_set():
            break
        img_path = None
        for split in ("val", "train"):
            candidate = dataset_root / split / image_name
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            try:
                write_record_fn(
                    Path(cache_path),
                    {
                        "image": image_name,
                        "dataset_id": dataset_id,
                        "detections": [],
                        "warnings": ["deep_prepass_image_missing"],
                    },
                )
            except Exception:
                pass
            if progress_queue is not None:
                progress_queue.put(1)
            continue
        try:
            with Image.open(img_path) as img:
                pil_img = img.convert("RGB")
        except Exception as exc:
            try:
                write_record_fn(
                    Path(cache_path),
                    {
                        "image": image_name,
                        "dataset_id": dataset_id,
                        "detections": [],
                        "warnings": [f"deep_prepass_image_open_failed:{exc}"],
                    },
                )
            except Exception:
                pass
            if progress_queue is not None:
                progress_queue.put(1)
            continue
        try:
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
                "dataset_id": dataset_id,
                "detections": detections,
                "warnings": warnings,
            }
            if isinstance(provenance, dict):
                record["provenance"] = provenance
            write_record_fn(Path(cache_path), record)
        except Exception as exc:
            try:
                write_record_fn(
                    Path(cache_path),
                    {
                        "image": image_name,
                        "dataset_id": dataset_id,
                        "detections": [],
                        "warnings": [f"deep_prepass_worker_failed:{exc}"],
                    },
                )
            except Exception:
                pass
        if progress_queue is not None:
            progress_queue.put(1)
