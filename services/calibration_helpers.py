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

from utils.io import _path_is_within_root_impl


_CALIBRATION_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def _calibration_sample_images(images: List[str], *, max_images: int, seed: int) -> List[str]:
    if max_images <= 0 or len(images) <= max_images:
        return list(images)
    rng = random.Random(seed)
    picks = list(images)
    rng.shuffle(picks)
    return picks[:max_images]


def _safe_calibration_image_file(path: Path, root: Path) -> bool:
    if path.is_symlink():
        return False
    try:
        resolved_path = path.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except Exception:
        return False
    if not _path_is_within_root_impl(resolved_path, resolved_root):
        return False
    return resolved_path.is_file()


def _calibration_list_images(
    dataset_id: str,
    *,
    resolve_dataset_fn: Callable[[str], Path],
) -> List[str]:
    dataset_root = resolve_dataset_fn(dataset_id)
    try:
        dataset_root_resolved = dataset_root.resolve(strict=True)
    except Exception:
        return []
    images: List[str] = []
    seen: set[str] = set()
    for split in ("val", "train"):
        split_root = dataset_root / split
        if not split_root.exists() or not split_root.is_dir() or split_root.is_symlink():
            continue
        image_root = split_root / "images"
        if image_root.exists() and image_root.is_dir() and not image_root.is_symlink():
            try:
                image_root_resolved = image_root.resolve(strict=True)
            except Exception:
                image_root_resolved = None
            if image_root_resolved is not None and _path_is_within_root_impl(
                image_root_resolved, dataset_root_resolved
            ):
                for entry in sorted(image_root.rglob("*")):
                    if entry.suffix.lower() not in _CALIBRATION_IMAGE_EXTS:
                        continue
                    if not _safe_calibration_image_file(entry, image_root):
                        continue
                    rel = entry.relative_to(image_root).as_posix()
                    if rel in seen:
                        continue
                    seen.add(rel)
                    images.append(rel)
        for entry in sorted(split_root.iterdir()):
            if entry.suffix.lower() not in _CALIBRATION_IMAGE_EXTS:
                continue
            if not _safe_calibration_image_file(entry, split_root):
                continue
            # Cache keys and downstream lookups are relative-path based; avoid duplicate
            # names across splits colliding onto a single cache record.
            rel = entry.name
            if rel in seen:
                continue
            seen.add(rel)
            images.append(rel)
    return images


def _calibration_resolve_image_path(dataset_root: Path, image_name: str) -> Optional[Path]:
    rel_path = Path(str(image_name or ""))
    if (
        not str(image_name or "").strip()
        or rel_path.is_absolute()
        or any(part in {"", ".", ".."} for part in rel_path.parts)
    ):
        return None
    try:
        dataset_root_resolved = dataset_root.resolve(strict=True)
    except Exception:
        return None
    for split in ("val", "train"):
        for candidate in (
            dataset_root / split / "images" / rel_path,
            dataset_root / split / rel_path,
        ):
            if not _safe_calibration_image_file(candidate, dataset_root):
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except Exception:
                continue
            if not _path_is_within_root_impl(resolved, dataset_root_resolved):
                continue
            return resolved
    return None


def _calibration_hash_payload(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _path_identity(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except RuntimeError:
        return path.absolute()


def _path_has_symlink_component(path: Path) -> bool:
    candidate = path if path.is_absolute() else path.absolute()
    checks = [candidate]
    checks.extend(candidate.parents)
    for component in checks:
        if component == component.parent:
            continue
        if component.is_symlink():
            return True
    return False


def _unlink_self_referential_symlink(path: Path) -> bool:
    if not path.is_symlink():
        return False
    try:
        target = Path(os.readlink(path))
    except OSError:
        return False
    if not target.is_absolute():
        target = path.parent / target
    if _path_identity(target) != _path_identity(path):
        return False
    path.unlink(missing_ok=True)
    return True


def _calibration_safe_link(src: Path, dest: Path) -> None:
    try:
        src_resolved = src.resolve()
        if src_resolved == _path_identity(dest):
            return
        _unlink_self_referential_symlink(dest)
        if dest.is_symlink() and not dest.exists():
            dest.unlink()
        if dest.exists() or dest.is_symlink():
            return
        if _path_has_symlink_component(dest.parent):
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        if _path_has_symlink_component(dest.parent):
            return
        os.symlink(str(src_resolved), dest)
    except Exception:
        try:
            if src.resolve() == _path_identity(dest):
                return
            _unlink_self_referential_symlink(dest)
            if dest.exists():
                return
            if _path_has_symlink_component(dest.parent):
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            if _path_has_symlink_component(dest.parent):
                return
            tmp_path = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
            if tmp_path.is_symlink():
                tmp_path.unlink(missing_ok=True)
            elif tmp_path.exists():
                if tmp_path.is_dir():
                    return
                tmp_path.unlink()
            try:
                shutil.copy2(src, tmp_path)
                os.replace(tmp_path, dest)
            finally:
                if tmp_path.exists() or tmp_path.is_symlink():
                    tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _calibration_write_record_atomic(path: Path, record: Dict[str, Any]) -> None:
    if _path_has_symlink_component(path.parent):
        raise ValueError("calibration_record_parent_symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink_component(path.parent):
        raise ValueError("calibration_record_parent_symlink")
    parent_resolved = path.parent.resolve(strict=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    for candidate in (tmp_path, path):
        if candidate.is_symlink():
            candidate.unlink(missing_ok=True)
        elif candidate.exists() and candidate.is_dir():
            raise ValueError("calibration_record_target_is_directory")
        try:
            candidate.resolve(strict=False).relative_to(parent_resolved)
        except Exception as exc:
            raise ValueError("calibration_record_path_not_allowed") from exc
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
    except Exception as exc:
        raise RuntimeError(f"deep_prepass_payload_invalid:{exc}") from exc
    dataset_root = resolve_dataset_fn(dataset_id)
    for image_name, cache_path in tasks:
        if cancel_event is not None and cancel_event.is_set():
            break
        img_path = _calibration_resolve_image_path(dataset_root, image_name)
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
