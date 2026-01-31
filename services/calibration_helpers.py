from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List


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
