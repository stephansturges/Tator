from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _calibration_sample_images(images: List[str], *, max_images: int, seed: int) -> List[str]:
    if max_images <= 0 or len(images) <= max_images:
        return list(images)
    rng = random.Random(seed)
    picks = list(images)
    rng.shuffle(picks)
    return picks[:max_images]


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
