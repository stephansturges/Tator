from __future__ import annotations

from pathlib import Path
from typing import List


def _iter_yolo_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
