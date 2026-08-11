"""Dataset path/metadata helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List


def _iter_yolo_images(images_dir: Path) -> List[Path]:
    root = images_dir.resolve()
    if not root.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in exts
            and _path_is_within_root(p.resolve(), root)
        ]
    )


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False
