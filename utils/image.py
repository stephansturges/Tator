"""Image utility helpers."""

from __future__ import annotations

import logging
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
import numpy as np
from fastapi import HTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_428_PRECONDITION_REQUIRED,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

logger = logging.getLogger(__name__)


def _slice_image_sahi(
    pil_img: Image.Image,
    slice_size: int,
    overlap: float,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    try:
        from sahi.slicing import slice_image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"sahi_unavailable:{exc}",
        ) from exc
    array = np.array(pil_img)
    result = slice_image(
        image=array,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )
    slices = getattr(result, "images", None)
    starts = getattr(result, "starting_pixels", None)
    if slices is None or starts is None:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="sahi_slice_failed")
    return slices, starts


def _decode_image_base64_impl(
    image_base64: str,
    *,
    max_bytes: Optional[int],
    max_dim: Optional[int],
    allow_downscale: bool,
) -> Tuple[Image.Image, np.ndarray]:
    """Decode base64 image with size/dimension guards and optional downscale."""
    if not image_base64:
        raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="image_payload_missing")
    raw = image_base64
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    if max_bytes:
        est_bytes = (len(raw) * 3) // 4
        if est_bytes > max_bytes * 2:
            raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="image_base64_too_large")
    try:
        data = base64.b64decode(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_base64:{exc}") from exc
    if max_bytes and len(data) > max_bytes:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="image_bytes_too_large")
    try:
        pil_img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_image:{exc}") from exc
    if max_dim:
        width, height = pil_img.size
        if width > max_dim or height > max_dim:
            if not allow_downscale:
                raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="image_too_large_dim")
            try:
                resample = getattr(Image, "Resampling", Image).LANCZOS  # Pillow 10 compat
            except Exception:
                resample = Image.LANCZOS
            pil_img = pil_img.copy()
            pil_img.thumbnail((max_dim, max_dim), resample)
    np_img = np.array(pil_img)
    return pil_img, np_img


def _image_path_for_label_impl(
    labels_dir: Path,
    images_dir: Path,
    label_file: Path,
    image_exts: set[str],
) -> Optional[Path]:
    stem = label_file.stem
    try:
        rel_label = label_file.relative_to(labels_dir)
    except Exception:
        rel_label = Path(label_file.name)
    # Prefer mirrored subdirectory structure when present.
    for ext in image_exts:
        candidate = images_dir / rel_label.with_suffix(ext)
        if candidate.exists():
            return candidate
    for ext in image_exts:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    for candidate in images_dir.rglob(f"{stem}.*"):
        if candidate.suffix.lower() in image_exts:
            return candidate
    return None


def _resolve_coco_image_path_impl(
    file_name: str,
    images_dir: Path,
    split_name: str,
    dataset_root: Path,
) -> Optional[Path]:
    if not file_name:
        return None
    rel_path = Path(file_name)
    candidates: List[Path] = []
    if rel_path.is_absolute():
        candidates.append(rel_path)
    candidates.append(images_dir / rel_path)
    candidates.append(images_dir / rel_path.name)
    candidates.append(dataset_root / rel_path)
    candidates.append(dataset_root / split_name / "images" / rel_path.name)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _label_relpath_for_image_impl(file_name: str) -> Path:
    rel_path = Path(file_name)
    if rel_path.is_absolute():
        rel_path = Path(rel_path.name)
    if "images" in rel_path.parts:
        idx = rel_path.parts.index("images")
        rel_path = Path(*rel_path.parts[idx + 1 :])
    return rel_path.with_suffix(".txt")
