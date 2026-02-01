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


def _load_image_size(image_path: Path) -> Tuple[int, int]:
    try:
        with Image.open(image_path) as im:
            width, height = im.size
            return int(width), int(height)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"image_read_failed:{image_path.name}:{exc}") from exc


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
