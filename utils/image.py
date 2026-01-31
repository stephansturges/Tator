from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import numpy as np
from fastapi import HTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
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
