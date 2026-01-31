from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from PIL import Image
from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

logger = logging.getLogger(__name__)


def _load_image_size(image_path: Path) -> Tuple[int, int]:
    try:
        with Image.open(image_path) as im:
            width, height = im.size
            return int(width), int(height)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"image_read_failed:{image_path.name}:{exc}") from exc
