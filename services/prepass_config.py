from __future__ import annotations

from typing import Any, List, Optional

from fastapi import HTTPException
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE


def _normalize_recipe_thresholds(thresholds: Optional[List[float]], fallback: float, limit: int = 20) -> List[float]:
    values = thresholds if thresholds is not None else [fallback]
    cleaned: List[float] = []
    seen = set()
    for raw in values:
        try:
            val = float(raw)
        except Exception:
            continue
        if val < 0.0 or val > 1.0:
            continue
        key = round(val, 4)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(val)
        if len(cleaned) >= limit:
            break
    return cleaned


def _require_sam3_for_prepass(
    enable_text: bool,
    enable_similarity: bool,
    *,
    sam3_import_error: Optional[Exception],
    build_sam3_image_model: Any,
    sam3_image_processor: Any,
) -> None:
    if not (enable_text or enable_similarity):
        return
    if sam3_import_error is not None or build_sam3_image_model is None or sam3_image_processor is None:
        detail = f"sam3_unavailable:{sam3_import_error}"
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
