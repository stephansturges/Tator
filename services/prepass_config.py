from __future__ import annotations

from typing import List, Optional


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
