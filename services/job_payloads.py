"""Shared helpers for job API payloads."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


def json_sanitize(value: Any) -> Any:
    """Return a JSON-safe copy of a nested job payload."""
    if isinstance(value, dict):
        return {str(key): json_sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if hasattr(value, "tolist"):
        try:
            return json_sanitize(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return json_sanitize(value.item())
        except Exception:
            pass
    if isinstance(value, Real):
        try:
            if not math.isfinite(float(value)):
                return None
        except Exception:
            return None
        return value
    return value


def clamp_progress(
    value: Any,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
    fallback: float | None = None,
) -> float | None:
    """Clamp finite progress values without promoting NaN/inf to completion."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    lower = float(minimum)
    upper = float(maximum)
    if upper < lower:
        lower, upper = upper, lower
    return max(lower, min(upper, parsed))
