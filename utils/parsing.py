from __future__ import annotations

import json
import re
from typing import Any, Optional


def _coerce_int(value: Any, fallback: int, *, minimum: Optional[int] = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = fallback
    if minimum is not None and result < minimum:
        result = minimum
    return result


def _coerce_float(
    value: Any,
    fallback: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = fallback
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _normalise_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    return trimmed


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_run_name(desired: Optional[str], fallback: str) -> str:
    name = desired or fallback
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("_") or fallback


def _normalize_device_list(devices: Optional[list[Any]]) -> list[int]:
    if not devices:
        return []
    cleaned: list[int] = []
    for value in devices:
        try:
            cleaned.append(int(value))
        except Exception:
            continue
    return [value for value in cleaned if value >= 0]


def _parse_device_ids_string(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return []
    ids: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"invalid_device_token:{part}")
        ids.append(int(part))
    return ids


def _agent_extract_json_array(text: str) -> Optional[list[Any]]:
    if not text:
        return None
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except Exception:
        return None
    return parsed if isinstance(parsed, list) else None
