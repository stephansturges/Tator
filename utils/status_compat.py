"""Compatibility aliases for Starlette HTTP status constants."""

from __future__ import annotations

from starlette import status as _status


def _status_constant(primary_name: str, legacy_name: str, fallback: int) -> int:
    primary = getattr(_status, primary_name, None)
    if primary is not None:
        return primary
    return getattr(_status, legacy_name, fallback)


HTTP_413_CONTENT_TOO_LARGE = _status_constant(
    "HTTP_413_CONTENT_TOO_LARGE",
    "HTTP_413_REQUEST_ENTITY_TOO_LARGE",
    413,
)

HTTP_422_UNPROCESSABLE_CONTENT = _status_constant(
    "HTTP_422_UNPROCESSABLE_CONTENT",
    "HTTP_422_UNPROCESSABLE_ENTITY",
    422,
)
