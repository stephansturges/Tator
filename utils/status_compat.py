"""Compatibility aliases for Starlette HTTP status constants."""

from __future__ import annotations

from starlette import status as _status

HTTP_413_CONTENT_TOO_LARGE = getattr(
    _status,
    "HTTP_413_CONTENT_TOO_LARGE",
    getattr(_status, "HTTP_413_REQUEST_ENTITY_TOO_LARGE", 413),
)

HTTP_422_UNPROCESSABLE_CONTENT = getattr(
    _status,
    "HTTP_422_UNPROCESSABLE_CONTENT",
    getattr(_status, "HTTP_422_UNPROCESSABLE_ENTITY", 422),
)
