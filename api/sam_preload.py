"""APIRouter for SAM preload endpoint."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter


def build_sam_preload_router(
    *,
    preload_fn: Callable[[Any], Any],
    request_cls: Type[Any],
    response_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/sam_preload", response_model=response_cls)
    def sam_preload(payload: request_cls):  # type: ignore[valid-type]
        return preload_fn(payload)

    return router
