"""APIRouter for Qwen caption endpoint."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter


def build_qwen_caption_router(
    *,
    caption_fn: Callable[[Any], Any],
    request_cls: Type[Any],
    response_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/qwen/caption", response_model=response_cls)
    def qwen_caption(payload: request_cls):  # type: ignore[valid-type]
        return caption_fn(payload)

    return router
