"""APIRouter for Qwen inference endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter


def build_qwen_infer_router(
    *,
    infer_fn: Callable[[Any], Any],
    request_cls: Type[Any],
    response_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/qwen/infer", response_model=response_cls)
    def qwen_infer(payload: request_cls):  # type: ignore[valid-type]
        return infer_fn(payload)

    return router
