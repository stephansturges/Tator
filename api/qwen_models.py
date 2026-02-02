"""APIRouter for Qwen model registry endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter


def build_qwen_models_router(
    *,
    list_models_fn: Callable[[], Any],
    activate_fn: Callable[[Any], Any],
    activate_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/qwen/models")
    def list_qwen_models():
        return list_models_fn()

    @router.post("/qwen/models/activate")
    def activate_qwen_model(payload: activate_cls):  # type: ignore[valid-type]
        return activate_fn(payload)

    return router
