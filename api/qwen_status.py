"""APIRouter for Qwen status and settings endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter


def build_qwen_status_router(
    *,
    status_fn: Callable[[], Any],
    get_settings_fn: Callable[[], Any],
    update_settings_fn: Callable[[Any], Any],
    unload_fn: Callable[[], Any],
    settings_cls: Type[Any],
    update_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/qwen/status")
    def qwen_status():
        return status_fn()

    @router.get("/qwen/settings", response_model=settings_cls)
    def qwen_settings():
        return get_settings_fn()

    @router.post("/qwen/settings", response_model=settings_cls)
    def update_qwen_settings(payload: update_cls):  # type: ignore[valid-type]
        return update_settings_fn(payload)

    @router.post("/qwen/unload")
    def qwen_unload():
        return unload_fn()

    return router
