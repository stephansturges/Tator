"""APIRouter for detector default selection endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter


def build_detectors_default_router(
    *,
    load_default_fn: Callable[[], Any],
    save_default_fn: Callable[[dict], Any],
    request_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/detectors/default")
    def get_default_detector():
        return load_default_fn()

    @router.post("/detectors/default")
    def set_default_detector(payload: request_cls):  # type: ignore[valid-type]
        data = {"mode": payload.mode}
        return save_default_fn(data)

    return router
