"""APIRouter for CLIP active model endpoints."""


from typing import Any, Callable, Type

from fastapi import APIRouter


def build_clip_active_model_router(
    *,
    get_fn: Callable[[], Any],
    set_fn: Callable[[Any], Any],
    request_cls: Type[Any],
    response_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/clip/active_model", response_model=response_cls)
    def get_active_model():
        return get_fn()

    @router.post("/clip/active_model", response_model=response_cls)
    def set_active_model(payload: request_cls):  # type: ignore[valid-type]
        return set_fn(payload)

    return router
