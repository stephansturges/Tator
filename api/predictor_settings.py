"""APIRouter for predictor settings endpoints."""


from typing import Any, Callable, Type

from fastapi import APIRouter


def build_predictor_settings_router(
    *,
    get_payload_fn: Callable[[], Any],
    update_fn: Callable[[Any], Any],
    settings_cls: Type[Any],
    update_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/predictor_settings", response_model=settings_cls)
    def get_predictor_settings():
        return get_payload_fn()

    @router.post("/predictor_settings", response_model=settings_cls)
    def update_predictor_settings(payload: update_cls):  # type: ignore[valid-type]
        return update_fn(payload)

    return router
