"""APIRouter for base64 prediction endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter


def build_predict_base64_router(
    *,
    predict_fn: Callable[[Any], Any],
    request_cls: Type[Any],
    response_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/predict_base64", response_model=response_cls)
    def predict_base64(payload: request_cls):  # type: ignore[valid-type]
        return predict_fn(payload)

    return router
