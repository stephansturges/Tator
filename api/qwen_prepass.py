"""APIRouter for Qwen prepass endpoint."""


from typing import Any, Callable, Type

from fastapi import APIRouter


def build_qwen_prepass_router(
    *,
    prepass_fn: Callable[[Any], Any],
    request_cls: Type[Any],
    response_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/qwen/prepass", response_model=response_cls)
    def qwen_prepass(payload: request_cls):  # type: ignore[valid-type]
        return prepass_fn(payload)

    return router
