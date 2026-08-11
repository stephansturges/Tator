"""APIRouter for Qwen caption endpoint."""


from typing import Any, Callable, Optional, Type

from fastapi import APIRouter, Query


def build_qwen_caption_router(
    *,
    caption_fn: Callable[[Any], Any],
    preview_fn: Optional[Callable[[Any], Any]] = None,
    cancel_fn: Optional[Callable[..., Any]] = None,
    request_cls: Type[Any],
    response_cls: Type[Any],
    preview_response_cls: Optional[Type[Any]] = None,
) -> APIRouter:
    router = APIRouter()

    @router.post("/qwen/caption", response_model=response_cls)
    def qwen_caption(payload: request_cls):  # type: ignore[valid-type]
        return caption_fn(payload)

    if preview_fn is not None and preview_response_cls is not None:

        @router.post("/qwen/caption/preview_prompt", response_model=preview_response_cls)
        def qwen_caption_preview_prompt(payload: request_cls):  # type: ignore[valid-type]
            return preview_fn(payload)

    @router.post("/qwen/caption/cancel")
    def qwen_caption_cancel(force: bool = Query(False)):
        if cancel_fn is None:
            return {"cancelled": False, "message": "Qwen caption cancellation is unavailable."}
        return cancel_fn(force=force)

    return router
