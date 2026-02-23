"""APIRouter for SAM slot status/activation endpoints."""


from typing import Any, Callable, Type

from fastapi import APIRouter


def build_sam_slots_router(
    *,
    status_fn: Callable[[], Any],
    activate_fn: Callable[[Any], Any],
    status_cls: Type[Any],
    activate_req_cls: Type[Any],
    activate_resp_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/sam_slots", response_model=status_cls)
    def sam_slots():
        return status_fn()

    @router.post("/sam_activate_slot", response_model=activate_resp_cls)
    def sam_activate_slot(payload: activate_req_cls):  # type: ignore[valid-type]
        return activate_fn(payload)

    return router
