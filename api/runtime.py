"""APIRouter for runtime control endpoints."""


from typing import Any, Callable

from fastapi import APIRouter


def build_runtime_router(
    *,
    unload_all_fn: Callable[[], None],
) -> APIRouter:
    router = APIRouter()

    @router.post("/runtime/unload")
    def unload_all_runtimes():
        unload_all_fn()
        return {"status": "unloaded"}

    return router
