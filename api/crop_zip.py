"""APIRouter for crop ZIP export endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter, Query


def build_crop_zip_router(
    *,
    init_fn: Callable[[], Any],
    chunk_fn: Callable[[Any, str], Any],
    finalize_fn: Callable[[str], Any],
    request_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/crop_zip_init")
    def crop_zip_init():
        return init_fn()

    @router.post("/crop_zip_chunk")
    def crop_zip_chunk(request: request_cls, jobId: str = Query(...)):  # type: ignore[valid-type]
        return chunk_fn(request, jobId)

    @router.get("/crop_zip_finalize")
    def crop_zip_finalize(jobId: str):
        return finalize_fn(jobId)

    return router
