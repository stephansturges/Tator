"""APIRouter for SAM3 training endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter


def build_sam3_training_router(
    *,
    create_job_fn: Callable[[Any], Any],
    list_jobs_fn: Callable[[], Any],
    get_job_fn: Callable[[str], Any],
    cancel_job_fn: Callable[[str], Any],
    cache_size_fn: Callable[[], Any],
    cache_purge_fn: Callable[[], Any],
    request_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/sam3/train/jobs")
    def create_sam3_training_job(payload: request_cls):  # type: ignore[valid-type]
        return create_job_fn(payload)

    @router.get("/sam3/train/jobs")
    def list_sam3_training_jobs():
        return list_jobs_fn()

    @router.get("/sam3/train/jobs/{job_id}")
    def get_sam3_training_job(job_id: str):
        return get_job_fn(job_id)

    @router.post("/sam3/train/jobs/{job_id}/cancel")
    def cancel_sam3_training_job(job_id: str):
        return cancel_job_fn(job_id)

    @router.get("/sam3/train/cache_size")
    def sam3_train_cache_size():
        return cache_size_fn()

    @router.post("/sam3/train/cache/purge")
    def sam3_train_cache_purge():
        return cache_purge_fn()

    return router
