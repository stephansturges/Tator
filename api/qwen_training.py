"""APIRouter for Qwen training job endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter, Request


def build_qwen_training_router(
    *,
    create_job_fn: Callable[[Any], Any],
    list_jobs_fn: Callable[[Request], Any],
    get_job_fn: Callable[[str, Request], Any],
    cancel_job_fn: Callable[[str], Any],
    cache_size_fn: Callable[[], Any],
    cache_purge_fn: Callable[[], Any],
    request_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/qwen/train/jobs")
    def create_qwen_training_job(payload: request_cls):  # type: ignore[valid-type]
        return create_job_fn(payload)

    @router.get("/qwen/train/jobs")
    def list_qwen_training_jobs(request: Request):
        return list_jobs_fn(request)

    @router.get("/qwen/train/jobs/{job_id}")
    def get_qwen_training_job(job_id: str, request: Request):
        return get_job_fn(job_id, request)

    @router.post("/qwen/train/jobs/{job_id}/cancel")
    def cancel_qwen_training_job(job_id: str):
        return cancel_job_fn(job_id)

    @router.get("/qwen/train/cache_size")
    def qwen_train_cache_size():
        return cache_size_fn()

    @router.post("/qwen/train/cache/purge")
    def qwen_train_cache_purge():
        return cache_purge_fn()

    return router
