"""APIRouter for RF-DETR training job endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter


def build_rfdetr_training_router(
    *,
    create_job_fn: Callable[[Any], Any],
    list_jobs_fn: Callable[[], Any],
    get_job_fn: Callable[[str], Any],
    cancel_job_fn: Callable[[str], Any],
    request_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/rfdetr/train/jobs")
    def create_rfdetr_training_job(payload: request_cls):  # type: ignore[valid-type]
        return create_job_fn(payload)

    @router.get("/rfdetr/train/jobs")
    def list_rfdetr_training_jobs():
        return list_jobs_fn()

    @router.get("/rfdetr/train/jobs/{job_id}")
    def get_rfdetr_training_job(job_id: str):
        return get_job_fn(job_id)

    @router.post("/rfdetr/train/jobs/{job_id}/cancel")
    def cancel_rfdetr_training_job(job_id: str):
        return cancel_job_fn(job_id)

    return router
