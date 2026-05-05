"""APIRouter for dataset-scale automatic labeling jobs."""

from typing import Any, Callable, Type

from fastapi import APIRouter


def build_auto_label_router(
    *,
    start_fn: Callable[[Any], Any],
    list_fn: Callable[[], Any],
    get_fn: Callable[[str], Any],
    cancel_fn: Callable[[str], Any],
    request_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/auto_label/jobs")
    def start_auto_label_job(payload: request_cls):  # type: ignore[valid-type]
        return start_fn(payload)

    @router.get("/auto_label/jobs")
    def list_auto_label_jobs():
        return list_fn()

    @router.get("/auto_label/jobs/{job_id}")
    def get_auto_label_job(job_id: str):
        return get_fn(job_id)

    @router.post("/auto_label/jobs/{job_id}/cancel")
    def cancel_auto_label_job(job_id: str):
        return cancel_fn(job_id)

    return router

