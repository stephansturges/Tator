"""APIRouter for YOLO training and head-graft job endpoints."""


from typing import Any, Callable, Type

from fastapi import APIRouter


def build_yolo_training_router(
    *,
    create_job_fn: Callable[[Any], Any],
    list_jobs_fn: Callable[[], Any],
    get_job_fn: Callable[[str], Any],
    cancel_job_fn: Callable[[str], Any],
    head_graft_create_fn: Callable[[Any], Any],
    head_graft_dry_run_fn: Callable[[Any], Any],
    head_graft_list_fn: Callable[[], Any],
    head_graft_get_fn: Callable[[str], Any],
    head_graft_cancel_fn: Callable[[str], Any],
    train_request_cls: Type[Any],
    head_graft_request_cls: Type[Any],
    head_graft_dry_run_request_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/yolo/train/jobs")
    def create_yolo_training_job(payload: train_request_cls):  # type: ignore[valid-type]
        return create_job_fn(payload)

    @router.get("/yolo/train/jobs")
    def list_yolo_training_jobs():
        return list_jobs_fn()

    @router.get("/yolo/train/jobs/{job_id}")
    def get_yolo_training_job(job_id: str):
        return get_job_fn(job_id)

    @router.post("/yolo/train/jobs/{job_id}/cancel")
    def cancel_yolo_training_job(job_id: str):
        return cancel_job_fn(job_id)

    @router.post("/yolo/head_graft/jobs")
    def create_yolo_head_graft_job(payload: head_graft_request_cls):  # type: ignore[valid-type]
        return head_graft_create_fn(payload)

    @router.post("/yolo/head_graft/dry_run")
    def yolo_head_graft_dry_run(payload: head_graft_dry_run_request_cls):  # type: ignore[valid-type]
        return head_graft_dry_run_fn(payload)

    @router.get("/yolo/head_graft/jobs")
    def list_yolo_head_graft_jobs():
        return head_graft_list_fn()

    @router.get("/yolo/head_graft/jobs/{job_id}")
    def get_yolo_head_graft_job(job_id: str):
        return head_graft_get_fn(job_id)

    @router.post("/yolo/head_graft/jobs/{job_id}/cancel")
    def cancel_yolo_head_graft_job(job_id: str):
        return head_graft_cancel_fn(job_id)

    return router
