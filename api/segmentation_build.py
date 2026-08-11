"""APIRouter for segmentation build job endpoints."""


from typing import Any, Callable, Type

from fastapi import APIRouter


def build_segmentation_build_router(
    *,
    start_fn: Callable[[Any], Any],
    list_fn: Callable[[], Any],
    get_fn: Callable[[str], Any],
    request_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/segmentation/build/jobs")
    def start_segmentation_build_job(payload: request_cls):  # type: ignore[valid-type]
        return start_fn(payload)

    @router.get("/segmentation/build/jobs")
    def list_segmentation_build_jobs():
        return list_fn()

    @router.get("/segmentation/build/jobs/{job_id}")
    def get_segmentation_build_job(job_id: str):
        return get_fn(job_id)

    return router
