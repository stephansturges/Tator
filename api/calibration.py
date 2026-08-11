"""APIRouter for calibration job endpoints."""


from typing import Any, Callable, Optional, Type

from fastapi import APIRouter, Body


def build_calibration_router(
    *,
    start_fn: Callable[[Any], Any],
    list_fn: Callable[[], Any],
    get_fn: Callable[[str], Any],
    cancel_fn: Callable[[str], Any],
    report_bundle_fn: Optional[Callable[[str], Any]] = None,
    request_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/calibration/jobs")
    def start_calibration_job(payload: request_cls = Body(...)):  # type: ignore[valid-type]
        return start_fn(payload)

    @router.get("/calibration/jobs")
    def list_calibration_jobs():
        return list_fn()

    @router.get("/calibration/jobs/{job_id}")
    def get_calibration_job(job_id: str):
        return get_fn(job_id)

    @router.post("/calibration/jobs/{job_id}/cancel")
    def cancel_calibration_job(job_id: str):
        return cancel_fn(job_id)

    if report_bundle_fn is not None:

        @router.get("/calibration/jobs/{job_id}/artifacts/report_bundle")
        def get_calibration_report_bundle(job_id: str):
            return report_bundle_fn(job_id)

    return router
