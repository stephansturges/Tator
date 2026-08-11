"""APIRouter for YOLO registry and inference endpoints."""


from typing import Any, Callable, Optional, Type

from fastapi import APIRouter, Query


def build_yolo_router(
    *,
    list_variants_fn: Callable[[Optional[str]], Any],
    list_runs_fn: Callable[[], Any],
    get_active_fn: Callable[[], Any],
    set_active_fn: Callable[[Any], Any],
    predict_region_fn: Callable[[Any], Any],
    predict_full_fn: Callable[[Any], Any],
    predict_windowed_fn: Callable[[Any], Any],
    download_run_fn: Callable[[str], Any],
    summary_fn: Callable[[str], Any],
    head_graft_bundle_fn: Callable[[str], Any],
    delete_run_fn: Callable[[str], Any],
    active_request_cls: Type[Any],
    region_request_cls: Type[Any],
    full_request_cls: Type[Any],
    windowed_request_cls: Type[Any],
    region_response_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/yolo/variants")
    def list_yolo_variants(task: Optional[str] = Query(None)):
        return list_variants_fn(task)

    @router.get("/yolo/runs")
    def list_yolo_runs():
        return list_runs_fn()

    @router.get("/yolo/active")
    def get_yolo_active():
        return get_active_fn()

    @router.post("/yolo/active")
    def set_yolo_active(payload: active_request_cls):  # type: ignore[valid-type]
        return set_active_fn(payload)

    @router.post("/yolo/predict_region", response_model=region_response_cls)
    def yolo_predict_region(payload: region_request_cls):  # type: ignore[valid-type]
        return predict_region_fn(payload)

    @router.post("/yolo/predict_full", response_model=region_response_cls)
    def yolo_predict_full(payload: full_request_cls):  # type: ignore[valid-type]
        return predict_full_fn(payload)

    @router.post("/yolo/predict_windowed", response_model=region_response_cls)
    def yolo_predict_windowed(payload: windowed_request_cls):  # type: ignore[valid-type]
        return predict_windowed_fn(payload)

    @router.get("/yolo/runs/{run_id}/download")
    def download_yolo_run(run_id: str):
        return download_run_fn(run_id)

    @router.get("/yolo/runs/{run_id}/summary")
    def yolo_run_summary(run_id: str):
        return summary_fn(run_id)

    @router.get("/yolo/head_graft/jobs/{job_id}/bundle")
    def download_yolo_head_graft_bundle(job_id: str):
        return head_graft_bundle_fn(job_id)

    @router.delete("/yolo/runs/{run_id}")
    def delete_yolo_run(run_id: str):
        return delete_run_fn(run_id)

    return router
