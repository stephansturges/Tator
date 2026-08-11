"""APIRouter for RF-DETR registry and inference endpoints."""


from typing import Any, Callable, Optional, Type

from fastapi import APIRouter, Query


def build_rfdetr_router(
    *,
    list_variants_fn: Callable[[Optional[str]], Any],
    list_runs_fn: Callable[[], Any],
    get_active_fn: Callable[[], Any],
    set_active_fn: Callable[[Any], Any],
    download_run_fn: Callable[[str], Any],
    summary_fn: Callable[[str], Any],
    delete_run_fn: Callable[[str], Any],
    predict_region_fn: Callable[[Any], Any],
    predict_full_fn: Callable[[Any], Any],
    predict_windowed_fn: Callable[[Any], Any],
    active_request_cls: Type[Any],
    region_request_cls: Type[Any],
    full_request_cls: Type[Any],
    windowed_request_cls: Type[Any],
    region_response_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/rfdetr/variants")
    def list_rfdetr_variants(task: Optional[str] = Query(None)):
        return list_variants_fn(task)

    @router.get("/rfdetr/runs")
    def list_rfdetr_runs():
        return list_runs_fn()

    @router.get("/rfdetr/active")
    def get_rfdetr_active():
        return get_active_fn()

    @router.post("/rfdetr/active")
    def set_rfdetr_active(payload: active_request_cls):  # type: ignore[valid-type]
        return set_active_fn(payload)

    @router.get("/rfdetr/runs/{run_id}/download")
    def download_rfdetr_run(run_id: str):
        return download_run_fn(run_id)

    @router.get("/rfdetr/runs/{run_id}/summary")
    def rfdetr_run_summary(run_id: str):
        return summary_fn(run_id)

    @router.delete("/rfdetr/runs/{run_id}")
    def delete_rfdetr_run(run_id: str):
        return delete_run_fn(run_id)

    @router.post("/rfdetr/predict_region", response_model=region_response_cls)
    def rfdetr_predict_region(payload: region_request_cls):  # type: ignore[valid-type]
        return predict_region_fn(payload)

    @router.post("/rfdetr/predict_full", response_model=region_response_cls)
    def rfdetr_predict_full(payload: full_request_cls):  # type: ignore[valid-type]
        return predict_full_fn(payload)

    @router.post("/rfdetr/predict_windowed", response_model=region_response_cls)
    def rfdetr_predict_windowed(payload: windowed_request_cls):  # type: ignore[valid-type]
        return predict_windowed_fn(payload)

    return router
