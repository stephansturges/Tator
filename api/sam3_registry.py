"""APIRouter for SAM3 storage/model registry endpoints."""


from typing import Any, Callable, Type

from fastapi import APIRouter, Query


def build_sam3_registry_router(
    *,
    list_runs_fn: Callable[[str], Any],
    delete_run_fn: Callable[[str, str, str], Any],
    promote_run_fn: Callable[[str, str], Any],
    list_models_fn: Callable[[str, bool], Any],
    activate_model_fn: Callable[[Any], Any],
    activate_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/sam3/storage/runs")
    def list_sam3_runs(variant: str = Query("sam3")):
        return list_runs_fn(variant)

    @router.delete("/sam3/storage/runs/{run_id}")
    def delete_sam3_run(run_id: str, variant: str = Query("sam3"), scope: str = Query("all")):
        return delete_run_fn(run_id, variant, scope)

    @router.post("/sam3/storage/runs/{run_id}/promote")
    def promote_sam3_run(run_id: str, variant: str = Query("sam3")):
        return promote_run_fn(run_id, variant)

    @router.get("/sam3/models/available")
    def list_sam3_available_models(variant: str = Query("sam3"), promoted_only: bool = Query(False)):
        return list_models_fn(variant, promoted_only)

    @router.get("/sam3/models")
    def list_sam3_models_alias(variant: str = Query("sam3"), promoted_only: bool = Query(False)):
        return list_models_fn(variant, promoted_only)

    @router.post("/sam3/models/activate")
    def activate_sam3_model(payload: activate_cls):  # type: ignore[valid-type]
        return activate_model_fn(payload)

    return router
