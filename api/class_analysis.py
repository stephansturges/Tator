"""APIRouter for class embedding analysis jobs."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from fastapi import APIRouter, Body, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse


def build_class_analysis_router(
    *,
    capabilities_fn: Callable[[], Any],
    create_job_fn: Callable[[dict], Any],
    create_active_workspace_job_fn: Callable[[str, list[UploadFile]], Any],
    get_job_fn: Callable[[str], Any],
    get_result_fn: Callable[[str], Any],
    get_projection_fn: Callable[[str, str], Any],
    get_thumbnail_fn: Callable[[str, str], FileResponse],
    create_cluster_search_fn: Callable[[str, dict], Any],
    get_cluster_search_fn: Callable[[str], Any],
    cancel_cluster_search_fn: Callable[[str], Any],
    cancel_job_fn: Callable[[str], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/class_analysis/capabilities")
    def class_analysis_capabilities():
        return capabilities_fn()

    @router.post("/class_analysis/jobs")
    def create_class_analysis_job(payload: dict = Body(...)):  # noqa: B008
        return create_job_fn(payload or {})

    @router.post("/class_analysis/jobs/active_workspace")
    async def create_active_workspace_class_analysis_job(request: Request):
        form = await request.form(
            max_files=float("inf"),
            max_fields=10_000,
            max_part_size=512 * 1024 * 1024,
        )
        manifest = form.get("manifest")
        if manifest is None or hasattr(manifest, "filename"):
            raise HTTPException(status_code=400, detail="active_workspace_manifest_required")
        files = [
            item
            for item in form.getlist("files")
            if hasattr(item, "filename") and hasattr(item, "read")
        ]
        result = create_active_workspace_job_fn(str(manifest), files)
        if inspect.isawaitable(result):
            return await result
        return result

    @router.get("/class_analysis/jobs/{job_id}")
    def get_class_analysis_job(job_id: str):
        return get_job_fn(job_id)

    @router.get("/class_analysis/jobs/{job_id}/result")
    def get_class_analysis_result(job_id: str):
        return get_result_fn(job_id)

    @router.get("/class_analysis/jobs/{job_id}/projection/{mode}")
    def get_class_analysis_projection(job_id: str, mode: str):
        return get_projection_fn(job_id, mode)

    @router.get("/class_analysis/jobs/{job_id}/thumbnail/{point_id}")
    def get_class_analysis_thumbnail(job_id: str, point_id: str):
        return get_thumbnail_fn(job_id, point_id)

    @router.post("/class_analysis/jobs/{job_id}/cluster_search")
    def create_class_analysis_cluster_search(job_id: str, payload: dict = Body(default_factory=dict)):  # noqa: B008
        return create_cluster_search_fn(job_id, payload or {})

    @router.get("/class_analysis/cluster_search/{cluster_job_id}")
    def get_class_analysis_cluster_search(cluster_job_id: str):
        return get_cluster_search_fn(cluster_job_id)

    @router.post("/class_analysis/cluster_search/{cluster_job_id}/cancel")
    def cancel_class_analysis_cluster_search(cluster_job_id: str):
        return cancel_cluster_search_fn(cluster_job_id)

    @router.post("/class_analysis/jobs/{job_id}/cancel")
    def cancel_class_analysis_job(job_id: str):
        return cancel_job_fn(job_id)

    return router
