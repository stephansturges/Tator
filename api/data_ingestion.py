"""APIRouter for data ingestion and local SALAD diversity jobs."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile


def build_data_ingestion_router(
    *,
    capabilities_fn: Callable[[], Any],
    create_analysis_job_fn: Callable[[str, list[Any], list[Any]], Any],
    create_salad_train_job_fn: Callable[[str, list[Any]], Any],
    export_reference_profile_fn: Callable[[str], Any],
    import_reference_profile_fn: Callable[[UploadFile], Any],
    preview_accepted_export_fn: Callable[[str, dict], Any],
    get_candidate_thumbnail_fn: Callable[[str, str], Any],
    get_reference_thumbnail_fn: Callable[[str, str], Any],
    get_distribution_fn: Callable[[str], Any],
    get_accepted_export_thumbnail_fn: Callable[[str, str, str], Any],
    download_accepted_export_fn: Callable[[str, dict], Any],
    get_job_fn: Callable[[str], Any],
    get_result_fn: Callable[[str], Any],
    cancel_job_fn: Callable[[str], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/data_ingestion/capabilities")
    def data_ingestion_capabilities():
        return capabilities_fn()

    @router.post("/data_ingestion/jobs")
    async def create_data_ingestion_job(request: Request):
        form = await request.form(
            max_files=float("inf"),
            max_fields=10_000,
            max_part_size=1024 * 1024 * 1024,
        )
        manifest = form.get("manifest")
        if manifest is None or hasattr(manifest, "filename"):
            raise HTTPException(status_code=400, detail="data_ingestion_manifest_required")
        candidates = [
            item
            for item in form.getlist("candidate_files")
            if hasattr(item, "filename") and hasattr(item, "read")
        ]
        references = [
            item
            for item in form.getlist("reference_files")
            if hasattr(item, "filename") and hasattr(item, "read")
        ]
        result = create_analysis_job_fn(str(manifest), candidates, references)
        if inspect.isawaitable(result):
            return await result
        return result

    @router.post("/data_ingestion/salad_train_jobs")
    async def create_salad_training_job(request: Request):
        form = await request.form(
            max_files=float("inf"),
            max_fields=10_000,
            max_part_size=1024 * 1024 * 1024,
        )
        manifest = form.get("manifest")
        if manifest is None or hasattr(manifest, "filename"):
            raise HTTPException(status_code=400, detail="salad_train_manifest_required")
        files = [
            item
            for item in form.getlist("train_files")
            if hasattr(item, "filename") and hasattr(item, "read")
        ]
        result = create_salad_train_job_fn(str(manifest), files)
        if inspect.isawaitable(result):
            return await result
        return result

    @router.get("/data_ingestion/reference_profiles/{profile_id}/export")
    def export_reference_profile(profile_id: str):
        return export_reference_profile_fn(profile_id)

    @router.post("/data_ingestion/reference_profiles/import")
    async def import_reference_profile(file: UploadFile = File(...)):  # noqa: B008
        result = import_reference_profile_fn(file)
        if inspect.isawaitable(result):
            return await result
        return result

    @router.get("/data_ingestion/jobs/{job_id}")
    def get_data_ingestion_job(job_id: str):
        return get_job_fn(job_id)

    @router.get("/data_ingestion/jobs/{job_id}/result")
    def get_data_ingestion_result(job_id: str):
        return get_result_fn(job_id)

    @router.post("/data_ingestion/jobs/{job_id}/accepted_export/preview")
    def preview_accepted_export(job_id: str, payload: dict = Body(...)):  # noqa: B008
        return preview_accepted_export_fn(job_id, payload or {})

    @router.get("/data_ingestion/jobs/{job_id}/candidate_thumbnail/{item_id}")
    def get_candidate_thumbnail(job_id: str, item_id: str):
        return get_candidate_thumbnail_fn(job_id, item_id)

    @router.get("/data_ingestion/jobs/{job_id}/reference_thumbnail/{point_id}")
    def get_reference_thumbnail(job_id: str, point_id: str):
        return get_reference_thumbnail_fn(job_id, point_id)

    @router.get("/data_ingestion/jobs/{job_id}/distribution")
    def get_distribution(job_id: str):
        return get_distribution_fn(job_id)

    @router.get("/data_ingestion/jobs/{job_id}/accepted_export/{preview_id}/thumbnail/{output_id}")
    def get_accepted_export_thumbnail(job_id: str, preview_id: str, output_id: str):
        return get_accepted_export_thumbnail_fn(job_id, preview_id, output_id)

    @router.post("/data_ingestion/jobs/{job_id}/accepted_export/download")
    def download_accepted_export(job_id: str, payload: dict = Body(...)):  # noqa: B008
        return download_accepted_export_fn(job_id, payload or {})

    @router.post("/data_ingestion/jobs/{job_id}/cancel")
    def cancel_data_ingestion_job(job_id: str):
        return cancel_job_fn(job_id)

    return router
