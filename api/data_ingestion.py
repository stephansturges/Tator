"""APIRouter for data ingestion and local SALAD diversity jobs."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request


def build_data_ingestion_router(
    *,
    capabilities_fn: Callable[[], Any],
    create_analysis_job_fn: Callable[[str, list[Any], list[Any]], Any],
    create_salad_train_job_fn: Callable[[str, list[Any]], Any],
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

    @router.get("/data_ingestion/jobs/{job_id}")
    def get_data_ingestion_job(job_id: str):
        return get_job_fn(job_id)

    @router.get("/data_ingestion/jobs/{job_id}/result")
    def get_data_ingestion_result(job_id: str):
        return get_result_fn(job_id)

    @router.post("/data_ingestion/jobs/{job_id}/cancel")
    def cancel_data_ingestion_job(job_id: str):
        return cancel_job_fn(job_id)

    return router

