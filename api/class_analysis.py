"""APIRouter for class embedding analysis jobs."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from fastapi import APIRouter, Body, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse


def build_class_analysis_router(
    *,
    capabilities_fn: Callable[[], Any],
    create_job_fn: Callable[[dict], Any],
    create_active_workspace_job_fn: Callable[[str, list[UploadFile]], Any],
    start_active_workspace_upload_fn: Callable[[dict], Any],
    batch_active_workspace_upload_fn: Callable[[str, str, list[UploadFile]], Any],
    finalize_active_workspace_upload_fn: Callable[[str], Any],
    create_active_workspace_snapshot_job_fn: Callable[[str, dict], Any],
    cancel_active_workspace_upload_fn: Callable[[str], Any],
    get_job_fn: Callable[[str], Any],
    get_result_fn: Callable[[str], Any],
    get_projection_fn: Callable[[str, str], Any],
    get_thumbnail_fn: Callable[..., FileResponse],
    record_review_disposition_fn: Callable[[str, str, dict], Any],
    authorize_training_capture_request_fn: Callable[[Request], Any],
    record_training_action_fn: Callable[[str, dict], Any],
    training_action_status_fn: Callable[[], Any],
    export_training_actions_fn: Callable[[], FileResponse],
    create_cluster_search_fn: Callable[[str, dict], Any],
    get_cluster_search_fn: Callable[[str], Any],
    cancel_cluster_search_fn: Callable[[str], Any],
    cancel_job_fn: Callable[[str], Any],
    create_qwen_review_fn: Callable[[str, str, dict], Any],
    list_qwen_reviews_fn: Callable[[str], Any],
    get_qwen_review_fn: Callable[[str], Any],
    cancel_qwen_review_fn: Callable[[str], Any],
    get_qwen_review_evidence_fn: Callable[[str, str], FileResponse],
    get_refinement_preview_fn: Optional[Callable[[str, str], Any]] = None,
    commit_dual_bbox_annotation_transaction_fn: Optional[
        Callable[[str, str, dict], Any]
    ] = None,
    delete_review_history_fn: Optional[Callable[[str, dict], Any]] = None,
    cache_status_fn: Optional[Callable[[], Any]] = None,
    purge_cache_fn: Optional[Callable[[dict], Any]] = None,
    set_cache_budget_fn: Optional[Callable[[dict], Any]] = None,
    recalibrate_review_ranking_fn: Optional[Callable[[str, dict], Any]] = None,
    reset_review_ranking_fn: Optional[Callable[[str, dict], Any]] = None,
    latest_session_fn: Optional[Callable[[], Any]] = None,
    open_session_annotation_source_fn: Optional[Callable[[str], Any]] = None,
    get_session_manifest_fn: Optional[Callable[[str], Any]] = None,
    get_session_graph_fn: Optional[Callable[..., Any]] = None,
    get_session_review_queue_fn: Optional[Callable[..., Any]] = None,
    get_session_review_history_fn: Optional[Callable[..., Any]] = None,
    get_session_point_detail_fn: Optional[Callable[[str, str], Any]] = None,
    get_session_point_evidence_fn: Optional[Callable[[str, str], Any]] = None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/class_analysis/capabilities")
    def class_analysis_capabilities():
        return capabilities_fn()

    if cache_status_fn is not None:

        @router.get("/class_analysis/cache")
        def get_class_analysis_cache_status():
            return cache_status_fn()

    if purge_cache_fn is not None:

        @router.post("/class_analysis/cache/purge")
        def purge_class_analysis_cache(
            payload: dict = Body(default_factory=dict),  # noqa: B008
        ):
            return purge_cache_fn(payload or {})

    if set_cache_budget_fn is not None:

        @router.post("/class_analysis/cache/budget")
        def set_class_analysis_cache_budget(
            payload: dict = Body(...),  # noqa: B008
        ):
            return set_cache_budget_fn(payload or {})

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

    @router.post("/class_analysis/jobs/active_workspace/upload_session/start")
    def start_active_workspace_upload_session(payload: dict = Body(default_factory=dict)):  # noqa: B008
        return start_active_workspace_upload_fn(payload or {})

    @router.post("/class_analysis/jobs/active_workspace/upload_session/{session_id}/batch")
    async def batch_active_workspace_upload_session(session_id: str, request: Request):
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
        result = batch_active_workspace_upload_fn(session_id, str(manifest), files)
        if inspect.isawaitable(result):
            return await result
        return result

    @router.post("/class_analysis/jobs/active_workspace/upload_session/{session_id}/finalize")
    def finalize_active_workspace_upload_session(session_id: str):
        return finalize_active_workspace_upload_fn(session_id)

    @router.post("/class_analysis/jobs/active_workspace/snapshots/{snapshot_id}")
    def create_active_workspace_snapshot_job(snapshot_id: str, payload: dict = Body(...)):  # noqa: B008
        return create_active_workspace_snapshot_job_fn(snapshot_id, payload or {})

    @router.post("/class_analysis/jobs/active_workspace/upload_session/{session_id}/cancel")
    def cancel_active_workspace_upload_session(session_id: str):
        return cancel_active_workspace_upload_fn(session_id)

    if latest_session_fn is not None:

        @router.get("/class_analysis/sessions/latest")
        def get_latest_class_analysis_session():
            return latest_session_fn()

    if open_session_annotation_source_fn is not None:

        @router.post("/class_analysis/jobs/{job_id}/annotation_session")
        def open_class_analysis_session_annotation_source(job_id: str):
            return open_session_annotation_source_fn(job_id)

    if get_session_manifest_fn is not None:

        @router.get("/class_analysis/jobs/{job_id}/manifest")
        def get_class_analysis_session_manifest(job_id: str):
            return get_session_manifest_fn(job_id)

    @router.get("/class_analysis/jobs/{job_id}")
    def get_class_analysis_job(job_id: str):
        return get_job_fn(job_id)

    @router.get("/class_analysis/jobs/{job_id}/result")
    def get_class_analysis_result(job_id: str):
        return get_result_fn(job_id)

    if recalibrate_review_ranking_fn is not None:

        @router.post("/class_analysis/jobs/{job_id}/review-ranking/recalibrate")
        def recalibrate_class_analysis_review_ranking(
            job_id: str,
            payload: dict = Body(default_factory=dict),  # noqa: B008
        ):
            return recalibrate_review_ranking_fn(job_id, payload or {})

    if reset_review_ranking_fn is not None:

        @router.delete("/class_analysis/jobs/{job_id}/review-ranking")
        def reset_class_analysis_review_ranking(
            job_id: str,
            payload: dict = Body(default_factory=dict),  # noqa: B008
        ):
            return reset_review_ranking_fn(job_id, payload or {})

    @router.get("/class_analysis/jobs/{job_id}/projection/{mode}")
    def get_class_analysis_projection(job_id: str, mode: str):
        return get_projection_fn(job_id, mode)

    @router.get("/class_analysis/jobs/{job_id}/thumbnail/{point_id}")
    def get_class_analysis_thumbnail(
        job_id: str,
        point_id: str,
        context: Optional[str] = None,
    ):
        # Preserve the legacy two-argument callback contract for ordinary
        # thumbnails; only pass the optional mode when the caller requests it.
        if context is None or not str(context).strip():
            return get_thumbnail_fn(job_id, point_id)
        return get_thumbnail_fn(job_id, point_id, context)

    if get_refinement_preview_fn is not None:

        @router.get(
            "/class_analysis/jobs/{job_id}/points/{point_id}/refinement_preview"
        )
        def get_class_analysis_refinement_preview(
            job_id: str,
            point_id: str,
        ):
            return get_refinement_preview_fn(job_id, point_id)

    if commit_dual_bbox_annotation_transaction_fn is not None:

        @router.post(
            "/class_analysis/jobs/{job_id}/points/{point_id}/dual_bbox_annotation_transaction"
        )
        def commit_dual_bbox_annotation_transaction(
            job_id: str,
            point_id: str,
            payload: dict = Body(default_factory=dict),  # noqa: B008
        ):
            return commit_dual_bbox_annotation_transaction_fn(
                job_id,
                point_id,
                payload or {},
            )

    @router.post("/class_analysis/jobs/{job_id}/points/{point_id}/review_disposition")
    def record_class_analysis_review_disposition(
        job_id: str,
        point_id: str,
        request: Request,
        payload: dict = Body(default_factory=dict),  # noqa: B008
    ):
        clean_payload = dict(payload or {})
        denied = None
        if clean_payload.get("capture_training_data") is True:
            try:
                clean_payload["_training_authorization"] = (
                    authorize_training_capture_request_fn(request)
                )
            except HTTPException as exc:
                clean_payload["capture_training_data"] = False
                denied = {
                    "status": "denied",
                    "detail": str(
                        exc.detail or "training_capture_forbidden"
                    ),
                }
        result = record_review_disposition_fn(
            job_id,
            point_id,
            clean_payload,
        )
        if denied is not None and isinstance(result, dict):
            result = {**result, "training_capture": denied}
        return result

    if delete_review_history_fn is not None:

        @router.post("/class_analysis/jobs/{job_id}/review_history/delete")
        def delete_class_analysis_review_history(
            job_id: str,
            payload: dict = Body(default_factory=dict),  # noqa: B008
        ):
            return delete_review_history_fn(job_id, payload or {})

    @router.post("/class_analysis/jobs/{job_id}/training_actions")
    def record_class_analysis_vignette_training_action(
        job_id: str,
        request: Request,
        payload: dict = Body(default_factory=dict),  # noqa: B008
    ):
        clean_payload = dict(payload or {})
        clean_payload["_training_authorization"] = (
            authorize_training_capture_request_fn(request)
        )
        return record_training_action_fn(job_id, clean_payload)

    @router.get("/class_analysis/training_actions/status")
    def get_class_analysis_vignette_training_status(request: Request):
        authorize_training_capture_request_fn(request)
        return training_action_status_fn()

    @router.post("/class_analysis/training_actions/export")
    def export_class_analysis_vignette_training_actions(request: Request):
        authorize_training_capture_request_fn(request)
        return export_training_actions_fn()

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

    @router.post("/class_analysis/jobs/{job_id}/points/{point_id}/qwen_review")
    def create_class_analysis_qwen_review(job_id: str, point_id: str, payload: dict = Body(default_factory=dict)):  # noqa: B008
        return create_qwen_review_fn(job_id, point_id, payload or {})

    @router.get("/class_analysis/jobs/{job_id}/qwen_reviews")
    def list_class_analysis_qwen_reviews(job_id: str):
        return list_qwen_reviews_fn(job_id)

    @router.get("/class_analysis/qwen_review/{review_id}")
    def get_class_analysis_qwen_review(review_id: str):
        return get_qwen_review_fn(review_id)

    @router.post("/class_analysis/qwen_review/{review_id}/cancel")
    def cancel_class_analysis_qwen_review(review_id: str):
        return cancel_qwen_review_fn(review_id)

    @router.get("/class_analysis/qwen_review/{review_id}/evidence/{evidence_id}")
    def get_class_analysis_qwen_review_evidence(review_id: str, evidence_id: str):
        return get_qwen_review_evidence_fn(review_id, evidence_id)

    if get_session_graph_fn is not None:

        @router.get("/class_analysis/jobs/{job_id}/graph")
        def get_class_analysis_session_graph(
            job_id: str,
            projection_mode: Optional[str] = None,
            class_name: Optional[str] = None,
            objects: str = "all",
            object_size: str = "all",
            reviewed: str = "any",
            limit: int = 50_000,
            cursor: Optional[str] = None,
        ):
            return get_session_graph_fn(
                job_id,
                projection_mode=projection_mode,
                class_name=class_name,
                objects=objects,
                object_size=object_size,
                reviewed=reviewed,
                limit=limit,
                cursor=cursor,
            )

    if get_session_review_queue_fn is not None:

        @router.get("/class_analysis/jobs/{job_id}/review_queue")
        def get_class_analysis_session_review_queue(
            job_id: str,
            category: str = "review",
            cursor: Optional[str] = None,
            limit: int = 36,
        ):
            return get_session_review_queue_fn(
                job_id, category=category, cursor=cursor, limit=limit
            )

    if get_session_review_history_fn is not None:

        @router.get("/class_analysis/jobs/{job_id}/review_history")
        def get_class_analysis_session_review_history(
            job_id: str,
            projection_mode: Optional[str] = None,
            limit: int = 250,
        ):
            return get_session_review_history_fn(
                job_id,
                projection_mode=projection_mode,
                limit=limit,
            )

    if get_session_point_detail_fn is not None:

        @router.get("/class_analysis/jobs/{job_id}/points/{point_id}")
        def get_class_analysis_session_point_detail(job_id: str, point_id: str):
            return get_session_point_detail_fn(job_id, point_id)

    if get_session_point_evidence_fn is not None:

        @router.get("/class_analysis/jobs/{job_id}/points/{point_id}/evidence")
        def get_class_analysis_session_point_evidence(job_id: str, point_id: str):
            return get_session_point_evidence_fn(job_id, point_id)

    return router
