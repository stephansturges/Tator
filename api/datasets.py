"""APIRouter for dataset registry endpoints."""

from typing import Any, Callable, Optional

from fastapi import APIRouter, UploadFile, File, Form, Body, HTTPException, Query


def _instruction_export_proof_is_ok(value: Any, *, expected_format: Optional[str] = None) -> bool:
    if not isinstance(value, dict):
        return False
    if expected_format is not None and str(value.get("format") or "").strip() != expected_format:
        return False
    if value.get("ok") is not True:
        return False
    error_count = value.get("error_count")
    if isinstance(error_count, bool):
        return False
    try:
        parsed_error_count = int(error_count)
    except (TypeError, ValueError, OverflowError):
        return False
    if parsed_error_count != 0:
        return False
    errors = value.get("errors")
    return isinstance(errors, list) and not errors


def _instruction_export_not_ready_reason(result: Any) -> str:
    if not isinstance(result, dict):
        return "unknown"
    report = result.get("instruction_report") if isinstance(result.get("instruction_report"), dict) else {}
    readiness = report.get("training_readiness") if isinstance(report.get("training_readiness"), dict) else {}
    readiness_status = str(readiness.get("status") or "unknown").strip() or "unknown"
    if readiness_status != "ready":
        return readiness_status

    payload_export_validation = result.get("instruction_export_validation")
    report_export_validation = report.get("instruction_export_validation")
    if not _instruction_export_proof_is_ok(payload_export_validation):
        return "instruction_export_validation"
    if not _instruction_export_proof_is_ok(report_export_validation):
        return "instruction_export_validation"
    if payload_export_validation != report_export_validation:
        return "instruction_export_validation_mismatch"

    payload_consistency = result.get("instruction_artifact_consistency")
    report_consistency = report.get("instruction_artifact_consistency")
    archive = result.get("instruction_archive") if isinstance(result.get("instruction_archive"), dict) else {}
    archive_consistency = archive.get("instruction_artifact_consistency")
    consistency_format = "tator_caption_instruction_artifact_consistency_v1"
    if not _instruction_export_proof_is_ok(payload_consistency, expected_format=consistency_format):
        return "instruction_artifact_consistency"
    if not _instruction_export_proof_is_ok(report_consistency, expected_format=consistency_format):
        return "instruction_artifact_consistency"
    if not _instruction_export_proof_is_ok(archive_consistency, expected_format=consistency_format):
        return "instruction_artifact_consistency"
    if payload_consistency != report_consistency or payload_consistency != archive_consistency:
        return "instruction_artifact_consistency_mismatch"
    return ""


def build_datasets_router(
    *,
    list_fn: Callable[[], Any],
    list_trash_fn: Callable[[], Any],
    upload_fn: Callable[[UploadFile, Optional[str], Optional[str], Optional[str]], Any],
    upload_session_list_fn: Callable[[], Any],
    upload_session_get_fn: Callable[[str], Any],
    upload_session_start_fn: Callable[[dict], Any],
    upload_session_batch_fn: Callable[[str, str, list[UploadFile]], Any],
    upload_session_finalize_fn: Callable[[str], Any],
    upload_session_cancel_fn: Callable[[str], Any],
    delete_fn: Callable[[str], Any],
    restore_fn: Callable[[str, Optional[str]], Any],
    download_fn: Callable[[str], Any],
    build_qwen_fn: Callable[[str], Any],
    check_fn: Callable[[str], Any],
    get_glossary_fn: Callable[[str], Any],
    set_glossary_fn: Callable[[str, str], Any],
    get_text_label_fn: Callable[[str, str], Any],
    get_text_labels_fn: Callable[[str, list[str]], Any],
    set_text_label_fn: Callable[[str, str, dict], Any],
    get_captions_fn: Callable[[str, str], Any],
    get_captions_batch_fn: Callable[[str, list[str]], Any],
    add_caption_fn: Callable[[str, str, dict], Any],
    update_caption_fn: Callable[[str, str, dict], Any],
    delete_caption_fn: Callable[[str, str, dict], Any],
    export_captions_fn: Callable[..., Any],
    apply_caption_instruction_review_fn: Callable[[str, Any], Any],
    register_path_fn: Callable[
        [
            str,
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[bool],
            Optional[bool],
        ],
        Any,
    ],
    open_path_fn: Callable[[str, Optional[bool]], Any],
    save_transient_fn: Callable[
        [str, Optional[str], Optional[str], Optional[str], Optional[str]], Any
    ],
    delete_transient_fn: Callable[[str], Any],
    annotation_session_start_fn: Callable[[str, dict], Any],
    annotation_session_heartbeat_fn: Callable[[str, dict], Any],
    annotation_session_stop_fn: Callable[[str, dict], Any],
    annotation_manifest_fn: Callable[[str], Any],
    annotation_image_fn: Callable[[str, str, str], Any],
    annotation_snapshot_fn: Callable[[str, dict], Any],
    annotation_meta_patch_fn: Callable[[str, dict], Any],
    transient_annotation_manifest_fn: Callable[[str], Any],
    transient_annotation_image_fn: Callable[[str, str, str], Any],
    transient_annotation_snapshot_fn: Callable[[str, dict], Any],
    transient_annotation_meta_patch_fn: Callable[[str, dict], Any],
    transient_annotation_session_start_fn: Callable[[str, dict], Any],
    transient_annotation_session_heartbeat_fn: Callable[[str, dict], Any],
    transient_annotation_session_stop_fn: Callable[[str, dict], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/datasets")
    def list_datasets():
        return list_fn()

    @router.post("/datasets/upload")
    def upload_dataset(
        file: UploadFile = File(...),  # noqa: B008
        dataset_id: Optional[str] = Form(None),
        dataset_type: Optional[str] = Form(None),
        context: Optional[str] = Form(None),
    ):
        return upload_fn(file, dataset_id, dataset_type, context)

    @router.get("/datasets/upload_sessions")
    def list_dataset_upload_sessions():
        return upload_session_list_fn()

    @router.get("/datasets/upload_session/{session_id}")
    def get_dataset_upload_session(session_id: str):
        return upload_session_get_fn(session_id)

    @router.post("/datasets/upload_session/start")
    def start_dataset_upload_session(payload: dict = Body(...)):  # noqa: B008
        return upload_session_start_fn(payload or {})

    @router.post("/datasets/upload_session/{session_id}/batch")
    def upload_dataset_session_batch(
        session_id: str,
        manifest: str = Form(...),  # noqa: B008
        files: list[UploadFile] = File(...),  # noqa: B008
    ):
        return upload_session_batch_fn(session_id, manifest, files)

    @router.post("/datasets/upload_session/{session_id}/finalize")
    def finalize_dataset_upload_session(session_id: str):
        return upload_session_finalize_fn(session_id)

    @router.post("/datasets/upload_session/{session_id}/cancel")
    def cancel_dataset_upload_session(session_id: str):
        return upload_session_cancel_fn(session_id)

    @router.get("/datasets/trash")
    def list_dataset_trash():
        return list_trash_fn()

    @router.post("/datasets/trash/{trash_id}/restore")
    def restore_dataset(trash_id: str, payload: dict = Body(None)):  # noqa: B008
        restore_id = (payload or {}).get("dataset_id")
        return restore_fn(trash_id, restore_id)

    @router.delete("/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str):
        return delete_fn(dataset_id)

    @router.get("/datasets/{dataset_id}/download")
    def download_dataset(dataset_id: str):
        return download_fn(dataset_id)

    @router.post("/datasets/{dataset_id}/build/qwen")
    def build_qwen_dataset(dataset_id: str):
        return build_qwen_fn(dataset_id)

    @router.get("/datasets/{dataset_id}/check")
    def check_dataset(dataset_id: str):
        return check_fn(dataset_id)

    @router.get("/datasets/{dataset_id}/glossary")
    def get_dataset_glossary(dataset_id: str):
        return get_glossary_fn(dataset_id)

    @router.post("/datasets/{dataset_id}/glossary")
    def set_dataset_glossary(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        glossary = str((payload or {}).get("glossary") or "")
        return set_glossary_fn(dataset_id, glossary)

    @router.post("/datasets/{dataset_id}/text_labels/batch")
    def get_text_labels(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        raw_names = (payload or {}).get("image_names") or []
        if not isinstance(raw_names, list):
            raw_names = []
        image_names = [str(name or "").strip() for name in raw_names if str(name or "").strip()]
        return get_text_labels_fn(dataset_id, image_names)

    @router.get("/datasets/{dataset_id}/text_labels/{image_name:path}")
    def get_text_label(dataset_id: str, image_name: str):
        return get_text_label_fn(dataset_id, image_name)

    @router.post("/datasets/{dataset_id}/text_labels/{image_name:path}")
    def set_text_label(dataset_id: str, image_name: str, payload: dict = Body(...)):  # noqa: B008
        return set_text_label_fn(dataset_id, image_name, payload or {})

    @router.get("/datasets/{dataset_id}/captions/export")
    def export_captions(
        dataset_id: str,
        include_caption0_in_training: bool = Query(True),
        include_generated_qa_in_training: bool = Query(True),
        include_deterministic_metadata_qa: bool = Query(False),
        qa_mix: str = Query("balanced"),
        answer_format: str = Query("natural"),
        require_ready_instruction_export: bool = Query(False),
    ):
        result = export_captions_fn(
            dataset_id,
            {
                "include_caption0_in_training": include_caption0_in_training,
                "include_generated_qa_in_training": include_generated_qa_in_training,
                "include_deterministic_metadata_qa": include_deterministic_metadata_qa,
                "qa_mix": qa_mix,
                "answer_format": answer_format,
            },
        )
        not_ready_reason = _instruction_export_not_ready_reason(result)
        if require_ready_instruction_export and not_ready_reason:
            raise HTTPException(
                status_code=409,
                detail=f"instruction_export_not_ready:{not_ready_reason}",
            )
        return result

    @router.post("/datasets/{dataset_id}/captions/instruction_review")
    def apply_caption_instruction_review(dataset_id: str, payload: Any = Body(...)):  # noqa: B008
        return apply_caption_instruction_review_fn(dataset_id, payload)

    @router.post("/datasets/{dataset_id}/captions/batch")
    def get_captions_batch(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        raw_names = (payload or {}).get("image_names") or []
        if not isinstance(raw_names, list):
            raw_names = []
        image_names = [str(name or "").strip() for name in raw_names if str(name or "").strip()]
        return get_captions_batch_fn(dataset_id, image_names)

    @router.patch("/datasets/{dataset_id}/captions/by_id/{caption_id}")
    def update_caption(dataset_id: str, caption_id: str, payload: dict = Body(...)):  # noqa: B008
        return update_caption_fn(dataset_id, caption_id, payload or {})

    @router.delete("/datasets/{dataset_id}/captions/by_id/{caption_id}")
    def delete_caption(dataset_id: str, caption_id: str, session_id: str = Query("")):
        payload = {"session_id": session_id} if str(session_id or "").strip() else {}
        return delete_caption_fn(dataset_id, caption_id, payload)

    @router.get("/datasets/{dataset_id}/captions/{image_name:path}")
    def get_captions(dataset_id: str, image_name: str):
        return get_captions_fn(dataset_id, image_name)

    @router.post("/datasets/{dataset_id}/captions/{image_name:path}")
    def add_caption(dataset_id: str, image_name: str, payload: dict = Body(...)):  # noqa: B008
        return add_caption_fn(dataset_id, image_name, payload or {})

    @router.post("/datasets/register_path")
    def register_dataset_path(payload: dict = Body(...)):  # noqa: B008
        return register_path_fn(
            str((payload or {}).get("path") or ""),
            (payload or {}).get("dataset_id"),
            (payload or {}).get("label"),
            (payload or {}).get("context"),
            (payload or {}).get("notes"),
            (payload or {}).get("force_new"),
            (payload or {}).get("strict"),
        )

    @router.post("/datasets/open_path")
    def open_dataset_path(payload: dict = Body(...)):  # noqa: B008
        return open_path_fn(
            str((payload or {}).get("path") or ""),
            (payload or {}).get("strict"),
        )

    @router.post("/datasets/transient/{session_id}/save")
    def save_transient_dataset(session_id: str, payload: dict = Body(...)):  # noqa: B008
        return save_transient_fn(
            session_id,
            (payload or {}).get("dataset_id"),
            (payload or {}).get("label"),
            (payload or {}).get("context"),
            (payload or {}).get("notes"),
        )

    @router.delete("/datasets/transient/{session_id}")
    def delete_transient_dataset(session_id: str):
        return delete_transient_fn(session_id)

    @router.post("/datasets/{dataset_id}/annotation/session/start")
    def start_annotation_session(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        return annotation_session_start_fn(dataset_id, payload or {})

    @router.post("/datasets/{dataset_id}/annotation/session/heartbeat")
    def heartbeat_annotation_session(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        return annotation_session_heartbeat_fn(dataset_id, payload or {})

    @router.post("/datasets/{dataset_id}/annotation/session/stop")
    def stop_annotation_session(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        return annotation_session_stop_fn(dataset_id, payload or {})

    @router.get("/datasets/{dataset_id}/annotation/manifest")
    def get_annotation_manifest(dataset_id: str):
        return annotation_manifest_fn(dataset_id)

    @router.get("/datasets/{dataset_id}/annotation/image")
    def get_annotation_image(
        dataset_id: str,
        split: str = Query(...),  # noqa: B008
        image_relpath: str = Query(...),  # noqa: B008
    ):
        return annotation_image_fn(dataset_id, split, image_relpath)

    @router.post("/datasets/{dataset_id}/annotation/snapshot")
    def save_annotation_snapshot(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        return annotation_snapshot_fn(dataset_id, payload or {})

    @router.patch("/datasets/{dataset_id}/annotation/meta")
    def patch_annotation_meta(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        return annotation_meta_patch_fn(dataset_id, payload or {})

    @router.post("/datasets/transient/{session_id}/annotation/session/start")
    def start_transient_annotation_session(
        session_id: str, payload: dict = Body(...)
    ):  # noqa: B008
        return transient_annotation_session_start_fn(session_id, payload or {})

    @router.post("/datasets/transient/{session_id}/annotation/session/heartbeat")
    def heartbeat_transient_annotation_session(
        session_id: str, payload: dict = Body(...)
    ):  # noqa: B008
        return transient_annotation_session_heartbeat_fn(session_id, payload or {})

    @router.post("/datasets/transient/{session_id}/annotation/session/stop")
    def stop_transient_annotation_session(session_id: str, payload: dict = Body(...)):  # noqa: B008
        return transient_annotation_session_stop_fn(session_id, payload or {})

    @router.get("/datasets/transient/{session_id}/annotation/manifest")
    def get_transient_annotation_manifest(session_id: str):
        return transient_annotation_manifest_fn(session_id)

    @router.get("/datasets/transient/{session_id}/annotation/image")
    def get_transient_annotation_image(
        session_id: str,
        split: str = Query(...),  # noqa: B008
        image_relpath: str = Query(...),  # noqa: B008
    ):
        return transient_annotation_image_fn(session_id, split, image_relpath)

    @router.post("/datasets/transient/{session_id}/annotation/snapshot")
    def save_transient_annotation_snapshot(
        session_id: str, payload: dict = Body(...)
    ):  # noqa: B008
        return transient_annotation_snapshot_fn(session_id, payload or {})

    @router.patch("/datasets/transient/{session_id}/annotation/meta")
    def patch_transient_annotation_meta(session_id: str, payload: dict = Body(...)):  # noqa: B008
        return transient_annotation_meta_patch_fn(session_id, payload or {})

    return router
