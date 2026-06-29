"""APIRouter for dataset registry endpoints."""

import json
import re
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


def _instruction_export_nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    if parsed < 0:
        return None
    return parsed


def _instruction_export_normalized_image_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    raw = re.sub(r"/+", "/", raw)
    while raw.startswith("./"):
        raw = raw[2:]
    return "/".join(part for part in raw.split("/") if part and part != ".")


def _instruction_export_normalized_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _instruction_export_review_decision(value: Any) -> str:
    decision = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    if decision in {"accept", "accepted", "approve", "approved", "keep", "kept", "pass", "passed"}:
        return "accepted"
    if decision in {"reject", "rejected", "deny", "denied", "drop", "dropped", "fail", "failed"}:
        return "rejected"
    if decision in {"revise", "revised", "needs_revision", "needs_review", "needs_rewrite", "edit", "edited"}:
        return "needs_revision"
    return decision or "unreviewed"


def _instruction_export_training_rows_are_valid(rows: list[Any]) -> bool:
    image_question_pairs: set[tuple[str, str]] = set()
    trainable_validation_statuses = {"accepted", "machine_validated"}
    nontrainable_validation_statuses = {"rejected", "failed", "invalid"}
    trainable_review_statuses = {"accepted", "unreviewed", "machine_validated"}
    nontrainable_review_statuses = {"rejected", "needs_revision"}
    for row in rows:
        if not isinstance(row, dict):
            return False
        image_path = str(row.get("image_path") or "").strip()
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        metadata = row.get("metadata")
        if not image_path or not question or not answer or not isinstance(metadata, dict):
            return False
        qa_id = str(metadata.get("qa_id") or "").strip()
        row_type = str(metadata.get("row_type") or "").strip()
        answer_source = str(metadata.get("answer_source") or "").strip()
        source_archive = str(metadata.get("source_archive") or "").strip()
        answer_format = str(metadata.get("answer_format") or "").strip().lower()
        validation_status = str(metadata.get("validation_status") or "").strip().lower()
        raw_review_values = [
            value for value in (metadata.get("review_status"), metadata.get("review_decision"))
            if str(value or "").strip()
        ]
        review_statuses = [_instruction_export_review_decision(value) for value in raw_review_values]
        if not qa_id or not row_type or not answer_source or not answer_format:
            return False
        if source_archive != "tator_caption_instruction_archive_v1":
            return False
        if not validation_status:
            return False
        if validation_status in nontrainable_validation_statuses:
            return False
        if validation_status not in trainable_validation_statuses:
            return False
        if not raw_review_values:
            return False
        if any(status in nontrainable_review_statuses for status in review_statuses):
            return False
        if any(status not in trainable_review_statuses for status in review_statuses):
            return False
        if row_type.startswith("deterministic_") or answer_format == "json" or answer_format.endswith("_json"):
            try:
                json.loads(answer)
            except Exception:
                return False
        image_key = _instruction_export_normalized_image_path(image_path) or image_path
        question_key = _instruction_export_normalized_question(question)
        pair_key = (image_key, question_key)
        if pair_key in image_question_pairs:
            return False
        image_question_pairs.add(pair_key)
    return True


def _instruction_export_archive_rows_are_valid(rows: list[Any]) -> bool:
    image_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        image_path = str(row.get("image_path") or "").strip()
        if not image_path:
            return False
        image_key = _instruction_export_normalized_image_path(image_path) or image_path
        if image_key in image_paths:
            return False
        image_paths.add(image_key)
        if not isinstance(row.get("source_annotations"), dict):
            return False
        if not isinstance(row.get("language_annotations"), dict):
            return False
        if not isinstance(row.get("deterministic_metadata_qa_pairs"), list):
            return False
        if not isinstance(row.get("export_metadata"), dict):
            return False
    return True


def _instruction_export_review_rows_are_valid(rows: list[Any]) -> bool:
    image_qa_ids: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        if str(row.get("format") or "").strip() != "tator_caption_instruction_review_rows_v1":
            return False
        image_path = str(row.get("image_path") or row.get("image_name") or row.get("image") or "").strip()
        split = str(row.get("split") or "").strip()
        if split:
            normalized_image_path = _instruction_export_normalized_image_path(f"{split}/{image_path}")
        else:
            normalized_image_path = _instruction_export_normalized_image_path(image_path)
        image_key = normalized_image_path or image_path
        qa_id = str(row.get("qa_id") or "").strip()
        row_origin = str(row.get("row_origin") or "").strip()
        question = str(row.get("question") or "").strip()
        candidate_answer = str(row.get("candidate_answer") or "").strip()
        training_answer = str(row.get("training_answer") or "").strip()
        validation_status = str(row.get("validation_status") or "").strip()
        if not image_path or not qa_id or not row_origin or not question or not candidate_answer or not validation_status:
            return False
        if not isinstance(row.get("selected_for_training"), bool):
            return False
        if not isinstance(row.get("requires_manual_review"), bool):
            return False
        if row.get("selected_for_training") is True and not training_answer:
            return False
        if not isinstance(row.get("source_summary"), dict):
            return False
        if not isinstance(row.get("rejection_reasons"), list):
            return False
        if "review_decision" not in row or "review_notes" not in row:
            return False
        review_decision = _instruction_export_review_decision(row.get("review_decision"))
        raw_review_decision = str(row.get("review_decision") or "").strip()
        if raw_review_decision and review_decision not in {"accepted", "rejected", "needs_revision", "unreviewed"}:
            return False
        pair_key = (image_key, qa_id)
        if pair_key in image_qa_ids:
            return False
        image_qa_ids.add(pair_key)
    return True


def _instruction_export_not_ready_reason(result: Any) -> str:
    if not isinstance(result, dict):
        return "unknown"
    report = result.get("instruction_report") if isinstance(result.get("instruction_report"), dict) else {}
    if str(report.get("format") or "").strip() != "tator_caption_instruction_report_v1":
        return "instruction_report"
    readiness = report.get("training_readiness") if isinstance(report.get("training_readiness"), dict) else {}
    readiness_status = str(readiness.get("status") or "unknown").strip() or "unknown"
    if readiness_status != "ready":
        return readiness_status
    if readiness.get("ready_for_training") is not True:
        return "training_readiness"
    for field_name in ("blocking_reasons", "required_actions", "quality_warnings"):
        values = readiness.get(field_name)
        if not isinstance(values, list) or values:
            return "training_readiness"

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

    training_rows = result.get("instruction_training_rows")
    archive_rows = result.get("instruction_archive_rows")
    review_rows = result.get("instruction_review_rows")
    if not isinstance(training_rows, list):
        return "instruction_training_rows"
    if not isinstance(archive_rows, list):
        return "instruction_archive_rows"
    if not isinstance(review_rows, list):
        return "instruction_review_rows"
    if not _instruction_export_training_rows_are_valid(training_rows):
        return "instruction_training_rows"
    if not _instruction_export_archive_rows_are_valid(archive_rows):
        return "instruction_archive_rows"
    if not _instruction_export_review_rows_are_valid(review_rows):
        return "instruction_review_rows"

    consistency_counts = payload_consistency.get("counts") if isinstance(payload_consistency.get("counts"), dict) else {}
    report_selected_count = _instruction_export_nonnegative_int(report.get("selected_flattened_row_count"))
    if report_selected_count is None:
        return "instruction_report"
    report_image_count = _instruction_export_nonnegative_int(report.get("image_count"))
    if report_image_count is None:
        return "instruction_report"
    report_review_count = _instruction_export_nonnegative_int(report.get("instruction_review_row_count"))
    if report_review_count is None:
        return "instruction_report"
    report_manual_review_count = _instruction_export_nonnegative_int(report.get("manual_review_required_count"))
    if report_manual_review_count is None:
        return "instruction_report"
    metrics = report.get("corpus_quality_metrics") if isinstance(report.get("corpus_quality_metrics"), dict) else {}
    if not metrics:
        return "corpus_quality_metrics"
    metrics_selected_count = _instruction_export_nonnegative_int(metrics.get("selected_flattened_row_count"))
    metrics_image_count = _instruction_export_nonnegative_int(metrics.get("image_count"))
    if metrics_selected_count is None or metrics_image_count is None:
        return "corpus_quality_metrics"
    if metrics_selected_count != report_selected_count or metrics_image_count != report_image_count:
        return "corpus_quality_metrics"

    training_row_count = len(training_rows)
    expected_training_count = _instruction_export_nonnegative_int(payload_export_validation.get("row_count"))
    if expected_training_count is None:
        return "instruction_export_validation"
    if training_row_count != expected_training_count:
        return "instruction_training_rows"
    if training_row_count != report_selected_count:
        return "instruction_training_rows"
    consistency_training_count = _instruction_export_nonnegative_int(consistency_counts.get("training_row_count"))
    if consistency_training_count is not None and training_row_count != consistency_training_count:
        return "instruction_training_rows"
    consistency_report_selected_count = _instruction_export_nonnegative_int(
        consistency_counts.get("report_selected_flattened_row_count")
    )
    if consistency_report_selected_count is not None and training_row_count != consistency_report_selected_count:
        return "instruction_training_rows"
    consistency_export_row_count = _instruction_export_nonnegative_int(
        consistency_counts.get("instruction_export_validation_row_count")
    )
    if consistency_export_row_count is not None and training_row_count != consistency_export_row_count:
        return "instruction_training_rows"
    readiness_training_count = _instruction_export_nonnegative_int(readiness.get("selected_training_row_count"))
    if readiness_training_count is not None and training_row_count != readiness_training_count:
        return "training_readiness"

    archive_row_count = len(archive_rows)
    if archive_row_count != report_image_count:
        return "instruction_archive_rows"
    consistency_archive_count = _instruction_export_nonnegative_int(consistency_counts.get("archive_row_count"))
    if consistency_archive_count is not None and archive_row_count != consistency_archive_count:
        return "instruction_archive_rows"
    consistency_report_image_count = _instruction_export_nonnegative_int(consistency_counts.get("report_image_count"))
    if consistency_report_image_count is not None and archive_row_count != consistency_report_image_count:
        return "instruction_archive_rows"

    review_row_count = len(review_rows)
    if review_row_count != report_review_count:
        return "instruction_review_rows"
    consistency_review_count = _instruction_export_nonnegative_int(consistency_counts.get("review_row_count"))
    if consistency_review_count is not None and review_row_count != consistency_review_count:
        return "instruction_review_rows"
    consistency_report_review_count = _instruction_export_nonnegative_int(
        consistency_counts.get("report_instruction_review_row_count")
    )
    if consistency_report_review_count is not None and review_row_count != consistency_report_review_count:
        return "instruction_review_rows"
    selected_review_row_count = sum(
        1 for row in review_rows if isinstance(row, dict) and row.get("selected_for_training") is True
    )
    if selected_review_row_count != training_row_count:
        return "instruction_review_rows"
    consistency_selected_review_count = _instruction_export_nonnegative_int(
        consistency_counts.get("selected_review_row_count")
    )
    if consistency_selected_review_count is not None and selected_review_row_count != consistency_selected_review_count:
        return "instruction_review_rows"
    readiness_selected_review_count = _instruction_export_nonnegative_int(readiness.get("selected_review_row_count"))
    if readiness_selected_review_count is not None and selected_review_row_count != readiness_selected_review_count:
        return "training_readiness"
    manual_review_count = sum(
        1 for row in review_rows if isinstance(row, dict) and row.get("requires_manual_review") is True
    )
    if manual_review_count != report_manual_review_count:
        return "instruction_review_rows"
    consistency_manual_review_count = _instruction_export_nonnegative_int(
        consistency_counts.get("manual_review_required_count")
    )
    if consistency_manual_review_count is not None and manual_review_count != consistency_manual_review_count:
        return "instruction_review_rows"
    consistency_report_manual_review_count = _instruction_export_nonnegative_int(
        consistency_counts.get("report_manual_review_required_count")
    )
    if consistency_report_manual_review_count is not None and manual_review_count != consistency_report_manual_review_count:
        return "instruction_review_rows"
    selected_manual_review_count = sum(
        1
        for row in review_rows
        if isinstance(row, dict)
        and row.get("selected_for_training") is True
        and row.get("requires_manual_review") is True
    )
    readiness_selected_manual_count = _instruction_export_nonnegative_int(
        readiness.get("selected_manual_review_row_count")
    )
    if readiness_selected_manual_count is not None and selected_manual_review_count != readiness_selected_manual_count:
        return "training_readiness"
    for readiness_count_name in (
        "pending_manual_review_row_count",
        "rejected_manual_review_row_count",
        "needs_revision_manual_review_row_count",
    ):
        readiness_count = _instruction_export_nonnegative_int(readiness.get(readiness_count_name))
        if readiness_count is not None and readiness_count != 0:
            return "training_readiness"
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
                "block_active_caption_jobs": True,
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
