from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import localinferenceapi as api


def _dataset_export_route_client(
    export_payload,
    apply_review_fn=None,
    download_fn=None,
    set_glossary_fn=None,
    get_text_label_fn=None,
    get_text_labels_fn=None,
    get_captions_fn=None,
    get_captions_batch_fn=None,
):
    from api.datasets import build_datasets_router

    def export_captions(dataset_id, options):
        if callable(export_payload):
            return export_payload(dataset_id, options)
        return export_payload

    def noop(*_args, **_kwargs):
        return {}

    app = FastAPI()
    app.include_router(
        build_datasets_router(
            list_fn=lambda: [],
            list_trash_fn=lambda: [],
            upload_fn=noop,
            upload_session_list_fn=lambda: [],
            upload_session_get_fn=noop,
            upload_session_start_fn=noop,
            upload_session_batch_fn=noop,
            upload_session_finalize_fn=noop,
            upload_session_cancel_fn=noop,
            delete_fn=noop,
            restore_fn=noop,
            download_fn=download_fn or noop,
            build_qwen_fn=noop,
            check_fn=noop,
            get_glossary_fn=noop,
            set_glossary_fn=set_glossary_fn or noop,
            get_text_label_fn=get_text_label_fn or noop,
            get_text_labels_fn=get_text_labels_fn or noop,
            set_text_label_fn=noop,
            get_captions_fn=get_captions_fn or noop,
            get_captions_batch_fn=get_captions_batch_fn or noop,
            add_caption_fn=noop,
            update_caption_fn=noop,
            delete_caption_fn=noop,
            export_captions_fn=export_captions,
            apply_caption_instruction_review_fn=apply_review_fn or noop,
            register_path_fn=noop,
            open_path_fn=noop,
            save_transient_fn=noop,
            delete_transient_fn=noop,
            annotation_session_start_fn=noop,
            annotation_session_heartbeat_fn=noop,
            annotation_session_stop_fn=noop,
            annotation_manifest_fn=noop,
            annotation_image_fn=noop,
            annotation_snapshot_fn=noop,
            annotation_meta_patch_fn=noop,
            transient_annotation_manifest_fn=noop,
            transient_annotation_image_fn=noop,
            transient_annotation_snapshot_fn=noop,
            transient_annotation_meta_patch_fn=noop,
            transient_annotation_session_start_fn=noop,
            transient_annotation_session_heartbeat_fn=noop,
            transient_annotation_session_stop_fn=noop,
        )
    )
    return TestClient(app)


def test_caption_export_route_blocks_when_backend_caption_job_is_active() -> None:
    observed_options = {}

    def export_captions(_dataset_id, options):
        observed_options.update(options)
        raise api.HTTPException(
            status_code=409,
            detail="caption_export_busy:qcap_busy:running",
        )

    client = _dataset_export_route_client(export_captions)

    response = client.get("/datasets/ds/captions/export")

    assert response.status_code == 409
    assert response.json()["detail"] == "caption_export_busy:qcap_busy:running"
    assert observed_options["block_active_caption_jobs"] is True


def test_export_captions_blocks_active_backend_caption_job_before_dataset_read(monkeypatch) -> None:
    job = api.QwenCaptionDatasetJob(job_id="qcap_busy", status="running")
    job.request = {"dataset_id": "ds"}
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {job.job_id: job})

    def fail_resolve(_dataset_id):
        raise AssertionError("export should block before reading a mutating dataset")

    monkeypatch.setattr(api, "_resolve_dataset_entry", fail_resolve)

    with pytest.raises(api.HTTPException) as exc_info:
        api.export_captions("ds", {"block_active_caption_jobs": True})

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "caption_export_busy:qcap_busy:running"
    assert api._qwen_caption_dataset_active_export_job("other") is None


def test_caption_export_blocks_live_persisted_runner_before_dataset_read(
    tmp_path,
    monkeypatch,
) -> None:
    jobs_root = tmp_path / "jobs"
    artifact_dir = tmp_path / "artifact"
    job_dir = jobs_root / "qcap_persisted"
    job_dir.mkdir(parents=True)
    artifact_dir.mkdir()
    (artifact_dir / ".runner.lock").write_text(json.dumps({"pid": 12345}), encoding="utf-8")
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "qcap_persisted",
                "status": "running",
                "request": {"dataset_id": "ds", "output_dir": str(artifact_dir)},
                "output_dir": str(artifact_dir),
                "created_at": 10.0,
                "updated_at": 20.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)
    monkeypatch.setattr(api, "_qwen_caption_dataset_pid_is_alive", lambda _pid: True)
    with api.QWEN_CAPTION_DATASET_JOBS_LOCK:
        original_jobs = dict(api.QWEN_CAPTION_DATASET_JOBS)
        api.QWEN_CAPTION_DATASET_JOBS.clear()

    def fail_resolve(_dataset_id):
        raise AssertionError("persisted live runner should block before dataset reads")

    monkeypatch.setattr(api, "_resolve_dataset_entry", fail_resolve)

    try:
        with pytest.raises(api.HTTPException) as exc_info:
            api.export_captions("ds", {"block_active_caption_jobs": True})
    finally:
        with api.QWEN_CAPTION_DATASET_JOBS_LOCK:
            api.QWEN_CAPTION_DATASET_JOBS.clear()
            api.QWEN_CAPTION_DATASET_JOBS.update(original_jobs)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "caption_export_busy:qcap_persisted:running"


def test_caption_dataset_job_start_blocks_live_persisted_runner_before_registration(
    tmp_path,
    monkeypatch,
) -> None:
    jobs_root = tmp_path / "jobs"
    artifact_dir = tmp_path / "artifact"
    job_dir = jobs_root / "qcap_persisted"
    job_dir.mkdir(parents=True)
    artifact_dir.mkdir()
    (artifact_dir / ".runner.lock").write_text(json.dumps({"pid": 12345}), encoding="utf-8")
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "qcap_persisted",
                "status": "running",
                "request": {"dataset_id": "ds", "output_dir": str(artifact_dir)},
                "output_dir": str(artifact_dir),
                "created_at": 10.0,
                "updated_at": 20.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)
    monkeypatch.setattr(api, "_qwen_caption_dataset_pid_is_alive", lambda _pid: True)
    with api.QWEN_CAPTION_DATASET_JOBS_LOCK:
        original_jobs = dict(api.QWEN_CAPTION_DATASET_JOBS)
        api.QWEN_CAPTION_DATASET_JOBS.clear()

    try:
        with pytest.raises(api.HTTPException) as exc_info:
            api._start_qwen_caption_dataset_job(
                api.QwenCaptionDatasetJobRequest(
                    dataset_id="ds",
                    caption_request={"user_prompt": "Describe it"},
                )
            )
        with api.QWEN_CAPTION_DATASET_JOBS_LOCK:
            live_jobs = dict(api.QWEN_CAPTION_DATASET_JOBS)
    finally:
        with api.QWEN_CAPTION_DATASET_JOBS_LOCK:
            api.QWEN_CAPTION_DATASET_JOBS.clear()
            api.QWEN_CAPTION_DATASET_JOBS.update(original_jobs)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "qwen_caption_dataset_job_active:qcap_persisted:running"
    assert live_jobs == {}


def test_download_dataset_entry_blocks_active_backend_caption_job_before_dataset_read(
    monkeypatch,
) -> None:
    job = api.QwenCaptionDatasetJob(job_id="qcap_busy", status="running")
    job.request = {"dataset_id": "ds"}
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {job.job_id: job})

    def fail_resolve(_dataset_id):
        raise AssertionError("dataset download should block before reading a mutating dataset")

    monkeypatch.setattr(api, "_resolve_dataset_entry", fail_resolve)

    with pytest.raises(api.HTTPException) as exc_info:
        api.download_dataset_entry("ds")

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "dataset_download_busy:qcap_busy:running"


def test_dataset_download_route_blocks_when_backend_caption_job_is_active() -> None:
    def download_dataset(_dataset_id):
        raise api.HTTPException(
            status_code=409,
            detail="dataset_download_busy:qcap_busy:running",
        )

    client = _dataset_export_route_client({}, download_fn=download_dataset)

    response = client.get("/datasets/ds/download")

    assert response.status_code == 409
    assert response.json()["detail"] == "dataset_download_busy:qcap_busy:running"


def test_set_dataset_glossary_blocks_active_backend_caption_job_before_dataset_read(
    monkeypatch,
) -> None:
    job = api.QwenCaptionDatasetJob(job_id="qcap_busy", status="running")
    job.request = {"dataset_id": "ds"}
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {job.job_id: job})

    def fail_resolve(_dataset_id):
        raise AssertionError("glossary save should block before reading a mutating dataset")

    monkeypatch.setattr(api, "_resolve_dataset_entry", fail_resolve)

    with pytest.raises(api.HTTPException) as exc_info:
        api.set_dataset_glossary("ds", '{"car": ["vehicle"]}')

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "caption_metadata_busy:qcap_busy:running"


def test_dataset_glossary_route_blocks_when_backend_caption_job_is_active() -> None:
    def set_glossary(_dataset_id, _glossary):
        raise api.HTTPException(
            status_code=409,
            detail="caption_metadata_busy:qcap_busy:running",
        )

    client = _dataset_export_route_client({}, set_glossary_fn=set_glossary)

    response = client.post("/datasets/ds/glossary", json={"glossary": "{}"})

    assert response.status_code == 409
    assert response.json()["detail"] == "caption_metadata_busy:qcap_busy:running"


def test_instruction_review_import_blocks_active_backend_caption_job_before_dataset_read(monkeypatch) -> None:
    job = api.QwenCaptionDatasetJob(job_id="qcap_busy", status="running")
    job.request = {"dataset_id": "ds"}
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {job.job_id: job})

    def fail_resolve(_dataset_id):
        raise AssertionError("review import should block before reading a mutating dataset")

    monkeypatch.setattr(api, "_resolve_dataset_entry", fail_resolve)

    with pytest.raises(api.HTTPException) as exc_info:
        api.apply_caption_instruction_review("ds", {"rows": []})

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "caption_review_import_busy:qcap_busy:running"


def test_instruction_review_route_blocks_when_backend_caption_job_is_active() -> None:
    def apply_review(_dataset_id, _payload):
        raise api.HTTPException(
            status_code=409,
            detail="caption_review_import_busy:qcap_busy:running",
        )

    client = _dataset_export_route_client({}, apply_review_fn=apply_review)

    response = client.post("/datasets/ds/captions/instruction_review", json={"rows": []})

    assert response.status_code == 409
    assert response.json()["detail"] == "caption_review_import_busy:qcap_busy:running"


@pytest.mark.parametrize(
    ("operation", "call"),
    [
        ("single_caption_read", lambda: api.get_captions("ds", "img.jpg")),
        ("batch_caption_read", lambda: api.get_captions_batch("ds", ["img.jpg"])),
    ],
)
def test_caption_reads_block_active_backend_caption_job_before_dataset_read(
    monkeypatch,
    operation,
    call,
) -> None:
    job = api.QwenCaptionDatasetJob(job_id="qcap_busy", status="running")
    job.request = {"dataset_id": "ds"}
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {job.job_id: job})

    def fail_resolve(_dataset_id):
        raise AssertionError(f"{operation} should block before reading a mutating dataset")

    monkeypatch.setattr(api, "_resolve_dataset_entry", fail_resolve)

    with pytest.raises(api.HTTPException) as exc_info:
        call()

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "caption_read_busy:qcap_busy:running"


def test_caption_read_routes_block_when_backend_caption_job_is_active() -> None:
    def read_caption(_dataset_id, _image_name):
        raise api.HTTPException(
            status_code=409,
            detail="caption_read_busy:qcap_busy:running",
        )

    def read_caption_batch(_dataset_id, _image_names):
        raise api.HTTPException(
            status_code=409,
            detail="caption_read_busy:qcap_busy:running",
        )

    client = _dataset_export_route_client(
        {},
        get_captions_fn=read_caption,
        get_captions_batch_fn=read_caption_batch,
    )

    response = client.get("/datasets/ds/captions/img.jpg")

    assert response.status_code == 409
    assert response.json()["detail"] == "caption_read_busy:qcap_busy:running"

    response = client.post("/datasets/ds/captions/batch", json={"image_names": ["img.jpg"]})

    assert response.status_code == 409
    assert response.json()["detail"] == "caption_read_busy:qcap_busy:running"


@pytest.mark.parametrize(
    ("operation", "call"),
    [
        ("single_text_label_read", lambda: api.get_text_label("ds", "img.jpg")),
        ("batch_text_label_read", lambda: api.get_text_labels("ds", ["img.jpg"])),
    ],
)
def test_text_label_reads_block_active_backend_caption_job_before_dataset_read(
    monkeypatch,
    operation,
    call,
) -> None:
    job = api.QwenCaptionDatasetJob(job_id="qcap_busy", status="running")
    job.request = {"dataset_id": "ds"}
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {job.job_id: job})

    def fail_resolve(_dataset_id):
        raise AssertionError(f"{operation} should block before reading a mutating dataset")

    monkeypatch.setattr(api, "_resolve_dataset_entry", fail_resolve)

    with pytest.raises(api.HTTPException) as exc_info:
        call()

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "caption_read_busy:qcap_busy:running"


def test_text_label_read_routes_block_when_backend_caption_job_is_active() -> None:
    def read_text_label(_dataset_id, _image_name):
        raise api.HTTPException(
            status_code=409,
            detail="caption_read_busy:qcap_busy:running",
        )

    def read_text_labels_batch(_dataset_id, _image_names):
        raise api.HTTPException(
            status_code=409,
            detail="caption_read_busy:qcap_busy:running",
        )

    client = _dataset_export_route_client(
        {},
        get_text_label_fn=read_text_label,
        get_text_labels_fn=read_text_labels_batch,
    )

    response = client.get("/datasets/ds/text_labels/img.jpg")

    assert response.status_code == 409
    assert response.json()["detail"] == "caption_read_busy:qcap_busy:running"

    response = client.post("/datasets/ds/text_labels/batch", json={"image_names": ["img.jpg"]})

    assert response.status_code == 409
    assert response.json()["detail"] == "caption_read_busy:qcap_busy:running"


@pytest.mark.parametrize(
    ("operation", "call"),
    [
        ("text_label", lambda: api.set_text_label("ds", "img.jpg", {"caption": "updated"})),
        ("add_caption", lambda: api.add_caption("ds", "img.jpg", {"caption": "new"})),
        ("update_caption", lambda: api.update_caption("ds", "cap_1", {"caption": "updated"})),
        ("delete_caption", lambda: api.delete_caption("ds", "cap_1", {})),
    ],
)
def test_caption_mutations_block_active_backend_caption_job_before_dataset_read(
    monkeypatch,
    operation,
    call,
) -> None:
    job = api.QwenCaptionDatasetJob(job_id="qcap_busy", status="running")
    job.request = {"dataset_id": "ds"}
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {job.job_id: job})

    def fail_resolve(_dataset_id):
        raise AssertionError(f"{operation} should block before reading a mutating dataset")

    monkeypatch.setattr(api, "_resolve_dataset_entry", fail_resolve)

    with pytest.raises(api.HTTPException) as exc_info:
        call()

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "caption_mutation_busy:qcap_busy:running"


def _ready_instruction_export_payload():
    export_validation = {"ok": True, "error_count": 0, "errors": [], "row_count": 1}
    artifact_consistency = {
        "format": "tator_caption_instruction_artifact_consistency_v1",
        "ok": True,
        "error_count": 0,
        "errors": [],
        "counts": {
            "training_row_count": 1,
            "archive_row_count": 1,
            "review_row_count": 1,
            "selected_review_row_count": 1,
            "manual_review_required_count": 0,
            "report_image_count": 1,
            "report_selected_flattened_row_count": 1,
            "report_instruction_review_row_count": 1,
            "report_manual_review_required_count": 0,
            "instruction_export_validation_row_count": 1,
        },
    }
    report = {
        "format": "tator_caption_instruction_report_v1",
        "selected_flattened_row_count": 1,
        "image_count": 1,
        "instruction_review_row_count": 1,
        "manual_review_required_count": 0,
        "corpus_quality_metrics": {"selected_flattened_row_count": 1, "image_count": 1},
        "training_readiness": {
            "status": "ready",
            "ready_for_training": True,
            "blocking_reasons": [],
            "required_actions": [],
            "quality_warnings": [],
            "selected_training_row_count": 1,
            "selected_review_row_count": 1,
            "selected_manual_review_row_count": 0,
            "pending_manual_review_row_count": 0,
            "rejected_manual_review_row_count": 0,
            "needs_revision_manual_review_row_count": 0,
        },
        "instruction_export_validation": dict(export_validation),
        "instruction_artifact_consistency": dict(artifact_consistency),
    }
    training_row = {
        "image_path": "frame.jpg",
        "question": "What is shown?",
        "answer": "A building.",
        "metadata": {
            "qa_id": "qa-1",
            "row_type": "generated_qa",
            "answer_source": "generated_qa_record",
            "source_archive": "tator_caption_instruction_archive_v1",
            "answer_format": "natural",
            "validation_status": "accepted",
            "review_status": "accepted",
        },
    }
    archive_row = {
        "image_path": "frame.jpg",
        "source_annotations": {},
        "language_annotations": {},
        "deterministic_metadata_qa_pairs": [],
        "export_metadata": {},
    }
    review_row = {
        "format": "tator_caption_instruction_review_rows_v1",
        "dataset_id": "ds",
        "image_path": "frame.jpg",
        "row_origin": "generated_qa",
        "qa_id": "qa-1",
        "question": "What is shown?",
        "candidate_answer": "A building.",
        "training_answer": "A building.",
        "validation_status": "accepted",
        "selected_for_training": True,
        "requires_manual_review": False,
        "review_decision": "",
        "review_notes": "",
        "source_summary": {},
        "rejection_reasons": [],
    }
    return {
        "instruction_report": report,
        "instruction_export_validation": dict(export_validation),
        "instruction_artifact_consistency": dict(artifact_consistency),
        "instruction_archive": {"instruction_artifact_consistency": dict(artifact_consistency)},
        "instruction_training_rows": [training_row],
        "instruction_archive_rows": [archive_row],
        "instruction_review_rows": [review_row],
    }


def test_caption_instruction_strict_export_gate_requires_ready_proofs() -> None:
    from api.datasets import _instruction_export_not_ready_reason

    export_validation = {"ok": True, "error_count": 0, "errors": [], "row_count": 1}
    artifact_consistency = {
        "format": "tator_caption_instruction_artifact_consistency_v1",
        "ok": True,
        "error_count": 0,
        "errors": [],
        "counts": {
            "training_row_count": 1,
            "archive_row_count": 1,
            "review_row_count": 1,
            "selected_review_row_count": 1,
            "manual_review_required_count": 0,
            "report_image_count": 1,
            "report_selected_flattened_row_count": 1,
            "report_instruction_review_row_count": 1,
            "report_manual_review_required_count": 0,
            "instruction_export_validation_row_count": 1,
        },
    }
    report = {
        "format": "tator_caption_instruction_report_v1",
        "selected_flattened_row_count": 1,
        "image_count": 1,
        "instruction_review_row_count": 1,
        "manual_review_required_count": 0,
        "corpus_quality_metrics": {"selected_flattened_row_count": 1, "image_count": 1},
        "training_readiness": {
            "status": "ready",
            "ready_for_training": True,
            "blocking_reasons": [],
            "required_actions": [],
            "quality_warnings": [],
            "selected_training_row_count": 1,
            "selected_review_row_count": 1,
            "selected_manual_review_row_count": 0,
            "pending_manual_review_row_count": 0,
            "rejected_manual_review_row_count": 0,
            "needs_revision_manual_review_row_count": 0,
        },
        "instruction_export_validation": dict(export_validation),
        "instruction_artifact_consistency": dict(artifact_consistency),
    }
    training_row = {
        "image_path": "frame.jpg",
        "question": "What is shown?",
        "answer": "A building.",
        "metadata": {
            "qa_id": "qa-1",
            "row_type": "generated_qa",
            "answer_source": "generated_qa_record",
            "source_archive": "tator_caption_instruction_archive_v1",
            "answer_format": "natural",
            "validation_status": "accepted",
            "review_status": "accepted",
        },
    }
    archive_row = {
        "image_path": "frame.jpg",
        "source_annotations": {},
        "language_annotations": {},
        "deterministic_metadata_qa_pairs": [],
        "export_metadata": {},
    }
    review_row = {
        "format": "tator_caption_instruction_review_rows_v1",
        "dataset_id": "ds",
        "image_path": "frame.jpg",
        "row_origin": "generated_qa",
        "qa_id": "qa-1",
        "question": "What is shown?",
        "candidate_answer": "A building.",
        "training_answer": "A building.",
        "validation_status": "accepted",
        "selected_for_training": True,
        "requires_manual_review": False,
        "review_decision": "",
        "review_notes": "",
        "source_summary": {},
        "rejection_reasons": [],
    }
    payload = {
        "instruction_report": report,
        "instruction_export_validation": dict(export_validation),
        "instruction_artifact_consistency": dict(artifact_consistency),
        "instruction_archive": {"instruction_artifact_consistency": dict(artifact_consistency)},
        "instruction_training_rows": [training_row],
        "instruction_archive_rows": [archive_row],
        "instruction_review_rows": [review_row],
    }

    assert _instruction_export_not_ready_reason(payload) == ""
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_report": {
                **report,
                "training_readiness": {**report["training_readiness"], "status": "needs_review"},
            },
        }
    ) == "needs_review"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_report": {
                **report,
                "format": "wrong_format",
            },
        }
    ) == "instruction_report"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_report": {
                **report,
                "training_readiness": {**report["training_readiness"], "ready_for_training": False},
            },
        }
    ) == "training_readiness"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_report": {
                **report,
                "training_readiness": {**report["training_readiness"], "quality_warnings": ["review first"]},
            },
        }
    ) == "training_readiness"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_report": {
                **report,
                "corpus_quality_metrics": {},
            },
        }
    ) == "corpus_quality_metrics"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_report": {
                **report,
                "corpus_quality_metrics": {**report["corpus_quality_metrics"], "selected_flattened_row_count": 2},
            },
        }
    ) == "corpus_quality_metrics"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_export_validation": {
                **export_validation,
                "ok": False,
                "error_count": 1,
                "errors": ["bad row"],
            },
        }
    ) == "instruction_export_validation"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_export_validation": {
                **export_validation,
                "row_count": 2,
            },
        }
    ) == "instruction_export_validation_mismatch"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_archive": {},
        }
    ) == "instruction_artifact_consistency"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_artifact_consistency": {
                **artifact_consistency,
                "format": "wrong_format",
            },
        }
    ) == "instruction_artifact_consistency"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_archive": {
                "instruction_artifact_consistency": {
                    **artifact_consistency,
                    "counts": {"training_row_count": 2},
                },
            },
        }
    ) == "instruction_artifact_consistency_mismatch"
    assert _instruction_export_not_ready_reason(
        {
            key: value
            for key, value in payload.items()
            if key != "instruction_training_rows"
        }
    ) == "instruction_training_rows"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_training_rows": [],
        }
    ) == "instruction_training_rows"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_training_rows": [
                {
                    **training_row,
                    "metadata": {**training_row["metadata"], "source_archive": "wrong_archive"},
                }
            ],
        }
    ) == "instruction_training_rows"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_training_rows": [
                {
                    **training_row,
                    "metadata": {**training_row["metadata"], "review_status": "needs_revision"},
                }
            ],
        }
    ) == "instruction_training_rows"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_archive_rows": [],
        }
    ) == "instruction_archive_rows"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_archive_rows": [
                {
                    key: value
                    for key, value in archive_row.items()
                    if key != "source_annotations"
                }
            ],
        }
    ) == "instruction_archive_rows"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_review_rows": [],
        }
    ) == "instruction_review_rows"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_review_rows": [
                {
                    key: value
                    for key, value in review_row.items()
                    if key != "training_answer"
                }
            ],
        }
    ) == "instruction_review_rows"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_review_rows": [{**review_row, "selected_for_training": False}],
        }
    ) == "instruction_review_rows"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_report": {
                **report,
                "manual_review_required_count": 1,
            },
        }
    ) == "instruction_review_rows"
    assert _instruction_export_not_ready_reason(
        {
            **payload,
            "instruction_report": {
                **report,
                "training_readiness": {**report["training_readiness"], "pending_manual_review_row_count": 1},
            },
        }
    ) == "training_readiness"


@pytest.mark.parametrize(
    ("row_key", "expected_reason"),
    (
        ("instruction_training_rows", "instruction_training_rows"),
        ("instruction_archive_rows", "instruction_archive_rows"),
        ("instruction_review_rows", "instruction_review_rows"),
    ),
)
def test_caption_instruction_strict_export_route_blocks_malformed_rows_when_ready_required(
    row_key, expected_reason
) -> None:
    payload = {
        **_ready_instruction_export_payload(),
        row_key: {"bad": True},
    }
    client = _dataset_export_route_client(payload)

    diagnostic_response = client.get("/datasets/ds/captions/export")
    assert diagnostic_response.status_code == 200
    assert diagnostic_response.json()[row_key] == {"bad": True}

    strict_response = client.get("/datasets/ds/captions/export?require_ready_instruction_export=true")
    assert strict_response.status_code == 409
    assert strict_response.json()["detail"] == f"instruction_export_not_ready:{expected_reason}"


def test_delete_linked_dataset_only_removes_registry_record(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "keep.txt").write_text("source", encoding="utf-8")

    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    record_root = registry_root / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    (record_root / api.DATASET_META_NAME).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )

    out = api.delete_dataset_entry("ds_linked")

    assert out["status"] == "deleted"
    assert out["storage_mode"] == "linked"
    assert source_root.exists(), "linked source must never be deleted"
    assert not record_root.exists(), "registry record should be removed"


def test_delete_linked_dataset_reports_registry_remove_failure_without_source_delete(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "keep.txt").write_text("source", encoding="utf-8")

    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    record_root = registry_root / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    (record_root / api.DATASET_META_NAME).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )

    original_rmtree = api.shutil.rmtree

    def fail_rmtree(path, *args, **kwargs):
        if Path(path).resolve(strict=False) == record_root.resolve(strict=False):
            raise OSError("simulated registry remove failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(api.shutil, "rmtree", fail_rmtree)

    with pytest.raises(api.HTTPException) as exc:
        api.delete_dataset_entry("ds_linked")

    assert exc.value.status_code == 500
    assert str(exc.value.detail).startswith("dataset_delete_failed:")
    assert source_root.exists(), "linked source must never be deleted"
    assert record_root.exists(), "registry record must remain when deletion fails"


def test_delete_linked_dataset_unlinks_registry_symlink_without_target_delete(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    target_record = registry_root / "target_record"
    target_record.mkdir(parents=True, exist_ok=True)
    (target_record / "payload.bin").write_bytes(b"target")
    record_link = registry_root / "ds_linked"
    try:
        record_link.symlink_to(target_record, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_link),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )

    out = api.delete_dataset_entry("ds_linked")

    assert out["status"] == "deleted"
    assert not record_link.exists()
    assert not record_link.is_symlink()
    assert (target_record / "payload.bin").read_bytes() == b"target"
    assert source_root.exists()


def test_delete_linked_dataset_rejects_symlinked_registry_parent_without_target_delete(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    target_parent = registry_root / "target_parent"
    target_parent.mkdir(parents=True, exist_ok=True)
    target_record = target_parent / "ds_linked"
    target_record.mkdir()
    (target_record / "payload.bin").write_bytes(b"target")
    parent_link = registry_root / "linked_parent"
    try:
        parent_link.symlink_to(target_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(parent_link / "ds_linked"),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.delete_dataset_entry("ds_linked")

    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_delete_forbidden"
    assert (target_record / "payload.bin").read_bytes() == b"target"
    assert source_root.exists()


def test_delete_managed_dataset_rejects_symlinked_registry_parent_without_target_delete(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    target_parent = registry_root / "target_parent"
    target_parent.mkdir(parents=True, exist_ok=True)
    target_root = target_parent / "managed_ds"
    target_root.mkdir()
    (target_root / "payload.bin").write_bytes(b"target")
    parent_link = registry_root / "linked_parent"
    try:
        parent_link.symlink_to(target_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "dataset_root": str(parent_link / "managed_ds"),
            "registry_root": str(parent_link / "managed_ds"),
            "storage_mode": "managed",
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.delete_dataset_entry("managed_ds")

    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_delete_forbidden"
    assert (target_root / "payload.bin").read_bytes() == b"target"


def test_delete_managed_dataset_moves_to_trash_and_restores(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    dataset_root = registry_root / "managed_ds"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "payload.bin").write_bytes(b"dataset")
    (dataset_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "managed_ds", "label": "Managed DS"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "label": "Managed DS",
            "dataset_root": str(dataset_root),
            "registry_root": str(dataset_root),
            "storage_mode": "managed",
            "source": "registry",
        },
    )

    out = api.delete_dataset_entry("managed_ds")

    assert out["status"] == "trashed"
    assert out["restore_available"] is True
    assert not dataset_root.exists()
    trash_entries = api.list_dataset_trash_entries()
    assert [entry["trash_id"] for entry in trash_entries] == [out["trash_id"]]
    assert trash_entries[0]["original_id"] == "managed_ds"
    trashed_payload = Path(trash_entries[0]["dataset_path"]) / "payload.bin"
    assert trashed_payload.read_bytes() == b"dataset"

    restored = api.restore_dataset_trash_entry(out["trash_id"])

    assert restored["status"] == "restored"
    assert restored["id"] == "managed_ds"
    assert dataset_root.exists()
    assert (dataset_root / "payload.bin").read_bytes() == b"dataset"
    restored_meta = json.loads((dataset_root / api.DATASET_META_NAME).read_text(encoding="utf-8"))
    assert restored_meta["id"] == "managed_ds"
    assert restored_meta["restored_from_trash_id"] == out["trash_id"]
    assert api.list_dataset_trash_entries() == []


def test_restore_managed_dataset_uses_unique_id_when_original_exists(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    dataset_root = registry_root / "managed_ds"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "payload.bin").write_bytes(b"dataset")
    (dataset_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "managed_ds", "label": "Managed DS"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "label": "Managed DS",
            "dataset_root": str(dataset_root),
            "registry_root": str(dataset_root),
            "storage_mode": "managed",
        },
    )
    trashed = api.delete_dataset_entry("managed_ds")
    replacement = registry_root / "managed_ds"
    replacement.mkdir()
    (replacement / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "managed_ds", "label": "Replacement"}),
        encoding="utf-8",
    )

    restored = api.restore_dataset_trash_entry(trashed["trash_id"])

    assert restored["id"] == "managed_ds_1"
    restored_root = registry_root / "managed_ds_1"
    assert restored_root.exists()
    assert (restored_root / "payload.bin").read_bytes() == b"dataset"
    restored_meta = json.loads((restored_root / api.DATASET_META_NAME).read_text(encoding="utf-8"))
    assert restored_meta["id"] == "managed_ds_1"
    assert (replacement / api.DATASET_META_NAME).exists()


def test_restore_managed_dataset_rolls_back_when_metadata_write_fails(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    dataset_root = registry_root / "managed_ds"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "payload.bin").write_bytes(b"dataset")
    (dataset_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "managed_ds", "label": "Managed DS"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "label": "Managed DS",
            "dataset_root": str(dataset_root),
            "registry_root": str(dataset_root),
            "storage_mode": "managed",
        },
    )
    trashed = api.delete_dataset_entry("managed_ds")
    trash_entry = api.list_dataset_trash_entries()[0]
    trash_dataset_root = Path(trash_entry["dataset_path"])

    def fail_metadata_write(*_args, **_kwargs):
        raise api.HTTPException(status_code=400, detail="metadata_write_failed")

    monkeypatch.setattr(api, "_write_dataset_metadata_json", fail_metadata_write)

    with pytest.raises(api.HTTPException) as exc:
        api.restore_dataset_trash_entry(trashed["trash_id"])

    assert exc.value.detail == "metadata_write_failed"
    assert not dataset_root.exists()
    assert trash_dataset_root.exists()
    assert (trash_dataset_root / "payload.bin").read_bytes() == b"dataset"


def test_restore_managed_dataset_rejects_symlinked_trash_entry_without_target_write(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    trash_root = registry_root / api.DATASET_TRASH_DIRNAME
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    trash_root.mkdir(parents=True)
    try:
        (trash_root / "linked-trash").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    marker = outside / "payload.bin"
    marker.write_bytes(b"outside")
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")

    with pytest.raises(api.HTTPException) as exc:
        api.restore_dataset_trash_entry("linked-trash")

    assert exc.value.status_code == 404
    assert exc.value.detail == "dataset_trash_entry_not_found"
    assert marker.read_bytes() == b"outside"


def test_delete_linked_dataset_blocks_active_annotation_lock(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)

    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    record_root = registry_root / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": "ds_linked",
        "annotation_lock": {
            "holder": "annotator",
            "session_id": "sess-active",
            "expires_at": time.time() + 300.0,
        },
    }
    (record_root / api.DATASET_META_NAME).write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.delete_dataset_entry("ds_linked")

    assert exc.value.status_code == 409
    assert exc.value.detail == "dataset_delete_blocked_annotation_lock"
    assert record_root.exists()
    assert source_root.exists()


def test_delete_linked_dataset_blocks_active_job_reference(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)

    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    record_root = registry_root / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    (record_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "ds_linked"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )
    with api.DATA_INGESTION_JOBS_LOCK:
        original_jobs = dict(api.DATA_INGESTION_JOBS)
        api.DATA_INGESTION_JOBS.clear()
        api.DATA_INGESTION_JOBS["di_active"] = api.DataIngestionJob(
            job_id="di_active",
            status="running",
            request={"reference_dataset_id": "ds_linked"},
        )

    try:
        with pytest.raises(api.HTTPException) as exc:
            api.delete_dataset_entry("ds_linked")
    finally:
        with api.DATA_INGESTION_JOBS_LOCK:
            api.DATA_INGESTION_JOBS.clear()
            api.DATA_INGESTION_JOBS.update(original_jobs)

    assert exc.value.status_code == 409
    assert exc.value.detail == "dataset_delete_blocked_active_jobs:data_ingestion"
    assert record_root.exists()
    assert source_root.exists()


def test_delete_linked_dataset_blocks_active_caption_dataset_job(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)

    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    record_root = registry_root / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    (record_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "ds_linked"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )
    job = api.QwenCaptionDatasetJob(job_id="qcap_active", status="running")
    job.request = {"dataset_id": "ds_linked"}
    with api.QWEN_CAPTION_DATASET_JOBS_LOCK:
        original_jobs = dict(api.QWEN_CAPTION_DATASET_JOBS)
        api.QWEN_CAPTION_DATASET_JOBS.clear()
        api.QWEN_CAPTION_DATASET_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as exc:
            api.delete_dataset_entry("ds_linked")
    finally:
        with api.QWEN_CAPTION_DATASET_JOBS_LOCK:
            api.QWEN_CAPTION_DATASET_JOBS.clear()
            api.QWEN_CAPTION_DATASET_JOBS.update(original_jobs)

    assert exc.value.status_code == 409
    assert exc.value.detail == "dataset_delete_blocked_active_jobs:qwen_caption_dataset"
    assert record_root.exists()
    assert source_root.exists()


def test_delete_linked_dataset_allows_completed_caption_dataset_job(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)

    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    record_root = registry_root / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    (record_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "ds_linked"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )
    job = api.QwenCaptionDatasetJob(job_id="qcap_done", status="completed")
    job.request = {"dataset_id": "ds_linked"}
    with api.QWEN_CAPTION_DATASET_JOBS_LOCK:
        original_jobs = dict(api.QWEN_CAPTION_DATASET_JOBS)
        api.QWEN_CAPTION_DATASET_JOBS.clear()
        api.QWEN_CAPTION_DATASET_JOBS[job.job_id] = job

    try:
        out = api.delete_dataset_entry("ds_linked")
    finally:
        with api.QWEN_CAPTION_DATASET_JOBS_LOCK:
            api.QWEN_CAPTION_DATASET_JOBS.clear()
            api.QWEN_CAPTION_DATASET_JOBS.update(original_jobs)

    assert out == {"status": "deleted", "id": "ds_linked", "storage_mode": "linked"}
    assert not record_root.exists()
    assert source_root.exists()


def test_delete_managed_dataset_blocks_active_annotation_lock(tmp_path, monkeypatch) -> None:
    registry_root = tmp_path / "registry"
    dataset_root = registry_root / "managed_ds"
    dataset_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": "managed_ds",
        "annotation_lock": {
            "holder": "annotator",
            "session_id": "sess-active",
            "expires_at": time.time() + 300.0,
        },
    }
    (dataset_root / api.DATASET_META_NAME).write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "dataset_root": str(dataset_root),
            "registry_root": str(dataset_root),
            "storage_mode": "managed",
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.delete_dataset_entry("managed_ds")

    assert exc.value.status_code == 409
    assert exc.value.detail == "dataset_delete_blocked_annotation_lock"
    assert dataset_root.exists()


def test_delete_managed_dataset_ignores_completed_job_reference(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    dataset_root = registry_root / "managed_ds"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "payload.bin").write_bytes(b"dataset")
    (dataset_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "managed_ds", "label": "Managed DS"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "label": "Managed DS",
            "dataset_root": str(dataset_root),
            "registry_root": str(dataset_root),
            "storage_mode": "managed",
            "source": "registry",
        },
    )
    with api.DATA_INGESTION_JOBS_LOCK:
        original_jobs = dict(api.DATA_INGESTION_JOBS)
        api.DATA_INGESTION_JOBS.clear()
        api.DATA_INGESTION_JOBS["di_done"] = api.DataIngestionJob(
            job_id="di_done",
            status="completed",
            request={
                "reference_dataset_id": "managed_ds",
                "reference_uploads": [{"path": str(dataset_root / "payload.bin")}],
            },
        )

    try:
        out = api.delete_dataset_entry("managed_ds")
    finally:
        with api.DATA_INGESTION_JOBS_LOCK:
            api.DATA_INGESTION_JOBS.clear()
            api.DATA_INGESTION_JOBS.update(original_jobs)

    assert out["status"] == "trashed"
    assert not dataset_root.exists()
    trash_entries = api.list_dataset_trash_entries()
    assert len(trash_entries) == 1
    assert trash_entries[0]["original_id"] == "managed_ds"


def test_save_transient_dataset_persists_overlay_content(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)

    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "_validate_linked_dataset_path", lambda _p: source_root)

    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS.clear()
        api.DATASET_TRANSIENT_SESSIONS["sess1"] = {
            "session_id": "sess1",
            "dataset_root": str(source_root),
            "label": "Transient DS",
            "classes": ["car"],
            "yolo_layout": "flat",
            "annotation_status": "in_progress",
            "annotation_notes": "half done",
            "annotation_cursor": {"split": "train", "image_relpath": "img1.jpg"},
            "annotation_progress": {"images_total": 1},
            "overlay_labels": {"train:img1.jpg": ["0 0.5 0.5 0.2 0.2"]},
            "overlay_text": {"train:img1.jpg": "car in frame"},
        }

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry_impl",
        lambda dataset_id, **_kwargs: {
            "id": dataset_id,
            "dataset_root": str(source_root),
            "registry_root": str(registry_root / dataset_id),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
            "classes": ["car"],
        },
    )

    out = api.save_transient_dataset(
        "sess1",
        dataset_id="saved_linked",
        label="Saved Linked",
        context="context",
        notes="notes",
    )

    record_root = registry_root / "saved_linked"
    overlay_label = (
        record_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "labels" / "train" / "img1.txt"
    )
    overlay_text = record_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "text_labels" / "img1.txt"
    registry_labelmap = record_root / "labelmap.txt"
    meta_path = record_root / api.DATASET_META_NAME

    assert out["id"] == "saved_linked"
    assert not (source_root / "labelmap.txt").exists()
    assert registry_labelmap.read_text(encoding="utf-8") == "car\n"
    assert overlay_label.exists()
    assert overlay_label.read_text(encoding="utf-8").strip() == "0 0.5 0.5 0.2 0.2"
    assert overlay_text.exists()
    assert overlay_text.read_text(encoding="utf-8").strip() == "car in frame"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["annotation_status"] == "in_progress"
    assert meta["annotation_notes"] == "notes"
    assert meta["classes"] == ["car"]
    assert meta["yolo_labelmap_path"] == str(registry_labelmap)
    assert meta["labelmap_source"] == "registry_overlay"


def test_save_transient_dataset_rejects_symlinked_registry_root(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_registry"
    outside.mkdir()
    registry_root = tmp_path / "registry"
    try:
        registry_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "_validate_linked_dataset_path", lambda _p: source_root)

    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS.clear()
        api.DATASET_TRANSIENT_SESSIONS["sess1"] = {
            "session_id": "sess1",
            "dataset_root": str(source_root),
            "label": "Transient DS",
            "classes": ["car"],
            "yolo_layout": "flat",
            "annotation_status": "in_progress",
            "overlay_labels": {"train:img1.jpg": ["0 0.5 0.5 0.2 0.2"]},
            "overlay_text": {"train:img1.jpg": "car in frame"},
        }

    with pytest.raises(api.HTTPException) as exc_info:
        api.save_transient_dataset(
            "sess1",
            dataset_id="saved_linked",
            label="Saved Linked",
            context="context",
            notes="notes",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_register_target_invalid"
    assert list(outside.iterdir()) == []


def test_register_dataset_path_rollback_revalidates_registry_root_before_rmtree(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    registry_root = tmp_path / "registry"
    outside_registry = tmp_path / "outside_registry"
    marker_box: dict[str, Path] = {}
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "_validate_linked_dataset_path", lambda _p: source_root)
    monkeypatch.setattr(
        api,
        "_validate_linked_dataset_shape",
        lambda *_args, **_kwargs: {"yolo_layout": "flat"},
    )

    def fail_metadata_write(_path, root, _meta):
        registry_dir = Path(root)
        dataset_name = registry_dir.name
        shutil.rmtree(registry_root)
        outside_registry.mkdir()
        outside_dataset = outside_registry / dataset_name
        outside_dataset.mkdir()
        marker = outside_dataset / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        marker_box["marker"] = marker
        registry_root.symlink_to(outside_registry, target_is_directory=True)
        raise api.HTTPException(status_code=400, detail="metadata_write_failed")

    monkeypatch.setattr(api, "_write_dataset_metadata_json", fail_metadata_write)

    with pytest.raises(api.HTTPException) as exc_info:
        api.register_dataset_path(
            str(source_root),
            dataset_id="saved_linked",
            label="Saved Linked",
            context="context",
            notes="notes",
        )

    assert exc_info.value.detail == "metadata_write_failed"
    assert marker_box["marker"].read_text(encoding="utf-8") == "keep"
    assert registry_root.is_symlink()
    registry_root.unlink()


def test_save_transient_dataset_rolls_back_when_metadata_write_fails(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "_validate_linked_dataset_path", lambda _p: source_root)

    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS.clear()
        api.DATASET_TRANSIENT_SESSIONS["sess1"] = {
            "session_id": "sess1",
            "dataset_root": str(source_root),
            "label": "Transient DS",
            "classes": ["car"],
            "yolo_layout": "flat",
            "annotation_status": "in_progress",
            "overlay_labels": {"train:img1.jpg": ["0 0.5 0.5 0.2 0.2"]},
            "overlay_text": {"train:img1.jpg": "car in frame"},
        }

    def fail_metadata_write(*_args, **_kwargs):
        raise api.HTTPException(status_code=400, detail="metadata_write_failed")

    monkeypatch.setattr(api, "_write_dataset_metadata_json", fail_metadata_write)

    with pytest.raises(api.HTTPException) as exc_info:
        api.save_transient_dataset(
            "sess1",
            dataset_id="saved_linked",
            label="Saved Linked",
            context="context",
            notes="notes",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "metadata_write_failed"
    assert not (registry_root / "saved_linked").exists()
    assert not (source_root / "labelmap.txt").exists()


def test_save_transient_dataset_rollback_revalidates_registry_root_before_rmtree(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    registry_root = tmp_path / "registry"
    outside_registry = tmp_path / "outside_registry"
    marker_box: dict[str, Path] = {}
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "_validate_linked_dataset_path", lambda _p: source_root)
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS.clear()
        api.DATASET_TRANSIENT_SESSIONS["sess1"] = {
            "session_id": "sess1",
            "dataset_root": str(source_root),
            "label": "Transient DS",
            "classes": [],
            "yolo_layout": "flat",
        }

    def fail_metadata_write(_path, root, _meta):
        registry_dir = Path(root)
        dataset_name = registry_dir.name
        shutil.rmtree(registry_root)
        outside_registry.mkdir()
        outside_dataset = outside_registry / dataset_name
        outside_dataset.mkdir()
        marker = outside_dataset / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        marker_box["marker"] = marker
        registry_root.symlink_to(outside_registry, target_is_directory=True)
        raise api.HTTPException(status_code=400, detail="metadata_write_failed")

    monkeypatch.setattr(api, "_write_dataset_metadata_json", fail_metadata_write)

    with pytest.raises(api.HTTPException) as exc_info:
        api.save_transient_dataset(
            "sess1",
            dataset_id="saved_linked",
            label="Saved Linked",
            context="context",
            notes="notes",
        )

    assert exc_info.value.detail == "metadata_write_failed"
    assert marker_box["marker"].read_text(encoding="utf-8") == "keep"
    assert registry_root.is_symlink()
    registry_root.unlink()


def test_download_dataset_entry_applies_overlay_files(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    (source_root / "images").mkdir(parents=True, exist_ok=True)
    (source_root / "labels").mkdir(parents=True, exist_ok=True)
    (source_root / "text_labels").mkdir(parents=True, exist_ok=True)
    (source_root / "labels" / "img1.txt").write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    (source_root / "text_labels" / "img1.txt").write_text("old", encoding="utf-8")

    record_root = tmp_path / "registry" / "ds_linked"
    overlay_root = record_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (overlay_root / "text_labels").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "img1.txt").write_text(
        "0 0.9 0.9 0.2 0.2\n", encoding="utf-8"
    )
    (overlay_root / "text_labels" / "img1.txt").write_text("new", encoding="utf-8")

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
        },
    )

    response = api.download_dataset_entry("ds_linked")

    zip_path = Path(response.path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        label_name = f"{source_root.name}/labels/img1.txt"
        text_name = f"{source_root.name}/text_labels/img1.txt"
        assert label_name in names
        assert text_name in names
        assert (
            f"{source_root.name}/{api.DATASET_ANNOTATION_OVERLAY_DIRNAME}/labels/train/img1.txt"
            not in names
        )
        assert zf.read(label_name).decode("utf-8").strip() == "0 0.9 0.9 0.2 0.2"
        assert zf.read(text_name).decode("utf-8").strip() == "new"


def test_download_dataset_entry_fails_if_planned_overlay_disappears(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    (source_root / "labels").mkdir(parents=True, exist_ok=True)
    (source_root / "labels" / "img1.txt").write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    record_root = tmp_path / "registry" / "ds_linked"
    overlay_root = record_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    overlay_label = overlay_root / "labels" / "train" / "img1.txt"
    overlay_label.write_text("0 0.9 0.9 0.2 0.2\n", encoding="utf-8")
    entry = {
        "id": "ds_linked",
        "dataset_root": str(source_root),
        "registry_root": str(record_root),
        "storage_mode": "linked",
        "linked_root": str(source_root),
        "yolo_layout": "flat",
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    real_overlay_entries = api._annotation_overlay_archive_entries

    def disappearing_overlay_entries(entry_arg):
        entries = real_overlay_entries(entry_arg)
        overlay_label.unlink()
        return entries

    monkeypatch.setattr(api, "_annotation_overlay_archive_entries", disappearing_overlay_entries)

    with pytest.raises(api.HTTPException) as exc:
        api.download_dataset_entry("ds_linked")

    assert exc.value.status_code == 412
    assert exc.value.detail == {
        "error": "dataset_export_override_unavailable",
        "path": "labels/img1.txt",
    }


def test_download_dataset_entry_fails_if_planned_overlay_becomes_symlink(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    (source_root / "labels").mkdir(parents=True, exist_ok=True)
    (source_root / "labels" / "img1.txt").write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    record_root = tmp_path / "registry" / "ds_linked"
    overlay_root = record_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    overlay_label = overlay_root / "labels" / "train" / "img1.txt"
    overlay_label.write_text("0 0.9 0.9 0.2 0.2\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    entry = {
        "id": "ds_linked",
        "dataset_root": str(source_root),
        "registry_root": str(record_root),
        "storage_mode": "linked",
        "linked_root": str(source_root),
        "yolo_layout": "flat",
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    real_overlay_entries = api._annotation_overlay_archive_entries

    def symlinked_overlay_entries(entry_arg):
        entries = real_overlay_entries(entry_arg)
        overlay_label.unlink()
        try:
            overlay_label.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"symlink unsupported: {exc}")
        return entries

    monkeypatch.setattr(api, "_annotation_overlay_archive_entries", symlinked_overlay_entries)

    with pytest.raises(api.HTTPException) as exc:
        api.download_dataset_entry("ds_linked")

    assert exc.value.status_code == 412
    assert exc.value.detail == {
        "error": "dataset_export_override_unavailable",
        "path": "labels/img1.txt",
    }


def test_download_dataset_entry_rejects_not_allowlisted_linked_record(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "outside_linked_source"
    (source_root / "images").mkdir(parents=True, exist_ok=True)
    (source_root / "labels").mkdir(parents=True, exist_ok=True)
    (source_root / "labelmap.txt").write_text("car\n", encoding="utf-8")
    record_root = tmp_path / "registry" / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "linked_root_status": "not_allowlisted",
            "yolo_layout": "flat",
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.download_dataset_entry("ds_linked")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_path_not_allowlisted"


def test_download_linked_dataset_applies_registry_labelmap_override(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "labelmap.txt").write_text("old\n", encoding="utf-8")

    record_root = tmp_path / "registry" / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    registry_labelmap = record_root / "labelmap.txt"
    registry_labelmap.write_text("old\nnew\n", encoding="utf-8")

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
            "yolo_labelmap_path": str(registry_labelmap),
            "labelmap_source": "registry_overlay",
        },
    )

    response = api.download_dataset_entry("ds_linked")

    with zipfile.ZipFile(Path(response.path), "r") as zf:
        labelmap_name = f"{source_root.name}/labelmap.txt"
        assert labelmap_name in set(zf.namelist())
        assert zf.read(labelmap_name).decode("utf-8") == "old\nnew\n"


def test_download_split_dataset_applies_split_scoped_text_overlays(
    tmp_path, monkeypatch
) -> None:
    entry = _split_entry_for_annotation(tmp_path)
    source_root = Path(entry["dataset_root"])
    for split, old_text, new_text in (
        ("train", "old train", "new train"),
        ("val", "old val", "new val"),
    ):
        source_text = source_root / split / "text_labels"
        source_text.mkdir(parents=True, exist_ok=True)
        (source_text / "shared.txt").write_text(old_text, encoding="utf-8")
        overlay_text = (
            Path(entry["registry_root"])
            / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
            / "text_labels"
            / split
        )
        overlay_text.mkdir(parents=True, exist_ok=True)
        (overlay_text / "shared.txt").write_text(new_text, encoding="utf-8")

    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    response = api.download_dataset_entry("ds")

    with zipfile.ZipFile(Path(response.path), "r") as zf:
        names = set(zf.namelist())
        train_name = f"{source_root.name}/train/text_labels/shared.txt"
        val_name = f"{source_root.name}/val/text_labels/shared.txt"
        assert train_name in names
        assert val_name in names
        assert zf.read(train_name).decode("utf-8") == "new train"
        assert zf.read(val_name).decode("utf-8") == "new val"
        assert f"{source_root.name}/text_labels/train/shared.txt" not in names


def test_download_dataset_entry_skips_dataset_symlink_escape(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    (source_root / "images").mkdir(parents=True, exist_ok=True)
    (source_root / "images" / "ok.txt").write_text("ok", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (source_root / "images" / "escape.txt").symlink_to(outside)
    record_root = tmp_path / "registry" / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
        },
    )

    response = api.download_dataset_entry("ds_linked")

    with zipfile.ZipFile(Path(response.path), "r") as zf:
        names = set(zf.namelist())
        assert f"{source_root.name}/images/ok.txt" in names
        assert f"{source_root.name}/images/escape.txt" not in names


def test_download_dataset_entry_skips_overlay_symlink_escape(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    record_root = tmp_path / "registry" / "ds_linked"
    overlay_root = record_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (overlay_root / "labels" / "train" / "escape.txt").symlink_to(outside)

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
        },
    )

    response = api.download_dataset_entry("ds_linked")

    with zipfile.ZipFile(Path(response.path), "r") as zf:
        assert f"{source_root.name}/labels/escape.txt" not in set(zf.namelist())


def test_download_linked_dataset_rejects_symlinked_registry_labelmap(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "labelmap.txt").write_text("old\n", encoding="utf-8")
    record_root = tmp_path / "registry" / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_labelmap.txt"
    outside.write_text("secret\n", encoding="utf-8")
    registry_labelmap = record_root / "labelmap.txt"
    try:
        registry_labelmap.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
            "yolo_labelmap_path": str(registry_labelmap),
            "labelmap_source": "registry_overlay",
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.download_dataset_entry("ds_linked")

    assert exc.value.status_code == 400
    assert exc.value.detail == "labelmap_path_forbidden"


def test_set_dataset_glossary_for_linked_dataset_writes_registry_meta(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)

    record_root = tmp_path / "registry" / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    (record_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "ds_linked", "label": "ds_linked"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "classes": ["gas_tank"],
        },
    )

    default = api.get_dataset_glossary("ds_linked")

    assert default["classes"] == ["gas_tank"]
    assert "storage tank" in default["default_glossary"]

    out = api.set_dataset_glossary("ds_linked", '{"car": ["vehicle"]}')

    assert out["dataset_id"] == "ds_linked"
    meta = json.loads((record_root / api.DATASET_META_NAME).read_text(encoding="utf-8"))
    assert "labelmap_glossary" in meta
    assert not (source_root / api.DATASET_META_NAME).exists()


def test_set_dataset_glossary_rejects_symlinked_registry_parent_without_write(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    outside_registry = tmp_path / "outside_registry"
    outside_registry.mkdir()
    linked_registry = tmp_path / "linked_registry"
    try:
        linked_registry.symlink_to(outside_registry, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(linked_registry / "ds_linked"),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "classes": ["gas_tank"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.set_dataset_glossary("ds_linked", '{"car": ["vehicle"]}')

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_metadata_path_forbidden"
    assert list(outside_registry.iterdir()) == []


def test_set_dataset_glossary_rejects_symlinked_metadata_without_target_write(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    record_root = tmp_path / "registry" / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    outside_meta = tmp_path / "outside_meta.json"
    outside_meta.write_text('{"secret": true}', encoding="utf-8")
    try:
        (record_root / api.DATASET_META_NAME).symlink_to(outside_meta)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "classes": ["gas_tank"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.set_dataset_glossary("ds_linked", '{"car": ["vehicle"]}')

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_metadata_path_forbidden"
    assert outside_meta.read_text(encoding="utf-8") == '{"secret": true}'


def _entry_for_annotation(tmp_path: Path) -> dict:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    for relpath in (
        "img.jpg",
        "img_ok.jpg",
        "nested/img.jpg",
        "sub/img.jpg",
        "sub_a/img.jpg",
        "sub_b/img.png",
    ):
        _write_test_image(dataset_root / "images" / relpath)
    registry_root = tmp_path / "registry" / "ds"
    registry_root.mkdir(parents=True, exist_ok=True)
    return {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "classes": ["car"],
    }


def _split_entry_for_annotation(tmp_path: Path) -> dict:
    dataset_root = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)
        _write_test_image(dataset_root / split / "images" / "shared.jpg")
    registry_root = tmp_path / "registry" / "ds"
    registry_root.mkdir(parents=True, exist_ok=True)
    return {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "split",
        "classes": ["car"],
    }


def _active_lock(session_id: str = "sess-1") -> dict:
    return {
        "holder": "annotator",
        "session_id": session_id,
        "expires_at": time.time() + 300.0,
    }


def _write_test_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(20, 40, 80))
    image.save(path)


def test_manifest_does_not_expose_meta_path(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    manifest = api.get_dataset_annotation_manifest("ds")
    assert "meta_path" not in manifest


def test_persistent_snapshot_requires_active_lock(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {"records": [{"split": "train", "image_relpath": "img.jpg", "label_lines": []}]},
        )
    assert exc.value.status_code == 409
    assert exc.value.detail == "annotation_lock_session_required"

    with pytest.raises(api.HTTPException) as exc2:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "wrong",
                "records": [{"split": "train", "image_relpath": "img.jpg", "label_lines": []}],
            },
        )
    assert exc2.value.status_code == 409
    assert exc2.value.detail == "annotation_lock_active"


def test_persistent_meta_patch_requires_lock_owner(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.patch_dataset_annotation_meta("ds", {"status": "in_progress"})
    assert exc.value.status_code == 409
    assert exc.value.detail == "annotation_lock_session_required"


def test_persistent_snapshot_validates_status_before_overlay_writes(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "status": "not_a_status",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "img.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                        "text_label": "caption",
                    }
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_annotation_status"
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    assert not (overlay_root / "labels" / "train" / "img.txt").exists()
    assert not (overlay_root / "text_labels" / "img.txt").exists()
    assert "annotation_status" not in meta


def test_persistent_snapshot_normalises_all_records_before_overlay_writes(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "img_ok.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    },
                    {
                        "split": "train",
                        "image_relpath": "../bad.jpg",
                        "label_lines": [],
                    },
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_relative_path"
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    assert not (overlay_root / "labels" / "train" / "img_ok.txt").exists()


def test_persistent_snapshot_rejects_records_for_missing_images(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "missing.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    }
                ],
            },
        )

    assert exc.value.status_code == 404
    assert exc.value.detail == "annotation_image_not_found"
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    assert not (overlay_root / "labels" / "train" / "missing.txt").exists()


def test_persistent_snapshot_text_only_update_preserves_existing_label_overlay(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    label_path = overlay_root / "labels" / "train" / "img.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    out = api.save_dataset_annotation_snapshot(
        "ds",
        {
            "session_id": "sess-lock",
            "records": [
                {
                    "split": "train",
                    "image_relpath": "img.jpg",
                    "text_label": "new caption",
                }
            ],
        },
    )

    assert out["status"] == "saved"
    assert label_path.read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"
    text_path = overlay_root / "text_labels" / "img.txt"
    assert text_path.read_text(encoding="utf-8") == "new caption"


def test_split_snapshot_text_overlays_are_split_scoped(tmp_path, monkeypatch) -> None:
    entry = _split_entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    out = api.save_dataset_annotation_snapshot(
        "ds",
        {
            "session_id": "sess-lock",
            "records": [
                {
                    "split": "train",
                    "image_relpath": "shared.jpg",
                    "text_label": "train caption",
                },
                {
                    "split": "val",
                    "image_relpath": "shared.jpg",
                    "text_label": "val caption",
                },
            ],
        },
    )

    assert out["status"] == "saved"
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    assert (
        overlay_root / "text_labels" / "train" / "shared.txt"
    ).read_text(encoding="utf-8") == "train caption"
    assert (
        overlay_root / "text_labels" / "val" / "shared.txt"
    ).read_text(encoding="utf-8") == "val caption"
    assert not (overlay_root / "text_labels" / "shared.txt").exists()

    manifest = api.get_dataset_annotation_manifest("ds")
    captions = {
        (row["split"], row["image_relpath"]): row["text_label"]
        for row in manifest["images"]
    }
    assert captions[("train", "shared.jpg")] == "train caption"
    assert captions[("val", "shared.jpg")] == "val caption"


def test_persistent_snapshot_replaces_overlay_file_symlink_without_touching_target(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    label_dir = overlay_root / "labels" / "train"
    label_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (label_dir / "img.txt").symlink_to(outside)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    out = api.save_dataset_annotation_snapshot(
        "ds",
        {
            "session_id": "sess-lock",
            "records": [
                {
                    "split": "train",
                    "image_relpath": "img.jpg",
                    "label_lines": ["0 0.5 0.5 0.1 0.1"],
                }
            ],
        },
    )

    assert out["status"] == "saved"
    assert outside.read_text(encoding="utf-8") == "secret"
    label_path = label_dir / "img.txt"
    assert not label_path.is_symlink()
    assert label_path.read_text(encoding="utf-8").strip() == "0 0.5 0.5 0.1 0.1"


def test_persistent_snapshot_rejects_overlay_parent_symlink_escape(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    label_parent = overlay_root / "labels"
    label_parent.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside_labels"
    outside_dir.mkdir()
    (label_parent / "train").symlink_to(outside_dir, target_is_directory=True)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "img.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    }
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "annotation_overlay_path_forbidden"
    assert not (outside_dir / "img.txt").exists()


def test_annotation_overlay_text_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
) -> None:
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    overlay_root = tmp_path / "overlay"
    overlay_root.mkdir()
    label_path = overlay_root / "labels" / "train" / "img.txt"
    label_path.parent.mkdir(parents=True)
    tmp_path_link = label_path.with_suffix(f"{label_path.suffix}.{FixedUUID.hex}.tmp")
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final = tmp_path / "outside_final.txt"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    try:
        tmp_path_link.symlink_to(outside_tmp)
        label_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: FixedUUID())

    api._annotation_write_text_within_root(
        label_path,
        overlay_root,
        "0 0.5 0.5 0.1 0.1\n",
    )

    assert not tmp_path_link.exists()
    assert not label_path.is_symlink()
    assert label_path.read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_persistent_snapshot_rejects_overlay_ancestor_symlink_escape(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    overlay_root.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside_labels"
    outside_dir.mkdir()
    try:
        (overlay_root / "labels").symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "nested/img.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    }
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "annotation_overlay_path_forbidden"
    assert list(outside_dir.iterdir()) == []


def test_persistent_snapshot_rejects_symlinked_registry_parent_before_overlay_creation(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    outside_registry = tmp_path / "outside_registry"
    outside_registry.mkdir()
    linked_parent = tmp_path / "linked_registry"
    try:
        linked_parent.symlink_to(outside_registry, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    entry["registry_root"] = str(linked_parent / "nested" / "ds")
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "nested/img.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    }
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "annotation_overlay_path_forbidden"
    assert not (outside_registry / "nested").exists()


def test_annotation_manifest_ignores_overlay_and_source_symlink_escapes(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    _write_test_image(Path(entry["dataset_root"]) / "images" / "img.jpg")
    outside_label = tmp_path / "outside_label.txt"
    outside_label.write_text("0 0.9 0.9 0.1 0.1\n", encoding="utf-8")
    outside_text = tmp_path / "outside_text.txt"
    outside_text.write_text("secret caption", encoding="utf-8")
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    overlay_label_dir = overlay_root / "labels" / "train"
    overlay_label_dir.mkdir(parents=True, exist_ok=True)
    (overlay_label_dir / "img.txt").symlink_to(outside_label)
    text_dir = Path(entry["dataset_root"]) / "text_labels"
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "img.txt").symlink_to(outside_text)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    manifest = api.get_dataset_annotation_manifest("ds")

    assert manifest["images"][0]["label_lines"] == []
    assert manifest["images"][0]["text_label"] == ""


def test_transient_annotation_manifest_ignores_source_label_symlink_escape(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labelmap.txt").write_text("car\n", encoding="utf-8")
    outside_label = tmp_path / "outside_label.txt"
    outside_label.write_text("0 0.9 0.9 0.1 0.1\n", encoding="utf-8")
    try:
        (dataset_root / "labels" / "img.txt").symlink_to(outside_label)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])

    opened = api.open_dataset_path(str(dataset_root), strict=True)
    session_id = opened["session_id"]
    try:
        manifest = api.get_transient_annotation_manifest(session_id)
    finally:
        with api.DATASET_TRANSIENT_LOCK:
            api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)

    assert manifest["images"][0]["label_lines"] == []


def test_resolve_annotation_image_path_rejects_symlink_escape(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    images_root = dataset_root / "images"
    images_root.mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"not really an image")
    (images_root / "escape.jpg").symlink_to(outside)

    with pytest.raises(api.HTTPException) as exc:
        api._resolve_annotation_image_path(
            dataset_root,
            "flat",
            "train",
            Path("escape.jpg"),
        )

    assert exc.value.status_code == 404
    assert exc.value.detail == "annotation_image_not_found"


def test_persistent_meta_patch_validates_status_before_labelmap_write(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.patch_dataset_annotation_meta(
            "ds",
            {
                "session_id": "sess-lock",
                "status": "not_a_status",
                "labelmap": ["car", "truck"],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_annotation_status"
    assert not (Path(entry["dataset_root"]) / "labelmap.txt").exists()
    assert "annotation_status" not in meta


def test_persistent_meta_patch_rejects_labelmap_path_outside_dataset_roots(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    entry["storage_mode"] = "managed"
    entry["linked_root"] = None
    entry["registry_root"] = None
    outside = tmp_path / "outside" / "labelmap.txt"
    entry["yolo_labelmap_path"] = str(outside)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.patch_dataset_annotation_meta(
            "ds",
            {
                "session_id": "sess-lock",
                "labelmap": ["car", "truck"],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "labelmap_path_forbidden"
    assert not outside.exists()
    assert set(meta) == {"annotation_lock"}


def test_persistent_meta_patch_linked_labelmap_stays_in_registry(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    dataset_root = Path(entry["dataset_root"])
    registry_root = Path(entry["registry_root"])
    source_labelmap = dataset_root / "labelmap.txt"
    source_labelmap.write_text("car\n", encoding="utf-8")
    meta = {
        "id": "ds",
        "label": "ds",
        "annotation_lock": _active_lock("sess-lock"),
        "classes": ["car"],
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_load_or_create_meta",
        lambda _entry: (registry_root / api.DATASET_META_NAME, meta),
    )

    out = api.patch_dataset_annotation_meta(
        "ds",
        {
            "session_id": "sess-lock",
            "labelmap": ["car", "truck"],
        },
    )

    registry_labelmap = registry_root / "labelmap.txt"
    saved_meta = json.loads((registry_root / api.DATASET_META_NAME).read_text(encoding="utf-8"))
    assert out["labelmap"] == ["car", "truck"]
    assert source_labelmap.read_text(encoding="utf-8") == "car\n"
    assert registry_labelmap.read_text(encoding="utf-8") == "car\ntruck\n"
    assert saved_meta["classes"] == ["car", "truck"]
    assert saved_meta["yolo_labelmap_path"] == str(registry_labelmap)
    assert saved_meta["labelmap_source"] == "registry_overlay"


def test_annotation_labelmap_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
) -> None:
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    labelmap_path = dataset_root / "labelmap.txt"
    tmp_path_link = labelmap_path.with_suffix(f"{labelmap_path.suffix}.{FixedUUID.hex}.tmp")
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final = tmp_path / "outside_final.txt"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    try:
        tmp_path_link.symlink_to(outside_tmp)
        labelmap_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: FixedUUID())

    api._annotation_write_labelmap_file(labelmap_path, [dataset_root.resolve()], ["car", "truck"])

    assert not tmp_path_link.exists()
    assert not labelmap_path.is_symlink()
    assert labelmap_path.read_text(encoding="utf-8") == "car\ntruck\n"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_persistent_meta_patch_rejects_symlinked_registry_parent_before_metadata_write(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    outside_registry = tmp_path / "outside_registry"
    outside_registry.mkdir()
    linked_parent = tmp_path / "linked_registry"
    try:
        linked_parent.symlink_to(outside_registry, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    entry["registry_root"] = str(linked_parent / "nested" / "ds")
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )

    with pytest.raises(api.HTTPException) as exc:
        api.patch_dataset_annotation_meta(
            "ds",
            {
                "session_id": "sess-lock",
                "notes": "blocked",
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_metadata_path_forbidden"
    assert not (outside_registry / "nested").exists()


def test_stop_session_requires_matching_session_or_force(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.stop_dataset_annotation_session("ds", {})
    assert exc.value.status_code == 409
    assert exc.value.detail == "annotation_lock_session_mismatch"

    out = api.stop_dataset_annotation_session("ds", {"force": True})
    assert out["status"] == "ok"
    assert meta.get("annotation_lock") == {}


def test_start_session_rejects_same_holder_name_with_different_session(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {
        "annotation_lock": {
            "holder": "webui:tab-a",
            "session_id": "sess-a",
            "expires_at": time.time() + 300.0,
        }
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    out = api.start_dataset_annotation_session(
        "ds",
        {"session_id": "sess-b", "editor_name": "webui:tab-a"},
    )

    assert out["status"] == "warning"
    assert out["warning"] == "annotation_lock_active"
    assert meta["annotation_lock"]["session_id"] == "sess-a"


def test_transient_snapshot_and_meta_patch_require_lock(monkeypatch) -> None:
    session_id = "transient-lock"
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": "/tmp/path",
            "overlay_labels": {},
            "overlay_text": {},
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }

    with pytest.raises(api.HTTPException) as exc:
        api.save_transient_annotation_snapshot(session_id, {"records": []})
    assert exc.value.status_code == 409
    assert exc.value.detail == "annotation_lock_session_required"

    with pytest.raises(api.HTTPException) as exc2:
        api.patch_transient_annotation_meta(session_id, {"session_id": "bad", "notes": "x"})
    assert exc2.value.status_code == 409
    assert exc2.value.detail == "annotation_lock_active"

    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_snapshot_validates_status_before_overlay_mutation(monkeypatch) -> None:
    session_id = "transient-invalid-status"
    now = time.time()
    overlay_labels = {}
    overlay_text = {}
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": "/tmp/path",
            "overlay_labels": overlay_labels,
            "overlay_text": overlay_text,
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }

    with pytest.raises(api.HTTPException) as exc:
        api.save_transient_annotation_snapshot(
            session_id,
            {
                "session_id": "sess-lock",
                "status": "not_a_status",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "img.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                        "text_label": "caption",
                    }
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_annotation_status"
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["overlay_labels"] == {}
        assert session["overlay_text"] == {}
        assert "annotation_status" not in session
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_snapshot_rejects_records_for_missing_images(tmp_path, monkeypatch) -> None:
    session_id = "transient-missing-image"
    source_root = tmp_path / "source"
    (source_root / "images").mkdir(parents=True, exist_ok=True)
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": str(source_root),
            "yolo_layout": "flat",
            "overlay_labels": {},
            "overlay_text": {},
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }
    monkeypatch.setattr(api, "_validate_linked_dataset_path", lambda _path: source_root)

    with pytest.raises(api.HTTPException) as exc:
        api.save_transient_annotation_snapshot(
            session_id,
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "missing.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    }
                ],
            },
        )

    assert exc.value.status_code == 404
    assert exc.value.detail == "annotation_image_not_found"
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["overlay_labels"] == {}
        assert session["overlay_text"] == {}
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_meta_patch_validates_status_before_labelmap_write(
    tmp_path, monkeypatch
) -> None:
    session_id = "transient-invalid-meta"
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": str(source_root),
            "classes": ["old"],
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }

    with pytest.raises(api.HTTPException) as exc:
        api.patch_transient_annotation_meta(
            session_id,
            {
                "session_id": "sess-lock",
                "status": "not_a_status",
                "labelmap": ["car", "truck"],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_annotation_status"
    assert not (source_root / "labelmap.txt").exists()
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["classes"] == ["old"]
        assert "annotation_status" not in session
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_meta_patch_does_not_touch_forbidden_source_root(
    tmp_path, monkeypatch
) -> None:
    session_id = "transient-forbidden-root"
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": str(source_root),
            "classes": ["old"],
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }

    def reject_linked_path(_path: str) -> Path:
        raise api.HTTPException(status_code=400, detail="dataset_path_not_allowlisted")

    monkeypatch.setattr(api, "_validate_linked_dataset_path", reject_linked_path)

    out = api.patch_transient_annotation_meta(
        session_id,
        {
            "session_id": "sess-lock",
            "labelmap": ["car", "truck"],
        },
    )

    assert out["labelmap"] == ["car", "truck"]
    assert not (source_root / "labelmap.txt").exists()
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["classes"] == ["car", "truck"]
        assert session["labelmap_source"] == "transient_session"
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_meta_patch_keeps_labelmap_in_session_memory(
    tmp_path, monkeypatch
) -> None:
    session_id = "transient-allowed-root"
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": str(source_root),
            "classes": ["old"],
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }
    out = api.patch_transient_annotation_meta(
        session_id,
        {
            "session_id": "sess-lock",
            "labelmap": ["car", "truck"],
        },
    )

    assert out["labelmap"] == ["car", "truck"]
    assert not (source_root / "labelmap.txt").exists()
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["classes"] == ["car", "truck"]
        assert session["labelmap_source"] == "transient_session"
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_meta_patch_never_replaces_source_labelmap_symlink(
    tmp_path, monkeypatch
) -> None:
    session_id = "transient-labelmap-symlink"
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    outside_final = tmp_path / "outside_labelmap.txt"
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final.write_text("external final", encoding="utf-8")
    outside_tmp.write_text("external tmp", encoding="utf-8")
    labelmap_path = source_root / "labelmap.txt"
    tmp_link = labelmap_path.with_suffix(f"{labelmap_path.suffix}.feedface.tmp")
    try:
        labelmap_path.symlink_to(outside_final)
        tmp_link.symlink_to(outside_tmp)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": str(source_root),
            "classes": ["old"],
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }
    out = api.patch_transient_annotation_meta(
        session_id,
        {
            "session_id": "sess-lock",
            "labelmap": ["car", "truck"],
        },
    )

    assert out["labelmap"] == ["car", "truck"]
    assert labelmap_path.is_symlink()
    assert tmp_link.is_symlink()
    assert outside_final.read_text(encoding="utf-8") == "external final"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["classes"] == ["car", "truck"]
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_delete_and_expiry(monkeypatch) -> None:
    active_id = "transient-active"
    expired_id = "transient-expired"
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[active_id] = {
            "session_id": active_id,
            "expires_at": now + 300.0,
        }
        api.DATASET_TRANSIENT_SESSIONS[expired_id] = {
            "session_id": expired_id,
            "expires_at": now - 1.0,
        }

    with pytest.raises(api.HTTPException) as exc:
        api.delete_transient_dataset(expired_id)
    assert exc.value.status_code == 410
    assert exc.value.detail == "transient_session_expired"
    out = api.delete_transient_dataset(active_id)
    assert out["status"] == "deleted"

    with api.DATASET_TRANSIENT_LOCK:
        assert active_id not in api.DATASET_TRANSIENT_SESSIONS
        assert expired_id not in api.DATASET_TRANSIENT_SESSIONS


def test_text_overlay_paths_keep_nested_relpaths(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    api.save_dataset_annotation_snapshot(
        "ds",
        {
            "session_id": "sess-lock",
            "records": [
                {
                    "split": "train",
                    "image_relpath": "sub_a/img.jpg",
                    "label_lines": [],
                    "text_label": "A",
                },
                {
                    "split": "train",
                    "image_relpath": "sub_b/img.png",
                    "label_lines": [],
                    "text_label": "B",
                },
            ],
        },
    )

    overlay_root = (
        Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "text_labels"
    )
    assert (overlay_root / "sub_a" / "img.txt").read_text(encoding="utf-8") == "A"
    assert (overlay_root / "sub_b" / "img.txt").read_text(encoding="utf-8") == "B"
    assert not (overlay_root / "img.txt").exists()


def test_effective_text_label_falls_back_to_legacy_flat_text_labels(tmp_path) -> None:
    entry = _entry_for_annotation(tmp_path)
    dataset_root = Path(entry["dataset_root"])
    text_root = dataset_root / "text_labels"
    text_root.mkdir(parents=True, exist_ok=True)
    (text_root / "img.txt").write_text("legacy", encoding="utf-8")
    value = api._annotation_effective_text_label(entry, Path("sub_a/img.jpg"))
    assert value == "legacy"


def test_get_text_labels_batch_reads_overlay_source_and_legacy(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    dataset_root = Path(entry["dataset_root"])
    source_text = dataset_root / "text_labels" / "sub_a"
    source_text.mkdir(parents=True, exist_ok=True)
    (source_text / "img.txt").write_text("source caption", encoding="utf-8")
    legacy_text = dataset_root / "text_labels"
    legacy_text.mkdir(parents=True, exist_ok=True)
    (legacy_text / "legacy.txt").write_text("legacy caption", encoding="utf-8")
    overlay_root = (
        Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "text_labels"
    )
    (overlay_root / "sub_b").mkdir(parents=True, exist_ok=True)
    (overlay_root / "sub_b" / "img.txt").write_text("overlay caption", encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    result = api.get_text_labels(
        "ds",
        ["sub_a/img.jpg", "sub_b/img.png", "nested/legacy.jpg", "missing.jpg"],
    )

    assert result["captions"] == {
        "sub_a/img.jpg": "source caption",
        "sub_b/img.png": "overlay caption",
        "nested/legacy.jpg": "legacy caption",
    }
    assert result["missing"] == ["missing.jpg"]


def test_text_label_endpoints_read_split_prefixed_source_captions(
    tmp_path, monkeypatch
) -> None:
    entry = _split_entry_for_annotation(tmp_path)
    dataset_root = Path(entry["dataset_root"])
    train_text = dataset_root / "train" / "text_labels"
    val_text = dataset_root / "val" / "text_labels"
    train_text.mkdir(parents=True, exist_ok=True)
    val_text.mkdir(parents=True, exist_ok=True)
    (train_text / "shared.txt").write_text("train source caption", encoding="utf-8")
    (val_text / "shared.txt").write_text("val source caption", encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    assert api.get_text_label("ds", "train/shared.jpg") == {
        "caption": "train source caption",
    }
    result = api.get_text_labels(
        "ds",
        ["train/shared.jpg", "val/shared.jpg", "shared.jpg"],
    )

    assert result["captions"] == {
        "train/shared.jpg": "train source caption",
        "val/shared.jpg": "val source caption",
    }
    assert result["missing"] == ["shared.jpg"]


def test_text_label_routes_accept_encoded_split_prefixed_image_names(
    tmp_path, monkeypatch
) -> None:
    entry = _split_entry_for_annotation(tmp_path)
    dataset_root = Path(entry["dataset_root"])
    train_text = dataset_root / "train" / "text_labels"
    train_text.mkdir(parents=True, exist_ok=True)
    (train_text / "shared.txt").write_text("train source caption", encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    client = TestClient(api.app)
    response = client.get("/datasets/ds/text_labels/train%2Fshared.jpg")

    assert response.status_code == 200
    assert response.json() == {"caption": "train source caption"}

    response = client.post(
        "/datasets/ds/text_labels/val%2Fshared.jpg",
        json={"caption": "new val caption"},
    )

    assert response.status_code == 200
    assert response.json()["caption"] == "new val caption"
    assert (
        Path(entry["registry_root"])
        / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
        / "text_labels"
        / "val"
        / "shared.txt"
    ).read_text(encoding="utf-8") == "new val caption"


def test_caption_alternate_routes_append_update_export_and_delete(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    client = TestClient(api.app)

    response = client.post(
        "/datasets/ds/captions/sub%2Fimg.jpg",
        json={"caption": "primary caption", "source": "manual"},
    )
    assert response.status_code == 200
    first = response.json()["caption"]
    assert first["caption"] == "primary caption"
    assert first["is_primary"] is True
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    assert (overlay_root / "text_labels" / "sub" / "img.txt").read_text(encoding="utf-8") == "primary caption"

    response = client.post(
        "/datasets/ds/captions/sub%2Fimg.jpg",
        json={"caption": "alternate caption", "source": "qwen_caption_job"},
    )
    assert response.status_code == 200
    second = response.json()["caption"]
    assert second["caption"] == "alternate caption"
    assert second["is_primary"] is False

    response = client.get("/datasets/ds/captions/sub%2Fimg.jpg")
    assert response.status_code == 200
    bundle = response.json()
    assert bundle["primary_caption"] == "primary caption"
    assert [item["caption"] for item in bundle["captions"]] == [
        "primary caption",
        "alternate caption",
    ]

    response = client.patch(
        f"/datasets/ds/captions/by_id/{second['id']}",
        json={"caption": "promoted alternate", "make_primary": True},
    )
    assert response.status_code == 200
    assert response.json()["caption"]["is_primary"] is True
    assert (overlay_root / "text_labels" / "sub" / "img.txt").read_text(encoding="utf-8") == "promoted alternate"

    response = client.patch(
        f"/datasets/ds/captions/by_id/{second['id']}",
        json={"caption": "edited promoted alternate"},
    )
    assert response.status_code == 200
    assert response.json()["caption"]["is_primary"] is True
    assert (overlay_root / "text_labels" / "sub" / "img.txt").read_text(encoding="utf-8") == "edited promoted alternate"

    response = client.post(
        "/datasets/ds/captions/sub%2Fimg.jpg",
        json={"caption": "third alternate", "source": "manual"},
    )
    assert response.status_code == 200
    third = response.json()["caption"]
    assert third["caption"] == "third alternate"
    assert third["is_primary"] is False

    response = client.get("/datasets/ds/captions/export")
    assert response.status_code == 200
    export_payload = response.json()
    assert export_payload["summary"] == {
        "image_count": 1,
        "caption_count": 3,
        "training_row_count": 3,
        "images_with_multiple_captions": 1,
        "max_captions_per_image": 3,
    }
    exported = export_payload["captions"]
    assert any(item["caption"] == "primary caption" for item in exported)
    assert any(item["caption"] == "edited promoted alternate" for item in exported)
    exported_for_image = [item for item in exported if item["image_name"] == "sub/img.jpg"]
    assert [item["caption"] for item in exported_for_image] == [
        "edited promoted alternate",
        "primary caption",
        "third alternate",
    ]
    assert [item["caption_index"] for item in exported_for_image] == [1, 2, 3]
    assert exported_for_image[0]["is_primary"] is True
    assert [item["caption"] for item in export_payload["grouped"]["sub/img.jpg"]] == [
        "edited promoted alternate",
        "primary caption",
        "third alternate",
    ]
    archive = export_payload["archive"]
    assert archive["format"] == "tator_caption_grouped_v1"
    assert archive["dataset_id"] == "ds"
    assert archive["image_count"] == 1
    assert archive["caption_count"] == 3
    assert archive["summary"] == export_payload["summary"]
    archived_image = next(item for item in archive["images"] if item["image_name"] == "sub/img.jpg")
    assert archived_image["caption_count"] == 3
    assert archived_image["primary_caption"] == "edited promoted alternate"
    assert [item["caption"] for item in archived_image["captions"]] == [
        "edited promoted alternate",
        "primary caption",
        "third alternate",
    ]
    assert [item["caption_index"] for item in archived_image["captions"]] == [1, 2, 3]
    assert archived_image["captions"][0]["is_primary"] is True
    assert archived_image["captions"][0]["caption_source"] == "qwen_caption_job"
    assert archived_image["captions"][2]["caption_source"] == "manual"
    assert "metadata" in archived_image["captions"][0]
    training_rows = export_payload["training_rows"]
    training_rows_for_image = [item for item in training_rows if item["image_path"] == "sub/img.jpg"]
    assert [json.loads(item["answer"]) for item in training_rows_for_image] == [
        {"caption": "edited promoted alternate"},
        {"caption": "primary caption"},
        {"caption": "third alternate"},
    ]
    assert len({item["question"] for item in training_rows_for_image}) == 3
    assert training_rows_for_image[0]["metadata"]["row_type"] == "caption0"
    assert training_rows_for_image[1]["metadata"]["row_type"] == "alternate_caption"
    assert training_rows_for_image[0]["metadata"]["caption_index"] == 1
    assert training_rows_for_image[0]["metadata"]["dataset_id"] == "ds"
    assert export_payload["instruction_summary"] == {
        "instruction_training_row_count": 1,
        "generated_qa_pair_count": 0,
        "deterministic_metadata_qa_pair_count": 0,
        "instruction_review_row_count": 1,
        "manual_review_required_count": 1,
        "training_readiness_status": "needs_review",
        "instruction_export_validation_ok": True,
        "instruction_export_validation_error_count": 0,
        "instruction_artifact_consistency_ok": True,
        "instruction_artifact_consistency_error_count": 0,
        "rejected_training_row_count": 0,
    }
    instruction_archive = export_payload["instruction_archive"]
    assert instruction_archive["format"] == "tator_caption_instruction_archive_v1"
    assert instruction_archive["settings"] == {
        "include_caption0_in_training": True,
        "include_generated_qa_in_training": True,
        "include_deterministic_metadata_qa": False,
        "qa_mix": "balanced",
        "answer_format": "natural",
    }
    assert instruction_archive["training_row_count"] == 1
    assert instruction_archive["rejection_reason_counts"] == {}
    assert len(export_payload["instruction_archive_rows"]) == instruction_archive["image_count"]
    assert len(export_payload["instruction_review_rows"]) == 1
    assert export_payload["instruction_report"]["instruction_review_row_count"] == 1
    assert export_payload["instruction_export_validation"]["ok"] is True
    assert export_payload["instruction_artifact_consistency"]["ok"] is True
    assert export_payload["instruction_artifact_consistency"] == instruction_archive["instruction_artifact_consistency"]
    assert export_payload["instruction_report"]["instruction_export_validation"]["ok"] is True
    assert export_payload["instruction_report"]["instruction_artifact_consistency"]["ok"] is True
    assert export_payload["instruction_report"]["training_readiness"]["status"] == "needs_review"
    assert (
        export_payload["instruction_report"]["training_readiness"]["instruction_export_validation_error_count"]
        == 0
    )
    assert (
        export_payload["instruction_report"]["training_readiness"]["instruction_artifact_consistency_error_count"]
        == 0
    )
    assert export_payload["instruction_report"]["training_readiness"]["pending_manual_review_row_count"] == 1
    strict_export_response = client.get("/datasets/ds/captions/export?require_ready_instruction_export=true")
    assert strict_export_response.status_code == 409
    assert "instruction_export_not_ready:needs_review" in strict_export_response.text
    assert all(
        row["format"] == "tator_caption_instruction_review_rows_v1"
        and "review_decision" in row
        and "review_notes" in row
        for row in export_payload["instruction_review_rows"]
    )
    archive_row_by_path = {
        row["image_path"]: row
        for row in export_payload["instruction_archive_rows"]
    }
    assert archive_row_by_path["sub/img.jpg"]["source_annotations"]["format"] == "tator_source_annotations_v1"
    assert "export_metadata" in archive_row_by_path["sub/img.jpg"]
    assert not any(
        row["source_annotations"]["status"] == "source_manifest_row_missing"
        for row in export_payload["instruction_archive_rows"]
    )
    assert export_payload["instruction_report"]["format"] == "tator_caption_instruction_report_v1"
    assert export_payload["instruction_report"]["image_count"] == instruction_archive["image_count"]
    assert export_payload["instruction_report"]["selected_flattened_row_count"] == 1
    assert len(export_payload["instruction_training_rows"]) == 1
    assert {row["metadata"]["row_type"] for row in export_payload["instruction_training_rows"]} == {
        "caption0"
    }
    assert all(
        row["metadata"]["source_archive"] == "tator_caption_instruction_archive_v1"
        for row in export_payload["instruction_training_rows"]
    )
    selected_review_row = next(row for row in export_payload["instruction_review_rows"] if row["selected_for_training"])
    mismatched_review_row = {
        **selected_review_row,
        "dataset_id": "other-dataset",
        "review_decision": "accepted",
        "review_notes": "wrong dataset should fail closed",
    }
    mismatched_review_response = client.post(
        "/datasets/ds/captions/instruction_review",
        json={"rows": [mismatched_review_row]},
    )
    assert mismatched_review_response.status_code == 400
    assert "review_rows_dataset_id_mismatch:row_1:other-dataset!=ds" in mismatched_review_response.text
    malformed_review_response = client.post(
        "/datasets/ds/captions/instruction_review",
        json=123,
    )
    assert malformed_review_response.status_code == 400
    assert "review_rows_list_required" in malformed_review_response.text
    array_review_row = {
        **selected_review_row,
        "review_decision": "accepted",
        "review_notes": "route-level array review import",
    }
    array_review_response = client.post(
        "/datasets/ds/captions/instruction_review",
        json=[array_review_row],
    )
    assert array_review_response.status_code == 200
    assert array_review_response.json()["applied_count"] == 1
    selected_review_row = {
        **selected_review_row,
        "review_decision": "accepted",
        "review_notes": "route-level object review import",
    }
    review_response = client.post(
        "/datasets/ds/captions/instruction_review",
        json=selected_review_row,
    )
    assert review_response.status_code == 200
    review_result = review_response.json()
    assert review_result["applied_count"] == 1
    assert review_result["skipped_count"] == 0
    reviewed_export_response = client.get("/datasets/ds/captions/export")
    assert reviewed_export_response.status_code == 200
    reviewed_export_payload = reviewed_export_response.json()
    reviewed_row = next(
        row
        for row in reviewed_export_payload["instruction_review_rows"]
        if row["qa_id"] == selected_review_row["qa_id"]
    )
    assert reviewed_row["review_decision"] == "accepted"
    assert reviewed_row["review_notes"] == "route-level object review import"
    assert (
        reviewed_export_payload["instruction_report"]["training_readiness"]["accepted_manual_review_row_count"]
        >= 1
    )
    deterministic_response = client.get(
        "/datasets/ds/captions/export?include_caption0_in_training=false"
        "&include_generated_qa_in_training=false&include_deterministic_metadata_qa=true"
    )
    assert deterministic_response.status_code == 200
    deterministic_payload = deterministic_response.json()
    assert deterministic_payload["instruction_summary"]["instruction_training_row_count"] >= 0
    assert all(
        str(row["metadata"]["row_type"]).startswith("deterministic_")
        for row in deterministic_payload["instruction_training_rows"]
    )
    assert all(
        json.loads(row["answer"])
        for row in deterministic_payload["instruction_training_rows"]
    )

    response = client.delete(f"/datasets/ds/captions/by_id/{first['id']}")
    assert response.status_code == 200
    response = client.get("/datasets/ds/captions/sub%2Fimg.jpg")
    assert [item["caption"] for item in response.json()["captions"]] == [
        "edited promoted alternate",
        "third alternate",
    ]


def test_set_text_label_requires_active_lock_owner_when_locked(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {
        "id": "ds",
        "annotation_lock": _active_lock("sess-lock"),
    }
    meta_path = Path(entry["registry_root"]) / api.DATASET_META_NAME
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    with pytest.raises(api.HTTPException) as exc:
        api.set_text_label("ds", "sub/img.jpg", {"caption": "blocked"})
    assert exc.value.status_code == 409
    assert exc.value.detail == "annotation_lock_session_required"

    with pytest.raises(api.HTTPException) as exc2:
        api.set_text_label("ds", "sub/img.jpg", {"session_id": "wrong", "caption": "blocked"})
    assert exc2.value.status_code == 409
    assert exc2.value.detail == "annotation_lock_active"

    out = api.set_text_label(
        "ds",
        "sub/img.jpg",
        {"session_id": "sess-lock", "caption": "saved"},
    )

    assert out == {"status": "saved", "caption": "saved"}
    overlay_text = (
        Path(entry["registry_root"])
        / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
        / "text_labels"
        / "sub"
        / "img.txt"
    )
    assert overlay_text.read_text(encoding="utf-8") == "saved"


def test_caption_alternate_routes_require_active_lock_owner_when_locked(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {
        "id": "ds",
        "annotation_lock": _active_lock("sess-lock"),
    }
    meta_path = Path(entry["registry_root"]) / api.DATASET_META_NAME
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    client = TestClient(api.app)

    response = client.post(
        "/datasets/ds/captions/sub%2Fimg.jpg",
        json={"caption": "blocked"},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "annotation_lock_session_required"

    response = client.post(
        "/datasets/ds/captions/sub%2Fimg.jpg",
        json={"caption": "saved", "session_id": "sess-lock"},
    )
    assert response.status_code == 200
    caption_id = response.json()["caption"]["id"]

    response = client.patch(
        f"/datasets/ds/captions/by_id/{caption_id}",
        json={"caption": "wrong", "session_id": "wrong"},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "annotation_lock_active"

    response = client.patch(
        f"/datasets/ds/captions/by_id/{caption_id}",
        json={"caption": "updated", "session_id": "sess-lock"},
    )
    assert response.status_code == 200
    assert response.json()["caption"]["caption"] == "updated"

    response = client.delete(f"/datasets/ds/captions/by_id/{caption_id}")
    assert response.status_code == 409
    assert response.json()["detail"] == "annotation_lock_session_required"

    response = client.delete(
        f"/datasets/ds/captions/by_id/{caption_id}?session_id=sess-lock"
    )
    assert response.status_code == 200


def test_set_text_label_allows_direct_write_without_active_lock(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta_path = Path(entry["registry_root"]) / api.DATASET_META_NAME
    meta_path.write_text(json.dumps({"id": "ds", "annotation_lock": {}}), encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    out = api.set_text_label("ds", "img.jpg", {"caption": "direct caption"})

    assert out == {"status": "saved", "caption": "direct caption"}
    overlay_text = (
        Path(entry["registry_root"])
        / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
        / "text_labels"
        / "img.txt"
    )
    assert overlay_text.read_text(encoding="utf-8") == "direct caption"


def test_build_qwen_dataset_from_yolo_uses_flat_annotation_overlay_and_registry_glossary(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "ds", "labelmap_glossary": "new: overlay class"}),
        encoding="utf-8",
    )

    qwen_root = tmp_path / "qwen"
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "registry_root": str(registry_root),
            "storage_mode": "linked",
            "linked_root": str(dataset_root),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "classes": ["old", "new"],
            "context": "site imagery",
        },
    )

    out = api.build_qwen_dataset_from_yolo("ds")

    assert out["train_count"] == 1
    assert out["val_count"] == 0
    assert out["labelmap_glossary"] == "new: overlay class"
    ann_path = qwen_root / out["id"] / "train" / "annotations.jsonl"
    annotation = json.loads(ann_path.read_text(encoding="utf-8").strip())
    assert annotation["image"] == "nested/img.jpg"
    assert annotation["detections"][0]["label"] == "new"
    assert annotation["detections"][0]["bbox"] == [30, 30, 70, 70]
    assert (qwen_root / out["id"] / "train" / "nested" / "img.jpg").exists()


def test_qwen_dataset_text_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
) -> None:
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    qwen_root = tmp_path / "qwen_dataset"
    annotations_path = qwen_root / "train" / "annotations.jsonl"
    annotations_path.parent.mkdir(parents=True)
    tmp_path_link = annotations_path.with_suffix(
        f"{annotations_path.suffix}.{FixedUUID.hex}.tmp"
    )
    outside_tmp = tmp_path / "outside_tmp.jsonl"
    outside_final = tmp_path / "outside_final.jsonl"
    outside_tmp.write_text("external tmp\n", encoding="utf-8")
    outside_final.write_text("external final\n", encoding="utf-8")
    try:
        tmp_path_link.symlink_to(outside_tmp)
        annotations_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: FixedUUID())

    api._qwen_dataset_write_text_within_root(
        annotations_path,
        qwen_root,
        '{"image":"nested/img.jpg"}\n',
    )

    assert not tmp_path_link.exists()
    assert not annotations_path.is_symlink()
    assert annotations_path.read_text(encoding="utf-8") == '{"image":"nested/img.jpg"}\n'
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp\n"
    assert outside_final.read_text(encoding="utf-8") == "external final\n"


def test_build_qwen_dataset_from_yolo_rejects_symlinked_qwen_root_before_write(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")
    outside = tmp_path / "outside_qwen"
    outside.mkdir()
    qwen_root = tmp_path / "qwen"
    try:
        qwen_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "classes": ["building"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.build_qwen_dataset_from_yolo("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_target_invalid"
    assert list(outside.iterdir()) == []


def test_build_qwen_dataset_from_yolo_rejects_symlinked_qwen_root_parent_before_write(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")
    outside = tmp_path / "outside_qwen"
    outside.mkdir()
    link_parent = tmp_path / "linked_parent"
    try:
        link_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", link_parent / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "classes": ["building"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.build_qwen_dataset_from_yolo("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_target_invalid"
    assert list(outside.iterdir()) == []


def test_build_qwen_dataset_from_yolo_rejects_labelmap_path_outside_dataset_roots(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    outside_labelmap = tmp_path / "outside" / "labelmap.txt"
    outside_labelmap.parent.mkdir(parents=True, exist_ok=True)
    outside_labelmap.write_text("secret\n", encoding="utf-8")
    qwen_root = tmp_path / "qwen"
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(outside_labelmap),
            "classes": ["building"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.build_qwen_dataset_from_yolo("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "labelmap_path_forbidden"
    assert not qwen_root.exists()


def test_build_qwen_dataset_from_yolo_rejects_symlinked_labelmap_before_write(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    outside_labelmap = tmp_path / "outside_labelmap.txt"
    outside_labelmap.write_text("secret\n", encoding="utf-8")
    try:
        (dataset_root / "labelmap.txt").symlink_to(outside_labelmap)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    qwen_root = tmp_path / "qwen"
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "classes": ["building"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.build_qwen_dataset_from_yolo("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "labelmap_path_forbidden"
    assert not qwen_root.exists()


def test_build_qwen_dataset_from_yolo_rejects_symlinked_registry_root_before_read(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")
    outside_registry = tmp_path / "outside_registry"
    outside_registry.mkdir()
    linked_registry = tmp_path / "linked_registry"
    try:
        linked_registry.symlink_to(outside_registry, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    qwen_root = tmp_path / "qwen"
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "registry_root": str(linked_registry),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "classes": ["building"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.build_qwen_dataset_from_yolo("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_metadata_path_forbidden"
    assert not qwen_root.exists()
    assert list(outside_registry.iterdir()) == []


def test_resolve_sam3_dataset_meta_materializes_annotation_overlay_for_linked_flat_yolo(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["old", "new"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(api, "_resolve_sam3_or_qwen_dataset", lambda _dataset_id: dataset_root)

    meta = api._resolve_sam3_dataset_meta("ds")

    assert meta["source"] == "annotation_overlay"
    assert meta["source_dataset_root"] == str(dataset_root.resolve())
    materialized_root = Path(meta["dataset_root"])
    assert materialized_root == (
        registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "sam3_materialized"
    ).resolve()
    assert not (dataset_root / "train").exists()
    coco_train = json.loads(Path(meta["coco_train_json"]).read_text(encoding="utf-8"))
    assert coco_train["categories"][1]["name"] == "new"
    assert coco_train["images"][0]["file_name"] == "images/nested/img.jpg"
    assert coco_train["annotations"][0]["category_id"] == 2
    assert coco_train["annotations"][0]["bbox"] == [30.0, 30.0, 40.0, 40.0]
    assert (materialized_root / "train" / "images" / "nested" / "img.jpg").exists()


def test_resolve_sam3_dataset_meta_rejects_symlinked_materialized_root_without_target_delete(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "img.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    victim = overlay_root / "victim"
    victim.mkdir(parents=True)
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    try:
        (overlay_root / "sam3_materialized").symlink_to(victim, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["building"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(api, "_resolve_sam3_or_qwen_dataset", lambda _dataset_id: dataset_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api._resolve_sam3_dataset_meta("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sam3_materialize_path_invalid"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_materialized_dataset_text_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
) -> None:
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    materialized_root = tmp_path / "materialized"
    label_path = materialized_root / "train" / "labels" / "img.txt"
    label_path.parent.mkdir(parents=True)
    tmp_path_link = label_path.with_suffix(f"{label_path.suffix}.{FixedUUID.hex}.tmp")
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final = tmp_path / "outside_final.txt"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    try:
        tmp_path_link.symlink_to(outside_tmp)
        label_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: FixedUUID())

    api._materialized_dataset_write_text_within_root(
        label_path,
        materialized_root,
        "0 0.5 0.5 0.2 0.2\n",
        detail="sam3_materialize_path_invalid",
    )

    assert not tmp_path_link.exists()
    assert not label_path.is_symlink()
    assert label_path.read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_resolve_sam3_dataset_meta_rejects_symlinked_registry_root_before_materialize(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")
    outside_registry = tmp_path / "outside_registry"
    outside_registry.mkdir()
    linked_registry = tmp_path / "linked_registry"
    try:
        linked_registry.symlink_to(outside_registry, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(linked_registry),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["building"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(api, "_resolve_sam3_or_qwen_dataset", lambda _dataset_id: dataset_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api._resolve_sam3_dataset_meta("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sam3_materialize_path_invalid"
    assert list(outside_registry.iterdir()) == []


def test_reset_materialized_dataset_root_rejects_nested_symlinked_allowed_root_before_mkdir(
    tmp_path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        api._reset_materialized_dataset_root(
            linked_parent / "nested" / "cache" / "materialized",
            linked_parent / "nested" / "cache",
            detail="yolo_cache_path_invalid",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "yolo_cache_path_invalid"
    assert list(outside.iterdir()) == []


def test_reset_materialized_dataset_root_rejects_nested_symlinked_target_before_mkdir(
    tmp_path,
) -> None:
    allowed_root = tmp_path / "allowed"
    outside = tmp_path / "outside_target"
    allowed_root.mkdir()
    outside.mkdir()
    linked_child = allowed_root / "linked_child"
    try:
        linked_child.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        api._reset_materialized_dataset_root(
            linked_child / "nested" / "materialized",
            allowed_root,
            detail="sam3_materialize_path_invalid",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sam3_materialize_path_invalid"
    assert list(outside.iterdir()) == []


def test_resolve_yolo_training_dataset_materializes_annotation_overlay_for_linked_flat_yolo(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["old", "new"],
    }
    cache_root = tmp_path / "yolo_cache"
    prepared_root = cache_root / "ds_cache"
    monkeypatch.setattr(api, "YOLO_DATASET_CACHE_ROOT", cache_root)
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(
        api,
        "_yolo_training_dataset_base_resolver",
        lambda _payload: {
            "dataset_id": "ds",
            "dataset_root": str(dataset_root),
            "prepared_root": str(dataset_root),
            "cache_root": str(prepared_root),
            "yolo_ready": True,
            "yolo_layout": "flat",
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "source": "registry",
        },
    )

    out = api._resolve_yolo_training_dataset(api.YoloTrainRequest(dataset_id="ds", accept_tos=True))

    assert out["source"] == "annotation_overlay_cache"
    assert out["prepared_root"] == str(prepared_root.resolve())
    assert out["yolo_layout"] == "split"
    assert Path(out["yolo_labelmap_path"]).read_text(encoding="utf-8") == "old\nnew\n"
    label_path = prepared_root / "train" / "labels" / "nested" / "img.txt"
    assert label_path.read_text(encoding="utf-8") == "1 0.5 0.5 0.4 0.4\n"
    assert (prepared_root / "train" / "images" / "nested" / "img.jpg").exists()
    assert not (dataset_root / "train").exists()


def test_resolve_yolo_training_dataset_rejects_symlinked_cache_materialization_target(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "img.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["building"],
    }
    cache_root = tmp_path / "yolo_cache"
    victim = cache_root / "victim"
    victim.mkdir(parents=True)
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    prepared_root = cache_root / "ds_cache"
    try:
        prepared_root.symlink_to(victim, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "YOLO_DATASET_CACHE_ROOT", cache_root)
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(
        api,
        "_yolo_training_dataset_base_resolver",
        lambda _payload: {
            "dataset_id": "ds",
            "dataset_root": str(dataset_root),
            "prepared_root": str(dataset_root),
            "cache_root": str(prepared_root),
            "yolo_ready": True,
            "yolo_layout": "flat",
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "source": "registry",
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api._resolve_yolo_training_dataset(api.YoloTrainRequest(dataset_id="ds", accept_tos=True))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "yolo_cache_path_invalid"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_resolve_rfdetr_training_dataset_materializes_annotation_overlay_for_linked_flat_yolo(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["old", "new"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])

    out = api._resolve_rfdetr_training_dataset(
        api.RfDetrTrainRequest(dataset_id="ds", accept_tos=True)
    )

    assert out["source"] == "annotation_overlay"
    materialized_root = Path(out["dataset_root"])
    assert materialized_root == (
        registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "sam3_materialized"
    ).resolve()
    coco_train = json.loads(Path(out["coco_train_json"]).read_text(encoding="utf-8"))
    assert coco_train["annotations"][0]["category_id"] == 2
    assert coco_train["annotations"][0]["bbox"] == [30.0, 30.0, 40.0, 40.0]
    assert not (dataset_root / "train").exists()


def test_prompt_helper_suggest_materializes_annotation_overlay_for_linked_flat_yolo(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["old", "new"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])

    out = api.prompt_helper_suggest(
        api.PromptHelperSuggestRequest(dataset_id="ds", max_synonyms=0, use_qwen=False)
    )

    assert any(row["class_id"] == 2 and row["class_name"] == "new" for row in out["classes"])
    materialized_root = (
        registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "sam3_materialized"
    )
    coco_train = json.loads(
        (materialized_root / "train" / "_annotations.coco.json").read_text(encoding="utf-8")
    )
    assert coco_train["annotations"][0]["category_id"] == 2
    assert coco_train["annotations"][0]["bbox"] == [30.0, 30.0, 40.0, 40.0]
    assert not (dataset_root / "train").exists()


def test_plan_segmentation_build_materializes_annotation_overlay_for_linked_flat_yolo(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["old", "new"],
    }
    sam3_root = tmp_path / "sam3_outputs"
    seg_root = tmp_path / "seg_jobs"
    sam3_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", sam3_root)
    monkeypatch.setattr(api, "SEG_BUILDER_ROOT", seg_root)

    planned_meta, planned_layout = api._plan_segmentation_build(
        api.SegmentationBuildRequest(source_dataset_id="ds", output_name="ds_seg")
    )

    materialized_root = (
        registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "sam3_materialized"
    ).resolve()
    assert planned_meta["source_dataset_root"] == str(materialized_root)
    assert planned_meta["source_dataset_id"] == "ds"
    assert planned_meta["classes"] == ["old", "new"]
    assert Path(planned_layout["dataset_root"]) == (sam3_root / "ds_seg").resolve()
    assert (
        materialized_root / "train" / "labels" / "nested" / "img.txt"
    ).read_text(encoding="utf-8").strip() == "1 0.5 0.5 0.4 0.4"
    assert not (dataset_root / "train").exists()


def test_plan_segmentation_build_rejects_symlinked_sam3_output_parent(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "source_split"
    _write_test_image(dataset_root / "train" / "images" / "img.jpg")
    (dataset_root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "train" / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")
    outside = tmp_path / "outside_sam3_outputs"
    outside.mkdir()
    link_parent = tmp_path / "linked_parent"
    try:
        link_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "storage_mode": "managed",
        "yolo_layout": "split",
        "yolo_ready": True,
        "classes": ["building"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", link_parent / "sam3_outputs")

    with pytest.raises(api.HTTPException) as exc_info:
        api._plan_segmentation_build(
            api.SegmentationBuildRequest(source_dataset_id="ds", output_name="ds_seg")
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "segmentation_output_path_invalid"
    assert list(outside.iterdir()) == []


def test_prepare_segmentation_output_root_rejects_symlink_swap(
    tmp_path, monkeypatch
) -> None:
    sam3_root = tmp_path / "sam3_outputs"
    sam3_root.mkdir()
    outside = tmp_path / "outside_output"
    outside.mkdir()
    try:
        (sam3_root / "ds_seg").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", sam3_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api._prepare_segmentation_output_root(
            {"id": "ds_seg"},
            {"dataset_root": str(sam3_root / "ds_seg")},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "segmentation_output_path_invalid"
    assert list(outside.iterdir()) == []


def test_write_segmentation_output_labelmap_skips_source_symlink_escape(
    tmp_path,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    dataset_root = tmp_path / "source"
    dataset_root.mkdir()
    outside = tmp_path / "outside_labelmap.txt"
    try:
        (dataset_root / "labelmap.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    api._write_segmentation_output_labelmap(
        output_root=output_root,
        dataset_root=dataset_root,
        classes=["building", "vehicle"],
    )

    assert (output_root / "labelmap.txt").read_text(encoding="utf-8") == "building\nvehicle\n"
    assert not outside.exists()


def test_segmentation_output_text_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
) -> None:
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    output_root = tmp_path / "output"
    label_path = output_root / "train" / "labels" / "img.txt"
    label_path.parent.mkdir(parents=True)
    tmp_path_link = label_path.with_suffix(f"{label_path.suffix}.{FixedUUID.hex}.tmp")
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final = tmp_path / "outside_final.txt"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    try:
        tmp_path_link.symlink_to(outside_tmp)
        label_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: FixedUUID())

    api._segmentation_write_text_within_root(
        label_path,
        output_root,
        "0 0.1 0.1 0.2 0.1 0.2 0.2\n",
    )

    assert not tmp_path_link.exists()
    assert not label_path.is_symlink()
    assert label_path.read_text(encoding="utf-8") == "0 0.1 0.1 0.2 0.1 0.2 0.2\n"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_segmentation_build_fails_when_image_worker_fails(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "source_split"
    _write_test_image(dataset_root / "train" / "images" / "img.jpg")
    (dataset_root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "train" / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")
    output_root = tmp_path / "sam3_outputs"
    output_root.mkdir()

    class FailingMiningPool:
        def __init__(self, _devices):
            self.workers = [self]

        def process_image(self, **_kwargs):
            raise RuntimeError("simulated mask failure")

        def close(self):
            return None

    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", output_root)
    monkeypatch.setattr(
        api,
        "_plan_segmentation_build",
        lambda _request: (
            {"id": "seg_out", "classes": ["building"]},
            {"dataset_root": str(output_root / "seg_out")},
        ),
    )
    monkeypatch.setattr(
        api,
        "_resolve_sam3_dataset_meta",
        lambda _dataset_id: {
            "id": "source",
            "dataset_root": str(dataset_root),
            "classes": ["building"],
        },
    )
    monkeypatch.setattr(api, "_resolve_sam3_mining_devices_impl", lambda *_args, **_kwargs: ["cpu"])
    monkeypatch.setattr(api, "_Sam3MiningPool", FailingMiningPool)

    def run_job_immediately(*, job, registry, lock, target, args, name=None):
        with lock:
            registry[job.job_id] = job
        target(*args)

    monkeypatch.setattr(api, "_register_job_and_start_thread", run_job_immediately)
    monkeypatch.setattr(
        api,
        "_convert_yolo_dataset_to_coco_impl",
        lambda _root: (_ for _ in ()).throw(AssertionError("must not publish failed build")),
    )
    with api.SEGMENTATION_BUILD_JOBS_LOCK:
        api.SEGMENTATION_BUILD_JOBS.clear()

    job = api._start_segmentation_build_job(
        api.SegmentationBuildRequest(source_dataset_id="source", output_name="seg_out")
    )

    assert job.status == "failed"
    assert job.error is not None
    assert job.error.startswith("segmentation_builder_worker_failed:1:")
    assert "simulated mask failure" in job.error
    assert job.result is None


def test_register_path_dedupes_existing_linked_entry(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "linked_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labelmap.txt").write_text("car\n", encoding="utf-8")

    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])
    existing = {
        "id": "existing",
        "label": "Existing",
        "storage_mode": "linked",
        "linked_root": str(dataset_root.resolve()),
        "signature": api._compute_dir_signature_impl(dataset_root),
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [existing])

    out = api.register_dataset_path(
        str(dataset_root), None, None, None, None, force_new=False, strict=True
    )
    assert out["id"] == "existing"


def test_register_path_rejects_symlinked_registry_root(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "linked_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labelmap.txt").write_text("car\n", encoding="utf-8")
    outside = tmp_path / "outside_registry"
    outside.mkdir()
    registry_root = tmp_path / "registry"
    try:
        registry_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api.register_dataset_path(
            str(dataset_root),
            "linked_ds",
            None,
            None,
            None,
            force_new=True,
            strict=True,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_register_target_invalid"
    assert list(outside.iterdir()) == []


def test_register_path_rejects_symlinked_labelmap_before_registry_write(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_labelmap.txt"
    outside.write_text("secret\n", encoding="utf-8")
    try:
        (dataset_root / "labelmap.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    registry_root = tmp_path / "registry"
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api.register_dataset_path(
            str(dataset_root),
            "linked_ds",
            None,
            None,
            None,
            force_new=True,
            strict=True,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "labelmap_path_forbidden"
    assert not registry_root.exists()
    assert outside.read_text(encoding="utf-8") == "secret\n"


def test_register_path_rejects_target_symlink_without_target_write(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labelmap.txt").write_text("car\n", encoding="utf-8")
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    outside = tmp_path / "outside_registry"
    outside.mkdir()
    outside_target = outside / "ghost"
    try:
        (registry_root / "linked_ds").symlink_to(outside_target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api.register_dataset_path(
            str(dataset_root),
            "linked_ds",
            None,
            None,
            None,
            force_new=True,
            strict=True,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_register_target_invalid"
    assert not outside_target.exists()


def test_register_path_rolls_back_when_metadata_write_fails(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labelmap.txt").write_text("car\n", encoding="utf-8")
    registry_root = tmp_path / "registry"
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)

    def fail_metadata_write(*_args, **_kwargs):
        raise api.HTTPException(status_code=400, detail="metadata_write_failed")

    monkeypatch.setattr(api, "_write_dataset_metadata_json", fail_metadata_write)

    with pytest.raises(api.HTTPException) as exc_info:
        api.register_dataset_path(
            str(dataset_root),
            "linked_ds",
            None,
            None,
            None,
            force_new=True,
            strict=True,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "metadata_write_failed"
    assert not (registry_root / "linked_ds").exists()
    assert dataset_root.exists()


def test_open_path_strict_requires_labelmap(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "bad_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])

    with pytest.raises(api.HTTPException) as exc:
        api.open_dataset_path(str(dataset_root), strict=True)
    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_shape_missing_labelmap"


def test_open_path_strict_rejects_symlinked_labelmap(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "bad_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_labelmap.txt"
    outside.write_text("secret\n", encoding="utf-8")
    try:
        (dataset_root / "labelmap.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])

    with pytest.raises(api.HTTPException) as exc:
        api.open_dataset_path(str(dataset_root), strict=True)

    assert exc.value.status_code == 400
    assert exc.value.detail == "labelmap_path_forbidden"
