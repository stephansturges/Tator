from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time

from PIL import Image
import pytest

from models.schemas import QwenCaptionDatasetJobRequest


def test_caption_dataset_job_request_strips_image_specific_template_fields() -> None:
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={
            "user_prompt": "Describe it",
            "image_base64": "bad",
            "image_name": "bad.jpg",
            "label_hints": [{"label": "Building"}],
            "image_width": 10,
        },
        attempts=99,
        per_image_timeout_seconds=5,
    )

    assert payload.caption_request == {"user_prompt": "Describe it"}
    assert payload.attempts == 5
    assert payload.per_image_timeout_seconds == 30.0
    assert payload.runner_no_output_timeout_seconds == 300.0


def test_caption_dataset_job_request_defaults_runner_watchdog_after_image_timeout() -> None:
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Describe it"},
        per_image_timeout_seconds=900,
    )

    assert payload.per_image_timeout_seconds == 900.0
    assert payload.runner_no_output_timeout_seconds == 1080.0
    assert payload.runner_heartbeat_interval_seconds == 30.0
    assert payload.runner_artifact_log_bytes == 1048576
    assert payload.runner_min_free_gb == 5.0
    assert payload.runner_disk_safety_factor == 1.25
    assert payload.max_failures == 0
    assert payload.pilot_min_cases == 300
    assert payload.pilot_deterministic_recovery_confidence == 0.95


def test_caption_dataset_job_request_normalizes_set_and_forget_controls() -> None:
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        annotation_session_id=" sess-lock ",
        caption_request={"user_prompt": "Describe it"},
        set_and_forget="true",
        allow_model_download="true",
        auto_resume_count=10000,
        max_auto_resumes="7",
        max_failed_case_rate=2,
        max_quality_failure_rate="-1",
        max_recovery_event_case_rate="0.5",
        max_loop_recovery_case_rate="bad",
        max_failed_attempt_row_rate="0.33",
        min_rate_cases=0,
        resume_reprocess_recovery_events="true",
        resume="false",
        require_pilot_certification="true",
        pilot_output_dir=" /tmp/pilot ",
        pilot_target_cases=99999999999,
        pilot_max_duration_hours="bad",
        pilot_max_p95_duration_hours="-1",
        pilot_min_cases=0,
        pilot_duration_safety_factor=0.5,
        pilot_require_prompt_budget_data="false",
        pilot_max_prompt_tokens=999999999,
        pilot_max_prompt_budget_adapted_case_rate="-1",
        pilot_deterministic_recovery_confidence=2,
    )

    assert payload.set_and_forget is True
    assert payload.annotation_session_id == "sess-lock"
    assert payload.allow_model_download is True
    assert payload.auto_resume_count == 1000
    assert payload.max_auto_resumes == 7
    assert payload.max_failed_case_rate == 1.0
    assert payload.max_quality_failure_rate == -1.0
    assert payload.max_recovery_event_case_rate == 0.5
    assert payload.max_loop_recovery_case_rate == 0.0
    assert payload.max_deterministic_recovery_case_rate == 0.01
    assert payload.max_failed_attempt_row_rate == 0.33
    assert payload.max_signal_exit_attempt_row_rate == 0.05
    assert payload.attempts == 3
    assert payload.min_rate_cases == 1
    assert payload.resume_reprocess_recovery_events is True
    assert payload.resume is False
    assert payload.require_pilot_certification is True
    assert payload.pilot_output_dir == "/tmp/pilot"
    assert payload.pilot_target_cases == 10_000_000
    assert payload.pilot_max_duration_hours == 336.0
    assert payload.pilot_max_p95_duration_hours == -1.0
    assert payload.pilot_min_cases == 1
    assert payload.pilot_duration_safety_factor == 1.0
    assert payload.pilot_require_prompt_budget_data is False
    assert payload.pilot_max_prompt_tokens == 1_000_000
    assert payload.pilot_max_prompt_budget_adapted_case_rate == -1.0
    assert payload.pilot_deterministic_recovery_confidence == 0.999999


def test_caption_dataset_job_request_uses_set_and_forget_loop_recovery_default() -> None:
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Describe it"},
        set_and_forget=True,
    )
    strict_payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Describe it"},
        set_and_forget=True,
        max_loop_recovery_case_rate=0,
        max_deterministic_recovery_case_rate=0,
        max_signal_exit_attempt_row_rate=0,
    )

    assert payload.max_loop_recovery_case_rate == 0.05
    assert payload.max_deterministic_recovery_case_rate == 0.01
    assert payload.max_signal_exit_attempt_row_rate == 0.05
    assert payload.attempts == 3
    assert strict_payload.max_loop_recovery_case_rate == 0.0
    assert strict_payload.max_deterministic_recovery_case_rate == 0.0
    assert strict_payload.max_signal_exit_attempt_row_rate == 0.0
    assert strict_payload.attempts == 3


def test_caption_dataset_job_request_normalizes_instruction_dataset_controls() -> None:
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Describe it"},
        instruction_dataset="true",
        subcaptions_per_image=99,
        include_caption0_in_training="false",
        include_generated_qa_in_training="0",
        include_deterministic_metadata_qa="yes",
        include_source_annotations_in_generator_context="off",
        strict_grounding="no",
        qa_mix="",
        answer_format="json",
    )
    negative = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Describe it"},
        subcaptions_per_image=-12,
    )

    assert payload.instruction_dataset is True
    assert payload.subcaptions_per_image == 20
    assert payload.include_caption0_in_training is False
    assert payload.include_generated_qa_in_training is False
    assert payload.include_deterministic_metadata_qa is True
    assert payload.include_source_annotations_in_generator_context is False
    assert payload.strict_grounding is False
    assert payload.qa_mix == "balanced"
    assert payload.answer_format == "json"
    assert negative.subcaptions_per_image == 0


def test_caption_dataset_job_request_rejects_instruction_dataset_with_no_row_family() -> None:
    with pytest.raises(ValueError, match="instruction_dataset_requires_training_row_family"):
        QwenCaptionDatasetJobRequest(
            dataset_id="ds",
            caption_request={"user_prompt": "Describe it"},
            instruction_dataset=True,
            include_caption0_in_training=False,
            include_generated_qa_in_training=False,
            include_deterministic_metadata_qa=False,
        )


def test_caption_dataset_job_request_normalizes_runner_artifact_log_limit() -> None:
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Describe it"},
        runner_artifact_log_bytes="0",
    )
    assert payload.runner_artifact_log_bytes == 0

    capped = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Describe it"},
        runner_artifact_log_bytes=9999999999,
    )
    assert capped.runner_artifact_log_bytes == 1_073_741_824

    disk = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Describe it"},
        runner_min_free_gb="-1",
        runner_disk_safety_factor="0.2",
    )
    assert disk.runner_min_free_gb == 0.0
    assert disk.runner_disk_safety_factor == 1.0


def test_caption_dataset_cases_use_effective_labels_and_skip_existing(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "train" / "images" / "frame.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16)).save(image_path)
    entry = {
        "id": "ds",
        "label": "Dataset",
        "yolo_layout": "split",
        "classes": ["Building", "Person"],
        "linked_root": str(dataset_root),
    }
    manifest = {
        "dataset_id": "ds",
        "dataset_label": "Dataset",
        "yolo_layout": "split",
        "labelmap": ["Building", "Person"],
        "images": [
            {
                "split": "train",
                "image_relpath": "frame.jpg",
                "image_name": "frame.jpg",
                "label_lines": ["0 0.5 0.5 0.5 0.5", "1 0.2 0.2 0.1 0.1"],
                "text_label": "",
            },
            {
                "split": "train",
                "image_relpath": "done.jpg",
                "image_name": "done.jpg",
                "label_lines": [],
                "text_label": "already captioned",
            },
        ],
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: dataset_root)
    monkeypatch.setattr(api, "_resolve_annotation_image_path", lambda *_args: image_path)

    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"caption_mode": "windowed"},
        overwrite=False,
    )
    _manifest, cases = api._qwen_caption_dataset_cases(payload, output_dir=tmp_path / "job")

    assert len(cases) == 1
    assert cases[0]["case_id"] == "image:train/frame.jpg:windowed"
    assert cases[0]["class_counts"] == {"Building": 1, "Person": 1}
    assert Path(cases[0]["label_path"]).read_text().splitlines() == [
        "0 0.5 0.5 0.5 0.5",
        "1 0.2 0.2 0.1 0.1",
    ]


def test_caption_dataset_save_appends_caption_record_and_primary_text_label(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    result_dir = tmp_path / "attempt"
    result_dir.mkdir()
    (result_dir / "result.json").write_text(
        """
        {
          "case": {"image_name": "frame.jpg", "split": "train"},
          "response": {"caption": "A caption."}
        }
        """
    )
    saved = []
    monkeypatch.setattr(
        api,
        "add_caption",
        lambda dataset_id, image_name, payload: saved.append(
            {"dataset_id": dataset_id, "image_name": image_name, "payload": payload}
        ) or {"status": "saved"},
    )
    job = api.QwenCaptionDatasetJob(job_id="job")

    image_name = api._qwen_caption_dataset_save_caption_from_row(
        dataset_id="ds",
        row={"artifact_dir": str(result_dir)},
        job=job,
    )

    assert image_name == "frame.jpg"
    assert saved[0] == {
        "dataset_id": "ds",
        "image_name": "frame.jpg",
        "payload": {
            "caption": "A caption.",
            "split": "train",
            "source": "qwen_caption_job",
            "title": "",
            "make_primary": False,
            "metadata": {
                "case_id": "",
                "used_counts": {},
                "used_boxes": None,
                "truncated": False,
                "recovery_events": [],
                "caption_quality": {},
                "artifact_dir": str(result_dir),
            },
        },
    }
    api._qwen_caption_dataset_save_caption_from_row(
        dataset_id="ds",
        row={"artifact_dir": str(result_dir)},
        job=job,
        make_primary=True,
    )
    assert saved[1]["payload"]["make_primary"] is True


def test_caption_dataset_caption_record_from_row_exposes_job_result(tmp_path: Path) -> None:
    import localinferenceapi as api

    result_dir = tmp_path / "attempt"
    result_dir.mkdir()
    (result_dir / "result.json").write_text(
        """
        {
          "case": {"case_id": "case-1", "image_name": "frame.jpg", "image_path": "/data/frame.jpg", "split": "train"},
          "response": {
            "caption": "A caption.",
            "used_counts": {"Building": 2},
            "used_boxes": 2,
            "truncated": true,
            "recovery_events": [{"stage": "safe_retry"}]
          },
          "caption_quality": {"characters": 10}
        }
        """
    )

    record = api._qwen_caption_dataset_caption_record_from_row(
        {"artifact_dir": str(result_dir), "case_id": "row-case", "case": "row-name", "stem": "frame"}
    )

    assert record == {
        "case_id": "row-case",
        "case": "row-name",
        "stem": "frame",
        "image_name": "frame.jpg",
        "image_path": "/data/frame.jpg",
        "split": "train",
        "caption": "A caption.",
        "used_counts": {"Building": 2},
        "used_boxes": 2,
        "truncated": True,
        "recovery_events": [{"stage": "safe_retry"}],
        "artifact_dir": str(result_dir),
        "caption_quality": {"characters": 10},
        "generated_qa_pair_count": 0,
    }


def test_caption_dataset_save_instruction_records_from_row_persists_generated_qa(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api

    result_dir = tmp_path / "attempt"
    result_dir.mkdir()
    (result_dir / "result.json").write_text(
        """
        {
          "case": {"case_id": "case-1", "image_name": "frame.jpg", "image_path": "/data/frame.jpg", "split": "train"},
          "response": {
            "caption": "A broad caption.",
            "generated_qa_pairs": [
              {
                "question": "What borders the water?",
                "answer": "Buildings border the water.",
                "answer_format": "object_count_json",
                "validated_against": ["source_annotations.object_counts"],
                "source_fields": ["source_annotations.object_counts.Building"],
                "validation_status": "accepted",
                "review_status": "machine_validated"
              },
              {"question": "How many boats are described?", "answer": "Two boats are described."}
            ]
          },
          "caption_quality": {"characters": 16}
        }
        """
    )
    captured = []
    monkeypatch.setattr(
        api,
        "_dataset_caption_add_instruction_records",
        lambda dataset_id, image_name, records, split=None: captured.append(
            {
                "dataset_id": dataset_id,
                "image_name": image_name,
                "records": list(records),
                "split": split,
            }
        )
        or len(records),
    )
    job = api.QwenCaptionDatasetJob(job_id="job")

    added = api._qwen_caption_dataset_save_instruction_records_from_row(
        dataset_id="ds",
        row={"artifact_dir": str(result_dir), "case_id": "row-case", "case": "row-name"},
        job=job,
    )

    assert added == 2
    assert captured[0]["dataset_id"] == "ds"
    assert captured[0]["image_name"] == "frame.jpg"
    assert captured[0]["split"] == "train"
    assert [record["row_type"] for record in captured[0]["records"]] == ["generated_qa", "generated_qa"]
    assert captured[0]["records"][0]["caption"] == "A broad caption."
    assert captured[0]["records"][0]["caption_id"] == "row-case"
    assert captured[0]["records"][0]["answer_format"] == "object_count_json"
    assert captured[0]["records"][0]["validated_against"] == ["source_annotations.object_counts"]
    assert captured[0]["records"][0]["source_fields"] == ["source_annotations.object_counts.Building"]
    assert captured[0]["records"][0]["validation_status"] == "accepted"
    assert captured[0]["records"][0]["review_status"] == "machine_validated"
    assert captured[0]["records"][0]["metadata"]["case_id"] == "row-case"
    assert captured[0]["records"][0]["metadata"]["artifact_dir"] == str(result_dir)
    assert job.logs[-1]["generated_qa_rows"] == 2


def test_caption_dataset_add_instruction_records_persists_validation_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api

    overlay_root = tmp_path / "overlay"
    records_path = overlay_root / "captions" / "instruction_records.jsonl"
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: {"id": "ds"})
    monkeypatch.setattr(
        api,
        "_dataset_caption_context",
        lambda _entry, image_name, _payload=None: ("train", Path(image_name), f"train/{image_name}"),
    )
    monkeypatch.setattr(
        api,
        "_dataset_caption_instruction_records_path",
        lambda _entry, ensure=False: (overlay_root, records_path),
    )

    added = api._dataset_caption_add_instruction_records(
        "ds",
        "frame.jpg",
        [
            {
                "id": "qa-1",
                "question": "How many buildings are present?",
                "answer": '{"object_counts":{"Building":1}}',
                "answer_format": "object_count_json",
                "validated_against": ["source_annotations.object_counts"],
                "source_fields": ["source_annotations.object_counts.Building"],
                "validation_status": "accepted",
                "review_status": "machine_validated",
                "validation": {"status": "accepted"},
            }
        ],
        split="train",
    )
    loaded = api._load_dataset_caption_instruction_records({"id": "ds"})
    raw = json.loads(records_path.read_text().splitlines()[0])

    assert added == 1
    assert loaded[0]["answer_format"] == "object_count_json"
    assert loaded[0]["validated_against"] == ["source_annotations.object_counts"]
    assert loaded[0]["source_fields"] == ["source_annotations.object_counts.Building"]
    assert loaded[0]["validation_status"] == "accepted"
    assert loaded[0]["review_status"] == "machine_validated"
    assert raw["source_fields"] == ["source_annotations.object_counts.Building"]
    assert raw["validation"] == {"status": "accepted"}


def test_caption_instruction_archive_separates_generated_qa_from_source_annotations(
    monkeypatch,
) -> None:
    import localinferenceapi as api

    manifest = {
        "labelmap": ["Boat", "Building"],
        "images": [
            {
                "split": "train",
                "image_relpath": "frame.jpg",
                "image_name": "frame.jpg",
                "label_lines": [
                    "0 0.5 0.5 0.1 0.1",
                    "1 0.2 0.2 0.1 0.1",
                ],
            }
        ],
    }
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    captions = [
        {
            "id": "caption-1",
            "image_name": "frame.jpg",
            "image_key": "train/frame.jpg",
            "split": "train",
            "caption": "A caption about the waterfront.",
            "source": "qwen_caption_job",
            "is_primary": True,
            "caption_index": 1,
        }
    ]
    generated = [
        {
            "id": "qa-1",
            "image_name": "frame.jpg",
            "image_key": "train/frame.jpg",
            "split": "train",
            "question": "What borders the water?",
            "answer": "Buildings border the water.",
            "row_type": "generated_qa",
            "answer_source": "vlm_generated",
        },
        {
            "id": "qa-2",
            "image_name": "frame.jpg",
            "image_key": "train/frame.jpg",
            "split": "train",
            "question": "What borders the water?",
            "answer": "A duplicate question should be rejected.",
            "row_type": "generated_qa",
            "answer_source": "vlm_generated",
        },
    ]

    archive = api._dataset_caption_instruction_archive(
        captions,
        generated,
        dataset_id="ds",
        entry={"id": "ds"},
        settings=api._caption_instruction_export_settings({}),
        exported_at="2026-01-01T00:00:00Z",
    )
    image = archive["images"][0]
    archive_row = archive["archive_rows"][0]
    report = archive["captioning_report"]

    assert archive["format"] == "tator_caption_instruction_archive_v1"
    assert archive["training_row_count"] == 2
    assert archive["rejected_training_row_count"] == 1
    assert archive["deterministic_metadata_qa_pair_count"] == 0
    assert archive["review_rows_format"] == "tator_caption_instruction_review_rows_v1"
    assert len(archive["instruction_review_rows"]) == 3
    assert archive_row["image_id"] == "frame"
    assert archive_row["image_path"] == "frame.jpg"
    assert archive_row["split"] == "train"
    assert set(archive_row) == {
        "image_id",
        "image_path",
        "split",
        "source_annotations",
        "language_annotations",
        "deterministic_metadata_qa_pairs",
        "export_metadata",
    }
    assert archive_row["export_metadata"]["selected_training_row_count"] == 2
    assert archive_row["export_metadata"]["generated_qa_candidate_count"] == 2
    assert archive_row["export_metadata"]["accepted_generated_qa_count"] == 2
    assert report["format"] == "tator_caption_instruction_report_v1"
    assert report["image_count"] == 1
    assert report["generated_qa_candidate_count"] == 2
    assert report["accepted_generated_qa_count"] == 2
    assert report["selected_flattened_row_count"] == 2
    assert report["instruction_review_row_count"] == 3
    assert report["manual_review_required_count"] == 3
    assert report["split_training_row_counts"] == {"train": 2}
    assert archive["instruction_export_validation"] == report["instruction_export_validation"]
    assert archive["instruction_export_validation"]["ok"] is True
    assert archive["instruction_export_validation"]["error_count"] == 0
    assert archive["instruction_export_validation"]["row_count"] == 2
    consistency = archive["instruction_artifact_consistency"]
    assert consistency == report["instruction_artifact_consistency"]
    assert consistency["format"] == "tator_caption_instruction_artifact_consistency_v1"
    assert consistency["ok"] is True
    assert consistency["error_count"] == 0
    assert consistency["counts"]["training_row_count"] == 2
    assert consistency["counts"]["archive_row_count"] == 1
    assert consistency["counts"]["review_row_count"] == 3
    assert consistency["counts"]["selected_review_row_count"] == 2
    assert consistency["counts"]["manual_review_required_count"] == 3
    readiness = report["training_readiness"]
    assert readiness["status"] == "needs_review"
    assert readiness["ready_for_training"] is False
    assert readiness["instruction_export_validation_error_count"] == 0
    assert readiness["instruction_artifact_consistency_error_count"] == 0
    assert readiness["selected_training_row_count"] == 2
    assert readiness["selected_manual_review_row_count"] == 2
    assert readiness["pending_manual_review_row_count"] == 2
    assert "review_selected_language_rows" in readiness["required_actions"]
    assert "generated_qa_question_diversity_ratio_below_threshold" in readiness["quality_warnings"]
    assert "source_class_coverage_rate_below_threshold" in readiness["quality_warnings"]
    metrics = report["corpus_quality_metrics"]
    assert archive["corpus_quality_metrics"] == metrics
    assert metrics["generated_qa_candidate_count"] == 2
    assert metrics["accepted_generated_qa_count"] == 2
    assert metrics["generated_qa_unique_question_count"] == 1
    assert metrics["generated_qa_global_duplicate_question_count"] == 1
    assert metrics["generated_qa_per_image_duplicate_question_count"] == 1
    assert metrics["generated_qa_question_diversity_ratio"] == 0.5
    assert metrics["generated_qa_acceptance_rate"] == 1.0
    assert metrics["duplicate_image_question_rejection_count"] == 1
    assert metrics["image_training_coverage_rate"] == 1.0
    assert metrics["source_validated_training_row_count"] == 2
    assert metrics["source_validated_training_row_rate"] == 1.0
    assert metrics["source_classes"] == ["Boat", "Building"]
    assert metrics["source_classes_covered_by_training_rows"] == []
    assert metrics["source_class_coverage_rate"] == 0.0
    assert metrics["training_answer_format_distribution"] == {"natural": 2}
    assert image["source_annotations"]["object_counts"] == {"Boat": 1, "Building": 1}
    assert image["source_annotations"]["annotations"][0]["class_name"] == "Boat"
    assert image["language_annotations"]["caption0"]["caption"] == "A caption about the waterfront."
    assert len(image["language_annotations"]["generated_qa_pairs"]) == 2
    assert "question" not in image["source_annotations"]["annotations"][0]
    assert [row["metadata"]["row_type"] for row in archive["training_rows"]] == [
        "caption0",
        "generated_qa",
    ]
    review_rows = archive["instruction_review_rows"]
    assert {row["format"] for row in review_rows} == {"tator_caption_instruction_review_rows_v1"}
    assert {row["row_origin"] for row in review_rows} == {"caption0", "generated_qa"}
    assert all("review_decision" in row and "review_notes" in row for row in review_rows)
    assert all(row["source_summary"]["object_counts"] == {"Boat": 1, "Building": 1} for row in review_rows)
    assert sum(1 for row in review_rows if row["selected_for_training"]) == 2
    assert sum(1 for row in review_rows if row["requires_manual_review"]) == 3
    assert any(
        row["row_origin"] == "generated_qa"
        and row["question"] == "What borders the water?"
        and not row["selected_for_training"]
        for row in review_rows
    )
    assert archive["rejections"][0]["reason"] == "duplicate_image_question"

    deterministic = api._dataset_caption_instruction_archive(
        captions,
        generated,
        dataset_id="ds",
        entry={"id": "ds"},
        settings=api._caption_instruction_export_settings(
            {
                "include_caption0_in_training": False,
                "include_generated_qa_in_training": False,
                "include_deterministic_metadata_qa": True,
            }
        ),
        exported_at="2026-01-01T00:00:00Z",
    )
    assert deterministic["deterministic_metadata_qa_pair_count"] == 8
    assert deterministic["training_row_count"] == 8
    row_types = {row["metadata"]["row_type"] for row in deterministic["training_rows"]}
    assert {
        "deterministic_class_list",
        "deterministic_object_count_schema",
        "deterministic_count",
        "deterministic_presence",
        "deterministic_spatial",
    }.issubset(row_types)
    for row in deterministic["training_rows"]:
        assert json.loads(row["answer"])
        assert row["metadata"]["validation_status"] == "machine_validated"
        assert row["metadata"]["source_fields"]
    assert deterministic["qa_type_distribution"]["deterministic_count"] == 2
    assert deterministic["split_training_row_counts"] == {"train": 8}
    assert deterministic["captioning_report"]["deterministic_metadata_qa_count"] == 8
    assert deterministic["captioning_report"]["instruction_review_row_count"] == 11
    assert deterministic["captioning_report"]["manual_review_required_count"] == 3
    assert sum(1 for row in deterministic["instruction_review_rows"] if row["selected_for_training"]) == 8
    assert deterministic["captioning_report"]["training_readiness"]["status"] == "ready"
    assert deterministic["captioning_report"]["training_readiness"]["ready_for_training"] is True
    assert deterministic["captioning_report"]["instruction_export_validation"]["ok"] is True
    assert deterministic["instruction_export_validation"]["row_count"] == 8
    assert deterministic["captioning_report"]["training_readiness"]["pending_manual_review_row_count"] == 0
    assert deterministic["archive_rows"][0]["export_metadata"]["deterministic_metadata_qa_pair_count"] == 8
    deterministic_metrics = deterministic["corpus_quality_metrics"]
    assert deterministic_metrics["source_validated_training_row_count"] == 8
    assert deterministic_metrics["source_validated_training_row_rate"] == 1.0
    assert deterministic_metrics["source_classes_covered_by_training_rows"] == ["Boat", "Building"]
    assert deterministic_metrics["source_class_coverage_rate"] == 1.0
    assert deterministic_metrics["training_answer_format_distribution"] == {
        "boolean_json": 2,
        "object_count_json": 3,
        "spatial_fact_json": 2,
        "visible_class_json": 1,
    }


def test_caption_instruction_review_import_persists_review_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    import localinferenceapi as api

    entry = {"id": "ds", "dataset_root": str(tmp_path), "registry_root": str(tmp_path)}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_manifest_for_entry",
        lambda _entry: {
            "labelmap": [],
            "images": [
                {
                    "image_name": "frame.jpg",
                    "image_relpath": "frame.jpg",
                    "split": "train",
                    "label_source_present": True,
                    "label_lines": [],
                }
            ],
        },
    )
    api._write_dataset_caption_instruction_records(
        entry,
        [
            {
                "id": "qa-1",
                "image_name": "frame.jpg",
                "image_key": "train/frame.jpg",
                "split": "train",
                "question": "What is the scene type?",
                "answer": "A waterfront area.",
                "row_type": "generated_qa",
                "answer_source": "vlm_generated",
                "validation_status": "accepted",
            }
        ],
    )
    text_label_path = tmp_path / "text_labels" / "frame.txt"
    text_label_path.parent.mkdir(parents=True, exist_ok=True)
    text_label_path.write_text("A primary caption.", encoding="utf-8")
    synthetic_caption_id = (
        "primary_"
        + hashlib.sha1("ds|train/frame.jpg|A primary caption.".encode("utf-8")).hexdigest()[:16]
    )

    result = api.apply_caption_instruction_review(
        "ds",
        {
            "reviewer": "qa-review",
            "rows": [
                {
                    "format": "tator_caption_instruction_review_rows_v1",
                    "dataset_id": "ds",
                    "image_path": "frame.jpg",
                    "split": "train",
                    "row_origin": "generated_qa",
                    "qa_id": "qa-1",
                    "row_type": "generated_qa",
                    "question": "What is the scene type?",
                    "candidate_answer": "A waterfront area.",
                    "training_answer": "A waterfront area.",
                    "validation_status": "accepted",
                    "selected_for_training": True,
                    "requires_manual_review": True,
                    "review_decision": "accepted",
                    "review_notes": "grounded",
                    "rejection_reasons": [],
                    "source_summary": {"status": "empty_label_file"},
                },
                {
                    "format": "tator_caption_instruction_review_rows_v1",
                    "dataset_id": "ds",
                    "image_path": "frame.jpg",
                    "split": "train",
                    "row_origin": "caption0",
                    "qa_id": synthetic_caption_id,
                    "row_type": "caption0",
                    "question": "Describe this image in detail.",
                    "candidate_answer": "A primary caption.",
                    "training_answer": "A primary caption.",
                    "answer_source": "text_label",
                    "validation_status": "accepted",
                    "selected_for_training": True,
                    "requires_manual_review": True,
                    "review_decision": "reject",
                    "review_notes": "too vague",
                    "rejection_reasons": [],
                    "source_summary": {"status": "empty_label_file"},
                    "metadata": {"synthetic": True},
                },
                {
                    "format": "tator_caption_instruction_review_rows_v1",
                    "dataset_id": "ds",
                    "image_path": "frame.jpg",
                    "split": "train",
                    "row_origin": "deterministic_metadata_qa",
                    "qa_id": "frame.jpg__metadata_object_counts",
                    "row_type": "deterministic_object_count_schema",
                    "question": "Return the labeled object counts for this image as JSON.",
                    "candidate_answer": "{}",
                    "training_answer": "{}",
                    "validation_status": "machine_validated",
                    "selected_for_training": False,
                    "requires_manual_review": False,
                    "review_decision": "accepted",
                    "review_notes": "",
                    "rejection_reasons": [],
                    "source_summary": {"status": "empty_label_file"},
                },
            ],
        },
    )

    assert result["status"] == "applied"
    assert result["received_row_count"] == 3
    assert result["applied_count"] == 2
    assert result["generated_qa_review_applied_count"] == 1
    assert result["caption_review_applied_count"] == 1
    assert result["created_caption_review_record_count"] == 1
    assert result["skipped_count"] == 1
    assert result["skipped_rows"][0]["reason"] == "deterministic_review_not_persisted"

    instruction_records = api._load_dataset_caption_instruction_records(entry)
    assert instruction_records[0]["review_status"] == "accepted"
    assert instruction_records[0]["metadata"]["review_decision"] == "accepted"
    assert instruction_records[0]["metadata"]["review_notes"] == "grounded"
    assert instruction_records[0]["metadata"]["reviewer"] == "qa-review"

    caption_records = api._load_dataset_caption_records(entry)
    assert len(caption_records) == 1
    assert caption_records[0]["id"] == synthetic_caption_id
    assert caption_records[0]["caption"] == "A primary caption."
    assert caption_records[0]["is_primary"] is False
    assert caption_records[0]["metadata"]["review_decision"] == "rejected"
    assert caption_records[0]["metadata"]["review_notes"] == "too vague"

    archive = api._dataset_caption_instruction_archive(
        caption_records,
        instruction_records,
        dataset_id="ds",
        entry=entry,
        settings=api._caption_instruction_export_settings({}),
        exported_at="2026-01-01T00:00:00Z",
    )
    review_rows = archive["instruction_review_rows"]
    caption_review = next(row for row in review_rows if row["row_origin"] == "caption0")
    generated_review = next(row for row in review_rows if row["row_origin"] == "generated_qa")
    assert caption_review["review_decision"] == "rejected"
    assert caption_review["selected_for_training"] is False
    assert generated_review["review_decision"] == "accepted"
    training_rows = archive["training_rows"]
    assert len(training_rows) == 1
    assert training_rows[0]["metadata"]["qa_id"] == "qa-1"
    assert all(row["metadata"]["qa_id"] != synthetic_caption_id for row in training_rows)
    assert archive["captioning_report"]["rejection_reason_counts"]["caption0_manual_review_rejected"] == 1
    readiness = archive["captioning_report"]["training_readiness"]
    assert readiness["status"] == "ready"
    assert readiness["rejected_manual_review_row_count"] == 0
    assert "selected_row_rejected_by_manual_review" not in readiness["blocking_reasons"]


def test_caption_instruction_review_decision_normalizes_external_review_values() -> None:
    import localinferenceapi as api

    assert api._caption_instruction_review_decision("accepted") == "accepted"
    assert api._caption_instruction_review_decision("reject") == "rejected"
    assert api._caption_instruction_review_decision("needs-revision") == "needs_revision"
    assert api._caption_instruction_review_decision("needs review") == "needs_revision"
    assert api._caption_instruction_review_decision("Needs-Rewrite") == "needs_revision"


def test_caption_instruction_review_import_rejects_mismatched_dataset_id(
    monkeypatch,
    tmp_path,
) -> None:
    import localinferenceapi as api

    entry = {"id": "ds", "dataset_root": str(tmp_path), "registry_root": str(tmp_path)}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_manifest_for_entry",
        lambda _entry: {
            "labelmap": [],
            "images": [
                {
                    "image_name": "frame.jpg",
                    "image_relpath": "frame.jpg",
                    "split": "train",
                    "label_source_present": True,
                    "label_lines": [],
                }
            ],
        },
    )
    api._write_dataset_caption_instruction_records(
        entry,
        [
            {
                "id": "qa-1",
                "image_name": "frame.jpg",
                "image_key": "train/frame.jpg",
                "split": "train",
                "question": "What is the scene type?",
                "answer": "A waterfront area.",
                "row_type": "generated_qa",
                "answer_source": "vlm_generated",
                "validation_status": "accepted",
            }
        ],
    )

    with pytest.raises(api.HTTPException) as excinfo:
        api.apply_caption_instruction_review(
            "ds",
            {
                "rows": [
                    {
                        "format": "tator_caption_instruction_review_rows_v1",
                        "dataset_id": "other-dataset",
                        "image_path": "frame.jpg",
                        "split": "train",
                        "row_origin": "generated_qa",
                        "qa_id": "qa-1",
                        "row_type": "generated_qa",
                        "question": "What is the scene type?",
                        "candidate_answer": "A waterfront area.",
                        "training_answer": "A waterfront area.",
                        "validation_status": "accepted",
                        "selected_for_training": True,
                        "requires_manual_review": True,
                        "review_decision": "rejected",
                        "review_notes": "wrong dataset",
                        "rejection_reasons": [],
                        "source_summary": {"status": "empty_label_file"},
                    }
                ]
            },
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "review_rows_dataset_id_mismatch:row_1:other-dataset!=ds"
    records = api._load_dataset_caption_instruction_records(entry)
    assert records[0]["review_status"] == ""
    assert "review_decision" not in records[0]["metadata"]


@pytest.mark.parametrize("row_origin", ["generated_qa", "caption0"])
def test_caption_instruction_review_import_rejects_missing_dataset_id(
    monkeypatch,
    tmp_path,
    row_origin,
) -> None:
    import localinferenceapi as api

    entry = {"id": "ds", "dataset_root": str(tmp_path), "registry_root": str(tmp_path)}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_manifest_for_entry",
        lambda _entry: {
            "labelmap": [],
            "images": [
                {
                    "image_name": "frame.jpg",
                    "image_relpath": "frame.jpg",
                    "split": "train",
                    "label_source_present": True,
                    "label_lines": [],
                }
            ],
        },
    )
    if row_origin == "generated_qa":
        api._write_dataset_caption_instruction_records(
            entry,
            [
                {
                    "id": "qa-1",
                    "image_name": "frame.jpg",
                    "image_key": "train/frame.jpg",
                    "split": "train",
                    "question": "What is the scene type?",
                    "answer": "A waterfront area.",
                    "row_type": "generated_qa",
                    "answer_source": "vlm_generated",
                    "validation_status": "accepted",
                }
            ],
        )
        row = {
            "format": "tator_caption_instruction_review_rows_v1",
            "image_path": "frame.jpg",
            "split": "train",
            "row_origin": "generated_qa",
            "qa_id": "qa-1",
            "row_type": "generated_qa",
            "question": "What is the scene type?",
            "candidate_answer": "A waterfront area.",
            "training_answer": "A waterfront area.",
            "validation_status": "accepted",
            "selected_for_training": True,
            "requires_manual_review": True,
            "review_decision": "accepted",
            "review_notes": "missing dataset id should fail",
            "rejection_reasons": [],
            "source_summary": {"status": "empty_label_file"},
        }
    else:
        api._write_dataset_caption_records(
            entry,
            [
                {
                    "id": "caption-existing",
                    "image_name": "frame.jpg",
                    "image_key": "train/frame.jpg",
                    "split": "train",
                    "caption": "A current caption.",
                    "source": "manual",
                    "metadata": {},
                }
            ],
        )
        row = {
            "format": "tator_caption_instruction_review_rows_v1",
            "image_path": "frame.jpg",
            "split": "train",
            "row_origin": "caption0",
            "qa_id": "caption-existing",
            "row_type": "caption0",
            "question": "Describe this image in detail.",
            "candidate_answer": "A current caption.",
            "training_answer": "A current caption.",
            "validation_status": "accepted",
            "selected_for_training": True,
            "requires_manual_review": True,
            "review_decision": "accepted",
            "review_notes": "missing dataset id should fail",
            "rejection_reasons": [],
            "source_summary": {"status": "empty_label_file"},
        }

    with pytest.raises(api.HTTPException) as excinfo:
        api.apply_caption_instruction_review("ds", {"rows": [row]})

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "review_rows_dataset_id_missing:row_1"
    if row_origin == "generated_qa":
        records = api._load_dataset_caption_instruction_records(entry)
    else:
        records = api._load_dataset_caption_records(entry)
    assert records[0].get("review_status", "") == ""
    assert "review_decision" not in records[0]["metadata"]


@pytest.mark.parametrize(
    ("second_decision", "expected_detail"),
    [
        ("rejected", "review_rows_conflicting_duplicate_target:row_1:row_2"),
        ("accepted", "review_rows_duplicate_target:row_1:row_2"),
    ],
)
def test_caption_instruction_review_import_rejects_duplicate_actionable_targets(
    monkeypatch,
    tmp_path,
    second_decision,
    expected_detail,
) -> None:
    import localinferenceapi as api

    entry = {"id": "ds", "dataset_root": str(tmp_path), "registry_root": str(tmp_path)}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_manifest_for_entry",
        lambda _entry: {
            "labelmap": [],
            "images": [
                {
                    "image_name": "frame.jpg",
                    "image_relpath": "frame.jpg",
                    "split": "train",
                    "label_source_present": True,
                    "label_lines": [],
                }
            ],
        },
    )
    api._write_dataset_caption_instruction_records(
        entry,
        [
            {
                "id": "qa-1",
                "image_name": "frame.jpg",
                "image_key": "train/frame.jpg",
                "split": "train",
                "question": "What is the scene type?",
                "answer": "A waterfront area.",
                "row_type": "generated_qa",
                "answer_source": "vlm_generated",
                "validation_status": "accepted",
            }
        ],
    )

    base_row = {
        "format": "tator_caption_instruction_review_rows_v1",
        "dataset_id": "ds",
        "image_path": "frame.jpg",
        "split": "train",
        "row_origin": "generated_qa",
        "qa_id": "qa-1",
        "row_type": "generated_qa",
        "question": "What is the scene type?",
        "candidate_answer": "A waterfront area.",
        "training_answer": "A waterfront area.",
        "validation_status": "accepted",
        "selected_for_training": True,
        "requires_manual_review": True,
        "review_decision": "accepted",
        "review_notes": "first decision",
        "rejection_reasons": [],
        "source_summary": {"status": "empty_label_file"},
    }
    duplicate_row = {
        **base_row,
        "row_type": "unexpected_external_type",
        "review_decision": second_decision,
        "review_notes": "duplicate decision",
    }

    with pytest.raises(api.HTTPException) as excinfo:
        api.apply_caption_instruction_review("ds", {"rows": [base_row, duplicate_row]})

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == expected_detail
    records = api._load_dataset_caption_instruction_records(entry)
    assert records[0]["review_status"] == ""
    assert "review_decision" not in records[0]["metadata"]


@pytest.mark.parametrize(
    ("row_origin", "second_decision", "expected_detail"),
    [
        ("generated_qa", "rejected", "review_rows_conflicting_duplicate_resolved_target:row_1:row_2"),
        ("generated_qa", "accepted", "review_rows_duplicate_resolved_target:row_1:row_2"),
        ("caption0", "rejected", "review_rows_conflicting_duplicate_resolved_target:row_1:row_2"),
        ("caption0", "accepted", "review_rows_duplicate_resolved_target:row_1:row_2"),
    ],
)
def test_caption_instruction_review_import_rejects_duplicate_resolved_actionable_targets(
    monkeypatch,
    tmp_path,
    row_origin,
    second_decision,
    expected_detail,
) -> None:
    import localinferenceapi as api

    entry = {"id": "ds", "dataset_root": str(tmp_path), "registry_root": str(tmp_path)}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_manifest_for_entry",
        lambda _entry: {
            "labelmap": [],
            "images": [
                {
                    "image_name": "frame.jpg",
                    "image_relpath": "frame.jpg",
                    "split": "train",
                    "label_source_present": True,
                    "label_lines": [],
                }
            ],
        },
    )
    if row_origin == "generated_qa":
        api._write_dataset_caption_instruction_records(
            entry,
            [
                {
                    "id": "qa-1",
                    "image_name": "frame.jpg",
                    "image_key": "train/frame.jpg",
                    "split": "train",
                    "question": "What is the scene type?",
                    "answer": "A waterfront area.",
                    "row_type": "generated_qa",
                    "answer_source": "vlm_generated",
                    "validation_status": "accepted",
                }
            ],
        )
        base_row = {
            "format": "tator_caption_instruction_review_rows_v1",
            "dataset_id": "ds",
            "image_path": "frame.jpg",
            "split": "train",
            "row_origin": "generated_qa",
            "qa_id": "qa-1",
            "row_type": "generated_qa",
            "question": "What is the scene type?",
            "candidate_answer": "A waterfront area.",
            "training_answer": "A waterfront area.",
            "validation_status": "accepted",
            "selected_for_training": True,
            "requires_manual_review": True,
            "review_decision": "accepted",
            "review_notes": "first decision",
            "rejection_reasons": [],
            "source_summary": {"status": "empty_label_file"},
        }
    else:
        api._write_dataset_caption_records(
            entry,
            [
                {
                    "id": "caption-existing",
                    "image_name": "frame.jpg",
                    "image_key": "train/frame.jpg",
                    "split": "train",
                    "caption": "A current caption.",
                    "source": "manual",
                    "metadata": {},
                }
            ],
        )
        base_row = {
            "format": "tator_caption_instruction_review_rows_v1",
            "dataset_id": "ds",
            "image_path": "frame.jpg",
            "split": "train",
            "row_origin": "caption0",
            "qa_id": "caption-existing",
            "row_type": "caption0",
            "question": "Describe this image in detail.",
            "candidate_answer": "A current caption.",
            "training_answer": "A current caption.",
            "validation_status": "accepted",
            "selected_for_training": True,
            "requires_manual_review": True,
            "review_decision": "accepted",
            "review_notes": "first decision",
            "rejection_reasons": [],
            "source_summary": {"status": "empty_label_file"},
        }
    duplicate_row = {
        **base_row,
        "review_decision": second_decision,
        "review_notes": "same stored record through content match",
    }
    duplicate_row.pop("qa_id")

    with pytest.raises(api.HTTPException) as excinfo:
        api.apply_caption_instruction_review("ds", {"rows": [base_row, duplicate_row]})

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == expected_detail
    if row_origin == "generated_qa":
        records = api._load_dataset_caption_instruction_records(entry)
    else:
        records = api._load_dataset_caption_records(entry)
    assert records[0].get("review_status", "") == ""
    assert "review_decision" not in records[0]["metadata"]


@pytest.mark.parametrize(
    ("row_origin", "row_update", "expected_detail"),
    [
        (
            "generated_qa",
            {"question": ""},
            "review_rows_generated_qa_text_missing:row_1",
        ),
        (
            "generated_qa",
            {"candidate_answer": "", "training_answer": ""},
            "review_rows_generated_qa_text_missing:row_1",
        ),
        (
            "caption0",
            {"candidate_answer": "", "training_answer": ""},
            "review_rows_caption0_answer_missing:row_1",
        ),
    ],
)
def test_caption_instruction_review_import_rejects_rows_missing_current_text(
    monkeypatch,
    tmp_path,
    row_origin,
    row_update,
    expected_detail,
) -> None:
    import localinferenceapi as api

    entry = {"id": "ds", "dataset_root": str(tmp_path), "registry_root": str(tmp_path)}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_manifest_for_entry",
        lambda _entry: {
            "labelmap": [],
            "images": [
                {
                    "image_name": "frame.jpg",
                    "image_relpath": "frame.jpg",
                    "split": "train",
                    "label_source_present": True,
                    "label_lines": [],
                }
            ],
        },
    )
    if row_origin == "generated_qa":
        api._write_dataset_caption_instruction_records(
            entry,
            [
                {
                    "id": "qa-1",
                    "image_name": "frame.jpg",
                    "image_key": "train/frame.jpg",
                    "split": "train",
                    "question": "What is the scene type?",
                    "answer": "A waterfront area.",
                    "row_type": "generated_qa",
                    "answer_source": "vlm_generated",
                    "validation_status": "accepted",
                }
            ],
        )
        row = {
            "format": "tator_caption_instruction_review_rows_v1",
            "dataset_id": "ds",
            "image_path": "frame.jpg",
            "split": "train",
            "row_origin": "generated_qa",
            "qa_id": "qa-1",
            "row_type": "generated_qa",
            "question": "What is the scene type?",
            "candidate_answer": "A waterfront area.",
            "training_answer": "A waterfront area.",
            "validation_status": "accepted",
            "selected_for_training": True,
            "requires_manual_review": True,
            "review_decision": "accepted",
            "review_notes": "text-stripped review should fail",
            "rejection_reasons": [],
            "source_summary": {"status": "empty_label_file"},
        }
    else:
        api._write_dataset_caption_records(
            entry,
            [
                {
                    "id": "caption-existing",
                    "image_name": "frame.jpg",
                    "image_key": "train/frame.jpg",
                    "split": "train",
                    "caption": "A current caption.",
                    "source": "manual",
                    "metadata": {},
                }
            ],
        )
        row = {
            "format": "tator_caption_instruction_review_rows_v1",
            "dataset_id": "ds",
            "image_path": "frame.jpg",
            "split": "train",
            "row_origin": "caption0",
            "qa_id": "caption-existing",
            "row_type": "caption0",
            "question": "Describe this image in detail.",
            "candidate_answer": "A current caption.",
            "training_answer": "A current caption.",
            "validation_status": "accepted",
            "selected_for_training": True,
            "requires_manual_review": True,
            "review_decision": "accepted",
            "review_notes": "text-stripped review should fail",
            "rejection_reasons": [],
            "source_summary": {"status": "empty_label_file"},
        }
    row.update(row_update)

    with pytest.raises(api.HTTPException) as excinfo:
        api.apply_caption_instruction_review("ds", {"rows": [row]})

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == expected_detail
    if row_origin == "generated_qa":
        records = api._load_dataset_caption_instruction_records(entry)
    else:
        records = api._load_dataset_caption_records(entry)
    assert records[0].get("review_status", "") == ""
    assert "review_decision" not in records[0]["metadata"]


@pytest.mark.parametrize(
    ("bad_row_update", "expected_detail"),
    [
        (
            {
                "qa_id": "qa-missing",
                "question": "What unsupported question is this?",
                "candidate_answer": "An unmatched answer.",
                "training_answer": "An unmatched answer.",
            },
            "review_rows_generated_qa_not_found:row_2",
        ),
        (
            {
                "image_path": "other.jpg",
                "image_name": "other.jpg",
                "image": "other.jpg",
            },
            "review_rows_generated_qa_not_found:row_2",
        ),
        (
            {
                "image_path": "",
                "image_name": "",
                "image": "",
            },
            "review_rows_missing_image_path:row_2",
        ),
        (
            {
                "row_origin": "freeform_review",
            },
            "review_rows_unsupported_row_origin:row_2:freeform_review",
        ),
    ],
)
def test_caption_instruction_review_import_rejects_unmatchable_actionable_rows_atomically(
    monkeypatch,
    tmp_path,
    bad_row_update,
    expected_detail,
) -> None:
    import localinferenceapi as api

    entry = {"id": "ds", "dataset_root": str(tmp_path), "registry_root": str(tmp_path)}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_manifest_for_entry",
        lambda _entry: {
            "labelmap": [],
            "images": [
                {
                    "image_name": "frame.jpg",
                    "image_relpath": "frame.jpg",
                    "split": "train",
                    "label_source_present": True,
                    "label_lines": [],
                }
            ],
        },
    )
    api._write_dataset_caption_instruction_records(
        entry,
        [
            {
                "id": "qa-1",
                "image_name": "frame.jpg",
                "image_key": "train/frame.jpg",
                "split": "train",
                "question": "What is the scene type?",
                "answer": "A waterfront area.",
                "row_type": "generated_qa",
                "answer_source": "vlm_generated",
                "validation_status": "accepted",
            }
        ],
    )

    base_row = {
        "format": "tator_caption_instruction_review_rows_v1",
        "dataset_id": "ds",
        "image_path": "frame.jpg",
        "split": "train",
        "row_origin": "generated_qa",
        "qa_id": "qa-1",
        "row_type": "generated_qa",
        "question": "What is the scene type?",
        "candidate_answer": "A waterfront area.",
        "training_answer": "A waterfront area.",
        "validation_status": "accepted",
        "selected_for_training": True,
        "requires_manual_review": True,
        "review_decision": "accepted",
        "review_notes": "valid first row should not be partially applied",
        "rejection_reasons": [],
        "source_summary": {"status": "empty_label_file"},
    }
    bad_row = {**base_row, **bad_row_update, "review_decision": "rejected"}

    with pytest.raises(api.HTTPException) as excinfo:
        api.apply_caption_instruction_review("ds", {"rows": [base_row, bad_row]})

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == expected_detail
    records = api._load_dataset_caption_instruction_records(entry)
    assert records[0]["review_status"] == ""
    assert "review_decision" not in records[0]["metadata"]


@pytest.mark.parametrize(
    "bad_row_update",
    [
        {"question": "What stale question was reviewed?"},
        {
            "candidate_answer": "A stale reviewed answer.",
            "training_answer": "A stale reviewed answer.",
        },
    ],
)
def test_caption_instruction_review_import_rejects_stale_generated_qa_text(
    monkeypatch,
    tmp_path,
    bad_row_update,
) -> None:
    import localinferenceapi as api

    entry = {"id": "ds", "dataset_root": str(tmp_path), "registry_root": str(tmp_path)}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_manifest_for_entry",
        lambda _entry: {
            "labelmap": [],
            "images": [
                {
                    "image_name": "frame.jpg",
                    "image_relpath": "frame.jpg",
                    "split": "train",
                    "label_source_present": True,
                    "label_lines": [],
                }
            ],
        },
    )
    api._write_dataset_caption_instruction_records(
        entry,
        [
            {
                "id": "qa-1",
                "image_name": "frame.jpg",
                "image_key": "train/frame.jpg",
                "split": "train",
                "question": "What is the scene type?",
                "answer": "A waterfront area.",
                "row_type": "generated_qa",
                "answer_source": "vlm_generated",
                "validation_status": "accepted",
            }
        ],
    )
    row = {
        "format": "tator_caption_instruction_review_rows_v1",
        "dataset_id": "ds",
        "image_path": "frame.jpg",
        "split": "train",
        "row_origin": "generated_qa",
        "qa_id": "qa-1",
        "row_type": "generated_qa",
        "question": "What is the scene type?",
        "candidate_answer": "A waterfront area.",
        "training_answer": "A waterfront area.",
        "validation_status": "accepted",
        "selected_for_training": True,
        "requires_manual_review": True,
        "review_decision": "accepted",
        "review_notes": "stale text should fail",
        "rejection_reasons": [],
        "source_summary": {"status": "empty_label_file"},
        **bad_row_update,
    }

    with pytest.raises(api.HTTPException) as excinfo:
        api.apply_caption_instruction_review("ds", {"rows": [row]})

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "review_rows_generated_qa_not_found:row_1"
    records = api._load_dataset_caption_instruction_records(entry)
    assert records[0]["review_status"] == ""
    assert "review_decision" not in records[0]["metadata"]


def test_caption_instruction_review_import_rejects_stale_caption0_text(
    monkeypatch,
    tmp_path,
) -> None:
    import localinferenceapi as api

    entry = {"id": "ds", "dataset_root": str(tmp_path), "registry_root": str(tmp_path)}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_manifest_for_entry",
        lambda _entry: {
            "labelmap": [],
            "images": [
                {
                    "image_name": "frame.jpg",
                    "image_relpath": "frame.jpg",
                    "split": "train",
                    "label_source_present": True,
                    "label_lines": [],
                }
            ],
        },
    )
    api._write_dataset_caption_records(
        entry,
        [
            {
                "id": "caption-existing",
                "image_name": "frame.jpg",
                "image_key": "train/frame.jpg",
                "split": "train",
                "caption": "Current caption text.",
                "source": "manual",
                "metadata": {},
            }
        ],
    )

    with pytest.raises(api.HTTPException) as excinfo:
        api.apply_caption_instruction_review(
            "ds",
            {
                "rows": [
                    {
                        "format": "tator_caption_instruction_review_rows_v1",
                        "dataset_id": "ds",
                        "image_path": "frame.jpg",
                        "split": "train",
                        "row_origin": "caption0",
                        "qa_id": "caption-existing",
                        "row_type": "caption0",
                        "question": "Describe this image in detail.",
                        "candidate_answer": "Stale caption text.",
                        "training_answer": "Stale caption text.",
                        "validation_status": "accepted",
                        "selected_for_training": True,
                        "requires_manual_review": True,
                        "review_decision": "accepted",
                        "review_notes": "stale review should fail",
                        "rejection_reasons": [],
                        "source_summary": {"status": "empty_label_file"},
                    }
                ]
            },
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "review_rows_caption0_not_found:row_1"
    records = api._load_dataset_caption_records(entry)
    assert len(records) == 1
    assert records[0]["caption"] == "Current caption text."
    assert "review_decision" not in records[0]["metadata"]


@pytest.mark.parametrize(
    "row_update",
    [
        {"qa_id": "primary-review"},
        {"metadata": {"synthetic": True}},
        {"answer_source": "text_label"},
        {
            "qa_id": "primary_0000000000000000",
            "metadata": {"synthetic": True},
            "answer_source": "text_label",
        },
    ],
)
def test_caption_instruction_review_import_rejects_arbitrary_caption0_creation(
    monkeypatch,
    tmp_path,
    row_update,
) -> None:
    import localinferenceapi as api

    entry = {"id": "ds", "dataset_root": str(tmp_path), "registry_root": str(tmp_path)}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_manifest_for_entry",
        lambda _entry: {
            "labelmap": [],
            "images": [
                {
                    "image_name": "frame.jpg",
                    "image_relpath": "frame.jpg",
                    "split": "train",
                    "label_source_present": True,
                    "label_lines": [],
                }
            ],
        },
    )
    synthetic_caption_id = (
        "primary_"
        + hashlib.sha1("ds|train/frame.jpg|A primary caption.".encode("utf-8")).hexdigest()[:16]
    )
    text_label_path = tmp_path / "text_labels" / "frame.txt"
    text_label_path.parent.mkdir(parents=True, exist_ok=True)
    text_label_path.write_text("A primary caption.", encoding="utf-8")
    row = {
        "format": "tator_caption_instruction_review_rows_v1",
        "dataset_id": "ds",
        "image_path": "frame.jpg",
        "split": "train",
        "row_origin": "caption0",
        "qa_id": synthetic_caption_id,
        "row_type": "caption0",
        "question": "Describe this image in detail.",
        "candidate_answer": "A primary caption.",
        "training_answer": "A primary caption.",
        "validation_status": "accepted",
        "selected_for_training": True,
        "requires_manual_review": True,
        "review_decision": "accepted",
        "review_notes": "arbitrary caption creation should fail",
        "rejection_reasons": [],
        "source_summary": {"status": "empty_label_file"},
    }
    row.update(row_update)

    with pytest.raises(api.HTTPException) as excinfo:
        api.apply_caption_instruction_review("ds", {"rows": [row]})

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "review_rows_caption0_creation_not_allowed:row_1"
    assert api._load_dataset_caption_records(entry) == []


def test_caption_instruction_training_readiness_blocks_selected_needs_revision_rows() -> None:
    import localinferenceapi as api

    readiness = api._caption_instruction_training_readiness(
        corpus_quality_metrics={
            "selected_flattened_row_count": 2,
            "image_count": 1,
            "generated_qa_candidate_count": 0,
            "source_class_count": 0,
        },
        review_rows=[
            {
                "selected_for_training": True,
                "requires_manual_review": True,
                "review_decision": "accepted",
            },
            {
                "selected_for_training": True,
                "requires_manual_review": True,
                "review_decision": "needs-revision",
            },
        ],
        settings={"include_generated_qa_in_training": True},
    )

    assert readiness["status"] == "blocked"
    assert readiness["ready_for_training"] is False
    assert readiness["accepted_manual_review_row_count"] == 1
    assert readiness["pending_manual_review_row_count"] == 0
    assert readiness["needs_revision_manual_review_row_count"] == 1
    assert "selected_row_needs_revision_by_manual_review" in readiness["blocking_reasons"]
    assert "revise_selected_language_rows" in readiness["required_actions"]


def test_caption_instruction_training_readiness_blocks_invalid_export_rows() -> None:
    import localinferenceapi as api

    source_archive = api.CAPTION_INSTRUCTION_ARCHIVE_FORMAT
    validation = api._caption_instruction_validate_training_rows(
        [
            {
                "image_path": "frame.jpg",
                "question": "Return the object counts as JSON.",
                "answer": "{not json",
                "metadata": {
                    "qa_id": "qa-1",
                    "row_type": "deterministic_object_count_schema",
                    "answer_source": "source_annotations.object_counts",
                    "answer_format": "object_count_json",
                    "source_archive": source_archive,
                    "validation_status": "machine_validated",
                    "review_status": "machine_validated",
                },
            },
            {
                "image_path": "frame.jpg",
                "question": "Return the object counts as JSON.",
                "answer": "{}",
                "metadata": {
                    "qa_id": "qa-2",
                    "row_type": "deterministic_object_count_schema",
                    "answer_source": "source_annotations.object_counts",
                    "answer_format": "object_count_json",
                    "source_archive": source_archive,
                    "validation_status": "machine_validated",
                    "review_status": "machine_validated",
                },
            },
            {
                "image_path": "frame.jpg",
                "question": "Describe the scene.",
                "answer": "A caption.",
                "metadata": {
                    "qa_id": "qa-3",
                    "row_type": "caption0",
                    "answer_source": "caption_record",
                    "answer_format": "natural",
                    "source_archive": source_archive,
                    "validation_status": "accepted",
                    "review_status": "rejected",
                },
            },
            {
                "image_path": "frame.jpg",
                "question": "Describe the invalid row.",
                "answer": "A caption.",
                "metadata": {
                    "qa_id": "qa-4",
                    "row_type": "caption0",
                    "answer_source": "caption_record",
                    "answer_format": "natural",
                    "source_archive": source_archive,
                    "validation_status": "invalid",
                    "review_status": "unreviewed",
                },
            },
            {
                "image_path": "frame.jpg",
                "question": "Describe the row needing revision.",
                "answer": "A caption.",
                "metadata": {
                    "qa_id": "qa-5",
                    "row_type": "generated_qa",
                    "answer_source": "vlm_generated",
                    "answer_format": "natural",
                    "source_archive": source_archive,
                    "validation_status": "accepted",
                    "review_decision": "needs-revision",
                },
            },
        ]
    )

    assert validation["ok"] is False
    assert validation["error_count"] == 5
    assert "row 1 answer is not valid JSON" in validation["errors"]
    assert "duplicate image_path + question at row 2" in validation["errors"]
    assert "row 3 has non-trainable review status" in validation["errors"]
    assert "row 4 was rejected by archive validation" in validation["errors"]
    assert "row 5 has non-trainable review status" in validation["errors"]

    readiness = api._caption_instruction_training_readiness(
        corpus_quality_metrics={
            "selected_flattened_row_count": 5,
            "image_count": 1,
            "generated_qa_candidate_count": 0,
            "source_class_count": 0,
        },
        review_rows=[],
        settings={"include_generated_qa_in_training": True},
        export_validation=validation,
    )

    assert readiness["status"] == "blocked"
    assert readiness["ready_for_training"] is False
    assert readiness["instruction_export_validation_error_count"] == 5
    assert "instruction_training_rows_invalid" in readiness["blocking_reasons"]
    assert "fix_instruction_training_rows" in readiness["required_actions"]


def test_caption_instruction_artifact_consistency_validator_blocks_mismatched_backend_counts() -> None:
    import localinferenceapi as api

    validation = api._caption_instruction_artifact_consistency_validation(
        training_rows=[
            {
                "image_path": "frame.jpg",
                "question": "Describe this image.",
                "answer": "A caption.",
                "metadata": {"qa_id": "caption0"},
            }
        ],
        archive_rows=[
            {"image_path": "frame.jpg"},
            {"image_path": "frame.jpg"},
        ],
        review_rows=[
            {
                "image_path": "frame.jpg",
                "qa_id": "caption0",
                "selected_for_training": True,
                "requires_manual_review": True,
            }
        ],
        report={
            "format": "tator_caption_instruction_report_v1",
            "image_count": 2,
            "selected_flattened_row_count": 2,
            "instruction_review_row_count": 2,
            "manual_review_required_count": 2,
            "corpus_quality_metrics": {
                "image_count": 2,
                "selected_flattened_row_count": 2,
            },
            "instruction_export_validation": {
                "ok": True,
                "error_count": 0,
                "errors": [],
                "row_count": 2,
            },
        },
        archive_image_count=2,
    )

    assert validation["format"] == "tator_caption_instruction_artifact_consistency_v1"
    assert validation["ok"] is False
    assert validation["error_count"] >= 4
    assert "training row count 1 does not match report selected row count 2" in validation["errors"]
    assert "review row count 1 does not match report review row count 2" in validation["errors"]
    assert "selected review row count 1 does not match report selected row count 2" in validation["errors"]
    assert "manual review row count 1 does not match report manual review count 2" in validation["errors"]
    assert "duplicate archive image_path frame.jpg" in validation["errors"]
    assert validation["counts"]["training_row_count"] == 1
    assert validation["counts"]["archive_row_count"] == 2
    assert validation["counts"]["review_row_count"] == 1


def test_caption_instruction_artifact_consistency_validator_blocks_same_count_identity_mismatches() -> None:
    import localinferenceapi as api

    report = {
        "format": "tator_caption_instruction_report_v1",
        "image_count": 1,
        "selected_flattened_row_count": 1,
        "instruction_review_row_count": 1,
        "manual_review_required_count": 1,
        "corpus_quality_metrics": {
            "image_count": 1,
            "selected_flattened_row_count": 1,
        },
        "instruction_export_validation": {
            "ok": True,
            "error_count": 0,
            "errors": [],
            "row_count": 1,
        },
    }
    validation = api._caption_instruction_artifact_consistency_validation(
        training_rows=[
            {
                "image_path": "frame.jpg",
                "question": "Describe this image in detail.",
                "answer": "A correct caption.",
                "metadata": {"qa_id": "caption0"},
            }
        ],
        archive_rows=[
            {
                "image_path": "frame.jpg",
                "language_annotations": {
                    "caption0": {
                        "qa_id": "caption0",
                        "question": "Describe this image in detail.",
                        "answer": "A stale archive caption.",
                    }
                },
                "deterministic_metadata_qa_pairs": [],
                "export_metadata": {"selected_training_row_count": 1},
            }
        ],
        review_rows=[
            {
                "image_path": "frame.jpg",
                "qa_id": "different-caption0",
                "question": "Describe this image in detail.",
                "training_answer": "A correct caption.",
                "selected_for_training": True,
                "requires_manual_review": True,
            }
        ],
        report=report,
        archive_image_count=1,
    )

    assert validation["ok"] is False
    assert validation["error_count"] >= 3
    assert (
        "training row qa_id caption0 image frame.jpg question 'describe this image in detail.' "
        "is missing from selected review rows"
    ) in validation["errors"]
    assert (
        "selected review row qa_id different-caption0 image frame.jpg question 'describe this image in detail.' "
        "is missing from training rows"
    ) in validation["errors"]
    assert (
        "archive candidate qa_id caption0 image frame.jpg question 'describe this image in detail.' "
        "answer does not match training row answer"
    ) in validation["errors"]
    assert validation["counts"]["training_identity_count"] == 1
    assert validation["counts"]["selected_review_identity_count"] == 1
    assert validation["counts"]["archive_candidate_identity_count"] == 1


def test_caption_instruction_training_validator_requires_complete_row_metadata() -> None:
    import localinferenceapi as api

    source_archive = api.CAPTION_INSTRUCTION_ARCHIVE_FORMAT
    rows = [
        {
            "image_path": "frame.jpg",
            "question": "Describe the row missing validation status.",
            "answer": "A caption.",
            "metadata": {
                "qa_id": "qa-missing-validation",
                "row_type": "caption0",
                "answer_source": "caption_record",
                "answer_format": "natural",
                "source_archive": source_archive,
                "review_status": "unreviewed",
            },
        },
        {
            "image_path": "frame.jpg",
            "question": "Describe the row missing review status.",
            "answer": "A caption.",
            "metadata": {
                "qa_id": "qa-missing-review",
                "row_type": "generated_qa",
                "answer_source": "vlm_generated",
                "answer_format": "natural",
                "source_archive": source_archive,
                "validation_status": "accepted",
            },
        },
        {
            "image_path": "frame.jpg",
            "question": "Describe the row with an unknown validation status.",
            "answer": "A caption.",
            "metadata": {
                "qa_id": "qa-unknown-validation",
                "row_type": "generated_qa",
                "answer_source": "vlm_generated",
                "answer_format": "natural",
                "source_archive": source_archive,
                "validation_status": "maybe",
                "review_status": "unreviewed",
            },
        },
        {
            "image_path": "frame.jpg",
            "question": "Describe the row with an unknown review status.",
            "answer": "A caption.",
            "metadata": {
                "qa_id": "qa-unknown-review",
                "row_type": "generated_qa",
                "answer_source": "vlm_generated",
                "answer_format": "natural",
                "source_archive": source_archive,
                "validation_status": "accepted",
                "review_status": "maybe",
            },
        },
        {
            "image_path": "frame.jpg",
            "question": "Describe the row missing archive provenance.",
            "answer": "A caption.",
            "metadata": {
                "qa_id": "qa-missing-archive",
                "row_type": "generated_qa",
                "answer_source": "vlm_generated",
                "answer_format": "natural",
                "validation_status": "accepted",
                "review_status": "unreviewed",
            },
        },
        {
            "image_path": "frame.jpg",
            "question": "Describe the row with conflicting review fields.",
            "answer": "A caption.",
            "metadata": {
                "qa_id": "qa-conflicting-review",
                "row_type": "generated_qa",
                "answer_source": "vlm_generated",
                "answer_format": "natural",
                "source_archive": source_archive,
                "validation_status": "accepted",
                "review_status": "unreviewed",
                "review_decision": "needs-revision",
            },
        },
    ]

    validation = api._caption_instruction_validate_training_rows(rows)

    assert validation["ok"] is False
    assert validation["error_count"] == 6
    assert "row 1 metadata missing validation_status" in validation["errors"]
    assert "row 2 metadata missing review_status" in validation["errors"]
    assert "row 3 metadata validation_status is unsupported" in validation["errors"]
    assert "row 4 metadata review_status is unsupported" in validation["errors"]
    assert "row 5 metadata missing source_archive" in validation["errors"]
    assert "row 6 has non-trainable review status" in validation["errors"]


def test_caption_instruction_archive_excludes_needs_revision_generated_qa_from_training_rows(
    monkeypatch,
    tmp_path,
) -> None:
    import localinferenceapi as api

    entry = {"id": "ds", "dataset_root": str(tmp_path), "registry_root": str(tmp_path)}
    monkeypatch.setattr(
        api,
        "_annotation_manifest_for_entry",
        lambda _entry: {
            "labelmap": [],
            "images": [
                {
                    "image_name": "frame.jpg",
                    "image_relpath": "frame.jpg",
                    "split": "train",
                    "label_source_present": True,
                    "label_lines": [],
                }
            ],
        },
    )
    caption_records = [
        {
            "id": "cap-1",
            "image_name": "frame.jpg",
            "image_key": "train/frame.jpg",
            "split": "train",
            "caption": "A waterfront scene.",
            "source": "text_label",
            "is_primary": True,
        }
    ]
    instruction_records = [
        {
            "id": "qa-revise",
            "image_name": "frame.jpg",
            "image_key": "train/frame.jpg",
            "split": "train",
            "question": "What is the scene type?",
            "answer": "A waterfront area.",
            "row_type": "generated_qa",
            "answer_source": "vlm_generated",
            "validation_status": "accepted",
            "metadata": {"review_decision": "needs-revision"},
        }
    ]

    archive = api._dataset_caption_instruction_archive(
        caption_records,
        instruction_records,
        dataset_id="ds",
        entry=entry,
        settings=api._caption_instruction_export_settings({}),
        exported_at="2026-01-01T00:00:00Z",
    )

    training_qa_ids = {row["metadata"]["qa_id"] for row in archive["training_rows"]}
    assert training_qa_ids == {"cap-1"}
    generated_review = next(row for row in archive["instruction_review_rows"] if row["row_origin"] == "generated_qa")
    assert generated_review["review_decision"] == "needs-revision"
    assert generated_review["selected_for_training"] is False
    report = archive["captioning_report"]
    assert report["rejection_reason_counts"]["generated_qa_manual_review_needs_revision"] == 1
    assert report["training_readiness"]["status"] == "needs_review"
    assert report["training_readiness"]["pending_manual_review_row_count"] == 1


def test_caption_instruction_training_rows_import_into_qwen_trainer(
    monkeypatch,
    tmp_path,
) -> None:
    import localinferenceapi as api
    from tools import qwen_training as training

    manifest = {
        "labelmap": ["Boat", "Building"],
        "images": [
            {
                "split": "train",
                "image_relpath": "frame.jpg",
                "image_name": "frame.jpg",
                "label_lines": ["0 0.25 0.25 0.10 0.10"],
                "label_source_present": True,
                "label_source": "dataset_label_file",
            }
        ],
    }
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    captions = [
        {
            "id": "caption-1",
            "image_name": "frame.jpg",
            "image_key": "train/frame.jpg",
            "split": "train",
            "caption": "A high-angle scene shows one boat near the shoreline.",
            "source": "qwen_caption_job",
            "is_primary": True,
            "caption_index": 1,
        }
    ]
    generated = [
        {
            "id": "qa-1",
            "image_name": "frame.jpg",
            "image_key": "train/frame.jpg",
            "split": "train",
            "question": "How many boats are visible?",
            "answer": "Two boats are visible.",
            "row_type": "generated_qa",
            "answer_source": "vlm_generated",
        }
    ]
    archive = api._dataset_caption_instruction_archive(
        captions,
        generated,
        dataset_id="ds",
        entry={"id": "ds"},
        settings=api._caption_instruction_export_settings(
            {
                "include_caption0_in_training": True,
                "include_generated_qa_in_training": True,
                "include_deterministic_metadata_qa": True,
            }
        ),
        exported_at="2026-01-01T00:00:00Z",
    )
    training_rows = archive["training_rows"]
    dataset_root = tmp_path / "qwen_dataset"
    image_dir = dataset_root / "train" / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), (40, 50, 60)).save(image_dir / "frame.jpg")
    (dataset_root / "train" / "annotations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in training_rows),
        encoding="utf-8",
    )

    dataset = training.QwenConversationDataset(dataset_root, "train", processor=object())
    row_types = {entry["metadata"]["row_type"] for entry in dataset.entries}
    generated_count = next(
        entry for entry in dataset.entries if entry["metadata"]["row_type"] == "generated_count_validated"
    )

    assert len(dataset) == archive["training_row_count"] == len(training_rows)
    assert {"caption0", "generated_count_validated", "deterministic_count"}.issubset(row_types)
    assert generated_count["conversations"][0] == {
        "from": "human",
        "value": "<image>\nHow many boats are visible?",
    }
    assert json.loads(generated_count["conversations"][1]["value"]) == {"object_counts": {"Boat": 1}}
    item = dataset[0]
    assert item["messages"][0]["content"][0]["type"] == "image"
    assert item["images"][0].size == (8, 8)


def test_caption_instruction_archive_rejects_structured_generated_qa_without_source_labels(
    monkeypatch,
) -> None:
    import localinferenceapi as api

    manifest = {
        "labelmap": ["Boat", "Building"],
        "images": [
            {
                "split": "train",
                "image_relpath": "missing-label.jpg",
                "image_name": "missing-label.jpg",
                "label_lines": [],
                "label_source_present": False,
                "label_source": "missing_label_file",
            }
        ],
    }
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    captions = [
        {
            "id": "caption-1",
            "image_name": "missing-label.jpg",
            "image_key": "train/missing-label.jpg",
            "split": "train",
            "caption": "A caption about the scene.",
            "source": "qwen_caption_job",
            "is_primary": True,
            "caption_index": 1,
        }
    ]
    generated = [
        {
            "id": "qa-1",
            "image_name": "missing-label.jpg",
            "image_key": "train/missing-label.jpg",
            "split": "train",
            "question": "How many boats are visible?",
            "answer": "Two boats are visible.",
            "row_type": "generated_qa",
            "answer_source": "vlm_generated",
        }
    ]

    archive = api._dataset_caption_instruction_archive(
        captions,
        generated,
        dataset_id="ds",
        entry={"id": "ds"},
        settings=api._caption_instruction_export_settings(
            {
                "include_caption0_in_training": False,
                "include_generated_qa_in_training": True,
                "include_deterministic_metadata_qa": True,
            }
        ),
        exported_at="2026-01-01T00:00:00Z",
    )
    image = archive["images"][0]

    assert image["source_annotations"]["status"] == "missing_label_file"
    assert image["source_annotations"]["uncertainty"][0]["type"] == "missing_source_annotations"
    assert image["language_annotations"]["generated_qa_pairs"][0]["validation_status"] == "rejected"
    assert archive["deterministic_metadata_qa_pair_count"] == 0
    assert archive["training_row_count"] == 0
    assert archive["rejection_reason_counts"] == {"generated_qa_validation_rejected": 1}
    assert archive["captioning_report"]["rejected_generated_qa_count"] == 1
    assert archive["captioning_report"]["rows_excluded_because_generated_answers_contradicted_source_annotations"] == 1
    assert archive["archive_rows"][0]["export_metadata"]["selected_training_row_count"] == 0


def test_caption_instruction_archive_rejects_caption0_count_contradiction(
    monkeypatch,
) -> None:
    import localinferenceapi as api

    manifest = {
        "labelmap": ["Boat"],
        "images": [
            {
                "split": "train",
                "image_relpath": "frame.jpg",
                "image_name": "frame.jpg",
                "label_lines": ["0 0.5 0.5 0.1 0.1"],
                "label_source_present": True,
                "label_source": "dataset_label_file",
            }
        ],
    }
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    captions = [
        {
            "id": "caption-1",
            "image_name": "frame.jpg",
            "image_key": "train/frame.jpg",
            "split": "train",
            "caption": "A top-down view shows two boats near the dock.",
            "source": "qwen_caption_job",
            "is_primary": True,
            "caption_index": 1,
        }
    ]

    archive = api._dataset_caption_instruction_archive(
        captions,
        [],
        dataset_id="ds",
        entry={"id": "ds"},
        settings=api._caption_instruction_export_settings(
            {
                "include_caption0_in_training": True,
                "include_generated_qa_in_training": True,
                "include_deterministic_metadata_qa": False,
            }
        ),
        exported_at="2026-01-01T00:00:00Z",
    )
    caption0 = archive["images"][0]["language_annotations"]["caption0"]

    assert caption0["validation_status"] == "rejected"
    assert caption0["rejection_reasons"] == ["caption0_count_contradicts_source_annotations"]
    assert caption0["validation"]["exact_count_claims"] == [
        {"class_name": "Boat", "count": 2, "text": "two boats"}
    ]
    assert archive["training_rows"] == []
    assert archive["training_row_count"] == 0
    assert archive["rejection_reason_counts"] == {"caption0_validation_rejected": 1}
    assert archive["captioning_report"]["rows_excluded_because_generated_answers_contradicted_source_annotations"] == 1
    assert archive["archive_rows"][0]["export_metadata"]["selected_training_row_count"] == 0


def test_caption_instruction_archive_keeps_non_manifest_records_out_of_training_rows(
    monkeypatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"labelmap": ["Boat"], "images": []})
    captions = [
        {
            "id": "caption-1",
            "image_name": "orphan.jpg",
            "image_key": "train/orphan.jpg",
            "split": "train",
            "caption": "A caption for an image that is no longer in the manifest.",
            "source": "qwen_caption_job",
            "is_primary": True,
            "caption_index": 1,
        }
    ]
    generated = [
        {
            "id": "qa-1",
            "image_name": "orphan.jpg",
            "image_key": "train/orphan.jpg",
            "split": "train",
            "question": "What setting is shown?",
            "answer": "A waterfront setting is shown.",
            "row_type": "generated_qa",
            "answer_source": "vlm_generated",
        }
    ]

    archive = api._dataset_caption_instruction_archive(
        captions,
        generated,
        dataset_id="ds",
        entry={"id": "ds"},
        settings=api._caption_instruction_export_settings(
            {
                "include_caption0_in_training": True,
                "include_generated_qa_in_training": True,
                "include_deterministic_metadata_qa": True,
            }
        ),
        exported_at="2026-01-01T00:00:00Z",
    )
    image = archive["images"][0]
    generated_pair = image["language_annotations"]["generated_qa_pairs"][0]

    assert image["source_annotations"]["status"] == "source_manifest_row_missing"
    assert image["export_metadata"]["flattening_eligible"] is False
    assert image["export_metadata"]["selected_training_row_count"] == 0
    assert generated_pair["validation_status"] == "accepted"
    assert generated_pair["validated_against"] == ["image", "language_annotations.caption0"]
    assert archive["training_rows"] == []
    assert archive["training_row_count"] == 0
    assert archive["deterministic_metadata_qa_pair_count"] == 0
    assert archive["rejection_reason_counts"] == {"source_manifest_row_missing": 1}
    assert archive["captioning_report"]["selected_flattened_row_count"] == 0


def test_caption_instruction_archive_rewrites_supported_structured_generated_qa_from_source_labels(
    monkeypatch,
) -> None:
    import localinferenceapi as api

    manifest = {
        "labelmap": ["Boat", "Building"],
        "images": [
            {
                "split": "train",
                "image_relpath": "frame.jpg",
                "image_name": "frame.jpg",
                "label_lines": ["0 0.5 0.5 0.1 0.1"],
                "label_source_present": True,
                "label_source": "dataset_label_file",
            }
        ],
    }
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    captions = [
        {
            "id": "caption-1",
            "image_name": "frame.jpg",
            "image_key": "train/frame.jpg",
            "split": "train",
            "caption": "A caption about a waterfront.",
            "source": "qwen_caption_job",
            "is_primary": True,
            "caption_index": 1,
        }
    ]
    generated = [
        {
            "id": "qa-1",
            "image_name": "frame.jpg",
            "image_key": "train/frame.jpg",
            "split": "train",
            "question": "How many boats are visible?",
            "answer": "Two boats are visible.",
            "row_type": "generated_qa",
            "answer_source": "vlm_generated",
        }
    ]

    archive = api._dataset_caption_instruction_archive(
        captions,
        generated,
        dataset_id="ds",
        entry={"id": "ds"},
        settings=api._caption_instruction_export_settings(
            {
                "include_caption0_in_training": False,
                "include_generated_qa_in_training": True,
                "include_deterministic_metadata_qa": False,
            }
        ),
        exported_at="2026-01-01T00:00:00Z",
    )

    generated_pair = archive["images"][0]["language_annotations"]["generated_qa_pairs"][0]
    assert generated_pair["validation_status"] == "accepted"
    assert generated_pair["row_type"] == "generated_count_validated"
    assert generated_pair["answer_source"] == "source_annotations.object_counts.Boat"
    assert json.loads(generated_pair["answer"]) == {"object_counts": {"Boat": 1}}
    assert generated_pair["metadata"]["candidate_answer"] == "Two boats are visible."
    assert archive["training_row_count"] == 1
    assert archive["training_rows"][0]["metadata"]["row_type"] == "generated_count_validated"
    assert archive["training_rows"][0]["metadata"]["answer_format"] == "object_count_json"
    assert archive["training_rows"][0]["metadata"]["source_fields"] == [
        "source_annotations.object_counts.Boat"
    ]
    assert json.loads(archive["training_rows"][0]["answer"]) == {"object_counts": {"Boat": 1}}
    metrics = archive["corpus_quality_metrics"]
    assert metrics["structured_rewrite_count"] == 1
    assert metrics["structured_rewrite_rate"] == 1.0
    assert metrics["source_validated_training_row_count"] == 1
    assert metrics["source_classes_covered_by_training_rows"] == ["Boat"]
    assert metrics["source_class_coverage_rate"] == 1.0


def test_caption_dataset_runner_summary_counts_completed_cases_not_attempts() -> None:
    import localinferenceapi as api

    counts = api._qwen_caption_dataset_result_summary_from_runner_summary(
        {
            "rows": [
                {"case_id": "a", "final_status": "ok"},
                {"case_id": "b", "final_status": "failed", "quality_failures": ["missing counts"]},
                {"case_id": "c", "final_status": "skipped_existing_caption"},
            ]
        }
    )

    assert counts == {
        "processed": 3,
        "failed": 1,
        "quality_failed": 1,
        "runner_totals": {"ok": 1, "failed": 1, "skipped_existing_caption": 1},
        "degraded_rates": {
            "processed_cases": 3,
            "attempt_rows": 3,
            "failed_cases": 1,
            "quality_failed_cases": 1,
            "failed_attempt_rows": 0,
            "signal_exit_attempt_rows": 0,
            "recovery_event_cases": 0,
            "recovery_events": 0,
            "loop_recovery_cases": 0,
            "deterministic_recovery_cases": 0,
            "prompt_budget_adapted_cases": 0,
            "max_prompt_tokens": 0,
            "stream_loop_detected_cases": 0,
            "stream_loop_detected_events": 0,
            "loop_trim_cases": 0,
            "loop_trim_events": 0,
            "failed_case_rate": 1 / 3,
            "quality_failure_rate": 1 / 3,
            "failed_attempt_row_rate": 0.0,
            "signal_exit_attempt_row_rate": 0.0,
            "recovery_event_case_rate": 0.0,
            "loop_recovery_case_rate": 0.0,
            "deterministic_recovery_case_rate": 0.0,
            "stream_loop_detected_case_rate": 0.0,
            "loop_trim_case_rate": 0.0,
        },
    }


def test_caption_dataset_runner_summary_uses_totals_when_rows_are_truncated() -> None:
    import localinferenceapi as api

    counts = api._qwen_caption_dataset_result_summary_from_runner_summary(
        {
            "total_cases": 500,
            "quality_failed_cases": 3,
            "totals": {
                "ok": 490,
                "failed": 8,
                "skipped_existing_caption": 2,
            },
            "row_count": 500,
            "row_limit": 2,
            "rows_truncated": True,
            "rows": [
                {"case_id": "recent_a", "final_status": "ok"},
                {"case_id": "recent_b", "final_status": "failed"},
            ],
        }
    )

    assert counts == {
        "processed": 500,
        "failed": 8,
        "quality_failed": 3,
        "runner_totals": {"ok": 490, "failed": 8, "skipped_existing_caption": 2},
    }


def test_caption_dataset_degraded_rates_from_rows_tracks_retries_and_recovery() -> None:
    import localinferenceapi as api

    rates = api._qwen_caption_dataset_degraded_rates_from_rows(
        [
            {
                "case_id": "a",
                "final_status": "failed_attempt",
                "status": "exception",
                "attempt_failure_kind": "signal_exit",
                "return_signal": 6,
                "return_signal_name": "SIGABRT",
                "quality_failures": [],
                "recovery_events": [],
            },
            {
                "case_id": "a",
                "final_status": "ok",
                "status": "ok",
                "quality_failures": [],
                "recovery_events": [{"action": "loop_detected"}],
                "preview_prompt_budget": {"adapted_sections": 2, "max_prompt_tokens": 9000},
                "qwen_caption_io": {
                    "event_counts": {
                        "stream_loop_detected": 1,
                        "loop_trim": 1,
                    }
                },
            },
            {
                "case_id": "b",
                "final_status": "ok",
                "status": "ok",
                "quality_failures": ["missing counts"],
                "recovery_events": [{"action": "deterministic_recovery_succeeded"}],
            },
        ]
    )

    assert rates["processed_cases"] == 2
    assert rates["attempt_rows"] == 3
    assert rates["failed_cases"] == 0
    assert rates["quality_failed_cases"] == 1
    assert rates["failed_attempt_rows"] == 1
    assert rates["signal_exit_attempt_rows"] == 1
    assert rates["recovery_event_cases"] == 2
    assert rates["recovery_events"] == 2
    assert rates["loop_recovery_cases"] == 1
    assert rates["deterministic_recovery_cases"] == 1
    assert rates["prompt_budget_adapted_cases"] == 1
    assert rates["max_prompt_tokens"] == 9000
    assert rates["stream_loop_detected_cases"] == 1
    assert rates["stream_loop_detected_events"] == 1
    assert rates["loop_trim_cases"] == 1
    assert rates["loop_trim_events"] == 1
    assert rates["quality_failure_rate"] == 0.5
    assert rates["failed_attempt_row_rate"] == 1 / 3
    assert rates["signal_exit_attempt_row_rate"] == 1 / 3
    assert rates["recovery_event_case_rate"] == 1.0
    assert rates["loop_recovery_case_rate"] == 0.5
    assert rates["deterministic_recovery_case_rate"] == 0.5
    assert rates["stream_loop_detected_case_rate"] == 0.5
    assert rates["loop_trim_case_rate"] == 0.5


def test_caption_dataset_degraded_policy_defers_low_sample_loop_rate() -> None:
    import localinferenceapi as api

    policy = api._qwen_caption_dataset_degraded_policy(
        {
            "processed_cases": 28,
            "failed_case_rate": 0.0,
            "quality_failure_rate": 0.0,
            "recovery_event_case_rate": 2 / 28,
            "loop_recovery_case_rate": 2 / 28,
            "failed_attempt_row_rate": 0.0,
        },
        {
            "set_and_forget": True,
            "max_recovery_event_case_rate": 0.25,
            "min_rate_cases": 20,
        },
        terminal=False,
    )

    assert policy["active"] is True
    assert policy["min_loop_recovery_rate_cases"] == 60
    assert policy["violations"][0]["rate"] == "loop_recovery_case_rate"
    assert policy["violations"][0]["active"] is False
    assert policy["violations"][0]["deferred_by_sample_floor"] is True
    assert policy["active_violations"] == []


def test_caption_dataset_degraded_policy_enforces_terminal_loop_rate() -> None:
    import localinferenceapi as api

    policy = api._qwen_caption_dataset_degraded_policy(
        {
            "processed_cases": 28,
            "failed_case_rate": 0.0,
            "quality_failure_rate": 0.0,
            "recovery_event_case_rate": 2 / 28,
            "loop_recovery_case_rate": 2 / 28,
            "failed_attempt_row_rate": 0.0,
        },
        {
            "set_and_forget": True,
            "max_recovery_event_case_rate": 0.25,
            "min_rate_cases": 20,
        },
        terminal=True,
    )

    assert policy["active_violations"][0]["rate"] == "loop_recovery_case_rate"


def test_caption_dataset_degraded_policy_enforces_terminal_deterministic_recovery_rate() -> None:
    import localinferenceapi as api

    policy = api._qwen_caption_dataset_degraded_policy(
        {
            "processed_cases": 100,
            "failed_case_rate": 0.0,
            "quality_failure_rate": 0.0,
            "recovery_event_case_rate": 0.02,
            "loop_recovery_case_rate": 0.02,
            "deterministic_recovery_case_rate": 0.02,
            "failed_attempt_row_rate": 0.0,
        },
        {
            "set_and_forget": True,
            "max_recovery_event_case_rate": 0.25,
            "max_loop_recovery_case_rate": 0.05,
            "max_deterministic_recovery_case_rate": 0.01,
            "min_rate_cases": 20,
        },
        terminal=True,
    )

    assert policy["thresholds"]["max_deterministic_recovery_case_rate"] == 0.01
    assert policy["active_violations"][0]["rate"] == "deterministic_recovery_case_rate"


def test_caption_dataset_effective_request_fields_resolve_implicit_model() -> None:
    import localinferenceapi as api

    fields, model_fields = api._qwen_caption_dataset_effective_request_fields(
        {
            "user_prompt": "Describe it",
            "model_id": "auto",
            "model_variant": "Thinking",
            "caption_fallback_model_id": "",
            "caption_loop_recovery_mode": "bad",
        }
    )

    expected_model = api._resolve_qwen_variant_model_id_impl(
        api.QWEN_MLX_CAPTION_MODEL_NAME,
        "Thinking",
    )
    assert fields["model_id"] == expected_model
    assert fields["model_variant"] == "Thinking"
    assert fields["caption_fallback_model_id"] == "auto"
    assert fields["caption_loop_recovery_mode"] == "safe_retry_fallback"
    assert model_fields == {
        "model_id": expected_model,
        "model_variant": "Thinking",
        "refinement_model_id": "same",
        "fallback_model_id": "auto",
        "loop_recovery": "safe_retry_fallback",
    }


def test_caption_dataset_effective_request_fields_defaults_windowed_set_and_forget_to_text_only() -> None:
    import localinferenceapi as api

    auto_fields, _model_fields = api._qwen_caption_dataset_effective_request_fields(
        {
            "user_prompt": "Describe it",
            "caption_mode": "windowed",
        },
        set_and_forget=True,
    )
    explicit_fields, _explicit_model_fields = api._qwen_caption_dataset_effective_request_fields(
        {
            "user_prompt": "Describe it",
            "caption_mode": "windowed",
            "caption_windowed_full_image_strategy": "visual",
        },
        set_and_forget=True,
    )
    full_fields, _full_model_fields = api._qwen_caption_dataset_effective_request_fields(
        {
            "user_prompt": "Describe it",
            "caption_mode": "full",
        },
        set_and_forget=True,
    )

    assert auto_fields["caption_windowed_full_image_strategy"] == "text_only"
    assert explicit_fields["caption_windowed_full_image_strategy"] == "visual"
    assert full_fields["caption_windowed_full_image_strategy"] == "visual"


def test_caption_dataset_runner_heartbeat_read_and_message(tmp_path: Path) -> None:
    import localinferenceapi as api

    heartbeat_path = tmp_path / "heartbeat.json"
    heartbeat_path.write_text(
        json.dumps(
            {
                "seq": 3,
                "phase": "attempt_running",
                "case": "image_000123",
                "attempt": 2,
                "processed": 122,
                "total_cases": 10000,
            }
        )
    )

    mtime, heartbeat = api._qwen_caption_dataset_read_runner_heartbeat(
        heartbeat_path,
        last_mtime=0.0,
    )

    assert heartbeat is not None
    assert heartbeat["seq"] == 3
    message = api._qwen_caption_dataset_runner_heartbeat_message(heartbeat)
    assert "attempt 2" in message
    assert "122/10000 complete" in message
    progress_message = api._qwen_caption_dataset_runner_heartbeat_message(
        {
            **heartbeat,
            "worker_progress": {
                "step_label": "Compose full-image caption",
                "message": "Generating response tokens",
                "generated_tokens": 120,
                "max_new_tokens": 3000,
            },
        }
    )
    assert "Qwen: Compose full-image caption: Generating response tokens" in progress_message
    assert "120/3000 tokens" in progress_message
    next_mtime, next_heartbeat = api._qwen_caption_dataset_read_runner_heartbeat(
        heartbeat_path,
        last_mtime=mtime,
    )
    assert next_mtime == mtime
    assert next_heartbeat is None


def test_caption_dataset_job_preflight_error_blocks_runner_launch(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api
    from tools import preflight_qwen_caption_soak as preflight

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start after preflight failure")

    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )
    monkeypatch.setattr(
        preflight,
        "preflight_soak",
        lambda _args: {
            "status": "error",
            "checks": [
                {
                    "name": "disk_budget",
                    "status": "error",
                    "detail": "free disk is below estimated need",
                }
            ],
        },
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_preflight", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "failed"
    assert job.error == "caption_runner_preflight_failed"
    assert "free disk is below estimated need" in job.message
    assert job.result["preflight"]["status"] == "error"


def test_caption_dataset_job_large_set_and_forget_requires_pilot_certification(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api
    from tools import preflight_qwen_caption_soak as preflight

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start without required pilot certification")

    def fail_preflight(*_args, **_kwargs):
        raise AssertionError("preflight should not run before the large-run pilot gate")

    cases = [
        {
            "case_id": f"image:train/frame_{index}.jpg:full",
            "name": f"image_{index:06d}",
            "stem": f"frame_{index}",
            "image_path": str(tmp_path / f"frame_{index}.jpg"),
            "label_path": str(tmp_path / f"frame_{index}.txt"),
            "label_count": 1,
            "class_counts": {"Building": 1},
            "caption_mode": "full",
        }
        for index in range(3)
    ]

    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_REQUIRE_PILOT_CASES", 3)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(preflight, "preflight_soak", fail_preflight)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: ({"dataset_label": "Dataset"}, cases),
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_pilot_required", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        require_pilot_certification=False,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "failed"
    assert job.error == "caption_runner_pilot_required"
    assert "Certified pilot required" in job.message
    assert job.result["total_cases"] == 3
    assert "preflight" not in job.result
    pilot_report = job.result["required_pilot_certification"]
    assert pilot_report["status"] == "error"
    assert pilot_report["required"] is True
    assert pilot_report["threshold_cases"] == 3
    assert pilot_report["checks"][0]["name"] == "set_and_forget_pilot_required"


def test_caption_dataset_job_large_set_and_forget_rejects_weak_pilot_gates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api
    from tools import certify_qwen_caption_soak as certify
    from tools import preflight_qwen_caption_soak as preflight

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start with weak pilot gates")

    def fail_preflight(*_args, **_kwargs):
        raise AssertionError("preflight should not run before the large-run pilot gate")

    def fail_certify(*_args, **_kwargs):
        raise AssertionError("certification should not run before pilot gate validation")

    cases = [
        {
            "case_id": f"image:train/frame_{index}.jpg:full",
            "name": f"image_{index:06d}",
            "stem": f"frame_{index}",
            "image_path": str(tmp_path / f"frame_{index}.jpg"),
            "label_path": str(tmp_path / f"frame_{index}.txt"),
            "label_count": 1,
            "class_counts": {"Building": 1},
            "caption_mode": "full",
        }
        for index in range(3)
    ]

    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_REQUIRE_PILOT_CASES", 3)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(preflight, "preflight_soak", fail_preflight)
    monkeypatch.setattr(certify, "certify_soak", fail_certify)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: ({"dataset_label": "Dataset"}, cases),
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_weak_pilot_gates", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        require_pilot_certification=True,
        pilot_output_dir=str(tmp_path / "pilot"),
        pilot_min_cases=1,
        pilot_require_prompt_budget_data=False,
        pilot_max_prompt_tokens=0,
        pilot_max_p95_duration_hours=-1,
        pilot_max_prompt_budget_adapted_case_rate=-1,
        pilot_deterministic_recovery_confidence=0,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "failed"
    assert job.error == "caption_runner_pilot_certification_failed"
    assert "Pilot certification configuration failed" in job.message
    assert "preflight" not in job.result
    report = job.result["required_pilot_certification"]
    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert report["total_cases"] == 3
    assert checks["pilot_output_dir_isolated"]["status"] == "ok"
    assert checks["pilot_min_cases"]["status"] == "error"
    assert checks["pilot_prompt_budget_required"]["status"] == "error"
    assert checks["pilot_prompt_size_ceiling"]["status"] == "error"
    assert checks["pilot_p95_duration_gate"]["status"] == "error"
    assert checks["pilot_prompt_budget_adaptation_gate"]["status"] == "error"
    assert checks["pilot_deterministic_recovery_confidence"]["status"] == "error"


def test_caption_dataset_job_large_set_and_forget_requires_backend_supervision(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api
    from tools import certify_qwen_caption_soak as certify
    from tools import preflight_qwen_caption_soak as preflight

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start without backend crash supervision")

    def fail_preflight(*_args, **_kwargs):
        raise AssertionError("preflight should not run before backend supervision gate")

    def fail_certify(*_args, **_kwargs):
        raise AssertionError("pilot certification should not run before backend supervision gate")

    cases = [
        {
            "case_id": f"image:train/frame_{index}.jpg:full",
            "name": f"image_{index:06d}",
            "stem": f"frame_{index}",
            "image_path": str(tmp_path / f"frame_{index}.jpg"),
            "label_path": str(tmp_path / f"frame_{index}.txt"),
            "label_count": 1,
            "class_counts": {"Building": 1},
            "caption_mode": "full",
        }
        for index in range(3)
    ]

    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_REQUIRE_PILOT_CASES", 3)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_REQUIRE_BACKEND_SUPERVISION", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(preflight, "preflight_soak", fail_preflight)
    monkeypatch.setattr(certify, "certify_soak", fail_certify)
    monkeypatch.setattr(
        api,
        "_qwen_backend_supervision_status",
        lambda: {
            "launcher": None,
            "restart_capable": False,
            "restart_on_crash": False,
            "message": "Backend process is not advertising crash-restart supervision.",
        },
    )
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: ({"dataset_label": "Dataset"}, cases),
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_supervision_required", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        require_pilot_certification=True,
        pilot_output_dir=str(tmp_path / "pilot"),
        pilot_min_cases=api.QWEN_CAPTION_DEFAULT_PILOT_MIN_CASES,
        pilot_require_prompt_budget_data=True,
        pilot_max_prompt_tokens=8400,
        pilot_max_p95_duration_hours=72,
        pilot_max_prompt_budget_adapted_case_rate=0.2,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "failed"
    assert job.error == "caption_runner_backend_supervision_required"
    assert "Backend supervision required" in job.message
    assert "preflight" not in job.result
    report = job.result["backend_supervision"]
    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert report["total_cases"] == 3
    assert checks["backend_crash_supervision"]["status"] == "error"
    assert checks["backend_crash_supervision"]["restart_capable"] is False


def test_caption_dataset_job_large_set_and_forget_requires_backend_restart_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api
    from tools import certify_qwen_caption_soak as certify
    from tools import preflight_qwen_caption_soak as preflight

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start with an under-sized backend restart policy")

    def fail_preflight(*_args, **_kwargs):
        raise AssertionError("preflight should not run before backend restart policy gate")

    def fail_certify(*_args, **_kwargs):
        raise AssertionError("pilot certification should not run before backend restart policy gate")

    cases = [
        {
            "case_id": f"image:train/frame_{index}.jpg:full",
            "name": f"image_{index:06d}",
            "stem": f"frame_{index}",
            "image_path": str(tmp_path / f"frame_{index}.jpg"),
            "label_path": str(tmp_path / f"frame_{index}.txt"),
            "label_count": 1,
            "class_counts": {"Building": 1},
            "caption_mode": "full",
        }
        for index in range(3)
    ]

    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_REQUIRE_PILOT_CASES", 3)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_REQUIRE_BACKEND_SUPERVISION", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setenv("TATOR_BACKEND_LAUNCHER", "tools/run_macos_backend.sh")
    monkeypatch.setenv("TATOR_BACKEND_LAUNCHER_RESTARTS_CRASHES", "1")
    monkeypatch.setenv("TATOR_BACKEND_LAUNCHER_RESTART_MAX", "1")
    monkeypatch.setenv("TATOR_BACKEND_LAUNCHER_RESTART_DELAY", "1")
    monkeypatch.setenv("TATOR_BACKEND_LAUNCHER_RESTART_MAX_DELAY", "30")
    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(preflight, "preflight_soak", fail_preflight)
    monkeypatch.setattr(certify, "certify_soak", fail_certify)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: ({"dataset_label": "Dataset"}, cases),
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_restart_policy_required", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        require_pilot_certification=True,
        pilot_output_dir=str(tmp_path / "pilot"),
        pilot_min_cases=api.QWEN_CAPTION_DEFAULT_PILOT_MIN_CASES,
        pilot_require_prompt_budget_data=True,
        pilot_max_prompt_tokens=8400,
        pilot_max_p95_duration_hours=72,
        pilot_max_prompt_budget_adapted_case_rate=0.2,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "failed"
    assert job.error == "caption_runner_backend_supervision_required"
    assert "Backend supervision required" in job.message
    assert "preflight" not in job.result
    report = job.result["backend_supervision"]
    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["backend_crash_supervision"]["status"] == "ok"
    assert checks["backend_restart_policy"]["status"] == "error"
    assert checks["backend_restart_policy"]["restart_policy_ready"] is False
    policy_checks = {check["name"]: check for check in checks["backend_restart_policy"]["policy_checks"]}
    assert policy_checks["restart_count_budget"]["status"] == "error"
    assert policy_checks["restart_count_budget"]["restart_max"] == 1


def test_caption_dataset_job_preflight_inputs_do_not_touch_explicit_artifact_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api
    from tools import preflight_qwen_caption_soak as preflight

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start after preflight failure")

    artifact_dir = tmp_path / "shared_artifacts"
    metadata_dir = tmp_path / "job_metadata"
    artifact_dir.mkdir()
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )

    def fake_preflight(args):
        assert args.output_dir == artifact_dir
        assert args.cases_json == metadata_dir / "cases.json"
        assert args.request_json == metadata_dir / "request_fields.json"
        assert not (artifact_dir / "cases.json").exists()
        assert not (artifact_dir / "request_fields.json").exists()
        return {
            "status": "error",
            "checks": [
                {
                    "name": "runner_lock",
                    "status": "error",
                    "detail": "output directory is currently owned by a live runner",
                }
            ],
        }

    monkeypatch.setattr(preflight, "preflight_soak", fake_preflight)
    job = api.QwenCaptionDatasetJob(job_id="qcap_explicit_artifact_preflight", output_dir=str(metadata_dir))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        output_dir=str(artifact_dir),
        attempts=1,
        per_image_timeout_seconds=30,
        allow_model_download=True,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "failed"
    assert job.error == "caption_runner_preflight_failed"
    assert (metadata_dir / "cases.json").exists()
    assert (metadata_dir / "request_fields.json").exists()
    assert not (artifact_dir / "cases.json").exists()
    assert not (artifact_dir / "request_fields.json").exists()


def test_caption_dataset_job_pilot_certification_error_blocks_runner_launch(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api
    from tools import preflight_qwen_caption_soak as preflight

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start after pilot certification failure")

    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )
    monkeypatch.setattr(preflight, "preflight_soak", lambda _args: {"status": "ok", "checks": []})
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_required_pilot_certification",
        lambda *_args, **_kwargs: {
            "status": "error",
            "checks": [
                {
                    "name": "pilot_sample_size",
                    "status": "error",
                    "detail": "2 timed cases is below minimum 50",
                }
            ],
        },
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_pilot_gate", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        require_pilot_certification=True,
        pilot_output_dir=str(tmp_path / "pilot"),
        pilot_require_prompt_budget_data=False,
        pilot_max_p95_duration_hours=72,
        pilot_max_prompt_tokens=8400,
        pilot_max_prompt_budget_adapted_case_rate=0.2,
        pilot_deterministic_recovery_confidence=0.8,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "failed"
    assert job.error == "caption_runner_pilot_certification_failed"
    assert "2 timed cases is below minimum 50" in job.message
    assert job.result["required_pilot_certification"]["status"] == "error"


def test_caption_dataset_job_pilot_certification_requires_generated_evidence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api
    from tools import certify_qwen_caption_soak as certify
    from tools import preflight_qwen_caption_soak as preflight

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start after missing generated pilot evidence")

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(preflight, "preflight_soak", lambda _args: {"status": "ok", "checks": []})
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )
    monkeypatch.setattr(
        certify,
        "certify_soak",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "target_cases": 10000,
            "pilot_cases": 100,
            "prompt_budget": {
                "pilot_cases": 100,
                "rows_with_prompt_budget": 100,
                "max_prompt_tokens": 1200,
                "adapted_case_rate": 0.0,
            },
            "qwen_caption_io_sources": {
                "required_rows": 100,
                "runtime_prompt_budget_rows": 100,
                "valid_runtime_prompt_budget_rows": 100,
                "invalid_runtime_rows_count": 0,
                "missing_runtime_rows_count": 0,
                "source_counts": {"qwen_caption_io_per_run": 100},
                "accepted_sources": ["qwen_caption_io_per_run"],
            },
            "checks": [
                {"name": "artifact_audit", "status": "ok"},
                {"name": "qwen_caption_io_source", "status": "ok"},
            ],
        },
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_missing_generated_pilot", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        require_pilot_certification=True,
        pilot_output_dir=str(tmp_path / "pilot"),
        pilot_min_cases=50,
        pilot_require_prompt_budget_data=True,
        pilot_max_p95_duration_hours=72,
        pilot_max_prompt_tokens=8400,
        pilot_max_prompt_budget_adapted_case_rate=0.2,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    report = job.result["required_pilot_certification"]
    checks = {check["name"]: check for check in report["checks"]}
    generated_gate = checks["backend_generated_pilot_evidence_gate"]
    assert job.status == "failed"
    assert job.error == "caption_runner_pilot_certification_failed"
    assert report["status"] == "error"
    assert generated_gate["status"] == "error"
    assert generated_gate["generated_pilot_cases"] == 0
    assert generated_gate["min_pilot_cases"] == 50


def test_caption_dataset_job_pilot_certification_requires_qwen_caption_io_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api
    from tools import certify_qwen_caption_soak as certify
    from tools import preflight_qwen_caption_soak as preflight

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start after stale pilot certification")

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(preflight, "preflight_soak", lambda _args: {"status": "ok", "checks": []})
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )
    monkeypatch.setattr(
        certify,
        "certify_soak",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "target_cases": 10000,
            "pilot_cases": 100,
            "generated_pilot_cases": 100,
            "generated_case_evidence": {
                "latest_cases": 100,
                "generated_cases": 100,
                "latest_skipped_cases": 0,
                "latest_skipped_cases_with_prior_generated_success": 0,
            },
            "prompt_budget": {
                "pilot_cases": 100,
                "rows_with_prompt_budget": 100,
                "max_prompt_tokens": 1200,
                "adapted_case_rate": 0.0,
            },
            "checks": [{"name": "prompt_budget_data", "status": "ok"}],
        },
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_stale_pilot_source", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        require_pilot_certification=True,
        pilot_output_dir=str(tmp_path / "pilot"),
        pilot_min_cases=100,
        pilot_require_prompt_budget_data=True,
        pilot_max_p95_duration_hours=72,
        pilot_max_prompt_tokens=8400,
        pilot_max_prompt_budget_adapted_case_rate=0.2,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    report = job.result["required_pilot_certification"]
    checks = {check["name"]: check for check in report["checks"]}
    assert job.status == "failed"
    assert job.error == "caption_runner_pilot_certification_failed"
    assert report["status"] == "error"
    assert checks["backend_qwen_caption_io_source_gate"]["status"] == "error"
    assert checks["backend_qwen_caption_io_source_gate"]["has_source_report"] is False


def test_caption_dataset_job_pilot_certification_rejects_unbound_qwen_latest_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api
    from tools import certify_qwen_caption_soak as certify
    from tools import preflight_qwen_caption_soak as preflight

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start with unbound qwen latest evidence")

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(preflight, "preflight_soak", lambda _args: {"status": "ok", "checks": []})
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )
    monkeypatch.setattr(
        certify,
        "certify_soak",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "target_cases": 10000,
            "pilot_cases": 100,
            "generated_pilot_cases": 100,
            "generated_case_evidence": {
                "latest_cases": 100,
                "generated_cases": 100,
                "latest_skipped_cases": 0,
                "latest_skipped_cases_with_prior_generated_success": 0,
            },
            "prompt_budget": {
                "pilot_cases": 100,
                "rows_with_prompt_budget": 100,
                "max_prompt_tokens": 1200,
                "adapted_case_rate": 0.0,
            },
            "qwen_caption_io_sources": {
                "required_rows": 100,
                "runtime_prompt_budget_rows": 100,
                "valid_runtime_prompt_budget_rows": 100,
                "invalid_runtime_rows_count": 0,
                "missing_runtime_rows_count": 0,
                "source_counts": {"qwen_caption_io_latest": 100},
                "accepted_sources": ["qwen_caption_io_per_run"],
            },
            "checks": [{"name": "qwen_caption_io_source", "status": "ok"}],
        },
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_unbound_pilot_source", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        require_pilot_certification=True,
        pilot_output_dir=str(tmp_path / "pilot"),
        pilot_min_cases=100,
        pilot_require_prompt_budget_data=True,
        pilot_max_p95_duration_hours=72,
        pilot_max_prompt_tokens=8400,
        pilot_max_prompt_budget_adapted_case_rate=0.2,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    report = job.result["required_pilot_certification"]
    checks = {check["name"]: check for check in report["checks"]}
    source_gate = checks["backend_qwen_caption_io_source_gate"]
    assert job.status == "failed"
    assert job.error == "caption_runner_pilot_certification_failed"
    assert report["status"] == "error"
    assert source_gate["status"] == "error"
    assert source_gate["unsupported_sources"] == ["qwen_caption_io_latest"]
    assert source_gate["accepted_observed_sources"] == []


def test_caption_dataset_job_pilot_certification_rejects_target_output_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api
    from tools import certify_qwen_caption_soak as certify
    from tools import preflight_qwen_caption_soak as preflight

    output_dir = tmp_path / "caption_artifacts"

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start after pilot output-dir collision")

    def fail_certify(*_args, **_kwargs):
        raise AssertionError("pilot certification should not run against the target output dir")

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(preflight, "preflight_soak", lambda _args: {"status": "ok", "checks": []})
    monkeypatch.setattr(certify, "certify_soak", fail_certify)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_pilot_collision", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        output_dir=str(output_dir),
        set_and_forget=True,
        require_pilot_certification=True,
        pilot_output_dir=str(output_dir),
        pilot_require_prompt_budget_data=False,
        pilot_max_p95_duration_hours=72,
        pilot_max_prompt_tokens=8400,
        pilot_max_prompt_budget_adapted_case_rate=0.2,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "failed"
    assert job.error == "caption_runner_pilot_certification_failed"
    assert "must be separate" in job.message
    report = job.result["required_pilot_certification"]
    assert report["status"] == "error"
    assert report["checks"][0]["name"] == "pilot_output_dir_isolated"
    assert report["checks"][0]["pilot_output_dir"] == str(output_dir.resolve(strict=False))
    assert report["checks"][0]["target_output_dir"] == str(output_dir.resolve(strict=False))


def test_caption_dataset_job_pilot_certification_uses_requested_fingerprint(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api
    from tools import certify_qwen_caption_soak as certify
    from tools import preflight_qwen_caption_soak as preflight

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("runner subprocess should not start after pilot fingerprint mismatch")

    captured: dict[str, object] = {}

    def fake_certify(pilot_output_dir, **kwargs):
        captured["pilot_output_dir"] = pilot_output_dir
        captured["kwargs"] = kwargs
        return {
            "status": "error",
            "checks": [
                {
                    "name": "run_settings_fingerprint",
                    "status": "error",
                    "detail": "pilot run settings do not match the requested launch settings",
                    "expected_fingerprint": kwargs.get("expected_run_settings_fingerprint"),
                    "pilot_fingerprint": "pilot",
                }
            ],
        }

    monkeypatch.setattr(api.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(preflight, "preflight_soak", lambda _args: {"status": "ok", "checks": []})
    monkeypatch.setattr(certify, "certify_soak", fake_certify)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_pilot_fingerprint", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence.", "model_id": "caption-model"},
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        require_pilot_certification=True,
        pilot_output_dir=str(tmp_path / "pilot"),
        pilot_require_prompt_budget_data=False,
        pilot_max_p95_duration_hours=72,
        pilot_max_prompt_tokens=8400,
        pilot_max_prompt_budget_adapted_case_rate=0.2,
        pilot_deterministic_recovery_confidence=0.8,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    expected_fingerprint = job.result["requested_run_settings"]["fingerprint"]
    certify_kwargs = captured["kwargs"]
    assert certify_kwargs["expected_run_settings_fingerprint"] == expected_fingerprint
    assert certify_kwargs["max_p95_duration_hours"] == 72
    assert certify_kwargs["require_prompt_budget_data"] is False
    assert certify_kwargs["max_prompt_tokens"] == 8400
    assert certify_kwargs["max_prompt_budget_adapted_case_rate"] == 0.2
    assert certify_kwargs["deterministic_recovery_confidence"] == 0.8
    assert expected_fingerprint
    assert job.status == "failed"
    assert job.error == "caption_runner_pilot_certification_failed"
    assert job.result["required_pilot_certification"]["checks"][0]["expected_fingerprint"] == expected_fingerprint


def test_caption_dataset_job_watchdog_fails_stalled_runner(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    class BlockingStdout:
        def __init__(self, proc):
            self.proc = proc

        def __iter__(self):
            return self

        def __next__(self):
            while not self.proc.terminated:
                time.sleep(0.01)
            raise StopIteration

    class StalledProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False
            self.stdout = BlockingStdout(self)

        def poll(self):
            return -15 if self.terminated or self.killed else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            return self.poll()

    stalled = StalledProcess()
    captured = {}

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return stalled

    monkeypatch.setattr(api.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_stalled", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        runner_no_output_timeout_seconds=0.1,
        runner_heartbeat_interval_seconds=0.05,
        runner_artifact_log_bytes=2048,
        allow_model_download=True,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert "--heartbeat-interval" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--heartbeat-interval") + 1] == "0.05"
    assert "--max-artifact-log-bytes" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--max-artifact-log-bytes") + 1] == "2048"
    assert stalled.terminated
    assert job.status == "failed"
    assert job.error == "caption_runner_no_output_timeout"
    assert job.result["watchdog_timeout_seconds"] == 0.1
    assert job.result["processed"] == 0
    assert job.result["heartbeat_json"].endswith("heartbeat.json")


def test_caption_dataset_job_watchdog_ignores_chattery_runner_stdout(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    class ChatteryStdout:
        def __init__(self, proc):
            self.proc = proc

        def __iter__(self):
            return self

        def __next__(self):
            if self.proc.terminated or self.proc.finished:
                raise StopIteration
            if time.monotonic() - self.proc.started_at > 0.4:
                self.proc.finished = True
                raise StopIteration
            time.sleep(0.01)
            return "runner is still printing but not reporting progress\n"

    class ChatteryProcess:
        def __init__(self):
            self.started_at = time.monotonic()
            self.terminated = False
            self.killed = False
            self.finished = False
            self.stdout = ChatteryStdout(self)

        def poll(self):
            if self.terminated or self.killed:
                return -15
            return 0 if self.finished else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            return self.poll()

    process = ChatteryProcess()
    monkeypatch.setattr(api.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )

    job = api.QwenCaptionDatasetJob(job_id="qcap_chatty_no_progress", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        runner_no_output_timeout_seconds=0.1,
        runner_heartbeat_interval_seconds=0.05,
        allow_model_download=True,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert process.terminated
    assert job.status == "failed"
    assert job.error == "caption_runner_no_output_timeout"
    assert "no heartbeat or result progress" in job.message
    assert job.result["processed"] == 0


def test_caption_dataset_job_runner_uses_preflighted_effective_model(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api
    from tools import preflight_qwen_caption_soak as preflight

    class EmptyStdout:
        def __iter__(self):
            return iter(())

    class CompletedProcess:
        stdout = EmptyStdout()

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    captured = {}

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return CompletedProcess()

    def fake_preflight(args):
        captured["preflight_args"] = args
        return {"status": "ok", "checks": []}

    monkeypatch.setattr(api.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(preflight, "preflight_soak", fake_preflight)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )
    job = api.QwenCaptionDatasetJob(job_id="qcap_effective_model", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={
            "user_prompt": "Write one sentence.",
            "model_id": "auto",
            "model_variant": "Thinking",
        },
        attempts=1,
        per_image_timeout_seconds=30,
        allow_model_download=True,
        save_text_labels=True,
        set_and_forget=True,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    expected_model = api._resolve_qwen_variant_model_id_impl(
        api.QWEN_MLX_CAPTION_MODEL_NAME,
        "Thinking",
    )
    cmd = captured["cmd"]
    assert cmd[cmd.index("--model-id") + 1] == expected_model
    assert cmd[cmd.index("--refinement-model-id") + 1] == "same"
    assert cmd[cmd.index("--fallback-model-id") + 1] == "auto"
    assert cmd[cmd.index("--loop-recovery") + 1] == "safe_retry_fallback"
    assert cmd[cmd.index("--attempts") + 1] == "1"
    assert cmd[cmd.index("--mlx-max-image-side") + 1] == "224"
    assert cmd[cmd.index("--retry-image-side-scale") + 1] == "0.75"
    assert cmd[cmd.index("--min-retry-image-side") + 1] == "192"
    assert cmd[cmd.index("--cooldown-after-success") + 1] == "5.0"
    assert "--save-dataset-text-labels" in cmd
    preflight_args = captured["preflight_args"]
    assert preflight_args.model_id == expected_model
    assert preflight_args.save_dataset_text_labels is True
    assert preflight_args.mlx_max_image_side == 224
    assert preflight_args.retry_image_side_scale == 0.75
    assert preflight_args.min_retry_image_side == 192
    assert preflight_args.cooldown_after_success == 5.0
    assert preflight_args.request_json == Path(job.result["output_dir"]) / "request_fields.json"
    request_fields = json.loads(preflight_args.request_json.read_text())
    assert request_fields["model_id"] == expected_model
    assert request_fields["model_variant"] == "Thinking"
    assert job.request["caption_request"]["model_id"] == expected_model
    assert job.result["effective_model_fields"]["model_id"] == expected_model
    assert job.request["max_signal_exit_attempt_row_rate"] == 0.05
    assert job.result["runner_profile"]["mlx_max_image_side"] == 224
    assert job.result["runner_profile"]["min_retry_image_side"] == 192
    assert job.result["runner_profile"]["cooldown_after_success"] == 5.0


def test_caption_dataset_runner_profile_preserves_explicit_success_cooldown() -> None:
    import localinferenceapi as api

    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={},
        set_and_forget=True,
        cooldown_after_success_seconds=0,
    )
    assert api._qwen_caption_dataset_runner_profile(payload)["cooldown_after_success"] == 0.0

    request_payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"runner_cooldown_after_success_seconds": "1.5"},
        set_and_forget=True,
    )
    assert api._qwen_caption_dataset_runner_profile(request_payload)["cooldown_after_success"] == 1.5


def test_caption_dataset_job_completes_when_runner_finished_with_failed_cases(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    class ScriptedStdout:
        def __init__(self, rows):
            self._lines = [json.dumps(row) + "\n" for row in rows]

        def __iter__(self):
            return iter(self._lines)

    class CompletedProcess:
        def __init__(self, rows):
            self.stdout = ScriptedStdout(rows)

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    rows = [
        {
            "case_id": "image:a.jpg:full",
            "case": "image_000001",
            "stem": "a",
            "exit_code": 1,
            "status": "exception",
            "final_status": "failed",
            "quality_failures": [],
            "artifact_dir": None,
        },
        {
            "case_id": "image:b.jpg:full",
            "case": "image_000002",
            "stem": "b",
            "exit_code": 0,
            "status": "ok",
            "final_status": "ok",
            "quality_failures": [],
            "artifact_dir": None,
        },
    ]
    monkeypatch.setattr(api.subprocess, "Popen", lambda *_args, **_kwargs: CompletedProcess(rows))
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:a.jpg:full",
                    "name": "image_000001",
                    "stem": "a",
                    "image_path": str(tmp_path / "a.jpg"),
                    "label_path": str(tmp_path / "a.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                },
                {
                    "case_id": "image:b.jpg:full",
                    "name": "image_000002",
                    "stem": "b",
                    "image_path": str(tmp_path / "b.jpg"),
                    "label_path": str(tmp_path / "b.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                },
            ],
        ),
    )
    job = api.QwenCaptionDatasetJob(job_id="qcap_done_with_failures", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        max_failures=0,
        allow_model_download=True,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "completed"
    assert job.error is None
    assert job.result["processed"] == 2
    assert job.result["failed"] == 1
    assert job.message == "Caption dataset job complete with 1 failed"


def test_caption_dataset_job_recovers_caption_from_skipped_completed_row(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api
    from tools import preflight_qwen_caption_soak as preflight

    class ScriptedStdout:
        def __init__(self, rows):
            self._lines = [json.dumps(row) + "\n" for row in rows]

        def __iter__(self):
            return iter(self._lines)

    class CompletedProcess:
        def __init__(self, rows):
            self.stdout = ScriptedStdout(rows)

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    attempt_dir = tmp_path / "artifacts" / "case" / "attempt_01"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "result.json").write_text(
        json.dumps({
            "case": {
                "case_id": "image:a.jpg:full",
                "name": "image_000001",
                "stem": "a",
                "image_name": "a.jpg",
                "image_path": str(tmp_path / "a.jpg"),
                "split": "train",
            },
            "response": {
                "caption": "The scene contains 1 Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 1,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {"missing_label_mentions": []},
        })
    )
    rows = [
        {
            "case_id": "image:a.jpg:full",
            "case": "image_000001",
            "stem": "a",
            "exit_code": 0,
            "status": "ok",
            "final_status": "skipped_completed",
            "quality_failures": [],
            "recovery_events": [{"action": "text_recovery_succeeded"}],
            "preview_prompt_budget": {"adapted_sections": 1, "max_prompt_tokens": 7000},
            "artifact_dir": str(attempt_dir),
            "resumed_skip": True,
        }
    ]
    saved_labels = []

    monkeypatch.setattr(api.subprocess, "Popen", lambda *_args, **_kwargs: CompletedProcess(rows))
    monkeypatch.setattr(preflight, "preflight_soak", lambda _args: {"status": "ok", "checks": []})
    monkeypatch.setattr(api, "set_text_label", lambda *args, **kwargs: saved_labels.append((args, kwargs)))
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:a.jpg:full",
                    "name": "image_000001",
                    "stem": "a",
                    "image_path": str(tmp_path / "a.jpg"),
                    "label_path": str(tmp_path / "a.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )
    job = api.QwenCaptionDatasetJob(job_id="qcap_recovered_caption", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        allow_model_download=True,
        save_text_labels=True,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "completed"
    assert job.result["runner_totals"] == {"skipped_completed": 1}
    assert job.result["degraded_rates"]["processed_cases"] == 1
    assert job.result["degraded_rates"]["recovery_event_cases"] == 1
    assert job.result["degraded_rates"]["recovery_events"] == 1
    assert job.result["degraded_rates"]["prompt_budget_adapted_cases"] == 1
    assert job.result["degraded_rates"]["max_prompt_tokens"] == 7000
    assert job.result["latest_caption"]["caption"] == "The scene contains 1 Building."
    assert job.result["captions"][0]["image_name"] == "a.jpg"
    assert job.result["saved_text_labels"] == 1
    assert saved_labels == [
        (
            ("ds", "a.jpg", {"caption": "The scene contains 1 Building.", "split": "train"}),
            {},
        )
    ]


def test_caption_dataset_set_and_forget_live_runner_failure_auto_resumes(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    class EmptyStdout:
        def __iter__(self):
            return iter(())

    class FailedProcess:
        stdout = EmptyStdout()

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

        def terminate(self):
            pass

        def kill(self):
            pass

    monkeypatch.setattr(api.subprocess, "Popen", lambda *_args, **_kwargs: FailedProcess())
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )
    captured = {}

    def fake_start(payload, *, force_default_output_dir=False):
        captured["payload"] = payload
        captured["force_default_output_dir"] = force_default_output_dir
        return api.QwenCaptionDatasetJob(job_id="qcap_auto", output_dir=str(tmp_path / "qcap_auto"))

    monkeypatch.setattr(api, "_start_qwen_caption_dataset_job", fake_start)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_MAX_AUTO_RESUMES", 25)
    job = api.QwenCaptionDatasetJob(job_id="qcap_live_retry", output_dir=str(tmp_path / "job"))
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {job.job_id: job})
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={
            "user_prompt": "Write one sentence.",
            "model_id": "auto",
            "model_variant": "Thinking",
        },
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        auto_resume_count=0,
        allow_model_download=True,
    )

    try:
        api._run_qwen_caption_dataset_job(job, payload)
    finally:
        api.QWEN_CAPTION_DATASET_JOBS.pop(job.job_id, None)

    assert job.status == "failed"
    assert job.error == "caption_runner_exit_1"
    assert job.message == "Auto-resumed as qcap_auto"
    assert job.result["auto_resumed_job_id"] == "qcap_auto"
    assert job.result["auto_resumed_from_error"] == "caption_runner_exit_1"
    expected_model = api._resolve_qwen_variant_model_id_impl(
        api.QWEN_MLX_CAPTION_MODEL_NAME,
        "Thinking",
    )
    assert job.request["caption_request"]["model_id"] == expected_model
    resumed_payload = captured["payload"]
    assert resumed_payload.resume is True
    assert resumed_payload.set_and_forget is True
    assert resumed_payload.allow_model_download is True
    assert resumed_payload.caption_request["model_id"] == expected_model
    assert resumed_payload.caption_request["model_variant"] == "Thinking"
    assert resumed_payload.auto_resume_count == 1
    assert resumed_payload.output_dir == str(tmp_path / "job")
    assert captured["force_default_output_dir"] is True
    assert any(log["message"] == "scheduled set-and-forget auto-resume" for log in job.logs)


def test_caption_dataset_set_and_forget_degraded_completion_auto_resumes_recovered_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api
    from tools import preflight_qwen_caption_soak as preflight

    row = {
        "case_id": "image:train/frame.jpg:full",
        "case": "image_000001",
        "stem": "frame",
        "caption_mode": "full",
        "exit_code": 0,
        "status": "ok",
        "final_status": "ok",
        "quality_failures": [],
        "recovery_events": [{"action": "loop_detected"}],
        "preview_prompt_budget": {"max_prompt_tokens": 1200},
    }

    class LinesStdout:
        def __iter__(self):
            return iter([json.dumps(row) + "\n"])

    class CompletedProcess:
        stdout = LinesStdout()

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    monkeypatch.setattr(api.subprocess, "Popen", lambda *_args, **_kwargs: CompletedProcess())
    monkeypatch.setattr(preflight, "preflight_soak", lambda *_args, **_kwargs: {"status": "ok", "checks": []})
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )
    captured = {}

    def fake_start(payload, *, force_default_output_dir=False):
        captured["payload"] = payload
        captured["force_default_output_dir"] = force_default_output_dir
        return api.QwenCaptionDatasetJob(job_id="qcap_auto_degraded", output_dir=str(tmp_path / "qcap_auto"))

    monkeypatch.setattr(api, "_start_qwen_caption_dataset_job", fake_start)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_MAX_AUTO_RESUMES", 25)
    job = api.QwenCaptionDatasetJob(job_id="qcap_degraded_retry", output_dir=str(tmp_path / "job"))
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {job.job_id: job})
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        auto_resume_count=0,
        allow_model_download=True,
        max_loop_recovery_case_rate=0.0,
    )

    try:
        api._run_qwen_caption_dataset_job(job, payload)
    finally:
        api.QWEN_CAPTION_DATASET_JOBS.pop(job.job_id, None)

    assert job.status == "failed"
    assert job.error == "caption_runner_degraded_rates"
    assert job.message == "Auto-resumed as qcap_auto_degraded"
    assert job.result["auto_resumed_job_id"] == "qcap_auto_degraded"
    assert job.result["auto_resumed_from_error"] == "caption_runner_degraded_rates"
    assert job.result["degraded_rates"]["loop_recovery_case_rate"] == 1.0
    violation_rates = {item["rate"] for item in job.result["degraded_rate_violations"]}
    assert "loop_recovery_case_rate" in violation_rates
    resumed_payload = captured["payload"]
    assert resumed_payload.resume is True
    assert resumed_payload.set_and_forget is True
    assert resumed_payload.resume_reprocess_recovery_events is True
    assert resumed_payload.auto_resume_count == 1
    assert resumed_payload.output_dir == str(tmp_path / "job")
    assert captured["force_default_output_dir"] is True


def test_caption_dataset_set_and_forget_strict_audit_failure_auto_resumes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api
    from tools import preflight_qwen_caption_soak as preflight

    row = {
        "case_id": "image:train/frame.jpg:full",
        "case": "image_000001",
        "stem": "frame",
        "caption_mode": "full",
        "exit_code": 0,
        "status": "ok",
        "final_status": "ok",
        "quality_failures": [],
        "recovery_events": [],
    }

    class LinesStdout:
        def __iter__(self):
            return iter([json.dumps(row) + "\n"])

    class CompletedProcess:
        stdout = LinesStdout()

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    monkeypatch.setattr(api.subprocess, "Popen", lambda *_args, **_kwargs: CompletedProcess())
    monkeypatch.setattr(preflight, "preflight_soak", lambda *_args, **_kwargs: {"status": "ok", "checks": []})
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_strict_audit",
        lambda *_args, **_kwargs: {
            "status": "error",
            "checks": [
                {
                    "name": "case_coverage",
                    "status": "error",
                    "detail": "1/2 cases have latest result rows",
                }
            ],
        },
    )
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_cases",
        lambda payload, output_dir: (
            {"dataset_label": "Dataset"},
            [
                {
                    "case_id": "image:train/frame.jpg:full",
                    "name": "image_000001",
                    "stem": "frame",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "label_path": str(tmp_path / "frame.txt"),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        ),
    )
    captured = {}

    def fake_start(payload, *, force_default_output_dir=False):
        captured["payload"] = payload
        captured["force_default_output_dir"] = force_default_output_dir
        return api.QwenCaptionDatasetJob(job_id="qcap_auto_audit", output_dir=str(tmp_path / "qcap_auto"))

    monkeypatch.setattr(api, "_start_qwen_caption_dataset_job", fake_start)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_MAX_AUTO_RESUMES", 25)
    job = api.QwenCaptionDatasetJob(job_id="qcap_strict_audit_retry", output_dir=str(tmp_path / "job"))
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {job.job_id: job})
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence."},
        attempts=1,
        per_image_timeout_seconds=30,
        set_and_forget=True,
        auto_resume_count=0,
        allow_model_download=True,
    )

    try:
        api._run_qwen_caption_dataset_job(job, payload)
    finally:
        api.QWEN_CAPTION_DATASET_JOBS.pop(job.job_id, None)

    assert job.status == "failed"
    assert job.error == "caption_runner_strict_audit_failed"
    assert job.message == "Auto-resumed as qcap_auto_audit"
    assert job.result["strict_audit"]["status"] == "error"
    assert job.result["auto_resumed_job_id"] == "qcap_auto_audit"
    assert job.result["auto_resumed_from_error"] == "caption_runner_strict_audit_failed"
    resumed_payload = captured["payload"]
    assert resumed_payload.resume is True
    assert resumed_payload.set_and_forget is True
    assert resumed_payload.auto_resume_count == 1
    assert resumed_payload.output_dir == str(tmp_path / "job")
    assert captured["force_default_output_dir"] is True


def test_caption_dataset_job_resume_reuses_persisted_request_and_output_dir(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    jobs_root = tmp_path / "jobs"
    old_output = jobs_root / "qcap_old"
    old_output.mkdir(parents=True)
    (old_output / "job.json").write_text(
        """
        {
          "job_id": "qcap_old",
          "status": "failed",
          "request": {
            "dataset_id": "ds",
            "caption_request": {"user_prompt": "Describe it"},
            "image_names": ["frame.jpg"],
            "resume": false,
            "allow_model_download": true,
            "save_text_labels": true,
            "generated_make_primary": true
          },
          "output_dir": "__OLD_OUTPUT__"
        }
        """.replace("__OLD_OUTPUT__", str(old_output))
    )
    captured = {}

    def fake_start(payload, *, force_default_output_dir=False):
        captured["payload"] = payload
        captured["force_default_output_dir"] = force_default_output_dir
        return api.QwenCaptionDatasetJob(job_id="qcap_new", output_dir=str(tmp_path / "jobs" / "qcap_new"))

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)
    monkeypatch.setattr(api, "_start_qwen_caption_dataset_job", fake_start)

    resumed = api._resume_qwen_caption_dataset_job("qcap_old")

    payload = captured["payload"]
    assert resumed.job_id == "qcap_new"
    assert payload.dataset_id == "ds"
    assert payload.caption_request == {"user_prompt": "Describe it"}
    assert payload.image_names == ["frame.jpg"]
    assert payload.resume is True
    assert payload.allow_model_download is True
    assert payload.save_text_labels is True
    assert payload.generated_make_primary is True
    assert payload.output_dir == str(old_output)
    assert captured["force_default_output_dir"] is True
    assert resumed.output_dir != str(old_output)


def test_caption_dataset_job_resume_rejects_live_running_job(monkeypatch) -> None:
    import localinferenceapi as api

    job = api.QwenCaptionDatasetJob(job_id="qcap_running", status="running")
    with api.QWEN_CAPTION_DATASET_JOBS_LOCK:
        api.QWEN_CAPTION_DATASET_JOBS[job.job_id] = job
    try:
        with pytest.raises(api.HTTPException) as exc_info:
            api._resume_qwen_caption_dataset_job(job.job_id)
    finally:
        with api.QWEN_CAPTION_DATASET_JOBS_LOCK:
            api.QWEN_CAPTION_DATASET_JOBS.pop(job.job_id, None)

    assert exc_info.value.status_code == api.HTTP_409_CONFLICT
    assert exc_info.value.detail == "qwen_caption_dataset_job_still_running"


def test_caption_dataset_job_resume_route_exists() -> None:
    import localinferenceapi as api

    routes = {
        (next(iter(getattr(route, "methods", {"GET"}))), getattr(route, "path", ""))
        for route in api.app.routes
    }

    assert ("POST", "/qwen/caption/jobs/{job_id}/resume") in routes


def test_caption_dataset_job_list_includes_persisted_interrupted_jobs(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    jobs_root = tmp_path / "jobs"
    job_dir = jobs_root / "qcap_old"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "qcap_old",
                "status": "running",
                "message": "Captioned 100/10000",
                "request": {"dataset_id": "ds", "caption_request": {"user_prompt": "Describe it"}},
                "result": {"dataset_id": "ds", "processed": 100, "total_cases": 10000},
                "output_dir": str(job_dir),
                "created_at": 10.0,
                "updated_at": 20.0,
                "logs": [],
            }
        )
    )
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)

    jobs = api.list_qwen_caption_dataset_jobs()

    by_id = {job["job_id"]: job for job in jobs}

    assert by_id["qcap_old"]["status"] == "interrupted"
    assert by_id["qcap_old"]["message"] == "Interrupted while backend was not tracking the job"
    assert by_id["qcap_old"]["request"]["dataset_id"] == "ds"


def test_caption_dataset_job_can_use_separate_metadata_and_artifact_dirs(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "train" / "images" / "frame.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color=(40, 60, 80)).save(image_path)
    (dataset_root / "labelmap.txt").write_text("Building\n")
    entry = {
        "id": "ds",
        "label": "Dataset",
        "yolo_layout": "split",
        "classes": ["Building"],
        "linked_root": str(dataset_root),
    }
    manifest = {
        "dataset_id": "ds",
        "dataset_label": "Dataset",
        "yolo_layout": "split",
        "labelmap": ["Building"],
        "images": [
            {
                "split": "train",
                "image_relpath": "frame.jpg",
                "image_name": "frame.jpg",
                "label_lines": ["0 0.5 0.5 0.5 0.5"],
                "text_label": "",
            }
        ],
    }
    artifact_dir = tmp_path / "artifacts"
    metadata_dir = tmp_path / "metadata"
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: dataset_root)
    monkeypatch.setattr(api, "_resolve_annotation_image_path", lambda *_args: image_path)
    job = api.QwenCaptionDatasetJob(job_id="qcap_separate", output_dir=str(metadata_dir))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence.", "max_new_tokens": 64},
        preview_only=True,
        attempts=1,
        per_image_timeout_seconds=120,
        output_dir=str(artifact_dir),
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "completed"
    assert Path(job.output_dir) == metadata_dir
    assert Path(job.result["job_output_dir"]) == metadata_dir
    assert Path(job.result["output_dir"]) == artifact_dir
    assert (metadata_dir / "job.json").exists()
    assert (artifact_dir / "summary.json").exists()


def test_caption_dataset_job_resume_preserves_source_job_metadata(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "train" / "images" / "frame.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color=(40, 60, 80)).save(image_path)
    (dataset_root / "labelmap.txt").write_text("Building\n")
    entry = {
        "id": "ds",
        "label": "Dataset",
        "yolo_layout": "split",
        "classes": ["Building"],
        "linked_root": str(dataset_root),
    }
    manifest = {
        "dataset_id": "ds",
        "dataset_label": "Dataset",
        "yolo_layout": "split",
        "labelmap": ["Building"],
        "images": [
            {
                "split": "train",
                "image_relpath": "frame.jpg",
                "image_name": "frame.jpg",
                "label_lines": ["0 0.5 0.5 0.5 0.5"],
                "text_label": "",
            }
        ],
    }
    artifact_dir = tmp_path / "artifacts"
    metadata_dir = tmp_path / "metadata"
    artifact_dir.mkdir(parents=True)
    old_job_json = json.dumps(
        {
            "job_id": "qcap_old",
            "status": "failed",
            "output_dir": str(artifact_dir),
        },
        indent=2,
        sort_keys=True,
    )
    (artifact_dir / "job.json").write_text(old_job_json)
    (artifact_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image:train/frame.jpg:full",
                "case": "image_000001",
                "stem": "frame",
                "caption_mode": "full",
                "exit_code": 0,
                "status": "preview_only",
                "final_status": "preview_only",
                "quality_failures": [],
                "artifact_dir": None,
            },
            sort_keys=True,
        )
        + "\n"
    )
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: dataset_root)
    monkeypatch.setattr(api, "_resolve_annotation_image_path", lambda *_args: image_path)
    job = api.QwenCaptionDatasetJob(job_id="qcap_resume_new", output_dir=str(metadata_dir))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence.", "max_new_tokens": 64},
        preview_only=True,
        resume=True,
        attempts=1,
        per_image_timeout_seconds=120,
        output_dir=str(artifact_dir),
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "completed"
    assert Path(job.output_dir) == metadata_dir
    assert Path(job.result["job_output_dir"]) == metadata_dir
    assert Path(job.result["output_dir"]) == artifact_dir
    assert job.result["processed"] == 1
    assert job.result["runner_totals"] == {"preview_only": 1}
    assert (metadata_dir / "job.json").exists()
    assert (artifact_dir / "job.json").read_text() == old_job_json


def test_caption_dataset_set_and_forget_auto_resume_selects_newest_artifact_owner(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    jobs_root = tmp_path / "jobs"
    artifact_dir = tmp_path / "artifacts"
    original_dir = jobs_root / "qcap_original"
    newer_dir = jobs_root / "qcap_resume"
    artifact_dir.mkdir(parents=True)
    original_dir.mkdir(parents=True)
    newer_dir.mkdir(parents=True)
    original_payload = {
        "job_id": "qcap_original",
        "status": "running",
        "message": "Captioned 10/100",
        "request": {
            "dataset_id": "ds",
            "caption_request": {"user_prompt": "Describe it"},
            "set_and_forget": True,
            "auto_resume_count": 0,
        },
        "result": {"dataset_id": "ds", "processed": 10, "total_cases": 100},
        "output_dir": str(artifact_dir),
        "created_at": 10.0,
        "updated_at": 20.0,
        "logs": [],
    }
    newer_payload = {
        "job_id": "qcap_resume",
        "status": "interrupted",
        "message": "Interrupted",
        "request": {
            "dataset_id": "ds",
            "caption_request": {"user_prompt": "Describe it"},
            "set_and_forget": True,
            "allow_model_download": True,
            "auto_resume_count": 1,
            "output_dir": str(artifact_dir),
        },
        "result": {"dataset_id": "ds", "processed": 25, "total_cases": 100},
        "output_dir": str(newer_dir),
        "created_at": 30.0,
        "updated_at": 40.0,
        "logs": [],
    }
    (original_dir / "job.json").write_text(json.dumps(original_payload))
    (newer_dir / "job.json").write_text(json.dumps(newer_payload))
    captured = {}

    def fake_start(payload, *, force_default_output_dir=False):
        captured["payload"] = payload
        captured["force_default_output_dir"] = force_default_output_dir
        return api.QwenCaptionDatasetJob(job_id="qcap_auto", output_dir=str(jobs_root / "qcap_auto"))

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {})
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_MAX_AUTO_RESUMES", 25)
    monkeypatch.setattr(api, "_start_qwen_caption_dataset_job", fake_start)

    resumed = api._auto_resume_qwen_caption_dataset_jobs()

    payload = captured["payload"]
    assert [job["job_id"] for job in resumed] == ["qcap_auto"]
    assert payload.dataset_id == "ds"
    assert payload.resume is True
    assert payload.set_and_forget is True
    assert payload.allow_model_download is True
    assert payload.auto_resume_count == 2
    assert payload.output_dir == str(artifact_dir)
    assert captured["force_default_output_dir"] is True


def test_caption_dataset_set_and_forget_auto_resume_reconciles_completed_external_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api

    jobs_root = tmp_path / "jobs"
    artifact_dir = tmp_path / "artifacts"
    metadata_dir = jobs_root / "qcap_interrupted"
    artifact_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    case = {
        "case_id": "image:train/frame.jpg:full",
        "name": "image_000001",
        "stem": "frame",
        "caption_mode": "full",
    }
    (artifact_dir / "manifest.json").write_text(json.dumps({"cases": [case]}))
    (artifact_dir / "results.jsonl").write_text(
        json.dumps({
            "case_id": "image:train/frame.jpg:full",
            "case": "image_000001",
            "stem": "frame",
            "status": "ok",
            "final_status": "ok",
            "quality_failures": [],
        })
        + "\n"
    )
    (artifact_dir / "summary.json").write_text(json.dumps({"total_cases": 1, "totals": {"ok": 1}}))
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps({"case_id": "image:train/frame.jpg:full", "caption": "A caption."})
        + "\n"
    )
    (artifact_dir / "heartbeat.json").write_text(
        json.dumps({
            "status": "completed",
            "phase": "finished",
            "heartbeat_epoch": time.time(),
        })
    )
    (metadata_dir / "job.json").write_text(
        json.dumps({
            "job_id": "qcap_interrupted",
            "status": "running",
            "message": "Captioned 1/1",
            "request": {
                "dataset_id": "ds",
                "caption_request": {"user_prompt": "Describe it"},
                "set_and_forget": True,
                "auto_resume_count": 0,
                "output_dir": str(artifact_dir),
            },
            "result": {"dataset_id": "ds", "processed": 1, "total_cases": 1},
            "output_dir": str(metadata_dir),
            "created_at": 10.0,
            "updated_at": 20.0,
            "logs": [],
        })
    )

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {})
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_MAX_AUTO_RESUMES", 25)
    monkeypatch.setattr(
        api,
        "_start_qwen_caption_dataset_job",
        lambda *_args, **_kwargs: pytest.fail("completed external artifacts should not start a resume job"),
    )

    assert api._auto_resume_qwen_caption_dataset_jobs() == []

    completed_payload = json.loads((metadata_dir / "job.json").read_text())
    assert completed_payload["status"] == "completed"
    assert completed_payload["progress"] == 1.0
    assert completed_payload["error"] is None
    assert completed_payload["message"] == "Caption dataset job complete from existing artifacts"
    assert completed_payload["result"]["output_dir"] == str(artifact_dir.resolve(strict=False))
    assert completed_payload["result"]["strict_audit"]["status"] == "ok"
    assert completed_payload["result"]["processed"] == 1
    assert completed_payload["result"]["total_cases"] == 1
    assert completed_payload["logs"][-1]["message"] == "set-and-forget artifacts already completed; no auto-resume needed"


def test_caption_dataset_set_and_forget_auto_resume_suppressed_by_newer_completed_owner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api

    jobs_root = tmp_path / "jobs"
    artifact_dir = tmp_path / "artifacts"
    old_dir = jobs_root / "qcap_old_failed"
    completed_dir = jobs_root / "qcap_completed"
    artifact_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    completed_dir.mkdir(parents=True)
    base_request = {
        "dataset_id": "ds",
        "caption_request": {"user_prompt": "Describe it"},
        "set_and_forget": True,
        "output_dir": str(artifact_dir),
    }
    (old_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "qcap_old_failed",
                "status": "failed",
                "error": "caption_runner_exit_1",
                "request": {**base_request, "auto_resume_count": 0},
                "output_dir": str(old_dir),
                "updated_at": 20.0,
            }
        )
    )
    (completed_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "qcap_completed",
                "status": "completed",
                "request": {**base_request, "auto_resume_count": 1},
                "output_dir": str(completed_dir),
                "updated_at": 40.0,
            }
        )
    )

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {})
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_MAX_AUTO_RESUMES", 25)
    monkeypatch.setattr(
        api,
        "_start_qwen_caption_dataset_job",
        lambda *_args, **_kwargs: pytest.fail("older failed artifact owner should be suppressed"),
    )

    assert api._auto_resume_qwen_caption_dataset_jobs() == []


def test_caption_dataset_set_and_forget_auto_resume_skips_cancelled_and_limit(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    jobs_root = tmp_path / "jobs"
    cancelled_dir = jobs_root / "qcap_cancelled"
    exhausted_dir = jobs_root / "qcap_exhausted"
    cancelled_dir.mkdir(parents=True)
    exhausted_dir.mkdir(parents=True)
    base_request = {
        "dataset_id": "ds",
        "caption_request": {"user_prompt": "Describe it"},
        "set_and_forget": True,
    }
    (cancelled_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "qcap_cancelled",
                "status": "cancelled",
                "request": {**base_request, "auto_resume_count": 0},
                "output_dir": str(cancelled_dir),
                "updated_at": 20.0,
            }
        )
    )
    (exhausted_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "qcap_exhausted",
                "status": "failed",
                "request": {**base_request, "auto_resume_count": 2, "max_auto_resumes": 2},
                "output_dir": str(exhausted_dir),
                "updated_at": 30.0,
            }
        )
    )
    pilot_failed_dir = jobs_root / "qcap_pilot_failed"
    pilot_failed_dir.mkdir(parents=True)
    (pilot_failed_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "qcap_pilot_failed",
                "status": "failed",
                "error": "caption_runner_pilot_certification_failed",
                "request": {**base_request, "auto_resume_count": 0, "max_auto_resumes": 25},
                "output_dir": str(pilot_failed_dir),
                "updated_at": 40.0,
            }
        )
    )
    pilot_required_dir = jobs_root / "qcap_pilot_required"
    pilot_required_dir.mkdir(parents=True)
    (pilot_required_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "qcap_pilot_required",
                "status": "failed",
                "error": "caption_runner_pilot_required",
                "request": {**base_request, "auto_resume_count": 0, "max_auto_resumes": 25},
                "output_dir": str(pilot_required_dir),
                "updated_at": 50.0,
            }
        )
    )
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {})
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(
        api,
        "_start_qwen_caption_dataset_job",
        lambda *_args, **_kwargs: pytest.fail("auto-resume should not start"),
    )

    assert api._auto_resume_qwen_caption_dataset_jobs() == []


def test_caption_dataset_set_and_forget_auto_resume_skips_nonrecoverable_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api

    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir(parents=True)
    base_request = {
        "dataset_id": "ds",
        "caption_request": {"user_prompt": "Describe it"},
        "set_and_forget": True,
        "auto_resume_count": 0,
        "max_auto_resumes": 25,
    }
    failures = [
        (
            "qcap_preflight_failed",
            {
                "error": "caption_runner_preflight_failed",
                "result": {
                    "preflight": {
                        "status": "error",
                        "checks": [{"name": "model_cache", "status": "error"}],
                    }
                },
            },
        ),
        (
            "qcap_generic_failed",
            {
                "error": "unexpected configuration error",
                "result": {},
            },
        ),
        (
            "qcap_corrupt_results",
            {
                "error": "caption_runner_strict_audit_failed",
                "result": {
                    "strict_audit": {
                        "status": "error",
                        "checks": [
                            {
                                "name": "results_jsonl",
                                "status": "error",
                                "invalid_rows": [{"line": 8, "preview": "{bad"}],
                            }
                        ],
                    }
                },
            },
        ),
        (
            "qcap_corrupt_captions",
            {
                "error": "caption_runner_strict_audit_failed",
                "result": {
                    "strict_audit": {
                        "status": "error",
                        "checks": [
                            {
                                "name": "captions_jsonl",
                                "status": "error",
                                "invalid_rows": [{"line": 3, "preview": "[bad]"}],
                            }
                        ],
                    }
                },
            },
        ),
    ]
    for index, (job_id, extra) in enumerate(failures, start=1):
        job_dir = jobs_root / job_id
        job_dir.mkdir()
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "status": "failed",
                    "request": {**base_request, "output_dir": str(tmp_path / f"artifacts_{index}")},
                    "output_dir": str(job_dir),
                    "updated_at": float(20 + index),
                    **extra,
                }
            )
        )

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {})
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(
        api,
        "_start_qwen_caption_dataset_job",
        lambda *_args, **_kwargs: pytest.fail("nonrecoverable failure should not auto-resume"),
    )

    assert api._auto_resume_qwen_caption_dataset_jobs() == []


def test_caption_dataset_set_and_forget_auto_resume_skips_live_artifact_owner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api

    jobs_root = tmp_path / "jobs"
    artifact_dir = tmp_path / "artifacts"
    persisted_dir = jobs_root / "qcap_failed"
    artifact_dir.mkdir(parents=True)
    persisted_dir.mkdir(parents=True)
    persisted_payload = {
        "job_id": "qcap_failed",
        "status": "failed",
        "error": "caption_runner_exit_1",
        "request": {
            "dataset_id": "ds",
            "caption_request": {"user_prompt": "Describe it"},
            "set_and_forget": True,
            "auto_resume_count": 0,
            "output_dir": str(artifact_dir),
        },
        "output_dir": str(persisted_dir),
        "updated_at": 20.0,
    }
    (persisted_dir / "job.json").write_text(json.dumps(persisted_payload))
    live_job = api.QwenCaptionDatasetJob(job_id="qcap_live_running", output_dir=str(artifact_dir))
    live_job.status = "running"
    live_job.request = dict(persisted_payload["request"])

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {live_job.job_id: live_job})
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_MAX_AUTO_RESUMES", 25)
    monkeypatch.setattr(
        api,
        "_start_qwen_caption_dataset_job",
        lambda *_args, **_kwargs: pytest.fail("auto-resume should not duplicate a live artifact owner"),
    )

    assert api._auto_resume_qwen_caption_dataset_jobs() == []


def test_caption_dataset_set_and_forget_auto_resume_ignores_failed_live_record(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api

    jobs_root = tmp_path / "jobs"
    artifact_dir = tmp_path / "artifacts"
    persisted_dir = jobs_root / "qcap_failed"
    artifact_dir.mkdir(parents=True)
    persisted_dir.mkdir(parents=True)
    persisted_payload = {
        "job_id": "qcap_failed",
        "status": "failed",
        "error": "caption_runner_exit_1",
        "request": {
            "dataset_id": "ds",
            "caption_request": {"user_prompt": "Describe it"},
            "set_and_forget": True,
            "allow_model_download": True,
            "auto_resume_count": 0,
            "output_dir": str(artifact_dir),
        },
        "output_dir": str(persisted_dir),
        "updated_at": 20.0,
    }
    (persisted_dir / "job.json").write_text(json.dumps(persisted_payload))
    live_failed = api.QwenCaptionDatasetJob(job_id="qcap_live_failed", output_dir=str(artifact_dir))
    live_failed.status = "failed"
    live_failed.error = "caption_runner_exit_1"
    live_failed.request = dict(persisted_payload["request"])
    captured = {}

    def fake_start(payload, *, force_default_output_dir=False):
        captured["payload"] = payload
        captured["force_default_output_dir"] = force_default_output_dir
        return api.QwenCaptionDatasetJob(job_id="qcap_auto_retry", output_dir=str(jobs_root / "qcap_auto_retry"))

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {live_failed.job_id: live_failed})
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_MAX_AUTO_RESUMES", 25)
    monkeypatch.setattr(api, "_start_qwen_caption_dataset_job", fake_start)

    resumed = api._auto_resume_qwen_caption_dataset_jobs()

    payload = captured["payload"]
    assert [job["job_id"] for job in resumed] == ["qcap_auto_retry"]
    assert payload.dataset_id == "ds"
    assert payload.resume is True
    assert payload.set_and_forget is True
    assert payload.allow_model_download is True
    assert payload.auto_resume_count == 1
    assert payload.output_dir == str(artifact_dir)
    assert captured["force_default_output_dir"] is True


def test_caption_dataset_set_and_forget_auto_resume_adopts_live_disk_runner_lock(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api

    jobs_root = tmp_path / "jobs"
    artifact_dir = tmp_path / "artifacts"
    persisted_dir = jobs_root / "qcap_interrupted"
    artifact_dir.mkdir(parents=True)
    persisted_dir.mkdir(parents=True)
    (artifact_dir / ".runner.lock").write_text(
        json.dumps(
            {
                "runner_id": "external-runner",
                "pid": os.getpid(),
                "heartbeat_epoch": time.time(),
            }
        )
    )
    (persisted_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "qcap_interrupted",
                "status": "interrupted",
                "request": {
                    "dataset_id": "ds",
                    "caption_request": {"user_prompt": "Describe it"},
                    "set_and_forget": True,
                    "auto_resume_count": 0,
                    "output_dir": str(artifact_dir),
                },
                "output_dir": str(persisted_dir),
                "updated_at": 20.0,
            }
        )
    )

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", jobs_root)
    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOBS", {})
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_MAX_AUTO_RESUMES", 25)
    registered = {}

    def fake_register_job_and_start_thread(*, job, registry, lock, target, args, name=None):
        registered["job"] = job
        registered["target"] = target
        registered["args"] = args
        registered["name"] = name
        with lock:
            registry[job.job_id] = job

    monkeypatch.setattr(api, "_register_job_and_start_thread", fake_register_job_and_start_thread)
    monkeypatch.setattr(
        api,
        "_start_qwen_caption_dataset_job",
        lambda *_args, **_kwargs: pytest.fail("auto-resume should not overtake an adopted live disk runner"),
    )

    adopted = api._auto_resume_qwen_caption_dataset_jobs()

    assert [job["job_id"] for job in adopted] == ["qcap_interrupted"]
    assert adopted[0]["status"] == "queued"
    assert adopted[0]["message"] == "Adopting live set-and-forget caption runner"
    assert registered["job"].job_id == "qcap_interrupted"
    assert registered["target"] is api._monitor_adopted_qwen_caption_dataset_runner
    assert registered["args"][2] == str(artifact_dir.resolve(strict=False))
    assert registered["name"] == "qwen-caption-adopt-qcap_interrupted"
    assert api.QWEN_CAPTION_DATASET_JOBS["qcap_interrupted"] is registered["job"]


def test_caption_dataset_adopted_live_runner_monitor_reconciles_completed_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import localinferenceapi as api

    artifact_dir = tmp_path / "artifacts"
    metadata_dir = tmp_path / "jobs" / "qcap_adopted"
    artifact_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    case = {
        "case_id": "image:train/frame.jpg:full",
        "name": "image_000001",
        "stem": "frame",
        "caption_mode": "full",
    }
    (artifact_dir / "manifest.json").write_text(json.dumps({"cases": [case]}))
    (artifact_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image:train/frame.jpg:full",
                "case": "image_000001",
                "stem": "frame",
                "status": "ok",
                "final_status": "ok",
                "quality_failures": [],
            }
        )
        + "\n"
    )
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps({"case_id": "image:train/frame.jpg:full", "caption": "A caption."})
        + "\n"
    )
    (artifact_dir / "summary.json").write_text(json.dumps({"total_cases": 1, "totals": {"ok": 1}}))
    (artifact_dir / "heartbeat.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "phase": "finished",
                "heartbeat_epoch": time.time(),
                "processed": 1,
                "total_cases": 1,
            }
        )
    )
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_ADOPTION_POLL_SECONDS", 0.5)
    monkeypatch.setattr(api, "_resume_qwen_caption_dataset_job", lambda *_args, **_kwargs: pytest.fail("completed adopted runner should not resume"))
    job = api.QwenCaptionDatasetJob(job_id="qcap_adopted", output_dir=str(metadata_dir))
    job.request = {
        "dataset_id": "ds",
        "caption_request": {"user_prompt": "Describe it"},
        "set_and_forget": True,
        "output_dir": str(artifact_dir),
    }

    api._monitor_adopted_qwen_caption_dataset_runner(
        job,
        job.request,
        str(artifact_dir.resolve(strict=False)),
    )

    assert job.status == "completed"
    assert job.progress == 1.0
    assert job.error is None
    assert job.message == "Caption dataset job complete from adopted runner"
    assert job.result["adopted_live_runner"] is True
    assert job.result["strict_audit"]["status"] == "ok"
    assert job.result["processed"] == 1
    assert job.result["total_cases"] == 1
    persisted = json.loads((metadata_dir / "job.json").read_text())
    assert persisted["status"] == "completed"
    assert persisted["result"]["output_dir"] == str(artifact_dir.resolve(strict=False))


def test_caption_dataset_audits_require_saved_text_labels_when_requested(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api
    from tools import audit_qwen_caption_soak as caption_audit

    calls = []

    def fake_audit_soak(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return {"status": "ok", "checks": []}

    monkeypatch.setattr(caption_audit, "audit_soak", fake_audit_soak)

    strict_report = api._qwen_caption_dataset_strict_audit(
        tmp_path,
        {"save_text_labels": "true", "set_and_forget": True},
        max_heartbeat_age_seconds=60,
    )
    live_report = api._qwen_caption_dataset_live_audit(
        tmp_path,
        {"save_text_labels": False, "set_and_forget": True},
        max_heartbeat_age_seconds=60,
    )

    assert strict_report["status"] == "ok"
    assert live_report["status"] == "ok"
    assert calls[0][1]["allow_running_incomplete"] is False
    assert calls[0][1]["set_and_forget"] is True
    assert calls[0][1]["require_saved_text_labels"] is True
    assert calls[1][1]["allow_running_incomplete"] is True
    assert calls[1][1]["set_and_forget"] is True
    assert calls[1][1]["require_saved_text_labels"] is False


def test_caption_dataset_set_and_forget_startup_hook_runs_once(monkeypatch) -> None:
    import localinferenceapi as api

    calls = []
    called = threading.Event()

    def fake_auto_resume():
        calls.append("called")
        called.set()
        return []

    assert api.startup_qwen_caption_set_and_forget_jobs in api.app.router.on_startup
    monkeypatch.setattr(api, "_auto_resume_qwen_caption_dataset_jobs", fake_auto_resume)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_SWEEP_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(api, "_QWEN_CAPTION_SET_AND_FORGET_STARTUP_DONE", False)
    monkeypatch.setattr(api, "_QWEN_CAPTION_SET_AND_FORGET_SWEEPER_STOP", threading.Event())

    api.startup_qwen_caption_set_and_forget_jobs()

    assert called.wait(timeout=2.0)
    api.startup_qwen_caption_set_and_forget_jobs()
    time.sleep(0.05)
    assert calls == ["called"]


def test_caption_dataset_set_and_forget_startup_hook_runs_periodic_sweeper(monkeypatch) -> None:
    import localinferenceapi as api

    calls = []
    second_call = threading.Event()
    stop_event = threading.Event()

    def fake_auto_resume():
        calls.append("called")
        if len(calls) >= 2:
            second_call.set()
            stop_event.set()
        return []

    monkeypatch.setattr(api, "_auto_resume_qwen_caption_dataset_jobs", fake_auto_resume)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME", True)
    monkeypatch.setattr(api, "QWEN_CAPTION_SET_AND_FORGET_SWEEP_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(api, "_QWEN_CAPTION_SET_AND_FORGET_STARTUP_DONE", False)
    monkeypatch.setattr(api, "_QWEN_CAPTION_SET_AND_FORGET_SWEEPER_STOP", stop_event)

    api.startup_qwen_caption_set_and_forget_jobs()

    assert second_call.wait(timeout=2.0)
    assert len(calls) >= 2


def test_caption_dataset_job_preview_subprocess_smoke(monkeypatch, tmp_path: Path) -> None:
    import localinferenceapi as api

    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "train" / "images" / "frame.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color=(40, 60, 80)).save(image_path)
    (dataset_root / "labelmap.txt").write_text("Building\n")
    entry = {
        "id": "ds",
        "label": "Dataset",
        "yolo_layout": "split",
        "classes": ["Building"],
        "linked_root": str(dataset_root),
    }
    manifest = {
        "dataset_id": "ds",
        "dataset_label": "Dataset",
        "yolo_layout": "split",
        "labelmap": ["Building"],
        "images": [
            {
                "split": "train",
                "image_relpath": "frame.jpg",
                "image_name": "frame.jpg",
                "label_lines": ["0 0.5 0.5 0.5 0.5"],
                "text_label": "",
            }
        ],
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: dataset_root)
    monkeypatch.setattr(api, "_resolve_annotation_image_path", lambda *_args: image_path)
    job = api.QwenCaptionDatasetJob(job_id="qcap_test", output_dir=str(tmp_path / "job"))
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="ds",
        caption_request={"user_prompt": "Write one sentence.", "max_new_tokens": 64},
        preview_only=True,
        attempts=1,
        per_image_timeout_seconds=120,
    )

    api._run_qwen_caption_dataset_job(job, payload)

    assert job.status == "completed"
    assert job.result["processed"] == 1
    assert Path(job.result["summary_json"]).exists()
