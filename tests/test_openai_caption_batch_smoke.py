from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.parse

from PIL import Image
import pytest

from tools import run_openai_caption_batch_smoke as batch_smoke
from models.schemas import QwenCaptionDatasetJobRequest


def test_batch_line_uses_responses_vision_file_and_glossary_terms(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image_path = images / "scene.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(image_path)
    label_path = labels / "scene.txt"
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (dataset / "labelmap.txt").write_text("RawThing\n", encoding="utf-8")
    request_json = tmp_path / "request.json"
    request_json.write_text(
        json.dumps({"labelmap_glossary": {"RawThing": ["canonical object"]}}),
        encoding="utf-8",
    )
    case = {
        "name": "sample_001",
        "stem": "scene",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "class_counts": {"RawThing": 1},
        "caption_mode": "full",
    }
    args = SimpleNamespace(
        request_json=request_json,
        max_output_tokens=3200,
        max_boxes=50,
        qa_count=8,
        model="gpt-5.5",
        reasoning_effort="high",
        image_detail="original",
        instruction_qa_imposed_questions=["What object is counted"],
        instruction_qa_restrict_speculative_language=True,
        include_source_annotations_in_generator_context=True,
        strict_grounding=False,
        qa_mix="object",
        answer_format="json",
    )

    line = batch_smoke.build_batch_line(
        case=case,
        file_id="file_vision_123",
        dataset_root=dataset,
        args=args,
    )

    assert line["method"] == "POST"
    assert line["url"] == "/v1/responses"
    body = line["body"]
    assert body["model"] == "gpt-5.5"
    assert body["reasoning"] == {"effort": "high"}
    assert body["text"]["format"]["type"] == "json_schema"
    content = body["input"][0]["content"]
    assert content[1] == {"type": "input_image", "file_id": "file_vision_123", "detail": "original"}
    prompt = content[0]["text"]
    assert "canonical object" in prompt
    assert '"canonical object": 1' in prompt
    assert '"RawThing": 1' not in prompt
    assert "Required user questions" in prompt
    assert "What object is counted?" in prompt
    assert "Use the computed object-heavy QA diversity plan" in prompt
    assert "QA diversity plan" in prompt
    assert "blue sedan, white van, red pickup, small tuk-tuk, or light vehicle" in prompt
    assert "JSON-encoded strings" in prompt
    assert "Restrict speculative language" in prompt
    assert "Never output raw labelmap spellings" in prompt
    assert "annotation coordinates" in prompt
    assert "Use image coordinates only when a required question explicitly asks for them" in prompt


def test_qa_category_plan_is_parametric_and_imposed_questions_consume_slots() -> None:
    case = {"case_id": "case_a", "image_name": "scene.jpg"}

    plan = batch_smoke.build_qa_category_plan(
        case=case,
        target_qa=8,
        imposed_questions=[],
        qa_mix="balanced",
    )

    assert sum(plan["category_counts"].values()) == 8
    assert plan["category_counts"]["count_presence"] == 2
    assert plan["category_counts"]["spatial_layout"] == 2
    assert plan["category_counts"]["relationship"] == 1
    assert plan["category_counts"]["appearance_attribute"] == 1
    assert plan["category_counts"]["scene_context"] == 1
    assert plan["category_counts"]["visibility_limit"] == 1

    masked = batch_smoke.build_qa_category_plan(
        case=case,
        target_qa=8,
        imposed_questions=["How many cars are visible?", "What color is the roof?"],
        qa_mix="balanced",
    )

    assert sum(masked["category_counts"].values()) == 6
    assert masked["category_counts"]["count_presence"] == 1
    assert masked["category_counts"]["appearance_attribute"] == 0
    assert masked["consumed_by_required_questions"] == [
        {"question": "How many cars are visible?", "category": "count_presence"},
        {"question": "What color is the roof?", "category": "appearance_attribute"},
    ]


def test_openai_caption_batch_args_preserve_qa_controls(tmp_path: Path) -> None:
    import localinferenceapi as api

    args = api._openai_caption_batch_args(
        QwenCaptionDatasetJobRequest(
            dataset_id="dataset_a",
            caption_provider="openai",
            openai_service_tier="batch",
            openai_reasoning_effort="low",
            instruction_dataset=True,
            include_generated_qa_in_training=True,
            subcaptions_per_image=8,
            target_generated_qa_per_image=8,
            instruction_qa_imposed_questions=["What is the main object"],
            instruction_qa_restrict_speculative_language=True,
            include_source_annotations_in_generator_context=False,
            strict_grounding=False,
            qa_mix="object",
            answer_format="json",
        ),
        request_path=tmp_path / "request.json",
    )

    assert args.qa_count == 8
    assert args.reasoning_effort == "low"
    assert args.instruction_qa_imposed_questions == ["What is the main object?"]
    assert args.instruction_qa_restrict_speculative_language is True
    assert args.include_source_annotations_in_generator_context is False
    assert args.strict_grounding is False
    assert args.qa_mix == "object"
    assert args.answer_format == "json"
    assert args.max_output_tokens == 10000


def test_imposed_questions_strip_output_format_instructions() -> None:
    payload = QwenCaptionDatasetJobRequest(
        dataset_id="dataset_a",
        caption_provider="openai",
        openai_service_tier="batch",
        instruction_dataset=True,
        include_generated_qa_in_training=True,
        subcaptions_per_image=8,
        target_generated_qa_per_image=8,
        instruction_qa_imposed_questions=[
            "What would be the safest place to land a drone in this image? Reply with a JSON object string containing x and y.",
            "Can you spot a safe area to drop a delivery in this image? Assuming a small parcel that is lowered slowly, where should we put it? Respong with X / Y coordinates and a description.",
        ],
    )

    assert payload.instruction_qa_imposed_questions == [
        "What would be the safest place to land a drone in this image?",
        "Can you spot a safe area to drop a delivery in this image? Assuming a small parcel that is lowered slowly, where should we put it?",
    ]
    assert batch_smoke.missing_required_questions(
        [{"question": "What would be the safest place to land a drone in this image?", "answer": "{}"}],
        [
            "What would be the safest place to land a drone in this image? Reply with a JSON object string containing x and y.",
        ],
    ) == []


def test_openai_caption_batch_args_strip_imposed_questions_when_generated_qa_disabled(
    tmp_path: Path,
) -> None:
    import localinferenceapi as api

    args = api._openai_caption_batch_args(
        QwenCaptionDatasetJobRequest(
            dataset_id="dataset_a",
            caption_provider="openai",
            openai_service_tier="batch",
            instruction_dataset=True,
            include_generated_qa_in_training=False,
            subcaptions_per_image=8,
            target_generated_qa_per_image=8,
            instruction_qa_imposed_questions=["What is the main object?"],
        ),
        request_path=tmp_path / "request.json",
    )

    assert args.qa_count == 0
    assert args.instruction_qa_imposed_questions == []


def test_batch_prompt_for_zero_qa_is_not_contradictory(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image_path = images / "scene.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(image_path)
    label_path = labels / "scene.txt"
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (dataset / "labelmap.txt").write_text("RawThing\n", encoding="utf-8")
    request_json = tmp_path / "request.json"
    request_json.write_text(
        json.dumps({"labelmap_glossary": {"RawThing": ["canonical object"]}}),
        encoding="utf-8",
    )
    case = {
        "name": "sample_001",
        "stem": "scene",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "class_counts": {"RawThing": 1},
        "caption_mode": "full",
    }
    args = SimpleNamespace(
        request_json=request_json,
        max_output_tokens=3200,
        max_boxes=50,
        qa_count=0,
        model="gpt-5.5",
        reasoning_effort="high",
        image_detail="original",
        instruction_qa_imposed_questions=[],
        instruction_qa_restrict_speculative_language=False,
        include_source_annotations_in_generator_context=True,
        strict_grounding=True,
        qa_mix="balanced",
        answer_format="natural",
    )

    line = batch_smoke.build_batch_line(
        case=case,
        file_id="file_vision_123",
        dataset_root=dataset,
        args=args,
    )

    schema = line["body"]["text"]["format"]["schema"]["properties"]["qa_pairs"]
    prompt = line["body"]["input"][0]["content"][0]["text"]
    assert schema["minItems"] == 0
    assert schema["maxItems"] == 0
    assert "Do not generate question-answer pairs" in prompt
    assert '"qa_pairs": []' in prompt
    assert "question text?" not in prompt
    assert "Required user questions" not in prompt


def test_batch_prompt_honors_per_case_generated_qa_deficit(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image_path = images / "scene.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(image_path)
    label_path = labels / "scene.txt"
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (dataset / "labelmap.txt").write_text("RawThing\n", encoding="utf-8")
    request_json = tmp_path / "request.json"
    request_json.write_text(
        json.dumps({"labelmap_glossary": {"RawThing": ["canonical object"]}}),
        encoding="utf-8",
    )
    case = {
        "name": "sample_001",
        "stem": "scene",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "class_counts": {"RawThing": 1},
        "caption_mode": "full",
        "_generated_qa_request_count": 3,
    }
    args = SimpleNamespace(
        request_json=request_json,
        max_output_tokens=3200,
        max_boxes=50,
        qa_count=8,
        model="gpt-5.5",
        reasoning_effort="high",
        image_detail="original",
        instruction_qa_imposed_questions=[],
        instruction_qa_restrict_speculative_language=False,
        include_source_annotations_in_generator_context=True,
        strict_grounding=True,
        qa_mix="balanced",
        answer_format="natural",
    )

    line = batch_smoke.build_batch_line(
        case=case,
        file_id="file_vision_123",
        dataset_root=dataset,
        args=args,
    )

    schema = line["body"]["text"]["format"]["schema"]["properties"]["qa_pairs"]
    prompt = line["body"]["input"][0]["content"][0]["text"]
    assert schema["minItems"] == 3
    assert schema["maxItems"] == 3
    assert "Generate exactly 3 question-answer pairs" in prompt


def test_openai_caption_dataset_cases_include_missing_imposed_questions_even_when_counts_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    image_path = images / "scene.jpg"
    Image.new("RGB", (16, 16), color=(20, 30, 40)).save(image_path)
    entry = {
        "id": "dataset_a",
        "label": "Dataset A",
        "dataset_root": str(dataset),
        "classes": ["RawThing"],
        "yolo_layout": "flat",
    }
    manifest = {
        "dataset_id": "dataset_a",
        "dataset_label": "Dataset A",
        "dataset_root": str(dataset),
        "labelmap": ["RawThing"],
        "yolo_layout": "flat",
        "images": [
            {
                "split": "train",
                "image_relpath": "scene.jpg",
                "image_name": "scene.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
                "text_label": "Existing base caption.",
            }
        ],
    }
    existing_qa = [
        {
            "id": f"qa_{index}",
            "image_name": "scene.jpg",
            "image_key": "scene.jpg",
            "question": f"What ordinary detail {index} is visible?",
            "answer": f"Detail {index} is visible.",
            "row_type": "generated_qa",
            "answer_source": "vlm_generated",
        }
        for index in range(8)
    ]
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: dataset)
    monkeypatch.setattr(api, "_annotation_effective_text_label", lambda *args, **kwargs: "Existing base caption.")
    monkeypatch.setattr(api, "_load_dataset_caption_records", lambda _entry: [])
    monkeypatch.setattr(api, "_load_dataset_caption_instruction_records", lambda _entry: existing_qa)

    _manifest, cases = api._qwen_caption_dataset_cases(
        QwenCaptionDatasetJobRequest(
            dataset_id="dataset_a",
            caption_provider="openai",
            openai_service_tier="batch",
            instruction_dataset=True,
            include_generated_qa_in_training=True,
            save_text_labels=True,
            write_policy="fill_missing",
            completion_mode="per_image_totals",
            target_base_captions_per_image=1,
            target_generated_qa_per_image=8,
            subcaptions_per_image=8,
            instruction_qa_imposed_questions=["What color is the roof?"],
        ),
        output_dir=tmp_path / "case_output",
    )

    assert len(cases) == 1
    assert cases[0]["completion_counts"]["base_caption_count"] == 1
    assert cases[0]["completion_counts"]["generated_qa_count"] == 8
    assert cases[0]["completion_counts"]["missing_imposed_questions"] == ["What color is the roof?"]
    assert cases[0]["_generated_qa_request_count"] == 1
    assert cases[0]["_openai_batch_missing_required_questions"] == ["What color is the roof?"]


def test_openai_batch_process_preview_uses_actual_combined_batch_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    image_path = images / "scene.png"
    Image.new("RGB", (48, 32), color=(30, 40, 50)).save(image_path)
    (dataset / "labelmap.txt").write_text("RawThing\n", encoding="utf-8")
    entry = {
        "id": "dataset_a",
        "dataset_id": "dataset_a",
        "dataset_root": str(dataset),
        "storage_mode": "managed",
        "yolo_layout": "flat",
    }
    manifest = {
        "dataset_id": "dataset_a",
        "dataset_label": "Dataset A",
        "yolo_layout": "flat",
        "labelmap": ["RawThing"],
        "images": [
            {
                "image_name": "scene.png",
                "image_relpath": "scene.png",
                "split": "train",
                "label_lines": ["0 0.5 0.5 0.25 0.25"],
            }
        ],
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_completion_snapshot",
        lambda *args, **kwargs: {
            "images": [
                {
                    "image_key": "train/scene.png",
                    "complete": False,
                    "base_caption_count": 0,
                    "generated_qa_count": 0,
                }
            ]
        },
    )

    preview = api._qwen_caption_dataset_process_preview(
        QwenCaptionDatasetJobRequest(
            dataset_id="dataset_a",
            caption_provider="openai",
            openai_service_tier="batch",
            openai_model="gpt-5.5",
            openai_image_detail="original",
            openai_reasoning_effort="high",
            instruction_dataset=True,
            include_generated_qa_in_training=True,
            subcaptions_per_image=8,
            target_generated_qa_per_image=8,
            caption_request={
                "request_text": "Describe it.",
                "labelmap_glossary": json.dumps({"RawThing": ["canonical object"]}),
            },
        )
    )

    section_titles = [section.title for section in preview.sections]
    assert section_titles == ["OpenAI Batch combined caption+QA prompt template"]
    assert "Return only one valid JSON object with keys caption and qa_pairs" in preview.full_text
    assert "Generated QA verifier prompt template" not in preview.full_text
    assert '"canonical object": 1' in preview.full_text
    assert '"RawThing": 1' not in preview.full_text
    assert preview.meta["instruction_qa_prompt"]["batch_combined_request"] is True
    assert preview.meta["instruction_qa_prompt"]["verifier"] == {
        "enabled": False,
        "reason": "openai_batch_combined_request",
    }


def test_openai_caption_provider_metadata_and_detail_validation() -> None:
    import localinferenceapi as api

    metadata = api._openai_caption_provider_metadata()
    models = {item["id"]: item for item in metadata["models"]}

    assert metadata["default_reasoning_effort"] == "high"
    assert metadata["pricing_last_verified"]
    assert models["gpt-5.5"]["supports_original_image_detail"] is True
    assert "original" in models["gpt-5.5"]["image_details"]
    assert models["gpt-4o"]["supports_original_image_detail"] is False
    assert "original" not in models["gpt-4o"]["image_details"]
    assert api._openai_caption_validate_model_detail("gpt-5.5", "original") == "original"
    assert api._openai_caption_validate_model_detail("gpt-4o", "high") == "high"
    with pytest.raises(api.HTTPException) as exc_info:
        api._openai_caption_validate_model_detail("gpt-4o", "original")
    assert exc_info.value.status_code == 400
    assert "openai_image_detail_unsupported:gpt-4o:original" in str(exc_info.value.detail)


def test_openai_batch_start_preflights_normalized_openai_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    import localinferenceapi as api

    seen: list[tuple[str, str, str]] = []

    def fake_preflight(payload):
        seen.append((payload.caption_provider, payload.openai_service_tier, payload.openai_image_detail))
        api._openai_caption_validate_model_detail(payload.openai_model, payload.openai_image_detail)

    monkeypatch.setattr(api, "_preflight_openai_caption_provider_request", fake_preflight)
    with pytest.raises(api.HTTPException) as exc_info:
        api._start_openai_caption_batch_job(
            QwenCaptionDatasetJobRequest(
                dataset_id="dataset_a",
                caption_provider="local_qwen",
                openai_model="gpt-4o",
                openai_image_detail="original",
                instruction_dataset=True,
            )
        )

    assert seen == [("openai", "batch", "original")]
    assert exc_info.value.status_code == 400
    assert "openai_image_detail_unsupported:gpt-4o:original" in str(exc_info.value.detail)


def test_openai_caption_batch_prompt_can_disable_source_annotation_context() -> None:
    prompt = batch_smoke.build_prompt(
        case={"name": "scene"},
        label_hints=[{"label": "canonical object", "bbox_2d": [1, 2, 3, 4]}],
        glossary_context={"canonical object": ["RawThing"]},
        class_counts={"canonical object": 2},
        target_qa=1,
        max_boxes=10,
        include_source_annotations=False,
    )

    assert "Ignore source annotation boxes" in prompt
    assert 'Authoritative object counts: {}' in prompt
    assert 'Representative annotation boxes: []' in prompt
    assert "RawThing" in prompt


def test_openai_caption_batch_collect_splits_incomplete_qa_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    case = {
        "name": "sample_001",
        "stem": "scene",
        "image_path": str(tmp_path / "scene.jpg"),
    }
    cid = batch_smoke.case_key(case)
    (output_dir / "batch_output.jsonl").write_text(
        json.dumps(
            {
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "resp_1",
                        "output_text": json.dumps(
                            {
                                "caption": "A grounded caption.",
                                "qa_pairs": [{"question": "What is present?", "answer": "A structure is present."}],
                            }
                        ),
                        "usage": {"input_tokens": 100, "output_tokens": 50},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = batch_smoke.collect_results(cases=[case], output_dir=output_dir, target_qa=2)

    assert summary["caption_rows"] == 0
    assert summary["incomplete_caption_rows"] == 1
    assert summary["accepted_cases"] == 0
    assert summary["incomplete_cases"] == 1


def test_openai_caption_batch_collect_marks_cap_hit_partial_json_for_full_retry(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    case = {
        "name": "sample_001",
        "stem": "scene",
        "image_path": str(tmp_path / "scene.jpg"),
    }
    cid = batch_smoke.case_key(case)
    (output_dir / "batch_output.jsonl").write_text(
        json.dumps(
            {
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "resp_1",
                        "output_text": '{"caption":"A grounded caption.","qa_pairs":[{"question":"What is present?","answer":"A structure',
                        "usage": {"input_tokens": 100, "output_tokens": 2500},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = batch_smoke.collect_results(
        cases=[case],
        output_dir=output_dir,
        target_qa=2,
        max_output_tokens=2500,
    )
    rows = [json.loads(line) for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["caption_rows"] == 0
    assert summary["incomplete_caption_rows"] == 0
    assert summary["failed_cases"] == 1
    assert rows[0]["failure_reason"] == "output_truncated"
    assert rows[0]["output_truncated"] is True
    assert batch_smoke.read_jsonl(output_dir / "captions.jsonl") == []
    assert batch_smoke.read_jsonl(output_dir / "incomplete_captions.jsonl") == []


def test_openai_caption_batch_collect_marks_missing_result_rows_failed(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    first = {
        "name": "sample_001",
        "stem": "first",
        "image_name": "first.jpg",
        "image_path": str(tmp_path / "first.jpg"),
    }
    second = {
        "name": "sample_002",
        "stem": "second",
        "image_name": "second.jpg",
        "image_path": str(tmp_path / "second.jpg"),
    }
    first_id = batch_smoke.case_key(first)
    second_id = batch_smoke.case_key(second)
    (output_dir / "batch_output.jsonl").write_text(
        json.dumps(
            {
                "custom_id": first_id,
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "resp_1",
                        "output_text": json.dumps({"caption": "A complete caption.", "qa_pairs": []}),
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = batch_smoke.collect_results(cases=[first, second], output_dir=output_dir, target_qa=0)

    assert summary["total_cases"] == 2
    assert summary["caption_rows"] == 1
    assert summary["failed_cases"] == 1
    assert summary["missing_result_rows"] == 1
    assert summary["missing_result_case_ids"] == [second_id]
    result_rows = batch_smoke.read_jsonl(output_dir / "results.jsonl")
    missing = [row for row in result_rows if row["case_id"] == second_id]
    assert missing == [
        {
            "case_id": second_id,
            "final_status": "failed",
            "failure_reason": "missing_batch_result",
            "generated_qa_pair_count": 0,
            "generated_qa_target_pair_count": 0,
            "image_name": "second.jpg",
        }
    ]


def test_openai_caption_batch_collect_is_local_idempotent_after_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_collected"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "summary.json").write_text(
        json.dumps({"total_cases": 1, "caption_rows": 1, "incomplete_caption_rows": 0}),
        encoding="utf-8",
    )
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image:scene.jpg:full",
                "image_name": "scene.jpg",
                "caption": "A previously collected caption.",
                "generated_qa_pairs": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_collected",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "openai_api_key_path": str(tmp_path / "missing_key"),
                "instruction_dataset": True,
                "subcaptions_per_image": 0,
                "caption_request": {"user_prompt": "Describe it."},
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "openai_batch_id": "batch_collected",
            "logs": [],
        }
    )

    result = api._openai_caption_batch_collect_job("ocap_collected")

    assert result["status"] == "collected"
    assert result["job_id"] == "ocap_collected"


def test_openai_caption_batch_collect_preserves_json_answer_format(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    case = {
        "name": "sample_001",
        "stem": "scene",
        "image_path": str(tmp_path / "scene.jpg"),
    }
    cid = batch_smoke.case_key(case)
    (output_dir / "batch_output.jsonl").write_text(
        json.dumps(
            {
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "resp_1",
                        "output_text": json.dumps(
                            {
                                "caption": "A grounded caption.",
                                "qa_pairs": [
                                    {
                                        "question": "What is present?",
                                        "answer": {"answer": "A structure is present."},
                                    }
                                ],
                            }
                        ),
                        "usage": {"input_tokens": 100, "output_tokens": 50},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = batch_smoke.collect_results(
        cases=[case],
        output_dir=output_dir,
        target_qa=1,
        answer_format="json",
    )

    assert summary["caption_rows"] == 1
    rows = batch_smoke.read_jsonl(output_dir / "captions.jsonl")
    assert rows[0]["answer_format"] == "json"
    assert rows[0]["generated_qa_pairs"][0]["answer"] == '{"answer":"A structure is present."}'
    assert batch_smoke.read_jsonl(output_dir / "results.jsonl")[0]["answer_format"] == "json"


def test_openai_caption_batch_collect_holds_rows_missing_imposed_questions(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    case = {
        "name": "sample_001",
        "stem": "scene",
        "image_path": str(tmp_path / "scene.jpg"),
    }
    cid = batch_smoke.case_key(case)
    (output_dir / "batch_output.jsonl").write_text(
        json.dumps(
            {
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "resp_1",
                        "output_text": json.dumps(
                            {
                                "caption": "A grounded caption.",
                                "qa_pairs": [
                                    {
                                        "question": "What is present?",
                                        "answer": "A structure is present.",
                                    },
                                    {
                                        "question": "Where is it?",
                                        "answer": "It is near the center.",
                                    },
                                ],
                            }
                        ),
                        "usage": {"input_tokens": 100, "output_tokens": 50},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = batch_smoke.collect_results(
        cases=[case],
        output_dir=output_dir,
        target_qa=2,
        imposed_questions=["What color is the roof?"],
    )

    assert summary["caption_rows"] == 0
    assert summary["incomplete_caption_rows"] == 1
    incomplete = batch_smoke.read_jsonl(output_dir / "incomplete_captions.jsonl")[0]
    result = batch_smoke.read_jsonl(output_dir / "results.jsonl")[0]
    assert incomplete["failure_reason"] == "required_qa_missing"
    assert incomplete["missing_required_questions"] == ["What color is the roof?"]
    assert result["missing_required_questions"] == ["What color is the roof?"]


def test_openai_caption_batch_collect_merges_visual_catchup_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    case = {
        "name": "sample_001",
        "stem": "scene",
        "image_path": str(tmp_path / "scene.jpg"),
        "_openai_batch_catchup_source_job_id": "ocap_source",
        "_openai_batch_existing_caption": "Existing caption.",
        "_openai_batch_existing_qa_pairs": [{"question": "What is already known?", "answer": "One object is present."}],
        "_openai_batch_existing_qa_count": 1,
        "_openai_batch_total_qa_target": 2,
        "_openai_batch_target_qa": 1,
    }
    cid = batch_smoke.case_key(case)
    (output_dir / "batch_output.jsonl").write_text(
        json.dumps(
            {
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "resp_1",
                        "output_text": json.dumps(
                            {
                                "caption": "Existing caption.",
                                "qa_pairs": [{"question": "Where is it located?", "answer": "It is near the center."}],
                            }
                        ),
                        "usage": {"input_tokens": 120, "output_tokens": 60},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = batch_smoke.collect_results(cases=[case], output_dir=output_dir, target_qa=8)

    assert summary["caption_rows"] == 1
    assert summary["incomplete_caption_rows"] == 0
    rows = batch_smoke.read_jsonl(output_dir / "captions.jsonl")
    assert rows[0]["generated_qa_pair_count"] == 2
    assert rows[0]["generated_qa_new_pair_count"] == 1
    assert rows[0]["openai_batch_catchup"]["source_job_id"] == "ocap_source"
    assert [pair["question"] for pair in rows[0]["generated_qa_pairs"]] == [
        "What is already known?",
        "Where is it located?",
    ]


def test_openai_caption_batch_collect_accepts_required_question_catchup_merge(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    case = {
        "name": "sample_001",
        "stem": "scene",
        "image_path": str(tmp_path / "scene.jpg"),
        "_openai_batch_catchup_source_job_id": "ocap_source",
        "_openai_batch_existing_caption": "Existing caption.",
        "_openai_batch_existing_qa_pairs": [
            {"question": "What type is visible?", "answer": "An object is visible."}
        ],
        "_openai_batch_existing_qa_count": 1,
        "_openai_batch_total_qa_target": 2,
        "_openai_batch_target_qa": 1,
        "_openai_batch_missing_required_questions": ["What color is the roof?"],
    }
    cid = batch_smoke.case_key(case)
    (output_dir / "batch_output.jsonl").write_text(
        json.dumps(
            {
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "resp_1",
                        "output_text": json.dumps(
                            {
                                "caption": "Existing caption.",
                                "qa_pairs": [
                                    {
                                        "question": "What color is the roof?",
                                        "answer": "The roof color cannot be determined from the image.",
                                    }
                                ],
                            }
                        ),
                        "usage": {"input_tokens": 120, "output_tokens": 60},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = batch_smoke.collect_results(
        cases=[case],
        output_dir=output_dir,
        target_qa=2,
        imposed_questions=["What type is visible?", "What color is the roof?"],
    )

    assert summary["caption_rows"] == 1
    assert summary["incomplete_caption_rows"] == 0
    row = batch_smoke.read_jsonl(output_dir / "captions.jsonl")[0]
    assert row["missing_required_questions"] == []
    assert [pair["question"] for pair in row["generated_qa_pairs"]] == [
        "What type is visible?",
        "What color is the roof?",
    ]


def test_openai_caption_batch_catchup_job_builds_visual_partial_row_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    class FakeThread:
        def __init__(self, target, args=(), name=None, daemon=None):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api.threading, "Thread", FakeThread)
    source_dir = tmp_path / "jobs" / "openai_batches" / "ocap_source"
    source_dir.mkdir(parents=True)
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (16, 16), color=(20, 30, 40)).save(image_path)
    label_path = tmp_path / "scene.txt"
    label_path.write_text("", encoding="utf-8")
    (tmp_path / "labelmap.txt").write_text("Object\n", encoding="utf-8")
    case = {
        "case_id": "image:scene.jpg:full",
        "name": "scene",
        "stem": "scene",
        "image_name": "scene.jpg",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "split": "train",
        "caption_mode": "full",
    }
    (source_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (source_dir / "incomplete_captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": case["case_id"],
                "image_name": "scene.jpg",
                "caption": "A partial paid caption.",
                "generated_qa_pairs": [{"question": "What is visible?", "answer": "One object is visible."}],
                "generated_qa_pair_count": 1,
                "generated_qa_target_pair_count": 3,
                "failure_reason": "generated_qa_incomplete",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_source",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "dataset_label": "Dataset A",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "openai_batch_shard_size": 100,
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "subcaptions_per_image": 3,
                "target_generated_qa_per_image": 3,
                "caption_request": {"request_text": "Describe it."},
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(source_dir),
            "logs": [],
        }
    )

    catchup = api._openai_caption_batch_catchup_job("ocap_source")

    assert catchup["kind"] == "openai_caption_batch"
    assert catchup["catchup_for_job_id"] == "ocap_source"
    assert catchup["case_count"] == 1
    catchup_dir = Path(catchup["output_dir"])
    catchup_cases = json.loads((catchup_dir / "cases.json").read_text(encoding="utf-8"))
    assert catchup_cases[0]["_openai_batch_existing_caption"] == "A partial paid caption."
    assert catchup_cases[0]["_openai_batch_existing_qa_count"] == 1
    assert catchup_cases[0]["_openai_batch_target_qa"] == 2
    batch_line = batch_smoke.build_batch_line(
        case=catchup_cases[0],
        file_id="file_visual_retry",
        dataset_root=tmp_path,
        args=SimpleNamespace(
            request_json=catchup_dir / "request_fields.json",
            max_output_tokens=3200,
            max_boxes=50,
            qa_count=3,
            model="gpt-5.5",
            reasoning_effort="high",
            image_detail="original",
        ),
    )
    content = batch_line["body"]["input"][0]["content"]
    assert any(part.get("type") == "input_image" and part.get("file_id") == "file_visual_retry" for part in content)
    prompt = next(part["text"] for part in content if part.get("type") == "input_text")
    assert "This is a catch-up request" in prompt
    assert "Generate exactly 2 additional" in prompt
    assert "Do not repeat any existing question" in prompt

    with pytest.raises(Exception) as exc_info:
        api._openai_caption_batch_catchup_job("ocap_source")
    assert getattr(exc_info.value, "status_code", None) == 409
    assert getattr(exc_info.value, "detail", "") == "openai_caption_batch_catchup_already_queued"


def test_openai_caption_batch_catchup_reprocesses_output_cap_truncated_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    class FakeThread:
        def __init__(self, target, args=(), name=None, daemon=None):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api.threading, "Thread", FakeThread)
    source_dir = tmp_path / "jobs" / "openai_batches" / "ocap_truncated_source"
    source_dir.mkdir(parents=True)
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (16, 16), color=(20, 30, 40)).save(image_path)
    case = {
        "case_id": "image:scene.jpg:full",
        "name": "scene",
        "stem": "scene",
        "image_name": "scene.jpg",
        "image_path": str(image_path),
        "split": "train",
        "caption_mode": "full",
    }
    (source_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (source_dir / "incomplete_captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": case["case_id"],
                "image_name": "scene.jpg",
                "caption": "A cut-off paid caption.",
                "generated_qa_pairs": [{"question": "What is visible?", "answer": "One object is visible."}],
                "generated_qa_pair_count": 1,
                "generated_qa_target_pair_count": 3,
                "failure_reason": "generated_qa_incomplete",
                "usage": {"output_tokens": 2500},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_truncated_source",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "dataset_label": "Dataset A",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "openai_batch_shard_size": 100,
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "subcaptions_per_image": 3,
                "target_generated_qa_per_image": 3,
                "caption_request": {"request_text": "Describe it.", "max_new_tokens": 2500},
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(source_dir),
            "logs": [],
        }
    )

    catchup = api._openai_caption_batch_catchup_job("ocap_truncated_source")

    catchup_dir = Path(catchup["output_dir"])
    catchup_cases = json.loads((catchup_dir / "cases.json").read_text(encoding="utf-8"))
    request_fields = json.loads((catchup_dir / "request_fields.json").read_text(encoding="utf-8"))
    assert "_openai_batch_existing_caption" not in catchup_cases[0]
    assert "_openai_batch_existing_qa_pairs" not in catchup_cases[0]
    assert catchup_cases[0]["_openai_batch_truncated_retry"] is True
    assert catchup_cases[0]["_openai_batch_target_qa"] == 3
    assert request_fields["max_new_tokens"] == 10000


def test_openai_caption_batch_detects_billing_hard_limit_error() -> None:
    import localinferenceapi as api

    assert api._openai_caption_batch_hit_billing_hard_limit(
        {
            "error": {
                "payload": {
                    "detail": {
                        "error": {
                            "message": "Billing hard limit has been reached",
                            "code": "billing_hard_limit_reached",
                        }
                    }
                }
            }
        }
    ) is True
    assert api._openai_caption_batch_hit_billing_hard_limit({"message": "ordinary failure"}) is False


def test_openai_caption_batch_submit_worker_cleans_orphan_uploads_on_billing_hard_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: {})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: str(tmp_path))
    monkeypatch.setattr(api, "_openai_caption_batch_api_key", lambda _path: "test-key")

    job_id = "ocap_hard_limit_cleanup"
    artifact_dir = tmp_path / "jobs" / "openai_batches" / job_id
    artifact_dir.mkdir(parents=True)
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(image_path)
    case = {
        "case_id": "image:scene.jpg:full",
        "image_name": "scene.jpg",
        "image_path": str(image_path),
        "caption_mode": "full",
    }
    (artifact_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (artifact_dir / "request_fields.json").write_text(json.dumps({"request_text": "Describe it."}), encoding="utf-8")
    api._openai_caption_batch_write(
        {
            "job_id": job_id,
            "kind": "openai_caption_batch",
            "status": "preparing",
            "dataset_id": "dataset_a",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "caption_request": {"request_text": "Describe it."},
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "request_json": str(artifact_dir / "request_fields.json"),
            "cases_json": str(artifact_dir / "cases.json"),
            "logs": [],
        }
    )
    cleanup_calls: list[dict[str, object]] = []

    def fake_upload_images(**_kwargs):
        return {case["case_id"]: {"case_id": case["case_id"], "file_id": "file_img_1"}}

    def fake_write_batch_input(**kwargs):
        path = Path(kwargs["output_dir"]) / "batch_input.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        return path

    def fake_submit_batch(**_kwargs):
        raise batch_smoke.OpenAIRequestError(
            operation="openai_http_error",
            status_code=400,
            detail=json.dumps({"error": {"code": "billing_hard_limit_reached", "message": "Billing hard limit has been reached"}}),
        )

    def fake_cleanup(**kwargs):
        cleanup_calls.append(dict(kwargs))
        return {"status": "ok", "deleted": 1, "file_count": 1}

    monkeypatch.setattr(batch_smoke, "upload_images", fake_upload_images)
    monkeypatch.setattr(batch_smoke, "write_batch_input", fake_write_batch_input)
    monkeypatch.setattr(batch_smoke, "submit_batch", fake_submit_batch)
    monkeypatch.setattr(batch_smoke, "cleanup_unsubmitted_uploaded_files", fake_cleanup)

    api._openai_caption_batch_submit_worker(job_id)

    persisted = api._openai_caption_batch_read(job_id)
    assert persisted["status"] == "failed"
    assert cleanup_calls
    assert cleanup_calls[0]["output_dir"] == artifact_dir
    assert persisted["error"]["orphan_file_cleanup"]["status"] == "ok"
    assert persisted["error"]["orphan_file_cleanup"]["deleted"] == 1


def test_openai_caption_batch_catchup_job_retries_missing_result_rows_as_full_visual_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    class FakeThread:
        def __init__(self, target, args=(), name=None, daemon=None):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api.threading, "Thread", FakeThread)
    source_dir = tmp_path / "jobs" / "openai_batches" / "ocap_missing_result_source"
    source_dir.mkdir(parents=True)
    image_path = tmp_path / "missing-result.jpg"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(image_path)
    label_path = tmp_path / "missing-result.txt"
    label_path.write_text("", encoding="utf-8")
    (tmp_path / "labelmap.txt").write_text("Object\n", encoding="utf-8")
    case = {
        "case_id": "image:train/missing-result.jpg:full",
        "name": "missing-result",
        "stem": "missing-result",
        "image_name": "missing-result.jpg",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "split": "train",
        "caption_mode": "full",
    }
    (source_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (source_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "case_id": case["case_id"],
                "image_name": "missing-result.jpg",
                "final_status": "failed",
                "failure_reason": "missing_batch_result",
                "generated_qa_pair_count": 0,
                "generated_qa_target_pair_count": "bad-local-artifact-count",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_missing_result_source",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "dataset_label": "Dataset A",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "openai_batch_shard_size": 100,
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "subcaptions_per_image": 2,
                "target_generated_qa_per_image": 2,
                "caption_request": {"request_text": "Describe it."},
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(source_dir),
            "logs": [],
        }
    )

    catchup = api._openai_caption_batch_catchup_job("ocap_missing_result_source")

    assert catchup["kind"] == "openai_caption_batch"
    assert catchup["catchup_for_job_id"] == "ocap_missing_result_source"
    assert catchup["case_count"] == 1
    catchup_dir = Path(catchup["output_dir"])
    catchup_cases = json.loads((catchup_dir / "cases.json").read_text(encoding="utf-8"))
    assert catchup_cases[0]["_openai_batch_missing_result_retry"] is True
    assert "_openai_batch_existing_caption" not in catchup_cases[0]
    assert catchup_cases[0]["_openai_batch_target_qa"] == 2
    batch_line = batch_smoke.build_batch_line(
        case=catchup_cases[0],
        file_id="file_missing_result_retry",
        dataset_root=tmp_path,
        args=SimpleNamespace(
            request_json=catchup_dir / "request_fields.json",
            max_output_tokens=3200,
            max_boxes=50,
            qa_count=2,
            model="gpt-5.5",
            reasoning_effort="high",
            image_detail="original",
        ),
    )
    content = batch_line["body"]["input"][0]["content"]
    assert any(part.get("type") == "input_image" and part.get("file_id") == "file_missing_result_retry" for part in content)
    prompt = next(part["text"] for part in content if part.get("type") == "input_text")
    assert "This is a catch-up request" not in prompt
    assert "Generate exactly 2 question-answer pairs" in prompt


def test_openai_caption_batch_catchup_retries_batch_error_rows_as_full_visual_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    class FakeThread:
        def __init__(self, target, args=(), name=None, daemon=None):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api.threading, "Thread", FakeThread)
    source_dir = tmp_path / "jobs" / "openai_batches" / "ocap_batch_error_source"
    source_dir.mkdir(parents=True)
    image_path = tmp_path / "server-error.jpg"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(image_path)
    label_path = tmp_path / "server-error.txt"
    label_path.write_text("", encoding="utf-8")
    (tmp_path / "labelmap.txt").write_text("Object\n", encoding="utf-8")
    case = {
        "case_id": "image:train/server-error.jpg:full",
        "name": "server-error",
        "stem": "server-error",
        "image_name": "server-error.jpg",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "split": "train",
        "caption_mode": "full",
    }
    (source_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (source_dir / "batch_error.jsonl").write_text(
        json.dumps(
            {
                "custom_id": case["case_id"],
                "response": {
                    "status_code": 503,
                    "body": {
                        "error": {
                            "type": "service_unavailable_error",
                            "code": "server_is_overloaded",
                            "message": "Our servers are currently overloaded. Please try again later.",
                        }
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = batch_smoke.collect_results(cases=[case], output_dir=source_dir, target_qa=2)
    results = api._openai_caption_batch_read_jsonl(source_dir / "results.jsonl")
    assert summary["failed_cases"] == 1
    assert results[0]["failure_reason"] == "batch_row_error"
    assert results[0]["status_code"] == 503
    assert results[0]["error"]["code"] == "server_is_overloaded"
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_batch_error_source",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "dataset_label": "Dataset A",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "openai_batch_shard_size": 100,
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "subcaptions_per_image": 2,
                "target_generated_qa_per_image": 2,
                "caption_request": {"request_text": "Describe it."},
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(source_dir),
            "logs": [],
        }
    )

    catchup = api._openai_caption_batch_catchup_job("ocap_batch_error_source")

    assert catchup["kind"] == "openai_caption_batch"
    assert catchup["catchup_for_job_id"] == "ocap_batch_error_source"
    assert catchup["case_count"] == 1
    catchup_dir = Path(catchup["output_dir"])
    catchup_cases = json.loads((catchup_dir / "cases.json").read_text(encoding="utf-8"))
    assert catchup_cases[0]["_openai_batch_failed_result_retry"] is True
    assert catchup_cases[0]["_openai_batch_missing_result_retry"] is False
    assert catchup_cases[0]["_openai_batch_retry_failure_reason"] == "batch_row_error"
    assert catchup_cases[0]["_openai_batch_target_qa"] == 2
    batch_line = batch_smoke.build_batch_line(
        case=catchup_cases[0],
        file_id="file_failed_result_retry",
        dataset_root=tmp_path,
        args=SimpleNamespace(
            request_json=catchup_dir / "request_fields.json",
            max_output_tokens=3200,
            max_boxes=50,
            qa_count=2,
            model="gpt-5.5",
            reasoning_effort="high",
            image_detail="original",
        ),
    )
    content = batch_line["body"]["input"][0]["content"]
    assert any(part.get("type") == "input_image" and part.get("file_id") == "file_failed_result_retry" for part in content)
    prompt = next(part["text"] for part in content if part.get("type") == "input_text")
    assert "This is a catch-up request" not in prompt
    assert "Generate exactly 2 question-answer pairs" in prompt


def test_openai_caption_batch_catchup_missing_linked_job_is_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    class FakeThread:
        def __init__(self, target, args=(), name=None, daemon=None):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api.threading, "Thread", FakeThread)
    source_dir = tmp_path / "jobs" / "openai_batches" / "ocap_stale_source"
    source_dir.mkdir(parents=True)
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (16, 16), color=(20, 30, 40)).save(image_path)
    label_path = tmp_path / "scene.txt"
    label_path.write_text("", encoding="utf-8")
    (tmp_path / "labelmap.txt").write_text("Object\n", encoding="utf-8")
    case = {
        "case_id": "image:scene.jpg:full",
        "name": "scene",
        "stem": "scene",
        "image_name": "scene.jpg",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "split": "train",
        "caption_mode": "full",
    }
    (source_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (source_dir / "incomplete_captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": case["case_id"],
                "image_name": "scene.jpg",
                "caption": "A partial paid caption.",
                "generated_qa_pairs": [{"question": "What is visible?", "answer": "One object is visible."}],
                "generated_qa_pair_count": 1,
                "generated_qa_target_pair_count": 2,
                "failure_reason": "generated_qa_incomplete",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_stale_source",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "dataset_label": "Dataset A",
            "last_catchup_job_id": "ocap_missing_catchup",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "openai_batch_shard_size": 100,
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "subcaptions_per_image": 2,
                "target_generated_qa_per_image": 2,
                "caption_request": {"request_text": "Describe it."},
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(source_dir),
            "logs": [],
        }
    )

    public_before = api._openai_caption_batch_job_public(api._openai_caption_batch_read("ocap_stale_source"))
    assert public_before["incomplete_resolution"]["status"] == "missing"
    assert public_before["incomplete_resolution"]["state"] == "retryable"

    catchup = api._openai_caption_batch_catchup_job("ocap_stale_source")

    assert catchup["catchup_for_job_id"] == "ocap_stale_source"
    assert catchup["case_count"] == 1
    source_after = api._openai_caption_batch_read("ocap_stale_source")
    assert source_after["last_catchup_job_id"] == catchup["job_id"]


def test_openai_caption_batch_catchup_handles_required_question_missing_without_count_deficit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    class FakeThread:
        def __init__(self, target, args=(), name=None, daemon=None):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api.threading, "Thread", FakeThread)
    source_dir = tmp_path / "jobs" / "openai_batches" / "ocap_required_missing"
    source_dir.mkdir(parents=True)
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (16, 16), color=(30, 40, 50)).save(image_path)
    label_path = tmp_path / "scene.txt"
    label_path.write_text("", encoding="utf-8")
    (tmp_path / "labelmap.txt").write_text("Object\n", encoding="utf-8")
    case = {
        "case_id": "image:scene.jpg:full",
        "name": "scene",
        "stem": "scene",
        "image_name": "scene.jpg",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "split": "train",
        "caption_mode": "full",
    }
    (source_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (source_dir / "incomplete_captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": case["case_id"],
                "image_name": "scene.jpg",
                "caption": "A paid caption with enough QA rows but one required question missing.",
                "generated_qa_pairs": [
                    {"question": "What is visible?", "answer": "One object is visible."},
                    {"question": "What type is visible?", "answer": "An object is visible."},
                ],
                "generated_qa_pair_count": 2,
                "generated_qa_target_pair_count": 2,
                "failure_reason": "required_qa_missing",
                "missing_required_questions": ["What color is the roof?"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_required_missing",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "dataset_label": "Dataset A",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "openai_batch_shard_size": 100,
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "instruction_qa_imposed_questions": [
                    "What type is visible?",
                    "What color is the roof?",
                ],
                "subcaptions_per_image": 2,
                "target_generated_qa_per_image": 2,
                "caption_request": {"request_text": "Describe it."},
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(source_dir),
            "logs": [],
        }
    )

    catchup = api._openai_caption_batch_catchup_job("ocap_required_missing")

    catchup_dir = Path(catchup["output_dir"])
    catchup_cases = json.loads((catchup_dir / "cases.json").read_text(encoding="utf-8"))
    assert catchup_cases[0]["_openai_batch_existing_caption"].startswith("A paid caption")
    assert catchup_cases[0]["_openai_batch_existing_qa_count"] == 1
    assert catchup_cases[0]["_openai_batch_existing_qa_pairs"] == [
        {"question": "What type is visible?", "answer": "An object is visible."}
    ]
    assert catchup_cases[0]["_openai_batch_total_qa_target"] == 2
    assert catchup_cases[0]["_openai_batch_target_qa"] == 1
    assert catchup_cases[0]["_openai_batch_missing_required_questions"] == ["What color is the roof?"]
    batch_line = batch_smoke.build_batch_line(
        case=catchup_cases[0],
        file_id="file_visual_retry",
        dataset_root=tmp_path,
        args=SimpleNamespace(
            request_json=catchup_dir / "request_fields.json",
            max_output_tokens=3200,
            max_boxes=50,
            qa_count=2,
            model="gpt-5.5",
            reasoning_effort="high",
            image_detail="original",
            instruction_qa_imposed_questions=[
                "What type is visible?",
                "What color is the roof?",
            ],
        ),
    )
    prompt = next(part["text"] for part in batch_line["body"]["input"][0]["content"] if part.get("type") == "input_text")
    required_block = prompt.split("After required questions", 1)[0]
    assert "What color is the roof?" in required_block
    assert "What type is visible?" not in required_block
    assert "Generate exactly 1 additional" in prompt


def test_openai_caption_batch_public_incomplete_rows_are_not_label_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_incomplete"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_cases": 3,
                "caption_rows": 2,
                "accepted_cases": 2,
                "incomplete_caption_rows": 1,
                "incomplete_cases": 1,
            }
        ),
        encoding="utf-8",
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_incomplete",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
            "warning_count": 1,
        }
    )

    public = api._openai_caption_batch_job_public(api._openai_caption_batch_read("ocap_incomplete"))

    assert public["output_summary"]["incomplete_caption_rows"] == 1
    assert public["warning_count"] == 1


def test_openai_caption_batch_detail_exposes_collected_output_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_samples"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image:scene.jpg:full",
                "image_name": "scene.jpg",
                "caption": "A collected caption.",
                "generated_qa_pairs": [
                    {"question": "What is visible?", "answer": "A building is visible."},
                    {"question": "Where is it?", "answer": "It is near the center."},
                ],
                "generated_qa_pair_count": 2,
                "generated_qa_target_pair_count": 2,
                "answer_format": "natural",
                "final_status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "incomplete_captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image:held.jpg:full",
                "image_name": "held.jpg",
                "caption": "A held caption.",
                "generated_qa_pairs": [
                    {"question": "What is visible?", "answer": "A building is visible."},
                ],
                "generated_qa_pair_count": 1,
                "generated_qa_target_pair_count": 2,
                "failure_reason": "required_qa_missing",
                "missing_required_questions": ["What color is the roof?"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image:scene.jpg:full",
                "image_name": "scene.jpg",
                "final_status": "ok",
                "caption_chars": 20,
                "generated_qa_pair_count": 2,
                "generated_qa_target_pair_count": 2,
                "answer_format": "natural",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_samples",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    detail = api._openai_caption_batch_detail("ocap_samples")

    samples = detail["collected_output_samples"]
    assert samples["accepted_caption_row_count"] == 1
    assert samples["incomplete_caption_row_count"] == 1
    assert samples["result_row_count"] == 1
    assert samples["accepted_caption_rows"][0]["caption"] == "A collected caption."
    assert samples["accepted_caption_rows"][0]["qa_pairs"][0]["question"] == "What is visible?"
    assert samples["incomplete_caption_rows"][0]["failure_reason"] == "required_qa_missing"
    assert samples["incomplete_caption_rows"][0]["missing_required_questions"] == ["What color is the roof?"]
    assert samples["result_rows"][0]["usage"] == {"input_tokens": 100, "output_tokens": 50}


def test_submit_batch_reuses_uploaded_input_file_and_records_create_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    batch_input = output_dir / "batch_input.jsonl"
    batch_input.write_text("{}\n", encoding="utf-8")
    batch_smoke.atomic_write_json(
        output_dir / "batch_input_file.json",
        {"response": {"id": "file_batch_existing"}},
    )
    uploads: list[str] = []

    def fail_upload(**kwargs):
        uploads.append(str(kwargs.get("file_path")))
        raise AssertionError("batch input should not be uploaded again")

    def fail_create(**kwargs):
        assert kwargs["path"] == "/batches"
        assert kwargs["body"]["input_file_id"] == "file_batch_existing"
        raise batch_smoke.OpenAIRequestError(
            operation="openai_http_error",
            status_code=400,
            detail=json.dumps({"error": {"code": "billing_hard_limit_reached"}}),
            headers={"x-request-id": "req_123"},
        )

    monkeypatch.setattr(batch_smoke, "multipart_upload", fail_upload)
    monkeypatch.setattr(batch_smoke, "request_json", fail_create)

    with pytest.raises(batch_smoke.OpenAIRequestError):
        batch_smoke.submit_batch(
            key="test-key",
            batch_input=batch_input,
            output_dir=output_dir,
            args=SimpleNamespace(
                timeout=10,
                model="gpt-5.5",
                reasoning_effort="high",
                image_detail="original",
                qa_count=8,
            ),
        )

    assert uploads == []
    error = json.loads((output_dir / "batch_create_error.json").read_text(encoding="utf-8"))
    assert error["input_file_id"] == "file_batch_existing"
    assert error["detail"]["error"]["code"] == "billing_hard_limit_reached"
    assert error["headers"]["x-request-id"] == "req_123"


def test_cleanup_unsubmitted_uploaded_files_deletes_remote_files_and_archives_local_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "image_files.jsonl").write_text(
        json.dumps({"case_id": "one", "file_id": "file_img_1"}) + "\n"
        + json.dumps({"case_id": "two", "file_id": "file_img_2"}) + "\n",
        encoding="utf-8",
    )
    batch_smoke.atomic_write_json(
        output_dir / "batch_input_file.json",
        {"response": {"id": "file_batch_input"}},
    )
    (output_dir / "batch_input.jsonl").write_text("{}\n", encoding="utf-8")
    deleted: list[str] = []

    def fake_request_json(**kwargs):
        assert kwargs["method"] == "DELETE"
        deleted.append(kwargs["path"].rsplit("/", 1)[-1])
        return {"deleted": True, "id": deleted[-1]}, {"x-request-id": f"req_{len(deleted)}"}

    monkeypatch.setattr(batch_smoke, "request_json", fake_request_json)

    report = batch_smoke.cleanup_unsubmitted_uploaded_files(
        key="test-key",
        output_dir=output_dir,
        timeout=10,
        reason="test_cleanup",
    )

    assert deleted == ["file_img_1", "file_img_2", "file_batch_input"]
    assert report["status"] == "ok"
    assert report["deleted"] == 3
    assert not (output_dir / "image_files.jsonl").exists()
    assert not (output_dir / "batch_input_file.json").exists()
    assert not (output_dir / "batch_input.jsonl").exists()
    assert len(list(output_dir.glob("image_files.jsonl.archived_after_orphan_cleanup_*.bak"))) == 1
    persisted = json.loads((output_dir / "orphan_file_cleanup_report.json").read_text(encoding="utf-8"))
    assert persisted["reason"] == "test_cleanup"
    assert persisted["archived_local_manifests"]


def test_cleanup_unsubmitted_uploaded_files_skips_when_batch_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "image_files.jsonl").write_text(
        json.dumps({"case_id": "one", "file_id": "file_img_1"}) + "\n",
        encoding="utf-8",
    )
    batch_smoke.atomic_write_json(output_dir / "batch.json", {"response": {"id": "batch_123"}})

    def fail_request_json(**_kwargs):
        raise AssertionError("submitted batch files must not be deleted")

    monkeypatch.setattr(batch_smoke, "request_json", fail_request_json)

    report = batch_smoke.cleanup_unsubmitted_uploaded_files(
        key="test-key",
        output_dir=output_dir,
        timeout=10,
    )

    assert report["status"] == "skipped"
    assert report["reason"] == "batch_already_submitted"
    assert (output_dir / "image_files.jsonl").exists()


def test_openai_caption_batch_job_persists_target_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    class FakeThread:
        def __init__(self, target, args=(), name=None, daemon=None):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api.threading, "Thread", FakeThread)
    api.OPENAI_CAPTION_BATCH_THREADS.clear()
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (16, 16), color=(20, 30, 40)).save(image_path)

    def fake_cases(payload, output_dir):
        label_path = output_dir / "case_labels" / "train" / "scene.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        return (
            {"dataset_label": "Test dataset"},
            [
                {
                    "case_id": "image:scene.jpg:full",
                    "name": "image_000001",
                    "stem": "scene",
                    "image_name": "scene.jpg",
                    "image_key": "scene.jpg",
                    "image_relpath": "scene.jpg",
                    "split": "train",
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ],
        )

    monkeypatch.setattr(api, "_qwen_caption_dataset_cases", fake_cases)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_effective_request_fields",
        lambda caption_request, set_and_forget=False: (dict(caption_request), {}),
    )

    job = api._start_openai_caption_batch_job(
        QwenCaptionDatasetJobRequest(
            dataset_id="dataset_a",
            caption_provider="openai",
            openai_service_tier="batch",
            caption_request={"user_prompt": "Describe the scene.", "caption_max_boxes": 12},
            instruction_dataset=True,
            subcaptions_per_image=8,
        )
    )

    assert job["kind"] == "openai_caption_batch"
    assert job["status"] == "preparing"
    artifact_dir = Path(job["output_dir"])
    assert (artifact_dir / "openai_batch_job.json").exists()
    assert (artifact_dir / "cases.json").exists()
    target = json.loads((artifact_dir / "target_manifest.json").read_text(encoding="utf-8"))
    assert target["dataset_id"] == "dataset_a"
    assert target["case_count"] == 1
    assert target["cases"][0]["image_key"] == "scene.jpg"
    assert target["cases"][0]["label_sha256"] == hashlib.sha256(b"0 0.5 0.5 0.2 0.2\n").hexdigest()
    assert target["dataset_fingerprint"]
    assert target["label_fingerprint"]
    assert target["target_identity"]["case_count"] == 1
    listed = api._openai_caption_batch_list_jobs()
    assert [item["job_id"] for item in listed] == [job["job_id"]]


def test_openai_caption_batch_shards_large_runs_under_one_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    class FakeThread:
        def __init__(self, target, args=(), name=None, daemon=None):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api.threading, "Thread", FakeThread)
    api.OPENAI_CAPTION_BATCH_THREADS.clear()
    image_paths = []
    for index in range(5):
        path = tmp_path / f"scene_{index}.jpg"
        Image.new("RGB", (16, 16), color=(20 + index, 30, 40)).save(path)
        image_paths.append(path)

    def fake_cases(payload, output_dir):
        cases = []
        for index, image_path in enumerate(image_paths):
            label_path = output_dir / "case_labels" / "train" / f"scene_{index}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            cases.append(
                {
                    "case_id": f"image:scene_{index}.jpg:full",
                    "name": f"image_{index + 1:06d}",
                    "stem": image_path.stem,
                    "image_name": image_path.name,
                    "image_key": image_path.name,
                    "image_relpath": image_path.name,
                    "split": "train",
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            )
        return {"dataset_label": "Test dataset"}, cases

    monkeypatch.setattr(api, "_qwen_caption_dataset_cases", fake_cases)
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_effective_request_fields",
        lambda caption_request, set_and_forget=False: (dict(caption_request), {}),
    )

    job = api._start_openai_caption_batch_job(
        QwenCaptionDatasetJobRequest(
            dataset_id="dataset_a",
            caption_provider="openai",
            openai_service_tier="batch",
            openai_batch_shard_size=2,
            caption_request={"user_prompt": "Describe the scene."},
            instruction_dataset=True,
            subcaptions_per_image=8,
        )
    )

    assert job["kind"] == "openai_caption_batch_collection"
    assert job["shard_count"] == 3
    assert job["shard_summary"]["total_shards"] == 3
    assert job["shard_summary"]["case_count"] == 5
    assert len(job["child_job_ids"]) == 3
    assert set(api.OPENAI_CAPTION_BATCH_THREADS) == {job["job_id"]}
    assert api.OPENAI_CAPTION_BATCH_THREADS[job["job_id"]].name.startswith("openai-caption-batch-collection-")
    listed = api._openai_caption_batch_list_jobs()
    assert [item["job_id"] for item in listed] == [job["job_id"]]
    child = api._openai_caption_batch_read(job["child_job_ids"][0])
    assert child["parent_job_id"] == job["job_id"]
    assert child["case_count"] == 2


def test_openai_caption_batch_poll_restarts_single_unsubmitted_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    class FakeThread:
        def __init__(self, target, args=(), name=None, daemon=None):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api.threading, "Thread", FakeThread)
    api.OPENAI_CAPTION_BATCH_THREADS.clear()
    job_id = "ocap_single_resume"
    artifact_dir = tmp_path / "jobs" / "openai_batches" / job_id
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "cases.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "request_fields.json").write_text("{}", encoding="utf-8")
    api._openai_caption_batch_write(
        {
            "job_id": job_id,
            "kind": "openai_caption_batch",
            "status": "preparing",
            "dataset_id": "dataset_a",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "instruction_dataset": True,
            },
            "output_dir": str(artifact_dir),
            "request_json": str(artifact_dir / "request_fields.json"),
            "cases_json": str(artifact_dir / "cases.json"),
        }
    )

    result = api._openai_caption_batch_poll_job(job_id)

    assert result["status"] == "preparing"
    assert set(api.OPENAI_CAPTION_BATCH_THREADS) == {job_id}
    thread = api.OPENAI_CAPTION_BATCH_THREADS[job_id]
    assert thread.started is True
    assert thread.name == f"openai-caption-batch-{job_id}"


def test_openai_caption_batch_no_case_instruction_run_materializes_locally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "_preflight_openai_caption_provider_request", lambda _payload: None)
    manifest = {
        "dataset_id": "dataset_a",
        "dataset_label": "Dataset A",
        "yolo_layout": "flat",
        "labelmap": ["Building"],
        "images": [],
    }
    monkeypatch.setattr(api, "_qwen_caption_dataset_cases", lambda payload, output_dir: (manifest, []))
    captured_artifacts = []

    def fake_materialize(*, dataset_id, payload, artifact_dir):
        captured_artifacts.append((dataset_id, payload, artifact_dir))
        return {
            "status": "ok",
            "artifact_dir": str(artifact_dir),
            "training_row_count": 3,
            "bundle_zip": str(artifact_dir / "bundle.zip"),
        }

    monkeypatch.setattr(api, "_openai_caption_batch_materialize_instruction_artifacts", fake_materialize)
    api.OPENAI_CAPTION_BATCH_THREADS.clear()

    result = api._start_openai_caption_batch_job(
        QwenCaptionDatasetJobRequest(
            dataset_id="dataset_a",
            caption_provider="openai",
            openai_service_tier="batch",
            instruction_dataset=True,
            include_caption0_in_training=False,
            include_generated_qa_in_training=False,
            include_deterministic_metadata_qa=True,
        )
    )

    assert result["status"] == "imported"
    assert result["local_materialization"] is True
    assert result["case_count"] == 0
    assert result["openai"]["remote_submission_skipped"] is True
    assert result["result"]["instruction_artifacts"]["status"] == "ok"
    assert not api.OPENAI_CAPTION_BATCH_THREADS
    assert len(captured_artifacts) == 1
    stored = api._openai_caption_batch_read(result["job_id"])
    assert json.loads(Path(stored["cases_json"]).read_text()) == []
    public = api._openai_caption_batch_job_public(stored, dataset_id="dataset_a", include_match=True)
    assert public["match_status"] == "local_materialization"
    assert public["import_ready"] is False


def test_openai_caption_batch_poll_restarts_collection_submitter_for_unsubmitted_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    class FakeThread:
        def __init__(self, target, args=(), name=None, daemon=None):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api.threading, "Thread", FakeThread)
    api.OPENAI_CAPTION_BATCH_THREADS.clear()
    parent_dir = tmp_path / "jobs" / "openai_batches" / "ocap_parent_resume_submit"
    child_dir = tmp_path / "jobs" / "openai_batches" / "ocap_child_resume_submit"
    parent_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    (child_dir / "cases.json").write_text(json.dumps([{"case_id": "case_1", "image_name": "scene.jpg"}]), encoding="utf-8")
    (child_dir / "request_fields.json").write_text(json.dumps({"user_prompt": "Describe."}), encoding="utf-8")
    base_request = {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"}
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_child_resume_submit",
            "kind": "openai_caption_batch",
            "status": "preparing",
            "dataset_id": "dataset_a",
            "case_count": 1,
            "request": dict(base_request),
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(child_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_parent_resume_submit",
            "kind": "openai_caption_batch_collection",
            "status": "in_progress",
            "dataset_id": "dataset_a",
            "child_job_ids": ["ocap_child_resume_submit"],
            "request": dict(base_request),
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(parent_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_poll_job("ocap_parent_resume_submit")

    assert result["status"] == "in_progress"
    assert set(api.OPENAI_CAPTION_BATCH_THREADS) == {"ocap_parent_resume_submit"}
    assert api.OPENAI_CAPTION_BATCH_THREADS["ocap_parent_resume_submit"].name.startswith("openai-caption-batch-collection-")


def test_openai_caption_batch_submit_artifacts_present_accepts_workspace_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.chdir(tmp_path)
    artifact_dir = Path("uploads/qwen_caption_jobs/openai_batches/ocap_child_workspace_relative")
    artifact_dir.mkdir(parents=True)
    cases_path = artifact_dir / "cases.json"
    request_path = artifact_dir / "request_fields.json"
    cases_path.write_text("[]", encoding="utf-8")
    request_path.write_text("{}", encoding="utf-8")

    assert api._openai_caption_batch_submit_artifacts_present(
        {
            "output_dir": str(artifact_dir),
            "cases_json": str(cases_path),
            "request_json": str(request_path),
        }
    )


def test_openai_caption_batch_target_verification_uses_image_hash_as_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_identity"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "target_manifest.json").write_text(
        json.dumps(
            {
                "format": "tator_openai_caption_batch_target_v1",
                "dataset_id": "dataset_a",
                "case_count": 1,
                "cases": [
                    {
                        "image_key": "scene.jpg",
                        "image_name": "scene.jpg",
                        "image_sha256": "same-image",
                        "label_sha256": "old-labels",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_current_manifest_rows",
        lambda dataset_id, selected_image_keys=None: {
            "scene.jpg": {
                "image_sha256": "same-image",
                "label_sha256": "new-labels",
            }
        },
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_identity",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    ok, report = api._openai_caption_batch_verify_target(
        api._openai_caption_batch_read("ocap_identity"),
        dataset_id="dataset_a",
    )

    assert ok is True
    assert report["match_status"] == "exact_dataset_match_label_warning"
    assert report["warning_count"] == 1
    assert report["warnings"][0]["reason"] == "label_sha256_mismatch"

    monkeypatch.setattr(
        api,
        "_openai_caption_batch_current_manifest_rows",
        lambda dataset_id, selected_image_keys=None: {
            "scene.jpg": {
                "image_sha256": "different-image",
                "label_sha256": "old-labels",
            }
        },
    )

    ok, report = api._openai_caption_batch_verify_target(
        api._openai_caption_batch_read("ocap_identity"),
        dataset_id="dataset_a",
    )

    assert ok is False
    assert report["mismatches"][0]["reason"] == "image_sha256_mismatch"

    monkeypatch.setattr(
        api,
        "_openai_caption_batch_current_manifest_rows",
        lambda dataset_id, selected_image_keys=None: {
            "scene.jpg": {
                "image_sha256": "",
                "label_sha256": "old-labels",
            }
        },
    )

    ok, report = api._openai_caption_batch_verify_target(
        api._openai_caption_batch_read("ocap_identity"),
        dataset_id="dataset_a",
    )

    assert ok is False
    assert report["mismatches"][0]["reason"] == "image_sha256_missing"


def test_openai_caption_batch_verify_hashes_only_target_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_subset"
    artifact_dir.mkdir(parents=True)
    cases = [
        {
            "image_key": "first.jpg",
            "image_name": "first.jpg",
            "image_relpath": "train/first.jpg",
            "image_sha256": "first-image",
            "label_sha256": "first-label",
        },
        {
            "image_key": "second.jpg",
            "image_name": "second.jpg",
            "image_relpath": "train/second.jpg",
            "image_sha256": "second-image",
            "label_sha256": "second-label",
        },
    ]
    (artifact_dir / "target_manifest.json").write_text(
        json.dumps(
            {
                "format": "tator_openai_caption_batch_target_v1",
                "dataset_id": "dataset_a",
                "case_count": 2,
                "cases": cases,
                "target_identity": api._openai_caption_batch_identity_from_rows(cases),
            }
        ),
        encoding="utf-8",
    )
    captured_selected: list[set[str]] = []

    def fake_current_rows(dataset_id, selected_image_keys=None):
        captured_selected.append(set(selected_image_keys or set()))
        return {"first.jpg": {"image_sha256": "first-image", "label_sha256": "first-label"}}

    monkeypatch.setattr(api, "_openai_caption_batch_current_manifest_rows", fake_current_rows)
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_subset",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    ok, report = api._openai_caption_batch_verify_target(
        api._openai_caption_batch_read("ocap_subset"),
        dataset_id="dataset_a",
        selected_image_keys=["first.jpg"],
    )

    assert ok is True
    assert report["case_count"] == 1
    assert captured_selected == [{"first.jpg", "train/first.jpg"}]


def test_openai_caption_batch_verify_matches_current_rows_by_image_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_alias"
    artifact_dir.mkdir(parents=True)
    cases = [
        {
            "image_key": "train/scene.jpg",
            "image_name": "scene.jpg",
            "image_relpath": "scene.jpg",
            "image_sha256": "same-image",
            "label_sha256": "same-label",
        }
    ]
    (artifact_dir / "target_manifest.json").write_text(
        json.dumps(
            {
                "format": "tator_openai_caption_batch_target_v1",
                "dataset_id": "dataset_a",
                "case_count": 1,
                "cases": cases,
                "target_identity": api._openai_caption_batch_identity_from_rows(cases),
            }
        ),
        encoding="utf-8",
    )
    captured_selected: list[set[str]] = []

    def fake_current_rows(dataset_id, selected_image_keys=None):
        captured_selected.append(set(selected_image_keys or set()))
        return {
            "scene.jpg": {
                "image_key": "scene.jpg",
                "image_name": "scene.jpg",
                "image_relpath": "scene.jpg",
                "image_sha256": "same-image",
                "label_sha256": "same-label",
            }
        }

    monkeypatch.setattr(api, "_openai_caption_batch_current_manifest_rows", fake_current_rows)
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_alias",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    ok, report = api._openai_caption_batch_verify_target(
        api._openai_caption_batch_read("ocap_alias"),
        dataset_id="dataset_a",
    )

    assert ok is True
    assert report["match_status"] == "exact_dataset_match"
    assert report["matched_case_count"] == 1
    assert captured_selected == [{"train/scene.jpg", "scene.jpg"}]


def test_openai_caption_batch_verify_allows_same_fingerprint_with_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_renamed"
    artifact_dir.mkdir(parents=True)
    cases = [
        {
            "image_key": "scene.jpg",
            "image_name": "scene.jpg",
            "image_sha256": "same-image",
            "label_sha256": "same-label",
        }
    ]
    identity = api._openai_caption_batch_identity_from_rows(cases)
    (artifact_dir / "target_manifest.json").write_text(
        json.dumps(
            {
                "format": "tator_openai_caption_batch_target_v1",
                "dataset_id": "dataset_old",
                "case_count": 1,
                "dataset_fingerprint": identity["image_fingerprint"],
                "label_fingerprint": identity["label_fingerprint"],
                "target_identity": identity,
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_current_manifest_rows",
        lambda dataset_id, selected_image_keys=None: {"scene.jpg": {"image_sha256": "same-image", "label_sha256": "same-label"}},
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_renamed",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_old",
            "request": {
                "dataset_id": "dataset_old",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    ok, report = api._openai_caption_batch_verify_target(
        api._openai_caption_batch_read("ocap_renamed"),
        dataset_id="dataset_new",
    )
    assert ok is False
    assert report["reason"] == "dataset_id_confirmation_required"
    assert report["match_status"] == "same_images_different_dataset_id"

    ok, report = api._openai_caption_batch_verify_target(
        api._openai_caption_batch_read("ocap_renamed"),
        dataset_id="dataset_new",
        import_mode="allow_same_fingerprint_dataset_id_change",
    )
    assert ok is True
    assert report["match_status"] == "same_images_different_dataset_id"


def test_openai_caption_batch_scan_and_adopt_classifies_unregistered_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "legacy_run"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "batch_status.json").write_text(
        json.dumps(
            {
                "response": {
                    "id": "batch_legacy",
                    "status": "completed",
                    "request_counts": {"total": 1, "completed": 1, "failed": 0},
                    "metadata": {"model": "gpt-5.5", "image_detail": "original", "qa_count": "8"},
                }
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "target_manifest.json").write_text(
        json.dumps(
            {
                "format": "tator_openai_caption_batch_target_v1",
                "dataset_id": "dataset_a",
                "case_count": 1,
                "cases": [{"image_key": "scene.jpg", "image_sha256": "hash"}],
            }
        ),
        encoding="utf-8",
    )

    scan = api._openai_caption_batch_scan({"roots": [str(tmp_path / "jobs")]})

    artifacts = [item for item in scan["artifacts"] if item["openai_batch_id"] == "batch_legacy"]
    assert artifacts
    assert artifacts[0]["status"] == "adoptable"
    assert artifacts[0]["adoptable"] is True

    adopted = api._openai_caption_batch_adopt({"artifact_dir": str(artifact_dir), "dataset_id": "dataset_a"})

    assert adopted["openai_batch_id"] == "batch_legacy"
    assert adopted["target_snapshot_status"] == "present"
    assert adopted["dataset_id"] == "dataset_a"


def test_openai_caption_batch_scan_prioritizes_explicit_artifact_path_when_broad_root_is_capped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "OPENAI_CAPTION_BATCH_SCAN_VISIT_LIMIT", 2)
    broad_root = tmp_path / "jobs" / "broad"
    broad_root.mkdir(parents=True)
    current = broad_root
    for index in range(8):
        current = current / f"nested_{index}"
        current.mkdir()
    artifact_dir = tmp_path / "jobs" / "explicit_legacy"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "batch_status.json").write_text(
        json.dumps({"response": {"id": "batch_explicit", "status": "completed"}}),
        encoding="utf-8",
    )

    scan = api._openai_caption_batch_scan({"artifact_dirs": [str(artifact_dir)]})

    assert scan["truncated"] is True
    assert scan["artifacts"][0]["openai_batch_id"] == "batch_explicit"
    assert scan["roots"][0] == str(artifact_dir.resolve(strict=False))


def test_openai_caption_batch_adopt_synthesizes_target_snapshot_from_legacy_cases_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "legacy_script"
    artifact_dir.mkdir(parents=True)
    image_path = tmp_path / "images" / "scene.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (16, 16), color=(60, 70, 80)).save(image_path)
    label_path = tmp_path / "labels" / "scene.txt"
    label_path.parent.mkdir()
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    cases_json = tmp_path / "legacy_cases.json"
    cases_json.write_text(
        json.dumps(
            [
                {
                    "case_id": "image:scene.jpg:full",
                    "image_name": "scene.jpg",
                    "image_key": "scene.jpg",
                    "image_relpath": "scene.jpg",
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "label_count": 1,
                    "class_counts": {"Building": 1},
                    "caption_mode": "full",
                }
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"dataset_root": str(tmp_path), "cases_json": str(cases_json), "qa_count": 8}),
        encoding="utf-8",
    )
    (artifact_dir / "batch_status.json").write_text(
        json.dumps(
            {
                "response": {
                    "id": "batch_legacy_script",
                    "status": "completed",
                    "request_counts": {"total": 1, "completed": 1, "failed": 0},
                }
            }
        ),
        encoding="utf-8",
    )

    adopted = api._openai_caption_batch_adopt({"artifact_dir": str(artifact_dir), "dataset_id": "dataset_a"})

    assert adopted["openai_batch_id"] == "batch_legacy_script"
    assert adopted["target_snapshot_status"] == "legacy_synthesized"
    target_path = Path(adopted["target_manifest"])
    assert target_path.exists()
    assert target_path.parent == Path(adopted["output_dir"])
    target = json.loads(target_path.read_text(encoding="utf-8"))
    assert target["source"] == "legacy_artifact_adoption"
    assert target["dataset_id"] == "dataset_a"
    assert target["cases"][0]["image_sha256"] == api._openai_caption_batch_file_sha256(image_path)


def test_openai_caption_batch_scan_reports_limit_when_recursive_search_is_capped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "OPENAI_CAPTION_BATCH_SCAN_VISIT_LIMIT", 3)
    broad_root = tmp_path / "jobs" / "broad"
    broad_root.mkdir(parents=True)
    current = broad_root
    for index in range(8):
        current = current / f"nested_{index}"
        current.mkdir()

    scan = api._openai_caption_batch_scan({"roots": [str(broad_root)]})

    assert scan["truncated"] is True
    assert scan["scan_limit_reached"] is True
    assert scan["visited_dirs"] >= 3


def test_openai_caption_batch_public_payload_redacts_api_key_paths() -> None:
    import localinferenceapi as api

    public = api._openai_caption_batch_job_public(
        {
            "job_id": "ocap_redact",
            "kind": "openai_caption_batch",
            "status": "submitted",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_api_key_path": "/Users/example/openAI_API_KEY_DoNotCommit",
                "openai_api_key_file": "/Users/example/also_secret",
            },
        }
    )

    assert "openai_api_key_path" not in public["request"]
    assert "openai_api_key_file" not in public["request"]
    assert public["request"]["openai_api_key_configured"] is True
    assert public["request"]["openai_api_key_source"] == "file"


def test_openai_caption_batch_import_requires_explicit_dataset_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_no_dataset"
    artifact_dir.mkdir(parents=True)
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_no_dataset",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    with pytest.raises(api.HTTPException) as exc:
        api._openai_caption_batch_import_job("ocap_no_dataset", dataset_id="")

    assert exc.value.status_code == 400
    assert exc.value.detail == "openai_caption_batch_import_dataset_id_required"


def test_openai_caption_batch_verify_blocks_empty_target_snapshot_before_dataset_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_current_manifest_rows",
        lambda *args, **kwargs: pytest.fail("empty target snapshots must not hash the current dataset"),
    )
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_empty_target"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "target_manifest.json").write_text(
        json.dumps(
            {
                "format": "tator_openai_caption_batch_target_v1",
                "dataset_id": "dataset_a",
                "case_count": 0,
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_empty_target",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    ok, report = api._openai_caption_batch_verify_target(
        api._openai_caption_batch_read("ocap_empty_target"),
        dataset_id="dataset_a",
    )

    assert ok is False
    assert report["reason"] == "target_cases_missing"
    assert report["match_status"] == "unknown_missing_snapshot"


def test_openai_caption_batch_collect_skips_imported_child_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "_openai_caption_batch_api_key", lambda path_value: "test-key")
    downloads: list[str] = []

    def fake_download_outputs(**kwargs):
        downloads.append(str(kwargs["output_dir"]))

    monkeypatch.setattr(batch_smoke, "download_outputs", fake_download_outputs)
    monkeypatch.setattr(
        batch_smoke,
        "collect_results",
            lambda *, cases, output_dir, target_qa, answer_format="natural", imposed_questions=None, **_kwargs: {
            "total_cases": len(cases),
            "caption_rows": len(cases),
            "generated_qa_rows": 0,
        },
    )
    parent_dir = tmp_path / "jobs" / "openai_batches" / "ocap_parent"
    imported_dir = tmp_path / "jobs" / "openai_batches" / "ocap_child_imported"
    collected_dir = tmp_path / "jobs" / "openai_batches" / "ocap_child_collected"
    for path in (parent_dir, imported_dir, collected_dir):
        path.mkdir(parents=True)
    (collected_dir / "batch_status.json").write_text(json.dumps({"response": {"id": "batch_child"}}), encoding="utf-8")
    (collected_dir / "cases.json").write_text(json.dumps([{"case_id": "case_1"}]), encoding="utf-8")
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_child_imported",
            "kind": "openai_caption_batch",
            "status": "imported",
            "dataset_id": "dataset_a",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "import_report": {"saved_captions": 3, "saved_generated_qa_rows": 4},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(imported_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_child_collected",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_child",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(collected_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_parent",
            "kind": "openai_caption_batch_collection",
            "status": "collected",
            "dataset_id": "dataset_a",
            "child_job_ids": ["ocap_child_imported", "ocap_child_collected"],
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(parent_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_collect_job("ocap_parent")

    assert result["status"] == "collected"
    assert api._openai_caption_batch_read("ocap_child_imported")["status"] == "imported"
    assert len(downloads) == 1
    assert downloads[0] == str(collected_dir.resolve(strict=False))


def test_openai_caption_batch_cancel_preserves_final_single_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    calls: list[str] = []
    monkeypatch.setattr(batch_smoke, "request_json", lambda **kwargs: calls.append(kwargs["path"]))
    job_dir = tmp_path / "jobs" / "openai_batches" / "ocap_collected_final"
    job_dir.mkdir(parents=True)
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_collected_final",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_collected",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(job_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_cancel_job("ocap_collected_final")

    assert result["status"] == "collected"
    assert api._openai_caption_batch_read("ocap_collected_final")["status"] == "collected"
    assert calls == []


def test_openai_caption_batch_archive_rejects_active_single_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    job_dir = tmp_path / "jobs" / "openai_batches" / "ocap_active_archive"
    job_dir.mkdir(parents=True)
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_active_archive",
            "kind": "openai_caption_batch",
            "status": "in_progress",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_active",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(job_dir),
            "logs": [],
        }
    )

    with pytest.raises(api.HTTPException) as exc:
        api._openai_caption_batch_archive_job("ocap_active_archive", archived=True)

    assert exc.value.status_code == 409
    assert exc.value.detail == "openai_caption_batch_archive_active_job"
    assert not api._openai_caption_batch_read("ocap_active_archive").get("archived")


def test_openai_caption_batch_archive_rejects_collection_with_active_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    parent_dir = tmp_path / "jobs" / "openai_batches" / "ocap_parent_active_archive"
    active_dir = tmp_path / "jobs" / "openai_batches" / "ocap_child_active_archive"
    collected_dir = tmp_path / "jobs" / "openai_batches" / "ocap_child_collected_archive"
    for path in (parent_dir, active_dir, collected_dir):
        path.mkdir(parents=True)
    base_request = {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"}
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_child_active_archive",
            "kind": "openai_caption_batch",
            "status": "submitted",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_active",
            "request": dict(base_request),
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(active_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_child_collected_archive",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_collected",
            "request": dict(base_request),
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(collected_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_parent_active_archive",
            "kind": "openai_caption_batch_collection",
            "status": "in_progress",
            "dataset_id": "dataset_a",
            "child_job_ids": ["ocap_child_active_archive", "ocap_child_collected_archive"],
            "request": dict(base_request),
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(parent_dir),
            "logs": [],
        }
    )

    with pytest.raises(api.HTTPException) as exc:
        api._openai_caption_batch_archive_job("ocap_parent_active_archive", archived=True)

    assert exc.value.status_code == 409
    assert exc.value.detail == "openai_caption_batch_archive_active_collection"
    parent = api._openai_caption_batch_read("ocap_parent_active_archive")
    assert not parent.get("archived")
    assert parent["shard_summary"]["active_shards"] == 1


def test_openai_caption_batch_archive_and_restore_terminal_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    job_dir = tmp_path / "jobs" / "openai_batches" / "ocap_terminal_archive"
    job_dir.mkdir(parents=True)
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_terminal_archive",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_collected",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(job_dir),
            "logs": [],
        }
    )

    archived = api._openai_caption_batch_archive_job("ocap_terminal_archive", archived=True)
    restored = api._openai_caption_batch_archive_job("ocap_terminal_archive", archived=False)

    assert archived["archived"] is True
    assert archived["archived_at"]
    assert restored["archived"] is False
    assert restored["archived_at"] == ""


def test_openai_caption_batch_parent_cancel_preserves_usable_child_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "_openai_caption_batch_api_key", lambda path_value: "test-key")
    cancelled_paths: list[str] = []

    def fake_request_json(**kwargs):
        cancelled_paths.append(str(kwargs["path"]))
        return {"id": "batch_active", "status": "cancelling"}, {}

    monkeypatch.setattr(batch_smoke, "request_json", fake_request_json)
    batch_root = tmp_path / "jobs" / "openai_batches"
    parent_dir = batch_root / "ocap_parent_cancel"
    imported_dir = batch_root / "ocap_cancel_imported"
    collected_dir = batch_root / "ocap_cancel_collected"
    active_dir = batch_root / "ocap_cancel_active"
    for path in (parent_dir, imported_dir, collected_dir, active_dir):
        path.mkdir(parents=True)
    base_request = {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"}
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_cancel_imported",
            "kind": "openai_caption_batch",
            "status": "imported",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_imported",
            "request": dict(base_request),
            "import_report": {"saved_captions": 1, "saved_generated_qa_rows": 8},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(imported_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_cancel_collected",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_collected",
            "request": dict(base_request),
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(collected_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_cancel_active",
            "kind": "openai_caption_batch",
            "status": "in_progress",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_active",
            "request": dict(base_request),
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(active_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_parent_cancel",
            "kind": "openai_caption_batch_collection",
            "status": "in_progress",
            "dataset_id": "dataset_a",
            "child_job_ids": ["ocap_cancel_imported", "ocap_cancel_collected", "ocap_cancel_active"],
            "request": dict(base_request),
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(parent_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_cancel_job("ocap_parent_cancel")

    assert cancelled_paths == ["/batches/batch_active/cancel"]
    assert api._openai_caption_batch_read("ocap_cancel_imported")["status"] == "imported"
    assert api._openai_caption_batch_read("ocap_cancel_collected")["status"] == "collected"
    assert api._openai_caption_batch_read("ocap_cancel_active")["status"] == "cancelled"
    assert result["status"] == "partial_failed"
    assert result["shard_summary"]["status_counts"]["imported"] == 1
    assert result["shard_summary"]["status_counts"]["collected"] == 1
    assert result["shard_summary"]["status_counts"]["cancelled"] == 1
    assert "preserved" in result["message"]


def test_openai_caption_batch_parent_collect_surfaces_child_collection_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "_openai_caption_batch_api_key", lambda path_value: "test-key")
    monkeypatch.setattr(batch_smoke, "poll_batch", lambda **kwargs: {"response": {"id": "batch_child", "status": "completed"}})
    monkeypatch.setattr(batch_smoke, "download_outputs", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("download boom")))
    parent_dir = tmp_path / "jobs" / "openai_batches" / "ocap_parent_error"
    child_dir = tmp_path / "jobs" / "openai_batches" / "ocap_child_error"
    for path in (parent_dir, child_dir):
        path.mkdir(parents=True)
    (child_dir / "batch_status.json").write_text(json.dumps({"response": {"id": "batch_child"}}), encoding="utf-8")
    (child_dir / "cases.json").write_text(json.dumps([{"case_id": "case_1"}]), encoding="utf-8")
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_child_error",
            "kind": "openai_caption_batch",
            "status": "completed_remote",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_child",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(child_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_parent_error",
            "kind": "openai_caption_batch_collection",
            "status": "completed_remote",
            "dataset_id": "dataset_a",
            "child_job_ids": ["ocap_child_error"],
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(parent_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_collect_job("ocap_parent_error")

    assert result["status"] == "partial_failed"
    assert result["shard_summary"]["collection_error_shards"] == 1
    assert result["result"]["collection_errors"][0]["message"] == "download boom"
    child = api._openai_caption_batch_read("ocap_child_error")
    assert child["status"] == "completed_remote"
    assert child["collection_error"]["message"] == "download boom"


def test_openai_caption_batch_collect_success_clears_stale_collection_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "_openai_caption_batch_api_key", lambda path_value: "test-key")
    monkeypatch.setattr(batch_smoke, "poll_batch", lambda **kwargs: {"response": {"id": "batch_child", "status": "completed"}})
    monkeypatch.setattr(batch_smoke, "download_outputs", lambda **kwargs: None)
    monkeypatch.setattr(
        batch_smoke,
        "collect_results",
            lambda *, cases, output_dir, target_qa, answer_format="natural", imposed_questions=None, **_kwargs: {
            "total_cases": len(cases),
            "caption_rows": len(cases),
            "generated_qa_rows": 0,
        },
    )
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_retry_child"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "batch_status.json").write_text(json.dumps({"response": {"id": "batch_child"}}), encoding="utf-8")
    (artifact_dir / "cases.json").write_text(json.dumps([{"case_id": "case_1"}]), encoding="utf-8")
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_retry_child",
            "kind": "openai_caption_batch",
            "status": "completed_remote",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_child",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "collection_error": {"message": "old error"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_collect_job("ocap_retry_child")

    assert result["status"] == "collected"
    assert "collection_error" not in api._openai_caption_batch_read("ocap_retry_child")


def test_openai_caption_batch_collect_normalizes_raw_completed_and_uses_best_batch_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "_openai_caption_batch_api_key", lambda path_value: "test-key")
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_poll_job",
        lambda job_id: api._openai_caption_batch_job_public(api._openai_caption_batch_read(job_id)),
    )
    downloaded: list[str] = []

    def fake_download_outputs(**kwargs):
        response = kwargs["batch"].get("response")
        downloaded.append(response["output_file_id"])

    monkeypatch.setattr(batch_smoke, "download_outputs", fake_download_outputs)
    monkeypatch.setattr(
        batch_smoke,
        "collect_results",
        lambda *, cases, output_dir, target_qa, answer_format="natural", imposed_questions=None, **_kwargs: {
            "total_cases": len(cases),
            "caption_rows": len(cases),
            "generated_qa_rows": 0,
        },
    )
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_raw_completed"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "batch_status.json").write_text(
        json.dumps({"response": {"id": "batch_child", "status": "in_progress"}}),
        encoding="utf-8",
    )
    (artifact_dir / "cases.json").write_text(json.dumps([{"case_id": "case_1"}]), encoding="utf-8")
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_raw_completed",
            "kind": "openai_caption_batch",
            "status": "completed",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_child",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "result": {
                "batch": {
                    "id": "batch_child",
                    "status": "completed",
                    "output_file_id": "file-output",
                    "error_file_id": "file-error",
                    "request_counts": {"total": 1, "completed": 1, "failed": 0},
                }
            },
            "logs": [],
        }
    )
    persisted = json.loads((artifact_dir / "openai_batch_job.json").read_text(encoding="utf-8"))
    persisted["status"] = "completed"
    (artifact_dir / "openai_batch_job.json").write_text(json.dumps(persisted), encoding="utf-8")

    result = api._openai_caption_batch_collect_job("ocap_raw_completed")

    assert result["status"] == "collected"
    assert downloaded == ["file-output"]
    persisted_status = json.loads((artifact_dir / "batch_status.json").read_text(encoding="utf-8"))
    assert persisted_status["response"]["status"] == "completed"
    assert persisted_status["response"]["output_file_id"] == "file-output"


def test_openai_caption_batch_collection_summary_normalizes_raw_openai_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    parent_dir = tmp_path / "jobs" / "openai_batches" / "ocap_parent_raw_status"
    child_dir = tmp_path / "jobs" / "openai_batches" / "ocap_child_raw_status"
    parent_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_child_raw_status",
            "kind": "openai_caption_batch",
            "status": "completed",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_child",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "case_count": 1,
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(child_dir),
            "logs": [],
        }
    )
    persisted = json.loads((child_dir / "openai_batch_job.json").read_text(encoding="utf-8"))
    persisted["status"] = "completed"
    (child_dir / "openai_batch_job.json").write_text(json.dumps(persisted), encoding="utf-8")
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_parent_raw_status",
            "kind": "openai_caption_batch_collection",
            "status": "in_progress",
            "dataset_id": "dataset_a",
            "child_job_ids": ["ocap_child_raw_status"],
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(parent_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_refresh_collection_job("ocap_parent_raw_status")

    assert result["status"] == "completed_remote"
    assert result["shard_summary"]["status_counts"] == {"completed_remote": 1}


def test_openai_caption_batch_import_blocks_when_target_snapshot_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_blocked"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "target_manifest.json").write_text(
        json.dumps(
            {
                "format": "tator_openai_caption_batch_target_v1",
                "dataset_id": "dataset_a",
                "case_count": 1,
                "cases": [{"image_key": "scene.jpg", "label_sha256": "old"}],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps({"case_id": "case", "caption": "Caption.", "generated_qa_pairs": []}) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "cases.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_current_manifest_rows",
        lambda dataset_id, selected_image_keys=None: {"scene.jpg": {"label_sha256": "new"}},
    )
    job = api._openai_caption_batch_write(
        {
            "job_id": "ocap_blocked",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "openai_batch_require_label_hash_match": True,
                "instruction_dataset": True,
                "subcaptions_per_image": 8,
                "test_outputs_count_toward_completion": True,
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )
    assert job["status"] == "collected"

    result = api._openai_caption_batch_import_job("ocap_blocked", dataset_id="dataset_a")

    assert result["status"] == "import_blocked"
    assert result["import_report"]["reason"] == "target_snapshot_mismatch"


def test_openai_caption_batch_import_saves_caption_and_generated_qa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_import"
    artifact_dir.mkdir(parents=True)
    case = {
        "case_id": "image:scene.jpg:full",
        "image_name": "scene.jpg",
        "image_path": str(tmp_path / "scene.jpg"),
        "split": "train",
    }
    (artifact_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image:scene.jpg:full",
                "image_name": "scene.jpg",
                "caption": "A grounded caption.",
                "generated_qa_pairs": [
                    {"question": "What is present?", "answer": "A building is present."},
                    {"question": "Where is it?", "answer": "It is near the center."},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_verify_target",
        lambda job, dataset_id=None, **kwargs: (
            True,
            {"reason": "ok", "dataset_id": "dataset_a", "case_count": 1, "target_hash": "hash"},
        ),
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_image_completion_counts",
        lambda dataset_id, image_name, split=None: {"generated_qa_count": 0},
    )
    monkeypatch.setattr(api, "_qwen_caption_dataset_soft_archive_generated_outputs", lambda *args, **kwargs: {})
    saved_captions = []
    saved_qa = []

    def fake_add_caption(dataset_id, image_name, payload, *, allow_active_caption_job_id=None):
        saved_captions.append((dataset_id, image_name, dict(payload)))
        return {"status": "saved"}

    def fake_add_qa(dataset_id, image_name, records, *, split=None):
        saved_qa.extend(dict(record) for record in records)
        return len(records)

    monkeypatch.setattr(api, "_add_caption_impl", fake_add_caption)
    monkeypatch.setattr(api, "_dataset_caption_add_instruction_records", fake_add_qa)
    materialized: list[tuple[str, Path]] = []

    def fake_materialize_instruction_artifacts(*, dataset_id, payload, artifact_dir):
        materialized.append((dataset_id, Path(artifact_dir)))
        return {
            "status": "ok",
            "artifact_dir": str(artifact_dir),
            "bundle_zip": str(Path(artifact_dir) / "bundle.zip"),
            "report_json": str(Path(artifact_dir) / "caption_instruction_report.json"),
            "training_row_count": 3,
        }

    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_materialize_instruction_artifacts",
        fake_materialize_instruction_artifacts,
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_import",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_123",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "save_text_labels": True,
                "generated_make_primary": False,
                "subcaptions_per_image": 2,
                "target_generated_qa_per_image": 2,
                "test_outputs_count_toward_completion": True,
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_import_job("ocap_import", dataset_id="dataset_a")

    assert result["status"] == "imported"
    assert result["import_report"]["saved_captions"] == 1
    assert result["import_report"]["saved_generated_qa_rows"] == 2
    assert saved_captions[0][0:2] == ("dataset_a", "scene.jpg")
    assert saved_captions[0][2]["source"] == "openai_caption_batch"
    assert [record["question"] for record in saved_qa] == ["What is present?", "Where is it?"]
    assert [record["answer_format"] for record in saved_qa] == ["natural", "natural"]
    assert materialized == [("dataset_a", artifact_dir / "instruction_artifacts")]
    assert result["result"]["instruction_artifacts"]["status"] == "ok"
    assert result["import_report"]["instruction_artifacts"]["training_row_count"] == 3


def test_openai_caption_batch_fill_missing_qa_deficit_does_not_duplicate_existing_caption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_qa_deficit"
    artifact_dir.mkdir(parents=True)
    case = {
        "case_id": "image:scene.jpg:full",
        "image_name": "scene.jpg",
        "image_path": str(tmp_path / "scene.jpg"),
        "split": "train",
    }
    (artifact_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": case["case_id"],
                "image_name": "scene.jpg",
                "caption": "A paid caption that should only support QA import.",
                "generated_qa_pairs": [
                    {"question": f"What detail {index} is visible?", "answer": f"Detail {index} is visible."}
                    for index in range(1, 6)
                ],
                "generated_qa_pair_count": 5,
                "generated_qa_target_pair_count": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_verify_target",
        lambda job, dataset_id=None, **kwargs: (
            True,
            {"reason": "ok", "dataset_id": "dataset_a", "case_count": 1, "target_hash": "hash"},
        ),
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_image_completion_counts",
        lambda dataset_id, image_name, split=None: {"base_caption_count": 1, "generated_qa_count": 3},
    )
    monkeypatch.setattr(
        api,
        "_add_caption_impl",
        lambda *args, **kwargs: pytest.fail("fill_missing QA-deficit import must not duplicate an existing base caption"),
    )
    saved_qa: list[dict] = []
    monkeypatch.setattr(
        api,
        "_dataset_caption_add_instruction_records",
        lambda dataset_id, image_name, records, *, split=None: saved_qa.extend(dict(record) for record in records) or len(records),
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_materialize_instruction_artifacts",
        lambda *, dataset_id, payload, artifact_dir: {"status": "ok", "artifact_dir": str(artifact_dir)},
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_qa_deficit",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_123",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "save_text_labels": True,
                "write_policy": "fill_missing",
                "completion_mode": "per_image_totals",
                "target_base_captions_per_image": 1,
                "target_generated_qa_per_image": 8,
                "subcaptions_per_image": 5,
                "test_outputs_count_toward_completion": True,
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_import_job("ocap_qa_deficit", dataset_id="dataset_a")

    assert result["status"] == "imported"
    assert result["import_report"]["saved_captions"] == 0
    assert result["import_report"]["saved_generated_qa_rows"] == 5
    assert len(saved_qa) == 5


def test_openai_caption_batch_incremental_qa_run_does_not_save_caption_when_base_increment_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_incremental_qa"
    artifact_dir.mkdir(parents=True)
    case = {
        "case_id": "image:scene.jpg:full",
        "image_name": "scene.jpg",
        "image_path": str(tmp_path / "scene.jpg"),
        "split": "train",
    }
    (artifact_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": case["case_id"],
                "image_name": "scene.jpg",
                "caption": "A paid caption that should not become a new variant.",
                "generated_qa_pairs": [
                    {"question": "What is visible?", "answer": "A visible object is present."},
                    {"question": "Where is it?", "answer": "It is near the center."},
                ],
                "generated_qa_pair_count": 2,
                "generated_qa_target_pair_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_verify_target",
        lambda job, dataset_id=None, **kwargs: (
            True,
            {"reason": "ok", "dataset_id": "dataset_a", "case_count": 1, "target_hash": "hash"},
        ),
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_image_completion_counts",
        lambda dataset_id, image_name, split=None: {"base_caption_count": 4, "generated_qa_count": 6},
    )
    monkeypatch.setattr(
        api,
        "_add_caption_impl",
        lambda *args, **kwargs: pytest.fail("incremental QA-only request with zero base increment must not save a caption"),
    )
    saved_qa: list[dict] = []
    monkeypatch.setattr(
        api,
        "_dataset_caption_add_instruction_records",
        lambda dataset_id, image_name, records, *, split=None: saved_qa.extend(dict(record) for record in records) or len(records),
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_materialize_instruction_artifacts",
        lambda *, dataset_id, payload, artifact_dir: {"status": "ok", "artifact_dir": str(artifact_dir)},
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_incremental_qa",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_123",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "save_text_labels": True,
                "write_policy": "fill_missing",
                "completion_mode": "incremental",
                "increment_base_captions_per_image": 0,
                "increment_generated_qa_per_image": 2,
                "subcaptions_per_image": 0,
                "test_outputs_count_toward_completion": True,
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_import_job("ocap_incremental_qa", dataset_id="dataset_a")

    assert result["status"] == "imported"
    assert result["import_report"]["saved_captions"] == 0
    assert result["import_report"]["saved_generated_qa_rows"] == 2
    assert len(saved_qa) == 2


def test_openai_caption_batch_required_question_catchup_import_replaces_unretained_qa_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_required_catchup_import"
    artifact_dir.mkdir(parents=True)
    case = {
        "case_id": "image:scene.jpg:full",
        "image_name": "scene.jpg",
        "image_path": str(tmp_path / "scene.jpg"),
        "split": "train",
    }
    (artifact_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": case["case_id"],
                "image_name": "scene.jpg",
                "caption": "A catch-up caption.",
                "generated_qa_pairs": [
                    {"question": "What is present?", "answer": "A building is present."},
                    {"question": "What color is the roof?", "answer": "The roof is gray."},
                ],
                "generated_qa_new_pairs": [
                    {"question": "What color is the roof?", "answer": "The roof is gray."},
                ],
                "generated_qa_pair_count": 2,
                "generated_qa_new_pair_count": 1,
                "generated_qa_existing_pair_count": 1,
                "generated_qa_target_pair_count": 2,
                "openai_batch_catchup": {
                    "source_job_id": "ocap_source",
                    "existing_pair_count": 1,
                    "new_pair_count": 1,
                    "total_target_pair_count": 2,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_verify_target",
        lambda job, dataset_id=None, **kwargs: (
            True,
            {"reason": "ok", "dataset_id": "dataset_a", "case_count": 1, "target_hash": "hash"},
        ),
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_image_completion_counts",
        lambda dataset_id, image_name, split=None: {"base_caption_count": 1, "generated_qa_count": 2},
    )
    monkeypatch.setattr(
        api,
        "_add_caption_impl",
        lambda *args, **kwargs: pytest.fail("required-question catch-up must not duplicate caption text"),
    )
    archive_calls: list[dict] = []

    def fake_archive(dataset_id, image_name, *, keep_questions, limit, job_id, split=None, reason=""):
        archive_calls.append(
            {
                "dataset_id": dataset_id,
                "image_name": image_name,
                "keep_questions": list(keep_questions),
                "limit": limit,
                "job_id": job_id,
                "split": split,
                "reason": reason,
            }
        )
        return 1

    monkeypatch.setattr(api, "_qwen_caption_dataset_soft_archive_generated_qa_not_in_questions", fake_archive)
    saved_qa: list[dict] = []
    monkeypatch.setattr(
        api,
        "_dataset_caption_add_instruction_records",
        lambda dataset_id, image_name, records, *, split=None: saved_qa.extend(dict(record) for record in records) or len(records),
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_materialize_instruction_artifacts",
        lambda *, dataset_id, payload, artifact_dir: {"status": "ok", "artifact_dir": str(artifact_dir)},
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_required_catchup_import",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_123",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "save_text_labels": True,
                "write_policy": "fill_missing",
                "completion_mode": "per_image_totals",
                "target_base_captions_per_image": 1,
                "target_generated_qa_per_image": 2,
                "subcaptions_per_image": 2,
                "instruction_qa_imposed_questions": ["What color is the roof?"],
                "test_outputs_count_toward_completion": True,
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_import_job("ocap_required_catchup_import", dataset_id="dataset_a")

    assert result["status"] == "imported"
    assert result["import_report"]["saved_captions"] == 0
    assert result["import_report"]["saved_generated_qa_rows"] == 1
    assert result["import_report"]["archived_generated_qa_rows"] == 1
    assert archive_calls == [
        {
            "dataset_id": "dataset_a",
            "image_name": "scene.jpg",
            "keep_questions": ["What is present?", "What color is the roof?"],
            "limit": 1,
            "job_id": "ocap_required_catchup_import",
            "split": "train",
            "reason": "openai_batch_catchup_replaced_unretained_qa",
        }
    ]
    assert [record["question"] for record in saved_qa] == ["What color is the roof?"]
    assert saved_qa[0]["lifecycle_status"] == "active"


def test_openai_caption_batch_import_reholds_accepted_row_missing_imposed_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_import_missing_required"
    artifact_dir.mkdir(parents=True)
    case = {
        "case_id": "image:scene.jpg:full",
        "image_name": "scene.jpg",
        "image_path": str(tmp_path / "scene.jpg"),
        "split": "train",
    }
    (artifact_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": case["case_id"],
                "image_name": "scene.jpg",
                "caption": "A legacy accepted caption.",
                "generated_qa_pairs": [
                    {"question": "What is present?", "answer": "A building is present."},
                    {"question": "Where is it?", "answer": "It is near the center."},
                ],
                "generated_qa_pair_count": 2,
                "generated_qa_target_pair_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_verify_target",
        lambda job, dataset_id=None, **kwargs: (
            True,
            {"reason": "ok", "dataset_id": "dataset_a", "case_count": 1, "target_hash": "hash"},
        ),
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_image_completion_counts",
        lambda dataset_id, image_name, split=None: {"generated_qa_count": 0},
    )
    saved_captions = []
    saved_qa = []
    monkeypatch.setattr(
        api,
        "_add_caption_impl",
        lambda dataset_id, image_name, payload, *, allow_active_caption_job_id=None: saved_captions.append((dataset_id, image_name, dict(payload))) or {"status": "saved"},
    )
    monkeypatch.setattr(
        api,
        "_dataset_caption_add_instruction_records",
        lambda dataset_id, image_name, records, *, split=None: saved_qa.extend(dict(record) for record in records) or len(records),
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_import_missing_required",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_123",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "save_text_labels": True,
                "subcaptions_per_image": 2,
                "target_generated_qa_per_image": 2,
                "instruction_qa_imposed_questions": ["What color is the roof?"],
                "test_outputs_count_toward_completion": True,
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_import_job("ocap_import_missing_required", dataset_id="dataset_a")

    assert result["status"] == "collected"
    assert result["import_report"]["status"] == "partial_imported"
    assert result["import_report"]["saved_captions"] == 0
    assert result["import_report"]["saved_generated_qa_rows"] == 0
    assert result["import_report"]["incomplete_rows"] == 1
    assert result["import_report"]["incomplete_row_samples"][0]["failure_reason"] == "required_qa_missing"
    assert result["import_report"]["incomplete_row_samples"][0]["missing_required_questions"] == [
        "What color is the roof?"
    ]
    held = batch_smoke.read_jsonl(artifact_dir / "incomplete_captions.jsonl")
    assert held[0]["case_id"] == case["case_id"]
    assert held[0]["missing_required_questions"] == ["What color is the roof?"]
    assert saved_captions == []
    assert saved_qa == []

    second = api._openai_caption_batch_import_job("ocap_import_missing_required", dataset_id="dataset_a")

    assert second["status"] == "collected"
    assert second["import_report"]["incomplete_rows"] == 1
    assert len(batch_smoke.read_jsonl(artifact_dir / "incomplete_captions.jsonl")) == 1
    assert saved_captions == []
    assert saved_qa == []


def test_openai_caption_batch_partial_import_stays_collected_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_partial_import"
    artifact_dir.mkdir(parents=True)
    case = {
        "case_id": "image:scene.jpg:full",
        "image_name": "scene.jpg",
        "image_path": str(tmp_path / "scene.jpg"),
        "split": "train",
    }
    (artifact_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": case["case_id"],
                "image_name": "scene.jpg",
                "caption": "A complete accepted caption.",
                "generated_qa_pairs": [
                    {"question": "What is present?", "answer": "A building is present."},
                    {"question": "Where is it?", "answer": "It is near the center."},
                ],
                "generated_qa_pair_count": 2,
                "generated_qa_target_pair_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "incomplete_captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image:other.jpg:full",
                "image_name": "other.jpg",
                "caption": "A paid partial caption.",
                "generated_qa_pairs": [{"question": "What is visible?", "answer": "An object is visible."}],
                "generated_qa_pair_count": 1,
                "generated_qa_target_pair_count": 2,
                "failure_reason": "generated_qa_incomplete",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_verify_target",
        lambda job, dataset_id=None, **kwargs: (
            True,
            {"reason": "ok", "dataset_id": "dataset_a", "case_count": 1, "target_hash": "hash"},
        ),
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_image_completion_counts",
        lambda dataset_id, image_name, split=None: {"generated_qa_count": 0},
    )
    monkeypatch.setattr(api, "_qwen_caption_dataset_soft_archive_generated_outputs", lambda *args, **kwargs: {})
    saved_captions: list[tuple[str, str, dict]] = []
    saved_qa: list[dict] = []

    def fake_add_caption(dataset_id, image_name, payload, *, allow_active_caption_job_id=None):
        saved_captions.append((dataset_id, image_name, dict(payload)))
        return {"status": "saved"}

    def fake_add_qa(dataset_id, image_name, records, *, split=None):
        saved_qa.extend(dict(record) for record in records)
        return len(records)

    monkeypatch.setattr(api, "_add_caption_impl", fake_add_caption)
    monkeypatch.setattr(api, "_dataset_caption_add_instruction_records", fake_add_qa)
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_partial_import",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_123",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "save_text_labels": True,
                "subcaptions_per_image": 2,
                "target_generated_qa_per_image": 2,
                "test_outputs_count_toward_completion": True,
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    first = api._openai_caption_batch_import_job("ocap_partial_import", dataset_id="dataset_a")
    second = api._openai_caption_batch_import_job("ocap_partial_import", dataset_id="dataset_a")

    assert first["status"] == "collected"
    assert first["import_report"]["status"] == "partial_imported"
    assert first["import_report"]["saved_captions"] == 1
    assert first["import_report"]["saved_generated_qa_rows"] == 2
    assert first["import_report"]["incomplete_rows"] == 1
    assert second["status"] == "collected"
    assert second["import_report"]["status"] == "partial_imported"
    assert second["import_report"]["saved_captions"] == 0
    assert second["import_report"]["saved_generated_qa_rows"] == 0
    assert second["import_report"]["already_imported_rows"] == 1
    assert len(saved_captions) == 1
    assert len(saved_qa) == 2
    imported_rows = api._openai_caption_batch_read_jsonl(artifact_dir / "imported_rows.jsonl")
    assert len(imported_rows) == 1
    assert imported_rows[0]["case_id"] == case["case_id"]


def test_openai_caption_batch_import_with_missing_result_rows_stays_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_missing_result_import"
    artifact_dir.mkdir(parents=True)
    accepted_case = {
        "case_id": "image:accepted.jpg:full",
        "image_name": "accepted.jpg",
        "image_path": str(tmp_path / "accepted.jpg"),
        "split": "train",
    }
    missing_case = {
        "case_id": "image:missing.jpg:full",
        "image_name": "missing.jpg",
        "image_path": str(tmp_path / "missing.jpg"),
        "split": "train",
    }
    (artifact_dir / "cases.json").write_text(json.dumps([accepted_case, missing_case]), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_cases": 2,
                "caption_rows": 1,
                "failed_cases": 1,
                "missing_result_rows": 1,
                "incomplete_caption_rows": 0,
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "captions.jsonl").write_text(
        json.dumps(
            {
                "case_id": accepted_case["case_id"],
                "image_name": "accepted.jpg",
                "caption": "An accepted caption.",
                "generated_qa_pairs": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "case_id": missing_case["case_id"],
                "image_name": "missing.jpg",
                "final_status": "failed",
                "failure_reason": "missing_batch_result",
                "generated_qa_pair_count": 0,
                "generated_qa_target_pair_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_verify_target",
        lambda job, dataset_id=None, **kwargs: (
            True,
            {"reason": "ok", "dataset_id": "dataset_a", "case_count": 2, "target_hash": "hash"},
        ),
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_image_completion_counts",
        lambda dataset_id, image_name, split=None: {"generated_qa_count": 0},
    )
    saved_captions: list[tuple[str, str, dict]] = []

    def fake_add_caption(dataset_id, image_name, payload, *, allow_active_caption_job_id=None):
        saved_captions.append((dataset_id, image_name, dict(payload)))
        return {"status": "saved"}

    monkeypatch.setattr(api, "_add_caption_impl", fake_add_caption)
    monkeypatch.setattr(api, "_dataset_caption_add_instruction_records", lambda *args, **kwargs: 0)
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_missing_result_import",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_missing_result",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "save_text_labels": True,
                "test_outputs_count_toward_completion": True,
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_import_job("ocap_missing_result_import", dataset_id="dataset_a")

    assert result["status"] == "collected"
    assert result["import_report"]["status"] == "partial_imported"
    assert result["import_report"]["saved_captions"] == 1
    assert result["import_report"]["failed_output_rows"] == 1
    assert result["import_report"]["missing_result_rows"] == 1
    assert "missing result row" in result["message"]
    assert [item[1] for item in saved_captions] == ["accepted.jpg"]


def test_openai_caption_batch_import_is_idempotent_for_imported_single_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_imported_once"
    artifact_dir.mkdir(parents=True)
    monkeypatch.setattr(api, "_add_caption_impl", lambda *args, **kwargs: pytest.fail("must not write captions"))
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_imported_once",
            "kind": "openai_caption_batch",
            "status": "imported",
            "dataset_id": "dataset_a",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
            },
            "import_report": {
                "status": "imported",
                "dataset_id": "dataset_a",
                "saved_captions": 1,
                "saved_generated_qa_rows": 2,
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(artifact_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_import_job("ocap_imported_once", dataset_id="dataset_a")

    assert result["status"] == "imported"
    assert result["import_report"]["saved_captions"] == 1


def test_openai_caption_batch_collection_import_does_not_reimport_imported_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "_openai_caption_batch_api_key", lambda path_value: "test-key")
    monkeypatch.setattr(batch_smoke, "download_outputs", lambda **kwargs: None)
    monkeypatch.setattr(
        batch_smoke,
        "collect_results",
        lambda *, cases, output_dir, target_qa, answer_format="natural", imposed_questions=None: {
            "total_cases": len(cases),
            "caption_rows": len(cases),
            "generated_qa_rows": 0,
        },
    )
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_verify_target",
        lambda job, dataset_id=None, **kwargs: (True, {"reason": "ok", "dataset_id": "dataset_a"}),
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_image_completion_counts",
        lambda dataset_id, image_name, split=None: {"generated_qa_count": 0},
    )
    monkeypatch.setattr(api, "_qwen_caption_dataset_soft_archive_generated_outputs", lambda *args, **kwargs: {})
    saved_captions: list[tuple[str, str, dict]] = []

    def fake_add_caption(dataset_id, image_name, payload, *, allow_active_caption_job_id=None):
        saved_captions.append((dataset_id, image_name, dict(payload)))
        return {"status": "saved"}

    monkeypatch.setattr(api, "_add_caption_impl", fake_add_caption)
    monkeypatch.setattr(api, "_dataset_caption_add_instruction_records", lambda *args, **kwargs: 0)
    materialized: list[tuple[str, Path]] = []

    def fake_materialize_instruction_artifacts(*, dataset_id, payload, artifact_dir):
        materialized.append((dataset_id, Path(artifact_dir)))
        return {
            "status": "ok",
            "artifact_dir": str(artifact_dir),
            "bundle_zip": str(Path(artifact_dir) / "bundle.zip"),
            "report_json": str(Path(artifact_dir) / "caption_instruction_report.json"),
            "training_row_count": 6,
        }

    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_materialize_instruction_artifacts",
        fake_materialize_instruction_artifacts,
    )

    parent_dir = tmp_path / "jobs" / "openai_batches" / "ocap_parent_import"
    imported_dir = tmp_path / "jobs" / "openai_batches" / "ocap_imported_child"
    collected_dir = tmp_path / "jobs" / "openai_batches" / "ocap_collected_child"
    for path in (parent_dir, imported_dir, collected_dir):
        path.mkdir(parents=True)
    case = {"case_id": "image:scene.jpg:full", "image_name": "scene.jpg", "split": "train"}
    (collected_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (collected_dir / "captions.jsonl").write_text(
        json.dumps({"case_id": case["case_id"], "image_name": "scene.jpg", "caption": "Caption.", "generated_qa_pairs": []}) + "\n",
        encoding="utf-8",
    )
    (collected_dir / "batch_status.json").write_text(json.dumps({"response": {"id": "batch_child"}}), encoding="utf-8")
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_imported_child",
            "kind": "openai_caption_batch",
            "status": "imported",
            "dataset_id": "dataset_a",
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "import_report": {"saved_captions": 5, "saved_generated_qa_rows": 6},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(imported_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_collected_child",
            "kind": "openai_caption_batch",
            "status": "collected",
            "dataset_id": "dataset_a",
            "openai_batch_id": "batch_child",
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "save_text_labels": True,
                "subcaptions_per_image": 0,
                "test_outputs_count_toward_completion": True,
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(collected_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_parent_import",
            "kind": "openai_caption_batch_collection",
            "status": "collected",
            "dataset_id": "dataset_a",
            "child_job_ids": ["ocap_imported_child", "ocap_collected_child"],
            "request": {
                "dataset_id": "dataset_a",
                "caption_provider": "openai",
                "openai_service_tier": "batch",
                "instruction_dataset": True,
                "include_generated_qa_in_training": True,
                "save_text_labels": True,
                "subcaptions_per_image": 0,
                "test_outputs_count_toward_completion": True,
            },
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(parent_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_import_job("ocap_parent_import", dataset_id="dataset_a")

    assert result["status"] == "imported"
    assert result["import_report"]["imported_shards"] == 2
    assert result["import_report"]["saved_captions"] == 6
    assert len(saved_captions) == 1
    assert saved_captions[0][0:2] == ("dataset_a", "scene.jpg")
    assert api._openai_caption_batch_read("ocap_imported_child")["import_report"]["saved_captions"] == 5
    assert materialized == [
        ("dataset_a", collected_dir / "instruction_artifacts"),
        ("dataset_a", parent_dir / "instruction_artifacts"),
    ]
    assert result["result"]["instruction_artifacts"]["status"] == "ok"


def test_openai_caption_batch_collection_import_skips_shards_outside_selected_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "_openai_caption_batch_api_key", lambda path_value: "test-key")
    monkeypatch.setattr(batch_smoke, "download_outputs", lambda **kwargs: None)
    monkeypatch.setattr(
        batch_smoke,
        "collect_results",
        lambda *, cases, output_dir, target_qa, answer_format="natural", imposed_questions=None: {
            "total_cases": len(cases),
            "caption_rows": len(cases),
            "generated_qa_rows": 0,
        },
    )
    monkeypatch.setattr(
        api,
        "_openai_caption_batch_current_manifest_rows",
        lambda dataset_id, selected_image_keys=None: {
            "first.jpg": {"image_sha256": "first-image", "label_sha256": "first-label"},
            "second.jpg": {"image_sha256": "second-image", "label_sha256": "second-label"},
        },
    )
    monkeypatch.setattr(
        api,
        "_qwen_caption_dataset_image_completion_counts",
        lambda dataset_id, image_name, split=None: {"generated_qa_count": 0},
    )
    monkeypatch.setattr(api, "_qwen_caption_dataset_soft_archive_generated_outputs", lambda *args, **kwargs: {})
    saved_captions: list[tuple[str, str, dict]] = []

    def fake_add_caption(dataset_id, image_name, payload, *, allow_active_caption_job_id=None):
        saved_captions.append((dataset_id, image_name, dict(payload)))
        return {"status": "saved"}

    monkeypatch.setattr(api, "_add_caption_impl", fake_add_caption)
    monkeypatch.setattr(api, "_dataset_caption_add_instruction_records", lambda *args, **kwargs: 0)

    parent_dir = tmp_path / "jobs" / "openai_batches" / "ocap_parent_subset"
    first_dir = tmp_path / "jobs" / "openai_batches" / "ocap_first_child"
    second_dir = tmp_path / "jobs" / "openai_batches" / "ocap_second_child"
    for path in (parent_dir, first_dir, second_dir):
        path.mkdir(parents=True)
    child_specs = [
        ("ocap_first_child", first_dir, "first.jpg", "first-image", "first-label", "First caption."),
        ("ocap_second_child", second_dir, "second.jpg", "second-image", "second-label", "Second caption."),
    ]
    for child_id, child_dir, image_name, image_sha, label_sha, caption in child_specs:
        case = {
            "case_id": f"image:{image_name}:full",
            "image_key": image_name,
            "image_name": image_name,
            "image_relpath": f"train/{image_name}",
            "image_sha256": image_sha,
            "label_sha256": label_sha,
            "split": "train",
        }
        (child_dir / "target_manifest.json").write_text(
            json.dumps(
                {
                    "format": "tator_openai_caption_batch_target_v1",
                    "dataset_id": "dataset_a",
                    "case_count": 1,
                    "cases": [case],
                    "target_identity": api._openai_caption_batch_identity_from_rows([case]),
                }
            ),
            encoding="utf-8",
        )
        (child_dir / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
        (child_dir / "captions.jsonl").write_text(
            json.dumps({"case_id": case["case_id"], "image_name": image_name, "caption": caption, "generated_qa_pairs": []}) + "\n",
            encoding="utf-8",
        )
        (child_dir / "batch_status.json").write_text(json.dumps({"response": {"id": f"batch_{child_id}"}}), encoding="utf-8")
        api._openai_caption_batch_write(
            {
                "job_id": child_id,
                "kind": "openai_caption_batch",
                "status": "collected",
                "dataset_id": "dataset_a",
                "openai_batch_id": f"batch_{child_id}",
                "request": {
                    "dataset_id": "dataset_a",
                    "caption_provider": "openai",
                    "openai_service_tier": "batch",
                    "save_text_labels": True,
                    "test_outputs_count_toward_completion": True,
                },
                "created_at": api._openai_caption_batch_now(),
                "output_dir": str(child_dir),
                "target_manifest": str(child_dir / "target_manifest.json"),
                "logs": [],
            }
        )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_parent_subset",
            "kind": "openai_caption_batch_collection",
            "status": "collected",
            "dataset_id": "dataset_a",
            "child_job_ids": ["ocap_first_child", "ocap_second_child"],
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(parent_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_import_job(
        "ocap_parent_subset",
        dataset_id="dataset_a",
        selected_image_keys=["first.jpg"],
    )

    assert result["import_report"]["imported_shards"] == 1
    assert result["import_report"]["skipped_shards"] == 1
    assert result["import_report"]["blocked_shards"] == 0
    assert [report["status"] for report in result["import_report"]["child_reports"]] == ["imported", "skipped"]
    assert [item[1] for item in saved_captions] == ["first.jpg"]


def test_openai_caption_batch_collection_import_reports_no_ready_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    parent_dir = tmp_path / "jobs" / "openai_batches" / "ocap_parent_waiting"
    child_dir = tmp_path / "jobs" / "openai_batches" / "ocap_child_waiting"
    parent_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_child_waiting",
            "kind": "openai_caption_batch",
            "status": "in_progress",
            "dataset_id": "dataset_a",
            "case_count": 1,
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(child_dir),
            "logs": [],
        }
    )
    api._openai_caption_batch_write(
        {
            "job_id": "ocap_parent_waiting",
            "kind": "openai_caption_batch_collection",
            "status": "in_progress",
            "dataset_id": "dataset_a",
            "child_job_ids": ["ocap_child_waiting"],
            "request": {"dataset_id": "dataset_a", "caption_provider": "openai", "openai_service_tier": "batch"},
            "created_at": api._openai_caption_batch_now(),
            "output_dir": str(parent_dir),
            "logs": [],
        }
    )

    result = api._openai_caption_batch_import_job("ocap_parent_waiting", dataset_id="dataset_a")

    assert result["status"] == "in_progress"
    assert result["message"].startswith("No OpenAI Batch shard outputs are ready")
    assert result["import_report"]["ready_shards"] == 0
    assert result["import_report"]["imported_shards"] == 0
    assert result["import_report"]["saved_captions"] == 0


def test_openai_caption_batch_cost_summary_uses_collected_response_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_cost"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case_1",
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "total_tokens": 1200,
                    "input_tokens_details": {"cached_tokens": 100},
                    "output_tokens_details": {"reasoning_tokens": 25},
                },
            }
        )
        + "\n"
        + json.dumps({"case_id": "case_2", "usage": {"input_tokens": 3000, "output_tokens": 400}})
        + "\n",
        encoding="utf-8",
    )
    job = {
        "job_id": "ocap_cost",
        "kind": "openai_caption_batch",
        "status": "collected",
        "dataset_id": "dataset_a",
        "request": {
            "dataset_id": "dataset_a",
            "caption_provider": "openai",
            "openai_service_tier": "batch",
            "openai_model": "gpt-5.5",
            "openai_image_detail": "original",
            "instruction_dataset": True,
            "subcaptions_per_image": 8,
        },
        "openai": {"model": "gpt-5.5", "image_detail": "original", "qa_count": 8, "max_boxes": 120, "max_output_tokens": 3200},
        "case_count": 2,
        "output_dir": str(artifact_dir),
        "result": {"batch": {"id": "batch_123"}},
    }

    summary = api._openai_caption_batch_cost_summary(job)

    assert summary["estimate"]["available"] is True
    assert summary["actual"]["available"] is True
    assert summary["actual"]["source"] == "collected_response_rows"
    assert summary["actual"]["usage"]["input_tokens"] == 4000
    assert summary["actual"]["usage"]["output_tokens"] == 600
    assert summary["actual"]["usage"]["cached_input_tokens"] == 100
    assert summary["actual"]["usage"]["reasoning_tokens"] == 25
    assert summary["actual"]["cost_usd"] == 0.019


def test_openai_caption_batch_cost_summary_uses_top_level_batch_usage_when_result_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_top_level_cost"
    artifact_dir.mkdir(parents=True)
    job = {
        "job_id": "ocap_top_level_cost",
        "kind": "openai_caption_batch",
        "status": "collected",
        "dataset_id": "dataset_a",
        "request": {
            "dataset_id": "dataset_a",
            "caption_provider": "openai",
            "openai_service_tier": "batch",
            "openai_model": "gpt-5.5",
            "openai_image_detail": "original",
            "instruction_dataset": True,
            "subcaptions_per_image": 8,
        },
        "openai": {"model": "gpt-5.5", "image_detail": "original", "qa_count": 8, "max_boxes": 120, "max_output_tokens": 3200},
        "case_count": 2,
        "output_dir": str(artifact_dir),
        "batch": {
            "id": "batch_123",
            "usage": {
                "input_tokens": 10000,
                "output_tokens": 2000,
                "total_tokens": 12000,
                "output_tokens_details": {"reasoning_tokens": 1500},
            },
        },
        "result": {},
    }

    summary = api._openai_caption_batch_cost_summary(job)

    assert summary["actual"]["available"] is True
    assert summary["actual"]["source"] == "batch_retrieve_usage"
    assert summary["actual"]["usage"]["input_tokens"] == 10000
    assert summary["actual"]["usage"]["output_tokens"] == 2000
    assert summary["actual"]["usage"]["reasoning_tokens"] == 1500
    assert summary["actual"]["cost_usd"] == 0.055


def test_openai_caption_batch_collection_cost_summary_recomputes_stale_child_actual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_stale_child_cost"
    artifact_dir.mkdir(parents=True)
    child = {
        "job_id": "ocap_stale_child_cost",
        "kind": "openai_caption_batch",
        "status": "collected",
        "dataset_id": "dataset_a",
        "request": {
            "dataset_id": "dataset_a",
            "caption_provider": "openai",
            "openai_service_tier": "batch",
            "openai_model": "gpt-5.5",
            "openai_image_detail": "original",
            "instruction_dataset": True,
            "subcaptions_per_image": 8,
        },
        "openai": {"model": "gpt-5.5", "image_detail": "original", "qa_count": 8, "max_boxes": 120, "max_output_tokens": 3200},
        "case_count": 1,
        "output_dir": str(artifact_dir),
        "result": {"batch": {"id": "batch_123", "usage": {"input_tokens": 1000, "output_tokens": 100}}},
        "cost_summary": {
            "estimate": {"available": True, "cost_usd": 1.0, "usage": {"input_tokens": 1, "output_tokens": 1}},
            "actual": {"available": False, "cost_usd": 0.0, "usage": {"input_tokens": 0, "output_tokens": 0}},
        },
    }

    summary = api._openai_caption_batch_collection_cost_summary([child])

    assert summary["actual"]["available"] is True
    assert summary["actual"]["shards"] == 1
    assert summary["actual"]["usage"]["input_tokens"] == 1000
    assert summary["actual"]["usage"]["output_tokens"] == 100
    assert summary["actual"]["cost_usd"] == 0.004


def test_openai_caption_batch_cost_estimate_uses_per_case_qa_deficits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_estimate"
    artifact_dir.mkdir(parents=True)
    cases_path = artifact_dir / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {"case_id": "case_1", "_generated_qa_request_count": 1},
                {"case_id": "case_2", "_generated_qa_request_count": 3},
            ]
        ),
        encoding="utf-8",
    )
    job = {
        "job_id": "ocap_estimate",
        "kind": "openai_caption_batch",
        "status": "submitted",
        "dataset_id": "dataset_a",
        "request": {
            "dataset_id": "dataset_a",
            "caption_provider": "openai",
            "openai_service_tier": "batch",
            "openai_model": "gpt-5.5",
            "openai_image_detail": "original",
            "instruction_dataset": True,
            "subcaptions_per_image": 8,
        },
        "openai": {
            "model": "gpt-5.5",
            "image_detail": "original",
            "qa_count": 8,
            "max_boxes": 120,
            "max_output_tokens": 3200,
        },
        "case_count": 99,
        "cases_json": str(cases_path),
        "output_dir": str(artifact_dir),
        "result": {},
    }

    summary = api._openai_caption_batch_cost_summary(job)

    assert summary["estimate"]["available"] is True
    assert summary["estimate"]["row_count"] == 2
    assert summary["estimate"]["usage"]["input_tokens"] == 17640
    assert summary["estimate"]["usage"]["output_tokens"] == 2520
    assert summary["estimate"]["granular"]["unit_cost_usd"]["caption"] > 0
    assert summary["estimate"]["granular"]["unit_cost_usd"]["qa_pair"] > 0
    assert summary["estimate"]["assumptions"]["case_qa_count_min"] == 1
    assert summary["estimate"]["assumptions"]["case_qa_count_max"] == 3
    assert summary["estimate"]["assumptions"]["case_qa_count_total"] == 4
    assert summary["estimate"]["assumptions"]["case_qa_counts_source"] == "cases_json"


def test_openai_caption_cost_calibration_exposes_caption_and_qa_unit_costs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    monkeypatch.setattr(api, "QWEN_CAPTION_DATASET_JOB_ROOT", tmp_path / "jobs")
    api.OPENAI_CAPTION_COST_CALIBRATION_CACHE = {"root": "", "ts": 0.0, "samples": []}
    artifact_dir = tmp_path / "jobs" / "openai_batches" / "ocap_calibration"
    artifact_dir.mkdir(parents=True)
    job = {
        "job_id": "ocap_calibration",
        "kind": "openai_caption_batch",
        "status": "collected",
        "dataset_id": "dataset_a",
        "request": {
            "dataset_id": "dataset_a",
            "caption_provider": "openai",
            "openai_service_tier": "batch",
            "openai_model": "gpt-5.5",
            "openai_image_detail": "original",
            "openai_reasoning_effort": "high",
            "instruction_dataset": True,
            "subcaptions_per_image": 8,
        },
        "openai": {
            "model": "gpt-5.5",
            "image_detail": "original",
            "reasoning_effort": "high",
            "qa_count": 8,
            "max_boxes": 120,
            "max_output_tokens": 2500,
        },
        "case_count": 2,
        "output_dir": str(artifact_dir),
    }
    (artifact_dir / "openai_batch_job.json").write_text(json.dumps(job), encoding="utf-8")
    rows = [
        {
            "case_id": "case_1",
            "generated_qa_target_pair_count": 8,
            "generated_qa_pair_count": 8,
            "usage": {"input_tokens": 3921, "output_tokens": 2103, "total_tokens": 6024},
        },
        {
            "case_id": "case_2",
            "generated_qa_target_pair_count": 8,
            "generated_qa_pair_count": 8,
            "usage": {"input_tokens": 3921, "output_tokens": 2103, "total_tokens": 6024},
        },
    ]
    (artifact_dir / "captions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    metadata = api._openai_caption_provider_metadata()
    sample = metadata["cost_model"]["calibration"][0]

    assert sample["row_count"] == 2
    assert sample["average_qa_per_row"] == 8
    assert sample["cost_usd"]["caption"] > 0
    assert sample["cost_usd"]["qa_pair"] > 0
    assert sample["cost_usd"]["observed_caption_plus_average_qa"] == pytest.approx(
        sample["cost_usd"]["observed_average_row"],
        abs=0.000001,
    )

    estimate = api._openai_caption_granular_cost_estimate(
        image_count=100,
        caption_count_per_image=10,
        qa_count_per_image=1000,
        model="gpt-5.5",
        service_tier="batch",
        image_detail="original",
        reasoning_effort="high",
        max_boxes=120,
        calibration=sample,
    )

    assert estimate["source"] == "calibrated_linear_estimate"
    assert estimate["unit_cost_usd"]["caption"] == pytest.approx(sample["cost_usd"]["caption"], abs=0.0000001)
    assert estimate["unit_cost_usd"]["qa_pair"] == pytest.approx(sample["cost_usd"]["qa_pair"], abs=0.0000001)
    assert estimate["caption_count_per_image"] == 10
    assert estimate["qa_count_per_image"] == 1000
    assert estimate["cost_usd"] == pytest.approx(
        100 * ((10 * sample["cost_usd"]["caption"]) + (1000 * sample["cost_usd"]["qa_pair"])),
        abs=0.001,
    )


def test_openai_standard_dataset_summary_accumulates_response_usage() -> None:
    import localinferenceapi as api

    summary = {
        "processed": 1,
        "openai_model": "gpt-5.5",
        "openai_service_tier": "standard",
    }
    row = {
        "qwen_caption_io": {
            "usage_rows": 2,
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
                "input_tokens_details": {"cached_tokens": 50},
                "output_tokens_details": {"reasoning_tokens": 25},
            },
        }
    }

    api._qwen_caption_dataset_add_openai_usage(summary, row)

    assert summary["openai_usage_call_rows"] == 2
    assert summary["openai_usage"]["input_tokens"] == 1000
    assert summary["openai_usage"]["output_tokens"] == 100
    assert summary["openai_usage"]["cached_input_tokens"] == 50
    assert summary["openai_usage"]["reasoning_tokens"] == 25
    assert summary["openai_cost_summary"]["source"] == "standard_runner_response_usage"
    assert summary["openai_cost_summary"]["service_tier"] == "standard"
    assert summary["openai_cost_summary"]["available"] is True


def test_openai_caption_spend_summary_groups_costs_by_api_key_model_and_batch() -> None:
    import localinferenceapi as api

    response = {
        "data": [
            {
                "start_time": 10,
                "end_time": 20,
                "results": [
                    {
                        "amount": {"value": 1.25, "currency": "usd"},
                        "api_key_id": "key_a",
                        "model": "gpt-5.5",
                        "batch": True,
                    },
                    {
                        "amount": {"value": 0.75, "currency": "usd"},
                        "api_key_id": "key_a",
                        "model": "gpt-5.5",
                        "batch": True,
                    },
                    {
                        "amount": {"value": 0.5, "currency": "usd"},
                        "api_key_id": "key_b",
                        "model": "gpt-5.4-mini",
                        "batch": False,
                    },
                ],
            }
        ]
    }

    summary = api._openai_caption_summarize_costs_response(response, ["api_key_id", "model", "batch"])

    assert summary["total_cost_usd"] == 2.5
    assert summary["bucket_count"] == 1
    assert summary["result_count"] == 3
    assert summary["groups"][0]["cost_usd"] == 2.0
    assert summary["groups"][0]["group"] == {"api_key_id": "key_a", "model": "gpt-5.5", "batch": "True"}
    assert summary["groups"][1]["cost_usd"] == 0.5


def test_openai_caption_fetch_spend_uses_admin_key_and_costs_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import localinferenceapi as api

    admin_key = tmp_path / "admin.key"
    admin_key.write_text("admin-test-key\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs):
        captured.update(kwargs)
        return (
            {
                "data": [
                    {
                        "start_time": 100,
                        "end_time": 200,
                        "results": [
                            {
                                "amount": {"value": 1.5, "currency": "usd"},
                                "api_key_id": "key_a",
                                "model": "gpt-5.5",
                                "batch": True,
                            }
                        ],
                    }
                ]
            },
            {"x-request-id": "req_costs"},
        )

    monkeypatch.setattr(batch_smoke, "request_json", fake_request_json)

    result = api._openai_caption_fetch_spend(
        {
            "admin_key_path": str(admin_key),
            "start_time": 100,
            "end_time": 200,
            "group_by": ["api_key_id", "model", "batch"],
        }
    )

    assert captured["key"] == "admin-test-key"
    assert captured["method"] == "GET"
    path = str(captured["path"])
    assert path.startswith("/organization/costs?")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
    assert query["group_by"] == ["api_key_id", "model", "batch"]
    assert query["start_time"] == ["100"]
    assert query["end_time"] == ["200"]
    assert result["total_cost_usd"] == 1.5
    assert result["admin_key_source"] == "file:admin.key"
