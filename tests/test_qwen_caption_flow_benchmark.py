from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace
from pathlib import Path

from PIL import Image
import pytest

from tools import run_qwen_caption_flow_benchmark as bench


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(120, 130, 140)).save(path)


def _parent_args(dataset: Path, output_dir: Path, **overrides):
    data = {
        "dataset_root": dataset,
        "output_dir": output_dir,
        "cases_json": None,
        "request_json": None,
        "all_images": True,
        "caption_mode": "full",
        "windowed_full_image_strategy": "visual",
        "sample_size": 0,
        "sample_seed": 13,
        "case": [],
        "limit": 0,
        "resume": False,
        "record_resume_skips": False,
        "skip_existing_captions": False,
        "attempts": 1,
        "cooldown_after_crash": 0,
        "cooldown_after_success": 0,
        "cooldown_backoff_multiplier": bench.DEFAULT_COOLDOWN_BACKOFF_MULTIPLIER,
        "max_cooldown_after_crash": bench.DEFAULT_MAX_COOLDOWN_AFTER_CRASH,
        "max_failures": 0,
        "continue_on_quality_failures": False,
        "save_dataset_text_labels": False,
        "timeout": 30,
        "heartbeat_interval": 0,
        "artifact_lock_timeout": 0,
        "artifact_lock_stale_seconds": 3600,
        "artifact_lock_poll_seconds": 0.01,
        "max_artifact_log_bytes": bench.DEFAULT_ARTIFACT_LOG_BYTES,
        "summary_row_limit": bench.DEFAULT_SUMMARY_ROW_LIMIT,
        "instruction_dataset": False,
        "subcaptions_per_image": 0,
        "instruction_max_new_tokens": None,
        "include_source_annotations_in_generator_context": True,
        "strict_grounding": True,
        "qa_mix": "balanced",
        "answer_format": "natural",
        "model_id": "model",
        "refinement_model_id": "same",
        "fallback_model_id": "auto",
        "loop_recovery": "safe_retry_fallback",
        "max_boxes": 0,
        "max_new_tokens": None,
        "final_sentences": 8,
        "window_size": 672,
        "window_overlap": 0.1,
        "mlx_max_image_side": 512,
        "retry_image_side_scale": bench.DEFAULT_RETRY_IMAGE_SIDE_SCALE,
        "min_retry_image_side": bench.DEFAULT_MIN_RETRY_IMAGE_SIDE,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
        "prompt": "Describe.",
        "preview_only": False,
        "use_sampling": False,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_discover_items_includes_images_without_label_files(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Boat\nBuilding\n")
    _write_image(dataset / "train" / "images" / "has_label.jpg")
    _write_image(dataset / "train" / "images" / "missing_label.jpg")
    label_dir = dataset / "train" / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "has_label.txt").write_text("0 0.5 0.5 0.25 0.25\n")

    items = bench.discover_items(dataset)

    by_stem = {item.stem: item for item in items}
    assert sorted(by_stem) == ["has_label", "missing_label"]
    assert by_stem["has_label"].label_count == 1
    assert by_stem["has_label"].class_counts == {"Boat": 1}
    assert by_stem["missing_label"].label_count == 0
    assert by_stem["missing_label"].class_counts == {}
    assert by_stem["missing_label"].label_path == label_dir / "missing_label.txt"


def test_summarize_qwen_caption_io_events_counts_stream_guard(tmp_path: Path) -> None:
    path = tmp_path / "qwen_caption_io.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"event": "input"}),
                json.dumps({"event": "stream_loop_detected"}),
                json.dumps({"event": "loop_trim"}),
                json.dumps({"event": "prompt_budget"}),
                "not json",
                json.dumps({"event": "stream_loop_detected"}),
            ]
        )
    )

    summary = bench.summarize_qwen_caption_io_events(path)

    assert summary["rows"] == 5
    assert summary["invalid_rows"] == 1
    assert summary["event_counts"]["stream_loop_detected"] == 2
    assert summary["stream_loop_detected_events"] == 2
    assert summary["loop_trim_events"] == 1
    assert summary["loop_guard_events"] == 3
    assert summary["prompt_budget_events"] == 1


def test_retry_image_side_profile_downshifts_later_attempts() -> None:
    assert bench.retry_mlx_image_side(512, attempt=1) == 512
    assert bench.retry_mlx_image_side(512, attempt=2) == 384
    assert bench.retry_mlx_image_side(512, attempt=3) == 288
    assert bench.retry_mlx_image_side(
        512,
        attempt=2,
        previous_failure_kind="signal_exit",
    ) == 256
    assert bench.retry_mlx_image_side(
        512,
        attempt=2,
        scale=1.0,
        previous_failure_kind="signal_exit",
    ) == 512
    assert bench.retry_mlx_image_side(512, attempt=4, min_image_side=320) == 320
    assert bench.retry_mlx_image_side(512, attempt=2, scale=1.0) == 512


def test_instruction_qa_normalizer_limits_rows_and_rejects_duplicates() -> None:
    pairs, rejections = bench._normalize_generated_qa_pairs(
        {
            "qa_pairs": [
                {"question": "What is next to the dock?", "answer": "A boat is next to the dock."},
                {"question": "What is next to the dock?", "answer": "Duplicate question."},
                {"question": "", "answer": "Missing question."},
                "bad",
                {"question": "How is the scene viewed?", "answer": "It is viewed from above."},
                {"question": "This should be clipped", "answer": "Too many."},
            ]
        },
        requested=2,
        case={"case_id": "case-1"},
    )

    assert [pair["question"] for pair in pairs] == [
        "What is next to the dock?",
        "How is the scene viewed?",
    ]
    assert [pair["qa_id"] for pair in pairs] == [
        "case-1__generated_qa_0001",
        "case-1__generated_qa_0002",
    ]
    assert pairs[0]["answer_source"] == "vlm_generated"
    assert pairs[0]["metadata"]["generator"] == "qwen_caption_instruction_pass"
    assert pairs[0]["validation"]["status"] == "candidate"
    assert [item["reason"] for item in rejections] == [
        "duplicate_question",
        "missing_question_or_answer",
        "pair_not_object",
    ]


def test_instruction_qa_normalizer_serializes_json_answers_and_rejects_invalid_json() -> None:
    pairs, rejections = bench._normalize_generated_qa_pairs(
        {
            "qa_pairs": [
                {
                    "question": "What is the general setting?",
                    "answer": {"answer": "A waterfront scene."},
                    "answer_format": "json",
                },
                {
                    "question": "What is invalid?",
                    "answer": "not json",
                    "answer_format": "json",
                },
            ]
        },
        requested=2,
        case={"case_id": "case-json"},
        answer_format="json",
    )

    assert pairs[0]["answer"] == '{"answer":"A waterfront scene."}'
    assert pairs[0]["answer_format"] == "json"
    assert [item["reason"] for item in rejections] == ["invalid_json_answer"]


def test_instruction_qa_token_budget_scales_and_respects_override() -> None:
    assert bench._instruction_qa_max_new_tokens(
        SimpleNamespace(instruction_max_new_tokens=None),
        requested=8,
    ) == 1792
    assert bench._instruction_qa_max_new_tokens(
        SimpleNamespace(instruction_max_new_tokens=64),
        requested=8,
    ) == 128
    assert bench._instruction_qa_max_new_tokens(
        SimpleNamespace(instruction_max_new_tokens=999999),
        requested=8,
    ) == 8192


def test_worker_qwen_caption_io_artifacts_follow_worker_run_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(bench, "REPO_ROOT", repo_root)
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    bench.append_jsonl(attempt_dir / bench.WORKER_PROGRESS_JSONL, {"run_id": "run-a", "seq": 1})
    bench.append_jsonl(attempt_dir / bench.WORKER_PROGRESS_JSONL, {"run_id": "run-b", "seq": 2})
    bench.append_jsonl(attempt_dir / bench.WORKER_PROGRESS_JSONL, {"run_id": "run-a", "seq": 3})
    (attempt_dir / bench.WORKER_PROGRESS_JSON).write_text(
        json.dumps({"run_id": "run-b", "seq": 4})
    )
    run_log_dir = repo_root / "logs" / "qwen_caption_io"
    run_log_dir.mkdir(parents=True)
    (run_log_dir / "run-a.jsonl").write_text(
        json.dumps({"event": "prompt_budget", "prompt_tokens": 100})
        + "\n"
        + json.dumps({"event": "output", "output_text": "a" * 80})
        + "\n"
    )
    (run_log_dir / "run-b.jsonl").write_text(
        json.dumps({"event": "prompt_budget", "prompt_tokens": 200}) + "\n"
    )
    (run_log_dir / "run-a.log").write_text("run a log\n")
    (run_log_dir / "run-b.log").write_text("run b log\n")

    artifacts, errors = bench.copy_worker_qwen_caption_io_artifacts(
        attempt_dir,
        max_bytes=120,
    )

    assert errors == []
    assert bench.collect_worker_progress_run_ids(attempt_dir) == ["run-a", "run-b"]
    assert artifacts["qwen_caption_io_jsonl"]["source"] == "qwen_caption_io_per_run"
    assert artifacts["qwen_caption_io_jsonl"]["source_run_ids"] == ["run-a", "run-b"]
    assert artifacts["qwen_caption_io_jsonl"]["truncated"] is True
    summary = bench.read_qwen_caption_io_summary(attempt_dir)
    assert summary["source"] == "qwen_caption_io_per_run"
    assert summary["rows"] == 3
    assert summary["prompt_budget_events"] == 2
    assert summary["max_prompt_tokens"] == 200
    assert summary["event_counts"]["output"] == 1


def test_worker_qwen_caption_io_artifacts_reject_unbound_latest_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(bench, "REPO_ROOT", repo_root)
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    latest_dir = repo_root / "logs"
    latest_dir.mkdir(parents=True)
    (latest_dir / "qwen_caption_io_latest.jsonl").write_text(
        json.dumps({"event": "prompt_budget", "run_id": "stale-run", "prompt_tokens": 999})
        + "\n"
    )
    (latest_dir / "qwen_caption_io_latest.log").write_text("stale global trace\n")

    artifacts, errors = bench.copy_worker_qwen_caption_io_artifacts(
        attempt_dir,
        max_bytes=120,
    )

    assert errors == []
    assert "qwen_caption_io_jsonl" not in artifacts
    assert "qwen_caption_io_log" not in artifacts
    assert not (attempt_dir / "qwen_caption_io.jsonl").exists()
    assert not (attempt_dir / "qwen_caption_io.log").exists()
    summary = bench.read_qwen_caption_io_summary(attempt_dir)
    assert summary["source"] == "worker_progress_run_ids"
    assert summary["missing_trace"] is True
    assert summary["missing_trace_reason"] == "worker progress did not expose a qwen caption run id"
    assert summary["fallback_skipped"] == "qwen_caption_io_latest"
    assert summary["rows"] == 0
    assert summary["prompt_budget_events"] == 0
    assert summary["max_prompt_tokens"] == 0


def test_worker_qwen_caption_io_artifacts_report_missing_per_run_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(bench, "REPO_ROOT", repo_root)
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    bench.append_jsonl(attempt_dir / bench.WORKER_PROGRESS_JSONL, {"run_id": "run-missing"})
    (repo_root / "logs").mkdir(parents=True)
    (repo_root / "logs" / "qwen_caption_io_latest.jsonl").write_text(
        json.dumps({"event": "prompt_budget", "run_id": "other-run", "prompt_tokens": 999})
        + "\n"
    )

    artifacts, errors = bench.copy_worker_qwen_caption_io_artifacts(
        attempt_dir,
        max_bytes=120,
    )

    assert errors == []
    assert "qwen_caption_io_jsonl" not in artifacts
    summary = bench.read_qwen_caption_io_summary(attempt_dir)
    assert summary["source_run_ids"] == ["run-missing"]
    assert summary["missing_trace"] is True
    assert (
        summary["missing_trace_reason"]
        == "worker qwen caption run ids had no matching per-run trace files"
    )
    assert summary["source_records"][0]["run_id"] == "run-missing"
    assert artifacts["qwen_caption_io_sources"]["source"] == "qwen_caption_io_sources"


def test_worker_progress_snapshot_carries_prompt_stack_and_io_events() -> None:
    snapshot = {
        "active": True,
        "run_id": "caption-run",
        "phase": "generate",
        "phase_label": "Captioning",
        "progress": 0.5,
        "message": "Generating response tokens",
        "step_id": "caption",
        "step_index": 3,
        "step_total": 4,
        "step_label": "Compose full-image caption",
        "step_plan": [
            {"id": "prepare", "label": "Prepare image and prompts"},
            {"id": "prompt_stack", "label": "Build prompt stack"},
            {"id": "caption", "label": "Compose full-image caption"},
        ],
        "token_preview": "The scene contains a vehicle.",
        "live_output": "The scene contains a vehicle.",
        "io_events": [
            {
                "event": "input",
                "kind": "prompt",
                "title": "input - Compose full-image caption",
                "text": "rendered prompt stack",
            },
            {
                "event": "output",
                "kind": "output",
                "title": "output - Compose full-image caption",
                "text": "The scene contains a vehicle.",
            },
        ],
    }

    progress = bench.worker_progress_snapshot(snapshot, seq=7)

    assert progress["seq"] == 7
    assert progress["step_plan"][1]["label"] == "Build prompt stack"
    assert progress["io_event_count"] == 2
    assert [event["kind"] for event in progress["io_events"]] == ["prompt", "output"]
    assert progress["io_events"][-1]["text"] == "The scene contains a vehicle."
    assert progress["last_io_event"]["text_tail"] == "The scene contains a vehicle."


def test_all_image_cases_have_stable_resume_keys(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "a.jpg")
    _write_image(dataset / "images" / "b.jpg")
    items = bench.discover_items(dataset)

    cases = bench.select_all_image_cases(items, caption_mode="windowed")

    assert [case["caption_mode"] for case in cases] == ["windowed", "windowed"]
    assert [bench.case_key(case) for case in cases] == [
        "image:a:windowed",
        "image:b:windowed",
    ]


def test_sample_cases_keeps_representative_stress_cases_before_random_fill() -> None:
    cases = [
        {
            "case_id": "image:empty:full",
            "name": "image_000001",
            "stem": "empty",
            "label_count": 0,
            "class_counts": {},
            "caption_mode": "full",
        },
        {
            "case_id": "image:sparse:full",
            "name": "image_000002",
            "stem": "sparse",
            "label_count": 1,
            "class_counts": {"Boat": 1},
            "caption_mode": "full",
        },
        {
            "case_id": "image:diverse:full",
            "name": "image_000003",
            "stem": "diverse",
            "label_count": 30,
            "class_counts": {"Boat": 8, "Building": 8, "Person": 7, "Vehicle": 7},
            "caption_mode": "full",
        },
        {
            "case_id": "image:dominant:full",
            "name": "image_000004",
            "stem": "dominant",
            "label_count": 90,
            "class_counts": {"Boat": 90},
            "caption_mode": "full",
        },
        {
            "case_id": "image:dense:full",
            "name": "image_000005",
            "stem": "dense",
            "label_count": 120,
            "class_counts": {"Building": 70, "Boat": 50},
            "caption_mode": "full",
        },
        {
            "case_id": "image:ordinary:full",
            "name": "image_000006",
            "stem": "ordinary",
            "label_count": 12,
            "class_counts": {"Building": 12},
            "caption_mode": "full",
        },
    ]

    sampled, meta = bench.sample_cases_with_meta(cases, sample_size=3, sample_seed=99)

    assert [bench.case_key(case) for case in sampled] == [
        "image:dense:full",
        "image:diverse:full",
        "image:dominant:full",
    ]
    assert meta["strategy"] == bench.SAMPLE_STRATEGY_STRESS_PLUS_RANDOM
    assert meta["source_cases"] == 6
    assert meta["selected_cases"] == 3
    assert meta["stress_case_keys"] == [
        "image:dense:full",
        "image:diverse:full",
        "image:dominant:full",
    ]
    assert meta["random_fill_case_keys"] == []


def test_sample_cases_fills_after_stress_cases_deterministically() -> None:
    cases = [
        {
            "case_id": f"image:{index}:full",
            "name": f"image_{index:06d}",
            "stem": str(index),
            "label_count": index,
            "class_counts": {"Building": index},
            "caption_mode": "full",
        }
        for index in range(1, 9)
    ]

    first, first_meta = bench.sample_cases_with_meta(cases, sample_size=6, sample_seed=7)
    second, second_meta = bench.sample_cases_with_meta(cases, sample_size=6, sample_seed=7)

    assert [bench.case_key(case) for case in first] == [bench.case_key(case) for case in second]
    assert first_meta == second_meta
    assert first_meta["stress_case_keys"]
    assert first_meta["random_fill_case_keys"]
    assert len({bench.case_key(case) for case in first}) == 6


def test_parent_waits_between_successful_cases_when_configured(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "a.jpg")
    _write_image(dataset / "images" / "b.jpg")
    output_dir = tmp_path / "run"
    sleeps = []
    heartbeats = []

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains 1 Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(bench.time, "sleep", lambda seconds: sleeps.append(seconds))
    original_write_heartbeat = bench.write_heartbeat

    def fake_write_heartbeat(root: Path, payload):
        heartbeats.append(dict(payload))
        original_write_heartbeat(root, payload)

    monkeypatch.setattr(bench, "write_heartbeat", fake_write_heartbeat)

    assert bench.run_parent(_parent_args(dataset, output_dir, cooldown_after_success=2.5)) == 0

    assert sleeps == [2.5]
    cooldown_rows = [row for row in heartbeats if row.get("phase") == "case_success_cooldown"]
    assert len(cooldown_rows) == 1
    assert cooldown_rows[0]["case_id"] == "image:a:full"
    assert cooldown_rows[0]["next_case_index"] == 2
    assert cooldown_rows[0]["cooldown_seconds"] == 2.5


def test_parent_manifest_records_stress_sample_selection(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Boat\nBuilding\nPerson\nVehicle\n")
    label_dir = dataset / "train" / "labels"
    for stem in ["empty", "sparse", "diverse", "dominant", "dense"]:
        _write_image(dataset / "train" / "images" / f"{stem}.jpg")
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "empty.txt").write_text("")
    (label_dir / "sparse.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (label_dir / "diverse.txt").write_text(
        "".join(f"{index % 4} 0.5 0.5 0.1 0.1\n" for index in range(20))
    )
    (label_dir / "dominant.txt").write_text("0 0.5 0.5 0.1 0.1\n" * 40)
    (label_dir / "dense.txt").write_text(("1 0.5 0.5 0.1 0.1\n" * 45) + ("0 0.5 0.5 0.1 0.1\n" * 15))

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    output_dir = tmp_path / "run"
    assert bench.run_parent(_parent_args(dataset, output_dir, sample_size=2, sample_seed=123)) == 0

    manifest = json.loads((output_dir / "manifest.json").read_text())
    selected_keys = [bench.case_key(case) for case in manifest["cases"]]
    assert selected_keys == ["image:dense:full", "image:diverse:full"]
    assert manifest["sample_selection"]["strategy"] == bench.SAMPLE_STRATEGY_STRESS_PLUS_RANDOM
    assert manifest["sample_selection"]["source_cases"] == 5
    assert manifest["sample_selection"]["selected_cases"] == 2
    assert manifest["sample_selection"]["stress_case_keys"] == selected_keys


def test_dataset_text_label_path_matches_split_and_flat_layouts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"

    assert bench.dataset_text_label_path(
        dataset,
        dataset / "train" / "images" / "nested" / "frame.jpg",
    ) == dataset / "train" / "text_labels" / "nested" / "frame.txt"

    assert bench.dataset_text_label_path(
        dataset,
        dataset / "images" / "frame.jpg",
    ) == dataset / "text_labels" / "frame.txt"


def test_results_jsonl_resume_uses_latest_row(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    bench.append_jsonl(
        results_path,
        {
            "case_id": "image:a:full",
            "exit_code": 1,
            "status": "exception",
            "quality_failures": [],
        },
    )
    bench.append_jsonl(
        results_path,
        {
            "case_id": "image:a:full",
            "exit_code": 0,
            "status": "ok",
            "quality_failures": [],
        },
    )

    latest = bench.load_latest_rows(results_path)

    assert latest["image:a:full"]["status"] == "ok"
    assert bench.row_succeeded(latest["image:a:full"], ignore_quality_failures=False)


def test_row_succeeded_can_ignore_quality_failures() -> None:
    row = {
        "case_id": "image:a:full",
        "exit_code": 0,
        "status": "ok",
        "quality_failures": ["missing counts: 2 Boat"],
    }

    assert not bench.row_succeeded(row, ignore_quality_failures=False)
    assert bench.row_succeeded(row, ignore_quality_failures=True)


def test_caption_quality_accepts_people_as_person_label() -> None:
    quality = bench.summarize_caption(
        "A total of 21 people are present on the field, each positioned in various spots.",
        {"Person": 21},
    )

    assert quality["missing_label_mentions"] == []
    assert quality["missing_count_digits"] == []


def test_quality_label_variants_do_not_double_pluralize_plural_label() -> None:
    variants = bench.quality_label_variants("Solar Panels")

    assert "solar panels" in variants
    assert "solar panelses" not in variants


def test_caption_quality_does_not_treat_distant_negative_context_as_count_contradiction() -> None:
    quality = bench.summarize_caption(
        (
            "Three boats are floating on a dark body of water. "
            "There are no visible landmarks or structures in the background, "
            "emphasizing the focus on the boats and their arrangement."
        ),
        {"Boat": 3, "Person": 2},
    )

    assert quality["count_contradiction_sentences"] == []


def test_row_has_recovery_events_detects_recovered_success() -> None:
    assert bench.row_has_recovery_events({"recovery_events": [{"action": "text_recovery_succeeded"}]})
    assert not bench.row_has_recovery_events({"recovery_events": []})
    assert not bench.row_has_recovery_events({})


def test_summary_file_is_atomic_and_counts_latest_status(tmp_path: Path) -> None:
    rows = [
        {"case_id": "a", "final_status": "failed", "status": "exception"},
        {"case_id": "a", "final_status": "ok", "status": "ok"},
        {"case_id": "b", "final_status": "skipped_existing_caption", "status": "skipped_existing_caption"},
    ]

    bench.write_summary(tmp_path, rows)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["total_cases"] == 2
    assert summary["row_count"] == 2
    assert summary["rows_truncated"] is False
    assert summary["rows_omitted"] == 0
    assert summary["totals"] == {"ok": 1, "skipped_existing_caption": 1}


def test_summary_file_caps_row_snapshot_without_losing_totals(tmp_path: Path) -> None:
    rows = [
        {"case_id": f"case_{index}", "final_status": "ok", "status": "ok"}
        for index in range(5)
    ]

    bench.write_summary(tmp_path, rows, row_limit=2)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["total_cases"] == 5
    assert summary["row_count"] == 5
    assert summary["totals"] == {"ok": 5}
    assert summary["row_limit"] == 2
    assert summary["rows_truncated"] is True
    assert summary["rows_omitted"] == 3
    assert summary["rows_sample_policy"] == "latest"
    assert [row["case_id"] for row in summary["rows"]] == ["case_3", "case_4"]


def test_text_artifact_limit_preserves_tail_and_records_truncation(tmp_path: Path) -> None:
    artifact_path = tmp_path / "stdout.txt"
    meta = bench.write_text_artifact(
        artifact_path,
        "prefix\n" + ("middle\n" * 20) + "important tail\n",
        max_bytes=90,
        source="child_stdout",
    )

    text = artifact_path.read_text()
    assert meta["truncated"] is True
    assert meta["original_bytes"] > meta["written_bytes"]
    assert meta["written_bytes"] <= 90
    assert "artifact truncated" in text
    assert text.endswith("important tail\n")
    assert "prefix" not in text


def test_jsonl_artifact_limit_keeps_valid_marker_line(tmp_path: Path) -> None:
    src = tmp_path / "source.jsonl"
    src.write_text("".join(json.dumps({"index": index}) + "\n" for index in range(40)))
    dst = tmp_path / "copy.jsonl"

    meta = bench.copy_text_artifact(src, dst, max_bytes=160, source="source.jsonl")

    lines = [json.loads(line) for line in dst.read_text().splitlines() if line.strip()]
    assert meta["truncated"] is True
    assert lines[0]["event"] == "artifact_truncated"
    assert lines[-1]["index"] == 39


def test_case_payload_merges_request_template_but_preserves_image_fields(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    image_path = dataset / "train" / "images" / "frame.jpg"
    _write_image(image_path)
    label_path = dataset / "train" / "labels" / "frame.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.25 0.25\n")
    request_json = tmp_path / "request.json"
    request_json.write_text(json.dumps({
        "user_prompt": "Template prompt",
        "caption_mode": "windowed",
        "max_boxes": 12,
        "model_id": "template-model",
        "model_variant": "Thinking",
        "refinement_model_id": "template-refinement-model",
        "caption_fallback_model_id": "none",
        "caption_loop_recovery_mode": "off",
        "image_name": "wrong.jpg",
        "label_hints": [],
        "image_width": 999,
    }))
    args = SimpleNamespace(
        prompt="Default prompt",
        max_boxes=0,
        max_new_tokens=None,
        model_id="model",
        refinement_model_id="same",
        loop_recovery="safe_retry_fallback",
        fallback_model_id="auto",
        window_size=672,
        window_overlap=0.1,
        final_sentences=8,
        use_sampling=False,
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        request_json=request_json,
    )
    case = {
        "name": "case",
        "image_name": "frame.jpg",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "caption_mode": "full",
    }

    payload = bench.case_payload(case, dataset, args)

    assert payload["user_prompt"] == "Template prompt"
    assert payload["caption_mode"] == "windowed"
    assert payload["max_boxes"] == 12
    assert payload["model_id"] == "template-model"
    assert payload["model_variant"] == "Thinking"
    assert payload["refinement_model_id"] == "template-refinement-model"
    assert payload["caption_fallback_model_id"] == "none"
    assert payload["caption_loop_recovery_mode"] == "off"
    assert payload["image_name"] == "frame.jpg"
    assert payload["image_width"] == 32
    assert payload["label_hints"] and payload["label_hints"][0]["label"] == "Building"


def test_case_payload_errors_when_explicit_request_template_is_missing(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    image_path = dataset / "train" / "images" / "frame.jpg"
    _write_image(image_path)
    label_path = dataset / "train" / "labels" / "frame.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("")
    args = SimpleNamespace(
        prompt="Prompt",
        max_boxes=0,
        max_new_tokens=None,
        model_id="model",
        refinement_model_id="same",
        loop_recovery="safe_retry_fallback",
        fallback_model_id="auto",
        window_size=672,
        window_overlap=0.1,
        final_sentences=8,
        use_sampling=False,
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        request_json=tmp_path / "missing.json",
    )
    case = {
        "name": "case",
        "image_name": "frame.jpg",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "caption_mode": "full",
    }

    with pytest.raises(FileNotFoundError, match="request template not found"):
        bench.case_payload(case, dataset, args)


def test_run_settings_payload_strips_image_fields_and_detects_prompt_drift(tmp_path: Path) -> None:
    request_json = tmp_path / "request.json"
    request_json.write_text(
        json.dumps({
            "model_id": "template-model",
            "user_prompt": "Template prompt",
            "image_name": "wrong.jpg",
            "label_hints": [],
            "image_width": 999,
        })
    )
    args = _parent_args(
        tmp_path / "dataset",
        tmp_path / "run",
        request_json=request_json,
        model_id="cli-model",
        prompt="CLI prompt",
    )

    settings = bench.run_settings_payload(args)
    same_settings = bench.run_settings_payload(
        _parent_args(
            tmp_path / "dataset",
            tmp_path / "run",
            request_json=request_json,
            model_id="cli-model",
            prompt="CLI prompt",
        )
    )
    changed_prompt_settings = bench.run_settings_payload(
        _parent_args(
            tmp_path / "dataset",
            tmp_path / "run",
            request_json=request_json,
            model_id="cli-model",
            prompt="Different CLI prompt",
        )
    )

    assert settings["request_template"] == {
        "model_id": "template-model",
        "user_prompt": "Template prompt",
    }
    assert settings["caption_args"]["model_id"] == "cli-model"
    assert settings["caption_args"]["prompt"] == "CLI prompt"
    assert settings["caption_args"]["windowed_full_image_strategy"] == "visual"
    assert settings["fingerprint"] == same_settings["fingerprint"]
    assert settings["fingerprint"] != changed_prompt_settings["fingerprint"]


def test_run_settings_payload_fingerprints_windowed_full_image_strategy(tmp_path: Path) -> None:
    visual_settings = bench.run_settings_payload(
        _parent_args(
            tmp_path / "dataset",
            tmp_path / "run",
            caption_mode="windowed",
            windowed_full_image_strategy="visual",
        )
    )
    text_only_settings = bench.run_settings_payload(
        _parent_args(
            tmp_path / "dataset",
            tmp_path / "run",
            caption_mode="windowed",
            windowed_full_image_strategy="text_only",
        )
    )

    assert visual_settings["caption_args"]["windowed_full_image_strategy"] == "visual"
    assert text_only_settings["caption_args"]["windowed_full_image_strategy"] == "text_only"
    assert visual_settings["fingerprint"] != text_only_settings["fingerprint"]


def test_run_settings_payload_uses_request_template_contents_not_path(tmp_path: Path) -> None:
    template_a = tmp_path / "job_a" / "request_fields.json"
    template_b = tmp_path / "job_b" / "request_fields.json"
    template_a.parent.mkdir()
    template_b.parent.mkdir()
    payload = {
        "model_id": "template-model",
        "user_prompt": "Stable prompt",
        "image_name": "ignored.jpg",
        "label_hints": [{"label": "ignored"}],
    }
    template_a.write_text(json.dumps(payload))
    template_b.write_text(json.dumps(payload))
    changed_template = tmp_path / "job_c" / "request_fields.json"
    changed_template.parent.mkdir()
    changed_template.write_text(json.dumps({**payload, "user_prompt": "Changed prompt"}))

    settings_a = bench.run_settings_payload(
        _parent_args(tmp_path / "dataset", tmp_path / "run", request_json=template_a)
    )
    settings_b = bench.run_settings_payload(
        _parent_args(tmp_path / "dataset", tmp_path / "run", request_json=template_b)
    )
    changed_settings = bench.run_settings_payload(
        _parent_args(tmp_path / "dataset", tmp_path / "run", request_json=changed_template)
    )

    assert settings_a["request_template"] == {
        "model_id": "template-model",
        "user_prompt": "Stable prompt",
    }
    assert settings_a["fingerprint"] == settings_b["fingerprint"]
    assert settings_a["fingerprint"] != changed_settings["fingerprint"]


def test_run_settings_payload_ignores_cases_json_path(tmp_path: Path) -> None:
    cases_a = tmp_path / "job_a" / "cases.json"
    cases_b = tmp_path / "job_b" / "cases.json"
    cases_a.parent.mkdir()
    cases_b.parent.mkdir()
    cases_payload = [
        {
            "case_id": "image:a:full",
            "name": "image_000001",
            "stem": "a",
            "caption_mode": "full",
        }
    ]
    cases_a.write_text(json.dumps(cases_payload))
    cases_b.write_text(json.dumps(cases_payload))

    settings_a = bench.run_settings_payload(
        _parent_args(tmp_path / "dataset", tmp_path / "run", cases_json=cases_a)
    )
    settings_b = bench.run_settings_payload(
        _parent_args(tmp_path / "dataset", tmp_path / "run", cases_json=cases_b)
    )

    assert settings_a["fingerprint"] == settings_b["fingerprint"]


def test_parent_resume_rejects_mismatched_run_settings_before_appending(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    image_path = dataset / "images" / "frame.jpg"
    _write_image(image_path)
    output_dir = tmp_path / "run"
    child_calls = []

    def fake_run(cmd, **_kwargs):
        child_calls.append(cmd)
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    assert bench.run_parent(_parent_args(dataset, output_dir, prompt="Describe.")) == 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["run_settings"]["caption_args"]["prompt"] == "Describe."
    assert manifest["run_settings"]["fingerprint"]

    assert bench.run_parent(_parent_args(dataset, output_dir, resume=True, prompt="Describe.")) == 0
    before_mismatch_results = (output_dir / "results.jsonl").read_text()
    with pytest.raises(SystemExit, match="resume_settings_mismatch"):
        bench.run_parent(_parent_args(dataset, output_dir, resume=True, prompt="Different prompt."))

    assert len(child_calls) == 1
    assert (output_dir / "results.jsonl").read_text() == before_mismatch_results


def test_parent_resume_rejects_invalid_results_jsonl_before_appending(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    image_path = dataset / "images" / "frame.jpg"
    _write_image(image_path)
    output_dir = tmp_path / "run"
    child_calls = []

    def fake_run(cmd, **_kwargs):
        child_calls.append(cmd)
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    assert bench.run_parent(_parent_args(dataset, output_dir, prompt="Describe.")) == 0
    results_path = output_dir / "results.jsonl"
    results_path.write_text(results_path.read_text() + "{not valid json}\n")
    before_resume_results = results_path.read_text()

    with pytest.raises(SystemExit, match="resume_results_jsonl_invalid:line 2"):
        bench.run_parent(_parent_args(dataset, output_dir, resume=True, prompt="Describe."))

    assert len(child_calls) == 1
    assert results_path.read_text() == before_resume_results


def test_parent_resume_rejects_invalid_captions_jsonl_before_appending(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    image_path = dataset / "images" / "frame.jpg"
    _write_image(image_path)
    output_dir = tmp_path / "run"
    child_calls = []

    def fake_run(cmd, **_kwargs):
        child_calls.append(cmd)
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    assert bench.run_parent(_parent_args(dataset, output_dir, prompt="Describe.")) == 0
    captions_path = output_dir / "captions.jsonl"
    results_path = output_dir / "results.jsonl"
    captions_path.write_text(captions_path.read_text() + "{not valid json}\n")
    before_resume_captions = captions_path.read_text()
    before_resume_results = results_path.read_text()

    with pytest.raises(SystemExit, match="resume_captions_jsonl_invalid:line 2"):
        bench.run_parent(_parent_args(dataset, output_dir, resume=True, prompt="Describe."))

    assert len(child_calls) == 1
    assert captions_path.read_text() == before_resume_captions
    assert results_path.read_text() == before_resume_results


def test_parent_results_row_records_saved_text_label(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    image_path = dataset / "train" / "images" / "frame.jpg"
    _write_image(image_path)
    label_path = dataset / "train" / "labels" / "frame.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.25 0.25\n")
    output_dir = tmp_path / "run"

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains 1 Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 1,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    args = SimpleNamespace(
        dataset_root=dataset,
        output_dir=output_dir,
        cases_json=None,
        request_json=None,
        all_images=True,
        caption_mode="full",
        sample_size=0,
        sample_seed=13,
        case=[],
        limit=0,
        resume=False,
        skip_existing_captions=False,
        attempts=1,
        cooldown_after_crash=0,
        max_failures=0,
        continue_on_quality_failures=False,
        save_dataset_text_labels=True,
        timeout=30,
        model_id="model",
        refinement_model_id="same",
        fallback_model_id="auto",
        loop_recovery="safe_retry_fallback",
        max_boxes=0,
        max_new_tokens=None,
        final_sentences=8,
        window_size=672,
        window_overlap=0.1,
        mlx_max_image_side=512,
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        prompt="Describe.",
        preview_only=False,
        use_sampling=False,
    )

    assert bench.run_parent(args) == 0

    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[-1]["final_status"] == "ok"
    assert rows[-1]["saved_text_label"].endswith("train/text_labels/frame.txt")
    assert (dataset / "train" / "text_labels" / "frame.txt").read_text().strip() == (
        "The scene contains 1 Building."
    )


def test_parent_limits_raw_attempt_logs_and_passes_limit_to_worker(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    image_path = dataset / "train" / "images" / "frame.jpg"
    _write_image(image_path)
    output_dir = tmp_path / "run"
    observed_limits = []

    def fake_run(cmd, **_kwargs):
        observed_limits.append(cmd[cmd.index("--max-artifact-log-bytes") + 1])
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains 1 Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 1,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(
            returncode=0,
            stdout="stdout prefix\n" + ("x\n" * 80) + "stdout tail\n",
            stderr="stderr prefix\n" + ("y\n" * 80) + "stderr tail\n",
        )

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    args = SimpleNamespace(
        dataset_root=dataset,
        output_dir=output_dir,
        cases_json=None,
        request_json=None,
        all_images=True,
        caption_mode="full",
        sample_size=0,
        sample_seed=13,
        case=[],
        limit=0,
        resume=False,
        skip_existing_captions=False,
        attempts=1,
        cooldown_after_crash=0,
        max_failures=0,
        continue_on_quality_failures=False,
        save_dataset_text_labels=False,
        timeout=30,
        max_artifact_log_bytes=120,
        model_id="model",
        refinement_model_id="same",
        fallback_model_id="auto",
        loop_recovery="safe_retry_fallback",
        max_boxes=0,
        max_new_tokens=None,
        final_sentences=8,
        window_size=672,
        window_overlap=0.1,
        mlx_max_image_side=512,
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        prompt="Describe.",
        preview_only=False,
        use_sampling=False,
    )

    assert bench.run_parent(args) == 0

    assert observed_limits == ["120"]
    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    attempt_dir = Path(rows[-1]["artifact_dir"])
    stdout_text = (attempt_dir / "stdout.txt").read_text()
    stderr_text = (attempt_dir / "stderr.txt").read_text()
    assert "artifact truncated" in stdout_text
    assert stdout_text.endswith("stdout tail\n")
    assert "stdout prefix" not in stdout_text
    assert "artifact truncated" in stderr_text
    assert stderr_text.endswith("stderr tail\n")
    assert rows[-1]["artifact_limits"]["stdout"]["truncated"] is True
    assert rows[-1]["artifact_limits"]["stderr"]["truncated"] is True


def test_parent_records_signal_exit_and_uses_backoff_before_retry(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "frame.jpg")
    output_dir = tmp_path / "run"
    calls = []
    sleeps = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if len(calls) < 3:
            return SimpleNamespace(returncode=-6, stdout="abort stdout", stderr="abort stderr")
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains 1 Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 1,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(bench.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert bench.run_parent(
        _parent_args(
            dataset,
            output_dir,
            attempts=3,
            cooldown_after_crash=1,
            cooldown_backoff_multiplier=2,
            max_cooldown_after_crash=3,
        )
    ) == 0

    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [row["attempt_failure_kind"] for row in rows] == [
        "signal_exit",
        "signal_exit",
        "none",
    ]
    assert [cmd[cmd.index("--mlx-max-image-side") + 1] for cmd in calls] == [
        "512",
        "256",
        "256",
    ]
    assert [row["attempt_profile"]["mlx_max_image_side"] for row in rows] == [
        512,
        256,
        256,
    ]
    assert [row["attempt_profile"]["adaptive_retry_reason"] for row in rows] == [
        "initial_attempt",
        "previous_signal_exit_min_side",
        "previous_signal_exit_min_side",
    ]
    assert [row["attempt_profile"]["previous_attempt_failure_kind"] for row in rows] == [
        None,
        "signal_exit",
        "signal_exit",
    ]
    assert rows[0]["attempt_profile"]["adaptive_retry_image_side"] is False
    assert rows[1]["attempt_profile"]["adaptive_retry_image_side"] is True
    assert rows[2]["attempt_profile"]["adaptive_retry_image_side"] is True
    assert rows[0]["exit_code"] == -6
    assert rows[0]["return_signal"] == 6
    assert rows[0]["return_signal_name"] == "SIGABRT"
    assert rows[0]["next_attempt_cooldown_seconds"] == 1
    assert rows[1]["next_attempt_cooldown_seconds"] == 2
    assert rows[-1]["final_status"] == "ok"
    assert sleeps == [1, 2]

    assert json.loads((output_dir / "heartbeat.json").read_text())["phase"] == "finished"
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["totals"] == {"ok": 1}


def test_parent_appends_terminal_failed_row_after_exhausted_attempts(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "frame.jpg")
    output_dir = tmp_path / "run"

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        attempt_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "exception",
            "exception": "simulated failure",
            "caption_quality": {},
        }
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=1, stdout="failure stdout", stderr="failure stderr")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    assert bench.run_parent(_parent_args(dataset, output_dir, attempts=1)) == 0

    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [row["final_status"] for row in rows] == ["failed_attempt", "failed"]
    assert rows[-1]["terminal_failure"] is True
    assert rows[-1]["status"] == "exception"
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["totals"] == {"failed": 1}
    assert json.loads((output_dir / "heartbeat.json").read_text())["status"] == "completed"


def test_parent_recovers_exhausted_attempts_with_deterministic_count_caption(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "frame.jpg")
    label_dir = dataset / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "frame.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    output_dir = tmp_path / "run"

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        attempt_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "exception",
            "exception": {"type": "RuntimeError", "message": "simulated failure"},
            "caption_quality": {},
        }
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=1, stdout="failure stdout", stderr="failure stderr")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    assert bench.run_parent(_parent_args(dataset, output_dir, attempts=1)) == 0

    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [row["final_status"] for row in rows] == ["failed_attempt", "ok"]
    recovered = rows[-1]
    assert recovered["parent_deterministic_recovery"] is True
    assert recovered["source_attempt_status"] == "exception"
    assert recovered["source_attempt_exit_code"] == 1
    assert recovered["attempt_failure_kind"] == "parent_deterministic_recovery"
    assert recovered["quality_failures"] == []
    assert recovered["recovery_events"][-1]["action"] == "deterministic_recovery_succeeded"
    assert recovered["recovery_events"][-1]["attempt"] == "parent_deterministic_recovery"

    captions = [
        json.loads(line)
        for line in (output_dir / "captions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(captions) == 1
    assert "1 building" in captions[0]["caption"].lower()
    assert captions[0]["used_counts"] == {"Building": 1}
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["totals"] == {"ok": 1}
    heartbeat = json.loads((output_dir / "heartbeat.json").read_text())
    assert heartbeat["status"] == "completed"
    assert heartbeat["failed_cases"] == 0


def test_parent_does_not_recover_generated_qa_training_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "frame.jpg")
    label_dir = dataset / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "frame.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    output_dir = tmp_path / "run"

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        attempt_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "instruction_qa_failed",
            "exception": {
                "type": "InstructionQaGenerationError",
                "message": "generated QA was requested, but no valid generated QA pairs were produced",
            },
            "response": {
                "caption": "The image shows one building.",
                "used_counts": {"Building": 1},
                "generated_qa_pair_count": 0,
            },
            "caption_quality": {},
        }
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=1, stdout="failure stdout", stderr="failure stderr")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    args = _parent_args(
        dataset,
        output_dir,
        attempts=1,
        instruction_dataset=True,
        subcaptions_per_image=8,
        max_failures=1,
    )
    assert bench.run_parent(args) == 1

    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [row["final_status"] for row in rows] == ["failed_attempt", "failed"]
    terminal = rows[-1]
    assert terminal["terminal_failure"] is True
    assert terminal["status"] == "instruction_qa_failed"
    assert terminal["parent_deterministic_recovery_skipped"]["reason"] == "generated_qa_required"
    assert "parent_deterministic_recovery" not in terminal
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["totals"] == {"failed": 1}
    heartbeat = json.loads((output_dir / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
    assert heartbeat["failed_cases"] == 1


def test_worker_generates_instruction_qa_pairs_without_image_path_name_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    image_path = dataset / "images" / "frame.jpg"
    _write_image(image_path)
    label_dir = dataset / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / "frame.txt"
    label_path.write_text("0 0.5 0.5 0.25 0.25\n")
    case = {
        "case_id": "image:frame:full",
        "case": "frame",
        "stem": "frame",
        "name": "frame",
        "image_name": "frame.jpg",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "caption_mode": "full",
        "label_count": 1,
        "class_counts": {"Building": 1},
    }
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    output_dir = tmp_path / "attempt"
    qa_images: list[tuple[int, int]] = []

    def fake_run_qwen_inference(*_args, **_kwargs):
        pil_img = _args[1]
        qa_images.append(tuple(pil_img.size))
        return (
            json.dumps(
                {
                    "qa_pairs": [
                        {
                            "question": "How many buildings are present?",
                            "answer": "1 building is present.",
                        },
                        {
                            "question": "What is the main visible object?",
                            "answer": "The main visible object is a building.",
                        },
                    ]
                }
            ),
            pil_img.size[0],
            pil_img.size[1],
        )

    fake_api = SimpleNamespace(
        qwen_caption_prompt_preview=lambda _request: {"full_text": "preview"},
        qwen_caption=lambda _request: {
            "caption": "The image shows one building.",
            "used_counts": {"Building": 1},
            "used_boxes": 1,
            "truncated": False,
        },
        qwen_progress=lambda: {},
        _run_qwen_inference=fake_run_qwen_inference,
    )
    monkeypatch.setitem(sys.modules, "localinferenceapi", fake_api)

    args = _parent_args(
        dataset,
        output_dir,
        instruction_dataset=True,
        subcaptions_per_image=2,
        instruction_max_new_tokens=512,
    )
    assert bench.run_worker(case_path, output_dir, dataset, args) == 0

    result = json.loads((output_dir / "result.json").read_text())
    assert result["status"] == "ok"
    assert qa_images == [(32, 24)]
    assert result["instruction_qa"]["status"] == "ok"
    assert result["response"]["generated_qa_pair_count"] == 2
    assert [pair["row_type"] for pair in result["response"]["generated_qa_pairs"]] == [
        "generated_qa",
        "generated_qa",
    ]


def test_worker_fails_generated_qa_training_case_when_pairs_are_empty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    image_path = dataset / "images" / "frame.jpg"
    _write_image(image_path)
    label_dir = dataset / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / "frame.txt"
    label_path.write_text("0 0.5 0.5 0.25 0.25\n")
    case = {
        "case_id": "image:frame:full",
        "case": "frame",
        "stem": "frame",
        "name": "frame",
        "image_name": "frame.jpg",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "caption_mode": "full",
        "label_count": 1,
        "class_counts": {"Building": 1},
    }
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    output_dir = tmp_path / "attempt"

    fake_api = SimpleNamespace(
        qwen_caption_prompt_preview=lambda _request: {"full_text": "preview"},
        qwen_caption=lambda _request: {
            "caption": "The image shows one building.",
            "used_counts": {"Building": 1},
            "used_boxes": 1,
            "truncated": False,
        },
        qwen_progress=lambda: {},
        _run_qwen_inference=lambda *_args, **_kwargs: (
            json.dumps({"qa_pairs": []}),
            32,
            24,
        ),
    )
    monkeypatch.setitem(sys.modules, "localinferenceapi", fake_api)

    args = _parent_args(
        dataset,
        output_dir,
        instruction_dataset=True,
        subcaptions_per_image=2,
        instruction_max_new_tokens=512,
    )
    assert bench.run_worker(case_path, output_dir, dataset, args) == 1

    result = json.loads((output_dir / "result.json").read_text())
    assert result["status"] == "instruction_qa_failed"
    assert result["instruction_qa"]["status"] == "empty"
    assert result["response"]["generated_qa_pair_count"] == 0
    assert result["exception"]["type"] == "InstructionQaGenerationError"


def test_parent_honors_restart_request_between_cases(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "frame_a.jpg")
    _write_image(dataset / "images" / "frame_b.jpg")
    output_dir = tmp_path / "run"
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        attempt_dir.mkdir(parents=True, exist_ok=True)
        calls.append(str(attempt_dir))
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        (attempt_dir / "result.json").write_text(json.dumps(result))
        (output_dir / bench.RUNNER_RESTART_REQUEST_NAME).write_text(
            json.dumps({"reason": "test_upgrade"})
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    assert bench.run_parent(_parent_args(dataset, output_dir, attempts=1)) == 0

    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(calls) == 1
    assert len(rows) == 1
    assert rows[0]["final_status"] == "ok"
    assert not (output_dir / bench.RUNNER_RESTART_REQUEST_NAME).exists()
    ack = json.loads((output_dir / bench.RUNNER_RESTART_ACK_NAME).read_text())
    assert ack["request"]["reason"] == "test_upgrade"
    assert ack["processed"] == 1
    heartbeat = json.loads((output_dir / "heartbeat.json").read_text())
    assert heartbeat["status"] == "restart_requested"
    assert heartbeat["phase"] == "restart_requested"
    assert bench.RUNNER_CAPABILITY_GRACEFUL_RESTART in heartbeat["runner_capabilities"]
    assert bench.RUNNER_CAPABILITY_PARENT_DETERMINISTIC_RECOVERY in heartbeat["runner_capabilities"]
    assert bench.RUNNER_CAPABILITY_CAPTION_IO_EVENT_SUMMARY in heartbeat["runner_capabilities"]
    assert bench.RUNNER_CAPABILITY_ADAPTIVE_RETRY_PROFILE in heartbeat["runner_capabilities"]
    assert heartbeat["processed"] == 1
    assert heartbeat["total_cases"] == 2
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert bench.RUNNER_CAPABILITY_GRACEFUL_RESTART in manifest["runner_capabilities"]
    assert bench.RUNNER_CAPABILITY_PARENT_DETERMINISTIC_RECOVERY in manifest["runner_capabilities"]
    assert bench.RUNNER_CAPABILITY_CAPTION_IO_EVENT_SUMMARY in manifest["runner_capabilities"]
    assert bench.RUNNER_CAPABILITY_ADAPTIVE_RETRY_PROFILE in manifest["runner_capabilities"]


def test_parent_writes_heartbeat_during_attempt(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    image_path = dataset / "images" / "frame.jpg"
    _write_image(image_path)
    output_dir = tmp_path / "run"
    observed_heartbeat = []
    observed_worker_heartbeat = []
    observed_lock_epochs = []

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        heartbeat_path = output_dir / "heartbeat.json"
        deadline = bench.time.time() + 1.0
        while bench.time.time() < deadline:
            if heartbeat_path.exists():
                heartbeat = json.loads(heartbeat_path.read_text())
                if heartbeat.get("phase") == "attempt_running":
                    observed_heartbeat.append(heartbeat)
                    break
            bench.time.sleep(0.01)
        bench.atomic_write_json(
            attempt_dir / bench.WORKER_PROGRESS_JSON,
            {
                "seq": 1,
                "phase": "generate",
                "phase_label": "Generating",
                "step_id": "generate_full",
                "step_label": "Compose full-image caption",
                "message": "Generating response tokens",
                "generated_tokens": 17,
                "max_new_tokens": 3000,
            },
        )
        deadline = bench.time.time() + 1.0
        while bench.time.time() < deadline:
            if heartbeat_path.exists():
                heartbeat = json.loads(heartbeat_path.read_text())
                if heartbeat.get("worker_step_label") == "Compose full-image caption":
                    observed_worker_heartbeat.append(heartbeat)
                    break
            bench.time.sleep(0.01)
        lock_path = output_dir / bench.RUNNER_LOCK_NAME
        first_lock = json.loads(lock_path.read_text())
        bench.time.sleep(0.06)
        second_lock = json.loads(lock_path.read_text())
        observed_lock_epochs.extend([
            float(first_lock.get("heartbeat_epoch") or 0.0),
            float(second_lock.get("heartbeat_epoch") or 0.0),
        ])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    args = SimpleNamespace(
        dataset_root=dataset,
        output_dir=output_dir,
        cases_json=None,
        request_json=None,
        all_images=True,
        caption_mode="full",
        sample_size=0,
        sample_seed=13,
        case=[],
        limit=0,
        resume=False,
        skip_existing_captions=False,
        attempts=1,
        cooldown_after_crash=0,
        max_failures=0,
        continue_on_quality_failures=False,
        save_dataset_text_labels=False,
        timeout=30,
        heartbeat_interval=0.02,
        model_id="model",
        refinement_model_id="same",
        fallback_model_id="auto",
        loop_recovery="safe_retry_fallback",
        max_boxes=0,
        max_new_tokens=None,
        final_sentences=8,
        window_size=672,
        window_overlap=0.1,
        mlx_max_image_side=512,
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        prompt="Describe.",
        preview_only=False,
        use_sampling=False,
    )

    assert bench.run_parent(args) == 0

    assert observed_heartbeat
    assert observed_heartbeat[0]["phase"] == "attempt_running"
    assert observed_heartbeat[0]["attempt"] == 1
    assert observed_heartbeat[0]["image_name"] == "frame.jpg"
    assert observed_worker_heartbeat
    assert observed_worker_heartbeat[0]["worker_message"] == "Generating response tokens"
    assert observed_worker_heartbeat[0]["worker_generated_tokens"] == 17
    assert observed_worker_heartbeat[0]["image_name"] == "frame.jpg"
    assert observed_worker_heartbeat[0]["seq"] > observed_heartbeat[0]["seq"]
    assert len(observed_lock_epochs) == 2
    assert observed_lock_epochs[1] > observed_lock_epochs[0]
    final_heartbeat = json.loads((output_dir / "heartbeat.json").read_text())
    assert final_heartbeat["phase"] == "finished"
    assert final_heartbeat["status"] == "completed"
    assert final_heartbeat["processed"] == 1
    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[-1]["worker_progress"]["step_label"] == "Compose full-image caption"
    assert rows[-1]["image_name"] == "frame.jpg"


def test_parent_artifact_lock_blocks_duplicate_writer_before_mutation(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "frame.jpg")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    results_path = output_dir / "results.jsonl"
    original_results = json.dumps(
        {
            "case_id": "image:frame:full",
            "exit_code": 0,
            "status": "ok",
            "final_status": "ok",
            "quality_failures": [],
        }
    ) + "\n"
    results_path.write_text(original_results)
    (output_dir / bench.RUNNER_LOCK_NAME).write_text(
        json.dumps(
            {
                "runner_id": "live-runner",
                "pid": os.getpid(),
                "heartbeat_epoch": time.time(),
            }
        )
    )
    args = SimpleNamespace(
        dataset_root=dataset,
        output_dir=output_dir,
        cases_json=None,
        request_json=None,
        all_images=True,
        caption_mode="full",
        sample_size=0,
        sample_seed=13,
        case=[],
        limit=0,
        resume=False,
        skip_existing_captions=False,
        attempts=1,
        cooldown_after_crash=0,
        max_failures=0,
        continue_on_quality_failures=False,
        save_dataset_text_labels=False,
        timeout=30,
        heartbeat_interval=0,
        artifact_lock_timeout=0.05,
        artifact_lock_stale_seconds=3600,
        artifact_lock_poll_seconds=0.01,
        model_id="model",
        refinement_model_id="same",
        fallback_model_id="auto",
        loop_recovery="safe_retry_fallback",
        max_boxes=0,
        max_new_tokens=None,
        final_sentences=8,
        window_size=672,
        window_overlap=0.1,
        mlx_max_image_side=512,
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        prompt="Describe.",
        preview_only=False,
        use_sampling=False,
    )

    with pytest.raises(SystemExit, match="artifact_lock_active"):
        bench.run_parent(args)

    assert results_path.read_text() == original_results


def test_parent_artifact_lock_does_not_take_over_stale_live_owner(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "frame.jpg")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    lock_path = output_dir / bench.RUNNER_LOCK_NAME
    lock_payload = {
        "runner_id": "stale-but-live",
        "pid": os.getpid(),
        "heartbeat_epoch": time.time() - 7200,
    }
    lock_path.write_text(json.dumps(lock_payload))
    args = SimpleNamespace(
        dataset_root=dataset,
        output_dir=output_dir,
        cases_json=None,
        request_json=None,
        all_images=True,
        caption_mode="full",
        sample_size=0,
        sample_seed=13,
        case=[],
        limit=0,
        resume=True,
        skip_existing_captions=False,
        attempts=1,
        cooldown_after_crash=0,
        max_failures=0,
        continue_on_quality_failures=False,
        save_dataset_text_labels=False,
        timeout=30,
        heartbeat_interval=0,
        artifact_lock_timeout=0.05,
        artifact_lock_stale_seconds=60,
        artifact_lock_poll_seconds=0.01,
        model_id="model",
        refinement_model_id="same",
        fallback_model_id="auto",
        loop_recovery="safe_retry_fallback",
        max_boxes=0,
        max_new_tokens=None,
        final_sentences=8,
        window_size=672,
        window_overlap=0.1,
        mlx_max_image_side=512,
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        prompt="Describe.",
        preview_only=False,
        use_sampling=False,
    )

    with pytest.raises(SystemExit, match="artifact_lock_active"):
        bench.run_parent(args)

    assert json.loads(lock_path.read_text()) == lock_payload


def test_parent_artifact_lock_removes_dead_owner_and_releases_after_run(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "frame.jpg")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / bench.RUNNER_LOCK_NAME).write_text(
        json.dumps(
            {
                "runner_id": "dead-runner",
                "pid": 999999999,
                "heartbeat_epoch": time.time() - 3600,
            }
        )
    )

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    args = SimpleNamespace(
        dataset_root=dataset,
        output_dir=output_dir,
        cases_json=None,
        request_json=None,
        all_images=True,
        caption_mode="full",
        sample_size=0,
        sample_seed=13,
        case=[],
        limit=0,
        resume=False,
        skip_existing_captions=False,
        attempts=1,
        cooldown_after_crash=0,
        max_failures=0,
        continue_on_quality_failures=False,
        save_dataset_text_labels=False,
        timeout=30,
        heartbeat_interval=0,
        artifact_lock_timeout=0.1,
        artifact_lock_stale_seconds=3600,
        artifact_lock_poll_seconds=0.01,
        model_id="model",
        refinement_model_id="same",
        fallback_model_id="auto",
        loop_recovery="safe_retry_fallback",
        max_boxes=0,
        max_new_tokens=None,
        final_sentences=8,
        window_size=672,
        window_overlap=0.1,
        mlx_max_image_side=512,
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        prompt="Describe.",
        preview_only=False,
        use_sampling=False,
    )

    assert bench.run_parent(args) == 0

    assert not (output_dir / bench.RUNNER_LOCK_NAME).exists()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["total_cases"] == 1
    assert summary["totals"] == {"ok": 1}


def test_parent_artifact_lock_removes_invalid_lock_and_releases_after_run(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "frame.jpg")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / bench.RUNNER_LOCK_NAME).write_text("{not-json")

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    assert bench.run_parent(_parent_args(dataset, output_dir)) == 0

    assert not (output_dir / bench.RUNNER_LOCK_NAME).exists()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["total_cases"] == 1
    assert summary["totals"] == {"ok": 1}


def test_parent_unlimited_failures_completes_case_list_with_failed_rows(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "a.jpg")
    _write_image(dataset / "images" / "b.jpg")
    output_dir = tmp_path / "run"

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        case = json.loads(Path(cmd[cmd.index("--case-json") + 1]).read_text())
        if case["stem"] == "a":
            result = {
                "status": "exception",
                "exception": {"type": "RuntimeError", "message": "simulated failure"},
            }
            return_code = 1
        else:
            result = {
                "status": "ok",
                "response": {
                    "caption": "The scene contains a Building.",
                    "used_counts": {"Building": 1},
                    "used_boxes": 0,
                    "truncated": False,
                    "recovery_events": [],
                },
                "caption_quality": {},
            }
            return_code = 0
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=return_code, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    args = SimpleNamespace(
        dataset_root=dataset,
        output_dir=output_dir,
        cases_json=None,
        request_json=None,
        all_images=True,
        caption_mode="full",
        sample_size=0,
        sample_seed=13,
        case=[],
        limit=0,
        resume=False,
        skip_existing_captions=False,
        attempts=1,
        cooldown_after_crash=0,
        max_failures=0,
        continue_on_quality_failures=False,
        save_dataset_text_labels=False,
        timeout=30,
        model_id="model",
        refinement_model_id="same",
        fallback_model_id="auto",
        loop_recovery="safe_retry_fallback",
        max_boxes=0,
        max_new_tokens=None,
        final_sentences=8,
        window_size=672,
        window_overlap=0.1,
        mlx_max_image_side=512,
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        prompt="Describe.",
        preview_only=False,
        use_sampling=False,
    )

    assert bench.run_parent(args) == 0

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["total_cases"] == 2
    assert summary["totals"] == {"failed": 1, "ok": 1}


def test_parent_resume_skips_completed_cases_at_dataset_scale(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    for index in range(20):
        _write_image(dataset / "images" / f"frame_{index:03d}.jpg")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    results_path = output_dir / "results.jsonl"
    completed_keys = {
        f"image:frame_{index:03d}:full"
        for index in range(8)
    }
    for key in sorted(completed_keys):
        stem = key.split(":")[1]
        bench.append_jsonl(
            results_path,
            {
                "case_id": key,
                "case": stem,
                "stem": stem,
                "caption_mode": "full",
                "exit_code": 0,
                "status": "ok",
                "final_status": "ok",
                "quality_failures": [],
                "artifact_dir": None,
            },
        )
    processed_stems = []

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        case = json.loads(Path(cmd[cmd.index("--case-json") + 1]).read_text())
        processed_stems.append(case["stem"])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    args = SimpleNamespace(
        dataset_root=dataset,
        output_dir=output_dir,
        cases_json=None,
        request_json=None,
        all_images=True,
        caption_mode="full",
        sample_size=0,
        sample_seed=13,
        case=[],
        limit=0,
        resume=True,
        skip_existing_captions=False,
        attempts=1,
        cooldown_after_crash=0,
        max_failures=0,
        continue_on_quality_failures=False,
        save_dataset_text_labels=False,
        timeout=30,
        heartbeat_interval=0,
        model_id="model",
        refinement_model_id="same",
        fallback_model_id="auto",
        loop_recovery="safe_retry_fallback",
        max_boxes=0,
        max_new_tokens=None,
        final_sentences=8,
        window_size=672,
        window_overlap=0.1,
        mlx_max_image_side=512,
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        prompt="Describe.",
        preview_only=False,
        use_sampling=False,
    )

    assert bench.run_parent(args) == 0

    assert processed_stems == [f"frame_{index:03d}" for index in range(8, 20)]
    rows = [
        json.loads(line)
        for line in results_path.read_text().splitlines()
        if line.strip()
    ]
    latest = {row["case_id"]: row for row in rows}
    assert len(latest) == 20
    assert len(rows) == 20
    assert sum(1 for row in rows if row.get("resumed_skip")) == 0
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["total_cases"] == 20
    assert summary["totals"] == {"ok": 20}
    heartbeat = json.loads((output_dir / "heartbeat.json").read_text())
    assert heartbeat["status"] == "completed"
    assert heartbeat["processed"] == 20


def test_parent_resume_can_record_legacy_skip_rows_when_requested(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    for stem in ("a", "b", "c"):
        _write_image(dataset / "images" / f"{stem}.jpg")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    results_path = output_dir / "results.jsonl"
    bench.append_jsonl(
        results_path,
        {
            "case_id": "image:a:full",
            "case": "a",
            "stem": "a",
            "caption_mode": "full",
            "exit_code": 0,
            "status": "ok",
            "final_status": "ok",
            "quality_failures": [],
            "artifact_dir": None,
        },
    )
    processed_stems = []

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        case = json.loads(Path(cmd[cmd.index("--case-json") + 1]).read_text())
        processed_stems.append(case["stem"])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    assert bench.run_parent(
        _parent_args(dataset, output_dir, resume=True, record_resume_skips=True)
    ) == 0

    assert processed_stems == ["b", "c"]
    rows = [
        json.loads(line)
        for line in results_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 4
    assert sum(1 for row in rows if row.get("resumed_skip")) == 1
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["total_cases"] == 3
    assert summary["totals"] == {"skipped_completed": 1, "ok": 2}


def test_parent_resume_all_completed_refreshes_summary_without_ledger_churn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    for stem in ("a", "b"):
        _write_image(dataset / "images" / f"{stem}.jpg")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    results_path = output_dir / "results.jsonl"
    for stem in ("a", "b"):
        bench.append_jsonl(
            results_path,
            {
                "case_id": f"image:{stem}:full",
                "case": stem,
                "stem": stem,
                "caption_mode": "full",
                "exit_code": 0,
                "status": "ok",
                "final_status": "ok",
                "quality_failures": [],
                "artifact_dir": None,
            },
        )
    initial_results = results_path.read_text()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("completed resume should not invoke child workers")

    monkeypatch.setattr(bench.subprocess, "run", fail_if_called)

    assert bench.run_parent(_parent_args(dataset, output_dir, resume=True)) == 0

    assert results_path.read_text() == initial_results
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["total_cases"] == 2
    assert summary["totals"] == {"ok": 2}
    heartbeat = json.loads((output_dir / "heartbeat.json").read_text())
    assert heartbeat["status"] == "completed"
    assert heartbeat["processed"] == 2


def test_parent_summary_row_limit_bounds_running_summary_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    for index in range(4):
        _write_image(dataset / "images" / f"frame_{index}.jpg")
    output_dir = tmp_path / "run"

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    assert bench.run_parent(_parent_args(dataset, output_dir, summary_row_limit=2)) == 0

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["total_cases"] == 4
    assert summary["row_count"] == 4
    assert summary["totals"] == {"ok": 4}
    assert summary["rows_truncated"] is True
    assert summary["rows_omitted"] == 2
    assert len(summary["rows"]) == 2


def test_parent_resume_reprocesses_recovered_rows_when_requested(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labelmap.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "labelmap.txt").write_text("Building\n")
    _write_image(dataset / "images" / "frame.jpg")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    results_path = output_dir / "results.jsonl"
    bench.append_jsonl(
        results_path,
        {
            "case_id": "image:frame:full",
            "case": "frame",
            "stem": "frame",
            "caption_mode": "full",
            "exit_code": 0,
            "status": "ok",
            "final_status": "ok",
            "quality_failures": [],
            "recovery_events": [{"action": "loop_detected"}],
            "artifact_dir": None,
        },
    )
    processed_stems = []

    def fake_run(cmd, **_kwargs):
        attempt_dir = Path(cmd[cmd.index("--output-dir") + 1])
        case = json.loads(Path(cmd[cmd.index("--case-json") + 1]).read_text())
        processed_stems.append(case["stem"])
        result = {
            "status": "ok",
            "response": {
                "caption": "The scene contains a Building.",
                "used_counts": {"Building": 1},
                "used_boxes": 0,
                "truncated": False,
                "recovery_events": [],
            },
            "caption_quality": {},
        }
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    args = _parent_args(
        dataset,
        output_dir,
        resume=True,
        resume_reprocess_recovery_events=True,
    )

    assert bench.run_parent(args) == 0
    assert processed_stems == ["frame"]
    rows = [
        json.loads(line)
        for line in results_path.read_text().splitlines()
        if line.strip()
    ]
    assert rows[-1]["final_status"] == "ok"
    assert not rows[-1].get("recovery_events")
