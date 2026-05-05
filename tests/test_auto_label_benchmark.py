from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_auto_label_benchmark import (
    build_benchmark_cases,
    compare_to_baseline,
    load_optional_json,
    load_sample_manifest,
    summarize_case_result,
    build_case_payload,
)


pytestmark = [pytest.mark.auto_label_perf]


def test_build_benchmark_cases_has_expected_profiles() -> None:
    nightly = build_benchmark_cases("nightly")
    weekly = build_benchmark_cases("weekly")

    assert len(nightly) == 12
    assert len(weekly) == 4
    assert nightly[0].name == "edr_full_image"
    assert weekly[-1].name == "edr_planner_with_caption"


def test_load_sample_manifest_accepts_image_dict_payload(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(
        json.dumps(
            {
                "dataset_id": "qwen_dataset",
                "target_mode": "segmentation",
                "images": [
                    {"image_relpath": "train/img1.jpg"},
                    {"image_name": "train/img2.jpg"},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = load_sample_manifest(sample_path)

    assert payload["dataset_id"] == "qwen_dataset"
    assert payload["target_mode"] == "segmentation"
    assert payload["image_relpaths"] == ["train/img1.jpg", "train/img2.jpg"]


def test_compare_to_baseline_flags_relative_regressions() -> None:
    baseline = {
        "throughput_img_per_sec": 2.0,
        "avg_latency_sec": 1.0,
        "p95_latency_sec": 1.5,
        "planner_avg_sec_per_image": 0.2,
        "falcon_queries_per_image": 1.0,
    }
    current = {
        "status": "completed",
        "throughput_img_per_sec": 1.2,
        "avg_latency_sec": 1.3,
        "p95_latency_sec": 2.0,
        "planner_avg_sec_per_image": 0.3,
        "falcon_queries_per_image": 1.2,
    }

    failures = compare_to_baseline(current, baseline)

    assert "throughput_regressed:-0.4000" in failures
    assert "avg_latency_regressed:0.3000" in failures
    assert "p95_latency_regressed:0.3333" in failures
    assert "planner_latency_regressed:0.5000" in failures
    assert "falcon_query_regressed:0.2000" in failures


def test_summarize_case_result_computes_latency_quantiles() -> None:
    case = build_benchmark_cases("weekly")[0]
    job = {
        "job_id": "al_bench",
        "status": "completed",
        "result": {
            "dataset_id": "qwen_dataset",
            "images_processed": 4,
            "images_total": 4,
            "labels_added": 8,
            "duplicates_dropped": 1,
            "zero_write_images": 0,
            "writes_applied": 8,
            "falcon_query_count": 6,
            "image_times_sec": [1.0, 2.0, 3.0, 4.0],
            "timings_sec": {"planner": 2.0, "total": 10.0},
        },
    }

    summary = summarize_case_result(case, job, elapsed_sec=8.0)

    assert summary["throughput_img_per_sec"] == pytest.approx(0.5)
    assert summary["avg_latency_sec"] == pytest.approx(2.5)
    assert summary["p50_latency_sec"] == pytest.approx(2.5)
    assert summary["p95_latency_sec"] == pytest.approx(3.85)
    assert summary["planner_avg_sec_per_image"] == pytest.approx(0.5)
    assert summary["falcon_queries_per_image"] == pytest.approx(1.5)
    assert summary["dataset_id"] == "qwen_dataset"


def test_load_optional_json_accepts_missing_path() -> None:
    assert load_optional_json(None) == {}
    assert load_optional_json("") == {}


def test_build_case_payload_applies_overrides() -> None:
    case = build_benchmark_cases("weekly")[0]
    payload = build_case_payload(
        dataset_id="linked_ds_case_a",
        image_relpaths=["img1.jpg", "img2.jpg"],
        case=case,
        edr_package_id="edr_pkg_1",
        target_mode="detection",
        split="all",
        class_names=["person"],
        overrides={
            "falcon_device": "cuda:1",
            "use_planner_caption": True,
            "enable_yolo": False,
        },
    )

    assert payload["dataset_id"] == "linked_ds_case_a"
    assert payload["falcon_device"] == "cuda:1"
    assert payload["use_planner_caption"] is True
    assert payload["enable_yolo"] is False
    assert payload["class_names"] == ["person"]
