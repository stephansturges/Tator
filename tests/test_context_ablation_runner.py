import json
from pathlib import Path

import numpy as np

from tools.context_feature_variants import compute_feature_schema_hash
from tools.run_context_feature_ablation import (
    LaneConfig,
    _gate0_decision,
    _gate1_decision,
    _gate2_decision,
    _load_lane_runtime_info,
    _method_gate_type,
)


def test_gate0_rejects_zero_heavy_non_varying_duplicate_block():
    decision = _gate0_decision(
        {
            "zero_fraction": 0.99,
            "varying_fraction": 0.05,
            "duplicate_fraction": 0.95,
            "variant_type": "global-only",
        }
    )
    assert not decision["pass"]
    assert "zero_fraction_exceeds_0.95" in decision["reasons"]
    assert "varying_fraction_below_0.20" in decision["reasons"]
    assert "duplicate_fraction_exceeds_0.90" in decision["reasons"]


def test_gate1_rejects_candidate_specific_method_when_gain_is_too_small():
    decision = _gate1_decision(
        "candidate-specific",
        [
            {
                "dataset_variant": "nonwindow",
                "delta_f1": 0.0010,
                "coverage_drop": 0.0010,
                "feature_gain_share": 0.08,
            },
            {
                "dataset_variant": "window",
                "delta_f1": 0.0012,
                "coverage_drop": 0.0015,
                "feature_gain_share": 0.09,
            },
        ],
    )
    assert not decision["pass"]
    assert "best_delta_below_0.0025" in decision["reasons"]
    assert "mean_delta_below_0.0015" in decision["reasons"]


def test_gate1_accepts_candidate_specific_method_with_clear_margin():
    decision = _gate1_decision(
        "candidate-specific",
        [
            {
                "dataset_variant": "nonwindow",
                "delta_f1": 0.0030,
                "coverage_drop": 0.0010,
                "feature_gain_share": 0.08,
            },
            {
                "dataset_variant": "window",
                "delta_f1": 0.0026,
                "coverage_drop": 0.0020,
                "feature_gain_share": 0.05,
            },
        ],
    )
    assert decision["pass"]


def test_gate2_rejects_method_when_mean_delta_is_non_positive():
    decision = _gate2_decision(
        [
            {"delta_f1": 0.0005, "coverage_drop": 0.001, "feature_gain_share": 0.07},
            {"delta_f1": -0.0006, "coverage_drop": 0.001, "feature_gain_share": 0.07},
        ]
    )
    assert not decision["pass"]
    assert "mean_delta_non_positive" in decision["reasons"]


def test_method_gate_type_uses_strictest_variant_type_across_rows():
    assert (
        _method_gate_type(
            [
                {"dataset_variant": "nonwindow", "variant_type": "candidate-specific"},
                {"dataset_variant": "window", "variant_type": "global-only"},
            ]
        )
        == "global-only"
    )


def test_load_lane_runtime_info_fails_when_summary_schema_is_stale(tmp_path: Path):
    labeled = tmp_path / "derived.npz"
    feature_names = np.asarray(["cand_score_yolo", "imgctx_scene_detector_supported_ratio"], dtype=object)
    schema_hash = compute_feature_schema_hash(
        feature_names,
        classifier_classes=["person"],
        labelmap=["person"],
        context_variant_id="scene_summary_v1",
        variant_config={"drop_img_probs": True},
    )
    np.savez(
        labeled,
        X=np.asarray([[0.5, 1.0]], dtype=np.float32),
        meta=np.asarray([json.dumps({"image": "img.jpg", "label": "person", "score": 0.9})], dtype=object),
        feature_names=feature_names,
        labelmap=np.asarray(["person"], dtype=object),
        classifier_classes=np.asarray(["person"], dtype=object),
        feature_schema_hash=np.asarray(schema_hash),
        context_variant_id=np.asarray("scene_summary_v1"),
        parent_feature_npz=np.asarray("/tmp/base.npz"),
        parent_feature_schema_hash=np.asarray("base-hash"),
        variant_config_json=np.asarray('{"drop_img_probs":true}'),
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"feature_schema_hash": "stale-hash", "summary": {"variant_type": "candidate-specific"}}),
        encoding="utf-8",
    )
    lane = LaneConfig(
        lane_id="scene_nonwindow",
        dataset_variant="nonwindow",
        method="scene_summary_v1",
        labeled_npz=labeled,
        prepass_jsonl=tmp_path / "prepass.jsonl",
        derivation_summary_json=summary,
    )
    try:
        _load_lane_runtime_info(lane)
    except SystemExit as exc:
        assert "derivation_summary_schema_mismatch" in str(exc)
    else:
        raise AssertionError("expected stale derivation summary to fail closed")


def test_load_lane_runtime_info_uses_payload_stats_without_summary(tmp_path: Path):
    labeled = tmp_path / "derived.npz"
    feature_names = np.asarray(["cand_score_yolo", "imgctx_scene_detector_supported_ratio"], dtype=object)
    schema_hash = compute_feature_schema_hash(
        feature_names,
        classifier_classes=["person"],
        labelmap=["person"],
        context_variant_id="scene_summary_v1",
        variant_config={"drop_img_probs": True},
    )
    np.savez(
        labeled,
        X=np.asarray([[0.7, 1.0], [0.2, 0.0]], dtype=np.float32),
        meta=np.asarray(
            [
                json.dumps({"image": "img_a.jpg", "label": "person", "score": 0.9}),
                json.dumps({"image": "img_a.jpg", "label": "person", "score": 0.2}),
            ],
            dtype=object,
        ),
        feature_names=feature_names,
        labelmap=np.asarray(["person"], dtype=object),
        classifier_classes=np.asarray(["person"], dtype=object),
        feature_schema_hash=np.asarray(schema_hash),
        context_variant_id=np.asarray("scene_summary_v1"),
        parent_feature_npz=np.asarray("/tmp/base.npz"),
        parent_feature_schema_hash=np.asarray("base-hash"),
        variant_config_json=np.asarray('{"drop_img_probs":true}'),
    )
    lane = LaneConfig(
        lane_id="scene_window",
        dataset_variant="window",
        method="scene_summary_v1",
        labeled_npz=labeled,
        prepass_jsonl=tmp_path / "prepass.jsonl",
        derivation_summary_json=None,
    )
    info = _load_lane_runtime_info(lane)
    assert info.context_variant_id == "scene_summary_v1"
    assert info.derivation_stats["new_feature_count"] == 1
