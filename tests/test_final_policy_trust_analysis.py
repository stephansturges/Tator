from __future__ import annotations

import json
import sys
from pathlib import Path

from tools import run_final_policy_trust_analysis as runner


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _debug_variant(
    *,
    f1: float,
    precision: float,
    recall: float,
    delta_f1: float,
    sam_only: int,
    baseline_sam_only: int,
    accepted_p99: float,
    baseline_p99: float,
) -> dict:
    return {
        "metrics": {
            "tp": 1000,
            "fp": 100,
            "fn": 200,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "compare_to_baseline_hand": {
            "delta_tp": 0,
            "delta_fp": 0,
            "delta_fn": 0,
            "delta_precision": 0.0,
            "delta_recall": 0.0,
            "delta_f1": delta_f1,
        },
        "acceptance_audit": {
            "accepted_rows": 1000,
            "accepted_rate": 0.5,
            "accepted_by_subgroup": {
                "sam_only": sam_only,
                "sam3_similarity_primary": 100,
                "sam3_text_primary": 200,
                "detector_supported": 700,
            },
        },
        "duplicate_density": {
            "accepted_per_image_stats": {
                "mean": 10.0,
                "p95": 20.0,
                "p99": accepted_p99,
                "max": 50.0,
            },
            "deduped_per_image_stats": {
                "mean": 8.0,
                "p95": 16.0,
                "p99": 20.0,
                "max": 30.0,
            },
            "accepted_to_deduped_ratio": 1.2,
        },
        "top_fp_sources": [{"label": "person", "primary_source": "sam3_text", "tp": 10, "fp": 50, "accepted": 60}],
    }


def test_final_policy_trust_analysis_rejects_learned_and_promotes_similarity(
    monkeypatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    refined_tag = "text_m1p4__sim_m1p2"

    baseline_dir = run_root / "postrun_sam_bias_magnitude_sweep" / "full" / refined_tag / "full" / "seed_42"
    _write_json(
        baseline_dir / "model_dynamic.meta.json",
        {
            "split_head": {"enabled": False},
            "xgb_params": {"max_depth": 8, "eta": 0.01, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "gamma": 0.0, "lambda": 1.0, "alpha": 0.0},
            "n_estimators": 900,
        },
    )
    _write_json(
        baseline_dir / "policy.json",
        {
            "sam_bias_scope": "sam_only",
            "sam_only_min_prob_default": 0.15,
            "consensus_iou_default": 0.7,
            "logit_bias_by_source_class": {"sam3_text": {"__default__": -1.4}, "sam3_similarity": {"__default__": -1.2}},
        },
    )

    nonwindow_dir = run_root / "final_matrix" / "nonwindow" / "full" / "seed_42"
    _write_json(
        nonwindow_dir / "model_foo.meta.json",
        {
            "split_head": {"enabled": True},
            "xgb_params": {"max_depth": 12, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "gamma": 0.0, "lambda": 1.0, "alpha": 0.0},
            "n_estimators": 600,
        },
    )
    _write_json(nonwindow_dir / "policy_bar.json", {"consensus_iou_default": 0.7})

    for seed in ("42", "1337", "2025"):
        _write_json(
            run_root / "out" / "debug_policy_layer_surface_mismatch" / f"seed_{seed}" / "results_summary.json",
            {
                "seed": seed,
                "thresholds": {"base_default": 0.65, "learned_default": 0.42},
                "variants": {
                    "full": {
                        "baseline_hand": _debug_variant(
                            f1=0.842,
                            precision=0.889,
                            recall=0.800,
                            delta_f1=0.0,
                            sam_only=200,
                            baseline_sam_only=200,
                            accepted_p99=25.0,
                            baseline_p99=25.0,
                        ),
                        "learned_selected": _debug_variant(
                            f1=0.705,
                            precision=0.580,
                            recall=0.896,
                            delta_f1=-0.137,
                            sam_only=40000,
                            baseline_sam_only=200,
                            accepted_p99=200.0,
                            baseline_p99=25.0,
                        ),
                        "learned_base_thresholds": _debug_variant(
                            f1=0.766,
                            precision=0.897,
                            recall=0.669,
                            delta_f1=-0.076,
                            sam_only=4000,
                            baseline_sam_only=200,
                            accepted_p99=80.0,
                            baseline_p99=25.0,
                        ),
                        "learned_gate_only": _debug_variant(
                            f1=0.774,
                            precision=0.917,
                            recall=0.669,
                            delta_f1=-0.068,
                            sam_only=3000,
                            baseline_sam_only=200,
                            accepted_p99=70.0,
                            baseline_p99=25.0,
                        ),
                        "learned_full_hand": _debug_variant(
                            f1=0.816,
                            precision=0.922,
                            recall=0.732,
                            delta_f1=-0.026,
                            sam_only=500,
                            baseline_sam_only=200,
                            accepted_p99=32.0,
                            baseline_p99=25.0,
                        ),
                    },
                    "val": {
                        "baseline_hand": _debug_variant(
                            f1=0.841,
                            precision=0.886,
                            recall=0.801,
                            delta_f1=0.0,
                            sam_only=40,
                            baseline_sam_only=40,
                            accepted_p99=15.0,
                            baseline_p99=15.0,
                        ),
                        "learned_selected": _debug_variant(
                            f1=0.702,
                            precision=0.579,
                            recall=0.892,
                            delta_f1=-0.139,
                            sam_only=8000,
                            baseline_sam_only=40,
                            accepted_p99=100.0,
                            baseline_p99=15.0,
                        ),
                        "learned_base_thresholds": _debug_variant(
                            f1=0.773,
                            precision=0.895,
                            recall=0.680,
                            delta_f1=-0.068,
                            sam_only=900,
                            baseline_sam_only=40,
                            accepted_p99=50.0,
                            baseline_p99=15.0,
                        ),
                        "learned_gate_only": _debug_variant(
                            f1=0.780,
                            precision=0.915,
                            recall=0.680,
                            delta_f1=-0.061,
                            sam_only=600,
                            baseline_sam_only=40,
                            accepted_p99=40.0,
                            baseline_p99=15.0,
                        ),
                        "learned_full_hand": _debug_variant(
                            f1=0.820,
                            precision=0.919,
                            recall=0.739,
                            delta_f1=-0.022,
                            sam_only=100,
                            baseline_sam_only=40,
                            accepted_p99=18.0,
                            baseline_p99=15.0,
                        ),
                    },
                },
            },
        )

    _write_json(
        run_root / "postrun_similarity_quality_full_window_eval" / "results_summary.json",
        {
            "ranked": [{"tag": "a0p5", "mean_f1": 0.8435, "mean_delta_f1": 0.0011}],
            "rows": [
                {
                    "tag": "a0p5",
                    "alpha": 0.5,
                    "seed": seed,
                    "metrics": {"tp": 1000, "fp": 90, "fn": 199, "precision": 0.891, "recall": 0.801, "f1": 0.8435, "coverage_preservation": 0.844},
                    "baseline_metrics": {"tp": 1000, "fp": 100, "fn": 200, "precision": 0.889, "recall": 0.800, "f1": 0.8424, "coverage_preservation": 0.843},
                    "compare_to_baseline": {"delta_tp": 0, "delta_fp": -10, "delta_fn": -1, "delta_precision": 0.002, "delta_recall": 0.001, "delta_f1": 0.0011, "delta_coverage_preservation": 0.001},
                }
                for seed in ("42", "1337", "2025")
            ],
        },
    )

    class Result:
        def __init__(self) -> None:
            self.stdout = ""

    commands = []

    def fake_run(cmd, cwd=None, check=None, capture_output=False, text=False):
        commands.append(list(cmd))
        if "tools/build_calibration_report_bundle.py" in cmd:
            out_json = Path(cmd[cmd.index("--output-json") + 1])
            out_md = Path(cmd[cmd.index("--output-md") + 1])
            out_json.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            out_md.write_text("# stub\n", encoding="utf-8")
            return Result()
        raise AssertionError(cmd)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_policy_trust_analysis.py",
            "--run-root",
            str(run_root),
            "--output-dir",
            str(run_root / "out"),
        ],
    )

    runner.main()

    decision = json.loads((run_root / "out" / "decision_summary.json").read_text())
    assert decision["learned_scoring_trust_status"] == "rejected"
    assert decision["best_learned_variant"] == "learned_full_hand"
    assert decision["similarity_quality_status"] == "promoted"
    assert decision["recommended_canonical_stack"]["windowed"] == "refined_hand_policy_plus_similarity_quality"

    ranked = json.loads((run_root / "out" / "results_ranked.json").read_text())
    assert ranked["full"][0]["tag"] == "similarity_quality_best"
    assert any("tools/build_calibration_report_bundle.py" in cmd for cmd in commands)


def test_final_policy_trust_analysis_allows_window_only_runs(
    monkeypatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    refined_tag = "text_m1p4__sim_m1p2"

    baseline_dir = run_root / "postrun_sam_bias_magnitude_sweep" / "full" / refined_tag / "full" / "seed_42"
    _write_json(
        baseline_dir / "model_dynamic.meta.json",
        {
            "split_head": {"enabled": False},
            "xgb_params": {"max_depth": 8, "eta": 0.01},
            "n_estimators": 900,
        },
    )
    _write_json(
        baseline_dir / "policy.json",
        {
            "sam_bias_scope": "sam_only",
            "sam_only_min_prob_default": 0.15,
            "consensus_iou_default": 0.7,
        },
    )

    for seed in ("42", "1337", "2025"):
        _write_json(
            run_root / "out" / "debug_policy_layer_surface_mismatch" / f"seed_{seed}" / "results_summary.json",
            {
                "seed": seed,
                "thresholds": {"base_default": 0.65, "learned_default": 0.42},
                "variants": {
                    "full": {
                        "baseline_hand": _debug_variant(
                            f1=0.842,
                            precision=0.889,
                            recall=0.800,
                            delta_f1=0.0,
                            sam_only=200,
                            baseline_sam_only=200,
                            accepted_p99=25.0,
                            baseline_p99=25.0,
                        ),
                        "learned_selected": _debug_variant(
                            f1=0.705,
                            precision=0.580,
                            recall=0.896,
                            delta_f1=-0.137,
                            sam_only=40000,
                            baseline_sam_only=200,
                            accepted_p99=200.0,
                            baseline_p99=25.0,
                        ),
                        "learned_base_thresholds": _debug_variant(
                            f1=0.766,
                            precision=0.897,
                            recall=0.669,
                            delta_f1=-0.076,
                            sam_only=4000,
                            baseline_sam_only=200,
                            accepted_p99=80.0,
                            baseline_p99=25.0,
                        ),
                        "learned_gate_only": _debug_variant(
                            f1=0.774,
                            precision=0.917,
                            recall=0.669,
                            delta_f1=-0.068,
                            sam_only=3000,
                            baseline_sam_only=200,
                            accepted_p99=70.0,
                            baseline_p99=25.0,
                        ),
                        "learned_full_hand": _debug_variant(
                            f1=0.816,
                            precision=0.922,
                            recall=0.732,
                            delta_f1=-0.026,
                            sam_only=500,
                            baseline_sam_only=200,
                            accepted_p99=32.0,
                            baseline_p99=25.0,
                        ),
                    },
                    "val": {
                        "baseline_hand": _debug_variant(
                            f1=0.841,
                            precision=0.886,
                            recall=0.801,
                            delta_f1=0.0,
                            sam_only=40,
                            baseline_sam_only=40,
                            accepted_p99=15.0,
                            baseline_p99=15.0,
                        ),
                    },
                },
            },
        )

    _write_json(
        run_root / "postrun_similarity_quality_full_window_eval" / "results_summary.json",
        {
            "ranked": [{"tag": "a0p5", "mean_f1": 0.8435, "mean_delta_f1": 0.0011}],
            "rows": [
                {
                    "tag": "a0p5",
                    "alpha": 0.5,
                    "seed": seed,
                    "metrics": {"tp": 1000, "fp": 90, "fn": 199, "precision": 0.891, "recall": 0.801, "f1": 0.8435, "coverage_preservation": 0.844},
                    "baseline_metrics": {"tp": 1000, "fp": 100, "fn": 200, "precision": 0.889, "recall": 0.800, "f1": 0.8424, "coverage_preservation": 0.843},
                    "compare_to_baseline": {"delta_tp": 0, "delta_fp": -10, "delta_fn": -1, "delta_precision": 0.002, "delta_recall": 0.001, "delta_f1": 0.0011, "delta_coverage_preservation": 0.001},
                }
                for seed in ("42", "1337", "2025")
            ],
        },
    )

    class Result:
        def __init__(self) -> None:
            self.stdout = ""

    def fake_run(cmd, cwd=None, check=None, capture_output=False, text=False):
        if "tools/build_calibration_report_bundle.py" in cmd:
            out_json = Path(cmd[cmd.index("--output-json") + 1])
            out_md = Path(cmd[cmd.index("--output-md") + 1])
            out_json.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            out_md.write_text("# stub\n", encoding="utf-8")
            return Result()
        raise AssertionError(cmd)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_policy_trust_analysis.py",
            "--run-root",
            str(run_root),
            "--output-dir",
            str(run_root / "out"),
        ],
    )

    runner.main()

    decision = json.loads((run_root / "out" / "decision_summary.json").read_text())
    assert decision["learned_scoring_trust_status"] == "rejected"
    assert decision["canonical_nonwindowed_recipe"]["validation_status"] == "not_available_for_window_only_run"
    assert decision["canonical_nonwindowed_recipe"]["lane"] == "nonwindow"
