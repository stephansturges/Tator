import json
from pathlib import Path
from types import SimpleNamespace

from tools.build_calibration_report_bundle import _build_bundle_from_args


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_single_run_report_bundle(tmp_path: Path):
    eval_json = tmp_path / "eval.json"
    analysis_json = tmp_path / "analysis.json"
    meta_json = tmp_path / "meta.json"
    selection_json = tmp_path / "selection.json"
    _write_json(
        eval_json,
        {
            "tp": 8,
            "fp": 2,
            "fn": 1,
            "metric_tiers": {
                "post_xgb": {"accepted_all": {"tp": 8, "fp": 2, "fn": 1, "precision": 0.8, "recall": 0.8889, "f1": 0.8421}},
                "post_cluster": {"source_attributed": {"yolo_rfdetr_union": {"f1": 0.8}}},
            },
            "coverage_upper_bound": {"candidate_all": {"recall_upper_bound": 0.95}},
        },
    )
    _write_json(
        analysis_json,
        {
            "per_class": [
                {"label": "person", "tp": 5, "fp": 1, "fn": 0, "support_gt": 5, "support_pred": 6},
                {"label": "truck", "tp": 3, "fp": 1, "fn": 1, "support_gt": 4, "support_pred": 4},
            ],
            "per_class_per_source": [
                {"label": "person", "primary_source": "yolo", "tp": 4, "fp": 1, "accepted": 5},
                {"label": "truck", "primary_source": "sam3_similarity", "tp": 1, "fp": 1, "accepted": 2},
            ],
            "boundary_hits": {
                "decision_rows": 10,
                "hard_gate_rejections": {"sam_only_floor": 1},
                "buckets": {
                    "within_0p01": {
                        "accepted": 2,
                        "rejected": 1,
                        "positive_rows": 2,
                        "negative_rows": 1,
                        "by_label": {"person": {"accepted": 2, "rejected": 0, "positive_rows": 2, "negative_rows": 0}},
                        "by_source": {"yolo": {"accepted": 2, "rejected": 0, "positive_rows": 2, "negative_rows": 0}},
                    }
                },
            },
            "calibration_diagnostics": {
                "candidate_count": 10,
                "positive_count": 6,
                "brier_sum_sq": 1.2,
                "bins": [
                    {"bin_index": 0, "lower": 0.0, "upper": 0.1, "count": 2, "sum_prob": 0.1, "sum_positive": 0.0},
                    {"bin_index": 9, "lower": 0.9, "upper": 1.0, "count": 3, "sum_prob": 2.85, "sum_positive": 3.0},
                ],
            },
        },
    )
    _write_json(
        meta_json,
        {
            "policy_layer_summary": {
                "requested_variant": "bakeoff",
                "selected_variant": "xgb",
                "baseline_f1": 0.81,
                "selected_f1": 0.8421,
                "delta_vs_baseline_f1": 0.0321,
            }
        },
    )
    _write_json(selection_json, {"requested_variant": "bakeoff", "selected_variant": "xgb", "trained_variants": ["lreg", "xgb"]})

    args = SimpleNamespace(
        eval_json=str(eval_json),
        analysis_json=str(analysis_json),
        analysis_json_glob=[],
        meta_json=str(meta_json),
        policy_selection_json=str(selection_json),
        results_summary_json=None,
        ranked_json=None,
        model_family="mlp",
    )

    bundle = _build_bundle_from_args(args)
    assert bundle["report_kind"] == "calibration_job"
    assert bundle["model_family"] == "mlp"
    assert bundle["policy_layer"]["selected_variant"] == "xgb"
    assert bundle["overall_metrics"]["f1"] > 0.84
    assert bundle["per_class"][0]["label"] == "person"
    assert bundle["per_class_per_source"][0]["primary_source"] in {"sam3_similarity", "yolo"}
    assert bundle["boundary_hits"]["decision_rows"] == 10
    assert bundle["calibration_diagnostics"]["candidate_count"] == 10


def test_build_postrun_bundle_prefers_best_tag_analysis_rows(tmp_path: Path):
    best_analysis = tmp_path / "best_analysis.json"
    worse_analysis = tmp_path / "worse_analysis.json"
    summary_json = tmp_path / "results_summary.json"
    _write_json(
        best_analysis,
        {
            "per_class": [{"label": "person", "tp": 4, "fp": 1, "fn": 1, "support_gt": 5, "support_pred": 5}],
            "per_class_per_source": [{"label": "person", "primary_source": "yolo", "tp": 4, "fp": 1, "accepted": 5}],
            "boundary_hits": {"decision_rows": 4, "hard_gate_rejections": {}, "buckets": {}},
            "calibration_diagnostics": {"candidate_count": 4, "positive_count": 3, "brier_sum_sq": 0.2, "bins": []},
        },
    )
    _write_json(
        worse_analysis,
        {
            "per_class": [{"label": "truck", "tp": 1, "fp": 4, "fn": 3, "support_gt": 4, "support_pred": 5}],
            "per_class_per_source": [{"label": "truck", "primary_source": "sam3_similarity", "tp": 1, "fp": 4, "accepted": 5}],
            "boundary_hits": {"decision_rows": 5, "hard_gate_rejections": {}, "buckets": {}},
            "calibration_diagnostics": {"candidate_count": 5, "positive_count": 1, "brier_sum_sq": 1.0, "bins": []},
        },
    )
    _write_json(
        summary_json,
        {
            "run_root": str(tmp_path),
            "rows": [
                {"tag": "best", "seed": "42", "metrics": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "coverage_preservation": 0.9}, "analysis_json": str(best_analysis)},
                {"tag": "worse", "seed": "42", "metrics": {"precision": 0.3, "recall": 0.25, "f1": 0.27, "coverage_preservation": 0.5}, "analysis_json": str(worse_analysis)},
            ],
            "ranked": [
                {"tag": "best", "mean_precision": 0.8, "mean_recall": 0.8, "mean_f1": 0.8},
                {"tag": "worse", "mean_precision": 0.3, "mean_recall": 0.25, "mean_f1": 0.27},
            ],
        },
    )

    args = SimpleNamespace(
        eval_json=None,
        analysis_json=None,
        analysis_json_glob=[str(tmp_path / "*.json")],
        meta_json=None,
        policy_selection_json=None,
        results_summary_json=str(summary_json),
        ranked_json=None,
        model_family="xgb",
    )

    bundle = _build_bundle_from_args(args)
    assert bundle["report_kind"] == "postrun_comparison"
    assert bundle["selection_summary"]["winner"] == "best"
    assert bundle["per_class"][0]["label"] == "person"
    assert bundle["per_class_per_source"][0]["primary_source"] == "yolo"
