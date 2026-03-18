import pytest

from tools import run_postrun_sam_bias_magnitude_sweep as bias_sweep


def test_parse_bias_grid_uses_defaults_and_dedupes():
    assert bias_sweep._parse_bias_grid("", default=[-1.0, -0.8]) == [-1.0, -0.8]
    assert bias_sweep._parse_bias_grid("-1.0,-0.8,-1.0", default=[]) == [-1.0, -0.8]


def test_promote_tags_applies_delta_gate():
    ranked = [
        {"tag": "a", "mean_delta_vs_baseline_f1": 0.01},
        {"tag": "b", "mean_delta_vs_baseline_f1": 0.0},
        {"tag": "c", "mean_delta_vs_baseline_f1": -0.001},
    ]
    assert bias_sweep._promote_tags(ranked, limit=2, min_delta=0.0) == ["a", "b"]
    assert bias_sweep._promote_tags(ranked, limit=2, min_delta=0.001) == ["a"]


def test_summarize_uses_view_specific_baseline():
    rows = [
        {
            "tag": "text_m1p0__sim_m0p8",
            "text_bias": -1.0,
            "sim_bias": -0.8,
            "view": "intersection",
            "precision": 0.88,
            "recall": 0.80,
            "f1": 0.838,
            "delta_vs_union_f1": 0.08,
            "coverage_preservation": 0.84,
        },
        {
            "tag": "text_m1p2__sim_m1p0",
            "text_bias": -1.2,
            "sim_bias": -1.0,
            "view": "intersection",
            "precision": 0.89,
            "recall": 0.80,
            "f1": 0.842,
            "delta_vs_union_f1": 0.082,
            "coverage_preservation": 0.84,
        },
    ]
    baseline = {"intersection": {"f1": 0.839, "precision": 0.0, "recall": 0.0, "delta_vs_union_f1": 0.0, "coverage_preservation": 0.0}}
    ranked = bias_sweep._summarize(rows, baseline_by_view=baseline)
    assert ranked["intersection"][0]["tag"] == "text_m1p2__sim_m1p0"
    assert ranked["intersection"][0]["mean_delta_vs_baseline_f1"] == pytest.approx(0.003)
