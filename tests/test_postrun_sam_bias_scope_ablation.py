import pytest

from tools import run_postrun_sam_bias_scope_ablation as scope_ablation


def test_parse_scopes_defaults_to_sam_only():
    assert scope_ablation._parse_scopes("") == ["sam_only"]
    assert scope_ablation._parse_scopes("sam_only,primary_source") == [
        "sam_only",
        "primary_source",
    ]


def test_summaries_compute_delta_vs_primary_source_baseline():
    rows = [
        {
            "scope": "primary_source",
            "view": "intersection",
            "precision": 0.89,
            "recall": 0.76,
            "f1": 0.82,
            "delta_vs_union_f1": 0.06,
            "coverage_preservation": 0.80,
        },
        {
            "scope": "sam_only",
            "view": "intersection",
            "precision": 0.90,
            "recall": 0.77,
            "f1": 0.825,
            "delta_vs_union_f1": 0.065,
            "coverage_preservation": 0.81,
        },
    ]

    ranked = scope_ablation._summaries(rows)

    assert ranked["intersection"][0]["scope"] == "sam_only"
    assert ranked["intersection"][0]["mean_delta_vs_baseline_f1"] == pytest.approx(0.005)
    assert ranked["intersection"][1]["scope"] == "primary_source"
    assert ranked["intersection"][1]["mean_delta_vs_baseline_f1"] == pytest.approx(0.0)
