from __future__ import annotations

from tools.run_postrun_similarity_policy_followup import _choose_similarity_tag, _compare_metrics


def test_choose_similarity_tag_uses_first_positive_pilot_row() -> None:
    payload = {
        "pilot": [
            {"tag": "a0p2", "mean_delta_f1": -0.001},
            {"tag": "a0p35", "mean_delta_f1": 0.0},
            {"tag": "a0p5", "mean_delta_f1": 0.002},
            {"tag": "a0p8", "mean_delta_f1": 0.003},
        ]
    }
    assert _choose_similarity_tag(payload) == "a0p5"


def test_compare_metrics_reports_signed_deltas() -> None:
    metrics = {"f1": 0.83, "tp": 100, "fp": 10, "fn": 20, "precision": 0.91, "recall": 0.83}
    baseline = {"f1": 0.81, "tp": 98, "fp": 14, "fn": 22, "precision": 0.875, "recall": 0.816}
    compare = _compare_metrics(metrics, baseline)
    assert round(compare["delta_f1"], 6) == 0.02
    assert compare["delta_tp"] == 2.0
    assert compare["delta_fp"] == -4.0
    assert compare["delta_fn"] == -2.0
