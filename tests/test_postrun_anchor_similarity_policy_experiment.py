from __future__ import annotations

import json
from pathlib import Path

from tools.run_postrun_anchor_similarity_policy_experiment import (
    _resolve_base_source,
    _resolve_best_similarity,
    _resolve_baseline_to_beat,
)


def test_resolve_best_similarity_requires_positive_delta() -> None:
    assert _resolve_best_similarity({"ranked": [{"tag": "a0p5", "mean_delta_f1": -0.001}]}) is None
    best = _resolve_best_similarity({"ranked": [{"tag": "a0p5", "mean_delta_f1": 0.002, "mean_f1": 0.84}]})
    assert best is not None
    assert best["tag"] == "a0p5"


def test_resolve_base_source_prefers_positive_similarity_winner(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    refined_tag = "text_m1p4__sim_m1p2"
    for seed in ("42", "1337", "2025"):
        base_dir = run_root / "postrun_sam_bias_magnitude_sweep" / "full" / refined_tag / "full" / f"seed_{seed}"
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "eval.json").write_text(json.dumps({"f1": 0.84, "precision": 0.88, "recall": 0.8}), encoding="utf-8")
    sim_summary = {
        "ranked": [
            {"tag": "a0p5", "mean_f1": 0.845, "mean_delta_f1": 0.003},
        ]
    }
    sim_path = run_root / "postrun_similarity_quality_full_window_eval" / "results_summary.json"
    sim_path.parent.mkdir(parents=True, exist_ok=True)
    sim_path.write_text(json.dumps(sim_summary), encoding="utf-8")

    base_source = _resolve_base_source(run_root, refined_tag=refined_tag)
    assert base_source["name"] == "similarity_quality"
    assert base_source["tag"] == "a0p5"


def test_resolve_baseline_to_beat_prefers_best_mean_f1(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    refined_tag = "text_m1p4__sim_m1p2"
    for seed, f1 in {"42": 0.84, "1337": 0.841, "2025": 0.842}.items():
        base_dir = run_root / "postrun_sam_bias_magnitude_sweep" / "full" / refined_tag / "full" / f"seed_{seed}"
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "eval.json").write_text(json.dumps({"f1": f1, "precision": 0.88, "recall": 0.8}), encoding="utf-8")
    learned_summary = {
        "rows": [
            {"seed": "42", "metrics": {"f1": 0.846}},
            {"seed": "1337", "metrics": {"f1": 0.845}},
            {"seed": "2025", "metrics": {"f1": 0.847}},
        ],
        "metrics_mean": {"mean_f1": 0.846},
    }
    learned_path = run_root / "postrun_learned_xgb_full_window_eval" / "results_summary.json"
    learned_path.parent.mkdir(parents=True, exist_ok=True)
    learned_path.write_text(json.dumps(learned_summary), encoding="utf-8")

    best = _resolve_baseline_to_beat(run_root, refined_tag=refined_tag)
    assert best["name"] == "learned_xgb"
    assert abs(best["mean_f1"] - 0.846) < 1e-9
