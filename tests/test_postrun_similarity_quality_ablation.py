from __future__ import annotations

import json
from pathlib import Path

from tools.run_postrun_similarity_quality_ablation import (
    _extract_eval_metrics,
    _resolve_best_refined_tag,
)


def test_resolve_best_refined_tag_prefers_full(tmp_path: Path) -> None:
    ranked_path = tmp_path / "postrun_sam_bias_magnitude_sweep" / "results_ranked.json"
    ranked_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_path.write_text(
        json.dumps(
            {
                "pilot": [{"tag": "pilot_tag"}],
                "full": [{"tag": "full_tag"}],
            }
        ),
        encoding="utf-8",
    )
    assert _resolve_best_refined_tag(tmp_path) == "full_tag"


def test_extract_eval_metrics_uses_post_xgb_and_coverage() -> None:
    payload = {
        "precision": 0.1,
        "recall": 0.2,
        "f1": 0.15,
        "tp": 10,
        "fp": 5,
        "fn": 7,
        "metric_tiers": {
            "post_xgb": {
                "accepted_all": {
                    "precision": 0.8,
                    "recall": 0.5,
                    "f1": 0.615384615,
                }
            },
            "post_cluster": {
                "source_attributed": {
                    "yolo_rfdetr_union": {
                        "f1": 0.4,
                    }
                }
            },
        },
        "coverage_upper_bound": {
            "candidate_all": {
                "recall_upper_bound": 0.625,
            }
        },
    }
    metrics = _extract_eval_metrics(payload)
    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 0.5
    assert round(metrics["f1"], 6) == round(0.615384615, 6)
    assert round(metrics["delta_vs_union_f1"], 6) == round(0.215384615, 6)
    assert round(metrics["coverage_preservation"], 6) == round(0.8, 6)
