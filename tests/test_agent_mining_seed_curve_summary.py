import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import _summarize_seed_threshold_curve_for_prompt  # noqa: E402


def test_seed_curve_summary_is_json_serializable_and_bounded():
    gt_best_scores = {"a": 0.9, "b": 0.6, "c": 0.55}
    fp_scores = [0.8, 0.4, 0.2]
    summary = _summarize_seed_threshold_curve_for_prompt(
        gt_best_scores=gt_best_scores,
        fp_scores=fp_scores,
        base_seed_threshold=0.05,
        curve_limit=12,
    )
    assert isinstance(summary, dict)
    assert "seed_threshold_curve" in summary
    assert len(summary["seed_threshold_curve"]) <= 12
    assert 0.0 <= float(summary["seed_threshold_base"]) <= 1.0
    assert 0.0 <= float(summary["seed_threshold_recommended"]) <= 1.0
    json.dumps(summary)

