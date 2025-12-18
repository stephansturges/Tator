import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import _update_best_clip_head_sweep_summary  # noqa: E402


def _run_sweep(candidates, *, target_precision: float):
    best = None
    best_key = None
    for c in candidates:
        best, best_key = _update_best_clip_head_sweep_summary(
            best_summary=best,
            best_key=best_key,
            total_gt=10,
            total_images=5,
            matched=c["matched"],
            fps=c["fps"],
            duplicates=c.get("duplicates", 0),
            preds=c.get("preds", c["matched"] + c["fps"]),
            det_images=c.get("det_images", 1),
            min_prob=c["min_prob"],
            margin=c.get("margin", 0.0),
            target_precision=target_precision,
        )
    return best


def test_head_sweep_prefers_meeting_target_then_matches():
    candidates = [
        {"min_prob": 0.1, "matched": 8, "fps": 8},  # precision 0.5
        {"min_prob": 0.5, "matched": 6, "fps": 0},  # precision 1.0
        {"min_prob": 0.9, "matched": 2, "fps": 0},  # precision 1.0
    ]
    best = _run_sweep(candidates, target_precision=0.6)
    assert best is not None
    assert abs(float(best["clip_head_min_prob"]) - 0.5) < 1e-9
    assert best["clip_head_meets_target_precision"] is True


def test_head_sweep_falls_back_to_best_precision_when_target_unreachable():
    candidates = [
        {"min_prob": 0.1, "matched": 8, "fps": 8},  # precision 0.5
        {"min_prob": 0.5, "matched": 6, "fps": 1},  # precision ~0.857
        {"min_prob": 0.9, "matched": 7, "fps": 3},  # precision 0.7
    ]
    best = _run_sweep(candidates, target_precision=0.95)
    assert best is not None
    assert abs(float(best["clip_head_min_prob"]) - 0.5) < 1e-9
    assert best["clip_head_meets_target_precision"] is False

