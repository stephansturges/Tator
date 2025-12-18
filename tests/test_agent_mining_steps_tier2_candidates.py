import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import _build_steps_tier2_candidate_grid  # noqa: E402


def test_steps_tier2_candidate_grid_is_deterministic_and_bounded():
    c1 = _build_steps_tier2_candidate_grid(base_seed_dedupe_iou=0.9, base_dedupe_iou=0.5, max_trials=9)
    c2 = _build_steps_tier2_candidate_grid(base_seed_dedupe_iou=0.9, base_dedupe_iou=0.5, max_trials=9)
    assert c1 == c2
    assert 1 <= len(c1) <= 9
    assert any(abs(float(c.get("seed_dedupe_iou")) - 0.9) < 1e-9 and abs(float(c.get("dedupe_iou")) - 0.5) < 1e-9 for c in c1)
    for cand in c1:
        assert 0.0 <= float(cand.get("seed_dedupe_iou")) <= 1.0
        assert 0.0 <= float(cand.get("dedupe_iou")) <= 1.0

