import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import _select_steps_from_seed_prompt_stats  # noqa: E402


def test_step_selection_changes_with_target_precision():
    # Prompt A: high coverage but noisy unless threshold is high.
    prompt_a = {
        "prompt": "A",
        "gt_best_scores": {1: 0.95, 2: 0.95, 3: 0.2, 4: 0.2},
        "seed_threshold_curve": [
            {"threshold": 0.1, "matches": 4, "fps": 6, "precision": 0.4},
            {"threshold": 0.9, "matches": 2, "fps": 0, "precision": 1.0},
        ],
    }
    # Prompt B: clean but only covers the remaining GTs.
    prompt_b = {
        "prompt": "B",
        "gt_best_scores": {3: 0.95, 4: 0.95},
        "seed_threshold_curve": [
            {"threshold": 0.1, "matches": 2, "fps": 0, "precision": 1.0},
            {"threshold": 0.9, "matches": 2, "fps": 0, "precision": 1.0},
        ],
    }

    selected_lo, _info_lo = _select_steps_from_seed_prompt_stats([prompt_a, prompt_b], max_steps=6, target_precision=0.3)
    assert [s["prompt"] for s in selected_lo] == ["A"]
    assert abs(float(selected_lo[0].get("selected_seed_threshold") or 0.0) - 0.1) < 1e-6

    selected_hi, _info_hi = _select_steps_from_seed_prompt_stats([prompt_a, prompt_b], max_steps=6, target_precision=0.9)
    prompts_hi = [s["prompt"] for s in selected_hi]
    assert set(prompts_hi) == {"A", "B"}
    by_prompt = {s["prompt"]: s for s in selected_hi}
    assert abs(float(by_prompt["A"].get("selected_seed_threshold") or 0.0) - 0.9) < 1e-6
    assert abs(float(by_prompt["B"].get("selected_seed_threshold") or 0.0) - 0.1) < 1e-6


def test_step_selection_picks_at_most_one_threshold_per_prompt():
    prompt = {
        "prompt": "A",
        "gt_best_scores": {1: 0.95, 2: 0.2},
        "seed_threshold_curve": [
            {"threshold": 0.1, "matches": 2, "fps": 50, "precision": 2 / 52},
            {"threshold": 0.9, "matches": 1, "fps": 0, "precision": 1.0},
        ],
    }
    selected, _info = _select_steps_from_seed_prompt_stats([prompt], max_steps=2, target_precision=0.9, max_candidates_per_prompt=4)
    assert len(selected) == 1
    assert selected[0]["prompt"] == "A"
    assert abs(float(selected[0].get("selected_seed_threshold") or 0.0) - 0.9) < 1e-6
