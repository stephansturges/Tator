import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import _refine_steps_prompt_subset_seed_stage  # noqa: E402


def test_refine_prompt_subset_can_drop_redundant_step():
    prompt_stats = []
    selected = [
        {"prompt": "a", "matched_keys": {1, 2}, "fps": 10, "precision": 2 / 12, "selected_seed_threshold": 0.5},
        {"prompt": "b", "matched_keys": {2}, "fps": 100, "precision": 1 / 101, "selected_seed_threshold": 0.5},
    ]
    refined, info = _refine_steps_prompt_subset_seed_stage(
        prompt_stats,
        selected,
        max_steps=6,
        target_precision=0.1,
        max_iters=0,
        top_k=3,
        base_seed_threshold=0.05,
    )
    assert info.get("enabled") is True
    assert [c.get("prompt") for c in refined] == ["a"]
    assert any(h.get("op") == "drop_redundant" and h.get("dropped") == "b" for h in info.get("history") or [])


def test_refine_prompt_subset_can_swap_to_improve_coverage():
    prompt_stats = [
        {
            "prompt": "c",
            "gt_best_scores": {3: 0.9, 4: 0.9},
            "seed_threshold_recommended": 0.5,
            "seed_threshold_curve": [{"threshold": 0.5, "matches": 2, "fps": 1, "precision": 2 / 3}],
        }
    ]
    selected = [
        {"prompt": "a", "matched_keys": {1, 2}, "fps": 10, "precision": 2 / 12, "selected_seed_threshold": 0.5},
        {"prompt": "b", "matched_keys": {3}, "fps": 1, "precision": 1 / 2, "selected_seed_threshold": 0.5},
    ]
    refined, info = _refine_steps_prompt_subset_seed_stage(
        prompt_stats,
        selected,
        max_steps=2,
        target_precision=0.1,
        max_iters=2,
        top_k=5,
        base_seed_threshold=0.05,
    )
    prompts = {str(c.get("prompt") or "") for c in refined}
    assert "c" in prompts
    assert "b" not in prompts
    assert info.get("enabled") is True

