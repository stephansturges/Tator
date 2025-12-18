import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import (  # noqa: E402
    AgentMiningRequest,
    _normalize_steps_for_head_tuning,
    _successive_halving_search,
)


def test_successive_halving_evaluates_all_then_prunes():
    calls = []
    candidates = [0, 1, 2, 3]

    def evaluator(cand, budget):
        calls.append((cand, budget))
        return (cand, budget), {"cand": cand, "budget": budget}

    best, _history = _successive_halving_search(
        candidates=candidates,
        budgets=[10, 20],
        evaluator=evaluator,
        keep_ratio=0.5,
    )
    assert best == 3

    stage1 = {c for c, b in calls if b == 10}
    stage2 = [c for c, b in calls if b == 20]
    assert stage1 == {0, 1, 2, 3}
    assert stage2 == [3, 2]  # top half by key, stable order


def test_successive_halving_rejects_non_increasing_budgets():
    raised = False
    try:
        _successive_halving_search(
            candidates=[0, 1],
            budgets=[10, 10],
            evaluator=lambda cand, budget: ((0,), None),
        )
    except Exception:
        raised = True
    assert raised


def test_normalize_steps_for_head_tuning_honors_per_step_values():
    payload = AgentMiningRequest(dataset_id="ds", seed_threshold=0.05, expand_threshold=0.3, steps_max_visual_seeds_per_step=5)
    steps = [
        {"enabled": False, "prompt": "skip", "seed_threshold": 0.9},
        {
            "enabled": True,
            "prompt": "car",
            "seed_threshold": 0.0,
            "expand_threshold": 0.7,
            "max_visual_seeds": 1,
            "seed_dedupe_iou": 0.8,
            "dedupe_iou": 0.4,
            "max_results": 12,
        },
    ]
    norm = _normalize_steps_for_head_tuning(steps, payload=payload)
    assert len(norm) == 1
    assert norm[0]["prompt"] == "car"
    assert abs(float(norm[0]["seed_threshold"]) - 0.0) < 1e-9
    assert abs(float(norm[0]["expand_threshold"]) - 0.7) < 1e-9
    assert int(norm[0]["max_visual_seeds"]) == 1
    assert abs(float(norm[0]["seed_dedupe_iou"]) - 0.8) < 1e-9
    assert abs(float(norm[0]["dedupe_iou"]) - 0.4) < 1e-9
    assert int(norm[0]["max_results"]) == 12


def test_agent_mining_request_validates_tier1_bounds():
    # eval_cap has ge=10
    raised = False
    try:
        AgentMiningRequest(dataset_id="ds", steps_optimize_tier1_eval_cap=0)
    except Exception:
        raised = True
    assert raised

