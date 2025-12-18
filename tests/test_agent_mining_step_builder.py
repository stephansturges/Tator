import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import (  # noqa: E402
    AgentMiningRequest,
    _build_steps_recipe_step_list_from_selected_stats,
)


def test_build_steps_recipe_step_list_honors_per_prompt_seed_thresholds():
    payload = AgentMiningRequest(dataset_id="ds")
    selected = [
        {"prompt": "car", "seed_threshold": 0.2},
        {"prompt": "automobile", "selected_seed_threshold": 0.1},
    ]
    prompts, steps = _build_steps_recipe_step_list_from_selected_stats(selected, prompts_fallback=["fallback"], payload=payload)
    assert prompts == ["car", "automobile"]
    assert steps[0]["prompt"] == "car"
    assert abs(float(steps[0]["seed_threshold"]) - 0.2) < 1e-9
    assert steps[1]["prompt"] == "automobile"
    assert abs(float(steps[1]["seed_threshold"]) - 0.1) < 1e-9


def test_build_steps_recipe_step_list_falls_back_to_payload_seed_threshold():
    payload = AgentMiningRequest(dataset_id="ds", seed_threshold=0.07)
    selected = [{"prompt": "car"}]
    prompts, steps = _build_steps_recipe_step_list_from_selected_stats(selected, prompts_fallback=None, payload=payload)
    assert prompts == ["car"]
    assert abs(float(steps[0]["seed_threshold"]) - 0.07) < 1e-9


def test_build_steps_recipe_step_list_uses_fallback_prompt_when_selection_empty():
    payload = AgentMiningRequest(dataset_id="ds", seed_threshold=0.07)
    prompts, steps = _build_steps_recipe_step_list_from_selected_stats([], prompts_fallback=["light_vehicle"], payload=payload)
    assert prompts == ["light_vehicle"]
    assert steps[0]["prompt"] == "light_vehicle"
    assert abs(float(steps[0]["seed_threshold"]) - 0.07) < 1e-9

