import sys
from pathlib import Path

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import _classify_agent_recipe_mode, _normalize_agent_recipe_execution_plan, _validate_agent_recipe_structure


def test_validate_agent_recipe_schema_v2_steps_accepts_minimal_prompt_step():
    recipe = {
        "schema_version": 2,
        "mode": "sam3_steps",
        "steps": [{"prompt": "car"}],
    }
    _validate_agent_recipe_structure(recipe)
    assert _classify_agent_recipe_mode(recipe) == "sam3_steps"
    plan = _normalize_agent_recipe_execution_plan(recipe)
    assert plan["mode"] == "sam3_steps"
    assert plan["schema_version"] == 2


def test_validate_agent_recipe_schema_v2_steps_does_not_misclassify_as_greedy_when_positives_exist():
    recipe = {
        "schema_version": 2,
        "mode": "sam3_steps",
        "positives": [],
        "steps": [{"prompt": "car"}],
    }
    _validate_agent_recipe_structure(recipe)
    assert _classify_agent_recipe_mode(recipe) == "sam3_steps"


def test_validate_agent_recipe_legacy_greedy_without_mode_is_still_supported():
    recipe = {
        "text_prompts": ["car", "automobile"],
        "positives": [],
        "params": {"seed_threshold": 0.05},
    }
    _validate_agent_recipe_structure(recipe)
    assert _classify_agent_recipe_mode(recipe) == "sam3_greedy"


def test_validate_agent_recipe_schema_v2_steps_rejects_step_without_prompt():
    recipe = {"schema_version": 2, "mode": "sam3_steps", "steps": [{}]}
    try:
        _validate_agent_recipe_structure(recipe)
    except HTTPException:
        return
    assert False, "expected validation to reject missing step prompts"

