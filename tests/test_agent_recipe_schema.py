import sys
from pathlib import Path

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import _classify_agent_recipe_mode, _normalize_agent_recipe_execution_plan, _validate_agent_recipe_structure
import localinferenceapi


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


def test_persist_agent_recipe_preserves_schema_v2_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(localinferenceapi, "AGENT_MINING_RECIPES_ROOT", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    recipe = {
        "schema_version": 2,
        "mode": "sam3_steps",
        "optimizer": {"algorithm": "sam3_steps_v2", "version": 1},
        "steps": [{"prompt": "car", "seed_threshold": 0.05}],
    }
    saved = localinferenceapi._persist_agent_recipe(
        dataset_id=None,
        class_id=1,
        class_name="light_vehicle",
        label="test",
        recipe=recipe,
        meta_overrides={
            "dataset_signature": "test_sig",
            "labelmap_hash": "test_hash",
            "labelmap": ["light_vehicle"],
        },
    )
    assert saved["recipe"]["mode"] == "sam3_steps"
    assert saved["recipe"]["steps"][0]["prompt"] == "car"
    assert saved["recipe"]["optimizer"]["algorithm"] == "sam3_steps_v2"


def test_import_agent_recipe_zip_preserves_top_level_params(tmp_path, monkeypatch):
    monkeypatch.setattr(localinferenceapi, "AGENT_MINING_RECIPES_ROOT", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    data = {
        "id": "ar_source",
        "dataset_id": "",
        "dataset_signature": "sig",
        "labelmap_hash": "hash",
        "labelmap": ["light_vehicle"],
        "class_id": 1,
        "class_name": "light_vehicle",
        "label": "imported",
        "created_at": 0.0,
        "params": {"seed_threshold": 0.33, "dedupe_iou": 0.77},
        "recipe": {
            "schema_version": 2,
            "mode": "sam3_steps",
            "optimizer": {"algorithm": "sam3_steps_v2", "version": 1, "val_split_hash": "abc123"},
            "steps": [{"prompt": "car", "seed_threshold": 0.05}],
        },
    }

    import json
    import zipfile
    from io import BytesIO

    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("recipe.json", json.dumps(data, ensure_ascii=False, indent=2))

    _, persisted = localinferenceapi._import_agent_recipe_zip_bytes(bio.getvalue())
    assert persisted["recipe"]["mode"] == "sam3_steps"
    assert persisted["recipe"]["steps"][0]["prompt"] == "car"
    assert persisted["params"]["seed_threshold"] == 0.33
    assert persisted["params"]["dedupe_iou"] == 0.77
    assert persisted["recipe"]["optimizer"]["val_split_hash"] == "abc123"
