import pytest
from fastapi import HTTPException

from services.prepass_recipes import (
    _list_prepass_recipes_impl,
    _load_prepass_recipe_meta,
    _save_prepass_recipe_impl,
    _validate_prepass_recipe_config_impl,
    _write_prepass_recipe_meta,
)


def test_prepass_recipe_config_rejects_legacy_cross_iou():
    with pytest.raises(HTTPException) as exc_info:
        _validate_prepass_recipe_config_impl({"cross_iou": 0.8, "enable_yolo": True})
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prepass_recipe_legacy_cross_iou_unsupported"


def test_prepass_recipe_config_accepts_current_cross_class_dedupe_fields():
    config = {
        "cross_class_dedupe_enabled": True,
        "cross_class_dedupe_iou": 0.8,
        "enable_yolo": True,
    }
    normalized = _validate_prepass_recipe_config_impl(config)
    assert normalized == config
    assert normalized is not config


def test_prepass_recipe_config_rejects_non_dict_payload():
    with pytest.raises(HTTPException) as exc_info:
        _validate_prepass_recipe_config_impl(["bad", "config"])
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prepass_recipe_invalid_config"
    with pytest.raises(HTTPException) as exc_info_2:
        _validate_prepass_recipe_config_impl([])
    assert exc_info_2.value.status_code == 400
    assert exc_info_2.value.detail == "prepass_recipe_invalid_config"


def test_prepass_recipe_config_allows_missing_config_as_empty():
    assert _validate_prepass_recipe_config_impl(None) == {}


def test_save_prepass_recipe_preserves_created_at(tmp_path):
    payload = {
        "name": "Recipe A",
        "description": "first",
        "config": {"enable_yolo": True},
        "glossary": "",
    }
    first = _save_prepass_recipe_impl(
        payload,
        recipe_id="recipe_a",
        prepass_schema_version=1,
        recipes_root=tmp_path,
        sanitize_run_id_fn=lambda value: value,
        normalize_glossary_fn=lambda value: value or "",
        write_meta_fn=_write_prepass_recipe_meta,
    )
    updated = _save_prepass_recipe_impl(
        {**payload, "description": "second"},
        recipe_id="recipe_a",
        prepass_schema_version=1,
        recipes_root=tmp_path,
        sanitize_run_id_fn=lambda value: value,
        normalize_glossary_fn=lambda value: value or "",
        write_meta_fn=_write_prepass_recipe_meta,
    )
    assert updated["created_at"] == first["created_at"]
    assert updated["updated_at"] >= first["updated_at"]


def test_load_prepass_recipe_meta_supports_legacy_recipe_json(tmp_path):
    recipe_dir = tmp_path / "legacy_recipe"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    (recipe_dir / "recipe.json").write_text(
        '{"id":"legacy_recipe","name":"Legacy","description":"x","created_at":1.0,"updated_at":2.0}',
        encoding="utf-8",
    )
    loaded = _load_prepass_recipe_meta(recipe_dir)
    assert loaded.get("id") == "legacy_recipe"
    assert loaded.get("name") == "Legacy"


def test_list_prepass_recipes_supports_legacy_recipe_json(tmp_path):
    recipe_dir = tmp_path / "legacy_recipe"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    (recipe_dir / "recipe.json").write_text(
        '{"id":"legacy_recipe","name":"Legacy","description":"x","created_at":1.0,"updated_at":2.0}',
        encoding="utf-8",
    )
    listed = _list_prepass_recipes_impl(recipes_root=tmp_path, meta_filename="prepass.meta.json")
    assert listed
    assert listed[0]["id"] == "legacy_recipe"
    assert listed[0]["name"] == "Legacy"


def test_list_prepass_recipes_handles_mixed_timestamp_types(tmp_path):
    first = tmp_path / "first"
    first.mkdir(parents=True, exist_ok=True)
    (first / "prepass.meta.json").write_text(
        '{"id":"first","name":"First","description":"","created_at":1.0,"updated_at":1.0}',
        encoding="utf-8",
    )

    second = tmp_path / "second"
    second.mkdir(parents=True, exist_ok=True)
    (second / "prepass.meta.json").write_text(
        '{"id":"second","name":"Second","description":"","created_at":"2026-02-09T00:00:00Z","updated_at":"2026-02-09T01:00:00Z"}',
        encoding="utf-8",
    )

    third = tmp_path / "third"
    third.mkdir(parents=True, exist_ok=True)
    (third / "prepass.meta.json").write_text(
        '{"id":"third","name":"Third","description":"","created_at":"bad","updated_at":"bad"}',
        encoding="utf-8",
    )

    listed = _list_prepass_recipes_impl(recipes_root=tmp_path, meta_filename="prepass.meta.json")
    assert [item["id"] for item in listed] == ["second", "first", "third"]
