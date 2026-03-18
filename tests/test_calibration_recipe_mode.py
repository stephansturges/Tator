from __future__ import annotations

from types import SimpleNamespace

from services.calibration import (
    _apply_canonical_recipe_to_payload,
    _resolve_lane_selection,
    _resolve_recipe_mode,
)


def test_resolve_recipe_mode_defaults_to_auto() -> None:
    assert _resolve_recipe_mode(SimpleNamespace()) == "auto"
    assert _resolve_recipe_mode(SimpleNamespace(recipe_mode="")) == "auto"
    assert _resolve_recipe_mode(SimpleNamespace(recipe_mode="reuse_only")) == "reuse_only"
    assert _resolve_recipe_mode(SimpleNamespace(recipe_mode="force_rediscover")) == "force_rediscover"
    assert _resolve_recipe_mode(SimpleNamespace(recipe_mode="weird")) == "auto"


def test_resolve_lane_selection_defaults_to_window() -> None:
    assert _resolve_lane_selection(SimpleNamespace()) == "window"
    assert _resolve_lane_selection(SimpleNamespace(lane_selection="")) == "window"
    assert _resolve_lane_selection(SimpleNamespace(lane_selection="nonwindow")) == "nonwindow"
    assert _resolve_lane_selection(SimpleNamespace(lane_selection="compare_both")) == "compare_both"
    assert _resolve_lane_selection(SimpleNamespace(lane_selection="weird")) == "window"


def test_apply_canonical_recipe_to_payload_overrides_executor_knobs() -> None:
    payload = SimpleNamespace(
        sam3_text_window_extension=True,
        similarity_window_extension=True,
        apply_default_ensemble_policy=True,
        ensemble_policy_json=None,
        split_head_by_support=None,
        train_sam3_text_quality=True,
        sam3_text_quality_alpha=None,
        train_sam3_similarity_quality=None,
        sam3_similarity_quality_alpha=None,
        policy_layer_variant="none",
        image_embed_proj_dim=0,
        image_embed_proj_seed=4242,
    )
    recipe = {
        "winner_lane": "nonwindow",
        "scenario": {
            "split_head": True,
            "train_sam3_text_quality": True,
            "sam3_text_quality_alpha": 0.5,
            "train_sam3_similarity_quality": False,
            "sam3_similarity_quality_alpha": None,
        },
        "policy": {"sam_bias_scope": "sam_only"},
        "xgb_hparams": {"max_depth": 8, "n_estimators": 900},
    }

    updated = _apply_canonical_recipe_to_payload(payload, recipe)

    assert updated.sam3_text_window_extension is False
    assert updated.similarity_window_extension is False
    assert updated.apply_default_ensemble_policy is False
    assert updated.split_head_by_support is True
    assert updated.sam3_text_quality_alpha == 0.5
    assert updated.policy_layer_variant == "none"
    assert updated.image_embed_proj_dim == 0
    assert updated.image_embed_proj_seed == 4242
    assert updated.xgb_max_depth == 8
    assert updated.xgb_n_estimators == 900


def test_apply_canonical_recipe_to_payload_preserves_explicit_policy_layer_override() -> None:
    payload = SimpleNamespace(
        sam3_text_window_extension=True,
        similarity_window_extension=True,
        apply_default_ensemble_policy=True,
        ensemble_policy_json=None,
        split_head_by_support=None,
        train_sam3_text_quality=True,
        sam3_text_quality_alpha=None,
        train_sam3_similarity_quality=None,
        sam3_similarity_quality_alpha=None,
        policy_layer_variant="xgb",
        image_embed_proj_dim=0,
        image_embed_proj_seed=4242,
    )
    recipe = {
        "winner_lane": "window",
        "scenario": {
            "split_head": False,
            "train_sam3_text_quality": True,
            "sam3_text_quality_alpha": 0.8,
            "train_sam3_similarity_quality": True,
            "sam3_similarity_quality_alpha": 0.5,
        },
        "policy": {"sam_bias_scope": "sam_only"},
        "xgb_hparams": {"max_depth": 8, "n_estimators": 900},
    }

    updated = _apply_canonical_recipe_to_payload(payload, recipe)

    assert updated.policy_layer_variant == "xgb"
