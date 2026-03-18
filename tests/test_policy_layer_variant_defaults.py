from types import SimpleNamespace

from services.calibration import _resolve_policy_layer_variant
from models.schemas import CalibrationRequest


def test_resolve_policy_layer_variant_defaults_to_none():
    assert _resolve_policy_layer_variant(SimpleNamespace()) == 'none'
    assert _resolve_policy_layer_variant(SimpleNamespace(policy_layer_variant='')) == 'none'
    assert _resolve_policy_layer_variant(SimpleNamespace(policy_layer_variant='none')) == 'none'
    assert _resolve_policy_layer_variant(SimpleNamespace(policy_layer_variant='xgb')) == 'xgb'
    assert _resolve_policy_layer_variant(SimpleNamespace(policy_layer_variant='lreg')) == 'lreg'
    assert _resolve_policy_layer_variant(SimpleNamespace(policy_layer_variant='bakeoff')) == 'bakeoff'
    assert _resolve_policy_layer_variant(SimpleNamespace(policy_layer_variant='weird')) == 'none'


def test_calibration_request_defaults_recipe_mode_auto_and_policy_layer_none():
    payload = CalibrationRequest(dataset_id="demo")
    assert payload.recipe_mode == "auto"
    assert payload.policy_layer_variant == "none"
