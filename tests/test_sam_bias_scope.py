import json

from tools import eval_ensemble_xgb_dedupe as eval_xgb
from tools import score_ensemble_candidates_xgb as score_xgb
from tools import train_ensemble_xgb as train_xgb


def test_primary_source_scope_keeps_bias_for_detector_backed_sam_primary():
    policy = {"sam_bias_scope": "primary_source"}

    assert eval_xgb._should_apply_source_bias(
        policy,
        primary_source="sam3_text",
        has_detector_support=True,
    )
    assert score_xgb._should_apply_source_bias(
        policy,
        primary_source="sam3_text",
        has_detector_support=True,
    )


def test_sam_only_scope_skips_bias_for_detector_backed_sam_primary():
    policy = {"sam_bias_scope": "sam_only"}

    assert not eval_xgb._should_apply_source_bias(
        policy,
        primary_source="sam3_text",
        has_detector_support=True,
    )
    assert not score_xgb._should_apply_source_bias(
        policy,
        primary_source="sam3_text",
        has_detector_support=True,
    )


def test_sam_only_scope_keeps_bias_for_true_sam_only_candidates():
    policy = {"sam_bias_scope": "sam_only"}

    assert eval_xgb._should_apply_source_bias(
        policy,
        primary_source="sam3_similarity",
        has_detector_support=False,
    )
    assert score_xgb._should_apply_source_bias(
        policy,
        primary_source="sam3_similarity",
        has_detector_support=False,
    )


def test_sam_only_scope_does_not_affect_non_sam_sources():
    policy = {"sam_bias_scope": "sam_only"}

    assert not eval_xgb._should_apply_source_bias(
        policy,
        primary_source="rfdetr",
        has_detector_support=True,
    )
    assert not score_xgb._should_apply_source_bias(
        policy,
        primary_source="rfdetr",
        has_detector_support=True,
    )


def test_policy_loaders_accept_long_inline_json_payloads():
    policy = {
        "sam_bias_scope": "sam_only",
        "threshold_by_class_override": {f"class_{idx}": 0.1 + (idx * 0.001) for idx in range(80)},
    }
    raw = json.dumps(policy, separators=(",", ":"))
    assert len(raw) > 255

    assert train_xgb._load_policy(raw) == policy
    assert eval_xgb._load_policy(raw) == policy
    assert score_xgb._load_policy(raw) == policy
