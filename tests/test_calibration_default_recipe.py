from types import SimpleNamespace

from services import calibration


def test_windowed_calibration_defaults_follow_winning_recipe():
    payload = SimpleNamespace(
        ensemble_policy_json=None,
        apply_default_ensemble_policy=True,
        split_head_by_support=None,
        sam3_text_quality_alpha=None,
        train_sam3_similarity_quality=None,
        sam3_similarity_quality_alpha=None,
    )
    sam3_window = {"enabled": True, "mode": "grid"}
    sim_window = {"enabled": True, "mode": "grid"}

    policy_json = calibration._resolve_default_ensemble_policy_json(
        payload,
        sam3_text_window_cfg=sam3_window,
        similarity_window_cfg=sim_window,
    )

    assert policy_json is not None
    assert '"sam_only_min_prob_default":0.15' in policy_json
    assert '"consensus_iou_default":0.7' in policy_json
    assert '"sam_bias_scope":"sam_only"' in policy_json
    assert '"sam3_text":{"__default__":-1.4}' in policy_json
    assert '"sam3_similarity":{"__default__":-1.2}' in policy_json
    assert (
        calibration._resolve_default_split_head_by_support(
            payload,
            sam3_text_window_cfg=sam3_window,
            similarity_window_cfg=sim_window,
        )
        is False
    )
    assert calibration._resolve_default_sam3_text_quality_alpha(
        payload,
        sam3_text_window_cfg=sam3_window,
        similarity_window_cfg=sim_window,
    ) == 0.8
    assert (
        calibration._resolve_default_train_sam3_similarity_quality(
            payload,
            sam3_text_window_cfg=sam3_window,
            similarity_window_cfg=sim_window,
        )
        is True
    )
    assert calibration._resolve_default_sam3_similarity_quality_alpha(
        payload,
        sam3_text_window_cfg=sam3_window,
        similarity_window_cfg=sim_window,
    ) == 0.5


def test_nonwindowed_calibration_defaults_keep_nonwindow_fallback():
    payload = SimpleNamespace(
        ensemble_policy_json=None,
        apply_default_ensemble_policy=True,
        split_head_by_support=None,
        sam3_text_quality_alpha=None,
        train_sam3_similarity_quality=None,
        sam3_similarity_quality_alpha=None,
    )
    sam3_window = {"enabled": False, "mode": "grid"}
    sim_window = {"enabled": False, "mode": "grid"}

    policy_json = calibration._resolve_default_ensemble_policy_json(
        payload,
        sam3_text_window_cfg=sam3_window,
        similarity_window_cfg=sim_window,
    )

    assert policy_json is not None
    assert '"sam_only_min_prob_default":0.15' in policy_json
    assert '"consensus_iou_default":0.7' in policy_json
    assert '"sam_bias_scope":"sam_only"' in policy_json
    assert '"sam3_text":{"__default__":-1.4}' in policy_json
    assert '"sam3_similarity":{"__default__":-1.2}' in policy_json
    assert (
        calibration._resolve_default_split_head_by_support(
            payload,
            sam3_text_window_cfg=sam3_window,
            similarity_window_cfg=sim_window,
        )
        is True
    )
    assert calibration._resolve_default_sam3_text_quality_alpha(
        payload,
        sam3_text_window_cfg=sam3_window,
        similarity_window_cfg=sim_window,
    ) == 0.5
    assert (
        calibration._resolve_default_train_sam3_similarity_quality(
            payload,
            sam3_text_window_cfg=sam3_window,
            similarity_window_cfg=sim_window,
        )
        is False
    )


def test_explicit_calibration_overrides_win_over_default_recipe():
    payload = SimpleNamespace(
        ensemble_policy_json=None,
        apply_default_ensemble_policy=True,
        split_head_by_support=True,
        sam3_text_quality_alpha=0.35,
        train_sam3_similarity_quality=False,
        sam3_similarity_quality_alpha=0.8,
    )
    sam3_window = {"enabled": True, "mode": "grid"}
    sim_window = {"enabled": True, "mode": "grid"}

    assert (
        calibration._resolve_default_split_head_by_support(
            payload,
            sam3_text_window_cfg=sam3_window,
            similarity_window_cfg=sim_window,
        )
        is True
    )
    assert calibration._resolve_default_sam3_text_quality_alpha(
        payload,
        sam3_text_window_cfg=sam3_window,
        similarity_window_cfg=sim_window,
    ) == 0.35
    assert (
        calibration._resolve_default_train_sam3_similarity_quality(
            payload,
            sam3_text_window_cfg=sam3_window,
            similarity_window_cfg=sim_window,
        )
        is False
    )
    assert calibration._resolve_default_sam3_similarity_quality_alpha(
        payload,
        sam3_text_window_cfg=sam3_window,
        similarity_window_cfg=sim_window,
    ) == 0.8
