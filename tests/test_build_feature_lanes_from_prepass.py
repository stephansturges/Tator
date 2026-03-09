from tools.build_feature_lanes_from_prepass import _lane_config, _lane_id


def test_lane_id_keeps_positive_image_embed_dims_distinct():
    assert _lane_id("window", 64) == "window_imgctx64"
    assert _lane_id("window", 128) == "window_imgctx128"
    assert _lane_id("window", 1024) == "window_imgctx1024"
    assert len({_lane_id("window", 64), _lane_id("window", 128), _lane_id("window", 1024)}) == 3


def test_lane_config_uses_actual_image_embed_dim():
    assert _lane_config("cachekey", 0) == "cachekey_noimg"
    assert _lane_config("cachekey", 256) == "cachekey_imgctx256"
