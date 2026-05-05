from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from services.falcon_perception import (
    _disable_unsafe_segmentation_hr_cache,
    _falcon_retry_dimensions,
    _normalize_prediction,
    resize_mask_rle,
)


def _tiny_rle() -> dict:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))
    return {"size": [4, 4], "counts": encoded["counts"].decode("utf-8")}


def test_normalize_prediction_recovers_geometry_from_mask_rle():
    pred = {
        "xy": np.ones((4, 4), dtype=np.float32),
        "hw": np.ones((4, 4), dtype=np.float32),
        "mask_rle": _tiny_rle(),
    }
    normalized = _normalize_prediction(pred, width=4, height=4)
    assert normalized is not None
    assert normalized["mask_rle"] == _tiny_rle()
    assert normalized["xy"] == {"x": 0.5, "y": 0.5}
    assert normalized["hw"] == {"w": 0.5, "h": 0.5}


def test_normalize_prediction_prefers_mask_geometry_over_scalar_xyhw():
    pred = {
        "xy": {"x": 0.5, "y": 0.5},
        "hw": {"w": 1.0, "h": 1.0},
        "mask_rle": _tiny_rle(),
    }
    normalized = _normalize_prediction(pred, width=4, height=4)
    assert normalized is not None
    assert normalized["xy"] == {"x": 0.5, "y": 0.5}
    assert normalized["hw"] == {"w": 0.5, "h": 0.5}


def test_falcon_retry_dimensions_descend_and_deduplicate():
    assert _falcon_retry_dimensions(1024, 256) == [1024, 768, 640, 512, 448, 384, 320, 256]
    assert _falcon_retry_dimensions(640, 256) == [640, 512, 448, 384, 320, 256]


def test_resize_mask_rle_updates_size_and_preserves_signal():
    resized = resize_mask_rle(_tiny_rle(), target_height=8, target_width=8)
    assert resized["size"] == [8, 8]
    normalized = _normalize_prediction({"mask_rle": resized}, width=8, height=8)
    assert normalized is not None
    assert normalized["mask_rle"]["size"] == [8, 8]
    assert normalized["hw"]["w"] > 0.0
    assert normalized["hw"]["h"] > 0.0


def test_disable_unsafe_segmentation_hr_cache_turns_cache_off_and_clears_entries():
    class DummyEngine:
        def __init__(self):
            self.enable_hr_cache = True
            self._hr_features_cache = {"abc": object()}

    engine = DummyEngine()
    _disable_unsafe_segmentation_hr_cache(engine)
    assert engine.enable_hr_cache is False
    assert engine._hr_features_cache == {}
