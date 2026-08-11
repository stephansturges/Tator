from __future__ import annotations

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from services.falcon_perception import (
    _FALCON_FLEX_IMPORT_NEW,
    _FALCON_FLEX_IMPORT_OLD,
    _disable_unsafe_segmentation_hr_cache,
    _falcon_retry_dimensions,
    _normalize_prediction,
    _patch_falcon_modeling_file,
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


def test_falcon_source_patch_replaces_snapshot_symlink_without_mutating_blob(tmp_path):
    source_root = tmp_path / "snapshot"
    source_root.mkdir()
    shared_blob = tmp_path / "blob.py"
    original = _FALCON_FLEX_IMPORT_OLD + "\nclass Demo:\n    pass\n"
    shared_blob.write_text(original, encoding="utf-8")
    snapshot_file = source_root / "modeling_falcon_perception.py"
    try:
        snapshot_file.symlink_to(shared_blob)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _patch_falcon_modeling_file(snapshot_file)

    assert not snapshot_file.is_symlink()
    assert shared_blob.read_text(encoding="utf-8") == original
    patched = snapshot_file.read_text(encoding="utf-8")
    assert _FALCON_FLEX_IMPORT_NEW in patched
    assert _FALCON_FLEX_IMPORT_OLD not in patched
