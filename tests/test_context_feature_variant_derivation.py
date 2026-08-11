import json

import numpy as np

from tools.context_feature_variants import (
    COMBINED_VARIANT,
    SCENE_SUMMARY_VARIANT,
    TRUSTED_CENTROID_VARIANT,
    compute_payload_feature_block_stats,
    derive_variant_payload,
)


def _payload():
    feature_names = [
        "cand_score_yolo",
        "cand_score_rfdetr",
        "cand_has_yolo",
        "cand_has_rfdetr",
        "cand_has_sam3_text",
        "cand_has_sam3_similarity",
        "support_count_total",
        "clf_emb_rp::000",
        "clf_emb_rp::001",
    ]
    X = np.asarray(
        [
            [0.9, 0.8, 1.0, 1.0, 0.0, 0.0, 4.0, 1.0, 0.0],
            [0.8, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.9, 0.1],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [0.7, 0.6, 1.0, 1.0, 0.0, 0.0, 3.0, 0.95, 0.05],
        ],
        dtype=np.float32,
    )
    meta = np.asarray(
        [
            json.dumps({"image": "img_a.jpg", "label": "person", "score": 0.95}),
            json.dumps({"image": "img_a.jpg", "label": "person", "score": 0.80}),
            json.dumps({"image": "img_a.jpg", "label": "truck", "score": 0.60}),
            json.dumps({"image": "img_b.jpg", "label": "person", "score": 0.88}),
        ],
        dtype=object,
    )
    return {
        "X": X,
        "y": np.asarray([1, 0, 0, 1], dtype=np.int64),
        "y_iou": np.asarray([1, 0, 0, 1], dtype=np.int64),
        "best_iou_any": np.asarray([0.8, 0.2, 0.1, 0.7], dtype=np.float32),
        "best_label_any": np.asarray(["person", "person", "truck", "person"], dtype=object),
        "meta": meta,
        "feature_names": np.asarray(feature_names, dtype=object),
        "labelmap": np.asarray(["person", "truck"], dtype=object),
        "classifier_classes": np.asarray(["person", "truck"], dtype=object),
    }


def test_scene_summary_variant_is_deterministic_and_preserves_row_order():
    payload = _payload()
    first, _ = derive_variant_payload(payload, variant=SCENE_SUMMARY_VARIANT, parent_feature_npz="/tmp/base.npz")
    second, _ = derive_variant_payload(payload, variant=SCENE_SUMMARY_VARIANT, parent_feature_npz="/tmp/base.npz")
    np.testing.assert_allclose(first["X"], second["X"])
    assert list(first["meta"]) == list(payload["meta"])
    assert first["X"].shape[0] == payload["X"].shape[0]
    assert any(str(name).startswith("imgctx_scene_") for name in first["feature_names"])


def test_trusted_centroid_variant_is_deterministic_and_appends_features():
    payload = _payload()
    first, _ = derive_variant_payload(payload, variant=TRUSTED_CENTROID_VARIANT, parent_feature_npz="/tmp/base.npz")
    second, _ = derive_variant_payload(payload, variant=TRUSTED_CENTROID_VARIANT, parent_feature_npz="/tmp/base.npz")
    np.testing.assert_allclose(first["X"], second["X"])
    assert first["X"].shape[1] > payload["X"].shape[1]
    assert any(str(name).startswith("imgctx_trusted_") for name in first["feature_names"])


def test_combined_variant_contains_both_feature_families():
    payload = _payload()
    combined, _ = derive_variant_payload(payload, variant=COMBINED_VARIANT, parent_feature_npz="/tmp/base.npz")
    names = [str(name) for name in combined["feature_names"]]
    assert any(name.startswith("imgctx_scene_") for name in names)
    assert any(name.startswith("imgctx_trusted_") for name in names)


def test_payload_feature_block_stats_detects_missing_scene_features():
    payload = _payload()
    stats = compute_payload_feature_block_stats(payload, expected_variant_id=SCENE_SUMMARY_VARIANT)
    assert stats["new_feature_count"] == 0
    assert stats["variant_type"] == "global-only"
