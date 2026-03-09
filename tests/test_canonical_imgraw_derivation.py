import json

import numpy as np

from tools.context_feature_variants import IMGRAW_VARIANT, derive_variant_payload


def test_imgraw_drops_image_probability_block_but_keeps_image_embedding_block():
    payload = {
        "X": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        "y": np.asarray([1], dtype=np.int64),
        "y_iou": np.asarray([1], dtype=np.int64),
        "best_iou_any": np.asarray([0.5], dtype=np.float32),
        "best_label_any": np.asarray(["person"], dtype=object),
        "meta": np.asarray([json.dumps({"image": "img.jpg", "label": "person", "score": 0.9})], dtype=object),
        "feature_names": np.asarray(
            [
                "img_clf_prob::person",
                "img_clf_prob_label",
                "img_clf_emb_rp::000",
                "cand_score_yolo",
            ],
            dtype=object,
        ),
        "labelmap": np.asarray(["person"], dtype=object),
        "classifier_classes": np.asarray(["person"], dtype=object),
    }

    derived, stats = derive_variant_payload(payload, variant=IMGRAW_VARIANT, parent_feature_npz="/tmp/base.npz")
    names = [str(name) for name in derived["feature_names"]]

    assert "img_clf_prob::person" not in names
    assert "img_clf_prob_label" not in names
    assert "img_clf_emb_rp::000" in names
    assert "cand_score_yolo" in names
    assert stats["dropped_img_prob_features"] == 2
    assert str(derived["context_variant_id"].item()) == IMGRAW_VARIANT
