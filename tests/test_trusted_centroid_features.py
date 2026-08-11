import json

import numpy as np

from tools.context_feature_variants import derive_trusted_centroid_block


def _payload():
    feature_names = [
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
            [1.0, 1.0, 0.0, 0.0, 5.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 3.0, 0.9, 0.1],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 4.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    meta = np.asarray(
        [
            json.dumps({"image": "img_a.jpg", "label": "person", "score": 0.95}),
            json.dumps({"image": "img_a.jpg", "label": "person", "score": 0.85}),
            json.dumps({"image": "img_a.jpg", "label": "truck", "score": 0.40}),
            json.dumps({"image": "img_b.jpg", "label": "person", "score": 0.92}),
        ],
        dtype=object,
    )
    return {
        "X": X,
        "meta": meta,
        "feature_names": np.asarray(feature_names, dtype=object),
    }


def test_trusted_centroid_block_uses_same_label_pool_and_emits_margin():
    block = derive_trusted_centroid_block(_payload())
    names = block.feature_names
    margin_idx = names.index("imgctx_trusted_same_other_margin")
    pool_idx = names.index("imgctx_trusted_same_pool_size")
    missing_idx = names.index("imgctx_trusted_missing_pool_flag")

    assert block.X[0, pool_idx] >= 2.0
    assert block.X[0, margin_idx] > 0.0
    assert block.X[0, missing_idx] == 0.0


def test_trusted_centroid_block_marks_missing_pool_when_no_detector_supported_same_label():
    payload = _payload()
    payload["X"][3, 0] = 0.0
    payload["X"][3, 1] = 0.0
    block = derive_trusted_centroid_block(payload)
    names = block.feature_names
    missing_idx = names.index("imgctx_trusted_missing_pool_flag")
    pool_idx = names.index("imgctx_trusted_same_pool_size")

    assert block.X[3, missing_idx] == 1.0
    assert block.X[3, pool_idx] == 0.0
