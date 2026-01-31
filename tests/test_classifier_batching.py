import numpy as np
from PIL import Image

import localinferenceapi as api


def test_classifier_review_batches(monkeypatch):
    detections = [
        {"label": "car", "bbox_xyxy_px": [0, 0, 10, 10], "score": 0.9},
        {"label": "car", "bbox_xyxy_px": [20, 20, 30, 30], "score": 0.9},
    ]
    head = {
        "classes": ["car", "background"],
        "background_labels": ["background"],
        "min_prob": 0.5,
        "margin": 0.0,
        "background_margin": 0.0,
    }

    def fake_encode(crops, head, max_batch_size=None, batch_size_override=None):
        return np.zeros((len(crops), 4), dtype=np.float32)

    def fake_predict(head, embeddings, return_prob=True):
        return np.array([[0.9, 0.1], [0.4, 0.6]], dtype=np.float32)

    def fake_keep_mask(proba, target_index, min_prob, margin, background_indices=None, background_guard=False, background_margin=0.0):
        return np.array([float(proba[0, target_index]) >= 0.5], dtype=bool)

    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)
    monkeypatch.setattr(api, "_clip_head_predict_proba", fake_predict)
    monkeypatch.setattr(api, "_clip_head_keep_mask", fake_keep_mask)

    pil_img = Image.new("RGB", (64, 64), (0, 0, 0))
    reviewed, counts = api._agent_classifier_review(detections, pil_img=pil_img, classifier_head=head)

    assert counts["classifier_checked"] == 2
    assert counts["classifier_rejected"] == 1
    assert len(reviewed) == 1
    assert reviewed[0]["classifier_accept"] is True
    assert detections[1]["classifier_accept"] is False
