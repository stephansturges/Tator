import numpy as np
from PIL import Image

from tools import build_ensemble_features as features


def test_encode_classifier_features_accepts_numpy_array_classes(monkeypatch):
    crops = [Image.new("RGB", (8, 8), (10, 20, 30))]
    head = {"classes": np.asarray(["car", "boat"], dtype=object)}

    monkeypatch.setattr(
        features.api,
        "_encode_pil_batch_for_head",
        lambda batch, *, head, device_override=None: np.ones((len(batch), 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        features.api,
        "_clip_head_predict_proba",
        lambda feats, head: np.asarray([[0.25, 0.75]], dtype=np.float32),
    )

    probs, embeds = features._encode_classifier_features(
        crops,
        head=head,
        batch_size=1,
        device_override=None,
        min_crop_size=4,
        embed_proj_dim=3,
        embed_proj_seed=7,
        embed_l2_normalize=True,
    )

    assert len(probs) == 1
    assert np.allclose(probs[0], [0.25, 0.75])
    assert len(embeds) == 1
    assert embeds[0].shape == (3,)
