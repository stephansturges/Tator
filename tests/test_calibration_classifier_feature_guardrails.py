import numpy as np
import pytest

from services.calibration import _validate_classifier_feature_matrix


def _write_npz(path, *, X, feature_names, classifier_classes, embed_proj_dim):
    np.savez(
        path,
        X=np.asarray(X, dtype=np.float32),
        feature_names=np.asarray(feature_names, dtype=object),
        classifier_classes=np.asarray(classifier_classes, dtype=object),
        embed_proj_dim=int(embed_proj_dim),
    )


def test_validate_classifier_feature_matrix_rejects_missing_classifier(tmp_path):
    path = tmp_path / "bad_features.npz"
    _write_npz(
        path,
        X=[[0.1, 0.2]],
        feature_names=["cand_score_yolo", "cand_score_rfdetr"],
        classifier_classes=[],
        embed_proj_dim=0,
    )
    with pytest.raises(RuntimeError, match="classifier_classes_empty"):
        _validate_classifier_feature_matrix(path)


def test_validate_classifier_feature_matrix_accepts_nonzero_embed_and_probs(tmp_path):
    path = tmp_path / "good_features.npz"
    _write_npz(
        path,
        X=[
            [0.2, 0.8, -0.1, 0.3],
            [0.9, 0.1, 0.2, -0.4],
        ],
        feature_names=[
            "clf_prob::person",
            "clf_prob::vehicle",
            "clf_emb_rp::000",
            "clf_emb_rp::001",
        ],
        classifier_classes=["person", "vehicle"],
        embed_proj_dim=2,
    )
    _validate_classifier_feature_matrix(path)
