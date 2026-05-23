import os
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch

import localinferenceapi as api
from tools import clip_training
from utils.local_salad import LOCAL_SALAD_CACHE_VERSION, LOCAL_SALAD_POLICY, LOCAL_SALAD_TRAINER, LocalSALADConfig, LocalSALADHead
from utils.local_salad_mlx import is_mlx_local_salad_head, local_salad_mlx_available
from localinferenceapi import (
    _copy2_if_different as _api_copy2_if_different,
    _link_or_copy_file,
    _unlink_self_referential_symlink,
)
from services.calibration_helpers import _calibration_safe_link
from services.canonical_edr_completion import _copy2_if_different as _canonical_copy2_if_different
from services.detectors import _copy2_if_different as _detector_copy2_if_different
from services.edr_packages import _copy2_if_different as _edr_copy2_if_different
from services.prepass_recipes import _copy2_if_different as _prepass_copy2_if_different


def test_link_or_copy_file_noops_when_source_is_destination(tmp_path):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"classifier")

    _link_or_copy_file(artifact, artifact, overwrite=True)

    assert artifact.read_bytes() == b"classifier"
    assert not artifact.is_symlink()


def test_logistic_regression_constructor_matches_installed_sklearn():
    clf = clip_training._make_logistic_regression(
        random_state=0,
        max_iter=1,
        solver="lbfgs",
        class_weight=None,
        C=1.0,
        warm_start=True,
        verbose=0,
        tol=1e-4,
    )

    assert clf.max_iter == 1
    assert clf.solver == "lbfgs"


def test_labels_from_proba_uses_classifier_class_order():
    labels = clip_training._labels_from_proba(
        ["negative", "positive"],
        np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32),
    )

    assert labels.tolist() == ["positive", "negative"]


def test_labels_from_proba_rejects_shape_mismatch():
    with pytest.raises(clip_training.TrainingError, match="Probability matrix shape"):
        clip_training._labels_from_proba(
            ["negative", "positive"],
            np.asarray([[0.1], [0.8]], dtype=np.float32),
        )


def test_unlink_self_referential_symlink_removes_broken_artifact(tmp_path):
    artifact = tmp_path / "model.pkl"
    os.symlink(str(artifact), artifact)

    assert artifact.is_symlink()
    assert _unlink_self_referential_symlink(artifact) is True
    assert not artifact.exists()
    assert not artifact.is_symlink()


@pytest.mark.parametrize(
    "copy_fn",
    [
        _api_copy2_if_different,
        _canonical_copy2_if_different,
        _detector_copy2_if_different,
        _edr_copy2_if_different,
        _prepass_copy2_if_different,
    ],
)
def test_copy_helpers_noop_when_source_is_destination(tmp_path, copy_fn):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"payload")

    copy_fn(artifact, artifact)

    assert artifact.read_bytes() == b"payload"
    assert not artifact.is_symlink()


@pytest.mark.parametrize(
    "copy_fn",
    [
        _api_copy2_if_different,
        _canonical_copy2_if_different,
        _detector_copy2_if_different,
        _edr_copy2_if_different,
        _prepass_copy2_if_different,
    ],
)
def test_copy_helpers_replace_self_referential_destination(tmp_path, copy_fn):
    source = tmp_path / "source.bin"
    dest = tmp_path / "dest.bin"
    source.write_bytes(b"payload")
    os.symlink(str(dest), dest)

    copy_fn(source, dest)

    assert dest.read_bytes() == b"payload"
    assert not dest.is_symlink()


def test_calibration_safe_link_replaces_self_referential_destination(tmp_path):
    source = tmp_path / "features.npz"
    dest = tmp_path / "cached_features.npz"
    source.write_bytes(b"features")
    os.symlink(str(dest), dest)

    _calibration_safe_link(source, dest)

    assert dest.exists()
    assert dest.resolve() == source.resolve()


def test_clip_auto_predict_loads_legacy_active_classifier_head(tmp_path, monkeypatch):
    classifier_path = tmp_path / "legacy_logreg.pkl"
    classifier = SimpleNamespace(
        classes_=np.array(["LightVehicle", "Person"], dtype=object),
        coef_=np.zeros((2, 512), dtype=np.float32),
        intercept_=np.zeros(2, dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    classifier.coef_[0, 0] = 3.0
    classifier.coef_[1, 1] = 3.0
    joblib.dump(classifier, classifier_path)

    monkeypatch.setattr(api, "active_classifier_head", None)
    monkeypatch.setattr(api, "active_classifier_path", str(classifier_path))

    features = np.zeros((1, 512), dtype=np.float32)
    features[0, 0] = 1.0
    details = api._clip_auto_predict_details(features, background_guard=False)

    assert details["error"] is None
    assert details["label"] == "LightVehicle"
    assert isinstance(api.active_classifier_head, dict)


def test_auto_class_local_salad_runtime_prefers_mlx_when_requested(tmp_path, monkeypatch):
    if not local_salad_mlx_available():
        pytest.skip("MLX is not available in this environment")

    config = LocalSALADConfig(num_channels=8, num_clusters=4, cluster_dim=8, token_dim=8, hidden_dim=64, dropout=0.0)
    torch_head = LocalSALADHead(config)
    path = tmp_path / "unit_head.pt"
    torch.save(
        {
            "format": LOCAL_SALAD_CACHE_VERSION,
            "config": config.to_dict(),
            "state_dict": torch_head.state_dict(),
            "metadata": {
                "id": "unit_head",
                "label": "Unit Head",
                "encoder_type": "dinov3",
                "policy": LOCAL_SALAD_POLICY,
                "trainer": LOCAL_SALAD_TRAINER,
            },
        },
        path,
    )
    monkeypatch.setenv("LOCAL_SALAD_BACKEND", "mlx")

    loaded, meta = clip_training._load_local_salad_runtime_head(path, device_name="cpu")
    desc = clip_training._encode_local_salad_head_np(
        loaded,
        torch.randn(2, 5, 8),
        torch.randn(2, 8),
    )

    assert is_mlx_local_salad_head(loaded)
    assert meta["runtime_backend"] == "mlx"
    assert desc.shape == (2, loaded.output_dim)
    assert np.allclose(np.linalg.norm(desc, axis=1), np.ones(2), atol=1e-5)
