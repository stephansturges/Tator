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
    _startup_copy2_if_different,
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


def test_link_or_copy_file_copies_when_hardlink_unavailable(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    dest = tmp_path / "dest.bin"
    source.write_bytes(b"payload")

    def fail_link(*_args, **_kwargs):
        raise OSError("hardlink unavailable")

    monkeypatch.setattr(api.os, "link", fail_link)

    _link_or_copy_file(source, dest)

    assert dest.read_bytes() == b"payload"
    assert not dest.is_symlink()


def test_link_or_copy_file_removes_partial_temp_after_copy_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    dest = tmp_path / "dest.bin"
    tmp_dest = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    source.write_bytes(b"payload")

    def fail_link(*_args, **_kwargs):
        raise OSError("hardlink unavailable")

    def fail_copy2(_src, target):
        target.write_bytes(b"partial")
        raise OSError("simulated link fallback copy failure")

    monkeypatch.setattr(api.os, "link", fail_link)
    monkeypatch.setattr(api.shutil, "copy2", fail_copy2)

    with pytest.raises(OSError, match="simulated link fallback copy failure"):
        _link_or_copy_file(source, dest)

    assert not dest.exists()
    assert not tmp_dest.exists()


def test_link_or_copy_file_preserves_existing_dest_after_copy_failure(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.bin"
    dest = tmp_path / "dest.bin"
    tmp_dest = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    source.write_bytes(b"payload")
    dest.write_bytes(b"existing")

    def fail_link(*_args, **_kwargs):
        raise OSError("hardlink unavailable")

    def fail_copy2(_src, target):
        target.write_bytes(b"partial")
        raise OSError("simulated overwrite copy failure")

    monkeypatch.setattr(api.os, "link", fail_link)
    monkeypatch.setattr(api.shutil, "copy2", fail_copy2)

    with pytest.raises(OSError, match="simulated overwrite copy failure"):
        _link_or_copy_file(source, dest, overwrite=True)

    assert dest.read_bytes() == b"existing"
    assert not tmp_dest.exists()


def test_link_or_copy_file_rejects_nested_symlinked_parent_before_mkdir(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        _link_or_copy_file(source, linked_parent / "nested" / "artifacts" / "dest.bin")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_relative_path"
    assert list(outside.iterdir()) == []


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


def test_trained_classifier_labelmap_drops_filtered_and_background_classes():
    labelmap = ["car", "boat", "plane"]

    trained = clip_training._trained_classifier_labelmap(
        labelmap,
        [0, 1, 2],
        ["boat", "__bg_0", "car"],
    )

    assert trained == ["car", "boat"]


def test_trained_classifier_labelmap_synthesizes_missing_class_labels():
    trained = clip_training._trained_classifier_labelmap(
        None,
        [0, 2, 3],
        ["class_2", "__bg_0", "custom_extra"],
    )

    assert trained == ["class_2", "custom_extra"]


def test_scan_yolo_label_files_returns_raw_class_counts(tmp_path):
    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    nested = labels_root / "nested"
    nested.mkdir()
    (labels_root / "a.txt").write_text(
        "1 0.5 0.5 0.2 0.2\nbad\n2 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    (nested / "b.TXT").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    label_map, raw_counts = clip_training._scan_yolo_label_files(
        ["a.jpg", "nested/b.png", "missing.jpg"],
        str(labels_root),
    )

    assert label_map["a.jpg"].endswith("a.txt")
    assert label_map["nested/b.png"].lower().endswith(os.path.join("nested", "b.txt"))
    assert label_map["missing.jpg"] is None
    assert dict(raw_counts) == {1: 1, 2: 2}


def test_embedding_cache_metadata_preserves_raw_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_training, "EMBED_CACHE_ROOT", tmp_path)
    signature = "cache_sig"
    cache_dir = tmp_path / signature
    cache_dir.mkdir()
    chunk_path = cache_dir / "embeddings_0000.dat"
    chunk_path.write_bytes(b"chunk")

    clip_training._write_cache_metadata(
        signature,
        cache_dir,
        [(str(chunk_path), 0, 1)],
        ["car"],
        [0],
        ["image-a"],
        [[0.0, 0.0, 1.0, 1.0]],
        {0},
        raw_counts={0: 3, 2: 1},
    )

    loaded = clip_training._load_cached_embeddings(signature)

    assert loaded is not None
    assert loaded["raw_counts"] == {0: 3, 2: 1}


def test_split_train_test_falls_back_when_group_split_loses_class():
    labels = np.asarray(["car", "car", "boat", "boat"], dtype=object)
    groups = np.asarray(["car_group", "car_group", "boat_group", "boat_group"], dtype=object)

    train_idx, test_idx, used_group_split = clip_training._split_train_test_indices(
        labels,
        groups,
        test_size=0.5,
        random_seed=0,
    )

    assert used_group_split is False
    assert {str(labels[idx]) for idx in train_idx} == {"car", "boat"}
    assert set(train_idx).isdisjoint(set(test_idx))


def test_split_train_test_can_preserve_all_classes_with_empty_test_split():
    labels = np.asarray(["car", "boat"], dtype=object)
    groups = np.asarray(["car_group", "boat_group"], dtype=object)

    train_idx, test_idx, used_group_split = clip_training._split_train_test_indices(
        labels,
        groups,
        test_size=0.5,
        random_seed=0,
    )

    assert used_group_split is False
    assert {str(labels[idx]) for idx in train_idx} == {"car", "boat"}
    assert test_idx.tolist() == []


def test_require_two_training_classes_rejects_single_class():
    with pytest.raises(clip_training.TrainingError, match="at least two classes"):
        clip_training._require_two_training_classes(["car", "car"])


def test_require_two_training_classes_accepts_two_classes():
    clip_training._require_two_training_classes(["car", "boat"])


def test_split_embedding_matrix_allows_zero_rows(tmp_path):
    matrix_path = tmp_path / "test_embeddings.dat"

    matrix = clip_training._make_split_embedding_matrix(str(matrix_path), 0, 512)
    clip_training._flush_split_embedding_matrix(matrix)

    assert matrix.shape == (0, 512)
    assert not matrix_path.exists()


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
def test_copy_helpers_replace_symlink_destination_without_target_write(tmp_path, copy_fn):
    source = tmp_path / "source.bin"
    dest = tmp_path / "dest.bin"
    outside = tmp_path / "outside.bin"
    source.write_bytes(b"payload")
    outside.write_bytes(b"external")
    os.symlink(str(outside), dest)

    copy_fn(source, dest)

    assert dest.read_bytes() == b"payload"
    assert not dest.is_symlink()
    assert outside.read_bytes() == b"external"


@pytest.mark.parametrize(
    "copy_fn, copy2_target",
    [
        (_api_copy2_if_different, "localinferenceapi.shutil.copy2"),
        (_canonical_copy2_if_different, "services.canonical_edr_completion.shutil.copy2"),
        (_detector_copy2_if_different, "services.detectors.shutil.copy2"),
        (_edr_copy2_if_different, "services.edr_packages.shutil.copy2"),
        (_prepass_copy2_if_different, "services.prepass_recipes.shutil.copy2"),
    ],
)
def test_copy_helpers_remove_partial_temp_after_copy_failure(
    tmp_path,
    monkeypatch,
    copy_fn,
    copy2_target,
):
    source = tmp_path / "source.bin"
    dest = tmp_path / "dest.bin"
    tmp_dest = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    source.write_bytes(b"payload")

    def _failing_copy2(_src, target):
        target.write_bytes(b"partial")
        raise OSError("simulated copy failure")

    monkeypatch.setattr(copy2_target, _failing_copy2)

    with pytest.raises(OSError, match="simulated copy failure"):
        copy_fn(source, dest)

    assert not dest.exists()
    assert not tmp_dest.exists()


def test_api_copy_helper_rejects_nested_symlinked_parent_before_mkdir(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        _api_copy2_if_different(source, linked_parent / "nested" / "artifacts" / "dest.bin")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "artifact_path_invalid"
    assert list(outside.iterdir()) == []


def test_startup_copy_helper_replaces_symlink_destination_without_target_write(tmp_path):
    source = tmp_path / "source.bin"
    dest = tmp_path / "dest.bin"
    outside = tmp_path / "outside.bin"
    source.write_bytes(b"payload")
    outside.write_bytes(b"external")
    os.symlink(str(outside), dest)

    _startup_copy2_if_different(source, dest)

    assert dest.read_bytes() == b"payload"
    assert not dest.is_symlink()
    assert outside.read_bytes() == b"external"


def test_startup_copy_helper_removes_partial_temp_after_copy_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    dest = tmp_path / "dest.bin"
    tmp_dest = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    source.write_bytes(b"payload")

    def _failing_copy2(_src, target):
        target.write_bytes(b"partial")
        raise OSError("simulated startup copy failure")

    monkeypatch.setattr("localinferenceapi.shutil.copy2", _failing_copy2)

    with pytest.raises(OSError, match="simulated startup copy failure"):
        _startup_copy2_if_different(source, dest)

    assert not dest.exists()
    assert not tmp_dest.exists()


def test_startup_copy_helper_skips_nested_symlinked_parent_before_mkdir(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _startup_copy2_if_different(source, linked_parent / "nested" / "artifacts" / "dest.bin")

    assert list(outside.iterdir()) == []


def test_startup_copy_helper_replaces_source_target_symlink(tmp_path):
    source = tmp_path / "source.bin"
    dest = tmp_path / "dest.bin"
    source.write_bytes(b"payload")
    os.symlink(str(source), dest)

    _startup_copy2_if_different(source, dest)

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


def test_refresh_active_classifier_reloads_overwritten_training_artifact(tmp_path, monkeypatch):
    model_path = tmp_path / "head.pkl"
    meta_path = tmp_path / "head.meta.pkl"
    labelmap_path = tmp_path / "labels.pkl"
    old_clf = SimpleNamespace(
        classes_=np.asarray(["negative", "positive"], dtype=object),
        coef_=np.asarray([[0.0, 0.0]], dtype=np.float32),
        intercept_=np.asarray([0.0], dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    new_clf = SimpleNamespace(
        classes_=np.asarray(["negative", "positive"], dtype=object),
        coef_=np.asarray([[2.0, 0.0]], dtype=np.float32),
        intercept_=np.asarray([0.5], dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    api.joblib.dump(new_clf, model_path)
    api.joblib.dump(
        {
            "encoder_type": "clip",
            "encoder_model": "ViT-B/32",
            "clip_model": "ViT-B/32",
            "logit_adjustment_inference": True,
            "logit_adjustment": [0.0, 1.0],
            "calibration_temperature": 2.0,
        },
        meta_path,
    )
    api.joblib.dump(["negative", "positive"], labelmap_path)
    stale_head = {
        "classifier_type": "logreg",
        "classes": ["negative", "positive"],
        "coef": np.asarray([[0.0, 0.0]], dtype=np.float32),
        "intercept": np.asarray([0.0], dtype=np.float32),
        "proba_mode": "binary",
    }
    monkeypatch.setattr(api, "clf", old_clf)
    monkeypatch.setattr(api, "active_classifier_path", str(model_path))
    monkeypatch.setattr(api, "active_labelmap_path", str(labelmap_path))
    monkeypatch.setattr(api, "active_label_list", ["old"])
    monkeypatch.setattr(api, "active_classifier_meta", {})
    monkeypatch.setattr(api, "active_encoder_type", "clip")
    monkeypatch.setattr(api, "active_encoder_model", None)
    monkeypatch.setattr(api, "active_classifier_head", stale_head)

    refreshed = api._refresh_active_classifier_if_current(
        SimpleNamespace(
            model_path=str(model_path),
            meta_path=str(meta_path),
            labelmap_path=str(labelmap_path),
        )
    )

    assert refreshed is True
    assert api.clf.coef_.tolist() == [[2.0, 0.0]]
    assert api.active_classifier_head is not stale_head
    assert api.active_label_list == ["negative", "positive"]
    assert api.active_encoder_model == "ViT-B/32"
    assert api.active_classifier_head["logit_adjustment_inference"] is True
    actual = api._clip_head_predict_proba(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        api.active_classifier_head,
    )
    expected_pos = 1.0 / (1.0 + np.exp(-((2.0 + 0.5 + 1.0) / 2.0)))
    assert np.allclose(actual[0], [1.0 - expected_pos, expected_pos], atol=1e-6)


def test_resume_classifier_backbone_reloads_active_cradio(monkeypatch):
    fake_model = object()
    monkeypatch.setattr(api, "active_encoder_type", "cradio")
    monkeypatch.setattr(api, "active_encoder_model", api.CRADIO_DEFAULT_MODEL)
    monkeypatch.setattr(api, "cradio_model", None)
    monkeypatch.setattr(api, "cradio_processor", None)
    monkeypatch.setattr(api, "cradio_model_name", None)
    monkeypatch.setattr(api, "cradio_model_device", None)
    monkeypatch.setattr(api, "cradio_initialized", False)
    monkeypatch.setattr(api, "_clip_reload_needed", True)
    monkeypatch.setattr(api, "resolve_cradio_torch_device", lambda **_kwargs: "mlx")
    monkeypatch.setattr(
        api,
        "_load_cradio_backbone_cached",
        lambda model_name, target_device, raise_on_error=False: (
            fake_model,
            None,
            model_name,
            target_device,
        ),
    )

    api._resume_classifier_backbone()

    assert api.cradio_model is fake_model
    assert api.cradio_processor is None
    assert api.cradio_model_name == api.CRADIO_DEFAULT_MODEL
    assert api.cradio_model_device == "mlx"
    assert api.cradio_initialized is True
    assert api._clip_reload_needed is False


def test_resume_classifier_backbone_reloads_active_clip_model_name(monkeypatch):
    loaded = []
    fake_model = object()
    fake_preprocess = object()

    monkeypatch.setattr(api, "active_encoder_type", "clip")
    monkeypatch.setattr(api, "active_encoder_model", "ViT-L/14")
    monkeypatch.setattr(api, "clip_model", None)
    monkeypatch.setattr(api, "clip_preprocess", None)
    monkeypatch.setattr(api, "clip_model_name", "ViT-B/32")
    monkeypatch.setattr(api, "clip_initialized", False)
    monkeypatch.setattr(api, "_clip_reload_needed", True)
    monkeypatch.setattr(api, "clf", object())
    monkeypatch.setattr(
        api.clip,
        "load",
        lambda name, device=None: (
            loaded.append((name, device)) or fake_model,
            fake_preprocess,
        ),
    )

    api._resume_classifier_backbone()

    assert loaded == [("ViT-L/14", api.device)]
    assert api.clip_model is fake_model
    assert api.clip_preprocess is fake_preprocess
    assert api.clip_model_name == "ViT-L/14"
    assert api.clip_initialized is True
    assert api._clip_reload_needed is False


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
