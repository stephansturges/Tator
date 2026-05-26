from pathlib import Path
import types

import numpy as np
import pytest
import torch
from PIL import Image

import localinferenceapi as api
from services import mlx_dinov3
from tools import clip_training


def test_mlx_dinov3_auto_falls_back_when_worker_or_model_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOV3_BACKEND", "auto")
    monkeypatch.setenv("MLX_DINOV3_WORKER", str(tmp_path / "missing-worker"))
    monkeypatch.setenv("MLX_DINOV3_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(mlx_dinov3.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mlx_dinov3.platform, "machine", lambda: "arm64")

    status = mlx_dinov3.mlx_dinov3_status("facebook/dinov3-vitb16-pretrain-lvd1689m")

    assert status.resolved == "torch"
    assert status.available is False
    assert status.platform_supported is True
    assert mlx_dinov3.resolve_mlx_dinov3_backend("facebook/dinov3-vitb16-pretrain-lvd1689m") == "torch"


def test_mlx_dinov3_explicit_request_fails_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_DINOV3_WORKER", str(tmp_path / "missing-worker"))
    monkeypatch.setenv("MLX_DINOV3_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(mlx_dinov3.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mlx_dinov3.platform, "machine", lambda: "arm64")

    with pytest.raises(mlx_dinov3.MlxDinoV3Unavailable):
        mlx_dinov3.resolve_mlx_dinov3_backend(
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            requested="mlx",
        )


def test_data_ingestion_dinov3_tokens_can_use_mlx_worker_and_cleanup_tmp(tmp_path, monkeypatch):
    tmp_root = tmp_path / "mlx_tmp"
    monkeypatch.setattr(api, "DATA_INGESTION_MLX_DINOV3_TMP_ROOT", tmp_root)

    class FakeMlxDino:
        def encode_image_paths(self, image_paths, *, include_patch_tokens=True):
            assert include_patch_tokens is True
            assert len(image_paths) == 2
            assert all(Path(path).exists() for path in image_paths)
            return {
                "patch_tokens": np.ones((2, 4, 8), dtype=np.float32),
                "cls_token": np.full((2, 8), 2.0, dtype=np.float32),
            }

    monkeypatch.setattr(api, "is_mlx_dinov3_encoder", lambda value: isinstance(value, FakeMlxDino))

    images = [
        Image.new("RGB", (20, 18), (20, 40, 60)),
        Image.new("RGB", (22, 16), (120, 80, 40)),
    ]
    patches, cls = api._data_ingestion_dinov3_tokens(FakeMlxDino(), object(), "mlx", images)

    assert isinstance(patches, torch.Tensor)
    assert isinstance(cls, torch.Tensor)
    assert patches.shape == (2, 4, 8)
    assert cls.shape == (2, 8)
    assert tmp_root.exists()
    assert list(tmp_root.iterdir()) == []


def test_data_ingestion_pooled_encoding_uses_mlx_dinov3_paths(tmp_path, monkeypatch):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (24, 24), (30, 80, 120)).save(image_a)
    Image.new("RGB", (24, 24), (140, 70, 20)).save(image_b)
    prepared = [
        {"image_path": str(image_a), "filename": "a.jpg"},
        {"image_path": str(image_b), "filename": "b.jpg"},
    ]

    class FakeMlxDino:
        def encode_image_paths(self, image_paths, *, include_patch_tokens=True):
            assert image_paths == [str(image_a), str(image_b)]
            assert include_patch_tokens is False
            return {
                "cls_token": np.asarray(
                    [
                        [3.0, 4.0],
                        [0.0, 5.0],
                    ],
                    dtype=np.float32,
                )
            }

    monkeypatch.setattr(api, "is_mlx_dinov3_encoder", lambda value: isinstance(value, FakeMlxDino))
    monkeypatch.setattr(api, "_data_ingestion_get_dinov3", lambda model_name: (FakeMlxDino(), FakeMlxDino(), "unit", "mlx"))

    job = api.DataIngestionJob(job_id="unit", kind="analysis", request={})
    features = api._data_ingestion_encode_prepared_images(
        prepared,
        job=job,
        encoder="dinov3_pooled",
        model_name="unit",
        batch_size=2,
    )

    assert features.shape == (2, 2)
    assert np.allclose(np.linalg.norm(features, axis=1), np.ones(2), atol=1e-6)


def test_data_ingestion_capabilities_report_dinov3_backend():
    caps = api._data_ingestion_capabilities()

    assert "dinov3_backend" in caps
    assert caps["dinov3_backend"]["default"] == "auto"
    assert caps["dinov3_backend"]["resolved"] in {"torch", "mlx", "unavailable"}


def test_set_active_model_can_use_mlx_dinov3_worker(tmp_path, monkeypatch):
    classifiers_root = tmp_path / "classifiers"
    labelmaps_root = tmp_path / "labelmaps"
    classifiers_root.mkdir()
    labelmaps_root.mkdir()
    classifier_path = classifiers_root / "dino_mlx.pkl"
    meta_path = classifiers_root / "dino_mlx.meta.pkl"
    labelmap_path = labelmaps_root / "labels.pkl"
    classifier = types.SimpleNamespace(
        classes_=np.asarray(["car", "boat"], dtype=object),
        coef_=np.zeros((2, 768), dtype=np.float32),
        intercept_=np.zeros(2, dtype=np.float32),
    )
    api.joblib.dump(classifier, classifier_path)
    api.joblib.dump(
        {
            "encoder_type": "dinov3",
            "encoder_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "embedding_dim": 768,
        },
        meta_path,
    )
    api.joblib.dump(["car", "boat"], labelmap_path)

    class FakeMlxDino:
        hidden_size = 768

    fake = FakeMlxDino()

    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(api, "dinov3_model", None)
    monkeypatch.setattr(api, "dinov3_processor", None)
    monkeypatch.setattr(api, "dinov3_initialized", False)
    monkeypatch.setattr(api, "resolve_mlx_dinov3_backend", lambda *_args, **_kwargs: "mlx")
    monkeypatch.setattr(api, "is_mlx_dinov3_encoder", lambda value: isinstance(value, FakeMlxDino))

    def fake_get_dino(model_name, device_name=None):
        assert device_name == "mlx"
        return fake, fake, model_name, "mlx"

    monkeypatch.setattr(api, "_data_ingestion_get_dinov3", fake_get_dino)

    payload = api.set_active_model(
        api.ActiveModelRequest(
            classifier_path=str(classifier_path),
            labelmap_path=str(labelmap_path),
        )
    )

    assert payload["encoder_type"] == "dinov3"
    assert payload["encoder_ready"] is True
    assert api.dinov3_model is fake
    assert api.dinov3_processor is fake
    assert api.dinov3_model_device == "mlx"


def test_resume_classifier_backbone_can_use_mlx_dinov3_worker(monkeypatch):
    class FakeMlxDino:
        hidden_size = 768

    fake = FakeMlxDino()
    monkeypatch.setattr(api, "active_encoder_type", "dinov3")
    monkeypatch.setattr(api, "active_encoder_model", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    monkeypatch.setattr(api, "dinov3_model", None)
    monkeypatch.setattr(api, "dinov3_processor", None)
    monkeypatch.setattr(api, "dinov3_model_name", None)
    monkeypatch.setattr(api, "dinov3_model_device", None)
    monkeypatch.setattr(api, "dinov3_initialized", False)
    monkeypatch.setattr(api, "resolve_mlx_dinov3_backend", lambda *_args, **_kwargs: "mlx")

    def fake_get_dino(model_name, device_name=None):
        assert device_name == "mlx"
        return fake, fake, model_name, "mlx"

    monkeypatch.setattr(api, "_data_ingestion_get_dinov3", fake_get_dino)

    api._resume_classifier_backbone()

    assert api.dinov3_model is fake
    assert api.dinov3_processor is fake
    assert api.dinov3_model_name == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert api.dinov3_model_device == "mlx"
    assert api.dinov3_initialized is True


def test_clip_training_dinov3_loader_and_encoder_can_use_mlx(monkeypatch):
    class FakeMlxDino:
        hidden_size = 2

        def encode_image_paths(self, image_paths, *, include_patch_tokens=True):
            assert len(image_paths) == 2
            assert include_patch_tokens is True
            assert all(Path(path).exists() for path in image_paths)
            return {
                "cls_token": np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32),
                "patch_tokens": np.asarray(
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[2.0, 0.0], [0.0, 2.0]],
                    ],
                    dtype=np.float32,
                ),
            }

    fake = FakeMlxDino()
    monkeypatch.setenv("DINOV3_BACKEND", "auto")
    monkeypatch.setattr(clip_training, "resolve_mlx_dinov3_backend", lambda *_args, **_kwargs: "mlx")
    monkeypatch.setattr(clip_training, "get_mlx_dinov3_worker", lambda _model_name: fake)
    monkeypatch.setattr(clip_training, "is_mlx_dinov3_encoder", lambda value: isinstance(value, FakeMlxDino))

    model, processor = clip_training._load_dinov3("facebook/dinov3-vitb16-pretrain-lvd1689m", "cpu")
    assert model is fake
    assert processor is fake

    feats = clip_training._encode_batch_dinov3(
        model,
        processor,
        "cpu",
        [Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))],
        pooling="cls_patch_concat",
        normalize=True,
    )

    assert feats.shape == (2, 4)
    assert np.allclose(np.linalg.norm(feats, axis=1), np.ones(2), atol=1e-6)
