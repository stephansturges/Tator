from pathlib import Path
import json
import os
import types

import numpy as np
import pytest
import torch
from PIL import Image

import localinferenceapi as api
from services import mlx_dinov3
from tools import clip_training


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_DINOV3_PARITY") != "1",
    reason="set RUN_REAL_DINOV3_PARITY=1 to load the local Torch and MLX DINOv3 checkpoints",
)
def test_real_mlx_dinov3_patch_tokens_match_torch_ordering():
    """Gate refinement on the real 14x14 backend token contract.

    The normal unit suite stays offline and lightweight. Release verification
    opts into this test on Apple Silicon with the already-converted MLX model
    and Hugging Face checkpoint available locally. A spatially asymmetric input
    makes a transpose, flip, or other token permutation observable.
    """

    from transformers import AutoImageProcessor, AutoModel

    model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    y_grid, x_grid = np.mgrid[0:224, 0:224]
    pixels = np.empty((224, 224, 3), dtype=np.uint8)
    pixels[..., 0] = (x_grid * 255 // 223).astype(np.uint8)
    pixels[..., 1] = (y_grid * 255 // 223).astype(np.uint8)
    pixels[..., 2] = (
        ((x_grid // 16) * 29 + (y_grid // 16) * 47) % 256
    ).astype(np.uint8)
    image = Image.fromarray(pixels, "RGB")
    processor = AutoImageProcessor.from_pretrained(
        model_id,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_id,
        local_files_only=True,
    ).eval()
    try:
        with torch.inference_mode():
            output = model(**processor(images=[image], return_tensors="pt"))
        register_count = int(
            getattr(model.config, "num_register_tokens", 0) or 0
        )
        torch_patches = (
            output.last_hidden_state[:, 1 + register_count :, :]
            .float()
            .cpu()
            .numpy()[0]
        )
        worker = mlx_dinov3.get_mlx_dinov3_worker(model_id)
        mlx_patches = np.asarray(
            worker.encode_pixels(
                pixels[None],
                include_patch_tokens=True,
            )["patch_tokens"],
            dtype=np.float32,
        )[0]

        def normalized(values):
            return values / np.maximum(
                np.linalg.norm(values, axis=-1, keepdims=True),
                1e-12,
            )

        similarities = normalized(torch_patches) @ normalized(mlx_patches).T
        best_mlx_index = similarities.argmax(axis=1)
        diagonal = np.diag(similarities)
        assert torch_patches.shape == mlx_patches.shape == (196, 768)
        assert np.array_equal(best_mlx_index, np.arange(196))
        assert float(diagonal.min()) >= 0.999
        assert float(diagonal.mean()) >= 0.9999
    finally:
        image.close()
        mlx_dinov3.stop_mlx_dinov3_workers()


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


def test_mlx_pil_encoding_prefers_packed_pixel_transport(tmp_path, monkeypatch):
    tmp_root = tmp_path / "mlx_tmp"
    monkeypatch.setattr(api, "DATA_INGESTION_MLX_DINOV3_TMP_ROOT", tmp_root)
    monkeypatch.delenv("MLX_DINOV3_TRANSPORT", raising=False)

    class FakeMlxDino:
        def encode_pixels(self, pixels, *, include_patch_tokens=True):
            assert pixels.dtype == np.uint8
            assert pixels.shape == (2, 224, 224, 3)
            assert include_patch_tokens is False
            return {
                "cls_token": np.asarray(
                    [[3.0, 4.0], [0.0, 5.0]],
                    dtype=np.float32,
                )
            }

        def encode_image_paths(self, *_args, **_kwargs):
            raise AssertionError("packed transport must avoid image files")

    images = [
        Image.new("RGB", (20, 18), (20, 40, 60)),
        Image.new("RGB", (22, 16), (120, 80, 40)),
    ]
    try:
        result = api._encode_mlx_dinov3_pil_images(
            FakeMlxDino(),
            images,
            include_patch_tokens=False,
        )
    finally:
        for image in images:
            image.close()

    assert np.array_equal(
        result["cls_token"],
        np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32),
    )
    assert not tmp_root.exists()


def test_mlx_worker_encode_pixels_writes_versioned_request(tmp_path, monkeypatch):
    worker = object.__new__(mlx_dinov3.MlxDinoV3Worker)
    worker._lock = mlx_dinov3.threading.Lock()
    worker._process = types.SimpleNamespace(
        stdin=types.SimpleNamespace(
            payload="",
            write=lambda value: setattr(worker._process.stdin, "payload", value),
            flush=lambda: None,
        )
    )
    worker.timeout_seconds = 5.0
    monkeypatch.setattr(worker, "_ensure_started", lambda: None)

    def fake_response(*, timeout_seconds):
        assert timeout_seconds == 5.0
        request = json.loads(worker._process.stdin.payload)
        assert request["protocol_version"] == 2
        assert Path(request["input_path"]).is_file()
        assert request["include_patch_tokens"] is False
        from safetensors.numpy import load_file, save_file

        packed = load_file(request["input_path"])["pixels"]
        assert packed.shape == (2, 224, 224, 3)
        save_file(
            {"cls_token": np.ones((2, 4), dtype=np.float32)},
            request["output_path"],
        )
        return {"id": request["id"], "ok": True}

    monkeypatch.setattr(worker, "_read_json_line", fake_response)

    result = worker.encode_pixels(
        np.zeros((2, 224, 224, 3), dtype=np.uint8),
        include_patch_tokens=False,
    )

    assert result["cls_token"].shape == (2, 4)


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
