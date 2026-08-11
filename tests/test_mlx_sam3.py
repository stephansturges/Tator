from __future__ import annotations

import importlib.util
import inspect

import numpy as np

from services import mlx_sam3


def test_point_prompt_transform_matches_meta_input_normalization():
    coords, labels = mlx_sam3._build_sparse_prompt_arrays(
        point_coords=np.asarray([[320.0, 180.0], [100.0, 50.0]]),
        point_labels=np.asarray([1, 0]),
        box=None,
        orig_hw=(360, 640),
        image_size=1008,
        embedding_size=72,
    )

    assert labels.tolist() == [1, 0, -1]
    reconstructed = (coords + 0.5) / 72.0
    expected_model = np.asarray([[504.0, 504.0], [157.5, 140.0], [0.0, 0.0]])
    np.testing.assert_allclose(reconstructed, (expected_model + 0.5) / 1008.0, atol=1e-7)


def test_box_corners_precede_points_without_padding():
    coords, labels = mlx_sam3._build_sparse_prompt_arrays(
        point_coords=np.asarray([[50.0, 60.0]]),
        point_labels=np.asarray([1]),
        box=np.asarray([10.0, 20.0, 90.0, 120.0]),
        orig_hw=(200, 100),
        image_size=1008,
        embedding_size=72,
    )

    assert coords.shape == (3, 2)
    assert labels.tolist() == [2, 3, 1]


def test_mask_input_is_resized_to_tracker_low_resolution():
    mask = np.arange(64, dtype=np.float32).reshape(1, 8, 8)

    normalized = mlx_sam3._normalize_mask_input(mask, target_size=288)

    assert normalized.shape == (1, 288, 288, 1)
    assert normalized.dtype == np.float32


def test_channel_last_mask_input_keeps_spatial_axes():
    mask = np.arange(64, dtype=np.float32).reshape(8, 8, 1)

    normalized = mlx_sam3._normalize_mask_input(mask, target_size=8)

    assert normalized.shape == (1, 8, 8, 1)
    np.testing.assert_array_equal(normalized[0, :, :, 0], mask[:, :, 0])


def test_dense_position_grid_uses_pixel_centers():
    coords = mlx_sam3._dense_position_coords(2, 4)

    assert coords.shape == (1, 8, 2)
    np.testing.assert_allclose(coords[0, 0], [0.125, 0.25])
    np.testing.assert_allclose(coords[0, -1], [0.875, 0.75])


def test_explicit_cached_model_path_is_available_on_apple_silicon(tmp_path, monkeypatch):
    model = tmp_path / "sam3"
    model.mkdir()
    for name in ("config.json", "processor_config.json", "model.safetensors"):
        (model / name).write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SAM3_MLX_MODEL_PATH", str(model))
    monkeypatch.setattr(mlx_sam3.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mlx_sam3.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(mlx_sam3, "_mlx_vlm_version_error", lambda: None)
    monkeypatch.setattr(mlx_sam3, "_mlx_runtime_error", lambda: None)

    status = mlx_sam3.resolve_mlx_sam3_config()

    assert status.available is True
    assert status.model_path == model.resolve()


def test_runtime_notice_exposes_torch_parity_override():
    assert mlx_sam3.DEFAULT_MLX_SAM3_MODEL_ID == "mlx-community/sam3-bf16"
    assert "BF16" in mlx_sam3.MLX_SAM3_RUNTIME_NOTICE
    assert "SAM3_BACKEND=torch" in mlx_sam3.MLX_SAM3_RUNTIME_NOTICE


def test_runtime_registry_exposes_only_locally_passing_variants():
    assert tuple(mlx_sam3.MLX_SAM3_VARIANTS) == (
        "mlx-bf16",
        "mlx-8bit",
        "mlx-5bit",
        "mlx-mxfp8",
        "mlx-mxfp4",
        "mlx-4bit",
    )
    assert "mlx-6bit" not in mlx_sam3.MLX_SAM3_VARIANTS
    assert "mlx-nvfp4" not in mlx_sam3.MLX_SAM3_VARIANTS
    assert mlx_sam3.normalize_mlx_sam3_runtime("mxfp4") == "mlx-mxfp4"


def test_explicit_runtime_ignores_default_environment_checkpoint(tmp_path, monkeypatch):
    hf_home = tmp_path / "hf"
    spec = mlx_sam3.MLX_SAM3_VARIANTS["mlx-8bit"]
    model = hf_home / "hub" / "models--mlx-community--sam3-8bit" / "snapshots" / spec.revision
    model.mkdir(parents=True)
    for name in ("config.json", "processor_config.json", "model.safetensors"):
        (model / name).write_text("{}", encoding="utf-8")
    wrong = tmp_path / "wrong-default"
    wrong.mkdir()
    for name in ("config.json", "processor_config.json", "model.safetensors"):
        (wrong / name).write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("SAM3_MLX_MODEL_PATH", str(wrong))
    monkeypatch.setattr(mlx_sam3.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mlx_sam3.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(mlx_sam3, "_mlx_vlm_version_error", lambda: None)
    monkeypatch.setattr(mlx_sam3, "_mlx_runtime_error", lambda: None)

    config = mlx_sam3.resolve_mlx_sam3_config("mlx-8bit")

    assert config.available is True
    assert config.model_path == model.resolve()
    assert config.model_id == spec.model_id


def test_decoder_compatibility_path_preserves_meta_invariants():
    source = inspect.getsource(mlx_sam3._run_mlx_sam3_mask_decoder)
    transformer_source = inspect.getsource(
        mlx_sam3._run_mlx_sam3_two_way_transformer
    )

    assert source.index("decoder.obj_score_token.weight") < source.index(
        "decoder.iou_token.weight"
    )
    assert "decoder.upscale_conv1(src) + decoder.conv_s1(high_res_features[1])" in source
    assert "decoder.upscale_conv2(upscaled) + decoder.conv_s0(high_res_features[0])" in source
    assert "mx.sigmoid(decoder.iou_prediction_head(iou_token))" in source
    assert "if layer_index == 0" in transformer_source
    assert "queries = layer.self_attn(queries, queries, queries)" in transformer_source
