import numpy as np
import pytest

from services import mlx_sam


def _force_apple_mlx(monkeypatch):
    monkeypatch.setattr(mlx_sam.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mlx_sam.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(mlx_sam.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(mlx_sam, "_mlx_runtime_error", lambda: None)


def test_resolve_mlx_sam_config_accepts_converted_model_and_examples_root(tmp_path, monkeypatch):
    _force_apple_mlx(monkeypatch)
    model_dir = tmp_path / "sam-vit-base-mlx"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"test")
    package_dir = tmp_path / "mlx-examples" / "segment_anything" / "segment_anything"
    package_dir.mkdir(parents=True)
    (package_dir / "sam.py").write_text("", encoding="utf-8")
    (package_dir / "predictor.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("SAM_MLX_MODEL_PATH", str(model_dir))
    monkeypatch.setenv("SAM_MLX_ROOT", str(tmp_path / "mlx-examples" / "segment_anything"))

    config = mlx_sam.resolve_mlx_sam_config()

    assert config.available is True
    assert config.model_path == model_dir
    assert config.package_dir == package_dir.resolve()
    assert config.reason is None


def test_explicit_mlx_sam_reports_missing_assets(monkeypatch):
    _force_apple_mlx(monkeypatch)
    monkeypatch.delenv("SAM_MLX_MODEL_PATH", raising=False)
    monkeypatch.delenv("MLX_SAM_MODEL_PATH", raising=False)
    monkeypatch.delenv("SAM_CHECKPOINT_PATH", raising=False)

    with pytest.raises(mlx_sam.MlxSamUnavailable, match="mlx_sam_model_path_missing"):
        mlx_sam.should_use_mlx_sam("mlx")


def test_resolve_mlx_sam_config_rejects_unusable_mlx_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(mlx_sam.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mlx_sam.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(mlx_sam.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(mlx_sam, "_mlx_runtime_error", lambda: "No Metal device available")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"test")
    monkeypatch.setenv("SAM_MLX_MODEL_PATH", str(model_dir))

    config = mlx_sam.resolve_mlx_sam_config()

    assert config.available is False
    assert config.reason.startswith("mlx_runtime_unavailable")


def test_mlx_sam_outputs_match_segment_anything_numpy_shapes():
    masks = np.zeros((1, 7, 11, 3), dtype=bool)
    scores = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    logits = np.zeros((1, 4, 5, 3), dtype=np.float32)

    out_masks, out_scores, out_logits = mlx_sam._to_segment_anything_outputs(
        masks, scores, logits
    )

    assert out_masks.shape == (3, 7, 11)
    assert out_scores.shape == (3,)
    assert out_logits.shape == (3, 4, 5)


def test_normalize_predict_kwargs_converts_meta_mask_logits_to_mlx_layout():
    pytest.importorskip("mlx.core")

    normalized = mlx_sam._normalize_predict_kwargs(
        {
            "point_coords": np.array([[1, 2]], dtype=np.float32),
            "point_labels": np.array([1], dtype=np.int64),
            "mask_input": np.zeros((1, 8, 9), dtype=np.float32),
            "return_logits": True,
        }
    )

    assert tuple(normalized["point_coords"].shape) == (1, 1, 2)
    assert tuple(normalized["point_labels"].shape) == (1, 1)
    assert tuple(normalized["mask_input"].shape) == (1, 8, 9, 1)
    assert normalized["return_logits"] is True


def test_build_mlx_sam_predictor_loads_external_package_without_name_collision(tmp_path, monkeypatch):
    _force_apple_mlx(monkeypatch)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"test")
    package_dir = tmp_path / "mlx_examples" / "segment_anything" / "segment_anything"
    package_dir.mkdir(parents=True)
    (package_dir / "sam.py").write_text(
        "def load(path):\n    return {'path': str(path)}\n",
        encoding="utf-8",
    )
    (package_dir / "predictor.py").write_text(
        "import numpy as np\n\n"
        "class SamPredictor:\n"
        "    def __init__(self, model):\n"
        "        self.model = model\n"
        "        self.image_shape = None\n"
        "    def set_image(self, image):\n"
        "        self.image_shape = image.shape\n"
        "    def predict(self, **kwargs):\n"
        "        return (\n"
        "            np.zeros((1, 3, 4, 1), dtype=bool),\n"
        "            np.ones((1, 1), dtype=np.float32),\n"
        "            np.zeros((1, 2, 2, 1), dtype=np.float32),\n"
        "        )\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SAM_MLX_MODEL_PATH", str(model_dir))
    monkeypatch.setenv("SAM_MLX_ROOT", str(tmp_path / "mlx_examples" / "segment_anything"))

    predictor = mlx_sam.build_mlx_sam_predictor()
    predictor.set_image(np.zeros((6, 8, 3), dtype=np.uint8))
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[1, 2]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int64),
        multimask_output=False,
    )

    assert predictor.predictor.image_shape == (6, 8, 3)
    assert masks.shape == (1, 3, 4)
    assert scores.shape == (1,)
    assert logits.shape == (1, 2, 2)
