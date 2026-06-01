"""Optional MLX runtime adapter for SAM1 interactive annotation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
import platform
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


MLX_SAM_MODEL_ENV_KEYS = ("SAM_MLX_MODEL_PATH", "MLX_SAM_MODEL_PATH")
MLX_SAM_ROOT_ENV_KEYS = ("SAM_MLX_ROOT", "MLX_SAM_ROOT", "MLX_EXAMPLES_ROOT")


class MlxSamUnavailable(RuntimeError):
    """Raised when MLX SAM was explicitly requested but cannot be built."""


@dataclass(frozen=True)
class MlxSamConfig:
    available: bool
    reason: Optional[str]
    model_path: Optional[Path]
    package_dir: Optional[Path]
    apple_silicon: bool
    mlx_installed: bool


class MlxSamPredictorAdapter:
    """Normalize the MLX example predictor to Meta SAM's numpy API shape."""

    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor

    def set_image(self, np_img: np.ndarray) -> None:
        self.predictor.set_image(np.ascontiguousarray(np_img))

    def predict(self, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = _normalize_predict_kwargs(kwargs)
        masks, scores, logits = self.predictor.predict(**normalized)
        return _to_segment_anything_outputs(masks, scores, logits)


def mlx_sam_status() -> Dict[str, Any]:
    config = resolve_mlx_sam_config()
    return {
        "available": config.available,
        "reason": config.reason,
        "model_path": str(config.model_path) if config.model_path else None,
        "package_dir": str(config.package_dir) if config.package_dir else None,
        "apple_silicon": config.apple_silicon,
        "mlx_installed": config.mlx_installed,
    }


def should_use_mlx_sam(preference: str = "auto") -> bool:
    pref = _normalize_preference(preference)
    if pref == "torch":
        return False
    config = resolve_mlx_sam_config()
    if pref == "mlx":
        if not config.available:
            raise MlxSamUnavailable(config.reason or "mlx_sam_unavailable")
        return True
    return config.available


def build_mlx_sam_predictor() -> MlxSamPredictorAdapter:
    config = resolve_mlx_sam_config()
    if not config.available or not config.model_path or not config.package_dir:
        raise MlxSamUnavailable(config.reason or "mlx_sam_unavailable")
    sam_mod, predictor_mod = _load_external_mlx_sam_modules(config.package_dir)
    sam_model = sam_mod.load(config.model_path)
    predictor = predictor_mod.SamPredictor(sam_model)
    return MlxSamPredictorAdapter(predictor)


def resolve_mlx_sam_config() -> MlxSamConfig:
    apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }
    mlx_installed = importlib.util.find_spec("mlx") is not None
    if not apple_silicon:
        return MlxSamConfig(False, "not_apple_silicon", None, None, apple_silicon, mlx_installed)
    if not mlx_installed:
        return MlxSamConfig(False, "mlx_not_installed", None, None, apple_silicon, mlx_installed)
    runtime_error = _mlx_runtime_error()
    if runtime_error:
        return MlxSamConfig(
            False,
            f"mlx_runtime_unavailable: {runtime_error}",
            None,
            None,
            apple_silicon,
            mlx_installed,
        )

    model_path = _resolve_mlx_sam_model_path()
    if model_path is None:
        return MlxSamConfig(False, "mlx_sam_model_path_missing", None, None, apple_silicon, mlx_installed)
    package_dir = _resolve_mlx_sam_package_dir()
    if package_dir is None:
        return MlxSamConfig(False, "mlx_sam_root_missing", model_path, None, apple_silicon, mlx_installed)
    return MlxSamConfig(True, None, model_path, package_dir, apple_silicon, mlx_installed)


def _resolve_mlx_sam_model_path() -> Optional[Path]:
    for key in MLX_SAM_MODEL_ENV_KEYS:
        raw = os.environ.get(key)
        path = _valid_model_path(raw) if raw else None
        if path is not None:
            return path
    checkpoint = os.environ.get("SAM_CHECKPOINT_PATH")
    return _valid_model_path(checkpoint) if checkpoint else None


def _valid_model_path(raw: Optional[str]) -> Optional[Path]:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.is_dir() and (path / "config.json").is_file() and (path / "model.safetensors").is_file():
        return path
    return None


def _resolve_mlx_sam_package_dir() -> Optional[Path]:
    for key in MLX_SAM_ROOT_ENV_KEYS:
        raw = os.environ.get(key)
        if not raw:
            continue
        candidate = _package_dir_from_root(Path(raw).expanduser())
        if candidate is not None:
            return candidate
    return None


def _package_dir_from_root(root: Path) -> Optional[Path]:
    if not root.is_absolute():
        root = Path.cwd() / root
    candidates = [
        root,
        root / "segment_anything",
        root / "segment_anything" / "segment_anything",
    ]
    for candidate in candidates:
        if (candidate / "sam.py").is_file() and (candidate / "predictor.py").is_file():
            return candidate.resolve()
    return None


def _load_external_mlx_sam_modules(package_dir: Path) -> Tuple[Any, Any]:
    package_dir = package_dir.resolve()
    digest = hashlib.sha1(str(package_dir).encode("utf-8")).hexdigest()[:12]
    package_name = f"_tator_mlx_sam_{digest}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        package.__loader__ = None
        package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        sys.modules[package_name] = package
    sam_mod = importlib.import_module(f"{package_name}.sam")
    predictor_mod = importlib.import_module(f"{package_name}.predictor")
    return sam_mod, predictor_mod


def _normalize_predict_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "point_coords": _to_mlx_array(kwargs.get("point_coords")),
        "point_labels": _to_mlx_array(kwargs.get("point_labels")),
        "box": _to_mlx_array(kwargs.get("box")),
        "mask_input": _to_mlx_array(kwargs.get("mask_input")),
        "multimask_output": kwargs.get("multimask_output", True),
        "return_logits": kwargs.get("return_logits", False),
    }
    point_coords = normalized["point_coords"]
    point_labels = normalized["point_labels"]
    box = normalized["box"]
    mask_input = normalized["mask_input"]

    if point_coords is not None and len(point_coords.shape) == 2:
        normalized["point_coords"] = point_coords[None, :, :]
    if point_labels is not None and len(point_labels.shape) == 1:
        normalized["point_labels"] = point_labels[None, :]
    if box is not None and len(box.shape) == 1:
        normalized["box"] = box[None, :]
    if mask_input is not None:
        normalized["mask_input"] = _normalize_mask_input(mask_input)
    return normalized


def _to_mlx_array(value: Any) -> Any:
    if value is None:
        return None
    if _is_mlx_array(value):
        return value
    import mlx.core as mx

    try:
        return mx.array(value)
    except RuntimeError as exc:
        if _is_mlx_device_unavailable_error(exc):
            return np.asarray(value)
        raise


def _normalize_mask_input(mask_input: Any) -> Any:
    if len(mask_input.shape) == 2:
        return mask_input[None, :, :, None]
    if len(mask_input.shape) == 3:
        # Meta SAM logits are CxHxW; MLX SAM expects BxHxWxC.
        if mask_input.shape[0] in {1, 3} and mask_input.shape[-1] not in {1, 3}:
            if _is_mlx_array(mask_input):
                import mlx.core as mx

                return mx.moveaxis(mask_input, 0, -1)[None, :, :, :]
            return np.moveaxis(mask_input, 0, -1)[None, :, :, :]
        return mask_input[None, :, :, :]
    return mask_input


def _is_mlx_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("mlx.")


def _is_mlx_device_unavailable_error(exc: BaseException) -> bool:
    text = str(exc)
    return "No Metal device" in text or "metal::load_device" in text


def _mlx_runtime_error() -> Optional[str]:
    try:
        import mlx.core as mx

        probe = mx.array([0.0])
        if hasattr(mx, "eval"):
            mx.eval(probe)
        return None
    except Exception as exc:  # pragma: no cover - depends on local Metal availability.
        return str(exc) or exc.__class__.__name__


def _to_segment_anything_outputs(masks: Any, scores: Any, logits: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks_np = _mlx_to_numpy(masks)
    scores_np = _mlx_to_numpy(scores)
    logits_np = _mlx_to_numpy(logits)
    return (
        _normalize_mlx_mask_output(masks_np),
        _normalize_mlx_score_output(scores_np),
        _normalize_mlx_mask_output(logits_np),
    )


def _mlx_to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value)


def _normalize_mlx_score_output(scores: np.ndarray) -> np.ndarray:
    if scores.ndim == 2 and scores.shape[0] == 1:
        return scores[0]
    return scores


def _normalize_mlx_mask_output(masks: np.ndarray) -> np.ndarray:
    if masks.ndim != 4:
        return masks
    if masks.shape[0] == 1:
        return np.moveaxis(masks[0], -1, 0)
    if masks.shape[-1] == 1:
        return masks[..., 0]
    return np.moveaxis(masks, -1, 1)


def _normalize_preference(preference: str) -> str:
    pref = (preference or "auto").strip().lower()
    if pref in {"mlx", "torch", "auto"}:
        return pref
    return "auto"
