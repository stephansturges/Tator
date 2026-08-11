"""Shared C-RADIOv4 embedding helpers.

C-RADIOv4 returns a global ``summary`` tensor and flattened spatial tokens.  We
keep the model-specific loading and pooling rules here so Class Split, data
ingestion, local SALAD, and auto-class all use the same contract.  On Mac, the
``auto`` backend prefers the local ``~/cradio_mlx`` MLX runtime when it is
present; CUDA/MPS/CPU Torch remains the fallback path.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


CRADIO_DEFAULT_MODEL = os.environ.get("CRADIO_MODEL_NAME", "nvidia/C-RADIOv4-SO400M")
CRADIO_SUPPORTED_MODELS = [
    "nvidia/C-RADIOv4-SO400M",
    "nvidia/C-RADIOv4-H",
]
CRADIO_POOLING_MODES = ["summary", "spatial_mean", "summary_spatial_concat"]
CRADIO_DEFAULT_POOLING = "summary"
CRADIO_MLX_DTYPE = os.environ.get("CRADIO_MLX_DTYPE", "bfloat16")


@dataclass(frozen=True)
class CRadioBackendStatus:
    requested: str
    resolved: str
    available: bool
    detail: str


def _cradio_torch_backend_status(raw: str) -> CRadioBackendStatus:
    if raw in {"cuda", "torch_cuda"}:
        return CRadioBackendStatus(
            requested=raw,
            resolved="cuda",
            available=torch.cuda.is_available(),
            detail="CUDA Torch backend" if torch.cuda.is_available() else "CUDA is not available",
        )
    if raw in {"mps", "metal", "torch_mps"}:
        available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        return CRadioBackendStatus(
            requested=raw,
            resolved="mps",
            available=available,
            detail="Apple Metal/MPS Torch backend" if available else "MPS is not available",
        )
    if raw in {"cpu", "torch_cpu"}:
        return CRadioBackendStatus(
            requested=raw,
            resolved="cpu",
            available=True,
            detail="CPU Torch backend",
        )
    if torch.cuda.is_available():
        return CRadioBackendStatus(requested=raw, resolved="cuda", available=True, detail="CUDA Torch backend")
    if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return CRadioBackendStatus(requested=raw, resolved="mps", available=True, detail="Apple Metal/MPS Torch backend")
    return CRadioBackendStatus(requested=raw, resolved="cpu", available=True, detail="CPU Torch backend")


def _cradio_mlx_source_root() -> Path:
    return Path(os.environ.get("CRADIO_MLX_SRC", str(_cradio_mlx_root() / "src"))).expanduser()


def _cradio_mlx_root() -> Path:
    return Path(os.environ.get("CRADIO_MLX_ROOT", "~/cradio_mlx")).expanduser()


def _cradio_mlx_variant(model_name: Any) -> str:
    model = normalize_cradio_model(model_name).lower()
    if "radiov4-h" in model or model.endswith("-h") or model.endswith("/h"):
        return "h"
    return "so400m"


def _cradio_mlx_checkpoint(model_name: Any) -> Path:
    variant = _cradio_mlx_variant(model_name)
    generic_override = os.environ.get("CRADIO_MLX_CHECKPOINT")
    if generic_override:
        return Path(generic_override).expanduser()
    env_key = "CRADIO_MLX_H_CHECKPOINT" if variant == "h" else "CRADIO_MLX_SO400M_CHECKPOINT"
    specific_override = os.environ.get(env_key)
    if specific_override:
        return Path(specific_override).expanduser()
    checkpoint_name = "c-radiov4-h" if variant == "h" else "c-radiov4-so400m"
    return _cradio_mlx_root() / "checkpoints" / checkpoint_name


def _import_cradio_mlx() -> Tuple[Any, Any]:
    try:
        from cradio_mlx import MLXHEncoder, MLXSO400MEncoder

        return MLXHEncoder, MLXSO400MEncoder
    except Exception as first_exc:  # noqa: BLE001
        source_root = _cradio_mlx_source_root()
        if source_root.exists():
            source_text = str(source_root)
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
            try:
                from cradio_mlx import MLXHEncoder, MLXSO400MEncoder

                return MLXHEncoder, MLXSO400MEncoder
            except Exception as second_exc:  # noqa: BLE001
                raise RuntimeError(f"cradio_mlx_import_failed:{second_exc}") from second_exc
        raise RuntimeError(f"cradio_mlx_import_failed:{first_exc}") from first_exc


def _cradio_mlx_backend_status(model_name: Optional[str] = None, *, requested: str = "mlx") -> CRadioBackendStatus:
    checkpoint = _cradio_mlx_checkpoint(model_name or CRADIO_DEFAULT_MODEL)
    model_path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    if not model_path.exists():
        return CRadioBackendStatus(
            requested=requested,
            resolved="mlx",
            available=False,
            detail=f"Local MLX C-RADIOv4 checkpoint not found: {model_path}",
        )
    try:
        _import_cradio_mlx()
        import mlx.core as mx  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return CRadioBackendStatus(
            requested=requested,
            resolved="mlx",
            available=False,
            detail=str(exc),
        )
    return CRadioBackendStatus(
        requested=requested,
        resolved="mlx",
        available=True,
        detail=f"Local MLX C-RADIOv4 backend ({model_path})",
    )


def normalize_cradio_pooling(value: Any) -> str:
    mode = str(value or CRADIO_DEFAULT_POOLING).strip().lower()
    if mode in {"spatial_mean", "spatial", "patch_mean", "token_mean", "features_mean"}:
        return "spatial_mean"
    if mode in {"concat", "summary_spatial", "summary+spatial", "summary_spatial_concat"}:
        return "summary_spatial_concat"
    return "summary"


def normalize_cradio_model(value: Any) -> str:
    model = str(value or CRADIO_DEFAULT_MODEL).strip()
    return model or CRADIO_DEFAULT_MODEL


def cradio_backend_status(
    requested: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
) -> CRadioBackendStatus:
    raw = str(requested or os.environ.get("CRADIO_BACKEND") or "auto").strip().lower()
    if raw in {"mlx", "apple_mlx"}:
        return _cradio_mlx_backend_status(model_name, requested=raw)
    if raw == "auto":
        mlx = _cradio_mlx_backend_status(model_name, requested=raw)
        if platform.system().lower() == "darwin" and mlx.available:
            return mlx
    return _cradio_torch_backend_status(raw)


def cradio_capabilities() -> Dict[str, Any]:
    auto = cradio_backend_status("auto")
    cuda = cradio_backend_status("cuda")
    mps = cradio_backend_status("mps")
    mlx = cradio_backend_status("mlx")
    return {
        "models": list(CRADIO_SUPPORTED_MODELS),
        "default_model": CRADIO_DEFAULT_MODEL,
        "pooling_modes": list(CRADIO_POOLING_MODES),
        "default_pooling": CRADIO_DEFAULT_POOLING,
        "backend": {
            "default": "auto",
            "auto_resolved": auto.resolved,
            "torch_cuda": {"available": cuda.available, "detail": cuda.detail},
            "torch_mps": {"available": mps.available, "detail": mps.detail},
            "torch_cpu": {"available": True, "detail": "CPU Torch backend"},
            "mlx": {"available": mlx.available, "detail": mlx.detail},
        },
    }


def resolve_cradio_torch_device(
    requested: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
) -> str:
    status = cradio_backend_status(requested, model_name=model_name)
    if not status.available:
        raise RuntimeError(status.detail)
    return status.resolved


def resolve_cradio_aux_torch_device(requested: Optional[str] = None) -> str:
    """Resolve a Torch device for layers that consume C-RADIO numpy outputs.

    The C-RADIO encoder may run in MLX, but local SALAD heads are PyTorch
    modules.  This helper keeps those heads on a valid Torch device without
    changing the encoder backend choice.
    """

    raw = str(requested or os.environ.get("CRADIO_AUX_TORCH_DEVICE") or "auto").strip().lower()
    if raw in {"mlx", "apple_mlx"}:
        raw = "auto"
    status = _cradio_torch_backend_status(raw)
    if not status.available:
        raise RuntimeError(status.detail)
    return status.resolved


def load_cradio_backbone(
    model_name: Optional[str] = None,
    device_name: Optional[str] = None,
    *,
    backend: Optional[str] = None,
) -> Tuple[Any, Any, str, str]:
    resolved_model = normalize_cradio_model(model_name)
    target_device = device_name or resolve_cradio_torch_device(backend, model_name=resolved_model)
    if target_device == "mlx":
        status = _cradio_mlx_backend_status(resolved_model, requested=backend or "mlx")
        if not status.available:
            raise RuntimeError(status.detail)
        try:
            MLXHEncoder, MLXSO400MEncoder = _import_cradio_mlx()
            variant = _cradio_mlx_variant(resolved_model)
            encoder_cls = MLXHEncoder if variant == "h" else MLXSO400MEncoder
            model = encoder_cls.load(_cradio_mlx_checkpoint(resolved_model), dtype=CRADIO_MLX_DTYPE)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"cradio_mlx_load_failed:{exc}") from exc
        return model, None, resolved_model, "mlx"
    try:
        from transformers import AutoModel, CLIPImageProcessor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cradio_transformers_unavailable:{exc}") from exc
    try:
        processor = CLIPImageProcessor.from_pretrained(resolved_model)
        model = AutoModel.from_pretrained(resolved_model, trust_remote_code=True)
        model.eval().to(target_device)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cradio_load_failed:{exc}") from exc
    return model, processor, resolved_model, target_device


def _unpack_cradio_outputs(outputs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    def first_present(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        return outputs[0], outputs[1]
    if isinstance(outputs, dict):
        summary = first_present(outputs.get("summary"), outputs.get("pooler_output"), outputs.get("image_embeds"))
        spatial = first_present(outputs.get("spatial_features"), outputs.get("last_hidden_state"))
        if summary is not None and spatial is not None:
            return summary, spatial
    summary = first_present(getattr(outputs, "summary", None), getattr(outputs, "pooler_output", None))
    spatial = first_present(getattr(outputs, "spatial_features", None), getattr(outputs, "last_hidden_state", None))
    if summary is None or spatial is None:
        raise RuntimeError("cradio_output_contract_unrecognized")
    return summary, spatial


def _l2_normalize_np(values: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(denom, 1e-12)


def _coerce_mlx_size_dimension(value: int) -> int:
    value = max(16, int(value or 512))
    return max(16, int(round(value / 16.0)) * 16)


def _resolve_cradio_mlx_image_size(images: Sequence[Image.Image]) -> int | Tuple[int, int]:
    override = os.environ.get("CRADIO_MLX_IMAGE_SIZE")
    if override:
        try:
            value = _coerce_mlx_size_dimension(int(float(override)))
            return value
        except Exception:
            pass
    preserve_input = str(os.environ.get("CRADIO_MLX_PRESERVE_INPUT_SIZE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not preserve_input:
        return 512
    sizes = [getattr(img, "size", None) for img in images if getattr(img, "size", None)]
    if sizes and all(size == sizes[0] for size in sizes):
        width, height = sizes[0]
        width = _coerce_mlx_size_dimension(int(width))
        height = _coerce_mlx_size_dimension(int(height))
        if width == height:
            return width
        return (height, width)
    return 512


def _encode_cradio_images_mlx(
    model: Any,
    images: Sequence[Image.Image],
    *,
    pooling: str = CRADIO_DEFAULT_POOLING,
    normalize: bool = True,
    return_tokens: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = model.encode_batch(list(images), image_size=_resolve_cradio_mlx_image_size(images))
    summary = np.asarray(result.summary, dtype=np.float32)
    spatial = np.asarray(result.spatial, dtype=np.float32)
    if summary.ndim == 1:
        summary = summary.reshape(1, -1)
    if spatial.ndim == 2:
        spatial = spatial.reshape(summary.shape[0], -1, spatial.shape[-1])
    spatial_mean = spatial.mean(axis=1) if spatial.size else np.empty((summary.shape[0], 0), dtype=np.float32)
    pooling_norm = normalize_cradio_pooling(pooling)
    if pooling_norm == "spatial_mean":
        feats = spatial_mean
    elif pooling_norm == "summary_spatial_concat":
        feats = np.concatenate(
            [
                _l2_normalize_np(summary, axis=-1),
                _l2_normalize_np(spatial_mean, axis=-1),
            ],
            axis=-1,
        )
    else:
        feats = summary
    feats = np.asarray(feats, dtype=np.float32)
    if normalize:
        feats = _l2_normalize_np(feats, axis=-1)
    if not return_tokens:
        return feats
    return feats, spatial.astype(np.float32, copy=False), summary.astype(np.float32, copy=False)


def encode_cradio_images(
    model: Any,
    processor: Any,
    device_name: str,
    images: Sequence[Image.Image],
    *,
    pooling: str = CRADIO_DEFAULT_POOLING,
    normalize: bool = True,
    return_tokens: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not images:
        empty = np.empty((0, 0), dtype=np.float32)
        if return_tokens:
            return empty, np.empty((0, 0, 0), dtype=np.float32), empty
        return empty
    if str(device_name or "").strip().lower() == "mlx" or processor is None:
        return _encode_cradio_images_mlx(
            model,
            images,
            pooling=pooling,
            normalize=normalize,
            return_tokens=return_tokens,
        )
    with torch.no_grad():
        inputs = processor(images=list(images), return_tensors="pt", do_resize=True)
        pixel_values = inputs["pixel_values"].to(device_name)
        outputs = model(pixel_values)
        summary, spatial = _unpack_cradio_outputs(outputs)
        summary = summary.float()
        spatial = spatial.float()
        spatial_mean = spatial.mean(dim=1)
        pooling_norm = normalize_cradio_pooling(pooling)
        if pooling_norm == "spatial_mean":
            feats = spatial_mean
        elif pooling_norm == "summary_spatial_concat":
            feats = torch.cat(
                [
                    torch.nn.functional.normalize(summary, dim=-1),
                    torch.nn.functional.normalize(spatial_mean, dim=-1),
                ],
                dim=-1,
            )
        else:
            feats = summary
        if normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
    feats_np = feats.detach().cpu().numpy().astype(np.float32)
    if not return_tokens:
        return feats_np
    return (
        feats_np,
        spatial.detach().cpu().numpy().astype(np.float32),
        summary.detach().cpu().numpy().astype(np.float32),
    )
