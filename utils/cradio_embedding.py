"""Shared C-RADIOv4 embedding helpers.

C-RADIOv4 is published as Hugging Face remote-code models that return a
global ``summary`` tensor and flattened ``spatial_features`` tokens.  We keep
the model-specific loading and pooling rules here so Class Split, data
ingestion, local SALAD, and auto-class all use the same contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CRadioBackendStatus:
    requested: str
    resolved: str
    available: bool
    detail: str


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


def cradio_backend_status(requested: Optional[str] = None) -> CRadioBackendStatus:
    raw = str(requested or os.environ.get("CRADIO_BACKEND") or "auto").strip().lower()
    if raw in {"mlx", "apple_mlx"}:
        return CRadioBackendStatus(
            requested=raw,
            resolved="mlx",
            available=False,
            detail=(
                "C-RADIOv4 is currently published as a Hugging Face/Torch "
                "remote-code model; no official MLX C-RADIOv4 implementation "
                "or converted checkpoint is available."
            ),
        )
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


def resolve_cradio_torch_device(requested: Optional[str] = None) -> str:
    status = cradio_backend_status(requested)
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
    target_device = device_name or resolve_cradio_torch_device(backend)
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
