"""DINOv3 runtime helpers."""

from __future__ import annotations

import os
from pathlib import Path


def native_dinov3_package(model_name):
    root = os.environ.get("DINOV3_TORCH_MODEL_ROOT")
    if not root:
        return None
    if any(part in {"", ".", ".."} for part in model_name.split("/")) or "\\" in model_name:
        raise ValueError("dinov3_model_name_invalid")
    path = Path(root).expanduser() / model_name.replace("/", "--")
    if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        raise RuntimeError("dinov3_verified_package_missing:" + model_name)
    return path


class DinoV3RecipeProcessor:
    """The established MLX recipe's PIL resize followed by ImageNet scaling.

    Used with explicitly staged native packages so a Transformers processor
    release cannot silently move resizing before scaling or change resolution.
    """
    def __call__(self, *, images, return_tensors="pt", **kwargs):
        import numpy as np
        import torch
        from PIL import Image
        if return_tensors != "pt" or kwargs:
            raise ValueError("dinov3_recipe_processor_options_invalid")
        pixels = []
        for image in images:
            rgb = image.convert("RGB")
            resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
            pixels.append(np.asarray(resized, dtype=np.float32))
            resized.close()
            rgb.close()
        batch = np.stack(pixels) / np.float32(255)
        batch = (batch - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        return {"pixel_values": torch.from_numpy(batch.transpose(0, 3, 1, 2).copy())}


def _dinov3_resolve_device_impl(requested: str, *, cuda_disabled: bool) -> str:
    normalized = str(requested or "").strip() or "cpu"
    if cuda_disabled and normalized.startswith("cuda"):
        return "cpu"
    return normalized
