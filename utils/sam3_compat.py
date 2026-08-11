"""Compatibility helpers for importing upstream SAM3 on macOS."""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any, Optional


def _sam3_prepare_macos_import_impl() -> Optional[dict[str, Any]]:
    """Install a temporary Triton import shim when SAM3 is used on macOS without Triton."""
    if sys.platform != "darwin":
        return None
    if importlib.util.find_spec("triton") is not None:
        return None
    existing = sys.modules.get("triton")
    if existing is not None and not getattr(existing, "_tator_sam3_stub", False):
        return None

    # Torchvision imports torch._dynamo, which probes Triton if a module named
    # triton exists. Load torchvision first so our import shim is only seen by
    # SAM3's CUDA-only helper modules.
    try:
        import torchvision  # noqa: F401
    except Exception:
        pass

    installed: list[str] = []

    def _decorator(*args: Any, **_kwargs: Any) -> Any:
        if args and callable(args[0]) and len(args) == 1:
            return args[0]
        return lambda fn: fn

    triton = types.ModuleType("triton")
    triton._tator_sam3_stub = True  # type: ignore[attr-defined]
    triton.jit = _decorator  # type: ignore[attr-defined]
    triton.autotune = lambda *_args, **_kwargs: (lambda fn: fn)  # type: ignore[attr-defined]
    triton.heuristics = lambda *_args, **_kwargs: (lambda fn: fn)  # type: ignore[attr-defined]
    triton.cdiv = lambda x, y: (x + y - 1) // y  # type: ignore[attr-defined]

    class _Config:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    triton.Config = _Config  # type: ignore[attr-defined]

    language = types.ModuleType("triton.language")

    class _DType:
        pass

    language.dtype = _DType  # type: ignore[attr-defined]
    language.constexpr = object  # type: ignore[attr-defined]
    triton.language = language  # type: ignore[attr-defined]

    for name, module in (("triton", triton), ("triton.language", language)):
        if name not in sys.modules:
            sys.modules[name] = module
            installed.append(name)

    return {"installed": installed}


def _sam3_finalize_macos_import_impl(token: Optional[dict[str, Any]]) -> None:
    """Patch CUDA-only SAM3 helpers and remove the temporary Triton import shim."""
    if token is None:
        return
    _patch_sam3_position_encoding_precompute()
    _patch_sam3_decoder_cuda_coord_precompute()
    _patch_sam3_fused_addmm()
    _patch_sam3_pin_memory_noop()
    _patch_sam3_edt_fallback()
    _patch_sam3_connected_components_fallback()
    for name in reversed(token.get("installed") or []):
        module = sys.modules.get(name)
        if getattr(module, "_tator_sam3_stub", False) or name == "triton.language":
            sys.modules.pop(name, None)


def _patch_sam3_position_encoding_precompute() -> None:
    try:
        import torch
        import sam3.model.position_encoding as position_encoding
    except Exception:
        return
    if torch.cuda.is_available():
        return
    cls = getattr(position_encoding, "PositionEmbeddingSine", None)
    if cls is None or getattr(cls, "_tator_macos_precompute_patch", False):
        return
    original_init = cls.__init__

    def _init_without_cuda_precompute(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["precompute_resolution"] = None
        original_init(self, *args, **kwargs)

    cls.__init__ = _init_without_cuda_precompute
    cls._tator_macos_precompute_patch = True


def _patch_sam3_decoder_cuda_coord_precompute() -> None:
    try:
        import torch
        import sam3.model.decoder as decoder
    except Exception:
        return
    if torch.cuda.is_available():
        return
    cls = getattr(decoder, "TransformerDecoder", None)
    if cls is None or getattr(cls, "_tator_macos_coord_patch", False):
        return
    original_get_coords = cls._get_coords

    def _get_coords_without_cuda(H: int, W: int, device: Any) -> Any:
        if str(device).startswith("cuda"):
            device = "cpu"
        return original_get_coords(H, W, device)

    cls._get_coords = staticmethod(_get_coords_without_cuda)
    cls._tator_macos_coord_patch = True


def _patch_sam3_fused_addmm() -> None:
    try:
        import torch
        import torch.nn.functional as F
        import sam3.model.vitdet as vitdet
        import sam3.perflib.fused as fused
    except Exception:
        return
    if torch.cuda.is_available():
        return
    if getattr(fused, "_tator_macos_addmm_patch", False):
        return

    def _addmm_act_float32(activation: Any, linear: Any, mat1: Any) -> Any:
        weight_dtype = getattr(getattr(linear, "weight", None), "dtype", None)
        if weight_dtype is not None and hasattr(mat1, "to"):
            mat1 = mat1.to(dtype=weight_dtype)
        y = linear(mat1)
        if activation in {F.relu, torch.nn.ReLU}:
            return F.relu(y)
        if activation in {F.gelu, torch.nn.GELU}:
            return F.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")

    fused.addmm_act = _addmm_act_float32
    fused._tator_macos_addmm_patch = True
    vitdet.addmm_act = _addmm_act_float32


def _patch_sam3_pin_memory_noop() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available() or getattr(torch.Tensor, "_tator_macos_pin_patch", False):
        return
    original_pin_memory = torch.Tensor.pin_memory

    def _pin_memory_noop(self: Any, device: Any = None) -> Any:
        return self

    torch.Tensor.pin_memory = _pin_memory_noop  # type: ignore[assignment]
    torch.Tensor._tator_macos_pin_patch = True  # type: ignore[attr-defined]
    torch.Tensor._tator_macos_original_pin_memory = original_pin_memory  # type: ignore[attr-defined]


def _patch_sam3_edt_fallback() -> None:
    try:
        import numpy as np
        import torch
        from scipy import ndimage
    except Exception:
        return

    def _edt_fallback(data: Any) -> Any:
        if not isinstance(data, torch.Tensor):
            raise TypeError("edt_fallback_requires_tensor")
        if data.dim() != 3:
            raise AssertionError("edt_fallback_requires_bhw_tensor")
        device = data.device
        masks = data.detach().to("cpu").bool().numpy()
        distances = [ndimage.distance_transform_edt(mask) for mask in masks]
        out = torch.from_numpy(np.stack(distances).astype("float32", copy=False))
        return out.to(device=device)

    for module_name in ("sam3.model.edt", "sam3.model.sam3_tracker_utils"):
        module = sys.modules.get(module_name)
        if module is not None:
            try:
                setattr(module, "edt_triton", _edt_fallback)
            except Exception:
                pass


def _patch_sam3_connected_components_fallback() -> None:
    try:
        import numpy as np
        import torch
        from scipy import ndimage
        import sam3.perflib.connected_components as connected_components
    except Exception:
        return

    def _connected_components_cpu_single(values: Any) -> Any:
        if not isinstance(values, torch.Tensor):
            raise TypeError("connected_components_requires_tensor")
        labels_np, num = ndimage.label(values.detach().to("cpu").numpy().astype(bool))
        labels_np = labels_np.astype("int64", copy=False)
        if num <= 0:
            counts_np = np.zeros_like(labels_np, dtype="int64")
        else:
            component_sizes = np.bincount(labels_np.reshape(-1), minlength=num + 1)
            counts_np = component_sizes[labels_np].astype("int64", copy=False)
        labels = torch.from_numpy(labels_np)
        counts = torch.from_numpy(counts_np)
        return labels, counts

    connected_components.connected_components_cpu_single = _connected_components_cpu_single
