"""GPU device helpers."""

from __future__ import annotations

from typing import Any, Sequence


def _torch_mps_available_impl(torch_module: Any) -> bool:
    """Return whether PyTorch can dispatch work to Apple Metal/MPS."""
    try:
        mps_backend = getattr(torch_module.backends, "mps", None)
        return bool(mps_backend and mps_backend.is_available())
    except Exception:
        return False


def _torch_mps_built_impl(torch_module: Any) -> bool:
    try:
        mps_backend = getattr(torch_module.backends, "mps", None)
        is_built = getattr(mps_backend, "is_built", None)
        return bool(mps_backend and is_built and is_built())
    except Exception:
        return False


def _resolve_torch_inference_device_impl(
    device_pref: str,
    *,
    torch_module: Any,
    prefer_mps: bool = True,
) -> str:
    """Resolve an inference device across CUDA, Apple MPS, and CPU."""
    pref = str(device_pref or "auto").strip().lower()
    if pref in {"", "auto"}:
        if torch_module.cuda.is_available():
            return "cuda"
        if prefer_mps and _torch_mps_available_impl(torch_module):
            return "mps"
        return "cpu"
    if pref.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("cuda_requested_but_unavailable")
    if pref.startswith("mps") and not _torch_mps_available_impl(torch_module):
        raise RuntimeError("mps_requested_but_unavailable")
    return pref


def _validate_cuda_device_ids_impl(
    device_ids: Sequence[int],
    *,
    torch_module: Any,
    http_exception_cls: Any,
) -> None:
    if not device_ids:
        return
    if not torch_module.cuda.is_available():
        raise http_exception_cls(status_code=400, detail="qwen_devices_unavailable")
    max_id = torch_module.cuda.device_count() - 1
    invalid = [device for device in device_ids if device < 0 or device > max_id]
    if invalid:
        raise http_exception_cls(
            status_code=400,
            detail=f"qwen_invalid_devices:available=0-{max_id}",
        )
