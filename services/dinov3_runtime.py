from __future__ import annotations


def _dinov3_resolve_device_impl(requested: str, *, cuda_disabled: bool) -> str:
    normalized = str(requested or "").strip() or "cpu"
    if cuda_disabled and normalized.startswith("cuda"):
        return "cpu"
    return normalized
