from __future__ import annotations

from typing import Any


def _sam3_clear_device_pinned_caches_impl(model: Any) -> None:
    """
    SAM3 upstream precomputes some internal caches on `cuda` (i.e. cuda:0) during module
    construction to help torch.compile. Those caches are stored in plain Python containers and
    are NOT moved by `model.to(cuda:N)`, which can break multi-GPU inference (device mismatch).

    We don't rely on torch.compile in this server path, so it's safe to clear these caches after
    moving the model to its target device.
    """
    if model is None:
        return
    try:
        modules = model.modules()
    except Exception:
        return
    for m in modules:
        try:
            cls_name = getattr(m, "__class__", type(m)).__name__
            cls_mod = getattr(getattr(m, "__class__", type(m)), "__module__", "")
        except Exception:
            cls_name = ""
            cls_mod = ""
        # Position encoding cache: dict of tensors keyed by shape (upstream).
        if cls_name == "PositionEmbeddingSine" or str(cls_mod).endswith("position_encoding"):
            try:
                cache = getattr(m, "cache", None)
                if isinstance(cache, dict):
                    cache.clear()
            except Exception:
                pass
        # Decoder cache: precomputed coord cache tuple (upstream).
        if hasattr(m, "compilable_cord_cache"):
            try:
                setattr(m, "compilable_cord_cache", None)
            except Exception:
                pass
        if hasattr(m, "coord_cache"):
            try:
                cache = getattr(m, "coord_cache", None)
                if isinstance(cache, dict):
                    cache.clear()
            except Exception:
                pass
