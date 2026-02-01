from __future__ import annotations

from typing import Any


def _set_sam3_device_pref_impl(
    device_index: int,
    *,
    torch_module: Any,
    state: dict,
) -> None:
    if torch_module.cuda.is_available():
        state["device_pref"] = f"cuda:{device_index}"
    else:
        state["device_pref"] = "cpu"


def _resolve_sam3_device_impl(
    sam3_device_pref: str,
    *,
    torch_module: Any,
    http_exception_cls: Any,
    http_400: int,
) -> Any:
    if sam3_device_pref in {"", "auto"}:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    try:
        return torch_module.device(sam3_device_pref)
    except Exception as exc:  # noqa: BLE001
        raise http_exception_cls(
            status_code=http_400,
            detail=f"invalid_sam3_device:{sam3_device_pref}:{exc}",
        ) from exc


def _resolve_sam3_mining_devices_impl(
    sam3_device_pref: str,
    *,
    torch_module: Any,
    logger: Any,
) -> list:
    """
    Resolve the list of devices to use for agent mining. If SAM3_DEVICE specifies an explicit device
    (or comma-separated list), honor it; otherwise fan out across all available CUDA devices, falling
    back to CPU when needed.
    """
    devices: list = []
    if sam3_device_pref not in {"", "auto"}:
        for part in sam3_device_pref.split(","):
            name = part.strip()
            if not name:
                continue
            try:
                devices.append(torch_module.device(name))
            except Exception:
                logger.warning("Invalid SAM3 device in SAM3_DEVICE=%s", name)
    if not devices and torch_module.cuda.is_available():
        try:
            for idx in range(torch_module.cuda.device_count()):
                devices.append(torch_module.device(f"cuda:{idx}"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to enumerate CUDA devices for mining: %s", exc)
            devices = []
    if not devices:
        devices = [torch_module.device("cpu")]
    return devices


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
