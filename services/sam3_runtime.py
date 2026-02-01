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


def _reset_sam3_runtime_impl(
    *,
    state: dict,
    predictor_manager: Any,
    torch_module: Any,
) -> None:
    state["sam3_text_model"] = None
    state["sam3_text_processor"] = None
    state["sam3_text_device"] = None
    try:
        predictor_manager.unload_all()
    except Exception:
        pass
    if torch_module.cuda.is_available():
        try:
            torch_module.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _build_backend_for_variant_impl(
    variant: str,
    *,
    sam3_backend_cls: Any,
    sam1_backend_cls: Any,
) -> Any:
    normalized = (variant or "sam1").lower()
    if normalized == "sam3":
        return sam3_backend_cls()
    return sam1_backend_cls()


def _ensure_sam3_text_runtime_impl(
    *,
    state: dict,
    lock: Any,
    resolve_device_fn: Any,
    sam3_import_error: Optional[str],
    build_model_fn: Any,
    processor_cls: Any,
    sam3_checkpoint: Optional[str],
    sam3_bpe_path: Any,
    clear_caches_fn: Any,
    http_exception_cls: Any,
    http_503: int,
    http_500: int,
) -> tuple:
    with lock:
        if (
            state.get("sam3_text_model") is not None
            and state.get("sam3_text_processor") is not None
            and state.get("sam3_text_device") is not None
        ):
            return state["sam3_text_model"], state["sam3_text_processor"], state["sam3_text_device"]
        device = resolve_device_fn()
        if sam3_import_error is not None or build_model_fn is None or processor_cls is None:
            detail = f"sam3_text_unavailable:{sam3_import_error}"
            raise http_exception_cls(status_code=http_503, detail=detail)
        try:
            device_str = str(device) if getattr(device, "type", None) == "cuda" else "cpu"
            enable_seg = True
            if sam3_checkpoint:
                model = build_model_fn(
                    checkpoint_path=sam3_checkpoint,
                    device=device_str,
                    load_from_HF=False,
                    enable_segmentation=enable_seg,
                    bpe_path=str(sam3_bpe_path),
                ).to(device)
            else:
                model = build_model_fn(
                    device=device_str,
                    enable_segmentation=enable_seg,
                    bpe_path=str(sam3_bpe_path),
                ).to(device)
            clear_caches_fn(model)
            processor = processor_cls(model, device=device_str)
        except Exception as exc:  # noqa: BLE001
            raise http_exception_cls(status_code=http_500, detail=f"sam3_text_load_failed:{exc}") from exc
        state["sam3_text_model"] = model
        state["sam3_text_processor"] = processor
        state["sam3_text_device"] = device
        return state["sam3_text_model"], state["sam3_text_processor"], state["sam3_text_device"]


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
