from __future__ import annotations

from typing import Any, Callable, Optional


def _reset_qwen_runtime_impl(
    *,
    state: dict,
    torch_module: Any,
) -> None:
    state["qwen_model"] = None
    state["qwen_processor"] = None
    state["qwen_device"] = None
    state["loaded_qwen_model_id"] = None
    state["qwen_last_error"] = None
    if torch_module.cuda.is_available():
        try:
            torch_module.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _unload_qwen_runtime_impl(
    *,
    state: dict,
    torch_module: Any,
    gc_module: Any,
    logger: Any,
    deque_factory: Callable[[], Any],
) -> None:
    try:
        del state["qwen_model"]
    except Exception:
        pass
    try:
        del state["qwen_processor"]
    except Exception:
        pass
    state["qwen_model"] = None
    state["qwen_processor"] = None
    state["loaded_qwen_model_id"] = None
    state["qwen_caption_cache"] = {}
    state["qwen_caption_order"] = deque_factory()
    cuda_alloc: Optional[int] = None
    cuda_reserved: Optional[int] = None
    if torch_module.cuda.is_available():
        try:
            torch_module.cuda.empty_cache()
            torch_module.cuda.ipc_collect()
            cuda_alloc = int(torch_module.cuda.memory_allocated())
            cuda_reserved = int(torch_module.cuda.memory_reserved())
        except Exception:
            pass
    state["qwen_device"] = None
    try:
        gc_module.collect()
    except Exception:
        pass
    if cuda_alloc is not None or cuda_reserved is not None:
        logger.info(
            "[qwen] after unload: cuda_alloc=%s bytes cuda_reserved=%s bytes",
            cuda_alloc,
            cuda_reserved,
        )


def _resolve_qwen_device_impl(
    qwen_device_pref: str,
    *,
    torch_module: Any,
) -> str:
    if qwen_device_pref and qwen_device_pref != "auto":
        if qwen_device_pref.startswith("cuda") and not torch_module.cuda.is_available():
            raise RuntimeError("cuda_requested_but_unavailable")
        if qwen_device_pref.startswith("mps"):
            mps_backend = getattr(torch_module.backends, "mps", None)
            if not mps_backend or not mps_backend.is_available():  # type: ignore[attr-defined]
                raise RuntimeError("mps_requested_but_unavailable")
        return qwen_device_pref
    if torch_module.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend and mps_backend.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def _ensure_qwen_ready_for_caption_impl(
    model_id_override: str,
    *,
    state: dict,
    qwen_lock: Any,
    import_error: Optional[str],
    qwen_model_cls: Any,
    qwen_processor_cls: Any,
    process_vision_info: Any,
    packaging_version: Any,
    min_transformers: str,
    resolve_device_fn: Any,
    device_pref: str,
    torch_module: Any,
    load_qwen_model_fn: Any,
    hf_offline_enabled_fn: Any,
    set_hf_offline_fn: Any,
    enable_hf_offline_defaults_fn: Any,
    strip_model_suffix_fn: Any,
    format_load_error_fn: Any,
    min_pixels: int,
    max_pixels: int,
    caption_cache_limit: int,
    evict_entry_fn: Any,
    logger: Any,
) -> tuple:
    if import_error is not None or qwen_model_cls is None or qwen_processor_cls is None or process_vision_info is None:
        detail = f"qwen_dependencies_missing:{import_error}"
        raise RuntimeError(detail)
    if packaging_version is not None:
        try:
            import transformers  # local import to avoid import-time failures

            if packaging_version.parse(transformers.__version__) < packaging_version.parse(min_transformers):
                raise RuntimeError(f"qwen_transformers_too_old:{transformers.__version__}<{min_transformers}")
        except RuntimeError:
            raise
        except Exception:
            pass
    cache_key = f"caption:{model_id_override}"
    cache_limit = max(0, int(caption_cache_limit or 0))
    if cache_limit == 0 and state.get("qwen_caption_cache"):
        for key, entry in list(state["qwen_caption_cache"].items()):
            evict_entry_fn(key, entry)
        state["qwen_caption_cache"].clear()
        state["qwen_caption_order"].clear()
    cached = state["qwen_caption_cache"].get(cache_key)
    if cached and cache_limit:
        try:
            state["qwen_caption_order"].remove(cache_key)
        except ValueError:
            pass
        state["qwen_caption_order"].append(cache_key)
        return cached
    with qwen_lock:
        cached = state["qwen_caption_cache"].get(cache_key)
        if cached and cache_limit:
            try:
                state["qwen_caption_order"].remove(cache_key)
            except ValueError:
                pass
            state["qwen_caption_order"].append(cache_key)
            return cached
        try:
            device = resolve_device_fn()
        except RuntimeError as exc:  # noqa: BLE001
            state["qwen_last_error"] = str(exc)
            raise RuntimeError(f"qwen_device_unavailable:{exc}") from exc
        use_auto_map = device_pref == "auto" and str(device).startswith("cuda") and torch_module.cuda.is_available()
        if use_auto_map:
            load_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        else:
            dtype = torch_module.float16 if str(device).startswith(("cuda", "mps")) else torch_module.float32
            load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}

        def _load_candidate(candidate_id: str) -> tuple:
            local_only = hf_offline_enabled_fn()
            model_local = load_qwen_model_fn(str(candidate_id), load_kwargs, local_files_only=local_only)
            if not load_kwargs.get("device_map"):
                model_local.to(device)
            model_local.eval()
            processor_local = qwen_processor_cls.from_pretrained(
                str(candidate_id),
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                local_files_only=local_only,
            )
            return model_local, processor_local

        def _load_with_online_retry(candidate_id: str) -> tuple:
            try:
                return _load_candidate(candidate_id)
            except Exception as exc:  # noqa: BLE001
                if hf_offline_enabled_fn():
                    logger.warning("[qwen] offline load failed; retrying with HF online: %s", exc)
                    set_hf_offline_fn(False)
                    try:
                        return _load_candidate(candidate_id)
                    finally:
                        enable_hf_offline_defaults_fn()
                raise

        try:
            model, processor = _load_with_online_retry(str(model_id_override))
        except Exception as exc:  # noqa: BLE001
            fallback_id = strip_model_suffix_fn(str(model_id_override))
            if fallback_id:
                try:
                    logger.warning("Qwen model %s not found; falling back to %s", model_id_override, fallback_id)
                    model, processor = _load_with_online_retry(str(fallback_id))
                    state["qwen_caption_cache"][cache_key] = (model, processor)
                    state["qwen_caption_order"].append(cache_key)
                except Exception as fallback_exc:  # noqa: BLE001
                    state["qwen_last_error"] = str(fallback_exc)
                    detail = format_load_error_fn(fallback_exc)
                    raise RuntimeError(f"qwen_load_failed:{detail}") from fallback_exc
            else:
                state["qwen_last_error"] = str(exc)
                detail = format_load_error_fn(exc)
                raise RuntimeError(f"qwen_load_failed:{detail}") from exc
        state["qwen_device"] = device
        state["qwen_last_error"] = None
        if cache_limit:
            state["qwen_caption_cache"][cache_key] = (model, processor)
            state["qwen_caption_order"].append(cache_key)
            while len(state["qwen_caption_order"]) > cache_limit:
                evict_key = state["qwen_caption_order"].popleft()
                evict_model = state["qwen_caption_cache"].pop(evict_key, None)
                evict_entry_fn(evict_key, evict_model)
        if torch_module.cuda.is_available():
            try:
                torch_module.cuda.empty_cache()
            except Exception:
                pass
        enable_hf_offline_defaults_fn()
        return model, processor


def _evict_qwen_caption_entry_impl(
    cache_key: str,
    cache_entry: Optional[tuple],
    *,
    torch_module: Any,
    gc_module: Any,
) -> None:
    if not cache_entry:
        return
    try:
        model, processor = cache_entry
        try:
            del model
        except Exception:
            pass
        try:
            del processor
        except Exception:
            pass
    except Exception:
        pass
    if torch_module.cuda.is_available():
        try:
            torch_module.cuda.empty_cache()
            torch_module.cuda.ipc_collect()
        except Exception:
            pass
    try:
        gc_module.collect()
    except Exception:
        pass
