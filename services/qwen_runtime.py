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
