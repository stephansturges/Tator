from __future__ import annotations

from typing import Any, Callable


def _unload_sam3_text_runtime_impl(*, state: dict, lock: Any) -> None:
    """Release SAM3 text prompt model to free device memory."""
    with lock:
        try:
            del state["sam3_text_model"]
        except Exception:
            pass
        try:
            del state["sam3_text_processor"]
        except Exception:
            pass
        state["sam3_text_model"] = None
        state["sam3_text_processor"] = None
        state["sam3_text_device"] = None


def _unload_dinov3_backbone_impl(
    *,
    state: dict,
    lock: Any,
    agent_backbones: dict,
    agent_locks: dict,
) -> None:
    """Release DINOv3 encoder + per-device caches."""
    with lock:
        try:
            del state["dinov3_model"]
        except Exception:
            pass
        try:
            del state["dinov3_processor"]
        except Exception:
            pass
        state["dinov3_model"] = None
        state["dinov3_processor"] = None
        state["dinov3_model_name"] = None
        state["dinov3_model_device"] = None
        state["dinov3_initialized"] = False
    try:
        agent_backbones.clear()
        agent_locks.clear()
    except Exception:
        pass


def _unload_detector_inference_impl(*, state: dict) -> None:
    """Release detector inference models (YOLO/RF-DETR) to free GPU memory."""
    try:
        del state["yolo_infer_model"]
    except Exception:
        pass
    state["yolo_infer_model"] = None
    state["yolo_infer_path"] = None
    state["yolo_infer_labelmap"] = []
    state["yolo_infer_task"] = None
    try:
        del state["rfdetr_infer_model"]
    except Exception:
        pass
    state["rfdetr_infer_model"] = None
    state["rfdetr_infer_path"] = None
    state["rfdetr_infer_labelmap"] = []
    state["rfdetr_infer_task"] = None
    state["rfdetr_infer_variant"] = None


def _unload_non_qwen_runtimes_impl(
    *,
    predictor_manager: Any,
    unload_sam3_text_fn: Callable[[], None],
    suspend_clip_fn: Callable[[], None],
    unload_dinov3_fn: Callable[[], None],
    unload_detector_fn: Callable[[], None],
    torch_module: Any,
    logger: Any,
) -> None:
    """Free heavy inference runtimes except Qwen (SAM, detectors, classifier backbones)."""
    try:
        predictor_manager.unload_all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to unload SAM predictors: %s", exc)
    unload_sam3_text_fn()
    suspend_clip_fn()
    unload_dinov3_fn()
    unload_detector_fn()
    if torch_module.cuda.is_available():
        try:
            torch_module.cuda.empty_cache()
            torch_module.cuda.ipc_collect()
        except Exception:
            pass


def _unload_inference_runtimes_impl(
    *,
    unload_non_qwen_fn: Callable[[], None],
    unload_qwen_fn: Callable[[], None],
    torch_module: Any,
) -> None:
    """Free heavy inference runtimes (SAM, detectors, Qwen, classifier backbones)."""
    unload_non_qwen_fn()
    unload_qwen_fn()
    if torch_module.cuda.is_available():
        try:
            device_count = torch_module.cuda.device_count()
        except Exception:
            device_count = 0
        for device_index in range(device_count):
            try:
                torch_module.cuda.set_device(device_index)
                torch_module.cuda.empty_cache()
            except Exception:
                continue
