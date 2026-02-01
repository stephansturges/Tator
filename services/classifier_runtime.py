"""Classifier runtime loading/unloading helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict


def _resume_classifier_backbone_impl(
    *,
    state: Dict[str, Any],
    device: Any,
    dinov3_lock: Any,
    dinov3_cuda_disabled: bool,
    dinov3_resolve_device_fn: Callable[[str], str],
    load_dinov3_fn: Callable[[str, str], tuple],
    resume_clip_fn: Callable[[], None],
) -> None:
    """Reload the active encoder backbone after training, based on user-selected classifier."""
    encoder_type = str(state.get("active_encoder_type") or "clip").strip().lower()
    if encoder_type == "dinov3":
        model_name = str(state.get("active_encoder_model") or "").strip()
        if not model_name:
            state["dinov3_initialized"] = False
            return
        target_device = dinov3_resolve_device_fn(device)
        with dinov3_lock:
            if (
                state.get("dinov3_model") is not None
                and state.get("dinov3_processor") is not None
                and state.get("dinov3_model_name") == model_name
            ):
                if dinov3_cuda_disabled and not state.get("dinov3_model_device"):
                    pass
                elif state.get("dinov3_model_device") and state.get("dinov3_model_device") != target_device:
                    pass
                else:
                    state["dinov3_initialized"] = True
                    return
            model, processor = load_dinov3_fn(model_name, target_device)
            if model is None or processor is None:
                state["dinov3_model"] = None
                state["dinov3_processor"] = None
                state["dinov3_model_name"] = None
                state["dinov3_model_device"] = None
                state["dinov3_initialized"] = False
                return
            state["dinov3_model"] = model
            state["dinov3_processor"] = processor
            state["dinov3_model_name"] = model_name
            state["dinov3_model_device"] = target_device
            state["dinov3_initialized"] = True
        return
    if encoder_type != "clip":
        state["_clip_reload_needed"] = False
        return
    if state.get("active_encoder_model"):
        state["clip_model_name"] = state.get("active_encoder_model")
    resume_clip_fn()
