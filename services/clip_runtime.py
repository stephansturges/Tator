"""CLIP runtime helpers."""

from __future__ import annotations

from typing import Any, Dict


def _suspend_clip_backbone_impl(
    *,
    state: Dict[str, Any],
    lock: Any,
    agent_backbones: Dict[Any, Any],
    agent_locks: Dict[Any, Any],
    logger: Any,
) -> None:
    with lock:
        if state.get("clip_model") is None:
            return
        if str(state.get("active_encoder_type") or "clip").strip().lower() == "clip":
            logger.info("Suspending CLIP backbone to free GPU memory for training.")
        else:
            logger.debug("Suspending CLIP backbone (inactive classifier) to free GPU memory for training.")
        state["clip_model"] = None
        state["clip_preprocess"] = None
        state["clip_initialized"] = False
        state["_clip_reload_needed"] = True
    try:
        agent_backbones.clear()
        agent_locks.clear()
    except Exception:
        pass


def _resume_clip_backbone_impl(
    *,
    state: Dict[str, Any],
    lock: Any,
    clip_module: Any,
    device: Any,
    default_model: str,
    clf: Any,
    logger: Any,
) -> None:
    if not state.get("_clip_reload_needed"):
        return
    with lock:
        if state.get("clip_model") is not None:
            state["_clip_reload_needed"] = False
            state["clip_initialized"] = True
            return
        clip_name = state.get("clip_model_name") or default_model
        try:
            clip_model, clip_preprocess = clip_module.load(clip_name, device=device)
            state["clip_model"] = clip_model
            state["clip_preprocess"] = clip_preprocess
            state["clip_initialized"] = bool(clf is not None and clip_model is not None)
            logger.info("Reloaded CLIP backbone %s after training.", clip_name)
        except Exception as exc:  # noqa: BLE001
            state["clip_model"] = None
            state["clip_preprocess"] = None
            state["clip_initialized"] = False
            logger.warning("Failed to reload CLIP backbone %s: %s", clip_name, exc)
        finally:
            state["_clip_reload_needed"] = False


