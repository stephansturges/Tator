"""Preserved first-pane Qwen orchestration with explicit worker dependencies.

Extracted from localinferenceapi.py on 2026-10-15. No application import, route
registration, filesystem discovery, model activation or service startup occurs
here. bind_pipeline() gives every callable a private dependency namespace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from types import FunctionType
import hashlib
import json
import logging
import math
import queue
import re
import threading
import time
import uuid
import numpy as np
import torch

logger = logging.getLogger(__name__)
from fastapi import HTTPException as HTTPException
from starlette.status import HTTP_400_BAD_REQUEST as HTTP_400_BAD_REQUEST
from utils.status_compat import HTTP_422_UNPROCESSABLE_CONTENT as HTTP_422_UNPROCESSABLE_CONTENT
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE as HTTP_503_SERVICE_UNAVAILABLE
from typing import Mapping as Mapping
from services.qwen_mlx import QWEN_MLX_VISION_MODEL_OPTIONS as QWEN_MLX_VISION_MODEL_OPTIONS
from services.qwen_mlx import QWEN_PLATFORM_MLX as QWEN_PLATFORM_MLX
from services.qwen_mlx import QWEN_PLATFORM_TRANSFORMERS as QWEN_PLATFORM_TRANSFORMERS
from services.qwen_model_catalog import QWEN_TRANSFORMERS_MODEL_OPTIONS as QWEN_TRANSFORMERS_MODEL_OPTIONS
from models.schemas import QwenCaptionHint as QwenCaptionHint
from models.schemas import QwenCaptionPromptPreviewResponse as QwenCaptionPromptPreviewResponse
from models.schemas import QwenCaptionPromptPreviewSection as QwenCaptionPromptPreviewSection
from models.schemas import QwenCaptionResponse as QwenCaptionResponse
from models.schemas import QwenDetection as QwenDetection
from models.schemas import QwenInferenceResponse as QwenInferenceResponse
from services.qwen import _adjust_prompt_for_thinking as _adjust_prompt_for_thinking_impl
from services.qwen import _allowed_caption_labels_impl as _allowed_caption_labels_impl
from services.qwen import _build_qwen_caption_prompt as _build_qwen_caption_prompt_impl
from services.qwen import _caption_count_conflicts as _caption_count_conflicts_impl
from services.qwen import _caption_degenerate_reason as _caption_degenerate_reason_impl
from services.qwen import _caption_demote_unstable_glossary_subtypes as _caption_demote_unstable_glossary_subtypes_impl
from services.qwen import _caption_editor_preserve_broad_terms_instruction as _caption_editor_preserve_broad_terms_instruction_impl
from services.qwen import _caption_glossary_map as _caption_glossary_map_impl
from services.qwen import _caption_has_meta as _caption_has_meta_impl
from services.qwen import _caption_is_degenerate_impl as _caption_is_degenerate_impl
from services.qwen import _caption_length_instruction as _caption_length_instruction_impl
from services.qwen import _caption_missing_exact_counts as _caption_missing_exact_counts_impl
from services.qwen import _caption_needs_completion as _caption_needs_completion_impl
from services.qwen import _caption_needs_english_rewrite as _caption_needs_english_rewrite_impl
from services.qwen import _caption_needs_refine as _caption_needs_refine_impl
from services.qwen import _caption_needs_short_form as _caption_needs_short_form_impl
from services.qwen import _caption_preferred_label as _caption_preferred_label_impl
from services.qwen import _caption_repair_count_text_artifacts as _caption_repair_count_text_artifacts_impl
from services.qwen import _caption_repetition_loop_detected as _caption_repetition_loop_detected_impl
from services.qwen import _caption_trim_to_complete_sentences as _caption_trim_to_complete_sentences_impl
from services.qwen import _caption_user_request_instruction as _caption_user_request_instruction_impl
from services.qwen import _caption_user_requested_short as _caption_user_requested_short_impl
from services.qwen import _clean_caption_source_context_text as _clean_caption_source_context_text_impl
from services.qwen import _collapse_whitespace as _collapse_whitespace_impl
from services.qwen import _extract_caption_from_text as _extract_caption_from_text_impl
from utils.coords import _extract_numeric_sequence as _extract_numeric_sequence
from services.qwen import _extract_qwen_json_block_impl as _extract_qwen_json_block_impl
from services.qwen import _format_caption_source_output_context as _format_caption_source_output_context_impl
from services.qwen import _format_caption_window_observation_lines as _format_caption_window_observation_lines_impl
from services.qwen import _get_qwen_prompt_config_impl as _get_qwen_prompt_config_impl
from utils.coords import _mask_to_bounding_box as _mask_to_bounding_box
from services.qwen_generation import _qwen_append_logits_processor as _qwen_append_logits_processor
from services.qwen_generation import _qwen_build_immediate_action_processor as _qwen_build_immediate_action_processor
from services.qwen_generation import _qwen_build_thinking_effort_processor as _qwen_build_thinking_effort_processor
from services.qwen_generation import _qwen_effective_input_len as _qwen_effective_input_len
from services.qwen_generation import _qwen_supports_presence_penalty as _qwen_supports_presence_penalty
from services.qwen import _render_qwen_prompt_impl as _render_qwen_prompt_impl
from services.qwen import _resolve_caption_all_windows as _resolve_caption_all_windows_impl
from services.qwen import _resolve_qwen_caption_decode as _resolve_qwen_caption_decode_impl
from services.qwen_runtime import _resolve_qwen_device_impl as _resolve_qwen_device_impl
from services.qwen_generation import _resolve_qwen_max_seq_len as _resolve_qwen_max_seq_len
from services.qwen import _resolve_qwen_variant_model_id_impl as _resolve_qwen_variant_model_id_impl
from services.qwen import _resolve_qwen_window_overlap as _resolve_qwen_window_overlap_impl
from services.qwen import _resolve_qwen_window_size as _resolve_qwen_window_size_impl
from services.qwen import _run_qwen_caption_cleanup as _run_qwen_caption_cleanup_impl
from services.qwen import _run_qwen_caption_merge as _run_qwen_caption_merge_impl
from services.qwen import _sanitize_qwen_caption as _sanitize_qwen_caption_impl
from utils.coords import _scale_bbox_to_image as _scale_bbox_to_image
from utils.coords import _scale_point_to_image as _scale_point_to_image
from services.qwen import _thinking_caption_needs_cleanup as _thinking_caption_needs_cleanup_impl
from services.qwen import _truncate_repeated_caption_loop as _truncate_repeated_caption_loop_impl
from services.qwen import _window_positions_impl as _window_positions_impl
from utils.coords import _xyxy_to_yolo_norm_list as _xyxy_to_yolo_norm_list
from services.qwen_model_catalog import is_qwen_mlx_model_id as is_qwen_mlx_model_id
from models.schemas import QwenCaptionRequest, QwenInferenceRequest

QWEN_CAPTION_CACHE_LIMIT = 1
QWEN_CAPTION_INSTRUCT_MERGE_AUTO_MAX_NEW_TOKENS = 1500
QWEN_CAPTION_INSTRUCT_WINDOW_AUTO_MAX_NEW_TOKENS = 1200
QWEN_CAPTION_LOOP_COOLDOWN_SECONDS = 3600
QWEN_CAPTION_MAX_NEW_TOKENS = 4096
QWEN_CAPTION_PROMPT_AWARE_MIN_NEW_TOKENS = 512
QWEN_CAPTION_PROMPT_OUTPUT_TOKEN_TARGET = 6000
QWEN_CAPTION_THINKING_PROMPT_OUTPUT_TOKEN_TARGET = 12000
QWEN_DO_SAMPLE = False
QWEN_MAX_NEW_TOKENS = 2000
QWEN_TEMPERATURE = 0.2
QWEN_TOP_P = 0.9
QWEN_WINDOW_DEFAULT_OVERLAP = 0.2
QWEN_WINDOW_DEFAULT_SIZE = 672
QWEN_MLX_CAPTION_MODEL_NAME = 'mlx-community/Qwen3-VL-4B-Instruct-4bit'


def _emit_caption_window(x0: int, y0: int, size: int, caption: str) -> None:
    hook = _CAPTION_WINDOW_HOOK.get()
    if hook is None:
        return
    try:
        hook(int(x0), int(y0), int(size), str(caption))
    except Exception:
        return


@dataclass
class QwenRuntime:
    model: Any
    processor: Any
    platform: str = QWEN_PLATFORM_TRANSFORMERS
    model_id: Optional[str] = None
    config: Optional[Any] = None

    def __iter__(self):
        yield self.model
        yield self.processor


class QwenCancellationRequested(RuntimeError):
    """Raised when the active Qwen request has been cancelled."""


class QwenCaptionLoopDetected(RuntimeError):
    """Raised when caption generation enters a repeated output loop."""

    def __init__(
        self,
        message: str = "qwen_caption_repetition_loop",
        *,
        raw_output: Any = None,
        trimmed_output: Any = None,
        degenerate_reason: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.raw_output = str(raw_output or "")
        self.trimmed_output = str(trimmed_output or "")
        self.degenerate_reason = str(degenerate_reason or "").strip() or None
        self.generated_chars = len(self.raw_output)

    def diagnostic_payload(self, *, max_chars: int = 4000) -> Dict[str, Any]:
        raw = self.raw_output
        trimmed = self.trimmed_output
        max_chars = max(200, int(max_chars or 4000))
        return {
            "type": type(self).__name__,
            "message": str(self),
            "degenerate_reason": self.degenerate_reason,
            "generated_chars": self.generated_chars,
            "raw_output_head": raw[:max_chars],
            "raw_output_tail": raw[-max_chars:] if raw else "",
            "raw_output_truncated": len(raw) > max_chars,
            "trimmed_output": trimmed[:max_chars],
            "trimmed_output_truncated": len(trimmed) > max_chars,
        }


def _qwen_caption_loop_output_salvageable(
    raw_output: Any,
    trimmed_output: Any = "",
    *,
    degenerate_reason: Optional[str] = None,
) -> bool:
    """Return whether a looped caption stream contains enough language to salvage."""

    text = str(trimmed_output or raw_output or "").strip()
    raw_text = str(raw_output or "").strip()
    if not text and not raw_text:
        return False
    combined = f"{text}\n{raw_text}".strip()
    compact = re.sub(r"\s+", "", combined)
    if not compact:
        return False
    if re.fullmatch(r"[^A-Za-z0-9]+", compact):
        return False
    reason = str(degenerate_reason or "").strip().lower()
    alpha_words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", combined)
    if reason in {"punctuation_loop", "alnum_char_loop"} and len(alpha_words) < 12:
        return False
    if len(alpha_words) < 12:
        return False
    unique_word_ratio = len({word.lower() for word in alpha_words}) / max(1, len(alpha_words))
    if len(alpha_words) >= 24 and unique_word_ratio < 0.18:
        return False
    sentence_like = re.findall(r"[A-Z][^.!?]{20,}[.!?]", combined)
    caption_markers = re.search(
        r"\b(final(?:\s+caption| polish)?|revised caption|caption)\s*:",
        combined,
        re.IGNORECASE,
    )
    return bool(sentence_like or caption_markers)


def _qwen_caption_compact_loop_salvage_text(
    raw_output: Any,
    trimmed_output: Any = "",
    *,
    max_chars: int = 6000,
) -> str:
    """Build a bounded raw-output excerpt for an LLM salvage pass."""

    pieces: List[Tuple[str, str]] = []
    trimmed = str(trimmed_output or "").strip()
    raw = str(raw_output or "").strip()
    if trimmed:
        pieces.append(("trimmed loop output", trimmed))
    if raw and raw != trimmed:
        pieces.append(("raw loop output", raw))
    if not pieces:
        return ""
    budget = max(1000, int(max_chars or 6000))
    per_piece = max(500, budget // len(pieces))
    sections: List[str] = []
    for label, text in pieces:
        if len(text) > per_piece:
            half = max(200, per_piece // 2)
            text = (
                f"{text[:half]}\n"
                f"... omitted {len(text) - (half * 2)} chars from middle ...\n"
                f"{text[-half:]}"
            )
        sections.append(f"{label}:\n{text}")
    return "\n\n".join(sections)


def _qwen_caption_compact_original_prompt_context(prompt: Any, *, max_chars: int = 4000) -> str:
    text = str(prompt or "").strip()
    if not text:
        return ""
    limit = max(1000, int(max_chars or 4000))
    if len(text) <= limit:
        return text
    half = max(500, limit // 2)
    return (
        f"{text[:half]}\n"
        f"... omitted {len(text) - (half * 2)} chars from middle ...\n"
        f"{text[-half:]}"
    )


class QwenCaptionRecoverableGenerationError(RuntimeError):
    """Raised when caption generation fails in a way the caption ladder can recover."""


def _normalize_qwen_caption_loop_recovery_mode(value: Any) -> str:
    mode = str(value or "safe_retry_fallback").strip().lower()
    if mode not in {"off", "safe_retry", "safe_retry_fallback"}:
        return "safe_retry_fallback"
    return mode


def _qwen_caption_safe_retry_decode(
    decode_override: Optional[Mapping[str, Any]],
    *,
    is_thinking: bool,
) -> Dict[str, Any]:
    """Return a lower-risk decode override after a caption loop is detected."""

    base = dict(decode_override or {})

    def _safe_float(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return parsed if math.isfinite(parsed) else default

    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    if is_thinking:
        safe = {
            "do_sample": True,
            "temperature": 0.35,
            "top_p": 0.75,
            "top_k": 20,
            "presence_penalty": 0.0,
        }
    else:
        safe = {"do_sample": False}
    safe["repetition_penalty"] = max(1.15, _safe_float(base.get("repetition_penalty"), 1.0))
    safe["repetition_context_size"] = max(256, _safe_int(base.get("repetition_context_size"), 128))
    safe["no_repeat_ngram_size"] = max(8, _safe_int(base.get("no_repeat_ngram_size"), 0))
    return safe


def _qwen_caption_safe_retry_tokens(max_new_tokens: Optional[int]) -> int:
    try:
        requested = int(max_new_tokens) if max_new_tokens is not None else 1000
    except (TypeError, ValueError, OverflowError):
        requested = 1000
    requested = max(256, min(requested, QWEN_CAPTION_MAX_NEW_TOKENS))
    return max(768, min(requested, 2200))


def _qwen_caption_recovery_key(
    *,
    stage_id: str,
    stage_label: str,
    call_kind: str,
    model_id: str,
    prompt: str,
    system_prompt: Optional[str],
    decode_override: Optional[Mapping[str, Any]],
) -> str:
    prompt_hash = hashlib.sha256(
        f"{system_prompt or ''}\n{prompt or ''}".encode("utf-8", "ignore")
    ).hexdigest()[:16]
    decode_hash = hashlib.sha256(
        json.dumps(dict(decode_override or {}), sort_keys=True, default=str).encode(
            "utf-8", "ignore"
        )
    ).hexdigest()[:12]
    raw = json.dumps(
        {
            "stage_id": stage_id,
            "stage_label": stage_label,
            "kind": call_kind,
            "model_id": model_id,
            "prompt": prompt_hash,
            "decode": decode_hash,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()


def _qwen_caption_step_entry(step_id: str, label: str, detail: str = "") -> Dict[str, str]:
    entry = {"id": step_id, "label": label}
    if detail:
        entry["detail"] = detail
    return entry


def _build_qwen_caption_step_plan(
    *,
    caption_mode: str,
    total_windows: int,
    two_stage: bool,
    is_thinking: bool,
    force_unload: bool,
    caption_model_id: str,
    refinement_model_id: str,
    windowed_full_image_strategy: str = "visual",
) -> List[Dict[str, str]]:
    plan: List[Dict[str, str]] = [
        _qwen_caption_step_entry("prepare", "Prepare image and prompts"),
        _qwen_caption_step_entry(
            "load_model",
            "Load caption model",
            (
                f"Caption model: {caption_model_id}; refinement model: {refinement_model_id}"
                if refinement_model_id and refinement_model_id != caption_model_id
                else f"Caption model: {caption_model_id}"
            ),
        ),
    ]
    if caption_mode == "windowed":
        for window_index in range(1, max(1, int(total_windows or 1)) + 1):
            plan.append(
                _qwen_caption_step_entry(
                    f"window_{window_index}",
                    f"Window observation {window_index}/{max(1, int(total_windows or 1))}",
                    (
                        "Crop caption used as source material before final composition; "
                        "Draft + refine does not run per crop."
                    ),
                )
            )
    if two_stage and is_thinking:
        plan.extend(
            [
                _qwen_caption_step_entry(
                    "draft_caption",
                    "Two-stage 1/2: draft full image",
                    "Thinking-only full-image draft pass after any window observations are collected.",
                ),
                _qwen_caption_step_entry(
                    "refine_draft",
                    "Two-stage 2/2: refine draft",
                    "Editor pass over the draft, full prompt context, and raw first-stage outputs. Separate from window merge.",
                ),
            ]
        )
    else:
        full_compose_detail = "Uses the full image, generated detection context, and any window observations already collected."
        if caption_mode == "windowed" and windowed_full_image_strategy == "text_only":
            full_compose_detail = (
                "Uses completed window observations, authoritative counts, and prompt context without "
                "sending a second full-image tensor."
            )
        plan.append(
            _qwen_caption_step_entry(
                "generate_full",
                "Compose full-image caption",
                full_compose_detail,
            )
        )
    if caption_mode == "windowed" and not (
        windowed_full_image_strategy == "text_only" and not (two_stage and is_thinking)
    ):
        plan.append(
            _qwen_caption_step_entry(
                "merge_windows",
                "Window merge: fold in sub-captions",
                "Windowed-only editor pass that runs after full-image composition when crop captions exist.",
            )
        )
    plan.append(
        _qwen_caption_step_entry(
            "postprocess",
            "Quality guard checks",
            "Cleanup, coverage/count guards, and language checks run only if needed.",
        )
    )
    if force_unload:
        plan.append(_qwen_caption_step_entry("unload_model", "Unload caption model"))
    plan.append(_qwen_caption_step_entry("finalize", "Finalize caption"))
    return plan


def _qwen_maybe_trim_repetition_loop_output(
    output_text: str,
    *,
    triggered: bool = False,
    replace_live_output: bool = True,
) -> str:
    if not output_text:
        return output_text
    if not triggered and not _caption_repetition_loop_detected_impl(output_text):
        return output_text
    cleaned = _truncate_repeated_caption_loop_impl(output_text)
    if not cleaned:
        cleaned = output_text
    _qwen_caption_io_record(
        {
            "event": "loop_trim",
            "call_id": qwen_llm_call_id_var.get(),
            "raw_output": output_text,
            "trimmed_output": cleaned,
            "loop_detected": True,
        }
    )
    update_kwargs: Dict[str, Any] = {
        "phase": "cleanup",
        "phase_label": "Stopping Loop",
        "progress": 0.90,
        "message": "Stopped a repeated caption loop; cleaning generated text",
        "token_preview": cleaned,
    }
    if replace_live_output:
        update_kwargs["live_output"] = cleaned
    _qwen_progress_update(**update_kwargs)
    logger.warning("[qwen] stopped repeated caption loop after %s generated chars", len(output_text))
    return cleaned


class _QwenCaptionStreamLoopGuard:
    """Incremental repeated-output guard for live caption token streams."""

    def __init__(
        self,
        *,
        min_chars: int = 32,
        check_chars: int = 24,
        max_tail_chars: int = 4096,
    ) -> None:
        self.min_chars = max(1, int(min_chars))
        self.check_chars = max(1, int(check_chars))
        self.max_tail_chars = max(self.min_chars, int(max_tail_chars))
        self._parts: List[str] = []
        self._tail = ""
        self._total_chars = 0
        self._chars_since_check = 0
        self.triggered = False
        self.reason: Optional[str] = None

    def append(self, piece: str) -> bool:
        if self.triggered:
            return True
        text = str(piece or "")
        if not text:
            return False
        self._parts.append(text)
        self._total_chars += len(text)
        self._chars_since_check += len(text)
        self._tail = (self._tail + text)[-self.max_tail_chars :]
        if self._total_chars < self.min_chars or self._chars_since_check < self.check_chars:
            return False
        self._chars_since_check = 0
        if _caption_repetition_loop_detected_impl(self._tail):
            self.triggered = True
            self.reason = (
                _caption_degenerate_reason_impl(self._tail, allow_short_caption=True)
                or "repetition_loop"
            )
        return self.triggered

    def text(self) -> str:
        return "".join(self._parts)


def _qwen_caption_stream_loop_detected_event(
    *,
    source: str,
    model_id: str,
    runtime_platform: Optional[str],
    guard: _QwenCaptionStreamLoopGuard,
) -> None:
    raw_output = guard.text()
    _qwen_caption_io_record(
        {
            "event": "stream_loop_detected",
            "call_id": qwen_llm_call_id_var.get(),
            "source": source,
            "runtime_platform": runtime_platform,
            "model_id": model_id,
            "loop_detected": True,
            "degenerate_reason": guard.reason,
            "generated_chars": len(raw_output),
            "raw_output_tail": raw_output[-4000:] if raw_output else "",
        }
    )
    _qwen_progress_update(
        phase="generate",
        phase_label="Stopping Loop",
        progress=0.88,
        message="Detected repeated caption text during generation",
        token_preview=raw_output,
    )


def _qwen_caption_generation_active() -> bool:
    with qwen_progress_lock:
        return qwen_progress_state.get("kind") == "caption"


def _http_exception_detail_text(exc: HTTPException) -> str:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, str):
        return detail
    if detail is None:
        return ""
    try:
        return json.dumps(detail, ensure_ascii=False)
    except Exception:
        return str(detail)


def _extract_qwen_json_block(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    snippet, detections = _extract_qwen_json_block_impl(text)
    if not detections:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail="qwen_parse_error:no_json_block_found",
        )
    return snippet, detections


def _qwen_items_from_payload(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in items if isinstance(item, dict)]


def _qwen_bbox_results(
    items: List[Dict[str, Any]],
    proc_w: int,
    proc_h: int,
    full_w: int,
    full_h: int,
    *,
    limit: int,
) -> List['QwenDetection']:
    results: List[QwenDetection] = []
    for item in items:
        bbox = (
            _extract_numeric_sequence(item.get("bbox_2d"), length=4)
            or _extract_numeric_sequence(item.get("bbox"), length=4)
            or _extract_numeric_sequence(item.get("box"), length=4)
        )
        if not bbox:
            continue
        scaled = _scale_bbox_to_image(bbox, proc_w, proc_h, full_w, full_h)
        if not scaled:
            continue
        left, top, right, bottom = scaled
        yolo_box = _xyxy_to_yolo_norm_list(full_w, full_h, left, top, right, bottom)
        label = item.get("label") or item.get("class") or item.get("name")
        results.append(
            QwenDetection(bbox=yolo_box, qwen_label=str(label) if label else None, source="bbox")
        )
        if len(results) >= limit:
            break
    return results


def _qwen_bbox_sam_results(
    items: List[Dict[str, Any]],
    proc_w: int,
    proc_h: int,
    pil_img: Image.Image,
    np_img: np.ndarray,
    token: Optional[str],
    variant: str,
    *,
    image_name: Optional[str],
    limit: int,
) -> List['QwenDetection']:
    results: List[QwenDetection] = []
    for item in items:
        bbox = (
            _extract_numeric_sequence(item.get("bbox_2d"), length=4)
            or _extract_numeric_sequence(item.get("bbox"), length=4)
            or _extract_numeric_sequence(item.get("box"), length=4)
        )
        if not bbox:
            continue
        scaled = _scale_bbox_to_image(bbox, proc_w, proc_h, pil_img.width, pil_img.height)
        if not scaled:
            continue
        sub_box = np.array(list(scaled), dtype=np.float32)
        masks, _, _ = _predict_with_cache(
            np_img,
            token,
            variant,
            image_name=image_name,
            box=sub_box,
            multimask_output=False,
        )
        mask = masks[0]
        left, top, right, bottom = _mask_to_bounding_box(mask)
        if right <= left or bottom <= top:
            continue
        yolo_box = _xyxy_to_yolo_norm_list(pil_img.width, pil_img.height, left, top, right, bottom)
        label = item.get("label") or item.get("class") or item.get("name")
        results.append(
            QwenDetection(
                bbox=yolo_box, qwen_label=str(label) if label else None, source="bbox_sam"
            )
        )
        if len(results) >= limit:
            break
    return results


def _qwen_point_results(
    items: List[Dict[str, Any]],
    proc_w: int,
    proc_h: int,
    pil_img: Image.Image,
    np_img: np.ndarray,
    token: Optional[str],
    variant: str,
    *,
    image_name: Optional[str],
    limit: int,
) -> List['QwenDetection']:
    results: List[QwenDetection] = []
    for item in items:
        point = _extract_numeric_sequence(item.get("point_2d") or item.get("point"), length=2)
        if not point:
            continue
        scaled_point = _scale_point_to_image(point, proc_w, proc_h, pil_img.width, pil_img.height)
        if not scaled_point:
            continue
        coords = np.array([[scaled_point[0], scaled_point[1]]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)
        masks, _, _ = _predict_with_cache(
            np_img,
            token,
            variant,
            image_name=image_name,
            point_coords=coords,
            point_labels=labels,
            multimask_output=False,
        )
        mask = masks[0]
        left, top, right, bottom = _mask_to_bounding_box(mask)
        if right <= left or bottom <= top:
            continue
        yolo_box = _xyxy_to_yolo_norm_list(pil_img.width, pil_img.height, left, top, right, bottom)
        label = item.get("label") or item.get("class") or item.get("name")
        results.append(
            QwenDetection(bbox=yolo_box, qwen_label=str(label) if label else None, source="point")
        )
        if len(results) >= limit:
            break
    return results


def _qwen_runtime_platform_from_runtime(runtime: Any) -> str:
    if isinstance(runtime, QwenRuntime):
        return runtime.platform
    return QWEN_PLATFORM_TRANSFORMERS


def _qwen_runtime_model_id(runtime: Any) -> Optional[str]:
    if isinstance(runtime, QwenRuntime):
        return runtime.model_id
    return None


def _group_hints_by_window(
    label_hints: Sequence[QwenCaptionHint],
    x_positions: Sequence[int],
    y_positions: Sequence[int],
    window: int,
) -> Dict[Tuple[int, int], List[QwenCaptionHint]]:
    grouped: Dict[Tuple[int, int], List[QwenCaptionHint]] = {
        (x0, y0): [] for y0 in y_positions for x0 in x_positions
    }
    for hint in label_hints:
        if not hint.bbox or len(hint.bbox) != 4:
            continue
        try:
            bx1, by1, bx2, by2 = [float(value) for value in hint.bbox[:4]]
        except (TypeError, ValueError):
            continue
        if bx2 <= bx1 or by2 <= by1:
            continue
        for x0 in x_positions:
            win_x1 = float(x0)
            win_x2 = win_x1 + float(window)
            ix1 = max(bx1, win_x1)
            ix2 = min(bx2, win_x2)
            if ix2 <= ix1:
                continue
            for y0 in y_positions:
                win_y1 = float(y0)
                win_y2 = win_y1 + float(window)
                iy1 = max(by1, win_y1)
                iy2 = min(by2, win_y2)
                if iy2 <= iy1:
                    continue
                grouped[(x0, y0)].append(
                    QwenCaptionHint(
                        label=hint.label,
                        bbox=[
                            ix1 - win_x1,
                            iy1 - win_y1,
                            ix2 - win_x1,
                            iy2 - win_y1,
                        ],
                        confidence=hint.confidence,
                        source_id=getattr(hint, "source_id", None),
                    )
                )
    return grouped


def _normalize_caption_hints_for_decoded_image(
    label_hints: Sequence[QwenCaptionHint],
    source_width: Optional[int],
    source_height: Optional[int],
    decoded_width: int,
    decoded_height: int,
) -> List[QwenCaptionHint]:
    try:
        safe_decoded_width = max(1.0, float(decoded_width))
        safe_decoded_height = max(1.0, float(decoded_height))
    except (TypeError, ValueError, OverflowError):
        safe_decoded_width = safe_decoded_height = 1.0
    try:
        safe_source_width = float(source_width) if source_width is not None else safe_decoded_width
        safe_source_height = float(source_height) if source_height is not None else safe_decoded_height
    except (TypeError, ValueError, OverflowError):
        safe_source_width = safe_decoded_width
        safe_source_height = safe_decoded_height
    if not math.isfinite(safe_source_width) or safe_source_width <= 0:
        safe_source_width = safe_decoded_width
    if not math.isfinite(safe_source_height) or safe_source_height <= 0:
        safe_source_height = safe_decoded_height
    scale_x = safe_decoded_width / safe_source_width
    scale_y = safe_decoded_height / safe_source_height
    normalized: List[QwenCaptionHint] = []
    for hint in label_hints or []:
        label = str(getattr(hint, "label", "") or "").strip()
        if not label:
            continue
        bbox = getattr(hint, "bbox", None)
        normalized_bbox: Optional[List[float]] = None
        if bbox and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError, OverflowError):
                x1 = y1 = x2 = y2 = 0.0
            else:
                if all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y
                    x1 = max(0.0, min(safe_decoded_width, x1))
                    x2 = max(0.0, min(safe_decoded_width, x2))
                    y1 = max(0.0, min(safe_decoded_height, y1))
                    y2 = max(0.0, min(safe_decoded_height, y2))
                    if x2 > x1 and y2 > y1:
                        normalized_bbox = [x1, y1, x2, y2]
        normalized.append(
            QwenCaptionHint(
                label=label,
                bbox=normalized_bbox,
                confidence=getattr(hint, "confidence", None),
                source_id=getattr(hint, "source_id", None),
            )
        )
    return normalized


def _caption_position_phrase_for_bbox(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
) -> str:
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return ""
    safe_width = max(1.0, float(image_width or 1))
    safe_height = max(1.0, float(image_height or 1))
    cx = max(0.0, min(safe_width, (x1 + x2) * 0.5))
    cy = max(0.0, min(safe_height, (y1 + y2) * 0.5))
    horiz = "left" if cx < safe_width / 3.0 else "right" if cx > safe_width * 2.0 / 3.0 else "center"
    vert = "upper" if cy < safe_height / 3.0 else "lower" if cy > safe_height * 2.0 / 3.0 else "middle"
    if horiz == "center" and vert == "middle":
        return "near the center"
    if horiz == "center":
        return f"in the {vert} center"
    if vert == "middle":
        return f"near the {horiz} side"
    return f"in the {vert}-{horiz}"


def _build_caption_overlap_guidance(
    label_hints: Sequence[QwenCaptionHint],
    grouped_hints: Mapping[Tuple[int, int], Sequence[QwenCaptionHint]],
    image_width: int,
    image_height: int,
    *,
    labelmap_glossary: Optional[str] = None,
    max_examples: int = 12,
) -> str:
    originals: Dict[str, QwenCaptionHint] = {}
    for hint in label_hints or []:
        source_id = str(getattr(hint, "source_id", None) or "").strip()
        if source_id and source_id not in originals:
            originals[source_id] = hint
    if not originals:
        return ""
    memberships: Dict[str, int] = {}
    for hints in grouped_hints.values():
        for hint in hints or []:
            source_id = str(getattr(hint, "source_id", None) or "").strip()
            if source_id:
                memberships[source_id] = memberships.get(source_id, 0) + 1
    glossary_map = _caption_glossary_map_impl(
        labelmap_glossary,
        [hint.label for hint in label_hints if getattr(hint, "label", None)],
    )
    examples: List[str] = []
    for source_id, count in memberships.items():
        if count <= 1:
            continue
        hint = originals.get(source_id)
        if not hint:
            continue
        label = _caption_preferred_label_impl(hint.label, glossary_map)
        position = _caption_position_phrase_for_bbox(
            hint.bbox or [],
            image_width,
            image_height,
        )
        examples.append(f"{label} {position}".strip())
        if len(examples) >= max_examples:
            break
    if not examples:
        return ""
    examples_text = "; ".join(examples)
    if len(examples) < len([sid for sid, count in memberships.items() if count > 1]):
        examples_text = f"{examples_text}; and other repeated crop-edge objects"
    return (
        "Overlap deduplication guidance: detailed/windowed captioning uses overlapping crops. "
        "Some observations therefore describe the same physical object more than once. "
        "Merge repeated crop-edge descriptions into one object in the final caption. "
        "Do not mention crop overlap, labels, hints, boxes, or this guidance. "
        f"Shared objects to describe once if visible: {examples_text}."
    )


def _qwen_model_id_is_thinking(model_id: Optional[str]) -> bool:
    return "thinking" in str(model_id or "").lower()


def _resolve_qwen_caption_max_new_tokens(
    requested: Optional[Any],
    *,
    model_id: Optional[str],
    variant: Optional[str],
    caption_mode: Optional[str],
    final_caption_max_sentences: Optional[int],
    many_windowed_steps: bool = False,
    label_hint_count: int = 0,
) -> int:
    """Resolve a caption output-token budget.

    A numeric request is an explicit user cap. None means Auto, where Thinking
    models and large/windowed prompt stacks get a larger default budget.
    """

    if requested is not None:
        try:
            parsed = int(requested)
        except (TypeError, ValueError, OverflowError):
            parsed = 0
        if parsed > 0:
            return max(32, min(parsed, QWEN_CAPTION_MAX_NEW_TOKENS))

    try:
        sentence_count = int(final_caption_max_sentences or 10)
    except (TypeError, ValueError, OverflowError):
        sentence_count = 10
    sentence_budget = max(1000, min(QWEN_CAPTION_MAX_NEW_TOKENS, max(1, sentence_count) * 100))
    is_thinking = _qwen_model_id_is_thinking(model_id) or str(variant or "").lower() == "thinking"
    mode = str(caption_mode or "full").strip().lower()
    if is_thinking:
        return QWEN_CAPTION_MAX_NEW_TOKENS
    if mode == "windowed" or many_windowed_steps:
        try:
            window_budget = int(QWEN_CAPTION_INSTRUCT_WINDOW_AUTO_MAX_NEW_TOKENS or 0)
        except (TypeError, ValueError, OverflowError):
            window_budget = 0
        return min(QWEN_CAPTION_MAX_NEW_TOKENS, max(sentence_budget, window_budget))
    return sentence_budget


def _estimate_qwen_prompt_tokens(
    rendered_prompt: str,
    *,
    tokenizer: Optional[Any] = None,
) -> int:
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                rendered_prompt,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
            if isinstance(input_ids, list):
                if input_ids and isinstance(input_ids[0], list):
                    return max(1, len(input_ids[0]))
                return max(1, len(input_ids))
        except Exception:
            pass
    return max(1, int(math.ceil(len(str(rendered_prompt or "")) / 4.0)))


def _caption_explicit_token_budget_active() -> bool:
    if not _qwen_caption_generation_active():
        return False
    with qwen_progress_lock:
        return bool(qwen_progress_state.get("caption_explicit_max_new_tokens"))


def _qwen_caption_prompt_aware_max_new_tokens(
    requested: Optional[Any],
    *,
    prompt_tokens: int,
    model_id: Optional[str],
    explicit: Optional[bool] = None,
) -> int:
    try:
        requested_max = int(requested) if requested is not None else QWEN_MAX_NEW_TOKENS
    except (TypeError, ValueError, OverflowError):
        requested_max = QWEN_MAX_NEW_TOKENS
    requested_max = max(1, min(requested_max, QWEN_CAPTION_MAX_NEW_TOKENS))
    explicit_budget = (
        _caption_explicit_token_budget_active() if explicit is None else bool(explicit)
    )
    if explicit_budget:
        return requested_max
    try:
        prompt_count = max(0, int(prompt_tokens or 0))
    except (TypeError, ValueError, OverflowError):
        prompt_count = 0
    is_thinking = _qwen_model_id_is_thinking(model_id)
    target = (
        QWEN_CAPTION_THINKING_PROMPT_OUTPUT_TOKEN_TARGET
        if is_thinking
        else QWEN_CAPTION_PROMPT_OUTPUT_TOKEN_TARGET
    )
    try:
        target = max(1, int(target or 0))
    except (TypeError, ValueError, OverflowError):
        target = 1
    min_new = (
        max(1024, int(QWEN_CAPTION_PROMPT_AWARE_MIN_NEW_TOKENS or 0))
        if is_thinking
        else max(256, int(QWEN_CAPTION_PROMPT_AWARE_MIN_NEW_TOKENS or 0))
    )
    if prompt_count + requested_max <= target:
        return requested_max
    adapted = target - prompt_count
    return min(requested_max, max(min_new, adapted))


def _qwen_disable_thinking_requested(chat_template_kwargs: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(chat_template_kwargs, dict):
        return False
    return chat_template_kwargs.get("enable_thinking") is False


def _qwen_messages_with_no_think(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    last_user_idx: Optional[int] = None
    for idx, message in enumerate(messages):
        copied_message = dict(message)
        role = str(copied_message.get("role") or "").lower()
        content = copied_message.get("content")
        if isinstance(content, list):
            copied_message["content"] = [
                dict(part) if isinstance(part, dict) else part for part in content
            ]
        copied.append(copied_message)
        if role == "user":
            last_user_idx = idx
    if last_user_idx is None:
        return copied
    content = copied[last_user_idx].get("content")
    if isinstance(content, str):
        if "/no_think" not in content:
            copied[last_user_idx]["content"] = f"{content.rstrip()}\n/no_think"
        return copied
    if isinstance(content, list):
        last_text_idx: Optional[int] = None
        for idx, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "text":
                last_text_idx = idx
        if last_text_idx is None:
            content.append({"type": "text", "text": "/no_think"})
        else:
            part = content[last_text_idx]
            text = str(part.get("text") or "")
            if "/no_think" not in text:
                part["text"] = f"{text.rstrip()}\n/no_think"
        return copied
    copied[last_user_idx]["content"] = [{"type": "text", "text": "/no_think"}]
    return copied


def _qwen_neutralize_thinking_prefill(
    rendered_prompt: Any,
    *,
    disable_thinking: bool,
    model_id: Optional[str],
) -> Any:
    if (
        not disable_thinking
        or not _qwen_model_id_is_thinking(model_id)
        or not isinstance(rendered_prompt, str)
    ):
        return rendered_prompt
    stripped = rendered_prompt.rstrip()
    if stripped.endswith("<think>"):
        return f"{stripped}\n\n</think>\n\n"
    return rendered_prompt


def _apply_qwen_chat_template(
    processor: Any,
    messages: Sequence[Dict[str, Any]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    tools: Optional[List[Dict[str, Any]]] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    base_kwargs: Dict[str, Any] = {
        "tokenize": tokenize,
        "add_generation_prompt": bool(add_generation_prompt),
    }
    if tools is not None:
        base_kwargs["tools"] = tools
    template_kwargs = {
        key: value for key, value in (chat_template_kwargs or {}).items() if value is not None
    }
    template_call_kwargs: Dict[str, Any] = {}
    processor_call_kwargs: Dict[str, Any] = {}
    if template_kwargs:
        try:
            from transformers.processing_utils import _get_template_variables

            chat_template = getattr(processor, "chat_template", None)
            if isinstance(chat_template, Mapping):
                chat_template = chat_template.get("default")
            template_vars = set(_get_template_variables(chat_template)) if chat_template else set()
            template_call_kwargs = {
                key: value for key, value in template_kwargs.items() if key in template_vars
            }
            processor_call_kwargs = {
                key: value for key, value in template_kwargs.items() if key not in template_vars
            }
        except Exception:
            template_call_kwargs = dict(template_kwargs)
            processor_call_kwargs = {}
    attempts: List[Dict[str, Any]] = []
    if template_kwargs:
        first_attempt = {**base_kwargs, **template_call_kwargs}
        if processor_call_kwargs:
            first_attempt["processor_kwargs"] = processor_call_kwargs
        attempts.append(first_attempt)
        if template_call_kwargs != template_kwargs:
            attempts.append({**base_kwargs, **template_kwargs})
        attempts.append({**base_kwargs, "chat_template_kwargs": template_kwargs})
    attempts.append(dict(base_kwargs))
    if tools is not None:
        without_tools = dict(base_kwargs)
        without_tools.pop("tools", None)
        attempts.append(without_tools)
    last_type_error: Optional[TypeError] = None
    for kwargs in attempts:
        try:
            return processor.apply_chat_template(messages, **kwargs)
        except TypeError as exc:
            last_type_error = exc
    if last_type_error is not None:
        raise last_type_error
    return processor.apply_chat_template(messages, **base_kwargs)


def _run_qwen_text_inference(
    prompt: str,
    *,
    max_new_tokens: Optional[int] = None,
    system_prompt_override: Optional[str] = None,
    model_id_override: Optional[str] = None,
    runtime_override: Optional[Tuple[Any, Any]] = None,
    decode_override: Optional[Dict[str, Any]] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int]:
    """Run a text-only caption editor pass with the selected Qwen runtime.

    Window merge is a reconciliation stage over already generated visual
    observations. Keeping it text-only avoids resubmitting a large full-image
    tensor to MLX for a prompt whose job is editing, not image inspection.
    """
    _raise_if_qwen_cancelled()
    if runtime_override is not None:
        runtime = runtime_override
    elif model_id_override:
        runtime = _ensure_qwen_ready_for_caption(model_id_override)
    else:
        runtime = _ensure_qwen_ready()
    _raise_if_qwen_cancelled()
    runtime_platform = _qwen_runtime_platform_from_runtime(runtime)
    resolved_model_id = (
        model_id_override
        or _qwen_runtime_model_id(runtime)
        or (active_qwen_metadata or {}).get("model_id")
        or QWEN_MODEL_NAME
    )
    sys_prompt = (
        system_prompt_override
        if system_prompt_override is not None
        else (active_qwen_metadata or {}).get("system_prompt")
    )
    messages: List[Dict[str, Any]] = []
    if sys_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    llm_call_id = uuid.uuid4().hex[:12]
    token = qwen_llm_call_id_var.set(llm_call_id)
    try:
        _qwen_caption_io_input(
            call_id=llm_call_id,
            source="qwen_text_inference",
            runtime_platform=runtime_platform,
            model_id=resolved_model_id,
            messages=messages,
            prompt_text=prompt,
            system_prompt=str(sys_prompt or ""),
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
            decode_override=decode_override,
            chat_template_kwargs=chat_template_kwargs,
        )
        output_text = _run_qwen_chat(
            messages,
            max_new_tokens=max_new_tokens,
            model_id_override=resolved_model_id,
            runtime_override=runtime,
            decode_override=decode_override,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=True,
        )
    finally:
        qwen_llm_call_id_var.reset(token)
    _qwen_caption_io_output(
        call_id=llm_call_id,
        source="qwen_text_inference",
        runtime_platform=runtime_platform,
        model_id=resolved_model_id,
        output_text=output_text,
        loop_detected=_caption_repetition_loop_detected_impl(output_text),
        degenerate_reason=_caption_degenerate_reason_impl(output_text, allow_short_caption=True),
    )
    return output_text, 0, 0


def _run_qwen_inference(
    prompt: str,
    pil_img: Image.Image,
    max_new_tokens: Optional[int] = None,
    system_prompt_override: Optional[str] = None,
    model_id_override: Optional[str] = None,
    runtime_override: Optional[Tuple[Any, Any]] = None,
    decode_override: Optional[Dict[str, Any]] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int]:
    """Execute a Qwen 3 VL inference following the reference recipe."""
    _raise_if_qwen_cancelled()
    if runtime_override is not None:
        runtime = runtime_override
    elif model_id_override:
        runtime = _ensure_qwen_ready_for_caption(model_id_override)
    else:
        runtime = _ensure_qwen_ready()
    _raise_if_qwen_cancelled()
    if isinstance(runtime, QwenRuntime) and runtime.platform == QWEN_PLATFORM_MLX:
        try:
            return _run_qwen_inference_mlx(
                runtime,
                prompt,
                pil_img,
                max_new_tokens=max_new_tokens,
                system_prompt_override=system_prompt_override,
                decode_override=decode_override,
                chat_template_kwargs=chat_template_kwargs,
            )
        finally:
            _qwen_mlx_post_generation_cleanup()
    model, processor = runtime
    resolved_model_id = (
        model_id_override
        or (active_qwen_metadata or {}).get("model_id")
        or QWEN_MODEL_NAME
    )
    messages: List[Dict[str, Any]] = []
    sys_prompt = (
        system_prompt_override
        if system_prompt_override is not None
        else (active_qwen_metadata or {}).get("system_prompt")
    )
    if sys_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }
    )
    disable_thinking = _qwen_disable_thinking_requested(chat_template_kwargs)
    if disable_thinking and _qwen_model_id_is_thinking(resolved_model_id):
        messages = _qwen_messages_with_no_think(messages)
    text = _apply_qwen_chat_template(
        processor,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    text = _qwen_neutralize_thinking_prefill(
        text,
        disable_thinking=disable_thinking,
        model_id=resolved_model_id,
    )
    llm_call_id = uuid.uuid4().hex[:12]
    qwen_llm_call_id_var.set(llm_call_id)
    _qwen_caption_io_input(
        call_id=llm_call_id,
        source="qwen_inference",
        model_id=resolved_model_id,
        messages=messages,
        prompt_text=prompt,
        rendered_prompt=text,
        system_prompt=str(sys_prompt or ""),
        user_prompt=prompt,
        image_width=int(pil_img.width),
        image_height=int(pil_img.height),
        max_new_tokens=max_new_tokens,
        decode_override=decode_override,
        chat_template_kwargs=chat_template_kwargs,
    )
    _agent_full_trace_write(
        {
            "type": "llm_input",
            "source": "qwen_inference",
            "model_id": resolved_model_id,
            "messages": messages,
            "prompt_text": text,
            "max_new_tokens": int(max_new_tokens) if max_new_tokens is not None else None,
            "decode_override": decode_override,
            "chat_template_kwargs": chat_template_kwargs,
        }
    )
    image_inputs, video_inputs = process_vision_info(messages)
    max_seq_len = _resolve_qwen_max_seq_len(model)
    max_input_len = None
    requested_max = int(max_new_tokens) if max_new_tokens is not None else QWEN_MAX_NEW_TOKENS
    if max_seq_len:
        if requested_max >= max_seq_len:
            requested_max = max(1, max_seq_len - 1)
    original_requested_max = requested_max
    preview_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    input_len = int(preview_inputs.input_ids.shape[1])
    num_images = len(image_inputs) if image_inputs is not None else 0
    effective_len, vision_tokens = _qwen_effective_input_len(preview_inputs, input_len, num_images)
    explicit_budget = _caption_explicit_token_budget_active()
    adapted_requested_max = _qwen_caption_prompt_aware_max_new_tokens(
        requested_max,
        prompt_tokens=int(effective_len),
        model_id=resolved_model_id,
        explicit=explicit_budget,
    )
    if (
        _qwen_caption_generation_active()
        and not explicit_budget
        and adapted_requested_max != requested_max
    ):
        logger.info(
            "[qwen-caption] prompt-aware Transformers output budget %s -> %s for %s prompt tokens",
            requested_max,
            adapted_requested_max,
            effective_len,
        )
        requested_max = adapted_requested_max
    _qwen_caption_io_prompt_budget(
        call_id=llm_call_id,
        source="qwen_inference",
        runtime_platform=QWEN_PLATFORM_TRANSFORMERS,
        model_id=resolved_model_id,
        prompt_tokens=int(effective_len),
        requested_max_new_tokens=original_requested_max,
        effective_max_new_tokens=requested_max,
        explicit_max_new_tokens=explicit_budget,
    )
    _qwen_progress_update(
        phase="prompt",
        phase_label="Preparing Prompt",
        progress=0.45,
        message="Tokenizing image prompt",
        input_tokens=int(effective_len),
        max_new_tokens=requested_max,
    )
    if max_seq_len:
        if effective_len + requested_max > max_seq_len:
            requested_max = max(1, max_seq_len - effective_len)
        if effective_len > max_seq_len:
            logger.warning(
                "[qwen] effective input length %s exceeds max_seq_len %s; truncating prompt.",
                effective_len,
                max_seq_len,
            )
            if vision_tokens is not None:
                max_input_len = max(1, max_seq_len - requested_max - vision_tokens + num_images)
            else:
                max_input_len = max(1, max_seq_len - requested_max)
    max_new_tokens = requested_max
    if max_input_len is None:
        inputs = preview_inputs
    else:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=max_input_len,
            return_tensors="pt",
        )
    model_device = None
    try:
        emb = getattr(model, "get_input_embeddings", None)
        if callable(emb):
            emb_layer = emb()
            if emb_layer is not None and hasattr(emb_layer, "weight"):
                model_device = emb_layer.weight.device
    except Exception:
        model_device = None
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except Exception:
            model_device = None
    device = (
        model_device
        or qwen_device
        or _resolve_qwen_device_impl(QWEN_DEVICE_PREF, torch_module=torch)
    )

    def _move_nested_to_device(val: Any, target_device: Any) -> Any:
        if torch.is_tensor(val):
            return val.to(target_device)
        if isinstance(val, dict):
            return {k: _move_nested_to_device(v, target_device) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return type(val)(_move_nested_to_device(v, target_device) for v in val)
        return val

    try:
        inputs = inputs.to(device)
    except Exception:
        pass
    try:
        if hasattr(inputs, "data") and isinstance(inputs.data, dict):
            inputs.data = _move_nested_to_device(inputs.data, device)
        else:
            inputs = _move_nested_to_device(inputs, device)
    except Exception:
        inputs = inputs.to(device)
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": (
            int(max_new_tokens) if max_new_tokens is not None else QWEN_MAX_NEW_TOKENS
        ),
    }
    use_sampling = QWEN_DO_SAMPLE
    if decode_override is not None and "do_sample" in decode_override:
        use_sampling = bool(decode_override.get("do_sample"))
    if not use_sampling:
        gen_config = getattr(model, "generation_config", None)
        if gen_config is not None and hasattr(gen_config, "clone"):
            try:
                gen_config = gen_config.clone()
            except Exception:
                gen_config = None
        if gen_config is not None:
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(gen_config, attr):
                    setattr(gen_config, attr, None)
            if hasattr(gen_config, "do_sample"):
                gen_config.do_sample = False
            gen_kwargs["generation_config"] = gen_config
    if use_sampling:
        temperature = QWEN_TEMPERATURE
        top_p = QWEN_TOP_P
        top_k = None
        presence_penalty = None
        repetition_penalty = None
        no_repeat_ngram_size = None
        if decode_override is not None:
            temperature = decode_override.get("temperature", temperature)
            top_p = decode_override.get("top_p", top_p)
            top_k = decode_override.get("top_k", top_k)
            presence_penalty = decode_override.get("presence_penalty", presence_penalty)
            repetition_penalty = decode_override.get("repetition_penalty", repetition_penalty)
            no_repeat_ngram_size = decode_override.get(
                "no_repeat_ngram_size", no_repeat_ngram_size
            )
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        if presence_penalty is not None and _qwen_supports_presence_penalty(model):
            gen_kwargs["presence_penalty"] = float(presence_penalty)
    else:
        gen_kwargs["do_sample"] = False
        repetition_penalty = decode_override.get("repetition_penalty") if decode_override else None
        no_repeat_ngram_size = (
            decode_override.get("no_repeat_ngram_size") if decode_override else None
        )
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)
    if no_repeat_ngram_size is not None:
        gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    tokenizer = getattr(processor, "tokenizer", None)
    try:
        input_prompt_len = int(inputs.input_ids.shape[1])
    except Exception:
        try:
            input_prompt_len = int(inputs["input_ids"].shape[1])
        except Exception:
            input_prompt_len = 0
    loop_guard_state: Dict[str, Any] = {"triggered": False}
    loop_guard_stop = threading.Event()
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _QwenCancelStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):  # type: ignore[no-untyped-def]
                return _qwen_cancel_requested() or loop_guard_stop.is_set()

        class _QwenRepetitionStoppingCriteria(StoppingCriteria):
            def __init__(self) -> None:
                self._last_check_len = 0

            def __call__(self, input_ids, scores, **kwargs):  # type: ignore[no-untyped-def]
                if tokenizer is None or _qwen_cancel_requested():
                    return False
                try:
                    generated_len = int(input_ids.shape[-1]) - input_prompt_len
                    if generated_len < 32 or generated_len - self._last_check_len < 8:
                        return False
                    self._last_check_len = generated_len
                    generated_ids = input_ids[0][input_prompt_len:]
                    if hasattr(generated_ids, "detach"):
                        generated_ids = generated_ids.detach().cpu().tolist()
                    text = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if _caption_repetition_loop_detected_impl(str(text or "")):
                        loop_guard_state["triggered"] = True
                        loop_guard_state["raw_output"] = str(text or "")
                        loop_guard_state["reason"] = (
                            _caption_degenerate_reason_impl(
                                str(text or ""),
                                allow_short_caption=True,
                            )
                            or "repetition_loop"
                        )
                        return True
                except Exception:
                    return False
                return False

        criteria = [_QwenCancelStoppingCriteria()]
        if tokenizer is not None and input_prompt_len > 0:
            criteria.append(_QwenRepetitionStoppingCriteria())
        existing_criteria = gen_kwargs.get("stopping_criteria")
        if existing_criteria is None:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(criteria)
        elif hasattr(existing_criteria, "append"):
            for criterion in criteria:
                existing_criteria.append(criterion)
    except Exception:
        pass
    _qwen_progress_update(
        phase="generate",
        phase_label="Generating",
        progress=0.55,
        message="Starting Qwen generation",
        max_new_tokens=int(max_new_tokens) if max_new_tokens is not None else QWEN_MAX_NEW_TOKENS,
    )
    _raise_if_qwen_cancelled()
    output_text: Optional[str] = None
    streamed_output = False
    if tokenizer is not None:
        try:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                timeout=1.0,
            )
            stream_kwargs = dict(gen_kwargs)
            stream_kwargs["streamer"] = streamer
            generated_parts: List[str] = []
            generation_errors: List[BaseException] = []
            stream_loop_guard = (
                _QwenCaptionStreamLoopGuard() if _qwen_caption_generation_active() else None
            )

            def _generate_stream() -> None:
                try:
                    _raise_if_qwen_cancelled()
                    with torch.inference_mode():
                        try:
                            model.generate(**inputs, **stream_kwargs)
                        except RuntimeError as exc:
                            if QWEN_DO_SAMPLE and "probability tensor" in str(exc).lower():
                                fallback_kwargs = {**stream_kwargs}
                                fallback_kwargs["do_sample"] = False
                                fallback_kwargs.pop("temperature", None)
                                fallback_kwargs.pop("top_p", None)
                                model.generate(**inputs, **fallback_kwargs)
                            else:
                                raise
                except BaseException as exc:  # noqa: BLE001
                    generation_errors.append(exc)

            thread = threading.Thread(
                target=_generate_stream,
                name="qwen-vl-inference-streamer",
                daemon=True,
            )
            thread.start()
            iterator = iter(streamer)
            while True:
                try:
                    new_text = next(iterator)
                except StopIteration:
                    break
                except queue.Empty:
                    if generation_errors:
                        break
                    if _qwen_cancel_requested():
                        generation_errors.append(QwenCancellationRequested("qwen_cancelled"))
                        break
                    if thread.is_alive():
                        _qwen_progress_update(
                            phase="generate",
                            phase_label="Generating",
                            message="Generating response tokens",
                        )
                        continue
                    break
                if not new_text:
                    continue
                _raise_if_qwen_cancelled()
                streamed_output = True
                generated_parts.append(new_text)
                _qwen_progress_token(
                    new_text,
                    generated_tokens=None,
                    max_new_tokens=int(max_new_tokens)
                    if max_new_tokens is not None
                    else QWEN_MAX_NEW_TOKENS,
                )
                if stream_loop_guard is not None and stream_loop_guard.append(new_text):
                    loop_guard_state["triggered"] = True
                    loop_guard_state["reason"] = stream_loop_guard.reason
                    loop_guard_stop.set()
                    _qwen_caption_stream_loop_detected_event(
                        source="qwen_inference",
                        runtime_platform=QWEN_PLATFORM_TRANSFORMERS,
                        model_id=resolved_model_id,
                        guard=stream_loop_guard,
                    )
                    break
            thread.join()
            if generation_errors:
                raise generation_errors[0]
            _raise_if_qwen_cancelled()
            output_text = "".join(generated_parts)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, QwenCancellationRequested):
                raise
            logger.warning("[qwen] token streaming failed; falling back to blocking generate: %s", exc)
            output_text = None
    if output_text is None:
        _raise_if_qwen_cancelled()
        with torch.inference_mode():
            try:
                generated_ids = model.generate(**inputs, **gen_kwargs)
            except RuntimeError as exc:
                if QWEN_DO_SAMPLE and "probability tensor" in str(exc).lower():
                    fallback_kwargs = {**gen_kwargs}
                    fallback_kwargs["do_sample"] = False
                    fallback_kwargs.pop("temperature", None)
                    fallback_kwargs.pop("top_p", None)
                    generated_ids = model.generate(**inputs, **fallback_kwargs)
                else:
                    raise
        _raise_if_qwen_cancelled()
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        raw_loop_output = (
            str(loop_guard_state.get("raw_output") or "")
            if bool(loop_guard_state.get("triggered"))
            else output_text
        )
        output_text = _qwen_maybe_trim_repetition_loop_output(
            output_text,
            triggered=bool(loop_guard_state.get("triggered")),
            replace_live_output=False,
        )
        if bool(loop_guard_state.get("triggered")) and _qwen_caption_generation_active():
            raise QwenCaptionLoopDetected(
                "qwen_caption_repetition_loop",
                raw_output=raw_loop_output,
                trimmed_output=output_text,
                degenerate_reason=str(loop_guard_state.get("reason") or "") or None,
            )
        _qwen_progress_update(
            phase="generate",
            phase_label="Generated",
            progress=0.90,
            message="Qwen response generated",
            token_preview=output_text,
            append_token_preview_to_live_output=not streamed_output,
        )
    else:
        raw_loop_output = output_text
        output_text = _qwen_maybe_trim_repetition_loop_output(
            output_text,
            triggered=bool(loop_guard_state.get("triggered")),
            replace_live_output=True,
        )
        if bool(loop_guard_state.get("triggered")) and _qwen_caption_generation_active():
            raise QwenCaptionLoopDetected(
                "qwen_caption_repetition_loop",
                raw_output=raw_loop_output,
                trimmed_output=output_text,
                degenerate_reason=str(loop_guard_state.get("reason") or "") or None,
            )
        _qwen_progress_update(
            phase="generate",
            phase_label="Generated",
            progress=0.90,
            message="Qwen response generated",
            token_preview=output_text,
            append_token_preview_to_live_output=not streamed_output,
        )
    _agent_full_trace_write(
        {
            "type": "llm_output",
            "source": "qwen_inference",
            "model_id": resolved_model_id,
            "output_text": output_text,
        }
    )
    _qwen_caption_io_output(
        call_id=llm_call_id,
        source="qwen_inference",
        model_id=resolved_model_id,
        output_text=output_text,
        loop_detected=_caption_repetition_loop_detected_impl(output_text),
        degenerate_reason=_caption_degenerate_reason_impl(output_text, allow_short_caption=True),
    )
    grid = inputs.get("image_grid_thw")
    patch_size = 14
    try:
        vision_cfg = getattr(model, "config", None)
        vision_cfg = getattr(vision_cfg, "vision_config", None)
        if vision_cfg is not None and getattr(vision_cfg, "patch_size", None):
            patch_size = int(vision_cfg.patch_size)
        elif getattr(processor, "image_processor", None) is not None:
            patch = getattr(processor.image_processor, "patch_size", None)
            if patch:
                patch_size = int(patch)
    except Exception:
        patch_size = 14
    if grid is not None:
        grid_values = grid[0]
        input_height = int(grid_values[1].item() * patch_size)
        input_width = int(grid_values[2].item() * patch_size)
    else:
        input_height = pil_img.height
        input_width = pil_img.width
    return output_text, input_width, input_height


def _run_qwen_chat(
    messages: List[Dict[str, Any]],
    *,
    max_new_tokens: Optional[int] = None,
    model_id_override: Optional[str] = None,
    runtime_override: Optional[Tuple[Any, Any]] = None,
    decode_override: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    add_generation_prompt: bool = True,
    assistant_prefix: Optional[str] = None,
    thinking_effort: Optional[float] = None,
    thinking_scale_factor: Optional[float] = None,
    immediate_action_bias: Optional[bool] = None,
    immediate_action_min_chars: Optional[int] = None,
    immediate_action_min_seconds: Optional[float] = None,
    immediate_action_logit_bias: Optional[float] = None,
) -> str:
    if runtime_override is not None:
        runtime = runtime_override
    elif model_id_override:
        runtime = _ensure_qwen_ready_for_caption(model_id_override)
    else:
        runtime = _ensure_qwen_ready()
    if isinstance(runtime, QwenRuntime) and runtime.platform == QWEN_PLATFORM_MLX:
        try:
            output_text = _run_qwen_chat_mlx(
                runtime,
                messages,
                max_new_tokens=max_new_tokens,
                decode_override=decode_override,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
                add_generation_prompt=add_generation_prompt,
                assistant_prefix=assistant_prefix,
            )
        finally:
            _qwen_mlx_post_generation_cleanup()
        _agent_full_trace_write(
            {
                "type": "llm_output",
                "runtime_platform": QWEN_PLATFORM_MLX,
                "model_id": runtime.model_id,
                "output_text": output_text,
            }
        )
        return output_text
    model, processor = runtime
    if assistant_prefix:
        add_generation_prompt = True
    text = _apply_qwen_chat_template(
        processor,
        messages,
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    if assistant_prefix:
        text = f"{text}{assistant_prefix}"
    resolved_model_id = (
        model_id_override
        or (active_qwen_metadata or {}).get("model_id")
        or QWEN_MODEL_NAME
    )
    _agent_full_trace_write(
        {
            "type": "llm_input",
            "model_id": resolved_model_id,
            "messages": messages,
            "prompt_text": text,
            "max_new_tokens": int(max_new_tokens) if max_new_tokens is not None else None,
            "decode_override": decode_override,
            "thinking_effort": thinking_effort,
            "thinking_scale_factor": thinking_scale_factor,
            "immediate_action_bias": immediate_action_bias,
            "immediate_action_min_chars": immediate_action_min_chars,
            "immediate_action_min_seconds": immediate_action_min_seconds,
            "immediate_action_logit_bias": immediate_action_logit_bias,
            "tools": tools,
            "chat_template_kwargs": chat_template_kwargs,
            "assistant_prefix": assistant_prefix,
        }
    )
    tokenizer = getattr(processor, "tokenizer", None)
    image_inputs, video_inputs = process_vision_info(messages)
    max_seq_len = _resolve_qwen_max_seq_len(model)
    requested_max = int(max_new_tokens) if max_new_tokens is not None else QWEN_MAX_NEW_TOKENS
    if max_seq_len:
        if requested_max >= max_seq_len:
            requested_max = max(1, max_seq_len - 1)
    original_requested_max = requested_max
    preview_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    input_len = int(preview_inputs.input_ids.shape[1])
    num_images = len(image_inputs) if image_inputs is not None else 0
    effective_len, vision_tokens = _qwen_effective_input_len(preview_inputs, input_len, num_images)
    explicit_budget = _caption_explicit_token_budget_active()
    adapted_requested_max = _qwen_caption_prompt_aware_max_new_tokens(
        requested_max,
        prompt_tokens=int(effective_len),
        model_id=resolved_model_id,
        explicit=explicit_budget,
    )
    if (
        _qwen_caption_generation_active()
        and not explicit_budget
        and adapted_requested_max != requested_max
    ):
        logger.info(
            "[qwen-caption] prompt-aware Transformers chat output budget %s -> %s for %s prompt tokens",
            requested_max,
            adapted_requested_max,
            effective_len,
        )
        requested_max = adapted_requested_max
    _qwen_caption_io_prompt_budget(
        call_id=qwen_llm_call_id_var.get(),
        source="qwen_chat",
        runtime_platform=QWEN_PLATFORM_TRANSFORMERS,
        model_id=resolved_model_id,
        prompt_tokens=int(effective_len),
        requested_max_new_tokens=original_requested_max,
        effective_max_new_tokens=requested_max,
        explicit_max_new_tokens=explicit_budget,
    )
    _qwen_progress_update(
        phase="prompt",
        phase_label="Preparing Prompt",
        progress=0.45,
        message="Tokenizing chat prompt",
        input_tokens=int(effective_len),
        max_new_tokens=requested_max,
    )
    max_input_len = None
    if max_seq_len:
        if effective_len + requested_max > max_seq_len:
            requested_max = max(1, max_seq_len - effective_len)
        if effective_len > max_seq_len:
            if vision_tokens is not None:
                max_input_len = max(1, max_seq_len - requested_max - vision_tokens + num_images)
            else:
                max_input_len = max(1, max_seq_len - requested_max)
    max_new_tokens = requested_max
    if max_input_len is None:
        inputs = preview_inputs
    else:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=max_input_len,
            return_tensors="pt",
        )
    device = qwen_device or _resolve_qwen_device_impl(QWEN_DEVICE_PREF, torch_module=torch)
    inputs = inputs.to(device)
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": (
            int(max_new_tokens) if max_new_tokens is not None else QWEN_MAX_NEW_TOKENS
        ),
    }
    use_sampling = QWEN_DO_SAMPLE
    if decode_override is not None and "do_sample" in decode_override:
        use_sampling = bool(decode_override.get("do_sample"))
    if use_sampling:
        temperature = QWEN_TEMPERATURE
        top_p = QWEN_TOP_P
        top_k = None
        presence_penalty = None
        if decode_override is not None:
            temperature = decode_override.get("temperature", temperature)
            top_p = decode_override.get("top_p", top_p)
            top_k = decode_override.get("top_k", top_k)
            presence_penalty = decode_override.get("presence_penalty", presence_penalty)
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        if presence_penalty is not None and _qwen_supports_presence_penalty(model):
            gen_kwargs["presence_penalty"] = float(presence_penalty)
    else:
        gen_kwargs["do_sample"] = False
    thinking_processor = _qwen_build_thinking_effort_processor(
        tokenizer, thinking_effort, thinking_scale_factor
    )
    immediate_processor = _qwen_build_immediate_action_processor(
        tokenizer,
        immediate_action_bias,
        immediate_action_min_chars,
        immediate_action_min_seconds,
        immediate_action_logit_bias,
    )
    _qwen_append_logits_processor(gen_kwargs, thinking_processor)
    _qwen_append_logits_processor(gen_kwargs, immediate_processor)
    _qwen_progress_update(
        phase="generate",
        phase_label="Generating",
        progress=0.55,
        message="Starting Qwen chat generation",
        max_new_tokens=int(max_new_tokens) if max_new_tokens is not None else QWEN_MAX_NEW_TOKENS,
    )
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    output_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    _qwen_progress_update(
        phase="generate",
        phase_label="Generated",
        progress=0.90,
        message="Qwen chat response generated",
        token_preview=output_text,
    )
    _agent_full_trace_write(
        {
            "type": "llm_output",
            "model_id": model_id_override
            or (active_qwen_metadata or {}).get("model_id")
            or QWEN_MODEL_NAME,
            "output_text": output_text,
        }
    )
    return output_text


def qwen_infer(payload: QwenInferenceRequest):
    prompt_type = payload.prompt_type.lower()
    if prompt_type not in {"bbox", "point", "bbox_sam"}:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_prompt_type")
    base_model_id = (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME
    platform_name = _resolve_qwen_runtime_platform(
        str(base_model_id),
        adapter_path=active_qwen_model_path,
        metadata=active_qwen_metadata,
    )
    _qwen_progress_start(
        kind="inference",
        model_id=str(base_model_id),
        platform=platform_name,
        message="Preparing Qwen detection request",
        max_new_tokens=QWEN_MAX_NEW_TOKENS,
    )
    try:
        pil_img, np_img, token = resolve_image_payload(
            payload.image_base64,
            payload.image_token,
            getattr(payload, "sam_variant", None),
        )
    except HTTPException:
        _qwen_progress_error("Qwen inference image payload failed")
        raise
    except Exception as exc:  # noqa: BLE001
        _qwen_progress_error(f"Qwen inference image payload failed: {exc}")
        raise
    try:
        manual_prompt = (payload.prompt or "").strip()
        if manual_prompt:
            final_prompt = manual_prompt
        else:
            item_list = (payload.item_list or "").strip()
            final_prompt = _render_qwen_prompt_impl(
                prompt_type,
                items=item_list,
                image_type=(payload.image_type or "").strip() or None,
                extra_context=(payload.extra_context or "").strip() or None,
                get_config_fn=lambda: _get_qwen_prompt_config_impl(
                    qwen_prompt_config, qwen_config_lock
                ),
                http_exception_cls=HTTPException,
                http_422=HTTP_422_UNPROCESSABLE_CONTENT,
            )
    except HTTPException:
        _qwen_progress_error("Qwen prompt rendering failed")
        raise
    except Exception as exc:  # noqa: BLE001
        _qwen_progress_error(f"Qwen prompt rendering failed: {exc}")
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"qwen_prompt_render_failed:{exc}",
        ) from exc
    try:
        _qwen_progress_update(
            phase="prepare",
            phase_label="Preparing Image",
            progress=0.06,
            message="Preparing image for Qwen inference",
        )
        qwen_text, proc_w, proc_h = _run_qwen_inference(final_prompt, pil_img)
    except HTTPException as exc:
        detail = _http_exception_detail_text(exc)
        _qwen_progress_error(
            f"Qwen inference failed: {detail}" if detail else "Qwen inference failed"
        )
        raise
    except Exception as exc:  # noqa: BLE001
        _qwen_progress_error(f"Qwen inference failed: {exc}")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_inference_failed:{exc}"
        ) from exc
    print("[Qwen prompt]", final_prompt)
    print("[Qwen raw output]", qwen_text)
    _qwen_progress_update(
        phase="postprocess",
        phase_label="Parsing",
        progress=0.92,
        message="Parsing Qwen detections",
        token_preview=qwen_text,
    )
    warnings: List[str] = []
    try:
        _, items = _extract_qwen_json_block(qwen_text)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        warnings.append(f"parse_error:{detail}")
        print(f"[Qwen parse error] {detail}; raw text follows:\n{qwen_text}")
        _qwen_progress_finish("Qwen finished, but the response did not parse", token_preview=qwen_text)
        return QwenInferenceResponse(
            boxes=[],
            raw_response=qwen_text,
            prompt=final_prompt,
            prompt_type=prompt_type,  # type: ignore[arg-type]
            warnings=warnings,
            image_token=token,
        )
    normalized_items = _qwen_items_from_payload(items)
    if not normalized_items:
        print("[Qwen parsed but empty list]", qwen_text)
        warnings.append("no_results")
        _qwen_progress_finish("Qwen finished with no detections", token_preview=qwen_text)
        return QwenInferenceResponse(
            boxes=[],
            raw_response=qwen_text,
            prompt=final_prompt,
            prompt_type=prompt_type,  # type: ignore[arg-type]
            warnings=warnings,
            image_token=token,
        )
    variant = _default_variant(getattr(payload, "sam_variant", None))
    limit = payload.max_results or 8
    image_name = getattr(payload, "image_name", None)
    if prompt_type == "bbox":
        boxes = _qwen_bbox_results(
            normalized_items, proc_w, proc_h, pil_img.width, pil_img.height, limit=limit
        )
    elif prompt_type == "bbox_sam":
        boxes = _qwen_bbox_sam_results(
            normalized_items,
            proc_w,
            proc_h,
            pil_img,
            np_img,
            token,
            variant,
            image_name=image_name,
            limit=limit,
        )
    else:
        boxes = _qwen_point_results(
            normalized_items,
            proc_w,
            proc_h,
            pil_img,
            np_img,
            token,
            variant,
            image_name=image_name,
            limit=limit,
        )
    if not boxes:
        warnings.append("no_results")
    _qwen_progress_finish(
        f"Qwen finished with {len(boxes)} detection{'s' if len(boxes) != 1 else ''}",
        token_preview=qwen_text,
    )
    return QwenInferenceResponse(
        boxes=boxes,
        raw_response=qwen_text,
        prompt=final_prompt,
        prompt_type=prompt_type,  # type: ignore[arg-type]
        warnings=warnings,
        image_token=token,
    )


def _resolve_qwen_caption_refinement_model_id(
    refinement_model_id: Optional[str],
    *,
    desired_model_id: str,
    active_model_id: Optional[str],
) -> str:
    raw = str(refinement_model_id or "").strip()
    if not raw or raw.lower() == "auto":
        return _qwen_caption_default_refinement_model_id(desired_model_id)
    if raw.lower() == "same":
        return desired_model_id
    if raw.lower() == "active":
        return str(active_model_id or "").strip() or QWEN_MODEL_NAME
    return raw


def _qwen_caption_known_model_ids(model_id: Optional[str]) -> Set[str]:
    if is_qwen_mlx_model_id(model_id):
        return {str(entry.get("id") or "") for entry in QWEN_MLX_VISION_MODEL_OPTIONS}
    return {str(entry.get("id") or "") for entry in QWEN_TRANSFORMERS_MODEL_OPTIONS}


def _qwen_caption_default_refinement_model_id(desired_model_id: Optional[str]) -> str:
    desired = str(desired_model_id or "").strip() or QWEN_MODEL_NAME
    if not _qwen_model_id_is_thinking(desired):
        return desired
    known_ids = _qwen_caption_known_model_ids(desired)
    exact_instruct = desired.replace("Thinking", "Instruct")
    platform = QWEN_PLATFORM_MLX if is_qwen_mlx_model_id(desired) else QWEN_PLATFORM_TRANSFORMERS
    if exact_instruct in known_ids:
        availability = _qwen_model_local_state(exact_instruct, platform)
        if availability.get("local") and not availability.get("partial"):
            return exact_instruct
    if is_qwen_mlx_model_id(desired):
        caption_instruct = _resolve_qwen_variant_model_id_impl(
            QWEN_MLX_CAPTION_MODEL_NAME,
            "Instruct",
        )
        return _effective_qwen_model_id_for_platform(caption_instruct, QWEN_PLATFORM_MLX)
    small_instruct = "Qwen/Qwen3-VL-4B-Instruct"
    small_availability = _qwen_model_local_state(small_instruct, QWEN_PLATFORM_TRANSFORMERS)
    if small_availability.get("local") and not small_availability.get("partial"):
        return small_instruct
    if exact_instruct in known_ids:
        return exact_instruct
    return small_instruct


def _resolve_qwen_caption_default_model_id(
    base_model_id: Optional[str],
    variant: Optional[str],
) -> str:
    """Resolve the implicit caption model for "Use active model" requests.

    The general Apple Silicon Qwen default is intentionally allowed to be a
    large experimental MLX checkpoint for single-call inference and agent
    review. Captioning is different: one click may run many window, cleanup,
    and merge generations. Use a smaller MLX vision model unless the caller
    explicitly selects a large model.
    """

    base = str(base_model_id or QWEN_MODEL_NAME).strip() or QWEN_MODEL_NAME
    platform_name = _resolve_qwen_runtime_platform(
        base,
        adapter_path=active_qwen_model_path,
        metadata=active_qwen_metadata,
    )
    if platform_name != QWEN_PLATFORM_MLX:
        return _resolve_qwen_variant_model_id_impl(base, variant)
    caption_base = _resolve_qwen_variant_model_id_impl(
        QWEN_MLX_CAPTION_MODEL_NAME,
        variant,
    )
    return _effective_qwen_model_id_for_platform(caption_base, QWEN_PLATFORM_MLX)


def _qwen_caption_prompt_preview_section(
    *,
    title: str,
    user_prompt: str,
    kind: str = "prompt",
    model_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    note: Optional[str] = None,
    image_region: Optional[Dict[str, Any]] = None,
) -> QwenCaptionPromptPreviewSection:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    if user_prompt:
        messages.append({"role": "user", "content": str(user_prompt)})
    return QwenCaptionPromptPreviewSection(
        title=title,
        kind=kind,
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        note=note,
        image_region=image_region,
        chat_messages=messages,
    )


def _qwen_caption_preview_section_prompt_text(section: QwenCaptionPromptPreviewSection) -> str:
    parts = [
        str(getattr(section, "system_prompt", "") or "").strip(),
        str(getattr(section, "user_prompt", "") or "").strip(),
    ]
    return "\n".join(part for part in parts if part)


def _qwen_caption_prompt_preview_budget(
    sections: Sequence[QwenCaptionPromptPreviewSection],
    *,
    requested_max_new_tokens: Optional[Any],
    variant: Optional[str],
    caption_mode: Optional[str],
    final_caption_max_sentences: Optional[int],
    explicit_max_new_tokens: bool,
) -> Dict[str, Any]:
    section_budgets: List[Dict[str, Any]] = []
    total_prompt_chars = 0
    max_prompt_tokens = 0
    max_prompt_title = ""
    min_effective: Optional[int] = None
    max_effective: Optional[int] = None
    adapted_sections = 0
    for section in sections:
        prompt_text = _qwen_caption_preview_section_prompt_text(section)
        prompt_chars = len(prompt_text)
        prompt_tokens = _estimate_qwen_prompt_tokens(prompt_text)
        model_id = str(getattr(section, "model_id", "") or "").strip()
        section_requested = _resolve_qwen_caption_max_new_tokens(
            requested_max_new_tokens,
            model_id=model_id,
            variant=variant,
            caption_mode=caption_mode,
            final_caption_max_sentences=final_caption_max_sentences,
        )
        section_effective = _qwen_caption_prompt_aware_max_new_tokens(
            section_requested,
            prompt_tokens=prompt_tokens,
            model_id=model_id,
            explicit=explicit_max_new_tokens,
        )
        if section_effective != section_requested:
            adapted_sections += 1
        total_prompt_chars += prompt_chars
        if prompt_tokens > max_prompt_tokens:
            max_prompt_tokens = prompt_tokens
            max_prompt_title = section.title
        min_effective = section_effective if min_effective is None else min(min_effective, section_effective)
        max_effective = section_effective if max_effective is None else max(max_effective, section_effective)
        section_budgets.append(
            {
                "title": section.title,
                "kind": section.kind,
                "model_id": model_id or None,
                "prompt_chars": prompt_chars,
                "estimated_prompt_tokens": prompt_tokens,
                "requested_max_new_tokens": section_requested,
                "effective_max_new_tokens": section_effective,
                "adapted": section_effective != section_requested,
            }
        )
    return {
        "explicit_max_new_tokens": bool(explicit_max_new_tokens),
        "requested_max_new_tokens": (
            int(requested_max_new_tokens)
            if isinstance(requested_max_new_tokens, int)
            else requested_max_new_tokens
        ),
        "total_prompt_chars": total_prompt_chars,
        "max_prompt_tokens": max_prompt_tokens,
        "max_prompt_section": max_prompt_title,
        "effective_max_new_tokens_min": min_effective,
        "effective_max_new_tokens_max": max_effective,
        "adapted_sections": adapted_sections,
        "sections": section_budgets,
        "estimate_note": (
            "Preview uses an approximate text-token estimate. Runtime remeasures each rendered "
            "chat prompt with the active processor/tokenizer when available."
        ),
    }


def _format_qwen_caption_prompt_preview_text(
    sections: Sequence[QwenCaptionPromptPreviewSection],
    *,
    warnings: Sequence[str],
    meta: Mapping[str, Any],
) -> str:
    lines: List[str] = [
        "Complete Qwen caption prompt flow preview",
        f"Image: {meta.get('image_name') or '(current image)'}",
        f"Mode: {meta.get('caption_mode') or 'full'}",
        f"Caption model: {meta.get('model_id') or '(active/default)'}",
        f"Refinement model: {meta.get('refinement_model_id') or '(same/default)'}",
    ]
    prompt_budget = meta.get("prompt_budget")
    if isinstance(prompt_budget, Mapping):
        min_effective = prompt_budget.get("effective_max_new_tokens_min")
        max_effective = prompt_budget.get("effective_max_new_tokens_max")
        if min_effective == max_effective:
            effective_text = str(min_effective)
        else:
            effective_text = f"{min_effective}-{max_effective}"
        lines.extend(
            [
                (
                    "Prompt size estimate: largest section "
                    f"~{prompt_budget.get('max_prompt_tokens') or 0} tokens"
                    + (
                        f" ({prompt_budget.get('max_prompt_section')})"
                        if prompt_budget.get("max_prompt_section")
                        else ""
                    )
                    + f"; effective output budget {effective_text} tokens"
                    + (
                        " (explicit cap)"
                        if prompt_budget.get("explicit_max_new_tokens")
                        else " (Auto, prompt-aware)"
                    )
                    + "."
                ),
                str(prompt_budget.get("estimate_note") or "").strip(),
            ]
        )
    if warnings:
        lines.extend(["", "Warnings / generated-dependent stages:"])
        lines.extend([f"- {warning}" for warning in warnings if str(warning or "").strip()])
    for section in sections:
        lines.extend(["", "=" * 88, section.title, "=" * 88])
        if section.note:
            lines.append(f"Note: {section.note}")
        if section.model_id:
            lines.append(f"Model: {section.model_id}")
        if section.image_region:
            region_bits = ", ".join(
                f"{key}={value}" for key, value in section.image_region.items()
                if value is not None
            )
            if region_bits:
                lines.append(f"Image region: {region_bits}")
        if section.system_prompt:
            lines.extend(["", "--- system ---", section.system_prompt])
        if section.user_prompt:
            lines.extend(["", "--- user ---", section.user_prompt])
    return "\n".join(lines).rstrip() + "\n"


def _format_caption_window_sentence_instruction(payload: QwenCaptionRequest) -> str:
    try:
        min_sentences = int(getattr(payload, "caption_window_min_sentences", 1) or 1)
    except (TypeError, ValueError, OverflowError):
        min_sentences = 1
    try:
        max_sentences = int(getattr(payload, "caption_window_max_sentences", 3) or 3)
    except (TypeError, ValueError, OverflowError):
        max_sentences = 3
    min_sentences = max(1, min(min_sentences, 10))
    max_sentences = max(min_sentences, min(max_sentences, 10))
    if min_sentences == max_sentences:
        suffix = "" if min_sentences == 1 else "s"
        return f"Write {min_sentences} concrete sentence{suffix} about this region only. "
    return f"Write {min_sentences}-{max_sentences} concrete sentences about this region only. "


def _caption_window_detection_context_prompt() -> str:
    return (
        "Window observation policy: this call describes only the supplied crop/window. "
        "Use the user request as broad guidance, but do not follow full-image length instructions here. "
        "Use the crop image as truth and use crop counts/boxes as priors for the crop only. "
        "State crop counts as ordinary visual facts when they are useful, using digits. "
        "Describe relative crop layout in natural words, but never mention coordinates. "
        "Do not mention labels, hints, boxes, glossary text, prompts, or that counts were supplied. "
        "Do not invent object categories outside the crop evidence."
    )


def qwen_caption_prompt_preview(payload: QwenCaptionRequest) -> QwenCaptionPromptPreviewResponse:
    pil_img, _, _ = resolve_image_payload(payload.image_base64, payload.image_token, None)
    request_image_name = str(getattr(payload, "image_name", "") or "").strip()
    user_prompt = (payload.user_prompt or "").strip()
    include_counts = bool(payload.include_counts)
    include_coords = bool(payload.include_coords)
    max_boxes = payload.max_boxes if payload.max_boxes is not None else 0
    source_image_width = payload.image_width
    source_image_height = payload.image_height
    image_width = pil_img.width
    image_height = pil_img.height
    label_hints = _normalize_caption_hints_for_decoded_image(
        payload.label_hints or [],
        source_image_width,
        source_image_height,
        image_width,
        image_height,
    )
    allowed_labels = _allowed_caption_labels_impl(label_hints)
    caption_mode = payload.caption_mode or "full"
    restrict_to_labels = (
        payload.restrict_to_labels if payload.restrict_to_labels is not None else True
    )
    final_caption_max_sentences = payload.final_caption_max_sentences
    caption_all_windows = _resolve_caption_all_windows_impl(
        caption_mode,
        payload.caption_all_windows,
        has_label_hints=bool(label_hints),
        restrict_to_labels=bool(restrict_to_labels),
    )
    detailed_mode = caption_mode == "windowed"
    glossary_map = _caption_glossary_map_impl(
        payload.labelmap_glossary,
        [hint.label for hint in label_hints if hint.label],
    )
    allowed_labels_prompt = (
        [_caption_preferred_label_impl(label, glossary_map) for label in allowed_labels]
        if allowed_labels
        else []
    )
    prompt_text, counts, used_boxes, truncated = _build_qwen_caption_prompt_impl(
        user_prompt,
        label_hints,
        image_width,
        image_height,
        include_counts,
        include_coords,
        max_boxes,
        detailed_mode,
        restrict_to_labels=restrict_to_labels,
        labelmap_glossary=payload.labelmap_glossary,
        max_sentences=final_caption_max_sentences,
        context_prompt=payload.caption_detection_context_prompt,
    )
    glossary_line = ""
    if glossary_map:
        glossary_line = (
            "Use each glossary broad term as the canonical name for that labeled class. "
            "If no glossary term exists, use a natural English class term instead of the raw label name. "
            "Glossary variants are possible narrower members only when clearly visible. "
            "Never output labelmap class names, especially tokens with underscores or odd internal spellings."
        )
        prompt_text = f"{prompt_text}\n{glossary_line}"
    base_model_id = (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME
    variant = payload.model_variant or "auto"
    model_id_override = str(payload.model_id or "").strip()
    if model_id_override.lower() == "active":
        model_id_override = ""
    desired_model_id = (
        model_id_override
        if model_id_override
        else _resolve_qwen_caption_default_model_id(base_model_id, variant)
    )
    caption_refinement_model_id = _resolve_qwen_caption_refinement_model_id(
        payload.refinement_model_id,
        desired_model_id=desired_model_id,
        active_model_id=base_model_id,
    )
    final_only = bool(payload.final_answer_only)
    final_caption_length_instruction = _caption_length_instruction_impl(final_caption_max_sentences)
    caption_user_request_note = _caption_user_request_instruction_impl(user_prompt)
    broad_term_editor_note = _caption_editor_preserve_broad_terms_instruction_impl()
    short_caption_requested = _caption_user_requested_short_impl(
        user_prompt,
        final_caption_max_sentences,
    )
    two_stage = bool(payload.two_stage_refine)
    is_thinking = "Thinking" in desired_model_id or variant == "Thinking"
    if is_thinking:
        prompt_text = _adjust_prompt_for_thinking_impl(prompt_text)
    if final_caption_length_instruction and "final caption length:" not in prompt_text.lower():
        prompt_text = f"{prompt_text}\nFinal caption length: {final_caption_length_instruction}"
    default_system_prompt = (
        "You are a detailed captioning assistant. Use the image as truth. "
        "Provide a rich, multi-sentence caption when there is a lot to see. "
        "Do not mention labels, hints, bounding boxes, coordinates, glossary text, or that counts were provided. "
        "Do not output internal dataset class names or any token with underscores. "
        "Use natural-language class terms from the provided context when available. "
        "Use narrower subtypes only when the image clearly supports them. "
        "If the hints conflict with the image, mention the uncertainty briefly. "
        "Stop when the caption is complete; never repeat a sentence or keep writing to fill the token budget.\n"
        "Respond in English only.\n"
        "Return only the final caption. Do not include reasoning or preamble."
    )
    custom_system_prompt = bool((payload.caption_system_prompt or "").strip())
    system_prompt = (payload.caption_system_prompt or "").strip() or default_system_prompt
    if not custom_system_prompt:
        if final_only and is_thinking and not two_stage:
            system_prompt = f"{system_prompt} Respond with exactly <final>...</final> and nothing else."
    default_editor_system_prompt = (
        "You are a caption editor. Use the image, draft caption, generated observations, "
        "glossary meanings, and authoritative counts as truth. Respond in English only. "
        "Return only the final revised caption, with no reasoning or preamble."
    )
    editor_system_prompt = (
        _collapse_whitespace_impl(payload.caption_editor_system_prompt or "")
        or default_editor_system_prompt
    )
    coverage_policy = (
        _collapse_whitespace_impl(payload.caption_coverage_prompt or "")
        or (
            "Apply the requested class coverage and count corrections while keeping the caption grounded in the image. "
            "Mention missing required object categories only when they are supported by the image or authoritative detection context. "
            "Preserve authoritative counts as natural scene facts using digits. Edit minimally unless the draft is mostly unusable. "
            "Do not mention labels, hints, boxes, coordinates, prompts, or that counts were supplied."
        )
    )
    language_rewrite_policy = (
        _collapse_whitespace_impl(payload.caption_language_rewrite_prompt or "")
        or (
            "Rewrite the caption in English only, preserving meaning, concrete visual detail, broad class terms, and authoritative counts. "
            "Do not add new objects or actions. Do not mention labels, hints, boxes, coordinates, prompts, or that counts were supplied."
        )
    )
    overlap: Optional[float] = None
    window_size: Optional[int] = None
    x_positions: List[int] = []
    y_positions: List[int] = []
    grouped_hints: Dict[Tuple[int, int], List[QwenCaptionHint]] = {}
    overlap_guidance = ""
    total_windows = 0
    if caption_mode == "windowed":
        overlap = _resolve_qwen_window_overlap_impl(
            payload.window_overlap, default_overlap=QWEN_WINDOW_DEFAULT_OVERLAP
        )
        window_size = _resolve_qwen_window_size_impl(
            payload.window_size,
            image_width,
            image_height,
            overlap=overlap,
            default_size=QWEN_WINDOW_DEFAULT_SIZE,
            default_overlap=QWEN_WINDOW_DEFAULT_OVERLAP,
        )
        x_positions = _window_positions_impl(image_width, window_size, overlap)
        y_positions = _window_positions_impl(image_height, window_size, overlap)
        grouped_hints = _group_hints_by_window(label_hints, x_positions, y_positions, window_size)
        overlap_guidance = _build_caption_overlap_guidance(
            label_hints,
            grouped_hints,
            image_width,
            image_height,
            labelmap_glossary=payload.labelmap_glossary,
        )
        total_windows = max(1, len(x_positions) * len(y_positions))
    caption_step_plan = _build_qwen_caption_step_plan(
        caption_mode=caption_mode,
        total_windows=total_windows,
        two_stage=two_stage,
        is_thinking=is_thinking,
        force_unload=False,
        caption_model_id=desired_model_id,
        refinement_model_id=caption_refinement_model_id,
        windowed_full_image_strategy=payload.caption_windowed_full_image_strategy or "visual",
    )

    def _format_authoritative_counts_note(
        counts_map: Dict[str, int],
        glossary: Optional[Dict[str, List[str]]],
    ) -> str:
        if not include_counts or not restrict_to_labels or not counts_map:
            return ""
        pieces: List[str] = []
        for label, count in counts_map.items():
            try:
                count_int = int(count)
            except (TypeError, ValueError):
                continue
            if count_int <= 0:
                continue
            preferred = _caption_preferred_label_impl(label, glossary).strip()
            if preferred:
                pieces.append(f"{preferred}: {count_int}")
        if not pieces:
            return ""
        return (
            "Authoritative object counts that must appear in the caption as natural scene facts: "
            + ", ".join(pieces)
            + ". State counts with digits; do not say counts were provided."
        )

    authoritative_counts_note = _format_authoritative_counts_note(counts, glossary_map)
    full_region = {
        "x": 0,
        "y": 0,
        "width": image_width,
        "height": image_height,
        "image_width": image_width,
        "image_height": image_height,
        "image_name": request_image_name,
        "kind": "full_image",
    }
    sections: List[QwenCaptionPromptPreviewSection] = []
    warnings: List[str] = []
    planned_windows: List[Tuple[int, int, int, str]] = []
    pending_window_merge_section: Optional[QwenCaptionPromptPreviewSection] = None
    skipped_windows = 0
    windowed_text_only_compose_preview = False
    if caption_mode == "windowed" and window_size is not None:
        window_index = 0
        window_is_thinking = "Thinking" in desired_model_id
        for y0 in y_positions:
            for x0 in x_positions:
                window_index += 1
                window_hints = grouped_hints.get((x0, y0), [])
                if not window_hints and not caption_all_windows:
                    skipped_windows += 1
                    continue
                window_allowed = _allowed_caption_labels_impl(window_hints)
                window_glossary_map = _caption_glossary_map_impl(
                    payload.labelmap_glossary,
                    [hint.label for hint in window_hints if hint.label],
                )
                window_allowed_prompt = (
                    [
                        _caption_preferred_label_impl(label, window_glossary_map)
                        for label in window_allowed
                    ]
                    if window_allowed
                    else []
                )
                window_prompt, _window_counts, _, _ = _build_qwen_caption_prompt_impl(
                    user_prompt,
                    window_hints,
                    window_size,
                    window_size,
                    include_counts,
                    include_coords,
                    max_boxes,
                    detailed_mode=True,
                    restrict_to_labels=restrict_to_labels,
                    labelmap_glossary=payload.labelmap_glossary,
                    max_sentences=final_caption_max_sentences,
                    context_prompt=_caption_window_detection_context_prompt(),
                )
                if payload.caption_window_prompt:
                    window_task_prompt = payload.caption_window_prompt.strip()
                else:
                    window_caption_instruction = _format_caption_window_sentence_instruction(payload)
                    window_task_prompt = (
                        "This crop is a subregion of the full image.\n"
                        "Apply the crop caption context above to this crop only.\n"
                        "Use the crop image as truth and treat the provided object context as priors, not as text to mention.\n"
                        "Describe crop-local objects of interest with natural-language class meanings, visible attributes, counts, "
                        "and spatial relations when available. Mention surrounding scene details only when they clarify those objects or the crop. "
                        "Do not add unsupported object categories or treat background context as extra counted inventory.\n"
                        f"{window_caption_instruction}No reasoning or preamble. "
                        "Do not mention labels, hints, bounding boxes, coordinates, glossary text, or that counts were provided, "
                        "but if counts are provided then do make sure to keep those counts in your output. "
                        "Do not output internal dataset class names or any token with underscores. "
                        "Use narrower subtypes only when the crop clearly supports them."
                    )
                if glossary_line:
                    window_prompt = f"{window_prompt}\n{glossary_line}"
                if restrict_to_labels and window_allowed_prompt:
                    window_prompt = (
                        f"{window_prompt}\nAllowed classes: {', '.join(window_allowed_prompt)}. "
                        "This window has labeled support boxes. Use the crop detection context to describe those labeled objects, "
                        "their visible attributes, and surrounding crop details that clarify them. "
                        "Do not promote unrelated background objects into extra counted categories, and do not introduce unsupported entity types."
                    )
                elif restrict_to_labels and label_hints and caption_all_windows:
                    window_prompt = (
                        f"{window_prompt}\nThis window has no labeled support boxes. "
                        "Use it only for broad scene context; do not enumerate object categories, counts, or new object lists."
                    )
                window_prompt = f"{window_prompt}\n\nCrop task:\n{window_task_prompt}"
                if window_is_thinking:
                    window_prompt = _adjust_prompt_for_thinking_impl(window_prompt)
                window_region = {
                    "x": x0,
                    "y": y0,
                    "width": min(window_size, max(1, image_width - x0)),
                    "height": min(window_size, max(1, image_height - y0)),
                    "image_width": image_width,
                    "image_height": image_height,
                    "image_name": request_image_name,
                    "kind": "window",
                    "label": f"Window {window_index}/{total_windows}",
                }
                sections.append(
                    _qwen_caption_prompt_preview_section(
                        title=f"Window {window_index}/{total_windows} caption prompt",
                        model_id=desired_model_id,
                        system_prompt=system_prompt,
                        user_prompt=window_prompt,
                        note="Exact prompt for this crop/window model call.",
                        image_region=window_region,
                    )
                )
                planned_windows.append((x0, y0, window_size, f"<generated caption from window {window_index}/{total_windows}>"))
        if skipped_windows:
            warnings.append(
                f"{skipped_windows} empty window(s) would be skipped because caption_all_windows is disabled."
            )
        windowed_text_only_compose_preview = (
            caption_mode == "windowed"
            and bool(planned_windows)
            and (payload.caption_windowed_full_image_strategy or "visual") == "text_only"
            and not (two_stage and is_thinking)
        )
        if planned_windows:
            warnings.append(
                "Text-only full-image composition prompts contain placeholders in this preview because their exact text depends on generated window captions."
                if windowed_text_only_compose_preview
                else "Window merge and final composition prompts contain placeholders in this preview because their exact text depends on generated window captions."
            )
            window_lines = _format_caption_window_observation_lines_impl(
                planned_windows,
                image_width=image_width,
                image_height=image_height,
                full_label_hints=label_hints,
                window_hints_by_window=grouped_hints,
                glossary_map=glossary_map,
            )
            window_lines.append(
                "Treat these observations as close-up source notes, not a separate object inventory."
            )
            if overlap_guidance:
                window_lines.append(overlap_guidance)
            if short_caption_requested:
                if include_counts and restrict_to_labels:
                    window_lines.append(
                        "Now write the full-image caption at the requested length. Use the close-up observations only for the most important visible details. "
                        "Mention only central labeled object types, and summarize repetitive objects instead of listing them. "
                        "Do not mention labels, hints, coordinates, or that counts were provided."
                    )
                    window_lines.append(
                        "If window observations conflict, trust the full image and the authoritative counts. Avoid self-contradictions."
                    )
                else:
                    window_lines.append(
                        "Now write the full-image caption at the requested length. Use the close-up observations only for the most important visible details. "
                        "Do not mention labels, hints, or coordinates."
                    )
                    window_lines.append(
                        "If window observations conflict, trust the full image. Avoid self-contradictions."
                    )
                window_lines.append(
                    "Do not expand beyond the requested caption length just because window observations are available."
                )
            elif include_counts and restrict_to_labels:
                window_lines.append(
                    "Now describe the full image in detail. Use all labeled object counts and the close-up observations. "
                    "Mention every class that appears in the hints, and summarize repetitive objects (e.g., many cars as a parking lot) "
                    "unless only a few are present or a specific action stands out. "
                    "Do not mention labels, hints, coordinates, or that counts were provided."
                )
                window_lines.append(
                    "If window observations conflict, trust the full image and the authoritative counts. Avoid self-contradictions."
                )
            else:
                window_lines.append(
                    "Now describe the full image in detail using the close-up observations. "
                    "Use detector hints as suggestions, and mention other visible objects. "
                    "Do not mention labels, hints, or coordinates."
                )
                window_lines.append(
                    "If window observations conflict, trust the full image. Avoid self-contradictions."
                )
            window_lines.append(
                (
                    "In the final caption, include only the most important window details that fit the requested length."
                    if short_caption_requested
                    else (
                        "In the final caption, preserve specific details about labeled or clearly central objects from the windows "
                        "(e.g., counts, actions, and notable attributes). "
                        "Do not promote unlabeled background observations into extra counted objects; longer captions are acceptable when needed."
                    )
                )
            )
            if broad_term_editor_note:
                window_lines.append(broad_term_editor_note)
            prompt_text = f"{prompt_text}\n" + "\n".join(window_lines)
            merge_detail_policy = (
                "Keep the revised caption concise. Fold in only the most important missing visible details from the windows. "
                if short_caption_requested
                else (
                    "Use multiple sentences if needed. Preserve specific counts, actions, and notable attributes from the windows; "
                    "treat window text as supporting evidence, not a complete object inventory. Ignore extra object lists "
                    "or quantities that conflict with the full image or authoritative counts. "
                )
            )
            merge_policy = (
                _collapse_whitespace_impl(payload.caption_merge_prompt or "")
                or (
                    "Revise the draft caption so it includes all distinct object details "
                    "from the window observations that are missing in the draft. "
                    "Do not invent new objects, and do not turn background window descriptions into extra counted object categories. "
                    f"{broad_term_editor_note} "
                    f"{merge_detail_policy}"
                    f"{final_caption_length_instruction}"
                    "Do not mention labels, hints, or coordinates."
                )
            )
            merge_prompt = (
                f"{caption_user_request_note}"
                f"{_collapse_whitespace_impl(glossary_line or '') + ' ' if glossary_line else ''}"
                f"{_collapse_whitespace_impl(authoritative_counts_note or '') + ' ' if authoritative_counts_note else ''}"
                f"{merge_policy}\n"
                "Draft caption: <generated full-image caption before window merge>\n"
                + "\n".join(window_lines)
            )
            merge_system = (
                _collapse_whitespace_impl(editor_system_prompt or "")
                or "You are a caption editor. Return only the revised caption in English."
            )
            if not windowed_text_only_compose_preview:
                pending_window_merge_section = _qwen_caption_prompt_preview_section(
                    title="Window merge prompt template",
                    kind="template",
                    model_id=caption_refinement_model_id,
                    system_prompt=merge_system,
                    user_prompt=merge_prompt,
                    note="Runs only after window captions and a full-image caption exist; placeholders stand in for generated text.",
                    image_region=full_region,
                )
    if two_stage and is_thinking:
        if payload.caption_draft_refine_prompt:
            draft_prompt = (
                f"{payload.caption_draft_refine_prompt.strip()}\n\n"
                f"{prompt_text}\n"
                "Respond with: DRAFT: <caption>"
            )
        else:
            draft_prompt = (
                "Step 1: Look at the image and form a draft caption.\n"
                f"{prompt_text}\n"
                "Respond with: DRAFT: <caption>"
            )
        draft_system = f"{system_prompt} Return only a line starting with 'DRAFT:'."
        sections.append(
            _qwen_caption_prompt_preview_section(
                title="Two-stage draft prompt",
                model_id=desired_model_id,
                system_prompt=draft_system,
                user_prompt=draft_prompt,
                note="Exact first pass prompt for Thinking-model draft mode.",
                image_region=full_region,
            )
        )
        allowed_note = ""
        if restrict_to_labels and allowed_labels_prompt:
            allowed_note = f"Allowed classes: {', '.join(allowed_labels_prompt)}. Do not introduce any other entity types."
        elif not restrict_to_labels:
            allowed_note = "You may mention additional visible objects beyond the hints."
        refine_prompt = f"{prompt_text}\nDraft caption: <generated draft caption>"
        if planned_windows:
            refine_prompt = (
                f"{refine_prompt}\n\n"
                "First-stage model output context for detail preservation. "
                "Use this only to recover concrete visible details; ignore reasoning, instructions, labels, hints, counts, or coordinates.\n"
                "<raw window and draft model outputs inserted here>"
            )
        if allowed_note:
            refine_prompt = f"{refine_prompt}\n{allowed_note}"
        if payload.caption_draft_refine_prompt:
            refine_prompt = (
                f"{refine_prompt}\n"
                f"{payload.caption_draft_refine_prompt.strip()} "
                f"{final_caption_length_instruction}"
                "Return only the final caption."
            )
        else:
            refine_prompt = (
                f"{refine_prompt}\n"
                "Edit the draft with minimal changes. Do not introduce new objects or actions. "
                f"{broad_term_editor_note} "
                f"{final_caption_length_instruction}"
                "Return only the final caption."
            )
        refine_system = editor_system_prompt
        sections.append(
            _qwen_caption_prompt_preview_section(
                title="Two-stage refinement prompt template",
                kind="template",
                model_id=caption_refinement_model_id,
                system_prompt=refine_system,
                user_prompt=refine_prompt,
                note="Runs after the draft pass; generated draft text is represented by a placeholder.",
                image_region=full_region,
            )
        )
        warnings.append(
            "Two-stage refinement prompt includes a generated draft placeholder because preview does not run Qwen."
        )
    else:
        full_prompt_note = "Exact full-image model prompt."
        full_prompt_model_id = desired_model_id
        full_prompt_system = system_prompt
        full_prompt_title = "Full-image caption prompt"
        if windowed_text_only_compose_preview:
            prompt_text = (
                f"{prompt_text}\n"
                "Text-only full-image composition instruction: compose the final full-image caption "
                "from the completed window observations, authoritative full-image counts, class inventory, "
                "and user request above. Treat the observations as visual evidence for the full image, "
                "deduplicate repeated crop-edge objects, and do not mention labels, hints, boxes, coordinates, "
                "or prompts. Return only the final caption."
            )
            full_prompt_note = (
                "Text-only full-image composition prompt; generated window observations are represented "
                "as placeholders and no second full-image tensor is sent."
            )
            full_prompt_model_id = caption_refinement_model_id
            full_prompt_system = editor_system_prompt
            full_prompt_title = "Text-only full-image composition prompt"
        if planned_windows:
            full_prompt_note = (
                "Exact full-image model prompt; generated window observations are "
                "represented as placeholders in this preview."
            )
            if windowed_text_only_compose_preview:
                full_prompt_note = (
                    "Text-only full-image composition prompt; generated window observations are represented "
                    "as placeholders and no second full-image tensor is sent."
                )
        sections.append(
            _qwen_caption_prompt_preview_section(
                title=full_prompt_title,
                model_id=full_prompt_model_id,
                system_prompt=full_prompt_system,
                user_prompt=prompt_text,
                note=full_prompt_note,
                image_region=full_region,
            )
        )
    if pending_window_merge_section is not None:
        sections.append(pending_window_merge_section)
    cleanup_policy = (
        _collapse_whitespace_impl(payload.caption_cleanup_prompt or "")
        or (
            "Remove repetition, avoid coordinates, and remove any mention of labels, hints, or counts being provided. "
            "Never copy planning phrases such as 'we can mention', 'we need to', or 'the user wants'. "
            f"{broad_term_editor_note} "
            "If the draft is mostly reasoning or planning text, ignore that draft and write a fresh image-grounded caption. "
            "Keep the caption grounded in the image."
        )
    )
    cleanup_allowed_note = ""
    if restrict_to_labels and allowed_labels_prompt:
        cleanup_allowed_note = (
            f"Only mention these classes if they appear: {', '.join(sorted(set(allowed_labels_prompt)))}. "
            "Do not introduce any other entity types. "
        )
    cleanup_minimal_note = "Edit the draft with minimal changes. Do not introduce new objects or actions. "
    cleanup_prompt = (
        f"{final_caption_length_instruction}"
        f"{cleanup_allowed_note}"
        f"{cleanup_minimal_note}"
        f"{caption_user_request_note}"
        f"{_collapse_whitespace_impl(glossary_line or '') + ' ' if glossary_line else ''}"
        f"{_collapse_whitespace_impl(authoritative_counts_note or '') + ' ' if authoritative_counts_note else ''}"
        f"{cleanup_policy}\n"
        "Draft caption: <caption that triggered cleanup, coverage, count, repetition, or language guards>\n\n"
        "First-stage model output context:\n"
        "<raw first-stage model output inserted here when available>"
    )
    sections.append(
        _qwen_caption_prompt_preview_section(
            title="Conditional cleanup / guard prompt template",
            kind="template",
            model_id=caption_refinement_model_id,
            system_prompt=editor_system_prompt,
            user_prompt=cleanup_prompt,
            note="Runs only if caption quality guards fire; generated caption text is represented by a placeholder.",
            image_region=full_region,
        )
    )
    coverage_prompt = (
        f"{prompt_text}\n"
        "Draft caption: <caption with missing classes or count conflicts>\n"
        "Guard findings: <missing classes and/or count corrections inserted here>\n"
        f"{coverage_policy}\n"
        f"{cleanup_policy}\n"
        f"{final_caption_length_instruction}"
        "Return only the final caption with no coordinates."
    )
    sections.append(
        _qwen_caption_prompt_preview_section(
            title="Conditional coverage/count guard prompt template",
            kind="template",
            model_id=caption_refinement_model_id,
            system_prompt=editor_system_prompt,
            user_prompt=coverage_prompt,
            note="Runs only when class coverage or exact-count guards find missing or contradictory caption content.",
            image_region=full_region,
        )
    )
    rewrite_prompt = (
        f"{language_rewrite_policy}\n"
        f"{cleanup_policy}\n"
        f"{final_caption_length_instruction}\n"
        "Caption: <caption that triggered the English-language guard>"
    )
    sections.append(
        _qwen_caption_prompt_preview_section(
            title="Conditional English rewrite prompt template",
            kind="template",
            model_id=caption_refinement_model_id,
            system_prompt=editor_system_prompt,
            user_prompt=rewrite_prompt,
            note="Runs only when the language guard detects a non-English or mixed-language caption.",
            image_region=full_region,
        )
    )
    warnings.append(
        "Cleanup, coverage, count, repetition, and language guard prompts run only if their checks fire after generation."
    )
    meta = {
        "image_name": request_image_name,
        "image_width": image_width,
        "image_height": image_height,
        "caption_mode": caption_mode,
        "caption_windowed_full_image_strategy": payload.caption_windowed_full_image_strategy or "visual",
        "model_id": desired_model_id,
        "refinement_model_id": caption_refinement_model_id,
        "two_stage_refine": two_stage,
        "is_thinking": is_thinking,
        "planned_windows": len(planned_windows),
        "skipped_windows": skipped_windows,
        "caption_all_windows": caption_all_windows,
        "include_counts": include_counts,
        "include_coords": include_coords,
        "max_boxes": max_boxes,
        "caption_window_min_sentences": payload.caption_window_min_sentences,
        "caption_window_max_sentences": payload.caption_window_max_sentences,
    }
    prompt_budget = _qwen_caption_prompt_preview_budget(
        sections,
        requested_max_new_tokens=payload.max_new_tokens,
        variant=variant,
        caption_mode=caption_mode,
        final_caption_max_sentences=final_caption_max_sentences,
        explicit_max_new_tokens=payload.max_new_tokens is not None,
    )
    meta["prompt_budget"] = prompt_budget
    if truncated:
        warnings.append(
            f"Max boxes is Auto/representative for this image: {used_boxes} of {len(label_hints)} boxes are shown in prompt box lists; counts still cover all label hints."
        )
    if prompt_budget.get("adapted_sections") and payload.max_new_tokens is None:
        warnings.append(
            "Auto output tokens will be reduced for large rendered prompt sections. Enter Max output tokens only when you want a hard override."
        )
    full_text = _format_qwen_caption_prompt_preview_text(sections, warnings=warnings, meta=meta)
    return QwenCaptionPromptPreviewResponse(
        sections=sections,
        full_text=full_text,
        used_counts=counts,
        used_boxes=used_boxes,
        truncated=truncated,
        step_plan=caption_step_plan,
        warnings=warnings,
        meta=meta,
    )


def qwen_caption(payload: QwenCaptionRequest):
    request_image_name = str(getattr(payload, "image_name", "") or "").strip()
    fast_mode = bool(payload.fast_mode)
    force_unload = payload.force_unload
    if force_unload is None:
        force_unload = QWEN_CAPTION_CACHE_LIMIT == 0
    if fast_mode:
        force_unload = False
    multi_model_cache = bool(payload.multi_model_cache or fast_mode)
    active_model_id: Optional[str] = None
    active_runtime: Optional[Tuple[Any, Any]] = None
    request_model_cache: Dict[str, Tuple[Any, Any]] = {}
    default_caption_model_id: Optional[str] = None
    base_model_hint = (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME
    requested_model_hint = str(payload.model_id or "").strip()
    if requested_model_hint.lower() == "active":
        requested_model_hint = ""
    requested_variant_hint = payload.model_variant or "auto"
    caption_model_hint = (
        requested_model_hint
        or _resolve_qwen_caption_default_model_id(base_model_hint, requested_variant_hint)
    )
    caption_platform_hint = _resolve_qwen_runtime_platform(str(caption_model_hint), metadata=active_qwen_metadata)
    initial_max_new_tokens = _resolve_qwen_caption_max_new_tokens(
        payload.max_new_tokens,
        model_id=caption_model_hint,
        variant=requested_variant_hint,
        caption_mode=payload.caption_mode,
        final_caption_max_sentences=payload.final_caption_max_sentences,
    )
    _qwen_progress_start(
        kind="caption",
        model_id=str(caption_model_hint),
        platform=caption_platform_hint,
        message="Preparing Qwen caption request",
        max_new_tokens=initial_max_new_tokens,
    )

    def get_runtime(model_id: Optional[str]) -> Tuple[Any, Any]:
        nonlocal active_model_id, active_runtime
        resolved_default = (
            default_caption_model_id
            or (active_qwen_metadata or {}).get("model_id")
            or QWEN_MODEL_NAME
        )
        if multi_model_cache:
            key = model_id or "__active__"
            cached = request_model_cache.get(key)
            if cached:
                return cached
            if model_id:
                runtime = _ensure_qwen_ready_for_caption(model_id)
            else:
                runtime = _ensure_qwen_ready_for_caption(resolved_default)
            request_model_cache[key] = runtime
            return runtime
        if active_runtime is not None and active_model_id != model_id:
            logger.info(
                "[qwen-caption] switching model %s -> %s; unloading current runtime",
                active_model_id,
                model_id,
            )
            _unload_qwen_runtime()
            active_runtime = None
            active_model_id = None
        if active_runtime is None:
            if model_id:
                active_runtime = _ensure_qwen_ready_for_caption(model_id)
                active_model_id = model_id
            else:
                active_runtime = _ensure_qwen_ready_for_caption(resolved_default)
                active_model_id = None
        return active_runtime

    try:
        _raise_if_qwen_cancelled()
        if payload.unload_others and not fast_mode:
            _qwen_progress_update(
                phase="unload",
                phase_label="Freeing Memory",
                progress=0.04,
                message="Unloading other inference models before captioning",
                step_id="prepare",
            )
            _unload_non_qwen_runtimes_impl(
                predictor_manager=predictor_manager,
                unload_sam3_text_fn=_unload_sam3_text_runtime,
                suspend_clip_fn=_suspend_clip_backbone,
                unload_dinov3_fn=_unload_dinov3_backbone,
                unload_detector_fn=_unload_detector_inference,
                torch_module=torch,
                logger=logger,
            )
        _qwen_progress_update(
            phase="prepare",
            phase_label="Preparing Image",
            progress=0.06,
            message="Preparing image and caption hints",
            step_id="prepare",
            live_output_reset=True,
        )
        _raise_if_qwen_cancelled()
        pil_img, _, _ = resolve_image_payload(payload.image_base64, payload.image_token, None)
        user_prompt = (payload.user_prompt or "").strip()
        include_counts = bool(payload.include_counts)
        include_coords = bool(payload.include_coords)
        max_boxes = payload.max_boxes if payload.max_boxes is not None else 0
        explicit_max_new_tokens = payload.max_new_tokens is not None
        with qwen_progress_lock:
            qwen_progress_state["caption_explicit_max_new_tokens"] = explicit_max_new_tokens
        max_new_tokens = payload.max_new_tokens
        source_image_width = payload.image_width
        source_image_height = payload.image_height
        image_width = pil_img.width
        image_height = pil_img.height
        label_hints = _normalize_caption_hints_for_decoded_image(
            payload.label_hints or [],
            source_image_width,
            source_image_height,
            image_width,
            image_height,
        )
        allowed_labels = _allowed_caption_labels_impl(label_hints)
        caption_mode = payload.caption_mode or "full"
        windowed_full_image_strategy = payload.caption_windowed_full_image_strategy or "visual"
        restrict_to_labels = (
            payload.restrict_to_labels if payload.restrict_to_labels is not None else True
        )
        final_caption_max_sentences = payload.final_caption_max_sentences
        caption_all_windows = _resolve_caption_all_windows_impl(
            caption_mode,
            payload.caption_all_windows,
            has_label_hints=bool(label_hints),
            restrict_to_labels=bool(restrict_to_labels),
        )
        detailed_mode = caption_mode == "windowed"
        glossary_map = _caption_glossary_map_impl(
            payload.labelmap_glossary,
            [hint.label for hint in label_hints if hint.label],
        )
        allowed_labels_prompt = (
            [_caption_preferred_label_impl(label, glossary_map) for label in allowed_labels]
            if allowed_labels
            else []
        )
        prompt_text, counts, used_boxes, truncated = _build_qwen_caption_prompt_impl(
            user_prompt,
            label_hints,
            image_width,
            image_height,
            include_counts,
            include_coords,
            max_boxes,
            detailed_mode,
            restrict_to_labels=restrict_to_labels,
            labelmap_glossary=payload.labelmap_glossary,
            max_sentences=final_caption_max_sentences,
            context_prompt=payload.caption_detection_context_prompt,
        )
        glossary_line = ""
        if glossary_map:
            glossary_line = (
                "Use each glossary broad term as the canonical name for that labeled class. "
                "If no glossary term exists, use a natural English class term instead of the raw label name. "
                "Glossary variants are possible narrower members only when clearly visible. "
                "Never output labelmap class names, especially tokens with underscores or odd internal spellings."
            )
            prompt_text = f"{prompt_text}\n{glossary_line}"
        base_model_id = (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME
        variant = payload.model_variant or "auto"
        model_id_override = str(payload.model_id or "").strip()
        if model_id_override.lower() == "active":
            model_id_override = ""
        if model_id_override:
            desired_model_id = model_id_override
            default_caption_model_id = base_model_id
        else:
            desired_model_id = _resolve_qwen_caption_default_model_id(base_model_id, variant)
            default_caption_model_id = desired_model_id
        if active_qwen_model_path and desired_model_id != base_model_id:
            logger.info(
                "[qwen-caption] using base model override (%s) while adapter %s is active",
                desired_model_id,
                active_qwen_model_id,
            )
        caption_base_model_id = desired_model_id
        caption_refinement_model_id = _resolve_qwen_caption_refinement_model_id(
            payload.refinement_model_id,
            desired_model_id=desired_model_id,
            active_model_id=base_model_id,
        )
        caption_loop_recovery_mode = _normalize_qwen_caption_loop_recovery_mode(
            payload.caption_loop_recovery_mode
        )
        caption_recovery_events: List[Dict[str, Any]] = []
        caption_loop_cooldown = bool(payload.caption_loop_cooldown)
        caption_fallback_model_id = str(payload.caption_fallback_model_id or "").strip()
        if caption_fallback_model_id.lower() in {"", "auto", "none", "active"}:
            caption_fallback_model_id = ""
        if not caption_fallback_model_id and caption_loop_recovery_mode == "safe_retry_fallback":
            caption_fallback_model_id = _qwen_caption_default_refinement_model_id(desired_model_id)
        if caption_fallback_model_id == desired_model_id:
            caption_fallback_model_id = ""
        final_only = bool(payload.final_answer_only)
        final_caption_length_instruction = _caption_length_instruction_impl(final_caption_max_sentences)
        caption_user_request_note = _caption_user_request_instruction_impl(user_prompt)
        broad_term_editor_note = _caption_editor_preserve_broad_terms_instruction_impl()
        short_caption_requested = _caption_user_requested_short_impl(
            user_prompt,
            final_caption_max_sentences,
        )
        two_stage = bool(payload.two_stage_refine)
        is_thinking = (
            _qwen_model_id_is_thinking(desired_model_id)
            or str(variant or "").lower() == "thinking"
        )
        max_new_tokens = _resolve_qwen_caption_max_new_tokens(
            max_new_tokens,
            model_id=desired_model_id,
            variant=variant,
            caption_mode=caption_mode,
            final_caption_max_sentences=final_caption_max_sentences,
            label_hint_count=len(label_hints),
        )
        caption_chat_template_kwargs = (
            {"enable_thinking": False} if final_only and is_thinking else None
        )
        postprocess_chat_template_kwargs = {"enable_thinking": False}
        decode_params = _resolve_qwen_caption_decode_impl(payload, is_thinking)
        deterministic_decode = {
            "do_sample": False,
            "repetition_penalty": 1.08,
            "repetition_context_size": 128,
            "no_repeat_ngram_size": 8,
        }
        postprocess_decode_params = (
            _resolve_qwen_caption_decode_impl(payload, is_thinking=True)
            if _qwen_model_id_is_thinking(caption_refinement_model_id)
            else deterministic_decode
        )
        if is_thinking:
            prompt_text = _adjust_prompt_for_thinking_impl(prompt_text)
        if final_caption_length_instruction and "final caption length:" not in prompt_text.lower():
            prompt_text = f"{prompt_text}\nFinal caption length: {final_caption_length_instruction}"
            if not explicit_max_new_tokens:
                max_new_tokens = max(
                    max_new_tokens,
                    min(QWEN_CAPTION_MAX_NEW_TOKENS, max(1000, int(final_caption_max_sentences or 10) * 100)),
                )
        # Keep caption max_new_tokens consistent across full/windowed/refine paths.
        # Avoid per-path caps here (we previously caused repeated CUDA asserts by diverging).
        if is_thinking and not explicit_max_new_tokens:
            max_new_tokens = max(max_new_tokens, 1000)
        max_new_tokens = min(max_new_tokens, QWEN_CAPTION_MAX_NEW_TOKENS)
        refine_max_tokens = max_new_tokens
        default_system_prompt = (
            "You are a detailed captioning assistant. Use the image as truth. "
            "Provide a rich, multi-sentence caption when there is a lot to see. "
            "Do not mention labels, hints, bounding boxes, coordinates, glossary text, or that counts were provided. "
            "Do not output internal dataset class names or any token with underscores. "
            "Use natural-language class terms from the provided context when available. "
            "Use narrower subtypes only when the image clearly supports them. "
            "If the hints conflict with the image, mention the uncertainty briefly. "
            "Stop when the caption is complete; never repeat a sentence or keep writing to fill the token budget.\n"
            "Respond in English only.\n"
            "Return only the final caption. Do not include reasoning or preamble."
        )
        custom_system_prompt = bool((payload.caption_system_prompt or "").strip())
        system_prompt = (payload.caption_system_prompt or "").strip() or default_system_prompt
        if not custom_system_prompt:
            if final_only and is_thinking and not two_stage:
                system_prompt = (
                    f"{system_prompt} Respond with exactly <final>...</final> and nothing else."
                )
        default_editor_system_prompt = (
            "You are a caption editor. Use the image, draft caption, generated observations, "
            "glossary meanings, and authoritative counts as truth. Respond in English only. "
            "Return only the final revised caption, with no reasoning or preamble."
        )
        editor_system_prompt = (
            _collapse_whitespace_impl(payload.caption_editor_system_prompt or "")
            or default_editor_system_prompt
        )
        coverage_policy = (
            _collapse_whitespace_impl(payload.caption_coverage_prompt or "")
            or (
                "Apply the requested class coverage and count corrections while keeping the caption grounded in the image. "
                "Mention missing required object categories only when they are supported by the image or authoritative detection context. "
                "Preserve authoritative counts as natural scene facts using digits. Edit minimally unless the draft is mostly unusable. "
                "Do not mention labels, hints, boxes, coordinates, prompts, or that counts were supplied."
            )
        )
        language_rewrite_policy = (
            _collapse_whitespace_impl(payload.caption_language_rewrite_prompt or "")
            or (
                "Rewrite the caption in English only, preserving meaning, concrete visual detail, broad class terms, and authoritative counts. "
                "Do not add new objects or actions. Do not mention labels, hints, boxes, coordinates, prompts, or that counts were supplied."
            )
        )
        use_caption_cache = True
        if (
            active_qwen_model_path
            and not model_id_override
            and desired_model_id == base_model_id
            and variant == "auto"
        ):
            use_caption_cache = False
        overlap: Optional[float] = None
        window_size: Optional[int] = None
        x_positions: List[int] = []
        y_positions: List[int] = []
        grouped_hints: Dict[Tuple[int, int], List[QwenCaptionHint]] = {}
        overlap_guidance = ""
        total_windows = 0
        if caption_mode == "windowed":
            overlap = _resolve_qwen_window_overlap_impl(
                payload.window_overlap, default_overlap=QWEN_WINDOW_DEFAULT_OVERLAP
            )
            window_size = _resolve_qwen_window_size_impl(
                payload.window_size,
                image_width,
                image_height,
                overlap=overlap,
                default_size=QWEN_WINDOW_DEFAULT_SIZE,
                default_overlap=QWEN_WINDOW_DEFAULT_OVERLAP,
            )
            x_positions = _window_positions_impl(
                image_width, window_size, overlap
            )
            y_positions = _window_positions_impl(
                image_height, window_size, overlap
            )
            grouped_hints = _group_hints_by_window(
                label_hints, x_positions, y_positions, window_size
            )
            overlap_guidance = _build_caption_overlap_guidance(
                label_hints,
                grouped_hints,
                image_width,
                image_height,
                labelmap_glossary=payload.labelmap_glossary,
            )
            total_windows = max(1, len(x_positions) * len(y_positions))
            logger.info(
                "[qwen-caption] window grid=%sx%s total=%s window=%s overlap=%.3f image=%sx%s",
                len(x_positions),
                len(y_positions),
                total_windows,
                window_size,
                overlap or 0.0,
                image_width,
                image_height,
            )
            if (
                is_thinking
                and (total_windows > 1 or len(label_hints) > 40)
                and not explicit_max_new_tokens
            ):
                max_new_tokens = min(QWEN_CAPTION_MAX_NEW_TOKENS, max(max_new_tokens, 3000))
                refine_max_tokens = max_new_tokens
        _qwen_progress_update(max_new_tokens=max_new_tokens)
        caption_step_plan = _build_qwen_caption_step_plan(
            caption_mode=caption_mode,
            total_windows=total_windows,
            two_stage=two_stage,
            is_thinking=is_thinking,
            force_unload=bool(force_unload),
            caption_model_id=desired_model_id,
            refinement_model_id=caption_refinement_model_id,
            windowed_full_image_strategy=payload.caption_windowed_full_image_strategy or "visual",
        )
        full_step_region = {
            "x": 0,
            "y": 0,
            "width": image_width,
            "height": image_height,
            "image_width": image_width,
            "image_height": image_height,
            "image_name": request_image_name,
            "kind": "full_image",
            "full_image": True,
        }
        _qwen_progress_update(
            phase="plan",
            phase_label="Planning",
            progress=0.08,
            message=(
                f"Caption workflow planned: {len(caption_step_plan)} steps"
                + (f", {total_windows} window pass{'es' if total_windows != 1 else ''}" if caption_mode == "windowed" else "")
            ),
            step_plan=caption_step_plan,
            step_id="prepare",
            step_region=full_step_region,
            live_output_reset=True,
        )

        def _caption_cleanup(
            prompt: str,
            img: Image.Image,
            tokens: int,
            base_id: str,
            cache_ok: bool,
            *,
            model_id_override: Optional[str] = None,
            runtime_override: Optional[Tuple[Any, Any]] = None,
            allowed_labels: Optional[List[str]] = None,
            strict: bool = False,
            minimal_edit: bool = False,
            max_sentences: Optional[int] = None,
            source_output: Optional[str] = None,
            source_outputs: Optional[Sequence[Tuple[str, str]]] = None,
            authoritative_counts_note_override: Optional[str] = None,
        ) -> str:
            counts_note = (
                authoritative_counts_note
                if authoritative_counts_note_override is None
                else authoritative_counts_note_override
            )
            return _run_qwen_caption_cleanup_impl(
                prompt,
                img,
                tokens,
                base_id,
                cache_ok,
                model_id_override=model_id_override,
                runtime_override=runtime_override,
                allowed_labels=allowed_labels,
                strict=strict,
                minimal_edit=minimal_edit,
                max_sentences=max_sentences,
                user_prompt=user_prompt,
                source_output=source_output,
                source_outputs=source_outputs,
                authoritative_counts_note=counts_note,
                glossary_line=glossary_line or None,
                cleanup_prompt_override=payload.caption_cleanup_prompt,
                cleanup_system_prompt_override=editor_system_prompt,
                chat_template_kwargs=postprocess_chat_template_kwargs,
                run_qwen_inference_fn=_caption_visual_inference,
                run_qwen_text_inference_fn=_caption_text_inference,
                resolve_variant_fn=_resolve_qwen_variant_model_id_impl,
                extract_caption_fn=_extract_caption_from_text_impl,
                sanitize_caption_fn=_sanitize_qwen_caption_impl,
            )

        def _caption_merge(
            draft: str,
            windows: Sequence[Tuple[int, int, int, str]],
            *,
            pil_img: Image.Image,
            base_model_id: str,
            runtime_resolver: Callable[[str], Tuple[Any, Any]],
            max_new_tokens: int,
            glossary_line: Optional[str] = None,
            model_id_override: Optional[str] = None,
            max_sentences: Optional[int] = None,
            overlap_guidance: Optional[str] = None,
            source_outputs: Optional[Sequence[Tuple[str, str]]] = None,
            full_label_hints: Optional[Sequence[QwenCaptionHint]] = None,
            window_hints_by_window: Optional[Mapping[Tuple[int, int], Sequence[QwenCaptionHint]]] = None,
            glossary_map: Optional[Dict[str, List[str]]] = None,
        ) -> str:
            return _run_qwen_caption_merge_impl(
                draft,
                windows,
                pil_img=pil_img,
                base_model_id=base_model_id,
                runtime_resolver=runtime_resolver,
                max_new_tokens=max_new_tokens,
                glossary_line=glossary_line,
                model_id_override=model_id_override,
                max_sentences=max_sentences,
                user_prompt=user_prompt,
                overlap_guidance=overlap_guidance,
                source_outputs=source_outputs,
                authoritative_counts_note=authoritative_counts_note,
                full_label_hints=full_label_hints,
                window_hints_by_window=window_hints_by_window,
                glossary_map=glossary_map,
                merge_prompt_override=payload.caption_merge_prompt,
                merge_system_prompt_override=editor_system_prompt,
                chat_template_kwargs=postprocess_chat_template_kwargs,
                run_qwen_inference_fn=_caption_visual_inference,
                run_qwen_text_inference_fn=_caption_text_inference,
                resolve_variant_fn=_resolve_qwen_variant_model_id_impl,
                extract_caption_fn=_extract_caption_from_text_impl,
                sanitize_caption_fn=_sanitize_qwen_caption_impl,
            )

        def _accept_guard_caption(candidate: str, fallback: str, *, stage: str) -> str:
            cleaned = _sanitize_qwen_caption_impl(candidate)
            reason = _caption_degenerate_reason_impl(cleaned, allow_short_caption=True)
            if not reason:
                return cleaned
            preserved = _sanitize_qwen_caption_impl(fallback)
            _qwen_caption_io_record(
                {
                    "event": "guard_output_rejected",
                    "stage": stage,
                    "reason": reason,
                    "rejected_output": cleaned,
                    "preserved_caption": preserved,
                }
            )
            _qwen_progress_update(
                phase="cleanup",
                phase_label="Postprocessing",
                message=f"Rejected degenerate {stage} output; preserving previous caption",
                token_preview=preserved,
            )
            return preserved

        def resolve_main_runtime() -> Tuple[Any, Any]:
            if model_id_override:
                return get_runtime(desired_model_id)
            if use_caption_cache:
                return get_runtime(desired_model_id)
            return get_runtime(None)

        def resolve_refinement_runtime() -> Tuple[Any, Any]:
            if str(caption_refinement_model_id or "") != str(desired_model_id or ""):
                return get_runtime(caption_refinement_model_id)
            return resolve_main_runtime()

        def _caption_current_stage() -> Tuple[str, str]:
            with qwen_progress_lock:
                step_id = str(qwen_progress_state.get("step_id") or "caption_stage")
                step_label = str(qwen_progress_state.get("step_label") or step_id)
            return step_id, step_label

        def _caption_mark_loop_cooldown(key: str) -> None:
            if not key or not caption_loop_cooldown:
                return
            qwen_caption_loop_cooldowns[key] = time.time() + max(
                1,
                int(QWEN_CAPTION_LOOP_COOLDOWN_SECONDS or 0),
            )

        def _caption_unload_after_loop() -> None:
            nonlocal active_runtime, active_model_id
            request_model_cache.clear()
            _unload_qwen_runtime()
            active_runtime = None
            active_model_id = None

        def _caption_record_recovery_event(
            *,
            stage_id: str,
            stage_label: str,
            action: str,
            model_id: str,
            call_kind: str,
            message: str,
            attempt: str,
            detail: Optional[str] = None,
        ) -> None:
            event = {
                "stage_id": stage_id,
                "stage_label": stage_label,
                "action": action,
                "attempt": attempt,
                "model_id": model_id,
                "call_kind": call_kind,
                "message": message,
                "detail": detail,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
            caption_recovery_events.append(event)
            _qwen_caption_io_record({"event": "recovery", **event})
            _qwen_progress_update(
                phase="recovery",
                phase_label="Recovering",
                message=message,
                model_id=model_id,
                step_id=stage_id,
                step_detail=detail or message,
                live_output_piece=f"\n[recovery] {message}\n",
            )

        def _caption_effective_model_id(
            *,
            model_id_override: Optional[str],
            runtime_override: Optional[Tuple[Any, Any]],
            call_kind: str,
        ) -> str:
            return (
                str(model_id_override or "").strip()
                or str(_qwen_runtime_model_id(runtime_override) or "").strip()
                or (caption_refinement_model_id if call_kind == "text" else desired_model_id)
            )

        def _caption_run_call(
            *,
            call_kind: str,
            prompt: str,
            img: Optional[Image.Image],
            max_new_tokens: Optional[int],
            system_prompt_override: Optional[str],
            model_id_override: Optional[str],
            runtime_override: Optional[Tuple[Any, Any]],
            decode_override: Optional[Dict[str, Any]],
            chat_template_kwargs: Optional[Dict[str, Any]],
        ) -> Tuple[str, int, int]:
            if call_kind == "visual":
                if img is None:
                    raise RuntimeError("qwen_caption_visual_image_missing")
                return _run_qwen_inference(
                    prompt,
                    img,
                    max_new_tokens=max_new_tokens,
                    system_prompt_override=system_prompt_override,
                    model_id_override=model_id_override,
                    runtime_override=runtime_override,
                    decode_override=decode_override,
                    chat_template_kwargs=chat_template_kwargs,
                )
            return _run_qwen_text_inference(
                prompt,
                max_new_tokens=max_new_tokens,
                system_prompt_override=system_prompt_override,
                model_id_override=model_id_override,
                runtime_override=runtime_override,
                decode_override=decode_override,
                chat_template_kwargs=chat_template_kwargs,
            )

        def _caption_run_with_loop_recovery(
            *,
            call_kind: str,
            prompt: str,
            img: Optional[Image.Image] = None,
            max_new_tokens: Optional[int] = None,
            system_prompt_override: Optional[str] = None,
            model_id_override: Optional[str] = None,
            runtime_override: Optional[Tuple[Any, Any]] = None,
            decode_override: Optional[Dict[str, Any]] = None,
            chat_template_kwargs: Optional[Dict[str, Any]] = None,
            text_recovery_prompt: Optional[str] = None,
            text_recovery_system_prompt: Optional[str] = None,
            text_recovery_model_id: Optional[str] = None,
            text_recovery_runtime_resolver: Optional[Callable[[], Tuple[Any, Any]]] = None,
            text_recovery_decode_override: Optional[Dict[str, Any]] = None,
            text_recovery_chat_template_kwargs: Optional[Dict[str, Any]] = None,
            text_recovery_max_new_tokens: Optional[int] = None,
            text_recovery_detail: Optional[str] = None,
            text_recovery_after_safe_retry: bool = False,
            deterministic_recovery_text: Optional[str] = None,
            deterministic_recovery_detail: Optional[str] = None,
        ) -> Tuple[str, int, int]:
            stage_id, stage_label = _caption_current_stage()
            primary_model_id = _caption_effective_model_id(
                model_id_override=model_id_override,
                runtime_override=runtime_override,
                call_kind=call_kind,
            )
            primary_key = _qwen_caption_recovery_key(
                stage_id=stage_id,
                stage_label=stage_label,
                call_kind=call_kind,
                model_id=primary_model_id,
                prompt=prompt,
                system_prompt=system_prompt_override,
                decode_override=decode_override,
            )
            safe_decode = _qwen_caption_safe_retry_decode(
                decode_override,
                is_thinking=_qwen_model_id_is_thinking(primary_model_id),
            )
            safe_tokens = _qwen_caption_safe_retry_tokens(max_new_tokens)
            safe_key = _qwen_caption_recovery_key(
                stage_id=stage_id,
                stage_label=stage_label,
                call_kind=call_kind,
                model_id=primary_model_id,
                prompt=prompt,
                system_prompt=system_prompt_override,
                decode_override=safe_decode,
            )
            allow_safe_retry = caption_loop_recovery_mode in {
                "safe_retry",
                "safe_retry_fallback",
            }
            allow_fallback = caption_loop_recovery_mode == "safe_retry_fallback"
            text_recovery_prompt_clean = str(text_recovery_prompt or "").strip()
            text_recovery_available = (
                call_kind == "visual"
                and allow_fallback
                and bool(text_recovery_prompt_clean)
            )
            last_recovery_exc: Optional[BaseException] = None

            def _try_text_recovery() -> Optional[Tuple[str, int, int]]:
                nonlocal last_recovery_exc
                if not text_recovery_available:
                    return None
                recovery_model_id = str(
                    text_recovery_model_id or caption_refinement_model_id or primary_model_id
                ).strip()
                recovery_decode = _qwen_caption_safe_retry_decode(
                    text_recovery_decode_override,
                    is_thinking=_qwen_model_id_is_thinking(recovery_model_id),
                )
                recovery_key = _qwen_caption_recovery_key(
                    stage_id=stage_id,
                    stage_label=stage_label,
                    call_kind="text",
                    model_id=recovery_model_id,
                    prompt=text_recovery_prompt_clean,
                    system_prompt=text_recovery_system_prompt,
                    decode_override=recovery_decode,
                )
                recovery_cooldown_remaining = (
                    _qwen_caption_loop_cooldown_remaining(recovery_key)
                    if caption_loop_cooldown
                    else 0.0
                )
                if recovery_cooldown_remaining > 0:
                    _caption_record_recovery_event(
                        stage_id=stage_id,
                        stage_label=stage_label,
                        action="cooldown_skip",
                        attempt="text_recovery",
                        model_id=recovery_model_id,
                        call_kind="text",
                        message=(
                            f"Skipping text-only recovery for {stage_label}; cooldown has "
                            f"{int(recovery_cooldown_remaining)}s left."
                        ),
                        detail="Text-only recovery looped recently for this exact stage.",
                    )
                    return None
                recovery_source_label = (
                    "text-only prompt context"
                    if text_recovery_after_safe_retry
                    else "text-only window evidence"
                )
                _caption_record_recovery_event(
                    stage_id=stage_id,
                    stage_label=stage_label,
                    action="text_recovery_start",
                    attempt="text_recovery",
                    model_id=recovery_model_id,
                    call_kind="text",
                    message=(
                        f"Recovering {stage_label} with {recovery_source_label} after "
                        "the visual stage could not complete safely."
                    ),
                    detail=(
                        text_recovery_detail
                        or "Uses completed window observations and authoritative counts without resending the full image."
                    ),
                )
                try:
                    runtime = (
                        text_recovery_runtime_resolver()
                        if text_recovery_runtime_resolver is not None
                        else get_runtime(recovery_model_id)
                    )
                    result = _caption_run_call(
                        call_kind="text",
                        prompt=text_recovery_prompt_clean,
                        img=None,
                        max_new_tokens=text_recovery_max_new_tokens or safe_tokens,
                        system_prompt_override=(
                            text_recovery_system_prompt or editor_system_prompt
                        ),
                        model_id_override=recovery_model_id,
                        runtime_override=runtime,
                        decode_override=recovery_decode,
                        chat_template_kwargs=(
                            text_recovery_chat_template_kwargs
                            or postprocess_chat_template_kwargs
                        ),
                    )
                    if _caption_repetition_loop_detected_impl(result[0]):
                        raise QwenCaptionLoopDetected("qwen_caption_repetition_loop")
                    _caption_record_recovery_event(
                        stage_id=stage_id,
                        stage_label=stage_label,
                        action="text_recovery_succeeded",
                        attempt="text_recovery",
                        model_id=recovery_model_id,
                        call_kind="text",
                        message=f"Recovered {stage_label} with {recovery_source_label}.",
                        detail=(
                            "The visual stage could not complete safely, so the caption was composed "
                            "from completed window observations, authoritative prompt context, "
                            "and full-frame counts."
                        ),
                    )
                    return result
                except (QwenCaptionLoopDetected, QwenCaptionRecoverableGenerationError) as exc:
                    last_recovery_exc = exc
                    _caption_mark_loop_cooldown(recovery_key)
                    is_loop = isinstance(exc, QwenCaptionLoopDetected)
                    _caption_record_recovery_event(
                        stage_id=stage_id,
                        stage_label=stage_label,
                        action="loop_detected" if is_loop else "recoverable_generation_error",
                        attempt="text_recovery",
                        model_id=recovery_model_id,
                        call_kind="text",
                        message=(
                            f"Text-only recovery also looped in {stage_label}."
                            if is_loop
                            else f"Text-only recovery hit a recoverable generation error in {stage_label}."
                        ),
                        detail=(
                            "The runtime was unloaded before trying the remaining recovery path."
                            if is_loop
                            else str(exc)
                        ),
                    )
                    _caption_unload_after_loop()
                    return None

            def _try_loop_output_salvage(
                exc: BaseException,
                *,
                source_attempt: str,
            ) -> Optional[Tuple[str, int, int]]:
                nonlocal last_recovery_exc
                if call_kind != "visual" or not allow_fallback:
                    return None
                if not isinstance(exc, QwenCaptionLoopDetected):
                    return None
                if not _qwen_caption_loop_output_salvageable(
                    exc.raw_output,
                    exc.trimmed_output,
                    degenerate_reason=exc.degenerate_reason,
                ):
                    _caption_record_recovery_event(
                        stage_id=stage_id,
                        stage_label=stage_label,
                        action="loop_salvage_skipped",
                        attempt=f"{source_attempt}_loop_salvage",
                        model_id=primary_model_id,
                        call_kind="text",
                        message=(
                            f"Skipping loop-output salvage for {stage_label}; loop output did not contain "
                            "enough caption-like language."
                        ),
                        detail=(
                            f"reason={exc.degenerate_reason or 'unknown'}; "
                            f"generated_chars={exc.generated_chars}"
                        ),
                    )
                    return None
                salvage_excerpt = _qwen_caption_compact_loop_salvage_text(
                    exc.raw_output,
                    exc.trimmed_output,
                )
                if not salvage_excerpt:
                    return None
                recovery_model_id = str(caption_refinement_model_id or primary_model_id).strip()
                recovery_decode = _qwen_caption_safe_retry_decode(
                    postprocess_decode_params,
                    is_thinking=_qwen_model_id_is_thinking(recovery_model_id),
                )
                original_context = _qwen_caption_compact_original_prompt_context(prompt)
                salvage_prompt = (
                    "Loop-output salvage instruction:\n"
                    "A visual caption generation was stopped because it entered a repeated-output loop. "
                    "The looped text below may still contain a usable image-grounded caption. Extract the "
                    "best final caption from that text, or minimally rewrite the usable caption content into "
                    "one clean final caption. Remove all reasoning, checklists, draft labels, planning notes, "
                    "repeated fragments, and recovery commentary. Do not add objects, actions, counts, or "
                    "spatial details that are not supported by the looped output or original caption context. "
                    "Preserve glossary broad terms and authoritative counts when they appear in the context. "
                    "If the looped text contains no usable image caption, return exactly NO_CAPTION.\n\n"
                )
                if original_context:
                    salvage_prompt = (
                        f"{salvage_prompt}"
                        "Original caption context, for constraints only:\n"
                        f"{original_context}\n\n"
                    )
                salvage_prompt = (
                    f"{salvage_prompt}"
                    "Looped model output to salvage:\n"
                    f"{salvage_excerpt}\n\n"
                    "Return only the salvaged final caption, or exactly NO_CAPTION."
                )
                recovery_key = _qwen_caption_recovery_key(
                    stage_id=stage_id,
                    stage_label=stage_label,
                    call_kind="text",
                    model_id=recovery_model_id,
                    prompt=salvage_prompt,
                    system_prompt=editor_system_prompt,
                    decode_override=recovery_decode,
                )
                recovery_cooldown_remaining = (
                    _qwen_caption_loop_cooldown_remaining(recovery_key)
                    if caption_loop_cooldown
                    else 0.0
                )
                if recovery_cooldown_remaining > 0:
                    _caption_record_recovery_event(
                        stage_id=stage_id,
                        stage_label=stage_label,
                        action="cooldown_skip",
                        attempt=f"{source_attempt}_loop_salvage",
                        model_id=recovery_model_id,
                        call_kind="text",
                        message=(
                            f"Skipping loop-output salvage for {stage_label}; cooldown has "
                            f"{int(recovery_cooldown_remaining)}s left."
                        ),
                        detail="Loop-output salvage looped recently for this exact stage.",
                    )
                    return None
                _caption_record_recovery_event(
                    stage_id=stage_id,
                    stage_label=stage_label,
                    action="loop_salvage_start",
                    attempt=f"{source_attempt}_loop_salvage",
                    model_id=recovery_model_id,
                    call_kind="text",
                    message=(
                        f"Trying to salvage {stage_label} from the looped visual output before "
                        "regenerating the image caption."
                    ),
                    detail=(
                        "Uses the raw looped caption text plus bounded original prompt context; "
                        "does not resend the image tensor."
                    ),
                )
                try:
                    runtime = get_runtime(recovery_model_id)
                    result = _caption_run_call(
                        call_kind="text",
                        prompt=salvage_prompt,
                        img=None,
                        max_new_tokens=min(safe_tokens or 700, 700),
                        system_prompt_override=editor_system_prompt,
                        model_id_override=recovery_model_id,
                        runtime_override=runtime,
                        decode_override=recovery_decode,
                        chat_template_kwargs=postprocess_chat_template_kwargs,
                    )
                    salvaged_text, _ = _extract_caption_from_text_impl(result[0], marker=None)
                    salvaged_text = _sanitize_qwen_caption_impl(salvaged_text)
                    rejected_reason = ""
                    if not salvaged_text or salvaged_text.strip().upper() == "NO_CAPTION":
                        rejected_reason = "no_caption"
                    else:
                        rejected_reason = (
                            _caption_degenerate_reason_impl(
                                salvaged_text,
                                allow_short_caption=True,
                            )
                            or ""
                        )
                    if rejected_reason:
                        _caption_record_recovery_event(
                            stage_id=stage_id,
                            stage_label=stage_label,
                            action="loop_salvage_rejected",
                            attempt=f"{source_attempt}_loop_salvage",
                            model_id=recovery_model_id,
                            call_kind="text",
                            message=f"Rejected loop-output salvage for {stage_label}.",
                            detail=f"reason={rejected_reason}",
                        )
                        return None
                    _caption_record_recovery_event(
                        stage_id=stage_id,
                        stage_label=stage_label,
                        action="loop_salvage_succeeded",
                        attempt=f"{source_attempt}_loop_salvage",
                        model_id=recovery_model_id,
                        call_kind="text",
                        message=f"Recovered {stage_label} by salvaging the looped visual output.",
                        detail=(
                            "The visual caption loop contained usable caption content; the editor model "
                            "removed reasoning, draft markers, and repeated fragments."
                        ),
                    )
                    return salvaged_text, result[1], result[2]
                except (QwenCaptionLoopDetected, QwenCaptionRecoverableGenerationError) as salvage_exc:
                    last_recovery_exc = salvage_exc
                    _caption_mark_loop_cooldown(recovery_key)
                    is_loop = isinstance(salvage_exc, QwenCaptionLoopDetected)
                    _caption_record_recovery_event(
                        stage_id=stage_id,
                        stage_label=stage_label,
                        action="loop_detected" if is_loop else "recoverable_generation_error",
                        attempt=f"{source_attempt}_loop_salvage",
                        model_id=recovery_model_id,
                        call_kind="text",
                        message=(
                            f"Loop-output salvage also looped in {stage_label}."
                            if is_loop
                            else f"Loop-output salvage hit a recoverable generation error in {stage_label}."
                        ),
                        detail=(
                            "The salvage runtime was unloaded before trying the remaining recovery path."
                            if is_loop
                            else str(salvage_exc)
                        ),
                    )
                    _caption_unload_after_loop()
                    return None

            primary_cooldown_remaining = (
                _qwen_caption_loop_cooldown_remaining(primary_key)
                if caption_loop_cooldown and caption_loop_recovery_mode != "off"
                else 0.0
            )
            if primary_cooldown_remaining > 0:
                _caption_record_recovery_event(
                    stage_id=stage_id,
                    stage_label=stage_label,
                    action="cooldown_skip",
                    attempt="primary",
                    model_id=primary_model_id,
                    call_kind=call_kind,
                    message=(
                        f"Skipping loop-prone caption stage on {primary_model_id}; "
                        f"cooldown has {int(primary_cooldown_remaining)}s left."
                    ),
                    detail="Exact stage/model/prompt/decode combination looped recently.",
                )
            else:
                try:
                    result = _caption_run_call(
                        call_kind=call_kind,
                        prompt=prompt,
                        img=img,
                        max_new_tokens=max_new_tokens,
                        system_prompt_override=system_prompt_override,
                        model_id_override=model_id_override,
                        runtime_override=runtime_override,
                        decode_override=decode_override,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                    if _caption_repetition_loop_detected_impl(result[0]):
                        raise QwenCaptionLoopDetected(
                            "qwen_caption_repetition_loop",
                            raw_output=result[0],
                            trimmed_output=_truncate_repeated_caption_loop_impl(result[0]),
                            degenerate_reason=_caption_degenerate_reason_impl(
                                result[0],
                                allow_short_caption=True,
                            ),
                        )
                    return result
                except (QwenCaptionLoopDetected, QwenCaptionRecoverableGenerationError) as exc:
                    last_recovery_exc = exc
                    is_loop = isinstance(exc, QwenCaptionLoopDetected)
                    _caption_mark_loop_cooldown(primary_key)
                    _caption_record_recovery_event(
                        stage_id=stage_id,
                        stage_label=stage_label,
                        action="loop_detected" if is_loop else "recoverable_generation_error",
                        attempt="primary",
                        model_id=primary_model_id,
                        call_kind=call_kind,
                        message=(
                            f"Caption loop detected in {stage_label}; unloading Qwen and recovering from window evidence."
                            if is_loop and text_recovery_available and not text_recovery_after_safe_retry
                            else f"Caption loop detected in {stage_label}; unloading Qwen and retrying safely."
                            if is_loop and allow_safe_retry
                            else f"Caption loop detected in {stage_label}; unloading Qwen and recovering from prompt context."
                            if is_loop and text_recovery_available
                            else f"Recoverable generation error in {stage_label}; unloading Qwen and recovering from prompt context."
                            if text_recovery_available and not text_recovery_after_safe_retry
                            else f"Recoverable generation error in {stage_label}; unloading Qwen and retrying safely."
                            if allow_safe_retry
                            else f"Caption loop detected in {stage_label}; recovery is disabled."
                            if is_loop
                            else f"Recoverable generation error in {stage_label}; recovery is disabled."
                        ),
                        detail=(
                            "Primary generation tripped the repetition loop guard."
                            if is_loop
                            else str(exc)
                        ),
                    )
                    _caption_unload_after_loop()
                    recovered_from_loop = _try_loop_output_salvage(
                        exc,
                        source_attempt="primary",
                    )
                    if recovered_from_loop is not None:
                        return recovered_from_loop
                    if not allow_safe_retry:
                        raise

            if not text_recovery_after_safe_retry:
                recovered_text = _try_text_recovery()
                if recovered_text is not None:
                    return recovered_text

            if allow_safe_retry:
                safe_cooldown_remaining = (
                    _qwen_caption_loop_cooldown_remaining(safe_key)
                    if caption_loop_cooldown
                    else 0.0
                )
                if safe_cooldown_remaining > 0:
                    _caption_record_recovery_event(
                        stage_id=stage_id,
                        stage_label=stage_label,
                        action="cooldown_skip",
                        attempt="safe_retry",
                        model_id=primary_model_id,
                        call_kind=call_kind,
                        message=(
                            f"Skipping safe retry for {stage_label}; cooldown has "
                            f"{int(safe_cooldown_remaining)}s left."
                        ),
                        detail="Safe decode already looped for this exact stage.",
                    )
                else:
                    try:
                        runtime = get_runtime(primary_model_id)
                        result = _caption_run_call(
                            call_kind=call_kind,
                            prompt=prompt,
                            img=img,
                            max_new_tokens=safe_tokens,
                            system_prompt_override=system_prompt_override,
                            model_id_override=primary_model_id,
                            runtime_override=runtime,
                            decode_override=safe_decode,
                            chat_template_kwargs=chat_template_kwargs,
                        )
                        if _caption_repetition_loop_detected_impl(result[0]):
                            raise QwenCaptionLoopDetected(
                                "qwen_caption_repetition_loop",
                                raw_output=result[0],
                                trimmed_output=_truncate_repeated_caption_loop_impl(result[0]),
                                degenerate_reason=_caption_degenerate_reason_impl(
                                    result[0],
                                    allow_short_caption=True,
                                ),
                            )
                        _caption_record_recovery_event(
                            stage_id=stage_id,
                            stage_label=stage_label,
                            action="safe_retry_succeeded",
                            attempt="safe_retry",
                            model_id=primary_model_id,
                            call_kind=call_kind,
                            message=f"Recovered {stage_label} with safe decode on {primary_model_id}.",
                            detail="Sampling and repetition controls were tightened for this retry.",
                        )
                        return result
                    except (QwenCaptionLoopDetected, QwenCaptionRecoverableGenerationError) as exc:
                        last_recovery_exc = exc
                        is_loop = isinstance(exc, QwenCaptionLoopDetected)
                        _caption_mark_loop_cooldown(safe_key)
                        _caption_record_recovery_event(
                            stage_id=stage_id,
                            stage_label=stage_label,
                            action="loop_detected" if is_loop else "recoverable_generation_error",
                            attempt="safe_retry",
                            model_id=primary_model_id,
                            call_kind=call_kind,
                            message=(
                                f"Safe retry also looped in {stage_label}."
                                if is_loop
                                else f"Safe retry hit a recoverable generation error in {stage_label}."
                            ),
                            detail=(
                                "The failed runtime was unloaded before trying any "
                                "fallback model or text-only recovery path."
                                if is_loop
                                else str(exc)
                            ),
                        )
                        _caption_unload_after_loop()
                        recovered_from_loop = _try_loop_output_salvage(
                            exc,
                            source_attempt="safe_retry",
                        )
                        if recovered_from_loop is not None:
                            return recovered_from_loop

            fallback_model_id = caption_fallback_model_id if allow_fallback else ""
            if fallback_model_id and fallback_model_id != primary_model_id:
                fallback_decode = _qwen_caption_safe_retry_decode(
                    decode_override,
                    is_thinking=_qwen_model_id_is_thinking(fallback_model_id),
                )
                fallback_key = _qwen_caption_recovery_key(
                    stage_id=stage_id,
                    stage_label=stage_label,
                    call_kind=call_kind,
                    model_id=fallback_model_id,
                    prompt=prompt,
                    system_prompt=system_prompt_override,
                    decode_override=fallback_decode,
                )
                fallback_cooldown_remaining = (
                    _qwen_caption_loop_cooldown_remaining(fallback_key)
                    if caption_loop_cooldown
                    else 0.0
                )
                if fallback_cooldown_remaining > 0:
                    _caption_record_recovery_event(
                        stage_id=stage_id,
                        stage_label=stage_label,
                        action="cooldown_skip",
                        attempt="fallback",
                        model_id=fallback_model_id,
                        call_kind=call_kind,
                        message=(
                            f"Skipping fallback model {fallback_model_id}; cooldown has "
                            f"{int(fallback_cooldown_remaining)}s left."
                        ),
                        detail="Fallback model looped recently for this exact stage.",
                    )
                else:
                    try:
                        runtime = get_runtime(fallback_model_id)
                        result = _caption_run_call(
                            call_kind=call_kind,
                            prompt=prompt,
                            img=img,
                            max_new_tokens=safe_tokens,
                            system_prompt_override=system_prompt_override,
                            model_id_override=fallback_model_id,
                            runtime_override=runtime,
                            decode_override=fallback_decode,
                            chat_template_kwargs=chat_template_kwargs,
                        )
                        if _caption_repetition_loop_detected_impl(result[0]):
                            raise QwenCaptionLoopDetected(
                                "qwen_caption_repetition_loop",
                                raw_output=result[0],
                                trimmed_output=_truncate_repeated_caption_loop_impl(result[0]),
                                degenerate_reason=_caption_degenerate_reason_impl(
                                    result[0],
                                    allow_short_caption=True,
                                ),
                            )
                        _caption_record_recovery_event(
                            stage_id=stage_id,
                            stage_label=stage_label,
                            action="fallback_succeeded",
                            attempt="fallback",
                            model_id=fallback_model_id,
                            call_kind=call_kind,
                            message=f"Recovered {stage_label} with fallback model {fallback_model_id}.",
                            detail="Fallback used safe decode after the primary model looped.",
                        )
                        return result
                    except (QwenCaptionLoopDetected, QwenCaptionRecoverableGenerationError) as exc:
                        last_recovery_exc = exc
                        is_loop = isinstance(exc, QwenCaptionLoopDetected)
                        _caption_mark_loop_cooldown(fallback_key)
                        _caption_record_recovery_event(
                            stage_id=stage_id,
                            stage_label=stage_label,
                            action="loop_detected" if is_loop else "recoverable_generation_error",
                            attempt="fallback",
                            model_id=fallback_model_id,
                            call_kind=call_kind,
                            message=(
                                f"Fallback model also looped in {stage_label}."
                                if is_loop
                                else f"Fallback model hit a recoverable generation error in {stage_label}."
                            ),
                            detail=(
                                "The failed runtime was unloaded before trying any "
                                "text-only or deterministic recovery path."
                                if is_loop
                                else str(exc)
                            ),
                        )
                        _caption_unload_after_loop()
                        recovered_from_loop = _try_loop_output_salvage(
                            exc,
                            source_attempt="fallback",
                        )
                        if recovered_from_loop is not None:
                            return recovered_from_loop

            if text_recovery_after_safe_retry:
                recovered_text = _try_text_recovery()
                if recovered_text is not None:
                    return recovered_text

            fallback_text = _sanitize_qwen_caption_impl(deterministic_recovery_text or "")
            if allow_fallback and fallback_text:
                _caption_record_recovery_event(
                    stage_id=stage_id,
                    stage_label=stage_label,
                    action="deterministic_recovery_succeeded",
                    attempt="deterministic_recovery",
                    model_id=primary_model_id,
                    call_kind="deterministic",
                    message=f"Recovered {stage_label} with deterministic count/layout fallback.",
                    detail=(
                        deterministic_recovery_detail
                        or "All model-based recovery paths looped, failed, or were unavailable; "
                        "the fallback used authoritative counts and spatial hints."
                    ),
                )
                return fallback_text, 0, 0

            if last_recovery_exc is not None:
                raise last_recovery_exc
            raise QwenCaptionLoopDetected("qwen_caption_repetition_loop")

        def _caption_visual_inference(
            prompt: str,
            img: Image.Image,
            max_new_tokens: Optional[int] = None,
            system_prompt_override: Optional[str] = None,
            model_id_override: Optional[str] = None,
            runtime_override: Optional[Tuple[Any, Any]] = None,
            decode_override: Optional[Dict[str, Any]] = None,
            chat_template_kwargs: Optional[Dict[str, Any]] = None,
            text_recovery_prompt: Optional[str] = None,
            text_recovery_system_prompt: Optional[str] = None,
            text_recovery_model_id: Optional[str] = None,
            text_recovery_runtime_resolver: Optional[Callable[[], Tuple[Any, Any]]] = None,
            text_recovery_decode_override: Optional[Dict[str, Any]] = None,
            text_recovery_chat_template_kwargs: Optional[Dict[str, Any]] = None,
            text_recovery_max_new_tokens: Optional[int] = None,
            text_recovery_detail: Optional[str] = None,
            text_recovery_after_safe_retry: bool = False,
            deterministic_recovery_text: Optional[str] = None,
            deterministic_recovery_detail: Optional[str] = None,
        ) -> Tuple[str, int, int]:
            return _caption_run_with_loop_recovery(
                call_kind="visual",
                prompt=prompt,
                img=img,
                max_new_tokens=max_new_tokens,
                system_prompt_override=system_prompt_override,
                model_id_override=model_id_override,
                runtime_override=runtime_override,
                decode_override=decode_override,
                chat_template_kwargs=chat_template_kwargs,
                text_recovery_prompt=text_recovery_prompt,
                text_recovery_system_prompt=text_recovery_system_prompt,
                text_recovery_model_id=text_recovery_model_id,
                text_recovery_runtime_resolver=text_recovery_runtime_resolver,
                text_recovery_decode_override=text_recovery_decode_override,
                text_recovery_chat_template_kwargs=text_recovery_chat_template_kwargs,
                text_recovery_max_new_tokens=text_recovery_max_new_tokens,
                text_recovery_detail=text_recovery_detail,
                text_recovery_after_safe_retry=text_recovery_after_safe_retry,
                deterministic_recovery_text=deterministic_recovery_text,
                deterministic_recovery_detail=deterministic_recovery_detail,
            )

        def _caption_text_inference(
            prompt: str,
            *,
            max_new_tokens: Optional[int] = None,
            system_prompt_override: Optional[str] = None,
            model_id_override: Optional[str] = None,
            runtime_override: Optional[Tuple[Any, Any]] = None,
            decode_override: Optional[Dict[str, Any]] = None,
            chat_template_kwargs: Optional[Dict[str, Any]] = None,
            deterministic_recovery_text: Optional[str] = None,
            deterministic_recovery_detail: Optional[str] = None,
        ) -> Tuple[str, int, int]:
            return _caption_run_with_loop_recovery(
                call_kind="text",
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                system_prompt_override=system_prompt_override,
                model_id_override=model_id_override,
                runtime_override=runtime_override,
                decode_override=decode_override,
                chat_template_kwargs=chat_template_kwargs,
                deterministic_recovery_text=deterministic_recovery_text,
                deterministic_recovery_detail=deterministic_recovery_detail,
            )

        windowed_captions: List[Tuple[int, int, int, str]] = []
        first_stage_output_sections: List[Tuple[str, str]] = []
        cleanup_count = 0
        refine_count = 0
        merge_count = 0

        def _record_first_stage_output(label: str, text: Optional[str]) -> None:
            cleaned = str(text or "").strip()
            if cleaned:
                first_stage_output_sections.append((label, cleaned))

        def _format_refine_labels(
            labels: Sequence[str],
            glossary: Optional[Dict[str, List[str]]],
        ) -> List[str]:
            display: List[str] = []
            seen: set[str] = set()
            for label in labels or []:
                preferred = _caption_preferred_label_impl(label, glossary).strip()
                if not preferred or preferred in seen:
                    continue
                seen.add(preferred)
                display.append(preferred)
            return display

        def _format_count_conflict_note(
            labels: Sequence[str],
            counts_map: Dict[str, int],
            glossary: Optional[Dict[str, List[str]]],
        ) -> str:
            pieces: List[str] = []
            for label in labels or []:
                try:
                    count = int(counts_map.get(label, 0))
                except (TypeError, ValueError):
                    continue
                if count <= 0:
                    continue
                preferred = _caption_preferred_label_impl(label, glossary).strip()
                if preferred:
                    pieces.append(f"{preferred}: {count}")
            if not pieces:
                return ""
            return (
                "Correct these authoritative object counts: "
                + ", ".join(pieces)
                + ". Remove contradictory plural, group, or quantity wording."
            )

        def _format_authoritative_counts_note(
            counts_map: Dict[str, int],
            glossary: Optional[Dict[str, List[str]]],
        ) -> str:
            if not include_counts or not restrict_to_labels or not counts_map:
                return ""
            pieces: List[str] = []
            for label, count in counts_map.items():
                try:
                    count_int = int(count)
                except (TypeError, ValueError):
                    continue
                if count_int <= 0:
                    continue
                preferred = _caption_preferred_label_impl(label, glossary).strip()
                if preferred:
                    pieces.append(f"{preferred}: {count_int}")
            if not pieces:
                return ""
            return (
                "Authoritative object counts that must appear in the caption as natural scene facts: "
                + ", ".join(pieces)
                + ". State counts with digits; do not say counts were provided."
            )

        authoritative_counts_note = _format_authoritative_counts_note(counts, glossary_map)

        def _caption_recovery_noun(label: str, count: int) -> str:
            preferred = _caption_preferred_label_impl(label, glossary_map).strip() or label
            noun = re.sub(r"\s+", " ", preferred.replace("_", " ")).strip().lower()
            if count == 1:
                return noun
            if noun == "person":
                return "people"
            if noun.endswith("s"):
                return noun
            if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
                return f"{noun[:-1]}ies"
            return f"{noun}s"

        def _caption_recovery_join(parts: Sequence[str]) -> str:
            cleaned = [str(part or "").strip() for part in parts if str(part or "").strip()]
            if not cleaned:
                return ""
            if len(cleaned) == 1:
                return cleaned[0]
            if len(cleaned) == 2:
                return f"{cleaned[0]} and {cleaned[1]}"
            return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

        def _caption_recovery_hint_region(hint: QwenCaptionHint) -> str:
            bbox = hint.bbox or []
            if len(bbox) != 4:
                return ""
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox]
            except (TypeError, ValueError):
                return ""
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                return ""
            max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
            if max_coord <= 1.5:
                x_den = y_den = 1.0
            elif max_coord <= 1000.0 and max(image_width, image_height, 1) < 900:
                x_den = y_den = 1000.0
            else:
                x_den = float(max(1, image_width))
                y_den = float(max(1, image_height))
            cx = max(0.0, min(1.0, ((x1 + x2) / 2.0) / x_den))
            cy = max(0.0, min(1.0, ((y1 + y2) / 2.0) / y_den))
            horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
            vertical = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
            if horizontal == "center" and vertical == "middle":
                return "center"
            if horizontal == "center":
                return f"{vertical} area"
            if vertical == "middle":
                return f"{horizontal} side"
            return f"{vertical}-{horizontal}"

        def _caption_deterministic_recovery_caption(
            hint_counts: Mapping[str, int],
            hints: Sequence[QwenCaptionHint],
            *,
            crop: bool = False,
        ) -> str:
            count_phrases: List[str] = []
            for label, count in hint_counts.items():
                try:
                    count_int = int(count)
                except (TypeError, ValueError):
                    continue
                if count_int <= 0:
                    continue
                noun = _caption_recovery_noun(label, count_int)
                count_phrases.append(f"{count_int} {noun}")
            if not count_phrases:
                labels = sorted({str(hint.label or "").strip() for hint in hints if hint.label})
                count_phrases = [
                    _caption_recovery_noun(label, 2)
                    for label in labels
                    if str(label or "").strip()
                ]
            if not count_phrases:
                return ""

            regions_by_label: Dict[str, Dict[str, int]] = {}
            for hint in hints:
                label = str(hint.label or "").strip()
                if not label:
                    continue
                region = _caption_recovery_hint_region(hint)
                if not region:
                    continue
                regions = regions_by_label.setdefault(label, {})
                regions[region] = regions.get(region, 0) + 1

            layout_parts: List[str] = []
            for label, region_counts in regions_by_label.items():
                try:
                    count_int = int(hint_counts.get(label, 0))
                except (TypeError, ValueError):
                    count_int = 0
                noun = _caption_recovery_noun(label, max(2, count_int))
                top_regions = sorted(
                    region_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:2]
                region_text = _caption_recovery_join([region for region, _count in top_regions])
                if not region_text:
                    continue
                if count_int == 1:
                    singular = _caption_recovery_noun(label, 1)
                    layout_parts.append(f"the single {singular} is positioned around the {region_text}")
                elif top_regions and sum(region_counts.values()) and top_regions[0][1] / sum(region_counts.values()) >= 0.55:
                    layout_parts.append(f"{noun} are concentrated around the {region_text}")
                else:
                    layout_parts.append(f"{noun} are spread across the {region_text}")

            opening = "The crop contains" if crop else (
                "From a high angle, the scene contains"
                if re.search(r"\b(high[- ]angle|top[- ]down|drone|aerial|looking down)\b", user_prompt or "", re.I)
                else "The scene contains"
            )
            sentences = [f"{opening} {_caption_recovery_join(count_phrases)}."]
            if layout_parts:
                sentences.append(f"{_caption_recovery_join(layout_parts).capitalize()}.")
            if not crop:
                sentences.append(
                    "The arrangement is described from the object distribution and surrounding scene context."
                )
            return _sanitize_qwen_caption_impl(" ".join(sentences))

        def _caption_term_guard_sources() -> List[Tuple[str, str]]:
            sources = list(first_stage_output_sections)
            for idx, (_x0, _y0, _size, caption) in enumerate(windowed_captions, start=1):
                if caption:
                    sources.append((f"Window {idx} cleaned caption", caption))
            return sources

        def _stabilize_glossary_subtypes(caption: str) -> str:
            guard_counts = counts or {label: 1 for label in glossary_map.keys()}
            if not caption or not guard_counts or not glossary_map:
                return caption
            return _caption_demote_unstable_glossary_subtypes_impl(
                caption,
                guard_counts,
                glossary_map,
                source_outputs=_caption_term_guard_sources(),
            )

        _qwen_progress_update(
            phase="load",
            phase_label="Loading",
            progress=0.12,
            message=f"Ensuring Qwen caption model is ready: {desired_model_id}",
            model_id=desired_model_id,
            step_id="load_model",
            step_region=full_step_region,
        )
        resolve_main_runtime()
        _raise_if_qwen_cancelled()
        if caption_mode == "windowed":
            if window_size is None:
                raise HTTPException(
                    status_code=HTTP_400_BAD_REQUEST,
                    detail="qwen_caption_window_size_invalid",
                )
            window_index = 0
            window_model_id = desired_model_id
            window_base_model_id = window_model_id
            window_is_thinking = "Thinking" in window_model_id
            for y0 in y_positions:
                for x0 in x_positions:
                    _raise_if_qwen_cancelled()
                    window_index += 1
                    window_hints = grouped_hints.get((x0, y0), [])
                    window_region = {
                        "x": x0,
                        "y": y0,
                        "width": min(window_size, max(1, image_width - x0)),
                        "height": min(window_size, max(1, image_height - y0)),
                        "image_width": image_width,
                        "image_height": image_height,
                        "image_name": request_image_name,
                        "kind": "window",
                        "label": f"Window {window_index}/{total_windows}",
                    }
                    window_detail = (
                        f"Region [{x0}, {y0}] to "
                        f"[{min(image_width, x0 + window_size)}, {min(image_height, y0 + window_size)}]; "
                        f"{len(window_hints)} hint{'s' if len(window_hints) != 1 else ''}"
                    )
                    _qwen_progress_update(
                        phase="window",
                        phase_label="Captioning Window",
                        progress=0.40 + (0.15 * (window_index - 1) / total_windows),
                        message=(
                            f"Captioning window {window_index}/{total_windows}"
                            if window_hints or caption_all_windows
                            else f"Skipping empty window {window_index}/{total_windows}"
                        ),
                        model_id=window_model_id,
                        step_id=f"window_{window_index}",
                        step_detail=window_detail,
                        step_region=window_region,
                    )
                    if not window_hints and not caption_all_windows:
                        continue
                    window_allowed = _allowed_caption_labels_impl(window_hints)
                    window_glossary_map = _caption_glossary_map_impl(
                        payload.labelmap_glossary,
                        [hint.label for hint in window_hints if hint.label],
                    )
                    window_allowed_prompt = (
                        [
                            _caption_preferred_label_impl(label, window_glossary_map)
                            for label in window_allowed
                        ]
                        if window_allowed
                        else []
                    )
                    window_prompt, window_counts, _, _ = _build_qwen_caption_prompt_impl(
                        user_prompt,
                        window_hints,
                        window_size,
                        window_size,
                        include_counts,
                        include_coords,
                        max_boxes,
                        detailed_mode=True,
                        restrict_to_labels=restrict_to_labels,
                        labelmap_glossary=payload.labelmap_glossary,
                        max_sentences=final_caption_max_sentences,
                        context_prompt=_caption_window_detection_context_prompt(),
                    )
                    if payload.caption_window_prompt:
                        window_task_prompt = payload.caption_window_prompt.strip()
                    else:
                        window_caption_instruction = _format_caption_window_sentence_instruction(payload)
                        window_task_prompt = (
                            "This crop is a subregion of the full image.\n"
                            "Apply the crop caption context above to this crop only.\n"
                            "Use the crop image as truth and treat the provided object context as priors, not as text to mention.\n"
                            "Describe crop-local objects of interest with natural-language class meanings, visible attributes, counts, "
                            "and spatial relations when available. Mention surrounding scene details only when they clarify those objects or the crop. "
                            "Do not add unsupported object categories or treat background context as extra counted inventory.\n"
                            f"{window_caption_instruction}No reasoning or preamble. "
                            "Do not mention labels, hints, bounding boxes, coordinates, glossary text, or that counts were provided, "
                            "but if counts are provided then do make sure to keep those counts in your output. "
                            "Do not output internal dataset class names or any token with underscores. "
                            "Use narrower subtypes only when the crop clearly supports them."
                        )
                    if glossary_line:
                        window_prompt = f"{window_prompt}\n{glossary_line}"
                    if restrict_to_labels and window_allowed_prompt:
                        window_prompt = (
                            f"{window_prompt}\nAllowed classes: {', '.join(window_allowed_prompt)}. "
                            "This window has labeled support boxes. Use the crop detection context to describe those labeled objects, "
                            "their visible attributes, and surrounding crop details that clarify them. "
                            "Do not promote unrelated background objects into extra counted categories, and do not introduce unsupported entity types."
                        )
                    elif restrict_to_labels and label_hints and caption_all_windows:
                        window_prompt = (
                            f"{window_prompt}\nThis window has no labeled support boxes. "
                            "Use it only for broad scene context; do not enumerate object categories, counts, or new object lists."
                        )
                    window_prompt = f"{window_prompt}\n\nCrop task:\n{window_task_prompt}"
                    if window_is_thinking:
                        window_prompt = _adjust_prompt_for_thinking_impl(window_prompt)
                    window_img = pil_img.crop((x0, y0, x0 + window_size, y0 + window_size))
                    window_max_tokens = max_new_tokens
                    window_text_recovery_prompt = (
                        f"{window_prompt}\n"
                        "Text-only recovery instruction: the visual crop generation could not complete safely. "
                        "Write a concise crop observation from the authoritative crop counts, class inventory, "
                        "representative box layout, and user request above. Treat the counts as truth, "
                        "do not mention recovery, labels, hints, boxes, coordinates, or prompts, and return only "
                        "1-3 concrete sentences about this crop."
                    )
                    window_deterministic_recovery_caption = _caption_deterministic_recovery_caption(
                        window_counts,
                        window_hints,
                        crop=True,
                    )
                    _qwen_progress_begin_output_section(f"Window {window_index}/{total_windows} output")
                    qwen_text, _, _ = _caption_visual_inference(
                        window_prompt,
                        window_img,
                        max_new_tokens=window_max_tokens,
                        system_prompt_override=system_prompt,
                        runtime_override=get_runtime(window_model_id),
                        decode_override=decode_params,
                        chat_template_kwargs=caption_chat_template_kwargs,
                        text_recovery_prompt=window_text_recovery_prompt,
                        text_recovery_system_prompt=editor_system_prompt,
                        text_recovery_model_id=caption_refinement_model_id,
                        text_recovery_runtime_resolver=resolve_refinement_runtime,
                        text_recovery_decode_override=postprocess_decode_params,
                        text_recovery_chat_template_kwargs=postprocess_chat_template_kwargs,
                        text_recovery_max_new_tokens=min(window_max_tokens or 1000, 1000),
                        text_recovery_detail=(
                            "Primary visual crop generation could not complete safely; this recovery path uses "
                            "the crop counts and representative layout text without sending another crop tensor."
                        ),
                        deterministic_recovery_text=window_deterministic_recovery_caption,
                        deterministic_recovery_detail=(
                            "All model-based crop recovery paths looped, failed, or were unavailable; "
                            "the fallback used authoritative crop counts and relative object layout."
                        ),
                    )
                    _raise_if_qwen_cancelled()
                    _record_first_stage_output(
                        f"Window {window_index}/{total_windows} raw output", qwen_text
                    )
                    window_caption, _ = _extract_caption_from_text_impl(qwen_text, marker=None)
                    window_caption = _sanitize_qwen_caption_impl(window_caption)
                    if window_is_thinking and _thinking_caption_needs_cleanup_impl(
                        window_caption, qwen_text
                    ):
                        cleanup_model = caption_refinement_model_id
                        _qwen_progress_update(
                            phase="cleanup",
                            phase_label="Cleaning Window",
                            progress=0.44 + (0.12 * window_index / total_windows),
                            message=f"Cleaning thinking output for window {window_index}/{total_windows}",
                            model_id=cleanup_model,
                            step_id=f"window_{window_index}",
                            step_detail=f"{window_detail}; cleaning thinking wrapper",
                            step_region=window_region,
                        )
                        _qwen_progress_begin_output_section(f"Window {window_index}/{total_windows} cleanup output")
                        window_caption = _caption_cleanup(
                            window_caption,
                            window_img,
                            window_max_tokens,
                            window_base_model_id,
                            use_caption_cache,
                            model_id_override=cleanup_model,
                            runtime_override=resolve_refinement_runtime(),
                            allowed_labels=(
                                window_allowed_prompt
                                if restrict_to_labels and window_allowed_prompt
                                else None
                            ),
                            strict=True,
                            minimal_edit=True,
                            source_output=qwen_text,
                            authoritative_counts_note_override=_format_authoritative_counts_note(
                                window_counts,
                                window_glossary_map,
                            ),
                        )
                        cleanup_count += 1
                    if _caption_is_degenerate_impl(window_caption):
                        cleanup_model = caption_refinement_model_id
                        _qwen_progress_update(
                            phase="cleanup",
                            phase_label="Cleaning Window",
                            progress=0.45 + (0.12 * window_index / total_windows),
                            message=f"Cleaning repeated or incomplete window {window_index}/{total_windows} caption",
                            model_id=cleanup_model,
                            step_id=f"window_{window_index}",
                            step_detail=f"{window_detail}; cleaning repeated or incomplete text",
                            step_region=window_region,
                        )
                        _qwen_progress_begin_output_section(f"Window {window_index}/{total_windows} cleanup output")
                        window_caption = _caption_cleanup(
                            window_caption,
                            window_img,
                            window_max_tokens,
                            window_base_model_id,
                            use_caption_cache,
                            model_id_override=cleanup_model,
                            runtime_override=resolve_refinement_runtime(),
                            allowed_labels=(
                                window_allowed_prompt
                                if restrict_to_labels and window_allowed_prompt
                                else None
                            ),
                            strict=True,
                            minimal_edit=True,
                            source_output=qwen_text,
                            authoritative_counts_note_override=_format_authoritative_counts_note(
                                window_counts,
                                window_glossary_map,
                            ),
                        )
                        cleanup_count += 1
                    if _caption_needs_completion_impl(window_caption) or _caption_has_meta_impl(
                        window_caption
                    ):
                        cleanup_model = caption_refinement_model_id
                        _qwen_progress_update(
                            phase="cleanup",
                            phase_label="Cleaning Window",
                            progress=0.46 + (0.12 * window_index / total_windows),
                            message=f"Removing meta text from window {window_index}/{total_windows}",
                            model_id=cleanup_model,
                            step_id=f"window_{window_index}",
                            step_detail=f"{window_detail}; removing meta text",
                            step_region=window_region,
                        )
                        _qwen_progress_begin_output_section(f"Window {window_index}/{total_windows} cleanup output")
                        window_caption = _caption_cleanup(
                            window_caption,
                            window_img,
                            window_max_tokens,
                            window_base_model_id,
                            use_caption_cache,
                            model_id_override=cleanup_model,
                            runtime_override=resolve_refinement_runtime(),
                            allowed_labels=(
                                window_allowed_prompt
                                if restrict_to_labels and window_allowed_prompt
                                else None
                            ),
                            strict=True,
                            minimal_edit=True,
                            source_output=qwen_text,
                            authoritative_counts_note_override=_format_authoritative_counts_note(
                                window_counts,
                                window_glossary_map,
                            ),
                        )
                        cleanup_count += 1
                    count_conflicts = (
                        _caption_count_conflicts_impl(
                            window_caption,
                            window_counts,
                            window_glossary_map,
                        )
                        if include_counts
                        else []
                    )
                    count_issues = sorted(
                        set(
                            count_conflicts
                            + (
                                _caption_missing_exact_counts_impl(
                                    window_caption,
                                    window_counts,
                                    window_glossary_map,
                                )
                                if include_counts
                                else []
                            )
                        )
                    )
                    needs_refine, missing = _caption_needs_refine_impl(
                        window_caption,
                        window_counts,
                        detailed_mode=True,
                        include_counts=include_counts,
                        glossary_map=window_glossary_map,
                    )
                    if needs_refine:
                        refine_model = caption_refinement_model_id
                        missing_display = _format_refine_labels(missing, window_glossary_map)
                        count_conflict_note = _format_count_conflict_note(
                            count_issues,
                            window_counts,
                            window_glossary_map,
                        )
                        _qwen_progress_update(
                            phase="guard",
                            phase_label="Guarding Window",
                            progress=0.48 + (0.12 * window_index / total_windows),
                            message=f"Applying window {window_index}/{total_windows} coverage/count guard",
                            model_id=refine_model,
                            step_id=f"window_{window_index}",
                            step_detail=(
                                f"{window_detail}; coverage/count guard ensuring "
                                f"{', '.join(missing_display) if missing_display else 'window class coverage'}"
                            ),
                            step_region=window_region,
                        )
                        allowed_note = ""
                        if restrict_to_labels and window_allowed_prompt:
                            allowed_note = (
                                f"Allowed classes: {', '.join(window_allowed_prompt)}. "
                                "Do not introduce any other entity types."
                            )
                        elif not restrict_to_labels:
                            allowed_note = (
                                "You may mention additional visible objects beyond the hints."
                            )
                        refine_notes = []
                        if missing_display:
                            refine_notes.append(
                                f"Ensure the caption mentions: {', '.join(missing_display)}."
                            )
                        if count_conflict_note:
                            refine_notes.append(count_conflict_note)
                        if not refine_notes:
                            refine_notes.append("Ensure all labeled classes in this window are mentioned.")
                        missing_note = " ".join(refine_notes)
                        refine_prompt = (
                            f"{window_prompt}\nDraft caption: {window_caption}\n{missing_note}"
                        )
                        if qwen_text:
                            first_stage_context = _format_caption_source_output_context_impl(
                                source_outputs=[
                                    (
                                        f"Window {window_index}/{total_windows} raw output",
                                        qwen_text,
                                    )
                                ]
                            )
                            if first_stage_context:
                                refine_prompt = f"{refine_prompt}\n\n{first_stage_context}"
                        if allowed_note:
                            refine_prompt = f"{refine_prompt}\n{allowed_note}"
                        refine_prompt = f"{refine_prompt}\n{coverage_policy}"
                        if payload.caption_cleanup_prompt:
                            refine_prompt = (
                                f"{refine_prompt}\n"
                                f"{payload.caption_cleanup_prompt.strip()} "
                                "Return only a concise, complete caption (1-3 sentences) with no coordinates."
                            )
                        else:
                            refine_prompt = (
                                f"{refine_prompt}\n"
                                "Edit the draft with minimal changes. Do not introduce new objects or actions. "
                                f"{broad_term_editor_note} "
                                "Return only a concise, complete caption (1-3 sentences) with no coordinates."
                            )
                        refine_system = editor_system_prompt
                        _qwen_progress_begin_output_section(f"Window {window_index}/{total_windows} guard output")
                        refine_text, _, _ = _caption_text_inference(
                            refine_prompt,
                            max_new_tokens=refine_max_tokens,
                            system_prompt_override=refine_system,
                            model_id_override=refine_model,
                            runtime_override=resolve_refinement_runtime(),
                            decode_override=postprocess_decode_params,
                            chat_template_kwargs=postprocess_chat_template_kwargs,
                        )
                        _raise_if_qwen_cancelled()
                        previous_window_caption = window_caption
                        window_caption, _ = _extract_caption_from_text_impl(
                            refine_text, marker=None
                        )
                        window_caption = _accept_guard_caption(
                            window_caption,
                            previous_window_caption,
                            stage=f"window {window_index}/{total_windows} coverage/count guard",
                        )
                        refine_count += 1
                    if window_caption:
                        window_caption_for_merge = _clean_caption_source_context_text_impl(window_caption)
                        if window_caption_for_merge:
                            windowed_captions.append((x0, y0, window_size, window_caption_for_merge))
                            _emit_caption_window(x0, y0, window_size, window_caption_for_merge)
            if windowed_captions:
                window_lines = _format_caption_window_observation_lines_impl(
                    windowed_captions,
                    image_width=image_width,
                    image_height=image_height,
                    full_label_hints=label_hints,
                    window_hints_by_window=grouped_hints,
                    glossary_map=glossary_map,
                )
                window_lines.append(
                    "Treat these observations as close-up source notes, not a separate object inventory."
                )
                if overlap_guidance:
                    window_lines.append(overlap_guidance)
                if short_caption_requested:
                    if include_counts and restrict_to_labels:
                        window_lines.append(
                            "Now write the full-image caption at the requested length. Use the close-up observations only for the most important visible details. "
                            "Mention only central labeled object types, and summarize repetitive objects instead of listing them. "
                            "Do not mention labels, hints, coordinates, or that counts were provided."
                        )
                        window_lines.append(
                            "If window observations conflict, trust the full image and the authoritative counts. Avoid self-contradictions."
                        )
                    else:
                        window_lines.append(
                            "Now write the full-image caption at the requested length. Use the close-up observations only for the most important visible details. "
                            "Do not mention labels, hints, or coordinates."
                        )
                        window_lines.append(
                            "If window observations conflict, trust the full image. Avoid self-contradictions."
                        )
                    window_lines.append(
                        "Do not expand beyond the requested caption length just because window observations are available."
                    )
                elif include_counts and restrict_to_labels:
                    window_lines.append(
                        "Now describe the full image in detail. Use all labeled object counts and the close-up observations. "
                        "Mention every class that appears in the hints, and summarize repetitive objects (e.g., many cars as a parking lot) "
                        "unless only a few are present or a specific action stands out. "
                        "Do not mention labels, hints, coordinates, or that counts were provided."
                    )
                    window_lines.append(
                        "If window observations conflict, trust the full image and the authoritative counts. Avoid self-contradictions."
                    )
                else:
                    window_lines.append(
                        "Now describe the full image in detail using the close-up observations. "
                        "Use detector hints as suggestions, and mention other visible objects. "
                        "Do not mention labels, hints, or coordinates."
                    )
                    window_lines.append(
                        "If window observations conflict, trust the full image. Avoid self-contradictions."
                    )
                window_lines.append(
                    (
                        "In the final caption, include only the most important window details that fit the requested length."
                        if short_caption_requested
                        else (
                            "In the final caption, preserve specific details about labeled or clearly central objects from the windows "
                            "(e.g., counts, actions, and notable attributes). "
                            "Do not promote unlabeled background observations into extra counted objects; longer captions are acceptable when needed."
                        )
                    )
                )
                if broad_term_editor_note:
                    window_lines.append(broad_term_editor_note)
                prompt_text = f"{prompt_text}\n" + "\n".join(window_lines)
        windowed_text_only_full_compose = (
            caption_mode == "windowed"
            and bool(windowed_captions)
            and windowed_full_image_strategy == "text_only"
            and not (two_stage and is_thinking)
        )
        if two_stage and is_thinking:
            _raise_if_qwen_cancelled()
            _qwen_progress_update(
                phase="draft",
                phase_label="Drafting",
                progress=0.55,
                message="Generating full-image draft caption",
                model_id=desired_model_id,
                step_id="draft_caption",
                step_region=full_step_region,
                step_detail="Two-stage 1/2: full-image draft pass with the caption model",
            )
            if payload.caption_draft_refine_prompt:
                draft_prompt = (
                    f"{payload.caption_draft_refine_prompt.strip()}\n\n"
                    f"{prompt_text}\n"
                    "Respond with: DRAFT: <caption>"
                )
            else:
                draft_prompt = (
                    "Step 1: Look at the image and form a draft caption.\n"
                    f"{prompt_text}\n"
                    "Respond with: DRAFT: <caption>"
                )
            draft_system = f"{system_prompt} Return only a line starting with 'DRAFT:'."
            _qwen_progress_begin_output_section("Full-image draft output")
            draft_text, _, _ = _caption_visual_inference(
                draft_prompt,
                pil_img,
                max_new_tokens=max_new_tokens,
                system_prompt_override=draft_system,
                runtime_override=resolve_main_runtime(),
                decode_override=decode_params,
                chat_template_kwargs=caption_chat_template_kwargs,
            )
            _raise_if_qwen_cancelled()
            _record_first_stage_output("Full-image draft raw output", draft_text)
            draft_caption, _ = _extract_caption_from_text_impl(draft_text, marker="DRAFT")
            draft_caption = _sanitize_qwen_caption_impl(draft_caption)
            allowed_note = ""
            if restrict_to_labels and allowed_labels_prompt:
                allowed_note = f"Allowed classes: {', '.join(allowed_labels_prompt)}. Do not introduce any other entity types."
            elif not restrict_to_labels:
                allowed_note = "You may mention additional visible objects beyond the hints."
            first_stage_context = _format_caption_source_output_context_impl(
                source_outputs=first_stage_output_sections
            )
            refine_prompt = f"{prompt_text}\nDraft caption: {draft_caption}"
            if first_stage_context:
                refine_prompt = f"{refine_prompt}\n\n{first_stage_context}"
            if allowed_note:
                refine_prompt = f"{refine_prompt}\n{allowed_note}"
            if payload.caption_draft_refine_prompt:
                refine_prompt = (
                    f"{refine_prompt}\n"
                    f"{payload.caption_draft_refine_prompt.strip()} "
                    f"{final_caption_length_instruction}"
                    "Return only the final caption."
                )
            else:
                refine_prompt = (
                    f"{refine_prompt}\n"
                    "Edit the draft with minimal changes. Do not introduce new objects or actions. "
                    f"{broad_term_editor_note} "
                    f"{final_caption_length_instruction}"
                    "Return only the final caption."
                )
            refine_system = editor_system_prompt
            refine_model = caption_refinement_model_id
            _qwen_progress_update(
                phase="refine",
                phase_label="Refining",
                progress=0.78,
                message="Refining draft caption",
                model_id=refine_model,
                step_id="refine_draft",
                step_region=full_step_region,
                step_detail=(
                    f"Two-stage 2/2: editor pass with {refine_model}; "
                    "separate from the window merge pass"
                ),
            )
            _qwen_progress_begin_output_section("Draft refinement output")
            qwen_text, _, _ = _caption_text_inference(
                refine_prompt,
                max_new_tokens=refine_max_tokens,
                system_prompt_override=refine_system,
                model_id_override=refine_model,
                runtime_override=resolve_refinement_runtime(),
                decode_override=postprocess_decode_params,
                chat_template_kwargs=postprocess_chat_template_kwargs,
            )
            _raise_if_qwen_cancelled()
            previous_caption_text = draft_caption
            caption_text, _ = _extract_caption_from_text_impl(qwen_text, marker=None)
            caption_text = _accept_guard_caption(
                caption_text,
                previous_caption_text,
                stage="draft refinement",
            )
        else:
            _raise_if_qwen_cancelled()
            _qwen_progress_update(
                phase="generate",
                phase_label="Generating",
                progress=0.50,
                message=(
                    "Composing full-image caption from window observations without resending the full image"
                    if windowed_text_only_full_compose
                    else "Composing full-image caption from image and window observations"
                    if caption_mode == "windowed" and windowed_captions
                    else "Generating caption"
                ),
                model_id=(
                    caption_refinement_model_id
                    if windowed_text_only_full_compose
                    else desired_model_id
                ),
                step_id="generate_full",
                step_region=full_step_region,
                step_detail=(
                    "Text-only full-image composition from completed window observations"
                    if windowed_text_only_full_compose
                    else
                    f"Full-image composition using {len(windowed_captions)} window observation"
                    f"{'s' if len(windowed_captions) != 1 else ''}"
                    if caption_mode == "windowed"
                    else "Full-image composition pass"
                ),
            )
            _qwen_progress_begin_output_section("Full-image caption output")
            full_visual_text_recovery_prompt = None
            full_visual_text_recovery_after_safe_retry = False
            if caption_mode == "windowed" and windowed_captions:
                full_visual_text_recovery_prompt = (
                    f"{prompt_text}\n"
                    "Text-only recovery instruction: compose the final full-image caption "
                    "from the completed window observations, authoritative full-image counts, "
                    "class inventory, and user request above. Treat the observations as visual "
                    "evidence for the full image, deduplicate repeated crop-edge objects, and "
                    "do not mention recovery, labels, hints, boxes, coordinates, or prompts. "
                    "Return only the final caption."
                )
            elif counts or label_hints:
                full_visual_text_recovery_after_safe_retry = True
                full_visual_text_recovery_prompt = (
                    f"{prompt_text}\n"
                    "Text-only recovery instruction: the visual full-image caption could not complete safely after "
                    "the image was inspected. Compose the final caption from the authoritative counts, "
                    "class inventory, representative spatial layout, and user request above. Treat the "
                    "counts as truth, do not mention recovery, labels, hints, boxes, coordinates, or prompts, "
                    "and return only the final caption."
                )
            full_deterministic_recovery_caption = _caption_deterministic_recovery_caption(
                counts,
                label_hints,
                crop=False,
            )
            if windowed_text_only_full_compose and full_visual_text_recovery_prompt:
                qwen_text, _, _ = _caption_text_inference(
                    full_visual_text_recovery_prompt.replace(
                        "Text-only recovery instruction:",
                        "Text-only full-image composition instruction:",
                        1,
                    ),
                    max_new_tokens=refine_max_tokens,
                    system_prompt_override=editor_system_prompt,
                    model_id_override=caption_refinement_model_id,
                    runtime_override=resolve_refinement_runtime(),
                    decode_override=postprocess_decode_params,
                    chat_template_kwargs=postprocess_chat_template_kwargs,
                    deterministic_recovery_text=full_deterministic_recovery_caption,
                    deterministic_recovery_detail=(
                        "Text-only full-image composition looped, failed, or was unavailable; "
                        "the fallback used authoritative counts and relative object layout."
                    ),
                )
            else:
                qwen_text, _, _ = _caption_visual_inference(
                    prompt_text,
                    pil_img,
                    max_new_tokens=max_new_tokens,
                    system_prompt_override=system_prompt,
                    runtime_override=resolve_main_runtime(),
                    decode_override=decode_params,
                    chat_template_kwargs=caption_chat_template_kwargs,
                    text_recovery_prompt=full_visual_text_recovery_prompt,
                    text_recovery_system_prompt=editor_system_prompt,
                    text_recovery_model_id=caption_refinement_model_id,
                    text_recovery_runtime_resolver=resolve_refinement_runtime,
                    text_recovery_decode_override=postprocess_decode_params,
                    text_recovery_chat_template_kwargs=postprocess_chat_template_kwargs,
                    text_recovery_detail=(
                        "Primary visual full-image composition could not complete safely; this recovery path uses "
                        "the completed window observations or authoritative prompt context without "
                        "sending another full-image tensor."
                    ),
                    text_recovery_after_safe_retry=full_visual_text_recovery_after_safe_retry,
                    deterministic_recovery_text=full_deterministic_recovery_caption,
                    deterministic_recovery_detail=(
                        "All model-based full-image recovery paths looped, failed, or were unavailable; "
                        "the fallback used authoritative counts and relative object layout."
                    ),
                )
            _raise_if_qwen_cancelled()
            _record_first_stage_output("Full-image caption raw output", qwen_text)
            caption_text, _ = _extract_caption_from_text_impl(qwen_text, marker=None)
            caption_text = _sanitize_qwen_caption_impl(caption_text)
            if is_thinking and _thinking_caption_needs_cleanup_impl(caption_text, qwen_text):
                cleanup_model = caption_refinement_model_id
                _qwen_progress_update(
                    phase="cleanup",
                    phase_label="Cleaning Up",
                    progress=0.84,
                    message="Cleaning thinking output from full-image caption",
                    model_id=cleanup_model,
                    step_id="postprocess",
                    step_region=full_step_region,
                    step_detail="Cleaning thinking-model wrapper or reasoning leakage",
                )
                _qwen_progress_begin_output_section("Full-image cleanup output")
                caption_text = _caption_cleanup(
                    caption_text,
                    pil_img,
                    refine_max_tokens,
                    caption_base_model_id,
                    use_caption_cache,
                    model_id_override=cleanup_model,
                    runtime_override=resolve_refinement_runtime(),
                    allowed_labels=(
                        allowed_labels_prompt
                        if restrict_to_labels and allowed_labels_prompt
                        else None
                    ),
                    strict=True,
                    minimal_edit=True,
                    max_sentences=final_caption_max_sentences,
                    source_outputs=first_stage_output_sections,
                )
                cleanup_count += 1
        if (
            caption_mode == "windowed"
            and windowed_captions
            and caption_text
            and not windowed_text_only_full_compose
        ):
            merge_tokens = (
                refine_max_tokens
                if explicit_max_new_tokens
                else min(
                    QWEN_CAPTION_MAX_NEW_TOKENS,
                    max(
                        refine_max_tokens,
                        (
                            QWEN_CAPTION_MAX_NEW_TOKENS
                            if _qwen_model_id_is_thinking(caption_refinement_model_id)
                            else QWEN_CAPTION_INSTRUCT_MERGE_AUTO_MAX_NEW_TOKENS
                        ),
                    ),
                )
            )
            _qwen_progress_update(
                phase="merge",
                phase_label="Merging",
                progress=0.82,
                message="Merging window captions into the full-image caption",
                model_id=caption_refinement_model_id,
                step_id="merge_windows",
                step_region=full_step_region,
                step_detail=(
                    f"Windowed-only merge pass folding {len(windowed_captions)} window caption"
                    f"{'s' if len(windowed_captions) != 1 else ''} into the full-image caption; "
                    "text-only editor pass, separate from Draft + refine"
                ),
            )
            _qwen_progress_begin_output_section("Window merge output")
            caption_text = _caption_merge(
                caption_text,
                windowed_captions,
                pil_img=pil_img,
                base_model_id=caption_base_model_id,
                runtime_resolver=lambda _model_id: resolve_refinement_runtime(),
                max_new_tokens=merge_tokens,
                glossary_line=glossary_line or None,
                model_id_override=caption_refinement_model_id,
                max_sentences=final_caption_max_sentences,
                overlap_guidance=overlap_guidance or None,
                source_outputs=first_stage_output_sections,
                full_label_hints=label_hints,
                window_hints_by_window=grouped_hints,
                glossary_map=glossary_map,
            )
            _raise_if_qwen_cancelled()
            merge_count += 1
            caption_text = _sanitize_qwen_caption_impl(caption_text)
        _qwen_progress_update(
            phase="postprocess",
            phase_label="Postprocessing",
            progress=0.84,
            message="Checking caption cleanup, coverage, and language",
            step_id="postprocess",
            step_region=full_step_region,
            step_detail="Checking repetition, meta text, missing classes, and language.",
        )
        if _caption_is_degenerate_impl(caption_text):
            cleanup_model = caption_refinement_model_id
            _qwen_progress_update(
                phase="cleanup",
                phase_label="Cleaning Up",
                progress=0.86,
                message="Cleaning up generated caption",
                model_id=cleanup_model,
                step_id="postprocess",
                step_region=full_step_region,
                step_detail="Removing repeated or incomplete generated text",
            )
            _qwen_progress_begin_output_section("Degenerate-caption cleanup output")
            caption_text = _caption_cleanup(
                caption_text,
                pil_img,
                refine_max_tokens,
                caption_base_model_id,
                use_caption_cache,
                model_id_override=cleanup_model,
                runtime_override=resolve_refinement_runtime(),
                allowed_labels=(
                    allowed_labels_prompt if restrict_to_labels and allowed_labels_prompt else None
                ),
                strict=True,
                minimal_edit=True,
                max_sentences=final_caption_max_sentences,
                source_outputs=first_stage_output_sections,
            )
            _raise_if_qwen_cancelled()
            cleanup_count += 1
        if _caption_needs_completion_impl(caption_text) or _caption_has_meta_impl(caption_text):
            cleanup_model = caption_refinement_model_id
            _qwen_progress_update(
                phase="cleanup",
                phase_label="Cleaning Up",
                progress=0.88,
                message="Removing meta text from caption",
                model_id=cleanup_model,
                step_id="postprocess",
                step_region=full_step_region,
                step_detail="Removing meta text or completing an unfinished caption",
            )
            _qwen_progress_begin_output_section("Meta-text cleanup output")
            caption_text = _caption_cleanup(
                caption_text,
                pil_img,
                refine_max_tokens,
                caption_base_model_id,
                use_caption_cache,
                model_id_override=cleanup_model,
                runtime_override=resolve_refinement_runtime(),
                allowed_labels=(
                    allowed_labels_prompt if restrict_to_labels and allowed_labels_prompt else None
                ),
                strict=True,
                minimal_edit=True,
                max_sentences=final_caption_max_sentences,
                source_outputs=first_stage_output_sections,
            )
            _raise_if_qwen_cancelled()
            cleanup_count += 1
        if (
            caption_mode == "windowed"
            and "4B" in desired_model_id
            and _caption_needs_short_form_impl(
                caption_text,
                max_words=max(80, int(final_caption_max_sentences or 2) * 80),
                max_sentences=int(final_caption_max_sentences or 2),
            )
        ):
            cleanup_model = caption_refinement_model_id
            _qwen_progress_update(
                phase="cleanup",
                phase_label="Cleaning Up",
                progress=0.89,
                message="Shortening oversized windowed caption",
                model_id=cleanup_model,
                step_id="postprocess",
                step_region=full_step_region,
                step_detail="Shortening a windowed 4B caption that grew too long",
            )
            _qwen_progress_begin_output_section("Short-form cleanup output")
            caption_text = _caption_cleanup(
                caption_text,
                pil_img,
                refine_max_tokens,
                caption_base_model_id,
                use_caption_cache,
                model_id_override=cleanup_model,
                runtime_override=resolve_refinement_runtime(),
                allowed_labels=(
                    allowed_labels_prompt if restrict_to_labels and allowed_labels_prompt else None
                ),
                strict=True,
                minimal_edit=True,
                max_sentences=final_caption_max_sentences,
                source_outputs=first_stage_output_sections,
            )
            _raise_if_qwen_cancelled()
            cleanup_count += 1
        count_conflicts = (
            _caption_count_conflicts_impl(caption_text, counts, glossary_map)
            if include_counts
            else []
        )
        count_issues = sorted(
            set(
                count_conflicts
                + (
                    _caption_missing_exact_counts_impl(caption_text, counts, glossary_map)
                    if include_counts
                    else []
                )
            )
        )
        needs_refine, missing = _caption_needs_refine_impl(
            caption_text,
            counts,
            detailed_mode=detailed_mode,
            include_counts=include_counts,
            glossary_map=glossary_map,
        )
        if needs_refine:
            refine_model = caption_refinement_model_id
            missing_display = _format_refine_labels(missing, glossary_map)
            count_conflict_note = _format_count_conflict_note(
                count_issues,
                counts,
                glossary_map,
            )
            _qwen_progress_update(
                phase="guard",
                phase_label="Guarding",
                progress=0.90,
                message="Applying final coverage/count guard",
                model_id=refine_model,
                step_id="postprocess",
                step_region=full_step_region,
                step_detail=(
                    f"Ensuring mentioned classes: {', '.join(missing_display)}"
                    if missing_display
                    else "Ensuring all labeled classes are mentioned"
                ),
            )
            allowed_note = ""
            if restrict_to_labels and allowed_labels_prompt:
                allowed_note = f"Allowed classes: {', '.join(allowed_labels_prompt)}. Do not introduce any other entity types."
            elif not restrict_to_labels:
                allowed_note = "You may mention additional visible objects beyond the hints."
            refine_notes = []
            if missing_display:
                refine_notes.append(
                    f"Ensure the caption mentions: {', '.join(missing_display)}."
                )
            if count_conflict_note:
                refine_notes.append(count_conflict_note)
            if not refine_notes:
                refine_notes.append("Ensure all labeled classes are mentioned.")
            missing_note = " ".join(refine_notes)
            refine_prompt = f"{prompt_text}\nDraft caption: {caption_text}\n{missing_note}"
            if first_stage_output_sections:
                first_stage_context = _format_caption_source_output_context_impl(
                    source_outputs=first_stage_output_sections
                )
                if first_stage_context:
                    refine_prompt = f"{refine_prompt}\n\n{first_stage_context}"
            if allowed_note:
                refine_prompt = f"{refine_prompt}\n{allowed_note}"
            refine_prompt = f"{refine_prompt}\n{coverage_policy}"
            if payload.caption_cleanup_prompt:
                refine_prompt = (
                    f"{refine_prompt}\n"
                    f"{payload.caption_cleanup_prompt.strip()} "
                    f"{final_caption_length_instruction}"
                    "Return only the final caption with no coordinates."
                )
            else:
                refine_prompt = (
                    f"{refine_prompt}\n"
                    "Edit the draft with minimal changes. Do not introduce new objects or actions. "
                    f"{broad_term_editor_note} "
                    f"{final_caption_length_instruction}"
                    "Return only the final caption with no coordinates."
                )
            refine_system = editor_system_prompt
            _qwen_progress_begin_output_section("Coverage/count guard output")
            refine_text, _, _ = _caption_text_inference(
                refine_prompt,
                max_new_tokens=refine_max_tokens,
                system_prompt_override=refine_system,
                model_id_override=refine_model,
                runtime_override=resolve_refinement_runtime(),
                decode_override=postprocess_decode_params,
                chat_template_kwargs=postprocess_chat_template_kwargs,
            )
            _raise_if_qwen_cancelled()
            previous_caption_text = caption_text
            caption_text, _ = _extract_caption_from_text_impl(refine_text, marker=None)
            caption_text = _accept_guard_caption(
                caption_text,
                previous_caption_text,
                stage="coverage/count guard",
            )
            refine_count += 1
        if caption_text and _caption_needs_english_rewrite_impl(caption_text):
            rewrite_model = caption_refinement_model_id
            _qwen_progress_update(
                phase="rewrite",
                phase_label="Rewriting",
                progress=0.92,
                message="Rewriting caption in English",
                model_id=rewrite_model,
                step_id="postprocess",
                step_region=full_step_region,
                step_detail="Rewriting non-English output in English",
            )
            rewrite_prompt = (
                f"{language_rewrite_policy} "
                f"{(payload.caption_cleanup_prompt.strip() + ' ') if payload.caption_cleanup_prompt else broad_term_editor_note + ' '}"
                f"{final_caption_length_instruction}\n"
                f"Caption: {caption_text}"
            )
            if first_stage_output_sections:
                first_stage_context = _format_caption_source_output_context_impl(
                    source_outputs=first_stage_output_sections
                )
                if first_stage_context:
                    rewrite_prompt = f"{rewrite_prompt}\n\n{first_stage_context}"
            rewrite_system = editor_system_prompt
            _qwen_progress_begin_output_section("English rewrite output")
            rewrite_text, _, _ = _caption_text_inference(
                rewrite_prompt,
                max_new_tokens=refine_max_tokens,
                system_prompt_override=rewrite_system,
                model_id_override=rewrite_model,
                runtime_override=resolve_refinement_runtime(),
                decode_override=postprocess_decode_params,
                chat_template_kwargs=postprocess_chat_template_kwargs,
            )
            _raise_if_qwen_cancelled()
            previous_caption_text = caption_text
            caption_text, _ = _extract_caption_from_text_impl(rewrite_text, marker=None)
            caption_text = _accept_guard_caption(
                caption_text,
                previous_caption_text,
                stage="English rewrite",
            )
        caption_text = _sanitize_qwen_caption_impl(caption_text)
        if _caption_needs_completion_impl(caption_text):
            repair_model = caption_refinement_model_id
            _qwen_progress_update(
                phase="cleanup",
                phase_label="Repairing",
                progress=0.94,
                message="Repairing truncated caption ending",
                model_id=repair_model,
                step_id="postprocess",
                step_region=full_step_region,
                step_detail=(
                    "Rewriting the generated caption into complete sentences before finalization"
                ),
            )
            _qwen_progress_begin_output_section("Truncated-caption repair output")
            repaired_caption = _caption_cleanup(
                caption_text,
                pil_img,
                min(refine_max_tokens, QWEN_CAPTION_MAX_NEW_TOKENS),
                caption_base_model_id,
                use_caption_cache,
                model_id_override=repair_model,
                runtime_override=resolve_refinement_runtime(),
                allowed_labels=(
                    allowed_labels_prompt if restrict_to_labels and allowed_labels_prompt else None
                ),
                strict=True,
                minimal_edit=False,
                max_sentences=final_caption_max_sentences,
            )
            _raise_if_qwen_cancelled()
            if repaired_caption:
                caption_text = _sanitize_qwen_caption_impl(repaired_caption)
            cleanup_count += 1
        missing_exact_counts_before_repair = _caption_missing_exact_counts_impl(
            caption_text,
            counts,
            glossary_map,
        )
        count_repaired_caption = _caption_repair_count_text_artifacts_impl(
            caption_text,
            counts,
            glossary_map,
        )
        sanitized_before_count_repair = _sanitize_qwen_caption_impl(caption_text)
        if count_repaired_caption != sanitized_before_count_repair:
            _qwen_caption_io_record(
                {
                    "event": "count_text_repair",
                    "missing_exact_counts": missing_exact_counts_before_repair,
                    "previous_caption": caption_text,
                    "repaired_caption": count_repaired_caption,
                }
            )
            caption_text = count_repaired_caption
            _qwen_progress_update(
                phase="postprocess",
                phase_label="Finalizing",
                progress=0.943,
                message="Repaired count wording in final caption",
                token_preview=caption_text,
                step_id="postprocess",
                step_region=full_step_region,
                step_detail="Final guard repaired authoritative count wording after editor output.",
            )
        caption_text = _sanitize_qwen_caption_impl(_stabilize_glossary_subtypes(caption_text))
        caption_text, trimmed_incomplete_tail = _caption_trim_to_complete_sentences_impl(
            caption_text,
            max_sentences=final_caption_max_sentences,
        )
        if trimmed_incomplete_tail:
            _qwen_progress_update(
                phase="postprocess",
                phase_label="Finalizing",
                progress=0.945,
                message="Removed incomplete trailing caption text",
                token_preview=caption_text,
                step_id="postprocess",
                step_region=full_step_region,
                step_detail="Final guard trimmed a dangling partial sentence.",
            )
        post_trim_count_repaired_caption = _caption_repair_count_text_artifacts_impl(
            caption_text,
            counts,
            glossary_map,
        )
        sanitized_before_post_trim_count_repair = _sanitize_qwen_caption_impl(caption_text)
        if post_trim_count_repaired_caption != sanitized_before_post_trim_count_repair:
            _qwen_caption_io_record(
                {
                    "event": "count_text_repair",
                    "phase": "after_sentence_trim",
                    "previous_caption": caption_text,
                    "repaired_caption": post_trim_count_repaired_caption,
                }
            )
            caption_text = post_trim_count_repaired_caption
            _qwen_progress_update(
                phase="postprocess",
                phase_label="Finalizing",
                progress=0.946,
                message="Repaired count wording after sentence trimming",
                token_preview=caption_text,
                step_id="postprocess",
                step_region=full_step_region,
                step_detail="Final guard restored authoritative count wording after sentence trimming.",
            )
            caption_text, _ = _caption_trim_to_complete_sentences_impl(
                caption_text,
                max_sentences=final_caption_max_sentences,
            )
        response = QwenCaptionResponse(
            caption=caption_text,
            used_counts=counts,
            used_boxes=used_boxes,
            truncated=truncated,
            recovery_events=caption_recovery_events,
        )
        _qwen_progress_begin_output_section("Final caption")
        _qwen_progress_update(
            phase="postprocess",
            phase_label="Finalizing",
            progress=0.95,
            message="Finalizing caption",
            token_preview=caption_text,
            append_token_preview_to_live_output=True,
            step_id="finalize",
            step_region=full_step_region,
            step_detail="Caption is being copied into the response",
        )
        word_count = len(caption_text.split()) if caption_text else 0
        logger.info(
            "[qwen-caption] hints=%s used=%s truncated=%s variant=%s model=%s refinement_model=%s final_only=%s windows=%s cleanup=%s refine=%s merge=%s words=%s",
            len(payload.label_hints or []),
            used_boxes,
            truncated,
            variant,
            desired_model_id,
            caption_refinement_model_id,
            final_only,
            len(windowed_captions) if caption_mode == "windowed" else 0,
            cleanup_count,
            refine_count,
            merge_count,
            word_count,
        )
    except QwenCancellationRequested as exc:
        _qwen_progress_cancelled("Qwen caption cancelled")
        if force_unload:
            request_model_cache.clear()
            _unload_qwen_runtime()
            active_runtime = None
            active_model_id = None
        raise HTTPException(status_code=499, detail="qwen_caption_cancelled") from exc
    except QwenCaptionLoopDetected as exc:
        _qwen_progress_error(
            "Qwen caption stopped because the model entered a repeated output loop. "
            "The Qwen runtime was unloaded to avoid a follow-up Metal crash."
        )
        logger.warning("[qwen-caption] repeated output loop -> forcing Qwen runtime unload")
        request_model_cache.clear()
        _unload_qwen_runtime()
        active_runtime = None
        active_model_id = None
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="qwen_caption_repetition_loop",
        ) from exc
    except HTTPException as exc:
        detail = _http_exception_detail_text(exc)
        _qwen_progress_error(
            f"Qwen caption failed: {detail}" if detail else "Qwen caption failed"
        )
        if force_unload:
            logger.warning("[qwen-caption] exception -> forcing unload")
            request_model_cache.clear()
            _unload_qwen_runtime()
            active_runtime = None
            active_model_id = None
        raise
    except Exception as exc:  # noqa: BLE001
        _qwen_progress_error(f"Qwen caption failed: {exc}")
        if force_unload:
            logger.warning("[qwen-caption] exception=%s -> forcing unload", exc)
            request_model_cache.clear()
            _unload_qwen_runtime()
            active_runtime = None
            active_model_id = None
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_caption_failed:{exc}"
        ) from exc
    if force_unload:
        _qwen_progress_update(
            phase="unload",
            phase_label="Unloading",
            progress=0.98,
            message="Unloading Qwen runtime after caption",
            step_id="unload_model",
            step_region=full_step_region,
            step_detail="Releasing Qwen model memory after captioning",
        )
        request_model_cache.clear()
        _unload_qwen_runtime()
        active_runtime = None
        active_model_id = None
        _qwen_progress_update(
            phase="postprocess",
            phase_label="Finalizing",
            progress=0.995,
            message="Caption ready",
            token_preview=response.caption,
            step_id="finalize",
            step_region=full_step_region,
            step_detail="Caption response is ready",
        )
    _qwen_progress_finish("Caption ready", token_preview=response.caption)
    return response


REQUIRED_BINDINGS = ('QWEN_DEVICE_PREF', 'QWEN_MODEL_NAME', '_CAPTION_WINDOW_HOOK', '_agent_full_trace_write', '_default_variant', '_effective_qwen_model_id_for_platform', '_ensure_qwen_ready', '_ensure_qwen_ready_for_caption', '_predict_with_cache', '_qwen_cancel_requested', '_qwen_caption_io_input', '_qwen_caption_io_output', '_qwen_caption_io_prompt_budget', '_qwen_caption_io_record', '_qwen_caption_loop_cooldown_remaining', '_qwen_mlx_post_generation_cleanup', '_qwen_model_local_state', '_qwen_progress_begin_output_section', '_qwen_progress_cancelled', '_qwen_progress_error', '_qwen_progress_finish', '_qwen_progress_start', '_qwen_progress_token', '_qwen_progress_update', '_raise_if_qwen_cancelled', '_resolve_qwen_runtime_platform', '_run_qwen_chat_mlx', '_run_qwen_inference_mlx', '_suspend_clip_backbone', '_unload_detector_inference', '_unload_dinov3_backbone', '_unload_non_qwen_runtimes_impl', '_unload_qwen_runtime', '_unload_sam3_text_runtime', 'active_qwen_metadata', 'active_qwen_model_id', 'active_qwen_model_path', 'predictor_manager', 'process_vision_info', 'qwen_caption_loop_cooldowns', 'qwen_config_lock', 'qwen_device', 'qwen_llm_call_id_var', 'qwen_progress_lock', 'qwen_progress_state', 'qwen_prompt_config', 'resolve_image_payload')
EXTRACTED_FUNCTIONS = ('_apply_qwen_chat_template', '_build_caption_overlap_guidance', '_build_qwen_caption_step_plan', '_caption_explicit_token_budget_active', '_caption_position_phrase_for_bbox', '_caption_window_detection_context_prompt', '_emit_caption_window', '_estimate_qwen_prompt_tokens', '_extract_qwen_json_block', '_format_caption_window_sentence_instruction', '_format_qwen_caption_prompt_preview_text', '_group_hints_by_window', '_http_exception_detail_text', '_normalize_caption_hints_for_decoded_image', '_normalize_qwen_caption_loop_recovery_mode', '_qwen_bbox_results', '_qwen_bbox_sam_results', '_qwen_caption_compact_loop_salvage_text', '_qwen_caption_compact_original_prompt_context', '_qwen_caption_default_refinement_model_id', '_qwen_caption_generation_active', '_qwen_caption_known_model_ids', '_qwen_caption_loop_output_salvageable', '_qwen_caption_preview_section_prompt_text', '_qwen_caption_prompt_aware_max_new_tokens', '_qwen_caption_prompt_preview_budget', '_qwen_caption_prompt_preview_section', '_qwen_caption_recovery_key', '_qwen_caption_safe_retry_decode', '_qwen_caption_safe_retry_tokens', '_qwen_caption_step_entry', '_qwen_caption_stream_loop_detected_event', '_qwen_disable_thinking_requested', '_qwen_items_from_payload', '_qwen_maybe_trim_repetition_loop_output', '_qwen_messages_with_no_think', '_qwen_model_id_is_thinking', '_qwen_neutralize_thinking_prefill', '_qwen_point_results', '_qwen_runtime_model_id', '_qwen_runtime_platform_from_runtime', '_resolve_qwen_caption_default_model_id', '_resolve_qwen_caption_max_new_tokens', '_resolve_qwen_caption_refinement_model_id', '_run_qwen_chat', '_run_qwen_inference', '_run_qwen_text_inference', 'qwen_caption', 'qwen_caption_prompt_preview', 'qwen_infer')


def bind_pipeline(bindings):
    """Bind explicit trusted worker services without mutating module globals."""
    missing = set(REQUIRED_BINDINGS) - set(bindings)
    if missing:
        raise ValueError("hosted_qwen_dependencies_missing:" + ",".join(sorted(missing)))
    allowed = set(REQUIRED_BINDINGS) | set(EXTRACTED_FUNCTIONS)
    if set(bindings) - allowed:
        raise ValueError("hosted_qwen_dependencies_unknown:" + ",".join(sorted(set(bindings)-allowed)))
    namespace = dict(globals())
    for name in EXTRACTED_FUNCTIONS:
        original = globals()[name]
        bound = FunctionType(original.__code__, namespace, name, original.__defaults__, original.__closure__)
        bound.__kwdefaults__ = original.__kwdefaults__
        namespace[name] = bound
    namespace.update(bindings)
    return namespace
