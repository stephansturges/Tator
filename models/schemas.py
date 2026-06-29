"""Shared Pydantic schemas (requests/responses)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, Literal

from pydantic import BaseModel, Field

from utils.pydantic_compat import root_validator_compat


QWEN_CAPTION_SET_AND_FORGET_MAX_LOOP_RECOVERY_RATE = 0.05
QWEN_CAPTION_SET_AND_FORGET_MAX_DETERMINISTIC_RECOVERY_RATE = 0.01
QWEN_CAPTION_SET_AND_FORGET_MAX_SIGNAL_EXIT_ATTEMPT_RATE = 0.05
QWEN_CAPTION_SET_AND_FORGET_ATTEMPTS = 3
QWEN_CAPTION_SET_AND_FORGET_INSTRUCTION_MAX_FAILURES = 1
QWEN_CAPTION_DEFAULT_PILOT_MIN_CASES = 300
QWEN_CAPTION_DEFAULT_PILOT_DETERMINISTIC_RECOVERY_CONFIDENCE = 0.95


class Base64Payload(BaseModel):
    image_base64: str
    image_token: Optional[str] = None
    uuid: Optional[str] = None
    background_guard: Optional[bool] = None
    bbox_xyxy: Optional[List[float]] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class PredictResponse(BaseModel):
    prediction: str
    proba: Optional[float] = None
    second_label: Optional[str] = None
    second_proba: Optional[float] = None
    margin: Optional[float] = None
    error: Optional[str] = None
    uuid: Optional[str] = None


class BboxModel(BaseModel):
    className: str
    x: float
    y: float
    width: float
    height: float


class CropImage(BaseModel):
    image_base64: str
    originalName: str
    bboxes: List[BboxModel]


class CropZipRequest(BaseModel):
    images: List[CropImage]


class PointPrompt(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    point_x: float
    point_y: float
    uuid: Optional[str] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None

    @root_validator_compat(skip_on_failure=True)
    def _ensure_point_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        return values


class BboxPrompt(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    uuid: Optional[str] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None

    @root_validator_compat(skip_on_failure=True)
    def _ensure_bbox_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        return values


class SamPreloadRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    sam_variant: Optional[str] = None
    preload_generation: Optional[int] = None
    image_name: Optional[str] = None
    slot: Optional[str] = "current"

    @root_validator_compat(skip_on_failure=True)
    def _ensure_preload_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        if values.get("slot") and values.get("slot") != "current" and not values.get("image_name"):
            raise ValueError("image_name_required_for_slot")
        return values


class SamPreloadResponse(BaseModel):
    status: str = "ready"
    width: int
    height: int
    token: str


class SamSlotStatus(BaseModel):
    slot: str
    image_name: Optional[str]
    token: Optional[str]
    variant: Optional[str]
    width: Optional[int]
    height: Optional[int]
    busy: bool
    last_loaded: float
    enabled: bool = True
    memory_bytes: Optional[int] = None


class SamActivateRequest(BaseModel):
    image_name: str
    sam_variant: Optional[str] = None


class SamActivateResponse(BaseModel):
    status: str
    slot: Optional[str] = None
    token: Optional[str] = None


class PredictorSettings(BaseModel):
    max_predictors: int
    min_predictors: int
    max_supported_predictors: int
    active_predictors: int
    loaded_predictors: int
    process_ram_mb: float
    total_ram_mb: float
    available_ram_mb: float
    image_ram_mb: float
    gpu_total_mb: Optional[float] = None
    gpu_free_mb: Optional[float] = None
    gpu_compute_capability: Optional[str] = None
    gpu_device_count: Optional[int] = None


class Sam3VisualPrompt(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    bbox_left: Optional[float] = None
    bbox_top: Optional[float] = None
    bbox_width: Optional[float] = None
    bbox_height: Optional[float] = None
    bboxes: Optional[List[List[float]]] = None
    bbox_labels: Optional[List[bool]] = None
    text_prompt: Optional[str] = None
    prompt: Optional[str] = None
    threshold: float = 0.55
    mask_threshold: float = 0.2
    simplify_epsilon: Optional[float] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None
    max_results: Optional[int] = None
    min_size: Optional[int] = None

    @root_validator_compat(skip_on_failure=True)
    def _validate_visual_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        raw_bboxes = values.get("bboxes")
        cleaned_bboxes: List[Tuple[float, float, float, float]] = []
        if isinstance(raw_bboxes, list) and raw_bboxes:
            for entry in raw_bboxes:
                coords = None
                if isinstance(entry, Mapping):
                    coords = [
                        entry.get("left", entry.get("x")),
                        entry.get("top", entry.get("y")),
                        entry.get("width", entry.get("w")),
                        entry.get("height", entry.get("h")),
                    ]
                else:
                    coords = entry
                if not isinstance(coords, (list, tuple)) or len(coords) < 4:
                    continue
                try:
                    cleaned = tuple(float(coords[idx]) for idx in range(4))
                except (TypeError, ValueError):
                    continue
                if cleaned[2] <= 0 or cleaned[3] <= 0:
                    continue
                cleaned_bboxes.append(cleaned)
        if cleaned_bboxes:
            values["bboxes"] = cleaned_bboxes
        else:
            values["bboxes"] = None
            for key in ("bbox_left", "bbox_top", "bbox_width", "bbox_height"):
                raw = values.get(key)
                try:
                    values[key] = float(raw)
                except (TypeError, ValueError):
                    raise ValueError(f"invalid_{key}") from None
            if values["bbox_width"] <= 0 or values["bbox_height"] <= 0:
                raise ValueError("invalid_bbox_dims")
        raw_labels = values.get("bbox_labels")
        cleaned_labels: Optional[List[bool]] = None
        if isinstance(raw_labels, (list, tuple)):
            cleaned_labels = []
            for entry in raw_labels:
                if isinstance(entry, bool):
                    cleaned_labels.append(entry)
                elif isinstance(entry, (int, float)):
                    cleaned_labels.append(bool(entry))
                elif isinstance(entry, str):
                    cleaned_labels.append(entry.strip().lower() in {"1", "true", "yes", "pos", "positive"})
                else:
                    cleaned_labels.append(True)
        if values.get("bboxes"):
            if cleaned_labels is None:
                values["bbox_labels"] = None
            else:
                if len(cleaned_labels) < len(values["bboxes"]):
                    cleaned_labels.extend([True] * (len(values["bboxes"]) - len(cleaned_labels)))
                values["bbox_labels"] = cleaned_labels[: len(values["bboxes"])]
        else:
            values["bbox_labels"] = None
        min_size = values.get("min_size")
        if min_size is not None:
            try:
                values["min_size"] = max(0, int(min_size))
            except (TypeError, ValueError):
                values["min_size"] = 0
        eps = values.get("simplify_epsilon")
        if eps is not None:
            try:
                eps_val = float(eps)
            except (TypeError, ValueError):
                eps_val = None
            values["simplify_epsilon"] = eps_val if eps_val is None or eps_val >= 0 else 0.0
        prompt = values.get("text_prompt") or values.get("prompt")
        if prompt is not None:
            prompt = str(prompt).strip()
            values["text_prompt"] = prompt or None
        return values


class SamPointAutoResponse(BaseModel):
    prediction: Optional[str] = None
    proba: Optional[float] = None
    second_label: Optional[str] = None
    second_proba: Optional[float] = None
    margin: Optional[float] = None
    bbox: List[float]
    uuid: Optional[str] = None
    error: Optional[str] = None
    image_token: Optional[str] = None
    score: Optional[float] = None
    mask: Optional[Dict[str, Any]] = None
    simplify_epsilon: Optional[float] = None


class QwenDetection(BaseModel):
    bbox: List[float]
    qwen_label: Optional[str] = None
    source: Literal["bbox", "point", "bbox_sam", "sam3_text"]
    score: Optional[float] = None
    mask: Optional[Dict[str, Any]] = None
    simplify_epsilon: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    clip_head_prob: Optional[float] = None
    clip_head_margin: Optional[float] = None
    clip_head_bg_prob: Optional[float] = None
    clip_head_bg_margin: Optional[float] = None


class QwenInferenceRequest(BaseModel):
    prompt: Optional[str] = None
    item_list: Optional[str] = None
    image_type: Optional[str] = None
    extra_context: Optional[str] = None
    prompt_type: Literal["bbox", "point", "bbox_sam"] = "bbox"
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None
    max_results: Optional[int] = 8

    @root_validator_compat(skip_on_failure=True)
    def _validate_qwen_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        prompt = (values.get("prompt") or "").strip()
        items = (values.get("item_list") or "").strip()
        if prompt:
            values["prompt"] = prompt
        elif items:
            values["item_list"] = items
        else:
            raise ValueError("prompt_or_items_required")
        max_results = values.get("max_results")
        if max_results is not None:
            try:
                max_int = int(max_results)
            except (TypeError, ValueError):
                max_int = 8
            values["max_results"] = max(1, min(max_int, 50))
        else:
            values["max_results"] = 8
        return values


class QwenInferenceResponse(BaseModel):
    boxes: List[QwenDetection] = Field(default_factory=list)
    raw_response: str
    prompt: str
    prompt_type: Literal["bbox", "point", "bbox_sam"]
    warnings: List[str] = Field(default_factory=list)
    image_token: Optional[str] = None


class QwenCaptionHint(BaseModel):
    label: str
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    source_id: Optional[str] = None

    @root_validator_compat(skip_on_failure=True)
    def _validate_hint(cls, values):  # noqa: N805
        label = (values.get("label") or "").strip()
        if not label:
            raise ValueError("label_required")
        values["label"] = label
        source_id = str(values.get("source_id") or "").strip()
        values["source_id"] = source_id or None
        bbox = values.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError("invalid_bbox")
            cleaned = []
            for val in bbox:
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    raise ValueError("invalid_bbox") from None
                if not math.isfinite(num):
                    raise ValueError("invalid_bbox")
                cleaned.append(num)
            values["bbox"] = cleaned
        confidence = values.get("confidence")
        if confidence is not None:
            try:
                confidence_val = float(confidence)
            except (TypeError, ValueError):
                confidence_val = None
            values["confidence"] = (
                confidence_val if confidence_val is not None and math.isfinite(confidence_val) else None
            )
        return values


class QwenCaptionRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    image_name: Optional[str] = None
    user_prompt: Optional[str] = None
    label_hints: Optional[List[QwenCaptionHint]] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    include_counts: Optional[bool] = True
    include_coords: Optional[bool] = True
    max_boxes: Optional[int] = 0
    max_new_tokens: Optional[int] = None
    model_variant: Optional[Literal["auto", "Instruct", "Thinking"]] = "auto"
    model_id: Optional[str] = None
    refinement_model_id: Optional[str] = None
    caption_loop_recovery_mode: Optional[
        Literal["off", "safe_retry", "safe_retry_fallback"]
    ] = "safe_retry_fallback"
    caption_fallback_model_id: Optional[str] = None
    caption_loop_cooldown: Optional[bool] = True
    final_answer_only: Optional[bool] = True
    final_caption_max_sentences: Optional[int] = 10
    two_stage_refine: Optional[bool] = False
    caption_mode: Optional[Literal["full", "windowed"]] = "full"
    caption_windowed_full_image_strategy: Optional[Literal["visual", "text_only"]] = "visual"
    window_size: Optional[int] = None
    window_overlap: Optional[float] = None
    caption_window_min_sentences: Optional[int] = 1
    caption_window_max_sentences: Optional[int] = 3
    restrict_to_labels: Optional[bool] = True
    caption_all_windows: Optional[bool] = None
    unload_others: Optional[bool] = False
    force_unload: Optional[bool] = None
    multi_model_cache: Optional[bool] = False
    fast_mode: Optional[bool] = False
    use_sampling: Optional[bool] = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    caption_system_prompt: Optional[str] = None
    caption_detection_context_prompt: Optional[str] = None
    caption_window_prompt: Optional[str] = None
    caption_draft_refine_prompt: Optional[str] = None
    caption_merge_prompt: Optional[str] = None
    caption_cleanup_prompt: Optional[str] = None
    caption_editor_system_prompt: Optional[str] = None
    caption_coverage_prompt: Optional[str] = None
    caption_language_rewrite_prompt: Optional[str] = None
    labelmap_glossary: Optional[str] = None

    @root_validator_compat(pre=True)
    def _normalize_caption_payload_input(cls, values):  # noqa: N805
        if not isinstance(values, Mapping):
            return values
        data = dict(values)
        caption_mode = str(data.get("caption_mode") or "full").strip().lower()
        if caption_mode == "hybrid":
            caption_mode = "windowed"
        if caption_mode not in {"full", "windowed"}:
            caption_mode = "full"
        data["caption_mode"] = caption_mode
        strategy = (
            str(data.get("caption_windowed_full_image_strategy") or "visual")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if strategy in {"text", "text_only", "window_text", "windowed_text", "observations", "skip_visual"}:
            strategy = "text_only"
        elif strategy not in {"visual", "text_only"}:
            strategy = "visual"
        data["caption_windowed_full_image_strategy"] = strategy
        data["model_variant"] = _normalize_qwen_variant_value(
            data.get("model_variant"),
            default="auto",
        )
        recovery_mode = str(
            data.get("caption_loop_recovery_mode") or "safe_retry_fallback"
        ).strip().lower()
        if recovery_mode not in {"off", "safe_retry", "safe_retry_fallback"}:
            recovery_mode = "safe_retry_fallback"
        data["caption_loop_recovery_mode"] = recovery_mode
        fallback_model_id = (data.get("caption_fallback_model_id") or "").strip()
        if fallback_model_id.lower() in {"", "auto", "none", "active"}:
            fallback_model_id = ""
        data["caption_fallback_model_id"] = fallback_model_id or None
        for field in (
            "image_width",
            "image_height",
            "max_boxes",
            "max_new_tokens",
            "final_caption_max_sentences",
            "temperature",
            "top_p",
            "top_k",
            "presence_penalty",
            "window_size",
            "window_overlap",
            "caption_window_min_sentences",
            "caption_window_max_sentences",
        ):
            if data.get(field) is None:
                continue
            try:
                numeric_value = float(data.get(field))
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(numeric_value):
                data[field] = None
        for field in ("image_token", "image_name"):
            if data.get(field) is not None:
                data[field] = str(data.get(field)).strip() or None
        return data

    @root_validator_compat(skip_on_failure=True)
    def _validate_caption_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        user_prompt = (values.get("user_prompt") or "").strip()
        values["user_prompt"] = user_prompt
        for prompt_field in (
            "caption_system_prompt",
            "caption_detection_context_prompt",
            "caption_window_prompt",
            "caption_draft_refine_prompt",
            "caption_merge_prompt",
            "caption_cleanup_prompt",
            "caption_editor_system_prompt",
            "caption_coverage_prompt",
            "caption_language_rewrite_prompt",
        ):
            values[prompt_field] = (values.get(prompt_field) or "").strip()
        max_boxes = values.get("max_boxes")
        if max_boxes is not None:
            try:
                max_boxes_val = int(max_boxes)
            except (TypeError, ValueError, OverflowError):
                max_boxes_val = 0
            values["max_boxes"] = max(0, min(max_boxes_val, 200))
        else:
            values["max_boxes"] = 0
        max_tokens = values.get("max_new_tokens")
        if max_tokens is not None:
            try:
                max_tokens_val = int(max_tokens)
            except (TypeError, ValueError, OverflowError):
                max_tokens_val = 0
            values["max_new_tokens"] = max(32, min(max_tokens_val, 4096)) if max_tokens_val > 0 else None
        else:
            values["max_new_tokens"] = None
        max_sentences = values.get("final_caption_max_sentences")
        if max_sentences is not None:
            try:
                max_sentences_val = int(max_sentences)
            except (TypeError, ValueError, OverflowError):
                max_sentences_val = 0
            values["final_caption_max_sentences"] = (
                max(1, min(max_sentences_val, 30)) if max_sentences_val > 0 else 10
            )
        else:
            values["final_caption_max_sentences"] = 10
        window_min = values.get("caption_window_min_sentences")
        try:
            window_min_val = int(window_min) if window_min is not None else 1
        except (TypeError, ValueError, OverflowError):
            window_min_val = 1
        window_min_val = max(1, min(window_min_val, 10))
        window_max = values.get("caption_window_max_sentences")
        try:
            window_max_val = int(window_max) if window_max is not None else 3
        except (TypeError, ValueError, OverflowError):
            window_max_val = 3
        window_max_val = max(window_min_val, min(window_max_val, 10))
        values["caption_window_min_sentences"] = window_min_val
        values["caption_window_max_sentences"] = window_max_val
        temp = values.get("temperature")
        if temp is not None:
            try:
                temp_val = float(temp)
                values["temperature"] = temp_val if math.isfinite(temp_val) else None
            except (TypeError, ValueError, OverflowError):
                values["temperature"] = None
        top_p = values.get("top_p")
        if top_p is not None:
            try:
                top_p_val = float(top_p)
                values["top_p"] = top_p_val if math.isfinite(top_p_val) else None
            except (TypeError, ValueError, OverflowError):
                values["top_p"] = None
        top_k = values.get("top_k")
        if top_k is not None:
            try:
                values["top_k"] = int(top_k)
            except (TypeError, ValueError, OverflowError):
                values["top_k"] = None
        presence = values.get("presence_penalty")
        if presence is not None:
            try:
                presence_val = float(presence)
                values["presence_penalty"] = presence_val if math.isfinite(presence_val) else None
            except (TypeError, ValueError, OverflowError):
                values["presence_penalty"] = None
        for key in ("image_width", "image_height"):
            val = values.get(key)
            if val is None:
                continue
            try:
                values[key] = max(1, int(val))
            except (TypeError, ValueError, OverflowError):
                values[key] = None
        caption_mode = (values.get("caption_mode") or "full").strip().lower()
        if caption_mode == "hybrid":
            caption_mode = "windowed"
        if caption_mode not in {"full", "windowed"}:
            caption_mode = "full"
        values["caption_mode"] = caption_mode
        strategy = (
            str(values.get("caption_windowed_full_image_strategy") or "visual")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if strategy in {"text", "text_only", "window_text", "windowed_text", "observations", "skip_visual"}:
            strategy = "text_only"
        elif strategy not in {"visual", "text_only"}:
            strategy = "visual"
        values["caption_windowed_full_image_strategy"] = strategy
        window_size = values.get("window_size")
        if window_size is not None:
            try:
                values["window_size"] = max(64, int(window_size))
            except (TypeError, ValueError, OverflowError):
                values["window_size"] = None
        window_overlap = values.get("window_overlap")
        if window_overlap is not None:
            try:
                overlap_val = float(window_overlap)
                values["window_overlap"] = overlap_val if math.isfinite(overlap_val) else None
            except (TypeError, ValueError, OverflowError):
                values["window_overlap"] = None
        model_id = (values.get("model_id") or "").strip()
        values["model_id"] = model_id or None
        refinement_model_id = (values.get("refinement_model_id") or "").strip()
        values["refinement_model_id"] = refinement_model_id or None
        recovery_mode = str(
            values.get("caption_loop_recovery_mode") or "safe_retry_fallback"
        ).strip().lower()
        if recovery_mode not in {"off", "safe_retry", "safe_retry_fallback"}:
            recovery_mode = "safe_retry_fallback"
        values["caption_loop_recovery_mode"] = recovery_mode
        fallback_model_id = (values.get("caption_fallback_model_id") or "").strip()
        if fallback_model_id.lower() in {"", "auto", "none", "active"}:
            fallback_model_id = ""
        values["caption_fallback_model_id"] = fallback_model_id or None
        glossary = values.get("labelmap_glossary")
        if glossary is not None:
            values["labelmap_glossary"] = str(glossary).strip() or None
        values["final_answer_only"] = bool(values.get("final_answer_only", True))
        values["two_stage_refine"] = bool(values.get("two_stage_refine", False))
        values["caption_loop_cooldown"] = bool(values.get("caption_loop_cooldown", True))
        return values


class QwenCaptionResponse(BaseModel):
    caption: str
    used_counts: Dict[str, int] = Field(default_factory=dict)
    used_boxes: int
    truncated: bool
    recovery_events: List[Dict[str, Any]] = Field(default_factory=list)


class QwenCaptionPromptPreviewSection(BaseModel):
    title: str
    kind: str = "prompt"
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: str = ""
    note: Optional[str] = None
    image_region: Optional[Dict[str, Any]] = None
    chat_messages: List[Dict[str, Any]] = Field(default_factory=list)


class QwenCaptionPromptPreviewResponse(BaseModel):
    sections: List[QwenCaptionPromptPreviewSection] = Field(default_factory=list)
    full_text: str
    used_counts: Dict[str, int] = Field(default_factory=dict)
    used_boxes: int = 0
    truncated: bool = False
    step_plan: List[Dict[str, str]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class QwenCaptionDatasetJobRequest(BaseModel):
    dataset_id: str
    annotation_session_id: Optional[str] = None
    caption_request: Dict[str, Any] = Field(default_factory=dict)
    image_names: List[str] = Field(default_factory=list)
    split: Optional[Literal["all", "train", "val"]] = "all"
    max_images: Optional[int] = None
    overwrite: Optional[bool] = False
    save_text_labels: Optional[bool] = False
    generated_make_primary: Optional[bool] = False
    instruction_dataset: Optional[bool] = False
    subcaptions_per_image: Optional[int] = 0
    include_caption0_in_training: Optional[bool] = True
    include_generated_qa_in_training: Optional[bool] = True
    include_deterministic_metadata_qa: Optional[bool] = False
    include_source_annotations_in_generator_context: Optional[bool] = True
    strict_grounding: Optional[bool] = True
    qa_mix: Optional[str] = "balanced"
    answer_format: Optional[str] = "natural"
    preview_only: Optional[bool] = False
    resume: Optional[bool] = True
    attempts: Optional[int] = 2
    per_image_timeout_seconds: Optional[float] = 900.0
    runner_no_output_timeout_seconds: Optional[float] = None
    runner_heartbeat_interval_seconds: Optional[float] = 30.0
    runner_artifact_log_bytes: Optional[int] = 1048576
    runner_min_free_gb: Optional[float] = 5.0
    runner_disk_safety_factor: Optional[float] = 1.25
    cooldown_after_crash_seconds: Optional[float] = 5.0
    cooldown_after_success_seconds: Optional[float] = None
    max_failures: Optional[int] = 0
    continue_on_quality_failures: Optional[bool] = False
    set_and_forget: Optional[bool] = False
    auto_resume_count: Optional[int] = 0
    max_auto_resumes: Optional[int] = None
    max_failed_case_rate: Optional[float] = 0.0
    max_quality_failure_rate: Optional[float] = 0.0
    max_recovery_event_case_rate: Optional[float] = 0.25
    max_loop_recovery_case_rate: Optional[float] = 0.0
    max_deterministic_recovery_case_rate: Optional[float] = 0.0
    max_failed_attempt_row_rate: Optional[float] = 0.25
    max_signal_exit_attempt_row_rate: Optional[float] = None
    min_rate_cases: Optional[int] = 20
    resume_reprocess_recovery_events: Optional[bool] = False
    allow_model_download: Optional[bool] = False
    require_pilot_certification: Optional[bool] = False
    pilot_output_dir: Optional[str] = None
    pilot_target_cases: Optional[int] = 10000
    pilot_max_duration_hours: Optional[float] = 336.0
    pilot_max_p95_duration_hours: Optional[float] = None
    pilot_min_cases: Optional[int] = QWEN_CAPTION_DEFAULT_PILOT_MIN_CASES
    pilot_duration_safety_factor: Optional[float] = 1.25
    pilot_require_prompt_budget_data: Optional[bool] = True
    pilot_max_prompt_tokens: Optional[int] = 0
    pilot_max_prompt_budget_adapted_case_rate: Optional[float] = 1.0
    pilot_deterministic_recovery_confidence: Optional[float] = (
        QWEN_CAPTION_DEFAULT_PILOT_DETERMINISTIC_RECOVERY_CONFIDENCE
    )
    run_name: Optional[str] = None
    output_dir: Optional[str] = None

    @root_validator_compat(pre=True)
    def _normalize_caption_job_input(cls, values):  # noqa: N805
        if not isinstance(values, Mapping):
            return values
        data = dict(values)
        set_and_forget_raw = data.get("set_and_forget")
        set_and_forget_requested = (
            set_and_forget_raw is True
            or str(set_and_forget_raw or "").strip().lower() in {"1", "true", "yes", "on"}
        )
        loop_rate_was_provided = (
            "max_loop_recovery_case_rate" in data
            and data.get("max_loop_recovery_case_rate") is not None
            and str(data.get("max_loop_recovery_case_rate")).strip() != ""
        )
        if set_and_forget_requested and not loop_rate_was_provided:
            data["max_loop_recovery_case_rate"] = QWEN_CAPTION_SET_AND_FORGET_MAX_LOOP_RECOVERY_RATE
        deterministic_rate_was_provided = (
            "max_deterministic_recovery_case_rate" in data
            and data.get("max_deterministic_recovery_case_rate") is not None
            and str(data.get("max_deterministic_recovery_case_rate")).strip() != ""
        )
        if set_and_forget_requested and not deterministic_rate_was_provided:
            data["max_deterministic_recovery_case_rate"] = (
                QWEN_CAPTION_SET_AND_FORGET_MAX_DETERMINISTIC_RECOVERY_RATE
            )
        signal_exit_rate_was_provided = (
            "max_signal_exit_attempt_row_rate" in data
            and data.get("max_signal_exit_attempt_row_rate") is not None
            and str(data.get("max_signal_exit_attempt_row_rate")).strip() != ""
        )
        if set_and_forget_requested and not signal_exit_rate_was_provided:
            data["max_signal_exit_attempt_row_rate"] = (
                QWEN_CAPTION_SET_AND_FORGET_MAX_SIGNAL_EXIT_ATTEMPT_RATE
            )
        attempts_was_provided = (
            "attempts" in data
            and data.get("attempts") is not None
            and str(data.get("attempts")).strip() != ""
        )
        if set_and_forget_requested and not attempts_was_provided:
            data["attempts"] = QWEN_CAPTION_SET_AND_FORGET_ATTEMPTS
        max_failures_was_provided = (
            "max_failures" in data
            and data.get("max_failures") is not None
            and str(data.get("max_failures")).strip() != ""
        )
        instruction_dataset_raw = data.get("instruction_dataset")
        instruction_dataset_requested = (
            instruction_dataset_raw is True
            or str(instruction_dataset_raw or "").strip().lower() in {"1", "true", "yes", "on"}
        )
        try:
            requested_subcaptions = int(data.get("subcaptions_per_image") or 0)
        except (TypeError, ValueError, OverflowError):
            requested_subcaptions = 0
        if (
            set_and_forget_requested
            and instruction_dataset_requested
            and requested_subcaptions > 0
            and not max_failures_was_provided
        ):
            data["max_failures"] = QWEN_CAPTION_SET_AND_FORGET_INSTRUCTION_MAX_FAILURES
        data["dataset_id"] = str(data.get("dataset_id") or "").strip()
        data["annotation_session_id"] = str(data.get("annotation_session_id") or "").strip() or None
        raw_request = data.get("caption_request")
        if not isinstance(raw_request, Mapping):
            raw_request = {}
        cleaned_request = dict(raw_request)
        for image_field in (
            "image_base64",
            "image_token",
            "image_name",
            "label_hints",
            "image_width",
            "image_height",
        ):
            cleaned_request.pop(image_field, None)
        data["caption_request"] = cleaned_request
        data["image_names"] = [
            str(name or "").strip()
            for name in (data.get("image_names") or [])
            if str(name or "").strip()
        ]
        split = str(data.get("split") or "all").strip().lower()
        data["split"] = split if split in {"all", "train", "val"} else "all"
        for field in (
            "max_images",
            "attempts",
            "per_image_timeout_seconds",
            "runner_no_output_timeout_seconds",
            "runner_heartbeat_interval_seconds",
            "runner_artifact_log_bytes",
            "runner_min_free_gb",
            "runner_disk_safety_factor",
            "cooldown_after_crash_seconds",
            "cooldown_after_success_seconds",
            "max_failures",
            "auto_resume_count",
            "max_auto_resumes",
            "max_failed_case_rate",
            "max_quality_failure_rate",
            "max_recovery_event_case_rate",
            "max_loop_recovery_case_rate",
            "max_deterministic_recovery_case_rate",
            "max_failed_attempt_row_rate",
            "max_signal_exit_attempt_row_rate",
            "min_rate_cases",
            "pilot_target_cases",
            "pilot_max_duration_hours",
            "pilot_max_p95_duration_hours",
            "pilot_min_cases",
            "pilot_duration_safety_factor",
            "pilot_max_prompt_tokens",
            "pilot_max_prompt_budget_adapted_case_rate",
            "pilot_deterministic_recovery_confidence",
            "subcaptions_per_image",
        ):
            if data.get(field) is None:
                continue
            try:
                numeric_value = float(data.get(field))
            except (TypeError, ValueError, OverflowError):
                data[field] = None
                continue
            if not math.isfinite(numeric_value):
                data[field] = None
        data["pilot_output_dir"] = str(data.get("pilot_output_dir") or "").strip() or None
        data["run_name"] = str(data.get("run_name") or "").strip() or None
        data["output_dir"] = str(data.get("output_dir") or "").strip() or None
        return data

    @root_validator_compat(skip_on_failure=True)
    def _validate_caption_job(cls, values):  # noqa: N805
        dataset_id = str(values.get("dataset_id") or "").strip()
        if not dataset_id:
            raise ValueError("dataset_id_required")
        values["dataset_id"] = dataset_id
        max_images = values.get("max_images")
        if max_images is not None:
            try:
                max_images_int = int(max_images)
            except (TypeError, ValueError, OverflowError):
                max_images_int = 0
            values["max_images"] = max(0, max_images_int) if max_images_int > 0 else None
        attempts = values.get("attempts")
        try:
            attempts_int = int(attempts) if attempts is not None else 2
        except (TypeError, ValueError, OverflowError):
            attempts_int = 2
        values["attempts"] = max(1, min(attempts_int, 5))
        timeout = values.get("per_image_timeout_seconds")
        try:
            timeout_float = float(timeout) if timeout is not None else 900.0
        except (TypeError, ValueError, OverflowError):
            timeout_float = 900.0
        values["per_image_timeout_seconds"] = max(30.0, min(timeout_float, 7200.0))
        watchdog = values.get("runner_no_output_timeout_seconds")
        if watchdog is None:
            watchdog_float = max(300.0, values["per_image_timeout_seconds"] + 180.0)
        else:
            try:
                watchdog_float = float(watchdog)
            except (TypeError, ValueError, OverflowError):
                watchdog_float = max(300.0, values["per_image_timeout_seconds"] + 180.0)
        if not math.isfinite(watchdog_float):
            watchdog_float = max(300.0, values["per_image_timeout_seconds"] + 180.0)
        values["runner_no_output_timeout_seconds"] = max(0.0, min(watchdog_float, 86400.0))
        heartbeat = values.get("runner_heartbeat_interval_seconds")
        try:
            heartbeat_float = float(heartbeat) if heartbeat is not None else 30.0
        except (TypeError, ValueError, OverflowError):
            heartbeat_float = 30.0
        if not math.isfinite(heartbeat_float):
            heartbeat_float = 30.0
        values["runner_heartbeat_interval_seconds"] = max(0.0, min(heartbeat_float, 3600.0))
        artifact_log_bytes = values.get("runner_artifact_log_bytes")
        try:
            artifact_log_bytes_int = int(artifact_log_bytes) if artifact_log_bytes is not None else 1048576
        except (TypeError, ValueError, OverflowError):
            artifact_log_bytes_int = 1048576
        values["runner_artifact_log_bytes"] = max(0, min(artifact_log_bytes_int, 1_073_741_824))
        min_free = values.get("runner_min_free_gb")
        try:
            min_free_float = float(min_free) if min_free is not None else 5.0
        except (TypeError, ValueError, OverflowError):
            min_free_float = 5.0
        if not math.isfinite(min_free_float):
            min_free_float = 5.0
        values["runner_min_free_gb"] = max(0.0, min(min_free_float, 100_000.0))
        disk_safety = values.get("runner_disk_safety_factor")
        try:
            disk_safety_float = float(disk_safety) if disk_safety is not None else 1.25
        except (TypeError, ValueError, OverflowError):
            disk_safety_float = 1.25
        if not math.isfinite(disk_safety_float):
            disk_safety_float = 1.25
        values["runner_disk_safety_factor"] = max(1.0, min(disk_safety_float, 10.0))
        cooldown = values.get("cooldown_after_crash_seconds")
        try:
            cooldown_float = float(cooldown) if cooldown is not None else 5.0
        except (TypeError, ValueError, OverflowError):
            cooldown_float = 5.0
        values["cooldown_after_crash_seconds"] = max(0.0, min(cooldown_float, 300.0))
        success_cooldown = values.get("cooldown_after_success_seconds")
        if success_cooldown is not None:
            try:
                success_cooldown_float = float(success_cooldown)
            except (TypeError, ValueError, OverflowError):
                success_cooldown_float = 0.0
            if not math.isfinite(success_cooldown_float):
                success_cooldown_float = 0.0
            values["cooldown_after_success_seconds"] = max(0.0, min(success_cooldown_float, 300.0))
        max_failures = values.get("max_failures")
        try:
            max_failures_int = int(max_failures) if max_failures is not None else 0
        except (TypeError, ValueError, OverflowError):
            max_failures_int = 0
        values["max_failures"] = max(0, max_failures_int)
        auto_resume_count = values.get("auto_resume_count")
        try:
            auto_resume_count_int = int(auto_resume_count) if auto_resume_count is not None else 0
        except (TypeError, ValueError, OverflowError):
            auto_resume_count_int = 0
        values["auto_resume_count"] = max(0, min(auto_resume_count_int, 1000))
        max_auto_resumes = values.get("max_auto_resumes")
        if max_auto_resumes is None:
            values["max_auto_resumes"] = None
        else:
            try:
                max_auto_resumes_int = int(max_auto_resumes)
            except (TypeError, ValueError, OverflowError):
                max_auto_resumes_int = 0
            values["max_auto_resumes"] = max(0, min(max_auto_resumes_int, 1000))
        rate_defaults = {
            "max_failed_case_rate": 0.0,
            "max_quality_failure_rate": 0.0,
            "max_recovery_event_case_rate": 0.25,
            "max_loop_recovery_case_rate": 0.0,
            "max_deterministic_recovery_case_rate": 0.0,
            "max_failed_attempt_row_rate": 0.25,
            "max_signal_exit_attempt_row_rate": 0.0,
        }
        for field, default in rate_defaults.items():
            try:
                rate_value = float(values.get(field)) if values.get(field) is not None else default
            except (TypeError, ValueError, OverflowError):
                rate_value = default
            if not math.isfinite(rate_value):
                rate_value = default
            values[field] = -1.0 if rate_value < 0 else max(0.0, min(rate_value, 1.0))
        min_rate_cases = values.get("min_rate_cases")
        try:
            min_rate_cases_int = int(min_rate_cases) if min_rate_cases is not None else 20
        except (TypeError, ValueError, OverflowError):
            min_rate_cases_int = 20
        values["min_rate_cases"] = max(1, min(min_rate_cases_int, 1_000_000))
        pilot_target_cases = values.get("pilot_target_cases")
        try:
            pilot_target_cases_int = int(pilot_target_cases) if pilot_target_cases is not None else 10000
        except (TypeError, ValueError, OverflowError):
            pilot_target_cases_int = 10000
        values["pilot_target_cases"] = max(1, min(pilot_target_cases_int, 10_000_000))
        pilot_max_duration_hours = values.get("pilot_max_duration_hours")
        try:
            pilot_max_duration_hours_float = (
                float(pilot_max_duration_hours) if pilot_max_duration_hours is not None else 336.0
            )
        except (TypeError, ValueError, OverflowError):
            pilot_max_duration_hours_float = 336.0
        if not math.isfinite(pilot_max_duration_hours_float):
            pilot_max_duration_hours_float = 336.0
        values["pilot_max_duration_hours"] = max(0.01, min(pilot_max_duration_hours_float, 100_000.0))
        pilot_max_p95_duration_hours = values.get("pilot_max_p95_duration_hours")
        if pilot_max_p95_duration_hours is not None:
            try:
                pilot_max_p95_duration_hours_float = float(pilot_max_p95_duration_hours)
            except (TypeError, ValueError, OverflowError):
                pilot_max_p95_duration_hours_float = pilot_max_duration_hours_float
            if not math.isfinite(pilot_max_p95_duration_hours_float):
                pilot_max_p95_duration_hours_float = pilot_max_duration_hours_float
            values["pilot_max_p95_duration_hours"] = (
                -1.0
                if pilot_max_p95_duration_hours_float < 0
                else max(0.01, min(pilot_max_p95_duration_hours_float, 100_000.0))
            )
        pilot_min_cases = values.get("pilot_min_cases")
        try:
            pilot_min_cases_int = (
                int(pilot_min_cases)
                if pilot_min_cases is not None
                else QWEN_CAPTION_DEFAULT_PILOT_MIN_CASES
            )
        except (TypeError, ValueError, OverflowError):
            pilot_min_cases_int = QWEN_CAPTION_DEFAULT_PILOT_MIN_CASES
        values["pilot_min_cases"] = max(1, min(pilot_min_cases_int, 1_000_000))
        pilot_duration_safety_factor = values.get("pilot_duration_safety_factor")
        try:
            pilot_duration_safety_factor_float = (
                float(pilot_duration_safety_factor)
                if pilot_duration_safety_factor is not None
                else 1.25
            )
        except (TypeError, ValueError, OverflowError):
            pilot_duration_safety_factor_float = 1.25
        if not math.isfinite(pilot_duration_safety_factor_float):
            pilot_duration_safety_factor_float = 1.25
        values["pilot_duration_safety_factor"] = max(1.0, min(pilot_duration_safety_factor_float, 10.0))
        pilot_max_prompt_tokens = values.get("pilot_max_prompt_tokens")
        try:
            pilot_max_prompt_tokens_int = (
                int(pilot_max_prompt_tokens) if pilot_max_prompt_tokens is not None else 0
            )
        except (TypeError, ValueError, OverflowError):
            pilot_max_prompt_tokens_int = 0
        values["pilot_max_prompt_tokens"] = max(0, min(pilot_max_prompt_tokens_int, 1_000_000))
        pilot_adapted_rate = values.get("pilot_max_prompt_budget_adapted_case_rate")
        try:
            pilot_adapted_rate_float = (
                float(pilot_adapted_rate) if pilot_adapted_rate is not None else 1.0
            )
        except (TypeError, ValueError, OverflowError):
            pilot_adapted_rate_float = 1.0
        if not math.isfinite(pilot_adapted_rate_float):
            pilot_adapted_rate_float = 1.0
        values["pilot_max_prompt_budget_adapted_case_rate"] = (
            -1.0 if pilot_adapted_rate_float < 0 else max(0.0, min(pilot_adapted_rate_float, 1.0))
        )
        pilot_confidence = values.get("pilot_deterministic_recovery_confidence")
        try:
            pilot_confidence_float = (
                float(pilot_confidence)
                if pilot_confidence is not None
                else QWEN_CAPTION_DEFAULT_PILOT_DETERMINISTIC_RECOVERY_CONFIDENCE
            )
        except (TypeError, ValueError, OverflowError):
            pilot_confidence_float = QWEN_CAPTION_DEFAULT_PILOT_DETERMINISTIC_RECOVERY_CONFIDENCE
        if not math.isfinite(pilot_confidence_float):
            pilot_confidence_float = QWEN_CAPTION_DEFAULT_PILOT_DETERMINISTIC_RECOVERY_CONFIDENCE
        values["pilot_deterministic_recovery_confidence"] = max(
            0.0,
            min(pilot_confidence_float, 0.999999),
        )

        def _coerce_bool(value: Any, default: bool = False) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            raw = str(value).strip().lower()
            if raw in {"1", "true", "yes", "on"}:
                return True
            if raw in {"0", "false", "no", "off"}:
                return False
            return default

        values["overwrite"] = _coerce_bool(values.get("overwrite"), False)
        values["save_text_labels"] = _coerce_bool(values.get("save_text_labels"), False)
        values["generated_make_primary"] = _coerce_bool(values.get("generated_make_primary"), False)
        values["instruction_dataset"] = _coerce_bool(values.get("instruction_dataset"), False)
        values["include_caption0_in_training"] = _coerce_bool(
            values.get("include_caption0_in_training"),
            True,
        )
        values["include_generated_qa_in_training"] = _coerce_bool(
            values.get("include_generated_qa_in_training"),
            True,
        )
        values["include_deterministic_metadata_qa"] = _coerce_bool(
            values.get("include_deterministic_metadata_qa"),
            False,
        )
        if bool(values["instruction_dataset"]) and not (
            bool(values["include_caption0_in_training"])
            or bool(values["include_generated_qa_in_training"])
            or bool(values["include_deterministic_metadata_qa"])
        ):
            raise ValueError("instruction_dataset_requires_training_row_family")
        values["include_source_annotations_in_generator_context"] = _coerce_bool(
            values.get("include_source_annotations_in_generator_context"),
            True,
        )
        values["strict_grounding"] = _coerce_bool(values.get("strict_grounding"), True)
        try:
            subcaptions = int(values.get("subcaptions_per_image") or 0)
        except (TypeError, ValueError, OverflowError):
            subcaptions = 0
        values["subcaptions_per_image"] = max(0, min(subcaptions, 20))
        qa_mix = str(values.get("qa_mix") or "balanced").strip().lower()
        values["qa_mix"] = qa_mix if qa_mix in {"balanced", "scene", "object", "caption"} else "balanced"
        answer_format = str(values.get("answer_format") or "natural").strip().lower()
        values["answer_format"] = answer_format if answer_format in {"natural", "json"} else "natural"
        values["preview_only"] = _coerce_bool(values.get("preview_only"), False)
        values["resume"] = _coerce_bool(values.get("resume"), True)
        values["continue_on_quality_failures"] = _coerce_bool(
            values.get("continue_on_quality_failures"),
            False,
        )
        values["set_and_forget"] = _coerce_bool(values.get("set_and_forget"), False)
        values["resume_reprocess_recovery_events"] = _coerce_bool(
            values.get("resume_reprocess_recovery_events"),
            False,
        )
        values["allow_model_download"] = _coerce_bool(values.get("allow_model_download"), False)
        values["require_pilot_certification"] = _coerce_bool(
            values.get("require_pilot_certification"),
            False,
        )
        values["pilot_require_prompt_budget_data"] = _coerce_bool(
            values.get("pilot_require_prompt_budget_data"),
            True,
        )
        return values


def _normalize_qwen_variant_value(value: Any, *, default: Optional[str] = "auto") -> Optional[str]:
    if value is None:
        return default
    raw = str(value).strip()
    if not raw:
        return default
    variant_lookup = {
        "auto": "auto",
        "instruct": "Instruct",
        "thinking": "Thinking",
    }
    return variant_lookup.get(raw.lower(), default)


class QwenPromptSection(BaseModel):
    base_prompt: str
    default_image_type: str = "image"
    default_extra_context: str = ""

    @root_validator_compat(skip_on_failure=True)
    def _validate_qwen_section(cls, values):  # noqa: N805
        template = values.get("base_prompt") or ""
        if "{items}" not in template:
            raise ValueError("base_prompt_missing_items_placeholder")
        if "{image_type}" not in template:
            raise ValueError("base_prompt_missing_image_type_placeholder")
        if "{extra_context}" not in template:
            raise ValueError("base_prompt_missing_extra_context_placeholder")
        return values


class QwenPromptConfig(BaseModel):
    bbox: QwenPromptSection
    point: QwenPromptSection


class QwenRuntimeSettings(BaseModel):
    trust_remote_code: bool = False
    inference_platform: Literal["auto", "transformers", "mlx_vlm"] = "auto"
    mlx_model_id: Optional[str] = None
    mlx_available: Optional[bool] = None
    mlx_models: List[Dict[str, Any]] = Field(default_factory=list)


class QwenRuntimeSettingsUpdate(BaseModel):
    trust_remote_code: Optional[bool] = None
    inference_platform: Optional[str] = None
    mlx_model_id: Optional[str] = None


class Sam3ModelActivateRequest(BaseModel):
    checkpoint_path: Optional[str] = None
    label: Optional[str] = None
    enable_segmentation: Optional[bool] = None


class QwenModelActivateRequest(BaseModel):
    model_id: str


class ActiveModelRequest(BaseModel):
    classifier_path: Optional[str] = None
    labelmap_path: Optional[str] = None
    clip_model: Optional[str] = None
    logit_adjustment_inference: Optional[bool] = None


class ActiveModelResponse(BaseModel):
    clip_model: Optional[str]
    encoder_type: Optional[str] = None
    encoder_model: Optional[str] = None
    classifier_path: Optional[str]
    labelmap_path: Optional[str]
    clip_ready: bool
    clip_error: Optional[str] = None
    clip_warnings: List[str] = Field(default_factory=list)
    encoder_ready: Optional[bool] = None
    encoder_error: Optional[str] = None
    labelmap_entries: List[str] = Field(default_factory=list)
    logit_adjustment_inference: Optional[bool] = None


class SegmentationBuildRequest(BaseModel):
    source_dataset_id: str = Field(..., description="Existing bbox dataset id (Qwen or SAM3)")
    output_name: Optional[str] = Field(None, description="Optional output dataset name")
    sam_variant: Literal["sam1", "sam3"] = Field("sam3", description="Generator to use for masks")
    output_format: Literal["yolo-seg"] = Field("yolo-seg", description="Target mask encoding (polygons)")
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Mask probability threshold")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Box confidence threshold")
    simplify_epsilon: float = Field(30.0, ge=0.0, description="Polygon simplification epsilon (px)")
    min_size: float = Field(0.0, ge=0.0, description="Minimum mask area (px^2)")
    max_results: int = Field(1, ge=1, description="Max detections per box prompt")


class QwenTrainRequest(BaseModel):
    dataset_id: Optional[str] = None
    run_name: Optional[str] = None
    model_id: Optional[str] = None
    training_mode: Optional[Literal["official_lora", "trl_qlora"]] = None
    system_prompt: Optional[str] = None
    devices: Optional[str] = None
    batch_size: Optional[int] = None
    max_epochs: Optional[int] = None
    lr: Optional[float] = None
    accumulate_grad_batches: Optional[int] = None
    warmup_steps: Optional[int] = None
    num_workers: Optional[int] = None
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[List[str]] = None
    log_every_n_steps: Optional[int] = None
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    max_length: Optional[int] = None
    seed: Optional[int] = None
    random_split: Optional[bool] = None
    val_percent: Optional[float] = None
    split_seed: Optional[int] = None
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None

    @root_validator_compat(pre=True)
    def _normalize_qwen_train_payload_input(cls, values):  # noqa: N805
        if not isinstance(values, Mapping):
            return values
        data = dict(values)
        mode = str(data.get("training_mode") or "").strip().lower().replace("-", "_")
        mode_lookup = {
            "": None,
            "official": "official_lora",
            "lora": "official_lora",
            "official_lora": "official_lora",
            "qlora": "trl_qlora",
            "trl": "trl_qlora",
            "trl_qlora": "trl_qlora",
        }
        if mode in mode_lookup:
            data["training_mode"] = mode_lookup[mode]
        for field in ("dataset_id", "run_name", "model_id", "system_prompt", "devices"):
            if data.get(field) is not None:
                data[field] = str(data.get(field)).strip() or None
        targets = data.get("lora_target_modules")
        if isinstance(targets, str):
            data["lora_target_modules"] = [
                item.strip() for item in targets.split(",") if item.strip()
            ]
        elif isinstance(targets, list):
            data["lora_target_modules"] = [
                str(item or "").strip() for item in targets if str(item or "").strip()
            ] or None
        for field in (
            "batch_size",
            "max_epochs",
            "lr",
            "accumulate_grad_batches",
            "warmup_steps",
            "num_workers",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "log_every_n_steps",
            "min_pixels",
            "max_pixels",
            "max_length",
            "seed",
            "val_percent",
            "split_seed",
            "train_limit",
            "val_limit",
        ):
            if data.get(field) is None:
                continue
            if isinstance(data.get(field), str) and not data.get(field).strip():
                data[field] = None
                continue
            try:
                numeric_value = float(data.get(field))
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(numeric_value):
                data[field] = None
        return data

    @root_validator_compat(skip_on_failure=True)
    def _validate_dataset_fields(cls, values):  # noqa: N805
        if not values.get("dataset_id"):
            raise ValueError("dataset_id_required")
        return values


class Sam3TrainRequest(BaseModel):
    dataset_id: str
    run_name: Optional[str] = None
    experiment_log_dir: Optional[str] = None
    train_batch_size: Optional[int] = None
    val_batch_size: Optional[int] = None
    num_train_workers: Optional[int] = None
    num_val_workers: Optional[int] = None
    max_epochs: Optional[int] = None
    resolution: Optional[int] = None
    lr_scale: Optional[float] = None
    gradient_accumulation_steps: Optional[int] = None
    val_epoch_freq: Optional[int] = None
    target_epoch_size: Optional[int] = None
    scheduler_warmup: Optional[int] = None
    scheduler_timescale: Optional[int] = None
    num_gpus: Optional[int] = None
    enable_inst_interactivity: Optional[bool] = None
    balance_classes: Optional[bool] = None
    balance_strategy: Optional[str] = None
    balance_power: Optional[float] = None
    balance_clip: Optional[float] = None
    balance_beta: Optional[float] = None
    balance_gamma: Optional[float] = None
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None
    log_freq: Optional[int] = None
    log_every_batch: Optional[bool] = None
    enable_segmentation_head: Optional[bool] = None
    train_segmentation: Optional[bool] = None
    freeze_language_backbone: Optional[bool] = None
    language_backbone_lr: Optional[float] = None
    prompt_variants: Optional[Dict[str, Any]] = None
    prompt_randomize: Optional[bool] = None
    val_score_thresh: Optional[float] = None
    val_max_dets: Optional[int] = None
    random_split: Optional[bool] = None
    val_percent: Optional[float] = None
    split_seed: Optional[int] = None


class YoloTrainRequest(BaseModel):
    dataset_id: Optional[str] = None
    dataset_root: Optional[str] = None
    run_name: Optional[str] = None
    task: Literal["detect", "segment"] = "detect"
    variant: Optional[str] = None
    from_scratch: Optional[bool] = None
    base_weights: Optional[str] = None
    epochs: Optional[int] = None
    img_size: Optional[int] = None
    batch: Optional[int] = None
    workers: Optional[int] = None
    accelerator: Optional[Literal["auto", "cuda", "mps", "cpu"]] = "auto"
    devices: Optional[List[int]] = None
    seed: Optional[int] = None
    augmentations: Optional[Dict[str, Any]] = None
    accept_tos: Optional[bool] = None

    @root_validator_compat(skip_on_failure=True)
    def _validate_dataset_fields(cls, values):  # noqa: N805
        if not (values.get("dataset_id") or values.get("dataset_root")):
            raise ValueError("dataset_id_or_root_required")
        return values


class YoloHeadGraftRequest(BaseModel):
    base_run_id: str
    dataset_id: Optional[str] = None
    dataset_root: Optional[str] = None
    run_name: Optional[str] = None
    epochs: Optional[int] = None
    img_size: Optional[int] = None
    batch: Optional[int] = None
    workers: Optional[int] = None
    accelerator: Optional[Literal["auto", "cuda", "mps", "cpu"]] = "auto"
    devices: Optional[List[int]] = None
    seed: Optional[int] = None
    export_onnx: Optional[bool] = None
    accept_tos: Optional[bool] = None

    @root_validator_compat(skip_on_failure=True)
    def _validate_dataset_fields(cls, values):  # noqa: N805
        if not (values.get("dataset_id") or values.get("dataset_root")):
            raise ValueError("dataset_id_or_root_required")
        return values


class YoloHeadGraftDryRunRequest(BaseModel):
    base_run_id: str
    dataset_id: Optional[str] = None
    dataset_root: Optional[str] = None

    @root_validator_compat(skip_on_failure=True)
    def _validate_dataset_fields(cls, values):  # noqa: N805
        if not (values.get("dataset_id") or values.get("dataset_root")):
            raise ValueError("dataset_id_or_root_required")
        return values


class RfDetrTrainRequest(BaseModel):
    dataset_id: Optional[str] = None
    dataset_root: Optional[str] = None
    run_name: Optional[str] = None
    task: Literal["detect", "segment"] = "detect"
    variant: Optional[str] = None
    epochs: Optional[int] = None
    batch: Optional[int] = None
    grad_accum: Optional[int] = None
    workers: Optional[int] = None
    devices: Optional[List[int]] = None
    seed: Optional[int] = None
    resolution: Optional[int] = None
    from_scratch: Optional[bool] = None
    pretrain_weights: Optional[str] = None
    use_ema: Optional[bool] = None
    early_stopping: Optional[bool] = None
    early_stopping_patience: Optional[int] = None
    multi_scale: Optional[bool] = None
    expanded_scales: Optional[bool] = None
    augmentations: Optional[Dict[str, Any]] = None
    accept_tos: Optional[bool] = None

    @root_validator_compat(skip_on_failure=True)
    def _validate_dataset_fields(cls, values):  # noqa: N805
        if not (values.get("dataset_id") or values.get("dataset_root")):
            raise ValueError("dataset_id_or_root_required")
        return values


class YoloActiveRequest(BaseModel):
    run_id: str


class RfDetrActiveRequest(BaseModel):
    run_id: str


class DetectorDefaultRequest(BaseModel):
    mode: str


class YoloRegionRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    region: List[float]
    conf: Optional[float] = 0.25
    iou: Optional[float] = 0.45
    max_det: Optional[int] = 300
    center_only: Optional[bool] = True
    image_is_cropped: Optional[bool] = False
    full_width: Optional[int] = None
    full_height: Optional[int] = None
    expected_labelmap: Optional[List[str]] = None

    @root_validator_compat(skip_on_failure=True)
    def _validate_region(cls, values):  # noqa: N805
        region = values.get("region")
        if not isinstance(region, list) or len(region) < 4:
            raise ValueError("region_required")
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_required")
        return values


class YoloRegionDetection(BaseModel):
    bbox: List[float]
    class_id: int
    class_name: Optional[str] = None
    score: Optional[float] = None


class YoloRegionResponse(BaseModel):
    detections: List[YoloRegionDetection]
    labelmap: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class YoloFullRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    conf: Optional[float] = 0.25
    iou: Optional[float] = 0.45
    max_det: Optional[int] = 300
    expected_labelmap: Optional[List[str]] = None

    @root_validator_compat(skip_on_failure=True)
    def _validate_image(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_required")
        return values


class YoloWindowedRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    conf: Optional[float] = 0.25
    iou: Optional[float] = 0.45
    max_det: Optional[int] = 300
    expected_labelmap: Optional[List[str]] = None
    slice_size: Optional[int] = 640
    overlap: Optional[float] = 0.2
    merge_iou: Optional[float] = 0.5

    @root_validator_compat(skip_on_failure=True)
    def _validate_image(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_required")
        return values


class RfDetrRegionRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    region: List[float]
    conf: Optional[float] = 0.25
    max_det: Optional[int] = 300
    center_only: Optional[bool] = True
    image_is_cropped: Optional[bool] = False
    full_width: Optional[int] = None
    full_height: Optional[int] = None
    expected_labelmap: Optional[List[str]] = None

    @root_validator_compat(skip_on_failure=True)
    def _validate_region(cls, values):  # noqa: N805
        region = values.get("region")
        if not isinstance(region, list) or len(region) < 4:
            raise ValueError("region_required")
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_required")
        return values


class RfDetrRegionDetection(BaseModel):
    bbox: List[float]
    class_id: int
    class_name: Optional[str] = None
    score: Optional[float] = None


class RfDetrRegionResponse(BaseModel):
    detections: List[RfDetrRegionDetection]
    labelmap: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class RfDetrFullRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    conf: Optional[float] = 0.25
    max_det: Optional[int] = 300
    expected_labelmap: Optional[List[str]] = None

    @root_validator_compat(skip_on_failure=True)
    def _validate_image(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_required")
        return values


class RfDetrWindowedRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    conf: Optional[float] = 0.25
    max_det: Optional[int] = 300
    expected_labelmap: Optional[List[str]] = None
    slice_size: Optional[int] = 640
    overlap: Optional[float] = 0.2
    merge_iou: Optional[float] = 0.5

    @root_validator_compat(skip_on_failure=True)
    def _validate_image(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_required")
        return values


class Sam3TextPromptResponse(BaseModel):
    detections: List[QwenDetection] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    image_token: Optional[str] = None
    # Optional masks aligned to detections (packed and base64-encoded to stay compact)
    masks: Optional[List[Dict[str, Any]]] = None


class Sam3TextPromptAutoResponse(BaseModel):
    detections: List[SamPointAutoResponse] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    image_token: Optional[str] = None


class PredictorSettingsUpdate(BaseModel):
    max_predictors: int


class MultiPointPrompt(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    positive_points: List[List[float]] = Field(default_factory=list)
    negative_points: List[List[float]] = Field(default_factory=list)
    uuid: Optional[str] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None

    @root_validator_compat(skip_on_failure=True)
    def _ensure_multi_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        return values


class YoloBboxOutput(BaseModel):
    class_id: str
    bbox: List[float]
    uuid: Optional[str] = None
    image_token: Optional[str] = None
    mask: Optional[Dict[str, Any]] = None
    simplify_epsilon: Optional[float] = None


class Sam3TextPrompt(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    text_prompt: str
    threshold: float = 0.5
    mask_threshold: float = 0.5
    simplify_epsilon: Optional[float] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None
    max_results: Optional[int] = None
    min_size: Optional[int] = None
    windowed: bool = False
    window_size: Optional[int] = None
    window_overlap: Optional[float] = None
    merge_iou: Optional[float] = None

    @root_validator_compat(skip_on_failure=True)
    def _ensure_text_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        if not values.get("text_prompt"):
            raise ValueError("text_prompt_required")
        min_size = values.get("min_size")
        if min_size is not None:
            try:
                min_size_int = max(0, int(min_size))
            except (TypeError, ValueError):
                min_size_int = 0
            values["min_size"] = min_size_int
        eps = values.get("simplify_epsilon")
        if eps is not None:
            try:
                eps_val = float(eps)
            except (TypeError, ValueError):
                eps_val = None
            values["simplify_epsilon"] = eps_val if eps_val is None or eps_val >= 0 else 0.0
        window_size = values.get("window_size")
        if window_size is not None:
            try:
                values["window_size"] = max(1, int(window_size))
            except (TypeError, ValueError):
                values["window_size"] = None
        for key in ("window_overlap", "merge_iou"):
            raw = values.get(key)
            if raw is not None:
                try:
                    values[key] = float(raw)
                except (TypeError, ValueError):
                    values[key] = None
        return values


class PrepassRecipeRequest(BaseModel):
    recipe_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    glossary: Optional[str] = None


class PrepassRecipeResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: float
    updated_at: float
    config: Dict[str, Any]
    glossary: Optional[str] = None
    schema_version: int
    renamed_from: Optional[str] = None
    notice: Optional[str] = None


class AgentApplyImageRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    image_name: Optional[str] = None
    sam_variant: Optional[str] = None
    recipe: Dict[str, Any]
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_size: int = Field(0, ge=0, le=10_000)
    simplify_epsilon: float = Field(0.0, ge=0.0, le=1_000.0)
    max_results: int = Field(1000, ge=1, le=5000)
    override_class_id: Optional[int] = Field(None, ge=0)
    override_class_name: Optional[str] = None
    clip_head_min_prob_override: Optional[float] = Field(None, ge=0.0, le=1.0)
    clip_head_margin_override: Optional[float] = Field(None, ge=0.0, le=1.0)
    extra_clip_classifier_path: Optional[str] = None
    extra_clip_min_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    extra_clip_margin: Optional[float] = Field(None, ge=0.0, le=1.0)

    @root_validator_compat(skip_on_failure=True)
    def _ensure_agent_apply_image_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        if not isinstance(values.get("recipe"), dict) or not values.get("recipe"):
            raise ValueError("recipe_required")
        return values


class AgentApplyChainStep(BaseModel):
    enabled: bool = True
    recipe_id: Optional[str] = None
    recipe: Optional[Dict[str, Any]] = None
    override_class_id: Optional[int] = Field(None, ge=0)
    override_class_name: Optional[str] = None
    dedupe_group: Optional[str] = None
    participate_cross_class_dedupe: bool = True
    clip_head_min_prob_override: Optional[float] = Field(None, ge=0.0, le=1.0)
    clip_head_margin_override: Optional[float] = Field(None, ge=0.0, le=1.0)
    extra_clip_classifier_path: Optional[str] = None
    extra_clip_min_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    extra_clip_margin: Optional[float] = Field(None, ge=0.0, le=1.0)

    @root_validator_compat(skip_on_failure=True)
    def _ensure_chain_step(cls, values):  # noqa: N805
        recipe_id = values.get("recipe_id")
        recipe_obj = values.get("recipe")
        if not recipe_id and not (isinstance(recipe_obj, dict) and recipe_obj):
            raise ValueError("recipe_required")
        return values


class AgentCascadeDedupeConfig(BaseModel):
    per_class_iou: float = Field(0.5, ge=0.0, le=1.0)
    cross_class_enabled: bool = False
    cross_class_iou: float = Field(0.5, ge=0.0, le=1.0)
    cross_class_scope: Literal["groups", "global"] = "groups"
    confidence: Literal["sam_score", "clip_head_prob", "clip_head_margin"] = "sam_score"
    clip_head_recipe_id: Optional[str] = None


class AgentApplyImageChainRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    image_name: Optional[str] = None
    sam_variant: Optional[str] = None
    steps: List[AgentApplyChainStep]
    dedupe: AgentCascadeDedupeConfig = Field(default_factory=AgentCascadeDedupeConfig)
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_size: int = Field(0, ge=0, le=10_000)
    simplify_epsilon: float = Field(0.0, ge=0.0, le=1_000.0)
    max_results: int = Field(1000, ge=1, le=5000)

    @root_validator_compat(skip_on_failure=True)
    def _ensure_agent_apply_chain_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        steps = values.get("steps") or []
        if not isinstance(steps, list) or not steps:
            raise ValueError("steps_required")
        enabled_steps: List[Any] = []
        for step in steps:
            if isinstance(step, dict):
                if step.get("enabled", True):
                    enabled_steps.append(step)
                continue
            if bool(getattr(step, "enabled", True)):
                enabled_steps.append(step)
        if not enabled_steps:
            raise ValueError("steps_required")
        return values


class AgentRecipeExportRequest(BaseModel):
    dataset_id: str
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    label: str = Field(..., min_length=1, max_length=128)
    recipe: Dict[str, Any]


class AgentCascadeSaveRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=128)
    steps: List[AgentApplyChainStep]
    dedupe: AgentCascadeDedupeConfig = Field(default_factory=AgentCascadeDedupeConfig)


class QwenPrepassRequest(BaseModel):
    dataset_id: Optional[str] = None
    recipe_source_dataset_id: Optional[str] = None
    edr_package_id: Optional[str] = None
    edr_package_apply_ensemble: Optional[bool] = True
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    image_name: Optional[str] = None
    model_id: Optional[str] = None
    model_variant: Optional[Literal["auto", "Instruct", "Thinking"]] = "auto"
    labelmap: Optional[List[str]] = None
    labelmap_glossary: Optional[str] = None
    detector_mode: Optional[Literal["yolo", "rfdetr"]] = "yolo"
    detector_id: Optional[str] = None
    enable_yolo: Optional[bool] = True
    enable_rfdetr: Optional[bool] = True
    yolo_id: Optional[str] = None
    rfdetr_id: Optional[str] = None
    sahi_window_size: Optional[int] = None
    sahi_overlap_ratio: Optional[float] = None
    classifier_id: Optional[str] = None
    sam_variant: Optional[str] = "sam3"
    enable_sam3_text: Optional[bool] = True
    sam3_text_synonym_budget: Optional[int] = 10
    sam3_text_window_extension: Optional[bool] = True
    sam3_text_window_mode: Optional[Literal["grid", "sahi"]] = "grid"
    sam3_text_window_size: Optional[int] = None
    sam3_text_window_overlap: Optional[float] = None
    enable_sam3_similarity: Optional[bool] = True
    similarity_min_exemplar_score: Optional[float] = 0.6
    similarity_exemplar_count: Optional[int] = 3
    similarity_exemplar_strategy: Optional[Literal["top", "random", "diverse"]] = "top"
    similarity_exemplar_seed: Optional[int] = None
    similarity_exemplar_fraction: Optional[float] = 0.2
    similarity_exemplar_min: Optional[int] = 3
    similarity_exemplar_max: Optional[int] = 12
    similarity_exemplar_source_quota: Optional[int] = 1
    similarity_window_mode: Optional[Literal["grid", "sahi"]] = "grid"
    similarity_window_size: Optional[int] = None
    similarity_window_overlap: Optional[float] = None
    similarity_window_extension: Optional[bool] = True
    prepass_mode: Optional[str] = "ensemble_sahi_sam3_text_similarity"
    prepass_only: Optional[bool] = True
    prepass_finalize: Optional[bool] = True
    prepass_keep_all: Optional[bool] = False
    prepass_sam3_text_thr: Optional[float] = 0.2
    prepass_similarity_score: Optional[float] = 0.3
    prepass_caption: Optional[bool] = True
    prepass_caption_profile: Optional[str] = "light"
    prepass_caption_model_id: Optional[str] = None
    prepass_caption_variant: Optional[Literal["auto", "Instruct", "Thinking"]] = None
    prepass_caption_max_tokens: Optional[int] = None
    use_detection_overlay: Optional[bool] = True
    grid_cols: Optional[int] = 2
    grid_rows: Optional[int] = 2
    grid_overlap_ratio: Optional[float] = None
    overlay_dot_radius: Optional[int] = None
    tighten_fp: Optional[bool] = True
    detector_conf: Optional[float] = 0.45
    detector_iou: Optional[float] = None
    detector_merge_iou: Optional[float] = None
    sam3_score_thr: Optional[float] = 0.2
    sam3_mask_threshold: Optional[float] = 0.2
    classifier_min_prob: Optional[float] = 0.35
    classifier_margin: Optional[float] = 0.05
    classifier_bg_margin: Optional[float] = 0.05
    scoreless_iou: Optional[float] = 0.0
    ensemble_enabled: Optional[bool] = False
    ensemble_job_id: Optional[str] = None
    iou: Optional[float] = 0.75
    fusion_mode: Optional[Literal["primary", "wbf", "source_weighted"]] = "primary"
    cross_class_dedupe_enabled: Optional[bool] = False
    cross_class_dedupe_iou: Optional[float] = 0.8
    max_new_tokens: Optional[int] = 4096
    thinking_effort: Optional[float] = None
    thinking_scale_factor: Optional[float] = None
    immediate_action_bias: Optional[bool] = True
    immediate_action_min_chars: Optional[int] = 200
    immediate_action_min_seconds: Optional[float] = 2.0
    immediate_action_logit_bias: Optional[float] = 6.0
    trace_verbose: Optional[bool] = False

    @root_validator_compat(pre=True)
    def _normalize_prepass_payload_input(cls, values):  # noqa: N805
        if not isinstance(values, Mapping):
            return values
        data = dict(values)
        data["model_variant"] = _normalize_qwen_variant_value(
            data.get("model_variant"),
            default="auto",
        )
        if "prepass_caption_variant" in data:
            data["prepass_caption_variant"] = _normalize_qwen_variant_value(
                data.get("prepass_caption_variant"),
                default=None,
            )
        for field in ("prepass_caption_max_tokens", "max_new_tokens"):
            if data.get(field) is None:
                continue
            try:
                numeric_value = float(data.get(field))
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(numeric_value):
                data[field] = None
        for field in (
            "image_token",
            "image_name",
            "model_id",
            "prepass_caption_model_id",
        ):
            if data.get(field) is not None:
                data[field] = str(data.get(field)).strip() or None
        return data


class AutoLabelPlannerCellAssignment(BaseModel):
    cell: str
    classes: List[str] = Field(default_factory=list)


class AutoLabelPlannerDecision(BaseModel):
    decision: Literal["skip", "full_image", "quadrants", "grid_cells"] = "full_image"
    scene_tags: List[Literal["crowded", "small_objects", "text_heavy", "coverage_gap", "spatially_localized"]] = Field(default_factory=list)
    global_classes: List[str] = Field(default_factory=list)
    cells: List[str] = Field(default_factory=list)
    cell_classes: List[AutoLabelPlannerCellAssignment] = Field(default_factory=list)
    reason: Optional[str] = None
    confidence: Literal["low", "medium", "high"] = "medium"


class AutoLabelRequest(BaseModel):
    dataset_id: str
    annotation_session_id: Optional[str] = None
    max_images: Optional[int] = Field(100, ge=1, le=100_000)
    split: Optional[Literal["all", "train", "val"]] = "all"
    unlabeled_only: bool = True
    image_relpaths: Optional[List[str]] = None
    target_mode: Literal["auto", "segmentation", "detection"] = "auto"
    falcon_window_mode: Literal["full_image", "quadrants", "planner_auto"] = "full_image"
    enable_falcon: bool = True
    falcon_overlap_ratio: float = Field(0.1, ge=0.0, le=0.45)
    dedupe_existing_same_class_iou: float = Field(0.5, ge=0.0, le=1.0)
    class_names: Optional[List[str]] = None
    edr_package_id: Optional[str] = None
    enable_yolo: bool = True
    enable_rfdetr: bool = True
    yolo_id: Optional[str] = None
    rfdetr_id: Optional[str] = None
    classifier_id: Optional[str] = None
    use_planner_caption: bool = True
    planner_grid_cols: int = Field(3, ge=1, le=6)
    planner_grid_rows: int = Field(3, ge=1, le=6)
    planner_model_id: Optional[str] = None
    planner_model_variant: Optional[Literal["auto", "Instruct", "Thinking"]] = "auto"
    falcon_model_id: Optional[str] = None
    falcon_device: Optional[str] = None
    falcon_backend: Literal["embedded", "server"] = "embedded"
    falcon_detection_strategy: Literal["native_detection", "segmentation_boxes"] = "native_detection"
    falcon_component_mode: Literal["largest_component", "component_split", "component_cluster"] = "component_split"
    falcon_local_files_only: bool = True
    falcon_min_dimension: int = Field(256, ge=64, le=4096)
    falcon_max_dimension: int = Field(1024, ge=128, le=8192)
    falcon_max_new_tokens: int = Field(1024, ge=32, le=4096)
    falcon_compile: bool = False
    falcon_coord_dedup_threshold: float = Field(0.01, ge=0.0, le=1.0)
    falcon_hr_upsample_ratio: int = Field(8, ge=1, le=32)
    falcon_segmentation_threshold: float = Field(0.3, ge=0.0, le=1.0)
    simplify_epsilon: float = Field(2.0, ge=0.0, le=100.0)
    force_annotation_lock: bool = False

    @root_validator_compat(pre=True)
    def _normalize_auto_label_payload_input(cls, values):  # noqa: N805
        if not isinstance(values, Mapping):
            return values
        data = dict(values)
        data["planner_model_variant"] = _normalize_qwen_variant_value(
            data.get("planner_model_variant"),
            default="auto",
        )
        for field in (
            "annotation_session_id",
            "planner_model_id",
            "falcon_model_id",
            "yolo_id",
            "rfdetr_id",
            "classifier_id",
        ):
            if data.get(field) is not None:
                data[field] = str(data.get(field)).strip() or None
        return data

    @root_validator_compat(skip_on_failure=True)
    def _validate_auto_label_request(cls, values):  # noqa: N805
        class_names = values.get("class_names")
        if isinstance(class_names, list):
            cleaned = [str(item or "").strip() for item in class_names if str(item or "").strip()]
            values["class_names"] = cleaned or None
        image_relpaths = values.get("image_relpaths")
        if isinstance(image_relpaths, list):
            seen = set()
            cleaned_relpaths = []
            for item in image_relpaths:
                normalized = str(item or "").strip().replace("\\", "/")
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                cleaned_relpaths.append(normalized)
            values["image_relpaths"] = cleaned_relpaths or None
        session_id = str(values.get("annotation_session_id") or "").strip()
        values["annotation_session_id"] = session_id or None
        return values


class CalibrationRequest(BaseModel):
    dataset_id: str
    max_images: Optional[int] = 2000
    seed: Optional[int] = 42
    recipe_mode: Optional[Literal["auto", "reuse_only", "force_rediscover"]] = "auto"
    lane_selection: Optional[Literal["window", "nonwindow", "compare_both"]] = "window"
    enable_yolo: Optional[bool] = True
    enable_rfdetr: Optional[bool] = True
    base_fp_ratio: Optional[float] = 0.2
    relax_fp_ratio: Optional[float] = 0.2
    recall_floor: Optional[float] = 0.6
    per_class_thresholds: Optional[bool] = True
    threshold_steps: Optional[int] = 200
    optimize_metric: Optional[str] = "f1"
    target_fp_ratio_by_label_json: Optional[str] = None
    min_recall_by_label_json: Optional[str] = None
    sam3_text_synonym_budget: Optional[int] = 10
    sam3_text_window_extension: Optional[bool] = True
    sam3_text_window_mode: Optional[Literal["grid", "sahi"]] = "grid"
    sam3_text_window_size: Optional[int] = None
    sam3_text_window_overlap: Optional[float] = None
    prepass_sam3_text_thr: Optional[float] = 0.2
    prepass_similarity_score: Optional[float] = 0.3
    sam3_score_thr: Optional[float] = 0.2
    sam3_mask_threshold: Optional[float] = 0.2
    image_embed_proj_dim: Optional[int] = 0
    image_embed_proj_seed: Optional[int] = 4242
    similarity_min_exemplar_score: Optional[float] = 0.6
    similarity_exemplar_count: Optional[int] = 3
    similarity_exemplar_strategy: Optional[Literal["top", "random", "diverse"]] = "top"
    similarity_exemplar_seed: Optional[int] = None
    similarity_exemplar_fraction: Optional[float] = 0.2
    similarity_exemplar_min: Optional[int] = 3
    similarity_exemplar_max: Optional[int] = 12
    similarity_exemplar_source_quota: Optional[int] = 1
    similarity_window_extension: Optional[bool] = True
    similarity_window_mode: Optional[Literal["grid", "sahi"]] = "grid"
    similarity_window_size: Optional[int] = None
    similarity_window_overlap: Optional[float] = None
    detector_conf: Optional[float] = 0.45
    sahi_window_size: Optional[int] = None
    sahi_overlap_ratio: Optional[float] = None
    classifier_id: Optional[str] = None
    support_iou: Optional[float] = 0.5
    context_radius: Optional[float] = 0.075
    label_iou: Optional[float] = 0.5
    eval_iou: Optional[float] = 0.5
    eval_iou_grid: Optional[str] = None
    dedupe_iou: Optional[float] = 0.75
    fusion_mode: Optional[Literal["primary", "wbf", "source_weighted"]] = "primary"
    cross_class_dedupe_enabled: Optional[bool] = False
    cross_class_dedupe_iou: Optional[float] = 0.8
    dedupe_iou_grid: Optional[str] = None
    scoreless_iou: Optional[float] = 0.0
    model_hidden: Optional[str] = "256,128"
    model_dropout: Optional[float] = 0.1
    model_epochs: Optional[int] = 20
    model_lr: Optional[float] = 1e-3
    model_weight_decay: Optional[float] = 1e-4
    model_seed: Optional[int] = 42
    calibration_model: Optional[Literal["mlp", "xgb"]] = "xgb"
    xgb_max_depth: Optional[int] = None
    xgb_n_estimators: Optional[int] = None
    xgb_learning_rate: Optional[float] = None
    xgb_subsample: Optional[float] = None
    xgb_colsample_bytree: Optional[float] = None
    xgb_min_child_weight: Optional[float] = None
    xgb_gamma: Optional[float] = None
    xgb_reg_lambda: Optional[float] = None
    xgb_reg_alpha: Optional[float] = None
    xgb_scale_pos_weight: Optional[float] = None
    xgb_tree_method: Optional[str] = None
    xgb_max_bin: Optional[int] = None
    xgb_early_stopping_rounds: Optional[int] = None
    xgb_log1p_counts: Optional[bool] = None
    xgb_standardize: Optional[bool] = None
    apply_default_ensemble_policy: Optional[bool] = True
    ensemble_policy_json: Optional[str] = None
    split_head_by_support: Optional[bool] = None
    train_sam3_text_quality: Optional[bool] = True
    sam3_text_quality_alpha: Optional[float] = None
    train_sam3_similarity_quality: Optional[bool] = None
    sam3_similarity_quality_alpha: Optional[float] = None
    policy_layer_variant: Optional[Literal["none", "bakeoff", "xgb", "lreg"]] = "none"


class AgentToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    call_id: Optional[str] = None


class AgentToolResult(BaseModel):
    name: str
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class AgentTraceEvent(BaseModel):
    step_id: int
    phase: Literal["intent", "tool_call", "tool_result", "merge"] = "intent"
    tool_name: Optional[str] = None
    summary: Optional[str] = None
    counts: Optional[Dict[str, int]] = None
    windows: Optional[List[Dict[str, Any]]] = None
    timestamp: Optional[float] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None
    model_output: Optional[str] = None


class QwenPrepassResponse(BaseModel):
    detections: List[Dict[str, Any]]
    trace: List[AgentTraceEvent]
    warnings: Optional[List[str]] = None
    caption: Optional[str] = None
    trace_path: Optional[str] = None
    trace_full_path: Optional[str] = None


class PromptHelperSuggestRequest(BaseModel):
    dataset_id: str
    max_synonyms: int = Field(3, ge=0, le=10)
    use_qwen: bool = True


class PromptHelperRequest(BaseModel):
    dataset_id: str
    sample_per_class: int = Field(10, ge=1, le=1000)
    max_synonyms: int = Field(3, ge=0, le=10)
    score_threshold: float = Field(0.2, ge=0.0, le=1.0)
    max_dets: int = Field(100, ge=1, le=2000)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    seed: int = 42
    use_qwen: bool = True
    prompts_by_class: Optional[Dict[int, List[str]]] = None


class PromptHelperSearchRequest(BaseModel):
    dataset_id: str
    sample_per_class: int = Field(20, ge=1, le=2000)
    negatives_per_class: int = Field(20, ge=0, le=2000)
    score_threshold: float = Field(0.2, ge=0.0, le=1.0)
    max_dets: int = Field(100, ge=1, le=2000)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    seed: int = 42
    precision_floor: float = Field(0.9, ge=0.0, le=1.0)
    prompts_by_class: Dict[int, List[str]]
    class_id: Optional[int] = None


class PromptRecipePrompt(BaseModel):
    prompt: str
    thresholds: Optional[List[float]] = None


class PromptRecipeRequest(BaseModel):
    dataset_id: str
    class_id: int
    prompts: List[PromptRecipePrompt]
    sample_size: int = Field(30, ge=1, le=5000)
    negatives: int = Field(0, ge=0, le=5000)
    max_dets: int = Field(100, ge=1, le=2000)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    seed: int = 42
    score_threshold: float = Field(0.2, ge=0.0, le=1.0)
    threshold_candidates: Optional[List[float]] = None


class PromptRecipeExpandRequest(BaseModel):
    dataset_id: str
    class_id: int
    base_prompts: List[str]
    max_new: int = Field(10, ge=0, le=50)


class AgentMiningRequest(BaseModel):
    dataset_id: str
    classes: Optional[List[int]] = None
    eval_image_count: int = Field(100, ge=1, le=50_000)
    split_seed: int = 42

    search_mode: Literal["steps"] = "steps"
    reuse_cache: bool = True

    steps_max_steps_per_recipe: int = Field(6, ge=1, le=50)
    steps_max_visual_seeds_per_step: int = Field(10, ge=0, le=500)
    steps_optimize_tier1: bool = False
    steps_optimize_tier1_eval_cap: int = Field(200, ge=10, le=50_000)
    steps_optimize_tier1_max_trials: int = Field(9, ge=1, le=256)
    steps_seed_eval_floor: Optional[float] = Field(None, ge=0.0, le=1.0)
    steps_seed_eval_max_results: Optional[int] = Field(None, ge=1, le=5000)
    steps_early_stop: bool = True
    steps_early_stop_mode: Literal["conservative", "balanced", "aggressive"] = "balanced"
    steps_prompt_prefilter: bool = True
    steps_prompt_prefilter_mode: Literal["conservative", "balanced", "aggressive"] = "balanced"
    steps_prompt_bg_drop: bool = True
    steps_prompt_bg_drop_mode: Literal["conservative", "balanced", "aggressive"] = "balanced"
    steps_optimize_tier2: bool = False
    steps_optimize_tier2_eval_cap: int = Field(200, ge=10, le=50_000)
    steps_optimize_tier2_max_trials: int = Field(12, ge=1, le=256)
    steps_refine_prompt_subset: bool = False
    steps_refine_prompt_subset_max_iters: int = Field(6, ge=0, le=100)
    steps_refine_prompt_subset_top_k: int = Field(6, ge=1, le=50)
    steps_optimize_global: bool = False
    steps_optimize_global_eval_caps: List[int] = Field(default_factory=lambda: [50, 200, 1000])
    steps_optimize_global_max_trials: int = Field(36, ge=1, le=4096)
    steps_optimize_global_keep_ratio: float = Field(0.5, ge=0.1, le=0.9)
    steps_optimize_global_rounds: int = Field(2, ge=1, le=20)
    steps_optimize_global_mutations_per_round: int = Field(24, ge=1, le=10_000)
    steps_optimize_global_max_steps_mutated: int = Field(2, ge=1, le=10)
    steps_optimize_global_enable_max_results: bool = False
    steps_optimize_global_enable_ordering: bool = False

    max_workers_per_device: int = Field(1, ge=1, le=8)
    max_workers: Optional[int] = Field(None, ge=1, le=256)

    text_prompts_by_class: Optional[Dict[int, List[str]]] = None
    prompt_llm_max_prompts: int = Field(10, ge=0, le=50)
    prompt_max_new_tokens: int = Field(160, ge=16, le=800)
    class_hints: Optional[Dict[str, str]] = None
    extra_prompts_by_class: Optional[Dict[str, List[str]]] = None

    clip_head_classifier_path: Optional[str] = None
    clip_head_min_prob: float = Field(0.5, ge=0.0, le=1.0)
    clip_head_margin: float = Field(0.0, ge=0.0, le=1.0)
    clip_head_auto_tune: bool = True
    clip_head_tune_margin: bool = True
    clip_head_target_precision: float = Field(0.75, ge=0.0, le=1.0)
    clip_head_background_guard: bool = True
    clip_head_background_margin: float = Field(0.0, ge=0.0, le=1.0)
    clip_head_background_auto_tune: bool = True
    clip_head_background_apply: Literal["seed", "final", "both"] = "final"
    clip_head_background_penalty: float = Field(0.0, ge=0.0, le=2.0)

    steps_hard_negative_export: bool = True
    steps_hard_negative_max_crops: int = Field(200, ge=0, le=5000)
    steps_hard_negative_min_prob: float = Field(0.1, ge=0.0, le=1.0)

    seed_threshold: float = Field(0.02, ge=0.0, le=1.0)
    expand_threshold: float = Field(0.15, ge=0.0, le=1.0)
    max_visual_seeds: int = Field(25, ge=0, le=500)
    seed_dedupe_iou: float = Field(0.9, ge=0.0, le=1.0)
    dedupe_iou: float = Field(0.5, ge=0.0, le=1.0)
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0)
    similarity_score: float = Field(0.25, ge=0.0, le=1.0)
    max_results: int = Field(1000, ge=1, le=5000)
    min_size: int = Field(0, ge=0, le=10_000)
    simplify_epsilon: float = Field(0.0, ge=0.0, le=1000.0)

    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
