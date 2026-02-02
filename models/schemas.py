"""Shared Pydantic schemas (requests/responses)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple, Literal

from pydantic import BaseModel, Field, root_validator


class Base64Payload(BaseModel):
    image_base64: str
    uuid: Optional[str] = None
    background_guard: Optional[bool] = None


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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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
    threshold: float = 0.5
    mask_threshold: float = 0.5
    simplify_epsilon: Optional[float] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None
    max_results: Optional[int] = None
    min_size: Optional[int] = None

    @root_validator(skip_on_failure=True)
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
                    raise ValueError(f"invalid_{key}")
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
    def _validate_hint(cls, values):  # noqa: N805
        label = (values.get("label") or "").strip()
        if not label:
            raise ValueError("label_required")
        values["label"] = label
        bbox = values.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError("invalid_bbox")
            cleaned = []
            for val in bbox:
                try:
                    cleaned.append(float(val))
                except (TypeError, ValueError):
                    cleaned.append(0.0)
            values["bbox"] = cleaned
        return values


class QwenCaptionRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    user_prompt: Optional[str] = None
    label_hints: Optional[List[QwenCaptionHint]] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    include_counts: Optional[bool] = True
    include_coords: Optional[bool] = True
    max_boxes: Optional[int] = 0
    max_new_tokens: Optional[int] = 256
    model_variant: Optional[Literal["auto", "Instruct", "Thinking"]] = "auto"
    model_id: Optional[str] = None
    final_answer_only: Optional[bool] = True
    two_stage_refine: Optional[bool] = False
    caption_mode: Optional[Literal["full", "windowed"]] = "full"
    window_size: Optional[int] = None
    window_overlap: Optional[float] = None
    restrict_to_labels: Optional[bool] = True
    caption_all_windows: Optional[bool] = True
    unload_others: Optional[bool] = False
    force_unload: Optional[bool] = None
    multi_model_cache: Optional[bool] = False
    fast_mode: Optional[bool] = False
    use_sampling: Optional[bool] = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    labelmap_glossary: Optional[str] = None

    @root_validator(skip_on_failure=True)
    def _validate_caption_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        user_prompt = (values.get("user_prompt") or "").strip()
        values["user_prompt"] = user_prompt
        max_boxes = values.get("max_boxes")
        if max_boxes is not None:
            try:
                max_boxes_val = int(max_boxes)
            except (TypeError, ValueError):
                max_boxes_val = 0
            values["max_boxes"] = max(0, min(max_boxes_val, 200))
        else:
            values["max_boxes"] = 0
        max_tokens = values.get("max_new_tokens")
        if max_tokens is not None:
            try:
                max_tokens_val = int(max_tokens)
            except (TypeError, ValueError):
                max_tokens_val = 256
            values["max_new_tokens"] = max(32, min(max_tokens_val, 2000))
        else:
            values["max_new_tokens"] = 256
        temp = values.get("temperature")
        if temp is not None:
            try:
                values["temperature"] = float(temp)
            except (TypeError, ValueError):
                values["temperature"] = None
        top_p = values.get("top_p")
        if top_p is not None:
            try:
                values["top_p"] = float(top_p)
            except (TypeError, ValueError):
                values["top_p"] = None
        top_k = values.get("top_k")
        if top_k is not None:
            try:
                values["top_k"] = int(top_k)
            except (TypeError, ValueError):
                values["top_k"] = None
        presence = values.get("presence_penalty")
        if presence is not None:
            try:
                values["presence_penalty"] = float(presence)
            except (TypeError, ValueError):
                values["presence_penalty"] = None
        for key in ("image_width", "image_height"):
            val = values.get(key)
            if val is None:
                continue
            try:
                values[key] = max(1, int(val))
            except (TypeError, ValueError):
                values[key] = None
        caption_mode = (values.get("caption_mode") or "full").strip().lower()
        if caption_mode == "hybrid":
            caption_mode = "windowed"
        if caption_mode not in {"full", "windowed"}:
            caption_mode = "full"
        values["caption_mode"] = caption_mode
        window_size = values.get("window_size")
        if window_size is not None:
            try:
                values["window_size"] = max(64, int(window_size))
            except (TypeError, ValueError):
                values["window_size"] = None
        window_overlap = values.get("window_overlap")
        if window_overlap is not None:
            try:
                values["window_overlap"] = float(window_overlap)
            except (TypeError, ValueError):
                values["window_overlap"] = None
        model_id = (values.get("model_id") or "").strip()
        values["model_id"] = model_id or None
        glossary = values.get("labelmap_glossary")
        if glossary is not None:
            values["labelmap_glossary"] = str(glossary).strip() or None
        values["final_answer_only"] = bool(values.get("final_answer_only", True))
        values["two_stage_refine"] = bool(values.get("two_stage_refine", False))
        return values


class QwenCaptionResponse(BaseModel):
    caption: str
    used_counts: Dict[str, int] = Field(default_factory=dict)
    used_boxes: int
    truncated: bool


class QwenPromptSection(BaseModel):
    base_prompt: str
    default_image_type: str = "image"
    default_extra_context: str = ""

    @root_validator(skip_on_failure=True)
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


class QwenRuntimeSettingsUpdate(BaseModel):
    trust_remote_code: Optional[bool] = None


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
    labelmap_entries: List[str] = []
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

    @root_validator(skip_on_failure=True)
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
    devices: Optional[List[int]] = None
    seed: Optional[int] = None
    augmentations: Optional[Dict[str, Any]] = None
    accept_tos: Optional[bool] = None

    @root_validator(skip_on_failure=True)
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
    devices: Optional[List[int]] = None
    seed: Optional[int] = None
    export_onnx: Optional[bool] = None
    accept_tos: Optional[bool] = None

    @root_validator(skip_on_failure=True)
    def _validate_dataset_fields(cls, values):  # noqa: N805
        if not (values.get("dataset_id") or values.get("dataset_root")):
            raise ValueError("dataset_id_or_root_required")
        return values


class YoloHeadGraftDryRunRequest(BaseModel):
    base_run_id: str
    dataset_id: Optional[str] = None
    dataset_root: Optional[str] = None

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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
    positive_points: List[List[float]] = []
    negative_points: List[List[float]] = []
    uuid: Optional[str] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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
    similarity_mid_conf_low: Optional[float] = None
    similarity_mid_conf_high: Optional[float] = None
    similarity_mid_conf_class_count: Optional[int] = None
    similarity_window_mode: Optional[Literal["grid", "sahi"]] = "grid"
    similarity_window_size: Optional[int] = None
    similarity_window_overlap: Optional[float] = None
    similarity_window_extension: Optional[bool] = False
    prepass_mode: Optional[str] = "ensemble_sahi_sam3_text_similarity"
    prepass_only: Optional[bool] = True
    prepass_finalize: Optional[bool] = True
    prepass_keep_all: Optional[bool] = False
    prepass_sam3_text_thr: Optional[float] = 0.2
    prepass_similarity_score: Optional[float] = 0.3
    prepass_similarity_per_class: Optional[int] = None
    prepass_inspect_topk: Optional[int] = None
    prepass_inspect_score: Optional[float] = None
    prepass_inspect_quadrants: Optional[bool] = None
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
    max_detections: Optional[int] = 2000
    iou: Optional[float] = 0.75
    cross_iou: Optional[float] = None
    max_new_tokens: Optional[int] = 4096
    thinking_effort: Optional[float] = None
    thinking_scale_factor: Optional[float] = None
    immediate_action_bias: Optional[bool] = True
    immediate_action_min_chars: Optional[int] = 200
    immediate_action_min_seconds: Optional[float] = 2.0
    immediate_action_logit_bias: Optional[float] = 6.0
    trace_verbose: Optional[bool] = False


class CalibrationRequest(BaseModel):
    dataset_id: str
    max_images: Optional[int] = 2000
    seed: Optional[int] = 42
    enable_yolo: Optional[bool] = True
    enable_rfdetr: Optional[bool] = True
    base_fp_ratio: Optional[float] = 0.2
    relax_fp_ratio: Optional[float] = 0.2
    recall_floor: Optional[float] = 0.6
    per_class_thresholds: Optional[bool] = True
    threshold_steps: Optional[int] = 200
    optimize_metric: Optional[str] = "f1"
    sam3_text_synonym_budget: Optional[int] = 10
    sam3_text_window_extension: Optional[bool] = True
    sam3_text_window_mode: Optional[Literal["grid", "sahi"]] = "grid"
    sam3_text_window_size: Optional[int] = None
    sam3_text_window_overlap: Optional[float] = None
    prepass_sam3_text_thr: Optional[float] = 0.2
    prepass_similarity_score: Optional[float] = 0.3
    sam3_score_thr: Optional[float] = 0.2
    sam3_mask_threshold: Optional[float] = 0.2
    similarity_min_exemplar_score: Optional[float] = 0.6
    similarity_window_extension: Optional[bool] = False
    detector_conf: Optional[float] = 0.45
    sahi_window_size: Optional[int] = None
    sahi_overlap_ratio: Optional[float] = None
    classifier_id: Optional[str] = None
    support_iou: Optional[float] = 0.5
    context_radius: Optional[float] = 0.075
    label_iou: Optional[float] = 0.9
    eval_iou: Optional[float] = 0.5
    eval_iou_grid: Optional[str] = None
    dedupe_iou: Optional[float] = 0.75
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
