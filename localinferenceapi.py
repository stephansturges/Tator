"""Unified FastAPI backend and inference orchestration."""

from __future__ import annotations

import base64, hashlib, io, zipfile, uuid, os, tempfile, shutil, time, logging, subprocess, sys, json, re, signal, random, gc, queue, functools, math
from contextvars import ContextVar
from pathlib import Path
import numpy as np
import yaml
from typing import Optional, List, Dict, Tuple, Any, Literal, Sequence, Mapping, Callable, Set
from collections import deque
import torch, clip, joblib
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Query, Body, HTTPException, Request
from api.detectors_default import build_detectors_default_router
from api.datasets import build_datasets_router
from api.glossaries import build_glossaries_router
from api.predictor_settings import build_predictor_settings_router
from api.runtime import build_runtime_router
from api.system import build_system_router
from api.qwen_status import build_qwen_status_router
from api.sam_slots import build_sam_slots_router
from api.qwen_training import build_qwen_training_router
from api.qwen_models import build_qwen_models_router
from api.qwen_datasets import build_qwen_datasets_router
from api.clip_active_model import build_clip_active_model_router
from api.sam_preload import build_sam_preload_router
from api.qwen_infer import build_qwen_infer_router
from api.qwen_caption import build_qwen_caption_router
from api.calibration import build_calibration_router
from api.prepass import build_prepass_router
from api.qwen_prepass import build_qwen_prepass_router
from api.clip_registry import build_clip_registry_router
from api.clip_training import build_clip_training_router
from api.fs_upload import build_fs_upload_router
from api.predict_base64 import build_predict_base64_router
from api.crop_zip import build_crop_zip_router
from api.agent_mining import build_agent_mining_router
from api.sam3_prompts import build_sam3_prompts_router
from api.sam3_prompt_helper import build_sam3_prompt_helper_router
from api.sam3_training import build_sam3_training_router
from api.sam3_registry import build_sam3_registry_router
from api.sam3_datasets import build_sam3_datasets_router
from api.segmentation_build import build_segmentation_build_router
from api.yolo_training import build_yolo_training_router
from api.rfdetr_training import build_rfdetr_training_router
from api.rfdetr import build_rfdetr_router
from api.yolo import build_yolo_router
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import (
    Base64Payload,
    PredictResponse,
    BboxModel,
    CropImage,
    CropZipRequest,
    PointPrompt,
    BboxPrompt,
    SamPreloadRequest,
    SamPreloadResponse,
    SamSlotStatus,
    SamActivateRequest,
    SamActivateResponse,
    PredictorSettings,
    Sam3VisualPrompt,
    SamPointAutoResponse,
    QwenDetection,
    QwenInferenceRequest,
    QwenInferenceResponse,
    QwenCaptionHint,
    QwenCaptionRequest,
    QwenCaptionResponse,
    QwenPrepassRequest,
    QwenPrepassResponse,
    CalibrationRequest,
    AgentToolCall,
    AgentToolResult,
    AgentTraceEvent,
    QwenPromptSection,
    QwenPromptConfig,
    QwenRuntimeSettings,
    QwenRuntimeSettingsUpdate,
    Sam3ModelActivateRequest,
    QwenModelActivateRequest,
    ActiveModelRequest,
    ActiveModelResponse,
    SegmentationBuildRequest,
    QwenTrainRequest,
    Sam3TrainRequest,
    YoloTrainRequest,
    YoloHeadGraftRequest,
    YoloHeadGraftDryRunRequest,
    RfDetrTrainRequest,
    YoloActiveRequest,
    RfDetrActiveRequest,
    DetectorDefaultRequest,
    YoloRegionRequest,
    YoloRegionDetection,
    YoloRegionResponse,
    YoloFullRequest,
    YoloWindowedRequest,
    RfDetrRegionRequest,
    RfDetrRegionDetection,
    RfDetrRegionResponse,
    RfDetrFullRequest,
    RfDetrWindowedRequest,
    Sam3TextPromptResponse,
    Sam3TextPromptAutoResponse,
    PredictorSettingsUpdate,
    MultiPointPrompt,
    YoloBboxOutput,
    Sam3TextPrompt,
    PromptHelperSuggestRequest,
    PromptHelperPreset,
    PromptHelperRequest,
    PromptHelperSearchRequest,
    PromptRecipePrompt,
    PromptRecipeRequest,
    PromptRecipeExpandRequest,
    PrepassRecipeRequest,
    PrepassRecipeResponse,
    AgentApplyImageRequest,
    AgentApplyChainStep,
    AgentCascadeDedupeConfig,
    AgentApplyImageChainRequest,
    AgentRecipeExportRequest,
    AgentCascadeSaveRequest,
    AgentMiningRequest,
)
from omegaconf import OmegaConf
import psutil
try:
    from packaging import version as packaging_version
except Exception:  # noqa: BLE001
    packaging_version = None
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_412_PRECONDITION_FAILED,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_404_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_428_PRECONDITION_REQUIRED,
    HTTP_409_CONFLICT,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from utils.io import (
    _ensure_directory,
    _load_json_metadata,
    _read_csv_last_row,
    _sanitize_yolo_run_id as _sanitize_yolo_run_id_impl,
    _compute_dir_signature as _compute_dir_signature_impl,
    _dir_size_bytes as _dir_size_bytes_impl,
    _path_is_within_root_impl as _path_is_within_root_impl,
)
_sanitize_rfdetr_run_id_impl = _sanitize_yolo_run_id_impl
from utils.network import _find_free_port_impl as _find_free_port_impl
from utils.coco import (
    _ensure_coco_supercategory_impl,
    _write_coco_annotations_impl,
    _encode_binary_mask_impl,
    _decode_binary_mask_impl,
    _mask_to_polygon_impl,
)
from utils.image import (
    _slice_image_sahi,
    _decode_image_base64_impl,
    _image_path_for_label_impl,
)
from utils.labels import (
    _read_labelmap_lines,
    _load_labelmap_file,
    _normalize_class_name_for_match,
    _apply_expected_labelmap_warnings,
    _raise_on_labelmap_mismatch,
    _agent_label_color_map,
    _agent_label_prefix_map,
    _agent_overlay_key_text,
    _agent_fuzzy_align_label,
)
from utils.classifier_utils import (
    _clip_head_background_indices,
    _agent_background_classes_from_head,
    _find_clip_head_target_index,
)
from utils.parsing import (
    _coerce_int,
    _coerce_float,
    _normalise_optional_path,
    _parse_bool,
    _safe_run_name,
    _normalize_device_list,
    _agent_extract_json_array,
)
from utils.gpu import _validate_cuda_device_ids_impl as _validate_cuda_device_ids_impl
from utils.errors import _agent_error_payload, _agent_error_from_detail
from utils.hashing import _stable_hash_impl as _stable_hash_impl
from utils.glossary import (
    _glossary_label_key,
    _normalize_labelmap_glossary,
    _default_agent_glossary_for_labelmap,
    _glossary_preview,
)
from utils.datasets import _iter_yolo_images
from services.prepass_config import (
    _normalize_recipe_thresholds,
    _require_sam3_for_prepass as _require_sam3_for_prepass_impl,
)
from services.prepass_recipes import (
    _write_prepass_recipe_meta,
    _load_prepass_recipe_meta,
    _parse_agent_recipe_schema_version as _parse_agent_recipe_schema_version_impl,
    _classify_agent_recipe_mode as _classify_agent_recipe_mode_impl,
    _validate_agent_recipe_structure as _validate_agent_recipe_structure_impl,
    _save_exemplar_crop_impl as _save_exemplar_crop_impl,
    _delete_agent_recipe_impl as _delete_agent_recipe_impl,
    _list_agent_recipes_impl as _list_agent_recipes_impl,
    _persist_agent_recipe_impl as _persist_agent_recipe_impl,
    _load_agent_recipe_impl as _load_agent_recipe_impl,
    _load_agent_recipe_json_only_impl as _load_agent_recipe_json_only_impl,
    _ensure_recipe_zip_impl as _ensure_recipe_zip_impl,
    _import_agent_recipe_zip_bytes_impl as _import_agent_recipe_zip_bytes_impl,
    _list_prepass_recipes_impl as _list_prepass_recipes_impl,
    _collect_recipe_assets_impl as _collect_recipe_assets_impl,
    _copy_tree_filtered_impl as _copy_tree_filtered_impl,
    _sha256_path_impl as _sha256_path_impl,
    _import_prepass_recipe_from_zip_impl as _import_prepass_recipe_from_zip_impl,
    _export_prepass_recipe_impl as _export_prepass_recipe_impl,
    _save_prepass_recipe_impl as _save_prepass_recipe_impl,
    _delete_prepass_recipe_impl as _delete_prepass_recipe_impl,
    _get_prepass_recipe_impl as _get_prepass_recipe_impl,
)
from services.agent_cascades import (
    _persist_agent_cascade_impl as _persist_agent_cascade_impl,
    _load_agent_cascade_impl as _load_agent_cascade_impl,
    _list_agent_cascades_impl as _list_agent_cascades_impl,
    _delete_agent_cascade_impl as _delete_agent_cascade_impl,
    _ensure_cascade_zip_impl as _ensure_cascade_zip_impl,
    _import_agent_cascade_zip_bytes_impl as _import_agent_cascade_zip_bytes_impl,
)
from services.prompt_helper_presets import (
    _list_prompt_helper_presets_impl as _list_prompt_helper_presets_impl,
    _load_prompt_helper_preset_impl as _load_prompt_helper_preset_impl,
    _save_prompt_helper_preset_impl as _save_prompt_helper_preset_impl,
)
from services.prompt_helper import _serialize_prompt_helper_job_impl as _serialize_prompt_helper_job_impl
from services.classifier_jobs import (
    _clip_job_append_metric_impl as _clip_job_append_metric_impl,
    _clip_job_log_impl as _clip_job_log_impl,
    _clip_job_update_impl as _clip_job_update_impl,
    _serialize_clip_job_impl as _serialize_clip_job_impl,
)
from services.detector_jobs import (
    _rfdetr_job_append_metric_impl as _rfdetr_job_append_metric_impl,
    _rfdetr_job_log_impl as _rfdetr_job_log_impl,
    _rfdetr_job_update_impl as _rfdetr_job_update_impl,
    _serialize_rfdetr_job_impl as _serialize_rfdetr_job_impl,
    _serialize_yolo_head_graft_job_impl as _serialize_yolo_head_graft_job_impl,
    _serialize_yolo_job_impl as _serialize_yolo_job_impl,
    _yolo_job_append_metric_impl as _yolo_job_append_metric_impl,
    _yolo_job_log_impl as _yolo_job_log_impl,
    _yolo_job_update_impl as _yolo_job_update_impl,
    _yolo_head_graft_job_log_impl as _yolo_head_graft_job_log_impl,
    _yolo_head_graft_job_update_impl as _yolo_head_graft_job_update_impl,
    _yolo_head_graft_audit_impl as _yolo_head_graft_audit_impl,
    _yolo_head_graft_force_stop_impl as _yolo_head_graft_force_stop_impl,
)
from services.qwen_jobs import (
    _log_qwen_get_request_impl as _log_qwen_get_request_impl,
    _qwen_job_append_metric_impl as _qwen_job_append_metric_impl,
    _qwen_job_log_impl as _qwen_job_log_impl,
    _qwen_job_update_impl as _qwen_job_update_impl,
    _summarize_qwen_metric_impl as _summarize_qwen_metric_impl,
    _serialize_qwen_job_impl as _serialize_qwen_job_impl,
)
from services.qwen_runtime import (
    _ensure_qwen_ready_for_caption_impl as _ensure_qwen_ready_for_caption_impl,
    _reset_qwen_runtime_impl as _reset_qwen_runtime_impl,
    _unload_qwen_runtime_impl as _unload_qwen_runtime_impl,
    _evict_qwen_caption_entry_impl as _evict_qwen_caption_entry_impl,
    _resolve_qwen_device_impl as _resolve_qwen_device_impl,
)
from services.qwen_generation import (
    _BASE_LOGITS_PROCESSOR,
    ThinkingEffortProcessor,
    ImmediateActionBiasProcessor,
    _resolve_qwen_max_seq_len,
    _qwen_estimate_vision_tokens,
    _qwen_effective_input_len,
    _qwen_supports_presence_penalty,
    _qwen_find_end_think_token_id,
    _qwen_build_thinking_effort_processor,
    _qwen_build_immediate_action_processor,
    _qwen_append_logits_processor,
)
from services.sam3_jobs import (
    _sam3_job_append_metric_impl as _sam3_job_append_metric_impl,
    _sam3_job_log_impl as _sam3_job_log_impl,
    _sam3_job_update_impl as _sam3_job_update_impl,
    _serialize_sam3_job_impl as _serialize_sam3_job_impl,
)
from services.sam3_runs import (
    _active_run_paths_for_variant_impl as _active_run_paths_for_variant_impl,
    _describe_run_dir_impl as _describe_run_dir_impl,
    _list_sam3_runs_impl as _list_sam3_runs_impl,
    _run_dir_for_request_impl as _run_dir_for_request_impl,
    _delete_run_scope_impl as _delete_run_scope_impl,
)
from services.sam3_runtime import (
    _sam3_clear_device_pinned_caches_impl as _sam3_clear_device_pinned_caches_impl,
    _set_sam3_device_pref_impl as _set_sam3_device_pref_impl,
    _resolve_sam3_device_impl as _resolve_sam3_device_impl,
    _resolve_sam3_mining_devices_impl as _resolve_sam3_mining_devices_impl,
    _reset_sam3_runtime_impl as _reset_sam3_runtime_impl,
    _build_backend_for_variant_impl as _build_backend_for_variant_impl,
    _ensure_sam3_text_runtime_impl as _ensure_sam3_text_runtime_impl,
)
from services.runtime_unload import (
    _unload_sam3_text_runtime_impl as _unload_sam3_text_runtime_impl,
    _unload_dinov3_backbone_impl as _unload_dinov3_backbone_impl,
    _unload_detector_inference_impl as _unload_detector_inference_impl,
    _unload_non_qwen_runtimes_impl as _unload_non_qwen_runtimes_impl,
    _unload_inference_runtimes_impl as _unload_inference_runtimes_impl,
    _prepare_for_training_impl as _prepare_for_training_impl,
    _finalize_training_environment_impl as _finalize_training_environment_impl,
)
from services.dinov3_runtime import (
    _dinov3_resolve_device_impl as _dinov3_resolve_device_impl,
)
from services.clip_runtime import (
    _suspend_clip_backbone_impl as _suspend_clip_backbone_impl,
    _resume_clip_backbone_impl as _resume_clip_backbone_impl,
)
from services.classifier_runtime import (
    _resume_classifier_backbone_impl as _resume_classifier_backbone_impl,
)
from services.segmentation import (
    _seg_job_log_impl as _seg_job_log_impl,
    _seg_job_update_impl as _seg_job_update_impl,
    _serialize_seg_job_impl as _serialize_seg_job_impl,
)
from services.datasets import (
    _load_dataset_glossary,
    _load_qwen_labelmap,
    _agent_load_labelmap_meta as _load_labelmap_meta,
    _detect_yolo_layout_impl as _detect_yolo_layout_impl,
    _yolo_labels_have_polygons_impl as _yolo_labels_have_polygons_impl,
    _resolve_dataset_entry_impl as _resolve_dataset_entry_impl,
    _resolve_dataset_legacy_impl as _resolve_dataset_legacy_impl,
    _resolve_sam3_or_qwen_dataset_impl as _resolve_sam3_or_qwen_dataset_impl,
    _yolo_resolve_split_paths_impl as _yolo_resolve_split_paths_impl,
    _resolve_yolo_training_dataset_impl as _resolve_yolo_training_dataset_impl,
    _resolve_rfdetr_training_dataset_impl as _resolve_rfdetr_training_dataset_impl,
    _compute_labelmap_hash_impl as _compute_labelmap_hash_impl,
    _compute_dataset_signature_impl as _compute_dataset_signature_impl,
    _load_registry_dataset_metadata_impl as _load_registry_dataset_metadata_impl,
    _persist_dataset_metadata_impl as _persist_dataset_metadata_impl,
    _coerce_dataset_metadata_impl as _coerce_dataset_metadata_impl,
    _load_qwen_dataset_metadata_impl as _load_qwen_dataset_metadata_impl,
    _load_sam3_dataset_metadata_impl as _load_sam3_dataset_metadata_impl,
    _persist_sam3_dataset_metadata_impl as _persist_sam3_dataset_metadata_impl,
    _count_dataset_images_impl as _count_dataset_images_impl,
    _count_caption_labels_impl as _count_caption_labels_impl,
    _list_all_datasets_impl as _list_all_datasets_impl,
    _collect_labels_from_qwen_jsonl_impl as _collect_labels_from_qwen_jsonl_impl,
    _discover_yolo_labelmap_impl as _discover_yolo_labelmap_impl,
    _convert_yolo_dataset_to_coco_impl as _convert_yolo_dataset_to_coco_impl,
    _convert_qwen_dataset_to_coco_impl as _convert_qwen_dataset_to_coco_impl,
    _convert_coco_dataset_to_yolo_impl as _convert_coco_dataset_to_yolo_impl,
    _resolve_sam3_dataset_meta_impl as _resolve_sam3_dataset_meta_impl,
    _load_coco_index_impl as _load_coco_index_impl,
)

_resolve_sam3_or_qwen_dataset = _resolve_sam3_or_qwen_dataset_impl
from services.prepass import (
    _agent_det_score,
    _agent_cluster_match,
    _agent_source_counts,
    _agent_format_source_counts,
    _agent_label_counts_summary,
    _agent_select_similarity_exemplars as _agent_select_similarity_exemplars_impl,
    _build_deep_prepass_runners_impl,
)
from services.cluster_helpers import _cluster_label_counts, _cluster_summaries
from services.context_store import (
    _agent_context_store as _agent_context_store_impl,
    _agent_context_chunk as _agent_context_chunk_impl,
)
from services.tile_context import _build_tile_context_payloads as _build_tile_context_payloads_impl
from services.prepass_grid import (
    _agent_grid_spec_for_payload,
    _agent_grid_cell_xyxy,
    _agent_grid_cell_for_window_bbox,
    _agent_grid_cell_for_detection,
    _agent_grid_usage_rows,
    _agent_grid_usage_text,
    _agent_grid_label_counts,
    _agent_tool_grid_cell_from_args as _grid_cell_from_args,
    _agent_record_grid_tool_usage as _record_grid_usage,
)
from utils.coords import (
    _xyxy_to_qwen_bbox,
    _qwen_bbox_to_xyxy,
    _normalize_window_xyxy,
    _window_local_bbox_2d_to_full_xyxy,
    _resolve_agent_bbox_xyxy,
    _agent_round_bbox_2d,
    _agent_clip_xyxy,
    _agent_expand_window_xyxy,
    _agent_xyxy_to_xywh,
    _xywh_to_xyxy,
    _yolo_to_xyxy,
    _yolo_to_xyxy_int,
    _xyxy_to_yolo_norm,
    _xyxy_to_yolo_norm_list,
    _mask_to_bounding_box,
    _agent_det_payload,
    _extract_numeric_sequence,
    _scale_bbox_to_image,
    _scale_point_to_image,
)
from utils.overlay import (
    _agent_render_detection_overlay,
    _agent_render_grid_overlay,
    _agent_overlay_labels,
)
from utils.trace_utils import (
    _agent_trace_sanitize_payload,
    _agent_trace_full_jsonable,
)
from services.readable import (
    _agent_readable_banner,
    _agent_detection_summary_lines,
    _agent_clean_observation_text,
    _agent_readable_format_bbox,
    _agent_readable_write as _agent_readable_write_impl,
)
from services.prepass_windows import _agent_sam3_text_windows
from services.prepass_similarity import (
    _agent_run_similarity_global,
    _agent_run_similarity_expansion,
)
from services.prepass_provenance import (
    _agent_attach_provenance,
    _agent_finalize_provenance,
)
from services.sam3_synonyms import _agent_generate_sam3_synonyms, _sam3_prompt_variants
from services.cluster_handles import (
    _agent_refresh_handle_index as _refresh_handle_index,
    _agent_cluster_handle as _cluster_handle,
    _agent_cluster_id_from_handle as _cluster_id_from_handle,
    _agent_handles_from_cluster_ids as _handles_from_cluster_ids,
    _agent_cluster_ids_from_handles as _cluster_ids_from_handles,
)
from services.classifier_select import (
    _agent_default_classifier_for_dataset as _select_default_classifier,
    _agent_classifier_matches_labelmap as _classifier_matches_labelmap,
    _agent_classifier_classes_for_path as _classifier_classes_for_path,
)
from services.classifier_batch import (
    _resolve_classifier_batch_size as _resolve_classifier_batch,
)
from services.classifier import (
    _predict_proba_batched_impl as _predict_proba_batched_impl,
    _resolve_agent_clip_classifier_path_impl as _resolve_agent_clip_classifier_path_impl,
    _load_clip_head_from_classifier_impl as _load_clip_head_from_classifier_impl,
    _resolve_head_normalize_embeddings_impl as _resolve_head_normalize_embeddings_impl,
    _resolve_active_head_normalize_embeddings_impl as _resolve_active_head_normalize_embeddings_impl,
    _infer_clip_model_from_embedding_dim_impl as _infer_clip_model_from_embedding_dim_impl,
    _load_labelmap_simple_impl as _load_labelmap_simple_impl,
    _validate_clip_dataset_impl as _validate_clip_dataset_impl,
    _resolve_clip_labelmap_path_impl as _resolve_clip_labelmap_path_impl,
    _find_labelmap_for_classifier_impl as _find_labelmap_for_classifier_impl,
    _list_clip_labelmaps_impl as _list_clip_labelmaps_impl,
    _list_clip_classifiers_impl as _list_clip_classifiers_impl,
)
from services.overlay_tools import (
    _agent_overlay_base_image as _overlay_base_image,
    _agent_overlay_crop_xyxy as _overlay_crop_xyxy,
)
from services.overlay_views import (
    _view_cell_raw as _view_cell_raw_impl,
    _view_cell_overlay as _view_cell_overlay_impl,
    _view_full_overlay as _view_full_overlay_impl,
)
from services.detector_merge import (
    _agent_merge_detections as _merge_detections,
    _merge_detections_nms as _merge_detections_nms,
)
from services.detector_params import (
    _clamp_conf_value as _clamp_conf_value_impl,
    _clamp_iou_value as _clamp_iou_value_impl,
    _clamp_max_det_value as _clamp_max_det_value_impl,
    _clamp_slice_params as _clamp_slice_params_impl,
)

# Legacy aliases used across handlers.
_clamp_conf_value = _clamp_conf_value_impl
_clamp_iou_value = _clamp_iou_value_impl
_clamp_max_det_value = _clamp_max_det_value_impl
_clamp_slice_params = _clamp_slice_params_impl
from services.calibration_helpers import (
    _calibration_list_images as _calibration_list_images_impl,
    _calibration_sample_images as _calibration_sample_images_impl,
    _calibration_hash_payload as _calibration_hash_payload_impl,
    _calibration_safe_link as _calibration_safe_link_impl,
    _calibration_write_record_atomic as _calibration_write_record_atomic_impl,
    _calibration_update as _calibration_update_impl,
    _calibration_cache_image as _calibration_cache_image_impl,
    _calibration_prepass_worker as _calibration_prepass_worker_impl,
)
from services.calibration import (
    CalibrationJob,
    _serialize_calibration_job as _serialize_calibration_job_impl,
    _run_calibration_job as _run_calibration_job_impl,
    _start_calibration_job as _start_calibration_job_impl,
    _cancel_calibration_job as _cancel_calibration_job_impl,
)
from services.calibration_metrics import (
    _build_gt_index_for_class_impl as _build_gt_index_for_class_impl,
    _evaluate_prompt_candidate_impl as _evaluate_prompt_candidate_impl,
    _collect_prompt_detections_impl as _collect_prompt_detections_impl,
    _build_prompt_recipe_impl as _build_prompt_recipe_impl,
)
from services.qwen import (
    _extract_balanced_json as _extract_balanced_json_impl,
    _generate_qwen_text as _generate_qwen_text_impl,
    _caption_glossary_map as _caption_glossary_map_impl,
    _caption_preferred_label as _caption_preferred_label_impl,
    _build_qwen_caption_prompt as _build_qwen_caption_prompt_impl,
    _collapse_whitespace as _collapse_whitespace_impl,
    _extract_caption_from_text as _extract_caption_from_text_impl,
    _caption_needs_english_rewrite as _caption_needs_english_rewrite_impl,
    _caption_starts_generic as _caption_starts_generic_impl,
    _caption_missing_labels as _caption_missing_labels_impl,
    _caption_needs_refine as _caption_needs_refine_impl,
    _sanitize_qwen_caption as _sanitize_qwen_caption_impl,
    _thinking_caption_needs_cleanup as _thinking_caption_needs_cleanup_impl,
    _caption_needs_completion as _caption_needs_completion_impl,
    _caption_has_meta as _caption_has_meta_impl,
    _caption_needs_short_form as _caption_needs_short_form_impl,
    _allowed_caption_labels_impl as _allowed_caption_labels_impl,
    _caption_is_degenerate_impl as _caption_is_degenerate_impl,
    _resolve_qwen_caption_decode as _resolve_qwen_caption_decode_impl,
    _window_positions_impl as _window_positions_impl,
    _adjust_prompt_for_thinking as _adjust_prompt_for_thinking_impl,
    _run_qwen_caption_cleanup as _run_qwen_caption_cleanup_impl,
    _run_qwen_caption_merge as _run_qwen_caption_merge_impl,
    _resolve_qwen_window_size as _resolve_qwen_window_size_impl,
    _resolve_qwen_window_overlap as _resolve_qwen_window_overlap_impl,
    _resolve_qwen_variant_model_id_impl as _resolve_qwen_variant_model_id_impl,
    _strip_qwen_model_suffix_impl as _strip_qwen_model_suffix_impl,
    _format_qwen_load_error_impl as _format_qwen_load_error_impl,
    _get_qwen_prompt_config_impl as _get_qwen_prompt_config_impl,
    _render_qwen_prompt_impl as _render_qwen_prompt_impl,
    _extract_qwen_json_block_impl as _extract_qwen_json_block_impl,
    _sanitize_prompts_impl as _sanitize_prompts_impl,
)

_extract_balanced_json = _extract_balanced_json_impl
_dir_size_bytes = _dir_size_bytes_impl
from services.detectors import (
    _agent_tool_run_detector_impl as _agent_tool_run_detector_impl,
    _ensure_yolo_inference_runtime_impl as _ensure_yolo_inference_runtime_impl,
    _ensure_rfdetr_inference_runtime_impl as _ensure_rfdetr_inference_runtime_impl,
    _set_yolo_infer_state_impl as _set_yolo_infer_state_impl,
    _set_rfdetr_infer_state_impl as _set_rfdetr_infer_state_impl,
    _load_yolo_active_impl as _load_yolo_active_impl,
    _save_yolo_active_impl as _save_yolo_active_impl,
    _load_rfdetr_active_impl as _load_rfdetr_active_impl,
    _save_rfdetr_active_impl as _save_rfdetr_active_impl,
    _load_detector_default_impl as _load_detector_default_impl,
    _save_detector_default_impl as _save_detector_default_impl,
    _yolo_run_dir_impl as _yolo_run_dir_impl,
    _yolo_load_run_meta_impl as _yolo_load_run_meta_impl,
    _yolo_write_run_meta_impl as _yolo_write_run_meta_impl,
    _yolo_prune_run_dir_impl as _yolo_prune_run_dir_impl,
    _yolo_device_arg_impl as _yolo_device_arg_impl,
    _yolo_p2_scale_impl as _yolo_p2_scale_impl,
    _rfdetr_variant_info_impl as _rfdetr_variant_info_impl,
    _rfdetr_best_checkpoint_impl as _rfdetr_best_checkpoint_impl,
    _rfdetr_parse_log_series_impl as _rfdetr_parse_log_series_impl,
    _rfdetr_sanitize_metric_impl as _rfdetr_sanitize_metric_impl,
    _rfdetr_normalize_aug_policy_impl as _rfdetr_normalize_aug_policy_impl,
    _rfdetr_install_augmentations_impl as _rfdetr_install_augmentations_impl,
    _rfdetr_restore_augmentations_impl as _rfdetr_restore_augmentations_impl,
    _rfdetr_latest_checkpoint_epoch_impl as _rfdetr_latest_checkpoint_epoch_impl,
    _rfdetr_monitor_training_impl as _rfdetr_monitor_training_impl,
    _yolo_build_aug_args_impl as _yolo_build_aug_args_impl,
    _yolo_parse_results_csv_impl as _yolo_parse_results_csv_impl,
    _yolo_monitor_training_impl as _yolo_monitor_training_impl,
    _strip_checkpoint_optimizer_impl as _strip_checkpoint_optimizer_impl,
    _yolo_load_labelmap_impl as _yolo_load_labelmap_impl,
    _yolo_load_run_labelmap_impl as _yolo_load_run_labelmap_impl,
    _rfdetr_load_labelmap_impl as _rfdetr_load_labelmap_impl,
    _rfdetr_remap_coco_ids_impl as _rfdetr_remap_coco_ids_impl,
    _rfdetr_prepare_dataset_impl as _rfdetr_prepare_dataset_impl,
    _yolo_write_data_yaml_impl as _yolo_write_data_yaml_impl,
    _yolo_resolve_model_source_impl as _yolo_resolve_model_source_impl,
    _yolo_variant_base_yaml_impl as _yolo_variant_base_yaml_impl,
    _yolo_write_variant_yaml_impl as _yolo_write_variant_yaml_impl,
    _yolo_write_head_graft_yaml_impl as _yolo_write_head_graft_yaml_impl,
    _yolo_find_detect_modules_impl as _yolo_find_detect_modules_impl,
    _yolo_detect_layer_index_impl as _yolo_detect_layer_index_impl,
    _rfdetr_ddp_worker_impl as _rfdetr_ddp_worker_impl,
    _rfdetr_run_dir_impl as _rfdetr_run_dir_impl,
    _rfdetr_load_run_meta_impl as _rfdetr_load_run_meta_impl,
    _rfdetr_write_run_meta_impl as _rfdetr_write_run_meta_impl,
    _rfdetr_prune_run_dir_impl as _rfdetr_prune_run_dir_impl,
    _collect_yolo_artifacts_impl as _collect_yolo_artifacts_impl,
    _collect_rfdetr_artifacts_impl as _collect_rfdetr_artifacts_impl,
    _yolo_extract_detections_impl as _yolo_extract_detections_impl,
    _rfdetr_extract_detections_impl as _rfdetr_extract_detections_impl,
    _resolve_detector_image_impl as _resolve_detector_image_impl,
    _yolo_metrics_summary_impl as _yolo_metrics_summary_impl,
    _rfdetr_metrics_summary_impl as _rfdetr_metrics_summary_impl,
    _clean_metric_summary_impl as _clean_metric_summary_impl,
    _list_yolo_runs_impl as _list_yolo_runs_impl,
    _list_rfdetr_runs_impl as _list_rfdetr_runs_impl,
)
from collections import OrderedDict
from segment_anything import sam_model_registry, SamPredictor


import threading
import queue
import itertools
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict

# Ensure we import the bundled SAM3 package (sam3/sam3) rather than shadowing it
# with the repo root folder name (sam3/). Without this, sam3 becomes a namespace
# that lacks the train.data modules needed for text prompting.
SAM3_SRC_ROOT = (Path(__file__).resolve().parent / "sam3").resolve()
if SAM3_SRC_ROOT.exists():
    sys.path.insert(0, str(SAM3_SRC_ROOT))

from tools.clip_training import (
    train_clip_from_yolo,
    TrainingError,
    TrainingArtifacts,
    _load_dinov3 as _clip_training_load_dinov3,
)
try:
    from tools.qwen_training import (
        QwenTrainingConfig,
        QwenTrainingResult,
        train_qwen_model,
        TrainingError as QwenTrainingError,
        DEFAULT_SYSTEM_PROMPT,
    )
except Exception as exc:  # noqa: BLE001
    QWEN_TRAINING_IMPORT_ERROR = exc
    QwenTrainingConfig = None  # type: ignore[assignment]
    QwenTrainingResult = None  # type: ignore[assignment]
    train_qwen_model = None  # type: ignore[assignment]
    QwenTrainingError = TrainingError  # type: ignore[assignment]
    DEFAULT_SYSTEM_PROMPT = (
        "You are an annotation assistant that only returns JSON objects shaped like {\"detections\":[{\"label\":\"class\","
        "\"bbox\":[x1,y1,x2,y2]} or {\"label\":\"class\",\"point\":[x,y]}]}"
    )
else:
    QWEN_TRAINING_IMPORT_ERROR = None

try:
    from transformers import (
        AutoProcessor,
        AutoConfig,
        AutoModelForCausalLM,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
    )
    from qwen_vl_utils import process_vision_info
except Exception as exc:  # noqa: BLE001
    QWEN_IMPORT_ERROR = exc
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]
    Qwen3VLMoeForConditionalGeneration = None  # type: ignore[assignment]
    AutoConfig = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]
    process_vision_info = None  # type: ignore[assignment]
else:
    QWEN_IMPORT_ERROR = None

def _try_import_qwen_agent() -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Exception]]:
    try:
        from qwen_agent.agents import FnCallAgent as _QwenFnCallAgent  # type: ignore
        from qwen_agent.llm.schema import Message as QwenAgentMessage, ContentItem as QwenAgentContentItem  # type: ignore
        from qwen_agent_llm import LocalQwenVLChatModel  # type: ignore
        from qwen_agent_tools import build_local_agent_tools  # type: ignore
        return _QwenFnCallAgent, QwenAgentMessage, QwenAgentContentItem, (LocalQwenVLChatModel, build_local_agent_tools), None
    except Exception as exc:  # noqa: BLE001
        return None, None, None, None, exc


QwenAgentAssistant, QwenAgentMessage, QwenAgentContentItem, _agent_payload, QWEN_AGENT_IMPORT_ERROR = _try_import_qwen_agent()
if QWEN_AGENT_IMPORT_ERROR is not None:
    local_repo = (Path(__file__).resolve().parent / "Qwen-Agent").resolve()
    if local_repo.exists():
        sys.path.insert(0, str(local_repo))
        QwenAgentAssistant, QwenAgentMessage, QwenAgentContentItem, _agent_payload, QWEN_AGENT_IMPORT_ERROR = _try_import_qwen_agent()
if _agent_payload is not None:
    LocalQwenVLChatModel, build_local_agent_tools = _agent_payload
else:
    LocalQwenVLChatModel = None  # type: ignore[assignment]
    build_local_agent_tools = None  # type: ignore[assignment]

BASE64_IMAGE_MAX_BYTES = int(os.environ.get("IMAGE_MAX_BYTES", str(100 * 1024 * 1024)))
BASE64_IMAGE_MAX_DIM = int(os.environ.get("IMAGE_MAX_DIM", "4096"))

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor as Sam3ImageProcessor
except Exception as exc:  # noqa: BLE001
    SAM3_NATIVE_IMAGE_IMPORT_ERROR = exc
    build_sam3_image_model = None  # type: ignore[assignment]
    Sam3ImageProcessor = None  # type: ignore[assignment]
else:
    SAM3_NATIVE_IMAGE_IMPORT_ERROR = None

try:
    from peft import PeftModel
except Exception as exc:  # noqa: BLE001
    PEFT_IMPORT_ERROR = exc
    PeftModel = None  # type: ignore[assignment]
else:
    PEFT_IMPORT_ERROR = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


MAX_PREDICTOR_SLOTS = 3
DATASET_ZIP_MAX_BYTES = _env_int("DATASET_ZIP_MAX_BYTES", 100 * 1024 * 1024 * 1024)
DATASET_ZIP_ENTRY_MAX_BYTES = _env_int("DATASET_ZIP_ENTRY_MAX_BYTES", 50 * 1024 * 1024 * 1024)
CLIP_DATASET_CHUNK_MAX_BYTES = _env_int("CLIP_DATASET_CHUNK_MAX_BYTES", 10 * 1024 * 1024 * 1024)
CLIP_DATASET_UPLOAD_QUOTA_BYTES = _env_int("CLIP_DATASET_UPLOAD_QUOTA_BYTES", 100 * 1024 * 1024 * 1024)
QWEN_DATASET_CHUNK_MAX_BYTES = _env_int("QWEN_DATASET_CHUNK_MAX_BYTES", 10 * 1024 * 1024 * 1024)
QWEN_DATASET_UPLOAD_QUOTA_BYTES = _env_int("QWEN_DATASET_UPLOAD_QUOTA_BYTES", 100 * 1024 * 1024 * 1024)
ASSET_MAX_BYTES = _env_int("ASSET_MAX_BYTES", 10 * 1024 * 1024 * 1024)
ASSET_UPLOAD_QUOTA_BYTES = _env_int("ASSET_UPLOAD_QUOTA_BYTES", 100 * 1024 * 1024 * 1024)
CLASSIFIER_ALLOWED_EXTS = {".pkl", ".joblib"}
LABELMAP_ALLOWED_EXTS = {".txt", ".pkl"}

SAM_PRELOAD_MAX_BYTES = _env_int("SAM_PRELOAD_MAX_BYTES", 2 * 1024 * 1024 * 1024)
MAX_RESPONSE_DETECTIONS = _env_int("MAX_RESPONSE_DETECTIONS", 5000)
MAX_RESPONSE_MASKS = _env_int("MAX_RESPONSE_MASKS", 2000)
MASK_ENCODE_MAX_BYTES = _env_int("MASK_ENCODE_MAX_BYTES", 64 * 1024 * 1024)
AGENT_MINING_CACHE_MAX_BYTES = _env_int("AGENT_MINING_CACHE_MAX_BYTES", 80 * 1024 * 1024 * 1024)
AGENT_MINING_CACHE_TTL_HOURS = _env_int("AGENT_MINING_CACHE_TTL_HOURS", 0)  # 0 = no TTL purge by default
AGENT_RECIPE_MAX_CROPS = _env_int("AGENT_RECIPE_MAX_CROPS", 1000)
AGENT_RECIPE_MAX_CROP_BYTES = _env_int("AGENT_RECIPE_MAX_CROP_BYTES", 512 * 1024 * 1024)
AGENT_RECIPE_MAX_CLIP_HEAD_BYTES = _env_int("AGENT_RECIPE_MAX_CLIP_HEAD_BYTES", 256 * 1024 * 1024)
AGENT_RECIPE_MAX_JSON_BYTES = _env_int("AGENT_RECIPE_MAX_JSON_BYTES", 10 * 1024 * 1024)
AGENT_RECIPE_MAX_BYTES = _env_int("AGENT_RECIPE_MAX_BYTES", 2 * 1024 * 1024 * 1024)
AGENT_CASCADE_MAX_JSON_BYTES = _env_int("AGENT_CASCADE_MAX_JSON_BYTES", 10 * 1024 * 1024)
AGENT_CASCADE_MAX_BYTES = _env_int("AGENT_CASCADE_MAX_BYTES", 8 * 1024 * 1024 * 1024)
CLIP_TRAIN_UPLOAD_MAX_BYTES = _env_int("CLIP_TRAIN_UPLOAD_MAX_BYTES", 10 * 1024 * 1024 * 1024)
CLIP_TRAIN_UPLOAD_QUOTA_BYTES = _env_int("CLIP_TRAIN_UPLOAD_QUOTA_BYTES", 100 * 1024 * 1024 * 1024)

QWEN_MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct")
QWEN_MIN_TRANSFORMERS = "4.57.0"
QWEN_MIN_PIXELS = _env_int("QWEN_MIN_PIXELS", 256 * 28 * 28)
QWEN_MAX_PIXELS = _env_int("QWEN_MAX_PIXELS", 1280 * 28 * 28)
QWEN_MAX_NEW_TOKENS = _env_int("QWEN_MAX_NEW_TOKENS", 2000)
QWEN_DO_SAMPLE = _env_bool("QWEN_DO_SAMPLE", False)
QWEN_TEMPERATURE = _env_float("QWEN_TEMPERATURE", 0.2)
QWEN_TOP_P = _env_float("QWEN_TOP_P", 0.9)
QWEN_DEVICE_PREF = os.environ.get("QWEN_DEVICE", "auto").strip().lower()
QWEN_TRUST_REMOTE_CODE = _env_bool("QWEN_TRUST_REMOTE_CODE", False)
QWEN_CAPTION_CACHE_LIMIT = _env_int("QWEN_CAPTION_CACHE_LIMIT", 0)
QWEN_WINDOW_DEFAULT_SIZE = _env_int("QWEN_WINDOW_SIZE", 672)
QWEN_WINDOW_DEFAULT_OVERLAP = _env_float("QWEN_WINDOW_OVERLAP", 0.2)

# Rough VRAM estimates (GB) for Qwen3 training defaults (batch=1, default pixel budget).
# These are approximate and should be treated as guidance, not a hard guarantee.
QWEN_VRAM_ESTIMATE_GB = {
    "official_lora": {
        "2B": 12.0,
        "4B": 20.0,
        "8B": 96.0,
        "32B": 192.0,
    },
    "trl_qlora": {
        "2B": 8.0,
        "4B": 10.0,
        "8B": 16.0,
        "32B": 48.0,
    },
}
QWEN_VRAM_THINKING_SCALE = 1.08
QWEN_VRAM_PIXEL_BASE = 451584
QWEN_VRAM_PIXEL_SCALE_MIN = 0.6
QWEN_VRAM_PIXEL_SCALE_MAX = 1.6

def _is_qwen_moe_model_id(model_id: str) -> bool:
    lowered = model_id.lower()
    return "a3b" in lowered or "moe" in lowered


qwen_model = None
qwen_processor = None
qwen_device: Optional[str] = None
qwen_last_error: Optional[str] = None
qwen_lock = threading.RLock()
qwen_config_lock = threading.RLock()
qwen_caption_cache: Dict[str, Tuple[Any, Any]] = {}
qwen_caption_order: deque[str] = deque()
_HF_OFFLINE_AUTO_ENABLED = False
_CAPTION_WINDOW_HOOK: ContextVar[Optional[Callable[[int, int, int, str], None]]] = ContextVar(
    "caption_window_hook",
    default=None,
)

QWEN_METADATA_FILENAME = "metadata.json"


def _default_qwen_metadata() -> Dict[str, Any]:
    return {
        "id": "default",
        "label": "Base Qwen 3",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "dataset_context": "",
        "classes": [],
        "model_id": QWEN_MODEL_NAME,
        "model_family": "qwen3",
        "source": "huggingface",
        "min_pixels": QWEN_MIN_PIXELS,
        "max_pixels": QWEN_MAX_PIXELS,
    }


def _enable_hf_offline_defaults() -> None:
    global _HF_OFFLINE_AUTO_ENABLED
    if _HF_OFFLINE_AUTO_ENABLED:
        return
    if not os.environ.get("HF_HUB_OFFLINE"):
        os.environ["HF_HUB_OFFLINE"] = "1"
    if not os.environ.get("TRANSFORMERS_OFFLINE"):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _HF_OFFLINE_AUTO_ENABLED = True
    logger.info("[qwen] HF offline mode enabled after initial download")


def _hf_offline_enabled() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _emit_caption_window(x0: int, y0: int, size: int, caption: str) -> None:
    hook = _CAPTION_WINDOW_HOOK.get()
    if hook is None:
        return
    try:
        hook(int(x0), int(y0), int(size), str(caption))
    except Exception:
        return


def _set_hf_offline(enabled: bool) -> None:
    if enabled:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"


active_qwen_model_id = "default"
active_qwen_model_path: Optional[Path] = None
active_qwen_metadata: Dict[str, Any] = _default_qwen_metadata()
loaded_qwen_model_id: Optional[str] = None


def _reset_qwen_runtime() -> None:
    global qwen_model, qwen_processor, qwen_last_error, loaded_qwen_model_id, qwen_device
    state = {
        "qwen_model": qwen_model,
        "qwen_processor": qwen_processor,
        "qwen_device": qwen_device,
        "loaded_qwen_model_id": loaded_qwen_model_id,
        "qwen_last_error": qwen_last_error,
    }
    _reset_qwen_runtime_impl(state=state, torch_module=torch)
    qwen_model = state["qwen_model"]
    qwen_processor = state["qwen_processor"]
    qwen_device = state["qwen_device"]
    loaded_qwen_model_id = state["loaded_qwen_model_id"]
    qwen_last_error = state["qwen_last_error"]


def _unload_sam3_text_runtime() -> None:
    """Release SAM3 text prompt model to free device memory."""
    global sam3_text_model, sam3_text_processor, sam3_text_device
    state = {
        "sam3_text_model": sam3_text_model,
        "sam3_text_processor": sam3_text_processor,
        "sam3_text_device": sam3_text_device,
    }
    _unload_sam3_text_runtime_impl(state=state, lock=sam3_text_lock)
    sam3_text_model = state["sam3_text_model"]
    sam3_text_processor = state["sam3_text_processor"]
    sam3_text_device = state["sam3_text_device"]


def _unload_dinov3_backbone() -> None:
    """Release DINOv3 encoder + per-device caches."""
    global dinov3_model, dinov3_processor, dinov3_model_name, dinov3_initialized, dinov3_model_device
    state = {
        "dinov3_model": dinov3_model,
        "dinov3_processor": dinov3_processor,
        "dinov3_model_name": dinov3_model_name,
        "dinov3_model_device": dinov3_model_device,
        "dinov3_initialized": dinov3_initialized,
    }
    _unload_dinov3_backbone_impl(
        state=state,
        lock=dinov3_lock,
        agent_backbones=_agent_dinov3_backbones,
        agent_locks=_agent_dinov3_locks,
    )
    dinov3_model = state["dinov3_model"]
    dinov3_processor = state["dinov3_processor"]
    dinov3_model_name = state["dinov3_model_name"]
    dinov3_model_device = state["dinov3_model_device"]
    dinov3_initialized = state["dinov3_initialized"]


def _load_dinov3_backbone(
    model_name: str,
    target_device: str,
    *,
    raise_on_error: bool = False,
) -> Tuple[Optional[Any], Optional[Any]]:
    try:
        model, processor = _clip_training_load_dinov3(model_name, target_device)
        return model, processor
    except Exception as exc:  # noqa: BLE001
        if raise_on_error:
            raise RuntimeError(str(exc)) from exc
        logger.warning("Failed to load DINOv3 backbone '%s' on %s: %s", model_name, target_device, exc)
        return None, None


def _unload_detector_inference() -> None:
    """Release detector inference models (YOLO/RF-DETR) to free GPU memory."""
    global yolo_infer_model, yolo_infer_path, yolo_infer_labelmap, yolo_infer_task
    global rfdetr_infer_model, rfdetr_infer_path, rfdetr_infer_labelmap, rfdetr_infer_task, rfdetr_infer_variant
    state = {
        "yolo_infer_model": yolo_infer_model,
        "yolo_infer_path": yolo_infer_path,
        "yolo_infer_labelmap": yolo_infer_labelmap,
        "yolo_infer_task": yolo_infer_task,
        "rfdetr_infer_model": rfdetr_infer_model,
        "rfdetr_infer_path": rfdetr_infer_path,
        "rfdetr_infer_labelmap": rfdetr_infer_labelmap,
        "rfdetr_infer_task": rfdetr_infer_task,
        "rfdetr_infer_variant": rfdetr_infer_variant,
    }
    _unload_detector_inference_impl(state=state)
    yolo_infer_model = state["yolo_infer_model"]
    yolo_infer_path = state["yolo_infer_path"]
    yolo_infer_labelmap = state["yolo_infer_labelmap"]
    yolo_infer_task = state["yolo_infer_task"]
    rfdetr_infer_model = state["rfdetr_infer_model"]
    rfdetr_infer_path = state["rfdetr_infer_path"]
    rfdetr_infer_labelmap = state["rfdetr_infer_labelmap"]
    rfdetr_infer_task = state["rfdetr_infer_task"]
    rfdetr_infer_variant = state["rfdetr_infer_variant"]


sam3_text_model = None
sam3_text_processor = None
sam3_text_device: Optional[torch.device] = None
sam3_text_lock = threading.RLock()


def _set_active_qwen_model_default() -> None:
    global active_qwen_model_id, active_qwen_model_path, active_qwen_metadata
    active_qwen_model_id = "default"
    active_qwen_model_path = None
    active_qwen_metadata = _default_qwen_metadata()
    _reset_qwen_runtime()


def _set_active_qwen_model_custom(model_id: str, ckpt_path: Path, metadata: Dict[str, Any]) -> None:
    global active_qwen_model_id, active_qwen_model_path, active_qwen_metadata
    active_qwen_model_id = model_id
    active_qwen_model_path = ckpt_path
    active_qwen_metadata = metadata or {}
    active_qwen_metadata.setdefault("id", model_id)
    _reset_qwen_runtime()


def _list_qwen_model_entries() -> List[Dict[str, Any]]:
    """Return registry entries for custom Qwen fine-tunes (if any)."""
    return []


def _get_qwen_model_entry(model_id: str) -> Optional[Dict[str, Any]]:
    for entry in _list_qwen_model_entries():
        if str(entry.get("id")) == str(model_id):
            return entry
    return None


def _prepare_for_qwen_training() -> None:
    _prepare_for_training_impl(
        unload_inference_runtimes_fn=lambda: _unload_inference_runtimes_impl(
            unload_non_qwen_fn=lambda: _unload_non_qwen_runtimes_impl(
                predictor_manager=predictor_manager,
                unload_sam3_text_fn=_unload_sam3_text_runtime,
                suspend_clip_fn=_suspend_clip_backbone,
                unload_dinov3_fn=_unload_dinov3_backbone,
                unload_detector_fn=_unload_detector_inference,
                torch_module=torch,
                logger=logger,
            ),
            unload_qwen_fn=_unload_qwen_runtime,
            torch_module=torch,
        )
    )


def _finalize_qwen_training_environment() -> None:
    _finalize_training_environment_impl(
        resume_classifier_fn=_resume_classifier_backbone,
        torch_module=torch,
    )


def _bytes_to_mb(value: int) -> float:
    return round(value / (1024 * 1024), 2)

# ----------------------------------------------------------------
# 1) Define a global error message and a global load-flag for CLIP
ERROR_MESSAGE = 0 # messy hack, making this an int because of the way we parse it later... the message has actually just been moved to the JS and appears when bbox uuid is None
clip_initialized = True
clip_last_error: Optional[str] = None
# ----------------------------------------------------------------

# 2) Attempt to load the logistic regression model (.pkl)
MODEL_PATH = "./my_logreg_model.pkl"
clf = None
if os.path.exists(MODEL_PATH):
    try:
        print("Loading logistic regression...")
        clf = joblib.load(MODEL_PATH)
        clip_last_error = None
    except Exception as e:
        print(f"Failed to load logistic regression model: {e}")
        clip_initialized = False
        clip_last_error = str(e)
else:
    print(f"File {MODEL_PATH} not found.")
    clip_initialized = False
    clip_last_error = "classifier_not_found"

LABELMAP_DEFAULT_PATH = "./my_label_list.pkl"
active_classifier_path: Optional[str] = MODEL_PATH if clf is not None else None
active_labelmap_path: Optional[str] = LABELMAP_DEFAULT_PATH if os.path.exists(LABELMAP_DEFAULT_PATH) else None
active_label_list: List[str] = []
active_encoder_type: str = "clip"
active_encoder_model: Optional[str] = None
active_classifier_meta: Dict[str, Any] = {}
active_head_normalize_embeddings: bool = True
active_classifier_head: Optional[Dict[str, Any]] = None
if active_labelmap_path:
    try:
        if active_labelmap_path.lower().endswith(".pkl"):
            loaded = joblib.load(active_labelmap_path)
            if isinstance(loaded, list):
                active_label_list = [str(item) for item in loaded]
            else:
                active_labelmap_path = None
        else:
            with open(active_labelmap_path, "r", encoding="utf-8") as handle:
                active_label_list = [line.strip() for line in handle if line.strip()]
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load labelmap {active_labelmap_path}: {exc}")
        active_labelmap_path = None
        active_label_list = []

try:
    if active_classifier_path and os.path.isfile(active_classifier_path):
        meta_path = os.path.splitext(active_classifier_path)[0] + ".meta.pkl"
        if os.path.exists(meta_path):
            meta_obj = joblib.load(meta_path)
            if isinstance(meta_obj, dict):
                active_classifier_meta = dict(meta_obj)
                active_encoder_type = meta_obj.get("encoder_type") or "clip"
                active_encoder_model = meta_obj.get("encoder_model") or meta_obj.get("clip_model")
                active_head_normalize_embeddings = _resolve_active_head_normalize_embeddings_impl(
                    meta_obj,
                    clf,
                    default=True,
                    resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
                )
except Exception:
    active_encoder_type = "clip"
    active_encoder_model = None
    active_classifier_head = None

# Keep default CLIP artifacts usable with path allowlists by mirroring them into uploads/.
# This preserves older workflows that write my_logreg_model.pkl/my_label_list.pkl at repo root.
try:
    _uploads_root_early = Path("uploads")
    _uploads_root_early.mkdir(exist_ok=True)
    _classifiers_root_early = (_uploads_root_early / "classifiers").resolve()
    _labelmaps_root_early = (_uploads_root_early / "labelmaps").resolve()
    _classifiers_root_early.mkdir(parents=True, exist_ok=True)
    _labelmaps_root_early.mkdir(parents=True, exist_ok=True)

    if active_classifier_path and os.path.isfile(active_classifier_path):
        src = Path(active_classifier_path).resolve()
        if not str(src).startswith(str(_classifiers_root_early)):
            dst = _classifiers_root_early / src.name
            try:
                if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
                active_classifier_path = str(dst)
            except Exception:
                pass

    if active_labelmap_path and os.path.isfile(active_labelmap_path):
        src = Path(active_labelmap_path).resolve()
        if not str(src).startswith(str(_labelmaps_root_early)):
            dst = _labelmaps_root_early / src.name
            try:
                if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
                active_labelmap_path = str(dst)
            except Exception:
                pass
except Exception:
    pass

# 3) Attempt to load the CLIP model (only when the active classifier uses CLIP)
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to enable TF32: %s", exc)
SUPPORTED_CLIP_MODELS = [
    "ViT-L/14",
    "ViT-B/16",
    "ViT-B/32",
]
DEFAULT_CLIP_MODEL = SUPPORTED_CLIP_MODELS[0]

clip_model = None
clip_preprocess = None
clip_model_name: Optional[str] = None
_clip_reload_needed = False
_skip_clip_load = os.environ.get("TATOR_SKIP_CLIP_LOAD", "").strip() == "1"
if _skip_clip_load:
    print("Skipping CLIP model load (TATOR_SKIP_CLIP_LOAD=1).")
    clip_initialized = False
    clip_model_name = None
else:
    encoder_type_norm = str(active_encoder_type or "clip").strip().lower()
    if encoder_type_norm != "clip" or not active_classifier_path:
        print(f"Skipping CLIP model load (active encoder={active_encoder_type}).")
        clip_initialized = False
        clip_model_name = None
    else:
        try:
            clip_name = active_encoder_model or DEFAULT_CLIP_MODEL
            print(f"Loading CLIP model ({clip_name})...")
            clip_model, clip_preprocess = clip.load(clip_name, device=device)
            clip_model_name = clip_name
        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
            clip_initialized = False
            clip_model_name = None

clip_lock = threading.Lock()
if clip_model is None or clf is None:
    clip_initialized = False

# Agent Mining often evaluates across multiple GPUs. The global CLIP backbone is pinned to
# `device` (typically "cuda" == GPU0) and guarded by `clip_lock`. To avoid serializing all CLIP
# embedding work onto a single GPU during mining/eval, we maintain per-device CLIP backbones
# (raw CLIP, not trained heads) with per-device locks.
_agent_clip_backbones: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
_agent_clip_locks: Dict[Tuple[str, str], threading.Lock] = {}
_agent_clip_backbones_lock = threading.Lock()

# Optional DINOv3 image encoder (frozen) for classifier heads.
dinov3_model: Optional[Any] = None
dinov3_processor: Optional[Any] = None
dinov3_model_name: Optional[str] = None
dinov3_model_device: Optional[str] = None
dinov3_initialized = False
dinov3_cuda_disabled = False
dinov3_lock = threading.Lock()
_agent_dinov3_backbones: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
_agent_dinov3_locks: Dict[Tuple[str, str], threading.Lock] = {}
_agent_dinov3_backbones_lock = threading.Lock()


def _suspend_clip_backbone() -> None:
    global clip_model, clip_preprocess, clip_initialized, _clip_reload_needed
    state = {
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "clip_initialized": clip_initialized,
        "_clip_reload_needed": _clip_reload_needed,
        "active_encoder_type": active_encoder_type,
    }
    _suspend_clip_backbone_impl(
        state=state,
        lock=clip_lock,
        agent_backbones=_agent_clip_backbones,
        agent_locks=_agent_clip_locks,
        logger=logger,
    )
    clip_model = state["clip_model"]
    clip_preprocess = state["clip_preprocess"]
    clip_initialized = state["clip_initialized"]
    _clip_reload_needed = state["_clip_reload_needed"]


def _resume_clip_backbone() -> None:
    global clip_model, clip_preprocess, clip_initialized, _clip_reload_needed
    state = {
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "clip_initialized": clip_initialized,
        "_clip_reload_needed": _clip_reload_needed,
        "clip_model_name": clip_model_name,
    }
    _resume_clip_backbone_impl(
        state=state,
        lock=clip_lock,
        clip_module=clip,
        device=device,
        default_model=DEFAULT_CLIP_MODEL,
        clf=clf,
        logger=logger,
    )
    clip_model = state["clip_model"]
    clip_preprocess = state["clip_preprocess"]
    clip_initialized = state["clip_initialized"]
    _clip_reload_needed = state["_clip_reload_needed"]


def _resume_classifier_backbone() -> None:
    """Reload the active encoder backbone after training, based on user-selected classifier."""
    global dinov3_model, dinov3_processor, dinov3_model_name, dinov3_initialized, dinov3_model_device
    global clip_model_name, _clip_reload_needed
    state = {
        "dinov3_model": dinov3_model,
        "dinov3_processor": dinov3_processor,
        "dinov3_model_name": dinov3_model_name,
        "dinov3_model_device": dinov3_model_device,
        "dinov3_initialized": dinov3_initialized,
        "clip_model_name": clip_model_name,
        "_clip_reload_needed": _clip_reload_needed,
        "active_encoder_type": active_encoder_type,
        "active_encoder_model": active_encoder_model,
    }
    _resume_classifier_backbone_impl(
        state=state,
        device=device,
        dinov3_lock=dinov3_lock,
        dinov3_cuda_disabled=dinov3_cuda_disabled,
        dinov3_resolve_device_fn=lambda device: _dinov3_resolve_device_impl(
            device, cuda_disabled=dinov3_cuda_disabled
        ),
        load_dinov3_fn=_load_dinov3_backbone,
        resume_clip_fn=_resume_clip_backbone,
    )
    dinov3_model = state["dinov3_model"]
    dinov3_processor = state["dinov3_processor"]
    dinov3_model_name = state["dinov3_model_name"]
    dinov3_model_device = state["dinov3_model_device"]
    dinov3_initialized = state["dinov3_initialized"]
    clip_model_name = state["clip_model_name"]
    _clip_reload_needed = state["_clip_reload_needed"]


# 4) Load the SAM model (segment-anything) as normal:
MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")
CHECKPOINT_PATH = os.environ.get("SAM_CHECKPOINT_PATH", "./sam_vit_h_4b8939.pth")
SAM3_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
SAM3_PROCESSOR_ID = os.environ.get("SAM3_PROCESSOR_ID", SAM3_MODEL_ID)
SAM3_CHECKPOINT_PATH = os.environ.get("SAM3_CHECKPOINT_PATH")
SAM3_DEVICE_PREF = os.environ.get("SAM3_DEVICE", "auto").strip().lower()
active_sam3_model_id = "default"
active_sam3_checkpoint = SAM3_CHECKPOINT_PATH
active_sam3_enable_segmentation = True
active_sam3_metadata: Dict[str, Any] = {
    "id": "default",
    "label": "Base SAM3",
    "checkpoint": SAM3_CHECKPOINT_PATH,
    "source": "env",
    "enable_segmentation": True,
}


def _set_sam3_device_pref(device_index: int) -> None:
    global SAM3_DEVICE_PREF
    state = {"device_pref": SAM3_DEVICE_PREF}
    _set_sam3_device_pref_impl(device_index, torch_module=torch, state=state)
    SAM3_DEVICE_PREF = state["device_pref"]


def _reset_sam3_runtime() -> None:
    global sam3_text_model, sam3_text_processor, sam3_text_device
    state = {
        "sam3_text_model": sam3_text_model,
        "sam3_text_processor": sam3_text_processor,
        "sam3_text_device": sam3_text_device,
    }
    _reset_sam3_runtime_impl(
        state=state,
        predictor_manager=predictor_manager,
        torch_module=torch,
    )
    sam3_text_model = state["sam3_text_model"]
    sam3_text_processor = state["sam3_text_processor"]
    sam3_text_device = state["sam3_text_device"]


def _resolve_sam1_devices() -> List[torch.device]:
    devices: List[torch.device] = []
    if torch.cuda.is_available():
        try:
            for idx in range(torch.cuda.device_count()):
                devices.append(torch.device(f"cuda:{idx}"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to enumerate CUDA devices for SAM1: %s", exc)
            devices = []
    if not devices:
        devices = [torch.device("cpu")]
    return devices


class _Sam1Backend:
    def __init__(self):
        self.predictor = SamPredictor(sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH))

    def set_image(self, np_img: np.ndarray) -> None:
        self.predictor.set_image(np_img)

    def predict(self, **kwargs):
        return self.predictor.predict(**kwargs)

    def unload(self) -> None:
        try:
            del self.predictor
        except Exception:  # noqa: BLE001
            pass
        self.predictor = None


class _Sam3Backend:
    def __init__(self):
        if SAM3_NATIVE_IMAGE_IMPORT_ERROR is not None or build_sam3_image_model is None:
            raise RuntimeError(f"sam3_unavailable:{SAM3_NATIVE_IMAGE_IMPORT_ERROR}")
        self.device = _resolve_sam3_device_impl(
            SAM3_DEVICE_PREF,
            torch_module=torch,
            http_exception_cls=HTTPException,
            http_400=HTTP_400_BAD_REQUEST,
        )
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        source = active_sam3_metadata.get("source") if isinstance(active_sam3_metadata, dict) else None
        try:
            model = build_sam3_image_model(
                device=device_str,
                checkpoint_path=active_sam3_checkpoint,
                load_from_HF=active_sam3_checkpoint is None,
                enable_inst_interactivity=True,
                enable_segmentation=active_sam3_enable_segmentation,
                bpe_path=str(SAM3_BPE_PATH),
            )
            if self.device:
                model = model.to(self.device)
            predictor = getattr(model, "inst_interactive_predictor", None)
            if predictor is None:
                raise RuntimeError("sam3_interactive_predictor_missing")
            tracker = getattr(predictor, "model", None)
            if tracker is None:
                raise RuntimeError("sam3_tracker_missing")
            if getattr(tracker, "backbone", None) is None:
                tracker.backbone = model.backbone
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"sam3_load_failed:{exc}") from exc
        self.model = model
        self.predictor = predictor

    def set_image(self, np_img: np.ndarray) -> None:
        arr = np.ascontiguousarray(np_img)
        self.predictor.set_image(arr)

    def predict(self, **kwargs):
        point_coords = kwargs.get("point_coords")
        point_labels = kwargs.get("point_labels")
        box = kwargs.get("box")
        mask_input = kwargs.get("mask_input")
        multimask_output = kwargs.get("multimask_output", True)
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )

    def unload(self) -> None:
        try:
            del self.predictor
        except Exception:  # noqa: BLE001
            pass
        try:
            del self.model
        except Exception:  # noqa: BLE001
            pass
        self.predictor = None
        self.model = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def _ensure_sam3_text_runtime():
    global sam3_text_model, sam3_text_processor, sam3_text_device
    state = {
        "sam3_text_model": sam3_text_model,
        "sam3_text_processor": sam3_text_processor,
        "sam3_text_device": sam3_text_device,
    }
    model, processor, device = _ensure_sam3_text_runtime_impl(
        state=state,
        lock=sam3_text_lock,
        resolve_device_fn=lambda: _resolve_sam3_device_impl(
            SAM3_DEVICE_PREF,
            torch_module=torch,
            http_exception_cls=HTTPException,
            http_400=HTTP_400_BAD_REQUEST,
        ),
        sam3_import_error=SAM3_NATIVE_IMAGE_IMPORT_ERROR,
        build_model_fn=build_sam3_image_model,
        processor_cls=Sam3ImageProcessor,
        sam3_checkpoint=active_sam3_checkpoint,
        sam3_bpe_path=SAM3_BPE_PATH,
        clear_caches_fn=_sam3_clear_device_pinned_caches_impl,
        http_exception_cls=HTTPException,
        http_503=HTTP_503_SERVICE_UNAVAILABLE,
        http_500=HTTP_500_INTERNAL_SERVER_ERROR,
    )
    sam3_text_model = state["sam3_text_model"]
    sam3_text_processor = state["sam3_text_processor"]
    sam3_text_device = state["sam3_text_device"]
    return model, processor, device


class PredictorSlot:
    def __init__(self, name: str):
        self.name = name
        self.backends: Dict[str, Any] = {}
        self.token: Optional[str] = None
        self.variant: Optional[str] = None
        self.image_shape: Optional[Tuple[int, int, int]] = None
        self.image_name: Optional[str] = None
        self.last_loaded: float = 0.0
        self.lock = threading.RLock()
        self._busy = threading.Event()
        self.image_memory_bytes: int = 0

    def set_image(self, np_img: np.ndarray, token: Optional[str], variant: Optional[str], image_name: Optional[str]) -> None:
        variant_name = (variant or "sam1").lower()
        with self.lock:
            self._busy.set()
            try:
                backend = self._ensure_backend(variant_name)
                backend.set_image(np_img)
                self.token = token
                self.variant = variant_name
                self.image_shape = np_img.shape
                self.image_name = image_name
                self.last_loaded = time.time()
                self.image_memory_bytes = int(np_img.nbytes)
            finally:
                self._busy.clear()

    def predict(self, **kwargs):
        with self.lock:
            self._busy.set()
            try:
                backend = self._ensure_backend((self.variant or "sam1").lower())
                return backend.predict(**kwargs)
            finally:
                self._busy.clear()

    def is_busy(self) -> bool:
        return self._busy.is_set()

    def clear(self) -> None:
        with self.lock:
            self.token = None
            self.variant = None
            self.image_shape = None
            self.image_name = None
            self.last_loaded = 0.0
            self.image_memory_bytes = 0

    def unload(self) -> None:
        with self.lock:
            self.clear()
            for backend in self.backends.values():
                try:
                    backend.unload()
                except Exception:  # noqa: BLE001
                    pass
            self.backends.clear()

    def _ensure_backend(self, variant: str):
        backend = self.backends.get(variant)
        if backend is None:
            backend = _build_backend_for_variant_impl(
                variant,
                sam3_backend_cls=_Sam3Backend,
                sam1_backend_cls=_Sam1Backend,
            )
            self.backends[variant] = backend
        return backend


class PredictorManager:
    def __init__(self):
        self.slots: Dict[str, PredictorSlot] = {
            "current": PredictorSlot("current"),
            "next": PredictorSlot("next"),
            "previous": PredictorSlot("previous"),
        }
        self.slot_order: List[str] = ["current", "next", "previous"]
        self.capacity_lock = threading.RLock()
        self.capacity: int = min(MAX_PREDICTOR_SLOTS, len(self.slot_order))
        self.enabled_slots: set[str] = set(self.slot_order[: self.capacity])
        self.token_index: Dict[Tuple[str, str], PredictorSlot] = {}
        self.image_index: Dict[Tuple[str, str], PredictorSlot] = {}
        self.queue: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, name="predictor-preload-worker", daemon=True)
        self.worker.start()

    def _slot_key(self, token: Optional[str], variant: Optional[str]) -> Optional[Tuple[str, str]]:
        if not token or not variant:
            return None
        return (token, variant)

    def _image_key(self, image_name: Optional[str], variant: Optional[str]) -> Optional[Tuple[str, str]]:
        if not image_name or not variant:
            return None
        return (variant, image_name)

    def is_slot_enabled(self, slot_name: str) -> bool:
        return slot_name in self.enabled_slots

    def resolve_slot(self, slot_name: Optional[str], *, allow_disabled_fallback: bool = True) -> str:
        """Return a normalised slot name.

        When ``allow_disabled_fallback`` is False we fail fast if the requested
        slot is currently disabled instead of silently falling back to the
        "current" slot. This prevents background preloads from clobbering the
        user's active predictor when the capacity shrinks.
        """

        candidate = (slot_name or "current").lower()
        if candidate not in self.slots:
            return "current"
        if self.is_slot_enabled(candidate):
            return candidate
        if allow_disabled_fallback:
            return "current"
        raise ValueError(f"slot_disabled:{candidate}")

    def capacity_limits(self) -> Tuple[int, int]:
        return (1, min(MAX_PREDICTOR_SLOTS, len(self.slot_order)))

    def get_capacity(self) -> int:
        with self.capacity_lock:
            return self.capacity

    def set_capacity(self, capacity: int) -> None:
        minimum, maximum = self.capacity_limits()
        normalized = max(minimum, min(maximum, capacity))
        with self.capacity_lock:
            if normalized == self.capacity:
                return
            self.capacity = normalized
            new_enabled = set(self.slot_order[: normalized])
            disabled = self.enabled_slots - new_enabled
            self.enabled_slots = new_enabled
            for slot_name in disabled:
                slot = self.slots.get(slot_name)
                if slot:
                    self._clear_slot_refs(slot)
                    slot.clear()

    def active_slot_count(self) -> int:
        return len(self.enabled_slots)

    def loaded_slot_count(self) -> int:
        return sum(1 for name, slot in self.slots.items() if name in self.enabled_slots and slot.token)

    def total_image_memory_bytes(self) -> int:
        return sum(slot.image_memory_bytes for name, slot in self.slots.items() if name in self.enabled_slots)

    def _clear_slot_refs(self, slot: PredictorSlot) -> None:
        remove_keys = [key for key, value in self.token_index.items() if value is slot]
        for key in remove_keys:
            self.token_index.pop(key, None)
        remove_image_keys = [key for key, value in self.image_index.items() if value is slot]
        for key in remove_image_keys:
            self.image_index.pop(key, None)

    def unload_all(self) -> None:
        with self.capacity_lock:
            for slot in self.slots.values():
                self._clear_slot_refs(slot)
                slot.unload()

    def set_slot(self, slot_name: str, np_img: np.ndarray, token: Optional[str], variant: Optional[str], image_name: Optional[str]) -> None:
        slot_name = self.resolve_slot(slot_name, allow_disabled_fallback=False)
        slot = self.slots[slot_name]
        self._clear_slot_refs(slot)
        slot.set_image(np_img, token, variant, image_name)
        key = self._slot_key(token, variant)
        if key:
            self.token_index[key] = slot
        image_key = self._image_key(image_name, variant)
        if image_key:
            self.image_index[image_key] = slot

    def ensure_current(self, np_img: np.ndarray, token: Optional[str], variant: Optional[str], image_name: Optional[str]) -> PredictorSlot:
        slot = self.token_index.get(self._slot_key(token, variant)) if token and variant else None
        if slot and slot.name == "current":
            return slot
        self.set_slot("current", np_img, token, variant, image_name)
        return self.slots["current"]

    def get_slot_for_token(self, token: Optional[str], variant: Optional[str]) -> Optional[PredictorSlot]:
        key = self._slot_key(token, variant)
        if key is None:
            return None
        return self.token_index.get(key)

    def get_slot_for_image(self, image_name: Optional[str], variant: Optional[str]) -> Optional[PredictorSlot]:
        key = self._image_key(image_name, variant)
        if key is None:
            return None
        return self.image_index.get(key)

    def promote_slot(self, slot_name: str) -> bool:
        if slot_name not in self.slots or slot_name == "current" or not self.is_slot_enabled(slot_name):
            return False
        if slot_name == "next":
            prev_slot = self.slots["previous"]
            curr_slot = self.slots["current"]
            next_slot = self.slots["next"]
            self.slots["previous"] = curr_slot
            self.slots["current"] = next_slot
            self.slots["next"] = prev_slot
        elif slot_name == "previous":
            prev_slot = self.slots["previous"]
            curr_slot = self.slots["current"]
            next_slot = self.slots["next"]
            self.slots["next"] = curr_slot
            self.slots["current"] = prev_slot
            self.slots["previous"] = next_slot
        else:
            return False
        self.slots["previous"].name = "previous"
        self.slots["current"].name = "current"
        self.slots["next"].name = "next"
        return True

    def predict(self, np_img: np.ndarray, token: Optional[str], variant: Optional[str], image_name: Optional[str], **predict_kwargs):
        slot = self.get_slot_for_token(token, variant)
        if slot is None:
            slot = self.ensure_current(np_img, token, variant, image_name)
        return slot.predict(**predict_kwargs)

    def set_slot_with_wait(self, slot_name: str, np_img: np.ndarray, token: Optional[str], variant: Optional[str], image_name: Optional[str]) -> None:
        slot_name = self.resolve_slot(slot_name, allow_disabled_fallback=False)
        if slot_name != "current":
            waited = 0.0
            # Give the "current" slot a brief head start so the active image always begins loading first,
            # but do not block background slots for the full duration of set_image.
            while (
                not self.stop_event.is_set()
                and waited < 0.2
                and not self.slots["current"].is_busy()
                and not self.slots["current"].token
            ):
                time.sleep(0.01)
                waited += 0.01
        self.set_slot(slot_name, np_img, token, variant, image_name)

    def stop(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=1.0)

    def schedule_slot(self, slot_name: str, payload: Dict[str, Any]) -> None:
        self.queue.put((slot_name, payload))

    def status(self) -> List[Dict[str, Any]]:
        info = []
        for name, slot in self.slots.items():
            entry: Dict[str, Any] = {
                "slot": name,
                "token": slot.token,
                "variant": slot.variant,
                "image_name": slot.image_name,
                "width": None,
                "height": None,
                "last_loaded": slot.last_loaded,
                "busy": slot.is_busy(),
                "enabled": self.is_slot_enabled(name),
                "memory_bytes": slot.image_memory_bytes,
            }
            if slot.image_shape:
                entry["height"] = slot.image_shape[0]
                entry["width"] = slot.image_shape[1]
            info.append(entry)
        return info

    def _materialize(self, payload: Dict[str, Any]) -> Tuple[np.ndarray, str, str, Optional[str]]:
        variant = _default_variant(payload.get("sam_variant"))
        image_name = payload.get("image_name")
        token = payload.get("image_token")
        if token:
            cached = _fetch_preloaded_image(token, variant)
            if cached is not None:
                return cached, token, variant, image_name
        base64_data = payload.get("image_base64")
        if not base64_data:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="image_payload_missing")
        _, np_img = _decode_image_base64_impl(base64_data, max_bytes=BASE64_IMAGE_MAX_BYTES, max_dim=BASE64_IMAGE_MAX_DIM, allow_downscale=True)
        token = hashlib.md5(np_img.tobytes()).hexdigest()
        _store_preloaded_image(token, np_img, variant)
        return np_img, token, variant, image_name

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                slot_name, payload = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                np_img, token, variant, image_name = self._materialize(payload)
                try:
                    self.set_slot_with_wait(slot_name, np_img, token, variant, image_name)
                except ValueError:
                    # Slot was disabled while this job was in flight; skip.
                    continue
            except Exception as exc:  # noqa: BLE001
                print(f"predictor preload failed: {exc}")


predictor_manager = PredictorManager()

# 5) Threading lock for SAM usage:
sam_lock = threading.Lock()

job_store: Dict[str, List["CropImage"]] = {}

app = FastAPI(title="Local Inference API (Multi-Predictor)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("localinferenceapi")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.propagate = False

# Cache for repeated calls
SAM_CACHE_LIMIT = 8
sam_cache_lock = threading.Lock()
sam_preload_cache: "OrderedDict[str, Tuple[np.ndarray, str]]" = OrderedDict()
sam_preload_cache_bytes = 0


def _store_preloaded_image(token: str, np_img: np.ndarray, variant: str) -> None:
    global sam_preload_cache_bytes
    arr = np.ascontiguousarray(np_img)
    arr_bytes = arr.nbytes
    if SAM_PRELOAD_MAX_BYTES > 0 and arr_bytes > SAM_PRELOAD_MAX_BYTES:
        logger.warning("Skipping preload store: image too large (%d bytes > %d)", arr_bytes, SAM_PRELOAD_MAX_BYTES)
        return
    with sam_cache_lock:
        # Remove existing entry bytes
        if token in sam_preload_cache:
            old_arr, _ = sam_preload_cache[token]
            sam_preload_cache_bytes -= getattr(old_arr, "nbytes", 0)
        sam_preload_cache[token] = (arr, variant)
        sam_preload_cache.move_to_end(token)
        sam_preload_cache_bytes += arr_bytes
        while len(sam_preload_cache) > SAM_CACHE_LIMIT or (
            SAM_PRELOAD_MAX_BYTES > 0 and sam_preload_cache_bytes > SAM_PRELOAD_MAX_BYTES
        ):
            _, (evicted_arr, _) = sam_preload_cache.popitem(last=False)
            sam_preload_cache_bytes -= getattr(evicted_arr, "nbytes", 0)


def _fetch_preloaded_image(token: str, variant: str) -> Optional[np.ndarray]:
    with sam_cache_lock:
        item = sam_preload_cache.get(token)
        if not item:
            return None
        arr, stored_variant = item
        # Allow reuse across variants; cache holds raw image arrays only.
        sam_preload_cache.move_to_end(token)
        return arr


_job_id_counter = itertools.count(1)


@dataclass
class SamPreloadJob:
    request_id: int
    variant: str
    slot: str
    generation: Optional[int]
    image_token: Optional[str]
    image_base64: Optional[str]
    image_name: Optional[str]
    event: threading.Event
    result: Optional[SamPreloadResponse] = None
    error: Optional[Exception] = None


class SamPreloadManager:
    def __init__(self):
        self.queue: "queue.Queue[SamPreloadJob]" = queue.Queue()
        self.lock = threading.Lock()
        self.latest_request_id: Dict[Tuple[str, str], int] = {}
        self.latest_generation: Dict[Tuple[str, str], int] = {}
        self.worker = threading.Thread(target=self._worker, name="sam-preload-worker", daemon=True)
        self.worker.start()

    def submit(
        self,
        *,
        variant: str,
        slot: str,
        generation: Optional[int],
        image_token: Optional[str],
        image_base64: Optional[str],
        image_name: Optional[str],
    ) -> SamPreloadResponse:
        job = SamPreloadJob(
            request_id=next(_job_id_counter),
            variant=variant,
            slot=slot,
            generation=generation,
            image_token=image_token,
            image_base64=image_base64,
            image_name=image_name,
            event=threading.Event(),
        )
        key = (variant, slot)
        with self.lock:
            self.latest_request_id[key] = job.request_id
            if generation is not None:
                prev = self.latest_generation.get(key)
                if prev is None or generation > prev:
                    self.latest_generation[key] = generation
        self.queue.put(job)
        job.event.wait()
        if job.error:
            raise job.error
        return job.result  # type: ignore[return-value]

    def _worker(self) -> None:
        while True:
            job = self.queue.get()
            try:
                if self._is_superseded(job):
                    job.result = self._superseded_response(job)
                else:
                    job.result = self._process_job(job)
            except Exception as exc:  # noqa: BLE001
                job.error = exc
            finally:
                job.event.set()
                self.queue.task_done()

    def _key(self, job: SamPreloadJob) -> Tuple[str, str]:
        return (job.variant, job.slot)

    def _is_superseded(self, job: SamPreloadJob) -> bool:
        with self.lock:
            latest_id = self.latest_request_id.get(self._key(job))
            latest_generation = self.latest_generation.get(self._key(job))
        if latest_id is not None and job.request_id < latest_id:
            return True
        if job.generation is not None and latest_generation is not None and job.generation < latest_generation:
            return True
        return False

    def _superseded_response(self, job: SamPreloadJob) -> SamPreloadResponse:
        width = 0
        height = 0
        if job.image_token:
            cached = _fetch_preloaded_image(job.image_token, job.variant)
            if cached is not None:
                height, width = cached.shape[:2]
        return SamPreloadResponse(status="superseded", width=int(width), height=int(height), token=job.image_token or "")

    def _process_job(self, job: SamPreloadJob) -> SamPreloadResponse:
        variant = job.variant
        slot = job.slot
        image_name = job.image_name

        if job.image_token:
            cached = _fetch_preloaded_image(job.image_token, variant)
            if cached is not None:
                if self._is_superseded(job):
                    height, width = cached.shape[:2]
                    return SamPreloadResponse(
                        status="superseded",
                        width=int(width),
                        height=int(height),
                        token=job.image_token,
                    )
                predictor_manager.set_slot_with_wait(slot, cached, job.image_token, variant, image_name)
                height, width = cached.shape[:2]
                return SamPreloadResponse(status="ready", width=int(width), height=int(height), token=job.image_token)
            if not job.image_base64:
                raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="image_token_not_found")

        if not job.image_base64:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="image_base64_required")

        np_img = self._decode_base64(job.image_base64)
        token = hashlib.md5(np_img.tobytes()).hexdigest()
        _store_preloaded_image(token, np_img, variant)

        if self._is_superseded(job):
            height, width = np_img.shape[:2]
            return SamPreloadResponse(status="superseded", width=int(width), height=int(height), token=token)

        predictor_manager.set_slot_with_wait(slot, np_img, token, variant, image_name)
        height, width = np_img.shape[:2]
        return SamPreloadResponse(status="ready", width=int(width), height=int(height), token=token)

    @staticmethod
    def _decode_base64(image_base64: str) -> np.ndarray:
        _, np_img = _decode_image_base64_impl(image_base64, max_bytes=BASE64_IMAGE_MAX_BYTES, max_dim=BASE64_IMAGE_MAX_DIM, allow_downscale=True)
        return np_img


sam_preload_manager = SamPreloadManager()


def _predict_with_cache(
    np_img: np.ndarray,
    token: Optional[str],
    variant: str,
    *,
    image_name: Optional[str] = None,
    **predict_kwargs: Any,
):
    normalized = _default_variant(variant)
    if normalized == "sam3" and not active_sam3_enable_segmentation:
        height, width = np_img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        box = predict_kwargs.get("box")
        point_coords = predict_kwargs.get("point_coords")
        if box is not None and len(box) >= 4:
            try:
                x1 = int(round(float(box[0])))
                y1 = int(round(float(box[1])))
                x2 = int(round(float(box[2])))
                y2 = int(round(float(box[3])))
            except (TypeError, ValueError):
                x1 = y1 = x2 = y2 = 0
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1
        elif point_coords is not None:
            try:
                px = int(round(float(point_coords[0][0])))
                py = int(round(float(point_coords[0][1])))
            except Exception:
                px = py = 0
            px = max(0, min(px, width - 1))
            py = max(0, min(py, height - 1))
            size = 2
            x1 = max(0, px - size)
            x2 = min(width, px + size)
            y1 = max(0, py - size)
            y2 = min(height, py + size)
            mask[y1:y2, x1:x2] = 1
        masks = np.asarray([mask], dtype=np.uint8)
        return masks, None, None
    return predictor_manager.predict(np_img, token, variant, image_name=image_name, **predict_kwargs)


def _default_variant(value: Optional[str]) -> str:
    return (value or "sam1").lower()


_job_id_counter = itertools.count(1)


@dataclass
class SamPreloadJob:
    request_id: int
    variant: str
    generation: Optional[int]
    image_token: Optional[str]
    image_base64: Optional[str]
    image_name: Optional[str]
    slot: str
    event: threading.Event
    result: Optional['SamPreloadResponse'] = None
    error: Optional[Exception] = None


class SamPreloadManager:
    def __init__(self):
        self.queue: "queue.Queue[SamPreloadJob]" = queue.Queue()
        self.lock = threading.Lock()
        self.latest_request_id: Dict[str, int] = {}
        self.latest_generation: Dict[str, int] = {}
        self.worker = threading.Thread(target=self._worker, name="sam-preload-worker", daemon=True)
        self.worker.start()

    def submit(
        self,
        *,
        variant: str,
        generation: Optional[int],
        image_token: Optional[str],
        image_base64: Optional[str],
        image_name: Optional[str],
        slot: str,
    ) -> 'SamPreloadResponse':
        job = SamPreloadJob(
            request_id=next(_job_id_counter),
            variant=variant,
            generation=generation,
            image_token=image_token,
            image_base64=image_base64,
            image_name=image_name,
            slot=slot,
            event=threading.Event(),
        )
        with self.lock:
            self.latest_request_id[variant] = job.request_id
            if generation is not None:
                prev = self.latest_generation.get(variant)
                if prev is None or generation > prev:
                    self.latest_generation[variant] = generation
        self.queue.put(job)
        job.event.wait()
        if job.error:
            raise job.error
        return job.result  # type: ignore[return-value]

    def _worker(self) -> None:
        while True:
            job = self.queue.get()
            try:
                if self._is_superseded(job):
                    job.result = SamPreloadResponse(status="superseded", width=0, height=0, token=job.image_token or "")
                else:
                    job.result = self._process_job(job)
            except Exception as exc:  # noqa: BLE001 - propagate to caller
                job.error = exc
            finally:
                job.event.set()
                self.queue.task_done()

    def _is_superseded(self, job: SamPreloadJob) -> bool:
        with self.lock:
            latest_id = self.latest_request_id.get(job.variant)
            latest_generation = self.latest_generation.get(job.variant)
        if latest_id is not None and job.request_id < latest_id:
            return True
        if job.generation is not None and latest_generation is not None and job.generation < latest_generation:
            return True
        return False

    def _process_job(self, job: SamPreloadJob) -> 'SamPreloadResponse':
        variant = job.variant
        try:
            slot_name = predictor_manager.resolve_slot(job.slot, allow_disabled_fallback=False)
        except ValueError:
            return SamPreloadResponse(status="slot_disabled", width=0, height=0, token=job.image_token or "")
        image_name = job.image_name

        if job.image_token:
            cached = _fetch_preloaded_image(job.image_token, variant)
            if cached is not None:
                if self._is_superseded(job):
                    return SamPreloadResponse(
                        status="superseded",
                        width=int(cached.shape[1]),
                        height=int(cached.shape[0]),
                        token=job.image_token,
                    )
                predictor_manager.set_slot_with_wait(slot_name, cached, job.image_token, variant, image_name)
                height, width = cached.shape[:2]
                return SamPreloadResponse(status="ready", width=int(width), height=int(height), token=job.image_token)
            if not job.image_base64:
                raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="image_token_not_found")

        if not job.image_base64:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="image_base64_required")

        np_img = self._decode_base64(job.image_base64)
        token = hashlib.md5(np_img.tobytes()).hexdigest()
        _store_preloaded_image(token, np_img, variant)

        if self._is_superseded(job):
            return SamPreloadResponse(status="superseded", width=int(np_img.shape[1]), height=int(np_img.shape[0]), token=token)

        predictor_manager.set_slot_with_wait(slot_name, np_img, token, variant, image_name)
        height, width = np_img.shape[:2]
        return SamPreloadResponse(status="ready", width=int(width), height=int(height), token=token)

    @staticmethod
    def _decode_base64(image_base64: str) -> np.ndarray:
        _, np_img = _decode_image_base64_impl(image_base64, max_bytes=BASE64_IMAGE_MAX_BYTES, max_dim=BASE64_IMAGE_MAX_DIM, allow_downscale=True)
        return np_img


def _extract_qwen_json_block(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    snippet, detections = _extract_qwen_json_block_impl(text)
    if not detections:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="qwen_parse_error:no_json_block_found")
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
        results.append(QwenDetection(bbox=yolo_box, qwen_label=str(label) if label else None, source="bbox"))
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
        results.append(QwenDetection(bbox=yolo_box, qwen_label=str(label) if label else None, source="bbox_sam"))
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
        results.append(QwenDetection(bbox=yolo_box, qwen_label=str(label) if label else None, source="point"))
        if len(results) >= limit:
            break
    return results


def _sam3_text_detections(
    pil_img: Image.Image,
    payload: Dict[str, Any],
    text_prompt: str,
    limit: Optional[int],
    *,
    min_score: Optional[float] = None,
    masks_arr: Optional[np.ndarray] = None,
    min_size: Optional[float] = None,
    simplify_epsilon: Optional[float] = None,
    collected_masks: Optional[List[np.ndarray]] = None,
) -> List[QwenDetection]:
    width, height = pil_img.width, pil_img.height
    boxes_source = payload.get("boxes")
    scores_source = payload.get("scores")
    masks = payload.get("masks")
    if isinstance(boxes_source, torch.Tensor):
        boxes_iter: Sequence[Any] = boxes_source.cpu().numpy()
    elif boxes_source is None:
        boxes_iter = []
    else:
        boxes_iter = boxes_source
    if isinstance(scores_source, torch.Tensor):
        scores_iter: Sequence[Any] = scores_source.cpu().numpy().tolist()
    elif scores_source is None:
        scores_iter = []
    else:
        scores_iter = scores_source
    if masks_arr is None and masks is not None:
        if isinstance(masks, torch.Tensor):
            masks_arr = masks.cpu().numpy()
        else:
            masks_arr = np.asarray(masks)
    detections: List[QwenDetection] = []
    if limit is None:
        numeric_limit: Optional[int] = None
    else:
        try:
            numeric_limit = int(limit)
        except (TypeError, ValueError):
            numeric_limit = None
        else:
            if numeric_limit <= 0:
                numeric_limit = None
    # Prefer highest-score boxes first when scores are available, since we may be limiting outputs.
    order = list(range(len(boxes_iter)))
    try:
        if scores_iter is not None and len(scores_iter) >= len(order):
            order.sort(
                key=lambda i: float(scores_iter[i]) if i < len(scores_iter) and scores_iter[i] is not None else -1e9,
                reverse=True,
            )
    except Exception:
        order = list(range(len(boxes_iter)))

    for idx in order:
        box = boxes_iter[idx]
        coords = np.asarray(box, dtype=np.float32).tolist()
        if len(coords) < 4:
            continue
        x_min, y_min, x_max, y_max = coords[:4]
        if x_max <= x_min or y_max <= y_min:
            continue
        yolo_box = _xyxy_to_yolo_norm_list(width, height, x_min, y_min, x_max, y_max)
        score_val = None
        if idx < len(scores_iter):
            try:
                score_val = float(scores_iter[idx])
            except (TypeError, ValueError):
                score_val = None
        if min_score is not None and score_val is not None and score_val < min_score:
            continue
        area = max(0.0, (x_max - x_min) * (y_max - y_min))
        if masks_arr is not None and idx < len(masks_arr):
            try:
                area = float(np.count_nonzero(masks_arr[idx]))
            except Exception:
                area = area
        if min_size is not None:
            try:
                if area < float(min_size):
                    continue
            except Exception:
                pass
        mask_payload = None
        mask_value = None
        if masks_arr is not None and idx < len(masks_arr):
            mask_value = masks_arr[idx]
            mask_payload = _encode_binary_mask_impl(mask_value, max_bytes=MASK_ENCODE_MAX_BYTES)
        if collected_masks is not None:
            collected_masks.append(mask_value)
        detections.append(
            QwenDetection(
                bbox=yolo_box,
                qwen_label=text_prompt,
                source="sam3_text",
                score=score_val,
                mask=mask_payload,
                simplify_epsilon=simplify_epsilon,
            )
        )
        if numeric_limit is not None and len(detections) >= numeric_limit:
            break
    if detections or masks_arr is None:
        return detections
    for idx, mask in enumerate(masks_arr):
        x_min, y_min, x_max, y_max = _mask_to_bounding_box(mask)
        if x_max <= x_min or y_max <= y_min:
            continue
        yolo_box = _xyxy_to_yolo_norm_list(width, height, x_min, y_min, x_max, y_max)
        score_val = None
        if idx < len(scores_iter):
            try:
                score_val = float(scores_iter[idx])
            except (TypeError, ValueError):
                score_val = None
        if min_score is not None and score_val is not None and score_val < min_score:
            continue
        area = max(0.0, (x_max - x_min) * (y_max - y_min))
        try:
            area = float(np.count_nonzero(mask))
        except Exception:
            area = area
        if min_size is not None:
            try:
                if area < float(min_size):
                    continue
            except Exception:
                pass
        detections.append(
            QwenDetection(
                bbox=yolo_box,
                qwen_label=text_prompt,
                source="sam3_text",
                score=score_val,
                mask=_encode_binary_mask_impl(mask, max_bytes=MASK_ENCODE_MAX_BYTES),
                simplify_epsilon=simplify_epsilon,
            )
        )
        if collected_masks is not None:
            collected_masks.append(mask)
        if numeric_limit is not None and len(detections) >= numeric_limit:
            break
    return detections


def _run_sam3_text_inference(
    pil_img: Image.Image,
    text_prompt: str,
    threshold: float,
    mask_threshold: float,
    limit: Optional[int],
    *,
    return_masks: bool = False,
    min_size: Optional[float] = None,
    simplify_epsilon: Optional[float] = None,
    processor_override: Optional[Any] = None,
    state: Optional[Any] = None,
) -> List[QwenDetection] | Tuple[List[QwenDetection], Optional[List[np.ndarray]]]:
    """
    Run SAM3 text inference. By default returns detections list; callers that need masks should
    inspect the second element when `return_masks=True`.
    """
    if processor_override is not None:
        processor = processor_override
    else:
        _, processor, _ = _ensure_sam3_text_runtime()
    try:
        processor.set_confidence_threshold(float(threshold))
    except Exception:
        # If the processor refuses the threshold, continue with its default.
        pass
    normalized_limit: Optional[int]
    if limit is None:
        normalized_limit = None
    else:
        try:
            normalized_limit = max(1, int(limit))
        except (TypeError, ValueError):
            normalized_limit = None
    img_state = state if state is not None else processor.set_image(pil_img)
    masks_arr: Optional[np.ndarray] = None
    try:
        output = processor.set_text_prompt(state=img_state, prompt=text_prompt)
    except KeyError:
        # Box-only checkpoints (enable_segmentation=False) do not emit pred_masks.
        # Fall back to raw model output and extract boxes/scores manually.
        try:
            raw = processor.model.forward_grounding(
                backbone_out=img_state.get("backbone_out", {}),
                find_input=processor.find_stage,
                find_target=None,
                geometric_prompt=img_state.get("geometric_prompt", processor.model._get_dummy_prompt()),
            )
            boxes_xyxy = raw.get("pred_boxes_xyxy")
            if boxes_xyxy is None:
                boxes_xyxy = raw.get("pred_boxes")
            scores = None
            logits = raw.get("pred_logits")
            if logits is not None:
                try:
                    scores = torch.sigmoid(logits.squeeze(-1))
                except Exception:  # noqa: BLE001
                    try:
                        scores = torch.sigmoid(logits)
                    except Exception:  # noqa: BLE001
                        scores = None
            output = {
                "boxes": boxes_xyxy,
                "scores": scores,
                # no masks for box-only checkpoints
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("SAM3 box-only text prompt fallback failed: %s", exc)
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_text_grounding_failed:{exc}") from exc
    try:
        if output is not None and hasattr(output, "pred_masks"):
            masks = output.pred_masks
            if masks is not None:
                try:
                    masks_arr = masks.cpu().numpy()
                except Exception:
                    try:
                        masks_arr = np.asarray(masks)
                    except Exception:
                        masks_arr = None
        if masks_arr is not None:
            try:
                masks_arr = (masks_arr >= float(mask_threshold)).astype(np.uint8)
            except Exception:
                # If thresholding fails, keep the raw masks.
                pass
    except Exception:
        masks_arr = None
    collected_masks: Optional[List[np.ndarray]] = [] if return_masks else None
    preds = _sam3_text_detections(
        pil_img,
        output,
        text_prompt,
        normalized_limit,
        min_score=float(threshold),
        masks_arr=masks_arr,
        min_size=min_size,
        simplify_epsilon=simplify_epsilon,
        collected_masks=collected_masks,
    )
    aligned_masks: Optional[List[np.ndarray]]
    if collected_masks is None:
        aligned_masks = None
    else:
        aligned_masks = collected_masks
    return (preds, aligned_masks) if return_masks else preds


def _run_sam3_visual_inference(
    pil_img: Image.Image,
    bbox_xywh: Tuple[float, float, float, float],
    threshold: float,
    mask_threshold: float,
    limit: Optional[int],
    *,
    return_masks: bool = False,
    min_size: Optional[float] = None,
    simplify_epsilon: Optional[float] = None,
    processor_override: Optional[Any] = None,
    state: Optional[Any] = None,
) -> List[QwenDetection] | Tuple[List[QwenDetection], Optional[List[np.ndarray]]]:
    """
    Run SAM3 with a single positive visual (box) prompt. By default returns detections list;
    callers that need masks should inspect the second element when `return_masks=True`.
    """
    if processor_override is not None:
        processor = processor_override
    else:
        _, processor, _ = _ensure_sam3_text_runtime()
    try:
        processor.set_confidence_threshold(float(threshold))
    except Exception:
        pass
    normalized_limit: Optional[int]
    if limit is None:
        normalized_limit = None
    else:
        try:
            normalized_limit = max(1, int(limit))
        except (TypeError, ValueError):
            normalized_limit = None
    img_state = state if state is not None else processor.set_image(pil_img)
    img_w, img_h = float(pil_img.width), float(pil_img.height)
    x, y, w, h = bbox_xywh
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    try:
        output = processor.add_geometric_prompt([cx, cy, w_norm, h_norm], True, state=img_state)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_visual_prompt_failed:{exc}") from exc
    masks_arr: Optional[np.ndarray] = None
    mask_logits = None
    if isinstance(output, Mapping):
        if "masks_logits" in output and output.get("masks_logits") is not None:
            mask_logits = output.get("masks_logits")
        elif "masks" in output and output.get("masks") is not None:
            mask_logits = output.get("masks")
    if mask_logits is None and isinstance(img_state, Mapping):
        if "masks_logits" in img_state and img_state.get("masks_logits") is not None:
            mask_logits = img_state.get("masks_logits")
        elif "masks" in img_state and img_state.get("masks") is not None:
            mask_logits = img_state.get("masks")
    try:
        threshold_val = float(mask_threshold)
    except Exception:
        threshold_val = 0.5
    threshold_val = max(0.0, min(1.0, threshold_val))
    # Normalize mask logits into a numpy array before thresholding.
    try:
        def _sigmoid_np(arr: np.ndarray) -> np.ndarray:
            try:
                return 1.0 / (1.0 + np.exp(-np.clip(arr, -50, 50)))
            except Exception:
                return 1.0 / (1.0 + np.exp(-arr))

        if isinstance(mask_logits, (list, tuple)):
            if any(isinstance(m, torch.Tensor) for m in mask_logits):
                stacked = [m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m) for m in mask_logits]
                mask_logits = np.stack(stacked)
            else:
                mask_logits = np.asarray(mask_logits)
        if isinstance(mask_logits, torch.Tensor):
            try:
                probs = mask_logits
                try:
                    min_v = float(probs.min())
                    max_v = float(probs.max())
                    if not (0.0 <= min_v <= 1.0 and 0.0 <= max_v <= 1.0):
                        probs = torch.sigmoid(probs)
                except Exception:
                    probs = torch.sigmoid(probs)
                masks_arr = (probs > threshold_val).cpu().numpy()
            except Exception:
                masks_arr = mask_logits.detach().cpu().numpy()
        elif mask_logits is not None:
            masks_np = np.asarray(mask_logits)
            if masks_np.dtype == object:
                try:
                    masks_np = np.stack([np.asarray(m) for m in masks_np])
                except Exception:
                    masks_np = None
            if masks_np is not None:
                if masks_np.dtype == bool or (
                    np.issubdtype(masks_np.dtype, np.floating)
                    and np.nanmin(masks_np) >= 0.0
                    and np.nanmax(masks_np) <= 1.0
                ):
                    probs_np = masks_np
                else:
                    probs_np = _sigmoid_np(masks_np)
                masks_arr = probs_np > threshold_val
        # Normalize mask shape to (N, H, W) where possible
        if masks_arr is not None:
            masks_arr = np.asarray(masks_arr)
            if masks_arr.dtype == object:
                flattened = [np.asarray(m) for m in masks_arr]
                masks_arr = np.stack(flattened)
            if masks_arr.ndim == 2:
                masks_arr = masks_arr[None, ...]
            elif masks_arr.ndim == 4 and masks_arr.shape[1] == 1:
                masks_arr = masks_arr[:, 0, ...]
            elif masks_arr.ndim == 4 and masks_arr.shape[-1] == 1:
                masks_arr = masks_arr[..., 0]
    except Exception:
        masks_arr = None
    def _to_numpy_safe(val: Any) -> Optional[np.ndarray]:
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            try:
                return val.detach().cpu().numpy()
            except Exception:
                return None
        try:
            return np.asarray(val)
        except Exception:
            return None

    payload_for_detection: Dict[str, Any] = {}
    if isinstance(output, Mapping):
        boxes_val = _to_numpy_safe(output.get("boxes"))
        scores_val = _to_numpy_safe(output.get("scores"))
        masks_val = _to_numpy_safe(output.get("masks"))
        if boxes_val is not None:
            payload_for_detection["boxes"] = boxes_val
        if scores_val is not None:
            payload_for_detection["scores"] = scores_val
        if masks_val is not None:
            payload_for_detection["masks"] = masks_val
    collected_masks: Optional[List[np.ndarray]] = [] if return_masks else None
    detections = _sam3_text_detections(
        pil_img,
        payload_for_detection,
        "visual",
        normalized_limit,
        min_score=float(threshold),
        masks_arr=masks_arr,
        min_size=min_size,
        simplify_epsilon=simplify_epsilon,
        collected_masks=collected_masks,
    )
    # Drop the seed box if SAM returns it again (dedupe by IoU against the input box).
    seed_xyxy = (bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3])
    def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    aligned_masks: Optional[List[np.ndarray]]
    if collected_masks is None:
        aligned_masks = None
    else:
        aligned_masks = collected_masks
    if detections:
        filtered_dets: List[QwenDetection] = []
        filtered_masks: List[np.ndarray] = []
        for det_idx, det in enumerate(detections):
            bbox = det.bbox or []
            if len(bbox) < 4:
                continue
            det_xyxy = _yolo_to_xyxy_int(pil_img.width, pil_img.height, bbox)
            if _iou(seed_xyxy, det_xyxy) > 0.9:
                continue
            filtered_dets.append(det)
            if aligned_masks is not None and det_idx < len(aligned_masks):
                filtered_masks.append(aligned_masks[det_idx])
        detections = filtered_dets
        if aligned_masks is not None:
            aligned_masks = filtered_masks
    return (detections, aligned_masks) if return_masks else detections


def _run_sam3_visual_inference_multi(
    pil_img: Image.Image,
    bboxes_xywh: List[Tuple[float, float, float, float]],
    bbox_labels: Optional[List[bool]],
    threshold: float,
    mask_threshold: float,
    limit: Optional[int],
    *,
    return_masks: bool = False,
    min_size: Optional[float] = None,
    simplify_epsilon: Optional[float] = None,
    processor_override: Optional[Any] = None,
    state: Optional[Any] = None,
) -> List[QwenDetection] | Tuple[List[QwenDetection], Optional[List[np.ndarray]]]:
    """
    Run SAM3 with multiple positive visual (box) prompts. Uses a shared image state
    to accumulate prompts and returns detections from the combined prompt set.
    """
    if processor_override is not None:
        processor = processor_override
    else:
        _, processor, _ = _ensure_sam3_text_runtime()
    try:
        processor.set_confidence_threshold(float(threshold))
    except Exception:
        pass
    normalized_limit: Optional[int]
    if limit is None:
        normalized_limit = None
    else:
        try:
            normalized_limit = max(1, int(limit))
        except (TypeError, ValueError):
            normalized_limit = None
    if not isinstance(bboxes_xywh, (list, tuple)) or not bboxes_xywh:
        empty = ([], []) if return_masks else []
        return empty
    cleaned_boxes: List[Tuple[float, float, float, float]] = []
    for bbox in bboxes_xywh:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            continue
        cleaned_boxes.append((x, y, w, h))
    if not cleaned_boxes:
        empty = ([], []) if return_masks else []
        return empty
    labels: List[bool]
    if bbox_labels is None:
        labels = [True] * len(cleaned_boxes)
    else:
        labels = list(bbox_labels)
        if len(labels) < len(cleaned_boxes):
            labels.extend([True] * (len(cleaned_boxes) - len(labels)))
        elif len(labels) > len(cleaned_boxes):
            labels = labels[: len(cleaned_boxes)]
    img_state = state if state is not None else processor.set_image(pil_img)
    img_w, img_h = float(pil_img.width), float(pil_img.height)
    output = None
    for bbox_xywh, label in zip(cleaned_boxes, labels):
        x, y, w, h = bbox_xywh
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        try:
            output = processor.add_geometric_prompt([cx, cy, w_norm, h_norm], bool(label), state=img_state)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_visual_prompt_failed:{exc}") from exc
    masks_arr: Optional[np.ndarray] = None
    mask_logits = None
    if isinstance(output, Mapping):
        if "masks_logits" in output and output.get("masks_logits") is not None:
            mask_logits = output.get("masks_logits")
        elif "masks" in output and output.get("masks") is not None:
            mask_logits = output.get("masks")
    if mask_logits is None and isinstance(img_state, Mapping):
        if "masks_logits" in img_state and img_state.get("masks_logits") is not None:
            mask_logits = img_state.get("masks_logits")
        elif "masks" in img_state and img_state.get("masks") is not None:
            mask_logits = img_state.get("masks")
    try:
        threshold_val = float(mask_threshold)
    except Exception:
        threshold_val = 0.5
    threshold_val = max(0.0, min(1.0, threshold_val))
    try:
        def _sigmoid_np(arr: np.ndarray) -> np.ndarray:
            try:
                return 1.0 / (1.0 + np.exp(-np.clip(arr, -50, 50)))
            except Exception:
                return 1.0 / (1.0 + np.exp(-arr))

        if isinstance(mask_logits, (list, tuple)):
            if any(isinstance(m, torch.Tensor) for m in mask_logits):
                stacked = [m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m) for m in mask_logits]
                mask_logits = np.stack(stacked)
            else:
                mask_logits = np.asarray(mask_logits)
        if isinstance(mask_logits, torch.Tensor):
            try:
                probs = mask_logits
                try:
                    min_v = float(probs.min())
                    max_v = float(probs.max())
                    if not (0.0 <= min_v <= 1.0 and 0.0 <= max_v <= 1.0):
                        probs = torch.sigmoid(probs)
                except Exception:
                    probs = torch.sigmoid(probs)
                masks_arr = (probs > threshold_val).cpu().numpy()
            except Exception:
                masks_arr = mask_logits.detach().cpu().numpy()
        elif mask_logits is not None:
            masks_np = np.asarray(mask_logits)
            if masks_np.dtype == bool or (
                np.issubdtype(masks_np.dtype, np.floating)
                and np.nanmin(masks_np) >= 0.0
                and np.nanmax(masks_np) <= 1.0
            ):
                probs_np = masks_np
            else:
                probs_np = _sigmoid_np(masks_np)
            masks_arr = probs_np > threshold_val
        if masks_arr is not None:
            masks_arr = np.asarray(masks_arr)
            if masks_arr.dtype == object:
                flattened = [np.asarray(m) for m in masks_arr]
                masks_arr = np.stack(flattened)
            if masks_arr.ndim == 2:
                masks_arr = masks_arr[None, ...]
            elif masks_arr.ndim == 4 and masks_arr.shape[1] == 1:
                masks_arr = masks_arr[:, 0, ...]
            elif masks_arr.ndim == 4 and masks_arr.shape[-1] == 1:
                masks_arr = masks_arr[..., 0]
    except Exception:
        masks_arr = None
    def _to_numpy_safe(val: Any) -> Optional[np.ndarray]:
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            try:
                return val.detach().cpu().numpy()
            except Exception:
                return None
        try:
            return np.asarray(val)
        except Exception:
            return None

    payload_for_detection: Dict[str, Any] = {}
    if isinstance(output, Mapping):
        boxes_val = _to_numpy_safe(output.get("boxes"))
        scores_val = _to_numpy_safe(output.get("scores"))
        masks_val = _to_numpy_safe(output.get("masks"))
        if boxes_val is not None:
            payload_for_detection["boxes"] = boxes_val
        if scores_val is not None:
            payload_for_detection["scores"] = scores_val
        if masks_val is not None:
            payload_for_detection["masks"] = masks_val
    collected_masks: Optional[List[np.ndarray]] = [] if return_masks else None
    detections = _sam3_text_detections(
        pil_img,
        payload_for_detection,
        "visual",
        normalized_limit,
        min_score=float(threshold),
        masks_arr=masks_arr,
        min_size=min_size,
        simplify_epsilon=simplify_epsilon,
        collected_masks=collected_masks,
    )
    seed_boxes_xyxy = [
        (bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3])
        for bx in bboxes_xywh
    ]
    def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    aligned_masks: Optional[List[np.ndarray]]
    if collected_masks is None:
        aligned_masks = None
    else:
        aligned_masks = collected_masks
    if detections:
        filtered_dets: List[QwenDetection] = []
        filtered_masks: List[np.ndarray] = []
        for det_idx, det in enumerate(detections):
            bbox = det.bbox or []
            if len(bbox) < 4:
                continue
            det_xyxy = _yolo_to_xyxy_int(pil_img.width, pil_img.height, bbox)
            if any(_iou(seed_xyxy, det_xyxy) > 0.9 for seed_xyxy in seed_boxes_xyxy):
                continue
            filtered_dets.append(det)
            if aligned_masks is not None and det_idx < len(aligned_masks):
                filtered_masks.append(aligned_masks[det_idx])
        detections = filtered_dets
        if aligned_masks is not None:
            aligned_masks = filtered_masks
    return (detections, aligned_masks) if return_masks else detections


def _ensure_qwen_ready():
    global qwen_model, qwen_processor, qwen_device, qwen_last_error, loaded_qwen_model_id
    if QWEN_IMPORT_ERROR is not None or Qwen3VLForConditionalGeneration is None or AutoProcessor is None or process_vision_info is None:
        detail = f"qwen_dependencies_missing:{QWEN_IMPORT_ERROR}"
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    if packaging_version is not None:
        try:
            import transformers  # local import to avoid import-time failures

            if packaging_version.parse(transformers.__version__) < packaging_version.parse(QWEN_MIN_TRANSFORMERS):
                raise HTTPException(
                    status_code=HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"qwen_transformers_too_old:{transformers.__version__}<{QWEN_MIN_TRANSFORMERS}",
                )
        except HTTPException:
            raise
        except Exception:
            # If we cannot resolve version info, continue and let the load fail if incompatible.
            pass
    if (
        qwen_model is not None
        and qwen_processor is not None
        and loaded_qwen_model_id == active_qwen_model_id
    ):
        return qwen_model, qwen_processor
    with qwen_lock:
        if (
            qwen_model is not None
            and qwen_processor is not None
            and loaded_qwen_model_id == active_qwen_model_id
        ):
            return qwen_model, qwen_processor
        try:
            device = _resolve_qwen_device_impl(QWEN_DEVICE_PREF, torch_module=torch)
        except RuntimeError as exc:  # noqa: BLE001
            qwen_last_error = str(exc)
            raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_device_unavailable:{exc}") from exc
        use_auto_map = QWEN_DEVICE_PREF == "auto" and device.startswith("cuda") and torch.cuda.is_available()
        load_kwargs: Dict[str, Any]
        if use_auto_map:
            load_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
            }
        else:
            dtype = torch.float16 if device.startswith(("cuda", "mps")) else torch.float32
            load_kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
            }
        adapter_path = active_qwen_model_path
        metadata = active_qwen_metadata or {}
        base_model_id = metadata.get("model_id") or QWEN_MODEL_NAME
        if adapter_path and PeftModel is None:
            detail = "qwen_peft_missing"
            if PEFT_IMPORT_ERROR is not None:
                detail = f"{detail}:{PEFT_IMPORT_ERROR}"
            raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
        def _load_candidate(candidate_id: str, processor_source: str) -> Tuple[Any, Any]:
            local_only = _hf_offline_enabled()
            model_local = _load_qwen_vl_model(str(candidate_id), load_kwargs, local_files_only=local_only)
            if adapter_path:
                model_local = PeftModel.from_pretrained(model_local, str(adapter_path))
            if not load_kwargs.get("device_map"):
                model_local.to(device)
            model_local.eval()
            processor_local = AutoProcessor.from_pretrained(
                processor_source,
                min_pixels=QWEN_MIN_PIXELS,
                max_pixels=QWEN_MAX_PIXELS,
                local_files_only=local_only,
            )
            return model_local, processor_local

        def _load_with_online_retry(candidate_id: str, processor_source: str) -> Tuple[Any, Any]:
            try:
                return _load_candidate(candidate_id, processor_source)
            except Exception as exc:  # noqa: BLE001
                if _hf_offline_enabled():
                    logger.warning("[qwen] offline load failed; retrying with HF online: %s", exc)
                    _set_hf_offline(False)
                    try:
                        return _load_candidate(candidate_id, processor_source)
                    finally:
                        _enable_hf_offline_defaults()
                raise

        try:
            processor_source = str(adapter_path) if adapter_path else str(base_model_id)
            model, processor = _load_with_online_retry(str(base_model_id), processor_source)
        except Exception as exc:  # noqa: BLE001
            fallback_id = _strip_qwen_model_suffix_impl(str(base_model_id))
            if fallback_id:
                try:
                    logger.warning("Qwen model %s not found; falling back to %s", base_model_id, fallback_id)
                    processor_source = str(adapter_path) if adapter_path else str(fallback_id)
                    model, processor = _load_with_online_retry(str(fallback_id), processor_source)
                except Exception as fallback_exc:  # noqa: BLE001
                    qwen_last_error = str(fallback_exc)
                    detail = _format_qwen_load_error_impl(fallback_exc, torch_module=torch)
                    raise HTTPException(
                        status_code=HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"qwen_load_failed:{detail}",
                    ) from fallback_exc
            else:
                qwen_last_error = str(exc)
                detail = _format_qwen_load_error_impl(exc, torch_module=torch)
                raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_load_failed:{detail}") from exc
        qwen_model = model
        qwen_processor = processor
        qwen_device = device
        qwen_last_error = None
        loaded_qwen_model_id = active_qwen_model_id
        _enable_hf_offline_defaults()
        return model, processor


def _unload_qwen_runtime() -> None:
    """Release Qwen model/processor to free device memory."""
    global qwen_model, qwen_processor, qwen_device, loaded_qwen_model_id
    global qwen_caption_cache, qwen_caption_order
    state = {
        "qwen_model": qwen_model,
        "qwen_processor": qwen_processor,
        "qwen_device": qwen_device,
        "loaded_qwen_model_id": loaded_qwen_model_id,
        "qwen_caption_cache": qwen_caption_cache,
        "qwen_caption_order": qwen_caption_order,
    }
    _unload_qwen_runtime_impl(
        state=state,
        torch_module=torch,
        gc_module=gc,
        logger=logger,
        deque_factory=deque,
    )
    qwen_model = state["qwen_model"]
    qwen_processor = state["qwen_processor"]
    qwen_device = state["qwen_device"]
    loaded_qwen_model_id = state["loaded_qwen_model_id"]
    qwen_caption_cache = state["qwen_caption_cache"]
    qwen_caption_order = state["qwen_caption_order"]


def _ensure_qwen_ready_for_caption(model_id_override: str) -> Tuple[Any, Any]:
    global qwen_device, qwen_last_error
    global qwen_caption_cache, qwen_caption_order
    state = {
        "qwen_caption_cache": qwen_caption_cache,
        "qwen_caption_order": qwen_caption_order,
        "qwen_device": qwen_device,
        "qwen_last_error": qwen_last_error,
    }
    device_pref = QWEN_DEVICE_PREF
    if device_pref == "auto":
        try:
            resolved_device = _resolve_qwen_device_impl(QWEN_DEVICE_PREF, torch_module=torch)
            device_pref = str(resolved_device)
        except Exception:
            device_pref = QWEN_DEVICE_PREF
    # Captioning should never use multi-GPU sharding; force a concrete device id if possible.
    if device_pref == "cuda":
        device_pref = "cuda:0"
    if device_pref != "auto" and state.get("qwen_caption_cache"):
        for cache_key, cache_entry in list(state["qwen_caption_cache"].items()):
            if not cache_entry:
                continue
            try:
                cached_model, _ = cache_entry
            except Exception:
                continue
            try:
                hf_map = getattr(cached_model, "hf_device_map", None)
                if hf_map:
                    unique_devices = {str(dev) for dev in hf_map.values()}
                    if len(unique_devices) > 1:
                        _evict_qwen_caption_entry_impl(
                            cache_key,
                            cache_entry,
                            torch_module=torch,
                            gc_module=gc,
                        )
                        state["qwen_caption_cache"].pop(cache_key, None)
                        try:
                            state["qwen_caption_order"].remove(cache_key)
                        except ValueError:
                            pass
            except Exception:
                continue
    try:
        model, processor = _ensure_qwen_ready_for_caption_impl(
            model_id_override,
            state=state,
            qwen_lock=qwen_lock,
            import_error=QWEN_IMPORT_ERROR,
            qwen_model_cls=Qwen3VLForConditionalGeneration,
            qwen_processor_cls=AutoProcessor,
            process_vision_info=process_vision_info,
            packaging_version=packaging_version,
            min_transformers=QWEN_MIN_TRANSFORMERS,
            resolve_device_fn=lambda: _resolve_qwen_device_impl(device_pref, torch_module=torch),
            device_pref=device_pref,
            torch_module=torch,
            load_qwen_model_fn=_load_qwen_vl_model,
            hf_offline_enabled_fn=_hf_offline_enabled,
            set_hf_offline_fn=_set_hf_offline,
            enable_hf_offline_defaults_fn=_enable_hf_offline_defaults,
            strip_model_suffix_fn=_strip_qwen_model_suffix_impl,
            format_load_error_fn=lambda exc: _format_qwen_load_error_impl(exc, torch_module=torch),
            min_pixels=QWEN_MIN_PIXELS,
            max_pixels=QWEN_MAX_PIXELS,
            caption_cache_limit=QWEN_CAPTION_CACHE_LIMIT,
            evict_entry_fn=lambda cache_key, cache_entry: _evict_qwen_caption_entry_impl(
                cache_key,
                cache_entry,
                torch_module=torch,
                gc_module=gc,
            ),
            logger=logger,
        )
    except RuntimeError as exc:  # noqa: BLE001
        detail = str(exc)
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=detail) from exc
    qwen_caption_cache = state["qwen_caption_cache"]
    qwen_caption_order = state["qwen_caption_order"]
    qwen_device = state["qwen_device"]
    qwen_last_error = state["qwen_last_error"]
    return model, processor


## NOTE: caption prompt helpers call *_impl directly to avoid wrapper drift.


_CAPTION_GENERIC_OPENERS = (
    "an aerial view",
    "aerial view",
    "from a high angle",
    "a drone image",
    "a bird's-eye view",
    "overhead view",
)

_QWEN_THINKING_REASONING_RE = re.compile(
    r"(?:\bgot it\b|\blet'?s\b|\bfirst\b|\bsecond\b|\bthird\b|\bstep\b|\bi need\b|\bnow\b|\bthe task\b)",
    re.IGNORECASE,
)
_QWEN_CAPTION_META_RE = re.compile(
    r"(authoritative|as indicated|label hint|bounding box|bbox|coordinates|hinted|counts are provided)",
    re.IGNORECASE,
)


## NOTE: caption cleanup/merge helpers call *_impl directly to avoid wrapper drift.


def _group_hints_by_window(
    label_hints: Sequence[QwenCaptionHint],
    x_positions: Sequence[int],
    y_positions: Sequence[int],
    window: int,
) -> Dict[Tuple[int, int], List[QwenCaptionHint]]:
    grouped: Dict[Tuple[int, int], List[QwenCaptionHint]] = {(x0, y0): [] for y0 in y_positions for x0 in x_positions}
    for hint in label_hints:
        if not hint.bbox or len(hint.bbox) != 4:
            continue
        bx1, by1, bx2, by2 = hint.bbox
        try:
            cx = (float(bx1) + float(bx2)) * 0.5
            cy = (float(by1) + float(by2)) * 0.5
        except (TypeError, ValueError):
            continue
        x0_match = None
        for x0 in x_positions:
            if x0 <= cx <= x0 + window:
                x0_match = x0
                break
        if x0_match is None:
            continue
        y0_match = None
        for y0 in y_positions:
            if y0 <= cy <= y0 + window:
                y0_match = y0
                break
        if y0_match is None:
            continue
        nx1 = max(0.0, min(float(bx1) - x0_match, window))
        ny1 = max(0.0, min(float(by1) - y0_match, window))
        nx2 = max(0.0, min(float(bx2) - x0_match, window))
        ny2 = max(0.0, min(float(by2) - y0_match, window))
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        grouped[(x0_match, y0_match)].append(
            QwenCaptionHint(
                label=hint.label,
                bbox=[nx1, ny1, nx2, ny2],
                confidence=hint.confidence,
            )
        )
    return grouped


def _run_qwen_inference(
    prompt: str,
    pil_img: Image.Image,
    max_new_tokens: Optional[int] = None,
    system_prompt_override: Optional[str] = None,
    model_id_override: Optional[str] = None,
    runtime_override: Optional[Tuple[Any, Any]] = None,
    decode_override: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int]:
    """Execute a Qwen 3 VL inference following the reference recipe."""
    if runtime_override is not None:
        model, processor = runtime_override
    elif model_id_override:
        model, processor = _ensure_qwen_ready_for_caption(model_id_override)
    else:
        model, processor = _ensure_qwen_ready()
    messages: List[Dict[str, Any]] = []
    sys_prompt = system_prompt_override if system_prompt_override is not None else (active_qwen_metadata or {}).get("system_prompt")
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _agent_full_trace_write(
        {
            "type": "llm_input",
            "source": "qwen_inference",
            "model_id": model_id_override or (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME,
            "messages": messages,
            "prompt_text": text,
            "max_new_tokens": int(max_new_tokens) if max_new_tokens is not None else None,
            "decode_override": decode_override,
        }
    )
    image_inputs, video_inputs = process_vision_info(messages)
    max_seq_len = _resolve_qwen_max_seq_len(model)
    max_input_len = None
    requested_max = int(max_new_tokens) if max_new_tokens is not None else QWEN_MAX_NEW_TOKENS
    if max_seq_len:
        if requested_max >= max_seq_len:
            requested_max = max(1, max_seq_len - 1)
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
    device = model_device or qwen_device or _resolve_qwen_device_impl(QWEN_DEVICE_PREF, torch_module=torch)
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
        "max_new_tokens": int(max_new_tokens) if max_new_tokens is not None else QWEN_MAX_NEW_TOKENS,
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
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    _agent_full_trace_write(
        {
            "type": "llm_output",
            "source": "qwen_inference",
            "model_id": model_id_override or (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME,
            "output_text": output_text,
        }
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
        model, processor = runtime_override
    elif model_id_override:
        model, processor = _ensure_qwen_ready_for_caption(model_id_override)
    else:
        model, processor = _ensure_qwen_ready()
    if assistant_prefix:
        add_generation_prompt = True
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    if assistant_prefix:
        text = f"{text}{assistant_prefix}"
    _agent_full_trace_write(
        {
            "type": "llm_input",
            "model_id": model_id_override or (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME,
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
        "max_new_tokens": int(max_new_tokens) if max_new_tokens is not None else QWEN_MAX_NEW_TOKENS,
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
    thinking_processor = _qwen_build_thinking_effort_processor(tokenizer, thinking_effort, thinking_scale_factor)
    immediate_processor = _qwen_build_immediate_action_processor(
        tokenizer,
        immediate_action_bias,
        immediate_action_min_chars,
        immediate_action_min_seconds,
        immediate_action_logit_bias,
    )
    _qwen_append_logits_processor(gen_kwargs, thinking_processor)
    _qwen_append_logits_processor(gen_kwargs, immediate_processor)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    output_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    _agent_full_trace_write(
        {
            "type": "llm_output",
            "model_id": model_id_override or (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME,
            "output_text": output_text,
        }
    )
    return output_text


def _run_qwen_chat_stream(
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
) -> Iterator[str]:
    if runtime_override is not None:
        model, processor = runtime_override
    elif model_id_override:
        model, processor = _ensure_qwen_ready_for_caption(model_id_override)
    else:
        model, processor = _ensure_qwen_ready()
    if assistant_prefix:
        add_generation_prompt = True
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    if assistant_prefix:
        text = f"{text}{assistant_prefix}"
    _agent_full_trace_write(
        {
            "type": "llm_input",
            "model_id": model_id_override or (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME,
            "messages": messages,
            "prompt_text": text,
            "max_new_tokens": int(max_new_tokens) if max_new_tokens is not None else None,
            "decode_override": decode_override,
            "thinking_effort": thinking_effort,
            "thinking_scale_factor": thinking_scale_factor,
            "tools": tools,
            "chat_template_kwargs": chat_template_kwargs,
            "assistant_prefix": assistant_prefix,
        }
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        output_text = _run_qwen_chat(
            messages,
            max_new_tokens=max_new_tokens,
            model_id_override=model_id_override,
            runtime_override=(model, processor),
            decode_override=decode_override,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            assistant_prefix=assistant_prefix,
            thinking_effort=thinking_effort,
            thinking_scale_factor=thinking_scale_factor,
            immediate_action_bias=immediate_action_bias,
            immediate_action_min_chars=immediate_action_min_chars,
            immediate_action_min_seconds=immediate_action_min_seconds,
            immediate_action_logit_bias=immediate_action_logit_bias,
        )
        yield output_text
        return
    try:
        from transformers import TextIteratorStreamer
    except Exception:
        output_text = _run_qwen_chat(
            messages,
            max_new_tokens=max_new_tokens,
            model_id_override=model_id_override,
            runtime_override=(model, processor),
            decode_override=decode_override,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            assistant_prefix=assistant_prefix,
            thinking_effort=thinking_effort,
            thinking_scale_factor=thinking_scale_factor,
            immediate_action_bias=immediate_action_bias,
            immediate_action_min_chars=immediate_action_min_chars,
            immediate_action_min_seconds=immediate_action_min_seconds,
            immediate_action_logit_bias=immediate_action_logit_bias,
        )
        yield output_text
        return
    image_inputs, video_inputs = process_vision_info(messages)
    max_seq_len = _resolve_qwen_max_seq_len(model)
    requested_max = int(max_new_tokens) if max_new_tokens is not None else QWEN_MAX_NEW_TOKENS
    if max_seq_len:
        if requested_max >= max_seq_len:
            requested_max = max(1, max_seq_len - 1)
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
    max_input_len = None
    if max_seq_len:
        if effective_len + requested_max > max_seq_len:
            requested_max = max(1, max_seq_len - effective_len)
        if effective_len > max_seq_len:
            if vision_tokens is not None:
                max_input_len = max(1, max_seq_len - requested_max - vision_tokens + num_images)
            else:
                max_input_len = max(1, max_seq_len - requested_max)
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
        "max_new_tokens": int(max_new_tokens) if max_new_tokens is not None else QWEN_MAX_NEW_TOKENS,
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
    thinking_processor = _qwen_build_thinking_effort_processor(tokenizer, thinking_effort, thinking_scale_factor)
    immediate_processor = _qwen_build_immediate_action_processor(
        tokenizer,
        immediate_action_bias,
        immediate_action_min_chars,
        immediate_action_min_seconds,
        immediate_action_logit_bias,
    )
    _qwen_append_logits_processor(gen_kwargs, thinking_processor)
    _qwen_append_logits_processor(gen_kwargs, immediate_processor)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        timeout=600.0,
    )
    gen_kwargs["streamer"] = streamer
    generated_text = ""

    def _generate() -> None:
        with torch.inference_mode():
            model.generate(**inputs, **gen_kwargs)

    thread = threading.Thread(target=_generate, name="qwen-chat-streamer", daemon=True)
    thread.start()
    try:
        for new_text in streamer:
            generated_text += new_text
            yield generated_text
    except queue.Empty:
        _agent_full_trace_write(
            {
                "type": "llm_stream_timeout",
                "model_id": model_id_override or (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME,
                "generated_chars": len(generated_text),
            }
        )
    thread.join()
    _agent_full_trace_write(
        {
            "type": "llm_output",
            "model_id": model_id_override or (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME,
            "output_text": generated_text,
        }
    )


def _load_qwen_vl_model(model_id: str, load_kwargs: Dict[str, Any], local_files_only: bool = False) -> Any:
    if AutoConfig is None or AutoModelForCausalLM is None:
        return Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_id), local_files_only=local_files_only, **load_kwargs
        )
    if not QWEN_TRUST_REMOTE_CODE:
        if _is_qwen_moe_model_id(str(model_id)) and Qwen3VLMoeForConditionalGeneration is not None:
            return Qwen3VLMoeForConditionalGeneration.from_pretrained(
                str(model_id), local_files_only=local_files_only, **load_kwargs
            )
        try:
            config = AutoConfig.from_pretrained(
                str(model_id), trust_remote_code=False, local_files_only=local_files_only
            )
            model_type = getattr(config, "model_type", None)
            if model_type == "qwen3_vl_moe" and Qwen3VLMoeForConditionalGeneration is not None:
                return Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    str(model_id), local_files_only=local_files_only, **load_kwargs
                )
            if model_type not in (None, "qwen3_vl", "qwen3_vl_moe"):
                logging.warning(
                    "Qwen model_type=%s may require trust_remote_code; set QWEN_TRUST_REMOTE_CODE=1 to enable.",
                    model_type,
                )
        except Exception as exc:
            logging.warning(
                "Qwen config load failed without trust_remote_code; set QWEN_TRUST_REMOTE_CODE=1 if needed. (%s)",
                exc,
            )
        return Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_id), local_files_only=local_files_only, **load_kwargs
        )
    try:
        config = AutoConfig.from_pretrained(
            str(model_id), trust_remote_code=True, local_files_only=local_files_only
        )
    except Exception:
        config = None
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if model_type == "qwen3_vl_moe" and Qwen3VLMoeForConditionalGeneration is not None:
            return Qwen3VLMoeForConditionalGeneration.from_pretrained(
                str(model_id), local_files_only=local_files_only, **load_kwargs
            )
        if model_type not in (None, "qwen3_vl", "qwen3_vl_moe"):
            return AutoModelForCausalLM.from_pretrained(
                str(model_id), trust_remote_code=True, local_files_only=local_files_only, **load_kwargs
            )
    return Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_id), local_files_only=local_files_only, **load_kwargs
    )


## NOTE: Qwen text helpers call *_impl directly where needed.


def resolve_image_payload(
    image_base64: Optional[str],
    image_token: Optional[str],
    sam_variant: Optional[str],
) -> Tuple[Image.Image, np.ndarray, str]:
    variant = _default_variant(sam_variant)
    if image_token:
        cached = _fetch_preloaded_image(image_token, variant)
        if cached is None:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="image_token_not_found")
        pil_img = Image.fromarray(cached)
        return pil_img, cached, image_token
    pil_img, np_img = _decode_image_base64_impl(image_base64, max_bytes=BASE64_IMAGE_MAX_BYTES, max_dim=BASE64_IMAGE_MAX_DIM, allow_downscale=True)
    token = hashlib.md5(np_img.tobytes()).hexdigest()
    _store_preloaded_image(token, np_img, variant)
    return pil_img, np_img, token


sam_preload_manager = SamPreloadManager()
AGENT_TOOL_REGISTRY: Dict[str, Any] = {}
_AGENT_ACTIVE_IMAGE_TOKEN: Optional[str] = None
_AGENT_ACTIVE_IMAGE_BASE64: Optional[str] = None
_AGENT_ACTIVE_DATASET_ID: Optional[str] = None
_AGENT_ACTIVE_LABELMAP: Optional[List[str]] = None
_AGENT_ACTIVE_GLOSSARY: Optional[str] = None
_AGENT_ACTIVE_OVERALL_CAPTION: Optional[str] = None
_AGENT_ACTIVE_WINDOWED_CAPTIONS: Optional[List[Dict[str, Any]]] = None
_AGENT_ACTIVE_INSPECTED_WINDOWS: Optional[Set[Tuple[int, int, int, int]]] = None
_AGENT_ACTIVE_CLASSIFIER_ID: Optional[str] = None
_AGENT_ACTIVE_TIGHTEN_FP: bool = False
_AGENT_ACTIVE_DETECTOR_CONF: Optional[float] = None
_AGENT_ACTIVE_SAM3_SCORE_THR: Optional[float] = None
_AGENT_ACTIVE_SAM3_MASK_THR: Optional[float] = None
_AGENT_ACTIVE_CLASSIFIER_MIN_PROB: Optional[float] = None
_AGENT_ACTIVE_CLASSIFIER_MARGIN: Optional[float] = None
_AGENT_ACTIVE_CLASSIFIER_BG_MARGIN: Optional[float] = None
_AGENT_ACTIVE_SCORELESS_IOU: Optional[float] = None
_AGENT_ACTIVE_DETECTIONS: Optional[List[Dict[str, Any]]] = None
_AGENT_ACTIVE_CANDIDATES: Optional[List[Dict[str, Any]]] = None
_AGENT_ACTIVE_CANDIDATE_INDEX: Dict[int, Dict[str, Any]] = {}
_AGENT_ACTIVE_CLUSTERS: Optional[List[Dict[str, Any]]] = None
_AGENT_ACTIVE_CLUSTER_INDEX: Dict[int, Dict[str, Any]] = {}
_AGENT_ACTIVE_GRID: Optional[Dict[str, Any]] = None
_AGENT_ACTIVE_GRID_IMAGE: Optional[Image.Image] = None
_AGENT_ACTIVE_OVERLAY_IMAGE: Optional[Image.Image] = None
_AGENT_ACTIVE_LABEL_COLORS: Optional[Dict[str, str]] = None
_AGENT_ACTIVE_LABEL_PREFIXES: Optional[Dict[str, str]] = None
_AGENT_ACTIVE_OVERLAY_DOT_RADIUS: Optional[int] = None
_AGENT_GRID_TOOL_USAGE: Dict[str, Dict[str, int]] = {}
_AGENT_GRID_TOOL_LAST: Dict[str, Dict[str, Any]] = {}
_AGENT_HANDLE_INDEX: Dict[str, int] = {}
_AGENT_NEXT_CANDIDATE_ID = 1
_AGENT_NEXT_CLUSTER_ID = 1
_AGENT_LAST_SUBMIT_DETECTIONS: Optional[List[Dict[str, Any]]] = None
_AGENT_PENDING_CLASSIFY_IDS: List[int] = []
_AGENT_CLASSIFIER_CLASS_CACHE: Dict[str, List[str]] = {}
_AGENT_TRACE_FULL_WRITER: Optional[Callable[[Dict[str, Any]], None]] = None
_AGENT_TRACE_READABLE_WRITER: Optional[Callable[[str], None]] = None
_AGENT_TILE_CONTEXT_STORE: Dict[str, Dict[str, Any]] = {}
_AGENT_GLOBAL_CONTEXT_STORE: Dict[str, Dict[str, Any]] = {}
_AGENT_TILE_SUMMARIES: List[Dict[str, Any]] = []
_AGENT_PREPASS_COMPLETE: bool = False

PREPASS_TIGHT_DEFAULT_DETECTOR_CONF = 0.45
PREPASS_TIGHT_DEFAULT_SAM3_SCORE = 0.5
PREPASS_TIGHT_DEFAULT_SAM3_MASK = 0.5
PREPASS_TIGHT_DEFAULT_CLASSIFIER_MIN_PROB = 0.35
PREPASS_TIGHT_DEFAULT_CLASSIFIER_MARGIN = 0.05
PREPASS_TIGHT_DEFAULT_CLASSIFIER_BG_MARGIN = 0.05
PREPASS_TIGHT_DEFAULT_SCORELESS_IOU = 0.3
_DEFAULT_SAM3_SYNONYMS: Dict[str, List[str]] = {}
PREPASS_CLASSIFIER_STRICT_MIN_PROB = 0.65
PREPASS_CLASSIFIER_STRICT_MARGIN = 0.15
PREPASS_CLASSIFIER_STRICT_BG_MARGIN = 0.15
PREPASS_STRICT_SAM3_MIN_SCORE = 0.7
PREPASS_GRID_OVERLAP_RATIO = 0.2
PREPASS_CONTEXT_CHUNK_BYTES = 5 * 1024 * 1024
PREPASS_MIN_ZOOM_WINDOW_PX = 200
PREPASS_READABLE_TO_CONSOLE = str(os.environ.get("PREPASS_READABLE_TO_CONSOLE", "1")).lower() not in {"0", "false", "no"}
PREPASS_CLUSTER_IOU = 0.75
PREPASS_INSPECT_MAX_OBJECTS = 0


def _register_agent_tool(name: str):
    def _wrap(func):
        AGENT_TOOL_REGISTRY[name] = func
        return func
    return _wrap


def _agent_set_active_clusters(clusters: Optional[Sequence[Dict[str, Any]]]) -> None:
    global _AGENT_ACTIVE_CLUSTERS, _AGENT_ACTIVE_DETECTIONS, _AGENT_ACTIVE_CLUSTER_INDEX
    if not clusters:
        _AGENT_ACTIVE_CLUSTERS = []
        _AGENT_ACTIVE_DETECTIONS = []
        _AGENT_ACTIVE_CLUSTER_INDEX = {}
        _agent_refresh_handle_index()
        return
    cluster_list = [dict(item) for item in clusters if isinstance(item, dict)]
    _AGENT_ACTIVE_CLUSTERS = cluster_list
    _AGENT_ACTIVE_DETECTIONS = cluster_list
    _AGENT_ACTIVE_CLUSTER_INDEX = {
        int(item.get("cluster_id")): item for item in cluster_list if item.get("cluster_id") is not None
    }
    _agent_refresh_handle_index()


def _agent_reset_registries() -> None:
    global _AGENT_ACTIVE_CANDIDATES, _AGENT_ACTIVE_CANDIDATE_INDEX
    global _AGENT_ACTIVE_CLUSTERS, _AGENT_ACTIVE_CLUSTER_INDEX
    global _AGENT_NEXT_CANDIDATE_ID, _AGENT_NEXT_CLUSTER_ID
    global _AGENT_LAST_SUBMIT_DETECTIONS
    global _AGENT_PENDING_CLASSIFY_IDS
    global _AGENT_ACTIVE_LABEL_COLORS, _AGENT_ACTIVE_LABEL_PREFIXES, _AGENT_ACTIVE_OVERLAY_DOT_RADIUS
    global _AGENT_GRID_TOOL_USAGE, _AGENT_GRID_TOOL_LAST
    global _AGENT_TILE_CONTEXT_STORE, _AGENT_GLOBAL_CONTEXT_STORE
    global _AGENT_ACTIVE_OVERALL_CAPTION, _AGENT_ACTIVE_WINDOWED_CAPTIONS
    global _AGENT_TILE_SUMMARIES, _AGENT_HANDLE_INDEX, _AGENT_PREPASS_COMPLETE
    _AGENT_ACTIVE_CANDIDATES = []
    _AGENT_ACTIVE_CANDIDATE_INDEX = {}
    _AGENT_ACTIVE_CLUSTERS = []
    _AGENT_ACTIVE_CLUSTER_INDEX = {}
    _AGENT_NEXT_CANDIDATE_ID = 1
    _AGENT_NEXT_CLUSTER_ID = 1
    _AGENT_LAST_SUBMIT_DETECTIONS = None
    _AGENT_PENDING_CLASSIFY_IDS = []
    _AGENT_ACTIVE_LABEL_COLORS = None
    _AGENT_ACTIVE_LABEL_PREFIXES = None
    _AGENT_ACTIVE_OVERLAY_DOT_RADIUS = None
    _AGENT_GRID_TOOL_USAGE = {}
    _AGENT_GRID_TOOL_LAST = {}
    _AGENT_HANDLE_INDEX = {}
    _AGENT_TILE_CONTEXT_STORE = {}
    _AGENT_GLOBAL_CONTEXT_STORE = {}
    _AGENT_ACTIVE_OVERALL_CAPTION = None
    _AGENT_ACTIVE_WINDOWED_CAPTIONS = None
    _AGENT_TILE_SUMMARIES = []
    _AGENT_PREPASS_COMPLETE = False


def _agent_default_classifier_for_dataset(dataset_id: Optional[str]) -> Optional[str]:
    return _select_default_classifier(
        dataset_id,
        load_labelmap_fn=_agent_load_labelmap,
        classifier_matches_fn=_agent_classifier_matches_labelmap,
        root_dir=UPLOAD_ROOT / "classifiers",
    )


def _agent_classifier_classes_for_path(path: Path) -> List[str]:
    key = str(path.resolve())
    cached = _AGENT_CLASSIFIER_CLASS_CACHE.get(key)
    if cached is not None:
        return cached
    classes = _classifier_classes_for_path(path, load_model_fn=joblib.load)
    _AGENT_CLASSIFIER_CLASS_CACHE[key] = classes
    return classes


def _agent_classifier_matches_labelmap(path: Path, labelmap: Sequence[str]) -> bool:
    return _classifier_matches_labelmap(
        path,
        labelmap,
        load_model_fn=joblib.load,
        normalize_label_fn=_normalize_class_name_for_match,
        bg_indices_fn=_clip_head_background_indices,
    )


## NOTE: clip classifier path resolution uses *_impl directly to avoid wrapper drift.


## NOTE: clip head loader uses *_impl directly to avoid wrapper drift.




def _dispatch_agent_tool(call: AgentToolCall) -> AgentToolResult:
    handler = AGENT_TOOL_REGISTRY.get(call.name)
    if handler is None:
        _agent_full_trace_write(
            {
                "type": "tool_dispatch_error",
                "tool": call.name,
                "error": "tool_not_found",
                "ts": time.time(),
            }
        )
        return AgentToolResult(
            name=call.name,
            result=_agent_error_payload("tool_failed", "tool_not_found", "Use a supported tool name."),
        )
    try:
        args = dict(call.arguments or {})
        try:
            import inspect
            handler_params = inspect.signature(handler).parameters
        except Exception:
            handler_params = {}
        if ("image_token" in handler_params or "image_base64" in handler_params) and not args.get("image_token") and not args.get("image_base64"):
            if _AGENT_ACTIVE_IMAGE_TOKEN:
                args["image_token"] = _AGENT_ACTIVE_IMAGE_TOKEN
            elif _AGENT_ACTIVE_IMAGE_BASE64:
                args["image_base64"] = _AGENT_ACTIVE_IMAGE_BASE64
        grid_cell = args.get("grid_cell")
        if grid_cell is not None:
            grid_cell = str(grid_cell).strip().upper()
            if grid_cell:
                args["grid_cell"] = grid_cell
        grid_tools = {
            "look_and_inspect",
            "image_zoom_in_tool",
            "zoom_and_detect",
            "run_detector",
            "sam3_text",
            "sam3_similarity",
            "qwen_infer",
            "view_cell_raw",
            "view_cell_overlay",
        }
        inspect_window_key: Optional[Tuple[int, int, int, int]] = None
        if grid_cell and call.name in grid_tools:
            if not _AGENT_ACTIVE_GRID:
                return AgentToolResult(
                    name=call.name,
                    result=_agent_error_payload(
                        "missing_grid_cell",
                        "grid_unavailable",
                        "Enable grid overlay or provide window_bbox_2d.",
                    ),
                )
            cell_xyxy = _agent_grid_cell_xyxy(
                _AGENT_ACTIVE_GRID,
                str(grid_cell),
                overlap_ratio=PREPASS_GRID_OVERLAP_RATIO,
            )
            if not cell_xyxy:
                cols = int(_AGENT_ACTIVE_GRID.get("cols") or 0)
                rows = int(_AGENT_ACTIVE_GRID.get("rows") or 0)
                labels = _AGENT_ACTIVE_GRID.get("col_labels") or []
                if labels:
                    col_range = f"{labels[0]}-{labels[-1]}"
                else:
                    col_range = "unknown"
                return AgentToolResult(
                    name=call.name,
                    result=_agent_error_payload(
                        "missing_grid_cell",
                        f"grid_cell_invalid:valid_cols={col_range},rows=1-{rows or 0}",
                        "Provide a valid grid_cell (e.g., C2).",
                    ),
                )
            img_w = int(_AGENT_ACTIVE_GRID.get("img_w") or 0)
            img_h = int(_AGENT_ACTIVE_GRID.get("img_h") or 0)
            if img_w <= 0 or img_h <= 0:
                return AgentToolResult(
                    name=call.name,
                    result=_agent_error_payload(
                        "missing_grid_cell",
                        "grid_unavailable",
                        "Enable grid overlay or provide window_bbox_2d.",
                    ),
                )
            cell_bbox_2d = list(_xyxy_to_qwen_bbox(img_w, img_h, *cell_xyxy))
            if call.name in {"view_cell_raw", "view_cell_overlay"}:
                pass
            elif call.name == "image_zoom_in_tool":
                if not args.get("bbox_2d") and not args.get("bbox_xyxy_px") and not args.get("window_bbox_2d"):
                    args["bbox_2d"] = cell_bbox_2d
            else:
                if not args.get("window_bbox_2d"):
                    args["window_bbox_2d"] = cell_bbox_2d
                if call.name == "look_and_inspect" and not args.get("bbox_space"):
                    args["bbox_space"] = "window"
        if call.name == "run_detector":
            if args.get("conf") is None and _AGENT_ACTIVE_DETECTOR_CONF is not None:
                args["conf"] = _AGENT_ACTIVE_DETECTOR_CONF
        bbox_2d = args.get("bbox_2d")
        if isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) < 4:
            args.pop("bbox_2d", None)
        bbox_xyxy_px = args.get("bbox_xyxy_px")
        if isinstance(bbox_xyxy_px, (list, tuple)) and len(bbox_xyxy_px) < 4:
            args.pop("bbox_xyxy_px", None)
        window_bbox_2d = args.get("window_bbox_2d")
        if isinstance(window_bbox_2d, (list, tuple)) and len(window_bbox_2d) < 4:
            args.pop("window_bbox_2d", None)
        if call.name == "look_and_inspect":
            if _AGENT_ACTIVE_IMAGE_TOKEN and not args.get("image_base64"):
                args["image_token"] = _AGENT_ACTIVE_IMAGE_TOKEN
            if _AGENT_ACTIVE_LABELMAP:
                args["labelmap"] = list(_AGENT_ACTIVE_LABELMAP)
            if _AGENT_PREPASS_COMPLETE and "register" not in args:
                args["register"] = False
            if _AGENT_ACTIVE_GLOSSARY:
                args["labelmap_glossary"] = str(_AGENT_ACTIVE_GLOSSARY)
                model_id = str(args.get("model_id") or "").strip().lower()
                if model_id in {"qwen", "default", "auto"}:
                    args.pop("model_id", None)
                window_vals = args.get("window_bbox_2d") or args.get("bbox_2d")
                if isinstance(window_vals, (list, tuple)) and len(window_vals) >= 4:
                    try:
                        window_key = tuple(int(round(float(v))) for v in window_vals[:4])
                    except Exception:
                        window_key = None
                    if window_key and _AGENT_ACTIVE_INSPECTED_WINDOWS is not None:
                        if window_key in _AGENT_ACTIVE_INSPECTED_WINDOWS:
                            already_result = {
                                "candidates": [],
                                "window_bbox_2d": list(window_vals[:4]),
                                "already_inspected": True,
                            }
                            _agent_full_trace_write(
                                {
                                    "type": "tool_dispatch",
                                    "tool": call.name,
                                    "args": args,
                                    "ts": time.time(),
                                }
                            )
                            _agent_full_trace_write(
                                {
                                    "type": "tool_dispatch_result",
                                    "tool": call.name,
                                    "result": already_result,
                                    "ts": time.time(),
                                }
                            )
                            return AgentToolResult(
                                name=call.name,
                                result=already_result,
                            )
                        inspect_window_key = window_key
        if call.name == "qwen_infer":
            if not args.get("items") and _AGENT_ACTIVE_LABELMAP:
                args["items"] = list(_AGENT_ACTIVE_LABELMAP)
            if not args.get("extra_context") and _AGENT_ACTIVE_GLOSSARY:
                args["extra_context"] = str(_AGENT_ACTIVE_GLOSSARY)
        if call.name == "sam3_text":
            label = str(args.get("label") or "").strip()
            prompt = str(args.get("prompt") or "").strip()
            if not label and prompt and _AGENT_ACTIVE_LABELMAP:
                aligned = _agent_fuzzy_align_label(prompt, _AGENT_ACTIVE_LABELMAP)
                if aligned:
                    label = aligned
                    args["label"] = aligned
            if not label and not prompt and _AGENT_ACTIVE_LABELMAP:
                label = str(_AGENT_ACTIVE_LABELMAP[0]).strip()
                if label:
                    args["label"] = label
            if not prompt and label:
                synonym_map, _ = _agent_generate_sam3_synonyms(
                    _AGENT_ACTIVE_LABELMAP or [],
                    _AGENT_ACTIVE_GLOSSARY or "",
                    max_synonyms=10,
                    generate_text_fn=lambda prompt, max_new_tokens=128, use_system_prompt=True: _generate_qwen_text_impl(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        use_system_prompt=use_system_prompt,
                        system_prompt=(active_qwen_metadata or {}).get("system_prompt"),
                        ensure_qwen_ready_fn=_ensure_qwen_ready,
                        resolve_qwen_device_fn=lambda: _resolve_qwen_device_impl(QWEN_DEVICE_PREF, torch_module=torch),
                    ),
                    extract_json_fn=_extract_balanced_json,
                    default_synonyms=_DEFAULT_SAM3_SYNONYMS,
                    label_key_fn=_glossary_label_key,
                )
                prompts = _sam3_prompt_variants(
                    label,
                    synonym_map,
                    max_prompts=1,
                    default_synonyms=_DEFAULT_SAM3_SYNONYMS,
                    label_key_fn=_glossary_label_key,
                )
                if prompts:
                    args["prompt"] = prompts[0]
                else:
                    args["prompt"] = label
            if args.get("score_thr") is None and _AGENT_ACTIVE_SAM3_SCORE_THR is not None:
                args["score_thr"] = _AGENT_ACTIVE_SAM3_SCORE_THR
            if args.get("mask_threshold") is None and _AGENT_ACTIVE_SAM3_MASK_THR is not None:
                args["mask_threshold"] = _AGENT_ACTIVE_SAM3_MASK_THR
        if call.name == "sam3_similarity":
            if args.get("score_thr") is None and _AGENT_ACTIVE_SAM3_SCORE_THR is not None:
                args["score_thr"] = _AGENT_ACTIVE_SAM3_SCORE_THR
        if call.name == "classify_crop":
            classifier_id = args.get("classifier_id")
            if isinstance(classifier_id, str):
                classifier_norm = classifier_id.strip().lower()
                if classifier_norm in {"default", "auto", "best"}:
                    classifier_id = None
                    args.pop("classifier_id", None)
                    if _AGENT_ACTIVE_CLASSIFIER_ID and not isinstance(active_classifier_head, dict):
                        args["classifier_id"] = _AGENT_ACTIVE_CLASSIFIER_ID
                elif Path(classifier_id).suffix.lower() not in CLASSIFIER_ALLOWED_EXTS:
                    args.pop("classifier_id", None)
            if not args.get("classifier_id") and not isinstance(active_classifier_head, dict):
                if _AGENT_ACTIVE_CLASSIFIER_ID:
                    args["classifier_id"] = _AGENT_ACTIVE_CLASSIFIER_ID
                else:
                    dataset_id = args.get("dataset_id") or _AGENT_ACTIVE_DATASET_ID
                    fallback = _agent_default_classifier_for_dataset(dataset_id)
                    if fallback:
                        args["classifier_id"] = fallback
                    else:
                        skipped = _agent_error_payload(
                            "classifier_unavailable",
                            "classifier_unavailable",
                            "Classifier unavailable; skip verification.",
                        )
                        _agent_full_trace_write(
                            {
                                "type": "tool_dispatch",
                                "tool": call.name,
                                "args": args,
                                "ts": time.time(),
                            }
                        )
                        _agent_full_trace_write(
                            {
                                "type": "tool_dispatch_result",
                                "tool": call.name,
                                "result": skipped,
                                "ts": time.time(),
                            }
                        )
                        return AgentToolResult(
                            name=call.name,
                            result=skipped,
                        )
        if not args.get("dataset_id") and _AGENT_ACTIVE_DATASET_ID:
            try:
                import inspect
                sig = inspect.signature(handler)
                if "dataset_id" in sig.parameters:
                    args["dataset_id"] = _AGENT_ACTIVE_DATASET_ID
            except Exception:
                # Best-effort: only inject when we can confirm the signature accepts it.
                pass
        _agent_full_trace_write(
            {
                "type": "tool_dispatch",
                "tool": call.name,
                "args": args,
                "ts": time.time(),
            }
        )
        result = handler(**args)
        if call.name == "look_and_inspect" and inspect_window_key and _AGENT_ACTIVE_INSPECTED_WINDOWS is not None:
            if not (isinstance(result, dict) and result.get("error")):
                _AGENT_ACTIVE_INSPECTED_WINDOWS.add(inspect_window_key)
        _agent_record_grid_tool_usage(call.name, args, result)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        code, message, fix_hint = _agent_error_from_detail(detail, call.name)
        _agent_full_trace_write(
            {
                "type": "tool_dispatch_error",
                "tool": call.name,
                "error": message,
                "ts": time.time(),
            }
        )
        return AgentToolResult(
            name=call.name,
            result=_agent_error_payload(code, message, fix_hint),
        )
    except Exception as exc:  # noqa: BLE001
        if call.name == "look_and_inspect":
            return AgentToolResult(
                name=call.name,
                result={"skipped": True, "reason": f"inspect_failed:{exc}"},
            )
        _agent_full_trace_write(
            {
                "type": "tool_dispatch_error",
                "tool": call.name,
                "error": str(exc),
                "ts": time.time(),
            }
        )
        return AgentToolResult(
            name=call.name,
            result=_agent_error_payload("tool_failed", f"tool_failed:{exc}", "Check tool arguments and retry once."),
        )
    _agent_full_trace_write(
        {
            "type": "tool_dispatch_result",
            "tool": call.name,
            "result": result,
            "ts": time.time(),
        }
    )
    return AgentToolResult(name=call.name, result=result or {})


def _agent_load_labelmap(dataset_id: Optional[str]) -> List[str]:
    labelmap, _ = _agent_load_labelmap_meta(dataset_id)
    return labelmap


@_register_agent_tool("look_and_inspect")
def _agent_tool_look_and_inspect(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    bbox_2d: Optional[Sequence[float]] = None,
    bbox_xyxy_px: Optional[Sequence[float]] = None,
    bbox_space: Optional[str] = None,
    window_bbox_2d: Optional[Sequence[float]] = None,
    grid_cell: Optional[str] = None,
    intent: Optional[str] = None,
    labelmap: Optional[List[str]] = None,
    labelmap_glossary: Optional[str] = None,
    max_objects: Optional[int] = None,
    model_id: Optional[str] = None,
    register: Optional[bool] = True,
    include_caption: Optional[bool] = True,
) -> Dict[str, Any]:
    try:
        if not model_id or str(model_id).strip().lower() in {"default", "auto"}:
            model_id = (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME
        pil_img, _, _ = _agent_resolve_image(image_base64, image_token)
        img_w, img_h = pil_img.size
        ann: Dict[str, Any] = {"bbox_space": bbox_space or "full"}
        if intent:
            ann["intent"] = str(intent)
        if bbox_2d is not None:
            ann["bbox_2d"] = list(bbox_2d)
        if bbox_xyxy_px is not None:
            ann["bbox_xyxy_px"] = list(bbox_xyxy_px)
        window_xyxy = None
        if ann.get("bbox_2d") is not None or ann.get("bbox_xyxy_px") is not None:
            window_xyxy = _resolve_agent_bbox_xyxy(ann, img_w, img_h, window_bbox_2d=window_bbox_2d)
        if window_xyxy is None and window_bbox_2d is not None:
            window_xyxy = _normalize_window_xyxy({"bbox_2d": window_bbox_2d}, img_w, img_h)
        if window_xyxy is None:
            window_xyxy = (0.0, 0.0, float(img_w), float(img_h))
        x1, y1, x2, y2 = window_xyxy
        crop = pil_img.crop((x1, y1, x2, y2))
        labels = [str(x) for x in (labelmap or [])]
        if not labels and _AGENT_ACTIVE_LABELMAP:
            labels = [str(x) for x in _AGENT_ACTIVE_LABELMAP]
        glossary = _normalize_labelmap_glossary(labelmap_glossary)
        if not glossary and _AGENT_ACTIVE_GLOSSARY:
            glossary = _normalize_labelmap_glossary(_AGENT_ACTIVE_GLOSSARY)
        max_items = PREPASS_INSPECT_MAX_OBJECTS
        prompt_lines = [
            "Inspect this window and list ALL visible objects from the labelmap.",
            "Return ONLY a JSON array. Each item: {\"label\": <labelmap label>, \"bbox_2d\": [x1,y1,x2,y2]}",
            "bbox_2d uses 0-1000 coordinates RELATIVE TO THIS WINDOW (not the full image).",
            "Only include labels from the labelmap. If none, return [].",
        ]
        if labels:
            prompt_lines.append("Labelmap: " + ", ".join(labels))
        if glossary:
            prompt_lines.append("Glossary:\n" + glossary)
        if max_items > 0:
            prompt_lines.append(f"Max objects: {max_items}.")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a vision inspector. Output JSON only."}]},
            {"role": "user", "content": [{"type": "image", "image": crop}, {"type": "text", "text": "\n".join(prompt_lines)}]},
        ]
        response = _run_qwen_chat(
            messages,
            max_new_tokens=512,
            decode_override={"temperature": 0.2, "top_p": 0.9},
            model_id_override=model_id,
        )
        items = _agent_extract_json_array(response) or []
        results = []
        label_set = set(labels)
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if label and label_set and label not in label_set:
                continue
            bbox = item.get("bbox_2d")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            results.append(
                {
                    "label": label,
                    "bbox_2d": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "bbox_space": "window",
                    "window_bbox_2d": [float(v) for v in _xyxy_to_qwen_bbox(img_w, img_h, *window_xyxy)],
                    "score": None,
                    "score_source": "qwen_inspect",
                }
            )
            if max_items > 0 and len(results) >= max_items:
                break
        candidate_counts: Dict[str, int] = {}
        for item in results:
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            candidate_counts[label] = candidate_counts.get(label, 0) + 1
        caption_text = ""
        if include_caption:
            try:
                caption_prompt = (
                    "Describe this window in 1 short sentence. "
                    "Mention visible objects using common terms, no coordinates."
                )
                caption_messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a visual captioner."}]},
                    {"role": "user", "content": [{"type": "image", "image": crop}, {"type": "text", "text": caption_prompt}]},
                ]
                caption_raw = _run_qwen_chat(
                    caption_messages,
                    max_new_tokens=120,
                    decode_override={"temperature": 0.3, "top_p": 0.9},
                    model_id_override=model_id,
                )
                caption_text = _sanitize_qwen_caption_impl(caption_raw)
            except Exception:
                caption_text = ""
        register_summary: Optional[Dict[str, Any]] = None
        if register:
            owner_cell = grid_cell or _agent_grid_cell_for_window_bbox(
                _AGENT_ACTIVE_GRID or {},
                _xyxy_to_qwen_bbox(img_w, img_h, *window_xyxy),
            )
            register_summary = _agent_register_detections(
                results,
                img_w=img_w,
                img_h=img_h,
                grid=_AGENT_ACTIVE_GRID,
                labelmap=labelmap,
                background=None,
                source_override="qwen_inspect",
                owner_cell=owner_cell,
            )
        new_cluster_ids = register_summary.get("new_cluster_ids") if isinstance(register_summary, dict) else []
        updated_cluster_ids = register_summary.get("updated_cluster_ids") if isinstance(register_summary, dict) else []
        new_summary = _agent_cluster_summaries(new_cluster_ids, include_ids=False)
        new_handles = _agent_handles_from_cluster_ids(new_cluster_ids or [])
        updated_handles = _agent_handles_from_cluster_ids(updated_cluster_ids or [])
        agent_view = {
            "grid_cell": grid_cell or _agent_grid_cell_for_window_bbox(_AGENT_ACTIVE_GRID or {}, _xyxy_to_qwen_bbox(img_w, img_h, *window_xyxy)),
            "caption": caption_text or None,
            "new_clusters": register_summary.get("new_clusters") if isinstance(register_summary, dict) else 0,
            "new_handles": new_handles,
            "updated_clusters": len(updated_cluster_ids or []),
            "updated_handles": updated_handles,
            "new_items": new_summary.get("items"),
            "new_items_total": new_summary.get("total"),
            "new_items_truncated": new_summary.get("truncated"),
            "label_counts": _agent_cluster_label_counts(new_cluster_ids or []),
            "candidate_count": len(results),
            "candidate_label_counts": candidate_counts,
        }
        return {
            "candidates": results,
            "window_xyxy_px": list(window_xyxy),
            "caption": caption_text or None,
            "register_summary": register_summary,
            "__agent_view__": agent_view,
        }
    except Exception as exc:  # noqa: BLE001
        reason = f"inspect_failed:{exc}"
        grid_text = None
        try:
            grid_text = grid_cell or _agent_grid_cell_for_window_bbox(
                _AGENT_ACTIVE_GRID or {}, _xyxy_to_qwen_bbox(img_w, img_h, *window_xyxy)  # type: ignore[arg-type]
            )
        except Exception:
            grid_text = grid_cell
        agent_view = {
            "grid_cell": grid_text,
            "caption": None,
            "new_clusters": 0,
            "new_handles": [],
            "updated_clusters": 0,
            "updated_handles": [],
            "new_items": [],
            "new_items_total": 0,
            "new_items_truncated": False,
            "label_counts": {},
            "candidate_count": 0,
            "skipped": True,
            "reason": reason,
        }
        return {
            "candidates": [],
            "window_xyxy_px": list(window_xyxy) if "window_xyxy" in locals() else None,
            "caption": None,
            "skipped": True,
            "reason": reason,
            "__agent_view__": agent_view,
        }


def _agent_load_labelmap_meta(dataset_id: Optional[str]) -> Tuple[List[str], str]:
    return _load_labelmap_meta(
        dataset_id,
        active_label_list=active_label_list,
        resolve_dataset=_resolve_sam3_or_qwen_dataset,
        discover_yolo_labelmap=lambda dataset_root: _discover_yolo_labelmap_impl(
            dataset_root, load_labelmap_file_fn=_load_labelmap_file
        ),
        load_qwen_labelmap=_load_qwen_labelmap,
        load_sam3_meta=lambda dataset_dir: _load_sam3_dataset_metadata_impl(
            dataset_dir,
            meta_name=SAM3_DATASET_META_NAME,
            load_json_metadata_fn=_load_json_metadata,
            persist_metadata_fn=lambda dataset_dir_inner, metadata: _persist_sam3_dataset_metadata_impl(
                dataset_dir_inner,
                metadata,
                meta_name=SAM3_DATASET_META_NAME,
                logger=logger,
            ),
        ),
        load_qwen_meta=lambda dataset_dir: _load_qwen_dataset_metadata_impl(
            dataset_dir,
            meta_name=QWEN_METADATA_FILENAME,
            load_json_metadata_fn=_load_json_metadata,
        ),
        normalize_glossary=_normalize_labelmap_glossary,
        default_glossary_fn=_default_agent_glossary_for_labelmap,
        collect_labels=_collect_labels_from_qwen_jsonl_impl,
    )


def _calibration_require_sam3(enable_text: bool, enable_similarity: bool) -> None:
    _require_sam3_for_prepass_impl(
        enable_text,
        enable_similarity,
        sam3_import_error=SAM3_NATIVE_IMAGE_IMPORT_ERROR,
        build_sam3_image_model=build_sam3_image_model,
        sam3_image_processor=Sam3ImageProcessor,
    )


def _calibration_unload_non_qwen() -> None:
    _unload_non_qwen_runtimes_impl(
        predictor_manager=predictor_manager,
        unload_sam3_text_fn=_unload_sam3_text_runtime,
        suspend_clip_fn=_suspend_clip_backbone,
        unload_dinov3_fn=_unload_dinov3_backbone,
        unload_detector_fn=_unload_detector_inference,
        torch_module=torch,
        logger=logger,
    )


def _calibration_unload_inference_runtimes() -> None:
    _unload_inference_runtimes_impl(
        unload_non_qwen_fn=_calibration_unload_non_qwen,
        unload_qwen_fn=_unload_qwen_runtime,
        torch_module=torch,
    )


def _calibration_prepare_for_training() -> None:
    _prepare_for_training_impl(
        unload_inference_runtimes_fn=_calibration_unload_inference_runtimes,
    )


def _calibration_load_yolo_active() -> dict:
    return _load_yolo_active_impl(YOLO_ACTIVE_PATH)


def _calibration_list_images(dataset_id: str) -> List[Path]:
    return _calibration_list_images_impl(
        dataset_id, resolve_dataset_fn=_resolve_sam3_or_qwen_dataset
    )


def _calibration_cache_image(pil_img: Image.Image, sam_variant: Optional[str]) -> Optional[str]:
    return _calibration_cache_image_impl(
        pil_img,
        sam_variant,
        store_preloaded_fn=_store_preloaded_image,
        default_variant_fn=_default_variant,
    )


def _calibration_unload_inference() -> None:
    _calibration_unload_inference_runtimes()


def _calibration_run_job(job: CalibrationJob, request: CalibrationRequest) -> None:
    _run_calibration_job_impl(
        job,
        request,
        jobs=CALIBRATION_JOBS,
        jobs_lock=CALIBRATION_JOBS_LOCK,
        update_fn=_calibration_update,
        require_sam3_fn=_calibration_require_sam3,
        prepare_for_training_fn=_calibration_prepare_for_training,
        load_yolo_active_fn=_calibration_load_yolo_active,
        load_rfdetr_active_fn=_load_rfdetr_active,
        load_labelmap_meta_fn=_agent_load_labelmap_meta,
        list_images_fn=_calibration_list_images,
        sample_images_fn=_calibration_sample_images_impl,
        calibration_root=CALIBRATION_ROOT,
        calibration_cache_root=CALIBRATION_CACHE_ROOT,
        prepass_request_cls=QwenPrepassRequest,
        active_classifier_head=active_classifier_head,
        active_classifier_path=active_classifier_path,
        default_classifier_for_dataset_fn=_agent_default_classifier_for_dataset,
        calibration_features_version=CALIBRATION_FEATURES_VERSION,
        write_record_fn=_calibration_write_record_atomic,
        hash_payload_fn=_calibration_hash_payload_impl,
        safe_link_fn=_calibration_safe_link_impl,
        prepass_worker_fn=_calibration_prepass_worker,
        unload_inference_runtimes_fn=_calibration_unload_inference,
        resolve_dataset_fn=_resolve_sam3_or_qwen_dataset,
        cache_image_fn=_calibration_cache_image,
        run_prepass_fn=_agent_run_deep_prepass,
        logger=logger,
        http_exception_cls=HTTPException,
        root_dir=Path(__file__).resolve().parent,
    )


def _encode_pil_batch_for_head(
    images: Sequence[Image.Image],
    *,
    head: Dict[str, Any],
    batch_size_override: Optional[int] = None,
    device_override: Optional[str] = None,
) -> Optional[np.ndarray]:
    if not images:
        return None
    encoder_type = str(head.get("encoder_type") or "clip").strip().lower()
    batch_size = int(batch_size_override or 0) or len(images)
    batch_size = max(1, min(batch_size, len(images)))
    features: List[np.ndarray] = []
    if encoder_type == "dinov3":
        if dinov3_model is None or dinov3_processor is None:
            return None
        device_name = device_override or dinov3_model_device or device
        for idx in range(0, len(images), batch_size):
            batch = images[idx : idx + batch_size]
            inputs = dinov3_processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device_name) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = dinov3_model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                feats = outputs.last_hidden_state.mean(dim=1)
            else:
                feats = outputs[0]
            feats_np = feats.detach().float().cpu().numpy()
            features.append(feats_np)
    else:
        if clip_model is None or clip_preprocess is None:
            return None
        device_name = device_override or device
        for idx in range(0, len(images), batch_size):
            batch = images[idx : idx + batch_size]
            batch_tensor = torch.stack([clip_preprocess(img) for img in batch]).to(device_name)
            with torch.no_grad():
                feats = clip_model.encode_image(batch_tensor).float()
            feats_np = feats.detach().cpu().numpy()
            features.append(feats_np)
    if not features:
        return None
    feats_np = np.concatenate(features, axis=0)
    center_vals = head.get("embedding_center_values")
    std_vals = head.get("embedding_std_values")
    if center_vals is not None:
        center_arr = np.asarray(center_vals, dtype=np.float32).reshape(1, -1)
        if center_arr.shape[1] == feats_np.shape[1]:
            feats_np = feats_np - center_arr
    if std_vals is not None:
        std_arr = np.asarray(std_vals, dtype=np.float32).reshape(1, -1)
        if std_arr.shape[1] == feats_np.shape[1]:
            std_arr = np.where(std_arr == 0, 1.0, std_arr)
            feats_np = feats_np / std_arr
    if head.get("normalize_embeddings"):
        denom = np.linalg.norm(feats_np, axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        feats_np = feats_np / denom
    return feats_np.astype(np.float32, copy=False)


def _encode_pil_batch_for_active(images: Sequence[Image.Image]) -> Optional[np.ndarray]:
    if isinstance(active_classifier_head, dict):
        head = active_classifier_head
    else:
        head = {
            "encoder_type": active_encoder_type or "clip",
            "normalize_embeddings": active_head_normalize_embeddings,
            "embedding_center_values": (active_classifier_meta or {}).get("embedding_center_values"),
            "embedding_std_values": (active_classifier_meta or {}).get("embedding_std_values"),
        }
    return _encode_pil_batch_for_head(images, head=head)


def _clip_head_predict_proba(
    feats: Any,
    head: Dict[str, Any],
    *,
    empty_cache_fn: Optional[Callable[[], None]] = None,
) -> Optional[np.ndarray]:
    if feats is None:
        return None
    feats_np = feats.detach().cpu().numpy() if isinstance(feats, torch.Tensor) else np.asarray(feats)
    if feats_np.ndim != 2:
        return None
    classifier_type = str(head.get("classifier_type") or "").lower()
    if classifier_type == "mlp":
        layers = head.get("layers") or []
        if not isinstance(layers, list) or not layers:
            return None
        x = feats_np.astype(np.float32, copy=False)
        for layer in layers:
            weight = np.asarray(layer.get("weight"), dtype=np.float32)
            bias = np.asarray(layer.get("bias"), dtype=np.float32)
            x = x @ weight.T + bias
            ln_weight = layer.get("layer_norm_weight")
            if ln_weight is not None:
                ln_weight = np.asarray(ln_weight, dtype=np.float32)
                ln_bias = np.asarray(layer.get("layer_norm_bias"), dtype=np.float32) if layer.get("layer_norm_bias") is not None else None
                eps = float(layer.get("layer_norm_eps") or 1e-5)
                mean = x.mean(axis=-1, keepdims=True)
                var = x.var(axis=-1, keepdims=True)
                x = (x - mean) / np.sqrt(var + eps)
                x = x * ln_weight
                if ln_bias is not None:
                    x = x + ln_bias
            activation = str(layer.get("activation") or "").lower()
            if activation in {"relu"}:
                x = np.maximum(0.0, x)
        logits = x
    else:
        coef = np.asarray(head.get("coef"), dtype=np.float32)
        intercept = np.asarray(head.get("intercept"), dtype=np.float32)
        if coef.ndim != 2:
            return None
        if intercept.ndim == 1:
            intercept = intercept.reshape(1, -1)
        logits = feats_np @ coef.T + intercept
    temp = head.get("temperature")
    if temp:
        try:
            logits = logits / float(temp)
        except Exception:
            pass
    if head.get("logit_adjustment_inference"):
        adjust = head.get("logit_adjustment")
        if adjust is not None:
            adj = np.asarray(adjust, dtype=np.float32).reshape(1, -1)
            if adj.shape[1] == logits.shape[1]:
                logits = logits + adj
    proba_mode = str(head.get("proba_mode") or "softmax").lower()
    if proba_mode == "binary":
        if logits.shape[1] != 1:
            logits = logits[:, :1]
        pos = 1.0 / (1.0 + np.exp(-logits))
        probs = np.concatenate([1.0 - pos, pos], axis=1)
    elif proba_mode == "ovr":
        probs = 1.0 / (1.0 + np.exp(-logits))
    else:
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    if empty_cache_fn is not None and torch.cuda.is_available():
        try:
            empty_cache_fn()
        except Exception:
            pass
    return probs.astype(np.float32, copy=False)


def _clip_auto_predict_details(feats_np: Any, *, background_guard: bool = True) -> Dict[str, Any]:
    if not isinstance(active_classifier_head, dict):
        return {"error": "classifier_head_unavailable"}
    head = active_classifier_head
    proba_arr = _clip_head_predict_proba(feats_np, head)
    if proba_arr is None or proba_arr.ndim != 2 or proba_arr.shape[0] == 0:
        return {"error": "classifier_failed"}
    classes = [str(c) for c in list(head.get("classes") or [])]
    if not classes:
        return {"error": "classifier_no_classes"}
    row = proba_arr[0]
    order = sorted(range(len(classes)), key=lambda idx: float(row[idx]), reverse=True)
    bg_indices = set(_clip_head_background_indices(classes))
    best_idx = order[0] if order else None
    if background_guard and best_idx is not None and best_idx in bg_indices:
        for idx in order:
            if idx not in bg_indices:
                best_idx = idx
                break
    best_label = classes[best_idx] if best_idx is not None else None
    best_prob = float(row[best_idx]) if best_idx is not None else None
    second_idx = None
    for idx in order[1:]:
        if best_idx is None or idx != best_idx:
            second_idx = idx
            break
    second_label = classes[second_idx] if second_idx is not None else None
    second_prob = float(row[second_idx]) if second_idx is not None else None
    margin = None
    if best_prob is not None and second_prob is not None:
        margin = float(best_prob - second_prob)
    return {
        "label": best_label,
        "proba": best_prob,
        "second_label": second_label,
        "second_proba": second_prob,
        "margin": margin,
        "error": None,
    }


def _clip_head_keep_mask(
    proba_arr: np.ndarray,
    *,
    target_index: int,
    min_prob: float,
    margin: float,
    background_indices: Optional[Sequence[int]] = None,
    background_guard: bool = True,
    background_margin: float = 0.0,
) -> Optional[np.ndarray]:
    if proba_arr is None or proba_arr.ndim != 2:
        return None
    num_classes = proba_arr.shape[1]
    if target_index < 0 or target_index >= num_classes:
        return None
    target = proba_arr[:, target_index]
    if num_classes > 1:
        mask = np.ones(num_classes, dtype=bool)
        mask[target_index] = False
        best_other = np.max(proba_arr[:, mask], axis=1)
    else:
        best_other = np.zeros_like(target)
    keep = (target >= float(min_prob)) & ((target - best_other) >= float(margin))
    if background_guard and background_indices:
        bg = np.max(proba_arr[:, list(background_indices)], axis=1)
        keep &= (target - bg) >= float(background_margin)
    return keep


def _save_clip_head_artifacts(
    *,
    recipe_dir: Path,
    head: Dict[str, Any],
    min_prob: float,
    margin: float,
) -> None:
    clip_dir = recipe_dir / "clip_head"
    clip_dir.mkdir(parents=True, exist_ok=True)
    head_path = clip_dir / "head.npz"
    np.savez(head_path, head=np.array([head], dtype=object))
    meta = {
        "min_prob": float(min_prob),
        "margin": float(margin),
        "clip_model": head.get("clip_model"),
        "encoder_type": head.get("encoder_type"),
        "encoder_model": head.get("encoder_model"),
        "proba_mode": head.get("proba_mode"),
        "background_margin": head.get("background_margin"),
    }
    (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True))


def _load_clip_head_artifacts(
    *,
    recipe_dir: Path,
    fallback_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    clip_dir = recipe_dir / "clip_head"
    head_path = clip_dir / "head.npz"
    if not head_path.exists():
        return None
    try:
        data = np.load(head_path, allow_pickle=True)
        head_arr = data.get("head")
        head = head_arr.item() if head_arr is not None else None
    except Exception:
        head = None
    if not isinstance(head, dict):
        return None
    meta_path = clip_dir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    if fallback_meta and not meta:
        meta = dict(fallback_meta)
    head["min_prob"] = float(meta.get("min_prob") or head.get("min_prob") or 0.5)
    head["margin"] = float(meta.get("margin") or head.get("margin") or 0.0)
    if meta.get("background_margin") is not None:
        head["background_margin"] = meta.get("background_margin")
    if meta.get("clip_model"):
        head["clip_model"] = meta.get("clip_model")
    if meta.get("encoder_type"):
        head["encoder_type"] = meta.get("encoder_type")
    if meta.get("encoder_model"):
        head["encoder_model"] = meta.get("encoder_model")
    if meta.get("proba_mode"):
        head["proba_mode"] = meta.get("proba_mode")
    return head


def _score_detections_with_clip_head(
    detections: Sequence[Any],
    *,
    pil_img: Image.Image,
    clip_head: Dict[str, Any],
    score_mode: str,
) -> Optional[Dict[int, float]]:
    if not detections or pil_img is None or not isinstance(clip_head, dict):
        return None
    img_w, img_h = pil_img.size
    crops: List[Image.Image] = []
    det_refs: List[Any] = []
    target_indices: List[Optional[int]] = []
    classes = [str(c) for c in list(clip_head.get("classes") or [])]
    for det in detections:
        label = None
        if isinstance(det, dict):
            label = det.get("label") or det.get("class_name")
        else:
            label = getattr(det, "label", None) or getattr(det, "class_name", None)
        target_idx = _find_clip_head_target_index(classes, label)
        if target_idx is None:
            target_indices.append(None)
            continue
        ann = det if isinstance(det, dict) else det.__dict__
        xyxy = _resolve_agent_bbox_xyxy(ann, img_w, img_h, window_bbox_2d=ann.get("window_bbox_2d"))
        if xyxy is None:
            target_indices.append(None)
            continue
        x1, y1, x2, y2 = xyxy
        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))
        if x2 <= x1 or y2 <= y1:
            target_indices.append(None)
            continue
        crops.append(pil_img.crop((x1, y1, x2, y2)))
        det_refs.append(det)
        target_indices.append(target_idx)
    if not crops:
        return None
    feats = _encode_pil_batch_for_head(crops, head=clip_head)
    if feats is None:
        return None
    proba = _clip_head_predict_proba(feats, clip_head)
    if proba is None or proba.ndim != 2:
        return None
    scores: Dict[int, float] = {}
    for idx, det in enumerate(det_refs):
        target_idx = target_indices[idx]
        if target_idx is None:
            continue
        row = proba[idx]
        target_prob = float(row[target_idx])
        if score_mode == "clip_head_margin":
            if len(row) > 1:
                best_other = float(np.max(np.delete(row, target_idx)))
            else:
                best_other = 0.0
            score_val = target_prob - best_other
        else:
            score_val = target_prob
        scores[id(det)] = score_val
    return scores


def _agent_classifier_review(
    detections: List[Dict[str, Any]],
    *,
    pil_img: Optional[Image.Image],
    classifier_head: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    counts = {"classifier_checked": 0, "classifier_rejected": 0}
    if not detections or pil_img is None or not isinstance(classifier_head, dict):
        return detections, counts
    classes = [str(c) for c in list(classifier_head.get("classes") or [])]
    if not classes:
        return detections, counts
    img_w, img_h = pil_img.size
    bg_indices = _clip_head_background_indices(classes)
    min_prob = float(classifier_head.get("min_prob") or 0.5)
    margin = float(classifier_head.get("margin") or 0.0)
    background_margin = float(classifier_head.get("background_margin") or 0.0)
    pending: List[Dict[str, Any]] = []
    pending_crops: List[Image.Image] = []
    reviewed: List[Dict[str, Any]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        counts["classifier_checked"] += 1
        label = str(det.get("label") or det.get("class_name") or "").strip()
        target_idx = _find_clip_head_target_index(classes, label)
        if target_idx is None:
            det["classifier_accept"] = False
            counts["classifier_rejected"] += 1
            continue
        xyxy = det.get("bbox_xyxy_px")
        if not xyxy:
            xyxy = _resolve_agent_bbox_xyxy(det, img_w, img_h, window_bbox_2d=det.get("window_bbox_2d"))
        if not xyxy or len(xyxy) < 4:
            det["classifier_accept"] = False
            counts["classifier_rejected"] += 1
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
        if x2 <= x1 or y2 <= y1:
            det["classifier_accept"] = False
            counts["classifier_rejected"] += 1
            continue
        pending.append({"entry": det, "target_idx": target_idx})
        pending_crops.append(pil_img.crop((x1, y1, x2, y2)))
    if not pending:
        return reviewed, counts
    empty_cache_fn = torch.cuda.empty_cache if torch.cuda.is_available() else None
    proba_arr = _predict_proba_batched_impl(
        pending_crops,
        classifier_head,
        batch_size=_resolve_classifier_batch(),
        encode_batch_fn=lambda items, head_obj, bs: _encode_pil_batch_for_head(
            items, head=head_obj, batch_size_override=bs
        ),
        predict_proba_fn=_clip_head_predict_proba,
        empty_cache_fn=empty_cache_fn,
    )
    if proba_arr is None or proba_arr.ndim != 2 or proba_arr.shape[0] != len(pending):
        for pending_entry in pending:
            pending_entry["entry"]["classifier_accept"] = False
        counts["classifier_rejected"] += len(pending)
        return reviewed, counts
    for row, pending_entry in zip(proba_arr, pending):
        entry = pending_entry["entry"]
        target_idx = pending_entry["target_idx"]
        keep_mask = _clip_head_keep_mask(
            row.reshape(1, -1),
            target_index=target_idx,
            min_prob=min_prob,
            margin=margin,
            background_indices=bg_indices,
            background_guard=True,
            background_margin=background_margin,
        )
        accept = bool(keep_mask[0]) if keep_mask is not None and len(keep_mask) else False
        entry["classifier_accept"] = accept
        if not accept:
            counts["classifier_rejected"] += 1
            continue
        reviewed.append(entry)
    return reviewed, counts


def _agent_current_label_colors(labels: Sequence[str]) -> Dict[str, str]:
    global _AGENT_ACTIVE_LABEL_COLORS
    if _AGENT_ACTIVE_LABEL_COLORS:
        if all(lbl in _AGENT_ACTIVE_LABEL_COLORS for lbl in labels):
            return _AGENT_ACTIVE_LABEL_COLORS
    label_colors = _agent_label_color_map(labels)
    _AGENT_ACTIVE_LABEL_COLORS = label_colors
    return label_colors


def _agent_current_label_prefixes(labels: Sequence[str]) -> Dict[str, str]:
    global _AGENT_ACTIVE_LABEL_PREFIXES
    if _AGENT_ACTIVE_LABEL_PREFIXES:
        if all(lbl in _AGENT_ACTIVE_LABEL_PREFIXES for lbl in labels):
            return _AGENT_ACTIVE_LABEL_PREFIXES
    label_prefixes = _agent_label_prefix_map(labels)
    _AGENT_ACTIVE_LABEL_PREFIXES = label_prefixes
    return label_prefixes


def _compute_steps_seed_eval_threshold(payload: "AgentMiningRequest") -> float:
    base = float(payload.seed_threshold)
    floor = payload.steps_seed_eval_floor
    if floor is None:
        return base
    try:
        floor_val = float(floor)
    except Exception:
        return base
    return min(base, floor_val)


def _compute_steps_seed_eval_max_results(payload: "AgentMiningRequest") -> int:
    override = payload.steps_seed_eval_max_results
    if override is None:
        return int(payload.max_results)
    try:
        return int(override)
    except Exception:
        return int(payload.max_results)


def _agent_refresh_handle_index() -> None:
    global _AGENT_HANDLE_INDEX
    _AGENT_HANDLE_INDEX = _refresh_handle_index(
        list(_AGENT_ACTIVE_CLUSTERS or []),
        handle_fn=_agent_cluster_handle,
    )


def _agent_cluster_handle(cluster: Mapping[str, Any]) -> Optional[str]:
    return _cluster_handle(
        cluster,
        label_prefixes=_AGENT_ACTIVE_LABEL_PREFIXES,
        labelmap=_AGENT_ACTIVE_LABELMAP or [],
        label_prefix_map_fn=_agent_current_label_prefixes,
    )


def _agent_cluster_id_from_handle(handle: Optional[str]) -> Optional[int]:
    cid = _cluster_id_from_handle(
        handle,
        handle_index=_AGENT_HANDLE_INDEX,
        cluster_index=_AGENT_ACTIVE_CLUSTER_INDEX,
    )
    if cid is not None:
        return cid
    text = str(handle or "").strip()
    if not text:
        return None
    match = re.search(r"(\\d+)$", text)
    if match:
        cid = int(match.group(1))
        cluster = _AGENT_ACTIVE_CLUSTER_INDEX.get(cid)
        if cluster and _agent_cluster_handle(cluster) == text:
            return cid
    return None


def _agent_handles_from_cluster_ids(cluster_ids: Sequence[int]) -> List[str]:
    return _handles_from_cluster_ids(
        cluster_ids,
        cluster_index=_AGENT_ACTIVE_CLUSTER_INDEX,
        handle_fn=_agent_cluster_handle,
    )


def _agent_cluster_ids_from_handles(handles: Sequence[str]) -> List[int]:
    return _cluster_ids_from_handles(
        handles,
        handle_index=_AGENT_HANDLE_INDEX,
        cluster_index=_AGENT_ACTIVE_CLUSTER_INDEX,
    )


def _agent_register_detections(
    detections: Sequence[Dict[str, Any]],
    *,
    img_w: int,
    img_h: int,
    grid: Optional[Mapping[str, Any]] = None,
    labelmap: Optional[Sequence[str]] = None,
    background: Optional[Sequence[str]] = None,
    source_override: Optional[str] = None,
    owner_cell: Optional[str] = None,
    iou_thr: Optional[float] = None,
) -> Dict[str, Any]:
    global _AGENT_ACTIVE_CANDIDATES, _AGENT_ACTIVE_CANDIDATE_INDEX
    global _AGENT_ACTIVE_CLUSTERS, _AGENT_ACTIVE_CLUSTER_INDEX
    global _AGENT_NEXT_CANDIDATE_ID, _AGENT_NEXT_CLUSTER_ID

    if not detections:
        return {"candidate_ids": [], "cluster_ids": [], "new_clusters": 0, "updated_clusters": 0, "rejected": 0}
    labelmap = list(labelmap or _AGENT_ACTIVE_LABELMAP or [])
    cleaned, rejected = _agent_sanitize_detection_items(
        list(detections),
        pil_img=None,
        classifier_head=None,
        img_w=img_w,
        img_h=img_h,
        labelmap=labelmap,
        background=background,
    )
    if _AGENT_ACTIVE_CANDIDATES is None:
        _AGENT_ACTIVE_CANDIDATES = []
    if _AGENT_ACTIVE_CLUSTERS is None:
        _AGENT_ACTIVE_CLUSTERS = []
    if iou_thr is None:
        iou_thr = PREPASS_CLUSTER_IOU
    new_cluster_ids: List[int] = []
    updated_cluster_ids: List[int] = []
    candidate_ids: List[int] = []
    for det in cleaned:
        candidate_id = _AGENT_NEXT_CANDIDATE_ID
        _AGENT_NEXT_CANDIDATE_ID += 1
        candidate_ids.append(candidate_id)
        source = source_override or det.get("source") or det.get("score_source") or "agent"
        source_primary = det.get("source_primary") or source
        source_prompt = det.get("source_prompt")
        source_exemplar_handles = det.get("source_exemplar_handles")
        source_detector_run_id = det.get("source_detector_run_id")
        source_list = set(det.get("source_list") or [])
        if source:
            source_list.add(str(source))
        cell = _agent_grid_cell_for_detection(det, img_w, img_h, grid)
        owner = owner_cell or cell
        candidate = {
            "candidate_id": candidate_id,
            "label": det.get("label"),
            "class_id": det.get("class_id"),
            "bbox_xyxy_px": det.get("bbox_xyxy_px"),
            "bbox_2d": det.get("bbox_2d"),
            "bbox_yolo": det.get("bbox_yolo"),
            "score": det.get("score"),
            "score_source": det.get("score_source") or det.get("source"),
            "source": source,
            "source_primary": source_primary,
            "source_prompt": source_prompt,
            "source_exemplar_handles": source_exemplar_handles,
            "source_detector_run_id": source_detector_run_id,
            "source_list": sorted(source_list) if source_list else None,
            "grid_cell": cell,
            "owner_cell": owner,
            "window_bbox_2d": det.get("window_bbox_2d"),
        }
        _AGENT_ACTIVE_CANDIDATES.append(candidate)
        _AGENT_ACTIVE_CANDIDATE_INDEX[candidate_id] = candidate
        cluster = _agent_cluster_match(det, _AGENT_ACTIVE_CLUSTERS, iou_thr=iou_thr)
        if cluster is None:
            cluster_id = _AGENT_NEXT_CLUSTER_ID
            _AGENT_NEXT_CLUSTER_ID += 1
            origin_tag = "prepass"
            cluster_entry = {
                "cluster_id": cluster_id,
                "label": det.get("label"),
                "class_id": det.get("class_id"),
                "bbox_xyxy_px": det.get("bbox_xyxy_px"),
                "bbox_2d": det.get("bbox_2d"),
                "bbox_yolo": det.get("bbox_yolo"),
                "score": det.get("score"),
                "score_source": det.get("score_source") or det.get("source"),
                "source": source,
                "source_primary": source_primary,
                "source_prompt": source_prompt,
                "source_exemplar_handles": source_exemplar_handles,
                "source_detector_run_id": source_detector_run_id,
                "source_list": sorted(source_list) if source_list else None,
                "origin": origin_tag,
                "candidate_ids": [candidate_id],
                "grid_cell": cell,
                "owner_cell": owner,
                "classifier_best": det.get("classifier_best"),
                "classifier_prob": det.get("classifier_prob"),
                "classifier_accept": det.get("classifier_accept"),
            }
            _AGENT_ACTIVE_CLUSTERS.append(cluster_entry)
            _AGENT_ACTIVE_CLUSTER_INDEX[cluster_id] = cluster_entry
            new_cluster_ids.append(cluster_id)
            candidate["cluster_id"] = cluster_id
            continue
        cluster_id = int(cluster.get("cluster_id"))
        candidate["cluster_id"] = cluster_id
        if owner and not cluster.get("owner_cell"):
            cluster["owner_cell"] = owner
        cluster.setdefault("candidate_ids", [])
        cluster["candidate_ids"] = list(cluster.get("candidate_ids") or []) + [candidate_id]
        cluster_sources = set(cluster.get("source_list") or [])
        if cluster.get("source"):
            cluster_sources.add(str(cluster.get("source")))
        cluster_sources.update(source_list)
        if cluster_sources:
            cluster["source_list"] = sorted(cluster_sources)
        cluster_score = _agent_det_score(cluster)
        det_score = _agent_det_score(det)
        replace = False
        if det_score is not None and (cluster_score is None or det_score > cluster_score):
            replace = True
        elif det_score is None and cluster_score is None and not cluster.get("bbox_xyxy_px"):
            replace = True
        if replace:
            keep_ids = list(cluster.get("candidate_ids") or [])
            keep_sources = list(cluster.get("source_list") or [])
            keep_classifier = {
                "classifier_best": cluster.get("classifier_best"),
                "classifier_prob": cluster.get("classifier_prob"),
                "classifier_accept": cluster.get("classifier_accept"),
            }
            cluster.update(
                {
                    "label": det.get("label"),
                    "class_id": det.get("class_id"),
                    "bbox_xyxy_px": det.get("bbox_xyxy_px"),
                    "bbox_2d": det.get("bbox_2d"),
                    "bbox_yolo": det.get("bbox_yolo"),
                    "score": det.get("score"),
                    "score_source": det.get("score_source") or det.get("source"),
                    "source": source,
                    "source_primary": source_primary,
                    "source_prompt": source_prompt,
                    "source_exemplar_handles": source_exemplar_handles,
                    "source_detector_run_id": source_detector_run_id,
                    "grid_cell": cell,
                }
            )
            cluster["candidate_ids"] = keep_ids
            if keep_sources:
                cluster["source_list"] = keep_sources
            for key, value in keep_classifier.items():
                if value is not None and cluster.get(key) is None:
                    cluster[key] = value
        else:
            if source_primary and cluster.get("source_primary") is None:
                cluster["source_primary"] = source_primary
            if source_prompt and cluster.get("source_prompt") is None:
                cluster["source_prompt"] = source_prompt
            if source_exemplar_handles and cluster.get("source_exemplar_handles") is None:
                cluster["source_exemplar_handles"] = list(source_exemplar_handles)
            if source_detector_run_id and cluster.get("source_detector_run_id") is None:
                cluster["source_detector_run_id"] = source_detector_run_id
        updated_cluster_ids.append(cluster_id)
    _agent_set_active_clusters(_AGENT_ACTIVE_CLUSTERS)
    return {
        "candidate_ids": candidate_ids,
        "cluster_ids": new_cluster_ids + [cid for cid in updated_cluster_ids if cid not in new_cluster_ids],
        "new_cluster_ids": new_cluster_ids,
        "updated_cluster_ids": updated_cluster_ids,
        "new_clusters": len(new_cluster_ids),
        "updated_clusters": len(updated_cluster_ids),
        "rejected": int(rejected),
    }


def _agent_cluster_label_counts(cluster_ids: Sequence[int]) -> Dict[str, int]:
    return _cluster_label_counts(cluster_ids, _AGENT_ACTIVE_CLUSTER_INDEX)


def _agent_cluster_summaries(
    cluster_ids: Sequence[int],
    *,
    max_items: int = 0,
    include_ids: bool = True,
) -> Dict[str, Any]:
    return _cluster_summaries(
        cluster_ids,
        _AGENT_ACTIVE_CLUSTER_INDEX,
        handle_fn=_agent_cluster_handle,
        round_bbox_fn=_agent_round_bbox_2d,
        max_items=max_items,
        include_ids=include_ids,
    )


def _agent_overlay_base_image() -> Optional[Image.Image]:
    return _overlay_base_image(
        grid_image=_AGENT_ACTIVE_GRID_IMAGE,
        image_base64=_AGENT_ACTIVE_IMAGE_BASE64,
        image_token=_AGENT_ACTIVE_IMAGE_TOKEN,
        image_resolver=_agent_resolve_image,
    )


def _agent_tool_grid_cell_from_args(
    tool_args: Mapping[str, Any],
    tool_result: Any,
) -> Optional[str]:
    return _grid_cell_from_args(
        tool_args,
        tool_result,
        grid=_AGENT_ACTIVE_GRID,
        cluster_index=_AGENT_ACTIVE_CLUSTER_INDEX,
    )


def _agent_record_grid_tool_usage(
    tool_name: str,
    tool_args: Mapping[str, Any],
    tool_result: Any,
) -> None:
    _record_grid_usage(
        tool_name,
        tool_args,
        tool_result,
        grid=_AGENT_ACTIVE_GRID,
        cluster_index=_AGENT_ACTIVE_CLUSTER_INDEX,
        usage=_AGENT_GRID_TOOL_USAGE,
        usage_last=_AGENT_GRID_TOOL_LAST,
    )




## NOTE: agent context helpers use *_impl directly to avoid wrapper drift.


def _agent_overlay_crop_xyxy(
    tool_args: Mapping[str, Any],
    tool_result: Any,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    return _overlay_crop_xyxy(
        tool_args,
        tool_result,
        img_w,
        img_h,
        grid=_AGENT_ACTIVE_GRID,
        cluster_index=_AGENT_ACTIVE_CLUSTER_INDEX,
        grid_cell_xyxy_fn=_agent_grid_cell_xyxy,
        clip_xyxy_fn=_agent_clip_xyxy,
        qwen_bbox_to_xyxy_fn=_qwen_bbox_to_xyxy,
        window_local_bbox_fn=_window_local_bbox_2d_to_full_xyxy,
        grid_overlap_ratio=PREPASS_GRID_OVERLAP_RATIO,
    )


def _agent_merge_detections(
    detections: List[Dict[str, Any]],
    *,
    iou_thr: float,
    max_det: Optional[int],
    cross_iou: Optional[float],
) -> List[Dict[str, Any]]:
    return _merge_detections(
        detections,
        iou_thr=iou_thr,
        max_det=max_det,
        cross_iou=cross_iou,
    )



## NOTE: classifier batch helpers use *_impl directly to avoid wrapper drift.


def _agent_sanitize_detection_items(
    items: List[Dict[str, Any]],
    *,
    pil_img: Optional[Image.Image] = None,
    classifier_head: Optional[Dict[str, Any]] = None,
    img_w: int,
    img_h: int,
    labelmap: List[str],
    background: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    background_set = set(background or [])
    label_index = {label: idx for idx, label in enumerate(labelmap)}
    cleaned: List[Dict[str, Any]] = []
    rejected = 0
    pending: List[Dict[str, Any]] = []
    pending_crops: List[Image.Image] = []
    for ann in items:
        label = str(ann.get("label") or ann.get("class_name") or "").strip()
        aligned = _agent_fuzzy_align_label(label, labelmap)
        if not aligned or aligned not in label_index or aligned in background_set:
            rejected += 1
            continue
        raw_score = ann.get("score")
        score: Optional[float]
        if raw_score is None:
            score = None
        else:
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = None
        score_source = ann.get("score_source")
        if not score_source:
            score_source = ann.get("source") or ("unknown" if score is None else None)
        window_bbox = ann.get("window_bbox_2d")
        xyxy = _resolve_agent_bbox_xyxy(ann, img_w, img_h, window_bbox_2d=window_bbox)
        if xyxy is None:
            rejected += 1
            continue
        x1, y1, x2, y2 = xyxy
        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))
        if x2 <= x1 or y2 <= y1:
            rejected += 1
            continue
        base_entry = {
            "bbox_xyxy_px": [x1, y1, x2, y2],
            "bbox_xywh_px": _agent_xyxy_to_xywh(x1, y1, x2, y2),
            "bbox_2d": list(_xyxy_to_qwen_bbox(img_w, img_h, x1, y1, x2, y2)),
            "bbox_yolo": list(_xyxy_to_yolo_norm(img_w, img_h, x1, y1, x2, y2)),
            "label": aligned,
            "class_id": label_index[aligned],
            "score": score,
            "score_source": score_source,
            "sam3_prompt_term": ann.get("sam3_prompt_term"),
            "sam3_prompt_label": ann.get("sam3_prompt_label"),
            "sam3_prompt_source": ann.get("sam3_prompt_source"),
            "source": ann.get("source") or "agent",
        }
        source_list = ann.get("source_list")
        if isinstance(source_list, (list, tuple)):
            base_entry["source_list"] = [str(item) for item in source_list if item]
        score_by_source = ann.get("score_by_source")
        if isinstance(score_by_source, dict):
            normalized_scores: Dict[str, float] = {}
            for raw_source, raw_source_score in score_by_source.items():
                source_key = str(raw_source or "").strip().lower()
                if not source_key:
                    continue
                try:
                    source_score_val = float(raw_source_score)
                except (TypeError, ValueError):
                    continue
                prev_val = normalized_scores.get(source_key)
                if prev_val is None or source_score_val > prev_val:
                    normalized_scores[source_key] = source_score_val
            if normalized_scores:
                base_entry["score_by_source"] = normalized_scores
        raw_atom_ids = ann.get("prepass_atom_ids")
        if isinstance(raw_atom_ids, (list, tuple)):
            atom_ids: List[str] = []
            for raw_atom_id in raw_atom_ids:
                atom_id = str(raw_atom_id or "").strip()
                if atom_id and atom_id not in atom_ids:
                    atom_ids.append(atom_id)
            if atom_ids:
                base_entry["prepass_atom_ids"] = atom_ids

        if pil_img is not None and isinstance(classifier_head, dict):
            classes = [str(c) for c in list(classifier_head.get("classes") or [])]
            target_idx = _find_clip_head_target_index(classes, aligned)
            if target_idx is None:
                rejected += 1
                continue
            source = str(score_source or ann.get("source") or "").strip().lower()
            strict_sources = {
                "sam3_similarity",
                "qwen_inspect",
                "qwen_infer",
                "yolo_zoom",
                "rfdetr_zoom",
            }
            strict = source in strict_sources
            if not strict and source == "sam3_text":
                if score is None or score < PREPASS_STRICT_SAM3_MIN_SCORE:
                    strict = True
            min_prob = float(classifier_head.get("min_prob") or 0.5)
            margin = float(classifier_head.get("margin") or 0.0)
            background_margin = float(classifier_head.get("background_margin") or 0.0)
            if strict:
                min_prob = max(min_prob, PREPASS_CLASSIFIER_STRICT_MIN_PROB)
                margin = max(margin, PREPASS_CLASSIFIER_STRICT_MARGIN)
                background_margin = max(background_margin, PREPASS_CLASSIFIER_STRICT_BG_MARGIN)
            pending.append(
                {
                    "entry": base_entry,
                    "target_idx": target_idx,
                    "min_prob": min_prob,
                    "margin": margin,
                    "background_margin": background_margin,
                }
            )
            pending_crops.append(pil_img.crop((x1, y1, x2, y2)))
        else:
            cleaned.append(base_entry)

    if pending and pil_img is not None and isinstance(classifier_head, dict):
        empty_cache_fn = torch.cuda.empty_cache if torch.cuda.is_available() else None
        proba_arr = _predict_proba_batched_impl(
            pending_crops,
            classifier_head,
            batch_size=_resolve_classifier_batch(),
            encode_batch_fn=lambda items, head_obj, bs: _encode_pil_batch_for_head(
                items, head=head_obj, batch_size_override=bs
            ),
            predict_proba_fn=_clip_head_predict_proba,
            empty_cache_fn=empty_cache_fn,
        )
        if proba_arr is None or proba_arr.ndim != 2 or proba_arr.shape[0] != len(pending):
            rejected += len(pending)
            return cleaned, rejected
        classes = [str(c) for c in list(classifier_head.get("classes") or [])]
        bg_indices = _clip_head_background_indices(classes)
        for row, pending_entry in zip(proba_arr, pending):
            entry = pending_entry["entry"]
            target_idx = pending_entry["target_idx"]
            min_prob = pending_entry["min_prob"]
            margin = pending_entry["margin"]
            background_margin = pending_entry["background_margin"]
            order = sorted(range(len(classes)), key=lambda idx: float(row[idx]), reverse=True)
            best_idx = order[0] if order else None
            best_label = classes[best_idx] if best_idx is not None else None
            best_prob = float(row[best_idx]) if best_idx is not None else None
            keep_mask = _clip_head_keep_mask(
                row.reshape(1, -1),
                target_index=target_idx,
                min_prob=min_prob,
                margin=margin,
                background_indices=bg_indices,
                background_guard=True,
                background_margin=background_margin,
            )
            entry["classifier_best"] = best_label
            entry["classifier_prob"] = best_prob
            entry["classifier_accept"] = bool(keep_mask[0]) if keep_mask is not None and len(keep_mask) else False
            if keep_mask is None or not bool(keep_mask[0]):
                rejected += 1
                continue
            cleaned.append(entry)
    return cleaned, rejected


def _agent_resolve_image(
    image_base64: Optional[str],
    image_token: Optional[str],
    sam_variant: Optional[str] = None,
) -> Tuple[Image.Image, np.ndarray, str]:
    if not image_base64 and not image_token:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="image_payload_missing")
    if image_token:
        variant = _default_variant(sam_variant)
        cached = _fetch_preloaded_image(image_token, variant)
        if cached is None:
            fallback_variant = "sam3" if variant == "sam1" else "sam1"
            cached = _fetch_preloaded_image(image_token, fallback_variant)
            if cached is not None:
                _store_preloaded_image(image_token, cached, variant)
        if cached is not None:
            pil_img = Image.fromarray(cached)
            return pil_img, cached, image_token
    return resolve_image_payload(image_base64, image_token, sam_variant)


@_register_agent_tool("run_detector")
def _agent_tool_run_detector(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    detector_id: Optional[str] = None,
    mode: Optional[str] = "yolo",
    conf: Optional[float] = None,
    sahi: Optional[Dict[str, Any]] = None,
    window: Optional[Any] = None,
    window_bbox_2d: Optional[Sequence[float]] = None,
    grid_cell: Optional[str] = None,
    max_det: Optional[int] = None,
    iou: Optional[float] = None,
    merge_iou: Optional[float] = None,
    expected_labelmap: Optional[Sequence[str]] = None,
    register: Optional[bool] = True,
) -> Dict[str, Any]:
    return _agent_tool_run_detector_impl(
        image_base64=image_base64,
        image_token=image_token,
        detector_id=detector_id,
        mode=mode,
        conf=conf,
        sahi=sahi,
        window=window,
        window_bbox_2d=window_bbox_2d,
        grid_cell=grid_cell,
        max_det=max_det,
        iou=iou,
        merge_iou=merge_iou,
        expected_labelmap=expected_labelmap,
        register=register,
        resolve_image_fn=_agent_resolve_image,
        normalize_window_fn=_normalize_window_xyxy,
        ensure_yolo_runtime_fn=_ensure_yolo_inference_runtime,
        ensure_rfdetr_runtime_fn=_ensure_rfdetr_inference_runtime,
        ensure_yolo_runtime_by_id_fn=_ensure_yolo_inference_runtime_for_detector,
        ensure_rfdetr_runtime_by_id_fn=_ensure_rfdetr_inference_runtime_for_detector,
        raise_labelmap_mismatch_fn=_raise_on_labelmap_mismatch,
        clamp_conf_fn=_clamp_conf_value,
        clamp_iou_fn=_clamp_iou_value,
        clamp_max_det_fn=_clamp_max_det_value,
        clamp_slice_params_fn=_clamp_slice_params,
        slice_image_fn=_slice_image_sahi,
        yolo_extract_fn=_yolo_extract_detections,
        rfdetr_extract_fn=_rfdetr_extract_detections,
        merge_nms_fn=_merge_detections_nms,
        xywh_to_xyxy_fn=_xywh_to_xyxy,
        det_payload_fn=_agent_det_payload,
        register_detections_fn=_agent_register_detections,
        cluster_summaries_fn=_agent_cluster_summaries,
        handles_from_cluster_ids_fn=_agent_handles_from_cluster_ids,
        cluster_label_counts_fn=_agent_cluster_label_counts,
        agent_labelmap=_AGENT_ACTIVE_LABELMAP or [],
        agent_grid=_AGENT_ACTIVE_GRID,
        yolo_lock=YOLO_INFER_LOCK,
        rfdetr_lock=RFDETR_INFER_LOCK,
        http_exception_cls=HTTPException,
    )


@_register_agent_tool("zoom_and_detect")
def _agent_tool_zoom_and_detect(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    detector_id: Optional[str] = None,
    mode: Optional[str] = "yolo",
    conf: Optional[float] = None,
    intent: Optional[str] = None,
    bbox_2d: Optional[Sequence[float]] = None,
    window_bbox_2d: Optional[Sequence[float]] = None,
    bbox_space: Optional[str] = None,
    grid_cell: Optional[str] = None,
    confirm_label: Optional[str] = None,
    confirm_topk: Optional[int] = None,
    max_det: Optional[int] = None,
    iou: Optional[float] = None,
    merge_iou: Optional[float] = None,
) -> Dict[str, Any]:
    if window_bbox_2d is None and bbox_2d is not None:
        if bbox_space and str(bbox_space).lower() == "window":
            raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="window_bbox_required")
        window_bbox_2d = bbox_2d
    if window_bbox_2d is None:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="window_bbox_required")
    if conf is None and _AGENT_ACTIVE_DETECTOR_CONF is not None:
        conf = _AGENT_ACTIVE_DETECTOR_CONF
    mode_norm = (mode or "yolo").strip().lower()
    pil_img, _, _ = _agent_resolve_image(image_base64, image_token)
    img_w, img_h = pil_img.size
    window_xyxy = _normalize_window_xyxy({"bbox_2d": window_bbox_2d}, img_w, img_h)
    sahi_cfg = {"enabled": True}
    result = _agent_tool_run_detector(
        image_base64=image_base64,
        image_token=image_token,
        detector_id=detector_id,
        mode=mode_norm,
        conf=conf,
        sahi=sahi_cfg,
        window_bbox_2d=window_bbox_2d,
        max_det=max_det,
        iou=iou,
        merge_iou=merge_iou,
        register=False,
    )
    detections = result.get("detections") if isinstance(result, dict) else None
    if isinstance(detections, list):
        for det in detections:
            if not isinstance(det, dict):
                continue
            if det.get("source") and not det.get("source_list"):
                det["source_list"] = [det.get("source")]
    register_summary = None
    if register and isinstance(detections, list):
        register_summary = _agent_register_detections(
            detections,
            img_w=img_w,
            img_h=img_h,
            grid=_AGENT_ACTIVE_GRID,
            labelmap=_AGENT_ACTIVE_LABELMAP or [],
            background=None,
            source_override=f"{mode_norm}_zoom",
            owner_cell=grid_cell,
        )
        result["register_summary"] = register_summary
    confirmation = None
    if confirm_label or confirm_topk:
        if window_xyxy is None:
            confirmation = {"label": confirm_label, "error": "window_bbox_invalid"}
        else:
            try:
                classify = _agent_tool_classify_crop(
                    image_base64=image_base64,
                    image_token=image_token,
                    bbox_xyxy_px=list(window_xyxy),
                    bbox_space="full",
                    topk=confirm_topk,
                )
                best = classify.get("best") if isinstance(classify, dict) else None
                best_label = best.get("label") if isinstance(best, dict) else None
                confirmation = {
                    "label": confirm_label,
                    "window_xyxy_px": list(window_xyxy),
                    "window_bbox_2d": list(_xyxy_to_qwen_bbox(img_w, img_h, *window_xyxy)),
                    "best": best,
                    "topk": classify.get("topk") if isinstance(classify, dict) else None,
                    "background_topk": classify.get("background_topk") if isinstance(classify, dict) else None,
                    "label_match": bool(best_label == confirm_label) if confirm_label and best_label else None,
                }
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                confirmation = {"label": confirm_label, "error": detail}
    if window_xyxy is not None:
        result["window_xyxy_px"] = list(window_xyxy)
        result["window_bbox_2d"] = list(window_bbox_2d or [])
    if confirmation is not None:
        result["confirmation"] = confirmation
    register_summary = result.get("register_summary") if isinstance(result, dict) else register_summary
    new_cluster_ids = register_summary.get("new_cluster_ids") if isinstance(register_summary, dict) else []
    updated_cluster_ids = register_summary.get("updated_cluster_ids") if isinstance(register_summary, dict) else []
    new_summary = _agent_cluster_summaries(new_cluster_ids, include_ids=False)
    new_handles = _agent_handles_from_cluster_ids(new_cluster_ids or [])
    updated_handles = _agent_handles_from_cluster_ids(updated_cluster_ids or [])
    agent_view = {
        "mode": mode_norm,
        "grid_cell": grid_cell,
        "intent": (intent or "").strip() or None,
        "new_clusters": register_summary.get("new_clusters") if isinstance(register_summary, dict) else 0,
        "new_handles": new_handles,
        "updated_clusters": len(updated_cluster_ids or []),
        "updated_handles": updated_handles,
        "new_items": new_summary.get("items"),
        "new_items_total": new_summary.get("total"),
        "new_items_truncated": new_summary.get("truncated"),
        "label_counts": _agent_cluster_label_counts(new_cluster_ids or []),
    }
    result["__agent_view__"] = agent_view
    return result


def _sam3_text_payloads_from_state(
    *,
    full_img: Image.Image,
    crop_img: Image.Image,
    prompt: str,
    label: Optional[str],
    score_thr: Optional[float],
    mask_threshold: Optional[float],
    max_results: Optional[int],
    window_xyxy: Optional[Sequence[float]] = None,
    processor_override: Optional[Any] = None,
    state: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[int]]:
    img_w, img_h = full_img.size
    offset_x = 0.0
    offset_y = 0.0
    if window_xyxy:
        offset_x, offset_y = float(window_xyxy[0]), float(window_xyxy[1])
    threshold_val = float(score_thr) if score_thr is not None else 0.2
    mask_val = float(mask_threshold) if mask_threshold is not None else 0.2
    detections = _run_sam3_text_inference(
        crop_img,
        prompt.strip(),
        threshold_val,
        mask_val,
        max_results,
        processor_override=processor_override,
        state=state,
    )
    aligned_label = _agent_fuzzy_align_label(label, _AGENT_ACTIVE_LABELMAP or [])
    assigned_label = aligned_label or (label or "").strip() or None
    class_id = None
    if assigned_label and _AGENT_ACTIVE_LABELMAP:
        for idx, item in enumerate(_AGENT_ACTIVE_LABELMAP):
            if str(item).strip().lower() == str(assigned_label).strip().lower():
                class_id = idx
                break
    payloads: List[Dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = _yolo_to_xyxy(crop_img.width, crop_img.height, det.bbox)
        x1 += offset_x
        y1 += offset_y
        x2 += offset_x
        y2 += offset_y
        payloads.append(
            _agent_det_payload(
                img_w,
                img_h,
                (x1, y1, x2, y2),
                label=assigned_label or det.qwen_label,
                class_id=class_id if assigned_label else det.class_id,
                score=det.score,
                source="sam3_text",
                window=window_xyxy,
            )
        )
    return payloads, assigned_label, class_id


@_register_agent_tool("sam3_text")
def _agent_tool_sam3_text(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    prompt: Optional[str] = None,
    label: Optional[str] = None,
    score_thr: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    max_results: Optional[int] = None,
    window: Optional[Any] = None,
    window_bbox_2d: Optional[Sequence[float]] = None,
    grid_cell: Optional[str] = None,
    register: Optional[bool] = True,
) -> Dict[str, Any]:
    pil_img, _, _ = _agent_resolve_image(image_base64, image_token, "sam3")
    img_w, img_h = pil_img.size
    window_xyxy = _normalize_window_xyxy(window, img_w, img_h)
    if window_xyxy is None and window_bbox_2d is not None:
        window_xyxy = _normalize_window_xyxy({"bbox_2d": window_bbox_2d}, img_w, img_h)
    crop_img = pil_img
    if window_xyxy:
        x1, y1, x2, y2 = window_xyxy
        crop_img = pil_img.crop((x1, y1, x2, y2))
    payloads, assigned_label, _ = _sam3_text_payloads_from_state(
        full_img=pil_img,
        crop_img=crop_img,
        prompt=(prompt or "").strip(),
        label=label,
        score_thr=score_thr,
        mask_threshold=mask_threshold,
        max_results=max_results,
        window_xyxy=window_xyxy,
    )
    register_summary: Optional[Dict[str, Any]] = None
    if register:
        source_override = "sam3_text"
        register_summary = _agent_register_detections(
            payloads,
            img_w=img_w,
            img_h=img_h,
            grid=_AGENT_ACTIVE_GRID,
            labelmap=_AGENT_ACTIVE_LABELMAP or [],
            background=None,
            source_override=source_override,
            owner_cell=grid_cell,
        )
    new_cluster_ids = register_summary.get("new_cluster_ids") if isinstance(register_summary, dict) else []
    updated_cluster_ids = register_summary.get("updated_cluster_ids") if isinstance(register_summary, dict) else []
    new_summary = _agent_cluster_summaries(new_cluster_ids, include_ids=False)
    new_handles = _agent_handles_from_cluster_ids(new_cluster_ids or [])
    updated_handles = _agent_handles_from_cluster_ids(updated_cluster_ids or [])
    agent_view = {
        "label": assigned_label,
        "prompt": (prompt or "").strip() or None,
        "grid_cell": grid_cell,
        "new_clusters": register_summary.get("new_clusters") if isinstance(register_summary, dict) else 0,
        "new_handles": new_handles,
        "updated_clusters": len(updated_cluster_ids or []),
        "updated_handles": updated_handles,
        "new_items": new_summary.get("items"),
        "new_items_total": new_summary.get("total"),
        "new_items_truncated": new_summary.get("truncated"),
        "label_counts": _agent_cluster_label_counts(new_cluster_ids or []),
    }
    return {
        "detections": payloads,
        "prompt": prompt,
        "label": assigned_label,
        "window": window_xyxy,
        "register_summary": register_summary,
        "__agent_view__": agent_view,
    }


@_register_agent_tool("sam3_similarity")
def _agent_tool_sam3_similarity(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    exemplar_boxes: Optional[List[Dict[str, Any]]] = None,
    exemplar_cluster_ids: Optional[List[int]] = None,
    exemplar_handles: Optional[List[str]] = None,
    label: Optional[str] = None,
    score_thr: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    max_results: Optional[int] = None,
    bbox_labels: Optional[List[bool]] = None,
    window: Optional[Any] = None,
    window_bbox_2d: Optional[Sequence[float]] = None,
    bbox_space: Optional[str] = None,
    grid_cell: Optional[str] = None,
    register: Optional[bool] = True,
) -> Dict[str, Any]:
    assigned_label = str(label).strip() if label is not None else ""
    try:
        pil_img, _, _ = _agent_resolve_image(image_base64, image_token, "sam3")
        img_w, img_h = pil_img.size
        if not assigned_label:
            raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="sam3_similarity_label_required")
        window_xyxy = _normalize_window_xyxy(window, img_w, img_h)
        if window_xyxy is None and window_bbox_2d is not None:
            window_xyxy = _normalize_window_xyxy({"bbox_2d": window_bbox_2d}, img_w, img_h)
        crop_img = pil_img
        offset_x = 0.0
        offset_y = 0.0
        if window_xyxy:
            x1, y1, x2, y2 = window_xyxy
            crop_img = pil_img.crop((x1, y1, x2, y2))
            offset_x, offset_y = x1, y1
        exemplar_boxes = list(exemplar_boxes or [])
        exemplar_ids: List[int] = []
        if exemplar_cluster_ids:
            exemplar_ids.extend([int(cid) for cid in exemplar_cluster_ids])
        if exemplar_handles:
            exemplar_ids.extend(_agent_cluster_ids_from_handles(exemplar_handles))
        for cid in exemplar_ids:
            cluster = _AGENT_ACTIVE_CLUSTER_INDEX.get(int(cid))
            if not cluster:
                continue
            bbox = cluster.get("bbox_2d")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                exemplar_boxes.append({"bbox_2d": list(bbox[:4]), "bbox_space": "full"})
        boxes_xywh: List[Tuple[float, float, float, float]] = []
        for box in exemplar_boxes:
            ann: Dict[str, Any] = {}
            window_ref = window_bbox_2d
            if isinstance(box, dict):
                ann["bbox_space"] = box.get("bbox_space") or bbox_space or "full"
                if "bbox_2d" in box:
                    ann["bbox_2d"] = box.get("bbox_2d")
                if "bbox_xyxy_px" in box:
                    ann["bbox_xyxy_px"] = box.get("bbox_xyxy_px")
                if box.get("window_bbox_2d") is not None:
                    window_ref = box.get("window_bbox_2d")
            elif isinstance(box, (list, tuple)) and len(box) >= 4:
                ann["bbox_xyxy_px"] = list(box[:4])
                ann["bbox_space"] = bbox_space or "full"
            xyxy_full = _resolve_agent_bbox_xyxy(ann, img_w, img_h, window_bbox_2d=window_ref)
            if xyxy_full is None:
                continue
            x1, y1, x2, y2 = xyxy_full
            if window_xyxy:
                x1 -= offset_x
                y1 -= offset_y
                x2 -= offset_x
                y2 -= offset_y
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            boxes_xywh.append((x1, y1, w, h))
        if not boxes_xywh:
            raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="sam3_similarity_exemplar_required")
        threshold_val = float(score_thr) if score_thr is not None else 0.2
        mask_val = float(mask_threshold) if mask_threshold is not None else 0.2
        detections = _run_sam3_visual_inference_multi(
            crop_img,
            boxes_xywh,
            bbox_labels,
            threshold_val,
            mask_val,
            max_results,
        )
    except TypeError as exc:
        if "has no len" in str(exc):
            return {
                "detections": [],
                "label": assigned_label,
                "window": None,
                "register_summary": None,
                "__agent_view__": {
                    "label": assigned_label,
                    "grid_cell": grid_cell,
                    "exemplar_handles": exemplar_handles or [],
                    "new_clusters": 0,
                    "new_handles": [],
                    "updated_clusters": 0,
                    "updated_handles": [],
                    "new_items": [],
                    "new_items_total": 0,
                    "new_items_truncated": False,
                    "label_counts": {},
                },
            }
        raise
    assigned_label = assigned_label
    payloads: List[Dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = _yolo_to_xyxy(crop_img.width, crop_img.height, det.bbox)
        x1 += offset_x
        y1 += offset_y
        x2 += offset_x
        y2 += offset_y
        payloads.append(
            _agent_det_payload(
                img_w,
                img_h,
                (x1, y1, x2, y2),
                label=assigned_label,
                class_id=None,
                score=det.score,
                source="sam3_similarity",
                window=window_xyxy,
            )
        )
    register_summary: Optional[Dict[str, Any]] = None
    if register:
        register_summary = _agent_register_detections(
            payloads,
            img_w=img_w,
            img_h=img_h,
            grid=_AGENT_ACTIVE_GRID,
            labelmap=_AGENT_ACTIVE_LABELMAP or [],
            background=None,
            source_override="sam3_similarity",
            owner_cell=grid_cell,
        )
    new_cluster_ids = register_summary.get("new_cluster_ids") if isinstance(register_summary, dict) else []
    updated_cluster_ids = register_summary.get("updated_cluster_ids") if isinstance(register_summary, dict) else []
    new_summary = _agent_cluster_summaries(new_cluster_ids, include_ids=False)
    new_handles = _agent_handles_from_cluster_ids(new_cluster_ids or [])
    updated_handles = _agent_handles_from_cluster_ids(updated_cluster_ids or [])
    exemplar_handles_out = exemplar_handles or _agent_handles_from_cluster_ids(exemplar_ids)
    agent_view = {
        "label": assigned_label,
        "grid_cell": grid_cell,
        "exemplar_handles": exemplar_handles_out,
        "new_clusters": register_summary.get("new_clusters") if isinstance(register_summary, dict) else 0,
        "new_handles": new_handles,
        "updated_clusters": len(updated_cluster_ids or []),
        "updated_handles": updated_handles,
        "new_items": new_summary.get("items"),
        "new_items_total": new_summary.get("total"),
        "new_items_truncated": new_summary.get("truncated"),
        "label_counts": _agent_cluster_label_counts(new_cluster_ids or []),
    }
    return {
        "detections": payloads,
        "label": assigned_label,
        "window": window_xyxy,
        "register_summary": register_summary,
        "__agent_view__": agent_view,
    }


@_register_agent_tool("qwen_infer")
def _agent_tool_qwen_infer(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    prompt: Optional[str] = None,
    item_list: Optional[str] = None,
    items: Optional[List[str]] = None,
    image_type: Optional[str] = None,
    extra_context: Optional[str] = None,
    prompt_type: Optional[str] = None,
    max_results: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    sam_variant: Optional[str] = None,
    window: Optional[Any] = None,
    window_bbox_2d: Optional[Sequence[float]] = None,
    grid_cell: Optional[str] = None,
    register: Optional[bool] = True,
) -> Dict[str, Any]:
    pil_img, np_img, token = _agent_resolve_image(image_base64, image_token, sam_variant)
    img_w, img_h = pil_img.size
    window_xyxy = _normalize_window_xyxy(window, img_w, img_h)
    if window_xyxy is None and window_bbox_2d is not None:
        window_xyxy = _normalize_window_xyxy({"bbox_2d": window_bbox_2d}, img_w, img_h)
    crop_img = pil_img
    crop_np = np_img
    offset_x = 0.0
    offset_y = 0.0
    crop_token = token
    if window_xyxy:
        x1, y1, x2, y2 = window_xyxy
        crop_img = pil_img.crop((x1, y1, x2, y2))
        crop_np = np.asarray(crop_img)
        crop_token = hashlib.md5(crop_np.tobytes()).hexdigest()
        _store_preloaded_image(crop_token, crop_np, _default_variant(sam_variant))
        offset_x, offset_y = x1, y1
    prompt_type = (prompt_type or "bbox").strip().lower()
    if prompt_type not in {"bbox", "point", "bbox_sam"}:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_qwen_prompt_type_invalid")
    manual_prompt = (prompt or "").strip()
    if not manual_prompt:
        if items:
            item_list = ", ".join([str(item).strip() for item in items if str(item).strip()])
        item_list = (item_list or "").strip()
        if not item_list:
            raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="agent_qwen_items_required")
    manual_prompt = _render_qwen_prompt_impl(
        prompt_type,
        items=item_list,
        image_type=(image_type or "").strip() or None,
        extra_context=(extra_context or "").strip() or None,
        get_config_fn=lambda: _get_qwen_prompt_config_impl(qwen_prompt_config, qwen_config_lock),
        http_exception_cls=HTTPException,
        http_422=HTTP_422_UNPROCESSABLE_ENTITY,
    )
    try:
        qwen_text, proc_w, proc_h = _run_qwen_inference(manual_prompt, crop_img, max_new_tokens=max_new_tokens)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_infer_failed:{exc}") from exc
    warnings: List[str] = []
    try:
        _, parsed_items = _extract_qwen_json_block(qwen_text)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        warnings.append(f"parse_error:{detail}")
        return {
            "detections": [],
            "warnings": warnings,
            "prompt": manual_prompt,
            "prompt_type": prompt_type,
            "raw_response": qwen_text,
            "__agent_view__": {
                "grid_cell": grid_cell,
                "prompt_type": prompt_type,
                "new_clusters": 0,
                "new_handles": [],
                "updated_clusters": 0,
                "updated_handles": [],
                "new_items": [],
                "new_items_total": 0,
                "new_items_truncated": False,
                "label_counts": {},
                "warnings": warnings,
            },
        }
    normalized_items = _qwen_items_from_payload(parsed_items)
    if not normalized_items:
        warnings.append("no_results")
        return {
            "detections": [],
            "warnings": warnings,
            "prompt": manual_prompt,
            "prompt_type": prompt_type,
            "raw_response": qwen_text,
            "__agent_view__": {
                "grid_cell": grid_cell,
                "prompt_type": prompt_type,
                "new_clusters": 0,
                "new_handles": [],
                "updated_clusters": 0,
                "updated_handles": [],
                "new_items": [],
                "new_items_total": 0,
                "new_items_truncated": False,
                "label_counts": {},
                "warnings": warnings,
            },
        }
    limit = int(max_results) if max_results is not None else 8
    variant = _default_variant(sam_variant)
    if prompt_type == "bbox":
        boxes = _qwen_bbox_results(normalized_items, proc_w, proc_h, crop_img.width, crop_img.height, limit=limit)
    elif prompt_type == "bbox_sam":
        boxes = _qwen_bbox_sam_results(
            normalized_items,
            proc_w,
            proc_h,
            crop_img,
            crop_np,
            crop_token,
            variant,
            image_name=None,
            limit=limit,
        )
    else:
        boxes = _qwen_point_results(
            normalized_items,
            proc_w,
            proc_h,
            crop_img,
            crop_np,
            crop_token,
            variant,
            image_name=None,
            limit=limit,
        )
    payloads: List[Dict[str, Any]] = []
    for det in boxes:
        x1, y1, x2, y2 = _yolo_to_xyxy_int(det.bbox, crop_img.width, crop_img.height)
        x1 += offset_x
        y1 += offset_y
        x2 += offset_x
        y2 += offset_y
        raw_label = det.qwen_label or det.class_name
        aligned_label = _agent_fuzzy_align_label(raw_label, _AGENT_ACTIVE_LABELMAP or [])
        payloads.append(
            _agent_det_payload(
                img_w,
                img_h,
                (x1, y1, x2, y2),
                label=aligned_label or raw_label,
                class_id=det.class_id,
                score=det.score,
                source=f"qwen_{det.source}",
                window=window_xyxy,
            )
        )
    if not payloads:
        warnings.append("no_results")
    register_summary: Optional[Dict[str, Any]] = None
    if register:
        register_summary = _agent_register_detections(
            payloads,
            img_w=img_w,
            img_h=img_h,
            grid=_AGENT_ACTIVE_GRID,
            labelmap=_AGENT_ACTIVE_LABELMAP or [],
            background=None,
            source_override="qwen_infer",
            owner_cell=grid_cell,
        )
    new_cluster_ids = register_summary.get("new_cluster_ids") if isinstance(register_summary, dict) else []
    updated_cluster_ids = register_summary.get("updated_cluster_ids") if isinstance(register_summary, dict) else []
    new_summary = _agent_cluster_summaries(new_cluster_ids, include_ids=False)
    new_handles = _agent_handles_from_cluster_ids(new_cluster_ids or [])
    updated_handles = _agent_handles_from_cluster_ids(updated_cluster_ids or [])
    agent_view = {
        "grid_cell": grid_cell,
        "prompt_type": prompt_type,
        "new_clusters": register_summary.get("new_clusters") if isinstance(register_summary, dict) else 0,
        "new_handles": new_handles,
        "updated_clusters": len(updated_cluster_ids or []),
        "updated_handles": updated_handles,
        "new_items": new_summary.get("items"),
        "new_items_total": new_summary.get("total"),
        "new_items_truncated": new_summary.get("truncated"),
        "label_counts": _agent_cluster_label_counts(new_cluster_ids or []),
        "warnings": warnings or None,
    }
    return {
        "detections": payloads,
        "warnings": warnings or None,
        "prompt": manual_prompt,
        "prompt_type": prompt_type,
        "raw_response": qwen_text,
        "window": window_xyxy,
        "register_summary": register_summary,
        "__agent_view__": agent_view,
    }


@_register_agent_tool("classify_crop")
def _agent_tool_classify_crop(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    bbox_2d: Optional[Sequence[float]] = None,
    bbox_xyxy_px: Optional[Sequence[float]] = None,
    window_bbox_2d: Optional[Sequence[float]] = None,
    bbox_space: Optional[str] = None,
    cluster_id: Optional[int] = None,
    handle: Optional[str] = None,
    label_hint: Optional[str] = None,
    classifier_id: Optional[str] = None,
    topk: Optional[int] = None,
) -> Dict[str, Any]:
    pil_img, np_img, _ = _agent_resolve_image(image_base64, image_token)
    img_w, img_h = pil_img.size
    ann = {"bbox_space": bbox_space or "full"}
    cluster = None
    if cluster_id is None and handle:
        cluster_id = _agent_cluster_id_from_handle(handle)
    if cluster_id is not None:
        cluster = _AGENT_ACTIVE_CLUSTER_INDEX.get(int(cluster_id))
        if not cluster:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="cluster_not_found")
        bbox_xyxy_px = cluster.get("bbox_xyxy_px")
        bbox_2d = cluster.get("bbox_2d")
    if bbox_xyxy_px is not None:
        ann["bbox_xyxy_px"] = list(bbox_xyxy_px[:4])
    if bbox_2d is not None:
        ann["bbox_2d"] = list(bbox_2d[:4])
    xyxy = _resolve_agent_bbox_xyxy(ann, img_w, img_h, window_bbox_2d=window_bbox_2d)
    if xyxy is None:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="bbox_required")
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="invalid_bbox")
    crop = pil_img.crop((x1, y1, x2, y2))
    head: Optional[Dict[str, Any]] = None
    if classifier_id:
        classifier_path = _resolve_agent_clip_classifier_path_impl(
            classifier_id,
            allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
            allowed_exts=CLASSIFIER_ALLOWED_EXTS,
            path_is_within_root_fn=_path_is_within_root_impl,
            http_exception_cls=HTTPException,
        )
        if classifier_path is not None:
            head = _load_clip_head_from_classifier_impl(
                classifier_path,
                joblib_load_fn=joblib.load,
                http_exception_cls=HTTPException,
                clip_head_background_indices_fn=_clip_head_background_indices,
                resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
                infer_clip_model_fn=_infer_clip_model_from_embedding_dim_impl,
                active_clip_model_name=clip_model_name,
                default_clip_model=DEFAULT_CLIP_MODEL,
                logger=logger,
            )
    elif isinstance(active_classifier_head, dict):
        head = active_classifier_head
    if _AGENT_TRACE_FULL_WRITER is not None:
        _trace_write_full(
            {
                "type": "classifier_head",
                "present": bool(head),
                "classifier_id": classifier_id,
                "head_type": type(head).__name__ if head is not None else None,
            }
        )
    if not isinstance(head, dict):
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="classifier_unavailable")
    feats = _encode_pil_batch_for_head([crop], head=head)
    if feats is None or not isinstance(feats, np.ndarray) or feats.size == 0:
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="classifier_encode_failed")
    proba_arr = _clip_head_predict_proba(feats, head)
    classes = [str(c) for c in list(head.get("classes") or [])]
    if proba_arr is None or proba_arr.ndim != 2 or proba_arr.shape[0] < 1 or proba_arr.shape[1] != len(classes):
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="classifier_predict_failed")
    row = proba_arr[0]
    k = max(1, min(int(topk) if topk is not None else 5, len(classes)))
    order = sorted(range(len(classes)), key=lambda idx: float(row[idx]), reverse=True)
    bg_indices = _clip_head_background_indices(classes)
    topk_items = [
        {"label": classes[idx], "prob": float(row[idx])}
        for idx in order[:k]
    ]
    background_topk = [
        {"label": classes[idx], "prob": float(row[idx])}
        for idx in order
        if idx in bg_indices
    ][:k]
    best = topk_items[0] if topk_items else {"label": "unknown", "prob": None}
    accept = None
    label_target = (label_hint or (cluster.get("label") if cluster else None) or "").strip()
    if label_target:
        target_idx = _find_clip_head_target_index(classes, label_target)
        if target_idx is not None:
            keep_mask = _clip_head_keep_mask(
                proba_arr,
                target_index=target_idx,
                min_prob=float(head.get("min_prob") or 0.5),
                margin=float(head.get("margin") or 0.0),
                background_indices=bg_indices,
                background_guard=True,
                background_margin=float(head.get("background_margin") or 0.0),
            )
            accept = bool(keep_mask[0]) if keep_mask is not None and len(keep_mask) else False
    if cluster is not None:
        cluster["classifier_best"] = best.get("label")
        cluster["classifier_prob"] = best.get("prob")
        cluster["classifier_accept"] = accept
    handle_out = _agent_cluster_handle(cluster) if cluster else None
    agent_view = {
        "handle": handle_out,
        "label_hint": label_target or None,
        "best": best,
        "topk": topk_items[:5],
        "accept": accept,
    }
    return {
        "topk": topk_items,
        "background_topk": background_topk,
        "best": best,
        "accept": accept,
        "__agent_view__": agent_view,
    }


@_register_agent_tool("image_zoom_in_tool")
def _agent_tool_image_zoom_in(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    bbox_2d: Optional[Sequence[float]] = None,
    bbox_xyxy_px: Optional[Sequence[float]] = None,
    window_bbox_2d: Optional[Sequence[float]] = None,
    bbox_space: Optional[str] = None,
    label: Optional[str] = None,
    grid_cell: Optional[str] = None,
    intent: Optional[str] = None,
    cluster_id: Optional[int] = None,
    handle: Optional[str] = None,
) -> Dict[str, Any]:
    pil_img, _, _ = _agent_resolve_image(image_base64, image_token)
    img_w, img_h = pil_img.size
    ann = {"bbox_space": bbox_space or "full"}
    cluster = None
    if cluster_id is None and handle:
        cluster_id = _agent_cluster_id_from_handle(handle)
    if cluster_id is not None:
        cluster = _AGENT_ACTIVE_CLUSTER_INDEX.get(int(cluster_id))
        if not cluster:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="cluster_not_found")
        bbox_xyxy_px = cluster.get("bbox_xyxy_px")
        bbox_2d = cluster.get("bbox_2d")
    if bbox_xyxy_px is not None:
        ann["bbox_xyxy_px"] = list(bbox_xyxy_px[:4])
    if bbox_2d is not None:
        ann["bbox_2d"] = list(bbox_2d[:4])
    xyxy = _resolve_agent_bbox_xyxy(ann, img_w, img_h, window_bbox_2d=window_bbox_2d)
    if xyxy is None:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="bbox_required")
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="invalid_bbox")
    orig_xyxy = (x1, y1, x2, y2)
    expanded = False
    if not grid_cell:
        (x1, y1, x2, y2), expanded = _agent_expand_window_xyxy(
            x1,
            y1,
            x2,
            y2,
            img_w,
            img_h,
            PREPASS_MIN_ZOOM_WINDOW_PX,
        )
    crop = pil_img.crop((x1, y1, x2, y2))
    crop_np = np.asarray(crop)
    token = hashlib.md5(crop_np.tobytes()).hexdigest()
    _store_preloaded_image(token, crop_np, _default_variant(None))
    window_bbox_2d = _xyxy_to_qwen_bbox(img_w, img_h, x1, y1, x2, y2)
    original_window_bbox_2d = _xyxy_to_qwen_bbox(img_w, img_h, *orig_xyxy)
    handle_out = _agent_cluster_handle(cluster) if cluster else None
    agent_view = {
        "grid_cell": grid_cell,
        "handle": handle_out,
        "label": (label or "").strip() or None,
        "intent": (intent or "").strip() or None,
        "width": int(crop.width),
        "height": int(crop.height),
        "expanded": expanded,
    }
    return {
        "image_token": token,
        "window_xyxy_px": [float(x1), float(y1), float(x2), float(y2)],
        "window_bbox_2d": list(window_bbox_2d),
        "label": (label or "").strip(),
        "width": int(crop.width),
        "height": int(crop.height),
        "expanded": expanded,
        "original_window_xyxy_px": list(orig_xyxy) if expanded else None,
        "original_window_bbox_2d": list(original_window_bbox_2d) if expanded else None,
        "__agent_view__": agent_view,
    }


@_register_agent_tool("view_cell_raw")
def _agent_tool_view_cell_raw(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    grid_cell: Optional[str] = None,
) -> Dict[str, Any]:
    if not grid_cell:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="grid_cell_required")
    if not _AGENT_ACTIVE_GRID:
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="grid_unavailable")
    cell_xyxy = _agent_grid_cell_xyxy(
        _AGENT_ACTIVE_GRID,
        str(grid_cell),
        overlap_ratio=PREPASS_GRID_OVERLAP_RATIO,
    )
    if not cell_xyxy:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="grid_cell_invalid")
    base_img = _AGENT_ACTIVE_GRID_IMAGE
    if base_img is None:
        base_img, _, _ = _agent_resolve_image(image_base64, image_token)
    return _view_cell_raw_impl(
        base_img=base_img,
        cell_xyxy=cell_xyxy,
        grid_cell=str(grid_cell),
        store_preloaded_fn=_store_preloaded_image,
        default_variant_fn=_default_variant,
    )


@_register_agent_tool("view_cell_overlay")
def _agent_tool_view_cell_overlay(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    grid_cell: Optional[str] = None,
) -> Dict[str, Any]:
    if not grid_cell:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="grid_cell_required")
    if not _AGENT_ACTIVE_GRID:
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="grid_unavailable")
    cell_xyxy = _agent_grid_cell_xyxy(
        _AGENT_ACTIVE_GRID,
        str(grid_cell),
        overlap_ratio=PREPASS_GRID_OVERLAP_RATIO,
    )
    if not cell_xyxy:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="grid_cell_invalid")
    base_img = _AGENT_ACTIVE_GRID_IMAGE
    if base_img is None:
        base_img, _, _ = _agent_resolve_image(image_base64, image_token)
    clusters = list(_AGENT_ACTIVE_CLUSTERS or [])
    labels = list(_AGENT_ACTIVE_LABELMAP or [])
    if not labels:
        labels = sorted(
            {
                str(cluster.get("label") or "").strip()
                for cluster in clusters
                if isinstance(cluster, dict) and cluster.get("label")
            }
        )
    result, overlay_img = _view_cell_overlay_impl(
        base_img=base_img,
        cell_xyxy=cell_xyxy,
        grid_cell=str(grid_cell),
        clusters=clusters,
        labelmap=labels,
        label_colors_fn=_agent_current_label_colors,
        label_prefixes_fn=_agent_current_label_prefixes,
        render_overlay_fn=_agent_render_detection_overlay,
        dot_radius=_AGENT_ACTIVE_OVERLAY_DOT_RADIUS,
        store_preloaded_fn=_store_preloaded_image,
        default_variant_fn=_default_variant,
    )
    if overlay_img is not None:
        _AGENT_ACTIVE_OVERLAY_IMAGE = overlay_img
    return result


@_register_agent_tool("view_full_overlay")
def _agent_tool_view_full_overlay(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
) -> Dict[str, Any]:
    base_img = _AGENT_ACTIVE_GRID_IMAGE
    if base_img is None:
        base_img, _, _ = _agent_resolve_image(image_base64, image_token)
    clusters = list(_AGENT_ACTIVE_CLUSTERS or [])
    labels = _agent_overlay_labels(clusters, _AGENT_ACTIVE_LABELMAP or [])
    result, overlay_img = _view_full_overlay_impl(
        base_img=base_img,
        clusters=clusters,
        labels=labels,
        label_colors_fn=_agent_current_label_colors,
        label_prefixes_fn=_agent_current_label_prefixes,
        render_overlay_fn=_agent_render_detection_overlay,
        dot_radius=_AGENT_ACTIVE_OVERLAY_DOT_RADIUS,
        grid=_AGENT_ACTIVE_GRID,
        grid_usage=_AGENT_GRID_TOOL_USAGE,
        grid_usage_last=_AGENT_GRID_TOOL_LAST,
        grid_usage_rows_fn=_agent_grid_usage_rows,
        grid_usage_text_fn=_agent_grid_usage_text,
        overlay_key_fn=_agent_overlay_key_text,
    )
    if overlay_img is not None:
        _AGENT_ACTIVE_OVERLAY_IMAGE = overlay_img
    return result


@_register_agent_tool("get_tile_context")
def _agent_tool_get_tile_context(
    grid_cell: Optional[str] = None,
    tile_id: Optional[str] = None,
) -> Dict[str, Any]:
    cell = (grid_cell or tile_id or "").strip()
    if not cell:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="grid_cell_required")
    if not _AGENT_ACTIVE_GRID:
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="grid_unavailable")
    payload, public_payload = _build_tile_context_payloads_impl(
        cell,
        clusters=_AGENT_ACTIVE_CLUSTERS or [],
        grid=_AGENT_ACTIVE_GRID,
        windowed_captions=_AGENT_ACTIVE_WINDOWED_CAPTIONS,
        handle_fn=_agent_cluster_handle,
        cluster_label_counts_fn=_agent_cluster_label_counts,
        overlay_labels_fn=_agent_overlay_labels,
        label_colors_fn=_agent_current_label_colors,
        label_prefixes_fn=_agent_current_label_prefixes,
        overlay_key_fn=_agent_overlay_key_text,
        labelmap=_AGENT_ACTIVE_LABELMAP or [],
        grid_usage=_AGENT_GRID_TOOL_USAGE,
        grid_usage_last=_AGENT_GRID_TOOL_LAST,
    )
    stored = _agent_context_store_impl(
        public_payload,
        kind="tile",
        max_bytes=int(PREPASS_CONTEXT_CHUNK_BYTES),
        tile_store=_AGENT_TILE_CONTEXT_STORE,
        global_store=_AGENT_GLOBAL_CONTEXT_STORE,
    )
    if stored.get("chunked"):
        preview = {
            "cluster_total": public_payload.get("cluster_total"),
            "tile_counts": public_payload.get("tile_counts"),
            "tool_usage": public_payload.get("tool_usage"),
            "tool_usage_last": public_payload.get("tool_usage_last"),
            "caption_hint": public_payload.get("caption_hint"),
        }
        return {
            "tile_id": cell,
            "grid_cell": cell,
            "chunked": True,
            "context_handle": stored.get("context_handle"),
            "chunk_total": stored.get("chunk_total"),
            "byte_size": stored.get("byte_size"),
            "preview": preview,
            "__agent_view__": {
                **public_payload,
                "chunked": True,
                "context_handle": stored.get("context_handle"),
                "chunk_total": stored.get("chunk_total"),
            },
        }
    return {
        **payload,
        "byte_size": stored.get("byte_size"),
        "chunked": False,
        "__agent_view__": {
            **public_payload,
            "chunked": False,
        },
    }


@_register_agent_tool("get_tile_context_chunk")
def _agent_tool_get_tile_context_chunk(
    context_handle: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> Dict[str, Any]:
    if not context_handle:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="context_handle_required")
    if chunk_index is None:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="context_chunk_index_required")
    payload = _agent_context_chunk_impl(
        context_handle,
        chunk_index=int(chunk_index),
        kind="tile",
        tile_store=_AGENT_TILE_CONTEXT_STORE,
        global_store=_AGENT_GLOBAL_CONTEXT_STORE,
    )
    return payload


@_register_agent_tool("get_global_context")
def _agent_tool_get_global_context() -> Dict[str, Any]:
    clusters = list(_AGENT_ACTIVE_CLUSTERS or [])
    counts: Dict[str, int] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        label = str(cluster.get("label") or "").strip()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    usage_rows = _agent_grid_usage_rows(_AGENT_ACTIVE_GRID, _AGENT_GRID_TOOL_USAGE, _AGENT_GRID_TOOL_LAST)
    labels = _agent_overlay_labels(clusters, _AGENT_ACTIVE_LABELMAP or [])
    label_colors = _agent_current_label_colors(labels) if labels else {}
    label_prefixes = _agent_current_label_prefixes(labels) if labels else {}
    payload = {
        "tile_summaries": list(_AGENT_TILE_SUMMARIES or []),
        "global_counts_by_label": counts,
        "tool_usage_heatmap": usage_rows,
        "overall_caption": _AGENT_ACTIVE_OVERALL_CAPTION or "",
        "windowed_captions": list(_AGENT_ACTIVE_WINDOWED_CAPTIONS or []),
        "overlay_key": _agent_overlay_key_text(label_colors, label_prefixes),
    }
    stored = _agent_context_store_impl(
        payload,
        kind="global",
        max_bytes=int(PREPASS_CONTEXT_CHUNK_BYTES),
        tile_store=_AGENT_TILE_CONTEXT_STORE,
        global_store=_AGENT_GLOBAL_CONTEXT_STORE,
    )
    if stored.get("chunked"):
        preview = {
            "tile_summaries_count": len(payload["tile_summaries"]),
            "global_counts_by_label": counts,
        }
        return {
            "chunked": True,
            "context_handle": stored.get("context_handle"),
            "chunk_total": stored.get("chunk_total"),
            "byte_size": stored.get("byte_size"),
            "preview": preview,
        }
    return {
        **payload,
        "byte_size": stored.get("byte_size"),
        "chunked": False,
    }


@_register_agent_tool("get_global_context_chunk")
def _agent_tool_get_global_context_chunk(
    context_handle: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> Dict[str, Any]:
    if not context_handle:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="context_handle_required")
    if chunk_index is None:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="context_chunk_index_required")
    payload = _agent_context_chunk_impl(
        context_handle,
        chunk_index=int(chunk_index),
        kind="global",
        tile_store=_AGENT_TILE_CONTEXT_STORE,
        global_store=_AGENT_GLOBAL_CONTEXT_STORE,
    )
    return payload


@_register_agent_tool("think_missed_objects")
def _agent_tool_think_missed_objects(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    overlay = _AGENT_ACTIVE_OVERLAY_IMAGE
    if overlay is None:
        base = _agent_overlay_base_image()
        if base is None:
            base, _, _ = _agent_resolve_image(image_base64, image_token)
        clusters = list(_AGENT_ACTIVE_CLUSTERS or [])
        labels = _agent_overlay_labels(clusters, _AGENT_ACTIVE_LABELMAP or [])
        label_colors = _agent_current_label_colors(labels) if labels else {}
        label_prefixes = _agent_current_label_prefixes(labels) if labels else {}
        overlay = base
        if clusters:
            overlay = _agent_render_detection_overlay(
                base,
                clusters,
                label_colors,
                dot_radius=_AGENT_ACTIVE_OVERLAY_DOT_RADIUS,
                label_prefixes=label_prefixes,
            )
    labels = list(_AGENT_ACTIVE_LABELMAP or [])
    usage_rows = _agent_grid_usage_rows(_AGENT_ACTIVE_GRID, _AGENT_GRID_TOOL_USAGE, _AGENT_GRID_TOOL_LAST)
    usage_text = _agent_grid_usage_text(usage_rows)
    prompt_lines = [
        "You are reviewing an annotated aerial image.",
        "Return JSON only: {\"missing_labels\": [...], \"missing_tiles\": [...], \"rationale\": \"...\"}.",
        "Use labelmap classes only. Use grid cells like A1, B2.",
        f"Labelmap: {', '.join(labels) if labels else 'none'}",
    ]
    if _AGENT_ACTIVE_OVERALL_CAPTION:
        prompt_lines.append(f"Overall caption: {_AGENT_ACTIVE_OVERALL_CAPTION}")
    if _AGENT_ACTIVE_WINDOWED_CAPTIONS:
        sample_caps = _AGENT_ACTIVE_WINDOWED_CAPTIONS[:8]
        cap_lines = []
        for entry in sample_caps:
            if isinstance(entry, dict):
                name = entry.get("window") or "window"
                cap = entry.get("caption") or ""
                cap_lines.append(f"{name}: {cap}")
        if cap_lines:
            prompt_lines.append("Windowed captions: " + " | ".join(cap_lines))
    if usage_text:
        prompt_lines.append(f"Tool usage by grid: {usage_text}")
    if notes:
        prompt_lines.append(f"Notes: {notes}")
    prompt = "\n".join(prompt_lines)
    try:
        raw, _, _ = _run_qwen_inference(
            prompt,
            overlay,
            max_new_tokens=256,
            system_prompt_override="Return JSON only.",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"think_missed_failed:{exc}") from exc
    json_text = _extract_balanced_json_impl(raw, "{", "}") or ""
    data: Dict[str, Any] = {}
    if json_text:
        try:
            data = json.loads(json_text)
        except Exception:
            data = {}
    missing_labels = data.get("missing_labels") if isinstance(data, dict) else None
    missing_tiles = data.get("missing_tiles") if isinstance(data, dict) else None
    rationale = data.get("rationale") if isinstance(data, dict) else None
    return {
        "missing_labels": missing_labels if isinstance(missing_labels, list) else [],
        "missing_tiles": missing_tiles if isinstance(missing_tiles, list) else [],
        "rationale": str(rationale or "").strip(),
    }


@_register_agent_tool("grid_label_counts")
def _agent_tool_grid_label_counts(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    del image_base64, image_token
    if not _AGENT_ACTIVE_GRID:
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="grid_unavailable")
    clusters = list(_AGENT_ACTIVE_CLUSTERS or [])
    summary = _agent_grid_label_counts(
        grid=_AGENT_ACTIVE_GRID,
        clusters=clusters,
        label=label,
        labelmap=_AGENT_ACTIVE_LABELMAP or [],
    )
    agent_cells = []
    for cell in summary:
        if not isinstance(cell, dict):
            continue
        agent_cells.append(
            {
                "grid_cell": cell.get("grid_cell"),
                "total": cell.get("total"),
                "counts": cell.get("counts"),
            }
        )
    return {
        "label": _agent_fuzzy_align_label(label, _AGENT_ACTIVE_LABELMAP or []) if label else None,
        "cells": summary,
        "total_cells": len(summary),
        "__agent_view__": {
            "label": _agent_fuzzy_align_label(label, _AGENT_ACTIVE_LABELMAP or []) if label else None,
            "cells": agent_cells,
            "total_cells": len(summary),
        },
    }


@_register_agent_tool("get_labelmap")
def _agent_tool_get_labelmap(
    dataset_id: Optional[str] = None,
    classifier_id: Optional[str] = None,
) -> Dict[str, Any]:
    classes, glossary = _agent_load_labelmap_meta(dataset_id)
    head: Optional[Dict[str, Any]] = None
    if classifier_id:
        classifier_path = _resolve_agent_clip_classifier_path_impl(
            classifier_id,
            allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
            allowed_exts=CLASSIFIER_ALLOWED_EXTS,
            path_is_within_root_fn=_path_is_within_root_impl,
            http_exception_cls=HTTPException,
        )
        if classifier_path is not None:
            head = _load_clip_head_from_classifier_impl(
                classifier_path,
                joblib_load_fn=joblib.load,
                http_exception_cls=HTTPException,
                clip_head_background_indices_fn=_clip_head_background_indices,
                resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
                infer_clip_model_fn=_infer_clip_model_from_embedding_dim_impl,
                active_clip_model_name=clip_model_name,
                default_clip_model=DEFAULT_CLIP_MODEL,
                logger=logger,
            )
    elif isinstance(active_classifier_head, dict):
        head = active_classifier_head
    background = _agent_background_classes_from_head(head)
    return {
        "classes": classes,
        "background_classes": background,
        "glossary": glossary,
        "rules": {
            "reject_background": True,
            "require_labelmap": True,
        },
    }


@_register_agent_tool("list_candidates")
def _agent_tool_list_candidates(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    label: Optional[str] = None,
    source: Optional[str] = None,
    min_score: Optional[float] = None,
    include_scoreless: Optional[bool] = True,
    sort_by: Optional[str] = "score_desc",
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    del image_base64, image_token
    active = list(_AGENT_ACTIVE_CLUSTERS or _AGENT_ACTIVE_DETECTIONS or [])
    if not active:
        return {"candidates": [], "total": 0, "returned": 0, "filters": {}}
    label_filter = str(label).strip() if label else None
    aligned_label = _agent_fuzzy_align_label(label_filter, _AGENT_ACTIVE_LABELMAP or [])
    if aligned_label:
        label_filter = aligned_label
    source_filter = str(source).strip() if source else None
    source_terms = {term for term in (source_filter or "").split(",") if term.strip()}
    source_terms = {term.strip().lower() for term in source_terms}
    try:
        min_score_val = float(min_score) if min_score is not None else None
    except (TypeError, ValueError):
        min_score_val = None
    include_scoreless = True if include_scoreless is None else bool(include_scoreless)
    sort_key = (sort_by or "score_desc").strip().lower()

    def score_value(det: Dict[str, Any]) -> Optional[float]:
        raw = det.get("score")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    filtered: List[Dict[str, Any]] = []
    for idx, det in enumerate(active):
        if not isinstance(det, dict):
            continue
        det_label = str(det.get("label") or det.get("class_name") or "").strip()
        if label_filter and det_label and det_label != label_filter:
            continue
        det_score = score_value(det)
        if det_score is None:
            if not include_scoreless:
                continue
        elif min_score_val is not None and det_score < min_score_val:
            continue
        if source_terms:
            det_sources: Set[str] = set()
            for key in ("source", "score_source"):
                val = det.get(key)
                if val:
                    det_sources.add(str(val).strip().lower())
            source_list = det.get("source_list")
            if isinstance(source_list, (list, tuple)):
                for item in source_list:
                    if item:
                        det_sources.add(str(item).strip().lower())
            if not det_sources.intersection(source_terms):
                continue
        entry = {
            "cluster_id": int(det.get("cluster_id")) if det.get("cluster_id") is not None else int(idx + 1),
            "handle": _agent_cluster_handle(det),
            "label": det_label,
            "class_id": det.get("class_id"),
            "score": det.get("score"),
            "score_source": det.get("score_source") or det.get("source") or "unknown",
            "source": det.get("source"),
            "source_list": det.get("source_list"),
            "candidate_count": len(det.get("candidate_ids") or []),
            "grid_cell": det.get("grid_cell"),
            "classifier_best": det.get("classifier_best"),
            "classifier_prob": det.get("classifier_prob"),
            "classifier_accept": det.get("classifier_accept"),
        }
        filtered.append(entry)

    if sort_key in {"score", "score_desc", "confidence"}:
        filtered.sort(key=lambda d: (score_value(d) if score_value(d) is not None else -1.0), reverse=True)
        sort_key = "score_desc"
    elif sort_key in {"score_asc", "confidence_asc"}:
        filtered.sort(key=lambda d: (score_value(d) if score_value(d) is not None else 1e9))
        sort_key = "score_asc"
    elif sort_key == "label":
        filtered.sort(key=lambda d: str(d.get("label") or ""))
    elif sort_key == "source":
        filtered.sort(key=lambda d: str(d.get("source") or ""))
    else:
        sort_key = "none"
    try:
        max_items_val = int(max_items) if max_items is not None else 0
    except (TypeError, ValueError):
        max_items_val = 0
    trimmed = filtered[:max_items_val] if max_items_val > 0 else filtered
    agent_candidates = []
    for item in trimmed:
        if not isinstance(item, dict):
            continue
        agent_candidates.append(
            {
                "handle": item.get("handle"),
                "label": item.get("label"),
                "grid_cell": item.get("grid_cell"),
                "score": item.get("score"),
                "score_source": item.get("score_source"),
                "source": item.get("source"),
                "source_list": item.get("source_list"),
                "verified": bool(item.get("classifier_accept")),
            }
        )
    return {
        "candidates": trimmed,
        "total": len(filtered),
        "returned": len(trimmed),
        "filters": {
            "label": label_filter,
            "source": source_filter,
            "min_score": min_score_val,
            "include_scoreless": include_scoreless,
            "sort_by": sort_key,
            "max_items": max_items_val,
        },
        "__agent_view__": {
            "candidates": agent_candidates,
            "total": len(filtered),
            "returned": len(trimmed),
            "filters": {
                "label": label_filter,
                "source": source_filter,
                "min_score": min_score_val,
                "include_scoreless": include_scoreless,
                "sort_by": sort_key,
                "max_items": max_items_val,
            },
        },
    }


@_register_agent_tool("submit_annotations")
def _agent_tool_submit_annotations(
    image_base64: Optional[str] = None,
    image_token: Optional[str] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
    cluster_ids: Optional[List[int]] = None,
    handles: Optional[List[str]] = None,
    include_all: Optional[bool] = None,
    dataset_id: Optional[str] = None,
    classifier_id: Optional[str] = None,
    window_bbox_2d: Optional[Sequence[float]] = None,
    iou: Optional[float] = None,
    cross_iou: Optional[float] = None,
    max_det: Optional[int] = None,
) -> Dict[str, Any]:
    global _AGENT_LAST_SUBMIT_DETECTIONS
    pil_img, _, _ = _agent_resolve_image(image_base64, image_token)
    img_w, img_h = pil_img.size
    labelmap = _agent_load_labelmap(dataset_id)
    head: Optional[Dict[str, Any]] = None
    if classifier_id:
        classifier_path = _resolve_agent_clip_classifier_path_impl(
            classifier_id,
            allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
            allowed_exts=CLASSIFIER_ALLOWED_EXTS,
            path_is_within_root_fn=_path_is_within_root_impl,
            http_exception_cls=HTTPException,
        )
        if classifier_path is not None:
            head = _load_clip_head_from_classifier_impl(
                classifier_path,
                joblib_load_fn=joblib.load,
                http_exception_cls=HTTPException,
                clip_head_background_indices_fn=_clip_head_background_indices,
                resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
                infer_clip_model_fn=_infer_clip_model_from_embedding_dim_impl,
                active_clip_model_name=clip_model_name,
                default_clip_model=DEFAULT_CLIP_MODEL,
                logger=logger,
            )
    elif isinstance(active_classifier_head, dict):
        head = active_classifier_head
    background = _agent_background_classes_from_head(head)
    normalized_annotations: List[Dict[str, Any]] = []
    cluster_id_list: List[int] = []
    label_overrides: Dict[int, str] = {}
    if include_all:
        cluster_id_list = [int(c.get("cluster_id")) for c in (_AGENT_ACTIVE_CLUSTERS or []) if c.get("cluster_id") is not None]
    if handles:
        cluster_id_list.extend(_agent_cluster_ids_from_handles(handles))
    if cluster_ids:
        cluster_id_list.extend([int(cid) for cid in cluster_ids])
    for ann in annotations or []:
        if isinstance(ann, (int, float)):
            cluster_id_list.append(int(ann))
            continue
        if isinstance(ann, dict) and ann.get("cluster_id") is not None:
            cid_val = int(ann.get("cluster_id"))
            cluster_id_list.append(cid_val)
            override_label = ann.get("label")
            if override_label:
                label_overrides[cid_val] = str(override_label)
            continue
        if window_bbox_2d is not None and isinstance(ann, dict) and ann.get("window_bbox_2d") is None:
            ann = {**ann, "window_bbox_2d": list(window_bbox_2d)}
        normalized_annotations.append(ann)
    if cluster_id_list:
        seen = set()
        for cid in cluster_id_list:
            if cid in seen:
                continue
            seen.add(cid)
            cluster = _AGENT_ACTIVE_CLUSTER_INDEX.get(int(cid))
            if not cluster:
                continue
            label = label_overrides.get(cid) or cluster.get("label")
            normalized_annotations.append(
                {
                    "label": label,
                    "bbox_xyxy_px": cluster.get("bbox_xyxy_px"),
                    "bbox_2d": cluster.get("bbox_2d"),
                    "bbox_yolo": cluster.get("bbox_yolo"),
                    "score": cluster.get("score"),
                    "score_source": cluster.get("score_source") or cluster.get("source"),
                    "class_id": cluster.get("class_id"),
                    "source": cluster.get("source") or "agent",
                    "source_list": cluster.get("source_list"),
                    "origin": cluster.get("origin"),
                }
            )
    cleaned, rejected = _agent_sanitize_detection_items(
        normalized_annotations,
        pil_img=pil_img,
        classifier_head=head,
        img_w=img_w,
        img_h=img_h,
        labelmap=labelmap,
        background=background,
    )

    def _unit_float(value: Optional[float], *, default: Optional[float]) -> Optional[float]:
        if value is None:
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(parsed):
            return default
        return max(0.0, min(1.0, parsed))

    iou_thr = _unit_float(iou, default=0.5)
    cross_iou_thr = _unit_float(cross_iou, default=0.8) if cross_iou is not None else None
    if cross_iou_thr is not None and cross_iou_thr <= 0.0:
        cross_iou_thr = None
    merged = _agent_merge_detections(
        cleaned,
        iou_thr=float(iou_thr if iou_thr is not None else 0.5),
        max_det=max_det,
        cross_iou=cross_iou_thr,
    )
    _AGENT_LAST_SUBMIT_DETECTIONS = list(merged)
    submitted_handles = _agent_handles_from_cluster_ids(cluster_id_list)
    agent_view = {
        "submitted_handles": sorted(set(submitted_handles)),
        "count": len(merged),
    }
    return {
        "detections": merged,
        "rejected": rejected,
        "count": len(merged),
        "__agent_view__": agent_view,
    }


def _agent_apply_ensemble_filter(
    detections: List[Dict[str, Any]],
    *,
    dataset_id: Optional[str],
    image_name: Optional[str],
    classifier_id: Optional[str],
    job_id: Optional[str],
    prepass_provenance: Optional[Dict[str, Any]] = None,
    trace_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_full_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_readable: Optional[Callable[[str], None]] = None,
    warnings: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not job_id:
        return detections
    if not dataset_id or not image_name:
        if warnings is not None:
            warnings.append("ensemble_filter_missing_dataset_or_image")
        return detections
    job_dir = CALIBRATION_ROOT / job_id
    model_path = job_dir / "ensemble_mlp.pt"
    meta_path = job_dir / "ensemble_mlp.meta.json"
    score_tool = "score_ensemble_candidates.py"
    if not model_path.exists():
        alt_model = job_dir / "ensemble_xgb.json"
        alt_meta = job_dir / "ensemble_xgb.meta.json"
        if alt_model.exists() and alt_meta.exists():
            model_path = alt_model
            meta_path = alt_meta
            score_tool = "score_ensemble_candidates_xgb.py"
    if not model_path.exists() or not meta_path.exists():
        if warnings is not None:
            warnings.append("ensemble_filter_model_missing")
        return detections
    if not detections:
        return detections
    classifier_path = None
    if classifier_id:
        classifier_path = _resolve_agent_clip_classifier_path_impl(
            classifier_id,
            allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
            allowed_exts=CLASSIFIER_ALLOWED_EXTS,
            path_is_within_root_fn=_path_is_within_root_impl,
            http_exception_cls=HTTPException,
        )
    if classifier_path is None:
        if warnings is not None:
            warnings.append("ensemble_filter_classifier_missing")
        return detections

    tmp_dir = UPLOAD_ROOT / "tmp_ensemble"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:8]
    jsonl_path = tmp_dir / f"ensemble_{stamp}.jsonl"
    features_path = tmp_dir / f"ensemble_{stamp}.npz"
    scored_path = tmp_dir / f"ensemble_{stamp}.scored.jsonl"
    try:
        with jsonl_path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "image": image_name,
                        "detections": detections,
                        "provenance": prepass_provenance if isinstance(prepass_provenance, dict) else None,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
        root_dir = Path(__file__).resolve().parent
        build_cmd = [
            sys.executable,
            str(root_dir / "tools" / "build_ensemble_features.py"),
            "--input",
            str(jsonl_path),
            "--dataset",
            dataset_id,
            "--output",
            str(features_path),
            "--support-iou",
            "0.5",
            "--min-crop-size",
            "4",
            "--device",
            "cuda",
            "--classifier-id",
            str(classifier_path),
        ]
        subprocess.run(build_cmd, check=True)
        score_cmd = [
            sys.executable,
            str(root_dir / "tools" / score_tool),
            "--model",
            str(model_path),
            "--meta",
            str(meta_path),
            "--data",
            str(features_path),
            "--output",
            str(scored_path),
        ]
        subprocess.run(score_cmd, check=True)
        accepted: List[Dict[str, Any]] = []
        for line in scored_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if entry.get("ensemble_accept"):
                accepted.append(entry)
        if trace_writer:
            trace_writer(
                {
                    "type": "ensemble_filter",
                    "accepted": len(accepted),
                    "total": len(detections),
                    "job_id": job_id,
                }
            )
        if trace_full_writer:
            trace_full_writer(
                {
                    "type": "ensemble_filter",
                    "accepted": len(accepted),
                    "total": len(detections),
                    "job_id": job_id,
                }
            )
        if trace_readable:
            trace_readable(
                f"ensemble filter: accepted={len(accepted)} total={len(detections)} job={job_id}"
            )
        return accepted
    except Exception as exc:  # noqa: BLE001
        if warnings is not None:
            warnings.append(f"ensemble_filter_failed:{exc}")
        return detections


@_register_agent_tool("log_observation")
def _agent_tool_log_observation(
    text: Optional[str] = None,
    bbox_2d: Optional[Sequence[float]] = None,
    window_bbox_2d: Optional[Sequence[float]] = None,
    grid_cell: Optional[str] = None,
    label_hint: Optional[str] = None,
) -> Dict[str, Any]:
    cleaned = _agent_clean_observation_text(text, max_len=160)
    if not cleaned:
        return {"ok": False, "error": "observation_missing"}
    payload: Dict[str, Any] = {
        "ok": True,
        "observation": cleaned,
    }
    if label_hint:
        payload["label_hint"] = str(label_hint)
    if grid_cell:
        payload["grid_cell"] = str(grid_cell)
    _ = bbox_2d
    _ = window_bbox_2d
    return payload


@_register_agent_tool("log_status")
def _agent_tool_log_status(text: Optional[str] = None) -> Dict[str, Any]:
    cleaned = _agent_clean_observation_text(text, max_len=160)
    if not cleaned:
        return {"ok": False, "error": "status_missing"}
    return {"ok": True, "status": cleaned}


def _agent_tool_specs_text(grid_enabled: bool = False) -> str:
    tool_names = [
        "get_tile_context",
        "get_tile_context_chunk",
        "get_global_context",
        "get_global_context_chunk",
        "view_cell_raw",
        "view_cell_overlay",
        "view_full_overlay",
        "list_candidates",
        "grid_label_counts",
        "look_and_inspect",
        "zoom_and_detect",
        "sam3_text",
        "sam3_similarity",
        "qwen_infer",
        "classify_crop",
        "image_zoom_in_tool",
        "think_missed_objects",
        "submit_annotations",
        "log_observation",
        "log_status",
    ]
    tools = _agent_tool_specs_facade(grid_enabled=bool(grid_enabled), tool_names=tool_names)
    return json.dumps(tools, ensure_ascii=False, indent=2)



def _agent_tool_specs(grid_enabled: bool = False) -> List[Dict[str, Any]]:
    """Return the tool specs as a list for Hermes-style tool templates."""
    try:
        return json.loads(_agent_tool_specs_text(grid_enabled=grid_enabled))
    except Exception:
        return []


def _agent_tool_specs_facade(
    *,
    grid_enabled: bool,
    tool_names: Sequence[str],
) -> List[Dict[str, Any]]:
    grid_cell = {
        "type": "string",
        "description": "Grid cell reference (e.g., C2). Columns are letters, rows are numbers; top-left is A1.",
    }
    required_grid = ["grid_cell"] if grid_enabled else []
    handle_list = {
        "type": "array",
        "description": "Handles returned by list_candidates/get_tile_context (e.g., LV12).",
        "items": {"type": "string"},
    }
    handle = {"type": "string", "description": "Handle returned by list_candidates/get_tile_context (e.g., LV12)."}
    label = {"type": "string", "description": "Canonical labelmap class."}
    prompt = {"type": "string", "description": "Single-term prompt/synonym for SAM3."}
    intent = {"type": "string", "description": "Short note on what you are searching for."}
    tool_map = {
        "get_tile_context": {
            "name": "get_tile_context",
            "description": "Return tile context (cluster handles, counts, captions, tool usage).",
            "parameters": {"type": "object", "properties": {"grid_cell": grid_cell}, "required": required_grid},
        },
        "get_tile_context_chunk": {
            "name": "get_tile_context_chunk",
            "description": "Fetch a chunk of a large tile context payload by handle + chunk_index.",
            "parameters": {
                "type": "object",
                "properties": {"context_handle": {"type": "string"}, "chunk_index": {"type": "number"}},
                "required": ["context_handle", "chunk_index"],
            },
        },
        "get_global_context": {
            "name": "get_global_context",
            "description": "Return global context (tile summaries, counts, captions).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "get_global_context_chunk": {
            "name": "get_global_context_chunk",
            "description": "Fetch a chunk of a large global context payload by handle + chunk_index.",
            "parameters": {
                "type": "object",
                "properties": {"context_handle": {"type": "string"}, "chunk_index": {"type": "number"}},
                "required": ["context_handle", "chunk_index"],
            },
        },
        "view_cell_raw": {
            "name": "view_cell_raw",
            "description": "Return the raw image crop for a grid cell (no detection dots).",
            "parameters": {"type": "object", "properties": {"grid_cell": grid_cell}, "required": required_grid},
        },
        "view_cell_overlay": {
            "name": "view_cell_overlay",
            "description": "Return the overlay crop for a grid cell (colored dots + handles).",
            "parameters": {"type": "object", "properties": {"grid_cell": grid_cell}, "required": required_grid},
        },
        "view_full_overlay": {
            "name": "view_full_overlay",
            "description": "Return the full image with grid + detection overlay and tool usage summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "list_candidates": {
            "name": "list_candidates",
            "description": "List current cluster candidates (handles, labels, scores, grid cells).",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "min_score": {"type": "number"},
                    "include_scoreless": {"type": "boolean"},
                    "sort_by": {"type": "string", "enum": ["score_desc", "score_asc", "label", "source", "none"]},
                    "max_items": {"type": "number"},
                },
                "required": [],
            },
        },
        "grid_label_counts": {
            "name": "grid_label_counts",
            "description": "Return per-grid-cell counts (no IDs).",
            "parameters": {"type": "object", "properties": {"label": {"type": "string"}}, "required": []},
        },
        "look_and_inspect": {
            "name": "look_and_inspect",
            "description": "Inspect a grid cell visually and propose candidate objects.",
            "parameters": {
                "type": "object",
                "properties": {"grid_cell": grid_cell, "intent": intent, "max_objects": {"type": "number"}},
                "required": required_grid,
            },
        },
        "zoom_and_detect": {
            "name": "zoom_and_detect",
            "description": "Run detector on a grid cell (zoomed crop).",
            "parameters": {
                "type": "object",
                "properties": {
                    "grid_cell": grid_cell,
                    "intent": intent,
                    "confirm_label": label,
                    "confirm_topk": {"type": "number"},
                },
                "required": required_grid,
            },
        },
        "sam3_text": {
            "name": "sam3_text",
            "description": "Run SAM3 text prompt for a single term; assign canonical label.",
            "parameters": {
                "type": "object",
                "properties": {"grid_cell": grid_cell, "label": label, "prompt": prompt},
                "required": (required_grid + ["label"]) if required_grid else ["label"],
            },
        },
        "sam3_similarity": {
            "name": "sam3_similarity",
            "description": "Run SAM3 similarity from exemplar handles; assign canonical label.",
            "parameters": {
                "type": "object",
                "properties": {"grid_cell": grid_cell, "label": label, "exemplar_handles": handle_list},
                "required": (required_grid + ["label", "exemplar_handles"]) if required_grid else ["label", "exemplar_handles"],
            },
        },
        "qwen_infer": {
            "name": "qwen_infer",
            "description": "Ask Qwen-VL to propose new boxes for a list of labels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grid_cell": grid_cell,
                    "items": {"type": "array", "items": {"type": "string"}},
                    "prompt_type": {"type": "string", "enum": ["bbox", "point"]},
                    "intent": intent,
                },
                "required": required_grid,
            },
        },
        "classify_crop": {
            "name": "classify_crop",
            "description": "Classify a cluster crop to verify its label.",
            "parameters": {
                "type": "object",
                "properties": {"handle": handle, "label_hint": label, "topk": {"type": "number"}},
                "required": ["handle"],
            },
        },
        "image_zoom_in_tool": {
            "name": "image_zoom_in_tool",
            "description": "Return a zoomed image for a grid cell or handle.",
            "parameters": {
                "type": "object",
                "properties": {"grid_cell": grid_cell, "handle": handle, "intent": intent},
                "required": [],
            },
        },
        "think_missed_objects": {
            "name": "think_missed_objects",
            "description": "Analyze overlay + captions to suggest missing labels/tiles.",
            "parameters": {"type": "object", "properties": {"notes": {"type": "string"}}, "required": []},
        },
        "submit_annotations": {
            "name": "submit_annotations",
            "description": "Submit final annotations using handles or include_all.",
            "parameters": {
                "type": "object",
                "properties": {"handles": handle_list, "include_all": {"type": "boolean"}},
                "required": [],
            },
        },
        "log_observation": {
            "name": "log_observation",
            "description": "Log a single-line observation about what you see (max 160 chars).",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
        "log_status": {
            "name": "log_status",
            "description": "Log a short status update (max 160 chars).",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    }
    specs: List[Dict[str, Any]] = []
    for name in tool_names:
        spec = tool_map.get(name)
        if not spec:
            continue
        specs.append({"type": "function", "function": spec})
    return specs


def _agent_full_trace_write(record: Dict[str, Any]) -> None:
    if _AGENT_TRACE_FULL_WRITER is None:
        return
    try:
        _AGENT_TRACE_FULL_WRITER(_agent_trace_full_jsonable(record))
    except Exception:
        return


_agent_readable_write = lambda line: _agent_readable_write_impl(  # noqa: E731
    line,
    writer=_AGENT_TRACE_READABLE_WRITER,
    to_console=PREPASS_READABLE_TO_CONSOLE,
)


(
    _agent_run_deep_prepass_part_a,
    _agent_deep_prepass_cleanup,
    _agent_run_deep_prepass,
    _agent_run_deep_prepass_caption,
) = _build_deep_prepass_runners_impl(
    run_detector_fn=lambda **kwargs: _agent_tool_run_detector(**kwargs),
    attach_provenance_fn=_agent_attach_provenance,
    generate_sam3_synonyms_fn=_agent_generate_sam3_synonyms,
    generate_text_fn=lambda prompt, max_new_tokens=128, use_system_prompt=True: _generate_qwen_text_impl(
        prompt,
        max_new_tokens=max_new_tokens,
        use_system_prompt=use_system_prompt,
        system_prompt=(active_qwen_metadata or {}).get("system_prompt"),
        ensure_qwen_ready_fn=_ensure_qwen_ready,
        resolve_qwen_device_fn=lambda: _resolve_qwen_device_impl(QWEN_DEVICE_PREF, torch_module=torch),
    ),
    extract_json_fn=_extract_balanced_json,
    default_synonyms=_DEFAULT_SAM3_SYNONYMS,
    label_key_fn=_glossary_label_key,
    sam3_text_windows_fn=_agent_sam3_text_windows,
    ensure_sam3_text_runtime_fn=_ensure_sam3_text_runtime,
    normalize_window_xyxy_fn=_normalize_window_xyxy,
    sam3_prompt_variants_fn=_sam3_prompt_variants,
    sam3_text_payloads_fn=_sam3_text_payloads_from_state,
    active_sam3_score_thr=_AGENT_ACTIVE_SAM3_SCORE_THR,
    active_sam3_mask_thr=_AGENT_ACTIVE_SAM3_MASK_THR,
    grid_overlap_ratio_default=PREPASS_GRID_OVERLAP_RATIO,
    resolve_classifier_path_fn=lambda path_str: _resolve_agent_clip_classifier_path_impl(
        path_str,
        allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
        allowed_exts=CLASSIFIER_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root_impl,
        http_exception_cls=HTTPException,
    ),
    load_classifier_head_fn=lambda classifier_path: _load_clip_head_from_classifier_impl(
        classifier_path,
        joblib_load_fn=joblib.load,
        http_exception_cls=HTTPException,
        clip_head_background_indices_fn=_clip_head_background_indices,
        resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
        infer_clip_model_fn=lambda embed_dim, active_name=None: _infer_clip_model_from_embedding_dim_impl(
            embed_dim, active_name=active_name or clip_model_name or DEFAULT_CLIP_MODEL
        ),
        active_clip_model_name=clip_model_name,
        default_clip_model=DEFAULT_CLIP_MODEL,
        logger=logger,
    ),
    active_classifier_head=active_classifier_head,
    background_from_head_fn=_agent_background_classes_from_head,
    sanitize_fn=_agent_sanitize_detection_items,
    default_iou=PREPASS_CLUSTER_IOU,
    select_exemplars_fn=lambda *args, **kwargs: _agent_select_similarity_exemplars(*args, **kwargs),
    run_similarity_global_fn=lambda *args, **kwargs: _agent_run_similarity_global(
        *args,
        **kwargs,
        sam3_similarity_fn=lambda **inner: _agent_tool_sam3_similarity(**inner),
    ),
    run_similarity_windowed_fn=lambda *args, **kwargs: _agent_run_similarity_expansion(
        *args,
        **kwargs,
        sam3_similarity_fn=lambda **inner: _agent_tool_sam3_similarity(**inner),
        grid_overlap_ratio_default=PREPASS_GRID_OVERLAP_RATIO,
    ),
    finalize_provenance_fn=_agent_finalize_provenance,
    caption_request_cls=QwenCaptionRequest,
    qwen_caption_fn=lambda payload: qwen_caption(payload),
    sanitize_caption_fn=_sanitize_qwen_caption_impl,
    label_counts_fn=_agent_label_counts_summary,
    qwen_bbox_to_xyxy_fn=_qwen_bbox_to_xyxy,
    xyxy_to_bbox_fn=_xyxy_to_qwen_bbox,
    grid_cell_for_window_bbox_fn=_agent_grid_cell_for_window_bbox,
    readable_format_bbox_fn=_agent_readable_format_bbox,
    unload_non_qwen_fn=lambda: _unload_non_qwen_runtimes_impl(
        predictor_manager=predictor_manager,
        unload_sam3_text_fn=_unload_sam3_text_runtime,
        suspend_clip_fn=_suspend_clip_backbone,
        unload_dinov3_fn=_unload_dinov3_backbone,
        unload_detector_fn=_unload_detector_inference,
        torch_module=torch,
        logger=logger,
    ),
    caption_window_hook=_CAPTION_WINDOW_HOOK,
    http_exception_cls=HTTPException,
    http_503_code=HTTP_503_SERVICE_UNAVAILABLE,
)


def _agent_select_similarity_exemplars(
    payload: QwenPrepassRequest,
    *,
    detections: List[Dict[str, Any]],
    trace_readable: Optional[Callable[[str], None]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    raw_min_score = payload.similarity_min_exemplar_score
    if raw_min_score is None:
        min_score = 0.6
    else:
        try:
            min_score = float(raw_min_score)
        except (TypeError, ValueError):
            min_score = 0.6
    if not math.isfinite(min_score):
        min_score = 0.6
    min_score = max(0.0, min(1.0, min_score))
    return _agent_select_similarity_exemplars_impl(
        min_score,
        detections=detections,
        max_per_label=payload.similarity_exemplar_count,
        strategy=payload.similarity_exemplar_strategy,
        seed=payload.similarity_exemplar_seed,
        exemplar_fraction=payload.similarity_exemplar_fraction,
        exemplar_min=payload.similarity_exemplar_min,
        exemplar_max=payload.similarity_exemplar_max,
        source_quota=payload.similarity_exemplar_source_quota,
        trace_readable=trace_readable,
    )


def _run_prepass_annotation_qwen(
    payload: QwenPrepassRequest,
    *,
    trace_sink: Optional[Any] = None,
    cancel_event: Optional[threading.Event] = None,
) -> QwenPrepassResponse:
    global QWEN_CAPTION_CACHE_LIMIT
    # Prepass-only mode is enforced; no agentic review loop is executed.
    payload = payload.copy(update={"prepass_only": True})
    _require_sam3_for_prepass_impl(
        bool(payload.enable_sam3_text),
        bool(payload.enable_sam3_similarity),
        sam3_import_error=SAM3_NATIVE_IMAGE_IMPORT_ERROR,
        build_sam3_image_model=build_sam3_image_model,
        sam3_image_processor=Sam3ImageProcessor,
    )
    if int(QWEN_CAPTION_CACHE_LIMIT or 0) < 1:
        QWEN_CAPTION_CACHE_LIMIT = 1
    pil_img, _, token = resolve_image_payload(payload.image_base64, payload.image_token, None)
    trace_path: Optional[str] = None
    full_trace_path: Optional[str] = None
    readable_trace_path: Optional[str] = None
    latest_readable_path: Optional[str] = None
    trace_file = QWEN_PREPASS_TRACE_ROOT / f"prepass_{int(time.time())}_{uuid.uuid4().hex[:8]}.jsonl"
    full_trace_file = QWEN_PREPASS_FULL_TRACE_ROOT / f"prepass_full_{int(time.time())}_{uuid.uuid4().hex[:8]}.jsonl"
    readable_trace_file = QWEN_PREPASS_READABLE_TRACE_ROOT / f"prepass_readable_{int(time.time())}_{uuid.uuid4().hex[:8]}.log"
    try:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_path = str(trace_file)
    except Exception:
        trace_path = None
    try:
        full_trace_file.parent.mkdir(parents=True, exist_ok=True)
        full_trace_path = str(full_trace_file)
    except Exception:
        full_trace_path = None
    try:
        readable_trace_file.parent.mkdir(parents=True, exist_ok=True)
        readable_trace_path = str(readable_trace_file)
    except Exception:
        readable_trace_path = None
    latest_full_path: Optional[str] = None
    try:
        latest_full_file = QWEN_PREPASS_FULL_TRACE_LATEST
        latest_full_file.parent.mkdir(parents=True, exist_ok=True)
        if not latest_full_file.exists():
            latest_full_file.write_text("", encoding="utf-8")
        latest_full_path = str(latest_full_file)
    except Exception:
        latest_full_path = None
    try:
        latest_readable_file = QWEN_PREPASS_READABLE_TRACE_LATEST
        latest_readable_file.parent.mkdir(parents=True, exist_ok=True)
        if not latest_readable_file.exists():
            latest_readable_file.write_text("", encoding="utf-8")
        latest_readable_path = str(latest_readable_file)
    except Exception:
        latest_readable_path = None

    def _trace_write(record: Dict[str, Any]) -> None:
        if not trace_path:
            return
        record["ts"] = record.get("ts") or time.time()
        try:
            with open(trace_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            pass
    def _trace_write_full(record: Dict[str, Any]) -> None:
        if not full_trace_path and not latest_full_path:
            return
        record["ts"] = record.get("ts") or time.time()
        try:
            payload = _agent_trace_full_jsonable(record)
        except Exception:
            payload = {"error": "trace_encode_failed", "record": str(record)}
        if full_trace_path:
            try:
                with open(full_trace_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass
        if latest_full_path and latest_full_path != full_trace_path:
            try:
                with open(latest_full_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass
    def _trace_write_readable(line: str) -> None:
        if not line or (not readable_trace_path and not latest_readable_path):
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        text = f"{stamp} {line}\n"
        if readable_trace_path:
            try:
                with open(readable_trace_path, "a", encoding="utf-8") as handle:
                    handle.write(text)
            except Exception:
                pass
        if latest_readable_path and latest_readable_path != readable_trace_path:
            try:
                with open(latest_readable_path, "a", encoding="utf-8") as handle:
                    handle.write(text)
            except Exception:
                pass
        if PREPASS_READABLE_TO_CONSOLE:
            logging.getLogger("prepass.readable").info(line)
    global _AGENT_ACTIVE_IMAGE_TOKEN, _AGENT_ACTIVE_IMAGE_BASE64, _AGENT_ACTIVE_DATASET_ID
    global _AGENT_ACTIVE_LABELMAP, _AGENT_ACTIVE_GLOSSARY, _AGENT_ACTIVE_INSPECTED_WINDOWS
    global _AGENT_ACTIVE_CLASSIFIER_ID, _AGENT_ACTIVE_TIGHTEN_FP
    global _AGENT_ACTIVE_DETECTOR_CONF, _AGENT_ACTIVE_SAM3_SCORE_THR, _AGENT_ACTIVE_SAM3_MASK_THR
    global _AGENT_ACTIVE_CLASSIFIER_MIN_PROB, _AGENT_ACTIVE_CLASSIFIER_MARGIN
    global _AGENT_ACTIVE_CLASSIFIER_BG_MARGIN, _AGENT_ACTIVE_SCORELESS_IOU
    global _AGENT_ACTIVE_DETECTIONS, _AGENT_ACTIVE_CLUSTERS, _AGENT_ACTIVE_GRID
    global _AGENT_ACTIVE_GRID_IMAGE, _AGENT_ACTIVE_OVERLAY_IMAGE
    global _AGENT_LAST_SUBMIT_DETECTIONS, _AGENT_PENDING_CLASSIFY_IDS
    global _AGENT_TRACE_FULL_WRITER, _AGENT_TRACE_READABLE_WRITER, _AGENT_PREPASS_COMPLETE
    def _agent_unit_float(value: Optional[float], fallback: Optional[float]) -> Optional[float]:
        if value is None:
            return fallback
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return fallback

    tighten_fp = bool(payload.tighten_fp if payload.tighten_fp is not None else True)
    detector_conf = _agent_unit_float(
        payload.detector_conf,
        PREPASS_TIGHT_DEFAULT_DETECTOR_CONF,
    )
    sam3_score_thr = _agent_unit_float(
        payload.sam3_score_thr,
        PREPASS_TIGHT_DEFAULT_SAM3_SCORE,
    )
    sam3_mask_thr = _agent_unit_float(
        payload.sam3_mask_threshold,
        PREPASS_TIGHT_DEFAULT_SAM3_MASK,
    )
    classifier_min_prob = _agent_unit_float(
        payload.classifier_min_prob,
        PREPASS_TIGHT_DEFAULT_CLASSIFIER_MIN_PROB,
    )
    classifier_margin = _agent_unit_float(
        payload.classifier_margin,
        PREPASS_TIGHT_DEFAULT_CLASSIFIER_MARGIN,
    )
    classifier_bg_margin = _agent_unit_float(
        payload.classifier_bg_margin,
        PREPASS_TIGHT_DEFAULT_CLASSIFIER_BG_MARGIN,
    )
    scoreless_iou = _agent_unit_float(
        payload.scoreless_iou,
        PREPASS_TIGHT_DEFAULT_SCORELESS_IOU if tighten_fp else 0.0,
    )

    _AGENT_ACTIVE_IMAGE_TOKEN = token
    _AGENT_ACTIVE_IMAGE_BASE64 = payload.image_base64
    _AGENT_ACTIVE_DATASET_ID = payload.dataset_id
    _AGENT_TRACE_FULL_WRITER = _trace_write_full if full_trace_path else None
    _AGENT_TRACE_READABLE_WRITER = _trace_write_readable if readable_trace_path or latest_readable_path else None
    _AGENT_ACTIVE_TIGHTEN_FP = tighten_fp
    _AGENT_ACTIVE_DETECTOR_CONF = detector_conf
    _AGENT_ACTIVE_SAM3_SCORE_THR = sam3_score_thr
    _AGENT_ACTIVE_SAM3_MASK_THR = sam3_mask_thr
    _AGENT_ACTIVE_CLASSIFIER_MIN_PROB = classifier_min_prob if tighten_fp else None
    _AGENT_ACTIVE_CLASSIFIER_MARGIN = classifier_margin if tighten_fp else None
    _AGENT_ACTIVE_CLASSIFIER_BG_MARGIN = classifier_bg_margin if tighten_fp else None
    _AGENT_ACTIVE_SCORELESS_IOU = scoreless_iou if tighten_fp else 0.0
    img_w, img_h = pil_img.size
    labelmap: List[str] = []
    glossary = ""
    if payload.labelmap:
        labelmap = [str(x).strip() for x in payload.labelmap if str(x).strip()]
        if labelmap:
            glossary = _default_agent_glossary_for_labelmap(labelmap)
    if not labelmap:
        labelmap, glossary = _agent_load_labelmap_meta(payload.dataset_id)
    labelmap = labelmap or []
    warnings: List[str] = []
    if not labelmap:
        warnings.append("labelmap_missing")
    if payload.labelmap_glossary is not None:
        glossary = _normalize_labelmap_glossary(payload.labelmap_glossary)
    _AGENT_ACTIVE_LABELMAP = labelmap
    _AGENT_ACTIVE_GLOSSARY = glossary
    _AGENT_ACTIVE_INSPECTED_WINDOWS = set()
    _agent_reset_registries()
    classifier_id_for_run = payload.classifier_id
    if not classifier_id_for_run and not isinstance(active_classifier_head, dict):
        classifier_id_for_run = _agent_default_classifier_for_dataset(payload.dataset_id)
    _AGENT_ACTIVE_CLASSIFIER_ID = classifier_id_for_run
    head: Optional[Dict[str, Any]] = None
    if classifier_id_for_run:
        classifier_path = _resolve_agent_clip_classifier_path_impl(
            classifier_id_for_run,
            allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
            allowed_exts=CLASSIFIER_ALLOWED_EXTS,
            path_is_within_root_fn=_path_is_within_root_impl,
            http_exception_cls=HTTPException,
        )
        if classifier_path is not None:
            head = _load_clip_head_from_classifier_impl(
                classifier_path,
                joblib_load_fn=joblib.load,
                http_exception_cls=HTTPException,
                clip_head_background_indices_fn=_clip_head_background_indices,
                resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
                infer_clip_model_fn=_infer_clip_model_from_embedding_dim_impl,
                active_clip_model_name=clip_model_name,
                default_clip_model=DEFAULT_CLIP_MODEL,
                logger=logger,
            )
    elif isinstance(active_classifier_head, dict):
        head = active_classifier_head
    if isinstance(head, dict):
        head = dict(head)
        if _AGENT_ACTIVE_TIGHTEN_FP:
            min_prob = float(head.get("min_prob") or 0.5)
            margin = float(head.get("margin") or 0.0)
            bg_margin = float(head.get("background_margin") or 0.0)
            if _AGENT_ACTIVE_CLASSIFIER_MIN_PROB is not None:
                min_prob = max(min_prob, _AGENT_ACTIVE_CLASSIFIER_MIN_PROB)
            if _AGENT_ACTIVE_CLASSIFIER_MARGIN is not None:
                margin = max(margin, _AGENT_ACTIVE_CLASSIFIER_MARGIN)
            if _AGENT_ACTIVE_CLASSIFIER_BG_MARGIN is not None:
                bg_margin = max(bg_margin, _AGENT_ACTIVE_CLASSIFIER_BG_MARGIN)
            head["min_prob"] = min_prob
            head["margin"] = margin
            head["background_margin"] = bg_margin
            _trace_write(
                {
                    "type": "precision_profile",
                    "tighten_fp": True,
                    "detector_conf": _AGENT_ACTIVE_DETECTOR_CONF,
                    "sam3_score_thr": _AGENT_ACTIVE_SAM3_SCORE_THR,
                    "sam3_mask_thr": _AGENT_ACTIVE_SAM3_MASK_THR,
                    "classifier_min_prob": head["min_prob"],
                    "classifier_margin": head["margin"],
                    "classifier_bg_margin": head["background_margin"],
                    "scoreless_iou": _AGENT_ACTIVE_SCORELESS_IOU,
                }
            )
            _trace_write_full(
                {
                    "type": "precision_profile",
                    "tighten_fp": True,
                    "detector_conf": _AGENT_ACTIVE_DETECTOR_CONF,
                    "sam3_score_thr": _AGENT_ACTIVE_SAM3_SCORE_THR,
                    "sam3_mask_thr": _AGENT_ACTIVE_SAM3_MASK_THR,
                    "classifier_min_prob": head["min_prob"],
                    "classifier_margin": head["margin"],
                    "classifier_bg_margin": head["background_margin"],
                    "scoreless_iou": _AGENT_ACTIVE_SCORELESS_IOU,
                }
            )
            _trace_write_readable(
                "precision_profile: "
                f"detector_conf={_AGENT_ACTIVE_DETECTOR_CONF or 0:.2f} "
                f"sam3_score_thr={_AGENT_ACTIVE_SAM3_SCORE_THR or 0:.2f} "
                f"sam3_mask_thr={_AGENT_ACTIVE_SAM3_MASK_THR or 0:.2f} "
                f"classifier_min_prob={head['min_prob']:.2f} "
                f"margin={head['margin']:.2f} "
                f"bg_margin={head['background_margin']:.2f} "
                f"scoreless_iou={(_AGENT_ACTIVE_SCORELESS_IOU or 0):.2f}"
            )
    background = _agent_background_classes_from_head(head)
    _trace_write({"type": "start", "payload": _agent_trace_sanitize_payload(payload, token)})
    try:
        _trace_write_full({"type": "start", "payload": payload.dict()})
    except Exception:
        _trace_write_full({"type": "start", "payload": str(payload)})
    readable_model_id = payload.model_id or (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME
    start_title = f"IMAGE START {payload.image_name or 'unknown'}"
    _trace_write_readable(_agent_readable_banner(start_title, fill="="))
    _trace_write_readable(
        "start: "
        f"dataset_id={payload.dataset_id or 'none'} "
        f"image={payload.image_name or 'unknown'} "
        f"model={readable_model_id} "
        f"variant={payload.model_variant or 'auto'}"
    )
    base_model_id = (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME
    desired_variant = (payload.model_variant or "auto").strip()
    if payload.model_id:
        model_id_override = payload.model_id
    elif desired_variant in {"Instruct", "Thinking"}:
        model_id_override = _resolve_qwen_variant_model_id_impl(base_model_id, desired_variant)
    else:
        model_id_override = base_model_id
    grid_spec: Optional[Dict[str, Any]] = _agent_grid_spec_for_payload(payload, img_w, img_h)
    _AGENT_ACTIVE_GRID = grid_spec
    trace: List[AgentTraceEvent] = []
    step_id = 0

    def _append_trace(phase: str, summary: str, counts: Optional[Dict[str, int]] = None) -> None:
        nonlocal step_id
        step_id += 1
        event = AgentTraceEvent(
            step_id=step_id,
            phase=phase,
            summary=summary,
            counts=counts,
            timestamp=time.time(),
        )
        trace.append(event)
        if trace_sink:
            trace_sink(event)

    _trace_write_readable(_agent_readable_banner("DEEP PREPASS START", fill="-"))
    deep_prepass = _agent_run_deep_prepass(
        payload,
        pil_img=pil_img,
        image_token=token,
        labelmap=labelmap,
        glossary=glossary,
        trace_writer=_trace_write,
        trace_full_writer=_trace_write_full,
        trace_readable=_trace_write_readable,
    )
    _trace_write_readable(_agent_readable_banner("DEEP PREPASS END", fill="-"))
    deep_detections = list(deep_prepass.get("detections") or [])
    deep_warnings = list(deep_prepass.get("warnings") or [])
    if deep_warnings:
        warnings.extend(deep_warnings)
    if payload.ensemble_enabled and payload.ensemble_job_id:
        deep_detections = _agent_apply_ensemble_filter(
            deep_detections,
            dataset_id=payload.dataset_id,
            image_name=payload.image_name,
            classifier_id=classifier_id_for_run,
            job_id=payload.ensemble_job_id,
            prepass_provenance=deep_prepass.get("provenance"),
            trace_writer=_trace_write,
            trace_full_writer=_trace_write_full,
            trace_readable=_trace_write_readable,
            warnings=warnings,
        )
    _append_trace("merge", "deep_prepass_complete", counts={"detections": len(deep_detections)})

    caption_text, caption_entries = _agent_run_deep_prepass_caption(
        payload,
        pil_img=pil_img,
        image_token=token,
        detections=deep_detections,
        model_id_override=model_id_override,
        glossary=glossary,
        grid_for_log=grid_spec,
        trace_writer=_trace_write,
        trace_full_writer=_trace_write_full,
        trace_readable=_trace_write_readable,
    )
    _AGENT_ACTIVE_OVERALL_CAPTION = caption_text
    _AGENT_ACTIVE_WINDOWED_CAPTIONS = caption_entries

    source_summary = _agent_format_source_counts(_agent_source_counts(deep_detections))
    _trace_write(
        {
            "type": "deep_prepass_summary",
            "detections": len(deep_detections),
            "source_summary": source_summary,
            "warnings": deep_warnings,
        }
    )
    _trace_write_full(
        {
            "type": "deep_prepass_summary",
            "detections": len(deep_detections),
            "source_summary": source_summary,
            "warnings": deep_warnings,
        }
    )

    detections: List[Dict[str, Any]] = []
    if deep_detections:
        _agent_register_detections(
            deep_detections,
            img_w=img_w,
            img_h=img_h,
            grid=grid_spec,
            labelmap=labelmap,
            background=background,
            source_override=None,
        )
        detections = list(_AGENT_ACTIVE_CLUSTERS or [])
    _agent_set_active_clusters(detections)
    _AGENT_PREPASS_COMPLETE = True

    overlay_image: Optional[Image.Image] = None
    overlay_radius: Optional[int] = None
    use_overlay = bool(payload.use_detection_overlay if payload.use_detection_overlay is not None else True)
    if payload.overlay_dot_radius is not None:
        try:
            overlay_radius = max(1, int(payload.overlay_dot_radius))
        except (TypeError, ValueError):
            overlay_radius = None
    grid_image = _agent_render_grid_overlay(pil_img, grid_spec) if grid_spec else pil_img
    _AGENT_ACTIVE_GRID_IMAGE = grid_image
    if use_overlay and labelmap:
        label_colors = _agent_current_label_colors(labelmap)
        label_prefixes = _agent_current_label_prefixes(labelmap)
        overlay_image = _agent_render_detection_overlay(
            grid_image,
            detections,
            label_colors,
            dot_radius=overlay_radius,
            label_prefixes=label_prefixes,
        )
        _AGENT_ACTIVE_LABEL_COLORS = label_colors
        _AGENT_ACTIVE_LABEL_PREFIXES = label_prefixes
        _AGENT_ACTIVE_OVERLAY_DOT_RADIUS = overlay_radius
        _trace_write(
            {
                "type": "overlay",
                "enabled": True,
                "detections": len(detections),
                "dot_radius": overlay_radius,
                "label_colors": label_colors,
                "grid": grid_spec,
            }
        )
        _trace_write_full(
            {
                "type": "overlay",
                "enabled": True,
                "detections": len(detections),
                "dot_radius": overlay_radius,
                "label_colors": label_colors,
                "grid": grid_spec,
            }
        )
        _trace_write_readable(
            f"deep prepass overlay: detections={len(detections)} "
            f"radius={overlay_radius if overlay_radius is not None else 'auto'}"
        )
    else:
        overlay_image = grid_image
        _trace_write(
            {
                "type": "overlay",
                "enabled": True,
                "detections": len(detections),
                "dot_radius": overlay_radius,
                "grid": grid_spec,
            }
        )
        _trace_write_full(
            {
                "type": "overlay",
                "enabled": True,
                "detections": len(detections),
                "dot_radius": overlay_radius,
                "grid": grid_spec,
            }
        )

    if labelmap:
        _AGENT_ACTIVE_LABEL_COLORS = _agent_current_label_colors(labelmap)
        _AGENT_ACTIVE_LABEL_PREFIXES = _agent_current_label_prefixes(labelmap)
    _AGENT_ACTIVE_OVERLAY_DOT_RADIUS = overlay_radius
    _AGENT_ACTIVE_OVERLAY_IMAGE = overlay_image
    _AGENT_ACTIVE_DETECTIONS = deep_detections

    if not payload.prepass_only:
        warnings.append("prepass_only_enforced")

    final_detections = list(_AGENT_ACTIVE_CLUSTERS or [])
    if payload.prepass_finalize:
        cross_iou = None
        if bool(payload.cross_class_dedupe_enabled):
            try:
                cross_iou = float(
                    payload.cross_class_dedupe_iou if payload.cross_class_dedupe_iou is not None else 0.8
                )
            except (TypeError, ValueError):
                cross_iou = 0.8
            cross_iou = max(0.0, min(1.0, cross_iou))
        submit_result = _agent_tool_submit_annotations(
            image_token=token,
            dataset_id=payload.dataset_id,
            classifier_id=payload.classifier_id,
            include_all=True,
            iou=payload.iou,
            cross_iou=cross_iou,
            max_det=None,
        )
        if isinstance(submit_result, dict) and submit_result.get("detections") is not None:
            final_detections = list(submit_result.get("detections") or [])
            rejected = submit_result.get("rejected")
            if rejected:
                warnings.append(f"prepass_finalize_rejected:{rejected}")
    summary_lines = _agent_detection_summary_lines(
        final_detections,
        grid=grid_spec,
        img_w=img_w,
        img_h=img_h,
        warnings=warnings,
    )
    _trace_write_readable(_agent_readable_banner("IMAGE END SUMMARY", fill="="))
    for line in summary_lines:
        _trace_write_readable(line)
    _trace_write_readable(_agent_readable_banner("END IMAGE", fill="="))
    _AGENT_TRACE_FULL_WRITER = None
    _AGENT_TRACE_READABLE_WRITER = None
    return QwenPrepassResponse(
        detections=final_detections,
        trace=trace,
        warnings=warnings or None,
        caption=None,
        trace_path=trace_path,
        trace_full_path=full_trace_path,
    )


def _run_prepass_annotation(
    payload: QwenPrepassRequest,
    *,
    trace_sink: Optional[Any] = None,
    cancel_event: Optional[threading.Event] = None,
) -> QwenPrepassResponse:
    return _run_prepass_annotation_qwen(payload, trace_sink=trace_sink, cancel_event=cancel_event)


DEFAULT_QWEN_PROMPT_CONFIG = QwenPromptConfig(
    bbox=QwenPromptSection(
        base_prompt=(
            "Output a JSON formatted list of very tight bounding boxes with coordinates in format (x1,y1,x2,y2) "
            "of detections in this {image_type}. Make a single bounding box for each unique instance of the things we want to detect. "
            "The objects we want to detect are: {items}. {extra_context}"
        ),
        default_image_type="image",
        default_extra_context="Return only JSON, no additional text.",
    ),
    point=QwenPromptSection(
        base_prompt=(
            "Output a JSON formatted list of positive click points with coordinates in format (x,y) for detections in this {image_type}. "
            "Each entry must contain \"point_2d\": [x, y] centered on the object so Segment Anything can turn it into a mask/bbox. "
            "Make one point per object. The objects we want to detect are: {items}. {extra_context}"
        ),
        default_image_type="image",
        default_extra_context="Respond with JSON only.",
    ),
)

qwen_prompt_config = DEFAULT_QWEN_PROMPT_CONFIG.copy(deep=True)


@dataclass
class ClipTrainingJob:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    temp_dir: Optional[str] = None
    images_dir: Optional[str] = None
    labels_dir: Optional[str] = None
    labelmap_path: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)



TRAINING_JOBS: Dict[str, ClipTrainingJob] = {}
TRAINING_JOBS_LOCK = threading.Lock()

QWEN_JOB_ROOT = Path(os.environ.get("QWEN_TRAINING_ROOT", "./uploads/qwen_runs"))
QWEN_JOB_ROOT.mkdir(parents=True, exist_ok=True)
QWEN_DATASET_ROOT = QWEN_JOB_ROOT / "datasets"
QWEN_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
SAM3_JOB_ROOT = Path(os.environ.get("SAM3_TRAINING_ROOT", "./uploads/sam3_runs"))
SAM3_JOB_ROOT.mkdir(parents=True, exist_ok=True)
SAM3_DATASET_ROOT = SAM3_JOB_ROOT / "datasets"
SAM3_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
SAM3_DATASET_META_NAME = "sam3_dataset.json"
YOLO_JOB_ROOT = Path(os.environ.get("YOLO_TRAINING_ROOT", "./uploads/yolo_runs"))
YOLO_JOB_ROOT.mkdir(parents=True, exist_ok=True)
YOLO_MODEL_ROOT = Path(os.environ.get("YOLO_MODEL_ROOT", "./uploads/yolo_models"))
YOLO_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
YOLO_ACTIVE_PATH = YOLO_MODEL_ROOT / "active.json"
YOLO_DATASET_CACHE_ROOT = YOLO_JOB_ROOT / "datasets"
YOLO_DATASET_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
YOLO_RUN_META_NAME = "run.json"
YOLO_KEEP_FILES = {
    "best.pt",
    "results.csv",
    "args.yaml",
    "data.yaml",
    "metrics.json",
    "metrics_series.json",
    "labelmap.txt",
    "head_graft_audit.jsonl",
    YOLO_RUN_META_NAME,
}

RFDETR_JOB_ROOT = Path(os.environ.get("RFDETR_TRAINING_ROOT", "./uploads/rfdetr_runs"))
RFDETR_JOB_ROOT.mkdir(parents=True, exist_ok=True)
RFDETR_MODEL_ROOT = Path(os.environ.get("RFDETR_MODEL_ROOT", "./uploads/rfdetr_models"))
RFDETR_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
RFDETR_ACTIVE_PATH = RFDETR_MODEL_ROOT / "active.json"
DETECTOR_PREFS_ROOT = Path(os.environ.get("DETECTOR_PREFS_ROOT", "./uploads/detectors"))
DETECTOR_PREFS_ROOT.mkdir(parents=True, exist_ok=True)
DETECTOR_DEFAULT_PATH = DETECTOR_PREFS_ROOT / "default.json"
RFDETR_RUN_META_NAME = "run.json"


def _load_rfdetr_active():
    return _load_rfdetr_active_impl(
        RFDETR_ACTIVE_PATH,
        RFDETR_JOB_ROOT,
        save_active_fn=lambda payload: _save_rfdetr_active_impl(payload, RFDETR_ACTIVE_PATH),
    )
RFDETR_KEEP_FILES = {
    "checkpoint_best_regular.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_total.pth",
    "checkpoint_best_optimized.pt",
    "results.json",
    "metrics_series.json",
    "metrics_plot.png",
    "log.txt",
    "labelmap.txt",
    RFDETR_RUN_META_NAME,
}

YOLO_INFER_LOCK = threading.RLock()
yolo_infer_model: Any = None
yolo_infer_path: Optional[str] = None
yolo_infer_labelmap: List[str] = []
yolo_infer_task: Optional[str] = None
RFDETR_INFER_LOCK = threading.RLock()
rfdetr_infer_model: Any = None
rfdetr_infer_path: Optional[str] = None
rfdetr_infer_labelmap: List[str] = []
rfdetr_infer_task: Optional[str] = None
rfdetr_infer_variant: Optional[str] = None


def _set_yolo_infer_state(
    model: Any,
    path: Optional[str],
    labelmap: List[str],
    task: Optional[str],
) -> None:
    global yolo_infer_model, yolo_infer_path, yolo_infer_labelmap, yolo_infer_task
    state = {
        "model": yolo_infer_model,
        "path": yolo_infer_path,
        "labelmap": yolo_infer_labelmap,
        "task": yolo_infer_task,
    }
    _set_yolo_infer_state_impl(model, path, labelmap, task, state=state)
    yolo_infer_model = state["model"]
    yolo_infer_path = state["path"]
    yolo_infer_labelmap = state["labelmap"]
    yolo_infer_task = state["task"]


def _set_rfdetr_infer_state(
    model: Any,
    path: Optional[str],
    labelmap: List[str],
    task: Optional[str],
    variant: Optional[str],
) -> None:
    global rfdetr_infer_model, rfdetr_infer_path, rfdetr_infer_labelmap, rfdetr_infer_task, rfdetr_infer_variant
    state = {
        "model": rfdetr_infer_model,
        "path": rfdetr_infer_path,
        "labelmap": rfdetr_infer_labelmap,
        "task": rfdetr_infer_task,
        "variant": rfdetr_infer_variant,
    }
    _set_rfdetr_infer_state_impl(model, path, labelmap, task, variant, state=state)
    rfdetr_infer_model = state["model"]
    rfdetr_infer_path = state["path"]
    rfdetr_infer_labelmap = state["labelmap"]
    rfdetr_infer_task = state["task"]
    rfdetr_infer_variant = state["variant"]
YOLO_VARIANTS = [
    {"id": "yolov8n", "label": "YOLOv8 Nano", "task": "detect"},
    {"id": "yolov8s", "label": "YOLOv8 Small", "task": "detect"},
    {"id": "yolov8m", "label": "YOLOv8 Medium", "task": "detect"},
    {"id": "yolov8l", "label": "YOLOv8 Large", "task": "detect"},
    {"id": "yolov8x", "label": "YOLOv8 XLarge", "task": "detect"},
    {"id": "yolov8n-seg", "label": "YOLOv8 Nano (seg)", "task": "segment"},
    {"id": "yolov8s-seg", "label": "YOLOv8 Small (seg)", "task": "segment"},
    {"id": "yolov8m-seg", "label": "YOLOv8 Medium (seg)", "task": "segment"},
    {"id": "yolov8l-seg", "label": "YOLOv8 Large (seg)", "task": "segment"},
    {"id": "yolov8x-seg", "label": "YOLOv8 XLarge (seg)", "task": "segment"},
    {"id": "yolov8n-p2", "label": "YOLOv8 Nano (P2)", "task": "detect"},
    {"id": "yolov8s-p2", "label": "YOLOv8 Small (P2)", "task": "detect"},
    {"id": "yolov8m-p2", "label": "YOLOv8 Medium (P2)", "task": "detect"},
    {"id": "yolov8l-p2", "label": "YOLOv8 Large (P2)", "task": "detect"},
    {"id": "yolov8x-p2", "label": "YOLOv8 XLarge (P2)", "task": "detect"},
]
RFDETR_VARIANTS = [
    {"id": "rfdetr-nano", "label": "RF-DETR Nano", "task": "detect"},
    {"id": "rfdetr-small", "label": "RF-DETR Small", "task": "detect"},
    {"id": "rfdetr-medium", "label": "RF-DETR Medium", "task": "detect"},
    {"id": "rfdetr-base", "label": "RF-DETR Base", "task": "detect"},
    {"id": "rfdetr-large", "label": "RF-DETR Large", "task": "detect"},
    {"id": "rfdetr-seg-preview", "label": "RF-DETR Seg Preview", "task": "segment"},
]
DATASET_REGISTRY_ROOT = Path(os.environ.get("DATASET_ROOT", "./uploads/datasets"))
DATASET_REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_META_NAME = "dataset.json"
PROMPT_HELPER_JOB_ROOT = Path(os.environ.get("SAM3_PROMPT_HELPER_ROOT", "./uploads/prompt_helper_jobs"))
PROMPT_HELPER_JOB_ROOT.mkdir(parents=True, exist_ok=True)
SEG_BUILDER_ROOT = Path(os.environ.get("SEGMENTATION_ROOT", "./uploads/seg_runs"))
SEG_BUILDER_ROOT.mkdir(parents=True, exist_ok=True)
SAM3_REPO_ROOT = Path(__file__).resolve().parent.resolve()
SAM3_VENDOR_ROOT = SAM3_REPO_ROOT / "sam3"
SAM3_PACKAGE_ROOT = SAM3_VENDOR_ROOT / "sam3"
SAM3_CONFIG_TEMPLATE = SAM3_REPO_ROOT / "sam3_local" / "local_yolo_ft.yaml"
SAM3_GENERATED_CONFIG_DIR = SAM3_PACKAGE_ROOT / "train/configs/generated"
SAM3_GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SAM3_BPE_PATH = SAM3_VENDOR_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
SAM3_MAX_LOG_LINES = 500
SAM3_MAX_METRIC_POINTS = 2000
SAM3_STORAGE_SCOPES = {"all", "checkpoints", "logs", "tensorboard", "dumps"}
YOLO_MAX_LOG_LINES = 300
YOLO_HEAD_GRAFT_PATCHED = False


@dataclass
class QwenTrainingJob:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    config: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class QwenDatasetUploadJob:
    job_id: str
    root_dir: Path
    run_name: Optional[str] = None
    train_count: int = 0
    val_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


QWEN_DATASET_UPLOADS: Dict[str, QwenDatasetUploadJob] = {}
QWEN_DATASET_UPLOADS_LOCK = threading.Lock()


@dataclass
class Sam3TrainingJob:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    config: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    process: Optional[subprocess.Popen] = None
    log_seq: int = 0


@dataclass
class YoloTrainingJob:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    config: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class YoloHeadGraftJob:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    config: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread_ident: Optional[int] = None


@dataclass
class RfDetrTrainingJob:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    config: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class SegmentationBuildJob:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    config: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class PromptHelperJob:
    job_id: str
    status: str = "queued"
    message: str = "Queued"
    progress: float = 0.0
    request: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0
    completed_steps: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class AgentMiningJob:
    job_id: str
    status: str = "queued"
    message: str = "Queued"
    progress: float = 0.0
    request: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)




QWEN_TRAINING_JOBS: Dict[str, QwenTrainingJob] = {}
QWEN_TRAINING_JOBS_LOCK = threading.Lock()
SAM3_TRAINING_JOBS: Dict[str, Sam3TrainingJob] = {}
SAM3_TRAINING_JOBS_LOCK = threading.Lock()
YOLO_TRAINING_JOBS: Dict[str, YoloTrainingJob] = {}
YOLO_TRAINING_JOBS_LOCK = threading.Lock()
YOLO_HEAD_GRAFT_JOBS: Dict[str, YoloHeadGraftJob] = {}
YOLO_HEAD_GRAFT_JOBS_LOCK = threading.Lock()
RFDETR_TRAINING_JOBS: Dict[str, RfDetrTrainingJob] = {}
RFDETR_TRAINING_JOBS_LOCK = threading.Lock()
SEGMENTATION_BUILD_JOBS: Dict[str, SegmentationBuildJob] = {}
SEGMENTATION_BUILD_JOBS_LOCK = threading.Lock()
PROMPT_HELPER_JOBS: Dict[str, PromptHelperJob] = {}
PROMPT_HELPER_JOBS_LOCK = threading.Lock()
AGENT_MINING_JOBS: Dict[str, AgentMiningJob] = {}
AGENT_MINING_JOBS_LOCK = threading.Lock()
CALIBRATION_JOBS: Dict[str, CalibrationJob] = {}
CALIBRATION_JOBS_LOCK = threading.Lock()
UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)
GLOSSARY_LIBRARY_ROOT = UPLOAD_ROOT / "glossaries"
GLOSSARY_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
PREPASS_RECIPE_ROOT = UPLOAD_ROOT / "prepass_recipes"
PREPASS_RECIPE_ROOT.mkdir(parents=True, exist_ok=True)
PREPASS_RECIPE_META = "prepass.meta.json"
PREPASS_RECIPE_ASSETS = "assets"
PREPASS_RECIPE_SCHEMA_VERSION = 1
PREPASS_RECIPE_TMP_ROOT = UPLOAD_ROOT / "tmp_prepass_recipes"
PREPASS_RECIPE_TMP_ROOT.mkdir(parents=True, exist_ok=True)
PREPASS_RECIPE_EXPORT_ROOT = Path(
    os.environ.get("PREPASS_RECIPE_EXPORT_ROOT", str(UPLOAD_ROOT / "prepass_recipe_exports"))
)
PREPASS_RECIPE_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
PROMPT_HELPER_PRESET_ROOT = UPLOAD_ROOT / "prompt_helper_presets"
PROMPT_HELPER_PRESET_ROOT.mkdir(parents=True, exist_ok=True)


CLIP_DATASET_UPLOAD_ROOT = UPLOAD_ROOT / "clip_dataset_uploads"
CLIP_DATASET_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_UPLOAD_ROOT = UPLOAD_ROOT / "dataset_uploads"
DATASET_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
CLIP_NEGATIVE_REPLAY_ROOT = UPLOAD_ROOT / "clip_negative_replay"
CLIP_NEGATIVE_REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
QWEN_PREPASS_TRACE_ROOT = UPLOAD_ROOT / "qwen_prepass_traces"
QWEN_PREPASS_TRACE_ROOT.mkdir(parents=True, exist_ok=True)
QWEN_PREPASS_FULL_TRACE_ROOT = UPLOAD_ROOT / "qwen_prepass_traces_full"
QWEN_PREPASS_FULL_TRACE_ROOT.mkdir(parents=True, exist_ok=True)
QWEN_PREPASS_FULL_TRACE_LATEST = QWEN_PREPASS_FULL_TRACE_ROOT / "latest.jsonl"
LOG_ROOT = Path("logs")
LOG_ROOT.mkdir(parents=True, exist_ok=True)
QWEN_PREPASS_READABLE_TRACE_ROOT = LOG_ROOT / "prepass_readable"
QWEN_PREPASS_READABLE_TRACE_ROOT.mkdir(parents=True, exist_ok=True)
QWEN_PREPASS_READABLE_TRACE_LATEST = QWEN_PREPASS_READABLE_TRACE_ROOT / "latest.log"
CALIBRATION_ROOT = UPLOAD_ROOT / "calibration_jobs"
CALIBRATION_CACHE_ROOT = UPLOAD_ROOT / "calibration_cache"
CALIBRATION_FEATURES_VERSION = 7
CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)


def _prune_job_registry(registry: Dict[str, Any], lock: threading.Lock, ttl_hours: Optional[int] = None) -> None:
    if ttl_hours is None:
        ttl_hours = JOB_REGISTRY_TTL_HOURS
    ttl_seconds = max(0, ttl_hours) * 3600
    if ttl_seconds == 0:
        return
    now = time.time()
    terminal = {"completed", "failed", "cancelled"}
    with lock:
        to_delete: List[str] = []
        for job_id, job in list(registry.items()):
            status = getattr(job, "status", "")
            updated = getattr(job, "updated_at", getattr(job, "created_at", now))
            if status in {"running", "queued", "cancelling"}:
                continue
            if status and status not in terminal:
                continue
            try:
                if now - float(updated) > ttl_seconds:
                    to_delete.append(job_id)
            except Exception:
                continue
        for job_id in to_delete:
            registry.pop(job_id, None)


JOB_REGISTRY_TTL_HOURS = _env_int("JOB_REGISTRY_TTL_HOURS", 72)
STAGING_TTL_HOURS = _env_int("STAGING_TTL_HOURS", 24)
AGENT_MINING_ROOT = UPLOAD_ROOT / "agent_mining"
AGENT_MINING_ROOT.mkdir(parents=True, exist_ok=True)
AGENT_MINING_JOB_ROOT = AGENT_MINING_ROOT / "jobs"
AGENT_MINING_JOB_ROOT.mkdir(parents=True, exist_ok=True)
AGENT_MINING_CACHE_ROOT = AGENT_MINING_ROOT / "cache"
AGENT_MINING_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
AGENT_MINING_META_ROOT = AGENT_MINING_ROOT / "meta"
AGENT_MINING_META_ROOT.mkdir(parents=True, exist_ok=True)
AGENT_MINING_DET_CACHE_ROOT = AGENT_MINING_ROOT / "detections"
AGENT_MINING_DET_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
AGENT_MINING_RECIPES_ROOT = AGENT_MINING_ROOT / "recipes"
AGENT_MINING_RECIPES_ROOT.mkdir(parents=True, exist_ok=True)
AGENT_MINING_CASCADES_ROOT = AGENT_MINING_ROOT / "cascades"
AGENT_MINING_CASCADES_ROOT.mkdir(parents=True, exist_ok=True)


def _enforce_agent_mining_cache_limits(cache_root: Path, allow_when_running: bool = False) -> None:
    """Prune agent mining cache by TTL and size caps."""
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    if not allow_when_running:
        with AGENT_MINING_JOBS_LOCK:
            for job in AGENT_MINING_JOBS.values():
                if getattr(job, "status", None) == "running":
                    return

    now = time.time()
    ttl_seconds = AGENT_MINING_CACHE_TTL_HOURS * 3600 if AGENT_MINING_CACHE_TTL_HOURS > 0 else 0

    def _iter_entries():
        for entry in cache_root.iterdir():
            try:
                stat = entry.stat()
            except Exception:
                continue
            yield entry, stat.st_mtime

    # TTL purge
    if ttl_seconds > 0:
        for entry, mtime in list(_iter_entries()):
            if now - mtime > ttl_seconds:
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink(missing_ok=True)
                except Exception:
                    pass

    # Size purge
    max_bytes = AGENT_MINING_CACHE_MAX_BYTES
    if max_bytes <= 0:
        return
    try:
        total = _dir_size_bytes_impl(cache_root)
    except Exception:
        return
    if total <= max_bytes:
        return
    entries = sorted(list(_iter_entries()), key=lambda pair: pair[1])
    for entry, _ in entries:
        if total <= max_bytes:
            break
        try:
            if entry.is_dir():
                size = _dir_size_bytes_impl(entry)
                shutil.rmtree(entry, ignore_errors=True)
            else:
                size = entry.stat().st_size
                entry.unlink(missing_ok=True)
            total = max(0, total - size)
        except Exception:
            continue


def _purge_directory(root: Path) -> int:
    """Delete all entries under a directory and return count removed."""
    removed = 0
    try:
        if not root.exists():
            return 0
        for entry in root.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                removed += 1
            except Exception:
                continue
    except Exception:
        return removed
    return removed


def _agent_cache_running_jobs() -> bool:
    """Return True if any agent mining job is currently running."""
    with AGENT_MINING_JOBS_LOCK:
        for job in AGENT_MINING_JOBS.values():
            if getattr(job, "status", None) == "running":
                return True
    return False


MAX_JOB_LOGS = 250
MAX_QWEN_METRIC_POINTS: Optional[int] = None

_job_log = functools.partial(_clip_job_log_impl, max_logs=MAX_JOB_LOGS, logger=logger)
_clip_job_append_metric = functools.partial(_clip_job_append_metric_impl, max_points=2000)
_job_update = functools.partial(_clip_job_update_impl, max_logs=MAX_JOB_LOGS, logger=logger)

_qwen_job_log = functools.partial(_qwen_job_log_impl, max_logs=MAX_JOB_LOGS, logger=logger)
_qwen_job_update = functools.partial(_qwen_job_update_impl, max_logs=MAX_JOB_LOGS, logger=logger)
_qwen_job_append_metric = functools.partial(_qwen_job_append_metric_impl, max_points=MAX_QWEN_METRIC_POINTS)

_sam3_job_log = functools.partial(_sam3_job_log_impl, max_logs=SAM3_MAX_LOG_LINES, logger=logger)
_sam3_job_append_metric = functools.partial(_sam3_job_append_metric_impl, max_points=SAM3_MAX_METRIC_POINTS)
_sam3_job_update = functools.partial(_sam3_job_update_impl, max_logs=SAM3_MAX_LOG_LINES, logger=logger)

_yolo_job_update = functools.partial(_yolo_job_update_impl)
_yolo_job_log = functools.partial(_yolo_job_log_impl, max_logs=YOLO_MAX_LOG_LINES, logger=logger)
_yolo_job_append_metric = functools.partial(_yolo_job_append_metric_impl, max_points=2000)

_yolo_head_graft_audit = functools.partial(_yolo_head_graft_audit_impl, time_fn=time.time)
_yolo_head_graft_job_update = functools.partial(_yolo_head_graft_job_update_impl, audit_fn=_yolo_head_graft_audit)
_yolo_head_graft_job_log = functools.partial(
    _yolo_head_graft_job_log_impl,
    max_logs=YOLO_MAX_LOG_LINES,
    audit_fn=_yolo_head_graft_audit,
    logger=logger,
)

_rfdetr_job_update = functools.partial(_rfdetr_job_update_impl)
_rfdetr_job_log = functools.partial(_rfdetr_job_log_impl, max_logs=MAX_JOB_LOGS, logger=logger)
_rfdetr_job_append_metric = functools.partial(_rfdetr_job_append_metric_impl, max_points=2000)

_seg_job_log = functools.partial(_seg_job_log_impl, max_logs=MAX_JOB_LOGS, logger=logger)
_seg_job_update = functools.partial(_seg_job_update_impl, max_logs=MAX_JOB_LOGS, logger=logger)

_log_qwen_get_request = functools.partial(_log_qwen_get_request_impl, logger=logger)


_list_all_datasets = functools.partial(
    _list_all_datasets_impl,
    prefer_registry=True,
    dataset_registry_root=DATASET_REGISTRY_ROOT,
    sam3_dataset_root=SAM3_DATASET_ROOT,
    qwen_dataset_root=QWEN_DATASET_ROOT,
    load_registry_meta_fn=lambda dataset_dir: _load_registry_dataset_metadata_impl(
        dataset_dir,
        load_json_metadata_fn=_load_json_metadata,
        meta_name=DATASET_META_NAME,
    ),
    load_sam3_meta_fn=lambda dataset_dir: _load_sam3_dataset_metadata_impl(
        dataset_dir,
        meta_name=SAM3_DATASET_META_NAME,
        load_json_metadata_fn=_load_json_metadata,
        persist_metadata_fn=lambda dataset_dir_inner, metadata: _persist_sam3_dataset_metadata_impl(
            dataset_dir_inner,
            metadata,
            meta_name=SAM3_DATASET_META_NAME,
            logger=logger,
        ),
    ),
    load_qwen_meta_fn=lambda dataset_dir: _load_qwen_dataset_metadata_impl(
        dataset_dir,
        meta_name=QWEN_METADATA_FILENAME,
        load_json_metadata_fn=_load_json_metadata,
    ),
    coerce_meta_fn=lambda dataset_dir, raw_meta, source: _coerce_dataset_metadata_impl(
        dataset_dir,
        raw_meta,
        source,
        dataset_context_key="dataset_context",
        compute_dir_signature_fn=_compute_dir_signature_impl,
        persist_metadata_fn=lambda dataset_dir_inner, metadata: _persist_dataset_metadata_impl(
            dataset_dir_inner,
            metadata,
            meta_name=DATASET_META_NAME,
            logger=logger,
        ),
    ),
    yolo_labels_have_polygons_fn=_yolo_labels_have_polygons_impl,
    convert_qwen_dataset_to_coco_fn=_convert_qwen_dataset_to_coco_impl,
    convert_coco_dataset_to_yolo_fn=lambda dataset_root: _convert_coco_dataset_to_yolo_impl(
        dataset_root,
        load_sam3_meta_fn=lambda dataset_dir: _load_sam3_dataset_metadata_impl(
            dataset_dir,
            meta_name=SAM3_DATASET_META_NAME,
            load_json_metadata_fn=_load_json_metadata,
            persist_metadata_fn=lambda dataset_dir_inner, metadata: _persist_sam3_dataset_metadata_impl(
                dataset_dir_inner,
                metadata,
                meta_name=SAM3_DATASET_META_NAME,
                logger=logger,
            ),
        ),
        persist_meta_fn=lambda dataset_dir_inner, metadata: _persist_sam3_dataset_metadata_impl(
            dataset_dir_inner,
            metadata,
            meta_name=SAM3_DATASET_META_NAME,
            logger=logger,
        ),
    ),
    load_dataset_glossary_fn=lambda dataset_dir: _load_dataset_glossary(
        dataset_dir,
        load_sam3_meta=lambda dataset_root: _load_sam3_dataset_metadata_impl(
            dataset_root,
            meta_name=SAM3_DATASET_META_NAME,
            load_json_metadata_fn=_load_json_metadata,
            persist_metadata_fn=lambda dataset_dir_inner, metadata: _persist_sam3_dataset_metadata_impl(
                dataset_dir_inner,
                metadata,
                meta_name=SAM3_DATASET_META_NAME,
                logger=logger,
            ),
        ),
        load_qwen_meta=lambda dataset_root: _load_qwen_dataset_metadata_impl(
            dataset_root,
            meta_name=QWEN_METADATA_FILENAME,
            load_json_metadata_fn=_load_json_metadata,
        ),
    ),
    glossary_preview_fn=_glossary_preview,
    count_caption_labels_fn=_count_caption_labels_impl,
    count_dataset_images_fn=lambda path: _count_dataset_images_impl(path, iter_images_fn=_iter_yolo_images),
    logger=logger,
)

_resolve_sam3_or_qwen_dataset = functools.partial(
    _resolve_sam3_or_qwen_dataset_impl,
    list_all_datasets_fn=_list_all_datasets,
    resolve_dataset_legacy_fn=lambda dataset_id: _resolve_dataset_legacy_impl(
        dataset_id,
        qwen_root=QWEN_DATASET_ROOT,
        sam3_root=SAM3_DATASET_ROOT,
        registry_root=DATASET_REGISTRY_ROOT,
        http_exception_cls=HTTPException,
    ),
)

_load_labelmap_simple = functools.partial(_load_labelmap_simple_impl, load_labelmap_file_fn=_load_labelmap_file)


def _unique_dataset_name(base: str, *, root: Path) -> str:
    candidate = base
    counter = 1
    while (root / candidate).exists():
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def _unwrap_single_root_dir(extract_root: Path) -> Path:
    entries = [p for p in extract_root.iterdir() if p.name not in {"__MACOSX"}]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_root


def list_datasets():
    return _list_all_datasets()


def _persist_dataset_meta(dataset_root: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    _persist_dataset_metadata_impl(dataset_root, meta, meta_name=DATASET_META_NAME, logger=logger)
    return meta


def upload_dataset_zip(
    file: UploadFile,
    dataset_id: Optional[str],
    dataset_type: Optional[str],
    context: Optional[str],
):
    temp_dir = Path(tempfile.mkdtemp(prefix="dataset_upload_"))
    try:
        zip_path = temp_dir / "upload.zip"
        with zip_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        extract_root = temp_dir / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
        source_root = _unwrap_single_root_dir(extract_root)
        base_id = dataset_id or Path(file.filename or "dataset").stem
        base_id = _sanitize_yolo_run_id_impl(base_id) or "dataset"
        dataset_id_final = _unique_dataset_name(base_id, root=DATASET_REGISTRY_ROOT)
        target_root = DATASET_REGISTRY_ROOT / dataset_id_final
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.copytree(source_root, target_root)

        meta_path = target_root / DATASET_META_NAME
        meta = _load_json_metadata(meta_path) or {}
        meta["id"] = dataset_id_final
        meta["label"] = meta.get("label") or dataset_id_final
        if dataset_type:
            meta["type"] = dataset_type
        else:
            meta.setdefault("type", "bbox")
        if context is not None:
            meta["context"] = context
        meta.setdefault("created_at", time.time())
        meta.setdefault("source", "registry")
        labelmap_path = target_root / "labelmap.txt"
        if labelmap_path.exists():
            try:
                labelmap = _load_labelmap_file(labelmap_path)
            except Exception:
                labelmap = []
            if labelmap and not meta.get("classes"):
                meta["classes"] = list(labelmap)
        train_images = target_root / "train" / "images"
        val_images = target_root / "val" / "images"
        train_count = len(_iter_yolo_images(train_images)) if train_images.exists() else 0
        val_count = len(_iter_yolo_images(val_images)) if val_images.exists() else 0
        if train_count or val_count:
            meta["train_count"] = train_count
            meta["val_count"] = val_count
            meta["image_count"] = train_count + val_count
        else:
            meta["image_count"] = _count_dataset_images_impl(target_root, iter_images_fn=_iter_yolo_images)
        meta["signature"] = _compute_dir_signature_impl(target_root)
        _persist_dataset_meta(target_root, meta)
        return meta
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _resolve_dataset_entry(dataset_id: str) -> Dict[str, Any]:
    entry = _resolve_dataset_entry_impl(dataset_id, list_all_datasets_fn=_list_all_datasets)
    if not entry:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="dataset_not_found")
    return entry


def delete_dataset_entry(dataset_id: str):
    entry = _resolve_dataset_entry(dataset_id)
    dataset_root = Path(entry.get("dataset_root") or "").resolve()
    if not dataset_root.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="dataset_root_missing")
    allowed_roots = [DATASET_REGISTRY_ROOT.resolve(), SAM3_DATASET_ROOT.resolve(), QWEN_DATASET_ROOT.resolve()]
    if not any(_path_is_within_root_impl(dataset_root, root) for root in allowed_roots):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_delete_forbidden")
    shutil.rmtree(dataset_root, ignore_errors=True)
    return {"status": "deleted", "id": dataset_id}


def download_dataset_entry(dataset_id: str):
    entry = _resolve_dataset_entry(dataset_id)
    dataset_root = Path(entry.get("dataset_root") or "")
    if not dataset_root.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="dataset_root_missing")
    tmp_dir = Path(tempfile.mkdtemp(prefix="dataset_export_"))
    try:
        zip_path = tmp_dir / f"{dataset_id}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in dataset_root.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(dataset_root)
                zf.write(path, arcname=str(Path(dataset_root.name) / rel))
        return FileResponse(path=str(zip_path), media_type="application/zip", filename=f"{dataset_id}.zip")
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"dataset_export_failed:{exc}") from exc


def _build_qwen_context(labelmap: Sequence[str], context: str) -> str:
    items = ", ".join(labelmap)
    ctx = (context or "").strip()
    if ctx:
        return f"{ctx}\nObjects of interest: {items}."
    return f"Objects of interest: {items}."


def _yolo_label_to_bbox(line: str, *, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except Exception:
        return None
    x1 = int(round((x - w / 2) * img_w))
    y1 = int(round((y - h / 2) * img_h))
    x2 = int(round((x + w / 2) * img_w))
    y2 = int(round((y + h / 2) * img_h))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_qwen_dataset_from_yolo(dataset_id: str):
    entry = _resolve_dataset_entry(dataset_id)
    if not entry.get("yolo_ready"):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_not_yolo_ready")
    dataset_root = Path(entry["dataset_root"])
    labelmap_path = Path(entry.get("yolo_labelmap_path") or dataset_root / "labelmap.txt")
    if not labelmap_path.exists():
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labelmap_missing")
    labelmap = [line.strip() for line in labelmap_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    context = entry.get("context") or ""
    glossary = _load_dataset_glossary(
        dataset_root,
        load_sam3_meta=lambda dataset_dir: _load_sam3_dataset_metadata_impl(
            dataset_dir,
            meta_name=SAM3_DATASET_META_NAME,
            load_json_metadata_fn=_load_json_metadata,
            persist_metadata_fn=lambda dataset_dir_inner, metadata: _persist_sam3_dataset_metadata_impl(
                dataset_dir_inner, metadata, meta_name=SAM3_DATASET_META_NAME, logger=logger
            ),
        ),
        load_qwen_meta=lambda dataset_dir: _load_qwen_dataset_metadata_impl(
            dataset_dir, meta_name=QWEN_METADATA_FILENAME, load_json_metadata_fn=_load_json_metadata
        ),
    )
    if not glossary:
        for _path, meta in _load_dataset_meta_candidates(dataset_root):
            raw = meta.get("labelmap_glossary")
            if raw:
                glossary = _normalize_labelmap_glossary(raw)
                break
    qwen_id = _unique_dataset_name(dataset_id, root=QWEN_DATASET_ROOT)
    target_root = QWEN_DATASET_ROOT / qwen_id
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "train").mkdir(parents=True, exist_ok=True)
    (target_root / "val").mkdir(parents=True, exist_ok=True)
    (target_root / "train").joinpath("annotations.jsonl").write_text("", encoding="utf-8")
    (target_root / "val").joinpath("annotations.jsonl").write_text("", encoding="utf-8")

    context_line = _build_qwen_context(labelmap, context)
    counts = {"train": 0, "val": 0}

    def _process_split(split: str):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            return
        ann_path = target_root / split / "annotations.jsonl"
        with ann_path.open("a", encoding="utf-8") as ann_handle:
            for label_path in labels_dir.rglob("*.txt"):
                image_path = _image_path_for_label_impl(label_path, images_dir)
                if image_path is None or not image_path.exists():
                    continue
                try:
                    img = Image.open(image_path)
                    img_w, img_h = img.size
                    img.close()
                except Exception:
                    continue
                detections = []
                lines = label_path.read_text(encoding="utf-8").splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_idx = int(float(parts[0]))
                    except Exception:
                        continue
                    if cls_idx < 0 or cls_idx >= len(labelmap):
                        continue
                    bbox = _yolo_label_to_bbox(line, img_w=img_w, img_h=img_h)
                    if not bbox:
                        continue
                    x1, y1, x2, y2 = bbox
                    detections.append(
                        {
                            "label": labelmap[cls_idx],
                            "bbox": [x1, y1, x2, y2],
                            "point": [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))],
                        }
                    )
                if not detections:
                    continue
                rel_name = image_path.name
                target_image = target_root / split / rel_name
                if not target_image.exists():
                    shutil.copy2(image_path, target_image)
                payload = {
                    "image": rel_name,
                    "context": context_line,
                    "detections": detections,
                }
                ann_handle.write(json.dumps(payload) + "\n")
                counts[split] += 1

    _process_split("train")
    _process_split("val")

    (target_root / "labelmap.txt").write_text("\n".join(labelmap) + "\n", encoding="utf-8")
    metadata = {
        "id": qwen_id,
        "label": qwen_id,
        "classes": labelmap,
        "context": context,
        "created_at": time.time(),
        "train_count": counts["train"],
        "val_count": counts["val"],
        "image_count": counts["train"] + counts["val"],
        "type": entry.get("type") or "bbox",
        "signature": _compute_dir_signature_impl(target_root),
    }
    if glossary:
        metadata["labelmap_glossary"] = glossary
    (target_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_meta = {"context": context, "classes": labelmap, "created_at": int(time.time() * 1000)}
    (target_root / "dataset_meta.json").write_text(json.dumps(dataset_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def check_dataset(dataset_id: str):
    entry = _resolve_dataset_entry(dataset_id)
    if entry.get("yolo_ready") and entry.get("yolo_images_dir") and entry.get("yolo_labels_dir"):
        inputs = {
            "images_dir": entry.get("yolo_images_dir"),
            "labels_dir": entry.get("yolo_labels_dir"),
            "labelmap_path": entry.get("yolo_labelmap_path"),
        }
        data = _validate_clip_dataset_impl(inputs, http_exception_cls=HTTPException, load_labelmap_simple_fn=_load_labelmap_simple)
        return {"ok": True, "format": "yolo", "detail": data}
    if entry.get("qwen_ready"):
        return {"ok": True, "format": "qwen", "detail": {"train": entry.get("train_count"), "val": entry.get("val_count")}}
    return {"ok": False, "format": entry.get("format") or "unknown", "detail": "dataset_format_unrecognized"}


def _load_dataset_meta_candidates(dataset_root: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    metas: List[Tuple[Path, Dict[str, Any]]] = []
    for name in (DATASET_META_NAME, QWEN_METADATA_FILENAME, SAM3_DATASET_META_NAME):
        path = dataset_root / name
        if not path.exists():
            continue
        meta = _load_json_metadata(path)
        if meta is not None:
            metas.append((path, meta))
    return metas


def get_dataset_glossary(dataset_id: str):
    entry = _resolve_dataset_entry(dataset_id)
    dataset_root = Path(entry["dataset_root"])
    glossary = _load_dataset_glossary(
        dataset_root,
        load_sam3_meta=lambda dataset_dir: _load_sam3_dataset_metadata_impl(
            dataset_dir,
            meta_name=SAM3_DATASET_META_NAME,
            load_json_metadata_fn=_load_json_metadata,
            persist_metadata_fn=lambda dataset_dir_inner, metadata: _persist_sam3_dataset_metadata_impl(
                dataset_dir_inner, metadata, meta_name=SAM3_DATASET_META_NAME, logger=logger
            ),
        ),
        load_qwen_meta=lambda dataset_dir: _load_qwen_dataset_metadata_impl(
            dataset_dir, meta_name=QWEN_METADATA_FILENAME, load_json_metadata_fn=_load_json_metadata
        ),
    )
    if not glossary:
        for _path, meta in _load_dataset_meta_candidates(dataset_root):
            raw = meta.get("labelmap_glossary")
            if raw:
                glossary = _normalize_labelmap_glossary(raw)
                break
    return {"dataset_id": dataset_id, "glossary": glossary or ""}


def set_dataset_glossary(dataset_id: str, glossary: str):
    entry = _resolve_dataset_entry(dataset_id)
    dataset_root = Path(entry["dataset_root"])
    normalized = _normalize_labelmap_glossary(glossary or "")
    updated = False
    for path, meta in _load_dataset_meta_candidates(dataset_root):
        meta["labelmap_glossary"] = normalized
        with path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        updated = True
    if not updated:
        meta = {"id": dataset_id, "labelmap_glossary": normalized}
        _persist_dataset_metadata_impl(dataset_root, meta, meta_name=DATASET_META_NAME, logger=logger)
    return {"dataset_id": dataset_id, "glossary": normalized}


def get_text_label(dataset_id: str, image_name: str):
    entry = _resolve_dataset_entry(dataset_id)
    dataset_root = Path(entry["dataset_root"])
    text_dir = dataset_root / "text_labels"
    text_path = text_dir / f"{Path(image_name).stem}.txt"
    if not text_path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="caption_not_found")
    return {"caption": text_path.read_text(encoding="utf-8").strip()}


def set_text_label(dataset_id: str, image_name: str, caption: str):
    entry = _resolve_dataset_entry(dataset_id)
    dataset_root = Path(entry["dataset_root"])
    text_dir = dataset_root / "text_labels"
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / f"{Path(image_name).stem}.txt"
    text_path.write_text(str(caption or "").strip(), encoding="utf-8")
    return {"status": "saved", "caption": str(caption or "").strip()}


def list_glossary_library():
    entries = []
    for path in sorted(GLOSSARY_LIBRARY_ROOT.glob("*.json")):
        data = _load_json_metadata(path) or {}
        name = data.get("name") or path.stem
        glossary = data.get("glossary") or ""
        entries.append(
            {
                "name": name,
                "preview": _glossary_preview(str(glossary or ""), []),
                "updated_at": path.stat().st_mtime,
            }
        )
    return entries


def get_glossary_entry(name: str):
    safe = _sanitize_yolo_run_id_impl(name) or name
    path = GLOSSARY_LIBRARY_ROOT / f"{safe}.json"
    if not path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="glossary_not_found")
    data = _load_json_metadata(path) or {}
    glossary = data.get("glossary") if isinstance(data, dict) else ""
    return {"name": data.get("name") or safe, "glossary": glossary or ""}


def save_glossary_entry(name: str, glossary: str):
    safe = _sanitize_yolo_run_id_impl(name) or name
    if not safe:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="glossary_name_required")
    path = GLOSSARY_LIBRARY_ROOT / f"{safe}.json"
    payload = {"name": name, "glossary": glossary}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return {"status": "saved", "name": name}


def delete_glossary_entry(name: str):
    safe = _sanitize_yolo_run_id_impl(name) or name
    path = GLOSSARY_LIBRARY_ROOT / f"{safe}.json"
    if not path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="glossary_not_found")
    path.unlink(missing_ok=True)
    return {"status": "deleted", "name": name}


def list_qwen_datasets():
    return [entry for entry in _list_all_datasets() if entry.get("source") == "qwen"]


def delete_qwen_dataset(dataset_id: str):
    dataset_root = QWEN_DATASET_ROOT / dataset_id
    if not dataset_root.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_dataset_not_found")
    if not _path_is_within_root_impl(dataset_root.resolve(), QWEN_DATASET_ROOT.resolve()):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="qwen_dataset_delete_forbidden")
    shutil.rmtree(dataset_root, ignore_errors=True)
    return {"status": "deleted", "id": dataset_id}


def init_qwen_dataset_upload(run_name: Optional[str]) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex
    root_dir = DATASET_UPLOAD_ROOT / f"qwen_upload_{job_id}"
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "train").mkdir(parents=True, exist_ok=True)
    (root_dir / "val").mkdir(parents=True, exist_ok=True)
    job = QwenDatasetUploadJob(job_id=job_id, root_dir=root_dir, run_name=run_name)
    with QWEN_DATASET_UPLOADS_LOCK:
        QWEN_DATASET_UPLOADS[job_id] = job
    return {"job_id": job_id}


def upload_qwen_dataset_chunk(job_id: str, split: str, image_name: str, annotation_line: str, file: UploadFile):
    with QWEN_DATASET_UPLOADS_LOCK:
        job = QWEN_DATASET_UPLOADS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_dataset_job_not_found")
    split = split.lower()
    if split not in {"train", "val"}:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="qwen_dataset_split_invalid")
    split_root = job.root_dir / split
    split_root.mkdir(parents=True, exist_ok=True)
    image_name = image_name or file.filename or f"{uuid.uuid4().hex}.jpg"
    image_path = split_root / image_name
    with image_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    ann_path = split_root / "annotations.jsonl"
    with ann_path.open("a", encoding="utf-8") as handle:
        handle.write(str(annotation_line).strip() + "\n")
    if split == "train":
        job.train_count += 1
    else:
        job.val_count += 1
    job.updated_at = time.time()
    return {"status": "ok", "job_id": job_id, "train_count": job.train_count, "val_count": job.val_count}


def finalize_qwen_dataset_upload(job_id: str, metadata: Dict[str, Any], run_name: Optional[str]):
    with QWEN_DATASET_UPLOADS_LOCK:
        job = QWEN_DATASET_UPLOADS.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_dataset_job_not_found")
    base_id = run_name or metadata.get("id") or job.run_name or job_id
    base_id = _sanitize_yolo_run_id_impl(str(base_id)) or str(job_id)
    dataset_id_final = _unique_dataset_name(base_id, root=QWEN_DATASET_ROOT)
    target_root = QWEN_DATASET_ROOT / dataset_id_final
    if target_root.exists():
        shutil.rmtree(target_root, ignore_errors=True)
    shutil.move(str(job.root_dir), target_root)
    classes = metadata.get("classes") or []
    labelmap = [str(c).strip() for c in classes if str(c).strip()]
    if labelmap:
        (target_root / "labelmap.txt").write_text("\n".join(labelmap) + "\n", encoding="utf-8")
    meta = {
        "id": dataset_id_final,
        "label": dataset_id_final,
        "classes": labelmap,
        "context": metadata.get("context") or "",
        "created_at": time.time(),
        "train_count": job.train_count,
        "val_count": job.val_count,
        "image_count": job.train_count + job.val_count,
        "signature": _compute_dir_signature_impl(target_root),
        "type": "bbox",
    }
    (target_root / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_meta = {"context": meta["context"], "classes": labelmap, "created_at": int(time.time() * 1000)}
    (target_root / "dataset_meta.json").write_text(json.dumps(dataset_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def cancel_qwen_dataset_upload(job_id: str):
    with QWEN_DATASET_UPLOADS_LOCK:
        job = QWEN_DATASET_UPLOADS.pop(job_id, None)
    if not job:
        return {"status": "missing", "job_id": job_id}
    shutil.rmtree(job.root_dir, ignore_errors=True)
    return {"status": "cancelled", "job_id": job_id}


def list_sam3_datasets():
    return _list_all_datasets()


def _resolve_sam3_dataset_meta(dataset_id: str) -> Dict[str, Any]:
    dataset_root = _resolve_sam3_or_qwen_dataset(dataset_id)
    annotations_path = dataset_root / "train" / "annotations.jsonl"
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    if annotations_path.exists():
        meta = _convert_qwen_dataset_to_coco_impl(dataset_root)
    elif train_images.exists() and train_labels.exists():
        meta = _convert_yolo_dataset_to_coco_impl(dataset_root)
    else:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_dataset_type_unsupported")
    meta["dataset_root"] = str(dataset_root)
    return meta


def convert_sam3_dataset(dataset_id: str):
    meta = _resolve_sam3_dataset_meta(dataset_id)
    return meta


def list_sam3_dataset_classes(dataset_id: str):
    entry = _resolve_dataset_entry(dataset_id)
    classes = entry.get("classes") or []
    if classes:
        return {"dataset_id": dataset_id, "classes": classes}
    dataset_root = Path(entry["dataset_root"])
    labelmap_path = dataset_root / "labelmap.txt"
    if labelmap_path.exists():
        classes = [line.strip() for line in labelmap_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"dataset_id": dataset_id, "classes": classes}


## NOTE: run dir/meta helpers use *_impl directly to avoid wrapper drift.


_yolo_labels_have_polygons = _yolo_labels_have_polygons_impl
_convert_coco_dataset_to_yolo = functools.partial(
    _convert_coco_dataset_to_yolo_impl,
    load_sam3_meta_fn=lambda dataset_dir: {},
    persist_meta_fn=lambda dataset_dir_inner, metadata: None,
)


_resolve_yolo_training_dataset = functools.partial(
    _resolve_yolo_training_dataset_impl,
    resolve_dataset_entry_fn=lambda dataset_id: _resolve_dataset_entry_impl(
        dataset_id,
        list_all_datasets_fn=_list_all_datasets,
    ),
    resolve_sam3_or_qwen_dataset_fn=_resolve_sam3_or_qwen_dataset,
    compute_dir_signature_fn=_compute_dir_signature_impl,
    sanitize_yolo_run_id_fn=_sanitize_yolo_run_id_impl,
    detect_yolo_layout_fn=_detect_yolo_layout_impl,
    yolo_labels_have_polygons_fn=_yolo_labels_have_polygons_impl,
    stable_hash_fn=_stable_hash_impl,
    yolo_cache_root=YOLO_DATASET_CACHE_ROOT,
    http_exception_cls=HTTPException,
)


_resolve_rfdetr_training_dataset = functools.partial(
    _resolve_rfdetr_training_dataset_impl,
    resolve_dataset_entry_fn=lambda dataset_id: _resolve_dataset_entry_impl(
        dataset_id,
        list_all_datasets_fn=_list_all_datasets,
    ),
    resolve_sam3_or_qwen_dataset_fn=_resolve_sam3_or_qwen_dataset,
    load_sam3_meta_fn=lambda dataset_dir: _load_sam3_dataset_metadata_impl(
        dataset_dir,
        meta_name=SAM3_DATASET_META_NAME,
        load_json_metadata_fn=_load_json_metadata,
        persist_metadata_fn=lambda dataset_dir_inner, metadata: _persist_sam3_dataset_metadata_impl(
            dataset_dir_inner,
            metadata,
            meta_name=SAM3_DATASET_META_NAME,
            logger=logger,
        ),
    ),
    detect_yolo_layout_fn=_detect_yolo_layout_impl,
    yolo_labels_have_polygons_fn=_yolo_labels_have_polygons_impl,
    convert_yolo_dataset_to_coco_fn=_convert_yolo_dataset_to_coco_impl,
    convert_qwen_dataset_to_coco_fn=_convert_qwen_dataset_to_coco_impl,
    load_qwen_dataset_metadata_fn=lambda dataset_dir: _load_qwen_dataset_metadata_impl(
        dataset_dir,
        meta_name=QWEN_METADATA_FILENAME,
        load_json_metadata_fn=_load_json_metadata,
    ),
    ensure_coco_supercategory_fn=_ensure_coco_supercategory_impl,
    http_exception_cls=HTTPException,
)


## NOTE: RF‑DETR helpers use *_impl directly to avoid wrapper drift.


## NOTE: YOLO dataset helpers use *_impl directly to avoid wrapper drift.


class ConcatHead(torch.nn.Module):
    """Concatenation layer for YOLOv8 Detect heads (head-grafting)."""

    def __init__(self, nc1: int = 80, nc2: int = 1, ch: Tuple[int, ...] = ()):
        super().__init__()
        self.nc1 = int(nc1)
        self.nc2 = int(nc2)

    def forward(self, x):
        # x is a list of length 2 (Detect outputs)
        if isinstance(x[0], tuple):
            preds1 = x[0][0]
            preds2 = x[1][0]
        elif isinstance(x[0], list):
            return [torch.cat((x0, x1), dim=1) for x0, x1 in zip(x[0], x[1])]
        else:
            preds1 = x[0]
            preds2 = x[1]

        preds = torch.cat((preds1[:, :4, :], preds2[:, :4, :]), dim=2)

        shape = list(preds1.shape)
        shape[-1] = preds1.shape[-1] + preds2.shape[-1]
        preds1_extended = torch.zeros(shape, device=preds1.device, dtype=preds1.dtype)
        preds1_extended[..., : preds1.shape[-1]] = preds1

        shape = list(preds2.shape)
        shape[-1] = preds1.shape[-1] + preds2.shape[-1]
        preds2_extended = torch.zeros(shape, device=preds2.device, dtype=preds2.dtype)
        preds2_extended[..., preds2.shape[-1] :] = preds2

        preds = torch.cat((preds, preds1_extended[:, 4:, :]), dim=1)
        preds = torch.cat((preds, preds2_extended[:, 4:, :]), dim=1)

        if isinstance(x[0], tuple):
            return (preds, x[0][1])
        return preds


def _patch_ultralytics_for_head_grafting() -> None:
    global YOLO_HEAD_GRAFT_PATCHED
    if YOLO_HEAD_GRAFT_PATCHED:
        return
    try:
        import ultralytics  # type: ignore
        from ultralytics.nn import tasks as yolo_tasks  # type: ignore
        from ultralytics.nn import modules as yolo_modules  # type: ignore
    except Exception:
        return
    try:
        version = getattr(ultralytics, "__version__", "")
        if version and not version.startswith("8."):
            logger.warning("Ultralytics %s not in supported 8.x range for head grafting", version)
    except Exception:
        pass
    try:
        if not hasattr(yolo_modules, "ConcatHead"):
            yolo_modules.ConcatHead = ConcatHead
        if hasattr(yolo_modules, "__all__") and "ConcatHead" not in yolo_modules.__all__:
            yolo_modules.__all__ = tuple(list(yolo_modules.__all__) + ["ConcatHead"])
        if not hasattr(yolo_tasks, "ConcatHead"):
            yolo_tasks.ConcatHead = ConcatHead
    except Exception:
        return

    original_parse_model = yolo_tasks.parse_model

    def parse_model_patched(d, ch, verbose=True):
        import ast
        import contextlib
        legacy = True
        max_channels = float("inf")
        nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        scale = d.get("scale")
        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
                yolo_tasks.LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            yolo_tasks.Conv.default_act = eval(act)
            if verbose:
                yolo_tasks.LOGGER.info(f"{yolo_tasks.colorstr('activation:')} {act}")

        if verbose:
            yolo_tasks.LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]
        def _opt(name: str):
            return getattr(yolo_tasks, name, None)

        base_modules = {
            _opt("Classify"),
            _opt("Conv"),
            _opt("ConvTranspose"),
            _opt("GhostConv"),
            _opt("Bottleneck"),
            _opt("GhostBottleneck"),
            _opt("SPP"),
            _opt("SPPF"),
            _opt("C2fPSA"),
            _opt("C2PSA"),
            _opt("DWConv"),
            _opt("Focus"),
            _opt("BottleneckCSP"),
            _opt("C1"),
            _opt("C2"),
            _opt("C2f"),
            _opt("C3k2"),
            _opt("RepNCSPELAN4"),
            _opt("ELAN1"),
            _opt("ADown"),
            _opt("AConv"),
            _opt("SPPELAN"),
            _opt("C2fAttn"),
            _opt("C3"),
            _opt("C3TR"),
            _opt("C3Ghost"),
            _opt("C3x"),
            _opt("RepC3"),
            _opt("PSA"),
            _opt("SCDown"),
            _opt("C2fCIB"),
            _opt("A2C2f"),
            torch.nn.ConvTranspose2d,
            _opt("DWConvTranspose2d"),
        }
        base_modules = frozenset(x for x in base_modules if x is not None)

        repeat_modules = {
            _opt("BottleneckCSP"),
            _opt("C1"),
            _opt("C2"),
            _opt("C2f"),
            _opt("C3k2"),
            _opt("C2fAttn"),
            _opt("C3"),
            _opt("C3TR"),
            _opt("C3Ghost"),
            _opt("C3x"),
            _opt("RepC3"),
            _opt("C2fPSA"),
            _opt("C2fCIB"),
            _opt("C2PSA"),
            _opt("A2C2f"),
        }
        repeat_modules = frozenset(x for x in repeat_modules if x is not None)

        detect_modules = [
            _opt("Detect"),
            _opt("WorldDetect"),
            _opt("YOLOEDetect"),
            _opt("Segment"),
            _opt("YOLOESegment"),
            _opt("Pose"),
            _opt("OBB"),
            _opt("ImagePoolingAttn"),
            _opt("v10Detect"),
        ]
        detect_modules = [m for m in detect_modules if m is not None]
        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            m_name = m
            m = (
                getattr(torch.nn, m[3:])
                if "nn." in m
                else getattr(__import__("torchvision").ops, m[16:])
                if "torchvision.ops." in m
                else getattr(yolo_tasks, m)
            )
            if m is None:
                raise KeyError(f"module_not_found:{m_name}")
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
            n = n_ = max(round(n * depth), 1) if n > 1 else n
            if m in base_modules:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = yolo_tasks.make_divisible(min(c2, max_channels) * width, 8)
                if _opt("C2fAttn") is not None and m is _opt("C2fAttn"):
                    args[1] = yolo_tasks.make_divisible(min(args[1], max_channels // 2) * width, 8)
                    args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
                if _opt("C3k2") is not None and m is _opt("C3k2"):
                    legacy = False
                    if scale in "mlx":
                        args[3] = True
                if _opt("A2C2f") is not None and m is _opt("A2C2f"):
                    legacy = False
                    if scale in "lx":
                        args.extend((True, 1.2))
                if _opt("C2fCIB") is not None and m is _opt("C2fCIB"):
                    legacy = False
            elif _opt("AIFI") is not None and m is _opt("AIFI"):
                args = [ch[f], *args]
            elif m in frozenset({x for x in [_opt("HGStem"), _opt("HGBlock")] if x is not None}):
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if _opt("HGBlock") is not None and m is _opt("HGBlock"):
                    args.insert(4, n)
                    n = 1
            elif _opt("ResNetLayer") is not None and m is _opt("ResNetLayer"):
                c2 = args[1] if args[3] else args[1] * 4
            elif m is torch.nn.BatchNorm2d:
                args = [ch[f]]
            elif _opt("Concat") is not None and m is _opt("Concat"):
                c2 = sum(ch[x] for x in f)
            elif m is ConcatHead:
                c2 = ch[f[-1]] if isinstance(f, list) else ch[f]
            elif m in frozenset(detect_modules):
                args.append([ch[x] for x in f])
                seg_cls = _opt("Segment")
                yoloe_seg_cls = _opt("YOLOESegment")
                if (seg_cls is not None and m is seg_cls) or (yoloe_seg_cls is not None and m is yoloe_seg_cls):
                    args[2] = yolo_tasks.make_divisible(min(args[2], max_channels) * width, 8)
                detect_like = {x for x in [_opt("Detect"), _opt("YOLOEDetect"), _opt("Segment"), _opt("YOLOESegment"), _opt("Pose"), _opt("OBB")] if x is not None}
                if m in detect_like:
                    m.legacy = legacy
            elif _opt("RTDETRDecoder") is not None and m is _opt("RTDETRDecoder"):
                args.insert(1, [ch[x] for x in f])
            elif _opt("CBLinear") is not None and m is _opt("CBLinear"):
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif _opt("CBFuse") is not None and m is _opt("CBFuse"):
                c2 = ch[f[-1]]
            elif m in frozenset({x for x in [_opt("TorchVision"), _opt("Index")] if x is not None}):
                c2 = args[0]
                c1 = ch[f]
                args = [*args[1:]]
            else:
                c2 = ch[f]

            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                yolo_tasks.LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return torch.nn.Sequential(*layers), sorted(save)

    def base_apply_patched(self, fn):
        self = torch.nn.Module._apply(self, fn)
        for m in self.model:
            if isinstance(m, yolo_tasks.Detect):
                m.stride = fn(m.stride)
                m.anchors = fn(m.anchors)
                m.strides = fn(m.strides)
        return self

    def detection_init_patched(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        yolo_tasks.BaseModel.__init__(self)
        self.yaml = cfg if isinstance(cfg, dict) else yolo_tasks.yaml_model_load(cfg)
        if self.yaml["backbone"][0][2] == "Silence":
            yolo_tasks.LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"
        self.yaml["channels"] = ch
        if nc and nc != self.yaml["nc"]:
            yolo_tasks.LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        self.model, self.save = yolo_tasks.parse_model(yolo_tasks.deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        for m in self.model:
            if isinstance(m, yolo_tasks.Detect):
                s = 256
                m.inplace = self.inplace

                def _forward(x):
                    if self.end2end:
                        return self.forward(x)["one2many"]
                    return self.forward(x)[0] if isinstance(m, (yolo_tasks.Segment, yolo_tasks.YOLOESegment, yolo_tasks.Pose, yolo_tasks.OBB)) else self.forward(x)

                self.model.eval()
                m.training = True
                m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
                self.stride = m.stride
                self.model.train()
                m.bias_init()
            else:
                self.stride = torch.Tensor([32])

        yolo_tasks.initialize_weights(self)
        if verbose:
            self.info()
            yolo_tasks.LOGGER.info("")

    yolo_tasks.parse_model = parse_model_patched
    yolo_tasks.BaseModel._apply = base_apply_patched
    yolo_tasks.DetectionModel.__init__ = detection_init_patched
    YOLO_HEAD_GRAFT_PATCHED = True



## NOTE: YOLO YAML/head helpers use *_impl directly to avoid wrapper drift.


def _ensure_yolo_inference_runtime() -> Tuple[Any, List[str], Optional[str]]:
    global yolo_infer_model, yolo_infer_path, yolo_infer_labelmap, yolo_infer_task
    return _ensure_yolo_inference_runtime_impl(
        load_active_fn=lambda: _load_yolo_active_impl(YOLO_ACTIVE_PATH),
        load_labelmap_fn=_yolo_load_labelmap_impl,
        patch_ultralytics_fn=_patch_ultralytics_for_head_grafting,
        yolo_lock=YOLO_INFER_LOCK,
        get_state_fn=lambda: (yolo_infer_model, yolo_infer_path, yolo_infer_labelmap, yolo_infer_task),
        set_state_fn=lambda model, path, labelmap, task: _set_yolo_infer_state(model, path, labelmap, task),
        import_yolo_fn=lambda: __import__("ultralytics").YOLO,  # type: ignore[attr-defined]
        http_exception_cls=HTTPException,
    )


def _import_rfdetr_variants() -> Dict[str, Any]:
    from rfdetr import (  # type: ignore
        RFDETRBase,
        RFDETRLarge,
        RFDETRNano,
        RFDETRSmall,
        RFDETRMedium,
        RFDETRSegPreview,
    )

    return {
        "rfdetr-nano": RFDETRNano,
        "rfdetr-small": RFDETRSmall,
        "rfdetr-medium": RFDETRMedium,
        "rfdetr-base": RFDETRBase,
        "rfdetr-large": RFDETRLarge,
        "rfdetr-seg-preview": RFDETRSegPreview,
    }


def _resolve_rfdetr_infer_device() -> str:
    return os.environ.get(
        "RFDETR_INFER_DEVICE",
        "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() == 1 else "cpu",
    )


def _ensure_rfdetr_inference_runtime() -> Tuple[Any, List[str], Optional[str]]:
    global rfdetr_infer_model, rfdetr_infer_path, rfdetr_infer_labelmap, rfdetr_infer_task, rfdetr_infer_variant

    return _ensure_rfdetr_inference_runtime_impl(
        load_active_fn=_load_rfdetr_active,
        load_labelmap_fn=_yolo_load_labelmap_impl,
        variant_info_fn=lambda task, variant: _rfdetr_variant_info_impl(
            task,
            variant,
            variants=RFDETR_VARIANTS,
            http_exception_cls=HTTPException,
        ),
        rfdetr_lock=RFDETR_INFER_LOCK,
        get_state_fn=lambda: (
            rfdetr_infer_model,
            rfdetr_infer_path,
            rfdetr_infer_labelmap,
            rfdetr_infer_task,
            rfdetr_infer_variant,
        ),
        set_state_fn=lambda model, path, labelmap, task, variant_id: _set_rfdetr_infer_state(
            model, path, labelmap, task, variant_id
        ),
        import_rfdetr_fn=_import_rfdetr_variants,
        http_exception_cls=HTTPException,
        torch_available=torch.cuda.is_available,
        resolve_device_fn=_resolve_rfdetr_infer_device,
    )


def _ensure_yolo_inference_runtime_for_detector(detector_id: Optional[str]) -> Tuple[Any, List[str], Optional[str]]:
    detector_run_id = str(detector_id or "").strip()
    if not detector_run_id:
        return _ensure_yolo_inference_runtime()
    active = _load_yolo_active_impl(YOLO_ACTIVE_PATH)
    active_run_id = str((active or {}).get("run_id") or "").strip()
    if active_run_id and detector_run_id == active_run_id:
        return _ensure_yolo_inference_runtime()
    run_dir = _yolo_run_dir_impl(
        detector_run_id,
        create=False,
        job_root=YOLO_JOB_ROOT,
        sanitize_fn=_sanitize_yolo_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_run_not_found")
    best_path = run_dir / "best.pt"
    if not best_path.exists():
        raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="yolo_best_missing")
    run_meta = _yolo_load_run_meta_impl(run_dir, meta_name=YOLO_RUN_META_NAME)
    config = run_meta.get("config") or {}
    dataset = config.get("dataset") or {}
    labelmap_path = run_dir / "labelmap.txt"
    labelmap = _yolo_load_labelmap_impl(labelmap_path) if labelmap_path.exists() else []
    with YOLO_INFER_LOCK:
        if yolo_infer_model is not None and yolo_infer_path == str(best_path):
            return yolo_infer_model, yolo_infer_labelmap, yolo_infer_task
        YOLO = __import__("ultralytics").YOLO  # type: ignore[attr-defined]
        _patch_ultralytics_for_head_grafting()
        model = YOLO(str(best_path))
        task = config.get("task") or dataset.get("task") or getattr(model, "task", None)
        _set_yolo_infer_state(model, str(best_path), labelmap, task)
        return model, labelmap, task


def _ensure_rfdetr_inference_runtime_for_detector(detector_id: Optional[str]) -> Tuple[Any, List[str], Optional[str]]:
    detector_run_id = str(detector_id or "").strip()
    if not detector_run_id:
        return _ensure_rfdetr_inference_runtime()
    active = _load_rfdetr_active()
    active_run_id = str((active or {}).get("run_id") or "").strip()
    if active_run_id and detector_run_id == active_run_id:
        return _ensure_rfdetr_inference_runtime()
    run_dir = _rfdetr_run_dir_impl(
        detector_run_id,
        create=False,
        job_root=RFDETR_JOB_ROOT,
        sanitize_fn=_sanitize_rfdetr_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="rfdetr_run_not_found")
    best_path = _rfdetr_best_checkpoint_impl(run_dir)
    if not best_path:
        raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="rfdetr_best_missing")
    run_meta = _rfdetr_load_run_meta_impl(run_dir, meta_name=RFDETR_RUN_META_NAME)
    config = run_meta.get("config") or {}
    dataset = config.get("dataset") or {}
    task = str(config.get("task") or dataset.get("task") or "detect")
    variant = config.get("variant")
    with RFDETR_INFER_LOCK:
        if rfdetr_infer_model is not None and rfdetr_infer_path == str(best_path):
            return rfdetr_infer_model, rfdetr_infer_labelmap, rfdetr_infer_task
        import_map = _import_rfdetr_variants()
        variant_info = _rfdetr_variant_info_impl(
            task,
            variant,
            variants=RFDETR_VARIANTS,
            http_exception_cls=HTTPException,
        )
        variant_id = variant_info.get("id")
        model_cls = import_map.get(variant_id)
        if not model_cls:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_variant_unknown")
        device_str = _resolve_rfdetr_infer_device()
        if isinstance(device_str, str) and device_str.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.set_device(device_str)
            except Exception:
                pass
        model_kwargs: Dict[str, Any] = {
            "pretrain_weights": str(best_path),
            "device": device_str,
        }
        if variant_id == "rfdetr-seg-preview" or task == "segment":
            model_kwargs["segmentation_head"] = True
        model = model_cls(**model_kwargs)
        labelmap_path = run_dir / "labelmap.txt"
        labelmap = _yolo_load_labelmap_impl(labelmap_path) if labelmap_path.exists() else []
        if labelmap:
            try:
                model.model.class_names = labelmap
            except Exception:
                pass
        _set_rfdetr_infer_state(model, str(best_path), labelmap, task, variant_id)
        return model, labelmap, task


def _promote_run(run_id: str, variant: str) -> Dict[str, Any]:
    run_dir = _run_dir_for_request_impl(
        run_id=run_id,
        variant=variant,
        job_root=SAM3_JOB_ROOT,
        http_exception_cls=HTTPException,
        http_400=HTTP_400_BAD_REQUEST,
        http_404=HTTP_404_NOT_FOUND,
    )
    active_paths = _active_run_paths_for_variant_impl(
        variant=variant,
        jobs_lock=SAM3_TRAINING_JOBS_LOCK,
        jobs=SAM3_TRAINING_JOBS,
    )
    if run_dir.resolve() in active_paths:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="sam3_run_active")
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="sam3_checkpoint_dir_missing")
    ckpts = [p for p in ckpt_dir.iterdir() if p.is_file() and p.suffix in {".ckpt", ".pth", ".pt"}]
    if not ckpts:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="sam3_checkpoints_missing")
    # choose keep candidate: prefer last.ckpt else newest
    keep = None
    for p in ckpts:
        if p.name == "last.ckpt":
            keep = p
            break
    if keep is None:
        keep = max(ckpts, key=lambda p: p.stat().st_mtime if p.exists() else 0)
    deleted = []
    freed = 0
    for p in ckpts:
        if p == keep:
            continue
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        try:
            p.unlink()
            deleted.append(str(p))
            freed += size
        except Exception:
            continue
    stripped, before, after = _strip_checkpoint_optimizer_impl(keep, torch_module=torch)
    freed += max(0, before - after)
    marker = run_dir / ".promoted"
    try:
        marker.write_text(json.dumps({"timestamp": time.time(), "keep": str(keep)}), encoding="utf-8")
    except Exception:
        pass
    return {
        "kept": str(keep),
        "kept_size_bytes": keep.stat().st_size if keep.exists() else 0,
        "stripped_optimizer": stripped,
        "deleted": deleted,
        "freed_bytes": freed,
        "run_path": str(run_dir),
        "promoted": True,
        "promoted_at": time.time(),
    }


_persist_agent_recipe = functools.partial(
    _persist_agent_recipe_impl,
    recipes_root=AGENT_MINING_RECIPES_ROOT,
    max_clip_head_bytes=AGENT_RECIPE_MAX_CLIP_HEAD_BYTES,
    max_crops=AGENT_RECIPE_MAX_CROPS,
    max_crop_bytes=AGENT_RECIPE_MAX_CROP_BYTES,
    resolve_dataset_fn=_resolve_sam3_or_qwen_dataset,
    load_coco_index_fn=_load_coco_index_impl,
    compute_dataset_signature_fn=_compute_dataset_signature_impl,
    compute_labelmap_hash_fn=_compute_labelmap_hash_impl,
    resolve_clip_classifier_fn=lambda path_str: _resolve_agent_clip_classifier_path_impl(
        path_str,
        allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
        allowed_exts=CLASSIFIER_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root_impl,
        http_exception_cls=HTTPException,
    ),
    load_clip_head_fn=lambda classifier_path: _load_clip_head_from_classifier_impl(
        classifier_path,
        joblib_load_fn=joblib.load,
        http_exception_cls=HTTPException,
        clip_head_background_indices_fn=_clip_head_background_indices,
        resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
        infer_clip_model_fn=_infer_clip_model_from_embedding_dim_impl,
        active_clip_model_name=clip_model_name,
        default_clip_model=DEFAULT_CLIP_MODEL,
        logger=logger,
    ),
    save_clip_head_artifacts_fn=_save_clip_head_artifacts,
    load_clip_head_artifacts_fn=_load_clip_head_artifacts,
    save_exemplar_crop_fn=_save_exemplar_crop_impl,
    sanitize_prompts_fn=_sanitize_prompts_impl,
    path_is_within_root_fn=_path_is_within_root_impl,
)


## NOTE: agent recipe loaders use *_impl directly to avoid wrapper drift.


_delete_agent_recipe = functools.partial(
    _delete_agent_recipe_impl,
    recipes_root=AGENT_MINING_RECIPES_ROOT,
    path_is_within_root_fn=_path_is_within_root_impl,
    http_exception_cls=HTTPException,
)

_list_agent_recipes = functools.partial(_list_agent_recipes_impl, recipes_root=AGENT_MINING_RECIPES_ROOT)


_ensure_recipe_zip = functools.partial(_ensure_recipe_zip_impl, recipes_root=AGENT_MINING_RECIPES_ROOT)


_import_agent_recipe_zip_bytes = functools.partial(
    _import_agent_recipe_zip_bytes_impl,
    recipes_root=AGENT_MINING_RECIPES_ROOT,
    max_json_bytes=AGENT_RECIPE_MAX_JSON_BYTES,
    max_clip_head_bytes=AGENT_RECIPE_MAX_CLIP_HEAD_BYTES,
    max_crops=AGENT_RECIPE_MAX_CROPS,
    max_crop_bytes=AGENT_RECIPE_MAX_CROP_BYTES,
    persist_recipe_fn=_persist_agent_recipe,
)


_persist_agent_cascade = functools.partial(
    _persist_agent_cascade_impl,
    cascades_root=AGENT_MINING_CASCADES_ROOT,
    path_is_within_root_fn=_path_is_within_root_impl,
)


## NOTE: agent cascade loader uses *_impl directly to avoid wrapper drift.


_list_agent_cascades = functools.partial(_list_agent_cascades_impl, cascades_root=AGENT_MINING_CASCADES_ROOT)

_delete_agent_cascade = functools.partial(
    _delete_agent_cascade_impl,
    cascades_root=AGENT_MINING_CASCADES_ROOT,
    path_is_within_root_fn=_path_is_within_root_impl,
)


_ensure_cascade_zip = functools.partial(
    _ensure_cascade_zip_impl,
    cascades_root=AGENT_MINING_CASCADES_ROOT,
    recipes_root=AGENT_MINING_RECIPES_ROOT,
    classifiers_root=(UPLOAD_ROOT / "classifiers"),
    path_is_within_root_fn=_path_is_within_root_impl,
    ensure_recipe_zip_fn=_ensure_recipe_zip,
    load_recipe_fn=lambda recipe_id: _load_agent_recipe_impl(
        recipe_id,
        recipes_root=AGENT_MINING_RECIPES_ROOT,
        path_is_within_root_fn=_path_is_within_root_impl,
    ),
    resolve_classifier_fn=lambda path_str: _resolve_agent_clip_classifier_path_impl(
        path_str,
        allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
        allowed_exts=CLASSIFIER_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root_impl,
        http_exception_cls=HTTPException,
    ),
)


_import_agent_cascade_zip_bytes = functools.partial(
    _import_agent_cascade_zip_bytes_impl,
    cascades_root=AGENT_MINING_CASCADES_ROOT,
    classifiers_root=(UPLOAD_ROOT / "classifiers"),
    max_json_bytes=AGENT_CASCADE_MAX_JSON_BYTES,
    classifier_allowed_exts=CLASSIFIER_ALLOWED_EXTS,
    path_is_within_root_fn=_path_is_within_root_impl,
    import_recipe_fn=_import_agent_recipe_zip_bytes,
    persist_cascade_fn=_persist_agent_cascade,
)


def _serialize_agent_mining_job(job: AgentMiningJob) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "progress": job.progress,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "request": job.request,
        "result": job.result,
        "logs": job.logs,
        "error": job.error,
    }


_calibration_update = _calibration_update_impl
_calibration_write_record_atomic = _calibration_write_record_atomic_impl


def _calibration_prepass_worker(
    device_index: int,
    tasks: List[Tuple[str, str]],
    dataset_id: str,
    labelmap: List[str],
    glossary: str,
    prepass_payload_dict: Dict[str, Any],
    cancel_event: Optional[Any],
    progress_queue: Optional[Any],
) -> None:
    _calibration_prepass_worker_impl(
        device_index,
        tasks,
        dataset_id,
        labelmap,
        glossary,
        prepass_payload_dict,
        cancel_event=cancel_event,
        progress_queue=progress_queue,
        resolve_dataset_fn=_resolve_sam3_or_qwen_dataset,
        prepass_request_cls=QwenPrepassRequest,
        cache_image_fn=_calibration_cache_image,
        run_prepass_fn=_agent_run_deep_prepass,
        write_record_fn=_calibration_write_record_atomic,
        set_device_pref_fn=_set_sam3_device_pref,
    )


## NOTE: calibration job runner uses *_impl directly to avoid wrapper drift.


def _run_prompt_helper_job(job: PromptHelperJob, payload: PromptHelperRequest) -> None:
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    job.status = "running"
    job.message = "Loading dataset…"
    job.request = payload.dict()
    job.updated_at = time.time()
    try:
        dataset_root = _resolve_sam3_or_qwen_dataset_impl(
            payload.dataset_id,
            list_all_datasets_fn=_list_all_datasets,
            resolve_dataset_legacy_fn=lambda dataset_id: _resolve_dataset_legacy_impl(
                dataset_id,
                qwen_root=QWEN_DATASET_ROOT,
                sam3_root=SAM3_DATASET_ROOT,
                registry_root=DATASET_REGISTRY_ROOT,
                http_exception_cls=HTTPException,
            ),
        )
        coco, gt_by_image_cat, images = _load_coco_index_impl(dataset_root)
        categories = coco.get("categories") or []
        cat_to_images: Dict[int, set[int]] = {}
        for ann in coco.get("annotations", []):
            try:
                cat_id = int(ann["category_id"])
                img_id = int(ann["image_id"])
            except Exception:
                continue
            cat_to_images.setdefault(cat_id, set()).add(img_id)
        prompts_map: Dict[int, List[str]] = {}
        if payload.prompts_by_class:
            for k, vals in payload.prompts_by_class.items():
                try:
                    cid = int(k)
                except Exception:
                    continue
                cleaned = [v.strip() for v in vals if isinstance(v, str) and v.strip()]
                if cleaned:
                    prompts_map[cid] = cleaned
        results: List[Dict[str, Any]] = []
        total_classes = len(categories) or 1
        image_cache: Dict[int, Image.Image] = {}
        # Precompute total steps for progress: each prompt * each sampled image.
        total_steps = 0
        for idx, cat in enumerate(categories):
            cat_id = int(cat.get("id", idx))
            prompts = prompts_map.get(cat_id)
            if not prompts:
                prompts = _generate_prompt_variants_for_class(
                    str(cat.get("name", f"class_{cat_id}")),
                    payload.max_synonyms,
                    payload.use_qwen,
                )
            sample_ids = _sample_images_for_category(
                cat_id,
                list(cat_to_images.get(cat_id, set())),
                payload.sample_per_class,
                payload.seed,
            )
            total_steps += len(prompts) * max(1, len(sample_ids))
        job.total_steps = total_steps
        job.completed_steps = 0
        for idx, cat in enumerate(categories):
            cat_id = int(cat.get("id", idx))
            class_name = str(cat.get("name", f"class_{cat_id}"))
            job.message = f"Evaluating {class_name} ({idx + 1}/{total_classes})…"
            job.progress = (idx) / total_classes
            job.updated_at = time.time()
            candidates = prompts_map.get(cat_id)
            if not candidates:
                candidates = _generate_prompt_variants_for_class(
                    class_name,
                    payload.max_synonyms,
                    payload.use_qwen,
                )
            sampled_images = _sample_images_for_category(
                cat_id,
                list(cat_to_images.get(cat_id, set())),
                payload.sample_per_class,
                payload.seed,
            )
            candidate_results: List[Dict[str, Any]] = []
            for prompt in candidates:
                step_label = f"{class_name}: '{prompt}'"
                try:
                    job.logs.append({"ts": time.time(), "msg": f"Running {step_label} on {len(sampled_images)} images"})
                    if len(job.logs) > MAX_JOB_LOGS:
                        job.logs[:] = job.logs[-MAX_JOB_LOGS:]
                except Exception:
                    pass
                metrics = _evaluate_prompt_for_class(
                    prompt,
                    cat_id=cat_id,
                    image_ids=sampled_images,
                    gt_by_image_cat=gt_by_image_cat,
                    images=images,
                    score_threshold=payload.score_threshold,
                    max_dets=payload.max_dets,
                    iou_threshold=payload.iou_threshold,
                    image_cache=image_cache,
                )
                candidate_results.append(metrics)
                job.completed_steps += max(1, len(sampled_images))
                if job.total_steps:
                    job.progress = min(1.0, job.completed_steps / job.total_steps)
                job.updated_at = time.time()
            candidate_results.sort(key=lambda m: (m.get("score", 0.0), m.get("recall", 0.0), m.get("precision", 0.0)), reverse=True)
            results.append(
                {
                    "class_id": cat_id,
                    "class_name": class_name,
                    "images_sampled": len(sampled_images),
                    "candidates": candidate_results,
                }
            )
            job.progress = (idx + 1) / total_classes
            job.updated_at = time.time()
        job.status = "completed"
        job.message = "Done"
        job.result = {
            "classes": results,
            "config": payload.dict(),
            "dataset_id": payload.dataset_id,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prompt helper job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "Failed"
    finally:
        job.updated_at = time.time()


def _run_prompt_helper_search_job(job: PromptHelperJob, payload: PromptHelperSearchRequest) -> None:
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    job.status = "running"
    job.message = "Loading dataset…"
    job.request = {"mode": "search", **payload.dict()}
    job.updated_at = time.time()
    try:
        dataset_root = _resolve_sam3_or_qwen_dataset_impl(
            payload.dataset_id,
            list_all_datasets_fn=_list_all_datasets,
            resolve_dataset_legacy_fn=lambda dataset_id: _resolve_dataset_legacy_impl(
                dataset_id,
                qwen_root=QWEN_DATASET_ROOT,
                sam3_root=SAM3_DATASET_ROOT,
                registry_root=DATASET_REGISTRY_ROOT,
                http_exception_cls=HTTPException,
            ),
        )
        coco, gt_by_image_cat, images = _load_coco_index_impl(dataset_root)
        categories = coco.get("categories") or []
        target_class_id = payload.class_id
        if not payload.prompts_by_class:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="search_prompts_required")
        cat_to_images: Dict[int, set[int]] = {}
        for ann in coco.get("annotations", []):
            try:
                cat_id = int(ann["category_id"])
                img_id = int(ann["image_id"])
            except Exception:
                continue
            cat_to_images.setdefault(cat_id, set()).add(img_id)
        prompts_map: Dict[int, List[str]] = {}
        for k, vals in payload.prompts_by_class.items():
            try:
                cid = int(k)
            except Exception:
                continue
            if not isinstance(vals, (list, tuple)):
                continue
            cleaned = [v.strip() for v in vals if isinstance(v, str) and v.strip()]
            if cleaned:
                prompts_map[cid] = cleaned
        if not prompts_map:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="search_prompts_empty")
        all_img_ids = list(images.keys())
        image_cache: Dict[int, Image.Image] = {}
        if target_class_id is not None:
            categories = [c for c in categories if int(c.get("id", categories.index(c))) == target_class_id]
            if not categories:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="search_class_not_found")

        total_steps = 0
        for idx, cat in enumerate(categories):
            cat_id = int(cat.get("id", idx))
            prompts = prompts_map.get(cat_id)
            if not prompts:
                continue
            pos_ids = _sample_images_for_category(
                cat_id,
                list(cat_to_images.get(cat_id, set())),
                payload.sample_per_class,
                payload.seed,
            )
            neg_ids = _sample_negative_images(
                cat_id,
                all_img_ids,
                cat_to_images,
                payload.negatives_per_class,
                payload.seed,
            )
            eval_count = len(set(pos_ids + neg_ids)) or 1
            total_steps += len(prompts) * eval_count
        job.total_steps = total_steps
        job.completed_steps = 0
        results: List[Dict[str, Any]] = []
        total_classes = len(categories) or 1
        for idx, cat in enumerate(categories):
            cat_id = int(cat.get("id", idx))
            class_name = str(cat.get("name", f"class_{cat_id}"))
            prompts = prompts_map.get(cat_id)
            if not prompts:
                continue
            job.message = f"Searching prompts for {class_name} ({idx + 1}/{total_classes})…"
            job.progress = idx / total_classes
            pos_ids = _sample_images_for_category(
                cat_id,
                list(cat_to_images.get(cat_id, set())),
                payload.sample_per_class,
                payload.seed,
            )
            neg_ids = _sample_negative_images(
                cat_id,
                all_img_ids,
                cat_to_images,
                payload.negatives_per_class,
                payload.seed,
            )
            eval_ids = list(dict.fromkeys([*pos_ids, *neg_ids]))
            candidate_results: List[Dict[str, Any]] = []
            for prompt in prompts:
                try:
                    job.logs.append(
                        {
                            "ts": time.time(),
                            "msg": f"Eval '{prompt}' on {len(eval_ids)} imgs (+{len(pos_ids)} pos / {len(neg_ids)} neg)",
                        }
                    )
                    if len(job.logs) > MAX_JOB_LOGS:
                        job.logs[:] = job.logs[-MAX_JOB_LOGS:]
                except Exception:
                    pass
                metrics = _evaluate_prompt_for_class(
                    prompt,
                    cat_id=cat_id,
                    image_ids=eval_ids,
                    gt_by_image_cat=gt_by_image_cat,
                    images=images,
                    score_threshold=payload.score_threshold,
                    max_dets=payload.max_dets,
                    iou_threshold=payload.iou_threshold,
                    image_cache=image_cache,
                )
                penalty = 1.0
                if payload.precision_floor > 0:
                    penalty = min(1.0, metrics["precision"] / max(payload.precision_floor, 1e-6))
                metrics["precision_penalty"] = penalty
                metrics["search_score"] = metrics["recall"] * (0.5 + 0.5 * metrics["det_rate"]) * penalty
                metrics["images_evaluated"] = len(eval_ids)
                metrics["positive_images"] = len(pos_ids)
                metrics["negative_images"] = len(neg_ids)
                candidate_results.append(metrics)
                job.completed_steps += max(1, len(eval_ids))
                if job.total_steps:
                    job.progress = min(1.0, job.completed_steps / job.total_steps)
                job.updated_at = time.time()
            candidate_results.sort(
                key=lambda m: (
                    m.get("search_score", 0.0),
                    m.get("recall", 0.0),
                    m.get("precision", 0.0),
                ),
                reverse=True,
            )
            best = candidate_results[0] if candidate_results else None
            results.append(
                {
                    "class_id": cat_id,
                    "class_name": class_name,
                    "positive_images": len(pos_ids),
                    "negative_images": len(neg_ids),
                    "best": best,
                    "candidates": candidate_results,
                }
            )
            job.progress = (idx + 1) / total_classes
            job.updated_at = time.time()
        job.status = "completed"
        job.message = "Done"
        job.result = {
            "classes": results,
            "config": payload.dict(),
            "dataset_id": payload.dataset_id,
            "mode": "search",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prompt search job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "Failed"
    finally:
        job.updated_at = time.time()


def _stable_sample_ids(ids: Sequence[int], cap: int, seed: int, salt: str = "") -> List[int]:
    pool = list(ids)
    if not pool or cap <= 0:
        return []
    rng = random.Random(f"{seed}:{salt}")
    rng.shuffle(pool)
    return pool[: min(len(pool), cap)]


def _sample_images_for_category(class_id: int, img_ids: Sequence[int], sample_size: int, seed: int) -> List[int]:
    return _stable_sample_ids(img_ids, cap=sample_size, seed=seed, salt=f"pos:{class_id}")


def _sample_negative_images(
    class_id: int,
    all_img_ids: Sequence[int],
    cat_to_images: Dict[int, set[int]],
    sample_size: int,
    seed: int,
) -> List[int]:
    positives = set(cat_to_images.get(class_id, set()))
    candidates = [img_id for img_id in all_img_ids if img_id not in positives]
    return _stable_sample_ids(candidates, cap=sample_size, seed=seed, salt=f"neg:{class_id}")


def _build_seed_threshold_sweep_grid(
    *, base_seed_threshold: float, observed_scores: Sequence[float], limit: int = 64
) -> List[float]:
    values = {0.0, float(max(0.0, min(1.0, base_seed_threshold)))}
    for score in observed_scores or []:
        try:
            values.add(float(max(0.0, min(1.0, score))))
        except Exception:
            continue
    ordered = sorted(values)
    if limit and len(ordered) > limit:
        step = max(1, int(len(ordered) / max(1, limit)))
        sampled = ordered[::step]
        if 0.0 not in sampled:
            sampled.insert(0, 0.0)
        if base_seed_threshold not in sampled:
            sampled.append(float(base_seed_threshold))
        ordered = sorted(set(sampled))[:limit]
    return ordered


def _compute_seed_threshold_curve(
    *,
    gt_best_scores: Dict[Any, float],
    fp_scores: Sequence[float],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    curve = []
    total_gt = len(gt_best_scores)
    for thr in sorted(thresholds):
        matches = sum(1 for v in gt_best_scores.values() if float(v) >= float(thr))
        fps = sum(1 for v in fp_scores if float(v) >= float(thr))
        precision = matches / (matches + fps) if (matches + fps) else 1.0
        recall = matches / total_gt if total_gt else 0.0
        curve.append(
            {
                "threshold": float(thr),
                "matches": int(matches),
                "fps": int(fps),
                "precision": float(precision),
                "recall": float(recall),
            }
        )
    return curve


def _select_seed_threshold_operating_point(
    curve: Sequence[Dict[str, Any]],
    *,
    min_precision: Optional[float] = None,
    max_fps: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not curve:
        return None
    candidates = []
    for point in curve:
        if min_precision is not None and float(point.get("precision", 0.0)) < float(min_precision):
            continue
        if max_fps is not None and int(point.get("fps", 0)) > int(max_fps):
            continue
        candidates.append(point)
    if candidates:
        candidates.sort(
            key=lambda p: (
                int(p.get("matches", 0)),
                float(p.get("precision", 0.0)),
                -float(p.get("threshold", 0.0)),
            ),
            reverse=True,
        )
        return candidates[0]
    # fallback: pick highest precision, then matches, then lowest threshold
    fallback = list(curve)
    fallback.sort(
        key=lambda p: (
            float(p.get("precision", 0.0)),
            int(p.get("matches", 0)),
            -float(p.get("threshold", 0.0)),
        ),
        reverse=True,
    )
    return fallback[0] if fallback else None


def _summarize_seed_threshold_curve_for_prompt(
    *,
    gt_best_scores: Dict[Any, float],
    fp_scores: Sequence[float],
    base_seed_threshold: float,
    curve_limit: int = 12,
) -> Dict[str, Any]:
    thresholds = _build_seed_threshold_sweep_grid(
        base_seed_threshold=base_seed_threshold,
        observed_scores=list(gt_best_scores.values()) + list(fp_scores),
        limit=curve_limit,
    )
    curve = _compute_seed_threshold_curve(
        gt_best_scores=gt_best_scores,
        fp_scores=fp_scores,
        thresholds=thresholds,
    )
    recommended = _select_seed_threshold_operating_point(curve, min_precision=None)
    rec_thr = float(recommended.get("threshold")) if recommended else float(base_seed_threshold)
    return {
        "seed_threshold_curve": curve[:curve_limit],
        "seed_threshold_base": float(base_seed_threshold),
        "seed_threshold_recommended": float(rec_thr),
    }


def _build_steps_recipe_step_list_from_selected_stats(
    selected: Sequence[Dict[str, Any]],
    *,
    prompts_fallback: Optional[Sequence[str]],
    payload: AgentMiningRequest,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    steps: List[Dict[str, Any]] = []
    prompts: List[str] = []
    if not selected and prompts_fallback:
        for prompt in prompts_fallback:
            prompts.append(str(prompt))
            steps.append(
                {
                    "prompt": str(prompt),
                    "seed_threshold": float(getattr(payload, "seed_threshold", 0.0)),
                }
            )
        return prompts, steps
    for entry in selected:
        prompt = str(entry.get("prompt") or "")
        if not prompt:
            continue
        seed_thr = entry.get("selected_seed_threshold")
        if seed_thr is None:
            seed_thr = entry.get("seed_threshold")
        if seed_thr is None:
            seed_thr = getattr(payload, "seed_threshold", 0.0)
        prompts.append(prompt)
        steps.append({"prompt": prompt, "seed_threshold": float(seed_thr)})
    return prompts, steps


def _normalize_steps_for_head_tuning(
    steps: Sequence[Dict[str, Any]], *, payload: AgentMiningRequest
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for step in steps or []:
        if not step.get("enabled", True):
            continue
        prompt = step.get("prompt")
        if not prompt:
            continue
        normalized.append(
            {
                "prompt": prompt,
                "seed_threshold": float(step.get("seed_threshold", payload.seed_threshold)),
                "expand_threshold": float(step.get("expand_threshold", payload.expand_threshold)),
                "max_visual_seeds": int(
                    step.get("max_visual_seeds", payload.steps_max_visual_seeds_per_step)
                ),
                "seed_dedupe_iou": float(step.get("seed_dedupe_iou", payload.seed_dedupe_iou)),
                "dedupe_iou": float(step.get("dedupe_iou", payload.dedupe_iou)),
                "max_results": int(step.get("max_results", payload.max_results)),
            }
        )
    return normalized


def _build_steps_tier2_candidate_grid(
    *, base_seed_dedupe_iou: float, base_dedupe_iou: float, max_trials: int
) -> List[Dict[str, Any]]:
    offsets = [-0.1, 0.0, 0.1]
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for off_seed in offsets:
        for off_det in offsets:
            seed_val = max(0.0, min(1.0, float(base_seed_dedupe_iou) + off_seed))
            det_val = max(0.0, min(1.0, float(base_dedupe_iou) + off_det))
            key = (round(seed_val, 4), round(det_val, 4))
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"seed_dedupe_iou": seed_val, "dedupe_iou": det_val})
    if max_trials:
        candidates = candidates[: max(1, int(max_trials))]
    return candidates


def _select_steps_from_seed_prompt_stats(
    prompt_stats: Sequence[Dict[str, Any]],
    *,
    max_steps: int,
    target_precision: float,
    max_candidates_per_prompt: int = 4,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for entry in prompt_stats:
        curve = entry.get("seed_threshold_curve") or []
        curve = list(curve)[: max_candidates_per_prompt]
        candidates = [
            c for c in curve if float(c.get("precision", 0.0)) >= float(target_precision)
        ]
        if candidates:
            candidates.sort(
                key=lambda c: (
                    int(c.get("matches", 0)),
                    float(c.get("precision", 0.0)),
                    -float(c.get("threshold", 0.0)),
                ),
                reverse=True,
            )
            best = candidates[0]
        elif curve:
            curve.sort(
                key=lambda c: (
                    float(c.get("precision", 0.0)),
                    int(c.get("matches", 0)),
                    -float(c.get("threshold", 0.0)),
                ),
                reverse=True,
            )
            best = curve[0]
        else:
            best = {"threshold": entry.get("seed_threshold", 0.0)}
        threshold = float(best.get("threshold", 0.0))
        gt_scores = entry.get("gt_best_scores") or {}
        matched_keys = {
            key for key, score in gt_scores.items() if float(score) >= float(threshold)
        }
        selected.append(
            {
                **entry,
                "selected_seed_threshold": threshold,
                "matched_keys": matched_keys,
            }
        )
    selected.sort(
        key=lambda e: (len(e.get("matched_keys") or []), float(e.get("precision", 0.0))),
        reverse=True,
    )
    coverage: set = set()
    chosen: List[Dict[str, Any]] = []
    for entry in selected:
        if len(chosen) >= int(max_steps):
            break
        matched_keys = set(entry.get("matched_keys") or [])
        if matched_keys and matched_keys.issubset(coverage):
            continue
        coverage |= matched_keys
        chosen.append(entry)
    return chosen, {"target_precision": float(target_precision)}


def _generate_steps_global_mutations(
    *,
    base_candidate: Dict[str, Any],
    seed_stats: Sequence[Dict[str, Any]],
    payload: AgentMiningRequest,
    max_mutations: int,
    target_precision: float,
    enable_max_results: bool,
    enable_ordering: bool,
) -> List[Dict[str, Any]]:
    base_steps = _normalize_steps_for_head_tuning(base_candidate.get("steps") or [], payload=payload)
    stats_by_prompt = {
        str(s.get("prompt") or ""): s for s in seed_stats if str(s.get("prompt") or "")
    }
    mutations: List[Dict[str, Any]] = []
    for prompt in sorted(stats_by_prompt.keys()):
        stat = stats_by_prompt[prompt]
        thr = stat.get("seed_threshold_recommended")
        if thr is None:
            curve = stat.get("seed_threshold_curve") or []
            if curve:
                thr = curve[0].get("threshold")
        if thr is None:
            thr = payload.seed_threshold
        steps = []
        seen = set()
        for step in base_steps:
            if step["prompt"] == prompt:
                step = {**step, "seed_threshold": float(thr)}
            if step["prompt"] in seen:
                continue
            seen.add(step["prompt"])
            steps.append(step)
        if prompt not in seen:
            steps.append(
                {
                    "prompt": prompt,
                    "seed_threshold": float(thr),
                    "expand_threshold": float(payload.expand_threshold),
                    "max_visual_seeds": int(payload.steps_max_visual_seeds_per_step),
                    "seed_dedupe_iou": float(payload.seed_dedupe_iou),
                    "dedupe_iou": float(payload.dedupe_iou),
                    "max_results": int(payload.max_results if enable_max_results else payload.max_results),
                }
            )
        if enable_ordering:
            steps = sorted(steps, key=lambda s: str(s.get("prompt") or ""))
        mutations.append({"steps": steps, "sig": f"{prompt}:{float(thr):.6f}"})
        if max_mutations and len(mutations) >= max_mutations:
            break
    return mutations


def _successive_halving_search(
    *,
    candidates: Sequence[Any],
    budgets: Sequence[int],
    evaluator: Any,
    keep_ratio: float,
) -> Tuple[Any, List[Dict[str, Any]]]:
    if not budgets or sorted(budgets) != list(budgets):
        raise ValueError("budgets must be increasing")
    history = []
    current = list(candidates)
    for budget in budgets:
        scored = []
        for cand in current:
            score, meta = evaluator(cand, budget)
            scored.append((score, cand, meta))
        scored.sort(key=lambda row: row[0], reverse=True)
        keep = max(1, int(len(scored) * keep_ratio))
        history.append({"budget": budget, "candidates": [row[1] for row in scored], "scores": [row[0] for row in scored]})
        current = [row[1] for row in scored[:keep]]
    best = current[0] if current else None
    return best, history


def _run_steps_global_successive_halving_rounds(
    *,
    base_candidate: Dict[str, Any],
    budgets: Sequence[int],
    keep_ratio: float,
    rounds: int,
    max_trials: int,
    mutate: Any,
    evaluator: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    best = dict(base_candidate)
    history: List[Dict[str, Any]] = []
    for round_idx in range(int(rounds)):
        candidates = mutate(best, round_idx) or []
        candidates = list(candidates)[: max_trials]
        best_candidate, stage_history = _successive_halving_search(
            candidates=candidates, budgets=budgets, evaluator=evaluator, keep_ratio=keep_ratio
        )
        if best_candidate is not None:
            best = best_candidate
        history.append({"round": round_idx, "history": stage_history, "best": best})
    return best, history


def _refine_steps_prompt_subset_seed_stage(
    prompt_stats: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
    *,
    max_steps: int,
    target_precision: float,
    max_iters: int,
    top_k: int,
    base_seed_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    info: Dict[str, Any] = {"enabled": True, "history": []}
    refined = [dict(s) for s in selected]
    # Drop redundant prompts.
    coverage_map = {}
    for entry in refined:
        matched_raw = entry.get("matched_keys")
        if matched_raw:
            matched = set(matched_raw)
            entry["matched_keys"] = matched
            coverage_map[entry.get("prompt")] = matched
            continue
        threshold = float(entry.get("selected_seed_threshold") or base_seed_threshold)
        gt_scores = entry.get("gt_best_scores") or {}
        matched = {k for k, v in gt_scores.items() if float(v) >= threshold}
        entry["matched_keys"] = matched
        coverage_map[entry.get("prompt")] = matched
    changed = True
    while changed:
        changed = False
        for entry in list(refined):
            prompt = entry.get("prompt")
            matched = set(entry.get("matched_keys") or [])
            others = set()
            for other in refined:
                if other is entry:
                    continue
                others |= set(other.get("matched_keys") or [])
            if matched and matched.issubset(others):
                refined.remove(entry)
                info["history"].append({"op": "drop_redundant", "dropped": prompt})
                changed = True
                break
    # Swap in a better prompt if it adds coverage.
    if max_iters and prompt_stats:
        current_coverage = set()
        for entry in refined:
            current_coverage |= set(entry.get("matched_keys") or [])
        for candidate in prompt_stats[:top_k]:
            prompt = candidate.get("prompt")
            if not prompt:
                continue
            threshold = float(candidate.get("seed_threshold_recommended") or base_seed_threshold)
            gt_scores = candidate.get("gt_best_scores") or {}
            cand_keys = {k for k, v in gt_scores.items() if float(v) >= threshold}
            if not cand_keys:
                continue
            if not cand_keys.issubset(current_coverage):
                # replace weakest prompt
                if refined:
                    refined.sort(key=lambda e: len(e.get("matched_keys") or []))
                    dropped = refined.pop(0)
                    info["history"].append({"op": "swap", "dropped": dropped.get("prompt"), "added": prompt})
                refined.append(
                    {
                        **candidate,
                        "selected_seed_threshold": threshold,
                        "matched_keys": cand_keys,
                    }
                )
                current_coverage |= cand_keys
                if len(refined) >= max_steps:
                    break
    return refined[:max_steps], info


def _agent_compact_tool_response(result: AgentToolResult) -> Dict[str, Any]:
    payload = {"tool": result.name, "result": result.result}
    if result.error:
        payload["error"] = result.error
    return payload


def _parse_tool_call_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not raw_text:
        return None, "empty"
    snippet = raw_text
    if "<tool_call>" in raw_text and "</tool_call>" in raw_text:
        try:
            snippet = raw_text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        except Exception:
            snippet = raw_text
    snippet = snippet.strip()
    try:
        payload = json.loads(snippet)
        return payload, None
    except Exception as exc:  # noqa: BLE001
        return None, f"parse_error:{exc}"


def _infer_clip_model_from_embedding_dim(embed_dim: int, active_name: Optional[str] = None) -> Optional[str]:
    return _infer_clip_model_from_embedding_dim_impl(
        embed_dim, active_name=active_name or clip_model_name or DEFAULT_CLIP_MODEL
    )


def _load_clip_head_from_classifier(path: Path) -> Optional[Dict[str, Any]]:
    return _load_clip_head_from_classifier_impl(
        path,
        joblib_load_fn=joblib.load,
        http_exception_cls=HTTPException,
        clip_head_background_indices_fn=_clip_head_background_indices,
        resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
        infer_clip_model_fn=_infer_clip_model_from_embedding_dim,
        active_clip_model_name=clip_model_name,
        default_clip_model=DEFAULT_CLIP_MODEL,
        logger=logger,
    )


def _build_prompt_recipe(
    candidates: List[Dict[str, Any]],
    all_gt_keys: set[str],
    per_image_gt: Dict[int, int],
    images: Dict[int, Dict[str, Any]],
    image_ids: List[int],
    gt_index: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    return _build_prompt_recipe_impl(candidates, all_gt_keys, per_image_gt, images, image_ids, gt_index)


def _build_qwen_caption_prompt(*args: Any, **kwargs: Any) -> Any:
    return _build_qwen_caption_prompt_impl(*args, **kwargs)


def _normalize_agent_recipe_execution_plan(recipe_obj: Dict[str, Any]) -> Dict[str, Any]:
    mode = _classify_agent_recipe_mode_impl(recipe_obj)
    if mode == "sam3_steps":
        return {
            "schema_version": int(recipe_obj.get("schema_version") or 2),
            "mode": "sam3_steps",
            "steps": recipe_obj.get("steps") or [],
        }
    if mode == "sam3_greedy":
        return {
            "schema_version": int(recipe_obj.get("schema_version") or 1),
            "mode": "sam3_greedy",
            "text_prompts": recipe_obj.get("text_prompts") or [],
            "params": recipe_obj.get("params") or {},
        }
    return recipe_obj


_classify_agent_recipe_mode = _classify_agent_recipe_mode_impl
_validate_agent_recipe_structure = _validate_agent_recipe_structure_impl


def _update_best_clip_head_sweep_summary(
    *,
    best_summary: Optional[Dict[str, Any]],
    best_key: Optional[Tuple[float, float]],
    total_gt: int,
    total_images: int,
    matched: int,
    fps: int,
    duplicates: int,
    preds: int,
    det_images: int,
    min_prob: float,
    margin: float,
    target_precision: float,
) -> Tuple[Dict[str, Any], Tuple[float, float]]:
    precision = matched / preds if preds else 0.0
    meets = precision >= float(target_precision)
    key = (float(min_prob), float(margin))
    summary = {
        "clip_head_min_prob": float(min_prob),
        "clip_head_margin": float(margin),
        "clip_head_meets_target_precision": bool(meets),
        "matched": int(matched),
        "fps": int(fps),
        "duplicates": int(duplicates),
        "preds": int(preds),
        "det_images": int(det_images),
        "precision": float(precision),
        "total_gt": int(total_gt),
        "total_images": int(total_images),
    }
    if best_summary is None:
        return summary, key
    best_meets = bool(best_summary.get("clip_head_meets_target_precision"))
    if meets and not best_meets:
        return summary, key
    if meets == best_meets:
        if meets:
            # prefer higher matched, then higher precision
            if (int(matched), float(precision)) > (
                int(best_summary.get("matched", 0)),
                float(best_summary.get("precision", 0.0)),
            ):
                return summary, key
        else:
            if float(precision) > float(best_summary.get("precision", 0.0)):
                return summary, key
    return best_summary, best_key or key


def _run_prompt_recipe_job(job: PromptHelperJob, payload: PromptRecipeRequest) -> None:
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    job.status = "running"
    job.message = "Loading dataset…"
    job.request = {"mode": "recipe", **payload.dict()}
    job.updated_at = time.time()
    try:
        if not payload.prompts:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="recipe_prompts_required")
        dataset_root = _resolve_sam3_or_qwen_dataset_impl(
            payload.dataset_id,
            list_all_datasets_fn=_list_all_datasets,
            resolve_dataset_legacy_fn=lambda dataset_id: _resolve_dataset_legacy_impl(
                dataset_id,
                qwen_root=QWEN_DATASET_ROOT,
                sam3_root=SAM3_DATASET_ROOT,
                registry_root=DATASET_REGISTRY_ROOT,
                http_exception_cls=HTTPException,
            ),
        )
        coco, gt_by_image_cat, images = _load_coco_index_impl(dataset_root)
        categories = coco.get("categories") or []
        cat_entry = next((c for c in categories if int(c.get("id", categories.index(c))) == payload.class_id), None)
        if not cat_entry:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="recipe_class_not_found")
        class_name = str(cat_entry.get("name", f"class_{payload.class_id}"))
        cat_to_images: Dict[int, set[int]] = {}
        for ann in coco.get("annotations", []):
            try:
                cat_id = int(ann["category_id"])
                img_id = int(ann["image_id"])
            except Exception:
                continue
            cat_to_images.setdefault(cat_id, set()).add(img_id)
        pos_ids = _sample_images_for_category(
            payload.class_id,
            list(cat_to_images.get(payload.class_id, set())),
            payload.sample_size,
            payload.seed,
        )
        all_img_ids = list(images.keys())
        neg_ids = _sample_negative_images(
            payload.class_id,
            all_img_ids,
            cat_to_images,
            payload.negatives,
            payload.seed,
        )
        eval_ids = list(dict.fromkeys([*pos_ids, *neg_ids]))
        image_cache: Dict[int, Image.Image] = {}
        gt_index_all, all_gt_keys_all, per_image_gt_all = _build_gt_index_for_class_impl(
            gt_by_image_cat,
            payload.class_id,
            xywh_to_xyxy_fn=_xywh_to_xyxy,
        )
        gt_index = {img_id: entries for img_id, entries in gt_index_all.items() if img_id in eval_ids}
        per_image_gt = {img_id: per_image_gt_all.get(img_id, 0) for img_id in eval_ids}
        all_gt_keys = set()
        for entries in gt_index.values():
            for key, _ in entries:
                all_gt_keys.add(key)
        if not eval_ids:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="recipe_no_images_sampled")
        if not all_gt_keys:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="recipe_no_gt_for_class")
        thresholds_cache: Dict[str, List[float]] = {}
        total_steps = 0
        for prompt_entry in payload.prompts:
            key = prompt_entry.prompt
            thresholds_cache[key] = _normalize_recipe_thresholds(
                prompt_entry.thresholds or payload.threshold_candidates,
                payload.score_threshold,
            )
            total_steps += len(thresholds_cache[key]) * max(1, len(eval_ids))
        job.total_steps = total_steps
        job.completed_steps = 0
        candidates: List[Dict[str, Any]] = []
        for idx, prompt_entry in enumerate(payload.prompts):
            thresholds = thresholds_cache.get(prompt_entry.prompt) or [payload.score_threshold]
            min_threshold = min(thresholds) if thresholds else payload.score_threshold
            detections = _collect_prompt_detections_impl(
                prompt_entry.prompt,
                min_threshold,
                image_ids=eval_ids,
                images=images,
                image_cache=image_cache,
                max_dets=payload.max_dets,
                run_sam3_text_inference_fn=_run_sam3_text_inference,
                yolo_to_xyxy_fn=_yolo_to_xyxy,
            )
            for thr in thresholds:
                try:
                    job.logs.append(
                        {
                            "ts": time.time(),
                            "msg": f"Eval prompt {idx + 1}/{len(payload.prompts)} @ {thr:.2f} on {len(eval_ids)} images",
                        }
                    )
                    if len(job.logs) > MAX_JOB_LOGS:
                        job.logs[:] = job.logs[-MAX_JOB_LOGS:]
                except Exception:
                    pass
                metrics = _evaluate_prompt_candidate_impl(
                    prompt_entry.prompt,
                    thr,
                    cat_id=payload.class_id,
                    image_ids=eval_ids,
                    gt_index=gt_index,
                    images=images,
                    iou_threshold=payload.iou_threshold,
                    max_dets=payload.max_dets,
                    image_cache=image_cache,
                    cached_detections=detections,
                    run_sam3_text_inference_fn=_run_sam3_text_inference,
                    yolo_to_xyxy_fn=_yolo_to_xyxy,
                    iou_fn=_iou,
                )
                metrics["class_name"] = class_name
                metrics["class_id"] = payload.class_id
                metrics["image_count"] = len(eval_ids)
                candidates.append(metrics)
                job.completed_steps += len(eval_ids)
                if job.total_steps:
                    job.progress = min(1.0, job.completed_steps / job.total_steps)
                job.message = f"Evaluated {prompt_entry.prompt} ({job.completed_steps}/{job.total_steps} images)"
                job.updated_at = time.time()
        recipe, coverage_by_image = _build_prompt_recipe_impl(
            candidates,
            all_gt_keys,
            per_image_gt,
            images,
            eval_ids,
            gt_index,
        )
        job.status = "completed"
        job.message = "Done"
        job.result = {
            "mode": "recipe",
            "dataset_id": payload.dataset_id,
            "class_id": payload.class_id,
            "class_name": class_name,
            "positive_images": len(pos_ids),
            "negative_images": len(neg_ids),
            "positive_image_ids": pos_ids,
            "negative_image_ids": neg_ids,
            "evaluated_image_ids": eval_ids,
            "gt_count": len(all_gt_keys),
            "config": payload.dict(),
            "candidates": [
                {
                    **{k: v for k, v in cand.items() if k not in {"matched_gt_keys", "matches_by_image"}},
                    "matched_gt": len(cand.get("matched_gt_keys") or []),
                }
                for cand in candidates
            ],
            "recipe": recipe,
            "coverage_by_image": coverage_by_image,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prompt recipe job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "Failed"
    finally:
        job.updated_at = time.time()


def _run_agent_mining_job(job: AgentMiningJob, payload: AgentMiningRequest) -> None:
    """
    SAM3 Recipe Mining (steps mode).

    Builds schema-v2 multi-step recipes using:
    - prompt bank (text prompts)
    - pretrained CLIP head (required) for filtering + cleanliness tuning

    Then evaluates the full steps pipeline on a deterministic sample of images.
    """
    with AGENT_MINING_JOBS_LOCK:
        AGENT_MINING_JOBS[job.job_id] = job
    job.status = "running"
    job.message = "Preparing image sample…"
    job.request = payload.dict()
    job.updated_at = time.time()
    start_ts = time.time()
    compute_estimate_info: Optional[Dict[str, Any]] = None

    def _log(msg: str) -> None:
        try:
            job.logs.append({"ts": time.time(), "msg": msg})
            if len(job.logs) > MAX_JOB_LOGS:
                job.logs[:] = job.logs[-MAX_JOB_LOGS:]
        except Exception:
            pass
        job.updated_at = time.time()

    def _cancelled() -> bool:
        return bool(job.cancel_event.is_set())

    try:
        _enforce_agent_mining_cache_limits(AGENT_MINING_DET_CACHE_ROOT, allow_when_running=False)

        dataset_root = _resolve_sam3_or_qwen_dataset_impl(
            payload.dataset_id,
            list_all_datasets_fn=_list_all_datasets,
            resolve_dataset_legacy_fn=lambda dataset_id: _resolve_dataset_legacy_impl(
                dataset_id,
                qwen_root=QWEN_DATASET_ROOT,
                sam3_root=SAM3_DATASET_ROOT,
                registry_root=DATASET_REGISTRY_ROOT,
                http_exception_cls=HTTPException,
            ),
        )
        _log(f"Dataset resolved at {dataset_root}")
        _log(
            "Request: "
            f"eval_images={payload.eval_image_count} "
            f"sample_seed={payload.split_seed}"
        )

        sample = _ensure_agent_mining_sample(
            payload.dataset_id,
            dataset_root,
            sample_size=payload.eval_image_count,
            seed=payload.split_seed,
        )
        eval_ids = [int(i) for i in (sample.get("sample_ids") or [])]
        if not eval_ids:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_empty_sample")
        _log(f"Prepared sample with {len(eval_ids)} images (cached={sample.get('_cached', False)})")
        sample_hash = "unknown"
        try:
            sample_hash = hashlib.sha256(",".join(str(i) for i in eval_ids).encode("utf-8")).hexdigest()[:12]
        except Exception:
            sample_hash = "unknown"
        job.progress = 0.05

        coco, gt_by_image_cat, images = _load_coco_index_impl(dataset_root)
        categories = coco.get("categories") or []
        _log(f"Loaded COCO with {len(categories)} categories")

        cat_filter: Optional[set[int]] = None
        if payload.classes:
            cat_filter = set()
            for cid in payload.classes:
                try:
                    cat_filter.add(int(cid))
                except Exception:
                    continue

        selected_categories: List[Tuple[int, str]] = []
        for idx, cat in enumerate(categories):
            try:
                cid = int(cat.get("id", idx))
            except Exception:
                cid = idx
            if cat_filter and cid not in cat_filter:
                continue
            selected_categories.append((cid, str(cat.get("name", f"class_{cid}"))))
        if not selected_categories:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_no_classes_selected")

        _log(
            "Config: "
            + f"text_thr={payload.seed_threshold} visual_thr={payload.expand_threshold} "
            + f"cand_iou={payload.seed_dedupe_iou} out_iou={payload.dedupe_iou} "
            + f"mask_thr={payload.mask_threshold} max_results={payload.max_results} iou_eval={payload.iou_threshold} "
            + f"llm_prompts={payload.prompt_llm_max_prompts} workers_per_gpu={getattr(payload, 'max_workers_per_device', 1)} "
            + f"steps(max_steps={int(payload.steps_max_steps_per_recipe)}, candidates/step={int(payload.steps_max_visual_seeds_per_step)}) "
            + f"bg_guard={bool(payload.clip_head_background_guard)} "
            + f"bg_apply={getattr(payload, 'clip_head_background_apply', 'final')} "
            + f"bg_margin={getattr(payload, 'clip_head_background_margin', 0.0)} "
            + f"bg_penalty={getattr(payload, 'clip_head_background_penalty', 0.0)}"
        )

        # Pretrained CLIP head (LogReg) is required for Agent Mining (recipe mining).
        clip_head_path = _resolve_agent_clip_classifier_path_impl(
            payload.clip_head_classifier_path,
            allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
            allowed_exts=CLASSIFIER_ALLOWED_EXTS,
            path_is_within_root_fn=_path_is_within_root_impl,
            http_exception_cls=HTTPException,
        )
        if clip_head_path is None:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_clip_head_required")
        clip_head = _load_clip_head_from_classifier_impl(
            clip_head_path,
            joblib_load_fn=joblib.load,
            http_exception_cls=HTTPException,
            clip_head_background_indices_fn=_clip_head_background_indices,
            resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
            infer_clip_model_fn=_infer_clip_model_from_embedding_dim_impl,
            active_clip_model_name=clip_model_name,
            default_clip_model=DEFAULT_CLIP_MODEL,
            logger=logger,
        )
        if not isinstance(clip_head, dict):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_clip_head_required")
        head_encoder_type = str(clip_head.get("encoder_type") or "clip").lower().strip()
        if not head_encoder_type:
            head_encoder_type = "clip"
        prefilter_allowed = head_encoder_type == "clip"
        head_mode_bits = ""
        try:
            if bool(getattr(payload, "clip_head_auto_tune", True)):
                head_mode_bits = (
                    f"(auto-tune: target_precision={float(getattr(payload, 'clip_head_target_precision', 0.9)):.2f}; "
                    f"seed min_prob={payload.clip_head_min_prob} margin={payload.clip_head_margin})"
                )
            else:
                head_mode_bits = f"(fixed thresholds: min_prob={payload.clip_head_min_prob} margin={payload.clip_head_margin})"
        except Exception:
            head_mode_bits = ""
        clip_name = clip_head.get("clip_model") if isinstance(clip_head, dict) else None
        clip_name_str = str(clip_name).strip() if isinstance(clip_name, str) and clip_name.strip() else "unknown"
        classes_list = clip_head.get("classes") if isinstance(clip_head.get("classes"), list) else []
        _log(
            "Pretrained CLIP head enabled: "
            f"{clip_head_path.name} "
            f"(clip={clip_name_str}, classes={len(classes_list)}, mode={clip_head.get('proba_mode')}) "
            f"{head_mode_bits}"
        )
        if classes_list:
            preview_limit = 60
            preview_items = []
            for idx, label in enumerate(classes_list):
                preview_items.append(f"{idx}:{label}")
                if len(preview_items) >= preview_limit:
                    break
            preview = ", ".join(preview_items)
            suffix = f", … (+{len(classes_list) - preview_limit} more)" if len(classes_list) > preview_limit else ""
            _log(f"CLIP head classes (index:name): {preview}{suffix}")

        sample_images = int(len(eval_ids))
        class_count = int(len(selected_categories))
        per_image_units = int(payload.steps_max_steps_per_recipe) * (1 + int(payload.steps_max_visual_seeds_per_step))
        speed_factor = _estimate_steps_speed_factor(payload, allow_prefilter=prefilter_allowed)
        base_units_per_class = float(sample_images * per_image_units) * float(speed_factor)
        global_enabled = bool(getattr(payload, "steps_optimize_global", False)) and bool(clip_head)
        global_units_per_class = 0.0
        budgets: List[int] = []
        invalid_budgets = False
        if global_enabled:
            global_images, budgets, invalid_budgets = _estimate_agent_global_optimizer_image_evals(
                val_images=sample_images,
                eval_caps=list(getattr(payload, "steps_optimize_global_eval_caps", []) or []),
                keep_ratio=float(getattr(payload, "steps_optimize_global_keep_ratio", 0.5) or 0.5),
                rounds=int(getattr(payload, "steps_optimize_global_rounds", 1) or 1),
                max_trials=int(getattr(payload, "steps_optimize_global_max_trials", 1) or 1),
                mutations_per_round=int(getattr(payload, "steps_optimize_global_mutations_per_round", 1) or 1),
            )
            if not invalid_budgets:
                global_units_per_class = float(global_images * per_image_units) * float(speed_factor)
        total_units_per_class = float(base_units_per_class + global_units_per_class)
        total_units_all = float(total_units_per_class * class_count) if class_count > 0 else None
        compute_estimate_info = {
            "sample_images": sample_images,
            "class_count": class_count,
            "steps": int(payload.steps_max_steps_per_recipe),
            "seeds_per_step": int(payload.steps_max_visual_seeds_per_step),
            "per_image_units": int(per_image_units),
            "speed_factor": float(speed_factor),
            "base_units_per_class": float(base_units_per_class),
            "global_units_per_class": float(global_units_per_class),
            "total_units_per_class": float(total_units_per_class),
            "total_units_all_classes": float(total_units_all) if total_units_all is not None else None,
            "global_optimizer": {
                "enabled": bool(global_enabled),
                "eval_caps": budgets,
                "invalid_budgets": bool(invalid_budgets),
            },
        }
        _log(
            "Compute estimate: "
            f"sample_images={sample_images} steps={int(payload.steps_max_steps_per_recipe)} candidates/step={int(payload.steps_max_visual_seeds_per_step)} "
            f"speed_factor={float(speed_factor):.2f} base_units/class={base_units_per_class:.0f} "
            f"global_units/class={global_units_per_class:.0f} total_units/class={total_units_per_class:.0f}"
            + (f" total_units/all_classes={total_units_all:.0f}" if total_units_all is not None else "")
        )

        # Prompt mining (Qwen) per class.
        base_prompts_all: List[str] = []
        if payload.extra_prompts_by_class and isinstance(payload.extra_prompts_by_class, dict):
            raw_base = payload.extra_prompts_by_class.get("__base__")
            if isinstance(raw_base, list):
                base_prompts_all = [p for p in raw_base if isinstance(p, str) and p.strip()]
            elif isinstance(raw_base, str) and raw_base.strip():
                base_prompts_all = [raw_base.strip()]
        base_prompts_all = _sanitize_prompts_impl(base_prompts_all)
        if base_prompts_all:
            _log(f"Base prompts (all classes): {', '.join(base_prompts_all)}")

        prefilter_cfg = _resolve_steps_prompt_prefilter_config(payload, allow_prefilter=prefilter_allowed)
        clip_model_for_prefilter: Optional[str] = None
        if isinstance(clip_head, dict):
            raw_model = clip_head.get("clip_model")
            if isinstance(raw_model, str) and raw_model.strip():
                clip_model_for_prefilter = raw_model.strip()
        if prefilter_cfg.get("enabled"):
            _log(
                "CLIP prompt prefilter enabled: "
                f"mode={prefilter_cfg.get('mode')} sample_size={prefilter_cfg.get('sample_size')} "
                f"keep_ratio={float(prefilter_cfg.get('keep_ratio') or 0.0):.2f}"
            )
        elif prefilter_cfg.get("requested") and not prefilter_allowed:
            _log(f"CLIP prompt prefilter disabled: head encoder_type={head_encoder_type}")

        bg_indices = _clip_head_background_indices(classes_list) if clip_head else []
        prompt_bg_drop_cfg = _resolve_steps_prompt_bg_drop_config(payload, allow_drop=bool(bg_indices))
        if prompt_bg_drop_cfg.get("enabled"):
            _log(
                "Prompt background drop enabled: "
                f"mode={prompt_bg_drop_cfg.get('mode')} min_checked={prompt_bg_drop_cfg.get('min_checked')} "
                f"drop_rate={float(prompt_bg_drop_cfg.get('drop_rate') or 0.0):.2f}"
            )
        elif prompt_bg_drop_cfg.get("requested") and prompt_bg_drop_cfg.get("disabled_reason") == "no_background_classes":
            _log("Prompt background drop disabled: no __bg_* classes in CLIP head.")

        prepared_prompts: Dict[int, List[str]] = {}
        prompt_prefilter_stats: Dict[int, Dict[str, Any]] = {}
        for cid, name in selected_categories:
            if _cancelled():
                break
            base = []
            if payload.text_prompts_by_class and cid in payload.text_prompts_by_class:
                base = [p for p in payload.text_prompts_by_class.get(cid) or [] if isinstance(p, str) and p.strip()]
            if not base:
                base = [name]
            user_extras: List[str] = []
            if payload.extra_prompts_by_class and isinstance(payload.extra_prompts_by_class, dict):
                raw_extra = payload.extra_prompts_by_class.get(name)
                if isinstance(raw_extra, list):
                    user_extras = [p for p in raw_extra if isinstance(p, str) and p.strip()]
                elif isinstance(raw_extra, str) and raw_extra.strip():
                    user_extras = [raw_extra.strip()]
            if user_extras:
                _log(f"User extra prompts for {name}: {', '.join(user_extras)}")
            base = _sanitize_prompts_impl([*base, *user_extras]) or [name]
            extras: List[str] = []
            if payload.prompt_llm_max_prompts > 0:
                extras = _expand_prompts_with_prompt_llm(
                    name,
                    base,
                    payload.prompt_llm_max_prompts,
                    log_fn=_log,
                    max_new_tokens=payload.prompt_max_new_tokens,
                )
            merged: List[str] = []
            seen = set()
            for p in [*base, *extras, *base_prompts_all]:
                key = str(p).lower().strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(str(p))
            base_keep = _sanitize_prompts_impl([*base, *base_prompts_all])
            prefilter_total = len(merged)
            if prefilter_cfg.get("enabled"):
                merged = _prefilter_prompts_with_clip(
                    merged,
                    keep_prompts=base_keep,
                    cat_id=int(cid),
                    class_name=str(name),
                    eval_ids=eval_ids,
                    images=images,
                    gt_by_image_cat=gt_by_image_cat,
                    clip_model_name=clip_model_for_prefilter,
                    sample_size=int(prefilter_cfg.get("sample_size") or 0),
                    keep_ratio=float(prefilter_cfg.get("keep_ratio") or 0.0),
                    seed=int(payload.split_seed) + int(cid),
                    log_fn=_log,
                )
            prompt_prefilter_stats[int(cid)] = {
                "enabled": bool(prefilter_cfg.get("enabled")),
                "mode": str(prefilter_cfg.get("mode") or "balanced"),
                "total": int(prefilter_total),
                "kept": int(len(merged)),
                "requested": bool(prefilter_cfg.get("requested")),
                "disabled_reason": prefilter_cfg.get("disabled_reason"),
            }
            if prefilter_cfg.get("enabled"):
                _log(
                    f"[steps] Prompt prefilter summary for {name}: kept {len(merged)}/{prefilter_total} "
                    f"(mode={prefilter_cfg.get('mode')})"
                )
            prepared_prompts[cid] = merged or base
            _log(f"Prompt list for {name}: {', '.join(prepared_prompts[cid])}")

        try:
            _unload_qwen_runtime()
            _log("Qwen prompt expansion runtime unloaded to free memory before SAM3 mining")
        except Exception:
            pass
        job.progress = max(job.progress, 0.07)

        eval_set = set(eval_ids)
        class_entries: List[Dict[str, Any]] = []
        total_classes = len(selected_categories)
        _log(f"Preparing {total_classes} class(es)…")

        for class_idx, (cid, name) in enumerate(selected_categories, start=1):
            if _cancelled():
                break
            job.message = f"Preparing {name} ({class_idx}/{total_classes})"
            job.updated_at = time.time()

            eval_gt = 0
            for img_id, cat_map in gt_by_image_cat.items():
                bboxes = cat_map.get(cid)
                if not bboxes:
                    continue
                if img_id in eval_set:
                    eval_gt += len(bboxes)

            prompts_for_class = prepared_prompts.get(cid) or [name]
            head_target_index: Optional[int] = None
            if clip_head:
                head_target_index = _find_clip_head_target_index(classes_list, name)
                if head_target_index is None:
                    _log(f"CLIP head mapping: class '{name}' not found; skipping head filter for this class.")
                else:
                    mapped_label = (
                        str(classes_list[head_target_index]) if 0 <= head_target_index < len(classes_list) else "unknown"
                    )
                    _log(f"CLIP head mapping: class '{name}' -> idx {head_target_index} (head='{mapped_label}')")
            class_entries.append(
                {
                    "id": int(cid),
                    "name": name,
                    "eval_gt": eval_gt,
                    "text_prompts": prompts_for_class,
                    "clip_head_target_index": head_target_index,
                }
            )

            job.progress = max(job.progress, 0.07 + 0.03 * (class_idx / max(1, total_classes)))
            job.updated_at = time.time()

        eval_payload = payload
        summaries: Dict[int, Dict[str, Any]] = {}
        recipes_by_class: Dict[int, Dict[str, Any]] = {}
        if _cancelled():
            summaries = {}
        else:
            job.message = "Evaluating recipes on sample…"
            job.updated_at = time.time()
            job.progress = max(job.progress, 0.1)
            eval_log_every = 5 if len(eval_ids) <= 50 else 50

            def _on_eval_progress(done: int, total: int) -> None:
                if total <= 0:
                    return
                if done != total and (eval_log_every <= 0 or done % eval_log_every != 0):
                    return
                frac = max(0.0, min(1.0, float(done) / float(total)))
                job.progress = max(job.progress, 0.1 + 0.9 * frac)
                job.message = f"Evaluating recipes: {done}/{total} sample images"
                job.updated_at = time.time()

            eval_workers: Optional[List[_Sam3GreedyEvalWorker]] = None
            try:
                eval_workers = _build_sam3_greedy_eval_workers(eval_payload, log_fn=_log)
                total_classes = len(class_entries)
                for class_idx, entry in enumerate(class_entries, start=1):
                    if _cancelled():
                        break
                    cid = int(entry.get("id"))
                    name = str(entry.get("name") or f"class_{cid}")
                    prompts_for_class = entry.get("text_prompts") or [name]
                    head_target_index = entry.get("clip_head_target_index")

                    class_base = 0.1 + 0.9 * ((class_idx - 1) / max(1, total_classes))
                    class_span = 0.9 * (1.0 / max(1, total_classes))
                    seed_span = class_span * 0.45
                    tune_span = class_span * 0.55

                    def _mk_phase_cb(base: float, span: float, label: str) -> Callable[[int, int], None]:
                        def _cb(done: int, total: int) -> None:
                            if total <= 0:
                                return
                            frac = max(0.0, min(1.0, float(done) / float(total)))
                            job.progress = max(job.progress, base + span * frac)
                            job.message = f"{label} {done}/{total} ({name})"
                            job.updated_at = time.time()

                        return _cb

                    job.message = f"[steps] Evaluating text candidates for {name} ({class_idx}/{total_classes})…"
                    job.updated_at = time.time()
                    seed_stats = _mine_seed_prompt_stats_image_first(
                        cat_id=cid,
                        prompts=prompts_for_class,
                        val_ids=eval_ids,
                        images=images,
                        gt_by_image_cat=gt_by_image_cat,
                        payload=eval_payload,
                        clip_head=clip_head,
                        clip_head_target_index=head_target_index,
                        clip_head_bg_indices=bg_indices,
                        prompt_bg_drop_cfg=prompt_bg_drop_cfg,
                        workers=eval_workers,
                        log_every=eval_log_every,
                        log_fn=_log,
                        cancel_event=job.cancel_event,
                        progress_callback=_mk_phase_cb(class_base, seed_span, "[steps] Candidate eval"),
                    )
                    prompt_bg_drop_summary: Optional[Dict[str, Any]] = None
                    if isinstance(prompt_bg_drop_cfg, dict) and (prompt_bg_drop_cfg.get("enabled") or prompt_bg_drop_cfg.get("requested")):
                        dropped = sum(1 for s in (seed_stats or []) if isinstance(s, dict) and s.get("bg_drop"))
                        prompt_bg_drop_summary = {
                            "enabled": bool(prompt_bg_drop_cfg.get("enabled")),
                            "mode": str(prompt_bg_drop_cfg.get("mode") or "balanced"),
                            "min_checked": int(prompt_bg_drop_cfg.get("min_checked") or 0),
                            "drop_rate": float(prompt_bg_drop_cfg.get("drop_rate") or 0.0),
                            "total": int(len(seed_stats or [])),
                            "dropped": int(dropped),
                            "requested": bool(prompt_bg_drop_cfg.get("requested")),
                            "disabled_reason": prompt_bg_drop_cfg.get("disabled_reason"),
                        }
                    target_prec = (
                        float(getattr(eval_payload, "clip_head_target_precision", 0.0) or 0.0)
                        if (clip_head is not None and head_target_index is not None)
                        else None
                    )
                    early_stop_cfg = _resolve_steps_early_stop_config(eval_payload, target_precision=target_prec)
                    if early_stop_cfg.get("enabled"):
                        _log(
                            "Early-stop enabled: "
                            f"mode={early_stop_cfg.get('mode')} min_steps={early_stop_cfg.get('min_steps')} "
                            f"window={early_stop_cfg.get('window')} min_increment={float(early_stop_cfg.get('min_increment') or 0.0):.3f} "
                            f"precision_margin={float(early_stop_cfg.get('precision_margin') or 0.0):.2f}"
                        )
                    selected, early_stop_info = _select_steps_from_seed_prompt_stats(
                        seed_stats,
                        max_steps=int(getattr(eval_payload, "steps_max_steps_per_recipe", 6) or 6),
                        target_precision=target_prec,
                        early_stop=early_stop_cfg,
                        log_fn=_log,
                    )
                    if early_stop_info.get("enabled"):
                        _log(
                            f"[steps] Early-stop summary for {name}: "
                            f"selected_steps={early_stop_info.get('selected_steps')}/{early_stop_info.get('max_steps')} "
                            f"mode={early_stop_info.get('mode')} reason={early_stop_info.get('reason')}"
                        )
                    refine_info: Optional[Dict[str, Any]] = None
                    if bool(getattr(eval_payload, "steps_refine_prompt_subset", False)):
                        selected, refine_info = _refine_steps_prompt_subset_seed_stage(
                            seed_stats,
                            selected,
                            max_steps=int(getattr(eval_payload, "steps_max_steps_per_recipe", 6) or 6),
                            target_precision=target_prec,
                            max_iters=int(getattr(eval_payload, "steps_refine_prompt_subset_max_iters", 6) or 6),
                            top_k=int(getattr(eval_payload, "steps_refine_prompt_subset_top_k", 6) or 6),
                            base_seed_threshold=float(eval_payload.seed_threshold),
                            log_fn=_log,
                        )
                    if isinstance(early_stop_info, dict):
                        early_stop_info["selected_steps_final"] = int(len(selected))
                    selected_prompts, step_list = _build_steps_recipe_step_list_from_selected_stats(
                        selected,
                        prompts_fallback=prompts_for_class,
                        payload=eval_payload,
                    )

                    tier1_info: Optional[Dict[str, Any]] = None
                    tier2_info: Optional[Dict[str, Any]] = None
                    global_info: Optional[Dict[str, Any]] = None
                    summary: Dict[str, Any]
                    if clip_head is not None and head_target_index is not None:
                        global_enabled = bool(getattr(eval_payload, "steps_optimize_global", False))
                        if global_enabled:
                            job.message = f"[steps] Global optimization for {name} ({class_idx}/{total_classes})…"
                            job.updated_at = time.time()
                            step_list, global_info = _tune_steps_global_optimizer_image_first(
                                cat_id=cid,
                                steps=step_list,
                                seed_stats=seed_stats,
                                val_ids=eval_ids,
                                images=images,
                                gt_by_image_cat=gt_by_image_cat,
                                payload=eval_payload,
                                clip_head=clip_head,
                                clip_head_target_index=int(head_target_index),
                                workers=eval_workers,
                                log_fn=_log,
                                cancel_event=job.cancel_event,
                            )
                        else:
                            if bool(getattr(eval_payload, "steps_optimize_tier1", False)):
                                job.message = f"[steps] Tier-1 grid search for {name} ({class_idx}/{total_classes})…"
                                job.updated_at = time.time()
                                step_list, tier1_info = _tune_steps_tier1_knobs_image_first(
                                    cat_id=cid,
                                    steps=step_list,
                                    val_ids=eval_ids,
                                    images=images,
                                    gt_by_image_cat=gt_by_image_cat,
                                    payload=eval_payload,
                                    clip_head=clip_head,
                                    clip_head_target_index=int(head_target_index),
                                    workers=eval_workers,
                                    log_fn=_log,
                                    cancel_event=job.cancel_event,
                                )
                            if bool(getattr(eval_payload, "steps_optimize_tier2", False)):
                                job.message = f"[steps] Tier-2 tuning for {name} ({class_idx}/{total_classes})…"
                                job.updated_at = time.time()
                                step_list, tier2_info = _tune_steps_tier2_knobs_image_first(
                                    cat_id=cid,
                                    steps=step_list,
                                    val_ids=eval_ids,
                                    images=images,
                                    gt_by_image_cat=gt_by_image_cat,
                                    payload=eval_payload,
                                    clip_head=clip_head,
                                    clip_head_target_index=int(head_target_index),
                                    workers=eval_workers,
                                    log_fn=_log,
                                    cancel_event=job.cancel_event,
                                )
                        job.message = f"[steps] Tuning CLIP head for {name} ({class_idx}/{total_classes})…"
                        job.updated_at = time.time()
                        summary = _tune_clip_head_for_selected_steps_image_first(
                            cat_id=cid,
                            class_name=name,
                            steps=step_list,
                            val_ids=eval_ids,
                            images=images,
                            gt_by_image_cat=gt_by_image_cat,
                            payload=eval_payload,
                            clip_head=clip_head,
                            clip_head_target_index=int(head_target_index),
                            workers=eval_workers,
                            log_every=eval_log_every,
                            log_fn=_log,
                            cancel_event=job.cancel_event,
                            progress_callback=_mk_phase_cb(class_base + seed_span, tune_span, "[steps] Final tune"),
                            export_hard_negatives=True,
                        )
                        if tier1_info and isinstance(summary, dict):
                            summary["tier1_tuning"] = tier1_info
                        if tier2_info and isinstance(summary, dict):
                            summary["tier2_tuning"] = tier2_info
                        if global_info and isinstance(summary, dict):
                            summary["global_optimizer"] = global_info
                        if refine_info and isinstance(summary, dict):
                            summary["prompt_subset_refinement"] = refine_info
                        if isinstance(summary, dict):
                            prefilter_summary = prompt_prefilter_stats.get(int(cid))
                            if isinstance(prefilter_summary, dict):
                                summary["prompt_prefilter"] = prefilter_summary
                            if isinstance(prompt_bg_drop_summary, dict):
                                summary["prompt_bg_drop"] = prompt_bg_drop_summary
                            if isinstance(early_stop_info, dict):
                                summary["early_stop"] = early_stop_info
                    else:
                        total_gt = int(entry.get("eval_gt") or 0)
                        summary = {
                            "gts": total_gt,
                            "matches": 0,
                            "fps": 0,
                            "duplicates": 0,
                            "preds": 0,
                            "precision": 0.0,
                            "recall": 0.0,
                            "coverage_rate": 0.0,
                            "det_rate": 0.0,
                        }
                        if refine_info:
                            summary["prompt_subset_refinement"] = refine_info
                        prefilter_summary = prompt_prefilter_stats.get(int(cid))
                        if isinstance(prefilter_summary, dict):
                            summary["prompt_prefilter"] = prefilter_summary
                        if isinstance(prompt_bg_drop_summary, dict):
                            summary["prompt_bg_drop"] = prompt_bg_drop_summary
                        if isinstance(early_stop_info, dict):
                            summary["early_stop"] = early_stop_info

                    summaries[cid] = summary
                    tuned_min_prob = None
                    tuned_margin = None
                    tuned_bg_margin = None
                    if isinstance(summary, dict):
                        tuned_min_prob = summary.get("clip_head_min_prob")
                        tuned_margin = summary.get("clip_head_margin")
                        tuned_bg_margin = summary.get("clip_head_background_margin")
                    try:
                        final_min_prob = float(tuned_min_prob) if tuned_min_prob is not None else float(payload.clip_head_min_prob)
                    except Exception:
                        final_min_prob = float(payload.clip_head_min_prob)
                    try:
                        final_margin = float(tuned_margin) if tuned_margin is not None else float(payload.clip_head_margin)
                    except Exception:
                        final_margin = float(payload.clip_head_margin)
                    try:
                        final_bg_margin = float(tuned_bg_margin) if tuned_bg_margin is not None else float(payload.clip_head_background_margin)
                    except Exception:
                        final_bg_margin = float(payload.clip_head_background_margin)
                    if step_list:
                        for step in step_list:
                            if not isinstance(step, dict):
                                continue
                            if "clip_seed" not in step:
                                step["clip_seed"] = {"min_prob": 0.0, "margin": 0.0}
                            if "clip_final" not in step:
                                step["clip_final"] = {"min_prob": float(final_min_prob), "margin": float(final_margin)}
                    tuned_expand = float(eval_payload.expand_threshold)
                    tuned_max_seeds = int(getattr(eval_payload, "steps_max_visual_seeds_per_step", 5) or 0)
                    tuned_seed_iou = float(eval_payload.seed_dedupe_iou)
                    tuned_out_iou = float(eval_payload.dedupe_iou)
                    tuned_max_results = int(eval_payload.max_results)
                    if step_list and isinstance(step_list[0], dict):
                        try:
                            if step_list[0].get("expand_threshold") is not None:
                                tuned_expand = float(step_list[0].get("expand_threshold"))
                        except Exception:
                            tuned_expand = float(eval_payload.expand_threshold)
                        try:
                            if step_list[0].get("max_visual_seeds") is not None:
                                tuned_max_seeds = int(step_list[0].get("max_visual_seeds"))
                        except Exception:
                            tuned_max_seeds = int(getattr(eval_payload, "steps_max_visual_seeds_per_step", 5) or 0)
                        try:
                            if step_list[0].get("seed_dedupe_iou") is not None:
                                tuned_seed_iou = float(step_list[0].get("seed_dedupe_iou"))
                        except Exception:
                            tuned_seed_iou = float(eval_payload.seed_dedupe_iou)
                        try:
                            if step_list[0].get("dedupe_iou") is not None:
                                tuned_out_iou = float(step_list[0].get("dedupe_iou"))
                        except Exception:
                            tuned_out_iou = float(eval_payload.dedupe_iou)
                        try:
                            if step_list[0].get("max_results") is not None:
                                tuned_max_results = int(step_list[0].get("max_results"))
                        except Exception:
                            tuned_max_results = int(eval_payload.max_results)
                    optimizer = {
                        "algorithm": "sam3_steps_v2",
                        "version": 1,
                        "created_at": float(time.time()),
                        "sample_seed": int(eval_payload.split_seed),
                        "sample_size": int(eval_payload.eval_image_count),
                        "sample_hash": str(sample_hash),
                        "sample_images": int(len(eval_ids)),
                        "target_precision": float(getattr(eval_payload, "clip_head_target_precision", 0.0) or 0.0),
                        "max_steps_per_recipe": int(getattr(eval_payload, "steps_max_steps_per_recipe", 6) or 6),
                        "seed_threshold": {
                            "base": float(eval_payload.seed_threshold),
                            "seed_eval_threshold": float(_compute_steps_seed_eval_threshold(eval_payload)),
                            "seed_eval_max_results": int(_compute_steps_seed_eval_max_results(eval_payload)),
                            "strategy": "curve_candidates",
                            "max_candidates_per_prompt": 6,
                        },
                        "global_optimizer": global_info
                        if isinstance(global_info, dict)
                        else {
                            "enabled": False,
                            "requested": bool(getattr(eval_payload, "steps_optimize_global", False)),
                            "eval_caps": list(getattr(eval_payload, "steps_optimize_global_eval_caps", []) or []),
                            "max_trials": int(getattr(eval_payload, "steps_optimize_global_max_trials", 0) or 0),
                            "keep_ratio": float(getattr(eval_payload, "steps_optimize_global_keep_ratio", 0.0) or 0.0),
                            "rounds": int(getattr(eval_payload, "steps_optimize_global_rounds", 0) or 0),
                            "mutations_per_round": int(getattr(eval_payload, "steps_optimize_global_mutations_per_round", 0) or 0),
                            "enable_ordering": bool(getattr(eval_payload, "steps_optimize_global_enable_ordering", False)),
                            "enable_max_results": bool(getattr(eval_payload, "steps_optimize_global_enable_max_results", False)),
                        },
                        "tier1": tier1_info
                        if isinstance(tier1_info, dict)
                        else {
                            "enabled": False,
                            "requested": bool(getattr(eval_payload, "steps_optimize_tier1", False)),
                            "eval_cap": int(getattr(eval_payload, "steps_optimize_tier1_eval_cap", 0) or 0),
                            "max_trials": int(getattr(eval_payload, "steps_optimize_tier1_max_trials", 0) or 0),
                        },
                        "tier2": tier2_info
                        if isinstance(tier2_info, dict)
                        else {
                            "enabled": False,
                            "requested": bool(getattr(eval_payload, "steps_optimize_tier2", False)),
                            "eval_cap": int(getattr(eval_payload, "steps_optimize_tier2_eval_cap", 0) or 0),
                            "max_trials": int(getattr(eval_payload, "steps_optimize_tier2_max_trials", 0) or 0),
                        },
                        "prompt_subset_refinement": refine_info
                        if isinstance(refine_info, dict)
                        else {
                            "enabled": False,
                            "requested": bool(getattr(eval_payload, "steps_refine_prompt_subset", False)),
                            "max_iters": int(getattr(eval_payload, "steps_refine_prompt_subset_max_iters", 0) or 0),
                            "top_k": int(getattr(eval_payload, "steps_refine_prompt_subset_top_k", 0) or 0),
                        },
                        "early_stop": {
                            **(early_stop_cfg or {}),
                            "requested": bool(getattr(eval_payload, "steps_early_stop", False)),
                        },
                        "prompt_prefilter": {
                            **(prefilter_cfg or {}),
                            "requested": bool(
                                prefilter_cfg.get("requested")
                                if isinstance(prefilter_cfg, dict)
                                else getattr(eval_payload, "steps_prompt_prefilter", False)
                            ),
                            "keep_base_prompts": True,
                        },
                    }
                    recipes_by_class[cid] = {
                        "schema_version": 2,
                        "mode": "sam3_steps",
                        "optimizer": optimizer,
                        "text_prompts": list(prompts_for_class),
                        "steps": step_list,
                        "params": {
                            "use_clip_fp_guard": False,
                            "use_negative_exemplars": False,
                            "negative_strength": 0.0,
                            "similarity_score": 0.0,
                            "seed_threshold": float(eval_payload.seed_threshold),
                            "expand_threshold": float(tuned_expand),
                            "max_visual_seeds": int(tuned_max_seeds),
                            "seed_dedupe_iou": float(tuned_seed_iou),
                            "dedupe_iou": float(tuned_out_iou),
                            "mask_threshold": float(eval_payload.mask_threshold),
                            "min_size": int(eval_payload.min_size),
                            "simplify_epsilon": float(eval_payload.simplify_epsilon),
                            "max_results": int(tuned_max_results),
                            "clip_head_background_guard": bool(getattr(eval_payload, "clip_head_background_guard", False)),
                            "clip_head_background_margin": float(final_bg_margin),
                            "clip_head_background_apply": str(getattr(eval_payload, "clip_head_background_apply", "final") or "final"),
                            "clip_head_background_penalty": float(getattr(eval_payload, "clip_head_background_penalty", 0.0) or 0.0),
                        },
                        "summary": {
                            **(summary or {}),
                            "seed_prompt_stats": [
                                {
                                    "prompt": s.get("prompt"),
                                    "matches": int(s.get("matches") or 0),
                                    "fps": int(s.get("fps") or 0),
                                    "precision": float(s.get("precision") or 0.0),
                                    "selected_seed_threshold": float(
                                        s.get("selected_seed_threshold")
                                        if s.get("selected_seed_threshold") is not None
                                        else (s.get("seed_threshold_recommended") if s.get("seed_threshold_recommended") is not None else eval_payload.seed_threshold)
                                    ),
                                    "selected_seed_threshold_point": s.get("selected_seed_threshold_point"),
                                    "seed_threshold_base": float(s.get("seed_threshold_base") or eval_payload.seed_threshold),
                                    "seed_threshold_recommended": float(
                                        s.get("seed_threshold_recommended") if s.get("seed_threshold_recommended") is not None else eval_payload.seed_threshold
                                    ),
                                    "seed_threshold_base_point": s.get("seed_threshold_base_point"),
                                    "seed_threshold_recommended_point": s.get("seed_threshold_recommended_point"),
                                    "seed_threshold_curve": s.get("seed_threshold_curve") or [],
                                }
                                for s in selected
                            ],
                        },
                    }
                job.progress = max(job.progress, 0.98)
            finally:
                if eval_workers:
                    for w in eval_workers:
                        try:
                            w.close()
                        except Exception:
                            continue

        results: List[Dict[str, Any]] = []
        for entry in class_entries:
            cid = int(entry.get("id"))
            name = str(entry.get("name") or f"class_{cid}")
            summary = summaries.get(cid)
            if summary is None:
                summary = {
                    "gts": int(entry.get("eval_gt") or 0),
                    "matches": 0,
                    "fps": 0,
                    "duplicates": 0,
                    "preds": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "coverage_rate": 0.0,
                    "det_rate": 0.0,
                }
            recipe = recipes_by_class.get(cid) or {}
            if clip_head is not None and isinstance(recipe, dict):
                try:
                    tuned_min_prob = summary.get("clip_head_min_prob")
                    tuned_margin = summary.get("clip_head_margin")
                    if tuned_min_prob is not None or tuned_margin is not None:
                        recipe["_clip_head_classifier_path"] = str(clip_head_path)
                    recipe["clip_head"] = {
                        "artifact": "clip_head/head.npz",
                        "clip_model": clip_head.get("clip_model"),
                        "proba_mode": clip_head.get("proba_mode"),
                        "classes": clip_head.get("classes") if isinstance(clip_head.get("classes"), list) else [],
                        "min_prob": float(tuned_min_prob) if tuned_min_prob is not None else float(payload.clip_head_min_prob),
                        "margin": float(tuned_margin) if tuned_margin is not None else float(payload.clip_head_margin),
                        "background_margin": float(tuned_bg_margin)
                        if tuned_bg_margin is not None
                        else float(getattr(payload, "clip_head_background_margin", 0.0) or 0.0),
                        "auto_tuned": bool(getattr(payload, "clip_head_auto_tune", True)),
                        "target_precision": float(getattr(payload, "clip_head_target_precision", 0.9)),
                    }
                except Exception:
                    pass

            results.append(
                {
                    "id": cid,
                    "name": name,
                    "eval_gt": int(entry.get("eval_gt") or 0),
                    "recipe": recipe,
                }
            )

        job.result = {
            "dataset_id": payload.dataset_id,
            "sample": {
                "count": len(eval_ids),
                "seed": payload.split_seed,
            },
            "compute_estimate": compute_estimate_info,
            "classes": results,
            "config": eval_payload.dict(),
            "note": "Agent mining completed (steps mode).",
        }
        if compute_estimate_info and total_units_per_class:
            runtime_sec = max(0.0, float(time.time() - start_ts))
            total_units = compute_estimate_info.get("total_units_all_classes") or compute_estimate_info.get("total_units_per_class")
            try:
                total_units_val = float(total_units) if total_units is not None else 0.0
            except Exception:
                total_units_val = 0.0
            sec_per_unit = runtime_sec / total_units_val if total_units_val > 0 else None
            compute_estimate_info["runtime_sec"] = runtime_sec
            compute_estimate_info["sec_per_unit"] = sec_per_unit
            _log(
                "Compute estimate calibration: "
                f"runtime_sec={runtime_sec:.1f} units={total_units_val:.0f} "
                + (f"sec_per_unit={sec_per_unit:.6f}" if sec_per_unit is not None else "sec_per_unit=n/a")
            )
        if _cancelled():
            job.status = "cancelled"
            job.message = "Cancelled"
        else:
            job.status = "completed"
            job.message = "Done"
            job.progress = 1.0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent mining job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "Failed"
    finally:
        job.updated_at = time.time()
def _start_prompt_helper_job(payload: PromptHelperRequest) -> PromptHelperJob:
    job_id = f"ph_{uuid.uuid4().hex[:8]}"
    job = PromptHelperJob(job_id=job_id)
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    thread = threading.Thread(target=_run_prompt_helper_job, args=(job, payload), daemon=True)
    thread.start()
    return job


def _start_agent_mining_job(payload: AgentMiningRequest) -> AgentMiningJob:
    clip_head_path = _resolve_agent_clip_classifier_path_impl(
        payload.clip_head_classifier_path,
        allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
        allowed_exts=CLASSIFIER_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root_impl,
        http_exception_cls=HTTPException,
    )
    if clip_head_path is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_clip_head_required")
    # Validate early so we fail fast (no background job created).
    _load_clip_head_from_classifier_impl(
        clip_head_path,
        joblib_load_fn=joblib.load,
        http_exception_cls=HTTPException,
        clip_head_background_indices_fn=_clip_head_background_indices,
        resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
        infer_clip_model_fn=_infer_clip_model_from_embedding_dim_impl,
        active_clip_model_name=clip_model_name,
        default_clip_model=DEFAULT_CLIP_MODEL,
        logger=logger,
    )
    job_id = f"am_{uuid.uuid4().hex[:8]}"
    job = AgentMiningJob(job_id=job_id)
    with AGENT_MINING_JOBS_LOCK:
        AGENT_MINING_JOBS[job.job_id] = job
    thread = threading.Thread(target=_run_agent_mining_job, args=(job, payload), daemon=True)
    thread.start()
    return job


def _cancel_agent_mining_job(job_id: str) -> AgentMiningJob:
    with AGENT_MINING_JOBS_LOCK:
        job = AGENT_MINING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_mining_job_not_found")
    if job.status in {"completed", "failed", "cancelled"}:
        return job
    job.cancel_event.set()
    job.status = "cancelled"
    job.message = "Cancelled"
    job.updated_at = time.time()
    return job


def _start_prompt_helper_search_job(payload: PromptHelperSearchRequest) -> PromptHelperJob:
    job_id = f"phs_{uuid.uuid4().hex[:8]}"
    job = PromptHelperJob(job_id=job_id)
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    thread = threading.Thread(target=_run_prompt_helper_search_job, args=(job, payload), daemon=True)
    thread.start()
    return job


def _start_prompt_recipe_job(payload: PromptRecipeRequest) -> PromptHelperJob:
    job_id = f"phr_{uuid.uuid4().hex[:8]}"
    job = PromptHelperJob(job_id=job_id)
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    thread = threading.Thread(target=_run_prompt_recipe_job, args=(job, payload), daemon=True)
    thread.start()
    return job


def start_agent_mining_job(payload: AgentMiningRequest):
    job = _start_agent_mining_job(payload)
    return _serialize_agent_mining_job(job)


def list_agent_mining_jobs():
    _prune_job_registry(AGENT_MINING_JOBS, AGENT_MINING_JOBS_LOCK)
    with AGENT_MINING_JOBS_LOCK:
        jobs = list(AGENT_MINING_JOBS.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_serialize_agent_mining_job(j) for j in jobs]


def get_agent_mining_job(job_id: str):
    with AGENT_MINING_JOBS_LOCK:
        job = AGENT_MINING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_mining_job_not_found")
    return _serialize_agent_mining_job(job)


def cancel_agent_mining_job(job_id: str):
    job = _cancel_agent_mining_job(job_id)
    return _serialize_agent_mining_job(job)


def get_latest_agent_mining_result():
    with AGENT_MINING_JOBS_LOCK:
        jobs = [j for j in AGENT_MINING_JOBS.values() if j.status in {"running", "completed", "failed", "cancelled"}]
    if not jobs:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_mining_result_not_found")
    jobs.sort(key=lambda j: j.updated_at, reverse=True)
    return _serialize_agent_mining_job(jobs[0])


def agent_mining_cache_size():
    cache_root = AGENT_MINING_DET_CACHE_ROOT
    # Light touch: enforce TTL/size only when no active job to avoid surprises.
    _enforce_agent_mining_cache_limits(cache_root, allow_when_running=False)
    total = 0
    files = 0
    try:
        for p in cache_root.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
                    files += 1
            except Exception:
                continue
    except Exception:
        total = 0
    return {
        "bytes": total,
        "files": files,
        "max_bytes": AGENT_MINING_CACHE_MAX_BYTES,
        "ttl_hours": AGENT_MINING_CACHE_TTL_HOURS,
    }


def agent_mining_cache_purge():
    cache_root = AGENT_MINING_DET_CACHE_ROOT
    if _agent_cache_running_jobs():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="agent_cache_purge_blocked_active_jobs")
    if not cache_root.exists():
        return {"status": "ok", "deleted_bytes": 0, "deleted_files": 0}
    deleted = 0
    deleted_files = 0
    paths = sorted(cache_root.rglob("*"), key=lambda x: len(x.parts), reverse=True)
    for p in paths:
        try:
            if p.is_file():
                deleted += p.stat().st_size
                deleted_files += 1
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        except Exception:
            continue
    return {"status": "ok", "deleted_bytes": deleted, "deleted_files": deleted_files}


def agent_mining_apply_image(payload: AgentApplyImageRequest):
    variant = _default_variant(payload.sam_variant or "sam3")
    if variant != "sam3":
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_apply_requires_sam3")
    pil_img, _, token = resolve_image_payload(payload.image_base64, payload.image_token, variant)
    warnings: List[str] = []
    recipe_meta = payload.recipe or {}

    class_id_val = recipe_meta.get("class_id")
    class_name_val = recipe_meta.get("class_name")
    if payload.override_class_id is not None or payload.override_class_name:
        warnings.append("class_override_used")
        if payload.override_class_id is not None:
            class_id_val = payload.override_class_id
        if payload.override_class_name:
            class_name_val = payload.override_class_name
    class_id_int: Optional[int] = None
    if class_id_val is not None:
        try:
            class_id_int = int(class_id_val)
        except Exception:
            class_id_int = None

    # Reuse the existing apply implementation by staging the in-memory image to a temp file.
    staging_dir = Path(tempfile.mkdtemp(prefix="agent_apply_image_"))
    try:
        img_path = staging_dir / "image.png"
        try:
            pil_img.save(img_path, format="PNG")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_apply_image_encode_failed:{exc}") from exc
        dets = _apply_agent_recipe_to_image(
            payload.recipe,
            image={"path": str(img_path)},
            dataset_id="image_payload",
            images={},
            mask_threshold=payload.mask_threshold,
            min_size=payload.min_size,
            simplify_epsilon=payload.simplify_epsilon,
            max_results=payload.max_results,
            class_id=class_id_int,
            class_name=str(class_name_val) if class_name_val is not None else None,
            clip_head_min_prob_override=payload.clip_head_min_prob_override,
            clip_head_margin_override=payload.clip_head_margin_override,
            extra_clip_classifier_path=payload.extra_clip_classifier_path,
            extra_clip_min_prob=payload.extra_clip_min_prob,
            extra_clip_margin=payload.extra_clip_margin,
            warnings=warnings,
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return Sam3TextPromptResponse(detections=dets, warnings=warnings, image_token=token)


def agent_mining_apply_image_chain(payload: AgentApplyImageChainRequest):
    variant = _default_variant(payload.sam_variant or "sam3")
    if variant != "sam3":
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_apply_requires_sam3")
    pil_img, _, token = resolve_image_payload(payload.image_base64, payload.image_token, variant)
    warnings: List[str] = []

    enabled_steps = [s for s in payload.steps if s.enabled]
    if not enabled_steps:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="steps_required")

    staging_dir = Path(tempfile.mkdtemp(prefix="agent_apply_chain_"))
    try:
        img_path = staging_dir / "image.png"
        try:
            pil_img.save(img_path, format="PNG")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_apply_image_encode_failed:{exc}") from exc

        all_dets: List[QwenDetection] = []
        det_meta: Dict[int, Dict[str, Any]] = {}

        for idx, step in enumerate(enabled_steps):
            recipe_obj: Optional[Dict[str, Any]] = None
            if isinstance(step.recipe, dict) and step.recipe:
                recipe_obj = step.recipe
            elif step.recipe_id:
                recipe_obj = _load_agent_recipe_json_only_impl(
                    step.recipe_id,
                    recipes_root=AGENT_MINING_RECIPES_ROOT,
                    path_is_within_root_fn=_path_is_within_root_impl,
                )
            if not isinstance(recipe_obj, dict) or not recipe_obj:
                continue

            class_id_val = recipe_obj.get("class_id")
            class_name_val = recipe_obj.get("class_name")
            if step.override_class_id is not None or step.override_class_name:
                warnings.append("class_override_used")
                if step.override_class_id is not None:
                    class_id_val = step.override_class_id
                if step.override_class_name:
                    class_name_val = step.override_class_name

            class_id_int: Optional[int] = None
            if class_id_val is not None:
                try:
                    class_id_int = int(class_id_val)
                except Exception:
                    class_id_int = None
            class_name_str = str(class_name_val) if class_name_val is not None else None

            dets = _apply_agent_recipe_to_image(
                recipe_obj,
                image={"path": str(img_path)},
                dataset_id="image_payload",
                images={},
                mask_threshold=payload.mask_threshold,
                min_size=payload.min_size,
                simplify_epsilon=payload.simplify_epsilon,
                max_results=payload.max_results,
                class_id=class_id_int,
                class_name=class_name_str,
                clip_head_min_prob_override=step.clip_head_min_prob_override,
                clip_head_margin_override=step.clip_head_margin_override,
                extra_clip_classifier_path=step.extra_clip_classifier_path,
                extra_clip_min_prob=step.extra_clip_min_prob,
                extra_clip_margin=step.extra_clip_margin,
                warnings=warnings,
            )
            group_val = (step.dedupe_group or "").strip() or "default"
            cross_val = bool(step.participate_cross_class_dedupe)
            for det in dets:
                det_meta[id(det)] = {
                    "step_index": idx,
                    "dedupe_group": group_val,
                    "cross_class": cross_val,
                }
            all_dets.extend(dets)

        if not all_dets:
            return Sam3TextPromptResponse(detections=[], warnings=warnings, image_token=token)

        original_scores: Dict[int, Optional[float]] = {}
        confidence_mode = payload.dedupe.confidence
        if confidence_mode in {"clip_head_prob", "clip_head_margin"}:
            head_recipe_id = (payload.dedupe.clip_head_recipe_id or "").strip() or None
            if head_recipe_id is None:
                for step in enabled_steps:
                    rid = None
                    if step.recipe_id:
                        rid = step.recipe_id
                    elif isinstance(step.recipe, dict):
                        rid = step.recipe.get("id")
                    if not rid:
                        continue
                    candidate = (AGENT_MINING_RECIPES_ROOT / str(rid) / "clip_head" / "head.npz").resolve()
                    if _path_is_within_root_impl(candidate, AGENT_MINING_RECIPES_ROOT.resolve()) and candidate.exists():
                        head_recipe_id = str(rid)
                        break

            clip_head: Optional[Dict[str, Any]] = None
            if head_recipe_id:
                try:
                    head_recipe = _load_agent_recipe_json_only_impl(
                        head_recipe_id,
                        recipes_root=AGENT_MINING_RECIPES_ROOT,
                        path_is_within_root_fn=_path_is_within_root_impl,
                    )
                except HTTPException:
                    head_recipe = None
                fallback_meta: Optional[Dict[str, Any]] = None
                if isinstance(head_recipe, dict):
                    recipe_block = head_recipe.get("recipe")
                    if isinstance(recipe_block, dict) and isinstance(recipe_block.get("clip_head"), dict):
                        fallback_meta = recipe_block.get("clip_head")
                    elif isinstance(head_recipe.get("clip_head"), dict):
                        fallback_meta = head_recipe.get("clip_head")
                recipe_root = (AGENT_MINING_RECIPES_ROOT / str(head_recipe_id)).resolve()
                if _path_is_within_root_impl(recipe_root, AGENT_MINING_RECIPES_ROOT.resolve()):
                    clip_head = _load_clip_head_artifacts(recipe_dir=recipe_root, fallback_meta=fallback_meta)

            clip_scores = (
                _score_detections_with_clip_head(
                    all_dets,
                    pil_img=pil_img,
                    clip_head=clip_head,
                    score_mode=confidence_mode,  # type: ignore[arg-type]
                )
                if clip_head
                else None
            )
            if clip_scores is None:
                warnings.append("clip_head_unavailable")
            else:
                if all_dets and not clip_scores:
                    warnings.append("clip_head_no_scores")
                for det in all_dets:
                    det_id = id(det)
                    original_scores[det_id] = det.score
                    if det_id in clip_scores:
                        det.score = float(clip_scores[det_id])

        # Always dedupe within each output class first.
        per_class_iou = float(payload.dedupe.per_class_iou)
        by_class: Dict[str, List[QwenDetection]] = {}
        for det in all_dets:
            class_name_key = str(det.class_name or "").strip()
            if class_name_key:
                key = f"name:{class_name_key}"
            elif det.class_id is not None:
                key = f"id:{int(det.class_id)}"
            else:
                key = "unknown"
            by_class.setdefault(key, []).append(det)
        deduped: List[QwenDetection] = []
        for group in by_class.values():
            deduped.extend(
                _dedupe_qwen_detections_iou(group, img_w=pil_img.width, img_h=pil_img.height, iou_thresh=per_class_iou)
            )
        final = deduped

        # Optional cross-class dedupe across steps (grouped or global).
        if payload.dedupe.cross_class_enabled:
            cross_iou = float(payload.dedupe.cross_class_iou)
            scope = payload.dedupe.cross_class_scope
            group_map: Dict[str, List[QwenDetection]] = {}
            for det in final:
                meta = det_meta.get(id(det)) or {}
                if not meta.get("cross_class", True):
                    continue
                group_key = "global" if scope == "global" else str(meta.get("dedupe_group") or "default")
                group_map.setdefault(group_key, []).append(det)

            kept_ids: set[int] = set()
            for group in group_map.values():
                kept = _dedupe_qwen_detections_iou(group, img_w=pil_img.width, img_h=pil_img.height, iou_thresh=cross_iou)
                kept_ids.update(id(d) for d in kept)

            if group_map:
                final = [
                    det
                    for det in final
                    if (id(det) in kept_ids) or not (det_meta.get(id(det)) or {}).get("cross_class", True)
                ]

        if original_scores:
            for det in all_dets:
                det_id = id(det)
                if det_id in original_scores:
                    det.score = original_scores[det_id]

        return Sam3TextPromptResponse(detections=final, warnings=warnings, image_token=token)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def agent_mining_save_recipe(payload: AgentRecipeExportRequest):
    recipe = _persist_agent_recipe(
        payload.dataset_id,
        payload.class_id,
        payload.class_name,
        payload.label,
        payload.recipe,
    )
    return recipe


def agent_mining_list_recipes(dataset_id: Optional[str] = None):
    return _list_agent_recipes(dataset_id=dataset_id)


def agent_mining_export_recipe(recipe_id: str):
    recipe = _load_agent_recipe_impl(
        recipe_id,
        recipes_root=AGENT_MINING_RECIPES_ROOT,
        path_is_within_root_fn=_path_is_within_root_impl,
    )
    zip_path = _ensure_recipe_zip(recipe)
    filename = f"{recipe_id}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    try:
        stream = zip_path.open("rb")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_export_failed:{exc}") from exc
    return StreamingResponse(stream, media_type="application/zip", headers=headers)


async def agent_mining_import_recipe(file: UploadFile = File(...)):
    staging_dir = Path(tempfile.mkdtemp(prefix="agent_recipe_import_", dir=str(AGENT_MINING_RECIPES_ROOT)))
    zip_path = staging_dir / "payload.zip"
    try:
        await _write_upload_file(
            file,
            zip_path,
            max_bytes=AGENT_RECIPE_MAX_BYTES,
            quota_root=staging_dir,
            quota_limit=AGENT_RECIPE_MAX_BYTES,
            allow_overwrite=True,
        )
        payload_bytes = zip_path.read_bytes()
        _, persisted = _import_agent_recipe_zip_bytes(payload_bytes)
        return persisted
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_recipe_import_failed:{exc}") from exc
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def agent_mining_delete_recipe(recipe_id: str):
    _delete_agent_recipe(recipe_id)
    return {"id": recipe_id, "deleted": True}


def agent_mining_save_cascade(payload: AgentCascadeSaveRequest):
    cascade_payload = {
        "steps": [s.dict() for s in payload.steps],
        "dedupe": payload.dedupe.dict(),
    }
    return _persist_agent_cascade(payload.label, cascade_payload)


def agent_mining_list_cascades():
    return _list_agent_cascades()


def agent_mining_delete_cascade(cascade_id: str):
    _delete_agent_cascade(cascade_id)
    return {"id": cascade_id, "deleted": True}


def agent_mining_export_cascade(cascade_id: str):
    cascade = _load_agent_cascade_impl(
        cascade_id,
        cascades_root=AGENT_MINING_CASCADES_ROOT,
        path_is_within_root_fn=_path_is_within_root_impl,
    )
    zip_path = _ensure_cascade_zip(cascade)
    filename = f"{cascade_id}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    try:
        stream = zip_path.open("rb")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_cascade_export_failed:{exc}") from exc
    return StreamingResponse(stream, media_type="application/zip", headers=headers)


async def agent_mining_import_cascade(file: UploadFile = File(...)):
    staging_dir = Path(tempfile.mkdtemp(prefix="agent_cascade_import_", dir=str(AGENT_MINING_CASCADES_ROOT)))
    zip_path = staging_dir / "payload.zip"
    try:
        await _write_upload_file(
            file,
            zip_path,
            max_bytes=AGENT_CASCADE_MAX_BYTES,
            quota_root=staging_dir,
            quota_limit=AGENT_CASCADE_MAX_BYTES,
            allow_overwrite=True,
        )
        payload_bytes = zip_path.read_bytes()
        return _import_agent_cascade_zip_bytes(payload_bytes)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_cascade_import_failed:{exc}") from exc
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


app.include_router(
    build_agent_mining_router(
        start_job_fn=start_agent_mining_job,
        list_jobs_fn=list_agent_mining_jobs,
        get_job_fn=get_agent_mining_job,
        cancel_job_fn=cancel_agent_mining_job,
        latest_result_fn=get_latest_agent_mining_result,
        cache_size_fn=agent_mining_cache_size,
        cache_purge_fn=agent_mining_cache_purge,
        apply_image_fn=agent_mining_apply_image,
        apply_chain_fn=agent_mining_apply_image_chain,
        save_recipe_fn=agent_mining_save_recipe,
        list_recipes_fn=agent_mining_list_recipes,
        get_recipe_fn=lambda recipe_id: _load_agent_recipe_impl(
            recipe_id,
            recipes_root=AGENT_MINING_RECIPES_ROOT,
            path_is_within_root_fn=_path_is_within_root_impl,
        ),
        export_recipe_fn=agent_mining_export_recipe,
        import_recipe_fn=agent_mining_import_recipe,
        delete_recipe_fn=agent_mining_delete_recipe,
        save_cascade_fn=agent_mining_save_cascade,
        list_cascades_fn=agent_mining_list_cascades,
        get_cascade_fn=lambda cascade_id: _load_agent_cascade_impl(
            cascade_id,
            cascades_root=AGENT_MINING_CASCADES_ROOT,
            path_is_within_root_fn=_path_is_within_root_impl,
        ),
        delete_cascade_fn=agent_mining_delete_cascade,
        export_cascade_fn=agent_mining_export_cascade,
        import_cascade_fn=agent_mining_import_cascade,
        job_request_cls=AgentMiningRequest,
        apply_image_request_cls=AgentApplyImageRequest,
        apply_chain_request_cls=AgentApplyImageChainRequest,
        recipe_request_cls=AgentRecipeExportRequest,
        cascade_request_cls=AgentCascadeSaveRequest,
        sam3_response_cls=Sam3TextPromptResponse,
    )
)


def prompt_helper_suggest(payload: PromptHelperSuggestRequest):
    return _suggest_prompts_for_dataset(payload)


def prompt_helper_expand(payload: PromptRecipeExpandRequest):
    dataset_root = _resolve_sam3_or_qwen_dataset_impl(
        payload.dataset_id,
        list_all_datasets_fn=_list_all_datasets,
        resolve_dataset_legacy_fn=lambda dataset_id: _resolve_dataset_legacy_impl(
            dataset_id,
            qwen_root=QWEN_DATASET_ROOT,
            sam3_root=SAM3_DATASET_ROOT,
            registry_root=DATASET_REGISTRY_ROOT,
            http_exception_cls=HTTPException,
        ),
    )
    coco, _, _ = _load_coco_index_impl(dataset_root)
    categories = coco.get("categories") or []
    cat_entry = next((c for c in categories if int(c.get("id", categories.index(c))) == payload.class_id), None)
    if not cat_entry:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="recipe_class_not_found")
    class_name = str(cat_entry.get("name", f"class_{payload.class_id}"))
    base_prompts = [p.strip() for p in payload.base_prompts if isinstance(p, str) and p.strip()]
    new_prompts = _expand_prompts_with_prompt_llm(
        class_name,
        base_prompts,
        payload.max_new,
        max_new_tokens=payload.max_new_tokens if hasattr(payload, "max_new_tokens") else 128,
    )
    combined: List[str] = []
    seen = set()
    for prompt in [*base_prompts, *new_prompts]:
        low = prompt.lower()
        if low in seen:
            continue
        seen.add(low)
        combined.append(prompt)
    return {
        "class_id": payload.class_id,
        "class_name": class_name,
        "base_prompts": base_prompts,
        "new_prompts": new_prompts,
        "combined": combined,
    }


def start_prompt_helper_job(payload: PromptHelperRequest):
    job = _start_prompt_helper_job(payload)
    return _serialize_prompt_helper_job_impl(job)


def start_prompt_helper_search(payload: PromptHelperSearchRequest):
    job = _start_prompt_helper_search_job(payload)
    return _serialize_prompt_helper_job_impl(job)


def start_prompt_helper_recipe(payload: PromptRecipeRequest):
    job = _start_prompt_recipe_job(payload)
    return _serialize_prompt_helper_job_impl(job)

def create_prompt_helper_preset(
    dataset_id: str = Form(...),
    label: str = Form(""),
    prompts_json: str = Form(...),
):
    try:
        prompts_by_class = json.loads(prompts_json)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_prompts:{exc}") from exc
    if not isinstance(prompts_by_class, dict):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prompts_must_be_object")
    normalized: Dict[int, List[str]] = {}
    for key, vals in prompts_by_class.items():
        try:
            cid = int(key)
        except Exception:
            continue
        if not isinstance(vals, (list, tuple)):
            continue
        cleaned = [str(v).strip() for v in vals if isinstance(v, str) and str(v).strip()]
        if cleaned:
            normalized[cid] = cleaned
    if not normalized:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="no_prompts_provided")
    preset = _save_prompt_helper_preset_impl(
        label,
        dataset_id,
        normalized,
        presets_root=PROMPT_HELPER_PRESET_ROOT,
        path_is_within_root_fn=_path_is_within_root_impl,
    )
    return preset


def list_prompt_helper_jobs():
    _prune_job_registry(PROMPT_HELPER_JOBS, PROMPT_HELPER_JOBS_LOCK)
    with PROMPT_HELPER_JOBS_LOCK:
        jobs = list(PROMPT_HELPER_JOBS.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_serialize_prompt_helper_job_impl(j) for j in jobs]


def get_prompt_helper_job(job_id: str):
    with PROMPT_HELPER_JOBS_LOCK:
        job = PROMPT_HELPER_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prompt_helper_job_not_found")
    return _serialize_prompt_helper_job_impl(job)


app.include_router(
    build_sam3_prompt_helper_router(
        suggest_fn=prompt_helper_suggest,
        expand_fn=prompt_helper_expand,
        create_job_fn=start_prompt_helper_job,
        search_fn=start_prompt_helper_search,
        recipe_fn=start_prompt_helper_recipe,
        list_presets_fn=lambda: _list_prompt_helper_presets_impl(presets_root=PROMPT_HELPER_PRESET_ROOT),
        get_preset_fn=lambda preset_id: _load_prompt_helper_preset_impl(
            preset_id,
            presets_root=PROMPT_HELPER_PRESET_ROOT,
            path_is_within_root_fn=_path_is_within_root_impl,
        ),
        save_preset_fn=create_prompt_helper_preset,
        list_jobs_fn=list_prompt_helper_jobs,
        get_job_fn=get_prompt_helper_job,
        request_suggest_cls=PromptHelperSuggestRequest,
        request_expand_cls=PromptRecipeExpandRequest,
        request_job_cls=PromptHelperRequest,
        request_search_cls=PromptHelperSearchRequest,
        request_recipe_cls=PromptRecipeRequest,
    )
)


def start_segmentation_build_job(request: SegmentationBuildRequest):
    job = _start_segmentation_build_job(request)
    return _serialize_seg_job_impl(job)


def list_segmentation_build_jobs():
    _prune_job_registry(SEGMENTATION_BUILD_JOBS, SEGMENTATION_BUILD_JOBS_LOCK)
    with SEGMENTATION_BUILD_JOBS_LOCK:
        jobs = list(SEGMENTATION_BUILD_JOBS.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_serialize_seg_job_impl(job) for job in jobs]


def get_segmentation_build_job(job_id: str):
    with SEGMENTATION_BUILD_JOBS_LOCK:
        job = SEGMENTATION_BUILD_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="segmentation_job_not_found")
    return _serialize_seg_job_impl(job)


app.include_router(
    build_segmentation_build_router(
        start_fn=start_segmentation_build_job,
        list_fn=list_segmentation_build_jobs,
        get_fn=get_segmentation_build_job,
        request_cls=SegmentationBuildRequest,
    )
)


## NOTE: RF‑DETR dataset helpers use *_impl directly to avoid wrapper drift.


def _rfdetr_ddp_worker(
    rank: int,
    world_size: int,
    variant_id: str,
    model_kwargs: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    aug_policy: Optional[Dict[str, Any]],
    dist_url: str,
) -> None:
    def _import_rfdetr() -> Dict[str, Any]:
        from rfdetr import (  # type: ignore
            RFDETRBase,
            RFDETRLarge,
            RFDETRNano,
            RFDETRSmall,
            RFDETRMedium,
            RFDETRSegPreview,
        )

        return {
            "rfdetr-nano": RFDETRNano,
            "rfdetr-small": RFDETRSmall,
            "rfdetr-medium": RFDETRMedium,
            "rfdetr-base": RFDETRBase,
            "rfdetr-large": RFDETRLarge,
            "rfdetr-seg-preview": RFDETRSegPreview,
        }

    _rfdetr_ddp_worker_impl(
        rank,
        world_size,
        variant_id,
        model_kwargs,
        train_kwargs,
        aug_policy,
        dist_url,
        os_module=os,
        torch_module=torch,
        import_rfdetr_fn=_import_rfdetr,
        normalize_aug_fn=_rfdetr_normalize_aug_policy_impl,
        install_aug_fn=_rfdetr_install_augmentations_impl,
        restore_aug_fn=_rfdetr_restore_augmentations_impl,
    )


## NOTE: keep these helpers as direct impl calls at the call sites to avoid wrapper drift.


def _prepare_sam3_training_split(
    dataset_root: Path,
    meta: Dict[str, Any],
    job_id: str,
    *,
    random_split: bool,
    val_percent: float,
    split_seed: int,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    log_messages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not random_split:
        return meta
    coco_train_path = Path(meta.get("coco_train_json", ""))
    coco_val_path = Path(meta.get("coco_val_json", ""))
    if not coco_train_path.exists() or not coco_val_path.exists():
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_coco_split_missing")
    try:
        with coco_train_path.open("r", encoding="utf-8") as handle:
            coco_train = json.load(handle)
        with coco_val_path.open("r", encoding="utf-8") as handle:
            coco_val = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_coco_load_failed:{exc}") from exc
    categories = coco_train.get("categories") or coco_val.get("categories") or []
    if not categories and meta.get("classes"):
        categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(meta.get("classes", []))]
    if not categories:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_categories_missing")
    images: Dict[int, Dict[str, Any]] = {}
    ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for coco_blob in (coco_train, coco_val):
        for img in coco_blob.get("images", []):
            try:
                img_id = int(img["id"])
            except Exception:
                continue
            images[img_id] = {**img, "id": img_id, "file_name": str(img.get("file_name", ""))}
        for ann in coco_blob.get("annotations", []):
            try:
                img_id = int(ann["image_id"])
            except Exception:
                continue
            ann_by_image.setdefault(img_id, []).append(ann)
    image_ids = list(images.keys())
    if not image_ids:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_training_no_images")
    rnd = random.Random(split_seed)
    rnd.shuffle(image_ids)
    total = len(image_ids)
    vp = max(0.0, min(float(val_percent), 0.9))
    val_count = int(total * vp)
    if val_count <= 0 and total > 1:
        val_count = 1
    if val_limit is not None and val_limit > 0:
        val_count = min(val_limit, val_count if val_count > 0 else val_limit, total - 1 if total > 1 else total)
    val_ids = image_ids[:val_count]
    train_ids = image_ids[val_count:]
    if train_limit is not None and train_limit > 0:
        train_ids = train_ids[:train_limit]
    if not train_ids or not val_ids:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_training_split_empty")
    split_root = (SAM3_JOB_ROOT / "splits" / job_id).resolve()
    split_root.parent.mkdir(parents=True, exist_ok=True)
    if split_root.exists():
        shutil.rmtree(split_root, ignore_errors=True)
    (split_root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_root / "val" / "images").mkdir(parents=True, exist_ok=True)

    def _find_image_source(file_name: str) -> Optional[Path]:
        rel_path = Path(file_name)
        candidates = [
            dataset_root / rel_path,
            dataset_root / "train" / rel_path,
            dataset_root / "val" / rel_path,
            dataset_root / "train" / "images" / rel_path,
            dataset_root / "val" / "images" / rel_path,
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        if rel_path.is_absolute() and rel_path.exists():
            return rel_path
        return None

    def _write_split(target_ids: List[int], split_name: str) -> Tuple[str, int]:
        images_out: List[Dict[str, Any]] = []
        anns_out: List[Dict[str, Any]] = []
        for img_id in target_ids:
            info = images.get(img_id)
            if not info:
                continue
            file_name = info.get("file_name")
            if not file_name:
                continue
            src_path = _find_image_source(file_name)
            if src_path is None:
                continue
            rel_name = _normalise_relative_path(file_name)
            if rel_name.is_absolute():
                rel_name = Path(rel_name.name)
            dst_path = split_root / split_name / rel_name
            _link_or_copy_file(src_path, dst_path)
            info_out = dict(info)
            info_out["file_name"] = rel_name.as_posix()
            images_out.append(info_out)
            anns_out.extend(ann_by_image.get(img_id, []))
        ann_path = split_root / split_name / "_annotations.coco.json"
        _write_coco_annotations_impl(
            ann_path,
            dataset_id=meta.get("id") or dataset_root.name,
            categories=categories,
            images=images_out,
            annotations=anns_out,
        )
        return str(ann_path), len(images_out)

    coco_train_new, train_count = _write_split(train_ids, "train")
    coco_val_new, val_count = _write_split(val_ids, "val")
    new_meta = {
        **meta,
        "dataset_root": str(split_root),
        "coco_train_json": coco_train_new,
        "coco_val_json": coco_val_new,
        "train_count": train_count,
        "val_count": val_count,
        "image_count": train_count + val_count,
        "signature": _compute_dir_signature_impl(split_root),
        "source": meta.get("source", "resplit"),
    }
    _persist_sam3_dataset_metadata_impl(
        split_root,
        new_meta,
        meta_name=SAM3_DATASET_META_NAME,
        logger=logger,
    )
    summary = (
        f"SAM3 split: {train_count} train / {val_count} val "
        f"(seed={split_seed}, val_percent={vp:.2f}, src={dataset_root}) -> {split_root}"
    )
    logger.info(summary)
    if log_messages is not None:
        log_messages.append(summary)
    return new_meta


def _plan_segmentation_build(request: SegmentationBuildRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset_root = _resolve_sam3_or_qwen_dataset_impl(
        request.source_dataset_id,
        list_all_datasets_fn=_list_all_datasets,
        resolve_dataset_legacy_fn=lambda dataset_id: _resolve_dataset_legacy_impl(
            dataset_id,
            qwen_root=QWEN_DATASET_ROOT,
            sam3_root=SAM3_DATASET_ROOT,
            registry_root=DATASET_REGISTRY_ROOT,
            http_exception_cls=HTTPException,
        ),
    )
    source_meta = _load_qwen_dataset_metadata_impl(
        dataset_root,
        meta_name=QWEN_METADATA_FILENAME,
        load_json_metadata_fn=_load_json_metadata,
    ) or _load_sam3_dataset_metadata_impl(
        dataset_root,
        meta_name=SAM3_DATASET_META_NAME,
        load_json_metadata_fn=_load_json_metadata,
        persist_metadata_fn=lambda dataset_dir_inner, metadata: _persist_sam3_dataset_metadata_impl(
            dataset_dir_inner,
            metadata,
            meta_name=SAM3_DATASET_META_NAME,
            logger=logger,
        ),
    )
    if not source_meta:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="segmentation_source_metadata_missing")
    dataset_type = source_meta.get("type", "bbox")
    if dataset_type != "bbox":
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="segmentation_builder_requires_bbox")
    source_id = source_meta.get("id") or dataset_root.name
    suggested_name = f"{source_id}_seg"
    output_id = _safe_run_name(request.output_name, suggested_name)
    output_root = (SAM3_DATASET_ROOT / output_id).resolve()
    if not str(output_root).startswith(str(SAM3_DATASET_ROOT.resolve())):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="segmentation_output_path_invalid")
    if output_root.exists():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="segmentation_output_exists")

    classes = source_meta.get("classes") or []
    context = source_meta.get("context") or source_meta.get("dataset_context") or ""
    source_signature = source_meta.get("signature") or _compute_dir_signature_impl(dataset_root)
    planned_meta = {
        "id": output_id,
        "label": source_meta.get("label") or source_id,
        "type": "seg",
        "source": "segmentation_builder",
        "source_dataset_id": source_id,
        "source_dataset_root": str(dataset_root),
        "source_signature": source_signature,
        "generator_variant": request.sam_variant,
        "output_format": request.output_format,
        "classes": classes,
        "context": context,
        "created_at": time.time(),
    }
    planned_layout = {
        "dataset_root": str(output_root),
        "images_dir": str(output_root / "images"),
        "labels_dir": str(output_root / "labels"),
        "metadata_path": str(output_root / SAM3_DATASET_META_NAME),
        "log_dir": str(SEG_BUILDER_ROOT / "logs" / output_id),
    }
    return planned_meta, planned_layout


def _start_segmentation_build_job(request: SegmentationBuildRequest) -> SegmentationBuildJob:
    planned_meta, planned_layout = _plan_segmentation_build(request)
    job_id = str(uuid.uuid4())
    job = SegmentationBuildJob(
        job_id=job_id,
        status="queued",
        message="Queued",
        progress=0.0,
        config={
            "source_dataset_id": request.source_dataset_id,
            "sam_variant": request.sam_variant,
            "output_format": request.output_format,
            "planned_metadata": planned_meta,
            "planned_layout": planned_layout,
        },
    )
    with SEGMENTATION_BUILD_JOBS_LOCK:
        SEGMENTATION_BUILD_JOBS[job_id] = job

    def worker() -> None:
        try:
            _seg_job_update(job, status="running", progress=0.02, message="Preparing segmentation build…", error=None)
            source_meta = _resolve_sam3_dataset_meta(request.source_dataset_id)
            classes = source_meta.get("classes") or []
            if not classes:
                # Try to load from labelmap.txt directly.
                try:
                    labelmap_file = _resolve_sam3_or_qwen_dataset_impl(
                        request.source_dataset_id,
                        list_all_datasets_fn=_list_all_datasets,
                        resolve_dataset_legacy_fn=lambda dataset_id: _resolve_dataset_legacy_impl(
                            dataset_id,
                            qwen_root=QWEN_DATASET_ROOT,
                            sam3_root=SAM3_DATASET_ROOT,
                            registry_root=DATASET_REGISTRY_ROOT,
                            http_exception_cls=HTTPException,
                        ),
                    ) / "labelmap.txt"
                    classes = _load_labelmap_file(labelmap_file)
                except Exception:
                    classes = []
            if not classes:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="segmentation_builder_no_classes")
            dataset_root = Path(
                source_meta.get("dataset_root")
                or _resolve_sam3_or_qwen_dataset_impl(
                    request.source_dataset_id,
                    list_all_datasets_fn=_list_all_datasets,
                    resolve_dataset_legacy_fn=lambda dataset_id: _resolve_dataset_legacy_impl(
                        dataset_id,
                        qwen_root=QWEN_DATASET_ROOT,
                        sam3_root=SAM3_DATASET_ROOT,
                        registry_root=DATASET_REGISTRY_ROOT,
                        http_exception_cls=HTTPException,
                    ),
                )
            )
            labelmap_file = dataset_root / "labelmap.txt"
            if not labelmap_file.exists() and classes:
                # Backfill labelmap file if missing.
                try:
                    labelmap_file.write_text("\n".join(classes), encoding="utf-8")
                except Exception:
                    pass
            output_root = Path(planned_layout["dataset_root"]).resolve()
            train_out = output_root / "train"
            val_out = output_root / "val"
            (train_out / "images").mkdir(parents=True, exist_ok=True)
            (train_out / "labels").mkdir(parents=True, exist_ok=True)
            (val_out / "images").mkdir(parents=True, exist_ok=True)
            (val_out / "labels").mkdir(parents=True, exist_ok=True)
            # Copy/link labelmap.
            if labelmap_file.exists():
                shutil.copy2(labelmap_file, output_root / "labelmap.txt")
            splits = []
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

            def _find_image_for_label(labels_dir: Path, images_dir: Path, label_file: Path) -> Optional[Tuple[Path, Path]]:
                stem = label_file.stem
                for ext in image_exts:
                    candidate = images_dir / f"{stem}{ext}"
                    if candidate.exists():
                        try:
                            rel = candidate.relative_to(images_dir)
                        except Exception:
                            rel = Path(candidate.name)
                        return candidate, rel
                for candidate in images_dir.rglob(f"{stem}.*"):
                    if candidate.suffix.lower() in image_exts:
                        try:
                            rel = candidate.relative_to(images_dir)
                        except Exception:
                            rel = Path(candidate.name)
                        return candidate, rel
                return None

            image_uid = 1
            for split in ("train", "val"):
                images_dir = dataset_root / split / "images"
                labels_dir = dataset_root / split / "labels"
                if not images_dir.exists() or not labels_dir.exists():
                    continue
                entries = []
                for label_file in sorted(labels_dir.rglob("*.txt")):
                    match = _find_image_for_label(labels_dir, images_dir, label_file)
                    if match is None:
                        continue
                    image_path, rel_path = match
                    boxes = []
                    try:
                        with label_file.open("r", encoding="utf-8") as handle:
                            lines = [ln.strip() for ln in handle if ln.strip()]
                    except Exception:
                        continue
                    for ln in lines:
                        parts = ln.split()
                        if len(parts) < 5:
                            continue
                        try:
                            cls_idx = int(float(parts[0]))
                            cx = float(parts[1])
                            cy = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                        except (TypeError, ValueError):
                            continue
                        if classes and (cls_idx < 0 or cls_idx >= len(classes)):
                            continue
                        boxes.append({"class_idx": cls_idx, "bbox": (cx, cy, w, h)})
                    entries.append(
                        {
                            "label_file": label_file,
                            "image_path": image_path,
                            "rel_path": rel_path,
                            "boxes": boxes,
                            "split": split,
                            "image_id": image_uid,
                        }
                    )
                    image_uid += 1
                splits.append((split, entries))
            total_images = sum(len(e) for _, e in splits)
            if total_images == 0:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="segmentation_builder_no_images")
            _seg_job_log(job, f"Queued {total_images} images for conversion using {request.sam_variant.upper()}")
            for split_name, entries in splits:
                _seg_job_log(job, f"{split_name}: {len(entries)} images")
            base_devices = (
                _resolve_sam3_mining_devices_impl(SAM3_DEVICE_PREF, torch_module=torch, logger=logger)
                if request.sam_variant == "sam3"
                else _resolve_sam1_devices()
            )
            expanded_devices: List[torch.device] = []
            per_dev = max(1, int(os.environ.get("SEG_BUILDER_WORKERS_PER_DEVICE", "1")))
            max_total_env = os.environ.get("SEG_BUILDER_MAX_WORKERS")
            max_total = None
            try:
                if max_total_env is not None:
                    max_total = max(1, int(max_total_env))
            except Exception:
                max_total = None
            for dev in base_devices:
                for _ in range(per_dev):
                    expanded_devices.append(dev)
            if max_total is not None:
                expanded_devices = expanded_devices[:max_total]
            if not expanded_devices:
                raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="segmentation_builder_no_devices")
            mining_pool = None
            sam1_workers: List[_Sam1SegWorker] = []
            try:
                if request.sam_variant == "sam3":
                    mining_pool = _Sam3MiningPool(expanded_devices)
                else:
                    for dev in expanded_devices:
                        sam1_workers.append(_Sam1SegWorker(dev))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
            processed = 0
            simplify_eps = float(request.simplify_epsilon)
            mask_threshold = float(request.mask_threshold)
            min_threshold = float(request.score_threshold)
            max_results = int(max(1, request.max_results))
            min_area = float(request.min_size)
            progress_lock = threading.Lock()

            def _link_or_copy(src: Path, dst: Path) -> None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    return
                try:
                    os.link(src, dst)
                except Exception:
                    shutil.copy2(src, dst)

            def _process_entry(entry: Dict[str, Any], worker: Any) -> None:
                nonlocal processed
                if job.cancel_event.is_set():
                    return
                image_path: Path = entry["image_path"]
                rel_path: Path = entry["rel_path"]
                split: str = entry["split"]
                boxes: List[Dict[str, Any]] = entry.get("boxes") or []
                tasks: List[Dict[str, Any]] = []
                try:
                    with Image.open(image_path) as im:
                        pil_img = im.convert("RGB")
                        width, height = pil_img.size
                        for idx, box in enumerate(boxes):
                            cx, cy, bw, bh = box["bbox"]
                            x1, y1, x2, y2 = _yolo_to_xyxy(width, height, (cx, cy, bw, bh))
                            tasks.append(
                                {
                                    "id": f"{rel_path}:{idx}",
                                    "type": "visual",
                                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                                    "class_idx": box["class_idx"],
                                    "fallback_poly": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                                }
                            )
                        if not tasks:
                            outputs = {}
                        else:
                            outputs = worker.process_image(
                                image_id=entry.get("image_id", 0),
                                pil_img=pil_img,
                                tasks=tasks,
                                min_threshold=min_threshold,
                                mask_threshold=mask_threshold,
                                max_results=max_results,
                                min_size=min_area,
                                simplify=simplify_eps,
                                return_masks=True,
                            ) or {}
                            label_lines = []
                            for task in tasks:
                                task_id = task["id"]
                                class_idx = task["class_idx"]
                                fallback = task["fallback_poly"]
                                dets = outputs.get(task_id) or []
                                best = None
                                if dets:
                                    best = max(dets, key=lambda d: d.get("score") or 0.0)
                                mask_arr = None
                                best_score = best.get("score") if best else None
                                if best:
                                    mask_arr = best.get("mask_array")
                                    if mask_arr is None and best.get("mask"):
                                        mask_arr = _decode_binary_mask_impl(best.get("mask"))
                                polygon = _mask_to_polygon_impl(mask_arr, simplify_eps) if mask_arr is not None else []
                                if best_score is None or best_score < min_threshold:
                                    polygon = []
                                if len(polygon) < 3:
                                    polygon = fallback
                                coords: List[float] = []
                                for x, y in polygon:
                                    coords.extend(
                                        [
                                            max(0.0, min(1.0, x / width)),
                                            max(0.0, min(1.0, y / height)),
                                        ]
                                    )
                                if len(coords) >= 6:
                                    label_lines.append(f"{class_idx} " + " ".join(f"{v:.6f}" for v in coords))
                        dest_labels = (train_out if split == "train" else val_out) / "labels" / f"{rel_path.stem}.txt"
                        dest_images = (train_out if split == "train" else val_out) / "images" / rel_path
                        dest_labels.parent.mkdir(parents=True, exist_ok=True)
                        dest_images.parent.mkdir(parents=True, exist_ok=True)
                        _link_or_copy(image_path, dest_images)
                        dest_labels.write_text("\n".join(label_lines), encoding="utf-8")
                finally:
                    with progress_lock:
                        processed += 1
                        progress_val = min(1.0, 0.05 + 0.9 * (processed / max(total_images, 1)))
                    _seg_job_update(
                        job,
                        progress=progress_val,
                        message=f"Processed {processed}/{total_images} images ({progress_val*100:.1f}%)",
                        log_message=False,
                    )

            # Dispatch over workers
            try:
                workers_list = mining_pool.workers if mining_pool is not None else sam1_workers
                if not workers_list:
                    raise RuntimeError("segmentation_builder_no_workers")
                with ThreadPoolExecutor(max_workers=max(1, len(workers_list))) as executor:
                    futures = []
                    task_idx = 0
                    for _, entries in splits:
                        for entry in entries:
                            worker = workers_list[task_idx % len(workers_list)]
                            futures.append(executor.submit(_process_entry, entry, worker))
                            task_idx += 1
                    for fut in as_completed(futures):
                        if job.cancel_event.is_set():
                            break
                        try:
                            fut.result()
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Segmentation build worker failed: %s", exc)
            finally:
                try:
                    if mining_pool is not None:
                        mining_pool.close()
                except Exception:
                    pass
                if sam1_workers:
                    for worker in sam1_workers:
                        try:
                            worker.close()
                        except Exception:
                            pass
            if job.cancel_event.is_set():
                _seg_job_update(job, status="cancelled", message="Cancelled", progress=job.progress)
                return
            _seg_job_log(job, "Converting output to COCO…")
            try:
                coco_meta = _convert_yolo_dataset_to_coco_impl(output_root)
            except Exception as exc:  # noqa: BLE001
                _seg_job_update(job, status="failed", message="COCO conversion failed", error=str(exc))
                return
            result_meta = _load_sam3_dataset_metadata_impl(
                output_root,
                meta_name=SAM3_DATASET_META_NAME,
                load_json_metadata_fn=_load_json_metadata,
                persist_metadata_fn=lambda dataset_dir_inner, metadata: _persist_sam3_dataset_metadata_impl(
                    dataset_dir_inner,
                    metadata,
                    meta_name=SAM3_DATASET_META_NAME,
                    logger=logger,
                ),
            ) or coco_meta or planned_meta
            _seg_job_update(
                job,
                status="completed",
                progress=1.0,
                message="Segmentation build complete.",
                result={
                    "planned_metadata": planned_meta,
                    "output_dataset_id": result_meta.get("id") if isinstance(result_meta, dict) else planned_meta.get("id"),
                    "output_root": str(output_root),
                    "classes": classes,
                    "train_count": len(next((e for s, e in splits if s == "train"), [])),
                    "val_count": len(next((e for s, e in splits if s == "val"), [])),
                },
            )
        except HTTPException as exc:
            _seg_job_update(job, status="failed", message=str(exc.detail), error=str(exc.detail))
        except Exception as exc:  # noqa: BLE001
            _seg_job_update(job, status="failed", message=str(exc), error=str(exc))

    threading.Thread(target=worker, daemon=True, name=f"seg-build-{job_id[:8]}").start()
    return job


def _latest_checkpoint_in_dir(checkpoint_dir: Path) -> Optional[str]:
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    return None


def _save_sam3_config(cfg: OmegaConf, job_id: str) -> Tuple[str, Path]:
    SAM3_GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_file = SAM3_GENERATED_CONFIG_DIR / f"{job_id}.yaml"
    yaml_text = OmegaConf.to_yaml(cfg)
    config_file.write_text("# @package _global_\n" + yaml_text, encoding="utf-8")
    return f"configs/generated/{config_file.name}", config_file


def _start_sam3_training_worker(
    job: Sam3TrainingJob,
    cfg: OmegaConf,
    num_gpus: int,
    *,
    val_score_thresh: Optional[float] = None,
    val_max_dets: Optional[int] = None,
) -> None:
    def worker():
        proc: Optional[subprocess.Popen] = None
        tail_logs: deque[str] = deque(maxlen=50)
        max_epochs = max(1, int(getattr(cfg.trainer, "max_epochs", 1) or 1))
        # Attempt to track steps per epoch from the config if present
        steps_per_epoch = None
        try:
            steps_per_epoch = int(cfg.scratch.target_epoch_size) if getattr(cfg.scratch, "target_epoch_size", None) else None
        except Exception:
            steps_per_epoch = None
        try:
            _prepare_for_training_impl(
                unload_inference_runtimes_fn=lambda: _unload_inference_runtimes_impl(
                    unload_non_qwen_fn=lambda: _unload_non_qwen_runtimes_impl(
                        predictor_manager=predictor_manager,
                        unload_sam3_text_fn=_unload_sam3_text_runtime,
                        suspend_clip_fn=_suspend_clip_backbone,
                        unload_dinov3_fn=_unload_dinov3_backbone,
                        unload_detector_fn=_unload_detector_inference,
                        torch_module=torch,
                        logger=logger,
                    ),
                    unload_qwen_fn=_unload_qwen_runtime,
                    torch_module=torch,
                )
            )
            _sam3_job_update(job, status="running", progress=0.05, message="Preparing SAM3 training job ...")
            config_name, config_file = _save_sam3_config(cfg, job.job_id)
            script_path = SAM3_PACKAGE_ROOT / "train" / "train.py"
            cmd = [sys.executable, str(script_path), "-c", config_name, "--use-cluster", "0"]
            if num_gpus is not None:
                cmd.extend(["--num-gpus", str(num_gpus)])
            env = os.environ.copy()
            existing_py = env.get("PYTHONPATH", "")
            py_root = f"{SAM3_VENDOR_ROOT}:{SAM3_REPO_ROOT}"
            env["PYTHONPATH"] = f"{py_root}:{existing_py}" if existing_py else py_root
            env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
            env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
            env.setdefault("NCCL_DEBUG", "INFO")
            # Enable runtime monkeypatches (loaded via sitecustomize.py) to keep vendor tree untouched.
            env.setdefault("SAM3_MONKEYPATCH", "1")
            if val_score_thresh is not None:
                try:
                    env["SAM3_VAL_SCORE_THRESH"] = str(float(val_score_thresh))
                except Exception:
                    pass
            if val_max_dets is not None:
                try:
                    env["SAM3_VAL_MAX_DETS"] = str(int(val_max_dets))
                except Exception:
                    pass
            proc = subprocess.Popen(
                cmd,
                cwd=str(SAM3_VENDOR_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            job.process = proc
            _sam3_job_log(job, f"Spawned {' '.join(cmd)}")
            while True:
                if proc.stdout is None:
                    break
                line = proc.stdout.readline()
                if line == "" and proc.poll() is not None:
                    break
                if not line:
                    continue
                if job.cancel_event.is_set() and proc.poll() is None:
                    proc.terminate()
                    _sam3_job_update(job, status="cancelling", message="Cancellation requested ...")
                    continue
                cleaned = line.rstrip("\n")
                tail_logs.append(cleaned)
                _sam3_job_log(job, cleaned)
                if "sam3-balance" in cleaned.lower() or cleaned.startswith("[sam3-balance]"):
                    job.result = job.result or {}
                    job.result["balance_info"] = cleaned
                try:
                    match = re.search(r"Train Epoch:\s*\[(\d+)\]\[\s*(\d+)\s*/\s*(\d+)\]", cleaned)
                    val_match = re.search(r"Val Epoch:\s*\[(\d+)\]\[\s*(\d+)\s*/\s*(\d+)\]", cleaned)
                    if val_match:
                        val_epoch_idx = int(val_match.group(1))
                        val_step_idx = int(val_match.group(2))
                        val_total_steps = max(1, int(val_match.group(3)))
                        val_frac = max(0.0, min(1.0, val_step_idx / val_total_steps))
                        prog_val = max(job.progress or 0.0, min(0.99, 0.9 + 0.1 * val_frac))
                        _sam3_job_update(
                            job,
                            progress=prog_val,
                            message=f"Validation running ({val_step_idx}/{val_total_steps})",
                            log_message=False,
                        )
                        _sam3_job_append_metric(
                            job,
                            {
                                "phase": "val",
                                "val_step": val_step_idx,
                                "val_total": val_total_steps,
                                "epoch": val_epoch_idx + 1,
                                "total_epochs": max_epochs,
                            },
                        )
                    if match:
                        epoch_idx = int(match.group(1))
                        step_idx = int(match.group(2))
                        total_steps = max(1, int(match.group(3)))
                        # Prefer log-reported total steps; fall back to config target if present
                        steps_in_epoch = total_steps or steps_per_epoch or total_steps
                        frac_epoch = (step_idx / steps_in_epoch) if steps_in_epoch else 0.0
                        frac = (epoch_idx + frac_epoch) / max_epochs
                        prog_val = max(0.05, min(0.99, frac))
                        _sam3_job_update(job, progress=prog_val, log_message=False)
                    loss_match = re.search(
                        r"Losses\/train_all_loss:\s*(?:(?:last|batch)=)?([0-9.+-eE]+)(?:.*?(?:avg\d*=?\s*([0-9.+-eE]+)|\(\s*([0-9.+-eE]+)\s*\)))?",
                        cleaned,
                    )
                    if loss_match and match:
                        instant = float(loss_match.group(1))
                        avg_loss = None
                        if loss_match.group(2):
                            avg_loss = float(loss_match.group(2))
                        elif loss_match.group(3):
                            avg_loss = float(loss_match.group(3))
                        total_steps = max(1, int(match.group(3)))
                        steps_in_epoch = total_steps or steps_per_epoch or total_steps
                        global_step = epoch_idx * steps_in_epoch + step_idx
                        metric_payload = {
                            "phase": "train",
                            "train_loss_batch": instant,
                            "train_loss_avg10": avg_loss,
                            "batch": step_idx,
                            "batches_per_epoch": steps_in_epoch,
                            "epoch": epoch_idx + 1,
                            "total_epochs": max_epochs,
                            "step": global_step,
                            "timestamp": time.time(),
                        }
                        _sam3_job_append_metric(job, metric_payload)
                    if "Meters:" in cleaned and "coco_eval_bbox_AP" in cleaned:
                        try:
                            # Extract key/value pairs like '...': np.float64(0.123)
                            pairs = re.findall(r"'([^']+)':\s*np\.float64\(([0-9.eE+-]+)\)", cleaned)
                            meter_map = {k: float(v) for k, v in pairs}
                            epoch_meta = re.search(r"'Trainer/epoch':\s*([0-9]+)", cleaned)
                            epoch_val = int(epoch_meta.group(1)) + 1 if epoch_meta else None
                            val_payload: Dict[str, Any] = {
                                "phase": "val",
                                "timestamp": time.time(),
                            }
                            if epoch_val is not None:
                                val_payload["epoch"] = epoch_val
                            # Pick the first coco_eval_bbox_* metrics if present.
                            for key, field in [
                                ("coco_eval_bbox_AP", "coco_ap"),
                                ("coco_eval_bbox_AP_50", "coco_ap50"),
                                ("coco_eval_bbox_AP_75", "coco_ap75"),
                                ("coco_eval_bbox_AR_maxDets@10", "coco_ar10"),
                                ("coco_eval_bbox_AR_maxDets@100", "coco_ar100"),
                            ]:
                                for meter_key, meter_val in meter_map.items():
                                    if meter_key.endswith(key):
                                        val_payload[field] = meter_val
                                        break
                            _sam3_job_append_metric(job, val_payload)
                        except Exception:
                            pass
                except Exception:
                    pass
                _sam3_job_update(job, message=cleaned[-200:], log_message=False)
            retcode = proc.wait() if proc else 1
            if job.cancel_event.is_set():
                _sam3_job_update(job, status="cancelled", message="Training cancelled")
                return
            if retcode != 0:
                sig_note = ""
                if retcode < 0:
                    sig_num = -retcode
                    try:
                        sig_name = signal.Signals(sig_num).name
                    except Exception:
                        sig_name = f"SIG{sig_num}"
                    sig_desc = signal.strsignal(sig_num) or sig_name
                    sig_note = f" (signal {sig_num}: {sig_desc})"
                tail_text = "\n".join(tail_logs)
                _sam3_job_update(
                    job,
                    status="failed",
                    message=f"Training failed (exit {retcode}{sig_note})",
                    error=f"exit_code:{retcode}{sig_note}\nlast_logs:\n{tail_text}",
                )
                return
            log_dir = Path(cfg.paths.experiment_log_dir)
            checkpoint_dir = log_dir / "checkpoints"
            latest_ckpt = _latest_checkpoint_in_dir(checkpoint_dir)
            seg_head = bool(getattr(cfg.scratch, "enable_segmentation_head", getattr(cfg.scratch, "enable_segmentation", True)))
            load_seg = bool(getattr(cfg.scratch, "load_segmentation", seg_head))
            result_payload = {
                "experiment_log_dir": str(log_dir),
                "checkpoint": latest_ckpt,
                "config_path": str(config_file),
                "enable_segmentation": seg_head,
                "enable_segmentation_head": seg_head,
                "load_segmentation": load_seg,
            }
            _sam3_job_update(job, status="succeeded", message="Training complete", progress=1.0, result=result_payload)
        except Exception as exc:  # noqa: BLE001
            _sam3_job_update(job, status="failed", message="Training crashed", error=str(exc))
        finally:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            _finalize_training_environment_impl(
                resume_classifier_fn=_resume_classifier_backbone,
                torch_module=torch,
            )

    thread = threading.Thread(target=worker, name=f"sam3-train-{job.job_id}", daemon=True)
    thread.start()


def _start_yolo_training_worker(job: YoloTrainingJob) -> None:
    def worker() -> None:
        run_dir = _yolo_run_dir_impl(
            job.job_id,
            create=True,
            job_root=YOLO_JOB_ROOT,
            sanitize_fn=_sanitize_yolo_run_id_impl,
            http_exception_cls=HTTPException,
        )
        write_run_meta = lambda meta: _yolo_write_run_meta_impl(
            run_dir,
            meta,
            meta_name=YOLO_RUN_META_NAME,
            time_fn=time.time,
        )
        config = dict(job.config or {})
        dataset_info = config.get("dataset") or {}
        task = str(dataset_info.get("task") or config.get("task") or "detect").lower()
        if job.cancel_event.is_set():
            _yolo_job_update(job, status="cancelled", message="Cancelled before start", progress=0.0)
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        if not dataset_info.get("yolo_ready"):
            _yolo_job_update(job, status="failed", message="Dataset is not YOLO-ready", error="yolo_not_ready")
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        try:
            _prepare_for_training_impl(
                unload_inference_runtimes_fn=lambda: _unload_inference_runtimes_impl(
                    unload_non_qwen_fn=lambda: _unload_non_qwen_runtimes_impl(
                        predictor_manager=predictor_manager,
                        unload_sam3_text_fn=_unload_sam3_text_runtime,
                        suspend_clip_fn=_suspend_clip_backbone,
                        unload_dinov3_fn=_unload_dinov3_backbone,
                        unload_detector_fn=_unload_detector_inference,
                        torch_module=torch,
                        logger=logger,
                    ),
                    unload_qwen_fn=_unload_qwen_runtime,
                    torch_module=torch,
                )
            )
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # noqa: BLE001
            _yolo_job_update(job, status="failed", message="Ultralytics not installed", error=str(exc))
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        _yolo_job_update(job, status="running", message="Starting YOLOv8 training", progress=0.0)
        _yolo_job_log(job, "Preparing dataset + data.yaml")
        dataset_root = Path(dataset_info.get("prepared_root") or dataset_info.get("dataset_root") or "")
        data_yaml = _yolo_write_data_yaml_impl(
            run_dir,
            dataset_root,
            dataset_info.get("yolo_layout"),
            dataset_info.get("yolo_labelmap_path"),
            resolve_split_paths_fn=_yolo_resolve_split_paths_impl,
            yolo_load_labelmap_fn=_yolo_load_labelmap_impl,
            yaml_dump_fn=lambda data: yaml.safe_dump(data, sort_keys=False),
            copy_file_fn=shutil.copy2,
        )
        from_scratch = bool(config.get("from_scratch"))
        base_weights = config.get("base_weights")
        variant = config.get("variant") or ""
        if task == "segment" and _yolo_p2_scale_impl(variant):
            _yolo_job_update(job, status="failed", message="P2 head is only supported for detection.", error="yolo_p2_segment")
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        _, model_source = _yolo_resolve_model_source_impl(
            variant,
            task,
            from_scratch,
            base_weights,
            p2_scale_fn=_yolo_p2_scale_impl,
        )
        device_arg = _yolo_device_arg_impl(config.get("devices"))
        train_kwargs = {
            "data": str(data_yaml),
            "task": task,
            "epochs": config.get("epochs"),
            "imgsz": config.get("img_size"),
            "batch": config.get("batch"),
            "workers": config.get("workers"),
            "seed": config.get("seed"),
            "device": device_arg,
            "project": str(run_dir),
            "name": "train",
            "exist_ok": True,
        }
        p2_scale = _yolo_p2_scale_impl(variant)
        if p2_scale and model_source.endswith("yolov8-p2.yaml"):
            try:
                import ultralytics  # type: ignore
            except Exception as exc:  # noqa: BLE001
                _yolo_job_update(job, status="failed", message="Ultralytics not installed", error=str(exc))
                write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
                return
            base_cfg = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "v8" / "yolov8-p2.yaml"
            cfg_payload = yaml.safe_load(base_cfg.read_text())
            cfg_payload["scale"] = p2_scale
            p2_cfg = run_dir / f"yolov8{p2_scale}-p2.yaml"
            p2_cfg.write_text(yaml.safe_dump(cfg_payload, sort_keys=False))
            model_source = str(p2_cfg)
            if base_weights:
                train_kwargs["pretrained"] = base_weights
            else:
                train_kwargs["pretrained"] = False
            _yolo_job_log(job, f"P2 variant: scale={p2_scale} (config={p2_cfg.name}, pretrained={train_kwargs.get('pretrained')})")
        _yolo_job_log(job, f"Model source: {model_source}")
        train_kwargs.update(_yolo_build_aug_args_impl(config.get("augmentations")))
        train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
        monitor_stop = threading.Event()
        monitor_thread = None
        try:
            model = YOLO(model_source)
            _yolo_job_log(job, "Training started")
            monitor_thread = threading.Thread(
                target=_yolo_monitor_training_impl,
                args=(job, run_dir, int(config.get("epochs") or 0), monitor_stop),
                kwargs={
                    "parse_results_fn": _yolo_parse_results_csv_impl,
                    "job_append_metric_fn": _yolo_job_append_metric,
                    "job_update_fn": _yolo_job_update,
                },
                name=f"yolo-monitor-{job.job_id[:8]}",
                daemon=True,
            )
            monitor_thread.start()
            results = model.train(**train_kwargs)
            train_dir = run_dir / "train"
            best_path = train_dir / "weights" / "best.pt"
            if best_path.exists():
                shutil.copy2(best_path, run_dir / "best.pt")
            results_csv = train_dir / "results.csv"
            args_yaml = train_dir / "args.yaml"
            if results_csv.exists():
                shutil.copy2(results_csv, run_dir / "results.csv")
            if args_yaml.exists():
                shutil.copy2(args_yaml, run_dir / "args.yaml")
            metrics_series: List[Dict[str, Any]] = []
            series_path = run_dir / "metrics_series.json"
            if (run_dir / "results.csv").exists():
                metrics_series = _yolo_parse_results_csv_impl(run_dir / "results.csv")
                if metrics_series:
                    try:
                        series_path.write_text(json.dumps(metrics_series, indent=2, sort_keys=True))
                        job.metrics = metrics_series
                    except Exception:
                        pass
            metrics_payload = {}
            try:
                metrics_payload = results.metrics if results else {}
            except Exception:
                metrics_payload = {}
            if metrics_payload:
                (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
            _yolo_prune_run_dir_impl(
                run_dir,
                keep_files_default=YOLO_KEEP_FILES,
                dir_size_fn=_dir_size_bytes,
                meta_name=YOLO_RUN_META_NAME,
            )
            result_payload = {
                "run_dir": str(run_dir),
                "best_path": str(run_dir / "best.pt") if (run_dir / "best.pt").exists() else None,
                "metrics_path": str(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else None,
                "metrics_series_path": str(series_path) if series_path.exists() else None,
            }
            _yolo_job_update(job, status="succeeded", message="Training complete", progress=1.0, result=result_payload)
        except Exception as exc:  # noqa: BLE001
            _yolo_job_update(job, status="failed", message="Training failed", error=str(exc))
        finally:
            monitor_stop.set()
            if monitor_thread:
                monitor_thread.join(timeout=2.0)
            _finalize_training_environment_impl(
                resume_classifier_fn=_resume_classifier_backbone,
                torch_module=torch,
            )
            write_run_meta(
                {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config, "result": job.result}
            )

    thread = threading.Thread(target=worker, name=f"yolo-train-{job.job_id}", daemon=True)
    thread.start()


def _start_yolo_head_graft_worker(job: YoloHeadGraftJob) -> None:
    def worker() -> None:
        job.thread_ident = threading.get_ident()
        run_dir = _yolo_run_dir_impl(
            job.job_id,
            create=True,
            job_root=YOLO_JOB_ROOT,
            sanitize_fn=_sanitize_yolo_run_id_impl,
            http_exception_cls=HTTPException,
        )
        write_run_meta = lambda meta: _yolo_write_run_meta_impl(
            run_dir,
            meta,
            meta_name=YOLO_RUN_META_NAME,
            time_fn=time.time,
        )
        config = dict(job.config or {})
        if run_dir:
            config.setdefault("paths", {})["run_dir"] = str(run_dir)
            job.config = config
        base_run_id = str(config.get("base_run_id") or "").strip()
        if not base_run_id:
            _yolo_head_graft_job_update(job, status="failed", message="Base YOLO run missing", error="yolo_base_missing")
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        if job.cancel_event.is_set():
            _yolo_head_graft_job_update(job, status="cancelled", message="Cancelled before start", progress=0.0)
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        base_run_dir = _yolo_run_dir_impl(
            base_run_id,
            create=False,
            job_root=YOLO_JOB_ROOT,
            sanitize_fn=_sanitize_yolo_run_id_impl,
            http_exception_cls=HTTPException,
        )
        base_best = base_run_dir / "best.pt"
        if not base_best.exists():
            _yolo_head_graft_job_update(job, status="failed", message="Base run is missing best.pt", error="yolo_base_missing_best")
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        base_meta = _yolo_load_run_meta_impl(base_run_dir, meta_name=YOLO_RUN_META_NAME)
        base_cfg = base_meta.get("config") or {}
        base_task = str(base_cfg.get("task") or "detect").lower()
        if base_task != "detect":
            _yolo_head_graft_job_update(job, status="failed", message="Base run is not detect", error="yolo_base_not_detect")
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        base_variant = config.get("variant") or base_cfg.get("variant")
        if not base_variant:
            _yolo_head_graft_job_update(
                job,
                status="failed",
                message="Base run missing variant (cannot infer architecture)",
                error="yolo_base_variant_missing",
            )
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        base_labelmap = _yolo_load_run_labelmap_impl(
            base_run_dir,
            yolo_load_labelmap_fn=_yolo_load_labelmap_impl,
            yaml_load_fn=yaml.safe_load,
        )
        if not base_labelmap:
            _yolo_head_graft_job_update(job, status="failed", message="Base labelmap missing", error="yolo_base_labelmap_missing")
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        try:
            dataset_payload = YoloTrainRequest(dataset_id=config.get("dataset_id"), dataset_root=config.get("dataset_root"))
            dataset_info = _resolve_yolo_training_dataset(dataset_payload)
        except Exception as exc:  # noqa: BLE001
            _yolo_head_graft_job_update(job, status="failed", message="Dataset not ready", error=str(exc))
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        if not dataset_info.get("yolo_ready"):
            _yolo_head_graft_job_update(job, status="failed", message="Dataset is not YOLO-ready", error="yolo_not_ready")
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        dataset_task = str(dataset_info.get("task") or "detect").lower()
        if dataset_task != "detect":
            _yolo_head_graft_job_update(job, status="failed", message="Head grafting only supports detect datasets", error="yolo_graft_detect_only")
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        new_labelmap = _yolo_load_labelmap_impl(Path(dataset_info.get("yolo_labelmap_path") or ""))
        if not new_labelmap:
            _yolo_head_graft_job_update(job, status="failed", message="New labelmap missing", error="yolo_new_labelmap_missing")
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        base_norm = {_normalize_class_name_for_match(n) for n in base_labelmap if n}
        new_norm = {_normalize_class_name_for_match(n) for n in new_labelmap if n}
        overlap = base_norm.intersection(new_norm)
        if overlap:
            _yolo_head_graft_job_update(
                job,
                status="failed",
                message="Base and new class lists overlap",
                error="yolo_labelmap_overlap",
            )
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        try:
            _prepare_for_training_impl(
                unload_inference_runtimes_fn=lambda: _unload_inference_runtimes_impl(
                    unload_non_qwen_fn=lambda: _unload_non_qwen_runtimes_impl(
                        predictor_manager=predictor_manager,
                        unload_sam3_text_fn=_unload_sam3_text_runtime,
                        suspend_clip_fn=_suspend_clip_backbone,
                        unload_dinov3_fn=_unload_dinov3_backbone,
                        unload_detector_fn=_unload_detector_inference,
                        torch_module=torch,
                        logger=logger,
                    ),
                    unload_qwen_fn=_unload_qwen_runtime,
                    torch_module=torch,
                )
            )
            _patch_ultralytics_for_head_grafting()
            import ultralytics  # type: ignore
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # noqa: BLE001
            _yolo_head_graft_job_update(job, status="failed", message="Ultralytics not installed", error=str(exc))
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        version = getattr(ultralytics, "__version__", "")
        if version and not version.startswith("8."):
            _yolo_head_graft_job_update(
                job,
                status="failed",
                message=f"Ultralytics {version} unsupported for head grafting",
                error="yolo_graft_ultralytics_version",
            )
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        _yolo_head_graft_job_update(job, status="running", message="Preparing head graft", progress=0.0)
        _yolo_head_graft_job_log(job, f"Base run: {base_run_id}")
        _yolo_head_graft_job_log(job, f"Variant: {base_variant}")
        _yolo_head_graft_audit(
            job,
            "head_graft_start",
            event="start",
            extra={"base_run_id": base_run_id, "variant": base_variant, "dataset_id": config.get("dataset_id")},
        )
        data_yaml = _yolo_write_data_yaml_impl(
            run_dir,
            Path(dataset_info.get("prepared_root") or dataset_info.get("dataset_root") or ""),
            dataset_info.get("yolo_layout"),
            dataset_info.get("yolo_labelmap_path"),
            resolve_split_paths_fn=_yolo_resolve_split_paths_impl,
            yolo_load_labelmap_fn=_yolo_load_labelmap_impl,
            yaml_dump_fn=lambda data: yaml.safe_dump(data, sort_keys=False),
            copy_file_fn=shutil.copy2,
        )
        variant_base_yaml_fn = lambda variant, task, run_dir=None: _yolo_variant_base_yaml_impl(
            variant,
            task,
            run_dir=run_dir,
            http_exception_cls=HTTPException,
            import_ultralytics_fn=lambda: __import__("ultralytics"),  # type: ignore
            yaml_load_fn=yaml.safe_load,
            yaml_dump_fn=lambda payload: yaml.safe_dump(payload, sort_keys=False),
            upload_root=UPLOAD_ROOT,
            p2_scale_fn=_yolo_p2_scale_impl,
        )
        find_detect_modules_fn = lambda model: _yolo_find_detect_modules_impl(
            model,
            import_detect_cls_fn=lambda: __import__("ultralytics.nn.tasks", fromlist=["Detect"]).Detect,  # type: ignore
        )
        nc_new = len(new_labelmap)
        nc_base = len(base_labelmap)
        head_yaml = _yolo_write_variant_yaml_impl(
            run_dir,
            base_variant,
            "detect",
            nc_new,
            variant_base_yaml_fn=variant_base_yaml_fn,
            yaml_load_fn=yaml.safe_load,
            yaml_dump_fn=lambda payload: yaml.safe_dump(payload, sort_keys=False),
        )
        device_arg = _yolo_device_arg_impl(config.get("devices"))
        train_kwargs = {
            "data": str(data_yaml),
            "task": "detect",
            "epochs": config.get("epochs"),
            "imgsz": config.get("img_size"),
            "batch": config.get("batch"),
            "workers": config.get("workers"),
            "seed": config.get("seed"),
            "device": device_arg,
            "project": str(run_dir),
            "name": "head",
            "exist_ok": True,
        }
        train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
        try:
            model = YOLO(str(head_yaml)).load(str(base_best))
            detect_idx = _yolo_detect_layer_index_impl(
                model.model,
                find_detect_modules_fn=find_detect_modules_fn,
            )

            def _freeze_bn(trainer):
                for idx, layer in enumerate(trainer.model.model):
                    if idx >= detect_idx:
                        continue
                    for sub in layer.modules():
                        if isinstance(sub, torch.nn.BatchNorm2d):
                            sub.eval()
                            sub.track_running_stats = False

            def _cancel_guard(trainer):
                if job.cancel_event.is_set():
                    trainer.stop = True
                    _yolo_head_graft_job_log(job, "Cancellation requested; stopping after current step")

            model.add_callback("on_train_epoch_start", _freeze_bn)
            model.add_callback("on_pretrain_routine_start", _freeze_bn)
            model.add_callback("on_train_epoch_start", _cancel_guard)
            model.add_callback("on_train_epoch_end", _cancel_guard)
            train_kwargs["freeze"] = detect_idx
            _yolo_head_graft_job_log(job, f"Training new head (freeze < layer {detect_idx})")
            model.train(**train_kwargs)
            train_dir = run_dir / "head"
            new_best = train_dir / "weights" / "best.pt"
            if not new_best.exists():
                raise RuntimeError("new_head_best_missing")
            if job.cancel_event.is_set():
                _yolo_head_graft_job_update(job, status="cancelled", message="Head training cancelled", progress=job.progress)
                write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
                _finalize_training_environment_impl(
                    resume_classifier_fn=_resume_classifier_backbone,
                    torch_module=torch,
                )
                return
        except Exception as exc:  # noqa: BLE001
            _yolo_head_graft_job_update(job, status="failed", message="Head training failed", error=str(exc))
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            _finalize_training_environment_impl(
                resume_classifier_fn=_resume_classifier_backbone,
                torch_module=torch,
            )
            return
        _yolo_head_graft_job_update(job, message="Merging heads", progress=0.7)
        merged_yaml = _yolo_write_head_graft_yaml_impl(
            run_dir,
            base_variant,
            nc_base,
            nc_new,
            variant_base_yaml_fn=variant_base_yaml_fn,
            yaml_load_fn=yaml.safe_load,
            yaml_dump_fn=lambda payload: yaml.safe_dump(payload, sort_keys=False),
            http_exception_cls=HTTPException,
        )
        try:
            merged = YOLO(str(merged_yaml)).load(str(base_best))
            new_model = YOLO(str(new_best))
            new_detects = find_detect_modules_fn(new_model.model)
            merged_detects = find_detect_modules_fn(merged.model)
            if len(new_detects) < 1 or len(merged_detects) < 2:
                raise RuntimeError("detect_module_missing")
            merged_detects[1].load_state_dict(new_detects[0].state_dict(), strict=False)
            merged_labelmap = list(base_labelmap) + list(new_labelmap)
            labelmap_path = run_dir / "labelmap.txt"
            labelmap_path.write_text("\n".join(merged_labelmap) + "\n")
            merged_names = {idx: name for idx, name in enumerate(merged_labelmap)}
            merged.model.names = merged_names
            try:
                merged.names = merged_names
            except Exception:
                pass
            merged.ckpt = {"model": merged.model}
            merged.save(str(run_dir / "best.pt"))
            # Sanity check inference shape / class index bounds
            sanity_errors: List[str] = []
            try:
                images_root = Path(dataset_info.get("yolo_images_dir") or "")
                if images_root.exists():
                    image_paths = sorted(list(images_root.glob("*")))[:3]
                    if image_paths:
                        merged.model.eval()
                        for image_path in image_paths:
                            try:
                                preds = merged.predict(str(image_path), verbose=False)
                            except Exception as exc:  # noqa: BLE001
                                sanity_errors.append(f"predict_failed:{image_path.name}:{exc}")
                                continue
                            for pred in preds or []:
                                boxes = getattr(pred, "boxes", None)
                                if boxes is None:
                                    continue
                                cls = getattr(boxes, "cls", None)
                                if cls is None:
                                    continue
                                try:
                                    max_cls = int(cls.max().item()) if hasattr(cls, "max") else int(max(cls))
                                except Exception:
                                    max_cls = None
                                if max_cls is not None and max_cls >= len(merged_labelmap):
                                    sanity_errors.append(f"class_index_out_of_range:{image_path.name}:{max_cls}")
                    else:
                        sanity_errors.append("sanity_images_missing")
                else:
                    sanity_errors.append("sanity_images_missing")
            except Exception as exc:  # noqa: BLE001
                sanity_errors.append(f"sanity_check_failed:{exc}")
            if sanity_errors:
                _yolo_head_graft_audit(job, "sanity_check_failed", event="sanity", level="warn", extra={"errors": sanity_errors})
            else:
                _yolo_head_graft_audit(job, "sanity_check_ok", event="sanity")
            export_path = None
            if config.get("export_onnx"):
                _yolo_head_graft_audit(
                    job,
                    "onnx_export_requested",
                    event="export",
                    level="warn",
                    extra={"note": "ConcatHead ONNX export may fail depending on runtime support."},
                )
                try:
                    export_path = merged.export(format="onnx")
                except Exception as exc:  # noqa: BLE001
                    _yolo_head_graft_job_log(job, f"ONNX export failed: {exc}")
            result_payload = {
                "run_dir": str(run_dir),
                "best_path": str(run_dir / "best.pt"),
                "labelmap_path": str(labelmap_path),
                "export_path": str(export_path) if export_path else None,
                "merged_yaml": str(merged_yaml),
            }
            _yolo_prune_run_dir_impl(
                run_dir,
                keep_files_default=YOLO_KEEP_FILES,
                dir_size_fn=_dir_size_bytes,
                meta_name=YOLO_RUN_META_NAME,
            )
            _yolo_head_graft_job_update(job, status="succeeded", message="Head graft complete", progress=1.0, result=result_payload)
            write_run_meta(
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "message": job.message,
                    "config": job.config,
                    "result": job.result,
                    "head_graft": {
                        "base_run_id": base_run_id,
                        "base_labelmap": base_labelmap,
                        "new_labelmap": new_labelmap,
                        "variant": base_variant,
                    },
                }
            )
            _yolo_head_graft_audit(job, "head_graft_complete", event="complete", extra={"result": result_payload})
        except Exception as exc:  # noqa: BLE001
            _yolo_head_graft_job_update(job, status="failed", message="Head merge failed", error=str(exc))
            write_run_meta({"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
        finally:
            _finalize_training_environment_impl(
                resume_classifier_fn=_resume_classifier_backbone,
                torch_module=torch,
            )

    thread = threading.Thread(target=worker, name=f"yolo-graft-{job.job_id}", daemon=True)
    thread.start()


def _start_rfdetr_training_worker(job: RfDetrTrainingJob) -> None:
    def worker() -> None:
        run_dir = _rfdetr_run_dir_impl(
            job.job_id,
            create=True,
            job_root=RFDETR_JOB_ROOT,
            sanitize_fn=_sanitize_rfdetr_run_id_impl,
            http_exception_cls=HTTPException,
        )
        config = dict(job.config or {})
        dataset_info = config.get("dataset") or {}
        if job.cancel_event.is_set():
            _rfdetr_job_update(job, status="cancelled", message="Cancelled before start", progress=0.0)
            _rfdetr_write_run_meta_impl(
                run_dir,
                {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config},
                meta_name=RFDETR_RUN_META_NAME,
                time_fn=time.time,
            )
            return
        try:
            _prepare_for_training_impl(
                unload_inference_runtimes_fn=lambda: _unload_inference_runtimes_impl(
                    unload_non_qwen_fn=lambda: _unload_non_qwen_runtimes_impl(
                        predictor_manager=predictor_manager,
                        unload_sam3_text_fn=_unload_sam3_text_runtime,
                        suspend_clip_fn=_suspend_clip_backbone,
                        unload_dinov3_fn=_unload_dinov3_backbone,
                        unload_detector_fn=_unload_detector_inference,
                        torch_module=torch,
                        logger=logger,
                    ),
                    unload_qwen_fn=_unload_qwen_runtime,
                    torch_module=torch,
                )
            )
            from rfdetr import (
                RFDETRBase,
                RFDETRLarge,
                RFDETRNano,
                RFDETRSmall,
                RFDETRMedium,
                RFDETRSegPreview,
            )
        except Exception as exc:  # noqa: BLE001
            _rfdetr_job_update(job, status="failed", message="RF-DETR not installed", error=str(exc))
            _rfdetr_write_run_meta_impl(
                run_dir,
                {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config},
                meta_name=RFDETR_RUN_META_NAME,
                time_fn=time.time,
            )
            return
        try:
            task = str(dataset_info.get("task") or config.get("task") or "detect").lower()
            variant_info = _rfdetr_variant_info_impl(
                task,
                config.get("variant"),
                variants=RFDETR_VARIANTS,
                http_exception_cls=HTTPException,
            )
            variant_id = variant_info.get("id")
            variant_label = variant_info.get("label")
            model_cls_map = {
                "rfdetr-nano": RFDETRNano,
                "rfdetr-small": RFDETRSmall,
                "rfdetr-medium": RFDETRMedium,
                "rfdetr-base": RFDETRBase,
                "rfdetr-large": RFDETRLarge,
                "rfdetr-seg-preview": RFDETRSegPreview,
            }
            model_cls = model_cls_map.get(variant_id)
            if not model_cls:
                raise RuntimeError("rfdetr_variant_unknown")
            dataset_root = Path(dataset_info.get("dataset_root") or "")
            coco_train = dataset_info.get("coco_train_json")
            coco_val = dataset_info.get("coco_val_json") or coco_train
            if not coco_train or not coco_val:
                raise RuntimeError("rfdetr_coco_missing")
            labelmap = _rfdetr_load_labelmap_impl(
                dataset_root,
                coco_train,
                yolo_load_labelmap_fn=_yolo_load_labelmap_impl,
                json_load_fn=json.loads,
            )
            if labelmap:
                (run_dir / "labelmap.txt").write_text("\n".join(labelmap) + "\n")
            model_kwargs: Dict[str, Any] = {}
            if config.get("resolution"):
                try:
                    model_kwargs["resolution"] = int(config.get("resolution"))
                except Exception:
                    pass
            model_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            if config.get("from_scratch"):
                model_kwargs["pretrain_weights"] = None
            elif config.get("pretrain_weights"):
                model_kwargs["pretrain_weights"] = config.get("pretrain_weights")
            if task == "segment":
                model_kwargs["segmentation_head"] = True
            _rfdetr_job_update(job, status="running", message=f"Starting RF-DETR training ({variant_label})", progress=0.0)
            _rfdetr_job_log(job, "Preparing dataset + COCO annotations")
            prepared_root = _rfdetr_prepare_dataset_impl(
                dataset_root,
                run_dir,
                coco_train,
                coco_val,
                remap_ids_fn=_rfdetr_remap_coco_ids_impl,
            )
            _rfdetr_job_log(job, f"RF-DETR dataset prepared at {prepared_root}")
            total_epochs = max(1, int(config.get("epochs") or 100))
            train_kwargs: Dict[str, Any] = {
                "dataset_dir": str(prepared_root),
                "dataset_file": "roboflow",
                "output_dir": str(run_dir),
                "epochs": total_epochs,
                "batch_size": config.get("batch"),
                "grad_accum_steps": config.get("grad_accum"),
                "num_workers": config.get("workers"),
                "seed": config.get("seed"),
                "use_ema": config.get("use_ema"),
                "early_stopping": config.get("early_stopping"),
                "early_stopping_patience": config.get("early_stopping_patience"),
                "run": config.get("run_name") or job.job_id[:8],
                "project": "rfdetr",
            }
            aug_policy = _rfdetr_normalize_aug_policy_impl(config.get("augmentations"))
            multi_scale = config.get("multi_scale")
            if multi_scale is not None:
                train_kwargs["multi_scale"] = bool(multi_scale)
                train_kwargs["expanded_scales"] = bool(config.get("expanded_scales")) if multi_scale else False
            if task == "segment":
                train_kwargs["segmentation_head"] = True
            train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
            device_ids = _normalize_device_list(config.get("devices"))
            if not device_ids and torch.cuda.is_available():
                device_ids = list(range(torch.cuda.device_count()))
            if device_ids:
                _validate_cuda_device_ids_impl(
                    device_ids,
                    torch_module=torch,
                    http_exception_cls=HTTPException,
                )
            cuda_visible = ",".join(str(d) for d in device_ids) if device_ids else None
            prev_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
            train_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            use_distributed = torch.cuda.is_available() and len(device_ids) > 1
            _rfdetr_job_log(job, f"Model variant: {variant_id}")
            if use_distributed:
                dist_url = f"tcp://127.0.0.1:{_find_free_port_impl()}"
                world_size = len(device_ids)
                _rfdetr_job_log(job, f"Multi-GPU enabled: devices={cuda_visible} world_size={world_size}")
                _rfdetr_job_log(job, f"Training started (epochs={total_epochs})")
                monitor_stop = threading.Event()
                monitor_thread = threading.Thread(
                    target=_rfdetr_monitor_training_impl,
                    args=(job, run_dir, total_epochs, monitor_stop),
                    kwargs={
                        "job_append_metric_fn": _rfdetr_job_append_metric,
                        "job_update_fn": _rfdetr_job_update,
                        "sanitize_metric_fn": _rfdetr_sanitize_metric_impl,
                        "latest_checkpoint_fn": _rfdetr_latest_checkpoint_epoch_impl,
                    },
                    name=f"rfdetr-monitor-{job.job_id[:8]}",
                    daemon=True,
                )
                monitor_thread.start()
                prev_skip_clip = os.environ.get("TATOR_SKIP_CLIP_LOAD")
                os.environ["TATOR_SKIP_CLIP_LOAD"] = "1"
                import torch.multiprocessing as mp
                try:
                    mp.spawn(
                        _rfdetr_ddp_worker,
                        args=(world_size, variant_id, model_kwargs, train_kwargs, aug_policy, dist_url),
                        nprocs=world_size,
                        join=True,
                    )
                finally:
                    monitor_stop.set()
                    monitor_thread.join(timeout=2.0)
                    if prev_skip_clip is None:
                        os.environ.pop("TATOR_SKIP_CLIP_LOAD", None)
                    else:
                        os.environ["TATOR_SKIP_CLIP_LOAD"] = prev_skip_clip
            else:
                rf_detr = model_cls(**model_kwargs)
                restore = _rfdetr_install_augmentations_impl(aug_policy)

                def on_fit_epoch_end(stats: Dict[str, Any]) -> None:
                    metric = _rfdetr_sanitize_metric_impl(stats)
                    if metric:
                        _rfdetr_job_append_metric(job, metric)
                    epoch = metric.get("epoch") if metric else None
                    if epoch is not None:
                        try:
                            epoch_idx = int(epoch)
                            progress = max(0.0, min(0.99, epoch_idx / total_epochs))
                            _rfdetr_job_update(job, progress=progress, message=f"Epoch {epoch_idx}/{total_epochs}")
                        except Exception:
                            pass
                    if job.cancel_event.is_set():
                        try:
                            rf_detr.model.request_early_stop()
                        except Exception:
                            pass

                rf_detr.callbacks["on_fit_epoch_end"].append(on_fit_epoch_end)
                _rfdetr_job_log(job, f"Training started (epochs={total_epochs})")
                try:
                    rf_detr.train(**train_kwargs)
                finally:
                    _rfdetr_restore_augmentations_impl(restore)
            if job.cancel_event.is_set():
                _rfdetr_job_update(job, status="cancelled", message="Training cancelled", progress=job.progress)
            else:
                _rfdetr_job_update(job, status="succeeded", message="Training complete", progress=1.0)
            metrics_series = job.metrics or []
            if not metrics_series:
                metrics_series = _rfdetr_parse_log_series_impl(run_dir / "log.txt")
                if metrics_series:
                    job.metrics = metrics_series
            if metrics_series:
                try:
                    (run_dir / "metrics_series.json").write_text(json.dumps(metrics_series, indent=2, sort_keys=True))
                except Exception:
                    pass
            best_path = _rfdetr_best_checkpoint_impl(run_dir)
            optimized_path = None
            if best_path:
                try:
                    export_kwargs = dict(model_kwargs)
                    export_kwargs["pretrain_weights"] = best_path
                    export_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
                    export_model = model_cls(**export_kwargs)
                    export_model.optimize_for_inference()
                    optimized_path = run_dir / "checkpoint_best_optimized.pt"
                    torch.jit.save(export_model.model.inference_model, str(optimized_path))
                    _rfdetr_job_log(job, f"Optimized export saved: {optimized_path.name}")
                except Exception as exc:  # noqa: BLE001
                    _rfdetr_job_log(job, f"Optimized export failed: {exc}")
            result_payload = {
                "run_dir": str(run_dir),
                "best_path": best_path,
                "optimized_path": str(optimized_path) if optimized_path else None,
                "results_path": str(run_dir / "results.json") if (run_dir / "results.json").exists() else None,
                "metrics_series_path": str(run_dir / "metrics_series.json") if (run_dir / "metrics_series.json").exists() else None,
                "log_path": str(run_dir / "log.txt") if (run_dir / "log.txt").exists() else None,
            }
            _rfdetr_prune_run_dir_impl(
                run_dir,
                keep_files_default=RFDETR_KEEP_FILES,
                dir_size_fn=_dir_size_bytes,
            )
            job.result = result_payload
        except Exception as exc:  # noqa: BLE001
            _rfdetr_job_update(job, status="failed", message="Training failed", error=str(exc))
        finally:
            if "prev_cuda_visible" in locals():
                if prev_cuda_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible
            _finalize_training_environment_impl(
                resume_classifier_fn=_resume_classifier_backbone,
                torch_module=torch,
            )
            _rfdetr_write_run_meta_impl(
                run_dir,
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "message": job.message,
                    "config": job.config,
                    "result": job.result,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                },
                meta_name=RFDETR_RUN_META_NAME,
                time_fn=time.time,
            )

    thread = threading.Thread(target=worker, name=f"rfdetr-train-{job.job_id}", daemon=True)
    thread.start()


def _get_sam3_job(job_id: str) -> Sam3TrainingJob:
    with SAM3_TRAINING_JOBS_LOCK:
        job = SAM3_TRAINING_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="sam3_job_not_found")
        return job


def _get_yolo_job(job_id: str) -> YoloTrainingJob:
    with YOLO_TRAINING_JOBS_LOCK:
        job = YOLO_TRAINING_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_job_not_found")
        return job


def _get_rfdetr_job(job_id: str) -> RfDetrTrainingJob:
    with RFDETR_TRAINING_JOBS_LOCK:
        job = RFDETR_TRAINING_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="rfdetr_job_not_found")
        return job


def _get_qwen_job(job_id: str) -> QwenTrainingJob:
    with QWEN_TRAINING_JOBS_LOCK:
        job = QWEN_TRAINING_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_job_not_found")
        return job


async def _save_upload_file(
    upload: UploadFile,
    root: Path,
    *,
    max_bytes: Optional[int] = None,
    quota_root: Optional[Path] = None,
    quota_limit: Optional[int] = None,
) -> Path:
    rel_path = _normalise_relative_path(upload.filename)
    dest = (root / rel_path).resolve()
    if not str(dest).startswith(str(root.resolve())):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_relative_path")
    if dest.exists():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="upload_exists")
    await _write_upload_file(
        upload,
        dest,
        max_bytes=max_bytes,
        quota_root=quota_root or root,
        quota_limit=quota_limit,
    )
    return dest


def _validate_upload_size(upload: UploadFile, *, max_bytes: int = BASE64_IMAGE_MAX_BYTES) -> None:
    if not max_bytes:
        return
    try:
        size = upload.size  # Starlette UploadFile may have size attr
    except Exception:
        size = None
    if size is not None and size > max_bytes:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="upload_too_large")


def _validate_upload_extension(filename: str, allowed_exts: set[str], detail: str) -> None:
    suffix = Path(filename).suffix.lower()
    if allowed_exts and suffix not in allowed_exts:
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=detail)


async def _save_asset(
    upload: UploadFile,
    *,
    subdir: str,
    allowed_exts: Optional[set[str]] = None,
    max_bytes: Optional[int] = None,
    quota_bytes: Optional[int] = None,
) -> str:
    dest_dir = UPLOAD_ROOT / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    rel_name = Path(upload.filename or f"asset_{uuid.uuid4().hex}").name
    dest_path = dest_dir / rel_name
    if allowed_exts:
        _validate_upload_extension(rel_name, allowed_exts, "upload_extension_not_allowed")
    if dest_path.exists():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="upload_exists")
    _validate_upload_size(upload, max_bytes=max_bytes or ASSET_MAX_BYTES)
    await _write_upload_file(
        upload,
        dest_path,
        max_bytes=max_bytes or ASSET_MAX_BYTES,
        quota_root=dest_dir,
        quota_limit=quota_bytes or ASSET_UPLOAD_QUOTA_BYTES,
    )
    return str(dest_path.resolve())


async def _write_upload_file(
    upload: UploadFile,
    dest: Path,
    *,
    max_bytes: Optional[int] = None,
    quota_root: Optional[Path] = None,
    quota_limit: Optional[int] = None,
    allow_overwrite: bool = False,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not allow_overwrite:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="upload_exists")
    written = 0
    existing = _dir_size_bytes_impl(quota_root) if quota_root and quota_limit else 0
    with dest.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if max_bytes and written > max_bytes:
                handle.close()
                try:
                    dest.unlink()
                except Exception:
                    pass
                raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="upload_too_large")
            if quota_root and quota_limit and existing + written > quota_limit:
                handle.close()
                try:
                    dest.unlink()
                except Exception:
                    pass
                raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="upload_quota_exceeded")
            handle.write(chunk)
    await upload.close()


def _artifacts_to_payload(artifacts: TrainingArtifacts) -> Dict[str, Any]:
    data = asdict(artifacts)
    return data


def _cleanup_job(job: ClipTrainingJob) -> None:
    if job.temp_dir and os.path.isdir(job.temp_dir):
        shutil.rmtree(job.temp_dir, ignore_errors=True)


def _publish_clip_training_artifacts(artifacts: TrainingArtifacts) -> TrainingArtifacts:
    classifiers_root = (UPLOAD_ROOT / "classifiers").resolve()
    labelmaps_root = (UPLOAD_ROOT / "labelmaps").resolve()
    classifiers_root.mkdir(parents=True, exist_ok=True)
    labelmaps_root.mkdir(parents=True, exist_ok=True)

    model_src = Path(artifacts.model_path).resolve()
    labelmap_src = Path(artifacts.labelmap_path).resolve()
    meta_src = Path(artifacts.meta_path).resolve()

    model_dst = classifiers_root / model_src.name
    labelmap_dst = labelmaps_root / labelmap_src.name
    meta_dst = classifiers_root / meta_src.name

    try:
        if model_src.exists():
            _link_or_copy_file(model_src, model_dst, overwrite=True)
            artifacts.model_path = str(model_dst)
    except Exception as exc:
        logger.warning("Failed to publish CLIP classifier %s: %s", model_src, exc)
    try:
        if meta_src.exists():
            _link_or_copy_file(meta_src, meta_dst, overwrite=True)
            artifacts.meta_path = str(meta_dst)
    except Exception as exc:
        logger.warning("Failed to publish CLIP meta %s: %s", meta_src, exc)
    try:
        if labelmap_src.exists():
            _link_or_copy_file(labelmap_src, labelmap_dst, overwrite=True)
            artifacts.labelmap_path = str(labelmap_dst)
    except Exception as exc:
        logger.warning("Failed to publish CLIP labelmap %s: %s", labelmap_src, exc)

    return artifacts


def _current_active_payload() -> Dict[str, Any]:
    encoder_ready = _active_encoder_ready()
    encoder_error = clip_last_error
    return {
        "clip_model": clip_model_name,
        "classifier_path": active_classifier_path,
        "labelmap_path": active_labelmap_path,
        "clip_ready": encoder_ready,
        "clip_error": clip_last_error,
        "labelmap_entries": list(active_label_list),
        "encoder_type": active_encoder_type,
        "encoder_model": active_encoder_model,
        "encoder_ready": encoder_ready,
        "encoder_error": encoder_error,
        "logit_adjustment_inference": (
            bool(active_classifier_head.get("logit_adjustment_inference"))
            if isinstance(active_classifier_head, dict)
            else None
        ),
    }


def _active_encoder_ready() -> bool:
    if clf is None:
        return False
    if str(active_encoder_type or "").strip().lower() == "dinov3":
        return bool(dinov3_initialized and dinov3_model is not None and dinov3_processor is not None)
    return bool(clip_initialized and clip_model is not None and clip_preprocess is not None)


def predict_base64(payload: Base64Payload):
    # If CLIP/logreg not loaded, return error message in "prediction"
    if not _active_encoder_ready():
        return PredictResponse(prediction=str(ERROR_MESSAGE), uuid=None, error="clip_unavailable") # messy ... returning the error message int as str. Crap logic needs cleanup

    pil_img, _np_img, _token = _resolve_detector_image_impl(
        payload.image_base64,
        payload.image_token,
        fetch_preloaded_fn=_fetch_preloaded_image,
        decode_image_fn=lambda b64: _decode_image_base64_impl(
            b64,
            max_bytes=BASE64_IMAGE_MAX_BYTES,
            max_dim=BASE64_IMAGE_MAX_DIM,
            allow_downscale=True,
        ),
        store_preloaded_fn=_store_preloaded_image,
        hash_fn=lambda payload: hashlib.md5(payload).hexdigest(),
    )
    feats_np = _encode_pil_batch_for_active([pil_img])
    if feats_np is None or not isinstance(feats_np, np.ndarray) or feats_np.size == 0:
        return PredictResponse(prediction=str(ERROR_MESSAGE), uuid=None, error="clip_unavailable")
    bg_guard = bool(payload.background_guard) if payload.background_guard is not None else False
    details = _clip_auto_predict_details(feats_np, background_guard=bg_guard)
    return PredictResponse(
        prediction=str(details.get("label") or "unknown"),
        proba=details.get("proba"),
        second_label=details.get("second_label"),
        second_proba=details.get("second_proba"),
        margin=details.get("margin"),
        error=details.get("error"),
        uuid=payload.uuid,
    )


app.include_router(
    build_predict_base64_router(
        predict_fn=predict_base64,
        request_cls=Base64Payload,
        response_cls=PredictResponse,
    )
)


def list_clip_backbones():
    return {
        "available": SUPPORTED_CLIP_MODELS,
        "active": clip_model_name,
    }


def download_clip_classifier(rel_path: str = Query(...)):
    classifier_path = _resolve_agent_clip_classifier_path_impl(
        rel_path,
        allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
        allowed_exts=CLASSIFIER_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root_impl,
        http_exception_cls=HTTPException,
    )
    if classifier_path is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="classifier_not_found")
    stream = classifier_path.open("rb")
    headers = {"Content-Disposition": f'attachment; filename="{classifier_path.name}"'}
    return StreamingResponse(stream, media_type="application/octet-stream", headers=headers)


def download_clip_classifier_zip(rel_path: str = Query(...)):
    classifier_path = _resolve_agent_clip_classifier_path_impl(
        rel_path,
        allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
        allowed_exts=CLASSIFIER_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root_impl,
        http_exception_cls=HTTPException,
    )
    if classifier_path is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="classifier_not_found")
    buffer = io.BytesIO()
    meta_path = Path(os.path.splitext(str(classifier_path))[0] + ".meta.pkl")
    labelmap_path = _find_labelmap_for_classifier_impl(
        classifier_path,
        upload_root=UPLOAD_ROOT,
        labelmap_exts=LABELMAP_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root_impl,
        joblib_load_fn=joblib.load,
        resolve_clip_labelmap_path_fn=lambda path_str, root_hint=None: _resolve_clip_labelmap_path_impl(
            path_str,
            root_hint=root_hint,
            upload_root=UPLOAD_ROOT,
            labelmap_exts=LABELMAP_ALLOWED_EXTS,
            path_is_within_root_fn=_path_is_within_root_impl,
        ),
    )
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(classifier_path, arcname=classifier_path.name)
        if meta_path.exists():
            zf.write(meta_path, arcname=meta_path.name)
        if labelmap_path is not None:
            zf.write(labelmap_path, arcname=labelmap_path.name)
    buffer.seek(0)
    filename = f"{classifier_path.stem}_clip_head.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


def delete_clip_classifier(rel_path: str = Query(...)):
    classifier_path = _resolve_agent_clip_classifier_path_impl(
        rel_path,
        allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
        allowed_exts=CLASSIFIER_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root_impl,
        http_exception_cls=HTTPException,
    )
    if classifier_path is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="classifier_not_found")
    try:
        classifier_path.unlink()
    except FileNotFoundError:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="classifier_not_found")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    meta_path = Path(os.path.splitext(str(classifier_path))[0] + ".meta.pkl")
    try:
        if meta_path.exists():
            meta_path.unlink()
    except Exception:
        pass
    return {"status": "deleted", "rel_path": rel_path}


def rename_clip_classifier(
    rel_path: str = Form(...),
    new_name: str = Form(...),
):
    classifier_path = _resolve_agent_clip_classifier_path_impl(
        rel_path,
        allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
        allowed_exts=CLASSIFIER_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root_impl,
        http_exception_cls=HTTPException,
    )
    if classifier_path is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="classifier_not_found")
    raw = str(new_name or "").strip()
    if not raw:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_new_name_required")
    # Strip any directory components; only allow file names.
    raw_name = Path(raw).name
    if not raw_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_new_name_invalid")
    current_suffix = classifier_path.suffix
    target_name = raw_name
    if not Path(target_name).suffix:
        target_name = f"{target_name}{current_suffix}"
    target_suffix = Path(target_name).suffix.lower()
    if target_suffix not in CLASSIFIER_ALLOWED_EXTS:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_extension_not_allowed")

    classifiers_root = (UPLOAD_ROOT / "classifiers").resolve()
    parent = classifier_path.parent.resolve()
    if not _path_is_within_root_impl(parent, classifiers_root):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_path_invalid")
    target_path = (parent / target_name).resolve()
    if not _path_is_within_root_impl(target_path, classifiers_root):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_path_invalid")

    if target_path == classifier_path:
        return {"status": "unchanged", "rel_path": str(classifier_path.relative_to(classifiers_root))}

    if target_path.exists():
        stem = target_path.stem
        suffix = target_path.suffix
        for idx in range(1, 1000):
            candidate = (parent / f"{stem}_{idx}{suffix}").resolve()
            if not candidate.exists():
                target_path = candidate
                break
        else:
            raise HTTPException(status_code=HTTP_409_CONFLICT, detail="classifier_rename_conflict")

    try:
        classifier_path.rename(target_path)
    except FileNotFoundError:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="classifier_not_found")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    old_meta = Path(os.path.splitext(str(classifier_path))[0] + ".meta.pkl")
    new_meta = Path(os.path.splitext(str(target_path))[0] + ".meta.pkl")
    try:
        if old_meta.exists():
            if new_meta.exists():
                try:
                    new_meta.unlink()
                except Exception:
                    pass
            old_meta.replace(new_meta)
    except Exception:
        pass

    try:
        global active_classifier_path
        if active_classifier_path and Path(active_classifier_path).resolve() == classifier_path.resolve():
            active_classifier_path = str(target_path)
    except Exception:
        pass

    return {
        "status": "renamed",
        "old_rel_path": str(classifier_path.relative_to(classifiers_root)),
        "new_rel_path": str(target_path.relative_to(classifiers_root)),
        "old_path": str(classifier_path),
        "new_path": str(target_path),
        "new_name": target_path.name,
    }


def download_clip_labelmap(rel_path: str = Query(...), root: Optional[str] = Query(None)):
    labelmap_path = _resolve_clip_labelmap_path_impl(
        rel_path,
        root_hint=root,
        upload_root=UPLOAD_ROOT,
        labelmap_exts=LABELMAP_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root_impl,
    )
    if labelmap_path is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="labelmap_not_found")
    stream = labelmap_path.open("rb")
    headers = {"Content-Disposition": f'attachment; filename="{labelmap_path.name}"'}
    return StreamingResponse(stream, media_type="application/octet-stream", headers=headers)


def delete_clip_labelmap(rel_path: str = Query(...), root: Optional[str] = Query(None)):
    labelmap_path = _resolve_clip_labelmap_path_impl(
        rel_path,
        root_hint=root,
        upload_root=UPLOAD_ROOT,
        labelmap_exts=LABELMAP_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root_impl,
    )
    if labelmap_path is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="labelmap_not_found")
    try:
        labelmap_path.unlink()
    except FileNotFoundError:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="labelmap_not_found")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return {"status": "deleted", "rel_path": rel_path}


app.include_router(
    build_clip_registry_router(
        list_backbones_fn=list_clip_backbones,
        list_classifiers_fn=lambda: _list_clip_classifiers_impl(
            upload_root=UPLOAD_ROOT,
            classifier_exts=CLASSIFIER_ALLOWED_EXTS,
            labelmap_exts=LABELMAP_ALLOWED_EXTS,
            path_is_within_root_fn=_path_is_within_root_impl,
            joblib_load_fn=joblib.load,
            resolve_clip_labelmap_path_fn=lambda path_str, root_hint=None: _resolve_clip_labelmap_path_impl(
                path_str,
                root_hint=root_hint,
                upload_root=UPLOAD_ROOT,
                labelmap_exts=LABELMAP_ALLOWED_EXTS,
                path_is_within_root_fn=_path_is_within_root_impl,
            ),
        ),
        list_labelmaps_fn=lambda: _list_clip_labelmaps_impl(
            upload_root=UPLOAD_ROOT,
            labelmap_exts=LABELMAP_ALLOWED_EXTS,
            load_labelmap_file_fn=_load_labelmap_file,
        ),
        download_classifier_fn=download_clip_classifier,
        download_classifier_zip_fn=download_clip_classifier_zip,
        delete_classifier_fn=delete_clip_classifier,
        rename_classifier_fn=rename_clip_classifier,
        download_labelmap_fn=download_clip_labelmap,
        delete_labelmap_fn=delete_clip_labelmap,
    )
)


async def upload_classifier(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="filename_required")
    saved_path = await _save_asset(
        file,
        subdir="classifiers",
        allowed_exts=CLASSIFIER_ALLOWED_EXTS,
        max_bytes=ASSET_MAX_BYTES,
        quota_bytes=ASSET_UPLOAD_QUOTA_BYTES,
    )
    return {"path": saved_path}


async def upload_labelmap(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="filename_required")
    saved_path = await _save_asset(
        file,
        subdir="labelmaps",
        allowed_exts=LABELMAP_ALLOWED_EXTS,
        max_bytes=ASSET_MAX_BYTES,
        quota_bytes=ASSET_UPLOAD_QUOTA_BYTES,
    )
    return {"path": saved_path}


app.include_router(
    build_fs_upload_router(
        upload_classifier_fn=upload_classifier,
        upload_labelmap_fn=upload_labelmap,
    )
)


def _validate_job_exists(job_id: str) -> ClipTrainingJob:
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="job_not_found")
        return job


def _start_training_worker(job: ClipTrainingJob, *, images_dir: str, labels_dir: str, labelmap_path: Optional[str],
                           clip_name: str, encoder_type: str, encoder_model: Optional[str],
                           output_dir: str, labelmap_dir: str, model_filename: str, labelmap_filename: str,
                           test_size: float, random_seed: int, batch_size: int, max_iter: int,
                           min_per_class: int, class_weight: str, effective_beta: float, C: float, device_override: Optional[str],
                           solver: str, classifier_type: str, mlp_hidden_sizes: str, mlp_dropout: float,
                           mlp_epochs: int, mlp_lr: float, mlp_weight_decay: float, mlp_label_smoothing: float,
                           mlp_loss_type: str, mlp_focal_gamma: float, mlp_focal_alpha: Optional[float],
                           mlp_sampler: str, mlp_mixup_alpha: float, mlp_normalize_embeddings: bool,
                           mlp_patience: int, mlp_activation: str, mlp_layer_norm: bool, mlp_hard_mining_epochs: int,
                           logit_adjustment_mode: str, logit_adjustment_inference: Optional[bool],
                           arcface_enabled: bool, arcface_margin: float, arcface_scale: float,
                           supcon_weight: float, supcon_temperature: float, supcon_projection_dim: int, supcon_projection_hidden: int,
                           embedding_center: bool, embedding_standardize: bool,
                           calibration_mode: str, calibration_max_iters: int, calibration_min_temp: float, calibration_max_temp: float,
                           reuse_embeddings: bool, hard_example_mining: bool,
                           hard_mining_misclassified_weight: float,
                           hard_mining_low_conf_weight: float,
                           hard_mining_low_conf_threshold: float,
                           hard_mining_margin_threshold: float,
                           convergence_tol: float,
                           bg_class_count: int,
                           cancel_event: threading.Event) -> None:

    def progress_cb(value: float, message: str) -> None:
        with TRAINING_JOBS_LOCK:
            if cancel_event.is_set() and job.status not in {"cancelled", "failed"}:
                _job_update(job, status="cancelling", message="Cancellation requested ...", progress=value)
                return
            _job_update(job, status="running", progress=value, message=message)

    def metrics_cb(metric: Dict[str, Any]) -> None:
        if not metric:
            return
        with TRAINING_JOBS_LOCK:
            _clip_job_append_metric(job, metric)

    def worker() -> None:
        try:
            _prepare_for_training_impl(
                unload_inference_runtimes_fn=lambda: _unload_inference_runtimes_impl(
                    unload_non_qwen_fn=lambda: _unload_non_qwen_runtimes_impl(
                        predictor_manager=predictor_manager,
                        unload_sam3_text_fn=_unload_sam3_text_runtime,
                        suspend_clip_fn=_suspend_clip_backbone,
                        unload_dinov3_fn=_unload_dinov3_backbone,
                        unload_detector_fn=_unload_detector_inference,
                        torch_module=torch,
                        logger=logger,
                    ),
                    unload_qwen_fn=_unload_qwen_runtime,
                    torch_module=torch,
                )
            )
            with TRAINING_JOBS_LOCK:
                if cancel_event.is_set():
                    _job_update(job, status="cancelled", progress=job.progress, message="Training cancelled before start.")
                    return
                _job_update(job, status="running", progress=0.01, message="Preparing training job ...")
            artifacts = train_clip_from_yolo(
                images_path=images_dir,
                labels_path=labels_dir,
                model_output=os.path.join(output_dir, model_filename),
                labelmap_output=os.path.join(labelmap_dir, labelmap_filename),
                clip_model=clip_name,
                encoder_type=encoder_type,
                encoder_model=encoder_model,
                input_labelmap=labelmap_path,
                test_size=test_size,
                random_seed=random_seed,
                batch_size=batch_size,
                max_iter=max_iter,
                min_per_class=min_per_class,
                class_weight=class_weight,
                effective_beta=effective_beta,
                C=C,
                solver=solver,
                classifier_type=classifier_type,
                mlp_hidden_sizes=mlp_hidden_sizes,
                mlp_dropout=mlp_dropout,
                mlp_epochs=mlp_epochs,
                mlp_lr=mlp_lr,
                mlp_weight_decay=mlp_weight_decay,
                mlp_label_smoothing=mlp_label_smoothing,
                mlp_loss_type=mlp_loss_type,
                mlp_focal_gamma=mlp_focal_gamma,
                mlp_focal_alpha=mlp_focal_alpha,
                mlp_sampler=mlp_sampler,
                mlp_mixup_alpha=mlp_mixup_alpha,
                mlp_normalize_embeddings=mlp_normalize_embeddings,
                mlp_patience=mlp_patience,
                mlp_activation=mlp_activation,
                mlp_layer_norm=mlp_layer_norm,
                mlp_hard_mining_epochs=mlp_hard_mining_epochs,
                logit_adjustment_mode=logit_adjustment_mode,
                logit_adjustment_inference=logit_adjustment_inference,
                arcface_enabled=arcface_enabled,
                arcface_margin=arcface_margin,
                arcface_scale=arcface_scale,
                supcon_weight=supcon_weight,
                supcon_temperature=supcon_temperature,
                supcon_projection_dim=supcon_projection_dim,
                supcon_projection_hidden=supcon_projection_hidden,
                embedding_center=embedding_center,
                embedding_standardize=embedding_standardize,
                calibration_mode=calibration_mode,
                calibration_max_iters=calibration_max_iters,
                calibration_min_temp=calibration_min_temp,
                calibration_max_temp=calibration_max_temp,
                reuse_embeddings=reuse_embeddings,
                hard_example_mining=hard_example_mining,
                hard_mining_misclassified_weight=hard_mining_misclassified_weight,
                hard_mining_low_conf_weight=hard_mining_low_conf_weight,
                hard_mining_low_conf_threshold=hard_mining_low_conf_threshold,
                hard_mining_margin_threshold=hard_mining_margin_threshold,
                convergence_tol=convergence_tol,
                bg_class_count=bg_class_count,
                device=device_override,
                progress_cb=progress_cb,
                metrics_cb=metrics_cb,
                should_cancel=cancel_event.is_set,
            )
            artifacts = _publish_clip_training_artifacts(artifacts)
            payload = _artifacts_to_payload(artifacts)
            with TRAINING_JOBS_LOCK:
                _job_update(job, status="succeeded", progress=1.0, message="Training completed.", artifacts=payload)
        except TrainingError as exc:
            with TRAINING_JOBS_LOCK:
                if str(exc) == "cancelled":
                    _job_update(job, status="cancelled", message="Training cancelled by user.")
                    logger.info("[clip-train %s] Training cancelled", job.job_id[:8])
                else:
                    _job_update(job, status="failed", message=str(exc), error=str(exc))
                    logger.warning("[clip-train %s] Training failed: %s", job.job_id[:8], exc)
        except Exception as exc:  # noqa: BLE001
            with TRAINING_JOBS_LOCK:
                _job_update(job, status="failed", message="Training crashed.", error=str(exc))
            logger.exception("[clip-train %s] Training crashed", job.job_id[:8])
        finally:
            _finalize_training_environment_impl(
                resume_classifier_fn=_resume_classifier_backbone,
                torch_module=torch,
            )
            _cleanup_job(job)

    threading.Thread(target=worker, name=f"clip-train-{job.job_id[:8]}", daemon=True).start()


async def start_clip_training(
    images: Optional[List[UploadFile]] = File(None),
    labels: Optional[List[UploadFile]] = File(None),
    labelmap: Optional[UploadFile] = File(None),
    clip_model_name: str = Form(DEFAULT_CLIP_MODEL),
    encoder_type: str = Form("clip"),
    encoder_model: Optional[str] = Form(None),
    output_dir: str = Form("."),
    model_filename: str = Form("my_logreg_model.pkl"),
    labelmap_filename: str = Form("my_label_list.pkl"),
    test_size: float = Form(0.2),
    random_seed: int = Form(42),
    batch_size: int = Form(64),
    max_iter: int = Form(1000),
    min_per_class: int = Form(2),
    class_weight: str = Form("balanced"),
    effective_beta: float = Form(0.9999),
    C: float = Form(1.0),
    device_override: Optional[str] = Form(None),
    images_path_native: Optional[str] = Form(None),
    labels_path_native: Optional[str] = Form(None),
    labelmap_path_native: Optional[str] = Form(None),
    solver: str = Form("saga"),
    classifier_type: str = Form("logreg"),
    mlp_hidden_sizes: str = Form("256"),
    mlp_dropout: float = Form(0.1),
    mlp_epochs: int = Form(50),
    mlp_lr: float = Form(1e-3),
    mlp_weight_decay: float = Form(1e-4),
    mlp_label_smoothing: float = Form(0.05),
    mlp_loss_type: str = Form("ce"),
    mlp_focal_gamma: float = Form(2.0),
    mlp_focal_alpha: float = Form(-1.0),
    mlp_sampler: str = Form("balanced"),
    mlp_mixup_alpha: float = Form(0.1),
    mlp_normalize_embeddings: Optional[str] = Form("true"),
    mlp_patience: int = Form(6),
    mlp_activation: str = Form("relu"),
    mlp_layer_norm: Optional[str] = Form("false"),
    mlp_hard_mining_epochs: int = Form(5),
    logit_adjustment_mode: str = Form("none"),
    logit_adjustment_inference: Optional[str] = Form(None),
    arcface_enabled: Optional[str] = Form("false"),
    arcface_margin: float = Form(0.2),
    arcface_scale: float = Form(30.0),
    supcon_weight: float = Form(0.0),
    supcon_temperature: float = Form(0.07),
    supcon_projection_dim: int = Form(128),
    supcon_projection_hidden: int = Form(0),
    embedding_center: Optional[str] = Form("false"),
    embedding_standardize: Optional[str] = Form("false"),
    calibration_mode: str = Form("none"),
    calibration_max_iters: int = Form(50),
    calibration_min_temp: float = Form(0.5),
    calibration_max_temp: float = Form(5.0),
    reuse_embeddings: Optional[str] = Form(None),
    hard_example_mining: Optional[str] = Form(None),
    hard_mis_weight: float = Form(3.0),
    hard_low_conf_weight: float = Form(2.0),
    hard_low_conf_threshold: float = Form(0.65),
    hard_margin_threshold: float = Form(0.15),
    convergence_tol: float = Form(1e-4),
    bg_class_count: int = Form(2),
    staged_temp_dir: Optional[str] = Form(None),
):
    images_path_native = _normalise_optional_path(images_path_native)
    labels_path_native = _normalise_optional_path(labels_path_native)
    labelmap_path_native = _normalise_optional_path(labelmap_path_native)

    encoder_type_norm = (encoder_type or "clip").strip().lower()
    if encoder_type_norm not in {"clip", "dinov3"}:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="clip_encoder_type_unsupported")
    encoder_model_name = (encoder_model or "").strip()
    if encoder_type_norm == "clip":
        if encoder_model_name:
            clip_model_name = encoder_model_name
        if clip_model_name not in SUPPORTED_CLIP_MODELS:
            SUPPORTED_CLIP_MODELS.append(clip_model_name)
    else:
        if not encoder_model_name:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="encoder_model_required")

    solver_name = (solver or "saga").strip().lower()
    if solver_name not in {"saga", "sag", "lbfgs", "liblinear", "newton-cg"}:
        solver_name = "saga"
    classifier_type_norm = (classifier_type or "logreg").strip().lower()
    if classifier_type_norm not in {"logreg", "mlp"}:
        classifier_type_norm = "logreg"
    reuse_embeddings_flag = _parse_bool(reuse_embeddings)
    hard_example_flag = _parse_bool(hard_example_mining)

    use_native_paths = bool(images_path_native and labels_path_native)
    if use_native_paths and (images or labels):
        logger.info("Ignoring uploaded files; using native dataset paths provided.")
    if reuse_embeddings_flag and not use_native_paths:
        logger.info("Embedding cache reuse requested but dataset is staged upload; disabling reuse for job %s", images_path_native or "<staged>")
        reuse_embeddings_flag = False

    if not use_native_paths:
        if not images:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="images_required")
        if not labels:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labels_required")

    classifiers_dir = _ensure_directory(str((UPLOAD_ROOT / "classifiers").resolve()))
    labelmaps_dir = _ensure_directory(str((UPLOAD_ROOT / "labelmaps").resolve()))
    if output_dir and output_dir not in {".", classifiers_dir}:
        logger.info("Ignoring CLIP output_dir=%s; saving under %s", output_dir, classifiers_dir)

    temp_root: Optional[str] = None
    images_dir: Optional[str] = None
    labels_dir: Optional[str] = None

    if use_native_paths:
        images_dir = _ensure_directory(images_path_native)
        labels_dir = _ensure_directory(labels_path_native)
    else:
        temp_root = tempfile.mkdtemp(prefix="clip_train_")
        images_dir = os.path.join(temp_root, "images")
        labels_dir = os.path.join(temp_root, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for upload in images or []:
            await _save_upload_file(
                upload,
                Path(images_dir),
                max_bytes=CLIP_TRAIN_UPLOAD_MAX_BYTES,
                quota_root=Path(temp_root),
                quota_limit=CLIP_TRAIN_UPLOAD_QUOTA_BYTES,
            )

        for upload in labels or []:
            await _save_upload_file(
                upload,
                Path(labels_dir),
                max_bytes=CLIP_TRAIN_UPLOAD_MAX_BYTES,
                quota_root=Path(temp_root),
                quota_limit=CLIP_TRAIN_UPLOAD_QUOTA_BYTES,
            )

    labelmap_path = None
    if labelmap_path_native:
        labelmap_path = os.path.abspath(labelmap_path_native)
        if not os.path.isfile(labelmap_path):
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="labelmap_not_found")
    elif labelmap is not None:
        if temp_root is None:
            temp_root = tempfile.mkdtemp(prefix="clip_train_")
        labelmap_path = str(
            await _save_upload_file(
                labelmap,
                Path(temp_root),
                max_bytes=ASSET_MAX_BYTES,
                quota_root=Path(temp_root),
                quota_limit=CLIP_TRAIN_UPLOAD_QUOTA_BYTES,
            )
        )

    job_id = uuid.uuid4().hex
    if images_dir is None or labels_dir is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_paths_unresolved")
    # Fail fast on obviously invalid staged datasets.
    _validate_clip_dataset_impl(
        {"images_dir": images_dir, "labels_dir": labels_dir, "labelmap_path": labelmap_path},
        http_exception_cls=HTTPException,
        load_labelmap_simple_fn=lambda path: _load_labelmap_simple_impl(
            path,
            load_labelmap_file_fn=_load_labelmap_file,
        ),
    )
    logger.info(
        "Starting training job %s (encoder=%s, model=%s, native_paths=%s)",
        job_id[:8],
        encoder_type_norm,
        clip_model_name,
        use_native_paths,
    )
    if staged_temp_dir:
        temp_root = os.path.abspath(staged_temp_dir)
    job = ClipTrainingJob(job_id=job_id, temp_dir=temp_root, images_dir=images_dir, labels_dir=labels_dir, labelmap_path=labelmap_path)
    job_message = "Job queued (native paths)" if use_native_paths else "Job queued (upload staging)"
    test_size_f = _coerce_float(test_size, 0.2, minimum=0.0, maximum=0.9)
    random_seed_i = _coerce_int(random_seed, 42)
    batch_size_i = _coerce_int(batch_size, 64, minimum=1)
    max_iter_i = _coerce_int(max_iter, 1000, minimum=1)
    min_per_class_i = _coerce_int(min_per_class, 2, minimum=1)
    class_weight_norm = (class_weight or "none").lower()
    if class_weight_norm not in {"balanced", "none", "effective"}:
        class_weight_norm = "none"
    C_f = _coerce_float(C, 1.0, minimum=0.0001)
    effective_beta_f = _coerce_float(effective_beta, 0.9999, minimum=0.5, maximum=0.99999)
    mlp_dropout_f = _coerce_float(mlp_dropout, 0.1, minimum=0.0, maximum=0.9)
    mlp_epochs_i = _coerce_int(mlp_epochs, 50, minimum=1)
    mlp_lr_f = _coerce_float(mlp_lr, 1e-3, minimum=1e-6)
    mlp_weight_decay_f = _coerce_float(mlp_weight_decay, 1e-4, minimum=0.0)
    mlp_label_smoothing_f = _coerce_float(mlp_label_smoothing, 0.05, minimum=0.0, maximum=0.3)
    mlp_loss_type_norm = (mlp_loss_type or "ce").strip().lower()
    if mlp_loss_type_norm not in {"ce", "focal"}:
        mlp_loss_type_norm = "ce"
    mlp_focal_gamma_f = _coerce_float(mlp_focal_gamma, 2.0, minimum=0.0)
    mlp_focal_alpha_f = _coerce_float(mlp_focal_alpha, -1.0)
    if mlp_focal_alpha_f < 0:
        mlp_focal_alpha_f = None
    mlp_sampler_norm = (mlp_sampler or "balanced").strip().lower()
    if mlp_sampler_norm not in {"balanced", "none", "shuffle"}:
        mlp_sampler_norm = "balanced"
    mlp_mixup_alpha_f = _coerce_float(mlp_mixup_alpha, 0.1, minimum=0.0)
    mlp_normalize_embeddings_flag = _parse_bool(mlp_normalize_embeddings)
    mlp_patience_i = _coerce_int(mlp_patience, 6, minimum=1)
    mlp_activation_norm = (mlp_activation or "relu").strip().lower()
    if mlp_activation_norm not in {"relu", "gelu"}:
        mlp_activation_norm = "relu"
    mlp_layer_norm_flag = _parse_bool(mlp_layer_norm)
    mlp_hard_mining_epochs_i = _coerce_int(mlp_hard_mining_epochs, 5, minimum=1)
    logit_adjustment_mode_norm = (logit_adjustment_mode or "none").strip().lower()
    if logit_adjustment_mode_norm not in {"none", "train", "infer", "both"}:
        logit_adjustment_mode_norm = "none"
    logit_adjustment_inference_flag = None
    if logit_adjustment_inference is not None:
        logit_adjustment_inference_flag = _parse_bool(logit_adjustment_inference)
    arcface_enabled_flag = _parse_bool(arcface_enabled)
    arcface_margin_f = _coerce_float(arcface_margin, 0.2, minimum=0.0)
    arcface_scale_f = _coerce_float(arcface_scale, 30.0, minimum=1.0)
    supcon_weight_f = _coerce_float(supcon_weight, 0.0, minimum=0.0)
    supcon_temperature_f = _coerce_float(supcon_temperature, 0.07, minimum=0.0001)
    supcon_projection_dim_i = _coerce_int(supcon_projection_dim, 128, minimum=0)
    supcon_projection_hidden_i = _coerce_int(supcon_projection_hidden, 0, minimum=0)
    embedding_center_flag = _parse_bool(embedding_center)
    embedding_standardize_flag = _parse_bool(embedding_standardize)
    calibration_mode_norm = (calibration_mode or "none").strip().lower()
    if calibration_mode_norm not in {"none", "temperature"}:
        calibration_mode_norm = "none"
    calibration_max_iters_i = _coerce_int(calibration_max_iters, 50, minimum=1)
    calibration_min_temp_f = _coerce_float(calibration_min_temp, 0.5, minimum=0.01)
    calibration_max_temp_f = _coerce_float(calibration_max_temp, 5.0, minimum=calibration_min_temp_f)
    device_override_clean = (device_override or None)
    hard_mis_weight_f = _coerce_float(hard_mis_weight, 3.0, minimum=1.0)
    hard_low_conf_weight_f = _coerce_float(hard_low_conf_weight, 2.0, minimum=1.0)
    hard_low_conf_threshold_f = _coerce_float(hard_low_conf_threshold, 0.65, minimum=0.0, maximum=0.9999)
    hard_margin_threshold_f = _coerce_float(hard_margin_threshold, 0.15, minimum=0.0)
    convergence_tol_f = _coerce_float(convergence_tol, 1e-4, minimum=1e-8)
    bg_class_count_i = _coerce_int(bg_class_count, 2, minimum=1)
    bg_class_count_i = max(1, min(10, bg_class_count_i))
    model_filename = Path(model_filename).name or "my_logreg_model.pkl"
    labelmap_filename = Path(labelmap_filename).name or "my_label_list.pkl"

    extras = [solver_name]
    extras.append(classifier_type_norm)
    if class_weight_norm and class_weight_norm != "none":
        extras.append(f"class_weight={class_weight_norm}")
    if reuse_embeddings_flag:
        extras.append("cache")
    if hard_example_flag:
        extras.append(f"hard({hard_mis_weight_f:.1f}/{hard_low_conf_weight_f:.1f})")
    extras.append(f"bg={bg_class_count_i}")
    job_message += f" [{', '.join(extras)}]"
    _job_log(job, job_message)

    with TRAINING_JOBS_LOCK:
        TRAINING_JOBS[job_id] = job

    _start_training_worker(
        job,
        images_dir=images_dir,
        labels_dir=labels_dir,
        labelmap_path=labelmap_path,
        clip_name=clip_model_name,
        encoder_type=encoder_type_norm,
        encoder_model=encoder_model_name or None,
        output_dir=classifiers_dir,
        labelmap_dir=labelmaps_dir,
        model_filename=model_filename,
        labelmap_filename=labelmap_filename,
        test_size=test_size_f,
        random_seed=random_seed_i,
        batch_size=batch_size_i,
        max_iter=max_iter_i,
        min_per_class=min_per_class_i,
        class_weight=class_weight_norm,
        effective_beta=effective_beta_f,
        C=C_f,
        device_override=device_override_clean,
        solver=solver_name,
        classifier_type=classifier_type_norm,
        mlp_hidden_sizes=str(mlp_hidden_sizes or "256"),
        mlp_dropout=mlp_dropout_f,
        mlp_epochs=mlp_epochs_i,
        mlp_lr=mlp_lr_f,
        mlp_weight_decay=mlp_weight_decay_f,
        mlp_label_smoothing=mlp_label_smoothing_f,
        mlp_loss_type=mlp_loss_type_norm,
        mlp_focal_gamma=mlp_focal_gamma_f,
        mlp_focal_alpha=mlp_focal_alpha_f,
        mlp_sampler=mlp_sampler_norm,
        mlp_mixup_alpha=mlp_mixup_alpha_f,
        mlp_normalize_embeddings=mlp_normalize_embeddings_flag,
        mlp_patience=mlp_patience_i,
        mlp_activation=mlp_activation_norm,
        mlp_layer_norm=mlp_layer_norm_flag,
        mlp_hard_mining_epochs=mlp_hard_mining_epochs_i,
        logit_adjustment_mode=logit_adjustment_mode_norm,
        logit_adjustment_inference=logit_adjustment_inference_flag,
        arcface_enabled=arcface_enabled_flag,
        arcface_margin=arcface_margin_f,
        arcface_scale=arcface_scale_f,
        supcon_weight=supcon_weight_f,
        supcon_temperature=supcon_temperature_f,
        supcon_projection_dim=supcon_projection_dim_i,
        supcon_projection_hidden=supcon_projection_hidden_i,
        embedding_center=embedding_center_flag,
        embedding_standardize=embedding_standardize_flag,
        calibration_mode=calibration_mode_norm,
        calibration_max_iters=calibration_max_iters_i,
        calibration_min_temp=calibration_min_temp_f,
        calibration_max_temp=calibration_max_temp_f,
        reuse_embeddings=reuse_embeddings_flag,
        hard_example_mining=hard_example_flag,
        hard_mining_misclassified_weight=hard_mis_weight_f,
        hard_mining_low_conf_weight=hard_low_conf_weight_f,
        hard_mining_low_conf_threshold=hard_low_conf_threshold_f,
        hard_mining_margin_threshold=hard_margin_threshold_f,
        convergence_tol=convergence_tol_f,
        bg_class_count=bg_class_count_i,
        cancel_event=job.cancel_event,
    )

    return {"job_id": job_id}


def _start_qwen_training_worker(job: QwenTrainingJob, config: QwenTrainingConfig) -> None:
    result_path = Path(config.result_path)

    def progress_cb(value: float, message: str) -> None:
        with QWEN_TRAINING_JOBS_LOCK:
            if job.cancel_event.is_set() and job.status not in {"cancelled", "failed"}:
                _qwen_job_update(job, status="cancelling", message="Cancelling ...", progress=value)
                return
            _qwen_job_update(job, status="running", message=message, progress=value)

    def metrics_cb(payload: Dict[str, Any]) -> None:
        if not payload:
            return
        with QWEN_TRAINING_JOBS_LOCK:
            _qwen_job_append_metric(job, payload)
            progress_val = payload.get("progress")
            progress = None
            if isinstance(progress_val, (int, float)):
                progress = max(0.0, min(float(progress_val), 0.999))
            message = _summarize_qwen_metric_impl(payload)
            _qwen_job_update(job, status="running", message=message, progress=progress, log_message=False)

    def cancel_cb() -> bool:
        return job.cancel_event.is_set()

    def worker() -> None:
        try:
            _prepare_for_qwen_training()
            with QWEN_TRAINING_JOBS_LOCK:
                if job.cancel_event.is_set():
                    _qwen_job_update(job, status="cancelled", message="Cancelled before start.")
                    return
                _qwen_job_update(job, status="running", progress=0.01, message="Preparing Qwen training job ...")
            result = train_qwen_model(config, progress_cb=progress_cb, cancel_cb=cancel_cb, metrics_cb=metrics_cb)
            run_metadata = _persist_qwen_run_metadata(result_path, config, result)
            payload = {
                "checkpoints": result.checkpoints,
                "latest": result.latest_checkpoint,
                "epochs_ran": result.epochs_ran,
                "metadata": run_metadata,
            }
            with QWEN_TRAINING_JOBS_LOCK:
                _qwen_job_update(job, status="succeeded", progress=1.0, message="Training complete", result=payload)
        except QwenTrainingError as exc:
            with QWEN_TRAINING_JOBS_LOCK:
                status = "cancelled" if job.cancel_event.is_set() else "failed"
                _qwen_job_update(job, status=status, message=str(exc), error=str(exc))
        except Exception as exc:  # noqa: BLE001
            with QWEN_TRAINING_JOBS_LOCK:
                _qwen_job_update(job, status="failed", message="Unexpected error", error=str(exc))
        finally:
            _finalize_qwen_training_environment()

    thread = threading.Thread(target=worker, name=f"qwen-train-{job.job_id}", daemon=True)
    thread.start()



def list_training_jobs():
    _prune_job_registry(TRAINING_JOBS, TRAINING_JOBS_LOCK)
    with TRAINING_JOBS_LOCK:
        jobs = sorted(TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        return [{"job_id": job.job_id, "status": job.status, "created_at": job.created_at} for job in jobs]


def get_training_job(job_id: str):
    job = _validate_job_exists(job_id)
    return _serialize_clip_job_impl(job)


def cancel_training_job(job_id: str):
    job = _validate_job_exists(job_id)
    next_status = job.status
    with TRAINING_JOBS_LOCK:
        if job.status in {"succeeded", "failed", "cancelled"}:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="job_not_cancellable")
        if job.cancel_event.is_set():
            return {"status": job.status}
        job.cancel_event.set()
        next_status = job.status if job.status not in {"running", "queued"} else "cancelling"
        _job_update(job, status=next_status, message="Cancellation requested ...")
    return {"status": next_status}


app.include_router(
    build_clip_training_router(
        start_fn=start_clip_training,
        list_fn=list_training_jobs,
        get_fn=get_training_job,
        cancel_fn=cancel_training_job,
    )
)


def _build_sam3_config(
    payload: Sam3TrainRequest,
    meta: Dict[str, Any],
    job_id: str,
    job_logs: Optional[List[str]] = None,
) -> Tuple[OmegaConf, int]:
    dataset_root = Path(meta.get("dataset_root") or SAM3_DATASET_ROOT)
    val_percent = payload.val_percent if payload.val_percent is not None else 0.3
    split_seed = int(payload.split_seed) if payload.split_seed is not None else 42
    random_split = payload.random_split if payload.random_split is not None else True
    train_limit = int(payload.train_limit) if payload.train_limit is not None and payload.train_limit > 0 else None
    val_limit = int(payload.val_limit) if payload.val_limit is not None and payload.val_limit > 0 else None
    meta = _prepare_sam3_training_split(
        dataset_root,
        meta,
        job_id,
        random_split=random_split,
        val_percent=val_percent,
        split_seed=split_seed,
        train_limit=train_limit,
        val_limit=val_limit,
        log_messages=job_logs,
    )
    cfg = OmegaConf.load(str(SAM3_CONFIG_TEMPLATE))
    if not hasattr(cfg.scratch, "enable_segmentation_head"):
        cfg.scratch.enable_segmentation_head = True
    if not hasattr(cfg.scratch, "load_segmentation"):
        cfg.scratch.load_segmentation = False
    train_ann = Path(meta["coco_train_json"]).resolve()
    val_ann = Path(meta["coco_val_json"]).resolve()
    cfg.paths.train_img_folder = str(train_ann.parent)
    cfg.paths.train_ann_file = str(train_ann)
    cfg.paths.val_img_folder = str(val_ann.parent)
    cfg.paths.val_ann_file = str(val_ann)
    run_name = _safe_run_name(payload.run_name, f"sam3_run_{job_id}")
    exp_dir = Path(payload.experiment_log_dir) if payload.experiment_log_dir else (SAM3_JOB_ROOT / run_name)
    if exp_dir.exists():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="run_name_exists")
    cfg.paths.experiment_log_dir = str(exp_dir.resolve())
    cfg.paths.bpe_path = str(SAM3_BPE_PATH)
    cfg.launcher.experiment_log_dir = cfg.paths.experiment_log_dir
    cfg.launcher.gpus_per_node = max(1, int(payload.num_gpus or cfg.launcher.gpus_per_node or 1))
    cfg.trainer.max_epochs = int(payload.max_epochs) if payload.max_epochs is not None else cfg.trainer.max_epochs
    cfg.trainer.val_epoch_freq = int(payload.val_epoch_freq) if payload.val_epoch_freq is not None else cfg.trainer.val_epoch_freq
    cfg.scratch.target_epoch_size = int(payload.target_epoch_size) if payload.target_epoch_size is not None else cfg.scratch.target_epoch_size
    dataset_type = meta.get("type", "bbox")
    seg_head_requested = payload.enable_segmentation_head
    train_seg_requested = payload.train_segmentation
    default_seg = dataset_type == "seg"
    enable_seg_head = bool(seg_head_requested) if seg_head_requested is not None else (bool(cfg.scratch.enable_segmentation_head) or default_seg)
    train_segmentation = bool(train_seg_requested) if train_seg_requested is not None else (bool(cfg.scratch.load_segmentation) or default_seg)
    cfg.scratch.enable_segmentation_head = enable_seg_head or train_segmentation
    cfg.scratch.load_segmentation = train_segmentation
    # Keep legacy flag aligned with head presence so downstream activation sees the capability.
    cfg.scratch.enable_segmentation = cfg.scratch.enable_segmentation_head
    if payload.resolution is not None:
        cfg.scratch.resolution = int(payload.resolution)
    if payload.lr_scale is not None:
        cfg.scratch.lr_scale = float(payload.lr_scale)
    if payload.gradient_accumulation_steps is not None:
        cfg.scratch.gradient_accumulation_steps = int(payload.gradient_accumulation_steps)
    cfg.trainer.gradient_accumulation_steps = cfg.scratch.gradient_accumulation_steps
    if cfg.trainer.gradient_accumulation_steps and cfg.trainer.gradient_accumulation_steps > 1:
        try:
            train_collate = cfg.trainer.data.train.collate_fn
            train_collate._target_ = "sam3.train.data.collator.collate_fn_api_with_chunking"
            train_collate.num_chunks = int(cfg.trainer.gradient_accumulation_steps)
            train_collate._partial_ = True
            if not hasattr(train_collate, "repeats"):
                train_collate.repeats = cfg.scratch.hybrid_repeats
        except Exception:
            pass
    if payload.scheduler_warmup is not None:
        cfg.scratch.scheduler_warmup = int(payload.scheduler_warmup)
    if payload.scheduler_timescale is not None:
        cfg.scratch.scheduler_timescale = int(payload.scheduler_timescale)
    if payload.train_batch_size is not None:
        cfg.scratch.train_batch_size = int(payload.train_batch_size)
    if payload.val_batch_size is not None:
        cfg.scratch.val_batch_size = int(payload.val_batch_size)
    if payload.num_train_workers is not None:
        cfg.scratch.num_train_workers = int(payload.num_train_workers)
    if payload.num_val_workers is not None:
        cfg.scratch.num_val_workers = int(payload.num_val_workers)
    if payload.enable_inst_interactivity is not None:
        cfg.scratch.enable_inst_interactivity = bool(payload.enable_inst_interactivity)
    if payload.train_limit is not None:
        cfg.dataset.num_images = int(payload.train_limit)
    elif payload.target_epoch_size is not None:
        try:
            batches = max(1, int(payload.target_epoch_size))
            batch_size = int(payload.train_batch_size) if payload.train_batch_size is not None else int(cfg.scratch.train_batch_size)
            cfg.dataset.num_images = max(1, batches * batch_size)
        except Exception:
            pass
    if payload.val_limit is not None:
        try:
            val_limit = max(1, int(payload.val_limit))
            cfg.dataset.val_num_images = val_limit
            if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "data") and hasattr(cfg.trainer.data, "val"):
                if hasattr(cfg.trainer.data.val, "dataset"):
                    cfg.trainer.data.val.dataset.limit_ids = val_limit
        except Exception:
            pass
    if payload.log_every_batch:
        try:
            cfg.trainer.logging.log_freq = 1
        except Exception:
            pass
    elif payload.log_freq is not None and "logging" in cfg.trainer:
        cfg.trainer.logging.log_freq = int(payload.log_freq)
    # Language backbone tuning (text alignment preservation)
    if payload.language_backbone_lr is not None:
        try:
            cfg.scratch.lr_language_backbone = float(payload.language_backbone_lr)
        except Exception:
            pass
    if payload.freeze_language_backbone:
        try:
            cfg.scratch.lr_language_backbone = 0.0
        except Exception:
            pass
    # Balance strategy/config
    if payload.balance_strategy is not None:
        cfg.dataset.balance_strategy = payload.balance_strategy
        cfg.dataset.class_balance = payload.balance_strategy != "none"
    if payload.balance_classes is not None:
        cfg.dataset.class_balance = bool(payload.balance_classes)
    if payload.balance_power is not None:
        cfg.dataset.balance_power = float(payload.balance_power)
    if payload.balance_clip is not None:
        cfg.dataset.balance_clip = float(payload.balance_clip)
    if payload.balance_beta is not None:
        cfg.dataset.balance_beta = float(payload.balance_beta)
    if payload.balance_gamma is not None:
        cfg.dataset.balance_gamma = float(payload.balance_gamma)
    cfg.trainer.checkpoint.save_dir = f"{cfg.launcher.experiment_log_dir}/checkpoints"
    if "meters" in cfg.trainer and "val" in cfg.trainer.meters:
        try:
            cfg.trainer.meters.val.roboflow100.detection.dump_dir = f"{cfg.launcher.experiment_log_dir}/dumps/local"
            cfg.trainer.meters.val.roboflow100.detection.pred_file_evaluators[0].gt_path = cfg.paths.val_ann_file
            # Apply val tuning
            if payload.val_max_dets is not None:
                cfg.trainer.meters.val.roboflow100.detection.maxdets = int(payload.val_max_dets)
        except Exception:
            pass
    # Prompt vocab overrides: allow multiple variants per class and optional randomization during training
    user_prompts = payload.prompt_variants or {}
    prompt_map: Dict[int, List[str]] = {}
    classes = meta.get("classes") or []
    if classes and user_prompts:
        def _normalise_variants(raw: Any) -> List[str]:
            if raw is None:
                return []
            if isinstance(raw, str):
                parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
                return parts if parts else [raw.strip()] if raw.strip() else []
            if isinstance(raw, (list, tuple, set)):
                return [str(p).strip() for p in raw if str(p).strip()]
            return []

        for idx, label in enumerate(classes):
            # allow lookup by label or by (1-based) category id
            cat_id = idx + 1
            custom = (
                user_prompts.get(label)
                or user_prompts.get(str(label))
                or user_prompts.get(cat_id)
                or user_prompts.get(str(cat_id))
            )
            variants = _normalise_variants(custom)
            if variants:
                prompt_map[cat_id] = variants

    if prompt_map:
        prompt_randomize = bool(payload.prompt_randomize) if payload.prompt_randomize is not None else True
        # Train loader
        try:
            train_loader_cfg = cfg.trainer.data.train.dataset.get("coco_json_loader")  # type: ignore[index]
        except Exception:
            train_loader_cfg = None
        if train_loader_cfg is None:
            cfg.trainer.data.train.dataset["coco_json_loader"] = {}
            train_loader_cfg = cfg.trainer.data.train.dataset.get("coco_json_loader")  # type: ignore[index]
        try:
            train_loader_cfg["_target_"] = "sam3.train.data.coco_json_loaders.COCO_FROM_JSON"
            train_loader_cfg["_partial_"] = True
            train_loader_cfg["prompts"] = prompt_map
            train_loader_cfg["prompt_randomize"] = prompt_randomize
        except Exception:
            pass
        # Val loader (deterministic prompts)
        try:
            val_loader_cfg = cfg.trainer.data.val.dataset.coco_json_loader  # type: ignore[assignment]
        except Exception:
            val_loader_cfg = None
        if val_loader_cfg is None:
            try:
                cfg.trainer.data.val.dataset["coco_json_loader"] = {}
                val_loader_cfg = cfg.trainer.data.val.dataset.coco_json_loader  # type: ignore[assignment]
            except Exception:
                val_loader_cfg = None
        if val_loader_cfg is not None:
            try:
                val_loader_cfg["_target_"] = "sam3.train.data.coco_json_loaders.COCO_FROM_JSON"
                val_loader_cfg["_partial_"] = True
                val_loader_cfg["prompts"] = prompt_map
                val_loader_cfg["prompt_randomize"] = False
            except Exception:
                pass
    cfg.launcher.num_nodes = 1
    cfg.submitit.use_cluster = False
    cfg.submitit.cpus_per_task = max(cfg.scratch.num_train_workers, cfg.submitit.cpus_per_task or 0)
    Path(cfg.paths.experiment_log_dir).mkdir(parents=True, exist_ok=True)
    return cfg, int(cfg.launcher.gpus_per_node)


def create_sam3_training_job(payload: Sam3TrainRequest):
    meta = _resolve_sam3_dataset_meta(payload.dataset_id)
    job_id = uuid.uuid4().hex
    prep_logs: List[str] = []
    cfg, num_gpus = _build_sam3_config(payload, meta, job_id, prep_logs)
    config_dict = OmegaConf.to_container(cfg, resolve=False)  # type: ignore[arg-type]
    job = Sam3TrainingJob(job_id=job_id, config=config_dict)
    with SAM3_TRAINING_JOBS_LOCK:
        SAM3_TRAINING_JOBS[job_id] = job
        for msg in prep_logs:
            _sam3_job_log(job, msg)
        _sam3_job_log(job, "Job queued")
    logger.info("[sam3-train %s] dataset=%s gpus=%s", job_id[:8], payload.dataset_id, num_gpus)
    _start_sam3_training_worker(
        job,
        cfg,
        num_gpus,
        val_score_thresh=payload.val_score_thresh,
        val_max_dets=payload.val_max_dets,
    )
    return {"job_id": job_id}


def list_sam3_training_jobs():
    _prune_job_registry(SAM3_TRAINING_JOBS, SAM3_TRAINING_JOBS_LOCK)
    with SAM3_TRAINING_JOBS_LOCK:
        jobs = sorted(SAM3_TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        return [_serialize_sam3_job_impl(job) for job in jobs]


def get_sam3_training_job(job_id: str):
    job = _get_sam3_job(job_id)
    return _serialize_sam3_job_impl(job)


def cancel_sam3_training_job(job_id: str):
    job = _get_sam3_job(job_id)
    with SAM3_TRAINING_JOBS_LOCK:
        if job.status in {"succeeded", "failed", "cancelled"}:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="job_not_cancellable")
        if job.cancel_event.is_set():
            return {"status": job.status}
        job.cancel_event.set()
        if job.process and job.process.poll() is None:
            try:
                job.process.terminate()
            except Exception:  # noqa: BLE001
                pass
        next_status = job.status if job.status not in {"running", "queued"} else "cancelling"
        _sam3_job_update(job, status=next_status, message="Cancellation requested ...")
    return {"status": job.status}


def sam3_train_cache_size():
    cache_root = SAM3_JOB_ROOT / "splits"
    return {"bytes": _dir_size_bytes_impl(cache_root)}


def sam3_train_cache_purge():
    cache_root = SAM3_JOB_ROOT / "splits"
    deleted = _purge_directory(cache_root)
    return {"status": "ok", "deleted_bytes": deleted}


app.include_router(
    build_sam3_training_router(
        create_job_fn=create_sam3_training_job,
        list_jobs_fn=list_sam3_training_jobs,
        get_job_fn=get_sam3_training_job,
        cancel_job_fn=cancel_sam3_training_job,
        cache_size_fn=sam3_train_cache_size,
        cache_purge_fn=sam3_train_cache_purge,
        request_cls=Sam3TrainRequest,
    )
)


def create_yolo_training_job(payload: YoloTrainRequest):
    if not payload.accept_tos:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_tos_required")
    job_id = uuid.uuid4().hex
    run_dir = _yolo_run_dir_impl(
        job_id,
        create=True,
        job_root=YOLO_JOB_ROOT,
        sanitize_fn=_sanitize_yolo_run_id_impl,
        http_exception_cls=HTTPException,
    )
    dataset_info = _resolve_yolo_training_dataset(payload)
    if payload.task == "segment" and dataset_info.get("yolo_ready") and not dataset_info.get("yolo_seg_ready"):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_seg_requires_polygons")
    config = payload.dict(exclude_none=True)
    config["paths"] = {"run_dir": str(run_dir)}
    config["dataset"] = dataset_info
    message = "Queued (training not started)"
    status = "queued"
    if not dataset_info.get("yolo_ready"):
        status = "blocked"
        message = "Dataset is not YOLO-ready; conversion required."
    job = YoloTrainingJob(job_id=job_id, config=config, message=message, status=status)
    with YOLO_TRAINING_JOBS_LOCK:
        YOLO_TRAINING_JOBS[job_id] = job
        _yolo_job_log(job, job.message)
    _yolo_write_run_meta_impl(
        run_dir,
        {
            "job_id": job_id,
            "status": job.status,
            "message": job.message,
            "config": job.config,
        },
        meta_name=YOLO_RUN_META_NAME,
        time_fn=time.time,
    )
    if job.status != "blocked":
        _start_yolo_training_worker(job)
    return {"job_id": job_id}


def list_yolo_training_jobs():
    _prune_job_registry(YOLO_TRAINING_JOBS, YOLO_TRAINING_JOBS_LOCK)
    with YOLO_TRAINING_JOBS_LOCK:
        jobs = sorted(YOLO_TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        return [_serialize_yolo_job_impl(job) for job in jobs]


def get_yolo_training_job(job_id: str):
    job = _get_yolo_job(job_id)
    return _serialize_yolo_job_impl(job)


def cancel_yolo_training_job(job_id: str):
    job = _get_yolo_job(job_id)
    with YOLO_TRAINING_JOBS_LOCK:
        if job.status in {"succeeded", "failed", "cancelled"}:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="job_not_cancellable")
        if job.cancel_event.is_set():
            return {"status": job.status}
        job.cancel_event.set()
        next_status = job.status if job.status not in {"running", "queued"} else "cancelled"
        _yolo_job_update(job, status=next_status, message="Cancellation requested ...")
        run_dir = _yolo_run_dir_impl(
            job.job_id,
            create=False,
            job_root=YOLO_JOB_ROOT,
            sanitize_fn=_sanitize_yolo_run_id_impl,
            http_exception_cls=HTTPException,
        )
        _yolo_write_run_meta_impl(
            run_dir,
            {
                "job_id": job.job_id,
                "status": job.status,
                "message": job.message,
                "config": job.config,
            },
            meta_name=YOLO_RUN_META_NAME,
            time_fn=time.time,
        )
    return {"status": job.status}


def create_yolo_head_graft_job(payload: YoloHeadGraftRequest):
    if not payload.accept_tos:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_tos_required")
    job_id = uuid.uuid4().hex
    if payload.run_name:
        safe_id = _sanitize_yolo_run_id_impl(payload.run_name)
        if safe_id:
            job_id = safe_id
    run_dir = _yolo_run_dir_impl(
        job_id,
        create=False,
        job_root=YOLO_JOB_ROOT,
        sanitize_fn=_sanitize_yolo_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if run_dir.exists():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="yolo_run_exists")
    config = payload.dict(exclude_none=True)
    job = YoloHeadGraftJob(job_id=job_id, config=config, message="Queued (head graft not started)")
    with YOLO_HEAD_GRAFT_JOBS_LOCK:
        YOLO_HEAD_GRAFT_JOBS[job_id] = job
        _yolo_head_graft_job_log(job, job.message)
    run_dir = _yolo_run_dir_impl(
        job_id,
        create=True,
        job_root=YOLO_JOB_ROOT,
        sanitize_fn=_sanitize_yolo_run_id_impl,
        http_exception_cls=HTTPException,
    )
    _yolo_write_run_meta_impl(
        run_dir,
        {"job_id": job_id, "status": job.status, "message": job.message, "config": job.config},
        meta_name=YOLO_RUN_META_NAME,
        time_fn=time.time,
    )
    _start_yolo_head_graft_worker(job)
    return _serialize_yolo_head_graft_job_impl(job)


def yolo_head_graft_dry_run(payload: YoloHeadGraftDryRunRequest):
    base_run_id = str(payload.base_run_id or "").strip()
    if not base_run_id:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_base_missing")
    base_run_dir = _yolo_run_dir_impl(
        base_run_id,
        create=False,
        job_root=YOLO_JOB_ROOT,
        sanitize_fn=_sanitize_yolo_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not base_run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_base_missing")
    base_meta = _yolo_load_run_meta_impl(base_run_dir, meta_name=YOLO_RUN_META_NAME)
    base_cfg = base_meta.get("config") or {}
    base_task = str(base_cfg.get("task") or "detect").lower()
    base_variant = base_cfg.get("variant")
    base_labelmap = _yolo_load_run_labelmap_impl(
        base_run_dir,
        yolo_load_labelmap_fn=_yolo_load_labelmap_impl,
        yaml_load_fn=yaml.safe_load,
    )
    dataset_payload = YoloTrainRequest(dataset_id=payload.dataset_id, dataset_root=payload.dataset_root)
    dataset_info = _resolve_yolo_training_dataset(dataset_payload)
    if not dataset_info.get("yolo_ready"):
        return {
            "ok": False,
            "error": "yolo_not_ready",
            "base_run_id": base_run_id,
            "base_task": base_task,
            "base_variant": base_variant,
        }
    dataset_task = str(dataset_info.get("task") or "detect").lower()
    new_labelmap = _yolo_load_labelmap_impl(Path(dataset_info.get("yolo_labelmap_path") or ""))
    base_norm = {_normalize_class_name_for_match(n) for n in base_labelmap if n}
    new_norm = {_normalize_class_name_for_match(n) for n in new_labelmap if n}
    overlap = sorted(base_norm.intersection(new_norm))
    ok = base_task == "detect" and dataset_task == "detect" and not overlap and bool(base_labelmap) and bool(new_labelmap)
    return {
        "ok": ok,
        "base_run_id": base_run_id,
        "base_task": base_task,
        "base_variant": base_variant,
        "base_label_count": len(base_labelmap),
        "new_label_count": len(new_labelmap),
        "overlap": overlap,
        "dataset_id": dataset_info.get("id") or dataset_info.get("dataset_id"),
        "yolo_ready": dataset_info.get("yolo_ready"),
        "dataset_task": dataset_task,
    }


def list_yolo_head_graft_jobs():
    _prune_job_registry(YOLO_HEAD_GRAFT_JOBS, YOLO_HEAD_GRAFT_JOBS_LOCK)
    with YOLO_HEAD_GRAFT_JOBS_LOCK:
        jobs = sorted(YOLO_HEAD_GRAFT_JOBS.values(), key=lambda job: job.created_at, reverse=True)
    return [_serialize_yolo_head_graft_job_impl(job) for job in jobs]


def get_yolo_head_graft_job(job_id: str):
    _prune_job_registry(YOLO_HEAD_GRAFT_JOBS, YOLO_HEAD_GRAFT_JOBS_LOCK)
    with YOLO_HEAD_GRAFT_JOBS_LOCK:
        job = YOLO_HEAD_GRAFT_JOBS.get(job_id)
    if job:
        return _serialize_yolo_head_graft_job_impl(job)
    raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="job_not_found")


def cancel_yolo_head_graft_job(job_id: str):
    with YOLO_HEAD_GRAFT_JOBS_LOCK:
        job = YOLO_HEAD_GRAFT_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="job_not_found")
        if job.status in {"succeeded", "failed", "cancelled"}:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="job_not_cancellable")
        if job.cancel_event.is_set():
            return {"status": job.status}
        job.cancel_event.set()
        stopped = False
        if job.status in {"running", "queued"}:
            stopped = _yolo_head_graft_force_stop_impl(job)
        next_status = "cancelled" if stopped else (job.status if job.status not in {"running", "queued"} else "cancelling")
        _yolo_head_graft_job_update(job, status=next_status, message="Cancellation requested ...")
        _yolo_head_graft_audit(job, "cancel_requested", event="cancel", extra={"forced": stopped})
    return {"status": job.status}


app.include_router(
    build_yolo_training_router(
        create_job_fn=create_yolo_training_job,
        list_jobs_fn=list_yolo_training_jobs,
        get_job_fn=get_yolo_training_job,
        cancel_job_fn=cancel_yolo_training_job,
        head_graft_create_fn=create_yolo_head_graft_job,
        head_graft_dry_run_fn=yolo_head_graft_dry_run,
        head_graft_list_fn=list_yolo_head_graft_jobs,
        head_graft_get_fn=get_yolo_head_graft_job,
        head_graft_cancel_fn=cancel_yolo_head_graft_job,
        train_request_cls=YoloTrainRequest,
        head_graft_request_cls=YoloHeadGraftRequest,
        head_graft_dry_run_request_cls=YoloHeadGraftDryRunRequest,
    )
)


def create_rfdetr_training_job(payload: RfDetrTrainRequest):
    if not payload.accept_tos:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_tos_required")
    job_id = uuid.uuid4().hex
    run_dir = _rfdetr_run_dir_impl(
        job_id,
        create=True,
        job_root=RFDETR_JOB_ROOT,
        sanitize_fn=_sanitize_rfdetr_run_id_impl,
        http_exception_cls=HTTPException,
    )
    dataset_info = _resolve_rfdetr_training_dataset(payload)
    config = payload.dict(exclude_none=True)
    config["paths"] = {"run_dir": str(run_dir)}
    config["dataset"] = dataset_info
    message = "Queued (training not started)"
    status = "queued"
    job = RfDetrTrainingJob(job_id=job_id, config=config, message=message, status=status)
    with RFDETR_TRAINING_JOBS_LOCK:
        RFDETR_TRAINING_JOBS[job_id] = job
        _rfdetr_job_log(job, job.message)
    _rfdetr_write_run_meta_impl(
        run_dir,
        {
            "job_id": job_id,
            "status": job.status,
            "message": job.message,
            "config": job.config,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        },
        meta_name=RFDETR_RUN_META_NAME,
        time_fn=time.time,
    )
    _start_rfdetr_training_worker(job)
    return {"job_id": job_id}


def list_rfdetr_training_jobs():
    _prune_job_registry(RFDETR_TRAINING_JOBS, RFDETR_TRAINING_JOBS_LOCK)
    with RFDETR_TRAINING_JOBS_LOCK:
        jobs = sorted(RFDETR_TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        return [_serialize_rfdetr_job_impl(job) for job in jobs]


def get_rfdetr_training_job(job_id: str):
    job = _get_rfdetr_job(job_id)
    return _serialize_rfdetr_job_impl(job)


def cancel_rfdetr_training_job(job_id: str):
    job = _get_rfdetr_job(job_id)
    with RFDETR_TRAINING_JOBS_LOCK:
        if job.status in {"succeeded", "failed", "cancelled"}:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="job_not_cancellable")
        if job.cancel_event.is_set():
            return {"status": job.status}
        job.cancel_event.set()
        next_status = job.status if job.status not in {"running", "queued"} else "cancelled"
        _rfdetr_job_update(job, status=next_status, message="Cancellation requested ...")
        run_dir = _rfdetr_run_dir_impl(
            job.job_id,
            create=False,
            job_root=RFDETR_JOB_ROOT,
            sanitize_fn=_sanitize_rfdetr_run_id_impl,
            http_exception_cls=HTTPException,
        )
        _rfdetr_write_run_meta_impl(
            run_dir,
            {
                "job_id": job.job_id,
                "status": job.status,
                "message": job.message,
                "config": job.config,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
            },
            meta_name=RFDETR_RUN_META_NAME,
            time_fn=time.time,
        )
    return {"status": job.status}


app.include_router(
    build_rfdetr_training_router(
        create_job_fn=create_rfdetr_training_job,
        list_jobs_fn=list_rfdetr_training_jobs,
        get_job_fn=get_rfdetr_training_job,
        cancel_job_fn=cancel_rfdetr_training_job,
        request_cls=RfDetrTrainRequest,
    )
)


def list_rfdetr_variants(task: Optional[str] = Query(None)):
    if task:
        task_norm = task.strip().lower()
        return [v for v in RFDETR_VARIANTS if v.get("task") == task_norm]
    return RFDETR_VARIANTS


def set_rfdetr_active(payload: RfDetrActiveRequest):
    run_dir = _rfdetr_run_dir_impl(
        payload.run_id,
        create=False,
        job_root=RFDETR_JOB_ROOT,
        sanitize_fn=_sanitize_rfdetr_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="rfdetr_run_not_found")
    best_path = _rfdetr_best_checkpoint_impl(run_dir)
    if not best_path:
        raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="rfdetr_best_missing")
    meta = _rfdetr_load_run_meta_impl(run_dir, meta_name=RFDETR_RUN_META_NAME)
    config = meta.get("config") or {}
    dataset = config.get("dataset") or {}
    active_payload = {
        "run_id": payload.run_id,
        "run_name": config.get("run_name") or dataset.get("label") or payload.run_id,
        "best_path": best_path,
        "labelmap_path": str(run_dir / "labelmap.txt") if (run_dir / "labelmap.txt").exists() else None,
        "task": config.get("task") or dataset.get("task"),
        "variant": config.get("variant"),
    }
    return _save_rfdetr_active_impl(active_payload, RFDETR_ACTIVE_PATH)


app.include_router(
    build_detectors_default_router(
        load_default_fn=lambda: _load_detector_default_impl(DETECTOR_DEFAULT_PATH),
        save_default_fn=lambda settings: _save_detector_default_impl(
            settings,
            DETECTOR_DEFAULT_PATH,
            HTTPException,
        ),
        request_cls=DetectorDefaultRequest,
    )
)


def download_rfdetr_run(run_id: str):
    run_dir = _rfdetr_run_dir_impl(
        run_id,
        create=False,
        job_root=RFDETR_JOB_ROOT,
        sanitize_fn=_sanitize_rfdetr_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="rfdetr_run_not_found")
    meta = _rfdetr_load_run_meta_impl(run_dir, meta_name=RFDETR_RUN_META_NAME)
    run_name = meta.get("config", {}).get("run_name") or meta.get("job_id") or run_id
    safe_name = _sanitize_yolo_run_id_impl(run_name)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename in sorted(RFDETR_KEEP_FILES):
            path = run_dir / filename
            if path.exists():
                zf.write(path, arcname=filename)
    buffer.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{safe_name}.zip"'}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


def yolo_run_summary(run_id: str):
    run_dir = _yolo_run_dir_impl(
        run_id,
        create=False,
        job_root=YOLO_JOB_ROOT,
        sanitize_fn=_sanitize_yolo_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_run_not_found")
    meta = _yolo_load_run_meta_impl(run_dir, meta_name=YOLO_RUN_META_NAME)
    config = meta.get("config") or {}
    dataset = config.get("dataset") or {}
    run_name = config.get("run_name") or dataset.get("label") or dataset.get("id") or run_id
    labelmap = _read_labelmap_lines(run_dir / "labelmap.txt")
    metrics = _clean_metric_summary_impl(
        _yolo_metrics_summary_impl(run_dir, read_csv_last_row_fn=_read_csv_last_row)
    )
    return {
        "run_id": run_id,
        "run_name": run_name,
        "dataset_label": dataset.get("label"),
        "dataset_id": dataset.get("id") or dataset.get("dataset_id"),
        "task": config.get("task"),
        "variant": config.get("variant"),
        "labelmap": labelmap,
        "metrics": metrics,
    }


def rfdetr_run_summary(run_id: str):
    run_dir = _rfdetr_run_dir_impl(
        run_id,
        create=False,
        job_root=RFDETR_JOB_ROOT,
        sanitize_fn=_sanitize_rfdetr_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="rfdetr_run_not_found")
    meta = _rfdetr_load_run_meta_impl(run_dir, meta_name=RFDETR_RUN_META_NAME)
    config = meta.get("config") or {}
    dataset = config.get("dataset") or {}
    run_name = config.get("run_name") or dataset.get("label") or dataset.get("id") or run_id
    labelmap = _read_labelmap_lines(run_dir / "labelmap.txt")
    metrics = _clean_metric_summary_impl(_rfdetr_metrics_summary_impl(run_dir))
    return {
        "run_id": run_id,
        "run_name": run_name,
        "dataset_label": dataset.get("label"),
        "dataset_id": dataset.get("id") or dataset.get("dataset_id"),
        "task": config.get("task"),
        "variant": config.get("variant"),
        "labelmap": labelmap,
        "metrics": metrics,
    }

def delete_rfdetr_run(run_id: str):
    run_dir = _rfdetr_run_dir_impl(
        run_id,
        create=False,
        job_root=RFDETR_JOB_ROOT,
        sanitize_fn=_sanitize_rfdetr_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="rfdetr_run_not_found")
    try:
        shutil.rmtree(run_dir)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return {"status": "deleted", "run_id": run_id}


def list_yolo_variants(task: Optional[str] = Query(None)):
    if task:
        task_norm = task.strip().lower()
        return [v for v in YOLO_VARIANTS if v.get("task") == task_norm]
    return YOLO_VARIANTS


def set_yolo_active(payload: YoloActiveRequest):
    run_dir = _yolo_run_dir_impl(
        payload.run_id,
        create=False,
        job_root=YOLO_JOB_ROOT,
        sanitize_fn=_sanitize_yolo_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_run_not_found")
    best_path = run_dir / "best.pt"
    if not best_path.exists():
        raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="yolo_best_missing")
    meta = _yolo_load_run_meta_impl(run_dir, meta_name=YOLO_RUN_META_NAME)
    config = meta.get("config") or {}
    dataset = config.get("dataset") or {}
    active_payload = {
        "run_id": payload.run_id,
        "run_name": config.get("run_name") or dataset.get("label") or payload.run_id,
        "best_path": str(best_path),
        "labelmap_path": str(run_dir / "labelmap.txt") if (run_dir / "labelmap.txt").exists() else None,
        "task": config.get("task"),
        "variant": config.get("variant"),
    }
    if meta.get("head_graft"):
        active_payload["head_graft"] = meta.get("head_graft")
    return _save_yolo_active_impl(active_payload, YOLO_ACTIVE_PATH)


def yolo_predict_region(payload: YoloRegionRequest):
    model, labelmap, task = _ensure_yolo_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_region_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image_impl(
        payload.image_base64,
        payload.image_token,
        fetch_preloaded_fn=_fetch_preloaded_image,
        decode_image_fn=lambda b64: _decode_image_base64_impl(
            b64,
            max_bytes=BASE64_IMAGE_MAX_BYTES,
            max_dim=BASE64_IMAGE_MAX_DIM,
            allow_downscale=True,
        ),
        store_preloaded_fn=_store_preloaded_image,
        hash_fn=lambda payload: hashlib.md5(payload).hexdigest(),
    )
    img_w, img_h = pil_img.size
    full_w = int(payload.full_width) if payload.full_width else img_w
    full_h = int(payload.full_height) if payload.full_height else img_h
    x, y, w, h = [float(v) for v in payload.region[:4]]
    left = max(0.0, min(full_w, x))
    top = max(0.0, min(full_h, y))
    right = max(left + 1.0, min(full_w, x + w))
    bottom = max(top + 1.0, min(full_h, y + h))
    warnings: List[str] = []
    if payload.image_is_cropped:
        if payload.full_width is None or payload.full_height is None:
            warnings.append("full_size_missing")
        if abs((right - left) - img_w) > 0.5 or abs((bottom - top) - img_h) > 0.5:
            warnings.append("region_crop_mismatch")
        right = min(full_w, left + img_w)
        bottom = min(full_h, top + img_h)
        crop = pil_img
    else:
        crop = pil_img.crop((left, top, right, bottom))
    conf = float(payload.conf) if payload.conf is not None else 0.25
    iou = float(payload.iou) if payload.iou is not None else 0.45
    max_det = int(payload.max_det) if payload.max_det is not None else 300
    if conf < 0 or conf > 1:
        conf = min(1.0, max(0.0, conf))
        warnings.append("conf_clamped")
    if iou < 0 or iou > 1:
        iou = min(1.0, max(0.0, iou))
        warnings.append("iou_clamped")
    if max_det < 1:
        max_det = 1
        warnings.append("max_det_clamped")
    if max_det > 5000:
        max_det = 5000
        warnings.append("max_det_clamped")
    expected = payload.expected_labelmap or []
    if expected and not labelmap:
        warnings.append("labelmap_missing")
    elif expected and labelmap and expected != labelmap:
        warnings.append("labelmap_mismatch")
    with YOLO_INFER_LOCK:
        with YOLO_INFER_LOCK:
            results = model.predict(crop, conf=conf, iou=iou, max_det=max_det, verbose=False)
    detections: List[YoloRegionDetection] = []
    if results:
        det_boxes = results[0].boxes
        if det_boxes is not None and det_boxes.xyxy is not None:
            xyxy = det_boxes.xyxy.cpu().numpy()
            confs = det_boxes.conf.cpu().numpy() if det_boxes.conf is not None else None
            classes = det_boxes.cls.cpu().numpy() if det_boxes.cls is not None else None
            for idx, box in enumerate(xyxy):
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                cx = (x1 + x2) / 2 + left
                cy = (y1 + y2) / 2 + top
                if payload.center_only and not (left <= cx <= right and top <= cy <= bottom):
                    continue
                abs_x = max(0.0, min(full_w, x1 + left))
                abs_y = max(0.0, min(full_h, y1 + top))
                abs_w = max(0.0, min(full_w - abs_x, (x2 - x1)))
                abs_h = max(0.0, min(full_h - abs_y, (y2 - y1)))
                class_id = int(classes[idx]) if classes is not None else -1
                class_name = None
                if class_id >= 0 and class_id < len(labelmap):
                    class_name = labelmap[class_id]
                score = float(confs[idx]) if confs is not None else None
                detections.append(
                    YoloRegionDetection(
                        bbox=[abs_x, abs_y, abs_w, abs_h],
                        class_id=class_id,
                        class_name=class_name,
                        score=score,
                    )
                )
    return YoloRegionResponse(detections=detections, labelmap=labelmap, warnings=warnings or None)


def rfdetr_predict_region(payload: RfDetrRegionRequest):
    model, labelmap, task = _ensure_rfdetr_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_region_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image_impl(
        payload.image_base64,
        payload.image_token,
        fetch_preloaded_fn=_fetch_preloaded_image,
        decode_image_fn=lambda b64: _decode_image_base64_impl(
            b64,
            max_bytes=BASE64_IMAGE_MAX_BYTES,
            max_dim=BASE64_IMAGE_MAX_DIM,
            allow_downscale=True,
        ),
        store_preloaded_fn=_store_preloaded_image,
        hash_fn=lambda payload: hashlib.md5(payload).hexdigest(),
    )
    img_w, img_h = pil_img.size
    full_w = int(payload.full_width) if payload.full_width else img_w
    full_h = int(payload.full_height) if payload.full_height else img_h
    x, y, w, h = [float(v) for v in payload.region[:4]]
    left = max(0.0, min(full_w, x))
    top = max(0.0, min(full_h, y))
    right = max(left + 1.0, min(full_w, x + w))
    bottom = max(top + 1.0, min(full_h, y + h))
    warnings: List[str] = []
    if payload.image_is_cropped:
        if payload.full_width is None or payload.full_height is None:
            warnings.append("full_size_missing")
        if abs((right - left) - img_w) > 0.5 or abs((bottom - top) - img_h) > 0.5:
            warnings.append("region_crop_mismatch")
        right = min(full_w, left + img_w)
        bottom = min(full_h, top + img_h)
        crop = pil_img
    else:
        crop = pil_img.crop((left, top, right, bottom))
    conf = float(payload.conf) if payload.conf is not None else 0.25
    max_det = int(payload.max_det) if payload.max_det is not None else 300
    if conf < 0 or conf > 1:
        conf = min(1.0, max(0.0, conf))
        warnings.append("conf_clamped")
    if max_det < 1:
        max_det = 1
        warnings.append("max_det_clamped")
    if max_det > 5000:
        max_det = 5000
        warnings.append("max_det_clamped")
    expected = payload.expected_labelmap or []
    if expected and not labelmap:
        warnings.append("labelmap_missing")
    elif expected and labelmap and expected != labelmap:
        warnings.append("labelmap_mismatch")
    detections: List[RfDetrRegionDetection] = []
    try:
        with RFDETR_INFER_LOCK:
            results = model.predict(crop, threshold=conf)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"rfdetr_predict_failed:{exc}") from exc
    if results is not None:
        xyxy = getattr(results, "xyxy", None)
        scores = getattr(results, "confidence", None)
        class_ids = getattr(results, "class_id", None)
        if xyxy is not None and len(xyxy):
            raw_entries: List[Tuple[Optional[float], RfDetrRegionDetection]] = []
            labelmap_shifted = False
            for idx, box in enumerate(xyxy):
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                cx = (x1 + x2) / 2 + left
                cy = (y1 + y2) / 2 + top
                if payload.center_only and not (left <= cx <= right and top <= cy <= bottom):
                    continue
                abs_x = max(0.0, min(full_w, x1 + left))
                abs_y = max(0.0, min(full_h, y1 + top))
                abs_w = max(0.0, min(full_w - abs_x, (x2 - x1)))
                abs_h = max(0.0, min(full_h - abs_y, (y2 - y1)))
                class_id = int(class_ids[idx]) if class_ids is not None else -1
                if labelmap and class_id >= len(labelmap) and 0 <= class_id - 1 < len(labelmap):
                    class_id -= 1
                    labelmap_shifted = True
                class_name = None
                if class_id >= 0 and class_id < len(labelmap):
                    class_name = labelmap[class_id]
                score = float(scores[idx]) if scores is not None else None
                raw_entries.append(
                    (
                        score,
                        RfDetrRegionDetection(
                            bbox=[abs_x, abs_y, abs_w, abs_h],
                            class_id=class_id,
                            class_name=class_name,
                            score=score,
                        ),
                    )
                )
            if raw_entries:
                if any(score is not None for score, _ in raw_entries):
                    raw_entries.sort(key=lambda item: item[0] if item[0] is not None else -1.0, reverse=True)
                detections = [entry for _, entry in raw_entries[:max_det]]
            if labelmap_shifted:
                warnings.append("labelmap_shifted")
    return RfDetrRegionResponse(detections=detections, labelmap=labelmap, warnings=warnings or None)


_yolo_extract_detections = _yolo_extract_detections_impl
_rfdetr_extract_detections = _rfdetr_extract_detections_impl


def yolo_predict_full(payload: YoloFullRequest):
    model, labelmap, task = _ensure_yolo_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_full_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image_impl(
        payload.image_base64,
        payload.image_token,
        fetch_preloaded_fn=_fetch_preloaded_image,
        decode_image_fn=lambda b64: _decode_image_base64_impl(
            b64,
            max_bytes=BASE64_IMAGE_MAX_BYTES,
            max_dim=BASE64_IMAGE_MAX_DIM,
            allow_downscale=True,
        ),
        store_preloaded_fn=_store_preloaded_image,
        hash_fn=lambda payload: hashlib.md5(payload).hexdigest(),
    )
    img_w, img_h = pil_img.size
    warnings: List[str] = []
    conf = _clamp_conf_value_impl(float(payload.conf) if payload.conf is not None else 0.25, warnings)
    iou = _clamp_iou_value_impl(float(payload.iou) if payload.iou is not None else 0.45, warnings)
    max_det = _clamp_max_det_value_impl(int(payload.max_det) if payload.max_det is not None else 300, warnings)
    _apply_expected_labelmap_warnings(payload.expected_labelmap, labelmap, warnings)
    def _predict_with_fallback(device: Optional[str] = None):
        kwargs = {"conf": conf, "iou": iou, "max_det": max_det, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        return model.predict(pil_img, **kwargs)

    try:
        with YOLO_INFER_LOCK:
            results = _predict_with_fallback()
    except RuntimeError as exc:  # noqa: BLE001
        msg = str(exc)
        if "CUDA" in msg or "device-side assert" in msg:
            warnings.append("yolo_cuda_error")
            _unload_detector_inference()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            # Retry on CPU to keep baseline functional.
            try:
                model, labelmap, task = _ensure_yolo_inference_runtime()
                if hasattr(model, "to"):
                    try:
                        model.to("cpu")
                    except Exception:
                        pass
                with YOLO_INFER_LOCK:
                    results = _predict_with_fallback(device="cpu")
                warnings.append("yolo_cuda_fallback_cpu")
            except Exception:
                raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="yolo_cuda_error") from exc
        else:
            raise
    raw = _yolo_extract_detections(results, labelmap, 0.0, 0.0, img_w, img_h)
    detections = [YoloRegionDetection(**item) for item in raw]
    return YoloRegionResponse(detections=detections, labelmap=labelmap, warnings=warnings or None)


def yolo_predict_windowed(payload: YoloWindowedRequest):
    model, labelmap, task = _ensure_yolo_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_windowed_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image_impl(
        payload.image_base64,
        payload.image_token,
        fetch_preloaded_fn=_fetch_preloaded_image,
        decode_image_fn=lambda b64: _decode_image_base64_impl(
            b64,
            max_bytes=BASE64_IMAGE_MAX_BYTES,
            max_dim=BASE64_IMAGE_MAX_DIM,
            allow_downscale=True,
        ),
        store_preloaded_fn=_store_preloaded_image,
        hash_fn=lambda payload: hashlib.md5(payload).hexdigest(),
    )
    img_w, img_h = pil_img.size
    warnings: List[str] = []
    conf = _clamp_conf_value_impl(float(payload.conf) if payload.conf is not None else 0.25, warnings)
    iou = _clamp_iou_value_impl(float(payload.iou) if payload.iou is not None else 0.45, warnings)
    max_det = _clamp_max_det_value_impl(int(payload.max_det) if payload.max_det is not None else 300, warnings)
    slice_size = int(payload.slice_size) if payload.slice_size is not None else 640
    overlap = float(payload.overlap) if payload.overlap is not None else 0.2
    merge_iou = float(payload.merge_iou) if payload.merge_iou is not None else 0.5
    slice_size, overlap, merge_iou = _clamp_slice_params_impl(slice_size, overlap, merge_iou, img_w, img_h, warnings)
    _apply_expected_labelmap_warnings(payload.expected_labelmap, labelmap, warnings)
    slices, starts = _slice_image_sahi(pil_img, slice_size, overlap)
    raw_detections: List[Dict[str, Any]] = []
    fallback_device: Optional[str] = None

    def _predict_with_fallback(crop_img: Image.Image) -> Any:
        kwargs = {"conf": conf, "iou": iou, "max_det": max_det, "verbose": False}
        if fallback_device is not None:
            kwargs["device"] = fallback_device
        return model.predict(crop_img, **kwargs)

    for tile, start in zip(slices, starts):
        offset_x, offset_y = float(start[0]), float(start[1])
        crop = Image.fromarray(tile)
        try:
            results = _predict_with_fallback(crop)
        except RuntimeError as exc:  # noqa: BLE001
            msg = str(exc)
            if "CUDA" in msg or "device-side assert" in msg:
                warnings.append("yolo_cuda_error")
                _unload_detector_inference()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                try:
                    model, labelmap, task = _ensure_yolo_inference_runtime()
                    if hasattr(model, "to"):
                        try:
                            model.to("cpu")
                        except Exception:
                            pass
                    fallback_device = "cpu"
                    results = _predict_with_fallback(crop)
                    warnings.append("yolo_cuda_fallback_cpu")
                except Exception:
                    raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="yolo_cuda_error") from exc
            else:
                raise
        raw_detections.extend(_yolo_extract_detections(results, labelmap, offset_x, offset_y, img_w, img_h))
    merged = _merge_detections_nms(raw_detections, merge_iou, max_det)
    detections = [YoloRegionDetection(**item) for item in merged]
    return YoloRegionResponse(detections=detections, labelmap=labelmap, warnings=warnings or None)


def rfdetr_predict_full(payload: RfDetrFullRequest):
    model, labelmap, task = _ensure_rfdetr_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_full_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image_impl(
        payload.image_base64,
        payload.image_token,
        fetch_preloaded_fn=_fetch_preloaded_image,
        decode_image_fn=lambda b64: _decode_image_base64_impl(
            b64,
            max_bytes=BASE64_IMAGE_MAX_BYTES,
            max_dim=BASE64_IMAGE_MAX_DIM,
            allow_downscale=True,
        ),
        store_preloaded_fn=_store_preloaded_image,
        hash_fn=lambda payload: hashlib.md5(payload).hexdigest(),
    )
    img_w, img_h = pil_img.size
    warnings: List[str] = []
    conf = _clamp_conf_value_impl(float(payload.conf) if payload.conf is not None else 0.25, warnings)
    max_det = _clamp_max_det_value_impl(int(payload.max_det) if payload.max_det is not None else 300, warnings)
    _apply_expected_labelmap_warnings(payload.expected_labelmap, labelmap, warnings)
    try:
        with RFDETR_INFER_LOCK:
            results = model.predict(pil_img, threshold=conf)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"rfdetr_predict_failed:{exc}") from exc
    raw, labelmap_shifted = _rfdetr_extract_detections(results, labelmap, 0.0, 0.0, img_w, img_h)
    raw.sort(key=lambda det: float(det.get("score") or 0.0), reverse=True)
    detections = [RfDetrRegionDetection(**item) for item in raw[:max_det]]
    if labelmap_shifted:
        warnings.append("labelmap_shifted")
    return RfDetrRegionResponse(detections=detections, labelmap=labelmap, warnings=warnings or None)


def rfdetr_predict_windowed(payload: RfDetrWindowedRequest):
    model, labelmap, task = _ensure_rfdetr_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_windowed_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image_impl(
        payload.image_base64,
        payload.image_token,
        fetch_preloaded_fn=_fetch_preloaded_image,
        decode_image_fn=lambda b64: _decode_image_base64_impl(
            b64,
            max_bytes=BASE64_IMAGE_MAX_BYTES,
            max_dim=BASE64_IMAGE_MAX_DIM,
            allow_downscale=True,
        ),
        store_preloaded_fn=_store_preloaded_image,
        hash_fn=lambda payload: hashlib.md5(payload).hexdigest(),
    )
    img_w, img_h = pil_img.size
    warnings: List[str] = []
    conf = _clamp_conf_value_impl(float(payload.conf) if payload.conf is not None else 0.25, warnings)
    max_det = _clamp_max_det_value_impl(int(payload.max_det) if payload.max_det is not None else 300, warnings)
    slice_size = int(payload.slice_size) if payload.slice_size is not None else 640
    overlap = float(payload.overlap) if payload.overlap is not None else 0.2
    merge_iou = float(payload.merge_iou) if payload.merge_iou is not None else 0.5
    slice_size, overlap, merge_iou = _clamp_slice_params_impl(slice_size, overlap, merge_iou, img_w, img_h, warnings)
    _apply_expected_labelmap_warnings(payload.expected_labelmap, labelmap, warnings)
    slices, starts = _slice_image_sahi(pil_img, slice_size, overlap)
    raw_detections: List[Dict[str, Any]] = []
    labelmap_shifted = False
    for tile, start in zip(slices, starts):
        offset_x, offset_y = float(start[0]), float(start[1])
        crop = Image.fromarray(tile)
        try:
            with RFDETR_INFER_LOCK:
                results = model.predict(crop, threshold=conf)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"rfdetr_predict_failed:{exc}") from exc
        extracted, shifted = _rfdetr_extract_detections(results, labelmap, offset_x, offset_y, img_w, img_h)
        labelmap_shifted = labelmap_shifted or shifted
        raw_detections.extend(extracted)
    merged = _merge_detections_nms(raw_detections, merge_iou, max_det)
    detections = [RfDetrRegionDetection(**item) for item in merged]
    if labelmap_shifted:
        warnings.append("labelmap_shifted")
    return RfDetrRegionResponse(detections=detections, labelmap=labelmap, warnings=warnings or None)


app.include_router(
    build_rfdetr_router(
        list_variants_fn=list_rfdetr_variants,
        list_runs_fn=lambda: _list_rfdetr_runs_impl(
            job_root=RFDETR_JOB_ROOT,
            active_payload=_load_rfdetr_active(),
            load_meta_fn=lambda run_dir: _rfdetr_load_run_meta_impl(run_dir, meta_name=RFDETR_RUN_META_NAME),
            collect_artifacts_fn=lambda run_dir: _collect_rfdetr_artifacts_impl(
                run_dir,
                meta_name=RFDETR_RUN_META_NAME,
            ),
            meta_name=RFDETR_RUN_META_NAME,
        ),
        get_active_fn=_load_rfdetr_active,
        set_active_fn=set_rfdetr_active,
        download_run_fn=download_rfdetr_run,
        summary_fn=rfdetr_run_summary,
        delete_run_fn=delete_rfdetr_run,
        predict_region_fn=rfdetr_predict_region,
        predict_full_fn=rfdetr_predict_full,
        predict_windowed_fn=rfdetr_predict_windowed,
        active_request_cls=RfDetrActiveRequest,
        region_request_cls=RfDetrRegionRequest,
        full_request_cls=RfDetrFullRequest,
        windowed_request_cls=RfDetrWindowedRequest,
        region_response_cls=RfDetrRegionResponse,
    )
)


def download_yolo_run(run_id: str):
    run_dir = _yolo_run_dir_impl(
        run_id,
        create=False,
        job_root=YOLO_JOB_ROOT,
        sanitize_fn=_sanitize_yolo_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_run_not_found")
    meta = _yolo_load_run_meta_impl(run_dir, meta_name=YOLO_RUN_META_NAME)
    run_name = meta.get("config", {}).get("run_name") or meta.get("job_id") or run_id
    safe_name = _sanitize_yolo_run_id_impl(run_name)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        keep_files = set(YOLO_KEEP_FILES)
        if meta.get("head_graft"):
            for yaml_path in run_dir.glob("*.yaml"):
                keep_files.add(yaml_path.name)
        for filename in sorted(keep_files):
            path = run_dir / filename
            if path.exists():
                zf.write(path, arcname=filename)
    buffer.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{safe_name}.zip"'}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


def download_yolo_head_graft_bundle(job_id: str):
    run_dir = _yolo_run_dir_impl(
        job_id,
        create=False,
        job_root=YOLO_JOB_ROOT,
        sanitize_fn=_sanitize_yolo_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_run_not_found")
    meta = _yolo_load_run_meta_impl(run_dir, meta_name=YOLO_RUN_META_NAME)
    if not meta.get("head_graft"):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_head_graft_not_found")
    run_name = meta.get("config", {}).get("run_name") or meta.get("job_id") or job_id
    safe_name = _sanitize_yolo_run_id_impl(run_name)
    required = {"best.pt", "labelmap.txt", "head_graft_audit.jsonl", YOLO_RUN_META_NAME}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename in sorted(required):
            path = run_dir / filename
            if path.exists():
                zf.write(path, arcname=filename)
        for yaml_path in sorted(run_dir.glob("*.yaml")):
            zf.write(yaml_path, arcname=yaml_path.name)
    buffer.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{safe_name}_head_graft_bundle.zip"'}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


def delete_yolo_run(run_id: str):
    run_dir = _yolo_run_dir_impl(
        run_id,
        create=False,
        job_root=YOLO_JOB_ROOT,
        sanitize_fn=_sanitize_yolo_run_id_impl,
        http_exception_cls=HTTPException,
    )
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_run_not_found")
    try:
        shutil.rmtree(run_dir)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return {"status": "deleted", "run_id": run_id}


app.include_router(
    build_yolo_router(
        list_variants_fn=list_yolo_variants,
        list_runs_fn=lambda: _list_yolo_runs_impl(
            job_root=YOLO_JOB_ROOT,
            dataset_cache_root=YOLO_DATASET_CACHE_ROOT,
            active_payload=_load_yolo_active_impl(YOLO_ACTIVE_PATH),
            load_meta_fn=lambda run_dir: _yolo_load_run_meta_impl(run_dir, meta_name=YOLO_RUN_META_NAME),
            collect_artifacts_fn=lambda run_dir: _collect_yolo_artifacts_impl(
                run_dir,
                meta_name=YOLO_RUN_META_NAME,
            ),
            meta_name=YOLO_RUN_META_NAME,
        ),
        get_active_fn=lambda: _load_yolo_active_impl(YOLO_ACTIVE_PATH),
        set_active_fn=set_yolo_active,
        predict_region_fn=yolo_predict_region,
        predict_full_fn=yolo_predict_full,
        predict_windowed_fn=yolo_predict_windowed,
        download_run_fn=download_yolo_run,
        summary_fn=yolo_run_summary,
        head_graft_bundle_fn=download_yolo_head_graft_bundle,
        delete_run_fn=delete_yolo_run,
        active_request_cls=YoloActiveRequest,
        region_request_cls=YoloRegionRequest,
        full_request_cls=YoloFullRequest,
        windowed_request_cls=YoloWindowedRequest,
        region_response_cls=YoloRegionResponse,
    )
)


def delete_sam3_run(run_id: str, variant: str = Query("sam3"), scope: str = Query("all")):
    normalized = "sam3"
    if scope not in SAM3_STORAGE_SCOPES:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_scope")
    run_dir = _run_dir_for_request_impl(
        run_id=run_id,
        variant=normalized,
        job_root=SAM3_JOB_ROOT,
        http_exception_cls=HTTPException,
        http_400=HTTP_400_BAD_REQUEST,
        http_404=HTTP_404_NOT_FOUND,
    )
    active_paths = _active_run_paths_for_variant_impl(
        variant=normalized,
        jobs_lock=SAM3_TRAINING_JOBS_LOCK,
        jobs=SAM3_TRAINING_JOBS,
    )
    if run_dir.resolve() in active_paths:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="sam3_run_active")
    deleted, freed = _delete_run_scope_impl(
        run_dir=run_dir,
        scope=scope,
        dir_size_fn=_dir_size_bytes,
        rmtree_fn=shutil.rmtree,
    )
    return {"deleted": deleted, "freed_bytes": freed}


def promote_sam3_run(run_id: str, variant: str = Query("sam3")):
    return _promote_run(run_id, "sam3")


def list_sam3_available_models(
    variant: str = Query("sam3"),
    promoted_only: bool = Query(False),
):
    """List run checkpoints for prompt model selection."""
    runs = _list_sam3_runs_impl(
        variant="sam3",
        job_root=SAM3_JOB_ROOT,
        dataset_root=SAM3_DATASET_ROOT,
        active_paths_fn=lambda variant: _active_run_paths_for_variant_impl(
            variant=variant,
            jobs_lock=SAM3_TRAINING_JOBS_LOCK,
            jobs=SAM3_TRAINING_JOBS,
        ),
        describe_fn=lambda run_dir, variant, active_paths: _describe_run_dir_impl(
            run_dir=run_dir,
            variant=variant,
            active_paths=active_paths,
            dir_size_fn=_dir_size_bytes,
        ),
    )
    models: List[Dict[str, Any]] = []
    # Always expose the base/active env model if available
    # Env/base model entry (always listed)
    env_base_path = SAM3_CHECKPOINT_PATH if SAM3_CHECKPOINT_PATH else None
    models.append(
        {
            "id": "Base SAM3",
            "key": "base",
            # Use None so activation loads from HF if no local checkpoint is present.
            "path": env_base_path,
            "size_bytes": None,
            "promoted": False,
            "active": active_sam3_checkpoint in {None, env_base_path},
            "variant": "sam3",
            "run_path": None,
            "source": "env",
        }
    )
    # Current active model entry (if different from env/base)
    if active_sam3_checkpoint and active_sam3_checkpoint != env_base_path:
        models.append(
            {
                "id": active_sam3_metadata.get("label") or active_sam3_metadata.get("id") or "active",
                "key": f"active:{active_sam3_checkpoint}",
                "path": active_sam3_checkpoint,
                "size_bytes": None,
                "promoted": False,
                "active": True,
                "variant": "sam3",
                "run_path": None,
                "source": active_sam3_metadata.get("source") or "env",
            }
        )
    for run in runs:
        if promoted_only and not run.get("promoted"):
            continue
        if run.get("active"):
            # allow listing active too, but mark status
            pass
        ckpts = run.get("checkpoints") or []
        if not ckpts:
            continue
        # prefer last.ckpt
        chosen = None
        for ck in ckpts:
            if ck.get("file") == "last.ckpt":
                chosen = ck
                break
        if chosen is None:
            chosen = ckpts[0]
        models.append(
            {
                "id": run.get("id"),
                "path": chosen.get("path"),
                "size_bytes": chosen.get("size_bytes"),
                "promoted": run.get("promoted", False),
                "active": run.get("active", False),
                "variant": run.get("variant"),
                "run_path": run.get("path"),
            }
        )
    return models

def activate_sam3_model(payload: Sam3ModelActivateRequest):
    global active_sam3_checkpoint, active_sam3_model_id, active_sam3_metadata, active_sam3_enable_segmentation
    checkpoint_path = payload.checkpoint_path
    source = "huggingface"
    resolved_path: Optional[Path] = None
    if checkpoint_path:
        resolved_path = Path(checkpoint_path).resolve()
        if not resolved_path.exists():
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="sam3_checkpoint_not_found")
        checkpoint_path = str(resolved_path)
        source = "custom"
    enable_seg = active_sam3_enable_segmentation if checkpoint_path is not None else True
    if payload.enable_segmentation is not None:
        enable_seg = bool(payload.enable_segmentation)
    active_sam3_checkpoint = checkpoint_path
    active_sam3_enable_segmentation = enable_seg
    active_sam3_model_id = payload.label or (resolved_path.stem if resolved_path else "facebook/sam3")
    active_sam3_metadata = {
        "id": active_sam3_model_id,
        "label": payload.label or active_sam3_model_id,
        "checkpoint": active_sam3_checkpoint,
        "source": source,
        "enable_segmentation": active_sam3_enable_segmentation,
    }
    _reset_sam3_runtime()
    return {"active": active_sam3_metadata}


app.include_router(
    build_sam3_registry_router(
        list_runs_fn=lambda variant="sam3": _list_sam3_runs_impl(
            variant="sam3",
            job_root=SAM3_JOB_ROOT,
            dataset_root=SAM3_DATASET_ROOT,
            active_paths_fn=lambda variant: _active_run_paths_for_variant_impl(
                variant=variant,
                jobs_lock=SAM3_TRAINING_JOBS_LOCK,
                jobs=SAM3_TRAINING_JOBS,
            ),
            describe_fn=lambda run_dir, variant, active_paths: _describe_run_dir_impl(
                run_dir=run_dir,
                variant=variant,
                active_paths=active_paths,
                dir_size_fn=_dir_size_bytes,
            ),
        ),
        delete_run_fn=delete_sam3_run,
        promote_run_fn=promote_sam3_run,
        list_models_fn=list_sam3_available_models,
        activate_model_fn=activate_sam3_model,
        activate_cls=Sam3ModelActivateRequest,
    )
)


def create_qwen_training_job(payload: QwenTrainRequest):
    if QWEN_TRAINING_IMPORT_ERROR is not None or train_qwen_model is None:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"qwen_training_unavailable:{QWEN_TRAINING_IMPORT_ERROR}",
        )
    if not payload.dataset_id:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_id_required")
    job_id = uuid.uuid4().hex
    prep_logs: List[str] = []
    config = _build_qwen_config(payload, job_id, prep_logs)
    config_dict = asdict(config)
    job = QwenTrainingJob(job_id=job_id, config=config_dict)
    logger.info(
        "[qwen-train %s] create job accelerator=%s devices=%s dataset=%s",
        job_id[:8],
        getattr(payload, "accelerator", None) or config_dict.get("accelerator"),
        getattr(payload, "devices", None) or config_dict.get("devices"),
        payload.dataset_id,
    )
    with QWEN_TRAINING_JOBS_LOCK:
        QWEN_TRAINING_JOBS[job_id] = job
        for msg in prep_logs:
            _qwen_job_log(job, msg)
        _qwen_job_log(job, "Job queued")
    _start_qwen_training_worker(job, config)
    return {"job_id": job_id}


def list_qwen_training_jobs(request: Request):
    _prune_job_registry(QWEN_TRAINING_JOBS, QWEN_TRAINING_JOBS_LOCK)
    with QWEN_TRAINING_JOBS_LOCK:
        jobs = sorted(QWEN_TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        _log_qwen_get_request(str(request.url.path), jobs)
        return [_serialize_qwen_job_impl(job) for job in jobs]


def get_qwen_training_job(job_id: str, request: Request):
    job = _get_qwen_job(job_id)
    _log_qwen_get_request(str(request.url.path), [job])
    return _serialize_qwen_job_impl(job)


def cancel_qwen_training_job(job_id: str):
    job = _get_qwen_job(job_id)
    with QWEN_TRAINING_JOBS_LOCK:
        if job.status in {"succeeded", "failed", "cancelled"}:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="job_not_cancellable")
        if job.cancel_event.is_set():
            return {"status": job.status}
        job.cancel_event.set()
        next_status = job.status if job.status not in {"running", "queued"} else "cancelling"
        _qwen_job_update(job, status=next_status, message="Cancellation requested ...")
        return {"status": next_status}


def qwen_train_cache_size():
    cache_root = QWEN_JOB_ROOT / "splits"
    return {"bytes": _dir_size_bytes_impl(cache_root)}


def qwen_train_cache_purge():
    cache_root = QWEN_JOB_ROOT / "splits"
    deleted = _purge_directory(cache_root)
    return {"status": "ok", "deleted_bytes": deleted}


def list_qwen_models():
    default_entry = {
        "id": "default",
        "label": "Base Qwen 3",
        "type": "builtin",
        "metadata": _default_qwen_metadata(),
        "path": None,
        "created_at": None,
        "active": active_qwen_model_id == "default",
    }
    entries = _list_qwen_model_entries()
    data = [default_entry]
    for entry in entries:
        entry["active"] = entry.get("id") == active_qwen_model_id
        data.append(entry)
    return {
        "active": active_qwen_model_id,
        "models": data,
    }


def activate_qwen_model(payload: QwenModelActivateRequest):
    model_id = (payload.model_id or "").strip()
    if not model_id:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="model_id_required")
    if model_id == "default":
        _set_active_qwen_model_default()
    else:
        entry = _get_qwen_model_entry(model_id)
        if not entry:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_model_not_found")
        latest = entry.get("path")
        if not latest or not Path(latest).exists():
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_model_missing_checkpoint")
        _set_active_qwen_model_custom(model_id, Path(latest), entry.get("metadata") or {})
    return {
        "active": active_qwen_model_id,
        "metadata": active_qwen_metadata,
    }


app.include_router(
    build_qwen_models_router(
        list_models_fn=list_qwen_models,
        activate_fn=activate_qwen_model,
        activate_cls=QwenModelActivateRequest,
    )
)


app.include_router(
    build_qwen_training_router(
        create_job_fn=create_qwen_training_job,
        list_jobs_fn=list_qwen_training_jobs,
        get_job_fn=get_qwen_training_job,
        cancel_job_fn=cancel_qwen_training_job,
        cache_size_fn=qwen_train_cache_size,
        cache_purge_fn=qwen_train_cache_purge,
        request_cls=QwenTrainRequest,
    )
)


def get_active_model():
    return _current_active_payload()


def set_active_model(payload: ActiveModelRequest):
    global clf, clip_model, clip_preprocess, clip_model_name, clip_initialized
    global active_classifier_path, active_labelmap_path, active_label_list, clip_last_error
    global active_encoder_type, active_encoder_model
    global dinov3_model, dinov3_processor, dinov3_model_name, dinov3_initialized, dinov3_model_device
    global active_classifier_meta, active_head_normalize_embeddings, active_classifier_head

    classifier_path = _normalise_optional_path(payload.classifier_path) or active_classifier_path
    labelmap_path = _normalise_optional_path(payload.labelmap_path)
    labelmap_provided = "labelmap_path" in payload.__fields_set__
    requested_clip_model = _normalise_optional_path(payload.clip_model)

    if not classifier_path:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_required")
    classifier_path_abs = os.path.abspath(classifier_path)
    if not os.path.isfile(classifier_path_abs):
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="classifier_not_found")
    allowed_root = (UPLOAD_ROOT / "classifiers").resolve()
    if not str(Path(classifier_path_abs).resolve()).startswith(str(allowed_root)):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_path_not_allowed")
    _validate_upload_extension(classifier_path_abs, CLASSIFIER_ALLOWED_EXTS, "classifier_extension_not_allowed")

    try:
        new_clf = joblib.load(classifier_path_abs)
    except Exception as exc:  # noqa: BLE001
        clip_last_error = str(exc)
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"classifier_load_failed:{exc}") from exc

    meta_clip_model = None
    meta_encoder_type = "clip"
    meta_encoder_model = None
    meta_found = False
    meta_obj: Optional[Dict[str, Any]] = None
    meta_path = os.path.splitext(classifier_path_abs)[0] + ".meta.pkl"
    if os.path.exists(meta_path):
        try:
            meta_candidate = joblib.load(meta_path)
            if isinstance(meta_candidate, dict):
                meta_obj = meta_candidate
                meta_found = True
                meta_clip_model = meta_obj.get("clip_model")
                meta_encoder_type = meta_obj.get("encoder_type") or "clip"
                meta_encoder_model = meta_obj.get("encoder_model") or meta_clip_model
        except Exception:
            meta_clip_model = None
            meta_encoder_type = "clip"
            meta_encoder_model = None
            meta_found = False
    encoder_type_norm = str(meta_encoder_type or "clip").strip().lower()
    encoder_model_norm = str(meta_encoder_model or "").strip() or (str(meta_clip_model).strip() if meta_clip_model else "")

    embed_dim = None
    try:
        if isinstance(new_clf, dict):
            embed_dim = new_clf.get("embedding_dim")
            if embed_dim is None:
                layers = new_clf.get("layers")
                if isinstance(layers, list) and layers:
                    weight = layers[0].get("weight")
                    if weight is not None and hasattr(weight, "shape"):
                        embed_dim = weight.shape[1]
        else:
            coef = getattr(new_clf, "coef_", None)
            if coef is not None:
                embed_dim = coef.shape[1]
    except Exception:
        embed_dim = None
    new_clip_model = None
    new_preprocess = None
    new_dinov3_model = None
    new_dinov3_processor = None
    encoder_model_for_active = None

    if not meta_found:
        if isinstance(new_clf, dict) and str(new_clf.get("classifier_type") or new_clf.get("head_type") or "") == "mlp":
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_meta_required")
        if embed_dim is not None and int(embed_dim) not in {512, 768}:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_meta_required")
        if encoder_type_norm != "clip":
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_meta_required")

    if encoder_type_norm == "clip":
        clip_name = requested_clip_model or str(meta_clip_model or "").strip() or clip_model_name or DEFAULT_CLIP_MODEL
        inferred = _infer_clip_model_from_embedding_dim_impl(embed_dim, active_name=clip_model_name or DEFAULT_CLIP_MODEL)
        if inferred and inferred != clip_name and not requested_clip_model:
            clip_name = inferred
        if clip_name not in SUPPORTED_CLIP_MODELS:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="clip_model_not_allowed")
        need_new_clip = clip_model is None or clip_model_name != clip_name
        if need_new_clip:
            try:
                new_clip_model, new_preprocess = clip.load(clip_name, device=device)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"clip_load_failed:{exc}") from exc
        else:
            new_clip_model = clip_model
            new_preprocess = clip_preprocess
        clip_dim = getattr(getattr(new_clip_model, "visual", None), "output_dim", None)
        if embed_dim is not None and clip_dim is not None and embed_dim != clip_dim:
            inferred = _infer_clip_model_from_embedding_dim_impl(embed_dim, active_name=clip_name)
            if inferred and inferred != clip_name:
                try:
                    new_clip_model, new_preprocess = clip.load(inferred, device=device)
                except Exception as exc:  # noqa: BLE001
                    raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"clip_load_failed:{exc}") from exc
                clip_name = inferred
                clip_dim = getattr(getattr(new_clip_model, "visual", None), "output_dim", None)
                logger.warning(
                    "CLIP classifier embedding dim %s mismatched requested backbone; falling back to %s.",
                    embed_dim,
                    inferred,
                )
        if embed_dim is not None and clip_dim is not None and embed_dim != clip_dim:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"dimension_mismatch:{embed_dim}!={clip_dim}")
        encoder_model_for_active = clip_name
    elif encoder_type_norm == "dinov3":
        if not encoder_model_norm:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="encoder_model_required")
        try:
            target_device = _dinov3_resolve_device_impl(device, cuda_disabled=dinov3_cuda_disabled)
            new_dinov3_model, new_dinov3_processor = _load_dinov3_backbone(
                encoder_model_norm,
                target_device,
                raise_on_error=True,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"dinov3_load_failed:{exc}") from exc
        dino_dim = None
        try:
            cfg = getattr(new_dinov3_model, "config", None)
            dino_dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "embed_dim", None)
        except Exception:
            dino_dim = None
        if embed_dim is not None and dino_dim is not None and int(embed_dim) != int(dino_dim):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"dimension_mismatch:{embed_dim}!={dino_dim}")
        encoder_model_for_active = encoder_model_norm
    else:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="clip_encoder_type_unsupported")

    labelmap_path_abs = None
    labelmap_entries: List[str] = []
    if labelmap_path is not None:
        labelmap_path_abs = os.path.abspath(labelmap_path)
        allowed_label_roots = [
            (UPLOAD_ROOT / "labelmaps").resolve(),
            (UPLOAD_ROOT / "classifiers").resolve(),
        ]
        if not any(str(Path(labelmap_path_abs).resolve()).startswith(str(root)) for root in allowed_label_roots):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labelmap_path_not_allowed")
        _validate_upload_extension(labelmap_path_abs, LABELMAP_ALLOWED_EXTS, "labelmap_extension_not_allowed")
        labelmap_entries = _load_labelmap_file(labelmap_path_abs, strict=True)
    elif not labelmap_provided and active_labelmap_path:
        labelmap_path_abs = active_labelmap_path
        labelmap_entries = list(active_label_list)
    classes_raw = getattr(new_clf, "classes_", None)
    if classes_raw is None and isinstance(new_clf, dict):
        classes_list = [str(c) for c in list(new_clf.get("classes") or [])]
    else:
        classes_list = [str(c) for c in list(classes_raw)] if classes_raw is not None else []
    clf_classes = len(classes_list) if classes_list else None
    if clf_classes is not None:
        if not labelmap_entries:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labelmap_required_for_classifier")
        bg_indices = _clip_head_background_indices(classes_list)
        non_bg_classes = [c for idx, c in enumerate(classes_list) if idx not in bg_indices]
        label_norm = {_normalize_class_name_for_match(n) for n in labelmap_entries if n}
        clf_norm = {_normalize_class_name_for_match(n) for n in non_bg_classes if n}
        if label_norm != clf_norm:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labelmap_classifier_class_mismatch")

    with clip_lock:
        clf = new_clf
        if encoder_type_norm == "clip":
            clip_model = new_clip_model
            clip_preprocess = new_preprocess
            clip_model_name = encoder_model_for_active
            clip_initialized = True
        else:
            clip_initialized = bool(clip_model is not None and clip_preprocess is not None)
        active_classifier_path = classifier_path_abs
        active_labelmap_path = labelmap_path_abs
        active_label_list = labelmap_entries
        active_encoder_type = encoder_type_norm
        active_encoder_model = encoder_model_for_active
        active_classifier_meta = dict(meta_obj) if isinstance(meta_obj, dict) else {}
        active_head_normalize_embeddings = _resolve_active_head_normalize_embeddings_impl(
            meta_obj,
            new_clf,
            default=True,
            resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
        )
        try:
            active_classifier_head = _load_clip_head_from_classifier_impl(
                Path(classifier_path_abs),
                joblib_load_fn=joblib.load,
                http_exception_cls=HTTPException,
                clip_head_background_indices_fn=_clip_head_background_indices,
                resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings_impl,
                infer_clip_model_fn=_infer_clip_model_from_embedding_dim_impl,
                active_clip_model_name=clip_model_name,
                default_clip_model=DEFAULT_CLIP_MODEL,
                logger=logger,
            )
        except Exception:
            active_classifier_head = None
        if payload.logit_adjustment_inference is not None and isinstance(active_classifier_head, dict):
            active_classifier_head["logit_adjustment_inference"] = bool(payload.logit_adjustment_inference)
        clip_last_error = None
    if encoder_type_norm == "dinov3":
        with dinov3_lock:
            dinov3_model = new_dinov3_model
            dinov3_processor = new_dinov3_processor
            dinov3_model_name = encoder_model_for_active
            dinov3_model_device = target_device
            dinov3_initialized = bool(dinov3_model is not None and dinov3_processor is not None)

    return _current_active_payload()


app.include_router(
    build_clip_active_model_router(
        get_fn=get_active_model,
        set_fn=set_active_model,
        request_cls=ActiveModelRequest,
        response_cls=ActiveModelResponse,
    )
)


# note this one is actually not used. For a while I thought it would be cool to send a smaller crop to SAM but I'm not sure it makes sense since
# now I'm caching / checking the file that is currently loaded in the predictor and not updating on every call so it's actually waaaay faster and we have the whole image
# ---------------------------------------------------------------------------
# SAM preload endpoint
# ---------------------------------------------------------------------------

def sam_preload(payload: SamPreloadRequest):
    variant = _default_variant(payload.sam_variant)
    try:
        slot_name = predictor_manager.resolve_slot(payload.slot, allow_disabled_fallback=False)
        return sam_preload_manager.submit(
            variant=variant,
            generation=payload.preload_generation,
            image_token=payload.image_token,
            image_base64=payload.image_base64,
            image_name=payload.image_name,
            slot=slot_name,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"sam_preload_failed:{exc}") from exc


app.include_router(
    build_sam_preload_router(
        preload_fn=sam_preload,
        request_cls=SamPreloadRequest,
        response_cls=SamPreloadResponse,
    )
)


def _sam_slots_status() -> List[SamSlotStatus]:
    return predictor_manager.status()


def _sam_activate_slot(payload: SamActivateRequest) -> SamActivateResponse:
    variant = _default_variant(payload.sam_variant)
    slot = predictor_manager.get_slot_for_image(payload.image_name, variant)
    if slot is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="slot_not_found")
    promoted = predictor_manager.promote_slot(slot.name)
    if not promoted and slot.name != "current":
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="slot_busy")
    return SamActivateResponse(status="promoted", slot="current", token=slot.token)


app.include_router(
    build_sam_slots_router(
        status_fn=_sam_slots_status,
        activate_fn=_sam_activate_slot,
        status_cls=List[SamSlotStatus],
        activate_req_cls=SamActivateRequest,
        activate_resp_cls=SamActivateResponse,
    )
)


def _predictor_settings_payload() -> PredictorSettings:
    min_cap, max_cap = predictor_manager.capacity_limits()
    current_cap = predictor_manager.get_capacity()
    active = predictor_manager.active_slot_count()
    loaded = predictor_manager.loaded_slot_count()
    image_memory = predictor_manager.total_image_memory_bytes()
    gpu_total_mb = None
    gpu_free_mb = None
    gpu_cc = None
    gpu_count = None
    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            gpu_total_mb = _bytes_to_mb(int(total_bytes))
            gpu_free_mb = _bytes_to_mb(int(free_bytes))
            major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
            gpu_cc = f"{major}.{minor}"
            gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_total_mb = None
            gpu_free_mb = None
            gpu_count = None
    vm = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    process_mb = _bytes_to_mb(process.memory_info().rss)
    total_mb = _bytes_to_mb(int(vm.total))
    available_mb = _bytes_to_mb(int(vm.available))
    image_mb = _bytes_to_mb(image_memory)
    return PredictorSettings(
        max_predictors=current_cap,
        min_predictors=min_cap,
        max_supported_predictors=max_cap,
        active_predictors=active,
        loaded_predictors=loaded,
        process_ram_mb=process_mb,
        total_ram_mb=total_mb,
        available_ram_mb=available_mb,
        image_ram_mb=image_mb,
        gpu_total_mb=gpu_total_mb,
        gpu_free_mb=gpu_free_mb,
        gpu_compute_capability=gpu_cc,
        gpu_device_count=gpu_count,
    )


def _update_predictor_settings(payload: PredictorSettingsUpdate) -> PredictorSettings:
    min_cap, max_cap = predictor_manager.capacity_limits()
    try:
        requested = int(payload.max_predictors)
    except Exception:
        requested = min_cap
    normalized = max(min_cap, min(max_cap, requested))
    predictor_manager.set_capacity(normalized)
    return _predictor_settings_payload()


app.include_router(
    build_predictor_settings_router(
        get_payload_fn=_predictor_settings_payload,
        update_fn=_update_predictor_settings,
        settings_cls=PredictorSettings,
        update_cls=PredictorSettingsUpdate,
    )
)


def _gpu_status_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = {"available": False, "device_count": 0, "devices": []}
    if not torch.cuda.is_available():
        return payload
    devices: List[Dict[str, Any]] = []
    try:
        device_count = torch.cuda.device_count()
        payload["available"] = True
        payload["device_count"] = device_count
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            device_info: Dict[str, Any] = {
                "index": idx,
                "name": props.name,
                "total_mb": _bytes_to_mb(int(props.total_memory)),
                "compute_capability": f"{props.major}.{props.minor}",
            }
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                device_info["free_mb"] = _bytes_to_mb(int(free_bytes))
                device_info["total_mb"] = _bytes_to_mb(int(total_bytes))
            except Exception:
                device_info["free_mb"] = None
            devices.append(device_info)
    except Exception as exc:  # noqa: BLE001
        payload["error"] = str(exc)
    payload["devices"] = devices
    return payload


def _storage_check_payload() -> Dict[str, Any]:
    roots = {
        "uploads": UPLOAD_ROOT,
        "dataset_registry": DATASET_REGISTRY_ROOT,
        "sam3_datasets": SAM3_DATASET_ROOT,
        "qwen_datasets": QWEN_DATASET_ROOT,
        "sam3_runs": SAM3_JOB_ROOT,
        "qwen_runs": QWEN_JOB_ROOT,
        "yolo_runs": YOLO_JOB_ROOT,
        "rfdetr_runs": RFDETR_JOB_ROOT,
        "calibration_jobs": CALIBRATION_ROOT,
        "calibration_cache": CALIBRATION_CACHE_ROOT,
        "prepass_recipes": PREPASS_RECIPE_ROOT,
        "clip_uploads": CLIP_DATASET_UPLOAD_ROOT,
        "dataset_uploads": DATASET_UPLOAD_ROOT,
    }
    results: List[Dict[str, Any]] = []
    ok = True
    for name, root in roots.items():
        entry = {"name": name, "path": str(root), "ok": True, "error": None}
        try:
            root.mkdir(parents=True, exist_ok=True)
            test_path = root / f".write_test_{uuid.uuid4().hex}"
            test_path.write_text("ok", encoding="utf-8")
            test_path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            entry["ok"] = False
            entry["error"] = str(exc)
            ok = False
        results.append(entry)
    return {"ok": ok, "roots": results}


def _system_health_summary() -> Dict[str, Any]:
    storage = _storage_check_payload()
    gpu = _gpu_status_payload()
    summary: Dict[str, Any] = {
        "ok": bool(storage.get("ok", False)),
        "gpu": gpu,
        "storage": storage,
        "datasets": {},
        "models": {},
        "errors": [],
    }
    try:
        datasets = _list_all_datasets()
        summary["datasets"] = {"count": len(datasets)}
    except Exception as exc:  # noqa: BLE001
        summary["datasets"] = {"count": None}
        summary["errors"].append(f"datasets_list_failed:{exc}")
        summary["ok"] = False
    try:
        summary["models"]["qwen"] = qwen_status()
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"qwen_status_failed:{exc}")
        summary["ok"] = False
    try:
        summary["models"]["sam3_runs"] = len(
            _list_sam3_runs_impl(
                variant="sam3",
                job_root=SAM3_JOB_ROOT,
                dataset_root=SAM3_DATASET_ROOT,
                active_paths_fn=lambda variant: _active_run_paths_for_variant_impl(
                    variant=variant,
                    jobs_lock=SAM3_TRAINING_JOBS_LOCK,
                    jobs=SAM3_TRAINING_JOBS,
                ),
                describe_fn=lambda run_dir, variant, active_paths: _describe_run_dir_impl(
                    run_dir=run_dir,
                    variant=variant,
                    active_paths=active_paths,
                    dir_size_fn=_dir_size_bytes,
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"sam3_runs_failed:{exc}")
        summary["ok"] = False
    try:
        summary["models"]["qwen_models"] = len(_list_qwen_model_entries())
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"qwen_models_failed:{exc}")
        summary["ok"] = False
    try:
        summary["models"]["clip_classifiers"] = len(
            _list_clip_classifiers_impl(
                upload_root=UPLOAD_ROOT,
                classifier_exts=CLASSIFIER_ALLOWED_EXTS,
                labelmap_exts=LABELMAP_ALLOWED_EXTS,
                path_is_within_root_fn=_path_is_within_root_impl,
                joblib_load_fn=joblib.load,
                resolve_clip_labelmap_path_fn=lambda path_str, root_hint=None: _resolve_clip_labelmap_path_impl(
                    path_str,
                    root_hint=root_hint,
                    upload_root=UPLOAD_ROOT,
                    labelmap_exts=LABELMAP_ALLOWED_EXTS,
                    path_is_within_root_fn=_path_is_within_root_impl,
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"clip_classifiers_failed:{exc}")
        summary["ok"] = False
    summary["models"]["yolo_variants"] = len(YOLO_VARIANTS)
    summary["models"]["rfdetr_variants"] = len(RFDETR_VARIANTS)
    return summary


app.include_router(
    build_system_router(
        gpu_payload_fn=_gpu_status_payload,
        storage_payload_fn=_storage_check_payload,
        health_summary_fn=_system_health_summary,
    )
)


def qwen_status():
    dependency_error = str(QWEN_IMPORT_ERROR) if QWEN_IMPORT_ERROR else None
    device_guess = qwen_device
    pending_error = qwen_last_error
    if not device_guess and not dependency_error:
        try:
            device_guess = _resolve_qwen_device_impl(QWEN_DEVICE_PREF, torch_module=torch)
        except RuntimeError as exc:  # noqa: BLE001
            pending_error = str(exc)
            device_guess = None
    return {
        "available": dependency_error is None,
        "loaded": qwen_model is not None,
        "model_name": QWEN_MODEL_NAME,
        "model_family": (active_qwen_metadata or {}).get("model_family", "qwen3"),
        "device": device_guess,
        "max_new_tokens": QWEN_MAX_NEW_TOKENS,
        "min_pixels": QWEN_MIN_PIXELS,
        "max_pixels": QWEN_MAX_PIXELS,
        "last_error": pending_error,
        "dependency_error": dependency_error,
        "active_model": active_qwen_model_id,
        "active_metadata": active_qwen_metadata,
    }


def qwen_settings():
    return QwenRuntimeSettings(trust_remote_code=QWEN_TRUST_REMOTE_CODE)


def update_qwen_settings(payload: QwenRuntimeSettingsUpdate):
    global QWEN_TRUST_REMOTE_CODE
    if payload.trust_remote_code is not None:
        desired = bool(payload.trust_remote_code)
        if desired != QWEN_TRUST_REMOTE_CODE:
            QWEN_TRUST_REMOTE_CODE = desired
            _unload_qwen_runtime()
    return QwenRuntimeSettings(trust_remote_code=QWEN_TRUST_REMOTE_CODE)


app.include_router(
    build_runtime_router(
        unload_all_fn=lambda: _unload_inference_runtimes_impl(
            unload_non_qwen_fn=lambda: _unload_non_qwen_runtimes_impl(
                predictor_manager=predictor_manager,
                unload_sam3_text_fn=_unload_sam3_text_runtime,
                suspend_clip_fn=_suspend_clip_backbone,
                unload_dinov3_fn=_unload_dinov3_backbone,
                unload_detector_fn=_unload_detector_inference,
                torch_module=torch,
                logger=logger,
            ),
            unload_qwen_fn=_unload_qwen_runtime,
            torch_module=torch,
        ),
    )
)

app.include_router(
    build_qwen_status_router(
        status_fn=qwen_status,
        get_settings_fn=qwen_settings,
        update_settings_fn=update_qwen_settings,
        unload_fn=lambda: (_unload_qwen_runtime() or {"status": "unloaded"}),
        settings_cls=QwenRuntimeSettings,
        update_cls=QwenRuntimeSettingsUpdate,
    )
)


def qwen_infer(payload: QwenInferenceRequest):
    prompt_type = payload.prompt_type.lower()
    if prompt_type not in {"bbox", "point", "bbox_sam"}:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_prompt_type")
    pil_img, np_img, token = resolve_image_payload(
        payload.image_base64,
        payload.image_token,
        getattr(payload, "sam_variant", None),
    )
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
            get_config_fn=lambda: _get_qwen_prompt_config_impl(qwen_prompt_config, qwen_config_lock),
            http_exception_cls=HTTPException,
            http_422=HTTP_422_UNPROCESSABLE_ENTITY,
        )
    try:
        qwen_text, proc_w, proc_h = _run_qwen_inference(final_prompt, pil_img)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_inference_failed:{exc}") from exc
    print("[Qwen prompt]", final_prompt)
    print("[Qwen raw output]", qwen_text)
    warnings: List[str] = []
    try:
        _, items = _extract_qwen_json_block(qwen_text)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        warnings.append(f"parse_error:{detail}")
        print(f"[Qwen parse error] {detail}; raw text follows:\n{qwen_text}")
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
        boxes = _qwen_bbox_results(normalized_items, proc_w, proc_h, pil_img.width, pil_img.height, limit=limit)
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
    return QwenInferenceResponse(
        boxes=boxes,
        raw_response=qwen_text,
        prompt=final_prompt,
        prompt_type=prompt_type,  # type: ignore[arg-type]
        warnings=warnings,
        image_token=token,
    )


def qwen_caption(payload: QwenCaptionRequest):
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
        if payload.unload_others and not fast_mode:
            _unload_non_qwen_runtimes_impl(
                predictor_manager=predictor_manager,
                unload_sam3_text_fn=_unload_sam3_text_runtime,
                suspend_clip_fn=_suspend_clip_backbone,
                unload_dinov3_fn=_unload_dinov3_backbone,
                unload_detector_fn=_unload_detector_inference,
                torch_module=torch,
                logger=logger,
            )
        pil_img, _, _ = resolve_image_payload(payload.image_base64, payload.image_token, None)
        user_prompt = (payload.user_prompt or "").strip()
        include_counts = bool(payload.include_counts)
        include_coords = bool(payload.include_coords)
        max_boxes = payload.max_boxes if payload.max_boxes is not None else 0
        max_new_tokens = payload.max_new_tokens if payload.max_new_tokens is not None else 128
        label_hints = payload.label_hints or []
        allowed_labels = _allowed_caption_labels_impl(label_hints)
        image_width = payload.image_width or pil_img.width
        image_height = payload.image_height or pil_img.height
        caption_mode = payload.caption_mode or "full"
        restrict_to_labels = payload.restrict_to_labels if payload.restrict_to_labels is not None else True
        caption_all_windows = True if caption_mode == "windowed" else bool(payload.caption_all_windows)
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
        )
        glossary_line = ""
        if payload.labelmap_glossary:
            glossary_line = (
                "Glossary (label synonyms): "
                f"{payload.labelmap_glossary}. "
                "Use glossary terms as optional synonym hints; do not copy the glossary verbatim. "
                "Never output labelmap class names (especially tokens with underscores)."
            )
            prompt_text = f"{prompt_text}\n{glossary_line}"
        base_model_id = (active_qwen_metadata or {}).get("model_id") or QWEN_MODEL_NAME
        default_caption_model_id = base_model_id
        variant = payload.model_variant or "auto"
        model_id_override = payload.model_id or ""
        if model_id_override:
            desired_model_id = model_id_override
        else:
            desired_model_id = _resolve_qwen_variant_model_id_impl(base_model_id, variant)
        if active_qwen_model_path and desired_model_id != base_model_id:
            logger.info(
                "[qwen-caption] using base model override (%s) while adapter %s is active",
                desired_model_id,
                active_qwen_model_id,
            )
        caption_base_model_id = desired_model_id if model_id_override else base_model_id
        final_only = bool(payload.final_answer_only)
        two_stage = bool(payload.two_stage_refine)
        is_thinking = "Thinking" in desired_model_id or variant == "Thinking"
        decode_params = _resolve_qwen_caption_decode_impl(payload, is_thinking)
        deterministic_decode = {"do_sample": False}
        if is_thinking:
            prompt_text = _adjust_prompt_for_thinking_impl(prompt_text)
        # Keep caption max_new_tokens consistent across full/windowed/refine paths; cap at 2000.
        # Avoid per-path caps here (we previously caused repeated CUDA asserts by diverging).
        if is_thinking:
            max_new_tokens = max(max_new_tokens, 2000)
        max_new_tokens = min(max_new_tokens, 2000)
        refine_max_tokens = max_new_tokens
        system_prompt = (
            "You are a detailed captioning assistant. Use the image as truth. "
            "Provide a rich, multi-sentence caption when there is a lot to see. "
            "Do not mention labels, hints, bounding boxes, coordinates, or that counts were provided. "
            "Do not output labelmap tags (e.g., light_vehicle). Use natural words like car, van, SUV. "
            "Avoid any label tokens that contain underscores. "
            "If the hints conflict with the image, mention the uncertainty briefly."
        )
        system_prompt = f"{system_prompt} Respond in English only."
        if final_only:
            system_prompt = f"{system_prompt} Return only the final caption. Do not include reasoning or preamble."
        if final_only and is_thinking and not two_stage:
            system_prompt = f"{system_prompt} Respond with exactly <final>...</final> and nothing else."
        use_caption_cache = True
        if active_qwen_model_path and not model_id_override and desired_model_id == base_model_id and variant == "auto":
            use_caption_cache = False

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
        ) -> str:
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
                run_qwen_inference_fn=_run_qwen_inference,
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
        ) -> str:
            return _run_qwen_caption_merge_impl(
                draft,
                windows,
                pil_img=pil_img,
                base_model_id=base_model_id,
                runtime_resolver=runtime_resolver,
                max_new_tokens=max_new_tokens,
                glossary_line=glossary_line,
                run_qwen_inference_fn=_run_qwen_inference,
                resolve_variant_fn=_resolve_qwen_variant_model_id_impl,
                extract_caption_fn=_extract_caption_from_text_impl,
                sanitize_caption_fn=_sanitize_qwen_caption_impl,
            )

        def resolve_main_runtime() -> Tuple[Any, Any]:
            if model_id_override:
                return get_runtime(desired_model_id)
            if use_caption_cache:
                return get_runtime(desired_model_id)
            return get_runtime(None)

        windowed_captions: List[Tuple[int, int, int, str]] = []
        cleanup_count = 0
        refine_count = 0
        merge_count = 0
        if caption_mode == "windowed":
            overlap = _resolve_qwen_window_overlap_impl(payload.window_overlap, default_overlap=QWEN_WINDOW_DEFAULT_OVERLAP)
            window_size = _resolve_qwen_window_size_impl(
                None,
                image_width,
                image_height,
                overlap=overlap,
                default_size=QWEN_WINDOW_DEFAULT_SIZE,
                default_overlap=QWEN_WINDOW_DEFAULT_OVERLAP,
            )
            force_two = True
            x_positions = _window_positions_impl(image_width, window_size, overlap, force_two=force_two)
            y_positions = _window_positions_impl(image_height, window_size, overlap, force_two=force_two)
            grouped_hints = _group_hints_by_window(label_hints, x_positions, y_positions, window_size)
            window_model_id = desired_model_id
            window_base_model_id = window_model_id
            window_is_thinking = "Thinking" in window_model_id
            for y0 in y_positions:
                for x0 in x_positions:
                    window_hints = grouped_hints.get((x0, y0), [])
                    if not window_hints and not caption_all_windows:
                        continue
                    window_allowed = _allowed_caption_labels_impl(window_hints)
                    window_glossary_map = _caption_glossary_map_impl(
                        payload.labelmap_glossary,
                        [hint.label for hint in window_hints if hint.label],
                    )
                    window_allowed_prompt = (
                        [_caption_preferred_label_impl(label, window_glossary_map) for label in window_allowed]
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
                    )
                    window_prompt = (
                        f"Window region in full image: [{x0}, {y0}] to [{x0 + window_size}, {y0 + window_size}].\n"
                        "Focus only on this region.\n"
                        "Write 1-3 detailed sentences about this region only. No reasoning or preamble. "
                        "Do not mention labels, hints, counts, or coordinates. "
                        "Do not output labelmap tags (e.g., light_vehicle); use natural words like car or van. "
                        "Avoid any token with underscores.\n"
                        f"{window_prompt}"
                    )
                    if glossary_line:
                        window_prompt = f"{window_prompt}\n{glossary_line}"
                    if restrict_to_labels and window_allowed_prompt:
                        window_prompt = (
                            f"{window_prompt}\nAllowed classes: {', '.join(window_allowed_prompt)}. "
                            "Do not introduce any other entity types."
                        )
                    if window_is_thinking:
                        window_prompt = _adjust_prompt_for_thinking_impl(window_prompt)
                    window_img = pil_img.crop((x0, y0, x0 + window_size, y0 + window_size))
                    window_max_tokens = max_new_tokens
                    qwen_text, _, _ = _run_qwen_inference(
                        window_prompt,
                        window_img,
                        max_new_tokens=window_max_tokens,
                        system_prompt_override=system_prompt,
                        runtime_override=get_runtime(window_model_id),
                        decode_override=decode_params,
                    )
                    window_caption, _ = _extract_caption_from_text_impl(qwen_text, marker=None)
                    window_caption = _sanitize_qwen_caption_impl(window_caption)
                    if window_is_thinking and _thinking_caption_needs_cleanup_impl(window_caption, qwen_text):
                        cleanup_model = _resolve_qwen_variant_model_id_impl(window_base_model_id, "Instruct")
                        window_caption = _caption_cleanup(
                            window_caption,
                            window_img,
                            window_max_tokens,
                            window_base_model_id,
                            use_caption_cache,
                            model_id_override=cleanup_model,
                            runtime_override=get_runtime(cleanup_model),
                            allowed_labels=window_allowed_prompt if restrict_to_labels and window_allowed_prompt else None,
                            strict=True,
                            minimal_edit=True,
                        )
                        cleanup_count += 1
                    if _caption_is_degenerate_impl(window_caption):
                        cleanup_model = _resolve_qwen_variant_model_id_impl(window_base_model_id, "Instruct")
                        window_caption = _caption_cleanup(
                            window_caption,
                            window_img,
                            window_max_tokens,
                            window_base_model_id,
                            use_caption_cache,
                            model_id_override=cleanup_model,
                            runtime_override=get_runtime(cleanup_model),
                            allowed_labels=window_allowed_prompt if restrict_to_labels and window_allowed_prompt else None,
                            strict=True,
                            minimal_edit=True,
                        )
                        cleanup_count += 1
                    if _caption_needs_completion_impl(window_caption) or _caption_has_meta_impl(window_caption):
                        cleanup_model = _resolve_qwen_variant_model_id_impl(window_base_model_id, "Instruct")
                        window_caption = _caption_cleanup(
                            window_caption,
                            window_img,
                            window_max_tokens,
                            window_base_model_id,
                            use_caption_cache,
                            model_id_override=cleanup_model,
                            runtime_override=get_runtime(cleanup_model),
                            allowed_labels=window_allowed_prompt if restrict_to_labels and window_allowed_prompt else None,
                            strict=True,
                            minimal_edit=True,
                        )
                        cleanup_count += 1
                    needs_refine, missing = _caption_needs_refine_impl(
                        window_caption,
                        window_counts,
                        detailed_mode=True,
                        include_counts=include_counts,
                        glossary_map=window_glossary_map,
                    )
                    if needs_refine:
                        refine_model = _resolve_qwen_variant_model_id_impl(window_base_model_id, "Instruct")
                        allowed_note = ""
                        if restrict_to_labels and window_allowed_prompt:
                            allowed_note = (
                                f"Allowed classes: {', '.join(window_allowed_prompt)}. "
                                "Do not introduce any other entity types."
                            )
                        elif not restrict_to_labels:
                            allowed_note = "You may mention additional visible objects beyond the hints."
                        missing_note = (
                            f"Ensure the caption mentions: {', '.join(missing)}."
                            if missing
                            else "Ensure all labeled classes in this window are mentioned."
                        )
                        refine_prompt = f"{window_prompt}\nDraft caption: {window_caption}\n{missing_note}"
                        if allowed_note:
                            refine_prompt = f"{refine_prompt}\n{allowed_note}"
                        refine_prompt = (
                            f"{refine_prompt}\n"
                            "Edit the draft with minimal changes. Do not introduce new objects or actions. "
                            "Return only a concise, complete caption (1-3 sentences) with no coordinates."
                        )
                        refine_system = (
                            "You are a concise captioning assistant. Return only the final caption in English."
                        )
                        refine_text, _, _ = _run_qwen_inference(
                            refine_prompt,
                            window_img,
                            max_new_tokens=refine_max_tokens,
                            system_prompt_override=refine_system,
                            runtime_override=get_runtime(refine_model),
                            decode_override=deterministic_decode,
                        )
                        window_caption, _ = _extract_caption_from_text_impl(refine_text, marker=None)
                        window_caption = _sanitize_qwen_caption_impl(window_caption)
                        refine_count += 1
                    if window_caption:
                        windowed_captions.append((x0, y0, window_size, window_caption))
                        _emit_caption_window(x0, y0, window_size, window_caption)
            if windowed_captions:
                window_lines = ["Close-up observations from subregions (use these to enrich the final caption):"]
                for x0, y0, size, caption in windowed_captions:
                    x_center = x0 + size / 2.0
                    y_center = y0 + size / 2.0
                    horiz = "left" if x_center < image_width / 3.0 else "right" if x_center > image_width * 2 / 3.0 else "center"
                    vert = "top" if y_center < image_height / 3.0 else "bottom" if y_center > image_height * 2 / 3.0 else "middle"
                    region = f"{vert}-{horiz}"
                    window_lines.append(
                        f"- {region} ([{x0},{y0},{x0 + size},{y0 + size}]): {caption}"
                    )
                if include_counts and restrict_to_labels:
                    window_lines.append(
                        "Now describe the full image in detail. Use all labeled object counts and the close-up observations. "
                        "Mention every class that appears in the hints, and summarize repetitive objects (e.g., many cars as a parking lot) "
                        "unless only a few are present or a specific action stands out. "
                        "Do not mention labels, hints, counts, or coordinates."
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
                    "In the final caption, preserve specific details from the windows "
                    "(e.g., people counts, actions, and notable objects). "
                    "Do not compress away window details; longer captions are preferred."
                )
                prompt_text = f"{prompt_text}\n" + "\n".join(window_lines)
        if two_stage and is_thinking:
            draft_prompt = (
                "Step 1: Look at the image and form a draft caption.\n"
                "Respond with: DRAFT: <caption>"
            )
            draft_system = f"{system_prompt} Return only a line starting with 'DRAFT:'."
            draft_text, _, _ = _run_qwen_inference(
                draft_prompt,
                pil_img,
                max_new_tokens=max_new_tokens,
                system_prompt_override=draft_system,
                runtime_override=resolve_main_runtime(),
                decode_override=decode_params,
            )
            draft_caption, _ = _extract_caption_from_text_impl(draft_text, marker="DRAFT")
            draft_caption = _sanitize_qwen_caption_impl(draft_caption)
            allowed_note = ""
            if restrict_to_labels and allowed_labels_prompt:
                allowed_note = (
                    f"Allowed classes: {', '.join(allowed_labels_prompt)}. Do not introduce any other entity types."
                )
            elif not restrict_to_labels:
                allowed_note = "You may mention additional visible objects beyond the hints."
            refine_prompt = f"{prompt_text}\nDraft caption: {draft_caption}"
            if allowed_note:
                refine_prompt = f"{refine_prompt}\n{allowed_note}"
            refine_prompt = (
                f"{refine_prompt}\n"
                "Edit the draft with minimal changes. Do not introduce new objects or actions. "
                "Return only the final caption."
            )
            refine_system = (
                "You are a captioning assistant. Use the image as truth. "
                "Return only the final caption. Respond in English only."
            )
            refine_model = _resolve_qwen_variant_model_id_impl(caption_base_model_id, "Instruct")
            qwen_text, _, _ = _run_qwen_inference(
                refine_prompt,
                pil_img,
                max_new_tokens=refine_max_tokens,
                system_prompt_override=refine_system,
                runtime_override=get_runtime(refine_model),
                decode_override=deterministic_decode,
            )
            caption_text, _ = _extract_caption_from_text_impl(qwen_text, marker=None)
            if final_only or is_thinking:
                caption_text = _sanitize_qwen_caption_impl(caption_text)
        else:
            qwen_text, _, _ = _run_qwen_inference(
                prompt_text,
                pil_img,
                max_new_tokens=max_new_tokens,
                system_prompt_override=system_prompt,
                runtime_override=resolve_main_runtime(),
                decode_override=decode_params,
            )
            caption_text, _ = _extract_caption_from_text_impl(qwen_text, marker=None)
            if final_only or is_thinking:
                caption_text = _sanitize_qwen_caption_impl(caption_text)
            if is_thinking and _thinking_caption_needs_cleanup_impl(caption_text, qwen_text):
                cleanup_model = _resolve_qwen_variant_model_id_impl(caption_base_model_id, "Instruct")
                caption_text = _caption_cleanup(
                    caption_text,
                    pil_img,
                    refine_max_tokens,
                    caption_base_model_id,
                    use_caption_cache,
                    model_id_override=cleanup_model,
                    runtime_override=get_runtime(cleanup_model),
                    allowed_labels=allowed_labels_prompt if restrict_to_labels and allowed_labels_prompt else None,
                    strict=True,
                    minimal_edit=True,
                )
                cleanup_count += 1
        if caption_mode == "windowed" and windowed_captions and caption_text:
            merge_tokens = min(refine_max_tokens, 256)
            caption_text = _caption_merge(
                caption_text,
                windowed_captions,
                pil_img=pil_img,
                base_model_id=caption_base_model_id,
                runtime_resolver=get_runtime,
                max_new_tokens=merge_tokens,
                glossary_line=glossary_line or None,
            )
            merge_count += 1
            if final_only or is_thinking:
                caption_text = _sanitize_qwen_caption_impl(caption_text)
        if _caption_is_degenerate_impl(caption_text):
            cleanup_model = _resolve_qwen_variant_model_id_impl(caption_base_model_id, "Instruct")
            caption_text = _caption_cleanup(
                caption_text,
                pil_img,
                refine_max_tokens,
                caption_base_model_id,
                use_caption_cache,
                model_id_override=cleanup_model,
                runtime_override=get_runtime(cleanup_model),
                allowed_labels=allowed_labels_prompt if restrict_to_labels and allowed_labels_prompt else None,
                strict=True,
                minimal_edit=True,
            )
            cleanup_count += 1
        if _caption_needs_completion_impl(caption_text) or _caption_has_meta_impl(caption_text):
            cleanup_model = _resolve_qwen_variant_model_id_impl(caption_base_model_id, "Instruct")
            caption_text = _caption_cleanup(
                caption_text,
                pil_img,
                refine_max_tokens,
                caption_base_model_id,
                use_caption_cache,
                model_id_override=cleanup_model,
                runtime_override=get_runtime(cleanup_model),
                allowed_labels=allowed_labels_prompt if restrict_to_labels and allowed_labels_prompt else None,
                strict=True,
                minimal_edit=True,
            )
            cleanup_count += 1
        if caption_mode == "windowed" and "4B" in desired_model_id and _caption_needs_short_form_impl(caption_text):
            cleanup_model = _resolve_qwen_variant_model_id_impl(caption_base_model_id, "Instruct")
            caption_text = _caption_cleanup(
                caption_text,
                pil_img,
                refine_max_tokens,
                caption_base_model_id,
                use_caption_cache,
                model_id_override=cleanup_model,
                runtime_override=get_runtime(cleanup_model),
                allowed_labels=allowed_labels_prompt if restrict_to_labels and allowed_labels_prompt else None,
                strict=True,
                minimal_edit=True,
            )
            cleanup_count += 1
        needs_refine, missing = _caption_needs_refine_impl(
            caption_text,
            counts,
            detailed_mode=detailed_mode,
            include_counts=include_counts,
            glossary_map=glossary_map,
        )
        if needs_refine:
            refine_model = _resolve_qwen_variant_model_id_impl(caption_base_model_id, "Instruct")
            allowed_note = ""
            if restrict_to_labels and allowed_labels_prompt:
                allowed_note = (
                    f"Allowed classes: {', '.join(allowed_labels_prompt)}. Do not introduce any other entity types."
                )
            elif not restrict_to_labels:
                allowed_note = "You may mention additional visible objects beyond the hints."
            missing_note = (
                f"Ensure the caption mentions: {', '.join(missing)}."
                if missing
                else "Ensure all labeled classes are mentioned."
            )
            refine_prompt = f"{prompt_text}\nDraft caption: {caption_text}\n{missing_note}"
            if allowed_note:
                refine_prompt = f"{refine_prompt}\n{allowed_note}"
            refine_prompt = (
                f"{refine_prompt}\n"
                "Edit the draft with minimal changes. Do not introduce new objects or actions. "
                "Return only the final caption with no coordinates."
            )
            refine_system = "You are a captioning assistant. Return only the final caption in English."
            refine_text, _, _ = _run_qwen_inference(
                refine_prompt,
                pil_img,
                max_new_tokens=refine_max_tokens,
                system_prompt_override=refine_system,
                runtime_override=get_runtime(refine_model),
                decode_override=deterministic_decode,
            )
            caption_text, _ = _extract_caption_from_text_impl(refine_text, marker=None)
            caption_text = _sanitize_qwen_caption_impl(caption_text)
            refine_count += 1
        if caption_text and _caption_needs_english_rewrite_impl(caption_text):
            rewrite_model = _resolve_qwen_variant_model_id_impl(base_model_id, "Instruct")
            rewrite_prompt = (
                "Rewrite the caption in English only, preserving meaning and brevity.\n"
                f"Caption: {caption_text}"
            )
            rewrite_system = "Return only the rewritten caption in English."
            rewrite_text, _, _ = _run_qwen_inference(
                rewrite_prompt,
                pil_img,
                max_new_tokens=refine_max_tokens,
                system_prompt_override=rewrite_system,
                runtime_override=get_runtime(rewrite_model),
                decode_override=deterministic_decode,
            )
            caption_text, _ = _extract_caption_from_text_impl(rewrite_text, marker=None)
            if final_only or is_thinking:
                caption_text = _sanitize_qwen_caption_impl(caption_text)
        response = QwenCaptionResponse(
            caption=caption_text,
            used_counts=counts,
            used_boxes=used_boxes,
            truncated=truncated,
        )
        word_count = len(caption_text.split()) if caption_text else 0
        logger.info(
            "[qwen-caption] hints=%s used=%s truncated=%s variant=%s model=%s final_only=%s windows=%s cleanup=%s refine=%s merge=%s words=%s",
            len(payload.label_hints or []),
            used_boxes,
            truncated,
            variant,
            desired_model_id,
            final_only,
            len(windowed_captions) if caption_mode == "windowed" else 0,
            cleanup_count,
            refine_count,
            merge_count,
            word_count,
        )
    except HTTPException:
        if force_unload:
            logger.warning("[qwen-caption] exception -> forcing unload")
            request_model_cache.clear()
            _unload_qwen_runtime()
            active_runtime = None
            active_model_id = None
        raise
    except Exception as exc:  # noqa: BLE001
        if force_unload:
            logger.warning("[qwen-caption] exception=%s -> forcing unload", exc)
            request_model_cache.clear()
            _unload_qwen_runtime()
            active_runtime = None
            active_model_id = None
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_caption_failed:{exc}") from exc
    if force_unload:
        request_model_cache.clear()
        _unload_qwen_runtime()
        active_runtime = None
        active_model_id = None
    return response


app.include_router(
    build_qwen_infer_router(
        infer_fn=qwen_infer,
        request_cls=QwenInferenceRequest,
        response_cls=QwenInferenceResponse,
    )
)

app.include_router(
    build_qwen_caption_router(
        caption_fn=qwen_caption,
        request_cls=QwenCaptionRequest,
        response_cls=QwenCaptionResponse,
    )
)


def qwen_prepass(payload: QwenPrepassRequest):
    try:
        payload = payload.copy(update={"prepass_only": True})
        return _run_prepass_annotation(payload)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_prepass_failed:{exc}") from exc


app.include_router(
    build_qwen_prepass_router(
        prepass_fn=qwen_prepass,
        request_cls=QwenPrepassRequest,
        response_cls=QwenPrepassResponse,
    )
)


def start_calibration_job(payload: CalibrationRequest = Body(...)):
    job = _start_calibration_job_impl(
        payload,
        job_cls=CalibrationJob,
        jobs=CALIBRATION_JOBS,
        jobs_lock=CALIBRATION_JOBS_LOCK,
        run_job_fn=_calibration_run_job,
    )
    return _serialize_calibration_job_impl(job)


def list_calibration_jobs():
    _prune_job_registry(CALIBRATION_JOBS, CALIBRATION_JOBS_LOCK)
    with CALIBRATION_JOBS_LOCK:
        jobs = list(CALIBRATION_JOBS.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_serialize_calibration_job_impl(job) for job in jobs]


def get_calibration_job(job_id: str):
    with CALIBRATION_JOBS_LOCK:
        job = CALIBRATION_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="calibration_job_not_found")
    return _serialize_calibration_job_impl(job)


def cancel_calibration_job(job_id: str):
    job = _cancel_calibration_job_impl(
        job_id,
        jobs=CALIBRATION_JOBS,
        jobs_lock=CALIBRATION_JOBS_LOCK,
        http_exception_cls=HTTPException,
        time_fn=time.time,
    )
    return _serialize_calibration_job_impl(job)


app.include_router(
    build_calibration_router(
        start_fn=start_calibration_job,
        list_fn=list_calibration_jobs,
        get_fn=get_calibration_job,
        cancel_fn=cancel_calibration_job,
        request_cls=CalibrationRequest,
    )
)


def get_prepass_recipe(recipe_id: str):
    data = _get_prepass_recipe_impl(
        recipe_id,
        recipes_root=PREPASS_RECIPE_ROOT,
        sanitize_run_id_fn=_sanitize_yolo_run_id_impl,
        load_meta_fn=_load_prepass_recipe_meta,
        prepass_schema_version=PREPASS_RECIPE_SCHEMA_VERSION,
    )
    return PrepassRecipeResponse(**data)


def save_prepass_recipe(payload: PrepassRecipeRequest):
    recipe_id = payload.recipe_id or uuid.uuid4().hex
    data = _save_prepass_recipe_impl(
        payload.dict(),
        recipe_id=recipe_id,
        prepass_schema_version=PREPASS_RECIPE_SCHEMA_VERSION,
        recipes_root=PREPASS_RECIPE_ROOT,
        sanitize_run_id_fn=_sanitize_yolo_run_id_impl,
        normalize_glossary_fn=_normalize_labelmap_glossary,
        write_meta_fn=_write_prepass_recipe_meta,
    )
    return PrepassRecipeResponse(**data)


def delete_prepass_recipe(recipe_id: str):
    _delete_prepass_recipe_impl(
        recipe_id,
        recipes_root=PREPASS_RECIPE_ROOT,
        sanitize_run_id_fn=_sanitize_yolo_run_id_impl,
    )
    return {"status": "deleted", "id": recipe_id}


def export_prepass_recipe(recipe_id: str):
    zip_path = _export_prepass_recipe_impl(
        recipe_id,
        prepass_recipe_meta=PREPASS_RECIPE_META,
        prepass_schema_version=PREPASS_RECIPE_SCHEMA_VERSION,
        prepass_recipe_export_root=PREPASS_RECIPE_EXPORT_ROOT,
        prepass_recipe_root=PREPASS_RECIPE_ROOT,
        sanitize_run_id_fn=_sanitize_yolo_run_id_impl,
        load_meta_fn=_load_prepass_recipe_meta,
        collect_assets_fn=lambda recipe_meta, temp_dir: _collect_recipe_assets_impl(
            recipe_meta,
            temp_dir,
            read_labelmap_lines_fn=_read_labelmap_lines,
            load_labelmap_meta_fn=_agent_load_labelmap_meta,
            active_labelmap_path=active_labelmap_path,
            sanitize_run_id_fn=_sanitize_yolo_run_id_impl,
            copy_tree_filtered_fn=_copy_tree_filtered_impl,
            sha256_fn=_sha256_path_impl,
            get_qwen_model_entry_fn=_get_qwen_model_entry,
            resolve_classifier_path_fn=lambda path_str: _resolve_agent_clip_classifier_path_impl(
                path_str,
                allowed_root=(UPLOAD_ROOT / "classifiers").resolve(),
                allowed_exts=CLASSIFIER_ALLOWED_EXTS,
                path_is_within_root_fn=_path_is_within_root_impl,
                http_exception_cls=HTTPException,
            ),
            yolo_job_root=YOLO_JOB_ROOT,
            rfdetr_job_root=RFDETR_JOB_ROOT,
            rfdetr_keep_files=RFDETR_KEEP_FILES,
            qwen_metadata_filename=QWEN_METADATA_FILENAME,
            qwen_job_root=QWEN_JOB_ROOT,
            upload_root=UPLOAD_ROOT,
            calibration_root=CALIBRATION_ROOT,
        ),
    )
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"prepass_recipe_{recipe_id}.zip",
    )


def _validate_prepass_recipe_manifest(manifest: Dict[str, Any], extract_dir: Path) -> None:
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prepass_recipe_manifest_invalid")
    schema = manifest.get("schema_version")
    if schema != PREPASS_RECIPE_SCHEMA_VERSION:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prepass_recipe_schema_mismatch")
    assets = manifest.get("assets")
    if not assets:
        return
    if isinstance(assets, dict):
        assets = list(assets.values())
    if not isinstance(assets, list):
        return
    for item in assets:
        if not isinstance(item, dict):
            continue
        rel = item.get("path")
        if not rel:
            continue
        target = (extract_dir / rel).resolve()
        if not str(target).startswith(str(extract_dir.resolve())):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prepass_recipe_manifest_path")
        if not target.exists():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prepass_recipe_manifest_missing")


def _unique_prepass_recipe_name(name: str) -> tuple[str, Optional[str]]:
    base = (name or "").strip() or "prepass_recipe"
    existing = set()
    for entry in PREPASS_RECIPE_ROOT.iterdir():
        if not entry.is_dir():
            continue
        try:
            meta = _load_prepass_recipe_meta(entry)
        except HTTPException:
            continue
        existing_name = (meta.get("name") or "").strip()
        if existing_name:
            existing.add(existing_name)
    if base not in existing:
        return base, None
    suffix = 1
    while True:
        candidate = f"{base} ({suffix})"
        if candidate not in existing:
            return candidate, base
        suffix += 1


def _import_prepass_recipe_from_zip(zip_path: Path) -> PrepassRecipeResponse:
    data = _import_prepass_recipe_from_zip_impl(
        zip_path,
        prepass_recipe_meta=PREPASS_RECIPE_META,
        prepass_schema_version=PREPASS_RECIPE_SCHEMA_VERSION,
        prepass_recipe_root=PREPASS_RECIPE_ROOT,
        prepass_tmp_root=PREPASS_RECIPE_TMP_ROOT,
        yolo_job_root=YOLO_JOB_ROOT,
        rfdetr_job_root=RFDETR_JOB_ROOT,
        rfdetr_keep_files=RFDETR_KEEP_FILES,
        qwen_job_root=QWEN_JOB_ROOT,
        qwen_metadata_filename=QWEN_METADATA_FILENAME,
        upload_root=UPLOAD_ROOT,
        calibration_root=CALIBRATION_ROOT,
        read_labelmap_lines_fn=_read_labelmap_lines,
        validate_manifest_fn=_validate_prepass_recipe_manifest,
        unique_name_fn=_unique_prepass_recipe_name,
        normalize_glossary_fn=_normalize_labelmap_glossary,
        write_meta_fn=_write_prepass_recipe_meta,
        sanitize_run_id_fn=_sanitize_yolo_run_id_impl,
    )
    return PrepassRecipeResponse(**data)


def import_prepass_recipe(file: UploadFile = File(...)):  # noqa: B008
    temp_dir = Path(tempfile.mkdtemp(prefix="prepass_recipe_import_"))
    try:
        zip_path = temp_dir / "upload.zip"
        with zip_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        return _import_prepass_recipe_from_zip(zip_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def import_prepass_recipe_raw(request: Request):
    if "application/zip" not in (request.headers.get("content-type") or ""):
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="prepass_recipe_invalid_media")
    temp_dir = Path(tempfile.mkdtemp(prefix="prepass_recipe_import_raw_"))
    try:
        zip_path = temp_dir / "upload.zip"
        with zip_path.open("wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        return _import_prepass_recipe_from_zip(zip_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


app.include_router(
    build_prepass_router(
        list_fn=lambda: _list_prepass_recipes_impl(
            recipes_root=PREPASS_RECIPE_ROOT,
            meta_filename=PREPASS_RECIPE_META,
        ),
        get_fn=get_prepass_recipe,
        save_fn=save_prepass_recipe,
        delete_fn=delete_prepass_recipe,
        export_fn=export_prepass_recipe,
        import_fn=import_prepass_recipe,
        import_raw_fn=import_prepass_recipe_raw,
        response_cls=PrepassRecipeResponse,
        request_cls=PrepassRecipeRequest,
    )
)

app.include_router(
    build_datasets_router(
        list_fn=list_datasets,
        upload_fn=upload_dataset_zip,
        delete_fn=delete_dataset_entry,
        download_fn=download_dataset_entry,
        build_qwen_fn=build_qwen_dataset_from_yolo,
        check_fn=check_dataset,
        get_glossary_fn=get_dataset_glossary,
        set_glossary_fn=set_dataset_glossary,
        get_text_label_fn=get_text_label,
        set_text_label_fn=set_text_label,
    )
)

app.include_router(
    build_glossaries_router(
        list_fn=list_glossary_library,
        get_fn=get_glossary_entry,
        save_fn=save_glossary_entry,
        delete_fn=delete_glossary_entry,
    )
)

app.include_router(
    build_qwen_datasets_router(
        list_fn=list_qwen_datasets,
        delete_fn=delete_qwen_dataset,
        init_fn=init_qwen_dataset_upload,
        chunk_fn=upload_qwen_dataset_chunk,
        finalize_fn=finalize_qwen_dataset_upload,
        cancel_fn=cancel_qwen_dataset_upload,
    )
)

app.include_router(
    build_sam3_datasets_router(
        list_fn=list_sam3_datasets,
        convert_fn=convert_sam3_dataset,
        classes_fn=list_sam3_dataset_classes,
    )
)


def sam3_text_prompt(payload: Sam3TextPrompt):
    variant = _default_variant(payload.sam_variant or "sam3")
    if variant != "sam3":
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_text_requires_sam3")
    pil_img, np_img, token = resolve_image_payload(payload.image_base64, payload.image_token, variant)
    effective_limit = payload.max_results
    detections, masks_arr = _run_sam3_text_inference(
        pil_img,
        payload.text_prompt,
        payload.threshold,
        payload.mask_threshold,
        effective_limit,
        return_masks=True,
        min_size=payload.min_size,
        simplify_epsilon=payload.simplify_epsilon,
    )
    warnings: List[str] = []
    if not detections:
        warnings.append("no_results")
    encoded_masks = None
    if detections:
        encoded_masks = []
        for idx, det in enumerate(detections):
            payload = det.mask if isinstance(det, QwenDetection) else None
            if payload is None and masks_arr is not None and idx < len(masks_arr) and masks_arr[idx] is not None:
                try:
                    payload = _encode_binary_mask_impl(masks_arr[idx], max_bytes=MASK_ENCODE_MAX_BYTES)
                except Exception:
                    payload = None
            encoded_masks.append(payload)
        if all(m is None for m in encoded_masks):
            encoded_masks = None
    return Sam3TextPromptResponse(
        detections=detections,
        warnings=warnings,
        image_token=token,
        masks=encoded_masks,
    )


def sam3_text_prompt_auto(payload: Sam3TextPrompt):
    variant = _default_variant(payload.sam_variant or "sam3")
    if variant != "sam3":
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_text_requires_sam3")
    if not _active_encoder_ready():
        return Sam3TextPromptAutoResponse(
            detections=[],
            warnings=["clip_unavailable"],
            image_token=None,
        )
    pil_img, np_img, token = resolve_image_payload(payload.image_base64, payload.image_token, variant)
    effective_limit = payload.max_results
    detections, masks_arr = _run_sam3_text_inference(
        pil_img,
        payload.text_prompt,
        payload.threshold,
        payload.mask_threshold,
        effective_limit,
        return_masks=True,
        min_size=payload.min_size,
        simplify_epsilon=payload.simplify_epsilon,
    )
    # TODO: enrich with masks for polygon mode consumers.
    responses: List[SamPointAutoResponse] = []
    warnings: List[str] = []
    if not detections:
        warnings.append("no_results")
    for idx, det in enumerate(detections):
        mask = masks_arr[idx] if masks_arr is not None and idx < len(masks_arr) else None
        mask_payload = det.mask if hasattr(det, "mask") else None
        if mask_payload is None and mask is not None:
            mask_payload = _encode_binary_mask_impl(mask, max_bytes=MASK_ENCODE_MAX_BYTES)
        if mask is not None:
            try:
                x_min, y_min, x_max, y_max = _mask_to_bounding_box(mask)
            except Exception:
                x_min, y_min, x_max, y_max = _yolo_to_xyxy_int(det.bbox, pil_img.width, pil_img.height)
        else:
            x_min, y_min, x_max, y_max = _yolo_to_xyxy_int(det.bbox, pil_img.width, pil_img.height)
        li = max(0, int(x_min))
        ti = max(0, int(y_min))
        ri = min(pil_img.width, int(x_max))
        bi = min(pil_img.height, int(y_max))
        if ri <= li or bi <= ti:
            responses.append(
                SamPointAutoResponse(
                    prediction="unknown",
                    bbox=det.bbox,
                    uuid=str(uuid.uuid4()),
                    error="empty_mask",
                    image_token=token,
                    score=det.score,
                    mask=mask_payload,
                    simplify_epsilon=getattr(det, "simplify_epsilon", None),
                )
            )
            continue
        subarr = np_img[ti:bi, li:ri, :]
        final_pil = Image.fromarray(subarr)
        feats_np = _encode_pil_batch_for_active([final_pil])
        if feats_np is None or not isinstance(feats_np, np.ndarray) or feats_np.size == 0:
            responses.append(
                SamPointAutoResponse(
                    prediction="unknown",
                    bbox=det.bbox,
                    uuid=str(uuid.uuid4()),
                    error="clip_unavailable",
                    image_token=token,
                    score=det.score,
                    mask=mask_payload,
                    simplify_epsilon=getattr(det, "simplify_epsilon", None),
                )
            )
            continue
        details = _clip_auto_predict_details(feats_np)
        err = details.get("error")
        if isinstance(err, str) and err.startswith("classifier_error") and err not in warnings:
            warnings.append(err)
        responses.append(
            SamPointAutoResponse(
                prediction=str(details.get("label") or "unknown"),
                proba=details.get("proba"),
                second_label=details.get("second_label"),
                second_proba=details.get("second_proba"),
                margin=details.get("margin"),
                bbox=det.bbox,
                uuid=str(uuid.uuid4()),
                image_token=token,
                score=det.score,
                mask=mask_payload,
                simplify_epsilon=getattr(det, "simplify_epsilon", None),
                error=err,
            )
        )
    return Sam3TextPromptAutoResponse(detections=responses, warnings=warnings, image_token=token)


def sam3_visual_prompt(payload: Sam3VisualPrompt):
    variant = _default_variant(payload.sam_variant or "sam3")
    if variant != "sam3":
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_visual_requires_sam3")
    pil_img, np_img, token = resolve_image_payload(payload.image_base64, payload.image_token, variant)
    effective_limit = payload.max_results
    try:
        if payload.bboxes:
            detections, masks_arr = _run_sam3_visual_inference_multi(
                pil_img,
                [tuple(bx) for bx in payload.bboxes],
                payload.bbox_labels,
                payload.threshold,
                payload.mask_threshold,
                effective_limit,
                return_masks=True,
                min_size=payload.min_size,
                simplify_epsilon=payload.simplify_epsilon,
            )
        else:
            detections, masks_arr = _run_sam3_visual_inference(
                pil_img,
                (
                    float(payload.bbox_left),
                    float(payload.bbox_top),
                    float(payload.bbox_width),
                    float(payload.bbox_height),
                ),
                payload.threshold,
                payload.mask_threshold,
                effective_limit,
                return_masks=True,
                min_size=payload.min_size,
                simplify_epsilon=payload.simplify_epsilon,
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_visual_failed:{exc}") from exc
    warnings: List[str] = []
    if not detections:
        warnings.append("no_results")
    encoded_masks = None
    if detections:
        encoded_masks = []
        for idx, det in enumerate(detections):
            payload_mask = det.mask if isinstance(det, QwenDetection) else None
            if payload_mask is None and masks_arr is not None and idx < len(masks_arr) and masks_arr[idx] is not None:
                try:
                    payload_mask = _encode_binary_mask_impl(masks_arr[idx], max_bytes=MASK_ENCODE_MAX_BYTES)
                except Exception:
                    payload_mask = None
            encoded_masks.append(payload_mask)
        if all(m is None for m in encoded_masks):
            encoded_masks = None
    return Sam3TextPromptResponse(
        detections=detections,
        warnings=warnings,
        image_token=token,
        masks=encoded_masks,
    )


def sam_point(prompt: PointPrompt):
    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    coords = np.array([[prompt.point_x, prompt.point_y]])
    labels = np.array([1])
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    left, top, right, bottom = _mask_to_bounding_box(mask_arr)
    yolo_box = _xyxy_to_yolo_norm_list(pil_img.width, pil_img.height, left, top, right, bottom)
    return YoloBboxOutput(
        class_id="0",
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
        mask=_encode_binary_mask_impl(mask_arr, max_bytes=MASK_ENCODE_MAX_BYTES),
        simplify_epsilon=None,
    )


def sam_bbox_auto(prompt: BboxPrompt):
    if not _active_encoder_ready():
        return SamPointAutoResponse(prediction=str(ERROR_MESSAGE), bbox=[], uuid=prompt.uuid)

    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    full_h, full_w = pil_img.height, pil_img.width
    left = max(0, prompt.bbox_left)
    top = max(0, prompt.bbox_top)
    right = min(full_w, left + prompt.bbox_width)
    bottom = min(full_h, top + prompt.bbox_height)
    if right <= left or bottom <= top:
        return SamPointAutoResponse(
            prediction="unknown",
            bbox=[0, 0, 0, 0],
            uuid=prompt.uuid,
            error="invalid_bbox",
            image_token=token,
        )
    sub_box = np.array([left, top, right, bottom], dtype=np.float32)
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        box=sub_box,
        multimask_output=False,
    )
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    x_min, y_min, x_max, y_max = _mask_to_bounding_box(mask_arr)
    yolo_box = _xyxy_to_yolo_norm_list(full_w, full_h, x_min, y_min, x_max, y_max)
    gx_min_i = max(0, int(x_min))
    gy_min_i = max(0, int(y_min))
    gx_max_i = min(full_w, int(x_max))
    gy_max_i = min(full_h, int(y_max))
    if gx_max_i <= gx_min_i or gy_max_i <= gy_min_i:
        return SamPointAutoResponse(
            prediction="unknown",
            bbox=yolo_box,
            uuid=prompt.uuid,
            error="empty_mask",
            image_token=token,
        )
    subarr = np_img[gy_min_i:gy_max_i, gx_min_i:gx_max_i, :]
    final_pil = Image.fromarray(subarr)
    feats_np = _encode_pil_batch_for_active([final_pil])
    if feats_np is None or not isinstance(feats_np, np.ndarray) or feats_np.size == 0:
        return SamPointAutoResponse(prediction="unknown", bbox=yolo_box, uuid=prompt.uuid, error="clip_unavailable", image_token=token)
    details = _clip_auto_predict_details(feats_np)
    return SamPointAutoResponse(
        prediction=str(details.get("label") or "unknown"),
        proba=details.get("proba"),
        second_label=details.get("second_label"),
        second_proba=details.get("second_proba"),
        margin=details.get("margin"),
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
        mask=_encode_binary_mask_impl(mask_arr, max_bytes=MASK_ENCODE_MAX_BYTES),
        simplify_epsilon=None,
        error=details.get("error"),
    )


def sam_point_auto(prompt: PointPrompt):
    if not _active_encoder_ready():
        return SamPointAutoResponse(prediction=str(ERROR_MESSAGE), bbox=[], uuid=prompt.uuid)

    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    coords = np.array([[prompt.point_x, prompt.point_y]])
    labels = np.array([1])
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    left, top, right, bottom = _mask_to_bounding_box(mask_arr)
    yolo_box = _xyxy_to_yolo_norm_list(pil_img.width, pil_img.height, left, top, right, bottom)
    li = max(0, int(left))
    ti = max(0, int(top))
    ri = min(pil_img.width, int(right))
    bi = min(pil_img.height, int(bottom))
    if ri <= li or bi <= ti:
        return SamPointAutoResponse(
            prediction="unknown",
            bbox=yolo_box,
            uuid=prompt.uuid,
            error="empty_mask",
            image_token=token,
            mask=_encode_binary_mask_impl(mask_arr, max_bytes=MASK_ENCODE_MAX_BYTES),
            simplify_epsilon=None,
        )
    subarr = np_img[ti:bi, li:ri, :]
    final_pil = Image.fromarray(subarr)
    feats_np = _encode_pil_batch_for_active([final_pil])
    if feats_np is None or not isinstance(feats_np, np.ndarray) or feats_np.size == 0:
        return SamPointAutoResponse(
            prediction="unknown",
            bbox=yolo_box,
            uuid=prompt.uuid,
            error="clip_unavailable",
            image_token=token,
            mask=_encode_binary_mask_impl(mask_arr, max_bytes=MASK_ENCODE_MAX_BYTES),
            simplify_epsilon=None,
        )
    details = _clip_auto_predict_details(feats_np)
    return SamPointAutoResponse(
        prediction=str(details.get("label") or "unknown"),
        proba=details.get("proba"),
        second_label=details.get("second_label"),
        second_proba=details.get("second_proba"),
        margin=details.get("margin"),
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
        mask=_encode_binary_mask_impl(mask_arr, max_bytes=MASK_ENCODE_MAX_BYTES),
        simplify_epsilon=None,
        error=details.get("error"),
    )


def sam_point_multi(prompt: MultiPointPrompt):
    positive = prompt.positive_points or []
    negative = prompt.negative_points or []
    if len(positive) == 0:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="positive_points_required")

    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    coords = np.array(positive + negative, dtype=np.float32)
    labels = np.array([1] * len(positive) + [0] * len(negative), dtype=np.int64)
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    left, top, right, bottom = _mask_to_bounding_box(mask_arr)
    yolo_box = _xyxy_to_yolo_norm_list(pil_img.width, pil_img.height, left, top, right, bottom)
    return YoloBboxOutput(
        class_id="0",
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
        mask=_encode_binary_mask_impl(mask_arr, max_bytes=MASK_ENCODE_MAX_BYTES),
        simplify_epsilon=None,
    )


def sam_point_multi_auto(prompt: MultiPointPrompt):
    if not _active_encoder_ready():
        return SamPointAutoResponse(prediction=str(ERROR_MESSAGE), bbox=[], uuid=prompt.uuid)

    positive = prompt.positive_points or []
    negative = prompt.negative_points or []
    if len(positive) == 0:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="positive_points_required")

    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    coords = np.array(positive + negative, dtype=np.float32)
    labels = np.array([1] * len(positive) + [0] * len(negative), dtype=np.int64)
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    left, top, right, bottom = _mask_to_bounding_box(mask_arr)
    yolo_box = _xyxy_to_yolo_norm_list(pil_img.width, pil_img.height, left, top, right, bottom)
    li = max(0, int(left))
    ti = max(0, int(top))
    ri = min(pil_img.width, int(right))
    bi = min(pil_img.height, int(bottom))
    if ri <= li or bi <= ti:
        return SamPointAutoResponse(prediction="unknown", bbox=yolo_box, uuid=prompt.uuid, error="empty_mask", image_token=token)
    subarr = np_img[ti:bi, li:ri, :]
    final_pil = Image.fromarray(subarr)
    feats_np = _encode_pil_batch_for_active([final_pil])
    if feats_np is None or not isinstance(feats_np, np.ndarray) or feats_np.size == 0:
        return SamPointAutoResponse(prediction="unknown", bbox=yolo_box, uuid=prompt.uuid, error="clip_unavailable", image_token=token)
    details = _clip_auto_predict_details(feats_np)
    return SamPointAutoResponse(
        prediction=str(details.get("label") or "unknown"),
        proba=details.get("proba"),
        second_label=details.get("second_label"),
        second_proba=details.get("second_proba"),
        margin=details.get("margin"),
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
        mask=_encode_binary_mask_impl(mask_arr, max_bytes=MASK_ENCODE_MAX_BYTES),
        simplify_epsilon=None,
        error=details.get("error"),
    )


def sam_bbox(prompt: BboxPrompt):
    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    full_h, full_w = pil_img.height, pil_img.width
    left = max(0, prompt.bbox_left)
    top = max(0, prompt.bbox_top)
    right = min(full_w, left + prompt.bbox_width)
    bottom = min(full_h, top + prompt.bbox_height)
    if right <= left or bottom <= top:
        return YoloBboxOutput(
            class_id="0",
            bbox=[0, 0, 0, 0],
            uuid=prompt.uuid
        )
    sub_box = np.array([left, top, right, bottom], dtype=np.float32)
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        box=sub_box,
        multimask_output=False,
    )
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    x_min, y_min, x_max, y_max = _mask_to_bounding_box(mask_arr)
    yolo_box = _xyxy_to_yolo_norm_list(full_w, full_h, x_min, y_min, x_max, y_max)
    gx_min_i = max(0, int(x_min))
    gy_min_i = max(0, int(y_min))
    gx_max_i = min(full_w, int(x_max))
    gy_max_i = min(full_h, int(y_max))
    if gx_max_i <= gx_min_i or gy_max_i <= gy_min_i:
        return YoloBboxOutput(
            class_id="0",
            bbox=yolo_box,
            uuid=prompt.uuid,
            image_token=token,
        )
    return YoloBboxOutput(
        class_id="0",
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
        mask=_encode_binary_mask_impl(mask_arr, max_bytes=MASK_ENCODE_MAX_BYTES),
        simplify_epsilon=None,
    )


app.include_router(
    build_sam3_prompts_router(
        sam3_text_fn=sam3_text_prompt,
        sam3_text_auto_fn=sam3_text_prompt_auto,
        sam3_visual_fn=sam3_visual_prompt,
        sam_point_fn=sam_point,
        sam_bbox_auto_fn=sam_bbox_auto,
        sam_point_auto_fn=sam_point_auto,
        sam_point_multi_fn=sam_point_multi,
        sam_point_multi_auto_fn=sam_point_multi_auto,
        sam_bbox_fn=sam_bbox,
        sam3_text_req=Sam3TextPrompt,
        sam3_text_auto_req=Sam3TextPrompt,
        sam3_visual_req=Sam3VisualPrompt,
        sam_point_req=PointPrompt,
        sam_bbox_req=BboxPrompt,
        sam_point_multi_req=MultiPointPrompt,
        sam3_text_resp=Sam3TextPromptResponse,
        sam3_text_auto_resp=Sam3TextPromptAutoResponse,
        sam_point_auto_resp=SamPointAutoResponse,
        yolo_bbox_resp=YoloBboxOutput,
    )
)

def crop_zip_init():
    jobId = str(uuid.uuid4())
    job_store[jobId] = []
    return {"jobId": jobId}

def crop_zip_chunk(request: CropZipRequest, jobId: str = Query(...)):
    if jobId not in job_store:
        raise HTTPException(status_code=400, detail="Invalid jobId")
    job_store[jobId].extend(request.images)
    return {"status": "ok", "count": len(request.images)}

def crop_zip_finalize(jobId: str):
    if jobId not in job_store:
        raise HTTPException(status_code=400, detail="Invalid jobId")
    all_images = job_store[jobId]
    if len(all_images) == 0:
        empty_buffer = io.BytesIO()
        with zipfile.ZipFile(empty_buffer, mode="w") as zf:
            pass
        empty_buffer.seek(0)
        del job_store[jobId]
        return StreamingResponse(
            empty_buffer,
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": "attachment; filename=crops.zip"}
        )
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, cropImage in enumerate(all_images):
            img_data = base64.b64decode(cropImage.image_base64)
            pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
            for bindex, bbox in enumerate(cropImage.bboxes):
                left = bbox.x
                top = bbox.y
                right = left + bbox.width
                bottom = top + bbox.height
                left = max(0, min(left, pil_img.width))
                right = max(0, min(right, pil_img.width))
                top = max(0, min(top, pil_img.height))
                bottom = max(0, min(bottom, pil_img.height))
                if right <= left or bottom <= top:
                    continue
                sub_img = pil_img.crop((left, top, right, bottom))
                stem = cropImage.originalName.rsplit(".",1)[0]
                out_name = f"{stem}-{bbox.className}-{bindex}.jpg"
                crop_buffer = io.BytesIO()
                sub_img.save(crop_buffer, format="JPEG")
                crop_buffer.seek(0)
                zf.writestr(out_name, crop_buffer.read())
    zip_buffer.seek(0)
    del job_store[jobId]
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=crops.zip"}
    )


app.include_router(
    build_crop_zip_router(
        init_fn=crop_zip_init,
        chunk_fn=crop_zip_chunk,
        finalize_fn=crop_zip_finalize,
        request_cls=CropZipRequest,
    )
)


if os.environ.get("COORD_ROUNDTRIP_TEST") == "1":
    _coord_roundtrip_smoke()
