from __future__ import annotations

import base64, colorsys, copy, hashlib, io, zipfile, math, uuid, os, tempfile, shutil, time, logging, subprocess, sys, json, re, signal, random, gzip, csv, socket, gc, queue, multiprocessing
from array import array
from contextvars import ContextVar
from pathlib import Path
import numpy as np
import yaml
from typing import Optional, List, Dict, Tuple, Any, Literal, Sequence, Mapping, Callable, Set
from collections import deque, Counter
import torch, clip, joblib, tiktoken
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, Form, Query, Body, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, root_validator, Field
from omegaconf import OmegaConf
import psutil
try:
    from packaging import version as packaging_version
except Exception:  # noqa: BLE001
    packaging_version = None
try:
    from transformers import LogitsProcessor, LogitsProcessorList
except Exception:  # noqa: BLE001
    LogitsProcessor = None
    LogitsProcessorList = None
_BASE_LOGITS_PROCESSOR = LogitsProcessor if LogitsProcessor is not None else object
from starlette.background import BackgroundTask
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
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
    _write_qwen_metadata,
    _atomic_write_text,
    _atomic_write_json,
    _read_csv_last_row,
    _sanitize_yolo_run_id as _sanitize_yolo_run_id_impl,
    _compute_dir_signature as _compute_dir_signature_impl,
    _dir_size_bytes as _dir_size_bytes_impl,
)
from utils.image import _load_image_size, _slice_image_sahi
from utils.labels import (
    _read_labelmap_lines,
    _load_labelmap_file,
    _normalize_class_name_for_match,
    _normalize_labelmap_entries,
    _apply_expected_labelmap_warnings,
    _labelmaps_match,
    _raise_on_labelmap_mismatch,
    _agent_label_prefix_candidates,
    _agent_label_color_map,
    _agent_label_prefix_map,
    _agent_overlay_key_text,
    _agent_fuzzy_align_label,
)
from utils.classifier_utils import (
    _is_background_class_name,
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
    _parse_device_ids_string,
    _agent_extract_json_array,
)
from utils.errors import _agent_error_payload, _agent_error_from_detail
from utils.glossary import (
    _glossary_label_key,
    _extract_glossary_synonyms,
    _normalize_labelmap_glossary,
    _normalize_glossary_name,
    _glossary_key,
    _parse_glossary_mapping,
    _parse_glossary_synonyms,
    _split_synonym_terms,
    _clean_sam3_synonym,
    _normalize_synonym_list,
    _dedupe_synonyms,
    _default_agent_glossary_for_labelmap,
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
    _normalize_agent_recipe_execution_plan as _normalize_agent_recipe_execution_plan_impl,
    _validate_agent_recipe_structure as _validate_agent_recipe_structure_impl,
    _save_exemplar_crop_impl as _save_exemplar_crop_impl,
    _delete_agent_recipe_impl as _delete_agent_recipe_impl,
    _list_agent_recipes_impl as _list_agent_recipes_impl,
    _normalize_agent_recipe_steps_impl as _normalize_agent_recipe_steps_impl,
    _persist_agent_recipe_impl as _persist_agent_recipe_impl,
    _load_agent_recipe_impl as _load_agent_recipe_impl,
    _load_agent_recipe_json_only_impl as _load_agent_recipe_json_only_impl,
    _ensure_recipe_zip_impl as _ensure_recipe_zip_impl,
    _import_agent_recipe_zip_bytes_impl as _import_agent_recipe_zip_bytes_impl,
    _prepass_recipe_dir_impl as _prepass_recipe_dir_impl,
    _prepass_recipe_meta_path_impl as _prepass_recipe_meta_path_impl,
    _prepass_recipe_assets_dir_impl as _prepass_recipe_assets_dir_impl,
    _sha256_path_impl as _sha256_path_impl,
    _copy_tree_filtered_impl as _copy_tree_filtered_impl,
    _unique_prepass_recipe_name_impl as _unique_prepass_recipe_name_impl,
    _validate_prepass_recipe_manifest_impl as _validate_prepass_recipe_manifest_impl,
    _list_prepass_recipes_impl as _list_prepass_recipes_impl,
    _collect_recipe_assets_impl as _collect_recipe_assets_impl,
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
    _ensure_clip_backbone_for_mining_impl as _ensure_clip_backbone_for_mining_impl,
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
    _find_any_file_impl as _find_any_file_impl,
    _count_dir_files_impl as _count_dir_files_impl,
    _dataset_integrity_report_impl as _dataset_integrity_report_impl,
    _resolve_yolo_training_dataset_impl as _resolve_yolo_training_dataset_impl,
    _resolve_rfdetr_training_dataset_impl as _resolve_rfdetr_training_dataset_impl,
    _compute_labelmap_hash_impl as _compute_labelmap_hash_impl,
    _compute_dataset_signature_impl as _compute_dataset_signature_impl,
    _purge_dataset_artifacts_impl as _purge_dataset_artifacts_impl,
    _ensure_qwen_dataset_signature_impl as _ensure_qwen_dataset_signature_impl,
    _find_qwen_dataset_by_signature_impl as _find_qwen_dataset_by_signature_impl,
    _load_registry_dataset_metadata_impl as _load_registry_dataset_metadata_impl,
    _persist_dataset_metadata_impl as _persist_dataset_metadata_impl,
    _coerce_dataset_metadata_impl as _coerce_dataset_metadata_impl,
    _load_qwen_dataset_metadata_impl as _load_qwen_dataset_metadata_impl,
    _persist_qwen_dataset_metadata_impl as _persist_qwen_dataset_metadata_impl,
    _load_sam3_dataset_metadata_impl as _load_sam3_dataset_metadata_impl,
    _persist_sam3_dataset_metadata_impl as _persist_sam3_dataset_metadata_impl,
    _count_dataset_images_impl as _count_dataset_images_impl,
    _count_caption_labels_impl as _count_caption_labels_impl,
    _list_all_datasets_impl as _list_all_datasets_impl,
    _collect_labels_from_qwen_jsonl_impl as _collect_labels_from_qwen_jsonl_impl,
    _extract_qwen_detections_from_payload_impl as _extract_qwen_detections_from_payload_impl,
    _discover_yolo_labelmap_impl as _discover_yolo_labelmap_impl,
)
from services.prepass import (
    _agent_merge_prepass_detections,
    _agent_filter_scoreless_detections,
    _agent_detection_has_source,
    _agent_det_score,
    _agent_cluster_match,
    _agent_source_counts,
    _agent_format_source_counts,
    _agent_label_counts_summary,
    _agent_compact_tool_result,
    _agent_select_similarity_exemplars as _agent_select_similarity_exemplars_impl,
    _agent_deep_prepass_cleanup_impl,
    _agent_run_deep_prepass_part_a_impl,
    _agent_run_deep_prepass_impl,
    _agent_run_deep_prepass_caption_impl,
    _agent_run_prepass_impl,
)
from services.cluster_helpers import _cluster_label_counts, _cluster_summaries
from services.context_store import (
    _context_store,
    _context_chunk,
    _agent_context_store as _agent_context_store_impl,
    _agent_context_chunk as _agent_context_chunk_impl,
)
from services.tile_context import (
    _cluster_owner_cell,
    _tile_clusters,
    _tile_cluster_payload,
    _tile_caption_hint,
    _build_tile_context_payloads as _build_tile_context_payloads_impl,
)
from services.prepass_grid import (
    _agent_grid_col_label,
    _agent_grid_col_index,
    _agent_grid_spec,
    _agent_grid_spec_for_payload,
    _agent_grid_cell_xyxy,
    _agent_grid_cell_for_window_bbox,
    _agent_grid_prompt_text,
    _agent_grid_cells,
    _agent_grid_cell_for_detection,
    _agent_grid_usage_rows,
    _agent_grid_usage_text,
    _agent_grid_label_counts,
    _agent_quadrant_windows_qwen,
    _agent_tool_grid_cell_from_args as _grid_cell_from_args,
    _agent_record_grid_tool_usage as _record_grid_usage,
)
from services.glossary_library import (
    _normalize_glossary_name,
    _glossary_key,
    _load_glossary_library,
    _persist_glossary_library,
    _find_glossary_entry,
    _upsert_glossary_entry_impl as _upsert_glossary_entry_impl,
    _delete_glossary_entry_impl as _delete_glossary_entry_impl,
)
from utils.coords import (
    _xyxy_to_qwen_bbox,
    _qwen_bbox_to_xyxy,
    _remap_window_xyxy_to_full,
    _normalize_window_xyxy,
    _window_bbox_2d_to_full_xyxy,
    _window_local_bbox_2d_to_full_xyxy,
    _window_local_xyxy_to_full_xyxy,
    _resolve_agent_bbox_xyxy,
    _agent_round_bbox_2d,
    _agent_clip_xyxy,
    _agent_expand_window_xyxy,
    _agent_xyxy_to_xywh,
    _yolo_to_xyxy,
    _xyxy_to_yolo_norm,
    _agent_det_payload,
    _agent_iou_xyxy,
    _extract_numeric_sequence,
    _scale_coord,
    _scale_bbox_to_image,
    _scale_point_to_image,
)
from utils.overlay import (
    _agent_detection_center_px,
    _agent_render_detection_overlay,
    _agent_render_grid_overlay,
    _agent_image_to_data_uri,
    _agent_overlay_labels,
)
from utils.text import _agent_clean_plan_text
from utils.llm import (
    _qwen_agent_message_text,
    _agent_content_to_text,
    _agent_stream_text_from_output,
    _agent_stream_tag_open,
    _agent_stream_extract_tool_name,
    _agent_parse_json_relaxed,
)
from utils.trace_utils import (
    _agent_trace_sanitize_payload,
    _agent_trace_sanitize_messages,
    _agent_trace_full_jsonable,
)
from services.readable import (
    _agent_readable_trim,
    _agent_readable_banner,
    _agent_detection_summary_lines,
    _agent_readable_detection_line,
    _agent_clean_observation_text,
    _agent_readable_format_bbox,
    _agent_readable_bbox_from_args,
    _agent_readable_tool_call_summary,
    _agent_readable_tool_result_summary,
    _agent_readable_line,
    _agent_readable_candidates_summary,
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
    _agent_classifier_review_impl as _agent_classifier_review_impl,
    _resolve_agent_clip_classifier_path_impl as _resolve_agent_clip_classifier_path_impl,
    _load_clip_head_from_classifier_impl as _load_clip_head_from_classifier_impl,
    _clip_head_predict_proba_impl as _clip_head_predict_proba_impl,
    _clip_head_keep_mask_impl as _clip_head_keep_mask_impl,
    _resolve_head_normalize_embeddings_impl as _resolve_head_normalize_embeddings_impl,
    _resolve_active_head_normalize_embeddings_impl as _resolve_active_head_normalize_embeddings_impl,
    _save_clip_head_artifacts_impl as _save_clip_head_artifacts_impl,
    _load_clip_head_artifacts_impl as _load_clip_head_artifacts_impl,
    _resolve_clip_head_background_settings_impl as _resolve_clip_head_background_settings_impl,
    _infer_clip_model_from_embedding_dim_impl as _infer_clip_model_from_embedding_dim_impl,
    _clip_auto_predict_label_impl as _clip_auto_predict_label_impl,
    _clip_auto_predict_details_impl as _clip_auto_predict_details_impl,
    _score_detections_with_clip_head_impl as _score_detections_with_clip_head_impl,
    _build_clip_head_sweep_grid_impl as _build_clip_head_sweep_grid_impl,
    _score_head_tuning_candidate_impl as _score_head_tuning_candidate_impl,
    _update_best_clip_head_sweep_summary_impl as _update_best_clip_head_sweep_summary_impl,
    _successive_halving_search_impl as _successive_halving_search_impl,
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
    _serialize_calibration_job as _serialize_calibration_job_impl,
    _run_calibration_job as _run_calibration_job_impl,
    _start_calibration_job as _start_calibration_job_impl,
    _cancel_calibration_job as _cancel_calibration_job_impl,
)
from services.calibration_metrics import (
    _expand_midpoints_impl as _expand_midpoints_impl,
    _build_gt_index_for_class_impl as _build_gt_index_for_class_impl,
    _evaluate_prompt_for_class_impl as _evaluate_prompt_for_class_impl,
    _evaluate_prompt_candidate_impl as _evaluate_prompt_candidate_impl,
    _collect_prompt_detections_impl as _collect_prompt_detections_impl,
    _build_prompt_recipe_impl as _build_prompt_recipe_impl,
    _score_greedy_eval_summaries_impl as _score_greedy_eval_summaries_impl,
    _gt_instance_key_impl as _gt_instance_key_impl,
    _build_seed_threshold_sweep_grid_impl as _build_seed_threshold_sweep_grid_impl,
    _compute_steps_seed_eval_threshold_impl as _compute_steps_seed_eval_threshold_impl,
    _compute_steps_seed_eval_max_results_impl as _compute_steps_seed_eval_max_results_impl,
    _compute_seed_threshold_curve_impl as _compute_seed_threshold_curve_impl,
    _select_seed_threshold_operating_point_impl as _select_seed_threshold_operating_point_impl,
    _select_seed_threshold_candidate_points_impl as _select_seed_threshold_candidate_points_impl,
    _summarize_seed_threshold_curve_for_prompt_impl as _summarize_seed_threshold_curve_for_prompt_impl,
    _select_steps_from_seed_prompt_stats_impl as _select_steps_from_seed_prompt_stats_impl,
    _resolve_steps_early_stop_config_impl as _resolve_steps_early_stop_config_impl,
    _build_seed_stage_candidate_from_prompt_stat_impl as _build_seed_stage_candidate_from_prompt_stat_impl,
    _refine_steps_prompt_subset_seed_stage_impl as _refine_steps_prompt_subset_seed_stage_impl,
    _resolve_steps_prompt_prefilter_config_impl as _resolve_steps_prompt_prefilter_config_impl,
    _resolve_steps_prompt_bg_drop_config_impl as _resolve_steps_prompt_bg_drop_config_impl,
    _resolve_steps_hard_negative_export_config_impl as _resolve_steps_hard_negative_export_config_impl,
    _estimate_steps_speed_factor_impl as _estimate_steps_speed_factor_impl,
    _estimate_agent_global_optimizer_image_evals_impl as _estimate_agent_global_optimizer_image_evals_impl,
    _build_steps_recipe_step_list_from_selected_stats_impl as _build_steps_recipe_step_list_from_selected_stats_impl,
    _collect_clip_prefilter_crops_impl as _collect_clip_prefilter_crops_impl,
    _normalize_steps_for_head_tuning_impl as _normalize_steps_for_head_tuning_impl,
    _prefilter_prompts_with_clip_impl as _prefilter_prompts_with_clip_impl,
    _export_hard_negative_replay_impl as _export_hard_negative_replay_impl,
)
from services.qwen import (
    _extract_balanced_json as _extract_balanced_json_impl,
    _extract_qwen_json_block_impl as _extract_qwen_json_block_impl,
    _generate_qwen_text as _generate_qwen_text_impl,
    _parse_prompt_candidates as _parse_prompt_candidates_impl,
    _generate_prompt_text as _generate_prompt_text_impl,
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
    _set_qwen_prompt_config_impl as _set_qwen_prompt_config_impl,
    _render_qwen_prompt_impl as _render_qwen_prompt_impl,
    _extract_qwen_json_block_impl as _extract_qwen_json_block_impl,
    _strip_qwen_model_suffix_impl as _strip_qwen_model_suffix_impl,
    _format_qwen_load_error_impl as _format_qwen_load_error_impl,
    _humanize_class_name_impl as _humanize_class_name_impl,
    _sanitize_prompts_impl as _sanitize_prompts_impl,
    _generate_prompt_variants_for_class_impl as _generate_prompt_variants_for_class_impl,
    _expand_prompts_with_prompt_llm_impl as _expand_prompts_with_prompt_llm_impl,
    _refine_prompts_with_qwen_impl as _refine_prompts_with_qwen_impl,
    _qwen_self_filter_prompts_impl as _qwen_self_filter_prompts_impl,
)
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
    _rfdetr_run_dir_impl as _rfdetr_run_dir_impl,
    _rfdetr_load_run_meta_impl as _rfdetr_load_run_meta_impl,
    _rfdetr_write_run_meta_impl as _rfdetr_write_run_meta_impl,
    _rfdetr_prune_run_dir_impl as _rfdetr_prune_run_dir_impl,
    _collect_yolo_artifacts_impl as _collect_yolo_artifacts_impl,
    _collect_rfdetr_artifacts_impl as _collect_rfdetr_artifacts_impl,
    _yolo_extract_detections_impl as _yolo_extract_detections_impl,
    _rfdetr_extract_detections_impl as _rfdetr_extract_detections_impl,
    _flatten_metrics_impl as _flatten_metrics_impl,
    _lookup_metric_impl as _lookup_metric_impl,
    _yolo_metrics_summary_impl as _yolo_metrics_summary_impl,
    _rfdetr_metrics_summary_impl as _rfdetr_metrics_summary_impl,
    _clean_metric_summary_impl as _clean_metric_summary_impl,
    _list_yolo_runs_impl as _list_yolo_runs_impl,
    _list_rfdetr_runs_impl as _list_rfdetr_runs_impl,
)
from collections import OrderedDict
try:
    from scipy.spatial import ConvexHull
except Exception:  # noqa: BLE001
    ConvexHull = None
from segment_anything import sam_model_registry, SamPredictor


def _message_text_for_tool_parse(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            try:
                item_type, item_value = item.get_type_and_value()
            except Exception:
                continue
            if item_type == "text" and item_value:
                parts.append(str(item_value))
        return "\n".join(parts)
    return ""


def _extract_balanced_json(text: str, start_char: str, end_char: str) -> Optional[str]:
    return _extract_balanced_json_impl(text, start_char, end_char)


def _parse_tool_call_payload(payload: str) -> Optional[Dict[str, Any]]:
    candidate = (payload or "").strip()
    if not candidate:
        return None
    for parser in ("json", "json5"):
        try:
            if parser == "json":
                return json.loads(candidate)
            import json5  # type: ignore
            return json5.loads(candidate)
        except Exception:
            continue
    for start_char, end_char in (("{", "}"), ("[", "]")):
        snippet = _extract_balanced_json(candidate, start_char, end_char)
        if not snippet:
            continue
        for parser in ("json", "json5"):
            try:
                if parser == "json":
                    return json.loads(snippet)
                import json5  # type: ignore
                return json5.loads(snippet)
            except Exception:
                continue
    return None


def _extract_tool_call_from_text(text: str) -> Optional[Tuple[str, Any]]:
    if not text:
        return None
    lower_text = text.lower()
    if "<tool_call>" in lower_text:
        start = lower_text.find("<tool_call>")
        end = lower_text.find("</tool_call>", start + 11)
        if end > start:
            payload = text[start + len("<tool_call>") : end].strip()
            parsed = _parse_tool_call_payload(payload)
            if isinstance(parsed, dict):
                name = str(parsed.get("name") or "").strip()
                args = parsed.get("arguments", {})
                if name:
                    return name, args
    if "✿function✿" in lower_text and "✿args✿" in lower_text:
        fn_idx = text.find("✿FUNCTION✿")
        args_idx = text.find("✿ARGS✿", fn_idx + 1)
        if fn_idx >= 0 and args_idx > fn_idx:
            name_chunk = text[fn_idx + len("✿FUNCTION✿") : args_idx]
            if ":" in name_chunk:
                name_chunk = name_chunk.split(":", 1)[1]
            name = name_chunk.strip().splitlines()[0].strip()
            args_chunk = text[args_idx + len("✿ARGS✿") :]
            stop_tokens = ["✿RESULT✿", "✿RETURN✿"]
            stop_pos = [args_chunk.find(tok) for tok in stop_tokens if tok in args_chunk]
            if stop_pos:
                args_chunk = args_chunk[: min(stop_pos)]
            parsed = _parse_tool_call_payload(args_chunk)
            if name:
                return name, parsed if parsed is not None else args_chunk.strip()
    parsed = _parse_tool_call_payload(text)
    if isinstance(parsed, dict):
        name = str(parsed.get("name") or "").strip()
        if name:
            return name, parsed.get("arguments", {})
    return None
import threading
import queue
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict

# Ensure we import the bundled SAM3 package (sam3/sam3) rather than shadowing it
# with the repo root folder name (sam3/). Without this, sam3 becomes a namespace
# that lacks the train.data modules needed for text prompting.
SAM3_SRC_ROOT = (Path(__file__).resolve().parent / "sam3").resolve()
if SAM3_SRC_ROOT.exists():
    sys.path.insert(0, str(SAM3_SRC_ROOT))

from tools.clip_training import train_clip_from_yolo, TrainingError, TrainingArtifacts
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
        import qwen_agent.settings as qwen_settings  # type: ignore
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


def _resolve_qwen_max_seq_len(model: Any) -> Optional[int]:
    config = getattr(model, "config", None)
    if config is None:
        return None

    def _read_seq_len(cfg: Any) -> Optional[int]:
        if cfg is None:
            return None
        for attr in ("max_position_embeddings", "max_sequence_length", "seq_length"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        # Some configs expose max_length as a generation hint (often tiny); treat as fallback only.
        val = getattr(cfg, "max_length", None)
        if isinstance(val, int) and val > 0:
            return val
        return None

    for cfg in (getattr(config, "text_config", None), getattr(config, "language_config", None), config):
        val = _read_seq_len(cfg)
        if isinstance(val, int) and val >= 256:
            return val
    return None


def _qwen_estimate_vision_tokens(preview_inputs: Any) -> Optional[int]:
    grid = None
    if isinstance(preview_inputs, dict):
        grid = preview_inputs.get("image_grid_thw")
    else:
        grid = getattr(preview_inputs, "image_grid_thw", None)
    if grid is None:
        return None
    try:
        grid_vals = grid if isinstance(grid, torch.Tensor) else torch.as_tensor(grid)
        if grid_vals.ndim == 3:
            grid_vals = grid_vals[0]
        if grid_vals.ndim == 2 and grid_vals.shape[-1] == 3:
            tokens = (grid_vals[:, 0] * grid_vals[:, 1] * grid_vals[:, 2]).sum()
            return int(tokens.item())
        if grid_vals.ndim == 1 and grid_vals.numel() == 3:
            tokens = grid_vals[0] * grid_vals[1] * grid_vals[2]
            return int(tokens.item())
    except Exception:
        return None
    return None


def _qwen_effective_input_len(preview_inputs: Any, input_len: int, num_images: int) -> Tuple[int, Optional[int]]:
    vision_tokens = _qwen_estimate_vision_tokens(preview_inputs)
    if vision_tokens is None or num_images <= 0:
        return input_len, vision_tokens
    effective_len = max(1, input_len - num_images + vision_tokens)
    return effective_len, vision_tokens


def _qwen_supports_presence_penalty(model: Any) -> bool:
    gen_config = getattr(model, "generation_config", None)
    if gen_config is None:
        return False
    if hasattr(gen_config, "to_dict"):
        try:
            return "presence_penalty" in gen_config.to_dict()
        except Exception:
            pass
    return hasattr(gen_config, "presence_penalty")


class ThinkingEffortProcessor(_BASE_LOGITS_PROCESSOR):
    """Scale the </think> token logit to reduce or increase chain-of-thought length."""

    def __init__(self, end_thinking_token_id: int, thinking_effort: float = 1.0, scale_factor: float = 2.0):
        super().__init__()
        self.end_thinking_token_id = int(end_thinking_token_id)
        self.thinking_effort = float(thinking_effort)
        self.scale_factor = float(scale_factor)
        self.finished_sequences: Set[int] = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.end_thinking_token_id >= scores.size(1):
            return scores
        scale = self.scale_factor ** (1.0 - self.thinking_effort)
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            if i in self.finished_sequences:
                continue
            if (input_ids[i] == self.end_thinking_token_id).any():
                self.finished_sequences.add(i)
                continue
            scores[i, self.end_thinking_token_id] *= scale
        return scores


class ImmediateActionBiasProcessor(_BASE_LOGITS_PROCESSOR):
    """Boost </think> when 'wait' appears inside a think block after a minimum threshold."""

    def __init__(
        self,
        tokenizer: Any,
        end_thinking_token_id: int,
        *,
        min_think_chars: int = 200,
        min_think_seconds: float = 2.0,
        logit_bias: float = 6.0,
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self.end_thinking_token_id = int(end_thinking_token_id)
        self.min_think_chars = max(1, int(min_think_chars))
        self.min_think_seconds = max(0.0, float(min_think_seconds))
        self.logit_bias = float(logit_bias)
        self._think_started_at: Dict[int, float] = {}
        self._wait_seen: Set[int] = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.logit_bias <= 0:
            return scores
        if self.end_thinking_token_id >= scores.size(1):
            return scores
        now = time.time()
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            if i in self._wait_seen:
                scores[i, self.end_thinking_token_id] += self.logit_bias
                continue
            try:
                text = self._tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=False)
            except Exception:
                continue
            think_text = self._extract_open_think_text(text)
            if think_text is None:
                if i in self._think_started_at:
                    del self._think_started_at[i]
                continue
            if i not in self._think_started_at:
                self._think_started_at[i] = now
            if len(think_text) < self.min_think_chars:
                continue
            if (now - self._think_started_at.get(i, now)) < self.min_think_seconds:
                continue
            if re.search(r"\bwait\b", think_text, flags=re.IGNORECASE):
                self._wait_seen.add(i)
                scores[i, self.end_thinking_token_id] += self.logit_bias
        return scores


def _qwen_find_end_think_token_id(tokenizer: Any) -> Optional[int]:
    if tokenizer is None:
        return None
    vocab_size = getattr(tokenizer, "vocab_size", None)
    vocab_size = int(vocab_size) if isinstance(vocab_size, int) and vocab_size > 0 else None
    candidates = [
        "</think>",
        "<|endofthink|>",
        "<|end_of_thought|>",
        "<|end_of_thinking|>",
    ]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in candidates:
        try:
            tok_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            tok_id = None
        if tok_id is not None and tok_id != unk_id:
            if vocab_size is not None and int(tok_id) >= vocab_size:
                tok_id = None
            else:
                return int(tok_id)
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
        except Exception:
            ids = []
        if isinstance(ids, list) and len(ids) == 1:
            tok_id = int(ids[0])
            if vocab_size is None or tok_id < vocab_size:
                return tok_id
    return None


def _qwen_build_thinking_effort_processor(
    tokenizer: Any,
    thinking_effort: Optional[float],
    scale_factor: Optional[float],
) -> Optional[ThinkingEffortProcessor]:
    if LogitsProcessorList is None or LogitsProcessor is None:
        return None
    if thinking_effort is None:
        return None
    try:
        effort_val = float(thinking_effort)
    except (TypeError, ValueError):
        return None
    end_token_id = _qwen_find_end_think_token_id(tokenizer)
    if end_token_id is None:
        return None
    scale_val = 2.0
    if scale_factor is not None:
        try:
            scale_val = float(scale_factor)
        except (TypeError, ValueError):
            scale_val = 2.0
    return ThinkingEffortProcessor(end_token_id, thinking_effort=effort_val, scale_factor=scale_val)


def _qwen_build_immediate_action_processor(
    tokenizer: Any,
    immediate_action_bias: Optional[bool],
    min_think_chars: Optional[int],
    min_think_seconds: Optional[float],
    logit_bias: Optional[float],
) -> Optional[ImmediateActionBiasProcessor]:
    if LogitsProcessorList is None or LogitsProcessor is None:
        return None
    if not immediate_action_bias:
        return None
    end_token_id = _qwen_find_end_think_token_id(tokenizer)
    if end_token_id is None:
        return None
    chars_val = 200 if min_think_chars is None else int(min_think_chars)
    secs_val = 2.0 if min_think_seconds is None else float(min_think_seconds)
    bias_val = 6.0 if logit_bias is None else float(logit_bias)
    return ImmediateActionBiasProcessor(
        tokenizer,
        end_token_id,
        min_think_chars=chars_val,
        min_think_seconds=secs_val,
        logit_bias=bias_val,
    )


def _qwen_append_logits_processor(
    gen_kwargs: Dict[str, Any],
    processor: Optional[_BASE_LOGITS_PROCESSOR],
) -> None:
    if processor is None:
        return
    processors = gen_kwargs.get("logits_processor")
    if processors is None:
        gen_kwargs["logits_processor"] = LogitsProcessorList([processor])
    elif isinstance(processors, LogitsProcessorList):
        processors.append(processor)
    else:
        try:
            gen_kwargs["logits_processor"] = LogitsProcessorList(list(processors) + [processor])
        except Exception:
            gen_kwargs["logits_processor"] = LogitsProcessorList([processor])

def _is_qwen_moe_model_id(model_id: str) -> bool:
    lowered = model_id.lower()
    return "a3b" in lowered or "moe" in lowered


def _infer_qwen_model_size(model_id: str) -> Optional[str]:
    for size in ("2B", "4B", "8B", "32B"):
        if size in model_id:
            return size
    return None


def _estimate_qwen_vram_mb(
    model_id: str,
    training_mode: str,
    *,
    max_pixels: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[Optional[float], Optional[str]]:
    size = _infer_qwen_model_size(model_id)
    if not size:
        return None, None
    mode = training_mode if training_mode in QWEN_VRAM_ESTIMATE_GB else "official_lora"
    base_gb = QWEN_VRAM_ESTIMATE_GB.get(mode, {}).get(size)
    if base_gb is None:
        return None, None
    scale = 1.0
    if "Thinking" in model_id:
        scale *= QWEN_VRAM_THINKING_SCALE
    if max_pixels:
        try:
            pixel_scale = max_pixels / max(1, QWEN_VRAM_PIXEL_BASE)
        except Exception:
            pixel_scale = 1.0
        pixel_scale = min(QWEN_VRAM_PIXEL_SCALE_MAX, max(QWEN_VRAM_PIXEL_SCALE_MIN, pixel_scale))
        scale *= pixel_scale
    if batch_size and batch_size > 1:
        scale *= float(batch_size)
    estimate_mb = base_gb * 1024.0 * scale
    note = None
    if mode == "official_lora" and size in {"8B", "32B"}:
        note = "Official LoRA is very VRAM-hungry; 8B/32B often exceed 48GB even at smaller pixel budgets."
    if mode == "trl_qlora" and size == "32B" and "Thinking" in model_id:
        note = "32B Thinking QLoRA is experimental; some setups hit device-map gradient issues."
    return estimate_mb, note

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


def _unload_non_qwen_runtimes() -> None:
    """Free heavy inference runtimes except Qwen (SAM, detectors, classifier backbones)."""
    _unload_non_qwen_runtimes_impl(
        predictor_manager=predictor_manager,
        unload_sam3_text_fn=_unload_sam3_text_runtime,
        suspend_clip_fn=_suspend_clip_backbone,
        unload_dinov3_fn=_unload_dinov3_backbone,
        unload_detector_fn=_unload_detector_inference,
        torch_module=torch,
        logger=logger,
    )


def _unload_inference_runtimes() -> None:
    """Free heavy inference runtimes (SAM, detectors, Qwen, classifier backbones)."""
    _unload_inference_runtimes_impl(
        unload_non_qwen_fn=_unload_non_qwen_runtimes,
        unload_qwen_fn=_unload_qwen_runtime,
        torch_module=torch,
    )


def _prepare_for_training() -> None:
    """Free heavy inference runtimes before starting a training job."""
    _prepare_for_training_impl(unload_inference_runtimes_fn=_unload_inference_runtimes)


def _finalize_training_environment() -> None:
    _finalize_training_environment_impl(
        resume_classifier_fn=_resume_classifier_backbone,
        torch_module=torch,
    )


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


def _prepare_for_qwen_training() -> None:
    _prepare_for_training()


def _finalize_qwen_training_environment() -> None:
    _finalize_training_environment()


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
                active_head_normalize_embeddings = _resolve_active_head_normalize_embeddings(meta_obj, clf, default=True)
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

def _infer_clip_model_from_embedding_dim(embedding_dim: Optional[int], *, active_name: Optional[str] = None) -> Optional[str]:
    return _infer_clip_model_from_embedding_dim_impl(embedding_dim, active_name=active_name)

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


def _dinov3_resolve_device(requested: str) -> str:
    return _dinov3_resolve_device_impl(requested, cuda_disabled=dinov3_cuda_disabled)


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
        dinov3_resolve_device_fn=_dinov3_resolve_device,
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


def _ensure_clip_backbone_for_mining() -> Tuple[Optional[Any], Optional[Any]]:
    """Ensure a CLIP backbone is available for exemplar embedding/fp guard (raw CLIP, no classifier required)."""
    global clip_model, clip_preprocess, clip_model_name, clip_initialized
    state = {
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "clip_model_name": clip_model_name,
        "clip_initialized": clip_initialized,
    }
    clip_model_local, clip_preprocess_local = _ensure_clip_backbone_for_mining_impl(
        state=state,
        lock=clip_lock,
        clip_module=clip,
        device=device,
        default_model=DEFAULT_CLIP_MODEL,
        logger=logger,
    )
    clip_model = state["clip_model"]
    clip_preprocess = state["clip_preprocess"]
    clip_model_name = state["clip_model_name"]
    clip_initialized = state["clip_initialized"]
    return clip_model_local, clip_preprocess_local

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


def _resolve_sam3_device() -> torch.device:
    return _resolve_sam3_device_impl(
        SAM3_DEVICE_PREF,
        torch_module=torch,
        http_exception_cls=HTTPException,
        http_400=HTTP_400_BAD_REQUEST,
    )


def _resolve_sam3_mining_devices() -> List[torch.device]:
    return _resolve_sam3_mining_devices_impl(
        SAM3_DEVICE_PREF,
        torch_module=torch,
        logger=logger,
    )


def _require_sam3_for_prepass(enable_text: bool, enable_similarity: bool) -> None:
    _require_sam3_for_prepass_impl(
        enable_text,
        enable_similarity,
        sam3_import_error=SAM3_NATIVE_IMAGE_IMPORT_ERROR,
        build_sam3_image_model=build_sam3_image_model,
        sam3_image_processor=Sam3ImageProcessor,
    )


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
        self.device = _resolve_sam3_device()
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


def _build_backend_for_variant(variant: str):
    return _build_backend_for_variant_impl(
        variant,
        sam3_backend_cls=_Sam3Backend,
        sam1_backend_cls=_Sam1Backend,
    )


def _sam3_clear_device_pinned_caches(model: Any) -> None:
    _sam3_clear_device_pinned_caches_impl(model)


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
        resolve_device_fn=_resolve_sam3_device,
        sam3_import_error=SAM3_NATIVE_IMAGE_IMPORT_ERROR,
        build_model_fn=build_sam3_image_model,
        processor_cls=Sam3ImageProcessor,
        sam3_checkpoint=active_sam3_checkpoint,
        sam3_bpe_path=SAM3_BPE_PATH,
        clear_caches_fn=_sam3_clear_device_pinned_caches,
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
            backend = _build_backend_for_variant(variant)
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
        _, np_img = _decode_image_base64(base64_data)
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
        _, np_img = _decode_image_base64(image_base64)
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
        _, np_img = _decode_image_base64(image_base64)
        return np_img


def _resolve_qwen_device() -> str:
    return _resolve_qwen_device_impl(QWEN_DEVICE_PREF, torch_module=torch)


def _get_qwen_prompt_config() -> QwenPromptConfig:
    return _get_qwen_prompt_config_impl(qwen_prompt_config, qwen_config_lock)


def _set_qwen_prompt_config(config: QwenPromptConfig) -> None:
    global qwen_prompt_config
    qwen_prompt_config = _set_qwen_prompt_config_impl(qwen_prompt_config, config, qwen_config_lock)


def _render_qwen_prompt(
    prompt_type: str,
    *,
    items: Optional[str],
    image_type: Optional[str],
    extra_context: Optional[str],
) -> str:
    return _render_qwen_prompt_impl(
        prompt_type,
        items=items,
        image_type=image_type,
        extra_context=extra_context,
        get_config_fn=_get_qwen_prompt_config,
        http_exception_cls=HTTPException,
        http_422=HTTP_422_UNPROCESSABLE_ENTITY,
    )


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
        yolo_box = to_yolo(full_w, full_h, left, top, right, bottom)
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
        left, top, right, bottom = mask_to_bounding_box(mask)
        if right <= left or bottom <= top:
            continue
        yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
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
        left, top, right, bottom = mask_to_bounding_box(mask)
        if right <= left or bottom <= top:
            continue
        yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
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
        yolo_box = to_yolo(width, height, x_min, y_min, x_max, y_max)
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
            mask_payload = encode_binary_mask(mask_value)
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
        x_min, y_min, x_max, y_max = mask_to_bounding_box(mask)
        if x_max <= x_min or y_max <= y_min:
            continue
        yolo_box = to_yolo(width, height, x_min, y_min, x_max, y_max)
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
                mask=encode_binary_mask(mask),
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
            det_xyxy = yolo_to_corners(bbox, pil_img.width, pil_img.height)
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
    if not bboxes_xywh:
        empty = ([], []) if return_masks else []
        return empty
    labels: List[bool]
    if bbox_labels is None:
        labels = [True] * len(bboxes_xywh)
    else:
        labels = list(bbox_labels)
        if len(labels) < len(bboxes_xywh):
            labels.extend([True] * (len(bboxes_xywh) - len(labels)))
        elif len(labels) > len(bboxes_xywh):
            labels = labels[: len(bboxes_xywh)]
    img_state = state if state is not None else processor.set_image(pil_img)
    img_w, img_h = float(pil_img.width), float(pil_img.height)
    output = None
    for bbox_xywh, label in zip(bboxes_xywh, labels):
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
            det_xyxy = yolo_to_corners(bbox, pil_img.width, pil_img.height)
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
            device = _resolve_qwen_device()
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
            fallback_id = _strip_qwen_model_suffix(str(base_model_id))
            if fallback_id:
                try:
                    logger.warning("Qwen model %s not found; falling back to %s", base_model_id, fallback_id)
                    processor_source = str(adapter_path) if adapter_path else str(fallback_id)
                    model, processor = _load_with_online_retry(str(fallback_id), processor_source)
                except Exception as fallback_exc:  # noqa: BLE001
                    qwen_last_error = str(fallback_exc)
                    detail = _format_qwen_load_error(fallback_exc)
                    raise HTTPException(
                        status_code=HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"qwen_load_failed:{detail}",
                    ) from fallback_exc
            else:
                qwen_last_error = str(exc)
                detail = _format_qwen_load_error(exc)
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


def _evict_qwen_caption_entry(cache_key: str, cache_entry: Optional[Tuple[Any, Any]]) -> None:
    _evict_qwen_caption_entry_impl(
        cache_key,
        cache_entry,
        torch_module=torch,
        gc_module=gc,
    )


def _ensure_qwen_ready_for_caption(model_id_override: str) -> Tuple[Any, Any]:
    global qwen_device, qwen_last_error
    global qwen_caption_cache, qwen_caption_order
    state = {
        "qwen_caption_cache": qwen_caption_cache,
        "qwen_caption_order": qwen_caption_order,
        "qwen_device": qwen_device,
        "qwen_last_error": qwen_last_error,
    }
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
            resolve_device_fn=_resolve_qwen_device,
            device_pref=QWEN_DEVICE_PREF,
            torch_module=torch,
            load_qwen_model_fn=_load_qwen_vl_model,
            hf_offline_enabled_fn=_hf_offline_enabled,
            set_hf_offline_fn=_set_hf_offline,
            enable_hf_offline_defaults_fn=_enable_hf_offline_defaults,
            strip_model_suffix_fn=_strip_qwen_model_suffix,
            format_load_error_fn=_format_qwen_load_error,
            min_pixels=QWEN_MIN_PIXELS,
            max_pixels=QWEN_MAX_PIXELS,
            caption_cache_limit=QWEN_CAPTION_CACHE_LIMIT,
            evict_entry_fn=_evict_qwen_caption_entry,
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


def _resolve_qwen_variant_model_id(base_model_id: str, variant: Optional[str]) -> str:
    return _resolve_qwen_variant_model_id_impl(base_model_id, variant)


def _strip_qwen_model_suffix(model_id: str) -> Optional[str]:
    return _strip_qwen_model_suffix_impl(model_id)


def _caption_glossary_map(labelmap_glossary: Optional[str], labels: Sequence[str]) -> Dict[str, List[str]]:
    return _caption_glossary_map_impl(labelmap_glossary, labels)


def _caption_preferred_label(label: str, glossary_map: Optional[Dict[str, List[str]]] = None) -> str:
    return _caption_preferred_label_impl(label, glossary_map)


def _build_qwen_caption_prompt(
    user_prompt: str,
    label_hints: Sequence[QwenCaptionHint],
    image_width: int,
    image_height: int,
    include_counts: bool,
    include_coords: bool,
    max_boxes: int,
    detailed_mode: bool,
    restrict_to_labels: bool = True,
    labelmap_glossary: Optional[str] = None,
) -> Tuple[str, Dict[str, int], int, bool]:
    return _build_qwen_caption_prompt_impl(
        user_prompt,
        label_hints,
        image_width,
        image_height,
        include_counts,
        include_coords,
        max_boxes,
        detailed_mode,
        restrict_to_labels=restrict_to_labels,
        labelmap_glossary=labelmap_glossary,
    )


def _collapse_whitespace(text: str) -> str:
    return _collapse_whitespace_impl(text)


def _extract_caption_from_text(text: str, marker: Optional[str] = None) -> Tuple[str, bool]:
    return _extract_caption_from_text_impl(text, marker)


def _caption_needs_english_rewrite(text: str) -> bool:
    return _caption_needs_english_rewrite_impl(text)

_CAPTION_GENERIC_OPENERS = (
    "an aerial view",
    "aerial view",
    "from a high angle",
    "a drone image",
    "a bird's-eye view",
    "overhead view",
)


def _caption_starts_generic(text: str) -> bool:
    return _caption_starts_generic_impl(text)


def _caption_missing_labels(
    text: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    return _caption_missing_labels_impl(text, counts, glossary_map)


def _caption_needs_refine(
    caption: str,
    counts: Dict[str, int],
    detailed_mode: bool,
    include_counts: bool,
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> Tuple[bool, List[str]]:
    return _caption_needs_refine_impl(caption, counts, detailed_mode, include_counts, glossary_map)

def _format_qwen_load_error(exc: Exception) -> str:
    return _format_qwen_load_error_impl(exc, torch_module=torch)


def _sanitize_qwen_caption(text: str) -> str:
    return _sanitize_qwen_caption_impl(text)


_QWEN_THINKING_REASONING_RE = re.compile(
    r"(?:\bgot it\b|\blet'?s\b|\bfirst\b|\bsecond\b|\bthird\b|\bstep\b|\bi need\b|\bnow\b|\bthe task\b)",
    re.IGNORECASE,
)
_QWEN_CAPTION_META_RE = re.compile(
    r"(authoritative|as indicated|label hint|bounding box|bbox|coordinates|hinted|counts are provided)",
    re.IGNORECASE,
)


def _thinking_caption_needs_cleanup(cleaned: str, raw: Optional[str]) -> bool:
    return _thinking_caption_needs_cleanup_impl(cleaned, raw)


def _caption_needs_completion(caption: str) -> bool:
    return _caption_needs_completion_impl(caption)


def _caption_has_meta(caption: str) -> bool:
    return _caption_has_meta_impl(caption)


def _caption_needs_short_form(caption: str, max_words: int = 80, max_sentences: int = 2) -> bool:
    return _caption_needs_short_form_impl(caption, max_words=max_words, max_sentences=max_sentences)


def _resolve_qwen_caption_decode(payload: QwenCaptionRequest, is_thinking: bool) -> Dict[str, Any]:
    return _resolve_qwen_caption_decode_impl(payload, is_thinking)


def _adjust_prompt_for_thinking(prompt_text: str) -> str:
    return _adjust_prompt_for_thinking_impl(prompt_text)


def _run_qwen_caption_cleanup(
    prompt: str,
    pil_img: Image.Image,
    max_new_tokens: int,
    base_model_id: str,
    use_caption_cache: bool,
    model_id_override: Optional[str] = None,
    runtime_override: Optional[Tuple[Any, Any]] = None,
    allowed_labels: Optional[List[str]] = None,
    strict: bool = False,
    minimal_edit: bool = False,
) -> str:
    return _run_qwen_caption_cleanup_impl(
        prompt,
        pil_img,
        max_new_tokens,
        base_model_id,
        use_caption_cache,
        model_id_override=model_id_override,
        runtime_override=runtime_override,
        allowed_labels=allowed_labels,
        strict=strict,
        minimal_edit=minimal_edit,
        run_qwen_inference_fn=_run_qwen_inference,
        resolve_variant_fn=_resolve_qwen_variant_model_id,
        extract_caption_fn=_extract_caption_from_text,
        sanitize_caption_fn=_sanitize_qwen_caption,
    )


def _run_qwen_caption_merge(
    draft_caption: str,
    windowed_captions: Sequence[Tuple[int, int, int, str]],
    *,
    pil_img: Image.Image,
    base_model_id: str,
    runtime_resolver: Callable[[str], Tuple[Any, Any]],
    max_new_tokens: int,
    glossary_line: Optional[str] = None,
) -> str:
    return _run_qwen_caption_merge_impl(
        draft_caption,
        windowed_captions,
        pil_img=pil_img,
        base_model_id=base_model_id,
        runtime_resolver=runtime_resolver,
        max_new_tokens=max_new_tokens,
        glossary_line=glossary_line,
        run_qwen_inference_fn=_run_qwen_inference,
        resolve_variant_fn=_resolve_qwen_variant_model_id,
        extract_caption_fn=_extract_caption_from_text,
        sanitize_caption_fn=_sanitize_qwen_caption,
    )


def _resolve_qwen_window_size(
    requested: Optional[int],
    image_width: int,
    image_height: int,
    *,
    overlap: Optional[float] = None,
) -> int:
    return _resolve_qwen_window_size_impl(
        requested,
        image_width,
        image_height,
        overlap=overlap,
        default_size=QWEN_WINDOW_DEFAULT_SIZE,
        default_overlap=QWEN_WINDOW_DEFAULT_OVERLAP,
    )


def _resolve_qwen_window_overlap(requested: Optional[float]) -> float:
    return _resolve_qwen_window_overlap_impl(requested, default_overlap=QWEN_WINDOW_DEFAULT_OVERLAP)


def _window_positions(
    total: int,
    window: int,
    overlap: float,
    *,
    force_two: bool = False,
) -> List[int]:
    return _window_positions_impl(total, window, overlap, force_two=force_two)


def _allowed_caption_labels(label_hints: Sequence[QwenCaptionHint]) -> List[str]:
    return _allowed_caption_labels_impl(label_hints)


def _caption_is_degenerate(caption: str) -> bool:
    return _caption_is_degenerate_impl(caption)


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


def _filter_hints_for_window(
    label_hints: Sequence[QwenCaptionHint],
    x0: int,
    y0: int,
    window: int,
    image_width: int,
    image_height: int,
) -> List[QwenCaptionHint]:
    # Deprecated: use _group_hints_by_window to avoid duplicates.
    window_hints: List[QwenCaptionHint] = []
    x1 = x0 + window
    y1 = y0 + window
    for hint in label_hints:
        if not hint.bbox or len(hint.bbox) != 4:
            continue
        bx1, by1, bx2, by2 = hint.bbox
        try:
            cx = (float(bx1) + float(bx2)) * 0.5
            cy = (float(by1) + float(by2)) * 0.5
        except (TypeError, ValueError):
            continue
        if cx < x0 or cx > x1 or cy < y0 or cy > y1:
            continue
        nx1 = max(0.0, min(float(bx1) - x0, window))
        ny1 = max(0.0, min(float(by1) - y0, window))
        nx2 = max(0.0, min(float(bx2) - x0, window))
        ny2 = max(0.0, min(float(by2) - y0, window))
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        window_hints.append(
            QwenCaptionHint(
                label=hint.label,
                bbox=[nx1, ny1, nx2, ny2],
                confidence=hint.confidence,
            )
        )
    return window_hints


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
    device = qwen_device or _resolve_qwen_device()
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
    device = qwen_device or _resolve_qwen_device()
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
    device = qwen_device or _resolve_qwen_device()
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


def _generate_qwen_text(
    prompt: str,
    *,
    max_new_tokens: int = 128,
    use_system_prompt: bool = True,
) -> str:
    """Text-only generation with Qwen for small helper tasks (no images)."""
    return _generate_qwen_text_impl(
        prompt,
        max_new_tokens=max_new_tokens,
        use_system_prompt=use_system_prompt,
        system_prompt=(active_qwen_metadata or {}).get("system_prompt"),
        ensure_qwen_ready_fn=_ensure_qwen_ready,
        resolve_qwen_device_fn=_resolve_qwen_device,
    )


def _parse_prompt_candidates(raw: str, seen: set[str], limit: int) -> List[str]:
    """Parse and validate a comma/list output into cleaned candidates; returns [] if invalid."""
    return _parse_prompt_candidates_impl(raw, seen, limit)


def _generate_prompt_text(
    prompt: str,
    *,
    max_new_tokens: int = 128,
) -> str:
    """
    Text-only helper for prompt brainstorming/critique.
    Uses Qwen (text-only) and returns empty string on failure.
    """
    return _generate_prompt_text_impl(
        prompt,
        max_new_tokens=max_new_tokens,
        generate_text_fn=lambda text, tokens: _generate_qwen_text(text, max_new_tokens=tokens, use_system_prompt=False),
    )


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
    pil_img, np_img = _decode_image_base64(image_base64)
    token = hashlib.md5(np_img.tobytes()).hexdigest()
    _store_preloaded_image(token, np_img, variant)
    return pil_img, np_img, token


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


sam_preload_manager = SamPreloadManager()


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


class QwenPrepassResponse(BaseModel):
    detections: List[Dict[str, Any]]
    trace: List[AgentTraceEvent]
    warnings: Optional[List[str]] = None
    caption: Optional[str] = None
    trace_path: Optional[str] = None
    trace_full_path: Optional[str] = None


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


def _agent_set_active_detections(detections: Optional[Sequence[Dict[str, Any]]]) -> None:
    global _AGENT_ACTIVE_DETECTIONS
    if not detections:
        _AGENT_ACTIVE_DETECTIONS = []
        return
    _AGENT_ACTIVE_DETECTIONS = [dict(det) for det in detections if isinstance(det, dict)]


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


_AgentToolRunner = None  # legacy tool runner removed


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
                    generate_text_fn=_generate_qwen_text,
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
                caption_text = _sanitize_qwen_caption(caption_raw)
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
        discover_yolo_labelmap=_discover_yolo_labelmap,
        load_qwen_labelmap=_load_qwen_labelmap,
        load_sam3_meta=_load_sam3_dataset_metadata,
        load_qwen_meta=_load_qwen_dataset_metadata,
        normalize_glossary=_normalize_labelmap_glossary,
        default_glossary_fn=_default_agent_glossary_for_labelmap,
        collect_labels=_collect_labels_from_qwen_jsonl,
    )


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




def _agent_context_store(
    payload: Dict[str, Any],
    *,
    kind: str,
    max_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    return _agent_context_store_impl(
        payload,
        kind=kind,
        max_bytes=int(max_bytes or PREPASS_CONTEXT_CHUNK_BYTES),
        tile_store=_AGENT_TILE_CONTEXT_STORE,
        global_store=_AGENT_GLOBAL_CONTEXT_STORE,
    )


def _agent_context_chunk(
    handle: str,
    *,
    chunk_index: int,
    kind: str,
) -> Dict[str, Any]:
    return _agent_context_chunk_impl(
        handle,
        chunk_index=chunk_index,
        kind=kind,
        tile_store=_AGENT_TILE_CONTEXT_STORE,
        global_store=_AGENT_GLOBAL_CONTEXT_STORE,
    )


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



def _resolve_classifier_batch_size() -> int:
    return _resolve_classifier_batch()


def _predict_proba_batched(
    crops: Sequence[Image.Image],
    head: Dict[str, Any],
    *,
    batch_size: int,
) -> Optional[np.ndarray]:
    empty_cache_fn = None
    if torch.cuda.is_available():
        empty_cache_fn = torch.cuda.empty_cache
    return _predict_proba_batched_impl(
        crops,
        head,
        batch_size=batch_size,
        encode_batch_fn=lambda items, head_obj, bs: _encode_pil_batch_for_head(
            items, head=head_obj, batch_size_override=bs
        ),
        predict_proba_fn=_clip_head_predict_proba,
        empty_cache_fn=empty_cache_fn,
    )


def _agent_classifier_review(
    detections: List[Dict[str, Any]],
    *,
    pil_img: Optional[Image.Image],
    classifier_head: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    return _agent_classifier_review_impl(
        detections,
        pil_img=pil_img,
        classifier_head=classifier_head,
        resolve_batch_size_fn=_resolve_classifier_batch_size,
        predict_proba_fn=lambda crops, head, bs: _predict_proba_batched(crops, head, batch_size=bs),
        clip_head_background_indices_fn=_clip_head_background_indices,
        find_target_index_fn=_find_clip_head_target_index,
        clip_head_keep_mask_fn=_clip_head_keep_mask,
        readable_write_fn=_agent_readable_write,
        readable_format_bbox_fn=_agent_readable_format_bbox,
    )


def _agent_finalize_detections(
    detections: List[Dict[str, Any]],
    *,
    pil_img: Optional[Image.Image] = None,
    classifier_head: Optional[Dict[str, Any]] = None,
    img_w: int,
    img_h: int,
    labelmap: List[str],
    background: Optional[Sequence[str]],
    iou_thr: float,
    cross_iou: Optional[float],
    max_det: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    cleaned, rejected = _agent_sanitize_detection_items(
        detections,
        pil_img=pil_img,
        classifier_head=None,
        img_w=img_w,
        img_h=img_h,
        labelmap=labelmap,
        background=background,
    )
    filtered_scoreless = 0
    scoreless_iou = _AGENT_ACTIVE_SCORELESS_IOU or 0.0
    if scoreless_iou > 0:
        cleaned, filtered_scoreless = _agent_filter_scoreless_detections(
            cleaned,
            iou_thr=scoreless_iou,
        )
        if filtered_scoreless:
            _agent_readable_write(
                f"scoreless_filter: removed={filtered_scoreless} "
                f"iou>={scoreless_iou:.2f}"
            )
    reviewed, classifier_counts = _agent_classifier_review(
        cleaned,
        pil_img=pil_img,
        classifier_head=classifier_head,
    )
    merged = _agent_merge_detections(
        reviewed,
        iou_thr=iou_thr,
        max_det=max_det,
        cross_iou=cross_iou,
    )
    if classifier_counts.get("classifier_checked") or classifier_counts.get("classifier_unavailable"):
        _agent_readable_write(
            "final_review: "
            f"accepted={len(merged)} "
            f"rejected_classifier={classifier_counts.get('classifier_rejected', 0)} "
            f"classifier_unavailable={classifier_counts.get('classifier_unavailable', 0)}"
        )
    counts = {
        "input": len(detections),
        "accepted": len(merged),
        "rejected": int(rejected),
        "filtered_scoreless": int(filtered_scoreless),
        **classifier_counts,
    }
    return merged, counts


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
        proba_arr = _predict_proba_batched(
            pending_crops,
            classifier_head,
            batch_size=_resolve_classifier_batch_size(),
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


@_register_agent_tool("sam3_text")
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
    pil_img, _, _ = _agent_resolve_image(image_base64, image_token, "sam3")
    img_w, img_h = pil_img.size
    assigned_label = str(label).strip() if label is not None else ""
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
        manual_prompt = _render_qwen_prompt(
            prompt_type,
            items=item_list,
            image_type=(image_type or "").strip() or None,
            extra_context=(extra_context or "").strip() or None,
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
        x1, y1, x2, y2 = yolo_to_corners(det.bbox, crop_img.width, crop_img.height)
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
        classifier_path = _resolve_agent_clip_classifier_path(classifier_id)
        if classifier_path is not None:
            head = _load_clip_head_from_classifier(classifier_path)
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
    stored = _agent_context_store(public_payload, kind="tile")
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
    payload = _agent_context_chunk(context_handle, chunk_index=int(chunk_index), kind="tile")
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
    stored = _agent_context_store(payload, kind="global")
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
    payload = _agent_context_chunk(context_handle, chunk_index=int(chunk_index), kind="global")
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
    json_text = _extract_balanced_json(raw, "{", "}") or ""
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
        classifier_path = _resolve_agent_clip_classifier_path(classifier_id)
        if classifier_path is not None:
            head = _load_clip_head_from_classifier(classifier_path)
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
        classifier_path = _resolve_agent_clip_classifier_path(classifier_id)
        if classifier_path is not None:
            head = _load_clip_head_from_classifier(classifier_path)
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
    merged = _agent_merge_detections(
        cleaned,
        iou_thr=float(iou or 0.5),
        max_det=max_det,
        cross_iou=float(cross_iou) if cross_iou is not None else None,
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
        classifier_path = _resolve_agent_clip_classifier_path(classifier_id)
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
                    {"image": image_name, "detections": detections},
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


def _agent_tool_prompt_qwen(grid_enabled: bool = False) -> str:
    tools = _agent_tool_specs(grid_enabled=grid_enabled)
    tool_names: List[str] = []
    tool_descs: List[str] = []
    for tool in tools:
        fn = tool.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        tool_names.append(name)
        desc = str(fn.get("description") or "").strip()
        params = fn.get("parameters") or {}
        params_text = json.dumps(params, ensure_ascii=False)
        tool_descs.append(f"### {name}\n{name}: {desc} Input parameters: {params_text}")
    names_text = ", ".join(tool_names)
    descs_text = "\n\n".join(tool_descs)
    return (
        "# Tools\n\n"
        "## You have access to the following tools:\n\n"
        f"{descs_text}\n\n"
        "## When you need to call a tool, please insert the following command in your reply:\n\n"
        f"✿FUNCTION✿: The tool to use, should be one of [{names_text}]\n"
        "✿ARGS✿: The input of the tool\n"
        "✿RESULT✿: Tool results\n"
        "✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)\n"
        "Return ONLY the tool call, no other text.\n"
        "If the response already starts after ✿FUNCTION✿:, continue with the tool name and do not repeat the token.\n"
        "Keep args minimal; use only grid_cell, handles, labels, and intent as needed.\n"
    )


def _agent_full_trace_write(record: Dict[str, Any]) -> None:
    if _AGENT_TRACE_FULL_WRITER is None:
        return
    try:
        _AGENT_TRACE_FULL_WRITER(_agent_trace_full_jsonable(record))
    except Exception:
        return


def _agent_readable_write(line: str) -> None:
    _agent_readable_write_impl(
        line,
        writer=_AGENT_TRACE_READABLE_WRITER,
        to_console=PREPASS_READABLE_TO_CONSOLE,
    )


def _agent_run_prepass(
    payload: QwenPrepassRequest,
    *,
    pil_img: Image.Image,
    image_token: str,
    labelmap: List[str],
    glossary: str,
    as_tool_messages: bool = True,
    trace_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_full_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    model_id_override: Optional[str] = None,
) -> Tuple[
    List[Any],
    List[Dict[str, Any]],
    List[str],
    List[AgentTraceEvent],
    bool,
    bool,
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    List[Dict[str, Any]],
]:
    return _agent_run_prepass_impl(
        payload,
        pil_img=pil_img,
        image_token=image_token,
        labelmap=labelmap,
        glossary=glossary,
        as_tool_messages=as_tool_messages,
        trace_writer=trace_writer,
        trace_full_writer=trace_full_writer,
        model_id_override=model_id_override,
        agent_message_cls=QwenAgentMessage,
        content_item_cls=QwenAgentContentItem,
        agent_trace_event_cls=AgentTraceEvent,
        grid_spec_for_payload_fn=_agent_grid_spec_for_payload,
        readable_detection_line_fn=_agent_readable_detection_line,
        readable_write_fn=_agent_readable_write,
        tool_run_detector_fn=_agent_tool_run_detector,
        generate_sam3_synonyms_fn=_agent_generate_sam3_synonyms,
        generate_text_fn=_generate_qwen_text,
        extract_json_fn=_extract_balanced_json,
        default_synonyms=_DEFAULT_SAM3_SYNONYMS,
        label_key_fn=_glossary_label_key,
        sam3_prompt_variants_fn=_sam3_prompt_variants,
        tool_sam3_text_fn=_agent_tool_sam3_text,
        tool_sam3_similarity_fn=_agent_tool_sam3_similarity,
        quadrant_windows_fn=_agent_quadrant_windows_qwen,
        tool_look_and_inspect_fn=_agent_tool_look_and_inspect,
        tool_qwen_infer_fn=_agent_tool_qwen_infer,
        qwen_bbox_to_xyxy_fn=_qwen_bbox_to_xyxy,
        resolve_window_overlap_fn=_resolve_qwen_window_overlap,
        resolve_window_size_fn=_resolve_qwen_window_size,
        window_positions_fn=_window_positions,
        run_qwen_inference_fn=_run_qwen_inference,
        extract_caption_fn=_extract_caption_from_text,
        sanitize_caption_fn=_sanitize_qwen_caption,
        caption_is_degenerate_fn=_caption_is_degenerate,
        caption_needs_completion_fn=_caption_needs_completion,
        caption_has_meta_fn=_caption_has_meta,
        qwen_caption_fn=qwen_caption,
        caption_request_cls=QwenCaptionRequest,
        qwen_caption_cleanup_fn=_run_qwen_caption_cleanup,
        resolve_qwen_variant_fn=_resolve_qwen_variant_model_id,
        unload_qwen_fn=_unload_qwen_runtime,
        det_source_summary_fn=_agent_format_source_counts,
        det_label_counts_fn=_agent_label_counts_summary,
        detection_has_source_fn=_agent_detection_has_source,
        source_counts_fn=_agent_source_counts,
        format_source_counts_fn=_agent_format_source_counts,
        merge_prepass_fn=_agent_merge_prepass_detections,
        compact_tool_result_fn=_agent_compact_tool_result,
        active_detector_conf=_AGENT_ACTIVE_DETECTOR_CONF,
        active_sam3_score_thr=_AGENT_ACTIVE_SAM3_SCORE_THR,
        active_sam3_mask_thr=_AGENT_ACTIVE_SAM3_MASK_THR,
        trace_readable_enabled=_AGENT_TRACE_READABLE_WRITER is not None,
    )

def _agent_run_deep_prepass_part_a(
    payload: QwenPrepassRequest,
    *,
    pil_img: Image.Image,
    image_token: str,
    labelmap: List[str],
    glossary: str,
    trace_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_full_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_readable: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    return _agent_run_deep_prepass_part_a_impl(
        payload,
        pil_img=pil_img,
        image_token=image_token,
        labelmap=labelmap,
        glossary=glossary,
        run_detector_fn=_agent_tool_run_detector,
        attach_provenance_fn=_agent_attach_provenance,
        generate_sam3_synonyms_fn=_agent_generate_sam3_synonyms,
        generate_text_fn=_generate_qwen_text,
        extract_json_fn=_extract_balanced_json,
        default_synonyms=_DEFAULT_SAM3_SYNONYMS,
        label_key_fn=_glossary_label_key,
        sam3_text_windows_fn=_agent_sam3_text_windows,
        ensure_sam3_text_runtime_fn=_ensure_sam3_text_runtime,
        normalize_window_xyxy_fn=_normalize_window_xyxy,
        sam3_prompt_variants_fn=_sam3_prompt_variants,
        sam3_text_payloads_fn=_sam3_text_payloads_from_state,
        trace_writer=trace_writer,
        trace_full_writer=trace_full_writer,
        trace_readable=trace_readable,
        active_sam3_score_thr=_AGENT_ACTIVE_SAM3_SCORE_THR,
        active_sam3_mask_thr=_AGENT_ACTIVE_SAM3_MASK_THR,
        grid_overlap_ratio_default=PREPASS_GRID_OVERLAP_RATIO,
    )


def _agent_run_deep_prepass(
    payload: QwenPrepassRequest,
    *,
    pil_img: Image.Image,
    image_token: str,
    labelmap: List[str],
    glossary: str,
    trace_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_full_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_readable: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    return _agent_run_deep_prepass_impl(
        payload,
        pil_img=pil_img,
        image_token=image_token,
        labelmap=labelmap,
        glossary=glossary,
        run_part_a_fn=_agent_run_deep_prepass_part_a,
        cleanup_fn=_agent_deep_prepass_cleanup,
        select_exemplars_fn=_agent_select_similarity_exemplars,
        run_similarity_global_fn=lambda *args, **kwargs: _agent_run_similarity_global(
            *args,
            **kwargs,
            sam3_similarity_fn=_agent_tool_sam3_similarity,
        ),
        run_similarity_windowed_fn=lambda *args, **kwargs: _agent_run_similarity_expansion(
            *args,
            **kwargs,
            sam3_similarity_fn=_agent_tool_sam3_similarity,
            grid_overlap_ratio_default=PREPASS_GRID_OVERLAP_RATIO,
        ),
        finalize_provenance_fn=_agent_finalize_provenance,
        trace_writer=trace_writer,
        trace_full_writer=trace_full_writer,
        trace_readable=trace_readable,
    )


def _agent_run_deep_prepass_caption(
    payload: QwenPrepassRequest,
    *,
    pil_img: Image.Image,
    image_token: str,
    detections: List[Dict[str, Any]],
    model_id_override: Optional[str],
    glossary: Optional[str] = None,
    grid_for_log: Optional[Dict[str, Any]] = None,
    trace_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_full_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_readable: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    return _agent_run_deep_prepass_caption_impl(
        payload,
        pil_img=pil_img,
        image_token=image_token,
        detections=detections,
        model_id_override=model_id_override,
        glossary=glossary,
        grid_for_log=grid_for_log,
        caption_request_cls=QwenCaptionRequest,
        qwen_caption_fn=qwen_caption,
        sanitize_caption_fn=_sanitize_qwen_caption,
        label_counts_fn=_agent_label_counts_summary,
        qwen_bbox_to_xyxy_fn=_qwen_bbox_to_xyxy,
        xyxy_to_bbox_fn=_xyxy_to_qwen_bbox,
        grid_cell_for_window_bbox_fn=_agent_grid_cell_for_window_bbox,
        readable_format_bbox_fn=_agent_readable_format_bbox,
        unload_non_qwen_fn=_unload_non_qwen_runtimes,
        caption_window_hook=_CAPTION_WINDOW_HOOK,
        http_exception_cls=HTTPException,
        http_503_code=HTTP_503_SERVICE_UNAVAILABLE,
        trace_writer=trace_writer,
        trace_full_writer=trace_full_writer,
        trace_readable=trace_readable,
    )


def _agent_deep_prepass_cleanup(
    payload: QwenPrepassRequest,
    *,
    detections: List[Dict[str, Any]],
    pil_img: Image.Image,
    labelmap: List[str],
) -> Dict[str, Any]:
    return _agent_deep_prepass_cleanup_impl(
        payload,
        detections=detections,
        pil_img=pil_img,
        labelmap=labelmap,
        resolve_classifier_path_fn=_resolve_agent_clip_classifier_path,
        load_classifier_head_fn=_load_clip_head_from_classifier,
        active_classifier_head=active_classifier_head,
        background_from_head_fn=_agent_background_classes_from_head,
        sanitize_fn=_agent_sanitize_detection_items,
        default_iou=PREPASS_CLUSTER_IOU,
    )


def _agent_select_similarity_exemplars(
    payload: QwenPrepassRequest,
    *,
    detections: List[Dict[str, Any]],
    trace_readable: Optional[Callable[[str], None]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    min_score = float(payload.similarity_min_exemplar_score or 0.6)
    return _agent_select_similarity_exemplars_impl(
        min_score,
        detections=detections,
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
    _require_sam3_for_prepass(bool(payload.enable_sam3_text), bool(payload.enable_sam3_similarity))
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
        classifier_path = _resolve_agent_clip_classifier_path(classifier_id_for_run)
        if classifier_path is not None:
            head = _load_clip_head_from_classifier(classifier_path)
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
        model_id_override = _resolve_qwen_variant_model_id(base_model_id, desired_variant)
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
        submit_result = _agent_tool_submit_annotations(
            image_token=token,
            dataset_id=payload.dataset_id,
            classifier_id=payload.classifier_id,
            include_all=True,
            iou=payload.iou,
            cross_iou=payload.cross_iou,
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


def _clamp_conf_value(conf: float, warnings: List[str]) -> float:
    return _clamp_conf_value_impl(conf, warnings)


def _clamp_iou_value(iou: float, warnings: List[str]) -> float:
    return _clamp_iou_value_impl(iou, warnings)


def _clamp_max_det_value(max_det: int, warnings: List[str]) -> int:
    return _clamp_max_det_value_impl(max_det, warnings)


def _clamp_slice_params(
    slice_size: int,
    overlap: float,
    merge_iou: float,
    img_w: int,
    img_h: int,
    warnings: List[str],
) -> Tuple[int, float, float]:
    return _clamp_slice_params_impl(
        slice_size, overlap, merge_iou, img_w, img_h, warnings
    )


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


@dataclass
class ClipDatasetUploadJob:
    job_id: str
    root_dir: Path
    images_dir: Path
    labels_dir: Path
    created_at: float = field(default_factory=time.time)
    image_count: int = 0
    label_count: int = 0
    completed: bool = False


@dataclass
class QwenDatasetUploadJob:
    job_id: str
    root_dir: Path
    train_dir: Path
    val_dir: Path
    train_annotations: Path
    val_annotations: Path
    created_at: float = field(default_factory=time.time)
    run_name: Optional[str] = None
    train_count: int = 0
    val_count: int = 0
    completed: bool = False


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

GLOSSARY_LIBRARY_ROOT = Path(os.environ.get("GLOSSARY_ROOT", "./uploads/glossaries"))
GLOSSARY_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
GLOSSARY_LIBRARY_PATH = GLOSSARY_LIBRARY_ROOT / "glossaries.json"
GLOSSARY_LIBRARY_LOCK = threading.Lock()

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


def _prepass_recipe_dir(recipe_id: str, *, create: bool = False) -> Path:
    return _prepass_recipe_dir_impl(
        recipe_id,
        create=create,
        recipes_root=PREPASS_RECIPE_ROOT,
        sanitize_id_fn=_sanitize_yolo_run_id,
    )


def _prepass_recipe_meta_path(recipe_id: str) -> Path:
    return _prepass_recipe_meta_path_impl(
        recipe_id,
        recipes_root=PREPASS_RECIPE_ROOT,
        meta_filename=PREPASS_RECIPE_META,
        sanitize_id_fn=_sanitize_yolo_run_id,
    )


def _prepass_recipe_assets_dir(recipe_id: str, *, create: bool = False) -> Path:
    return _prepass_recipe_assets_dir_impl(
        recipe_id,
        create=create,
        recipes_root=PREPASS_RECIPE_ROOT,
        assets_dirname=PREPASS_RECIPE_ASSETS,
        sanitize_id_fn=_sanitize_yolo_run_id,
    )


def _sha256_path(path: Path) -> str:
    return _sha256_path_impl(path)


def _copy_tree_filtered(src: Path, dest: Path, *, keep_files: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    return _copy_tree_filtered_impl(src, dest, keep_files=keep_files, sha256_fn=_sha256_path)


def _unique_prepass_recipe_name(name: str) -> Tuple[str, Optional[str]]:
    return _unique_prepass_recipe_name_impl(name, list_recipes_fn=_list_prepass_recipes)


def _validate_prepass_recipe_manifest(manifest: Dict[str, Any], extract_dir: Path) -> None:
    return _validate_prepass_recipe_manifest_impl(
        manifest,
        extract_dir,
        sha256_fn=_sha256_path,
        path_is_within_root_fn=_path_is_within_root,
    )


def _list_prepass_recipes() -> List[Dict[str, Any]]:
    return _list_prepass_recipes_impl(recipes_root=PREPASS_RECIPE_ROOT, meta_filename=PREPASS_RECIPE_META)


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


@dataclass
class CalibrationJob:
    job_id: str
    status: str = "queued"
    message: str = "Queued"
    phase: str = "queued"
    progress: float = 0.0
    processed: int = 0
    total: int = 0
    request: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
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
PREPASS_RECIPE_ROOT = UPLOAD_ROOT / "prepass_recipes"
PREPASS_RECIPE_ROOT.mkdir(parents=True, exist_ok=True)
PREPASS_RECIPE_META = "recipe.json"
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
    schema_version: int = PREPASS_RECIPE_SCHEMA_VERSION
    renamed_from: Optional[str] = None
    notice: Optional[str] = None
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
CALIBRATION_FEATURES_VERSION = 3
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


def _purge_staging_dirs(
    root: Path,
    *,
    ttl_hours: Optional[int] = None,
    active_roots: Optional[set[str]] = None,
    prefix: Optional[str] = None,
) -> Dict[str, int]:
    """Delete old staging directories that are not active. Returns stats."""
    stats = {"deleted": 0, "bytes": 0}
    if ttl_hours is None:
        ttl_hours = STAGING_TTL_HOURS
    if not root.exists() or ttl_hours <= 0:
        return stats
    cutoff = time.time() - ttl_hours * 3600
    active_roots = active_roots or set()
    for entry in root.iterdir():
        try:
            if not entry.is_dir():
                continue
            if prefix and not entry.name.startswith(prefix):
                continue
            if str(entry.resolve()) in active_roots:
                continue
            mtime = entry.stat().st_mtime
            if mtime > cutoff:
                continue
            stats["bytes"] += _purge_directory(entry)
            stats["deleted"] += 1
        except Exception:
            continue
    return stats
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


def _purge_dataset_artifacts(dataset_id: str) -> None:
    return _purge_dataset_artifacts_impl(
        dataset_id,
        normalise_relative_path_fn=_normalise_relative_path,
        agent_mining_meta_root=AGENT_MINING_META_ROOT,
        agent_mining_det_cache_root=AGENT_MINING_DET_CACHE_ROOT,
        prompt_helper_preset_root=PROMPT_HELPER_PRESET_ROOT,
    )
CLIP_DATASET_JOBS: Dict[str, ClipDatasetUploadJob] = {}
CLIP_DATASET_JOBS_LOCK = threading.Lock()
QWEN_DATASET_JOBS: Dict[str, QwenDatasetUploadJob] = {}
QWEN_DATASET_JOBS_LOCK = threading.Lock()

MAX_JOB_LOGS = 250
MAX_QWEN_METRIC_POINTS: Optional[int] = None


def _job_log(job: ClipTrainingJob, message: str) -> None:
    _clip_job_log_impl(job, message, max_logs=MAX_JOB_LOGS, logger=logger)


def _clip_job_append_metric(job: ClipTrainingJob, metric: Dict[str, Any]) -> None:
    _clip_job_append_metric_impl(job, metric, max_points=2000)


def _job_update(job: ClipTrainingJob, *, status: Optional[str] = None, message: Optional[str] = None,
                progress: Optional[float] = None, error: Optional[str] = None,
                artifacts: Optional[Dict[str, Any]] = None) -> None:
    _clip_job_update_impl(
        job,
        status=status,
        message=message,
        progress=progress,
        error=error,
        artifacts=artifacts,
        max_logs=MAX_JOB_LOGS,
        logger=logger,
    )


def _qwen_job_log(job: QwenTrainingJob, message: str) -> None:
    _qwen_job_log_impl(job, message, max_logs=MAX_JOB_LOGS, logger=logger)


def _qwen_job_update(
    job: QwenTrainingJob,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    log_message: bool = True,
) -> None:
    _qwen_job_update_impl(
        job,
        status=status,
        message=message,
        progress=progress,
        error=error,
        result=result,
        log_message=log_message,
        max_logs=MAX_JOB_LOGS,
        logger=logger,
    )


def _serialize_job(job: ClipTrainingJob) -> Dict[str, Any]:
    return _serialize_clip_job_impl(job)


def _serialize_qwen_job(job: QwenTrainingJob) -> Dict[str, Any]:
    return _serialize_qwen_job_impl(job)


def _sam3_job_log(job: Sam3TrainingJob, message: str) -> None:
    _sam3_job_log_impl(job, message, max_logs=SAM3_MAX_LOG_LINES, logger=logger)


def _sam3_job_append_metric(job: Sam3TrainingJob, metric: Dict[str, Any]) -> None:
    _sam3_job_append_metric_impl(job, metric, max_points=SAM3_MAX_METRIC_POINTS)


def _sam3_job_update(
    job: Sam3TrainingJob,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    log_message: bool = True,
) -> None:
    _sam3_job_update_impl(
        job,
        status=status,
        message=message,
        progress=progress,
        error=error,
        result=result,
        log_message=log_message,
        max_logs=SAM3_MAX_LOG_LINES,
        logger=logger,
    )


def _serialize_sam3_job(job: Sam3TrainingJob) -> Dict[str, Any]:
    return _serialize_sam3_job_impl(job)


def _serialize_yolo_job(job: YoloTrainingJob) -> Dict[str, Any]:
    return _serialize_yolo_job_impl(job)


def _serialize_yolo_head_graft_job(job: YoloHeadGraftJob) -> Dict[str, Any]:
    return _serialize_yolo_head_graft_job_impl(job)


def _yolo_job_update(
    job: YoloTrainingJob,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    _yolo_job_update_impl(
        job,
        status=status,
        message=message,
        progress=progress,
        error=error,
        result=result,
    )


def _yolo_job_log(job: YoloTrainingJob, message: str) -> None:
    _yolo_job_log_impl(job, message, max_logs=YOLO_MAX_LOG_LINES, logger=logger)


def _yolo_head_graft_job_update(
    job: YoloHeadGraftJob,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    _yolo_head_graft_job_update_impl(
        job,
        status=status,
        message=message,
        progress=progress,
        error=error,
        result=result,
        audit_fn=_yolo_head_graft_audit,
    )


def _yolo_head_graft_job_log(job: YoloHeadGraftJob, message: str) -> None:
    _yolo_head_graft_job_log_impl(
        job,
        message,
        max_logs=YOLO_MAX_LOG_LINES,
        audit_fn=_yolo_head_graft_audit,
        logger=logger,
    )


def _yolo_head_graft_audit(
    job: YoloHeadGraftJob,
    message: str,
    *,
    level: str = "info",
    event: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        run_dir_value = (job.config or {}).get("paths", {}).get("run_dir")
        if not run_dir_value:
            return
        run_dir = Path(run_dir_value)
        if not run_dir.exists():
            return
        payload = {
            "timestamp": time.time(),
            "level": level,
            "event": event or "log",
            "message": message,
        }
        if extra:
            payload["extra"] = extra
        audit_path = run_dir / "head_graft_audit.jsonl"
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        return


def _yolo_head_graft_force_stop(job: YoloHeadGraftJob) -> bool:
    ident = job.thread_ident
    if not ident:
        return False
    try:
        import ctypes
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(ident), ctypes.py_object(SystemExit))
        if res == 0:
            return False
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(ident), None)
            return False
        return True
    except Exception:
        return False


def _yolo_job_append_metric(job: YoloTrainingJob, metric: Dict[str, Any]) -> None:
    _yolo_job_append_metric_impl(job, metric, max_points=2000)


def _serialize_rfdetr_job(job: RfDetrTrainingJob) -> Dict[str, Any]:
    return _serialize_rfdetr_job_impl(job)


def _rfdetr_job_update(
    job: RfDetrTrainingJob,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    _rfdetr_job_update_impl(
        job,
        status=status,
        message=message,
        progress=progress,
        error=error,
        result=result,
    )


def _rfdetr_job_log(job: RfDetrTrainingJob, message: str) -> None:
    _rfdetr_job_log_impl(job, message, max_logs=MAX_JOB_LOGS, logger=logger)


def _rfdetr_job_append_metric(job: RfDetrTrainingJob, metric: Dict[str, Any]) -> None:
    _rfdetr_job_append_metric_impl(job, metric, max_points=2000)


def _seg_job_log(job: SegmentationBuildJob, message: str) -> None:
    _seg_job_log_impl(job, message, max_logs=MAX_JOB_LOGS, logger=logger)


def _seg_job_update(
    job: SegmentationBuildJob,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    log_message: bool = True,
) -> None:
    _seg_job_update_impl(
        job,
        status=status,
        message=message,
        progress=progress,
        error=error,
        result=result,
        log_message=log_message,
        max_logs=MAX_JOB_LOGS,
        logger=logger,
    )


def _serialize_seg_job(job: SegmentationBuildJob) -> Dict[str, Any]:
    return _serialize_seg_job_impl(job)


def _log_qwen_get_request(endpoint: str, jobs: Sequence[QwenTrainingJob]) -> None:
    _log_qwen_get_request_impl(endpoint, jobs, logger)


def _qwen_job_append_metric(job: QwenTrainingJob, metric: Dict[str, Any]) -> None:
    _qwen_job_append_metric_impl(job, metric, max_points=MAX_QWEN_METRIC_POINTS)


def _summarize_qwen_metric(metric: Dict[str, Any]) -> str:
    return _summarize_qwen_metric_impl(metric)


def _ensure_qwen_dataset_signature(dataset_dir: Path, metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    return _ensure_qwen_dataset_signature_impl(
        dataset_dir,
        metadata,
        compute_dir_signature_fn=_compute_dir_signature,
        persist_metadata_fn=_persist_qwen_dataset_metadata,
    )


def _find_qwen_dataset_by_signature(signature: str) -> Optional[Path]:
    return _find_qwen_dataset_by_signature_impl(
        signature,
        dataset_root=QWEN_DATASET_ROOT,
        load_metadata_fn=_load_qwen_dataset_metadata,
        ensure_signature_fn=_ensure_qwen_dataset_signature,
    )


def _load_registry_dataset_metadata(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    return _load_registry_dataset_metadata_impl(
        dataset_dir,
        load_json_metadata_fn=_load_json_metadata,
        meta_name=DATASET_META_NAME,
    )


def _persist_dataset_metadata(dataset_dir: Path, metadata: Dict[str, Any]) -> None:
    _persist_dataset_metadata_impl(
        dataset_dir,
        metadata,
        meta_name=DATASET_META_NAME,
        logger=logger,
    )


def _coerce_dataset_metadata(dataset_dir: Path, raw_meta: Optional[Dict[str, Any]], source: str) -> Dict[str, Any]:
    return _coerce_dataset_metadata_impl(
        dataset_dir,
        raw_meta,
        source,
        dataset_context_key="dataset_context",
        compute_dir_signature_fn=_compute_dir_signature,
        persist_metadata_fn=_persist_dataset_metadata,
    )


def _list_all_datasets(prefer_registry: bool = True) -> List[Dict[str, Any]]:
    return _list_all_datasets_impl(
        prefer_registry=prefer_registry,
        dataset_registry_root=DATASET_REGISTRY_ROOT,
        sam3_dataset_root=SAM3_DATASET_ROOT,
        qwen_dataset_root=QWEN_DATASET_ROOT,
        load_registry_meta_fn=_load_registry_dataset_metadata,
        load_sam3_meta_fn=_load_sam3_dataset_metadata,
        load_qwen_meta_fn=_load_qwen_dataset_metadata,
        coerce_meta_fn=_coerce_dataset_metadata,
        yolo_labels_have_polygons_fn=_yolo_labels_have_polygons,
        convert_qwen_dataset_to_coco_fn=_convert_qwen_dataset_to_coco,
        convert_coco_dataset_to_yolo_fn=_convert_coco_dataset_to_yolo,
        load_dataset_glossary_fn=_load_dataset_glossary,
        glossary_preview_fn=_glossary_preview,
        count_caption_labels_fn=_count_caption_labels,
        count_dataset_images_fn=_count_dataset_images,
        logger=logger,
    )


def _load_qwen_dataset_metadata(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    return _load_qwen_dataset_metadata_impl(
        dataset_dir,
        meta_name=QWEN_METADATA_FILENAME,
        load_json_metadata_fn=_load_json_metadata,
    )


def _persist_qwen_dataset_metadata(dataset_dir: Path, metadata: Dict[str, Any]) -> None:
    _persist_qwen_dataset_metadata_impl(
        dataset_dir,
        metadata,
        meta_name=QWEN_METADATA_FILENAME,
        write_qwen_metadata_fn=_write_qwen_metadata,
    )


def _load_sam3_dataset_metadata(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    return _load_sam3_dataset_metadata_impl(
        dataset_dir,
        meta_name=SAM3_DATASET_META_NAME,
        load_json_metadata_fn=_load_json_metadata,
        persist_metadata_fn=_persist_sam3_dataset_metadata,
    )


def _persist_sam3_dataset_metadata(dataset_dir: Path, metadata: Dict[str, Any]) -> None:
    _persist_sam3_dataset_metadata_impl(
        dataset_dir,
        metadata,
        meta_name=SAM3_DATASET_META_NAME,
        logger=logger,
    )


def _count_dataset_images(dataset_root: Path) -> int:
    return _count_dataset_images_impl(dataset_root, iter_images_fn=_iter_yolo_images)


def _count_caption_labels(dataset_root: Path) -> Tuple[int, bool]:
    return _count_caption_labels_impl(dataset_root)


def _dir_size_bytes(path: Path) -> int:
    return _dir_size_bytes_impl(path)


def _active_run_paths_for_variant(variant: str) -> set[Path]:
    return _active_run_paths_for_variant_impl(
        variant=variant,
        jobs_lock=SAM3_TRAINING_JOBS_LOCK,
        jobs=SAM3_TRAINING_JOBS,
    )


def _describe_run_dir(run_dir: Path, variant: str, active_paths: set[Path]) -> Dict[str, Any]:
    return _describe_run_dir_impl(
        run_dir=run_dir,
        variant=variant,
        active_paths=active_paths,
        dir_size_fn=_dir_size_bytes,
    )


def _list_sam3_runs(variant: str) -> List[Dict[str, Any]]:
    return _list_sam3_runs_impl(
        variant=variant,
        job_root=SAM3_JOB_ROOT,
        dataset_root=SAM3_DATASET_ROOT,
        active_paths_fn=_active_run_paths_for_variant,
        describe_fn=_describe_run_dir,
    )


def _run_dir_for_request(run_id: str, variant: str) -> Path:
    return _run_dir_for_request_impl(
        run_id=run_id,
        variant=variant,
        job_root=SAM3_JOB_ROOT,
        http_exception_cls=HTTPException,
        http_400=HTTP_400_BAD_REQUEST,
        http_404=HTTP_404_NOT_FOUND,
    )


def _delete_run_scope(run_dir: Path, scope: str) -> Tuple[List[str], int]:
    return _delete_run_scope_impl(
        run_dir=run_dir,
        scope=scope,
        dir_size_fn=_dir_size_bytes,
        rmtree_fn=shutil.rmtree,
    )


def _sanitize_yolo_run_id(raw: str) -> str:
    return _sanitize_yolo_run_id_impl(raw)


def _yolo_run_dir(run_id: str, *, create: bool = False) -> Path:
    return _yolo_run_dir_impl(
        run_id,
        create=create,
        job_root=YOLO_JOB_ROOT,
        sanitize_fn=_sanitize_yolo_run_id,
        http_exception_cls=HTTPException,
    )


def _yolo_load_run_meta(run_dir: Path) -> Dict[str, Any]:
    return _yolo_load_run_meta_impl(run_dir, meta_name=YOLO_RUN_META_NAME)


def _yolo_write_run_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    _yolo_write_run_meta_impl(
        run_dir,
        meta,
        meta_name=YOLO_RUN_META_NAME,
        time_fn=time.time,
    )


def _yolo_prune_run_dir(run_dir: Path, keep_files: Optional[set[str]] = None) -> Dict[str, Any]:
    return _yolo_prune_run_dir_impl(
        run_dir,
        keep_files=keep_files,
        keep_files_default=YOLO_KEEP_FILES,
        dir_size_fn=_dir_size_bytes,
        meta_name=YOLO_RUN_META_NAME,
    )


def _sanitize_rfdetr_run_id(raw: str) -> str:
    return _sanitize_yolo_run_id(raw)


def _rfdetr_run_dir(run_id: str, *, create: bool = False) -> Path:
    return _rfdetr_run_dir_impl(
        run_id,
        create=create,
        job_root=RFDETR_JOB_ROOT,
        sanitize_fn=_sanitize_rfdetr_run_id,
        http_exception_cls=HTTPException,
    )


def _rfdetr_load_run_meta(run_dir: Path) -> Dict[str, Any]:
    return _rfdetr_load_run_meta_impl(run_dir, meta_name=RFDETR_RUN_META_NAME)


def _rfdetr_write_run_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    _rfdetr_write_run_meta_impl(
        run_dir,
        meta,
        meta_name=RFDETR_RUN_META_NAME,
        time_fn=time.time,
    )


def _rfdetr_prune_run_dir(run_dir: Path, keep_files: Optional[set[str]] = None) -> Dict[str, Any]:
    return _rfdetr_prune_run_dir_impl(
        run_dir,
        keep_files=keep_files,
        keep_files_default=RFDETR_KEEP_FILES,
        dir_size_fn=_dir_size_bytes,
    )


def _collect_yolo_artifacts(run_dir: Path) -> Dict[str, bool]:
    return _collect_yolo_artifacts_impl(run_dir, meta_name=YOLO_RUN_META_NAME)


def _collect_rfdetr_artifacts(run_dir: Path) -> Dict[str, bool]:
    return _collect_rfdetr_artifacts_impl(run_dir, meta_name=RFDETR_RUN_META_NAME)


def _flatten_metrics(obj: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _flatten_metrics_impl(obj, prefix=prefix, out=out)


def _lookup_metric(flat: Dict[str, Any], keys: List[str]) -> Optional[float]:
    return _lookup_metric_impl(flat, keys)


def _yolo_metrics_summary(run_dir: Path) -> Dict[str, float]:
    return _yolo_metrics_summary_impl(run_dir, read_csv_last_row_fn=_read_csv_last_row)


def _rfdetr_metrics_summary(run_dir: Path) -> Dict[str, float]:
    return _rfdetr_metrics_summary_impl(run_dir)


def _clean_metric_summary(summary: Dict[str, Optional[float]]) -> Dict[str, float]:
    return _clean_metric_summary_impl(summary)


def _list_yolo_runs() -> List[Dict[str, Any]]:
    return _list_yolo_runs_impl(
        job_root=YOLO_JOB_ROOT,
        dataset_cache_root=YOLO_DATASET_CACHE_ROOT,
        active_payload=_load_yolo_active(),
        load_meta_fn=_yolo_load_run_meta,
        collect_artifacts_fn=_collect_yolo_artifacts,
        meta_name=YOLO_RUN_META_NAME,
    )


def _list_rfdetr_runs() -> List[Dict[str, Any]]:
    return _list_rfdetr_runs_impl(
        job_root=RFDETR_JOB_ROOT,
        active_payload=_load_rfdetr_active(),
        load_meta_fn=_rfdetr_load_run_meta,
        collect_artifacts_fn=_collect_rfdetr_artifacts,
        meta_name=RFDETR_RUN_META_NAME,
    )


def _load_yolo_active() -> Dict[str, Any]:
    return _load_yolo_active_impl(YOLO_ACTIVE_PATH)


def _save_yolo_active(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _save_yolo_active_impl(payload, YOLO_ACTIVE_PATH)


def _load_rfdetr_active() -> Dict[str, Any]:
    return _load_rfdetr_active_impl(RFDETR_ACTIVE_PATH, RFDETR_JOB_ROOT, _save_rfdetr_active)


def _save_rfdetr_active(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _save_rfdetr_active_impl(payload, RFDETR_ACTIVE_PATH)


def _load_detector_default() -> Dict[str, Any]:
    return _load_detector_default_impl(DETECTOR_DEFAULT_PATH)


def _save_detector_default(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _save_detector_default_impl(payload, DETECTOR_DEFAULT_PATH, HTTPException)


def _detect_yolo_layout(dataset_root: Path) -> Dict[str, Any]:
    return _detect_yolo_layout_impl(dataset_root)


def _yolo_labels_have_polygons(
    labels_dir: Optional[Path],
    *,
    max_files: int = 200,
    max_lines: int = 2000,
) -> bool:
    return _yolo_labels_have_polygons_impl(
        labels_dir,
        max_files=max_files,
        max_lines=max_lines,
    )


def _resolve_dataset_entry(dataset_id: str) -> Optional[Dict[str, Any]]:
    return _resolve_dataset_entry_impl(dataset_id, list_all_datasets_fn=_list_all_datasets)


def _resolve_yolo_training_dataset(payload: YoloTrainRequest) -> Dict[str, Any]:
    return _resolve_yolo_training_dataset_impl(
        payload,
        resolve_dataset_entry_fn=_resolve_dataset_entry,
        resolve_sam3_or_qwen_dataset_fn=_resolve_sam3_or_qwen_dataset,
        compute_dir_signature_fn=_compute_dir_signature,
        sanitize_yolo_run_id_fn=_sanitize_yolo_run_id,
        detect_yolo_layout_fn=_detect_yolo_layout,
        yolo_labels_have_polygons_fn=_yolo_labels_have_polygons,
        stable_hash_fn=_stable_hash,
        yolo_cache_root=YOLO_DATASET_CACHE_ROOT,
        http_exception_cls=HTTPException,
    )


def _resolve_rfdetr_training_dataset(payload: RfDetrTrainRequest) -> Dict[str, Any]:
    return _resolve_rfdetr_training_dataset_impl(
        payload,
        resolve_dataset_entry_fn=_resolve_dataset_entry,
        resolve_sam3_or_qwen_dataset_fn=_resolve_sam3_or_qwen_dataset,
        load_sam3_meta_fn=_load_sam3_dataset_metadata,
        detect_yolo_layout_fn=_detect_yolo_layout,
        yolo_labels_have_polygons_fn=_yolo_labels_have_polygons,
        convert_yolo_dataset_to_coco_fn=_convert_yolo_dataset_to_coco,
        convert_qwen_dataset_to_coco_fn=_convert_qwen_dataset_to_coco,
        load_qwen_dataset_metadata_fn=_load_qwen_dataset_metadata,
        ensure_coco_supercategory_fn=_ensure_coco_supercategory,
        http_exception_cls=HTTPException,
    )


def _yolo_resolve_split_paths(dataset_root: Path, layout: Optional[str]) -> Tuple[str, str]:
    return _yolo_resolve_split_paths_impl(dataset_root, layout)


def _yolo_load_labelmap(labelmap_path: Path) -> List[str]:
    try:
        return [line.strip() for line in labelmap_path.read_text().splitlines() if line.strip()]
    except Exception:
        return []


def _yolo_load_run_labelmap(run_dir: Path) -> List[str]:
    labelmap_path = run_dir / "labelmap.txt"
    labels = _yolo_load_labelmap(labelmap_path)
    if labels:
        return labels
    data_yaml = run_dir / "data.yaml"
    if data_yaml.exists():
        try:
            payload = yaml.safe_load(data_yaml.read_text())
            names = payload.get("names") if isinstance(payload, dict) else None
            if isinstance(names, dict):
                return [names[k] for k in sorted(names.keys())]
            if isinstance(names, list):
                return [str(x) for x in names]
        except Exception:
            pass
    return []


def _validate_yolo_label_ids(labels_dir: Path, label_count: int) -> None:
    if label_count <= 0:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labelmap_empty")
    if not labels_dir.exists():
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labels_dir_missing")
    for label_path in labels_dir.rglob("*.txt"):
        rel_name = None
        try:
            rel_name = str(label_path.relative_to(labels_dir))
        except Exception:
            rel_name = str(label_path.name)
        try:
            with label_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line_no, line in enumerate(handle, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    if not parts:
                        continue
                    try:
                        raw = float(parts[0])
                    except Exception:
                        raise HTTPException(
                            status_code=HTTP_400_BAD_REQUEST,
                            detail=f"labelmap_class_id_invalid:{rel_name}:{line_no}",
                        )
                    idx = int(raw)
                    if abs(raw - idx) > 1e-6:
                        raise HTTPException(
                            status_code=HTTP_400_BAD_REQUEST,
                            detail=f"labelmap_class_id_non_int:{rel_name}:{line_no}",
                        )
                    if idx < 0 or idx >= label_count:
                        raise HTTPException(
                            status_code=HTTP_400_BAD_REQUEST,
                            detail=f"labelmap_class_id_out_of_range:{idx}/{label_count}:{rel_name}:{line_no}",
                        )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"labelmap_label_read_failed:{rel_name}:{exc}",
            ) from exc


def _rfdetr_load_labelmap(dataset_root: Path, coco_train_json: Optional[str] = None) -> List[str]:
    labelmap_path = dataset_root / "labelmap.txt"
    if labelmap_path.exists():
        return _yolo_load_labelmap(labelmap_path)
    coco_path = Path(coco_train_json) if coco_train_json else None
    if coco_path and coco_path.exists():
        try:
            data = json.loads(coco_path.read_text())
            categories = data.get("categories", [])
            categories = [c for c in categories if isinstance(c, dict) and "id" in c and "name" in c]
            categories.sort(key=lambda c: int(c.get("id", 0)))
            return [str(c["name"]) for c in categories]
        except Exception:
            return []
    return []


def _rfdetr_variant_info(task: str, variant: Optional[str]) -> Dict[str, Any]:
    task_norm = (task or "detect").lower().strip()
    variant_norm = (variant or "").strip().lower()
    if task_norm == "segment":
        variant_norm = "rfdetr-seg-preview"
    if not variant_norm:
        variant_norm = "rfdetr-medium" if task_norm == "detect" else "rfdetr-seg-preview"
    variant_map = {entry["id"]: entry for entry in RFDETR_VARIANTS}
    info = variant_map.get(variant_norm)
    if not info:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_variant_unknown")
    if task_norm == "segment" and info.get("task") != "segment":
        variant_norm = "rfdetr-seg-preview"
        info = variant_map.get(variant_norm)
    return info or {}


def _rfdetr_best_checkpoint(run_dir: Path) -> Optional[str]:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        path = run_dir / name
        if path.exists():
            return str(path)
    return None


def _rfdetr_parse_log_series(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        return []
    series: List[Dict[str, Any]] = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            series.append(json.loads(line))
        except Exception:
            continue
    return series


def _rfdetr_sanitize_metric(metric: Dict[str, Any]) -> Dict[str, Any]:
    def _coerce(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            try:
                return obj.item()
            except Exception:
                return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        return obj

    try:
        return json.loads(json.dumps(metric, default=_coerce))
    except Exception:
        return {}


def _rfdetr_normalize_aug_policy(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not raw or not isinstance(raw, dict):
        return None
    def _clamp(value: Any, default: float = 0.0, maximum: float = 1.0) -> float:
        try:
            num = float(value)
        except Exception:
            num = default
        return max(0.0, min(maximum, num))

    policy = {
        "hsv_h": _clamp(raw.get("hsv_h"), 0.0, 0.5),
        "hsv_s": _clamp(raw.get("hsv_s"), 0.0, 1.0),
        "hsv_v": _clamp(raw.get("hsv_v"), 0.0, 1.0),
        "blur_prob": _clamp(raw.get("blur_prob"), 0.0, 1.0),
        "gray_prob": _clamp(raw.get("gray_prob"), 0.0, 1.0),
    }
    kernel = raw.get("blur_kernel")
    try:
        kernel = int(kernel)
    except Exception:
        kernel = 0
    if kernel and kernel % 2 == 0:
        kernel += 1
    policy["blur_kernel"] = max(0, kernel)
    if not any(
        [
            policy["hsv_h"],
            policy["hsv_s"],
            policy["hsv_v"],
            policy["blur_prob"],
            policy["gray_prob"],
        ]
    ):
        return None
    return policy


def _rfdetr_install_augmentations(policy: Optional[Dict[str, Any]]) -> Optional[Tuple[Any, Any]]:
    if not policy:
        return None
    try:
        import random as _random
        import torchvision.transforms as tvt
        import rfdetr.datasets.coco as coco_mod
    except Exception:
        return None

    class _ImageOnlyTransform:
        def __init__(self, transform, p: float = 1.0) -> None:
            self.transform = transform
            self.p = float(p)

        def __call__(self, img, target):
            if self.p < 1.0 and _random.random() > self.p:
                return img, target
            return self.transform(img), target

    aug_transforms = []
    hsv_h = float(policy.get("hsv_h") or 0.0)
    hsv_s = float(policy.get("hsv_s") or 0.0)
    hsv_v = float(policy.get("hsv_v") or 0.0)
    if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
        color_jitter = tvt.ColorJitter(
            brightness=hsv_v,
            contrast=hsv_v,
            saturation=hsv_s,
            hue=min(0.5, hsv_h),
        )
        aug_transforms.append(_ImageOnlyTransform(color_jitter, p=1.0))
    gray_prob = float(policy.get("gray_prob") or 0.0)
    if gray_prob > 0:
        aug_transforms.append(_ImageOnlyTransform(tvt.RandomGrayscale(p=1.0), p=gray_prob))
    blur_prob = float(policy.get("blur_prob") or 0.0)
    blur_kernel = int(policy.get("blur_kernel") or 0)
    if blur_prob > 0 and blur_kernel >= 3:
        blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        blur = tvt.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))
        aug_transforms.append(_ImageOnlyTransform(blur, p=blur_prob))

    if not aug_transforms:
        return None

    original_make = coco_mod.make_coco_transforms
    original_make_square = coco_mod.make_coco_transforms_square_div_64

    def _wrap_make(make_func):
        def _wrapped(*args, **kwargs):
            base = make_func(*args, **kwargs)
            try:
                if hasattr(base, "transforms") and isinstance(base.transforms, list):
                    insert_idx = max(0, len(base.transforms) - 1)
                    base.transforms[insert_idx:insert_idx] = list(aug_transforms)
            except Exception:
                pass
            return base
        return _wrapped

    coco_mod.make_coco_transforms = _wrap_make(original_make)
    coco_mod.make_coco_transforms_square_div_64 = _wrap_make(original_make_square)
    return (original_make, original_make_square)


def _rfdetr_restore_augmentations(restore: Optional[Tuple[Any, Any]]) -> None:
    if not restore:
        return
    try:
        import rfdetr.datasets.coco as coco_mod
    except Exception:
        return
    try:
        coco_mod.make_coco_transforms = restore[0]
        coco_mod.make_coco_transforms_square_div_64 = restore[1]
    except Exception:
        pass


def _rfdetr_latest_checkpoint_epoch(run_dir: Path) -> Optional[int]:
    try:
        best = None
        for path in run_dir.glob("checkpoint*.pth"):
            name = path.name
            if not name.startswith("checkpoint") or not name.endswith(".pth"):
                continue
            token = name[len("checkpoint") : -len(".pth")]
            if not token.isdigit():
                continue
            value = int(token)
            if best is None or value > best:
                best = value
        return best
    except Exception:
        return None


def _rfdetr_monitor_training(job: RfDetrTrainingJob, run_dir: Path, total_epochs: int, stop_event: threading.Event) -> None:
    log_path = run_dir / "log.txt"
    last_pos = 0
    pending = ""
    last_epoch: Optional[int] = None
    while not stop_event.is_set():
        if job.cancel_event.is_set() or job.status not in {"running", "queued"}:
            break
        new_metrics: List[Dict[str, Any]] = []
        if log_path.exists():
            try:
                with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                    handle.seek(last_pos)
                    chunk = handle.read()
                    last_pos = handle.tell()
            except Exception:
                chunk = ""
            if chunk:
                chunk = pending + chunk
                pending = ""
                if not chunk.endswith("\n"):
                    last_newline = chunk.rfind("\n")
                    if last_newline == -1:
                        pending = chunk
                        chunk = ""
                    else:
                        pending = chunk[last_newline + 1 :]
                        chunk = chunk[: last_newline + 1]
                for line in chunk.splitlines():
                    if not line.strip():
                        continue
                    try:
                        metric = json.loads(line)
                    except Exception:
                        continue
                    metric = _rfdetr_sanitize_metric(metric)
                    if metric:
                        new_metrics.append(metric)
        if new_metrics:
            for metric in new_metrics:
                _rfdetr_job_append_metric(job, metric)
            latest = new_metrics[-1]
            epoch = latest.get("epoch")
            if isinstance(epoch, (int, float)):
                try:
                    epoch_idx = int(epoch)
                except Exception:
                    epoch_idx = None
                if epoch_idx is not None and epoch_idx != last_epoch:
                    last_epoch = epoch_idx
                    if total_epochs > 0:
                        progress = max(0.0, min(0.99, epoch_idx / total_epochs))
                        _rfdetr_job_update(job, progress=progress, message=f"Epoch {epoch_idx}/{total_epochs}")
        else:
            checkpoint_epoch = _rfdetr_latest_checkpoint_epoch(run_dir)
            if checkpoint_epoch is not None and checkpoint_epoch != last_epoch:
                last_epoch = checkpoint_epoch
                if total_epochs > 0:
                    progress = max(0.0, min(0.99, checkpoint_epoch / total_epochs))
                    _rfdetr_job_update(job, progress=progress, message=f"Epoch {checkpoint_epoch}/{total_epochs}")
        stop_event.wait(15.0)


def _yolo_write_data_yaml(run_dir: Path, dataset_root: Path, layout: Optional[str], labelmap_path: Optional[str]) -> Path:
    train_rel, val_rel = _yolo_resolve_split_paths(dataset_root, layout)
    names = []
    if labelmap_path:
        names = _yolo_load_labelmap(Path(labelmap_path))
    data = {
        "path": str(dataset_root),
        "train": train_rel,
        "val": val_rel,
        "names": names,
    }
    data_path = run_dir / "data.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False))
    if labelmap_path:
        try:
            shutil.copy2(labelmap_path, run_dir / "labelmap.txt")
        except Exception:
            pass
    return data_path


def _yolo_device_arg(devices: Optional[List[int]]) -> Optional[str]:
    if not devices:
        return None
    cleaned = [str(int(d)) for d in devices if isinstance(d, (int, str)) and str(d).strip().isdigit()]
    return ",".join(cleaned) if cleaned else None


def _yolo_p2_scale(model_id: str) -> Optional[str]:
    match = re.match(r"^yolov8([nsmlx])-p2$", model_id)
    if match:
        return match.group(1)
    return None


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



def _yolo_resolve_model_source(
    variant: Optional[str],
    task: str,
    from_scratch: bool,
    base_weights: Optional[str],
) -> Tuple[str, str]:
    model_id = (variant or "yolov8n").strip()
    if _yolo_p2_scale(model_id):
        return "cfg", "yolov8-p2.yaml"
    if base_weights:
        return "custom", base_weights
    if from_scratch:
        suffix = "-seg" if task == "segment" and "seg" not in model_id else ""
        return "cfg", f"{model_id}{suffix}.yaml"
    if task == "segment" and "seg" not in model_id:
        model_id = f"{model_id}-seg"
    return "weights", f"{model_id}.pt"


def _yolo_variant_base_yaml(variant: str, task: str, *, run_dir: Optional[Path] = None) -> Path:
    try:
        import ultralytics  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"yolo_unavailable:{exc}") from exc
    model_id = (variant or "yolov8n").strip()
    p2_scale = _yolo_p2_scale(model_id)
    base_cfg_dir = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "v8"
    if p2_scale:
        base_cfg = base_cfg_dir / "yolov8-p2.yaml"
        cfg_payload = yaml.safe_load(base_cfg.read_text())
        cfg_payload["scale"] = p2_scale
        target_dir = run_dir or Path(tempfile.mkdtemp(prefix="yolo_p2_cfg_", dir=str(UPLOAD_ROOT)))
        target = target_dir / f"yolov8{p2_scale}-p2.yaml"
        target.write_text(yaml.safe_dump(cfg_payload, sort_keys=False))
        return target
    suffix = "-seg" if task == "segment" and "seg" not in model_id else ""
    base_cfg = base_cfg_dir / f"{model_id}{suffix}.yaml"
    if not base_cfg.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_variant_yaml_missing")
    return base_cfg


def _yolo_write_variant_yaml(run_dir: Path, variant: str, task: str, nc: int) -> Path:
    base_cfg = _yolo_variant_base_yaml(variant, task, run_dir=run_dir)
    cfg_payload = yaml.safe_load(base_cfg.read_text())
    cfg_payload["nc"] = int(nc)
    target = run_dir / f"{Path(base_cfg).stem}_nc{nc}.yaml"
    target.write_text(yaml.safe_dump(cfg_payload, sort_keys=False))
    return target


def _yolo_write_head_graft_yaml(run_dir: Path, variant: str, base_nc: int, new_nc: int) -> Path:
    base_cfg = _yolo_variant_base_yaml(variant, "detect", run_dir=run_dir)
    cfg_payload = yaml.safe_load(base_cfg.read_text())
    head = cfg_payload.get("head") or []
    detect_idx = None
    for idx, entry in enumerate(head):
        if len(entry) >= 3 and entry[2] == "Detect":
            detect_idx = idx
            break
    if detect_idx is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_detect_layer_missing")
    detect_entry = list(head[detect_idx])
    detect_args = list(detect_entry[3]) if isinstance(detect_entry[3], list) else [detect_entry[3]]
    if detect_args:
        detect_args[0] = int(base_nc)
    else:
        detect_args = [int(base_nc)]
    detect_entry[3] = detect_args
    head[detect_idx] = detect_entry
    new_detect_args = list(detect_args)
    new_detect_args[0] = int(new_nc)
    new_detect = [detect_entry[0], detect_entry[1], "Detect", new_detect_args]
    head.append(new_detect)
    head.append([[-2, -1], 1, "ConcatHead", [int(base_nc), int(new_nc)]])
    cfg_payload["head"] = head
    cfg_payload["nc"] = int(base_nc + new_nc)
    target = run_dir / f"{Path(base_cfg).stem}_2xhead.yaml"
    target.write_text(yaml.safe_dump(cfg_payload, sort_keys=False))
    return target


def _yolo_find_detect_modules(model: Any) -> List[Any]:
    try:
        from ultralytics.nn.tasks import Detect  # type: ignore
    except Exception:
        return []
    modules: List[Any] = []
    for m in getattr(model, "model", []):
        if isinstance(m, Detect):
            modules.append(m)
    return modules


def _yolo_detect_layer_index(model: Any) -> int:
    detects = _yolo_find_detect_modules(model)
    if not detects:
        return max(0, len(getattr(model, "model", [])) - 1)
    for idx, m in enumerate(getattr(model, "model", [])):
        if m is detects[0]:
            return idx
    return max(0, len(getattr(model, "model", [])) - 1)


def _ensure_yolo_inference_runtime() -> Tuple[Any, List[str], Optional[str]]:
    global yolo_infer_model, yolo_infer_path, yolo_infer_labelmap, yolo_infer_task
    return _ensure_yolo_inference_runtime_impl(
        load_active_fn=_load_yolo_active,
        load_labelmap_fn=_yolo_load_labelmap,
        patch_ultralytics_fn=_patch_ultralytics_for_head_grafting,
        yolo_lock=YOLO_INFER_LOCK,
        get_state_fn=lambda: (yolo_infer_model, yolo_infer_path, yolo_infer_labelmap, yolo_infer_task),
        set_state_fn=lambda model, path, labelmap, task: _set_yolo_infer_state(model, path, labelmap, task),
        import_yolo_fn=lambda: __import__("ultralytics").YOLO,  # type: ignore[attr-defined]
        http_exception_cls=HTTPException,
    )


def _ensure_rfdetr_inference_runtime() -> Tuple[Any, List[str], Optional[str]]:
    global rfdetr_infer_model, rfdetr_infer_path, rfdetr_infer_labelmap, rfdetr_infer_task, rfdetr_infer_variant

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

    return _ensure_rfdetr_inference_runtime_impl(
        load_active_fn=_load_rfdetr_active,
        load_labelmap_fn=_yolo_load_labelmap,
        variant_info_fn=_rfdetr_variant_info,
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
        import_rfdetr_fn=_import_rfdetr,
        http_exception_cls=HTTPException,
        torch_available=torch.cuda.is_available,
        resolve_device_fn=lambda: "cuda" if torch.cuda.is_available() else "cpu",
    )


def _yolo_build_aug_args(aug: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not aug:
        return {}
    payload = dict(aug)
    mapping = {
        "flip_lr": "fliplr",
        "flip_ud": "flipud",
        "hsv_h": "hsv_h",
        "hsv_s": "hsv_s",
        "hsv_v": "hsv_v",
        "mosaic": "mosaic",
        "mixup": "mixup",
        "copy_paste": "copy_paste",
        "scale": "scale",
        "translate": "translate",
        "degrees": "degrees",
        "shear": "shear",
        "perspective": "perspective",
        "erasing": "erasing",
    }
    aug_args: Dict[str, Any] = {}
    for key, dest in mapping.items():
        if key in payload:
            aug_args[dest] = payload[key]
    return {k: v for k, v in aug_args.items() if v is not None}


def _yolo_parse_results_csv(results_path: Path) -> List[Dict[str, Any]]:
    if not results_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, raw in enumerate(reader):
                if not raw:
                    continue
                parsed: Dict[str, Any] = {}
                for key, value in raw.items():
                    if key is None or value is None:
                        continue
                    name = str(key).strip()
                    if not name:
                        continue
                    text = str(value).strip()
                    if text == "":
                        continue
                    try:
                        num = float(text)
                    except ValueError:
                        continue
                    if name == "epoch":
                        parsed["epoch"] = int(num) if float(num).is_integer() else num
                    else:
                        parsed[name] = num
                if "epoch" not in parsed:
                    parsed["epoch"] = idx + 1
                if parsed:
                    rows.append(parsed)
    except Exception:  # noqa: BLE001
        return []
    return rows


def _yolo_monitor_training(job: YoloTrainingJob, run_dir: Path, total_epochs: int, stop_event: threading.Event) -> None:
    results_path = run_dir / "train" / "results.csv"
    last_len = 0
    while not stop_event.is_set():
        if job.cancel_event.is_set() or job.status not in {"running", "queued"}:
            break
        series = _yolo_parse_results_csv(results_path)
        if series and len(series) > last_len:
            new_entries = series[last_len:]
            for metric in new_entries:
                _yolo_job_append_metric(job, metric)
            last_len = len(series)
            latest = series[-1]
            epoch = latest.get("epoch")
            if isinstance(epoch, (int, float)):
                try:
                    epoch_idx = int(epoch)
                except Exception:
                    epoch_idx = None
                if epoch_idx is not None and total_epochs > 0:
                    progress = max(0.0, min(0.99, epoch_idx / total_epochs))
                    _yolo_job_update(job, progress=progress, message=f"Epoch {epoch_idx}/{total_epochs}")
        stop_event.wait(12.0)


def _strip_checkpoint_optimizer(ckpt_path: Path) -> Tuple[bool, int, int]:
    """Remove optimizer/scheduler state from a torch checkpoint to shrink size."""
    before = ckpt_path.stat().st_size if ckpt_path.exists() else 0
    if not ckpt_path.exists() or before == 0:
        return False, before, before
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
        removed = False
        for key in ["optimizer", "optimizers", "lr_schedulers", "schedulers", "trainer"]:
            if key in payload:
                payload.pop(key, None)
                removed = True
        if not removed:
            return False, before, before
        tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_size = tmp_path.stat().st_size
        tmp_path.replace(ckpt_path)
        return True, before, tmp_size
    except Exception:
        return False, before, before


def _promote_run(run_id: str, variant: str) -> Dict[str, Any]:
    run_dir = _run_dir_for_request(run_id, variant)
    active_paths = _active_run_paths_for_variant(variant)
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
    stripped, before, after = _strip_checkpoint_optimizer(keep)
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


def _resolve_dataset_legacy(dataset_id: str) -> Path:
    return _resolve_dataset_legacy_impl(
        dataset_id,
        qwen_root=QWEN_DATASET_ROOT,
        sam3_root=SAM3_DATASET_ROOT,
        registry_root=DATASET_REGISTRY_ROOT,
        http_exception_cls=HTTPException,
    )


def _resolve_sam3_or_qwen_dataset(dataset_id: str) -> Path:
    return _resolve_sam3_or_qwen_dataset_impl(
        dataset_id,
        list_all_datasets_fn=_list_all_datasets,
        resolve_dataset_legacy_fn=_resolve_dataset_legacy,
    )


def _stable_hash(entries: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for item in entries:
        digest.update(item.encode("utf-8"))
    return digest.hexdigest()


def _decode_image_base64(
    image_base64: str,
    *,
    max_bytes: Optional[int] = BASE64_IMAGE_MAX_BYTES,
    max_dim: Optional[int] = BASE64_IMAGE_MAX_DIM,
    allow_downscale: bool = True,
) -> Tuple[Image.Image, np.ndarray]:
    """Decode base64 image with size/dimension guards and optional downscale."""
    if not image_base64:
        raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="image_payload_missing")
    raw = image_base64
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    if max_bytes:
        est_bytes = (len(raw) * 3) // 4
        if est_bytes > max_bytes * 2:
            raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="image_base64_too_large")
    try:
        data = base64.b64decode(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_base64:{exc}") from exc
    if max_bytes and len(data) > max_bytes:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="image_bytes_too_large")
    try:
        pil_img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_image:{exc}") from exc
    if max_dim:
        width, height = pil_img.size
        if width > max_dim or height > max_dim:
            if not allow_downscale:
                raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="image_too_large_dim")
            try:
                resample = getattr(Image, "Resampling", Image).LANCZOS  # Pillow 10 compat
            except Exception:
                resample = Image.LANCZOS
            pil_img = pil_img.copy()
            pil_img.thumbnail((max_dim, max_dim), resample)
    np_img = np.array(pil_img)
    return pil_img, np_img


def _compute_dir_signature(root: Path, *, allowed_exts: Optional[set[str]] = None) -> str:
    return _compute_dir_signature_impl(root, allowed_exts=allowed_exts)


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _agent_mining_meta_dir(dataset_id: str) -> Path:
    cleaned = (dataset_id or "").strip().replace("\\", "/").strip("/")
    safe = re.sub(r"[^A-Za-z0-9._/-]", "_", cleaned)
    meta_dir = (AGENT_MINING_META_ROOT / safe).resolve()
    if not _path_is_within_root(meta_dir, AGENT_MINING_META_ROOT.resolve()):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_dataset_invalid")
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir


def _agent_mining_cache_dir(dataset_id: str) -> Path:
    cleaned = (dataset_id or "").strip().replace("\\", "/").strip("/")
    safe = re.sub(r"[^A-Za-z0-9._/-]", "_", cleaned)
    cache_dir = (AGENT_MINING_DET_CACHE_ROOT / safe).resolve()
    if not _path_is_within_root(cache_dir, AGENT_MINING_DET_CACHE_ROOT.resolve()):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_dataset_invalid")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _normalize_agent_recipe_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _normalize_agent_recipe_steps_impl(steps)


def _parse_agent_recipe_schema_version(recipe_obj: Dict[str, Any]) -> Optional[int]:
    return _parse_agent_recipe_schema_version_impl(recipe_obj)


def _classify_agent_recipe_mode(recipe_obj: Dict[str, Any]) -> Literal["sam3_steps", "sam3_greedy", "legacy_steps"]:
    return _classify_agent_recipe_mode_impl(recipe_obj)


def _normalize_agent_recipe_execution_plan(recipe_obj: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_agent_recipe_execution_plan_impl(recipe_obj)


def _validate_agent_recipe_structure(recipe_obj: Dict[str, Any]) -> None:
    _validate_agent_recipe_structure_impl(recipe_obj)


def _compute_labelmap_hash(categories: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    return _compute_labelmap_hash_impl(categories)


def _compute_dataset_signature(dataset_id: str, dataset_root: Path, images: Dict[int, Dict[str, Any]], categories: List[Dict[str, Any]]) -> str:
    return _compute_dataset_signature_impl(dataset_id, dataset_root, images, categories)


def _save_exemplar_crop(
    *,
    exemplar: Dict[str, Any],
    images: Dict[int, Dict[str, Any]],
    crop_dir: Path,
    step_idx: int,
    crop_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    return _save_exemplar_crop_impl(
        exemplar=exemplar,
        images=images,
        crop_dir=crop_dir,
        step_idx=step_idx,
        crop_name=crop_name,
    )


def _export_hard_negative_replay(
    *,
    dataset_id: str,
    class_id: int,
    class_name: str,
    entries: Sequence[Dict[str, Any]],
    max_crops: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    return _export_hard_negative_replay_impl(
        dataset_id=dataset_id,
        class_id=class_id,
        class_name=class_name,
        entries=entries,
        max_crops=max_crops,
        replay_root=CLIP_NEGATIVE_REPLAY_ROOT,
        path_is_within_root_fn=_path_is_within_root,
        time_fn=time.time,
        log_fn=log_fn,
    )


def _persist_agent_recipe(
    dataset_id: Optional[str],
    class_id: Optional[int],
    class_name: Optional[str],
    label: str,
    recipe: Dict[str, Any],
    *,
    crop_overrides: Optional[Dict[str, bytes]] = None,
    clip_head_overrides: Optional[Dict[str, bytes]] = None,
    meta_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _persist_agent_recipe_impl(
        dataset_id,
        class_id,
        class_name,
        label,
        recipe,
        crop_overrides=crop_overrides,
        clip_head_overrides=clip_head_overrides,
        meta_overrides=meta_overrides,
        recipes_root=AGENT_MINING_RECIPES_ROOT,
        max_clip_head_bytes=AGENT_RECIPE_MAX_CLIP_HEAD_BYTES,
        max_crops=AGENT_RECIPE_MAX_CROPS,
        max_crop_bytes=AGENT_RECIPE_MAX_CROP_BYTES,
        resolve_dataset_fn=_resolve_sam3_or_qwen_dataset,
        load_coco_index_fn=_load_coco_index,
        compute_dataset_signature_fn=_compute_dataset_signature,
        compute_labelmap_hash_fn=_compute_labelmap_hash,
        resolve_clip_classifier_fn=_resolve_agent_clip_classifier_path,
        load_clip_head_fn=_load_clip_head_from_classifier,
        save_clip_head_artifacts_fn=_save_clip_head_artifacts,
        load_clip_head_artifacts_fn=_load_clip_head_artifacts,
        save_exemplar_crop_fn=_save_exemplar_crop,
        sanitize_prompts_fn=_sanitize_prompts,
        path_is_within_root_fn=_path_is_within_root,
    )


def _load_agent_recipe(recipe_id: str) -> Dict[str, Any]:
    return _load_agent_recipe_impl(
        recipe_id,
        recipes_root=AGENT_MINING_RECIPES_ROOT,
        path_is_within_root_fn=_path_is_within_root,
    )


def _load_agent_recipe_json_only(recipe_id: str) -> Dict[str, Any]:
    """Load an agent recipe payload without inlining crop_base64 blobs (suitable for inference/export)."""
    return _load_agent_recipe_json_only_impl(
        recipe_id,
        recipes_root=AGENT_MINING_RECIPES_ROOT,
        path_is_within_root_fn=_path_is_within_root,
    )


def _delete_agent_recipe(recipe_id: str) -> None:
    return _delete_agent_recipe_impl(
        recipe_id,
        recipes_root=AGENT_MINING_RECIPES_ROOT,
        path_is_within_root_fn=_path_is_within_root,
        http_exception_cls=HTTPException,
    )


def _list_agent_recipes(dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return _list_agent_recipes_impl(recipes_root=AGENT_MINING_RECIPES_ROOT, dataset_id=dataset_id)


def _ensure_recipe_zip(recipe: Dict[str, Any]) -> Path:
    return _ensure_recipe_zip_impl(recipe, recipes_root=AGENT_MINING_RECIPES_ROOT)


def _import_agent_recipe_zip_bytes(zip_bytes: bytes) -> Tuple[Optional[str], Dict[str, Any]]:
    return _import_agent_recipe_zip_bytes_impl(
        zip_bytes,
        recipes_root=AGENT_MINING_RECIPES_ROOT,
        max_json_bytes=AGENT_RECIPE_MAX_JSON_BYTES,
        max_clip_head_bytes=AGENT_RECIPE_MAX_CLIP_HEAD_BYTES,
        max_crops=AGENT_RECIPE_MAX_CROPS,
        max_crop_bytes=AGENT_RECIPE_MAX_CROP_BYTES,
        persist_recipe_fn=_persist_agent_recipe,
    )


def _persist_agent_cascade(label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _persist_agent_cascade_impl(
        label,
        payload,
        cascades_root=AGENT_MINING_CASCADES_ROOT,
        path_is_within_root_fn=_path_is_within_root,
    )


def _load_agent_cascade(cascade_id: str) -> Dict[str, Any]:
    return _load_agent_cascade_impl(
        cascade_id,
        cascades_root=AGENT_MINING_CASCADES_ROOT,
        path_is_within_root_fn=_path_is_within_root,
    )


def _list_agent_cascades() -> List[Dict[str, Any]]:
    return _list_agent_cascades_impl(cascades_root=AGENT_MINING_CASCADES_ROOT)


def _delete_agent_cascade(cascade_id: str) -> None:
    return _delete_agent_cascade_impl(
        cascade_id,
        cascades_root=AGENT_MINING_CASCADES_ROOT,
        path_is_within_root_fn=_path_is_within_root,
    )


def _ensure_cascade_zip(cascade: Dict[str, Any]) -> Path:
    return _ensure_cascade_zip_impl(
        cascade,
        cascades_root=AGENT_MINING_CASCADES_ROOT,
        recipes_root=AGENT_MINING_RECIPES_ROOT,
        classifiers_root=(UPLOAD_ROOT / "classifiers"),
        path_is_within_root_fn=_path_is_within_root,
        ensure_recipe_zip_fn=_ensure_recipe_zip,
        load_recipe_fn=_load_agent_recipe,
        resolve_classifier_fn=_resolve_agent_clip_classifier_path,
    )


def _import_agent_cascade_zip_bytes(zip_bytes: bytes) -> Dict[str, Any]:
    return _import_agent_cascade_zip_bytes_impl(
        zip_bytes,
        cascades_root=AGENT_MINING_CASCADES_ROOT,
        classifiers_root=(UPLOAD_ROOT / "classifiers"),
        max_json_bytes=AGENT_CASCADE_MAX_JSON_BYTES,
        classifier_allowed_exts=CLASSIFIER_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root,
        import_recipe_fn=_import_agent_recipe_zip_bytes,
        persist_cascade_fn=_persist_agent_cascade,
    )


def _sanitize_prompts(prompts: List[str]) -> List[str]:
    return _sanitize_prompts_impl(prompts)


def _refine_prompts_with_qwen(prompts: List[str]) -> List[str]:
    if not prompts or Qwen3VLForConditionalGeneration is None or QWEN_IMPORT_ERROR:
        return _sanitize_prompts(prompts)
    return _refine_prompts_with_qwen_impl(
        prompts,
        generate_prompt_text_fn=lambda prompt_text, tokens: _generate_prompt_text(prompt_text, max_new_tokens=tokens),
        sanitize_prompts_fn=_sanitize_prompts,
    )


def _qwen_self_filter_prompts(class_name: str, prompts: List[str]) -> List[str]:
    if not prompts or Qwen3VLForConditionalGeneration is None or QWEN_IMPORT_ERROR:
        return _sanitize_prompts(prompts)
    return _qwen_self_filter_prompts_impl(
        class_name,
        prompts,
        generate_prompt_text_fn=lambda prompt_text, tokens: _generate_prompt_text(prompt_text, max_new_tokens=tokens),
        sanitize_prompts_fn=_sanitize_prompts,
        humanize_class_name_fn=_humanize_class_name,
    )


def _expand_midpoints(values: List[float], *, fine_step: float = 0.05, clamp: Tuple[float, float] = (0.0, 1.0), limit: int = 20) -> List[float]:
    return _expand_midpoints_impl(values, fine_step=fine_step, clamp=clamp, limit=limit)


def _build_gt_index_for_class(
    gt_by_image_cat: Dict[int, Dict[int, List[List[float]]]], target_class: int
) -> Tuple[Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]], set[str], Dict[int, int]]:
    return _build_gt_index_for_class_impl(
        gt_by_image_cat,
        target_class,
        xywh_to_xyxy_fn=_xywh_to_xyxy,
    )


def _evaluate_prompt_candidate(
    prompt: str,
    threshold: float,
    *,
    cat_id: int,
    image_ids: List[int],
    gt_index: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]],
    other_gt_index: Optional[Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]]] = None,
    images: Dict[int, Dict[str, Any]],
    iou_threshold: float,
    max_dets: int,
    image_cache: Dict[int, Image.Image],
    cached_detections: Optional[Dict[int, List[Tuple[float, float, float, float, Optional[float]]]]] = None,
) -> Dict[str, Any]:
    return _evaluate_prompt_candidate_impl(
        prompt,
        threshold,
        cat_id=cat_id,
        image_ids=image_ids,
        gt_index=gt_index,
        other_gt_index=other_gt_index,
        images=images,
        iou_threshold=iou_threshold,
        max_dets=max_dets,
        image_cache=image_cache,
        cached_detections=cached_detections,
        run_sam3_text_inference_fn=_run_sam3_text_inference,
        yolo_to_xyxy_fn=_yolo_to_xyxy,
        iou_fn=_iou,
    )


def _collect_prompt_detections(
    prompt: str,
    min_threshold: float,
    *,
    image_ids: List[int],
    images: Dict[int, Dict[str, Any]],
    image_cache: Dict[int, Image.Image],
    max_dets: int,
) -> Dict[int, List[Tuple[float, float, float, float, Optional[float]]]]:
    return _collect_prompt_detections_impl(
        prompt,
        min_threshold,
        image_ids=image_ids,
        images=images,
        image_cache=image_cache,
        max_dets=max_dets,
        run_sam3_text_inference_fn=_run_sam3_text_inference,
        yolo_to_xyxy_fn=_yolo_to_xyxy,
    )


def _build_prompt_recipe(
    candidates: List[Dict[str, Any]],
    all_gt_keys: set[str],
    per_image_gt: Dict[int, int],
    images: Dict[int, Dict[str, Any]],
    image_ids: List[int],
    gt_index: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    return _build_prompt_recipe_impl(
        candidates,
        all_gt_keys,
        per_image_gt,
        images,
        image_ids,
        gt_index,
    )


def _serialize_prompt_helper_job(job: PromptHelperJob) -> Dict[str, Any]:
    return _serialize_prompt_helper_job_impl(job)


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


def _serialize_calibration_job(job: CalibrationJob) -> Dict[str, Any]:
    return _serialize_calibration_job_impl(job)


def _calibration_update(job: CalibrationJob, **kwargs: Any) -> None:
    _calibration_update_impl(job, **kwargs)


def _calibration_write_record_atomic(path: Path, record: Dict[str, Any]) -> None:
    _calibration_write_record_atomic_impl(path, record)


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
    return _calibration_prepass_worker_impl(
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
        set_device_pref_fn=lambda idx: _set_sam3_device_pref(idx),
    )


def _calibration_list_images(dataset_id: str) -> List[str]:
    return _calibration_list_images_impl(
        dataset_id, resolve_dataset_fn=_resolve_sam3_or_qwen_dataset
    )


def _calibration_sample_images(images: List[str], *, max_images: int, seed: int) -> List[str]:
    return _calibration_sample_images_impl(images, max_images=max_images, seed=seed)


def _calibration_cache_image(pil_img: Image.Image, sam_variant: Optional[str]) -> str:
    return _calibration_cache_image_impl(
        pil_img,
        sam_variant,
        store_preloaded_fn=_store_preloaded_image,
        default_variant_fn=_default_variant,
    )


def _calibration_hash_payload(payload: Dict[str, Any]) -> str:
    return _calibration_hash_payload_impl(payload)


def _calibration_safe_link(src: Path, dest: Path) -> None:
    _calibration_safe_link_impl(src, dest)


def _run_calibration_job(job: CalibrationJob, payload: CalibrationRequest) -> None:
    return _run_calibration_job_impl(
        job,
        payload,
        jobs=CALIBRATION_JOBS,
        jobs_lock=CALIBRATION_JOBS_LOCK,
        update_fn=_calibration_update,
        require_sam3_fn=_require_sam3_for_prepass,
        prepare_for_training_fn=_prepare_for_training,
        load_yolo_active_fn=_load_yolo_active,
        load_rfdetr_active_fn=_load_rfdetr_active,
        load_labelmap_meta_fn=_agent_load_labelmap_meta,
        list_images_fn=_calibration_list_images,
        sample_images_fn=_calibration_sample_images,
        calibration_root=CALIBRATION_ROOT,
        calibration_cache_root=CALIBRATION_CACHE_ROOT,
        prepass_request_cls=QwenPrepassRequest,
        active_classifier_head=active_classifier_head,
        calibration_features_version=CALIBRATION_FEATURES_VERSION,
        write_record_fn=_calibration_write_record_atomic,
        hash_payload_fn=_calibration_hash_payload,
        safe_link_fn=_calibration_safe_link,
        prepass_worker_fn=_calibration_prepass_worker,
        unload_inference_runtimes_fn=_unload_inference_runtimes,
        resolve_dataset_fn=_resolve_sam3_or_qwen_dataset,
        cache_image_fn=_calibration_cache_image,
        run_prepass_fn=_agent_run_deep_prepass,
        logger=logger,
        http_exception_cls=HTTPException,
        root_dir=Path(__file__).resolve().parent,
    )


def _start_calibration_job(payload: CalibrationRequest) -> CalibrationJob:
    return _start_calibration_job_impl(
        payload,
        job_cls=CalibrationJob,
        jobs=CALIBRATION_JOBS,
        jobs_lock=CALIBRATION_JOBS_LOCK,
        run_job_fn=_run_calibration_job,
    )


def _cancel_calibration_job(job_id: str) -> CalibrationJob:
    return _cancel_calibration_job_impl(
        job_id,
        jobs=CALIBRATION_JOBS,
        jobs_lock=CALIBRATION_JOBS_LOCK,
        http_exception_cls=HTTPException,
        time_fn=time.time,
    )
def _run_prompt_helper_job(job: PromptHelperJob, payload: PromptHelperRequest) -> None:
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    job.status = "running"
    job.message = "Loading dataset…"
    job.request = payload.dict()
    job.updated_at = time.time()
    try:
        dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
        coco, gt_by_image_cat, images = _load_coco_index(dataset_root)
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
        dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
        coco, gt_by_image_cat, images = _load_coco_index(dataset_root)
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
        dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
        coco, gt_by_image_cat, images = _load_coco_index(dataset_root)
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
        gt_index_all, all_gt_keys_all, per_image_gt_all = _build_gt_index_for_class(gt_by_image_cat, payload.class_id)
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
            detections = _collect_prompt_detections(
                prompt_entry.prompt,
                min_threshold,
                image_ids=eval_ids,
                images=images,
                image_cache=image_cache,
                max_dets=payload.max_dets,
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
                metrics = _evaluate_prompt_candidate(
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
        recipe, coverage_by_image = _build_prompt_recipe(
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

        dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
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

        coco, gt_by_image_cat, images = _load_coco_index(dataset_root)
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
        clip_head_path = _resolve_agent_clip_classifier_path(payload.clip_head_classifier_path)
        if clip_head_path is None:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_clip_head_required")
        clip_head = _load_clip_head_from_classifier(clip_head_path)
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
        base_prompts_all = _sanitize_prompts(base_prompts_all)
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
            base = _sanitize_prompts([*base, *user_extras]) or [name]
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
            base_keep = _sanitize_prompts([*base, *base_prompts_all])
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
    clip_head_path = _resolve_agent_clip_classifier_path(payload.clip_head_classifier_path)
    if clip_head_path is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_clip_head_required")
    # Validate early so we fail fast (no background job created).
    _load_clip_head_from_classifier(clip_head_path)
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


@app.post("/agent_mining/jobs")
def start_agent_mining_job(payload: AgentMiningRequest):
    job = _start_agent_mining_job(payload)
    return _serialize_agent_mining_job(job)


@app.get("/agent_mining/jobs")
def list_agent_mining_jobs():
    _prune_job_registry(AGENT_MINING_JOBS, AGENT_MINING_JOBS_LOCK)
    with AGENT_MINING_JOBS_LOCK:
        jobs = list(AGENT_MINING_JOBS.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_serialize_agent_mining_job(j) for j in jobs]


@app.get("/agent_mining/jobs/{job_id}")
def get_agent_mining_job(job_id: str):
    with AGENT_MINING_JOBS_LOCK:
        job = AGENT_MINING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_mining_job_not_found")
    return _serialize_agent_mining_job(job)


@app.post("/agent_mining/jobs/{job_id}/cancel")
def cancel_agent_mining_job(job_id: str):
    job = _cancel_agent_mining_job(job_id)
    return _serialize_agent_mining_job(job)


@app.get("/agent_mining/results/latest")
def get_latest_agent_mining_result():
    with AGENT_MINING_JOBS_LOCK:
        jobs = [j for j in AGENT_MINING_JOBS.values() if j.status in {"running", "completed", "failed", "cancelled"}]
    if not jobs:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_mining_result_not_found")
    jobs.sort(key=lambda j: j.updated_at, reverse=True)
    return _serialize_agent_mining_job(jobs[0])


@app.get("/agent_mining/cache_size")
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


@app.post("/agent_mining/cache/purge")
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
    clip_head_min_prob_override: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional extra CLIP-head probability threshold applied in addition to the recipe's baked-in head thresholds. "
            "Effective min_prob is max(recipe_min_prob, clip_head_min_prob_override)."
        ),
    )
    clip_head_margin_override: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional extra CLIP-head margin threshold applied in addition to the recipe's baked-in head thresholds. "
            "Effective margin is max(recipe_margin, clip_head_margin_override)."
        ),
    )
    extra_clip_classifier_path: Optional[str] = Field(
        None,
        description=(
            "Optional extra CLIP classifier head (trained via the CLIP tab) to apply after the recipe runs. "
            "Useful for adding a classifier-based filter to crop-bank recipes."
        ),
    )
    extra_clip_min_prob: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum probability for the step output class when using extra_clip_classifier_path. "
            "This filter is applied in addition to any CLIP filtering already baked into the recipe."
        ),
    )
    extra_clip_margin: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional margin for the step output class when using extra_clip_classifier_path "
            "(p(target) - max(p(other)) must be >= margin)."
        ),
    )

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
    clip_head_min_prob_override: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional extra CLIP-head probability threshold applied in addition to the recipe's baked-in head thresholds. "
            "Effective min_prob is max(recipe_min_prob, clip_head_min_prob_override)."
        ),
    )
    clip_head_margin_override: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional extra CLIP-head margin threshold applied in addition to the recipe's baked-in head thresholds. "
            "Effective margin is max(recipe_margin, clip_head_margin_override)."
        ),
    )
    extra_clip_classifier_path: Optional[str] = Field(
        None,
        description=(
            "Optional extra CLIP classifier head (trained via the CLIP tab) to apply after the recipe runs. "
            "Useful for adding a classifier-based filter to crop-bank recipes."
        ),
    )
    extra_clip_min_prob: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum probability for the step output class when using extra_clip_classifier_path. "
            "This filter is applied in addition to any CLIP filtering already baked into the recipe."
        ),
    )
    extra_clip_margin: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional margin for the step output class when using extra_clip_classifier_path "
            "(p(target) - max(p(other)) must be >= margin)."
        ),
    )

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


@app.post("/agent_mining/apply_image", response_model=Sam3TextPromptResponse)
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


@app.post("/agent_mining/apply_image_chain", response_model=Sam3TextPromptResponse)
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
                recipe_obj = _load_agent_recipe_json_only(step.recipe_id)
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
                    if _path_is_within_root(candidate, AGENT_MINING_RECIPES_ROOT.resolve()) and candidate.exists():
                        head_recipe_id = str(rid)
                        break

            clip_head: Optional[Dict[str, Any]] = None
            if head_recipe_id:
                try:
                    head_recipe = _load_agent_recipe_json_only(head_recipe_id)
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
                if _path_is_within_root(recipe_root, AGENT_MINING_RECIPES_ROOT.resolve()):
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


@app.post("/agent_mining/recipes", response_model=Dict[str, Any])
def agent_mining_save_recipe(payload: AgentRecipeExportRequest):
    recipe = _persist_agent_recipe(
        payload.dataset_id,
        payload.class_id,
        payload.class_name,
        payload.label,
        payload.recipe,
    )
    return recipe


@app.get("/agent_mining/recipes", response_model=List[Dict[str, Any]])
def agent_mining_list_recipes(dataset_id: Optional[str] = None):
    return _list_agent_recipes(dataset_id)


@app.get("/agent_mining/recipes/{recipe_id}", response_model=Dict[str, Any])
def agent_mining_get_recipe(recipe_id: str):
    return _load_agent_recipe(recipe_id)


@app.get("/agent_mining/recipes/{recipe_id}/export")
def agent_mining_export_recipe(recipe_id: str):
    recipe = _load_agent_recipe(recipe_id)
    zip_path = _ensure_recipe_zip(recipe)
    filename = f"{recipe_id}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    try:
        stream = zip_path.open("rb")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_export_failed:{exc}") from exc
    return StreamingResponse(stream, media_type="application/zip", headers=headers)


@app.post("/agent_mining/recipes/import", response_model=Dict[str, Any])
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


@app.delete("/agent_mining/recipes/{recipe_id}")
def agent_mining_delete_recipe(recipe_id: str):
    _delete_agent_recipe(recipe_id)
    return {"id": recipe_id, "deleted": True}


@app.post("/agent_mining/cascades", response_model=Dict[str, Any])
def agent_mining_save_cascade(payload: AgentCascadeSaveRequest):
    cascade_payload = {
        "steps": [s.dict() for s in payload.steps],
        "dedupe": payload.dedupe.dict(),
    }
    return _persist_agent_cascade(payload.label, cascade_payload)


@app.get("/agent_mining/cascades", response_model=List[Dict[str, Any]])
def agent_mining_list_cascades():
    return _list_agent_cascades()


@app.get("/agent_mining/cascades/{cascade_id}", response_model=Dict[str, Any])
def agent_mining_get_cascade(cascade_id: str):
    return _load_agent_cascade(cascade_id)


@app.delete("/agent_mining/cascades/{cascade_id}")
def agent_mining_delete_cascade(cascade_id: str):
    _delete_agent_cascade(cascade_id)
    return {"id": cascade_id, "deleted": True}


@app.get("/agent_mining/cascades/{cascade_id}/export")
def agent_mining_export_cascade(cascade_id: str):
    cascade = _load_agent_cascade(cascade_id)
    zip_path = _ensure_cascade_zip(cascade)
    filename = f"{cascade_id}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    try:
        stream = zip_path.open("rb")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_cascade_export_failed:{exc}") from exc
    return StreamingResponse(stream, media_type="application/zip", headers=headers)


@app.post("/agent_mining/cascades/import", response_model=Dict[str, Any])
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


@app.post("/sam3/prompt_helper/suggest")
def prompt_helper_suggest(payload: PromptHelperSuggestRequest):
    return _suggest_prompts_for_dataset(payload)


@app.post("/sam3/prompt_helper/expand")
def prompt_helper_expand(payload: PromptRecipeExpandRequest):
    dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
    coco, _, _ = _load_coco_index(dataset_root)
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


def _list_prompt_helper_presets() -> List[Dict[str, Any]]:
    return _list_prompt_helper_presets_impl(presets_root=PROMPT_HELPER_PRESET_ROOT)


def _load_prompt_helper_preset(preset_id: str) -> Dict[str, Any]:
    return _load_prompt_helper_preset_impl(
        preset_id,
        presets_root=PROMPT_HELPER_PRESET_ROOT,
        path_is_within_root_fn=_path_is_within_root,
    )


def _save_prompt_helper_preset(label: str, dataset_id: str, prompts_by_class: Dict[int, List[str]]) -> Dict[str, Any]:
    return _save_prompt_helper_preset_impl(
        label,
        dataset_id,
        prompts_by_class,
        presets_root=PROMPT_HELPER_PRESET_ROOT,
        path_is_within_root_fn=_path_is_within_root,
    )


@app.post("/sam3/prompt_helper/jobs")
def start_prompt_helper_job(payload: PromptHelperRequest):
    job = _start_prompt_helper_job(payload)
    return _serialize_prompt_helper_job(job)


@app.post("/sam3/prompt_helper/search")
def start_prompt_helper_search(payload: PromptHelperSearchRequest):
    job = _start_prompt_helper_search_job(payload)
    return _serialize_prompt_helper_job(job)


@app.post("/sam3/prompt_helper/recipe")
def start_prompt_helper_recipe(payload: PromptRecipeRequest):
    job = _start_prompt_recipe_job(payload)
    return _serialize_prompt_helper_job(job)

@app.get("/sam3/prompt_helper/presets")
def list_prompt_helper_presets():
    return _list_prompt_helper_presets()


@app.get("/sam3/prompt_helper/presets/{preset_id}")
def get_prompt_helper_preset(preset_id: str):
    return _load_prompt_helper_preset(preset_id)


@app.post("/sam3/prompt_helper/presets")
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
    preset = _save_prompt_helper_preset(label, dataset_id, normalized)
    return preset


@app.get("/sam3/prompt_helper/jobs")
def list_prompt_helper_jobs():
    _prune_job_registry(PROMPT_HELPER_JOBS, PROMPT_HELPER_JOBS_LOCK)
    with PROMPT_HELPER_JOBS_LOCK:
        jobs = list(PROMPT_HELPER_JOBS.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_serialize_prompt_helper_job(j) for j in jobs]


@app.get("/sam3/prompt_helper/jobs/{job_id}")
def get_prompt_helper_job(job_id: str):
    with PROMPT_HELPER_JOBS_LOCK:
        job = PROMPT_HELPER_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prompt_helper_job_not_found")
    return _serialize_prompt_helper_job(job)


@app.post("/segmentation/build/jobs")
def start_segmentation_build_job(request: SegmentationBuildRequest):
    job = _start_segmentation_build_job(request)
    return _serialize_seg_job(job)


@app.get("/segmentation/build/jobs")
def list_segmentation_build_jobs():
    _prune_job_registry(SEGMENTATION_BUILD_JOBS, SEGMENTATION_BUILD_JOBS_LOCK)
    with SEGMENTATION_BUILD_JOBS_LOCK:
        jobs = list(SEGMENTATION_BUILD_JOBS.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_serialize_seg_job(job) for job in jobs]


@app.get("/segmentation/build/jobs/{job_id}")
def get_segmentation_build_job(job_id: str):
    with SEGMENTATION_BUILD_JOBS_LOCK:
        job = SEGMENTATION_BUILD_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="segmentation_job_not_found")
    return _serialize_seg_job(job)


def _collect_labels_from_qwen_jsonl(jsonl_path: Path) -> List[str]:
    return _collect_labels_from_qwen_jsonl_impl(
        jsonl_path,
        extract_detections_fn=_extract_qwen_detections_from_payload,
    )


def _extract_qwen_detections_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _extract_qwen_detections_from_payload_impl(payload)


def _discover_yolo_labelmap(dataset_root: Path) -> List[str]:
    return _discover_yolo_labelmap_impl(dataset_root, load_labelmap_file_fn=_load_labelmap_file)


def _coco_info_block(dataset_id: str) -> Dict[str, Any]:
    """Minimal COCO info section to keep pycocotools happy."""
    return {
        "description": f"{dataset_id} generated by tator",
        "version": "1.0",
        "year": int(time.strftime("%Y", time.gmtime())),
        "contributor": "tator",
        "date_created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_coco_annotations(
    output_path: Path,
    *,
    dataset_id: str,
    categories: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
) -> None:
    payload = {
        "info": _coco_info_block(dataset_id),
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _ensure_coco_info_fields(path: Path, dataset_id: str, categories: List[Dict[str, Any]]) -> str:
    """Backfill missing COCO 'info'/'licenses' for older conversions."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load COCO file %s to backfill info: %s", path, exc)
        return str(path)
    if not isinstance(data, dict):
        return str(path)
    modified = False
    if "info" not in data or not isinstance(data["info"], dict):
        data["info"] = _coco_info_block(dataset_id)
        modified = True
    if "licenses" not in data or not isinstance(data["licenses"], list):
        data["licenses"] = []
        modified = True
    if categories and (not isinstance(data.get("categories"), list) or not data["categories"]):
        data["categories"] = categories
        modified = True
    if modified:
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to rewrite COCO file %s: %s", path, exc)
    return str(path)


def _ensure_coco_supercategory(path: Path, default: str = "object") -> bool:
    """Ensure every COCO category has a supercategory (RF-DETR expects it)."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    categories = data.get("categories")
    if not isinstance(categories, list):
        return False
    modified = False
    for category in categories:
        if not isinstance(category, dict):
            continue
        if "supercategory" not in category:
            category["supercategory"] = default
            modified = True
    if modified:
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle)
        except Exception:
            return False
    return modified


def _rfdetr_remap_coco_ids(src_path: Path, dest_path: Path) -> None:
    """Create a 0-based COCO category id mapping for RF-DETR."""
    data = json.loads(src_path.read_text())
    categories = data.get("categories", [])
    annotations = data.get("annotations", [])
    if not isinstance(categories, list) or not isinstance(annotations, list):
        raise RuntimeError("rfdetr_coco_invalid")
    ordered = [c for c in categories if isinstance(c, dict) and "id" in c and "name" in c]
    ordered.sort(key=lambda c: int(c.get("id", 0)))
    mapping = {int(cat["id"]): idx for idx, cat in enumerate(ordered)}
    new_categories = []
    for idx, cat in enumerate(ordered):
        new_categories.append(
            {
                "id": idx,
                "name": str(cat.get("name")),
                "supercategory": cat.get("supercategory") or "object",
            }
        )
    new_annotations = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        try:
            cat_id = int(ann.get("category_id"))
        except Exception:
            continue
        if cat_id not in mapping:
            continue
        ann = dict(ann)
        ann["category_id"] = mapping[cat_id]
        new_annotations.append(ann)
    data["categories"] = new_categories
    data["annotations"] = new_annotations
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(json.dumps(data))


def _rfdetr_prepare_dataset(dataset_root: Path, run_dir: Path, coco_train: str, coco_val: str) -> Path:
    """Prepare a RF-DETR-compatible dataset layout with 0-based category ids."""
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_src = dataset_root / "train"
    valid_src = dataset_root / "valid"
    val_src = dataset_root / "val"
    test_src = dataset_root / "test"

    if not train_src.exists():
        train_src = dataset_root
    if not valid_src.exists():
        valid_src = val_src if val_src.exists() else train_src
    if not test_src.exists():
        test_src = valid_src if valid_src.exists() else train_src

    def _link_split(name: str, source: Path) -> None:
        dest = dataset_dir / name
        if dest.exists() or dest.is_symlink():
            return
        try:
            dest.symlink_to(source, target_is_directory=True)
        except Exception:
            shutil.copytree(source, dest)

    _link_split("train", train_src)
    _link_split("valid", valid_src)
    _link_split("test", test_src)

    train_dest = dataset_dir / "train" / "_annotations.coco.json"
    val_dest = dataset_dir / "valid" / "_annotations.coco.json"
    test_dest = dataset_dir / "test" / "_annotations.coco.json"
    _rfdetr_remap_coco_ids(Path(coco_train), train_dest)
    _rfdetr_remap_coco_ids(Path(coco_val), val_dest)
    _rfdetr_remap_coco_ids(Path(coco_val), test_dest)
    return dataset_dir


def _validate_cuda_device_ids(device_ids: Sequence[int]) -> None:
    if not device_ids:
        return
    if not torch.cuda.is_available():
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="qwen_devices_unavailable")
    max_id = torch.cuda.device_count() - 1
    invalid = [device for device in device_ids if device < 0 or device > max_id]
    if invalid:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"qwen_invalid_devices:available=0-{max_id}",
        )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _rfdetr_ddp_worker(
    rank: int,
    world_size: int,
    variant_id: str,
    model_kwargs: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    aug_policy: Optional[Dict[str, Any]],
    dist_url: str,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    if dist_url.startswith("tcp://"):
        try:
            host_port = dist_url.replace("tcp://", "")
            host, port = host_port.split(":", 1)
            os.environ["MASTER_ADDR"] = host
            os.environ["MASTER_PORT"] = port
        except Exception:
            pass
    try:
        from rfdetr import (
            RFDETRBase,
            RFDETRLarge,
            RFDETRNano,
            RFDETRSmall,
            RFDETRMedium,
            RFDETRSegPreview,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"rfdetr_import_failed:{exc}") from exc
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
    model_kwargs = dict(model_kwargs)
    model_kwargs["device"] = "cuda" if torch.cuda.is_available() else model_kwargs.get("device", "cpu")
    train_kwargs = dict(train_kwargs)
    train_kwargs["device"] = "cuda" if torch.cuda.is_available() else train_kwargs.get("device", "cpu")
    train_kwargs["world_size"] = world_size
    train_kwargs["dist_url"] = dist_url
    rf_detr = model_cls(**model_kwargs)
    restore = _rfdetr_install_augmentations(_rfdetr_normalize_aug_policy(aug_policy))
    try:
        rf_detr.train(**train_kwargs)
    finally:
        _rfdetr_restore_augmentations(restore)


def _convert_yolo_dataset_to_coco(dataset_root: Path) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    val_images = dataset_root / "val" / "images"
    val_labels = dataset_root / "val" / "labels"
    for path in (train_images, train_labels):
        if not path.exists():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_yolo_split_missing")
    has_val_images = val_images.exists()
    has_val_labels = val_labels.exists()
    if has_val_images != has_val_labels:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_yolo_val_split_incomplete")
    if not has_val_images:
        # Allow train-only YOLO zips. We create an empty val split so downstream
        # code paths (COCO conversion, random split, etc.) have a consistent layout.
        val_images.mkdir(parents=True, exist_ok=True)
        val_labels.mkdir(parents=True, exist_ok=True)

    labelmap = _discover_yolo_labelmap(dataset_root)
    if not labelmap:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_labelmap_missing")
    label_to_id = {label: idx + 1 for idx, label in enumerate(labelmap)}
    categories = [{"id": cid, "name": name, "supercategory": "object"} for name, cid in label_to_id.items()]
    signature = _compute_dir_signature(dataset_root)
    existing_meta = _load_sam3_dataset_metadata(dataset_root)
    # Preserve previously recorded metadata and infer dataset type (bbox vs seg) so downstream
    # training can enable masks when appropriate.
    dataset_type = (existing_meta or {}).get("type", "bbox")
    dataset_label = (existing_meta or {}).get("label", dataset_root.name)
    dataset_source = (existing_meta or {}).get("source", "yolo")
    def _coco_has_invalid_image_refs(path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return True
        if not isinstance(data, dict):
            return True
        images = data.get("images")
        anns = data.get("annotations")
        if not isinstance(images, list) or not isinstance(anns, list):
            return True
        image_ids = set()
        for img in images:
            if not isinstance(img, dict):
                continue
            try:
                image_ids.add(int(img.get("id")))
            except Exception:
                continue
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            try:
                img_id = int(ann.get("image_id"))
            except Exception:
                continue
            if img_id not in image_ids:
                return True
        return False

    def _coco_missing_segmentation(path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return True
        if not isinstance(data, dict):
            return True
        anns = data.get("annotations")
        if not isinstance(anns, list):
            return True
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            seg = ann.get("segmentation")
            if seg is not None and seg != []:
                return False
        return True

    if (
        existing_meta
        and existing_meta.get("signature") == signature
        and existing_meta.get("coco_train_json")
        and existing_meta.get("coco_val_json")
    ):
        coco_train_path = Path(existing_meta["coco_train_json"])
        coco_val_path = Path(existing_meta["coco_val_json"])
        rebuild = _coco_has_invalid_image_refs(coco_train_path) or _coco_has_invalid_image_refs(coco_val_path)
        if dataset_type == "seg":
            rebuild = rebuild or _coco_missing_segmentation(coco_train_path) or _coco_missing_segmentation(coco_val_path)
        if not rebuild:
            # Backfill missing COCO info if this dataset was converted before we added it.
            _ensure_coco_info_fields(coco_train_path, dataset_root.name, categories)
            _ensure_coco_info_fields(coco_val_path, dataset_root.name, categories)
            return existing_meta

    # Rebuild/conversion path: infer bbox vs seg directly from labels.
    dataset_type = "bbox"

    image_id_counter = 1
    annotation_id = 1
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def _image_path_for_label(labels_dir: Path, images_dir: Path, label_file: Path) -> Optional[Path]:
        stem = label_file.stem
        try:
            rel_label = label_file.relative_to(labels_dir)
        except Exception:
            rel_label = Path(label_file.name)
        # Prefer mirrored subdirectory structure when present.
        for ext in image_exts:
            candidate = images_dir / rel_label.with_suffix(ext)
            if candidate.exists():
                return candidate
        for ext in image_exts:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        for candidate in images_dir.rglob(f"{stem}.*"):
            if candidate.suffix.lower() in image_exts:
                return candidate
        return None

    def _convert_split(split_images: Path, split_labels: Path, split_name: str) -> str:
        nonlocal image_id_counter, annotation_id, dataset_type
        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        images_lookup: Dict[str, int] = {}
        image_sizes: Dict[str, Tuple[int, int]] = {}

        def _clamp01(val: float) -> float:
            return max(0.0, min(1.0, val))

        def _bbox_xyxy_from_cxcywh(cx: float, cy: float, w: float, h: float) -> Optional[Tuple[float, float, float, float]]:
            if w <= 0 or h <= 0:
                return None
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            x1 = _clamp01(x1)
            y1 = _clamp01(y1)
            x2 = _clamp01(x2)
            y2 = _clamp01(y2)
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)

        def _bbox_xyxy_from_polygon(coords: List[float]) -> Optional[Tuple[float, float, float, float]]:
            if len(coords) < 6 or len(coords) % 2 != 0:
                return None
            xs = coords[0::2]
            ys = coords[1::2]
            if not xs or not ys:
                return None
            min_x = _clamp01(min(xs))
            max_x = _clamp01(max(xs))
            min_y = _clamp01(min(ys))
            max_y = _clamp01(max(ys))
            if max_x <= min_x or max_y <= min_y:
                return None
            return (min_x, min_y, max_x, max_y)

        def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter_area = inter_w * inter_h
            if inter_area <= 0:
                return 0.0
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            denom = area_a + area_b - inter_area
            return inter_area / denom if denom > 0 else 0.0

        def _polygon_to_coco_segmentation(coords: List[float], width: int, height: int) -> Optional[List[List[float]]]:
            if len(coords) < 6 or len(coords) % 2 != 0:
                return None
            out: List[float] = []
            for idx in range(0, len(coords), 2):
                x = _clamp01(coords[idx]) * width
                y = _clamp01(coords[idx + 1]) * height
                out.extend([x, y])
            if len(out) < 6:
                return None
            return [out]

        for label_file in sorted(split_labels.rglob("*.txt")):
            image_path = _image_path_for_label(split_labels, split_images, label_file)
            if image_path is None:
                logger.warning("No matching image for label file %s", label_file)
                continue
            image_rel = str(image_path.relative_to(split_images.parent))
            if image_rel not in images_lookup:
                try:
                    with Image.open(image_path) as im:
                        width, height = im.size
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read image %s: %s", image_path, exc)
                    continue
                images_lookup[image_rel] = image_id_counter
                image_sizes[image_rel] = (width, height)
                images.append(
                    {
                        "id": image_id_counter,
                        "file_name": image_rel,
                        "width": width,
                        "height": height,
                    }
                )
                image_id_counter += 1
            image_id = images_lookup[image_rel]
            width, height = image_sizes.get(image_rel, (None, None))
            try:
                with label_file.open("r", encoding="utf-8") as handle:
                    lines = [ln.strip() for ln in handle if ln.strip()]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read YOLO labels from %s: %s", label_file, exc)
                continue
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_idx = int(float(parts[0]))
                except (TypeError, ValueError):
                    continue
                if class_idx < 0 or class_idx >= len(labelmap):
                    continue
                if width is None or height is None:
                    continue
                raw_vals = []
                for token in parts[1:]:
                    try:
                        raw_vals.append(float(token))
                    except (TypeError, ValueError):
                        raw_vals = []
                        break
                if not raw_vals:
                    continue
                bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
                segmentation: Optional[List[List[float]]] = None
                if len(raw_vals) == 4:
                    cx, cy, w, h = raw_vals
                    bbox_xyxy = _bbox_xyxy_from_cxcywh(cx, cy, w, h)
                else:
                    # Support YOLO-seg polygon-only format (YOLOv8) and bbox+polygon format.
                    poly_only = raw_vals if len(raw_vals) >= 6 and len(raw_vals) % 2 == 0 else None
                    bbox_plus_poly = None
                    if len(raw_vals) > 4 and (len(raw_vals) - 4) >= 6 and (len(raw_vals) - 4) % 2 == 0:
                        bbox_plus_poly = (raw_vals[:4], raw_vals[4:])
                    chosen_poly = None
                    if poly_only is not None and bbox_plus_poly is not None:
                        bbox_fields, poly_fields = bbox_plus_poly
                        bbox_from_fields = _bbox_xyxy_from_cxcywh(*bbox_fields)
                        bbox_from_poly = _bbox_xyxy_from_polygon(poly_fields)
                        if bbox_from_fields is not None and bbox_from_poly is not None and _bbox_iou(bbox_from_fields, bbox_from_poly) >= 0.9:
                            chosen_poly = poly_fields
                            bbox_xyxy = bbox_from_poly
                        else:
                            chosen_poly = poly_only
                            bbox_xyxy = _bbox_xyxy_from_polygon(poly_only)
                    elif bbox_plus_poly is not None:
                        _, poly_fields = bbox_plus_poly
                        chosen_poly = poly_fields
                        bbox_xyxy = _bbox_xyxy_from_polygon(poly_fields)
                    elif poly_only is not None:
                        chosen_poly = poly_only
                        bbox_xyxy = _bbox_xyxy_from_polygon(poly_only)
                    else:
                        # Unknown extra fields; treat first four as bbox and ignore remainder.
                        bbox_xyxy = _bbox_xyxy_from_cxcywh(*raw_vals[:4])
                    if chosen_poly is not None and bbox_xyxy is not None:
                        segmentation = _polygon_to_coco_segmentation(chosen_poly, int(width), int(height))
                        dataset_type = "seg"

                if bbox_xyxy is None:
                    continue
                x1_n, y1_n, x2_n, y2_n = bbox_xyxy
                x1 = x1_n * width
                y1 = y1_n * height
                abs_w = (x2_n - x1_n) * width
                abs_h = (y2_n - y1_n) * height
                if abs_w <= 0 or abs_h <= 0:
                    continue
                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_idx + 1,
                    "bbox": [x1, y1, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0,
                }
                if segmentation is not None:
                    ann["segmentation"] = segmentation
                annotations.append(ann)
                annotation_id += 1
        output_path = dataset_root / split_name / "_annotations.coco.json"
        try:
            _write_coco_annotations(
                output_path,
                dataset_id=dataset_root.name,
                categories=categories,
                images=images,
                annotations=annotations,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_coco_write_failed:{exc}") from exc
        return str(output_path)

    coco_train = _convert_split(train_images, train_labels, "train")
    coco_val = _convert_split(val_images, val_labels, "val")
    sam3_meta = {
        "id": dataset_root.name,
        "label": dataset_label,
        "source": dataset_source,
        "type": dataset_type,
        "dataset_root": str(dataset_root),
        "signature": signature,
        "classes": labelmap,
        "context": "",
        "image_count": None,
        "train_count": None,
        "val_count": None,
        "coco_train_json": coco_train,
        "coco_val_json": coco_val,
        "converted_at": time.time(),
    }
    _persist_sam3_dataset_metadata(dataset_root, sam3_meta)
    return sam3_meta


def _convert_qwen_dataset_to_coco(dataset_root: Path) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    metadata = _load_qwen_dataset_metadata(dataset_root) or {}
    metadata, signature = _ensure_qwen_dataset_signature(dataset_root, metadata)
    if "type" not in metadata:
        metadata["type"] = "bbox"
        _persist_qwen_dataset_metadata(dataset_root, metadata)
    dataset_id = metadata.get("id") or dataset_root.name
    labelmap = _load_qwen_labelmap(
        dataset_root,
        load_qwen_meta=_load_qwen_dataset_metadata,
        collect_labels=_collect_labels_from_qwen_jsonl,
    )
    if not labelmap:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_labelmap_missing")
    label_to_id = {label: idx + 1 for idx, label in enumerate(labelmap)}
    categories = [{"id": cid, "name": name, "supercategory": "object"} for name, cid in label_to_id.items()]
    existing_meta = _load_sam3_dataset_metadata(dataset_root)
    if (
        existing_meta
        and existing_meta.get("signature") == signature
        and existing_meta.get("coco_train_json")
        and existing_meta.get("coco_val_json")
    ):
        _ensure_coco_info_fields(Path(existing_meta["coco_train_json"]), dataset_id, categories)
        _ensure_coco_info_fields(Path(existing_meta["coco_val_json"]), dataset_id, categories)
        return existing_meta

    annotation_id = 1
    images_lookup: Dict[str, int] = {}
    image_sizes: Dict[str, Tuple[int, int]] = {}

    def _convert_split(split: str) -> str:
        nonlocal annotation_id
        jsonl_path = dataset_root / split / "annotations.jsonl"
        if not jsonl_path.exists():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"sam3_annotations_missing:{split}")
        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    image_rel = payload.get("image")
                    if not isinstance(image_rel, str):
                        continue
                    if image_rel not in images_lookup:
                        image_path = dataset_root / split / image_rel
                        if not image_path.exists():
                            logger.warning("Missing image referenced in %s: %s", jsonl_path, image_path)
                            continue
                        try:
                            with Image.open(image_path) as im:
                                width, height = im.size
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to read image %s: %s", image_path, exc)
                            continue
                        images_lookup[image_rel] = len(images_lookup) + 1
                        image_sizes[image_rel] = (width, height)
                        images.append(
                            {
                                "id": images_lookup[image_rel],
                                "file_name": image_rel,
                                "width": width,
                                "height": height,
                            }
                        )
                    image_id = images_lookup[image_rel]
                    width, height = image_sizes.get(image_rel, (None, None))
                    detections = _extract_qwen_detections_from_payload(payload)
                    for det in detections:
                        label = str(det.get("label", "")).strip()
                        if not label or label not in label_to_id:
                            continue
                        bbox = det.get("bbox")
                        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            try:
                                x1 = float(bbox[0])
                                y1 = float(bbox[1])
                                x2 = float(bbox[2])
                                y2 = float(bbox[3])
                            except (TypeError, ValueError):
                                continue
                            if width is not None and height is not None:
                                x1 = max(0.0, min(x1, width))
                                x2 = max(0.0, min(x2, width))
                                y1 = max(0.0, min(y1, height))
                                y2 = max(0.0, min(y2, height))
                            w = max(0.0, x2 - x1)
                            h = max(0.0, y2 - y1)
                            if w <= 0 or h <= 0:
                                continue
                            coco_bbox = [x1, y1, w, h]
                        else:
                            point = det.get("point")
                            if not (isinstance(point, (list, tuple)) and len(point) >= 2):
                                continue
                            try:
                                cx = float(point[0])
                                cy = float(point[1])
                            except (TypeError, ValueError):
                                continue
                            # Convert point to a tiny box to retain the signal.
                            size = 2.0
                            x1 = cx - size / 2.0
                            y1 = cy - size / 2.0
                            coco_bbox = [x1, y1, size, size]
                        area = coco_bbox[2] * coco_bbox[3]
                        if area <= 0:
                            continue
                        annotations.append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": label_to_id[label],
                                "bbox": coco_bbox,
                                "area": area,
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to convert %s to COCO: %s", jsonl_path, exc)
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_coco_conversion_failed:{split}")
        output_path = dataset_root / split / "_annotations.coco.json"
        try:
            _write_coco_annotations(
                output_path,
                dataset_id=dataset_id,
                categories=categories,
                images=images,
                annotations=annotations,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_coco_write_failed:{exc}") from exc
        return str(output_path)

    coco_train = _convert_split("train")
    coco_val = _convert_split("val")
    sam3_meta = {
        "id": metadata.get("id") or dataset_root.name,
        "label": metadata.get("label") or metadata.get("id") or dataset_root.name,
        "source": "qwen",
        "type": metadata.get("type", "bbox"),
        "dataset_root": str(dataset_root),
        "signature": signature,
        "classes": labelmap,
        "context": metadata.get("context", ""),
        "image_count": metadata.get("image_count"),
        "train_count": metadata.get("train_count"),
        "val_count": metadata.get("val_count"),
        "coco_train_json": coco_train,
        "coco_val_json": coco_val,
        "converted_at": time.time(),
    }
    _persist_sam3_dataset_metadata(dataset_root, sam3_meta)
    return sam3_meta


def _convert_coco_dataset_to_yolo(dataset_root: Path) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    ann_paths: List[Tuple[str, Path, Path]] = []
    for split in ("train", "val"):
        ann_path = dataset_root / split / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        images_dir = ann_path.parent / "images"
        if not images_dir.exists():
            images_dir = ann_path.parent
        ann_paths.append((split, ann_path, images_dir))
    if not ann_paths:
        ann_path, images_dir = _find_coco_split(dataset_root)
        ann_paths = [("train", ann_path, images_dir)]

    category_map: Dict[int, str] = {}
    has_segmentation = False
    for _, ann_path, _ in ann_paths:
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"coco_load_failed:{exc}") from exc
        for cat in data.get("categories", []) or []:
            try:
                cid = int(cat.get("id"))
            except Exception:
                continue
            name = str(cat.get("name") or f"class_{cid}")
            category_map.setdefault(cid, name)
        if not category_map:
            for ann in data.get("annotations", []) or []:
                try:
                    cid = int(ann.get("category_id"))
                except Exception:
                    continue
                category_map.setdefault(cid, f"class_{cid}")
        if not has_segmentation:
            for ann in data.get("annotations", []) or []:
                seg = ann.get("segmentation")
                if isinstance(seg, list) and any(isinstance(poly, list) and len(poly) >= 6 for poly in seg):
                    has_segmentation = True
                    break

    if not category_map:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="coco_categories_missing")

    sorted_ids = sorted(category_map.keys())
    labelmap = [category_map[cid] for cid in sorted_ids]
    labelmap_path = dataset_root / "labelmap.txt"
    labelmap_path.write_text("\n".join(labelmap) + "\n", encoding="utf-8")
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(sorted_ids)}

    def _resolve_image_path(file_name: str, images_dir: Path, split_name: str) -> Optional[Path]:
        if not file_name:
            return None
        rel_path = Path(file_name)
        candidates: List[Path] = []
        if rel_path.is_absolute():
            candidates.append(rel_path)
        candidates.append(images_dir / rel_path)
        candidates.append(images_dir / rel_path.name)
        candidates.append(dataset_root / rel_path)
        candidates.append(dataset_root / split_name / "images" / rel_path.name)
        for cand in candidates:
            if cand.exists():
                return cand
        return None

    def _label_relpath_for_image(file_name: str) -> Path:
        rel_path = Path(file_name)
        if rel_path.is_absolute():
            rel_path = Path(rel_path.name)
        if "images" in rel_path.parts:
            idx = rel_path.parts.index("images")
            rel_path = Path(*rel_path.parts[idx + 1 :])
        return rel_path.with_suffix(".txt")

    dataset_type = "seg" if has_segmentation else "bbox"
    for split_name, ann_path, images_dir in ann_paths:
        labels_dir = dataset_root / split_name / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"coco_load_failed:{exc}") from exc
        images = data.get("images", []) or []
        annotations = data.get("annotations", []) or []
        ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in annotations:
            try:
                img_id = int(ann.get("image_id"))
            except Exception:
                continue
            ann_by_image.setdefault(img_id, []).append(ann)
        for img in images:
            try:
                img_id = int(img.get("id"))
            except Exception:
                continue
            file_name = str(img.get("file_name") or "")
            img_path = _resolve_image_path(file_name, images_dir, split_name)
            if img_path is None:
                logger.warning("COCO->YOLO: missing image for %s in %s", file_name, dataset_root)
                continue
            width = img.get("width")
            height = img.get("height")
            if not width or not height:
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                except Exception as exc:  # noqa: BLE001
                    logger.warning("COCO->YOLO: failed to read image size for %s: %s", img_path, exc)
                    continue
            label_rel = _label_relpath_for_image(file_name)
            label_path = labels_dir / label_rel
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines: List[str] = []
            for ann in ann_by_image.get(img_id, []):
                try:
                    cat_id = int(ann.get("category_id"))
                except Exception:
                    continue
                if cat_id not in cat_id_to_idx:
                    continue
                class_idx = cat_id_to_idx[cat_id]
                bbox = ann.get("bbox") or []
                if len(bbox) < 4:
                    continue
                x, y, w, h = map(float, bbox[:4])
                if w <= 0 or h <= 0:
                    continue
                cx = (x + w / 2.0) / float(width)
                cy = (y + h / 2.0) / float(height)
                bw = w / float(width)
                bh = h / float(height)
                if dataset_type == "seg":
                    seg = ann.get("segmentation")
                    poly = None
                    if isinstance(seg, list):
                        for candidate in seg:
                            if isinstance(candidate, list) and len(candidate) >= 6:
                                poly = candidate
                                break
                    if poly is not None:
                        coords: List[str] = []
                        for idx in range(0, len(poly), 2):
                            px = float(poly[idx]) / float(width)
                            py = float(poly[idx + 1]) / float(height)
                            coords.append(f"{max(0.0, min(1.0, px)):.6f}")
                            coords.append(f"{max(0.0, min(1.0, py)):.6f}")
                        if len(coords) >= 6:
                            lines.append(f"{class_idx} " + " ".join(coords))
                            continue
                lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if lines:
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = _load_sam3_dataset_metadata(dataset_root) or {}
    meta.setdefault("id", dataset_root.name)
    meta.setdefault("label", dataset_root.name)
    meta.setdefault("source", meta.get("source") or "coco")
    meta["classes"] = labelmap
    meta["type"] = dataset_type
    meta["dataset_root"] = str(dataset_root)
    meta["signature"] = _compute_dir_signature(dataset_root)
    meta["yolo_converted_at"] = time.time()
    _persist_sam3_dataset_metadata(dataset_root, meta)
    return meta


def _list_sam3_datasets() -> List[Dict[str, Any]]:
    return _list_all_datasets()


def _resolve_sam3_dataset_meta(dataset_id: str) -> Dict[str, Any]:
    dataset_root = _resolve_sam3_or_qwen_dataset(dataset_id)
    annotations_path = dataset_root / "train" / "annotations.jsonl"
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    if annotations_path.exists():
        meta = _convert_qwen_dataset_to_coco(dataset_root)
    elif train_images.exists() and train_labels.exists():
        meta = _convert_yolo_dataset_to_coco(dataset_root)
    else:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_dataset_type_unsupported")
    meta["dataset_root"] = str(dataset_root)
    return meta


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
        _write_coco_annotations(
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
        "signature": _compute_dir_signature(split_root),
        "source": meta.get("source", "resplit"),
    }
    _persist_sam3_dataset_metadata(split_root, new_meta)
    summary = (
        f"SAM3 split: {train_count} train / {val_count} val "
        f"(seed={split_seed}, val_percent={vp:.2f}, src={dataset_root}) -> {split_root}"
    )
    logger.info(summary)
    if log_messages is not None:
        log_messages.append(summary)
    return new_meta


def _plan_segmentation_build(request: SegmentationBuildRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset_root = _resolve_sam3_or_qwen_dataset(request.source_dataset_id)
    source_meta = _load_qwen_dataset_metadata(dataset_root) or _load_sam3_dataset_metadata(dataset_root)
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
    source_signature = source_meta.get("signature") or _compute_dir_signature(dataset_root)
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
                    labelmap_file = _resolve_sam3_or_qwen_dataset(request.source_dataset_id) / "labelmap.txt"
                    classes = _load_labelmap_file(labelmap_file)
                except Exception:
                    classes = []
            if not classes:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="segmentation_builder_no_classes")
            dataset_root = Path(source_meta.get("dataset_root") or _resolve_sam3_or_qwen_dataset(request.source_dataset_id))
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
            base_devices = _resolve_sam3_mining_devices() if request.sam_variant == "sam3" else _resolve_sam1_devices()
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
                                        mask_arr = decode_binary_mask(best.get("mask"))
                                polygon = mask_to_polygon(mask_arr, simplify_eps) if mask_arr is not None else []
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
                coco_meta = _convert_yolo_dataset_to_coco(output_root)
            except Exception as exc:  # noqa: BLE001
                _seg_job_update(job, status="failed", message="COCO conversion failed", error=str(exc))
                return
            result_meta = _load_sam3_dataset_metadata(output_root) or coco_meta or planned_meta
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
            _prepare_for_training()
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
            _finalize_training_environment()

    thread = threading.Thread(target=worker, name=f"sam3-train-{job.job_id}", daemon=True)
    thread.start()


def _start_yolo_training_worker(job: YoloTrainingJob) -> None:
    def worker() -> None:
        run_dir = _yolo_run_dir(job.job_id, create=True)
        config = dict(job.config or {})
        dataset_info = config.get("dataset") or {}
        task = str(dataset_info.get("task") or config.get("task") or "detect").lower()
        if job.cancel_event.is_set():
            _yolo_job_update(job, status="cancelled", message="Cancelled before start", progress=0.0)
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        if not dataset_info.get("yolo_ready"):
            _yolo_job_update(job, status="failed", message="Dataset is not YOLO-ready", error="yolo_not_ready")
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        try:
            _prepare_for_training()
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # noqa: BLE001
            _yolo_job_update(job, status="failed", message="Ultralytics not installed", error=str(exc))
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        _yolo_job_update(job, status="running", message="Starting YOLOv8 training", progress=0.0)
        _yolo_job_log(job, "Preparing dataset + data.yaml")
        dataset_root = Path(dataset_info.get("prepared_root") or dataset_info.get("dataset_root") or "")
        data_yaml = _yolo_write_data_yaml(run_dir, dataset_root, dataset_info.get("yolo_layout"), dataset_info.get("yolo_labelmap_path"))
        from_scratch = bool(config.get("from_scratch"))
        base_weights = config.get("base_weights")
        variant = config.get("variant") or ""
        if task == "segment" and _yolo_p2_scale(variant):
            _yolo_job_update(job, status="failed", message="P2 head is only supported for detection.", error="yolo_p2_segment")
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        _, model_source = _yolo_resolve_model_source(variant, task, from_scratch, base_weights)
        device_arg = _yolo_device_arg(config.get("devices"))
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
        p2_scale = _yolo_p2_scale(variant)
        if p2_scale and model_source.endswith("yolov8-p2.yaml"):
            try:
                import ultralytics  # type: ignore
            except Exception as exc:  # noqa: BLE001
                _yolo_job_update(job, status="failed", message="Ultralytics not installed", error=str(exc))
                _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
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
        train_kwargs.update(_yolo_build_aug_args(config.get("augmentations")))
        train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
        monitor_stop = threading.Event()
        monitor_thread = None
        try:
            model = YOLO(model_source)
            _yolo_job_log(job, "Training started")
            monitor_thread = threading.Thread(
                target=_yolo_monitor_training,
                args=(job, run_dir, int(config.get("epochs") or 0), monitor_stop),
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
                metrics_series = _yolo_parse_results_csv(run_dir / "results.csv")
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
            _yolo_prune_run_dir(run_dir)
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
            _finalize_training_environment()
            _yolo_write_run_meta(
                run_dir,
                {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config, "result": job.result},
            )

    thread = threading.Thread(target=worker, name=f"yolo-train-{job.job_id}", daemon=True)
    thread.start()


def _start_yolo_head_graft_worker(job: YoloHeadGraftJob) -> None:
    def worker() -> None:
        job.thread_ident = threading.get_ident()
        run_dir = _yolo_run_dir(job.job_id, create=True)
        config = dict(job.config or {})
        if run_dir:
            config.setdefault("paths", {})["run_dir"] = str(run_dir)
            job.config = config
        base_run_id = str(config.get("base_run_id") or "").strip()
        if not base_run_id:
            _yolo_head_graft_job_update(job, status="failed", message="Base YOLO run missing", error="yolo_base_missing")
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        if job.cancel_event.is_set():
            _yolo_head_graft_job_update(job, status="cancelled", message="Cancelled before start", progress=0.0)
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        base_run_dir = _yolo_run_dir(base_run_id, create=False)
        base_best = base_run_dir / "best.pt"
        if not base_best.exists():
            _yolo_head_graft_job_update(job, status="failed", message="Base run is missing best.pt", error="yolo_base_missing_best")
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        base_meta = _yolo_load_run_meta(base_run_dir)
        base_cfg = base_meta.get("config") or {}
        base_task = str(base_cfg.get("task") or "detect").lower()
        if base_task != "detect":
            _yolo_head_graft_job_update(job, status="failed", message="Base run is not detect", error="yolo_base_not_detect")
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        base_variant = config.get("variant") or base_cfg.get("variant")
        if not base_variant:
            _yolo_head_graft_job_update(
                job,
                status="failed",
                message="Base run missing variant (cannot infer architecture)",
                error="yolo_base_variant_missing",
            )
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        base_labelmap = _yolo_load_run_labelmap(base_run_dir)
        if not base_labelmap:
            _yolo_head_graft_job_update(job, status="failed", message="Base labelmap missing", error="yolo_base_labelmap_missing")
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        try:
            dataset_payload = YoloTrainRequest(dataset_id=config.get("dataset_id"), dataset_root=config.get("dataset_root"))
            dataset_info = _resolve_yolo_training_dataset(dataset_payload)
        except Exception as exc:  # noqa: BLE001
            _yolo_head_graft_job_update(job, status="failed", message="Dataset not ready", error=str(exc))
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        if not dataset_info.get("yolo_ready"):
            _yolo_head_graft_job_update(job, status="failed", message="Dataset is not YOLO-ready", error="yolo_not_ready")
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        dataset_task = str(dataset_info.get("task") or "detect").lower()
        if dataset_task != "detect":
            _yolo_head_graft_job_update(job, status="failed", message="Head grafting only supports detect datasets", error="yolo_graft_detect_only")
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        new_labelmap = _yolo_load_labelmap(Path(dataset_info.get("yolo_labelmap_path") or ""))
        if not new_labelmap:
            _yolo_head_graft_job_update(job, status="failed", message="New labelmap missing", error="yolo_new_labelmap_missing")
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
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
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        try:
            _prepare_for_training()
            _patch_ultralytics_for_head_grafting()
            import ultralytics  # type: ignore
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # noqa: BLE001
            _yolo_head_graft_job_update(job, status="failed", message="Ultralytics not installed", error=str(exc))
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        version = getattr(ultralytics, "__version__", "")
        if version and not version.startswith("8."):
            _yolo_head_graft_job_update(
                job,
                status="failed",
                message=f"Ultralytics {version} unsupported for head grafting",
                error="yolo_graft_ultralytics_version",
            )
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
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
        data_yaml = _yolo_write_data_yaml(run_dir, Path(dataset_info.get("prepared_root") or dataset_info.get("dataset_root") or ""), dataset_info.get("yolo_layout"), dataset_info.get("yolo_labelmap_path"))
        nc_new = len(new_labelmap)
        nc_base = len(base_labelmap)
        head_yaml = _yolo_write_variant_yaml(run_dir, base_variant, "detect", nc_new)
        device_arg = _yolo_device_arg(config.get("devices"))
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
            detect_idx = _yolo_detect_layer_index(model.model)

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
                _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
                _finalize_training_environment()
                return
        except Exception as exc:  # noqa: BLE001
            _yolo_head_graft_job_update(job, status="failed", message="Head training failed", error=str(exc))
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            _finalize_training_environment()
            return
        _yolo_head_graft_job_update(job, message="Merging heads", progress=0.7)
        merged_yaml = _yolo_write_head_graft_yaml(run_dir, base_variant, nc_base, nc_new)
        try:
            merged = YOLO(str(merged_yaml)).load(str(base_best))
            new_model = YOLO(str(new_best))
            new_detects = _yolo_find_detect_modules(new_model.model)
            merged_detects = _yolo_find_detect_modules(merged.model)
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
            _yolo_prune_run_dir(run_dir)
            _yolo_head_graft_job_update(job, status="succeeded", message="Head graft complete", progress=1.0, result=result_payload)
            _yolo_write_run_meta(
                run_dir,
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
                },
            )
            _yolo_head_graft_audit(job, "head_graft_complete", event="complete", extra={"result": result_payload})
        except Exception as exc:  # noqa: BLE001
            _yolo_head_graft_job_update(job, status="failed", message="Head merge failed", error=str(exc))
            _yolo_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
        finally:
            _finalize_training_environment()

    thread = threading.Thread(target=worker, name=f"yolo-graft-{job.job_id}", daemon=True)
    thread.start()


def _start_rfdetr_training_worker(job: RfDetrTrainingJob) -> None:
    def worker() -> None:
        run_dir = _rfdetr_run_dir(job.job_id, create=True)
        config = dict(job.config or {})
        dataset_info = config.get("dataset") or {}
        if job.cancel_event.is_set():
            _rfdetr_job_update(job, status="cancelled", message="Cancelled before start", progress=0.0)
            _rfdetr_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        try:
            _prepare_for_training()
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
            _rfdetr_write_run_meta(run_dir, {"job_id": job.job_id, "status": job.status, "message": job.message, "config": job.config})
            return
        try:
            task = str(dataset_info.get("task") or config.get("task") or "detect").lower()
            variant_info = _rfdetr_variant_info(task, config.get("variant"))
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
            labelmap = _rfdetr_load_labelmap(dataset_root, coco_train)
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
            prepared_root = _rfdetr_prepare_dataset(dataset_root, run_dir, coco_train, coco_val)
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
            aug_policy = _rfdetr_normalize_aug_policy(config.get("augmentations"))
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
            cuda_visible = ",".join(str(d) for d in device_ids) if device_ids else None
            prev_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
            train_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            use_distributed = torch.cuda.is_available() and len(device_ids) > 1
            _rfdetr_job_log(job, f"Model variant: {variant_id}")
            if use_distributed:
                dist_url = f"tcp://127.0.0.1:{_find_free_port()}"
                world_size = len(device_ids)
                _rfdetr_job_log(job, f"Multi-GPU enabled: devices={cuda_visible} world_size={world_size}")
                _rfdetr_job_log(job, f"Training started (epochs={total_epochs})")
                monitor_stop = threading.Event()
                monitor_thread = threading.Thread(
                    target=_rfdetr_monitor_training,
                    args=(job, run_dir, total_epochs, monitor_stop),
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
                restore = _rfdetr_install_augmentations(aug_policy)

                def on_fit_epoch_end(stats: Dict[str, Any]) -> None:
                    metric = _rfdetr_sanitize_metric(stats)
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
                    _rfdetr_restore_augmentations(restore)
            if job.cancel_event.is_set():
                _rfdetr_job_update(job, status="cancelled", message="Training cancelled", progress=job.progress)
            else:
                _rfdetr_job_update(job, status="succeeded", message="Training complete", progress=1.0)
            metrics_series = job.metrics or []
            if not metrics_series:
                metrics_series = _rfdetr_parse_log_series(run_dir / "log.txt")
                if metrics_series:
                    job.metrics = metrics_series
            if metrics_series:
                try:
                    (run_dir / "metrics_series.json").write_text(json.dumps(metrics_series, indent=2, sort_keys=True))
                except Exception:
                    pass
            best_path = _rfdetr_best_checkpoint(run_dir)
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
            _rfdetr_prune_run_dir(run_dir)
            job.result = result_payload
        except Exception as exc:  # noqa: BLE001
            _rfdetr_job_update(job, status="failed", message="Training failed", error=str(exc))
        finally:
            if "prev_cuda_visible" in locals():
                if prev_cuda_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible
            _finalize_training_environment()
            _rfdetr_write_run_meta(
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
    existing = _dir_size_bytes(quota_root) if quota_root and quota_limit else 0
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


def _resolve_head_normalize_embeddings(head: Optional[Dict[str, Any]], *, default: bool = True) -> bool:
    return _resolve_head_normalize_embeddings_impl(head, default=default)


def _resolve_active_head_normalize_embeddings(
    meta_obj: Optional[Dict[str, Any]],
    clf_obj: Optional[object],
    *,
    default: bool = True,
) -> bool:
    return _resolve_active_head_normalize_embeddings_impl(
        meta_obj,
        clf_obj,
        default=default,
        resolve_head_normalize_embeddings_fn=_resolve_head_normalize_embeddings,
    )

def mask_to_bounding_box(mask: np.ndarray) -> tuple[int,int,int,int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return (0,0,0,0)
    y_min,y_max = np.where(rows)[0][[0,-1]]
    x_min,x_max = np.where(cols)[0][[0,-1]]
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def encode_binary_mask(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    try:
        mask_arr = np.asarray(mask)
    except Exception:
        return None
    if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
        mask_arr = mask_arr[0]
    if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
        mask_arr = mask_arr[..., 0]
    if mask_arr.ndim != 2:
        return None
    mask_bool = mask_arr.astype(bool)
    height, width = mask_bool.shape
    packed = np.packbits(mask_bool.astype(np.uint8), axis=None)
    try:
        packed_bytes = packed.tobytes()
    except Exception:
        return None
    if MASK_ENCODE_MAX_BYTES > 0 and len(packed_bytes) > MASK_ENCODE_MAX_BYTES:
        return None
    try:
        encoded = base64.b64encode(packed_bytes).decode("ascii")
    except Exception:
        return None
    return {"size": [int(height), int(width)], "counts": encoded}


def _prune_detections_for_response(dets: List[Any], warnings: Optional[List[str]] = None) -> List[Any]:
    """Clamp response payload size by limiting detection count and mask payloads."""
    if not dets:
        return dets
    limited: List[Any] = list(dets[: MAX_RESPONSE_DETECTIONS]) if MAX_RESPONSE_DETECTIONS > 0 else list(dets)
    if warnings is not None and MAX_RESPONSE_DETECTIONS > 0 and len(dets) > MAX_RESPONSE_DETECTIONS:
        warnings.append("detections_pruned")
    mask_budget = MAX_RESPONSE_MASKS if MAX_RESPONSE_MASKS > 0 else None
    masks_used = 0
    for det in limited:
        mask_val = getattr(det, "mask", None)
        if mask_val is not None and mask_budget is not None:
            if masks_used >= mask_budget:
                try:
                    det.mask = None  # type: ignore[attr-defined]
                except Exception:
                    pass
                else:
                    if warnings is not None:
                        warnings.append("masks_pruned")
            else:
                masks_used += 1
    return limited


def decode_binary_mask(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    if not payload:
        return None
    counts = payload.get("counts")
    size = payload.get("size") or []
    if not counts or len(size) != 2:
        return None
    try:
        packed = np.frombuffer(base64.b64decode(counts), dtype=np.uint8)
        bits = np.unpackbits(packed)[: int(size[0]) * int(size[1])]
        return bits.reshape(int(size[0]), int(size[1]))
    except Exception:
        return None


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker simplification for 2D points."""
    if points.shape[0] < 3 or epsilon <= 0:
        return points

    def _perp_dist(pt, start, end):
        if np.allclose(start, end):
            return np.linalg.norm(pt - start)
        return np.abs(np.cross(end - start, start - pt)) / np.linalg.norm(end - start)

    start_pt = points[0]
    end_pt = points[-1]
    dmax = 0.0
    idx = 0
    for i in range(1, len(points) - 1):
        d = _perp_dist(points[i], start_pt, end_pt)
        if d > dmax:
            idx = i
            dmax = d
    if dmax > epsilon:
        rec1 = _rdp(points[: idx + 1], epsilon)
        rec2 = _rdp(points[idx:], epsilon)
        return np.concatenate((rec1[:-1], rec2), axis=0)
    return np.array([start_pt, end_pt])


def mask_to_polygon(mask: np.ndarray, simplify_epsilon: float) -> List[Tuple[float, float]]:
    """Extract a coarse polygon outline from a binary mask."""
    try:
        mask_arr = np.asarray(mask).astype(bool)
    except Exception:
        return []
    if mask_arr.ndim != 2 or not mask_arr.any():
        return []
    coords = np.argwhere(mask_arr)  # y, x
    if coords.shape[0] < 3:
        return []
    points = np.stack([coords[:, 1], coords[:, 0]], axis=1)  # x, y
    hull_pts = points
    if ConvexHull is not None:
        try:
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]
        except Exception:
            hull_pts = points
    if simplify_epsilon and simplify_epsilon > 0:
        hull_pts = _rdp(hull_pts, simplify_epsilon)
    # Ensure at least 3 points.
    if hull_pts.shape[0] < 3:
        # Fallback to simple bounding box.
        xs, ys = points[:, 0], points[:, 1]
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        hull_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return [(float(x), float(y)) for x, y in hull_pts]

def to_yolo(w: int, h: int, left: int, top: int, right: int, bottom: int) -> List[float]:
    w_abs = float(right - left)
    h_abs = float(bottom - top)
    cx_abs = left + w_abs/2
    cy_abs = top + h_abs/2
    cx = cx_abs / w
    cy = cy_abs / h
    ww = w_abs / w
    hh = h_abs / h
    return [cx, cy, ww, hh]


def yolo_to_corners(box: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    if len(box) < 4:
        return (0, 0, 0, 0)
    cx, cy, ww, hh = box[:4]
    w_abs = max(0.0, float(ww) * w)
    h_abs = max(0.0, float(hh) * h)
    cx_abs = float(cx) * w
    cy_abs = float(cy) * h
    left = int(round(cx_abs - w_abs / 2))
    top = int(round(cy_abs - h_abs / 2))
    right = int(round(cx_abs + w_abs / 2))
    bottom = int(round(cy_abs + h_abs / 2))
    left = max(0, min(w, left))
    top = max(0, min(h, top))
    right = max(left, min(w, right))
    bottom = max(top, min(h, bottom))
    return left, top, right, bottom

@app.post("/predict_base64", response_model=PredictResponse)
def predict_base64(payload: Base64Payload):
    # If CLIP/logreg not loaded, return error message in "prediction"
    if not _active_encoder_ready():
        return PredictResponse(prediction=str(ERROR_MESSAGE), uuid=None, error="clip_unavailable") # messy ... returning the error message int as str. Crap logic needs cleanup

    pil_img, _np_img, _token = _resolve_detector_image(payload.image_base64, payload.image_token)
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


@app.get("/clip/backbones")
def list_clip_backbones():
    return {
        "available": SUPPORTED_CLIP_MODELS,
        "active": clip_model_name,
    }


def _list_clip_classifiers() -> List[Dict[str, Any]]:
    return _list_clip_classifiers_impl(
        upload_root=UPLOAD_ROOT,
        classifier_exts=CLASSIFIER_ALLOWED_EXTS,
        labelmap_exts=LABELMAP_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root,
        joblib_load_fn=joblib.load,
        resolve_clip_labelmap_path_fn=_resolve_clip_labelmap_path,
    )


@app.get("/clip/classifiers")
def list_clip_classifiers():
    return _list_clip_classifiers()


def _resolve_clip_labelmap_path(path_str: Optional[str], *, root_hint: Optional[str] = None) -> Optional[Path]:
    return _resolve_clip_labelmap_path_impl(
        path_str,
        root_hint=root_hint,
        upload_root=UPLOAD_ROOT,
        labelmap_exts=LABELMAP_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root,
    )


def _find_labelmap_for_classifier(classifier_path: Path) -> Optional[Path]:
    return _find_labelmap_for_classifier_impl(
        classifier_path,
        upload_root=UPLOAD_ROOT,
        labelmap_exts=LABELMAP_ALLOWED_EXTS,
        path_is_within_root_fn=_path_is_within_root,
        joblib_load_fn=joblib.load,
        resolve_clip_labelmap_path_fn=_resolve_clip_labelmap_path,
    )


def _list_clip_labelmaps() -> List[Dict[str, Any]]:
    return _list_clip_labelmaps_impl(
        upload_root=UPLOAD_ROOT,
        labelmap_exts=LABELMAP_ALLOWED_EXTS,
        load_labelmap_file_fn=_load_labelmap_file,
    )


@app.get("/clip/labelmaps")
def list_clip_labelmaps():
    return _list_clip_labelmaps()


@app.get("/clip/classifiers/download")
def download_clip_classifier(rel_path: str = Query(...)):
    classifier_path = _resolve_agent_clip_classifier_path(rel_path)
    if classifier_path is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="classifier_not_found")
    stream = classifier_path.open("rb")
    headers = {"Content-Disposition": f'attachment; filename="{classifier_path.name}"'}
    return StreamingResponse(stream, media_type="application/octet-stream", headers=headers)


@app.get("/clip/classifiers/download_zip")
def download_clip_classifier_zip(rel_path: str = Query(...)):
    classifier_path = _resolve_agent_clip_classifier_path(rel_path)
    if classifier_path is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="classifier_not_found")
    buffer = io.BytesIO()
    meta_path = Path(os.path.splitext(str(classifier_path))[0] + ".meta.pkl")
    labelmap_path = _find_labelmap_for_classifier(classifier_path)
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


@app.delete("/clip/classifiers")
def delete_clip_classifier(rel_path: str = Query(...)):
    classifier_path = _resolve_agent_clip_classifier_path(rel_path)
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


@app.post("/clip/classifiers/rename")
def rename_clip_classifier(
    rel_path: str = Form(...),
    new_name: str = Form(...),
):
    classifier_path = _resolve_agent_clip_classifier_path(rel_path)
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
    if not _path_is_within_root(parent, classifiers_root):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_path_invalid")
    target_path = (parent / target_name).resolve()
    if not _path_is_within_root(target_path, classifiers_root):
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


@app.get("/clip/labelmaps/download")
def download_clip_labelmap(rel_path: str = Query(...), root: Optional[str] = Query(None)):
    labelmap_path = _resolve_clip_labelmap_path(rel_path, root_hint=root)
    if labelmap_path is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="labelmap_not_found")
    stream = labelmap_path.open("rb")
    headers = {"Content-Disposition": f'attachment; filename="{labelmap_path.name}"'}
    return StreamingResponse(stream, media_type="application/octet-stream", headers=headers)


@app.delete("/clip/labelmaps")
def delete_clip_labelmap(rel_path: str = Query(...), root: Optional[str] = Query(None)):
    labelmap_path = _resolve_clip_labelmap_path(rel_path, root_hint=root)
    if labelmap_path is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="labelmap_not_found")
    try:
        labelmap_path.unlink()
    except FileNotFoundError:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="labelmap_not_found")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return {"status": "deleted", "rel_path": rel_path}


@app.post("/fs/upload_classifier")
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


@app.post("/fs/upload_labelmap")
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
            _prepare_for_training()
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
            _finalize_training_environment()
            _cleanup_job(job)

    threading.Thread(target=worker, name=f"clip-train-{job.job_id[:8]}", daemon=True).start()


def _load_labelmap_simple(path: Optional[str]) -> List[str]:
    return _load_labelmap_simple_impl(path, load_labelmap_file_fn=_load_labelmap_file)


def _validate_clip_dataset(inputs: Dict[str, str]) -> Dict[str, Any]:
    return _validate_clip_dataset_impl(
        inputs,
        http_exception_cls=HTTPException,
        load_labelmap_simple_fn=_load_labelmap_simple,
    )


@app.post("/clip/train")
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
    _validate_clip_dataset({"images_dir": images_dir, "labels_dir": labels_dir, "labelmap_path": labelmap_path})
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
            message = _summarize_qwen_metric(payload)
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



@app.get("/clip/train")
def list_training_jobs():
    _prune_job_registry(TRAINING_JOBS, TRAINING_JOBS_LOCK)
    with TRAINING_JOBS_LOCK:
        jobs = sorted(TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        return [{"job_id": job.job_id, "status": job.status, "created_at": job.created_at} for job in jobs]


@app.get("/clip/train/{job_id}")
def get_training_job(job_id: str):
    job = _validate_job_exists(job_id)
    return _serialize_job(job)


@app.post("/clip/train/{job_id}/cancel")
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


@app.post("/sam3/train/jobs")
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


@app.get("/sam3/train/jobs")
def list_sam3_training_jobs():
    _prune_job_registry(SAM3_TRAINING_JOBS, SAM3_TRAINING_JOBS_LOCK)
    with SAM3_TRAINING_JOBS_LOCK:
        jobs = sorted(SAM3_TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        return [_serialize_sam3_job(job) for job in jobs]


@app.get("/sam3/train/jobs/{job_id}")
def get_sam3_training_job(job_id: str):
    job = _get_sam3_job(job_id)
    return _serialize_sam3_job(job)


@app.post("/sam3/train/jobs/{job_id}/cancel")
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


@app.post("/yolo/train/jobs")
def create_yolo_training_job(payload: YoloTrainRequest):
    if not payload.accept_tos:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_tos_required")
    job_id = uuid.uuid4().hex
    run_dir = _yolo_run_dir(job_id, create=True)
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
    _yolo_write_run_meta(
        run_dir,
        {
            "job_id": job_id,
            "status": job.status,
            "message": job.message,
            "config": job.config,
        },
    )
    if job.status != "blocked":
        _start_yolo_training_worker(job)
    return {"job_id": job_id}


@app.get("/yolo/train/jobs")
def list_yolo_training_jobs():
    _prune_job_registry(YOLO_TRAINING_JOBS, YOLO_TRAINING_JOBS_LOCK)
    with YOLO_TRAINING_JOBS_LOCK:
        jobs = sorted(YOLO_TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        return [_serialize_yolo_job(job) for job in jobs]


@app.get("/yolo/train/jobs/{job_id}")
def get_yolo_training_job(job_id: str):
    job = _get_yolo_job(job_id)
    return _serialize_yolo_job(job)


@app.post("/yolo/train/jobs/{job_id}/cancel")
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
        run_dir = _yolo_run_dir(job.job_id, create=False)
        _yolo_write_run_meta(
            run_dir,
            {
                "job_id": job.job_id,
                "status": job.status,
                "message": job.message,
                "config": job.config,
            },
        )
    return {"status": job.status}


@app.post("/yolo/head_graft/jobs")
def create_yolo_head_graft_job(payload: YoloHeadGraftRequest):
    if not payload.accept_tos:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_tos_required")
    job_id = uuid.uuid4().hex
    if payload.run_name:
        safe_id = _sanitize_yolo_run_id(payload.run_name)
        if safe_id:
            job_id = safe_id
    run_dir = _yolo_run_dir(job_id, create=False)
    if run_dir.exists():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="yolo_run_exists")
    config = payload.dict(exclude_none=True)
    job = YoloHeadGraftJob(job_id=job_id, config=config, message="Queued (head graft not started)")
    with YOLO_HEAD_GRAFT_JOBS_LOCK:
        YOLO_HEAD_GRAFT_JOBS[job_id] = job
        _yolo_head_graft_job_log(job, job.message)
    run_dir = _yolo_run_dir(job_id, create=True)
    _yolo_write_run_meta(
        run_dir,
        {"job_id": job_id, "status": job.status, "message": job.message, "config": job.config},
    )
    _start_yolo_head_graft_worker(job)
    return _serialize_yolo_head_graft_job(job)


@app.post("/yolo/head_graft/dry_run")
def yolo_head_graft_dry_run(payload: YoloHeadGraftDryRunRequest):
    base_run_id = str(payload.base_run_id or "").strip()
    if not base_run_id:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_base_missing")
    base_run_dir = _yolo_run_dir(base_run_id, create=False)
    if not base_run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_base_missing")
    base_meta = _yolo_load_run_meta(base_run_dir)
    base_cfg = base_meta.get("config") or {}
    base_task = str(base_cfg.get("task") or "detect").lower()
    base_variant = base_cfg.get("variant")
    base_labelmap = _yolo_load_run_labelmap(base_run_dir)
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
    new_labelmap = _yolo_load_labelmap(Path(dataset_info.get("yolo_labelmap_path") or ""))
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


@app.get("/yolo/head_graft/jobs")
def list_yolo_head_graft_jobs():
    _prune_job_registry(YOLO_HEAD_GRAFT_JOBS, YOLO_HEAD_GRAFT_JOBS_LOCK)
    with YOLO_HEAD_GRAFT_JOBS_LOCK:
        jobs = sorted(YOLO_HEAD_GRAFT_JOBS.values(), key=lambda job: job.created_at, reverse=True)
    return [_serialize_yolo_head_graft_job(job) for job in jobs]


@app.get("/yolo/head_graft/jobs/{job_id}")
def get_yolo_head_graft_job(job_id: str):
    _prune_job_registry(YOLO_HEAD_GRAFT_JOBS, YOLO_HEAD_GRAFT_JOBS_LOCK)
    with YOLO_HEAD_GRAFT_JOBS_LOCK:
        job = YOLO_HEAD_GRAFT_JOBS.get(job_id)
    if job:
        return _serialize_yolo_head_graft_job(job)
    raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="job_not_found")


@app.post("/yolo/head_graft/jobs/{job_id}/cancel")
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
            stopped = _yolo_head_graft_force_stop(job)
        next_status = "cancelled" if stopped else (job.status if job.status not in {"running", "queued"} else "cancelling")
        _yolo_head_graft_job_update(job, status=next_status, message="Cancellation requested ...")
        _yolo_head_graft_audit(job, "cancel_requested", event="cancel", extra={"forced": stopped})
    return {"status": job.status}


@app.post("/rfdetr/train/jobs")
def create_rfdetr_training_job(payload: RfDetrTrainRequest):
    if not payload.accept_tos:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_tos_required")
    job_id = uuid.uuid4().hex
    run_dir = _rfdetr_run_dir(job_id, create=True)
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
    _rfdetr_write_run_meta(
        run_dir,
        {
            "job_id": job_id,
            "status": job.status,
            "message": job.message,
            "config": job.config,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        },
    )
    _start_rfdetr_training_worker(job)
    return {"job_id": job_id}


@app.get("/rfdetr/train/jobs")
def list_rfdetr_training_jobs():
    _prune_job_registry(RFDETR_TRAINING_JOBS, RFDETR_TRAINING_JOBS_LOCK)
    with RFDETR_TRAINING_JOBS_LOCK:
        jobs = sorted(RFDETR_TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        return [_serialize_rfdetr_job(job) for job in jobs]


@app.get("/rfdetr/train/jobs/{job_id}")
def get_rfdetr_training_job(job_id: str):
    job = _get_rfdetr_job(job_id)
    return _serialize_rfdetr_job(job)


@app.post("/rfdetr/train/jobs/{job_id}/cancel")
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
        run_dir = _rfdetr_run_dir(job.job_id, create=False)
        _rfdetr_write_run_meta(
            run_dir,
            {
                "job_id": job.job_id,
                "status": job.status,
                "message": job.message,
                "config": job.config,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
            },
        )
    return {"status": job.status}


@app.get("/rfdetr/variants")
def list_rfdetr_variants(task: Optional[str] = Query(None)):
    if task:
        task_norm = task.strip().lower()
        return [v for v in RFDETR_VARIANTS if v.get("task") == task_norm]
    return RFDETR_VARIANTS


@app.get("/rfdetr/runs")
def list_rfdetr_runs():
    return _list_rfdetr_runs()


@app.get("/rfdetr/active")
def get_rfdetr_active():
    return _load_rfdetr_active()


@app.post("/rfdetr/active")
def set_rfdetr_active(payload: RfDetrActiveRequest):
    run_dir = _rfdetr_run_dir(payload.run_id, create=False)
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="rfdetr_run_not_found")
    best_path = _rfdetr_best_checkpoint(run_dir)
    if not best_path:
        raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="rfdetr_best_missing")
    meta = _rfdetr_load_run_meta(run_dir)
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
    return _save_rfdetr_active(active_payload)


@app.get("/detectors/default")
def get_default_detector():
    return _load_detector_default()


class DetectorDefaultRequest(BaseModel):
    mode: str


@app.post("/detectors/default")
def set_default_detector(payload: DetectorDefaultRequest):
    data = {"mode": payload.mode}
    return _save_detector_default(data)


@app.get("/rfdetr/runs/{run_id}/download")
def download_rfdetr_run(run_id: str):
    run_dir = _rfdetr_run_dir(run_id, create=False)
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="rfdetr_run_not_found")
    meta = _rfdetr_load_run_meta(run_dir)
    run_name = meta.get("config", {}).get("run_name") or meta.get("job_id") or run_id
    safe_name = _sanitize_yolo_run_id(run_name)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename in sorted(RFDETR_KEEP_FILES):
            path = run_dir / filename
            if path.exists():
                zf.write(path, arcname=filename)
    buffer.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{safe_name}.zip"'}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


@app.get("/yolo/runs/{run_id}/summary")
def yolo_run_summary(run_id: str):
    run_dir = _yolo_run_dir(run_id, create=False)
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_run_not_found")
    meta = _yolo_load_run_meta(run_dir)
    config = meta.get("config") or {}
    dataset = config.get("dataset") or {}
    run_name = config.get("run_name") or dataset.get("label") or dataset.get("id") or run_id
    labelmap = _read_labelmap_lines(run_dir / "labelmap.txt")
    metrics = _clean_metric_summary(_yolo_metrics_summary(run_dir))
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


@app.get("/rfdetr/runs/{run_id}/summary")
def rfdetr_run_summary(run_id: str):
    run_dir = _rfdetr_run_dir(run_id, create=False)
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="rfdetr_run_not_found")
    meta = _rfdetr_load_run_meta(run_dir)
    config = meta.get("config") or {}
    dataset = config.get("dataset") or {}
    run_name = config.get("run_name") or dataset.get("label") or dataset.get("id") or run_id
    labelmap = _read_labelmap_lines(run_dir / "labelmap.txt")
    metrics = _clean_metric_summary(_rfdetr_metrics_summary(run_dir))
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

@app.delete("/rfdetr/runs/{run_id}")
def delete_rfdetr_run(run_id: str):
    run_dir = _rfdetr_run_dir(run_id, create=False)
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="rfdetr_run_not_found")
    try:
        shutil.rmtree(run_dir)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return {"status": "deleted", "run_id": run_id}


@app.get("/yolo/variants")
def list_yolo_variants(task: Optional[str] = Query(None)):
    if task:
        task_norm = task.strip().lower()
        return [v for v in YOLO_VARIANTS if v.get("task") == task_norm]
    return YOLO_VARIANTS


@app.get("/yolo/runs")
def list_yolo_runs():
    return _list_yolo_runs()


@app.get("/yolo/active")
def get_yolo_active():
    return _load_yolo_active()


@app.post("/yolo/active")
def set_yolo_active(payload: YoloActiveRequest):
    run_dir = _yolo_run_dir(payload.run_id, create=False)
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_run_not_found")
    best_path = run_dir / "best.pt"
    if not best_path.exists():
        raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="yolo_best_missing")
    meta = _yolo_load_run_meta(run_dir)
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
    return _save_yolo_active(active_payload)


@app.post("/yolo/predict_region", response_model=YoloRegionResponse)
def yolo_predict_region(payload: YoloRegionRequest):
    model, labelmap, task = _ensure_yolo_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_region_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image(payload.image_base64, payload.image_token)
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


@app.post("/rfdetr/predict_region", response_model=RfDetrRegionResponse)
def rfdetr_predict_region(payload: RfDetrRegionRequest):
    model, labelmap, task = _ensure_rfdetr_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_region_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image(payload.image_base64, payload.image_token)
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


def _yolo_extract_detections(
    results: Any,
    labelmap: List[str],
    offset_x: float,
    offset_y: float,
    full_w: int,
    full_h: int,
) -> List[Dict[str, Any]]:
    return _yolo_extract_detections_impl(
        results,
        labelmap,
        offset_x,
        offset_y,
        full_w,
        full_h,
    )


def _rfdetr_extract_detections(
    results: Any,
    labelmap: List[str],
    offset_x: float,
    offset_y: float,
    full_w: int,
    full_h: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    return _rfdetr_extract_detections_impl(
        results,
        labelmap,
        offset_x,
        offset_y,
        full_w,
        full_h,
    )


def _resolve_detector_image(
    image_base64: Optional[str],
    image_token: Optional[str],
) -> Tuple[Image.Image, np.ndarray, str]:
    if image_token:
        for variant in ("sam1", "sam3"):
            cached = _fetch_preloaded_image(image_token, variant)
            if cached is not None:
                pil_img = Image.fromarray(cached)
                return pil_img, cached, image_token
        if image_base64:
            pil_img, np_img = _decode_image_base64(image_base64)
            token = hashlib.md5(np_img.tobytes()).hexdigest()
            _store_preloaded_image(token, np_img, "sam1")
            return pil_img, np_img, token
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="image_token_not_found")
    pil_img, np_img = _decode_image_base64(image_base64)
    token = hashlib.md5(np_img.tobytes()).hexdigest()
    _store_preloaded_image(token, np_img, "sam1")
    return pil_img, np_img, token


@app.post("/yolo/predict_full", response_model=YoloRegionResponse)
def yolo_predict_full(payload: YoloFullRequest):
    model, labelmap, task = _ensure_yolo_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_full_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image(payload.image_base64, payload.image_token)
    img_w, img_h = pil_img.size
    warnings: List[str] = []
    conf = _clamp_conf_value(float(payload.conf) if payload.conf is not None else 0.25, warnings)
    iou = _clamp_iou_value(float(payload.iou) if payload.iou is not None else 0.45, warnings)
    max_det = _clamp_max_det_value(int(payload.max_det) if payload.max_det is not None else 300, warnings)
    _apply_expected_labelmap_warnings(payload.expected_labelmap, labelmap, warnings)
    with YOLO_INFER_LOCK:
        results = model.predict(pil_img, conf=conf, iou=iou, max_det=max_det, verbose=False)
    raw = _yolo_extract_detections(results, labelmap, 0.0, 0.0, img_w, img_h)
    detections = [YoloRegionDetection(**item) for item in raw]
    return YoloRegionResponse(detections=detections, labelmap=labelmap, warnings=warnings or None)


@app.post("/yolo/predict_windowed", response_model=YoloRegionResponse)
def yolo_predict_windowed(payload: YoloWindowedRequest):
    model, labelmap, task = _ensure_yolo_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_windowed_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image(payload.image_base64, payload.image_token)
    img_w, img_h = pil_img.size
    warnings: List[str] = []
    conf = _clamp_conf_value(float(payload.conf) if payload.conf is not None else 0.25, warnings)
    iou = _clamp_iou_value(float(payload.iou) if payload.iou is not None else 0.45, warnings)
    max_det = _clamp_max_det_value(int(payload.max_det) if payload.max_det is not None else 300, warnings)
    slice_size = int(payload.slice_size) if payload.slice_size is not None else 640
    overlap = float(payload.overlap) if payload.overlap is not None else 0.2
    merge_iou = float(payload.merge_iou) if payload.merge_iou is not None else 0.5
    slice_size, overlap, merge_iou = _clamp_slice_params(slice_size, overlap, merge_iou, img_w, img_h, warnings)
    _apply_expected_labelmap_warnings(payload.expected_labelmap, labelmap, warnings)
    slices, starts = _slice_image_sahi(pil_img, slice_size, overlap)
    raw_detections: List[Dict[str, Any]] = []
    for tile, start in zip(slices, starts):
        offset_x, offset_y = float(start[0]), float(start[1])
        crop = Image.fromarray(tile)
        results = model.predict(crop, conf=conf, iou=iou, max_det=max_det, verbose=False)
        raw_detections.extend(_yolo_extract_detections(results, labelmap, offset_x, offset_y, img_w, img_h))
    merged = _merge_detections_nms(raw_detections, merge_iou, max_det)
    detections = [YoloRegionDetection(**item) for item in merged]
    return YoloRegionResponse(detections=detections, labelmap=labelmap, warnings=warnings or None)


@app.post("/rfdetr/predict_full", response_model=RfDetrRegionResponse)
def rfdetr_predict_full(payload: RfDetrFullRequest):
    model, labelmap, task = _ensure_rfdetr_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_full_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image(payload.image_base64, payload.image_token)
    img_w, img_h = pil_img.size
    warnings: List[str] = []
    conf = _clamp_conf_value(float(payload.conf) if payload.conf is not None else 0.25, warnings)
    max_det = _clamp_max_det_value(int(payload.max_det) if payload.max_det is not None else 300, warnings)
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


@app.post("/rfdetr/predict_windowed", response_model=RfDetrRegionResponse)
def rfdetr_predict_windowed(payload: RfDetrWindowedRequest):
    model, labelmap, task = _ensure_rfdetr_inference_runtime()
    task_name = str(task).lower() if task else None
    if not task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_task_unknown")
    if "segment" in task_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="rfdetr_windowed_detect_requires_bbox")
    pil_img, _np_img, _token = _resolve_detector_image(payload.image_base64, payload.image_token)
    img_w, img_h = pil_img.size
    warnings: List[str] = []
    conf = _clamp_conf_value(float(payload.conf) if payload.conf is not None else 0.25, warnings)
    max_det = _clamp_max_det_value(int(payload.max_det) if payload.max_det is not None else 300, warnings)
    slice_size = int(payload.slice_size) if payload.slice_size is not None else 640
    overlap = float(payload.overlap) if payload.overlap is not None else 0.2
    merge_iou = float(payload.merge_iou) if payload.merge_iou is not None else 0.5
    slice_size, overlap, merge_iou = _clamp_slice_params(slice_size, overlap, merge_iou, img_w, img_h, warnings)
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


@app.get("/yolo/runs/{run_id}/download")
def download_yolo_run(run_id: str):
    run_dir = _yolo_run_dir(run_id, create=False)
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_run_not_found")
    meta = _yolo_load_run_meta(run_dir)
    run_name = meta.get("config", {}).get("run_name") or meta.get("job_id") or run_id
    safe_name = _sanitize_yolo_run_id(run_name)
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


@app.get("/yolo/head_graft/jobs/{job_id}/bundle")
def download_yolo_head_graft_bundle(job_id: str):
    run_dir = _yolo_run_dir(job_id, create=False)
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_run_not_found")
    meta = _yolo_load_run_meta(run_dir)
    if not meta.get("head_graft"):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_head_graft_not_found")
    run_name = meta.get("config", {}).get("run_name") or meta.get("job_id") or job_id
    safe_name = _sanitize_yolo_run_id(run_name)
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


@app.delete("/yolo/runs/{run_id}")
def delete_yolo_run(run_id: str):
    run_dir = _yolo_run_dir(run_id, create=False)
    if not run_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="yolo_run_not_found")
    try:
        shutil.rmtree(run_dir)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return {"status": "deleted", "run_id": run_id}


@app.get("/sam3/train/cache_size")
def sam3_train_cache_size():
    cache_root = SAM3_JOB_ROOT / "splits"
    return {"bytes": _dir_size_bytes(cache_root)}


@app.post("/sam3/train/cache/purge")
def sam3_train_cache_purge():
    cache_root = SAM3_JOB_ROOT / "splits"
    deleted = _purge_directory(cache_root)
    return {"status": "ok", "deleted_bytes": deleted}


@app.get("/sam3/storage/runs")
def list_sam3_runs(variant: str = Query("sam3")):
    # SAM3-lite removed; always use sam3
    return _list_sam3_runs("sam3")


@app.delete("/sam3/storage/runs/{run_id}")
def delete_sam3_run(run_id: str, variant: str = Query("sam3"), scope: str = Query("all")):
    normalized = "sam3"
    if scope not in SAM3_STORAGE_SCOPES:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_scope")
    run_dir = _run_dir_for_request(run_id, normalized)
    active_paths = _active_run_paths_for_variant(normalized)
    if run_dir.resolve() in active_paths:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="sam3_run_active")
    deleted, freed = _delete_run_scope(run_dir, scope)
    return {"deleted": deleted, "freed_bytes": freed}


@app.post("/sam3/storage/runs/{run_id}/promote")
def promote_sam3_run(run_id: str, variant: str = Query("sam3")):
    return _promote_run(run_id, "sam3")


@app.get("/sam3/models/available")
def list_sam3_available_models(
    variant: str = Query("sam3"),
    promoted_only: bool = Query(False),
):
    """List run checkpoints for prompt model selection."""
    runs = _list_sam3_runs("sam3")
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

@app.post("/sam3/models/activate")
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


@app.post("/qwen/train/jobs")
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


@app.get("/qwen/train/jobs")
def list_qwen_training_jobs(request: Request):
    _prune_job_registry(QWEN_TRAINING_JOBS, QWEN_TRAINING_JOBS_LOCK)
    with QWEN_TRAINING_JOBS_LOCK:
        jobs = sorted(QWEN_TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        _log_qwen_get_request(str(request.url.path), jobs)
        return [_serialize_qwen_job(job) for job in jobs]


@app.get("/qwen/train/jobs/{job_id}")
def get_qwen_training_job(job_id: str, request: Request):
    job = _get_qwen_job(job_id)
    _log_qwen_get_request(str(request.url.path), [job])
    return _serialize_qwen_job(job)


@app.post("/qwen/train/jobs/{job_id}/cancel")
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


@app.get("/qwen/train/cache_size")
def qwen_train_cache_size():
    cache_root = QWEN_JOB_ROOT / "splits"
    return {"bytes": _dir_size_bytes(cache_root)}


@app.post("/qwen/train/cache/purge")
def qwen_train_cache_purge():
    cache_root = QWEN_JOB_ROOT / "splits"
    deleted = _purge_directory(cache_root)
    return {"status": "ok", "deleted_bytes": deleted}


@app.get("/qwen/models")
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


@app.post("/qwen/models/activate")
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


@app.get("/clip/active_model", response_model=ActiveModelResponse)
def get_active_model():
    return _current_active_payload()


@app.post("/clip/active_model", response_model=ActiveModelResponse)
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
        inferred = _infer_clip_model_from_embedding_dim(embed_dim, active_name=clip_model_name or DEFAULT_CLIP_MODEL)
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
            inferred = _infer_clip_model_from_embedding_dim(embed_dim, active_name=clip_name)
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
            target_device = _dinov3_resolve_device(device)
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
        active_head_normalize_embeddings = _resolve_active_head_normalize_embeddings(meta_obj, new_clf, default=True)
        try:
            active_classifier_head = _load_clip_head_from_classifier(Path(classifier_path_abs))
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


# note this one is actually not used. For a while I thought it would be cool to send a smaller crop to SAM but I'm not sure it makes sense since
# now I'm caching / checking the file that is currently loaded in the predictor and not updating on every call so it's actually waaaay faster and we have the whole image
# ---------------------------------------------------------------------------
# SAM preload endpoint
# ---------------------------------------------------------------------------

@app.post("/sam_preload", response_model=SamPreloadResponse)
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


@app.get("/sam_slots", response_model=List[SamSlotStatus])
def sam_slots():
    return predictor_manager.status()


@app.post("/sam_activate_slot", response_model=SamActivateResponse)
def sam_activate_slot(payload: SamActivateRequest):
    variant = _default_variant(payload.sam_variant)
    slot = predictor_manager.get_slot_for_image(payload.image_name, variant)
    if slot is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="slot_not_found")
    promoted = predictor_manager.promote_slot(slot.name)
    if not promoted and slot.name != "current":
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="slot_busy")
    return SamActivateResponse(status="promoted", slot="current", token=slot.token)


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


@app.get("/predictor_settings", response_model=PredictorSettings)
def get_predictor_settings():
    return _predictor_settings_payload()


@app.post("/predictor_settings", response_model=PredictorSettings)
def update_predictor_settings(payload: PredictorSettingsUpdate):
    min_cap, max_cap = predictor_manager.capacity_limits()
    try:
        requested = int(payload.max_predictors)
    except Exception:
        requested = min_cap
    normalized = max(min_cap, min(max_cap, requested))
    predictor_manager.set_capacity(normalized)
    return _predictor_settings_payload()


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


@app.get("/system/gpu")
def get_system_gpu():
    return _gpu_status_payload()


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


@app.get("/system/storage_check")
def system_storage_check():
    return _storage_check_payload()


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
        datasets = _list_all_datasets(prefer_registry=True)
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
        summary["models"]["sam3_runs"] = len(_list_sam3_runs("sam3"))
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"sam3_runs_failed:{exc}")
        summary["ok"] = False
    try:
        summary["models"]["qwen_models"] = len(_list_qwen_model_entries())
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"qwen_models_failed:{exc}")
        summary["ok"] = False
    try:
        summary["models"]["clip_classifiers"] = len(_list_clip_classifiers())
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"clip_classifiers_failed:{exc}")
        summary["ok"] = False
    summary["models"]["yolo_variants"] = len(YOLO_VARIANTS)
    summary["models"]["rfdetr_variants"] = len(RFDETR_VARIANTS)
    return summary


@app.get("/system/health_summary")
def system_health_summary():
    return _system_health_summary()


@app.get("/qwen/status")
def qwen_status():
    dependency_error = str(QWEN_IMPORT_ERROR) if QWEN_IMPORT_ERROR else None
    device_guess = qwen_device
    pending_error = qwen_last_error
    if not device_guess and not dependency_error:
        try:
            device_guess = _resolve_qwen_device()
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


@app.get("/qwen/settings", response_model=QwenRuntimeSettings)
def qwen_settings():
    return QwenRuntimeSettings(trust_remote_code=QWEN_TRUST_REMOTE_CODE)


@app.post("/qwen/settings", response_model=QwenRuntimeSettings)
def update_qwen_settings(payload: QwenRuntimeSettingsUpdate):
    global QWEN_TRUST_REMOTE_CODE
    if payload.trust_remote_code is not None:
        desired = bool(payload.trust_remote_code)
        if desired != QWEN_TRUST_REMOTE_CODE:
            QWEN_TRUST_REMOTE_CODE = desired
            _unload_qwen_runtime()
    return QwenRuntimeSettings(trust_remote_code=QWEN_TRUST_REMOTE_CODE)


@app.post("/runtime/unload")
def unload_all_runtimes():
    _unload_inference_runtimes()
    return {"status": "unloaded"}


@app.post("/qwen/unload")
def qwen_unload():
    _unload_qwen_runtime()
    return {"status": "unloaded"}


@app.post("/qwen/infer", response_model=QwenInferenceResponse)
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
        final_prompt = _render_qwen_prompt(
            prompt_type,
            items=item_list,
            image_type=(payload.image_type or "").strip() or None,
            extra_context=(payload.extra_context or "").strip() or None,
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


@app.post("/qwen/caption", response_model=QwenCaptionResponse)
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

    def get_runtime(model_id: Optional[str]) -> Tuple[Any, Any]:
        nonlocal active_model_id, active_runtime
        if multi_model_cache:
            key = model_id or "__active__"
            cached = request_model_cache.get(key)
            if cached:
                return cached
            if model_id:
                runtime = _ensure_qwen_ready_for_caption(model_id)
            else:
                runtime = _ensure_qwen_ready()
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
                active_runtime = _ensure_qwen_ready()
                active_model_id = None
        return active_runtime

    try:
        if payload.unload_others and not fast_mode:
            _unload_non_qwen_runtimes()
        pil_img, _, _ = resolve_image_payload(payload.image_base64, payload.image_token, None)
        user_prompt = (payload.user_prompt or "").strip()
        include_counts = bool(payload.include_counts)
        include_coords = bool(payload.include_coords)
        max_boxes = payload.max_boxes if payload.max_boxes is not None else 0
        max_new_tokens = payload.max_new_tokens if payload.max_new_tokens is not None else 128
        label_hints = payload.label_hints or []
        allowed_labels = _allowed_caption_labels(label_hints)
        image_width = payload.image_width or pil_img.width
        image_height = payload.image_height or pil_img.height
        caption_mode = payload.caption_mode or "full"
        restrict_to_labels = payload.restrict_to_labels if payload.restrict_to_labels is not None else True
        caption_all_windows = True if caption_mode == "windowed" else bool(payload.caption_all_windows)
        detailed_mode = caption_mode == "windowed"
        glossary_map = _caption_glossary_map(
            payload.labelmap_glossary,
            [hint.label for hint in label_hints if hint.label],
        )
        allowed_labels_prompt = (
            [_caption_preferred_label(label, glossary_map) for label in allowed_labels]
            if allowed_labels
            else []
        )
        prompt_text, counts, used_boxes, truncated = _build_qwen_caption_prompt(
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
        variant = payload.model_variant or "auto"
        model_id_override = payload.model_id or ""
        if model_id_override:
            desired_model_id = model_id_override
        else:
            desired_model_id = _resolve_qwen_variant_model_id(base_model_id, variant)
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
        decode_params = _resolve_qwen_caption_decode(payload, is_thinking)
        deterministic_decode = {"do_sample": False}
        if is_thinking:
            prompt_text = _adjust_prompt_for_thinking(prompt_text)
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
            overlap = _resolve_qwen_window_overlap(payload.window_overlap)
            window_size = _resolve_qwen_window_size(None, image_width, image_height, overlap=overlap)
            force_two = True
            x_positions = _window_positions(image_width, window_size, overlap, force_two=force_two)
            y_positions = _window_positions(image_height, window_size, overlap, force_two=force_two)
            grouped_hints = _group_hints_by_window(label_hints, x_positions, y_positions, window_size)
            window_model_id = desired_model_id
            window_base_model_id = window_model_id
            window_is_thinking = "Thinking" in window_model_id
            for y0 in y_positions:
                for x0 in x_positions:
                    window_hints = grouped_hints.get((x0, y0), [])
                    if not window_hints and not caption_all_windows:
                        continue
                    window_allowed = _allowed_caption_labels(window_hints)
                    window_glossary_map = _caption_glossary_map(
                        payload.labelmap_glossary,
                        [hint.label for hint in window_hints if hint.label],
                    )
                    window_allowed_prompt = (
                        [_caption_preferred_label(label, window_glossary_map) for label in window_allowed]
                        if window_allowed
                        else []
                    )
                    window_prompt, window_counts, _, _ = _build_qwen_caption_prompt(
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
                        window_prompt = _adjust_prompt_for_thinking(window_prompt)
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
                    window_caption, _ = _extract_caption_from_text(qwen_text, marker=None)
                    window_caption = _sanitize_qwen_caption(window_caption)
                    if window_is_thinking and _thinking_caption_needs_cleanup(window_caption, qwen_text):
                        cleanup_model = _resolve_qwen_variant_model_id(window_base_model_id, "Instruct")
                        window_caption = _run_qwen_caption_cleanup(
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
                    if _caption_is_degenerate(window_caption):
                        cleanup_model = _resolve_qwen_variant_model_id(window_base_model_id, "Instruct")
                        window_caption = _run_qwen_caption_cleanup(
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
                    if _caption_needs_completion(window_caption) or _caption_has_meta(window_caption):
                        cleanup_model = _resolve_qwen_variant_model_id(window_base_model_id, "Instruct")
                        window_caption = _run_qwen_caption_cleanup(
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
                    needs_refine, missing = _caption_needs_refine(
                        window_caption,
                        window_counts,
                        detailed_mode=True,
                        include_counts=include_counts,
                        glossary_map=window_glossary_map,
                    )
                    if needs_refine:
                        refine_model = _resolve_qwen_variant_model_id(window_base_model_id, "Instruct")
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
                        window_caption, _ = _extract_caption_from_text(refine_text, marker=None)
                        window_caption = _sanitize_qwen_caption(window_caption)
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
            draft_caption, _ = _extract_caption_from_text(draft_text, marker="DRAFT")
            draft_caption = _sanitize_qwen_caption(draft_caption)
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
            refine_model = _resolve_qwen_variant_model_id(caption_base_model_id, "Instruct")
            qwen_text, _, _ = _run_qwen_inference(
                refine_prompt,
                pil_img,
                max_new_tokens=refine_max_tokens,
                system_prompt_override=refine_system,
                runtime_override=get_runtime(refine_model),
                decode_override=deterministic_decode,
            )
            caption_text, _ = _extract_caption_from_text(qwen_text, marker=None)
            if final_only or is_thinking:
                caption_text = _sanitize_qwen_caption(caption_text)
        else:
            qwen_text, _, _ = _run_qwen_inference(
                prompt_text,
                pil_img,
                max_new_tokens=max_new_tokens,
                system_prompt_override=system_prompt,
                runtime_override=resolve_main_runtime(),
                decode_override=decode_params,
            )
            caption_text, _ = _extract_caption_from_text(qwen_text, marker=None)
            if final_only or is_thinking:
                caption_text = _sanitize_qwen_caption(caption_text)
            if is_thinking and _thinking_caption_needs_cleanup(caption_text, qwen_text):
                cleanup_model = _resolve_qwen_variant_model_id(caption_base_model_id, "Instruct")
                caption_text = _run_qwen_caption_cleanup(
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
            caption_text = _run_qwen_caption_merge(
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
                caption_text = _sanitize_qwen_caption(caption_text)
        if _caption_is_degenerate(caption_text):
            cleanup_model = _resolve_qwen_variant_model_id(caption_base_model_id, "Instruct")
            caption_text = _run_qwen_caption_cleanup(
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
        if _caption_needs_completion(caption_text) or _caption_has_meta(caption_text):
            cleanup_model = _resolve_qwen_variant_model_id(caption_base_model_id, "Instruct")
            caption_text = _run_qwen_caption_cleanup(
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
        if caption_mode == "windowed" and "4B" in desired_model_id and _caption_needs_short_form(caption_text):
            cleanup_model = _resolve_qwen_variant_model_id(caption_base_model_id, "Instruct")
            caption_text = _run_qwen_caption_cleanup(
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
        needs_refine, missing = _caption_needs_refine(
            caption_text,
            counts,
            detailed_mode=detailed_mode,
            include_counts=include_counts,
            glossary_map=glossary_map,
        )
        if needs_refine:
            refine_model = _resolve_qwen_variant_model_id(caption_base_model_id, "Instruct")
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
            caption_text, _ = _extract_caption_from_text(refine_text, marker=None)
            caption_text = _sanitize_qwen_caption(caption_text)
            refine_count += 1
        if caption_text and _caption_needs_english_rewrite(caption_text):
            rewrite_model = _resolve_qwen_variant_model_id(base_model_id, "Instruct")
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
            caption_text, _ = _extract_caption_from_text(rewrite_text, marker=None)
            if final_only or is_thinking:
                caption_text = _sanitize_qwen_caption(caption_text)
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


@app.post("/qwen/prepass", response_model=QwenPrepassResponse)
def qwen_prepass(payload: QwenPrepassRequest):
    try:
        payload = payload.copy(update={"prepass_only": True})
        return _run_prepass_annotation(payload)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_prepass_failed:{exc}") from exc


@app.post("/calibration/jobs")
def start_calibration_job(payload: CalibrationRequest = Body(...)):
    job = _start_calibration_job(payload)
    return _serialize_calibration_job(job)


@app.get("/calibration/jobs")
def list_calibration_jobs():
    _prune_job_registry(CALIBRATION_JOBS, CALIBRATION_JOBS_LOCK)
    with CALIBRATION_JOBS_LOCK:
        jobs = list(CALIBRATION_JOBS.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_serialize_calibration_job(job) for job in jobs]


@app.get("/calibration/jobs/{job_id}")
def get_calibration_job(job_id: str):
    with CALIBRATION_JOBS_LOCK:
        job = CALIBRATION_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="calibration_job_not_found")
    return _serialize_calibration_job(job)


@app.post("/calibration/jobs/{job_id}/cancel")
def cancel_calibration_job(job_id: str):
    job = _cancel_calibration_job(job_id)
    return _serialize_calibration_job(job)


@app.get("/prepass/recipes")
def list_prepass_recipes():
    return _list_prepass_recipes()


@app.get("/prepass/recipes/{recipe_id}", response_model=PrepassRecipeResponse)
def get_prepass_recipe(recipe_id: str):
    data = _get_prepass_recipe_impl(
        recipe_id,
        recipes_root=PREPASS_RECIPE_ROOT,
        sanitize_run_id_fn=_sanitize_yolo_run_id,
        load_meta_fn=_load_prepass_recipe_meta,
        prepass_schema_version=PREPASS_RECIPE_SCHEMA_VERSION,
    )
    return PrepassRecipeResponse(**data)


@app.post("/prepass/recipes", response_model=PrepassRecipeResponse)
def save_prepass_recipe(payload: PrepassRecipeRequest):
    recipe_id = payload.recipe_id or uuid.uuid4().hex
    data = _save_prepass_recipe_impl(
        payload.dict(),
        recipe_id=recipe_id,
        prepass_schema_version=PREPASS_RECIPE_SCHEMA_VERSION,
        recipes_root=PREPASS_RECIPE_ROOT,
        sanitize_run_id_fn=_sanitize_yolo_run_id,
        normalize_glossary_fn=_normalize_labelmap_glossary,
        write_meta_fn=_write_prepass_recipe_meta,
    )
    return PrepassRecipeResponse(**data)


@app.delete("/prepass/recipes/{recipe_id}")
def delete_prepass_recipe(recipe_id: str):
    _delete_prepass_recipe_impl(
        recipe_id,
        recipes_root=PREPASS_RECIPE_ROOT,
        sanitize_run_id_fn=_sanitize_yolo_run_id,
    )
    return {"status": "deleted", "id": recipe_id}


def _collect_recipe_assets(recipe_meta: Dict[str, Any], temp_dir: Path) -> Dict[str, Any]:
    return _collect_recipe_assets_impl(
        recipe_meta,
        temp_dir,
        read_labelmap_lines_fn=_read_labelmap_lines,
        load_labelmap_meta_fn=_agent_load_labelmap_meta,
        active_labelmap_path=active_labelmap_path,
        sanitize_run_id_fn=_sanitize_yolo_run_id,
        copy_tree_filtered_fn=_copy_tree_filtered,
        sha256_fn=_sha256_path,
        get_qwen_model_entry_fn=_get_qwen_model_entry,
        resolve_classifier_path_fn=_resolve_agent_clip_classifier_path,
        yolo_job_root=YOLO_JOB_ROOT,
        rfdetr_job_root=RFDETR_JOB_ROOT,
        rfdetr_keep_files=RFDETR_KEEP_FILES,
        qwen_metadata_filename=QWEN_METADATA_FILENAME,
        qwen_job_root=QWEN_JOB_ROOT,
        upload_root=UPLOAD_ROOT,
        calibration_root=CALIBRATION_ROOT,
    )


@app.post("/prepass/recipes/{recipe_id}/export")
def export_prepass_recipe(recipe_id: str):
    zip_path = _export_prepass_recipe_impl(
        recipe_id,
        prepass_recipe_meta=PREPASS_RECIPE_META,
        prepass_schema_version=PREPASS_RECIPE_SCHEMA_VERSION,
        prepass_recipe_export_root=PREPASS_RECIPE_EXPORT_ROOT,
        prepass_recipe_root=PREPASS_RECIPE_ROOT,
        sanitize_run_id_fn=_sanitize_yolo_run_id,
        load_meta_fn=_load_prepass_recipe_meta,
        collect_assets_fn=_collect_recipe_assets,
    )
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"prepass_recipe_{recipe_id}.zip",
    )


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
        sanitize_run_id_fn=_sanitize_yolo_run_id,
    )
    return PrepassRecipeResponse(**data)


@app.post("/prepass/recipes/import", response_model=PrepassRecipeResponse)
def import_prepass_recipe(file: UploadFile = File(...)):  # noqa: B008
    temp_dir = Path(tempfile.mkdtemp(prefix="prepass_recipe_import_"))
    try:
        zip_path = temp_dir / "upload.zip"
        with zip_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        return _import_prepass_recipe_from_zip(zip_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/prepass/recipes/import-raw", response_model=PrepassRecipeResponse)
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

@app.post("/sam3/text_prompt", response_model=Sam3TextPromptResponse)
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
                    payload = encode_binary_mask(masks_arr[idx])
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


@app.post("/sam3/text_prompt_auto", response_model=Sam3TextPromptAutoResponse)
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
            mask_payload = encode_binary_mask(mask)
        if mask is not None:
            try:
                x_min, y_min, x_max, y_max = mask_to_bounding_box(mask)
            except Exception:
                x_min, y_min, x_max, y_max = yolo_to_corners(det.bbox, pil_img.width, pil_img.height)
        else:
            x_min, y_min, x_max, y_max = yolo_to_corners(det.bbox, pil_img.width, pil_img.height)
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


@app.post("/sam3/visual_prompt", response_model=Sam3TextPromptResponse)
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
                    payload_mask = encode_binary_mask(masks_arr[idx])
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


@app.post("/sam_point", response_model=YoloBboxOutput)
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
    left, top, right, bottom = mask_to_bounding_box(mask_arr)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
    return YoloBboxOutput(
        class_id="0",
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
    )


@app.post("/sam_bbox_auto", response_model=SamPointAutoResponse)
def sam_bbox_auto(prompt: BboxPrompt):
    if not _active_encoder_ready():
        return SamPointAutoResponse(prediction=ERROR_MESSAGE, bbox=[], uuid=prompt.uuid)

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
    x_min, y_min, x_max, y_max = mask_to_bounding_box(mask_arr)
    yolo_box = to_yolo(full_w, full_h, x_min, y_min, x_max, y_max)
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
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
        error=details.get("error"),
    )


@app.post("/sam_point_auto", response_model=SamPointAutoResponse)
def sam_point_auto(prompt: PointPrompt):
    if not _active_encoder_ready():
        return SamPointAutoResponse(prediction=ERROR_MESSAGE, bbox=[], uuid=prompt.uuid)

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
    left, top, right, bottom = mask_to_bounding_box(mask_arr)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
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
            mask=encode_binary_mask(mask_arr),
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
            mask=encode_binary_mask(mask_arr),
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
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
        error=details.get("error"),
    )


@app.post("/sam_point_multi", response_model=YoloBboxOutput)
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
    left, top, right, bottom = mask_to_bounding_box(mask_arr)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
    return YoloBboxOutput(
        class_id="0",
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
    )


@app.post("/sam_point_multi_auto", response_model=SamPointAutoResponse)
def sam_point_multi_auto(prompt: MultiPointPrompt):
    if not _active_encoder_ready():
        return SamPointAutoResponse(prediction=ERROR_MESSAGE, bbox=[], uuid=prompt.uuid)

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
    left, top, right, bottom = mask_to_bounding_box(mask_arr)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
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
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
        error=details.get("error"),
    )


@app.post("/sam_bbox", response_model=YoloBboxOutput)
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
    x_min, y_min, x_max, y_max = mask_to_bounding_box(mask_arr)
    yolo_box = to_yolo(full_w, full_h, x_min, y_min, x_max, y_max)
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
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
    )

@app.post("/crop_zip_init")
def crop_zip_init():
    jobId = str(uuid.uuid4())
    job_store[jobId] = []
    return {"jobId": jobId}

@app.post("/crop_zip_chunk")
def crop_zip_chunk(request: CropZipRequest, jobId: str = Query(...)):
    if jobId not in job_store:
        raise HTTPException(status_code=400, detail="Invalid jobId")
    job_store[jobId].extend(request.images)
    return {"status": "ok", "count": len(request.images)}

@app.get("/crop_zip_finalize")
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


if os.environ.get("COORD_ROUNDTRIP_TEST") == "1":
    _coord_roundtrip_smoke()
