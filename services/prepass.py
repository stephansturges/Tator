"""Prepass pipeline helpers (detectors + SAM3 + dedupe orchestration)."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from utils.coords import _agent_iou_xyxy


def _agent_source_counts(detections: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for det in detections:
        sources = det.get("source_list")
        if isinstance(sources, (list, tuple)) and sources:
            for src in sources:
                source = str(src or "unknown")
                counts[source] = counts.get(source, 0) + 1
            continue
        source = str(det.get("source") or det.get("score_source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _agent_format_source_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "none"
    parts = [f"{key}={counts[key]}" for key in sorted(counts.keys())]
    return ", ".join(parts)


def _agent_label_counts_summary(detections: Sequence[Dict[str, Any]], limit: int = 8) -> str:
    counts: Dict[str, int] = {}
    for det in detections:
        label = str(det.get("label") or det.get("class_name") or "").strip()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "none"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    parts = [f"{label}({count})" for label, count in ordered[:limit]]
    if len(ordered) > limit:
        parts.append(f"+{len(ordered) - limit} more")
    return ", ".join(parts)


def _agent_extract_atom_ids(det: Dict[str, Any]) -> List[str]:
    atom_ids: List[str] = []
    raw_ids = det.get("prepass_atom_ids")
    if isinstance(raw_ids, (list, tuple)):
        for raw in raw_ids:
            value = str(raw or "").strip()
            if value and value not in atom_ids:
                atom_ids.append(value)
    raw_single = det.get("prepass_atom_id")
    if raw_single is not None:
        value = str(raw_single).strip()
        if value and value not in atom_ids:
            atom_ids.append(value)
    return atom_ids


def _agent_stage_bucket_for_detection(det: Dict[str, Any]) -> Tuple[str, str]:
    source = str(det.get("source_primary") or det.get("source") or det.get("score_source") or "unknown").strip().lower()
    detector_run_id = str(det.get("source_detector_run_id") or "").strip()
    has_window = bool(det.get("window_bbox_2d") or det.get("grid_cell"))
    if source in {"yolo", "rfdetr"}:
        run_name = "full"
        if detector_run_id:
            run_name = detector_run_id.rsplit(":", 1)[-1].strip().lower() or "full"
        elif has_window:
            run_name = "windowed"
        return "detector", f"{source}_{run_name}"
    if source == "sam3_text":
        return "sam3_text", "windowed" if has_window else "full"
    if source == "sam3_similarity":
        return "sam3_similarity", "windowed" if has_window else "full"
    return source or "unknown", "default"


def _agent_register_provenance_atoms(
    detections: Sequence[Dict[str, Any]],
    *,
    next_atom_index: int,
    atoms: List[Dict[str, Any]],
    stage_atoms: Dict[str, Dict[str, List[str]]],
) -> int:
    for det in detections:
        if not isinstance(det, dict):
            continue
        atom_id = f"a{next_atom_index:07d}"
        next_atom_index += 1
        det["prepass_atom_id"] = atom_id
        det["prepass_atom_ids"] = [atom_id]
        stage_name, run_name = _agent_stage_bucket_for_detection(det)
        stage_bucket = stage_atoms.setdefault(stage_name, {})
        run_bucket = stage_bucket.setdefault(run_name, [])
        run_bucket.append(atom_id)
        bbox_xyxy = det.get("bbox_xyxy_px")
        if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) < 4:
            bbox_xyxy = None
        bbox_2d = det.get("bbox_2d")
        if not isinstance(bbox_2d, (list, tuple)) or len(bbox_2d) < 4:
            bbox_2d = None
        score_val = det.get("score")
        if score_val is not None:
            try:
                score_val = float(score_val)
            except (TypeError, ValueError):
                score_val = None
        atom_record = {
            "atom_id": atom_id,
            "stage": stage_name,
            "run": run_name,
            "label": str(det.get("label") or det.get("class_name") or "").strip() or None,
            "source": det.get("source"),
            "source_primary": det.get("source_primary"),
            "source_detector_run_id": det.get("source_detector_run_id"),
            "source_prompt": det.get("source_prompt"),
            "source_exemplar_handles": list(det.get("source_exemplar_handles") or []),
            "score": score_val,
            "score_source": det.get("score_source"),
            "bbox_xyxy_px": list(bbox_xyxy[:4]) if bbox_xyxy else None,
            "bbox_2d": list(bbox_2d[:4]) if bbox_2d else None,
            "window_bbox_2d": list(det.get("window_bbox_2d") or []) or None,
            "grid_cell": det.get("grid_cell"),
            "sam3_prompt_term": det.get("sam3_prompt_term"),
            "sam3_prompt_label": det.get("sam3_prompt_label"),
            "sam3_prompt_source": det.get("sam3_prompt_source"),
        }
        atoms.append(atom_record)
    return next_atom_index


def _agent_build_final_cluster_provenance(detections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    final_clusters: List[Dict[str, Any]] = []
    for idx, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        score_val = det.get("score")
        if score_val is not None:
            try:
                score_val = float(score_val)
            except (TypeError, ValueError):
                score_val = None
        bbox_xyxy = det.get("bbox_xyxy_px")
        if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) < 4:
            bbox_xyxy = None
        final_clusters.append(
            {
                "index": idx,
                "label": str(det.get("label") or det.get("class_name") or "").strip() or None,
                "source": det.get("source"),
                "score_source": det.get("score_source"),
                "score": score_val,
                "bbox_xyxy_px": list(bbox_xyxy[:4]) if bbox_xyxy else None,
                "atom_ids": _agent_extract_atom_ids(det),
                "source_list": list(det.get("source_list") or []),
                "score_by_source": dict(det.get("score_by_source") or {}),
            }
        )
    return final_clusters


def _agent_merge_prepass_detections(
    detections: List[Dict[str, Any]],
    *,
    iou_thr: float = 0.85,
    cross_class_iou_thr: Optional[float] = None,
    fusion_mode: str = "primary",
) -> Tuple[List[Dict[str, Any]], int]:
    if not detections or iou_thr <= 0:
        return detections, 0
    if cross_class_iou_thr is not None:
        try:
            cross_class_iou_thr = float(cross_class_iou_thr)
        except (TypeError, ValueError):
            cross_class_iou_thr = None
        if cross_class_iou_thr is not None and cross_class_iou_thr <= 0:
            cross_class_iou_thr = None
    fusion_mode = str(fusion_mode or "primary").strip().lower()
    use_wbf = fusion_mode == "wbf"

    def det_score(det: Dict[str, Any]) -> float:
        try:
            return float(det.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _normalized_source(value: Any) -> str:
        return str(value or "").strip().lower()

    def _score_by_source(det: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        raw_map = det.get("score_by_source")
        if isinstance(raw_map, Mapping):
            for raw_src, raw_score in raw_map.items():
                src = _normalized_source(raw_src)
                if not src:
                    continue
                try:
                    score_val = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if src not in out or score_val > out[src]:
                    out[src] = score_val
        primary = _normalized_source(det.get("score_source") or det.get("source"))
        raw_primary_score = det.get("score")
        if primary and raw_primary_score is not None:
            try:
                primary_score = float(raw_primary_score)
            except (TypeError, ValueError):
                primary_score = None
            if primary_score is not None and (primary not in out or primary_score > out[primary]):
                out[primary] = primary_score
        return out

    def _wbf_init(entry: Dict[str, Any]) -> None:
        if not use_wbf:
            return
        bbox = entry.get("bbox_xyxy_px")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return
        weight = det_score(entry)
        if weight <= 0:
            weight = 1e-6
        entry["_wbf_weight"] = float(weight)
        entry["_wbf_bbox_sum"] = [float(bbox[i]) * float(weight) for i in range(4)]

    def _wbf_merge(keep_det: Dict[str, Any], det_a: Dict[str, Any], det_b: Dict[str, Any]) -> None:
        if not use_wbf:
            return
        sum_a = det_a.get("_wbf_bbox_sum")
        weight_a = det_a.get("_wbf_weight")
        if not isinstance(sum_a, list) or len(sum_a) < 4 or not isinstance(weight_a, (int, float)):
            bbox_a = det_a.get("bbox_xyxy_px")
            if not isinstance(bbox_a, (list, tuple)) or len(bbox_a) < 4:
                return
            weight_a = det_score(det_a)
            if weight_a <= 0:
                weight_a = 1e-6
            sum_a = [float(bbox_a[i]) * float(weight_a) for i in range(4)]
        sum_b = det_b.get("_wbf_bbox_sum")
        weight_b = det_b.get("_wbf_weight")
        if not isinstance(sum_b, list) or len(sum_b) < 4 or not isinstance(weight_b, (int, float)):
            bbox_b = det_b.get("bbox_xyxy_px")
            if not isinstance(bbox_b, (list, tuple)) or len(bbox_b) < 4:
                return
            weight_b = det_score(det_b)
            if weight_b <= 0:
                weight_b = 1e-6
            sum_b = [float(bbox_b[i]) * float(weight_b) for i in range(4)]
        merged_weight = float(weight_a) + float(weight_b)
        if merged_weight <= 0:
            return
        merged_sum = [float(sum_a[i]) + float(sum_b[i]) for i in range(4)]
        keep_det["_wbf_weight"] = merged_weight
        keep_det["_wbf_bbox_sum"] = merged_sum
        keep_det["bbox_xyxy_px"] = [val / merged_weight for val in merged_sum]

    merged: List[Dict[str, Any]] = []
    removed = 0
    ordered = sorted(detections, key=det_score, reverse=True)
    for det in ordered:
        label = str(det.get("label") or det.get("class_name") or "").strip()
        box = det.get("bbox_xyxy_px")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            merged.append(det)
            continue
        matched_idx = None
        for idx, kept in enumerate(merged):
            kept_label = str(kept.get("label") or kept.get("class_name") or "").strip()
            merge_iou_thr = iou_thr
            if label and kept_label and label != kept_label:
                if cross_class_iou_thr is None:
                    continue
                merge_iou_thr = cross_class_iou_thr
            kept_box = kept.get("bbox_xyxy_px")
            if not isinstance(kept_box, (list, tuple)) or len(kept_box) < 4:
                continue
            if _agent_iou_xyxy(box, kept_box) >= merge_iou_thr:
                matched_idx = idx
                break
        if matched_idx is None:
            entry = dict(det)
            atom_ids = _agent_extract_atom_ids(entry)
            source_list = set(entry.get("source_list") or [])
            if entry.get("source"):
                source_list.add(entry.get("source"))
            score_map = _score_by_source(entry)
            source_list.update(score_map.keys())
            if source_list:
                entry["source_list"] = sorted(source_list)
            if score_map:
                entry["score_by_source"] = score_map
            if atom_ids:
                entry["prepass_atom_ids"] = atom_ids
            _wbf_init(entry)
            merged.append(entry)
        else:
            kept = merged[matched_idx]
            merged_atom_ids = _agent_extract_atom_ids(kept)
            for atom_id in _agent_extract_atom_ids(det):
                if atom_id not in merged_atom_ids:
                    merged_atom_ids.append(atom_id)
            source_list = set(kept.get("source_list") or [])
            if kept.get("source"):
                source_list.add(kept.get("source"))
            if det.get("source"):
                source_list.add(det.get("source"))
            kept_score_map = _score_by_source(kept)
            det_score_map = _score_by_source(det)
            merged_score_map = dict(kept_score_map)
            for src, src_score in det_score_map.items():
                if src not in merged_score_map or src_score > merged_score_map[src]:
                    merged_score_map[src] = src_score
            source_list.update(merged_score_map.keys())
            keep_det = dict(kept)
            if det_score(det) > det_score(kept):
                keep_det = dict(det)
            if source_list:
                keep_det["source_list"] = sorted(source_list)
            if merged_score_map:
                keep_det["score_by_source"] = merged_score_map
            if merged_atom_ids:
                keep_det["prepass_atom_ids"] = merged_atom_ids
            _wbf_merge(keep_det, kept, det)
            merged[matched_idx] = keep_det
            removed += 1
    if use_wbf:
        for det in merged:
            if not isinstance(det, dict):
                continue
            det.pop("_wbf_weight", None)
            det.pop("_wbf_bbox_sum", None)
    return merged, removed


def _agent_filter_scoreless_detections(
    detections: List[Dict[str, Any]],
    *,
    iou_thr: float,
) -> Tuple[List[Dict[str, Any]], int]:
    if not detections or iou_thr <= 0:
        return detections, 0
    anchors = [
        det
        for det in detections
        if det.get("score") is not None and (det.get("score_source") or det.get("source")) != "unknown"
    ]
    if not anchors:
        return detections, 0
    filtered: List[Dict[str, Any]] = []
    removed = 0
    for det in detections:
        score = det.get("score")
        score_source = det.get("score_source") or det.get("source") or "unknown"
        if score is None or score_source == "unknown":
            bbox = det.get("bbox_xyxy_px") or []
            has_overlap = False
            for anchor in anchors:
                anchor_bbox = anchor.get("bbox_xyxy_px") or []
                if _agent_iou_xyxy(bbox, anchor_bbox) >= iou_thr:
                    has_overlap = True
                    break
            if not has_overlap:
                removed += 1
                continue
        filtered.append(det)
    return filtered, removed


def _agent_det_score(det: Dict[str, Any]) -> Optional[float]:
    raw = det.get("score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _agent_float_with_default(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


def _agent_cluster_match(
    det: Dict[str, Any],
    clusters: Sequence[Dict[str, Any]],
    *,
    iou_thr: float,
) -> Optional[Dict[str, Any]]:
    label = str(det.get("label") or "").strip()
    bbox = det.get("bbox_xyxy_px")
    if not label or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_label = str(cluster.get("label") or "").strip()
        if cluster_label and cluster_label != label:
            continue
        cluster_bbox = cluster.get("bbox_xyxy_px")
        if not isinstance(cluster_bbox, (list, tuple)) or len(cluster_bbox) < 4:
            continue
        if _agent_iou_xyxy(bbox, cluster_bbox) >= iou_thr:
            return cluster
    return None


def _agent_deep_prepass_cleanup_impl(
    payload: Any,
    *,
    detections: List[Dict[str, Any]],
    pil_img: Any,
    labelmap: List[str],
    resolve_classifier_path_fn: Any,
    load_classifier_head_fn: Any,
    active_classifier_head: Any,
    background_from_head_fn: Any,
    sanitize_fn: Any,
    default_iou: float,
) -> Dict[str, Any]:
    img_w, img_h = pil_img.size
    iou_thr = _agent_float_with_default(getattr(payload, "iou", None), default_iou)
    iou_thr = max(0.0, min(1.0, iou_thr))
    cross_class_enabled = bool(getattr(payload, "cross_class_dedupe_enabled", False))
    cross_class_iou_thr = None
    if cross_class_enabled:
        raw_cross_class_iou = getattr(payload, "cross_class_dedupe_iou", None)
        try:
            cross_class_iou_thr = float(raw_cross_class_iou if raw_cross_class_iou is not None else 0.8)
        except (TypeError, ValueError):
            cross_class_iou_thr = 0.8
        cross_class_iou_thr = max(0.0, min(1.0, cross_class_iou_thr))
    merged, removed = _agent_merge_prepass_detections(
        detections,
        iou_thr=iou_thr,
        cross_class_iou_thr=cross_class_iou_thr,
        fusion_mode=str(getattr(payload, "fusion_mode", None) or "primary"),
    )
    scoreless_iou = _agent_float_with_default(getattr(payload, "scoreless_iou", None), 0.0)
    scoreless_iou = max(0.0, min(1.0, scoreless_iou))
    if scoreless_iou > 0.0:
        merged, scoreless_removed = _agent_filter_scoreless_detections(
            merged,
            iou_thr=float(scoreless_iou),
        )
    else:
        scoreless_removed = 0
    head: Optional[Dict[str, Any]] = None
    if not getattr(payload, "prepass_keep_all", False):
        classifier_id = getattr(payload, "classifier_id", None)
        if classifier_id:
            classifier_path = resolve_classifier_path_fn(classifier_id)
            if classifier_path is not None:
                head = load_classifier_head_fn(classifier_path)
        elif isinstance(active_classifier_head, dict):
            head = dict(active_classifier_head)
    background = background_from_head_fn(head)
    cleaned, rejected = sanitize_fn(
        merged,
        pil_img=pil_img,
        classifier_head=head,
        img_w=img_w,
        img_h=img_h,
        labelmap=labelmap,
        background=background,
    )
    return {
        "detections": cleaned,
        "removed": removed,
        "scoreless_removed": scoreless_removed,
        "rejected": rejected,
    }


def _agent_run_deep_prepass_part_a_impl(
    payload: Any,
    *,
    pil_img: Any,
    image_token: str,
    labelmap: List[str],
    glossary: str,
    run_detector_fn: Any,
    attach_provenance_fn: Any,
    generate_sam3_synonyms_fn: Any,
    generate_text_fn: Any,
    extract_json_fn: Any,
    default_synonyms: Any,
    label_key_fn: Any,
    sam3_text_windows_fn: Any,
    ensure_sam3_text_runtime_fn: Any,
    normalize_window_xyxy_fn: Any,
    sam3_prompt_variants_fn: Any,
    sam3_text_payloads_fn: Any,
    trace_writer: Optional[Any] = None,
    trace_full_writer: Optional[Any] = None,
    trace_readable: Optional[Any] = None,
    active_sam3_score_thr: Optional[float] = None,
    active_sam3_mask_thr: Optional[float] = None,
    grid_overlap_ratio_default: float = 0.0,
) -> Dict[str, Any]:
    img_w, img_h = pil_img.size
    detections: List[Dict[str, Any]] = []
    warnings: List[str] = []
    sam3_text_prompts: Dict[str, List[str]] = {}
    sam3_text_term_map: Dict[str, Dict[str, List[str]]] = {}

    def _log_step(event: str, payload_obj: Dict[str, Any]) -> None:
        if trace_writer:
            trace_writer({"type": event, **payload_obj, "ts": time.time()})
        if trace_full_writer:
            trace_full_writer({"type": event, **payload_obj, "ts": time.time()})

    raw_slice_size = getattr(payload, "sahi_window_size", None)
    try:
        slice_size = int(raw_slice_size) if raw_slice_size is not None else 640
    except (TypeError, ValueError):
        slice_size = 640
    if slice_size <= 0:
        slice_size = 640
    raw_overlap = getattr(payload, "sahi_overlap_ratio", None)
    overlap = _agent_float_with_default(raw_overlap, 0.2)
    if overlap <= 0.0 or overlap >= 1.0:
        overlap = 0.2
    sahi_cfg = {
        "enabled": True,
        "slice_size": slice_size,
        "overlap": overlap,
    }
    full_cfg = {"enabled": False}

    detector_conf = getattr(payload, "detector_conf", None)
    detector_iou = getattr(payload, "detector_iou", None)
    detector_merge_iou = getattr(payload, "detector_merge_iou", None)
    max_det = None

    for mode in ("yolo", "rfdetr"):
        if mode == "yolo" and getattr(payload, "enable_yolo", True) is False:
            continue
        if mode == "rfdetr" and getattr(payload, "enable_rfdetr", True) is False:
            continue
        det_id = getattr(payload, "yolo_id", None) if mode == "yolo" else getattr(payload, "rfdetr_id", None)
        if not det_id:
            if (getattr(payload, "detector_mode", None) or "yolo") == mode:
                det_id = getattr(payload, "detector_id", None)
        for run_name, run_sahi in (("full", full_cfg), ("sahi", sahi_cfg)):
            args = {
                "image_token": image_token,
                "detector_id": det_id,
                "mode": mode,
                "conf": detector_conf,
                "sahi": run_sahi,
                "max_det": max_det,
                "iou": detector_iou,
                "merge_iou": detector_merge_iou,
                "expected_labelmap": labelmap or None,
            }
            if trace_readable:
                if run_name == "sahi":
                    trace_readable(
                        f"deep_prepass detector:{mode} start sahi="
                        f"{sahi_cfg.get('slice_size')}x{sahi_cfg.get('slice_size')} "
                        f"overlap={sahi_cfg.get('overlap')}"
                    )
                else:
                    trace_readable(f"deep_prepass detector:{mode} start full")
            _log_step("deep_prepass_tool_call", {"tool": "run_detector", "args": args})
            try:
                result = run_detector_fn(
                    image_token=image_token,
                    detector_id=det_id,
                    mode=mode,
                    conf=detector_conf,
                    sahi=run_sahi,
                    max_det=max_det,
                    iou=detector_iou,
                    merge_iou=detector_merge_iou,
                    expected_labelmap=labelmap or None,
                    register=False,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"deep_prepass_detector_failed:{mode}:{run_name}:{exc}")
                continue
            _log_step("deep_prepass_tool_result", {"tool": "run_detector", "result": result})
            dets = list(result.get("detections") or [])
            if trace_readable:
                trace_readable(f"deep_prepass detector:{mode} {run_name} detections={len(dets)}")
            run_id = det_id
            if run_name:
                run_id = f"{det_id or mode}:{run_name}"
            attach_provenance_fn(
                dets,
                source=mode,
                source_primary=mode,
                source_detector_run_id=run_id,
            )
            detections.extend(dets)

    if getattr(payload, "enable_sam3_text", True) is not False and labelmap:
        labels = [lbl for lbl in labelmap if str(lbl).strip()]
        prompt_budget = getattr(payload, "sam3_text_synonym_budget", None)
        if prompt_budget is not None:
            prompt_budget = int(prompt_budget)
        synonym_map, term_meta = generate_sam3_synonyms_fn(
            labels,
            glossary or "",
            max_synonyms=prompt_budget,
            generate_text_fn=generate_text_fn,
            extract_json_fn=extract_json_fn,
            default_synonyms=default_synonyms,
            label_key_fn=label_key_fn,
        )
        sam3_text_term_map = term_meta
        score_thr = getattr(payload, "prepass_sam3_text_thr", None)
        if score_thr is None and active_sam3_score_thr is not None:
            score_thr = active_sam3_score_thr
        mask_thr = getattr(payload, "sam3_mask_threshold", None)
        if mask_thr is None and active_sam3_mask_thr is not None:
            mask_thr = active_sam3_mask_thr
        windowed = getattr(payload, "sam3_text_window_extension", True) is not False
        windows: List[Dict[str, Any]] = []
        if windowed:
            windows = sam3_text_windows_fn(
                payload,
                pil_img=pil_img,
                grid_overlap_ratio_default=grid_overlap_ratio_default,
            )
        _, sam3_processor, _ = ensure_sam3_text_runtime_fn()
        global_state = None
        try:
            global_state = sam3_processor.set_image(pil_img)
        except Exception:
            global_state = None
        if windows:
            for window in windows:
                window_xyxy = None
                if window.get("bbox_xyxy_px"):
                    window_xyxy = window.get("bbox_xyxy_px")
                elif window.get("bbox_2d") is not None:
                    window_xyxy = normalize_window_xyxy_fn(
                        {"bbox_2d": window.get("bbox_2d")}, img_w, img_h
                    )
                if not window_xyxy:
                    continue
                x1, y1, x2, y2 = window_xyxy
                crop_img = pil_img.crop((x1, y1, x2, y2))
                try:
                    window_state = sam3_processor.set_image(crop_img)
                except Exception:
                    window_state = None
                window["window_xyxy"] = window_xyxy
                window["crop_img"] = crop_img
                window["state"] = window_state
        prompt_plan: List[Tuple[str, str, str]] = []
        for label in labels:
            base_terms = term_meta.get(label, {}).get("base_terms", [])
            expanded_terms = term_meta.get(label, {}).get("expanded_terms", [])
            max_prompts = max(2, len(synonym_map.get(label, [])) + 2)
            prompts = sam3_prompt_variants_fn(
                label,
                synonym_map,
                max_prompts=max_prompts,
                default_synonyms=default_synonyms,
                label_key_fn=label_key_fn,
            )
            if not prompts:
                prompts = [str(label).replace("_", " ").strip() or str(label).strip()]
            for prompt in prompts:
                prompt_origin = "base"
                if prompt in expanded_terms:
                    prompt_origin = "expanded"
                elif prompt not in base_terms:
                    prompt_origin = "unknown"
                sam3_text_prompts.setdefault(label, []).append(prompt)
                prompt_plan.append((label, prompt, prompt_origin))

        contexts: List[Dict[str, Any]] = [
            {
                "name": "full",
                "grid_cell": None,
                "window_xyxy": None,
                "window_bbox_2d": None,
                "crop_img": pil_img,
                "state": global_state,
            }
        ]
        for window in windows:
            contexts.append(
                {
                    "name": window.get("name") or window.get("grid_cell") or "window",
                    "grid_cell": window.get("grid_cell"),
                    "window_xyxy": window.get("window_xyxy"),
                    "window_bbox_2d": window.get("bbox_2d"),
                    "crop_img": window.get("crop_img") or pil_img,
                    "state": window.get("state"),
                }
            )

        for ctx in contexts:
            window_name = ctx.get("name") or "window"
            for label, prompt, prompt_origin in prompt_plan:
                args = {
                    "image_token": image_token,
                    "prompt": prompt,
                    "label": label,
                    "score_thr": score_thr,
                    "mask_threshold": mask_thr,
                    "max_results": None,
                }
                if ctx.get("window_bbox_2d") is not None:
                    args["window_bbox_2d"] = ctx.get("window_bbox_2d")
                if ctx.get("grid_cell"):
                    args["grid_cell"] = ctx.get("grid_cell")
                _log_step("deep_prepass_tool_call", {"tool": "sam3_text", "args": args})
                try:
                    dets, assigned_label, _ = sam3_text_payloads_fn(
                        full_img=pil_img,
                        crop_img=ctx.get("crop_img") or pil_img,
                        prompt=prompt,
                        label=label,
                        score_thr=score_thr,
                        mask_threshold=mask_thr,
                        max_results=None,
                        window_xyxy=ctx.get("window_xyxy"),
                        processor_override=sam3_processor,
                        state=ctx.get("state"),
                    )
                except Exception as exc:  # noqa: BLE001
                    if ctx.get("window_xyxy"):
                        warnings.append(f"deep_prepass_sam3_text_failed:{label}:{window_name}:{exc}")
                    else:
                        warnings.append(f"deep_prepass_sam3_text_failed:{label}:{exc}")
                    continue
                _log_step("deep_prepass_tool_result", {"tool": "sam3_text", "result": {"detections": dets}})
                for det in dets:
                    if not isinstance(det, dict):
                        continue
                    det["sam3_prompt_term"] = prompt
                    det["sam3_prompt_label"] = assigned_label or label
                    det["sam3_prompt_source"] = prompt_origin
                if trace_readable:
                    if ctx.get("window_xyxy"):
                        trace_readable(
                            f"deep_prepass sam3_text label={label} prompt={prompt} window={window_name} detections={len(dets)}"
                        )
                    else:
                        trace_readable(
                            f"deep_prepass sam3_text label={label} prompt={prompt} detections={len(dets)}"
                        )
                attach_provenance_fn(
                    dets,
                    source="sam3_text",
                    source_primary="sam3_text",
                    source_prompt=prompt,
                )
                detections.extend(dets)

    return {
        "detections": detections,
        "warnings": warnings,
        "sam3_text_prompts": sam3_text_prompts,
        "sam3_text_term_map": sam3_text_term_map,
        "image_size": {"width": img_w, "height": img_h},
    }


def _agent_run_deep_prepass_impl(
    payload: Any,
    *,
    pil_img: Any,
    image_token: str,
    labelmap: List[str],
    glossary: str,
    run_part_a_fn: Any,
    cleanup_fn: Any,
    select_exemplars_fn: Any,
    run_similarity_global_fn: Any,
    run_similarity_windowed_fn: Any,
    finalize_provenance_fn: Any,
    trace_writer: Optional[Any] = None,
    trace_full_writer: Optional[Any] = None,
    trace_readable: Optional[Any] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    provenance_atoms: List[Dict[str, Any]] = []
    provenance_stage_atoms: Dict[str, Dict[str, List[str]]] = {}
    next_atom_index = 1
    part_a = run_part_a_fn(
        payload,
        pil_img=pil_img,
        image_token=image_token,
        labelmap=labelmap,
        glossary=glossary,
        trace_writer=trace_writer,
        trace_full_writer=trace_full_writer,
        trace_readable=trace_readable,
    )
    detections = list(part_a.get("detections") or [])
    next_atom_index = _agent_register_provenance_atoms(
        detections,
        next_atom_index=next_atom_index,
        atoms=provenance_atoms,
        stage_atoms=provenance_stage_atoms,
    )
    warnings.extend(list(part_a.get("warnings") or []))
    if trace_readable:
        trace_readable(f"deep prepass A: detections={len(detections)}")
    cleanup_a = cleanup_fn(
        payload,
        detections=detections,
        pil_img=pil_img,
        labelmap=labelmap,
    )
    detections = list(cleanup_a.get("detections") or [])
    removed = int(cleanup_a.get("removed") or 0)
    scoreless_removed = int(cleanup_a.get("scoreless_removed") or 0)
    rejected = int(cleanup_a.get("rejected") or 0)
    if trace_readable:
        trace_readable(
            "deep prepass A cleanup: "
            f"cleaned={len(detections)} removed={removed} "
            f"scoreless_removed={scoreless_removed} rejected={rejected}"
        )
    added_similarity = 0
    if getattr(payload, "enable_sam3_similarity", True) is not False and detections:
        exemplars_by_label = select_exemplars_fn(
            payload,
            detections=detections,
            trace_readable=trace_readable,
        )
        similarity_detections: List[Dict[str, Any]] = []
        if exemplars_by_label:
            global_result = run_similarity_global_fn(
                payload,
                pil_img=pil_img,
                image_token=image_token,
                exemplars_by_label=exemplars_by_label,
                trace_writer=trace_writer,
                trace_full_writer=trace_full_writer,
                trace_readable=trace_readable,
            )
            similarity_detections.extend(list(global_result.get("detections") or []))
            warnings.extend(list(global_result.get("warnings") or []))
            if getattr(payload, "similarity_window_extension", False):
                window_result = run_similarity_windowed_fn(
                    payload,
                    pil_img=pil_img,
                    image_token=image_token,
                    exemplars_by_label=exemplars_by_label,
                    trace_writer=trace_writer,
                    trace_full_writer=trace_full_writer,
                    trace_readable=trace_readable,
                )
                similarity_detections.extend(list(window_result.get("detections") or []))
                warnings.extend(list(window_result.get("warnings") or []))
        added_similarity = len(similarity_detections)
        if similarity_detections:
            next_atom_index = _agent_register_provenance_atoms(
                similarity_detections,
                next_atom_index=next_atom_index,
                atoms=provenance_atoms,
                stage_atoms=provenance_stage_atoms,
            )
            detections = detections + similarity_detections
            cleanup_b = cleanup_fn(
                payload,
                detections=detections,
                pil_img=pil_img,
                labelmap=labelmap,
            )
            detections = list(cleanup_b.get("detections") or [])
            removed_b = int(cleanup_b.get("removed") or 0)
            scoreless_removed_b = int(cleanup_b.get("scoreless_removed") or 0)
            rejected_b = int(cleanup_b.get("rejected") or 0)
            if trace_readable:
                trace_readable(
                    "deep prepass B: "
                    f"added_similarity={added_similarity} cleaned={len(detections)} "
                    f"removed={removed_b} scoreless_removed={scoreless_removed_b} rejected={rejected_b}"
                )
    finalize_provenance_fn(detections)
    final_clusters = _agent_build_final_cluster_provenance(detections)
    return {
        "detections": detections,
        "warnings": warnings,
        "sam3_text_prompts": part_a.get("sam3_text_prompts") or {},
        "sam3_text_term_map": part_a.get("sam3_text_term_map") or {},
        "image_size": part_a.get("image_size") or {},
        "provenance": {
            "schema_version": 1,
            "atoms": provenance_atoms,
            "stage_atoms": provenance_stage_atoms,
            "final_clusters": final_clusters,
        },
    }


def _agent_run_deep_prepass_caption_impl(
    payload: Any,
    *,
    pil_img: Any,
    image_token: str,
    detections: List[Dict[str, Any]],
    model_id_override: Optional[str],
    glossary: Optional[str],
    grid_for_log: Optional[Dict[str, Any]],
    caption_request_cls: Any,
    qwen_caption_fn: Any,
    sanitize_caption_fn: Any,
    label_counts_fn: Any,
    qwen_bbox_to_xyxy_fn: Any,
    xyxy_to_bbox_fn: Any,
    grid_cell_for_window_bbox_fn: Any,
    readable_format_bbox_fn: Any,
    unload_non_qwen_fn: Any,
    caption_window_hook: Any,
    http_exception_cls: Any,
    http_503_code: int,
    trace_writer: Optional[Any] = None,
    trace_full_writer: Optional[Any] = None,
    trace_readable: Optional[Any] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    if not getattr(payload, "prepass_caption", False):
        return "", []
    caption_hints: List[Any] = []
    hint_items = list(detections or [])
    hint_items.sort(key=lambda det: float(det.get("score") or 0.0), reverse=True)
    for det in hint_items[: min(len(hint_items), 160)]:
        label = str(det.get("label") or det.get("class_name") or "").strip()
        if not label:
            continue
        bbox = det.get("bbox_xyxy_px")
        if not bbox and det.get("bbox_2d"):
            bbox = qwen_bbox_to_xyxy_fn(pil_img.width, pil_img.height, det.get("bbox_2d") or [])
        if not bbox or len(bbox) < 4:
            continue
        caption_hints.append(
            {
                "label": label,
                "bbox": [float(v) for v in bbox[:4]],
                "confidence": det.get("score"),
            }
        )
    det_hint_summary = label_counts_fn(hint_items, limit=10)
    prepass_prompt = (
        "Write a detailed, multi-sentence caption. Use detection hints as suggestions, "
        "but mention other visible objects. Preserve specific details you see (counts, actions, "
        "notable attributes). Do not mention labels, hints, or coordinates. "
        "Never output labelmap tags (e.g., light_vehicle); use natural words like car or van. "
        "Avoid any token with underscores."
    )
    if det_hint_summary and det_hint_summary != "none":
        prepass_prompt = f"{prepass_prompt} Detection hints: {det_hint_summary}."

    caption_profile = (getattr(payload, "prepass_caption_profile", None) or "light").strip().lower()
    if caption_profile not in {"light", "deep"}:
        caption_profile = "light"
    caption_variant = getattr(payload, "prepass_caption_variant", None) or getattr(payload, "model_variant", None) or "auto"
    caption_model_id = (getattr(payload, "prepass_caption_model_id", None) or model_id_override or "").strip() or None
    caption_max_tokens = int(getattr(payload, "prepass_caption_max_tokens", None) or (512 if caption_profile == "light" else 1024))
    caption_mode = "windowed"
    caption_all_windows = True
    include_coords = False
    caption_payload = caption_request_cls(
        image_token=image_token,
        user_prompt=prepass_prompt,
        label_hints=caption_hints or None,
        image_width=pil_img.width,
        image_height=pil_img.height,
        include_counts=True,
        include_coords=include_coords,
        max_boxes=min(len(caption_hints), 120),
        max_new_tokens=caption_max_tokens,
        model_variant=caption_variant,
        model_id=caption_model_id,
        final_answer_only=True,
        two_stage_refine=caption_mode == "windowed",
        caption_mode=caption_mode,
        caption_all_windows=caption_all_windows,
        restrict_to_labels=False,
        fast_mode=True,
        multi_model_cache=True,
        labelmap_glossary=glossary,
    )

    def _is_cuda_oom(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or ("cuda error" in msg and "memory" in msg)

    windowed_captions: List[Tuple[int, int, int, str]] = []

    def _window_hook(x0: int, y0: int, size: int, caption: str) -> None:
        windowed_captions.append((x0, y0, size, caption))
        if trace_readable:
            cell = None
            if grid_for_log:
                bbox_2d = xyxy_to_bbox_fn(
                    pil_img.width,
                    pil_img.height,
                    float(x0),
                    float(y0),
                    float(x0 + size),
                    float(y0 + size),
                )
                cell = grid_cell_for_window_bbox_fn(grid_for_log, bbox_2d)
            cell_text = f" {cell}" if cell else ""
            trace_readable(
                f"prepass caption window{cell_text} "
                f"[{x0},{y0},{x0 + size},{y0 + size}]: {caption}"
            )
        if trace_writer:
            trace_writer(
                {
                    "type": "prepass_caption_window",
                    "window": [int(x0), int(y0), int(size)],
                    "grid_cell": grid_cell_for_window_bbox_fn(
                        grid_for_log,
                        xyxy_to_bbox_fn(
                            pil_img.width,
                            pil_img.height,
                            float(x0),
                            float(y0),
                            float(x0 + size),
                            float(y0 + size),
                        ),
                    )
                    if grid_for_log
                    else None,
                    "caption": caption,
                    "ts": time.time(),
                }
            )
        if trace_full_writer:
            trace_full_writer(
                {
                    "type": "prepass_caption_window",
                    "window": [int(x0), int(y0), int(size)],
                    "grid_cell": grid_cell_for_window_bbox_fn(
                        grid_for_log,
                        xyxy_to_bbox_fn(
                            pil_img.width,
                            pil_img.height,
                            float(x0),
                            float(y0),
                            float(x0 + size),
                            float(y0 + size),
                        ),
                    )
                    if grid_for_log
                    else None,
                    "caption": caption,
                    "ts": time.time(),
                }
            )

    def _run_caption_with_hook(request: Any) -> Any:
        token = None
        if caption_mode == "windowed":
            token = caption_window_hook.set(_window_hook)
        try:
            if trace_full_writer:
                trace_full_writer(
                    {
                        "type": "prepass_caption_call",
                        "payload": {
                            "model_id": request.model_id,
                            "variant": request.model_variant,
                            "caption_mode": request.caption_mode,
                            "caption_all_windows": request.caption_all_windows,
                            "restrict_to_labels": request.restrict_to_labels,
                            "max_new_tokens": request.max_new_tokens,
                            "hint_count": len(request.label_hints or []),
                        },
                        "ts": time.time(),
                    }
                )
            return qwen_caption_fn(request)
        finally:
            if token is not None:
                caption_window_hook.reset(token)

    try:
        response = _run_caption_with_hook(caption_payload)
    except Exception as exc:  # noqa: BLE001
        if _is_cuda_oom(exc):
            try:
                unload_non_qwen_fn()
                response = _run_caption_with_hook(caption_payload)
            except Exception as retry_exc:  # noqa: BLE001
                detail = str(retry_exc)
                raise http_exception_cls(
                    status_code=http_503_code,
                    detail=f"prepass_caption_failed:{detail}",
                ) from retry_exc
        else:
            detail = str(exc)
            raise http_exception_cls(
                status_code=http_503_code,
                detail=f"prepass_caption_failed:{detail}",
            ) from exc
    caption_text = sanitize_caption_fn(response.caption or "")
    if not caption_text:
        raise http_exception_cls(status_code=http_503_code, detail="prepass_caption_failed:empty")
    if trace_readable:
        trace_readable(f"prepass caption summary: {caption_text}")
    caption_entries: List[Dict[str, Any]] = []
    if windowed_captions:
        for x0, y0, size, caption in windowed_captions:
            bbox_2d = xyxy_to_bbox_fn(
                pil_img.width,
                pil_img.height,
                float(x0),
                float(y0),
                float(x0 + size),
                float(y0 + size),
            )
            window_name = None
            if grid_for_log:
                window_name = grid_cell_for_window_bbox_fn(grid_for_log, bbox_2d)
            if not window_name:
                window_name = f"window_{int(x0)}_{int(y0)}"
            caption_entries.append(
                {
                    "window": window_name,
                    "bbox_2d": list(bbox_2d),
                    "caption": caption,
                }
            )
            if trace_full_writer:
                trace_full_writer(
                    {
                        "type": "prepass_caption_result",
                        "window": {"name": window_name, "bbox_2d": list(bbox_2d)},
                        "caption": caption,
                        "ts": time.time(),
                    }
                )
            if trace_readable:
                coords = readable_format_bbox_fn(bbox_2d)
                trace_readable(f"prepass caption {window_name} {coords}: {caption}")
    return caption_text, caption_entries


def _build_deep_prepass_runners_impl(
    *,
    run_detector_fn: Any,
    attach_provenance_fn: Any,
    generate_sam3_synonyms_fn: Any,
    generate_text_fn: Any,
    extract_json_fn: Any,
    default_synonyms: Any,
    label_key_fn: Any,
    sam3_text_windows_fn: Any,
    ensure_sam3_text_runtime_fn: Any,
    normalize_window_xyxy_fn: Any,
    sam3_prompt_variants_fn: Any,
    sam3_text_payloads_fn: Any,
    active_sam3_score_thr: float,
    active_sam3_mask_thr: float,
    grid_overlap_ratio_default: float,
    resolve_classifier_path_fn: Any,
    load_classifier_head_fn: Any,
    active_classifier_head: Any,
    background_from_head_fn: Any,
    sanitize_fn: Any,
    default_iou: float,
    select_exemplars_fn: Any,
    run_similarity_global_fn: Any,
    run_similarity_windowed_fn: Any,
    finalize_provenance_fn: Any,
    caption_request_cls: Any,
    qwen_caption_fn: Any,
    sanitize_caption_fn: Any,
    label_counts_fn: Any,
    qwen_bbox_to_xyxy_fn: Any,
    xyxy_to_bbox_fn: Any,
    grid_cell_for_window_bbox_fn: Any,
    readable_format_bbox_fn: Any,
    unload_non_qwen_fn: Any,
    caption_window_hook: Any,
    http_exception_cls: Any,
    http_503_code: int,
) -> Tuple[
    Callable[..., Dict[str, Any]],
    Callable[..., Dict[str, Any]],
    Callable[..., Dict[str, Any]],
    Callable[..., Tuple[str, List[Dict[str, Any]]]],
]:
    def run_part_a(
        payload: Any,
        *,
        pil_img: Any,
        image_token: str,
        labelmap: List[str],
        glossary: str,
        trace_writer: Optional[Any] = None,
        trace_full_writer: Optional[Any] = None,
        trace_readable: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return _agent_run_deep_prepass_part_a_impl(
            payload,
            pil_img=pil_img,
            image_token=image_token,
            labelmap=labelmap,
            glossary=glossary,
            run_detector_fn=run_detector_fn,
            attach_provenance_fn=attach_provenance_fn,
            generate_sam3_synonyms_fn=generate_sam3_synonyms_fn,
            generate_text_fn=generate_text_fn,
            extract_json_fn=extract_json_fn,
            default_synonyms=default_synonyms,
            label_key_fn=label_key_fn,
            sam3_text_windows_fn=sam3_text_windows_fn,
            ensure_sam3_text_runtime_fn=ensure_sam3_text_runtime_fn,
            normalize_window_xyxy_fn=normalize_window_xyxy_fn,
            sam3_prompt_variants_fn=sam3_prompt_variants_fn,
            sam3_text_payloads_fn=sam3_text_payloads_fn,
            trace_writer=trace_writer,
            trace_full_writer=trace_full_writer,
            trace_readable=trace_readable,
            active_sam3_score_thr=active_sam3_score_thr,
            active_sam3_mask_thr=active_sam3_mask_thr,
            grid_overlap_ratio_default=grid_overlap_ratio_default,
        )

    def cleanup(
        payload: Any,
        *,
        detections: List[Dict[str, Any]],
        pil_img: Any,
        labelmap: List[str],
    ) -> Dict[str, Any]:
        return _agent_deep_prepass_cleanup_impl(
            payload,
            detections=detections,
            pil_img=pil_img,
            labelmap=labelmap,
            resolve_classifier_path_fn=resolve_classifier_path_fn,
            load_classifier_head_fn=load_classifier_head_fn,
            active_classifier_head=active_classifier_head,
            background_from_head_fn=background_from_head_fn,
            sanitize_fn=sanitize_fn,
            default_iou=default_iou,
        )

    def run(
        payload: Any,
        *,
        pil_img: Any,
        image_token: str,
        labelmap: List[str],
        glossary: str,
        trace_writer: Optional[Any] = None,
        trace_full_writer: Optional[Any] = None,
        trace_readable: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return _agent_run_deep_prepass_impl(
            payload,
            pil_img=pil_img,
            image_token=image_token,
            labelmap=labelmap,
            glossary=glossary,
            run_part_a_fn=run_part_a,
            cleanup_fn=cleanup,
            select_exemplars_fn=select_exemplars_fn,
            run_similarity_global_fn=run_similarity_global_fn,
            run_similarity_windowed_fn=run_similarity_windowed_fn,
            finalize_provenance_fn=finalize_provenance_fn,
            trace_writer=trace_writer,
            trace_full_writer=trace_full_writer,
            trace_readable=trace_readable,
        )

    def caption(
        payload: Any,
        *,
        pil_img: Any,
        image_token: str,
        detections: List[Dict[str, Any]],
        model_id_override: Optional[str],
        glossary: Optional[str] = None,
        grid_for_log: Optional[Dict[str, Any]] = None,
        trace_writer: Optional[Any] = None,
        trace_full_writer: Optional[Any] = None,
        trace_readable: Optional[Any] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        return _agent_run_deep_prepass_caption_impl(
            payload,
            pil_img=pil_img,
            image_token=image_token,
            detections=detections,
            model_id_override=model_id_override,
            glossary=glossary,
            grid_for_log=grid_for_log,
            caption_request_cls=caption_request_cls,
            qwen_caption_fn=qwen_caption_fn,
            sanitize_caption_fn=sanitize_caption_fn,
            label_counts_fn=label_counts_fn,
            qwen_bbox_to_xyxy_fn=qwen_bbox_to_xyxy_fn,
            xyxy_to_bbox_fn=xyxy_to_bbox_fn,
            grid_cell_for_window_bbox_fn=grid_cell_for_window_bbox_fn,
            readable_format_bbox_fn=readable_format_bbox_fn,
            unload_non_qwen_fn=unload_non_qwen_fn,
            caption_window_hook=caption_window_hook,
            http_exception_cls=http_exception_cls,
            http_503_code=http_503_code,
            trace_writer=trace_writer,
            trace_full_writer=trace_full_writer,
            trace_readable=trace_readable,
        )

    return run_part_a, cleanup, run, caption


def _agent_select_similarity_exemplars(
    min_score: float,
    *,
    detections: List[Dict[str, Any]],
    max_per_label: Optional[int] = None,
    strategy: Optional[str] = None,
    seed: Optional[int] = None,
    exemplar_fraction: Optional[float] = None,
    exemplar_min: Optional[int] = None,
    exemplar_max: Optional[int] = None,
    source_quota: Optional[int] = None,
    trace_readable: Optional[Any] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    try:
        min_score = float(min_score)
    except (TypeError, ValueError):
        min_score = 0.6
    if not math.isfinite(min_score):
        min_score = 0.6
    min_score = max(0.0, min(1.0, min_score))
    max_count = int(max_per_label or 3)
    if max_count < 1:
        max_count = 1
    strategy_norm = str(strategy or "top").strip().lower()
    if strategy_norm not in {"top", "random", "diverse"}:
        strategy_norm = "top"
    diverse_fraction = float(exemplar_fraction or 0.2)
    if not math.isfinite(diverse_fraction) or diverse_fraction <= 0:
        diverse_fraction = 0.2
    diverse_min = int(exemplar_min or 3)
    if diverse_min < 1:
        diverse_min = 1
    diverse_max = int(exemplar_max or 12)
    if diverse_max < diverse_min:
        diverse_max = diverse_min
    try:
        diverse_source_quota = int(source_quota) if source_quota is not None else 1
    except (TypeError, ValueError):
        diverse_source_quota = 1
    if diverse_source_quota < 0:
        diverse_source_quota = 0

    def _det_score(det: Dict[str, Any]) -> float:
        try:
            return float(det.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _det_source(det: Dict[str, Any]) -> str:
        return str(
            det.get("source_primary")
            or det.get("source")
            or det.get("score_source")
            or "unknown"
        ).strip().lower()

    def _det_center(det: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        bbox_xyxy = det.get("bbox_xyxy_px")
        if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) >= 4:
            try:
                x1 = float(bbox_xyxy[0])
                y1 = float(bbox_xyxy[1])
                x2 = float(bbox_xyxy[2])
                y2 = float(bbox_xyxy[3])
                return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
            except (TypeError, ValueError):
                return None
        bbox_2d = det.get("bbox_2d")
        if isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) >= 4:
            try:
                x = float(bbox_2d[0])
                y = float(bbox_2d[1])
                w = float(bbox_2d[2])
                h = float(bbox_2d[3])
                return (x + 0.5 * w, y + 0.5 * h)
            except (TypeError, ValueError):
                return None
        return None

    def _det_identity_signature(det: Dict[str, Any]) -> str:
        handle = str(det.get("handle") or "").strip()
        source = _det_source(det)
        atom_ids = det.get("prepass_atom_ids")
        atom_sig = ""
        if isinstance(atom_ids, (list, tuple)):
            cleaned = [str(value).strip() for value in atom_ids if str(value).strip()]
            if cleaned:
                atom_sig = ",".join(sorted(cleaned))
        bbox_sig = ""
        bbox_xyxy = det.get("bbox_xyxy_px")
        if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) >= 4:
            try:
                bbox_sig = ",".join(f"{float(value):.6f}" for value in bbox_xyxy[:4])
            except (TypeError, ValueError):
                bbox_sig = ""
        if not bbox_sig:
            bbox_2d = det.get("bbox_2d")
            if isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) >= 4:
                try:
                    bbox_sig = ",".join(f"{float(value):.6f}" for value in bbox_2d[:4])
                except (TypeError, ValueError):
                    bbox_sig = ""
        return "|".join([handle, source, atom_sig, bbox_sig])

    def _stable_tie_break(*, label: str, label_seed: int, det: Dict[str, Any]) -> float:
        token = _det_identity_signature(det)
        tie_input = f"{label_seed}:{label}:{token}".encode("utf-8", errors="ignore")
        digest = hashlib.sha1(tie_input).hexdigest()
        return int(digest, 16) / float((1 << 160) - 1)

    def _det_stable_key(det: Dict[str, Any]) -> str:
        try:
            normalized = json.dumps(det, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            normalized = _det_identity_signature(det)
        return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()

    def _select_diverse(
        label: str,
        eligible: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not eligible:
            return []
        base_seed = seed if seed is not None else 0
        seed_input = f"{base_seed}:{label}:diverse".encode("utf-8", errors="ignore")
        label_seed = int(hashlib.sha1(seed_input).hexdigest(), 16) % (2**32)
        target_from_fraction = int(math.ceil(len(eligible) * diverse_fraction))
        target_count = max(diverse_min, target_from_fraction)
        target_count = min(target_count, diverse_max, len(eligible))
        if target_count < 1:
            return []

        entries: List[Dict[str, Any]] = []
        for det in eligible:
            entries.append(
                {
                    "det": det,
                    "score": _det_score(det),
                    "source": _det_source(det),
                    "center": _det_center(det),
                    "tie_break": _stable_tie_break(label=label, label_seed=label_seed, det=det),
                    "stable_key": _det_stable_key(det),
                }
            )
        entries.sort(key=lambda entry: (-entry["score"], entry["tie_break"], entry["stable_key"]))
        max_score = max((entry["score"] for entry in entries), default=0.0)
        if max_score <= 0:
            max_score = 1.0
        centers = [entry["center"] for entry in entries if entry.get("center") is not None]
        if centers:
            xs = [float(pt[0]) for pt in centers]
            ys = [float(pt[1]) for pt in centers]
            diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if diag <= 1e-6:
                diag = 1.0
        else:
            diag = 1.0

        selected_indices: List[int] = []
        selected_index_set: set[int] = set()
        selected_source_counts: Dict[str, int] = {}
        selected_centers: List[Tuple[float, float]] = []

        def _candidate_value(entry: Dict[str, Any], *, quota_phase: bool) -> float:
            score_norm = max(0.0, min(1.0, float(entry["score"]) / max_score))
            center = entry.get("center")
            if not selected_centers:
                diversity = 1.0
            elif center is None:
                diversity = 0.0
            else:
                min_dist = min(
                    math.hypot(float(center[0]) - sx, float(center[1]) - sy)
                    for sx, sy in selected_centers
                )
                diversity = max(0.0, min(1.0, float(min_dist) / diag))
            source = str(entry.get("source") or "unknown")
            source_count = int(selected_source_counts.get(source, 0))
            source_bonus = 0.0
            if source_count == 0:
                source_bonus += 0.15
            if quota_phase and diverse_source_quota > 0 and source_count < diverse_source_quota:
                source_bonus += 0.20
            return 0.65 * score_norm + 0.35 * diversity + source_bonus

        def _pick_best(candidate_indices: List[int], *, quota_phase: bool) -> Optional[int]:
            best_idx: Optional[int] = None
            best_value = float("-inf")
            best_tie = float("-inf")
            best_stable_key = ""
            for idx in candidate_indices:
                if idx in selected_index_set:
                    continue
                entry = entries[idx]
                value = _candidate_value(entry, quota_phase=quota_phase)
                tie = float(entry["tie_break"])
                stable_key = str(entry.get("stable_key") or "")
                if value > best_value + 1e-12 or (
                    abs(value - best_value) <= 1e-12
                    and (
                        tie > best_tie + 1e-12
                        or (abs(tie - best_tie) <= 1e-12 and stable_key > best_stable_key)
                    )
                ):
                    best_idx = idx
                    best_value = value
                    best_tie = tie
                    best_stable_key = stable_key
            return best_idx

        source_to_indices: Dict[str, List[int]] = {}
        for idx, entry in enumerate(entries):
            source = str(entry.get("source") or "unknown")
            source_to_indices.setdefault(source, []).append(idx)

        if diverse_source_quota > 0 and source_to_indices:
            source_order = sorted(
                source_to_indices.keys(),
                key=lambda src: (
                    -max(
                        (entries[idx]["score"] for idx in source_to_indices.get(src, [])),
                        default=0.0,
                    ),
                    src,
                ),
            )
            while len(selected_indices) < target_count:
                progressed = False
                for source in source_order:
                    if int(selected_source_counts.get(source, 0)) >= diverse_source_quota:
                        continue
                    pick_idx = _pick_best(source_to_indices.get(source, []), quota_phase=True)
                    if pick_idx is None:
                        continue
                    selected_indices.append(pick_idx)
                    selected_index_set.add(pick_idx)
                    selected_source_counts[source] = int(selected_source_counts.get(source, 0)) + 1
                    center = entries[pick_idx].get("center")
                    if isinstance(center, tuple) and len(center) == 2:
                        selected_centers.append((float(center[0]), float(center[1])))
                    progressed = True
                    if len(selected_indices) >= target_count:
                        break
                if not progressed:
                    break

        all_indices = list(range(len(entries)))
        while len(selected_indices) < target_count:
            pick_idx = _pick_best(all_indices, quota_phase=False)
            if pick_idx is None:
                break
            source = str(entries[pick_idx].get("source") or "unknown")
            selected_indices.append(pick_idx)
            selected_index_set.add(pick_idx)
            selected_source_counts[source] = int(selected_source_counts.get(source, 0)) + 1
            center = entries[pick_idx].get("center")
            if isinstance(center, tuple) and len(center) == 2:
                selected_centers.append((float(center[0]), float(center[1])))
        return [entries[idx]["det"] for idx in selected_indices]

    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        if not isinstance(det, dict):
            continue
        label = str(det.get("label") or det.get("class_name") or "").strip()
        if not label:
            continue
        by_label.setdefault(label, []).append(det)
    selections: Dict[str, List[Dict[str, Any]]] = {}
    for label, dets in by_label.items():
        dets_sorted = sorted(dets, key=_det_score, reverse=True)
        high = [d for d in dets_sorted if _det_score(d) >= min_score]
        chosen: List[Dict[str, Any]] = []
        if high:
            if strategy_norm == "diverse":
                chosen.extend(_select_diverse(label, high))
            elif strategy_norm == "random" and len(high) > max_count:
                base_seed = seed if seed is not None else 0
                seed_input = f"{base_seed}:{label}".encode("utf-8", errors="ignore")
                label_seed = int(hashlib.sha1(seed_input).hexdigest(), 16) % (2**32)
                rng = random.Random(label_seed)
                candidates = list(high)
                rng.shuffle(candidates)
                chosen.extend(candidates[:max_count])
            else:
                chosen.extend(high[:max_count])
        if chosen:
            selections[label] = chosen
            if trace_readable:
                handles = []
                for det in chosen:
                    handle = det.get("handle")
                    if handle:
                        handles.append(str(handle))
                handle_text = ", ".join(handles) if handles else "n/a"
                source_counts = _agent_source_counts(chosen)
                source_text = _agent_format_source_counts(source_counts)
                trace_readable(
                    f"deep_prepass similarity exemplars label={label} count={len(chosen)} "
                    f"strategy={strategy_norm} sources={source_text} handles={handle_text}"
                )
    return selections
