from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

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


def _agent_compact_tool_result(result: Dict[str, Any], max_items: int = 0) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"summary": "tool_result_invalid"}
    if max_items <= 0:
        return result
    detections = result.get("detections")
    if not isinstance(detections, list):
        candidates = result.get("candidates")
        if not isinstance(candidates, list):
            return result
        total = len(candidates)
        if total <= max_items:
            return result
        trimmed = candidates[:max_items]
        return {
            **{k: v for k, v in result.items() if k != "candidates"},
            "candidates": trimmed,
            "candidate_count": total,
            "truncated": True,
        }
    total = len(detections)
    if total <= max_items:
        return result
    classes = {}
    for det in detections:
        label = str(det.get("label") or det.get("class") or "unknown")
        classes[label] = classes.get(label, 0) + 1
    trimmed = detections[:max_items]
    return {
        **{k: v for k, v in result.items() if k != "detections"},
        "detections": trimmed,
        "detection_count": total,
        "class_counts": classes,
        "truncated": True,
    }


def _agent_merge_prepass_detections(
    detections: List[Dict[str, Any]],
    *,
    iou_thr: float = 0.85,
) -> Tuple[List[Dict[str, Any]], int]:
    if not detections or iou_thr <= 0:
        return detections, 0

    def det_score(det: Dict[str, Any]) -> float:
        try:
            return float(det.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

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
            if label and kept_label and label != kept_label:
                continue
            kept_box = kept.get("bbox_xyxy_px")
            if not isinstance(kept_box, (list, tuple)) or len(kept_box) < 4:
                continue
            if _agent_iou_xyxy(box, kept_box) >= iou_thr:
                matched_idx = idx
                break
        if matched_idx is None:
            entry = dict(det)
            source_list = set(entry.get("source_list") or [])
            if entry.get("source"):
                source_list.add(entry.get("source"))
            if source_list:
                entry["source_list"] = sorted(source_list)
            merged.append(entry)
        else:
            kept = merged[matched_idx]
            source_list = set(kept.get("source_list") or [])
            if kept.get("source"):
                source_list.add(kept.get("source"))
            if det.get("source"):
                source_list.add(det.get("source"))
            keep_det = kept
            if det_score(det) > det_score(kept):
                keep_det = dict(det)
            keep_det["source_list"] = sorted(source_list) if source_list else keep_det.get("source_list")
            merged[matched_idx] = keep_det
            removed += 1
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


def _agent_detection_has_source(det: Dict[str, Any], sources: Set[str]) -> bool:
    if not det or not sources:
        return False
    source = str(det.get("source") or det.get("score_source") or "")
    if source and source in sources:
        return True
    source_list = det.get("source_list")
    if isinstance(source_list, (list, tuple)):
        for item in source_list:
            if str(item) in sources:
                return True
    return False


def _agent_det_score(det: Dict[str, Any]) -> Optional[float]:
    raw = det.get("score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


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
    iou_thr = float(getattr(payload, "iou", None) or default_iou)
    merged, removed = _agent_merge_prepass_detections(detections, iou_thr=iou_thr)
    scoreless_iou = getattr(payload, "scoreless_iou", None) or 0.0
    if scoreless_iou:
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

    sahi_cfg = {
        "enabled": True,
        "slice_size": int(getattr(payload, "sahi_window_size", None) or 640),
        "overlap": float(getattr(payload, "sahi_overlap_ratio", None) or 0.2),
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
    return {
        "detections": detections,
        "warnings": warnings,
        "sam3_text_prompts": part_a.get("sam3_text_prompts") or {},
        "sam3_text_term_map": part_a.get("sam3_text_term_map") or {},
        "image_size": part_a.get("image_size") or {},
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


def _agent_run_prepass_impl(
    payload: Any,
    *,
    pil_img: Any,
    image_token: str,
    labelmap: List[str],
    glossary: str,
    as_tool_messages: bool,
    trace_writer: Optional[Any],
    trace_full_writer: Optional[Any],
    model_id_override: Optional[str],
    agent_message_cls: Any,
    content_item_cls: Any,
    agent_trace_event_cls: Any,
    grid_spec_for_payload_fn: Any,
    readable_detection_line_fn: Any,
    readable_write_fn: Any,
    tool_run_detector_fn: Any,
    generate_sam3_synonyms_fn: Any,
    generate_text_fn: Any,
    extract_json_fn: Any,
    default_synonyms: Any,
    label_key_fn: Any,
    sam3_prompt_variants_fn: Any,
    tool_sam3_text_fn: Any,
    tool_sam3_similarity_fn: Any,
    quadrant_windows_fn: Any,
    tool_look_and_inspect_fn: Any,
    tool_qwen_infer_fn: Any,
    qwen_bbox_to_xyxy_fn: Any,
    resolve_window_overlap_fn: Any,
    resolve_window_size_fn: Any,
    window_positions_fn: Any,
    run_qwen_inference_fn: Any,
    extract_caption_fn: Any,
    sanitize_caption_fn: Any,
    caption_is_degenerate_fn: Any,
    caption_needs_completion_fn: Any,
    caption_has_meta_fn: Any,
    qwen_caption_fn: Any,
    caption_request_cls: Any,
    qwen_caption_cleanup_fn: Any,
    resolve_qwen_variant_fn: Any,
    unload_qwen_fn: Any,
    det_source_summary_fn: Any,
    det_label_counts_fn: Any,
    detection_has_source_fn: Any,
    source_counts_fn: Any,
    format_source_counts_fn: Any,
    merge_prepass_fn: Any,
    compact_tool_result_fn: Any,
    active_detector_conf: Optional[float],
    active_sam3_score_thr: Optional[float],
    active_sam3_mask_thr: Optional[float],
    trace_readable_enabled: bool,
) -> Tuple[
    List[Any],
    List[Dict[str, Any]],
    List[str],
    List[Any],
    bool,
    bool,
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    List[Dict[str, Any]],
]:
    img_w, img_h = pil_img.size
    mode = (getattr(payload, "prepass_mode", None) or "").strip().lower()
    if not mode or mode in {"none", "off", "false"}:
        return [], [], [], [], False, False, None, None, None, None, []
    prepass_messages: List[Any] = []
    prepass_records: List[Dict[str, Any]] = []
    prepass_detections: List[Dict[str, Any]] = []
    prepass_warnings: List[str] = []
    prepass_trace: List[Any] = []
    prepass_has_detector = False
    prepass_has_inspect = False
    qwen_failed = False
    step_id = 0
    max_items = int(getattr(payload, "prepass_inspect_topk", None)) if getattr(payload, "prepass_inspect_topk", None) is not None else 0
    caption_summary: Optional[str] = None
    caption_text: Optional[str] = None
    caption_entries: List[Dict[str, Any]] = []
    sam3_text_prompts: Dict[str, List[str]] = {}
    sam3_text_term_map: Dict[str, Dict[str, List[str]]] = {}
    sam3_text_summary: Optional[str] = None
    prepass_source_summary: Optional[str] = None
    grid_for_log: Optional[Dict[str, Any]] = None
    if getattr(payload, "use_detection_overlay", True) is not False:
        grid_for_log = grid_spec_for_payload_fn(payload, img_w, img_h)

    def add_tool_result(name: str, result: Dict[str, Any], summary: str) -> None:
        nonlocal step_id
        step_id += 1
        prepass_trace.append(
            agent_trace_event_cls(
                step_id=step_id,
                phase="tool_call",
                tool_name=name,
                summary="prepass_tool_call",
                timestamp=time.time(),
            )
        )
        if trace_writer:
            trace_writer(
                {
                    "type": "prepass_tool_call",
                    "tool": name,
                    "summary": summary,
                    "ts": time.time(),
                }
            )
        step_id += 1
        compact = compact_tool_result_fn(result, max_items=max_items)
        if as_tool_messages:
            prepass_messages.append(
                agent_message_cls(
                    role="function",
                    name=name,
                    content=[content_item_cls(text=json.dumps(compact, ensure_ascii=False))],
                )
            )
        else:
            prepass_records.append({"tool": name, "result": compact})
        prepass_trace.append(
            agent_trace_event_cls(
                step_id=step_id,
                phase="tool_result",
                tool_name=name,
                summary=summary,
                timestamp=time.time(),
            )
        )
        if trace_writer:
            trace_writer(
                {
                    "type": "prepass_tool_result",
                    "tool": name,
                    "summary": summary,
                    "result": result,
                    "ts": time.time(),
                }
            )
        if trace_readable_enabled:
            count = None
            count_key = None
            if isinstance(result, dict):
                for key in ("detections", "candidates", "annotations"):
                    items = result.get(key)
                    if isinstance(items, list):
                        count = len(items)
                        count_key = key
                        break
            pretty_summary = summary.replace("prepass_", "prepass ")
            line = f"{pretty_summary}"
            if count is not None and count_key:
                line = f"{line} -> {count} {count_key}"
            readable_write_fn(line)
            if isinstance(result, dict) and count_key:
                items = result.get(count_key)
                if isinstance(items, list) and items:
                    for item in items:
                        if isinstance(item, dict):
                            detail = readable_detection_line_fn(
                                item,
                                grid=grid_for_log,
                                img_w=img_w,
                                img_h=img_h,
                            )
                            readable_write_fn(f"{pretty_summary} item: {detail}")
        if isinstance(result.get("detections"), list):
            prepass_detections.extend(result.get("detections") or [])

    detector_modes: List[str] = []
    if "ensemble" in mode:
        detector_modes = ["yolo", "rfdetr"]
    else:
        detector_modes = [getattr(payload, "detector_mode", None) or "yolo"]
    sahi_enabled = "sahi" in mode
    detector_conf = getattr(payload, "detector_conf", None)
    if detector_conf is None and active_detector_conf is not None:
        detector_conf = active_detector_conf
    for det_mode in detector_modes:
        det_args = {
            "image_token": image_token,
            "detector_id": getattr(payload, "detector_id", None)
            if det_mode == (getattr(payload, "detector_mode", None) or "yolo")
            else None,
            "mode": det_mode,
            "conf": detector_conf,
            "sahi": {"enabled": True} if sahi_enabled else None,
        }
        if trace_full_writer:
            trace_full_writer(
                {
                    "type": "prepass_tool_call",
                    "tool": "run_detector",
                    "args": det_args,
                    "ts": time.time(),
                }
            )
        try:
            det_result = tool_run_detector_fn(
                image_token=det_args["image_token"],
                detector_id=det_args["detector_id"],
                mode=det_args["mode"],
                conf=det_args["conf"],
                sahi=det_args["sahi"],
                register=False,
            )
        except Exception as exc:  # noqa: BLE001
            prepass_warnings.append(f"prepass_detector_failed:{det_mode}:{exc}")
            continue
        det_result = {**det_result, "detector_mode": det_mode}
        if trace_full_writer:
            trace_full_writer(
                {
                    "type": "prepass_tool_result",
                    "tool": "run_detector",
                    "result": det_result,
                    "ts": time.time(),
                }
            )
        add_tool_result("run_detector", det_result, f"prepass_detector:{det_mode}")
        prepass_has_detector = True

    if "sam3_text" in mode and labelmap:
        labels = [lbl for lbl in labelmap if str(lbl).strip()]
        if labels:
            prompt_budget = 10 if getattr(payload, "sam3_text_synonym_budget", None) is None else int(getattr(payload, "sam3_text_synonym_budget", None))
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
            max_results = None
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
                    sam3_args = {
                        "image_token": image_token,
                        "prompt": prompt,
                        "label": label,
                        "score_thr": score_thr,
                        "mask_threshold": mask_thr,
                        "max_results": max_results,
                    }
                    if trace_full_writer:
                        trace_full_writer(
                            {
                                "type": "prepass_tool_call",
                                "tool": "sam3_text",
                                "args": sam3_args,
                                "ts": time.time(),
                            }
                        )
                    try:
                        sam3_result = tool_sam3_text_fn(
                            image_token=sam3_args["image_token"],
                            prompt=sam3_args["prompt"],
                            label=sam3_args["label"],
                            score_thr=sam3_args["score_thr"],
                            mask_threshold=sam3_args["mask_threshold"],
                            max_results=sam3_args["max_results"],
                            register=False,
                        )
                        dets = sam3_result.get("detections") if isinstance(sam3_result, dict) else None
                        if isinstance(dets, list):
                            for det in dets:
                                if not isinstance(det, dict):
                                    continue
                                det["sam3_prompt_term"] = prompt
                                det["sam3_prompt_label"] = label
                                det["sam3_prompt_source"] = prompt_origin
                        if trace_full_writer:
                            trace_full_writer(
                                {
                                    "type": "prepass_tool_result",
                                    "tool": "sam3_text",
                                    "result": sam3_result,
                                    "ts": time.time(),
                                }
                            )
                        summary = f"prepass_sam3_text:{label} prompt={prompt}"
                        add_tool_result("sam3_text", sam3_result, summary)
                    except Exception as exc:  # noqa: BLE001
                        prepass_warnings.append(f"prepass_sam3_text_failed:{label}:{exc}")

    if "similarity" in mode and prepass_detections:
        per_class = int(getattr(payload, "prepass_similarity_per_class", None) or 2)
        if getattr(payload, "prepass_similarity_score", None) is not None:
            score_thr = float(getattr(payload, "prepass_similarity_score", None))
        elif active_sam3_score_thr is not None:
            score_thr = float(active_sam3_score_thr)
        else:
            score_thr = 0.2
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for det in prepass_detections:
            label = str(det.get("label") or det.get("class_name") or "unknown")
            by_label.setdefault(label, []).append(det)
        any_similarity = False
        for label, dets in by_label.items():
            dets.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
            exemplar_boxes: List[Dict[str, Any]] = []
            for det in dets[:per_class]:
                if float(det.get("score") or 0.0) < score_thr:
                    continue
                bbox_2d = det.get("bbox_2d")
                if isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) >= 4:
                    exemplar_boxes.append({"bbox_2d": list(bbox_2d), "bbox_space": "full"})
            if not exemplar_boxes:
                continue
            any_similarity = True
            try:
                sim_args = {
                    "image_token": image_token,
                    "exemplar_boxes": exemplar_boxes,
                    "label": label,
                    "bbox_labels": [True for _ in exemplar_boxes],
                    "score_thr": getattr(payload, "prepass_similarity_score", None),
                }
                if trace_full_writer:
                    trace_full_writer(
                        {
                            "type": "prepass_tool_call",
                            "tool": "sam3_similarity",
                            "args": sim_args,
                            "ts": time.time(),
                        }
                    )
                sim_result = tool_sam3_similarity_fn(
                    image_token=sim_args["image_token"],
                    exemplar_boxes=sim_args["exemplar_boxes"],
                    label=sim_args["label"],
                    bbox_labels=sim_args["bbox_labels"],
                    score_thr=sim_args["score_thr"],
                    register=False,
                )
                if trace_full_writer:
                    trace_full_writer(
                        {
                            "type": "prepass_tool_result",
                            "tool": "sam3_similarity",
                            "result": sim_result,
                            "ts": time.time(),
                        }
                    )
                add_tool_result("sam3_similarity", sim_result, f"prepass_sam3_similarity:{label}")
            except Exception as exc:  # noqa: BLE001
                prepass_warnings.append(f"prepass_sam3_similarity_failed:{label}:{exc}")
        if not any_similarity:
            prepass_warnings.append("prepass_similarity_no_exemplars")

    if "inspect" in mode and getattr(payload, "prepass_inspect_quadrants", True) is not False:
        windows = quadrant_windows_fn(0.1)
        for window in windows:
            inspect_args = {
                "image_token": image_token,
                "window_bbox_2d": window.get("bbox_2d"),
                "labelmap": labelmap,
                "labelmap_glossary": glossary,
                "max_objects": getattr(payload, "prepass_inspect_topk", None),
            }
            if trace_full_writer:
                trace_full_writer(
                    {
                        "type": "prepass_tool_call",
                        "tool": "look_and_inspect",
                        "args": inspect_args,
                        "ts": time.time(),
                    }
                )
            try:
                inspect_result = tool_look_and_inspect_fn(
                    image_token=inspect_args["image_token"],
                    window_bbox_2d=inspect_args["window_bbox_2d"],
                    labelmap=inspect_args["labelmap"],
                    labelmap_glossary=inspect_args["labelmap_glossary"],
                    max_objects=inspect_args["max_objects"],
                    register=False,
                    include_caption=False,
                )
                if trace_full_writer:
                    trace_full_writer(
                        {
                            "type": "prepass_tool_result",
                            "tool": "look_and_inspect",
                            "result": inspect_result,
                            "ts": time.time(),
                        }
                    )
                add_tool_result("look_and_inspect", inspect_result, f"prepass_inspect:{window.get('name')}")
                prepass_has_inspect = True
            except Exception as exc:  # noqa: BLE001
                prepass_warnings.append(f"prepass_inspect_failed:{window.get('name')}:{exc}")
                qwen_failed = True

    if "qwen" in mode and labelmap:
        try:
            max_results = int(getattr(payload, "prepass_inspect_topk", None) or 12)
            if max_results > 30:
                max_results = 30
            qwen_args = {
                "image_token": image_token,
                "items": labelmap,
                "prompt_type": "bbox",
                "max_results": max_results,
                "max_new_tokens": 512,
                "extra_context": glossary,
            }
            if trace_full_writer:
                trace_full_writer(
                    {
                        "type": "prepass_tool_call",
                        "tool": "qwen_infer",
                        "args": qwen_args,
                        "ts": time.time(),
                    }
                )
            qwen_result = tool_qwen_infer_fn(
                image_token=qwen_args["image_token"],
                items=qwen_args["items"],
                prompt_type=qwen_args["prompt_type"],
                max_results=qwen_args["max_results"],
                max_new_tokens=qwen_args["max_new_tokens"],
                extra_context=qwen_args["extra_context"],
                register=False,
            )
            if trace_full_writer:
                trace_full_writer(
                    {
                        "type": "prepass_tool_result",
                        "tool": "qwen_infer",
                        "result": qwen_result,
                        "ts": time.time(),
                    }
                )
            add_tool_result("qwen_infer", qwen_result, "prepass_qwen_infer")
        except Exception as exc:  # noqa: BLE001
            prepass_warnings.append(f"prepass_qwen_infer_failed:{exc}")
            qwen_failed = True

    if getattr(payload, "prepass_caption", False):
        det_hint_sources = {"yolo", "rfdetr"}
        det_hint_items = [det for det in prepass_detections if detection_has_source_fn(det, det_hint_sources)]
        det_hint_items.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
        det_hint_summary = det_label_counts_fn(det_hint_items, limit=10)
        hint_limit = min(len(det_hint_items), 120)
        caption_hints: List[Any] = []
        for det in det_hint_items[:hint_limit]:
            label = str(det.get("label") or det.get("class_name") or "").strip()
            bbox = det.get("bbox_xyxy_px") or None
            if not bbox and det.get("bbox_2d"):
                bbox = qwen_bbox_to_xyxy_fn(
                    pil_img.width,
                    pil_img.height,
                    det.get("bbox_2d"),
                )
            if not label or not bbox or len(bbox) != 4:
                continue
            caption_hints.append(
                {
                    "label": label,
                    "bbox": [float(v) for v in bbox],
                    "confidence": det.get("score"),
                }
            )
        prepass_prompt = (
            "Describe the image region in detail. "
            "Use detector hints as suggestions, but also mention other visible objects. "
            "Do not mention labels, hints, or coordinates."
        )
        if det_hint_summary and det_hint_summary != "none":
            prepass_prompt = f"{prepass_prompt} Detector hints: {det_hint_summary}."
        caption_profile = (getattr(payload, "prepass_caption_profile", None) or "light").strip().lower()
        if caption_profile not in {"light", "deep"}:
            caption_profile = "light"

        caption_text = ""
        windowed_captions: List[Tuple[int, int, int, str]] = []

        if caption_profile == "light":
            caption_model_id = getattr(payload, "prepass_caption_model_id", None) or model_id_override
            caption_max_tokens = int(getattr(payload, "prepass_caption_max_tokens", None) or 256)
            system_prompt = (
                "You are a captioning assistant. Respond in English with a detailed sentence."
            )
            decode_override = {"do_sample": False}
            try:
                runtime = (
                    _ensure_qwen_ready_for_caption(caption_model_id)
                    if caption_model_id
                    else _ensure_qwen_ready()
                )
                full_prompt = (
                    "Describe the full image in detail. "
                    "Use detector hints as suggestions, but mention other visible objects. "
                    "Do not mention labels, hints, or coordinates."
                )
                if det_hint_summary and det_hint_summary != "none":
                    full_prompt = f"{full_prompt} Detector hints: {det_hint_summary}."
                caption_raw, _, _ = run_qwen_inference_fn(
                    full_prompt,
                    pil_img,
                    max_new_tokens=caption_max_tokens,
                    system_prompt_override=system_prompt,
                    runtime_override=runtime,
                    decode_override=decode_override,
                )
                caption_text, _ = extract_caption_fn(caption_raw, marker=None)
                caption_text = sanitize_caption_fn(caption_text)
                overlap = resolve_window_overlap_fn(None)
                window_size = resolve_window_size_fn(None, pil_img.width, pil_img.height, overlap=overlap)
                force_two = True
                x_positions = window_positions_fn(pil_img.width, window_size, overlap, force_two=force_two)
                y_positions = window_positions_fn(pil_img.height, window_size, overlap, force_two=force_two)
                window_prompt = (
                    "Describe this region in 1 detailed sentence. "
                    "Focus only on this region. "
                    "Do not mention labels, hints, or coordinates."
                )
                for y0 in y_positions:
                    for x0 in x_positions:
                        window_img = pil_img.crop((x0, y0, x0 + window_size, y0 + window_size))
                        window_raw, _, _ = run_qwen_inference_fn(
                            window_prompt,
                            window_img,
                            max_new_tokens=caption_max_tokens,
                            system_prompt_override=system_prompt,
                            runtime_override=runtime,
                            decode_override=decode_override,
                        )
                        window_caption, _ = extract_caption_fn(window_raw, marker=None)
                        window_caption = sanitize_caption_fn(window_caption)
                        if window_caption:
                            windowed_captions.append((x0, y0, window_size, window_caption))
                            cell = None
                            if grid_for_log:
                                bbox_2d = xyxy_to_bbox_fn(
                                    img_w,
                                    img_h,
                                    float(x0),
                                    float(y0),
                                    float(x0 + window_size),
                                    float(y0 + window_size),
                                )
                                cell = grid_cell_for_window_bbox_fn(grid_for_log, bbox_2d)
                            cell_text = f" {cell}" if cell else ""
                            readable_write_fn(
                                f"prepass caption window{cell_text} "
                                f"[{x0},{y0},{x0 + window_size},{y0 + window_size}]: {window_caption}"
                            )
                            if trace_writer:
                                trace_writer(
                                    {
                                        "type": "prepass_caption_window",
                                        "window": [int(x0), int(y0), int(window_size)],
                                        "grid_cell": cell,
                                        "caption": window_caption,
                                        "ts": time.time(),
                                    }
                                )
                            if trace_full_writer:
                                trace_full_writer(
                                    {
                                        "type": "prepass_caption_window",
                                        "window": [int(x0), int(y0), int(window_size)],
                                        "grid_cell": cell,
                                        "caption": window_caption,
                                        "ts": time.time(),
                                    }
                                )
            except Exception as exc:  # noqa: BLE001
                prepass_warnings.append(f"prepass_caption_failed:light:{exc}")
                qwen_failed = True
                caption_text = ""
                windowed_captions = []
        if caption_profile != "light":
            caption_variant = getattr(payload, "prepass_caption_variant", None) or getattr(payload, "model_variant", None) or "auto"
            caption_model_id = (getattr(payload, "prepass_caption_model_id", None) or model_id_override or "").strip() or None
            caption_max_tokens = int(getattr(payload, "prepass_caption_max_tokens", None) or 512)
            caption_payload = caption_request_cls(
                image_token=image_token,
                user_prompt=prepass_prompt,
                label_hints=caption_hints or None,
                image_width=pil_img.width,
                image_height=pil_img.height,
                include_counts=False,
                include_coords=True,
                max_boxes=min(len(caption_hints), 120),
                max_new_tokens=caption_max_tokens,
                model_variant=caption_variant,
                model_id=caption_model_id,
                final_answer_only=True,
                two_stage_refine=True,
                caption_mode="windowed",
                caption_all_windows=True,
                restrict_to_labels=False,
                fast_mode=True,
                multi_model_cache=True,
            )
            caption_base_model_id = (
                caption_model_id
                or model_id_override
                or getattr(getattr(payload, "active_qwen_metadata", None), "model_id", None)
                or None
            )
            caption_max_tokens = int(caption_payload.max_new_tokens or 512)

            def run_caption_attempt(request: Any) -> Tuple[str, List[Tuple[int, int, int, str]]]:
                attempt_windows: List[Tuple[int, int, int, str]] = []

                def _window_hook(x0: int, y0: int, size: int, caption: str) -> None:
                    attempt_windows.append((x0, y0, size, caption))
                    if not trace_readable_enabled or not caption:
                        return
                    cell = None
                    if grid_for_log:
                        bbox_2d = xyxy_to_bbox_fn(
                            img_w,
                            img_h,
                            float(x0),
                            float(y0),
                            float(x0 + size),
                            float(y0 + size),
                        )
                        cell = grid_cell_for_window_bbox_fn(grid_for_log, bbox_2d)
                    cell_text = f" {cell}" if cell else ""
                    readable_write_fn(
                        f"prepass caption window{cell_text} "
                        f"[{x0},{y0},{x0 + size},{y0 + size}]: {caption}"
                    )
                    if trace_writer:
                        trace_writer(
                            {
                                "type": "prepass_caption_window",
                                "window": [int(x0), int(y0), int(size)],
                                "grid_cell": cell,
                                "caption": caption,
                                "ts": time.time(),
                            }
                        )
                    if trace_full_writer:
                        trace_full_writer(
                            {
                                "type": "prepass_caption_window",
                                "window": [int(x0), int(y0), int(size)],
                                "grid_cell": cell,
                                "caption": caption,
                                "ts": time.time(),
                            }
                        )

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
                    response = qwen_caption_fn(request)
                    return response.caption or "", attempt_windows
                finally:
                    caption_window_hook.reset(token)

            def caption_is_bad(text: str) -> bool:
                return (
                    not text
                    or caption_is_degenerate_fn(text)
                    or caption_needs_completion_fn(text)
                    or caption_has_meta_fn(text)
                )

            try:
                caption_text, windowed_captions = run_caption_attempt(caption_payload)
            except Exception as exc:  # noqa: BLE001
                prepass_warnings.append(f"prepass_caption_failed:primary:{exc}")
                qwen_failed = True
                caption_text = ""
                windowed_captions = []

            if caption_is_bad(caption_text):
                fallback_payload = caption_payload.copy(
                    update={
                        "model_variant": "Instruct",
                        "use_sampling": False,
                        "two_stage_refine": True,
                    }
                )
                try:
                    caption_text, windowed_captions = run_caption_attempt(fallback_payload)
                except Exception as exc:  # noqa: BLE001
                    prepass_warnings.append(f"prepass_caption_failed:fallback:{exc}")
                    qwen_failed = True

            if caption_is_bad(caption_text):
                fallback_prompt = (
                    "Write a detailed, concrete caption describing the visible scene. "
                    "Mention the main objects and layout. Do not mention labels or coordinates."
                )
                fallback_system = (
                    "You are a captioning assistant. Use the image as truth. "
                    "Respond in English only. Return only the caption."
                )
                try:
                    fallback_raw, _, _ = run_qwen_inference_fn(
                        fallback_prompt,
                        pil_img,
                        max_new_tokens=caption_max_tokens,
                        system_prompt_override=fallback_system,
                        model_id_override=resolve_qwen_variant_fn(caption_base_model_id, "Instruct"),
                        decode_override={"do_sample": False},
                    )
                    caption_text, _ = extract_caption_fn(fallback_raw, marker=None)
                    caption_text = sanitize_caption_fn(caption_text)
                except Exception as exc:  # noqa: BLE001
                    prepass_warnings.append(f"prepass_caption_failed:direct:{exc}")
                    qwen_failed = True

            if caption_is_bad(caption_text):
                caption_text = qwen_caption_cleanup_fn(
                    caption_text or "Describe the image.",
                    pil_img,
                    caption_max_tokens,
                    caption_base_model_id,
                    use_caption_cache=True,
                    model_id_override=resolve_qwen_variant_fn(caption_base_model_id, "Instruct"),
                    runtime_override=None,
                    allowed_labels=None,
                    strict=True,
                    minimal_edit=False,
                )

        if trace_readable_enabled and caption_text:
            readable_write_fn(f"prepass caption summary: {caption_text}")

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
                x_center = x0 + size / 2.0
                y_center = y0 + size / 2.0
                horiz = "left" if x_center < pil_img.width / 3.0 else "right" if x_center > pil_img.width * 2 / 3.0 else "center"
                vert = "top" if y_center < pil_img.height / 3.0 else "bottom" if y_center > pil_img.height * 2 / 3.0 else "middle"
                name = f"{vert}_{horiz}"
                entry = {
                    "window": name,
                    "bbox_2d": list(bbox_2d),
                    "caption": caption,
                }
                caption_entries.append(entry)
                if trace_full_writer:
                    trace_full_writer(
                        {
                            "type": "prepass_caption_result",
                            "window": {"name": name, "bbox_2d": list(bbox_2d)},
                            "caption": caption,
                            "ts": time.time(),
                        }
                    )
                if trace_readable_enabled:
                    coords = readable_format_bbox_fn(bbox_2d)
                    readable_write_fn(f"prepass caption {name} {coords}: {caption}")

        caption_lines: List[str] = []
        if caption_text:
            caption_lines.append(f"Refined caption (Qwen VLM): {caption_text}")
        if caption_entries:
            caption_lines.append("Windowed captions (Qwen VLM). Use as hints for hidden objects:")
            for entry in caption_entries:
                coords = readable_format_bbox_fn(entry.get("bbox_2d"))
                name = entry.get("window") or "window"
                caption_lines.append(f"- {name} {coords}: {entry.get('caption')}")
        caption_summary = "\n".join(caption_lines) if caption_lines else "none"
        prepass_messages.append(
            agent_message_cls(
                role="user",
                content=[content_item_cls(text=caption_summary)],
            )
        )
        step_id += 1
        prepass_trace.append(
            agent_trace_event_cls(
                step_id=step_id,
                phase="intent",
                summary="prepass_caption",
                windows=caption_entries,
                timestamp=time.time(),
            )
        )
        if trace_writer:
            trace_writer({"type": "prepass_caption", "windows": caption_entries, "ts": time.time()})
        if trace_full_writer:
            trace_full_writer(
                {
                    "type": "prepass_caption",
                    "caption": caption_text,
                    "windows": caption_entries,
                    "ts": time.time(),
                }
            )

    if sam3_text_prompts:
        summary_parts = []
        for label in sorted(sam3_text_prompts.keys()):
            prompts = [p for p in sam3_text_prompts[label] if str(p).strip()]
            unique = []
            seen = set()
            for prompt in prompts:
                key = str(prompt).strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                unique.append(str(prompt).strip())
            if unique:
                summary_parts.append(f"{label}: {', '.join(unique[:8])}")
        if summary_parts:
            sam3_text_summary = "; ".join(summary_parts)
            prepass_messages.append(
                agent_message_cls(
                    role="user",
                    content=[
                        content_item_cls(
                            text=(
                                "Prepass sam3_text prompts used (label -> prompts). "
                                "Use these as hints; you may add new synonyms:\n"
                                f"{sam3_text_summary}"
                            )
                        )
                    ],
                )
            )
            if trace_full_writer:
                trace_full_writer(
                    {
                        "type": "prepass_sam3_text_summary",
                        "summary": sam3_text_summary,
                        "ts": time.time(),
                    }
                )
            if trace_readable_enabled:
                readable_write_fn(f"prepass sam3_text prompts: {sam3_text_summary}")

    if prepass_detections:
        raw_counts = source_counts_fn(prepass_detections)
        merged, removed = merge_prepass_fn(prepass_detections, iou_thr=0.85)
        merged_counts = source_counts_fn(merged)
        prepass_detections = merged
        prepass_source_summary = (
            f"raw_sources({format_source_counts_fn(raw_counts) }); "
            f"dedup_iou>=0.85 removed={removed} kept={len(merged)}; "
            f"merged_sources({format_source_counts_fn(merged_counts) })"
        )
        if trace_readable_enabled:
            readable_write_fn(f"prepass dedup: {prepass_source_summary}")
        prepass_messages.append(
            agent_message_cls(
                role="user",
                content=[
                    content_item_cls(
                        text=(
                            "Prepass detection sources: "
                            f"{prepass_source_summary}. "
                            "Caption hints use detector sources only (yolo/rfdetr); "
                            "SAM3 detections are supplemental and may overlap."
                        )
                    )
                ],
            )
        )

    if prepass_records and not as_tool_messages:
        prepass_messages.append(
            agent_message_cls(
                role="user",
                content=[
                    content_item_cls(
                        text=(
                            "Prepass tool results are available via list_candidates and tool calls; "
                            "they are not embedded here to keep prompts compact."
                        )
                    )
                ],
            )
        )
    if qwen_failed:
        try:
            unload_qwen_fn()
        except Exception:
            pass
    return (
        prepass_messages,
        prepass_detections,
        prepass_warnings,
        prepass_trace,
        prepass_has_detector,
        prepass_has_inspect,
        caption_summary,
        sam3_text_summary,
        prepass_source_summary,
        caption_text,
        caption_entries,
    )



def _agent_select_similarity_exemplars(
    min_score: float,
    *,
    detections: List[Dict[str, Any]],
    trace_readable: Optional[Any] = None,
) -> Dict[str, List[Dict[str, Any]]]:
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
        dets_sorted = sorted(dets, key=lambda d: float(d.get("score") or 0.0), reverse=True)
        high = [d for d in dets_sorted if float(d.get("score") or 0.0) >= min_score]
        chosen: List[Dict[str, Any]] = []
        if high:
            chosen.extend(high[:3])
        if chosen:
            selections[label] = chosen
            if trace_readable:
                handles = []
                for det in chosen:
                    handle = det.get("handle")
                    if handle:
                        handles.append(str(handle))
                handle_text = ", ".join(handles) if handles else "n/a"
                trace_readable(
                    f"deep_prepass similarity exemplars label={label} count={len(chosen)} handles={handle_text}"
                )
    return selections
