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
