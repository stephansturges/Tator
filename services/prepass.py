from __future__ import annotations

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
