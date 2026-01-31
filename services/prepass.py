from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

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
