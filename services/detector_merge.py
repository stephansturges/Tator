"""Detector NMS/merge helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _iou_xywh(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _nms_indices(boxes: List[List[float]], scores: List[float], iou_thr: float) -> List[int]:
    order = sorted(range(len(boxes)), key=lambda idx: scores[idx], reverse=True)
    keep: List[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        remaining = []
        for idx in order:
            if _iou_xywh(boxes[current], boxes[idx]) <= iou_thr:
                remaining.append(idx)
        order = remaining
    return keep


def _merge_detections_nms(
    detections: List[Dict[str, Any]],
    iou_thr: float,
    max_det: Optional[int],
) -> List[Dict[str, Any]]:
    if not detections:
        return []
    if iou_thr <= 0:
        merged = detections
    else:
        by_class: Dict[int, List[int]] = {}
        for idx, det in enumerate(detections):
            class_id = int(det.get("class_id", -1))
            by_class.setdefault(class_id, []).append(idx)
        keep_idx: List[int] = []
        for idxs in by_class.values():
            boxes = [detections[i]["bbox"] for i in idxs]
            scores = [float(detections[i].get("score") or 0.0) for i in idxs]
            for keep in _nms_indices(boxes, scores, iou_thr):
                keep_idx.append(idxs[keep])
        merged = [detections[i] for i in keep_idx]
    merged.sort(key=lambda det: float(det.get("score") or 0.0), reverse=True)
    if max_det:
        return merged[:max_det]
    return merged


def _agent_merge_detections(
    detections: List[Dict[str, Any]],
    *,
    iou_thr: float,
    max_det: Optional[int],
    cross_iou: Optional[float],
) -> List[Dict[str, Any]]:
    if not detections:
        return []
    by_class: Dict[int, List[int]] = {}
    for idx, det in enumerate(detections):
        class_id = int(det.get("class_id", -1))
        by_class.setdefault(class_id, []).append(idx)
    kept: List[int] = []
    for idxs in by_class.values():
        boxes = [detections[i]["bbox_xywh_px"] for i in idxs]
        scores = [float(detections[i].get("score") or 0.0) for i in idxs]
        if iou_thr <= 0:
            keep = list(range(len(idxs)))
        else:
            keep = _nms_indices(boxes, scores, iou_thr)
        kept.extend([idxs[k] for k in keep])
    merged = [detections[i] for i in kept]
    if cross_iou and cross_iou > 0:
        boxes = [det["bbox_xywh_px"] for det in merged]
        scores = [float(det.get("score") or 0.0) for det in merged]
        keep = _nms_indices(boxes, scores, cross_iou)
        merged = [merged[i] for i in keep]
    merged.sort(key=lambda det: float(det.get("score") or 0.0), reverse=True)
    if max_det:
        return merged[:max_det]
    return merged
