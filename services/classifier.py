from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _predict_proba_batched_impl(
    crops: Sequence[Any],
    head: Dict[str, Any],
    *,
    batch_size: int,
    encode_batch_fn: Callable[[Sequence[Any], Dict[str, Any], int], Any],
    predict_proba_fn: Callable[[Any, Dict[str, Any]], Any],
    empty_cache_fn: Optional[Callable[[], None]] = None,
) -> Optional[Any]:
    feats = encode_batch_fn(crops, head, batch_size)
    if feats is None:
        return None
    try:
        return predict_proba_fn(feats, head, empty_cache_fn=empty_cache_fn)  # type: ignore[call-arg]
    except TypeError:
        return predict_proba_fn(feats, head)


def _agent_classifier_review_impl(
    detections: List[Dict[str, Any]],
    *,
    pil_img: Optional[Any],
    classifier_head: Optional[Dict[str, Any]],
    resolve_batch_size_fn: Callable[[], int],
    predict_proba_fn: Callable[[Sequence[Any], Dict[str, Any], int], Optional[Any]],
    clip_head_background_indices_fn: Callable[[Sequence[str]], List[int]],
    find_target_index_fn: Callable[[Sequence[str], str], Optional[int]],
    clip_head_keep_mask_fn: Callable[..., Any],
    readable_write_fn: Callable[[str], None],
    readable_format_bbox_fn: Callable[[Sequence[float]], str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    counts = {
        "classifier_checked": 0,
        "classifier_rejected": 0,
        "classifier_errors": 0,
        "classifier_unavailable": 0,
    }
    if not detections:
        return detections, counts
    if pil_img is None or not isinstance(classifier_head, dict):
        counts["classifier_unavailable"] = len(detections)
        for det in detections:
            det["classifier_accept"] = None
            det["classifier_error"] = "unavailable"
        return detections, counts
    classes = [str(c) for c in list(classifier_head.get("classes") or [])]
    bg_indices = clip_head_background_indices_fn(classes)
    min_prob = float(classifier_head.get("min_prob") or 0.5)
    margin = float(classifier_head.get("margin") or 0.0)
    background_margin = float(classifier_head.get("background_margin") or 0.0)
    accepted: List[Dict[str, Any]] = []

    pending: List[Tuple[Dict[str, Any], int, Sequence[float]]] = []
    crops: List[Any] = []
    for det in detections:
        bbox = det.get("bbox_xyxy_px")
        label = str(det.get("label") or "").strip()
        if not bbox or len(bbox) < 4:
            counts["classifier_errors"] += 1
            det["classifier_accept"] = False
            det["classifier_error"] = "missing_bbox"
            continue
        target_idx = find_target_index_fn(classes, label)
        if target_idx is None:
            counts["classifier_errors"] += 1
            det["classifier_accept"] = False
            det["classifier_error"] = "label_not_in_classifier"
            continue
        x1, y1, x2, y2 = bbox[:4]
        crop = pil_img.crop((x1, y1, x2, y2))
        pending.append((det, target_idx, bbox[:4]))
        crops.append(crop)

    if pending:
        batch_size = resolve_batch_size_fn()
        proba_arr = predict_proba_fn(crops, classifier_head, batch_size)
        if proba_arr is None or getattr(proba_arr, "ndim", None) != 2:
            for det, _target_idx, _bbox in pending:
                counts["classifier_errors"] += 1
                det["classifier_accept"] = False
                det["classifier_error"] = "predict_failed"
            return [], counts
        for row, (det, target_idx, bbox) in zip(proba_arr, pending):
            order = sorted(range(len(classes)), key=lambda idx: float(row[idx]), reverse=True)
            best_idx = order[0] if order else None
            best_label = classes[best_idx] if best_idx is not None else "unknown"
            best_prob = float(row[best_idx]) if best_idx is not None else None
            det["classifier_best"] = best_label
            det["classifier_prob"] = best_prob
            keep_mask = clip_head_keep_mask_fn(
                row.reshape(1, -1),
                target_index=target_idx,
                min_prob=min_prob,
                margin=margin,
                background_indices=bg_indices,
                background_guard=True,
                background_margin=background_margin,
            )
            accept = bool(keep_mask[0]) if keep_mask is not None and len(keep_mask) else False
            det["classifier_accept"] = accept
            counts["classifier_checked"] += 1
            summary_bbox = readable_format_bbox_fn(bbox)
            prob_text = f"{best_prob:.3f}" if isinstance(best_prob, float) else "n/a"
            readable_write_fn(
                f"classifier check label={det.get('label')} bbox={summary_bbox} "
                f"best={best_label} prob={prob_text} accept={'yes' if accept else 'no'}"
            )
            if accept:
                accepted.append(det)
            else:
                counts["classifier_rejected"] += 1
    return accepted, counts
