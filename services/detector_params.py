from __future__ import annotations

from typing import List, Tuple


def _clamp_conf_value(conf: float, warnings: List[str]) -> float:
    if conf < 0 or conf > 1:
        warnings.append("conf_clamped")
    return max(0.0, min(1.0, float(conf)))


def _clamp_iou_value(iou: float, warnings: List[str]) -> float:
    if iou < 0 or iou > 1:
        warnings.append("iou_clamped")
    return max(0.0, min(1.0, float(iou)))


def _clamp_max_det_value(max_det: int, warnings: List[str]) -> int:
    if max_det < 1:
        warnings.append("max_det_clamped")
        return 1
    if max_det > 10000:
        warnings.append("max_det_clamped")
        return 10000
    return int(max_det)


def _clamp_slice_params(
    slice_size: int,
    overlap: float,
    merge_iou: float,
    img_w: int,
    img_h: int,
    warnings: List[str],
) -> Tuple[int, float, float]:
    max_dim = max(img_w, img_h, 1)
    if slice_size < 64:
        slice_size = 64
        warnings.append("slice_size_clamped")
    if slice_size > max_dim:
        slice_size = max_dim
        warnings.append("slice_size_clamped")
    if overlap < 0 or overlap >= 0.95:
        overlap = min(0.9, max(0.0, overlap))
        warnings.append("overlap_clamped")
    if merge_iou < 0 or merge_iou > 1:
        merge_iou = min(1.0, max(0.0, merge_iou))
        warnings.append("merge_iou_clamped")
    return slice_size, overlap, merge_iou
