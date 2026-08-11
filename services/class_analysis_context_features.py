"""Pure same-image geometry features for the class-analysis selector.

The pooled embedding graph answers whether an annotation looks unlike its
labelled class across the dataset.  This module provides a separate source of
evidence: whether the annotation's geometry is unusual relative to other
annotations of the *same class in the same image*.

The extractor is deliberately independent of selector status and ranking.  It
does not mutate input records, load images, fit a classifier, or decide whether
an annotation is valid.  The production utility selector learns how much weight
to give these numeric features alongside visual evidence.

Image dimensions must come from explicit source-image metadata.  Public
``width``/``height`` fields are bbox dimensions and are never interpreted as
source-image dimensions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


SAME_IMAGE_CONTEXT_FEATURE_CONTRACT = "same-image-object-context-features-v1"

_MAD_NORMAL_SCALE = 1.4826
_ROBUST_LOG_SCALE_FLOOR = 0.05
_LOCAL_RADII = (0.05, 0.10, 0.20)

ImageKey = Tuple[str, str]
ImageDimensions = Mapping[Union[ImageKey, str], Sequence[float]]


@dataclass(frozen=True)
class _Geometry:
    record_index: int
    point_id: str
    class_key: str
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float
    area: float
    log_width: float
    log_height: float
    log_area: float
    log_aspect: float
    outside_fraction: float


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _image_key(record: Mapping[str, Any]) -> ImageKey:
    return (
        str(record.get("split") or "train").strip() or "train",
        str(record.get("image_relpath") or record.get("image_name") or "").strip(),
    )


def _class_key(record: Mapping[str, Any]) -> str:
    class_name = str(record.get("class_name") or "").strip()
    if class_name:
        return f"name:{class_name}"
    class_id = record.get("class_id")
    if class_id is not None:
        return f"id:{class_id}"
    return ""


def _dimension_candidates(
    records: Sequence[Mapping[str, Any]],
    key: ImageKey,
    image_dimensions: Optional[ImageDimensions],
) -> List[Tuple[float, float]]:
    values: List[Tuple[float, float]] = []
    if image_dimensions is not None:
        supplied = image_dimensions.get(key)
        if supplied is None:
            supplied = image_dimensions.get(key[1])
        if isinstance(supplied, (list, tuple)) and len(supplied) >= 2:
            width = _finite_float(supplied[0])
            height = _finite_float(supplied[1])
            if width is not None and height is not None and width > 0 and height > 0:
                values.append((width, height))
    for record in records:
        width = _finite_float(
            record.get("_image_width")
            if record.get("_image_width") is not None
            else record.get("image_width")
        )
        height = _finite_float(
            record.get("_image_height")
            if record.get("_image_height") is not None
            else record.get("image_height")
        )
        if width is not None and height is not None and width > 0 and height > 0:
            values.append((width, height))
    return values


def _resolve_dimensions(
    records: Sequence[Mapping[str, Any]],
    key: ImageKey,
    image_dimensions: Optional[ImageDimensions],
) -> Tuple[Optional[Tuple[float, float]], str]:
    values = _dimension_candidates(records, key, image_dimensions)
    if not values:
        return None, "image_dimensions_missing"
    reference = values[0]
    if any(
        not math.isclose(width, reference[0], rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(height, reference[1], rel_tol=0.0, abs_tol=1e-6)
        for width, height in values[1:]
    ):
        return None, "image_dimensions_conflict"
    return reference, ""


def _geometry(
    record: Mapping[str, Any],
    *,
    record_index: int,
    image_width: float,
    image_height: float,
) -> Tuple[Optional[_Geometry], str]:
    bbox = record.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None, "bbox_missing"
    parsed = [_finite_float(value) for value in bbox[:4]]
    if any(value is None for value in parsed):
        return None, "bbox_nonfinite"
    raw_x1, raw_y1, raw_x2, raw_y2 = (float(value) for value in parsed)
    if raw_x2 <= raw_x1 or raw_y2 <= raw_y1:
        return None, "bbox_degenerate"

    raw_area = (raw_x2 - raw_x1) * (raw_y2 - raw_y1)
    x1 = min(image_width, max(0.0, raw_x1))
    y1 = min(image_height, max(0.0, raw_y1))
    x2 = min(image_width, max(0.0, raw_x2))
    y2 = min(image_height, max(0.0, raw_y2))
    if x2 <= x1 or y2 <= y1:
        return None, "bbox_outside_image"

    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    area = width * height
    clipped_area_px = (x2 - x1) * (y2 - y1)
    outside_fraction = max(0.0, min(1.0, 1.0 - clipped_area_px / raw_area))
    return (
        _Geometry(
            record_index=record_index,
            point_id=str(record.get("point_id") or ""),
            class_key=_class_key(record),
            x1=x1 / image_width,
            y1=y1 / image_height,
            x2=x2 / image_width,
            y2=y2 / image_height,
            center_x=(x1 + x2) / (2.0 * image_width),
            center_y=(y1 + y2) / (2.0 * image_height),
            width=width,
            height=height,
            area=area,
            log_width=math.log(max(width, 1e-12)),
            log_height=math.log(max(height, 1e-12)),
            log_area=math.log(max(area, 1e-12)),
            log_aspect=math.log(max(width / height, 1e-12)),
            outside_fraction=outside_fraction,
        ),
        "",
    )


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _median(values: Sequence[float]) -> float:
    return _quantile(values, 0.5)


def _robust_residual(value: float, peer_values: Sequence[float]) -> Tuple[float, float, float]:
    """Return log residual, bounded robust z-score, and peer robust scale."""

    if not peer_values:
        return 0.0, 0.0, 0.0
    median = _median(peer_values)
    absolute_deviations = [abs(peer - median) for peer in peer_values]
    robust_scale = max(
        _ROBUST_LOG_SCALE_FLOOR,
        _MAD_NORMAL_SCALE * _median(absolute_deviations),
    )
    residual = value - median
    return residual, max(-20.0, min(20.0, residual / robust_scale)), robust_scale


def _ridge_predict_log_size(
    target: _Geometry,
    peers: Sequence[_Geometry],
    *,
    ridge: float,
) -> Dict[str, Any]:
    """Fit leave-one-target-out log width/height ~ center x/y regressions."""

    if len(peers) < 4:
        return {
            "perspective_available": False,
            "perspective_peer_count": len(peers),
            "perspective_unavailable_reason": "too_few_same_class_peers",
        }
    design = np.asarray(
        [[1.0, peer.center_x, peer.center_y] for peer in peers],
        dtype=np.float64,
    )
    target_design = np.asarray([1.0, target.center_x, target.center_y], dtype=np.float64)
    penalty = np.diag([0.0, max(0.0, ridge), max(0.0, ridge)])
    gram = design.T @ design + penalty
    width_values = np.asarray([peer.log_width for peer in peers], dtype=np.float64)
    height_values = np.asarray([peer.log_height for peer in peers], dtype=np.float64)
    try:
        width_beta = np.linalg.solve(gram, design.T @ width_values)
        height_beta = np.linalg.solve(gram, design.T @ height_values)
    except np.linalg.LinAlgError:
        return {
            "perspective_available": False,
            "perspective_peer_count": len(peers),
            "perspective_unavailable_reason": "regression_singular",
        }
    expected_log_width = float(target_design @ width_beta)
    expected_log_height = float(target_design @ height_beta)
    width_residual = target.log_width - expected_log_width
    height_residual = target.log_height - expected_log_height
    fitted_width = design @ width_beta
    fitted_height = design @ height_beta

    def r_squared(values: np.ndarray, fitted: np.ndarray) -> float:
        total = float(np.sum((values - np.mean(values)) ** 2))
        if total <= 1e-12:
            return 0.0
        return max(-1.0, min(1.0, 1.0 - float(np.sum((values - fitted) ** 2)) / total))

    return {
        "perspective_available": True,
        "perspective_peer_count": len(peers),
        "perspective_unavailable_reason": "",
        "perspective_expected_log_width_norm": expected_log_width,
        "perspective_expected_log_height_norm": expected_log_height,
        "perspective_log_width_residual": width_residual,
        "perspective_log_height_residual": height_residual,
        # Positive scale residual means larger than expected at this location.
        "perspective_log_scale_residual": 0.5 * (width_residual + height_residual),
        # Positive shape residual means wider/shorter than expected.
        "perspective_log_aspect_residual": width_residual - height_residual,
        "perspective_residual_magnitude": math.hypot(width_residual, height_residual),
        "perspective_width_fit_r2": r_squared(width_values, fitted_width),
        "perspective_height_fit_r2": r_squared(height_values, fitted_height),
    }


def _intersection_metrics(left: _Geometry, right: _Geometry) -> Tuple[float, float]:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0, 0.0
    union = max(1e-12, left.area + right.area - intersection)
    return intersection / union, intersection / max(1e-12, left.area)


def _rank_fraction(value: float, peers: Sequence[float]) -> float:
    if not peers:
        return 0.5
    below = sum(peer < value for peer in peers)
    equal = sum(math.isclose(peer, value, rel_tol=0.0, abs_tol=1e-12) for peer in peers)
    return (below + 0.5 * equal) / len(peers)


def _base_output(record: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "contract": SAME_IMAGE_CONTEXT_FEATURE_CONTRACT,
        "point_id": str(record.get("point_id") or ""),
        "available": False,
        "unavailable_reason": reason,
    }


def extract_same_image_context_features(
    records: Sequence[Mapping[str, Any]],
    *,
    excluded_anchor_point_ids: Collection[str],
    target_point_ids: Optional[Collection[str]] = None,
    alternative_class_by_point_id: Optional[Mapping[str, str]] = None,
    image_dimensions: Optional[ImageDimensions] = None,
    perspective_ridge: float = 1e-3,
) -> List[Dict[str, Any]]:
    """Return deterministic feature rows in input order.

    Regressions and robust peer statistics are leave-one-target-out and use
    only anchors outside ``excluded_anchor_point_ids``.  The caller must pass
    the complete Stage-1 rough-candidate ID set; a suspected label must not
    become evidence that another suspected label has normal geometry.  Records
    carrying an explicit Stage-1 candidate flag are also excluded defensively.
    A missing/invalid record receives an unavailable row while valid peers in
    the same image continue to receive features.  When ``target_point_ids`` is
    provided, only those rows are returned, but every valid same-image record
    remains available as a potential trusted anchor.  This lets the selector
    score a 5k candidate queue without retaining 258k feature dictionaries.
    ``alternative_class_by_point_id`` optionally adds the same robust scale
    comparison against trusted same-image annotations of the proposed class;
    this is evidence for a learned selector, not a replacement-class decision.
    """

    excluded_anchor_ids = {
        str(point_id).strip()
        for point_id in excluded_anchor_point_ids
        if str(point_id).strip()
    }
    selected_target_ids = (
        None
        if target_point_ids is None
        else {
            str(point_id).strip()
            for point_id in target_point_ids
            if str(point_id).strip()
        }
    )
    selected_indices = [
        index
        for index, record in enumerate(records)
        if selected_target_ids is None
        or str(record.get("point_id") or "").strip() in selected_target_ids
    ]
    selected_index_set = set(selected_indices)

    def anchor_eligible(geometry: _Geometry) -> bool:
        if not geometry.point_id or geometry.point_id in excluded_anchor_ids:
            return False
        record = records[geometry.record_index]
        return not bool(
            record.get("is_wrong_class_candidate")
            or record.get("is_rough_outlier_candidate")
        )

    grouped_indices: Dict[ImageKey, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped_indices[_image_key(record)].append(index)

    output: Dict[int, Dict[str, Any]] = {}
    for key in sorted(grouped_indices):
        indices = grouped_indices[key]
        target_indices = [index for index in indices if index in selected_index_set]
        if not target_indices:
            continue
        group_records = [records[index] for index in indices]
        dimensions, dimensions_reason = _resolve_dimensions(
            group_records,
            key,
            image_dimensions,
        )
        if dimensions is None:
            for index in target_indices:
                output[index] = _base_output(records[index], dimensions_reason)
            continue
        image_width, image_height = dimensions
        geometry_by_index: Dict[int, _Geometry] = {}
        for index in indices:
            geometry, reason = _geometry(
                records[index],
                record_index=index,
                image_width=image_width,
                image_height=image_height,
            )
            if geometry is None and index in selected_index_set:
                output[index] = _base_output(records[index], reason)
            else:
                if geometry is not None:
                    geometry_by_index[index] = geometry

        valid_geometries = [geometry_by_index[index] for index in indices if index in geometry_by_index]
        distinct_classes = {geometry.class_key for geometry in valid_geometries if geometry.class_key}
        for index in target_indices:
            target = geometry_by_index.get(index)
            if target is None:
                continue
            same_class = [
                geometry
                for geometry in valid_geometries
                if geometry.class_key == target.class_key
            ]
            peers = [geometry for geometry in same_class if geometry.record_index != index]
            trusted_anchors = [geometry for geometry in peers if anchor_eligible(geometry)]
            alternative_class_name = str(
                (alternative_class_by_point_id or {}).get(target.point_id) or ""
            ).strip()
            alternative_class_key = (
                f"name:{alternative_class_name}" if alternative_class_name else ""
            )
            alternative_class_objects = [
                geometry
                for geometry in valid_geometries
                if alternative_class_key
                and geometry.class_key == alternative_class_key
                and geometry.record_index != index
            ]
            trusted_alternative_anchors = [
                geometry
                for geometry in alternative_class_objects
                if anchor_eligible(geometry)
            ]
            other_objects = [geometry for geometry in valid_geometries if geometry.record_index != index]

            peer_widths = [peer.width for peer in trusted_anchors]
            peer_heights = [peer.height for peer in trusted_anchors]
            peer_areas = [peer.area for peer in trusted_anchors]
            peer_log_widths = [peer.log_width for peer in trusted_anchors]
            peer_log_heights = [peer.log_height for peer in trusted_anchors]
            peer_log_areas = [peer.log_area for peer in trusted_anchors]
            peer_log_aspects = [peer.log_aspect for peer in trusted_anchors]
            width_residual, width_robust_z, width_scale = _robust_residual(
                target.log_width, peer_log_widths
            )
            height_residual, height_robust_z, height_scale = _robust_residual(
                target.log_height, peer_log_heights
            )
            area_residual, area_robust_z, area_scale = _robust_residual(
                target.log_area, peer_log_areas
            )
            aspect_residual, aspect_robust_z, aspect_scale = _robust_residual(
                target.log_aspect, peer_log_aspects
            )
            alternative_widths = [peer.width for peer in trusted_alternative_anchors]
            alternative_heights = [peer.height for peer in trusted_alternative_anchors]
            alternative_areas = [peer.area for peer in trusted_alternative_anchors]
            alternative_width_residual, alternative_width_robust_z, _ = _robust_residual(
                target.log_width,
                [peer.log_width for peer in trusted_alternative_anchors],
            )
            alternative_height_residual, alternative_height_robust_z, _ = _robust_residual(
                target.log_height,
                [peer.log_height for peer in trusted_alternative_anchors],
            )
            alternative_area_residual, alternative_area_robust_z, _ = _robust_residual(
                target.log_area,
                [peer.log_area for peer in trusted_alternative_anchors],
            )
            alternative_aspect_residual, alternative_aspect_robust_z, _ = _robust_residual(
                target.log_aspect,
                [peer.log_aspect for peer in trusted_alternative_anchors],
            )
            current_scale_reference_available = bool(trusted_anchors)
            alternative_scale_reference_available = bool(trusted_alternative_anchors)
            scale_contrast_available = bool(
                current_scale_reference_available
                and alternative_scale_reference_available
            )

            any_distances: List[float] = []
            same_distances: List[float] = []
            local_any = {radius: 0 for radius in _LOCAL_RADII}
            local_same = {radius: 0 for radius in _LOCAL_RADII}
            max_any_iou = 0.0
            max_same_iou = 0.0
            max_other_iou = 0.0
            max_any_target_cover = 0.0
            max_other_target_cover = 0.0
            overlap_any_count = 0
            overlap_same_count = 0
            overlap_other_count = 0
            for other in other_objects:
                distance = math.hypot(
                    target.center_x - other.center_x,
                    target.center_y - other.center_y,
                ) / math.sqrt(2.0)
                any_distances.append(distance)
                is_same = other.class_key == target.class_key
                if is_same:
                    same_distances.append(distance)
                for radius in _LOCAL_RADII:
                    if distance <= radius:
                        local_any[radius] += 1
                        if is_same:
                            local_same[radius] += 1
                iou, target_cover = _intersection_metrics(target, other)
                max_any_iou = max(max_any_iou, iou)
                max_any_target_cover = max(max_any_target_cover, target_cover)
                if is_same:
                    max_same_iou = max(max_same_iou, iou)
                else:
                    max_other_iou = max(max_other_iou, iou)
                    max_other_target_cover = max(
                        max_other_target_cover, target_cover
                    )
                if target_cover > 0.0:
                    overlap_any_count += 1
                    if is_same:
                        overlap_same_count += 1
                    else:
                        overlap_other_count += 1

            border_distance = min(
                target.x1,
                target.y1,
                1.0 - target.x2,
                1.0 - target.y2,
            )
            touches_border = border_distance <= 1e-9 or target.outside_fraction > 0.0
            feature_row: Dict[str, Any] = {
                "contract": SAME_IMAGE_CONTEXT_FEATURE_CONTRACT,
                "point_id": target.point_id,
                "available": True,
                "unavailable_reason": "",
                "image_width": int(round(image_width)),
                "image_height": int(round(image_height)),
                "image_object_count": len(valid_geometries),
                "image_distinct_class_count": len(distinct_classes),
                "same_class_count": len(same_class),
                "same_class_peer_count": len(peers),
                "trusted_same_class_anchor_count": len(trusted_anchors),
                "excluded_same_class_peer_count": len(peers) - len(trusted_anchors),
                "same_class_scale_reference_available": current_scale_reference_available,
                "same_class_reference_contract": (
                    "same-image-same-class-exclude-stage1-rough-and-self-v1"
                ),
                "alternative_class_name": alternative_class_name,
                "alternative_class_count": len(alternative_class_objects),
                "trusted_alternative_class_anchor_count": len(
                    trusted_alternative_anchors
                ),
                "excluded_alternative_class_anchor_count": (
                    len(alternative_class_objects) - len(trusted_alternative_anchors)
                ),
                "alternative_scale_reference_available": (
                    alternative_scale_reference_available
                ),
                "scale_contrast_available": scale_contrast_available,
                "same_class_fraction": len(same_class) / max(1, len(valid_geometries)),
                "bbox_center_x_norm": target.center_x,
                "bbox_center_y_norm": target.center_y,
                "bbox_width_norm": target.width,
                "bbox_height_norm": target.height,
                "bbox_area_fraction": target.area,
                "bbox_log_width_norm": target.log_width,
                "bbox_log_height_norm": target.log_height,
                "bbox_log_area_fraction": target.log_area,
                "bbox_log_aspect": target.log_aspect,
                "bbox_border_distance_norm": border_distance,
                "bbox_touches_border": touches_border,
                "bbox_outside_fraction": target.outside_fraction,
                "same_class_peer_width_median_norm": _median(peer_widths),
                "same_class_peer_width_q25_norm": _quantile(peer_widths, 0.25),
                "same_class_peer_width_q75_norm": _quantile(peer_widths, 0.75),
                "same_class_peer_height_median_norm": _median(peer_heights),
                "same_class_peer_height_q25_norm": _quantile(peer_heights, 0.25),
                "same_class_peer_height_q75_norm": _quantile(peer_heights, 0.75),
                "same_class_peer_area_median_fraction": _median(peer_areas),
                "same_class_peer_area_q25_fraction": _quantile(peer_areas, 0.25),
                "same_class_peer_area_q75_fraction": _quantile(peer_areas, 0.75),
                "same_class_log_width_residual": width_residual,
                "same_class_log_height_residual": height_residual,
                "same_class_log_area_residual": area_residual,
                "same_class_log_aspect_residual": aspect_residual,
                "same_class_width_robust_z": width_robust_z,
                "same_class_height_robust_z": height_robust_z,
                "same_class_area_robust_z": area_robust_z,
                "same_class_aspect_robust_z": aspect_robust_z,
                "same_class_width_robust_scale": width_scale,
                "same_class_height_robust_scale": height_scale,
                "same_class_area_robust_scale": area_scale,
                "same_class_aspect_robust_scale": aspect_scale,
                "same_class_area_rank_fraction": _rank_fraction(target.area, peer_areas),
                "same_class_area_rank_extremeness": abs(
                    2.0 * _rank_fraction(target.area, peer_areas) - 1.0
                ),
                "nearest_object_center_distance_norm": min(any_distances, default=1.0),
                "nearest_same_class_center_distance_norm": min(same_distances, default=1.0),
                "max_object_iou": max_any_iou,
                "max_same_class_iou": max_same_iou,
                "max_other_class_iou": max_other_iou,
                "max_target_coverage_by_any": max_any_target_cover,
                "max_target_coverage_by_other": max_other_target_cover,
                "overlapping_object_count": overlap_any_count,
                "overlapping_same_class_count": overlap_same_count,
                "overlapping_other_class_count": overlap_other_count,
                "alternative_class_anchor_width_median_norm": _median(
                    alternative_widths
                ),
                "alternative_class_anchor_height_median_norm": _median(
                    alternative_heights
                ),
                "alternative_class_anchor_area_median_fraction": _median(
                    alternative_areas
                ),
                "alternative_class_log_width_residual": (
                    alternative_width_residual
                ),
                "alternative_class_log_height_residual": (
                    alternative_height_residual
                ),
                "alternative_class_log_area_residual": alternative_area_residual,
                "alternative_class_log_aspect_residual": (
                    alternative_aspect_residual
                ),
                "alternative_class_width_robust_z": alternative_width_robust_z,
                "alternative_class_height_robust_z": alternative_height_robust_z,
                "alternative_class_area_robust_z": alternative_area_robust_z,
                "alternative_class_aspect_robust_z": alternative_aspect_robust_z,
                # Positive means the target is geometrically more typical for
                # the proposed alternative than for its current annotation.
                "current_minus_alternative_abs_scale_residual": (
                    abs(area_residual) - abs(alternative_area_residual)
                    if scale_contrast_available
                    else 0.0
                ),
                "current_minus_alternative_abs_aspect_residual": (
                    abs(aspect_residual) - abs(alternative_aspect_residual)
                    if scale_contrast_available
                    else 0.0
                ),
                "current_minus_alternative_geometry_residual_magnitude": (
                    math.hypot(width_residual, height_residual)
                    - math.hypot(
                        alternative_width_residual,
                        alternative_height_residual,
                    )
                    if scale_contrast_available
                    else 0.0
                ),
            }
            for radius in _LOCAL_RADII:
                suffix = f"r{int(round(radius * 100)):02d}"
                feature_row[f"local_object_count_{suffix}"] = local_any[radius]
                feature_row[f"local_same_class_count_{suffix}"] = local_same[radius]
            feature_row.update(
                _ridge_predict_log_size(
                    target,
                    trusted_anchors,
                    ridge=perspective_ridge,
                )
            )
            output[index] = feature_row

    return [
        output.get(index)
        or _base_output(records[index], "internal_grouping_error")
        for index in selected_indices
    ]


__all__ = [
    "SAME_IMAGE_CONTEXT_FEATURE_CONTRACT",
    "extract_same_image_context_features",
]
