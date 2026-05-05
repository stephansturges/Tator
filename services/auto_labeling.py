"""Helpers for dataset-scale automatic labeling."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from utils.coco import _mask_to_polygon_impl
from utils.coords import _xyxy_to_yolo_norm_list, _yolo_to_xyxy
from utils.glossary import _parse_glossary_mapping


AUTO_LABEL_TARGET_MODE_AUTO = "auto"
AUTO_LABEL_TARGET_MODE_DETECTION = "detection"
AUTO_LABEL_TARGET_MODE_SEGMENTATION = "segmentation"

AUTO_LABEL_WINDOW_MODE_FULL = "full_image"
AUTO_LABEL_WINDOW_MODE_QUADRANTS = "quadrants"
AUTO_LABEL_WINDOW_MODE_PLANNER = "planner_auto"


_FALCON_PROMPT_TERM_LIMIT = 6
_FALCON_QUERY_TIER_A = "A"
_FALCON_QUERY_TIER_B = "B"
_FALCON_QUERY_TIER_C = "C"

_FALCON_QUERY_PRIORITY_TERMS: Dict[str, List[str]] = {
    "bike": ["motorcycle", "scooter", "bike"],
    "boat": ["boat", "canoe", "kayak", "ship"],
    "building": ["building", "house", "office building", "warehouse"],
    "bus": ["bus", "coach", "autobus"],
    "container": ["shipping container", "truck container", "container"],
    "digger": ["excavator", "backhoe", "bulldozer", "digger", "dozer"],
    "gastank": ["storage tank", "oil tank", "silo", "pressure vessel"],
    "light_vehicle": ["car", "SUV", "sedan", "van", "pickup truck", "hatchback"],
    "person": ["person", "pedestrian", "cyclist", "walker", "swimmer"],
    "solarpanels": ["solar panel", "solar panel array"],
    "truck": ["truck", "lorry", "semi truck", "18-wheeler", "big rig"],
    "utility_pole": ["utility pole", "streetlight", "transmission tower", "mast", "power pylon"],
}

_FALCON_QUERY_BLOCKLIST: Dict[str, set[str]] = {
    "digger": {"construction vehicle", "heavy machinery", "tractor", "steam shovel"},
    "light_vehicle": {"light vehicle", "light_vehicle", "passenger vehicle", "automobile", "4x4"},
    "person": {"human", "individual", "passenger"},
    "solarpanels": {"array", "solarpanels"},
    "boat": {"surfboard"},
    "gastank": {"tank", "barrel", "silos"},
    "truck": {"commercial vehicle", "heavy-duty vehicle", "semi-trailer truck"},
    "utility_pole": {
        "antenna",
        "pole",
        "street fixture",
        "drying rack",
        "satellite dish",
        "mounting pole",
        "light fixture",
        "utility_pole",
    },
}

def _humanize_label(label: str) -> str:
    return str(label or "").replace("_", " ").strip()


def _clean_falcon_term(term: str) -> str:
    cleaned = str(term or "").strip()
    cleaned = re.sub(r"[_]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        return ""
    if any(token in cleaned.lower() for token in (",", ";", "/", " or ", " and ")):
        return ""
    return cleaned

def _falcon_query_from_term(term: str, *, query_frame: str = "term") -> str:
    cleaned = str(term or "").strip()
    if not cleaned:
        return ""
    frame = str(query_frame or "term").strip().lower()
    if frame == "all_instances":
        return f"all instances of {cleaned}"
    if frame == "the":
        return f"the {cleaned}"
    return cleaned


def _falcon_term_buckets_for_class(
    class_name: str,
    *,
    labelmap: Sequence[str],
    glossary: str = "",
) -> Dict[str, Any]:
    canonical = str(class_name or "").strip()
    if not canonical:
        return {"canonical": "", "priority": [], "glossary": []}
    blocked = {term.lower() for term in (_FALCON_QUERY_BLOCKLIST.get(canonical) or set())}
    canonical_term = _clean_falcon_term(_humanize_label(canonical))
    priority_terms: List[str] = []
    seen_priority: set[str] = set()
    for raw in (_FALCON_QUERY_PRIORITY_TERMS.get(canonical) or []):
        cleaned = _clean_falcon_term(raw)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in blocked or key in seen_priority:
            continue
        seen_priority.add(key)
        priority_terms.append(cleaned)
    glossary_terms: List[str] = []
    seen_glossary = set(seen_priority)
    if canonical_term:
        seen_glossary.add(canonical_term.lower())
    for raw in _parse_glossary_mapping(str(glossary or ""), labelmap).get(canonical) or []:
        cleaned = _clean_falcon_term(raw)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in blocked or key in seen_glossary:
            continue
        seen_glossary.add(key)
        glossary_terms.append(cleaned)
    return {
        "canonical": canonical_term,
        "priority": priority_terms[:_FALCON_PROMPT_TERM_LIMIT],
        "glossary": glossary_terms[:_FALCON_PROMPT_TERM_LIMIT],
    }


def _build_falcon_terms_for_class(
    class_name: str,
    *,
    labelmap: Sequence[str],
    glossary: str = "",
    prompt_style: str = "default",
) -> List[str]:
    buckets = _falcon_term_buckets_for_class(class_name, labelmap=labelmap, glossary=glossary or "")
    canonical_term = str(buckets.get("canonical") or "").strip()
    priority_terms = list(buckets.get("priority") or [])
    glossary_terms = list(buckets.get("glossary") or [])
    style = str(prompt_style or "default").strip().lower()

    if style == "canonical_only":
        return [canonical_term] if canonical_term else []
    if style == "priority_top1":
        if priority_terms:
            return priority_terms[:1]
        return [canonical_term] if canonical_term else []
    if style == "priority_only":
        return priority_terms or ([canonical_term] if canonical_term else [])
    if style == "glossary_only":
        return glossary_terms or priority_terms[:1] or ([canonical_term] if canonical_term else [])

    terms: List[str] = []
    if canonical_term and canonical_term.lower() not in {
        term.lower() for term in (_FALCON_QUERY_BLOCKLIST.get(str(class_name or "").strip()) or set())
    }:
        terms.append(canonical_term)
    if priority_terms:
        terms.extend(priority_terms)
    if glossary_terms:
        terms.extend(glossary_terms)
    deduped: List[str] = []
    seen: set[str] = set()
    for term in terms:
        key = str(term or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(str(term))
    return deduped[:_FALCON_PROMPT_TERM_LIMIT]


def _build_falcon_query_tier_rows_for_class(
    class_name: str,
    *,
    labelmap: Sequence[str],
    glossary: str = "",
    prompt_style: str = "default",
    query_frame: str = "term",
) -> List[Dict[str, str]]:
    canonical = str(class_name or "").strip()
    if not canonical:
        return []
    buckets = _falcon_term_buckets_for_class(canonical, labelmap=labelmap, glossary=glossary or "")
    canonical_term = str(buckets.get("canonical") or "").strip()
    priority_terms = list(buckets.get("priority") or [])
    glossary_terms = list(buckets.get("glossary") or [])
    blocked = {term.lower() for term in (_FALCON_QUERY_BLOCKLIST.get(canonical) or set())}
    style = str(prompt_style or "default").strip().lower()
    tier_a_terms: List[str] = []
    tier_b_terms: List[str] = []
    tier_c_terms: List[str] = []
    if style == "canonical_only":
        if canonical_term:
            tier_a_terms = [canonical_term]
    elif style == "priority_top1":
        if priority_terms:
            tier_a_terms = priority_terms[:1]
        elif canonical_term and canonical_term.lower() not in blocked:
            tier_a_terms = [canonical_term]
    elif style == "priority_only":
        if priority_terms:
            tier_a_terms = priority_terms[:1]
            tier_b_terms = priority_terms[1:]
        elif canonical_term and canonical_term.lower() not in blocked:
            tier_a_terms = [canonical_term]
    elif style == "glossary_only":
        if glossary_terms:
            tier_a_terms = glossary_terms[:1]
            tier_b_terms = glossary_terms[1:]
        elif priority_terms:
            tier_a_terms = priority_terms[:1]
        elif canonical_term and canonical_term.lower() not in blocked:
            tier_a_terms = [canonical_term]
    else:
        if canonical_term and canonical_term.lower() not in blocked:
            tier_a_terms = [canonical_term]
            if priority_terms:
                distinct_priority_terms = [
                    term for term in priority_terms if term.lower() != canonical_term.lower()
                ]
                if distinct_priority_terms:
                    tier_a_terms.append(distinct_priority_terms[0])
                    tier_b_terms = distinct_priority_terms[1:]
                else:
                    tier_b_terms = []
        elif priority_terms:
            tier_a_terms = priority_terms[:1]
            tier_b_terms = priority_terms[1:]
        seen = {term.lower() for term in tier_a_terms + tier_b_terms}
        for cleaned in glossary_terms:
            key = cleaned.lower()
            if key in blocked or key in seen:
                continue
            seen.add(key)
            tier_c_terms.append(cleaned)
    rows: List[Dict[str, str]] = []
    for tier_name, terms in (
        (_FALCON_QUERY_TIER_A, tier_a_terms),
        (_FALCON_QUERY_TIER_B, tier_b_terms),
        (_FALCON_QUERY_TIER_C, tier_c_terms),
    ):
        for term in terms[:_FALCON_PROMPT_TERM_LIMIT]:
            rows.append(
                {
                    "class_name": canonical,
                    "query": _falcon_query_from_term(term, query_frame=query_frame),
                    "term": term,
                    "tier": tier_name,
                }
            )
    return rows


def build_falcon_query_tiers(
    class_names: Sequence[str],
    *,
    labelmap: Sequence[str],
    glossary: str = "",
    prompt_style: str = "default",
    query_frame: str = "term",
) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for class_name in class_names:
        canonical = str(class_name or "").strip()
        if not canonical:
            continue
        out[canonical] = _build_falcon_query_tier_rows_for_class(
            canonical,
            labelmap=labelmap,
            glossary=glossary or "",
            prompt_style=prompt_style,
            query_frame=query_frame,
        )
    return out


def infer_dataset_annotation_mode(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bbox_count = 0
    polygon_count = 0
    image_count = 0
    for row in rows or []:
        lines = row.get("label_lines")
        if not isinstance(lines, list):
            continue
        image_count += 1
        for raw_line in lines:
            parts = str(raw_line or "").strip().split()
            if len(parts) < 5:
                continue
            if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                polygon_count += 1
            else:
                bbox_count += 1
    if polygon_count > 0:
        return {
            "mode": AUTO_LABEL_TARGET_MODE_SEGMENTATION,
            "reason": "existing_polygons",
            "bbox_count": bbox_count,
            "polygon_count": polygon_count,
            "images_seen": image_count,
        }
    if bbox_count > 0:
        return {
            "mode": AUTO_LABEL_TARGET_MODE_DETECTION,
            "reason": "existing_bboxes",
            "bbox_count": bbox_count,
            "polygon_count": polygon_count,
            "images_seen": image_count,
        }
    return {
        "mode": None,
        "reason": "no_existing_annotations",
        "bbox_count": bbox_count,
        "polygon_count": polygon_count,
        "images_seen": image_count,
    }


def parse_yolo_label_line(
    line: str,
    *,
    width: int,
    height: int,
) -> Optional[Dict[str, Any]]:
    parts = str(line or "").strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except (TypeError, ValueError):
        return None
    if len(coords) >= 6 and len(coords) % 2 == 0:
        polygon: List[Tuple[float, float]] = []
        for idx in range(0, len(coords), 2):
            x = max(0.0, min(float(width), coords[idx] * float(width)))
            y = max(0.0, min(float(height), coords[idx + 1] * float(height)))
            polygon.append((x, y))
        if len(polygon) >= 3:
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            return {
                "class_id": class_id,
                "bbox_xyxy": (
                    float(min(xs)),
                    float(min(ys)),
                    float(max(xs)),
                    float(max(ys)),
                ),
                "polygon_xy": polygon,
                "kind": "polygon",
            }
    try:
        x1, y1, x2, y2 = _yolo_to_xyxy(width, height, coords[:4])
    except Exception:
        return None
    return {
        "class_id": class_id,
        "bbox_xyxy": (float(x1), float(y1), float(x2), float(y2)),
        "polygon_xy": None,
        "kind": "bbox",
    }


def bbox_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def mask_iou(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> float:
    if mask_a is None or mask_b is None:
        return 0.0
    if mask_a.shape != mask_b.shape:
        return 0.0
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union <= 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)


def mask_bbox_xyxy(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    if mask is None:
        return None
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return (x1, y1, x2, y2)


def polygon_mask_from_yolo_line(
    line: str,
    *,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    parsed = parse_yolo_label_line(line, width=width, height=height)
    if not parsed or parsed.get("polygon_xy") is None:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = parsed.get("polygon_xy") or []
    if len(polygon) < 3:
        return None
    from PIL import Image, ImageDraw  # local import

    img = Image.fromarray(mask, mode="L")
    draw = ImageDraw.Draw(img)
    draw.polygon([(float(x), float(y)) for x, y in polygon], outline=1, fill=1)
    mask = np.array(img, dtype=np.uint8)
    return mask.astype(bool)


def build_quadrant_windows(
    width: int,
    height: int,
    *,
    overlap_ratio: float = 0.1,
) -> List[Dict[str, Any]]:
    overlap_ratio = max(0.0, min(0.45, float(overlap_ratio or 0.0)))
    half_w = float(width) / 2.0
    half_h = float(height) / 2.0
    overlap_w = half_w * overlap_ratio
    overlap_h = half_h * overlap_ratio
    mid_x = float(width) / 2.0
    mid_y = float(height) / 2.0
    windows = [
        ("Q1", 0.0, 0.0, min(float(width), mid_x + overlap_w), min(float(height), mid_y + overlap_h)),
        ("Q2", max(0.0, mid_x - overlap_w), 0.0, float(width), min(float(height), mid_y + overlap_h)),
        ("Q3", 0.0, max(0.0, mid_y - overlap_h), min(float(width), mid_x + overlap_w), float(height)),
        ("Q4", max(0.0, mid_x - overlap_w), max(0.0, mid_y - overlap_h), float(width), float(height)),
    ]
    return [
        {
            "id": window_id,
            "xyxy": (x1, y1, x2, y2),
        }
        for window_id, x1, y1, x2, y2 in windows
    ]


def build_grid_windows(
    width: int,
    height: int,
    *,
    cols: int,
    rows: int,
    overlap_ratio: float = 0.1,
) -> List[Dict[str, Any]]:
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    overlap_ratio = max(0.0, min(0.45, float(overlap_ratio or 0.0)))
    cell_w = float(width) / float(cols)
    cell_h = float(height) / float(rows)
    win_w = cell_w * (1.0 + overlap_ratio)
    win_h = cell_h * (1.0 + overlap_ratio)
    out: List[Dict[str, Any]] = []
    for row_idx in range(rows):
        for col_idx in range(cols):
            x_center = (col_idx + 0.5) * cell_w
            y_center = (row_idx + 0.5) * cell_h
            x1 = max(0.0, x_center - win_w / 2.0)
            y1 = max(0.0, y_center - win_h / 2.0)
            x2 = min(float(width), x_center + win_w / 2.0)
            y2 = min(float(height), y_center + win_h / 2.0)
            cell_id = f"{chr(ord('A') + col_idx)}{row_idx + 1}"
            out.append({"id": cell_id, "xyxy": (x1, y1, x2, y2)})
    return out


def serialize_bbox_label_line(
    class_id: int,
    bbox_xyxy: Sequence[float],
    *,
    width: int,
    height: int,
) -> str:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    bbox = _xyxy_to_yolo_norm_list(width, height, x1, y1, x2, y2)
    return f"{int(class_id)} " + " ".join(f"{float(v):.6f}" for v in bbox)


def serialize_bbox_polygon_label_line(
    class_id: int,
    bbox_xyxy: Sequence[float],
    *,
    width: int,
    height: int,
) -> str:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    polygon = (
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    )
    coords: List[str] = []
    for x, y in polygon:
        coords.append(f"{max(0.0, min(1.0, float(x) / float(width))):.6f}")
        coords.append(f"{max(0.0, min(1.0, float(y) / float(height))):.6f}")
    return f"{int(class_id)} " + " ".join(coords)


def serialize_mask_label_line(
    class_id: int,
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    simplify_epsilon: float = 2.0,
) -> str:
    polygon = _mask_to_polygon_impl(mask.astype(np.uint8), float(simplify_epsilon)) or []
    if len(polygon) < 3:
        bbox_xyxy = mask_bbox_xyxy(mask)
        if bbox_xyxy is None:
            raise ValueError("mask_empty")
        x1, y1, x2, y2 = bbox_xyxy
        polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    coords: List[str] = []
    for x, y in polygon:
        coords.append(f"{max(0.0, min(1.0, float(x) / float(width))):.6f}")
        coords.append(f"{max(0.0, min(1.0, float(y) / float(height))):.6f}")
    return f"{int(class_id)} " + " ".join(coords)


def _component_bbox_gap(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) < 4 or len(right) < 4:
        return float("inf")
    lx1, ly1, lx2, ly2 = [float(v) for v in left[:4]]
    rx1, ry1, rx2, ry2 = [float(v) for v in right[:4]]
    gap_x = max(0.0, max(lx1 - rx2, rx1 - lx2))
    gap_y = max(0.0, max(ly1 - ry2, ry1 - ly2))
    return float((gap_x**2 + gap_y**2) ** 0.5)


def _bbox_border_touch_count(
    bbox_xyxy: Sequence[float],
    *,
    width: int,
    height: int,
) -> int:
    if len(bbox_xyxy) < 4:
        return 0
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    touches = 0
    if x1 <= 0.0:
        touches += 1
    if y1 <= 0.0:
        touches += 1
    if x2 >= float(width):
        touches += 1
    if y2 >= float(height):
        touches += 1
    return touches


def score_falcon_candidate(
    *,
    bbox_area_fraction: float,
    border_touch_count: int,
    component_count: int,
    derivation_mode: str = "",
) -> float:
    """Rank Falcon boxes so object-scale components beat tiny scraps or huge blobs."""
    area = max(0.0, float(bbox_area_fraction or 0.0))
    border_touch = max(0, int(border_touch_count or 0))
    components = max(1, int(component_count or 1))
    mode = str(derivation_mode or "").strip().lower()

    if area <= 0.0005:
        area_score = 0.08
    elif area <= 0.003:
        area_score = 0.08 + 0.55 * ((area - 0.0005) / 0.0025)
    elif area <= 0.06:
        area_score = 0.82 + 0.12 * ((area - 0.003) / 0.057)
    elif area <= 0.18:
        area_score = 0.94 - 0.24 * ((area - 0.06) / 0.12)
    elif area <= 0.35:
        area_score = 0.70 - 0.45 * ((area - 0.18) / 0.17)
    else:
        area_score = 0.18

    score = area_score
    if border_touch >= 2:
        score -= 0.12
    elif border_touch == 1:
        score -= 0.04
    score -= 0.015 * min(max(0, components - 1), 6)
    if mode.startswith("edge_strip_cluster") and 0.003 <= area <= 0.08:
        score += 0.02
    return max(0.05, min(0.95, float(score)))


def adjust_falcon_candidate_score(
    *,
    class_name: str,
    query: str,
    tier: str,
    base_score: float,
    bbox_xyxy: Sequence[float],
    bbox_area_fraction: float,
) -> float:
    """Apply light class-aware priors so tiny-object queries are not drowned by coarse boxes."""
    score = float(base_score or 0.0)
    canonical = str(class_name or "").strip().lower()
    query_text = str(query or "").strip().lower()
    tier_name = str(tier or "").strip().upper()
    area = max(0.0, float(bbox_area_fraction or 0.0))
    bbox = tuple(float(v) for v in (bbox_xyxy or ())[:4])
    width = max(1.0, (bbox[2] - bbox[0]) if len(bbox) >= 4 else 1.0)
    height = max(1.0, (bbox[3] - bbox[1]) if len(bbox) >= 4 else 1.0)
    aspect_ratio = max(width, height) / max(1.0, min(width, height))

    if canonical == "utility_pole":
        if area > 0.0:
            ideal = 0.003
            distance = abs(math.log(max(area, 1e-6) / ideal))
            score += max(-0.10, min(0.08, 0.08 - 0.12 * distance))
        if aspect_ratio >= 2.5:
            score += 0.07
        elif aspect_ratio >= 1.8:
            score += 0.04
        elif aspect_ratio <= 1.35:
            score -= 0.06
        if "pylon" in query_text or "tower" in query_text:
            score += 0.03
        elif tier_name in {"B", "C"}:
            score += 0.01
    elif canonical == "light_vehicle":
        if area > 0.0:
            ideal = 0.025
            distance = abs(math.log(max(area, 1e-6) / ideal))
            score += max(-0.12, min(0.08, 0.08 - 0.12 * distance))
        if 1.2 <= aspect_ratio <= 4.5:
            score += 0.03
        if tier_name in {"B", "C"}:
            score += 0.02
    elif canonical == "person":
        if 0.002 <= area <= 0.03:
            score += 0.05
        elif area > 0.09:
            score -= 0.08
        if aspect_ratio >= 2.0:
            score += 0.03
    elif canonical == "building":
        if 0.03 <= area <= 0.4:
            score += 0.05
        elif area < 0.008:
            score -= 0.08
    return max(0.05, min(0.99, score))


def _mask_component_records(mask: np.ndarray) -> List[Dict[str, Any]]:
    if mask is None:
        return []
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        return []
    labeled, component_count = ndimage.label(binary)
    if component_count <= 0:
        return []
    object_slices = ndimage.find_objects(labeled)
    records: List[Dict[str, Any]] = []
    for component_id, obj_slice in enumerate(object_slices, start=1):
        if obj_slice is None or len(obj_slice) < 2:
            continue
        y_slice, x_slice = obj_slice[:2]
        y1 = int(y_slice.start or 0)
        y2 = int(y_slice.stop or 0)
        x1 = int(x_slice.start or 0)
        x2 = int(x_slice.stop or 0)
        if x2 <= x1 or y2 <= y1:
            continue
        component_view = labeled[obj_slice]
        area_px = int(np.count_nonzero(component_view == component_id))
        if area_px <= 0:
            continue
        bbox = (float(x1), float(y1), float(x2), float(y2))
        records.append(
            {
                "component_id": int(component_id),
                "bbox_xyxy": bbox,
                "area_px": area_px,
                "width_px": float(x2 - x1),
                "height_px": float(y2 - y1),
            }
        )
    return sorted(records, key=lambda item: int(item.get("area_px") or 0), reverse=True)


def _build_edge_strip_cluster_candidates(
    usable_components: Sequence[Dict[str, Any]],
    *,
    crop_width: int,
    crop_height: int,
    max_bbox_area_fraction: float,
    border_touch_area_fraction: float,
    edge_touch_tolerance_px: float = 6.0,
    edge_band_fraction: float = 0.08,
    edge_gap_fraction: float = 0.12,
) -> List[Dict[str, Any]]:
    if not usable_components:
        return []
    vertical_band_px = max(16.0, float(crop_width) * float(edge_band_fraction))
    horizontal_band_px = max(16.0, float(crop_height) * float(edge_band_fraction))
    vertical_gap_px = max(12.0, float(crop_height) * float(edge_gap_fraction))
    horizontal_gap_px = max(12.0, float(crop_width) * float(edge_gap_fraction))
    crop_area = max(1.0, float(crop_width * crop_height))

    def _eligible_for_edge(component: Dict[str, Any], edge_name: str) -> bool:
        bbox = tuple(float(v) for v in component.get("bbox_xyxy") or ())
        if len(bbox) < 4:
            return False
        x1, y1, x2, y2 = bbox
        if edge_name == "left":
            return x1 <= edge_touch_tolerance_px or x2 <= vertical_band_px
        if edge_name == "right":
            return x2 >= float(crop_width) - edge_touch_tolerance_px or x1 >= float(crop_width) - vertical_band_px
        if edge_name == "top":
            return y1 <= edge_touch_tolerance_px or y2 <= horizontal_band_px
        return y2 >= float(crop_height) - edge_touch_tolerance_px or y1 >= float(crop_height) - horizontal_band_px

    def _cluster_for_edge(edge_name: str) -> List[Dict[str, Any]]:
        edge_components = [item for item in usable_components if _eligible_for_edge(item, edge_name)]
        if len(edge_components) < 2:
            return []
        if edge_name in {"left", "right"}:
            axis_start_idx = 1
            axis_end_idx = 3
            ortho_start_idx = 0
            ortho_end_idx = 2
            gap_px = vertical_gap_px
            band_px = vertical_band_px
        else:
            axis_start_idx = 0
            axis_end_idx = 2
            ortho_start_idx = 1
            ortho_end_idx = 3
            gap_px = horizontal_gap_px
            band_px = horizontal_band_px
        edge_components = sorted(edge_components, key=lambda item: float(item["bbox_xyxy"][axis_start_idx]))
        clusters: List[List[Dict[str, Any]]] = []
        current_cluster: List[Dict[str, Any]] = []
        current_axis_end = 0.0
        cluster_ortho_start = 0.0
        cluster_ortho_end = 0.0
        for component in edge_components:
            bbox = tuple(float(v) for v in component["bbox_xyxy"])
            axis_start = bbox[axis_start_idx]
            axis_end = bbox[axis_end_idx]
            ortho_start = bbox[ortho_start_idx]
            ortho_end = bbox[ortho_end_idx]
            if not current_cluster:
                current_cluster = [component]
                current_axis_end = axis_end
                cluster_ortho_start = ortho_start
                cluster_ortho_end = ortho_end
                continue
            gap = max(0.0, axis_start - current_axis_end)
            union_ortho_start = min(cluster_ortho_start, ortho_start)
            union_ortho_end = max(cluster_ortho_end, ortho_end)
            union_ortho_span = max(0.0, union_ortho_end - union_ortho_start)
            if gap <= gap_px and union_ortho_span <= band_px:
                current_cluster.append(component)
                current_axis_end = max(current_axis_end, axis_end)
                cluster_ortho_start = union_ortho_start
                cluster_ortho_end = union_ortho_end
                continue
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            current_cluster = [component]
            current_axis_end = axis_end
            cluster_ortho_start = ortho_start
            cluster_ortho_end = ortho_end
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        out: List[Dict[str, Any]] = []
        seen_component_sets: set[Tuple[int, ...]] = set()
        for cluster in clusters:
            component_ids = tuple(sorted(int(item.get("component_id") or 0) for item in cluster))
            if component_ids in seen_component_sets:
                continue
            seen_component_sets.add(component_ids)
            x1 = min(float(item["bbox_xyxy"][0]) for item in cluster)
            y1 = min(float(item["bbox_xyxy"][1]) for item in cluster)
            x2 = max(float(item["bbox_xyxy"][2]) for item in cluster)
            y2 = max(float(item["bbox_xyxy"][3]) for item in cluster)
            bbox = (x1, y1, x2, y2)
            bbox_area_fraction = max(0.0, (x2 - x1) * (y2 - y1)) / crop_area
            border_touch_count = _bbox_border_touch_count(bbox, width=crop_width, height=crop_height)
            if bbox_area_fraction >= float(max_bbox_area_fraction):
                continue
            if border_touch_count >= 2 and bbox_area_fraction >= float(border_touch_area_fraction):
                continue
            out.append(
                {
                    "bbox_xyxy": bbox,
                    "area_px": int(sum(int(item.get("area_px") or 0) for item in cluster)),
                    "bbox_area_fraction": float(bbox_area_fraction),
                    "border_touch_count": int(border_touch_count),
                    "component_count": int(len(cluster)),
                    "component_ids": list(component_ids),
                    "derivation_mode": f"edge_strip_cluster_{edge_name}",
                    "degenerate": False,
                }
            )
        return out

    candidates: List[Dict[str, Any]] = []
    for edge_name in ("left", "right", "top", "bottom"):
        candidates.extend(_cluster_for_edge(edge_name))
    return candidates


def derive_mask_component_candidates(
    mask: np.ndarray,
    *,
    crop_width: int,
    crop_height: int,
    mode: str = "largest_component",
    max_components: int = 32,
    min_component_area_fraction: float = 0.00005,
    max_bbox_area_fraction: float = 0.85,
    border_touch_area_fraction: float = 0.05,
    cluster_gap_fraction: float = 0.02,
) -> Dict[str, Any]:
    crop_area = max(1.0, float(crop_width * crop_height))
    min_component_area_px = max(16.0, crop_area * float(min_component_area_fraction))
    raw_components = _mask_component_records(mask)
    usable: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for component in raw_components:
        bbox = tuple(float(v) for v in component.get("bbox_xyxy") or ())
        if len(bbox) < 4:
            dropped.append({**component, "drop_reason": "bbox_invalid"})
            continue
        bbox_area_fraction = (
            max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / crop_area
        )
        border_touch_count = _bbox_border_touch_count(bbox, width=crop_width, height=crop_height)
        enriched = {
            **component,
            "bbox_area_fraction": float(bbox_area_fraction),
            "border_touch_count": int(border_touch_count),
        }
        if float(component.get("area_px") or 0.0) < float(min_component_area_px):
            dropped.append({**enriched, "drop_reason": "component_too_small"})
            continue
        if float(component.get("width_px") or 0.0) < 3.0 or float(component.get("height_px") or 0.0) < 3.0:
            dropped.append({**enriched, "drop_reason": "bbox_too_thin"})
            continue
        if bbox_area_fraction >= float(max_bbox_area_fraction):
            dropped.append({**enriched, "drop_reason": "bbox_too_large"})
            continue
        if border_touch_count >= 2 and bbox_area_fraction >= float(border_touch_area_fraction):
            dropped.append({**enriched, "drop_reason": "border_hugging"})
            continue
        usable.append(enriched)

    if not usable:
        return {
            "candidates": [],
            "components": [dict(item) for item in raw_components],
            "dropped": [dict(item) for item in dropped],
        }

    if str(mode or "largest_component") == "largest_component":
        candidate_components = [usable[0]]
    elif str(mode or "largest_component") == "component_split":
        candidate_components = usable[: max(1, int(max_components))]
    else:
        diag = float((float(crop_width) ** 2 + float(crop_height) ** 2) ** 0.5)
        gap_threshold = max(1.0, diag * float(cluster_gap_fraction))
        clusters: List[List[Dict[str, Any]]] = []
        for component in usable:
            placed = False
            for cluster in clusters:
                if any(
                    _component_bbox_gap(component.get("bbox_xyxy") or (), other.get("bbox_xyxy") or ())
                    <= gap_threshold
                    for other in cluster
                ):
                    cluster.append(component)
                    placed = True
                    break
            if not placed:
                clusters.append([component])
        candidate_components = []
        for cluster in clusters:
            x1 = min(float(item["bbox_xyxy"][0]) for item in cluster)
            y1 = min(float(item["bbox_xyxy"][1]) for item in cluster)
            x2 = max(float(item["bbox_xyxy"][2]) for item in cluster)
            y2 = max(float(item["bbox_xyxy"][3]) for item in cluster)
            bbox = (x1, y1, x2, y2)
            bbox_area_fraction = max(0.0, (x2 - x1) * (y2 - y1)) / crop_area
            border_touch_count = _bbox_border_touch_count(bbox, width=crop_width, height=crop_height)
            candidate = {
                "component_ids": [int(item["component_id"]) for item in cluster],
                "bbox_xyxy": bbox,
                "area_px": int(sum(int(item.get("area_px") or 0) for item in cluster)),
                "width_px": float(x2 - x1),
                "height_px": float(y2 - y1),
                "bbox_area_fraction": float(bbox_area_fraction),
                "border_touch_count": int(border_touch_count),
                "component_count": int(len(cluster)),
            }
            if bbox_area_fraction >= float(max_bbox_area_fraction):
                dropped.append({**candidate, "drop_reason": "cluster_bbox_too_large"})
                continue
            if border_touch_count >= 2 and bbox_area_fraction >= float(border_touch_area_fraction):
                dropped.append({**candidate, "drop_reason": "cluster_border_hugging"})
                continue
            candidate_components.append(candidate)
        candidate_components = sorted(
            candidate_components,
            key=lambda item: int(item.get("area_px") or 0),
            reverse=True,
        )[: max(1, int(max_components))]

    edge_cluster_candidates = _build_edge_strip_cluster_candidates(
        usable,
        crop_width=crop_width,
        crop_height=crop_height,
        max_bbox_area_fraction=max_bbox_area_fraction,
        border_touch_area_fraction=border_touch_area_fraction,
    )

    candidates: List[Dict[str, Any]] = []
    seen_signatures: set[Tuple[Tuple[int, ...], Tuple[float, float, float, float], str]] = set()
    for item in candidate_components:
        bbox = tuple(float(v) for v in item.get("bbox_xyxy") or ())
        if len(bbox) < 4:
            continue
        candidate = {
            "bbox_xyxy": bbox,
            "area_px": int(item.get("area_px") or 0),
            "bbox_area_fraction": float(item.get("bbox_area_fraction") or 0.0),
            "border_touch_count": int(item.get("border_touch_count") or 0),
            "component_count": int(item.get("component_count") or 1),
            "component_ids": list(item.get("component_ids") or [int(item.get("component_id") or 0)]),
            "derivation_mode": str(item.get("derivation_mode") or mode or "largest_component"),
            "degenerate": False,
        }
        signature = (
            tuple(sorted(int(value) for value in candidate["component_ids"])),
            tuple(round(float(value), 3) for value in bbox),
            str(candidate["derivation_mode"]),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        candidates.append(candidate)
    for item in edge_cluster_candidates:
        bbox = tuple(float(v) for v in item.get("bbox_xyxy") or ())
        if len(bbox) < 4:
            continue
        candidate = {
            "bbox_xyxy": bbox,
            "area_px": int(item.get("area_px") or 0),
            "bbox_area_fraction": float(item.get("bbox_area_fraction") or 0.0),
            "border_touch_count": int(item.get("border_touch_count") or 0),
            "component_count": int(item.get("component_count") or 1),
            "component_ids": list(item.get("component_ids") or []),
            "derivation_mode": str(item.get("derivation_mode") or "edge_strip_cluster"),
            "degenerate": False,
        }
        signature = (
            tuple(sorted(int(value) for value in candidate["component_ids"])),
            tuple(round(float(value), 3) for value in bbox),
            str(candidate["derivation_mode"]),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        candidates.append(candidate)
    return {
        "candidates": candidates,
        "components": [dict(item) for item in raw_components],
        "dropped": [dict(item) for item in dropped],
    }
