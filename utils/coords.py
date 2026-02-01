from __future__ import annotations

import json
from typing import Any, Optional, Sequence, Tuple, List


def _xyxy_to_qwen_bbox(width: int, height: int, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0, 0.0
    qx1 = max(0.0, min(1000.0, x1 / float(width) * 1000.0))
    qy1 = max(0.0, min(1000.0, y1 / float(height) * 1000.0))
    qx2 = max(0.0, min(1000.0, x2 / float(width) * 1000.0))
    qy2 = max(0.0, min(1000.0, y2 / float(height) * 1000.0))
    if qx1 > qx2:
        qx1, qx2 = qx2, qx1
    if qy1 > qy2:
        qy1, qy2 = qy2, qy1
    return qx1, qy1, qx2, qy2


def _qwen_bbox_to_xyxy(width: int, height: int, bbox_2d: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(bbox_2d) < 4 or width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0, 0.0
    qx1, qy1, qx2, qy2 = map(float, bbox_2d[:4])
    qx1 = max(0.0, min(1000.0, qx1))
    qy1 = max(0.0, min(1000.0, qy1))
    qx2 = max(0.0, min(1000.0, qx2))
    qy2 = max(0.0, min(1000.0, qy2))
    if qx1 > qx2:
        qx1, qx2 = qx2, qx1
    if qy1 > qy2:
        qy1, qy2 = qy2, qy1
    x1 = qx1 / 1000.0 * float(width)
    y1 = qy1 / 1000.0 * float(height)
    x2 = qx2 / 1000.0 * float(width)
    y2 = qy2 / 1000.0 * float(height)
    return x1, y1, x2, y2


def _normalize_window_xyxy(window: Optional[Any], img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    if not window:
        return None
    if isinstance(window, (list, tuple)) and len(window) >= 4:
        x1, y1, x2, y2 = map(float, window[:4])
        return max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2)


def _extract_numeric_sequence(value: Any, *, length: int) -> Optional[List[float]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return None
    numbers: List[float] = []
    for idx in range(length):
        try:
            numbers.append(float(value[idx]))
        except (TypeError, ValueError):
            return None
    return numbers


def _scale_coord(value: float, src: int, dst: int) -> float:
    if src <= 0:
        return float(value)
    return float(value) * (float(dst) / float(src))


def _scale_bbox_to_image(
    bbox: List[float],
    proc_w: int,
    proc_h: int,
    full_w: int,
    full_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    if len(bbox) < 4:
        return None
    left = _scale_coord(bbox[0], proc_w, full_w)
    top = _scale_coord(bbox[1], proc_h, full_h)
    right = _scale_coord(bbox[2], proc_w, full_w)
    bottom = _scale_coord(bbox[3], proc_h, full_h)
    left_i = max(0, min(full_w, int(round(left))))
    top_i = max(0, min(full_h, int(round(top))))
    right_i = max(0, min(full_w, int(round(right))))
    bottom_i = max(0, min(full_h, int(round(bottom))))
    if right_i <= left_i or bottom_i <= top_i:
        return None
    return left_i, top_i, right_i, bottom_i


def _scale_point_to_image(
    point: List[float],
    proc_w: int,
    proc_h: int,
    full_w: int,
    full_h: int,
) -> Optional[Tuple[float, float]]:
    if len(point) < 2:
        return None
    x = _scale_coord(point[0], proc_w, full_w)
    y = _scale_coord(point[1], proc_h, full_h)
    x = float(min(max(x, 0.0), float(full_w)))
    y = float(min(max(y, 0.0), float(full_h)))
    return x, y
    if isinstance(window, dict):
        if "bbox_2d" in window:
            x1, y1, x2, y2 = _qwen_bbox_to_xyxy(img_w, img_h, window.get("bbox_2d") or [])
            return max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2)
        if all(k in window for k in ("x1", "y1", "x2", "y2")):
            x1 = float(window.get("x1"))
            y1 = float(window.get("y1"))
            x2 = float(window.get("x2"))
            y2 = float(window.get("y2"))
            return max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2)
    return None


def _window_bbox_2d_to_full_xyxy(
    img_w: int,
    img_h: int,
    window_bbox_2d: Optional[Sequence[float]],
) -> Optional[Tuple[float, float, float, float]]:
    if not window_bbox_2d:
        return None
    x1, y1, x2, y2 = _qwen_bbox_to_xyxy(img_w, img_h, window_bbox_2d)
    return x1, y1, x2, y2


def _window_local_bbox_2d_to_full_xyxy(
    img_w: int,
    img_h: int,
    window_bbox_2d: Optional[Sequence[float]],
    local_bbox_2d: Optional[Sequence[float]],
) -> Optional[Tuple[float, float, float, float]]:
    if not window_bbox_2d or not local_bbox_2d:
        return None
    window_xyxy = _window_bbox_2d_to_full_xyxy(img_w, img_h, window_bbox_2d)
    if not window_xyxy:
        return None
    wx1, wy1, wx2, wy2 = window_xyxy
    win_w = max(1.0, wx2 - wx1)
    win_h = max(1.0, wy2 - wy1)
    lx1, ly1, lx2, ly2 = _qwen_bbox_to_xyxy(int(win_w), int(win_h), local_bbox_2d)
    return lx1 + wx1, ly1 + wy1, lx2 + wx1, ly2 + wy1


def _window_local_xyxy_to_full_xyxy(
    window_xyxy: Optional[Tuple[float, float, float, float]],
    local_xyxy: Optional[Sequence[float]],
) -> Optional[Tuple[float, float, float, float]]:
    if not window_xyxy or not local_xyxy:
        return None
    wx1, wy1, _, _ = window_xyxy
    x1, y1, x2, y2 = map(float, local_xyxy[:4])
    return x1 + wx1, y1 + wy1, x2 + wx1, y2 + wy1


def _resolve_agent_bbox_xyxy(
    ann: Dict[str, Any],
    img_w: int,
    img_h: int,
    *,
    window_bbox_2d: Optional[Sequence[float]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    bbox_space = str(ann.get("bbox_space") or "full").strip().lower()
    if bbox_space == "window":
        window_xyxy = _window_bbox_2d_to_full_xyxy(img_w, img_h, window_bbox_2d)
        if window_xyxy is None:
            return None
        if "bbox_2d" in ann:
            return _window_local_bbox_2d_to_full_xyxy(img_w, img_h, window_bbox_2d, ann.get("bbox_2d"))
        if "bbox_xyxy_px" in ann:
            coords = ann.get("bbox_xyxy_px")
            if isinstance(coords, (list, tuple)) and len(coords) >= 4:
                return _window_local_xyxy_to_full_xyxy(window_xyxy, coords)
            return None
        return None
    if "bbox_xyxy_px" in ann:
        try:
            coords = ann.get("bbox_xyxy_px")
            if isinstance(coords, (list, tuple)) and len(coords) >= 4:
                return tuple(map(float, coords[:4]))  # type: ignore[return-value]
            return None
        except Exception:
            return None
    if "bbox_2d" in ann:
        return _qwen_bbox_to_xyxy(img_w, img_h, ann.get("bbox_2d") or [])
    return None


def _agent_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    except Exception:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _agent_round_bbox_2d(bbox: Any) -> Optional[List[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        return [round(float(v), 1) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None


def _agent_clip_xyxy(
    xyxy: Optional[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    if not xyxy:
        return None
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(img_w), float(x1)))
    y1 = max(0.0, min(float(img_h), float(y1)))
    x2 = max(0.0, min(float(img_w), float(x2)))
    y2 = max(0.0, min(float(img_h), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _agent_expand_window_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    min_size: int,
) -> Tuple[Tuple[float, float, float, float], bool]:
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    target_w = max(width, float(min_size))
    target_h = max(height, float(min_size))
    if target_w > img_w:
        target_w = float(img_w)
    if target_h > img_h:
        target_h = float(img_h)
    expanded = target_w > width or target_h > height
    if not expanded:
        return (x1, y1, x2, y2), False
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nx1 = cx - target_w / 2.0
    nx2 = cx + target_w / 2.0
    ny1 = cy - target_h / 2.0
    ny2 = cy + target_h / 2.0
    if nx1 < 0.0:
        nx2 -= nx1
        nx1 = 0.0
    if nx2 > img_w:
        nx1 -= nx2 - img_w
        nx2 = float(img_w)
    if ny1 < 0.0:
        ny2 -= ny1
        ny1 = 0.0
    if ny2 > img_h:
        ny1 -= ny2 - img_h
        ny2 = float(img_h)
    nx1 = max(0.0, min(float(img_w), nx1))
    nx2 = max(0.0, min(float(img_w), nx2))
    ny1 = max(0.0, min(float(img_h), ny1))
    ny2 = max(0.0, min(float(img_h), ny2))
    return (nx1, ny1, nx2, ny2), True


def _agent_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> List[float]:
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def _xyxy_to_yolo_norm(
    width: int,
    height: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0, 0.0
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return (
        cx / float(width),
        cy / float(height),
        w / float(width),
        h / float(height),
    )


def _agent_det_payload(
    img_w: int,
    img_h: int,
    xyxy: Tuple[float, float, float, float],
    *,
    label: Optional[str],
    class_id: Optional[int],
    score: Optional[float],
    source: str,
    window: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    x1, y1, x2, y2 = xyxy
    bbox_xywh = _agent_xyxy_to_xywh(x1, y1, x2, y2)
    bbox_2d = _xyxy_to_qwen_bbox(img_w, img_h, x1, y1, x2, y2)
    payload = {
        "bbox_2d": list(bbox_2d),
        "bbox_xyxy_px": [float(x1), float(y1), float(x2), float(y2)],
        "bbox_xywh_px": bbox_xywh,
        "bbox_yolo": list(_xyxy_to_yolo_norm(img_w, img_h, x1, y1, x2, y2)),
        "label": label,
        "class_id": class_id,
        "score": score,
        "score_source": source if score is not None else "unknown",
        "source": source,
        "bbox_space": "full",
    }
    if window:
        payload["window_xyxy_px"] = [float(v) for v in window]
        payload["window_bbox_2d"] = list(_xyxy_to_qwen_bbox(img_w, img_h, *window))
    return payload


def _yolo_to_xyxy(width: int, height: int, bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(bbox) < 4:
        return 0.0, 0.0, 0.0, 0.0
    cx, cy, bw, bh = map(float, bbox[:4])
    x1 = (cx - bw / 2.0) * float(width)
    y1 = (cy - bh / 2.0) * float(height)
    x2 = (cx + bw / 2.0) * float(width)
    y2 = (cy + bh / 2.0) * float(height)
    return x1, y1, x2, y2
