from __future__ import annotations

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


def _remap_window_xyxy_to_full(xyxy: Sequence[float], window_xyxy: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, xyxy[:4])
    wx1, wy1, wx2, wy2 = map(float, window_xyxy[:4])
    return x1 + wx1, y1 + wy1, x2 + wx1, y2 + wy1


def _normalize_window_xyxy(window: Optional[Any], img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    if not window:
        return None
    if isinstance(window, (list, tuple)) and len(window) >= 4:
        x1, y1, x2, y2 = map(float, window[:4])
        return max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2)
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


def _yolo_to_xyxy(width: int, height: int, bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(bbox) < 4:
        return 0.0, 0.0, 0.0, 0.0
    cx, cy, bw, bh = map(float, bbox[:4])
    x1 = (cx - bw / 2.0) * float(width)
    y1 = (cy - bh / 2.0) * float(height)
    x2 = (cx + bw / 2.0) * float(width)
    y2 = (cy + bh / 2.0) * float(height)
    return x1, y1, x2, y2
