from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple


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
