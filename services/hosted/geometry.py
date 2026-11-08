"""Box geometry rules shared by saving, importing, bridging and exporting.

Boxes are always clipped to the image: the part inside the image is kept and
the part outside is cut away (never shifted). A box with no area left inside
the image is empty; importers and exporters drop it, saving refuses it.
"""
from __future__ import annotations

import math


def clip_box(x: float, y: float, w: float, h: float, width: float, height: float) -> list[float] | None:
    """The pixel box ``[x, y, w, h]`` clipped to the image, or None when empty."""
    if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(float(width), x + w), min(float(height), y + h)
    clipped_w, clipped_h = right - left, bottom - top
    # Keep x + w inside the image even when float subtraction rounds outward.
    if left + clipped_w > width:
        clipped_w = math.nextafter(clipped_w, 0.0)
    if top + clipped_h > height:
        clipped_h = math.nextafter(clipped_h, 0.0)
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    return [left, top, clipped_w, clipped_h]


def clamp_polygon(points: list[list[float]], width: float, height: float) -> list[list[float]]:
    """Polygon points moved onto the image edge where they fall outside it."""
    return [[min(float(width), max(0.0, px)), min(float(height), max(0.0, py))] for px, py in points]
