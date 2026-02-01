from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from utils.coords import _qwen_bbox_to_xyxy, _yolo_to_xyxy


def _agent_overlay_labels(
    clusters: Sequence[Dict[str, Any]],
    labelmap: Optional[Sequence[str]] = None,
) -> List[str]:
    labels = list(labelmap or [])
    if labels:
        return labels
    label_set = {
        str(cluster.get("label") or "").strip()
        for cluster in clusters
        if isinstance(cluster, dict) and cluster.get("label")
    }
    return sorted(label for label in label_set if label)


def _agent_detection_center_px(det: Dict[str, Any], img_w: int, img_h: int) -> Optional[Tuple[float, float]]:
    bbox = det.get("bbox_xyxy_px")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bbox_2d = det.get("bbox_2d")
    if isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) >= 4:
        x1, y1, x2, y2 = _qwen_bbox_to_xyxy(img_w, img_h, bbox_2d)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bbox_yolo = det.get("bbox_yolo")
    if isinstance(bbox_yolo, (list, tuple)) and len(bbox_yolo) >= 4:
        x1, y1, x2, y2 = _yolo_to_xyxy(img_w, img_h, bbox_yolo)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return None


def _agent_render_detection_overlay(
    pil_img: Image.Image,
    detections: Sequence[Dict[str, Any]],
    label_colors: Mapping[str, str],
    *,
    dot_radius: Optional[int] = None,
    label_prefixes: Optional[Mapping[str, str]] = None,
) -> Image.Image:
    if not detections:
        return pil_img
    overlay = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    img_w, img_h = overlay.size
    if dot_radius is None or dot_radius <= 0:
        dot_radius = max(2, int(round(min(img_w, img_h) * 0.004)))
    try:
        id_font = ImageFont.load_default()
    except Exception:
        id_font = None
    for det in detections:
        if not isinstance(det, dict):
            continue
        center = _agent_detection_center_px(det, img_w, img_h)
        if not center:
            continue
        cx, cy = center
        label = str(det.get("label") or det.get("class_name") or "").strip()
        color = label_colors.get(label, "#FFFFFF")
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            outline = "#000000" if luminance > 140 else "#FFFFFF"
        except Exception:
            outline = "#000000"
        draw.ellipse(
            (cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius),
            fill=color,
            outline=outline,
            width=1,
        )
        cluster_id = det.get("cluster_id") or det.get("candidate_id")
        if cluster_id is not None:
            text = str(cluster_id)
            if label_prefixes is not None and label:
                prefix = label_prefixes.get(label)
                if prefix:
                    text = f"{prefix}{cluster_id}"
            tx = min(max(0, int(cx + dot_radius + 2)), img_w - 1)
            ty = min(max(0, int(cy - dot_radius - 2)), img_h - 1)
            draw.text((tx + 1, ty + 1), text, fill="#000000", font=id_font)
            draw.text((tx, ty), text, fill=outline, font=id_font)
    return overlay


def _agent_render_grid_overlay(
    pil_img: Image.Image,
    grid: Mapping[str, Any],
    *,
    line_color: Tuple[int, int, int, int] = (255, 255, 255, 200),
    text_color: Tuple[int, int, int, int] = (255, 255, 255, 90),
) -> Image.Image:
    if not grid:
        return pil_img
    cols = int(grid.get("cols") or 0)
    rows = int(grid.get("rows") or 0)
    if cols <= 0 or rows <= 0:
        return pil_img
    cell_w = float(grid.get("cell_w") or 0.0)
    cell_h = float(grid.get("cell_h") or 0.0)
    labels = grid.get("col_labels") or []
    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    img_w, img_h = base.size
    for col in range(1, cols):
        x = int(round(col * cell_w))
        draw.line([(x, 0), (x, img_h)], fill=line_color, width=1)
    for row in range(1, rows):
        y = int(round(row * cell_h))
        draw.line([(0, y), (img_w, y)], fill=line_color, width=1)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if labels:
        for col_idx, label in enumerate(labels):
            x = int(round(col_idx * cell_w + 4))
            draw.text((x, 2), str(label), fill=text_color, font=font)
    for row_idx in range(rows):
        y = int(round(row_idx * cell_h + 2))
        draw.text((2, y), str(row_idx + 1), fill=text_color, font=font)
    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")
