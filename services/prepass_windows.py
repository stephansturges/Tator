"""Window helpers for SAM3 text/similarity prepass runs."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from PIL import Image

from services.prepass_grid import _agent_grid_cell_xyxy, _agent_grid_cells, _agent_grid_spec_for_payload
from utils.coords import _resolve_agent_bbox_xyxy, _xyxy_to_qwen_bbox
from utils.image import _slice_image_sahi


def _agent_similarity_windows(
    payload: Any,
    *,
    pil_img: Image.Image,
    grid_overlap_ratio_default: float,
) -> List[Dict[str, Any]]:
    img_w, img_h = pil_img.size
    mode = (getattr(payload, "similarity_window_mode", None) or "grid").strip().lower()
    windows: List[Dict[str, Any]] = []
    if mode == "sahi":
        slice_size = int(getattr(payload, "similarity_window_size", None) or getattr(payload, "sahi_window_size", None) or 640)
        overlap = float(getattr(payload, "similarity_window_overlap", None) or getattr(payload, "sahi_overlap_ratio", None) or 0.2)
        _, starts = _slice_image_sahi(pil_img, slice_size, overlap)
        for idx, start in enumerate(starts):
            x1 = float(start[0])
            y1 = float(start[1])
            x2 = min(float(img_w), x1 + slice_size)
            y2 = min(float(img_h), y1 + slice_size)
            windows.append(
                {
                    "name": f"sahi_{idx}",
                    "bbox_xyxy_px": [x1, y1, x2, y2],
                    "bbox_2d": list(_xyxy_to_qwen_bbox(img_w, img_h, x1, y1, x2, y2)),
                }
            )
        return windows
    grid_spec = _agent_grid_spec_for_payload(payload, img_w, img_h)
    overlap_ratio = getattr(payload, "grid_overlap_ratio", None) or grid_overlap_ratio_default
    for cell in _agent_grid_cells(grid_spec):
        xyxy = _agent_grid_cell_xyxy(grid_spec, cell, overlap_ratio=overlap_ratio)
        if not xyxy:
            continue
        x1, y1, x2, y2 = xyxy
        windows.append(
            {
                "name": cell,
                "grid_cell": cell,
                "bbox_xyxy_px": [x1, y1, x2, y2],
                "bbox_2d": list(_xyxy_to_qwen_bbox(img_w, img_h, x1, y1, x2, y2)),
            }
        )
    return windows


def _agent_sam3_text_windows(
    payload: Any,
    *,
    pil_img: Image.Image,
    grid_overlap_ratio_default: float,
) -> List[Dict[str, Any]]:
    img_w, img_h = pil_img.size
    mode = (getattr(payload, "sam3_text_window_mode", None) or "grid").strip().lower()
    windows: List[Dict[str, Any]] = []
    if mode == "sahi":
        slice_size = int(getattr(payload, "sam3_text_window_size", None) or getattr(payload, "sahi_window_size", None) or 640)
        overlap = float(getattr(payload, "sam3_text_window_overlap", None) or getattr(payload, "sahi_overlap_ratio", None) or 0.2)
        _, starts = _slice_image_sahi(pil_img, slice_size, overlap)
        for idx, start in enumerate(starts):
            x1 = float(start[0])
            y1 = float(start[1])
            x2 = min(float(img_w), x1 + slice_size)
            y2 = min(float(img_h), y1 + slice_size)
            windows.append(
                {
                    "name": f"sahi_{idx}",
                    "bbox_xyxy_px": [x1, y1, x2, y2],
                    "bbox_2d": list(_xyxy_to_qwen_bbox(img_w, img_h, x1, y1, x2, y2)),
                }
            )
        return windows
    grid_spec = _agent_grid_spec_for_payload(payload, img_w, img_h)
    overlap_ratio = getattr(payload, "grid_overlap_ratio", None) or grid_overlap_ratio_default
    for cell in _agent_grid_cells(grid_spec):
        xyxy = _agent_grid_cell_xyxy(grid_spec, cell, overlap_ratio=overlap_ratio)
        if not xyxy:
            continue
        x1, y1, x2, y2 = xyxy
        windows.append(
            {
                "name": cell,
                "grid_cell": cell,
                "bbox_xyxy_px": [x1, y1, x2, y2],
                "bbox_2d": list(_xyxy_to_qwen_bbox(img_w, img_h, x1, y1, x2, y2)),
            }
        )
    return windows


def _agent_exemplars_for_window(
    exemplars: Sequence[Dict[str, Any]],
    *,
    img_w: int,
    img_h: int,
    window_xyxy: Sequence[float],
) -> List[Dict[str, Any]]:
    if not exemplars:
        return []
    if not window_xyxy or len(window_xyxy) < 4:
        return list(exemplars)
    wx1, wy1, wx2, wy2 = window_xyxy
    filtered: List[Dict[str, Any]] = []
    for det in exemplars:
        if not isinstance(det, dict):
            continue
        xyxy = _resolve_agent_bbox_xyxy(det, img_w, img_h)
        if xyxy is None:
            continue
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if wx1 <= cx <= wx2 and wy1 <= cy <= wy2:
            filtered.append(det)
    return filtered
