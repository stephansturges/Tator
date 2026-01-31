from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


def _view_cell_raw(
    *,
    base_img: Image.Image,
    cell_xyxy: Tuple[float, float, float, float],
    grid_cell: str,
    store_preloaded_fn: Callable[[str, Any, Any], None],
    default_variant_fn: Callable[[Any], Any],
) -> Dict[str, Any]:
    x1, y1, x2, y2 = cell_xyxy
    crop = base_img.crop((x1, y1, x2, y2))
    crop_np = np.asarray(crop)
    token = hashlib.md5(crop_np.tobytes()).hexdigest()
    store_preloaded_fn(token, crop_np, default_variant_fn(None))
    agent_view = {
        "grid_cell": str(grid_cell),
        "width": int(crop.width),
        "height": int(crop.height),
    }
    return {
        "image_token": token,
        "grid_cell": str(grid_cell),
        "width": int(crop.width),
        "height": int(crop.height),
        "__agent_view__": agent_view,
    }


def _view_cell_overlay(
    *,
    base_img: Image.Image,
    cell_xyxy: Tuple[float, float, float, float],
    grid_cell: str,
    clusters: Sequence[Mapping[str, Any]],
    labelmap: Sequence[str],
    label_colors_fn: Callable[[Sequence[str]], Dict[str, str]],
    label_prefixes_fn: Callable[[Sequence[str]], Dict[str, str]],
    render_overlay_fn: Callable[..., Image.Image],
    dot_radius: int,
    store_preloaded_fn: Callable[[str, Any, Any], None],
    default_variant_fn: Callable[[Any], Any],
) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
    overlay_img = base_img
    labels = list(labelmap)
    if clusters and not labels:
        labels = sorted(
            {
                str(cluster.get("label") or "").strip()
                for cluster in clusters
                if isinstance(cluster, dict) and cluster.get("label")
            }
        )
    if clusters:
        label_colors = label_colors_fn(labels)
        label_prefixes = label_prefixes_fn(labels)
        overlay_img = render_overlay_fn(
            base_img,
            clusters,
            label_colors,
            label_prefixes=label_prefixes,
            dot_radius=dot_radius,
        )
    x1, y1, x2, y2 = cell_xyxy
    crop = overlay_img.crop((x1, y1, x2, y2))
    crop_np = np.asarray(crop)
    token = hashlib.md5(crop_np.tobytes()).hexdigest()
    store_preloaded_fn(token, crop_np, default_variant_fn(None))
    agent_view = {
        "grid_cell": str(grid_cell),
        "width": int(crop.width),
        "height": int(crop.height),
    }
    payload = {
        "image_token": token,
        "grid_cell": str(grid_cell),
        "width": int(crop.width),
        "height": int(crop.height),
        "__agent_view__": agent_view,
    }
    return payload, overlay_img


def _view_full_overlay(
    *,
    base_img: Image.Image,
    clusters: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    label_colors_fn: Callable[[Sequence[str]], Dict[str, str]],
    label_prefixes_fn: Callable[[Sequence[str]], Dict[str, str]],
    render_overlay_fn: Callable[..., Image.Image],
    dot_radius: int,
    grid: Optional[Mapping[str, Any]],
    grid_usage: Mapping[str, Mapping[str, int]],
    grid_usage_last: Mapping[str, Mapping[str, float]],
    grid_usage_rows_fn: Callable[[Optional[Mapping[str, Any]], Mapping[str, Mapping[str, int]], Mapping[str, Mapping[str, float]]], Sequence[Dict[str, Any]]],
    grid_usage_text_fn: Callable[[Sequence[Dict[str, Any]]], str],
    overlay_key_fn: Callable[[Dict[str, str], Dict[str, str]], str],
) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
    overlay_img = base_img
    label_colors = label_colors_fn(labels) if labels else {}
    label_prefixes = label_prefixes_fn(labels) if labels else {}
    if clusters:
        overlay_img = render_overlay_fn(
            base_img,
            clusters,
            label_colors,
            dot_radius=dot_radius,
            label_prefixes=label_prefixes,
        )
    usage_rows = grid_usage_rows_fn(grid, grid_usage, grid_usage_last)
    usage_text = grid_usage_text_fn(usage_rows)
    agent_view = {
        "grid_usage": usage_rows,
        "grid_usage_text": usage_text,
        "total_cells": len(usage_rows),
        "overlay_key": overlay_key_fn(label_colors, label_prefixes),
    }
    payload = {
        "width": int(overlay_img.width),
        "height": int(overlay_img.height),
        "__agent_view__": agent_view,
    }
    return payload, overlay_img
