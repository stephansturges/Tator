"""Overlay image tooling helpers."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from PIL import Image


def _agent_overlay_base_image(
    *,
    grid_image: Optional[Image.Image],
    image_base64: Optional[str],
    image_token: Optional[str],
    image_resolver: Callable[[Optional[str], Optional[str]], Tuple[Image.Image, Any, str]],
) -> Optional[Image.Image]:
    if grid_image is not None:
        return grid_image
    if image_base64 or image_token:
        base_img, _, _ = image_resolver(image_base64, image_token)
        return base_img
    return None


def _agent_overlay_crop_xyxy(
    tool_args: Mapping[str, Any],
    tool_result: Any,
    img_w: int,
    img_h: int,
    *,
    grid: Optional[Mapping[str, Any]],
    cluster_index: Mapping[int, Mapping[str, Any]],
    grid_cell_xyxy_fn: Callable[[Mapping[str, Any], str, float], Optional[Tuple[float, float, float, float]]],
    clip_xyxy_fn: Callable[[Optional[Tuple[float, float, float, float]], int, int], Optional[Tuple[float, float, float, float]]],
    qwen_bbox_to_xyxy_fn: Callable[[int, int, Sequence[float]], Tuple[float, float, float, float]],
    window_local_bbox_fn: Callable[[int, int, Sequence[float], Sequence[float]], Optional[Tuple[float, float, float, float]]],
    grid_overlap_ratio: float,
) -> Optional[Tuple[float, float, float, float]]:
    agent_view = tool_result.get("__agent_view__") if isinstance(tool_result, dict) else None
    grid_cell = tool_args.get("grid_cell") or (
        agent_view.get("grid_cell") if isinstance(agent_view, dict) else None
    )
    if grid_cell and grid:
        cell_xyxy = grid_cell_xyxy_fn(grid, str(grid_cell), grid_overlap_ratio)
        return clip_xyxy_fn(cell_xyxy, img_w, img_h)

    window_xyxy = None
    for source in (tool_result, tool_args):
        if isinstance(source, dict):
            win = source.get("window_xyxy_px")
            if isinstance(win, (list, tuple)) and len(win) >= 4:
                window_xyxy = tuple(float(v) for v in win[:4])
                break
    if window_xyxy:
        return clip_xyxy_fn(window_xyxy, img_w, img_h)

    window_bbox_2d = None
    if isinstance(tool_result, dict):
        window_bbox_2d = tool_result.get("window_bbox_2d")
    if window_bbox_2d is None:
        window_bbox_2d = tool_args.get("window_bbox_2d")
    if isinstance(window_bbox_2d, (list, tuple)) and len(window_bbox_2d) >= 4:
        return clip_xyxy_fn(qwen_bbox_to_xyxy_fn(img_w, img_h, window_bbox_2d), img_w, img_h)

    window_arg = tool_args.get("window")
    if isinstance(window_arg, dict):
        if isinstance(window_arg.get("bbox_xyxy_px"), (list, tuple)) and len(window_arg.get("bbox_xyxy_px")) >= 4:
            xyxy = tuple(float(v) for v in window_arg.get("bbox_xyxy_px")[:4])
            return clip_xyxy_fn(xyxy, img_w, img_h)
        if isinstance(window_arg.get("bbox_2d"), (list, tuple)) and len(window_arg.get("bbox_2d")) >= 4:
            return clip_xyxy_fn(qwen_bbox_to_xyxy_fn(img_w, img_h, window_arg.get("bbox_2d")), img_w, img_h)

    cluster_id = tool_args.get("cluster_id")
    if cluster_id is not None:
        cluster = cluster_index.get(int(cluster_id))
        if cluster:
            bbox_xyxy = cluster.get("bbox_xyxy_px")
            if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) >= 4:
                xyxy = tuple(float(v) for v in bbox_xyxy[:4])
                return clip_xyxy_fn(xyxy, img_w, img_h)

    bbox_2d = tool_args.get("bbox_2d")
    bbox_space = str(tool_args.get("bbox_space") or "full").strip().lower()
    if isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) >= 4:
        if bbox_space == "window":
            window_bbox_2d = tool_args.get("window_bbox_2d")
            if isinstance(window_bbox_2d, (list, tuple)) and len(window_bbox_2d) >= 4:
                xyxy = window_local_bbox_fn(img_w, img_h, window_bbox_2d, bbox_2d)
                return clip_xyxy_fn(xyxy, img_w, img_h) if xyxy else None
        return clip_xyxy_fn(qwen_bbox_to_xyxy_fn(img_w, img_h, bbox_2d), img_w, img_h)

    bbox_xyxy_px = tool_args.get("bbox_xyxy_px")
    if isinstance(bbox_xyxy_px, (list, tuple)) and len(bbox_xyxy_px) >= 4:
        xyxy = tuple(float(v) for v in bbox_xyxy_px[:4])
        return clip_xyxy_fn(xyxy, img_w, img_h)

    return None
