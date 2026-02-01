from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from utils.coords import _qwen_bbox_to_xyxy
from utils.coords import _xyxy_to_qwen_bbox
from utils.overlay import _agent_detection_center_px
from utils.labels import _agent_fuzzy_align_label


def _agent_grid_col_label(index: int) -> str:
    label = ""
    idx = int(index) + 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def _agent_grid_col_index(label: str) -> Optional[int]:
    if not label:
        return None
    label = "".join(ch for ch in label.strip().upper() if "A" <= ch <= "Z")
    if not label:
        return None
    idx = 0
    for ch in label:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _agent_grid_spec(
    img_w: int,
    img_h: int,
    *,
    target: int = 200,
    min_cell: int = 160,
    max_cell: int = 260,
) -> Dict[str, Any]:
    def _fit(count: int, length: int) -> Tuple[int, float]:
        count = max(1, int(count))
        while count > 1 and length / count < min_cell:
            count -= 1
        while length / count > max_cell:
            count += 1
        return count, length / count

    cols_guess = max(1, int(round(img_w / float(target))))
    rows_guess = max(1, int(round(img_h / float(target))))
    cols, cell_w = _fit(cols_guess, img_w)
    rows, cell_h = _fit(rows_guess, img_h)
    col_labels = [_agent_grid_col_label(idx) for idx in range(cols)]
    return {
        "cols": cols,
        "rows": rows,
        "cell_w": float(cell_w),
        "cell_h": float(cell_h),
        "col_labels": col_labels,
        "img_w": int(img_w),
        "img_h": int(img_h),
    }


def _agent_grid_spec_for_payload(payload: Any, img_w: int, img_h: int) -> Dict[str, Any]:
    cols = getattr(payload, "grid_cols", None)
    rows = getattr(payload, "grid_rows", None)
    try:
        cols = int(cols) if cols is not None else None
    except (TypeError, ValueError):
        cols = None
    try:
        rows = int(rows) if rows is not None else None
    except (TypeError, ValueError):
        rows = None
    if cols and cols < 1:
        cols = None
    if rows and rows < 1:
        rows = None
    if cols or rows:
        if not cols and rows:
            cell_h = float(img_h) / float(rows)
            cols = max(1, int(round(float(img_w) / cell_h)))
        if not rows and cols:
            cell_w = float(img_w) / float(cols)
            rows = max(1, int(round(float(img_h) / cell_w)))
        cols = max(1, int(cols or 1))
        rows = max(1, int(rows or 1))
        cell_w = float(img_w) / float(cols)
        cell_h = float(img_h) / float(rows)
        col_labels = [_agent_grid_col_label(idx) for idx in range(cols)]
        return {
            "cols": cols,
            "rows": rows,
            "cell_w": float(cell_w),
            "cell_h": float(cell_h),
            "col_labels": col_labels,
            "img_w": int(img_w),
            "img_h": int(img_h),
        }
    return _agent_grid_spec(img_w, img_h)


def _agent_grid_cell_xyxy(
    grid: Mapping[str, Any],
    cell_label: str,
    *,
    overlap_ratio: float = 0.0,
) -> Optional[Tuple[float, float, float, float]]:
    if not grid or not cell_label:
        return None
    text = str(cell_label).strip()
    match = re.search(r"([A-Za-z]+)\\s*(\\d+)", text)
    col_label = None
    row_text = None
    if match:
        col_label, row_text = match.groups()
    else:
        match = re.search(r"(\\d+)\\s*([A-Za-z]+)", text)
        if match:
            row_text, col_label = match.groups()
    if not col_label or not row_text:
        return None
    col_idx = _agent_grid_col_index(col_label)
    try:
        row_idx = int(row_text) - 1
    except ValueError:
        return None
    cols = int(grid.get("cols") or 0)
    rows = int(grid.get("rows") or 0)
    if col_idx is None or col_idx < 0 or col_idx >= cols or row_idx < 0 or row_idx >= rows:
        return None
    cell_w = float(grid.get("cell_w") or 0.0)
    cell_h = float(grid.get("cell_h") or 0.0)
    img_w = int(grid.get("img_w") or 0)
    img_h = int(grid.get("img_h") or 0)
    x1 = col_idx * cell_w
    y1 = row_idx * cell_h
    x2 = img_w if col_idx == cols - 1 else (col_idx + 1) * cell_w
    y2 = img_h if row_idx == rows - 1 else (row_idx + 1) * cell_h
    ratio = max(0.0, float(overlap_ratio or 0.0))
    if ratio > 0.0:
        expand_x = (x2 - x1) * ratio * 0.5
        expand_y = (y2 - y1) * ratio * 0.5
        x1 = max(0.0, x1 - expand_x)
        y1 = max(0.0, y1 - expand_y)
        x2 = min(float(img_w), x2 + expand_x)
        y2 = min(float(img_h), y2 + expand_y)
    return (x1, y1, x2, y2)


def _agent_grid_cell_for_window_bbox(
    grid: Mapping[str, Any],
    window_bbox_2d: Sequence[float],
) -> Optional[str]:
    if not grid or not window_bbox_2d or len(window_bbox_2d) < 4:
        return None
    img_w = int(grid.get("img_w") or 0)
    img_h = int(grid.get("img_h") or 0)
    if img_w <= 0 or img_h <= 0:
        return None
    x1, y1, x2, y2 = _qwen_bbox_to_xyxy(img_w, img_h, window_bbox_2d)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    cell_w = float(grid.get("cell_w") or 0.0)
    cell_h = float(grid.get("cell_h") or 0.0)
    cols = int(grid.get("cols") or 0)
    rows = int(grid.get("rows") or 0)
    if cell_w <= 0 or cell_h <= 0 or cols <= 0 or rows <= 0:
        return None
    col_idx = min(cols - 1, max(0, int(cx / cell_w)))
    row_idx = min(rows - 1, max(0, int(cy / cell_h)))
    col_labels = grid.get("col_labels") or []
    if col_idx >= len(col_labels):
        return None
    return f"{col_labels[col_idx]}{row_idx + 1}"


def _agent_grid_cells(grid: Optional[Mapping[str, Any]]) -> List[str]:
    if not grid:
        return []
    labels = list(grid.get("col_labels") or [])
    rows = int(grid.get("rows") or 0)
    if not labels or rows <= 0:
        return []
    cells: List[str] = []
    for row in range(1, rows + 1):
        for col in labels:
            cells.append(f"{col}{row}")
    return cells


def _agent_tool_grid_cell_from_args(
    tool_args: Mapping[str, Any],
    tool_result: Any,
    *,
    grid: Optional[Mapping[str, Any]],
    cluster_index: Mapping[int, Mapping[str, Any]],
) -> Optional[str]:
    if not grid:
        return None
    grid_cell = tool_args.get("grid_cell")
    if grid_cell:
        return str(grid_cell)
    cluster_id = tool_args.get("cluster_id")
    if cluster_id is not None:
        cluster = cluster_index.get(int(cluster_id))
        if cluster and cluster.get("grid_cell"):
            return str(cluster.get("grid_cell"))
    if isinstance(tool_result, dict):
        agent_view = tool_result.get("__agent_view__")
        if isinstance(agent_view, dict) and agent_view.get("grid_cell"):
            return str(agent_view.get("grid_cell"))
    window_bbox_2d = tool_args.get("window_bbox_2d")
    if isinstance(window_bbox_2d, (list, tuple)) and len(window_bbox_2d) >= 4:
        return _agent_grid_cell_for_window_bbox(grid, window_bbox_2d)
    window_arg = tool_args.get("window")
    if isinstance(window_arg, dict):
        window_bbox_2d = window_arg.get("bbox_2d")
        if isinstance(window_bbox_2d, (list, tuple)) and len(window_bbox_2d) >= 4:
            return _agent_grid_cell_for_window_bbox(grid, window_bbox_2d)
    return None


def _agent_record_grid_tool_usage(
    tool_name: str,
    tool_args: Mapping[str, Any],
    tool_result: Any,
    *,
    grid: Optional[Mapping[str, Any]],
    cluster_index: Mapping[int, Mapping[str, Any]],
    usage: Dict[str, Dict[str, int]],
    usage_last: Dict[str, Dict[str, Any]],
    track_tools: Optional[Sequence[str]] = None,
) -> None:
    if not grid:
        return
    if track_tools is None:
        track_tools = (
            "look_and_inspect",
            "classify_crop",
            "view_cell_raw",
            "view_cell_overlay",
        )
    if tool_name not in set(track_tools):
        return
    cell = _agent_tool_grid_cell_from_args(
        tool_args, tool_result, grid=grid, cluster_index=cluster_index
    )
    if not cell:
        return
    cell_usage = usage.setdefault(cell, {})
    cell_usage[tool_name] = int(cell_usage.get(tool_name, 0)) + 1
    usage_last[cell] = {"tool": tool_name, "ts": time.time()}


def _agent_grid_cell_for_detection(
    det: Mapping[str, Any],
    img_w: int,
    img_h: int,
    grid: Optional[Mapping[str, Any]],
) -> Optional[str]:
    if not grid:
        return None
    cell_hint = det.get("grid_cell") if isinstance(det, Mapping) else None
    if cell_hint:
        return str(cell_hint)
    center = _agent_detection_center_px(dict(det), img_w, img_h)
    if not center:
        return None
    cx, cy = center
    bbox_2d = _xyxy_to_qwen_bbox(img_w, img_h, cx, cy, cx, cy)
    return _agent_grid_cell_for_window_bbox(grid, bbox_2d)


def _agent_grid_usage_rows(
    grid: Optional[Mapping[str, Any]],
    tool_usage: Mapping[str, Dict[str, int]],
    tool_last: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cells = _agent_grid_cells(grid)
    rows: List[Dict[str, Any]] = []
    for cell in cells:
        tool_counts = dict(tool_usage.get(cell, {}))
        total = sum(int(v) for v in tool_counts.values())
        last = tool_last.get(cell, {})
        rows.append(
            {
                "grid_cell": cell,
                "total_calls": total,
                "tools": tool_counts,
                "last_tool": last.get("tool"),
                "last_ts": last.get("ts"),
            }
        )
    return rows


def _agent_grid_usage_text(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    tool_short = {
        "look_and_inspect": "inspect",
        "image_zoom_in_tool": "zoom",
        "zoom_and_detect": "zoom_detect",
        "run_detector": "detector",
        "sam3_text": "sam3_text",
        "sam3_similarity": "sam3_sim",
        "qwen_infer": "qwen_infer",
        "classify_crop": "classify",
        "view_cell_raw": "view_raw",
        "view_cell_overlay": "view_overlay",
    }
    parts: List[str] = []
    for row in rows:
        cell = row.get("grid_cell")
        tools = row.get("tools") or {}
        last_tool = row.get("last_tool")
        if not tools:
            suffix = f" last={last_tool}" if last_tool else ""
            parts.append(f"{cell}: none{suffix}")
            continue
        tool_bits = []
        for tool_name, count in sorted(tools.items()):
            short = tool_short.get(tool_name, tool_name)
            tool_bits.append(f"{short}={int(count)}")
        if last_tool:
            tool_bits.append(f"last={last_tool}")
        parts.append(f"{cell}: " + ",".join(tool_bits))
    return "; ".join(parts)


def _agent_grid_label_counts(
    *,
    grid: Optional[Mapping[str, Any]],
    clusters: Sequence[Dict[str, Any]],
    label: Optional[str] = None,
    labelmap: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    if not grid:
        return []
    label_filter = _agent_fuzzy_align_label(label, labelmap or []) if label else None
    counts: Dict[str, Dict[str, Any]] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_label = str(cluster.get("label") or "").strip()
        if label_filter and cluster_label != label_filter:
            continue
        cell = cluster.get("owner_cell") or cluster.get("grid_cell")
        if not cell:
            bbox_2d = cluster.get("bbox_2d")
            if isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) >= 4:
                cell = _agent_grid_cell_for_window_bbox(grid, bbox_2d)
        if not cell:
            continue
        entry = counts.setdefault(str(cell), {"grid_cell": str(cell), "counts": {}, "cluster_ids": []})
        entry["cluster_ids"].append(int(cluster.get("cluster_id")))
        entry["counts"][cluster_label] = entry["counts"].get(cluster_label, 0) + 1
    summary: List[Dict[str, Any]] = []
    for cell, entry in sorted(counts.items()):
        counts_map = entry.get("counts") or {}
        total = sum(int(v) for v in counts_map.values())
        summary.append(
            {
                "grid_cell": cell,
                "total": total,
                "counts": counts_map,
                "cluster_ids": entry.get("cluster_ids") or [],
            }
        )
    return summary
