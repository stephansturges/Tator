from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from utils.coords import _qwen_bbox_to_xyxy


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


def _agent_grid_prompt_text(grid: Optional[Mapping[str, Any]]) -> str:
    if not grid:
        return ""
    cols = int(grid.get("cols") or 0)
    rows = int(grid.get("rows") or 0)
    labels = grid.get("col_labels") or []
    if not cols or not rows or not labels:
        return ""
    first = str(labels[0])
    last = str(labels[-1])
    cell_w = float(grid.get("cell_w") or 0.0)
    cell_h = float(grid.get("cell_h") or 0.0)
    return (
        f"Grid: columns {first}-{last}, rows 1-{rows}. "
        f"Cell size ~{cell_w:.0f}x{cell_h:.0f} px. "
        "Use grid_cell like C2 (column C, row 2, top-left origin) for windowed tools; "
        "do not use numeric coordinates when the grid is enabled."
    )


def _agent_quadrant_windows_qwen(overlap_ratio: float = 0.1) -> List[Dict[str, Any]]:
    overlap_ratio = max(0.0, min(float(overlap_ratio), 0.4))
    base = 1000.0
    if overlap_ratio <= 0.0:
        win = 500.0
    else:
        win = base / (2.0 - overlap_ratio)
    start = base - win
    x0 = 0.0
    y0 = 0.0
    x1 = max(0.0, min(base, start))
    y1 = max(0.0, min(base, start))
    win = max(1.0, min(base, win))
    windows = [
        {"name": "top_left", "bbox_2d": [x0, y0, x0 + win, y0 + win]},
        {"name": "top_right", "bbox_2d": [x1, y0, x1 + win, y0 + win]},
        {"name": "bottom_left", "bbox_2d": [x0, y1, x0 + win, y0 + win]},
        {"name": "bottom_right", "bbox_2d": [x1, y1, x1 + win, y1 + win]},
    ]
    clipped = []
    for window in windows:
        x1w, y1w, x2w, y2w = window["bbox_2d"]
        clipped.append(
            {
                "name": window["name"],
                "bbox_2d": [
                    max(0.0, min(base, x1w)),
                    max(0.0, min(base, y1w)),
                    max(0.0, min(base, x2w)),
                    max(0.0, min(base, y2w)),
                ],
            }
        )
    return clipped
