"""Readable logging helpers."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from services.prepass_grid import _agent_grid_cell_for_detection




def _agent_readable_write(
    line: str,
    *,
    writer: Optional[Callable[[str], None]] = None,
    to_console: bool = False,
    logger_name: str = "prepass.readable",
) -> None:
    if not line:
        return
    try:
        if writer is not None:
            writer(line)
        if to_console:
            logging.getLogger(logger_name).info(line)
    except Exception:
        return


def _agent_readable_banner(title: str, *, width: int = 88, fill: str = "=") -> str:
    cleaned = " ".join(str(title or "").strip().split())
    if not cleaned:
        return fill * width
    text = f" {cleaned} "
    if len(text) >= width:
        return text[:width]
    pad = width - len(text)
    left = pad // 2
    right = pad - left
    return f"{fill * left}{text}{fill * right}"


def _agent_detection_summary_lines(
    detections: Sequence[Dict[str, Any]],
    *,
    grid: Optional[Mapping[str, Any]] = None,
    img_w: int,
    img_h: int,
    warnings: Optional[Sequence[str]] = None,
    max_cells: int = 10,
) -> List[str]:
    total = len(detections)
    label_counts: Dict[str, int] = {}
    cell_counts: Dict[str, Dict[str, int]] = {}
    for det in detections:
        label = str(det.get("label") or det.get("class_name") or "").strip()
        if not label:
            continue
        label_counts[label] = label_counts.get(label, 0) + 1
        if grid:
            cell = _agent_grid_cell_for_detection(det, img_w, img_h, grid)
            if cell:
                cell_counts.setdefault(label, {})
                cell_counts[label][cell] = cell_counts[label].get(cell, 0) + 1
    lines: List[str] = []
    labels_total = len(label_counts)
    grid_state = "on" if grid else "off"
    warnings_text = ""
    if warnings:
        warnings_text = f" warnings={len(warnings)}"
    lines.append(f"summary: total={total} labels={labels_total} grid={grid_state}{warnings_text}")
    ordered = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
    for label, count in ordered:
        cell_text = ""
        cells = cell_counts.get(label) if grid else None
        if cells:
            cell_items = sorted(cells.items(), key=lambda item: (-item[1], item[0]))
            parts = [f"{cell}x{cnt}" for cell, cnt in cell_items[:max_cells]]
            if len(cell_items) > max_cells:
                parts.append(f"+{len(cell_items) - max_cells} more")
            cell_text = f" cells={','.join(parts)}"
        lines.append(f"label {label}: {count}{cell_text}")
    if warnings:
        warn_list = ", ".join(str(w) for w in warnings)
        lines.append(f"warnings: {warn_list}")
    return lines


def _agent_clean_observation_text(text: Optional[str], max_len: int = 160) -> str:
    if not text:
        return ""
    cleaned = " ".join(str(text).replace("\n", " ").replace("\t", " ").split())
    for prefix in ("observation:", "observations:", "note:", "notes:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3].rstrip() + "..."
    return cleaned


def _agent_readable_format_bbox(bbox: Any) -> str:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return ""
    values: List[str] = []
    for val in bbox[:4]:
        try:
            num = float(val)
        except (TypeError, ValueError):
            values.append("?")
            continue
        rounded = round(num)
        if abs(num - rounded) < 0.01:
            values.append(str(int(rounded)))
        else:
            values.append(f"{num:.1f}")
    return "[" + ", ".join(values) + "]"



