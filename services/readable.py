from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from services.prepass_grid import _agent_grid_cell_for_detection


def _agent_readable_trim(text: Optional[str], max_len: Optional[int] = None) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    if max_len is None or max_len <= 0 or len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


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


def _agent_readable_detection_line(
    det: Mapping[str, Any],
    *,
    grid: Optional[Mapping[str, Any]] = None,
    img_w: int,
    img_h: int,
) -> str:
    label = str(det.get("label") or det.get("class_name") or "unknown")
    bbox = None
    if isinstance(det.get("bbox_xyxy_px"), (list, tuple)) and len(det.get("bbox_xyxy_px")) >= 4:
        bbox = det.get("bbox_xyxy_px")
    elif isinstance(det.get("bbox_2d"), (list, tuple)) and len(det.get("bbox_2d")) >= 4:
        bbox = det.get("bbox_2d")
    bbox_text = _agent_readable_format_bbox(bbox) if bbox else "[]"
    score = det.get("score")
    if score is None:
        score_text = "score=n/a"
    else:
        try:
            score_text = f"score={float(score):.3f}"
        except (TypeError, ValueError):
            score_text = "score=n/a"
    source = str(det.get("source") or det.get("score_source") or "").strip()
    source_text = f" source={source}" if source else ""
    cluster_id = det.get("cluster_id")
    id_text = f"id={cluster_id} " if cluster_id is not None else ""
    cell_text = ""
    if grid:
        cell = _agent_grid_cell_for_detection(det, img_w, img_h, grid)
        if cell:
            cell_text = f" cell={cell}"
    return f"{id_text}label={label} bbox={bbox_text} {score_text}{source_text}{cell_text}"


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


def _agent_readable_bbox_from_args(args: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    if not isinstance(args, dict):
        return "", ""
    for key in ("window_bbox_2d", "bbox_2d", "bbox_xyxy_px"):
        bbox = args.get(key)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return key, _agent_readable_format_bbox(bbox)
    return "", ""


def _agent_readable_tool_call_summary(tool_name: str, args: Optional[Dict[str, Any]]) -> str:
    tool = str(tool_name or "")
    key, coords = _agent_readable_bbox_from_args(args)
    grid_cell = _agent_readable_trim((args or {}).get("grid_cell"))
    cluster_id = (args or {}).get("cluster_id")
    if tool == "run_detector":
        mode = str((args or {}).get("mode") or "yolo")
        grid_text = f" grid={grid_cell}" if grid_cell else ""
        return f"run_detector mode={mode}{grid_text}"
    if tool == "zoom_and_detect":
        mode = str((args or {}).get("mode") or "yolo")
        intent = _agent_readable_trim((args or {}).get("intent"))
        confirm_label = _agent_readable_trim((args or {}).get("confirm_label"))
        intent_text = f" intent=\"{intent}\"" if intent else ""
        confirm_text = f" confirm={confirm_label}" if confirm_label else ""
        grid_text = f" grid={grid_cell}" if grid_cell else ""
        return f"zoom_and_detect mode={mode}{intent_text}{confirm_text}{grid_text}"
    if tool == "look_and_inspect":
        max_objects = (args or {}).get("max_objects")
        intent = _agent_readable_trim((args or {}).get("intent"))
        extra = f" max_objects={max_objects}" if max_objects else ""
        if intent:
            extra = f"{extra} intent=\"{intent}\""
        grid_text = f" grid={grid_cell}" if grid_cell else ""
        return f"inspect cell{grid_text}{extra}"
    if tool == "image_zoom_in_tool":
        intent = _agent_readable_trim((args or {}).get("intent"))
        intent_text = f" intent=\"{intent}\"" if intent else ""
        grid_text = f" grid={grid_cell}" if grid_cell else ""
        cluster_text = f" cluster={cluster_id}" if cluster_id is not None else ""
        return f"looking closer{grid_text}{cluster_text}{intent_text}"
    if tool in {"view_cell_raw", "view_cell_overlay"}:
        grid_text = f" grid={grid_cell}" if grid_cell else ""
        return f"{tool}{grid_text}"
    if tool == "get_tile_context":
        grid_text = f" grid={grid_cell}" if grid_cell else ""
        return f"get_tile_context{grid_text}"
    if tool == "get_tile_context_chunk":
        handle = _agent_readable_trim((args or {}).get("context_handle"))
        idx = (args or {}).get("chunk_index")
        return f"get_tile_context_chunk handle={handle} idx={idx}"
    if tool == "get_global_context":
        return "get_global_context"
    if tool == "get_global_context_chunk":
        handle = _agent_readable_trim((args or {}).get("context_handle"))
        idx = (args or {}).get("chunk_index")
        return f"get_global_context_chunk handle={handle} idx={idx}"
    if tool == "think_missed_objects":
        return "think_missed_objects"
    if tool == "log_observation":
        observation = _agent_readable_trim((args or {}).get("text"))
        if observation and grid_cell:
            return f"observation {grid_cell} \"{observation}\""
        if observation:
            return f"observation \"{observation}\""
        return "observation"
    if tool == "log_status":
        status = _agent_readable_trim((args or {}).get("text"))
        if status:
            return f"status \"{status}\""
        return "status"
    if tool == "classify_crop":
        label = _agent_readable_trim((args or {}).get("label_hint"))
        label_text = f" label={label}" if label else ""
        cluster_text = f" cluster={cluster_id}" if cluster_id is not None else ""
        return f"classify crop{cluster_text}{label_text}"
    if tool == "sam3_text":
        prompt = _agent_readable_trim((args or {}).get("prompt"))
        label = _agent_readable_trim((args or {}).get("label"))
        details: List[str] = []
        if label:
            details.append(f"label={label}")
        if prompt:
            details.append(f"prompt=\"{prompt}\"")
        grid_text = f" grid={grid_cell}" if grid_cell else ""
        return ("sam3_text " + " ".join(details) if details else "sam3_text") + grid_text
    if tool == "sam3_similarity":
        exemplars = (args or {}).get("exemplar_cluster_ids")
        count = len(exemplars) if isinstance(exemplars, list) else 0
        grid_text = f" grid={grid_cell}" if grid_cell else ""
        return f"sam3_similarity exemplars={count}{grid_text}"
    if tool == "qwen_infer":
        prompt_type = (args or {}).get("prompt_type") or "bbox"
        items = (args or {}).get("items")
        item_list = (args or {}).get("item_list")
        count = len(items) if isinstance(items, list) else 0
        if not count and isinstance(item_list, str) and item_list.strip():
            count = len([item for item in item_list.split(",") if item.strip()])
        prompt = _agent_readable_trim((args or {}).get("prompt"))
        details = f"type={prompt_type} items={count}"
        if prompt:
            details += f" prompt=\"{prompt}\""
        grid_text = f" grid={grid_cell}" if grid_cell else ""
        return f"qwen_infer {details}{grid_text}"
    if tool == "list_candidates":
        label = _agent_readable_trim((args or {}).get("label"))
        source = _agent_readable_trim((args or {}).get("source"))
        min_score = (args or {}).get("min_score")
        max_items = (args or {}).get("max_items")
        details: List[str] = []
        if label:
            details.append(f"label={label}")
        if source:
            details.append(f"source={source}")
        if isinstance(min_score, (int, float)):
            details.append(f"min_score={float(min_score):.2f}")
        if isinstance(max_items, (int, float)):
            details.append(f"max_items={int(max_items)}")
        return "list_candidates " + " ".join(details) if details else "list_candidates"
    if tool == "grid_label_counts":
        label = _agent_readable_trim((args or {}).get("label"))
        return f"grid_label_counts label={label}" if label else "grid_label_counts"
    if tool == "submit_annotations":
        annotations = (args or {}).get("annotations")
        count = len(annotations) if isinstance(annotations, list) else 0
        clusters = (args or {}).get("cluster_ids")
        cluster_count = len(clusters) if isinstance(clusters, list) else 0
        handles = (args or {}).get("handles")
        handle_count = len(handles) if isinstance(handles, list) else 0
        include_all = bool((args or {}).get("include_all"))
        extra = ""
        if include_all:
            extra = " include_all"
        elif handle_count:
            extra = f" handles={handle_count}"
        elif cluster_count:
            extra = f" clusters={cluster_count}"
        return f"submit annotations count={count}{extra}"
    return tool


def _agent_readable_tool_result_summary(tool_name: str, result: Any) -> str:
    tool = str(tool_name or "")
    if not isinstance(result, dict):
        return f"{tool} result"
    if result.get("blocked"):
        reason = result.get("error") or "blocked"
        return f"{tool} blocked ({reason})"
    if result.get("error"):
        err = result.get("error")
        if isinstance(err, dict):
            code = err.get("code") or "error"
            return f"{tool} error ({code})"
        return f"{tool} error ({err})"
    if result.get("skipped"):
        reason = result.get("reason") or "skipped"
        return f"{tool} skipped ({reason})"
    if tool == "classify_crop":
        best = result.get("best") if isinstance(result, dict) else None
        if isinstance(best, dict):
            label = best.get("label")
            prob = best.get("prob")
            if label is not None and prob is not None:
                return f"classify_crop best={label} prob={prob:.3f}"
        return "classify_crop result"
    if tool in {"run_detector", "zoom_and_detect", "sam3_text", "sam3_similarity", "qwen_infer", "look_and_inspect"}:
        cluster_ids = result.get("cluster_ids")
        count = len(cluster_ids) if isinstance(cluster_ids, list) else None
        new_clusters = result.get("new_clusters")
        label_counts = result.get("label_counts") if isinstance(result, dict) else None
        caption = result.get("caption") if isinstance(result, dict) else None
        if count is not None:
            extra = f" new={new_clusters}" if isinstance(new_clusters, int) else ""
            label_text = ""
            if isinstance(label_counts, dict) and label_counts:
                label_parts = [f"{key}={label_counts[key]}" for key in sorted(label_counts.keys())]
                label_text = " labels=" + ",".join(label_parts[:6])
            caption_text = f" caption=\"{_agent_readable_trim(caption)}\"" if caption and tool == "look_and_inspect" else ""
            return f"{tool} returned {count} clusters{extra}{label_text}{caption_text}"
        return f"{tool} result"
    if tool in {"view_cell_raw", "view_cell_overlay"}:
        cell = result.get("grid_cell")
        cell_text = f" {cell}" if cell else ""
        return f"{tool}{cell_text} ready"
    if tool == "view_full_overlay":
        cells = result.get("grid_usage") if isinstance(result, dict) else None
        count = len(cells) if isinstance(cells, list) else 0
        return f"view_full_overlay cells={count}"
    if tool == "get_tile_context":
        total = result.get("cluster_total") if isinstance(result, dict) else None
        grid_cell = result.get("grid_cell") if isinstance(result, dict) else None
        if total is not None:
            return f"get_tile_context {grid_cell} clusters={total}"
        return "get_tile_context result"
    if tool == "get_tile_context_chunk":
        idx = result.get("chunk_index") if isinstance(result, dict) else None
        total = result.get("chunk_total") if isinstance(result, dict) else None
        return f"get_tile_context_chunk {idx}/{total}" if idx is not None else "get_tile_context_chunk result"
    if tool == "get_global_context":
        count = len(result.get("tile_summaries") or []) if isinstance(result, dict) else 0
        return f"get_global_context tiles={count}"
    if tool == "get_global_context_chunk":
        idx = result.get("chunk_index") if isinstance(result, dict) else None
        total = result.get("chunk_total") if isinstance(result, dict) else None
        return f"get_global_context_chunk {idx}/{total}" if idx is not None else "get_global_context_chunk result"
    if tool == "think_missed_objects":
        labels = result.get("missing_labels") if isinstance(result, dict) else None
        tiles = result.get("missing_tiles") if isinstance(result, dict) else None
        label_count = len(labels) if isinstance(labels, list) else 0
        tile_count = len(tiles) if isinstance(tiles, list) else 0
        return f"think_missed_objects labels={label_count} tiles={tile_count}"
    if tool == "grid_label_counts":
        cells = result.get("cells")
        count = len(cells) if isinstance(cells, list) else 0
        return f"grid_label_counts cells={count}"
    if tool == "submit_annotations":
        clusters = result.get("submitted_clusters")
        cluster_count = len(clusters) if isinstance(clusters, list) else 0
        count = result.get("count")
        if cluster_count:
            return f"submit_annotations clusters={cluster_count} count={count}"
    if tool == "log_observation":
        observation = _agent_readable_trim(result.get("observation")) if isinstance(result, dict) else ""
        if observation:
            return f"observation logged \"{observation}\""
        return "observation logged"
    if tool == "zoom_and_detect":
        confirmation = result.get("confirmation") if isinstance(result, dict) else None
        if isinstance(confirmation, dict):
            label = confirmation.get("label")
            match = confirmation.get("label_match")
            if label is not None and match is not None:
                return f"zoom_and_detect confirmation label={label} match={'yes' if match else 'no'}"
    if tool == "image_zoom_in_tool":
        cell = result.get("grid_cell") if isinstance(result, dict) else None
        cluster = result.get("cluster_id") if isinstance(result, dict) else None
        cell_text = f" grid={cell}" if cell else ""
        cluster_text = f" cluster={cluster}" if cluster is not None else ""
        if cell_text or cluster_text:
            return f"image_zoom_in_tool{cell_text}{cluster_text}"
        return "image_zoom_in_tool result"
    for key in ("detections", "candidates", "annotations"):
        items = result.get(key)
        if isinstance(items, list):
            details = _agent_readable_candidates_summary(items)
            suffix = f": {details}" if details else ""
            return f"{tool} returned {len(items)} {key}{suffix}"
    return f"{tool} result"


def _agent_readable_line(
    event_type: str,
    *,
    step_id: Optional[int] = None,
    tool_name: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    output_text: Optional[str] = None,
) -> str:
    prefix = f"step{step_id}: " if step_id is not None else ""
    if event_type == "tool_call" and tool_name:
        return prefix + _agent_readable_tool_call_summary(tool_name, args)
    if event_type == "tool_result" and tool_name:
        return prefix + _agent_readable_tool_result_summary(tool_name, result)
    if event_type == "model_output" and output_text:
        return prefix + f"model output \"{_agent_readable_trim(output_text)}\""
    if event_type == "prepass" and tool_name:
        return prefix + _agent_readable_tool_result_summary(tool_name, result)
    return ""


def _agent_readable_candidates_summary(items: List[Dict[str, Any]], limit: int = 8) -> str:
    if not items:
        return ""
    parts: List[str] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        cand_id = item.get("candidate_id")
        label = str(item.get("label") or item.get("class_name") or "unknown")
        bbox = None
        if isinstance(item.get("bbox_xyxy_px"), (list, tuple)) and len(item.get("bbox_xyxy_px")) >= 4:
            bbox = item.get("bbox_xyxy_px")
        elif isinstance(item.get("bbox_2d"), (list, tuple)) and len(item.get("bbox_2d")) >= 4:
            bbox = item.get("bbox_2d")
        bbox_text = _agent_readable_format_bbox(bbox) if bbox else "[]"
        score = item.get("score")
        if score is None:
            score_text = "score=n/a"
        else:
            try:
                score_text = f"score={float(score):.3f}"
            except (TypeError, ValueError):
                score_text = "score=n/a"
        id_text = f"id={cand_id} " if cand_id is not None else ""
        parts.append(f"{id_text}{label} {bbox_text} {score_text}")
    if len(items) > limit:
        parts.append(f"+{len(items) - limit} more")
    return "; ".join(parts)
