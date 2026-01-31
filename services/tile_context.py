from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from services.prepass_grid import _agent_grid_cell_for_window_bbox


def _cluster_owner_cell(cluster: Mapping[str, Any]) -> Optional[str]:
    owner = cluster.get("owner_cell")
    if owner:
        return str(owner)
    cell = cluster.get("grid_cell")
    if cell:
        return str(cell)
    return None


def _tile_clusters(grid_cell: str, clusters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not grid_cell:
        return list(clusters or [])
    tile = str(grid_cell)
    return [
        cluster
        for cluster in clusters
        if isinstance(cluster, dict) and _cluster_owner_cell(cluster) == tile
    ]


def _tile_cluster_payload(
    grid_cell: str,
    clusters: Sequence[Dict[str, Any]],
    *,
    handle_fn: Callable[[Mapping[str, Any]], Optional[str]],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for cluster in _tile_clusters(grid_cell, clusters):
        items.append(
            {
                "cluster_id": int(cluster.get("cluster_id")),
                "handle": handle_fn(cluster),
                "label": cluster.get("label"),
                "score": cluster.get("score"),
                "sources": cluster.get("source_list") or [cluster.get("source")],
                "grid_cell": cluster.get("grid_cell"),
                "owner_cell": cluster.get("owner_cell"),
                "verified": bool(cluster.get("classifier_accept")),
            }
        )
    return items


def _tile_caption_hint(
    grid_cell: str,
    windowed_captions: Optional[Sequence[Dict[str, Any]]],
    grid: Optional[Mapping[str, Any]],
) -> Optional[str]:
    if not windowed_captions or not grid:
        return None
    hints: List[str] = []
    for entry in windowed_captions:
        if not isinstance(entry, dict):
            continue
        bbox = entry.get("bbox_2d")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        cell = _agent_grid_cell_for_window_bbox(grid, bbox)
        if cell and str(cell) == str(grid_cell):
            text = str(entry.get("caption") or "").strip()
            if text:
                hints.append(text)
    if not hints:
        return None
    return " ".join(hints)


def _build_tile_context_payloads(
    grid_cell: str,
    *,
    clusters: Sequence[Dict[str, Any]],
    grid: Optional[Mapping[str, Any]],
    windowed_captions: Optional[Sequence[Dict[str, Any]]],
    handle_fn: Callable[[Mapping[str, Any]], Optional[str]],
    cluster_label_counts_fn: Callable[[Sequence[int]], Dict[str, int]],
    overlay_labels_fn: Callable[[Sequence[Dict[str, Any]], Sequence[str]], Sequence[str]],
    label_colors_fn: Callable[[Sequence[str]], Dict[str, str]],
    label_prefixes_fn: Callable[[Sequence[str]], Dict[str, str]],
    overlay_key_fn: Callable[[Dict[str, str], Dict[str, str]], str],
    labelmap: Sequence[str],
    grid_usage: Mapping[str, Mapping[str, int]],
    grid_usage_last: Mapping[str, Mapping[str, float]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cluster_list = _tile_cluster_payload(grid_cell, clusters, handle_fn=handle_fn)
    counts = cluster_label_counts_fn([item["cluster_id"] for item in cluster_list])
    tool_usage = dict(grid_usage.get(grid_cell, {}))
    tool_usage_last = dict(grid_usage_last.get(grid_cell, {}))
    caption_hint = _tile_caption_hint(grid_cell, windowed_captions, grid)
    labels = overlay_labels_fn(clusters, labelmap)
    label_colors = label_colors_fn(labels) if labels else {}
    label_prefixes = label_prefixes_fn(labels) if labels else {}
    agent_cluster_list = []
    for item in cluster_list:
        if not isinstance(item, dict):
            continue
        agent_cluster_list.append(
            {
                "handle": item.get("handle"),
                "label": item.get("label"),
                "score": item.get("score"),
                "sources": item.get("sources"),
                "grid_cell": item.get("grid_cell"),
                "owner_cell": item.get("owner_cell"),
                "verified": bool(item.get("verified")),
            }
        )
    payload = {
        "tile_id": grid_cell,
        "grid_cell": grid_cell,
        "tile_cluster_list": cluster_list,
        "tile_counts": counts,
        "tool_usage": tool_usage,
        "tool_usage_last": tool_usage_last,
        "caption_hint": caption_hint,
        "overlay_key": overlay_key_fn(label_colors, label_prefixes),
        "cluster_total": len(cluster_list),
    }
    public_payload = {
        "tile_id": grid_cell,
        "grid_cell": grid_cell,
        "tile_cluster_list": agent_cluster_list,
        "tile_counts": counts,
        "tool_usage": tool_usage,
        "tool_usage_last": tool_usage_last,
        "caption_hint": caption_hint,
        "overlay_key": overlay_key_fn(label_colors, label_prefixes),
        "cluster_total": len(cluster_list),
    }
    return payload, public_payload
