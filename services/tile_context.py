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
