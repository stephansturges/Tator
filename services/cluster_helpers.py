from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


def _cluster_label_counts(
    cluster_ids: Sequence[int],
    cluster_index: Mapping[int, Dict[str, Any]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cid in cluster_ids:
        cluster = cluster_index.get(int(cid))
        if not cluster:
            continue
        label = str(cluster.get("label") or "").strip()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def _cluster_summaries(
    cluster_ids: Sequence[int],
    cluster_index: Mapping[int, Dict[str, Any]],
    *,
    handle_fn: Callable[[Mapping[str, Any]], Optional[str]],
    round_bbox_fn: Callable[[Any], Optional[List[float]]],
    max_items: int = 0,
    include_ids: bool = True,
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for cid in cluster_ids:
        cluster = cluster_index.get(int(cid))
        if not cluster:
            continue
        item = {
            "handle": handle_fn(cluster),
            "label": cluster.get("label"),
            "grid_cell": cluster.get("grid_cell"),
            "score": cluster.get("score"),
            "score_source": cluster.get("score_source") or cluster.get("source"),
            "bbox_2d": round_bbox_fn(cluster.get("bbox_2d")),
        }
        if include_ids:
            item["cluster_id"] = int(cluster.get("cluster_id"))
        items.append(item)
    total = len(items)
    if max_items <= 0:
        return {"items": items, "total": total, "truncated": False}
    truncated = total > max_items
    return {"items": items[:max_items], "total": total, "truncated": truncated}
