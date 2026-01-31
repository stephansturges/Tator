from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, List


def _agent_refresh_handle_index(
    clusters: Sequence[Dict[str, Any]],
    *,
    handle_fn: Callable[[Mapping[str, Any]], Optional[str]],
) -> Dict[str, int]:
    handle_index: Dict[str, int] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        handle = handle_fn(cluster)
        cid = cluster.get("cluster_id")
        if handle and cid is not None:
            handle_index[str(handle)] = int(cid)
    return handle_index


def _agent_cluster_handle(
    cluster: Mapping[str, Any],
    *,
    label_prefixes: Optional[Dict[str, str]],
    labelmap: Sequence[str],
    label_prefix_map_fn: Callable[[Sequence[str]], Dict[str, str]],
) -> Optional[str]:
    cluster_id = cluster.get("cluster_id") or cluster.get("candidate_id")
    if cluster_id is None:
        return None
    label = str(cluster.get("label") or "").strip()
    prefix = None
    if label:
        prefix = (label_prefixes or {}).get(label)
        if prefix is None:
            labels = list(labelmap or [])
            if not labels:
                labels = [label]
            prefix_map = label_prefix_map_fn(labels)
            prefix = prefix_map.get(label)
    if prefix:
        return f"{prefix}{int(cluster_id)}"
    return str(cluster_id)


def _agent_cluster_id_from_handle(
    handle: Optional[str],
    *,
    handle_index: Dict[str, int],
    cluster_index: Dict[int, Dict[str, Any]],
) -> Optional[int]:
    if not handle:
        return None
    text = str(handle).strip()
    if not text:
        return None
    if text in handle_index:
        return int(handle_index[text])
    if text.isdigit():
        cid = int(text)
        if cid in cluster_index:
            return cid
    for cluster_id, cluster in cluster_index.items():
        if str(cluster.get("handle") or "").strip() == text:
            return int(cluster_id)
    return None


def _agent_handles_from_cluster_ids(
    cluster_ids: Sequence[int],
    *,
    cluster_index: Dict[int, Dict[str, Any]],
    handle_fn: Callable[[Mapping[str, Any]], Optional[str]],
) -> List[str]:
    handles = []
    for cid in cluster_ids:
        cluster = cluster_index.get(int(cid))
        if not cluster:
            continue
        handle = handle_fn(cluster)
        if handle:
            handles.append(str(handle))
    return handles


def _agent_cluster_ids_from_handles(
    handles: Sequence[str],
    *,
    handle_index: Dict[str, int],
    cluster_index: Dict[int, Dict[str, Any]],
) -> List[int]:
    cluster_ids: List[int] = []
    for handle in handles:
        cid = _agent_cluster_id_from_handle(
            handle, handle_index=handle_index, cluster_index=cluster_index
        )
        if cid is None:
            continue
        cluster_ids.append(int(cid))
    return cluster_ids
