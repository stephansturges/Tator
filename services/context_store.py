"""Context store helpers."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from fastapi import HTTPException
from starlette.status import HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_ENTITY


def _context_store(
    payload: Dict[str, Any],
    *,
    kind: str,
    max_bytes: Optional[int],
    tile_store: Dict[str, Dict[str, Any]],
    global_store: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=True)
    raw_bytes = raw.encode("utf-8", errors="ignore")
    byte_size = len(raw_bytes)
    limit = int(max_bytes or 0)
    if limit <= 0 or byte_size <= limit:
        return {"payload": payload, "byte_size": byte_size, "chunked": False}
    chunk_size = max(1, limit)
    chunks = []
    for idx in range(0, byte_size, chunk_size):
        chunk = raw_bytes[idx : idx + chunk_size].decode("utf-8", errors="ignore")
        chunks.append(chunk)
    handle = f"{kind}_{uuid.uuid4().hex[:10]}"
    store = {"chunks": chunks, "byte_size": byte_size}
    if kind == "tile":
        tile_store[handle] = store
    else:
        global_store[handle] = store
    return {
        "chunked": True,
        "context_handle": handle,
        "chunk_total": len(chunks),
        "byte_size": byte_size,
    }


def _context_chunk(
    handle: str,
    *,
    chunk_index: int,
    kind: str,
    tile_store: Dict[str, Dict[str, Any]],
    global_store: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    store = tile_store if kind == "tile" else global_store
    entry = store.get(handle)
    if not entry:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="context_handle_missing")
    chunks = entry.get("chunks") or []
    total = len(chunks)
    idx = int(chunk_index)
    if idx < 0 or idx >= total:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="context_chunk_index_invalid")
    return {
        "context_handle": handle,
        "chunk_index": idx,
        "chunk_total": total,
        "payload_chunk": chunks[idx],
        "byte_size": entry.get("byte_size"),
    }


def _agent_context_store(
    payload: Dict[str, Any],
    *,
    kind: str,
    max_bytes: Optional[int],
    tile_store: Dict[str, Dict[str, Any]],
    global_store: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return _context_store(
        payload,
        kind=kind,
        max_bytes=max_bytes,
        tile_store=tile_store,
        global_store=global_store,
    )


def _agent_context_chunk(
    handle: str,
    *,
    chunk_index: int,
    kind: str,
    tile_store: Dict[str, Dict[str, Any]],
    global_store: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return _context_chunk(
        handle,
        chunk_index=chunk_index,
        kind=kind,
        tile_store=tile_store,
        global_store=global_store,
    )
