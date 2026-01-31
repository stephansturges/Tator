from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence


def _agent_trace_sanitize_payload(payload: Any, image_token: Optional[str]) -> Dict[str, Any]:
    data = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload or {})
    if image_token:
        data["image_token"] = image_token
    if data.get("image_base64"):
        data["image_base64"] = "<redacted>"
    return data


def _agent_trace_sanitize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        item = dict(msg)
        content = item.get("content")
        if isinstance(content, list):
            stripped = []
            for entry in content:
                if isinstance(entry, dict) and entry.get("image"):
                    stripped.append({"type": "image"})
                elif isinstance(entry, dict) and "text" in entry:
                    stripped.append({"text": entry.get("text")})
                else:
                    stripped.append(entry)
            item["content"] = stripped
        sanitized.append(item)
    return sanitized


def _agent_trace_full_jsonable(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 6:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_agent_trace_full_jsonable(item, _depth=_depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _agent_trace_full_jsonable(item, _depth=_depth + 1)
            for key, item in value.items()
        }
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import torch
    except Exception:
        torch = None
    if np is not None and isinstance(value, np.ndarray):
        if value.ndim <= 1 and value.size <= 16:
            return value.tolist()
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if torch is not None and isinstance(value, torch.Tensor):
        try:
            arr = value.detach().cpu().numpy()
            return _agent_trace_full_jsonable(arr, _depth=_depth + 1)
        except Exception:
            return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if hasattr(value, "model_dump"):
        try:
            return _agent_trace_full_jsonable(value.model_dump(), _depth=_depth + 1)
        except Exception:
            return str(value)
    try:
        return json.loads(json.dumps(value))
    except Exception:
        return str(value)
