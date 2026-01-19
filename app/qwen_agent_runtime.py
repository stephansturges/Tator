from __future__ import annotations

from typing import Any, Dict

try:
    from importlib import metadata as _metadata
except Exception:  # pragma: no cover - importlib.metadata missing on old envs
    _metadata = None  # type: ignore[assignment]

try:
    import qwen_agent  # type: ignore
except Exception:
    qwen_agent = None  # type: ignore[assignment]


def qwen_agent_status() -> Dict[str, Any]:
    version = None
    if _metadata is not None and qwen_agent is not None:
        try:
            version = _metadata.version("qwen-agent")
        except Exception:
            version = None
    return {
        "available": qwen_agent is not None,
        "version": version,
    }


def qwen_agent_available() -> bool:
    return qwen_agent is not None
