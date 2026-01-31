from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Sequence


def _qwen_agent_message_text(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for item in content:
            try:
                item_type, item_value = item.get_type_and_value()
            except Exception:
                continue
            if item_type == "text" and item_value:
                texts.append(str(item_value))
        return "\n".join(texts)
    return ""


def _agent_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text") or ""))
            elif hasattr(item, "get_type_and_value"):
                try:
                    item_type, item_value = item.get_type_and_value()
                except Exception:
                    continue
                if item_type == "text" and item_value:
                    parts.append(str(item_value))
        return "".join(parts)
    return str(content)


def _agent_stream_text_from_output(output: Sequence[Any]) -> str:
    if not output:
        return ""
    for msg in reversed(output):
        if msg is None:
            continue
        if isinstance(msg, dict):
            if msg.get("role") == "assistant":
                return _agent_content_to_text(msg.get("content"))
            continue
        if getattr(msg, "role", None) == "assistant":
            text = _qwen_agent_message_text(msg)
            if text:
                return text
    return ""


def _agent_stream_tag_open(text: str, start: str, end: str) -> bool:
    if not text:
        return False
    start_idx = text.rfind(start)
    if start_idx < 0:
        return False
    end_idx = text.rfind(end)
    return end_idx < start_idx


def _agent_stream_extract_tool_name(text: str) -> Optional[str]:
    if not text:
        return None
    tail = text
    if "<tool_call>" in text:
        tail = text.split("<tool_call>")[-1]
    match = re.search(r"\"name\"\\s*:\\s*\"([^\"]+)\"", tail)
    if match:
        return match.group(1)
    return None


def _agent_parse_json_relaxed(payload: Any) -> Optional[Any]:
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        return payload
    if not isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return None
    text = payload.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None
