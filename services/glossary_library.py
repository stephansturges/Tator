from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_glossary_name(name: str) -> str:
    return " ".join(str(name or "").strip().split())


def _glossary_key(name: str) -> str:
    return _normalize_glossary_name(name).lower()


def _load_glossary_library(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    entries = data.get("glossaries") if isinstance(data, dict) else data
    return list(entries) if isinstance(entries, list) else []


def _persist_glossary_library(path: Path, entries: List[Dict[str, Any]]) -> None:
    payload = {"glossaries": entries}
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    tmp_path.replace(path)


def _find_glossary_entry(entries: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    key = _glossary_key(name)
    if not key:
        return None
    for entry in entries:
        if _glossary_key(entry.get("name")) == key:
            return entry
    return None
