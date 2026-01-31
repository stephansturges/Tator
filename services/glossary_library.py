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


def _upsert_glossary_entry_impl(
    name: str,
    glossary_text: str,
    *,
    normalize_name_fn,
    normalize_glossary_fn,
    load_entries_fn,
    find_entry_fn,
    persist_entries_fn,
    glossary_key_fn,
    lock,
    time_fn,
) -> Dict[str, Any]:
    normalized_name = normalize_name_fn(name)
    if not normalized_name:
        raise ValueError("glossary_name_required")
    glossary_text = normalize_glossary_fn(glossary_text)
    with lock:
        entries = load_entries_fn()
        entry = find_entry_fn(entries, normalized_name)
        now = float(time_fn())
        if entry:
            entry["name"] = normalized_name
            entry["glossary"] = glossary_text
            entry["updated_at"] = now
        else:
            entry = {
                "name": normalized_name,
                "glossary": glossary_text,
                "created_at": now,
                "updated_at": now,
            }
            entries.append(entry)
        entries.sort(key=lambda item: glossary_key_fn(item.get("name")))
        persist_entries_fn(entries)
    return entry


def _delete_glossary_entry_impl(
    name: str,
    *,
    normalize_name_fn,
    load_entries_fn,
    persist_entries_fn,
    glossary_key_fn,
    lock,
) -> bool:
    normalized_name = normalize_name_fn(name)
    if not normalized_name:
        return False
    with lock:
        entries = load_entries_fn()
        before = len(entries)
        entries = [entry for entry in entries if glossary_key_fn(entry.get("name")) != glossary_key_fn(normalized_name)]
        if len(entries) == before:
            return False
        persist_entries_fn(entries)
    return True
