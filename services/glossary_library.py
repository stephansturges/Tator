"""Glossary library helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


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
