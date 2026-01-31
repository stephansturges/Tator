from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence


def _glossary_label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).strip().lower())


def _extract_glossary_synonyms(text: str) -> List[str]:
    cleaned = re.sub(r"[()]", " ", str(text))
    parts = re.split(r"[;,/]|\\band\\b|\\bor\\b", cleaned, flags=re.IGNORECASE)
    synonyms: List[str] = []
    for part in parts:
        term = part.strip()
        if not term:
            continue
        term = re.sub(
            r"^(all kinds of|all kind of|all|including|include|including the|including those|such as|other)\\s+",
            "",
            term,
            flags=re.IGNORECASE,
        ).strip()
        term = term.strip(" .")
        if term:
            synonyms.append(term)
    return synonyms


def _parse_glossary_synonyms(glossary: str, labelmap: Sequence[str]) -> Dict[str, List[str]]:
    if not glossary:
        return {}
    norm_to_label = {_glossary_label_key(lbl): lbl for lbl in labelmap if str(lbl).strip()}
    mapping: Dict[str, List[str]] = {}
    for line in str(glossary).splitlines():
        raw = line.strip()
        if not raw:
            continue
        key = None
        rest = None
        if "->" in raw:
            key, rest = raw.split("->", 1)
        elif ":" in raw:
            key, rest = raw.split(":", 1)
        elif "=" in raw:
            key, rest = raw.split("=", 1)
        elif " - " in raw:
            key, rest = raw.split(" - ", 1)
        else:
            match = re.match(r"^(\\w+)\\s*\\((.+)\\)$", raw)
            if match:
                key, rest = match.group(1), match.group(2)
        if not key or rest is None:
            continue
        label = norm_to_label.get(_glossary_label_key(key))
        if not label:
            continue
        synonyms = _extract_glossary_synonyms(rest)
        if synonyms:
            mapping.setdefault(label, []).extend(synonyms)
    return mapping


def _parse_glossary_mapping(glossary: str, labelmap: Sequence[str]) -> Dict[str, List[str]]:
    text = str(glossary or "").strip()
    if not text:
        return {}
    norm_to_label = {_glossary_label_key(lbl): lbl for lbl in labelmap if str(lbl).strip()}
    parsed: Any = None
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
    mapping: Dict[str, List[str]] = {}
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            label = norm_to_label.get(_glossary_label_key(key))
            if not label:
                continue
            terms: List[str] = []
            if isinstance(value, (list, tuple)):
                terms = [str(item) for item in value]
            elif isinstance(value, str):
                terms = _split_synonym_terms(value)
            cleaned = _normalize_synonym_list(terms)
            if cleaned:
                mapping[label] = cleaned
        return mapping
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            key = item.get("label") or item.get("class") or item.get("name")
            if not key:
                continue
            label = norm_to_label.get(_glossary_label_key(key))
            if not label:
                continue
            value = item.get("synonyms") or item.get("terms") or item.get("values") or item.get("list")
            terms: List[str] = []
            if isinstance(value, (list, tuple)):
                terms = [str(val) for val in value]
            elif isinstance(value, str):
                terms = _split_synonym_terms(value)
            cleaned = _normalize_synonym_list(terms)
            if cleaned:
                mapping[label] = cleaned
        return mapping
    # Fallback: parse line‑based glossary
    mapping.update(_parse_glossary_synonyms(glossary, labelmap))
    return mapping


def _split_synonym_terms(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[;,/]|\\band\\b|\\bor\\b", str(text), flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def _clean_sam3_synonym(term: str) -> str:
    if not term:
        return ""
    cleaned = str(term).strip()
    cleaned = re.sub(r"[\"']", "", cleaned).strip()
    cleaned = re.sub(
        r"^(all kinds of|all kind of|all|including|include|including the|including those|such as|other|and|or)\\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = cleaned.strip(" .")
    if not cleaned:
        return ""
    if len(cleaned) > 40:
        return ""
    if "->" in cleaned or ":" in cleaned:
        return ""
    return cleaned


def _normalize_synonym_list(terms: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for term in terms:
        for part in _split_synonym_terms(term):
            cleaned = _clean_sam3_synonym(part)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(cleaned)
    return normalized


def _dedupe_synonyms(terms: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for term in terms:
        key = str(term).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(term)
    return output
