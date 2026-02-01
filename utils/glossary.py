"""Glossary normalization helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence


_DEFAULT_GLOSSARY_MAP: Dict[str, List[str]] = {
    "bike": ["bike", "motorbike", "scooter", "motorcycle"],
    "boat": ["boat", "canoe", "kayak", "surfboard", "ship"],
    "building": ["building", "house", "store", "office building", "residential building", "warehouse"],
    "bus": ["bus", "omnibus", "autobus", "coach"],
    "container": ["container", "truck container", "shipping container"],
    "digger": [
        "digger",
        "excavator",
        "tractor",
        "backhoe",
        "construction vehicle",
        "bulldozer",
        "steam shovel",
        "loader excavator",
        "dozer",
        "earthmover",
        "heavy machinery",
    ],
    "gastank": [
        "silos",
        "tank",
        "storage tank",
        "barrel",
        "pressure vessel",
        "oil silo",
        "storage silo",
        "oil tank",
    ],
    "light_vehicle": [
        "car",
        "light vehicle",
        "light_vehicle",
        "pickup truck",
        "sedan",
        "suv",
        "van",
        "4x4",
        "family car",
        "passenger vehicle",
        "automobile",
        "hatchback",
    ],
    "person": [
        "cyclist",
        "person",
        "swimmer",
        "human",
        "passenger",
        "pedestrian",
        "walker",
        "hiker",
        "individual",
    ],
    "solarpanels": ["array", "solar panel", "solarpanels"],
    "truck": [
        "truck",
        "lorry",
        "commercial vehicle",
        "semi truck",
        "articulated truck",
        "heavy-duty vehicle",
        "big rig",
        "18-wheeler",
        "semi-trailer truck",
    ],
    "utility_pole": [
        "antenna",
        "pole",
        "utility pole",
        "utility_pole",
        "street fixture",
        "drying rack",
        "streetlight",
        "street lamp",
        "electricity pylon",
        "power pylon",
        "transmission tower",
        "high-voltage pole",
        "lattice tower",
        "mast",
        "comms mast",
        "aerial mast",
        "satellite dish",
        "mounting pole",
        "light fixture",
    ],
}


def _glossary_label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).strip().lower())


def _normalize_labelmap_glossary(raw_glossary: Any) -> str:
    if raw_glossary is None:
        return ""
    if isinstance(raw_glossary, str):
        return raw_glossary.strip()
    if isinstance(raw_glossary, list):
        lines = [str(item).strip() for item in raw_glossary if str(item).strip()]
        return "\n".join(lines)
    if isinstance(raw_glossary, dict):
        lines = []
        for key, value in raw_glossary.items():
            if value is None:
                lines.append(f"{key}".strip())
                continue
            if isinstance(value, (list, tuple)):
                joined = ", ".join([str(item) for item in value if str(item).strip()])
                lines.append(f"{key}: {joined}".strip())
            else:
                lines.append(f"{key}: {value}".strip())
        return "\n".join([line for line in lines if line.strip()])
    return str(raw_glossary).strip()


def _normalize_glossary_name(name: str) -> str:
    return re.sub(r"\\s+", " ", str(name or "").strip())


def _glossary_key(name: str) -> str:
    return _normalize_glossary_name(name).lower()


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


def _default_agent_glossary_for_labelmap(labelmap: Sequence[str]) -> str:
    def _normalize(name: str) -> str:
        return "".join(ch for ch in name.lower().strip() if ch.isalnum() or ch == "_")
    mapped: Dict[str, List[str]] = {}
    for label in labelmap:
        norm = _normalize(label)
        synonyms = _DEFAULT_GLOSSARY_MAP.get(norm)
        if synonyms:
            mapped[label] = synonyms
    if not mapped:
        return ""
    return json.dumps(mapped, indent=2, ensure_ascii=True)
