"""Nonmutating meaning of stored hosted caption representations.

Raw fields and source addresses remain authoritative storage. These helpers
project text/lifecycle for display, review and export, never normalize history.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterator, Sequence

from .errors import DomainError


def caption_text(record: Any) -> str:
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return ""
    value = record.get("caption") if "caption" in record else record.get("text")
    return str(value or "").strip()


def caption_lifecycle(record: Any, parent: Any = None) -> str:
    def own(value):
        value = value.get("lifecycle_status") if isinstance(value, dict) else None
        status = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
        return status if status in {"active", "superseded", "overflow", "rejected", "deleted"} else "active"
    inherited = own(parent)
    return inherited if inherited != "active" else own(record)


def primary_caption_index(records: Sequence[Any], primary: str) -> int | None:
    primary = str(primary or "").strip()
    if not primary:
        return None

    def order(index):
        record = records[index] if isinstance(records[index], dict) else {}
        try:
            ordinal = max(0, int(record.get("caption_index") or 0))
        except (TypeError, ValueError, OverflowError):
            ordinal = 0
        return (not bool(record.get("is_primary")), ordinal, str(record.get("created_at") or ""), index)

    candidates = [index for index, record in enumerate(records)
                  if caption_lifecycle(record) == "active" and caption_text(record) == primary]
    return min(candidates, key=order) if candidates else None


@dataclass(frozen=True)
class CaptionProjection:
    collection: str
    index: int
    raw: Any

    @property
    def record(self) -> dict:
        return {"caption": self.raw, "source": "manual"} if isinstance(self.raw, str) else self.raw

    @property
    def text(self) -> str:
        return caption_text(self.raw)

    @property
    def lifecycle(self) -> str:
        return caption_lifecycle(self.raw)


def caption_records(annotations: dict) -> Iterator[CaptionProjection]:
    for collection in ("captions", "alternate_captions"):
        values = annotations.get(collection) or []
        if not isinstance(values, list):
            raise DomainError("caption_alternates_format_invalid", 409)
        for index, raw in enumerate(values):
            if isinstance(raw, (str, dict)):
                yield CaptionProjection(collection, index, raw)
