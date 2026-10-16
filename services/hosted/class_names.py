"""Class names must remain one entry in every supported text labelmap reader."""
from __future__ import annotations

import re

from .errors import DomainError


LINE_SEPARATORS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")


def validate_class_name(value: str) -> str:
    """Preserve existing trimming without hiding internal line delimiters."""
    if not isinstance(value, str) or not value.strip() or len(value) > 200 or "\0" in value:
        raise DomainError("invalid_field", 422, {"field": "class_name"})
    name = value.strip()
    if any(character in LINE_SEPARATORS for character in name):
        raise DomainError("invalid_class_name", 422, {"field": "class_name"})
    return name


def parse_class_lines(value: str) -> list[str]:
    """Split physical CR/LF lines before checking other Unicode separators."""
    return [validate_class_name(line) for line in re.split(r"\r\n|\r|\n", value) if line.strip()]


def validate_snapshot_classes(manifest: dict) -> None:
    """Historical maps stay readable; incompatible new exports fail explicitly."""
    for record in manifest["classes"]:
        try:
            validate_class_name(record.get("name"))
        except DomainError as exc:
            raise DomainError("invalid_snapshot_class_names", 409,
                              {"class_id": record.get("id"), "name": record.get("name")}) from exc
