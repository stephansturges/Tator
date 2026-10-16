"""Known language containers consumed by review and native instruction export.

Only container shape is constrained. Legacy nulls, plain caption strings,
unknown extensions and individual provenance-list members remain unchanged.
"""
from __future__ import annotations

from typing import Any

from .errors import DomainError


def validate_language_annotations(annotations: dict[str, Any]) -> None:
    def typed(value, field, kind):
        if value is not None and not isinstance(value, kind):
            raise DomainError("invalid_annotation_language", 422,
                              {"field": field, "expected": kind.__name__})

    def provenance(value, field):
        typed(value, field, dict)
        if value is None:
            return
        for key in ("validated_against", "source_fields"):
            typed(value.get(key), f"{field}.{key}", list)

    def qa_pairs(value, field):
        typed(value, field, list)
        for index, pair in enumerate(value or []):
            path = f"{field}[{index}]"
            if not isinstance(pair, dict):
                raise DomainError("invalid_annotation_language", 422, {"field": path, "expected": "dict"})
            for key in ("question", "answer"):
                typed(pair.get(key), f"{path}.{key}", str)
            provenance(pair, path)
            metadata = pair.get("metadata")
            provenance(metadata, f"{path}.metadata")
            # Both locations matter even when one currently shadows the other.
            provenance(pair.get("validation"), f"{path}.validation")
            provenance((metadata or {}).get("validation"), f"{path}.metadata.validation")

    typed(annotations.get("caption"), "caption", str)
    for collection in ("captions", "alternate_captions"):
        values = annotations.get(collection)
        typed(values, collection, list)
        for index, record in enumerate(values or []):
            if isinstance(record, str):
                continue
            path = f"{collection}[{index}]"
            if not isinstance(record, dict):
                raise DomainError("invalid_annotation_language", 422, {"field": path, "expected": "dict or str"})
            for key in ("caption", "text"):
                typed(record.get(key), f"{path}.{key}", str)
            metadata = record.get("metadata")
            typed(metadata, f"{path}.metadata", dict)
            qa_pairs(record.get("generated_qa_pairs"), f"{path}.generated_qa_pairs")
            qa_pairs((metadata or {}).get("generated_qa_pairs"), f"{path}.metadata.generated_qa_pairs")


def validate_instruction_manifest(manifest: dict[str, Any]) -> None:
    """Reject historical invalid content before native export; never repair it."""
    for image in manifest.get("images", []):
        try:
            validate_language_annotations(image["annotations"])
        except DomainError as exc:
            raise DomainError("invalid_annotation_language", 422,
                              {"image_id": image["id"], **exc.detail}) from exc
