"""Exact mutation preconditions without rewriting legacy command fingerprints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import DomainError


def validate_precondition(value: Any, *, field: str = "expected", validate_editor: bool = True) -> dict:
    if not isinstance(value, dict):
        raise DomainError("invalid_precondition", 422, {"field": field})
    version = value.get("expected_version", value.get("version_id"))
    if not isinstance(version, str) or not version.strip():
        raise DomainError("invalid_precondition", 422, {"field": field + ".expected_version"})
    generation = value.get("generation")
    if type(generation) is not int or generation < 0:
        raise DomainError("invalid_precondition", 422, {"field": field + ".generation"})
    if validate_editor and "editor_id" in value and (not isinstance(value["editor_id"], str) or not value["editor_id"].strip()):
        raise DomainError("invalid_precondition", 422, {"field": field + ".editor_id"})
    return value


class ImagePrecondition(BaseModel):
    # Extra keys and the supplied version spelling are part of legacy hashes.
    model_config = ConfigDict(extra="allow", strict=True)
    generation: int = Field(ge=0)

    @model_validator(mode="after")
    def exact_shape(self):
        try:
            # Validate the effective version, preserving an unused legacy
            # alias and editor_id as exact extras for the receipt hash. New
            # commands validate editor_id inside their transaction.
            validate_precondition(self.model_dump(exclude_unset=True), validate_editor=False)
        except DomainError as exc:
            raise ValueError(exc.detail["field"]) from exc
        return self


def precondition_payload(values: dict[str, ImagePrecondition] | None) -> dict | None:
    return None if values is None else {key: value.model_dump(exclude_unset=True) for key, value in values.items()}
