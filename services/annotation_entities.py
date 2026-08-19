"""Stable annotation-entity identity and revision-safe mutation helpers.

The Data Quality Explorer projects annotations into a derived graph.  Geometry,
class, and source-row hints are useful migration evidence, but they are not an
entity identity: duplicates can share all three and relabeling deliberately
changes one of them.  This module keeps identity separate from mutable label
content and provides a deliberately conservative migration for legacy records.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import uuid
from typing import Any, Iterable, Mapping, Sequence


ANNOTATION_ENTITY_SCHEMA_VERSION = 2


class AnnotationEntityError(ValueError):
    """Base class for annotation entity contract failures."""


class AnnotationIdentityConflict(AnnotationEntityError):
    """Legacy rows cannot be mapped to stable entities without guessing."""


class AnnotationRevisionConflict(AnnotationEntityError):
    """A caller attempted to mutate an entity or record from a stale revision."""


class AnnotationEntityNotFound(AnnotationEntityError):
    """A requested stable annotation entity is not present in the record."""


def _canonical_number(value: str) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise AnnotationIdentityConflict("label rows must contain finite numbers")
    if number == 0:
        number = 0.0
    return format(number, ".12g")


def canonical_label_line(line: Any) -> str:
    tokens = str(line or "").strip().split()
    if len(tokens) < 5:
        raise AnnotationIdentityConflict("label rows must contain a class and four coordinates")
    try:
        class_id = str(int(float(tokens[0])))
        geometry = [_canonical_number(token) for token in tokens[1:5]]
    except (TypeError, ValueError) as exc:
        raise AnnotationIdentityConflict("label rows must be numeric") from exc
    suffix = tokens[5:]
    return " ".join([class_id, *geometry, *suffix])


def label_geometry_key(line: Any) -> tuple[str, str, str, str]:
    tokens = canonical_label_line(line).split()
    return tuple(tokens[1:5])  # type: ignore[return-value]


def source_annotation_entity_id(source_identity: str, source_row_ordinal: int) -> str:
    material = f"{source_identity}\0{int(source_row_ordinal)}".encode("utf-8")
    return "ae1_" + hashlib.sha256(material).hexdigest()[:32]


def new_annotation_entity_id() -> str:
    return "ae2_" + uuid.uuid4().hex


def _entity_payload_for_revision(entity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "annotation_entity_id": str(entity.get("annotation_entity_id") or ""),
        "entity_revision": int(entity.get("entity_revision") or 0),
        "source_row_ordinal": entity.get("source_row_ordinal"),
        "label_line": str(entity.get("label_line") or ""),
        "deleted": bool(entity.get("deleted")),
    }


def annotation_record_revision(record: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": ANNOTATION_ENTITY_SCHEMA_VERSION,
        "annotation_source_identity": str(record.get("annotation_source_identity") or ""),
        "entities": [
            _entity_payload_for_revision(entity)
            for entity in record.get("entities", [])
            if isinstance(entity, Mapping)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "alr1_" + hashlib.sha256(encoded).hexdigest()


def _build_record(source_identity: str, entities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": ANNOTATION_ENTITY_SCHEMA_VERSION,
        "annotation_source_identity": str(source_identity),
        "entities": [dict(entity) for entity in entities],
    }
    record["record_revision"] = annotation_record_revision(record)
    return record


def _normalise_existing_record(record: Mapping[str, Any]) -> dict[str, Any]:
    source_identity = str(record.get("annotation_source_identity") or "")
    seen: set[str] = set()
    entities: list[dict[str, Any]] = []
    raw_entities = record.get("entities")
    if not isinstance(raw_entities, list):
        raise AnnotationIdentityConflict("annotation entity record has no entity list")
    for raw in raw_entities:
        if not isinstance(raw, Mapping):
            raise AnnotationIdentityConflict("annotation entity record contains an invalid entity")
        entity_id = str(raw.get("annotation_entity_id") or "")
        if not entity_id or entity_id in seen:
            raise AnnotationIdentityConflict("annotation entity IDs must be present and unique")
        seen.add(entity_id)
        deleted = bool(raw.get("deleted"))
        label_line = str(raw.get("label_line") or "")
        if not deleted:
            label_line = canonical_label_line(label_line)
        entities.append(
            {
                "annotation_entity_id": entity_id,
                "entity_revision": max(1, int(raw.get("entity_revision") or 1)),
                "source_row_ordinal": raw.get("source_row_ordinal"),
                "label_line": label_line,
                "deleted": deleted,
            }
        )
    return _build_record(source_identity, entities)


def _match_rows_to_entities(
    entities: Sequence[Mapping[str, Any]],
    current_lines: Sequence[str],
) -> tuple[dict[int, int], set[int], set[int]]:
    """Return current-row -> entity-index mapping without resolving ambiguities."""

    unmatched_entities = {
        index for index, entity in enumerate(entities) if not bool(entity.get("deleted"))
    }
    unmatched_rows = set(range(len(current_lines)))
    matches: dict[int, int] = {}

    exact_entities: dict[str, deque[int]] = defaultdict(deque)
    for index in sorted(unmatched_entities):
        exact_entities[canonical_label_line(entities[index].get("label_line"))].append(index)
    for row_index, line in enumerate(current_lines):
        queue = exact_entities.get(line)
        if queue:
            entity_index = queue.popleft()
            matches[row_index] = entity_index
            unmatched_rows.discard(row_index)
            unmatched_entities.discard(entity_index)

    by_geometry_entities: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    by_geometry_rows: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index in unmatched_entities:
        by_geometry_entities[label_geometry_key(entities[index].get("label_line"))].append(index)
    for row_index in unmatched_rows:
        by_geometry_rows[label_geometry_key(current_lines[row_index])].append(row_index)

    for geometry in sorted(set(by_geometry_entities) & set(by_geometry_rows)):
        entity_indices = by_geometry_entities[geometry]
        row_indices = by_geometry_rows[geometry]
        if len(entity_indices) == 1 and len(row_indices) == 1:
            entity_index = entity_indices[0]
            row_index = row_indices[0]
            matches[row_index] = entity_index
            unmatched_entities.discard(entity_index)
            unmatched_rows.discard(row_index)
            continue
        raise AnnotationIdentityConflict(
            "multiple legacy annotations share mutable geometry; migration would be ambiguous"
        )
    return matches, unmatched_entities, unmatched_rows


def migrate_legacy_annotation_record(
    *,
    source_identity: str,
    source_lines: Iterable[Any],
    current_lines: Iterable[Any],
    existing_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or reconcile a stable entity record without guessing.

    Exact duplicate source rows remain distinct because source-row ordinal is
    immutable.  A class-only legacy edit is migrated only when the remaining
    geometry match is one-to-one.  Any many-to-many geometry match fails closed.
    """

    source_identity = str(source_identity or "").strip()
    if not source_identity:
        raise AnnotationIdentityConflict("annotation source identity is required")
    source = [canonical_label_line(line) for line in source_lines]
    current = [canonical_label_line(line) for line in current_lines]

    if existing_record is None:
        entities: list[dict[str, Any]] = [
            {
                "annotation_entity_id": source_annotation_entity_id(source_identity, ordinal),
                "entity_revision": 1,
                "source_row_ordinal": ordinal,
                "label_line": line,
                "deleted": False,
            }
            for ordinal, line in enumerate(source)
        ]
    else:
        normalised = _normalise_existing_record(existing_record)
        if normalised["annotation_source_identity"] != source_identity:
            raise AnnotationIdentityConflict("annotation source identity changed")
        entities = deepcopy(normalised["entities"])

    matches, unmatched_entities, unmatched_rows = _match_rows_to_entities(entities, current)

    for row_index, entity_index in matches.items():
        entity = entities[entity_index]
        line = current[row_index]
        if canonical_label_line(entity.get("label_line")) != line or bool(entity.get("deleted")):
            entity["label_line"] = line
            entity["deleted"] = False
            entity["entity_revision"] = int(entity.get("entity_revision") or 0) + 1

    for entity_index in unmatched_entities:
        entity = entities[entity_index]
        if not bool(entity.get("deleted")):
            entity["deleted"] = True
            entity["entity_revision"] = int(entity.get("entity_revision") or 0) + 1

    for row_index in sorted(unmatched_rows):
        entities.append(
            {
                "annotation_entity_id": new_annotation_entity_id(),
                "entity_revision": 1,
                "source_row_ordinal": None,
                "label_line": current[row_index],
                "deleted": False,
            }
        )

    return _build_record(source_identity, entities)


def live_annotation_entities(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalised = _normalise_existing_record(record)
    return [entity for entity in normalised["entities"] if not entity["deleted"]]


def live_label_lines(record: Mapping[str, Any]) -> list[str]:
    return [str(entity["label_line"]) for entity in live_annotation_entities(record)]


def bind_label_rows_to_entities(
    record: Mapping[str, Any],
    label_lines: Sequence[str],
) -> list[dict[str, Any]]:
    """Bind each current label row to exactly one stable annotation entity.

    Source-row ordinals preserve the identity of geometrically identical boxes.
    A content fallback is accepted only when it has one unused candidate; an
    ambiguous legacy row must remain review-only rather than being guessed.
    """
    entities = live_annotation_entities(record)
    by_ordinal = {
        int(entity["source_row_ordinal"]): entity
        for entity in entities
        if entity.get("source_row_ordinal") is not None
    }
    unused = {str(entity["annotation_entity_id"]) for entity in entities}
    bound: list[dict[str, Any]] = []
    for row_ordinal, raw_line in enumerate(label_lines):
        line = canonical_label_line(raw_line)
        entity = by_ordinal.get(row_ordinal)
        if (
            entity is not None
            and str(entity["annotation_entity_id"]) in unused
            and canonical_label_line(entity.get("label_line")) == line
        ):
            unused.remove(str(entity["annotation_entity_id"]))
            bound.append(dict(entity))
            continue
        candidates = [
            candidate
            for candidate in entities
            if str(candidate["annotation_entity_id"]) in unused
            and canonical_label_line(candidate.get("label_line")) == line
        ]
        if len(candidates) != 1:
            raise AnnotationIdentityConflict(
                f"label row {row_ordinal} does not bind to exactly one annotation entity"
            )
        entity = candidates[0]
        unused.remove(str(entity["annotation_entity_id"]))
        bound.append(dict(entity))
    if unused:
        raise AnnotationIdentityConflict(
            "annotation entity record contains live rows absent from current labels"
        )
    return bound


@dataclass(frozen=True)
class AnnotationEntityActionResult:
    annotation_entity_id: str
    action: str
    entity_revision: int
    label_line: str
    deleted: bool


def apply_annotation_entity_actions(
    record: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    *,
    expected_record_revision: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply an ordered, atomic set of entity mutations with CAS preconditions."""

    working = _normalise_existing_record(record)
    current_revision = str(working["record_revision"])
    if expected_record_revision is None:
        raise AnnotationRevisionConflict("annotation record revision is required")
    if str(expected_record_revision) != current_revision:
        raise AnnotationRevisionConflict("annotation record revision changed")

    entities_by_id = {
        str(entity["annotation_entity_id"]): entity for entity in working["entities"]
    }
    requested_ids: set[str] = set()
    prepared: list[tuple[dict[str, Any], str, Mapping[str, Any]]] = []
    for raw_action in actions:
        entity_id = str(raw_action.get("annotation_entity_id") or "")
        action = str(raw_action.get("action") or "").strip().lower()
        if not entity_id or entity_id in requested_ids:
            raise AnnotationIdentityConflict("each transaction action must target one unique entity")
        requested_ids.add(entity_id)
        entity = entities_by_id.get(entity_id)
        if entity is None:
            raise AnnotationEntityNotFound(entity_id)
        expected_entity_revision = raw_action.get("expected_entity_revision")
        if expected_entity_revision is None:
            raise AnnotationRevisionConflict(
                f"annotation entity revision is required: {entity_id}"
            )
        if int(expected_entity_revision) != int(
            entity["entity_revision"]
        ):
            raise AnnotationRevisionConflict(f"annotation entity revision changed: {entity_id}")
        if bool(entity.get("deleted")):
            raise AnnotationRevisionConflict(f"annotation entity is already deleted: {entity_id}")
        if action not in {"relabel", "delete"}:
            raise AnnotationIdentityConflict(f"unsupported annotation entity action: {action}")
        if action == "relabel":
            try:
                target_class_value = float(str(raw_action.get("target_class_id")))
            except (TypeError, ValueError) as exc:
                raise AnnotationIdentityConflict("relabel requires a numeric target_class_id") from exc
            if (
                not math.isfinite(target_class_value)
                or not target_class_value.is_integer()
                or target_class_value < 0
            ):
                raise AnnotationIdentityConflict(
                    "relabel requires a non-negative integer target_class_id"
                )
        prepared.append((entity, action, raw_action))

    results: list[dict[str, Any]] = []
    for entity, action, raw_action in prepared:
        if action == "delete":
            entity["deleted"] = True
        else:
            tokens = canonical_label_line(entity["label_line"]).split()
            tokens[0] = str(int(float(str(raw_action["target_class_id"]))))
            entity["label_line"] = " ".join(tokens)
        entity["entity_revision"] = int(entity["entity_revision"]) + 1
        results.append(
            {
                "annotation_entity_id": str(entity["annotation_entity_id"]),
                "action": action,
                "entity_revision": int(entity["entity_revision"]),
                "label_line": str(entity["label_line"]),
                "deleted": bool(entity["deleted"]),
            }
        )

    updated = _build_record(str(working["annotation_source_identity"]), working["entities"])
    return updated, results
