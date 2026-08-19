from __future__ import annotations

import pytest

from services.annotation_entities import (
    AnnotationIdentityConflict,
    AnnotationRevisionConflict,
    apply_annotation_entity_actions,
    bind_label_rows_to_entities,
    live_label_lines,
    migrate_legacy_annotation_record,
)


SOURCE = "asi1_test_source"


def test_duplicate_source_rows_receive_distinct_stable_identities() -> None:
    line = "2 0.5 0.5 0.2 0.3"
    record = migrate_legacy_annotation_record(
        source_identity=SOURCE,
        source_lines=[line, line],
        current_lines=[line, line],
    )

    entities = record["entities"]
    assert len(entities) == 2
    assert entities[0]["annotation_entity_id"] != entities[1]["annotation_entity_id"]


def test_relabel_preserves_identity_and_increments_entity_revision() -> None:
    record = migrate_legacy_annotation_record(
        source_identity=SOURCE,
        source_lines=["2 0.5 0.5 0.2 0.3"],
        current_lines=["2 0.5 0.5 0.2 0.3"],
    )
    entity = record["entities"][0]

    updated, results = apply_annotation_entity_actions(
        record,
        [
            {
                "annotation_entity_id": entity["annotation_entity_id"],
                "expected_entity_revision": entity["entity_revision"],
                "action": "relabel",
                "target_class_id": 7,
            }
        ],
        expected_record_revision=record["record_revision"],
    )

    assert results[0]["annotation_entity_id"] == entity["annotation_entity_id"]
    assert results[0]["entity_revision"] == entity["entity_revision"] + 1
    assert live_label_lines(updated) == ["7 0.5 0.5 0.2 0.3"]


def test_delete_targets_only_one_exact_duplicate() -> None:
    line = "2 0.5 0.5 0.2 0.3"
    record = migrate_legacy_annotation_record(
        source_identity=SOURCE,
        source_lines=[line, line],
        current_lines=[line, line],
    )
    target = record["entities"][1]

    updated, _ = apply_annotation_entity_actions(
        record,
        [
            {
                "annotation_entity_id": target["annotation_entity_id"],
                "expected_entity_revision": 1,
                "action": "delete",
            }
        ],
        expected_record_revision=record["record_revision"],
    )

    assert live_label_lines(updated) == [line]
    assert updated["entities"][0]["deleted"] is False
    assert updated["entities"][1]["deleted"] is True


def test_stale_record_and_entity_revisions_fail_closed() -> None:
    record = migrate_legacy_annotation_record(
        source_identity=SOURCE,
        source_lines=["2 0.5 0.5 0.2 0.3"],
        current_lines=["2 0.5 0.5 0.2 0.3"],
    )
    entity_id = record["entities"][0]["annotation_entity_id"]

    with pytest.raises(AnnotationRevisionConflict):
        apply_annotation_entity_actions(
            record,
            [{"annotation_entity_id": entity_id, "action": "delete"}],
            expected_record_revision="alr1_stale",
        )
    with pytest.raises(AnnotationRevisionConflict):
        apply_annotation_entity_actions(
            record,
            [
                {
                    "annotation_entity_id": entity_id,
                    "expected_entity_revision": 99,
                    "action": "delete",
                }
            ],
        )
    with pytest.raises(AnnotationRevisionConflict, match="record revision is required"):
        apply_annotation_entity_actions(
            record,
            [
                {
                    "annotation_entity_id": entity_id,
                    "expected_entity_revision": 1,
                    "action": "delete",
                }
            ],
        )
    with pytest.raises(AnnotationRevisionConflict, match="entity revision is required"):
        apply_annotation_entity_actions(
            record,
            [{"annotation_entity_id": entity_id, "action": "delete"}],
            expected_record_revision=record["record_revision"],
        )


def test_legacy_class_only_change_migrates_when_geometry_is_unique() -> None:
    record = migrate_legacy_annotation_record(
        source_identity=SOURCE,
        source_lines=["2 0.5 0.5 0.2 0.3"],
        current_lines=["7 0.5 0.5 0.2 0.3"],
    )

    assert record["entities"][0]["annotation_entity_id"].startswith("ae1_")
    assert live_label_lines(record) == ["7 0.5 0.5 0.2 0.3"]


def test_legacy_many_to_many_geometry_change_is_not_guessed() -> None:
    with pytest.raises(AnnotationIdentityConflict):
        migrate_legacy_annotation_record(
            source_identity=SOURCE,
            source_lines=["2 0.5 0.5 0.2 0.3", "3 0.5 0.5 0.2 0.3"],
            current_lines=["7 0.5 0.5 0.2 0.3", "8 0.5 0.5 0.2 0.3"],
        )


def test_row_binding_preserves_duplicate_source_ordinals() -> None:
    line = "2 0.5 0.5 0.2 0.3"
    record = migrate_legacy_annotation_record(
        source_identity=SOURCE,
        source_lines=[line, line],
        current_lines=[line, line],
    )

    bound = bind_label_rows_to_entities(record, [line, line])

    assert [entity["source_row_ordinal"] for entity in bound] == [0, 1]
    assert bound[0]["annotation_entity_id"] != bound[1]["annotation_entity_id"]


def test_relabel_rejects_fractional_class_id() -> None:
    record = migrate_legacy_annotation_record(
        source_identity=SOURCE,
        source_lines=["2 0.5 0.5 0.2 0.3"],
        current_lines=["2 0.5 0.5 0.2 0.3"],
    )
    entity = record["entities"][0]

    with pytest.raises(AnnotationIdentityConflict, match="non-negative integer"):
        apply_annotation_entity_actions(
            record,
            [
                {
                    "annotation_entity_id": entity["annotation_entity_id"],
                    "expected_entity_revision": entity["entity_revision"],
                    "action": "relabel",
                    "target_class_id": 1.5,
                }
            ],
            expected_record_revision=record["record_revision"],
        )
