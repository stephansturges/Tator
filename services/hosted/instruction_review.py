"""Human language-review decisions on the exact saved image revision.

This records review evidence; it does not override the native exporter's
machine validation, quality checks, or corpus readiness calculations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from services.hosted.errors import DomainError
from services.hosted.types import MutationContext
from .caption_semantics import caption_lifecycle, caption_records, caption_text, primary_caption_index


class InstructionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_id: str = Field(min_length=1, max_length=100)
    decision: Literal["accepted", "rejected", "needs_revision", "unreviewed"]
    notes: str = Field(default="", max_length=4000)


class InstructionReviewApply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version: str
    generation: int = Field(ge=0)
    editor_id: str = Field(min_length=8, max_length=128)
    operation_id: str = Field(min_length=8, max_length=128)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions: list[InstructionDecision] = Field(min_length=1, max_length=2000)


def _source_digest(annotations):
    content = {key: annotations.get(key) for key in ("caption", "captions", "alternate_captions")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def review_content_digest(record, kind):
    """Bind human judgment to the text consumed by the native exporter."""
    text = {"kind": kind,
        "question": "Describe this image in detail." if kind == "caption0" else str(record.get("question") or "").strip(),
        "answer": caption_text(record) if kind == "caption0" else str(record.get("answer") or "").strip()}
    return hashlib.sha256(json.dumps(text, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bind_machine_validated_pairs(pairs):
    """Bind actual server outputs; callers must never pass browser-provided QA.

    The durable completed job or trusted proposal, rather than this public hash,
    grants authority to introduce a machine qualification on an annotator save.
    """
    result = copy.deepcopy(pairs) if isinstance(pairs, list) else []
    for pair in result:
        if not isinstance(pair, dict):
            continue
        meta = pair.get("metadata") if isinstance(pair.get("metadata"), dict) else {}
        validation = pair.get("validation") if isinstance(pair.get("validation"), dict) else {}
        statuses = [pair.get("validation_status"), pair.get("review_status"), meta.get("validation_status"),
                    meta.get("review_status"), validation.get("status")]
        if any(re.sub(r"[\s-]+", "_", str(status or "").strip().lower()) == "machine_validated" for status in statuses):
            pair["metadata"] = {**meta, "machine_validation_recorded_via": "hosted_caption_result",
                "machine_validation_content_sha256": review_content_digest(pair, "generated_qa")}
    return result


def _targets(annotations):
    """Yield server-derived addresses; the client cannot choose arbitrary keys."""
    primary = str(annotations.get("caption") or "").strip()
    projections = list(caption_records(annotations))
    selected = primary_caption_index([value.raw for value in projections], primary)
    for position, projected in enumerate(projections):
        collection, index, record = projected.collection, projected.index, projected.record
        if projected.lifecycle != "active":
            continue
        prefix = "captions:" if collection == "captions" else ""
        if projected.text:
            yield f"{prefix}caption:{index}", "caption0", record, (collection, index, None, None), position == selected
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        location = "top" if record.get("generated_qa_pairs") else "metadata"
        pairs = record.get("generated_qa_pairs") or metadata.get("generated_qa_pairs") or []
        if isinstance(pairs, list):
            for number, pair in enumerate(pairs):
                if isinstance(pair, dict) and caption_lifecycle(pair, record) == "active":
                    yield f"{prefix}qa:{index}:{location}:{number}", "generated_qa", pair, (collection, index, location, number), False
    if primary and selected is None:
        yield "primary", "caption0", {"caption": primary, "source": "manual", "lifecycle_status": "active", "is_primary": True}, ("alternate_captions", None, None, None), True



class InstructionReviews:
    def __init__(self, store):
        self.store = store

    def inspect(self, actor, image_id):
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            self.store._can_view(actor, self.store._image(con, image_id))
            image = self.store._image_result(con, image_id, content=True)
        rows = []
        for identifier, kind, value, _, primary in _targets(image["annotations"]):
            meta = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
            changed = meta.get("review_recorded_via") == "hosted_instruction_review" and meta.get("review_content_sha256") != review_content_digest(value, kind)
            machine_changed = kind == "generated_qa" and meta.get("machine_validation_recorded_via") == "hosted_caption_result" and meta.get("machine_validation_content_sha256") != review_content_digest(value, kind)
            rows.append({"target_id": identifier, "kind": kind,
                "question": value.get("question", "Describe this image in detail."),
                "answer": caption_text(value) if kind == "caption0" else value.get("answer", ""), "is_primary": primary,
                "decision": "unreviewed" if changed else meta.get("review_decision", value.get("review_status", "unreviewed")),
                "text_changed_since_review": changed,
                "text_changed_since_machine_validation": machine_changed,
                "notes": meta.get("review_notes", ""), "reviewer": meta.get("reviewer", ""),
                "reviewed_at": meta.get("reviewed_at", ""),
                "validation_status": value.get("validation_status", ""),
                "rejection_reasons": value.get("rejection_reasons", []),
                "source": value.get("source", value.get("answer_source", ""))})
        return {"image_id": image_id, "expected_version": image["current_version"],
            "generation": image["generation"], "source_sha256": _source_digest(image["annotations"]), "rows": rows}

    def apply(self, actor, image_id, body):
        ctx = MutationContext(image_id, body.expected_version, body.generation, body.editor_id, body.operation_id)

        def commit(con):
            self.store._guard(con, actor, ctx)
            version = con.execute("SELECT annotations_json,complete,negative FROM hosted_annotation_versions WHERE id=?", (ctx.expected_version,)).fetchone()
            annotations = json.loads(version["annotations_json"])
            if _source_digest(annotations) != body.source_sha256:
                raise DomainError("instruction_review_source_changed", 409)
            targets = {row[0]: row for row in _targets(annotations)}
            identifiers = [decision.target_id for decision in body.decisions]
            if len(set(identifiers)) != len(identifiers) or any(key not in targets for key in identifiers):
                raise DomainError("instruction_review_target_invalid", 422)
            now = datetime.fromtimestamp(self.store.clock(), timezone.utc).isoformat()
            for decision in body.decisions:
                _, kind, record, (collection, index, location, number), _ = targets[decision.target_id]
                if index is None:
                    if annotations.get("alternate_captions") is None:
                        annotations["alternate_captions"] = []
                    annotations.setdefault("alternate_captions", []).append(record)
                elif location is None:
                    # String-only imported captions acquire review metadata
                    # without changing their text or inventing model evidence.
                    annotations[collection][index] = record
                meta = dict(record.get("metadata")) if isinstance(record.get("metadata"), dict) else {}
                meta.update(review_decision=decision.decision, review_status=decision.decision, review_notes=decision.notes.strip(),
                    reviewer=actor.user_id, reviewed_at=now, review_row_origin=kind,
                    review_recorded_via="hosted_instruction_review", review_source_version=ctx.expected_version,
                    review_content_sha256=review_content_digest(record, kind))
                record.update(metadata=meta, review_status=decision.decision, updated_at=now)
            return self.store._save(con, actor, ctx, annotations,
                complete=bool(version["complete"]), negative=bool(version["negative"]))

        return self.store._command(actor, ctx.operation_id, "review_caption_instructions",
            {"image_id": image_id, **body.model_dump()}, commit, admin=True)
