"""Recoverable administration and frozen canonical filesystem adapters.

No method discovers canonical membership by scanning a directory. Archive is a
database visibility transition; retained versions and releases continue to own
their media. Only separately observed, completely unreferenced objects can be
physically purged after the minimum retention period.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .artifacts import ArtifactRef, ArtifactStore, fsync_directory
from .auth import AuthService
from .backup import ExportEntry, create_canonical_zip
from .errors import DomainError
from .store import SCHEMA_VERSION, HostedStore, _id, _json, _text
from .types import Actor, MutationContext


MINIMUM_RETENTION_SECONDS = 30 * 24 * 60 * 60


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


class HostedOperations:
    def __init__(self, store: HostedStore, artifacts: ArtifactStore,
                 root: str | Path | None = None) -> None:
        self.store = store
        self.artifacts = artifacts
        self.root = Path(root) if root is not None else artifacts.root.parent
        self.root.mkdir(parents=True, exist_ok=True)
        with store.transaction() as con:
            for statement in (
                """CREATE TABLE IF NOT EXISTS hosted_artifact_references (
                    kind TEXT NOT NULL, owner_id TEXT NOT NULL, sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(kind,owner_id,sha256))""",
                """CREATE TABLE IF NOT EXISTS hosted_exports (
                    id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL REFERENCES hosted_snapshots(id),
                    sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                    actor_id TEXT NOT NULL, created_at REAL NOT NULL)""",
                """CREATE TABLE IF NOT EXISTS hosted_orphan_observations (
                    sha256 TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL,
                    first_observed_at REAL NOT NULL)""",
                """CREATE TABLE IF NOT EXISTS hosted_purge_commands (
                    actor_id TEXT NOT NULL, operation_id TEXT NOT NULL, request_hash TEXT NOT NULL,
                    state TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(actor_id,operation_id))""",
                """CREATE TABLE IF NOT EXISTS hosted_purge_items (
                    actor_id TEXT NOT NULL, operation_id TEXT NOT NULL, sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, state TEXT NOT NULL,
                    PRIMARY KEY(actor_id,operation_id,sha256),
                    FOREIGN KEY(actor_id,operation_id) REFERENCES hosted_purge_commands(actor_id,operation_id))""",
            ):
                con.execute(statement)

    @staticmethod
    def artifact_references(connection: sqlite3.Connection) -> list[dict[str, Any]]:
        """Complete central registry, usable against a captured backup database.

        Runtime/model/configuration writers must register immutable custom
        assets here; absolute paths are never recovery references.
        """
        refs: dict[str, int] = {}
        for table in ("hosted_media", "hosted_artifact_references"):
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                if table == "hosted_media":
                    raise DomainError("hosted_schema_missing", 409)
                continue
            for row in connection.execute(f"SELECT sha256,size_bytes FROM {table}"):
                ref = ArtifactRef(row[0], row[1])
                if ref.sha256 in refs and refs[ref.sha256] != ref.size_bytes:
                    raise DomainError("artifact_size_conflict", 409)
                refs[ref.sha256] = ref.size_bytes
        return [{"sha256": digest, "size_bytes": size} for digest, size in sorted(refs.items())]

    @staticmethod
    def _referenced(con: sqlite3.Connection, digest: str) -> bool:
        return con.execute("SELECT 1 FROM hosted_media WHERE sha256=? UNION ALL "
                           "SELECT 1 FROM hosted_artifact_references WHERE sha256=? LIMIT 1", (digest, digest)).fetchone() is not None

    def register_artifact(self, actor: Actor, reference: ArtifactRef, kind: str,
                          owner_id: str, operation_id: str) -> dict[str, Any]:
        kind, owner_id = _text(kind, "kind", 100), _text(owner_id, "owner_id", 200)
        with self.artifacts.exclusive():
            self.artifacts.verify(reference)
            def run(con: sqlite3.Connection) -> dict[str, Any]:
                known = con.execute("SELECT size_bytes FROM hosted_artifact_references WHERE sha256=? UNION ALL "
                                    "SELECT size_bytes FROM hosted_media WHERE sha256=?", (reference.sha256, reference.sha256)).fetchall()
                if any(row[0] != reference.size_bytes for row in known):
                    raise DomainError("artifact_size_conflict", 409)
                con.execute("INSERT OR IGNORE INTO hosted_artifact_references VALUES (?,?,?,?,?)",
                            (kind, owner_id, reference.sha256, reference.size_bytes, self.store.clock()))
                con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (reference.sha256,))
                return {"kind": kind, "owner_id": owner_id, **reference.as_dict()}
            return self.store._command(actor, operation_id, "register_artifact", {"kind": kind, "owner_id": owner_id, **reference.as_dict()}, run, admin=True)

    def _archive(self, con: sqlite3.Connection, actor: Actor, *, kind: str,
                 target_id: str, dataset_id: str, images: list[sqlite3.Row], reason: str) -> dict[str, Any]:
        archive_id = _id("archive")
        captures = [{"image_id": image["id"], "version_id": image["current_version"],
                     "generation": image["generation"], "state": image["state"],
                     "media_hash": image["media_hash"]} for image in images]
        now = self.store.clock()
        con.execute("INSERT INTO hosted_archives(id,kind,target_id,dataset_id,state,archived_at,actor_id,reason,captured_json) "
                    "VALUES (?,?,?,?,'archived',?,?,?,?)", (archive_id, kind, target_id, dataset_id, now, actor.user_id, reason, _json(captures)))
        for image in images:
            con.execute("INSERT INTO hosted_archived_images VALUES (?,?,?)", (image["id"], archive_id, image["current_version"]))
            con.execute("UPDATE hosted_images SET generation=generation+1,updated_at=? WHERE id=?", (now, image["id"]))
            con.execute("DELETE FROM hosted_editor_leases WHERE image_id=?", (image["id"],))
            self.store._event(con, actor, dataset_id, image["id"], "image_archived", {"archive_id": archive_id, "version_id": image["current_version"], "reason": reason})
        if kind == "dataset":
            con.execute("INSERT INTO hosted_archived_datasets VALUES (?,?)", (dataset_id, archive_id))
            self.store._event(con, actor, dataset_id, None, "dataset_archived", {"archive_id": archive_id, "reason": reason})
        con.execute("UPDATE hosted_datasets SET revision=revision+1 WHERE id=?", (dataset_id,))
        return {"archive_id": archive_id, "kind": kind, "target_id": target_id, "dataset_id": dataset_id,
                "state": "archived", "image_count": len(images), "archived_at": now,
                "earliest_purge_at": now + MINIMUM_RETENTION_SECONDS, "files_deleted": 0}

    def archive_image(self, actor: Actor, ctx: MutationContext, reason: str = "") -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            image = self.store._guard(con, actor, ctx)
            return self._archive(con, actor, kind="image", target_id=ctx.image_id,
                                 dataset_id=image["dataset_id"], images=[image], reason=reason)
        return self.store._command(actor, ctx.operation_id, "archive_image", {"context": asdict(ctx), "reason": reason}, run, admin=True)

    def archive_dataset(self, actor: Actor, dataset_id: str, expected_images: dict[str, dict[str, Any]],
                        operation_id: str, reason: str = "", expected_revision: int | None = None) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            from .preconditions import validate_precondition
            if not isinstance(expected_images, dict):
                raise DomainError("invalid_precondition", 422, {"field": "expected_images"})
            for identifier, expected in expected_images.items():
                validate_precondition(expected, field=f"expected_images.{identifier}")
            dataset = self.store._dataset(con, dataset_id)
            if expected_revision is None or dataset["revision"] != expected_revision:
                raise DomainError("dataset_revision_changed", 409)
            images = con.execute("SELECT * FROM hosted_images WHERE dataset_id=? AND id NOT IN (SELECT image_id FROM hosted_archived_images) ORDER BY id", (dataset_id,)).fetchall()
            if set(expected_images) != {image["id"] for image in images}:
                raise DomainError("dataset_membership_changed", 409)
            for image in images:
                self.store._can_edit(actor, image)  # Requires explicit override first.
                expected = expected_images[image["id"]]
                self.store._expected(image, expected.get("expected_version", expected.get("version_id")), expected.get("generation"))
                lease = con.execute("SELECT * FROM hosted_editor_leases WHERE image_id=?", (image["id"],)).fetchone()
                if lease and lease["expires_at"] > self.store.clock():
                    context = MutationContext(image["id"], image["current_version"], image["generation"], expected.get("editor_id", ""), operation_id)
                    self.store._guard(con, actor, context)
            return self._archive(con, actor, kind="dataset", target_id=dataset_id,
                                 dataset_id=dataset_id, images=images, reason=reason)
        return self.store._command(actor, operation_id, "archive_dataset", dict(dataset_id=dataset_id,expected_images=expected_images,reason=reason,expected_revision=expected_revision), run, admin=True)

    def list_archives(self, actor: Actor, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        self.store._pagination(limit, offset)
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            return [dict(row) for row in con.execute("SELECT id AS archive_id,kind,target_id,dataset_id,state,archived_at,actor_id,reason,restored_at,restored_by FROM hosted_archives ORDER BY archived_at DESC,id LIMIT ? OFFSET ?", (limit, offset))]

    def restore_archive(self, actor: Actor, archive_id: str, operation_id: str) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            archive = con.execute("SELECT * FROM hosted_archives WHERE id=?", (archive_id,)).fetchone()
            if not archive:
                raise DomainError("archive_not_found", 404)
            if archive["state"] != "archived":
                raise DomainError("archive_already_restored", 409)
            parent = con.execute("SELECT archive_id FROM hosted_archived_datasets WHERE dataset_id=?", (archive["dataset_id"],)).fetchone()
            if parent and parent[0] != archive_id:
                raise DomainError("dataset_still_archived", 409)
            members = con.execute("SELECT * FROM hosted_archived_images WHERE archive_id=?", (archive_id,)).fetchall()
            captures = {row["image_id"]: row for row in json.loads(archive["captured_json"])}
            if set(captures) != {member["image_id"] for member in members}:
                raise DomainError("archive_membership_changed", 409)
            for member in members:
                image = self.store._image(con, member["image_id"], include_archived=True)
                if image["current_version"] != member["version_id"] or image["current_version"] != captures[image["id"]]["version_id"]:
                    raise DomainError("archive_version_changed", 409)
                con.execute("UPDATE hosted_images SET generation=generation+1,updated_at=? WHERE id=?", (self.store.clock(), image["id"]))
                self.store._event(con, actor, archive["dataset_id"], image["id"], "image_restored", {"archive_id": archive_id})
            con.execute("DELETE FROM hosted_archived_images WHERE archive_id=?", (archive_id,))
            con.execute("DELETE FROM hosted_archived_datasets WHERE archive_id=?", (archive_id,))
            con.execute("UPDATE hosted_archives SET state='restored',restored_at=?,restored_by=? WHERE id=?", (self.store.clock(), actor.user_id, archive_id))
            con.execute("UPDATE hosted_datasets SET revision=revision+1 WHERE id=?", (archive["dataset_id"],))
            return {"archive_id": archive_id, "state": "restored", "image_count": len(members), "files_deleted": 0}
        return self.store._command(actor, operation_id, "restore_archive", {"archive_id": archive_id}, run, admin=True)

    def _purge_preview(self, con: sqlite3.Connection, archive_id: str) -> dict[str, Any]:
        archive = con.execute("SELECT * FROM hosted_archives WHERE id=?", (archive_id,)).fetchone()
        if not archive or archive["state"] != "archived":
            raise DomainError("archive_not_found", 404)
        captures = json.loads(archive["captured_json"])
        digests = {capture["media_hash"] for capture in captures}
        snapshots = Counter()
        for row in con.execute("SELECT manifest_json FROM hosted_snapshots"):
            snapshots.update(image["media_hash"] for image in json.loads(row[0])["images"])
        rows = []
        for digest in sorted(digests):
            members = con.execute("SELECT COUNT(*) FROM hosted_images WHERE media_hash=?", (digest,)).fetchone()[0]
            history = con.execute("SELECT COUNT(*) FROM hosted_annotation_versions v JOIN hosted_images i ON i.id=v.image_id WHERE i.media_hash=?", (digest,)).fetchone()[0]
            additional = con.execute("SELECT COUNT(*) FROM hosted_artifact_references WHERE sha256=?", (digest,)).fetchone()[0]
            rows.append({"sha256": digest, "dataset_memberships": members, "annotation_versions": history,
                         "snapshot_references": snapshots[digest], "other_references": additional})
        eligible_at = archive["archived_at"] + MINIMUM_RETENTION_SECONDS
        result = {"archive_id": archive_id, "eligible_at": eligible_at,
                  "retention_elapsed": self.store.clock() >= eligible_at,
                  "references": rows, "can_purge": False,
                  "reason": "retained_annotation_history", "files_deleted": 0}
        result["preview_hash"] = _digest(result)
        return result

    def purge_preview(self, actor: Actor, archive_id: str) -> dict[str, Any]:
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            return self._purge_preview(con, archive_id)

    def purge(self, actor: Actor, archive_id: str, preview_hash: str, operation_id: str) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            preview = self._purge_preview(con, archive_id)
            if preview["preview_hash"] != preview_hash:
                raise DomainError("purge_preview_changed", 409)
            if not preview["retention_elapsed"]:
                raise DomainError("purge_retention_not_elapsed", 409, preview)
            # Retaining immutable versions is deliberate. Deleting media here
            # would make history/restore misleading even without active members.
            raise DomainError("purge_references_retained", 409, preview)
        with self.artifacts.exclusive():
            return self.store._command(actor, operation_id, "purge_archive", dict(archive_id=archive_id,preview_hash=preview_hash), run, admin=True)

    def _orphan_preview(self, con: sqlite3.Connection, physical: dict[str, ArtifactRef]) -> dict[str, Any]:
        rows = []
        now = self.store.clock()
        for digest, reference in sorted(physical.items()):
            referenced = self._referenced(con, digest)
            if referenced:
                con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (digest,))
                eligible_at = None
            else:
                con.execute("INSERT OR IGNORE INTO hosted_orphan_observations VALUES (?,?,?)", (digest, reference.size_bytes, now))
                observed = con.execute("SELECT * FROM hosted_orphan_observations WHERE sha256=?", (digest,)).fetchone()
                if observed["size_bytes"] != reference.size_bytes:
                    raise DomainError("artifact_size_conflict", 409)
                eligible_at = observed["first_observed_at"] + MINIMUM_RETENTION_SECONDS
            rows.append({**reference.as_dict(), "referenced": referenced, "eligible_at": eligible_at,
                         "eligible": eligible_at is not None and now >= eligible_at})
        result = {"artifacts": rows, "can_purge": bool(rows) and all(row["eligible"] for row in rows),
                  "minimum_retention_days": 30}
        result["preview_hash"] = _digest(result)
        return result

    def preview_orphan_purge(self, actor: Actor, digests: list[str]) -> dict[str, Any]:
        if not digests or len(set(digests)) != len(digests):
            raise DomainError("invalid_purge_targets", 422)
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
        with self.artifacts.exclusive():
            # Potentially large hashing happens outside the DB writer lock.
            physical = {digest: self.artifacts.verify(digest) for digest in digests}
            with self.store.transaction() as con:
                self.store._actor(con, actor, admin=True)
                return self._orphan_preview(con, physical)

    def discover_orphans(self, actor: Actor, *, after: str = "", limit: int = 500) -> dict[str, Any]:
        """Bounded scan of immutable storage, never of dataset membership."""
        if (after and not re.fullmatch(r'[0-9a-f]{64}', after)) or type(limit) is not int or not 1 <= limit <= 2000:
            raise DomainError('invalid_storage_scan', 422)
        physical, scanned, cursor, more = {}, 0, after, False
        with self.artifacts.exclusive():
            with self.store.read_connection() as con:
                self.store._actor(con, actor, admin=True)
                with os.scandir(self.artifacts.sha256_root) as entries:
                    shards = sorted(entry.name for entry in entries if re.fullmatch(r'[0-9a-f]{2}', entry.name) and entry.is_dir(follow_symlinks=False))
                for shard in shards:
                    if after and shard < after[:2]:
                        continue
                    with os.scandir(self.artifacts.sha256_root / shard) as entries:
                        names = sorted(entry.name for entry in entries if re.fullmatch(r'[0-9a-f]{64}', entry.name) and entry.name.startswith(shard) and entry.name > after and entry.is_file(follow_symlinks=False))
                    for digest in names:
                        if scanned >= limit:
                            more = True
                            break
                        scanned += 1
                        cursor = digest
                        if not self._referenced(con, digest):
                            physical[digest] = self.artifacts.verify(digest)
                    if more:
                        break
            with self.store.transaction() as con:
                self.store._actor(con, actor, admin=True)
                preview = self._orphan_preview(con, physical)
        return {**preview, 'scanned': scanned, 'next_cursor': cursor if more else None}

    def purge_orphans(self, actor: Actor, digests: list[str], preview_hash: str,
                      operation_id: str) -> dict[str, Any]:
        """Delete exact unreferenced blobs, with restartable per-item outcomes."""
        if not digests or len(set(digests)) != len(digests):
            raise DomainError("invalid_purge_targets", 422)
        operation_id = _text(operation_id, "operation_id", 200)
        payload = {"digests": sorted(digests), "preview_hash": preview_hash}
        request_hash = _digest(payload)
        replay = self._existing(actor, operation_id, "purge_orphans", payload)
        if replay is not None:
            return replay
        with self.artifacts.exclusive():
            replay = self._existing(actor, operation_id, "purge_orphans", payload)
            if replay is not None:
                return replay
            with self.store.read_connection() as con:
                self.store._actor(con, actor, admin=True)
                command = con.execute("SELECT * FROM hosted_purge_commands WHERE actor_id=? AND operation_id=?", (actor.user_id, operation_id)).fetchone()
            if command:
                if command["request_hash"] != request_hash:
                    raise DomainError("operation_id_reused", 409)
                if command["state"] == "interrupted":
                    raise DomainError("purge_requires_new_preview_after_restore", 409)
            else:
                physical = {digest: self.artifacts.verify(digest) for digest in digests}
                with self.store.transaction() as con:
                    self.store._actor(con, actor, admin=True)
                    if con.execute("SELECT 1 FROM hosted_receipts WHERE actor_id=? AND operation_id=?", (actor.user_id, operation_id)).fetchone():
                        raise DomainError("operation_id_reused", 409)
                    preview = self._orphan_preview(con, physical)
                    if preview["preview_hash"] != preview_hash:
                        raise DomainError("purge_preview_changed", 409)
                    if not preview["can_purge"]:
                        raise DomainError("purge_not_eligible", 409, preview)
                    con.execute("INSERT INTO hosted_purge_commands VALUES (?,?,?,'prepared',?)", (actor.user_id, operation_id, request_hash, self.store.clock()))
                    for row in preview["artifacts"]:
                        con.execute("INSERT INTO hosted_purge_items VALUES (?,?,?,?,'pending')", (actor.user_id, operation_id, row["sha256"], row["size_bytes"]))
            for digest in sorted(digests):
                with self.store.transaction() as con:
                    self.store._actor(con, actor, admin=True)
                    item = con.execute("SELECT * FROM hosted_purge_items WHERE actor_id=? AND operation_id=? AND sha256=?", (actor.user_id, operation_id, digest)).fetchone()
                    if not item:
                        raise DomainError("purge_journal_incomplete", 409)
                    if item["state"] != "pending":
                        continue
                    if self._referenced(con, digest):
                        raise DomainError("purge_references_changed", 409, {"sha256": digest})
                    deleted = self.artifacts.delete_unreferenced(digest, is_referenced=lambda value: self._referenced(con, value))
                    con.execute("UPDATE hosted_purge_items SET state=? WHERE actor_id=? AND operation_id=? AND sha256=?",
                                ("deleted" if deleted else "absent", actor.user_id, operation_id, digest))
                    con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (digest,))
            def finish(con: sqlite3.Connection) -> dict[str, Any]:
                rows = con.execute("SELECT * FROM hosted_purge_items WHERE actor_id=? AND operation_id=? ORDER BY sha256", (actor.user_id, operation_id)).fetchall()
                if any(row["state"] == "pending" for row in rows):
                    raise DomainError("purge_journal_incomplete", 409)
                con.execute("UPDATE hosted_purge_commands SET state='complete' WHERE actor_id=? AND operation_id=?", (actor.user_id, operation_id))
                return {"status": "complete", "deleted_digests": [row["sha256"] for row in rows if row["state"] == "deleted"],
                        "already_absent_digests": [row["sha256"] for row in rows if row["state"] == "absent"],
                        "logical_bytes_removed": sum(row["size_bytes"] for row in rows if row["state"] == "deleted")}
            return self.store._command(actor, operation_id, "purge_orphans", payload, finish, admin=True)

    def _snapshot_entries(self, manifest: dict[str, Any], *, unsplit_as_train=False, media_views=None) -> Iterator[ExportEntry]:
        from .media import HostedMedia, MEDIA_VIEW_POLICY, working_media_bytes, validate_working_view
        from .class_names import validate_snapshot_classes

        validate_snapshot_classes(manifest)

        classes = sorted(manifest["classes"], key=lambda item: item["position"])
        mapping = {item["id"]: index for index, item in enumerate(classes)}
        names = [item["name"] for item in classes]
        yield ExportEntry("classes.txt", data=("\n".join(names) + ("\n" if names else "")).encode())
        yield ExportEntry("labelmap.txt", data=("\n".join(names) + ("\n" if names else "")).encode())
        yield ExportEntry("labelmap.json", data=_json({"class_version": manifest["class_version"], "classes": classes}).encode())
        yield ExportEntry("hosted-snapshot.json", data=_json(manifest).encode())
        splits: dict[str, str] = {}
        caption_rows = []
        media_map = {"policy": MEDIA_VIEW_POLICY, "coordinate_space": "encoded", "images": {}}
        media = HostedMedia(self.store, self.artifacts) if hasattr(self, "store") else None
        for image in manifest["images"]:
            image_id, split = image["id"], image["split"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id) or split not in ("train", "val", "test", "unsplit"):
                raise DomainError("invalid_snapshot_identity", 409)
            if unsplit_as_train and split == "unsplit":
                split = "train"
            width, height = image["width"], image["height"]
            if width <= 0 or height <= 0:
                raise DomainError("invalid_snapshot_dimensions", 409)
            reference = ArtifactRef(image["media_hash"], image["size_bytes"])
            frozen = (media_views or {}).get(image_id) or image.get("working_media")
            if frozen is not None:
                view = frozen
                working_ref = HostedMedia.reference(view) if media is not None else reference
                with self.artifacts.open(working_ref) as stream:
                    working = stream.read()
                validate_working_view(working, view, source_sha256=reference.sha256,
                                      source_size_bytes=reference.size_bytes)
            elif media is not None:
                view = media.resolve(reference)
                working = None
            else:
                # Native children consume pinned immutable bytes without DB
                # access. Freeze their actual working hash beside the original
                # snapshot rather than replacing its provenance identity.
                with self.artifacts.open(reference) as stream:
                    working, decoded_view = working_media_bytes(stream)
                view = decoded_view
                if (view["source_sha256"] != reference.sha256
                        or view["source_size_bytes"] != reference.size_bytes
                        or view["sha256"] != decoded_view["sha256"]
                        or view["size_bytes"] != len(working)
                        or (view["width"], view["height"]) != (decoded_view["width"], decoded_view["height"])):
                    raise DomainError("snapshot_media_mismatch", 409)
            if (view["width"], view["height"]) != (width, height):
                raise DomainError("snapshot_media_mismatch", 409)
            prefix = f"{split}/"
            image_path = f"{prefix}images/{image_id}{view['suffix']}"
            media_map["images"][image_id] = {**view, "path": image_path}
            yield (ExportEntry(image_path, artifact=HostedMedia.reference(view)) if media is not None
                   else ExportEntry(image_path, data=working))
            splits[split] = f"{split}/images"
            labels, polygons = [], []
            for obj in image["annotations"]["objects"]:
                if obj["class_id"] not in mapping:
                    raise DomainError("snapshot_class_missing", 409)
                index = mapping[obj["class_id"]]
                x, y, w, h = obj["bbox"]
                values = ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)
                labels.append(str(index) + " " + " ".join(format(value, ".12g") for value in values))
                if obj.get("polygon"):
                    points = [value for point in obj["polygon"] for value in (point[0] / width, point[1] / height)]
                    polygons.append(str(index) + " " + " ".join(format(value, ".12g") for value in points))
            yield ExportEntry(f"{prefix}labels/{image_id}.txt", data=("\n".join(labels) + ("\n" if labels else "")).encode())
            if polygons:
                yield ExportEntry(f"{prefix}segments/{image_id}.txt", data=("\n".join(polygons) + "\n").encode())
            yield ExportEntry(f"{prefix}annotations/{image_id}.json", data=_json({"version_id": image["version_id"], "annotations": image["annotations"], "negative": bool(image["negative"])}).encode())
            caption = image["annotations"].get("caption")
            if isinstance(caption, str):
                yield ExportEntry(f"{prefix}captions/{image_id}.txt", data=caption.encode())
            caption_rows.append({"image_id": image_id, "image_path": image_path, "version_id": image["version_id"],
                                 "caption": caption, "captions": image["annotations"].get("captions", []),
                                 "alternate_captions": image["annotations"].get("alternate_captions", []),
                                 "text_labels": image["annotations"].get("text_labels", [])})
        config: dict[str, Any] = {"path": ".", "nc": len(classes), "names": names}
        for split in ("train", "val", "test"):
            if split in splits:
                config[split] = splits[split]
        if "unsplit" in splits:
            # Preserve unsplit provenance; never manufacture a validation split.
            config["train"] = ([splits["train"], splits["unsplit"]]
                               if "train" in splits else splits["unsplit"])
        policies = {view["policy"] for view in media_map["images"].values()}
        media_map["policy"] = next(iter(policies)) if len(policies) == 1 else "mixed" if policies else MEDIA_VIEW_POLICY
        yield ExportEntry("hosted-media.json", data=_json(media_map).encode())
        yield ExportEntry("data.yaml", data=(_json(config) + "\n").encode())  # JSON is valid YAML.
        yield ExportEntry("captions.jsonl", data=("\n".join(_json(row) for row in caption_rows) + ("\n" if caption_rows else "")).encode())

    def materialize_snapshot(self, actor: Actor, snapshot_id: str, destination: str | Path, *, unsplit_as_train=False, media_views=None) -> dict[str, Any]:
        """Publish an isolated read-only view for a legacy canonical consumer."""
        manifest = self.store.get_snapshot(actor, snapshot_id)["manifest"]
        destination = Path(destination).absolute()
        if (not destination.parent.is_dir() or destination.parent.is_symlink() or destination.is_symlink()
                or (destination.exists() and (not destination.is_dir() or any(destination.iterdir())))):
            raise DomainError("materialization_destination_not_empty", 409)
        stage = Path(tempfile.mkdtemp(prefix=".hosted-view-", dir=destination.parent))
        published = False
        try:
            with self.artifacts.exclusive():
                for entry in self._snapshot_entries(manifest, unsplit_as_train=unsplit_as_train, media_views=media_views):
                    target = stage / entry.path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as output:
                        if entry.artifact is not None:
                            with self.artifacts.open(entry.artifact) as source:
                                shutil.copyfileobj(source, output, length=1024 * 1024)
                        else:
                            output.write(entry.data)
                        output.flush()
                        os.fchmod(output.fileno(), 0o400)
                        os.fsync(output.fileno())
                for directory, _, _ in os.walk(stage, topdown=False):
                    fsync_directory(Path(directory))
                # Short final admission: revoke/logout cannot interleave with
                # publication after this authenticated transaction is admitted.
                with self.store.transaction() as con:
                    self.store._actor(con, actor, admin=True)
                    os.rename(stage, destination)
                    published = True
                    fsync_directory(destination.parent)
            return {"root": str(destination), "snapshot_id": snapshot_id, "dataset_id": manifest["dataset_id"],
                    "dataset_revision": manifest["revision"], "image_count": len(manifest["images"]),
                    "manifest_sha256": _digest(manifest)}
        finally:
            if not published:
                shutil.rmtree(stage)

    def _existing(self, actor: Actor, operation_id: str, operation: str, payload: Any) -> dict[str, Any] | None:
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            row = con.execute("SELECT request_hash,response_json FROM hosted_receipts WHERE actor_id=? AND operation_id=?", (actor.user_id, operation_id)).fetchone()
            if row:
                if row["request_hash"] != _digest({"operation": operation, "payload": payload}):
                    raise DomainError("operation_id_reused", 409)
                return self.store.disclose_receipt_result(actor, operation_id, operation,
                                                         json.loads(row["response_json"]), connection=con)
            self.store._actor(con, actor, admin=True)
            return None

    def _copy_export(self, actor: Actor, reference: ArtifactRef, destination: Path) -> None:
        if not destination.parent.is_dir() or destination.parent.is_symlink() or destination.is_symlink():
            raise DomainError("export_destination_unsafe", 409)
        with self.artifacts.exclusive():
            if destination.exists():
                from .artifacts import hash_stream, open_regular
                with open_regular(destination) as source:
                    digest, size = hash_stream(source)
                if (digest, size) != (reference.sha256, reference.size_bytes):
                    raise DomainError("export_destination_exists", 409)
                with self.store.read_connection() as con:
                    self.store._actor(con, actor, admin=True)
                return
            fd, name = tempfile.mkstemp(prefix=".hosted-export-", dir=destination.parent)
            stage = Path(name)
            try:
                with os.fdopen(fd, "wb") as output, self.artifacts.open(reference) as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                with self.store.transaction() as con:
                    self.store._actor(con, actor, admin=True)
                    os.link(stage, destination, follow_symlinks=False)
                    fsync_directory(destination.parent)
            finally:
                stage.unlink(missing_ok=True)
                fsync_directory(destination.parent)

    def export_canonical(self, actor: Actor, dataset_id: str, destination: str | Path,
                         operation_id: str, snapshot_id: str | None = None) -> dict[str, Any]:
        """Receipt identifies an immutable export blob; delivery can be retried."""
        destination = Path(destination).absolute()
        payload = {"dataset_id": dataset_id, "destination": str(destination), "snapshot_id": snapshot_id}
        result = self._existing(actor, operation_id, "export_canonical", payload)
        if result is None:
            if snapshot_id is None:
                capture_op = "export-snapshot-" + _digest({"actor": actor.user_id, "operation_id": operation_id})
                frozen = self.store.create_snapshot(actor, dataset_id, capture_op, "Canonical export")
            else:
                frozen = self.store.get_snapshot(actor, snapshot_id)
            manifest = frozen["manifest"]
            if manifest["dataset_id"] != dataset_id:
                raise DomainError("snapshot_dataset_mismatch", 409)
            with tempfile.TemporaryDirectory(prefix="export-", dir=self.artifacts.staging) as directory:
                archive = Path(directory) / "dataset.zip"
                create_canonical_zip(archive, self._snapshot_entries(manifest), artifacts=self.artifacts)
                with self.artifacts.exclusive():
                    reference = self.artifacts.put_file(archive)
                    def run(con: sqlite3.Connection) -> dict[str, Any]:
                        export_id = _id("export")
                        con.execute("INSERT INTO hosted_exports VALUES (?,?,?,?,?,?)", (export_id, frozen["snapshot_id"], reference.sha256, reference.size_bytes, actor.user_id, self.store.clock()))
                        con.execute("INSERT INTO hosted_artifact_references VALUES ('export',?,?,?,?)", (export_id, reference.sha256, reference.size_bytes, self.store.clock()))
                        con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (reference.sha256,))
                        return {"export_id": export_id, "snapshot_id": frozen["snapshot_id"], "dataset_id": dataset_id,
                                "dataset_revision": manifest["revision"], "image_count": len(manifest["images"]), **reference.as_dict()}
                    result = self.store._command(actor, operation_id, "export_canonical", payload, run, admin=True)
        self._copy_export(actor, ArtifactRef(result["sha256"], result["size_bytes"]), destination)
        return {**result, "path": str(destination)}

    def export_reference(self, actor: Actor, export_id: str) -> ArtifactRef:
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            row = con.execute("SELECT sha256,size_bytes FROM hosted_exports WHERE id=?", (export_id,)).fetchone()
            if not row:
                raise DomainError("export_not_found", 404)
            return ArtifactRef(row["sha256"], row["size_bytes"])

    @staticmethod
    def restore_hook(connection: sqlite3.Connection, manifest: dict[str, Any]) -> None:
        """Schema-aware closure and authority reset before restored-root publication."""
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"hosted_schema", "hosted_users", "hosted_meta", "hosted_images", "hosted_annotation_versions",
                    "hosted_datasets", "hosted_class_versions", "hosted_receipts", "hosted_media"}
        if not required.issubset(tables):
            raise DomainError("recovery_schema_missing", 409, {"tables": sorted(required - tables)})
        version = connection.execute("SELECT version FROM hosted_schema WHERE component='store'").fetchone()
        if not version or version[0] != SCHEMA_VERSION:
            raise DomainError("recovery_schema_unsupported", 409)
        available = {row["sha256"]: row["size_bytes"] for row in manifest["artifacts"]}
        for reference in HostedOperations.artifact_references(connection):
            if available.get(reference["sha256"]) != reference["size_bytes"]:
                raise DomainError("recovery_artifact_missing", 409, {"sha256": reference["sha256"]})
        if "hosted_media_views" in tables:
            for source_hash, policy, encoded in connection.execute("SELECT source_sha256,policy,view_json FROM hosted_media_views"):
                try:
                    view = json.loads(encoded)
                    if (view["source_sha256"] != source_hash or view["policy"] != policy
                            or view["coordinate_space"] != "encoded" or view["width"] <= 0 or view["height"] <= 0):
                        raise ValueError("media view identity changed")
                    for digest, size in ((source_hash, view["source_size_bytes"]), (view["sha256"], view["size_bytes"])):
                        registered = connection.execute("SELECT 1 FROM hosted_artifact_references WHERE kind='media_view' AND owner_id=? AND sha256=? AND size_bytes=?", (source_hash + ":" + policy, digest, size)).fetchone()
                        if registered is None or available.get(digest) != size:
                            raise ValueError("working media reference missing")
                except (KeyError, TypeError, ValueError) as error:
                    raise DomainError("recovery_media_view_missing", 409) from error
        if 'hosted_ingestion_candidates' in tables:
            for identifier, dataset_id, encoded in connection.execute('SELECT id,dataset_id,manifest_json FROM hosted_ingestion_candidates'):
                try:
                    captured = json.loads(encoded)
                    if (captured['candidate_set_id'] != identifier or captured['dataset_id'] != dataset_id
                            or captured['schema'] != 'hosted-incoming-candidates-v1'):
                        raise ValueError('candidate identity changed')
                    for image in captured['images']:
                        row = connection.execute('SELECT i.dataset_id,i.media_hash,m.size_bytes,v.image_id,v.annotations_json FROM hosted_images i JOIN hosted_media m ON m.sha256=i.media_hash JOIN hosted_annotation_versions v ON v.id=? WHERE i.id=?', (image['version_id'], image['id'])).fetchone()
                        if (row is None or tuple(row[:4]) != (dataset_id, image['media_hash'], image['size_bytes'], image['id'])
                                or json.loads(row[4]) != image['annotations']
                                or available.get(image['media_hash']) != image['size_bytes']):
                            raise ValueError('candidate authority or media missing')
                except (KeyError, TypeError, ValueError) as error:
                    raise DomainError('recovery_ingestion_candidates_missing', 409) from error
        # Secondary indices must not name blobs outside the captured recovery
        # closure, even if their main job row and other artifacts are intact.
        if 'hosted_provider_results' in tables:
            for call_id, encoded in connection.execute('SELECT call_id,result_json FROM hosted_provider_results'):
                try:
                    result = json.loads(encoded)
                    inputs = result.get('input_artifacts', [])
                    if (not isinstance(inputs, list) or ('stage_id' in result and not inputs)
                            or result.get('call_id', call_id) != call_id):
                        raise ValueError('paid stage input manifest invalid')
                    for item in inputs:
                        digest, size = item['sha256'], item['size_bytes']
                        if not isinstance(digest, str) or type(size) is not int or size < 0:
                            raise ValueError('paid stage artifact identity invalid')
                        ref = connection.execute("SELECT 1 FROM hosted_artifact_references WHERE kind='paid_stage' AND owner_id=? AND sha256=? AND size_bytes=?", (call_id, digest, size)).fetchone()
                        if ref is None or available.get(digest) != size:
                            raise ValueError('paid stage input missing')
                except (KeyError, TypeError, ValueError, AttributeError) as error:
                    raise DomainError('recovery_paid_stage_artifact_missing', 409) from error
        if "hosted_analysis_artifacts" in tables:
            for owner, digest, size in connection.execute("SELECT job_id,sha256,size_bytes FROM hosted_analysis_artifacts"):
                ref = connection.execute("SELECT 1 FROM hosted_artifact_references WHERE kind='analysis_result' AND owner_id=? AND sha256=? AND size_bytes=?", (owner, digest, size)).fetchone()
                if ref is None or available.get(digest) != size:
                    raise DomainError("recovery_analysis_artifact_missing", 409)
        if 'hosted_paid_batches' in tables:
            if 'hosted_paid_batch_rows' not in tables or connection.execute(
                    'SELECT 1 FROM hosted_paid_batch_rows r LEFT JOIN hosted_paid_batches b ON b.id=r.job_id WHERE b.id IS NULL LIMIT 1').fetchone():
                raise DomainError('recovery_paid_batch_missing', 409)
            for owner, actor_id, encoded, request_encoded, result_encoded in connection.execute(
                    'SELECT id,actor_id,manifest_json,request_json,result_json FROM hosted_paid_batches'):
                try:
                    captured, request = json.loads(encoded), json.loads(request_encoded)
                    images = {image['id']: image for image in captured['images']}
                    if (len(images) != len(captured['images']) or captured['dataset_id'] != request['dataset_id']):
                        raise ValueError('invalid captured population')
                    classes = connection.execute('SELECT classes_json FROM hosted_class_versions WHERE dataset_id=? AND version=?',
                                                 (captured['dataset_id'], captured['class_version'])).fetchone()
                    if classes is None or json.loads(classes[0]) != captured['classes']:
                        raise ValueError('missing captured class version')
                    refs = []
                    result = json.loads(result_encoded) if result_encoded else {}
                    refs.extend(result.get('raw_evidence', []))
                    if result.get('evidence'):
                        refs.append(result['evidence'])
                    rows = connection.execute('SELECT image_id,call_id,output_json FROM hosted_paid_batch_rows WHERE job_id=?', (owner,)).fetchall()
                    if {row[0] for row in rows} != set(images):
                        raise ValueError('batch rows differ from captured population')
                    for image_id, call_id, output_encoded in rows:
                        image = images[image_id]
                        record = connection.execute('SELECT i.dataset_id,i.media_hash,m.size_bytes,v.image_id,v.annotations_json FROM hosted_images i JOIN hosted_media m ON m.sha256=i.media_hash JOIN hosted_annotation_versions v ON v.id=? WHERE i.id=?', (image['version_id'], image_id)).fetchone()
                        if (record is None or tuple(record[:4]) != (captured['dataset_id'], image['media_hash'], image['size_bytes'], image_id)
                                or json.loads(record[4]) != image['annotations'] or available.get(image['media_hash']) != image['size_bytes']):
                            raise ValueError('batch source version or media missing')
                        if call_id:
                            call = connection.execute('SELECT actor_id,image_id,annotation_version,model,operation_id FROM hosted_provider_calls WHERE id=?', (call_id,)).fetchone()
                            operation = 'batch-row-' + hashlib.sha256(_json([owner, image_id]).encode()).hexdigest()
                            if call is None or tuple(call) != (actor_id, image_id, image['version_id'], request['provider_options']['model'], operation):
                                raise ValueError('batch billing authority mismatch')
                        output = json.loads(output_encoded) if output_encoded else {}
                        if output and output.get('call_id') != call_id:
                            raise ValueError('batch output call mismatch')
                        refs.extend(output.get('evidence', []))
                    for ref in refs:
                        digest, size = ref['sha256'], ref['size_bytes']
                        registered = connection.execute("SELECT 1 FROM hosted_artifact_references WHERE kind='paid_batch' AND owner_id=? AND sha256=? AND size_bytes=?", (owner, digest, size)).fetchone()
                        if type(size) is not int or not isinstance(digest, str) or registered is None or available.get(digest) != size:
                            raise ValueError('batch evidence missing')
                except (KeyError, TypeError, ValueError, AttributeError, sqlite3.Error) as error:
                    raise DomainError('recovery_paid_batch_missing', 409, {'job_id': owner}) from error
        if 'hosted_caption_batch_attempts' in tables:
            for owner, digest, size in connection.execute('SELECT batch_id,sha256,size_bytes FROM hosted_caption_batch_attempts'):
                ref = connection.execute("SELECT 1 FROM hosted_artifact_references WHERE kind='caption_batch' AND owner_id=? AND sha256=? AND size_bytes=?", (owner, digest, size)).fetchone()
                if ref is None or available.get(digest) != size:
                    raise DomainError('recovery_caption_artifact_missing', 409)
        if "hosted_pruning_checkpoints" in tables:
            for owner, encoded in connection.execute("SELECT job_id,manifest_json FROM hosted_pruning_checkpoints"):
                try:
                    rows = json.loads(encoded)["files"]
                    for name, item in rows.items():
                        if (not isinstance(name, str) or "\\" in name or PurePosixPath(name).is_absolute()
                                or any(part in {"", ".", ".."} for part in name.split("/"))):
                            raise ValueError("invalid checkpoint path")
                        ref = connection.execute("SELECT 1 FROM hosted_artifact_references WHERE kind='pruning_review' AND owner_id=? AND sha256=? AND size_bytes=?", (owner, item["sha256"], item["size_bytes"])).fetchone()
                        if ref is None or available.get(item["sha256"]) != item["size_bytes"]:
                            raise ValueError("unretained checkpoint blob")
                except (KeyError, TypeError, ValueError, AttributeError) as error:
                    raise DomainError("recovery_pruning_artifact_missing", 409) from error
        if 'hosted_salad_heads' in tables:
            heads = {}
            for head_id, digest, size in connection.execute('SELECT head_id,sha256,size_bytes FROM hosted_salad_heads'):
                ref = connection.execute("SELECT 1 FROM hosted_artifact_references WHERE kind='salad_head' AND owner_id=? AND sha256=? AND size_bytes=?", (head_id, digest, size)).fetchone()
                if ref is None or available.get(digest) != size:
                    raise DomainError('recovery_salad_head_missing', 409)
                heads[head_id] = digest
        else:
            heads = {}
        for settings_table in ('hosted_dataset_settings', 'hosted_dataset_setting_versions'):
            if settings_table in tables:
                for encoded, in connection.execute(f'SELECT settings_json FROM {settings_table}'):
                    try:
                        saved = json.loads(encoded).get('salad_profiles', {})
                        profiles = saved.get('profiles', {})
                        if any(head not in heads for head in saved.get('head_ids', [])):
                            raise ValueError('unknown attached head')
                        for head_id, profile in profiles.items():
                            if heads.get(head_id) != profile.get('head_sha256'):
                                raise ValueError('profile head identity changed')
                    except (TypeError, ValueError, AttributeError) as error:
                        raise DomainError('recovery_salad_profile_invalid', 409) from error
        if "hosted_tool_models" in tables:
            for model_id, definition_json in connection.execute("SELECT model_id,definition_json FROM hosted_tool_models"):
                try:
                    definition = json.loads(definition_json)
                    if not isinstance(definition, dict):
                        raise ValueError("model definition must be an object")
                    if not definition.get("checkpoint_sha256"):
                        raise ValueError("model definition has no checkpoint")
                    if definition.get("kind") == "classifier" and not definition.get("metadata_sha256"):
                        raise ValueError("classifier definition has no recipe metadata")
                    if definition.get("kind") in {"qwen", "edr", "sam3_recipe"} and not definition.get("files"):
                        raise ValueError("directory model has no file manifest")
                    if definition.get("kind") == "edr":
                        if not isinstance(definition.get("components"), dict) or not definition["components"]:
                            raise ValueError("ensemble has no components")
                        for role, component_id in definition["components"].items():
                            component = connection.execute("SELECT definition_json FROM hosted_tool_models WHERE model_id=?", (component_id,)).fetchone()
                            expected_kind = "qwen" if role in {"qwen", "qwen_refinement", "qwen_fallback"} else role
                            if (role not in {"yolo", "rfdetr", "sam3", "classifier", "qwen", "qwen_refinement", "qwen_fallback"} or component is None
                                    or json.loads(component[0]).get("kind") != expected_kind):
                                raise ValueError("ensemble component missing or incompatible")
                        if not definition.get("recipe_file"):
                            raise ValueError("ensemble recipe missing")
                        for field in ("recipe_file", "ensemble_model_file", "ensemble_meta_file", "feature_contract_file"):
                            if definition.get(field) and definition[field] not in definition["files"]:
                                raise ValueError("ensemble file missing from manifest")
                    if definition.get('kind') == 'sam3_recipe':
                        components = definition.get('components')
                        if not isinstance(components, dict) or 'sam3' not in components or set(components) - {'sam3', 'extra_classifier'}:
                            raise ValueError('SAM3 recipe components missing or incompatible')
                        for role, component_id in components.items():
                            component = connection.execute('SELECT definition_json FROM hosted_tool_models WHERE model_id=?', (component_id,)).fetchone()
                            expected_kind = 'classifier' if role == 'extra_classifier' else 'sam3'
                            if component is None or json.loads(component[0]).get('kind') != expected_kind:
                                raise ValueError('SAM3 recipe component missing or incompatible')
                        if definition.get('recipe_file') not in definition['files']:
                            raise ValueError('SAM3 recipe file missing from manifest')
                    hashes = [definition[key] for key in ("checkpoint_sha256", "bpe_sha256", "metadata_sha256") if definition.get(key)]
                    for field in ("files", "adapter_files"):
                        contents = definition.get(field, {})
                        if not isinstance(contents, dict):
                            raise ValueError("invalid model directory manifest")
                        for name, file_digest in contents.items():
                            if (not isinstance(name, str) or not name or "\\" in name or "\0" in name
                                    or PurePosixPath(name).is_absolute()
                                    or any(part in {"", ".", ".."} for part in name.split("/"))):
                                raise ValueError("invalid model directory path")
                            hashes.append(file_digest)
                    for digest in hashes:
                        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                            raise ValueError("invalid model artifact digest")
                except (TypeError, ValueError) as error:
                    raise DomainError("recovery_model_definition_invalid", 409, {"model_id": model_id}) from error
                for digest in hashes:
                    registered = (connection.execute("SELECT size_bytes FROM hosted_artifact_references WHERE kind='tool_model' AND owner_id=? AND sha256=?", (model_id, digest)).fetchone()
                                  if "hosted_artifact_references" in tables else None)
                    if registered is None or available.get(digest) != registered[0]:
                        raise DomainError("recovery_model_artifact_missing", 409, {"model_id": model_id, "sha256": digest})
        if not connection.execute("SELECT 1 FROM hosted_users WHERE role='admin' AND disabled=0 LIMIT 1").fetchone():
            raise DomainError("recovery_has_no_active_administrator", 409)
        AuthService.invalidate_restored_sessions(connection)
        from .job_authority import invalidate_restored_jobs
        invalidate_restored_jobs(connection)
        from .providers import ProviderAccounts
        ProviderAccounts.after_restore(connection)
        if "hosted_analysis_jobs" in tables:
            connection.execute("UPDATE hosted_analysis_jobs SET state='interrupted' WHERE state='captured'")
        for table in ('hosted_caption_jobs', 'hosted_caption_batches'):
            if table in tables:
                connection.execute(f"UPDATE {table} SET state='interrupted',error='control_restarted' WHERE state IN ('queued','running','paused','retrying','waiting_for_editor')")
        if 'hosted_paid_batches' in tables:
            connection.execute("UPDATE hosted_paid_batches SET state='submission_unknown',error='restored_during_submission' WHERE state='creating'")
            connection.execute("UPDATE hosted_paid_batches SET state='interrupted',error='server_restored' WHERE state IN ('queued','uploading','paused') AND remote_id IS NULL")
        if 'hosted_dqe_batches' in tables:
            connection.execute("UPDATE hosted_dqe_batches SET state='interrupted' WHERE state IN ('queued','running')")
        connection.execute("UPDATE hosted_images SET generation=generation+1")
        if "hosted_purge_commands" in tables:
            connection.execute("UPDATE hosted_purge_commands SET state='interrupted' WHERE state='prepared'")
        # Inference workers are recreated in the new root; running predictions
        # cannot be resurrected as successful annotation writes. Dataset/member
        # and submitted annotation versions are deliberately unchanged.


def recovery_artifact_references(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Standard hosted-schema callback for the trusted recovery CLI."""
    return HostedOperations.artifact_references(connection)


def prepare_restored_database(connection: sqlite3.Connection, manifest: dict[str, Any]) -> None:
    """Validate hosted recovery closure and revoke the previous installation."""
    HostedOperations.restore_hook(connection, manifest)
