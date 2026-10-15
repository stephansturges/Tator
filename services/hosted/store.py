"""Hosted annotation authority.

Only approval admits Incoming into canonical membership. Annotation versions,
submissions, class mappings and command receipts are immutable. Ownership and
the current-version pointers change together in short SQLite transactions.

AuthService owns hosted_users. Every admission rechecks that table *inside*
the transaction, including delayed operations carrying an earlier Actor.
Blob publication and garbage collection are external: callers must hold the
artifact-store lock before a transaction adding/removing media references.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

from .errors import DomainError
from .types import Actor, MutationContext


SCHEMA_VERSION = 1
STATES = ("unassigned", "assigned", "returned", "submitted", "confirmed")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise DomainError("invalid_json", 422) from exc


def _text(value: Any, field: str, maximum: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise DomainError("invalid_field", 422, {"field": field})
    return value.strip()


class HostedStore:
    def __init__(self, db_path: str | Path, *, clock: Callable[[], float] = time.time,
                 lease_seconds: float = 120) -> None:
        self.path = Path(db_path)
        self.clock = clock
        self.lease_seconds = float(lease_seconds)
        if not math.isfinite(self.lease_seconds) or self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be finite and positive")
        if self.path.is_symlink():
            raise ValueError("Hosted database must not be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        target = self.path.resolve().as_uri() + "?mode=ro" if readonly else str(self.path)
        con = sqlite3.connect(target, uri=readonly, timeout=15, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=15000")
        if readonly:
            con.execute("PRAGMA query_only=ON")
        else:
            con.execute("PRAGMA synchronous=FULL")
        return con

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        """One coherent read snapshot; safe for auth and backup consumers."""
        con = self._connect(readonly=True)
        try:
            con.execute("BEGIN")
            yield con
        finally:
            con.rollback()
            con.close()

    def _initialize(self) -> None:
        with closing(self._connect()) as con:
            if con.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_schema'").fetchone():
                previous = con.execute("SELECT version FROM hosted_schema WHERE component='store'").fetchone()
                if previous and previous[0] != SCHEMA_VERSION:
                    raise RuntimeError("Unsupported hosted store schema")
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS hosted_schema (
                    component TEXT PRIMARY KEY, version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_datasets (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    class_version INTEGER NOT NULL DEFAULT 0,
                    created_by TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_classes (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
                    name TEXT NOT NULL, name_key TEXT NOT NULL, position INTEGER NOT NULL,
                    retired INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(dataset_id,name_key), UNIQUE(dataset_id,position));
                CREATE TABLE IF NOT EXISTS hosted_class_versions (
                    dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
                    version INTEGER NOT NULL, classes_json TEXT NOT NULL,
                    actor_id TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(dataset_id,version));
                CREATE TABLE IF NOT EXISTS hosted_dataset_settings (
                    dataset_id TEXT PRIMARY KEY REFERENCES hosted_datasets(id),
                    version INTEGER NOT NULL, settings_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_dataset_setting_versions (
                    dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
                    version INTEGER NOT NULL, settings_json TEXT NOT NULL,
                    actor_id TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(dataset_id,version));
                CREATE TABLE IF NOT EXISTS hosted_media (
                    sha256 TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL,
                    width INTEGER NOT NULL, height INTEGER NOT NULL,
                    media_path TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_assignments (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
                    annotator_id TEXT NOT NULL, created_by TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_images (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
                    media_hash TEXT NOT NULL REFERENCES hosted_media(sha256), filename TEXT NOT NULL,
                    split TEXT NOT NULL DEFAULT 'train',
                    state TEXT NOT NULL CHECK(state IN ('unassigned','assigned','returned','submitted','confirmed')),
                    assignee_id TEXT, assignment_id TEXT REFERENCES hosted_assignments(id),
                    current_version TEXT NOT NULL REFERENCES hosted_annotation_versions(id) DEFERRABLE INITIALLY DEFERRED,
                    confirmed_version TEXT REFERENCES hosted_annotation_versions(id),
                    active_submission TEXT REFERENCES hosted_submissions(id),
                    generation INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    UNIQUE(dataset_id,media_hash));
                CREATE TABLE IF NOT EXISTS hosted_annotation_versions (
                    id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES hosted_images(id) DEFERRABLE INITIALLY DEFERRED,
                    parent_version TEXT REFERENCES hosted_annotation_versions(id),
                    annotations_json TEXT NOT NULL, content_hash TEXT NOT NULL,
                    complete INTEGER NOT NULL, negative INTEGER NOT NULL,
                    class_version INTEGER NOT NULL, actor_id TEXT NOT NULL,
                    kind TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_assignment_members (
                    assignment_id TEXT NOT NULL REFERENCES hosted_assignments(id),
                    image_id TEXT NOT NULL REFERENCES hosted_images(id),
                    initial_version TEXT NOT NULL REFERENCES hosted_annotation_versions(id),
                    state TEXT NOT NULL, PRIMARY KEY(assignment_id,image_id));
                CREATE TABLE IF NOT EXISTS hosted_editor_leases (
                    image_id TEXT PRIMARY KEY REFERENCES hosted_images(id),
                    actor_id TEXT NOT NULL, editor_id TEXT NOT NULL,
                    generation INTEGER NOT NULL, expires_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_submissions (
                    id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES hosted_images(id),
                    version_id TEXT NOT NULL REFERENCES hosted_annotation_versions(id),
                    assignment_id TEXT NOT NULL REFERENCES hosted_assignments(id),
                    actor_id TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_reviews (
                    id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES hosted_images(id),
                    submission_id TEXT REFERENCES hosted_submissions(id),
                    version_id TEXT NOT NULL REFERENCES hosted_annotation_versions(id),
                    action TEXT NOT NULL, feedback TEXT NOT NULL, actor_id TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_receipts (
                    actor_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                    operation TEXT NOT NULL, request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(actor_id,operation_id));
                CREATE TABLE IF NOT EXISTS hosted_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, actor_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
                    image_id TEXT REFERENCES hosted_images(id), kind TEXT NOT NULL,
                    data_json TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_import_conflicts (
                    id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES hosted_images(id),
                    filename TEXT NOT NULL, annotations_json TEXT NOT NULL,
                    class_version INTEGER NOT NULL, actor_id TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_import_conflict_decisions (
                    conflict_id TEXT PRIMARY KEY REFERENCES hosted_import_conflicts(id),
                    image_id TEXT NOT NULL REFERENCES hosted_images(id), actor_id TEXT NOT NULL,
                    choice TEXT NOT NULL CHECK(choice IN ('keep_current','use_imported_objects')),
                    previous_version TEXT NOT NULL REFERENCES hosted_annotation_versions(id),
                    result_version TEXT NOT NULL REFERENCES hosted_annotation_versions(id), created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_class_requests (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
                    actor_id TEXT NOT NULL, name TEXT NOT NULL, reason TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending', class_id TEXT REFERENCES hosted_classes(id),
                    reviewed_by TEXT, feedback TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL, reviewed_at REAL);
                CREATE TABLE IF NOT EXISTS hosted_snapshots (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
                    revision INTEGER NOT NULL, name TEXT NOT NULL, manifest_json TEXT NOT NULL,
                    actor_id TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS hosted_archives (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('image','dataset')),
                    target_id TEXT NOT NULL, dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
                    state TEXT NOT NULL CHECK(state IN ('archived','restored')),
                    archived_at REAL NOT NULL, actor_id TEXT NOT NULL, reason TEXT NOT NULL,
                    captured_json TEXT NOT NULL, restored_at REAL, restored_by TEXT);
                CREATE TABLE IF NOT EXISTS hosted_archived_images (
                    image_id TEXT PRIMARY KEY REFERENCES hosted_images(id),
                    archive_id TEXT NOT NULL REFERENCES hosted_archives(id),
                    version_id TEXT NOT NULL REFERENCES hosted_annotation_versions(id));
                CREATE TABLE IF NOT EXISTS hosted_archived_datasets (
                    dataset_id TEXT PRIMARY KEY REFERENCES hosted_datasets(id),
                    archive_id TEXT NOT NULL REFERENCES hosted_archives(id));
                CREATE INDEX IF NOT EXISTS hosted_images_work ON hosted_images(assignee_id,state,dataset_id,id);
                CREATE INDEX IF NOT EXISTS hosted_images_dataset ON hosted_images(dataset_id,state,id);
                CREATE INDEX IF NOT EXISTS hosted_versions_image ON hosted_annotation_versions(image_id,created_at,id);
                CREATE INDEX IF NOT EXISTS hosted_events_image ON hosted_events(image_id,id);
                COMMIT;
            """)
            row = con.execute("SELECT version FROM hosted_schema WHERE component='store'").fetchone()
            if row and row[0] != SCHEMA_VERSION:
                raise RuntimeError("Unsupported hosted store schema")
            con.execute("INSERT OR IGNORE INTO hosted_schema VALUES ('store',?)", (SCHEMA_VERSION,))
            for table in ("hosted_annotation_versions", "hosted_class_versions", "hosted_dataset_setting_versions", "hosted_submissions",
                          "hosted_reviews", "hosted_receipts", "hosted_events", "hosted_snapshots",
                          "hosted_import_conflicts", "hosted_import_conflict_decisions"):
                for verb in ("UPDATE", "DELETE"):
                    con.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_immutable_{verb.lower()} "
                                f"BEFORE {verb} ON {table} BEGIN SELECT RAISE(ABORT,'immutable hosted history'); END")

    def _actor(self, con: sqlite3.Connection, actor: Actor, *, admin: bool = False) -> sqlite3.Row:
        if actor.session_id or actor.instance_id:
            session = con.execute("SELECT user_id,expires_at,instance_id FROM hosted_sessions WHERE token_hash=?",
                                  (actor.session_id,)).fetchone()
            instance = con.execute("SELECT value FROM hosted_meta WHERE key='instance_id'").fetchone()
            if (not actor.session_id or not actor.instance_id or not session or not instance
                    or session["user_id"] != actor.user_id or session["expires_at"] <= self.clock()
                    or session["instance_id"] != actor.instance_id or instance[0] != actor.instance_id):
                raise DomainError("login_required", 401)
        row = con.execute("SELECT id,role,disabled FROM hosted_users WHERE id=?", (actor.user_id,)).fetchone()
        if not row or row["disabled"] or row["role"] != actor.role:
            raise DomainError("actor_inactive", 401)
        if admin and row["role"] != "admin":
            raise DomainError("admin_required", 403)
        return row

    @staticmethod
    def _annotator(con: sqlite3.Connection, user_id: str) -> None:
        row = con.execute("SELECT role,disabled FROM hosted_users WHERE id=?", (user_id,)).fetchone()
        if not row or row["disabled"] or row["role"] != "annotator":
            raise DomainError("annotator_unavailable", 409)

    def _command(self, actor: Actor, operation_id: str, operation: str, payload: Any,
                 run: Callable[[sqlite3.Connection], dict[str, Any]], *, admin: bool = False) -> dict[str, Any]:
        operation_id = _text(operation_id, "operation_id", 200)
        digest = hashlib.sha256(_json({"operation": operation, "payload": payload}).encode()).hexdigest()
        with self.transaction() as con:
            self._actor(con, actor, admin=admin)
            existing = con.execute("SELECT request_hash,response_json FROM hosted_receipts "
                                   "WHERE actor_id=? AND operation_id=?", (actor.user_id, operation_id)).fetchone()
            if existing:
                if existing["request_hash"] != digest:
                    raise DomainError("operation_id_reused", 409)
                return json.loads(existing["response_json"])
            # Long filesystem purges reserve their operation before unlinking.
            # A second semantic command must not steal that receipt identity
            # while the per-item journal is awaiting recovery.
            if con.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_purge_commands'").fetchone():
                reserved = con.execute("SELECT request_hash FROM hosted_purge_commands WHERE actor_id=? AND operation_id=?", (actor.user_id, operation_id)).fetchone()
                if reserved and (operation != "purge_orphans" or reserved[0] != hashlib.sha256(_json(payload).encode()).hexdigest()):
                    raise DomainError("operation_id_reused", 409)
            response = run(con)
            con.execute("INSERT INTO hosted_receipts VALUES (?,?,?,?,?,?)",
                        (actor.user_id, operation_id, operation, digest, _json(response), self.clock()))
            return response

    def receipt(self, actor: Actor, operation_id: str) -> dict[str, Any] | None:
        with self.read_connection() as con:
            self._actor(con, actor)
            row = con.execute("SELECT operation,response_json,created_at FROM hosted_receipts "
                              "WHERE actor_id=? AND operation_id=?", (actor.user_id, operation_id)).fetchone()
            return None if not row else {"operation_id": operation_id, "operation": row["operation"],
                                         "result": json.loads(row["response_json"]), "committed_at": row["created_at"]}

    @staticmethod
    def _dataset(con: sqlite3.Connection, dataset_id: str, *, include_archived: bool = False) -> sqlite3.Row:
        row = con.execute("SELECT * FROM hosted_datasets WHERE id=?", (dataset_id,)).fetchone()
        if not row:
            raise DomainError("dataset_not_found", 404)
        if not include_archived and con.execute("SELECT 1 FROM hosted_archived_datasets WHERE dataset_id=?", (dataset_id,)).fetchone():
            raise DomainError("dataset_not_found", 404)
        return row

    @staticmethod
    def _image(con: sqlite3.Connection, image_id: str, *, include_archived: bool = False) -> sqlite3.Row:
        row = con.execute("SELECT * FROM hosted_images WHERE id=?", (image_id,)).fetchone()
        if not row:
            raise DomainError("image_not_found", 404)
        if not include_archived and con.execute(
                "SELECT 1 FROM hosted_archived_images WHERE image_id=? UNION ALL "
                "SELECT 1 FROM hosted_archived_datasets WHERE dataset_id=? LIMIT 1", (image_id, row["dataset_id"])).fetchone():
            raise DomainError("image_not_found", 404)
        return row

    @staticmethod
    def _can_view(actor: Actor, image: sqlite3.Row) -> None:
        if actor.role != "admin" and not (image["assignee_id"] == actor.user_id
                                          and image["state"] in ("assigned", "returned")):
            raise DomainError("image_not_found", 404)

    @staticmethod
    def _can_edit(actor: Actor, image: sqlite3.Row) -> None:
        if actor.role == "admin":
            if image["state"] in ("assigned", "returned"):
                raise DomainError("image_owned_by_annotator", 409)
        elif image["assignee_id"] != actor.user_id or image["state"] not in ("assigned", "returned"):
            raise DomainError("image_not_editable", 403)

    @staticmethod
    def _expected(image: sqlite3.Row, version: str, generation: int) -> None:
        if not isinstance(version, str) or not version:
            raise DomainError("expected_version_required", 422)
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise DomainError("generation_required", 422)
        if image["current_version"] != version:
            raise DomainError("annotation_version_changed", 409, {"current_version": image["current_version"]})
        if image["generation"] != generation:
            raise DomainError("ownership_changed", 409, {"generation": image["generation"]})

    def _guard(self, con: sqlite3.Connection, actor: Actor, ctx: MutationContext, *, renew: bool = True) -> sqlite3.Row:
        image = self._image(con, ctx.image_id)
        self._can_view(actor, image)
        self._can_edit(actor, image)
        self._expected(image, ctx.expected_version, ctx.generation)
        lease = con.execute("SELECT * FROM hosted_editor_leases WHERE image_id=?", (ctx.image_id,)).fetchone()
        if not lease or lease["expires_at"] <= self.clock():
            raise DomainError("editor_lease_expired", 409)
        if (lease["actor_id"], lease["editor_id"], lease["generation"]) != (
                actor.user_id, ctx.editor_id, ctx.generation):
            raise DomainError("editor_lease_conflict", 423)
        if renew:
            con.execute("UPDATE hosted_editor_leases SET expires_at=? WHERE image_id=?",
                        (self.clock() + self.lease_seconds, ctx.image_id))
        return image

    def authorize_context(self, actor: Actor, ctx: MutationContext, *, write: bool = True) -> dict[str, Any]:
        """Read-only admission check for queued inference and delayed results.

        This grants no future write right. The eventual save must repeat the
        same checks atomically with its mutation.
        """
        with self.read_connection() as con:
            self._actor(con, actor)
            if write:
                self._guard(con, actor, ctx, renew=False)
            else:
                image = self._image(con, ctx.image_id)
                self._can_view(actor, image)
                self._expected(image, ctx.expected_version, ctx.generation)
            return self._image_result(con, ctx.image_id, content=True)

    def _event(self, con: sqlite3.Connection, actor: Actor, dataset_id: str, image_id: str | None,
               kind: str, data: Any) -> None:
        con.execute("INSERT INTO hosted_events(actor_id,dataset_id,image_id,kind,data_json,created_at) VALUES (?,?,?,?,?,?)",
                    (actor.user_id, dataset_id, image_id, kind, _json(data), self.clock()))

    @staticmethod
    def _classes(con: sqlite3.Connection, dataset_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in con.execute(
            "SELECT id,name,position,retired FROM hosted_classes WHERE dataset_id=? ORDER BY position", (dataset_id,))]

    def _class_revision(self, con: sqlite3.Connection, actor: Actor, dataset_id: str) -> int:
        con.execute("UPDATE hosted_datasets SET class_version=class_version+1,revision=revision+1 WHERE id=?", (dataset_id,))
        version = self._dataset(con, dataset_id)["class_version"]
        con.execute("INSERT INTO hosted_class_versions VALUES (?,?,?,?,?)",
                    (dataset_id, version, _json(self._classes(con, dataset_id)), actor.user_id, self.clock()))
        return version

    def _add_class(self, con: sqlite3.Connection, actor: Actor, dataset_id: str, name: str) -> dict[str, Any]:
        name = _text(name, "class_name", 200)
        if "\n" in name or "\r" in name:
            raise DomainError("invalid_class_name", 422)
        if con.execute("SELECT 1 FROM hosted_classes WHERE dataset_id=? AND name_key=?", (dataset_id, name.casefold())).fetchone():
            raise DomainError("class_name_exists", 409)
        position = con.execute("SELECT COALESCE(MAX(position),-1)+1 FROM hosted_classes WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
        class_id = _id("class")
        con.execute("INSERT INTO hosted_classes(id,dataset_id,name,name_key,position) VALUES (?,?,?,?,?)",
                    (class_id, dataset_id, name, name.casefold(), position))
        return {"id": class_id, "name": name, "position": position, "retired": 0}

    def create_dataset(self, actor: Actor, name: str, classes: list[str] | tuple[str, ...] = (),
                       operation_id: str = "") -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            dataset_id = _id("dataset")
            con.execute("INSERT INTO hosted_datasets(id,name,created_by,created_at) VALUES (?,?,?,?)",
                        (dataset_id, _text(name, "name"), actor.user_id, self.clock()))
            for class_name in classes:
                self._add_class(con, actor, dataset_id, class_name)
            con.execute("INSERT INTO hosted_class_versions VALUES (?,?,?,?,?)",
                        (dataset_id, 0, _json(self._classes(con, dataset_id)), actor.user_id, self.clock()))
            self._event(con, actor, dataset_id, None, "dataset_created", {})
            return self._dataset_result(con, actor, dataset_id)
        return self._command(actor, operation_id, "create_dataset", {"name": name, "classes": classes}, run, admin=True)

    def _dataset_result(self, con: sqlite3.Connection, actor: Actor, dataset_id: str) -> dict[str, Any]:
        dataset = dict(self._dataset(con, dataset_id))
        query, args = ("SELECT state,COUNT(*) AS n FROM hosted_images WHERE dataset_id=? "
                       "AND id NOT IN (SELECT image_id FROM hosted_archived_images)"), [dataset_id]
        if actor.role != "admin":
            query += " AND assignee_id=? AND state IN ('assigned','returned')"
            args.append(actor.user_id)
        dataset["counts"] = {row["state"]: row["n"] for row in con.execute(query + " GROUP BY state", args)}
        dataset["classes"] = self._classes(con, dataset_id)
        settings = con.execute("SELECT version,settings_json FROM hosted_dataset_settings WHERE dataset_id=?", (dataset_id,)).fetchone()
        dataset["settings"] = json.loads(settings["settings_json"]) if settings else {}
        dataset["settings_version"] = settings["version"] if settings else 0
        return dataset

    def set_dataset_settings(self, actor: Actor, dataset_id: str, settings: dict[str, Any],
                             expected_revision: int, operation_id: str) -> dict[str, Any]:
        """Version model/recipe configuration; route adapters validate its schema."""
        return self._change_dataset_settings(actor, dataset_id, settings, expected_revision, operation_id, patch=False)

    def patch_dataset_settings(self, actor: Actor, dataset_id: str, patch: dict[str, Any],
                               expected_revision: int, operation_id: str) -> dict[str, Any]:
        """Merge top-level settings atomically, with a receipt for the stable patch.

        Replaying a successful request returns its original result even when
        another command has since changed an unrelated setting. Nested values
        are replaced as units; this is deliberately not a recursive JSON merge.
        """
        return self._change_dataset_settings(actor, dataset_id, patch, expected_revision, operation_id, patch=True)

    def _change_dataset_settings(self, actor: Actor, dataset_id: str, settings: dict[str, Any],
                                 expected_revision: int, operation_id: str, *, patch: bool) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            dataset = self._dataset(con, dataset_id)
            if dataset["revision"] != expected_revision:
                raise DomainError("dataset_revision_changed", 409)
            if not isinstance(settings, dict):
                raise DomainError("invalid_dataset_settings", 422)
            value = settings
            if patch:
                current = con.execute("SELECT settings_json FROM hosted_dataset_settings WHERE dataset_id=?", (dataset_id,)).fetchone()
                value = {**(json.loads(current[0]) if current else {}), **settings}
            encoded = _json(value)
            if len(encoded.encode()) > 1024 * 1024:
                raise DomainError("dataset_settings_too_large", 413)
            version = dataset["revision"] + 1
            con.execute("INSERT INTO hosted_dataset_settings VALUES (?,?,?) ON CONFLICT(dataset_id) DO UPDATE SET version=excluded.version,settings_json=excluded.settings_json", (dataset_id, version, encoded))
            con.execute("INSERT INTO hosted_dataset_setting_versions VALUES (?,?,?,?,?)", (dataset_id, version, encoded, actor.user_id, self.clock()))
            con.execute("UPDATE hosted_datasets SET revision=? WHERE id=?", (version, dataset_id))
            self._event(con, actor, dataset_id, None, "dataset_settings_changed", {"settings_version": version})
            return self._dataset_result(con, actor, dataset_id)
        operation = "patch_dataset_settings" if patch else "set_dataset_settings"
        return self._command(actor, operation_id, operation, dict(dataset_id=dataset_id,settings=settings,expected_revision=expected_revision), run, admin=True)

    def get_dataset(self, actor: Actor, dataset_id: str) -> dict[str, Any]:
        with self.read_connection() as con:
            self._actor(con, actor)
            if actor.role != "admin" and not con.execute(
                    "SELECT 1 FROM hosted_images WHERE dataset_id=? AND assignee_id=? AND state IN ('assigned','returned') LIMIT 1",
                    (dataset_id, actor.user_id)).fetchone():
                raise DomainError("dataset_not_found", 404)
            return self._dataset_result(con, actor, dataset_id)

    @staticmethod
    def _pagination(limit: int, offset: int) -> tuple[int, int]:
        if not isinstance(limit, int) or not isinstance(offset, int) or not 1 <= limit <= 1000 or offset < 0:
            raise DomainError("invalid_pagination", 422)
        return limit, offset

    def list_datasets(self, actor: Actor, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        self._pagination(limit, offset)
        with self.read_connection() as con:
            self._actor(con, actor)
            query, args = "SELECT id FROM hosted_datasets WHERE id NOT IN (SELECT dataset_id FROM hosted_archived_datasets)", []
            if actor.role != "admin":
                query += " AND id IN (SELECT dataset_id FROM hosted_images WHERE assignee_id=? AND state IN ('assigned','returned') AND id NOT IN (SELECT image_id FROM hosted_archived_images))"
                args.append(actor.user_id)
            args.extend([limit, offset])
            return [self._dataset_result(con, actor, row[0]) for row in con.execute(query + " ORDER BY created_at,id LIMIT ? OFFSET ?", args)]

    @staticmethod
    def _validate_language_annotations(annotations: dict[str, Any]) -> None:
        """Validate the structures shared by review and canonical consumers.

        Optional legacy fields may be null, and caption collections may contain
        plain strings. Validate known fields without normalizing text, dropping
        extension metadata or changing stored review/model evidence.
        """
        def typed(value, field, kind):
            if value is not None and not isinstance(value, kind):
                raise DomainError("invalid_annotation_language", 422,
                                  {"field": field, "expected": kind.__name__})

        def qa_pairs(value, field):
            typed(value, field, list)
            for index, pair in enumerate(value or []):
                path = f"{field}[{index}]"
                if not isinstance(pair, dict):
                    raise DomainError("invalid_annotation_language", 422,
                                      {"field": path, "expected": "dict"})
                for key in ("question", "answer"):
                    typed(pair.get(key), f"{path}.{key}", str)
                typed(pair.get("metadata"), f"{path}.metadata", dict)
                typed(pair.get("validation"), f"{path}.validation", dict)

        typed(annotations.get("caption"), "caption", str)
        for collection in ("captions", "alternate_captions"):
            values = annotations.get(collection)
            typed(values, collection, list)
            for index, record in enumerate(values or []):
                if isinstance(record, str):
                    continue
                path = f"{collection}[{index}]"
                if not isinstance(record, dict):
                    raise DomainError("invalid_annotation_language", 422,
                                      {"field": path, "expected": "dict or str"})
                for key in ("caption", "text"):
                    typed(record.get(key), f"{path}.{key}", str)
                metadata = record.get("metadata")
                typed(metadata, f"{path}.metadata", dict)
                # Check both locations even if the first currently shadows the
                # second; a later edit must not uncover malformed saved data.
                qa_pairs(record.get("generated_qa_pairs"), f"{path}.generated_qa_pairs")
                qa_pairs((metadata or {}).get("generated_qa_pairs"),
                         f"{path}.metadata.generated_qa_pairs")

    def _annotations(self, con: sqlite3.Connection, dataset_id: str, annotations: Any,
                     *, width: int, height: int) -> dict[str, Any]:
        if not isinstance(annotations, dict):
            raise DomainError("invalid_annotations", 422)
        encoded = _json(annotations)
        if len(encoded.encode()) > 16 * 1024 * 1024:
            raise DomainError("annotations_too_large", 413)
        result = json.loads(encoded)
        self._validate_language_annotations(result)
        objects = result.setdefault("objects", [])
        if not isinstance(objects, list):
            raise DomainError("invalid_objects", 422)
        valid = {row["id"] for row in self._classes(con, dataset_id) if not row["retired"]}
        seen = set()
        for item in objects:
            if not isinstance(item, dict):
                raise DomainError("invalid_object", 422)
            object_id = _text(item.get("id"), "object.id", 200)
            if object_id in seen:
                raise DomainError("duplicate_object_id", 422)
            seen.add(object_id)
            item["id"] = object_id
            if not isinstance(item.get("class_id"), str) or item["class_id"] not in valid:
                raise DomainError("invalid_class", 422, {"class_id": item.get("class_id")})
            bbox = item.get("bbox")
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)):
                raise DomainError("invalid_bbox", 422, {"object_id": object_id})
            x, y, w, h = bbox
            if width <= 0 or height <= 0:
                raise DomainError("image_dimensions_required", 422)
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise DomainError("bbox_out_of_bounds", 422, {"object_id": object_id})
            if "polygon" in item:
                polygon = item["polygon"]
                if (not isinstance(polygon, list) or len(polygon) < 3 or any(
                        not isinstance(point, list) or len(point) != 2 or any(
                            type(v) not in (int, float) or not math.isfinite(v) for v in point)
                        for point in polygon)):
                    raise DomainError("invalid_polygon", 422, {"object_id": object_id})
                if any(not (0 <= point[0] <= width and 0 <= point[1] <= height) for point in polygon):
                    raise DomainError("polygon_out_of_bounds", 422, {"object_id": object_id})
        return result

    def _version(self, con: sqlite3.Connection, actor: Actor, image_id: str, dataset_id: str,
                 annotations: dict[str, Any], *, parent: str | None, kind: str,
                 complete: bool = False, negative: bool = False) -> str:
        if type(complete) is not bool or type(negative) is not bool:
            raise DomainError("invalid_completion", 422)
        if negative and annotations["objects"]:
            raise DomainError("negative_has_objects", 422)
        version_id = _id("version")
        value = _json(annotations)
        con.execute("INSERT INTO hosted_annotation_versions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (version_id, image_id, parent, value, hashlib.sha256(value.encode()).hexdigest(),
                     int(complete or negative), int(negative), self._dataset(con, dataset_id)["class_version"],
                     actor.user_id, kind, self.clock()))
        return version_id

    def _image_result(self, con: sqlite3.Connection, image_id: str, *, content: bool = False) -> dict[str, Any]:
        image = dict(self._image(con, image_id))
        version = con.execute("SELECT * FROM hosted_annotation_versions WHERE id=?", (image["current_version"],)).fetchone()
        image.update({"version_id": version["id"], "complete": bool(version["complete"]), "negative": bool(version["negative"]),
                      "class_version": version["class_version"], "dataset_revision": self._dataset(con, image["dataset_id"])["revision"]})
        lease = con.execute("SELECT actor_id,editor_id,generation,expires_at FROM hosted_editor_leases WHERE image_id=?", (image_id,)).fetchone()
        image["editor_lease"] = dict(lease) if lease and lease["expires_at"] > self.clock() else None
        if content:
            image["annotations"] = json.loads(version["annotations_json"])
            media = con.execute("SELECT size_bytes,width,height FROM hosted_media WHERE sha256=?", (image["media_hash"],)).fetchone()
            image.update(dict(media))
        return image

    def image(self, actor: Actor, image_id: str) -> dict[str, Any]:
        with self.read_connection() as con:
            self._actor(con, actor)
            self._can_view(actor, self._image(con, image_id))
            return self._image_result(con, image_id, content=True)

    def list_images(self, actor: Actor, dataset_id: str | None = None, state: str | None = None,
                    *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        self._pagination(limit, offset)
        if state is not None and state not in STATES + ("incoming",):
            raise DomainError("invalid_state", 422)
        with self.read_connection() as con:
            self._actor(con, actor)
            conditions, args = ["id NOT IN (SELECT image_id FROM hosted_archived_images)",
                                "dataset_id NOT IN (SELECT dataset_id FROM hosted_archived_datasets)"], []
            if dataset_id:
                conditions.append("dataset_id=?")
                args.append(dataset_id)
            if state == "incoming":
                conditions.append("state!='confirmed'")
            elif state:
                conditions.append("state=?")
                args.append(state)
            if actor.role != "admin":
                conditions.extend(["assignee_id=?", "state IN ('assigned','returned')"])
                args.append(actor.user_id)
            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            rows = con.execute("SELECT id FROM hosted_images" + where + " ORDER BY created_at,id LIMIT ? OFFSET ?", [*args, limit, offset])
            return [self._image_result(con, row[0]) for row in rows]

    def import_image(self, actor: Actor, dataset_id: str, media_hash: str, filename: str,
                     annotations: dict[str, Any] | None = None, operation_id: str = "", *,
                     media_path: str = "", width: int = 0, height: int = 0,
                     size_bytes: int = 0, split: str = "train") -> dict[str, Any]:
        payload = dict(dataset_id=dataset_id, media_hash=media_hash, filename=filename, annotations=annotations,
                       media_path=media_path, width=width, height=height, size_bytes=size_bytes, split=split)
        return self._command(actor, operation_id, "import_image", payload,
                             lambda con: self._import_image(con, actor, **payload), admin=True)

    def _import_image(self, con: sqlite3.Connection, actor: Actor, *, dataset_id: str, media_hash: str,
                      filename: str, annotations: dict[str, Any] | None, media_path: str = "",
                      width: int = 0, height: int = 0, size_bytes: int = 0, split: str = "train") -> dict[str, Any]:
        dataset = self._dataset(con, dataset_id)
        if not isinstance(media_hash, str) or not _SHA256.fullmatch(media_hash):
            raise DomainError("invalid_media_hash", 422)
        name = _text(filename, "filename", 1000)
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
            raise DomainError("invalid_filename", 422)
        if split not in ("train", "val", "test", "unsplit"):
            raise DomainError("invalid_split", 422)
        if any(type(v) is not int or v < 0 for v in (width, height, size_bytes)):
            raise DomainError("invalid_media_dimensions", 422)
        body = self._annotations(con, dataset_id, annotations if annotations is not None else {}, width=width, height=height)
        existing = con.execute("SELECT * FROM hosted_images WHERE dataset_id=? AND media_hash=?", (dataset_id, media_hash)).fetchone()
        if existing:
            if con.execute("SELECT 1 FROM hosted_archived_images WHERE image_id=?", (existing["id"],)).fetchone():
                raise DomainError("duplicate_image_archived", 409, {"image_id": existing["id"]})
            current = con.execute("SELECT annotations_json FROM hosted_annotation_versions WHERE id=?", (existing["current_version"],)).fetchone()[0]
            conflict_id = None
            if annotations is not None and _json(body) != current:
                conflict_id = _id("importconflict")
                con.execute("INSERT INTO hosted_import_conflicts VALUES (?,?,?,?,?,?,?)",
                            (conflict_id, existing["id"], name, _json(body), dataset["class_version"], actor.user_id, self.clock()))
            self._event(con, actor, dataset_id, existing["id"], "duplicate_import", {"filename": name, "conflict_id": conflict_id})
            return {**self._image_result(con, existing["id"]), "duplicate": True, "conflict_id": conflict_id}
        media = con.execute("SELECT * FROM hosted_media WHERE sha256=?", (media_hash,)).fetchone()
        if media:
            for key, value in (("size_bytes", size_bytes), ("width", width), ("height", height)):
                if value and media[key] and value != media[key]:
                    raise DomainError("media_metadata_conflict", 409)
        else:
            con.execute("INSERT INTO hosted_media VALUES (?,?,?,?,?,?)", (media_hash, size_bytes, width, height, media_path, self.clock()))
        image_id, initial_version = _id("image"), _id("reserved")
        now = self.clock()
        con.execute("INSERT INTO hosted_images(id,dataset_id,media_hash,filename,split,state,current_version,created_at,updated_at) "
                    "VALUES (?,?,?,?,?,'unassigned',?,?,?)", (image_id, dataset_id, media_hash, name, split, initial_version, now, now))
        version_id = self._version(con, actor, image_id, dataset_id, body, parent=None, kind="import")
        con.execute("UPDATE hosted_images SET current_version=? WHERE id=?", (version_id, image_id))
        collisions = [r[0] for r in con.execute("SELECT id FROM hosted_images WHERE dataset_id=? AND filename=? AND id!=?", (dataset_id, name, image_id))]
        self._event(con, actor, dataset_id, image_id, "imported", {"version_id": version_id, "filename_collisions": collisions})
        return {**self._image_result(con, image_id), "duplicate": False, "conflict_id": None, "filename_collisions": collisions}

    def import_yolo_image(self, actor: Actor, dataset_id: str, media_hash: str, filename: str,
                          label_text: str, class_names: list[str], expected_class_version: int,
                          operation_id: str, *, width: int, height: int, size_bytes: int, split: str = 'unsplit') -> dict[str, Any]:
        payload = dict(dataset_id=dataset_id, media_hash=media_hash, filename=filename, label_text=label_text,
                       class_names=class_names, expected_class_version=expected_class_version,
                       width=width, height=height, size_bytes=size_bytes, split=split)
        def run(con):
            dataset = self._dataset(con, dataset_id)
            if type(expected_class_version) is not int or dataset['class_version'] != expected_class_version:
                raise DomainError('class_version_changed', 409)
            if (not isinstance(label_text, str) or len(label_text.encode()) > 8 * 1024 * 1024
                    or not isinstance(class_names, list) or len(class_names) > 10000
                    or any(not isinstance(name, str) or not name.strip() or '\n' in name or '\r' in name for name in class_names)):
                raise DomainError('invalid_yolo_import', 422)
            names = [name.strip() for name in class_names]
            if len({name.casefold() for name in names}) != len(names):
                raise DomainError('ambiguous_import_labelmap', 422)
            known = {row['name'].casefold(): row['id'] for row in self._classes(con, dataset_id) if not row['retired']}
            missing = [name for name in names if name.casefold() not in known]
            if missing:
                raise DomainError('import_classes_missing', 422, {'classes': missing})
            from .migration import _objects
            objects = _objects(label_text, width, height, names, clip=False, key=filename)
            annotations = {'objects': [{**{key: value for key, value in item.items() if key != 'class_index'},
                                        'class_id': known[names[item['class_index']].casefold()]} for item in objects],
                           'import_provenance': {'format': 'yolo', 'filename': filename, 'label_text': label_text,
                                                 'class_names': names, 'class_version': expected_class_version}}
            return self._import_image(con, actor, dataset_id=dataset_id, media_hash=media_hash, filename=filename,
                                      annotations=annotations, width=width, height=height, size_bytes=size_bytes, split=split)
        return self._command(actor, operation_id, 'import_yolo_image', payload, run, admin=True)

    def resolve_import_conflict(self, actor: Actor, ctx: MutationContext, conflict_id: str,
                                choice: str, *, negative: bool = False) -> dict[str, Any]:
        def run(con):
            image = self._guard(con, actor, ctx)
            conflict = con.execute('SELECT * FROM hosted_import_conflicts WHERE id=? AND image_id=?', (conflict_id, ctx.image_id)).fetchone()
            if conflict is None:
                raise DomainError('import_conflict_not_found', 404)
            if con.execute('SELECT 1 FROM hosted_import_conflict_decisions WHERE conflict_id=?', (conflict_id,)).fetchone():
                raise DomainError('import_conflict_already_reviewed', 409)
            if choice not in ('keep_current', 'use_imported_objects') or type(negative) is not bool:
                raise DomainError('invalid_conflict_choice', 422)
            if choice == 'use_imported_objects':
                current = con.execute('SELECT annotations_json FROM hosted_annotation_versions WHERE id=?', (ctx.expected_version,)).fetchone()
                annotations = json.loads(current[0])
                annotations['objects'] = json.loads(conflict['annotations_json'])['objects']
                if not annotations['objects'] and not negative:
                    raise DomainError('empty_annotation_requires_negative_review', 409)
                provenance = annotations.setdefault('import_reviews', [])
                if not isinstance(provenance, list):
                    raise DomainError('invalid_import_review_metadata', 422)
                provenance.append({'conflict_id': conflict_id, 'previous_version': ctx.expected_version, 'actor_id': actor.user_id})
                self._save(con, actor, ctx, annotations, complete=image['state'] == 'confirmed', negative=negative)
            result = self._image_result(con, ctx.image_id)
            con.execute('INSERT INTO hosted_import_conflict_decisions VALUES(?,?,?,?,?,?,?)',
                        (conflict_id, ctx.image_id, actor.user_id, choice, ctx.expected_version, result['current_version'], self.clock()))
            self._event(con, actor, image['dataset_id'], ctx.image_id, 'import_conflict_resolved',
                        {'conflict_id': conflict_id, 'choice': choice, 'previous_version': ctx.expected_version, 'result_version': result['current_version']})
            return result
        return self._command(actor, ctx.operation_id, 'resolve_import_conflict',
                             {'context': asdict(ctx), 'conflict_id': conflict_id, 'choice': choice, 'negative': negative}, run, admin=True)

    def acquire_editor(self, actor: Actor, image_id: str, editor_id: str,
                       expected_version: str | None = None, generation: int | None = None,
                       *, takeover: bool = False) -> dict[str, Any]:
        editor_id = _text(editor_id, "editor_id", 200)
        with self.transaction() as con:
            self._actor(con, actor)
            image = self._image(con, image_id)
            self._can_view(actor, image)
            self._can_edit(actor, image)
            if expected_version is not None or generation is not None:
                self._expected(image, expected_version, generation)
            lease = con.execute("SELECT * FROM hosted_editor_leases WHERE image_id=?", (image_id,)).fetchone()
            if lease and lease["expires_at"] > self.clock():
                if (lease["actor_id"], lease["editor_id"]) != (actor.user_id, editor_id):
                    if not takeover or lease["actor_id"] != actor.user_id:
                        raise DomainError("editor_lease_conflict", 423, {"expires_at": lease["expires_at"]})
                    self._expected(image, expected_version, generation)
                    con.execute("UPDATE hosted_images SET generation=generation+1 WHERE id=?", (image_id,))
                    con.execute("UPDATE hosted_editor_leases SET editor_id=?,generation=generation+1,expires_at=? WHERE image_id=?",
                                (editor_id, self.clock() + self.lease_seconds, image_id))
                    self._event(con, actor, image["dataset_id"], image_id, "editor_taken_over", {"previous_editor_id": lease["editor_id"], "editor_id": editor_id})
                else:
                    con.execute("UPDATE hosted_editor_leases SET expires_at=? WHERE image_id=?", (self.clock() + self.lease_seconds, image_id))
            else:
                con.execute("UPDATE hosted_images SET generation=generation+1 WHERE id=?", (image_id,))
                con.execute("INSERT INTO hosted_editor_leases VALUES (?,?,?,?,?) ON CONFLICT(image_id) DO UPDATE SET "
                            "actor_id=excluded.actor_id,editor_id=excluded.editor_id,generation=excluded.generation,expires_at=excluded.expires_at",
                            (image_id, actor.user_id, editor_id, image["generation"] + 1, self.clock() + self.lease_seconds))
            return self._image_result(con, image_id, content=True)

    def _handoff(self, con: sqlite3.Connection, image_id: str, state: str, *, assignee: str | None,
                 assignment: str | None, submission: str | None = None) -> None:
        image = self._image(con, image_id)
        if image["assignment_id"]:
            con.execute("UPDATE hosted_assignment_members SET state=? WHERE assignment_id=? AND image_id=?",
                        (state if state != "unassigned" else "overridden", image["assignment_id"], image_id))
        con.execute("UPDATE hosted_images SET state=?,assignee_id=?,assignment_id=?,active_submission=?,generation=generation+1,updated_at=? WHERE id=?",
                    (state, assignee, assignment, submission, self.clock(), image_id))
        con.execute("DELETE FROM hosted_editor_leases WHERE image_id=?", (image_id,))

    def release_editor(self, actor: Actor, ctx: MutationContext) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            image = self._guard(con, actor, ctx)
            con.execute("DELETE FROM hosted_editor_leases WHERE image_id=?", (ctx.image_id,))
            con.execute("UPDATE hosted_images SET generation=generation+1 WHERE id=?", (ctx.image_id,))
            self._event(con, actor, image["dataset_id"], ctx.image_id, "editor_released", {})
            return self._image_result(con, ctx.image_id)
        return self._command(actor, ctx.operation_id, "release_editor", asdict(ctx), run)

    def assign(self, actor: Actor, dataset_id: str, image_ids: list[str], annotator_id: str,
               operation_id: str, expected: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            self._dataset(con, dataset_id)
            self._annotator(con, annotator_id)
            if not image_ids or len(set(image_ids)) != len(image_ids) or not expected or set(expected) != set(image_ids):
                raise DomainError("exact_assignment_preconditions_required", 422)
            for image_id in image_ids:
                image = self._image(con, image_id)
                if image["dataset_id"] != dataset_id or image["state"] != "unassigned":
                    raise DomainError("image_not_assignable", 409, {"image_id": image_id})
                pre = expected[image_id]
                self._expected(image, pre.get("expected_version", pre.get("version_id")), pre.get("generation"))
                lease = con.execute("SELECT expires_at FROM hosted_editor_leases WHERE image_id=?", (image_id,)).fetchone()
                if lease and lease[0] > self.clock():
                    raise DomainError("editor_lease_conflict", 423, {"image_id": image_id})
            assignment_id = _id("assignment")
            con.execute("INSERT INTO hosted_assignments VALUES (?,?,?,?,?)", (assignment_id, dataset_id, annotator_id, actor.user_id, self.clock()))
            for image_id in image_ids:
                image = self._image(con, image_id)
                con.execute("INSERT INTO hosted_assignment_members VALUES (?,?,?,'assigned')", (assignment_id, image_id, image["current_version"]))
                self._handoff(con, image_id, "assigned", assignee=annotator_id, assignment=assignment_id)
                self._event(con, actor, dataset_id, image_id, "assigned", {"assignment_id": assignment_id, "annotator_id": annotator_id})
            return {"assignment_id": assignment_id, "images": [self._image_result(con, image_id) for image_id in image_ids]}
        return self._command(actor, operation_id, "assign", dict(dataset_id=dataset_id,image_ids=image_ids,annotator_id=annotator_id,expected=expected), run, admin=True)

    @staticmethod
    def _split_population(con, dataset_id):
        """Confirmed membership and transitive capture groups; no pending rows."""
        from .splits import capture_groups
        rows, parent, labels, owners = {}, {}, {}, {}
        def find(identifier):
            while parent[identifier] != identifier:
                parent[identifier] = parent[parent[identifier]]
                identifier = parent[identifier]
            return identifier
        query = ("SELECT i.id,i.filename,i.split,i.current_version,i.generation,v.annotations_json "
                 "FROM hosted_images i JOIN hosted_annotation_versions v ON v.id=i.confirmed_version "
                 "WHERE i.dataset_id=? AND i.state='confirmed' "
                 "AND NOT EXISTS(SELECT 1 FROM hosted_archived_images a WHERE a.image_id=i.id) ORDER BY i.id")
        for row in con.execute(query, (dataset_id,)):
            item = dict(row); groups = capture_groups(json.loads(item.pop('annotations_json')))
            identifier = item['id']; rows[identifier] = item; parent[identifier] = identifier; labels[identifier] = groups
            for group in groups:
                other = owners.setdefault(group, identifier)
                parent[find(identifier)] = find(other)
        cohorts = {}
        for identifier in rows:
            cohorts.setdefault(find(identifier), []).append(identifier)
        return rows, cohorts, labels

    def preview_splits(self, actor: Actor, dataset_id: str, image_ids: list[str], target_split: str) -> dict[str, Any]:
        from .splits import SPLITS
        if (not isinstance(image_ids, list) or not 1 <= len(image_ids) <= 50000
                or any(not isinstance(value, str) for value in image_ids) or len(set(image_ids)) != len(image_ids)
                or target_split not in SPLITS):
            raise DomainError('invalid_split_selection', 422)
        with self.read_connection() as con:
            self._actor(con, actor, admin=True)
            dataset = self._dataset(con, dataset_id)
            rows, cohorts, labels = self._split_population(con, dataset_id)
            selected = set(image_ids)
            if not selected <= set(rows):
                raise DomainError('split_image_not_confirmed', 409)
            affected, groups = set(), []
            for members in cohorts.values():
                if selected.intersection(members):
                    affected.update(members)
                    groups.append({'image_ids': members, 'labels': [list(value) for value in sorted({value for identifier in members for value in labels[identifier]})]})
            if len(affected) > 50000:
                raise DomainError('split_group_selection_too_large', 422)
            before = {split: sum(row['split'] == split for row in rows.values()) for split in SPLITS}
            after = dict(before)
            for identifier in affected:
                after[rows[identifier]['split']] -= 1; after[target_split] += 1
            assignments = {identifier: target_split for identifier in sorted(affected)}
            return {'dataset_id': dataset_id, 'expected_revision': dataset['revision'],
                    'selected_count': len(selected), 'group_added_count': len(affected - selected),
                    'change_count': sum(rows[identifier]['split'] != target_split for identifier in affected),
                    'counts_before': before, 'counts_after': after, 'groups': groups,
                    'assignments': assignments, 'expected': {identifier: {'expected_version': rows[identifier]['current_version'], 'generation': rows[identifier]['generation']} for identifier in assignments},
                    'images': [{**rows[identifier], 'selected': identifier in selected, 'target_split': target_split} for identifier in assignments]}

    def set_splits(self, actor: Actor, dataset_id: str, assignments: dict[str, str], expected_revision: int,
                   expected: dict[str, dict[str, Any]], operation_id: str) -> dict[str, Any]:
        from .splits import SPLITS
        def run(con):
            dataset = self._dataset(con, dataset_id)
            if type(expected_revision) is not int or dataset['revision'] != expected_revision:
                raise DomainError('dataset_revision_changed', 409)
            if (not isinstance(assignments, dict) or not 1 <= len(assignments) <= 50000
                    or any(not isinstance(key, str) or value not in SPLITS for key, value in assignments.items())
                    or not isinstance(expected, dict) or set(assignments) != set(expected)):
                raise DomainError('exact_split_preconditions_required', 422)
            rows, cohorts, labels = self._split_population(con, dataset_id)
            if not set(assignments) <= set(rows):
                raise DomainError('split_image_not_confirmed', 409)
            for identifier, pre in expected.items():
                if not isinstance(pre, dict):
                    raise DomainError('exact_split_preconditions_required', 422)
                self._expected(rows[identifier], pre.get('expected_version'), pre.get('generation'))
                lease = con.execute('SELECT expires_at FROM hosted_editor_leases WHERE image_id=?', (identifier,)).fetchone()
                if lease and lease[0] > self.clock():
                    raise DomainError('editor_lease_conflict', 423, {'image_id': identifier})
            targets = set(assignments)
            for members in cohorts.values():
                if targets.intersection(members) and len({assignments.get(identifier, rows[identifier]['split']) for identifier in members}) > 1:
                    raise DomainError('split_group_conflict', 409, {'image_ids': members})
            changed = 0
            for identifier, split in assignments.items():
                old = rows[identifier]['split']
                if old == split:
                    continue
                con.execute('UPDATE hosted_images SET split=?,generation=generation+1,updated_at=? WHERE id=?', (split, self.clock(), identifier))
                con.execute('DELETE FROM hosted_editor_leases WHERE image_id=?', (identifier,))
                self._event(con, actor, dataset_id, identifier, 'split_changed', {'from': old, 'to': split, 'groups': [list(value) for value in labels[identifier]]})
                changed += 1
            if changed:
                con.execute('UPDATE hosted_datasets SET revision=revision+1 WHERE id=?', (dataset_id,))
            counts = {split: 0 for split in SPLITS}
            for identifier, row in rows.items():
                counts[assignments.get(identifier, row['split'])] += 1
            return {'dataset_id': dataset_id, 'revision': self._dataset(con, dataset_id)['revision'], 'changed_count': changed,
                    'assignments': assignments,
                    'counts': counts, 'images': [self._image_result(con, identifier) for identifier in sorted(assignments)]}
        return self._command(actor, operation_id, 'set_splits', dict(dataset_id=dataset_id,
            assignments=assignments, expected_revision=expected_revision, expected=expected), run, admin=True)

    def save(self, actor: Actor, ctx: MutationContext, annotations: dict[str, Any], *,
             complete: bool = False, negative: bool = False) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            return self._save(con, actor, ctx, annotations, complete=complete, negative=negative)
        return self._command(actor, ctx.operation_id, "save", {"context": asdict(ctx), "annotations": annotations, "complete": complete, "negative": negative}, run)

    def _save(self, con: sqlite3.Connection, actor: Actor, ctx: MutationContext,
              annotations: dict[str, Any], *, complete: bool, negative: bool,
              trusted_instruction_pairs=()) -> dict[str, Any]:
        """Shared commit mechanism for interactive and guarded analysis edits."""
        image = self._guard(con, actor, ctx)
        media = con.execute("SELECT width,height FROM hosted_media WHERE sha256=?", (image["media_hash"],)).fetchone()
        body = self._annotations(con, image["dataset_id"], annotations, width=media["width"], height=media["height"])
        if actor.role != 'admin':
            previous = con.execute('SELECT annotations_json FROM hosted_annotation_versions WHERE id=?',
                                   (ctx.expected_version,)).fetchone()
            previous = json.loads(previous['annotations_json'])
            self._preserve_instruction_reviews(previous, body)
            self._preserve_machine_validations(con, actor, ctx, previous, body,
                                               trusted_instruction_pairs=trusted_instruction_pairs)
        if image["state"] == "confirmed" and not (complete or negative):
            raise DomainError("confirmed_annotation_requires_completion", 409)
        version_id = self._version(con, actor, ctx.image_id, image["dataset_id"], body,
                                   parent=ctx.expected_version, kind="canonical_edit" if image["state"] == "confirmed" else "draft",
                                   complete=complete, negative=negative)
        if image["state"] == "confirmed":
            self._require_complete(con, version_id)
            con.execute("UPDATE hosted_images SET current_version=?,confirmed_version=?,updated_at=? WHERE id=?", (version_id, version_id, self.clock(), ctx.image_id))
            con.execute("UPDATE hosted_datasets SET revision=revision+1 WHERE id=?", (image["dataset_id"],))
        else:
            con.execute("UPDATE hosted_images SET current_version=?,updated_at=? WHERE id=?", (version_id, self.clock(), ctx.image_id))
        self._event(con, actor, image["dataset_id"], ctx.image_id, "annotation_saved", {"version_id": version_id, "previous_version": ctx.expected_version})
        return self._image_result(con, ctx.image_id)

    @staticmethod
    def _preserve_instruction_reviews(previous, current):
        """Annotators can edit language, but cannot create administrator review.

        Check both native and hosted review paths. Hosted stamps bind their own
        text digest, so edited text can keep its historical evidence safely.
        Older native stamps lack that binding and must keep the reviewed text.
        Removing a complete record is allowed; immutable history retains it.
        """
        from collections import Counter

        protected = {'review_decision', 'review_status', 'review_notes', 'reviewer',
                     'reviewed_at', 'review_row_origin', 'review_recorded_via',
                     'review_source_version', 'review_content_sha256'}
        neutral = {'', 'unreviewed', 'machine_validated', 'unknown', 'pending'}

        def records(annotations):
            for collection in ('captions', 'alternate_captions'):
                values = annotations.get(collection)
                if not isinstance(values, list):
                    continue
                for value in values:
                    if not isinstance(value, dict):
                        continue
                    yield 'caption0', value
                    metadata = value.get('metadata')
                    metadata = metadata if isinstance(metadata, dict) else {}
                    # Inspect both locations, including a currently shadowed
                    # list: a later edit must not expose a forged old stamp.
                    for pairs in (value.get('generated_qa_pairs'), metadata.get('generated_qa_pairs')):
                        if isinstance(pairs, list):
                            for pair in pairs:
                                if isinstance(pair, dict):
                                    yield 'generated_qa', pair

        def evidence(kind, record):
            metadata = record.get('metadata')
            metadata = metadata if isinstance(metadata, dict) else {}
            def fields(value):
                return {key: value[key] for key in protected.intersection(value)
                        if value[key] is not None and str(value[key]).strip()}
            top, meta = fields(record), fields(metadata)
            allowed_neutral = neutral - {'machine_validated'} if kind == 'caption0' else neutral
            def meaningful(values):
                return any(key not in {'review_decision', 'review_status'} or
                           re.sub(r'[\s-]+', '_', str(value).strip().lower()) not in allowed_neutral
                           for key, value in values.items())
            if not meaningful(top) and not meaningful(meta):
                return None
            # UI normalization adds image names, positions and empty dates.
            # None of those fields grants review authority. Bind only human
            # evidence, its reviewed text, and comparable stable identities.
            payload = {'top': top, 'metadata': meta}
            bound = (metadata.get('review_recorded_via') == 'hosted_instruction_review'
                    and isinstance(metadata.get('review_content_sha256'), str)
                    and re.fullmatch(r'[0-9a-f]{64}', metadata['review_content_sha256']))
            if not bound:
                payload['text'] = {key: str(record.get(key) or '').strip()
                                   for key in (('caption',) if kind == 'caption0' else ('question', 'answer'))}
            return _json([kind, payload])

        available = {}
        known_ids = {}
        for kind, record in records(previous):
            signature = evidence(kind, record)
            if signature is not None:
                identifier = str(record.get('id') or '').strip()
                available.setdefault(signature, Counter())[identifier] += 1
                if identifier:
                    known_ids.setdefault((kind, identifier), set()).add(signature)
        for kind, record in records(current):
            signature = evidence(kind, record)
            identifier = str(record.get('id') or '').strip()
            identity = (kind, identifier)
            if signature is None:
                if identity in known_ids:
                    raise DomainError('instruction_review_requires_admin', 403)
                continue
            identities = available.get(signature, Counter())
            # The native caption UI assigns IDs to imported records that did
            # not have one. Once present, that ID cannot be switched or lost.
            match = identifier if identities[identifier] else ''
            if identities[match] <= 0:
                raise DomainError('instruction_review_requires_admin', 403)
            identities[match] -= 1

    def _preserve_machine_validations(self, con, actor, ctx, previous, current, *, trusted_instruction_pairs=()):
        """Machine validation belongs to a saved server result, never a client.

        ``trusted_instruction_pairs`` is private service input for a verified
        immutable proposal. Public save bodies cannot set it. Existing evidence
        may travel through UI normalization; content-bound stamps make later
        text edits unqualified at export without rewriting annotation history.
        """
        from collections import Counter

        def pairs(annotations):
            for collection in ('captions', 'alternate_captions'):
                captions = annotations.get(collection)
                if not isinstance(captions, list):
                    continue
                for caption in captions:
                    if not isinstance(caption, dict):
                        continue
                    metadata = caption.get('metadata')
                    metadata = metadata if isinstance(metadata, dict) else {}
                    for values in (caption.get('generated_qa_pairs'), metadata.get('generated_qa_pairs')):
                        if isinstance(values, list):
                            for value in values:
                                if isinstance(value, dict):
                                    yield caption, value

        def signature(pair, *, permit_edits=False, caption=None):
            metadata = pair.get('metadata')
            metadata = metadata if isinstance(metadata, dict) else {}
            flags = {}
            for prefix, container, keys in (
                    ('top', pair, ('validation_status', 'review_status', 'review_decision')),
                    ('metadata', metadata, ('validation_status', 'review_status', 'review_decision')),
                    ('validation', pair.get('validation'), ('status',)),
                    ('metadata.validation', metadata.get('validation'), ('status',))):
                if isinstance(container, dict):
                    for key in keys:
                        value = re.sub(r'[\s-]+', '_', str(container.get(key) or '').strip().lower())
                        if value:
                            flags[prefix + '.' + key] = value
            stamp = {key: metadata[key] for key in ('machine_validation_recorded_via', 'machine_validation_content_sha256')
                     if key in metadata}
            if 'machine_validated' not in flags.values() and not stamp:
                return None
            # Keep verifier/source/quality details exact. Only UI identity,
            # positioning and lifecycle display fields may differ from a
            # server result; none qualifies its language for training.
            display = {'id', 'qa_id', 'image', 'image_name', 'image_key', 'split',
                       'caption_index', 'title', 'is_primary', 'created_at', 'updated_at',
                       'caption_id', 'caption', 'question', 'answer', 'metadata', 'validation',
                       'validation_status', 'review_status', 'review_decision'}
            meta_excluded = {'validation_status', 'review_status', 'review_decision', 'validation',
                             'machine_validation_recorded_via', 'machine_validation_content_sha256'}
            def verifier(value):
                return {key: item for key, item in value.items() if key != 'status'} if isinstance(value, dict) else {}
            payload = {'flags': flags, 'stamp': stamp,
                       'evidence': {key: value for key, value in pair.items() if key not in display},
                       'metadata': {key: value for key, value in metadata.items() if key not in meta_excluded},
                       'validation': verifier(pair.get('validation')),
                       'metadata_validation': verifier(metadata.get('validation'))}
            if caption is not None:
                payload['caption_evidence'] = {key: caption.get(key) for key in ('caption_quality', 'evidence', 'backend_job_id')}
            bound = (stamp.get('machine_validation_recorded_via') == 'hosted_caption_result'
                     and isinstance(stamp.get('machine_validation_content_sha256'), str)
                     and re.fullmatch(r'[0-9a-f]{64}', stamp['machine_validation_content_sha256']))
            if not (permit_edits and bound):
                payload['text'] = {key: str(pair.get(key) or '').strip() for key in ('question', 'answer')}
            return _json(payload)

        previous_evidence = Counter(signature(pair, permit_edits=True, caption=caption) for caption, pair in pairs(previous))
        trusted = Counter(signature(pair) for pair in trusted_instruction_pairs if isinstance(pair, dict))
        jobs = {}
        for caption, pair in pairs(current):
            claimed = signature(pair, caption=caption)
            if claimed is None:
                continue
            carried = signature(pair, permit_edits=True, caption=caption)
            if previous_evidence[carried] > 0:
                previous_evidence[carried] -= 1
                continue
            trusted_claim = signature(pair)
            if trusted[trusted_claim] > 0:
                trusted[trusted_claim] -= 1
                continue
            job_id = caption.get('backend_job_id')
            if not isinstance(job_id, str) or not job_id or len(job_id) > 128:
                raise DomainError('machine_validation_requires_server_evidence', 403)
            if job_id not in jobs:
                exists = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_caption_jobs'").fetchone()
                job = con.execute('SELECT * FROM hosted_caption_jobs WHERE id=?', (job_id,)).fetchone() if exists else None
                if job is None or job['actor_id'] != actor.user_id or job['state'] != 'completed' or not job['result_json']:
                    raise DomainError('machine_validation_requires_server_evidence', 403)
                request = json.loads(job['request_json'])
                image = self._image(con, ctx.image_id)
                if (request.get('dataset_id') != image['dataset_id'] or request.get('image_id') != ctx.image_id
                        or request.get('expected_version') != ctx.expected_version or request.get('generation') != ctx.generation
                        or request.get('editor_id') != ctx.editor_id):
                    raise DomainError('machine_validation_source_changed', 409)
                self._actor(con, Actor(**json.loads(job['actor_json'])))
                result = json.loads(job['result_json'])
                # latest_caption repeats one of captions; never count it twice.
                jobs[job_id] = Counter(signature(value, caption={**record, 'evidence': result.get('evidence'), 'backend_job_id': job_id})
                                      for record, value in pairs({'captions': result.get('captions', [])}))
            if jobs[job_id][claimed] <= 0:
                raise DomainError('machine_validation_requires_server_evidence', 403)
            jobs[job_id][claimed] -= 1

    @staticmethod
    def _require_complete(con: sqlite3.Connection, version_id: str) -> None:
        version = con.execute("SELECT * FROM hosted_annotation_versions WHERE id=?", (version_id,)).fetchone()
        if not version["complete"]:
            raise DomainError("annotation_not_complete", 409)
        annotations = json.loads(version["annotations_json"])
        # Recheck historical drafts admitted before schema validation existed.
        # Reject them for correction, rather than poisoning a canonical snapshot.
        HostedStore._validate_language_annotations(annotations)
        if not annotations["objects"] and not version["negative"]:
            raise DomainError("empty_annotation_requires_negative_review", 409)

    def submit(self, actor: Actor, ctx: MutationContext) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            image = self._guard(con, actor, ctx)
            if actor.role != "annotator" or image["state"] not in ("assigned", "returned"):
                raise DomainError("image_not_submittable", 409)
            self._require_complete(con, ctx.expected_version)
            submission_id = _id("submission")
            con.execute("INSERT INTO hosted_submissions VALUES (?,?,?,?,?,?)", (submission_id, ctx.image_id, ctx.expected_version, image["assignment_id"], actor.user_id, self.clock()))
            self._handoff(con, ctx.image_id, "submitted", assignee=image["assignee_id"], assignment=image["assignment_id"], submission=submission_id)
            self._event(con, actor, image["dataset_id"], ctx.image_id, "submitted", {"submission_id": submission_id, "version_id": ctx.expected_version})
            return self._image_result(con, ctx.image_id)
        return self._command(actor, ctx.operation_id, "submit", asdict(ctx), run)

    def approve(self, actor: Actor, ctx: MutationContext, feedback: str = "") -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            image = self._guard(con, actor, ctx)
            return self._approve(con, actor, ctx, image, feedback)
        return self._command(actor, ctx.operation_id, "approve", {"context": asdict(ctx), "feedback": feedback}, run, admin=True)

    def approve_from_review_queue(self, actor: Actor, ctx: MutationContext, feedback: str = "", *,
                                  dataset_id: str | None = None) -> dict[str, Any]:
        """Approve an exact idle submission without acquiring a transient lease."""
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            image = self._image(con, ctx.image_id)
            if dataset_id is not None and image["dataset_id"] != dataset_id:
                raise DomainError("image_dataset_mismatch", 409)
            self._expected(image, ctx.expected_version, ctx.generation)
            if image["state"] != "submitted":
                raise DomainError("image_not_approvable", 409)
            lease = con.execute("SELECT expires_at FROM hosted_editor_leases WHERE image_id=?", (ctx.image_id,)).fetchone()
            if lease and lease[0] > self.clock():
                raise DomainError("editor_lease_conflict", 423)
            return self._approve(con, actor, ctx, image, feedback)
        result = self._command(actor, ctx.operation_id, "approve_from_review_queue", {"context": asdict(ctx), "feedback": feedback}, run, admin=True)
        # A replay is the historical command outcome, even if its image is now
        # archived. Bind the route to that outcome without reading mutable
        # visibility first. Keep the payload digest compatible with old receipts.
        if dataset_id is not None and result["dataset_id"] != dataset_id:
            raise DomainError("image_dataset_mismatch", 409)
        return result

    def _approve(self, con: sqlite3.Connection, actor: Actor, ctx: MutationContext,
                 image: sqlite3.Row, feedback: str) -> dict[str, Any]:
        if image["state"] not in ("unassigned", "submitted"):
            raise DomainError("image_not_approvable", 409)
        self._require_complete(con, ctx.expected_version)
        review_id = self._review(con, actor, image, "approved", feedback)
        con.execute("UPDATE hosted_images SET confirmed_version=? WHERE id=?", (ctx.expected_version, ctx.image_id))
        self._handoff(con, ctx.image_id, "confirmed", assignee=None, assignment=None)
        con.execute("UPDATE hosted_datasets SET revision=revision+1 WHERE id=?", (image["dataset_id"],))
        self._event(con, actor, image["dataset_id"], ctx.image_id, "approved", {"review_id": review_id, "version_id": ctx.expected_version})
        return self._image_result(con, ctx.image_id)

    def _review(self, con: sqlite3.Connection, actor: Actor, image: sqlite3.Row, action: str, feedback: str) -> str:
        if not isinstance(feedback, str) or len(feedback) > 20000:
            raise DomainError("invalid_feedback", 422)
        review_id = _id("review")
        con.execute("INSERT INTO hosted_reviews VALUES (?,?,?,?,?,?,?,?)", (review_id, image["id"], image["active_submission"], image["current_version"], action, feedback, actor.user_id, self.clock()))
        return review_id

    def return_image(self, actor: Actor, ctx: MutationContext, feedback: str) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            image = self._guard(con, actor, ctx)
            if image["state"] != "submitted":
                raise DomainError("image_not_returnable", 409)
            self._annotator(con, image["assignee_id"])
            text = _text(feedback, "feedback", 20000)
            review_id = self._review(con, actor, image, "returned", text)
            self._handoff(con, ctx.image_id, "returned", assignee=image["assignee_id"], assignment=image["assignment_id"])
            self._event(con, actor, image["dataset_id"], ctx.image_id, "returned", {"review_id": review_id, "feedback": text})
            return self._image_result(con, ctx.image_id)
        return self._command(actor, ctx.operation_id, "return", {"context": asdict(ctx), "feedback": feedback}, run, admin=True)

    def override(self, actor: Actor, ctx: MutationContext, reason: str = "") -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            image = self._image(con, ctx.image_id)
            self._expected(image, ctx.expected_version, ctx.generation)
            if image["state"] not in ("assigned", "returned"):
                raise DomainError("image_not_overridable", 409)
            _text(ctx.editor_id, "editor_id", 200)
            self._handoff(con, ctx.image_id, "unassigned", assignee=None, assignment=None)
            self._event(con, actor, image["dataset_id"], ctx.image_id, "ownership_overridden", {"previous_assignee": image["assignee_id"], "reason": reason, "version_id": ctx.expected_version})
            return self._image_result(con, ctx.image_id)
        return self._command(actor, ctx.operation_id, "override", {"context": asdict(ctx), "reason": reason}, run, admin=True)

    def recall_images(self, actor: Actor, dataset_id: str, image_ids: list[str],
                      expected: dict[str, dict[str, Any]], reason: str,
                      operation_id: str) -> dict[str, Any]:
        """Recall one explicit selection atomically, without changing its drafts.

        A selection may span assignments and annotators. Unlike the historical
        assignment-remainder helper, no changed or submitted member is skipped:
        either every captured member still qualifies, or no ownership changes.
        """
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            self._dataset(con, dataset_id)
            explanation = _text(reason, "reason", 10000)
            if (not isinstance(image_ids, list) or not image_ids or len(image_ids) > 50000
                    or any(not isinstance(identifier, str) for identifier in image_ids)
                    or len(set(image_ids)) != len(image_ids) or not isinstance(expected, dict)
                    or set(expected) != set(image_ids)):
                raise DomainError("exact_assignment_preconditions_required", 422)
            captured = []
            for image_id in image_ids:
                image = self._image(con, image_id)
                if (image["dataset_id"] != dataset_id or image["state"] not in ("assigned", "returned")
                        or not image["assignee_id"] or not image["assignment_id"]):
                    raise DomainError("image_not_overridable", 409, {"image_id": image_id})
                pre = expected[image_id]
                if not isinstance(pre, dict):
                    raise DomainError("exact_assignment_preconditions_required", 422)
                self._expected(image, pre.get("expected_version"), pre.get("generation"))
                captured.append(image)
            for image in captured:
                self._handoff(con, image["id"], "unassigned", assignee=None, assignment=None)
                self._event(con, actor, dataset_id, image["id"], "ownership_overridden", {
                    "assignment_id": image["assignment_id"], "previous_assignee": image["assignee_id"],
                    "reason": explanation, "version_id": image["current_version"], "operation_id": operation_id})
            return {"dataset_id": dataset_id, "images": [self._image_result(con, image["id"]) for image in captured]}
        return self._command(actor, operation_id, "recall_images", dict(dataset_id=dataset_id,
            image_ids=image_ids, expected=expected, reason=reason), run, admin=True)

    def reassign(self, actor: Actor, ctx: MutationContext, annotator_id: str) -> dict[str, Any]:
        """Assign an explicitly recovered image; never silently override its owner."""
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            image = self._guard(con, actor, ctx)
            if image["state"] != "unassigned":
                raise DomainError("explicit_override_required", 409)
            self._annotator(con, annotator_id)
            assignment_id = _id("assignment")
            con.execute("INSERT INTO hosted_assignments VALUES (?,?,?,?,?)", (assignment_id, image["dataset_id"], annotator_id, actor.user_id, self.clock()))
            con.execute("INSERT INTO hosted_assignment_members VALUES (?,?,?,'assigned')", (assignment_id, ctx.image_id, ctx.expected_version))
            self._handoff(con, ctx.image_id, "assigned", assignee=annotator_id, assignment=assignment_id)
            self._event(con, actor, image["dataset_id"], ctx.image_id, "reassigned", {"assignment_id": assignment_id, "annotator_id": annotator_id})
            return self._image_result(con, ctx.image_id)
        return self._command(actor, ctx.operation_id, "reassign", {"context": asdict(ctx), "annotator_id": annotator_id}, run, admin=True)

    def override_assignment(self, actor: Actor, assignment_id: str, operation_id: str,
                            expected: dict[str, dict[str, Any]], reason: str = "") -> dict[str, Any]:
        """Recover the captured remaining members; a concurrent submission stays in review.

        Members not explicitly captured never join the command later. Any stale
        editable member rejects the transaction rather than causing a takeover.
        """
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            assignment = con.execute("SELECT * FROM hosted_assignments WHERE id=?", (assignment_id,)).fetchone()
            if not assignment:
                raise DomainError("assignment_not_found", 404)
            if not expected:
                raise DomainError("exact_assignment_preconditions_required", 422)
            members = {row[0] for row in con.execute("SELECT image_id FROM hosted_assignment_members WHERE assignment_id=?", (assignment_id,))}
            if not set(expected).issubset(members):
                raise DomainError("assignment_member_mismatch", 409)
            recovered, skipped = [], []
            for image_id, pre in expected.items():
                image = self._image(con, image_id)
                if image["assignment_id"] != assignment_id or image["state"] not in ("assigned", "returned"):
                    skipped.append({"image_id": image_id, "reason": "no_longer_annotator_owned"})
                    continue
                self._expected(image, pre.get("expected_version", pre.get("version_id")), pre.get("generation"))
                self._handoff(con, image_id, "unassigned", assignee=None, assignment=None)
                self._event(con, actor, assignment["dataset_id"], image_id, "ownership_overridden", {
                    "assignment_id": assignment_id, "previous_assignee": image["assignee_id"], "reason": reason,
                    "version_id": image["current_version"]})
                recovered.append(self._image_result(con, image_id))
            return {"assignment_id": assignment_id, "images": recovered, "skipped": skipped}
        return self._command(actor, operation_id, "override_assignment", dict(assignment_id=assignment_id,expected=expected,reason=reason), run, admin=True)

    def bulk_approve(self, actor: Actor, contexts: list[MutationContext], feedback: str = "") -> list[dict[str, Any]]:
        """Independent exact-version commits with a durable receipt for each image."""
        if (not contexts or len({ctx.image_id for ctx in contexts}) != len(contexts)
                or len({ctx.operation_id for ctx in contexts}) != len(contexts)):
            raise DomainError("invalid_bulk_targets", 422)
        results = []
        for ctx in contexts:
            try:
                results.append({"image_id": ctx.image_id, "status": "committed", "result": self.approve(actor, ctx, feedback)})
            except DomainError as exc:
                results.append({"image_id": ctx.image_id, "status": "rejected", "error": exc.as_dict(), "http_status": exc.status})
        return results

    def own_submission_status(self, actor: Actor, dataset_id: str | None = None, *,
                              limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Receipt-like progress after handoff, without access to pixels or labels."""
        self._pagination(limit, offset)
        with self.read_connection() as con:
            self._actor(con, actor)
            query = ("SELECT s.id AS submission_id,s.image_id,s.version_id,s.created_at,i.dataset_id,i.filename,d.name AS dataset_name "
                     "FROM hosted_submissions s JOIN hosted_images i ON i.id=s.image_id "
                     "JOIN hosted_datasets d ON d.id=i.dataset_id WHERE s.actor_id=?")
            args = [actor.user_id]
            if dataset_id:
                query += " AND i.dataset_id=?"
                args.append(dataset_id)
            result = []
            for row in con.execute(query + " ORDER BY s.rowid DESC LIMIT ? OFFSET ?", [*args, limit, offset]):
                value = dict(row)
                review = con.execute("SELECT action,feedback,created_at FROM hosted_reviews WHERE submission_id=? ORDER BY rowid DESC LIMIT 1", (row["submission_id"],)).fetchone()
                value.update({"status": review["action"] if review else "awaiting_review",
                              "feedback": review["feedback"] if review else "",
                              "reviewed_at": review["created_at"] if review else None})
                archived = con.execute("SELECT a.reason,a.archived_at FROM hosted_archived_images i JOIN hosted_archives a ON a.id=i.archive_id WHERE i.image_id=?", (row["image_id"],)).fetchone()
                value["archived"] = archived is not None
                if archived:
                    value.update({"status": "archived", "archive_reason": archived["reason"], "archived_at": archived["archived_at"]})
                result.append(value)
            return result

    def history(self, actor: Actor, image_id: str) -> dict[str, Any]:
        with self.read_connection() as con:
            self._actor(con, actor)
            self._can_view(actor, self._image(con, image_id))
            result = {"image_id": image_id}
            for key, table in (("versions", "hosted_annotation_versions"), ("submissions", "hosted_submissions"),
                               ("reviews", "hosted_reviews"), ("events", "hosted_events"), ("import_conflicts", "hosted_import_conflicts"),
                               ("import_conflict_decisions", "hosted_import_conflict_decisions")):
                values = []
                for row in con.execute(f"SELECT * FROM {table} WHERE image_id=? ORDER BY rowid", (image_id,)):
                    value = dict(row)
                    for field in ("annotations_json", "data_json"):
                        if field in value:
                            value[field.removesuffix("_json")] = json.loads(value.pop(field))
                    values.append(value)
                result[key] = values
            return result

    def add_class(self, actor: Actor, dataset_id: str, name: str, operation_id: str,
                  expected_class_version: int) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            if self._dataset(con, dataset_id)["class_version"] != expected_class_version:
                raise DomainError("class_version_changed", 409)
            item = self._add_class(con, actor, dataset_id, name)
            version = self._class_revision(con, actor, dataset_id)
            self._event(con, actor, dataset_id, None, "class_added", item)
            return {"class": item, "class_version": version}
        return self._command(actor, operation_id, "add_class", {"dataset_id": dataset_id, "name": name, "expected_class_version": expected_class_version}, run, admin=True)

    def rename_class(self, actor: Actor, dataset_id: str, class_id: str, name: str,
                     operation_id: str, expected_class_version: int) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            if self._dataset(con, dataset_id)["class_version"] != expected_class_version:
                raise DomainError("class_version_changed", 409)
            row = con.execute("SELECT * FROM hosted_classes WHERE id=? AND dataset_id=?", (class_id, dataset_id)).fetchone()
            if not row:
                raise DomainError("class_not_found", 404)
            value = _text(name, "class_name", 200)
            if "\n" in value or "\r" in value:
                raise DomainError("invalid_class_name", 422)
            if con.execute("SELECT 1 FROM hosted_classes WHERE dataset_id=? AND name_key=? AND id!=?", (dataset_id, value.casefold(), class_id)).fetchone():
                raise DomainError("class_name_exists", 409)
            con.execute("UPDATE hosted_classes SET name=?,name_key=? WHERE id=?", (value, value.casefold(), class_id))
            version = self._class_revision(con, actor, dataset_id)
            self._event(con, actor, dataset_id, None, "class_renamed", {"class_id": class_id, "previous_name": row["name"], "name": value})
            return {"class_id": class_id, "name": value, "class_version": version}
        return self._command(actor, operation_id, "rename_class", dict(dataset_id=dataset_id,class_id=class_id,name=name,expected_class_version=expected_class_version), run, admin=True)

    def request_class(self, actor: Actor, dataset_id: str, name: str, reason: str,
                      operation_id: str) -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            if not isinstance(name, str) or "\n" in name or "\r" in name:
                raise DomainError("invalid_class_name", 422)
            self._dataset(con, dataset_id)
            if actor.role != "admin" and not con.execute("SELECT 1 FROM hosted_images WHERE dataset_id=? AND assignee_id=? AND state IN ('assigned','returned') LIMIT 1", (dataset_id, actor.user_id)).fetchone():
                raise DomainError("dataset_not_found", 404)
            request_id = _id("classrequest")
            con.execute("INSERT INTO hosted_class_requests(id,dataset_id,actor_id,name,reason,created_at) VALUES (?,?,?,?,?,?)",
                        (request_id, dataset_id, actor.user_id, _text(name,"class_name",200), _text(reason,"reason",5000), self.clock()))
            self._event(con, actor, dataset_id, None, "class_requested", {"request_id": request_id})
            return dict(con.execute("SELECT * FROM hosted_class_requests WHERE id=?", (request_id,)).fetchone())
        return self._command(actor, operation_id, "request_class", dict(dataset_id=dataset_id,name=name,reason=reason), run)

    def list_class_requests(self, actor: Actor, dataset_id: str | None = None) -> list[dict[str, Any]]:
        with self.read_connection() as con:
            self._actor(con, actor)
            where, args = [], []
            if actor.role != "admin":
                where.append("actor_id=?")
                args.append(actor.user_id)
            if dataset_id:
                where.append("dataset_id=?")
                args.append(dataset_id)
            return [dict(row) for row in con.execute("SELECT * FROM hosted_class_requests" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at,id", args)]

    def resolve_class_request(self, actor: Actor, request_id: str, approved: bool, operation_id: str,
                              expected_class_version: int, feedback: str = "") -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            row = con.execute("SELECT * FROM hosted_class_requests WHERE id=?", (request_id,)).fetchone()
            if not row:
                raise DomainError("class_request_not_found", 404)
            if row["state"] != "pending":
                raise DomainError("class_request_already_reviewed", 409)
            if type(approved) is not bool:
                raise DomainError("invalid_review", 422)
            if self._dataset(con, row["dataset_id"])["class_version"] != expected_class_version:
                raise DomainError("class_version_changed", 409)
            class_id = None
            if approved:
                existing = con.execute("SELECT id FROM hosted_classes WHERE dataset_id=? AND name_key=? AND retired=0", (row["dataset_id"], row["name"].casefold())).fetchone()
                if existing:
                    class_id = existing[0]
                else:
                    class_id = self._add_class(con, actor, row["dataset_id"], row["name"])["id"]
                    self._class_revision(con, actor, row["dataset_id"])
            con.execute("UPDATE hosted_class_requests SET state=?,class_id=?,reviewed_by=?,feedback=?,reviewed_at=? WHERE id=?",
                        ("approved" if approved else "rejected", class_id, actor.user_id, feedback, self.clock(), request_id))
            self._event(con, actor, row["dataset_id"], None, "class_request_reviewed", {"request_id": request_id, "approved": approved, "class_id": class_id})
            return dict(con.execute("SELECT * FROM hosted_class_requests WHERE id=?", (request_id,)).fetchone())
        return self._command(actor, operation_id, "resolve_class_request", dict(request_id=request_id,approved=approved,expected_class_version=expected_class_version,feedback=feedback), run, admin=True)

    def _snapshot(self, con: sqlite3.Connection, dataset_id: str) -> dict[str, Any]:
        dataset = self._dataset(con, dataset_id)
        rows = con.execute("SELECT i.id,i.media_hash,i.filename,i.split,i.confirmed_version AS version_id,v.annotations_json,v.complete,v.negative,v.class_version,m.width,m.height,m.size_bytes "
                           "FROM hosted_images i JOIN hosted_annotation_versions v ON v.id=i.confirmed_version JOIN hosted_media m ON m.sha256=i.media_hash "
                           "WHERE i.dataset_id=? AND i.state='confirmed' AND i.id NOT IN (SELECT image_id FROM hosted_archived_images) ORDER BY i.id", (dataset_id,))
        images = []
        for row in rows:
            image = dict(row)
            image["annotations"] = json.loads(image.pop("annotations_json"))
            images.append(image)
        settings = con.execute("SELECT version,settings_json FROM hosted_dataset_settings WHERE dataset_id=?", (dataset_id,)).fetchone()
        return {"schema": "hosted-canonical-snapshot-v1", "dataset_id": dataset_id, "dataset_name": dataset["name"],
                "revision": dataset["revision"], "class_version": dataset["class_version"],
                "classes": self._classes(con, dataset_id), "images": images,
                "settings": json.loads(settings["settings_json"]) if settings else {},
                "settings_version": settings["version"] if settings else 0}

    def snapshot(self, actor: Actor, dataset_id: str) -> dict[str, Any]:
        with self.read_connection() as con:
            self._actor(con, actor, admin=True)
            return self._snapshot(con, dataset_id)

    def create_snapshot(self, actor: Actor, dataset_id: str, operation_id: str, name: str = "") -> dict[str, Any]:
        def run(con: sqlite3.Connection) -> dict[str, Any]:
            manifest = self._snapshot(con, dataset_id)
            snapshot_id = _id("snapshot")
            con.execute("INSERT INTO hosted_snapshots VALUES (?,?,?,?,?,?,?)", (snapshot_id, dataset_id, manifest["revision"], name, _json(manifest), actor.user_id, self.clock()))
            return {"snapshot_id": snapshot_id, "manifest": manifest}
        return self._command(actor, operation_id, "create_snapshot", dict(dataset_id=dataset_id,name=name), run, admin=True)

    def get_snapshot(self, actor: Actor, snapshot_id: str) -> dict[str, Any]:
        with self.read_connection() as con:
            self._actor(con, actor, admin=True)
            row = con.execute("SELECT * FROM hosted_snapshots WHERE id=?", (snapshot_id,)).fetchone()
            if not row:
                raise DomainError("snapshot_not_found", 404)
            return {"snapshot_id": snapshot_id, "name": row["name"], "manifest": json.loads(row["manifest_json"])}

    def media_references(self, connection: sqlite3.Connection | None = None) -> list[str]:
        if connection is not None:
            return [row[0] for row in connection.execute("SELECT sha256 FROM hosted_media ORDER BY sha256")]
        with self.read_connection() as con:
            return self.media_references(con)
