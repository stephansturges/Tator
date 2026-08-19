"""Small durable store for non-exported annotation entity metadata.

YOLO label files intentionally remain portable and unchanged.  Stable IDs,
record revisions, and idempotency receipts therefore live in this sidecar store
and are keyed by the annotation source identity plus image key.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Any, Iterator, Mapping


_TRANSACTION_TRANSPORT_KEYS = frozenset(
    {
        "annotation_save",
        "lock_session_id",
        "lock_token",
        "session_id",
        "_training_authorization",
        "training_authorization",
    }
)

_TERMINAL_OPERATION_STATUSES = frozenset(
    {"complete", "conflict", "rejected", "superseded", "legacy_unrecoverable"}
)

_OPERATION_PHASE_BY_STATE = {
    "prepared": "prepared",
    "annotation_committed": "annotation_committed",
    "entity_committed": "entity_committed",
    "review_committed": "review_committed",
    "recovery_required": "entity_committed",
    "complete": "review_committed",
    "conflict": "prepared",
    "rejected": "prepared",
    "superseded": "prepared",
}

_OPERATION_STATUS_BY_STATE = {
    "prepared": "active",
    "annotation_committed": "active",
    "entity_committed": "active",
    "review_committed": "active",
    "recovery_required": "recovery_required",
    "complete": "complete",
    "conflict": "conflict",
    "rejected": "rejected",
    "superseded": "superseded",
}


def _strip_transaction_transport_credentials(value: Any, *, root: bool = True) -> Any:
    """Remove replayable transport credentials from durable journal payloads."""
    if isinstance(value, Mapping):
        return {
            str(key): _strip_transaction_transport_credentials(item, root=False)
            for key, item in value.items()
            if str(key)
            not in (
                _TRANSACTION_TRANSPORT_KEYS
                if root
                else {
                    "annotation_save",
                    "_training_authorization",
                    "training_authorization",
                }
            )
        }
    if isinstance(value, list):
        return [
            _strip_transaction_transport_credentials(item, root=False)
            for item in value
        ]
    return value


def _semantic_json(value: Mapping[str, Any] | None) -> str:
    return json.dumps(
        _strip_transaction_transport_credentials(dict(value or {})),
        sort_keys=True,
        separators=(",", ":"),
    )


def default_annotation_entity_store_path() -> Path:
    configured = str(os.environ.get("TATOR_ANNOTATION_ENTITY_STORE_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser()
    configured_state = str(os.environ.get("TATOR_STATE_DIR") or "").strip()
    if configured_state:
        state_root = Path(configured_state).expanduser()
    elif sys.platform == "darwin":
        state_root = Path.home() / "Library" / "Application Support" / "Tator"
    else:
        state_root = Path.home() / ".local" / "share" / "tator"
    return state_root / "annotation_entities.sqlite3"


class AnnotationEntityStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or default_annotation_entity_store_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialise(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS annotation_entity_records (
                    source_identity TEXT NOT NULL,
                    image_key TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (source_identity, image_key)
                );
            CREATE TABLE IF NOT EXISTS annotation_entity_transactions (
                operation_id TEXT PRIMARY KEY,
                request_hash TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                job_id TEXT NOT NULL DEFAULT '',
                source_identity TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL DEFAULT 'complete',
                request_json TEXT NOT NULL DEFAULT '{}',
                updated_at REAL NOT NULL DEFAULT 0,
                error_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS analysis_point_entities (
                job_id TEXT NOT NULL,
                point_id TEXT NOT NULL,
                source_identity TEXT NOT NULL,
                image_key TEXT NOT NULL,
                annotation_entity_id TEXT NOT NULL,
                entity_revision INTEGER NOT NULL,
                record_revision TEXT NOT NULL,
                status TEXT NOT NULL,
                attestation TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(job_id, point_id)
            );
            CREATE TABLE IF NOT EXISTS annotation_job_sources (
                job_id TEXT NOT NULL,
                source_identity TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY(job_id, source_identity)
            );
                CREATE TABLE IF NOT EXISTS annotation_entity_conflicts (
                    source_identity TEXT NOT NULL,
                    image_key TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (source_identity, image_key)
                );
                CREATE TABLE IF NOT EXISTS annotation_source_analysis_bindings (
                    source_mode TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    binding_generation INTEGER NOT NULL DEFAULT 1,
                    binding_state TEXT NOT NULL DEFAULT 'active',
                    source_descriptor_json TEXT NOT NULL DEFAULT '{}',
                    checkpoint_revision TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (source_mode, source_id)
                );
                CREATE TABLE IF NOT EXISTS annotation_operation_batches (
                    batch_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    source_mode TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    binding_generation INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    target_class_name TEXT NOT NULL DEFAULT '',
                    declared_count INTEGER NOT NULL,
                    manifest_hash TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'draft',
                    active_operation_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    error_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS annotation_operation_batch_items (
                    batch_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    point_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    state TEXT NOT NULL DEFAULT 'pending',
                    operation_id TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (batch_id, sequence),
                    FOREIGN KEY (batch_id) REFERENCES annotation_operation_batches(batch_id)
                        ON DELETE CASCADE
                );
                """
        )
            transaction_columns = {
                str(row[1]) for row in connection.execute(
                    "PRAGMA table_info(annotation_entity_transactions)"
                ).fetchall()
            }
            for column, definition in (
                ("job_id", "TEXT NOT NULL DEFAULT ''"),
                ("source_identity", "TEXT NOT NULL DEFAULT ''"),
                ("state", "TEXT NOT NULL DEFAULT 'complete'"),
                ("request_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("updated_at", "REAL NOT NULL DEFAULT 0"),
                ("error_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("source_mode", "TEXT NOT NULL DEFAULT ''"),
                ("source_id", "TEXT NOT NULL DEFAULT ''"),
                ("binding_generation", "INTEGER NOT NULL DEFAULT 0"),
                ("transaction_kind", "TEXT NOT NULL DEFAULT ''"),
                ("phase", "TEXT NOT NULL DEFAULT 'prepared'"),
                ("status", "TEXT NOT NULL DEFAULT 'active'"),
                ("retryable", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if column not in transaction_columns:
                    connection.execute(
                        f"ALTER TABLE annotation_entity_transactions ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                "UPDATE annotation_entity_transactions SET state = 'complete' "
                "WHERE state IS NULL OR state = ''"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_annotation_entity_transactions_scope "
                "ON annotation_entity_transactions(job_id, source_identity, state)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_annotation_job_sources_source "
                "ON annotation_job_sources(source_identity, job_id)"
            )
            binding_columns = {
                str(row[1]) for row in connection.execute(
                    "PRAGMA table_info(annotation_source_analysis_bindings)"
                ).fetchall()
            }
            for column, definition in (
                ("binding_generation", "INTEGER NOT NULL DEFAULT 1"),
                ("binding_state", "TEXT NOT NULL DEFAULT 'active'"),
                ("source_descriptor_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("checkpoint_revision", "TEXT NOT NULL DEFAULT ''"),
                ("created_at", "REAL NOT NULL DEFAULT 0"),
            ):
                if column not in binding_columns:
                    connection.execute(
                        f"ALTER TABLE annotation_source_analysis_bindings ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                "UPDATE annotation_source_analysis_bindings SET "
                "binding_generation = MAX(1, binding_generation), "
                "created_at = CASE WHEN created_at > 0 THEN created_at ELSE updated_at END"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_annotation_transactions_binding "
                "ON annotation_entity_transactions(source_mode, source_id, "
                "binding_generation, status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_annotation_batches_binding "
                "ON annotation_operation_batches(source_mode, source_id, "
                "binding_generation, state)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_annotation_batch_items_state "
                "ON annotation_operation_batch_items(batch_id, state, sequence)"
            )
            connection.execute(
                "UPDATE annotation_entity_transactions SET "
                "status = 'legacy_unrecoverable', retryable = 0 "
                "WHERE binding_generation <= 0 AND state NOT IN ('complete', 'rejected')"
            )
            batch_item_columns = {
                str(row[1]) for row in connection.execute(
                    "PRAGMA table_info(annotation_operation_batch_items)"
                ).fetchall()
            }
            if "payload_json" not in batch_item_columns:
                connection.execute(
                    "ALTER TABLE annotation_operation_batch_items "
                    "ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'"
                )
            for row in connection.execute(
                "SELECT operation_id, request_hash, request_json, response_json, error_json, state, "
                "source_mode, source_id, binding_generation "
                "FROM annotation_entity_transactions"
            ).fetchall():
                try:
                    request = json.loads(str(row["request_json"] or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    request = {}
                try:
                    response = json.loads(str(row["response_json"] or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    response = {}
                try:
                    error = json.loads(str(row["error_json"] or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    error = {}
                scrubbed = _strip_transaction_transport_credentials(request)
                scrubbed_response = _strip_transaction_transport_credentials(response)
                scrubbed_error = _strip_transaction_transport_credentials(error)
                source = scrubbed.get("annotation_source")
                if not isinstance(source, Mapping):
                    source = scrubbed.get("annotation_target")
                source = source if isinstance(source, Mapping) else {}
                source_mode = str(row["source_mode"] or source.get("kind") or source.get("mode") or "")
                source_id = str(row["source_id"] or source.get("id") or source.get("source_id") or source.get("session_id") or "")
                transaction_kind = str(scrubbed.get("transaction_kind") or "")
                state = str(row["state"] or "complete")
                phase = _OPERATION_PHASE_BY_STATE.get(state, "prepared")
                status = _OPERATION_STATUS_BY_STATE.get(state, "legacy_unrecoverable")
                generation = int(row["binding_generation"] or 0)
                if source_mode and source_id and generation <= 0:
                    binding = connection.execute(
                        "SELECT job_id, binding_generation FROM annotation_source_analysis_bindings "
                        "WHERE source_mode = ? AND source_id = ?",
                        (source_mode, source_id),
                    ).fetchone()
                    if binding is not None and str(binding["job_id"]) == str(scrubbed.get("job_id") or ""):
                        generation = int(binding["binding_generation"] or 1)
                if state != "complete" and (
                    not source_mode or not source_id or generation <= 0
                ):
                    status = "legacy_unrecoverable"
                    state = "rejected"
                hash_payload = dict(scrubbed)
                hash_payload.pop("transaction_kind", None)
                migrated_hash = (
                    str(row["request_hash"])
                    if state == "complete" and not hash_payload
                    else self.request_hash(hash_payload)
                )
                connection.execute(
                    "UPDATE annotation_entity_transactions SET request_json = ?, "
                    "response_json = ?, error_json = ?, request_hash = ?, "
                    "source_mode = ?, source_id = ?, binding_generation = ?, "
                    "transaction_kind = ?, state = ?, phase = ?, status = ?, retryable = ? "
                    "WHERE operation_id = ?",
                    (
                        _semantic_json(scrubbed),
                        _semantic_json(scrubbed_response),
                        _semantic_json(scrubbed_error),
                        migrated_hash,
                        source_mode,
                        source_id,
                        generation,
                        transaction_kind,
                        state,
                        phase,
                        status,
                        1 if status == "recovery_required" else 0,
                        str(row["operation_id"]),
                    ),
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    @staticmethod
    def request_hash(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            _strip_transaction_transport_credentials(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get_record(
        self,
        source_identity: str,
        image_key: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any] | None:
        owns_connection = connection is None
        current = connection or self._connect()
        try:
            row = current.execute(
                "SELECT record_json FROM annotation_entity_records "
                "WHERE source_identity = ? AND image_key = ?",
                (str(source_identity), str(image_key)),
            ).fetchone()
            return json.loads(str(row["record_json"])) if row is not None else None
        finally:
            if owns_connection:
                current.close()

    def put_record(
        self,
        source_identity: str,
        image_key: str,
        record: Mapping[str, Any],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        owns_connection = connection is None
        current = connection or self._connect()
        try:
            current.execute(
                "INSERT INTO annotation_entity_records "
                "(source_identity, image_key, record_json, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(source_identity, image_key) DO UPDATE SET "
                "record_json = excluded.record_json, updated_at = excluded.updated_at",
                (
                    str(source_identity),
                    str(image_key),
                    json.dumps(dict(record), sort_keys=True, separators=(",", ":")),
                    time.time(),
                ),
            )
            current.execute(
                "DELETE FROM annotation_entity_conflicts "
                "WHERE source_identity = ? AND image_key = ?",
                (str(source_identity), str(image_key)),
            )
            if owns_connection:
                current.commit()
        finally:
            if owns_connection:
                current.close()

    def mark_conflict(self, source_identity: str, image_key: str, detail: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO annotation_entity_conflicts "
                "(source_identity, image_key, detail, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(source_identity, image_key) DO UPDATE SET "
                "detail = excluded.detail, updated_at = excluded.updated_at",
                (str(source_identity), str(image_key), str(detail), time.time()),
            )

    def list_conflicts(self, source_identity: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if source_identity:
                rows = connection.execute(
                    "SELECT source_identity, image_key, detail, updated_at "
                    "FROM annotation_entity_conflicts WHERE source_identity = ? "
                    "ORDER BY image_key",
                    (str(source_identity),),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT source_identity, image_key, detail, updated_at "
                    "FROM annotation_entity_conflicts ORDER BY source_identity, image_key"
                ).fetchall()
        return [dict(row) for row in rows]

    def bind_job_sources(
        self, job_id: str, source_identities: Iterator[str] | list[str] | set[str]
    ) -> None:
        safe_job_id = str(job_id or "").strip()
        identities = sorted(
            {
                str(source_identity or "").strip()
                for source_identity in source_identities
                if str(source_identity or "").strip()
            }
        )
        if not safe_job_id or not identities:
            return
        now = time.time()
        with self.transaction() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO annotation_job_sources "
                "(job_id, source_identity, created_at) VALUES (?, ?, ?)",
                [(safe_job_id, identity, now) for identity in identities],
            )

    def list_job_sources(self, job_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source_identity FROM annotation_job_sources "
                "WHERE job_id = ? ORDER BY source_identity",
                (str(job_id or "").strip(),),
            ).fetchall()
        return [str(row["source_identity"]) for row in rows]

    def get_source_job_binding(self, source_mode: str, source_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * "
                "FROM annotation_source_analysis_bindings "
                "WHERE source_mode = ? AND source_id = ?",
                (str(source_mode), str(source_id)),
            ).fetchone()
        return dict(row) if row is not None else None

    def claim_source_job(
        self,
        source_mode: str,
        source_id: str,
        job_id: str,
        *,
        source_descriptor: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically claim a source generation without bypassing old work."""

        now = time.time()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM annotation_source_analysis_bindings "
                "WHERE source_mode = ? AND source_id = ?",
                (str(source_mode), str(source_id)),
            ).fetchone()
            if existing is not None and str(existing["job_id"]) == str(job_id):
                return dict(existing)
            generation = 1
            if existing is not None:
                old_job_id = str(existing["job_id"] or "")
                blockers = connection.execute(
                    "SELECT operation_id FROM annotation_entity_transactions "
                    "WHERE job_id = ? AND status NOT IN "
                    "('complete', 'conflict', 'rejected', 'superseded', 'legacy_unrecoverable') "
                    "LIMIT 1",
                    (old_job_id,),
                ).fetchone()
                active_batch = connection.execute(
                    "SELECT batch_id FROM annotation_operation_batches "
                    "WHERE source_mode = ? AND source_id = ? AND binding_generation = ? "
                    "AND state NOT IN ('complete', 'partial', 'cancelled', 'conflict') LIMIT 1",
                    (
                        str(source_mode),
                        str(source_id),
                        int(existing["binding_generation"] or 1),
                    ),
                ).fetchone()
                if blockers is not None or active_batch is not None:
                    raise ValueError(f"annotation_source_has_uncheckpointed_analysis:{old_job_id}")
                generation = int(existing["binding_generation"] or 1) + 1
                connection.execute(
                    "UPDATE annotation_entity_transactions SET state = 'superseded', "
                    "status = 'superseded', retryable = 0, updated_at = ? "
                    "WHERE job_id = ? AND status NOT IN ('complete', 'conflict', 'rejected', "
                    "'superseded', 'legacy_unrecoverable')",
                    (now, old_job_id),
                )
            connection.execute(
                "INSERT INTO annotation_source_analysis_bindings "
                "(source_mode, source_id, job_id, binding_generation, binding_state, "
                "source_descriptor_json, checkpoint_revision, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'active', ?, '', ?, ?) "
                "ON CONFLICT(source_mode, source_id) DO UPDATE SET "
                "job_id = excluded.job_id, binding_generation = excluded.binding_generation, "
                "binding_state = 'active', source_descriptor_json = excluded.source_descriptor_json, "
                "checkpoint_revision = '', updated_at = excluded.updated_at",
                (
                    str(source_mode),
                    str(source_id),
                    str(job_id),
                    generation,
                    _semantic_json(source_descriptor),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM annotation_source_analysis_bindings "
                "WHERE source_mode = ? AND source_id = ?",
                (str(source_mode), str(source_id)),
            ).fetchone()
            return dict(row)

    def bind_source_job(self, source_mode: str, source_id: str, job_id: str) -> dict[str, Any]:
        return self.claim_source_job(source_mode, source_id, job_id)

    def get_job_source_binding(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM annotation_source_analysis_bindings WHERE job_id = ? "
                "ORDER BY updated_at DESC LIMIT 1",
                (str(job_id),),
            ).fetchone()
        return dict(row) if row is not None else None

    def assert_source_generation(
        self,
        source_mode: str,
        source_id: str,
        job_id: str,
        binding_generation: int,
    ) -> dict[str, Any]:
        binding = self.get_source_job_binding(source_mode, source_id)
        if (
            binding is None
            or str(binding.get("job_id") or "") != str(job_id)
            or int(binding.get("binding_generation") or 0) != int(binding_generation)
        ):
            raise ValueError("annotation_source_binding_superseded")
        return binding

    def unbind_source_job(self, source_mode: str, source_id: str, job_id: str) -> bool:
        with self.transaction() as connection:
            blockers = connection.execute(
                "SELECT operation_id FROM annotation_entity_transactions WHERE job_id = ? "
                "AND status NOT IN ('complete', 'conflict', 'rejected', 'superseded', "
                "'legacy_unrecoverable') LIMIT 1",
                (str(job_id),),
            ).fetchone()
            if blockers is not None:
                raise ValueError("annotation_source_has_uncheckpointed_analysis")
            cursor = connection.execute(
                "DELETE FROM annotation_source_analysis_bindings "
                "WHERE source_mode = ? AND source_id = ? AND job_id = ?",
                (str(source_mode), str(source_id), str(job_id)),
            )
            return bool(cursor.rowcount)

    def get_receipt(
        self,
        operation_id: str,
        request_hash: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any] | None:
        owns_connection = connection is None
        current = connection or self._connect()
        try:
            row = current.execute(
                "SELECT request_hash, response_json, state FROM annotation_entity_transactions "
                "WHERE operation_id = ?",
                (str(operation_id),),
            ).fetchone()
            if row is None:
                return None
            if str(row["request_hash"]) != str(request_hash):
                raise ValueError("annotation_transaction_id_reused_with_different_payload")
            if str(row["state"] or "complete") != "complete":
                return None
            return json.loads(str(row["response_json"]))
        finally:
            if owns_connection:
                current.close()

    def put_receipt(
        self,
        operation_id: str,
        request_hash: str,
        response: Mapping[str, Any],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        owns_connection = connection is None
        current = connection or self._connect()
        try:
            now = time.time()
            existing = current.execute(
                "SELECT request_hash FROM annotation_entity_transactions "
                "WHERE operation_id = ?",
                (str(operation_id),),
            ).fetchone()
            if existing is not None and str(existing["request_hash"]) != str(request_hash):
                raise ValueError("annotation_transaction_id_reused_with_different_payload")
            current.execute(
                "INSERT INTO annotation_entity_transactions "
                "(operation_id, request_hash, response_json, created_at, state, updated_at) "
                "VALUES (?, ?, ?, ?, 'complete', ?) "
                "ON CONFLICT(operation_id) DO UPDATE SET "
                "request_hash = excluded.request_hash, response_json = excluded.response_json, "
                "state = 'complete', updated_at = excluded.updated_at, error_json = '{}'",
                (
                    str(operation_id),
                    str(request_hash),
                    _semantic_json(response),
                    now,
                    now,
                ),
            )
            if owns_connection:
                current.commit()
        finally:
            if owns_connection:
                current.close()

    @staticmethod
    def _operation_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "operation_id": str(row["operation_id"]),
            "request_hash": str(row["request_hash"]),
            "job_id": str(row["job_id"] or ""),
            "source_identity": str(row["source_identity"] or ""),
            "state": str(row["state"] or "complete"),
            "source_mode": str(row["source_mode"] or ""),
            "source_id": str(row["source_id"] or ""),
            "binding_generation": int(row["binding_generation"] or 0),
            "transaction_kind": str(row["transaction_kind"] or ""),
            "phase": str(row["phase"] or "prepared"),
            "status": str(row["status"] or "active"),
            "retryable": bool(row["retryable"]),
            "request": json.loads(str(row["request_json"] or "{}")),
            "response": json.loads(str(row["response_json"] or "{}")),
            "error": json.loads(str(row["error_json"] or "{}")),
            "created_at": float(row["created_at"] or 0),
            "updated_at": float(row["updated_at"] or 0),
        }

    def begin_operation(
        self,
        *,
        operation_id: str,
        request_hash: str,
        request: Mapping[str, Any],
        job_id: str,
        source_identity: str,
        source_mode: str = "",
        source_id: str = "",
        binding_generation: int = 0,
        transaction_kind: str = "",
        response: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self.transaction() as connection:
            if source_mode or source_id or binding_generation:
                if not source_mode or not source_id or int(binding_generation) < 1:
                    raise ValueError("annotation_source_generation_required")
                binding = connection.execute(
                    "SELECT job_id, binding_generation FROM annotation_source_analysis_bindings "
                    "WHERE source_mode = ? AND source_id = ?",
                    (str(source_mode), str(source_id)),
                ).fetchone()
                if (
                    binding is None
                    or str(binding["job_id"] or "") != str(job_id)
                    or int(binding["binding_generation"] or 0)
                    != int(binding_generation)
                ):
                    raise ValueError("annotation_source_generation_stale")
            row = connection.execute(
                "SELECT * FROM annotation_entity_transactions WHERE operation_id = ?",
                (str(operation_id),),
            ).fetchone()
            if row is not None:
                if str(row["request_hash"]) != str(request_hash):
                    raise ValueError("annotation_transaction_id_reused_with_different_payload")
                return self._operation_from_row(row)
            connection.execute(
                "INSERT INTO annotation_entity_transactions "
                "(operation_id, request_hash, response_json, created_at, job_id, "
                "source_identity, state, request_json, updated_at, error_json, "
                "source_mode, source_id, binding_generation, transaction_kind, phase, "
                "status, retryable) VALUES (?, ?, ?, ?, ?, ?, 'prepared', ?, ?, '{}', "
                "?, ?, ?, ?, 'prepared', 'active', 0)",
                (
                    str(operation_id),
                    str(request_hash),
                    _semantic_json(response),
                    now,
                    str(job_id),
                    str(source_identity),
                    json.dumps(
                        _strip_transaction_transport_credentials(dict(request)),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    now,
                    str(source_mode),
                    str(source_id),
                    int(binding_generation),
                    str(transaction_kind),
                ),
            )
        operation = self.get_operation(operation_id, request_hash=request_hash)
        if operation is None:
            raise RuntimeError("annotation transaction journal insert was not readable")
        return operation

    def get_operation(
        self,
        operation_id: str,
        *,
        request_hash: str | None = None,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM annotation_entity_transactions WHERE operation_id = ?",
                (str(operation_id),),
            ).fetchone()
        if row is None:
            return None
        if request_hash is not None and str(row["request_hash"]) != str(request_hash):
            raise ValueError("annotation_transaction_id_reused_with_different_payload")
        return self._operation_from_row(row)

    def advance_operation(
        self,
        operation_id: str,
        state: str,
        *,
        response: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        order = {
            "prepared": 0,
            "annotation_committed": 1,
            "entity_committed": 2,
            "review_committed": 3,
            "recovery_required": 3,
            "complete": 4,
            "conflict": 4,
            "rejected": 4,
        }
        terminal = {"complete", "conflict", "rejected", "superseded"}
        if state not in order:
            raise ValueError(f"unsupported_annotation_transaction_state:{state}")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM annotation_entity_transactions WHERE operation_id = ?",
                (str(operation_id),),
            ).fetchone()
            if row is None:
                raise KeyError(operation_id)
            current = str(row["state"] or "complete")
            if current in terminal and current != state:
                raise ValueError(f"terminal_annotation_transaction:{current}")
            if current == "recovery_required" and state not in {
                "recovery_required",
                "review_committed",
                "complete",
                "conflict",
                "rejected",
            }:
                raise ValueError(
                    f"annotation_transaction_state_regression:{current}:{state}"
                )
            if current != "recovery_required" and order[state] < order.get(current, 0):
                raise ValueError(f"annotation_transaction_state_regression:{current}:{state}")
            next_response = (
                dict(response)
                if response is not None
                else json.loads(str(row["response_json"] or "{}"))
            )
            next_error = dict(error or {})
            phase = _OPERATION_PHASE_BY_STATE.get(state, str(row["phase"] or "prepared"))
            status = _OPERATION_STATUS_BY_STATE.get(state, state)
            connection.execute(
                "UPDATE annotation_entity_transactions SET state = ?, phase = ?, status = ?, "
                "retryable = ?, response_json = ?, error_json = ?, updated_at = ? "
                "WHERE operation_id = ?",
                (
                    state,
                    phase,
                    status,
                    1 if status == "recovery_required" else 0,
                    _semantic_json(next_response),
                    _semantic_json(next_error),
                    time.time(),
                    str(operation_id),
                ),
            )
        operation = self.get_operation(operation_id)
        if operation is None:
            raise RuntimeError("annotation transaction journal update was not readable")
        return operation

    def put_record_and_advance_operation(
        self,
        *,
        operation_id: str,
        source_identity: str,
        image_key: str,
        record: Mapping[str, Any],
        state: str,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Commit an entity-sidecar mutation and its journal phase atomically."""

        if state != "entity_committed":
            raise ValueError(f"unsupported_atomic_annotation_state:{state}")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM annotation_entity_transactions WHERE operation_id = ?",
                (str(operation_id),),
            ).fetchone()
            if row is None:
                raise KeyError(operation_id)
            current = str(row["state"] or "complete")
            if current == state:
                return self._operation_from_row(row)
            if current != "annotation_committed":
                raise ValueError(
                    f"annotation_transaction_state_regression:{current}:{state}"
                )
            self.put_record(
                source_identity,
                image_key,
                record,
                connection=connection,
            )
            connection.execute(
                "UPDATE annotation_entity_transactions SET state = ?, phase = ?, status = 'active', "
                "retryable = 0, response_json = ?, "
                "error_json = '{}', updated_at = ? WHERE operation_id = ?",
                (
                    state,
                    _OPERATION_PHASE_BY_STATE[state],
                    _semantic_json(response),
                    time.time(),
                    str(operation_id),
                ),
            )
        operation = self.get_operation(operation_id)
        if operation is None:
            raise RuntimeError("atomic annotation transaction update was not readable")
        return operation

    def mark_operation_conflict(
        self,
        *,
        operation_id: str,
        source_identity: str,
        image_key: str,
        detail: str | Mapping[str, Any],
        response: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist the identity diagnostic and terminal journal state together."""

        if isinstance(detail, Mapping):
            structured_detail = dict(detail)
            diagnostic = str(
                structured_detail.get("message")
                or structured_detail.get("detail")
                or structured_detail.get("code")
                or "annotation_entity_conflict"
            )
        else:
            diagnostic = str(detail)
            structured_detail = {
                "code": "annotation_entity_conflict",
                "message": diagnostic,
                "rerun_required": True,
            }
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM annotation_entity_transactions WHERE operation_id = ?",
                (str(operation_id),),
            ).fetchone()
            if row is None:
                raise KeyError(operation_id)
            current = str(row["state"] or "complete")
            if current == "complete":
                return self._operation_from_row(row)
            next_response = (
                dict(response)
                if response is not None
                else json.loads(str(row["response_json"] or "{}"))
            )
            next_response["terminal_recovery"] = structured_detail
            next_response["rerun_required"] = bool(
                structured_detail.get("rerun_required", True)
            )
            connection.execute(
                "INSERT INTO annotation_entity_conflicts "
                "(source_identity, image_key, detail, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(source_identity, image_key) DO UPDATE SET "
                "detail = excluded.detail, updated_at = excluded.updated_at",
                (
                    str(source_identity),
                    str(image_key),
                    diagnostic,
                    time.time(),
                ),
            )
            connection.execute(
                "UPDATE annotation_entity_transactions SET state = 'conflict', "
                "status = 'conflict', retryable = 0, response_json = ?, error_json = ?, "
                "updated_at = ? "
                "WHERE operation_id = ?",
                (
                    _semantic_json(next_response),
                    _semantic_json({"detail": structured_detail}),
                    time.time(),
                    str(operation_id),
                ),
            )
        operation = self.get_operation(operation_id)
        if operation is None:
            raise RuntimeError("conflicted annotation transaction was not readable")
        return operation

    def list_operations(
        self,
        *,
        job_id: str,
        source_identity: str | None = None,
        include_terminal: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = ["job_id = ?"]
        values: list[Any] = [str(job_id)]
        if source_identity is not None:
            clauses.append("source_identity = ?")
            values.append(str(source_identity))
        if not include_terminal:
            clauses.append(
                "status NOT IN ('complete', 'conflict', 'rejected', 'superseded', "
                "'legacy_unrecoverable')"
            )
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM annotation_entity_transactions WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at, operation_id",
                tuple(values),
            ).fetchall()
        return [self._operation_from_row(row) for row in rows]

    def list_checkpoint_operations(self, *, job_id: str) -> list[dict[str, Any]]:
        """Return only states that can affect recovery or checkpoint authority."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM annotation_entity_transactions "
                "WHERE job_id = ? AND status NOT IN ('complete', 'rejected', "
                "'superseded', 'legacy_unrecoverable') "
                "ORDER BY created_at, operation_id",
                (str(job_id),),
            ).fetchall()
        return [self._operation_from_row(row) for row in rows]

    def put_point_binding(
        self,
        *,
        job_id: str,
        point_id: str,
        source_identity: str,
        image_key: str,
        annotation_entity_id: str,
        entity_revision: int,
        record_revision: str,
        status: str,
        attestation: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO analysis_point_entities "
                "(job_id, point_id, source_identity, image_key, annotation_entity_id, "
                "entity_revision, record_revision, status, attestation, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(job_id, point_id) DO UPDATE SET "
                "source_identity = excluded.source_identity, image_key = excluded.image_key, "
                "annotation_entity_id = excluded.annotation_entity_id, "
                "entity_revision = excluded.entity_revision, "
                "record_revision = excluded.record_revision, status = excluded.status, "
                "attestation = excluded.attestation, updated_at = excluded.updated_at",
                (
                    str(job_id),
                    str(point_id),
                    str(source_identity),
                    str(image_key),
                    str(annotation_entity_id),
                    int(entity_revision),
                    str(record_revision),
                    str(status),
                    str(attestation),
                    time.time(),
                ),
            )

    def get_point_binding(self, job_id: str, point_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM analysis_point_entities WHERE job_id = ? AND point_id = ?",
                (str(job_id), str(point_id)),
            ).fetchone()
        return dict(row) if row is not None else None

    def create_batch(
        self,
        *,
        batch_id: str,
        job_id: str,
        source_mode: str,
        source_id: str,
        binding_generation: int,
        action: str,
        target_class_name: str,
        declared_count: int,
        manifest_hash: str,
    ) -> dict[str, Any]:
        now = time.time()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
            if existing is not None:
                current = dict(existing)
                if any(
                    (
                        str(current["job_id"]) != str(job_id),
                        str(current["source_mode"]) != str(source_mode),
                        str(current["source_id"]) != str(source_id),
                        int(current["binding_generation"]) != int(binding_generation),
                        str(current["action"]) != str(action),
                        str(current["target_class_name"]) != str(target_class_name),
                        int(current["declared_count"]) != int(declared_count),
                        str(current["manifest_hash"]) != str(manifest_hash),
                    )
                ):
                    raise ValueError("annotation_batch_id_reused_with_different_payload")
                return self._batch_from_row(connection, existing)
            connection.execute(
                "INSERT INTO annotation_operation_batches "
                "(batch_id, job_id, source_mode, source_id, binding_generation, action, "
                "target_class_name, declared_count, manifest_hash, state, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)",
                (
                    str(batch_id), str(job_id), str(source_mode), str(source_id),
                    int(binding_generation), str(action), str(target_class_name),
                    int(declared_count), str(manifest_hash), now, now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
            return self._batch_from_row(connection, row)

    def append_batch_items(
        self, batch_id: str, items: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        now = time.time()
        with self.transaction() as connection:
            batch = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
            if batch is None:
                raise KeyError(batch_id)
            if str(batch["state"]) not in {"draft", "uploading"}:
                raise ValueError("annotation_batch_manifest_closed")
            for item in items:
                sequence = int(item.get("sequence", -1))
                point_id = str(item.get("point_id") or "").strip()
                payload = (
                    dict(item.get("payload") or {})
                    if isinstance(item.get("payload"), Mapping)
                    else {}
                )
                if sequence < 0 or sequence >= int(batch["declared_count"]) or not point_id:
                    raise ValueError("annotation_batch_item_invalid")
                existing = connection.execute(
                    "SELECT point_id, payload_json FROM annotation_operation_batch_items "
                    "WHERE batch_id = ? AND sequence = ?",
                    (str(batch_id), sequence),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing["point_id"]) != point_id
                        or str(existing["payload_json"] or "{}") != _semantic_json(payload)
                    ):
                        raise ValueError("annotation_batch_sequence_conflict")
                    continue
                connection.execute(
                    "INSERT OR IGNORE INTO annotation_operation_batch_items "
                    "(batch_id, sequence, point_id, payload_json, state, updated_at) "
                    "VALUES (?, ?, ?, ?, 'pending', ?)",
                    (str(batch_id), sequence, point_id, _semantic_json(payload), now),
                )
            connection.execute(
                "UPDATE annotation_operation_batches SET state = 'uploading', updated_at = ? "
                "WHERE batch_id = ?",
                (now, str(batch_id)),
            )
            row = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
            return self._batch_from_row(connection, row)

    def start_batch(self, batch_id: str) -> dict[str, Any]:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
            if row is None:
                raise KeyError(batch_id)
            count = int(connection.execute(
                "SELECT COUNT(*) FROM annotation_operation_batch_items WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()[0])
            if count != int(row["declared_count"]):
                raise ValueError("annotation_batch_manifest_incomplete")
            manifest_rows = connection.execute(
                "SELECT sequence, point_id, payload_json "
                "FROM annotation_operation_batch_items WHERE batch_id = ? ORDER BY sequence",
                (str(batch_id),),
            ).fetchall()
            manifest = [
                {
                    "sequence": int(item["sequence"]),
                    "point_id": str(item["point_id"]),
                    "payload": json.loads(str(item["payload_json"] or "{}")),
                }
                for item in manifest_rows
            ]
            if self.request_hash(manifest) != str(row["manifest_hash"]):
                raise ValueError("annotation_batch_manifest_hash_mismatch")
            connection.execute(
                "UPDATE annotation_operation_batches SET state = 'ready', updated_at = ? "
                "WHERE batch_id = ? AND state IN ('draft', 'uploading')",
                (time.time(), str(batch_id)),
            )
            row = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
            return self._batch_from_row(connection, row)

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
            return self._batch_from_row(connection, row) if row is not None else None

    def list_active_batches(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE job_id = ? "
                "AND state NOT IN ('complete', 'partial', 'cancelled', 'conflict') "
                "ORDER BY created_at, batch_id",
                (str(job_id),),
            ).fetchall()
            return [self._batch_from_row(connection, row) for row in rows]

    def get_batch_items(
        self,
        batch_id: str,
        *,
        states: tuple[str, ...] | None = None,
        limit: int = 500,
        after_sequence: int = -1,
    ) -> list[dict[str, Any]]:
        clauses = ["batch_id = ?", "sequence > ?"]
        values: list[Any] = [str(batch_id), int(after_sequence)]
        if states:
            clauses.append("state IN (" + ",".join("?" for _ in states) + ")")
            values.extend(states)
        values.append(max(1, min(int(limit), 500)))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM annotation_operation_batch_items WHERE "
                + " AND ".join(clauses)
                + " ORDER BY sequence LIMIT ?",
                tuple(values),
            ).fetchall()
        return [self._batch_item_from_row(row) for row in rows]

    def settle_batch_items(
        self,
        batch_id: str,
        updates: list[Mapping[str, Any]],
        *,
        active_operation_id: str = "",
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            for update in updates:
                connection.execute(
                    "UPDATE annotation_operation_batch_items SET state = ?, operation_id = ?, "
                    "result_json = ?, error_json = ?, updated_at = ? "
                    "WHERE batch_id = ? AND sequence = ?",
                    (
                        str(update.get("state") or "failed"),
                        str(update.get("operation_id") or active_operation_id),
                        _semantic_json(update.get("result") if isinstance(update.get("result"), Mapping) else {}),
                        _semantic_json(update.get("error") if isinstance(update.get("error"), Mapping) else {}),
                        time.time(), str(batch_id), int(update.get("sequence", -1)),
                    ),
                )
            counts = {
                str(row["state"]): int(row["count"])
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM annotation_operation_batch_items "
                    "WHERE batch_id = ? GROUP BY state",
                    (str(batch_id),),
                ).fetchall()
            }
            pending = sum(counts.get(state, 0) for state in ("pending", "running", "recovery_required"))
            failed = counts.get("failed", 0) + counts.get("conflict", 0)
            state = "running" if pending else "partial" if failed else "complete"
            connection.execute(
                "UPDATE annotation_operation_batches SET state = ?, active_operation_id = ?, "
                "updated_at = ? WHERE batch_id = ?",
                (state, str(active_operation_id), time.time(), str(batch_id)),
            )
            row = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
            return self._batch_from_row(connection, row)

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE annotation_operation_batch_items SET state = 'cancelled', updated_at = ? "
                "WHERE batch_id = ? AND state = 'pending'",
                (time.time(), str(batch_id)),
            )
            connection.execute(
                "UPDATE annotation_operation_batches SET state = 'cancelled', updated_at = ? "
                "WHERE batch_id = ? AND state NOT IN ('complete', 'partial', 'conflict')",
                (time.time(), str(batch_id)),
            )
            row = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
            if row is None:
                raise KeyError(batch_id)
            return self._batch_from_row(connection, row)

    def retry_batch_items(
        self, batch_id: str, sequences: list[int] | None = None
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
            if row is None:
                raise KeyError(batch_id)
            clauses = [
                "batch_id = ?",
                "state IN ('failed', 'conflict', 'recovery_required')",
            ]
            values: list[Any] = [str(batch_id)]
            if sequences:
                normalized = sorted({int(value) for value in sequences})
                clauses.append(
                    "sequence IN (" + ",".join("?" for _ in normalized) + ")"
                )
                values.extend(normalized)
            connection.execute(
                "UPDATE annotation_operation_batch_items SET state = 'pending', "
                "error_json = '{}', updated_at = ? WHERE " + " AND ".join(clauses),
                (time.time(), *values),
            )
            connection.execute(
                "UPDATE annotation_operation_batches SET state = 'ready', "
                "active_operation_id = '', updated_at = ? WHERE batch_id = ?",
                (time.time(), str(batch_id)),
            )
            row = connection.execute(
                "SELECT * FROM annotation_operation_batches WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
            return self._batch_from_row(connection, row)

    @staticmethod
    def _batch_item_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "batch_id": str(row["batch_id"]),
            "sequence": int(row["sequence"]),
            "point_id": str(row["point_id"]),
            "payload": json.loads(str(row["payload_json"] or "{}")),
            "state": str(row["state"]),
            "operation_id": str(row["operation_id"] or ""),
            "result": json.loads(str(row["result_json"] or "{}")),
            "error": json.loads(str(row["error_json"] or "{}")),
            "updated_at": float(row["updated_at"] or 0),
        }

    def _batch_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        counts = {
            str(item["state"]): int(item["count"])
            for item in connection.execute(
                "SELECT state, COUNT(*) AS count FROM annotation_operation_batch_items "
                "WHERE batch_id = ? GROUP BY state",
                (str(row["batch_id"]),),
            ).fetchall()
        }
        return {
            **dict(row),
            "binding_generation": int(row["binding_generation"]),
            "declared_count": int(row["declared_count"]),
            "counts": counts,
            "error": json.loads(str(row["error_json"] or "{}")),
        }
