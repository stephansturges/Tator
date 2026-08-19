"""Bounded, normalized persistence for recoverable class-analysis sessions."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


SESSION_STORE_SCHEMA = "class-analysis-session-store-v3"
SESSION_STORE_FILENAME = "session.sqlite3"
SESSION_STORE_VALIDATION_ABI = "immutable-analysis-v2"
GRAPH_DEFAULT_ROWS = 50_000
GRAPH_MAX_ROWS = 50_000
GRAPH_MAX_BYTES = 64 * 1024 * 1024
QUEUE_DEFAULT_ROWS = 36
QUEUE_MAX_ROWS = 100
QUEUE_MAX_BYTES = 4 * 1024 * 1024
REVIEW_HISTORY_DEFAULT_ROWS = 250
REVIEW_HISTORY_MAX_ROWS = 500
REVIEW_HISTORY_MAX_BYTES = 16 * 1024 * 1024
DETAIL_MAX_BYTES = 256 * 1024
EVIDENCE_MAX_BYTES = 2 * 1024 * 1024
_SESSION_STORE_VALIDATION_CACHE_LOCK = threading.RLock()
_SESSION_STORE_VALIDATION_CACHE: dict[
    tuple[str, int, int, str, str], dict[str, Any]
] = {}
_SESSION_STORE_VALIDATION_CACHE_MAX = 32


class SessionStoreError(RuntimeError):
    """A stable transport or session-store contract failure."""

    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = int(status_code)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_text(value: Any) -> str:
    return _json_bytes(value).decode("utf-8")


def _bounded_payload(value: Any, max_bytes: int, detail: str) -> dict[str, Any]:
    payload = _json_bytes(value)
    if len(payload) > int(max_bytes):
        raise SessionStoreError(detail, status_code=413)
    return value


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    result = _finite_float(value, float("nan"))
    return result if math.isfinite(result) else None


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _boolean(value: Any) -> int:
    return 1 if bool(value) else 0


def _mode_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pca": "global_pca",
        "class_balanced": "class_balanced_pca",
        "between_class": "between_class_pca",
        "within_filter": "within_filter_pca",
    }
    return aliases.get(text, text)


def _selected_projection_mode(
    request: Mapping[str, Any], summary: Mapping[str, Any]
) -> str:
    for source in (request, summary):
        for key in (
            "projection",
            "projection_mode",
            "map_layout",
            "selected_projection",
        ):
            mode = _mode_name(source.get(key))
            if mode:
                return mode
    return "umap"


def _projection_pair(value: Any, ordinal: int) -> Optional[tuple[float, float]]:
    try:
        row = value[ordinal]
        if len(row) < 2:
            return None
        x = _optional_float(row[0])
        y = _optional_float(row[1])
    except (IndexError, KeyError, TypeError):
        return None
    if x is None or y is None:
        return None
    return (x, y)


def _point_projection(point: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    projection = point.get("projection")
    if isinstance(projection, Sequence) and not isinstance(projection, (str, bytes)):
        if len(projection) >= 2:
            x = _optional_float(projection[0])
            y = _optional_float(projection[1])
            if x is not None and y is not None:
                return (x, y)
    if isinstance(projection, Mapping):
        x = _optional_float(projection.get("x"))
        y = _optional_float(projection.get("y"))
        if x is not None and y is not None:
            return (x, y)
    for x_key, y_key in (("x", "y"), ("projection_x", "projection_y")):
        x = _optional_float(point.get(x_key))
        y = _optional_float(point.get(y_key))
        if x is not None and y is not None:
            return (x, y)
    return None


def _bbox(point: Mapping[str, Any]) -> tuple[Optional[float], ...]:
    value = point.get("bbox_xyxy") or point.get("bbox")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) >= 4:
            return tuple(_optional_float(value[index]) for index in range(4))
    return (None, None, None, None)


def _display_rank(point_id: str) -> int:
    digest = hashlib.sha256(point_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _evidence_payload(point: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    for key in ("refined_outlier", "spatial_evidence"):
        value = point.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _detail_payload(point: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "refined_outlier",
        "spatial_evidence",
        "token_cut_overlay",
        "projection",
    }
    return {key: value for key, value in point.items() if key not in excluded}


def _point_identity_values(
    point: Mapping[str, Any], point_id: str
) -> tuple[str, int, str, str, str, str, str]:
    entity_id = str(point.get("annotation_entity_id") or "").strip()
    entity_revision = _integer(point.get("annotation_entity_revision"), 0)
    record_revision = str(
        point.get("annotation_entity_record_revision") or ""
    ).strip()
    source_identity = str(point.get("annotation_source_identity") or "").strip()
    source_record_key = str(point.get("source_record_key") or "").strip()
    identity_status = str(point.get("identity_status") or "").strip()
    attestation = str(point.get("annotation_attestation") or "").strip()
    if identity_status == "ready":
        if not all(
            (
                entity_id,
                entity_revision > 0,
                record_revision,
                source_identity,
                source_record_key,
                attestation,
            )
        ):
            raise SessionStoreError(
                f"class_analysis_point_identity_invalid:{point_id}",
                status_code=409,
            )
    elif identity_status == "identity_conflict":
        if not source_identity or not source_record_key:
            raise SessionStoreError(
                f"class_analysis_point_identity_conflict_unscoped:{point_id}",
                status_code=409,
            )
    else:
        raise SessionStoreError(
            f"class_analysis_point_identity_status_invalid:{point_id}",
            status_code=409,
        )
    return (
        entity_id,
        entity_revision,
        record_revision,
        source_identity,
        source_record_key,
        identity_status,
        attestation,
    )


def _review_categories(
    point: Mapping[str, Any], evidence: Optional[Mapping[str, Any]]
) -> list[tuple[str, int, float]]:
    score = _finite_float(
        point.get("review_priority_score"),
        _finite_float(
            point.get("quality_score"),
            _finite_float(point.get("wrong_class_suspicion")),
        ),
    )
    rank_hint = 2_000_000_000
    if evidence:
        rank_hint = _integer(
            evidence.get("human_review_rank"),
            _integer(evidence.get("selector_priority_rank"), rank_hint),
        )
    categories: list[tuple[str, int, float]] = []
    quality_candidate = bool(point.get("quality_review_candidate"))
    wrong_candidate = bool(point.get("is_wrong_class_candidate")) or bool(
        point.get("proposed_class_differs")
    )
    rough_candidate = bool(point.get("is_rough_outlier_candidate")) or evidence is not None
    overlap_candidate = any(
        bool(point.get(key))
        for key in (
            "is_close_overlap_candidate",
            "is_dual_bbox_conflict",
            "pair_review_key",
        )
    )
    if quality_candidate or wrong_candidate or rough_candidate or overlap_candidate:
        categories.append(("review", rank_hint, score))
    if wrong_candidate:
        categories.append(("wrong_class", rank_hint, score))
    if rough_candidate:
        categories.append(("spatial_evidence", rank_hint, score))
    if overlap_candidate:
        categories.append(("overlap", rank_hint, score))
    if bool(point.get("tiny_object")) or bool(point.get("low_source_detail")):
        categories.append(("tiny_object", rank_hint, score))
    return categories


def _schema_sql() -> str:
    return """
    CREATE TABLE session_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE points_core (
        ordinal INTEGER PRIMARY KEY,
        point_id TEXT NOT NULL UNIQUE,
        split TEXT,
        image_relpath TEXT,
        frontend_image_key TEXT,
        source_key TEXT,
        class_id TEXT,
        class_name TEXT,
        x1 REAL,
        y1 REAL,
        x2 REAL,
        y2 REAL,
        width REAL,
        height REAL,
        cluster_id TEXT,
        outlier_score REAL,
        quality_score REAL,
        wrong_class_suspicion REAL,
        review_priority_score REAL,
        proposed_class TEXT,
        quality_review_candidate INTEGER NOT NULL DEFAULT 0,
        quality_queue_bucket TEXT,
        tiny_object INTEGER NOT NULL DEFAULT 0,
        low_source_detail INTEGER NOT NULL DEFAULT 0,
        relative_small_object INTEGER NOT NULL DEFAULT 0,
        is_wrong_class_candidate INTEGER NOT NULL DEFAULT 0,
        is_close_overlap_candidate INTEGER NOT NULL DEFAULT 0,
        is_dual_bbox_conflict INTEGER NOT NULL DEFAULT 0,
        is_rough_outlier_candidate INTEGER NOT NULL DEFAULT 0,
        reviewed INTEGER NOT NULL DEFAULT 0,
        review_object_key TEXT,
        pair_review_key TEXT,
        display_rank INTEGER NOT NULL,
        annotation_entity_id TEXT,
        annotation_entity_revision INTEGER,
        annotation_entity_record_revision TEXT,
        annotation_source_identity TEXT,
        source_record_key TEXT,
        identity_status TEXT,
        annotation_attestation TEXT
    );
    CREATE TABLE point_projections (
        point_id TEXT NOT NULL,
        mode TEXT NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        PRIMARY KEY (point_id, mode)
    );
    CREATE TABLE point_details (
        point_id TEXT PRIMARY KEY,
        payload TEXT NOT NULL
    );
    CREATE TABLE review_queue (
        category TEXT NOT NULL,
        rank INTEGER NOT NULL DEFAULT 0,
        rank_hint INTEGER NOT NULL,
        point_id TEXT NOT NULL,
        score REAL NOT NULL,
        display_rank INTEGER NOT NULL,
        PRIMARY KEY (category, point_id)
    );
    CREATE TABLE overlap_pairs (
        pair_review_key TEXT NOT NULL,
        point_id TEXT NOT NULL,
        other_point_id TEXT,
        payload TEXT,
        PRIMARY KEY (pair_review_key, point_id)
    );
    CREATE TABLE review_state (
        point_id TEXT PRIMARY KEY,
        disposition TEXT,
        revision INTEGER,
        reviewed_at TEXT,
        payload TEXT
    );
    CREATE TABLE evidence (
        point_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        payload BLOB NOT NULL,
        sidecar_row INTEGER,
        fingerprint TEXT,
        updated_at REAL NOT NULL
    );
    CREATE TABLE evidence_tasks (
        point_id TEXT PRIMARY KEY,
        priority INTEGER NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        source_key TEXT,
        candidate_payload TEXT NOT NULL DEFAULT '{}',
        lease_token TEXT,
        updated_at REAL NOT NULL
    );
    """


def _flush_rows(
    connection: sqlite3.Connection,
    buffers: dict[str, list[tuple[Any, ...]]],
) -> None:
    statements = {
        "points": "INSERT INTO points_core VALUES ("
        + ",".join("?" for _ in range(40))
        + ")",
        "projections": "INSERT OR REPLACE INTO point_projections VALUES (?,?,?,?)",
        "details": "INSERT INTO point_details VALUES (?,?)",
        "queues": "INSERT OR REPLACE INTO review_queue VALUES (?,?,?,?,?,?)",
        "overlaps": "INSERT OR REPLACE INTO overlap_pairs VALUES (?,?,?,?)",
        "reviews": "INSERT OR REPLACE INTO review_state VALUES (?,?,?,?,?)",
        "evidence": "INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?,?)",
        "tasks": "INSERT OR REPLACE INTO evidence_tasks VALUES (?,?,?,?,?,?,?,?,?)",
    }
    for key, statement in statements.items():
        rows = buffers[key]
        if rows:
            connection.executemany(statement, rows)
            rows.clear()


def _normalize_queue_ranks(connection: sqlite3.Connection) -> None:
    categories = [
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT category FROM review_queue ORDER BY category"
        )
    ]
    for category in categories:
        cursor = connection.execute(
            """SELECT point_id FROM review_queue
               WHERE category = ?
               ORDER BY rank_hint ASC, score DESC, display_rank ASC, point_id ASC""",
            (category,),
        )
        rank = 0
        while True:
            rows = cursor.fetchmany(2_000)
            if not rows:
                break
            updates = []
            for row in rows:
                rank += 1
                updates.append((rank, category, str(row[0])))
            connection.executemany(
                "UPDATE review_queue SET rank = ? WHERE category = ? AND point_id = ?",
                updates,
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_class_analysis_session_store(
    target_path: Path | str,
    points: Iterable[Mapping[str, Any]],
    *,
    summary: Optional[Mapping[str, Any]] = None,
    request: Optional[Mapping[str, Any]] = None,
    projection_coordinates: Optional[Mapping[str, Any]] = None,
    expected_point_count: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """Write, validate, fsync, and atomically publish one normalized store."""

    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    if target.exists():
        raise SessionStoreError("class_analysis_session_store_exists", status_code=409)
    summary_value = dict(summary or {})
    request_value = dict(request or {})
    selected_mode = _selected_projection_mode(request_value, summary_value)
    projection_sources = {
        _mode_name(mode): values
        for mode, values in dict(projection_coordinates or {}).items()
        if _mode_name(mode)
    }
    available_modes: set[str] = set(projection_sources)
    point_count = 0
    evidence_count = 0
    queue_count = 0
    class_counts: dict[str, int] = {}
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = sqlite3.connect(str(partial))
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.executescript(_schema_sql())
        connection.execute("BEGIN")
        buffers: dict[str, list[tuple[Any, ...]]] = {
            key: []
            for key in (
                "points",
                "projections",
                "details",
                "queues",
                "overlaps",
                "reviews",
                "evidence",
                "tasks",
            )
        }
        now = time.time()
        for ordinal, raw_point in enumerate(points):
            if cancel_check is not None and ordinal % 1_000 == 0 and cancel_check():
                raise RuntimeError("cancelled")
            if not isinstance(raw_point, Mapping):
                continue
            point = dict(raw_point)
            point_id = str(point.get("point_id") or "").strip()
            if not point_id:
                raise SessionStoreError("class_analysis_point_id_required")
            identity_values = _point_identity_values(point, point_id)
            x1, y1, x2, y2 = _bbox(point)
            class_name = str(point.get("class_name") or "")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            display_rank = _display_rank(point_id)
            evidence = _evidence_payload(point)
            reviewed = bool(
                point.get("reviewed")
                or point.get("review_disposition")
                or point.get("human_review_disposition")
            )
            buffers["points"].append(
                (
                    ordinal,
                    point_id,
                    str(point.get("split") or ""),
                    str(point.get("image_relpath") or ""),
                    str(point.get("frontend_image_key") or ""),
                    str(
                        point.get("source_key")
                        or point.get("frontend_image_key")
                        or point.get("image_relpath")
                        or ""
                    ),
                    str(point.get("class_id") or ""),
                    class_name,
                    x1,
                    y1,
                    x2,
                    y2,
                    _optional_float(point.get("width")),
                    _optional_float(point.get("height")),
                    str(point.get("cluster_id") or ""),
                    _optional_float(point.get("outlier_score")),
                    _optional_float(point.get("quality_score")),
                    _optional_float(point.get("wrong_class_suspicion")),
                    _optional_float(point.get("review_priority_score")),
                    str(point.get("proposed_class") or ""),
                    _boolean(point.get("quality_review_candidate")),
                    str(point.get("quality_queue_bucket") or ""),
                    _boolean(point.get("tiny_object")),
                    _boolean(point.get("low_source_detail")),
                    _boolean(point.get("relative_small_object")),
                    _boolean(point.get("is_wrong_class_candidate")),
                    _boolean(point.get("is_close_overlap_candidate")),
                    _boolean(point.get("is_dual_bbox_conflict")),
                    _boolean(point.get("is_rough_outlier_candidate")),
                    _boolean(reviewed),
                    str(point.get("review_object_key") or ""),
                    str(point.get("pair_review_key") or ""),
                    display_rank,
                    *identity_values,
                )
            )
            selected_projection = _point_projection(point)
            if selected_projection is not None:
                available_modes.add(selected_mode)
                buffers["projections"].append(
                    (point_id, selected_mode, selected_projection[0], selected_projection[1])
                )
            for mode, values in projection_sources.items():
                projection = _projection_pair(values, ordinal)
                if projection is not None:
                    buffers["projections"].append(
                        (point_id, mode, projection[0], projection[1])
                    )
            detail = _detail_payload(point)
            detail_bytes = _json_bytes(detail)
            if len(detail_bytes) > DETAIL_MAX_BYTES:
                raise SessionStoreError(
                    f"class_analysis_point_detail_too_large:{point_id}",
                    status_code=413,
                )
            buffers["details"].append((point_id, detail_bytes.decode("utf-8")))
            categories = _review_categories(point, evidence)
            for category, rank_hint, score in categories:
                buffers["queues"].append(
                    (category, 0, rank_hint, point_id, score, display_rank)
                )
                queue_count += 1
            pair_key = str(point.get("pair_review_key") or "").strip()
            if pair_key:
                other_id = str(
                    point.get("overlap_other_point_id")
                    or point.get("paired_point_id")
                    or ""
                )
                overlap_payload = point.get("overlap_pair")
                buffers["overlaps"].append(
                    (
                        pair_key,
                        point_id,
                        other_id,
                        _json_text(overlap_payload) if overlap_payload is not None else None,
                    )
                )
            if reviewed:
                review_payload = point.get("review_state")
                buffers["reviews"].append(
                    (
                        point_id,
                        str(
                            point.get("review_disposition")
                            or point.get("human_review_disposition")
                            or "reviewed"
                        ),
                        _integer(point.get("human_review_revision"), 0),
                        str(point.get("reviewed_at") or ""),
                        _json_text(review_payload) if review_payload is not None else None,
                    )
                )
            if evidence is not None:
                evidence_bytes = _json_bytes(evidence)
                if len(evidence_bytes) > EVIDENCE_MAX_BYTES:
                    raise SessionStoreError(
                        f"class_analysis_point_evidence_too_large:{point_id}",
                        status_code=413,
                    )
                buffers["evidence"].append(
                    (
                        point_id,
                        "completed",
                        sqlite3.Binary(zlib.compress(evidence_bytes, level=6)),
                        _integer(evidence.get("sidecar_row"), -1),
                        str(evidence.get("evidence_fingerprint") or ""),
                        now,
                    )
                )
                buffers["tasks"].append(
                    (
                        point_id,
                        _integer(evidence.get("human_review_rank"), ordinal + 1),
                        "completed",
                        1,
                        None,
                        str(
                            point.get("source_key")
                            or point.get("frontend_image_key")
                            or point.get("image_relpath")
                            or ""
                        ),
                        _json_text(_detail_payload(point)),
                        None,
                        now,
                    )
                )
                evidence_count += 1
            point_count += 1
            if point_count % 1_000 == 0:
                _flush_rows(connection, buffers)
        _flush_rows(connection, buffers)
        if expected_point_count is not None and point_count != int(expected_point_count):
            raise SessionStoreError(
                f"class_analysis_session_point_count_mismatch:{point_count}:"
                f"{int(expected_point_count)}"
            )
        _normalize_queue_ranks(connection)
        metadata = {
            "schema": SESSION_STORE_SCHEMA,
            "store_validation_id": uuid.uuid4().hex,
            "created_at": time.time(),
            "summary": summary_value,
            "request": request_value,
            "selected_projection_mode": selected_mode,
            "projection_modes": [
                selected_mode,
                *sorted(mode for mode in available_modes if mode != selected_mode),
            ],
            "point_count": point_count,
            "evidence_count": evidence_count,
            "queue_entry_count": queue_count,
            "class_counts": class_counts,
        }
        connection.executemany(
            "INSERT INTO session_meta(key, value) VALUES (?, ?)",
            [(key, _json_text(value)) for key, value in metadata.items()],
        )
        connection.executescript(
            """
            CREATE INDEX points_core_class_idx ON points_core(class_name, ordinal);
            CREATE INDEX points_core_review_idx ON points_core(reviewed, ordinal);
            CREATE INDEX points_core_flags_idx ON points_core(
                quality_review_candidate, is_wrong_class_candidate,
                is_rough_outlier_candidate, is_close_overlap_candidate
            );
            CREATE INDEX point_projections_mode_idx ON point_projections(mode, point_id);
            CREATE UNIQUE INDEX review_queue_rank_idx ON review_queue(category, rank);
            CREATE INDEX review_queue_point_idx ON review_queue(point_id);
            CREATE INDEX overlap_pairs_point_idx ON overlap_pairs(point_id);
            CREATE INDEX evidence_status_idx ON evidence(status, point_id);
            CREATE INDEX evidence_tasks_status_idx ON evidence_tasks(status, priority, point_id);
            """
        )
        connection.commit()
        check = connection.execute("PRAGMA quick_check").fetchone()
        if not check or str(check[0]).lower() != "ok":
            raise SessionStoreError("class_analysis_session_store_integrity_failed")
        connection.close()
        connection = None
        _fsync_path(partial)
        validation = validate_class_analysis_session_store(
            partial,
            expected_point_count=expected_point_count,
            expected_evidence_count=evidence_count,
        )
        os.replace(partial, target)
        _remember_session_store_validation(target, validation)
        _fsync_directory(target.parent)
        size_bytes = target.stat().st_size
        digest = _sha256_file(target)
        return {
            "session_store_file": target.name,
            "session_store_schema": SESSION_STORE_SCHEMA,
            "session_store_sha256": digest,
            "session_store_bytes": size_bytes,
            "session_store_point_count": point_count,
            "session_store_evidence_count": evidence_count,
            "transport": {
                "schema": SESSION_STORE_SCHEMA,
                "graph_default_rows": GRAPH_DEFAULT_ROWS,
                "graph_max_rows": GRAPH_MAX_ROWS,
                "queue_default_rows": QUEUE_DEFAULT_ROWS,
                "queue_max_rows": QUEUE_MAX_ROWS,
                "point_count": point_count,
                "evidence_count": evidence_count,
                "projection_modes": metadata["projection_modes"],
            },
        }
    except Exception:
        if connection is not None:
            connection.close()
        partial.unlink(missing_ok=True)
        raise


def _open_readonly(
    path: Path | str,
    *,
    require_validated: bool = True,
) -> sqlite3.Connection:
    store = Path(path)
    if not store.is_file():
        raise SessionStoreError("class_analysis_session_store_not_found", status_code=404)
    if require_validated:
        ensure_class_analysis_session_store_validated(store)
    connection = sqlite3.connect(f"file:{store.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


_POINT_IDENTITY_COLUMNS = (
    "annotation_entity_id",
    "annotation_entity_revision",
    "annotation_entity_record_revision",
    "annotation_source_identity",
    "source_record_key",
    "identity_status",
    "annotation_attestation",
)


def _require_current_session_store_schema(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT value FROM session_meta WHERE key = 'schema'"
    ).fetchone()
    try:
        schema = json.loads(str(row[0])) if row is not None else ""
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SessionStoreError(
            "class_analysis_session_store_schema_invalid", status_code=409
        ) from exc
    if schema != SESSION_STORE_SCHEMA:
        raise SessionStoreError(
            "class_analysis_session_store_schema_unsupported", status_code=409
        )
    available = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(points_core)")
    }
    if any(name not in available for name in _POINT_IDENTITY_COLUMNS):
        raise SessionStoreError(
            "class_analysis_session_store_identity_schema_invalid",
            status_code=409,
        )


def _point_identity_select_sql(
    connection: sqlite3.Connection, alias: str = "p"
) -> str:
    _require_current_session_store_schema(connection)
    return ", ".join(
        f"{alias}.{name} AS {name}" for name in _POINT_IDENTITY_COLUMNS
    )


def read_session_store_metadata(path: Path | str) -> dict[str, Any]:
    with _open_readonly(path) as connection:
        _require_current_session_store_schema(connection)
        rows = connection.execute("SELECT key, value FROM session_meta").fetchall()
    metadata = {str(row["key"]): json.loads(str(row["value"])) for row in rows}
    return metadata


def get_class_analysis_session_source_identities(
    path: Path | str,
) -> list[str]:
    return list(
        get_class_analysis_session_identity_summary(path)["source_identities"]
    )


def validate_class_analysis_session_store(
    path: Path | str,
    *,
    expected_point_count: Optional[int] = None,
    expected_evidence_count: Optional[int] = None,
) -> dict[str, Any]:
    with _open_readonly(path, require_validated=False) as connection:
        _require_current_session_store_schema(connection)
        check = connection.execute("PRAGMA quick_check").fetchone()
        point_count = int(connection.execute("SELECT COUNT(*) FROM points_core").fetchone()[0])
        evidence_count = int(connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0])
        projection_count = int(
            connection.execute("SELECT COUNT(*) FROM point_projections").fetchone()[0]
        )
        distinct_points = int(
            connection.execute("SELECT COUNT(DISTINCT point_id) FROM points_core").fetchone()[0]
        )
        invalid_identity_count = int(
            connection.execute(
                """SELECT COUNT(*) FROM points_core
                   WHERE (identity_status = 'ready' AND (
                              annotation_entity_id = ''
                              OR annotation_entity_revision <= 0
                              OR annotation_entity_record_revision = ''
                              OR annotation_source_identity = ''
                              OR source_record_key = ''
                              OR annotation_attestation = ''
                          ))
                      OR (identity_status = 'identity_conflict' AND (
                              annotation_source_identity = ''
                              OR source_record_key = ''
                          ))
                      OR identity_status NOT IN ('ready', 'identity_conflict')"""
            ).fetchone()[0]
        )
        ready_identity_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM points_core WHERE identity_status = 'ready'"
            ).fetchone()[0]
        )
        identity_conflict_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM points_core "
                "WHERE identity_status = 'identity_conflict'"
            ).fetchone()[0]
        )
    if not check or str(check[0]).lower() != "ok":
        raise SessionStoreError("class_analysis_session_store_integrity_failed")
    if point_count != distinct_points:
        raise SessionStoreError("class_analysis_session_point_identity_collision")
    if invalid_identity_count:
        raise SessionStoreError(
            "class_analysis_session_point_identity_invalid", status_code=409
        )
    if expected_point_count is not None and point_count != int(expected_point_count):
        raise SessionStoreError("class_analysis_session_point_count_mismatch")
    if expected_evidence_count is not None and evidence_count != int(expected_evidence_count):
        raise SessionStoreError("class_analysis_session_evidence_count_mismatch")
    if point_count and not projection_count:
        raise SessionStoreError("class_analysis_session_projection_missing")
    return {
        "schema": SESSION_STORE_SCHEMA,
        "point_count": point_count,
        "evidence_count": evidence_count,
        "projection_count": projection_count,
        "ready_identity_count": ready_identity_count,
        "identity_conflict_count": identity_conflict_count,
        "invalid_identity_count": invalid_identity_count,
    }


def _session_store_validation_key(
    path: Path | str,
) -> tuple[str, int, int, str, str]:
    store = Path(path)
    if not store.is_file():
        raise SessionStoreError(
            "class_analysis_session_store_not_found",
            status_code=404,
        )
    stat = store.stat()
    try:
        connection = sqlite3.connect(
            f"file:{store.resolve()}?mode=ro",
            uri=True,
        )
        try:
            row = connection.execute(
                "SELECT value FROM session_meta WHERE key = 'store_validation_id'"
            ).fetchone()
        finally:
            connection.close()
        validation_id = str(json.loads(str(row[0]))) if row is not None else ""
    except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SessionStoreError(
            "class_analysis_session_store_validation_id_unreadable",
            status_code=409,
        ) from exc
    if not validation_id:
        raise SessionStoreError(
            "class_analysis_session_store_validation_id_missing",
            status_code=409,
        )
    return (
        str(store.resolve()),
        int(stat.st_dev),
        int(stat.st_ino),
        validation_id,
        SESSION_STORE_VALIDATION_ABI,
    )


def ensure_class_analysis_session_store_validated(
    path: Path | str,
) -> dict[str, Any]:
    """Validate one immutable artifact once for its exact filesystem identity."""

    key = _session_store_validation_key(path)
    with _SESSION_STORE_VALIDATION_CACHE_LOCK:
        cached = _SESSION_STORE_VALIDATION_CACHE.get(key)
        if cached is not None:
            return dict(cached)
        result = validate_class_analysis_session_store(path)
        _remember_session_store_validation(path, result)
        return dict(result)


def _remember_session_store_validation(
    path: Path | str,
    result: Mapping[str, Any],
) -> None:
    key = _session_store_validation_key(path)
    with _SESSION_STORE_VALIDATION_CACHE_LOCK:
        resolved = key[0]
        for stale_key in tuple(_SESSION_STORE_VALIDATION_CACHE):
            if stale_key[0] == resolved and stale_key != key:
                _SESSION_STORE_VALIDATION_CACHE.pop(stale_key, None)
        while len(_SESSION_STORE_VALIDATION_CACHE) >= _SESSION_STORE_VALIDATION_CACHE_MAX:
            _SESSION_STORE_VALIDATION_CACHE.pop(
                next(iter(_SESSION_STORE_VALIDATION_CACHE))
            )
        _SESSION_STORE_VALIDATION_CACHE[key] = dict(result)


def get_class_analysis_session_identity_summary(
    path: Path | str,
) -> dict[str, Any]:
    validation = ensure_class_analysis_session_store_validated(path)
    with _open_readonly(path) as connection:
        rows = connection.execute(
            "SELECT annotation_source_identity, identity_status, COUNT(*) AS count "
            "FROM points_core GROUP BY annotation_source_identity, identity_status"
        ).fetchall()
    source_identities = sorted(
        {
            str(row["annotation_source_identity"])
            for row in rows
            if str(row["annotation_source_identity"] or "")
        }
    )
    return {
        "schema": SESSION_STORE_SCHEMA,
        "point_count": int(validation["point_count"]),
        "ready_identity_count": int(validation["ready_identity_count"]),
        "identity_conflict_count": int(validation["identity_conflict_count"]),
        "invalid_identity_count": int(validation["invalid_identity_count"]),
        "source_identities": source_identities,
    }


def _graph_predicates(
    *,
    class_name: Optional[str],
    objects: str,
    object_size: str,
    reviewed: str,
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    values: list[Any] = []
    if class_name and class_name != "__all__":
        clauses.append("p.class_name = ?")
        values.append(str(class_name))
    object_mode = str(objects or "all")
    object_clauses = {
        "all": None,
        "review": "p.quality_review_candidate = 1",
        "wrong_class": "p.is_wrong_class_candidate = 1",
        "spatial_evidence": "p.is_rough_outlier_candidate = 1",
        "overlap": "(p.is_close_overlap_candidate = 1 OR p.is_dual_bbox_conflict = 1)",
    }
    if object_mode not in object_clauses:
        raise SessionStoreError("class_analysis_graph_object_filter_invalid")
    if object_clauses[object_mode]:
        clauses.append(str(object_clauses[object_mode]))
    size_mode = str(object_size or "all")
    size_clauses = {
        "all": None,
        "tiny_only": "(p.tiny_object = 1 OR p.low_source_detail = 1)",
        "exclude_tiny": "(p.tiny_object = 0 AND p.low_source_detail = 0)",
        "relative_small": "p.relative_small_object = 1",
    }
    if size_mode not in size_clauses:
        raise SessionStoreError("class_analysis_graph_size_filter_invalid")
    if size_clauses[size_mode]:
        clauses.append(str(size_clauses[size_mode]))
    effective_reviewed = (
        "CASE WHEN r.point_id IS NULL THEN p.reviewed "
        "WHEN COALESCE(r.disposition, '') = '' THEN 0 ELSE 1 END"
    )
    reviewed_mode = str(reviewed or "any")
    if reviewed_mode == "reviewed":
        clauses.append(f"({effective_reviewed}) = 1")
    elif reviewed_mode == "unreviewed":
        clauses.append(f"({effective_reviewed}) = 0")
    elif reviewed_mode != "any":
        raise SessionStoreError("class_analysis_graph_review_filter_invalid")
    return clauses, values


def _review_state_version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT value FROM session_meta WHERE key = 'review_state_version'"
    ).fetchone()
    if row is None:
        return 0
    try:
        return int(json.loads(str(row[0])))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0


def _bump_review_state_version(connection: sqlite3.Connection) -> int:
    version = _review_state_version(connection) + 1
    connection.execute(
        "INSERT OR REPLACE INTO session_meta(key, value) VALUES (?, ?)",
        ("review_state_version", _json_text(version)),
    )
    return version


def upsert_class_analysis_review_state(
    path: Path | str,
    *,
    point_id: str,
    disposition: str,
    revision: Any = "",
    reviewed_at: Any = "",
    payload: Optional[Mapping[str, Any]] = None,
) -> int:
    """Project one durable receipt into the normalized session store."""

    identifier = str(point_id or "").strip()
    state = str(disposition or "").strip().lower()
    if not identifier or not state:
        raise SessionStoreError("class_analysis_review_state_invalid")
    ensure_class_analysis_session_store_validated(path)
    connection = sqlite3.connect(str(Path(path)))
    try:
        # Receipt projection is a disposable index over the durable ledger.
        # Do not hold an interactive review request behind a long store writer.
        connection.execute("PRAGMA busy_timeout=1000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("BEGIN IMMEDIATE")
        review_object_key = str(
            (payload or {}).get("review_object_key")
            if isinstance(payload, Mapping)
            else ""
        ).strip()
        identifiers = {
            str(row[0])
            for row in connection.execute(
                """SELECT point_id FROM points_core
                   WHERE ((? <> '' AND (review_object_key = ? OR pair_review_key = ?))
                          OR (? = '' AND point_id = ?))""",
                (
                    review_object_key,
                    review_object_key,
                    review_object_key,
                    review_object_key,
                    identifier,
                ),
            )
        }
        if not identifiers:
            connection.rollback()
            return _review_state_version(connection)
        connection.executemany(
            """INSERT OR REPLACE INTO review_state
               (point_id, disposition, revision, reviewed_at, payload)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    target_id,
                    state,
                    str(revision or ""),
                    str(reviewed_at or ""),
                    _json_text(dict(payload)) if isinstance(payload, Mapping) else None,
                )
                for target_id in identifiers
            ],
        )
        version = _bump_review_state_version(connection)
        connection.commit()
        return version
    finally:
        connection.close()


def clear_class_analysis_review_state(
    path: Path | str,
    *,
    point_id: str,
    review_object_key: str = "",
) -> int:
    """Persist an explicit unreviewed tombstone over immutable build-time state."""

    identifier = str(point_id or "").strip()
    if not identifier:
        raise SessionStoreError("class_analysis_review_state_invalid")
    ensure_class_analysis_session_store_validated(path)
    connection = sqlite3.connect(str(Path(path)))
    try:
        connection.execute("PRAGMA busy_timeout=1000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("BEGIN IMMEDIATE")
        review_key = str(review_object_key or "").strip()
        identifiers = {
            str(row[0])
            for row in connection.execute(
                """SELECT point_id FROM points_core
                   WHERE ((? <> '' AND (review_object_key = ? OR pair_review_key = ?))
                          OR (? = '' AND point_id = ?))""",
                (review_key, review_key, review_key, review_key, identifier),
            )
        }
        if not identifiers:
            connection.rollback()
            return _review_state_version(connection)
        connection.executemany(
            """INSERT OR REPLACE INTO review_state
               (point_id, disposition, revision, reviewed_at, payload)
               VALUES (?, '', '', '', NULL)""",
            [(target_id,) for target_id in identifiers],
        )
        version = _bump_review_state_version(connection)
        connection.commit()
        return version
    finally:
        connection.close()


def replace_class_analysis_review_state(
    path: Path | str,
    entries: Sequence[Mapping[str, Any]],
) -> int:
    """Reconcile a restored session with the authoritative receipt ledger."""

    ensure_class_analysis_session_store_validated(path)
    connection = sqlite3.connect(str(Path(path)))
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("BEGIN IMMEDIATE")
        def review_state_digest() -> bytes:
            digest = hashlib.sha256()
            for row in connection.execute(
                """SELECT point_id, disposition, revision, reviewed_at, payload
                   FROM review_state ORDER BY point_id"""
            ):
                encoded = _json_bytes(list(row))
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
            return digest.digest()

        previous_digest = review_state_digest()
        # This table is a ledger projection, not the source of build-time
        # review state. Deleting absent rows preserves points_core.reviewed as
        # the fallback; explicit clear receipts are projected as blank rows.
        connection.execute("DELETE FROM review_state")
        rows = []
        for entry in entries:
            point_id = str(entry.get("point_id") or "").strip()
            review_key = str(entry.get("review_object_key") or "").strip()
            disposition = str(entry.get("disposition") or "").strip().lower()
            if (not point_id and not review_key) or not disposition:
                continue
            rows.append(
                (
                    point_id,
                    "" if disposition == "clear" else disposition,
                    str(entry.get("entry_revision") or ""),
                    str(entry.get("updated_at") or ""),
                    _json_text(dict(entry)),
                    review_key,
                )
            )
        if rows:
            for point_id, disposition, revision, reviewed_at, payload, review_key in rows:
                connection.execute(
                    """INSERT OR REPLACE INTO review_state
                       (point_id, disposition, revision, reviewed_at, payload)
                       SELECT p.point_id, ?, ?, ?, ?
                       FROM points_core p
                       WHERE ((? <> '' AND (p.review_object_key = ? OR p.pair_review_key = ?))
                              OR (? = '' AND p.point_id = ?))""",
                    (
                        disposition,
                        revision,
                        reviewed_at,
                        payload,
                        review_key,
                        review_key,
                        review_key,
                        review_key,
                        point_id,
                    ),
                )
        current_digest = review_state_digest()
        version = (
            _review_state_version(connection)
            if current_digest == previous_digest
            else _bump_review_state_version(connection)
        )
        connection.commit()
        return version
    finally:
        connection.close()


def _graph_cursor_query_key(
    *,
    mode: str,
    class_name: Optional[str],
    objects: str,
    object_size: str,
    reviewed: str,
    review_state_version: int,
) -> str:
    return hashlib.sha256(
        _json_bytes(
            [
                mode,
                str(class_name or ""),
                str(objects or "all"),
                str(object_size or "all"),
                str(reviewed or "any"),
                int(review_state_version),
            ]
        )
    ).hexdigest()[:24]


def _encode_graph_cursor(query_key: str, row: sqlite3.Row) -> str:
    payload = _json_bytes(
        [
            query_key,
            int(row["review_bucket"]),
            _finite_float(row["priority_sort"]),
            int(row["display_rank"]),
            int(row["ordinal"]),
        ]
    )
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_graph_cursor(
    cursor: Optional[str], query_key: str
) -> Optional[tuple[int, float, int, int]]:
    if not cursor:
        return None
    try:
        padded = str(cursor) + "=" * (-len(str(cursor)) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        if not isinstance(value, list) or len(value) != 5 or value[0] != query_key:
            raise ValueError
        return (
            int(value[1]),
            float(value[2]),
            int(value[3]),
            int(value[4]),
        )
    except Exception as exc:  # noqa: BLE001
        raise SessionStoreError(
            "class_analysis_graph_cursor_invalid", status_code=409
        ) from exc


def get_class_analysis_graph_payload(
    path: Path | str,
    *,
    projection_mode: Optional[str] = None,
    class_name: Optional[str] = None,
    objects: str = "all",
    object_size: str = "all",
    reviewed: str = "any",
    limit: int = GRAPH_DEFAULT_ROWS,
    cursor: Optional[str] = None,
) -> dict[str, Any]:
    row_limit = _integer(limit, GRAPH_DEFAULT_ROWS)
    if row_limit < 1 or row_limit > GRAPH_MAX_ROWS:
        raise SessionStoreError("class_analysis_graph_limit_invalid")
    metadata = read_session_store_metadata(path)
    modes = [str(mode) for mode in metadata.get("projection_modes") or []]
    mode = _mode_name(projection_mode or metadata.get("selected_projection_mode"))
    if mode not in modes:
        raise SessionStoreError("class_analysis_projection_mode_unavailable", status_code=409)
    clauses, values = _graph_predicates(
        class_name=class_name,
        objects=objects,
        object_size=object_size,
        reviewed=reviewed,
    )
    where = " AND ".join(["q.mode = ?", *clauses])
    parameters = [mode, *values]
    with _open_readonly(path) as connection:
        review_version = _review_state_version(connection)
        query_key = _graph_cursor_query_key(
            mode=mode,
            class_name=class_name,
            objects=objects,
            object_size=object_size,
            reviewed=reviewed,
            review_state_version=review_version,
        )
        cursor_values = _decode_graph_cursor(cursor, query_key)
        total = int(
            connection.execute(
                f"""SELECT COUNT(*) FROM point_projections q
                    JOIN points_core p ON p.point_id = q.point_id
                    LEFT JOIN review_state r ON r.point_id = p.point_id
                    WHERE {where}""",
                parameters,
            ).fetchone()[0]
        )
        review_bucket = (
            "CASE WHEN p.quality_review_candidate = 1 "
            "OR p.is_wrong_class_candidate = 1 "
            "OR p.is_rough_outlier_candidate = 1 "
            "OR p.is_close_overlap_candidate = 1 THEN 0 ELSE 1 END"
        )
        priority_sort = "-COALESCE(p.review_priority_score, 0)"
        page_where = where
        page_parameters = list(parameters)
        if cursor_values is not None:
            page_where += (
                f" AND ({review_bucket}, {priority_sort}, p.display_rank, p.ordinal) "
                "> (?, ?, ?, ?)"
            )
            page_parameters.extend(cursor_values)
        identity_select = _point_identity_select_sql(connection)
        rows = connection.execute(
            f"""SELECT p.ordinal, p.point_id, p.class_id, p.class_name,
                       CASE WHEN r.point_id IS NULL THEN p.reviewed
                            WHEN COALESCE(r.disposition, '') = '' THEN 0 ELSE 1 END
                            AS effective_reviewed,
                       p.quality_review_candidate,
                       p.is_wrong_class_candidate, p.is_rough_outlier_candidate,
                       p.is_close_overlap_candidate, p.is_dual_bbox_conflict,
                       p.tiny_object, p.low_source_detail,
                       p.review_priority_score, p.wrong_class_suspicion,
                       p.proposed_class, p.display_rank, q.x, q.y,
                       {identity_select},
                       {review_bucket} AS review_bucket,
                       {priority_sort} AS priority_sort
                FROM point_projections q
                JOIN points_core p ON p.point_id = q.point_id
                LEFT JOIN review_state r ON r.point_id = p.point_id
                WHERE {page_where}
                ORDER BY review_bucket ASC, priority_sort ASC,
                    p.display_rank ASC,
                    p.ordinal ASC
                LIMIT ?""",
            [*page_parameters, row_limit + 1],
        ).fetchall()
    has_more = len(rows) > row_limit
    page_rows = rows[:row_limit]
    columns = {
        "ordinal": [int(row["ordinal"]) for row in page_rows],
        "point_id": [str(row["point_id"]) for row in page_rows],
        "x": [float(row["x"]) for row in page_rows],
        "y": [float(row["y"]) for row in page_rows],
        "class_id": [str(row["class_id"] or "") for row in page_rows],
        "class_name": [str(row["class_name"] or "") for row in page_rows],
        "reviewed": [bool(row["effective_reviewed"]) for row in page_rows],
        "quality_review_candidate": [bool(row["quality_review_candidate"]) for row in page_rows],
        "wrong_class_candidate": [bool(row["is_wrong_class_candidate"]) for row in page_rows],
        "spatial_evidence_candidate": [bool(row["is_rough_outlier_candidate"]) for row in page_rows],
        "overlap_candidate": [
            bool(row["is_close_overlap_candidate"] or row["is_dual_bbox_conflict"])
            for row in page_rows
        ],
        "tiny_object": [bool(row["tiny_object"] or row["low_source_detail"]) for row in page_rows],
        "review_priority_score": [
            _optional_float(row["review_priority_score"]) for row in page_rows
        ],
        "wrong_class_suspicion": [
            _optional_float(row["wrong_class_suspicion"]) for row in page_rows
        ],
        "proposed_class": [str(row["proposed_class"] or "") for row in page_rows],
        "annotation_entity_id": [
            str(row["annotation_entity_id"] or "") for row in page_rows
        ],
        "annotation_entity_revision": [
            _integer(row["annotation_entity_revision"], 0) for row in page_rows
        ],
        "annotation_entity_record_revision": [
            str(row["annotation_entity_record_revision"] or "") for row in page_rows
        ],
        "annotation_source_identity": [
            str(row["annotation_source_identity"] or "") for row in page_rows
        ],
        "source_record_key": [str(row["source_record_key"] or "") for row in page_rows],
        "identity_status": [str(row["identity_status"] or "") for row in page_rows],
        "annotation_attestation": [
            str(row["annotation_attestation"] or "") for row in page_rows
        ],
    }
    result = {
        "schema": "class-analysis-graph-v2",
        "projection_mode": mode,
        "available_projection_modes": modes,
        "session_summary": metadata.get("summary") or {},
        "class_counts": metadata.get("class_counts") or {},
        "point_count": int(metadata.get("point_count") or 0),
        "evidence_count": int(metadata.get("evidence_count") or 0),
        "total_matching": total,
        "returned": len(page_rows),
        "truncated": has_more or bool(cursor),
        "limit": row_limit,
        "next_cursor": (
            _encode_graph_cursor(query_key, page_rows[-1])
            if has_more and page_rows
            else None
        ),
        "review_state_version": review_version,
        "columns": columns,
    }
    return _bounded_payload(result, GRAPH_MAX_BYTES, "class_analysis_graph_payload_too_large")


def _encode_cursor(category: str, rank: int, point_id: str) -> str:
    payload = _json_bytes([category, int(rank), point_id])
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(cursor: Optional[str], category: str) -> tuple[int, str]:
    if not cursor:
        return (0, "")
    try:
        padded = str(cursor) + "=" * (-len(str(cursor)) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        if not isinstance(value, list) or len(value) != 3 or value[0] != category:
            raise ValueError
        return (_integer(value[1]), str(value[2]))
    except Exception as exc:  # noqa: BLE001
        raise SessionStoreError("class_analysis_review_queue_cursor_invalid") from exc


def get_class_analysis_review_queue_payload(
    path: Path | str,
    *,
    category: str = "review",
    cursor: Optional[str] = None,
    limit: int = QUEUE_DEFAULT_ROWS,
) -> dict[str, Any]:
    page_limit = _integer(limit, QUEUE_DEFAULT_ROWS)
    if page_limit < 1 or page_limit > QUEUE_MAX_ROWS:
        raise SessionStoreError("class_analysis_review_queue_limit_invalid")
    queue_category = str(category or "review")
    after_rank, after_point_id = _decode_cursor(cursor, queue_category)
    with _open_readonly(path) as connection:
        category_exists = connection.execute(
            "SELECT 1 FROM review_queue WHERE category = ? LIMIT 1", (queue_category,)
        ).fetchone()
        if category_exists is None:
            known = {
                str(row[0])
                for row in connection.execute("SELECT DISTINCT category FROM review_queue")
            }
            if queue_category not in known and queue_category not in {
                "review",
                "wrong_class",
                "spatial_evidence",
                "overlap",
                "tiny_object",
            }:
                raise SessionStoreError("class_analysis_review_queue_category_invalid")
        total = int(
            connection.execute(
                """SELECT COUNT(*) FROM review_queue q
                   JOIN points_core p ON p.point_id = q.point_id
                   LEFT JOIN review_state r ON r.point_id = p.point_id
                   WHERE q.category = ?
                     AND (CASE WHEN r.point_id IS NULL THEN p.reviewed
                               WHEN COALESCE(r.disposition, '') = '' THEN 0 ELSE 1 END) = 0""",
                (queue_category,),
            ).fetchone()[0]
        )
        identity_select = _point_identity_select_sql(connection)
        rows = connection.execute(
            f"""SELECT q.rank, q.score, p.*,
                      CASE WHEN r.point_id IS NULL THEN p.reviewed
                           WHEN COALESCE(r.disposition, '') = '' THEN 0 ELSE 1 END
                           AS effective_reviewed,
                      {identity_select}
               FROM review_queue q
               JOIN points_core p ON p.point_id = q.point_id
               LEFT JOIN review_state r ON r.point_id = p.point_id
               WHERE q.category = ?
                 AND (CASE WHEN r.point_id IS NULL THEN p.reviewed
                           WHEN COALESCE(r.disposition, '') = '' THEN 0 ELSE 1 END) = 0
                 AND (q.rank > ? OR (q.rank = ? AND q.point_id > ?))
               ORDER BY q.rank ASC, q.point_id ASC
               LIMIT ?""",
            (queue_category, after_rank, after_rank, after_point_id, page_limit + 1),
        ).fetchall()
    has_more = len(rows) > page_limit
    page_rows = rows[:page_limit]
    items = []
    for row in page_rows:
        bbox = [row["x1"], row["y1"], row["x2"], row["y2"]]
        items.append(
            {
                "rank": int(row["rank"]),
                "point_id": str(row["point_id"]),
                "score": _finite_float(row["score"]),
                "split": str(row["split"] or ""),
                "image_relpath": str(row["image_relpath"] or ""),
                "frontend_image_key": str(row["frontend_image_key"] or ""),
                "class_id": str(row["class_id"] or ""),
                "class_name": str(row["class_name"] or ""),
                "bbox_xyxy": bbox,
                "width": _optional_float(row["width"]),
                "height": _optional_float(row["height"]),
                "proposed_class": str(row["proposed_class"] or ""),
                "reviewed": bool(row["effective_reviewed"]),
                "tiny_object": bool(row["tiny_object"] or row["low_source_detail"]),
                "review_object_key": str(row["review_object_key"] or ""),
                "pair_review_key": str(row["pair_review_key"] or ""),
                "annotation_entity_id": str(row["annotation_entity_id"] or ""),
                "annotation_entity_revision": _integer(
                    row["annotation_entity_revision"], 0
                ),
                "annotation_entity_record_revision": str(
                    row["annotation_entity_record_revision"] or ""
                ),
                "annotation_source_identity": str(
                    row["annotation_source_identity"] or ""
                ),
                "source_record_key": str(row["source_record_key"] or ""),
                "identity_status": str(row["identity_status"] or ""),
                "annotation_attestation": str(row["annotation_attestation"] or ""),
            }
        )
    next_cursor = None
    if has_more and page_rows:
        last = page_rows[-1]
        next_cursor = _encode_cursor(
            queue_category, int(last["rank"]), str(last["point_id"])
        )
    result = {
        "schema": "class-analysis-review-queue-v2",
        "category": queue_category,
        "total": total,
        "items": items,
        "next_cursor": next_cursor,
    }
    return _bounded_payload(result, QUEUE_MAX_BYTES, "class_analysis_review_queue_payload_too_large")


def get_class_analysis_review_history_payload(
    path: Path | str,
    *,
    projection_mode: Optional[str] = None,
    limit: int = REVIEW_HISTORY_DEFAULT_ROWS,
) -> dict[str, Any]:
    """Return durable review receipts independently from graph visibility."""

    row_limit = _integer(limit, REVIEW_HISTORY_DEFAULT_ROWS)
    if row_limit < 1 or row_limit > REVIEW_HISTORY_MAX_ROWS:
        raise SessionStoreError("class_analysis_review_history_limit_invalid")
    metadata = read_session_store_metadata(path)
    modes = [str(mode) for mode in metadata.get("projection_modes") or []]
    selected_mode = _mode_name(
        projection_mode or metadata.get("selected_projection_mode")
    )
    if selected_mode not in modes:
        raise SessionStoreError(
            "class_analysis_projection_mode_unavailable",
            status_code=409,
        )
    with _open_readonly(path) as connection:
        total = int(
            connection.execute(
                """SELECT COUNT(*) FROM review_state
                   WHERE COALESCE(disposition, '') <> ''"""
            ).fetchone()[0]
        )
        rows = connection.execute(
            """SELECT r.point_id, r.disposition, r.revision, r.reviewed_at,
                      r.payload AS review_payload,
                      d.payload AS detail_payload
               FROM review_state r
               JOIN point_details d ON d.point_id = r.point_id
               WHERE COALESCE(r.disposition, '') <> ''
               ORDER BY r.rowid DESC, r.point_id ASC
               LIMIT ?""",
            (row_limit,),
        ).fetchall()
        point_ids = [str(row["point_id"]) for row in rows]
        projection_rows = (
            connection.execute(
                f"""SELECT point_id, mode, x, y
                    FROM point_projections
                    WHERE point_id IN ({','.join('?' for _ in point_ids)})""",
                point_ids,
            ).fetchall()
            if point_ids
            else []
        )
    projections: dict[str, dict[str, list[float]]] = {}
    for row in projection_rows:
        projections.setdefault(str(row["point_id"]), {})[str(row["mode"])] = [
            float(row["x"]),
            float(row["y"]),
        ]
    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            point = json.loads(str(row["detail_payload"] or "{}"))
        except json.JSONDecodeError:
            point = {}
        if not isinstance(point, dict):
            point = {}
        try:
            receipt = json.loads(str(row["review_payload"] or "{}"))
        except json.JSONDecodeError:
            receipt = {}
        receipt = receipt if isinstance(receipt, dict) else {}
        point_id = str(row["point_id"])
        point_coordinates = projections.get(point_id, {})
        point.update(
            {
                "point_id": point_id,
                "reviewed": True,
                "human_review_disposition": str(row["disposition"] or ""),
                "human_review_revision": str(row["revision"] or ""),
                "human_reviewed_at": str(row["reviewed_at"] or ""),
                "human_review_origin": str(receipt.get("origin") or ""),
                "human_review_persistence": "durable",
                "_bounded_projection_coordinates": point_coordinates,
                "projection": point_coordinates.get(selected_mode, [0.0, 0.0]),
            }
        )
        items.append(point)
    return _bounded_payload(
        {
            "schema": "class-analysis-review-history-v1",
            "projection_mode": selected_mode,
            "total": total,
            "returned": len(items),
            "truncated": total > len(items),
            "items": items,
        },
        REVIEW_HISTORY_MAX_BYTES,
        "class_analysis_review_history_payload_too_large",
    )


def get_class_analysis_point_detail_payload(
    path: Path | str, point_id: str
) -> dict[str, Any]:
    identifier = str(point_id or "").strip()
    with _open_readonly(path) as connection:
        row = connection.execute(
            "SELECT payload FROM point_details WHERE point_id = ?", (identifier,)
        ).fetchone()
    if row is None:
        raise SessionStoreError("class_analysis_point_not_found", status_code=404)
    result = {
        "schema": "class-analysis-point-detail-v2",
        "point_id": identifier,
        "point": json.loads(str(row["payload"])),
    }
    return _bounded_payload(result, DETAIL_MAX_BYTES, "class_analysis_point_detail_too_large")


def get_class_analysis_point_evidence_payload(
    path: Path | str, point_id: str
) -> dict[str, Any]:
    identifier = str(point_id or "").strip()
    with _open_readonly(path) as connection:
        row = connection.execute(
            "SELECT status, payload, sidecar_row, fingerprint FROM evidence WHERE point_id = ?",
            (identifier,),
        ).fetchone()
    if row is None:
        raise SessionStoreError("class_analysis_point_evidence_not_found", status_code=404)
    try:
        payload = zlib.decompress(bytes(row["payload"]))
    except zlib.error as exc:
        raise SessionStoreError(
            "class_analysis_point_evidence_corrupt", status_code=409
        ) from exc
    if len(payload) > EVIDENCE_MAX_BYTES:
        raise SessionStoreError("class_analysis_point_evidence_too_large", status_code=413)
    evidence = json.loads(payload.decode("utf-8"))
    if isinstance(evidence, dict):
        evidence.pop("_artifact", None)
    result = {
        "schema": "class-analysis-point-evidence-v2",
        "point_id": identifier,
        "status": str(row["status"]),
        "sidecar_row": int(row["sidecar_row"]),
        "fingerprint": str(row["fingerprint"] or ""),
        "evidence": evidence,
    }
    return _bounded_payload(result, EVIDENCE_MAX_BYTES, "class_analysis_point_evidence_too_large")


def _upgrade_evidence_task_schema(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(evidence_tasks)")
    }
    if "candidate_payload" not in columns:
        connection.execute(
            "ALTER TABLE evidence_tasks ADD COLUMN candidate_payload TEXT NOT NULL DEFAULT '{}'"
        )
    if "lease_token" not in columns:
        connection.execute("ALTER TABLE evidence_tasks ADD COLUMN lease_token TEXT")
    if "lease_owner" not in columns:
        connection.execute("ALTER TABLE evidence_tasks ADD COLUMN lease_owner TEXT")
    if "lease_expires_at" not in columns:
        connection.execute("ALTER TABLE evidence_tasks ADD COLUMN lease_expires_at REAL")


def get_class_analysis_evidence_candidates(path: Path | str) -> list[dict[str, Any]]:
    """Return the frozen candidate set used to prepare one session context."""

    connection = sqlite3.connect(str(Path(path)))
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        _upgrade_evidence_task_schema(connection)
        rows = connection.execute(
            """SELECT candidate_payload FROM evidence_tasks
               ORDER BY priority ASC, point_id ASC"""
        ).fetchall()
    finally:
        connection.close()
    candidates = []
    for row in rows:
        try:
            value = json.loads(str(row["candidate_payload"] or "{}"))
        except json.JSONDecodeError:
            value = {}
        if isinstance(value, dict) and str(value.get("point_id") or "").strip():
            candidates.append(value)
    return candidates


def claim_class_analysis_evidence_worker(
    path: Path | str,
    *,
    owner_id: str,
    lease_seconds: float = 120.0,
) -> bool:
    owner = str(owner_id or "").strip()
    if not owner:
        raise SessionStoreError("class_analysis_evidence_worker_owner_required")
    now = time.time()
    connection = sqlite3.connect(str(Path(path)))
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT value FROM session_meta WHERE key = 'evidence_worker_lease'"
        ).fetchone()
        lease: dict[str, Any] = {}
        if row is not None:
            try:
                parsed = json.loads(str(row[0]))
                if isinstance(parsed, dict):
                    lease = parsed
            except json.JSONDecodeError:
                lease = {}
        active_owner = str(lease.get("owner_id") or "")
        active_until = _finite_float(lease.get("expires_at"), 0.0)
        if active_owner and active_owner != owner and active_until > now:
            connection.rollback()
            return False
        connection.execute(
            "INSERT OR REPLACE INTO session_meta(key, value) VALUES (?, ?)",
            (
                "evidence_worker_lease",
                _json_text(
                    {
                        "owner_id": owner,
                        "expires_at": now + max(30.0, float(lease_seconds)),
                        "updated_at": now,
                    }
                ),
            ),
        )
        connection.commit()
        return True
    finally:
        connection.close()


def heartbeat_class_analysis_evidence_worker(
    path: Path | str,
    *,
    owner_id: str,
    lease_seconds: float = 120.0,
) -> bool:
    owner = str(owner_id or "").strip()
    connection = sqlite3.connect(str(Path(path)))
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT value FROM session_meta WHERE key = 'evidence_worker_lease'"
        ).fetchone()
        try:
            lease = json.loads(str(row[0])) if row is not None else {}
        except json.JSONDecodeError:
            lease = {}
        if not isinstance(lease, dict) or str(lease.get("owner_id") or "") != owner:
            connection.rollback()
            return False
        now = time.time()
        connection.execute(
            "UPDATE session_meta SET value = ? WHERE key = 'evidence_worker_lease'",
            (
                _json_text(
                    {
                        "owner_id": owner,
                        "expires_at": now + max(30.0, float(lease_seconds)),
                        "updated_at": now,
                    }
                ),
            ),
        )
        connection.commit()
        return True
    finally:
        connection.close()


def release_class_analysis_evidence_worker(
    path: Path | str,
    *,
    owner_id: str,
) -> bool:
    owner = str(owner_id or "").strip()
    connection = sqlite3.connect(str(Path(path)))
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT value FROM session_meta WHERE key = 'evidence_worker_lease'"
        ).fetchone()
        try:
            lease = json.loads(str(row[0])) if row is not None else {}
        except json.JSONDecodeError:
            lease = {}
        if not isinstance(lease, dict) or str(lease.get("owner_id") or "") != owner:
            connection.rollback()
            return False
        connection.execute(
            "DELETE FROM session_meta WHERE key = 'evidence_worker_lease'"
        )
        connection.commit()
        return True
    finally:
        connection.close()


def initialize_class_analysis_evidence_tasks(
    path: Path | str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Create durable pending tasks without replacing completed evidence."""

    store = Path(path)
    connection = sqlite3.connect(str(store))
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        _upgrade_evidence_task_schema(connection)
        now = time.time()
        rows = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                continue
            point_id = str(candidate.get("point_id") or "").strip()
            if not point_id:
                continue
            source_key = str(
                candidate.get("source_key")
                or candidate.get("frontend_image_key")
                or candidate.get("image_relpath")
                or ""
            )
            rows.append(
                (
                    point_id,
                    index + 1,
                    "pending",
                    0,
                    None,
                    source_key,
                    _json_text(dict(candidate)),
                    None,
                    now,
                )
            )
        connection.executemany(
            """INSERT OR IGNORE INTO evidence_tasks
               (point_id, priority, status, attempts, error, source_key,
                candidate_payload, lease_token, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        connection.execute(
            """UPDATE evidence_tasks
               SET status = 'completed', error = NULL, lease_token = NULL,
                   lease_owner = NULL, lease_expires_at = NULL,
                   updated_at = ?
               WHERE point_id IN (SELECT point_id FROM evidence)""",
            (now,),
        )
        connection.commit()
        counts = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM evidence_tasks GROUP BY status"
            ).fetchall()
        )
        return {str(key): int(value) for key, value in counts.items()}
    finally:
        connection.close()


def claim_class_analysis_evidence_batch(
    path: Path | str,
    *,
    limit: int = 36,
    lease_seconds: float = 1800.0,
    max_attempts: int = 3,
    owner_id: str = "",
) -> dict[str, Any]:
    """Lease one deterministic source-grouped batch for a single worker."""

    batch_limit = max(1, min(256, int(limit)))
    now = time.time()
    lease_duration = max(30.0, float(lease_seconds))
    owner = str(owner_id or f"legacy:{os.getpid()}").strip()
    lease_token = uuid.uuid4().hex
    connection = sqlite3.connect(str(Path(path)))
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        _upgrade_evidence_task_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """UPDATE evidence_tasks
               SET status = CASE WHEN attempts < ? THEN 'retry' ELSE 'failed' END,
                   lease_token = NULL, lease_owner = NULL, lease_expires_at = NULL,
                   error = COALESCE(error, 'worker_lease_expired'),
                   updated_at = ?
               WHERE status = 'processing'
                 AND COALESCE(lease_expires_at, updated_at + ?) < ?""",
            (int(max_attempts), now, lease_duration, now),
        )
        candidates = connection.execute(
            """SELECT point_id, priority, status, attempts, source_key,
                      candidate_payload
               FROM evidence_tasks
               WHERE status IN ('pending', 'retry') AND attempts < ?
               ORDER BY priority ASC, source_key ASC, point_id ASC
               LIMIT ?""",
            (int(max_attempts), batch_limit * 8),
        ).fetchall()
        selected: list[sqlite3.Row] = []
        if candidates:
            source_order: list[str] = []
            by_source: dict[str, list[sqlite3.Row]] = {}
            for row in candidates:
                source_key = str(row["source_key"] or "")
                if source_key not in by_source:
                    source_order.append(source_key)
                    by_source[source_key] = []
                by_source[source_key].append(row)
            for source_key in source_order:
                selected.extend(by_source[source_key])
                if len(selected) >= batch_limit:
                    selected = selected[:batch_limit]
                    break
        if selected:
            connection.executemany(
                """UPDATE evidence_tasks
                   SET status = 'processing', attempts = attempts + 1,
                       error = NULL, lease_token = ?, lease_owner = ?,
                       lease_expires_at = ?, updated_at = ?
                   WHERE point_id = ? AND status IN ('pending', 'retry')""",
                [
                    (
                        lease_token,
                        owner,
                        now + lease_duration,
                        now,
                        str(row["point_id"]),
                    )
                    for row in selected
                ],
            )
        connection.commit()
        items = []
        for row in selected:
            try:
                candidate = json.loads(str(row["candidate_payload"] or "{}"))
            except json.JSONDecodeError:
                candidate = {}
            items.append(
                {
                    "point_id": str(row["point_id"]),
                    "priority": int(row["priority"]),
                    "attempt": int(row["attempts"]) + 1,
                    "source_key": str(row["source_key"] or ""),
                    "candidate": candidate if isinstance(candidate, dict) else {},
                }
            )
        return {
            "lease_token": lease_token if items else "",
            "lease_owner": owner if items else "",
            "items": items,
        }
    finally:
        connection.close()


def complete_class_analysis_evidence_batch(
    path: Path | str,
    *,
    lease_token: str,
    owner_id: str = "",
    rows: Sequence[Mapping[str, Any]],
) -> int:
    token = str(lease_token or "").strip()
    if not token:
        raise SessionStoreError("class_analysis_evidence_lease_required")
    owner = str(owner_id or "").strip()
    now = time.time()
    connection = sqlite3.connect(str(Path(path)))
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        _upgrade_evidence_task_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        completed = 0
        for row in rows:
            point_id = str(row.get("point_id") or "").strip()
            evidence = row.get("evidence")
            if not point_id or not isinstance(evidence, Mapping):
                continue
            task = connection.execute(
                """SELECT 1 FROM evidence_tasks
                   WHERE point_id = ? AND status = 'processing' AND lease_token = ?
                     AND (? = '' OR lease_owner = ?)""",
                (point_id, token, owner, owner),
            ).fetchone()
            if task is None:
                raise SessionStoreError("class_analysis_evidence_lease_stale", status_code=409)
            evidence_bytes = _json_bytes(dict(evidence))
            if len(evidence_bytes) > EVIDENCE_MAX_BYTES:
                raise SessionStoreError(
                    f"class_analysis_point_evidence_too_large:{point_id}",
                    status_code=413,
                )
            connection.execute(
                """INSERT OR REPLACE INTO evidence
                   (point_id, status, payload, sidecar_row, fingerprint, updated_at)
                   VALUES (?, 'completed', ?, ?, ?, ?)""",
                (
                    point_id,
                    sqlite3.Binary(zlib.compress(evidence_bytes, level=6)),
                    _integer(evidence.get("sidecar_row"), -1),
                    str(row.get("fingerprint") or ""),
                    now,
                ),
            )
            connection.execute(
                """UPDATE evidence_tasks
                   SET status = 'completed', error = NULL, lease_token = NULL,
                       lease_owner = NULL, lease_expires_at = NULL,
                       updated_at = ?
                   WHERE point_id = ? AND lease_token = ?
                     AND (? = '' OR lease_owner = ?)""",
                (now, point_id, token, owner, owner),
            )
            completed += 1
        connection.commit()
        return completed
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def fail_class_analysis_evidence_batch(
    path: Path | str,
    *,
    lease_token: str,
    owner_id: str = "",
    error: str,
    max_attempts: int = 3,
) -> int:
    connection = sqlite3.connect(str(Path(path)))
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        _upgrade_evidence_task_schema(connection)
        now = time.time()
        owner = str(owner_id or "").strip()
        cursor = connection.execute(
            """UPDATE evidence_tasks
               SET status = CASE WHEN attempts < ? THEN 'retry' ELSE 'failed' END,
                   error = ?, lease_token = NULL, lease_owner = NULL,
                   lease_expires_at = NULL, updated_at = ?
               WHERE status = 'processing' AND lease_token = ?
                 AND (? = '' OR lease_owner = ?)""",
            (
                int(max_attempts),
                str(error or "evidence_batch_failed")[:1000],
                now,
                str(lease_token),
                owner,
                owner,
            ),
        )
        connection.commit()
        return max(0, int(cursor.rowcount))
    finally:
        connection.close()


def promote_class_analysis_evidence_task(path: Path | str, point_id: str) -> bool:
    connection = sqlite3.connect(str(Path(path)))
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        _upgrade_evidence_task_schema(connection)
        cursor = connection.execute(
            """UPDATE evidence_tasks SET priority = -1, updated_at = ?
               WHERE point_id = ? AND status IN ('pending', 'retry')""",
            (time.time(), str(point_id or "").strip()),
        )
        connection.commit()
        return int(cursor.rowcount) > 0
    finally:
        connection.close()


def class_analysis_evidence_progress(path: Path | str) -> dict[str, Any]:
    with _open_readonly(path) as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM evidence_tasks GROUP BY status"
        ).fetchall()
        first_update = connection.execute(
            "SELECT MIN(updated_at), MAX(updated_at) FROM evidence_tasks"
        ).fetchone()
    counts = {str(row["status"]): int(row["count"]) for row in rows}
    total = sum(counts.values())
    completed = counts.get("completed", 0)
    return {
        "total": total,
        "completed": completed,
        "failed": counts.get("failed", 0),
        "pending": counts.get("pending", 0) + counts.get("retry", 0),
        "processing": counts.get("processing", 0),
        "counts": counts,
        "started_at": float(first_update[0] or 0) if first_update else 0.0,
        "updated_at": float(first_update[1] or 0) if first_update else 0.0,
    }


def get_class_analysis_internal_evidence(
    path: Path | str, point_id: str
) -> Optional[dict[str, Any]]:
    with _open_readonly(path) as connection:
        row = connection.execute(
            "SELECT payload FROM evidence WHERE point_id = ? AND status = 'completed'",
            (str(point_id or "").strip(),),
        ).fetchone()
    if row is None:
        return None
    try:
        payload = zlib.decompress(bytes(row["payload"]))
        value = json.loads(payload.decode("utf-8"))
    except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionStoreError("class_analysis_point_evidence_corrupt", status_code=409) from exc
    return value if isinstance(value, dict) else None


def get_class_analysis_evidence_task(
    path: Path | str,
    point_id: str,
) -> Optional[dict[str, Any]]:
    """Return one task's durable state without exposing its candidate payload."""

    identifier = str(point_id or "").strip()
    if not identifier:
        return None
    with _open_readonly(path) as connection:
        row = connection.execute(
            """
            SELECT point_id, status, priority, attempts AS attempt_count,
                   source_key, error AS last_error, lease_owner,
                   lease_expires_at, updated_at
            FROM evidence_tasks
            WHERE point_id = ?
            """,
            (identifier,),
        ).fetchone()
    return dict(row) if row is not None else None


def heartbeat_class_analysis_evidence_lease(
    path: Path | str,
    *,
    lease_token: str,
    owner_id: str,
    lease_seconds: float = 1800.0,
) -> int:
    connection = sqlite3.connect(str(Path(path)))
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        _upgrade_evidence_task_schema(connection)
        now = time.time()
        cursor = connection.execute(
            """UPDATE evidence_tasks
               SET lease_expires_at = ?, updated_at = ?
               WHERE status = 'processing' AND lease_token = ? AND lease_owner = ?""",
            (
                now + max(30.0, float(lease_seconds)),
                now,
                str(lease_token or ""),
                str(owner_id or ""),
            ),
        )
        connection.commit()
        return max(0, int(cursor.rowcount or 0))
    finally:
        connection.close()


def release_class_analysis_evidence_leases(
    path: Path | str,
    *,
    owner_id: str = "",
    expired_only: bool = True,
) -> int:
    """Release only this worker's leases, or leases proven expired."""

    connection = sqlite3.connect(str(Path(path)))
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        _upgrade_evidence_task_schema(connection)
        now = time.time()
        owner = str(owner_id or "").strip()
        if owner:
            predicate = "status = 'processing' AND lease_owner = ?"
            parameters: tuple[Any, ...] = (now, owner)
        elif expired_only:
            predicate = (
                "status = 'processing' "
                "AND COALESCE(lease_expires_at, updated_at) < ?"
            )
            parameters = (now, now)
        else:
            raise SessionStoreError("class_analysis_evidence_lease_owner_required")
        cursor = connection.execute(
            f"""UPDATE evidence_tasks
                SET status = 'retry', lease_token = NULL, lease_owner = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE {predicate}""",
            parameters,
        )
        connection.commit()
        return max(0, int(cursor.rowcount or 0))
    finally:
        connection.close()


def get_class_analysis_qwen_context(
    path: Path | str,
    point_id: str,
    *,
    same_source_limit: int = 256,
    per_class_limit: int = 8,
) -> dict[str, Any]:
    identifier = str(point_id or "").strip()
    metadata = read_session_store_metadata(path)
    with _open_readonly(path) as connection:
        target = connection.execute(
            "SELECT source_key, class_name, proposed_class FROM points_core WHERE point_id = ?",
            (identifier,),
        ).fetchone()
        if target is None:
            raise SessionStoreError("class_analysis_point_not_found", status_code=404)
        ids = [identifier]
        ids.extend(
            str(row[0])
            for row in connection.execute(
                """SELECT point_id FROM points_core
                   WHERE source_key = ? AND point_id <> ?
                   ORDER BY ordinal LIMIT ?""",
                (str(target["source_key"] or ""), identifier, int(same_source_limit)),
            )
        )
        class_names = [
            str(name)
            for name in (metadata.get("class_counts") or {})
            if str(name)
        ]
        for class_name in class_names:
            ids.extend(
                str(row[0])
                for row in connection.execute(
                    """SELECT point_id FROM points_core
                       WHERE class_name = ? AND point_id <> ?
                       ORDER BY COALESCE(review_priority_score, 0) DESC,
                                display_rank ASC LIMIT ?""",
                    (class_name, identifier, int(per_class_limit)),
                )
            )
        unique_ids = list(dict.fromkeys(ids))
        points = []
        for offset in range(0, len(unique_ids), 500):
            chunk = unique_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT point_id, payload FROM point_details WHERE point_id IN ({placeholders})",
                chunk,
            ).fetchall()
            by_id = {str(row["point_id"]): json.loads(str(row["payload"])) for row in rows}
            points.extend(by_id[pid] for pid in chunk if pid in by_id)
    evidence = get_class_analysis_internal_evidence(path, identifier)
    for point in points:
        if str(point.get("point_id") or "") == identifier and evidence is not None:
            public_evidence = dict(evidence)
            artifact = public_evidence.pop("_artifact", None)
            point["refined_outlier"] = public_evidence
            if isinstance(artifact, Mapping):
                point["_evidence_artifact"] = dict(artifact)
            break
    summary = dict(metadata.get("summary") or {})
    summary.setdefault("class_counts", metadata.get("class_counts") or {})
    summary.setdefault("object_count", metadata.get("point_count") or 0)
    return {"summary": summary, "points": points}
