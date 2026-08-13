"""Bounded, normalized persistence for recoverable class-analysis sessions."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


SESSION_STORE_SCHEMA = "class-analysis-session-store-v2"
SESSION_STORE_FILENAME = "session.sqlite3"
GRAPH_DEFAULT_ROWS = 50_000
GRAPH_MAX_ROWS = 300_000
GRAPH_MAX_BYTES = 64 * 1024 * 1024
QUEUE_DEFAULT_ROWS = 36
QUEUE_MAX_ROWS = 100
QUEUE_MAX_BYTES = 4 * 1024 * 1024
DETAIL_MAX_BYTES = 256 * 1024
EVIDENCE_MAX_BYTES = 2 * 1024 * 1024


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
        display_rank INTEGER NOT NULL
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
        updated_at REAL NOT NULL
    );
    """


def _flush_rows(
    connection: sqlite3.Connection,
    buffers: dict[str, list[tuple[Any, ...]]],
) -> None:
    statements = {
        "points": "INSERT INTO points_core VALUES ("
        + ",".join("?" for _ in range(33))
        + ")",
        "projections": "INSERT OR REPLACE INTO point_projections VALUES (?,?,?,?)",
        "details": "INSERT INTO point_details VALUES (?,?)",
        "queues": "INSERT OR REPLACE INTO review_queue VALUES (?,?,?,?,?,?)",
        "overlaps": "INSERT OR REPLACE INTO overlap_pairs VALUES (?,?,?,?)",
        "reviews": "INSERT OR REPLACE INTO review_state VALUES (?,?,?,?,?)",
        "evidence": "INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?,?)",
        "tasks": "INSERT OR REPLACE INTO evidence_tasks VALUES (?,?,?,?,?,?,?)",
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
        os.replace(partial, target)
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


def _open_readonly(path: Path | str) -> sqlite3.Connection:
    store = Path(path)
    if not store.is_file():
        raise SessionStoreError("class_analysis_session_store_not_found", status_code=404)
    connection = sqlite3.connect(f"file:{store.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def read_session_store_metadata(path: Path | str) -> dict[str, Any]:
    with _open_readonly(path) as connection:
        rows = connection.execute("SELECT key, value FROM session_meta").fetchall()
    metadata = {str(row["key"]): json.loads(str(row["value"])) for row in rows}
    if metadata.get("schema") != SESSION_STORE_SCHEMA:
        raise SessionStoreError("class_analysis_session_store_schema_invalid", status_code=409)
    return metadata


def validate_class_analysis_session_store(
    path: Path | str,
    *,
    expected_point_count: Optional[int] = None,
    expected_evidence_count: Optional[int] = None,
) -> dict[str, Any]:
    with _open_readonly(path) as connection:
        check = connection.execute("PRAGMA quick_check").fetchone()
        point_count = int(connection.execute("SELECT COUNT(*) FROM points_core").fetchone()[0])
        evidence_count = int(connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0])
        projection_count = int(
            connection.execute("SELECT COUNT(*) FROM point_projections").fetchone()[0]
        )
        distinct_points = int(
            connection.execute("SELECT COUNT(DISTINCT point_id) FROM points_core").fetchone()[0]
        )
    if not check or str(check[0]).lower() != "ok":
        raise SessionStoreError("class_analysis_session_store_integrity_failed")
    if point_count != distinct_points:
        raise SessionStoreError("class_analysis_session_point_identity_collision")
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
    reviewed_mode = str(reviewed or "any")
    if reviewed_mode == "reviewed":
        clauses.append("p.reviewed = 1")
    elif reviewed_mode == "unreviewed":
        clauses.append("p.reviewed = 0")
    elif reviewed_mode != "any":
        raise SessionStoreError("class_analysis_graph_review_filter_invalid")
    return clauses, values


def get_class_analysis_graph_payload(
    path: Path | str,
    *,
    projection_mode: Optional[str] = None,
    class_name: Optional[str] = None,
    objects: str = "all",
    object_size: str = "all",
    reviewed: str = "any",
    limit: int = GRAPH_DEFAULT_ROWS,
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
        total = int(
            connection.execute(
                f"""SELECT COUNT(*) FROM point_projections q
                    JOIN points_core p ON p.point_id = q.point_id
                    WHERE {where}""",
                parameters,
            ).fetchone()[0]
        )
        rows = connection.execute(
            f"""SELECT p.ordinal, p.point_id, p.class_id, p.class_name,
                       p.reviewed, p.quality_review_candidate,
                       p.is_wrong_class_candidate, p.is_rough_outlier_candidate,
                       p.is_close_overlap_candidate, p.is_dual_bbox_conflict,
                       p.tiny_object, p.low_source_detail,
                       p.review_priority_score, p.wrong_class_suspicion,
                       p.proposed_class, q.x, q.y
                FROM point_projections q
                JOIN points_core p ON p.point_id = q.point_id
                WHERE {where}
                ORDER BY
                    CASE WHEN p.quality_review_candidate = 1
                              OR p.is_wrong_class_candidate = 1
                              OR p.is_rough_outlier_candidate = 1
                              OR p.is_close_overlap_candidate = 1
                         THEN 0 ELSE 1 END,
                    COALESCE(p.review_priority_score, 0) DESC,
                    p.display_rank ASC,
                    p.ordinal ASC
                LIMIT ?""",
            [*parameters, row_limit],
        ).fetchall()
    columns = {
        "ordinal": [int(row["ordinal"]) for row in rows],
        "point_id": [str(row["point_id"]) for row in rows],
        "x": [float(row["x"]) for row in rows],
        "y": [float(row["y"]) for row in rows],
        "class_id": [str(row["class_id"] or "") for row in rows],
        "class_name": [str(row["class_name"] or "") for row in rows],
        "reviewed": [bool(row["reviewed"]) for row in rows],
        "quality_review_candidate": [bool(row["quality_review_candidate"]) for row in rows],
        "wrong_class_candidate": [bool(row["is_wrong_class_candidate"]) for row in rows],
        "spatial_evidence_candidate": [bool(row["is_rough_outlier_candidate"]) for row in rows],
        "overlap_candidate": [
            bool(row["is_close_overlap_candidate"] or row["is_dual_bbox_conflict"])
            for row in rows
        ],
        "tiny_object": [bool(row["tiny_object"] or row["low_source_detail"]) for row in rows],
        "review_priority_score": [
            _optional_float(row["review_priority_score"]) for row in rows
        ],
        "wrong_class_suspicion": [
            _optional_float(row["wrong_class_suspicion"]) for row in rows
        ],
        "proposed_class": [str(row["proposed_class"] or "") for row in rows],
    }
    result = {
        "schema": "class-analysis-graph-v2",
        "projection_mode": mode,
        "available_projection_modes": modes,
        "total_matching": total,
        "returned": len(rows),
        "truncated": total > len(rows),
        "limit": row_limit,
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
                "SELECT COUNT(*) FROM review_queue WHERE category = ?", (queue_category,)
            ).fetchone()[0]
        )
        rows = connection.execute(
            """SELECT q.rank, q.score, p.* FROM review_queue q
               JOIN points_core p ON p.point_id = q.point_id
               WHERE q.category = ?
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
                "reviewed": bool(row["reviewed"]),
                "tiny_object": bool(row["tiny_object"] or row["low_source_detail"]),
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
    result = {
        "schema": "class-analysis-point-evidence-v2",
        "point_id": identifier,
        "status": str(row["status"]),
        "sidecar_row": int(row["sidecar_row"]),
        "fingerprint": str(row["fingerprint"] or ""),
        "evidence": json.loads(payload.decode("utf-8")),
    }
    return _bounded_payload(result, EVIDENCE_MAX_BYTES, "class_analysis_point_evidence_too_large")
