"""Durable management index for existing Tator datasets and review sessions.

Annotation and analysis stores remain authoritative.  This database owns only
management state: names, pins, references, reservations and resumable operations.
"""

from __future__ import annotations

import contextlib
import contextvars
import base64
import errno
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping


API_VERSION = 1
_METADATA_RESERVE_BYTES = 8 * 1024 * 1024


class StorageLibraryError(RuntimeError):
    def __init__(self, code: str, message: str = "", *, status_code: int = 409, **context: Any):
        self.status_code = status_code
        self.detail = {"code": code, "message": message or code, **context}
        super().__init__(message or code)


def encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


def absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def contained(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def reject_symlinks(path: Path) -> None:
    path = absolute(path)
    for candidate in [*reversed(path.parents), path]:
        if candidate.is_symlink():
            raise StorageLibraryError("storage_path_symlink", "Storage path contains a symbolic link.")


class StorageLibrary:
    def __init__(self, root: Path | str, *, roots: Mapping[str, Path | str] | None = None):
        self.root = absolute(root)
        reject_symlinks(self.root)
        if len(self.root.parts) < 3:
            raise StorageLibraryError("storage_root_unsafe")
        self.directory = self.root / "library"
        reject_symlinks(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "catalog.sqlite3"
        reject_symlinks(self.path)
        self.roots = {str(key): absolute(value) for key, value in (roots or {}).items()}
        if self.root not in self.roots.values():
            self.roots["class_analysis"] = self.root
        self._admission_context = contextvars.ContextVar("storage_library_admission", default=None)
        self._lock = threading.RLock()
        self._dirty = threading.Event()
        self._metadata_bootstrapped = False
        self._reserve_path = self.directory / "metadata.reserve"
        self.replenish_metadata_reserve()
        try:
            self._initialise()
        except StorageLibraryError as exc:
            if exc.detail.get("code") != "library_metadata_retry":
                raise
            self._initialise()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=15.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=15000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextlib.contextmanager
    def transaction(self):
        with self._lock:
            try:
                with contextlib.closing(self.connect()) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        yield connection
                        connection.commit()
                    except BaseException:
                        try:
                            connection.rollback()
                        except sqlite3.Error:
                            pass
                        raise
            except (sqlite3.Error, OSError) as exc:
                disk_full = getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL or getattr(exc, "errno", None) in {errno.ENOSPC, errno.EDQUOT}
                if disk_full and self._release_metadata_reserve():
                    raise StorageLibraryError("library_metadata_retry", "The disk filled up. Reserved management space was released; retry this action.", status_code=503) from exc
                raise

    def _release_metadata_reserve(self) -> bool:
        try:
            reject_symlinks(self._reserve_path)
            if not self._reserve_path.is_file():
                return False
            self._reserve_path.unlink()
            return True
        except (OSError, StorageLibraryError):
            return False

    def replenish_metadata_reserve(self) -> None:
        """Keep physically allocated room for durable removal/recovery intent."""
        descriptor = None
        created = False
        try:
            reject_symlinks(self._reserve_path)
            if self._reserve_path.exists():
                return
            volume = os.statvfs(self.directory)
            if volume.f_bavail * volume.f_frsize < _METADATA_RESERVE_BYTES * 4:
                return
            descriptor = os.open(self._reserve_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            created = True
            # Write actual blocks: truncation alone would create a sparse file.
            chunk = bytes(256 * 1024)
            remaining = _METADATA_RESERVE_BYTES
            while remaining:
                written = os.write(descriptor, chunk[:min(remaining, len(chunk))])
                if written <= 0:
                    raise OSError(errno.ENOSPC, "Metadata reserve allocation failed")
                remaining -= written
            os.fsync(descriptor)
        except (OSError, StorageLibraryError):
            if created:
                try:
                    self._reserve_path.unlink()
                except OSError:
                    pass
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def metadata_storage(self) -> dict[str, int]:
        """Stat fixed management files without indexing the catalog into itself."""
        logical = allocated = reserve = 0
        seen = set()
        for path in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm"), self._reserve_path):
            try:
                info = path.lstat()
                if path.is_symlink() or (info.st_dev, info.st_ino) in seen:
                    continue
                seen.add((info.st_dev, info.st_ino))
                logical += info.st_size
                allocated += int(getattr(info, "st_blocks", (info.st_size + 511) // 512)) * 512
                if path == self._reserve_path:
                    reserve = info.st_size
            except OSError:
                continue
        return {"logical_bytes": logical, "allocated_bytes": allocated, "reserve_bytes": reserve}

    def _initialise(self) -> None:
        with self.transaction() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS library_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, authority TEXT NOT NULL,
                    source_revision TEXT NOT NULL, summary TEXT NOT NULL,
                    label_override TEXT, pinned INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'available', revision INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL, last_opened_at REAL,
                    discovered_generation TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS library_items_kind ON items(kind,state,updated_at,id);
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY, root_id TEXT NOT NULL, relpath TEXT NOT NULL,
                    category TEXT NOT NULL, ownership TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'present',
                    generation INTEGER NOT NULL DEFAULT 1, signature TEXT NOT NULL DEFAULT '',
                    logical_bytes INTEGER, allocated_bytes INTEGER, shared_bytes INTEGER NOT NULL DEFAULT 0,
                    file_count INTEGER, measured_at REAL, scan_status TEXT NOT NULL DEFAULT 'unknown',
                    error TEXT, metadata TEXT NOT NULL DEFAULT '{}', UNIQUE(root_id,relpath)
                );
                CREATE TABLE IF NOT EXISTS asset_references (
                    item_id TEXT NOT NULL REFERENCES items(id), asset_id TEXT NOT NULL REFERENCES assets(id),
                    role TEXT NOT NULL, PRIMARY KEY(item_id,asset_id,role)
                );
                CREATE INDEX IF NOT EXISTS library_asset_consumers ON asset_references(asset_id,item_id);
                CREATE TABLE IF NOT EXISTS leases (
                    id TEXT PRIMARY KEY, token TEXT NOT NULL, owner TEXT NOT NULL, purpose TEXT NOT NULL,
                    item_ids TEXT NOT NULL, asset_ids TEXT NOT NULL, paths TEXT NOT NULL,
                    created_at REAL NOT NULL, expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, idempotency_key TEXT UNIQUE,
                    request_hash TEXT NOT NULL, request TEXT NOT NULL, state TEXT NOT NULL,
                    progress TEXT NOT NULL DEFAULT '{}', result TEXT NOT NULL DEFAULT '{}', error TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS operation_items (
                    operation_id TEXT NOT NULL REFERENCES operations(id), asset_id TEXT NOT NULL,
                    descriptor TEXT NOT NULL, quarantine TEXT, phase TEXT NOT NULL DEFAULT 'planned',
                    error TEXT, PRIMARY KEY(operation_id,asset_id)
                );
            """)
            connection.execute("INSERT OR IGNORE INTO library_meta VALUES ('backend_id',?)", (encode(str(uuid.uuid4())),))
            connection.execute("INSERT OR IGNORE INTO library_meta VALUES ('schema',?)", (encode(API_VERSION),))
            connection.execute("INSERT OR IGNORE INTO library_meta VALUES ('discovery',?)", (encode({"status": "pending"}),))
        self.backend_id = str(self.get_meta("backend_id"))

    def envelope(self, **payload: Any) -> dict[str, Any]:
        return {"api_version": API_VERSION, "backend_id": self.backend_id, **payload}

    def get_meta(self, key: str, default: Any = None) -> Any:
        with contextlib.closing(self.connect()) as connection:
            row = connection.execute("SELECT value FROM library_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key: str, value: Any) -> None:
        with self.transaction() as connection:
            connection.execute("INSERT INTO library_meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, encode(value)))

    def invalidate(self, *paths: Path | str, item_ids: Iterable[str] = ()) -> None:
        """Cheap producer hook; reconciliation performs the actual discovery."""
        self._dirty.set()
        targets = [absolute(path) for path in paths]
        item_ids = tuple(item_ids)
        if not targets and not item_ids:
            return
        with self.transaction() as connection:
            rows = connection.execute("SELECT id,root_id,relpath FROM assets WHERE state!='deleted'").fetchall() if targets else []
            for row in rows:
                path = self.asset_path(dict(row))
                if any(contained(path, target) or contained(target, path) for target in targets):
                    connection.execute("UPDATE assets SET scan_status='stale',generation=generation+1 WHERE id=?", (row["id"],))
            for item_id in item_ids:
                connection.execute("UPDATE items SET revision=revision+1,updated_at=? WHERE id=?", (time.time(), str(item_id)))

    def locate(self, path: Path | str) -> tuple[str, str]:
        target = absolute(path)
        matches = [(key, root) for key, root in self.roots.items() if contained(target, root)]
        if not matches:
            raise StorageLibraryError("storage_path_outside_roots")
        key, root = max(matches, key=lambda pair: len(pair[1].parts))
        return key, target.relative_to(root).as_posix()

    def asset_path(self, asset: Mapping[str, Any]) -> Path:
        root = self.roots.get(str(asset["root_id"]))
        if root is None:
            raise StorageLibraryError("storage_root_unavailable")
        relative = Path(str(asset["relpath"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise StorageLibraryError("storage_path_invalid")
        path = absolute(root / relative)
        if not contained(path, root):
            raise StorageLibraryError("storage_path_invalid")
        return path

    def register_asset(self, path: Path | str, *, category: str, ownership: str = "managed", metadata: Mapping[str, Any] | None = None) -> str:
        root_id, relative = self.locate(path)
        metadata = dict(metadata or {})
        if ownership == "external":
            metadata["root_path"] = str(self.roots[root_id])
        identifier = "asset:" + digest([root_id, relative])[:32]
        target = absolute(path)
        try:
            info = target.lstat()
            signature = encode([info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode])
            state = "present"
        except FileNotFoundError:
            signature, state = "missing", "missing"
        except OSError:
            signature, state = "unreadable", "unknown"
        with self.transaction() as connection:
            previous = connection.execute("SELECT * FROM assets WHERE id=?", (identifier,)).fetchone()
            if previous is not None and previous["state"] in {"deleting", "quarantined"}:
                # A known producer explicitly invalidates/reconciles replacement
                # assets; discovery must not resurrect deletion tombstones.
                return identifier
            if previous is not None and previous["state"] == "deleted" and (state != "present" or signature == previous["signature"]):
                return identifier
            connection.execute("""INSERT INTO assets(id,root_id,relpath,category,ownership,state,signature,metadata)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                category=excluded.category,ownership=excluded.ownership,state=excluded.state,
                generation=assets.generation+CASE WHEN assets.signature!=excluded.signature THEN 1 ELSE 0 END,
                scan_status=CASE WHEN assets.signature!=excluded.signature THEN 'stale' ELSE assets.scan_status END,
                signature=excluded.signature,metadata=excluded.metadata""",
                (identifier, root_id, relative, category, ownership, state, signature, encode(metadata)))
        return identifier

    def register_item(self, identifier: str, *, kind: str, authority: Mapping[str, Any], summary: Mapping[str, Any], references: Iterable[tuple[str, str]] = (), source_revision: str = "", generation: str = "") -> dict[str, Any]:
        now = time.time()
        references = list(dict.fromkeys(references))
        encoded_summary, encoded_authority = encode(dict(summary)), encode(dict(authority))
        revision = source_revision or digest([authority, summary])
        with self.transaction() as connection:
            existing = connection.execute("SELECT * FROM items WHERE id=?", (identifier,)).fetchone()
            replace_cache_tombstone = existing and existing["state"] == "deleted" and authority.get("kind") in {"cache", "export", "upload"} and any(
                connection.execute("SELECT 1 FROM assets WHERE id=? AND state='present'", (asset_id,)).fetchone() for asset_id, _ in references
            )
            restored_dataset = existing and (existing["state"] == "trashed" or (existing["state"] == "unlinked" and summary.get("restored_from_trash_id"))) and authority.get("kind") == "dataset" and Path(str(authority.get("path") or "")).is_dir()
            if existing and existing["state"] in {"deleted", "deleting", "analysis_cleared", "unlinked", "trashed"} and not (replace_cache_tombstone or restored_dataset):
                return self._item(existing)
            state = str(summary.get("state") or "available")
            connection.execute("""INSERT INTO items(id,kind,authority,source_revision,summary,state,created_at,updated_at,discovered_generation)
                VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                kind=excluded.kind,authority=excluded.authority,source_revision=excluded.source_revision,
                summary=excluded.summary,state=excluded.state,
                revision=items.revision+CASE WHEN items.source_revision!=excluded.source_revision OR items.summary!=excluded.summary THEN 1 ELSE 0 END,
                updated_at=CASE WHEN items.source_revision!=excluded.source_revision OR items.summary!=excluded.summary THEN excluded.updated_at ELSE items.updated_at END,
                discovered_generation=excluded.discovered_generation""",
                (identifier, kind, encoded_authority, revision, encoded_summary, state, float(summary.get("created_at") or now), now, generation))
            connection.execute("DELETE FROM asset_references WHERE item_id=?", (identifier,))
            connection.executemany("INSERT OR IGNORE INTO asset_references VALUES (?,?,?)", [(identifier, asset_id, role) for asset_id, role in references])
        return self.get_item(identifier)

    @staticmethod
    def _item(row: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        summary = json.loads(item.pop("summary"))
        authority = json.loads(item.pop("authority"))
        result = {**summary, **item, "authority": authority, "pinned": bool(item["pinned"])}
        result["label"] = item.get("label_override") or summary.get("label") or item["id"]
        result.pop("label_override", None)
        result.pop("discovered_generation", None)
        return result

    @staticmethod
    def _asset(row: Mapping[str, Any]) -> dict[str, Any]:
        asset = dict(row)
        asset["metadata"] = json.loads(asset["metadata"])
        return asset

    def get_item(self, identifier: str) -> dict[str, Any]:
        with contextlib.closing(self.connect()) as connection:
            row = connection.execute("SELECT * FROM items WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise StorageLibraryError("library_item_not_found", status_code=404)
        item = self._item(row)
        item["assets"] = self.assets_for_items([identifier])
        item["storage"] = self.item_storage(item["assets"])
        return item

    @staticmethod
    def item_storage(assets: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        rows = list(assets)
        measured = [row for row in rows if row.get("state") != "deleted"]
        return {"logical_bytes": sum(int(row.get("logical_bytes") or 0) for row in measured),
                "allocated_bytes": sum(int(row.get("allocated_bytes") or 0) for row in measured),
                "shared_bytes": sum(int(row.get("logical_bytes") or 0) for row in measured if int(row.get("consumer_count") or 0) > 1),
                "measured_at": min((float(row["measured_at"]) for row in measured if row.get("measured_at")), default=None),
                "status": "complete" if all(row.get("scan_status") == "complete" for row in measured) else "partial",
                "unknown_assets": sum(row.get("logical_bytes") is None for row in measured)}

    def assets_for_items(self, identifiers: Iterable[str]) -> list[dict[str, Any]]:
        identifiers = list(dict.fromkeys(str(value) for value in identifiers))
        if not identifiers:
            return []
        placeholders = ",".join("?" for _ in identifiers)
        with contextlib.closing(self.connect()) as connection:
            rows = connection.execute(f"""SELECT DISTINCT a.*,
                (SELECT COUNT(DISTINCT r.item_id) FROM asset_references r JOIN items i ON i.id=r.item_id
                 WHERE r.asset_id=a.id AND i.state NOT IN ('deleted','unlinked')) consumer_count
                FROM assets a JOIN asset_references r ON r.asset_id=a.id WHERE r.item_id IN ({placeholders})""", identifiers).fetchall()
        return [self._asset(row) for row in rows]

    def asset_consumers(self, asset_id: str) -> list[dict[str, Any]]:
        with contextlib.closing(self.connect()) as connection:
            rows = connection.execute("SELECT DISTINCT i.* FROM items i JOIN asset_references r ON r.item_id=i.id WHERE r.asset_id=? AND i.state NOT IN ('deleted','unlinked')", (asset_id,)).fetchall()
        return [self._item(row) for row in rows]

    def all_assets(self) -> list[dict[str, Any]]:
        with contextlib.closing(self.connect()) as connection:
            rows = connection.execute("SELECT * FROM assets WHERE state!='deleted' ORDER BY root_id,relpath").fetchall()
        return [self._asset(row) for row in rows]

    def list_items(self, *, kind: str = "", q: str = "", cursor: str = "", limit: int = 50, include_deleted: bool = False, parent_id: str = "", source_id: str = "") -> dict[str, Any]:
        limit = max(1, min(100, int(limit)))
        clauses, params = [], []
        if not include_deleted:
            clauses.append("state NOT IN ('deleted','unlinked')")
            clauses.append("(state!='trashed' OR kind='trash')")
        if kind:
            clauses.append("kind=?")
            params.append(kind)
        if parent_id:
            clauses.append("json_extract(summary,'$.parent_id')=?")
            params.append(parent_id)
        if source_id:
            clauses.append("json_extract(summary,'$.source.id')=?")
            params.append(source_id)
        if q:
            clauses.append("(summary LIKE ? ESCAPE '\\' OR COALESCE(label_override,'') LIKE ? ESCAPE '\\')")
            needle = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            params.extend([needle, needle])
        if cursor:
            try:
                position = json.loads(base64.urlsafe_b64decode(cursor.encode() + b"=" * (-len(cursor) % 4)))
                timestamp, last_id = float(position[0]), str(position[1])
            except (ValueError, TypeError, IndexError) as exc:
                raise StorageLibraryError("library_cursor_invalid", status_code=400) from exc
            clauses.append("(COALESCE(last_opened_at,created_at)<? OR (COALESCE(last_opened_at,created_at)=? AND id>?))")
            params.extend([timestamp, timestamp, last_id])
        with contextlib.closing(self.connect()) as connection:
            rows = connection.execute("SELECT * FROM items" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY COALESCE(last_opened_at,created_at) DESC,id LIMIT ?", [*params, limit + 1]).fetchall()
        items = []
        for row in rows[:limit]:
            item = self._item(row)
            item["storage"] = self.item_storage(self.assets_for_items([item["id"]]))
            items.append(item)
        next_cursor = base64.urlsafe_b64encode(encode([items[-1].get("last_opened_at") or items[-1]["created_at"], items[-1]["id"]]).encode()).decode().rstrip("=") if len(rows) > limit else None
        return self.envelope(items=items, next_cursor=next_cursor)

    def update_item(self, identifier: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        current = self.get_item(identifier)
        if current["state"] in {"deleted", "deleting", "unlinked"}:
            raise StorageLibraryError("library_item_removed", item=current)
        if payload.get("expected_revision") != current["revision"]:
            raise StorageLibraryError("library_revision_changed", item=current)
        label = payload.get("label", current["label"])
        if not isinstance(label, str) or not label.strip() or len(label) > 240:
            raise StorageLibraryError("library_label_invalid", status_code=400)
        pinned = payload.get("pinned", current["pinned"])
        if not isinstance(pinned, bool):
            raise StorageLibraryError("library_pin_invalid", status_code=400)
        with self.transaction() as connection:
            updated = connection.execute("UPDATE items SET label_override=?,pinned=?,revision=revision+1,updated_at=? WHERE id=? AND revision=? AND state NOT IN ('deleted','deleting','unlinked')", (label.strip(), int(pinned), time.time(), identifier, current["revision"]))
            if updated.rowcount != 1:
                raise StorageLibraryError("library_revision_changed")
        return self.envelope(item=self.get_item(identifier))

    @contextlib.contextmanager
    def admission(self, *, exclusive: bool = False, blocking: bool = True):
        """Reentrant process-safe admission around actual filesystem access."""
        import fcntl
        current = self._admission_context.get()
        if current is not None:
            if exclusive and not current:
                raise StorageLibraryError("library_lock_upgrade_forbidden")
            yield
            return
        path = self.directory / "filesystem.lock"
        reject_symlinks(path)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            try:
                fcntl.flock(descriptor, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise StorageLibraryError("library_storage_busy", "Files are in use. Retry after the active operation finishes.") from exc
            context_token = self._admission_context.set(exclusive)
            try:
                yield
            finally:
                self._admission_context.reset(context_token)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def create_lease(self, *, item_ids: Iterable[str] = (), paths: Iterable[Path | str] = (), owner: str = "", purpose: str = "attachment", ttl: float = 120.0) -> dict[str, Any]:
        identifiers = list(dict.fromkeys(item_ids))
        assets = self.assets_for_items(identifiers)
        for identifier in identifiers:
            if self.get_item(identifier)["state"] in {"deleting", "deleted", "unlinked"}:
                raise StorageLibraryError("library_item_removed")
        lease_id, token, now = "lease_" + uuid.uuid4().hex, uuid.uuid4().hex, time.time()
        explicit_paths = [str(absolute(value)) for value in paths]
        all_paths = list(dict.fromkeys([*explicit_paths, *(str(self.asset_path(asset)) for asset in assets)]))
        with self.transaction() as connection:
            connection.execute("INSERT INTO leases VALUES (?,?,?,?,?,?,?,?,?)", (lease_id, token, owner, purpose, encode(identifiers), encode([a["id"] for a in assets]), encode(all_paths), now, now + max(30.0, min(3600.0, ttl))))
        return {"lease_id": lease_id, "lease_token": token, "expires_at": now + max(30.0, min(3600.0, ttl))}

    def heartbeat(self, lease_id: str, token: str) -> dict[str, Any]:
        expires = time.time() + 120.0
        with self.admission(), self.transaction() as connection:
            row = connection.execute("SELECT * FROM leases WHERE id=? AND token=?", (lease_id, token)).fetchone()
            if row is None or float(row["expires_at"]) <= time.time():
                raise StorageLibraryError("library_lease_expired", "Reopen the item to establish a fresh reservation.")
            connection.execute("UPDATE leases SET expires_at=? WHERE id=?", (expires, lease_id))
        return self.envelope(lease_id=lease_id, expires_at=expires)

    def release(self, lease_id: str, token: str) -> dict[str, Any]:
        with self.transaction() as connection:
            connection.execute("DELETE FROM leases WHERE id=? AND token=?", (lease_id, token))
        return self.envelope(lease_id=lease_id, state="released")

    def active_leases(self) -> list[dict[str, Any]]:
        with contextlib.closing(self.connect()) as connection:
            rows = connection.execute("SELECT * FROM leases WHERE expires_at>?", (time.time(),)).fetchall()
        result = []
        for row in rows:
            lease = dict(row)
            lease.pop("token", None)
            for key in ("item_ids", "asset_ids", "paths"):
                lease[key] = json.loads(lease[key])
            result.append(lease)
        return result

    @contextlib.contextmanager
    def protect(self, *, paths: Iterable[Path | str] = (), item_ids: Iterable[str] = (), owner: str = "", purpose: str = "operation"):
        paths, item_ids = tuple(paths), tuple(item_ids)
        with self.admission(blocking=False):
            if not paths and not item_ids:
                yield {}
                return
            lease = self.create_lease(item_ids=item_ids, paths=paths, owner=owner, purpose=purpose, ttl=3600)
            try:
                yield lease
            finally:
                self.release(lease["lease_id"], lease["lease_token"])

    def create_operation(self, kind: str, payload: Mapping[str, Any], *, state: str = "queued", idempotency_key: str = "", operation_items: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
        encoded = encode(dict(payload))
        request_hash = digest([kind, payload])
        descriptors = [(str(asset["id"]), encode(dict(asset))) for asset in operation_items]
        with self.transaction() as connection:
            if idempotency_key:
                row = connection.execute("SELECT * FROM operations WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if row:
                    if row["request_hash"] != request_hash:
                        raise StorageLibraryError("library_idempotency_conflict")
                    return {**self._operation(row), "_created": False}
            identifier, now = "libop_" + uuid.uuid4().hex, time.time()
            connection.execute("INSERT INTO operations(id,kind,idempotency_key,request_hash,request,state,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)", (identifier, kind, idempotency_key or None, request_hash, encoded, state, now, now))
            # Publish executable intent and its full file list together. A
            # running worker must never observe a queued, empty removal plan.
            connection.executemany("INSERT INTO operation_items(operation_id,asset_id,descriptor) VALUES (?,?,?)",
                                   [(identifier, asset_id, descriptor) for asset_id, descriptor in descriptors])
        return {**self.get_operation(identifier, internal=True), "_created": True}

    @staticmethod
    def _operation(row: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["operation_id"] = result.pop("id")
        result["type"] = result.pop("kind")
        for field in ("request", "progress", "result"):
            result[field] = json.loads(result[field])
        if result.get("error"):
            result["error"] = json.loads(result["error"])
        return result

    def get_operation(self, identifier: str, *, internal: bool = False) -> dict[str, Any]:
        with contextlib.closing(self.connect()) as connection:
            row = connection.execute("SELECT * FROM operations WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise StorageLibraryError("library_operation_not_found", status_code=404)
        result = self._operation(row)
        if not internal:
            result.pop("request", None)
            result.pop("request_hash", None)
            result.pop("idempotency_key", None)
        return self.envelope(**result)

    def update_operation(self, identifier: str, *, state: str | None = None, progress: Any = None, result: Any = None, error: Any = None) -> dict[str, Any]:
        values: dict[str, Any] = {"updated_at": time.time()}
        if state is not None:
            values["state"] = state
        for field, value in (("progress", progress), ("result", result), ("error", error)):
            if value is not None:
                values[field] = encode(value)
        with self.transaction() as connection:
            connection.execute("UPDATE operations SET " + ",".join(f"{key}=?" for key in values) + " WHERE id=?", [*values.values(), identifier])
        return self.get_operation(identifier)
