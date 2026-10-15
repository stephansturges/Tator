"""Server-owned accounts and revocable opaque sessions for hosted Tator.

Tokens are returned only at creation and stored as hashes. This service knows
nothing about browser-supplied roles or caller-selected dataset roots.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from .errors import DomainError
from .types import Actor


@dataclass(frozen=True)
class Login:
    actor: Actor
    token: str
    csrf: str
    expires_at: float


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _username(value: str) -> str:
    value = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", value):
        raise DomainError("invalid_username", 422, "Use 3–64 letters, numbers, dots, underscores or hyphens.")
    return value


def _password(value: str) -> str:
    if not isinstance(value, str) or not 12 <= len(value) <= 1024:
        raise DomainError("invalid_password", 422, "Use a password of at least 12 characters.")
    return value


class AuthService:
    """Uses the authority database, including its short transaction boundary."""

    def __init__(self, store: Any, *, clock=time.time, session_seconds=12 * 3600):
        self.store = store
        self.clock = clock
        self.session_seconds = session_seconds
        self.hasher = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)
        self._dummy_hash = self.hasher.hash(secrets.token_urlsafe(32))
        with store.transaction() as db:
            # Execute individually: executescript would implicitly commit an
            # existing transaction and defeat last-admin/bootstrap locking.
            for sql in (
                """CREATE TABLE IF NOT EXISTS hosted_users (
                    id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin','annotator')),
                    disabled INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL, password_changed_at REAL NOT NULL)""",
                """CREATE TABLE IF NOT EXISTS hosted_sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES hosted_users(id),
                    csrf_hash TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL,
                    instance_id TEXT NOT NULL)""",
                """CREATE TABLE IF NOT EXISTS hosted_auth_links (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES hosted_users(id),
                    purpose TEXT NOT NULL, expires_at REAL NOT NULL, used_at REAL)""",
                """CREATE TABLE IF NOT EXISTS hosted_auth_attempts (
                    id INTEGER PRIMARY KEY, username TEXT NOT NULL, address_hash TEXT NOT NULL,
                    at REAL NOT NULL, succeeded INTEGER NOT NULL DEFAULT 0)""",
                "CREATE INDEX IF NOT EXISTS hosted_auth_attempt_time ON hosted_auth_attempts(at)",
                "CREATE TABLE IF NOT EXISTS hosted_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)",
            ):
                db.execute(sql)
            db.execute("INSERT OR IGNORE INTO hosted_meta VALUES ('instance_id',?)", (str(uuid.uuid4()),))

    @property
    def instance_id(self) -> str:
        with self.store.read_connection() as db:
            return db.execute("SELECT value FROM hosted_meta WHERE key='instance_id'").fetchone()[0]

    @staticmethod
    def _actor(row) -> Actor:
        return Actor(user_id=row["id"], role=row["role"], username=row["username"])

    @staticmethod
    def _public(row) -> dict:
        return {key: row[key] for key in ("id", "username", "role", "disabled", "created_at")}

    def require_admin(self, actor: Actor, db=None) -> None:
        if db is None:
            with self.store.read_connection() as conn:
                return self.require_admin(actor, conn)
        row = db.execute("SELECT role,disabled FROM hosted_users WHERE id=?", (actor.user_id,)).fetchone()
        if row is None or row["disabled"] or row["role"] != "admin":
            raise DomainError("administrator_required", 403)
        self.store._actor(db, actor, admin=True)

    def bootstrap(self, username: str, password: str) -> dict:
        username, encoded = _username(username), self.hasher.hash(_password(password))
        now, user_id = self.clock(), str(uuid.uuid4())
        with self.store.transaction() as db:
            if db.execute("SELECT 1 FROM hosted_users LIMIT 1").fetchone():
                raise DomainError("already_bootstrapped", 409)
            db.execute("INSERT INTO hosted_users VALUES (?,?,?,?,0,?,?)", (user_id, username, encoded, "admin", now, now))
        return {"id": user_id, "username": username, "role": "admin"}

    def create_user(self, actor: Actor, username: str, role="annotator") -> dict:
        username = _username(username)
        if role not in ("admin", "annotator"):
            raise DomainError("invalid_role", 422)
        user_id, now = str(uuid.uuid4()), self.clock()
        encoded = self.hasher.hash(secrets.token_urlsafe(48))
        token = secrets.token_urlsafe(32)
        with self.store.transaction() as db:
            self.require_admin(actor, db)
            if db.execute("SELECT 1 FROM hosted_users WHERE username=?", (username,)).fetchone():
                raise DomainError("username_exists", 409)
            db.execute("INSERT INTO hosted_users VALUES (?,?,?,?,0,?,?)", (user_id, username, encoded, role, now, now))
            db.execute("INSERT INTO hosted_auth_links VALUES (?,?,?,?,NULL)", (_digest(token), user_id, "invite", now + 86400))
        return {"id": user_id, "username": username, "role": role, "invite_token": token, "expires_at": now + 86400}

    def list_users(self, actor: Actor) -> list[dict]:
        with self.store.read_connection() as db:
            self.require_admin(actor, db)
            return [self._public(row) for row in db.execute("SELECT * FROM hosted_users ORDER BY username")]

    def set_user(self, actor: Actor, user_id: str, *, disabled: bool, role: str,
                 expected_role: str | None = None, expected_disabled: bool | None = None) -> dict:
        if role not in ("admin", "annotator"):
            raise DomainError("invalid_role", 422)
        with self.store.transaction() as db:
            self.require_admin(actor, db)
            row = db.execute("SELECT * FROM hosted_users WHERE id=?", (user_id,)).fetchone()
            if row is None:
                raise DomainError("user_not_found", 404)
            if expected_role is not None and (row["role"] != expected_role or bool(row["disabled"]) != expected_disabled):
                raise DomainError("user_changed", 409, "This account changed. Refresh before editing it again.")
            if row["role"] == "admin" and not row["disabled"] and (disabled or role != "admin"):
                count = db.execute("SELECT count(*) FROM hosted_users WHERE role='admin' AND disabled=0").fetchone()[0]
                if count <= 1:
                    raise DomainError("last_administrator", 409)
            db.execute("UPDATE hosted_users SET disabled=?,role=? WHERE id=?", (int(disabled), role, user_id))
            db.execute("DELETE FROM hosted_sessions WHERE user_id=?", (user_id,))
            if disabled:
                db.execute("DELETE FROM hosted_auth_links WHERE user_id=?", (user_id,))
            return self._public(db.execute("SELECT * FROM hosted_users WHERE id=?", (user_id,)).fetchone())

    def reset_link(self, actor: Actor, user_id: str) -> dict:
        now, token = self.clock(), secrets.token_urlsafe(32)
        with self.store.transaction() as db:
            self.require_admin(actor, db)
            row = db.execute("SELECT disabled FROM hosted_users WHERE id=?", (user_id,)).fetchone()
            if row is None or row[0]:
                raise DomainError("user_not_available", 404)
            db.execute("DELETE FROM hosted_auth_links WHERE user_id=?", (user_id,))
            db.execute("DELETE FROM hosted_sessions WHERE user_id=?", (user_id,))
            db.execute("INSERT INTO hosted_auth_links VALUES (?,?,?,?,NULL)", (_digest(token), user_id, "reset", now + 3600))
        return {"reset_token": token, "expires_at": now + 3600}

    def redeem(self, token: str, password: str) -> None:
        encoded, now = self.hasher.hash(_password(password)), self.clock()
        with self.store.transaction() as db:
            row = db.execute("SELECT l.*,u.disabled FROM hosted_auth_links l JOIN hosted_users u ON u.id=l.user_id WHERE l.token_hash=?", (_digest(token),)).fetchone()
            if row is None or row["used_at"] is not None or row["expires_at"] <= now or row["disabled"]:
                raise DomainError("invalid_or_expired_link", 400)
            db.execute("UPDATE hosted_users SET password_hash=?,password_changed_at=? WHERE id=?", (encoded, now, row["user_id"]))
            db.execute("UPDATE hosted_auth_links SET used_at=? WHERE token_hash=?", (now, _digest(token)))
            db.execute("DELETE FROM hosted_sessions WHERE user_id=?", (row["user_id"],))

    def login(self, username: str, password: str, address: str) -> Login:
        # Record attempts before expensive hashing so parallel callers cannot
        # all pass a stale rate-limit check.
        username, now = str(username).strip().lower()[:128], self.clock()
        address_hash = _digest(address)
        with self.store.transaction() as db:
            db.execute("DELETE FROM hosted_auth_attempts WHERE at<?", (now - 900,))
            per_user = db.execute("SELECT count(*) FROM hosted_auth_attempts WHERE username=? AND succeeded=0", (username,)).fetchone()[0]
            per_address = db.execute("SELECT count(*) FROM hosted_auth_attempts WHERE address_hash=?", (address_hash,)).fetchone()[0]
            if per_user >= 10 or per_address >= 100:
                raise DomainError("login_rate_limited", 429)
            attempt = db.execute("INSERT INTO hosted_auth_attempts(username,address_hash,at) VALUES (?,?,?)", (username, address_hash, now)).lastrowid
            row = db.execute("SELECT * FROM hosted_users WHERE username=?", (username,)).fetchone()
        encoded = row["password_hash"] if row else self._dummy_hash
        valid = False
        try:
            valid = len(password) <= 1024 and self.hasher.verify(encoded, password)
        except (VerificationError, InvalidHashError, TypeError):
            pass
        if not valid or row is None or row["disabled"]:
            raise DomainError("invalid_credentials", 401, "Username or password is incorrect.")
        token = secrets.token_urlsafe(32)
        csrf = _digest("tator-csrf\0" + token)
        expires_at = now + self.session_seconds
        with self.store.transaction() as db:
            current = db.execute("SELECT * FROM hosted_users WHERE id=?", (row["id"],)).fetchone()
            if current is None or current["disabled"] or current["password_hash"] != encoded:
                raise DomainError("invalid_credentials", 401)
            instance = db.execute("SELECT value FROM hosted_meta WHERE key='instance_id'").fetchone()[0]
            db.execute("DELETE FROM hosted_sessions WHERE expires_at<=?", (now,))
            db.execute("INSERT INTO hosted_sessions VALUES (?,?,?,?,?,?)", (_digest(token), row["id"], _digest(csrf), now, expires_at, instance))
            db.execute("UPDATE hosted_auth_attempts SET succeeded=1 WHERE id=?", (attempt,))
            actor = self._actor(current)
        return Login(actor, token, csrf, expires_at)

    def authenticate(self, token: str, *, csrf: str | None = None) -> Actor:
        with self.store.read_connection() as db:
            row = db.execute("""SELECT u.*,s.csrf_hash,s.expires_at,s.instance_id
                FROM hosted_sessions s JOIN hosted_users u ON u.id=s.user_id
                WHERE s.token_hash=?""", (_digest(token),)).fetchone()
            instance = db.execute("SELECT value FROM hosted_meta WHERE key='instance_id'").fetchone()[0]
        if row is None or row["disabled"] or row["expires_at"] <= self.clock() or row["instance_id"] != instance:
            raise DomainError("login_required", 401)
        if csrf is not None and not hmac.compare_digest(row["csrf_hash"], _digest(csrf)):
            raise DomainError("csrf_rejected", 403)
        return Actor(user_id=row["id"], role=row["role"], username=row["username"],
                     session_id=_digest(token), instance_id=instance)

    def session_csrf(self, token: str) -> str:
        # Stable within a login so a second tab cannot invalidate the first
        # tab's next save. Rotating the opaque login token rotates this too.
        self.authenticate(token)
        return _digest("tator-csrf\0" + token)

    def logout(self, token: str) -> None:
        with self.store.transaction() as db:
            db.execute("DELETE FROM hosted_sessions WHERE token_hash=?", (_digest(token),))

    @staticmethod
    def invalidate_restored_sessions(db) -> None:
        for table in ("hosted_sessions", "hosted_auth_links", "hosted_editor_leases"):
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                db.execute(f"DELETE FROM {table}")
        db.execute("INSERT OR REPLACE INTO hosted_meta VALUES ('instance_id',?)", (str(uuid.uuid4()),))
