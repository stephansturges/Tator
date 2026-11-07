"""Shared, transactional hosted v1 -> v2 upgrade and tranche invariants.

Called with an exclusive maintenance connection before normal write transactions.
The assignment parent is rebuilt in SQLite's documented order; its identity and
all referring submission/member rows survive unchanged.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time

from .errors import DomainError

SCHEMA_VERSION = 2

DDL = (
    """CREATE TABLE IF NOT EXISTS hosted_tranches (
        id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
        name TEXT NOT NULL, instructions TEXT NOT NULL DEFAULT '', annotator_id TEXT,
        lifecycle TEXT NOT NULL CHECK(lifecycle IN ('draft','active','closed')),
        revision INTEGER NOT NULL DEFAULT 0, created_by TEXT NOT NULL, updated_by TEXT NOT NULL,
        created_at REAL NOT NULL, updated_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS hosted_tranche_members (
        id TEXT PRIMARY KEY, tranche_id TEXT NOT NULL REFERENCES hosted_tranches(id),
        image_id TEXT NOT NULL REFERENCES hosted_images(id),
        assignment_id TEXT REFERENCES hosted_assignments(id), annotator_id TEXT,
        outcome TEXT NOT NULL CHECK(outcome IN ('reserved','assigned','returned','submitted','approved','withdrawn','transferred','archived')),
        is_current INTEGER NOT NULL CHECK(is_current IN (0,1)),
        joined_at REAL NOT NULL, ended_at REAL, reason TEXT NOT NULL DEFAULT '',
        joined_by TEXT NOT NULL, ended_by TEXT,
        CHECK((is_current=1 AND outcome IN ('reserved','assigned','returned','submitted') AND ended_at IS NULL)
           OR (is_current=0 AND outcome IN ('approved','withdrawn','transferred','archived') AND ended_at IS NOT NULL)))""",
    "CREATE UNIQUE INDEX IF NOT EXISTS hosted_tranche_current_image ON hosted_tranche_members(image_id) WHERE is_current=1",
    "CREATE INDEX IF NOT EXISTS hosted_tranche_members_order ON hosted_tranche_members(tranche_id,joined_at,id)",
    "CREATE INDEX IF NOT EXISTS hosted_tranche_members_latest ON hosted_tranche_members(tranche_id,image_id,joined_at,id)",
    "CREATE INDEX IF NOT EXISTS hosted_tranche_members_personal ON hosted_tranche_members(annotator_id,tranche_id,image_id)",
    "CREATE INDEX IF NOT EXISTS hosted_tranche_members_episode ON hosted_tranche_members(assignment_id,image_id)",
    "CREATE INDEX IF NOT EXISTS hosted_tranches_list ON hosted_tranches(dataset_id,lifecycle,created_at,id)",
    "CREATE INDEX IF NOT EXISTS hosted_tranches_assignee ON hosted_tranches(annotator_id,lifecycle)",
    """CREATE TABLE IF NOT EXISTS hosted_tranche_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, tranche_id TEXT NOT NULL REFERENCES hosted_tranches(id),
        actor_id TEXT NOT NULL, kind TEXT NOT NULL, data_json TEXT NOT NULL, created_at REAL NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS hosted_tranche_events_order ON hosted_tranche_events(tranche_id,id)",
    """CREATE TABLE IF NOT EXISTS hosted_migration_outcomes (
        id TEXT PRIMARY KEY, migration TEXT NOT NULL, image_id TEXT NOT NULL REFERENCES hosted_images(id),
        submission_id TEXT REFERENCES hosted_submissions(id), archive_id TEXT REFERENCES hosted_archives(id),
        outcome TEXT NOT NULL, actor_kind TEXT NOT NULL CHECK(actor_kind='system'), created_at REAL NOT NULL)""",
)


def validate_tranches(con: sqlite3.Connection) -> None:
    """Validate relational ownership, including legacy writers and restored state."""
    bad = con.execute("""SELECT i.id FROM hosted_images i
        LEFT JOIN hosted_assignments a ON a.id=i.assignment_id
        LEFT JOIN hosted_tranche_members m ON m.image_id=i.id AND m.is_current=1
        LEFT JOIN hosted_tranches t ON t.id=m.tranche_id
        WHERE (i.state IN ('assigned','returned','submitted') AND (
          a.id IS NULL OR m.id IS NULL OR a.tranche_id!=m.tranche_id OR m.assignment_id IS NULL OR m.assignment_id!=a.id
          OR a.dataset_id!=i.dataset_id OR t.dataset_id!=i.dataset_id OR t.lifecycle!='active'
          OR i.assignee_id IS NULL OR a.annotator_id!=i.assignee_id OR t.annotator_id!=i.assignee_id
          OR m.annotator_id IS NULL OR m.annotator_id!=i.assignee_id OR m.outcome!=i.state
          OR NOT EXISTS(SELECT 1 FROM hosted_assignment_members am WHERE am.assignment_id=a.id AND am.image_id=i.id)
          OR EXISTS(SELECT 1 FROM hosted_archived_images ar WHERE ar.image_id=i.id)))
        OR (i.state NOT IN ('assigned','returned','submitted') AND (i.assignment_id IS NOT NULL OR i.assignee_id IS NOT NULL OR i.active_submission IS NOT NULL))
        OR (m.id IS NOT NULL AND (t.dataset_id!=i.dataset_id OR EXISTS(SELECT 1 FROM hosted_archived_images ar WHERE ar.image_id=i.id) OR EXISTS(SELECT 1 FROM hosted_archived_datasets ad WHERE ad.dataset_id=i.dataset_id) OR
          (m.outcome='reserved' AND (i.state!='unassigned' OR t.lifecycle!='draft' OR m.assignment_id IS NOT NULL OR m.annotator_id IS NOT NULL))
          OR (m.outcome!='reserved' AND i.state NOT IN ('assigned','returned','submitted'))))
        OR (i.state='submitted' AND NOT EXISTS(SELECT 1 FROM hosted_submissions s
          WHERE s.id=i.active_submission AND s.image_id=i.id AND s.assignment_id=i.assignment_id AND s.actor_id=i.assignee_id))
        OR (i.state!='submitted' AND i.active_submission IS NOT NULL)
        LIMIT 1""").fetchone()
    if bad:
        raise DomainError('tranche_ownership_inconsistent', 409, {'image_id': bad[0]})
    bad = con.execute("""SELECT a.id FROM hosted_assignments a LEFT JOIN hosted_tranches t ON t.id=a.tranche_id
        WHERE t.id IS NULL OR t.dataset_id!=a.dataset_id LIMIT 1""").fetchone()
    if bad:
        raise DomainError('tranche_episode_inconsistent', 409, {'assignment_id': bad[0]})


def _legacy_id(kind, value):
    return kind + '_legacy_' + hashlib.sha256(value.encode()).hexdigest()[:32]


def migrate_store(con: sqlite3.Connection, *, now=None) -> None:
    """Upgrade supported schema under maintenance exclusivity, or validate v2.

    Generic recovery archives without hosted state are left to their own hook.
    Foreign-key enforcement is restored even after a failed migration.
    """
    if not con.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_schema'").fetchone():
        return
    version = con.execute("SELECT version FROM hosted_schema WHERE component='store'").fetchone()
    if not version or version[0] not in (1, SCHEMA_VERSION):
        raise RuntimeError('Unsupported hosted store schema')
    if version[0] == SCHEMA_VERSION:
        validate_tranches(con)
        return
    if con.in_transaction:
        raise RuntimeError('Hosted migration requires an exclusive connection outside a transaction')
    old_factory = con.row_factory
    con.row_factory = sqlite3.Row
    stamp = time.time() if now is None else now
    con.execute('PRAGMA foreign_keys=OFF')
    try:
        con.execute('BEGIN IMMEDIATE')
        # Another startup may have completed while we waited for the lock.
        if con.execute("SELECT version FROM hosted_schema WHERE component='store'").fetchone()[0] == SCHEMA_VERSION:
            con.commit()
            validate_tranches(con)
            return
        for statement in DDL:
            con.execute(statement)
        assignments = con.execute('SELECT * FROM hosted_assignments ORDER BY created_at,id').fetchall()
        for assignment in assignments:
            tranche_id = _legacy_id('tranche', assignment['id'])
            active = con.execute("SELECT 1 FROM hosted_images i WHERE assignment_id=? AND state IN ('assigned','returned','submitted') AND NOT EXISTS(SELECT 1 FROM hosted_archived_images x WHERE x.image_id=i.id) LIMIT 1", (assignment['id'],)).fetchone()
            con.execute('INSERT INTO hosted_tranches VALUES(?,?,?,?,?,?,?,?,?,?,?)', (
                tranche_id, assignment['dataset_id'], 'Imported assignment ' + assignment['id'][-8:], '',
                assignment['annotator_id'], 'active' if active else 'closed', 0, assignment['created_by'],
                assignment['created_by'], assignment['created_at'], stamp))
            members = con.execute('''SELECT m.*,i.assignment_id AS current_assignment,i.state AS image_state,
                x.archive_id FROM hosted_assignment_members m JOIN hosted_images i ON i.id=m.image_id
                LEFT JOIN hosted_archived_images x ON x.image_id=i.id WHERE m.assignment_id=?''', (assignment['id'],)).fetchall()
            for member in members:
                pending = member['current_assignment'] == assignment['id'] and member['image_state'] in ('assigned','returned','submitted')
                current = bool(pending and not member['archive_id'])
                outcome = member['image_state'] if current else ('archived' if pending and member['archive_id'] else
                    'approved' if member['state'] == 'confirmed' else 'withdrawn' if member['state'] in ('overridden','unassigned') else 'transferred')
                con.execute('INSERT INTO hosted_tranche_members VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', (
                    _legacy_id('member', assignment['id'] + ':' + member['image_id']), tranche_id, member['image_id'],
                    assignment['id'], assignment['annotator_id'], outcome, int(current), assignment['created_at'],
                    None if current else stamp, 'Backfilled from assignment history', assignment['created_by'], None if current else 'system:schema-v2'))
        # Preserve every old parent identifier. Do not rename the old table first.
        con.execute('''CREATE TABLE hosted_assignments_v2 (
            id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
            annotator_id TEXT NOT NULL, created_by TEXT NOT NULL, created_at REAL NOT NULL,
            tranche_id TEXT NOT NULL REFERENCES hosted_tranches(id))''')
        for row in assignments:
            con.execute('INSERT INTO hosted_assignments_v2 VALUES(?,?,?,?,?,?)',
                        tuple(row[key] for key in ('id','dataset_id','annotator_id','created_by','created_at')) + (_legacy_id('tranche', row['id']),))
        con.execute('DROP TABLE hosted_assignments')
        con.execute('ALTER TABLE hosted_assignments_v2 RENAME TO hosted_assignments')
        con.execute('CREATE INDEX hosted_assignments_tranche ON hosted_assignments(tranche_id,created_at,id)')
        archived = con.execute("SELECT i.*,x.archive_id FROM hosted_images i JOIN hosted_archived_images x ON x.image_id=i.id WHERE i.state!='confirmed'").fetchall()
        for row in archived:
            con.execute("INSERT INTO hosted_migration_outcomes VALUES(?,?,?,?,?,?,'system',?)", (
                _legacy_id('migration', row['id']), 'store-v1-to-v2', row['id'], row['active_submission'], row['archive_id'], 'archived', stamp))
            con.execute("UPDATE hosted_images SET state='unassigned',assignee_id=NULL,assignment_id=NULL,active_submission=NULL,generation=generation+1 WHERE id=?", (row['id'],))
            con.execute('DELETE FROM hosted_editor_leases WHERE image_id=?', (row['id'],))
        for table in ('hosted_tranche_events', 'hosted_migration_outcomes'):
            for verb in ('UPDATE','DELETE'):
                con.execute(f"CREATE TRIGGER {table}_immutable_{verb.lower()} BEFORE {verb} ON {table} BEGIN SELECT RAISE(ABORT,'immutable hosted history'); END")
        validate_tranches(con)
        if con.execute('PRAGMA foreign_key_check').fetchone():
            raise RuntimeError('Hosted migration foreign-key validation failed')
        con.execute("UPDATE hosted_schema SET version=? WHERE component='store'", (SCHEMA_VERSION,))
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.row_factory = old_factory
        con.execute('PRAGMA foreign_keys=ON')
        if con.execute('PRAGMA foreign_keys').fetchone()[0] != 1:
            raise RuntimeError('Hosted migration failed to restore foreign-key enforcement')
