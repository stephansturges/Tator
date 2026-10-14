"""Disk-backed pair evidence; large datasets never retain bbox matches in RAM."""
import contextlib
import json
import sqlite3
from pathlib import Path
import tempfile


COMPACT_PAIR_SCHEMA = 'strict-pair-projection-v1'


def compact_pair(edge):
    """Fields needed by strict ordering/coverage; native proof stays in source.

    Rejected matches never affect visual grouping. Keep every measured pair,
    including original gate failures which a new object weight can promote.
    """
    fields = ('left', 'right', 'scene', 'exact_image', 'assignment_identity',
              'left_object', 'right_object', 'left_class_coverage', 'right_class_coverage',
              'left_unmatched_rows', 'right_unmatched_rows', 'left_valid', 'right_valid', 'valid')
    result = {key: edge[key] for key in fields}
    result['matches'] = [{key: match[key] for key in ('left_feature', 'right_feature', 'scale', 'accepted')}
                         for match in edge['matches'] if match['accepted']]
    return result


class PairStore:
    def __init__(self,path: Path):
        self.path = path
        self.writer = None

    def append(self,edge):
        if self.writer is None:
            self.writer = sqlite3.connect(self.path)
            self.writer.execute("CREATE TABLE IF NOT EXISTS pairs (a INTEGER,b INTEGER,valid INTEGER,payload TEXT,PRIMARY KEY(a,b))")
            self.writer.execute("CREATE INDEX IF NOT EXISTS pair_b ON pairs(b,valid)")
            self.writer.execute("CREATE INDEX IF NOT EXISTS pair_a ON pairs(a,valid)")
        self.writer.execute("INSERT OR REPLACE INTO pairs VALUES(?,?,?,?)",(edge["left"],edge["right"],int(edge["valid"]),json.dumps(edge,separators=(",",":"))))

    def finish(self):
        if self.writer is None:
            # An empty source graph is still a complete durable artifact.
            self.writer = sqlite3.connect(self.path)
            self.writer.execute("CREATE TABLE IF NOT EXISTS pairs (a INTEGER,b INTEGER,valid INTEGER,payload TEXT,PRIMARY KEY(a,b))")
        self.writer.commit(); self.writer.close(); self.writer=None

    def abort(self):
        if self.writer is not None:
            self.writer.rollback(); self.writer.close(); self.writer=None

    def __iter__(self):
        with contextlib.closing(sqlite3.connect(f"file:{self.path}?mode=ro",uri=True)) as db:
            for row in db.execute("SELECT payload FROM pairs ORDER BY a,b"):
                yield json.loads(row[0])

    def valid_links(self):
        """Topology needs pair endpoints, not the large assignment payloads."""
        with contextlib.closing(sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)) as db:
            yield from db.execute('SELECT a,b FROM pairs WHERE valid=1 ORDER BY a,b')

    def edges_for(self,index):
        with contextlib.closing(sqlite3.connect(f"file:{self.path}?mode=ro",uri=True)) as db:
            for row in db.execute("SELECT payload FROM pairs WHERE valid=1 AND a=? UNION ALL SELECT payload FROM pairs WHERE valid=1 AND b=?",(index,index)):
                yield json.loads(row[0])

    def all_edges_for(self,index):
        """Measured pairs before gate evaluation, for cached reweighting only."""
        with contextlib.closing(sqlite3.connect(f"file:{self.path}?mode=ro",uri=True)) as db:
            for row in db.execute("SELECT payload FROM pairs WHERE a=? UNION ALL SELECT payload FROM pairs WHERE b=?",(index,index)):
                yield json.loads(row[0])

    def pair(self, left, right):
        a, b = sorted((left, right))
        with contextlib.closing(sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)) as db:
            row = db.execute("SELECT payload FROM pairs WHERE a=? AND b=?", (a, b)).fetchone()
        return json.loads(row[0]) if row else None

    def comparisons_for(self, members):
        """All measured family pairs, including measured gate failures."""
        with contextlib.closing(sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)) as db:
            db.execute('CREATE TEMP TABLE selected_members (id INTEGER PRIMARY KEY)')
            db.executemany('INSERT INTO selected_members VALUES (?)', ((int(index),) for index in members))
            for row in db.execute('SELECT p.payload FROM selected_members l CROSS JOIN pairs p ON p.a=l.id JOIN selected_members r ON p.b=r.id ORDER BY p.a,p.b'):
                yield json.loads(row[0])


class CandidatePairs:
    """Deduplicate a large retrieval pool on disk instead of a Python set."""
    def __init__(self, directory=None):
        descriptor, name = tempfile.mkstemp(prefix='.retrieved-pairs-', suffix='.sqlite3', dir=directory)
        import os
        os.close(descriptor)
        self.path = Path(name)
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA journal_mode=OFF')
        self.db.execute('CREATE TABLE candidates (a INTEGER, b INTEGER, PRIMARY KEY(a,b)) WITHOUT ROWID')

    def update(self, pairs):
        chunk = []
        for pair in pairs:
            chunk.append(pair)
            if len(chunk) >= 1024:
                self.db.executemany('INSERT OR IGNORE INTO candidates VALUES (?, ?)', chunk)
                chunk = []
        if chunk: self.db.executemany('INSERT OR IGNORE INTO candidates VALUES (?, ?)', chunk)

    def add(self, pair):
        self.db.execute('INSERT OR IGNORE INTO candidates VALUES (?, ?)', pair)

    def __len__(self):
        return self.db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]

    def __iter__(self):
        self.db.commit()
        yield from self.db.execute('SELECT a,b FROM candidates ORDER BY a,b')

    def close(self):
        self.db.close()
        self.path.unlink(missing_ok=True)
