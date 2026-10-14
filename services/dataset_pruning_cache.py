"""Content/recipe keyed descriptors shared with ingestion and DQE.

DQE matrices stay in place: references are bound to artifact file state. Only
newly encoded descriptors occupy cache blobs. Annotation classes and splits
are absent from keys, so relabeling and moving images do not re-encode pixels.
"""
from __future__ import annotations
import contextlib
import json
import sqlite3
import threading
import hashlib
from pathlib import Path
import numpy as np
from services.dataset_pruning import digest


def file_state(path: Path) -> list[int]:
    s = path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


class DescriptorCache:
    def __init__(self, root: Path):
        root.mkdir(parents=True,exist_ok=True)
        self.path = root / "descriptors.sqlite3"
        self.lock = threading.RLock()
        self.maps = {}
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS descriptors (key TEXT PRIMARY KEY, dim INTEGER, vector BLOB, reference TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS pixel_hashes (identity TEXT PRIMARY KEY, sha256 TEXT)")

    @contextlib.contextmanager
    def connect(self):
        with self.lock, contextlib.closing(sqlite3.connect(self.path,timeout=30)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db

    @staticmethod
    def key(image_sha: str, recipe: dict, geometry: str = "") -> str:
        return digest({"image":image_sha,"recipe":recipe,"geometry":geometry})

    def get(self, key: str):
        with self.lock:
            return self._get(key)

    def get_many(self, keys):
        """Read a bounded set of rows with one SQLite connection.

        Preserve input order and missing entries, including duplicate keys.
        Every referenced producer row receives the same content-state checks
        and normalization as a single lookup. Callers bound vector memory by
        passing their existing replay batch rather than the full dataset.
        """
        keys = list(keys)
        if not keys:
            return []
        with self.lock:
            rows = {}
            unique = list(dict.fromkeys(keys))
            with self.connect() as db:
                for start in range(0, len(unique), 512):
                    chunk = unique[start:start + 512]
                    query = "SELECT key,dim,vector,reference FROM descriptors WHERE key IN (" + ",".join("?" for _ in chunk) + ")"
                    for key, dim, blob, reference in db.execute(query, chunk):
                        rows[key] = (dim, blob, reference)
            decoded = {key: self._decode(row) for key, row in rows.items()}
            return [decoded.get(key) for key in keys]

    def _get(self, key: str):
        with self.connect() as db:
            row = db.execute("SELECT dim,vector,reference FROM descriptors WHERE key=?",(key,)).fetchone()
        return self._decode(row)

    def _decode(self, row):
        if row is None:
            return None
        try:
            dim,blob,raw = row
            if blob is not None:
                vector = np.frombuffer(blob,dtype=np.float32)
            else:
                ref = json.loads(raw)
                path = Path(ref["path"])
                if path.is_symlink() or file_state(path) != ref["state"]:
                    return None
                map_key = (str(path),tuple(ref["state"]))
                if map_key not in self.maps:
                    self.maps[map_key] = np.load(path,mmap_mode="r",allow_pickle=False)
                vector = np.array(self.maps[map_key][ref["row"],ref.get("start", 0):ref["end"]],np.float32,copy=True)
                if file_state(path)!=ref["state"]: return None
            if vector.shape != (dim,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 1e-8:
                return None
            return vector / max(float(np.linalg.norm(vector)),1e-12)
        except (OSError,ValueError,KeyError,IndexError,TypeError):
            return None

    def put(self, key: str, vector: np.ndarray):
        self.put_many([(key, vector)])

    def put_many(self, entries):
        rows = []
        for key, vector in entries:
            value = np.asarray(vector,np.float32).reshape(-1)
            if not value.size or not np.isfinite(value).all() or np.linalg.norm(value) <= 1e-8:
                raise ValueError("Invalid cached descriptor")
            value = value/max(float(np.linalg.norm(value)),1e-12)
            rows.append((key,len(value),value.tobytes()))
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO descriptors VALUES (?,?,?,NULL)", rows)

    def put_references(self, entries: list[tuple[str,int]], path: Path, end: int, *, expected_state=None, start=0):
        state = expected_state if expected_state is not None else file_state(path)
        if file_state(path)!=state: raise ValueError("Producer artifact changed before cache reference publication")
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO descriptors VALUES (?,?,NULL,?)",
                [(key,end-start,json.dumps({"path":str(path.resolve()),"state":state,"row":row,"start":start,"end":end})) for key,row in entries])

    def reference_dependencies(self, keys):
        """Bound producer files actually consumed by this component only."""
        dependencies = {}
        with self.connect() as db:
            for start in range(0, len(keys), 512):
                chunk = keys[start:start + 512]
                if not chunk: continue
                rows = db.execute('SELECT reference FROM descriptors WHERE key IN (' + ','.join('?' for _ in chunk) + ') AND reference IS NOT NULL', chunk)
                for (raw,) in rows:
                    reference = json.loads(raw)
                    path = Path(reference['path'])
                    if path.is_symlink() or file_state(path) != reference['state']:
                        raise ValueError('A consumed component artifact changed before publication')
                    previous = dependencies.get(str(path))
                    if previous and previous['state'] != reference['state']:
                        raise ValueError('A component mixed producer artifact generations')
                    dependencies[str(path)] = {'path': str(path), 'state': reference['state'], 'role': 'component_source',
                                               'required': False, 'sha256': reference.get('sha256'),
                                               'producer_receipt_sha256': reference.get('producer_receipt_sha256'),
                                               'content_attestation': 'exact terminal producer artifact and row mapping' if reference.get('producer_receipt_sha256') else 'producer-bound descriptor rows; whole artifact SHA may be absent'}
        return list(dependencies.values())

    def put_attested_references(self, entries, artifacts, *, producer_receipt_sha256, expected_rows):
        """Atomically publish terminal shared-producer rows without vector copies.

        Each entry is (key, artifact_index, row). The caller has bound the key
        to actual encoded image bytes and exact prepared object geometry. All
        matrices and consumed rows are checked before committing references.
        """
        if type(expected_rows) is not int or expected_rows <= 0 or not artifacts:
            raise ValueError('Invalid component reference population')
        loaded = []
        try:
            for artifact in artifacts:
                path = Path(artifact['path']).resolve()
                if Path(artifact['path']).is_symlink() or file_state(path) != artifact['state']:
                    raise ValueError('Producer artifact changed before publication')
                sha = hashlib.sha256()
                with path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''): sha.update(chunk)
                if sha.hexdigest() != artifact['sha256'] or file_state(path) != artifact['state']:
                    raise ValueError('Producer artifact content does not match its terminal receipt')
                matrix = np.load(path, mmap_mode='r', allow_pickle=False)
                start, end = int(artifact.get('start', 0)), int(artifact['end'])
                if matrix.ndim != 2 or matrix.dtype != np.float32 or not 0 <= start < end <= matrix.shape[1]:
                    raise ValueError('Invalid producer component slice')
                loaded.append((path, matrix, start, end, artifact))
            if len({end - start for _, _, start, end, _ in loaded}) != 1:
                raise ValueError('A component cannot mix incompatible dimensions')
            count = 0
            with self.connect() as db:
                # The cache is an acceleration store, never a head-space
                # conversion. Duplicate keys must resolve identical vectors.
                db.execute('PRAGMA temp_store=FILE')
                db.execute('CREATE TEMP TABLE publish_seen (key TEXT PRIMARY KEY, segment INTEGER, row INTEGER) WITHOUT ROWID')
                for key, segment, row in entries:
                    if type(segment) is not int or not 0 <= segment < len(loaded) or type(row) is not int:
                        raise ValueError('Invalid component row reference')
                    path, matrix, start, end, artifact = loaded[segment]
                    if not 0 <= row < len(matrix): raise ValueError('Producer row is outside its artifact')
                    vector = np.asarray(matrix[row, start:end])
                    if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 1e-8:
                        raise ValueError('Invalid terminal producer descriptor')
                    previous = db.execute('SELECT segment,row FROM publish_seen WHERE key=?', (key,)).fetchone()
                    if previous:
                        old_segment, old_row = previous; _, old_matrix, old_start, old_end, _ = loaded[old_segment]
                        if not np.array_equal(vector, old_matrix[old_row, old_start:old_end]):
                            raise ValueError('Duplicate pixel/geometry keys have conflicting producer vectors')
                    else: db.execute('INSERT INTO publish_seen VALUES (?,?,?)', (key, segment, row))
                    reference = {'path': str(path), 'state': artifact['state'], 'sha256': artifact['sha256'],
                                 'producer_receipt_sha256': producer_receipt_sha256, 'row': row, 'start': start, 'end': end}
                    db.execute('INSERT OR REPLACE INTO descriptors VALUES (?,?,NULL,?)', (key, end - start, json.dumps(reference, separators=(',', ':'))))
                    count += 1
                if count != expected_rows: raise ValueError('Producer row population is incomplete')
                if any(file_state(path) != artifact['state'] for path, _, _, _, artifact in loaded):
                    raise ValueError('Producer artifact changed during publication')
            return {'published_rows': count, 'artifact_count': len(loaded), 'vectors_copied': 0}
        finally:
            for _, matrix, _, _, _ in loaded:
                if getattr(matrix, '_mmap', None) is not None: matrix._mmap.close()

    def close_maps(self):
        with self.lock: self.maps.clear()

    def pixel_hash(self,path: Path,state: list[int],compute):
        identity = digest({"path":str(path.resolve()),"state":state})
        with self.connect() as db:
            row=db.execute("SELECT sha256 FROM pixel_hashes WHERE identity=?",(identity,)).fetchone()
        if row is not None: return row[0]
        value=compute()
        if file_state(path)!=state: raise ValueError("Source image changed while hashing")
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO pixel_hashes VALUES(?,?)",(identity,value))
        return value
