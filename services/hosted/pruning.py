"""Hosted persistence/admission for the unchanged pruning review service.

The numerical planner, direct pair matcher, revisioned ledger, Undo, grouping
and approved-only ZIP export remain services.dataset_pruning implementations.
Only immutable input hydration, auth and recoverable storage are adapted here.
"""
from contextlib import contextmanager, closing
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import sys
import tempfile
import threading
import uuid
import zipfile

import numpy as np
from fastapi import HTTPException

from services.dataset_pruning import digest
from services.dataset_pruning_cache import file_state
from services.dataset_pruning_comparison import SelectedObjectFeatures
from services.dataset_pruning_jobs import PruningService, read_json, write_json
from services.dataset_pruning_review import ReviewLedger
from .artifacts import ArtifactRef
from .errors import DomainError
from .store import _json
from .pruning_components import RetainedComponents, recover_retained_components


def _relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise DomainError("pruning_artifact_path_invalid", 409)
    return path


class FrozenRows:
    def __init__(self, base, directory):
        self.base, self.directory = base, directory
        self.raw = RetainedComponents(directory)
    def __getattr__(self, name):
        return getattr(self.base, name)
    def replay_object_rows(self, images, recipe, manifest):
        original = read_json(self.directory / "result.json")
        if digest(recipe) != digest(original["recipe"]) or digest(manifest) != digest(original['recipe'].get("component_manifest", {})):
            return self.raw.replay(images, recipe, manifest, original['recipe'])
        path = self.directory / "hosted_objects.npy"
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            rows = {obj["feature_index"]: np.array(values[obj["feature_index"]], copy=True)
                    for image in images for obj in image["objects"]}
            if values.ndim != 2 or values.dtype != np.float32 or any(not np.isfinite(row).all() or abs(float(np.linalg.norm(row)) - 1) > 1e-4 for row in rows.values()):
                raise HTTPException(409, "Saved fitted object rows are invalid")
            return SelectedObjectFeatures(rows, values.shape[1])
        finally:
            values._mmap.close()


class FrozenAdapter:
    def __init__(self, owner, actor, job_id, manifest):
        self.owner, self.actor, self.job_id, self.manifest = owner, actor, job_id, manifest
        self.api = None  # Final fitted feature replay needs no encoder globals.
    def review_ledger(self, path):
        return FencedLedger(path, self)
    @contextmanager
    def calibration_commit_guard(self):
        # Diagnostic profiles are separate SQLite records. Hold authority only
        # while publishing them, never while measuring recipe variants.
        with self.owner.store.transaction() as con:
            self.owner.store._actor(con, self.actor, admin=True)
            current = self.owner.store._dataset(con, self.manifest['dataset_id'])
            if current['revision'] != self.manifest['revision']:
                raise HTTPException(409, 'The approved dataset changed before calibration commit')
            yield
    @contextmanager
    def guard(self):
        self.owner.analysis.status(self.actor, self.job_id)
        yield
        self.owner.analysis.status(self.actor, self.job_id)
    def check_revision(self, source):
        if self.owner.analysis.status(self.actor, self.job_id)["stale"]:
            raise HTTPException(409, "The approved dataset changed. Run a new analysis before changing this pruning review.")
        if source["revision"] != self.manifest["revision"]:
            raise HTTPException(409, "Frozen source revision changed")
    def check_inventory(self, source, images):
        wanted = {row["id"]: row for row in self.manifest["images"]}
        if len(images) != len(wanted):
            raise HTTPException(409, "Frozen pruning membership changed")
        for image in images:
            identifier = Path(image["image_relpath"]).stem
            frozen = wanted.get(identifier)
            if frozen is None or frozen["media_hash"] != image.get("source_image_sha256", image["image_sha256"]) or file_state(Path(image["image_path"])) != image["image_state"]:
                raise HTTPException(409, "Frozen pruning media changed")
    @staticmethod
    def export_file_state(path):
        value = Path(path).stat()
        return value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino
    def disk_capacity(self, directory, plan):
        size = sum(len(row["data"]) if isinstance(row.get("data"), bytes) else row["state"][0] for row in plan)
        if shutil.disk_usage(directory).free < 2 * size + max(512 * 1024**2, min(4 * 1024**3, size // 10)):
            raise HTTPException(507, "Insufficient space for the frozen dataset variant")
        return size
    def freeze_export(self, plan, directory, progress):
        directory.mkdir(parents=True)
        frozen = []
        for index, entry in enumerate(plan, 1):
            if isinstance(entry.get("data"), bytes):
                data = entry["data"]
                frozen.append({**entry, "content_sha256": hashlib.sha256(data).hexdigest(), "content_size": len(data)})
            else:
                source, expected = Path(entry["path"]), tuple(entry["state"])
                if self.export_file_state(source) != expected:
                    raise HTTPException(409, "Export source changed")
                target = directory / str(index)
                value, size = hashlib.sha256(), 0
                with source.open("rb") as src, target.open("xb") as dst:
                    for block in iter(lambda: src.read(1024 * 1024), b""):
                        dst.write(block); value.update(block); size += len(block)
                if self.export_file_state(source) != expected:
                    raise HTTPException(409, "Export source changed")
                frozen.append({**entry, "path": target, "state": self.export_file_state(target), "content_sha256": value.hexdigest(), "content_size": size})
            progress(index, len(plan))
        return frozen
    def write_export(self, plan, path, progress):
        temporary = path.with_name("." + uuid.uuid4().hex + ".zip")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for index, row in enumerate(plan, 1):
                    _relative(row["arcname"])
                    if isinstance(row.get("data"), bytes):
                        archive.writestr(row["arcname"], row["data"])
                    else:
                        if self.export_file_state(row["path"]) != tuple(row["state"]):
                            raise HTTPException(409, "Frozen export source changed")
                        archive.write(row["path"], row["arcname"])
                        if self.export_file_state(row["path"]) != tuple(row["state"]):
                            raise HTTPException(409, "Frozen export source changed")
                    progress(index, len(plan))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    def register_export(self, path):
        # Terminal run_export performs the checkpoint after state publication.
        self.owner.analysis.status(self.actor, self.job_id)


class FencedLedger(ReviewLedger):
    def __init__(self, path, adapter):
        super().__init__(path)
        self.adapter = adapter
    @contextmanager
    def transaction(self):
        fence = None
        try:
            with super().transaction() as state:
                yield state
                # Only the final ledger commit holds the canonical writer
                # fence. Expensive planning and preview work happened earlier.
                store = self.adapter.owner.store
                pending = store.transaction()
                con = pending.__enter__()
                fence = pending
                store._actor(con, self.adapter.actor, admin=True)
                dataset = store._dataset(con, self.adapter.manifest['dataset_id'])
                if dataset['revision'] != self.adapter.manifest['revision']:
                    raise HTTPException(409, 'The approved dataset changed before review commit')
                if con.execute('SELECT 1 FROM hosted_archived_datasets WHERE dataset_id=?', (dataset['id'],)).fetchone():
                    raise HTTPException(409, 'The approved dataset was archived before review commit')
        finally:
            if fence is not None:
                fence.__exit__(*sys.exc_info())


class HostedPruningService(PruningService):
    def recipe_lab(self, job_id):
        value = super().recipe_lab(job_id)
        if value['available'] and not (self.directory(job_id) / 'hosted_raw_components/manifest.json').is_file():
            reason = self.directory(job_id) / 'hosted-raw-unavailable.json'
            value.update(available=False, reason=read_json(reason)['reason'] if reason.is_file() else
                'Raw object components were not retained by this analysis. Run this recipe again to enable comparisons.')
        return value
    def _run_settings_preview(self, *args):
        try:
            return super()._run_settings_preview(*args)
        finally:
            self.adapter.owner.checkpoint(self.adapter.job_id, self)
    def run_export(self, *args):
        try:
            return super().run_export(*args)
        finally:
            self.adapter.owner.checkpoint(self.adapter.job_id, self)


class HostedPruning:
    def __init__(self, analysis):
        self.analysis, self.store, self.artifacts = analysis, analysis.store, analysis.artifacts
        self.root = analysis.root / "pruning-reviews"
        self.root.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.job_locks, self.services, self.file_refs = {}, {}, {}
        with self.store.transaction() as con:
            con.execute("CREATE TABLE IF NOT EXISTS hosted_pruning_checkpoints(job_id TEXT PRIMARY KEY REFERENCES hosted_analysis_jobs(id),manifest_json TEXT NOT NULL,updated_at REAL NOT NULL)")
    def _job_lock(self, job_id):
        with self.lock:
            return self.job_locks.setdefault(job_id, threading.RLock())
    def service(self, actor, job_id):
        self.analysis.status(actor, job_id)
        key = (job_id, actor.user_id, actor.session_id, actor.instance_id)
        with self._job_lock(job_id):
            if key in self.services:
                return self.services[key]
            with self.store.read_connection() as con:
                self.store._actor(con, actor, admin=True)
                job, manifest, _ = self.analysis._job(con, job_id)
                row = con.execute("SELECT payload_json FROM hosted_analysis_views WHERE job_id=?", (job_id,)).fetchone()
                saved = con.execute("SELECT manifest_json FROM hosted_pruning_checkpoints WHERE job_id=?", (job_id,)).fetchone()
                refs = con.execute("SELECT relative_path,sha256,size_bytes FROM hosted_analysis_artifacts WHERE job_id=?", (job_id,)).fetchall()
            if job["consumer"] != "pruning" or job["state"] != "complete" or row is None:
                raise DomainError("pruning_analysis_unavailable", 409)
            metadata = json.loads(row[0]); native_id = metadata["native_job_id"]
            if not isinstance(native_id, str) or not __import__('re').fullmatch(r'dp_[a-f0-9]{16}', native_id):
                raise DomainError("pruning_job_identity_invalid", 409)
            base = self.root / job_id
            target = base / native_id
            if not (target / "hosted-hydrated.json").is_file():
                if base.exists():
                    # An interrupted hydration has no published ledger authority.
                    shutil.rmtree(base)
                base.mkdir()
                prefix = f"class_analysis/pruning/{native_id}/"
                inventory = json.loads(saved[0])["files"] if saved else {
                    value["relative_path"][len(prefix):]: {"sha256": value["sha256"], "size_bytes": value["size_bytes"]}
                    for value in refs if value["relative_path"].startswith(prefix)
                    and not value["relative_path"][len(prefix):].startswith(("planning/", "thumbnails/", "exports/"))}
                target.mkdir()
                for relative, ref in inventory.items():
                    destination = target / _relative(relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with self.artifacts.open(ArtifactRef(**ref)) as stream, destination.open("xb") as out:
                        shutil.copyfileobj(stream, out)
                if not (target / "hosted_objects.npy").is_file():
                    raise DomainError("pruning_fitted_rows_unavailable", 409, "Run the native pruning analysis again to retain its fitted object rows.")
                prepared = self.analysis.prepare(actor, job_id)
                media = {Path(row["image_relpath"]).stem: row for row in prepared["manifest"]["images"]}
                images = read_json(target / "images.json")
                if not (target / 'hosted_raw_components/manifest.json').is_file():
                    result = read_json(target / 'result.json')
                    try:
                        recover_retained_components(target, images, result['recipe'],
                            result['recipe'].get('component_manifest', {}), self.artifacts, refs)
                    except (HTTPException, ValueError, KeyError, IndexError, sqlite3.DatabaseError) as error:
                        write_json(target / 'hosted-raw-unavailable.json', {'reason':
                            'This older analysis cannot recover its raw components from retained artifacts: ' +
                            str(error.detail if isinstance(error, HTTPException) else error)})
                for image in images:
                    row = media.get(Path(image["image_relpath"]).stem)
                    if row is None or row["image_sha256"] != image["image_sha256"]:
                        raise DomainError("pruning_snapshot_membership_changed", 409)
                    image.update(image_path=row["image_path"], image_state=file_state(Path(row["image_path"])))
                write_json(target / "images.json", images)
                source = read_json(target / "source.json")
                source.update(dataset_root=prepared["dataset_root"], annotation_states={})
                write_json(target / "source.json", source)
                # Restored file identities invalidate old proof/preview bindings.
                # Keep intent and receipts survive; old approvals cannot silently
                # become approval of freshly computed accounting.
                if saved:
                    with ReviewLedger(target / "review.sqlite3").transaction() as state:
                        state["plans"] = {}
                        for decision in state["decisions"].values():
                            if decision.get("status") == "approved": decision["status"] = "superseded"
                        state["review_revision"] += 1; state["plan_revision"] += 1
                        state["hosted_recovery_notice"] = "Review restored. Recalculate proposals and review approvals against the restored file identities."
                    shutil.rmtree(target / "planning", ignore_errors=True)
                    shutil.rmtree(target / "exports", ignore_errors=True)
                write_json(target / "hosted-hydrated.json", {"job_id": job_id, "snapshot_id": job["snapshot_id"]})
            adapter = FrozenAdapter(self, actor, job_id, manifest)
            service = HostedPruningService(base, adapter)
            service.lock = self._job_lock(job_id)
            service.cache = FrozenRows(service.cache, target)
            service.hosted_native_id = native_id
            self.services[key] = service
            return service
    def checkpoint(self, job_id, service):
        """Freeze files plus the native SQLite ledger into the recovery closure."""
        with self._job_lock(job_id), self.artifacts.exclusive():
            target = service.directory(service.hosted_native_id)
            rows = {}
            known = self.file_refs.setdefault(job_id, {})
            def generation_of(path):
                primary = file_state(path)
                try:
                    wal = file_state(Path(str(path) + '-wal'))
                except FileNotFoundError:
                    wal = None
                return primary, wal
            with tempfile.TemporaryDirectory(prefix="pruning-checkpoint-", dir=self.root) as temporary:
                # Planning includes concurrently-created/deleted coverage spill
                # directories. Recovery intentionally invalidates those proofs;
                # preserve the ledger and immutable inputs, not live scratch.
                paths = []
                for current, directories, filenames in os.walk(target, topdown=True):
                    directories[:] = [name for name in directories if not name.startswith('.')
                                      and name not in {'planning', 'thumbnails', 'snapshot'}]
                    paths.extend(Path(current) / name for name in filenames if not name.startswith('.'))
                for path in sorted(paths):
                    relative = path.relative_to(target).as_posix()
                    if not path.is_file() or path.is_symlink() or any(part.startswith(".") for part in path.relative_to(target).parts) or relative.startswith("thumbnails/") or "/snapshot/" in relative or path.name.endswith(("-wal", "-shm")):
                        continue
                    if relative.startswith('exports/'):
                        state_path = path.parent / 'state.json'
                        if not state_path.is_file() or read_json(state_path).get('status') != 'completed':
                            continue
                    generation = generation_of(path)
                    previous = known.get(relative)
                    if previous and previous[0] == generation:
                        rows[relative] = previous[1]
                        continue
                    # sqlite backup includes committed WAL state, never half a
                    # writer transaction; other files publish by atomic replace.
                    if path.suffix == ".sqlite3":
                        copy_path = Path(temporary) / (uuid.uuid4().hex + ".sqlite3")
                        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as src, closing(sqlite3.connect(copy_path)) as dst:
                            src.backup(dst)
                        ref = self.artifacts.put_file(copy_path)
                    else:
                        ref = self.artifacts.put_file(path)
                    rows[relative] = {"sha256": ref.sha256, "size_bytes": ref.size_bytes}
                    if generation == generation_of(path):
                        known[relative] = (generation, rows[relative])
                with self.store.transaction() as con:
                    for ref in rows.values():
                        con.execute("INSERT OR IGNORE INTO hosted_artifact_references VALUES(?,?,?,?,?)",
                                    ("pruning_review", job_id, ref["sha256"], ref["size_bytes"], self.store.clock()))
                        con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (ref["sha256"],))
                    con.execute("INSERT OR REPLACE INTO hosted_pruning_checkpoints VALUES(?,?,?)", (job_id, _json({"files": rows}), self.store.clock()))
    def call(self, actor, job_id, method, *args, **kwargs):
        service = self.service(actor, job_id)
        with self._job_lock(job_id):
            with service.adapter.guard():
                value = getattr(service, method)(service.hosted_native_id, *args, **kwargs)
            # Reads can reconcile cross-analysis Keep events; retain that state.
            if method not in {"thumbnail", "crop", "download", "status"}:
                self.checkpoint(job_id, service)
            return value
