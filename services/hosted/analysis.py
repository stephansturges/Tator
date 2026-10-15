"""Canonical analysis admission and adapters; no API or model globals imported.

Numerical work remains owned by the existing DQE/pruning/training engines. This
module binds their input and output to an immutable approved dataset snapshot.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace
from typing import Any

from utils.capture_metadata import CAPTURE_METADATA_FIELDS, portable_capture_metadata

from .artifacts import ArtifactRef, ArtifactStore, open_regular
from .errors import DomainError
from .operations import HostedOperations
from .store import HostedStore, _id, _json
from .types import Actor, MutationContext


CONSUMERS = {"dqe", "dqe_review", "qwen_caption", "caption_export", "ingestion", "salad_training", "pruning", "statistics", "yolo_training", "sam3_training", "rfdetr_training", "qwen_training", "clip_training"}
SNAPSHOT_SCHEMA = "hosted-canonical-snapshot-v1"
CAPTURE_FIELDS = set(CAPTURE_METADATA_FIELDS)


def _hash_file(path: Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with open_regular(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _inside(path: str | Path, root: Path) -> Path:
    candidate = Path(path).absolute()
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise DomainError("analysis_path_invalid", 422)
    candidate = candidate.resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise DomainError("analysis_path_invalid", 422)
    return candidate


def read_canonical_snapshot(reference: dict, *, allowed_root: str | Path) -> dict:
    """Read the supervisor's pinned manifest, never a client-selected dataset.

    All paths are confined to one private capture directory. Hashes cover the
    exact manifest and image bytes; IDs and revisions must match its short IPC
    reference. This helper needs no application, registry or model imports.
    """
    root = Path(allowed_root).absolute()
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise DomainError("analysis_path_invalid", 422)
    root = root.resolve(strict=True)
    path = _inside(reference["manifest_path"], root)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise DomainError("analysis_manifest_too_large", 413)
    checksum, _ = _hash_file(path)
    if checksum != reference.get("manifest_sha256"):
        raise DomainError("analysis_manifest_changed", 409)
    with open_regular(path) as stream:
        manifest = json.load(stream, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if manifest.get("schema") != SNAPSHOT_SCHEMA:
        raise DomainError("analysis_snapshot_schema_invalid", 422)
    for key in ("snapshot_id", "dataset_id", "revision", "class_version", "settings_version"):
        if manifest.get(key) != reference.get(key):
            raise DomainError("analysis_snapshot_identity_changed", 409, {"field": key})
    seen = set()
    for image in manifest["images"]:
        if image["id"] in seen:
            raise DomainError("analysis_snapshot_duplicate_image", 422)
        seen.add(image["id"])
        media = _inside(image["media_path"], path.parent)
        actual, size = _hash_file(media)
        if actual != image["media_hash"] or size != image["size_bytes"]:
            raise DomainError("analysis_media_changed", 409, {"image_id": image["id"]})
    if _hash_file(path)[0] != checksum:
        raise DomainError("analysis_manifest_changed", 409)
    return manifest


def read_candidate_set(reference: dict, *, allowed_root: str | Path) -> dict:
    """Read separately captured Incoming media; this grants no canonical admission."""
    root = Path(allowed_root).absolute()
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise DomainError('analysis_path_invalid', 422)
    root = root.resolve(strict=True)
    path = _inside(reference['manifest_path'], root)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise DomainError('analysis_manifest_too_large', 413)
    checksum = _hash_file(path)[0]
    if checksum != reference.get('manifest_sha256'):
        raise DomainError('ingestion_candidates_changed', 409)
    with open_regular(path) as stream:
        manifest = json.load(stream, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if manifest.get('schema') != 'hosted-incoming-candidates-v1':
        raise DomainError('ingestion_candidate_schema_invalid', 422)
    if any(manifest.get(key) != reference.get(key) for key in ('candidate_set_id', 'dataset_id')):
        raise DomainError('ingestion_candidate_identity_changed', 409)
    seen = set()
    for image in manifest['images']:
        if image['id'] in seen:
            raise DomainError('ingestion_candidate_duplicate', 409)
        seen.add(image['id'])
        media = _inside(image['media_path'], path.parent)
        if _hash_file(media) != (image['media_hash'], image['size_bytes']):
            raise DomainError('ingestion_candidate_media_changed', 409)
    if _hash_file(path)[0] != checksum:
        raise DomainError('ingestion_candidates_changed', 409)
    return manifest


def _lines(image: dict, class_indices: dict[str, int]) -> tuple[list[str], list[dict]]:
    lines, mappings = [], []
    width, height = image["width"], image["height"]
    for ordinal, item in enumerate(image["annotations"].get("objects", [])):
        x, y, w, h = item["bbox"]
        numeric_class = class_indices[item["class_id"]]
        lines.append(" ".join([str(numeric_class), *(format(value, ".17g") for value in ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height))]))
        mappings.append({"label_row": ordinal, "object_id": item["id"], "class_id": item["class_id"], "class_index": numeric_class,
                         "bbox": item["bbox"], "polygon": item.get("polygon")})
    return lines, mappings


def legacy_source_manifest(manifest: dict, directory: str | Path) -> dict:
    """Translate identities and geometry only, without changing any recipe.

    The materialized tree uses stable hosted image IDs. Label-row maps recover
    stable object IDs when legacy results refer to numeric rows. Negative images
    remain explicit members even though DQE has no crop to embed for them.
    """
    directory = Path(directory)
    classes = sorted(manifest["classes"], key=lambda row: row["position"])
    indices = {row["id"]: index for index, row in enumerate(classes)}
    rows = []
    for image in manifest["images"]:
        # Materializer records the actual encoded suffix, independent of an old
        # user filename. Resolve exactly this ID, not arbitrary folder members.
        split = image["split"] if image["split"] != "unsplit" else "train"
        choices = list((directory / split / "images").glob(image["id"] + ".*"))
        if len(choices) != 1:
            raise DomainError("analysis_materialized_image_missing", 409, {"image_id": image["id"]})
        path = _inside(choices[0], directory.resolve())
        if _hash_file(path) != (image["media_hash"], image["size_bytes"]):
            raise DomainError("analysis_media_changed", 409, {"image_id": image["id"]})
        lines, objects = _lines(image, indices)
        annotations = image["annotations"]
        caption_records = []
        for key in ('captions', 'alternate_captions'):
            value = annotations.get(key, [])
            if value is not None and not isinstance(value, list):
                raise DomainError('analysis_caption_records_invalid', 409, {'image_id': image['id'], 'field': key})
            caption_records.extend(value or [])
        migration = annotations.get("migration") if isinstance(annotations.get("migration"), dict) else {}
        metadata = migration.get("metadata") if isinstance(migration.get("metadata"), dict) else {}
        explicit_capture = annotations.get("capture_metadata") if isinstance(annotations.get("capture_metadata"), dict) else {}
        capture = portable_capture_metadata({**metadata, **explicit_capture})
        rows.append({"split": split, "original_split": image["split"], "image_relpath": path.name, "image_path": str(path),
                     "image_name": image["filename"], "image_sha256": image["media_hash"], "label_lines": lines,
                     "label_source_present": True, "text_label": annotations.get("caption", ""),
                     "caption_records": caption_records,
                     "alternate_captions": annotations.get('alternate_captions', []),
                     "width": image["width"], "height": image["height"], "hosted_image_id": image["id"], "hosted_version_id": image["version_id"],
                     "hosted_objects": objects, "negative": bool(image["negative"]), "capture_metadata": capture,
                     "annotation_source_identity": "hosted:" + manifest["dataset_id"],
                     "annotation_entity_record_revision": image["version_id"], "annotation_identity_status": "ready",
                     "annotation_entities_by_label_row": [{"annotation_entity_id": "hosted:" + image["id"] + ":" + obj["object_id"],
                        "entity_revision": int(hashlib.sha256(image["version_id"].encode()).hexdigest()[:12], 16) + 1} for obj in objects], **capture})
    return {"snapshot_id": manifest.get("snapshot_id"), "dataset_id": manifest["dataset_id"], "session_revision": manifest["revision"],
            "labelmap": [row["name"] for row in classes], "class_ids": [row["id"] for row in classes], "images": rows,
            "image_count": len(rows), "dataset_label": manifest["dataset_name"], "yolo_layout": "split", "storage_mode": "hosted_snapshot"}


class CanonicalAnalysis:
    def __init__(self, store: HostedStore, artifacts: ArtifactStore, root: str | Path):
        self.store, self.artifacts, self.root = store, artifacts, Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.operations = HostedOperations(store, artifacts)
        with store.transaction() as con:
            con.execute("CREATE TABLE IF NOT EXISTS hosted_analysis_jobs (id TEXT PRIMARY KEY,snapshot_id TEXT NOT NULL REFERENCES hosted_snapshots(id),actor_id TEXT NOT NULL,consumer TEXT NOT NULL,request_json TEXT NOT NULL,state TEXT NOT NULL,result_sha256 TEXT,created_at REAL NOT NULL,updated_at REAL NOT NULL)")
            if "error_json" not in {row[1] for row in con.execute("PRAGMA table_info(hosted_analysis_jobs)")}:
                con.execute("ALTER TABLE hosted_analysis_jobs ADD COLUMN error_json TEXT")
            con.execute("CREATE TABLE IF NOT EXISTS hosted_analysis_artifacts(job_id TEXT NOT NULL REFERENCES hosted_analysis_jobs(id),relative_path TEXT NOT NULL,sha256 TEXT NOT NULL,size_bytes INTEGER NOT NULL,PRIMARY KEY(job_id,relative_path))")
            con.execute("CREATE TABLE IF NOT EXISTS hosted_analysis_views(job_id TEXT PRIMARY KEY REFERENCES hosted_analysis_jobs(id),payload_json TEXT NOT NULL)")
            con.execute('CREATE TABLE IF NOT EXISTS hosted_ingestion_candidates(id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),manifest_json TEXT NOT NULL,created_by TEXT NOT NULL,created_at REAL NOT NULL)')
            for verb in ('UPDATE', 'DELETE'):
                con.execute(f"CREATE TRIGGER IF NOT EXISTS hosted_ingestion_candidates_immutable_{verb.lower()} BEFORE {verb} ON hosted_ingestion_candidates BEGIN SELECT RAISE(ABORT,'immutable ingestion candidates'); END")

    def capture_incoming(self, actor: Actor, dataset_id: str, image_ids: list[str], operation_id: str) -> dict:
        if (not isinstance(image_ids, list) or not image_ids or len(image_ids) > 100000
                or any(not isinstance(value, str) or not value for value in image_ids) or len(set(image_ids)) != len(image_ids)):
            raise DomainError('ingestion_candidate_ids_invalid', 422)
        def run(con):
            dataset = self.store._dataset(con, dataset_id)
            images = []
            for image_id in image_ids:
                current = self.store._image(con, image_id)
                if current['dataset_id'] != dataset_id or current['state'] != 'unassigned' or current['assignee_id'] is not None:
                    raise DomainError('ingestion_candidate_not_unassigned_incoming', 409, {'image_id': image_id})
                row = con.execute('SELECT i.id,i.media_hash,i.filename,i.split,i.current_version AS version_id,i.generation,v.annotations_json,v.complete,v.negative,v.class_version,m.width,m.height,m.size_bytes FROM hosted_images i JOIN hosted_annotation_versions v ON v.id=i.current_version JOIN hosted_media m ON m.sha256=i.media_hash WHERE i.id=?', (image_id,)).fetchone()
                image = dict(row); image['annotations'] = json.loads(image.pop('annotations_json')); images.append(image)
            identifier = _id('candidates')
            manifest = {'schema': 'hosted-incoming-candidates-v1', 'candidate_set_id': identifier,
                        'dataset_id': dataset_id, 'dataset_name': dataset['name'], 'images': images}
            con.execute('INSERT INTO hosted_ingestion_candidates VALUES(?,?,?,?,?)', (identifier, dataset_id, _json(manifest), actor.user_id, self.store.clock()))
            return {'candidate_set_id': identifier, 'dataset_id': dataset_id, 'manifest': manifest}
        return self.store._command(actor, operation_id, 'capture_ingestion_candidates', {'dataset_id': dataset_id, 'image_ids': image_ids}, run, admin=True)

    def get_candidate_set(self, actor: Actor, candidate_set_id: str) -> dict:
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            row = con.execute('SELECT dataset_id,manifest_json FROM hosted_ingestion_candidates WHERE id=?', (candidate_set_id,)).fetchone()
            if row is None:
                raise DomainError('ingestion_candidate_set_not_found', 404)
            return {'candidate_set_id': candidate_set_id, 'dataset_id': row[0], 'manifest': json.loads(row[1])}

    def capture(self, actor: Actor, dataset_id: str, consumer: str, parameters: dict,
                operation_id: str, *, snapshot_id: str | None = None) -> dict:
        if consumer not in CONSUMERS or not isinstance(parameters, dict) or len(_json(parameters).encode()) > 1024 * 1024:
            raise DomainError("analysis_request_invalid", 422)
        payload = {"dataset_id": dataset_id, "consumer": consumer, "parameters": parameters, "snapshot_id": snapshot_id}
        def run(con):
            if snapshot_id is None:
                manifest = self.store._snapshot(con, dataset_id)
                captured_id = _id("snapshot")
                con.execute("INSERT INTO hosted_snapshots VALUES (?,?,?,?,?,?,?)", (captured_id, dataset_id, manifest["revision"], consumer, _json(manifest), actor.user_id, self.store.clock()))
            else:
                row = con.execute("SELECT dataset_id,manifest_json FROM hosted_snapshots WHERE id=?", (snapshot_id,)).fetchone()
                if row is None or row["dataset_id"] != dataset_id:
                    raise DomainError("snapshot_not_found", 404)
                captured_id, manifest = snapshot_id, json.loads(row["manifest_json"])
            job_id, now = _id("analysis"), self.store.clock()
            request = {**payload, "snapshot_id": captured_id, "dataset_revision": manifest["revision"], "class_version": manifest["class_version"], "settings_version": manifest.get("settings_version", 0),
                       "snapshot_sha256": hashlib.sha256(_json(manifest).encode()).hexdigest(), "settings": manifest.get("settings", {})}
            con.execute("INSERT INTO hosted_analysis_jobs(id,snapshot_id,actor_id,consumer,request_json,state,result_sha256,created_at,updated_at) VALUES (?,?,?,?,?,'captured',NULL,?,?)", (job_id, captured_id, actor.user_id, consumer, _json(request), now, now))
            return {"job_id": job_id, "snapshot_id": captured_id, "consumer": consumer, "dataset_id": dataset_id, "dataset_revision": manifest["revision"], "state": "captured"}
        return self.store._command(actor, operation_id, "capture_analysis", payload, run, admin=True)

    def _job(self, con: sqlite3.Connection, job_id: str) -> tuple[dict, dict, dict]:
        row = con.execute("SELECT j.*,s.manifest_json FROM hosted_analysis_jobs j JOIN hosted_snapshots s ON s.id=j.snapshot_id WHERE j.id=?", (job_id,)).fetchone()
        if row is None:
            raise DomainError("analysis_not_found", 404)
        job = dict(row)
        manifest, request = json.loads(job.pop("manifest_json")), json.loads(job.pop("request_json"))
        error = job.pop("error_json", None)
        job["error"] = json.loads(error) if error else None
        return job, manifest, request

    def mark_terminal(self, actor: Actor, job_id: str, state: str, error: dict | str | None, operation_id: str) -> dict:
        if state not in {"failed", "interrupted", "cancelled"} or len(_json(error).encode()) > 1024 * 1024:
            raise DomainError("analysis_terminal_state_invalid", 422)
        def run(con):
            job, _, _ = self._job(con, job_id)
            if job["state"] != "captured":
                raise DomainError("analysis_already_terminal", 409)
            con.execute("UPDATE hosted_analysis_jobs SET state=?,error_json=?,updated_at=? WHERE id=?", (state, _json(error), self.store.clock(), job_id))
            return {"job_id": job_id, "snapshot_id": job["snapshot_id"], "state": state, "error": error}
        return self.store._command(actor, operation_id, "analysis_terminal", {"job_id": job_id, "state": state, "error": error}, run, admin=True)

    def status(self, actor: Actor, job_id: str) -> dict:
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            job, manifest, request = self._job(con, job_id)
            current = con.execute("SELECT revision FROM hosted_datasets WHERE id=?", (manifest["dataset_id"],)).fetchone()
            archived = con.execute("SELECT 1 FROM hosted_archived_datasets WHERE dataset_id=?", (manifest["dataset_id"],)).fetchone()
            return {**job, "request": request, "dataset_id": manifest["dataset_id"], "dataset_revision": manifest["revision"],
                    "confirmed_count": len(manifest["images"]),
                    "stale": current is None or current[0] != manifest["revision"] or bool(archived)}

    def image_reference(self, actor: Actor, job_id: str, image_id: str) -> ArtifactRef:
        """Authorize historical media through this job's frozen populations."""
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            job, manifest, request = self._job(con, job_id)
            image = next((row for row in manifest["images"] if row["id"] == image_id), None)
            candidate_id = request.get("parameters", {}).get("candidate_set_id")
            if image is None and job["consumer"] == "ingestion" and isinstance(candidate_id, str):
                row = con.execute("SELECT dataset_id,manifest_json FROM hosted_ingestion_candidates WHERE id=?", (candidate_id,)).fetchone()
                if row and row["dataset_id"] == manifest["dataset_id"]:
                    candidates = json.loads(row["manifest_json"])
                    image = next((row for row in candidates["images"] if row["id"] == image_id), None)
            if image is None:
                raise DomainError("analysis_image_not_found", 404)
            return ArtifactRef(image["media_hash"], image["size_bytes"])

    def prepare(self, actor: Actor, job_id: str) -> dict:
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            job, manifest, request = self._job(con, job_id)
        directory = self.root / job_id
        destination = directory / "dataset"
        directory.mkdir(exist_ok=True)
        if not destination.exists():
            self.operations.materialize_snapshot(actor, job["snapshot_id"], destination, unsplit_as_train=True)
        # Verify against the frozen manifest even when a preparation is resumed.
        expected_files = set()
        for entry in self.operations._snapshot_entries(manifest, unsplit_as_train=True):
            expected_files.add(entry.path)
            path = _inside(destination / entry.path, destination.resolve())
            wanted = ((entry.artifact.sha256, entry.artifact.size_bytes) if entry.artifact is not None
                      else (hashlib.sha256(entry.data).hexdigest(), len(entry.data)))
            if _hash_file(path) != wanted:
                raise DomainError("analysis_materialization_changed", 409, {"path": entry.path})
        if {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()} != expected_files:
            raise DomainError("analysis_materialization_changed", 409)
        source = legacy_source_manifest({**manifest, "snapshot_id": job["snapshot_id"]}, destination)
        mapping = {row["hosted_image_id"]: {"version_id": row["hosted_version_id"], "split": row["split"], "original_split": row["original_split"], "image_relpath": row["image_relpath"], "objects": row["hosted_objects"]} for row in source["images"]}
        package = {"job_id": job_id, "snapshot_id": job["snapshot_id"], "dataset_root": str(destination), "manifest": source, "identity_map": mapping,
                   "request": request, "provenance": {"snapshot_sha256": request["snapshot_sha256"], "dataset_revision": manifest["revision"], "class_version": manifest["class_version"], "settings_version": manifest.get("settings_version", 0)}}
        with self.artifacts.exclusive():
            reference = self.artifacts.put_bytes(_json(package).encode())
            self.operations.register_artifact(actor, reference, "analysis_input", job_id, job_id + "-input-" + reference.sha256[:24])
        return {**package, "input_artifact": reference.as_dict()}

    def prepare_review_parent(self, actor: Actor, job_id: str, pin_directory: str | Path) -> dict:
        """Pin the exact retained DQE session for an isolated VLM review.

        The caller owns the private pin directory until the worker terminates.
        Browser requests never supply native paths or artifact inventories.
        """
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            job, manifest, _ = self._job(con, job_id)
            current = self.store._dataset(con, manifest['dataset_id'])
            row = con.execute('SELECT payload_json FROM hosted_analysis_views WHERE job_id=?', (job_id,)).fetchone()
            if job['consumer'] != 'dqe' or job['state'] != 'complete' or row is None:
                raise DomainError('dqe_result_unavailable', 409)
            if current['revision'] != manifest['revision']:
                raise DomainError('analysis_snapshot_stale', 409)
            native_id = json.loads(row[0]).get('native_job_id')
            if not isinstance(native_id, str) or not native_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in native_id):
                raise DomainError('analysis_native_job_invalid', 409)
            prefix = 'class_analysis/' + native_id + '/'
            rows = con.execute('SELECT relative_path,sha256,size_bytes FROM hosted_analysis_artifacts WHERE job_id=? ORDER BY relative_path', (job_id,)).fetchall()
        selected = [row for row in rows if row[0].startswith(prefix)]
        required = {'config.json', 'session.sqlite3'}
        if not required.issubset({row[0][len(prefix):] for row in selected}):
            raise DomainError('dqe_parent_artifacts_missing', 409)
        directory = Path(pin_directory)
        directory.mkdir(parents=True, exist_ok=True)
        inputs = []
        with self.artifacts.exclusive():
            for relative, digest, size in selected:
                name = relative[len(prefix):]
                if not name or '\\' in name or any(part in {'', '.', '..'} for part in name.split('/')):
                    raise DomainError('analysis_native_artifact_invalid', 409)
                self.artifacts.pin(ArtifactRef(digest, size), directory)
                inputs.append({'namespace': 'dqe_parent', 'relative_path': name, 'path': str((directory / digest).resolve()), 'sha256': digest, 'size_bytes': size})
        return {'snapshot_id': job['snapshot_id'], 'dataset_id': manifest['dataset_id'],
                'parent_analysis': {'job_id': job_id, 'native_job_id': native_id, 'snapshot_id': job['snapshot_id']},
                'cache_inputs': inputs}

    def dqe_request(self, actor: Actor, job_id: str) -> dict:
        """Native DQE active-workspace contract, bound to this capture only.

        The isolated engine must place CLASS_ANALYSIS_ROOT above dataset_root.
        All source/identity fields are server-owned and override request values.
        Existing native normalization still resolves the full numerical recipe.
        """
        package = self.prepare(actor, job_id)
        if package["request"]["consumer"] != "dqe":
            raise DomainError("analysis_consumer_mismatch", 409)
        return {**package["request"]["parameters"], "source_mode": "active_workspace", "mode": "active_workspace",
                "workspace_dir": package["dataset_root"], "workspace_manifest": package["manifest"],
                "workspace_manifest_path": "", "snapshot_id": package["snapshot_id"], "dataset_id": "", "session_id": "",
                "yolo_layout": "split", "labelmap": package["manifest"]["labelmap"]}

    def resolve_yolo_input(self, actor: Actor, job_id: str, *, task="detect") -> dict:
        """Call Tator's existing dataset resolver with one frozen registry entry."""
        from fastapi import HTTPException
        from services.datasets import _resolve_yolo_training_dataset_impl, _detect_yolo_layout_impl, _yolo_labels_have_polygons_impl
        from utils.io import _compute_dir_signature, _sanitize_yolo_run_id

        package = self.prepare(actor, job_id)
        if package["request"]["consumer"] != "yolo_training" or task not in {"detect", "segment"}:
            raise DomainError("analysis_consumer_mismatch", 409)
        if task == "segment":
            # Export keeps boxes and polygons separately. A segmentation
            # training adapter must explicitly select the polygon view.
            raise DomainError("analysis_polygon_training_adapter_required", 409)
        splits = {image["original_split"] for image in package["manifest"]["images"]}
        if "train" not in splits or "val" not in splits:
            raise DomainError("analysis_validation_split_required", 409, "Create an explicit grouped train/validation split before training; no image is silently reused for validation.")
        dataset_id = package["request"]["dataset_id"]
        entry = {"id": dataset_id, "dataset_root": package["dataset_root"], "signature": package["provenance"]["snapshot_sha256"]}
        def resolve(identifier):
            if identifier != dataset_id:
                raise DomainError("analysis_snapshot_identity_changed", 409)
            return entry
        def no_legacy_fallback(_identifier):
            raise DomainError("analysis_snapshot_identity_changed", 409)
        result = _resolve_yolo_training_dataset_impl(SimpleNamespace(dataset_id=dataset_id, task=task),
            resolve_dataset_entry_fn=resolve, resolve_sam3_or_qwen_dataset_fn=no_legacy_fallback,
            compute_dir_signature_fn=_compute_dir_signature, sanitize_yolo_run_id_fn=_sanitize_yolo_run_id,
            detect_yolo_layout_fn=_detect_yolo_layout_impl, yolo_labels_have_polygons_fn=_yolo_labels_have_polygons_impl,
            stable_hash_fn=lambda value: hashlib.sha256(_json(value).encode()).hexdigest(),
            yolo_cache_root=self.root / job_id / "training-cache", http_exception_cls=HTTPException)
        if result["prepared_root"] != package["dataset_root"]:
            raise DomainError("analysis_unqualified_training_cache", 409)
        return {**result, "hosted_snapshot_id": package["snapshot_id"], "hosted_provenance": package["provenance"], "identity_map": package["identity_map"]}

    def publish_result(self, actor: Actor, job_id: str, result: dict, operation_id: str,
                       *, artifacts: list[dict] | None = None) -> dict:
        """Attach engine output and exact recipe/cache/fit provenance unchanged.

        Retained artifacts must already be immutable blobs. Numerical results may
        remain useful after edits, but stale output cannot authorize mutations.
        """
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            job, _, request = self._job(con, job_id)
        wrapped = {"job_id": job_id, "snapshot_id": job["snapshot_id"], "request": request, "result": result, "artifacts": artifacts or []}
        with self.artifacts.exclusive():
            retained = []
            for item in artifacts or []:
                reference = ArtifactRef(item["sha256"], item["size_bytes"])
                self.artifacts.verify(reference)
                retained.append(reference)
            reference = self.artifacts.put_bytes(_json(wrapped).encode())
            def run(con):
                current, _, _ = self._job(con, job_id)
                if current["state"] != "captured":
                    raise DomainError("analysis_already_terminal", 409)
                for ref in [*retained, reference]:
                    con.execute("INSERT OR IGNORE INTO hosted_artifact_references VALUES ('analysis_result',?,?,?,?)", (job_id, ref.sha256, ref.size_bytes, self.store.clock()))
                    con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (ref.sha256,))
                con.execute("UPDATE hosted_analysis_jobs SET state='complete',result_sha256=?,updated_at=? WHERE id=?", (reference.sha256, self.store.clock(), job_id))
                return {"job_id": job_id, "state": "complete", "snapshot_id": job["snapshot_id"], "result": reference.as_dict()}
            return self.store._command(actor, operation_id, "publish_analysis_result", {"job_id": job_id, "sha256": reference.sha256}, run, admin=True)

    def result_reference(self, actor: Actor, job_id: str) -> ArtifactRef:
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            job, _, _ = self._job(con, job_id)
            row = con.execute("SELECT size_bytes FROM hosted_artifact_references WHERE kind='analysis_result' AND owner_id=? AND sha256=?", (job_id, job["result_sha256"])).fetchone()
            if row is None:
                raise DomainError("analysis_result_unavailable", 409)
            return ArtifactRef(job["result_sha256"], row[0])

    def publish_native_result(self, actor: Actor, job_id: str, output: dict,
                              operation_id: str, *, native_root: str | Path) -> dict:
        """Retain verified native files before the private worker tree expires.

        The small worker response identifies a hashed receipt, never thousands
        of filenames over IPC. All fit/model/cache outputs are registered with
        their relative paths so backup and later cache rehydration are possible.
        Exact retries remain usable after that temporary tree has been removed.
        """
        payload = {"job_id": job_id, "output": output}
        if self.store.receipt(actor, operation_id):
            return self.store._command(actor, operation_id, "publish_native_analysis", payload,
                lambda con: (_ for _ in ()).throw(DomainError("analysis_receipt_missing", 409)), admin=True)
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            job, manifest, request = self._job(con, job_id)
        if (output.get("snapshot_id") != job["snapshot_id"] or output.get("dataset_id") != manifest["dataset_id"]
                or output.get("revision") != manifest["revision"] or output.get("operation") != job["consumer"]):
            raise DomainError("analysis_native_result_mismatch", 409)
        root = Path(native_root).absolute().resolve(strict=True)
        directory = Path(output["native_directory"]).absolute()
        receipt_path = _inside(directory / "receipt.json", root)
        if _hash_file(receipt_path) != (output["receipt_sha256"], output["receipt_size_bytes"]):
            raise DomainError("analysis_native_receipt_changed", 409)
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("state") != "complete" or receipt.get("operation") != job["consumer"]:
            raise DomainError("analysis_native_result_mismatch", 409)
        relative_names = set()
        with self.artifacts.exclusive():
            retained = []
            result = None
            for item in receipt["artifacts"]:
                relative = item["relative_path"]
                if (not isinstance(relative, str) or "\\" in relative or relative.startswith("/")
                        or any(part in {"", ".", ".."} for part in relative.split("/")) or relative in relative_names):
                    raise DomainError("analysis_native_artifact_invalid", 409)
                relative_names.add(relative)
                path = _inside(directory / relative, directory.resolve(strict=True))
                reference = self.artifacts.put_file(path, expected_sha256=item["sha256"], expected_size=item["size_bytes"])
                retained.append({"relative_path": relative, **reference.as_dict()})
                if relative == receipt["result_path"]:
                    with self.artifacts.open(reference) as stream:
                        result = json.load(stream)
            if (not isinstance(result, dict) or result.get("snapshot_id") != job["snapshot_id"]
                    or result.get("dataset_id") != manifest["dataset_id"] or result.get("revision") != manifest["revision"]):
                raise DomainError("analysis_native_result_mismatch", 409)
            wrapped = {"job_id": job_id, "snapshot_id": job["snapshot_id"], "request": request,
                       "result": result, "artifacts": retained, "native_receipt_sha256": output["receipt_sha256"]}
            engine = result.get("engine_result") or {}
            public = engine.get("result") if isinstance(engine.get("result"), dict) else engine
            view = {"native_job_id": engine.get("native_job_id"), "consumer": job["consumer"],
                    "image_count": len(manifest["images"]),
                    "result": {key: value for key, value in public.items() if key not in {"points"}},
                    "dataset_id": manifest["dataset_id"], "snapshot_id": job["snapshot_id"], "revision": manifest["revision"]}
            reference = self.artifacts.put_bytes(_json(wrapped).encode())
            def run(con):
                current, _, _ = self._job(con, job_id)
                if current["state"] != "captured":
                    raise DomainError("analysis_already_terminal", 409)
                for item in [*retained, reference.as_dict()]:
                    con.execute("INSERT OR IGNORE INTO hosted_artifact_references VALUES ('analysis_result',?,?,?,?)",
                                (job_id, item["sha256"], item["size_bytes"], self.store.clock()))
                    con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (item["sha256"],))
                con.executemany("INSERT INTO hosted_analysis_artifacts VALUES (?,?,?,?)",
                                [(job_id, item["relative_path"], item["sha256"], item["size_bytes"]) for item in retained])
                con.execute("INSERT INTO hosted_analysis_views VALUES (?,?)", (job_id, _json(view)))
                con.execute("UPDATE hosted_analysis_jobs SET state='complete',result_sha256=?,updated_at=? WHERE id=?",
                            (reference.sha256, self.store.clock(), job_id))
                return {"job_id": job_id, "state": "complete", "snapshot_id": job["snapshot_id"], "result": reference.as_dict()}
            return self.store._command(actor, operation_id, "publish_native_analysis", payload, run, admin=True)

    def apply_annotations(self, actor: Actor, job_id: str, context: MutationContext,
                          annotations: dict, *, complete=True, negative=False) -> dict:
        """Explicit human application under the same image and population fence.

        No legacy annotation writer is called. The current editor lease and full
        snapshot revision are checked in the transaction creating the version.
        """
        payload = {"job_id": job_id, "context": asdict(context), "annotations": annotations, "complete": complete, "negative": negative}
        def run(con):
            job, manifest, _ = self._job(con, job_id)
            if job["state"] != "complete":
                raise DomainError("analysis_result_unavailable", 409)
            dataset = self.store._dataset(con, manifest["dataset_id"])
            if dataset["revision"] != manifest["revision"]:
                raise DomainError("analysis_snapshot_stale", 409)
            original = next((image for image in manifest["images"] if image["id"] == context.image_id), None)
            if original is None or original["version_id"] != context.expected_version:
                raise DomainError("analysis_image_not_in_snapshot", 409)
            image = self.store._guard(con, actor, context)
            if image["state"] != "confirmed" or image["dataset_id"] != manifest["dataset_id"]:
                raise DomainError("analysis_image_not_confirmed", 409)
            # Use the exact store save implementation in this transaction; it
            # owns geometry/completeness validation, versions and canonical revision.
            return self.store._save(con, actor, context, annotations, complete=complete, negative=negative)
        return self.store._command(actor, context.operation_id, "apply_analysis_annotations", payload, run, admin=True)
