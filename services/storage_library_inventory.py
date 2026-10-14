"""Passive discovery of existing authorities and bounded filesystem accounting."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import sqlite3
import stat
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from services.storage_library import StorageLibrary, StorageLibraryError, absolute, contained, digest, encode, reject_symlinks


_JOB = re.compile(r"ca_[A-Za-z0-9_-]+\Z")
_CACHE_CATEGORIES = {"image_packs", "resume_embeddings", "embeddings", "thumbnails"}
_SOURCE_FILES = {"config.json", "session_manifest.json", "job_state.json", "metadata.jsonl", "thumbnail_index.json", "points.sqlite3", "session.sqlite3"}


def read_metadata(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
    reject_symlinks(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 1 < info.st_size <= max_bytes:
        raise ValueError("Metadata is missing or exceeds its bounded size.")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Metadata is not an object.")
    return payload


class StorageLibraryInventory:
    def __init__(self, library: StorageLibrary, *, dataset_roots: Mapping[str, Path | str] | None = None,
                 transient_root: Path | str | None = None, export_root: Path | str | None = None,
                 upload_roots: Mapping[str, Path | str] | None = None, entity_store_path: Path | str | None = None):
        self.library = library
        self.dataset_roots = {key: absolute(value) for key, value in (dataset_roots or {}).items()}
        self.transient_root = absolute(transient_root) if transient_root else None
        self.export_root = absolute(export_root) if export_root else None
        self.upload_roots = {key: absolute(value) for key, value in (upload_roots or {}).items()}
        self.entity_store_path = absolute(entity_store_path) if entity_store_path else None
        self.library.roots.update(self.dataset_roots)
        self.library.roots.update(self.upload_roots)
        for key, value in (("transient", self.transient_root), ("exports", self.export_root), ("entity", self.entity_store_path.parent if self.entity_store_path else None)):
            if value is not None:
                self.library.roots.setdefault(key, value)
        self._generation = ""
        self._errors: list[dict[str, str]] = []

    def _children(self, path: Path):
        if not path.exists():
            return
        try:
            reject_symlinks(path)
            with os.scandir(path) as iterator:
                for entry in iterator:
                    if entry.name.startswith(".tator-delete-"):
                        continue
                    yield Path(entry.path)
        except (OSError, StorageLibraryError) as exc:
            self._errors.append({"path": str(path), "error": str(exc)})

    def _asset(self, path: Path, category: str, *, ownership: str = "managed", **metadata: Any) -> str:
        return self.library.register_asset(path, category=category, ownership=ownership, metadata=metadata)

    def _item(self, identifier: str, kind: str, authority: dict[str, Any], summary: dict[str, Any], refs=()):
        return self.library.register_item(identifier, kind=kind, authority=authority, summary=summary, references=refs, generation=self._generation)

    def _unclassified(self, path: Path, label: str = "Unclassified files") -> None:
        asset = self._asset(path, "unknown", ownership="protected")
        self._item("unknown:" + digest(str(path))[:24], "cleanup", {"kind": "unknown", "path": str(path)},
                   {"label": label, "state": "unclassified", "capabilities": {"cleanup_actions": [], "reasons": ["Ownership has not been established; these files are protected."]}}, [(asset, "protected")])

    def _datasets(self) -> dict[str, list[str]]:
        identities = {}
        for root_id, root in self.dataset_roots.items():
            for child in self._children(root):
                if child.name.startswith(".") or not child.is_dir() or child.is_symlink():
                    continue
                metadata = None
                error = ""
                for name in ("dataset.json", "sam3_dataset.json", "metadata.json"):
                    if (child / name).exists():
                        try:
                            metadata = read_metadata(child / name)
                        except (OSError, ValueError, StorageLibraryError) as exc:
                            error = str(exc)
                        break
                if metadata is None:
                    self._unclassified(child, child.name)
                    continue
                dataset_id = str(metadata.get("id") or child.name)
                identifier = f"dataset:{root_id}:{dataset_id}"
                identities.setdefault(dataset_id, []).append(identifier)
                mode = str(metadata.get("storage_mode") or ("linked" if metadata.get("linked_root") else "managed"))
                linked = str(metadata.get("linked_root") or "")
                source_root = absolute(linked) if mode == "linked" and linked else child
                refs = [(self._asset(child, "dataset_metadata" if mode == "linked" else "dataset", storage_mode=mode), "dataset")]
                if mode == "linked" and linked:
                    external_key = "external:" + digest(str(source_root))[:24]
                    self.library.roots[external_key] = source_root
                    refs.append((self._asset(source_root, "external_source", ownership="external"), "source"))
                source_ok = source_root.is_dir() and not source_root.is_symlink()
                if (child / ".annotation_overlay").exists():
                    refs.append((self._asset(child / ".annotation_overlay", "human_work", ownership="protected"), "human_work"))
                self._item(identifier, "dataset", {"kind": "dataset", "root_id": root_id, "key": dataset_id, "path": str(child)},
                    {"label": str(metadata.get("label") or dataset_id), "state": "available" if source_ok else "source_missing",
                     "created_at": metadata.get("created_at"), "storage_mode": mode,
                     "restored_from_trash_id": metadata.get("restored_from_trash_id"),
                     "source": {"mode": "linked", "id": dataset_id, "dataset_root": str(source_root), "annotation_revision": metadata.get("annotation_revision", 0)},
                     "image_count": metadata.get("image_count"), "classes": metadata.get("classes") or [],
                     "capabilities": {"open_dataset": source_ok, "can_edit": source_ok, "cleanup_actions": ["unlink_dataset"] if mode == "linked" else ["trash_dataset"], "reasons": [error] if error else ([] if source_ok else ["The source directory is unavailable."])}}, refs)
            for trash in self._children(root / ".trash"):
                if not trash.is_dir() or trash.is_symlink():
                    continue
                try:
                    metadata = read_metadata(trash / "deleted_dataset.json")
                except (OSError, ValueError, StorageLibraryError):
                    self._unclassified(trash, "Unrecognized dataset trash")
                    continue
                asset = self._asset(trash, "trash", original_path=metadata.get("original_path"))
                trash_identifier = f"trash:{root_id}:{trash.name}"
                self._item(trash_identifier, "trash", {"kind": "trash", "root_id": root_id, "key": trash.name, "path": str(trash)},
                           {"label": metadata.get("label") or metadata.get("original_id") or trash.name,
                            "state": "trashed", "deleted_at": metadata.get("deleted_at"), "source": {"mode": "trash", "id": trash.name},
                            "capabilities": {"restore": True, "restore_url": "/storage/library/items/" + trash_identifier + "/restore", "cleanup_actions": ["empty_trash"], "reasons": []}}, [(asset, "trash")])
        return identities

    def _sessions(self, dataset_ids: Mapping[str, list[str]]) -> None:
        for job_dir in self._children(self.library.root):
            if not _JOB.fullmatch(job_dir.name) or not job_dir.is_dir() or job_dir.is_symlink():
                continue
            manifest, config, errors = {}, {}, []
            for name, target in (("session_manifest.json", manifest), ("config.json", config)):
                try:
                    target.update(read_metadata(job_dir / name))
                except (OSError, ValueError, StorageLibraryError) as exc:
                    errors.append(f"{name}: {exc}")
            manifest_source = manifest.get("source")
            invalid_source = manifest_source is not None and not isinstance(manifest_source, Mapping)
            if invalid_source:
                errors.append("session_manifest.json: source metadata is not an object; this review remains protected.")
            source = {} if invalid_source else dict(manifest_source or {})
            source.setdefault("mode", str(config.get("source_mode") or ""))
            source.setdefault("id", str(config.get("dataset_id") or config.get("session_id") or config.get("source_id") or ""))
            source.setdefault("snapshot_id", str(config.get("snapshot_id") or config.get("workspace_id") or ""))
            refs, source_refs = [], []
            dataset_matches = list(dataset_ids.get(str(source.get("id") or ""), []))
            parent_id = dataset_matches[0] if len(dataset_matches) == 1 else ""
            if len(dataset_matches) > 1:
                errors.append("Dataset identity is ambiguous across configured registries; reconnect the exact source.")
                refs.extend((asset["id"], "ambiguous_source") for asset in self.library.assets_for_items(dataset_matches))
            workspace_value = str(config.get("workspace_dir") or "")
            if source.get("mode") == "active_workspace" and workspace_value:
                workspace = absolute(workspace_value)
                if contained(workspace, self.library.root) and not workspace.is_symlink():
                    snapshot_id = str(source.get("snapshot_id") or "unknown")
                    parent_id = f"snapshot:{snapshot_id}:{digest(str(workspace))[:12]}"
                    for path in (workspace / "images", workspace / "snapshot.json"):
                        source_refs.append((self._asset(path, "snapshot_images" if path.name == "images" else "snapshot_identity"), "source"))
                    input_manifest = absolute(config.get("workspace_manifest_path") or workspace / "manifest.json")
                    if contained(input_manifest, workspace):
                        refs.append((self._asset(input_manifest, "annotation_input", ownership="protected"), "annotation_input"))
                    source.update({"workspace_dir": str(workspace), "workspace_manifest_path": str(input_manifest), "job_id": job_dir.name})
                    source_ok = (workspace / "images").is_dir() and input_manifest.is_file()
                    try:
                        snapshot = read_metadata(workspace / "snapshot.json")
                    except (OSError, ValueError, StorageLibraryError):
                        snapshot = {}
                    self._item(parent_id, "dataset", {"kind": "snapshot", "key": snapshot_id, "path": str(workspace), "job_id": job_dir.name},
                        {"label": snapshot.get("dataset_label") or source.get("dataset_label") or "Uploaded analysis dataset",
                         "state": "available" if source_ok else "source_missing", "source": source,
                         "image_count": source.get("image_count"), "storage_mode": "snapshot",
                         "capabilities": {"open_dataset": source_ok, "can_edit": source_ok, "cleanup_actions": ["trash_dataset"], "reasons": [] if source_ok else ["The saved source files are missing."]}}, source_refs)
                else:
                    errors.append("Snapshot workspace is outside the managed analysis root.")
            refs.extend(source_refs)
            if parent_id and not source_refs:
                try:
                    refs.extend((asset["id"], "source") for asset in self.library.assets_for_items([parent_id]))
                except StorageLibraryError:
                    pass
            source_ok = bool(source_refs) and all(self.library.asset_path(asset).exists() for asset in self.library.assets_for_items([parent_id])) if source_refs else bool(parent_id)
            known = set()
            for child in self._children(job_dir):
                if child.name == "active_workspace":
                    continue
                if child.name == "thumbnails":
                    thumbnail_asset = self._asset(child, "rebuildable_cache", cache_category="session_thumbnails")
                    self._item("cache:" + digest(str(child))[:24], "cleanup", {"kind": "cache", "path": str(child), "category": "session_thumbnails"},
                               {"label": "Review thumbnails · " + job_dir.name, "parent_id": f"session:{job_dir.name}", "state": "available",
                                "capabilities": {"cleanup_actions": ["clear_cache"], "reasons": []}}, [(thumbnail_asset, "cache")])
                    continue
                known.add(child.name)
                if child.name in {"session.sqlite3", "session.sqlite3-wal", "session.sqlite3-shm", "points.sqlite3", "points.sqlite3-wal", "points.sqlite3-shm"}:
                    category = "analysis_files"
                elif child.name == "retained_review.sqlite3":
                    category = "human_work"
                elif child.name in _SOURCE_FILES or child.name.endswith((".sqlite3-wal", ".sqlite3-shm")):
                    category = "session_identity"
                elif child.name == "annotation_source":
                    category = "human_work"
                elif child.name == "qwen_reviews":
                    category = "review_evidence"
                elif child.name in {"embeddings.npy", "projection_coordinates.npz", "result.json", "thumbnails", "patch_refinement.npz", "patch_refinement_manifest.json", "review_ranking.json"} or child.name.startswith("projection_"):
                    category = "analysis_files"
                else:
                    category = "unknown"
                refs.append((self._asset(child, category, ownership="protected" if category == "unknown" else "managed"), category))
            plot_ok = not invalid_source and manifest.get("schema") == "class-analysis-session-manifest-v1" and (job_dir / "session.sqlite3").is_file()
            legacy_ok = not invalid_source and manifest.get("schema") == "class-analysis-session-manifest-v1" and (job_dir / "result.json").is_file()
            actions = [] if invalid_source else ["delete_review"]
            if not invalid_source and any(role == "analysis_files" for _, role in refs):
                actions.insert(0, "clear_analysis_files")
            self._item(f"session:{job_dir.name}", "session", {"kind": "session", "key": job_dir.name, "path": str(job_dir)},
                {"label": source.get("dataset_label") or source.get("id") or job_dir.name, "job_id": job_dir.name,
                 "parent_id": parent_id, "state": "available" if plot_ok or legacy_ok else "incomplete",
                 "created_at": manifest.get("created_at"), "source": source, "recipe": manifest.get("recipe") or {},
                 "analysis_scope": config.get("analysis_scope") or config.get("scope") or "selected_class", "class_names": config.get("class_names") or ([config["class_name"]] if config.get("class_name") else []),
                 "object_count": source.get("object_count"), "image_count": source.get("image_count"),
                 "capabilities": {"open_review": bool(plot_ok or legacy_ok), "plot_only": bool(plot_ok or legacy_ok), "can_edit": bool(source_ok and (plot_ok or legacy_ok)), "cleanup_actions": actions,
                                  "reasons": errors + ([] if source_ok else ["The exact annotation source needs reconnection."])}}, refs)

    def _transients(self) -> None:
        if self.transient_root is None:
            return
        for path in self._children(self.transient_root):
            if path.suffix != ".json" or path.is_symlink():
                continue
            try:
                payload = read_metadata(path, max_bytes=128 * 1024 * 1024)
                session_id = str(payload.get("session_id") or "")
                if not session_id:
                    raise ValueError("Missing session identity")
            except (OSError, ValueError, StorageLibraryError):
                self._unclassified(path, "Unrecognized saved annotation state")
                continue
            refs = [(self._asset(path, "human_work", ownership="protected"), "human_work")]
            job_id = str(payload.get("class_analysis_job_id") or "")
            source_path = absolute(payload.get("dataset_root") or path.parent)
            source_ok = source_path.is_dir() and not source_path.is_symlink()
            source_paths = [source_path]
            annotation_source = str(payload.get("class_analysis_annotation_source_root") or "")
            if annotation_source:
                source_paths.append(absolute(annotation_source))
            for dependency in source_paths:
                matching = [asset for asset in self.library.all_assets()
                            if asset["state"] != "deleted" and (contained(self.library.asset_path(asset), dependency) or contained(dependency, self.library.asset_path(asset)))
                            and asset["category"] not in {"rebuildable_cache", "export", "unknown"}]
                if matching:
                    refs.extend((asset["id"], "source") for asset in matching)
                else:
                    try:
                        self.library.locate(dependency)
                        ownership = "protected"
                    except StorageLibraryError:
                        self.library.roots["external:" + digest(str(dependency))[:24]] = dependency
                        ownership = "external"
                    refs.append((self._asset(dependency, "annotation_source_dependency", ownership=ownership), "source"))
            self._item(f"transient:{session_id}", "dataset", {"kind": "transient", "key": session_id, "path": str(path), "job_id": job_id},
                {"label": payload.get("label") or "Saved annotation workspace", "state": "available" if source_ok else "source_missing",
                 "source": {"mode": "transient", "id": session_id, "session_id": session_id, "dataset_root": str(source_path), "job_id": job_id, "annotation_revision": payload.get("_state_revision", 0)},
                 "storage_mode": "transient", "capabilities": {"open_dataset": source_ok, "can_edit": source_ok, "cleanup_actions": ["delete_review"], "reasons": [] if source_ok else ["Source images are unavailable; saved edits are retained."]}}, refs)
            if job_id:
                with self.library.transaction() as connection:
                    if connection.execute("SELECT 1 FROM items WHERE id=? AND state!='deleted'", (f"session:{job_id}",)).fetchone():
                        connection.execute("INSERT OR IGNORE INTO asset_references VALUES (?,?,?)", (f"session:{job_id}", refs[0][0], "human_work"))

    def _generated(self) -> None:
        cache_root = self.library.roots.get("cache", self.library.root / "cache")
        self.library.roots.setdefault("cache", cache_root)
        for category_dir in self._children(cache_root):
            category = category_dir.name
            if category not in _CACHE_CATEGORIES:
                self._unclassified(category_dir, "Protected cache: " + category)
                continue
            # Immutable cache shards are safe units. They may contain many
            # files; accounting is deferred to the background size scan.
            for unit in self._children(category_dir):
                asset = self._asset(unit, "rebuildable_cache", cache_category=category)
                self._item("cache:" + digest(str(unit))[:24], "cleanup", {"kind": "cache", "path": str(unit), "category": category},
                           {"label": category.replace("_", " ") + " · " + unit.name, "state": "available", "capabilities": {"cleanup_actions": ["clear_cache"], "reasons": []}}, [(asset, "cache")])
        if self.export_root:
            for identity_dir in self._children(self.export_root):
                if not identity_dir.is_dir() or identity_dir.is_symlink():
                    continue
                for archive in self._children(identity_dir):
                    if archive.suffix != ".zip" or archive.is_symlink():
                        continue
                    self.register_export(archive)
        for root_id, root in self.upload_roots.items():
            for child in self._children(root):
                if child.is_symlink():
                    continue
                payload = {}
                for name in ("upload_session.json", "session.json"):
                    try:
                        payload = read_metadata(child / name)
                        break
                    except (OSError, ValueError, StorageLibraryError):
                        continue
                asset = self._asset(child, "upload", ownership="managed" if payload else "protected")
                status = str(payload.get("status") or "unknown")
                self._item(f"upload:{root_id}:{child.name}", "cleanup", {"kind": "upload", "root_id": root_id, "key": child.name, "path": str(child)},
                           {"label": payload.get("dataset_id") or "Staged upload · " + child.name, "state": status,
                            "capabilities": {"cleanup_actions": ["remove_abandoned_upload"] if payload else [], "reasons": [] if payload else ["Upload ownership is unknown."]}}, [(asset, "upload")])
        protected = [(self.library.root / "audit", "Review and audit records"), (self.library.root / "exports", "Review exports")]
        if self.entity_store_path:
            protected.extend((Path(str(self.entity_store_path) + suffix), "Annotation transaction database" + suffix) for suffix in ("", "-wal", "-shm"))
        for path, label in protected:
            if not path.exists():
                continue
            asset = self._asset(path, "human_work", ownership="protected")
            self._item("protected:" + digest(str(path))[:24], "cleanup", {"kind": "protected", "path": str(path)},
                       {"label": label, "state": "protected", "capabilities": {"cleanup_actions": [], "reasons": ["Shared human work; not a disposable cache."]}}, [(asset, "protected")])

    def register_salad_profiles(self) -> None:
        """Persisted deployment references protect exact external head assets.

        Descriptor caches remain regenerable. Referenced checkpoints are durable
        human-selected model versions even when they originated in a report.
        """
        root = self.library.roots.get('salad_heads')
        if root is None or not (root / '.dataset_profiles.json').is_file():
            return
        management = read_metadata(root / '.dataset_profiles.json')
        references = management.get('head_references', {})
        if not isinstance(references, dict):
            raise StorageLibraryError('library_salad_references_invalid')
        management_asset = self._asset(root / '.dataset_profiles.json', 'human_work', ownership='protected')
        self._item('salad:management', 'cleanup', {'kind': 'salad_profiles', 'path': str(root / '.dataset_profiles.json')},
                   {'label': 'Dataset SALAD profiles', 'state': 'protected',
                    'capabilities': {'cleanup_actions': [], 'reasons': ['Saved dataset attachments and immutable model versions.']}},
                   [(management_asset, 'profile_management')])
        for head_id, reference in references.items():
            path = absolute(str(reference.get('path') or ''))
            reject_symlinks(path)
            try: self.library.locate(path)
            except StorageLibraryError:
                self.library.roots['external:' + digest(str(path.parent))[:24]] = path.parent
            root_id, _ = self.library.locate(path)
            head_asset = self._asset(path, 'head_artifact', ownership='protected',
                root_path=str(self.library.roots[root_id]), bound_state=reference.get('state'),
                bound_sha256=reference.get('head_sha256'), head_id=head_id)
            attached = [identity for identity, dataset in management.get('datasets', {}).items()
                        if head_id in dataset.get('head_ids', [])]
            profile = management.get('profiles', {}).get(head_id, {})
            self._item('salad:' + head_id, 'cleanup', {'kind': 'salad_head', 'head_id': head_id, 'path': str(path)},
                {'label': profile.get('display_label') or head_id, 'state': 'protected' if path.is_file() else 'source_missing',
                 'deployment_datasets': attached, 'head_sha256': reference.get('head_sha256'),
                 'capabilities': {'cleanup_actions': [], 'reasons': ['Immutable registered SALAD checkpoint; preserve this file.']}},
                [(head_asset, 'immutable_head'), (management_asset, 'profile_management')]
                + [(management_asset, 'deployment:' + identity) for identity in attached])

    def register_pruning(self, directory: Path | str) -> dict[str, Any]:
        """Register a saved pruning selection and its source dependency."""
        directory = absolute(directory)
        if not contained(directory,self.library.root / "pruning") or not re.fullmatch(r"dp_[a-f0-9]{16}",directory.name):
            raise StorageLibraryError("library_pruning_path_invalid")
        reject_symlinks(directory)
        source = read_metadata(directory / "source.json") if (directory / "source.json").is_file() else {}
        state = read_metadata(directory / "state.json")
        refs = []
        for path in self._children(directory):
            if path.name in {"exports","features"} or path.is_symlink(): continue
            human = path.name == 'review.sqlite3'
            refs.append((self._asset(path, 'human_work' if human else "session_identity" if path.suffix==".json" else "analysis_files",
                                     ownership='protected' if human else 'managed'),"pruning_review"))
        dependency_path = directory / 'dependencies.json'
        if dependency_path.is_file():
            dependencies = read_metadata(dependency_path).get('dependencies', [])
            if not isinstance(dependencies, list) or len(dependencies) > 128:
                raise StorageLibraryError('library_pruning_dependencies_invalid')
            for dependency in dependencies:
                if not dependency.get('required'): continue
                path = absolute(str(dependency.get('path') or ''))
                if not path.is_file() or path.is_symlink():
                    raise StorageLibraryError('library_pruning_dependency_unavailable')
                reject_symlinks(path)
                try: self.library.locate(path)
                except StorageLibraryError:
                    key = 'external:' + digest(str(path.parent))[:24]
                    self.library.roots[key] = path.parent
                refs.append((self._asset(path, 'head_artifact' if dependency.get('role') == 'salad_head' else 'component_artifact',
                                         ownership='protected', dependency_role=dependency.get('role'),
                                         root_path=str(self.library.roots[self.library.locate(path)[0]]),
                                         bound_state=dependency.get('state'), bound_sha256=dependency.get('sha256')), 'recipe_dependency'))
        registry=source.get("registry_path")
        if source.get("kind")=="dataset" and registry:
            # Retain linked registration and annotation overlay, not only pixels.
            with contextlib.closing(self.library.connect()) as db:
                candidates=db.execute("SELECT id,authority FROM items WHERE kind='dataset'").fetchall()
            for item_id,raw in candidates:
                authority=json.loads(raw)
                if authority.get("path") and absolute(authority["path"])==absolute(registry):
                    refs.extend((asset["id"],"source") for asset in self.library.assets_for_items([item_id]))
        source_root = source.get("dataset_root")
        if source_root and Path(source_root).is_dir():
            path = absolute(source_root)
            try:
                root_id,relative = self.library.locate(path)
            except StorageLibraryError:
                root_id = "external:" + digest(str(path))[:24]
                self.library.roots[root_id] = path
                relative = "."
            with contextlib.closing(self.library.connect()) as db:
                existing = db.execute("SELECT id FROM assets WHERE root_id=? AND relpath=?",(root_id,relative)).fetchone()
            asset = existing[0] if existing else self._asset(path,"snapshot_images",ownership="external" if source.get("storage_mode")=="linked" else "managed")
            refs.append((asset,"source"))
        return self._item("pruning:"+directory.name,"cleanup",{"kind":"pruning","path":str(directory),"key":directory.name},
             {"label":"Pruning review · "+str(source.get("dataset_label") or directory.name),"state":"available","job_id":directory.name,
              "created_at":state.get("created_at"),"source":source,
              "capabilities":{"cleanup_actions":["delete_review"],"reasons":[]}},refs)

    def register_pruning_export(self, archive: Path | str) -> dict[str, Any]:
        archive = absolute(archive)
        if not contained(archive,self.library.root/"pruning") or archive.name!="dataset.zip":
            raise StorageLibraryError("library_pruning_export_path_invalid")
        reject_symlinks(archive)
        asset = self._asset(archive,"export")
        return self._item("export:"+digest(str(archive))[:24],"cleanup",{"kind":"export","path":str(archive)},
             {"label":"Pruned dataset export","state":"available","modified_at":archive.stat().st_mtime,
              "capabilities":{"cleanup_actions":["clear_export"],"reasons":[]}},[(asset,"export")])

    def register_export(self, archive: Path | str) -> dict[str, Any]:
        """Known producer hook: two file stats, no directory traversal."""
        archive = absolute(archive)
        if self.export_root is None or not contained(archive, self.export_root) or archive.suffix != ".zip":
            raise StorageLibraryError("library_export_path_invalid")
        refs = []
        for path in (archive, archive.with_suffix(".json")):
            if not path.exists():
                continue
            reject_symlinks(path)
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise StorageLibraryError("library_export_path_invalid")
            asset_id = self._asset(path, "export")
            with self.library.transaction() as connection:
                connection.execute("UPDATE assets SET logical_bytes=?,allocated_bytes=?,shared_bytes=?,file_count=1,measured_at=?,scan_status='complete' WHERE id=? AND state='present'",
                                   (info.st_size, int(getattr(info, "st_blocks", (info.st_size + 511) // 512)) * 512, info.st_size if info.st_nlink > 1 else 0, time.time(), asset_id))
            refs.append((asset_id, "export"))
        return self._item("export:" + digest(str(archive))[:24], "cleanup", {"kind": "export", "path": str(archive)},
                          {"label": "Annotation export · " + archive.stem[:12], "state": "available", "modified_at": archive.stat().st_mtime,
                           "capabilities": {"cleanup_actions": ["clear_export"], "reasons": []}}, refs)

    def register_thumbnail_cache(self, thumb_dir: Path | str, parent_id: str = "") -> dict[str, Any]:
        """Reconcile one known cache, without rediscovering other datasets."""
        path = absolute(thumb_dir)
        if not contained(path, self.library.root) or path.name != "thumbnails":
            raise StorageLibraryError("library_thumbnail_path_invalid")
        reject_symlinks(path)
        parent_id = parent_id or "session:" + path.parent.name
        asset_id = self._asset(path, "rebuildable_cache", cache_category="session_thumbnails")
        item = self._item("cache:" + digest(str(path))[:24], "cleanup", {"kind": "cache", "path": str(path), "category": "session_thumbnails"},
                          {"label": "Review thumbnails · " + path.parent.name, "parent_id": parent_id, "state": "available",
                           "capabilities": {"cleanup_actions": ["clear_cache"], "reasons": []}}, [(asset_id, "cache")])
        self.measure(asset_ids={asset_id})
        return self.library.get_item(item["id"])

    def _reconcile_restored_trash(self) -> None:
        """Retire catalog rows after the authoritative registry restores them.

        This only stats previously known trash directories. An unavailable
        root, unreadable path or unfinished restore remains visible/protected.
        No source files are moved or removed here.
        """
        with contextlib.closing(self.library.connect()) as connection:
            rows = connection.execute("SELECT id,authority FROM items WHERE kind='trash' AND state='trashed'").fetchall()
        for row in rows:
            authority = json.loads(row["authority"])
            if authority.get("kind") != "trash":
                continue
            root = self.dataset_roots.get(str(authority.get("root_id") or ""))
            path = absolute(authority.get("path") or "/")
            if root is None or path.parent != root / ".trash":
                continue
            try:
                reject_symlinks(root)
                if not root.is_dir():
                    continue
                reject_symlinks(path)
                path.lstat()
                continue
            except FileNotFoundError:
                pass
            except (OSError, StorageLibraryError):
                continue
            assets = self.library.assets_for_items([row["id"]])
            with self.library.transaction() as connection:
                connection.execute("UPDATE items SET state='deleted',revision=revision+1,updated_at=? WHERE id=? AND state='trashed'", (time.time(), row["id"]))
                for asset in assets:
                    if contained(self.library.asset_path(asset), path):
                        connection.execute("UPDATE assets SET state='deleted',generation=generation+1 WHERE id=? AND state NOT IN ('deleting','quarantined','deleted')", (asset["id"],))

    def _restore_external_roots(self) -> None:
        """Recover reference-only locations even when their link is in trash.

        These paths never become managed deletion roots or measurement targets.
        Older catalog entries can recover the same hashed location from their
        retained source descriptor; new external assets store it explicitly.
        """
        with contextlib.closing(self.library.connect()) as connection:
            assets = connection.execute("SELECT root_id,metadata FROM assets WHERE ownership='external' OR (ownership='protected' AND category='head_artifact')").fetchall()
            rows = connection.execute("SELECT summary FROM items").fetchall()
        candidates = {}
        for row in rows:
            source = json.loads(row["summary"]).get("source") or {}
            for value in (source.get("dataset_root"), source.get("annotation_source_root")):
                if value:
                    path = absolute(value)
                    candidates["external:" + digest(str(path))[:24]] = path
        for asset in assets:
            root_id = asset["root_id"]
            if root_id in self.library.roots or not root_id.startswith("external:"):
                continue
            value = json.loads(asset["metadata"]).get("root_path")
            path = absolute(value) if value else candidates.get(root_id)
            if path is not None and root_id == "external:" + digest(str(path))[:24]:
                self.library.roots[root_id] = path

    def discover(self, *, measure: bool = True) -> dict[str, Any]:
        self._generation, self._errors = uuid.uuid4().hex, []
        # Producers may publish while shared discovery is running. Clear only
        # before reading, so an invalidation during the scan survives it.
        self.library._dirty.clear()
        self.library._metadata_bootstrapped = False
        started = time.time()
        previous = self.library.get_meta("discovery", {})
        self.library.set_meta("discovery", {"status": "discovering", "started_at": started, "generation": self._generation, "previous_complete": bool(previous.get("metadata_complete"))})
        with self.library.admission():
            self._restore_external_roots()
            self.register_salad_profiles()
            dataset_ids = self._datasets()
            self._sessions(dataset_ids)
            self._transients()
            self._generated()
            pruning_root = self.library.root / "pruning"
            for directory in self._children(pruning_root):
                if not re.fullmatch(r"dp_[a-f0-9]{16}",directory.name) or not (directory/"state.json").is_file(): continue
                self.register_pruning(directory)
                for export in self._children(directory/"exports"):
                    if (export/"dataset.zip").is_file(): self.register_pruning_export(export/"dataset.zip")
            if (pruning_root/"cache").is_dir():
                path=pruning_root/"cache"
                asset=self._asset(path,"rebuildable_cache",cache_category="pruning_descriptors_and_matching")
                self._item("cache:"+digest(str(path))[:24],"cleanup",{"kind":"cache","path":str(path),"category":"pruning"},
                     {"label":"Pruning descriptors and matching cache","state":"available","capabilities":{"cleanup_actions":["clear_cache"],"reasons":[]}},[(asset,"cache")])
            self._reconcile_restored_trash()
            discovery = {"status": "measuring" if measure else "ready", "metadata_complete": not self._errors,
                         "generation": self._generation, "started_at": started, "updated_at": time.time(), "errors": self._errors}
            self.library.set_meta("discovery", discovery)
            self.library._metadata_bootstrapped = discovery["metadata_complete"]
            if measure:
                self.measure()
            discovery.update(status="ready" if not self._errors else "partial", updated_at=time.time(), errors=self._errors)
            self.library.set_meta("discovery", discovery)
        return self.library.envelope(discovery=discovery)

    def measure(self, *, asset_ids: set[str] | None = None) -> None:
        assets = self.library.all_assets()
        owned = [(asset, self.library.asset_path(asset)) for asset in assets if asset["ownership"] != "external" and asset["state"] not in {"deleting", "quarantined"}]
        paths = {str(path): asset["id"] for asset, path in owned}
        if asset_ids is not None:
            owned = [(asset, path) for asset, path in owned if asset["id"] in asset_ids]
        temporary = self.library.directory / (".measurement-" + uuid.uuid4().hex + ".sqlite3")
        connection = sqlite3.connect(str(temporary))
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("CREATE TABLE inodes(device INTEGER,inode INTEGER,size INTEGER,allocated INTEGER,links INTEGER,seen INTEGER DEFAULT 1,PRIMARY KEY(device,inode))")
        unique_logical = sum(int(asset.get("logical_bytes") or 0) for asset in assets if asset["state"] in {"deleting", "quarantined"})
        unique_allocated = sum(int(asset.get("allocated_bytes") or 0) for asset in assets if asset["state"] in {"deleting", "quarantined"})
        try:
            for asset, path in sorted(owned, key=lambda pair: len(pair[1].parts), reverse=True):
                logical = allocated = files = hardlinked = 0
                errors = []
                stack = [path]
                while stack:
                    current = stack.pop()
                    if current != path and str(current) in paths:
                        continue
                    try:
                        info = current.lstat()
                        if stat.S_ISLNK(info.st_mode):
                            errors.append("Symbolic link skipped: " + str(current))
                            continue
                        if stat.S_ISDIR(info.st_mode):
                            with os.scandir(current) as children:
                                stack.extend(Path(child.path) for child in children)
                        elif stat.S_ISREG(info.st_mode):
                            size = max(0, int(info.st_size))
                            blocks = max(0, int(getattr(info, "st_blocks", (size + 511) // 512)) * 512)
                            files += 1
                            logical += size
                            if info.st_nlink > 1:
                                hardlinked += size
                            inserted = connection.execute("INSERT OR IGNORE INTO inodes(device,inode,size,allocated,links) VALUES (?,?,?,?,?)", (info.st_dev, info.st_ino, size, blocks, info.st_nlink)).rowcount
                            if inserted:
                                allocated += blocks
                                unique_logical += size
                                unique_allocated += blocks
                            else:
                                connection.execute("UPDATE inodes SET seen=seen+1 WHERE device=? AND inode=?", (info.st_dev, info.st_ino))
                            if files % 2048 == 0:
                                connection.commit()
                                time.sleep(0)
                    except FileNotFoundError as exc:
                        consumers = self.library.asset_consumers(asset["id"]) if current == path else [{}]
                        expected_absence = all(consumer.get("kind") == "dataset" and consumer.get("state") == "trashed" for consumer in consumers)
                        # SQLite removes these optional files when its last
                        # connection closes. Their absence is zero usage, not
                        # an incomplete scan of the annotation database.
                        if current == path and self.entity_store_path and path in {
                            Path(str(self.entity_store_path) + "-wal"),
                            Path(str(self.entity_store_path) + "-shm"),
                        }:
                            expected_absence = True
                        # Native registry trash/unlink moves a known source
                        # without deleting it. Its former path contributes zero;
                        # the authoritative trash asset accounts for the bytes.
                        if not expected_absence and len(errors) < 12:
                            errors.append(str(exc))
                    except OSError as exc:
                        if len(errors) < 12:
                            errors.append(str(exc))
                with self.library.transaction() as database:
                    database.execute("UPDATE assets SET logical_bytes=?,allocated_bytes=?,shared_bytes=?,file_count=?,measured_at=?,scan_status=?,error=? WHERE id=? AND generation=?",
                                     (logical, allocated, hardlinked, files, time.time(), "partial" if errors else "complete", encode(errors) if errors else None, asset["id"], asset["generation"]))
                if errors:
                    self._errors.append({"path": str(path), "error": "; ".join(errors)})
            connection.commit()
            if asset_ids is None:
                self.library.set_meta("usage", {"logical_bytes": unique_logical, "allocated_bytes": unique_allocated, "measured_at": time.time(), "status": "partial" if self._errors else "complete"})
            else:
                self.library.set_meta("usage", {**self.library.get_meta("usage", {}), "status": "stale"})
        finally:
            connection.close()
            temporary.unlink(missing_ok=True)

    def locations(self) -> list[dict[str, Any]]:
        locations: dict[int, dict[str, Any]] = {}
        for root_id, root in self.library.roots.items():
            if root_id.startswith("external:"):
                continue
            try:
                existing = root
                while not existing.exists() and existing != existing.parent:
                    existing = existing.parent
                info = existing.stat()
                usage = shutil.disk_usage(existing)
                row = locations.setdefault(info.st_dev, {"id": str(info.st_dev), "device": info.st_dev, "label": str(existing), "total_bytes": usage.total, "free_bytes": usage.free, "roots": []})
                row["roots"].append({"id": root_id, "path": str(root)})
            except OSError:
                continue
        return list(locations.values())
