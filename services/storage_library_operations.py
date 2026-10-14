"""Connected library operations with durable intent and restart recovery."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from services.storage_library import StorageLibrary, StorageLibraryError, absolute, contained, digest, encode, reject_symlinks
from services.storage_library_inventory import StorageLibraryInventory


ACTIONS = frozenset({"clear_cache", "clear_analysis_files", "delete_review", "trash_dataset", "empty_trash", "unlink_dataset", "remove_abandoned_upload", "clear_export"})
_ACTION_CATEGORIES = {
    "clear_cache": {"rebuildable_cache"}, "clear_analysis_files": {"analysis_files"},
    "empty_trash": {"trash", "snapshot_images", "snapshot_identity"},
    "remove_abandoned_upload": {"upload"}, "clear_export": {"export"},
    "unlink_dataset": {"dataset_metadata"},
}


class StorageLibraryOperations:
    def __init__(self, library: StorageLibrary, inventory: StorageLibraryInventory, *,
                 open_adapter: Callable | None = None, action_adapter: Callable | None = None,
                 active_users_adapter: Callable | None = None, on_change_adapter: Callable | None = None):
        self.library, self.inventory = library, inventory
        self.open_adapter, self.action_adapter = open_adapter, action_adapter
        self.active_users_adapter, self.on_change_adapter = active_users_adapter, on_change_adapter
        self._worker_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._wake_requested = False

    def summary(self) -> dict[str, Any]:
        discovery = self.library.get_meta("discovery", {"status": "pending"})
        refresh_operation_id = None
        if not self.library._metadata_bootstrapped or discovery.get("status") == "pending" or self.library._dirty.is_set():
            refresh_operation_id = self.refresh({})["operation_id"]
        else:
            with contextlib.closing(self.library.connect()) as connection:
                pending = connection.execute("SELECT id FROM operations WHERE kind='refresh' AND state IN ('queued','running') ORDER BY created_at LIMIT 1").fetchone()
            if pending:
                refresh_operation_id = pending[0]
        assets = self.library.all_assets()
        categories: dict[str, int] = {}
        for asset in assets:
            categories[asset["category"]] = categories.get(asset["category"], 0) + int(asset.get("logical_bytes") or 0)
        usage = dict(self.library.get_meta("usage", {"status": "unknown"}))
        metadata = self.library.metadata_storage()
        for field in ("logical_bytes", "allocated_bytes"):
            if usage.get(field) is not None:
                usage[field] += metadata[field]
        categories["library_metadata"] = metadata["logical_bytes"]
        discovery = self.library.get_meta("discovery", discovery)
        inventory_pending = (not self.library._metadata_bootstrapped
                             or not discovery.get("metadata_complete") or self.library._dirty.is_set())
        return self.library.envelope(locations=self.inventory.locations(), usage={**usage, "categories": categories, "protected_metadata": metadata},
                                     discovery=discovery, refresh_operation_id=refresh_operation_id, inventory_pending=inventory_pending,
                                     retention={"automatic": "unused_rebuildable_cache", "saved_work": "explicit_removal_only"})

    def list_items(self, **options: Any) -> dict[str, Any]:
        return self.library.list_items(**options)

    def get_item(self, identifier: str) -> dict[str, Any]:
        return self.library.envelope(item=self.library.get_item(identifier))

    def update_item(self, identifier: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.library.admission(blocking=False):
            return self.library.update_item(identifier, payload)

    def restore_item(self, identifier: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.library.admission(exclusive=True, blocking=False):
            self._require_discovery()
            item = self.library.get_item(identifier)
            authority_kind = item["authority"].get("kind")
            if item["state"] != "trashed" or authority_kind not in {"snapshot", "trash"}:
                raise StorageLibraryError("library_restore_unavailable")
            if payload.get("expected_revision") != item["revision"]:
                raise StorageLibraryError("library_revision_changed")
            if authority_kind == "trash":
                if self.action_adapter is None:
                    raise StorageLibraryError("library_dataset_adapter_unavailable")
                result = self.action_adapter("restore_dataset", item, dict(payload)) or {}
                self.inventory._reconcile_restored_trash()
                self.library.invalidate()
                return self.library.envelope(item=self.library.get_item(identifier), state="restored", result=result)
            if not all(self.library.asset_path(asset).exists() for asset in item["assets"]):
                raise StorageLibraryError("library_source_missing")
            with self.library.transaction() as connection:
                row = connection.execute("SELECT summary FROM items WHERE id=?", (identifier,)).fetchone()
                summary = json.loads(row[0])
                summary["capabilities"] = summary.pop("pre_trash_capabilities", {"open_dataset": True, "can_edit": True, "cleanup_actions": ["trash_dataset"], "reasons": []})
                summary["state"] = "available"
                connection.execute("UPDATE items SET kind='dataset',state='available',summary=?,revision=revision+1,updated_at=? WHERE id=?", (encode(summary), time.time(), identifier))
            return self.library.envelope(item=self.library.get_item(identifier), state="restored")

    def refresh(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with contextlib.closing(self.library.connect()) as connection:
            existing = connection.execute("SELECT id FROM operations WHERE kind='refresh' AND state IN ('queued','running') ORDER BY created_at LIMIT 1").fetchone()
        if existing:
            operation = self.library.get_operation(existing[0])
        else:
            operation = self.library.create_operation("refresh", {})
        self._wake()
        return operation

    def resume_pending_operations(self) -> None:
        """Called by composition startup; does not synchronously scan sources."""
        with self.library.transaction() as connection:
            connection.execute("UPDATE operations SET state='queued' WHERE kind='cleanup' AND state='partial_failed'")
        self._wake()

    def _require_discovery(self) -> None:
        discovery = self.library.get_meta("discovery", {})
        if not self.library._metadata_bootstrapped or not discovery.get("metadata_complete") or self.library._dirty.is_set():
            self.refresh({})
            raise StorageLibraryError("library_inventory_not_ready", "Storage discovery must finish before files can be removed.", discovery=discovery)

    def _users(self, item_ids: list[str]) -> list[dict[str, Any]]:
        if self.active_users_adapter is None:
            return []
        # Legacy editors may hold a source annotation lease without a new
        # library attachment. Include the exact registered/transient sources
        # shared by a selected review (or its cache), not merely its job id.
        related = set(item_ids)
        for identifier in tuple(related):
            parent = self.library.get_item(identifier).get("parent_id")
            if parent:
                related.add(str(parent))
        for asset in self.library.assets_for_items(related):
            related.update(consumer["id"] for consumer in self.library.asset_consumers(asset["id"])
                           if consumer.get("kind") == "dataset")
        return [dict(row) for row in (self.active_users_adapter(item_ids=sorted(related)) or [])]

    def _validate_item(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        identifier = str(payload.get("item_id") or "")
        item = self.library.get_item(identifier)
        if item["state"] in {"deleted", "deleting", "unlinked", "trashed"}:
            raise StorageLibraryError("library_item_unavailable", item=item)
        if payload.get("expected_revision") != item["revision"]:
            raise StorageLibraryError("library_revision_changed", item=item)
        return item

    def open(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        mode = {"open_review": "review", "open_dataset": "dataset"}.get(str(payload.get("mode") or ""), str(payload.get("mode") or "review"))
        if mode not in {"review", "dataset", "plot_only"}:
            raise StorageLibraryError("library_open_mode_invalid", status_code=400)
        key = str(payload.get("idempotency_key") or "").strip()
        if not key or len(key) > 240:
            raise StorageLibraryError("library_idempotency_key_required", status_code=400)
        self._require_discovery()
        operation = self.library.create_operation("open", payload, state="preparing", idempotency_key="open:" + key)
        operation_id = operation["operation_id"]
        if operation["state"] in {"prepared", "completed"}:
            return self.library.envelope(operation_id=operation_id, state=operation["state"], **operation["result"])
        if operation["state"] in {"failed", "cancelled"}:
            return self.library.get_operation(operation_id)
        if not operation.get("_created"):
            raise StorageLibraryError("library_open_in_progress", "This open request is already being prepared.", operation_id=operation_id)
        lease = None
        try:
            with self.library.admission(blocking=False):
                item = self._validate_item(payload)
                capability = "open_dataset" if mode == "dataset" else "plot_only" if mode == "plot_only" else "open_review"
                if not bool(item.get("capabilities", {}).get(capability)):
                    raise StorageLibraryError("library_open_unavailable", "This item cannot be opened in the requested mode.", item=item)
                lease = self.library.create_lease(item_ids=[item["id"]], owner=str(payload.get("client_id") or ""), purpose="prepare_open")
                # Persist the lease before any source materialization so an
                # interrupted prepare can release exactly its own reservation.
                self.library.update_operation(operation_id, result={**lease, "item_id": item["id"]})
                if self.open_adapter is None:
                    descriptor = {"source": item.get("source", {}), "job_id": item.get("job_id"), "mode": mode}
                else:
                    descriptor = self.open_adapter(item, mode, payload)
                result = {**lease, "item": {key: value for key, value in item.items() if key != "assets"}, "descriptor": descriptor}
                self.library.update_operation(operation_id, state="prepared", result=result)
            return self.library.envelope(operation_id=operation_id, state="prepared", **result)
        except Exception as exc:
            if lease:
                self.library.release(lease["lease_id"], lease["lease_token"])
            self.library.update_operation(operation_id, state="failed", error=self._error(exc))
            raise

    def commit_open(self, identifier: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        operation = self.library.get_operation(identifier, internal=True)
        if operation["type"] != "open" or operation["state"] not in {"prepared", "completed"}:
            raise StorageLibraryError("library_open_not_prepared")
        result = operation["result"]
        token = str(payload.get("lease_token") or "")
        if not token or token != result.get("lease_token"):
            raise StorageLibraryError("library_lease_invalid", status_code=403)
        if operation["state"] == "completed":
            return self.library.envelope(operation_id=identifier, state="completed", **result)
        with self.library.admission(blocking=False):
            self.library.heartbeat(result["lease_id"], token)
            item_id = result.get("item", {}).get("id") or result.get("item_id")
            item = self.library.get_item(item_id)
            if item["state"] in {"deleted", "deleting", "unlinked"}:
                raise StorageLibraryError("library_item_removed")
            with self.library.transaction() as connection:
                connection.execute("UPDATE items SET last_opened_at=? WHERE id=?", (time.time(), item_id))
                connection.execute("UPDATE leases SET purpose='attachment' WHERE id=?", (result["lease_id"],))
            self.library.update_operation(identifier, state="completed")
        return self.library.envelope(operation_id=identifier, state="completed", **result)

    def cancel_open(self, identifier: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        operation = self.library.get_operation(identifier, internal=True)
        if operation["type"] != "open":
            raise StorageLibraryError("library_operation_kind_invalid")
        result = operation["result"]
        token = str(payload.get("lease_token") or "")
        if result.get("lease_token") and token != result["lease_token"]:
            raise StorageLibraryError("library_lease_invalid", status_code=403)
        if result.get("lease_id"):
            self.library.release(result["lease_id"], token)
        return self.library.update_operation(identifier, state="cancelled")

    def heartbeat(self, identifier: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.library.heartbeat(identifier, str(payload.get("lease_token") or ""))

    def release(self, identifier: str, token: str) -> dict[str, Any]:
        return self.library.release(identifier, token)

    def _preview(self, payload: Mapping[str, Any], *, check_users: bool = True) -> dict[str, Any]:
        self._require_discovery()
        action = str(payload.get("action") or "")
        raw_ids = payload.get("item_ids")
        if action not in ACTIONS or not isinstance(raw_ids, list) or not raw_ids or len(raw_ids) > 200 or not all(isinstance(value, str) for value in raw_ids):
            raise StorageLibraryError("library_cleanup_request_invalid", status_code=400)
        identifiers = sorted(set(raw_ids))
        items = [self.library.get_item(identifier) for identifier in identifiers]
        selected, retained, blockers = [], [], []
        selected_ids = set(identifiers)
        protected_item_ids = selected_ids | {str(item.get("parent_id")) for item in items if item.get("parent_id")}
        transient_discard = all(item.get("authority", {}).get("kind") == "transient" for item in items)
        input_assets = self.library.assets_for_items(identifiers)
        keeps_human_work = action == "delete_review" and not transient_discard and any(
            asset["category"] == "human_work" and (asset["ownership"] == "protected" or any(row["id"] not in selected_ids for row in self.library.asset_consumers(asset["id"])))
            for asset in input_assets
        )
        for item in items:
            if item["state"] in {"deleted", "deleting", "unlinked"}:
                blockers.append({"code": "item_unavailable", "item_id": item["id"], "message": "This item was already removed or is being removed."})
            if item["pinned"]:
                blockers.append({"code": "item_pinned", "item_id": item["id"], "message": "Turn off Keep before removing this item."})
            parent_id = str(item.get("parent_id") or "")
            if parent_id:
                try:
                    parent = self.library.get_item(parent_id)
                except StorageLibraryError:
                    parent = {}
                if parent.get("pinned"):
                    blockers.append({"code": "parent_pinned", "item_id": item["id"], "message": "The owning review is marked Keep.", "parent_id": parent_id})
            actions = item.get("capabilities", {}).get("cleanup_actions", [])
            if action not in actions:
                blockers.append({"code": "action_unavailable", "item_id": item["id"], "message": "This cleanup action is unavailable for the selected item."})
        for asset in input_assets:
            if asset["state"] == "deleted":
                continue
            remaining = [row for row in self.library.asset_consumers(asset["id"]) if row["id"] not in selected_ids]
            reason = ""
            if remaining:
                reason = "Still used by another retained item."
            elif asset["ownership"] == "external":
                reason = "External originals are never deleted."
            elif asset["category"] == "unknown":
                reason = "Unknown ownership remains protected."
            elif action in _ACTION_CATEGORIES and asset["category"] not in _ACTION_CATEGORIES[action]:
                reason = "Kept by this cleanup action."
            elif action == "delete_review" and asset["category"] in {"snapshot_images", "snapshot_identity", "annotation_input"}:
                reason = "Source or acknowledged human work is retained unless the owning store explicitly removes it."
            elif keeps_human_work and asset["category"] == "session_identity":
                reason = "Identity and recovery evidence for retained annotations."
            elif action in {"trash_dataset", "unlink_dataset"}:
                reason = "Handled by the authoritative dataset registry."
            elif asset["ownership"] == "protected" and not (transient_discard and asset["category"] == "human_work"):
                reason = "Durable human work remains protected."
            if reason:
                retained.append({**asset, "reason": reason, "consumers": [{"id": row["id"], "label": row["label"]} for row in remaining]})
            else:
                selected.append(asset)
        if transient_discard and any(asset["category"] == "human_work" and asset.get("consumers") for asset in retained):
            blockers.append({"code": "retained_annotation_dependency", "message": "A retained review still depends on these saved annotations."})
        all_assets = self.library.all_assets()
        selected_asset_ids = {asset["id"] for asset in selected}
        for asset in list(selected):
            path = self.library.asset_path(asset)
            descendants = [other for other in all_assets if other["id"] != asset["id"] and contained(self.library.asset_path(other), path)]
            protected_children = [other for other in descendants if other["id"] not in selected_asset_ids]
            if protected_children:
                selected.remove(asset)
                selected_asset_ids.remove(asset["id"])
                retained.append({**asset, "reason": "Contains retained child assets.", "consumers": []})
                if action in {"unlink_dataset", "empty_trash"}:
                    blockers.append({"code": "retained_child_assets", "message": "Retained annotations or shared files must be preserved before removing this item.", "asset_id": asset["id"]})
        # Trash/unlink mutates an existing registry path. It must not move a
        # directory that any retained session or annotation branch still uses.
        if action in {"trash_dataset", "unlink_dataset", "empty_trash"}:
            for item in items:
                path_value = item.get("authority", {}).get("path")
                if not path_value:
                    continue
                path = absolute(path_value)
                for asset in all_assets:
                    if not contained(self.library.asset_path(asset), path):
                        continue
                    other_users = [row for row in self.library.asset_consumers(asset["id"]) if row["id"] not in selected_ids]
                    if other_users:
                        blockers.append({"code": "retained_dependency", "item_id": item["id"], "message": "A retained review or dataset still needs these files.", "consumers": [row["id"] for row in other_users]})
                        break
        affected_paths = [self.library.asset_path(asset) for asset in selected]
        if action in {"trash_dataset", "unlink_dataset", "delete_review"}:
            affected_paths.extend(absolute(item["authority"]["path"]) for item in items if item.get("authority", {}).get("path"))
        for lease in self.library.active_leases():
            if protected_item_ids.intersection(lease["item_ids"]) or any(contained(path, absolute(value)) or contained(absolute(value), path) for path in affected_paths for value in lease["paths"]):
                blockers.append({"code": "item_in_use", "message": "A browser or operation is using the selected files.", "lease_id": lease["id"], "purpose": lease["purpose"]})
        if check_users:
            for user in self._users(identifiers):
                blockers.append({"code": "active_operation", "message": "An active operation is using managed storage.", "user": user})
        # A preview is an identity-bound plan, not a sum of displayed rows.
        plan_digest = digest({"action": action, "items": [[item["id"], item["revision"], item["source_revision"]] for item in items],
                              "assets": [[asset["id"], asset["generation"], asset["signature"]] for asset in selected],
                              "retained": [[asset["id"], asset["generation"], asset["consumer_count"]] for asset in retained]})
        logical = sum(int(asset.get("logical_bytes") or 0) for asset in selected)
        # Physical estimates exclude multiply-linked bytes unless all aliases
        # can be proved selected. The UI also reports observed volume free space.
        allocated = sum(int(asset.get("allocated_bytes") or 0) for asset in selected if not int(asset.get("shared_bytes") or 0))
        effects = ["Shared sources and retained items stay available."]
        if action == "trash_dataset":
            effects = ["Moves this dataset out of the active library. Trash still occupies disk space."]
        elif action == "clear_analysis_files":
            effects.append("Saved annotations and review identity remain; removed analysis files may require recomputation.")
        elif action == "delete_review":
            effects.append("Removes this saved review. Shared review records and retained annotations are not erased.")
            if transient_discard:
                effects = ["Permanently discards this saved annotation workspace and its local edits. Original images are not deleted."]
        elif action == "unlink_dataset":
            effects = ["Unlinks the app's dataset record. Saved edits are kept in Deleted datasets until you permanently empty them. External original files remain untouched."]
        return {"action": action, "item_ids": identifiers, "items": [{key: value for key, value in item.items() if key != "assets"} for item in items],
                "assets": selected, "retained_assets": retained, "blockers": blockers, "digest": plan_digest,
                "estimated_bytes": logical if action not in {"trash_dataset", "unlink_dataset"} else 0,
                "estimated_allocated_bytes": allocated if action not in {"trash_dataset", "unlink_dataset"} else 0,
                "estimate_status": "complete" if all(asset.get("scan_status") == "complete" for asset in selected) else "partial",
                "effects": effects, "can_commit": not blockers}

    def preview_cleanup(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.library.admission(blocking=False):
            preview = self._preview(payload)
            operation = self.library.create_operation("preview", {"action": preview["action"], "item_ids": preview["item_ids"]}, state="prepared")
            self.library.update_operation(operation["operation_id"], result=preview)
        return self.library.envelope(preview_id=operation["operation_id"], **preview)

    def commit_cleanup(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        preview_id = str(payload.get("preview_id") or "")
        previous = self.library.get_operation(preview_id, internal=True)
        if previous["type"] != "preview":
            raise StorageLibraryError("library_preview_invalid")
        requested_digest = str(payload.get("digest") or "")
        if requested_digest != previous["result"].get("digest"):
            raise StorageLibraryError("library_preview_changed")
        key = str(payload.get("idempotency_key") or "").strip()
        if not key or len(key) > 240:
            raise StorageLibraryError("library_idempotency_key_required", status_code=400)
        request = {"preview_id": preview_id, "digest": requested_digest, "action": previous["request"]["action"], "item_ids": previous["request"]["item_ids"]}
        # Resolve an existing acknowledgement before rechecking a now-deleted
        # target, so retrying a lost response remains idempotent.
        with contextlib.closing(self.library.connect()) as connection:
            existing = connection.execute("SELECT id,request_hash FROM operations WHERE idempotency_key=?", ("cleanup:" + key,)).fetchone()
        if existing:
            if existing["request_hash"] != digest(["cleanup", request]):
                raise StorageLibraryError("library_idempotency_conflict")
            return self.library.get_operation(existing["id"])
        with self.library.admission(exclusive=True, blocking=False):
            current = self._preview(previous["request"])
            if current["digest"] != requested_digest:
                raise StorageLibraryError("library_preview_changed", "The selected files or their dependencies changed. Review a new preview.")
            if current["blockers"]:
                raise StorageLibraryError("library_cleanup_blocked", "The selected files are protected or in use.", blockers=current["blockers"])
            operation = self.library.create_operation("cleanup", request, idempotency_key="cleanup:" + key,
                                                      operation_items=current["assets"])
        self._wake()
        return self.library.get_operation(operation["operation_id"])

    def get_operation(self, identifier: str) -> dict[str, Any]:
        return self.library.get_operation(identifier)

    def retry_operation(self, identifier: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        operation = self.library.get_operation(identifier, internal=True)
        if operation["type"] not in {"cleanup", "refresh"} or operation["state"] not in {"partial_failed", "failed", "queued", "running"}:
            raise StorageLibraryError("library_operation_not_retryable")
        if operation["state"] in {"partial_failed", "failed"}:
            self.library.update_operation(identifier, state="queued")
        self._wake()
        return self.library.get_operation(identifier)

    def _wake(self) -> None:
        with self._worker_lock:
            self._wake_requested = True
            if self._worker and self._worker.is_alive():
                return
            self._wake_requested = False
            self._worker = threading.Thread(target=self._work, daemon=True, name="storage-library")
            self._worker.start()

    @staticmethod
    def _error(exc: Exception) -> dict[str, Any]:
        return dict(exc.detail) if isinstance(exc, StorageLibraryError) else {"code": "library_operation_failed", "message": str(exc)}

    def _work(self) -> None:
        import fcntl
        path = self.library.directory / "worker.lock"
        reject_symlinks(path)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            # Wait in this background thread if another process is finishing
            # its queue. Returning here could lose a request queued after that
            # worker's final empty read; it cannot see our in-process wake flag.
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            while True:
                try:
                    with self.library.transaction() as connection:
                        row = connection.execute("SELECT id,kind FROM operations WHERE kind IN ('refresh','cleanup') AND state IN ('queued','running') ORDER BY created_at LIMIT 1").fetchone()
                        if row is None:
                            return
                        connection.execute("UPDATE operations SET state='running',updated_at=? WHERE id=?", (time.time(), row["id"]))
                except StorageLibraryError as exc:
                    if exc.detail.get("code") == "library_metadata_retry":
                        continue
                    raise
                try:
                    if row["kind"] == "refresh":
                        result = self.inventory.discover(measure=True)
                        self.library.update_operation(row["id"], state="completed", result=result)
                    else:
                        if not self.library._metadata_bootstrapped or self.library._dirty.is_set():
                            self.inventory.discover(measure=False)
                        self._execute_cleanup(row["id"])
                except Exception as exc:
                    if isinstance(exc, StorageLibraryError) and exc.detail.get("code") in {"library_storage_busy", "library_metadata_retry"}:
                        self.library.update_operation(row["id"], state="queued", progress={"waiting_for_storage": True, "reason": exc.detail["code"]})
                        time.sleep(0.2)
                        continue
                    with contextlib.closing(self.library.connect()) as connection:
                        changed = connection.execute("SELECT 1 FROM operation_items WHERE operation_id=? AND phase!='planned' LIMIT 1", (row["id"],)).fetchone()
                    if not changed:
                        saved_result = self.library.get_operation(row["id"], internal=True)["result"]
                        changed = bool(saved_result.get("authoritative_results") or saved_result.get("authoritative_started"))
                    self.library.update_operation(row["id"], state="partial_failed" if changed else "failed", error=self._error(exc))
        finally:
            os.close(descriptor)
            # A request may arrive after the final empty queue read while this
            # thread still appears alive. Hand it off only after releasing the
            # process lock, so the replacement worker can actually claim it.
            with self._worker_lock:
                if self._worker is threading.current_thread():
                    self._worker = None
                wake_requested = self._wake_requested
            if wake_requested:
                self._wake()

    def _execute_cleanup(self, operation_id: str) -> None:
        operation = self.library.get_operation(operation_id, internal=True)
        request = operation["request"]
        action, identifiers = request["action"], request["item_ids"]
        callback_results = operation["result"].get("authoritative_results", {})
        callback_started = set(operation["result"].get("authoritative_started", []))
        with contextlib.closing(self.library.connect()) as connection:
            steps = [dict(row) for row in connection.execute("SELECT * FROM operation_items WHERE operation_id=? ORDER BY asset_id", (operation_id,))]
        resumed = any(step["phase"] != "planned" for step in steps) or bool(callback_results) or bool(callback_started)
        approved_preview = self.library.get_operation(request["preview_id"], internal=True)["result"]
        with self.library.admission(exclusive=True, blocking=False):
            self._require_discovery()
            if not resumed:
                current = self._preview(request)
                if current["digest"] != request["digest"]:
                    raise StorageLibraryError("library_preview_changed", "Storage changed before removal began. Nothing was removed.")
                if current["blockers"]:
                    raise StorageLibraryError("library_cleanup_blocked", blockers=current["blockers"])
            else:
                users = self._users(identifiers)
                if users:
                    raise StorageLibraryError("library_cleanup_blocked", active_users=users)
            approved_assets = {asset["id"]: asset for asset in approved_preview.get("assets", [])}
            recorded_asset_ids = {step["asset_id"] for step in steps}
            if recorded_asset_ids - approved_assets.keys():
                raise StorageLibraryError("library_cleanup_plan_invalid", "The stored removal plan does not match its approved preview.")
            if approved_assets.keys() - recorded_asset_ids:
                # Older versions committed the queue row before the file list.
                # Recover that interrupted intent from its durable preview;
                # normal dependency/signature checks below still apply.
                with self.library.transaction() as connection:
                    connection.executemany("INSERT INTO operation_items(operation_id,asset_id,descriptor) VALUES (?,?,?)",
                                           [(operation_id, asset_id, encode(approved_assets[asset_id]))
                                            for asset_id in sorted(approved_assets.keys() - recorded_asset_ids)])
                    steps = [dict(row) for row in connection.execute("SELECT * FROM operation_items WHERE operation_id=? ORDER BY asset_id", (operation_id,))]
            items = [self.library.get_item(identifier) for identifier in identifiers]
            current_assets = self.library.all_assets()
            selected_asset_ids = {step["asset_id"] for step in steps}

            def retained_path_consumers(path):
                return sorted({consumer["id"] for candidate in current_assets
                               if contained(self.library.asset_path(candidate), path)
                               for consumer in self.library.asset_consumers(candidate["id"])
                               if consumer["id"] not in identifiers})

            if resumed:
                protected_ids = set(identifiers) | {str(item["parent_id"]) for item in items if item.get("parent_id")}
                protected_paths = [self.library.asset_path(json.loads(step["descriptor"])) for step in steps]
                protected_paths.extend(absolute(item["authority"]["path"]) for item in items if item.get("authority", {}).get("path"))
                for identifier in protected_ids:
                    if self.library.get_item(identifier).get("pinned"):
                        raise StorageLibraryError("library_cleanup_blocked", "A selected item or its owner is marked Keep.")
                for lease in self.library.active_leases():
                    if protected_ids.intersection(lease["item_ids"]) or any(
                        contained(path, absolute(value)) or contained(absolute(value), path)
                        for path in protected_paths for value in lease["paths"]
                    ):
                        raise StorageLibraryError("library_cleanup_blocked", "An active browser or operation is using the selected files.", lease_id=lease["id"])
            if action == "delete_review":
                for item in items:
                    if item.get("kind") != "session":
                        continue
                    retained = [asset for asset in approved_preview.get("retained_assets", []) if asset["id"] in {row["id"] for row in item["assets"]}]
                    if not any(asset["category"] == "human_work" for asset in retained):
                        continue
                    visible_workspace = any(
                        consumer.get("kind") == "dataset" and consumer["id"] not in identifiers
                        for asset in retained if asset["category"] == "human_work"
                        for consumer in self.library.asset_consumers(asset["id"])
                    )
                    if visible_workspace:
                        continue
                    work_source = dict(item.get("source") or {})
                    work_source.setdefault("job_id", item.get("job_id") or item["authority"].get("key"))
                    self.library.register_item("saved_work:" + str(item["authority"].get("key")), kind="dataset",
                        authority={"kind": "saved_work", "key": item["authority"].get("key"), "path": item["authority"].get("path")},
                        summary={"label": "Saved annotations — " + item["label"], "state": "saved_work", "source": work_source,
                                 "capabilities": {"open_dataset": bool(item.get("capabilities", {}).get("can_edit")), "can_edit": bool(item.get("capabilities", {}).get("can_edit")),
                                                  "cleanup_actions": [], "reasons": ["Analysis removed; saved annotation work and its exact source identity are retained."]}},
                        references=[(asset["id"], "retained_work") for asset in retained])
            if action in {"trash_dataset", "unlink_dataset", "delete_review", "clear_analysis_files", "remove_abandoned_upload"}:
                for item in items:
                    if item["id"] in callback_results:
                        continue
                    if action in {"trash_dataset", "unlink_dataset"}:
                        users = retained_path_consumers(absolute(item["authority"]["path"]))
                        if users:
                            raise StorageLibraryError("library_dependency_changed", "A retained item now needs this dataset.", item_id=item["id"], consumers=users)
                    callback_started.add(item["id"])
                    self.library.update_operation(operation_id, result={"authoritative_started": sorted(callback_started), "authoritative_results": callback_results})
                    if action == "trash_dataset" and item["authority"].get("kind") == "snapshot":
                        callback_results[item["id"]] = {"status": "trashed", "logical_trash": True}
                    elif self.action_adapter is not None:
                        callback_results[item["id"]] = self.action_adapter(action, item, {**request, "operation_id": operation_id}) or {}
                    elif action in {"trash_dataset", "unlink_dataset"}:
                        raise StorageLibraryError("library_dataset_adapter_unavailable")
                    else:
                        callback_results[item["id"]] = {"status": "retained_human_work"}
                    self.library.update_operation(operation_id, result={"authoritative_started": sorted(callback_started), "authoritative_results": callback_results})
                    for preserved in callback_results[item["id"]].get("retained_assets", []):
                        path = absolute(preserved)
                        if not contained(path, self.library.root):
                            raise StorageLibraryError("library_retained_asset_outside_root")
                        asset_id = self.library.register_asset(path, category="human_work", ownership="protected")
                        with self.library.transaction() as connection:
                            connection.execute("INSERT OR IGNORE INTO asset_references VALUES (?,?,?)", (item["id"], asset_id, "retained_work"))
            with self.library.transaction() as connection:
                for identifier in identifiers:
                    connection.execute("UPDATE items SET state='deleting' WHERE id=?", (identifier,))
            for step in steps:
                if step["phase"] in {"quarantined", "deleted"}:
                    continue
                asset = json.loads(step["descriptor"])
                path = self.library.asset_path(asset)
                reject_symlinks(path)
                root = self.library.roots[asset["root_id"]]
                if path == root or contained(self.library.directory, path) or contained(path, self.library.directory):
                    raise StorageLibraryError("library_delete_root_forbidden")
                if step["phase"] == "moving" and step.get("quarantine"):
                    saved_quarantine = absolute(step["quarantine"])
                    if saved_quarantine.exists() and not path.exists():
                        with self.library.transaction() as connection:
                            connection.execute("UPDATE operation_items SET phase='quarantined' WHERE operation_id=? AND asset_id=?", (operation_id, asset["id"]))
                            connection.execute("UPDATE assets SET state='quarantined' WHERE id=?", (asset["id"],))
                        continue
                if not path.exists():
                    with self.library.transaction() as connection:
                        connection.execute("UPDATE operation_items SET phase='deleted' WHERE operation_id=? AND asset_id=?", (operation_id, asset["id"]))
                        connection.execute("UPDATE assets SET state='deleted',generation=generation+1 WHERE id=?", (asset["id"],))
                    continue
                retained_consumers = [consumer["id"] for consumer in self.library.asset_consumers(asset["id"])
                                      if consumer["id"] not in identifiers]
                descendants = [candidate for candidate in current_assets if candidate["id"] != asset["id"]
                               and contained(self.library.asset_path(candidate), path)]
                retained_children = [candidate["id"] for candidate in descendants if candidate["id"] not in selected_asset_ids]
                if retained_children:
                    raise StorageLibraryError("library_dependency_changed", "New retained assets are nested inside the selected files.", asset_id=asset["id"], retained_assets=retained_children)
                if retained_consumers:
                    raise StorageLibraryError("library_dependency_changed", "A retained item now needs files selected by this operation.", asset_id=asset["id"], consumers=retained_consumers)
                info = path.lstat()
                signature = encode([info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode])
                if signature != asset["signature"]:
                    raise StorageLibraryError("library_asset_changed", "A selected asset changed since its preview.", asset_id=asset["id"])
                # The destination is a sibling on the same filesystem. Intent
                # is committed before rename, covering a crash at either side.
                quarantine = path.parent / (".tator-delete-" + operation_id + "-" + digest(asset["id"])[:12])
                reject_symlinks(quarantine)
                with self.library.transaction() as connection:
                    connection.execute("UPDATE operation_items SET quarantine=?,phase='moving' WHERE operation_id=? AND asset_id=?", (str(quarantine), operation_id, asset["id"]))
                    connection.execute("UPDATE assets SET state='deleting' WHERE id=?", (asset["id"],))
                if quarantine.exists():
                    raise StorageLibraryError("library_quarantine_exists")
                os.replace(path, quarantine)
                with self.library.transaction() as connection:
                    connection.execute("UPDATE operation_items SET phase='quarantined' WHERE operation_id=? AND asset_id=?", (operation_id, asset["id"]))
                    connection.execute("UPDATE assets SET state='quarantined' WHERE id=?", (asset["id"],))
        # Slow deletion happens after quarantine, outside the global admission
        # lock. Existing consumers see a durable removed/deleting item state.
        removed_bytes = 0
        with contextlib.closing(self.library.connect()) as connection:
            steps = [dict(row) for row in connection.execute("SELECT * FROM operation_items WHERE operation_id=?", (operation_id,))]
        for index, step in enumerate(steps):
            asset = json.loads(step["descriptor"])
            quarantine_value = step.get("quarantine")
            if quarantine_value:
                quarantine = absolute(quarantine_value)
                path = self.library.asset_path(asset)
                expected_name = ".tator-delete-" + operation_id + "-" + digest(asset["id"])[:12]
                if quarantine.parent != path.parent or quarantine.name != expected_name:
                    raise StorageLibraryError("library_quarantine_invalid")
                reject_symlinks(quarantine)
                if quarantine.exists():
                    if quarantine.is_dir():
                        shutil.rmtree(quarantine)
                    else:
                        quarantine.unlink()
            removed_bytes += int(asset.get("logical_bytes") or 0)
            with self.library.transaction() as connection:
                connection.execute("UPDATE operation_items SET phase='deleted',error=NULL WHERE operation_id=? AND asset_id=?", (operation_id, asset["id"]))
                connection.execute("UPDATE assets SET state='deleted',generation=generation+1 WHERE id=?", (asset["id"],))
            self.library.update_operation(operation_id, progress={"completed_assets": index + 1, "total_assets": len(steps), "removed_logical_bytes": removed_bytes})
        # Files are already absent and item state is 'deleting'. Publishing
        # the terminal database state needs no second filesystem admission;
        # unrelated readers must not strand a completed deletion here.
        with contextlib.nullcontext():
            with self.library.transaction() as connection:
                for item in items:
                    if action == "clear_analysis_files":
                        state = "analysis_cleared"
                    elif action == "trash_dataset":
                        state = "trashed"
                    elif action == "unlink_dataset":
                        state = "unlinked"
                    else:
                        state = "deleted"
                    summary = {key: value for key, value in item.items() if key not in {"assets", "storage", "authority", "id", "revision", "source_revision"}}
                    if action == "clear_analysis_files":
                        summary["capabilities"] = {**summary.get("capabilities", {}), "open_review": False, "plot_only": False,
                                                   "open_dataset": bool(summary.get("capabilities", {}).get("can_edit")),
                                                   "cleanup_actions": ["delete_review"], "reasons": ["Analysis files were cleared; saved annotations and review identity remain available."]}
                    elif state == "trashed" and item["authority"].get("kind") == "snapshot":
                        summary["pre_trash_capabilities"] = summary.get("capabilities", {})
                        summary["capabilities"] = {"cleanup_actions": ["empty_trash"], "restore": True,
                                                   "restore_url": "/storage/library/items/" + item["id"] + "/restore", "reasons": []}
                        connection.execute("UPDATE items SET kind='trash' WHERE id=?", (item["id"],))
                    elif state in {"deleted", "unlinked"}:
                        summary["deleted_at"] = time.time()
                        summary["deleted_by_operation"] = operation_id
                    connection.execute("UPDATE items SET state=?,summary=?,revision=revision+1,updated_at=? WHERE id=?", (state, encode(summary), time.time(), item["id"]))
            result = {"removed_logical_bytes": removed_bytes, "estimated_bytes": removed_bytes,
                      "authoritative_results": callback_results, "locations": self.inventory.locations(), "item_ids": identifiers, "action": action}
            if self.on_change_adapter is not None:
                self.on_change_adapter(action, items, result)
            self.library.update_operation(operation_id, state="completed", result=result)
        self.library.replenish_metadata_reserve()
        self.library._dirty.set()
