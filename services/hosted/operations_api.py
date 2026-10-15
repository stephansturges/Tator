"""Administrator-only export, archive and recovery download endpoints."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import threading
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .backup import BackupError, create_encrypted_backup
from .errors import DomainError
from .operations import HostedOperations
from .types import MutationContext


class Command(BaseModel):
    operation_id: str = Field(min_length=8, max_length=128)


class Export(Command):
    snapshot_id: str | None = None


class Snapshot(Command):
    name: str = Field(default="", max_length=200)


class ImageArchive(Command):
    expected_version: str
    generation: int = Field(ge=0)
    editor_id: str
    reason: str = Field(default="", max_length=10000)


class DatasetArchive(Command):
    expected_images: dict
    expected_revision: int = Field(ge=0)
    reason: str = Field(default="", max_length=10000)


class RecoveryBackup(Command):
    # This is a PUBLIC encryption recipient. Private decryption keys never enter
    # this HTTP service and are needed only by the operator's restore CLI.
    recipient: str = Field(pattern=r"^age1[0-9a-z]{58}$")


class OrphanScan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    after: str = Field(default='', max_length=64)
    limit: int = Field(default=500, ge=1, le=2000)


class OrphanSelection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    digests: list[Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]] = Field(min_length=1, max_length=2000)


class OrphanPurge(OrphanSelection, Command):
    preview_hash: str = Field(pattern=r'^[0-9a-f]{64}$')


def install_operations_api(app, store, auth, artifacts, admin, root):
    operations = HostedOperations(store, artifacts, root)
    app.state.operations = operations
    router = APIRouter(prefix="/api/hosted")
    export_dir, backup_dir = Path(root) / "downloads", Path(root) / "recovery"
    export_dir.mkdir(exist_ok=True, mode=0o700)
    backup_dir.mkdir(exist_ok=True, mode=0o700)
    backup_lock = threading.Lock()
    with store.transaction() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS hosted_backup_downloads (
            id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
            actor_id TEXT NOT NULL, created_at REAL NOT NULL)""")

    @router.post("/datasets/{dataset_id}/snapshots")
    def snapshot(dataset_id: str, body: Snapshot, identity=Depends(admin)):
        return store.create_snapshot(identity, dataset_id, body.operation_id, body.name)

    @router.get("/snapshots/{snapshot_id}")
    def get_snapshot(snapshot_id: str, identity=Depends(admin)):
        return store.get_snapshot(identity, snapshot_id)

    @router.post("/datasets/{dataset_id}/export")
    def export(dataset_id: str, body: Export, identity=Depends(admin)):
        name = hashlib.sha256((identity.user_id + "\0" + body.operation_id).encode()).hexdigest()
        destination = export_dir / (name + ".zip")
        result = operations.export_canonical(identity, dataset_id, destination, body.operation_id, body.snapshot_id)
        # The durable content-addressed artifact serves retries. This additional
        # materialized download is temporary and is not a second dataset source.
        destination.unlink(missing_ok=True)
        return {key: value for key, value in result.items() if key != "path"} | {"download_url": f"/api/hosted/exports/{result['export_id']}"}

    @router.get("/exports/{export_id}")
    def download(export_id: str, identity=Depends(admin)):
        reference = operations.export_reference(identity, export_id)
        def chunks():
            with artifacts.open(reference) as stream:
                while chunk := stream.read(1024 * 1024):
                    yield chunk
        return StreamingResponse(chunks(), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="tator-dataset.zip"'})

    @router.post("/images/{image_id}/archive")
    def archive_image(image_id: str, body: ImageArchive, identity=Depends(admin)):
        context = MutationContext(image_id, body.expected_version, body.generation, body.editor_id, body.operation_id)
        return operations.archive_image(identity, context, body.reason)

    @router.post("/datasets/{dataset_id}/archive")
    def archive_dataset(dataset_id: str, body: DatasetArchive, identity=Depends(admin)):
        return operations.archive_dataset(identity, dataset_id, **body.model_dump())

    @router.get("/archives")
    def archives(limit: int = 100, offset: int = 0, identity=Depends(admin)):
        return {"archives": operations.list_archives(identity, limit=limit, offset=offset)}

    @router.post("/archives/{archive_id}/restore")
    def restore(archive_id: str, body: Command, identity=Depends(admin)):
        return operations.restore_archive(identity, archive_id, body.operation_id)

    @router.get("/archives/{archive_id}/purge-preview")
    def purge_preview(archive_id: str, identity=Depends(admin)):
        return operations.purge_preview(identity, archive_id)

    @router.post('/storage/orphans/scan')
    def scan_orphans(body: OrphanScan, identity=Depends(admin)):
        return operations.discover_orphans(identity, **body.model_dump())

    @router.post('/storage/orphans/preview')
    def preview_orphans(body: OrphanSelection, identity=Depends(admin)):
        return operations.preview_orphan_purge(identity, body.digests)

    @router.post('/storage/orphans/purge')
    def purge_orphans(body: OrphanPurge, identity=Depends(admin)):
        return operations.purge_orphans(identity, **body.model_dump())

    @router.post("/recovery-backups")
    def backup(body: RecoveryBackup, identity=Depends(admin)):
        backup_id = hashlib.sha256((identity.user_id + "\0" + body.operation_id).encode()).hexdigest()
        payload = {"recipient": body.recipient}
        with backup_lock:
            prior = operations._existing(identity, body.operation_id, "recovery_backup", payload)
            if prior is not None:
                saved = backup_dir / (prior["backup_id"] + ".tar.age")
                if saved.is_symlink() or not saved.is_file():
                    raise DomainError("backup_download_unavailable", 409, "This earlier backup download is unavailable after recovery. Create a new backup.")
                return prior
            destination = backup_dir / (backup_id + ".tar.age")
            # A previous process may have produced a complete file before its DB
            # receipt. Preserve that file and choose a fresh staging suffix.
            # Atomic backup publication will never overwrite it silently.
            if destination.exists():
                raise DomainError("backup_receipt_missing", 409, "An earlier backup file exists without a recorded outcome. An administrator must reconcile it before retrying this command.")
            try:
                result = create_encrypted_backup(store.path, artifacts, destination,
                    recipient=body.recipient, references=operations.artifact_references,
                    metadata={"format": "tator-hosted", "instance_id": auth.instance_id})
            except BackupError as error:
                raise DomainError("backup_failed", 422, str(error)) from error
            def commit(db):
                db.execute("INSERT INTO hosted_backup_downloads VALUES (?,?,?,?,?)", (backup_id, result.sha256, result.size_bytes, identity.user_id, store.clock()))
                return {"backup_id": backup_id, "sha256": result.sha256, "size_bytes": result.size_bytes,
                        "artifact_count": result.artifact_count, "download_url": f"/api/hosted/recovery-backups/{backup_id}"}
            return store._command(identity, body.operation_id, "recovery_backup", payload, commit, admin=True)

    @router.get("/recovery-backups/{backup_id}")
    def download_backup(backup_id: str, identity=Depends(admin)):
        with store.read_connection() as db:
            store._actor(db, identity, admin=True)
            row = db.execute("SELECT * FROM hosted_backup_downloads WHERE id=?", (backup_id,)).fetchone()
            if not row:
                raise DomainError("backup_not_found", 404)
        path = backup_dir / (row["id"] + ".tar.age")
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["size_bytes"]:
            raise DomainError("backup_download_unavailable", 404)
        return FileResponse(path, filename="tator-recovery.tar.age", media_type="application/octet-stream")

    app.include_router(router)
    return operations
