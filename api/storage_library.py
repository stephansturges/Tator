"""Thin routes for the persistent dataset, session and storage library."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

from services.storage_library import StorageLibraryError


def build_storage_library_router(service: Any) -> APIRouter:
    """Accept a service or lazy factory; route registration performs no I/O."""
    router = APIRouter(prefix="/storage/library", tags=["storage library"])

    def call(method: str, *args, **kwargs):
        try:
            target = service() if callable(service) else service
            return getattr(target, method)(*args, **kwargs)
        except StorageLibraryError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @router.get("/summary")
    def summary():
        return call("summary")

    @router.get("/items")
    def list_items(kind: str = "", q: str = "", cursor: str = "", limit: int = Query(50, ge=1, le=100), parent_id: str = "", source_id: str = ""):
        return call("list_items", kind=kind, q=q, cursor=cursor, limit=limit, parent_id=parent_id, source_id=source_id)

    @router.get("/items/{item_id}")
    def get_item(item_id: str):
        return call("get_item", item_id)

    @router.patch("/items/{item_id}")
    def update_item(item_id: str, payload: dict = Body(...)):
        return call("update_item", item_id, payload)

    @router.post("/items/{item_id}/restore")
    def restore_item(item_id: str, payload: dict = Body(...)):
        return call("restore_item", item_id, payload)

    @router.post("/refresh")
    def refresh(payload: dict = Body(default_factory=dict)):
        return call("refresh", payload)

    @router.post("/open")
    def open_item(payload: dict = Body(...)):
        return call("open", payload)

    @router.post("/open/{operation_id}/commit")
    def commit_open(operation_id: str, payload: dict = Body(...)):
        return call("commit_open", operation_id, payload)

    @router.post("/open/{operation_id}/cancel")
    def cancel_open(operation_id: str, payload: dict = Body(default_factory=dict)):
        return call("cancel_open", operation_id, payload)

    @router.post("/leases/{lease_id}/heartbeat")
    def heartbeat(lease_id: str, payload: dict = Body(...)):
        return call("heartbeat", lease_id, payload)

    @router.delete("/leases/{lease_id}")
    def release(lease_id: str, lease_token: str = ""):
        return call("release", lease_id, lease_token)

    @router.post("/cleanup/preview")
    def preview(payload: dict = Body(...)):
        return call("preview_cleanup", payload)

    @router.post("/cleanup/commit")
    def commit(payload: dict = Body(...)):
        return call("commit_cleanup", payload)

    @router.get("/operations/{operation_id}")
    def operation(operation_id: str):
        return call("get_operation", operation_id)

    @router.post("/operations/{operation_id}/retry")
    def retry_operation(operation_id: str, payload: dict = Body(default_factory=dict)):
        return call("retry_operation", operation_id, payload)

    return router
