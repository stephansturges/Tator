"""APIRouter for system status endpoints."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter


def build_system_router(
    *,
    gpu_payload_fn: Callable[[], Any],
    storage_payload_fn: Callable[[], Any],
    health_summary_fn: Callable[[], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/system/gpu")
    def get_system_gpu():
        return gpu_payload_fn()

    @router.get("/system/storage_check")
    def system_storage_check():
        return storage_payload_fn()

    @router.get("/system/health_summary")
    def system_health_summary():
        return health_summary_fn()

    return router
