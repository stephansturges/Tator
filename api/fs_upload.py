"""APIRouter for filesystem upload endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter, File, UploadFile


def build_fs_upload_router(
    *,
    upload_classifier_fn: Callable[[UploadFile], Any],
    upload_labelmap_fn: Callable[[UploadFile], Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/fs/upload_classifier")
    async def upload_classifier(file: UploadFile = File(...)):
        return await upload_classifier_fn(file)

    @router.post("/fs/upload_labelmap")
    async def upload_labelmap(file: UploadFile = File(...)):
        return await upload_labelmap_fn(file)

    return router
