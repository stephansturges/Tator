"""APIRouter for CLIP classifier/labelmap registry endpoints."""


from typing import Any, Callable

from fastapi import APIRouter, Query, Form
from fastapi.responses import StreamingResponse


def build_clip_registry_router(
    *,
    list_backbones_fn: Callable[[], Any],
    list_classifiers_fn: Callable[[], Any],
    list_labelmaps_fn: Callable[[], Any],
    download_classifier_fn: Callable[[str], StreamingResponse],
    download_classifier_zip_fn: Callable[[str], StreamingResponse],
    delete_classifier_fn: Callable[[str], Any],
    rename_classifier_fn: Callable[[str, str], Any],
    download_labelmap_fn: Callable[[str, str | None], StreamingResponse],
    delete_labelmap_fn: Callable[[str, str | None], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/clip/backbones")
    def list_clip_backbones():
        return list_backbones_fn()

    @router.get("/clip/classifiers")
    def list_clip_classifiers():
        return list_classifiers_fn()

    @router.get("/clip/labelmaps")
    def list_clip_labelmaps():
        return list_labelmaps_fn()

    @router.get("/clip/classifiers/download")
    def download_clip_classifier(rel_path: str = Query(...)):
        return download_classifier_fn(rel_path)

    @router.get("/clip/classifiers/download_zip")
    def download_clip_classifier_zip(rel_path: str = Query(...)):
        return download_classifier_zip_fn(rel_path)

    @router.delete("/clip/classifiers")
    def delete_clip_classifier(rel_path: str = Query(...)):
        return delete_classifier_fn(rel_path)

    @router.post("/clip/classifiers/rename")
    def rename_clip_classifier(rel_path: str = Form(...), new_name: str = Form(...)):
        return rename_classifier_fn(rel_path, new_name)

    @router.get("/clip/labelmaps/download")
    def download_clip_labelmap(rel_path: str = Query(...), root: str | None = Query(None)):
        return download_labelmap_fn(rel_path, root)

    @router.delete("/clip/labelmaps")
    def delete_clip_labelmap(rel_path: str = Query(...), root: str | None = Query(None)):
        return delete_labelmap_fn(rel_path, root)

    return router
