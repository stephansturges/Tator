"""APIRouter for dataset registry endpoints."""

from typing import Any, Callable, Optional

from fastapi import APIRouter, UploadFile, File, Form, Body


def build_datasets_router(
    *,
    list_fn: Callable[[], Any],
    upload_fn: Callable[[UploadFile, Optional[str], Optional[str], Optional[str]], Any],
    delete_fn: Callable[[str], Any],
    download_fn: Callable[[str], Any],
    build_qwen_fn: Callable[[str], Any],
    check_fn: Callable[[str], Any],
    get_glossary_fn: Callable[[str], Any],
    set_glossary_fn: Callable[[str, str], Any],
    get_text_label_fn: Callable[[str, str], Any],
    set_text_label_fn: Callable[[str, str, str], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/datasets")
    def list_datasets():
        return list_fn()

    @router.post("/datasets/upload")
    def upload_dataset(
        file: UploadFile = File(...),  # noqa: B008
        dataset_id: Optional[str] = Form(None),
        dataset_type: Optional[str] = Form(None),
        context: Optional[str] = Form(None),
    ):
        return upload_fn(file, dataset_id, dataset_type, context)

    @router.delete("/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str):
        return delete_fn(dataset_id)

    @router.get("/datasets/{dataset_id}/download")
    def download_dataset(dataset_id: str):
        return download_fn(dataset_id)

    @router.post("/datasets/{dataset_id}/build/qwen")
    def build_qwen_dataset(dataset_id: str):
        return build_qwen_fn(dataset_id)

    @router.get("/datasets/{dataset_id}/check")
    def check_dataset(dataset_id: str):
        return check_fn(dataset_id)

    @router.get("/datasets/{dataset_id}/glossary")
    def get_dataset_glossary(dataset_id: str):
        return get_glossary_fn(dataset_id)

    @router.post("/datasets/{dataset_id}/glossary")
    def set_dataset_glossary(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        glossary = str((payload or {}).get("glossary") or "")
        return set_glossary_fn(dataset_id, glossary)

    @router.get("/datasets/{dataset_id}/text_labels/{image_name}")
    def get_text_label(dataset_id: str, image_name: str):
        return get_text_label_fn(dataset_id, image_name)

    @router.post("/datasets/{dataset_id}/text_labels/{image_name}")
    def set_text_label(dataset_id: str, image_name: str, payload: dict = Body(...)):  # noqa: B008
        caption = str((payload or {}).get("caption") or "")
        return set_text_label_fn(dataset_id, image_name, caption)

    return router
