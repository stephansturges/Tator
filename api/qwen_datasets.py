"""APIRouter for Qwen dataset upload/list endpoints."""

from typing import Any, Callable, Optional

from fastapi import APIRouter, UploadFile, File, Form


def build_qwen_datasets_router(
    *,
    list_fn: Callable[[], Any],
    delete_fn: Callable[[str], Any],
    init_fn: Callable[[Optional[str]], Any],
    chunk_fn: Callable[[str, str, str, str, UploadFile], Any],
    finalize_fn: Callable[[str, dict, Optional[str]], Any],
    cancel_fn: Callable[[str], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/qwen/datasets")
    def list_qwen_datasets():
        return list_fn()

    @router.delete("/qwen/datasets/{dataset_id}")
    def delete_qwen_dataset(dataset_id: str):
        return delete_fn(dataset_id)

    @router.post("/qwen/dataset/init")
    def init_qwen_dataset(run_name: Optional[str] = Form(None)):
        return init_fn(run_name)

    @router.post("/qwen/dataset/chunk")
    def upload_qwen_dataset_chunk(
        job_id: str = Form(...),
        split: str = Form(...),
        image_name: str = Form(...),
        annotation_line: str = Form(...),
        file: UploadFile = File(...),  # noqa: B008
    ):
        return chunk_fn(job_id, split, image_name, annotation_line, file)

    @router.post("/qwen/dataset/finalize")
    def finalize_qwen_dataset(
        job_id: str = Form(...),
        metadata: str = Form("{}"),
        run_name: Optional[str] = Form(None),
    ):
        meta = {}
        try:
            import json

            meta = json.loads(metadata) if metadata else {}
        except Exception:
            meta = {}
        return finalize_fn(job_id, meta, run_name)

    @router.post("/qwen/dataset/cancel")
    def cancel_qwen_dataset(job_id: str = Form(...)):
        return cancel_fn(job_id)

    return router
