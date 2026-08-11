"""APIRouter for SAM3 dataset endpoints."""

from typing import Any, Callable

from fastapi import APIRouter


def build_sam3_datasets_router(
    *,
    list_fn: Callable[[], Any],
    convert_fn: Callable[[str], Any],
    classes_fn: Callable[[str], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/sam3/datasets")
    def list_sam3_datasets():
        return list_fn()

    @router.post("/sam3/datasets/{dataset_id}/convert")
    def convert_sam3_dataset(dataset_id: str):
        return convert_fn(dataset_id)

    @router.get("/sam3/datasets/{dataset_id}/classes")
    def list_sam3_dataset_classes(dataset_id: str):
        return classes_fn(dataset_id)

    return router
