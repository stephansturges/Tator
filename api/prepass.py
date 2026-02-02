"""APIRouter for prepass recipe endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse


def build_prepass_router(
    *,
    list_fn: Callable[[], Any],
    get_fn: Callable[[str], Any],
    save_fn: Callable[[Any], Any],
    delete_fn: Callable[[str], Any],
    export_fn: Callable[[str], Any],
    import_fn: Callable[[UploadFile], Any],
    import_raw_fn: Callable[[Request], Any],
    response_cls: Type[Any],
    request_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/prepass/recipes")
    def list_prepass_recipes():
        return list_fn()

    @router.get("/prepass/recipes/{recipe_id}", response_model=response_cls)
    def get_prepass_recipe(recipe_id: str):
        return get_fn(recipe_id)

    @router.post("/prepass/recipes", response_model=response_cls)
    def save_prepass_recipe(payload: request_cls):  # type: ignore[valid-type]
        return save_fn(payload)

    @router.delete("/prepass/recipes/{recipe_id}")
    def delete_prepass_recipe(recipe_id: str):
        return delete_fn(recipe_id)

    @router.post("/prepass/recipes/{recipe_id}/export")
    def export_prepass_recipe(recipe_id: str):
        return export_fn(recipe_id)

    @router.post("/prepass/recipes/import", response_model=response_cls)
    def import_prepass_recipe(file: UploadFile = File(...)):  # noqa: B008
        return import_fn(file)

    @router.post("/prepass/recipes/import-raw", response_model=response_cls)
    async def import_prepass_recipe_raw(request: Request):
        return await import_raw_fn(request)

    return router
