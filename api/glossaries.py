"""APIRouter for glossary library endpoints."""

from typing import Any, Callable

from fastapi import APIRouter, Body


def build_glossaries_router(
    *,
    list_fn: Callable[[], Any],
    get_fn: Callable[[str], Any],
    save_fn: Callable[[str, str], Any],
    delete_fn: Callable[[str], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/glossaries")
    def list_glossaries():
        return list_fn()

    @router.get("/glossaries/{name}")
    def get_glossary(name: str):
        return get_fn(name)

    @router.post("/glossaries")
    def save_glossary(payload: dict = Body(...)):  # noqa: B008
        name = str((payload or {}).get("name") or "").strip()
        glossary = str((payload or {}).get("glossary") or "")
        return save_fn(name, glossary)

    @router.delete("/glossaries/{name}")
    def delete_glossary(name: str):
        return delete_fn(name)

    return router
