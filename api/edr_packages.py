"""APIRouter for hermetic EDR package endpoints."""

from typing import Any, Callable

from fastapi import APIRouter, File, Request, UploadFile


def build_edr_packages_router(
    *,
    list_fn: Callable[[], Any],
    get_fn: Callable[[str], Any],
    export_fn: Callable[[str], Any],
    import_fn: Callable[[UploadFile], Any],
    import_raw_fn: Callable[[Request], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/edr/packages")
    def list_edr_packages():
        return list_fn()

    @router.get("/edr/packages/{package_id}")
    def get_edr_package(package_id: str):
        return get_fn(package_id)

    @router.get("/edr/packages/{package_id}/export")
    def export_edr_package(package_id: str):
        return export_fn(package_id)

    @router.post("/edr/packages/import")
    def import_edr_package(file: UploadFile = File(...)):  # noqa: B008
        return import_fn(file)

    @router.post("/edr/packages/import-raw")
    async def import_edr_package_raw(request: Request):
        return await import_raw_fn(request)

    return router
