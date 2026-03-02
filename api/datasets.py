"""APIRouter for dataset registry endpoints."""

from typing import Any, Callable, Optional

from fastapi import APIRouter, UploadFile, File, Form, Body, Query


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
    register_path_fn: Callable[
        [
            str,
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[bool],
            Optional[bool],
        ],
        Any,
    ],
    open_path_fn: Callable[[str, Optional[bool]], Any],
    save_transient_fn: Callable[
        [str, Optional[str], Optional[str], Optional[str], Optional[str]], Any
    ],
    delete_transient_fn: Callable[[str], Any],
    annotation_session_start_fn: Callable[[str, dict], Any],
    annotation_session_heartbeat_fn: Callable[[str, dict], Any],
    annotation_session_stop_fn: Callable[[str, dict], Any],
    annotation_manifest_fn: Callable[[str], Any],
    annotation_image_fn: Callable[[str, str, str], Any],
    annotation_snapshot_fn: Callable[[str, dict], Any],
    annotation_meta_patch_fn: Callable[[str, dict], Any],
    transient_annotation_manifest_fn: Callable[[str], Any],
    transient_annotation_image_fn: Callable[[str, str, str], Any],
    transient_annotation_snapshot_fn: Callable[[str, dict], Any],
    transient_annotation_meta_patch_fn: Callable[[str, dict], Any],
    transient_annotation_session_start_fn: Callable[[str, dict], Any],
    transient_annotation_session_heartbeat_fn: Callable[[str, dict], Any],
    transient_annotation_session_stop_fn: Callable[[str, dict], Any],
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

    @router.post("/datasets/register_path")
    def register_dataset_path(payload: dict = Body(...)):  # noqa: B008
        return register_path_fn(
            str((payload or {}).get("path") or ""),
            (payload or {}).get("dataset_id"),
            (payload or {}).get("label"),
            (payload or {}).get("context"),
            (payload or {}).get("notes"),
            (payload or {}).get("force_new"),
            (payload or {}).get("strict"),
        )

    @router.post("/datasets/open_path")
    def open_dataset_path(payload: dict = Body(...)):  # noqa: B008
        return open_path_fn(
            str((payload or {}).get("path") or ""),
            (payload or {}).get("strict"),
        )

    @router.post("/datasets/transient/{session_id}/save")
    def save_transient_dataset(session_id: str, payload: dict = Body(...)):  # noqa: B008
        return save_transient_fn(
            session_id,
            (payload or {}).get("dataset_id"),
            (payload or {}).get("label"),
            (payload or {}).get("context"),
            (payload or {}).get("notes"),
        )

    @router.delete("/datasets/transient/{session_id}")
    def delete_transient_dataset(session_id: str):
        return delete_transient_fn(session_id)

    @router.post("/datasets/{dataset_id}/annotation/session/start")
    def start_annotation_session(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        return annotation_session_start_fn(dataset_id, payload or {})

    @router.post("/datasets/{dataset_id}/annotation/session/heartbeat")
    def heartbeat_annotation_session(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        return annotation_session_heartbeat_fn(dataset_id, payload or {})

    @router.post("/datasets/{dataset_id}/annotation/session/stop")
    def stop_annotation_session(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        return annotation_session_stop_fn(dataset_id, payload or {})

    @router.get("/datasets/{dataset_id}/annotation/manifest")
    def get_annotation_manifest(dataset_id: str):
        return annotation_manifest_fn(dataset_id)

    @router.get("/datasets/{dataset_id}/annotation/image")
    def get_annotation_image(
        dataset_id: str,
        split: str = Query(...),  # noqa: B008
        image_relpath: str = Query(...),  # noqa: B008
    ):
        return annotation_image_fn(dataset_id, split, image_relpath)

    @router.post("/datasets/{dataset_id}/annotation/snapshot")
    def save_annotation_snapshot(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        return annotation_snapshot_fn(dataset_id, payload or {})

    @router.patch("/datasets/{dataset_id}/annotation/meta")
    def patch_annotation_meta(dataset_id: str, payload: dict = Body(...)):  # noqa: B008
        return annotation_meta_patch_fn(dataset_id, payload or {})

    @router.post("/datasets/transient/{session_id}/annotation/session/start")
    def start_transient_annotation_session(
        session_id: str, payload: dict = Body(...)
    ):  # noqa: B008
        return transient_annotation_session_start_fn(session_id, payload or {})

    @router.post("/datasets/transient/{session_id}/annotation/session/heartbeat")
    def heartbeat_transient_annotation_session(
        session_id: str, payload: dict = Body(...)
    ):  # noqa: B008
        return transient_annotation_session_heartbeat_fn(session_id, payload or {})

    @router.post("/datasets/transient/{session_id}/annotation/session/stop")
    def stop_transient_annotation_session(session_id: str, payload: dict = Body(...)):  # noqa: B008
        return transient_annotation_session_stop_fn(session_id, payload or {})

    @router.get("/datasets/transient/{session_id}/annotation/manifest")
    def get_transient_annotation_manifest(session_id: str):
        return transient_annotation_manifest_fn(session_id)

    @router.get("/datasets/transient/{session_id}/annotation/image")
    def get_transient_annotation_image(
        session_id: str,
        split: str = Query(...),  # noqa: B008
        image_relpath: str = Query(...),  # noqa: B008
    ):
        return transient_annotation_image_fn(session_id, split, image_relpath)

    @router.post("/datasets/transient/{session_id}/annotation/snapshot")
    def save_transient_annotation_snapshot(
        session_id: str, payload: dict = Body(...)
    ):  # noqa: B008
        return transient_annotation_snapshot_fn(session_id, payload or {})

    @router.patch("/datasets/transient/{session_id}/annotation/meta")
    def patch_transient_annotation_meta(session_id: str, payload: dict = Body(...)):  # noqa: B008
        return transient_annotation_meta_patch_fn(session_id, payload or {})

    return router
