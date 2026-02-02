"""APIRouter for SAM3 prompt and SAM legacy endpoints."""

from __future__ import annotations

from typing import Any, Callable, Type

from fastapi import APIRouter


def build_sam3_prompts_router(
    *,
    sam3_text_fn: Callable[[Any], Any],
    sam3_text_auto_fn: Callable[[Any], Any],
    sam3_visual_fn: Callable[[Any], Any],
    sam_point_fn: Callable[[Any], Any],
    sam_bbox_auto_fn: Callable[[Any], Any],
    sam_point_auto_fn: Callable[[Any], Any],
    sam_point_multi_fn: Callable[[Any], Any],
    sam_point_multi_auto_fn: Callable[[Any], Any],
    sam_bbox_fn: Callable[[Any], Any],
    sam3_text_req: Type[Any],
    sam3_text_auto_req: Type[Any],
    sam3_visual_req: Type[Any],
    sam_point_req: Type[Any],
    sam_bbox_req: Type[Any],
    sam_point_multi_req: Type[Any],
    sam3_text_resp: Type[Any],
    sam3_text_auto_resp: Type[Any],
    sam_point_auto_resp: Type[Any],
    yolo_bbox_resp: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/sam3/text_prompt", response_model=sam3_text_resp)
    def sam3_text_prompt(payload: sam3_text_req):  # type: ignore[valid-type]
        return sam3_text_fn(payload)

    @router.post("/sam3/text_prompt_auto", response_model=sam3_text_auto_resp)
    def sam3_text_prompt_auto(payload: sam3_text_auto_req):  # type: ignore[valid-type]
        return sam3_text_auto_fn(payload)

    @router.post("/sam3/visual_prompt", response_model=sam3_text_resp)
    def sam3_visual_prompt(payload: sam3_visual_req):  # type: ignore[valid-type]
        return sam3_visual_fn(payload)

    @router.post("/sam_point", response_model=yolo_bbox_resp)
    def sam_point(payload: sam_point_req):  # type: ignore[valid-type]
        return sam_point_fn(payload)

    @router.post("/sam_bbox_auto", response_model=sam_point_auto_resp)
    def sam_bbox_auto(payload: sam_bbox_req):  # type: ignore[valid-type]
        return sam_bbox_auto_fn(payload)

    @router.post("/sam_point_auto", response_model=sam_point_auto_resp)
    def sam_point_auto(payload: sam_point_req):  # type: ignore[valid-type]
        return sam_point_auto_fn(payload)

    @router.post("/sam_point_multi", response_model=yolo_bbox_resp)
    def sam_point_multi(payload: sam_point_multi_req):  # type: ignore[valid-type]
        return sam_point_multi_fn(payload)

    @router.post("/sam_point_multi_auto", response_model=sam_point_auto_resp)
    def sam_point_multi_auto(payload: sam_point_multi_req):  # type: ignore[valid-type]
        return sam_point_multi_auto_fn(payload)

    @router.post("/sam_bbox", response_model=yolo_bbox_resp)
    def sam_bbox(payload: sam_bbox_req):  # type: ignore[valid-type]
        return sam_bbox_fn(payload)

    return router
