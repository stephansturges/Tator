"""SAM3 similarity expansion helpers."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from services.prepass_provenance import _agent_attach_provenance
from services.prepass_windows import _agent_exemplars_for_window, _agent_similarity_windows


def _agent_run_similarity_global(
    payload: Any,
    *,
    pil_img: Image.Image,
    image_token: str,
    exemplars_by_label: Dict[str, List[Dict[str, Any]]],
    sam3_similarity_fn: Callable[..., Dict[str, Any]],
    trace_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_full_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_readable: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    detections: List[Dict[str, Any]] = []
    warnings: List[str] = []
    score_thr = payload.prepass_similarity_score
    mask_thr = payload.sam3_mask_threshold

    def _log_step(event: str, payload_obj: Dict[str, Any]) -> None:
        if trace_writer:
            trace_writer({"type": event, **payload_obj, "ts": time.time()})
        if trace_full_writer:
            trace_full_writer({"type": event, **payload_obj, "ts": time.time()})

    for label, exemplars in exemplars_by_label.items():
        exemplar_boxes = []
        exemplar_handles = []
        for det in exemplars:
            bbox = det.get("bbox_2d")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                exemplar_boxes.append({"bbox_2d": list(bbox[:4]), "bbox_space": "full"})
            handle = det.get("handle")
            if handle:
                exemplar_handles.append(str(handle))
        if not exemplar_boxes:
            continue
        args = {
            "image_token": image_token,
            "label": label,
            "exemplar_boxes": exemplar_boxes,
            "score_thr": score_thr,
            "mask_threshold": mask_thr,
        }
        _log_step("deep_prepass_tool_call", {"tool": "sam3_similarity", "args": args})
        try:
            result = sam3_similarity_fn(
                image_token=image_token,
                exemplar_boxes=exemplar_boxes,
                label=label,
                score_thr=score_thr,
                mask_threshold=mask_thr,
                register=False,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"deep_prepass_similarity_failed:{label}:full:{exc}")
            continue
        _log_step("deep_prepass_tool_result", {"tool": "sam3_similarity", "result": result})
        dets = list(result.get("detections") or [])
        if trace_readable:
            trace_readable(
                f"deep_prepass sam3_similarity label={label} window=full detections={len(dets)}"
            )
        _agent_attach_provenance(
            dets,
            source="sam3_similarity",
            source_primary="sam3_similarity",
            source_exemplar_handles=exemplar_handles or None,
        )
        detections.extend(dets)
    return {"detections": detections, "warnings": warnings}


def _agent_run_similarity_expansion(
    payload: Any,
    *,
    pil_img: Image.Image,
    image_token: str,
    exemplars_by_label: Dict[str, List[Dict[str, Any]]],
    sam3_similarity_fn: Callable[..., Dict[str, Any]],
    grid_overlap_ratio_default: float,
    trace_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_full_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_readable: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    img_w, img_h = pil_img.size
    detections: List[Dict[str, Any]] = []
    warnings: List[str] = []
    windows = _agent_similarity_windows(
        payload, pil_img=pil_img, grid_overlap_ratio_default=grid_overlap_ratio_default
    )
    score_thr = payload.prepass_similarity_score
    mask_thr = payload.sam3_mask_threshold

    def _log_step(event: str, payload_obj: Dict[str, Any]) -> None:
        if trace_writer:
            trace_writer({"type": event, **payload_obj, "ts": time.time()})
        if trace_full_writer:
            trace_full_writer({"type": event, **payload_obj, "ts": time.time()})

    for label, exemplars in exemplars_by_label.items():
        for window in windows:
            window_xyxy = window.get("bbox_xyxy_px") or []
            window_bbox_2d = window.get("bbox_2d")
            window_exemplars = _agent_exemplars_for_window(
                exemplars, img_w=img_w, img_h=img_h, window_xyxy=window_xyxy
            )
            if not window_exemplars:
                continue
            exemplar_boxes = []
            exemplar_handles = []
            for det in window_exemplars:
                bbox = det.get("bbox_2d")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    exemplar_boxes.append({"bbox_2d": list(bbox[:4]), "bbox_space": "full"})
                handle = det.get("handle")
                if handle:
                    exemplar_handles.append(str(handle))
            if not exemplar_boxes:
                continue
            args = {
                "image_token": image_token,
                "label": label,
                "exemplar_boxes": exemplar_boxes,
                "score_thr": score_thr,
                "mask_threshold": mask_thr,
                "window_bbox_2d": window_bbox_2d,
            }
            if window.get("grid_cell"):
                args["grid_cell"] = window.get("grid_cell")
            _log_step("deep_prepass_tool_call", {"tool": "sam3_similarity", "args": args})
            try:
                result = sam3_similarity_fn(
                    image_token=image_token,
                    exemplar_boxes=exemplar_boxes,
                    label=label,
                    score_thr=score_thr,
                    mask_threshold=mask_thr,
                    window_bbox_2d=window_bbox_2d,
                    grid_cell=window.get("grid_cell"),
                    register=False,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"deep_prepass_similarity_failed:{label}:{window.get('name')}:{exc}")
                continue
            _log_step("deep_prepass_tool_result", {"tool": "sam3_similarity", "result": result})
            dets = list(result.get("detections") or [])
            if trace_readable:
                window_name = window.get("name") or window.get("grid_cell") or "window"
                trace_readable(
                    f"deep_prepass sam3_similarity label={label} window={window_name} detections={len(dets)}"
                )
            _agent_attach_provenance(
                dets,
                source="sam3_similarity",
                source_primary="sam3_similarity",
                source_exemplar_handles=exemplar_handles or None,
            )
            detections.extend(dets)
    return {"detections": detections, "warnings": warnings}
