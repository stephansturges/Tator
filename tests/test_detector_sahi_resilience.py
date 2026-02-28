from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import HTTPException
from PIL import Image

from services.detectors import _agent_tool_run_detector_impl


class _NoOpLock:
    def __enter__(self) -> "_NoOpLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@dataclass
class _DummyYolo:
    calls: int = 0

    def predict(self, *_args, **_kwargs) -> List[Dict[str, Any]]:
        self.calls += 1
        return [{"ok": True}]


@dataclass
class _DummyRfDetr:
    calls: int = 0

    def predict(self, *_args, **_kwargs) -> Dict[str, Any]:
        self.calls += 1
        return {"ok": True}


def _resolve_image(
    _image_base64: Optional[str], _image_token: Optional[str], _cache_key: Optional[str]
) -> Tuple[Image.Image, None, str]:
    return Image.new("RGB", (64, 64)), None, "img-token"


def _normalize_window(_window: Optional[Any], _img_w: int, _img_h: int) -> None:
    return None


def _clamp_conf(value: float, _warnings: List[str]) -> float:
    return value


def _clamp_iou(value: float, _warnings: List[str]) -> float:
    return value


def _clamp_max_det(value: int, _warnings: List[str]) -> int:
    return value


def _clamp_slice(
    slice_size: int, overlap: float, merge_iou: float, _w: int, _h: int, _warnings: List[str]
) -> Tuple[int, float, float]:
    return slice_size, overlap, merge_iou


def _slice_image(_img: Any, _slice_size: int, _overlap: float) -> Tuple[List[Any], List[Tuple[int, int]]]:
    # Intentionally mismatched metadata: no start offsets for provided slices.
    return [object()], []


def _xywh_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox[:4]
    return float(x), float(y), float(x + w), float(y + h)


def _det_payload(
    _img_w: int,
    _img_h: int,
    bbox_xyxy: Tuple[float, float, float, float],
    *,
    label: Optional[str],
    class_id: Optional[int],
    score: Optional[float],
    source: str,
    window: Optional[Tuple[float, float, float, float]],
) -> Dict[str, Any]:
    return {
        "bbox_xyxy": list(bbox_xyxy),
        "label": label,
        "class_id": class_id,
        "score": score,
        "source": source,
        "window": list(window) if window else None,
    }


def _merge_passthrough(raw: List[Dict[str, Any]], _iou: float, _max_det: int) -> List[Dict[str, Any]]:
    return raw


def _register_none(*_args, **_kwargs) -> None:
    return None


def _cluster_summary(_ids: Sequence[int], include_ids: bool = False) -> Dict[str, Any]:
    _ = include_ids
    return {"items": [], "total": 0, "truncated": False}


def _handles(_ids: Sequence[int]) -> List[str]:
    return []


def _label_counts(_ids: Sequence[int]) -> Dict[str, int]:
    return {}


def test_yolo_sahi_mismatched_slice_metadata_falls_back_to_full_image() -> None:
    model = _DummyYolo()

    out = _agent_tool_run_detector_impl(
        image_base64=None,
        image_token=None,
        detector_id=None,
        mode="yolo",
        conf=0.25,
        sahi={"enabled": True, "slice_size": 640, "overlap": 0.2, "merge_iou": 0.5},
        window=None,
        window_bbox_2d=None,
        grid_cell=None,
        max_det=300,
        iou=0.45,
        merge_iou=0.5,
        expected_labelmap=None,
        register=False,
        resolve_image_fn=_resolve_image,
        normalize_window_fn=_normalize_window,
        ensure_yolo_runtime_fn=lambda: (model, ["vehicle"], "detect"),
        ensure_rfdetr_runtime_fn=lambda: (_DummyRfDetr(), ["vehicle"], "detect"),
        ensure_yolo_runtime_by_id_fn=None,
        ensure_rfdetr_runtime_by_id_fn=None,
        raise_labelmap_mismatch_fn=lambda expected, actual, context: None,
        clamp_conf_fn=_clamp_conf,
        clamp_iou_fn=_clamp_iou,
        clamp_max_det_fn=_clamp_max_det,
        clamp_slice_params_fn=_clamp_slice,
        slice_image_fn=_slice_image,
        yolo_extract_fn=lambda *_args, **_kwargs: [
            {"bbox": [1.0, 1.0, 8.0, 8.0], "class_name": "vehicle", "class_id": 0, "score": 0.9}
        ],
        rfdetr_extract_fn=lambda *_args, **_kwargs: ([], False),
        merge_nms_fn=_merge_passthrough,
        xywh_to_xyxy_fn=_xywh_to_xyxy,
        det_payload_fn=_det_payload,
        register_detections_fn=_register_none,
        cluster_summaries_fn=_cluster_summary,
        handles_from_cluster_ids_fn=_handles,
        cluster_label_counts_fn=_label_counts,
        agent_labelmap=["vehicle"],
        agent_grid=None,
        yolo_lock=_NoOpLock(),
        rfdetr_lock=_NoOpLock(),
        http_exception_cls=HTTPException,
    )

    assert model.calls == 1
    assert len(out["detections"]) == 1
    assert out["detections"][0]["source"] == "yolo"


def test_rfdetr_sahi_mismatched_slice_metadata_falls_back_to_full_image() -> None:
    model = _DummyRfDetr()

    out = _agent_tool_run_detector_impl(
        image_base64=None,
        image_token=None,
        detector_id=None,
        mode="rfdetr",
        conf=0.25,
        sahi={"enabled": True, "slice_size": 640, "overlap": 0.2, "merge_iou": 0.5},
        window=None,
        window_bbox_2d=None,
        grid_cell=None,
        max_det=300,
        iou=None,
        merge_iou=0.5,
        expected_labelmap=None,
        register=False,
        resolve_image_fn=_resolve_image,
        normalize_window_fn=_normalize_window,
        ensure_yolo_runtime_fn=lambda: (_DummyYolo(), ["vehicle"], "detect"),
        ensure_rfdetr_runtime_fn=lambda: (model, ["vehicle"], "detect"),
        ensure_yolo_runtime_by_id_fn=None,
        ensure_rfdetr_runtime_by_id_fn=None,
        raise_labelmap_mismatch_fn=lambda expected, actual, context: None,
        clamp_conf_fn=_clamp_conf,
        clamp_iou_fn=_clamp_iou,
        clamp_max_det_fn=_clamp_max_det,
        clamp_slice_params_fn=_clamp_slice,
        slice_image_fn=_slice_image,
        yolo_extract_fn=lambda *_args, **_kwargs: [],
        rfdetr_extract_fn=lambda *_args, **_kwargs: (
            [{"bbox": [2.0, 2.0, 6.0, 6.0], "class_name": "vehicle", "class_id": 0, "score": 0.8}],
            False,
        ),
        merge_nms_fn=_merge_passthrough,
        xywh_to_xyxy_fn=_xywh_to_xyxy,
        det_payload_fn=_det_payload,
        register_detections_fn=_register_none,
        cluster_summaries_fn=_cluster_summary,
        handles_from_cluster_ids_fn=_handles,
        cluster_label_counts_fn=_label_counts,
        agent_labelmap=["vehicle"],
        agent_grid=None,
        yolo_lock=_NoOpLock(),
        rfdetr_lock=_NoOpLock(),
        http_exception_cls=HTTPException,
    )

    assert model.calls == 1
    assert len(out["detections"]) == 1
    assert out["detections"][0]["source"] == "rfdetr"
