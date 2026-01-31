from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _ensure_yolo_inference_runtime_impl(
    *,
    load_active_fn: Callable[[], Dict[str, Any]],
    load_labelmap_fn: Callable[[Any], List[str]],
    patch_ultralytics_fn: Callable[[], None],
    yolo_lock: Any,
    get_state_fn: Callable[[], Tuple[Any, Optional[str], List[str], Optional[str]]],
    set_state_fn: Callable[[Any, Optional[str], List[str], Optional[str]], None],
    import_yolo_fn: Callable[[], Any],
    http_exception_cls: Any,
) -> Tuple[Any, List[str], Optional[str]]:
    active = load_active_fn()
    if not isinstance(active, dict):
        raise http_exception_cls(status_code=412, detail="yolo_active_missing")
    best_path = active.get("best_path")
    if not best_path:
        raise http_exception_cls(status_code=412, detail="yolo_active_missing")
    labelmap_path = active.get("labelmap_path")
    task = active.get("task")
    with yolo_lock:
        model, path, labelmap, cached_task = get_state_fn()
        if model is not None and path == best_path:
            return model, labelmap, cached_task
        try:
            YOLO = import_yolo_fn()
        except Exception as exc:  # noqa: BLE001
            raise http_exception_cls(status_code=503, detail=f"yolo_unavailable:{exc}") from exc
        patch_ultralytics_fn()
        model = YOLO(best_path)
        labelmap = load_labelmap_fn(labelmap_path) if labelmap_path else []
        resolved_task = task or getattr(model, "task", None)
        set_state_fn(model, best_path, labelmap, resolved_task)
        return model, labelmap, resolved_task


def _ensure_rfdetr_inference_runtime_impl(
    *,
    load_active_fn: Callable[[], Dict[str, Any]],
    load_labelmap_fn: Callable[[Any], List[str]],
    variant_info_fn: Callable[[str, Optional[str]], Dict[str, Any]],
    rfdetr_lock: Any,
    get_state_fn: Callable[[], Tuple[Any, Optional[str], List[str], Optional[str], Optional[str]]],
    set_state_fn: Callable[[Any, Optional[str], List[str], Optional[str], Optional[str]], None],
    import_rfdetr_fn: Callable[[], Dict[str, Any]],
    http_exception_cls: Any,
    torch_available: Callable[[], bool],
    resolve_device_fn: Callable[[], str],
) -> Tuple[Any, List[str], Optional[str]]:
    active = load_active_fn()
    if not isinstance(active, dict):
        raise http_exception_cls(status_code=412, detail="rfdetr_active_missing")
    best_path = active.get("best_path")
    if not best_path:
        raise http_exception_cls(status_code=412, detail="rfdetr_active_missing")
    if not Path(best_path).exists():
        raise http_exception_cls(status_code=412, detail="rfdetr_active_missing_weights")
    labelmap_path = active.get("labelmap_path")
    task = active.get("task") or "detect"
    variant = active.get("variant")
    with rfdetr_lock:
        model, path, labelmap, cached_task, cached_variant = get_state_fn()
        if model is not None and path == best_path:
            return model, labelmap, cached_task
        try:
            import_map = import_rfdetr_fn()
        except Exception as exc:  # noqa: BLE001
            raise http_exception_cls(status_code=503, detail=f"rfdetr_unavailable:{exc}") from exc
        variant_info = variant_info_fn(task, variant)
        variant_id = variant_info.get("id")
        model_cls = import_map.get(variant_id)
        if not model_cls:
            raise http_exception_cls(status_code=400, detail="rfdetr_variant_unknown")
        model_kwargs: Dict[str, Any] = {
            "pretrain_weights": best_path,
            "device": resolve_device_fn() if torch_available() else "cpu",
        }
        if variant_id == "rfdetr-seg-preview" or task == "segment":
            model_kwargs["segmentation_head"] = True
        model = model_cls(**model_kwargs)
        labelmap = load_labelmap_fn(labelmap_path) if labelmap_path else []
        if labelmap:
            try:
                model.model.class_names = labelmap
            except Exception:
                pass
        set_state_fn(model, best_path, labelmap, task, variant_id)
    return model, labelmap, task


def _load_yolo_active_impl(yolo_active_path: Path) -> Dict[str, Any]:
    if not yolo_active_path.exists():
        return {}
    try:
        return json.loads(yolo_active_path.read_text())
    except Exception:
        return {}


def _save_yolo_active_impl(payload: Dict[str, Any], yolo_active_path: Path) -> Dict[str, Any]:
    yolo_active_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload or {})
    data["updated_at"] = time.time()
    if "created_at" not in data:
        data["created_at"] = data["updated_at"]
    yolo_active_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data


def _load_rfdetr_active_impl(
    rfdetr_active_path: Path,
    rfdetr_job_root: Path,
    save_active_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    def _load_from(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    active = _load_from(rfdetr_active_path)
    best_path = str(active.get("best_path") or "")
    if best_path and Path(best_path).exists():
        return active

    fallback_path = rfdetr_job_root / "active.json"
    fallback = _load_from(fallback_path)
    fallback_best = str(fallback.get("best_path") or "")
    if fallback_best and Path(fallback_best).exists():
        try:
            save_active_fn(fallback)
        except Exception:
            pass
        return fallback
    return {}


def _save_rfdetr_active_impl(payload: Dict[str, Any], rfdetr_active_path: Path) -> Dict[str, Any]:
    rfdetr_active_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload or {})
    data["updated_at"] = time.time()
    if "created_at" not in data:
        data["created_at"] = data["updated_at"]
    rfdetr_active_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data


def _load_detector_default_impl(detector_default_path: Path) -> Dict[str, Any]:
    if not detector_default_path.exists():
        return {"mode": "rfdetr"}
    try:
        payload = json.loads(detector_default_path.read_text())
        if isinstance(payload, dict):
            mode = str(payload.get("mode") or "").strip().lower()
            if mode in {"yolo", "rfdetr"}:
                return payload
    except Exception:
        pass
    return {"mode": "rfdetr"}


def _save_detector_default_impl(
    payload: Dict[str, Any],
    detector_default_path: Path,
    http_exception_cls: Any,
) -> Dict[str, Any]:
    detector_default_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload or {})
    mode = str(data.get("mode") or "").strip().lower()
    if mode not in {"yolo", "rfdetr"}:
        raise http_exception_cls(status_code=400, detail="detector_mode_invalid")
    data["mode"] = mode
    data["updated_at"] = time.time()
    if "created_at" not in data:
        data["created_at"] = data["updated_at"]
    detector_default_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data


def _agent_tool_run_detector_impl(
    *,
    image_base64: Optional[str],
    image_token: Optional[str],
    detector_id: Optional[str],
    mode: Optional[str],
    conf: Optional[float],
    sahi: Optional[Dict[str, Any]],
    window: Optional[Any],
    window_bbox_2d: Optional[Sequence[float]],
    grid_cell: Optional[str],
    max_det: Optional[int],
    iou: Optional[float],
    merge_iou: Optional[float],
    expected_labelmap: Optional[Sequence[str]],
    register: Optional[bool],
    resolve_image_fn: Callable[[Optional[str], Optional[str], Optional[str]], Tuple[Any, Any, str]],
    normalize_window_fn: Callable[[Optional[Any], int, int], Optional[Tuple[float, float, float, float]]],
    ensure_yolo_runtime_fn: Callable[[], Tuple[Any, List[str], str]],
    ensure_rfdetr_runtime_fn: Callable[[], Tuple[Any, List[str], str]],
    raise_labelmap_mismatch_fn: Callable[[Optional[Sequence[str]], Optional[Sequence[str]], str], None],
    clamp_conf_fn: Callable[[float, List[str]], float],
    clamp_iou_fn: Callable[[float, List[str]], float],
    clamp_max_det_fn: Callable[[int, List[str]], int],
    clamp_slice_params_fn: Callable[[int, float, float, int, int, List[str]], Tuple[int, float, float]],
    slice_image_fn: Callable[[Any, int, float], Tuple[List[Any], List[Tuple[int, int]]]],
    yolo_extract_fn: Callable[[Any, List[str], float, float, int, int], List[Dict[str, Any]]],
    rfdetr_extract_fn: Callable[[Any, List[str], float, float, int, int], Tuple[List[Dict[str, Any]], bool]],
    merge_nms_fn: Callable[[List[Dict[str, Any]], float, int], List[Dict[str, Any]]],
    xywh_to_xyxy_fn: Callable[[Sequence[float]], Tuple[float, float, float, float]],
    det_payload_fn: Callable[..., Dict[str, Any]],
    register_detections_fn: Callable[..., Optional[Dict[str, Any]]],
    cluster_summaries_fn: Callable[[Sequence[int], bool], Dict[str, Any]],
    handles_from_cluster_ids_fn: Callable[[Sequence[int]], List[str]],
    cluster_label_counts_fn: Callable[[Sequence[int]], Dict[str, int]],
    agent_labelmap: Optional[Sequence[str]],
    agent_grid: Any,
    yolo_lock: Any,
    rfdetr_lock: Any,
    http_exception_cls: Any,
) -> Dict[str, Any]:
    pil_img, _, _ = resolve_image_fn(image_base64, image_token, None)
    img_w, img_h = pil_img.size
    mode_norm = (mode or "yolo").strip().lower()
    if mode_norm not in {"yolo", "rfdetr"}:
        raise http_exception_cls(status_code=400, detail="agent_detector_mode_invalid")
    window_xyxy = normalize_window_fn(window, img_w, img_h)
    if window_xyxy is None and window_bbox_2d is not None:
        window_xyxy = normalize_window_fn({"bbox_2d": window_bbox_2d}, img_w, img_h)
    crop_img = pil_img
    offset_x = 0.0
    offset_y = 0.0
    if window_xyxy:
        x1, y1, x2, y2 = window_xyxy
        crop_img = pil_img.crop((x1, y1, x2, y2))
        offset_x, offset_y = x1, y1
    detections: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if mode_norm == "yolo":
        model, labelmap, _task = ensure_yolo_runtime_fn()
        expected = list(expected_labelmap or (agent_labelmap or []))
        raise_labelmap_mismatch_fn(expected=expected or None, actual=labelmap, context="yolo")
        conf_val = clamp_conf_fn(float(conf) if conf is not None else 0.25, warnings)
        iou_val = clamp_iou_fn(float(iou) if iou is not None else 0.45, warnings)
        max_det_val = clamp_max_det_fn(int(max_det) if max_det is not None else 300, warnings)
        raw: List[Dict[str, Any]] = []
        if sahi and sahi.get("enabled"):
            try:
                slice_size = int(sahi.get("slice_size") or 640)
                overlap = float(sahi.get("overlap") or 0.2)
                merge_iou_val = float(merge_iou or sahi.get("merge_iou") or 0.5)
                slice_size, overlap, merge_iou_val = clamp_slice_params_fn(
                    slice_size, overlap, merge_iou_val, crop_img.width, crop_img.height, warnings
                )
                slices, starts = slice_image_fn(crop_img, slice_size, overlap)
                for tile, start in zip(slices, starts):
                    tile_offset_x = float(start[0]) + offset_x
                    tile_offset_y = float(start[1]) + offset_y
                    with yolo_lock:
                        results = model.predict(
                            __import__("PIL").Image.fromarray(tile),
                            conf=conf_val,
                            iou=iou_val,
                            max_det=max_det_val,
                            verbose=False,
                        )
                    raw.extend(yolo_extract_fn(results, labelmap, tile_offset_x, tile_offset_y, img_w, img_h))
                raw = merge_nms_fn(raw, merge_iou_val, max_det_val)
            except http_exception_cls as exc:
                if "sahi_unavailable" in str(exc.detail):
                    warnings.append(str(exc.detail))
                else:
                    warnings.append(f"sahi_failed:{exc.detail}")
                raw = []
        if not raw:
            with yolo_lock:
                results = model.predict(
                    crop_img,
                    conf=conf_val,
                    iou=iou_val,
                    max_det=max_det_val,
                    verbose=False,
                )
            raw = yolo_extract_fn(results, labelmap, offset_x, offset_y, img_w, img_h)
        for det in raw:
            x1, y1, x2, y2 = xywh_to_xyxy_fn(det.get("bbox") or [])
            detections.append(
                det_payload_fn(
                    img_w,
                    img_h,
                    (x1, y1, x2, y2),
                    label=det.get("class_name"),
                    class_id=det.get("class_id"),
                    score=det.get("score"),
                    source="yolo",
                    window=window_xyxy,
                )
            )
    else:
        model, labelmap, _task = ensure_rfdetr_runtime_fn()
        expected = list(expected_labelmap or (agent_labelmap or []))
        raise_labelmap_mismatch_fn(expected=expected or None, actual=labelmap, context="rfdetr")
        conf_val = clamp_conf_fn(float(conf) if conf is not None else 0.25, warnings)
        max_det_val = clamp_max_det_fn(int(max_det) if max_det is not None else 300, warnings)
        raw: List[Dict[str, Any]] = []
        if sahi and sahi.get("enabled"):
            try:
                slice_size = int(sahi.get("slice_size") or 640)
                overlap = float(sahi.get("overlap") or 0.2)
                merge_iou_val = float(merge_iou or sahi.get("merge_iou") or 0.5)
                slice_size, overlap, merge_iou_val = clamp_slice_params_fn(
                    slice_size, overlap, merge_iou_val, crop_img.width, crop_img.height, warnings
                )
                slices, starts = slice_image_fn(crop_img, slice_size, overlap)
                for tile, start in zip(slices, starts):
                    tile_offset_x = float(start[0]) + offset_x
                    tile_offset_y = float(start[1]) + offset_y
                    try:
                        with rfdetr_lock:
                            results = model.predict(__import__("PIL").Image.fromarray(tile), threshold=conf_val)
                    except Exception as exc:  # noqa: BLE001
                        raise http_exception_cls(status_code=500, detail=f"rfdetr_predict_failed:{exc}") from exc
                    extracted, shifted = rfdetr_extract_fn(
                        results, labelmap, tile_offset_x, tile_offset_y, img_w, img_h
                    )
                    if shifted:
                        if expected:
                            raise http_exception_cls(
                                status_code=412,
                                detail="detector_labelmap_shifted:rfdetr",
                            )
                        warnings.append("rfdetr_labelmap_shifted")
                    raw.extend(extracted)
                raw = merge_nms_fn(raw, merge_iou_val, max_det_val)
            except http_exception_cls as exc:
                if "sahi_unavailable" in str(exc.detail):
                    warnings.append(str(exc.detail))
                else:
                    warnings.append(f"sahi_failed:{exc.detail}")
                raw = []
        if not raw:
            try:
                with rfdetr_lock:
                    results = model.predict(crop_img, threshold=conf_val)
            except Exception as exc:  # noqa: BLE001
                raise http_exception_cls(status_code=500, detail=f"rfdetr_predict_failed:{exc}") from exc
            raw, shifted = rfdetr_extract_fn(results, labelmap, offset_x, offset_y, img_w, img_h)
            if shifted:
                if expected:
                    raise http_exception_cls(
                        status_code=412,
                        detail="detector_labelmap_shifted:rfdetr",
                    )
                warnings.append("rfdetr_labelmap_shifted")
        raw.sort(key=lambda det: float(det.get("score") or 0.0), reverse=True)
        for det in raw[:max_det_val]:
            x1, y1, x2, y2 = xywh_to_xyxy_fn(det.get("bbox") or [])
            detections.append(
                det_payload_fn(
                    img_w,
                    img_h,
                    (x1, y1, x2, y2),
                    label=det.get("class_name"),
                    class_id=det.get("class_id"),
                    score=det.get("score"),
                    source="rfdetr",
                    window=window_xyxy,
                )
            )
    register_summary: Optional[Dict[str, Any]] = None
    if register:
        register_summary = register_detections_fn(
            detections,
            img_w=img_w,
            img_h=img_h,
            grid=agent_grid,
            labelmap=agent_labelmap or [],
            background=None,
            source_override=None,
            owner_cell=grid_cell,
        )
    new_cluster_ids = register_summary.get("new_cluster_ids") if isinstance(register_summary, dict) else []
    updated_cluster_ids = register_summary.get("updated_cluster_ids") if isinstance(register_summary, dict) else []
    new_summary = cluster_summaries_fn(new_cluster_ids, include_ids=False)
    new_handles = handles_from_cluster_ids_fn(new_cluster_ids or [])
    updated_handles = handles_from_cluster_ids_fn(updated_cluster_ids or [])
    agent_view = {
        "mode": mode_norm,
        "grid_cell": grid_cell,
        "warnings": warnings or None,
        "new_clusters": register_summary.get("new_clusters") if isinstance(register_summary, dict) else 0,
        "new_handles": new_handles,
        "updated_clusters": len(updated_cluster_ids or []),
        "updated_handles": updated_handles,
        "new_items": new_summary.get("items"),
        "new_items_total": new_summary.get("total"),
        "new_items_truncated": new_summary.get("truncated"),
        "label_counts": cluster_label_counts_fn(new_cluster_ids or []),
    }
    return {
        "detections": detections,
        "warnings": warnings or None,
        "register_summary": register_summary,
        "__agent_view__": agent_view,
    }
