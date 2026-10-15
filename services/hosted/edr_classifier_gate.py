"""Preserved prepass classifier gate with explicit per-pipeline bindings."""
from __future__ import annotations
from types import FunctionType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from PIL import Image
import numpy as np
from utils.coords import _agent_xyxy_to_xywh, _resolve_agent_bbox_xyxy, _xyxy_to_qwen_bbox, _xyxy_to_yolo_norm
from utils.labels import _agent_fuzzy_align_label
from utils.classifier_utils import _clip_head_background_indices, _clip_head_classes, _find_clip_head_target_index
from services.prepass import _agent_cluster_match, _agent_det_score
from services.prepass_grid import _agent_grid_cell_for_detection
from utils.embedding_recipe import make_embedding_crop_views as _embedding_make_crop_views
from utils.embedding_recipe import normalize_background_mode as _embedding_normalize_background_mode
from utils.embedding_recipe import normalize_preprocess_mode as _embedding_normalize_preprocess
from utils.embedding_recipe import normalize_embedding_view_mode as _embedding_normalize_view_mode

def _clip_head_keep_mask(
    proba_arr: np.ndarray,
    *,
    target_index: int,
    min_prob: float,
    margin: float,
    background_indices: Optional[Sequence[int]] = None,
    background_guard: bool = True,
    background_margin: float = 0.0,
) -> Optional[np.ndarray]:
    if proba_arr is None or proba_arr.ndim != 2:
        return None
    num_classes = proba_arr.shape[1]
    if target_index < 0 or target_index >= num_classes:
        return None
    target = proba_arr[:, target_index]
    if num_classes > 1:
        mask = np.ones(num_classes, dtype=bool)
        mask[target_index] = False
        best_other = np.max(proba_arr[:, mask], axis=1)
    else:
        best_other = np.zeros_like(target)
    keep = (target >= float(min_prob)) & ((target - best_other) >= float(margin))
    if background_guard and background_indices:
        bg = np.max(proba_arr[:, list(background_indices)], axis=1)
        keep &= (target - bg) >= float(background_margin)
    return keep


def _classifier_crop_for_head(
    pil_img: Image.Image,
    xyxy: Sequence[float],
    head: Optional[Mapping[str, Any]],
) -> Tuple[Any, Dict[str, Any]]:
    """Crop and preprocess an object using the classifier's saved embedding recipe."""
    img_w, img_h = pil_img.size
    x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
    crop_mode = str((head or {}).get("embedding_crop_mode") or (head or {}).get("crop_mode") or "padded_square")
    padding = _coerce_float(
        (head or {}).get("embedding_crop_padding_ratio")
        if (head or {}).get("embedding_crop_padding_ratio") is not None
        else (head or {}).get("crop_padding_ratio"),
        0.08,
        minimum=0.0,
        maximum=1.0,
    )
    preprocess_mode = _embedding_normalize_preprocess((head or {}).get("preprocess_mode"))
    canonical_size = min(1024, _coerce_int((head or {}).get("canonical_size"), 336, minimum=64))
    background_mode = _embedding_normalize_background_mode((head or {}).get("background_mode"))
    view_mode = _embedding_normalize_view_mode((head or {}).get("embedding_view_mode"))
    views, crop_xyxy, view_metadata = _embedding_make_crop_views(
        pil_img,
        [x1, y1, x2, y2],
        crop_mode=crop_mode,
        padding_ratio=padding,
        preprocess_mode=preprocess_mode,
        canonical_size=canonical_size,
        background_mode=background_mode,
        view_mode=view_mode,
    )
    record = {
        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        "crop_xyxy": [int(v) for v in crop_xyxy],
        "width": int(round(max(1.0, x2 - x1))),
        "height": int(round(max(1.0, y2 - y1))),
        "background_mode": background_mode,
        "embedding_view_mode": view_mode,
        "embedding_views": view_metadata,
    }
    return (tuple(views) if len(views) > 1 else views[0]), record


def _agent_sanitize_detection_items(
    items: List[Dict[str, Any]],
    *,
    pil_img: Optional[Image.Image] = None,
    classifier_head: Optional[Dict[str, Any]] = None,
    img_w: int,
    img_h: int,
    labelmap: List[str],
    background: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    background_set = set(background or [])
    label_index = {label: idx for idx, label in enumerate(labelmap)}
    cleaned: List[Dict[str, Any]] = []
    rejected = 0
    pending: List[Dict[str, Any]] = []
    pending_crops: List[Any] = []
    pending_crop_records: List[Dict[str, Any]] = []
    for ann in items:
        label = str(ann.get("label") or ann.get("class_name") or "").strip()
        aligned = _agent_fuzzy_align_label(label, labelmap)
        if not aligned or aligned not in label_index or aligned in background_set:
            rejected += 1
            continue
        raw_score = ann.get("score")
        score: Optional[float]
        if raw_score is None:
            score = None
        else:
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = None
        score_source = ann.get("score_source")
        if not score_source:
            score_source = ann.get("source") or ("unknown" if score is None else None)
        window_bbox = ann.get("window_bbox_2d")
        xyxy = _resolve_agent_bbox_xyxy(ann, img_w, img_h, window_bbox_2d=window_bbox)
        if xyxy is None:
            rejected += 1
            continue
        x1, y1, x2, y2 = xyxy
        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))
        if x2 <= x1 or y2 <= y1:
            rejected += 1
            continue
        base_entry = {
            "bbox_xyxy_px": [x1, y1, x2, y2],
            "bbox_xywh_px": _agent_xyxy_to_xywh(x1, y1, x2, y2),
            "bbox_2d": list(_xyxy_to_qwen_bbox(img_w, img_h, x1, y1, x2, y2)),
            "bbox_yolo": list(_xyxy_to_yolo_norm(img_w, img_h, x1, y1, x2, y2)),
            "label": aligned,
            "class_id": label_index[aligned],
            "score": score,
            "score_source": score_source,
            "sam3_prompt_term": ann.get("sam3_prompt_term"),
            "sam3_prompt_label": ann.get("sam3_prompt_label"),
            "sam3_prompt_source": ann.get("sam3_prompt_source"),
            "source": ann.get("source") or "agent",
        }
        source_list = ann.get("source_list")
        if isinstance(source_list, (list, tuple)):
            base_entry["source_list"] = [str(item) for item in source_list if item]
        score_by_source = ann.get("score_by_source")
        if isinstance(score_by_source, dict):
            normalized_scores: Dict[str, float] = {}
            for raw_source, raw_source_score in score_by_source.items():
                source_key = str(raw_source or "").strip().lower()
                if not source_key:
                    continue
                try:
                    source_score_val = float(raw_source_score)
                except (TypeError, ValueError):
                    continue
                prev_val = normalized_scores.get(source_key)
                if prev_val is None or source_score_val > prev_val:
                    normalized_scores[source_key] = source_score_val
            if normalized_scores:
                base_entry["score_by_source"] = normalized_scores
        raw_atom_ids = ann.get("prepass_atom_ids")
        if isinstance(raw_atom_ids, (list, tuple)):
            atom_ids: List[str] = []
            for raw_atom_id in raw_atom_ids:
                atom_id = str(raw_atom_id or "").strip()
                if atom_id and atom_id not in atom_ids:
                    atom_ids.append(atom_id)
            if atom_ids:
                base_entry["prepass_atom_ids"] = atom_ids

        if pil_img is not None and isinstance(classifier_head, dict):
            classes = _clip_head_classes(classifier_head)
            target_idx = _find_clip_head_target_index(classes, aligned)
            if target_idx is None:
                rejected += 1
                continue
            source = str(score_source or ann.get("source") or "").strip().lower()
            strict_sources = {
                "sam3_similarity",
                "qwen_inspect",
                "qwen_infer",
                "yolo_zoom",
                "rfdetr_zoom",
            }
            strict = source in strict_sources
            if not strict and source == "sam3_text":
                if score is None or score < PREPASS_STRICT_SAM3_MIN_SCORE:
                    strict = True
            min_prob = float(classifier_head.get("min_prob") or 0.5)
            margin = float(classifier_head.get("margin") or 0.0)
            background_margin = float(classifier_head.get("background_margin") or 0.0)
            if strict:
                min_prob = max(min_prob, PREPASS_CLASSIFIER_STRICT_MIN_PROB)
                margin = max(margin, PREPASS_CLASSIFIER_STRICT_MARGIN)
                background_margin = max(background_margin, PREPASS_CLASSIFIER_STRICT_BG_MARGIN)
            pending.append(
                {
                    "entry": base_entry,
                    "target_idx": target_idx,
                    "min_prob": min_prob,
                    "margin": margin,
                    "background_margin": background_margin,
                }
            )
            crop, crop_record = _classifier_crop_for_head(pil_img, (x1, y1, x2, y2), classifier_head)
            pending_crops.append(crop)
            pending_crop_records.append(crop_record)
        else:
            cleaned.append(base_entry)

    if pending and pil_img is not None and isinstance(classifier_head, dict):
        empty_cache_fn = torch.cuda.empty_cache if torch.cuda.is_available() else None
        try:
            proba_arr = _predict_proba_batched_impl(
                pending_crops,
                classifier_head,
                batch_size=_resolve_classifier_batch(),
                encode_batch_fn=lambda items, head_obj, bs: _encode_embedding_items_for_head(
                    items, head=head_obj, batch_size_override=bs, geometry_records=pending_crop_records
                ),
                predict_proba_fn=_clip_head_predict_proba,
                empty_cache_fn=empty_cache_fn,
            )
        finally:
            for crop in pending_crops:
                _close_crop_item(crop)
        if proba_arr is None or proba_arr.ndim != 2 or proba_arr.shape[0] != len(pending):
            rejected += len(pending)
            return cleaned, rejected
        classes = _clip_head_classes(classifier_head)
        bg_indices = _clip_head_background_indices(classes)
        for row, pending_entry in zip(proba_arr, pending, strict=False):
            entry = pending_entry["entry"]
            target_idx = pending_entry["target_idx"]
            min_prob = pending_entry["min_prob"]
            margin = pending_entry["margin"]
            background_margin = pending_entry["background_margin"]
            order = sorted(range(len(classes)), key=lambda idx: float(row[idx]), reverse=True)
            best_idx = order[0] if order else None
            best_label = classes[best_idx] if best_idx is not None else None
            best_prob = float(row[best_idx]) if best_idx is not None else None
            keep_mask = _clip_head_keep_mask(
                row.reshape(1, -1),
                target_index=target_idx,
                min_prob=min_prob,
                margin=margin,
                background_indices=bg_indices,
                background_guard=True,
                background_margin=background_margin,
            )
            entry["classifier_best"] = best_label
            entry["classifier_prob"] = best_prob
            entry["classifier_accept"] = (
                bool(keep_mask[0]) if keep_mask is not None and len(keep_mask) else False
            )
            if keep_mask is None or not bool(keep_mask[0]):
                rejected += 1
                continue
            cleaned.append(entry)
    return cleaned, rejected


def _agent_register_detections(
    detections: Sequence[Dict[str, Any]],
    *,
    img_w: int,
    img_h: int,
    grid: Optional[Mapping[str, Any]] = None,
    labelmap: Optional[Sequence[str]] = None,
    background: Optional[Sequence[str]] = None,
    source_override: Optional[str] = None,
    owner_cell: Optional[str] = None,
    iou_thr: Optional[float] = None,
) -> Dict[str, Any]:
    global _AGENT_ACTIVE_CANDIDATES, _AGENT_ACTIVE_CANDIDATE_INDEX
    global _AGENT_ACTIVE_CLUSTERS, _AGENT_ACTIVE_CLUSTER_INDEX
    global _AGENT_NEXT_CANDIDATE_ID, _AGENT_NEXT_CLUSTER_ID

    if not detections:
        return {
            "candidate_ids": [],
            "cluster_ids": [],
            "new_clusters": 0,
            "updated_clusters": 0,
            "rejected": 0,
        }
    labelmap = list(labelmap or _AGENT_ACTIVE_LABELMAP or [])
    cleaned, rejected = _agent_sanitize_detection_items(
        list(detections),
        pil_img=None,
        classifier_head=None,
        img_w=img_w,
        img_h=img_h,
        labelmap=labelmap,
        background=background,
    )
    if _AGENT_ACTIVE_CANDIDATES is None:
        _AGENT_ACTIVE_CANDIDATES = []
    if _AGENT_ACTIVE_CLUSTERS is None:
        _AGENT_ACTIVE_CLUSTERS = []
    if iou_thr is None:
        iou_thr = PREPASS_CLUSTER_IOU
    new_cluster_ids: List[int] = []
    updated_cluster_ids: List[int] = []
    candidate_ids: List[int] = []
    for det in cleaned:
        candidate_id = _AGENT_NEXT_CANDIDATE_ID
        _AGENT_NEXT_CANDIDATE_ID += 1
        candidate_ids.append(candidate_id)
        source = source_override or det.get("source") or det.get("score_source") or "agent"
        source_primary = det.get("source_primary") or source
        source_prompt = det.get("source_prompt")
        source_exemplar_handles = det.get("source_exemplar_handles")
        source_detector_run_id = det.get("source_detector_run_id")
        source_list = set(det.get("source_list") or [])
        if source:
            source_list.add(str(source))
        cell = _agent_grid_cell_for_detection(det, img_w, img_h, grid)
        owner = owner_cell or cell
        candidate = {
            "candidate_id": candidate_id,
            "label": det.get("label"),
            "class_id": det.get("class_id"),
            "bbox_xyxy_px": det.get("bbox_xyxy_px"),
            "bbox_2d": det.get("bbox_2d"),
            "bbox_yolo": det.get("bbox_yolo"),
            "score": det.get("score"),
            "score_source": det.get("score_source") or det.get("source"),
            "source": source,
            "source_primary": source_primary,
            "source_prompt": source_prompt,
            "source_exemplar_handles": source_exemplar_handles,
            "source_detector_run_id": source_detector_run_id,
            "source_list": sorted(source_list) if source_list else None,
            "grid_cell": cell,
            "owner_cell": owner,
            "window_bbox_2d": det.get("window_bbox_2d"),
        }
        _AGENT_ACTIVE_CANDIDATES.append(candidate)
        _AGENT_ACTIVE_CANDIDATE_INDEX[candidate_id] = candidate
        cluster = _agent_cluster_match(det, _AGENT_ACTIVE_CLUSTERS, iou_thr=iou_thr)
        if cluster is None:
            cluster_id = _AGENT_NEXT_CLUSTER_ID
            _AGENT_NEXT_CLUSTER_ID += 1
            origin_tag = "prepass"
            cluster_entry = {
                "cluster_id": cluster_id,
                "label": det.get("label"),
                "class_id": det.get("class_id"),
                "bbox_xyxy_px": det.get("bbox_xyxy_px"),
                "bbox_2d": det.get("bbox_2d"),
                "bbox_yolo": det.get("bbox_yolo"),
                "score": det.get("score"),
                "score_source": det.get("score_source") or det.get("source"),
                "source": source,
                "source_primary": source_primary,
                "source_prompt": source_prompt,
                "source_exemplar_handles": source_exemplar_handles,
                "source_detector_run_id": source_detector_run_id,
                "source_list": sorted(source_list) if source_list else None,
                "origin": origin_tag,
                "candidate_ids": [candidate_id],
                "grid_cell": cell,
                "owner_cell": owner,
                "classifier_best": det.get("classifier_best"),
                "classifier_prob": det.get("classifier_prob"),
                "classifier_accept": det.get("classifier_accept"),
            }
            _AGENT_ACTIVE_CLUSTERS.append(cluster_entry)
            _AGENT_ACTIVE_CLUSTER_INDEX[cluster_id] = cluster_entry
            new_cluster_ids.append(cluster_id)
            candidate["cluster_id"] = cluster_id
            continue
        cluster_id = int(cluster.get("cluster_id"))
        candidate["cluster_id"] = cluster_id
        if owner and not cluster.get("owner_cell"):
            cluster["owner_cell"] = owner
        cluster.setdefault("candidate_ids", [])
        cluster["candidate_ids"] = list(cluster.get("candidate_ids") or []) + [candidate_id]
        cluster_sources = set(cluster.get("source_list") or [])
        if cluster.get("source"):
            cluster_sources.add(str(cluster.get("source")))
        cluster_sources.update(source_list)
        if cluster_sources:
            cluster["source_list"] = sorted(cluster_sources)
        cluster_score = _agent_det_score(cluster)
        det_score = _agent_det_score(det)
        replace = False
        if det_score is not None and (cluster_score is None or det_score > cluster_score):
            replace = True
        elif det_score is None and cluster_score is None and not cluster.get("bbox_xyxy_px"):
            replace = True
        if replace:
            keep_ids = list(cluster.get("candidate_ids") or [])
            keep_sources = list(cluster.get("source_list") or [])
            keep_classifier = {
                "classifier_best": cluster.get("classifier_best"),
                "classifier_prob": cluster.get("classifier_prob"),
                "classifier_accept": cluster.get("classifier_accept"),
            }
            cluster.update(
                {
                    "label": det.get("label"),
                    "class_id": det.get("class_id"),
                    "bbox_xyxy_px": det.get("bbox_xyxy_px"),
                    "bbox_2d": det.get("bbox_2d"),
                    "bbox_yolo": det.get("bbox_yolo"),
                    "score": det.get("score"),
                    "score_source": det.get("score_source") or det.get("source"),
                    "source": source,
                    "source_primary": source_primary,
                    "source_prompt": source_prompt,
                    "source_exemplar_handles": source_exemplar_handles,
                    "source_detector_run_id": source_detector_run_id,
                    "grid_cell": cell,
                }
            )
            cluster["candidate_ids"] = keep_ids
            if keep_sources:
                cluster["source_list"] = keep_sources
            for key, value in keep_classifier.items():
                if value is not None and cluster.get(key) is None:
                    cluster[key] = value
        else:
            if source_primary and cluster.get("source_primary") is None:
                cluster["source_primary"] = source_primary
            if source_prompt and cluster.get("source_prompt") is None:
                cluster["source_prompt"] = source_prompt
            if source_exemplar_handles and cluster.get("source_exemplar_handles") is None:
                cluster["source_exemplar_handles"] = list(source_exemplar_handles)
            if source_detector_run_id and cluster.get("source_detector_run_id") is None:
                cluster["source_detector_run_id"] = source_detector_run_id
        updated_cluster_ids.append(cluster_id)
    _agent_set_active_clusters(_AGENT_ACTIVE_CLUSTERS)
    return {
        "candidate_ids": candidate_ids,
        "cluster_ids": new_cluster_ids
        + [cid for cid in updated_cluster_ids if cid not in new_cluster_ids],
        "new_cluster_ids": new_cluster_ids,
        "updated_cluster_ids": updated_cluster_ids,
        "new_clusters": len(new_cluster_ids),
        "updated_clusters": len(updated_cluster_ids),
        "rejected": int(rejected),
    }


def bind_gate(bindings):
    namespace = dict(globals())
    namespace.update(bindings)
    for name in ('_clip_head_keep_mask', '_classifier_crop_for_head', '_agent_sanitize_detection_items', '_agent_register_detections'):
        original = globals()[name]
        namespace[name] = FunctionType(original.__code__, namespace, name, original.__defaults__, original.__closure__)
        namespace[name].__kwdefaults__ = original.__kwdefaults__
    return namespace['_agent_sanitize_detection_items']


def register_clusters(detections, *, width, height, labelmap, gate, grid):
    namespace = dict(globals())
    namespace.update(_AGENT_ACTIVE_CANDIDATES=[], _AGENT_ACTIVE_CANDIDATE_INDEX={},
        _AGENT_ACTIVE_CLUSTERS=[], _AGENT_ACTIVE_CLUSTER_INDEX={}, _AGENT_ACTIVE_LABELMAP=labelmap,
        _AGENT_NEXT_CANDIDATE_ID=1, _AGENT_NEXT_CLUSTER_ID=1, PREPASS_CLUSTER_IOU=.75,
        _agent_sanitize_detection_items=gate, _agent_set_active_clusters=lambda value: None)
    original = _agent_register_detections
    register = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    register.__kwdefaults__ = original.__kwdefaults__
    register(detections, img_w=width, img_h=height, labelmap=labelmap, grid=grid, background=[])
    return namespace['_AGENT_ACTIVE_CLUSTERS']
