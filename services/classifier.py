from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _predict_proba_batched_impl(
    crops: Sequence[Any],
    head: Dict[str, Any],
    *,
    batch_size: int,
    encode_batch_fn: Callable[[Sequence[Any], Dict[str, Any], int], Any],
    predict_proba_fn: Callable[[Any, Dict[str, Any]], Any],
    empty_cache_fn: Optional[Callable[[], None]] = None,
) -> Optional[Any]:
    feats = encode_batch_fn(crops, head, batch_size)
    if feats is None:
        return None
    try:
        return predict_proba_fn(feats, head, empty_cache_fn=empty_cache_fn)  # type: ignore[call-arg]
    except TypeError:
        return predict_proba_fn(feats, head)


def _agent_classifier_review_impl(
    detections: List[Dict[str, Any]],
    *,
    pil_img: Optional[Any],
    classifier_head: Optional[Dict[str, Any]],
    resolve_batch_size_fn: Callable[[], int],
    predict_proba_fn: Callable[[Sequence[Any], Dict[str, Any], int], Optional[Any]],
    clip_head_background_indices_fn: Callable[[Sequence[str]], List[int]],
    find_target_index_fn: Callable[[Sequence[str], str], Optional[int]],
    clip_head_keep_mask_fn: Callable[..., Any],
    readable_write_fn: Callable[[str], None],
    readable_format_bbox_fn: Callable[[Sequence[float]], str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    counts = {
        "classifier_checked": 0,
        "classifier_rejected": 0,
        "classifier_errors": 0,
        "classifier_unavailable": 0,
    }
    if not detections:
        return detections, counts
    if pil_img is None or not isinstance(classifier_head, dict):
        counts["classifier_unavailable"] = len(detections)
        for det in detections:
            det["classifier_accept"] = None
            det["classifier_error"] = "unavailable"
        return detections, counts
    classes = [str(c) for c in list(classifier_head.get("classes") or [])]
    bg_indices = clip_head_background_indices_fn(classes)
    min_prob = float(classifier_head.get("min_prob") or 0.5)
    margin = float(classifier_head.get("margin") or 0.0)
    background_margin = float(classifier_head.get("background_margin") or 0.0)
    accepted: List[Dict[str, Any]] = []

    pending: List[Tuple[Dict[str, Any], int, Sequence[float]]] = []
    crops: List[Any] = []
    for det in detections:
        bbox = det.get("bbox_xyxy_px")
        label = str(det.get("label") or "").strip()
        if not bbox or len(bbox) < 4:
            counts["classifier_errors"] += 1
            det["classifier_accept"] = False
            det["classifier_error"] = "missing_bbox"
            continue
        target_idx = find_target_index_fn(classes, label)
        if target_idx is None:
            counts["classifier_errors"] += 1
            det["classifier_accept"] = False
            det["classifier_error"] = "label_not_in_classifier"
            continue
        x1, y1, x2, y2 = bbox[:4]
        crop = pil_img.crop((x1, y1, x2, y2))
        pending.append((det, target_idx, bbox[:4]))
        crops.append(crop)

    if pending:
        batch_size = resolve_batch_size_fn()
        proba_arr = predict_proba_fn(crops, classifier_head, batch_size)
        if proba_arr is None or getattr(proba_arr, "ndim", None) != 2:
            for det, _target_idx, _bbox in pending:
                counts["classifier_errors"] += 1
                det["classifier_accept"] = False
                det["classifier_error"] = "predict_failed"
            return [], counts
        for row, (det, target_idx, bbox) in zip(proba_arr, pending):
            order = sorted(range(len(classes)), key=lambda idx: float(row[idx]), reverse=True)
            best_idx = order[0] if order else None
            best_label = classes[best_idx] if best_idx is not None else "unknown"
            best_prob = float(row[best_idx]) if best_idx is not None else None
            det["classifier_best"] = best_label
            det["classifier_prob"] = best_prob
            keep_mask = clip_head_keep_mask_fn(
                row.reshape(1, -1),
                target_index=target_idx,
                min_prob=min_prob,
                margin=margin,
                background_indices=bg_indices,
                background_guard=True,
                background_margin=background_margin,
            )
            accept = bool(keep_mask[0]) if keep_mask is not None and len(keep_mask) else False
            det["classifier_accept"] = accept
            counts["classifier_checked"] += 1
            summary_bbox = readable_format_bbox_fn(bbox)
            prob_text = f"{best_prob:.3f}" if isinstance(best_prob, float) else "n/a"
            readable_write_fn(
                f"classifier check label={det.get('label')} bbox={summary_bbox} "
                f"best={best_label} prob={prob_text} accept={'yes' if accept else 'no'}"
            )
            if accept:
                accepted.append(det)
            else:
                counts["classifier_rejected"] += 1
    return accepted, counts


def _resolve_agent_clip_classifier_path_impl(
    path_str: Optional[str],
    *,
    allowed_root: Path,
    allowed_exts: Sequence[str],
    path_is_within_root_fn: Callable[[Path, Path], bool],
    http_exception_cls: Any,
) -> Optional[Path]:
    if not path_str:
        return None
    raw = Path(str(path_str))
    candidate = Path(os.path.abspath(str(path_str))).resolve()
    if not path_is_within_root_fn(candidate, allowed_root):
        try:
            candidate_alt = (allowed_root / raw).resolve()
        except Exception:
            candidate_alt = None
        if candidate_alt is None or not path_is_within_root_fn(candidate_alt, allowed_root):
            raise http_exception_cls(status_code=400, detail="agent_clip_classifier_path_not_allowed")
        candidate = candidate_alt
    if candidate.suffix.lower() not in allowed_exts:
        raise http_exception_cls(status_code=400, detail="agent_clip_classifier_ext_not_allowed")
    if not candidate.exists() or not candidate.is_file():
        raise http_exception_cls(status_code=404, detail="agent_clip_classifier_not_found")
    return candidate


def _load_clip_head_from_classifier_impl(
    classifier_path: Path,
    *,
    joblib_load_fn: Callable[[str], Any],
    http_exception_cls: Any,
    clip_head_background_indices_fn: Callable[[Sequence[str]], List[int]],
    resolve_head_normalize_embeddings_fn: Callable[[Dict[str, Any], bool], bool],
    infer_clip_model_fn: Callable[[int, Optional[str]], Optional[str]],
    active_clip_model_name: Optional[str],
    default_clip_model: str,
    logger: Any,
) -> Optional[Dict[str, Any]]:
    try:
        clf_obj = joblib_load_fn(str(classifier_path))
    except Exception as exc:  # noqa: BLE001
        raise http_exception_cls(status_code=400, detail=f"agent_clip_classifier_load_failed:{exc}") from exc

    clip_model_used = None
    encoder_type_used = None
    encoder_model_used = None
    meta_found = False
    solver_used = None
    normalize_embeddings: Optional[bool] = None
    embedding_center_values: Optional[Sequence[float]] = None
    embedding_std_values: Optional[Sequence[float]] = None
    calibration_temperature: Optional[float] = None
    logit_adjustment: Optional[Sequence[float]] = None
    logit_adjustment_inference: bool = False
    arcface_enabled: bool = False
    arcface_margin: Optional[float] = None
    arcface_scale: Optional[float] = None
    meta_path = os.path.splitext(str(classifier_path))[0] + ".meta.pkl"
    if os.path.exists(meta_path):
        try:
            meta_obj = joblib_load_fn(meta_path)
            if isinstance(meta_obj, dict):
                meta_found = True
                clip_model_used = meta_obj.get("clip_model")
                encoder_type_used = meta_obj.get("encoder_type")
                encoder_model_used = meta_obj.get("encoder_model")
                solver_used = meta_obj.get("solver")
                if meta_obj.get("mlp_normalize_embeddings") is not None:
                    normalize_embeddings = bool(meta_obj.get("mlp_normalize_embeddings"))
                if meta_obj.get("embedding_center_values") is not None:
                    embedding_center_values = meta_obj.get("embedding_center_values")
                if meta_obj.get("embedding_std_values") is not None:
                    embedding_std_values = meta_obj.get("embedding_std_values")
                if meta_obj.get("calibration_temperature") is not None:
                    try:
                        calibration_temperature = float(meta_obj.get("calibration_temperature"))
                    except Exception:
                        calibration_temperature = None
                if meta_obj.get("logit_adjustment") is not None:
                    logit_adjustment = meta_obj.get("logit_adjustment")
                if meta_obj.get("logit_adjustment_inference") is not None:
                    try:
                        logit_adjustment_inference = bool(meta_obj.get("logit_adjustment_inference"))
                    except Exception:
                        logit_adjustment_inference = False
                if meta_obj.get("arcface_enabled") is not None:
                    arcface_enabled = bool(meta_obj.get("arcface_enabled"))
                if meta_obj.get("arcface_margin") is not None:
                    try:
                        arcface_margin = float(meta_obj.get("arcface_margin"))
                    except Exception:
                        arcface_margin = None
                if meta_obj.get("arcface_scale") is not None:
                    try:
                        arcface_scale = float(meta_obj.get("arcface_scale"))
                    except Exception:
                        arcface_scale = None
        except Exception:
            clip_model_used = None
            encoder_type_used = None
            encoder_model_used = None
            solver_used = None
            meta_found = False
    if not solver_used:
        try:
            raw_solver = getattr(clf_obj, "solver", None)
            if raw_solver is not None and str(raw_solver).strip():
                solver_used = str(raw_solver).strip()
        except Exception:
            solver_used = None

    if isinstance(clf_obj, dict) and str(clf_obj.get("classifier_type") or clf_obj.get("head_type") or "").lower() == "mlp":
        classes = [str(c) for c in list(clf_obj.get("classes") or [])]
        layers_raw = clf_obj.get("layers")
        if not classes or not isinstance(layers_raw, list) or not layers_raw:
            raise http_exception_cls(status_code=400, detail="agent_clip_classifier_invalid")
        if not meta_found:
            raise http_exception_cls(status_code=400, detail="agent_clip_classifier_meta_required")
        layers: List[Dict[str, Any]] = []
        embedding_dim = int(clf_obj.get("embedding_dim") or 0)
        total_layers = len(layers_raw)
        for idx, layer in enumerate(layers_raw):
            try:
                weight = np.asarray(layer.get("weight"), dtype=np.float32)
                bias = np.asarray(layer.get("bias"), dtype=np.float32).reshape(-1)
            except Exception as exc:  # noqa: BLE001
                raise http_exception_cls(status_code=400, detail=f"agent_clip_classifier_invalid:{exc}") from exc
            if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
                raise http_exception_cls(status_code=400, detail="agent_clip_classifier_invalid_shape")
            if embedding_dim <= 0:
                embedding_dim = int(weight.shape[1])
            activation = str(layer.get("activation") or "").strip().lower()
            if not activation:
                activation = "linear" if idx == total_layers - 1 else "relu"
            if activation not in {"relu", "linear", "none", "identity"}:
                activation = "relu" if idx < total_layers - 1 else "linear"
            layer_entry: Dict[str, Any] = {
                "weight": weight,
                "bias": bias,
                "activation": activation,
            }
            if layer.get("layer_norm_weight") is not None:
                layer_entry["layer_norm_weight"] = np.asarray(layer.get("layer_norm_weight"), dtype=np.float32)
                if layer.get("layer_norm_bias") is not None:
                    layer_entry["layer_norm_bias"] = np.asarray(layer.get("layer_norm_bias"), dtype=np.float32)
                if layer.get("layer_norm_eps") is not None:
                    try:
                        layer_entry["layer_norm_eps"] = float(layer.get("layer_norm_eps"))
                    except Exception:
                        pass
            layers.append(layer_entry)
        if embedding_dim <= 0:
            raise http_exception_cls(status_code=400, detail="agent_clip_classifier_invalid_shape")
        bg_indices = clip_head_background_indices_fn(classes)
        bg_classes = [classes[idx] for idx in bg_indices] if bg_indices else []
        if not isinstance(encoder_type_used, str) or not encoder_type_used.strip():
            encoder_type_used = "clip"
        if not isinstance(encoder_model_used, str) or not encoder_model_used.strip():
            encoder_model_used = clip_model_used
        normalize_flag = normalize_embeddings if normalize_embeddings is not None else resolve_head_normalize_embeddings_fn(clf_obj, True)
        return {
            "classes": classes,
            "background_indices": bg_indices,
            "background_classes": bg_classes,
            "layers": layers,
            "clip_model": str(clip_model_used) if clip_model_used else None,
            "encoder_type": str(encoder_type_used),
            "encoder_model": str(encoder_model_used) if encoder_model_used else None,
            "embedding_dim": int(embedding_dim),
            "proba_mode": "softmax",
            "classifier_type": "mlp",
            "normalize_embeddings": bool(normalize_flag),
            "embedding_center_values": embedding_center_values,
            "embedding_std_values": embedding_std_values,
            "temperature": calibration_temperature,
            "logit_adjustment": logit_adjustment,
            "logit_adjustment_inference": bool(logit_adjustment_inference),
            "arcface": bool(arcface_enabled),
            "arcface_margin": arcface_margin,
            "arcface_scale": arcface_scale,
        }

    classes_raw = getattr(clf_obj, "classes_", None)
    coef_raw = getattr(clf_obj, "coef_", None)
    intercept_raw = getattr(clf_obj, "intercept_", None)
    if classes_raw is None or coef_raw is None or intercept_raw is None:
        raise http_exception_cls(status_code=400, detail="agent_clip_classifier_invalid")
    try:
        classes = [str(c) for c in list(classes_raw)]
        coef = np.asarray(coef_raw, dtype=np.float32)
        intercept = np.asarray(intercept_raw, dtype=np.float32).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        raise http_exception_cls(status_code=400, detail=f"agent_clip_classifier_invalid:{exc}") from exc
    if coef.ndim != 2 or intercept.ndim != 1:
        raise http_exception_cls(status_code=400, detail="agent_clip_classifier_invalid_shape")
    if coef.shape[0] != intercept.shape[0]:
        if not (coef.shape[0] == 1 and intercept.shape[0] == 1 and len(classes) == 2):
            raise http_exception_cls(status_code=400, detail="agent_clip_classifier_invalid_shape")

    embedding_dim = 0
    try:
        embedding_dim = int(coef.shape[1]) if hasattr(coef, "shape") else 0
    except Exception:
        embedding_dim = 0
    if not meta_found:
        if embedding_dim and int(embedding_dim) not in {512, 768}:
            raise http_exception_cls(status_code=400, detail="agent_clip_classifier_meta_required")
        inferred = infer_clip_model_fn(embedding_dim, active_clip_model_name or default_clip_model)
        if inferred:
            clip_model_used = inferred
    if clip_model_used:
        try:
            expected_dim = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768}.get(str(clip_model_used))
            if expected_dim and embedding_dim and int(expected_dim) != int(embedding_dim):
                inferred = infer_clip_model_fn(embedding_dim, active_clip_model_name or default_clip_model)
                if inferred and inferred != clip_model_used:
                    logger.warning(
                        "CLIP head %s embedding dim %s mismatches clip_model=%s; using inferred %s instead.",
                        classifier_path.name,
                        embedding_dim,
                        clip_model_used,
                        inferred,
                    )
                    clip_model_used = inferred
                else:
                    logger.warning(
                        "CLIP head %s embedding dim %s mismatches clip_model=%s (expected %s); head probabilities may be unavailable.",
                        classifier_path.name,
                        embedding_dim,
                        clip_model_used,
                        expected_dim,
                    )
        except Exception:
            pass
    if not isinstance(encoder_type_used, str) or not encoder_type_used.strip():
        encoder_type_used = "clip"
    if not isinstance(encoder_model_used, str) or not encoder_model_used.strip():
        encoder_model_used = clip_model_used
    multi_class_used = None
    try:
        raw_multi = getattr(clf_obj, "multi_class", None)
        if raw_multi is not None and str(raw_multi).strip():
            multi_class_used = str(raw_multi).strip()
    except Exception:
        multi_class_used = None

    n_classes = len(classes)
    if n_classes == 2 and coef.shape[0] == 1:
        proba_mode = "binary"
    elif (solver_used and str(solver_used).strip().lower() == "liblinear") or (
        multi_class_used and str(multi_class_used).strip().lower() == "ovr"
    ):
        proba_mode = "ovr"
    else:
        proba_mode = "softmax"

    bg_indices = clip_head_background_indices_fn(classes)
    bg_classes = [classes[idx] for idx in bg_indices] if bg_indices else []
    if normalize_embeddings is None:
        normalize_embeddings = True

    return {
        "classes": classes,
        "background_indices": bg_indices,
        "background_classes": bg_classes,
        "coef": coef,
        "intercept": intercept,
        "clip_model": str(clip_model_used) if clip_model_used else None,
        "encoder_type": str(encoder_type_used),
        "encoder_model": str(encoder_model_used) if encoder_model_used else None,
        "embedding_dim": int(embedding_dim),
        "proba_mode": proba_mode,
        "classifier_type": "logreg",
        "normalize_embeddings": bool(normalize_embeddings),
        "embedding_center_values": embedding_center_values,
        "embedding_std_values": embedding_std_values,
        "temperature": calibration_temperature,
        "logit_adjustment": logit_adjustment,
        "logit_adjustment_inference": bool(logit_adjustment_inference),
    }


def _clip_head_predict_proba_impl(feats: np.ndarray, head: Dict[str, Any]) -> Optional[np.ndarray]:
    """Compute predict_proba(feats) for an exported classifier head (logreg or MLP)."""
    if feats is None:
        return None
    classes = head.get("classes") or []
    if not isinstance(classes, list) or not classes:
        return None
    try:
        X = np.asarray(feats, dtype=np.float32)
    except Exception:
        return None
    if X.ndim != 2:
        return None
    temperature = head.get("temperature")
    try:
        temperature_val = float(temperature) if temperature is not None else None
    except Exception:
        temperature_val = None
    if temperature_val is not None and temperature_val <= 0:
        temperature_val = None
    if isinstance(head.get("classifier_type"), str) and head.get("classifier_type") == "mlp" or head.get("layers"):
        layers = head.get("layers")
        if not isinstance(layers, list) or not layers:
            return None
        arcface_enabled = bool(head.get("arcface"))
        try:
            arcface_scale = float(head.get("arcface_scale") or 1.0)
        except Exception:
            arcface_scale = 1.0
        out = X
        total_layers = len(layers)
        last_weight = None
        for idx, layer in enumerate(layers):
            try:
                W = np.asarray(layer.get("weight"), dtype=np.float32)
                b = np.asarray(layer.get("bias"), dtype=np.float32).reshape(-1)
            except Exception:
                return None
            if W.ndim != 2 or b.ndim != 1 or W.shape[0] != b.shape[0] or out.shape[1] != W.shape[1]:
                return None
            is_last = idx == total_layers - 1
            if arcface_enabled and is_last:
                last_weight = W
                break
            out = out @ W.T + b.reshape(1, -1)
            ln_weight = layer.get("layer_norm_weight")
            ln_bias = layer.get("layer_norm_bias")
            if ln_weight is not None:
                try:
                    gamma = np.asarray(ln_weight, dtype=np.float32).reshape(1, -1)
                    beta = np.asarray(ln_bias, dtype=np.float32).reshape(1, -1) if ln_bias is not None else 0.0
                    eps = float(layer.get("layer_norm_eps") or 1e-5)
                    mean = np.mean(out, axis=1, keepdims=True)
                    var = np.mean((out - mean) ** 2, axis=1, keepdims=True)
                    out = (out - mean) / np.sqrt(var + eps)
                    out = out * gamma + beta
                except Exception:
                    pass
            activation = str(layer.get("activation") or "").strip().lower()
            if not activation:
                activation = "linear" if is_last else "relu"
            if activation == "relu":
                out = np.maximum(out, 0.0)
            elif activation == "gelu":
                out = 0.5 * out * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (out + 0.044715 * (out ** 3))))
            elif activation in {"linear", "none", "identity"}:
                pass
            else:
                out = np.maximum(out, 0.0)
        if arcface_enabled:
            if last_weight is None:
                return None
            if out.shape[1] != last_weight.shape[1]:
                return None
            feats = out
            feat_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            weight_norm = last_weight / (np.linalg.norm(last_weight, axis=1, keepdims=True) + 1e-8)
            out = feat_norm @ weight_norm.T
            try:
                arcface_scale = float(head.get("arcface_scale") or arcface_scale)
            except Exception:
                arcface_scale = arcface_scale
            out = out * arcface_scale
        if temperature_val is not None:
            out = out / temperature_val
        if head.get("logit_adjustment_inference") and head.get("logit_adjustment") is not None:
            try:
                adjustment = np.asarray(head.get("logit_adjustment"), dtype=np.float32).reshape(1, -1)
                if adjustment.shape[1] == out.shape[1]:
                    out = out + adjustment
            except Exception:
                pass
        exp = np.exp(out - out.max(axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    coef = np.asarray(head.get("coef"), dtype=np.float32)
    intercept = np.asarray(head.get("intercept"), dtype=np.float32).reshape(-1)
    if coef.ndim != 2 or intercept.ndim != 1:
        return None
    if coef.shape[0] != intercept.shape[0]:
        if not (coef.shape[0] == 1 and intercept.shape[0] == 1 and len(classes) == 2):
            return None
    logits = X @ coef.T + intercept.reshape(1, -1)
    if temperature_val is not None:
        logits = logits / temperature_val
    if head.get("logit_adjustment_inference") and head.get("logit_adjustment") is not None:
        try:
            adjustment = np.asarray(head.get("logit_adjustment"), dtype=np.float32).reshape(1, -1)
            if adjustment.shape[1] == logits.shape[1]:
                logits = logits + adjustment
        except Exception:
            pass
    if logits.shape[1] == 1:
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.concatenate([1.0 - probs, probs], axis=1)
    else:
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
    return probs


def _clip_head_keep_mask_impl(
    proba: np.ndarray,
    *,
    target_index: int,
    min_prob: float,
    margin: float,
    background_indices: Optional[Sequence[int]] = None,
    background_guard: bool = False,
    background_margin: float = 0.0,
) -> Optional[np.ndarray]:
    """Return boolean keep mask for rows in proba."""
    try:
        probs = np.asarray(proba, dtype=np.float32)
    except Exception:
        return None
    if probs.ndim != 2 or probs.shape[0] == 0:
        return None
    if target_index < 0 or target_index >= probs.shape[1]:
        return None
    try:
        min_prob_f = float(min_prob)
    except Exception:
        min_prob_f = 0.0
    try:
        margin_f = float(margin)
    except Exception:
        margin_f = 0.0

    p_target = probs[:, target_index]
    keep = p_target >= min_prob_f

    try:
        bg_margin_f = float(background_margin)
    except Exception:
        bg_margin_f = 0.0

    if background_guard:
        bg_indices: List[int] = []
        if background_indices:
            for idx in background_indices:
                if isinstance(idx, int) and 0 <= idx < probs.shape[1] and idx != target_index:
                    bg_indices.append(idx)
        if bg_indices:
            p_bg = np.max(probs[:, bg_indices], axis=1)
            keep &= p_target >= (p_bg + max(0.0, bg_margin_f))

    # "Margin" is optional: a value of 0 disables the margin check (i.e., do not require the target
    # class to be the argmax). When enabled, require p(target) >= p(best_other) + margin.
    if margin_f > 0.0:
        if probs.shape[1] > 1:
            masked = probs.copy()
            masked[:, target_index] = -1.0
            p_other = np.max(masked, axis=1)
        else:
            p_other = np.zeros_like(p_target)
        keep &= p_target >= (p_other + margin_f)
    return keep
