from __future__ import annotations

import json
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


def _resolve_head_normalize_embeddings_impl(head: Optional[Dict[str, Any]], *, default: bool = True) -> bool:
    if not head:
        return default
    raw = head.get("normalize_embeddings")
    if raw is None:
        raw = head.get("mlp_normalize_embeddings")
    if raw is None:
        return default
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _resolve_active_head_normalize_embeddings_impl(
    meta_obj: Optional[Dict[str, Any]],
    clf_obj: Optional[object],
    *,
    default: bool = True,
    resolve_head_normalize_embeddings_fn: Callable[[Optional[Dict[str, Any]], bool], bool],
) -> bool:
    if isinstance(clf_obj, dict):
        head_type = str(clf_obj.get("classifier_type") or clf_obj.get("head_type") or "").lower()
        if head_type == "mlp":
            return resolve_head_normalize_embeddings_fn(clf_obj, default=default)
    if isinstance(meta_obj, dict) and str(meta_obj.get("classifier_type") or "").lower() == "mlp":
        raw = meta_obj.get("mlp_normalize_embeddings")
        if raw is not None:
            return bool(raw)
    return default


def _save_clip_head_artifacts_impl(
    *,
    recipe_dir: Path,
    head: Dict[str, Any],
    min_prob: float,
    margin: float,
    path_is_within_root_fn: Callable[[Path, Path], bool],
    http_exception_cls: Any,
    resolve_head_normalize_embeddings_fn: Callable[[Optional[Dict[str, Any]], bool], bool],
    max_bytes: int,
) -> None:
    """Persist a portable CLIP head artifact into a recipe package directory."""
    clip_dir = (recipe_dir / "clip_head").resolve()
    if not path_is_within_root_fn(clip_dir, recipe_dir.resolve()):
        raise http_exception_cls(status_code=400, detail="agent_recipe_clip_head_path_invalid")
    clip_dir.mkdir(parents=True, exist_ok=True)

    npz_path = clip_dir / "head.npz"
    meta_path = clip_dir / "meta.json"
    classifier_type = str(head.get("classifier_type") or "").strip().lower()
    if not classifier_type and head.get("layers"):
        classifier_type = "mlp"
    if classifier_type == "mlp":
        layers = head.get("layers")
        if not isinstance(layers, list) or not layers:
            raise http_exception_cls(status_code=400, detail="agent_recipe_clip_head_invalid")
        arrays: Dict[str, np.ndarray] = {}
        layers_meta: List[Dict[str, str]] = []
        total_layers = len(layers)
        normalize_flag = resolve_head_normalize_embeddings_fn(head, default=True)
        for idx, layer in enumerate(layers):
            try:
                weight = np.asarray(layer.get("weight"), dtype=np.float32)
                bias = np.asarray(layer.get("bias"), dtype=np.float32).reshape(-1)
            except Exception as exc:  # noqa: BLE001
                raise http_exception_cls(status_code=400, detail=f"agent_recipe_clip_head_invalid:{exc}") from exc
            if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
                raise http_exception_cls(status_code=400, detail="agent_recipe_clip_head_invalid")
            weight_key = f"layer{idx}_weight"
            bias_key = f"layer{idx}_bias"
            arrays[weight_key] = weight
            arrays[bias_key] = bias
            activation = str(layer.get("activation") or "").strip().lower()
            if not activation:
                activation = "linear" if idx == total_layers - 1 else "relu"
            if activation not in {"relu", "linear", "none", "identity"}:
                activation = "relu" if idx < total_layers - 1 else "linear"
            layers_meta.append({
                "weight": weight_key,
                "bias": bias_key,
                "activation": activation,
            })
        try:
            np.savez_compressed(str(npz_path), **arrays)
        except Exception as exc:  # noqa: BLE001
            raise http_exception_cls(status_code=500, detail=f"agent_recipe_clip_head_write_failed:{exc}") from exc
        classes = head.get("classes") if isinstance(head.get("classes"), list) else []
        encoder_type = head.get("encoder_type") if isinstance(head.get("encoder_type"), str) else "clip"
        encoder_model = head.get("encoder_model")
        if not isinstance(encoder_model, str) or not encoder_model.strip():
            encoder_model = head.get("clip_model")
        meta = {
            "clip_model": head.get("clip_model"),
            "encoder_type": encoder_type,
            "encoder_model": encoder_model,
            "proba_mode": "softmax",
            "classifier_type": "mlp",
            "classes": [str(c) for c in classes],
            "layers": layers_meta,
            "normalize_embeddings": bool(normalize_flag),
            "embedding_center_values": head.get("embedding_center_values").tolist()
            if isinstance(head.get("embedding_center_values"), np.ndarray)
            else head.get("embedding_center_values"),
            "embedding_std_values": head.get("embedding_std_values").tolist()
            if isinstance(head.get("embedding_std_values"), np.ndarray)
            else head.get("embedding_std_values"),
            "calibration_temperature": head.get("temperature")
            if head.get("temperature") is not None
            else head.get("calibration_temperature"),
            "logit_adjustment": head.get("logit_adjustment"),
            "logit_adjustment_inference": bool(head.get("logit_adjustment_inference")),
            "arcface": bool(head.get("arcface")),
            "arcface_margin": head.get("arcface_margin"),
            "arcface_scale": head.get("arcface_scale"),
            "min_prob": float(min_prob),
            "margin": float(margin),
        }
    else:
        try:
            coef = np.asarray(head.get("coef"), dtype=np.float32)
            intercept = np.asarray(head.get("intercept"), dtype=np.float32).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            raise http_exception_cls(status_code=400, detail=f"agent_recipe_clip_head_invalid:{exc}") from exc
        if coef.ndim != 2 or intercept.ndim != 1:
            raise http_exception_cls(status_code=400, detail="agent_recipe_clip_head_invalid")

        try:
            np.savez_compressed(str(npz_path), coef=coef, intercept=intercept)
        except Exception as exc:  # noqa: BLE001
            raise http_exception_cls(status_code=500, detail=f"agent_recipe_clip_head_write_failed:{exc}") from exc

        classes = head.get("classes") if isinstance(head.get("classes"), list) else []
        encoder_type = head.get("encoder_type") if isinstance(head.get("encoder_type"), str) else "clip"
        encoder_model = head.get("encoder_model")
        if not isinstance(encoder_model, str) or not encoder_model.strip():
            encoder_model = head.get("clip_model")
        normalize_flag = resolve_head_normalize_embeddings_fn(head, default=True)
        meta = {
            "clip_model": head.get("clip_model"),
            "encoder_type": encoder_type,
            "encoder_model": encoder_model,
            "proba_mode": head.get("proba_mode"),
            "classifier_type": "logreg",
            "classes": [str(c) for c in classes],
            "normalize_embeddings": bool(normalize_flag),
            "embedding_center_values": head.get("embedding_center_values").tolist()
            if isinstance(head.get("embedding_center_values"), np.ndarray)
            else head.get("embedding_center_values"),
            "embedding_std_values": head.get("embedding_std_values").tolist()
            if isinstance(head.get("embedding_std_values"), np.ndarray)
            else head.get("embedding_std_values"),
            "calibration_temperature": head.get("temperature")
            if head.get("temperature") is not None
            else head.get("calibration_temperature"),
            "logit_adjustment": head.get("logit_adjustment"),
            "logit_adjustment_inference": bool(head.get("logit_adjustment_inference")),
            "min_prob": float(min_prob),
            "margin": float(margin),
        }
    try:
        with meta_path.open("w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        raise http_exception_cls(status_code=500, detail=f"agent_recipe_clip_head_meta_write_failed:{exc}") from exc

    try:
        total = npz_path.stat().st_size + meta_path.stat().st_size
    except Exception:
        total = 0
    if total and max_bytes > 0 and total > max_bytes:
        raise http_exception_cls(status_code=413, detail="agent_recipe_clip_head_too_large")


def _load_clip_head_artifacts_impl(
    *,
    recipe_dir: Path,
    fallback_meta: Optional[Dict[str, Any]] = None,
    path_is_within_root_fn: Callable[[Path, Path], bool],
    clip_head_background_indices_fn: Callable[[Sequence[str]], List[int]],
    infer_clip_model_fn: Callable[[Optional[int], Optional[str]], Optional[str]],
    active_clip_model_name: Optional[str],
    default_clip_model: str,
) -> Optional[Dict[str, Any]]:
    """Load a portable CLIP head artifact from a recipe package directory."""
    clip_dir = (recipe_dir / "clip_head").resolve()
    npz_path = (clip_dir / "head.npz").resolve()
    meta_path = (clip_dir / "meta.json").resolve()
    if not path_is_within_root_fn(npz_path, clip_dir) or not path_is_within_root_fn(meta_path, clip_dir):
        return None
    if not npz_path.exists() or not npz_path.is_file():
        return None
    meta: Dict[str, Any] = {}
    if meta_path.exists() and meta_path.is_file():
        try:
            with meta_path.open("r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            if isinstance(loaded, dict):
                meta = loaded
        except Exception:
            meta = {}
    if not meta and isinstance(fallback_meta, dict):
        meta = fallback_meta
    classes_raw = meta.get("classes") if isinstance(meta.get("classes"), list) else []
    classes = [str(c) for c in classes_raw]
    proba_mode = meta.get("proba_mode")
    min_prob = meta.get("min_prob")
    margin = meta.get("margin")
    classifier_type = str(meta.get("classifier_type") or "").strip().lower()
    layers_meta = meta.get("layers") if isinstance(meta.get("layers"), list) else None
    normalize_flag = meta.get("normalize_embeddings")
    center_vals = meta.get("embedding_center_values")
    std_vals = meta.get("embedding_std_values")
    temperature_val = meta.get("calibration_temperature")
    logit_adjustment = meta.get("logit_adjustment")
    logit_adjustment_inference = meta.get("logit_adjustment_inference")
    arcface_flag = meta.get("arcface")
    if arcface_flag is None:
        arcface_flag = meta.get("arcface_enabled")
    arcface_margin = meta.get("arcface_margin")
    arcface_scale = meta.get("arcface_scale")

    min_prob_val: Optional[float] = None
    margin_val: Optional[float] = None
    try:
        if min_prob is not None:
            min_prob_val = float(min_prob)
    except Exception:
        min_prob_val = None
    try:
        if margin is not None:
            margin_val = float(margin)
    except Exception:
        margin_val = None

    clip_model_val: Optional[str] = None
    try:
        raw = meta.get("clip_model")
        if isinstance(raw, str) and raw.strip():
            clip_model_val = raw.strip()
    except Exception:
        clip_model_val = None

    encoder_type = meta.get("encoder_type") if isinstance(meta.get("encoder_type"), str) else "clip"
    encoder_model = meta.get("encoder_model")
    if not isinstance(encoder_model, str) or not encoder_model.strip():
        encoder_model = clip_model_val

    if classifier_type == "mlp" or layers_meta:
        try:
            with np.load(str(npz_path)) as data:
                layers: List[Dict[str, Any]] = []
                total_layers = len(layers_meta or [])
                for idx, layer in enumerate(layers_meta or []):
                    weight_key = layer.get("weight")
                    bias_key = layer.get("bias")
                    if not weight_key or not bias_key:
                        return None
                    weight = np.asarray(data.get(weight_key), dtype=np.float32)
                    bias = np.asarray(data.get(bias_key), dtype=np.float32).reshape(-1)
                    if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
                        return None
                    activation = str(layer.get("activation") or "").strip().lower()
                    if not activation:
                        activation = "linear" if idx == total_layers - 1 else "relu"
                    if activation not in {"relu", "linear", "none", "identity"}:
                        activation = "relu" if idx < total_layers - 1 else "linear"
                    layers.append({
                        "weight": weight,
                        "bias": bias,
                        "activation": activation,
                    })
        except Exception:
            return None
        bg_indices = clip_head_background_indices_fn(classes)
        bg_classes = [classes[idx] for idx in bg_indices] if bg_indices else []
        embedding_dim = 0
        try:
            embedding_dim = int(layers[0].get("weight").shape[1]) if layers else 0
        except Exception:
            embedding_dim = 0
        return {
            "classes": classes,
            "background_indices": bg_indices,
            "background_classes": bg_classes,
            "layers": layers,
            "clip_model": clip_model_val,
            "encoder_type": encoder_type,
            "encoder_model": encoder_model,
            "embedding_dim": embedding_dim,
            "proba_mode": "softmax",
            "classifier_type": "mlp",
            "normalize_embeddings": bool(normalize_flag) if normalize_flag is not None else True,
            "embedding_center_values": center_vals,
            "embedding_std_values": std_vals,
            "temperature": temperature_val,
            "logit_adjustment": logit_adjustment,
            "logit_adjustment_inference": bool(logit_adjustment_inference),
            "arcface": bool(arcface_flag),
            "arcface_margin": arcface_margin,
            "arcface_scale": arcface_scale,
            "min_prob": min_prob_val,
            "margin": margin_val,
        }

    try:
        with np.load(str(npz_path)) as data:
            coef = np.asarray(data["coef"], dtype=np.float32)
            intercept = np.asarray(data["intercept"], dtype=np.float32).reshape(-1)
    except Exception:
        return None

    if not isinstance(proba_mode, str) or not proba_mode:
        if coef.shape[0] == 1 and len(classes) == 2:
            proba_mode = "binary"
        else:
            proba_mode = "softmax"

    if not clip_model_val and coef.ndim == 2:
        emb_dim = 0
        try:
            emb_dim = int(coef.shape[1])
        except Exception:
            emb_dim = 0
        clip_model_val = infer_clip_model_fn(emb_dim, active_name=active_clip_model_name or default_clip_model)

    bg_indices = clip_head_background_indices_fn(classes)
    bg_classes = [classes[idx] for idx in bg_indices] if bg_indices else []

    return {
        "classes": classes,
        "background_indices": bg_indices,
        "background_classes": bg_classes,
        "coef": coef,
        "intercept": intercept,
        "clip_model": clip_model_val,
        "encoder_type": encoder_type,
        "encoder_model": encoder_model,
        "proba_mode": proba_mode,
        "classifier_type": "logreg",
        "normalize_embeddings": bool(normalize_flag) if normalize_flag is not None else True,
        "embedding_center_values": center_vals,
        "embedding_std_values": std_vals,
        "temperature": temperature_val,
        "logit_adjustment": logit_adjustment,
        "logit_adjustment_inference": bool(logit_adjustment_inference),
        "min_prob": min_prob_val,
        "margin": margin_val,
    }


def _resolve_clip_head_background_settings_impl(payload: Any) -> Tuple[bool, bool, float, str]:
    try:
        guard = bool(getattr(payload, "clip_head_background_guard", False))
    except Exception:
        guard = False
    try:
        apply_raw = str(getattr(payload, "clip_head_background_apply", "final") or "final").strip().lower()
    except Exception:
        apply_raw = "final"
    apply_mode = apply_raw if apply_raw in {"seed", "final", "both"} else "final"
    try:
        margin_val = float(getattr(payload, "clip_head_background_margin", 0.0) or 0.0)
    except Exception:
        margin_val = 0.0
    margin_val = max(0.0, min(1.0, margin_val))
    guard_seed = bool(guard and apply_mode in {"seed", "both"})
    guard_final = bool(guard and apply_mode in {"final", "both"})
    return guard_seed, guard_final, float(margin_val), apply_mode


def _infer_clip_model_from_embedding_dim_impl(
    embedding_dim: Optional[int], *, active_name: Optional[str] = None
) -> Optional[str]:
    try:
        emb = int(embedding_dim or 0)
    except Exception:
        return None
    if emb == 768:
        return "ViT-L/14"
    if emb == 512:
        active = str(active_name or "").strip()
        if active in {"ViT-B/32", "ViT-B/16"}:
            return active
        return "ViT-B/32"
    return None


def _clip_auto_predict_label_impl(
    feats_np: np.ndarray,
    *,
    clf_obj: Optional[Any],
    active_head: Optional[Dict[str, Any]],
    background_guard: bool = False,
    clip_head_predict_proba_fn: Callable[[np.ndarray, Dict[str, Any]], Optional[np.ndarray]],
    clip_head_background_indices_fn: Callable[[Sequence[str]], List[int]],
    is_background_class_name_fn: Callable[[str], bool],
) -> Tuple[str, Optional[float], Optional[str]]:
    """Return (label, probability, error) for auto-classification using the active CLIP head."""
    if clf_obj is None:
        return "unknown", None, "clip_unavailable"
    head = active_head if isinstance(active_head, dict) else None
    classes_raw = getattr(clf_obj, "classes_", None)
    if head is not None:
        classes = [str(c) for c in list(head.get("classes") or [])]
    elif classes_raw is None and isinstance(clf_obj, dict):
        classes = [str(c) for c in list(clf_obj.get("classes") or [])]
    else:
        classes = [str(c) for c in list(classes_raw)] if classes_raw is not None else []
    proba_arr: Optional[np.ndarray] = None
    if head is not None:
        proba_arr = clip_head_predict_proba_fn(feats_np, head)
    elif hasattr(clf_obj, "predict_proba"):
        try:
            proba = clf_obj.predict_proba(feats_np)
            proba_arr = np.asarray(proba, dtype=np.float32)
        except Exception:
            proba_arr = None
    elif isinstance(clf_obj, dict):
        proba_arr = clip_head_predict_proba_fn(feats_np, clf_obj)
    if proba_arr is not None and proba_arr.ndim == 2 and proba_arr.shape[0] >= 1 and proba_arr.shape[1] == len(classes):
        row = proba_arr[0]
        bg_indices = clip_head_background_indices_fn(classes)
        if bg_indices:
            non_bg_indices = [idx for idx in range(len(classes)) if idx not in bg_indices]
            if not non_bg_indices:
                return "unknown", float(np.max(row)) if row.size else None, "clip_background"
            best_non_bg = non_bg_indices[int(np.argmax(row[non_bg_indices]))]
            p_non_bg = float(row[best_non_bg])
            if background_guard:
                p_bg = float(np.max(row[bg_indices])) if bg_indices else -1.0
                if p_bg >= p_non_bg:
                    return "unknown", p_bg, "clip_background"
            return str(classes[best_non_bg]), p_non_bg, None
        best_idx = int(np.argmax(row))
        return str(classes[best_idx]), float(row[best_idx]), None
    try:
        pred_cls = clf_obj.predict(feats_np)[0]
    except Exception as exc:  # noqa: BLE001
        return "unknown", None, f"classifier_error:{exc}"
    label = str(pred_cls)
    if is_background_class_name_fn(label):
        return "unknown", None, "clip_background"
    return label, None, None


def _clip_auto_predict_details_impl(
    feats_np: np.ndarray,
    *,
    clf_obj: Optional[Any],
    active_head: Optional[Dict[str, Any]],
    background_guard: bool = False,
    clip_head_predict_proba_fn: Callable[[np.ndarray, Dict[str, Any]], Optional[np.ndarray]],
    clip_head_background_indices_fn: Callable[[Sequence[str]], List[int]],
) -> Dict[str, Optional[object]]:
    if clf_obj is None:
        return {
            "label": "unknown",
            "proba": None,
            "second_label": None,
            "second_proba": None,
            "margin": None,
            "error": "clip_unavailable",
        }
    head = active_head if isinstance(active_head, dict) else None
    classes_raw = getattr(clf_obj, "classes_", None)
    if head is not None:
        classes = [str(c) for c in list(head.get("classes") or [])]
    elif classes_raw is None and isinstance(clf_obj, dict):
        classes = [str(c) for c in list(clf_obj.get("classes") or [])]
    else:
        classes = [str(c) for c in list(classes_raw)] if classes_raw is not None else []
    proba_arr: Optional[np.ndarray] = None
    if head is not None:
        proba_arr = clip_head_predict_proba_fn(feats_np, head)
    elif hasattr(clf_obj, "predict_proba"):
        try:
            proba = clf_obj.predict_proba(feats_np)
            proba_arr = np.asarray(proba, dtype=np.float32)
        except Exception:
            proba_arr = None
    elif isinstance(clf_obj, dict):
        proba_arr = clip_head_predict_proba_fn(feats_np, clf_obj)
    if proba_arr is not None and proba_arr.ndim == 2 and proba_arr.shape[0] >= 1 and proba_arr.shape[1] == len(classes):
        row = proba_arr[0]
        bg_indices = clip_head_background_indices_fn(classes)

        def _best_two(indices: Sequence[int]) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[float]]:
            if not indices:
                return None, None, None, None
            ordered = sorted(indices, key=lambda i: float(row[i]), reverse=True)
            best_idx = ordered[0]
            second_idx = ordered[1] if len(ordered) > 1 else None
            best_val = float(row[best_idx])
            second_val = float(row[second_idx]) if second_idx is not None else None
            return best_idx, best_val, second_idx, second_val

        if bg_indices:
            non_bg = [idx for idx in range(len(classes)) if idx not in bg_indices]
            best_idx, best_val, second_idx, second_val = _best_two(non_bg)
            if best_idx is None:
                return {
                    "label": "unknown",
                    "proba": float(np.max(row)) if row.size else None,
                    "second_label": None,
                    "second_proba": None,
                    "margin": None,
                    "error": "clip_background",
                }
            if background_guard:
                p_bg = float(np.max(row[bg_indices])) if bg_indices else -1.0
                if p_bg >= best_val:
                    return {
                        "label": "unknown",
                        "proba": p_bg,
                        "second_label": str(classes[best_idx]),
                        "second_proba": best_val,
                        "margin": float(p_bg - best_val),
                        "error": "clip_background",
                    }
            margin = float(best_val - second_val) if second_val is not None else None
            return {
                "label": str(classes[best_idx]),
                "proba": best_val,
                "second_label": str(classes[second_idx]) if second_idx is not None else None,
                "second_proba": second_val,
                "margin": margin,
                "error": None,
            }

        best_idx, best_val, second_idx, second_val = _best_two(list(range(len(classes))))
        margin = float(best_val - second_val) if best_val is not None and second_val is not None else None
        return {
            "label": str(classes[best_idx]) if best_idx is not None else "unknown",
            "proba": best_val,
            "second_label": str(classes[second_idx]) if second_idx is not None else None,
            "second_proba": second_val,
            "margin": margin,
            "error": None,
        }

    try:
        pred = clf_obj.predict(feats_np)
        label = str(pred[0]) if pred is not None and len(pred) else "unknown"
    except Exception as exc:  # noqa: BLE001
        return {
            "label": "unknown",
            "proba": None,
            "second_label": None,
            "second_proba": None,
            "margin": None,
            "error": f"classifier_error:{exc}",
        }
    return {
        "label": label,
        "proba": None,
        "second_label": None,
        "second_proba": None,
        "margin": None,
        "error": None,
    }


def _score_detections_with_clip_head_impl(
    dets: Sequence[Any],
    *,
    pil_img: Any,
    clip_head: Dict[str, Any],
    score_mode: str,
    bbox_to_xyxy_pixels_fn: Callable[[Sequence[float], int, int], Optional[Tuple[float, float, float, float]]],
    encode_pil_batch_for_head_fn: Callable[[Sequence[Any], Dict[str, Any]], Optional[np.ndarray]],
    clip_head_predict_proba_fn: Callable[[np.ndarray, Dict[str, Any]], Optional[np.ndarray]],
    clip_head_background_indices_fn: Callable[[Sequence[str]], List[int]],
    find_target_index_fn: Callable[[Sequence[str], Optional[str]], Optional[int]],
) -> Optional[Dict[int, float]]:
    """
    Compute CLIP-head-based scores for a list of detections.

    Returns a dict mapping id(det)->score, and also populates det.clip_head_prob/margin where available.
    """
    if not dets:
        return {}
    if not isinstance(clip_head, dict):
        return None
    classes = clip_head.get("classes") if isinstance(clip_head.get("classes"), list) else []
    if not classes:
        return None
    crops: List[Any] = []
    det_refs: List[Any] = []
    for det in dets:
        bbox_xyxy = bbox_to_xyxy_pixels_fn(det.bbox or [], pil_img.width, pil_img.height)
        if bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            crops.append(pil_img.crop((x1, y1, x2, y2)))
        except Exception:
            continue
        det_refs.append(det)
    if not det_refs:
        return {}

    feats = encode_pil_batch_for_head_fn(crops, head=clip_head)
    if feats is None or not isinstance(feats, np.ndarray) or feats.size == 0:
        return None
    proba = clip_head_predict_proba_fn(feats, clip_head)
    if proba is None:
        return None

    bg_indices = clip_head_background_indices_fn(classes)
    scores: Dict[int, float] = {}
    for det, row in zip(det_refs, proba):
        label = det.class_name or det.qwen_label
        t_idx = find_target_index_fn(classes, label)
        if t_idx is None or t_idx < 0 or t_idx >= row.shape[0]:
            continue
        try:
            p_t = float(row[int(t_idx)])
        except Exception:
            continue
        p_other = 0.0
        try:
            if row.shape[0] > 1:
                other = np.asarray(row, dtype=np.float32).copy()
                other[int(t_idx)] = -1.0
                p_other = float(np.max(other))
        except Exception:
            p_other = 0.0
        p_bg: Optional[float] = None
        if bg_indices:
            try:
                p_bg = float(np.max(row[bg_indices]))
            except Exception:
                p_bg = None
        det.clip_head_prob = p_t
        det.clip_head_margin = float(p_t - p_other)
        if p_bg is not None:
            det.clip_head_bg_prob = float(p_bg)
            det.clip_head_bg_margin = float(p_t - p_bg)
        score = p_t if score_mode == "clip_head_prob" else float(p_t - p_other)
        scores[id(det)] = float(score)
    return scores


def _build_clip_head_sweep_grid_impl(
    payload: Any,
    *,
    base_min_prob: float,
    base_margin: float,
    base_bg_margin: float,
    allow_bg_tune: bool,
    allow_margin_tune: Optional[bool] = None,
) -> Tuple[List[float], List[float], List[float], float]:
    """Return (min_prob candidates, margin candidates, bg_margin candidates, target_precision)."""
    try:
        base_min = float(base_min_prob)
    except Exception:
        base_min = 0.5
    try:
        base_mar = float(base_margin)
    except Exception:
        base_mar = 0.0
    try:
        base_bg = float(base_bg_margin)
    except Exception:
        base_bg = 0.0
    base_min = max(0.0, min(1.0, base_min))
    base_mar = max(0.0, min(1.0, base_mar))
    base_bg = max(0.0, min(1.0, base_bg))
    try:
        target_precision = float(getattr(payload, "clip_head_target_precision", 0.9))
    except Exception:
        target_precision = 0.9
    target_precision = max(0.0, min(1.0, target_precision))

    auto_tune = True
    try:
        auto_tune = bool(getattr(payload, "clip_head_auto_tune", True))
    except Exception:
        auto_tune = True
    bg_auto_tune = True
    try:
        bg_auto_tune = bool(getattr(payload, "clip_head_background_auto_tune", True))
    except Exception:
        bg_auto_tune = True
    margin_auto_tune = True
    try:
        margin_auto_tune = bool(
            getattr(payload, "clip_head_tune_margin", True)
            if allow_margin_tune is None
            else allow_margin_tune
        )
    except Exception:
        margin_auto_tune = True

    if not auto_tune:
        return [base_min], [base_mar], [base_bg], target_precision

    min_probs_raw: List[float] = []
    min_probs_raw.extend([0.0, 0.001, 0.002, 0.005])
    min_probs_raw.extend([round(i * 0.01, 3) for i in range(0, 11)])  # 0.00..0.10
    min_probs_raw.extend([round(i * 0.05, 3) for i in range(3, 20)])  # 0.15..0.95
    min_probs_raw.extend([0.975, 0.99])
    min_probs_raw.append(base_min)
    min_probs = sorted({float(max(0.0, min(1.0, p))) for p in min_probs_raw})

    margins = [0.0, 0.05, 0.1, 0.2]
    margins.append(base_mar)
    margins = sorted({float(max(0.0, min(1.0, m))) for m in margins})
    if not margin_auto_tune:
        margins = [base_mar]
    bg_margins = [0.0, 0.02, 0.05, 0.1, 0.2]
    bg_margins.append(base_bg)
    bg_margins = sorted({float(max(0.0, min(1.0, m))) for m in bg_margins})
    if not allow_bg_tune or not bg_auto_tune:
        bg_margins = [base_bg]
    return min_probs, margins, bg_margins, target_precision


def _score_head_tuning_candidate_impl(
    *,
    matched: int,
    fps: int,
    precision: float,
    min_prob: float,
    margin: float,
    bg_margin: float,
    target_precision: float,
) -> Tuple[int, float, int, float, float, float, float]:
    meets_target = bool(matched > 0 and precision >= float(target_precision))
    if meets_target:
        return (1, float(matched), -int(fps), float(precision), -float(min_prob), -float(margin), -float(bg_margin))
    return (0, float(precision), int(matched), -int(fps), -float(min_prob), -float(margin), -float(bg_margin))


def _update_best_clip_head_sweep_summary_impl(
    *,
    best_summary: Optional[Dict[str, Any]],
    best_key: Optional[Tuple[int, float, int, float, float, float, float]],
    total_gt: int,
    total_images: int,
    matched: int,
    fps: int,
    duplicates: int,
    preds: int,
    det_images: int,
    min_prob: float,
    margin: float,
    bg_margin: float,
    target_precision: float,
    debug: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[int, float, int, float, float, float, float]]]:
    recall = matched / total_gt if total_gt else 0.0
    precision = matched / max(1, matched + fps)
    det_rate = det_images / total_images if total_images else 0.0
    key = _score_head_tuning_candidate_impl(
        matched=int(matched),
        fps=int(fps),
        precision=float(precision),
        min_prob=float(min_prob),
        margin=float(margin),
        bg_margin=float(bg_margin),
        target_precision=float(target_precision),
    )
    if best_key is not None and key <= best_key:
        return best_summary, best_key
    summary: Dict[str, Any] = {
        "gts": int(total_gt),
        "matches": int(matched),
        "fps": int(fps),
        "duplicates": int(duplicates),
        "preds": int(preds),
        "precision": float(precision),
        "recall": float(recall),
        "coverage_rate": float(recall),
        "det_rate": float(det_rate),
        "clip_head_min_prob": float(min_prob),
        "clip_head_margin": float(margin),
        "clip_head_background_margin": float(bg_margin),
        "clip_head_target_precision": float(target_precision),
        "clip_head_meets_target_precision": bool(float(precision) >= float(target_precision)),
    }
    if debug is not None:
        summary["debug"] = debug
    return summary, key


def _successive_halving_search_impl(
    *,
    candidates: Sequence[Any],
    budgets: Sequence[int],
    evaluator: Callable[[Any, int], Tuple[Tuple[Any, ...], Any]],
    keep_ratio: float = 0.5,
    log_fn: Optional[Callable[[str], None]] = None,
    log_prefix: str = "",
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Deterministic successive-halving controller.

    - candidates: initial list of candidate configs
    - budgets: increasing list of evaluation budgets (ints); larger budgets are more expensive but more reliable
    - evaluator: function(candidate, budget) -> (key, result), where `key` is a comparable tuple; larger is better
    - keep_ratio: fraction of candidates to keep between stages (0 < keep_ratio <= 1)
    """
    if not candidates:
        raise ValueError("no_candidates")
    if not budgets:
        raise ValueError("no_budgets")
    budgets_clean: List[int] = []
    last = 0
    for b in budgets:
        try:
            b_int = int(b)
        except Exception:
            continue
        if b_int <= 0:
            continue
        if b_int <= last:
            raise ValueError("budgets_must_be_increasing")
        budgets_clean.append(b_int)
        last = b_int
    if not budgets_clean:
        raise ValueError("no_valid_budgets")

    history: List[Dict[str, Any]] = []
    remaining = list(candidates)
    for stage_idx, budget in enumerate(budgets_clean):
        stage_results: List[Tuple[Tuple[Any, ...], Any, Any]] = []
        for candidate in remaining:
            key, result = evaluator(candidate, budget)
            stage_results.append((key, result, candidate))
        stage_results.sort(key=lambda item: item[0], reverse=True)
        best_key, best_result, best_candidate = stage_results[0]
        history.append({
            "stage": stage_idx,
            "budget": int(budget),
            "best_key": best_key,
            "best_result": best_result,
            "candidates": len(stage_results),
        })
        if log_fn is not None:
            log_fn(f"{log_prefix}stage={stage_idx} budget={budget} best={best_key}")
        if stage_idx == len(budgets_clean) - 1:
            return best_candidate, history
        keep_n = max(1, int(len(stage_results) * float(keep_ratio)))
        remaining = [candidate for _key, _res, candidate in stage_results[:keep_n]]
    return remaining[0], history


def _load_labelmap_simple_impl(
    path: Optional[str],
    *,
    load_labelmap_file_fn: Callable[[Path], Sequence[str]],
) -> List[str]:
    if not path:
        return []
    try:
        classes = load_labelmap_file_fn(Path(path))
        return [str(c) for c in list(classes)]
    except Exception:
        return []


def _validate_clip_dataset_impl(
    inputs: Dict[str, str],
    *,
    http_exception_cls: Any,
    load_labelmap_simple_fn: Callable[[Optional[str]], List[str]],
) -> Dict[str, Any]:
    """
    Light validation of staged CLIP dataset to fail fast before launching a job.
    Expects keys: images_dir, labels_dir, optional labelmap_path.
    """
    images_dir = inputs.get("images_dir")
    labels_dir = inputs.get("labels_dir")
    labelmap_path = inputs.get("labelmap_path")
    if not images_dir or not labels_dir:
        raise http_exception_cls(status_code=400, detail="clip_dataset_missing_paths")
    img_root = Path(images_dir)
    lbl_root = Path(labels_dir)
    if not img_root.is_dir() or not lbl_root.is_dir():
        raise http_exception_cls(status_code=400, detail="clip_dataset_missing_paths")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files: List[Path] = []
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_exts:
            image_files.append(p)
    if not image_files:
        raise http_exception_cls(status_code=400, detail="clip_images_missing")
    label_files = [p for p in lbl_root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"]
    if not label_files:
        raise http_exception_cls(status_code=400, detail="clip_labels_missing")
    labelmap = load_labelmap_simple_fn(labelmap_path)
    max_cid = -1
    box_count = 0
    for lf in label_files:
        try:
            for line in lf.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                max_cid = max(max_cid, cid)
                box_count += 1
        except Exception:
            continue
    if box_count == 0:
        raise http_exception_cls(status_code=400, detail="clip_labels_empty")
    if labelmap and max_cid >= len(labelmap):
        raise http_exception_cls(status_code=400, detail="clip_labelmap_class_mismatch")
    return {
        "images": len(image_files),
        "labels": len(label_files),
        "boxes": box_count,
        "labelmap_classes": len(labelmap),
    }
