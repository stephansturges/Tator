from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

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


def _resolve_clip_labelmap_path_impl(
    path_str: Optional[str],
    *,
    root_hint: Optional[str],
    upload_root: Path,
    labelmap_exts: Sequence[str],
    path_is_within_root_fn: Callable[[Path, Path], bool],
) -> Optional[Path]:
    if not path_str:
        return None
    raw = str(path_str).strip()
    if not raw:
        return None
    roots: List[Path] = []
    labelmaps_root = (upload_root / "labelmaps").resolve()
    classifiers_root = (upload_root / "classifiers").resolve()
    if root_hint == "classifiers":
        roots = [classifiers_root]
    elif root_hint == "labelmaps":
        roots = [labelmaps_root]
    else:
        roots = [labelmaps_root, classifiers_root]
    for root in roots:
        try:
            candidate = (root / raw).resolve()
        except Exception:
            continue
        if not path_is_within_root_fn(candidate, root):
            continue
        if candidate.suffix.lower() not in labelmap_exts:
            continue
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _find_labelmap_for_classifier_impl(
    classifier_path: Path,
    *,
    upload_root: Path,
    labelmap_exts: Sequence[str],
    path_is_within_root_fn: Callable[[Path, Path], bool],
    joblib_load_fn: Callable[[str], Any],
    resolve_clip_labelmap_path_fn: Callable[[Optional[str], Optional[str]], Optional[Path]],
) -> Optional[Path]:
    meta_path = Path(os.path.splitext(str(classifier_path))[0] + ".meta.pkl")
    if meta_path.exists():
        try:
            meta_obj = joblib_load_fn(str(meta_path))
            if isinstance(meta_obj, dict):
                labelmap_hint = meta_obj.get("labelmap_filename") or meta_obj.get("labelmap_path")
                resolved = resolve_clip_labelmap_path_fn(labelmap_hint, "labelmaps")
                if resolved is not None:
                    return resolved
        except Exception:
            pass
    stem = classifier_path.stem
    roots = [
        (upload_root / "labelmaps").resolve(),
        (upload_root / "classifiers").resolve(),
    ]
    for root in roots:
        if not root.exists():
            continue
        for ext in labelmap_exts:
            candidate = (root / f"{stem}{ext}").resolve()
            if candidate.exists() and candidate.is_file() and path_is_within_root_fn(candidate, root):
                return candidate
    return None


def _list_clip_labelmaps_impl(
    *,
    upload_root: Path,
    labelmap_exts: Sequence[str],
    load_labelmap_file_fn: Callable[[Path], Sequence[str]],
) -> List[Dict[str, Any]]:
    labelmaps_root = (upload_root / "labelmaps").resolve()
    entries: List[Dict[str, Any]] = []
    root = labelmaps_root
    if not root.exists():
        return entries
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in labelmap_exts:
            continue
        if path.name.endswith(".meta.pkl"):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        try:
            classes = load_labelmap_file_fn(path)
        except Exception:
            classes = []
        if not classes:
            continue
        entries.append(
            {
                "filename": path.name,
                "path": str(path.resolve()),
                "rel_path": str(path.relative_to(root)),
                "root": "labelmaps",
                "n_classes": len(classes),
                "modified_at": stat.st_mtime,
            }
        )
    entries.sort(key=lambda item: (item.get("modified_at") or 0), reverse=True)
    return entries


def _list_clip_classifiers_impl(
    *,
    upload_root: Path,
    classifier_exts: Sequence[str],
    labelmap_exts: Sequence[str],
    path_is_within_root_fn: Callable[[Path, Path], bool],
    joblib_load_fn: Callable[[str], Any],
    resolve_clip_labelmap_path_fn: Callable[[Optional[str], Optional[str]], Optional[Path]],
) -> List[Dict[str, Any]]:
    """List classifier heads available for CLIP filtering (typically trained via the CLIP training tab)."""
    root = (upload_root / "classifiers").resolve()
    labelmaps_root = (upload_root / "labelmaps").resolve()
    classifiers: List[Dict[str, Any]] = []
    if not root.exists():
        return classifiers

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in classifier_exts:
            continue
        if path.name.endswith(".meta.pkl"):
            continue
        if not path_is_within_root_fn(path.resolve(), root):
            continue

        entry: Dict[str, Any] = {
            "filename": path.name,
            "path": str(path.resolve()),
            "rel_path": str(path.relative_to(root)),
        }

        meta_path = os.path.splitext(str(path))[0] + ".meta.pkl"
        if os.path.exists(meta_path):
            try:
                meta_obj = joblib_load_fn(meta_path)
                if isinstance(meta_obj, dict):
                    entry["clip_model"] = meta_obj.get("clip_model")
                    entry["encoder_type"] = meta_obj.get("encoder_type") or "clip"
                    entry["encoder_model"] = meta_obj.get("encoder_model") or entry.get("clip_model")
                    entry["solver"] = meta_obj.get("solver")
                    entry["classifier_type"] = meta_obj.get("classifier_type")
                    entry["embedding_dim"] = meta_obj.get("embedding_dim")
                    entry["n_samples_train"] = meta_obj.get("n_samples_train")
                    entry["n_samples_test"] = meta_obj.get("n_samples_test")
                    labelmap_hint = meta_obj.get("labelmap_filename") or meta_obj.get("labelmap_path")
                    if labelmap_hint:
                        resolved = resolve_clip_labelmap_path_fn(labelmap_hint, "labelmaps")
                        if resolved is not None:
                            entry["labelmap_guess"] = str(resolved)
                            entry["labelmap_guess_rel"] = str(resolved.relative_to(labelmaps_root))
            except Exception:
                pass

        try:
            clf_obj = joblib_load_fn(str(path))
            classes_raw = getattr(clf_obj, "classes_", None)
            if classes_raw is not None:
                entry["classes"] = [str(c) for c in list(classes_raw)]
                entry["n_classes"] = len(entry["classes"])
            elif isinstance(clf_obj, dict) and clf_obj.get("classes"):
                entry["classes"] = [str(c) for c in list(clf_obj.get("classes") or [])]
                entry["n_classes"] = len(entry["classes"])
                entry["classifier_type"] = clf_obj.get("classifier_type") or clf_obj.get("head_type")
            coef = getattr(clf_obj, "coef_", None)
            if coef is not None and hasattr(coef, "shape") and len(getattr(coef, "shape", [])) >= 2:
                entry["embedding_dim"] = int(coef.shape[1])
        except Exception as exc:  # noqa: BLE001
            entry["load_error"] = str(exc)
        if "encoder_type" not in entry:
            entry["encoder_type"] = "clip"
        if "encoder_model" not in entry:
            entry["encoder_model"] = entry.get("clip_model")

        try:
            entry["modified_at"] = path.stat().st_mtime
        except Exception:
            entry["modified_at"] = None
        try:
            stem = path.stem
            for ext in labelmap_exts:
                candidate = (labelmaps_root / f"{stem}{ext}").resolve()
                if candidate.exists() and path_is_within_root_fn(candidate, labelmaps_root):
                    entry["labelmap_guess"] = str(candidate)
                    entry["labelmap_guess_rel"] = str(candidate.relative_to(labelmaps_root))
                    break
        except Exception:
            pass
        classifiers.append(entry)

    classifiers.sort(key=lambda c: (c.get("modified_at") or 0), reverse=True)
    return classifiers
