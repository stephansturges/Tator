"""Pruning-only correction strengths over the existing DQE geometry fit.

Raw descriptor identities never include a correction strength or a fusion
weight. Saved coefficients retain the existing replay schema: its coefficients
are the effective, strength-scaled prediction, including the fitted offset.
The unscaled fit is retained separately so another strength does not fit again
or interpolate already normalized descriptors.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from services.dataset_pruning import digest

FIT_SCHEMA = "pruning-geometry-base-fit-v1"
FORMULA = "unit(raw - strength * (offset + geometry_prediction))"


def validate_strength(value, *, default=1.0):
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Correction strength must be a finite number between 0 and 1")
    return float(value)


def effective_transform(transform, strength):
    """Rescale a saved complete prediction; compatible with the shared replay.

An older applied transform has full strength. A transform already rescaled by
this module always starts from its retained base coefficients, rather than
scaling the effective coefficients a second time.
    """
    alpha = validate_strength(strength)
    result = copy.deepcopy(transform)
    fitted = result.get("fitted_transform") or {}
    base = fitted.get("base_coefficients", fitted.get("coefficients"))
    if not transform.get("applied") and not fitted.get("base_coefficients"):
        result.update(correction_strength=0.0, requested_correction_strength=alpha,
                      correction_formula=FORMULA)
        return result
    if fitted.get("schema") != "independent_size_residualization_v1" or fitted.get("normalization") != "subtract_design_then_l2":
        raise ValueError("The complete saved geometry prediction is unavailable")
    beta = np.asarray(base, np.float32)
    if beta.ndim != 2 or not beta.size or not np.isfinite(beta).all():
        raise ValueError("Saved base geometry coefficients are invalid")
    fitted["base_coefficients"] = beta.tolist()
    fitted["base_coefficients_sha256"] = hashlib.sha256(beta.tobytes()).hexdigest()
    fitted["coefficients"] = (beta * np.float32(alpha)).tolist()
    fitted["correction_strength"] = alpha
    fitted["prediction_scope"] = "complete fitted offset and geometry terms"
    result.update(fitted_transform=fitted, applied=alpha > 0,
                  correction_strength=alpha, requested_correction_strength=alpha,
                  correction_formula=FORMULA)
    if alpha == 0:
        result["skipped_reason"] = "zero_correction_strength"
    elif result.get("skipped_reason") == "zero_correction_strength":
        result.pop("skipped_reason")
    return result


def _fit_identity(api, raw, records, raw_recipe):
    covariates, names = api._class_analysis_record_covariates(records)
    # Include actual geometry inputs, not merely the row count or raw width.
    # Function source identifies the existing fit without encoder code changes.
    functions = [api._class_analysis_record_covariates,
                 api._class_analysis_streamed_small_design_lstsq,
                 api._class_analysis_adjust_embeddings_to_memmap]
    implementations = {function.__name__: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
                       for function in functions}
    return {"schema": FIT_SCHEMA, "raw_recipe": raw_recipe,
            "record_digest": api._class_analysis_embedding_record_digest(records),
            "covariates": names, "covariates_sha256": hashlib.sha256(np.asarray(covariates, np.float32).tobytes()).hexdigest(),
            "raw_shape": list(raw.shape), "implementation": implementations}


def _read_base_fit(path, identity):
    if path.is_symlink():
        raise ValueError("Saved geometry fit cache is unavailable or changed")
    if not path.exists():
        return None
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Saved geometry fit cache is unavailable or changed")
    value = json.loads(path.read_text())
    transform = value.get("transform")
    if value.get("schema") != FIT_SCHEMA or value.get("identity") != identity or not isinstance(transform, dict) or value.get("transform_digest") != digest(transform):
        raise ValueError("Saved geometry fit cache does not match its recorded population")
    return transform


def _write_base_fit(path, identity, transform):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema": FIT_SCHEMA, "identity": identity, "transform": transform,
             "transform_digest": digest(transform)}
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def adjust_component_to_memmap(api, raw, records, *, mode, output_path,
                               strength=None, output_array=None, chunk_size=2048,
                               fit_cache_root=None, raw_recipe=None, base_transform=None):
    """Reuse or fit one geometry predictor, then apply the requested strength.

Only owned derived files are written. The existing shared fitter and replay
perform the math; the input matrix, saved heads and pixel caches are untouched.
Legacy absent strength means full correction when enabled, zero when raw.
    """
    source = np.asarray(raw, np.float32)
    if source.ndim != 2 or len(records) != len(source):
        raise ValueError("Correction rows do not match their geometry records")
    mode_norm = api._class_analysis_normalize_embedding_adjustment(mode)
    alpha = validate_strength(strength, default=0.0 if mode_norm == "none" else 1.0)
    if mode_norm == "none" and alpha != 0:
        raise ValueError("Raw components must use correction strength 0")
    if output_array is not None and np.shares_memory(source, output_array):
        raise ValueError("Geometry correction must not overwrite raw descriptors")
    cache_path = None
    identity = None
    base = base_transform
    if mode_norm != "none" and alpha > 0 and fit_cache_root is not None and raw_recipe is not None:
        identity = _fit_identity(api, source, records, raw_recipe)
        cache_path = Path(fit_cache_root) / (digest(identity) + ".json")
        if base is None:
            base = _read_base_fit(cache_path, identity)
    reused = base is not None
    if base is None:
        adjusted, base = api._class_analysis_adjust_embeddings_to_memmap(
            source, records, mode="none" if alpha == 0 else mode_norm,
            output_path=Path(output_path), chunk_size=chunk_size, output_array=output_array)
        if cache_path is not None and base.get("applied"):
            _write_base_fit(cache_path, identity, base)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        adjusted = output_array if output_array is not None else np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=source.shape)
        if adjusted.shape != source.shape or adjusted.dtype != np.float32:
            raise ValueError("Correction output shape does not match the raw component")
    transform = effective_transform(base, alpha)
    transform["mode"] = mode_norm
    transform["base_fit_reused"] = reused
    if identity is not None:
        transform["base_fit_identity"] = digest(identity)
    # Fresh full-strength fitting already produced the exact legacy output.
    # A changed strength or saved fit replays only bounded row chunks.
    if reused or (alpha != 1 and base.get("applied")):
        for start in range(0, len(source), max(1, int(chunk_size))):
            end = min(len(source), start + max(1, int(chunk_size)))
            adjusted[start:end] = api._class_analysis_replay_fitted_embedding_adjustment(
                source[start:end], records[start:end], transform)
        adjusted.flush()
    return adjusted, transform
