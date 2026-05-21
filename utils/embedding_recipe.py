"""Shared crop and embedding recipe helpers for object-crop encoders."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def normalize_crop_mode(value: Any) -> str:
    mode = str(value or "padded_square").strip().lower()
    if mode in {"tight", "bbox", "raw_bbox"}:
        return "tight"
    if mode in {"padded", "pad"}:
        return "padded"
    return "padded_square"


def normalize_preprocess_mode(value: Any) -> str:
    mode = str(value or "canonical").strip().lower()
    if mode in {"native", "raw", "processor"}:
        return "native"
    return "canonical"


def normalize_embedding_adjustment(value: Any) -> str:
    mode = str(value or "remove_size_bias").strip().lower()
    if mode in {"none", "off", "raw"}:
        return "none"
    if mode in {"remove_size_bias", "size_residualized", "residualize_size", "area_residualized"}:
        return "remove_size_bias"
    return "remove_size_bias"


def normalize_dinov3_pooling(value: Any) -> str:
    mode = str(value or "pooler").strip().lower()
    if mode in {"cls", "cls_token"}:
        return "cls"
    if mode in {"patch_mean", "mean_patch", "patches_mean"}:
        return "patch_mean"
    if mode in {"cls_patch_concat", "concat", "cls+patch"}:
        return "cls_patch_concat"
    return "pooler"


def normalize_embedding_aggregation(value: Any) -> str:
    mode = str(value or "pooled").strip().lower()
    if mode in {"local_salad", "salad", "salad_local"}:
        return "local_salad"
    return "pooled"


def normalize_background_mode(value: Any) -> str:
    mode = str(value or "full_crop").strip().lower()
    if mode in {"none", "off", "raw", "full", "full_crop"}:
        return "full_crop"
    if mode in {"mean", "mean_fill", "mean_fill_outside_box", "fill_outside_box"}:
        return "mean_fill_outside_box"
    if mode in {"blur", "blur_outside_box"}:
        return "blur_outside_box"
    if mode in {"darken", "dim", "darken_outside_box", "dim_outside_box"}:
        return "darken_outside_box"
    return "full_crop"


def normalize_embedding_view_mode(value: Any) -> str:
    mode = str(value or "single").strip().lower()
    if mode in {"single", "one", "standard"}:
        return "single"
    if mode in {"tight_standard", "tight_plus_standard", "tight+standard"}:
        return "tight_standard"
    if mode in {"standard_context", "standard_plus_context", "context", "standard+context"}:
        return "standard_context"
    if mode in {"tight_context", "tight_plus_context", "multi_scale", "multiscale", "tight+context"}:
        return "tight_context"
    return "single"


def embedding_view_specs(
    *,
    crop_mode: str,
    padding_ratio: float,
    view_mode: str,
    context_padding_ratio: float = 0.25,
) -> List[Tuple[str, float, str]]:
    base_mode = normalize_crop_mode(crop_mode)
    base_pad = max(0.0, float(padding_ratio or 0.0))
    context_pad = max(base_pad, float(context_padding_ratio or 0.25))
    mode = normalize_embedding_view_mode(view_mode)
    if mode == "tight_standard":
        return [("tight", 0.0, "tight"), (base_mode, base_pad, "standard")]
    if mode == "standard_context":
        return [(base_mode, base_pad, "standard"), ("padded_square", context_pad, "context")]
    if mode == "tight_context":
        return [("tight", 0.0, "tight"), ("padded_square", context_pad, "context")]
    return [(base_mode, base_pad, "standard")]


def crop_bounds(
    bbox_xyxy: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    crop_mode: str,
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad = max(0.0, float(padding_ratio or 0.0))
    mode = normalize_crop_mode(crop_mode)
    if mode == "tight":
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    elif mode == "padded":
        nx1 = x1 - width * pad
        nx2 = x2 + width * pad
        ny1 = y1 - height * pad
        ny2 = y2 + height * pad
    else:
        side = max(width, height) * (1.0 + 2.0 * pad)
        cx = x1 + width / 2.0
        cy = y1 + height / 2.0
        nx1 = cx - side / 2.0
        nx2 = cx + side / 2.0
        ny1 = cy - side / 2.0
        ny2 = cy + side / 2.0
    left = max(0, min(int(math.floor(nx1)), int(image_width)))
    top = max(0, min(int(math.floor(ny1)), int(image_height)))
    right = max(0, min(int(math.ceil(nx2)), int(image_width)))
    bottom = max(0, min(int(math.ceil(ny2)), int(image_height)))
    if right <= left:
        right = min(int(image_width), left + 1)
    if bottom <= top:
        bottom = min(int(image_height), top + 1)
    return left, top, right, bottom


def _local_bbox_for_crop(
    *,
    bbox_xyxy: Sequence[float],
    crop_xyxy: Sequence[float],
    crop_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    bx1, by1, bx2, by2 = [float(v) for v in bbox_xyxy[:4]]
    cx1, cy1, _cx2, _cy2 = [float(v) for v in crop_xyxy[:4]]
    width, height = crop_size
    left = max(0, min(int(math.floor(bx1 - cx1)), int(width)))
    top = max(0, min(int(math.floor(by1 - cy1)), int(height)))
    right = max(0, min(int(math.ceil(bx2 - cx1)), int(width)))
    bottom = max(0, min(int(math.ceil(by2 - cy1)), int(height)))
    if right <= left:
        right = min(int(width), left + 1)
    if bottom <= top:
        bottom = min(int(height), top + 1)
    return left, top, right, bottom


def apply_background_mode(
    crop: Image.Image,
    *,
    bbox_xyxy: Sequence[float],
    crop_xyxy: Sequence[float],
    mode: str,
) -> Image.Image:
    mode_norm = normalize_background_mode(mode)
    rgb = crop.convert("RGB")
    if mode_norm == "full_crop":
        return rgb.copy()
    box = _local_bbox_for_crop(bbox_xyxy=bbox_xyxy, crop_xyxy=crop_xyxy, crop_size=rgb.size)
    if mode_norm == "blur_outside_box":
        background = rgb.filter(ImageFilter.GaussianBlur(radius=max(2, int(max(rgb.size) * 0.04))))
    elif mode_norm == "darken_outside_box":
        background = ImageEnhance.Brightness(rgb).enhance(0.28)
    else:
        try:
            mean = tuple(int(v) for v in np.asarray(rgb.resize((1, 1), Image.Resampling.BOX))[0, 0, :3])
        except Exception:
            mean = (0, 0, 0)
        background = Image.new("RGB", rgb.size, mean)
    background.paste(rgb.crop(box), box)
    return background


def preprocess_crop(crop: Image.Image, *, mode: str, canonical_size: int) -> Image.Image:
    if normalize_preprocess_mode(mode) == "native":
        return crop.convert("RGB").copy()
    size = max(64, min(1024, int(canonical_size or 336)))
    rgb = crop.convert("RGB")
    width, height = rgb.size
    if width <= 0 or height <= 0:
        return Image.new("RGB", (size, size), (0, 0, 0))
    if width == height:
        return rgb.resize((size, size), Image.Resampling.LANCZOS)
    try:
        fill = tuple(int(v) for v in np.asarray(rgb.resize((1, 1), Image.Resampling.BOX))[0, 0, :3])
    except Exception:
        fill = (0, 0, 0)
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(rgb, ((side - width) // 2, (side - height) // 2))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def make_embedding_crop_views(
    image: Image.Image,
    bbox_xyxy: Sequence[float],
    *,
    crop_mode: str,
    padding_ratio: float,
    preprocess_mode: str,
    canonical_size: int,
    background_mode: str = "full_crop",
    view_mode: str = "single",
    context_padding_ratio: float = 0.25,
) -> Tuple[List[Image.Image], Tuple[int, int, int, int], List[Dict[str, Any]]]:
    width, height = image.size
    views: List[Image.Image] = []
    metadata: List[Dict[str, Any]] = []
    primary_bounds: Optional[Tuple[int, int, int, int]] = None
    for view_crop_mode, view_padding, view_name in embedding_view_specs(
        crop_mode=crop_mode,
        padding_ratio=padding_ratio,
        view_mode=view_mode,
        context_padding_ratio=context_padding_ratio,
    ):
        bounds = crop_bounds(
            bbox_xyxy,
            image_width=width,
            image_height=height,
            crop_mode=view_crop_mode,
            padding_ratio=view_padding,
        )
        if primary_bounds is None:
            primary_bounds = bounds
        raw_crop = image.crop(bounds)
        try:
            masked = apply_background_mode(
                raw_crop,
                bbox_xyxy=bbox_xyxy,
                crop_xyxy=bounds,
                mode=background_mode,
            )
        finally:
            try:
                raw_crop.close()
            except Exception:
                pass
        try:
            processed = preprocess_crop(masked, mode=preprocess_mode, canonical_size=canonical_size)
        finally:
            try:
                masked.close()
            except Exception:
                pass
        views.append(processed)
        metadata.append(
            {
                "view": view_name,
                "crop_mode": normalize_crop_mode(view_crop_mode),
                "padding_ratio": float(view_padding),
                "crop_xyxy": [int(v) for v in bounds],
            }
        )
    if primary_bounds is None:
        primary_bounds = crop_bounds(
            bbox_xyxy,
            image_width=width,
            image_height=height,
            crop_mode=crop_mode,
            padding_ratio=padding_ratio,
        )
    return views, primary_bounds, metadata


def covariate_row(
    *,
    bbox_xyxy: Sequence[float],
    crop_xyxy: Optional[Sequence[float]] = None,
) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if crop_xyxy is not None and len(crop_xyxy) >= 4:
        cx1, cy1, cx2, cy2 = [float(v) for v in crop_xyxy[:4]]
        crop_width = max(1.0, cx2 - cx1)
        crop_height = max(1.0, cy2 - cy1)
    else:
        crop_width = width
        crop_height = height
    return [
        math.log1p(width * height),
        math.log1p(crop_width * crop_height),
        math.log(max(width / height, 1e-6)),
        math.log(max(crop_width / crop_height, 1e-6)),
    ]


COVARIATE_NAMES = [
    "log_bbox_area",
    "log_crop_area",
    "log_bbox_aspect",
    "log_crop_aspect",
]


def covariates_from_records(records: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    rows: List[List[float]] = []
    for record in records:
        bbox = record.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            width = max(1.0, float(record.get("width") or 1.0))
            height = max(1.0, float(record.get("height") or 1.0))
            bbox = [0.0, 0.0, width, height]
        crop = record.get("crop_xyxy")
        rows.append(covariate_row(bbox_xyxy=bbox, crop_xyxy=crop if isinstance(crop, (list, tuple)) else None))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.asarray(rows, dtype=np.float32), list(COVARIATE_NAMES)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        return arr
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return (arr / denom).astype(np.float32, copy=False)


def fit_size_bias_residualizer(
    embeddings: np.ndarray,
    covariates: np.ndarray,
) -> Optional[Dict[str, Any]]:
    emb = np.asarray(embeddings, dtype=np.float32)
    cov = np.asarray(covariates, dtype=np.float32)
    if emb.ndim != 2 or cov.ndim != 2 or emb.shape[0] != cov.shape[0] or emb.shape[0] < 6:
        return None
    std = cov.std(axis=0)
    keep = std > 1e-6
    if not np.any(keep):
        return None
    selected = cov[:, keep]
    mean = selected.mean(axis=0)
    selected_std = selected.std(axis=0)
    selected_std = np.where(selected_std <= 1e-6, 1.0, selected_std)
    design = np.concatenate(
        [
            np.ones((selected.shape[0], 1), dtype=np.float32),
            ((selected - mean) / selected_std).astype(np.float32),
        ],
        axis=1,
    )
    beta, *_ = np.linalg.lstsq(design, emb, rcond=None)
    return {
        "mode": "remove_size_bias",
        "covariate_names": [name for name, use in zip(COVARIATE_NAMES, keep.tolist()) if use],
        "keep_mask": keep.astype(bool).tolist(),
        "mean": mean.astype(np.float32).tolist(),
        "std": selected_std.astype(np.float32).tolist(),
        "beta": np.asarray(beta, dtype=np.float32).tolist(),
    }


def apply_size_bias_residualizer(
    embeddings: np.ndarray,
    covariates: np.ndarray,
    transform: Optional[Mapping[str, Any]],
    *,
    normalize: bool = True,
) -> np.ndarray:
    emb = np.asarray(embeddings, dtype=np.float32)
    if not transform:
        return normalize_rows(emb) if normalize else emb
    cov = np.asarray(covariates, dtype=np.float32)
    try:
        keep = np.asarray(transform.get("keep_mask"), dtype=bool)
        mean = np.asarray(transform.get("mean"), dtype=np.float32).reshape(1, -1)
        std = np.asarray(transform.get("std"), dtype=np.float32).reshape(1, -1)
        beta = np.asarray(transform.get("beta"), dtype=np.float32)
        if cov.ndim != 2 or keep.ndim != 1 or cov.shape[1] != keep.shape[0] or cov.shape[0] != emb.shape[0]:
            return normalize_rows(emb) if normalize else emb
        selected = cov[:, keep]
        if selected.shape[1] != mean.shape[1] or selected.shape[1] != std.shape[1]:
            return normalize_rows(emb) if normalize else emb
        std = np.where(std <= 1e-6, 1.0, std)
        design = np.concatenate(
            [
                np.ones((selected.shape[0], 1), dtype=np.float32),
                ((selected - mean) / std).astype(np.float32),
            ],
            axis=1,
        )
        if beta.ndim != 2 or beta.shape[0] != design.shape[1] or beta.shape[1] != emb.shape[1]:
            return normalize_rows(emb) if normalize else emb
        adjusted = emb - design @ beta
        return normalize_rows(adjusted.astype(np.float32, copy=False)) if normalize else adjusted.astype(np.float32, copy=False)
    except Exception:
        return normalize_rows(emb) if normalize else emb
