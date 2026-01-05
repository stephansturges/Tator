"""Reusable helpers for training CLIP + Logistic Regression models.

This module centralises the logic that used to live in
``tools/train_clip_regression_from_YOLO.py`` so that the FastAPI layer and the
CLI script can share the same implementation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import clip
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from PIL import Image, ImageFile
try:
    import albumentations as A
except Exception:  # noqa: BLE001
    A = None
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.cluster import MiniBatchKMeans

# The datasets we work with can include truncated images; be lenient when reading.
ImageFile.LOAD_TRUNCATED_IMAGES = True

EMBED_CACHE_ROOT = Path(os.environ.get("CLIP_EMBED_CACHE", "./uploads/clip_embeddings"))
CACHE_VERSION = 1

logger = logging.getLogger(__name__)

CLIP_AUG_HFLIP_P = 0.5
CLIP_AUG_VFLIP_P = 0.1
CLIP_AUG_BRIGHTNESS_LIMIT = 0.2
CLIP_AUG_CONTRAST_LIMIT = 0.2
CLIP_AUG_HUE_SHIFT = 8
CLIP_AUG_SAT_SHIFT = 20
CLIP_AUG_VAL_SHIFT = 10
CLIP_AUG_GRAY_P = 0.05
CLIP_AUG_SAFE_ROTATE_P = 0.1
CLIP_AUG_SAFE_ROTATE_LIMIT = (-90, 90)
CLIP_AUG_ISO_P = 0.05
CLIP_AUG_ISO_COLOR_SHIFT = (0.01, 0.05)
CLIP_AUG_ISO_INTENSITY = (0.1, 0.4)
CLIP_AUG_GAUSS_P = 0.05
CLIP_AUG_GAUSS_VAR = (5.0, 20.0)
CLIP_OVERSAMPLE_TARGET_PCTL_LOW = 50.0
CLIP_OVERSAMPLE_TARGET_PCTL_HIGH = 90.0
CLIP_OVERSAMPLE_TARGET_BLEND = 0.5
CLIP_OVERSAMPLE_ALPHA = 0.6
CLIP_OVERSAMPLE_MAX_MULTIPLIER = 4.0


ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]


class TrainingError(RuntimeError):
    """Raised when the training pipeline fails in a recoverable way."""


@dataclass
class TrainingArtifacts:
    model_path: str
    labelmap_path: str
    meta_path: str
    accuracy: float
    classes_seen: int
    samples_train: int
    samples_test: int
    clip_model: str
    encoder_type: str
    encoder_model: str
    embedding_dim: int
    device: str
    classification_report: str
    confusion_matrix: List[List[int]]
    label_order: List[str]
    iterations_run: int
    converged: bool
    convergence_trace: List[Dict[str, Optional[float]]]
    solver: str
    hard_example_mining: bool
    class_weight: str
    effective_beta: float
    per_class_metrics: List[Dict[str, Optional[float]]]
    hard_mining_misclassified_weight: float
    hard_mining_low_conf_weight: float
    hard_mining_low_conf_threshold: float
    hard_mining_margin_threshold: float
    convergence_tol: float
    background_class_count: int
    background_classes: List[str]
    negative_crop_policy: Dict[str, Any]
    augmentation_policy: Dict[str, Any]
    oversample_policy: Dict[str, Any]
    classifier_type: str
    mlp_hidden_sizes: List[int]
    mlp_dropout: float
    mlp_epochs: int
    mlp_lr: float
    mlp_weight_decay: float
    mlp_label_smoothing: float
    mlp_loss_type: str
    mlp_focal_gamma: float
    mlp_focal_alpha: Optional[float]
    mlp_sampler: str
    mlp_mixup_alpha: float
    mlp_normalize_embeddings: bool
    mlp_patience: int
    mlp_activation: str
    mlp_layer_norm: bool
    mlp_hard_mining_epochs: int
    logit_adjustment_mode: str
    logit_adjustment_inference: bool
    logit_adjustment: Optional[List[float]]
    arcface_enabled: bool
    arcface_margin: float
    arcface_scale: float
    supcon_weight: float
    supcon_temperature: float
    supcon_projection_dim: int
    supcon_projection_hidden: int
    embedding_center: bool
    embedding_standardize: bool
    calibration_mode: str
    calibration_temperature: Optional[float]
    phase_timings: Dict[str, float]


def _safe_progress(progress_cb: Optional[ProgressCallback], value: float, message: str) -> None:
    if progress_cb:
        capped = max(0.0, min(1.0, value))
        try:
            progress_cb(capped, message)
        except Exception:
            # Never let callback issues break the training job.
            pass


def _parse_hidden_sizes(raw: Optional[str]) -> List[int]:
    if raw is None:
        return [256]
    if isinstance(raw, (list, tuple)):
        sizes = [int(s) for s in raw if int(s) > 0]
        return sizes or [256]
    text = str(raw).strip()
    if not text:
        return [256]
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    sizes: List[int] = []
    for part in parts:
        if not part:
            continue
        try:
            value = int(float(part))
        except Exception:
            continue
        if value > 0:
            sizes.append(value)
    return sizes or [256]


def _effective_number_weights(counts: Dict[int, int], beta: float) -> Dict[int, float]:
    if not counts:
        return {}
    weights: Dict[int, float] = {}
    for cls_id, count in counts.items():
        if count <= 0:
            weights[int(cls_id)] = 0.0
            continue
        effective_num = 1.0 - (beta ** float(count))
        weight = (1.0 - beta) / max(effective_num, 1e-12)
        weights[int(cls_id)] = float(weight)
    mean_val = sum(weights.values()) / max(1, len(weights))
    if mean_val > 0:
        for key in list(weights.keys()):
            weights[key] = weights[key] / mean_val
    return weights


def _logit_adjustment_from_counts(counts: Dict[int, int], num_classes: int) -> Optional[np.ndarray]:
    if not counts or num_classes <= 0:
        return None
    prior = np.ones(num_classes, dtype=np.float32)
    for idx in range(num_classes):
        count = counts.get(idx, 0)
        prior[idx] = max(1.0, float(count))
    total = float(prior.sum())
    if total <= 0:
        return None
    prior = prior / total
    return -np.log(prior + 1e-12)


def _effective_number_weight_map(labels: Sequence[object], beta: float) -> Dict[object, float]:
    counts = Counter(labels)
    if not counts:
        return {}
    weights: Dict[object, float] = {}
    for label, count in counts.items():
        if count <= 0:
            weights[label] = 0.0
            continue
        effective_num = 1.0 - (beta ** float(count))
        weight = (1.0 - beta) / max(effective_num, 1e-12)
        weights[label] = float(weight)
    mean_val = sum(weights.values()) / max(1, len(weights))
    if mean_val > 0:
        for key in list(weights.keys()):
            weights[key] = weights[key] / mean_val
    return weights


def _ensure_cache_root() -> None:
    EMBED_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _detect_dataset_signature(images_path: str, labels_path: str) -> Optional[str]:
    candidates: List[Path] = []
    for root in (Path(images_path), Path(labels_path)):
        candidates.append(root / "metadata.json")
        candidates.append(root / "dataset_meta.json")
        candidates.append(root.parent / "metadata.json")
        candidates.append(root.parent / "dataset_meta.json")
    for path in candidates:
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            signature = data.get("signature")
            if signature:
                return str(signature)
        except Exception:
            continue
    return None


def _label_fingerprint(label_root: Path) -> Optional[str]:
    try:
        count = 0
        max_mtime = 0
        total_size = 0
        for label_file in sorted(label_root.glob("**/*.txt")):
            try:
                stat = label_file.stat()
            except FileNotFoundError:
                continue
            count += 1
            if stat.st_mtime_ns > max_mtime:
                max_mtime = stat.st_mtime_ns
            total_size += stat.st_size
        return f"{count}:{max_mtime}:{total_size}"
    except Exception:
        return None


def _compute_dataset_signature(
    images_path: str,
    labels_path: str,
    clip_model: str,
    *,
    encoder_type: str = "clip",
    encoder_model: Optional[str] = None,
    bg_class_count: int,
    labelmap_path: Optional[str] = None,
    bg_policy: Optional[str] = None,
    aug_policy: Optional[str] = None,
    oversample_policy: Optional[str] = None,
    embed_norm: Optional[bool] = None,
) -> str:
    encoder_name = str(encoder_model or clip_model or "").strip()
    entries: List[str] = [f"encoder:{encoder_type}:{encoder_name}", f"bg:{bg_class_count}"]
    if labelmap_path:
        try:
            lm_path = Path(labelmap_path)
            if lm_path.exists() and lm_path.is_file():
                stat = lm_path.stat()
                lm_hash = hashlib.sha256(lm_path.read_bytes()).hexdigest()
                entries.append(f"LM:{lm_path.name}:{stat.st_mtime_ns}:{stat.st_size}:{lm_hash}")
        except Exception:
            pass
    if bg_policy:
        entries.append(f"bg_policy:{bg_policy}")
    if aug_policy:
        entries.append(f"aug_policy:{aug_policy}")
    if oversample_policy:
        entries.append(f"oversample_policy:{oversample_policy}")
    if embed_norm is not None:
        entries.append(f"embed_norm:{int(bool(embed_norm))}")
    dataset_signature = _detect_dataset_signature(images_path, labels_path)
    label_root = Path(labels_path)
    if dataset_signature:
        entries.append(f"dataset_sig:{dataset_signature}")
        entries.append(f"images_root:{Path(images_path).name}")
        entries.append(f"labels_root:{Path(labels_path).name}")
        label_fp = _label_fingerprint(label_root)
        if label_fp:
            entries.append(f"labels_fp:{label_fp}")
        digest = hashlib.sha256("|".join(entries).encode("utf-8")).hexdigest()
        return digest
    image_root = Path(images_path)

    for label_file in sorted(label_root.glob("**/*.txt")):
        try:
            stat = label_file.stat()
        except FileNotFoundError:
            continue
        rel = label_file.relative_to(label_root)
        entries.append(f"L:{rel}:{stat.st_mtime_ns}:{stat.st_size}")

    for image_file in sorted(image_root.glob("**/*")):
        if image_file.is_dir():
            continue
        if image_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}:
            continue
        try:
            stat = image_file.stat()
        except FileNotFoundError:
            continue
        rel = image_file.relative_to(image_root)
        entries.append(f"I:{rel}:{stat.st_mtime_ns}:{stat.st_size}")

    digest = hashlib.sha256("|".join(entries).encode("utf-8")).hexdigest()
    return digest


def _load_cached_embeddings(signature: str) -> Optional[Dict[str, object]]:
    cache_dir = EMBED_CACHE_ROOT / signature
    meta_path = cache_dir / "metadata.joblib"
    if not meta_path.exists():
        return None
    try:
        meta = joblib.load(meta_path)
    except Exception:
        return None
    if not isinstance(meta, dict) or meta.get("version") != CACHE_VERSION:
        return None

    chunk_records = []
    for chunk_info in meta.get("chunks", []):
        rel = chunk_info.get("filename")
        start = chunk_info.get("start", 0)
        count = chunk_info.get("count", 0)
        if rel is None:
            continue
        chunk_path = cache_dir / rel
        if not chunk_path.exists():
            return None
        chunk_records.append((str(chunk_path), int(start), int(count)))

    if not chunk_records:
        return None

    return {
        "chunk_dir": cache_dir,
        "chunk_records": chunk_records,
        "y_class_names": meta.get("y_class_names", []),
        "y_numeric": meta.get("y_numeric", []),
        "groups": meta.get("groups", []),
        "encountered_cids": set(meta.get("encountered_cids", [])),
        "bg_policy": meta.get("bg_policy"),
        "aug_policy": meta.get("aug_policy"),
        "oversample_policy": meta.get("oversample_policy"),
        "background_classes": meta.get("background_classes", []),
        "embed_norm": meta.get("embed_norm"),
    }


def _write_cache_metadata(signature: str,
                          chunk_dir: Path,
                          chunk_records: List[Tuple[str, int, int]],
                          y_class_names: List[str],
                          y_numeric: List[int],
                          groups: List[str],
                          encountered_cids: set[int],
                          *,
                          bg_policy: Optional[Dict[str, float]] = None,
                          aug_policy: Optional[Dict[str, float]] = None,
                          oversample_policy: Optional[Dict[str, float]] = None,
                          background_classes: Optional[List[str]] = None,
                          labelmap_path: Optional[str] = None,
                          labelmap_hash: Optional[str] = None,
                          embed_norm: Optional[bool] = None) -> None:
    meta_path = chunk_dir / "metadata.joblib"
    rel_chunks = []
    for chunk_path, start, count in chunk_records:
        rel_chunks.append({
            "filename": os.path.basename(chunk_path),
            "start": int(start),
            "count": int(count),
        })
    payload = {
        "version": CACHE_VERSION,
        "signature": signature,
        "y_class_names": list(y_class_names),
        "y_numeric": list(map(int, y_numeric)),
        "groups": list(groups),
        "encountered_cids": list(map(int, encountered_cids)),
        "chunks": rel_chunks,
        "bg_policy": bg_policy,
        "aug_policy": aug_policy,
        "oversample_policy": oversample_policy,
        "background_classes": list(background_classes or []),
        "labelmap_path": labelmap_path,
        "labelmap_hash": labelmap_hash,
        "embed_norm": bool(embed_norm) if embed_norm is not None else None,
    }
    joblib.dump(payload, meta_path, compress=3)


def _load_labelmap(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    lower = path.lower()
    if lower.endswith(".pkl"):
        obj = joblib.load(path)
        if not isinstance(obj, list):
            raise TrainingError("Labelmap pickle must contain a list of class names.")
        return [str(x) for x in obj]
    if not os.path.isfile(path):
        raise TrainingError(f"Labelmap file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        entries = [ln.strip() for ln in handle if ln.strip()]
    if not entries:
        raise TrainingError("Labelmap text file is empty.")
    return entries


def _resolve_device(requested: Optional[str]) -> str:
    if requested:
        req = requested.strip().lower()
        if req == "cpu":
            return "cpu"
        if req == "cuda":
            if not torch.cuda.is_available():
                raise TrainingError("CUDA requested but no GPU is available.")
            return "cuda"
        if req.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise TrainingError("CUDA requested but no GPU is available.")
            try:
                idx = int(req.split(":", 1)[1])
            except Exception as exc:
                raise TrainingError(f"Invalid CUDA device '{requested}'. Use cuda:0, cuda:1, etc.") from exc
            if idx < 0 or idx >= torch.cuda.device_count():
                raise TrainingError(f"CUDA device index {idx} is out of range.")
            return f"cuda:{idx}"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_clip(arch: str, device: str) -> Tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    try:
        model, preprocess = clip.load(arch, device=device)
    except Exception as exc:
        raise TrainingError(f"Failed to load CLIP backbone '{arch}': {exc}") from exc
    model.eval()
    return model, preprocess


def _load_dinov3(model_name: str, device: str) -> Tuple[torch.nn.Module, Any]:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"DINOv3 requires transformers: {exc}") from exc

    def _format_hf_error(err: Exception) -> str:
        msg = str(err)
        lowered = msg.lower()
        if any(token in lowered for token in ("401", "403", "unauthorized", "forbidden", "gated")):
            return (
                "Access denied. If this model is gated, run `huggingface-cli login` or set HF_TOKEN "
                "and accept the model license on Hugging Face."
            )
        if "not found" in lowered or "404" in lowered:
            return "Model not found. Check the model name and spelling."
        if "connection" in lowered or "offline" in lowered or "timeout" in lowered:
            return "Network error while fetching the model. Check connectivity or pre-download the weights."
        return msg

    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"Failed to load DINOv3 backbone '{model_name}': {_format_hf_error(exc)}") from exc
    model.eval()
    model.to(device)
    return model, processor


def _clamp_bbox(x_min: float, y_min: float, x_max: float, y_max: float, w_img: int, h_img: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0.0, min(x_min, float(w_img)))
    x2 = max(0.0, min(x_max, float(w_img)))
    y1 = max(0.0, min(y_min, float(h_img)))
    y2 = max(0.0, min(y_max, float(h_img)))
    if x2 <= x1 or y2 <= y1:
        return None
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = float(inter_x2 - inter_x1) * float(inter_y2 - inter_y1)
    area_a = float(max(0, ax2 - ax1)) * float(max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1)) * float(max(0, by2 - by1))
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return float(inter_area / denom)


def _pad_bbox(
    box: Tuple[int, int, int, int],
    pad_ratio: float,
    w_img: int,
    h_img: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = int(round(width * pad_ratio))
    pad_y = int(round(height * pad_ratio))
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w_img, x2 + pad_x)
    ny2 = min(h_img, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def _sample_background_boxes(
    rng: random.Random,
    w_img: int,
    h_img: int,
    gt_boxes: Sequence[Tuple[int, int, int, int]],
    *,
    sample_count: int,
    max_attempts: int,
    pad_ratio: float,
    iou_max: float,
    min_scale: float,
    max_scale: float,
) -> List[Tuple[int, int, int, int]]:
    if sample_count <= 0 or w_img <= 1 or h_img <= 1:
        return []
    padded = [_pad_bbox(box, pad_ratio, w_img, h_img) for box in gt_boxes]
    samples: List[Tuple[int, int, int, int]] = []
    attempts = 0
    while len(samples) < sample_count and attempts < max_attempts:
        attempts += 1
        bw = rng.uniform(min_scale, max_scale) * w_img
        bh = rng.uniform(min_scale, max_scale) * h_img
        if bw < 2.0 or bh < 2.0:
            continue
        max_x = max(0.0, w_img - bw)
        max_y = max(0.0, h_img - bh)
        if max_x <= 0.0 or max_y <= 0.0:
            continue
        x1 = rng.uniform(0.0, max_x)
        y1 = rng.uniform(0.0, max_y)
        x2 = x1 + bw
        y2 = y1 + bh
        candidate = _clamp_bbox(x1, y1, x2, y2, w_img, h_img)
        if candidate is None:
            continue
        if any(_bbox_iou(candidate, box) > iou_max for box in padded):
            continue
        if any(_bbox_iou(candidate, box) > 0.3 for box in samples):
            continue
        samples.append(candidate)
    return samples


def _encode_batch(
    clip_model: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    device: str,
    images: Sequence[Image.Image],
    *,
    normalize: bool = True,
) -> np.ndarray:
    if not images:
        return np.empty((0, 512), dtype=np.float32)
    with torch.no_grad():
        batch = torch.stack([preprocess(img) for img in images]).to(device)
        feats = clip_model.encode_image(batch)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy().astype(np.float32)


def _encode_batch_dinov3(
    model: torch.nn.Module,
    processor: Any,
    device: str,
    images: Sequence[Image.Image],
    *,
    normalize: bool = True,
) -> np.ndarray:
    if not images:
        hidden = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
        if hidden <= 0:
            hidden = int(getattr(getattr(model, "config", None), "embed_dim", 0) or 0)
        return np.empty((0, hidden or 0), dtype=np.float32)
    with torch.no_grad():
        batch = processor(images=list(images), return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        feats = getattr(outputs, "pooler_output", None)
        if feats is None:
            feats = getattr(outputs, "last_hidden_state", None)
            if feats is None:
                raise TrainingError("DINOv3 output missing pooler_output/last_hidden_state.")
            feats = feats[:, 0, :]
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy().astype(np.float32)


def _clip_aug_signature(enabled: bool) -> str:
    if not enabled:
        return "albumentations:disabled"
    return (
        "albumentations:"
        f"hflip={CLIP_AUG_HFLIP_P};vflip={CLIP_AUG_VFLIP_P};"
        f"brightness={CLIP_AUG_BRIGHTNESS_LIMIT};contrast={CLIP_AUG_CONTRAST_LIMIT};"
        f"hsv={CLIP_AUG_HUE_SHIFT}/{CLIP_AUG_SAT_SHIFT}/{CLIP_AUG_VAL_SHIFT};"
        f"gray={CLIP_AUG_GRAY_P};"
        f"safe_rotate={CLIP_AUG_SAFE_ROTATE_LIMIT}:{CLIP_AUG_SAFE_ROTATE_P};"
        f"iso={CLIP_AUG_ISO_COLOR_SHIFT}:{CLIP_AUG_ISO_INTENSITY}:{CLIP_AUG_ISO_P};"
        f"gauss={CLIP_AUG_GAUSS_VAR}:{CLIP_AUG_GAUSS_P}"
    )


def _build_clip_augmenter() -> Optional["A.Compose"]:
    if A is None:
        logger.warning("Albumentations is unavailable; CLIP training will run without augmentation.")
        return None
    return A.Compose(
        [
            A.HorizontalFlip(p=CLIP_AUG_HFLIP_P),
            A.VerticalFlip(p=CLIP_AUG_VFLIP_P),
            A.RandomBrightnessContrast(
                brightness_limit=CLIP_AUG_BRIGHTNESS_LIMIT,
                contrast_limit=CLIP_AUG_CONTRAST_LIMIT,
                p=0.3,
            ),
            A.HueSaturationValue(
                hue_shift_limit=CLIP_AUG_HUE_SHIFT,
                sat_shift_limit=CLIP_AUG_SAT_SHIFT,
                val_shift_limit=CLIP_AUG_VAL_SHIFT,
                p=0.3,
            ),
            A.ToGray(p=CLIP_AUG_GRAY_P),
            A.SafeRotate(
                limit=CLIP_AUG_SAFE_ROTATE_LIMIT,
                interpolation=1,
                border_mode=0,
                rotate_method="largest_box",
                mask_interpolation=0,
                fill=0,
                fill_mask=0,
                p=CLIP_AUG_SAFE_ROTATE_P,
            ),
            A.ISONoise(
                color_shift=CLIP_AUG_ISO_COLOR_SHIFT,
                intensity=CLIP_AUG_ISO_INTENSITY,
                p=CLIP_AUG_ISO_P,
            ),
            A.GaussNoise(
                var_limit=CLIP_AUG_GAUSS_VAR,
                p=CLIP_AUG_GAUSS_P,
            ),
        ]
    )


def _apply_augmenter(augmenter: Optional["A.Compose"], crop: Image.Image) -> Image.Image:
    if augmenter is None:
        return crop
    arr = np.asarray(crop)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]
    try:
        augmented = augmenter(image=arr)
    except Exception:
        return crop
    out = augmented.get("image", arr)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    elif out.shape[2] > 3:
        out = out[:, :, :3]
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _parse_yolo_box_line(parts: Sequence[str],
                         w_img: int,
                         h_img: int) -> Optional[Tuple[int, Tuple[int, int, int, int]]]:
    if len(parts) < 5:
        return None
    try:
        cid = int(float(parts[0]))
    except Exception:
        return None
    coords = []
    try:
        coords = [float(v) for v in parts[1:]]
    except Exception:
        return None
    if len(coords) == 4:
        x_c, y_c, w_n, h_n = coords
        x_min = (x_c - 0.5 * w_n) * w_img
        y_min = (y_c - 0.5 * h_n) * h_img
        x_max = x_min + w_n * w_img
        y_max = y_min + h_n * h_img
    else:
        if len(coords) < 6 or len(coords) % 2 != 0:
            return None
        xs = coords[0::2]
        ys = coords[1::2]
        if not xs or not ys:
            return None
        x_min = min(xs) * w_img
        y_min = min(ys) * h_img
        x_max = max(xs) * w_img
        y_max = max(ys) * h_img
    bbox = _clamp_bbox(x_min, y_min, x_max, y_max, w_img, h_img)
    if bbox is None:
        return None
    return cid, bbox


def _compute_oversample_multipliers(raw_counts: Dict[int, int]) -> Dict[int, float]:
    if not raw_counts:
        return {}
    counts = np.array([max(1, int(v)) for v in raw_counts.values()], dtype=np.float64)
    if counts.size == 0:
        return {}
    p_low = float(np.percentile(counts, CLIP_OVERSAMPLE_TARGET_PCTL_LOW))
    p_high = float(np.percentile(counts, CLIP_OVERSAMPLE_TARGET_PCTL_HIGH))
    p_low = max(p_low, 1.0)
    p_high = max(p_high, p_low)
    target = math.exp(
        (1.0 - CLIP_OVERSAMPLE_TARGET_BLEND) * math.log(p_low)
        + CLIP_OVERSAMPLE_TARGET_BLEND * math.log(p_high)
    )
    multipliers: Dict[int, float] = {}
    for cid, count in raw_counts.items():
        denom = max(1.0, float(count))
        ratio = target / denom
        mult = ratio ** CLIP_OVERSAMPLE_ALPHA
        mult = max(1.0, min(CLIP_OVERSAMPLE_MAX_MULTIPLIER, mult))
        multipliers[int(cid)] = float(mult)
    return multipliers


def _sample_repeat_count(multiplier: float, rng: random.Random) -> int:
    if not math.isfinite(multiplier) or multiplier <= 1.0:
        return 1
    base = int(math.floor(multiplier))
    frac = multiplier - base
    if frac > 0 and rng.random() < frac:
        base += 1
    return max(1, base)


def train_clip_from_yolo(
    images_path: str,
    labels_path: str,
    model_output: str,
    labelmap_output: str,
    *,
    clip_model: str = "ViT-B/32",
    encoder_type: str = "clip",
    encoder_model: Optional[str] = None,
    input_labelmap: Optional[str] = None,
    test_size: float = 0.2,
    random_seed: int = 42,
    max_iter: int = 1000,
    device: Optional[str] = None,
    batch_size: int = 64,
    min_per_class: int = 2,
    class_weight: str = "balanced",
    effective_beta: float = 0.9999,
    C: float = 1.0,
    solver: str = "saga",
    classifier_type: str = "logreg",
    mlp_hidden_sizes: Optional[str] = None,
    mlp_dropout: float = 0.1,
    mlp_epochs: int = 50,
    mlp_lr: float = 1e-3,
    mlp_weight_decay: float = 1e-4,
    mlp_label_smoothing: float = 0.05,
    mlp_loss_type: str = "ce",
    mlp_focal_gamma: float = 2.0,
    mlp_focal_alpha: float = -1.0,
    mlp_sampler: str = "balanced",
    mlp_mixup_alpha: float = 0.1,
    mlp_normalize_embeddings: bool = True,
    mlp_patience: int = 6,
    mlp_activation: str = "relu",
    mlp_layer_norm: bool = False,
    mlp_hard_mining_epochs: int = 5,
    logit_adjustment_mode: str = "none",
    logit_adjustment_inference: Optional[str] = None,
    arcface_enabled: bool = False,
    arcface_margin: float = 0.2,
    arcface_scale: float = 30.0,
    supcon_weight: float = 0.0,
    supcon_temperature: float = 0.07,
    supcon_projection_dim: int = 128,
    supcon_projection_hidden: int = 0,
    embedding_center: bool = False,
    embedding_standardize: bool = False,
    calibration_mode: str = "none",
    calibration_max_iters: int = 50,
    calibration_min_temp: float = 0.5,
    calibration_max_temp: float = 5.0,
    reuse_embeddings: bool = False,
    hard_example_mining: bool = False,
    hard_mining_misclassified_weight: float = 3.0,
    hard_mining_low_conf_weight: float = 2.0,
    hard_mining_low_conf_threshold: float = 0.65,
    hard_mining_margin_threshold: float = 0.15,
    convergence_tol: float = 1e-4,
    bg_class_count: int = 2,
    progress_cb: Optional[ProgressCallback] = None,
    should_cancel: Optional[CancelCallback] = None,
) -> TrainingArtifacts:
    """Train a CLIP+LogReg model from a YOLO-style dataset.

    Parameters mirror the CLI script, but instead of printing output this
    function returns a :class:`TrainingArtifacts` instance and optionally
    reports progress through ``progress_cb``.
    """

    if not os.path.isdir(images_path):
        raise TrainingError(f"Images folder not found: {images_path}")
    if not os.path.isdir(labels_path):
        raise TrainingError(f"Labels folder not found: {labels_path}")

    encoder_type = (encoder_type or "clip").strip().lower()
    if encoder_type not in {"clip", "dinov3"}:
        raise TrainingError(f"encoder_type_unsupported:{encoder_type}")
    encoder_model_name = (encoder_model or "").strip()
    if encoder_type == "clip":
        if encoder_model_name:
            clip_model = encoder_model_name
        encoder_model_name = clip_model
    else:
        if not encoder_model_name:
            raise TrainingError("encoder_model_required")
        clip_model = encoder_model_name

    class_weight = (class_weight or "none").lower()
    if class_weight not in {"none", "balanced", "effective"}:
        class_weight = "none"
    classifier_type = (classifier_type or "logreg").strip().lower()
    if classifier_type not in {"logreg", "mlp"}:
        classifier_type = "logreg"
    mlp_hidden_sizes_list = _parse_hidden_sizes(mlp_hidden_sizes)
    mlp_dropout = float(max(0.0, min(0.9, mlp_dropout)))
    mlp_epochs = int(max(1, mlp_epochs))
    mlp_lr = float(max(1e-6, mlp_lr))
    mlp_weight_decay = float(max(0.0, mlp_weight_decay))
    mlp_label_smoothing = float(max(0.0, min(0.3, mlp_label_smoothing)))
    mlp_loss_type = str(mlp_loss_type or "ce").strip().lower()
    if mlp_loss_type not in {"ce", "focal"}:
        mlp_loss_type = "ce"
    mlp_focal_gamma = float(max(0.0, mlp_focal_gamma))
    mlp_focal_alpha = float(mlp_focal_alpha) if mlp_focal_alpha is not None else -1.0
    if mlp_focal_alpha < 0:
        mlp_focal_alpha = None
    mlp_sampler = str(mlp_sampler or "balanced").strip().lower()
    if mlp_sampler not in {"balanced", "none", "shuffle"}:
        mlp_sampler = "balanced"
    mlp_mixup_alpha = float(max(0.0, mlp_mixup_alpha))
    if isinstance(mlp_normalize_embeddings, str):
        mlp_normalize_embeddings = mlp_normalize_embeddings.strip().lower() in {"1", "true", "yes", "on"}
    else:
        mlp_normalize_embeddings = bool(mlp_normalize_embeddings)
    mlp_patience = int(max(1, mlp_patience))
    mlp_activation = str(mlp_activation or "relu").strip().lower()
    if mlp_activation not in {"relu", "gelu"}:
        mlp_activation = "relu"
    if isinstance(mlp_layer_norm, str):
        mlp_layer_norm = mlp_layer_norm.strip().lower() in {"1", "true", "yes", "on"}
    else:
        mlp_layer_norm = bool(mlp_layer_norm)
    mlp_hard_mining_epochs = int(max(1, mlp_hard_mining_epochs))
    logit_adjustment_mode = str(logit_adjustment_mode or "none").strip().lower()
    if logit_adjustment_mode not in {"none", "train", "infer", "both"}:
        logit_adjustment_mode = "none"
    logit_adjustment_inference_override = logit_adjustment_inference is not None
    if isinstance(logit_adjustment_inference, str):
        logit_adjustment_inference_flag = logit_adjustment_inference.strip().lower() in {"1", "true", "yes", "on"}
    elif logit_adjustment_inference is None:
        logit_adjustment_inference_flag = None
    else:
        logit_adjustment_inference_flag = bool(logit_adjustment_inference)
    logit_adjustment_train = logit_adjustment_mode in {"train", "both"}
    if logit_adjustment_inference_flag is None:
        logit_adjustment_inference_flag = logit_adjustment_mode in {"infer", "both"}
    if isinstance(arcface_enabled, str):
        arcface_enabled = arcface_enabled.strip().lower() in {"1", "true", "yes", "on"}
    else:
        arcface_enabled = bool(arcface_enabled)
    arcface_margin = float(max(0.0, arcface_margin))
    arcface_scale = float(max(1.0, arcface_scale))
    supcon_weight = float(max(0.0, supcon_weight))
    supcon_temperature = float(max(1e-4, supcon_temperature))
    supcon_projection_dim = int(max(0, supcon_projection_dim))
    supcon_projection_hidden = int(max(0, supcon_projection_hidden))
    if classifier_type != "mlp":
        if arcface_enabled:
            logger.warning("ArcFace is only supported for MLP heads; disabling ArcFace.")
        arcface_enabled = False
        supcon_weight = 0.0
        if logit_adjustment_train:
            logger.warning("Logit adjustment during training is only supported for MLP heads; disabling train-time adjustment.")
            logit_adjustment_train = False
            if logit_adjustment_mode == "train":
                logit_adjustment_mode = "none"
            elif logit_adjustment_mode == "both":
                logit_adjustment_mode = "infer"
            if not logit_adjustment_inference_override:
                logit_adjustment_inference_flag = logit_adjustment_mode in {"infer", "both"}
    if arcface_enabled and (logit_adjustment_train or logit_adjustment_inference_flag):
        logger.warning("ArcFace is incompatible with logit adjustment; disabling logit adjustment.")
        logit_adjustment_train = False
        logit_adjustment_inference_flag = False
        logit_adjustment_mode = "none"
    if arcface_enabled:
        if mlp_loss_type != "ce":
            logger.warning("ArcFace requires CE loss; switching mlp_loss_type to 'ce'.")
            mlp_loss_type = "ce"
        if mlp_label_smoothing > 0:
            logger.warning("ArcFace requires hard targets; disabling label smoothing.")
            mlp_label_smoothing = 0.0
        if mlp_mixup_alpha > 0:
            logger.warning("ArcFace requires hard targets; disabling mixup.")
            mlp_mixup_alpha = 0.0
    if supcon_weight > 0 and mlp_mixup_alpha > 0:
        logger.warning("SupCon with mixup is unsupported; disabling mixup.")
        mlp_mixup_alpha = 0.0
    if isinstance(embedding_center, str):
        embedding_center = embedding_center.strip().lower() in {"1", "true", "yes", "on"}
    else:
        embedding_center = bool(embedding_center)
    if isinstance(embedding_standardize, str):
        embedding_standardize = embedding_standardize.strip().lower() in {"1", "true", "yes", "on"}
    else:
        embedding_standardize = bool(embedding_standardize)
    if embedding_standardize:
        embedding_center = True
    calibration_mode = str(calibration_mode or "none").strip().lower()
    if calibration_mode not in {"none", "temperature"}:
        calibration_mode = "none"
    calibration_max_iters = int(max(1, calibration_max_iters))
    calibration_min_temp = float(max(1e-3, calibration_min_temp))
    calibration_max_temp = float(max(calibration_min_temp, calibration_max_temp))

    hard_mining_misclassified_weight = float(max(1.0, hard_mining_misclassified_weight))
    hard_mining_low_conf_weight = float(max(1.0, hard_mining_low_conf_weight))
    hard_mining_low_conf_threshold = float(max(0.0, min(0.9999, hard_mining_low_conf_threshold)))
    hard_mining_margin_threshold = float(max(0.0, hard_mining_margin_threshold))
    convergence_tol = float(max(1e-8, convergence_tol))
    bg_class_count = max(1, min(10, int(bg_class_count)))
    effective_beta = float(max(0.5, min(0.99999, effective_beta)))
    class_weight = str(class_weight or "none").strip().lower()
    if class_weight not in {"balanced", "none", "effective"}:
        class_weight = "none"

    bg_samples_per_image = 3
    bg_max_ratio = 0.4
    bg_pad_ratio = 0.1
    bg_iou_max = 0.01
    bg_min_scale = 0.08
    bg_max_scale = 0.4
    bg_max_attempts = 30
    bg_policy = {
        "samples_per_image": bg_samples_per_image,
        "max_ratio": bg_max_ratio,
        "pad_ratio": bg_pad_ratio,
        "iou_max": bg_iou_max,
        "min_scale": bg_min_scale,
        "max_scale": bg_max_scale,
        "max_attempts": bg_max_attempts,
    }
    bg_policy_signature = (
        f"samples={bg_samples_per_image};max_ratio={bg_max_ratio};pad={bg_pad_ratio};"
        f"iou={bg_iou_max};scale={bg_min_scale}-{bg_max_scale};attempts={bg_max_attempts}"
    )
    oversample_policy = {
        "target_percentile_low": CLIP_OVERSAMPLE_TARGET_PCTL_LOW,
        "target_percentile_high": CLIP_OVERSAMPLE_TARGET_PCTL_HIGH,
        "target_blend": CLIP_OVERSAMPLE_TARGET_BLEND,
        "alpha": CLIP_OVERSAMPLE_ALPHA,
        "max_multiplier": CLIP_OVERSAMPLE_MAX_MULTIPLIER,
    }
    oversample_policy_signature = (
        f"pctl={CLIP_OVERSAMPLE_TARGET_PCTL_LOW}-{CLIP_OVERSAMPLE_TARGET_PCTL_HIGH};"
        f"blend={CLIP_OVERSAMPLE_TARGET_BLEND};alpha={CLIP_OVERSAMPLE_ALPHA};"
        f"max_mult={CLIP_OVERSAMPLE_MAX_MULTIPLIER}"
    )
    rng = random.Random(int(random_seed))

    # Prepare paths early to fail fast on unwritable destinations.
    model_dir = os.path.dirname(os.path.abspath(model_output)) or "."
    labelmap_dir = os.path.dirname(os.path.abspath(labelmap_output)) or "."
    for path in {model_dir, labelmap_dir}:
        if path and not os.path.isdir(path):
            raise TrainingError(f"Output directory does not exist: {path}")

    _safe_progress(progress_cb, 0.0, "Loading configuration ...")
    phase_timings: Dict[str, float] = {}
    total_start = time.perf_counter()

    def _check_cancel() -> None:
        if should_cancel and should_cancel():
            raise TrainingError("cancelled")

    _check_cancel()
    labelmap_list = _load_labelmap(input_labelmap)
    labelmap_hash: Optional[str] = None
    if input_labelmap:
        try:
            lm_path = Path(input_labelmap)
            if lm_path.exists() and lm_path.is_file():
                labelmap_hash = hashlib.sha256(lm_path.read_bytes()).hexdigest()
        except Exception:
            labelmap_hash = None
    resolved_device = _resolve_device(device)
    normalize_embeddings = True
    if classifier_type == "mlp":
        normalize_embeddings = bool(mlp_normalize_embeddings)
    if encoder_type == "clip":
        clip_net, preprocess = _load_clip(clip_model, resolved_device)
        def encode_batch(images: Sequence[Image.Image]) -> np.ndarray:
            return _encode_batch(clip_net, preprocess, resolved_device, images, normalize=normalize_embeddings)
    else:
        dino_model, dino_processor = _load_dinov3(encoder_model_name, resolved_device)
        def encode_batch(images: Sequence[Image.Image]) -> np.ndarray:
            return _encode_batch_dinov3(dino_model, dino_processor, resolved_device, images, normalize=normalize_embeddings)
    augmenter = _build_clip_augmenter()
    aug_enabled = augmenter is not None
    aug_policy_signature = _clip_aug_signature(aug_enabled)
    aug_policy = {
        "enabled": aug_enabled,
        "horizontal_flip_p": CLIP_AUG_HFLIP_P,
        "vertical_flip_p": CLIP_AUG_VFLIP_P,
        "brightness_limit": CLIP_AUG_BRIGHTNESS_LIMIT,
        "contrast_limit": CLIP_AUG_CONTRAST_LIMIT,
        "hue_shift_limit": CLIP_AUG_HUE_SHIFT,
        "sat_shift_limit": CLIP_AUG_SAT_SHIFT,
        "val_shift_limit": CLIP_AUG_VAL_SHIFT,
        "to_gray_p": CLIP_AUG_GRAY_P,
        "safe_rotate_limit": CLIP_AUG_SAFE_ROTATE_LIMIT,
        "safe_rotate_p": CLIP_AUG_SAFE_ROTATE_P,
        "iso_color_shift": CLIP_AUG_ISO_COLOR_SHIFT,
        "iso_intensity": CLIP_AUG_ISO_INTENSITY,
        "iso_p": CLIP_AUG_ISO_P,
        "gauss_var_limit": CLIP_AUG_GAUSS_VAR,
        "gauss_p": CLIP_AUG_GAUSS_P,
    }

    cache_signature = None
    cache_payload: Optional[Dict[str, object]] = None
    using_cached_embeddings = False
    should_cleanup_chunks = True
    cache_persisted = False

    if reuse_embeddings:
        _ensure_cache_root()
        cache_signature = _compute_dataset_signature(
            images_path,
            labels_path,
            clip_model,
            encoder_type=encoder_type,
            encoder_model=encoder_model_name,
            bg_class_count=bg_class_count,
            labelmap_path=input_labelmap,
            bg_policy=bg_policy_signature,
            aug_policy=aug_policy_signature,
            oversample_policy=oversample_policy_signature,
            embed_norm=normalize_embeddings,
        )
        _check_cancel()
        cache_payload = _load_cached_embeddings(cache_signature)
        if cache_payload:
            using_cached_embeddings = True
            cache_persisted = True
            should_cleanup_chunks = False
            _safe_progress(progress_cb, 0.02, f"Reusing cached embeddings (signature {cache_signature[:12]})")
            logger.info("Embedding cache hit for signature %s", cache_signature)
        else:
            _safe_progress(progress_cb, 0.02, "No cached embeddings found; generating new embeddings ...")
            logger.info("Embedding cache miss for signature %s", cache_signature)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    scan_start = time.perf_counter()
    image_files: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(images_path):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() not in valid_exts:
                continue
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, images_path)
            image_files.append(rel_path)
    image_files.sort()
    phase_timings["scan"] = time.perf_counter() - scan_start
    _check_cancel()
    if not image_files:
        raise TrainingError("No supported images found in the provided folder.")

    embed_start = time.perf_counter()
    chunk_records: List[Tuple[str, int, int]] = []
    y_class_names: List[str]
    y_numeric: List[int]
    groups: List[str]
    encountered_cids: set[int]
    background_classes: List[str] = []

    if using_cached_embeddings and cache_payload:
        _check_cancel()
        _safe_progress(progress_cb, 0.05, f"Found cached embeddings for signature {cache_signature[:8]}…")
        chunk_dir = Path(cache_payload["chunk_dir"])
        chunk_records = list(cache_payload["chunk_records"])
        y_class_names = [str(v) for v in cache_payload["y_class_names"]]
        y_numeric = [int(v) for v in cache_payload["y_numeric"]]
        groups = [str(v) for v in cache_payload["groups"]]
        encountered_cids = {int(v) for v in cache_payload["encountered_cids"]}
        background_classes = [str(v) for v in cache_payload.get("background_classes") or []]
        should_cleanup_chunks = False
    else:
        _safe_progress(progress_cb, 0.05, f"Found {len(image_files)} candidate images.")
        _safe_progress(progress_cb, 0.08, f"Encoding {encoder_type} embeddings on {resolved_device} (batch size={batch_size}) ...")

        if reuse_embeddings and cache_signature:
            cache_dir = EMBED_CACHE_ROOT / cache_signature
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            chunk_dir = cache_dir
            should_cleanup_chunks = False
        else:
            chunk_dir = Path(tempfile.mkdtemp(prefix="clip_chunks_"))

        chunk_records = []
        y_class_names = []
        y_numeric = []
        groups = []
        encountered_cids = set()

        batch_crops: List[Image.Image] = []
        batch_meta: List[Tuple[str, int, str]] = []  # (class_name, cid, group)
        bg_embeddings: List[np.ndarray] = []
        bg_groups: List[str] = []
        bg_crop_batch: List[Image.Image] = []
        bg_group_batch: List[str] = []

    def flush_batch() -> None:
        nonlocal batch_crops, batch_meta
        if not batch_crops:
            return
        _check_cancel()
        embs = encode_batch(batch_crops)
        chunk_start = len(y_class_names)
        chunk_path = chunk_dir / f"chunk_{len(chunk_records):06d}.npy"
        np.save(chunk_path, embs, allow_pickle=False)
        for cls_name, cid, grp in batch_meta:
            y_class_names.append(cls_name)
            y_numeric.append(cid)
            groups.append(grp)
        chunk_records.append((str(chunk_path), chunk_start, len(embs)))
        batch_crops.clear()
        batch_meta.clear()

    def append_embeddings(
        embs: np.ndarray,
        class_names: List[str],
        cids: List[int],
        group_names: List[str],
    ) -> None:
        if embs.size == 0:
            return
        if len(class_names) != len(embs) or len(cids) != len(embs) or len(group_names) != len(embs):
            raise TrainingError("Embedding metadata length mismatch.")
        chunk_start = len(y_class_names)
        chunk_path = chunk_dir / f"chunk_{len(chunk_records):06d}.npy"
        np.save(chunk_path, embs, allow_pickle=False)
        y_class_names.extend(class_names)
        y_numeric.extend(cids)
        groups.extend(group_names)
        chunk_records.append((str(chunk_path), chunk_start, len(embs)))

    def flush_bg_batch() -> None:
        nonlocal bg_crop_batch, bg_group_batch
        if not bg_crop_batch:
            return
        _check_cancel()
        embs = encode_batch(bg_crop_batch)
        bg_embeddings.append(embs)
        bg_groups.extend(bg_group_batch)
        bg_crop_batch.clear()
        bg_group_batch.clear()

    total_pos = 0
    total_valid = 0
    raw_counts: Counter[int] = Counter()
    oversample_multipliers: Dict[int, float] = {}
    label_map: Dict[str, Optional[str]] = {}

    if not using_cached_embeddings:
        label_exts = [".txt", ".TXT"]
        for img_rel in image_files:
            base = os.path.splitext(img_rel)[0]
            label_file = None
            for ext in label_exts:
                candidate = os.path.join(labels_path, base + ext)
                if os.path.isfile(candidate):
                    label_file = candidate
                    break
            label_map[img_rel] = label_file
            if label_file is None:
                continue
            try:
                lines = open(label_file, "r", encoding="utf-8").read().strip().splitlines()
            except Exception:
                continue
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                raw_counts[cid] += 1
        oversample_multipliers = _compute_oversample_multipliers(raw_counts)

    if not using_cached_embeddings:
        label_exts = [".txt", ".TXT"]
        for idx, img_rel in enumerate(image_files, start=1):
            base = os.path.splitext(img_rel)[0]
            label_file = label_map.get(img_rel)
            if not label_file:
                continue
            img_path = os.path.join(images_path, img_rel)
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as exc:
                raise TrainingError(f"Failed to open image '{img_rel}': {exc}") from exc
            w_img, h_img = pil_img.size

            try:
                lines = open(label_file, "r", encoding="utf-8").read().strip().splitlines()
            except Exception as exc:
                raise TrainingError(f"Failed to read label file '{label_file}': {exc}") from exc

            gt_boxes: List[Tuple[int, int, int, int]] = []
            for ln in lines:
                _check_cancel()
                parts = ln.split()
                parsed = _parse_yolo_box_line(parts, w_img, h_img)
                if not parsed:
                    continue
                cid, bbox = parsed
                X1, Y1, X2, Y2 = bbox
                gt_boxes.append(bbox)
                try:
                    base_crop = pil_img.crop((X1, Y1, X2, Y2))
                except Exception:
                    continue

                if labelmap_list and 0 <= cid < len(labelmap_list):
                    cls_name = str(labelmap_list[cid])
                else:
                    cls_name = f"class_{cid}"

                encountered_cids.add(cid)
                repeat = _sample_repeat_count(oversample_multipliers.get(cid, 1.0), rng)
                for _ in range(repeat):
                    aug_crop = _apply_augmenter(augmenter, base_crop)
                    batch_crops.append(aug_crop)
                    batch_meta.append((cls_name, cid, base))
                    total_pos += 1
                    if len(batch_crops) >= batch_size:
                        flush_batch()

            if bg_class_count > 0:
                bg_boxes = _sample_background_boxes(
                    rng,
                    w_img,
                    h_img,
                    gt_boxes,
                    sample_count=bg_samples_per_image,
                    max_attempts=bg_max_attempts,
                    pad_ratio=bg_pad_ratio,
                    iou_max=bg_iou_max,
                    min_scale=bg_min_scale,
                    max_scale=bg_max_scale,
                )
                for bg_box in bg_boxes:
                    try:
                        crop = pil_img.crop(bg_box)
                    except Exception:
                        continue
                    crop = _apply_augmenter(augmenter, crop)
                    bg_crop_batch.append(crop)
                    bg_group_batch.append(base)
                    if len(bg_crop_batch) >= batch_size:
                        flush_bg_batch()

            if idx % 25 == 0 or idx == len(image_files):
                frac = idx / max(1, len(image_files))
                _safe_progress(progress_cb, 0.05 + 0.30 * frac, f"Processed {idx}/{len(image_files)} images (accumulated crops={total_pos}) ...")

        flush_batch()
        flush_bg_batch()
        _check_cancel()

        total_valid = total_pos
        if bg_embeddings:
            bg_feats = np.concatenate(bg_embeddings, axis=0)
            bg_total = bg_feats.shape[0]
            if total_pos > 0 and bg_max_ratio > 0.0:
                max_bg = int(total_pos * (bg_max_ratio / max(1e-6, (1.0 - bg_max_ratio))))
                if max_bg > 0 and bg_total > max_bg:
                    sel = rng.sample(range(bg_total), max_bg)
                    bg_feats = bg_feats[sel]
                    bg_groups = [bg_groups[i] for i in sel]
                    bg_total = max_bg

            min_cluster = max(2, min_per_class)
            max_k = max(1, bg_total // max(1, min_cluster))
            bg_k = min(bg_class_count, max_k) if bg_total > 0 else 0
            if bg_k >= 1 and bg_total >= bg_k:
                if bg_k == 1:
                    labels = np.zeros(bg_total, dtype=int)
                else:
                    try:
                        kmeans = MiniBatchKMeans(
                            n_clusters=bg_k,
                            random_state=int(random_seed),
                            batch_size=max(128, batch_size),
                            n_init=5,
                        )
                        labels = kmeans.fit_predict(bg_feats)
                    except Exception:
                        labels = np.zeros(bg_total, dtype=int)
                        bg_k = 1

                max_dataset_cid = max(encountered_cids) if encountered_cids else -1
                if labelmap_list:
                    max_dataset_cid = max(max_dataset_cid, len(labelmap_list) - 1)
                bg_class_names = [f"__bg_{i}" for i in range(bg_k)]
                background_classes = list(bg_class_names)
                bg_cids = [max_dataset_cid + 1 + i for i in range(bg_k)]
                bg_names = [bg_class_names[int(idx)] for idx in labels]
                bg_ids = [bg_cids[int(idx)] for idx in labels]
                append_embeddings(bg_feats, bg_names, bg_ids, bg_groups)
                total_valid += len(bg_names)
    else:
        total_valid = len(y_numeric)

    if not encountered_cids:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        raise TrainingError("Not enough labelled boxes to train a model.")
    if total_valid == 0 or len(y_numeric) < 2:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        raise TrainingError("Not enough labelled boxes to train a model.")

    y_numeric_np = np.array(y_numeric, dtype=int)
    groups_np = np.array(groups)
    y_class_names_arr = np.array(y_class_names, dtype=object)
    if not background_classes:
        background_classes = sorted({name for name in y_class_names if name.startswith("__bg_")})

    counts = Counter(y_numeric_np.tolist())
    keep_mask = np.array([
        (raw_counts.get(c, counts[c]) >= max(1, min_per_class))
        for c in y_numeric_np
    ])
    if not np.any(keep_mask):
        shutil.rmtree(chunk_dir, ignore_errors=True)
        raise TrainingError("Not enough samples after filtering low-frequency classes.")

    kept_indices = np.flatnonzero(keep_mask)
    new_index_map = np.full(len(keep_mask), -1, dtype=np.int64)
    new_index_map[kept_indices] = np.arange(kept_indices.size)

    y_numeric_np = y_numeric_np[keep_mask]
    groups_np = groups_np[keep_mask]
    y_class_names_arr = y_class_names_arr[keep_mask]

    _safe_progress(progress_cb, 0.38, f"Retained {len(y_numeric_np)} samples across {len(set(y_class_names_arr))} classes; building train/test split ...")

    unique_groups = np.unique(groups_np)
    use_group_split = len(unique_groups) >= 2
    if use_group_split:
        try:
            _check_cancel()
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
            train_idx, test_idx = next(splitter.split(np.arange(len(y_numeric_np)), y_class_names_arr, groups=groups_np))
        except Exception:
            use_group_split = False
    if not use_group_split:
        stratify = y_class_names_arr if len(set(y_class_names_arr)) > 1 else None
        _check_cancel()
        train_idx, test_idx = train_test_split(
            np.arange(len(y_numeric_np)),
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify,
        )

    train_size = len(train_idx)
    test_size = len(test_idx)

    train_positions = np.full(len(y_numeric_np), -1, dtype=np.int64)
    train_positions[train_idx] = np.arange(train_size, dtype=np.int64)
    test_positions = np.full(len(y_numeric_np), -1, dtype=np.int64)
    test_positions[test_idx] = np.arange(test_size, dtype=np.int64)

    train_memmap_path = os.path.join(chunk_dir, "train_embeddings.dat")
    test_memmap_path = os.path.join(chunk_dir, "test_embeddings.dat")
    embedding_dim: Optional[int] = None
    for chunk_path, _, _ in chunk_records:
        try:
            probe = np.load(chunk_path, mmap_mode="r")
        except Exception as exc:
            raise TrainingError(f"Failed to load embedding chunk '{chunk_path}': {exc}") from exc
        if probe.ndim != 2 or probe.shape[1] <= 0:
            raise TrainingError(f"Invalid embedding chunk shape for '{chunk_path}': {probe.shape}")
        embedding_dim = int(probe.shape[1])
        break
    if embedding_dim is None:
        raise TrainingError("No embeddings available to build train/test split.")
    X_train_mm = np.memmap(train_memmap_path, dtype=np.float64, mode="w+", shape=(train_size, embedding_dim))
    X_test_mm = np.memmap(test_memmap_path, dtype=np.float64, mode="w+", shape=(test_size, embedding_dim))

    for chunk_path, old_start, count in chunk_records:
        _check_cancel()
        chunk = np.load(chunk_path)
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != embedding_dim:
            raise TrainingError(
                f"Embedding dimension mismatch for '{chunk_path}': got {chunk.shape}, expected (*, {embedding_dim})"
            )
        old_indices = np.arange(old_start, old_start + count)
        new_indices = new_index_map[old_indices]
        valid_mask = new_indices >= 0
        if not np.any(valid_mask):
            if not reuse_embeddings:
                os.remove(chunk_path)
            continue
        chunk = chunk[valid_mask]
        new_indices = new_indices[valid_mask]
        train_dest = train_positions[new_indices]
        train_mask = train_dest >= 0
        if np.any(train_mask):
            X_train_mm[train_dest[train_mask]] = chunk[train_mask]
        test_dest = test_positions[new_indices]
        test_mask = test_dest >= 0
        if np.any(test_mask):
            X_test_mm[test_dest[test_mask]] = chunk[test_mask]
        if not reuse_embeddings:
            os.remove(chunk_path)

    X_train_mm.flush()
    X_test_mm.flush()
    X_train = np.asarray(X_train_mm)
    X_test = np.asarray(X_test_mm)
    phase_timings["embed"] = time.perf_counter() - embed_start
    if classifier_type == "mlp" and mlp_normalize_embeddings:
        _safe_progress(progress_cb, 0.40, "Normalizing embeddings ...")
        train_norms = np.linalg.norm(X_train, axis=1, keepdims=True)
        train_norms = np.maximum(train_norms, 1e-12)
        X_train = X_train / train_norms
        if X_test.size:
            test_norms = np.linalg.norm(X_test, axis=1, keepdims=True)
            test_norms = np.maximum(test_norms, 1e-12)
            X_test = X_test / test_norms
    embedding_center_values: Optional[np.ndarray] = None
    embedding_std_values: Optional[np.ndarray] = None
    if embedding_center or embedding_standardize:
        _safe_progress(progress_cb, 0.41, "Centering embeddings ...")
        embedding_center_values = np.mean(X_train, axis=0).astype(np.float32)
        X_train = X_train - embedding_center_values
        if X_test.size:
            X_test = X_test - embedding_center_values
        if embedding_standardize:
            _safe_progress(progress_cb, 0.42, "Standardizing embeddings ...")
            embedding_std_values = np.std(X_train, axis=0).astype(np.float32)
            embedding_std_values = np.maximum(embedding_std_values, 1e-6)
            X_train = X_train / embedding_std_values
            if X_test.size:
                X_test = X_test / embedding_std_values
    y_train = y_class_names_arr[train_idx]
    y_test = y_class_names_arr[test_idx]

    def _calibrate_temperature(
        logits: np.ndarray,
        labels: Sequence[int],
        num_classes: int,
    ) -> Optional[float]:
        if calibration_mode != "temperature":
            return None
        if logits is None or not isinstance(logits, np.ndarray):
            return None
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != num_classes:
            return None
        labels_arr = np.asarray(labels, dtype=int)
        if labels_arr.shape[0] != logits.shape[0]:
            return None
        calib_start = time.perf_counter()
        temps = np.linspace(calibration_min_temp, calibration_max_temp, num=max(3, calibration_max_iters), dtype=np.float64)
        best_temp = 1.0
        best_loss = None
        for temp in temps:
            scaled = logits / float(temp)
            max_logit = np.max(scaled, axis=1, keepdims=True)
            exp_logits = np.exp(scaled - max_logit)
            denom = exp_logits.sum(axis=1, keepdims=True) + 1e-8
            probs = exp_logits / denom
            try:
                loss = float(log_loss(labels_arr, probs, labels=list(range(num_classes))))
            except Exception:
                continue
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_temp = float(temp)
        phase_timings["calibration"] = time.perf_counter() - calib_start
        return best_temp

    train_start = time.perf_counter()
    convergence_trace: List[Dict[str, Optional[float]]] = []
    converged = False
    accuracy = 0.0
    calibration_temperature: Optional[float] = None
    report = ""
    matrix: List[List[int]] = []
    label_list: List[str] = []
    class_weight_param = "balanced" if class_weight == "balanced" else None
    classifier_solver = solver
    clf: Optional[object] = None
    labels_for_cm: List[str] = []
    report_dict: Dict[str, Any] = {}
    per_class_metrics: List[Dict[str, Optional[float]]] = []
    iterations_run = 0

    try:
        logit_adjustment_vec: Optional[np.ndarray] = None
        if classifier_type == "mlp":
            _safe_progress(progress_cb, 0.45, "Training MLP head ...")
            classes_list = sorted(set(map(str, y_train)) | set(map(str, y_test)))
            class_to_idx = {name: idx for idx, name in enumerate(classes_list)}
            y_train_idx = np.array([class_to_idx[str(name)] for name in y_train], dtype=np.int64)
            y_test_idx = np.array([class_to_idx[str(name)] for name in y_test], dtype=np.int64) if y_test.size else None

            counts = Counter(y_train_idx.tolist())
            class_weights = None
            weights = None
            effective_weights = None
            if class_weight == "effective" and counts:
                effective_weights = _effective_number_weights(counts, effective_beta)
                weights = np.array(
                    [effective_weights.get(idx, 1.0) for idx in range(len(classes_list))],
                    dtype=np.float32,
                )
                class_weights = torch.tensor(weights, device=resolved_device, dtype=torch.float32)
            elif class_weight_param == "balanced" and counts:
                n_classes = len(classes_list)
                weights = np.ones(n_classes, dtype=np.float32)
                total = float(sum(counts.values()))
                for idx, count in counts.items():
                    if count:
                        weights[idx] = total / (n_classes * float(count))
                class_weights = torch.tensor(weights, device=resolved_device, dtype=torch.float32)
            logit_adjustment_vec = None
            logit_adjustment_tensor = None
            if logit_adjustment_train or logit_adjustment_inference_flag:
                logit_adjustment_vec = _logit_adjustment_from_counts(counts, len(classes_list))
                if logit_adjustment_vec is not None:
                    logit_adjustment_tensor = torch.tensor(
                        logit_adjustment_vec,
                        device=resolved_device,
                        dtype=torch.float32,
                    )

            input_dim = int(X_train.shape[1]) if X_train.size else 0
            if input_dim <= 0:
                raise TrainingError("No embeddings available to train MLP classifier.")
            if len(classes_list) < 2:
                raise TrainingError("Need at least two classes to train MLP classifier.")

            hidden_sizes = list(mlp_hidden_sizes_list or [256])
            layer_dims = [input_dim] + hidden_sizes

            class _MLPHead(torch.nn.Module):
                def __init__(
                    self,
                    dims: List[int],
                    output_dim: int,
                    *,
                    activation: str,
                    dropout: float,
                    layer_norm: bool,
                    arcface: bool,
                    arcface_scale: float,
                ) -> None:
                    super().__init__()
                    self.hidden_linears = torch.nn.ModuleList()
                    self.hidden_norms = torch.nn.ModuleList()
                    self.hidden_dropouts = torch.nn.ModuleList()
                    self.hidden_activations: List[str] = []
                    for idx in range(len(dims) - 1):
                        linear = torch.nn.Linear(dims[idx], dims[idx + 1])
                        self.hidden_linears.append(linear)
                        if layer_norm:
                            self.hidden_norms.append(torch.nn.LayerNorm(dims[idx + 1]))
                        else:
                            self.hidden_norms.append(torch.nn.Identity())
                        self.hidden_activations.append(activation)
                        if dropout > 0:
                            self.hidden_dropouts.append(torch.nn.Dropout(dropout))
                        else:
                            self.hidden_dropouts.append(torch.nn.Identity())
                    self.output = torch.nn.Linear(dims[-1], output_dim, bias=not arcface)
                    self.arcface_enabled = bool(arcface)
                    self.arcface_scale = float(arcface_scale)

                def forward_features(self, x: torch.Tensor) -> torch.Tensor:
                    out = x
                    for linear, norm, act, drop in zip(
                        self.hidden_linears,
                        self.hidden_norms,
                        self.hidden_activations,
                        self.hidden_dropouts,
                    ):
                        out = linear(out)
                        out = norm(out)
                        if act == "gelu":
                            out = torch.nn.functional.gelu(out)
                        elif act == "relu":
                            out = torch.nn.functional.relu(out)
                        out = drop(out)
                    return out

                def forward_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    feats = self.forward_features(x)
                    if self.arcface_enabled:
                        feats_norm = torch.nn.functional.normalize(feats, dim=1)
                        weight_norm = torch.nn.functional.normalize(self.output.weight, dim=1)
                        logits = feats_norm @ weight_norm.t()
                        logits = logits * self.arcface_scale
                        return logits, feats
                    logits = self.output(feats)
                    return logits, feats

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    logits, _ = self.forward_logits(x)
                    return logits

            model = _MLPHead(
                layer_dims,
                len(classes_list),
                activation=mlp_activation,
                dropout=mlp_dropout,
                layer_norm=mlp_layer_norm,
                arcface=arcface_enabled,
                arcface_scale=arcface_scale,
            ).to(resolved_device)
            projection_head: Optional[torch.nn.Module] = None
            if supcon_weight > 0.0:
                proj_in = layer_dims[-1]
                if supcon_projection_dim <= 0:
                    supcon_projection_dim = proj_in
                if supcon_projection_hidden > 0:
                    projection_head = torch.nn.Sequential(
                        torch.nn.Linear(proj_in, supcon_projection_hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(supcon_projection_hidden, supcon_projection_dim),
                    ).to(resolved_device)
                else:
                    projection_head = torch.nn.Linear(proj_in, supcon_projection_dim).to(resolved_device)
            optimizer_params = list(model.parameters())
            if projection_head is not None:
                optimizer_params += list(projection_head.parameters())
            optimizer = torch.optim.AdamW(optimizer_params, lr=mlp_lr, weight_decay=mlp_weight_decay)
            use_soft_targets = mlp_label_smoothing > 0 or mlp_mixup_alpha > 0 or mlp_loss_type == "focal"

            def predict_logits(data: np.ndarray) -> np.ndarray:
                model.eval()
                with torch.no_grad():
                    tensor = torch.tensor(data, dtype=torch.float32, device=resolved_device)
                    logits = model(tensor)
                return logits.cpu().numpy()

            def predict_probs(data: np.ndarray) -> np.ndarray:
                logits = predict_logits(data)
                if logit_adjustment_inference_flag and logit_adjustment_vec is not None:
                    try:
                        adj = np.asarray(logit_adjustment_vec, dtype=np.float32).reshape(1, -1)
                        if adj.shape[1] == logits.shape[1]:
                            logits = logits + adj
                    except Exception:
                        pass
                max_logit = np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits - max_logit)
                denom = exp_logits.sum(axis=1, keepdims=True) + 1e-8
                return (exp_logits / denom).astype(np.float32)

            def _build_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
                targets = torch.zeros((labels.shape[0], num_classes), device=labels.device, dtype=torch.float32)
                targets.scatter_(1, labels.unsqueeze(1), 1.0)
                if mlp_label_smoothing > 0:
                    smooth = mlp_label_smoothing
                    targets = targets * (1.0 - smooth) + smooth / float(num_classes)
                return targets

            def _compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                loss = -(targets * log_probs).sum(dim=1)
                if class_weights is not None:
                    weight_vec = (targets * class_weights).sum(dim=1)
                    loss = loss * weight_vec
                if mlp_loss_type == "focal":
                    probs = torch.exp(log_probs)
                    pt = (targets * probs).sum(dim=1)
                    focal_factor = (1.0 - pt).clamp(min=0.0) ** mlp_focal_gamma
                    loss = loss * focal_factor
                    if mlp_focal_alpha is not None and mlp_focal_alpha > 0:
                        loss = loss * mlp_focal_alpha
                return loss.mean()

            arcface_cos_m = math.cos(arcface_margin) if arcface_enabled else 0.0
            arcface_sin_m = math.sin(arcface_margin) if arcface_enabled else 0.0

            def _apply_arcface_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                if not arcface_enabled or arcface_margin <= 0:
                    return logits
                if arcface_scale <= 0:
                    return logits
                cosine = logits / float(arcface_scale)
                cosine = cosine.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
                sine = torch.sqrt((1.0 - cosine ** 2).clamp(min=0.0))
                phi = cosine * arcface_cos_m - sine * arcface_sin_m
                output = cosine.clone()
                idx = torch.arange(output.shape[0], device=output.device)
                output[idx, labels] = phi[idx, labels]
                return output * float(arcface_scale)

            def _supcon_loss(features: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
                if features is None or features.ndim != 2 or features.shape[0] < 2:
                    return None
                labels = labels.view(-1, 1)
                mask = torch.eq(labels, labels.T).float()
                logits = torch.div(torch.matmul(features, features.T), float(supcon_temperature))
                logits_max, _ = torch.max(logits, dim=1, keepdim=True)
                logits = logits - logits_max.detach()
                logits_mask = torch.ones_like(mask) - torch.eye(features.shape[0], device=features.device)
                mask = mask * logits_mask
                exp_logits = torch.exp(logits) * logits_mask
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
                mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
                return -mean_log_prob_pos.mean()

            train_tensor = torch.tensor(X_train, dtype=torch.float32)
            train_labels = torch.tensor(y_train_idx, dtype=torch.long)
            sampler = None
            shuffle = False
            if mlp_sampler == "balanced" and counts:
                if weights is None:
                    weights = np.ones(len(classes_list), dtype=np.float32)
                    total = float(sum(counts.values()))
                    for idx, count in counts.items():
                        if count:
                            weights[idx] = total / (len(classes_list) * float(count))
                sample_weights = torch.tensor(weights[train_labels.numpy()], dtype=torch.float32)
                sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            elif mlp_sampler == "shuffle":
                shuffle = True
            train_loader = DataLoader(
                TensorDataset(train_tensor, train_labels),
                batch_size=min(max(8, batch_size), len(train_tensor)),
                shuffle=shuffle if sampler is None else False,
                sampler=sampler,
            )
            val_loader = None
            if y_test.size and y_test_idx is not None:
                val_tensor = torch.tensor(X_test, dtype=torch.float32)
                val_labels = torch.tensor(y_test_idx, dtype=torch.long)
                val_loader = DataLoader(
                    TensorDataset(val_tensor, val_labels),
                    batch_size=min(max(8, batch_size), len(val_tensor)),
                    shuffle=False,
                )

            best_state = None
            best_val = None
            epochs_no_improve = 0
            supcon_enabled = supcon_weight > 0.0 and projection_head is not None

            for epoch in range(1, mlp_epochs + 1):
                _check_cancel()
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(resolved_device)
                    batch_y = batch_y.to(resolved_device)
                    perm = None
                    lam = None
                    if mlp_mixup_alpha > 0:
                        lam = np.random.beta(mlp_mixup_alpha, mlp_mixup_alpha, size=batch_x.size(0)).astype("float32")
                        lam = torch.tensor(lam, device=batch_x.device).view(-1, 1)
                        perm = torch.randperm(batch_x.size(0), device=batch_x.device)
                        batch_x = lam * batch_x + (1.0 - lam) * batch_x[perm]
                    optimizer.zero_grad()
                    logits, feats = model.forward_logits(batch_x)
                    logits_for_loss = _apply_arcface_margin(logits, batch_y)
                    if logit_adjustment_train and logit_adjustment_tensor is not None:
                        logits_for_loss = logits_for_loss + logit_adjustment_tensor
                    if use_soft_targets:
                        targets = _build_targets(batch_y, logits_for_loss.shape[1])
                        if mlp_mixup_alpha > 0:
                            targets = lam * targets + (1.0 - lam) * targets[perm]
                        loss = _compute_loss(logits_for_loss, targets)
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            logits_for_loss,
                            batch_y,
                            weight=class_weights,
                            label_smoothing=mlp_label_smoothing,
                        )
                    if supcon_enabled:
                        proj_feats = feats
                        if projection_head is not None:
                            proj_feats = projection_head(feats)
                        proj_feats = torch.nn.functional.normalize(proj_feats, dim=1)
                        sup_loss = _supcon_loss(proj_feats, batch_y)
                        if sup_loss is not None:
                            loss = loss + supcon_weight * sup_loss
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item()) * batch_y.size(0)
                    preds = logits_for_loss.argmax(dim=1)
                    correct += int((preds == batch_y).sum().item())
                    total += int(batch_y.size(0))

                train_loss = running_loss / max(1, total)
                train_acc = float(correct) / max(1, total)

                val_loss = None
                val_acc = None
                if val_loader is not None:
                    model.eval()
                    val_running = 0.0
                    val_correct = 0
                    val_total = 0
                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            batch_x = batch_x.to(resolved_device)
                            batch_y = batch_y.to(resolved_device)
                            logits, _feats = model.forward_logits(batch_x)
                            logits_for_loss = _apply_arcface_margin(logits, batch_y)
                            if logit_adjustment_train and logit_adjustment_tensor is not None:
                                logits_for_loss = logits_for_loss + logit_adjustment_tensor
                            if use_soft_targets:
                                targets = _build_targets(batch_y, logits_for_loss.shape[1])
                                loss = _compute_loss(logits_for_loss, targets)
                            else:
                                loss = torch.nn.functional.cross_entropy(
                                    logits_for_loss,
                                    batch_y,
                                    weight=class_weights,
                                    label_smoothing=mlp_label_smoothing,
                                )
                            val_running += float(loss.item()) * batch_y.size(0)
                            preds = logits_for_loss.argmax(dim=1)
                            val_correct += int((preds == batch_y).sum().item())
                            val_total += int(batch_y.size(0))
                    val_loss = val_running / max(1, val_total)
                    val_acc = float(val_correct) / max(1, val_total)

                    if best_val is None or val_loss < best_val - 1e-4:
                        best_val = val_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= mlp_patience:
                            converged = True
                            break

                convergence_trace.append({
                    "iteration": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "coef_delta": None,
                    "loss_delta": None,
                })
                iterations_run = epoch
                progress_fraction = 0.45 + 0.15 * (epoch / max(1, mlp_epochs))
                status_parts = [
                    f"epoch {epoch}",
                    f"train_loss={train_loss:.4f}",
                    f"train_acc={train_acc:.3f}",
                ]
                if val_loss is not None:
                    status_parts.append(f"val_loss={val_loss:.4f}")
                if val_acc is not None:
                    status_parts.append(f"val_acc={val_acc:.3f}")
                _safe_progress(progress_cb, progress_fraction, "MLP: " + ", ".join(status_parts))

            if best_state:
                model.load_state_dict(best_state)

            if hard_example_mining:
                _safe_progress(progress_cb, 0.62, "MLP hard mining: scoring training set ...")
                proba_train = predict_probs(X_train)
                train_pred_idx = np.argmax(proba_train, axis=1)
                sample_weight = np.ones(len(y_train_idx), dtype=np.float32)

                if hard_mining_misclassified_weight > 1.0:
                    misclassified_mask = train_pred_idx != y_train_idx
                    if np.any(misclassified_mask):
                        sample_weight[misclassified_mask] = np.maximum(
                            sample_weight[misclassified_mask], hard_mining_misclassified_weight
                        )

                if hard_mining_low_conf_weight > 1.0:
                    top1 = proba_train.max(axis=1)
                    low_conf_mask = np.zeros_like(top1, dtype=bool)
                    if hard_mining_low_conf_threshold > 0.0:
                        low_conf_mask |= top1 < hard_mining_low_conf_threshold
                    if hard_mining_margin_threshold > 0.0 and proba_train.shape[1] > 1:
                        second_best = np.partition(proba_train, -2, axis=1)[:, -2]
                        margin_gap = top1 - second_best
                        low_conf_mask |= margin_gap < hard_mining_margin_threshold
                    if np.any(low_conf_mask):
                        sample_weight[low_conf_mask] = np.maximum(
                            sample_weight[low_conf_mask], hard_mining_low_conf_weight
                        )

                if not np.any(sample_weight != 1.0):
                    _safe_progress(progress_cb, 0.64, "MLP hard mining skipped — no hard samples found.")
                else:
                    sampler = WeightedRandomSampler(
                        torch.tensor(sample_weight, dtype=torch.float32),
                        num_samples=len(sample_weight),
                        replacement=True,
                    )
                    hard_loader = DataLoader(
                        TensorDataset(train_tensor, train_labels),
                        batch_size=min(max(8, batch_size), len(train_tensor)),
                        shuffle=False,
                        sampler=sampler,
                    )
                    for hard_epoch in range(1, mlp_hard_mining_epochs + 1):
                        _check_cancel()
                        model.train()
                        running_loss = 0.0
                        correct = 0
                        total = 0
                        for batch_x, batch_y in hard_loader:
                            batch_x = batch_x.to(resolved_device)
                            batch_y = batch_y.to(resolved_device)
                            perm = None
                            lam = None
                            if mlp_mixup_alpha > 0:
                                lam = np.random.beta(mlp_mixup_alpha, mlp_mixup_alpha, size=batch_x.size(0)).astype("float32")
                                lam = torch.tensor(lam, device=batch_x.device).view(-1, 1)
                                perm = torch.randperm(batch_x.size(0), device=batch_x.device)
                                batch_x = lam * batch_x + (1.0 - lam) * batch_x[perm]
                            optimizer.zero_grad()
                            logits, feats = model.forward_logits(batch_x)
                            logits_for_loss = _apply_arcface_margin(logits, batch_y)
                            if logit_adjustment_train and logit_adjustment_tensor is not None:
                                logits_for_loss = logits_for_loss + logit_adjustment_tensor
                            if use_soft_targets:
                                targets = _build_targets(batch_y, logits_for_loss.shape[1])
                                if mlp_mixup_alpha > 0:
                                    targets = lam * targets + (1.0 - lam) * targets[perm]
                                loss = _compute_loss(logits_for_loss, targets)
                            else:
                                loss = torch.nn.functional.cross_entropy(
                                    logits_for_loss,
                                    batch_y,
                                    weight=class_weights,
                                    label_smoothing=mlp_label_smoothing,
                                )
                            if supcon_enabled:
                                proj_feats = feats
                                if projection_head is not None:
                                    proj_feats = projection_head(feats)
                                proj_feats = torch.nn.functional.normalize(proj_feats, dim=1)
                                sup_loss = _supcon_loss(proj_feats, batch_y)
                                if sup_loss is not None:
                                    loss = loss + supcon_weight * sup_loss
                            loss.backward()
                            optimizer.step()
                            running_loss += float(loss.item()) * batch_y.size(0)
                            preds = logits_for_loss.argmax(dim=1)
                            correct += int((preds == batch_y).sum().item())
                            total += int(batch_y.size(0))
                        train_loss = running_loss / max(1, total)
                        train_acc = float(correct) / max(1, total)
                        iterations_run += 1
                        _safe_progress(
                            progress_cb,
                            0.62 + 0.02 * (hard_epoch / max(1, mlp_hard_mining_epochs)),
                            f"MLP hard mining epoch {hard_epoch}: loss={train_loss:.4f}, acc={train_acc:.3f}",
                        )

            classifier_solver = "mlp"
            layer_specs: List[Dict[str, Any]] = []
            try:
                for idx, linear in enumerate(model.hidden_linears):
                    layer_norm = model.hidden_norms[idx] if hasattr(model, "hidden_norms") else None
                    activation = model.hidden_activations[idx] if hasattr(model, "hidden_activations") else mlp_activation
                    layer_specs.append({
                        "linear": linear,
                        "activation": activation,
                        "layer_norm": layer_norm if isinstance(layer_norm, torch.nn.LayerNorm) else None,
                    })
                layer_specs.append({
                    "linear": model.output,
                    "activation": "linear",
                    "layer_norm": None,
                })
            except Exception:
                layer_specs = []
            clf = {
                "classifier_type": "mlp",
                "classes": classes_list,
                "layers": [],
                "normalize_embeddings": bool(mlp_normalize_embeddings),
                "embedding_center": bool(embedding_center),
                "embedding_standardize": bool(embedding_standardize),
                "embedding_center_values": embedding_center_values,
                "embedding_std_values": embedding_std_values,
                "calibration_temperature": calibration_temperature,
                "logit_adjustment": logit_adjustment_vec,
                "logit_adjustment_inference": bool(logit_adjustment_inference_flag),
                "arcface": bool(arcface_enabled),
                "arcface_margin": float(arcface_margin),
                "arcface_scale": float(arcface_scale),
            }
            for spec in layer_specs:
                linear = spec.get("linear")
                activation = spec.get("activation")
                layer_norm = spec.get("layer_norm")
                if not isinstance(linear, torch.nn.Linear):
                    continue
                layer_entry: Dict[str, Any] = {
                    "weight": linear.weight.detach().cpu().numpy(),
                    "bias": linear.bias.detach().cpu().numpy() if linear.bias is not None else np.zeros(linear.weight.shape[0], dtype=np.float32),
                    "activation": activation or "linear",
                }
                if isinstance(layer_norm, torch.nn.LayerNorm):
                    layer_entry["layer_norm_weight"] = layer_norm.weight.detach().cpu().numpy()
                    layer_entry["layer_norm_bias"] = layer_norm.bias.detach().cpu().numpy()
                    layer_entry["layer_norm_eps"] = float(layer_norm.eps)
                clf["layers"].append(layer_entry)

            _safe_progress(progress_cb, 0.65, "Evaluating classifier on validation split ...")
            if y_test.size:
                proba_val = predict_probs(X_test)
                val_pred_idx = np.argmax(proba_val, axis=1)
                y_pred = np.array([classes_list[i] for i in val_pred_idx], dtype=object)
                accuracy = float((y_pred == y_test).mean())
                report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
                report = classification_report(y_test, y_pred, zero_division=0, digits=4)
                labels_for_cm = sorted(set(y_test) | set(y_pred))
                matrix = confusion_matrix(y_test, y_pred, labels=labels_for_cm).tolist()
            else:
                proba_train = predict_probs(X_train)
                train_pred_idx = np.argmax(proba_train, axis=1)
                train_pred = np.array([classes_list[i] for i in train_pred_idx], dtype=object)
                accuracy = float((train_pred == y_train).mean())
                report_dict = classification_report(y_train, train_pred, zero_division=0, output_dict=True)
                report = classification_report(y_train, train_pred, zero_division=0, digits=4)
                labels_for_cm = sorted(set(y_train) | set(train_pred))
                matrix = confusion_matrix(y_train, train_pred, labels=labels_for_cm).tolist()

            if calibration_mode == "temperature":
                calib_logits = predict_logits(X_test if y_test.size else X_train)
                if logit_adjustment_inference_flag and logit_adjustment_vec is not None:
                    try:
                        adj = np.asarray(logit_adjustment_vec, dtype=np.float32).reshape(1, -1)
                        if adj.shape[1] == calib_logits.shape[1]:
                            calib_logits = calib_logits + adj
                    except Exception:
                        pass
                calib_labels = y_test_idx if y_test_idx is not None and y_test.size else y_train_idx
                calibration_temperature = _calibrate_temperature(
                    calib_logits,
                    calib_labels,
                    num_classes=len(classes_list),
                )
        else:
            clf = LogisticRegression(
                random_state=random_seed,
                max_iter=1,
                multi_class="auto",
                solver=solver,
                class_weight=class_weight_param,
                C=C,
                warm_start=True,
                verbose=0,
                tol=convergence_tol,
            )
            tol = convergence_tol
            prev_coef: Optional[np.ndarray] = None
            prev_loss: Optional[float] = None
            last_train_pred: Optional[np.ndarray] = None
            last_val_pred: Optional[np.ndarray] = None
            last_train_proba: Optional[np.ndarray] = None
            last_val_proba: Optional[np.ndarray] = None
            effective_sample_weight: Optional[np.ndarray] = None

            if class_weight == "effective":
                weight_map = _effective_number_weight_map(y_train, effective_beta)
                if weight_map:
                    effective_sample_weight = np.array(
                        [weight_map.get(label, 1.0) for label in y_train],
                        dtype=np.float64,
                    )

            count_by_label = Counter([str(label) for label in y_train])
            logit_adjustment_vec = None

            def _resolve_logit_adjustment_vec(classes: Sequence[str]) -> Optional[np.ndarray]:
                counts_idx: Dict[int, int] = {}
                for idx, name in enumerate(classes):
                    counts_idx[idx] = int(count_by_label.get(str(name), 0))
                return _logit_adjustment_from_counts(counts_idx, len(classes))

            def _predict_proba_with_adjustment(data: np.ndarray) -> np.ndarray:
                raw_logits = clf.decision_function(data)
                logits = np.asarray(raw_logits)
                if logits.ndim == 1:
                    logits = np.stack([np.zeros_like(logits), logits], axis=1)
                if logit_adjustment_inference_flag and logit_adjustment_vec is not None:
                    adj = np.asarray(logit_adjustment_vec, dtype=np.float32).reshape(1, -1)
                    if adj.shape[1] == logits.shape[1]:
                        logits = logits + adj
                max_logit = np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits - max_logit)
                denom = exp_logits.sum(axis=1, keepdims=True) + 1e-8
                return (exp_logits / denom).astype(np.float32)

            iteration_counter = 0

            for iteration in range(1, max_iter + 1):
                _check_cancel()
                if effective_sample_weight is not None:
                    clf.fit(X_train, y_train, sample_weight=effective_sample_weight)
                else:
                    clf.fit(X_train, y_train)
                if logit_adjustment_inference_flag and logit_adjustment_vec is None:
                    logit_adjustment_vec = _resolve_logit_adjustment_vec(clf.classes_)
                if logit_adjustment_inference_flag and logit_adjustment_vec is not None:
                    proba_train = _predict_proba_with_adjustment(X_train)
                else:
                    proba_train = clf.predict_proba(X_train)
                train_loss = float(log_loss(y_train, proba_train, labels=clf.classes_))
                train_pred = clf.predict(X_train)
                train_acc = float((train_pred == y_train).mean())
                last_train_pred = train_pred
                last_train_proba = proba_train

                val_loss: Optional[float] = None
                val_acc: Optional[float] = None
                if y_test.size:
                    if logit_adjustment_inference_flag and logit_adjustment_vec is not None:
                        proba_val = _predict_proba_with_adjustment(X_test)
                    else:
                        proba_val = clf.predict_proba(X_test)
                    val_loss = float(log_loss(y_test, proba_val, labels=clf.classes_))
                    val_pred = clf.predict(X_test)
                    val_acc = float((val_pred == y_test).mean())
                    last_val_pred = val_pred
                    last_val_proba = proba_val

                coef_delta: Optional[float] = None
                if prev_coef is not None:
                    coef_delta = float(np.linalg.norm(clf.coef_ - prev_coef))

                loss_delta: Optional[float] = None
                if prev_loss is not None:
                    loss_delta = float(abs(prev_loss - train_loss))

                convergence_trace.append({
                    "iteration": iteration,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "coef_delta": coef_delta,
                    "loss_delta": loss_delta,
                })
                iteration_counter += 1

                progress_fraction = 0.45 + 0.15 * (iteration / max(1, max_iter))
                status_parts = [
                    f"iter {iteration}",
                    f"train_loss={train_loss:.4f}",
                    f"train_acc={train_acc:.3f}",
                ]
                if val_loss is not None:
                    status_parts.append(f"val_loss={val_loss:.4f}")
                if val_acc is not None:
                    status_parts.append(f"val_acc={val_acc:.3f}")
                _safe_progress(progress_cb, progress_fraction, "Convergence: " + ", ".join(status_parts))

                prev_coef = clf.coef_.copy()
                prev_loss = train_loss

                if coef_delta is not None and coef_delta <= tol:
                    converged = True
                    break
                if loss_delta is not None and loss_delta <= tol:
                    converged = True
                    break

            iterations_run = iteration_counter
            if not converged and iterations_run >= max_iter:
                _safe_progress(progress_cb, 0.62, "Warning: LogisticRegression hit max_iter; consider increasing it or loosening tolerance.")

            if hard_example_mining and last_train_pred is not None and last_train_proba is not None:
                _safe_progress(progress_cb, 0.66, "Starting hard example mining pass ...")
                if effective_sample_weight is not None:
                    sample_weight = effective_sample_weight.copy()
                else:
                    sample_weight = np.ones(len(y_train), dtype=np.float64)

                if hard_mining_misclassified_weight > 1.0:
                    misclassified_mask = last_train_pred != y_train
                    if np.any(misclassified_mask):
                        sample_weight[misclassified_mask] = np.maximum(
                            sample_weight[misclassified_mask], hard_mining_misclassified_weight
                        )

                if hard_mining_low_conf_weight > 1.0:
                    top1 = last_train_proba.max(axis=1)
                    low_conf_mask = np.zeros_like(top1, dtype=bool)
                    if hard_mining_low_conf_threshold > 0.0:
                        low_conf_mask |= top1 < hard_mining_low_conf_threshold
                    if hard_mining_margin_threshold > 0.0 and last_train_proba.shape[1] > 1:
                        second_best = np.partition(last_train_proba, -2, axis=1)[:, -2]
                        margin_gap = top1 - second_best
                        low_conf_mask |= margin_gap < hard_mining_margin_threshold
                    if np.any(low_conf_mask):
                        sample_weight[low_conf_mask] = np.maximum(
                            sample_weight[low_conf_mask], hard_mining_low_conf_weight
                        )

                if not np.any(sample_weight != 1.0):
                    _safe_progress(progress_cb, 0.7, "Hard mining skipped — no samples met weighting criteria.")
                else:
                    extra_iters = max(5, min(200, max_iter // 5))
                    for extra_iter in range(1, extra_iters + 1):
                        _check_cancel()
                        clf.fit(X_train, y_train, sample_weight=sample_weight)
                        proba_train = clf.predict_proba(X_train)
                        train_loss = float(log_loss(y_train, proba_train, labels=clf.classes_))
                        train_pred = clf.predict(X_train)
                        train_acc = float((train_pred == y_train).mean())
                        last_train_pred = train_pred
                        last_train_proba = proba_train

                        val_loss = None
                        val_acc = None
                        if y_test.size:
                            proba_val = clf.predict_proba(X_test)
                            val_loss = float(log_loss(y_test, proba_val, labels=clf.classes_))
                            val_pred = clf.predict(X_test)
                            val_acc = float((val_pred == y_test).mean())
                            last_val_pred = val_pred
                            last_val_proba = proba_val

                        coef_delta = float(np.linalg.norm(clf.coef_ - prev_coef)) if prev_coef is not None else None
                        loss_delta = float(abs(prev_loss - train_loss)) if prev_loss is not None else None

                        iteration_counter += 1
                        convergence_trace.append({
                            "iteration": iteration_counter,
                            "train_loss": train_loss,
                            "train_accuracy": train_acc,
                            "val_loss": val_loss,
                            "val_accuracy": val_acc,
                            "coef_delta": coef_delta,
                            "loss_delta": loss_delta,
                        })

                        progress_fraction = 0.66 + 0.09 * (extra_iter / extra_iters)
                        status_parts = [
                            f"hard_iter {extra_iter}",
                            f"train_loss={train_loss:.4f}",
                            f"train_acc={train_acc:.3f}",
                        ]
                        if val_loss is not None:
                            status_parts.append(f"val_loss={val_loss:.4f}")
                        if val_acc is not None:
                            status_parts.append(f"val_acc={val_acc:.3f}")
                        _safe_progress(progress_cb, progress_fraction, "Hard mining: " + ", ".join(status_parts))

                        prev_coef = clf.coef_.copy()
                        prev_loss = train_loss

                        if coef_delta is not None and coef_delta <= tol:
                            converged = True
                            break
                        if loss_delta is not None and loss_delta <= tol:
                            converged = True
                            break

                iterations_run = iteration_counter

            _safe_progress(progress_cb, 0.65, "Evaluating classifier on validation split ...")
            if y_test.size:
                y_pred = last_val_pred if last_val_pred is not None else clf.predict(X_test)
                accuracy = float((y_pred == y_test).mean())
                report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
                report = classification_report(y_test, y_pred, zero_division=0, digits=4)
                labels_for_cm = sorted(set(y_test) | set(y_pred))
                matrix = confusion_matrix(y_test, y_pred, labels=labels_for_cm).tolist()
            else:
                train_pred_final = last_train_pred if last_train_pred is not None else clf.predict(X_train)
                accuracy = float((train_pred_final == y_train).mean())
                report_dict = classification_report(y_train, train_pred_final, zero_division=0, output_dict=True)
                report = classification_report(y_train, train_pred_final, zero_division=0, digits=4)
                labels_for_cm = sorted(set(y_train) | set(train_pred_final))
                matrix = confusion_matrix(y_train, train_pred_final, labels=labels_for_cm).tolist()

            if calibration_mode == "temperature":
                try:
                    raw_logits = clf.decision_function(X_test if y_test.size else X_train)
                    logits = np.asarray(raw_logits)
                    if logits.ndim == 1:
                        logits = np.stack([np.zeros_like(logits), logits], axis=1)
                    if logit_adjustment_inference_flag and logit_adjustment_vec is not None:
                        adj = np.asarray(logit_adjustment_vec, dtype=np.float32).reshape(1, -1)
                        if adj.shape[1] == logits.shape[1]:
                            logits = logits + adj
                    classes_list = list(clf.classes_)
                    class_to_idx = {str(name): idx for idx, name in enumerate(classes_list)}
                    labels_raw = y_test if y_test.size else y_train
                    labels_idx = [class_to_idx.get(str(name), 0) for name in labels_raw]
                    calibration_temperature = _calibrate_temperature(
                        logits,
                        labels_idx,
                        num_classes=len(classes_list),
                    )
                except Exception:
                    calibration_temperature = None

        for label in labels_for_cm:
            key = str(label)
            metrics = report_dict.get(key) or report_dict.get(label)
            if not metrics:
                continue
            support_value = metrics.get("support")
            per_class_metrics.append({
                "label": label,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1-score"),
                "support": int(support_value) if support_value is not None else None,
            })

        phase_timings["train"] = time.perf_counter() - train_start

        _safe_progress(progress_cb, 0.75, f"Saving classifier to {model_output} and labelmap to {labelmap_output} ...")
        save_start = time.perf_counter()
        _check_cancel()
        joblib.dump(clf, model_output, compress=3)

        encountered_sorted = sorted(encountered_cids)
        if labelmap_list:
            label_list = list(labelmap_list)
            max_cid = max(encountered_sorted) if encountered_sorted else -1
            while len(label_list) <= max_cid:
                label_list.append(f"unused_cid_{len(label_list)}")
        else:
            max_cid = max(encountered_sorted) if encountered_sorted else -1
            label_list = [f"class_{i}" for i in range(max_cid + 1)] if max_cid >= 0 else []
            for cid in encountered_sorted:
                if cid < len(label_list):
                    label_list[cid] = f"class_{cid}"
                else:
                    label_list.append(f"class_{cid}")

        joblib.dump(label_list, labelmap_output, compress=3)
        n_train = int(X_train.shape[0])
        n_test = int(X_test.shape[0])
        embedding_dim = int(X_train.shape[1]) if X_train.size else 0
        phase_timings["save"] = time.perf_counter() - save_start

        meta = {
            "clip_model": clip_model,
            "encoder_type": encoder_type,
            "encoder_model": clip_model,
            "device": resolved_device,
            "test_size": test_size,
            "random_seed": random_seed,
            "min_per_class": min_per_class,
            "class_weight": class_weight,
            "effective_beta": effective_beta,
            "C": C,
            "solver": classifier_solver,
            "classifier_type": classifier_type,
            "hard_example_mining": hard_example_mining,
            "hard_mining_misclassified_weight": hard_mining_misclassified_weight,
            "hard_mining_low_conf_weight": hard_mining_low_conf_weight,
            "hard_mining_low_conf_threshold": hard_mining_low_conf_threshold,
            "hard_mining_margin_threshold": hard_mining_margin_threshold,
            "convergence_tol": convergence_tol,
            "mlp_hidden_sizes": list(mlp_hidden_sizes_list),
            "mlp_dropout": mlp_dropout,
            "mlp_epochs": mlp_epochs,
            "mlp_lr": mlp_lr,
            "mlp_weight_decay": mlp_weight_decay,
            "mlp_label_smoothing": mlp_label_smoothing,
            "mlp_loss_type": mlp_loss_type,
            "mlp_focal_gamma": mlp_focal_gamma,
            "mlp_focal_alpha": mlp_focal_alpha,
            "mlp_sampler": mlp_sampler,
            "mlp_mixup_alpha": mlp_mixup_alpha,
            "mlp_normalize_embeddings": mlp_normalize_embeddings,
            "mlp_patience": mlp_patience,
            "mlp_activation": mlp_activation,
            "mlp_layer_norm": mlp_layer_norm,
            "mlp_hard_mining_epochs": mlp_hard_mining_epochs,
            "logit_adjustment_mode": logit_adjustment_mode,
            "logit_adjustment_train": bool(logit_adjustment_train),
            "logit_adjustment_inference": bool(logit_adjustment_inference_flag),
            "logit_adjustment": logit_adjustment_vec.tolist() if logit_adjustment_vec is not None else None,
            "arcface_enabled": bool(arcface_enabled),
            "arcface_margin": arcface_margin,
            "arcface_scale": arcface_scale,
            "supcon_weight": supcon_weight,
            "supcon_temperature": supcon_temperature,
            "supcon_projection_dim": supcon_projection_dim,
            "supcon_projection_hidden": supcon_projection_hidden,
            "background_class_count": bg_class_count,
            "background_classes": list(background_classes),
            "negative_crop_policy": dict(bg_policy),
            "augmentation_policy": dict(aug_policy),
            "oversample_policy": dict(oversample_policy),
            "embedding_center": embedding_center,
            "embedding_standardize": embedding_standardize,
            "embedding_center_values": embedding_center_values.tolist() if embedding_center_values is not None else None,
            "embedding_std_values": embedding_std_values.tolist() if embedding_std_values is not None else None,
            "calibration_mode": calibration_mode,
            "calibration_temperature": calibration_temperature,
            "phase_timings": dict(phase_timings),
            "labelmap_filename": os.path.basename(labelmap_output),
            "labelmap_path": labelmap_output,
            "n_classes_seen": len(encountered_sorted),
            "n_samples_train": n_train,
            "n_samples_test": n_test,
            "embedding_dim": embedding_dim,
            "iterations_run": iterations_run,
            "converged": converged,
        }
        meta_path = os.path.splitext(model_output)[0] + ".meta.pkl"
        _check_cancel()
        joblib.dump(meta, meta_path, compress=3)

        if reuse_embeddings and not using_cached_embeddings and cache_signature:
            _check_cancel()
            _write_cache_metadata(
                cache_signature,
                chunk_dir,
                chunk_records,
                y_class_names,
                y_numeric,
                groups,
                encountered_cids,
                bg_policy=bg_policy,
                aug_policy=aug_policy,
                oversample_policy=oversample_policy,
                background_classes=background_classes,
                labelmap_path=input_labelmap,
                labelmap_hash=labelmap_hash,
                embed_norm=normalize_embeddings,
            )
            cache_persisted = True

        _safe_progress(progress_cb, 0.92, "Cleaning up ...")
        torch.cuda.empty_cache() if resolved_device == "cuda" else None

        _safe_progress(progress_cb, 1.0, "Training completed.")
        phase_timings["total"] = time.perf_counter() - total_start
        logger.info("CLIP training timings: %s", phase_timings)

        result = TrainingArtifacts(
            model_path=model_output,
            labelmap_path=labelmap_output,
            meta_path=meta_path,
            accuracy=accuracy,
            classes_seen=len(encountered_sorted),
            samples_train=n_train,
            samples_test=n_test,
            clip_model=clip_model,
            encoder_type=encoder_type,
            encoder_model=clip_model,
            embedding_dim=embedding_dim,
            device=resolved_device,
            classification_report=report,
            confusion_matrix=matrix,
            label_order=label_list,
            iterations_run=iterations_run,
            converged=converged,
            convergence_trace=convergence_trace,
            solver=classifier_solver,
            hard_example_mining=hard_example_mining,
            class_weight=class_weight,
            effective_beta=float(effective_beta),
            per_class_metrics=per_class_metrics,
            hard_mining_misclassified_weight=hard_mining_misclassified_weight,
            hard_mining_low_conf_weight=hard_mining_low_conf_weight,
            hard_mining_low_conf_threshold=hard_mining_low_conf_threshold,
            hard_mining_margin_threshold=hard_mining_margin_threshold,
            convergence_tol=convergence_tol,
            background_class_count=int(bg_class_count),
            background_classes=list(background_classes),
            negative_crop_policy=dict(bg_policy),
            augmentation_policy=dict(aug_policy),
            oversample_policy=dict(oversample_policy),
            classifier_type=classifier_type,
            mlp_hidden_sizes=list(mlp_hidden_sizes_list),
            mlp_dropout=float(mlp_dropout),
            mlp_epochs=int(mlp_epochs),
            mlp_lr=float(mlp_lr),
            mlp_weight_decay=float(mlp_weight_decay),
            mlp_label_smoothing=float(mlp_label_smoothing),
            mlp_loss_type=str(mlp_loss_type),
            mlp_focal_gamma=float(mlp_focal_gamma),
            mlp_focal_alpha=None if mlp_focal_alpha is None else float(mlp_focal_alpha),
            mlp_sampler=str(mlp_sampler),
            mlp_mixup_alpha=float(mlp_mixup_alpha),
            mlp_normalize_embeddings=bool(mlp_normalize_embeddings),
            mlp_patience=int(mlp_patience),
            mlp_activation=str(mlp_activation),
            mlp_layer_norm=bool(mlp_layer_norm),
            mlp_hard_mining_epochs=int(mlp_hard_mining_epochs),
            logit_adjustment_mode=str(logit_adjustment_mode),
            logit_adjustment_inference=bool(logit_adjustment_inference_flag),
            logit_adjustment=None if logit_adjustment_vec is None else [float(x) for x in logit_adjustment_vec],
            arcface_enabled=bool(arcface_enabled),
            arcface_margin=float(arcface_margin),
            arcface_scale=float(arcface_scale),
            supcon_weight=float(supcon_weight),
            supcon_temperature=float(supcon_temperature),
            supcon_projection_dim=int(supcon_projection_dim),
            supcon_projection_hidden=int(supcon_projection_hidden),
            embedding_center=bool(embedding_center),
            embedding_standardize=bool(embedding_standardize),
            calibration_mode=str(calibration_mode),
            calibration_temperature=None if calibration_temperature is None else float(calibration_temperature),
            phase_timings=dict(phase_timings),
        )

    finally:
        del X_train
        del X_test
        try:
            del X_train_mm
        except Exception:
            pass
        try:
            del X_test_mm
        except Exception:
            pass
        try:
            if os.path.exists(train_memmap_path):
                os.remove(train_memmap_path)
        except Exception:
            pass
        try:
            if os.path.exists(test_memmap_path):
                os.remove(test_memmap_path)
        except Exception:
            pass
        if should_cleanup_chunks:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        elif not cache_persisted:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    return result
