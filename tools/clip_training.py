"""Reusable helpers for training CLIP + Logistic Regression models.

This module centralises the logic that used to live in
``tools/train_clip_regression_from_YOLO.py`` so that the FastAPI layer and the
CLI script can share the same implementation.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import clip
import joblib
import numpy as np
import torch
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
    per_class_metrics: List[Dict[str, Optional[float]]]
    hard_mining_misclassified_weight: float
    hard_mining_low_conf_weight: float
    hard_mining_low_conf_threshold: float
    hard_mining_margin_threshold: float
    convergence_tol: float


def _safe_progress(progress_cb: Optional[ProgressCallback], value: float, message: str) -> None:
    if progress_cb:
        capped = max(0.0, min(1.0, value))
        try:
            progress_cb(capped, message)
        except Exception:
            # Never let callback issues break the training job.
            pass


def _ensure_cache_root() -> None:
    EMBED_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _compute_dataset_signature(
    images_path: str,
    labels_path: str,
    clip_model: str,
    *,
    bg_class_count: int,
    bg_policy: Optional[str] = None,
    aug_policy: Optional[str] = None,
) -> str:
    entries: List[str] = [f"clip:{clip_model}", f"bg:{bg_class_count}"]
    if bg_policy:
        entries.append(f"bg_policy:{bg_policy}")
    if aug_policy:
        entries.append(f"aug_policy:{aug_policy}")
    image_root = Path(images_path)
    label_root = Path(labels_path)

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
        "background_classes": meta.get("background_classes", []),
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
                          background_classes: Optional[List[str]] = None) -> None:
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
        "background_classes": list(background_classes or []),
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
        if req in {"cpu", "cuda"}:
            if req == "cuda" and not torch.cuda.is_available():
                raise TrainingError("CUDA requested but no GPU is available.")
            return req
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_clip(arch: str, device: str) -> Tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    try:
        model, preprocess = clip.load(arch, device=device)
    except Exception as exc:
        raise TrainingError(f"Failed to load CLIP backbone '{arch}': {exc}") from exc
    model.eval()
    return model, preprocess


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
) -> np.ndarray:
    if not images:
        return np.empty((0, 512), dtype=np.float32)
    with torch.no_grad():
        batch = torch.stack([preprocess(img) for img in images]).to(device)
        feats = clip_model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy().astype(np.float32)


def _clip_aug_signature() -> str:
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


def _build_clip_augmenter() -> "A.Compose":
    if A is None:
        raise TrainingError("Albumentations is required for CLIP training.")
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


def _apply_augmenter(augmenter: "A.Compose", crop: Image.Image) -> Image.Image:
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


def train_clip_from_yolo(
    images_path: str,
    labels_path: str,
    model_output: str,
    labelmap_output: str,
    *,
    clip_model: str = "ViT-B/32",
    input_labelmap: Optional[str] = None,
    test_size: float = 0.2,
    random_seed: int = 42,
    max_iter: int = 1000,
    device: Optional[str] = None,
    batch_size: int = 64,
    min_per_class: int = 2,
    class_weight: str = "none",
    C: float = 1.0,
    solver: str = "saga",
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

    class_weight = (class_weight or "none").lower()
    if class_weight not in {"none", "balanced"}:
        class_weight = "none"

    hard_mining_misclassified_weight = float(max(1.0, hard_mining_misclassified_weight))
    hard_mining_low_conf_weight = float(max(1.0, hard_mining_low_conf_weight))
    hard_mining_low_conf_threshold = float(max(0.0, min(0.9999, hard_mining_low_conf_threshold)))
    hard_mining_margin_threshold = float(max(0.0, hard_mining_margin_threshold))
    convergence_tol = float(max(1e-8, convergence_tol))
    bg_class_count = max(1, min(10, int(bg_class_count)))

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
    rng = random.Random(int(random_seed))

    # Prepare paths early to fail fast on unwritable destinations.
    model_dir = os.path.dirname(os.path.abspath(model_output)) or "."
    labelmap_dir = os.path.dirname(os.path.abspath(labelmap_output)) or "."
    for path in {model_dir, labelmap_dir}:
        if path and not os.path.isdir(path):
            raise TrainingError(f"Output directory does not exist: {path}")

    _safe_progress(progress_cb, 0.0, "Loading configuration ...")

    def _check_cancel() -> None:
        if should_cancel and should_cancel():
            raise TrainingError("cancelled")

    _check_cancel()
    labelmap_list = _load_labelmap(input_labelmap)
    resolved_device = _resolve_device(device)
    clip_net, preprocess = _load_clip(clip_model, resolved_device)
    augmenter = _build_clip_augmenter()
    aug_policy_signature = _clip_aug_signature()
    aug_policy = {
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
            bg_class_count=bg_class_count,
            bg_policy=bg_policy_signature,
            aug_policy=aug_policy_signature,
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
    image_files: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(images_path):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() not in valid_exts:
                continue
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, images_path)
            image_files.append(rel_path)
    image_files.sort()
    _check_cancel()
    if not image_files:
        raise TrainingError("No supported images found in the provided folder.")

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
        _safe_progress(progress_cb, 0.08, f"Encoding CLIP embeddings on {resolved_device} (batch size={batch_size}) ...")

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
        embs = _encode_batch(clip_net, preprocess, resolved_device, batch_crops)
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
        embs = _encode_batch(clip_net, preprocess, resolved_device, bg_crop_batch)
        bg_embeddings.append(embs)
        bg_groups.extend(bg_group_batch)
        bg_crop_batch.clear()
        bg_group_batch.clear()

    total_pos = 0
    total_valid = 0
    if not using_cached_embeddings:
        label_exts = [".txt", ".TXT"]
        for idx, img_rel in enumerate(image_files, start=1):
            base = os.path.splitext(img_rel)[0]
            label_file = None
            for ext in label_exts:
                candidate = os.path.join(labels_path, base + ext)
                if os.path.isfile(candidate):
                    label_file = candidate
                    break
            if label_file is None:
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
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                    x_c, y_c, w_n, h_n = map(float, parts[1:5])
                except Exception:
                    continue

                x_min = (x_c - 0.5 * w_n) * w_img
                y_min = (y_c - 0.5 * h_n) * h_img
                x_max = x_min + w_n * w_img
                y_max = y_min + h_n * h_img
                bbox = _clamp_bbox(x_min, y_min, x_max, y_max, w_img, h_img)
                if bbox is None:
                    continue
                X1, Y1, X2, Y2 = bbox
                gt_boxes.append(bbox)
                try:
                    crop = pil_img.crop((X1, Y1, X2, Y2))
                except Exception:
                    continue
                crop = _apply_augmenter(augmenter, crop)

                if labelmap_list and 0 <= cid < len(labelmap_list):
                    cls_name = str(labelmap_list[cid])
                else:
                    cls_name = f"class_{cid}"

                batch_crops.append(crop)
                batch_meta.append((cls_name, cid, base))
                encountered_cids.add(cid)
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
    keep_mask = np.array([counts[c] >= max(1, min_per_class) for c in y_numeric_np])
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
    X_train_mm = np.memmap(train_memmap_path, dtype=np.float64, mode="w+", shape=(train_size, 512))
    X_test_mm = np.memmap(test_memmap_path, dtype=np.float64, mode="w+", shape=(test_size, 512))

    for chunk_path, old_start, count in chunk_records:
        _check_cancel()
        chunk = np.load(chunk_path)
        chunk = np.asarray(chunk, dtype=np.float64)
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
    y_train = y_class_names_arr[train_idx]
    y_test = y_class_names_arr[test_idx]

    convergence_trace: List[Dict[str, Optional[float]]] = []
    converged = False
    accuracy = 0.0
    report = ""
    matrix: List[List[int]] = []
    label_list: List[str] = []
    class_weight_param = None if class_weight == "none" else "balanced"

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

    iteration_counter = 0

    try:
        for iteration in range(1, max_iter + 1):
            _check_cancel()
            clf.fit(X_train, y_train)
            proba_train = clf.predict_proba(X_train)
            train_loss = float(log_loss(y_train, proba_train, labels=clf.classes_))
            train_pred = clf.predict(X_train)
            train_acc = float((train_pred == y_train).mean())
            last_train_pred = train_pred
            last_train_proba = proba_train

            val_loss: Optional[float] = None
            val_acc: Optional[float] = None
            if y_test.size:
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
        per_class_metrics: List[Dict[str, Optional[float]]] = []
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

        _safe_progress(progress_cb, 0.75, f"Saving classifier to {model_output} and labelmap to {labelmap_output} ...")
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

        meta = {
            "clip_model": clip_model,
            "device": resolved_device,
            "test_size": test_size,
            "random_seed": random_seed,
            "min_per_class": min_per_class,
            "class_weight": class_weight,
            "C": C,
            "solver": solver,
            "hard_example_mining": hard_example_mining,
            "hard_mining_misclassified_weight": hard_mining_misclassified_weight,
            "hard_mining_low_conf_weight": hard_mining_low_conf_weight,
            "hard_mining_low_conf_threshold": hard_mining_low_conf_threshold,
            "hard_mining_margin_threshold": hard_mining_margin_threshold,
            "convergence_tol": convergence_tol,
            "background_class_count": bg_class_count,
            "background_classes": list(background_classes),
            "negative_crop_policy": dict(bg_policy),
            "augmentation_policy": dict(aug_policy),
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
                background_classes=background_classes,
            )
            cache_persisted = True

        _safe_progress(progress_cb, 0.92, "Cleaning up ...")
        torch.cuda.empty_cache() if resolved_device == "cuda" else None

        _safe_progress(progress_cb, 1.0, "Training completed.")

        result = TrainingArtifacts(
            model_path=model_output,
            labelmap_path=labelmap_output,
            meta_path=meta_path,
            accuracy=accuracy,
            classes_seen=len(encountered_sorted),
            samples_train=n_train,
            samples_test=n_test,
            clip_model=clip_model,
            device=resolved_device,
            classification_report=report,
            confusion_matrix=matrix,
            label_order=label_list,
            iterations_run=iterations_run,
            converged=converged,
            convergence_trace=convergence_trace,
            solver=solver,
            hard_example_mining=hard_example_mining,
            class_weight=class_weight,
            per_class_metrics=per_class_metrics,
            hard_mining_misclassified_weight=hard_mining_misclassified_weight,
            hard_mining_low_conf_weight=hard_mining_low_conf_weight,
            hard_mining_low_conf_threshold=hard_mining_low_conf_threshold,
            hard_mining_margin_threshold=hard_mining_margin_threshold,
            convergence_tol=convergence_tol,
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
