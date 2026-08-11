"""Apple Silicon MLX adapter for SAM3 interactive geometric prompting."""

from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
import platform
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


DEFAULT_MLX_SAM3_MODEL_ID = "mlx-community/sam3-bf16"
DEFAULT_MLX_SAM3_REVISION = "dfe573c3171dbcfda8399c650d9135afa7e94592"
MIN_MLX_VLM_VERSION = "0.6.6"
MLX_SAM3_RUNTIME_NOTICE = (
    "Apple Silicon is using the accelerated BF16 MLX SAM3 checkpoint. "
    "Masks can differ slightly from full-precision Torch; set SAM3_BACKEND=torch "
    "when exact Torch comparison is required."
)

_MODEL_LOCK = threading.Lock()
_EXECUTION_LOCK = threading.RLock()
_SHARED_MODELS: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


@dataclass(frozen=True)
class MlxSam3VariantSpec:
    runtime_id: str
    model_id: str
    revision: str
    label: str
    size_bytes: int
    quality_tier: str
    guidance: str
    preload_median_ms: float
    point_median_ms: float
    box_median_ms: float
    point_iou_mean: float
    box_iou_mean: float


# Official same-generation MLX checkpoints that passed the local geometric
# prompt gate. Integer 6-bit is omitted because one clear building point prompt
# collapsed to IoU 0.022. NVFP4 is omitted because it was both slower and less
# faithful than MXFP4. BF16 remains the default because it was also fastest.
MLX_SAM3_VARIANTS: Dict[str, MlxSam3VariantSpec] = {
    "mlx-bf16": MlxSam3VariantSpec(
        "mlx-bf16", DEFAULT_MLX_SAM3_MODEL_ID, DEFAULT_MLX_SAM3_REVISION,
        "MLX BF16 - Recommended", 1_724_814_602, "recommended",
        "Best overall local result and fastest preload; about 1.72 GB on disk.",
        220.48, 4.14, 4.53, 0.9255, 0.9273,
    ),
    "mlx-8bit": MlxSam3VariantSpec(
        "mlx-8bit", "mlx-community/sam3-8bit",
        "bcd63fddcd2082f3841c7f25dc67a1bcb9e95dcf",
        "MLX 8-bit - Balanced", 1_042_981_230, "balanced",
        "Close to BF16 quality at about 1.04 GB; slightly slower on this Mac.",
        251.84, 4.37, 4.47, 0.9166, 0.9271,
    ),
    "mlx-5bit": MlxSam3VariantSpec(
        "mlx-5bit", "mlx-community/sam3-5bit",
        "158c434fca8e9eb591433d93da12537ea3834636",
        "MLX 5-bit - Compact", 729_679_819, "compact",
        "Strong measured fidelity at about 0.73 GB, but slower than BF16 and 8-bit.",
        321.47, 5.53, 5.22, 0.9255, 0.9316,
    ),
    "mlx-mxfp8": MlxSam3VariantSpec(
        "mlx-mxfp8", "mlx-community/sam3-mxfp8",
        "715709b3f9e36ecbf23be76b94075f684cf97860",
        "MLX MXFP8 - Experimental", 964_505_600, "experimental",
        "Good point fidelity at about 0.96 GB; box masks varied more than BF16.",
        272.40, 4.45, 4.88, 0.9254, 0.9118,
    ),
    "mlx-mxfp4": MlxSam3VariantSpec(
        "mlx-mxfp4", "mlx-community/sam3-mxfp4",
        "38eced50afd50303f207c0165d0299991373c683",
        "MLX MXFP4 - Smallest good", 546_770_631, "compact",
        "Smallest passing option at about 0.55 GB; good mean fidelity, but slower than BF16.",
        313.46, 5.07, 6.06, 0.9225, 0.9307,
    ),
    "mlx-4bit": MlxSam3VariantSpec(
        "mlx-4bit", "mlx-community/sam3-4bit",
        "53c009b4ef4cfa5b1d9fa4549e7b153e881b5636",
        "MLX 4-bit - Experimental", 625_246_115, "experimental",
        "About 0.63 GB and usable, but the slowest offered MLX runtime with lower fidelity.",
        549.86, 7.41, 7.94, 0.9120, 0.9134,
    ),
}


class MlxSam3Unavailable(RuntimeError):
    """Raised when MLX SAM3 was explicitly requested but is unavailable."""


@dataclass(frozen=True)
class MlxSam3Config:
    available: bool
    reason: Optional[str]
    model_path: Optional[Path]
    model_id: str
    revision: str
    apple_silicon: bool
    mlx_installed: bool
    mlx_vlm_installed: bool
    runtime_id: str = "auto"
    label: str = "MLX SAM3"
    size_bytes: Optional[int] = None
    quality_tier: str = "custom"
    guidance: str = ""


class MlxSam3PredictorAdapter:
    """Expose mlx-vlm's SAM3 tracker through Meta's interactive numpy API."""

    runtime_notice = MLX_SAM3_RUNTIME_NOTICE

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        model_path: Path,
        runtime_notice: str = MLX_SAM3_RUNTIME_NOTICE,
    ) -> None:
        self.model = model
        self.processor = processor
        self.model_path = model_path
        self.runtime_notice = runtime_notice
        self._features = None
        self._orig_hw: Optional[Tuple[int, int]] = None
        self._image_size = int(getattr(processor, "image_size", 1008) or 1008)
        tracker_config = getattr(getattr(model, "config", None), "tracker_config", None)
        prompt_config = getattr(tracker_config, "prompt_encoder_config", None)
        patch_size = int(getattr(prompt_config, "patch_size", 14) or 14)
        self._embedding_size = max(1, self._image_size // patch_size)

    def set_image(self, np_img: np.ndarray) -> None:
        import mlx.core as mx

        arr = np.ascontiguousarray(np_img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"sam3_mlx_image_shape_invalid:{arr.shape}")
        with _EXECUTION_LOCK:
            inputs = self.processor.preprocess_image(arr)
            pixel_values = mx.array(inputs["pixel_values"])
            backbone = self.model.detector_model.vision_encoder.backbone(pixel_values)
            features = list(self.model.tracker_neck(backbone))
            # Meta adds this learned embedding to the lowest-resolution feature for
            # interactive image prediction because there is no video memory bank.
            features[2] = (
                features[2]
                + self.model.tracker_model.no_memory_embedding
            )
            mx.eval(backbone, *features)
            self._features = features
            self._orig_hw = (int(arr.shape[0]), int(arr.shape[1]))

    def predict(self, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import mlx.core as mx

        if self._features is None or self._orig_hw is None:
            raise RuntimeError("sam3_mlx_image_not_set")
        with _EXECUTION_LOCK:
            coords, labels = _build_sparse_prompt_arrays(
                point_coords=kwargs.get("point_coords"),
                point_labels=kwargs.get("point_labels"),
                box=kwargs.get("box"),
                orig_hw=self._orig_hw,
                image_size=self._image_size,
                embedding_size=self._embedding_size,
            )
            prompt_points = None
            if coords is not None and labels is not None:
                prompt_points = (mx.array(coords[None]), mx.array(labels[None]))
            prompt_masks_np = _normalize_mask_input(
                kwargs.get("mask_input"), target_size=self._embedding_size * 4
            )
            prompt_masks = mx.array(prompt_masks_np) if prompt_masks_np is not None else None
            low_res_mx, scores_mx = _run_mlx_sam3_mask_decoder(
                model=self.model,
                current_features=self._features[2],
                prompt_points=prompt_points,
                prompt_masks=prompt_masks,
                multimask_output=bool(kwargs.get("multimask_output", True)),
                high_res_features=[self._features[0], self._features[1]],
            )
            mx.eval(low_res_mx, scores_mx)

            low_res = np.array(low_res_mx.astype(mx.float32))
            scores = np.array(scores_mx.astype(mx.float32))
            if low_res.ndim == 4 and low_res.shape[0] == 1:
                low_res = low_res[0]
            if scores.ndim == 2 and scores.shape[0] == 1:
                scores = scores[0]
            low_res = np.clip(low_res, -32.0, 32.0)
            full_logits = np.stack(
                [_resize_float_mask(mask, self._orig_hw) for mask in low_res], axis=0
            )
            masks = full_logits if kwargs.get("return_logits", False) else full_logits > 0.0
            return masks, scores, low_res

    def unload(self) -> None:
        with _EXECUTION_LOCK:
            self._features = None
            self._orig_hw = None


def _run_mlx_sam3_mask_decoder(
    *,
    model: Any,
    current_features: Any,
    prompt_points: Any,
    prompt_masks: Any,
    multimask_output: bool,
    high_res_features: Any,
) -> Tuple[Any, Any]:
    """Run Meta-equivalent interactive decoding around mlx-vlm's SAM3 modules.

    mlx-vlm 0.6.6 has correct converted weights, but its tracker wrapper changes
    trained token order, first-layer attention, and skip placement. Keeping the
    compatibility path here makes those invariants explicit and removable once
    upstream provides a parity-tested image predictor.
    """
    import mlx.core as mx
    import mlx.nn as nn

    tracker = model.tracker_model
    decoder = tracker.mask_decoder
    sparse_embeddings, dense_embeddings = tracker.prompt_encoder(
        points=prompt_points,
        boxes=None,
        masks=prompt_masks,
    )
    batch, height, width, hidden_size = current_features.shape
    image_embeddings = current_features.reshape(batch, height * width, hidden_size)
    image_pe = tracker.prompt_encoder.shared_embedding.forward_with_coords(
        mx.array(_dense_position_coords(height, width))
    )
    if batch != 1:
        image_pe = mx.broadcast_to(image_pe, (batch, height * width, hidden_size))

    # SAM3 was trained with object, IoU, then mask tokens in this exact order.
    output_tokens = mx.concatenate(
        [
            mx.broadcast_to(decoder.obj_score_token.weight[None], (batch, 1, hidden_size)),
            mx.broadcast_to(decoder.iou_token.weight[None], (batch, 1, hidden_size)),
            mx.broadcast_to(
                decoder.mask_tokens.weight[None],
                (batch, decoder.num_mask_tokens, hidden_size),
            ),
        ],
        axis=1,
    )
    tokens = mx.concatenate([output_tokens, sparse_embeddings], axis=1)
    queries, src = _run_mlx_sam3_two_way_transformer(
        decoder.transformer,
        image_embeddings + dense_embeddings,
        image_pe,
        tokens,
    )
    iou_token = queries[:, 1]
    mask_tokens = queries[:, 2 : 2 + decoder.num_mask_tokens]
    src = src.reshape(batch, height, width, hidden_size)

    # Meta adds the 144px skip before normalization/GELU and the 288px skip
    # before the second GELU. mlx-vlm's wrapper reverses both order and timing.
    upscaled = decoder.upscale_conv1(src) + decoder.conv_s1(high_res_features[1])
    upscaled = nn.gelu(decoder.upscale_layer_norm(upscaled))
    upscaled = nn.gelu(
        decoder.upscale_conv2(upscaled) + decoder.conv_s0(high_res_features[0])
    )
    flat = upscaled.reshape(batch, -1, upscaled.shape[-1])
    all_masks = []
    for index in range(decoder.num_mask_tokens):
        hyper = decoder.output_hypernetworks_mlps[index](mask_tokens[:, index])
        mask = mx.sum(flat * hyper[:, None, :], axis=-1)
        all_masks.append(mask.reshape(batch, 1, upscaled.shape[1], upscaled.shape[2]))
    all_masks = mx.concatenate(all_masks, axis=1)
    all_scores = mx.sigmoid(decoder.iou_prediction_head(iou_token))
    return _select_mlx_sam3_masks(
        decoder,
        all_masks,
        all_scores,
        multimask_output=multimask_output,
    )


def _run_mlx_sam3_two_way_transformer(
    transformer: Any,
    image_embeddings: Any,
    image_pe: Any,
    point_embeddings: Any,
) -> Tuple[Any, Any]:
    queries = point_embeddings
    keys = image_embeddings
    for layer_index, layer in enumerate(transformer.layers):
        if layer_index == 0:
            queries = layer.self_attn(queries, queries, queries)
        else:
            query = queries + point_embeddings
            queries = queries + layer.self_attn(query, query, queries)
        queries = layer.layer_norm1(queries)

        query = queries + point_embeddings
        key = keys + image_pe
        queries = layer.layer_norm2(
            queries + layer.cross_attn_token_to_image(query, key, keys)
        )
        queries = layer.layer_norm3(queries + layer.mlp(queries))

        query = queries + point_embeddings
        key = keys + image_pe
        keys = layer.layer_norm4(
            keys + layer.cross_attn_image_to_token(key, query, queries)
        )

    query = queries + point_embeddings
    key = keys + image_pe
    queries = transformer.layer_norm_final_attn(
        queries + transformer.final_attn_token_to_image(query, key, keys)
    )
    return queries, keys


def _select_mlx_sam3_masks(
    decoder: Any,
    all_masks: Any,
    all_scores: Any,
    *,
    multimask_output: bool,
) -> Tuple[Any, Any]:
    import mlx.core as mx

    if multimask_output:
        return all_masks[:, 1:], all_scores[:, 1:]
    single_mask = all_masks[:, :1]
    single_score = all_scores[:, :1]
    if not bool(getattr(decoder, "dynamic_multimask_via_stability", False)):
        return single_mask, single_score

    delta = float(getattr(decoder, "dynamic_multimask_stability_delta", 0.05))
    threshold = float(getattr(decoder, "dynamic_multimask_stability_thresh", 0.98))
    area_intersection = mx.sum(single_mask > delta, axis=(-2, -1)).astype(mx.float32)
    area_union = mx.sum(single_mask > -delta, axis=(-2, -1)).astype(mx.float32)
    stability = mx.where(
        area_union > 0,
        area_intersection / mx.maximum(area_union, mx.ones_like(area_union)),
        mx.ones_like(area_union),
    )
    multi_scores = all_scores[:, 1:]
    best_indices = mx.argmax(multi_scores, axis=-1)
    batch_indices = mx.arange(all_masks.shape[0])
    best_masks = all_masks[:, 1:][batch_indices, best_indices][:, None]
    best_scores = multi_scores[batch_indices, best_indices][:, None]
    stable = stability >= threshold
    return (
        mx.where(stable[:, :, None, None], single_mask, best_masks),
        mx.where(stable, single_score, best_scores),
    )


def _dense_position_coords(height: int, width: int) -> np.ndarray:
    y = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    x = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)[None]


def mlx_sam3_status(runtime: str = "auto") -> Dict[str, Any]:
    config = resolve_mlx_sam3_config(runtime)
    return {
        "available": config.available,
        "reason": config.reason,
        "model_path": str(config.model_path) if config.model_path else None,
        "model_id": config.model_id,
        "revision": config.revision,
        "apple_silicon": config.apple_silicon,
        "mlx_installed": config.mlx_installed,
        "mlx_vlm_installed": config.mlx_vlm_installed,
        "minimum_mlx_vlm_version": MIN_MLX_VLM_VERSION,
        "runtime_id": config.runtime_id,
        "label": config.label,
        "size_bytes": config.size_bytes,
        "quality_tier": config.quality_tier,
        "guidance": config.guidance,
    }


def normalize_mlx_sam3_runtime(value: Optional[str]) -> str:
    normalized = str(value or "auto").strip().lower()
    aliases = {
        "bf16": "mlx-bf16",
        "8bit": "mlx-8bit",
        "5bit": "mlx-5bit",
        "mxfp8": "mlx-mxfp8",
        "mxfp4": "mlx-mxfp4",
        "4bit": "mlx-4bit",
        "mlx": "mlx-bf16",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"auto", "torch"} or normalized in MLX_SAM3_VARIANTS:
        return normalized
    raise ValueError(f"sam3_runtime_invalid:{normalized}")


def mlx_sam3_runtime_options() -> list[Dict[str, Any]]:
    options = []
    for runtime_id, spec in MLX_SAM3_VARIANTS.items():
        config = resolve_mlx_sam3_config(runtime_id)
        options.append(
            {
                "id": runtime_id,
                "label": spec.label,
                "available": config.available,
                "installed": config.model_path is not None,
                "reason": config.reason,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "size_bytes": spec.size_bytes,
                "quality_tier": spec.quality_tier,
                "guidance": spec.guidance,
                "benchmark": {
                    "sample_count": 12,
                    "preload_median_ms": spec.preload_median_ms,
                    "point_median_ms": spec.point_median_ms,
                    "box_median_ms": spec.box_median_ms,
                    "point_mask_iou_mean": spec.point_iou_mean,
                    "box_mask_iou_mean": spec.box_iou_mean,
                },
                "setup_command": f".venv-macos/bin/python tools/setup_mlx_sam3.py --variant {runtime_id}",
            }
        )
    return options


def should_use_mlx_sam3(preference: str = "auto", runtime: str = "auto") -> bool:
    runtime_id = normalize_mlx_sam3_runtime(runtime)
    if runtime_id == "torch":
        return False
    if runtime_id != "auto":
        config = resolve_mlx_sam3_config(runtime_id)
        if not config.available:
            raise MlxSam3Unavailable(config.reason or "mlx_sam3_unavailable")
        return True
    pref = _normalize_preference(preference)
    if pref == "torch":
        return False
    config = resolve_mlx_sam3_config("auto")
    if pref == "mlx":
        if not config.available:
            raise MlxSam3Unavailable(config.reason or "mlx_sam3_unavailable")
        return True
    return config.available


def build_mlx_sam3_predictor(runtime: str = "auto") -> MlxSam3PredictorAdapter:
    config = resolve_mlx_sam3_config(runtime)
    if not config.available or config.model_path is None:
        raise MlxSam3Unavailable(config.reason or "mlx_sam3_unavailable")
    cache_key = (str(config.model_path.resolve()), config.revision)
    with _MODEL_LOCK:
        cached = _SHARED_MODELS.get(cache_key)
        if cached is None:
            import mlx.core as mx
            from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
            from mlx_vlm.utils import load_model

            model = load_model(config.model_path, lazy=False, strict=True)
            model.eval()
            mx.eval(model.parameters())
            processor = Sam3Processor.from_pretrained(str(config.model_path))
            cached = (model, processor)
            # Slots evict superseded checkpoint-specific backends. Retain only
            # the newest shared loader reference so selector exploration does
            # not pin every checkpoint in process memory.
            _SHARED_MODELS.clear()
            _SHARED_MODELS[cache_key] = cached
        model, processor = cached
    notice = (
        f"Apple Silicon is using {config.label}. Masks can differ slightly from "
        "full-precision Torch; choose Torch reference for exact comparison."
    )
    return MlxSam3PredictorAdapter(
        model,
        processor,
        model_path=config.model_path,
        runtime_notice=notice,
    )


def clear_shared_mlx_sam3_models() -> None:
    with _MODEL_LOCK:
        _SHARED_MODELS.clear()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def resolve_mlx_sam3_config(runtime: str = "auto") -> MlxSam3Config:
    runtime_id = normalize_mlx_sam3_runtime(runtime)
    if runtime_id == "torch":
        return MlxSam3Config(
            False, "torch_runtime_selected", None,
            DEFAULT_MLX_SAM3_MODEL_ID, DEFAULT_MLX_SAM3_REVISION,
            platform.system() == "Darwin", False, False,
            runtime_id="torch", label="Torch reference", quality_tier="reference",
        )
    spec = MLX_SAM3_VARIANTS.get(runtime_id)
    if spec is None:
        model_id = str(os.environ.get("SAM3_MLX_MODEL_ID") or DEFAULT_MLX_SAM3_MODEL_ID)
        revision = str(os.environ.get("SAM3_MLX_MODEL_REVISION") or DEFAULT_MLX_SAM3_REVISION)
        default_spec = MLX_SAM3_VARIANTS["mlx-bf16"]
        label = default_spec.label if model_id == DEFAULT_MLX_SAM3_MODEL_ID else "Custom MLX SAM3"
        size_bytes = default_spec.size_bytes if model_id == DEFAULT_MLX_SAM3_MODEL_ID else None
        quality_tier = default_spec.quality_tier if model_id == DEFAULT_MLX_SAM3_MODEL_ID else "custom"
        guidance = default_spec.guidance if model_id == DEFAULT_MLX_SAM3_MODEL_ID else "Environment-selected MLX SAM3 checkpoint."
    else:
        model_id = spec.model_id
        revision = spec.revision
        label = spec.label
        size_bytes = spec.size_bytes
        quality_tier = spec.quality_tier
        guidance = spec.guidance
    apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }
    mlx_installed = importlib.util.find_spec("mlx") is not None
    mlx_vlm_installed = importlib.util.find_spec("mlx_vlm") is not None
    if not apple_silicon:
        return MlxSam3Config(
            False, "not_apple_silicon", None, model_id, revision,
            apple_silicon, mlx_installed, mlx_vlm_installed,
            runtime_id, label, size_bytes, quality_tier, guidance,
        )
    if not mlx_installed:
        return MlxSam3Config(
            False, "mlx_not_installed", None, model_id, revision,
            apple_silicon, mlx_installed, mlx_vlm_installed,
            runtime_id, label, size_bytes, quality_tier, guidance,
        )
    if not mlx_vlm_installed:
        return MlxSam3Config(
            False, "mlx_vlm_not_installed", None, model_id, revision,
            apple_silicon, mlx_installed, mlx_vlm_installed,
            runtime_id, label, size_bytes, quality_tier, guidance,
        )
    version_error = _mlx_vlm_version_error()
    if version_error:
        return MlxSam3Config(
            False, version_error, None, model_id, revision,
            apple_silicon, mlx_installed, mlx_vlm_installed,
            runtime_id, label, size_bytes, quality_tier, guidance,
        )
    runtime_error = _mlx_runtime_error()
    if runtime_error:
        return MlxSam3Config(
            False, f"mlx_runtime_unavailable: {runtime_error}", None, model_id, revision,
            apple_silicon, mlx_installed, mlx_vlm_installed,
            runtime_id, label, size_bytes, quality_tier, guidance,
        )
    model_path = _resolve_model_path(
        model_id,
        revision,
        allow_environment_path=runtime_id == "auto",
    )
    if model_path is None:
        return MlxSam3Config(
            False, "mlx_sam3_model_path_missing", None, model_id, revision,
            apple_silicon, mlx_installed, mlx_vlm_installed,
            runtime_id, label, size_bytes, quality_tier, guidance,
        )
    return MlxSam3Config(
        True, None, model_path, model_id, revision,
        apple_silicon, mlx_installed, mlx_vlm_installed,
        runtime_id, label, size_bytes, quality_tier, guidance,
    )


def _resolve_model_path(
    model_id: str,
    revision: str,
    *,
    allow_environment_path: bool = True,
) -> Optional[Path]:
    explicit = (
        str(os.environ.get("SAM3_MLX_MODEL_PATH") or "").strip()
        if allow_environment_path
        else ""
    )
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve() if _model_ready(candidate) else None

    hf_home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    repo_root = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    exact = repo_root / "snapshots" / revision
    if _model_ready(exact):
        return exact.resolve()
    ref = repo_root / "refs" / revision
    if ref.is_file():
        try:
            referenced = repo_root / "snapshots" / ref.read_text(encoding="utf-8").strip()
        except OSError:
            referenced = None
        if referenced is not None and _model_ready(referenced):
            return referenced.resolve()
    snapshots = repo_root / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(
            snapshots.iterdir(),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )
        for candidate in candidates:
            if _model_ready(candidate):
                return candidate.resolve()
    return None


def _model_ready(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "processor_config.json").is_file()
        and (
            (path / "model.safetensors").is_file()
            or (path / "model.safetensors.index.json").is_file()
        )
    )


def _build_sparse_prompt_arrays(
    *,
    point_coords: Any,
    point_labels: Any,
    box: Any,
    orig_hw: Tuple[int, int],
    image_size: int,
    embedding_size: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    coords_parts = []
    label_parts = []
    has_box = box is not None
    if has_box:
        box_arr = np.asarray(box, dtype=np.float32)
        if box_arr.size != 4:
            raise ValueError("sam3_mlx_box_shape_invalid")
        coords_parts.append(box_arr.reshape(2, 2))
        label_parts.append(np.asarray([2, 3], dtype=np.int32))
    if point_coords is not None:
        if point_labels is None:
            raise ValueError("sam3_mlx_point_labels_required")
        points_arr = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
        labels_arr = np.asarray(point_labels, dtype=np.int32).reshape(-1)
        if len(points_arr) != len(labels_arr):
            raise ValueError("sam3_mlx_point_shape_mismatch")
        coords_parts.append(points_arr)
        label_parts.append(labels_arr)
        if not has_box:
            coords_parts.append(np.zeros((1, 2), dtype=np.float32))
            label_parts.append(np.asarray([-1], dtype=np.int32))
    if not coords_parts:
        return None, None

    coords = np.concatenate(coords_parts, axis=0).astype(np.float32, copy=False)
    labels = np.concatenate(label_parts, axis=0).astype(np.int32, copy=False)
    height, width = orig_hw
    if height <= 0 or width <= 0:
        raise ValueError("sam3_mlx_original_size_invalid")
    model_coords = coords.copy()
    model_coords[:, 0] *= float(image_size) / float(width)
    model_coords[:, 1] *= float(image_size) / float(height)
    # mlx-vlm 0.6.1 normalizes point inputs by the embedding grid rather
    # than the input image size. This inverse transform reproduces Meta's
    # (pixel + 0.5) / input_size positional encoding exactly.
    adapted = ((model_coords + 0.5) / float(image_size)) * float(embedding_size) - 0.5
    return adapted.astype(np.float32), labels


def _normalize_mask_input(mask_input: Any, *, target_size: int) -> Optional[np.ndarray]:
    if mask_input is None:
        return None
    mask = np.asarray(mask_input, dtype=np.float32)
    while mask.ndim > 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            raise ValueError(f"sam3_mlx_mask_input_shape_invalid:{mask.shape}")
    if mask.ndim != 2:
        raise ValueError(f"sam3_mlx_mask_input_shape_invalid:{mask.shape}")
    if mask.shape != (target_size, target_size):
        mask = _resize_float_mask(mask, (target_size, target_size))
    return mask[None, :, :, None]


def _resize_float_mask(mask: np.ndarray, output_hw: Tuple[int, int]) -> np.ndarray:
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    arr = np.asarray(mask, dtype=np.float32)
    if arr.shape == (output_h, output_w):
        return arr
    resampling = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    resized = Image.fromarray(arr).resize((output_w, output_h), resampling)
    return np.asarray(resized, dtype=np.float32)


def _mlx_runtime_error() -> Optional[str]:
    try:
        import mlx.core as mx

        probe = mx.array([0.0])
        mx.eval(probe)
        return None
    except Exception as exc:  # pragma: no cover - depends on the host Metal runtime.
        return str(exc) or exc.__class__.__name__


def _mlx_vlm_version_error() -> Optional[str]:
    try:
        from packaging.version import Version

        installed = importlib.metadata.version("mlx-vlm")
        if Version(installed) < Version(MIN_MLX_VLM_VERSION):
            return f"mlx_vlm_too_old:{installed};requires>={MIN_MLX_VLM_VERSION}"
    except importlib.metadata.PackageNotFoundError:
        return "mlx_vlm_not_installed"
    except Exception as exc:
        return f"mlx_vlm_version_unreadable:{exc}"
    return None


def _normalize_preference(preference: str) -> str:
    pref = str(preference or "auto").strip().lower()
    return pref if pref in {"auto", "mlx", "torch"} else "auto"
