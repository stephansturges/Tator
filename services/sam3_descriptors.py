"""SAM3 tracker descriptors without changing interactive mask decoding.

The historical MLX recipe pools three 256-channel tracker maps, including
the no-memory embedding at the lowest resolution. Native SAM3 caches 32/64
channel projections for its decoder. Capture the inputs to those projections;
never substitute the decoder caches for the descriptor or modify their values.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import math
import os
import platform
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import marshal
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image

NATIVE_DESCRIPTOR_ABI = "sam3-native-preprojection-mask-pool-v1"
HISTORICAL_DESCRIPTOR_PROFILE = {
    "name": "historical_mlx_descriptor_v1",
    "vision_layer_norm_epsilon": 1e-6,
    "global_rope_position_scale": 1.0,
    "global_attention_blocks": [7, 15, 23, 31],
    "image_size": 1008,
    "patch_size": 14,
    "box_prompt": "two_corner_tokens_without_dummy_padding",
    "checkpoint_weights": "original_native_float32",
    "scope": "dedicated_analysis_model_only",
}


@contextmanager
def native_sam3_context(device):
    """Keep native caches/kernels on the assigned GPU and inference in FP32."""
    import torch
    target = torch.device(device)
    selected = torch.cuda.device(target) if target.type == "cuda" else nullcontext()
    with selected, torch.autocast(device_type="cuda", enabled=False):
        yield


def _autocast_nesting(torch):
    # PyTorch exposes no read-only nesting accessor. This balanced pair reads
    # the thread-local count without changing it or allocating a CUDA tensor.
    depth = torch.autocast_increment_nesting() - 1
    torch.autocast_decrement_nesting()
    return depth


def build_native_sam3_model(builder, *, device, **options):
    """Balance the pinned tracker's process-wide autocast entry, even on failure."""
    import torch
    with native_sam3_context(device):
        baseline = _autocast_nesting(torch)
        try:
            model = builder(device="cpu", **options)
            predictor = getattr(model, "inst_interactive_predictor", None)
            tracker = getattr(predictor, "model", None)
            constructor_context = getattr(tracker, "bf16_context", None)
            if constructor_context is not None:
                constructor_context.__exit__(None, None, None)
            return model.to(device)
        finally:
            # A failed native constructor can enter BF16 before returning its
            # tracker. Restore the nesting owned by this call as well as the
            # enabled/dtype/cache flags restored by native_sam3_context.
            while _autocast_nesting(torch) > baseline:
                torch.autocast_decrement_nesting()


def configure_historical_descriptor_geometry(model):
    """Reproduce the saved MLX descriptor geometry on a dedicated analysis model.

    The upstream annotation model intentionally uses different global RoPE
    scaling and LayerNorm epsilon. Never call this on a user's tool model.
    Check the complete expected structure before changing any model state.
    """
    import torch
    try:
        trunk = model.backbone.vision_backbone.trunk
        blocks = list(trunk.blocks)
        global_ids = HISTORICAL_DESCRIPTOR_PROFILE["global_attention_blocks"]
        norms = [trunk.ln_pre] + [norm for block in blocks for norm in (block.norm1, block.norm2)]
        valid = (
            model.inst_interactive_predictor.model.image_size == 1008
            and tuple(trunk.patch_embed.proj.kernel_size) == (14, 14)
            and tuple(trunk.patch_embed.proj.stride) == (14, 14)
            and len(blocks) == 32 and list(trunk.full_attn_ids) == global_ids
            and len(norms) == 65
            and all(isinstance(norm, torch.nn.LayerNorm)
                    and tuple(norm.normalized_shape) == (1024,) and norm.eps == 1e-5 for norm in norms)
            and all(block.window_size == (0 if i in global_ids else 24)
                    and tuple(block.attn.input_size) == ((72, 72) if i in global_ids else (24, 24))
                    and tuple(block.attn.rope_pt_size) == (24, 24)
                    and block.attn.head_dim == 64
                    and block.attn.rope_interp and block.attn.use_rope
                    for i, block in enumerate(blocks))
        )
        if not valid:
            raise ValueError("unexpected_native_vision_geometry")
        # Prepare replacement buffers before mutating the model. Buffer device
        # follows the loaded model, including non-default CUDA devices.
        replacements = []
        for i in global_ids:
            attention = blocks[i].attn
            frequencies = attention.compute_cis(end_x=72, end_y=72, scale_pos=1.0)
            if frequencies.shape != (5184, 32) or frequencies.shape != attention.freqs_cis.shape:
                raise ValueError("unexpected_native_rope_shape")
            replacements.append((attention, frequencies.to(attention.freqs_cis.device)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("sam3_historical_descriptor_geometry_unsupported") from exc
    for norm in norms:
        norm.eps = HISTORICAL_DESCRIPTOR_PROFILE["vision_layer_norm_epsilon"]
    for attention, frequencies in replacements:
        attention.rope_pt_size = attention.input_size
        attention.freqs_cis = frequencies
    model._tator_descriptor_profile = dict(HISTORICAL_DESCRIPTOR_PROFILE)
    return model


def _compiled_functions(path):
    """Match the final definition of each qualified name, including Python 3.10."""
    expected = {}

    def collect(code, prefix=""):
        for child in code.co_consts:
            if not inspect.iscode(child):
                continue
            name = prefix + child.co_name
            expected[name] = child
            # Class bodies have no NEWLOCALS flag; functions introduce the
            # <locals> qualifier used by nested functions and classes.
            suffix = ".<locals>." if child.co_flags & inspect.CO_NEWLOCALS else "."
            collect(child, name + suffix)

    collect(compile(path.read_bytes(), str(path), "exec", dont_inherit=True))
    return expected


def descriptor_status(api):
    if descriptor_backend() == "mlx-bf16":
        from services.mlx_sam3 import mlx_sam3_status
        return mlx_sam3_status("mlx-bf16")
    checkpoint = getattr(api, "active_sam3_checkpoint", None)
    return {"model_id": getattr(api, "SAM3_MODEL_ID", "facebook/sam3"),
            "model_path": str(checkpoint) if checkpoint else None,
            "available": bool(checkpoint and Path(checkpoint).is_file()), "runtime_id": "torch"}


def native_runtime_identity(api):
    """Attest checkpoint, package source and loaded Python code independently."""
    import torch
    status = descriptor_status(api)
    spec = importlib.util.find_spec("sam3")
    paths = sorted(Path(spec.origin).parent.rglob("*.py")) if spec and spec.origin else []
    paths.append(Path(__file__))
    sha = api._class_analysis_file_sha256_cached
    versions = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "numpy", "Pillow", "timm", "sam3"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        if not (name == "sam3" or name.startswith("sam3.") or name == __name__):
            continue
        path = Path(getattr(module, "__file__", "") or "")
        if path.suffix != ".py" or not path.is_file():
            continue
        expected = _compiled_functions(path)
        for label, value in vars(module).items():
            functions = [(label, value)] if inspect.isfunction(value) else list(vars(value).items()) if inspect.isclass(value) and value.__module__ == name else []
            for key, function in functions:
                if isinstance(function, (staticmethod, classmethod)):
                    function = function.__func__
                if not inspect.isfunction(function) or function.__module__ != name:
                    continue
                function = inspect.unwrap(function)
                code = function.__code__
                qualname = function.__qualname__
                # Generated dataclass helpers carry their own loaded-code digest;
                # every source-defined function must match the currently pinned file.
                generated = code.co_filename == "<string>"
                if not generated and expected.get(qualname) != code:
                    raise RuntimeError("sam3_loaded_source_changed:" + name + "." + key)
                if name == __name__:
                    # Validate every loaded native function, but identify the
                    # package by its complete source tree. Lazy native imports
                    # must not change the producer after the first model load.
                    loaded[name + "." + qualname] = hashlib.sha256(
                        (sha(path) + qualname).encode() + (marshal.dumps(code, 0) if generated else b"")
                    ).hexdigest()
    revision = None
    if spec and spec.origin:
        package = Path(spec.origin).parent
        if (package.parent / ".git").exists():
            result = subprocess.run(["git", "-C", str(package.parent), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                revision = result.stdout.strip()
        manifest = package.parent / "tator-source.json"
        if manifest.is_file():
            provenance = json.loads(manifest.read_text())
            actual = {str(p.relative_to(package.parent)): sha(p) for p in package.rglob("*.py")}
            if (not re.fullmatch(r"[0-9a-f]{40}", provenance.get("base_revision", ""))
                    or provenance.get("python_source_sha256") != actual):
                raise RuntimeError("sam3_vendor_source_manifest_changed")
            revision = provenance["base_revision"]
            paths.append(manifest)
    files = [Path(status["model_path"])] if status["available"] else []
    bpe = getattr(api, "SAM3_BPE_PATH", None)
    if bpe and Path(bpe).is_file():
        files.append(Path(bpe))
    return {"backend": "torch", "descriptor_scales": 3, "dimension": 768,
            "policy": "full_source_image_mask_pool_or_bbox_fallback", "status": status,
            "implementation_abi": NATIVE_DESCRIPTOR_ABI, "source_revision": revision,
            "descriptor_profile": dict(HISTORICAL_DESCRIPTOR_PROFILE),
            "source_revision_policy": "Base revision is context; exact implementation file hashes define compatibility.",
            "checkpoint_artifacts": [{"name": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for p in files],
            "library_versions": versions, "implementation_files": [{"path": str(p), "sha256": sha(p)} for p in paths],
            "loaded_implementation": loaded, "persistent_cache_safe": bool(status["available"] and spec),
            "device": str(getattr(api, "SAM3_DEVICE_PREF", "cpu")), "dtype": "float32",
            "math_policy": {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                            "float32_matmul_precision": torch.get_float32_matmul_precision(),
                            "autocast": "disabled_by_native_context"}}


def descriptor_backend():
    requested = os.environ.get("TATOR_SAM3_DESCRIPTOR_BACKEND", "auto").strip().lower()
    if requested == "auto":
        return "mlx-bf16" if platform.system() == "Darwin" and platform.machine() == "arm64" else "torch"
    if requested not in {"mlx-bf16", "torch"}:
        raise ValueError("sam3_descriptor_backend_invalid")
    return requested


class NativeSam3DescriptorPredictor:
    """Own one image's raw tracker maps beside the untouched native predictor."""

    def __init__(self, predictor, model=None):
        self.predictor = predictor
        self.model = model
        self.descriptor_features = None

    def set_image(self, image):
        self.descriptor_features = None
        captured = {}
        decoder = self.predictor.model.sam_mask_decoder
        handles = []
        try:
            for index, projection in enumerate((decoder.conv_s0, decoder.conv_s1)):
                def capture(_module, args, level=index):
                    feature = args[0]
                    if feature.ndim != 4 or feature.shape[0] != 1 or feature.shape[1] != 256:
                        raise RuntimeError("sam3_descriptor_preprojection_shape_invalid")
                    captured[level] = feature.detach()
                handles.append(projection.register_forward_pre_hook(capture))
            with native_sam3_context(getattr(self.predictor, "device", "cpu")):
                self.predictor.set_image(np.ascontiguousarray(image))
            lowest = self.predictor._features["image_embed"]
            if set(captured) != {0, 1} or lowest.ndim != 4 or lowest.shape[:2] != (1, 256):
                raise RuntimeError("sam3_descriptor_feature_maps_missing")
            # Native set_image already added no_mem_embed once. High resolution
            # maps remain before conv_s0/conv_s1; mask prediction keeps its caches.
            self.descriptor_features = [captured[0], captured[1], lowest.detach()]
        finally:
            for handle in handles:
                handle.remove()

    def predict(self, **kwargs):
        handle = None
        historical = getattr(self.model, "_tator_descriptor_profile", None) == HISTORICAL_DESCRIPTOR_PROFILE
        pure_box = (kwargs.get("box") is not None
                    and all(kwargs.get(key) is None for key in ("point_coords", "point_labels", "mask_input")))
        try:
            if historical and pure_box:
                import torch
                if np.asarray(kwargs["box"]).shape != (4,):
                    raise RuntimeError("sam3_historical_box_prompt_unsupported")
                encoder = self.predictor.model.sam_prompt_encoder
                calls = []

                def remove_dummy(module, args, options, output):
                    points = options.get("points")
                    sparse, dense = output
                    if (args or points is None or options.get("boxes") is not None
                            or tuple(points[0].shape) != (1, 2, 2)
                            or tuple(points[1].shape) != (1, 2)
                            or points[1].tolist() != [[2, 3]]
                            or tuple(sparse.shape) != (1, 3, 256)
                            or not torch.equal(sparse[:, -1], module.not_a_point_embed.weight)
                            or calls):
                        raise RuntimeError("sam3_historical_box_prompt_unsupported")
                    calls.append(True)
                    # Historical MLX passed only the two box corners. Native
                    # treats those corners as points and appends a dummy token.
                    return sparse[:, :2], dense

                handle = encoder.register_forward_hook(remove_dummy, with_kwargs=True)
            with native_sam3_context(getattr(self.predictor, "device", "cpu")):
                result = self.predictor.predict(**kwargs)
            if handle is not None and len(calls) != 1:
                raise RuntimeError("sam3_historical_box_prompt_unsupported")
            return result
        finally:
            if handle is not None:
                handle.remove()

    def unload(self):
        self.descriptor_features = None
        if hasattr(self.predictor, "reset_predictor"):
            self.predictor.reset_predictor()
        self.predictor = None
        self.model = None


def mask_feature_vector(predictor, bbox_xyxy, image_size):
    """Shared mask selection, interpolation, fallback and 3-scale normalization."""
    native = isinstance(predictor, NativeSam3DescriptorPredictor)
    features = predictor.descriptor_features if native else getattr(predictor, "_features", None)
    if not isinstance(features, (tuple, list)) or len(features) < 3:
        raise RuntimeError("sam3_mask_fusion_feature_maps_missing")
    width, height = image_size
    bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)[:4]
    if bbox.size != 4 or not np.isfinite(bbox).all() or width <= 0 or height <= 0:
        raise ValueError("sam3_descriptor_geometry_invalid")
    masks, scores, _ = predictor.predict(box=bbox, multimask_output=True)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    finite = np.flatnonzero(np.isfinite(scores))
    best = int(finite[np.argmax(scores[finite])]) if finite.size else -1
    chosen = np.asarray(masks[best], dtype=bool) if best >= 0 else np.zeros((height, width), dtype=bool)
    if chosen.shape != (height, width):
        raise RuntimeError("sam3_descriptor_mask_shape_invalid")
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
    use_mask = bool(chosen.any() and chosen[y1:y2, x1:x2].any())
    mask_image = Image.fromarray(chosen.astype(np.uint8) * 255) if use_mask else None
    components = []
    for feature in features[:3]:
        fh, fw = map(int, feature.shape[-2:] if native else feature.shape[1:3])
        channels = int(feature.shape[1] if native else feature.shape[-1])
        if channels != 256:
            raise RuntimeError("sam3_descriptor_channels_invalid")
        weights = np.asarray(mask_image.resize((fw, fh), Image.Resampling.BILINEAR), dtype=np.float32) / 255 if use_mask else None
        if not use_mask:
            fx1, fy1 = max(0, min(fw - 1, int(math.floor(float(bbox[0]) / width * fw)))), max(0, min(fh - 1, int(math.floor(float(bbox[1]) / height * fh))))
            fx2, fy2 = max(fx1 + 1, min(fw, int(math.ceil(float(bbox[2]) / width * fw)))), max(fy1 + 1, min(fh, int(math.ceil(float(bbox[3]) / height * fh))))
        if native:
            import torch
            # Accumulate in float32, including when a caller uses autocast.
            values = feature.float()
            if use_mask:
                weight = torch.as_tensor(weights, device=values.device)[None, None]
                pooled = (values * weight).sum((2, 3)) / weight.sum().clamp_min(1e-6)
            else:
                pooled = values[:, :, fy1:fy2, fx1:fx2].mean((2, 3))
            component = pooled.detach().cpu().numpy().reshape(-1)
        else:
            import mlx.core as mx
            if use_mask:
                weight = mx.array(weights)[None, :, :, None]
                pooled = mx.sum(feature * weight, axis=(1, 2)) / mx.maximum(mx.sum(weight), mx.array(1e-6))
            else:
                pooled = mx.mean(feature[:, fy1:fy2, fx1:fx2, :], axis=(1, 2))
            mx.eval(pooled)
            component = np.asarray(pooled, dtype=np.float32).reshape(-1)
        component /= max(float(np.linalg.norm(component)), 1e-12)
        components.append(component)
    vector = np.concatenate(components).astype(np.float32, copy=False)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    if not np.isfinite(vector).all() or vector.size != 768:
        raise RuntimeError("sam3_descriptor_invalid")
    return vector, not use_mask, float(scores[best]) if best >= 0 else None
