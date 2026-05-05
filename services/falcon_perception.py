"""Lazy Falcon-Perception runtime helpers."""

from __future__ import annotations

import base64
import importlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
import torch
from pycocotools import mask as mask_utils
from PIL import Image


FALCON_PERCEPTION_MIN_TORCH = (2, 5)

_FALCON_RUNTIME: Dict[str, Any] = {
    "model": None,
    "model_id": None,
    "device": None,
}

_FALCON_OFFICIAL_RUNTIME: Dict[str, Any] = {
    "engine": None,
    "tokenizer": None,
    "model_source": None,
    "device": None,
    "repo_root": None,
    "falcon_pkg": None,
    "paged_module": None,
    "visualization_module": None,
}
_FALCON_OFFICIAL_SEG_RUNTIME: Dict[str, Any] = {
    "engine": None,
    "tokenizer": None,
    "model_source": None,
    "device": None,
    "repo_root": None,
    "falcon_pkg": None,
    "paged_module": None,
    "visualization_module": None,
    "max_image_size": None,
}

_FALCON_OFFICIAL_REPO_URL = "https://github.com/tiiuae/Falcon-Perception"
_FALCON_SERVER_URL_ENV = "FALCON_SERVER_URL"

_FALCON_FLEX_IMPORT_OLD = """from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
)
"""
_FALCON_FLEX_IMPORT_NEW = "from torch.nn.attention.flex_attention import BlockMask\n"
_FALCON_FLEX_CALL_OLD = (
    "        output, aux_output = flex_fn(xq, xk, xv, block_mask=attention_masks, "
    "return_aux=AuxRequest(lse=True))\n"
    "        return self._post_attention(output, aux_output.lse)\n"
)
_FALCON_FLEX_CALL_NEW = (
    "        output, lse = flex_fn(xq, xk, xv, block_mask=attention_masks, return_lse=True)\n"
    "        return self._post_attention(output, lse)\n"
)
_FALCON_SQUARED_RELU_GATE_OLD = """def squared_relu_gate(packed: T, hidden_dim: int) -> T:
    packed_2d = packed.flatten(0, -2)
    n_rows = packed_2d.shape[0]
    n_cols = hidden_dim
    out_2d = torch.empty((n_rows, n_cols), device=packed.device, dtype=packed.dtype)
    n = n_rows * n_cols
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _squared_relu_gate_kernel[grid](
        packed_2d, out_2d, n_rows, n_cols,
        packed_2d.stride(0), packed_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        BLOCK_SIZE=1024,
    )
    return out_2d.view(*packed.shape[:-1], hidden_dim)
"""
_FALCON_SQUARED_RELU_GATE_NEW = """def squared_relu_gate(packed: T, hidden_dim: int) -> T:
    gate, up = packed.split(hidden_dim, dim=-1)
    gate = torch.relu(gate)
    return gate.square() * up
"""
_FALCON_ATTN_COMPILE_DECODE_OLD = "compiled_flex_attn_decode = torch.compile(flex_attention, fullgraph=True)\n"
_FALCON_ATTN_COMPILE_DECODE_NEW = "compiled_flex_attn_decode = flex_attention\n"
_FALCON_ATTN_COMPILE_PREFILL_OLD = "compiled_flex_attn_prefill = torch.compile(flex_attention, dynamic=True)\n"
_FALCON_ATTN_COMPILE_PREFILL_NEW = "compiled_flex_attn_prefill = flex_attention\n"
_FALCON_CREATE_BLOCK_MASK_OLD = """_compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=True
) # Note: can't use mode = 'reduce-overhead' here because it uses internal CUDA graph trees on private streams, causing manual capture to record empty graphs
"""
_FALCON_CREATE_BLOCK_MASK_NEW = "_compiled_create_block_mask = create_block_mask\n"
_FALCON_INDEX_ALIGN_HELPER = """

def _align_lookup_tensor(tensor, index):
    if tensor.device == index.device:
        return tensor
    if index.device.type == "cpu":
        return tensor.cpu()
    return tensor.to(index.device)
"""
_FALCON_INDEX_ALIGN_HELPER_TYPED = """

def _align_lookup_tensor(tensor: T, index: T) -> T:
    if tensor.device == index.device:
        return tensor
    if index.device.type == "cpu":
        return tensor.cpu()
    return tensor.to(index.device)
"""
_FALCON_DOCUMENT_MASK_OLD = """    def document_mask(b: T, h: T, q_idx: T, kv_idx: T) -> T:
        return sequence_indices[b, q_idx] == sequence_indices[b, kv_idx]
"""
_FALCON_DOCUMENT_MASK_NEW = """    def document_mask(b: T, h: T, q_idx: T, kv_idx: T) -> T:
        sequence_local = _align_lookup_tensor(sequence_indices, q_idx)
        return sequence_local[b, q_idx] == sequence_local[b, kv_idx]
"""
_FALCON_NONPAD_MASK_OLD = """    def mask_mod(b, h, q_idx, kv_idx):
        return non_pad_mask_id[b, kv_idx] > 0
"""
_FALCON_NONPAD_MASK_NEW = """    def mask_mod(b, h, q_idx, kv_idx):
        non_pad_local = _align_lookup_tensor(non_pad_mask_id, kv_idx)
        return non_pad_local[b, kv_idx] > 0
"""
_FALCON_IMAGE_PREFIX_MASK_OLD = """    def image_prefix_mask_mod(b, h, q_idx, kv_idx):
        # Check if both tokens are image tokens and belong to the same image
        is_img_tokens = img_mask[b, q_idx] & img_mask[b, kv_idx]
        is_same_image = img_indices[b, q_idx] == img_indices[b, kv_idx]
        return is_img_tokens & is_same_image
"""
_FALCON_IMAGE_PREFIX_MASK_NEW = """    def image_prefix_mask_mod(b, h, q_idx, kv_idx):
        # Check if both tokens are image tokens and belong to the same image
        img_mask_local = _align_lookup_tensor(img_mask, q_idx)
        img_indices_local = _align_lookup_tensor(img_indices, q_idx)
        is_img_tokens = img_mask_local[b, q_idx] & img_mask_local[b, kv_idx]
        is_same_image = img_indices_local[b, q_idx] == img_indices_local[b, kv_idx]
        return is_img_tokens & is_same_image
"""
_FALCON_ANYUP_MASK_OLD = """    def _mask_mod(b_idx, h_idx, q_idx, kv_idx):
        q_r_idx = q_idx // w
        q_c_idx = q_idx % w
        kv_r_idx = kv_idx // w_
        kv_c_idx = kv_idx % w_
        row_lower = kv_r_idx >= r0[q_r_idx, q_c_idx]
        row_upper = kv_r_idx < r1[q_r_idx, q_c_idx]
        col_lower = kv_c_idx >= c0[q_r_idx, q_c_idx]
        col_upper = kv_c_idx < c1[q_r_idx, q_c_idx]

        return row_lower & row_upper & col_lower & col_upper
"""
_FALCON_ANYUP_MASK_NEW = """    def _mask_mod(b_idx, h_idx, q_idx, kv_idx):
        q_r_idx = q_idx // w
        q_c_idx = q_idx % w
        kv_r_idx = kv_idx // w_
        kv_c_idx = kv_idx % w_
        r0_local = _align_lookup_tensor(r0, q_idx)
        r1_local = _align_lookup_tensor(r1, q_idx)
        c0_local = _align_lookup_tensor(c0, q_idx)
        c1_local = _align_lookup_tensor(c1, q_idx)
        row_lower = kv_r_idx >= r0_local[q_r_idx, q_c_idx]
        row_upper = kv_r_idx < r1_local[q_r_idx, q_c_idx]
        col_lower = kv_c_idx >= c0_local[q_r_idx, q_c_idx]
        col_upper = kv_c_idx < c1_local[q_r_idx, q_c_idx]

        return row_lower & row_upper & col_lower & col_upper
"""
_FALCON_UPSAMPLER_MASK_OLD = """    def get_upsampler_attn_mask(self, H, W, h, w, device):
        return create_attention_mask(
            get_upsampler_attn_mask_mod(H, W, h, w, device=device),
            B=None, H=None, Q_LEN=H * W, KV_LEN=h * w,
        )
"""
_FALCON_UPSAMPLER_MASK_NEW = """    def get_upsampler_attn_mask(self, H, W, h, w, device):
        return create_attention_mask(
            get_upsampler_attn_mask_mod(H, W, h, w, device=device),
            B=None, H=None, Q_LEN=H * W, KV_LEN=h * w, device=device,
        )
"""
_FALCON_BATCH_MASK_RETURN_OLD = "    return create_attention_mask(mask_mod, B, None, max_len, max_len)\n"
_FALCON_BATCH_MASK_RETURN_NEW = (
    "    return create_attention_mask(mask_mod, B, None, max_len, max_len, device=input_batch.device)\n"
)


def _torch_version_tuple(torch_module: Any = torch) -> Tuple[int, int]:
    raw = str(getattr(torch_module, "__version__", "0.0")).split("+", 1)[0]
    parts = raw.split(".")
    try:
        major = int(parts[0])
    except Exception:
        major = 0
    try:
        minor = int(parts[1])
    except Exception:
        minor = 0
    return major, minor


def falcon_runtime_supported(torch_module: Any = torch) -> bool:
    return _torch_version_tuple(torch_module) >= FALCON_PERCEPTION_MIN_TORCH


def falcon_runtime_error_detail(torch_module: Any = torch) -> Optional[str]:
    if falcon_runtime_supported(torch_module):
        return None
    major, minor = _torch_version_tuple(torch_module)
    return (
        f"falcon_requires_torch_{FALCON_PERCEPTION_MIN_TORCH[0]}_"
        f"{FALCON_PERCEPTION_MIN_TORCH[1]}:current={major}.{minor}"
    )


def _configure_falcon_torch_runtime(torch_module: Any = torch) -> None:
    try:
        import torch._dynamo as torch_dynamo  # local import

        torch_dynamo.config.capture_scalar_outputs = True
    except Exception:
        pass


def _patch_falcon_modeling_file(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return
    updated = raw
    if _FALCON_FLEX_IMPORT_OLD in updated:
        updated = updated.replace(_FALCON_FLEX_IMPORT_OLD, _FALCON_FLEX_IMPORT_NEW, 1)
    if _FALCON_FLEX_CALL_OLD in updated:
        updated = updated.replace(_FALCON_FLEX_CALL_OLD, _FALCON_FLEX_CALL_NEW, 1)
    if _FALCON_UPSAMPLER_MASK_OLD in updated:
        updated = updated.replace(_FALCON_UPSAMPLER_MASK_OLD, _FALCON_UPSAMPLER_MASK_NEW, 1)
    updated = re.sub(
        r"def squared_relu_gate\(packed: T, hidden_dim: int\) -> T:\n(?:    .*\n)+?    return out_2d.view\(\*packed.shape\[:-1\], hidden_dim\)\n",
        _FALCON_SQUARED_RELU_GATE_NEW,
        updated,
        count=1,
    )
    if updated == raw:
        return
    path.write_text(updated, encoding="utf-8")


def _patch_falcon_tokenizer_config(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if str(payload.get("tokenizer_class") or "").strip() != "TokenizersBackend":
        return
    payload["tokenizer_class"] = "PreTrainedTokenizerFast"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _patch_falcon_attention_file(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return
    updated = raw
    updated = updated.replace(_FALCON_INDEX_ALIGN_HELPER_TYPED, _FALCON_INDEX_ALIGN_HELPER)
    updated = updated.replace(_FALCON_ATTN_COMPILE_DECODE_OLD, _FALCON_ATTN_COMPILE_DECODE_NEW, 1)
    updated = updated.replace(_FALCON_ATTN_COMPILE_PREFILL_OLD, _FALCON_ATTN_COMPILE_PREFILL_NEW, 1)
    updated = updated.replace(_FALCON_CREATE_BLOCK_MASK_OLD, _FALCON_CREATE_BLOCK_MASK_NEW, 1)
    if "def _align_lookup_tensor" not in updated and "compiled_flex_attn_prefill = flex_attention" in updated:
        updated = updated.replace(
            "compiled_flex_attn_prefill = flex_attention",
            "compiled_flex_attn_prefill = flex_attention" + _FALCON_INDEX_ALIGN_HELPER,
            1,
        )
    updated = updated.replace(_FALCON_DOCUMENT_MASK_OLD, _FALCON_DOCUMENT_MASK_NEW, 1)
    updated = updated.replace(_FALCON_NONPAD_MASK_OLD, _FALCON_NONPAD_MASK_NEW, 1)
    updated = updated.replace(_FALCON_IMAGE_PREFIX_MASK_OLD, _FALCON_IMAGE_PREFIX_MASK_NEW, 1)
    updated = updated.replace(_FALCON_BATCH_MASK_RETURN_OLD, _FALCON_BATCH_MASK_RETURN_NEW, 1)
    if updated == raw:
        return
    path.write_text(updated, encoding="utf-8")


def _patch_falcon_anyup_file(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return
    updated = raw.replace(_FALCON_INDEX_ALIGN_HELPER_TYPED, _FALCON_INDEX_ALIGN_HELPER)
    updated = updated.replace(_FALCON_ATTN_COMPILE_PREFILL_OLD, _FALCON_ATTN_COMPILE_PREFILL_NEW, 1)
    if "def _align_lookup_tensor" not in updated and "compiled_flex_attn_prefill = flex_attention" in updated:
        updated = updated.replace(
            "compiled_flex_attn_prefill = flex_attention",
            "compiled_flex_attn_prefill = flex_attention" + _FALCON_INDEX_ALIGN_HELPER,
            1,
        )
    updated = re.sub(
        r"    def _mask_mod\(b_idx, h_idx, q_idx, kv_idx\):\n(?:        .*\n)+?        return row_lower & row_upper & col_lower & col_upper\n",
        _FALCON_ANYUP_MASK_NEW,
        updated,
        count=1,
    )
    if updated == raw:
        return
    path.write_text(updated, encoding="utf-8")


def _patch_falcon_source_tree(source_root: Path) -> None:
    modeling_path = source_root / "modeling_falcon_perception.py"
    if modeling_path.exists():
        _patch_falcon_modeling_file(modeling_path)
    tokenizer_config_path = source_root / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        _patch_falcon_tokenizer_config(tokenizer_config_path)
    attention_path = source_root / "attention.py"
    if attention_path.exists():
        _patch_falcon_attention_file(attention_path)
    anyup_path = source_root / "anyup.py"
    if anyup_path.exists():
        _patch_falcon_anyup_file(anyup_path)


def _patch_falcon_module_cache(snapshot_dir: Path) -> None:
    snapshot_name = str(snapshot_dir.name or "").strip()
    if not snapshot_name:
        return
    cache_root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    candidates = [cache_root / f"_{snapshot_name}"]
    for candidate in candidates:
        if candidate.exists():
            _patch_falcon_source_tree(candidate)


def _official_repo_root() -> Path:
    override = str(os.environ.get("FALCON_PERCEPTION_SOURCE_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".cache" / "tator" / "Falcon-Perception").resolve()


def _ensure_official_repo_source() -> Path:
    repo_root = _official_repo_root()
    if (repo_root / "falcon_perception").exists():
        return repo_root
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", _FALCON_OFFICIAL_REPO_URL, str(repo_root)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return repo_root


def _patch_official_model_file(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return
    updated = raw.replace(
        "from torch.nn.attention.flex_attention import (\n    AuxRequest,\n    BlockMask,\n)\n",
        "from torch.nn.attention.flex_attention import BlockMask\n",
    )
    updated = updated.replace(
        "        output, aux_output = flex_fn(\n"
        "            xq, xk, xv,\n"
        "            block_mask=attention_masks,\n"
        "            return_aux=AuxRequest(lse=True),\n"
        "            kernel_options=flex_attn_kernel_options,\n"
        "        )\n"
        "        output = self._post_attention(output, aux_output.lse)\n",
        "        output, lse = flex_fn(\n"
        "            xq, xk, xv,\n"
        "            block_mask=attention_masks,\n"
        "            return_lse=True,\n"
        "            kernel_options=flex_attn_kernel_options,\n"
        "        )\n"
        "        output = self._post_attention(output, lse)\n",
    )
    updated = re.sub(
        r"def squared_relu_gate\(packed: T, hidden_dim: int\) -> T:\n(?:    .*\n)+?    return out_2d.view\(\*packed.shape\[:-1\], hidden_dim\)\n",
        "def squared_relu_gate(packed: T, hidden_dim: int) -> T:\n"
        "    gate, up = packed.split(hidden_dim, dim=-1)\n"
        "    gate = torch.relu(gate)\n"
        "    return gate.square() * up\n",
        updated,
        count=1,
    )
    if updated != raw:
        path.write_text(updated, encoding="utf-8")


def _patch_official_attention_file(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return
    updated = raw.replace(
        "compiled_flex_attn_decode = torch.compile(flex_attention, fullgraph=True)\n",
        "compiled_flex_attn_decode = flex_attention\n",
    )
    updated = updated.replace(
        "compiled_flex_attn_prefill = torch.compile(flex_attention, dynamic=True)\n",
        "compiled_flex_attn_prefill = flex_attention\n",
    )
    updated = updated.replace(
        "_compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)\n",
        "_compiled_create_block_mask = create_block_mask\n",
    )
    helper = (
        "\n\ndef _align_lookup_tensor(tensor, index):\n"
        "    if tensor.device == index.device:\n"
        "        return tensor\n"
        "    if index.device.type == \"cpu\":\n"
        "        return tensor.cpu()\n"
        "    return tensor.to(index.device)\n"
    )
    if "def _align_lookup_tensor" not in updated:
        updated = updated.replace("def offset_mask_mod", helper + "\ndef offset_mask_mod", 1)
    updated = updated.replace(
        "    def document_mask(b: T, h: T, q_idx: T, kv_idx: T) -> T:\n"
        "        return sequence_indices[b, q_idx] == sequence_indices[b, kv_idx]\n",
        "    def document_mask(b: T, h: T, q_idx: T, kv_idx: T) -> T:\n"
        "        sequence_local = _align_lookup_tensor(sequence_indices, q_idx)\n"
        "        return sequence_local[b, q_idx] == sequence_local[b, kv_idx]\n",
    )
    updated = updated.replace(
        "    def mask_mod(b, h, q_idx, kv_idx):\n"
        "        return non_pad_mask_id[b, kv_idx] > 0\n",
        "    def mask_mod(b, h, q_idx, kv_idx):\n"
        "        non_pad_local = _align_lookup_tensor(non_pad_mask_id, kv_idx)\n"
        "        return non_pad_local[b, kv_idx] > 0\n",
    )
    updated = updated.replace(
        "    def image_prefix_mask_mod(b, h, q_idx, kv_idx):\n"
        "        # Check if both tokens are image tokens and belong to the same image\n"
        "        is_img_tokens = img_mask[b, q_idx] & img_mask[b, kv_idx]\n"
        "        is_same_image = img_indices[b, q_idx] == img_indices[b, kv_idx]\n"
        "        return is_img_tokens & is_same_image\n",
        "    def image_prefix_mask_mod(b, h, q_idx, kv_idx):\n"
        "        img_mask_local = _align_lookup_tensor(img_mask, q_idx)\n"
        "        img_indices_local = _align_lookup_tensor(img_indices, q_idx)\n"
        "        is_img_tokens = img_mask_local[b, q_idx] & img_mask_local[b, kv_idx]\n"
        "        is_same_image = img_indices_local[b, q_idx] == img_indices_local[b, kv_idx]\n"
        "        return is_img_tokens & is_same_image\n",
    )
    updated = updated.replace(
        "    return create_attention_mask(mask_mod, B, None, max_len, max_len)\n",
        "    return create_attention_mask(mask_mod, B, None, max_len, max_len, device=input_batch.device)\n",
    )
    if updated != raw:
        path.write_text(updated, encoding="utf-8")


def _patch_official_anyup_file(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return
    updated = raw
    helper = (
        "from falcon_perception.attention import compiled_flex_attn_prefill\n\n\n"
        "def _align_lookup_tensor(tensor, index):\n"
        "    if tensor.device == index.device:\n"
        "        return tensor\n"
        "    if index.device.type == \"cpu\":\n"
        "        return tensor.cpu()\n"
        "    return tensor.to(index.device)\n"
    )
    if "def _align_lookup_tensor" not in updated:
        updated = updated.replace(
            "from falcon_perception.attention import compiled_flex_attn_prefill\n",
            helper,
            1,
        )
    updated = updated.replace(
        "    def _mask_mod(b_idx, h_idx, q_idx, kv_idx):\n"
        "        q_r_idx = q_idx // w\n"
        "        q_c_idx = q_idx % w\n"
        "        kv_r_idx = kv_idx // w_\n"
        "        kv_c_idx = kv_idx % w_\n"
        "        row_lower = kv_r_idx >= r0[q_r_idx, q_c_idx]\n"
        "        row_upper = kv_r_idx < r1[q_r_idx, q_c_idx]\n"
        "        col_lower = kv_c_idx >= c0[q_r_idx, q_c_idx]\n"
        "        col_upper = kv_c_idx < c1[q_r_idx, q_c_idx]\n\n"
        "        return row_lower & row_upper & col_lower & col_upper\n",
        "    def _mask_mod(b_idx, h_idx, q_idx, kv_idx):\n"
        "        q_r_idx = q_idx // w\n"
        "        q_c_idx = q_idx % w\n"
        "        kv_r_idx = kv_idx // w_\n"
        "        kv_c_idx = kv_idx % w_\n"
        "        r0_local = _align_lookup_tensor(r0, q_idx)\n"
        "        r1_local = _align_lookup_tensor(r1, q_idx)\n"
        "        c0_local = _align_lookup_tensor(c0, q_idx)\n"
        "        c1_local = _align_lookup_tensor(c1, q_idx)\n"
        "        row_lower = kv_r_idx >= r0_local[q_r_idx, q_c_idx]\n"
        "        row_upper = kv_r_idx < r1_local[q_r_idx, q_c_idx]\n"
        "        col_lower = kv_c_idx >= c0_local[q_r_idx, q_c_idx]\n"
        "        col_upper = kv_c_idx < c1_local[q_r_idx, q_c_idx]\n\n"
        "        return row_lower & row_upper & col_lower & col_upper\n",
    )
    if updated != raw:
        path.write_text(updated, encoding="utf-8")


def _patch_official_paged_inference_file(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return
    updated = raw.replace(
        "seq.image_hash = int.from_bytes(hashlib.sha256(img.tobytes()).digest()[:8])",
        'seq.image_hash = int.from_bytes(hashlib.sha256(img.tobytes()).digest()[:8], "big")',
    )
    if updated != raw:
        path.write_text(updated, encoding="utf-8")


def _patch_official_data_file(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return
    updated = raw.replace(
        "    mask_thw = E.reduce(\n"
        "        pixel_mask_THW,\n"
        "        \"(t tp) (h hp) (w wp) -> t h w\",\n"
        "        reduction=\"any\",\n"
        "        tp=temporal_patch_size,\n"
        "        hp=spatial_patch_size,\n"
        "        wp=spatial_patch_size,\n"
        "    )\n"
        "    width = int(E.reduce(mask_thw.sum(axis=-1).astype(int), \"t h -> \", reduction=\"max\"))\n"
        "    height = int(E.reduce(mask_thw.sum(axis=-2).astype(int), \"t w -> \", reduction=\"max\"))\n",
        "    pixel_mask_THW = np.asarray(pixel_mask_THW, dtype=bool)\n"
        "    T, H, W = pixel_mask_THW.shape\n"
        "    mask_thw = pixel_mask_THW.reshape(\n"
        "        T // temporal_patch_size,\n"
        "        temporal_patch_size,\n"
        "        H // spatial_patch_size,\n"
        "        spatial_patch_size,\n"
        "        W // spatial_patch_size,\n"
        "        spatial_patch_size,\n"
        "    ).any(axis=(1, 3, 5))\n"
        "    width = int(mask_thw.sum(axis=-1).astype(int).max())\n"
        "    height = int(mask_thw.sum(axis=-2).astype(int).max())\n",
    )
    if updated != raw:
        path.write_text(updated, encoding="utf-8")


def _patch_official_repo_tree(repo_root: Path) -> None:
    package_root = repo_root / "falcon_perception"
    _patch_official_model_file(package_root / "model.py")
    _patch_official_attention_file(package_root / "attention.py")
    _patch_official_anyup_file(package_root / "anyup.py")
    _patch_official_paged_inference_file(package_root / "paged_inference.py")
    _patch_official_data_file(package_root / "data.py")


def _import_official_falcon_modules(repo_root: Path) -> Dict[str, Any]:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    importlib.invalidate_caches()
    falcon_pkg = importlib.import_module("falcon_perception")
    data_module = importlib.import_module("falcon_perception.data")
    paged_module = importlib.import_module("falcon_perception.paged_inference")
    visualization_module = importlib.import_module("falcon_perception.visualization_utils")
    return {
        "falcon_pkg": falcon_pkg,
        "data_module": data_module,
        "paged_module": paged_module,
        "visualization_module": visualization_module,
    }


def _resolve_falcon_model_source(*, model_id: str, local_files_only: bool) -> str:
    raw_model_id = str(model_id or "").strip()
    if not raw_model_id:
        raise RuntimeError("falcon_model_id_required")
    local_candidate = Path(raw_model_id).expanduser()
    if local_candidate.exists():
        resolved = local_candidate.resolve()
        _patch_falcon_source_tree(resolved)
        _patch_falcon_module_cache(resolved)
        return str(resolved)
    from huggingface_hub import snapshot_download  # local import

    snapshot_dir = Path(
        snapshot_download(
            repo_id=raw_model_id,
            local_files_only=bool(local_files_only),
        )
    ).resolve()
    _patch_falcon_source_tree(snapshot_dir)
    _patch_falcon_module_cache(snapshot_dir)
    return str(snapshot_dir)


def ensure_falcon_runtime(
    *,
    model_id: str,
    device: str,
    local_files_only: bool,
) -> Any:
    if not falcon_runtime_supported(torch):
        raise RuntimeError(falcon_runtime_error_detail(torch) or "falcon_requires_newer_torch")
    _configure_falcon_torch_runtime(torch)
    resolved_model_source = _resolve_falcon_model_source(
        model_id=model_id,
        local_files_only=local_files_only,
    )
    cached_model = _FALCON_RUNTIME.get("model")
    cached_model_id = str(_FALCON_RUNTIME.get("model_id") or "")
    cached_device = str(_FALCON_RUNTIME.get("device") or "")
    if cached_model is not None and cached_model_id == resolved_model_source and cached_device == str(device):
        return cached_model
    from transformers import AutoModelForCausalLM  # local import

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_source,
        trust_remote_code=True,
        device_map={"": device},
        local_files_only=True,
    )
    _FALCON_RUNTIME["model"] = model
    _FALCON_RUNTIME["model_id"] = resolved_model_source
    _FALCON_RUNTIME["device"] = str(device)
    return model


def ensure_official_falcon_detection_runtime(
    *,
    model_id: str,
    device: str,
    local_files_only: bool,
) -> Dict[str, Any]:
    if not falcon_runtime_supported(torch):
        raise RuntimeError(falcon_runtime_error_detail(torch) or "falcon_requires_newer_torch")
    _configure_falcon_torch_runtime(torch)
    repo_root = _ensure_official_repo_source()
    _patch_official_repo_tree(repo_root)
    modules = _import_official_falcon_modules(repo_root)
    resolved_model_source = _resolve_falcon_model_source(
        model_id=model_id,
        local_files_only=local_files_only,
    )
    cached_engine = _FALCON_OFFICIAL_RUNTIME.get("engine")
    if (
        cached_engine is not None
        and str(_FALCON_OFFICIAL_RUNTIME.get("model_source") or "") == resolved_model_source
        and str(_FALCON_OFFICIAL_RUNTIME.get("device") or "") == str(device)
        and str(_FALCON_OFFICIAL_RUNTIME.get("repo_root") or "") == str(repo_root)
    ):
        return _FALCON_OFFICIAL_RUNTIME

    falcon_pkg = modules["falcon_pkg"]
    data_module = modules["data_module"]
    paged_module = modules["paged_module"]
    visualization_module = modules["visualization_module"]

    falcon_pkg.setup_torch_config()
    model, tokenizer, model_args = falcon_pkg.load_from_hf_export(hf_local_dir=resolved_model_source)
    model = model.to(device=str(device), dtype=torch.float32)
    image_processor = data_module.ImageProcessor(
        patch_size=int(getattr(model_args, "spatial_patch_size", 16) or 16),
        merge_size=1,
    )
    engine = paged_module.PagedInferenceEngine(
        model,
        tokenizer,
        image_processor,
        max_batch_size=8,
        max_seq_length=8192,
        n_pages=128,
        page_size=128,
        prefill_length_limit=8192,
        enable_hr_cache=False,
        capture_cudagraph=False,
    )
    _FALCON_OFFICIAL_RUNTIME.update(
        {
            "engine": engine,
            "tokenizer": tokenizer,
            "model_source": resolved_model_source,
            "device": str(device),
            "repo_root": str(repo_root),
            "falcon_pkg": falcon_pkg,
            "paged_module": paged_module,
            "visualization_module": visualization_module,
        }
    )
    return _FALCON_OFFICIAL_RUNTIME


def ensure_official_falcon_segmentation_runtime(
    *,
    model_id: str,
    device: str,
    local_files_only: bool,
    max_image_size: int,
) -> Dict[str, Any]:
    if not falcon_runtime_supported(torch):
        raise RuntimeError(falcon_runtime_error_detail(torch) or "falcon_requires_newer_torch")
    _configure_falcon_torch_runtime(torch)
    repo_root = _ensure_official_repo_source()
    _patch_official_repo_tree(repo_root)
    modules = _import_official_falcon_modules(repo_root)
    resolved_model_source = _resolve_falcon_model_source(
        model_id=model_id,
        local_files_only=local_files_only,
    )
    cached_engine = _FALCON_OFFICIAL_SEG_RUNTIME.get("engine")
    if (
        cached_engine is not None
        and str(_FALCON_OFFICIAL_SEG_RUNTIME.get("model_source") or "") == resolved_model_source
        and str(_FALCON_OFFICIAL_SEG_RUNTIME.get("device") or "") == str(device)
        and str(_FALCON_OFFICIAL_SEG_RUNTIME.get("repo_root") or "") == str(repo_root)
        and int(_FALCON_OFFICIAL_SEG_RUNTIME.get("max_image_size") or 0) == int(max_image_size)
    ):
        _disable_unsafe_segmentation_hr_cache(cached_engine)
        return _FALCON_OFFICIAL_SEG_RUNTIME

    falcon_pkg = modules["falcon_pkg"]
    data_module = modules["data_module"]
    paged_module = modules["paged_module"]
    visualization_module = modules["visualization_module"]

    falcon_pkg.setup_torch_config()
    model, tokenizer, model_args = falcon_pkg.load_from_hf_export(hf_local_dir=resolved_model_source)
    model = model.to(device=str(device), dtype=torch.float32)
    image_processor = data_module.ImageProcessor(
        patch_size=int(getattr(model_args, "spatial_patch_size", 16) or 16),
        merge_size=1,
    )
    cfg = paged_module.engine_config_for_gpu(max_image_size=int(max_image_size), dtype=model.dtype)
    cfg["max_batch_size"] = min(int(cfg.get("max_batch_size") or 8), 8)
    cfg["n_pages"] = min(int(cfg.get("n_pages") or 128), 128)
    cfg["prefill_length_limit"] = min(int(cfg.get("prefill_length_limit") or 8192), 8192)
    cfg["max_decode_steps_between_prefills"] = min(
        int(cfg.get("max_decode_steps_between_prefills") or 8),
        8,
    )
    engine = paged_module.PagedInferenceEngine(
        model,
        tokenizer,
        image_processor,
        max_seq_length=8192,
        capture_cudagraph=False,
        enable_hr_cache=False,
        **cfg,
    )
    _disable_unsafe_segmentation_hr_cache(engine)
    _FALCON_OFFICIAL_SEG_RUNTIME.update(
        {
            "engine": engine,
            "tokenizer": tokenizer,
            "model_source": resolved_model_source,
            "device": str(device),
            "repo_root": str(repo_root),
            "falcon_pkg": falcon_pkg,
            "paged_module": paged_module,
            "visualization_module": visualization_module,
            "max_image_size": int(max_image_size),
        }
    )
    return _FALCON_OFFICIAL_SEG_RUNTIME


def unload_falcon_runtime() -> None:
    model = _FALCON_RUNTIME.get("model")
    _FALCON_RUNTIME["model"] = None
    _FALCON_RUNTIME["model_id"] = None
    _FALCON_RUNTIME["device"] = None
    if model is not None:
        try:
            del model
        except Exception:
            pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def unload_official_falcon_runtime() -> None:
    for cache in (_FALCON_OFFICIAL_RUNTIME, _FALCON_OFFICIAL_SEG_RUNTIME):
        engine = cache.get("engine")
        cache.update(
            {
                "engine": None,
                "tokenizer": None,
                "model_source": None,
                "device": None,
                "repo_root": None,
                "falcon_pkg": None,
                "paged_module": None,
                "visualization_module": None,
                "max_image_size": None,
            }
        )
        if engine is not None:
            try:
                del engine
            except Exception:
                pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _disable_unsafe_segmentation_hr_cache(engine: Any) -> None:
    """Disable query-unsafe HR cache reuse for Falcon segmentation.

    Falcon's segmentation upsampler derives HR image features from the current
    sequence hidden states, so reusing those features across different prompts
    on the same image is invalid. The upstream paged engine caches by image hash
    only, which can corrupt later query/window results.
    """
    if engine is None:
        return
    try:
        engine.enable_hr_cache = False
    except Exception:
        pass
    try:
        cache = getattr(engine, "_hr_features_cache", None)
        if hasattr(cache, "clear"):
            cache.clear()
    except Exception:
        pass


def _normalize_rle(mask_rle: Dict[str, Any]) -> Dict[str, Any]:
    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    return {"size": size, "counts": counts}


def resize_mask_rle(
    mask_rle: Dict[str, Any],
    *,
    target_height: int,
    target_width: int,
) -> Dict[str, Any]:
    normalized = _normalize_rle(mask_rle)
    size = normalized.get("size") or []
    if len(size) >= 2 and int(size[0]) == int(target_height) and int(size[1]) == int(target_width):
        counts = normalized.get("counts")
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {"size": [int(target_height), int(target_width)], "counts": counts}
    decoded = mask_utils.decode(normalized)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    resized = np.asfortranarray(
        np.array(
            Image.fromarray(decoded.astype(np.uint8)).resize(
                (int(target_width), int(target_height)),
                Image.NEAREST,
            )
        ).astype(np.uint8)
    )
    encoded = mask_utils.encode(resized)
    counts = encoded.get("counts")
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(target_height), int(target_width)], "counts": counts}


def decode_mask_rle(mask_rle: Dict[str, Any]) -> np.ndarray:
    decoded = mask_utils.decode(_normalize_rle(mask_rle))
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(bool)


def _mask_bbox_xyxy(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    if mask.ndim != 2:
        return None
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _normalized_xyhw_from_bbox(
    bbox_xyxy: Tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    x1, y1, x2, y2 = bbox_xyxy
    width_f = max(1.0, float(width))
    height_f = max(1.0, float(height))
    w = max(0.0, min(width_f, x2 - x1))
    h = max(0.0, min(height_f, y2 - y1))
    cx = max(0.0, min(width_f, x1 + w / 2.0))
    cy = max(0.0, min(height_f, y1 + h / 2.0))
    return (
        {"x": float(cx / width_f), "y": float(cy / height_f)},
        {"w": float(w / width_f), "h": float(h / height_f)},
    )


def _prediction_has_scalar_geometry(pred: Dict[str, Any]) -> bool:
    xy = pred.get("xy") if isinstance(pred.get("xy"), dict) else {}
    hw = pred.get("hw") if isinstance(pred.get("hw"), dict) else {}
    try:
        float(xy.get("x"))
        float(xy.get("y"))
        float(hw.get("w"))
        float(hw.get("h"))
        return True
    except Exception:
        return False


def _normalize_prediction(
    pred: Dict[str, Any],
    *,
    width: int,
    height: int,
) -> Optional[Dict[str, Any]]:
    if not isinstance(pred, dict):
        return None
    mask_rle = pred.get("mask_rle") if isinstance(pred.get("mask_rle"), dict) else None
    resized_mask_rle = None
    if mask_rle:
        try:
            resized_mask_rle = resize_mask_rle(mask_rle, target_height=int(height), target_width=int(width))
        except Exception:
            resized_mask_rle = mask_rle
    normalized: Dict[str, Any] = {"mask_rle": resized_mask_rle or {"counts": "", "size": [height, width]}}
    if mask_rle:
        try:
            mask = decode_mask_rle(resized_mask_rle or mask_rle)
        except Exception:
            mask = None
        if mask is not None:
            bbox = _mask_bbox_xyxy(mask)
            if bbox is not None:
                xy, hw = _normalized_xyhw_from_bbox(bbox, width=width, height=height)
                normalized["xy"] = xy
                normalized["hw"] = hw
                return normalized
    if _prediction_has_scalar_geometry(pred):
        normalized["xy"] = {
            "x": float(pred["xy"]["x"]),
            "y": float(pred["xy"]["y"]),
        }
        normalized["hw"] = {
            "w": float(pred["hw"]["w"]),
            "h": float(pred["hw"]["h"]),
        }
        return normalized
    return None


def _normalize_prediction_batch(
    batch: Any,
    *,
    width: int,
    height: int,
) -> List[List[Dict[str, Any]]]:
    if not isinstance(batch, list):
        return []
    normalized: List[List[Dict[str, Any]]] = []
    for item in batch:
        rows: List[Dict[str, Any]] = []
        if isinstance(item, list):
            for row in item:
                normalized_row = _normalize_prediction(row, width=width, height=height)
                if normalized_row is not None:
                    rows.append(normalized_row)
        normalized.append(rows)
    return normalized


def _falcon_retry_dimensions(max_dimension: int, min_dimension: int) -> List[int]:
    requested = max(int(max_dimension), int(min_dimension))
    lower_candidates = [768, 640, 512, 448, 384, 320, int(min_dimension)]
    dims: List[int] = []
    for value in [requested, *lower_candidates]:
        value = max(int(min_dimension), min(requested, int(value)))
        if value not in dims:
            dims.append(value)
    return dims


def _is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc or "")
    return isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in text


def _run_falcon_queries_raw_generate(
    *,
    pil_image: Any,
    queries: Sequence[str],
    model_id: str,
    device: str,
    local_files_only: bool = True,
    compile_model: bool = False,
    min_dimension: int = 256,
    max_dimension: int = 1024,
    max_new_tokens: int = 1024,
    segm_threshold: float = 0.3,
) -> List[List[Dict[str, Any]]]:
    if not queries:
        return []
    _configure_falcon_torch_runtime(torch)
    width, height = pil_image.size
    dims = _falcon_retry_dimensions(int(max_dimension), int(min_dimension))
    last_exc: Optional[BaseException] = None
    for dim in dims:
        try:
            model = ensure_falcon_runtime(model_id=model_id, device=device, local_files_only=local_files_only)
            images = [pil_image] * len(queries)
            batch = model.generate(
                images,
                list(queries),
                max_new_tokens=int(max_new_tokens),
                min_dimension=int(min_dimension),
                max_dimension=int(dim),
                compile=bool(compile_model),
                segm_threshold=float(segm_threshold),
            )
            return _normalize_prediction_batch(batch, width=width, height=height)
        except Exception as exc:
            last_exc = exc
            if _is_cuda_oom_error(exc) and dim != dims[-1]:
                unload_falcon_runtime()
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return []


def _falcon_server_url(override: Optional[str] = None) -> str:
    candidate = str(override or os.environ.get(_FALCON_SERVER_URL_ENV) or "").strip()
    if not candidate:
        raise RuntimeError("falcon_server_url_required")
    return candidate.rstrip("/")


def _image_to_base64_png(pil_image: Any) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _run_falcon_queries_server(
    *,
    pil_image: Any,
    queries: Sequence[str],
    task: str,
    server_url: str,
    min_dimension: int,
    max_dimension: int,
    max_new_tokens: int,
) -> List[List[Dict[str, Any]]]:
    if not queries:
        return []
    endpoint = f"{_falcon_server_url(server_url)}/v1/predictions"
    image_b64 = _image_to_base64_png(pil_image)
    out: List[List[Dict[str, Any]]] = []
    for query in queries:
        if not str(query or "").strip():
            out.append([])
            continue
        response = requests.post(
            endpoint,
            json={
                "image": {"base64": image_b64},
                "query": str(query).strip(),
                "task": str(task or "segmentation").strip(),
                "max_tokens": int(max_new_tokens),
                "min_image_size": int(min_dimension),
                "max_image_size": int(max_dimension),
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        rows: List[Dict[str, Any]] = []
        for item in payload.get("masks") or []:
            bbox = item.get("bbox") or []
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            width = max(1.0, float(pil_image.size[0]))
            height = max(1.0, float(pil_image.size[1]))
            rows.append(
                {
                    "xy": {
                        "x": float(((x1 + x2) / 2.0) / width),
                        "y": float(((y1 + y2) / 2.0) / height),
                    },
                    "hw": {
                        "w": float(max(0.0, x2 - x1) / width),
                        "h": float(max(0.0, y2 - y1) / height),
                    },
                    "mask_rle": item.get("rle") if isinstance(item.get("rle"), dict) else None,
                }
            )
        out.append(
            _normalize_prediction_batch([rows], width=pil_image.size[0], height=pil_image.size[1])[0]
            if rows
            else []
        )
    return out


def _run_falcon_queries_official_detection(
    *,
    pil_image: Any,
    queries: Sequence[str],
    model_id: str,
    device: str,
    local_files_only: bool = True,
    min_dimension: int = 256,
    max_dimension: int = 1024,
    max_new_tokens: int = 1024,
) -> List[List[Dict[str, Any]]]:
    if not queries:
        return []
    dims = _falcon_retry_dimensions(int(max_dimension), int(min_dimension))
    last_exc: Optional[BaseException] = None
    for dim in dims:
        try:
            runtime = ensure_official_falcon_detection_runtime(
                model_id=model_id,
                device=device,
                local_files_only=local_files_only,
            )
            falcon_pkg = runtime["falcon_pkg"]
            paged_module = runtime["paged_module"]
            visualization_module = runtime["visualization_module"]
            engine = runtime["engine"]
            tokenizer = runtime["tokenizer"]
            stop_token_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]
            sampling_params = paged_module.SamplingParams(
                max_new_tokens=int(max_new_tokens),
                stop_token_ids=stop_token_ids,
                segmentation_threshold=0.3,
            )
            sequences = [
                paged_module.Sequence(
                    text=falcon_pkg.build_prompt_for_task(str(query or "").strip(), "detection"),
                    image=pil_image,
                    min_image_size=int(min_dimension),
                    max_image_size=int(dim),
                    request_idx=idx + 1,
                    task="detection",
                )
                for idx, query in enumerate(queries)
                if str(query or "").strip()
            ]
            if not sequences:
                return []
            engine.generate(sequences, sampling_params=sampling_params, use_tqdm=False, print_stats=False)
            output: List[List[Dict[str, Any]]] = []
            for seq in sequences:
                rows: List[Dict[str, Any]] = []
                for bbox in visualization_module.pair_bbox_entries(seq.output_aux.bboxes_raw):
                    try:
                        rows.append(
                            {
                                "xy": {"x": float(bbox["x"]), "y": float(bbox["y"])},
                                "hw": {"w": float(bbox["w"]), "h": float(bbox["h"])},
                            }
                        )
                    except Exception:
                        continue
                output.append(rows)
            return output
        except Exception as exc:
            last_exc = exc
            if _is_cuda_oom_error(exc) and dim != dims[-1]:
                unload_official_falcon_runtime()
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return []


def _run_falcon_queries_official_segmentation(
    *,
    pil_image: Any,
    queries: Sequence[str],
    model_id: str,
    device: str,
    local_files_only: bool = True,
    min_dimension: int = 256,
    max_dimension: int = 1024,
    max_new_tokens: int = 1024,
    coord_dedup_threshold: float = 0.01,
    hr_upsample_ratio: int = 8,
    segmentation_threshold: float = 0.3,
) -> List[List[Dict[str, Any]]]:
    if not queries:
        return []
    dims = _falcon_retry_dimensions(int(max_dimension), int(min_dimension))
    last_exc: Optional[BaseException] = None
    width, height = pil_image.size
    for dim in dims:
        try:
            runtime = ensure_official_falcon_segmentation_runtime(
                model_id=model_id,
                device=device,
                local_files_only=local_files_only,
                max_image_size=int(dim),
            )
            falcon_pkg = runtime["falcon_pkg"]
            paged_module = runtime["paged_module"]
            visualization_module = runtime["visualization_module"]
            engine = runtime["engine"]
            tokenizer = runtime["tokenizer"]
            _disable_unsafe_segmentation_hr_cache(engine)
            stop_token_ids = [tokenizer.eos_token_id]
            if hasattr(tokenizer, "end_of_query_token_id"):
                stop_token_ids.append(tokenizer.end_of_query_token_id)
            sampling_params = paged_module.SamplingParams(
                max_new_tokens=int(max_new_tokens),
                stop_token_ids=stop_token_ids,
                coord_dedup_threshold=float(coord_dedup_threshold),
                hr_upsample_ratio=int(hr_upsample_ratio),
                segmentation_threshold=float(segmentation_threshold),
            )
            sequences = [
                paged_module.Sequence(
                    text=falcon_pkg.build_prompt_for_task(str(query or "").strip(), "segmentation"),
                    image=pil_image,
                    min_image_size=int(min_dimension),
                    max_image_size=int(dim),
                    request_idx=idx + 1,
                    task="segmentation",
                )
                for idx, query in enumerate(queries)
                if str(query or "").strip()
            ]
            if not sequences:
                return []
            engine.generate(sequences, sampling_params=sampling_params, use_tqdm=False, print_stats=False)
            output: List[List[Dict[str, Any]]] = []
            for seq in sequences:
                rows: List[Dict[str, Any]] = []
                paired_bboxes = visualization_module.pair_bbox_entries(seq.output_aux.bboxes_raw)
                raw_masks = list((seq.output_aux.masks_rle if seq.output_aux else []) or [])
                for idx, raw_mask in enumerate(raw_masks):
                    row: Dict[str, Any] = {"mask_rle": raw_mask}
                    if idx < len(paired_bboxes):
                        bbox = paired_bboxes[idx]
                        row["xy"] = {"x": float(bbox.get("x", 0.5)), "y": float(bbox.get("y", 0.5))}
                        row["hw"] = {"w": float(bbox.get("w", 1.0)), "h": float(bbox.get("h", 1.0))}
                    rows.append(row)
                if not rows and paired_bboxes:
                    for bbox in paired_bboxes:
                        rows.append(
                            {
                                "xy": {"x": float(bbox.get("x", 0.5)), "y": float(bbox.get("y", 0.5))},
                                "hw": {"w": float(bbox.get("w", 1.0)), "h": float(bbox.get("h", 1.0))},
                            }
                        )
                output.append(
                    [
                        normalized
                        for normalized in (
                            _normalize_prediction(row, width=width, height=height)
                            for row in rows
                        )
                        if normalized is not None
                    ]
                )
            return output
        except Exception as exc:
            last_exc = exc
            if _is_cuda_oom_error(exc) and dim != dims[-1]:
                unload_official_falcon_runtime()
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return []


def run_falcon_queries(
    *,
    pil_image: Any,
    queries: Sequence[str],
    model_id: str,
    device: str,
    local_files_only: bool = True,
    compile_model: bool = False,
    min_dimension: int = 256,
    max_dimension: int = 1024,
    max_new_tokens: int = 1024,
    task: str = "segmentation",
    backend: str = "embedded",
    server_url: Optional[str] = None,
    coord_dedup_threshold: float = 0.01,
    hr_upsample_ratio: int = 8,
    segmentation_threshold: float = 0.3,
) -> List[List[Dict[str, Any]]]:
    task_name = str(task or "segmentation").strip().lower()
    backend_name = str(backend or "embedded").strip().lower()
    if backend_name == "server":
        return _run_falcon_queries_server(
            pil_image=pil_image,
            queries=queries,
            task=task_name,
            server_url=str(server_url or ""),
            min_dimension=int(min_dimension),
            max_dimension=int(max_dimension),
            max_new_tokens=int(max_new_tokens),
        )
    if task_name == "detection":
        return _run_falcon_queries_official_detection(
            pil_image=pil_image,
            queries=queries,
            model_id=model_id,
            device=device,
            local_files_only=local_files_only,
            min_dimension=min_dimension,
            max_dimension=max_dimension,
            max_new_tokens=max_new_tokens,
        )
    if task_name == "segmentation":
        return _run_falcon_queries_official_segmentation(
            pil_image=pil_image,
            queries=queries,
            model_id=model_id,
            device=device,
            local_files_only=local_files_only,
            min_dimension=min_dimension,
            max_dimension=max_dimension,
            max_new_tokens=max_new_tokens,
            coord_dedup_threshold=coord_dedup_threshold,
            hr_upsample_ratio=hr_upsample_ratio,
            segmentation_threshold=segmentation_threshold,
        )
    return _run_falcon_queries_raw_generate(
        pil_image=pil_image,
        queries=queries,
        model_id=model_id,
        device=device,
        local_files_only=local_files_only,
        compile_model=compile_model,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
        max_new_tokens=max_new_tokens,
        segm_threshold=float(segmentation_threshold),
    )


def prediction_bbox_xyxy(pred: Dict[str, Any], *, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    xy = pred.get("xy") if isinstance(pred.get("xy"), dict) else {}
    hw = pred.get("hw") if isinstance(pred.get("hw"), dict) else {}
    try:
        cx = float(xy.get("x"))
        cy = float(xy.get("y"))
        w = float(hw.get("w"))
        h = float(hw.get("h"))
    except Exception:
        return None
    x1 = max(0.0, min(float(width), (cx - w / 2.0) * float(width)))
    y1 = max(0.0, min(float(height), (cy - h / 2.0) * float(height)))
    x2 = max(0.0, min(float(width), (cx + w / 2.0) * float(width)))
    y2 = max(0.0, min(float(height), (cy + h / 2.0) * float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)
