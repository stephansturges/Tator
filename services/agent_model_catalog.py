"""Inference-only VLM/agent model catalog.

These entries extend the existing Qwen runtime surface for agent-assisted
workflows. They are deliberately not training models: Qwen training remains
handled by services.qwen_model_catalog and tools.qwen_training.
"""

from __future__ import annotations

import importlib.metadata

from typing import Any, Dict, Iterable, List, Optional

from packaging import version as packaging_version

from services.qwen_mlx import (
    QWEN_AEON_QWEN36_27B_MLX_MODEL,
    QWEN_MLX_MODEL_OPTIONS,
    QWEN_PLATFORM_MLX,
    QWEN_PLATFORM_TRANSFORMERS,
)
from services.qwen_model_catalog import QWEN_CUDA_DEFAULT_MODEL

AGENT_MLX_NOTE = (
    "Inference-only MLX-VLM agent model candidate. Use for captioning, prepass, "
    "and Class Split review after local smoke validation; training is not wired."
)
AGENT_TRANSFORMERS_NOTE = (
    "Inference-only Transformers agent model candidate. Keep hidden until a model "
    "has a completed Class Split/VLM smoke on the target runtime."
)
AGENT_TRANSFORMERS5_NOTE = (
    "Inference-only Transformers 5.x VLM agent model candidate. Requires the "
    "unified Transformers 5.x backend profile used for Qwen3.5/3.6 models."
)
QWEN_MLX_RUNTIME_SUPPORTED_NOTE = (
    "Inference-only MLX Qwen3-VL model from the stable runtime catalog. Enabled for "
    "agent-assisted review because the underlying MLX-VLM path supports image "
    "inference; not part of the Qwen3.6 vignette matrix winners."
)

_AGENT_ENABLED_QWEN3_VL_MLX_IDS = (
    "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    "mlx-community/Qwen3-VL-2B-Instruct-4bit",
    "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    "mlx-community/Qwen3-VL-4B-Thinking-4bit",
    "mlx-community/Qwen3-VL-8B-Thinking-4bit",
    "EZCon/Huihui-Qwen3-VL-4B-Instruct-abliterated-4bit-mlx",
    "EZCon/Huihui-Qwen3-VL-2B-Instruct-abliterated-4bit-mlx",
    "alexgusevski/Huihui-Qwen3-VL-8B-Instruct-abliterated-q4-mlx",
    "nightmedia/Huihui-Qwen3-VL-32B-Thinking-abliterated-qx65-hi-mlx",
)


def _entry(
    model_id: str,
    label: str,
    *,
    family: str,
    runtime_platform: str,
    source: str,
    size: str,
    variant: str = "Instruct",
    quantization: Optional[str] = None,
    abliterated: bool = False,
    vision_inference_supported: bool = True,
    compatibility_note: Optional[str] = None,
    backend_status: str = "candidate",
    smoke_status: str = "metadata",
    min_transformers: Optional[str] = None,
) -> Dict[str, Any]:
    note = compatibility_note or (
        AGENT_MLX_NOTE if runtime_platform == QWEN_PLATFORM_MLX else AGENT_TRANSFORMERS_NOTE
    )
    return {
        "id": model_id,
        "label": label,
        "model_id": model_id,
        "runtime_platform": runtime_platform,
        "model_family": family,
        "agent_model": True,
        "agent_supported": vision_inference_supported,
        "inference_supported": vision_inference_supported,
        "vision_inference_supported": vision_inference_supported,
        "training_supported": False,
        "training_modes": [],
        "training_note": note,
        "compatibility_note": note,
        "source": source,
        "size": size,
        "variant": variant,
        "quantization": quantization,
        "quantized": bool(quantization),
        "abliterated": abliterated,
        "backend_status": backend_status,
        "smoke_status": smoke_status,
        "min_transformers": min_transformers,
        "dataset_context": (
            "Inference-only visual-language agent model for captioning, prepass, "
            "and Class Split review. It does not participate in Qwen adapter training."
        ),
    }


def _entry_from_mlx_runtime(model_id: str) -> Dict[str, Any]:
    for entry in QWEN_MLX_MODEL_OPTIONS:
        if str(entry.get("id") or entry.get("model_id") or "") != model_id:
            continue
        return _entry(
            model_id,
            str(entry.get("label") or model_id),
            family="qwen3_vl",
            runtime_platform=QWEN_PLATFORM_MLX,
            source=str(entry.get("source") or "mlx-community"),
            size=str(entry.get("size") or ""),
            variant=str(entry.get("variant") or "Instruct"),
            quantization=entry.get("quantization"),
            abliterated=bool(entry.get("abliterated")),
            vision_inference_supported=entry.get("vision_inference_supported", True) is not False,
            compatibility_note=QWEN_MLX_RUNTIME_SUPPORTED_NOTE,
            backend_status="validated_runtime",
            smoke_status="qwen_mlx_runtime_supported",
        )
    raise ValueError(f"Agent Qwen3-VL MLX model is not in the runtime catalog: {model_id}")


def _build_agent_model_options() -> List[Dict[str, Any]]:
    entries = [
        *(_entry_from_mlx_runtime(model_id) for model_id in _AGENT_ENABLED_QWEN3_VL_MLX_IDS),
        _entry(
            QWEN_AEON_QWEN36_27B_MLX_MODEL,
            "MLX AEON Qwen3.6 27B Ultimate Uncensored FP4",
            family="qwen3_6",
            runtime_platform=QWEN_PLATFORM_MLX,
            source="AEON-7",
            size="27B",
            variant="AEON Ultimate Uncensored",
            quantization="FP4",
            abliterated=True,
            vision_inference_supported=True,
            compatibility_note=(
                "Inference-only AEON Qwen3.6 27B multimodal MLX FP4 model. "
                "The checkpoint includes vision_tower weights and processor files; "
                "the backend normalizes its qwen3_5_vision config alias for mlx-vlm."
            ),
            backend_status="validated_runtime",
            smoke_status="mlx_vlm_qwen35_vision_alias_supported",
        ),
        _entry(
            QWEN_CUDA_DEFAULT_MODEL,
            "CUDA AEON Qwen3.6 27B Ultimate Uncensored NVFP4 MTP",
            family="qwen3_6",
            runtime_platform=QWEN_PLATFORM_TRANSFORMERS,
            source="AEON-7",
            size="27B",
            variant="AEON Ultimate Uncensored",
            quantization="NVFP4",
            abliterated=True,
            compatibility_note=(
                "Inference-only AEON Qwen3.6 27B multimodal NVFP4/MTP model. "
                "The upstream metadata advertises Transformers/vLLM image-text support; "
                "training is not wired."
            ),
            backend_status="metadata_verified",
            smoke_status="metadata_verified",
            min_transformers="5.7.0",
        ),
        _entry(
            "empero-ai/Qwable-9B-Claude-Fable-5",
            "Transformers5 Qwable 9B Claude Fable 5",
            family="qwen3_5",
            runtime_platform=QWEN_PLATFORM_TRANSFORMERS,
            source="empero-ai",
            size="9B",
            variant="Claude Fable 5",
            compatibility_note=(
                f"{AGENT_TRANSFORMERS5_NOTE} Metadata smoke passed under "
                "Transformers 5.7: config resolves as qwen3_5 and AutoProcessor "
                "returns Qwen3VLProcessor with an image processor."
            ),
            backend_status="transformers5_runtime_supported",
            smoke_status="transformers5_processor_passed",
            min_transformers="5.7.0",
        ),
        _entry(
            "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
            "Qwythos 9B Claude Mythos 5 1M (blocked visual)",
            family="qwen3_5",
            runtime_platform=QWEN_PLATFORM_TRANSFORMERS,
            source="empero-ai",
            size="9B",
            variant="Claude Mythos 5 1M",
            vision_inference_supported=False,
            compatibility_note=(
                "Blocked for image-assisted vignette review: the repo config resolves "
                "as qwen3_5 and advertises a vision config under Transformers 5.7, "
                "but the model repo is missing preprocessor_config.json / image "
                "processor files, so AutoProcessor cannot load an image processor."
            ),
            backend_status="blocked",
            smoke_status="missing_image_processor",
            min_transformers="5.7.0",
        ),
        _entry(
            "mlx-community/Qwen3.6-35B-A3B-4bit",
            "MLX Qwen3.6 35B-A3B 4bit",
            family="qwen3_6",
            runtime_platform=QWEN_PLATFORM_MLX,
            source="mlx-community",
            size="35B-A3B",
            quantization="4bit",
            compatibility_note=(
                "Inference-only MLX-VLM agent model. Local Class Split vignette smoke tests passed; "
                "training is not wired."
            ),
            smoke_status="class_split_benchmark_passed",
        ),
        _entry(
            "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit",
            "MLX Huihui Qwen3.6 35B-A3B abliterated 4bit",
            family="qwen3_6",
            runtime_platform=QWEN_PLATFORM_MLX,
            source="vanch007",
            size="35B-A3B",
            variant="Abliterated",
            quantization="4bit",
            abliterated=True,
            compatibility_note=(
                "Inference-only MLX-VLM agent model. Local Class Split vignette smoke tests passed; "
                "training is not wired."
            ),
            smoke_status="class_split_benchmark_passed",
        ),
    ]
    entries.sort(
        key=lambda entry: (
            1 if entry.get("backend_status") == "blocked" else 0,
            str(entry.get("runtime_platform") or ""),
            str(entry.get("model_family") or ""),
            str(entry.get("label") or ""),
        )
    )
    return entries


AGENT_MODEL_OPTIONS = _build_agent_model_options()
AGENT_MODEL_IDS = {str(entry["id"]) for entry in AGENT_MODEL_OPTIONS}
AGENT_MLX_MODEL_OPTIONS = [
    entry for entry in AGENT_MODEL_OPTIONS if entry.get("runtime_platform") == QWEN_PLATFORM_MLX
]
AGENT_MLX_MODEL_IDS = {str(entry["id"]) for entry in AGENT_MLX_MODEL_OPTIONS}
AGENT_TRANSFORMERS_MODEL_OPTIONS = [
    entry
    for entry in AGENT_MODEL_OPTIONS
    if entry.get("runtime_platform") == QWEN_PLATFORM_TRANSFORMERS
]
AGENT_TRANSFORMERS_MODEL_IDS = {str(entry["id"]) for entry in AGENT_TRANSFORMERS_MODEL_OPTIONS}


def agent_model_options(*, runtime_platform: Optional[str] = None) -> List[Dict[str, Any]]:
    entries: Iterable[Dict[str, Any]] = AGENT_MODEL_OPTIONS
    if runtime_platform:
        entries = (
            entry
            for entry in entries
            if str(entry.get("runtime_platform") or "") == str(runtime_platform)
        )
    return [dict(entry) for entry in entries]


def agent_model_metadata_for_model(model_id: str) -> Dict[str, Any]:
    raw = str(model_id or "").strip()
    for entry in AGENT_MODEL_OPTIONS:
        if str(entry.get("id")) == raw:
            return dict(entry)
    return {}


def is_agent_model_id(model_id: Optional[str]) -> bool:
    return str(model_id or "").strip() in AGENT_MODEL_IDS


def is_agent_mlx_model_id(model_id: Optional[str]) -> bool:
    return str(model_id or "").strip() in AGENT_MLX_MODEL_IDS


def is_agent_transformers_model_id(model_id: Optional[str]) -> bool:
    return str(model_id or "").strip() in AGENT_TRANSFORMERS_MODEL_IDS


def agent_model_block_detail(model_id: str) -> Optional[str]:
    metadata = agent_model_metadata_for_model(model_id)
    if not metadata:
        return None
    min_transformers = str(metadata.get("min_transformers") or "").strip()
    if min_transformers and metadata.get("runtime_platform") == QWEN_PLATFORM_TRANSFORMERS:
        try:
            installed = importlib.metadata.version("transformers")
        except Exception:
            installed = ""
        if not installed or packaging_version.parse(installed) < packaging_version.parse(min_transformers):
            return (
                f"{model_id}: requires transformers>={min_transformers}; "
                f"installed={installed or 'missing'}."
            )
    if (
        metadata.get("vision_inference_supported") is False
        or metadata.get("inference_supported") is False
        or metadata.get("backend_status") == "blocked"
    ):
        return str(
            metadata.get("compatibility_note")
            or f"{model_id}: this agent model is not enabled for vision inference."
        )
    return None
