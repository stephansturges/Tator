"""Inference-only VLM/agent model catalog.

These entries extend the existing Qwen runtime surface for agent-assisted
workflows. They are deliberately not training models: Qwen training remains
handled by services.qwen_model_catalog and tools.qwen_training.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from services.qwen_mlx import QWEN_PLATFORM_MLX, QWEN_PLATFORM_TRANSFORMERS

AGENT_MLX_NOTE = (
    "Inference-only MLX-VLM agent model candidate. Use for captioning, prepass, "
    "and Class Split review after local smoke validation; training is not wired."
)
AGENT_TRANSFORMERS_NOTE = (
    "Inference-only Transformers agent model candidate. Keep hidden until a model "
    "has a completed Class Split/VLM smoke on the target runtime."
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
        "dataset_context": (
            "Inference-only visual-language agent model for captioning, prepass, "
            "and Class Split review. It does not participate in Qwen adapter training."
        ),
    }


def _build_agent_model_options() -> List[Dict[str, Any]]:
    entries = [
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
