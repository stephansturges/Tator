"""Inference-only VLM/agent model catalog.

These entries extend the existing Qwen runtime surface for agent-assisted
workflows. They are deliberately not training models: Qwen training remains
handled by services.qwen_model_catalog and tools.qwen_training.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from services.qwen_mlx import QWEN_PLATFORM_MLX, QWEN_PLATFORM_TRANSFORMERS

AGENT_TRANSFORMERS5_NOTE = (
    "Inference-only agent model. Requires a Transformers 5.x-compatible runtime "
    "for this architecture; keep Qwen training on the Qwen3-VL catalog."
)
AGENT_MLX_NOTE = (
    "Inference-only MLX-VLM agent model candidate. Use for captioning, prepass, "
    "and Class Split review after local smoke validation; training is not wired."
)
BLOCKED_HERETIC_NOTE = (
    "Candidate Heretic Qwen3.6 35B-A3B MLX checkpoint is tracked but not enabled: "
    "local mlx-vlm 0.6.1 smoke tests loaded with the compatibility shim, but "
    "generated invalid text in the class-split vignette benchmark."
)
GEMMA4_UNIFIED_NOTE = (
    "Tracked candidate only. The current isolated smoke found a Gemma4-unified "
    "variant that needs a newer runtime path before visual inference is enabled."
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
        AGENT_MLX_NOTE if runtime_platform == QWEN_PLATFORM_MLX else AGENT_TRANSFORMERS5_NOTE
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
            "Jackrong/Qwopus3.6-27B-v2",
            "CUDA Qwopus3.6 27B v2",
            family="qwopus3_6",
            runtime_platform=QWEN_PLATFORM_TRANSFORMERS,
            source="Jackrong",
            size="27B",
            variant="v2",
            smoke_status="config_processor",
        ),
        _entry(
            "prithivMLmods/Qwen3.6-35B-A3B-abliterated-MAX",
            "CUDA Qwen3.6 35B-A3B abliterated MAX",
            family="qwen3_6",
            runtime_platform=QWEN_PLATFORM_TRANSFORMERS,
            source="prithivMLmods",
            size="35B-A3B",
            variant="Abliterated MAX",
            abliterated=True,
            smoke_status="config_processor",
        ),
        _entry(
            "nex-agi/Nex-N2-mini",
            "CUDA Nex-N2 Mini",
            family="nex_n2",
            runtime_platform=QWEN_PLATFORM_TRANSFORMERS,
            source="nex-agi",
            size="35B-A3B",
            variant="Mini",
            smoke_status="config_processor",
        ),
        _entry(
            "huihui-ai/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated",
            "CUDA Huihui Gemma 4 31B QAT abliterated",
            family="gemma4",
            runtime_platform=QWEN_PLATFORM_TRANSFORMERS,
            source="huihui-ai",
            size="31B",
            variant="QAT",
            quantization="q4_0 source",
            abliterated=True,
            smoke_status="config_processor",
        ),
        _entry(
            "mlx-community/Qwen3.6-35B-A3B-4bit",
            "MLX Qwen3.6 35B-A3B 4bit",
            family="qwen3_6",
            runtime_platform=QWEN_PLATFORM_MLX,
            source="mlx-community",
            size="35B-A3B",
            quantization="4bit",
            smoke_status="metadata",
        ),
        _entry(
            "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
            "MLX Unsloth Qwen3.6 35B-A3B UD 4bit",
            family="qwen3_6",
            runtime_platform=QWEN_PLATFORM_MLX,
            source="unsloth",
            size="35B-A3B",
            quantization="4bit",
            smoke_status="metadata",
        ),
        _entry(
            "froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit",
            "MLX Qwen3.6 35B-A3B Uncensored Heretic 4bit",
            family="qwen3_6",
            runtime_platform=QWEN_PLATFORM_MLX,
            source="froggeric",
            size="35B-A3B",
            variant="Heretic",
            quantization="4bit",
            abliterated=True,
            smoke_status="metadata",
        ),
        _entry(
            "mlx-community/Nex-N2-mini-nvfp4",
            "MLX Nex-N2 Mini NVFP4",
            family="nex_n2",
            runtime_platform=QWEN_PLATFORM_MLX,
            source="mlx-community",
            size="35B-A3B",
            variant="Mini",
            quantization="NVFP4",
            smoke_status="metadata",
        ),
        _entry(
            "mlx-community/gemma-4-31B-it-qat-4bit",
            "MLX Gemma 4 31B QAT 4bit",
            family="gemma4",
            runtime_platform=QWEN_PLATFORM_MLX,
            source="mlx-community",
            size="31B",
            variant="QAT",
            quantization="4bit",
            smoke_status="metadata",
        ),
        _entry(
            "vanch007/Huihui-gemma-4-26B-A4B-it-abliterated-mlx-4bit",
            "MLX Huihui Gemma 4 26B-A4B abliterated 4bit",
            family="gemma4",
            runtime_platform=QWEN_PLATFORM_MLX,
            source="vanch007",
            size="26B-A4B",
            quantization="4bit",
            abliterated=True,
            smoke_status="metadata",
        ),
        _entry(
            "prithivMLmods/gemma-4-31B-it-Uncensored-MAX-MLX",
            "MLX Gemma 4 31B Uncensored MAX",
            family="gemma4",
            runtime_platform=QWEN_PLATFORM_MLX,
            source="prithivMLmods",
            size="31B",
            variant="Uncensored MAX",
            abliterated=True,
            smoke_status="metadata",
        ),
        _entry(
            "huihui-ai/Huihui-gemma-4-12B-it-abliterated",
            "Gemma 4 12B abliterated unified candidate",
            family="gemma4",
            runtime_platform=QWEN_PLATFORM_TRANSFORMERS,
            source="huihui-ai",
            size="12B",
            abliterated=True,
            vision_inference_supported=False,
            compatibility_note=GEMMA4_UNIFIED_NOTE,
            backend_status="blocked",
            smoke_status="failed_config",
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
