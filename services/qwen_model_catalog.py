"""Shared Qwen model catalog and runtime policy helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

QWEN_PLATFORM_TRANSFORMERS = "transformers"

_QWEN_VARIANTS = ("Instruct", "Thinking")
_QWEN_DENSE_SIZES = ("2B", "4B", "8B", "32B")
_QWEN_MOE_SIZES = ("30B-A3B", "235B-A22B")


def _official_model_id(size: str, variant: str) -> str:
    return f"Qwen/Qwen3-VL-{size}-{variant}"


def _model_sort_key(entry: Dict[str, Any]) -> tuple:
    preferred_quant = {"none": 0, "awq": 1, "gptq": 2}
    size_order = {
        "2B": 0,
        "4B": 1,
        "8B": 2,
        "32B": 3,
        "30B-A3B": 4,
        "235B-A22B": 5,
    }
    return (
        1 if entry.get("abliterated") else 0,
        preferred_quant.get(str(entry.get("quantization_backend") or "none"), 20),
        size_order.get(str(entry.get("size") or ""), 20),
        str(entry.get("variant") or ""),
        str(entry.get("id") or ""),
    )


def _entry(
    model_id: str,
    label: str,
    *,
    size: str,
    variant: str,
    source: str,
    quantization: Optional[str] = None,
    quantization_backend: str = "none",
    abliterated: bool = False,
    training_model_id: Optional[str] = None,
    training_note: Optional[str] = None,
) -> Dict[str, Any]:
    train_base = str(training_model_id or model_id)
    quantized = quantization_backend != "none"
    return {
        "id": model_id,
        "label": label,
        "model_id": model_id,
        "runtime_platform": QWEN_PLATFORM_TRANSFORMERS,
        "size": size,
        "variant": variant,
        "source": source,
        "quantization": quantization,
        "quantization_backend": quantization_backend,
        "quantized": quantized,
        "abliterated": abliterated,
        "training_supported": True,
        "training_modes": ["official_lora", "trl_qlora"],
        "training_model_id": train_base,
        "training_note": training_note
        or (
            "Training starts from the matching full Transformers checkpoint; QLoRA applies bitsandbytes 4-bit at load time."
            if quantized
            else None
        ),
        "dataset_context": (
            "Abliterated Qwen3-VL Transformer checkpoint for CUDA/CPU inference and adapter training."
            if abliterated
            else "Qwen3-VL Transformer checkpoint for CUDA/CPU inference and adapter training."
        ),
    }


def _build_transformers_model_options() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for size in (*_QWEN_DENSE_SIZES, *_QWEN_MOE_SIZES):
        for variant in _QWEN_VARIANTS:
            model_id = _official_model_id(size, variant)
            entries.append(
                _entry(
                    model_id,
                    f"CUDA Qwen3-VL {size} {variant}",
                    size=size,
                    variant=variant,
                    source="Qwen",
                )
            )

    awq_specs = [
        ("2B", "Instruct"),
        ("2B", "Thinking"),
        ("4B", "Instruct"),
        ("4B", "Thinking"),
        ("8B", "Instruct"),
        ("8B", "Thinking"),
        ("32B", "Instruct"),
        ("32B", "Thinking"),
        ("30B-A3B", "Instruct"),
        ("30B-A3B", "Thinking"),
    ]
    for size, variant in awq_specs:
        model_id = f"cyankiwi/Qwen3-VL-{size}-{variant}-AWQ-4bit"
        entries.append(
            _entry(
                model_id,
                f"CUDA Qwen3-VL {size} {variant} AWQ 4-bit",
                size=size,
                variant=variant,
                source="cyankiwi",
                quantization="AWQ 4-bit",
                quantization_backend="awq",
                training_model_id=_official_model_id(size, variant),
            )
        )

    gptq_specs = [
        ("pramjana/Qwen3-VL-4B-Instruct-4bit-GPTQ", "4B", "Instruct"),
        ("pramjana/Qwen3-VL-4B-Thinking-4bit-GPTQ", "4B", "Thinking"),
        ("aonaon/Qwen3-VL-8B-Instruct-4bit-GPTQ", "8B", "Instruct"),
    ]
    for model_id, size, variant in gptq_specs:
        entries.append(
            _entry(
                model_id,
                f"CUDA Qwen3-VL {size} {variant} GPTQ 4-bit",
                size=size,
                variant=variant,
                source=model_id.split("/", 1)[0],
                quantization="GPTQ 4-bit",
                quantization_backend="gptq",
                training_model_id=_official_model_id(size, variant),
            )
        )

    abliterated_specs = [
        ("huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated", "2B", "Instruct"),
        ("huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated", "4B", "Instruct"),
        ("huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated", "8B", "Instruct"),
        ("huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated", "30B-A3B", "Instruct"),
    ]
    for model_id, size, variant in abliterated_specs:
        entries.append(
            _entry(
                model_id,
                f"CUDA Huihui Qwen3-VL {size} {variant} abliterated",
                size=size,
                variant=variant,
                source="huihui-ai",
                abliterated=True,
            )
        )

    entries.append(
        _entry(
            "JinRiYao2001/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-AWQ",
            "CUDA Huihui Qwen3-VL 30B-A3B Instruct abliterated AWQ",
            size="30B-A3B",
            variant="Instruct",
            source="JinRiYao2001",
            quantization="AWQ 4-bit",
            quantization_backend="awq",
            abliterated=True,
            training_model_id="huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated",
        )
    )

    entries.sort(key=_model_sort_key)
    return entries


QWEN_TRANSFORMERS_MODEL_OPTIONS = _build_transformers_model_options()
QWEN_TRANSFORMERS_MODEL_IDS = {str(entry["id"]) for entry in QWEN_TRANSFORMERS_MODEL_OPTIONS}
_TRAINING_MODEL_BY_ID = {
    str(entry["id"]): str(entry.get("training_model_id") or entry["id"])
    for entry in QWEN_TRANSFORMERS_MODEL_OPTIONS
}


def qwen_transformers_metadata_for_model(model_id: str) -> Dict[str, Any]:
    for entry in QWEN_TRANSFORMERS_MODEL_OPTIONS:
        if str(entry.get("id")) == str(model_id):
            return dict(entry)
    return {
        "id": str(model_id),
        "label": str(model_id),
        "model_id": str(model_id),
        "runtime_platform": QWEN_PLATFORM_TRANSFORMERS,
        "training_supported": True,
        "training_modes": ["official_lora", "trl_qlora"],
        "training_model_id": resolve_qwen_training_model_id(model_id),
    }


def infer_qwen_quantization_backend(model_id: Optional[str]) -> str:
    lowered = str(model_id or "").strip().lower()
    if "awq" in lowered:
        return "awq"
    if "gptq" in lowered:
        return "gptq"
    if "fp8" in lowered:
        return "fp8"
    return "none"


def is_qwen_mlx_model_id(model_id: Optional[str]) -> bool:
    raw = str(model_id or "").strip()
    lowered = raw.lower()
    return raw.startswith("mlx-community/") or lowered.endswith("-mlx") or "-mlx-" in lowered


def _strip_known_training_quant_suffix(model_id: str) -> Optional[str]:
    for suffix in ("-FP8", "-GPTQ-Int4", "-GPTQ-Int8", "-AWQ", "-INT4", "-INT8"):
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return None


def _infer_official_base_from_quantized_name(model_id: str) -> Optional[str]:
    basename = model_id.rsplit("/", 1)[-1]
    match = re.search(
        r"(Qwen3-VL-(?:2B|4B|8B|32B|30B-A3B|235B-A22B)-(?:Instruct|Thinking))",
        basename,
    )
    if not match:
        return None
    return f"Qwen/{match.group(1)}"


def resolve_qwen_training_model_id(model_id: Optional[str]) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return "Qwen/Qwen3-VL-4B-Instruct"
    if raw in _TRAINING_MODEL_BY_ID:
        return _TRAINING_MODEL_BY_ID[raw]
    stripped = _strip_known_training_quant_suffix(raw)
    if stripped and stripped.startswith("Qwen/Qwen3-VL-"):
        return stripped
    if infer_qwen_quantization_backend(raw) in {"awq", "gptq", "fp8"}:
        inferred = _infer_official_base_from_quantized_name(raw)
        if inferred:
            return inferred
    return raw


def qwen_transformers_load_kwargs(
    model_id: Optional[str],
    *,
    device: str,
    device_pref: str,
    torch_module: Any,
) -> Dict[str, Any]:
    device_text = str(device or "")
    pref = str(device_pref or "auto")
    quant_backend = infer_qwen_quantization_backend(model_id)
    cuda_available = bool(torch_module.cuda.is_available())
    on_cuda = device_text.startswith("cuda") and cuda_available
    if on_cuda and quant_backend in {"awq", "gptq"}:
        if pref == "auto" or device_text == "cuda":
            return {"torch_dtype": "auto", "device_map": "auto"}
        return {"torch_dtype": "auto", "device_map": {"": device_text}}
    if pref == "auto" and on_cuda:
        return {"torch_dtype": "auto", "device_map": "auto"}
    dtype = torch_module.float16 if device_text.startswith(("cuda", "mps")) else torch_module.float32
    return {"torch_dtype": dtype, "low_cpu_mem_usage": True}


__all__ = [
    "QWEN_TRANSFORMERS_MODEL_OPTIONS",
    "QWEN_TRANSFORMERS_MODEL_IDS",
    "infer_qwen_quantization_backend",
    "is_qwen_mlx_model_id",
    "qwen_transformers_metadata_for_model",
    "qwen_transformers_load_kwargs",
    "resolve_qwen_training_model_id",
]
