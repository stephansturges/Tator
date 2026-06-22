"""Shared Qwen model catalog and runtime policy helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

QWEN_PLATFORM_TRANSFORMERS = "transformers"

_QWEN_VARIANTS = ("Instruct", "Thinking")
_QWEN_DENSE_SIZES = ("2B", "4B", "8B", "32B")
_QWEN_MOE_SIZES = ("30B-A3B", "235B-A22B")
_QWEN3_5_6_INFERENCE_NOTE = (
    "Qwen3.5/3.6 visual checkpoints are exposed for CUDA/CPU inference and agent-assisted "
    "review only. Adapter training for this generation is not wired yet."
)
_QWEN_TRAINING_MODES = ["official_lora", "trl_qlora"]
_QWEN_MOE_TRAINING_NOTE = (
    "Qwen3-VL MoE adapter training uses the Transformers Qwen3VLMoe trainer path. "
    "Expect very large GPU memory requirements, especially without QLoRA or distributed training."
)


def _official_model_id(size: str, variant: str) -> str:
    return f"Qwen/Qwen3-VL-{size}-{variant}"


def _huihui_abliterated_model_id(size: str, variant: str) -> str:
    return f"huihui-ai/Huihui-Qwen3-VL-{size}-{variant}-abliterated"


def _supports_current_trainer(size: str) -> bool:
    return size in (*_QWEN_DENSE_SIZES, *_QWEN_MOE_SIZES)


def _model_sort_key(entry: Dict[str, Any]) -> tuple:
    preferred_quant = {"none": 0, "fp8": 1, "awq": 2, "gptq": 3}
    size_order = {
        "2B": 0,
        "4B": 1,
        "8B": 2,
        "9B": 3,
        "27B": 4,
        "32B": 5,
        "30B-A3B": 6,
        "35B-A3B": 7,
        "235B-A22B": 8,
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
    training_supported: bool = True,
    model_line: str = "Qwen3-VL",
) -> Dict[str, Any]:
    train_base = str(training_model_id or model_id)
    quantized = quantization_backend != "none"
    if not training_supported and not training_note:
        training_note = _QWEN_MOE_TRAINING_NOTE
    if training_supported and not training_note:
        if size in _QWEN_MOE_SIZES:
            training_note = _QWEN_MOE_TRAINING_NOTE
        elif quantized:
            training_note = (
                "Training starts from the matching unquantized abliterated Transformers checkpoint; "
                "QLoRA applies bitsandbytes 4-bit at load time."
                if abliterated
                else "Training starts from the matching full Transformers checkpoint; "
                "QLoRA applies bitsandbytes 4-bit at load time."
            )
    dataset_context = (
        f"Abliterated {model_line} Transformer checkpoint for CUDA/CPU inference and adapter training."
        if abliterated
        else f"{model_line} Transformer checkpoint for CUDA/CPU inference and adapter training."
    )
    if not training_supported:
        dataset_context = (
            f"Abliterated {model_line} Transformer checkpoint for CUDA/CPU inference. Adapter training is not wired for this model."
            if abliterated
            else f"{model_line} Transformer checkpoint for CUDA/CPU inference. Adapter training is not wired for this model."
        )
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
        "training_supported": training_supported,
        "training_modes": list(_QWEN_TRAINING_MODES) if training_supported else [],
        "training_model_id": train_base,
        "training_note": training_note,
        "dataset_context": dataset_context,
        "model_line": model_line,
    }


def _build_transformers_model_options() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for size in (*_QWEN_DENSE_SIZES, *_QWEN_MOE_SIZES):
        training_supported = _supports_current_trainer(size)
        for variant in _QWEN_VARIANTS:
            model_id = _official_model_id(size, variant)
            entries.append(
                _entry(
                    model_id,
                    f"CUDA Qwen3-VL {size} {variant}",
                    size=size,
                    variant=variant,
                    source="Qwen",
                    training_supported=training_supported,
                )
            )
            fp8_model_id = f"{model_id}-FP8"
            entries.append(
                _entry(
                    fp8_model_id,
                    f"CUDA Qwen3-VL {size} {variant} FP8",
                    size=size,
                    variant=variant,
                    source="Qwen",
                    quantization="FP8",
                    quantization_backend="fp8",
                    training_model_id=model_id,
                    training_note=(
                        "FP8 is for CUDA inference on compatible NVIDIA GPUs; adapter training starts "
                        "from the matching full Transformers checkpoint."
                        if training_supported
                        else _QWEN_MOE_TRAINING_NOTE
                    ),
                    training_supported=training_supported,
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
                training_supported=_supports_current_trainer(size),
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

    huihui_abliterated_specs = [
        (_huihui_abliterated_model_id(size, variant), size, variant)
        for size in ("2B", "4B", "8B", "32B", "30B-A3B")
        for variant in _QWEN_VARIANTS
    ]
    for model_id, size, variant in huihui_abliterated_specs:
        entries.append(
            _entry(
                model_id,
                f"CUDA Huihui Qwen3-VL {size} {variant} abliterated",
                size=size,
                variant=variant,
                source="huihui-ai",
                abliterated=True,
                training_supported=_supports_current_trainer(size),
            )
        )

    prithiv_abliterated_specs = [
        (f"prithivMLmods/Qwen3-VL-{size}-{variant}-abliterated-v1", size, variant, "v1")
        for size in ("2B", "4B", "8B", "32B", "30B-A3B")
        for variant in _QWEN_VARIANTS
    ]
    prithiv_abliterated_specs.append(
        ("prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2", "8B", "Instruct", "v2")
    )
    for model_id, size, variant, version in prithiv_abliterated_specs:
        entries.append(
            _entry(
                model_id,
                f"CUDA Prithiv Qwen3-VL {size} {variant} abliterated {version}",
                size=size,
                variant=variant,
                source="prithivMLmods",
                abliterated=True,
                training_supported=_supports_current_trainer(size),
            )
        )

    community_abliterated_specs = [
        ("Feiouex/Huihui-Qwen3-VL-8B-Instruct-abliterated", "8B", "Instruct", "Feiouex"),
        (
            "sonicrules1234/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated",
            "30B-A3B",
            "Instruct",
            "sonicrules1234",
        ),
        ("trithemius/Huihui-Qwen3-VL-2B-Instruct-abliterated", "2B", "Instruct", "trithemius"),
        ("Freesol/Huihui-Qwen3-VL-8B-Instruct-abliterated-merged", "8B", "Instruct", "Freesol"),
    ]
    for model_id, size, variant, source in community_abliterated_specs:
        entries.append(
            _entry(
                model_id,
                f"CUDA {source} Qwen3-VL {size} {variant} abliterated",
                size=size,
                variant=variant,
                source=source,
                abliterated=True,
                training_supported=_supports_current_trainer(size),
            )
        )

    quantized_abliterated_specs = [
        (
            "JinRiYao2001/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-AWQ",
            "30B-A3B",
            "Instruct",
            "AWQ 4-bit",
            "awq",
            "JinRiYao2001",
            _huihui_abliterated_model_id("30B-A3B", "Instruct"),
        ),
        (
            "nicklas373/Huihui-Qwen3-VL-8B-Thinking-abliterated-AWQ",
            "8B",
            "Thinking",
            "AWQ 4-bit",
            "awq",
            "nicklas373",
            _huihui_abliterated_model_id("8B", "Thinking"),
        ),
        (
            "nicklas373/Huihui-Qwen3-VL-8B-Thinking-abliterated-AWQ-8-bit",
            "8B",
            "Thinking",
            "AWQ 8-bit",
            "awq",
            "nicklas373",
            _huihui_abliterated_model_id("8B", "Thinking"),
        ),
        (
            "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated-FP8",
            "4B",
            "Instruct",
            "FP8",
            "fp8",
            "huihui-ai",
            _huihui_abliterated_model_id("4B", "Instruct"),
        ),
        (
            "Heouzen/Huihui-Qwen3-VL-8B-Instruct-FP8-abliterated",
            "8B",
            "Instruct",
            "FP8",
            "fp8",
            "Heouzen",
            _huihui_abliterated_model_id("8B", "Instruct"),
        ),
        (
            "Heouzen/Huihui-Qwen3-VL-32B-Instruct-FP8-abliterated",
            "32B",
            "Instruct",
            "FP8",
            "fp8",
            "Heouzen",
            _huihui_abliterated_model_id("32B", "Instruct"),
        ),
    ]
    for model_id, size, variant, quantization, quantization_backend, source, training_model_id in quantized_abliterated_specs:
        entries.append(
            _entry(
                model_id,
                f"CUDA Huihui Qwen3-VL {size} {variant} abliterated {quantization}",
                size=size,
                variant=variant,
                source=source,
                quantization=quantization,
                quantization_backend=quantization_backend,
                abliterated=True,
                training_model_id=training_model_id,
                training_supported=_supports_current_trainer(size),
            )
        )

    qwen35_36_specs = [
        ("Qwen/Qwen3.5-4B", "Qwen3.5", "4B", "Vision", "Qwen", None, "none", False, None),
        ("Qwen/Qwen3.5-9B", "Qwen3.5", "9B", "Vision", "Qwen", None, "none", False, None),
        ("Qwen/Qwen3.5-27B", "Qwen3.5", "27B", "Vision", "Qwen", None, "none", False, None),
        ("Qwen/Qwen3.5-35B-A3B", "Qwen3.5", "35B-A3B", "Vision", "Qwen", None, "none", False, None),
        ("cyankiwi/Qwen3.5-9B-AWQ-4bit", "Qwen3.5", "9B", "Vision", "cyankiwi", "AWQ 4-bit", "awq", False, "Qwen/Qwen3.5-9B"),
        ("cyankiwi/Qwen3.5-27B-AWQ-4bit", "Qwen3.5", "27B", "Vision", "cyankiwi", "AWQ 4-bit", "awq", False, "Qwen/Qwen3.5-27B"),
        ("huihui-ai/Huihui-Qwen3.5-9B-abliterated", "Qwen3.5", "9B", "Vision", "huihui-ai", None, "none", True, None),
        ("huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated", "Qwen3.5", "35B-A3B", "Vision", "huihui-ai", None, "none", True, None),
        ("prithivMLmods/Gliese-Qwen3.5-9B-Abliterated-Caption", "Qwen3.5", "9B", "Vision", "prithivMLmods", None, "none", True, "Qwen/Qwen3.5-9B"),
        ("Qwen/Qwen3.6-27B", "Qwen3.6", "27B", "Vision", "Qwen", None, "none", False, None),
        ("Qwen/Qwen3.6-35B-A3B", "Qwen3.6", "35B-A3B", "Vision", "Qwen", None, "none", False, None),
        ("Qwen/Qwen3.6-35B-A3B-FP8", "Qwen3.6", "35B-A3B", "Vision", "Qwen", "FP8", "fp8", False, "Qwen/Qwen3.6-35B-A3B"),
        ("cyankiwi/Qwen3.6-27B-AWQ-INT4", "Qwen3.6", "27B", "Vision", "cyankiwi", "AWQ 4-bit", "awq", False, "Qwen/Qwen3.6-27B"),
        ("btbtyler09/Qwen3.6-27B-GPTQ-4bit", "Qwen3.6", "27B", "Vision", "btbtyler09", "GPTQ 4-bit", "gptq", False, "Qwen/Qwen3.6-27B"),
        ("cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", "Qwen3.6", "35B-A3B", "Vision", "cyankiwi", "AWQ 4-bit", "awq", False, "Qwen/Qwen3.6-35B-A3B"),
        ("palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4", "Qwen3.6", "35B-A3B", "Vision", "palmfuture", "GPTQ 4-bit", "gptq", False, "Qwen/Qwen3.6-35B-A3B"),
        ("huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated", "Qwen3.6", "35B-A3B", "Vision", "huihui-ai", None, "none", True, None),
        ("shawnw3i/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP", "Qwen3.6", "27B", "Vision", "shawnw3i", "AWQ 4-bit", "awq", True, "Qwen/Qwen3.6-27B"),
        ("batsclamp/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-FP8", "Qwen3.6", "35B-A3B", "Vision", "batsclamp", "FP8", "fp8", True, "Qwen/Qwen3.6-35B-A3B"),
    ]
    for (
        model_id,
        model_line,
        size,
        variant,
        source,
        quantization,
        quantization_backend,
        abliterated,
        training_model_id,
    ) in qwen35_36_specs:
        quant_label = f" {quantization}" if quantization else ""
        ablit_label = " abliterated" if abliterated else ""
        entries.append(
            _entry(
                model_id,
                f"CUDA {source} {model_line} {size}{ablit_label}{quant_label}",
                size=size,
                variant=variant,
                source=source,
                quantization=quantization,
                quantization_backend=quantization_backend,
                abliterated=abliterated,
                training_model_id=training_model_id or model_id,
                training_supported=False,
                training_note=_QWEN3_5_6_INFERENCE_NOTE,
                model_line=model_line,
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


def _infer_qwen_size_variant(model_id: str) -> tuple[Optional[str], Optional[str]]:
    basename = str(model_id or "").rsplit("/", 1)[-1]
    match = re.search(
        r"Qwen3-VL-(?P<size>2B|4B|8B|32B|30B-A3B|235B-A22B)-(?P<variant>Instruct|Thinking)",
        basename,
    )
    if not match:
        match = re.search(
            r"Qwen3\.[56]-(?P<size>4B|9B|27B|35B-A3B)",
            basename,
        )
        if not match:
            return None, None
        return match.group("size"), "Vision"
    return match.group("size"), match.group("variant")


def _quantization_label(quantization_backend: str, model_id: str) -> Optional[str]:
    backend = str(quantization_backend or "none").lower()
    lowered = str(model_id or "").lower()
    if backend == "awq":
        return "AWQ 8-bit" if "8-bit" in lowered or "8bit" in lowered else "AWQ 4-bit"
    if backend == "gptq":
        return "GPTQ 4-bit"
    if backend == "fp8":
        return "FP8"
    if backend == "4bit":
        return "4-bit"
    if backend == "8bit":
        return "8-bit"
    return None


def qwen_transformers_metadata_for_model(model_id: str) -> Dict[str, Any]:
    for entry in QWEN_TRANSFORMERS_MODEL_OPTIONS:
        if str(entry.get("id")) == str(model_id):
            return dict(entry)
    raw = str(model_id)
    owner = raw.split("/", 1)[0] if "/" in raw else "huggingface"
    size, variant = _infer_qwen_size_variant(raw)
    quant_backend = infer_qwen_quantization_backend(raw)
    quantization = _quantization_label(quant_backend, raw)
    quantized = quant_backend != "none"
    abliterated = "abliterated" in raw.lower()
    training_model_id = resolve_qwen_training_model_id(raw)
    model_line = "Qwen3-VL"
    if "qwen3.6" in raw.lower():
        model_line = "Qwen3.6"
    elif "qwen3.5" in raw.lower():
        model_line = "Qwen3.5"
    label_parts = ["CUDA"]
    if owner and owner not in {"huggingface", "Qwen"}:
        label_parts.append(owner)
    label_parts.append(model_line)
    if size:
        label_parts.append(size)
    if variant:
        label_parts.append(variant)
    if abliterated:
        label_parts.append("abliterated")
    if quantization:
        label_parts.append(quantization)
    training_note = None
    if quantized and training_model_id != raw:
        training_note = (
            "Training starts from the resolved unquantized abliterated Transformers checkpoint; "
            "QLoRA applies bitsandbytes 4-bit at load time."
            if abliterated
            else "Training starts from the resolved full Transformers checkpoint; "
            "QLoRA applies bitsandbytes 4-bit at load time."
        )
    elif size in _QWEN_MOE_SIZES:
        training_note = _QWEN_MOE_TRAINING_NOTE
    return {
        "id": raw,
        "label": " ".join(label_parts) if len(label_parts) > 2 else raw,
        "model_id": raw,
        "runtime_platform": QWEN_PLATFORM_TRANSFORMERS,
        "size": size,
        "variant": variant,
        "source": owner,
        "quantization": quantization,
        "quantization_backend": quant_backend,
        "quantized": quantized,
        "abliterated": abliterated,
        "training_supported": True,
        "training_modes": ["official_lora", "trl_qlora"],
        "training_model_id": training_model_id,
        "training_note": training_note,
        "dataset_context": (
            f"Abliterated {model_line} Transformer checkpoint for CUDA/CPU inference and adapter training."
            if abliterated
            else f"{model_line} Transformer checkpoint for CUDA/CPU inference and adapter training."
        ),
        "model_line": model_line,
    }


def infer_qwen_quantization_backend(model_id: Optional[str]) -> str:
    lowered = str(model_id or "").strip().lower()
    if "awq" in lowered:
        return "awq"
    if "gptq" in lowered:
        return "gptq"
    if "fp8" in lowered:
        return "fp8"
    if re.search(r"(^|[-_])(4bit|4-bit|int4)([-_]|$)", lowered):
        return "4bit"
    if re.search(r"(^|[-_])(8bit|8-bit|int8)([-_]|$)", lowered):
        return "8bit"
    return "none"


def is_qwen_mlx_model_id(model_id: Optional[str]) -> bool:
    raw = str(model_id or "").strip()
    lowered = raw.lower()
    return (
        raw.startswith("mlx-community/")
        or lowered.endswith("-mlx")
        or "-mlx-" in lowered
        or lowered.startswith("goekdeniz-guelmez/josiefied-qwen3-vl-")
    )


def _strip_known_training_quant_suffix(model_id: str) -> Optional[str]:
    special = re.sub(r"(?i)-FP8-abliterated$", "-abliterated", model_id)
    if special != model_id:
        return special
    for suffix in (
        "-AWQ-8-bit",
        "-AWQ-8bit",
        "-AWQ-4bit",
        "-4bit-AWQ",
        "-4-bit-AWQ",
        "-8bit-AWQ",
        "-8-bit-AWQ",
        "-AWQ",
        "-GPTQ-Int4",
        "-GPTQ-Int8",
        "-GPTQ-4bit",
        "-4bit-GPTQ",
        "-4-bit-GPTQ",
        "-GPTQ",
        "-4bit",
        "-4-bit",
        "-8bit",
        "-8-bit",
        "-FP8",
        "-INT4",
        "-INT8",
    ):
        if model_id.lower().endswith(suffix.lower()):
            return model_id[: -len(suffix)]
    return None


def _infer_official_base_from_quantized_name(model_id: str) -> Optional[str]:
    size, variant = _infer_qwen_size_variant(model_id)
    if not size or not variant:
        return None
    basename = str(model_id or "").rsplit("/", 1)[-1]
    generation_match = re.search(r"Qwen3\.(?P<version>[56])-", basename)
    if generation_match:
        return f"Qwen/Qwen3.{generation_match.group('version')}-{size}"
    return _official_model_id(size, variant)


def _infer_abliterated_base_from_quantized_name(model_id: str) -> Optional[str]:
    size, variant = _infer_qwen_size_variant(model_id)
    if not size or not variant:
        return None
    lowered = str(model_id or "").lower()
    if "abliterated" not in lowered:
        return None
    stripped = _strip_known_training_quant_suffix(str(model_id or "").strip())
    if stripped and "abliterated" in stripped.lower():
        return stripped
    if "huihui" in lowered:
        return _huihui_abliterated_model_id(size, variant)
    return _huihui_abliterated_model_id(size, variant)


def resolve_qwen_training_model_id(model_id: Optional[str]) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return "Qwen/Qwen3-VL-4B-Instruct"
    if raw in _TRAINING_MODEL_BY_ID:
        return _TRAINING_MODEL_BY_ID[raw]
    stripped = _strip_known_training_quant_suffix(raw)
    is_abliterated = "abliterated" in raw.lower()
    if is_abliterated and stripped and "abliterated" in stripped.lower():
        return stripped
    if stripped and stripped.startswith("Qwen/Qwen3-VL-"):
        return stripped
    if infer_qwen_quantization_backend(raw) in {"awq", "gptq", "fp8", "4bit", "8bit"}:
        if is_abliterated:
            inferred_abliterated = _infer_abliterated_base_from_quantized_name(raw)
            if inferred_abliterated:
                return inferred_abliterated
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
    if on_cuda and quant_backend in {"awq", "gptq", "fp8", "4bit", "8bit"}:
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
