"""MLX-VLM helpers for Qwen runtime selection and built-in model options."""

from __future__ import annotations

import platform as platform_lib
from typing import Any, Dict, List, Optional

QWEN_PLATFORM_AUTO = "auto"
QWEN_PLATFORM_TRANSFORMERS = "transformers"
QWEN_PLATFORM_MLX = "mlx_vlm"
QWEN_PLATFORM_ALIASES = {
    "auto": QWEN_PLATFORM_AUTO,
    "hf": QWEN_PLATFORM_TRANSFORMERS,
    "torch": QWEN_PLATFORM_TRANSFORMERS,
    "transformer": QWEN_PLATFORM_TRANSFORMERS,
    "transformers": QWEN_PLATFORM_TRANSFORMERS,
    "mlx": QWEN_PLATFORM_MLX,
    "mlx-vlm": QWEN_PLATFORM_MLX,
    "mlx_vlm": QWEN_PLATFORM_MLX,
}
QWEN_AEON_QWEN36_27B_MLX_MODEL = (
    "AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4"
)
# General Apple Silicon Qwen inference should use the largest validated local
# vision-capable path by default. Captioning keeps a separate compact default
# below because one caption action can trigger many model calls.
QWEN_MLX_DEFAULT_MODEL = QWEN_AEON_QWEN36_27B_MLX_MODEL
QWEN_MLX_CAPTION_DEFAULT_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
QWEN_AEON_QWEN36_27B_MLX_NOTE = (
    "Inference-only AEON Qwen3.6 27B multimodal MLX FP4 checkpoint. "
    "mlx-vlm 0.6.x exposes the matching Qwen3.5 vision tower as qwen3_5, while "
    "this checkpoint stores it as qwen3_5_vision; the backend normalizes that "
    "config alias before load. Adapter training is not wired."
)
QWEN_VANCH007_QWEN36_35B_MLX_MODEL = "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit"
QWEN_VANCH007_QWEN36_35B_MLX_NOTE = (
    "Experimental Qwen3.6 35B-A3B abliterated MLX-VLM checkpoint. "
    "Class-split vignette smoke tests passed with mlx-vlm 0.6.1 plus the local "
    "Qwen3.5/3.6 MoE split-weight compatibility shim; adapter training is not "
    "enabled until tested on this architecture."
)
QWEN_KNOWN_INCOMPATIBLE_MLX_LANGUAGE_ONLY_NOTE = (
    "This MLX repack contains only language_model weights and no vision_tower weights, "
    "so it cannot run Qwen3-VL image captioning, detection, or vision LoRA training."
)
QWEN_KNOWN_INCOMPATIBLE_MLX_MODELS = {
    "Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit": (
        "Candidate Heretic Qwen3.6 35B-A3B MLX checkpoint was removed from the "
        "model list because smoke tests generated invalid text in the Class Split "
        "vignette benchmark."
    ),
    "introvoyz041/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx-mlx-4Bit": (
        QWEN_KNOWN_INCOMPATIBLE_MLX_LANGUAGE_ONLY_NOTE
    ),
    "introvoyz041/Huihui-Qwen3-VL-32B-Thinking-abliterated-qx65-hi-mlx-mlx-4Bit": (
        QWEN_KNOWN_INCOMPATIBLE_MLX_LANGUAGE_ONLY_NOTE
    ),
}


def normalize_qwen_platform(value: Optional[str]) -> str:
    key = str(value or QWEN_PLATFORM_AUTO).strip().lower()
    return QWEN_PLATFORM_ALIASES.get(key, QWEN_PLATFORM_AUTO)


def is_mac_hardware() -> bool:
    system = platform_lib.system().lower()
    machine = platform_lib.machine().lower()
    return system == "darwin" and machine in {"arm64", "aarch64"}


def is_mlx_model_id(model_id: Optional[str]) -> bool:
    raw = str(model_id or "").strip()
    if raw.startswith("mlx-community/"):
        return True
    try:
        if raw in QWEN_MLX_MODEL_IDS:
            return True
    except NameError:
        pass
    lowered = raw.lower()
    return (
        lowered.endswith("-mlx")
        or "-mlx-" in lowered
        or lowered.startswith("goekdeniz-guelmez/josiefied-qwen3-vl-")
    )


def qwen_known_incompatible_mlx_detail(model_id: str) -> Optional[str]:
    raw = str(model_id or "").strip()
    note = QWEN_KNOWN_INCOMPATIBLE_MLX_MODELS.get(raw)
    if not note:
        return None
    return f"{raw}: {note}"


def _model_entry(size: str, variant: str, quant: str, *, moe: bool = False) -> Dict[str, Any]:
    stem = f"Qwen3-VL-{size}-{variant}"
    if moe:
        stem = f"Qwen3-VL-{size}-A3B-{variant}"
    model_id = f"mlx-community/{stem}-{quant}"
    return {
        "id": model_id,
        "label": f"MLX Qwen3-VL {size}{'-A3B' if moe else ''} {variant} {quant}",
        "model_id": model_id,
        "size": size,
        "variant": variant,
        "quantization": quant,
        "runtime_platform": QWEN_PLATFORM_MLX,
        "source": "mlx-community",
        "vision_inference_supported": True,
        "training_supported": True,
    }


def _external_mlx_entry(
    model_id: str,
    *,
    label: str,
    size: str,
    variant: str,
    quantization: str,
    source: str,
    abliterated: bool = False,
    vision_inference_supported: bool = True,
    training_supported: bool = True,
    compatibility_note: Optional[str] = None,
    training_note: Optional[str] = None,
) -> Dict[str, Any]:
    entry = {
        "id": model_id,
        "label": label,
        "model_id": model_id,
        "size": size,
        "variant": variant,
        "quantization": quantization,
        "runtime_platform": QWEN_PLATFORM_MLX,
        "source": source,
        "abliterated": abliterated,
        "vision_inference_supported": vision_inference_supported,
        "training_supported": training_supported,
    }
    if compatibility_note:
        entry["compatibility_note"] = compatibility_note
    if training_note:
        entry["training_note"] = training_note
    return entry


def qwen_mlx_model_options() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    size_quants = {
        "2B": {
            "Instruct": ["3bit", "4bit", "5bit", "6bit", "8bit", "bf16"],
            "Thinking": ["3bit", "4bit", "5bit", "6bit", "8bit", "bf16"],
        },
        "4B": {
            "Instruct": ["3bit", "4bit", "5bit", "6bit", "8bit"],
            "Thinking": ["3bit", "4bit", "5bit", "6bit", "8bit", "bf16"],
        },
        "8B": {
            "Instruct": ["3bit", "4bit", "5bit", "6bit", "8bit", "bf16"],
            "Thinking": ["3bit", "4bit", "5bit", "6bit", "8bit", "bf16"],
        },
        "32B": {
            "Instruct": ["3bit", "4bit", "5bit", "6bit", "8bit", "bf16"],
            "Thinking": ["3bit", "4bit"],
        },
    }
    for size, variant_map in size_quants.items():
        for variant, quants in variant_map.items():
            for quant in quants:
                entries.append(_model_entry(size, variant, quant))
    for variant in ("Instruct", "Thinking"):
        for quant in ("3bit", "4bit", "5bit", "6bit", "8bit", "bf16"):
            entries.append(_model_entry("30B", variant, quant, moe=True))
    entries.extend(
        [
            {
                "id": "mlx-community/Qwen3-VL-235B-A22B-Instruct-3bit",
                "label": "MLX Qwen3-VL 235B-A22B Instruct 3bit",
                "model_id": "mlx-community/Qwen3-VL-235B-A22B-Instruct-3bit",
                "size": "235B-A22B",
                "variant": "Instruct",
                "quantization": "3bit",
                "runtime_platform": QWEN_PLATFORM_MLX,
            },
            {
                "id": "mlx-community/Qwen3-VL-235B-A22B-Instruct-4bit",
                "label": "MLX Qwen3-VL 235B-A22B Instruct 4bit",
                "model_id": "mlx-community/Qwen3-VL-235B-A22B-Instruct-4bit",
                "size": "235B-A22B",
                "variant": "Instruct",
                "quantization": "4bit",
                "runtime_platform": QWEN_PLATFORM_MLX,
            },
            {
                "id": "mlx-community/Qwen3-VL-235B-A22B-Thinking-3bit",
                "label": "MLX Qwen3-VL 235B-A22B Thinking 3bit",
                "model_id": "mlx-community/Qwen3-VL-235B-A22B-Thinking-3bit",
                "size": "235B-A22B",
                "variant": "Thinking",
                "quantization": "3bit",
                "runtime_platform": QWEN_PLATFORM_MLX,
            },
        ]
    )
    for size in ("2B", "4B"):
        for variant in ("Instruct", "Thinking"):
            base = f"EZCon/Huihui-Qwen3-VL-{size}-{variant}-abliterated"
            for quant, suffix in (
                ("mlx", "mlx"),
                ("4bit", "4bit-mlx"),
                ("8bit", "8bit-mlx"),
                ("mixed 4/8-bit MXFP4", "4bit-g32-mxfp4-mixed_4_8-mlx"),
            ):
                model_id = f"{base}-{suffix}"
                entries.append(
                    _external_mlx_entry(
                        model_id,
                        label=f"MLX Huihui Qwen3-VL {size} {variant} abliterated {quant}",
                        size=size,
                        variant=variant,
                        quantization=quant,
                        source="EZCon",
                        abliterated=True,
                    )
                )
    for size in ("4B", "8B"):
        for quant, suffix in (
            ("mlx", "mlx"),
            ("q2", "q2-mlx"),
            ("q3", "q3-mlx"),
            ("q4", "q4-mlx"),
            ("q6", "q6-mlx"),
            ("q8", "q8-mlx"),
        ):
            model_id = f"alexgusevski/Huihui-Qwen3-VL-{size}-Instruct-abliterated-{suffix}"
            entries.append(
                _external_mlx_entry(
                    model_id,
                    label=f"MLX Huihui Qwen3-VL {size} Instruct abliterated {quant}",
                    size=size,
                    variant="Instruct",
                    quantization=quant,
                    source="alexgusevski",
                    abliterated=True,
                )
            )
    for model_id, size, variant, quantization, source in (
        (
            QWEN_AEON_QWEN36_27B_MLX_MODEL,
            "27B",
            "AEON Ultimate Uncensored",
            "FP4",
            "AEON-7",
        ),
        (
            QWEN_VANCH007_QWEN36_35B_MLX_MODEL,
            "35B-A3B",
            "Abliterated",
            "4bit",
            "vanch007",
        ),
        (
            "nightmedia/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx",
            "30B-A3B",
            "Thinking",
            "qx86-hi",
            "nightmedia",
        ),
        (
            "nightmedia/Huihui-Qwen3-VL-32B-Thinking-abliterated-qx65-hi-mlx",
            "32B",
            "Thinking",
            "qx65-hi",
            "nightmedia",
        ),
        (
            "veeceey/Huihui-Qwen3-VL-8B-Instruct-abliterated-mlx-4bit",
            "8B",
            "Instruct",
            "4bit",
            "veeceey",
        ),
        (
            "Goekdeniz-Guelmez/Josiefied-Qwen3-VL-4B-Instruct-abliterated-beta-v1",
            "4B",
            "Instruct",
            "beta v1",
            "Goekdeniz-Guelmez",
        ),
    ):
        if source == "AEON-7":
            label = f"MLX AEON Qwen3.6 {size} {variant} {quantization}"
        elif source == "Goekdeniz-Guelmez":
            label = f"MLX Josiefied Qwen3-VL {size} {variant} abliterated {quantization}"
        elif source == "vanch007":
            label = f"MLX Qwen3.6 {size} abliterated {quantization}"
        else:
            label = f"MLX Huihui Qwen3-VL {size} {variant} abliterated {quantization}"
        is_aeon_qwen36 = model_id == QWEN_AEON_QWEN36_27B_MLX_MODEL
        is_vanch007_qwen36 = model_id == QWEN_VANCH007_QWEN36_35B_MLX_MODEL
        entries.append(
            _external_mlx_entry(
                model_id,
                label=label,
                size=size,
                variant=variant,
                quantization=quantization,
                source=source,
                abliterated=True,
                vision_inference_supported=True,
                training_supported=not (is_aeon_qwen36 or is_vanch007_qwen36),
                compatibility_note=(
                    QWEN_AEON_QWEN36_27B_MLX_NOTE
                    if is_aeon_qwen36
                    else
                    QWEN_VANCH007_QWEN36_35B_MLX_NOTE
                    if is_vanch007_qwen36
                    else None
                ),
                training_note=(
                    QWEN_AEON_QWEN36_27B_MLX_NOTE
                    if is_aeon_qwen36
                    else
                    QWEN_VANCH007_QWEN36_35B_MLX_NOTE
                    if is_vanch007_qwen36
                    else None
                ),
            )
        )
    preferred = {
        QWEN_AEON_QWEN36_27B_MLX_MODEL: 0,
        "mlx-community/Qwen3-VL-4B-Instruct-4bit": 1,
        "mlx-community/Qwen3-VL-2B-Instruct-4bit": 2,
        "mlx-community/Qwen3-VL-8B-Instruct-4bit": 3,
        "mlx-community/Qwen3-VL-4B-Thinking-4bit": 4,
        "mlx-community/Qwen3-VL-8B-Thinking-4bit": 5,
        "EZCon/Huihui-Qwen3-VL-4B-Instruct-abliterated-4bit-mlx": 6,
        "EZCon/Huihui-Qwen3-VL-2B-Instruct-abliterated-4bit-mlx": 7,
        "alexgusevski/Huihui-Qwen3-VL-8B-Instruct-abliterated-q4-mlx": 8,
        "nightmedia/Huihui-Qwen3-VL-32B-Thinking-abliterated-qx65-hi-mlx": 9,
        QWEN_VANCH007_QWEN36_35B_MLX_MODEL: 10,
    }
    entries.sort(
        key=lambda entry: (
            preferred.get(str(entry["id"]), 1000),
            str(entry.get("size") or ""),
            str(entry.get("variant") or ""),
            str(entry.get("quantization") or ""),
        )
    )
    return entries


QWEN_MLX_MODEL_OPTIONS = qwen_mlx_model_options()
QWEN_MLX_MODEL_IDS = {str(entry["id"]) for entry in QWEN_MLX_MODEL_OPTIONS}
QWEN_MLX_VISION_MODEL_OPTIONS = [
    entry for entry in QWEN_MLX_MODEL_OPTIONS if entry.get("vision_inference_supported", True) is not False
]
QWEN_MLX_VISION_MODEL_IDS = {str(entry["id"]) for entry in QWEN_MLX_VISION_MODEL_OPTIONS}


def mlx_available(import_error: Optional[BaseException], load_fn: Any, generate_fn: Any) -> bool:
    return import_error is None and load_fn is not None and generate_fn is not None


def select_qwen_platform(
    configured_platform: str,
    *,
    model_id: Optional[str],
    adapter_path: Optional[Any],
    mlx_import_error: Optional[BaseException],
    mlx_load_fn: Any,
    mlx_generate_fn: Any,
) -> str:
    configured = normalize_qwen_platform(configured_platform)
    if configured == QWEN_PLATFORM_TRANSFORMERS:
        return QWEN_PLATFORM_TRANSFORMERS
    if configured == QWEN_PLATFORM_MLX:
        return QWEN_PLATFORM_MLX
    if adapter_path and is_mlx_model_id(model_id):
        return QWEN_PLATFORM_MLX
    if adapter_path:
        return QWEN_PLATFORM_TRANSFORMERS
    if is_mlx_model_id(model_id):
        return QWEN_PLATFORM_MLX
    if is_mac_hardware() and mlx_available(mlx_import_error, mlx_load_fn, mlx_generate_fn):
        return QWEN_PLATFORM_MLX
    return QWEN_PLATFORM_TRANSFORMERS


def _split_quantized_qwen3_model_id(model_id: str) -> tuple[str, Optional[str]]:
    raw = str(model_id or "").strip()
    if "/" in raw:
        raw = raw.rsplit("/", 1)[1]
    for suffix in ("-3bit", "-4bit", "-5bit", "-6bit", "-8bit", "-bf16"):
        if raw.endswith(suffix):
            return raw[: -len(suffix)], suffix[1:]
    for suffix in ("-FP8", "-GPTQ-Int4", "-GPTQ-Int8", "-AWQ", "-INT4", "-INT8"):
        if raw.endswith(suffix):
            return raw[: -len(suffix)], None
    return raw, None


def resolve_mlx_model_id(
    model_id: Optional[str],
    *,
    default_mlx_model_id: str = QWEN_MLX_DEFAULT_MODEL,
    default_quantization: str = "4bit",
) -> str:
    candidate = str(model_id or "").strip()
    if not candidate:
        return default_mlx_model_id
    if is_mlx_model_id(candidate):
        return candidate
    stem, quant = _split_quantized_qwen3_model_id(candidate)
    if not stem.startswith("Qwen3-VL-"):
        return candidate
    quant = quant or str(default_quantization or "4bit").strip() or "4bit"
    resolved = f"mlx-community/{stem}-{quant}"
    if resolved in QWEN_MLX_MODEL_IDS:
        return resolved
    return default_mlx_model_id


def qwen_mlx_metadata_for_model(model_id: str) -> Dict[str, Any]:
    for entry in QWEN_MLX_MODEL_OPTIONS:
        if str(entry.get("id")) == str(model_id):
            return dict(entry)
    return {
        "id": model_id,
        "label": str(model_id),
        "model_id": str(model_id),
        "runtime_platform": QWEN_PLATFORM_MLX,
        "vision_inference_supported": True,
        "training_supported": True,
    }
