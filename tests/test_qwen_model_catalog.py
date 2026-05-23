from pathlib import Path
from types import SimpleNamespace

from services.qwen_model_catalog import (
    QWEN_TRANSFORMERS_MODEL_IDS,
    QWEN_TRANSFORMERS_MODEL_OPTIONS,
    qwen_transformers_metadata_for_model,
    qwen_transformers_load_kwargs,
    resolve_qwen_training_model_id,
)


class _Cuda:
    @staticmethod
    def is_available():
        return True


class _Torch:
    cuda = _Cuda()
    float16 = SimpleNamespace(name="float16")
    float32 = SimpleNamespace(name="float32")


def test_cuda_catalog_includes_all_official_qwen3_vl_full_and_fp8_models():
    sizes = ("2B", "4B", "8B", "32B", "30B-A3B", "235B-A22B")
    variants = ("Instruct", "Thinking")

    expected = {
        model_id
        for size in sizes
        for variant in variants
        for model_id in (
            f"Qwen/Qwen3-VL-{size}-{variant}",
            f"Qwen/Qwen3-VL-{size}-{variant}-FP8",
        )
    }

    assert expected <= QWEN_TRANSFORMERS_MODEL_IDS


def test_cuda_catalog_marks_fp8_as_quantized_inference_with_full_training_base():
    entry = next(
        item
        for item in QWEN_TRANSFORMERS_MODEL_OPTIONS
        if item["id"] == "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
    )

    assert entry["quantized"] is True
    assert entry["quantization_backend"] == "fp8"
    assert entry["training_supported"] is True
    assert entry["training_modes"] == ["official_lora", "trl_qlora"]
    assert entry["training_model_id"] == "Qwen/Qwen3-VL-235B-A22B-Thinking"


def test_cuda_catalog_marks_dense_and_moe_transformers_models_as_trainable():
    dense_entry = next(
        item
        for item in QWEN_TRANSFORMERS_MODEL_OPTIONS
        if item["id"] == "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"
    )
    dense_quantized_entry = next(
        item
        for item in QWEN_TRANSFORMERS_MODEL_OPTIONS
        if item["id"] == "nicklas373/Huihui-Qwen3-VL-8B-Thinking-abliterated-AWQ"
    )
    moe_entry = next(
        item
        for item in QWEN_TRANSFORMERS_MODEL_OPTIONS
        if item["id"] == "huihui-ai/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated"
    )
    moe_quantized_entry = next(
        item
        for item in QWEN_TRANSFORMERS_MODEL_OPTIONS
        if item["id"] == "JinRiYao2001/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-AWQ"
    )

    assert dense_entry["training_supported"] is True
    assert dense_entry["training_modes"] == ["official_lora", "trl_qlora"]
    assert dense_quantized_entry["training_supported"] is True
    assert dense_quantized_entry["training_model_id"] == "huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated"
    assert moe_entry["training_supported"] is True
    assert moe_entry["training_modes"] == ["official_lora", "trl_qlora"]
    assert "Qwen3VLMoe" in moe_entry["training_note"]
    assert moe_quantized_entry["training_supported"] is True
    assert moe_quantized_entry["training_model_id"] == "huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated"


def test_unknown_custom_qwen_transformers_model_defaults_to_trainable():
    metadata = qwen_transformers_metadata_for_model("local/qwen3-vl-custom")

    assert metadata["training_supported"] is True
    assert metadata["training_modes"] == ["official_lora", "trl_qlora"]


def test_unknown_custom_abliterated_transformers_model_metadata_is_inferred():
    metadata = qwen_transformers_metadata_for_model(
        "custom/Huihui-Qwen3-VL-4B-Instruct-abliterated-AWQ-4bit"
    )

    assert metadata["training_supported"] is True
    assert metadata["abliterated"] is True
    assert metadata["quantized"] is True
    assert metadata["quantization_backend"] == "awq"
    assert metadata["training_model_id"] == "custom/Huihui-Qwen3-VL-4B-Instruct-abliterated"
    assert "abliterated" in metadata["training_note"]


def test_cuda_catalog_includes_primary_abliterated_qwen3_vl_transformers_models():
    sizes = ("2B", "4B", "8B", "32B", "30B-A3B")
    variants = ("Instruct", "Thinking")
    expected_huihui = {
        f"huihui-ai/Huihui-Qwen3-VL-{size}-{variant}-abliterated"
        for size in sizes
        for variant in variants
    }
    expected_prithiv = {
        f"prithivMLmods/Qwen3-VL-{size}-{variant}-abliterated-v1"
        for size in sizes
        for variant in variants
    }
    expected_quantized = {
        "JinRiYao2001/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-AWQ",
        "nicklas373/Huihui-Qwen3-VL-8B-Thinking-abliterated-AWQ",
        "nicklas373/Huihui-Qwen3-VL-8B-Thinking-abliterated-AWQ-8-bit",
        "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated-FP8",
        "Heouzen/Huihui-Qwen3-VL-8B-Instruct-FP8-abliterated",
        "Heouzen/Huihui-Qwen3-VL-32B-Instruct-FP8-abliterated",
    }
    expected_community = {
        "Feiouex/Huihui-Qwen3-VL-8B-Instruct-abliterated",
        "sonicrules1234/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated",
        "trithemius/Huihui-Qwen3-VL-2B-Instruct-abliterated",
        "Freesol/Huihui-Qwen3-VL-8B-Instruct-abliterated-merged",
    }

    assert expected_huihui <= QWEN_TRANSFORMERS_MODEL_IDS
    assert expected_prithiv <= QWEN_TRANSFORMERS_MODEL_IDS
    assert "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2" in QWEN_TRANSFORMERS_MODEL_IDS
    assert expected_quantized <= QWEN_TRANSFORMERS_MODEL_IDS
    assert expected_community <= QWEN_TRANSFORMERS_MODEL_IDS


def test_awq_runtime_uses_device_map_for_explicit_cuda_device():
    kwargs = qwen_transformers_load_kwargs(
        "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit",
        device="cuda:1",
        device_pref="cuda:1",
        torch_module=_Torch,
    )

    assert kwargs["torch_dtype"] == "auto"
    assert kwargs["device_map"] == {"": "cuda:1"}


def test_fp8_runtime_uses_device_map_for_explicit_cuda_device():
    kwargs = qwen_transformers_load_kwargs(
        "Qwen/Qwen3-VL-4B-Instruct-FP8",
        device="cuda:1",
        device_pref="cuda:1",
        torch_module=_Torch,
    )

    assert kwargs["torch_dtype"] == "auto"
    assert kwargs["device_map"] == {"": "cuda:1"}


def test_quantized_training_model_resolves_to_full_base_checkpoint():
    assert (
        resolve_qwen_training_model_id("cyankiwi/Qwen3-VL-8B-Thinking-AWQ-4bit")
        == "Qwen/Qwen3-VL-8B-Thinking"
    )
    assert (
        resolve_qwen_training_model_id("pramjana/Qwen3-VL-4B-Instruct-4bit-GPTQ")
        == "Qwen/Qwen3-VL-4B-Instruct"
    )
    assert (
        resolve_qwen_training_model_id("Qwen/Qwen3-VL-235B-A22B-Thinking-FP8")
        == "Qwen/Qwen3-VL-235B-A22B-Thinking"
    )


def test_cuda_requirements_cover_qwen_training_and_quantized_inference():
    repo_root = Path(__file__).resolve().parents[1]
    requirements = (repo_root / "requirements.txt").read_text(encoding="utf-8")
    constraints = (repo_root / "constraints" / "falcon-cu118.txt").read_text(encoding="utf-8")

    for package_name in (
        "compressed-tensors",
        "gptqmodel",
        "optimum",
        "bitsandbytes",
        "peft",
        "trl",
    ):
        assert package_name in requirements
        assert package_name in constraints


def test_abliterated_training_model_stays_on_abliterated_checkpoint():
    assert (
        resolve_qwen_training_model_id("huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated")
        == "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"
    )
    assert (
        resolve_qwen_training_model_id("prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1")
        == "prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1"
    )
    assert (
        resolve_qwen_training_model_id("nicklas373/Huihui-Qwen3-VL-8B-Thinking-abliterated-AWQ")
        == "huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated"
    )
    assert (
        resolve_qwen_training_model_id("Heouzen/Huihui-Qwen3-VL-32B-Instruct-FP8-abliterated")
        == "huihui-ai/Huihui-Qwen3-VL-32B-Instruct-abliterated"
    )


def test_unknown_quantized_abliterated_training_model_keeps_abliterated_base():
    assert (
        resolve_qwen_training_model_id(
            "custom/Huihui-Qwen3-VL-4B-Instruct-abliterated-AWQ-4bit"
        )
        == "custom/Huihui-Qwen3-VL-4B-Instruct-abliterated"
    )
    assert (
        resolve_qwen_training_model_id(
            "custom/Qwen3-VL-8B-Thinking-abliterated-v1-4bit-GPTQ"
        )
        == "custom/Qwen3-VL-8B-Thinking-abliterated-v1"
    )
    assert (
        resolve_qwen_training_model_id(
            "custom/Huihui-Qwen3-VL-32B-Instruct-FP8-abliterated"
        )
        == "custom/Huihui-Qwen3-VL-32B-Instruct-abliterated"
    )


def test_generic_4bit_abliterated_cuda_model_resolves_to_abliterated_base():
    metadata = qwen_transformers_metadata_for_model(
        "custom/Huihui-Qwen3-VL-8B-Instruct-abliterated-4bit"
    )

    assert metadata["quantized"] is True
    assert metadata["quantization_backend"] == "4bit"
    assert metadata["training_model_id"] == "custom/Huihui-Qwen3-VL-8B-Instruct-abliterated"
    assert (
        resolve_qwen_training_model_id(
            "custom/Huihui-Qwen3-VL-8B-Instruct-abliterated-4bit"
        )
        == "custom/Huihui-Qwen3-VL-8B-Instruct-abliterated"
    )


def test_generic_4bit_cuda_runtime_uses_device_map_for_explicit_cuda_device():
    kwargs = qwen_transformers_load_kwargs(
        "custom/Qwen3-VL-4B-Instruct-4bit",
        device="cuda:1",
        device_pref="cuda:1",
        torch_module=_Torch,
    )

    assert kwargs["torch_dtype"] == "auto"
    assert kwargs["device_map"] == {"": "cuda:1"}
