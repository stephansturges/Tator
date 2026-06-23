from services.agent_model_catalog import (
    AGENT_MLX_MODEL_IDS,
    AGENT_MODEL_IDS,
    AGENT_MODEL_OPTIONS,
    AGENT_TRANSFORMERS_MODEL_IDS,
)
from services.qwen_mlx import QWEN_PLATFORM_MLX, QWEN_PLATFORM_TRANSFORMERS


def test_agent_catalog_includes_qwen_inference_family():
    expected = {
        "mlx-community/Qwen3-VL-2B-Instruct-4bit",
        "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        "mlx-community/Qwen3-VL-8B-Instruct-4bit",
        "mlx-community/Qwen3-VL-4B-Thinking-4bit",
        "mlx-community/Qwen3-VL-8B-Thinking-4bit",
        "EZCon/Huihui-Qwen3-VL-2B-Instruct-abliterated-4bit-mlx",
        "EZCon/Huihui-Qwen3-VL-4B-Instruct-abliterated-4bit-mlx",
        "alexgusevski/Huihui-Qwen3-VL-8B-Instruct-abliterated-q4-mlx",
        "nightmedia/Huihui-Qwen3-VL-32B-Thinking-abliterated-qx65-hi-mlx",
        "empero-ai/Qwable-9B-Claude-Fable-5",
        "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
        "AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4",
        "AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit",
    }
    removed_unusable = {
        "Jackrong/Qwopus3.6-27B-v2",
        "prithivMLmods/Qwen3.6-35B-A3B-abliterated-MAX",
        "nex-agi/Nex-N2-mini",
        "huihui-ai/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated",
        "mlx-community/gemma-4-31B-it-qat-4bit",
        "vanch007/Huihui-gemma-4-26B-A4B-it-abliterated-mlx-4bit",
    }

    assert expected <= AGENT_MODEL_IDS
    assert not (removed_unusable & AGENT_MODEL_IDS)
    assert (expected - AGENT_TRANSFORMERS_MODEL_IDS) <= AGENT_MLX_MODEL_IDS
    assert "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit" in AGENT_MLX_MODEL_IDS
    assert {
        "empero-ai/Qwable-9B-Claude-Fable-5",
        "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
        "AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS",
    } <= AGENT_TRANSFORMERS_MODEL_IDS


def test_agent_catalog_is_inference_only_not_training():
    for entry in AGENT_MODEL_OPTIONS:
        assert entry["training_supported"] is False
        assert entry["training_modes"] == []
        assert entry["agent_model"] is True
        assert entry["runtime_platform"] in {QWEN_PLATFORM_MLX, QWEN_PLATFORM_TRANSFORMERS}
        assert entry["smoke_status"] in {
            "class_split_benchmark_passed",
            "missing_image_processor",
            "metadata_verified",
            "transformers5_processor_passed",
            "qwen_mlx_runtime_supported",
        }


def test_agent_catalog_marks_qwen36_matrix_winners_separately():
    by_id = {entry["id"]: entry for entry in AGENT_MODEL_OPTIONS}

    assert by_id["mlx-community/Qwen3.6-35B-A3B-4bit"]["smoke_status"] == "class_split_benchmark_passed"
    assert (
        by_id["vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit"]["smoke_status"]
        == "class_split_benchmark_passed"
    )
    assert by_id["mlx-community/Qwen3-VL-4B-Instruct-4bit"]["smoke_status"] == "qwen_mlx_runtime_supported"
    assert by_id["mlx-community/Qwen3-VL-4B-Instruct-4bit"]["backend_status"] == "validated_runtime"
    aeon_mlx = by_id["AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4"]
    aeon_cuda = by_id["AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS"]
    assert aeon_mlx["runtime_platform"] == QWEN_PLATFORM_MLX
    assert aeon_mlx["smoke_status"] == "metadata_verified"
    assert aeon_cuda["runtime_platform"] == QWEN_PLATFORM_TRANSFORMERS
    assert aeon_cuda["smoke_status"] == "metadata_verified"


def test_agent_catalog_marks_empero_visual_capability():
    by_id = {entry["id"]: entry for entry in AGENT_MODEL_OPTIONS}
    qwable = by_id["empero-ai/Qwable-9B-Claude-Fable-5"]
    qwythos = by_id["empero-ai/Qwythos-9B-Claude-Mythos-5-1M"]

    assert qwable["runtime_platform"] == QWEN_PLATFORM_TRANSFORMERS
    assert qwable["vision_inference_supported"] is True
    assert qwable["smoke_status"] == "transformers5_processor_passed"
    assert qwable["min_transformers"] == "5.7.0"
    assert qwythos["runtime_platform"] == QWEN_PLATFORM_TRANSFORMERS
    assert qwythos["vision_inference_supported"] is False
    assert qwythos["smoke_status"] == "missing_image_processor"
    assert "missing" in qwythos["compatibility_note"].lower()


def test_agent_catalog_omits_known_bad_visual_candidates():
    removed = {
        "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
        "froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit",
    }

    assert not (removed & AGENT_MODEL_IDS)
