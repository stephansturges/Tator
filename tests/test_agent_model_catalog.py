from services.agent_model_catalog import (
    AGENT_MLX_MODEL_IDS,
    AGENT_MODEL_IDS,
    AGENT_MODEL_OPTIONS,
    AGENT_TRANSFORMERS_MODEL_IDS,
    agent_model_block_detail,
)
from services.qwen_mlx import QWEN_PLATFORM_MLX, QWEN_PLATFORM_TRANSFORMERS


def test_agent_catalog_includes_requested_inference_families():
    expected = {
        "Jackrong/Qwopus3.6-27B-v2",
        "prithivMLmods/Qwen3.6-35B-A3B-abliterated-MAX",
        "nex-agi/Nex-N2-mini",
        "huihui-ai/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        "mlx-community/gemma-4-31B-it-qat-4bit",
        "vanch007/Huihui-gemma-4-26B-A4B-it-abliterated-mlx-4bit",
    }

    assert expected <= AGENT_MODEL_IDS
    assert "Jackrong/Qwopus3.6-27B-v2" in AGENT_TRANSFORMERS_MODEL_IDS
    assert "mlx-community/gemma-4-31B-it-qat-4bit" in AGENT_MLX_MODEL_IDS


def test_agent_catalog_is_inference_only_not_training():
    for entry in AGENT_MODEL_OPTIONS:
        assert entry["training_supported"] is False
        assert entry["training_modes"] == []
        assert entry["agent_model"] is True
        assert entry["runtime_platform"] in {QWEN_PLATFORM_MLX, QWEN_PLATFORM_TRANSFORMERS}


def test_agent_catalog_blocks_known_bad_visual_candidates():
    gemma_unified = "huihui-ai/Huihui-gemma-4-12B-it-abliterated"

    assert "newer runtime path" in agent_model_block_detail(gemma_unified)
