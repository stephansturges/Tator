from services.agent_model_catalog import (
    AGENT_MLX_MODEL_IDS,
    AGENT_MODEL_IDS,
    AGENT_MODEL_OPTIONS,
    AGENT_TRANSFORMERS_MODEL_IDS,
)
from services.qwen_mlx import QWEN_PLATFORM_MLX, QWEN_PLATFORM_TRANSFORMERS


def test_agent_catalog_includes_qwen_inference_family():
    expected = {
        "Jackrong/Qwopus3.6-27B-v2",
        "prithivMLmods/Qwen3.6-35B-A3B-abliterated-MAX",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit",
    }
    removed_non_qwen = {
        "nex-agi/Nex-N2-mini",
        "huihui-ai/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated",
        "mlx-community/gemma-4-31B-it-qat-4bit",
        "vanch007/Huihui-gemma-4-26B-A4B-it-abliterated-mlx-4bit",
    }

    assert expected <= AGENT_MODEL_IDS
    assert not (removed_non_qwen & AGENT_MODEL_IDS)
    assert "Jackrong/Qwopus3.6-27B-v2" in AGENT_TRANSFORMERS_MODEL_IDS
    assert "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit" in AGENT_MLX_MODEL_IDS


def test_agent_catalog_is_inference_only_not_training():
    for entry in AGENT_MODEL_OPTIONS:
        assert entry["training_supported"] is False
        assert entry["training_modes"] == []
        assert entry["agent_model"] is True
        assert entry["runtime_platform"] in {QWEN_PLATFORM_MLX, QWEN_PLATFORM_TRANSFORMERS}


def test_agent_catalog_omits_known_bad_visual_candidates():
    removed = {
        "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
        "froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit",
    }

    assert not (removed & AGENT_MODEL_IDS)
