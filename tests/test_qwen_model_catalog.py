from types import SimpleNamespace

from services.qwen_model_catalog import (
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


def test_awq_runtime_uses_device_map_for_explicit_cuda_device():
    kwargs = qwen_transformers_load_kwargs(
        "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit",
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


def test_abliterated_training_model_stays_on_abliterated_checkpoint():
    assert (
        resolve_qwen_training_model_id("huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated")
        == "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"
    )
