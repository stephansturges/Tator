from types import SimpleNamespace

import pytest
from PIL import Image

import localinferenceapi as api


def _make_qwen_train_dataset(tmp_path, monkeypatch):
    qwen_root = tmp_path / "qwen"
    dataset_root = qwen_root / "demo"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "val").mkdir(parents=True)
    (dataset_root / "train" / "annotations.jsonl").write_text("{}\n", encoding="utf-8")
    (dataset_root / "val" / "annotations.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", tmp_path / "registry")
    monkeypatch.setattr(api, "QWEN_JOB_ROOT", tmp_path / "jobs")
    return dataset_root


def test_qwen_model_registry_exposes_mlx_quantized_models():
    models = api.list_qwen_models()["models"]
    ids = {entry["id"] for entry in models}

    assert "mlx-community/Qwen3-VL-4B-Instruct-4bit" in ids
    assert "mlx-community/Qwen3-VL-32B-Thinking-4bit" in ids
    assert "mlx-community/Qwen3-VL-235B-A22B-Thinking-3bit" in ids
    mlx_entry = next(entry for entry in models if entry["id"] == "mlx-community/Qwen3-VL-4B-Instruct-4bit")
    assert mlx_entry["type"] == "builtin_mlx"
    assert mlx_entry["metadata"]["runtime_platform"] == "mlx_vlm"


def test_qwen_model_registry_exposes_cuda_quantized_and_abliterated_models():
    models = api.list_qwen_models()["models"]
    by_id = {entry["id"]: entry for entry in models}

    fp8_entry = by_id["Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"]
    assert fp8_entry["type"] == "builtin_transformers"
    assert fp8_entry["metadata"]["runtime_platform"] == "transformers"
    assert fp8_entry["metadata"]["quantization_backend"] == "fp8"
    assert fp8_entry["metadata"]["training_supported"] is True
    assert fp8_entry["metadata"]["training_modes"] == ["official_lora", "trl_qlora"]
    assert fp8_entry["metadata"]["training_model_id"] == "Qwen/Qwen3-VL-235B-A22B-Thinking"

    awq_entry = by_id["cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit"]
    assert awq_entry["type"] == "builtin_transformers"
    assert awq_entry["metadata"]["runtime_platform"] == "transformers"
    assert awq_entry["metadata"]["quantization_backend"] == "awq"
    assert awq_entry["metadata"]["training_supported"] is True
    assert awq_entry["metadata"]["training_model_id"] == "Qwen/Qwen3-VL-4B-Instruct"

    abliterated_entry = by_id["huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"]
    assert abliterated_entry["metadata"]["abliterated"] is True
    assert abliterated_entry["metadata"]["training_supported"] is True
    assert abliterated_entry["metadata"]["training_model_id"] == abliterated_entry["metadata"]["model_id"]

    moe_entry = by_id["huihui-ai/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated"]
    assert moe_entry["metadata"]["training_supported"] is True
    assert moe_entry["metadata"]["training_modes"] == ["official_lora", "trl_qlora"]


def test_qwen_model_registry_exposes_abliterated_mlx_models():
    models = api.list_qwen_models()["models"]
    by_id = {entry["id"]: entry for entry in models}
    expected_ids = {
        "EZCon/Huihui-Qwen3-VL-4B-Instruct-abliterated-mlx",
        "EZCon/Huihui-Qwen3-VL-4B-Instruct-abliterated-4bit-mlx",
        "EZCon/Huihui-Qwen3-VL-4B-Instruct-abliterated-4bit-g32-mxfp4-mixed_4_8-mlx",
        "EZCon/Huihui-Qwen3-VL-2B-Thinking-abliterated-8bit-mlx",
        "alexgusevski/Huihui-Qwen3-VL-8B-Instruct-abliterated-q4-mlx",
        "nightmedia/Huihui-Qwen3-VL-32B-Thinking-abliterated-qx65-hi-mlx",
        "introvoyz041/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx-mlx-4Bit",
        "veeceey/Huihui-Qwen3-VL-8B-Instruct-abliterated-mlx-4bit",
        "Goekdeniz-Guelmez/Josiefied-Qwen3-VL-4B-Instruct-abliterated-beta-v1",
    }

    assert expected_ids <= set(by_id)
    for model_id in expected_ids:
        assert by_id[model_id]["type"] == "builtin_mlx"
        assert by_id[model_id]["metadata"]["runtime_platform"] == "mlx_vlm"
        assert by_id[model_id]["metadata"]["abliterated"] is True
        assert by_id[model_id]["metadata"]["training_supported"] is True
        assert by_id[model_id]["metadata"]["training_modes"] == ["official_lora", "trl_qlora"]


def test_qwen_training_config_accepts_moe_transformers_model(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)

    payload = api.QwenTrainRequest(
        dataset_id="demo",
        model_id="huihui-ai/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated",
    )
    config = api._build_qwen_config(payload, "job-moe")

    assert config.runtime_platform == api.QWEN_PLATFORM_TRANSFORMERS
    assert config.model_id == "huihui-ai/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated"


def test_qwen_training_config_accepts_mlx_model(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)

    payload = api.QwenTrainRequest(
        dataset_id="demo",
        model_id="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        training_mode="trl_qlora",
        accumulate_grad_batches=8,
    )
    config = api._build_qwen_config(payload, "job-mlx")

    assert config.runtime_platform == api.QWEN_PLATFORM_MLX
    assert config.model_id == "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    assert config.training_mode == "trl_qlora"
    assert config.accumulate_grad_batches == 1


def test_qwen_mlx_runtime_loads_adapter_path(monkeypatch, tmp_path):
    calls = {}
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    def fake_load(model_id, **kwargs):
        calls["model_id"] = model_id
        calls["kwargs"] = kwargs
        return object(), object()

    monkeypatch.setattr(api, "MLX_VLM_LOAD", fake_load)
    monkeypatch.setattr(api, "MLX_VLM_GENERATE", object())
    monkeypatch.setattr(api, "MLX_VLM_IMPORT_ERROR", None)
    monkeypatch.setattr(api, "MLX_LOAD_CONFIG", None)

    runtime = api._load_qwen_mlx_runtime(
        "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        adapter_path=adapter_dir,
    )

    assert runtime.platform == api.QWEN_PLATFORM_MLX
    assert calls["model_id"] == "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    assert calls["kwargs"]["adapter_path"] == str(adapter_dir)


def test_qwen_mlx_model_id_resolution_maps_hf_to_quantized_default():
    assert (
        api._effective_qwen_model_id_for_platform(
            "Qwen/Qwen3-VL-8B-Thinking",
            api.QWEN_PLATFORM_MLX,
        )
        == "mlx-community/Qwen3-VL-8B-Thinking-4bit"
    )


def test_qwen_mlx_model_id_resolution_keeps_explicit_abliterated_mlx_id():
    model_id = "EZCon/Huihui-Qwen3-VL-2B-Instruct-abliterated-4bit-mlx"
    assert api._effective_qwen_model_id_for_platform(model_id, api.QWEN_PLATFORM_MLX) == model_id


def test_qwen_cuda_catalog_model_forces_transformers_runtime():
    assert (
        api._resolve_qwen_runtime_platform("cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit")
        == api.QWEN_PLATFORM_TRANSFORMERS
    )
    assert (
        api._resolve_qwen_runtime_platform(
            "Qwen/Qwen3-VL-4B-Instruct",
            metadata={"id": "Qwen/Qwen3-VL-4B-Instruct", "runtime_platform": "transformers"},
        )
        == api.QWEN_PLATFORM_TRANSFORMERS
    )


def test_qwen_inference_uses_mlx_runtime(monkeypatch):
    calls = {}

    class DummyProcessor:
        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_kwargs"] = kwargs
            return "formatted prompt"

    def fake_generate(model, processor, prompt, image=None, **kwargs):
        calls["model"] = model
        calls["processor"] = processor
        calls["prompt"] = prompt
        calls["image"] = image
        calls["kwargs"] = kwargs
        return SimpleNamespace(text='[{"bbox_2d":[0,0,10,10],"label":"car"}]')

    monkeypatch.setattr(api, "MLX_VLM_GENERATE", fake_generate)
    runtime = api.QwenRuntime(
        model=object(),
        processor=DummyProcessor(),
        platform=api.QWEN_PLATFORM_MLX,
        model_id="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        config={"model_type": "qwen2_5_vl"},
    )

    img = Image.new("RGB", (32, 24), (0, 0, 0))
    text, width, height = api._run_qwen_inference(
        "Find cars",
        img,
        max_new_tokens=64,
        runtime_override=runtime,
        decode_override={"do_sample": False},
    )

    assert "car" in text
    assert (width, height) == (1000, 1000)
    assert calls["prompt"] == "formatted prompt"
    assert calls["image"] == [img]
    assert calls["kwargs"]["max_tokens"] == 64
    assert calls["kwargs"]["temperature"] == 0.0
    assert calls["messages"][-1]["content"][0] == {"type": "image"}


def test_qwen_mlx_message_normalization_keeps_falsey_image_objects():
    class BoolBomb:
        def __bool__(self):
            raise AssertionError("image object truthiness should not be evaluated")

    image = BoolBomb()
    normalized, images = api._normalize_qwen_messages_for_mlx(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "url": "fallback"},
                    {"type": "text", "text": "Describe this."},
                ],
            }
        ]
    )

    assert images == [image]
    assert normalized[0]["content"][0] == {"type": "image"}


def test_qwen_mlx_cached_runtime_preserves_config(monkeypatch):
    monkeypatch.setattr(api, "MLX_VLM_LOAD", lambda model_id: ("model", "processor"))
    monkeypatch.setattr(api, "MLX_VLM_GENERATE", lambda *args, **kwargs: "")
    monkeypatch.setattr(api, "MLX_LOAD_CONFIG", lambda model_id: {"model_id": model_id})
    monkeypatch.setattr(api, "active_qwen_metadata", {"model_id": "mlx-community/Qwen3-VL-4B-Instruct-4bit"})
    monkeypatch.setattr(api, "active_qwen_model_id", "default")
    api._unload_qwen_runtime()

    first = api._ensure_qwen_mlx_ready("mlx-community/Qwen3-VL-4B-Instruct-4bit")
    second = api._ensure_qwen_mlx_ready("mlx-community/Qwen3-VL-4B-Instruct-4bit")

    assert first.config == {"model_id": "mlx-community/Qwen3-VL-4B-Instruct-4bit"}
    assert second.config == first.config
    api._unload_qwen_runtime()


def test_generate_qwen_text_uses_runtime_aware_chat(monkeypatch):
    calls = []

    def fake_run_qwen_chat(messages, **kwargs):
        calls.append((messages, kwargs))
        return " expanded prompts "

    monkeypatch.setattr(api, "_run_qwen_chat", fake_run_qwen_chat)

    text = api._generate_qwen_text(
        "Expand class prompts",
        max_new_tokens=42,
        use_system_prompt=True,
        system_prompt="System prompt",
    )

    assert text == "expanded prompts"
    assert len(calls) == 1
    messages, kwargs = calls[0]
    assert messages[0] == {"role": "system", "content": [{"type": "text", "text": "System prompt"}]}
    assert messages[1] == {"role": "user", "content": [{"type": "text", "text": "Expand class prompts"}]}
    assert kwargs["max_new_tokens"] == 42
    assert kwargs["decode_override"] == {"do_sample": False}
