import json
import os
from pathlib import Path

import pytest
from PIL import Image
import torch

from tools import qwen_training as training


def test_qwen_training_model_class_routes_moe_models(monkeypatch):
    class DenseModel:
        pass

    class MoeModel:
        pass

    monkeypatch.setattr(training, "AutoConfig", None)
    monkeypatch.setattr(training, "Qwen3VLForConditionalGeneration", DenseModel)
    monkeypatch.setattr(training, "Qwen3VLMoeForConditionalGeneration", MoeModel)

    assert (
        training._qwen_training_model_class("Qwen/Qwen3-VL-30B-A3B-Instruct")
        is MoeModel
    )
    assert (
        training._qwen_training_model_class("Qwen/Qwen3-VL-235B-A22B-Thinking")
        is MoeModel
    )
    assert (
        training._qwen_training_model_class("Qwen/Qwen3-VL-4B-Instruct")
        is DenseModel
    )


def test_qwen_mlx_training_flavor_detects_qx_quantized_models():
    assert (
        training._mlx_training_flavor(
            "nightmedia/Huihui-Qwen3-VL-32B-Thinking-abliterated-qx65-hi-mlx"
        )
        == "mlx_qlora"
    )


def test_qwen_training_restores_cuda_visible_devices(monkeypatch, tmp_path):
    observed = {}

    def fake_train(config, progress_cb=None, cancel_cb=None, metrics_cb=None):
        observed["cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return training.QwenTrainingResult(
            config=config,
            checkpoints=[],
            latest_checkpoint=None,
            epochs_ran=0,
        )

    monkeypatch.setattr(training, "_train_official_lora", fake_train)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    config = training.QwenTrainingConfig(
        dataset_root=str(tmp_path / "dataset"),
        result_path=str(tmp_path / "run"),
        devices=" 2 ",
    )

    training.train_qwen_model(config)

    assert observed["cuda_visible"] == "2"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1"


def test_qwen_training_clears_temporary_cuda_visible_devices_on_error(monkeypatch, tmp_path):
    def fake_train(config, progress_cb=None, cancel_cb=None, metrics_cb=None):
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3"
        raise training.TrainingError("boom")

    monkeypatch.setattr(training, "_train_official_lora", fake_train)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    config = training.QwenTrainingConfig(
        dataset_root=str(tmp_path / "dataset"),
        result_path=str(tmp_path / "run"),
        devices="3",
    )

    with pytest.raises(training.TrainingError):
        training.train_qwen_model(config)

    assert os.environ.get("CUDA_VISIBLE_DEVICES") is None
    assert (
        training._mlx_training_flavor(
            "introvoyz041/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx-mlx-4Bit"
        )
        == "mlx_qlora"
    )


def test_qwen_training_image_resolver_rejects_traversal(tmp_path):
    dataset_root = tmp_path / "dataset"
    image_dir = dataset_root / "train" / "images"
    image_dir.mkdir(parents=True)
    safe_image = image_dir / "sample.png"
    safe_image.write_bytes(b"safe")
    outside_image = tmp_path / "outside.png"
    outside_image.write_bytes(b"outside")

    assert training._resolve_image_path(dataset_root, "train", "sample.png") == safe_image.resolve()
    assert training._resolve_image_path(dataset_root, "train", "../outside.png") is None
    assert training._resolve_image_path(dataset_root, "train", str(outside_image)) is None


def test_qwen_training_image_resolver_handles_bad_entries(tmp_path):
    dataset_root = tmp_path / "dataset"
    image_dir = dataset_root / "train" / "images"
    nested_dir = image_dir / "nested"
    nested_dir.mkdir(parents=True)
    safe_image = nested_dir / "sample.png"
    safe_image.write_bytes(b"safe")
    directory_image = image_dir / "directory.png"
    directory_image.mkdir()
    loop = image_dir / "loop.png"
    try:
        loop.symlink_to(loop)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert (
        training._resolve_image_path(dataset_root, "train", "nested\\sample.png")
        == safe_image.resolve()
    )
    assert training._resolve_image_path(dataset_root, "train", "directory.png") is None
    assert training._resolve_image_path(dataset_root, "train", "loop.png") is None


def test_qwen_training_image_resolver_rejects_symlinked_image_root_escape(tmp_path):
    dataset_root = tmp_path / "dataset"
    split_root = dataset_root / "train"
    split_root.mkdir(parents=True)
    outside_images = tmp_path / "outside_images"
    outside_images.mkdir()
    (outside_images / "external.png").write_bytes(b"external")
    try:
        (split_root / "images").symlink_to(outside_images, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert training._resolve_image_path(dataset_root, "train", "external.png") is None


def test_qwen_conversation_dataset_rejects_symlinked_annotation_escape(tmp_path):
    dataset_root = tmp_path / "dataset"
    split_root = dataset_root / "train"
    image_root = split_root / "images"
    image_root.mkdir(parents=True)
    Image.new("RGB", (2, 2), (0, 0, 0)).save(image_root / "sample.png")
    outside_annotations = tmp_path / "outside_annotations.jsonl"
    outside_annotations.write_text(
        '{"image":"sample.png","conversations":[{"from":"human","value":"<image> find car"},{"from":"gpt","value":"{}"}]}\n',
        encoding="utf-8",
    )
    try:
        (split_root / "annotations.jsonl").symlink_to(outside_annotations)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(training.TrainingError, match="qwen_annotations_missing"):
        training.QwenConversationDataset(dataset_root, "train", processor=object())


def test_qwen_conversation_collator_masks_prompt_vision_and_padding_tokens():
    class FakeTokenizer:
        pad_token_id = 0
        image_token_id = 151655

        def convert_tokens_to_ids(self, token):
            return {
                "<|image_pad|>": 151655,
                "<|vision_start|>": 151652,
                "<|vision_end|>": 151653,
            }.get(token, -1)

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            if add_generation_prompt:
                return "prompt"
            return "full"

        def __call__(self, *, text, images, padding, truncation, max_length, return_tensors):
            assert images and not isinstance(images[0], list)
            if all(item == "prompt" for item in text):
                return {
                    "input_ids": torch.tensor([[10, 151655, 11], [12, 151655, 13]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                }
            return {
                "input_ids": torch.tensor(
                    [
                        [10, 151655, 11, 21, 22, 0],
                        [12, 151655, 13, 23, 0, 0],
                    ]
                ),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 0, 0],
                    ]
                ),
            }

    image_a = Image.new("RGB", (2, 2), (0, 0, 0))
    image_b = Image.new("RGB", (2, 2), (255, 255, 255))
    batch = [
        {
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": image_a}]},
                {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
            ],
            "images": [image_a],
        },
        {
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": image_b}]},
                {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
            ],
            "images": [image_b],
        },
    ]

    inputs = training.QwenConversationCollator(FakeProcessor(), max_length=None)(batch)

    assert inputs["labels"].tolist() == [
        [-100, -100, -100, 21, 22, -100],
        [-100, -100, -100, 23, -100, -100],
    ]


def test_qwen_conversation_messages_keep_one_image_marker():
    image = Image.new("RGB", (2, 2), (0, 0, 0))

    missing_marker = training._conversation_to_messages(
        [
            {"from": "human", "value": "find cars"},
            {"from": "gpt", "value": "{}"},
        ],
        image,
    )
    repeated_marker = training._conversation_to_messages(
        [
            {"from": "human", "value": "<image> find cars <image>"},
            {"from": "gpt", "value": "{}"},
        ],
        image,
    )

    for messages in (missing_marker, repeated_marker):
        image_parts = [
            part
            for message in messages
            for part in message.get("content", [])
            if part.get("type") == "image"
        ]
        assert len(image_parts) == 1
        assert image_parts[0]["image"] is image


def test_qwen_mlx_lora_training_uses_mlx_backend(monkeypatch, tmp_path):
    dataset_root = tmp_path / "dataset"
    for split in ("train", "val"):
        image_dir = dataset_root / split / "images"
        image_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8), (0, 0, 0)).save(image_dir / "sample.png")
        (dataset_root / split / "annotations.jsonl").write_text(
            '{"image":"sample.png","conversations":[{"from":"human","value":"<image> find car"},{"from":"gpt","value":"{\\"detections\\":[]}"}]}\n',
            encoding="utf-8",
        )

    calls = {}

    class FakeConfig:
        model_type = "qwen3_vl"

    class FakeModel:
        config = FakeConfig()
        language_model = object()

    class FakeVisionDataset:
        def __init__(self, raw, config, processor):
            self.raw = raw
            self.config = config
            self.processor = processor

    class FakeTrainingArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOptimizers:
        class Adam:
            def __init__(self, learning_rate):
                self.learning_rate = learning_rate

    def fake_train(**kwargs):
        calls["train"] = kwargs

    monkeypatch.setattr(
        training,
        "_load_mlx_training_backend",
        lambda: {
            "optimizers": FakeOptimizers,
            "VisionDataset": FakeVisionDataset,
            "TrainingArgs": FakeTrainingArgs,
            "train": fake_train,
            "find_all_linear_names": lambda language_model: ["q_proj"],
            "get_peft_model": lambda model, modules, **kwargs: model,
            "not_supported_for_training": set(),
            "print_trainable_parameters": lambda model: None,
            "load": lambda model_id, **kwargs: (FakeModel(), object()),
        },
    )

    config = training.QwenTrainingConfig(
        dataset_root=str(dataset_root),
        result_path=str(tmp_path / "run"),
        model_id="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        runtime_platform=training.QWEN_PLATFORM_MLX,
        training_mode="trl_qlora",
        max_epochs=1,
        batch_size=1,
    )
    result = training.train_qwen_model(config)

    assert result.metadata["runtime_platform"] == training.QWEN_PLATFORM_MLX
    assert result.metadata["training_flavor"] == "mlx_qlora"
    assert Path(result.latest_checkpoint).name == "latest"
    assert calls["train"]["args"].kwargs["adapter_file"].endswith("adapters.safetensors")
    assert calls["train"]["train_on_completions"] is True
    adapter_config = json.loads((Path(result.latest_checkpoint) / "adapter_config.json").read_text())
    assert adapter_config["rank"] == 8
    assert adapter_config["alpha"] == 16.0


def test_qwen_mlx_legacy_backend_scales_lora_alpha(monkeypatch, tmp_path):
    dataset_root = tmp_path / "dataset"
    for split in ("train", "val"):
        image_dir = dataset_root / split / "images"
        image_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8), (0, 0, 0)).save(image_dir / "sample.png")
        (dataset_root / split / "annotations.jsonl").write_text(
            '{"image":"sample.png","conversations":[{"from":"human","value":"<image> find car"},{"from":"gpt","value":"{\\"detections\\":[]}"}]}\n',
            encoding="utf-8",
        )

    calls = {}

    class FakeConfig:
        model_type = "qwen3_vl"

    class FakeModel:
        config = FakeConfig()
        language_model = object()

        def train(self):
            calls["train_mode"] = True

        def trainable_parameters(self):
            return {}

    class FakeVisionDataset:
        def __init__(self, raw, config, processor, image_processor=None):
            self.raw = raw
            self.config = config
            self.processor = processor
            self.image_processor = image_processor

        def __len__(self):
            return len(self.raw)

        def __getitem__(self, idx):
            return self.raw[idx]

    class FakeOptimizers:
        class Adam:
            state = {}

            def __init__(self, learning_rate):
                self.learning_rate = learning_rate

    class FakeTrainer:
        def __init__(self, model, optimizer, train_on_completions=False):
            calls["train_on_completions"] = train_on_completions

        def train_step(self, batch):
            calls["batch"] = batch

            class FakeLoss:
                def item(self):
                    return 0.25

            return FakeLoss()

    class FakeMx:
        @staticmethod
        def eval(*args):
            calls["mx_eval"] = True

    def fake_get_peft_model(model, modules, **kwargs):
        calls["peft"] = kwargs
        return model

    def fake_save_adapter(model, adapter_file):
        calls["adapter_file"] = adapter_file

    monkeypatch.setattr(
        training,
        "_load_mlx_training_backend",
        lambda: {
            "api": "legacy",
            "mx": FakeMx,
            "optimizers": FakeOptimizers,
            "VisionDataset": FakeVisionDataset,
            "TrainingArgs": lambda **kwargs: kwargs,
            "Trainer": FakeTrainer,
            "save_adapter": fake_save_adapter,
            "find_all_linear_names": lambda language_model: ["q_proj"],
            "get_peft_model": fake_get_peft_model,
            "not_supported_for_training": set(),
            "print_trainable_parameters": lambda model: None,
            "load": lambda model_id, **kwargs: (FakeModel(), object()),
            "load_image_processor": lambda model_id: object(),
        },
    )

    config = training.QwenTrainingConfig(
        dataset_root=str(dataset_root),
        result_path=str(tmp_path / "run"),
        model_id="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        runtime_platform=training.QWEN_PLATFORM_MLX,
        max_epochs=1,
        batch_size=1,
        lora_rank=8,
        lora_alpha=16,
        log_every_n_steps=1,
    )
    result = training.train_qwen_model(config)

    assert calls["peft"]["alpha"] == 2.0
    assert calls["train_on_completions"] is True
    assert calls["mx_eval"] is True
    assert calls["adapter_file"].endswith("adapters.safetensors")
    adapter_config = json.loads((Path(calls["adapter_file"]).parent / "adapter_config.json").read_text())
    assert adapter_config == {"rank": 8, "alpha": 2.0, "dropout": 0.05}
    assert result.metadata["training_backend_api"] == "legacy"
