from pathlib import Path

from PIL import Image

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
    assert result.metadata["training_backend_api"] == "legacy"
