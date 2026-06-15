import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import localinferenceapi as api


class _ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, **_kwargs):
        self._target = target
        self._args = tuple(args or ())
        self._kwargs = dict(kwargs or {})

    def start(self):
        if self._target:
            self._target(*self._args, **self._kwargs)

    def join(self, *_args, **_kwargs):
        return None


def _fake_torch(*, cuda: bool = False, cuda_count: int = 0):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: cuda_count,
        )
    )


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


def _make_qwen_train_only_dataset(tmp_path, monkeypatch):
    qwen_root = tmp_path / "qwen"
    dataset_root = qwen_root / "demo"
    image_dir = dataset_root / "train" / "images"
    image_dir.mkdir(parents=True)
    for idx in range(4):
        name = f"sample_{idx}.png"
        Image.new("RGB", (8, 8), (idx, idx, idx)).save(image_dir / name)
    (dataset_root / "train" / "annotations.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "image": f"sample_{idx}.png",
                    "conversations": [
                        {"from": "human", "value": "<image> find car"},
                        {"from": "gpt", "value": "{\"detections\":[]}"},
                    ],
                },
                separators=(",", ":"),
            )
            for idx in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
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
    heretic_id = "Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit"
    assert heretic_id in by_id
    heretic_entry = by_id[heretic_id]
    assert heretic_entry["type"] == "builtin_mlx"
    assert heretic_entry["metadata"]["runtime_platform"] == "mlx_vlm"
    assert heretic_entry["metadata"]["abliterated"] is True
    assert heretic_entry["metadata"]["source"] == "Youssofal"
    assert heretic_entry["metadata"]["size"] == "35B-A3B"
    assert heretic_entry["metadata"]["variant"] == "Heretic"
    assert heretic_entry["metadata"]["vision_inference_supported"] is False
    assert heretic_entry["metadata"]["training_supported"] is False
    assert heretic_entry["metadata"]["training_modes"] == []
    assert "generated invalid text" in heretic_entry["metadata"]["compatibility_note"]
    assert "vignette benchmark" in heretic_entry["metadata"]["compatibility_note"]
    assert "Heretic" in heretic_entry["label"]
    qwen36_id = "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit"
    assert qwen36_id in by_id
    qwen36_entry = by_id[qwen36_id]
    assert qwen36_entry["type"] == "builtin_mlx"
    assert qwen36_entry["metadata"]["runtime_platform"] == "mlx_vlm"
    assert qwen36_entry["metadata"]["abliterated"] is True
    assert qwen36_entry["metadata"]["source"] == "vanch007"
    assert qwen36_entry["metadata"]["size"] == "35B-A3B"
    assert qwen36_entry["metadata"]["variant"] == "Abliterated"
    assert qwen36_entry["metadata"]["vision_inference_supported"] is True
    assert qwen36_entry["metadata"]["training_supported"] is False
    assert qwen36_entry["metadata"]["training_modes"] == []
    assert "smoke tests passed" in qwen36_entry["metadata"]["compatibility_note"]
    for model_id in (
        "introvoyz041/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx-mlx-4Bit",
        "introvoyz041/Huihui-Qwen3-VL-32B-Thinking-abliterated-qx65-hi-mlx-mlx-4Bit",
    ):
        assert model_id in by_id
        assert by_id[model_id]["type"] == "builtin_mlx"
        assert by_id[model_id]["metadata"]["abliterated"] is True
        assert by_id[model_id]["metadata"]["vision_inference_supported"] is False
        assert by_id[model_id]["metadata"]["training_supported"] is False
        assert by_id[model_id]["metadata"]["training_modes"] == []


def test_qwen_model_registry_exposes_inference_only_agent_models():
    models = api.list_qwen_models()["models"]
    by_id = {entry["id"]: entry for entry in models}

    expected = {
        "Jackrong/Qwopus3.6-27B-v2": "builtin_agent_transformers",
        "prithivMLmods/Qwen3.6-35B-A3B-abliterated-MAX": "builtin_agent_transformers",
        "nex-agi/Nex-N2-mini": "builtin_agent_transformers",
        "huihui-ai/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated": "builtin_agent_transformers",
        "mlx-community/Qwen3.6-35B-A3B-4bit": "builtin_agent_mlx",
        "mlx-community/gemma-4-31B-it-qat-4bit": "builtin_agent_mlx",
        "vanch007/Huihui-gemma-4-26B-A4B-it-abliterated-mlx-4bit": "builtin_agent_mlx",
    }

    for model_id, entry_type in expected.items():
        entry = by_id[model_id]
        assert entry["type"] == entry_type
        assert entry["metadata"]["agent_model"] is True
        assert entry["metadata"]["training_supported"] is False
        assert entry["metadata"]["training_modes"] == []
        assert entry["metadata"]["vision_inference_supported"] is True


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


def test_qwen_training_config_resolves_custom_quantized_abliterated_cuda_base(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)

    payload = api.QwenTrainRequest(
        dataset_id=" demo ",
        model_id=" custom/Huihui-Qwen3-VL-4B-Instruct-abliterated-AWQ-4bit ",
        training_mode="QLoRA",
        lora_target_modules="q_proj,k_proj",
    )
    config = api._build_qwen_config(payload, "job-custom-abliterated")

    assert config.runtime_platform == api.QWEN_PLATFORM_TRANSFORMERS
    assert config.requested_model_id == "custom/Huihui-Qwen3-VL-4B-Instruct-abliterated-AWQ-4bit"
    assert config.model_id == "custom/Huihui-Qwen3-VL-4B-Instruct-abliterated"
    assert config.training_mode == "trl_qlora"
    assert config.lora_target_modules == ["q_proj", "k_proj"]
    assert config.requested_model_metadata["abliterated"] is True
    assert config.requested_model_metadata["quantization_backend"] == "awq"


def test_qwen_training_config_clamps_direct_api_numeric_knobs(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)

    payload = api.QwenTrainRequest(
        dataset_id="demo",
        batch_size=0,
        max_epochs=0,
        lr=-1.0,
        accumulate_grad_batches=0,
        warmup_steps=-3,
        num_workers=-1,
        lora_rank=0,
        lora_alpha=-5,
        lora_dropout=3.5,
        log_every_n_steps=0,
        min_pixels=100000,
        max_pixels=10,
        max_length=8,
        seed=-1,
        train_limit=-1,
        val_limit=0,
    )
    config = api._build_qwen_config(payload, "job-clamp")

    assert config.batch_size == 1
    assert config.max_epochs == 3
    assert config.lr == 2e-4
    assert config.accumulate_grad_batches == 8
    assert config.warmup_steps == 50
    assert config.num_workers == 0
    assert config.lora_rank == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 1.0
    assert config.log_every_n_steps == 10
    assert config.min_pixels == 100000
    assert config.max_pixels >= config.min_pixels
    assert config.max_length == 2048
    assert config.seed == 1337
    assert config.train_limit is None
    assert config.val_limit is None


def test_qwen_training_config_rejects_cuda_devices_when_cuda_absent(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "torch", _fake_torch(cuda=False, cuda_count=0))

    payload = api.QwenTrainRequest(
        dataset_id="demo",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        devices="0",
    )

    with pytest.raises(api.HTTPException) as excinfo:
        api._build_qwen_config(payload, "job-no-cuda")

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen_cuda_devices_unavailable"


def test_qwen_training_config_normalizes_cuda_devices(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "torch", _fake_torch(cuda=True, cuda_count=2))

    payload = api.QwenTrainRequest(
        dataset_id="demo",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        devices=" 0, 01 ",
    )
    config = api._build_qwen_config(payload, "job-cuda-devices")

    assert config.devices == "0,1"


def test_qwen_training_config_rejects_out_of_range_cuda_devices(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "torch", _fake_torch(cuda=True, cuda_count=1))

    payload = api.QwenTrainRequest(
        dataset_id="demo",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        devices="1",
    )

    with pytest.raises(api.HTTPException) as excinfo:
        api._build_qwen_config(payload, "job-bad-cuda-devices")

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen_invalid_devices:available=0-0"


def test_qwen_training_config_random_split_materializes_qwen_split(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_only_dataset(tmp_path, monkeypatch)
    logs = []
    payload = api.QwenTrainRequest(
        dataset_id="demo",
        random_split=True,
        val_percent=0.5,
        split_seed=123,
    )

    config = api._build_qwen_config(payload, "job-random-split", logs)
    split_root = Path(config.dataset_root)

    assert split_root == (api.QWEN_JOB_ROOT / "splits" / "job-random-split").resolve()
    assert (split_root / "train" / "annotations.jsonl").exists()
    assert (split_root / "val" / "annotations.jsonl").exists()
    train_lines = (split_root / "train" / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
    val_lines = (split_root / "val" / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 2
    assert len(val_lines) == 2
    for split_name, lines in (("train", train_lines), ("val", val_lines)):
        for line in lines:
            payload_out = json.loads(line)
            assert ".." not in Path(payload_out["image"]).parts
            assert (split_root / split_name / "images" / payload_out["image"]).exists()
    metadata = json.loads((split_root / api.QWEN_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["random_split"] is True
    assert metadata["train_count"] == 2
    assert metadata["val_count"] == 2
    assert any("Qwen split:" in entry for entry in logs)


def test_qwen_training_split_text_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
):
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    split_root = tmp_path / "split"
    annotations_path = split_root / "train" / "annotations.jsonl"
    annotations_path.parent.mkdir(parents=True)
    tmp_path_link = annotations_path.with_suffix(
        f"{annotations_path.suffix}.{FixedUUID.hex}.tmp"
    )
    outside_tmp = tmp_path / "outside_tmp.jsonl"
    outside_final = tmp_path / "outside_final.jsonl"
    outside_tmp.write_text("external tmp\n", encoding="utf-8")
    outside_final.write_text("external final\n", encoding="utf-8")
    try:
        tmp_path_link.symlink_to(outside_tmp)
        annotations_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: FixedUUID())

    api._qwen_training_write_text_within_root(
        annotations_path,
        split_root,
        '{"image":"sample.png"}\n',
    )

    assert not tmp_path_link.exists()
    assert not annotations_path.is_symlink()
    assert annotations_path.read_text(encoding="utf-8") == '{"image":"sample.png"}\n'
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp\n"
    assert outside_final.read_text(encoding="utf-8") == "external final\n"


def test_qwen_training_random_split_copies_when_hardlink_unavailable(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_only_dataset(tmp_path, monkeypatch)

    def fail_link(*_args, **_kwargs):
        raise OSError("hardlink unavailable")

    monkeypatch.setattr(api.os, "link", fail_link)
    payload = api.QwenTrainRequest(dataset_id="demo", random_split=True, val_percent=0.5)

    config = api._build_qwen_config(payload, "job-copy-split", [])
    split_root = Path(config.dataset_root)
    images = list((split_root / "train" / "images").iterdir()) + list(
        (split_root / "val" / "images").iterdir()
    )

    assert images
    assert all(path.is_file() for path in images)
    assert all(not path.is_symlink() for path in images)


def test_qwen_training_random_split_rejects_symlinked_split_root(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_only_dataset(tmp_path, monkeypatch)
    api.QWEN_JOB_ROOT.mkdir(parents=True, exist_ok=True)
    outside_split_root = tmp_path / "outside_split_root"
    outside_split_root.mkdir()
    try:
        (api.QWEN_JOB_ROOT / "splits").symlink_to(outside_split_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    payload = api.QwenTrainRequest(dataset_id="demo", random_split=True, val_percent=0.5)

    with pytest.raises(api.HTTPException) as excinfo:
        api._build_qwen_config(payload, "job-split-link", [])

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen_split_path_invalid"
    assert list(outside_split_root.iterdir()) == []


def test_qwen_training_config_rejects_existing_run_name(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)
    (api.QWEN_JOB_ROOT / "runs" / "existing").mkdir(parents=True)
    payload = api.QwenTrainRequest(dataset_id="demo", run_name="existing")

    with pytest.raises(api.HTTPException) as excinfo:
        api._build_qwen_config(payload, "job-existing")

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail == "run_name_exists"


def test_qwen_training_config_rejects_symlinked_annotation_escape(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    dataset_root = _make_qwen_train_dataset(tmp_path, monkeypatch)
    outside_annotations = tmp_path / "outside_annotations.jsonl"
    outside_annotations.write_text("{}\n", encoding="utf-8")
    train_annotations = dataset_root / "train" / "annotations.jsonl"
    train_annotations.unlink()
    try:
        train_annotations.symlink_to(outside_annotations)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    payload = api.QwenTrainRequest(dataset_id="demo", run_name="annotation-link")

    with pytest.raises(api.HTTPException) as excinfo:
        api._build_qwen_config(payload, "job-annotation-link")

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen_train_split_missing"


def test_qwen_training_job_rejects_active_duplicate_run_name(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "_start_qwen_training_worker", lambda job, config: None)
    with api.QWEN_TRAINING_JOBS_LOCK:
        api.QWEN_TRAINING_JOBS.clear()
    payload = api.QwenTrainRequest(dataset_id="demo", run_name="duplicate")

    try:
        first = api.create_qwen_training_job(payload)
        assert first["job_id"]
        with pytest.raises(api.HTTPException) as excinfo:
            api.create_qwen_training_job(payload)
        assert excinfo.value.status_code == 409
        assert excinfo.value.detail == "run_name_exists"
    finally:
        with api.QWEN_TRAINING_JOBS_LOCK:
            api.QWEN_TRAINING_JOBS.clear()


def test_qwen_training_job_cleans_random_split_when_run_name_exists(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_only_dataset(tmp_path, monkeypatch)
    (api.QWEN_JOB_ROOT / "runs" / "existing").mkdir(parents=True)
    payload = api.QwenTrainRequest(
        dataset_id="demo",
        run_name="existing",
        random_split=True,
        val_percent=0.5,
    )

    with pytest.raises(api.HTTPException) as excinfo:
        api.create_qwen_training_job(payload)

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail == "run_name_exists"
    split_root = api.QWEN_JOB_ROOT / "splits"
    assert not split_root.exists() or list(split_root.iterdir()) == []


def test_qwen_training_job_cleans_random_split_for_active_duplicate(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_only_dataset(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "_start_qwen_training_worker", lambda job, config: None)
    with api.QWEN_TRAINING_JOBS_LOCK:
        api.QWEN_TRAINING_JOBS.clear()
    payload = api.QwenTrainRequest(
        dataset_id="demo",
        run_name="duplicate",
        random_split=True,
        val_percent=0.5,
    )

    try:
        first = api.create_qwen_training_job(payload)
        assert first["job_id"]
        with pytest.raises(api.HTTPException) as excinfo:
            api.create_qwen_training_job(payload)
        assert excinfo.value.status_code == 409
        assert excinfo.value.detail == "run_name_exists"
        splits = sorted((api.QWEN_JOB_ROOT / "splits").iterdir())
        assert [path.name for path in splits] == [first["job_id"]]
    finally:
        with api.QWEN_TRAINING_JOBS_LOCK:
            api.QWEN_TRAINING_JOBS.clear()


def test_qwen_train_request_drops_nonfinite_numeric_controls():
    payload = api.QwenTrainRequest(
        dataset_id="demo",
        training_mode="official",
        batch_size=float("nan"),
        lr=float("inf"),
        lora_dropout=float("-inf"),
        max_epochs=2,
    )

    assert payload.training_mode == "official_lora"
    assert payload.batch_size is None
    assert payload.lr is None
    assert payload.lora_dropout is None
    assert payload.max_epochs == 2


def test_qwen_train_request_treats_empty_numeric_controls_as_defaults():
    payload = api.QwenTrainRequest(
        dataset_id="demo",
        batch_size="",
        lr=" ",
        train_limit="",
    )

    assert payload.batch_size is None
    assert payload.lr is None
    assert payload.train_limit is None


def test_qwen_training_metadata_preserves_requested_abliterated_source(tmp_path):
    if api.QwenTrainingConfig is None or api.QwenTrainingResult is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    result_path = tmp_path / "run"
    config = api.QwenTrainingConfig(
        dataset_root=str(tmp_path / "dataset"),
        result_path=str(result_path),
        model_id="custom/Huihui-Qwen3-VL-4B-Instruct-abliterated",
        requested_model_id="custom/Huihui-Qwen3-VL-4B-Instruct-abliterated-AWQ-4bit",
        requested_model_metadata={
            "abliterated": True,
            "quantization": "AWQ 4-bit",
            "quantization_backend": "awq",
            "training_note": "Training starts from the resolved unquantized abliterated checkpoint.",
        },
        runtime_platform=api.QWEN_PLATFORM_TRANSFORMERS,
    )
    result = api.QwenTrainingResult(
        config=config,
        checkpoints=[str(result_path / "latest")],
        latest_checkpoint=str(result_path / "latest"),
        epochs_ran=1,
    )

    metadata = api._persist_qwen_run_metadata(result_path, config, result)

    assert metadata["requested_model_id"] == "custom/Huihui-Qwen3-VL-4B-Instruct-abliterated-AWQ-4bit"
    assert metadata["training_model_id"] == "custom/Huihui-Qwen3-VL-4B-Instruct-abliterated"
    assert metadata["abliterated"] is True
    assert metadata["quantization_backend"] == "awq"


def test_qwen_training_late_cancel_skips_metadata_publish(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None or api.QwenTrainingResult is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    result_path = tmp_path / "run"
    config = api.QwenTrainingConfig(
        dataset_root=str(tmp_path / "dataset"),
        result_path=str(result_path),
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        run_name="late-cancel",
    )
    job = api.QwenTrainingJob(job_id="qwen_late_cancel", config={})
    persist_calls = {"count": 0}

    def fake_train_qwen_model(_config, **_kwargs):
        job.cancel_event.set()
        return api.QwenTrainingResult(
            config=config,
            checkpoints=[str(result_path / "latest")],
            latest_checkpoint=str(result_path / "latest"),
            epochs_ran=1,
        )

    def fail_persist(*_args, **_kwargs):
        persist_calls["count"] += 1
        raise AssertionError("cancelled Qwen job should not publish metadata")

    monkeypatch.setattr(api.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(api, "_prepare_for_qwen_training", lambda: None)
    monkeypatch.setattr(api, "_finalize_qwen_training_environment", lambda: None)
    monkeypatch.setattr(api, "train_qwen_model", fake_train_qwen_model)
    monkeypatch.setattr(api, "_persist_qwen_run_metadata", fail_persist)

    api._start_qwen_training_worker(job, config)

    assert job.status == "cancelled"
    assert job.result is None
    assert persist_calls["count"] == 0
    assert not (result_path / api.QWEN_METADATA_FILENAME).exists()


def test_qwen_training_worker_fails_when_metadata_publish_fails(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None or api.QwenTrainingResult is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    result_path = tmp_path / "run"
    config = api.QwenTrainingConfig(
        dataset_root=str(tmp_path / "dataset"),
        result_path=str(result_path),
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        run_name="metadata-failure",
    )
    job = api.QwenTrainingJob(job_id="qwen_metadata_failure", config={})

    def fake_train_qwen_model(_config, **_kwargs):
        return api.QwenTrainingResult(
            config=config,
            checkpoints=[str(result_path / "latest")],
            latest_checkpoint=str(result_path / "latest"),
            epochs_ran=1,
        )

    monkeypatch.setattr(api.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(api, "_prepare_for_qwen_training", lambda: None)
    monkeypatch.setattr(api, "_finalize_qwen_training_environment", lambda: None)
    monkeypatch.setattr(api, "train_qwen_model", fake_train_qwen_model)
    monkeypatch.setattr(api, "_write_qwen_run_metadata_file", lambda *_args, **_kwargs: False)

    api._start_qwen_training_worker(job, config)

    assert job.status == "failed"
    assert job.error == "qwen_run_metadata_write_failed"
    assert job.result is None


def test_qwen_training_config_rejects_symlinked_runs_root(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)
    api.QWEN_JOB_ROOT.mkdir(parents=True, exist_ok=True)
    outside_runs = tmp_path / "outside_runs"
    outside_runs.mkdir()
    try:
        (api.QWEN_JOB_ROOT / "runs").symlink_to(outside_runs, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    payload = api.QwenTrainRequest(dataset_id="demo", run_name="escaped")

    with pytest.raises(api.HTTPException) as excinfo:
        api._build_qwen_config(payload, "job-runs-link")

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen_run_path_invalid"
    assert list(outside_runs.iterdir()) == []


def test_qwen_training_config_rejects_symlinked_job_root_parent_for_runs(
    tmp_path, monkeypatch
):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)
    outside_runs = tmp_path / "outside_runs"
    outside_runs.mkdir()
    link_parent = tmp_path / "linked_parent"
    try:
        link_parent.symlink_to(outside_runs, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "QWEN_JOB_ROOT", link_parent / "jobs")

    payload = api.QwenTrainRequest(dataset_id="demo", run_name="escaped")

    with pytest.raises(api.HTTPException) as excinfo:
        api._build_qwen_config(payload, "job-root-parent-link")

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen_run_path_invalid"
    assert list(outside_runs.iterdir()) == []


def test_qwen_training_metadata_replaces_symlinked_metadata_file(tmp_path):
    if api.QwenTrainingConfig is None or api.QwenTrainingResult is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    result_path = tmp_path / "run"
    result_path.mkdir()
    outside_meta = tmp_path / "outside_metadata.json"
    outside_meta.write_text("external", encoding="utf-8")
    metadata_path = result_path / api.QWEN_METADATA_FILENAME
    try:
        metadata_path.symlink_to(outside_meta)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    config = api.QwenTrainingConfig(
        dataset_root=str(tmp_path / "dataset"),
        result_path=str(result_path),
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        run_name="safe-meta",
    )
    result = api.QwenTrainingResult(
        config=config,
        checkpoints=[str(result_path / "latest")],
        latest_checkpoint=str(result_path / "latest"),
        epochs_ran=1,
    )

    metadata = api._persist_qwen_run_metadata(result_path, config, result)

    assert outside_meta.read_text(encoding="utf-8") == "external"
    assert not metadata_path.is_symlink()
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert persisted["id"] == metadata["id"] == "safe-meta"


def test_qwen_training_metadata_rejects_symlinked_result_dir_without_target_write(tmp_path):
    if api.QwenTrainingConfig is None or api.QwenTrainingResult is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    outside_run = tmp_path / "outside_run"
    outside_run.mkdir()
    result_path = tmp_path / "linked_run"
    try:
        result_path.symlink_to(outside_run, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    config = api.QwenTrainingConfig(
        dataset_root=str(tmp_path / "dataset"),
        result_path=str(result_path),
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        run_name="linked-run",
    )
    result = api.QwenTrainingResult(
        config=config,
        checkpoints=[str(result_path / "latest")],
        latest_checkpoint=str(result_path / "latest"),
        epochs_ran=1,
    )

    with pytest.raises(api.QwenTrainingError, match="qwen_run_metadata_write_failed"):
        api._persist_qwen_run_metadata(result_path, config, result)

    assert not (outside_run / api.QWEN_METADATA_FILENAME).exists()


def test_qwen_training_metadata_rejects_symlinked_result_parent_without_target_write(tmp_path):
    if api.QwenTrainingConfig is None or api.QwenTrainingResult is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    outside_run = tmp_path / "outside_run"
    outside_run.mkdir()
    link_parent = tmp_path / "linked_parent"
    try:
        link_parent.symlink_to(outside_run, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    result_path = link_parent / "run"
    config = api.QwenTrainingConfig(
        dataset_root=str(tmp_path / "dataset"),
        result_path=str(result_path),
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        run_name="linked-parent-run",
    )
    result = api.QwenTrainingResult(
        config=config,
        checkpoints=[str(result_path / "latest")],
        latest_checkpoint=str(result_path / "latest"),
        epochs_ran=1,
    )

    with pytest.raises(api.QwenTrainingError, match="qwen_run_metadata_write_failed"):
        api._persist_qwen_run_metadata(result_path, config, result)

    assert list(outside_run.iterdir()) == []


def test_qwen_training_metadata_fails_when_metadata_path_is_directory(tmp_path):
    if api.QwenTrainingConfig is None or api.QwenTrainingResult is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    result_path = tmp_path / "run"
    result_path.mkdir()
    (result_path / api.QWEN_METADATA_FILENAME).mkdir()
    config = api.QwenTrainingConfig(
        dataset_root=str(tmp_path / "dataset"),
        result_path=str(result_path),
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        run_name="metadata-dir",
    )
    result = api.QwenTrainingResult(
        config=config,
        checkpoints=[str(result_path / "latest")],
        latest_checkpoint=str(result_path / "latest"),
        epochs_ran=1,
    )

    with pytest.raises(api.QwenTrainingError, match="qwen_run_metadata_write_failed"):
        api._persist_qwen_run_metadata(result_path, config, result)

    assert (result_path / api.QWEN_METADATA_FILENAME).is_dir()


def test_qwen_model_registry_skips_transformers_runs_without_adapter_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "QWEN_JOB_ROOT", tmp_path / "qwen_jobs")
    broken_latest = api.QWEN_JOB_ROOT / "runs" / "broken" / "latest"
    broken_latest.mkdir(parents=True)
    (api.QWEN_JOB_ROOT / "runs" / "broken" / api.QWEN_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "id": "broken",
                "label": "Broken",
                "model_id": "Qwen/Qwen3-VL-4B-Instruct",
                "runtime_platform": api.QWEN_PLATFORM_TRANSFORMERS,
                "latest_checkpoint": str(broken_latest),
            }
        ),
        encoding="utf-8",
    )
    good_latest = api.QWEN_JOB_ROOT / "runs" / "good" / "latest"
    good_latest.mkdir(parents=True)
    (good_latest / "adapter_config.json").write_text("{}", encoding="utf-8")
    (good_latest / "adapter_model.safetensors").write_bytes(b"adapter")
    (api.QWEN_JOB_ROOT / "runs" / "good" / api.QWEN_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "id": "good",
                "label": "Good",
                "model_id": "Qwen/Qwen3-VL-4B-Instruct",
                "runtime_platform": api.QWEN_PLATFORM_TRANSFORMERS,
                "latest_checkpoint": str(good_latest),
            }
        ),
        encoding="utf-8",
    )

    ids = {entry["id"] for entry in api._list_qwen_model_entries()}

    assert "good" in ids
    assert "broken" not in ids


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


def test_qwen_training_config_ignores_cuda_devices_for_mlx_model(tmp_path, monkeypatch):
    if api.QwenTrainingConfig is None:
        pytest.skip("Qwen training dependencies are not importable in this environment")

    _make_qwen_train_dataset(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "torch", _fake_torch(cuda=False, cuda_count=0))
    logs = []

    payload = api.QwenTrainRequest(
        dataset_id="demo",
        model_id="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        devices="0",
    )
    config = api._build_qwen_config(payload, "job-mlx-devices", logs)

    assert config.runtime_platform == api.QWEN_PLATFORM_MLX
    assert config.devices is None
    assert "Ignoring CUDA device selection for MLX Qwen training" in logs


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
    monkeypatch.setattr(
        api, "_qwen_mlx_remote_checkpoint_incompatibility_detail", lambda _model_id: None
    )

    runtime = api._load_qwen_mlx_runtime(
        "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        adapter_path=adapter_dir,
    )

    assert runtime.platform == api.QWEN_PLATFORM_MLX
    assert calls["model_id"] == "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    assert calls["kwargs"]["adapter_path"] == str(adapter_dir)


def test_qwen_mlx_runtime_rejects_language_only_repack(monkeypatch):
    bad_model_id = (
        "introvoyz041/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx-mlx-4Bit"
    )

    def fail_load(*_args, **_kwargs):
        raise AssertionError("incompatible MLX repack should be rejected before load")

    monkeypatch.setattr(api, "MLX_VLM_LOAD", fail_load)
    monkeypatch.setattr(api, "MLX_VLM_GENERATE", object())
    monkeypatch.setattr(api, "MLX_VLM_IMPORT_ERROR", None)

    with pytest.raises(api.HTTPException) as excinfo:
        api._load_qwen_mlx_runtime(bad_model_id)

    assert excinfo.value.status_code == 400
    assert "qwen_mlx_incompatible_checkpoint" in str(excinfo.value.detail)
    assert "no vision_tower weights" in str(excinfo.value.detail)


def test_qwen_settings_excludes_language_only_mlx_repack_options():
    bad_model_id = (
        "introvoyz041/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx-mlx-4Bit"
    )

    settings = api.qwen_settings()

    assert bad_model_id not in {entry["id"] for entry in settings.mlx_models}


def test_qwen_settings_excludes_blocked_heretic_candidate():
    model_id = "Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit"

    settings = api.qwen_settings()

    assert model_id not in {entry["id"] for entry in settings.mlx_models}


def test_qwen_settings_includes_working_qwen36_candidate():
    model_id = "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit"

    settings = api.qwen_settings()
    by_id = {entry["id"]: entry for entry in settings.mlx_models}

    assert model_id in by_id
    assert by_id[model_id]["vision_inference_supported"] is True
    assert by_id[model_id]["training_supported"] is False


def test_qwen_settings_mlx_options_include_cache_availability():
    settings = api.qwen_settings()

    assert settings.mlx_models
    availability = settings.mlx_models[0].get("availability")
    assert availability is not None
    assert {"local", "partial", "needs_download", "loaded"} <= set(availability)


def test_qwen_cache_snapshot_path_prefers_hf_snapshot_dir(tmp_path, monkeypatch):
    model_id = "owner/model"
    commit = "abc123"
    repo = tmp_path / "hub" / "models--owner--model"
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit, encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.visual.blocks.0.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

    assert api._qwen_cache_snapshot_path(model_id) == snapshot.resolve()


def test_qwen_settings_excludes_cached_language_only_mlx_options(monkeypatch):
    bad_model_id = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

    def fake_cached_incompatibility(model_id, _availability):
        if model_id == bad_model_id:
            return f"{model_id}: language-only checkpoint"
        return None

    monkeypatch.setattr(
        api,
        "_qwen_mlx_cached_checkpoint_incompatibility_detail",
        fake_cached_incompatibility,
    )

    settings = api.qwen_settings()

    assert bad_model_id not in {entry["id"] for entry in settings.mlx_models}


def test_update_qwen_settings_rejects_invalid_inference_platform(monkeypatch):
    monkeypatch.setattr(api, "QWEN_INFERENCE_PLATFORM", api.QWEN_PLATFORM_AUTO)
    monkeypatch.setattr(api, "_unload_qwen_runtime", lambda: None)

    with pytest.raises(api.HTTPException) as excinfo:
        api.update_qwen_settings(api.QwenRuntimeSettingsUpdate(inference_platform="cuda"))

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen_inference_platform_invalid"
    assert api.QWEN_INFERENCE_PLATFORM == api.QWEN_PLATFORM_AUTO


def test_update_qwen_settings_accepts_inference_platform_alias(monkeypatch):
    unloads = []
    monkeypatch.setattr(api, "QWEN_INFERENCE_PLATFORM", api.QWEN_PLATFORM_AUTO)
    monkeypatch.setattr(api, "_unload_qwen_runtime", lambda: unloads.append("unloaded"))

    settings = api.update_qwen_settings(api.QwenRuntimeSettingsUpdate(inference_platform="torch"))

    assert settings.inference_platform == api.QWEN_PLATFORM_TRANSFORMERS
    assert api.QWEN_INFERENCE_PLATFORM == api.QWEN_PLATFORM_TRANSFORMERS
    assert unloads == ["unloaded"]


def test_qwen_mlx_checkpoint_index_detects_missing_visual_weights(tmp_path):
    index_path = tmp_path / "model.safetensors.index.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"model_type": "qwen3_vl", "vision_start_token_id": 151652}),
        encoding="utf-8",
    )
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                    "model.language_model.layers.0.mlp.up_proj.weight": "model.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    detail = api._qwen_mlx_checkpoint_index_incompatibility_detail(
        "example/Qwen3-VL-language-only", index_path
    )

    assert detail is not None
    assert "no visual/vision weights" in detail


def test_qwen_mlx_checkpoint_index_accepts_visual_weights(tmp_path):
    index_path = tmp_path / "model.safetensors.index.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8")
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                    "model.visual.blocks.0.attn.qkv.weight": "model.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    detail = api._qwen_mlx_checkpoint_index_incompatibility_detail(
        "example/Qwen3-VL-full", index_path
    )

    assert detail is None


def test_qwen_activation_rejects_language_only_mlx_repack():
    bad_model_id = (
        "introvoyz041/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx-mlx-4Bit"
    )

    with pytest.raises(api.HTTPException) as excinfo:
        api.activate_qwen_model(api.QwenModelActivateRequest(model_id=bad_model_id))

    assert excinfo.value.status_code == 400
    assert "qwen_mlx_incompatible_checkpoint" in str(excinfo.value.detail)


def test_qwen_mlx_runtime_rejects_blocked_heretic_candidate(monkeypatch):
    model_id = "Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit"
    load_called = False

    def fake_load(*_args, **_kwargs):
        nonlocal load_called
        load_called = True
        return object(), object()

    monkeypatch.setattr(api, "MLX_VLM_LOAD", fake_load)
    monkeypatch.setattr(api, "MLX_VLM_GENERATE", lambda *_args, **_kwargs: "")

    with pytest.raises(api.HTTPException) as excinfo:
        api._load_qwen_mlx_runtime(model_id)

    assert excinfo.value.status_code == 400
    assert "qwen_mlx_incompatible_checkpoint" in str(excinfo.value.detail)
    assert "generated invalid text" in str(excinfo.value.detail)
    assert load_called is False


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


def test_qwen_mlx_model_id_resolution_keeps_explicit_heretic_mlx_id():
    model_id = "Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit"
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
        chat_template_kwargs={"enable_thinking": False},
    )

    assert "car" in text
    assert (width, height) == (1000, 1000)
    assert calls["prompt"] == "formatted prompt"
    assert calls["image"] == [img]
    assert calls["kwargs"]["max_tokens"] == 64
    assert calls["kwargs"]["temperature"] == 0.0
    assert calls["template_kwargs"]["enable_thinking"] is False
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
