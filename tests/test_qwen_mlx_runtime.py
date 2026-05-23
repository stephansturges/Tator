import json
from pathlib import Path
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
