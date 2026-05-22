import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import localinferenceapi as api
from services.data_ingestion import greedy_diverse_indices, normalize_keep_fraction
from utils.local_salad import LocalSALADConfig, LocalSALADHead
from utils.local_salad_mlx import local_salad_mlx_available
from PIL import Image


def test_diverse_indices_use_reference_novelty_and_keep_fraction():
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    reference = np.asarray([[1.0, 0.0]], dtype=np.float32)

    selected, novelty = greedy_diverse_indices(embeddings, keep_fraction=0.5, reference_embeddings=reference)

    assert normalize_keep_fraction("20") == 0.2
    assert selected[0] == 3
    assert len(selected) == 2
    assert novelty[3] > novelty[2] > novelty[1]


def test_data_ingestion_capabilities_expose_reference_profile_flow():
    caps = api._data_ingestion_capabilities()

    assert caps["encoders"] == ["local_salad"]
    assert caps["default_encoder"] == "local_salad"
    assert caps["reference_profile_flow"] is True
    assert caps["analysis_encoders"] == ["local_salad"]
    assert caps["reference_profile_base_encoders"] == ["dinov3", "cradio"]
    assert caps["default_cradio_model"] == api.CRADIO_DEFAULT_MODEL
    assert "summary_spatial_concat" in caps["cradio_pooling_modes"]
    assert caps["salad_policy"] == "local_training_only"
    assert caps["local_salad_policy"] == api.LOCAL_SALAD_POLICY
    assert caps["local_salad_trainer"] == api.LOCAL_SALAD_TRAINER
    assert caps["local_salad_backend"]["default"] == "auto"
    assert caps["local_salad_backend"]["auto_resolved"] in {"mlx", "torch"}
    assert "local_salad_heads" in caps
    assert "data_ingestion_recipes" not in caps
    assert api._local_salad_training_stage(0.02) == "Preparing reference media"
    assert api._local_salad_training_stage(0.20) == "Encoding reference views"
    assert api._local_salad_training_stage(0.50) == "Training reference profile"
    assert api._local_salad_training_stage(0.80) == "Optimizing reference profile"
    assert api._local_salad_training_stage(0.99) == "Finalizing reference profile"


class _FakeUpload:
    def __init__(self, filename: str, payload: bytes):
        self.filename = filename
        self._payload = payload
        self._read = False

    async def read(self, _size: int = -1):
        if self._read:
            return b""
        self._read = True
        return self._payload


def _write_unit_local_salad_head(root: Path, head_id: str, metadata: dict | None = None) -> None:
    config = LocalSALADConfig(num_channels=8, num_clusters=2, cluster_dim=4, token_dim=6, hidden_dim=12, dropout=0.0)
    head = LocalSALADHead(config)
    payload_metadata = {
        "id": head_id,
        "label": head_id,
        "policy": api.LOCAL_SALAD_POLICY,
        "trainer": api.LOCAL_SALAD_TRAINER,
        **(metadata or {}),
    }
    torch.save(
        {
            "format": api.LOCAL_SALAD_CACHE_VERSION,
            "config": config.to_dict(),
            "state_dict": head.state_dict(),
            "metadata": payload_metadata,
        },
        root / f"{head_id}.pt",
    )


def test_data_ingestion_create_jobs_reject_empty_uploads_before_queueing(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", tmp_path)

    with pytest.raises(api.HTTPException) as analysis_error:
        asyncio.run(api.create_data_ingestion_analysis_job("{}", [], []))
    assert analysis_error.value.status_code == 400
    assert analysis_error.value.detail == "data_ingestion_no_candidate_files"

    with pytest.raises(api.HTTPException) as reference_error:
        asyncio.run(api.create_data_ingestion_analysis_job("{}", [_FakeUpload("candidate.jpg", b"candidate")], []))
    assert reference_error.value.status_code == 400
    assert reference_error.value.detail == "data_ingestion_no_reference_files"

    with pytest.raises(api.HTTPException) as encoder_error:
        asyncio.run(
            api.create_data_ingestion_analysis_job(
                json.dumps({"encoder": "dinov3_pooled"}),
                [_FakeUpload("candidate.jpg", b"candidate")],
                [_FakeUpload("reference.jpg", b"reference")],
            )
        )
    assert encoder_error.value.status_code == 400
    assert encoder_error.value.detail == "data_ingestion_encoder_unsupported"

    with pytest.raises(api.HTTPException) as salad_error:
        asyncio.run(api.create_local_salad_training_job("{}", []))
    assert salad_error.value.status_code == 400
    assert salad_error.value.detail == "local_salad_no_training_files"


def test_data_ingestion_backend_dataset_rows_and_training_queue(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    images_root = dataset_root / "images"
    images_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(images_root / "a.jpg")
    Image.new("RGB", (8, 8), (40, 50, 60)).save(images_root / "b.jpg")
    entry = {
        "id": "unit_dataset",
        "label": "Unit Dataset",
        "dataset_root": str(dataset_root),
        "yolo_layout": "flat",
    }
    jobs_root = tmp_path / "jobs"
    heads_root = tmp_path / "heads"
    jobs_root.mkdir()
    heads_root.mkdir()
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)

    rows = api._data_ingestion_dataset_media_rows("unit_dataset", field_name="reference", max_count=1)
    assert len(rows) == 1
    assert rows[0]["source_dataset_id"] == "unit_dataset"
    assert rows[0]["source_dataset_label"] == "Unit Dataset"
    assert rows[0]["field"] == "reference"

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(api.threading, "Thread", DummyThread)
    manifest = json.dumps({"reference_dataset_id": "unit_dataset", "reference_source": "backend_dataset"})
    result = asyncio.run(api.create_local_salad_training_job(manifest, []))
    job = api.DATA_INGESTION_JOBS[result["job_id"]]
    assert job.kind == "local_salad_train"
    assert len(job.request["train_uploads"]) == 2
    assert job.request["train_uploads"][0]["source_dataset_id"] == "unit_dataset"


def test_data_ingestion_active_reference_dataset_id_is_metadata_only(tmp_path, monkeypatch):
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)

    def fail_dataset_resolution(_dataset_id):
        raise AssertionError("active Label Images dataset id should not resolve as a backend dataset")

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(api, "_resolve_dataset_entry", fail_dataset_resolution)
    monkeypatch.setattr(api.threading, "Thread", DummyThread)
    _write_unit_local_salad_head(
        jobs_root,
        "open_label_images_head",
        {
            "reference_source": "active_label_images",
            "reference_dataset_id": "open_label_images_dataset",
        },
    )
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", jobs_root)
    manifest = json.dumps({
        "reference_source": "active_label_images",
        "reference_dataset_id": "open_label_images_dataset",
        "encoder": "local_salad",
        "salad_head_id": "open_label_images_head",
    })

    analysis_result = asyncio.run(
        api.create_data_ingestion_analysis_job(
            manifest,
            [_FakeUpload("candidate.jpg", b"candidate")],
            [_FakeUpload("reference.jpg", b"reference")],
        )
    )
    analysis_job = api.DATA_INGESTION_JOBS[analysis_result["job_id"]]
    assert analysis_job.request["reference_dataset_id"] == "open_label_images_dataset"
    assert len(analysis_job.request["reference_uploads"]) == 1

    train_result = asyncio.run(
        api.create_local_salad_training_job(
            manifest,
            [_FakeUpload("train_a.jpg", b"a"), _FakeUpload("train_b.jpg", b"b")],
        )
    )
    train_job = api.DATA_INGESTION_JOBS[train_result["job_id"]]
    assert train_job.request["reference_dataset_id"] == "open_label_images_dataset"
    assert len(train_job.request["train_uploads"]) == 2


def test_local_salad_reference_matching_requires_specific_metadata():
    assert not api._local_salad_head_reference_matches_request(
        {},
        {"reference_source": "active_label_images"},
    )
    assert api._local_salad_head_reference_matches_request(
        {"reference_source": "active_label_images"},
        {"reference_source": "active_label_images"},
    )
    assert not api._local_salad_head_reference_matches_request(
        {"reference_source": "backend_dataset"},
        {"reference_source": "backend_dataset", "reference_dataset_id": "dataset_a"},
    )
    assert api._local_salad_head_reference_matches_request(
        {"reference_source": "active_label_images", "reference_dataset_id": "dataset_a"},
        {"reference_source": "backend_dataset", "reference_dataset_id": "dataset_a"},
    )
    assert not api._local_salad_head_reference_matches_request(
        {"reference_source": "active_label_images", "reference_dataset_id": "dataset_a"},
        {"reference_source": "backend_dataset", "reference_dataset_id": "dataset_b"},
    )


def test_data_ingestion_rejects_mismatched_local_salad_profile_before_queue(tmp_path, monkeypatch):
    jobs_root = tmp_path / "jobs"
    heads_root = tmp_path / "heads"
    jobs_root.mkdir()
    heads_root.mkdir()
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    _write_unit_local_salad_head(
        heads_root,
        "dataset_a_head",
        {
            "reference_source": "backend_dataset",
            "reference_dataset_id": "dataset_a",
        },
    )
    manifest = json.dumps({
        "encoder": "local_salad",
        "salad_head_id": "dataset_a_head",
        "reference_source": "backend_dataset",
        "reference_dataset_id": "dataset_b",
    })

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api.create_data_ingestion_analysis_job(
                manifest,
                [_FakeUpload("candidate.jpg", b"candidate")],
                [_FakeUpload("reference.jpg", b"reference")],
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "local_salad_head_reference_mismatch"


def test_data_ingestion_runtime_rejects_mismatched_local_salad_profile(tmp_path, monkeypatch):
    heads_root = tmp_path / "heads"
    heads_root.mkdir()
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    _write_unit_local_salad_head(
        heads_root,
        "dataset_a_head",
        {
            "reference_source": "backend_dataset",
            "reference_dataset_id": "dataset_a",
        },
    )
    job = api.DataIngestionJob(
        job_id="di_mismatch_runtime",
        kind="analysis",
        request={
            "encoder": "local_salad",
            "salad_head_id": "dataset_a_head",
            "reference_source": "backend_dataset",
            "reference_dataset_id": "dataset_b",
            "candidate_uploads": [{"path": str(tmp_path / "candidate.jpg"), "filename": "candidate.jpg"}],
            "reference_uploads": [{"path": str(tmp_path / "reference.jpg"), "filename": "reference.jpg"}],
        },
    )

    api._run_data_ingestion_analysis_job(job)

    assert job.status == "failed"
    assert job.error == "local_salad_head_reference_mismatch"


def test_data_ingestion_runtime_requires_local_salad_encoder(tmp_path, monkeypatch):
    job = api.DataIngestionJob(
        job_id="di_pooled_runtime",
        kind="analysis",
        request={
            "encoder": "dinov3_pooled",
            "candidate_uploads": [{"path": str(tmp_path / "candidate.jpg"), "filename": "candidate.jpg"}],
            "reference_uploads": [{"path": str(tmp_path / "reference.jpg"), "filename": "reference.jpg"}],
        },
    )

    api._run_data_ingestion_analysis_job(job)

    assert job.status == "failed"
    assert job.error == "data_ingestion_encoder_unsupported"


def test_data_ingestion_analysis_requires_reference_profile_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", tmp_path / "jobs")
    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api.create_data_ingestion_analysis_job(
                json.dumps({"reference_source": "active_label_images"}),
                [_FakeUpload("candidate.jpg", b"candidate")],
                [_FakeUpload("reference.jpg", b"reference")],
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "local_salad_head_required"


def test_data_ingestion_cradio_pooled_uses_requested_pooling(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (12, 12), (20, 40, 60)).save(image_path)
    prepared = [
        {
            "image_path": str(image_path),
            "filename": "sample.jpg",
            "source_type": "image",
            "frame_index": 0,
            "width": 12,
            "height": 12,
        }
    ]
    captured = {}

    monkeypatch.setattr(
        api,
        "_load_cradio_backbone_cached",
        lambda model_name, target_device, raise_on_error=False: ("model", "processor", model_name, "cpu"),
    )
    monkeypatch.setattr(api, "resolve_cradio_torch_device", lambda _requested=None, **_kwargs: "cpu")

    def fake_encode(model, processor, device_name, images, *, pooling, normalize=True, return_tokens=False):
        captured["pooling"] = pooling
        captured["normalize"] = normalize
        return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(api, "encode_cradio_images", fake_encode)
    job = api.DataIngestionJob(job_id="di_cradio_unit")
    feats = api._data_ingestion_encode_prepared_images(
        prepared,
        job=job,
        encoder="cradio_pooled",
        model_name=api.CRADIO_DEFAULT_MODEL,
        cradio_pooling="spatial_mean",
    )

    assert feats.shape == (1, 3)
    assert captured == {"pooling": "spatial_mean", "normalize": True}
    assert "C-RADIOv4 pooled" in job.message


def test_local_salad_head_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", tmp_path)
    config = LocalSALADConfig(num_channels=8, num_clusters=2, cluster_dim=4, token_dim=6, hidden_dim=12, dropout=0.0)
    head = LocalSALADHead(config)
    path = api._local_salad_head_path("unit_head")
    torch.save(
        {
            "format": api.LOCAL_SALAD_CACHE_VERSION,
            "config": config.to_dict(),
            "state_dict": head.state_dict(),
            "metadata": {
                "id": "unit_head",
                "label": "Unit Head",
                "train_image_count": 3,
                "policy": api.LOCAL_SALAD_POLICY,
                "trainer": api.LOCAL_SALAD_TRAINER,
            },
        },
        path,
    )

    loaded, meta = api._load_local_salad_head("unit_head")
    patches = torch.randn(2, 5, 8)
    desc = api._encode_local_salad_head_np(loaded, patches)

    assert meta["id"] == "unit_head"
    assert desc.shape == (2, loaded.output_dim)
    assert np.allclose(np.linalg.norm(desc, axis=1), np.ones(2), atol=1e-5)


def test_local_salad_training_job_uses_mlx_backend_and_saves_compatible_head(tmp_path, monkeypatch):
    if not local_salad_mlx_available():
        pytest.skip("MLX is not available in this environment")

    jobs_root = tmp_path / "jobs"
    heads_root = tmp_path / "heads"
    jobs_root.mkdir()
    heads_root.mkdir()
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)

    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (24, 24), (30, 80, 120)).save(image_a)
    Image.new("RGB", (24, 24), (140, 70, 20)).save(image_b)
    prepared = [
        {"image_path": str(image_a), "filename": "a.jpg", "width": 24, "height": 24, "source_type": "image"},
        {"image_path": str(image_b), "filename": "b.jpg", "width": 24, "height": 24, "source_type": "image"},
    ]

    class DummyDino:
        class Config:
            hidden_size = 8

        config = Config()

    monkeypatch.setattr(api, "_data_ingestion_prepare_media", lambda *args, **kwargs: list(prepared))
    monkeypatch.setattr(api, "_data_ingestion_get_dinov3", lambda model_name: (DummyDino(), object(), "unit-dino", "cpu"))

    def fake_dinov3_tokens(model_obj, processor_obj, device_name, images):
        patches = []
        globals_ = []
        ramp = torch.arange(48, dtype=torch.float32).reshape(6, 8) / 100.0
        for img in images:
            base = float(np.asarray(img.convert("RGB"), dtype=np.float32).mean() / 255.0)
            sample = ramp + base
            patches.append(sample)
            globals_.append(sample.mean(dim=0))
        return torch.stack(patches), torch.stack(globals_)

    monkeypatch.setattr(api, "_data_ingestion_dinov3_tokens", fake_dinov3_tokens)

    job = api.DataIngestionJob(
        job_id="salad_mlx_unit",
        kind="local_salad_train",
        request={
            "train_uploads": prepared,
            "encoder_type": "dinov3",
            "encoder_model": "unit-dino",
            "head_name": "MLX Unit Head",
            "local_salad_backend": "mlx",
            "epochs": 1,
            "batch_size": 2,
            "num_clusters": 4,
            "cluster_dim": 8,
            "token_dim": 8,
            "hidden_dim": 64,
            "dropout": 0.0,
            "seed": 7,
        },
    )

    api._run_local_salad_training_job(job)

    assert job.status == "completed"
    assert job.result
    summary = job.result["summary"]
    assert summary["salad_backend"] == "mlx"
    assert summary["head_id"] == "MLX_Unit_Head"
    assert Path(summary["path"]).exists()

    loaded, meta = api._load_local_salad_head(summary["head_id"], device_name="cpu", backend="torch")
    assert isinstance(loaded, LocalSALADHead)
    assert meta["salad_backend"] == "mlx"
    assert meta["encoder_type"] == "dinov3"
    assert meta["encoder_model"] == "unit-dino"
    with Image.open(image_a) as opened_a, Image.open(image_b) as opened_b:
        test_patches, test_global = fake_dinov3_tokens(None, None, "cpu", [opened_a, opened_b])
    desc = api._encode_local_salad_head_np(loaded, test_patches, test_global)
    assert desc.shape == (2, summary["descriptor_dim"])
    assert np.allclose(np.linalg.norm(desc, axis=1), np.ones(2), atol=1e-5)


def test_local_salad_head_loader_rejects_external_or_stale_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", tmp_path)
    torch.save({"state_dict": {}, "metadata": {"label": "Not a Tator local head"}}, tmp_path / "external.pt")
    torch.save(
        {
            "format": api.LOCAL_SALAD_CACHE_VERSION,
            "config": LocalSALADConfig(num_channels=8).to_dict(),
            "state_dict": LocalSALADHead(LocalSALADConfig(num_channels=8)).state_dict(),
            "metadata": {
                "label": "Policy marker but no local trainer marker",
                "policy": api.LOCAL_SALAD_POLICY,
            },
        },
        tmp_path / "forged.pt",
    )

    with pytest.raises(Exception) as exc_info:
        api._load_local_salad_head("external")
    assert "local_salad_head_unsupported_format" in str(exc_info.value)
    with pytest.raises(Exception) as exc_info:
        api._load_local_salad_head("forged")
    assert "local_salad_head_trainer_required" in str(exc_info.value)


def test_local_salad_head_ids_do_not_overwrite_existing_heads(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", tmp_path)
    (tmp_path / "local_salad_head.pt").write_bytes(b"placeholder")

    assert api._unique_local_salad_head_id("local salad head") == "local_salad_head_2"
