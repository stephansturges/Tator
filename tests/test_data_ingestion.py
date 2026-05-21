import asyncio

import numpy as np
import pytest
import torch

import localinferenceapi as api
from services.data_ingestion import greedy_diverse_indices, normalize_keep_fraction
from utils.local_salad import LocalSALADConfig, LocalSALADHead
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


def test_data_ingestion_capabilities_expose_salad_and_cradio():
    caps = api._data_ingestion_capabilities()

    assert "local_salad" in caps["encoders"]
    assert "cradio_pooled" in caps["encoders"]
    assert caps["default_cradio_model"] == api.CRADIO_DEFAULT_MODEL
    assert "summary_spatial_concat" in caps["cradio_pooling_modes"]
    assert caps["salad_policy"] == "local_training_only"
    assert caps["local_salad_policy"] == api.LOCAL_SALAD_POLICY
    assert caps["local_salad_trainer"] == api.LOCAL_SALAD_TRAINER
    assert "local_salad_heads" in caps
    assert any(recipe["id"] == "local_salad_top20" for recipe in caps["data_ingestion_recipes"])
    assert any(recipe["id"] == "cradio_top20" and recipe["cradio_pooling"] == "summary" for recipe in caps["data_ingestion_recipes"])
    assert api._local_salad_training_stage(0.02) == "Selecting ingredients"
    assert api._local_salad_training_stage(0.20) == "Washing lettuce"
    assert api._local_salad_training_stage(0.50) == "Mixing dressing"
    assert api._local_salad_training_stage(0.80) == "Tossing salad"
    assert api._local_salad_training_stage(0.99) == "Finalizing SALAD"


def test_data_ingestion_create_jobs_reject_empty_uploads_before_queueing():
    with pytest.raises(api.HTTPException) as analysis_error:
        asyncio.run(api.create_data_ingestion_analysis_job("{}", [], []))
    assert analysis_error.value.status_code == 400
    assert analysis_error.value.detail == "data_ingestion_no_candidate_files"

    with pytest.raises(api.HTTPException) as salad_error:
        asyncio.run(api.create_local_salad_training_job("{}", []))
    assert salad_error.value.status_code == 400
    assert salad_error.value.detail == "local_salad_no_training_files"


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
    monkeypatch.setattr(api, "resolve_cradio_torch_device", lambda _requested=None: "cpu")

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
    desc = loaded(patches)

    assert meta["id"] == "unit_head"
    assert desc.shape == (2, loaded.output_dim)
    assert torch.allclose(desc.norm(dim=1), torch.ones(2), atol=1e-5)


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
