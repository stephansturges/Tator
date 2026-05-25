import asyncio
import json
import os
import random
import shutil
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

import localinferenceapi as api
from services.data_ingestion import (
    greedy_diverse_indices,
    greedy_diverse_indices_with_local_scores,
    greedy_diverse_indices_with_scores,
    local_vendi_metric,
    local_vendi_metrics_from_patch_tokens,
    normalize_keep_fraction,
    safe_media_name,
    score_percentiles,
)
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


def test_diverse_indices_return_candidate_coverage_scores():
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [-0.98, 0.02],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    reference = np.asarray([[1.0, 0.0]], dtype=np.float32)

    selected, novelty, coverage = greedy_diverse_indices_with_scores(
        embeddings,
        keep_fraction=0.5,
        reference_embeddings=reference,
    )

    assert selected == [3, 2]
    assert novelty[3] > novelty[1] > novelty[2]
    assert coverage[3] == pytest.approx(novelty[3])
    assert coverage[2] <= novelty[2]
    assert coverage[1] < novelty[1]


def test_local_vendi_metric_tracks_patch_token_effective_rank():
    identical = np.ones((8, 4), dtype=np.float32)
    diverse = np.eye(8, dtype=np.float32)

    identical_metric = local_vendi_metric(identical)
    diverse_metric = local_vendi_metric(diverse)

    assert identical_metric["local_vendi_effective_patches"] == pytest.approx(1.0)
    assert identical_metric["local_vendi_score"] == pytest.approx(0.0, abs=1e-6)
    assert diverse_metric["local_vendi_effective_patches"] == pytest.approx(8.0)
    assert diverse_metric["local_vendi_score"] > 0.99


def test_local_vendi_batch_metrics_use_deterministic_patch_cap():
    tokens = np.arange(2 * 12 * 4, dtype=np.float32).reshape(2, 12, 4)

    first = local_vendi_metrics_from_patch_tokens(tokens, max_patches=5)
    second = local_vendi_metrics_from_patch_tokens(tokens, max_patches=5)

    assert first == second
    assert all(metric["local_vendi_patch_count"] == 12 for metric in first)
    assert all(metric["local_vendi_used_patch_count"] == 5 for metric in first)


def test_score_percentiles_handles_nonfinite_values():
    scores = score_percentiles([0.5, np.nan, np.inf, -np.inf])

    assert np.all(np.isfinite(scores))
    assert scores[2] == pytest.approx(1.0)
    assert scores[3] == pytest.approx(0.0)


def test_diverse_indices_local_vendi_weight_can_break_coverage_ties():
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    reference = np.asarray([[1.0, 0.0]], dtype=np.float32)

    selected_base, _novelty, _coverage = greedy_diverse_indices_with_scores(
        embeddings,
        keep_fraction=1 / 3,
        reference_embeddings=reference,
    )
    selected_vendi, _novelty_vendi, coverage_vendi, selection_scores = greedy_diverse_indices_with_local_scores(
        embeddings,
        keep_fraction=1 / 3,
        reference_embeddings=reference,
        local_scores=[0.1, 0.2, 1.0],
        local_weight=0.2,
    )

    assert selected_base == [1]
    assert selected_vendi == [2]
    assert coverage_vendi[selected_vendi[0]] == pytest.approx(1.0)
    assert selection_scores[selected_vendi[0]] == pytest.approx(0.8)


def test_diverse_indices_local_vendi_zero_weight_preserves_coverage_selection():
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    reference = np.asarray([[1.0, 0.0]], dtype=np.float32)

    selected_base, novelty_base, coverage_base = greedy_diverse_indices_with_scores(
        embeddings,
        keep_fraction=0.5,
        reference_embeddings=reference,
    )
    selected_vendi, novelty_vendi, coverage_vendi, selection_vendi = greedy_diverse_indices_with_local_scores(
        embeddings,
        keep_fraction=0.5,
        reference_embeddings=reference,
        local_scores=[1.0, 0.0, 0.0, 0.0],
        local_weight=0.0,
    )

    assert selected_vendi == selected_base
    assert np.allclose(novelty_vendi, novelty_base)
    assert np.allclose(coverage_vendi, coverage_base)
    assert np.allclose(selection_vendi, coverage_base)


def test_safe_media_name_bounds_generated_filename_length():
    name = safe_media_name(f"{'a' * 300}.jpg")

    assert len(name) == 96
    assert name == "a" * 96


def test_reference_fingerprint_changes_with_file_mtime_identity():
    base_row = {
        "filename": "a.jpg",
        "source_type": "image",
        "frame_index": 0,
        "width": 8,
        "height": 8,
        "size": 10,
        "mtime_ns": 100,
    }

    first = api._data_ingestion_reference_fingerprint([base_row], source="backend_dataset")
    second = api._data_ingestion_reference_fingerprint([{**base_row, "mtime_ns": 200}], source="backend_dataset")

    assert first["content_hash"] != second["content_hash"]


def test_data_ingestion_novelty_score_metadata_explains_raw_distance_order():
    metadata = api._data_ingestion_novelty_score_metadata([0.2, 0.5, 0.1])

    assert [entry["reference_novelty_rank"] for entry in metadata] == [2, 1, 3]
    assert metadata[1]["reference_novelty_percentile"] == pytest.approx(100.0)
    assert metadata[2]["reference_novelty_percentile"] == pytest.approx(0.0)
    assert metadata[0]["reference_novelty_score"] == pytest.approx(0.2)


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
    assert caps["local_salad_augmentation_profile"] == api.LOCAL_SALAD_AUGMENTATION_PROFILE
    assert caps["local_salad_default_epochs"] == api.LOCAL_SALAD_DEFAULT_EPOCHS
    assert caps["local_vendi"]["enabled"] is True
    assert caps["local_vendi"]["default_enabled"] == api.DATA_INGESTION_LOCAL_VENDI_DEFAULT_ENABLED
    assert caps["local_vendi"]["default_weight"] == api.DATA_INGESTION_LOCAL_VENDI_DEFAULT_WEIGHT
    assert caps["local_vendi"]["max_patches"] == api.DATA_INGESTION_LOCAL_VENDI_MAX_PATCHES
    assert caps["local_salad_backend"]["default"] == "auto"
    assert caps["local_salad_backend"]["auto_resolved"] in {"mlx", "torch"}
    assert "local_salad_heads" in caps
    assert caps["max_extracted_frames_per_video"] == api.DATA_INGESTION_MAX_EXTRACTED_FRAMES_PER_VIDEO
    assert caps["media_prepare_workers"] >= 1
    assert caps["image_load_workers"] >= 1
    assert "data_ingestion_recipes" not in caps
    assert api._local_salad_training_stage(0.02) == "Preparing reference media"
    assert api._local_salad_training_stage(0.20) == "Encoding reference views"
    assert api._local_salad_training_stage(0.50) == "Training reference profile"
    assert api._local_salad_training_stage(0.80) == "Optimizing reference profile"
    assert api._local_salad_training_stage(0.99) == "Finalizing reference profile"


def test_local_salad_train_view_uses_strong_deterministic_augmentation():
    axis = np.linspace(0, 255, 128, dtype=np.uint8)
    xx, yy = np.meshgrid(axis, axis)
    img = Image.fromarray(np.dstack([xx, yy, np.full_like(xx, 128)]))

    first = api._data_ingestion_train_view(img, random.Random(7))
    repeated = api._data_ingestion_train_view(img, random.Random(7))
    assert first.mode == "RGB"
    assert np.array_equal(np.asarray(first), np.asarray(repeated))

    views = [api._data_ingestion_train_view(img, random.Random(seed)) for seed in range(1, 12)]
    assert len({view.size for view in views}) > 1
    channel_equal_fractions = []
    for view in views:
        arr = np.asarray(view)
        channel_equal_fractions.append(float(np.mean((arr[..., 0] == arr[..., 1]) & (arr[..., 1] == arr[..., 2]))))
    assert max(channel_equal_fractions) > 0.95
    assert min(channel_equal_fractions) < 0.05

    large = Image.new("RGB", (1600, 1200), (80, 110, 140))
    large_view = api._data_ingestion_train_view(large, random.Random(13))
    assert max(large_view.size) <= api.LOCAL_SALAD_TRAIN_VIEW_MAX_SIDE


def test_data_ingestion_capabilities_redact_backend_paths(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "cradio" / "model.safetensors"

    monkeypatch.setattr(
        api,
        "cradio_capabilities",
        lambda: {
            "backend": {
                "default": "auto",
                "auto_resolved": "mlx",
                "mlx": {
                    "available": True,
                    "detail": f"Local MLX C-RADIOv4 backend ({checkpoint_path})",
                    "checkpoint_path": str(checkpoint_path),
                },
                "torch_cpu": {"available": True, "detail": "CPU Torch backend"},
            },
        },
    )

    caps = api._data_ingestion_capabilities()
    class_caps = api._class_analysis_capabilities()

    assert str(tmp_path) not in json.dumps(caps)
    assert str(tmp_path) not in json.dumps(class_caps)
    assert caps["cradio_backend"]["mlx"]["detail"] == "Local MLX C-RADIOv4 backend (<local-path>)"
    assert class_caps["cradio_backend"]["mlx"]["detail"] == "Local MLX C-RADIOv4 backend (<local-path>)"
    assert "checkpoint_path" not in caps["cradio_backend"]["mlx"]
    assert "checkpoint_path" not in class_caps["cradio_backend"]["mlx"]


class _FakeUpload:
    def __init__(self, filename: str, payload: bytes):
        self.filename = filename
        self._payload = payload
        self._read = False
        self.closed = False

    async def read(self, _size: int = -1):
        if self._read:
            return b""
        self._read = True
        return self._payload

    async def close(self):
        self.closed = True
        return None


class _FailStartThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        raise RuntimeError("thread start failed")


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


def test_reference_profile_export_import_roundtrip_preserves_metadata(tmp_path, monkeypatch):
    heads_root = tmp_path / "heads"
    heads_root.mkdir()
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    fingerprint = {
        "version": 1,
        "source": "backend_dataset",
        "dataset_id": "dataset_a",
        "label": "Dataset A",
        "image_count": 2,
        "content_hash": "abc123",
    }
    _write_unit_local_salad_head(
        heads_root,
        "dataset_a_profile",
        {
            "reference_source": "backend_dataset",
            "reference_dataset_id": "dataset_a",
            "reference_dataset_label": "Dataset A",
            "reference_fingerprint": fingerprint,
        },
    )

    response = api.export_data_ingestion_reference_profile("dataset_a_profile")
    zip_path = Path(response.path)
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        assert set(zf.namelist()) == {"manifest.json", "profile.pt", "checksums.json"}
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["bundle_version"] == api.REFERENCE_PROFILE_BUNDLE_VERSION
        assert manifest["reference_fingerprint"] == fingerprint

    imported = api._import_data_ingestion_reference_profile_zip(zip_path)

    assert imported["id"] == "dataset_a_profile_2"
    loaded, metadata = api._load_local_salad_head(imported["id"], device_name="cpu", backend="torch")
    assert isinstance(loaded, LocalSALADHead)
    assert metadata["original_profile_id"] == "dataset_a_profile"
    assert metadata["reference_dataset_id"] == "dataset_a"
    assert metadata["reference_fingerprint"] == fingerprint

    asyncio.run(response.background())
    assert not zip_path.parent.exists()


def test_reference_profile_import_rejects_checksum_mismatch(tmp_path, monkeypatch):
    heads_root = tmp_path / "heads"
    heads_root.mkdir()
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    _write_unit_local_salad_head(heads_root, "profile_a", {"reference_source": "active_label_images"})
    response = api.export_data_ingestion_reference_profile("profile_a")
    source_zip = Path(response.path)
    bad_zip = tmp_path / "bad_profile.zip"
    with zipfile.ZipFile(source_zip) as src, zipfile.ZipFile(bad_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            payload = src.read(name)
            if name == "profile.pt":
                payload += b"tamper"
            dst.writestr(name, payload)

    with pytest.raises(api.HTTPException) as exc_info:
        api._import_data_ingestion_reference_profile_zip(bad_zip)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "reference_profile_import_checksum_mismatch"
    asyncio.run(response.background())


def test_reference_profile_import_rejects_duplicate_members(tmp_path, monkeypatch):
    heads_root = tmp_path / "heads"
    heads_root.mkdir()
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    _write_unit_local_salad_head(heads_root, "profile_a", {"reference_source": "active_label_images"})
    response = api.export_data_ingestion_reference_profile("profile_a")
    source_zip = Path(response.path)
    duplicate_zip = tmp_path / "duplicate_profile.zip"
    with zipfile.ZipFile(source_zip) as src, zipfile.ZipFile(duplicate_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            dst.writestr(name, src.read(name))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dst.writestr("manifest.json", src.read("manifest.json"))

    with pytest.raises(api.HTTPException) as exc_info:
        api._import_data_ingestion_reference_profile_zip(duplicate_zip)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "reference_profile_import_duplicate_files"
    asyncio.run(response.background())


def test_data_ingestion_job_serialization_hides_private_paths(tmp_path):
    job = api.DataIngestionJob(
        job_id="di_public_job",
        kind="analysis",
        status="completed",
        request={
            "candidate_uploads": [{"path": str(tmp_path / "candidate.jpg"), "filename": "candidate.jpg"}],
            "reference_uploads": [{"image_path": str(tmp_path / "reference.jpg"), "filename": "reference.jpg"}],
            "keep_fraction": 0.2,
        },
        result={
            "summary": {
                "path": str(tmp_path / "profile.pt"),
                "cache_path": str(tmp_path / "cache"),
                "checkpoint_path": str(tmp_path / "checkpoint.pt"),
                "selected_count": 1,
            }
        },
    )

    serialized = api._serialize_data_ingestion_job(job)

    dumped = json.dumps(serialized)
    assert str(tmp_path) not in dumped
    assert serialized["request"]["candidate_uploads"][0] == {"filename": "candidate.jpg"}
    assert serialized["request"]["reference_uploads"][0] == {"filename": "reference.jpg"}
    assert serialized["summary"] == {"selected_count": 1}


def _register_completed_ingestion_job(api_module, job_id: str, job_dir: Path, image_path: Path) -> None:
    reference_thumb_path, _thumb_dir = api_module._data_ingestion_thumbnail_cache_path(
        job_dir,
        "reference_thumbnails",
        "reference_000000",
    )
    Image.new("RGB", (20, 18), (180, 120, 20)).save(reference_thumb_path)
    result = {
        "summary": {"salad_head_id": "unit_profile", "selected_count": 1},
        "items": [
            {
                "item_id": "item_keep",
                "index": 0,
                "rank": 1,
                "keep": True,
                "diversity_score": 0.92,
                "filename": "candidate_a.jpg",
                "saved_name": "candidate_a.jpg",
                "source_type": "image",
                "frame_index": 0,
                "width": 16,
                "height": 10,
                "image_path": str(image_path),
                "source_path": str(image_path),
                "resolved_path": str(image_path),
            },
            {
                "item_id": "item_skip",
                "index": 1,
                "rank": None,
                "keep": False,
                "diversity_score": 0.12,
                "filename": "candidate_b.jpg",
                "saved_name": "candidate_b.jpg",
                "source_type": "image",
                "frame_index": 0,
                "width": 16,
                "height": 10,
                "image_path": str(image_path),
                "path": str(image_path),
            },
        ],
        "reference_items": [
            {
                "point_id": "reference_000000",
                "index": 0,
                "filename": "reference_a.jpg",
                "source_type": "image",
                "frame_index": 0,
                "width": 20,
                "height": 18,
                "has_thumbnail": True,
                "image_path": str(image_path),
            }
        ],
    }
    result_path = job_dir / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    job = api_module.DataIngestionJob(
        job_id=job_id,
        kind="analysis",
        status="completed",
        progress=1.0,
        result=result,
        result_path=str(result_path),
    )
    api_module.DATA_INGESTION_JOBS[job_id] = job


def test_data_ingestion_result_hides_source_paths_and_exposes_candidate_thumbnails(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_public_result"
    media_dir = job_dir / "media" / "candidates"
    media_dir.mkdir(parents=True)
    image_path = media_dir / "candidate_a.jpg"
    Image.new("RGB", (24, 16), (20, 100, 180)).save(image_path)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_public_result", job_dir, image_path)

    public_result = api.get_data_ingestion_result("di_public_result")

    assert public_result["items"][0]["thumbnail_url"] == "/data_ingestion/jobs/di_public_result/candidate_thumbnail/item_keep"
    assert "image_path" not in public_result["items"][0]
    assert "source_path" not in public_result["items"][0]
    assert "resolved_path" not in public_result["items"][0]
    assert "path" not in public_result["items"][1]
    assert str(tmp_path) not in json.dumps(public_result)


def test_data_ingestion_candidate_thumbnail_uses_internal_source_path(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_candidate_thumb"
    media_dir = job_dir / "media" / "candidates"
    media_dir.mkdir(parents=True)
    image_path = media_dir / "candidate_a.jpg"
    Image.new("RGB", (800, 200), (20, 100, 180)).save(image_path)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_candidate_thumb", job_dir, image_path)

    response = api.get_data_ingestion_candidate_thumbnail("di_candidate_thumb", "item_keep")

    thumb_path = Path(response.path)
    assert thumb_path.exists()
    with Image.open(thumb_path) as img:
        assert max(img.size) == api.DATA_INGESTION_CANDIDATE_THUMB_SIZE


def test_data_ingestion_reference_thumbnail_uses_cached_job_artifact(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_reference_thumb"
    media_dir = job_dir / "media" / "candidates"
    media_dir.mkdir(parents=True)
    image_path = media_dir / "candidate_a.jpg"
    Image.new("RGB", (24, 16), (20, 100, 180)).save(image_path)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_reference_thumb", job_dir, image_path)

    response = api.get_data_ingestion_reference_thumbnail("di_reference_thumb", "reference_000000")

    thumb_path = Path(response.path)
    assert thumb_path.exists()
    assert job_dir in thumb_path.parents
    with Image.open(thumb_path) as img:
        assert img.size == (20, 18)


def test_data_ingestion_reference_thumbnail_lazily_generates_from_internal_path(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_reference_thumb_lazy"
    media_dir = job_dir / "media" / "references"
    media_dir.mkdir(parents=True)
    image_path = media_dir / "reference_a.jpg"
    Image.new("RGB", (900, 300), (20, 100, 180)).save(image_path)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_reference_thumb_lazy", job_dir, image_path)
    cached_thumb, _thumb_dir = api._data_ingestion_thumbnail_cache_path(
        job_dir,
        "reference_thumbnails",
        "reference_000000",
    )
    cached_thumb.unlink()

    response = api.get_data_ingestion_reference_thumbnail("di_reference_thumb_lazy", "reference_000000")

    thumb_path = Path(response.path)
    assert thumb_path == cached_thumb
    assert thumb_path.exists()
    with Image.open(thumb_path) as img:
        assert max(img.size) == api.DATA_INGESTION_REFERENCE_THUMB_SIZE


def test_data_ingestion_reference_thumbnail_allows_backend_dataset_root(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_reference_thumb_dataset"
    candidate_dir = job_dir / "media" / "candidates"
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "candidate_a.jpg"
    Image.new("RGB", (24, 16), (20, 100, 180)).save(candidate_path)
    dataset_root = tmp_path / "backend_dataset"
    dataset_images = dataset_root / "images"
    dataset_images.mkdir(parents=True)
    reference_path = dataset_images / "reference_a.jpg"
    Image.new("RGB", (900, 300), (180, 120, 20)).save(reference_path)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda dataset_id: {
            "id": dataset_id,
            "dataset_root": str(dataset_root),
            "storage_mode": "managed",
        },
    )
    result = {
        "summary": {"salad_head_id": "unit_profile", "selected_count": 1},
        "items": [
            {
                "item_id": "item_keep",
                "index": 0,
                "rank": 1,
                "keep": True,
                "diversity_score": 0.92,
                "filename": "candidate_a.jpg",
                "saved_name": "candidate_a.jpg",
                "source_type": "image",
                "image_path": str(candidate_path),
            }
        ],
        "reference_items": [
            {
                "point_id": "reference_000000",
                "index": 0,
                "filename": "reference_a.jpg",
                "source_type": "image",
                "width": 900,
                "height": 300,
                "has_thumbnail": True,
                "image_path": str(reference_path),
                "_source_dataset_id": "reference_ds",
            }
        ],
    }
    result_path = job_dir / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    api.DATA_INGESTION_JOBS.clear()
    api.DATA_INGESTION_JOBS["di_reference_thumb_dataset"] = api.DataIngestionJob(
        job_id="di_reference_thumb_dataset",
        kind="analysis",
        status="completed",
        progress=1.0,
        result=result,
        result_path=str(result_path),
    )

    response = api.get_data_ingestion_reference_thumbnail("di_reference_thumb_dataset", "reference_000000")
    public_result = api.get_data_ingestion_result("di_reference_thumb_dataset")

    thumb_path = Path(response.path)
    assert job_dir in thumb_path.parents
    with Image.open(thumb_path) as img:
        assert max(img.size) == api.DATA_INGESTION_REFERENCE_THUMB_SIZE
    assert str(dataset_root) not in json.dumps(public_result)
    assert "_source_dataset_id" not in json.dumps(public_result)


def test_data_ingestion_reference_thumbnail_rejects_backend_dataset_escape(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_reference_thumb_escape"
    candidate_dir = job_dir / "media" / "candidates"
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "candidate_a.jpg"
    Image.new("RGB", (24, 16), (20, 100, 180)).save(candidate_path)
    dataset_root = tmp_path / "backend_dataset"
    dataset_root.mkdir()
    outside_path = tmp_path / "outside_reference.jpg"
    Image.new("RGB", (900, 300), (180, 120, 20)).save(outside_path)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda dataset_id: {
            "id": dataset_id,
            "dataset_root": str(dataset_root),
            "storage_mode": "managed",
        },
    )
    result = {
        "summary": {"salad_head_id": "unit_profile", "selected_count": 1},
        "items": [
            {
                "item_id": "item_keep",
                "index": 0,
                "rank": 1,
                "keep": True,
                "diversity_score": 0.92,
                "filename": "candidate_a.jpg",
                "source_type": "image",
                "image_path": str(candidate_path),
            }
        ],
        "reference_items": [
            {
                "point_id": "reference_000000",
                "index": 0,
                "filename": "reference_a.jpg",
                "source_type": "image",
                "has_thumbnail": True,
                "image_path": str(outside_path),
                "_source_dataset_id": "reference_ds",
            }
        ],
    }
    result_path = job_dir / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    api.DATA_INGESTION_JOBS.clear()
    api.DATA_INGESTION_JOBS["di_reference_thumb_escape"] = api.DataIngestionJob(
        job_id="di_reference_thumb_escape",
        kind="analysis",
        status="completed",
        progress=1.0,
        result=result,
        result_path=str(result_path),
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.get_data_ingestion_reference_thumbnail("di_reference_thumb_escape", "reference_000000")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "data_ingestion_reference_thumbnail_not_found"


def test_data_ingestion_distribution_projects_candidates_without_paths(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_distribution"
    media_dir = job_dir / "media" / "candidates"
    media_dir.mkdir(parents=True)
    image_path = media_dir / "candidate_a.jpg"
    Image.new("RGB", (24, 16), (20, 100, 180)).save(image_path)
    np.savez_compressed(
        job_dir / "embeddings.npz",
        candidate_embeddings=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        reference_embeddings=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_distribution", job_dir, image_path)

    distribution = api.get_data_ingestion_distribution("di_distribution")

    assert distribution["summary"]["projection"] == "pca"
    assert distribution["summary"]["projection_fit_basis"] == "reference"
    assert distribution["summary"]["candidate_count"] == 2
    assert distribution["summary"]["reference_count"] == 3
    assert distribution["summary"]["candidate_point_count"] == 2
    assert distribution["summary"]["reference_point_count"] == 3
    assert "image_path" not in json.dumps(distribution)
    assert str(tmp_path) not in json.dumps(distribution)
    reference_points = [point for point in distribution["points"] if point["kind"] == "reference"]
    candidate_points = [point for point in distribution["points"] if point["kind"] == "candidate"]
    assert len(reference_points) == 3
    assert len(candidate_points) == 2
    assert reference_points[0]["thumbnail_url"] == "/data_ingestion/jobs/di_distribution/reference_thumbnail/reference_000000"
    assert reference_points[0]["filename"] == "reference_a.jpg"
    assert candidate_points[0]["thumbnail_url"] == "/data_ingestion/jobs/di_distribution/candidate_thumbnail/item_keep"
    assert all(len(point["projection"]) == 2 for point in distribution["points"])
    assert all(np.isfinite(point["projection"]).all() for point in distribution["points"])


def test_accepted_export_preview_and_download_tiles_kept_items(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_accept_unit"
    media_dir = job_dir / "media" / "candidates"
    media_dir.mkdir(parents=True)
    image_path = media_dir / "candidate_a.jpg"
    Image.new("RGB", (16, 10), (20, 100, 180)).save(image_path)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_accept_unit", job_dir, image_path)

    payload = {
        "transform_mode": "tile",
        "target_width": 8,
        "target_height": 8,
        "tile_edge_policy": "cover_no_padding",
        "tile_overlap": 0,
        "limit": 10,
    }
    preview = api.preview_data_ingestion_accepted_export("di_accept_unit", payload)

    assert preview["total_outputs"] == 4
    assert len(preview["outputs"]) == 4
    assert preview["warnings"]
    assert all("thumbnail_url" in output for output in preview["outputs"])
    first_output_id = preview["outputs"][0]["output_id"]

    thumb = api.get_data_ingestion_accepted_export_thumbnail(
        "di_accept_unit",
        preview["preview_id"],
        first_output_id,
    )
    assert Path(thumb.path).exists()

    download = api.download_data_ingestion_accepted_export(
        "di_accept_unit",
        {**payload, "output_ids": [first_output_id]},
    )
    zip_path = Path(download.path)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "summary.json" in names
        image_names = [name for name in names if name.startswith("images/")]
        assert len(image_names) == 1
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["summary"]["output_count"] == 1
        assert manifest["outputs"][0]["item_id"] == "item_keep"
        assert str(tmp_path) not in json.dumps(manifest)
    asyncio.run(download.background())
    assert not zip_path.parent.exists()


def test_accepted_export_rejects_source_outside_job_dir(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_accept_escape"
    job_dir.mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(outside)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_accept_escape", job_dir, outside)

    with pytest.raises(api.HTTPException) as exc_info:
        api.preview_data_ingestion_accepted_export(
            "di_accept_escape",
            {"transform_mode": "tile", "target_width": 4, "target_height": 4},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "accepted_export_source_missing"


def test_accepted_export_empty_item_ids_do_not_fall_back_to_keep_defaults(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_accept_empty"
    media_dir = job_dir / "media" / "candidates"
    media_dir.mkdir(parents=True)
    image_path = media_dir / "candidate_a.jpg"
    Image.new("RGB", (16, 10), (20, 100, 180)).save(image_path)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_accept_empty", job_dir, image_path)

    preview = api.preview_data_ingestion_accepted_export(
        "di_accept_empty",
        {
            "item_ids": [],
            "transform_mode": "tile",
            "target_width": 8,
            "target_height": 8,
        },
    )

    assert preview["total_outputs"] == 0
    assert preview["outputs"] == []
    with pytest.raises(api.HTTPException) as exc_info:
        api.download_data_ingestion_accepted_export(
            "di_accept_empty",
            {
                "item_ids": [],
                "transform_mode": "tile",
                "target_width": 8,
                "target_height": 8,
            },
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "accepted_export_no_outputs"


def test_accepted_export_drop_partials_drops_too_small_sources(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_accept_drop"
    media_dir = job_dir / "media" / "candidates"
    media_dir.mkdir(parents=True)
    image_path = media_dir / "small.jpg"
    Image.new("RGB", (8, 8), (20, 100, 180)).save(image_path)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_accept_drop", job_dir, image_path)

    payload = {
        "transform_mode": "tile",
        "target_width": 16,
        "target_height": 16,
        "tile_edge_policy": "drop_partials",
    }
    preview = api.preview_data_ingestion_accepted_export("di_accept_drop", payload)

    assert preview["total_outputs"] == 0
    with pytest.raises(api.HTTPException) as exc_info:
        api.download_data_ingestion_accepted_export("di_accept_drop", payload)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "accepted_export_no_outputs"


def test_accepted_export_center_crop_reports_real_source_bounds_and_size(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_accept_center"
    media_dir = job_dir / "media" / "candidates"
    media_dir.mkdir(parents=True)
    image_path = media_dir / "wide.jpg"
    Image.new("RGB", (16, 10), (20, 100, 180)).save(image_path)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_accept_center", job_dir, image_path)

    payload = {"transform_mode": "center_crop", "target_width": 8, "target_height": 8}
    preview = api.preview_data_ingestion_accepted_export("di_accept_center", payload)

    assert preview["total_outputs"] == 1
    assert preview["outputs"][0]["source_bounds"] == [3, 0, 13, 10]
    download = api.download_data_ingestion_accepted_export("di_accept_center", payload)
    zip_path = Path(download.path)
    with zipfile.ZipFile(zip_path) as zf:
        image_names = [name for name in zf.namelist() if name.startswith("images/")]
        assert len(image_names) == 1
        with zf.open(image_names[0]) as handle:
            with Image.open(handle) as img:
                assert img.size == (8, 8)
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["outputs"][0]["source_bounds"] == [3, 0, 13, 10]
    asyncio.run(download.background())


def test_accepted_export_rejects_oversized_target_geometry(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_accept_huge"
    media_dir = job_dir / "media" / "candidates"
    media_dir.mkdir(parents=True)
    image_path = media_dir / "candidate_a.jpg"
    Image.new("RGB", (16, 10), (20, 100, 180)).save(image_path)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_accept_huge", job_dir, image_path)

    with pytest.raises(api.HTTPException) as exc_info:
        api.preview_data_ingestion_accepted_export(
            "di_accept_huge",
            {
                "transform_mode": "tile",
                "target_width": api.ACCEPTED_EXPORT_MAX_TARGET_EDGE + 1,
                "target_height": 8,
            },
        )

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "accepted_export_target_size_too_large"


def test_accepted_export_revalidates_original_source_before_render(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "ingestion"
    job_dir = ingestion_root / "di_accept_revalidate"
    media_dir = job_dir / "media" / "candidates"
    media_dir.mkdir(parents=True)
    image_path = media_dir / "candidate_a.jpg"
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (16, 10), (20, 100, 180)).save(image_path)
    Image.new("RGB", (16, 10), (1, 2, 3)).save(outside)
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    api.DATA_INGESTION_JOBS.clear()
    _register_completed_ingestion_job(api, "di_accept_revalidate", job_dir, image_path)
    _job, _result, config, outputs, _job_dir = api._data_ingestion_plan_accepted_outputs(
        "di_accept_revalidate",
        {"transform_mode": "original"},
    )
    assert len(outputs) == 1
    image_path.unlink()
    try:
        image_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        api._data_ingestion_render_output_image(outputs[0], config)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "accepted_export_source_missing"


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
    assert not any(tmp_path.iterdir())

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
    assert not any(tmp_path.iterdir())


def test_data_ingestion_analysis_cleans_staging_when_thread_start_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", tmp_path)
    monkeypatch.setattr(api, "_validate_local_salad_head_reference", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api.threading, "Thread", _FailStartThread)
    with api.DATA_INGESTION_JOBS_LOCK:
        api.DATA_INGESTION_JOBS.clear()

    candidate = _FakeUpload("candidate.jpg", b"candidate")
    reference = _FakeUpload("reference.jpg", b"reference")
    with pytest.raises(RuntimeError, match="thread start failed"):
        asyncio.run(
            api.create_data_ingestion_analysis_job(
                json.dumps({"encoder": "local_salad", "salad_head_id": "unit_head"}),
                [candidate],
                [reference],
            )
        )

    assert api.DATA_INGESTION_JOBS == {}
    assert not any(tmp_path.iterdir())
    assert candidate.closed is True
    assert reference.closed is True


def test_data_ingestion_analysis_cleanup_revalidates_root_before_rmtree(
    tmp_path, monkeypatch
):
    jobs_root = tmp_path / "jobs"
    outside_root = tmp_path / "outside_jobs"
    marker_box: dict[str, Path] = {}
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)
    monkeypatch.setattr(api, "_validate_local_salad_head_reference", lambda *_args, **_kwargs: None)

    class SwapRootFailThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            job_dirs = [path for path in jobs_root.iterdir() if path.is_dir()]
            assert len(job_dirs) == 1
            job_name = job_dirs[0].name
            shutil.rmtree(jobs_root)
            outside_root.mkdir()
            outside_job = outside_root / job_name
            outside_job.mkdir()
            marker = outside_job / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            marker_box["marker"] = marker
            jobs_root.symlink_to(outside_root, target_is_directory=True)
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(api.threading, "Thread", SwapRootFailThread)
    with api.DATA_INGESTION_JOBS_LOCK:
        api.DATA_INGESTION_JOBS.clear()

    with pytest.raises(RuntimeError, match="thread start failed"):
        asyncio.run(
            api.create_data_ingestion_analysis_job(
                json.dumps({"encoder": "local_salad", "salad_head_id": "unit_head"}),
                [_FakeUpload("candidate.jpg", b"candidate")],
                [_FakeUpload("reference.jpg", b"reference")],
            )
        )

    assert marker_box["marker"].read_text(encoding="utf-8") == "keep"
    assert jobs_root.is_symlink()
    jobs_root.unlink()


def test_data_ingestion_cleanup_unlinks_job_symlink_without_target_delete(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "data_ingestion"
    ingestion_root.mkdir()
    outside_job = tmp_path / "outside_job"
    outside_job.mkdir()
    marker = outside_job / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    job_link = ingestion_root / "di_link"
    try:
        job_link.symlink_to(outside_job, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)

    api._cleanup_data_ingestion_job_dir(job_link)

    assert not job_link.exists()
    assert not job_link.is_symlink()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_data_ingestion_analysis_rejects_symlinked_root_before_upload(tmp_path, monkeypatch):
    outside = tmp_path / "outside_data_ingestion"
    outside.mkdir()
    ingestion_root = tmp_path / "data_ingestion"
    try:
        ingestion_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    monkeypatch.setattr(api, "_validate_local_salad_head_reference", lambda *_args, **_kwargs: None)
    candidate = _FakeUpload("candidate.jpg", b"candidate")
    reference = _FakeUpload("reference.jpg", b"reference")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api.create_data_ingestion_analysis_job(
                json.dumps({
                    "encoder": "local_salad",
                    "salad_head_id": "unit_head",
                    "reference_source": "active_label_images",
                }),
                [candidate],
                [reference],
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "data_ingestion_path_invalid"
    assert list(outside.iterdir()) == []


def test_data_ingestion_analysis_rejects_symlinked_root_parent_before_upload(tmp_path, monkeypatch):
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", linked_parent / "data_ingestion")
    monkeypatch.setattr(api, "_validate_local_salad_head_reference", lambda *_args, **_kwargs: None)
    candidate = _FakeUpload("candidate.jpg", b"candidate")
    reference = _FakeUpload("reference.jpg", b"reference")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api.create_data_ingestion_analysis_job(
                json.dumps({
                    "encoder": "local_salad",
                    "salad_head_id": "unit_head",
                    "reference_source": "active_label_images",
                }),
                [candidate],
                [reference],
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "data_ingestion_path_invalid"
    assert list(outside.iterdir()) == []


def test_data_ingestion_job_dir_rejects_symlinked_existing_job_dir(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "data_ingestion"
    ingestion_root.mkdir()
    outside_job = tmp_path / "outside_job"
    outside_job.mkdir()
    try:
        (ingestion_root / "job_link").symlink_to(outside_job, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api._data_ingestion_job_dir("job_link", create=True)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "data_ingestion_path_invalid"
    assert list(outside_job.iterdir()) == []


def test_local_salad_training_cleans_staging_when_thread_start_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", tmp_path)
    monkeypatch.setattr(api.threading, "Thread", _FailStartThread)
    with api.DATA_INGESTION_JOBS_LOCK:
        api.DATA_INGESTION_JOBS.clear()

    upload = _FakeUpload("train.jpg", b"train")
    with pytest.raises(RuntimeError, match="thread start failed"):
        asyncio.run(api.create_local_salad_training_job("{}", [upload]))

    assert api.DATA_INGESTION_JOBS == {}
    assert not any(tmp_path.iterdir())
    assert upload.closed is True


def test_local_salad_training_rejects_symlinked_root_before_upload(tmp_path, monkeypatch):
    outside = tmp_path / "outside_data_ingestion"
    outside.mkdir()
    ingestion_root = tmp_path / "data_ingestion"
    try:
        ingestion_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api.create_local_salad_training_job("{}", [_FakeUpload("train.jpg", b"train")]))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "data_ingestion_path_invalid"
    assert list(outside.iterdir()) == []


def test_local_salad_training_rejects_symlinked_root_parent_before_upload(tmp_path, monkeypatch):
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", linked_parent / "data_ingestion")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api.create_local_salad_training_job("{}", [_FakeUpload("train.jpg", b"train")]))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "data_ingestion_path_invalid"
    assert list(outside.iterdir()) == []


def test_local_salad_training_cleans_saved_uploads_if_backend_reference_lookup_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", tmp_path)

    def _missing_dataset(_dataset_id):
        raise api.HTTPException(status_code=404, detail="dataset_not_found")

    monkeypatch.setattr(api, "_resolve_dataset_entry", _missing_dataset)
    upload = _FakeUpload("train.jpg", b"train")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api.create_local_salad_training_job(
                json.dumps({
                    "reference_source": "backend_dataset",
                    "reference_dataset_id": "missing_dataset",
                }),
                [upload],
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "dataset_not_found"
    assert upload.closed is True
    assert not any(tmp_path.iterdir())


def test_write_upload_file_rejects_broken_symlink_destination(tmp_path):
    outside = tmp_path / "outside.bin"
    dest = tmp_path / "upload.bin"
    dest.symlink_to(outside)

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api._write_upload_file(_FakeUpload("upload.bin", b"payload"), dest))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "upload_exists"
    assert not outside.exists()


def test_write_upload_file_overwrite_unlinks_symlink_destination(tmp_path):
    outside = tmp_path / "outside.bin"
    dest = tmp_path / "upload.bin"
    dest.symlink_to(outside)

    asyncio.run(
        api._write_upload_file(
            _FakeUpload("upload.bin", b"payload"),
            dest,
            allow_overwrite=True,
        )
    )

    assert not dest.is_symlink()
    assert dest.read_bytes() == b"payload"
    assert not outside.exists()


def test_write_upload_file_overwrite_replaces_symlink_without_target_write(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"external")
    dest = tmp_path / "upload.bin"
    dest.symlink_to(outside)

    asyncio.run(
        api._write_upload_file(
            _FakeUpload("upload.bin", b"payload"),
            dest,
            allow_overwrite=True,
        )
    )

    assert not dest.is_symlink()
    assert dest.read_bytes() == b"payload"
    assert outside.read_bytes() == b"external"


def test_write_upload_file_overwrite_preserves_existing_file_on_size_failure(tmp_path):
    dest = tmp_path / "upload.bin"
    dest.write_bytes(b"existing")
    tmp_dest = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api._write_upload_file(
                _FakeUpload("upload.bin", b"payload"),
                dest,
                max_bytes=3,
                allow_overwrite=True,
            )
        )

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "upload_too_large"
    assert dest.read_bytes() == b"existing"
    assert not tmp_dest.exists()


def test_write_upload_file_rejects_symlinked_parent_without_write(tmp_path):
    outside_dir = tmp_path / "outside_parent"
    outside_dir.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api._write_upload_file(
                _FakeUpload("upload.bin", b"payload"),
                linked_parent / "upload.bin",
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_relative_path"
    assert list(outside_dir.iterdir()) == []


def test_write_upload_file_rejects_symlinked_ancestor_without_write(tmp_path):
    outside_dir = tmp_path / "outside_ancestor"
    outside_dir.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api._write_upload_file(
                _FakeUpload("upload.bin", b"payload"),
                linked_parent / "nested" / "upload.bin",
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_relative_path"
    assert list(outside_dir.iterdir()) == []


def test_save_upload_file_rejects_symlinked_root_without_write(tmp_path):
    outside_dir = tmp_path / "outside_root"
    outside_dir.mkdir()
    upload_root = tmp_path / "upload_root"
    try:
        upload_root.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api._save_upload_file(_FakeUpload("upload.bin", b"payload"), upload_root))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_relative_path"
    assert list(outside_dir.iterdir()) == []


def test_save_upload_file_rejects_symlinked_nested_parent_without_write(tmp_path):
    upload_root = tmp_path / "upload_root"
    upload_root.mkdir()
    outside_dir = tmp_path / "outside_nested"
    outside_dir.mkdir()
    try:
        (upload_root / "nested").symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api._save_upload_file(
                _FakeUpload("nested/upload.bin", b"payload"),
                upload_root,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_relative_path"
    assert list(outside_dir.iterdir()) == []


def test_data_ingestion_save_uploads_closes_upload_handles(tmp_path):
    upload = _FakeUpload("candidate.jpg", b"payload")

    rows = asyncio.run(api._data_ingestion_save_uploads([upload], tmp_path, "candidate"))

    assert upload.closed is True
    assert len(rows) == 1
    assert Path(rows[0]["path"]).read_bytes() == b"payload"


def test_data_ingestion_save_uploads_rejects_unsupported_media_extension(tmp_path):
    upload = _FakeUpload("notes.txt", b"payload")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api._data_ingestion_save_uploads([upload], tmp_path, "candidate"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "data_ingestion_media_extension_unsupported"
    assert upload.closed is True
    assert list(tmp_path.iterdir()) == []


def test_data_ingestion_save_uploads_rejects_symlink_destination(tmp_path):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"external")
    link = tmp_path / "00000_candidate.jpg"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    upload = _FakeUpload("candidate.jpg", b"payload")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api._data_ingestion_save_uploads([upload], tmp_path, "candidate"))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "upload_exists"
    assert upload.closed is True
    assert outside.read_bytes() == b"external"


def test_data_ingestion_save_uploads_enforces_file_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_INGESTION_UPLOAD_MAX_BYTES", 3)
    upload = _FakeUpload("candidate.jpg", b"payload")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api._data_ingestion_save_uploads([upload], tmp_path, "candidate"))

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "upload_too_large"
    assert upload.closed is True
    assert not (tmp_path / "00000_candidate.jpg").exists()


def test_data_ingestion_save_uploads_enforces_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_INGESTION_UPLOAD_MAX_BYTES", 100)
    monkeypatch.setattr(api, "DATA_INGESTION_UPLOAD_QUOTA_BYTES", 8)
    (tmp_path / "existing.bin").write_bytes(b"12345")
    upload = _FakeUpload("candidate.jpg", b"payload")

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api._data_ingestion_save_uploads([upload], tmp_path, "candidate"))

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "upload_quota_exceeded"
    assert upload.closed is True
    assert (tmp_path / "existing.bin").read_bytes() == b"12345"
    assert not (tmp_path / "00000_candidate.jpg").exists()


def test_data_ingestion_video_extraction_applies_backend_frame_cap(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    out_dir = tmp_path / "frames"
    monkeypatch.setattr(api, "DATA_INGESTION_MAX_EXTRACTED_FRAMES_PER_VIDEO", 3)
    monkeypatch.setattr(api.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    seen: dict[str, list[str]] = {}

    def fake_run(cmd, **_kwargs):
        seen["cmd"] = list(cmd)
        pattern = Path(cmd[-1])
        pattern.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (10, 20, 30)).save(pattern.parent / "frame_000001.jpg")
        return api.subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(api.subprocess, "run", fake_run)
    job = api.DataIngestionJob(job_id="di_video_cap")

    rows = api._data_ingestion_prepare_media(
        job,
        [{"path": str(video_path), "filename": "sample.mp4", "saved_name": "sample.mp4", "size": len(b"fake-video")}],
        out_dir=out_dir,
        frame_interval=1.0,
        max_frames_per_video=0,
        progress_start=0.0,
        progress_end=1.0,
    )

    assert "-frames:v" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("-frames:v") + 1] == "3"
    assert len(rows) == 1
    assert rows[0]["source_type"] == "video_frame"


def test_data_ingestion_prepare_media_handles_multiple_videos_with_workers(tmp_path, monkeypatch):
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"fake-video-a")
    video_b.write_bytes(b"fake-video-b")
    out_dir = tmp_path / "frames"
    monkeypatch.setattr(api, "DATA_INGESTION_MEDIA_PREPARE_WORKERS", 2)
    monkeypatch.setattr(api.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    commands: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        commands.append(list(cmd))
        pattern = Path(cmd[-1])
        pattern.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (10, 20, 30)).save(pattern.parent / "frame_000001.jpg")
        return api.subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(api.subprocess, "run", fake_run)
    job = api.DataIngestionJob(job_id="di_multi_video")

    rows = api._data_ingestion_prepare_media(
        job,
        [
            {"path": str(video_a), "filename": "a.mp4", "saved_name": "a.mp4", "size": len(b"fake-video-a")},
            {"path": str(video_b), "filename": "b.mp4", "saved_name": "b.mp4", "size": len(b"fake-video-b")},
        ],
        out_dir=out_dir,
        frame_interval=1.0,
        max_frames_per_video=2,
        progress_start=0.0,
        progress_end=1.0,
    )

    assert len(commands) == 2
    assert [row["filename"] for row in rows] == ["a.mp4", "b.mp4"]
    assert all(row["source_type"] == "video_frame" for row in rows)


def test_local_salad_training_rejects_bad_encoder_before_saving_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", tmp_path)

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api.create_local_salad_training_job(
                json.dumps({"encoder_type": "clip"}),
                [_FakeUpload("train.jpg", b"train")],
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "local_salad_encoder_unsupported"
    assert not any(tmp_path.iterdir())


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


def test_data_ingestion_backend_dataset_rejects_not_allowlisted_linked_record(tmp_path, monkeypatch):
    dataset_root = tmp_path / "outside_dataset"
    (dataset_root / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(dataset_root / "images" / "a.jpg")
    entry = {
        "id": "blocked_link",
        "label": "Blocked Link",
        "dataset_root": str(dataset_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "linked_root_status": "not_allowlisted",
        "yolo_layout": "flat",
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)

    with pytest.raises(api.HTTPException) as exc_info:
        api._data_ingestion_dataset_media_rows("blocked_link", field_name="reference")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_path_not_allowlisted"


def test_data_ingestion_backend_dataset_rows_skip_symlinked_image_root_escape(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside_images = tmp_path / "outside_images"
    outside_images.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(outside_images / "outside.jpg")
    try:
        (dataset_root / "images").symlink_to(outside_images, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    entry = {
        "id": "symlink_dataset",
        "label": "Symlink Dataset",
        "dataset_root": str(dataset_root),
        "yolo_layout": "flat",
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)

    assert api._annotation_collect_images(entry) == []
    assert api._data_ingestion_dataset_media_rows("symlink_dataset", field_name="reference") == []


def test_data_ingestion_backend_dataset_rows_reject_symlinked_dataset_root(tmp_path, monkeypatch):
    outside_root = tmp_path / "outside_dataset"
    (outside_root / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(outside_root / "images" / "a.jpg")
    dataset_link = tmp_path / "dataset_link"
    try:
        dataset_link.symlink_to(outside_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    entry = {
        "id": "symlink_root_dataset",
        "label": "Symlink Root Dataset",
        "dataset_root": str(dataset_link),
        "yolo_layout": "flat",
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)

    with pytest.raises(api.HTTPException) as exc_info:
        api._data_ingestion_dataset_media_rows("symlink_root_dataset", field_name="reference")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_path_not_found"


def test_data_ingestion_backend_dataset_row_cap_counts_valid_rows(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    images_root = dataset_root / "images"
    images_root.mkdir(parents=True)
    valid_path = images_root / "valid.jpg"
    outside_path = tmp_path / "outside.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(valid_path)
    Image.new("RGB", (8, 8), (40, 50, 60)).save(outside_path)
    entry = {
        "id": "mixed_dataset",
        "label": "Mixed Dataset",
        "dataset_root": str(dataset_root),
        "yolo_layout": "flat",
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_collect_images",
        lambda _entry: [
            {
                "split": "train",
                "image_relpath": "outside.jpg",
                "image_name": "outside.jpg",
                "image_path": outside_path,
            },
            {
                "split": "train",
                "image_relpath": "valid.jpg",
                "image_name": "valid.jpg",
                "image_path": valid_path,
            },
        ],
    )

    rows = api._data_ingestion_dataset_media_rows("mixed_dataset", field_name="reference", max_count=1)

    assert len(rows) == 1
    assert rows[0]["filename"] == "train/valid.jpg"
    assert rows[0]["path"] == str(valid_path.resolve())


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


def test_data_ingestion_active_linked_reference_can_use_backend_dataset_rows(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    images_root = dataset_root / "images"
    images_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(images_root / "a.jpg")
    Image.new("RGB", (8, 8), (40, 50, 60)).save(images_root / "b.jpg")
    entry = {
        "id": "linked_active_dataset",
        "label": "Linked Active Dataset",
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

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(api.threading, "Thread", DummyThread)
    _write_unit_local_salad_head(
        heads_root,
        "linked_active_head",
        {
            "reference_source": "active_label_images",
            "reference_dataset_id": "linked_active_dataset",
        },
    )
    manifest = json.dumps({
        "reference_source": "active_label_images",
        "reference_dataset_id": "linked_active_dataset",
        "use_backend_reference_dataset": True,
        "encoder": "local_salad",
        "salad_head_id": "linked_active_head",
    })

    analysis_result = asyncio.run(
        api.create_data_ingestion_analysis_job(
            manifest,
            [_FakeUpload("candidate.jpg", b"candidate")],
            [],
        )
    )
    analysis_job = api.DATA_INGESTION_JOBS[analysis_result["job_id"]]
    assert len(analysis_job.request["reference_uploads"]) == 2
    assert analysis_job.request["reference_uploads"][0]["source_dataset_id"] == "linked_active_dataset"

    train_result = asyncio.run(api.create_local_salad_training_job(manifest, []))
    train_job = api.DATA_INGESTION_JOBS[train_result["job_id"]]
    assert len(train_job.request["train_uploads"]) == 2
    assert train_job.request["train_uploads"][0]["source_dataset_id"] == "linked_active_dataset"


def test_data_ingestion_active_transient_reference_uses_session_rows(tmp_path, monkeypatch):
    dataset_root = tmp_path / "transient_dataset"
    images_root = dataset_root / "images"
    images_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(images_root / "a.jpg")
    Image.new("RGB", (8, 8), (40, 50, 60)).save(images_root / "b.jpg")
    session_id = "transient_active_session"
    jobs_root = tmp_path / "jobs"
    heads_root = tmp_path / "heads"
    jobs_root.mkdir()
    heads_root.mkdir()
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path])
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": str(dataset_root),
            "label": "Transient Active",
            "yolo_layout": "flat",
            "created_at": api.time.time(),
            "updated_at": api.time.time(),
            "expires_at": api.time.time() + 3600,
        }

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(api.threading, "Thread", DummyThread)
    _write_unit_local_salad_head(
        heads_root,
        "transient_active_head",
        {
            "reference_source": "active_label_images",
            "reference_dataset_kind": "transient_session",
            "reference_dataset_id": f"transient:{session_id}",
            "reference_session_id": session_id,
        },
    )
    rows = api._data_ingestion_transient_session_media_rows(
        session_id,
        field_name="reference",
        max_count=1,
    )
    assert len(rows) == 1
    assert rows[0]["source_dataset_id"] == f"transient:{session_id}"
    assert rows[0]["source_session_id"] == session_id

    manifest = json.dumps({
        "reference_source": "active_label_images",
        "reference_dataset_kind": "transient_session",
        "reference_dataset_id": f"transient:{session_id}",
        "reference_session_id": session_id,
        "reference_open_path": str(dataset_root),
        "use_server_reference_dataset": True,
        "encoder": "local_salad",
        "salad_head_id": "transient_active_head",
    })

    analysis_result = asyncio.run(
        api.create_data_ingestion_analysis_job(
            manifest,
            [_FakeUpload("candidate.jpg", b"candidate")],
            [],
        )
    )
    analysis_job = api.DATA_INGESTION_JOBS[analysis_result["job_id"]]
    assert len(analysis_job.request["reference_uploads"]) == 2
    assert analysis_job.request["reference_uploads"][0]["source_session_id"] == session_id

    train_result = asyncio.run(api.create_local_salad_training_job(manifest, []))
    train_job = api.DATA_INGESTION_JOBS[train_result["job_id"]]
    assert len(train_job.request["train_uploads"]) == 2
    assert train_job.request["train_uploads"][0]["source_session_id"] == session_id

    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)
    fallback_train_result = asyncio.run(api.create_local_salad_training_job(manifest, []))
    fallback_train_job = api.DATA_INGESTION_JOBS[fallback_train_result["job_id"]]
    assert len(fallback_train_job.request["train_uploads"]) == 2
    assert fallback_train_job.request["train_uploads"][0]["source_open_path_recovered"] is True


def test_data_ingestion_backend_reference_embeddings_are_cached(tmp_path, monkeypatch):
    jobs_root = tmp_path / "jobs"
    heads_root = tmp_path / "heads"
    jobs_root.mkdir()
    heads_root.mkdir()
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    _write_unit_local_salad_head(
        heads_root,
        "cache_head",
        {
            "reference_source": "active_label_images",
            "reference_dataset_id": "dataset_cache",
        },
    )

    candidate_path = tmp_path / "candidate.jpg"
    ref_a_path = tmp_path / "ref_a.jpg"
    ref_b_path = tmp_path / "ref_b.jpg"
    Image.new("RGB", (8, 8), (200, 10, 10)).save(candidate_path)
    Image.new("RGB", (8, 8), (10, 200, 10)).save(ref_a_path)
    Image.new("RGB", (8, 8), (10, 10, 200)).save(ref_b_path)

    def media_row(path: Path, filename: str, *, source_dataset_id: str = "") -> dict:
        stat = path.stat()
        row = {
            "path": str(path),
            "filename": filename,
            "saved_name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "field": "reference" if source_dataset_id else "candidate",
        }
        if source_dataset_id:
            row.update(
                {
                    "source_dataset_id": source_dataset_id,
                    "source_dataset_label": "Dataset Cache",
                    "split": "train",
                    "image_relpath": filename,
                    "dataset_index": 0,
                }
            )
        return row

    request = {
        "encoder": "local_salad",
        "salad_head_id": "cache_head",
        "reference_source": "active_label_images",
        "reference_dataset_id": "dataset_cache",
        "candidate_uploads": [media_row(candidate_path, "candidate.jpg")],
        "reference_uploads": [
            media_row(ref_a_path, "ref_a.jpg", source_dataset_id="dataset_cache"),
            media_row(ref_b_path, "ref_b.jpg", source_dataset_id="dataset_cache"),
        ],
    }
    reference_encode_calls = {"count": 0}

    def fake_encode(prepared_rows, **_kwargs):
        rows = list(prepared_rows)
        if rows and all(str(row.get("source_dataset_id") or "") == "dataset_cache" for row in rows):
            reference_encode_calls["count"] += 1
        encoded = np.zeros((len(rows), 4), dtype=np.float32)
        for idx in range(len(rows)):
            encoded[idx, idx % 4] = 1.0
        return encoded

    monkeypatch.setattr(api, "_data_ingestion_encode_prepared_images", fake_encode)

    job_one = api.DataIngestionJob(job_id="di_cache_one", kind="analysis", request=dict(request))
    api._run_data_ingestion_analysis_job(job_one)
    assert job_one.status == "completed"
    assert reference_encode_calls["count"] == 1

    cache_files = list((jobs_root / "cache" / "reference_embeddings").glob("*.npz"))
    assert len(cache_files) == 1
    with np.load(jobs_root / job_one.job_id / "embeddings.npz", allow_pickle=False) as payload:
        assert "reference_cache_key" in payload.files
        assert "reference_embeddings" not in payload.files

    job_two = api.DataIngestionJob(job_id="di_cache_two", kind="analysis", request=dict(request))
    api._run_data_ingestion_analysis_job(job_two)
    assert job_two.status == "completed"
    assert reference_encode_calls["count"] == 1
    candidate_embeddings, reference_embeddings = api._data_ingestion_load_embeddings(jobs_root / job_two.job_id)
    assert candidate_embeddings.shape == (1, 4)
    assert reference_embeddings.shape == (2, 4)


def test_local_salad_reference_matching_requires_specific_metadata():
    assert not api._local_salad_head_reference_matches_request(
        {},
        {"reference_source": "active_label_images"},
    )
    assert api._local_salad_head_reference_matches_request(
        {"reference_source": "active_label_images"},
        {"reference_source": "active_label_images"},
    )
    assert api._local_salad_head_reference_matches_request(
        {"reference_source": "active_label_images", "reference_label": "Dataset A"},
        {"reference_source": "active_label_images", "reference_label": "Dataset A"},
    )
    assert not api._local_salad_head_reference_matches_request(
        {"reference_source": "active_label_images", "reference_label": "Dataset A"},
        {"reference_source": "active_label_images", "reference_label": "Dataset B"},
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
    assert api._local_salad_head_reference_matches_request(
        {
            "reference_source": "active_label_images",
            "reference_dataset_kind": "transient_session",
            "reference_dataset_id": "transient:session_a",
            "reference_session_id": "session_a",
        },
        {
            "reference_source": "active_label_images",
            "reference_dataset_kind": "transient_session",
            "reference_dataset_id": "transient:session_a",
            "reference_session_id": "session_a",
        },
    )
    assert not api._local_salad_head_reference_matches_request(
        {
            "reference_source": "active_label_images",
            "reference_dataset_kind": "transient_session",
            "reference_dataset_id": "transient:session_a",
            "reference_session_id": "session_a",
        },
        {
            "reference_source": "active_label_images",
            "reference_dataset_kind": "transient_session",
            "reference_dataset_id": "transient:session_b",
            "reference_session_id": "session_b",
        },
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
    assert list(jobs_root.iterdir()) == []


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


def test_data_ingestion_analysis_summary_uses_local_salad_head_metadata(tmp_path, monkeypatch):
    jobs_root = tmp_path / "jobs"
    heads_root = tmp_path / "heads"
    heads_root.mkdir()
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    _write_unit_local_salad_head(
        heads_root,
        "cradio_profile",
        {
            "reference_source": "active_label_images",
            "encoder_type": "cradio",
            "encoder_model": "nvidia/C-RADIOv4-H",
            "cradio_pooling": "spatial_mean",
            "salad_backend": "mlx",
        },
    )

    def fake_prepare(_job, rows, **_kwargs):
        return [
            {
                "image_path": str(tmp_path / f"{row.get('filename', 'image')}.jpg"),
                "filename": row.get("filename") or "image.jpg",
                "saved_name": row.get("saved_name") or "",
                "source_type": "image",
                "frame_index": 0,
                "width": 8,
                "height": 8,
            }
            for row in rows
        ]

    def fake_encode(prepared, **_kwargs):
        values = np.zeros((len(prepared), 3), dtype=np.float32)
        for idx in range(len(prepared)):
            values[idx, idx % 3] = 1.0
        return values

    monkeypatch.setattr(api, "_data_ingestion_prepare_media", fake_prepare)
    monkeypatch.setattr(api, "_data_ingestion_encode_prepared_images", fake_encode)
    job = api.DataIngestionJob(
        job_id="di_summary_meta",
        kind="analysis",
        request={
            "encoder": "local_salad",
            "encoder_model": "stale-request-model",
            "cradio_pooling": "summary_spatial_concat",
            "salad_head_id": "cradio_profile",
            "reference_source": "active_label_images",
            "candidate_uploads": [{"path": str(tmp_path / "candidate.jpg"), "filename": "candidate.jpg"}],
            "reference_uploads": [{"path": str(tmp_path / "reference.jpg"), "filename": "reference.jpg"}],
        },
    )

    api._run_data_ingestion_analysis_job(job)

    assert job.status == "completed"
    summary = job.result["summary"]
    assert summary["encoder"] == "local_salad"
    assert summary["encoder_type"] == "cradio"
    assert summary["base_encoder"] == "cradio"
    assert summary["encoder_model"] == "nvidia/C-RADIOv4-H"
    assert summary["cradio_pooling"] == "spatial_mean"
    assert summary["salad_head_backend"] == "mlx"


def test_data_ingestion_keep_fraction_applies_to_blended_upload_batch(tmp_path, monkeypatch):
    jobs_root = tmp_path / "jobs"
    heads_root = tmp_path / "heads"
    heads_root.mkdir()
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    _write_unit_local_salad_head(
        heads_root,
        "active_head",
        {"reference_source": "active_label_images"},
    )

    def fake_prepare(_job, rows, *, out_dir, **_kwargs):
        is_candidate = "candidates" in str(out_dir)
        prepared = []
        for row_idx, row in enumerate(rows):
            count = 5 if is_candidate else 1
            for frame_idx in range(count):
                prepared.append(
                    {
                        "image_path": str(tmp_path / f"{row.get('filename', 'item')}_{frame_idx}.jpg"),
                        "filename": row.get("filename") or "item.jpg",
                        "saved_name": row.get("saved_name") or "",
                        "source_type": "video_frame" if is_candidate else "image",
                        "frame_index": frame_idx,
                        "width": 8,
                        "height": 8,
                        "row_idx": row_idx,
                    }
                )
        return prepared

    def fake_encode(prepared, **kwargs):
        values = np.zeros((len(prepared), 3), dtype=np.float32)
        for idx in range(len(prepared)):
            values[idx, idx % 3] = 1.0
            values[idx, 2] += idx * 0.01
        if kwargs.get("return_local_vendi"):
            metrics = [
                {
                    "local_vendi_score": float(idx / max(1, len(prepared) - 1)),
                    "local_vendi_effective_patches": float(idx + 1),
                    "local_vendi_patch_count": 8,
                    "local_vendi_used_patch_count": 8,
                }
                for idx in range(len(prepared))
            ]
            return values, metrics
        return values

    monkeypatch.setattr(api, "_data_ingestion_prepare_media", fake_prepare)
    monkeypatch.setattr(api, "_data_ingestion_encode_prepared_images", fake_encode)
    job = api.DataIngestionJob(
        job_id="di_global_keep_fraction",
        kind="analysis",
        request={
            "encoder": "local_salad",
            "salad_head_id": "active_head",
            "reference_source": "active_label_images",
            "keep_fraction": 0.2,
            "candidate_uploads": [
                {"path": str(tmp_path / "a.mp4"), "filename": "a.mp4"},
                {"path": str(tmp_path / "b.mp4"), "filename": "b.mp4"},
            ],
            "reference_uploads": [{"path": str(tmp_path / "reference.jpg"), "filename": "reference.jpg"}],
        },
    )

    api._run_data_ingestion_analysis_job(job)

    assert job.status == "completed"
    summary = job.result["summary"]
    assert summary["candidate_upload_count"] == 2
    assert summary["candidate_image_count"] == 10
    assert summary["selected_count"] == 2
    assert len([item for item in job.result["items"] if item["keep"]]) == 2
    assert summary["local_vendi_enabled"] is True
    assert summary["selection_score_kind"] == "coverage_percentile_plus_local_vendi"
    assert "Local Vendi" in summary["selection_score_description"]
    assert "local_vendi_score" in job.result["items"][0]
    assert all(item["selection_score_kind"] == summary["selection_score_kind"] for item in job.result["items"])
    assert [item["selection_priority_rank"] for item in job.result["items"]] == list(range(1, 11))
    assert all(item["selection_priority_total"] == 10 for item in job.result["items"])


def test_data_ingestion_analysis_cancelled_before_result_write_leaves_no_result_artifact(tmp_path, monkeypatch):
    jobs_root = tmp_path / "jobs"
    heads_root = tmp_path / "heads"
    heads_root.mkdir()
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    _write_unit_local_salad_head(
        heads_root,
        "active_head",
        {"reference_source": "active_label_images"},
    )
    prepared = [
        {
            "image_path": str(tmp_path / "image.jpg"),
            "filename": "image.jpg",
            "saved_name": "image.jpg",
            "source_type": "image",
            "frame_index": 0,
            "width": 8,
            "height": 8,
        }
    ]
    monkeypatch.setattr(api, "_data_ingestion_prepare_media", lambda *_args, **_kwargs: list(prepared))
    job = api.DataIngestionJob(
        job_id="di_cancel_before_result",
        kind="analysis",
        request={
            "encoder": "local_salad",
            "salad_head_id": "active_head",
            "reference_source": "active_label_images",
            "candidate_uploads": [{"path": str(tmp_path / "candidate.jpg"), "filename": "candidate.jpg"}],
            "reference_uploads": [{"path": str(tmp_path / "reference.jpg"), "filename": "reference.jpg"}],
        },
    )
    encode_calls = {"count": 0}

    def fake_encode(prepared_rows, **_kwargs):
        encode_calls["count"] += 1
        if encode_calls["count"] == 2:
            job.cancel_event.set()
        values = np.zeros((len(prepared_rows), 3), dtype=np.float32)
        values[:, 0] = 1.0
        return values

    monkeypatch.setattr(api, "_data_ingestion_encode_prepared_images", fake_encode)

    api._run_data_ingestion_analysis_job(job)

    assert job.status == "cancelled"
    assert job.result is None
    assert job.result_path is None
    assert not (jobs_root / job.job_id / "result.json").exists()
    assert not (jobs_root / job.job_id / "embeddings.npz").exists()


def test_data_ingestion_result_rejects_symlinked_result_escape(tmp_path, monkeypatch):
    ingestion_root = tmp_path / "data_ingestion"
    job_root = ingestion_root / "job_escape"
    job_root.mkdir(parents=True)
    outside = tmp_path / "outside_result.json"
    outside.write_text('{"escaped":true}', encoding="utf-8")
    result_link = job_root / "result.json"
    try:
        result_link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    job = api.DataIngestionJob(
        job_id="job_escape",
        status="completed",
        result_path=str(result_link),
    )
    with api.DATA_INGESTION_JOBS_LOCK:
        api.DATA_INGESTION_JOBS.clear()
        api.DATA_INGESTION_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as exc_info:
            api.get_data_ingestion_result(job.job_id)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "result_not_found"
    finally:
        with api.DATA_INGESTION_JOBS_LOCK:
            api.DATA_INGESTION_JOBS.clear()


def test_data_ingestion_result_rejects_symlinked_storage_root(tmp_path, monkeypatch):
    outside_root = tmp_path / "outside_ingestion"
    job_root = outside_root / "job_escape"
    job_root.mkdir(parents=True)
    result_path = job_root / "result.json"
    result_path.write_text('{"escaped":true}', encoding="utf-8")
    ingestion_root = tmp_path / "data_ingestion"
    try:
        ingestion_root.symlink_to(outside_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", ingestion_root)
    job = api.DataIngestionJob(
        job_id="job_escape",
        status="completed",
        result_path=str(ingestion_root / "job_escape" / "result.json"),
    )
    with api.DATA_INGESTION_JOBS_LOCK:
        api.DATA_INGESTION_JOBS.clear()
        api.DATA_INGESTION_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as exc_info:
            api.get_data_ingestion_result(job.job_id)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "data_ingestion_path_invalid"
    finally:
        with api.DATA_INGESTION_JOBS_LOCK:
            api.DATA_INGESTION_JOBS.clear()


def test_data_ingestion_analysis_requires_reference_profile_by_default(tmp_path, monkeypatch):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)
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
    assert not jobs_root.exists()


def test_data_ingestion_analysis_cleans_saved_uploads_if_backend_reference_empty(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    jobs_root = tmp_path / "jobs"
    heads_root = tmp_path / "heads"
    heads_root.mkdir()
    monkeypatch.setattr(api, "DATA_INGESTION_ROOT", jobs_root)
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "empty_dataset",
            "label": "Empty Dataset",
            "dataset_root": str(dataset_root),
            "yolo_layout": "flat",
        },
    )
    _write_unit_local_salad_head(
        heads_root,
        "empty_dataset_head",
        {
            "reference_source": "backend_dataset",
            "reference_dataset_id": "empty_dataset",
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api.create_data_ingestion_analysis_job(
                json.dumps({
                    "reference_source": "backend_dataset",
                    "reference_dataset_id": "empty_dataset",
                    "salad_head_id": "empty_dataset_head",
                    "encoder": "local_salad",
                }),
                [_FakeUpload("candidate.jpg", b"candidate")],
                [],
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "data_ingestion_no_reference_files"
    assert list(jobs_root.iterdir()) == []


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
    assert summary["augmentation_profile"] == api.LOCAL_SALAD_AUGMENTATION_PROFILE
    assert Path(summary["path"]).exists()

    loaded, meta = api._load_local_salad_head(summary["head_id"], device_name="cpu", backend="torch")
    assert isinstance(loaded, LocalSALADHead)
    assert meta["salad_backend"] == "mlx"
    assert meta["encoder_type"] == "dinov3"
    assert meta["encoder_model"] == "unit-dino"
    assert meta["augmentation_profile"] == api.LOCAL_SALAD_AUGMENTATION_PROFILE
    with Image.open(image_a) as opened_a, Image.open(image_b) as opened_b:
        test_patches, test_global = fake_dinov3_tokens(None, None, "cpu", [opened_a, opened_b])
    desc = api._encode_local_salad_head_np(loaded, test_patches, test_global)
    assert desc.shape == (2, summary["descriptor_dim"])
    assert np.allclose(np.linalg.norm(desc, axis=1), np.ones(2), atol=1e-5)


def test_local_salad_training_cancelled_before_head_write_leaves_no_head(tmp_path, monkeypatch):
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

    job = api.DataIngestionJob(
        job_id="salad_cancel_before_head",
        kind="local_salad_train",
        request={
            "train_uploads": prepared,
            "encoder_type": "dinov3",
            "encoder_model": "unit-dino",
            "head_name": "Cancelled Head",
            "local_salad_backend": "torch",
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
    token_calls = {"count": 0}

    monkeypatch.setattr(api, "_data_ingestion_prepare_media", lambda *_args, **_kwargs: list(prepared))
    monkeypatch.setattr(api, "_data_ingestion_get_dinov3", lambda model_name: (DummyDino(), object(), "unit-dino", "cpu"))

    def fake_dinov3_tokens(_model_obj, _processor_obj, _device_name, images):
        token_calls["count"] += 1
        if token_calls["count"] >= 2:
            job.cancel_event.set()
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

    api._run_local_salad_training_job(job)

    assert job.status == "cancelled"
    assert job.result is None
    assert list(heads_root.glob("*.pt")) == []
    assert not (jobs_root / job.job_id / "result.json").exists()


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


def test_list_local_salad_heads_skips_symlink_escape(tmp_path, monkeypatch):
    heads_root = tmp_path / "heads"
    heads_root.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not a head")
    (heads_root / "escape.pt").symlink_to(outside)
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)

    def fail_load(*_args, **_kwargs):
        raise AssertionError("external symlinked head should not be loaded")

    monkeypatch.setattr(api, "load_local_salad_head_file", fail_load)

    assert api._list_local_salad_heads() == []


def test_list_local_salad_heads_rejects_symlinked_root(tmp_path, monkeypatch):
    outside = tmp_path / "outside_heads"
    outside.mkdir()
    (outside / "escape.pt").write_bytes(b"not a head")
    heads_root = tmp_path / "heads"
    try:
        heads_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)

    def fail_load(*_args, **_kwargs):
        raise AssertionError("external symlinked head should not be loaded")

    monkeypatch.setattr(api, "load_local_salad_head_file", fail_load)

    assert api._list_local_salad_heads() == []


def test_list_local_salad_heads_omits_filesystem_paths(tmp_path, monkeypatch):
    heads_root = tmp_path / "heads"
    heads_root.mkdir()
    (heads_root / "profile.pt").write_bytes(b"placeholder")
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)

    def fake_load(_path, device_name="cpu"):
        return None, {
            "label": "Profile",
            "encoder_type": "dinov3",
            "train_image_count": 3,
            "config": {
                "token_dim": 2,
                "num_clusters": 1,
                "cluster_dim": 4,
                "cache_path": str(tmp_path / "cache"),
            },
        }

    monkeypatch.setattr(api, "load_local_salad_head_file", fake_load)

    heads = api._list_local_salad_heads()

    assert len(heads) == 1
    assert heads[0]["id"] == "profile"
    assert "path" not in heads[0]
    assert "cache_path" not in heads[0]["config"]
    assert str(tmp_path) not in json.dumps(heads)


def test_list_local_salad_heads_redacts_unreadable_errors(tmp_path, monkeypatch):
    heads_root = tmp_path / "heads"
    heads_root.mkdir()
    (heads_root / "bad.pt").write_bytes(b"placeholder")
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", heads_root)

    def fake_load(path, device_name="cpu"):
        raise RuntimeError(f"failed to load {path}")

    monkeypatch.setattr(api, "load_local_salad_head_file", fake_load)

    heads = api._list_local_salad_heads()

    assert len(heads) == 1
    assert heads[0]["status"].startswith("unreadable:")
    assert str(tmp_path) not in json.dumps(heads)


def test_local_salad_head_ids_do_not_overwrite_existing_heads(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", tmp_path)
    (tmp_path / "local_salad_head.pt").write_bytes(b"placeholder")

    assert api._unique_local_salad_head_id("local salad head") == "local_salad_head_2"
