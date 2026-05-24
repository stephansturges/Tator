import asyncio
import json
import os
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


def test_local_salad_head_ids_do_not_overwrite_existing_heads(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "LOCAL_SALAD_HEAD_ROOT", tmp_path)
    (tmp_path / "local_salad_head.pt").write_bytes(b"placeholder")

    assert api._unique_local_salad_head_id("local salad head") == "local_salad_head_2"
