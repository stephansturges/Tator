from __future__ import annotations

import asyncio
import io
import json
import zipfile
from pathlib import Path

import pytest

import localinferenceapi as api
from services.detectors import (
    _copy_tree_within_root,
    _list_rfdetr_runs_impl,
    _list_yolo_runs_impl,
    _load_yolo_active_impl,
    _rfdetr_load_run_meta_impl,
    _rfdetr_prune_run_dir_impl,
    _rfdetr_run_dir_impl,
    _rfdetr_best_checkpoint_impl,
    _rfdetr_prepare_dataset_impl,
    _yolo_load_run_meta_impl,
    _yolo_prune_run_dir_impl,
    _yolo_run_dir_impl,
)


async def _stream_body(response) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return b"".join(chunks)


def _zip_names(response) -> set[str]:
    raw = asyncio.run(_stream_body(response))
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        return set(zf.namelist())


def test_load_yolo_active_ignores_missing_best_path(tmp_path: Path) -> None:
    active_path = tmp_path / "active.json"
    active_path.write_text(
        '{"run_id":"missing","best_path":"/tmp/tator_missing_yolo_best.pt"}',
        encoding="utf-8",
    )

    assert _load_yolo_active_impl(active_path) == {}


def test_load_yolo_active_ignores_best_symlink_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_text("secret", encoding="utf-8")
    best_path = run_dir / "best.pt"
    try:
        best_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    active_path = tmp_path / "active.json"
    active_path.write_text(
        '{"run_id":"run","best_path":"' + str(best_path) + '"}',
        encoding="utf-8",
    )

    assert _load_yolo_active_impl(active_path) == {}


def test_yolo_load_run_meta_skips_symlink_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside_meta.json"
    outside.write_text('{"job_id":"escaped"}', encoding="utf-8")
    try:
        (run_dir / "run_meta.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert _yolo_load_run_meta_impl(run_dir, meta_name="run_meta.json") == {}


def test_rfdetr_load_run_meta_skips_symlink_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside_meta.json"
    outside.write_text('{"job_id":"escaped"}', encoding="utf-8")
    try:
        (run_dir / "run_meta.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert _rfdetr_load_run_meta_impl(run_dir, meta_name="run_meta.json") == {}


def test_list_yolo_runs_skips_symlinked_run_dir_escape(tmp_path: Path) -> None:
    job_root = tmp_path / "yolo_runs"
    job_root.mkdir()
    outside = tmp_path / "outside_run"
    outside.mkdir()
    (outside / "run_meta.json").write_text('{"job_id":"escaped"}', encoding="utf-8")
    try:
        (job_root / "linked_run").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    runs = _list_yolo_runs_impl(
        job_root=job_root,
        dataset_cache_root=job_root / "datasets",
        active_payload={},
        load_meta_fn=lambda run_dir: _yolo_load_run_meta_impl(
            run_dir, meta_name="run_meta.json"
        ),
        collect_artifacts_fn=lambda _run_dir: {},
        meta_name="run_meta.json",
    )

    assert runs == []


def test_list_rfdetr_runs_skips_symlinked_run_dir_escape(tmp_path: Path) -> None:
    job_root = tmp_path / "rfdetr_runs"
    job_root.mkdir()
    outside = tmp_path / "outside_run"
    outside.mkdir()
    (outside / "run_meta.json").write_text('{"job_id":"escaped"}', encoding="utf-8")
    try:
        (job_root / "linked_run").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    runs = _list_rfdetr_runs_impl(
        job_root=job_root,
        active_payload={},
        load_meta_fn=lambda run_dir: _rfdetr_load_run_meta_impl(
            run_dir, meta_name="run_meta.json"
        ),
        collect_artifacts_fn=lambda _run_dir: {},
        meta_name="run_meta.json",
    )

    assert runs == []


def test_yolo_run_dir_rejects_symlinked_job_root_before_create(tmp_path: Path) -> None:
    outside = tmp_path / "outside_yolo_runs"
    outside.mkdir()
    job_root = tmp_path / "yolo_runs"
    try:
        job_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        _yolo_run_dir_impl(
            "run1",
            create=True,
            job_root=job_root,
            sanitize_fn=lambda value: value,
            http_exception_cls=api.HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"
    assert list(outside.iterdir()) == []


def test_rfdetr_run_dir_rejects_symlinked_job_root_before_create(tmp_path: Path) -> None:
    outside = tmp_path / "outside_rfdetr_runs"
    outside.mkdir()
    job_root = tmp_path / "rfdetr_runs"
    try:
        job_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        _rfdetr_run_dir_impl(
            "run1",
            create=True,
            job_root=job_root,
            sanitize_fn=lambda value: value,
            http_exception_cls=api.HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"
    assert list(outside.iterdir()) == []


def test_yolo_run_dir_rejects_symlinked_job_parent_before_create(tmp_path: Path) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        _yolo_run_dir_impl(
            "run1",
            create=True,
            job_root=linked_parent / "yolo_runs",
            sanitize_fn=lambda value: value,
            http_exception_cls=api.HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"
    assert list(outside.iterdir()) == []


def test_rfdetr_run_dir_rejects_symlinked_job_parent_before_create(tmp_path: Path) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        _rfdetr_run_dir_impl(
            "run1",
            create=True,
            job_root=linked_parent / "rfdetr_runs",
            sanitize_fn=lambda value: value,
            http_exception_cls=api.HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"
    assert list(outside.iterdir()) == []


def test_download_yolo_run_skips_symlink_keep_file_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_root = tmp_path / "yolo_runs"
    run_dir = job_root / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.pt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (run_dir / "best.pt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    (run_dir / "labelmap.txt").write_text("target\n", encoding="utf-8")

    monkeypatch.setattr(api, "YOLO_JOB_ROOT", job_root)

    names = _zip_names(api.download_yolo_run("run1"))

    assert "labelmap.txt" in names
    assert "best.pt" not in names


def test_yolo_detector_runtime_rejects_best_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_root = tmp_path / "yolo_runs"
    run_dir = job_root / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.pt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (run_dir / "best.pt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "YOLO_JOB_ROOT", job_root)

    with pytest.raises(api.HTTPException) as exc:
        api._ensure_yolo_inference_runtime_for_detector("run1")

    assert exc.value.status_code == 412
    assert exc.value.detail == "yolo_best_missing"


def test_set_yolo_active_rejects_symlinked_best_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_root = tmp_path / "yolo_runs"
    run_dir = job_root / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.pt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (run_dir / "best.pt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "YOLO_JOB_ROOT", job_root)

    with pytest.raises(api.HTTPException) as exc:
        api.set_yolo_active(api.YoloActiveRequest(run_id="run1"))

    assert exc.value.status_code == 412
    assert exc.value.detail == "yolo_best_missing"


def test_set_yolo_active_omits_symlinked_labelmap_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_root = tmp_path / "yolo_runs"
    run_dir = job_root / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best.pt").write_text("weights", encoding="utf-8")
    outside = tmp_path / "outside_labelmap.txt"
    outside.write_text("escaped\n", encoding="utf-8")
    try:
        (run_dir / "labelmap.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    active_path = tmp_path / "models" / "yolo" / "active.json"
    monkeypatch.setattr(api, "YOLO_JOB_ROOT", job_root)
    monkeypatch.setattr(api, "YOLO_ACTIVE_PATH", active_path)

    payload = api.set_yolo_active(api.YoloActiveRequest(run_id="run1"))

    assert payload["labelmap_path"] is None
    assert json.loads(active_path.read_text(encoding="utf-8"))["labelmap_path"] is None


def test_download_rfdetr_run_skips_symlink_keep_file_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_root = tmp_path / "rfdetr_runs"
    run_dir = job_root / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.pth"
    outside.write_text("secret", encoding="utf-8")
    try:
        (run_dir / "checkpoint_best_total.pth").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    (run_dir / "labelmap.txt").write_text("target\n", encoding="utf-8")

    monkeypatch.setattr(api, "RFDETR_JOB_ROOT", job_root)

    names = _zip_names(api.download_rfdetr_run("run1"))

    assert "labelmap.txt" in names
    assert "checkpoint_best_total.pth" not in names


def test_rfdetr_prepare_dataset_copy_fallback_skips_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    train_root = dataset_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    (train_root / "safe.jpg").write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside.jpg"
    outside.write_text("secret", encoding="utf-8")
    try:
        (train_root / "escape.jpg").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    def _disable_symlink(*_args, **_kwargs):
        raise OSError("symlink disabled")

    def _write_remapped(_src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('{"images":[],"annotations":[],"categories":[]}', encoding="utf-8")

    monkeypatch.setattr(Path, "symlink_to", _disable_symlink)

    prepared = _rfdetr_prepare_dataset_impl(
        dataset_root,
        tmp_path / "run",
        str(train_root / "_annotations.coco.json"),
        str(train_root / "_annotations.coco.json"),
        remap_ids_fn=_write_remapped,
    )

    assert (prepared / "train" / "safe.jpg").read_text(encoding="utf-8") == "safe"
    assert not (prepared / "train" / "escape.jpg").exists()


def test_detector_copy_tree_replaces_symlinked_dest_root_without_target_write(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "safe.txt").write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "dest"
    try:
        dest.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _copy_tree_within_root(src, dest)

    assert not dest.is_symlink()
    assert (dest / "safe.txt").read_text(encoding="utf-8") == "safe"
    assert not (outside / "safe.txt").exists()


def test_detector_copy_tree_replaces_symlinked_file_destination_without_target_write(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "safe.txt").write_text("safe", encoding="utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("external", encoding="utf-8")
    try:
        (dest / "safe.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _copy_tree_within_root(src, dest)

    assert not (dest / "safe.txt").is_symlink()
    assert (dest / "safe.txt").read_text(encoding="utf-8") == "safe"
    assert outside.read_text(encoding="utf-8") == "external"


def test_detector_copy_tree_rejects_symlinked_dest_parent_without_target_write(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "safe.txt").write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(RuntimeError, match="detector_path_invalid"):
        _copy_tree_within_root(src, linked_parent / "dest")

    assert list(outside.iterdir()) == []


def test_detector_copy_tree_rejects_internal_symlinked_dest_parent_without_target_write(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    (src / "linked_parent").mkdir(parents=True)
    (src / "linked_parent" / "safe.txt").write_text("safe", encoding="utf-8")
    dest = tmp_path / "dest"
    actual = dest / "actual"
    actual.mkdir(parents=True)
    try:
        (dest / "linked_parent").symlink_to(actual, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(RuntimeError, match="detector_path_invalid"):
        _copy_tree_within_root(src, dest)

    assert list(actual.iterdir()) == []


def test_yolo_prune_run_dir_unlinks_symlink_directory_without_target_delete(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "payload.bin").write_bytes(b"external")
    try:
        (run_dir / "linked").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    result = _yolo_prune_run_dir_impl(
        run_dir,
        keep_files=set(),
        keep_files_default=[],
        dir_size_fn=lambda _path: 999,
        meta_name="meta.json",
    )

    assert result["deleted"] == ["linked"]
    assert result["freed_bytes"] == 0
    assert not (run_dir / "linked").exists()
    assert (outside / "payload.bin").read_bytes() == b"external"


def test_rfdetr_prune_run_dir_unlinks_symlink_directory_without_target_delete(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "payload.bin").write_bytes(b"external")
    try:
        (run_dir / "linked").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    result = _rfdetr_prune_run_dir_impl(
        run_dir,
        keep_files=set(),
        keep_files_default=[],
        dir_size_fn=lambda _path: 999,
    )

    assert result["deleted"] == ["linked"]
    assert result["freed_bytes"] == 0
    assert not (run_dir / "linked").exists()
    assert (outside / "payload.bin").read_bytes() == b"external"


def test_rfdetr_detector_runtime_rejects_best_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_root = tmp_path / "rfdetr_runs"
    run_dir = job_root / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.pth"
    outside.write_text("secret", encoding="utf-8")
    try:
        (run_dir / "checkpoint_best_total.pth").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "RFDETR_JOB_ROOT", job_root)

    assert _rfdetr_best_checkpoint_impl(run_dir) is None
    with pytest.raises(api.HTTPException) as exc:
        api._ensure_rfdetr_inference_runtime_for_detector("run1")

    assert exc.value.status_code == 412
    assert exc.value.detail == "rfdetr_best_missing"


def test_delete_yolo_run_clears_active_and_cached_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "active_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_path.write_text("weights", encoding="utf-8")
    (run_dir / "labelmap.txt").write_text("target\n", encoding="utf-8")
    active_path = tmp_path / "models" / "yolo" / "active.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(
        (
            '{"run_id":"active_run","best_path":"'
            + str(best_path)
            + '","labelmap_path":"'
            + str(run_dir / "labelmap.txt")
            + '","task":"detect"}'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "YOLO_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api, "YOLO_ACTIVE_PATH", active_path)
    api._set_yolo_infer_state(object(), str(best_path), ["target"], "detect")

    try:
        out = api.delete_yolo_run(run_id)
        assert out == {"status": "deleted", "run_id": run_id}
        assert not run_dir.exists()
        assert not active_path.exists()
        assert api.yolo_infer_model is None
        assert api.yolo_infer_path is None
        assert api.yolo_infer_labelmap == []
        assert api.yolo_infer_task is None
    finally:
        api._set_yolo_infer_state(None, None, [], None)


def test_delete_yolo_run_blocks_active_training_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "active_train_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best.pt").write_text("weights", encoding="utf-8")
    monkeypatch.setattr(api, "YOLO_JOB_ROOT", tmp_path)
    job = api.YoloTrainingJob(
        job_id=run_id,
        status="running",
        config={"paths": {"run_dir": str(run_dir)}},
    )
    with api.YOLO_TRAINING_JOBS_LOCK:
        api.YOLO_TRAINING_JOBS.clear()
        api.YOLO_TRAINING_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as exc:
            api.delete_yolo_run(run_id)
        assert exc.value.status_code == 409
        assert exc.value.detail == "yolo_run_delete_blocked_active_jobs:yolo_training"
        assert run_dir.exists()
    finally:
        with api.YOLO_TRAINING_JOBS_LOCK:
            api.YOLO_TRAINING_JOBS.clear()


def test_delete_yolo_run_allows_blocked_nonrunning_training_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "blocked_train_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "YOLO_JOB_ROOT", tmp_path)
    job = api.YoloTrainingJob(
        job_id=run_id,
        status="blocked",
        config={"paths": {"run_dir": str(run_dir)}},
    )
    with api.YOLO_TRAINING_JOBS_LOCK:
        api.YOLO_TRAINING_JOBS.clear()
        api.YOLO_TRAINING_JOBS[job.job_id] = job

    try:
        assert api.delete_yolo_run(run_id) == {"status": "deleted", "run_id": run_id}
        assert not run_dir.exists()
    finally:
        with api.YOLO_TRAINING_JOBS_LOCK:
            api.YOLO_TRAINING_JOBS.clear()


def test_delete_yolo_run_blocks_active_head_graft_base_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "base_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "YOLO_JOB_ROOT", tmp_path)
    job = api.YoloHeadGraftJob(
        job_id="graft_job",
        status="queued",
        config={"base_run_id": run_id},
    )
    with api.YOLO_HEAD_GRAFT_JOBS_LOCK:
        api.YOLO_HEAD_GRAFT_JOBS.clear()
        api.YOLO_HEAD_GRAFT_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as exc:
            api.delete_yolo_run(run_id)
        assert exc.value.status_code == 409
        assert exc.value.detail == "yolo_run_delete_blocked_active_jobs:yolo_head_graft"
        assert run_dir.exists()
    finally:
        with api.YOLO_HEAD_GRAFT_JOBS_LOCK:
            api.YOLO_HEAD_GRAFT_JOBS.clear()


def test_delete_yolo_run_clears_corrupt_active_marker_inside_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "corrupt_active_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    active_path = tmp_path / "models" / "yolo" / "active.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    missing_best = run_dir / "best.pt"
    active_path.write_text(
        '{"run_id":"corrupt_active_run","best_path":"' + str(missing_best) + '"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "YOLO_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api, "YOLO_ACTIVE_PATH", active_path)

    out = api.delete_yolo_run(run_id)

    assert out == {"status": "deleted", "run_id": run_id}
    assert not active_path.exists()


def test_delete_rfdetr_run_clears_active_and_cached_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "active_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "checkpoint_best_total.pth"
    best_path.write_text("weights", encoding="utf-8")
    (run_dir / "labelmap.txt").write_text("target\n", encoding="utf-8")
    active_path = tmp_path / "models" / "rfdetr" / "active.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(
        (
            '{"run_id":"active_run","best_path":"'
            + str(best_path)
            + '","labelmap_path":"'
            + str(run_dir / "labelmap.txt")
            + '","task":"detect","variant":"rfdetr-nano"}'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "RFDETR_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api, "RFDETR_ACTIVE_PATH", active_path)
    api._set_rfdetr_infer_state(object(), str(best_path), ["target"], "detect", "rfdetr-nano")

    try:
        out = api.delete_rfdetr_run(run_id)
        assert out == {"status": "deleted", "run_id": run_id}
        assert not run_dir.exists()
        assert not active_path.exists()
        assert api.rfdetr_infer_model is None
        assert api.rfdetr_infer_path is None
        assert api.rfdetr_infer_labelmap == []
        assert api.rfdetr_infer_task is None
        assert api.rfdetr_infer_variant is None
    finally:
        api._set_rfdetr_infer_state(None, None, [], None, None)


def test_delete_rfdetr_run_blocks_active_training_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "active_rfdetr_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "RFDETR_JOB_ROOT", tmp_path)
    job = api.RfDetrTrainingJob(
        job_id=run_id,
        status="queued",
        config={"paths": {"run_dir": str(run_dir)}},
    )
    with api.RFDETR_TRAINING_JOBS_LOCK:
        api.RFDETR_TRAINING_JOBS.clear()
        api.RFDETR_TRAINING_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as exc:
            api.delete_rfdetr_run(run_id)
        assert exc.value.status_code == 409
        assert exc.value.detail == "rfdetr_run_delete_blocked_active_jobs:rfdetr_training"
        assert run_dir.exists()
    finally:
        with api.RFDETR_TRAINING_JOBS_LOCK:
            api.RFDETR_TRAINING_JOBS.clear()


def test_delete_rfdetr_run_clears_corrupt_active_marker_inside_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "corrupt_active_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    active_path = tmp_path / "models" / "rfdetr" / "active.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    missing_best = run_dir / "checkpoint_best_total.pth"
    active_path.write_text(
        '{"run_id":"corrupt_active_run","best_path":"' + str(missing_best) + '"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "RFDETR_JOB_ROOT", tmp_path)
    monkeypatch.setattr(api, "RFDETR_ACTIVE_PATH", active_path)

    out = api.delete_rfdetr_run(run_id)

    assert out == {"status": "deleted", "run_id": run_id}
    assert not active_path.exists()
