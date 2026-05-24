from __future__ import annotations

import asyncio
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import localinferenceapi as api
from services.detectors import (
    _collect_rfdetr_artifacts_impl,
    _collect_yolo_artifacts_impl,
    _copy_tree_within_root,
    _list_rfdetr_runs_impl,
    _list_yolo_runs_impl,
    _load_yolo_active_impl,
    _rfdetr_load_run_meta_impl,
    _rfdetr_prune_run_dir_impl,
    _rfdetr_run_dir_impl,
    _rfdetr_best_checkpoint_impl,
    _rfdetr_prepare_dataset_impl,
    _yolo_load_run_labelmap_impl,
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


class _ImmediateTrainingThread:
    def __init__(self, target=None, args=(), kwargs=None, name="", **_kwargs):
        self._target = target
        self._args = tuple(args or ())
        self._kwargs = dict(kwargs or {})
        self._name = str(name or "")

    def start(self):
        if "monitor" in self._name:
            return
        if self._target:
            self._target(*self._args, **self._kwargs)

    def join(self, *_args, **_kwargs):
        return None


def test_load_yolo_active_ignores_missing_best_path(tmp_path: Path) -> None:
    active_path = tmp_path / "active.json"
    active_path.write_text(
        '{"run_id":"missing","best_path":"/tmp/tator_missing_yolo_best.pt"}',
        encoding="utf-8",
    )

    assert _load_yolo_active_impl(active_path) == {}


def test_raw_detector_active_loader_ignores_symlinked_marker(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_active.json"
    outside.write_text('{"run_id":"active_run"}', encoding="utf-8")
    active_path = tmp_path / "models" / "yolo" / "active.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        active_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert api._load_detector_active_payload_raw(active_path) == {}
    assert active_path.is_symlink()
    assert outside.read_text(encoding="utf-8") == '{"run_id":"active_run"}'


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


def test_rfdetr_prepare_dataset_rejects_split_symlink_escape(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    outside = tmp_path / "outside_train"
    outside.mkdir(parents=True)
    (outside / "_annotations.coco.json").write_text(
        '{"images":[],"annotations":[],"categories":[]}', encoding="utf-8"
    )
    dataset_root.mkdir()
    try:
        (dataset_root / "train").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(RuntimeError, match="rfdetr_dataset_path_invalid"):
        _rfdetr_prepare_dataset_impl(
            dataset_root,
            tmp_path / "run",
            str(outside / "_annotations.coco.json"),
            str(outside / "_annotations.coco.json"),
            remap_ids_fn=lambda _src, _dest: None,
        )


def test_yolo_artifact_collection_ignores_symlink_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside_best.pt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (run_dir / "best.pt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    artifacts = _collect_yolo_artifacts_impl(run_dir, meta_name="run.json")

    assert artifacts["best_pt"] is False
    assert artifacts["metrics_json"] is True


def test_rfdetr_artifact_collection_ignores_symlink_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside_checkpoint.pth"
    outside.write_text("secret", encoding="utf-8")
    try:
        (run_dir / "checkpoint_best_total.pth").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    (run_dir / "results.json").write_text("{}", encoding="utf-8")

    artifacts = _collect_rfdetr_artifacts_impl(run_dir, meta_name="run.json")

    assert artifacts["best_total"] is False
    assert artifacts["results_json"] is True


def test_yolo_run_labelmap_ignores_symlinked_data_yaml_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside_data.yaml"
    outside.write_text("names: [escaped]\n", encoding="utf-8")
    try:
        (run_dir / "data.yaml").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    labels = _yolo_load_run_labelmap_impl(
        run_dir,
        yolo_load_labelmap_fn=lambda _path: [],
        yaml_load_fn=lambda text: {"names": ["escaped"]} if text else {},
    )

    assert labels == []


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


def test_yolo_cleanup_run_dir_unlinks_in_root_symlink_without_target_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "yolo_runs"
    root.mkdir()
    target = root / "target_run"
    target.mkdir()
    payload = target / "best.pt"
    payload.write_bytes(b"keep")
    in_root_link = root / "failed_run"
    outside_link = tmp_path / "outside_link"
    parent_target = root / "parent_target"
    parent_target.mkdir()
    run_through_parent_link = parent_target / "failed_parent_run"
    run_through_parent_link.mkdir()
    parent_payload = run_through_parent_link / "best.pt"
    parent_payload.write_bytes(b"parent")
    parent_link = root / "linked_parent"
    try:
        in_root_link.symlink_to(target, target_is_directory=True)
        outside_link.symlink_to(target, target_is_directory=True)
        parent_link.symlink_to(parent_target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "YOLO_JOB_ROOT", root)

    api._cleanup_yolo_run_dir(in_root_link)
    api._cleanup_yolo_run_dir(outside_link)
    api._cleanup_yolo_run_dir(parent_link / "failed_parent_run")

    assert not in_root_link.exists()
    assert outside_link.is_symlink()
    assert payload.read_bytes() == b"keep"
    assert parent_payload.read_bytes() == b"parent"


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


def test_rfdetr_cleanup_run_dir_unlinks_in_root_symlink_without_target_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "rfdetr_runs"
    root.mkdir()
    target = root / "target_run"
    target.mkdir()
    payload = target / "checkpoint_best_total.pth"
    payload.write_bytes(b"keep")
    in_root_link = root / "failed_run"
    outside_link = tmp_path / "outside_link"
    parent_target = root / "parent_target"
    parent_target.mkdir()
    run_through_parent_link = parent_target / "failed_parent_run"
    run_through_parent_link.mkdir()
    parent_payload = run_through_parent_link / "checkpoint_best_total.pth"
    parent_payload.write_bytes(b"parent")
    parent_link = root / "linked_parent"
    try:
        in_root_link.symlink_to(target, target_is_directory=True)
        outside_link.symlink_to(target, target_is_directory=True)
        parent_link.symlink_to(parent_target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "RFDETR_JOB_ROOT", root)

    api._cleanup_rfdetr_run_dir(in_root_link)
    api._cleanup_rfdetr_run_dir(outside_link)
    api._cleanup_rfdetr_run_dir(parent_link / "failed_parent_run")

    assert not in_root_link.exists()
    assert outside_link.is_symlink()
    assert payload.read_bytes() == b"keep"
    assert parent_payload.read_bytes() == b"parent"


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


def test_yolo_training_late_cancel_skips_artifact_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "late_cancel_yolo"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    labelmap_path = dataset_root / "labelmap.txt"
    labelmap_path.write_text("target\n", encoding="utf-8")
    yolo_root = tmp_path / "yolo_runs"
    job = api.YoloTrainingJob(
        job_id=run_id,
        config={
            "dataset": {
                "yolo_ready": True,
                "dataset_root": str(dataset_root),
                "yolo_layout": "flat",
                "yolo_labelmap_path": str(labelmap_path),
                "task": "detect",
            },
            "task": "detect",
            "variant": "yolov8n",
            "epochs": 1,
            "device_resolution": {
                "device_arg": "cpu",
                "device_label": "CPU",
                "resolved_accelerator": "cpu",
            },
        },
    )

    class FakeYOLO:
        def __init__(self, _model_source):
            pass

        def train(self, **kwargs):
            train_weights = Path(kwargs["project"]) / "train" / "weights"
            train_weights.mkdir(parents=True, exist_ok=True)
            (train_weights / "best.pt").write_text("weights", encoding="utf-8")
            job.cancel_event.set()
            return SimpleNamespace(metrics={"mAP50": 1.0})

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(api, "YOLO_JOB_ROOT", yolo_root)
    monkeypatch.setattr(api.threading, "Thread", _ImmediateTrainingThread)
    monkeypatch.setattr(api, "_prepare_for_training_impl", lambda **_kwargs: None)
    monkeypatch.setattr(api, "_finalize_training_environment_impl", lambda **_kwargs: None)
    monkeypatch.setattr(api, "_yolo_monitor_training_impl", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        api,
        "_yolo_write_data_yaml_impl",
        lambda run_dir, *_args, **_kwargs: run_dir / "data.yaml",
    )
    monkeypatch.setattr(
        api,
        "_yolo_resolve_model_source_impl",
        lambda *_args, **_kwargs: ("weights", "yolov8n.pt"),
    )

    def fail_after_cancel(*_args, **_kwargs):
        raise AssertionError("cancelled YOLO job should not publish artifacts")

    monkeypatch.setattr(api, "_copy2_if_different", fail_after_cancel)
    monkeypatch.setattr(api, "_write_detector_json_atomic", fail_after_cancel)
    monkeypatch.setattr(api, "_yolo_prune_run_dir_impl", fail_after_cancel)

    api._start_yolo_training_worker(job)

    run_dir = yolo_root / run_id
    meta = json.loads((run_dir / api.YOLO_RUN_META_NAME).read_text(encoding="utf-8"))
    assert job.status == "cancelled"
    assert job.result is None
    assert meta["status"] == "cancelled"
    assert meta.get("result") is None
    assert not (run_dir / "best.pt").exists()
    assert (run_dir / "train" / "weights" / "best.pt").exists()


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


def test_rfdetr_training_late_cancel_skips_artifact_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "late_cancel_rfdetr"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    coco_path = dataset_root / "train.json"
    coco_path.write_text('{"images":[],"annotations":[],"categories":[]}', encoding="utf-8")
    rfdetr_root = tmp_path / "rfdetr_runs"
    job = api.RfDetrTrainingJob(
        job_id=run_id,
        config={
            "dataset": {
                "dataset_root": str(dataset_root),
                "coco_train_json": str(coco_path),
                "coco_val_json": str(coco_path),
                "task": "detect",
            },
            "task": "detect",
            "variant": "rfdetr-nano",
            "epochs": 1,
            "device_resolution": {
                "device_arg": "cpu",
                "device_label": "CPU",
                "devices": [],
                "resolved_accelerator": "cpu",
            },
        },
    )

    class FakeRfDetr:
        def __init__(self, **_kwargs):
            self.callbacks = {"on_fit_epoch_end": []}
            self.model = SimpleNamespace(request_early_stop=lambda: None)

        def train(self, **kwargs):
            run_dir = Path(kwargs["output_dir"])
            (run_dir / "checkpoint_best_total.pth").write_text("weights", encoding="utf-8")
            job.cancel_event.set()

    fake_rfdetr = ModuleType("rfdetr")
    for name in (
        "RFDETRBase",
        "RFDETRLarge",
        "RFDETRNano",
        "RFDETRSmall",
        "RFDETRMedium",
        "RFDETRSegPreview",
    ):
        setattr(fake_rfdetr, name, FakeRfDetr)

    monkeypatch.setitem(sys.modules, "rfdetr", fake_rfdetr)
    monkeypatch.setattr(api, "RFDETR_JOB_ROOT", rfdetr_root)
    monkeypatch.setattr(api.threading, "Thread", _ImmediateTrainingThread)
    monkeypatch.setattr(api, "_prepare_for_training_impl", lambda **_kwargs: None)
    monkeypatch.setattr(api, "_finalize_training_environment_impl", lambda **_kwargs: None)
    monkeypatch.setattr(api, "_rfdetr_load_labelmap_impl", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        api,
        "_rfdetr_prepare_dataset_impl",
        lambda _dataset_root, run_dir, *_args, **_kwargs: run_dir / "dataset",
    )
    monkeypatch.setattr(api, "_rfdetr_install_augmentations_impl", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api, "_rfdetr_restore_augmentations_impl", lambda *_args, **_kwargs: None)

    def fail_after_cancel(*_args, **_kwargs):
        raise AssertionError("cancelled RF-DETR job should not publish artifacts")

    monkeypatch.setattr(api, "_rfdetr_best_checkpoint_impl", fail_after_cancel)
    monkeypatch.setattr(api, "_rfdetr_prune_run_dir_impl", fail_after_cancel)

    api._start_rfdetr_training_worker(job)

    run_dir = rfdetr_root / run_id
    meta = json.loads((run_dir / api.RFDETR_RUN_META_NAME).read_text(encoding="utf-8"))
    assert job.status == "cancelled"
    assert job.result is None
    assert meta["status"] == "cancelled"
    assert meta.get("result") is None
    assert not (run_dir / "metrics_series.json").exists()


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
