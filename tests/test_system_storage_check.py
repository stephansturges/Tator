from __future__ import annotations

from pathlib import Path

import pytest

import localinferenceapi as api


_STORAGE_ROOT_ATTRS = {
    "uploads": "UPLOAD_ROOT",
    "dataset_registry": "DATASET_REGISTRY_ROOT",
    "sam3_datasets": "SAM3_DATASET_ROOT",
    "qwen_datasets": "QWEN_DATASET_ROOT",
    "sam3_runs": "SAM3_JOB_ROOT",
    "qwen_runs": "QWEN_JOB_ROOT",
    "yolo_runs": "YOLO_JOB_ROOT",
    "rfdetr_runs": "RFDETR_JOB_ROOT",
    "calibration_jobs": "CALIBRATION_ROOT",
    "calibration_cache": "CALIBRATION_CACHE_ROOT",
    "prepass_recipes": "PREPASS_RECIPE_ROOT",
    "clip_uploads": "CLIP_DATASET_UPLOAD_ROOT",
    "dataset_uploads": "DATASET_UPLOAD_ROOT",
}


def _patch_storage_roots(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for name, attr in _STORAGE_ROOT_ATTRS.items():
        path = root / name
        monkeypatch.setattr(api, attr, path)
        roots[name] = path
    return roots


def test_storage_check_passes_for_regular_roots_and_cleans_probe_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _patch_storage_roots(monkeypatch, tmp_path / "storage")

    payload = api._storage_check_payload()

    assert payload["ok"] is True
    assert {entry["name"] for entry in payload["roots"]} == set(_STORAGE_ROOT_ATTRS)
    assert all(entry["ok"] is True for entry in payload["roots"])
    for root in roots.values():
        assert root.is_dir()
        assert not list(root.glob(".write_test_*"))


def test_storage_check_rejects_symlink_root_without_writing_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _patch_storage_roots(monkeypatch, tmp_path / "storage")
    outside = tmp_path / "outside_calibration"
    outside.mkdir()
    roots["calibration_jobs"].parent.mkdir(parents=True, exist_ok=True)
    try:
        roots["calibration_jobs"].symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    payload = api._storage_check_payload()
    entries = {entry["name"]: entry for entry in payload["roots"]}

    assert payload["ok"] is False
    assert entries["calibration_jobs"]["ok"] is False
    assert "storage_root_symlink" in str(entries["calibration_jobs"]["error"])
    assert list(outside.iterdir()) == []


def test_storage_check_rejects_symlinked_root_parent_without_writing_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _patch_storage_roots(monkeypatch, tmp_path / "storage")
    outside = tmp_path / "outside_calibration"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CALIBRATION_ROOT", linked_parent / "calibration_jobs")

    payload = api._storage_check_payload()
    entries = {entry["name"]: entry for entry in payload["roots"]}

    assert roots["calibration_jobs"] == tmp_path / "storage" / "calibration_jobs"
    assert payload["ok"] is False
    assert entries["calibration_jobs"]["ok"] is False
    assert "storage_root_symlink" in str(entries["calibration_jobs"]["error"])
    assert list(outside.iterdir()) == []
