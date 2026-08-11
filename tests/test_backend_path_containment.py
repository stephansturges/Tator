from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from services.classifier import _validate_clip_dataset_impl
from services.datasets import _resolve_dataset_legacy_impl
from services.detectors import _rfdetr_run_dir_impl, _yolo_run_dir_impl
import services.prompt_helper_presets as prompt_helper_presets_service
from services.prompt_helper_presets import (
    _list_prompt_helper_presets_impl,
    _load_prompt_helper_preset_impl,
    _save_prompt_helper_preset_impl,
)
from services.sam3_runs import _run_dir_for_request_impl
from utils.datasets import _iter_yolo_images
from utils.io import _compute_dir_signature, _dir_size_bytes, _sanitize_yolo_run_id


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def test_legacy_dataset_resolution_rejects_sibling_prefix_escape(tmp_path: Path) -> None:
    qwen_root = tmp_path / "qwen"
    sam3_root = tmp_path / "sam3"
    registry_root = tmp_path / "registry"
    escaped = tmp_path / "qwen_evil" / "dataset"
    escaped.mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        _resolve_dataset_legacy_impl(
            "../qwen_evil/dataset",
            qwen_root=qwen_root,
            sam3_root=sam3_root,
            registry_root=registry_root,
            http_exception_cls=HTTPException,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "sam3_dataset_not_found"


def test_sam3_run_lookup_rejects_sibling_prefix_escape(tmp_path: Path) -> None:
    job_root = tmp_path / "sam3_runs"
    escaped = tmp_path / "sam3_runs_evil" / "run1"
    escaped.mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        _run_dir_for_request_impl(
            run_id="../sam3_runs_evil/run1",
            variant="sam3",
            job_root=job_root,
            http_exception_cls=HTTPException,
            http_400=400,
            http_404=404,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"


def test_sam3_run_lookup_rejects_symlinked_job_root_without_target_delete(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_sam3_runs"
    (outside / "run1").mkdir(parents=True)
    (outside / "run1" / "payload.bin").write_bytes(b"target")
    job_root = tmp_path / "sam3_runs"
    try:
        job_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _run_dir_for_request_impl(
            run_id="run1",
            variant="sam3",
            job_root=job_root,
            http_exception_cls=HTTPException,
            http_400=400,
            http_404=404,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"
    assert (outside / "run1" / "payload.bin").read_bytes() == b"target"


def test_sam3_run_lookup_rejects_symlinked_job_root_parent_without_target_delete(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    (outside / "sam3_runs" / "run1").mkdir(parents=True)
    (outside / "sam3_runs" / "run1" / "payload.bin").write_bytes(b"target")
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _run_dir_for_request_impl(
            run_id="run1",
            variant="sam3",
            job_root=linked_parent / "sam3_runs",
            http_exception_cls=HTTPException,
            http_400=400,
            http_404=404,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"
    assert (outside / "sam3_runs" / "run1" / "payload.bin").read_bytes() == b"target"


def test_yolo_run_lookup_rejects_normalized_alias(tmp_path: Path) -> None:
    job_root = tmp_path / "yolo_runs"
    (job_root / "run1").mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        _yolo_run_dir_impl(
            "../run1",
            create=False,
            job_root=job_root,
            sanitize_fn=_sanitize_yolo_run_id,
            http_exception_cls=HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"


def test_yolo_run_lookup_rejects_blank_id(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _yolo_run_dir_impl(
            "",
            create=False,
            job_root=tmp_path / "yolo_runs",
            sanitize_fn=_sanitize_yolo_run_id,
            http_exception_cls=HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"


def test_list_prompt_helper_presets_skips_symlink_escape(tmp_path: Path) -> None:
    presets_root = tmp_path / "presets"
    presets_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"id":"outside","created_at":1}', encoding="utf-8")
    try:
        (presets_root / "outside.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert _list_prompt_helper_presets_impl(presets_root=presets_root) == []


def test_list_prompt_helper_presets_skips_symlinked_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside_presets"
    outside.mkdir()
    (outside / "phset_escape.json").write_text('{"id":"escape","created_at":1}', encoding="utf-8")
    presets_root = tmp_path / "presets"
    try:
        presets_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert _list_prompt_helper_presets_impl(presets_root=presets_root) == []


def test_list_prompt_helper_presets_skips_symlinked_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside_parent"
    presets_dir = outside / "presets"
    presets_dir.mkdir(parents=True)
    (presets_dir / "phset_escape.json").write_text('{"id":"escape","created_at":1}', encoding="utf-8")
    presets_parent = tmp_path / "linked_parent"
    try:
        presets_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert _list_prompt_helper_presets_impl(presets_root=presets_parent / "presets") == []


def test_load_prompt_helper_preset_rejects_symlinked_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside_presets"
    outside.mkdir()
    (outside / "phset_escape.json").write_text('{"id":"escape","created_at":1}', encoding="utf-8")
    presets_root = tmp_path / "presets"
    try:
        presets_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _load_prompt_helper_preset_impl(
            "phset_escape",
            presets_root=presets_root,
            path_is_within_root_fn=_path_within_root,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "prompt_helper_preset_not_found"


def test_save_prompt_helper_preset_rejects_symlinked_root_without_write(tmp_path: Path) -> None:
    outside = tmp_path / "outside_presets"
    outside.mkdir()
    presets_root = tmp_path / "presets"
    try:
        presets_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _save_prompt_helper_preset_impl(
            "demo",
            "dataset",
            {0: ["prompt"]},
            presets_root=presets_root,
            path_is_within_root_fn=_path_within_root,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prompt_helper_preset_path_invalid"
    assert list(outside.iterdir()) == []


def test_save_prompt_helper_preset_rejects_symlinked_parent_without_write(tmp_path: Path) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    presets_parent = tmp_path / "linked_parent"
    try:
        presets_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _save_prompt_helper_preset_impl(
            "demo",
            "dataset",
            {0: ["prompt"]},
            presets_root=presets_parent / "presets",
            path_is_within_root_fn=_path_within_root,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prompt_helper_preset_path_invalid"
    assert list(outside.iterdir()) == []


def test_save_prompt_helper_preset_rejects_nested_symlinked_parent_without_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    presets_parent = tmp_path / "linked_parent"
    try:
        presets_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _save_prompt_helper_preset_impl(
            "demo",
            "dataset",
            {0: ["prompt"]},
            presets_root=presets_parent / "nested" / "presets",
            path_is_within_root_fn=_path_within_root,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prompt_helper_preset_path_invalid"
    assert list(outside.iterdir()) == []


def test_save_prompt_helper_preset_failure_does_not_leave_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    def partial_dump(_payload, handle, **_kwargs) -> None:
        handle.write("{partial")
        raise RuntimeError("serializer failed")

    presets_root = tmp_path / "presets"
    monkeypatch.setattr(prompt_helper_presets_service.uuid, "uuid4", lambda: FixedUUID())
    monkeypatch.setattr(prompt_helper_presets_service.json, "dump", partial_dump)

    with pytest.raises(HTTPException) as exc_info:
        _save_prompt_helper_preset_impl(
            "demo",
            "dataset",
            {0: ["prompt"]},
            presets_root=presets_root,
            path_is_within_root_fn=_path_within_root,
        )

    assert exc_info.value.status_code == 500
    assert str(exc_info.value.detail).startswith("prompt_helper_preset_save_failed:")
    assert not (presets_root / "phset_deadbeef.json").exists()
    assert list(presets_root.iterdir()) == []


def test_save_prompt_helper_preset_replaces_symlink_leaves_without_target_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PresetUUID:
        hex = "deadbeef000000000000000000000000"

    class TempUUID:
        hex = "feedface000000000000000000000000"

    uuids = iter([PresetUUID(), TempUUID()])
    presets_root = tmp_path / "presets"
    presets_root.mkdir()
    outside_final = tmp_path / "outside_preset.json"
    outside_tmp = tmp_path / "outside_tmp.json"
    outside_final.write_text('{"external":"final"}', encoding="utf-8")
    outside_tmp.write_text('{"external":"tmp"}', encoding="utf-8")
    preset_path = presets_root / "phset_deadbeef.json"
    tmp_path_link = presets_root / (
        f".phset_deadbeef.{os.getpid()}.feedface000000000000000000000000.tmp"
    )
    try:
        preset_path.symlink_to(outside_final)
        tmp_path_link.symlink_to(outside_tmp)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(prompt_helper_presets_service.uuid, "uuid4", lambda: next(uuids))

    out = _save_prompt_helper_preset_impl(
        "demo",
        "dataset",
        {0: ["prompt"]},
        presets_root=presets_root,
        path_is_within_root_fn=_path_within_root,
    )

    assert out["id"] == "phset_deadbeef"
    assert not preset_path.is_symlink()
    assert not tmp_path_link.exists()
    assert '"external"' not in preset_path.read_text(encoding="utf-8")
    assert outside_final.read_text(encoding="utf-8") == '{"external":"final"}'
    assert outside_tmp.read_text(encoding="utf-8") == '{"external":"tmp"}'


def test_yolo_run_lookup_rejects_symlinked_run_id_without_target_delete(tmp_path: Path) -> None:
    job_root = tmp_path / "yolo_runs"
    job_root.mkdir()
    target = job_root / "target_run"
    target.mkdir()
    (target / "best.pt").write_text("weights", encoding="utf-8")
    try:
        (job_root / "linked_run").symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _yolo_run_dir_impl(
            "linked_run",
            create=False,
            job_root=job_root,
            sanitize_fn=_sanitize_yolo_run_id,
            http_exception_cls=HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"
    assert (target / "best.pt").read_text(encoding="utf-8") == "weights"


def test_rfdetr_run_lookup_rejects_normalized_alias(tmp_path: Path) -> None:
    job_root = tmp_path / "rfdetr_runs"
    (job_root / "run1").mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        _rfdetr_run_dir_impl(
            "../run1",
            create=False,
            job_root=job_root,
            sanitize_fn=_sanitize_yolo_run_id,
            http_exception_cls=HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"


def test_rfdetr_run_lookup_rejects_symlinked_run_id_without_target_delete(tmp_path: Path) -> None:
    job_root = tmp_path / "rfdetr_runs"
    job_root.mkdir()
    target = job_root / "target_run"
    target.mkdir()
    (target / "checkpoint_best_total.pth").write_text("weights", encoding="utf-8")
    try:
        (job_root / "linked_run").symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _rfdetr_run_dir_impl(
            "linked_run",
            create=False,
            job_root=job_root,
            sanitize_fn=_sanitize_yolo_run_id,
            http_exception_cls=HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"
    assert (target / "checkpoint_best_total.pth").read_text(encoding="utf-8") == "weights"


def test_compute_dir_signature_skips_symlink_escape(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "local.txt").write_text("local", encoding="utf-8")
    before = _compute_dir_signature(dataset_root)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (dataset_root / "escape.txt").symlink_to(outside)

    assert _compute_dir_signature(dataset_root) == before


def test_dir_size_bytes_skips_file_symlinks(tmp_path: Path) -> None:
    quota_root = tmp_path / "quota"
    quota_root.mkdir()
    (quota_root / "local.bin").write_bytes(b"1234")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x" * 100)
    (quota_root / "escape.bin").symlink_to(outside)

    assert _dir_size_bytes(quota_root) == 4


def test_iter_yolo_images_skips_symlink_escape(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    images_root.mkdir()
    local = images_root / "local.jpg"
    local.write_bytes(b"local")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"secret")
    (images_root / "escape.jpg").symlink_to(outside)

    assert _iter_yolo_images(images_root) == [local]


def test_clip_dataset_validation_skips_symlinked_images(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    labels_root = tmp_path / "labels"
    images_root.mkdir()
    labels_root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"not an image")
    (images_root / "escape.jpg").symlink_to(outside)
    (labels_root / "local.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        _validate_clip_dataset_impl(
            {
                "images_dir": str(images_root),
                "labels_dir": str(labels_root),
            },
            http_exception_cls=HTTPException,
            load_labelmap_simple_fn=lambda _path: ["car"],
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "clip_images_missing"


def test_clip_dataset_validation_skips_symlinked_labels(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    labels_root = tmp_path / "labels"
    images_root.mkdir()
    labels_root.mkdir()
    (images_root / "local.jpg").write_bytes(b"not an image")
    outside = tmp_path / "outside.txt"
    outside.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels_root / "escape.txt").symlink_to(outside)

    with pytest.raises(HTTPException) as exc_info:
        _validate_clip_dataset_impl(
            {
                "images_dir": str(images_root),
                "labels_dir": str(labels_root),
            },
            http_exception_cls=HTTPException,
            load_labelmap_simple_fn=lambda _path: ["car"],
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "clip_labels_missing"
