from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path

import pytest

import localinferenceapi as api
from services.detectors import _load_yolo_active_impl


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
