from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image

from services.calibration_helpers import _calibration_prepass_worker


class _ProgressQueue:
    def __init__(self) -> None:
        self.items: List[int] = []

    def put(self, value: int) -> None:
        self.items.append(int(value))


def test_calibration_prepass_worker_raises_on_invalid_payload(tmp_path: Path) -> None:
    class _BadPayload:
        def __init__(self, **_kwargs: Any) -> None:
            raise ValueError("bad_payload")

    with pytest.raises(RuntimeError, match="deep_prepass_payload_invalid"):
        _calibration_prepass_worker(
            0,
            [],
            "dataset",
            [],
            "",
            {},
            cancel_event=None,
            progress_queue=None,
            resolve_dataset_fn=lambda _dataset_id: tmp_path,
            prepass_request_cls=_BadPayload,
            cache_image_fn=lambda _img, _variant: "tok",
            run_prepass_fn=lambda *_args, **_kwargs: {"detections": [], "warnings": []},
            write_record_fn=lambda _path, _record: None,
            set_device_pref_fn=None,
        )


def test_calibration_prepass_worker_writes_missing_image_record(tmp_path: Path) -> None:
    class _Payload:
        def __init__(self, **_kwargs: Any) -> None:
            self.sam_variant = None

    records: Dict[Path, Dict[str, Any]] = {}
    progress = _ProgressQueue()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    cache_path = tmp_path / "cache" / "missing.json"

    _calibration_prepass_worker(
        0,
        [("missing.jpg", str(cache_path))],
        "dataset",
        [],
        "",
        {},
        cancel_event=None,
        progress_queue=progress,
        resolve_dataset_fn=lambda _dataset_id: dataset_root,
        prepass_request_cls=_Payload,
        cache_image_fn=lambda _img, _variant: "tok",
        run_prepass_fn=lambda *_args, **_kwargs: {"detections": [], "warnings": []},
        write_record_fn=lambda path, record: records.setdefault(Path(path), record),
        set_device_pref_fn=None,
    )

    assert cache_path in records
    assert records[cache_path]["warnings"] == ["deep_prepass_image_missing"]
    assert progress.items == [1]


def test_calibration_prepass_worker_writes_open_failure_record(tmp_path: Path) -> None:
    class _Payload:
        def __init__(self, **_kwargs: Any) -> None:
            self.sam_variant = None

    records: Dict[Path, Dict[str, Any]] = {}
    progress = _ProgressQueue()
    dataset_root = tmp_path / "dataset"
    val_dir = dataset_root / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    bad_image = val_dir / "bad.jpg"
    bad_image.write_bytes(b"not-an-image")
    cache_path = tmp_path / "cache" / "bad.json"

    _calibration_prepass_worker(
        0,
        [("bad.jpg", str(cache_path))],
        "dataset",
        [],
        "",
        {},
        cancel_event=None,
        progress_queue=progress,
        resolve_dataset_fn=lambda _dataset_id: dataset_root,
        prepass_request_cls=_Payload,
        cache_image_fn=lambda _img, _variant: "tok",
        run_prepass_fn=lambda *_args, **_kwargs: {"detections": [], "warnings": []},
        write_record_fn=lambda path, record: records.setdefault(Path(path), record),
        set_device_pref_fn=None,
    )

    assert cache_path in records
    warning = records[cache_path]["warnings"][0]
    assert warning.startswith("deep_prepass_image_open_failed:")
    assert progress.items == [1]


def test_calibration_prepass_worker_reads_yolo_split_image_dirs(tmp_path: Path) -> None:
    class _Payload:
        def __init__(self, **_kwargs: Any) -> None:
            self.sam_variant = None

    records: Dict[Path, Dict[str, Any]] = {}
    progress = _ProgressQueue()
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "val" / "images" / "nested" / "ok.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color="white").save(image_path)
    cache_path = tmp_path / "cache" / "ok.json"

    _calibration_prepass_worker(
        0,
        [("nested/ok.jpg", str(cache_path))],
        "dataset",
        ["object"],
        "",
        {},
        cancel_event=None,
        progress_queue=progress,
        resolve_dataset_fn=lambda _dataset_id: dataset_root,
        prepass_request_cls=_Payload,
        cache_image_fn=lambda _img, _variant: "tok",
        run_prepass_fn=lambda *_args, **_kwargs: {
            "detections": [{"label": "object", "score": 0.9}],
            "warnings": [],
        },
        write_record_fn=lambda path, record: records.setdefault(Path(path), record),
        set_device_pref_fn=None,
    )

    assert records[cache_path]["detections"] == [{"label": "object", "score": 0.9}]
    assert records[cache_path]["warnings"] == []
    assert progress.items == [1]


def test_calibration_prepass_worker_treats_image_symlink_escape_as_missing(
    tmp_path: Path,
) -> None:
    class _Payload:
        def __init__(self, **_kwargs: Any) -> None:
            self.sam_variant = None

    records: Dict[Path, Dict[str, Any]] = {}
    dataset_root = tmp_path / "dataset"
    image_root = dataset_root / "val" / "images"
    image_root.mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (8, 8), color="white").save(outside)
    try:
        (image_root / "escaped.jpg").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    cache_path = tmp_path / "cache" / "escaped.json"

    _calibration_prepass_worker(
        0,
        [("escaped.jpg", str(cache_path))],
        "dataset",
        ["object"],
        "",
        {},
        cancel_event=None,
        progress_queue=None,
        resolve_dataset_fn=lambda _dataset_id: dataset_root,
        prepass_request_cls=_Payload,
        cache_image_fn=lambda _img, _variant: "tok",
        run_prepass_fn=lambda *_args, **_kwargs: {
            "detections": [{"label": "object", "score": 0.9}],
            "warnings": [],
        },
        write_record_fn=lambda path, record: records.setdefault(Path(path), record),
        set_device_pref_fn=None,
    )

    assert records[cache_path]["warnings"] == ["deep_prepass_image_missing"]
