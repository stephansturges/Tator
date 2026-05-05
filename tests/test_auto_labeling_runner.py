from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

import localinferenceapi as api
from models.schemas import QwenPrepassResponse


pytestmark = [pytest.mark.auto_label_full]


def _build_auto_label_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, rows, labelmap, glossary=""):
    dataset_root = tmp_path / "dataset"
    images_root = dataset_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": "ds_auto",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root / "ds_auto"),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "classes": list(labelmap),
    }
    manifest = {"images": list(rows), "labelmap": list(labelmap)}
    overlay = {}
    saved_payloads = []
    session_events = {"start": 0, "stop": 0, "heartbeat": 0}

    for row in rows:
        split = str(row.get("split") or "train").strip().lower()
        relpath = Path(str(row.get("image_relpath") or row.get("image_name") or "").strip())
        image_path = images_root / relpath
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color="white").save(image_path)
        overlay[(split, relpath.as_posix())] = list(row.get("label_lines") or [])

    monkeypatch.setattr(api, "_falcon_runtime_error_detail", lambda _torch: None)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: manifest)
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: dataset_root)
    monkeypatch.setattr(
        api,
        "_resolve_annotation_image_path",
        lambda _root, _layout, _split, relpath: images_root / relpath,
    )
    monkeypatch.setattr(
        api,
        "_annotation_effective_label_lines",
        lambda _entry, split, relpath: list(overlay.get((str(split).lower(), relpath.as_posix()), [])),
    )
    monkeypatch.setattr(api, "get_dataset_glossary", lambda _dataset_id: {"glossary": glossary})
    monkeypatch.setattr(
        api,
        "start_dataset_annotation_session",
        lambda _dataset_id, _payload: session_events.__setitem__("start", session_events["start"] + 1) or {"status": "ok"},
    )
    monkeypatch.setattr(
        api,
        "heartbeat_dataset_annotation_session",
        lambda _dataset_id, _payload: session_events.__setitem__("heartbeat", session_events["heartbeat"] + 1) or {"status": "ok"},
    )
    monkeypatch.setattr(
        api,
        "stop_dataset_annotation_session",
        lambda _dataset_id, _payload: session_events.__setitem__("stop", session_events["stop"] + 1) or {"status": "ok"},
    )

    def _save_snapshot(_dataset_id, payload):
        saved_payloads.append(payload)
        for record in payload.get("records") or []:
            split = str(record.get("split") or "train").strip().lower()
            relpath = str(record.get("image_relpath") or "").strip()
            overlay[(split, relpath)] = list(record.get("label_lines") or [])
        return {"status": "ok"}

    monkeypatch.setattr(api, "save_dataset_annotation_snapshot", _save_snapshot)
    return {
        "entry": entry,
        "manifest": manifest,
        "overlay": overlay,
        "saved_payloads": saved_payloads,
        "session_events": session_events,
    }


def _prepass_response(detections):
    return QwenPrepassResponse(detections=list(detections), trace=[])


def _encode_mask(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded.get("counts")
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": list(encoded["size"]), "counts": counts}


@pytest.mark.auto_label_smoke
def test_auto_label_runner_segmentation_keeps_baseline_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _build_auto_label_fixture(
        tmp_path,
        monkeypatch,
        rows=[{"split": "train", "image_relpath": "img1.jpg", "label_lines": []}],
        labelmap=["car"],
    )

    monkeypatch.setattr(
        api,
        "_run_prepass_annotation_qwen",
        lambda *_args, **_kwargs: _prepass_response(
            [{"label": "car", "bbox_yolo": [0.5, 0.5, 0.25, 0.25], "score": 0.9}]
        ),
    )
    monkeypatch.setattr(
        api,
        "_auto_label_falcon_candidates_for_window",
        lambda **_kwargs: [],
    )

    payload = api.AutoLabelRequest(
        dataset_id="ds_auto",
        target_mode="segmentation",
        image_relpaths=["img1.jpg"],
        class_names=["car"],
        edr_package_id="canonical_edr_pkg",
        enable_yolo=False,
        enable_rfdetr=False,
    )
    job = api.AutoLabelJob(job_id="al_seg_baseline")
    api._run_auto_label_job(job, payload)

    assert job.status == "completed"
    assert job.result["baseline_candidate_count"] == 1
    assert job.result["falcon_query_count"] == 1
    assert job.result["labels_added"] == 1
    assert state["saved_payloads"], "expected snapshot write"
    saved_line = state["saved_payloads"][0]["records"][0]["label_lines"][0]
    parts = saved_line.split()
    assert len(parts) == 9, "segmentation mode should serialize baseline bbox as a polygon"


@pytest.mark.auto_label_smoke
def test_auto_label_runner_uses_raw_package_baseline_without_prepass_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_auto_label_fixture(
        tmp_path,
        monkeypatch,
        rows=[{"split": "train", "image_relpath": "img1.jpg", "label_lines": []}],
        labelmap=["car"],
    )

    seen_requests = []

    def _fake_prepass(request, **_kwargs):
        seen_requests.append(request)
        return _prepass_response([])

    monkeypatch.setattr(api, "_run_prepass_annotation_qwen", _fake_prepass)
    monkeypatch.setattr(api, "_auto_label_falcon_candidates_for_window", lambda **_kwargs: [])

    payload = api.AutoLabelRequest(
        dataset_id="ds_auto",
        target_mode="detection",
        image_relpaths=["img1.jpg"],
        class_names=["car"],
        edr_package_id="canonical_edr_pkg",
        enable_yolo=False,
        enable_rfdetr=False,
    )
    job = api.AutoLabelJob(job_id="al_raw_package_baseline")
    api._run_auto_label_job(job, payload)

    assert job.status == "completed"
    assert seen_requests, "expected baseline prepass request"
    request = seen_requests[0]
    assert request.edr_package_apply_ensemble is False
    assert request.prepass_keep_all is True
    assert request.prepass_finalize is False


@pytest.mark.auto_label_smoke
def test_auto_label_runner_manual_quadrants_do_not_suppress_falcon_for_explicit_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_auto_label_fixture(
        tmp_path,
        monkeypatch,
        rows=[{"split": "train", "image_relpath": "img1.jpg", "label_lines": []}],
        labelmap=["car"],
    )

    monkeypatch.setattr(
        api,
        "_run_prepass_annotation_qwen",
        lambda *_args, **_kwargs: _prepass_response(
            [{"label": "car", "bbox_yolo": [0.2, 0.2, 0.1, 0.1], "score": 0.9}]
        ),
    )

    falcon_calls = []

    def _fake_falcon(**kwargs):
        falcon_calls.append(kwargs)
        return [
            {
                "class_id": 0,
                "class_name": "car",
                "bbox_xyxy": (40.0, 40.0, 56.0, 56.0),
                "mask": None,
                "score": 0.7,
                "source": "falcon_fill_in",
            }
        ]

    monkeypatch.setattr(api, "_auto_label_falcon_candidates_for_window", _fake_falcon)

    payload = api.AutoLabelRequest(
        dataset_id="ds_auto",
        target_mode="detection",
        image_relpaths=["img1.jpg"],
        class_names=["car"],
        falcon_window_mode="quadrants",
        edr_package_id="canonical_edr_pkg",
        enable_yolo=False,
        enable_rfdetr=False,
    )
    job = api.AutoLabelJob(job_id="al_quadrants_explicit_subset")
    api._run_auto_label_job(job, payload)

    assert job.status == "completed"
    assert job.result["baseline_candidate_count"] == 1
    assert job.result["falcon_window_count"] == 4
    assert job.result["falcon_query_count"] == 4
    assert len(falcon_calls) == 4
    assert job.result["falcon_candidate_count"] == 4


@pytest.mark.auto_label_smoke
def test_auto_label_parse_planner_decision_filters_requested_classes() -> None:
    raw = json_string = (
        '{"decision":"full_image","global_classes":["person","car"],'
        '"cell_classes":[{"cell":"A1","classes":["person","car"]}],"cells":["A1"]}'
    )
    parsed = api._auto_label_parse_planner_decision(
        json_string,
        labelmap=["car", "person"],
        valid_cells=["A1", "A2"],
        allowed_classes=["car"],
    )
    assert parsed is not None
    assert parsed.global_classes == ["car"]
    assert parsed.cell_classes[0].classes == ["car"]


@pytest.mark.auto_label_smoke
def test_auto_label_run_planner_retries_once_before_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            "not-json",
            '{"decision":"full_image","global_classes":["car"],"cells":[],"cell_classes":[],"confidence":"high"}',
        ]
    )
    monkeypatch.setattr(api, "_run_qwen_chat", lambda *_args, **_kwargs: next(responses))
    image = Image.new("RGB", (64, 64), color="white")

    summary = api._auto_label_run_planner(
        image,
        model_id_override=None,
        labelmap=["car", "person"],
        target_classes=["car"],
        baseline_counts={"car": 0},
        window_overlap_ratio=0.1,
        use_caption=False,
        grid_cols=2,
        grid_rows=2,
    )

    assert summary["retry_count"] == 1
    assert summary["attempt_count"] == 2
    assert len(summary["raw_attempts"]) == 2
    assert summary["parsed"]["global_classes"] == ["car"]


@pytest.mark.auto_label_smoke
def test_auto_label_runner_respects_image_relpath_subset_and_records_zero_write_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _build_auto_label_fixture(
        tmp_path,
        monkeypatch,
        rows=[
            {"split": "train", "image_relpath": "img1.jpg", "label_lines": []},
            {"split": "train", "image_relpath": "img2.jpg", "label_lines": []},
        ],
        labelmap=["car"],
    )
    monkeypatch.setattr(api, "_run_prepass_annotation_qwen", lambda *_args, **_kwargs: _prepass_response([]))
    monkeypatch.setattr(api, "_auto_label_falcon_candidates_for_window", lambda **_kwargs: [])

    payload = api.AutoLabelRequest(
        dataset_id="ds_auto",
        target_mode="detection",
        image_relpaths=["img2.jpg"],
        class_names=["car"],
        enable_yolo=False,
        enable_rfdetr=False,
    )
    job = api.AutoLabelJob(job_id="al_subset")
    api._run_auto_label_job(job, payload)

    assert job.status == "completed"
    assert job.result["images_total"] == 1
    assert job.result["images_processed"] == 1
    assert job.result["zero_write_images"] == 1
    assert job.result["writes_applied"] == 0
    assert not state["saved_payloads"]


@pytest.mark.auto_label_full
def test_auto_label_runner_tracks_candidate_and_write_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _build_auto_label_fixture(
        tmp_path,
        monkeypatch,
        rows=[{"split": "train", "image_relpath": "img1.jpg", "label_lines": []}],
        labelmap=["car"],
    )
    monkeypatch.setattr(api, "_run_prepass_annotation_qwen", lambda *_args, **_kwargs: _prepass_response([]))
    monkeypatch.setattr(
        api,
        "_auto_label_falcon_candidates_for_window",
        lambda **_kwargs: [
            {
                "class_id": 0,
                "class_name": "car",
                "bbox_xyxy": (10.0, 10.0, 30.0, 30.0),
                "mask": None,
                "score": 0.8,
                "source": "falcon_fill_in",
            }
        ],
    )

    payload = api.AutoLabelRequest(
        dataset_id="ds_auto",
        target_mode="detection",
        image_relpaths=["img1.jpg"],
        class_names=["car"],
        falcon_window_mode="full_image",
        enable_yolo=False,
        enable_rfdetr=False,
    )
    job = api.AutoLabelJob(job_id="al_metrics")
    api._run_auto_label_job(job, payload)

    assert job.status == "completed"
    assert job.result["falcon_candidate_count"] == 1
    assert job.result["kept_candidate_count"] == 1
    assert job.result["writes_attempted"] == 1
    assert job.result["writes_applied"] == 1
    assert job.result["image_times_sec"]
    assert job.result["timings_sec"]["total"] >= 0.0


@pytest.mark.auto_label_smoke
def test_auto_label_runner_reuses_existing_annotation_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _build_auto_label_fixture(
        tmp_path,
        monkeypatch,
        rows=[{"split": "train", "image_relpath": "img1.jpg", "label_lines": []}],
        labelmap=["car"],
    )
    meta = {
        "annotation_lock": {
            "holder": "webui:tab-a",
            "session_id": "sess-lock",
            "expires_at": 10**12,
        }
    }
    monkeypatch.setattr(
        api,
        "_annotation_load_or_create_meta",
        lambda _entry: (tmp_path / "meta.json", meta),
    )
    monkeypatch.setattr(api, "_run_prepass_annotation_qwen", lambda *_args, **_kwargs: _prepass_response([]))
    monkeypatch.setattr(api, "_auto_label_falcon_candidates_for_window", lambda **_kwargs: [])

    payload = api.AutoLabelRequest(
        dataset_id="ds_auto",
        annotation_session_id="sess-lock",
        target_mode="detection",
        image_relpaths=["img1.jpg"],
        class_names=["car"],
        enable_yolo=False,
        enable_rfdetr=False,
    )
    job = api.AutoLabelJob(job_id="al_reuse_session")
    api._run_auto_label_job(job, payload)

    assert job.status == "completed"
    assert job.result["annotation_session_id"] == "sess-lock"
    assert state["session_events"]["start"] == 0
    assert state["session_events"]["stop"] == 0


@pytest.mark.auto_label_smoke
def test_auto_label_runner_fails_for_non_owner_annotation_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _build_auto_label_fixture(
        tmp_path,
        monkeypatch,
        rows=[{"split": "train", "image_relpath": "img1.jpg", "label_lines": []}],
        labelmap=["car"],
    )
    meta = {
        "annotation_lock": {
            "holder": "webui:tab-a",
            "session_id": "sess-lock",
            "expires_at": 10**12,
        }
    }
    monkeypatch.setattr(
        api,
        "_annotation_load_or_create_meta",
        lambda _entry: (tmp_path / "meta.json", meta),
    )

    payload = api.AutoLabelRequest(
        dataset_id="ds_auto",
        annotation_session_id="wrong-session",
        target_mode="detection",
        image_relpaths=["img1.jpg"],
        class_names=["car"],
        enable_yolo=False,
        enable_rfdetr=False,
    )
    job = api.AutoLabelJob(job_id="al_wrong_session")
    api._run_auto_label_job(job, payload)

    assert job.status == "failed"
    assert job.error == "annotation_lock_active"
    assert state["session_events"]["start"] == 0
    assert state["session_events"]["stop"] == 0


@pytest.mark.auto_label_smoke
def test_auto_label_falcon_window_runs_fallback_tiers_when_tier_a_is_weak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_auto_label_fixture(
        tmp_path,
        monkeypatch,
        rows=[{"split": "train", "image_relpath": "img1.jpg", "label_lines": []}],
        labelmap=["light_vehicle"],
    )

    queries_seen = []

    def _fake_run(**kwargs):
        queries_seen.append(
            {
                "task": kwargs.get("task"),
                "backend": kwargs.get("backend"),
                "queries": list(kwargs.get("queries") or []),
            }
        )
        out = []
        for query in kwargs.get("queries") or []:
            query = str(query or "")
            if query == "car":
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[20:24, 18:22] = 1
                out.append([{"mask_rle": _encode_mask(mask)}])
            elif query == "SUV":
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[20:34, 18:30] = 1
                out.append([{"mask_rle": _encode_mask(mask)}])
            else:
                out.append([])
        return out

    monkeypatch.setattr(api, "_run_falcon_queries_impl", _fake_run)
    image = Image.new("RGB", (64, 64), color="white")
    payload = api.AutoLabelRequest(
        dataset_id="ds_auto",
        target_mode="detection",
        class_names=["light_vehicle"],
        falcon_detection_strategy="segmentation_boxes",
        falcon_backend="embedded",
        falcon_component_mode="component_split",
        enable_yolo=False,
        enable_rfdetr=False,
    )

    result = api._auto_label_falcon_candidates_for_window(
        pil_img=image,
        crop_window={"id": "FULL", "xyxy": (0.0, 0.0, 64.0, 64.0)},
        class_names=["light_vehicle"],
        class_id_map={"light_vehicle": 0},
        labelmap=["light_vehicle"],
        glossary="",
        payload=payload,
        target_mode="detection",
    )

    assert result["query_count"] >= 2
    assert len(queries_seen) >= 2
    assert queries_seen[0]["task"] == "segmentation"
    assert queries_seen[0]["queries"] == ["car"]
    assert any("SUV" in call["queries"] for call in queries_seen[1:])
    assert result["candidates"]
    assert result["candidates"][0]["class_name"] == "light_vehicle"
    assert result["candidates"][0]["derivation_mode"] == "component_split"
    assert result["candidates"][0]["score"] > 0.5


@pytest.mark.auto_label_smoke
def test_auto_label_dedupe_prefers_higher_scored_falcon_candidate() -> None:
    low_score = {
        "class_name": "truck",
        "bbox_xyxy": (100.0, 100.0, 300.0, 220.0),
        "score": 0.35,
        "source": "falcon_fill_in",
        "component_area_px": 24000,
        "bbox_area_fraction_crop": 0.28,
        "border_touch_count": 1,
        "component_count": 1,
    }
    better = {
        "class_name": "truck",
        "bbox_xyxy": (128.0, 116.0, 252.0, 206.0),
        "score": 0.88,
        "source": "falcon_fill_in",
        "component_area_px": 11160,
        "bbox_area_fraction_crop": 0.08,
        "border_touch_count": 0,
        "component_count": 1,
    }

    kept, dropped = api._auto_label_dedupe_candidates(
        [low_score, better],
        existing=[],
        target_mode="detection",
        iou_threshold=0.1,
    )

    assert dropped == 1
    assert kept == [better]


@pytest.mark.auto_label_smoke
def test_auto_label_falcon_window_collapses_duplicate_query_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_auto_label_fixture(
        tmp_path,
        monkeypatch,
        rows=[{"split": "train", "image_relpath": "img1.jpg", "label_lines": []}],
        labelmap=["utility_pole"],
    )

    def _fake_run(**kwargs):
        out = []
        for query in kwargs.get("queries") or []:
            if query == "power pylon":
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[30:34, 18:40] = 1
                out.append([{"mask_rle": _encode_mask(mask)} for _ in range(5)])
            else:
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[10:28, 8:28] = 1
                out.append([{"mask_rle": _encode_mask(mask)}])
        return out

    monkeypatch.setattr(api, "_run_falcon_queries_impl", _fake_run)
    image = Image.new("RGB", (64, 64), color="white")
    payload = api.AutoLabelRequest(
        dataset_id="ds_auto",
        target_mode="detection",
        class_names=["utility_pole"],
        falcon_detection_strategy="segmentation_boxes",
        falcon_backend="embedded",
        falcon_component_mode="component_split",
        enable_yolo=False,
        enable_rfdetr=False,
    )

    result = api._auto_label_falcon_candidates_for_window(
        pil_img=image,
        crop_window={"id": "FULL", "xyxy": (0.0, 0.0, 64.0, 64.0)},
        class_names=["utility_pole"],
        class_id_map={"utility_pole": 0},
        labelmap=["utility_pole"],
        glossary='{"utility_pole":["streetlight","power pylon"]}',
        payload=payload,
        target_mode="detection",
    )

    power_pylon = [row for row in result["candidates"] if row.get("query") == "power pylon"]
    assert power_pylon
    assert len(power_pylon) <= 6


@pytest.mark.auto_label_smoke
def test_auto_label_falcon_window_collapses_cross_query_duplicate_boxes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_auto_label_fixture(
        tmp_path,
        monkeypatch,
        rows=[{"split": "train", "image_relpath": "img1.jpg", "label_lines": []}],
        labelmap=["utility_pole"],
    )

    def _fake_run(**kwargs):
        out = []
        for _query in kwargs.get("queries") or []:
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[30:34, 18:40] = 1
            out.append([{"mask_rle": _encode_mask(mask)} for _ in range(3)])
        return out

    monkeypatch.setattr(api, "_run_falcon_queries_impl", _fake_run)
    image = Image.new("RGB", (64, 64), color="white")
    payload = api.AutoLabelRequest(
        dataset_id="ds_auto",
        target_mode="detection",
        class_names=["utility_pole"],
        falcon_detection_strategy="segmentation_boxes",
        falcon_backend="embedded",
        falcon_component_mode="component_split",
        enable_yolo=False,
        enable_rfdetr=False,
    )

    result = api._auto_label_falcon_candidates_for_window(
        pil_img=image,
        crop_window={"id": "FULL", "xyxy": (0.0, 0.0, 64.0, 64.0)},
        class_names=["utility_pole"],
        class_id_map={"utility_pole": 0},
        labelmap=["utility_pole"],
        glossary='{"utility_pole":["streetlight","power pylon","comms mast"]}',
        payload=payload,
        target_mode="detection",
    )

    unique_boxes = {
        tuple(round(float(v), 3) for v in row.get("bbox_xyxy") or ())
        for row in result["candidates"]
    }
    assert len(unique_boxes) == 1
    assert len(result["candidates"]) == 1


@pytest.mark.auto_label_smoke
def test_auto_label_falcon_window_stops_after_strong_tier_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_auto_label_fixture(
        tmp_path,
        monkeypatch,
        rows=[{"split": "train", "image_relpath": "img1.jpg", "label_lines": []}],
        labelmap=["boat"],
    )

    queries_seen = []

    def _fake_run(**kwargs):
        queries_seen.append(list(kwargs.get("queries") or []))
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:36, 14:34] = 1
        return [[{"mask_rle": _encode_mask(mask)}] for _ in (kwargs.get("queries") or [])]

    monkeypatch.setattr(api, "_run_falcon_queries_impl", _fake_run)
    image = Image.new("RGB", (64, 64), color="white")
    payload = api.AutoLabelRequest(
        dataset_id="ds_auto",
        target_mode="detection",
        class_names=["boat"],
        falcon_detection_strategy="segmentation_boxes",
        falcon_backend="embedded",
        falcon_component_mode="component_split",
        enable_yolo=False,
        enable_rfdetr=False,
    )

    result = api._auto_label_falcon_candidates_for_window(
        pil_img=image,
        crop_window={"id": "FULL", "xyxy": (0.0, 0.0, 64.0, 64.0)},
        class_names=["boat"],
        class_id_map={"boat": 0},
        labelmap=["boat"],
        glossary="",
        payload=payload,
        target_mode="detection",
    )

    assert result["query_count"] == 2
    assert queries_seen == [["boat", "canoe"]]
