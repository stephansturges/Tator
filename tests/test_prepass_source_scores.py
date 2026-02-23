from types import SimpleNamespace

from PIL import Image

from services.prepass import (
    _agent_deep_prepass_cleanup_impl,
    _agent_merge_prepass_detections,
    _agent_run_deep_prepass_part_a_impl,
    _agent_select_similarity_exemplars,
)

import pytest
import localinferenceapi as api


def test_merge_preserves_per_source_scores():
    detections = [
        {
            "label": "car",
            "bbox_xyxy_px": [10.0, 10.0, 50.0, 50.0],
            "score": 0.91,
            "score_source": "rfdetr",
            "source": "rfdetr",
            "source_list": ["rfdetr"],
        },
        {
            "label": "car",
            "bbox_xyxy_px": [11.0, 11.0, 51.0, 51.0],
            "score": 0.72,
            "score_source": "yolo",
            "source": "yolo",
            "source_list": ["yolo"],
        },
    ]
    merged, removed = _agent_merge_prepass_detections(detections, iou_thr=0.5)
    assert removed == 1
    assert len(merged) == 1
    merged_det = merged[0]
    assert set(merged_det.get("source_list") or []) == {"rfdetr", "yolo"}
    score_map = merged_det.get("score_by_source") or {}
    assert score_map.get("rfdetr") == 0.91
    assert score_map.get("yolo") == 0.72


def test_sanitize_keeps_score_by_source():
    cleaned, rejected = api._agent_sanitize_detection_items(
        [
            {
                "label": "car",
                "bbox_xyxy_px": [5.0, 5.0, 30.0, 40.0],
                "score": 0.8,
                "score_source": "rfdetr",
                "source": "rfdetr",
                "source_list": ["rfdetr", "yolo"],
                "score_by_source": {
                    "rfdetr": 0.8,
                    "yolo": "0.7",
                    "invalid": "not-a-number",
                },
            }
        ],
        pil_img=None,
        classifier_head=None,
        img_w=100,
        img_h=100,
        labelmap=["car"],
        background=None,
    )
    assert rejected == 0
    assert len(cleaned) == 1
    score_map = cleaned[0].get("score_by_source") or {}
    assert score_map == {"rfdetr": 0.8, "yolo": 0.7}


def test_merge_cross_class_disabled_by_default():
    detections = [
        {
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 100.0, 100.0],
            "score": 0.9,
            "score_source": "yolo",
            "source": "yolo",
        },
        {
            "label": "truck",
            "bbox_xyxy_px": [2.0, 2.0, 98.0, 98.0],
            "score": 0.85,
            "score_source": "rfdetr",
            "source": "rfdetr",
        },
    ]
    merged, removed = _agent_merge_prepass_detections(detections, iou_thr=0.5)
    assert removed == 0
    assert len(merged) == 2


def test_merge_cross_class_enabled_uses_cross_class_iou():
    detections = [
        {
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 100.0, 100.0],
            "score": 0.9,
            "score_source": "yolo",
            "source": "yolo",
        },
        {
            "label": "truck",
            "bbox_xyxy_px": [2.0, 2.0, 98.0, 98.0],
            "score": 0.85,
            "score_source": "rfdetr",
            "source": "rfdetr",
        },
    ]
    merged, removed = _agent_merge_prepass_detections(detections, iou_thr=0.5, cross_class_iou_thr=0.8)
    assert removed == 1
    assert len(merged) == 1
    merged_det = merged[0]
    assert set(merged_det.get("source_list") or []) == {"yolo", "rfdetr"}
    score_map = merged_det.get("score_by_source") or {}
    assert score_map.get("yolo") == 0.9
    assert score_map.get("rfdetr") == 0.85


def test_merge_preserves_atom_membership_union():
    detections = [
        {
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 100.0, 100.0],
            "score": 0.9,
            "source": "yolo",
            "prepass_atom_ids": ["a0000001"],
        },
        {
            "label": "car",
            "bbox_xyxy_px": [1.0, 1.0, 99.0, 99.0],
            "score": 0.8,
            "source": "rfdetr",
            "prepass_atom_ids": ["a0000002"],
        },
    ]
    merged, removed = _agent_merge_prepass_detections(detections, iou_thr=0.5)
    assert removed == 1
    assert len(merged) == 1
    assert set(merged[0].get("prepass_atom_ids") or []) == {"a0000001", "a0000002"}


def test_merge_wbf_fuses_bbox_geometry():
    detections = [
        {
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 10.0, 10.0],
            "score": 0.9,
            "source": "yolo",
        },
        {
            "label": "car",
            "bbox_xyxy_px": [1.0, 1.0, 11.0, 11.0],
            "score": 0.3,
            "source": "rfdetr",
        },
    ]
    merged, removed = _agent_merge_prepass_detections(detections, iou_thr=0.5, fusion_mode="wbf")
    assert removed == 1
    assert len(merged) == 1
    # Weighted by scores: (0*0.9 + 1*0.3)/1.2 = 0.25 and (10*0.9 + 11*0.3)/1.2 = 10.25
    bbox = merged[0].get("bbox_xyxy_px") or []
    assert bbox and len(bbox) >= 4
    assert bbox[0] == pytest.approx(0.25)
    assert bbox[1] == pytest.approx(0.25)
    assert bbox[2] == pytest.approx(10.25)
    assert bbox[3] == pytest.approx(10.25)


def test_similarity_exemplar_diverse_is_deterministic_and_source_aware():
    detections = [
        {
            "handle": "yolo_a",
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 20.0, 20.0],
            "score": 0.96,
            "source": "yolo",
        },
        {
            "handle": "yolo_b",
            "label": "car",
            "bbox_xyxy_px": [2.0, 2.0, 22.0, 22.0],
            "score": 0.93,
            "source": "yolo",
        },
        {
            "handle": "rfdetr_a",
            "label": "car",
            "bbox_xyxy_px": [60.0, 60.0, 84.0, 84.0],
            "score": 0.91,
            "source": "rfdetr",
        },
        {
            "handle": "rfdetr_b",
            "label": "car",
            "bbox_xyxy_px": [62.0, 62.0, 86.0, 86.0],
            "score": 0.89,
            "source": "rfdetr",
        },
        {
            "handle": "sam3_text_a",
            "label": "car",
            "bbox_xyxy_px": [110.0, 24.0, 130.0, 44.0],
            "score": 0.86,
            "source": "sam3_text",
        },
        {
            "handle": "sam3_sim_a",
            "label": "car",
            "bbox_xyxy_px": [26.0, 110.0, 46.0, 130.0],
            "score": 0.84,
            "source": "sam3_similarity",
        },
    ]
    first = _agent_select_similarity_exemplars(
        0.8,
        detections=detections,
        strategy="diverse",
        seed=17,
        exemplar_fraction=0.5,
        exemplar_min=2,
        exemplar_max=4,
        source_quota=1,
    )
    second = _agent_select_similarity_exemplars(
        0.8,
        detections=detections,
        strategy="diverse",
        seed=17,
        exemplar_fraction=0.5,
        exemplar_min=2,
        exemplar_max=4,
        source_quota=1,
    )
    first_handles = [det.get("handle") for det in first.get("car") or []]
    second_handles = [det.get("handle") for det in second.get("car") or []]
    assert first_handles == second_handles
    assert len(first_handles) == 3
    assert len(set(det.get("source") for det in first.get("car") or [])) >= 2


def test_similarity_exemplar_diverse_is_order_invariant_for_equal_scores():
    detections = [
        {
            "handle": "yolo_a",
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 20.0, 20.0],
            "score": 0.95,
            "source": "yolo",
        },
        {
            "handle": "rfdetr_a",
            "label": "car",
            "bbox_xyxy_px": [80.0, 0.0, 100.0, 20.0],
            "score": 0.95,
            "source": "rfdetr",
        },
        {
            "handle": "sam3_text_a",
            "label": "car",
            "bbox_xyxy_px": [0.0, 80.0, 20.0, 100.0],
            "score": 0.95,
            "source": "sam3_text",
        },
        {
            "handle": "yolo_b",
            "label": "car",
            "bbox_xyxy_px": [22.0, 2.0, 42.0, 22.0],
            "score": 0.92,
            "source": "yolo",
        },
        {
            "handle": "rfdetr_b",
            "label": "car",
            "bbox_xyxy_px": [82.0, 2.0, 102.0, 22.0],
            "score": 0.92,
            "source": "rfdetr",
        },
        {
            "handle": "sam3_sim_a",
            "label": "car",
            "bbox_xyxy_px": [2.0, 82.0, 22.0, 102.0],
            "score": 0.92,
            "source": "sam3_similarity",
        },
    ]
    params = dict(
        min_score=0.9,
        strategy="diverse",
        seed=123,
        exemplar_fraction=0.5,
        exemplar_min=3,
        exemplar_max=4,
        source_quota=1,
    )
    first = _agent_select_similarity_exemplars(
        params["min_score"],
        detections=detections,
        strategy=params["strategy"],
        seed=params["seed"],
        exemplar_fraction=params["exemplar_fraction"],
        exemplar_min=params["exemplar_min"],
        exemplar_max=params["exemplar_max"],
        source_quota=params["source_quota"],
    )
    second = _agent_select_similarity_exemplars(
        params["min_score"],
        detections=list(reversed(detections)),
        strategy=params["strategy"],
        seed=params["seed"],
        exemplar_fraction=params["exemplar_fraction"],
        exemplar_min=params["exemplar_min"],
        exemplar_max=params["exemplar_max"],
        source_quota=params["source_quota"],
    )
    first_handles = [det.get("handle") for det in first.get("car") or []]
    second_handles = [det.get("handle") for det in second.get("car") or []]
    assert first_handles == second_handles


def test_similarity_exemplar_diverse_handles_signature_tie_order_invariant():
    detections = [
        {
            "id": "a",
            "handle": "shared",
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 20.0, 20.0],
            "score": 0.95,
            "source": "yolo",
            "debug": "first",
        },
        {
            "id": "b",
            "handle": "shared",
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 20.0, 20.0],
            "score": 0.95,
            "source": "yolo",
            "debug": "second",
        },
    ]
    first = _agent_select_similarity_exemplars(
        0.9,
        detections=detections,
        strategy="diverse",
        seed=99,
        exemplar_fraction=0.5,
        exemplar_min=1,
        exemplar_max=1,
        source_quota=0,
    )
    second = _agent_select_similarity_exemplars(
        0.9,
        detections=list(reversed(detections)),
        strategy="diverse",
        seed=99,
        exemplar_fraction=0.5,
        exemplar_min=1,
        exemplar_max=1,
        source_quota=0,
    )
    first_ids = [det.get("id") for det in first.get("car") or []]
    second_ids = [det.get("id") for det in second.get("car") or []]
    assert first_ids == second_ids


def test_similarity_exemplar_diverse_respects_zero_source_quota():
    detections = [
        {
            "handle": "yolo_top",
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 20.0, 20.0],
            "score": 0.99,
            "source": "yolo",
        },
        {
            "handle": "yolo_second",
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 20.0, 20.0],
            "score": 0.98,
            "source": "yolo",
        },
        {
            "handle": "rfdetr_low",
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 20.0, 20.0],
            "score": 0.50,
            "source": "rfdetr",
        },
    ]
    with_quota = _agent_select_similarity_exemplars(
        0.0,
        detections=detections,
        strategy="diverse",
        seed=7,
        exemplar_fraction=0.8,
        exemplar_min=2,
        exemplar_max=2,
        source_quota=1,
    )
    without_quota = _agent_select_similarity_exemplars(
        0.0,
        detections=detections,
        strategy="diverse",
        seed=7,
        exemplar_fraction=0.8,
        exemplar_min=2,
        exemplar_max=2,
        source_quota=0,
    )
    with_quota_handles = [det.get("handle") for det in with_quota.get("car") or []]
    without_quota_handles = [det.get("handle") for det in without_quota.get("car") or []]
    assert "rfdetr_low" in with_quota_handles
    assert without_quota_handles == ["yolo_top", "yolo_second"]


def test_deep_prepass_cleanup_respects_explicit_zero_iou():
    payload = SimpleNamespace(
        iou=0.0,
        cross_class_dedupe_enabled=False,
        scoreless_iou=0.0,
        prepass_keep_all=True,
        classifier_id=None,
        fusion_mode="primary",
    )
    detections = [
        {
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 50.0, 50.0],
            "bbox_2d": [0.0, 0.0, 50.0, 50.0],
            "score": 0.95,
            "source": "yolo",
        },
        {
            "label": "car",
            "bbox_xyxy_px": [1.0, 1.0, 51.0, 51.0],
            "bbox_2d": [1.0, 1.0, 50.0, 50.0],
            "score": 0.90,
            "source": "rfdetr",
        },
    ]
    result = _agent_deep_prepass_cleanup_impl(
        payload,
        detections=detections,
        pil_img=Image.new("RGB", (64, 64)),
        labelmap=["car"],
        resolve_classifier_path_fn=lambda _classifier_id: None,
        load_classifier_head_fn=lambda _path: None,
        active_classifier_head=None,
        background_from_head_fn=lambda _head: set(),
        sanitize_fn=lambda dets, **_kwargs: (dets, 0),
        default_iou=0.75,
    )
    assert len(result["detections"]) == 2
    assert result["removed"] == 0


def test_deep_prepass_cleanup_rejects_non_finite_scoreless_iou():
    payload = SimpleNamespace(
        iou=0.75,
        cross_class_dedupe_enabled=False,
        scoreless_iou=float("nan"),
        prepass_keep_all=True,
        classifier_id=None,
        fusion_mode="primary",
    )
    detections = [
        {
            "label": "car",
            "bbox_xyxy_px": [0.0, 0.0, 50.0, 50.0],
            "bbox_2d": [0.0, 0.0, 50.0, 50.0],
            "score": None,
            "source": "unknown",
        },
        {
            "label": "car",
            "bbox_xyxy_px": [1.0, 1.0, 51.0, 51.0],
            "bbox_2d": [1.0, 1.0, 50.0, 50.0],
            "score": 0.90,
            "source": "rfdetr",
        },
    ]
    result = _agent_deep_prepass_cleanup_impl(
        payload,
        detections=detections,
        pil_img=Image.new("RGB", (64, 64)),
        labelmap=["car"],
        resolve_classifier_path_fn=lambda _classifier_id: None,
        load_classifier_head_fn=lambda _path: None,
        active_classifier_head=None,
        background_from_head_fn=lambda _head: set(),
        sanitize_fn=lambda dets, **_kwargs: (dets, 0),
        default_iou=0.75,
    )
    # scoreless_iou should canonicalize to 0.0 and avoid unexpected filtering.
    assert result["scoreless_removed"] == 0


def test_deep_prepass_detector_sahi_defaults_match_zero_inputs():
    calls = []

    def _run_detector_stub(**kwargs):
        calls.append(kwargs)
        return {"detections": [], "warnings": []}

    payload = SimpleNamespace(
        sahi_window_size=0,
        sahi_overlap_ratio=0.0,
        enable_yolo=True,
        enable_rfdetr=False,
        yolo_id=None,
        rfdetr_id=None,
        detector_mode="yolo",
        detector_id=None,
        detector_conf=None,
        detector_iou=None,
        detector_merge_iou=None,
        enable_sam3_text=False,
        enable_sam3_similarity=False,
    )
    _agent_run_deep_prepass_part_a_impl(
        payload,
        pil_img=Image.new("RGB", (64, 64)),
        image_token="token",
        labelmap=["car"],
        glossary="",
        run_detector_fn=_run_detector_stub,
        attach_provenance_fn=lambda *_args, **_kwargs: None,
        generate_sam3_synonyms_fn=lambda *_args, **_kwargs: {},
        generate_text_fn=lambda *_args, **_kwargs: "",
        extract_json_fn=lambda *_args, **_kwargs: {},
        default_synonyms={},
        label_key_fn=lambda value: value,
        sam3_text_windows_fn=lambda *_args, **_kwargs: [],
        ensure_sam3_text_runtime_fn=lambda: None,
        normalize_window_xyxy_fn=lambda *_args, **_kwargs: None,
        sam3_prompt_variants_fn=lambda *_args, **_kwargs: [],
        sam3_text_payloads_fn=lambda *_args, **_kwargs: [],
    )
    assert len(calls) == 2
    sahi_calls = [call for call in calls if call.get("sahi", {}).get("enabled")]
    assert len(sahi_calls) == 1
    assert sahi_calls[0]["sahi"]["slice_size"] == 640
    assert sahi_calls[0]["sahi"]["overlap"] == pytest.approx(0.2)


def test_deep_prepass_detector_sahi_defaults_match_invalid_inputs():
    calls = []

    def _run_detector_stub(**kwargs):
        calls.append(kwargs)
        return {"detections": [], "warnings": []}

    payload = SimpleNamespace(
        sahi_window_size=-128,
        sahi_overlap_ratio=1.5,
        enable_yolo=True,
        enable_rfdetr=False,
        yolo_id=None,
        rfdetr_id=None,
        detector_mode="yolo",
        detector_id=None,
        detector_conf=None,
        detector_iou=None,
        detector_merge_iou=None,
        enable_sam3_text=False,
        enable_sam3_similarity=False,
    )
    _agent_run_deep_prepass_part_a_impl(
        payload,
        pil_img=Image.new("RGB", (64, 64)),
        image_token="token",
        labelmap=["car"],
        glossary="",
        run_detector_fn=_run_detector_stub,
        attach_provenance_fn=lambda *_args, **_kwargs: None,
        generate_sam3_synonyms_fn=lambda *_args, **_kwargs: {},
        generate_text_fn=lambda *_args, **_kwargs: "",
        extract_json_fn=lambda *_args, **_kwargs: {},
        default_synonyms={},
        label_key_fn=lambda value: value,
        sam3_text_windows_fn=lambda *_args, **_kwargs: [],
        ensure_sam3_text_runtime_fn=lambda: None,
        normalize_window_xyxy_fn=lambda *_args, **_kwargs: None,
        sam3_prompt_variants_fn=lambda *_args, **_kwargs: [],
        sam3_text_payloads_fn=lambda *_args, **_kwargs: [],
    )
    sahi_calls = [call for call in calls if call.get("sahi", {}).get("enabled")]
    assert len(sahi_calls) == 1
    assert sahi_calls[0]["sahi"]["slice_size"] == 640
    assert sahi_calls[0]["sahi"]["overlap"] == pytest.approx(0.2)


def test_api_similarity_min_score_preserves_explicit_zero(monkeypatch):
    captured = {}

    def _fake_selector(min_score, **kwargs):
        captured["min_score"] = min_score
        captured["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(api, "_agent_select_similarity_exemplars_impl", _fake_selector)
    payload = SimpleNamespace(
        similarity_min_exemplar_score=0.0,
        similarity_exemplar_count=3,
        similarity_exemplar_strategy="top",
        similarity_exemplar_seed=None,
        similarity_exemplar_fraction=None,
        similarity_exemplar_min=None,
        similarity_exemplar_max=None,
        similarity_exemplar_source_quota=None,
    )
    api._agent_select_similarity_exemplars(payload, detections=[{"label": "car", "score": 0.1, "source": "yolo"}])
    assert captured["min_score"] == 0.0


def test_api_similarity_min_score_non_finite_falls_back(monkeypatch):
    captured = {}

    def _fake_selector(min_score, **kwargs):
        captured["min_score"] = min_score
        captured["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(api, "_agent_select_similarity_exemplars_impl", _fake_selector)
    payload = SimpleNamespace(
        similarity_min_exemplar_score=float("nan"),
        similarity_exemplar_count=3,
        similarity_exemplar_strategy="top",
        similarity_exemplar_seed=None,
        similarity_exemplar_fraction=None,
        similarity_exemplar_min=None,
        similarity_exemplar_max=None,
        similarity_exemplar_source_quota=None,
    )
    api._agent_select_similarity_exemplars(payload, detections=[{"label": "car", "score": 0.1, "source": "yolo"}])
    assert captured["min_score"] == pytest.approx(0.6)


def test_submit_annotations_preserves_explicit_zero_iou(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        api,
        "_agent_resolve_image",
        lambda *_args, **_kwargs: (Image.new("RGB", (64, 64)), None, "token"),
    )
    monkeypatch.setattr(api, "_agent_load_labelmap", lambda _dataset_id: ["car"])
    monkeypatch.setattr(api, "_agent_background_classes_from_head", lambda _head: set())
    monkeypatch.setattr(
        api,
        "_agent_sanitize_detection_items",
        lambda *_args, **_kwargs: ([{"label": "car", "bbox_xyxy_px": [0.0, 0.0, 10.0, 10.0]}], 0),
    )

    def _fake_merge(detections, *, iou_thr, max_det, cross_iou):
        captured["iou_thr"] = iou_thr
        captured["max_det"] = max_det
        captured["cross_iou"] = cross_iou
        return detections

    monkeypatch.setattr(api, "_agent_merge_detections", _fake_merge)

    api._agent_tool_submit_annotations(
        annotations=[],
        iou=0.0,
        cross_iou=0.0,
        max_det=None,
    )
    assert captured["iou_thr"] == 0.0
    assert captured["cross_iou"] is None
