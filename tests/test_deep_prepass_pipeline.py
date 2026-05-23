from contextvars import ContextVar
from PIL import Image
from types import SimpleNamespace

from fastapi import HTTPException

import localinferenceapi as api
from services.prepass import _agent_run_deep_prepass_caption_impl
from services.prepass_similarity import _agent_run_similarity_global


def test_deep_prepass_runs_detectors_and_similarity(monkeypatch):
    calls = []
    sim_calls = []

    def fake_run_detector(
        image_token=None,
        detector_id=None,
        mode=None,
        conf=None,
        sahi=None,
        max_det=None,
        iou=None,
        merge_iou=None,
        expected_labelmap=None,
        register=None,
    ):
        calls.append((mode, bool(sahi and sahi.get("enabled"))))
        return {
            "detections": [
                {
                    "label": "car",
                    "score": 0.9,
                    "bbox_xyxy_px": [0.0, 0.0, 10.0, 10.0],
                    "bbox_2d": [0, 0, 100, 100],
                    "source": mode,
                }
            ]
        }

    def fake_sam3_similarity(*args, **kwargs):
        sim_calls.append(kwargs.get("label"))
        return {"detections": []}

    def fake_similarity_payloads(*args, **kwargs):
        sim_calls.append(kwargs.get("label"))
        return []

    monkeypatch.setattr(api, "_agent_tool_run_detector", fake_run_detector)
    monkeypatch.setattr(api, "_agent_tool_sam3_similarity", fake_sam3_similarity)
    monkeypatch.setattr(api, "_sam3_similarity_payloads_from_state", fake_similarity_payloads)
    monkeypatch.setattr(api, "_agent_tool_sam3_text", lambda **_: {"detections": []})
    monkeypatch.setattr(
        api,
        "_agent_generate_sam3_synonyms",
        lambda labels, glossary, max_synonyms=None: ({}, {label: {"base_terms": [label], "expanded_terms": []} for label in labels}),
    )

    payload = api.QwenPrepassRequest(
        dataset_id="",
        enable_yolo=True,
        enable_rfdetr=True,
        enable_sam3_text=False,
        enable_sam3_similarity=True,
        prepass_keep_all=True,
        detector_conf=0.2,
        sahi_window_size=64,
        sahi_overlap_ratio=0.2,
        prepass_similarity_score=0.2,
        sam3_mask_threshold=0.2,
        similarity_window_extension=False,
    )
    pil_img = Image.new("RGB", (64, 64), (0, 0, 0))
    result = api._agent_run_deep_prepass(
        payload,
        pil_img=pil_img,
        image_token="tok",
        labelmap=["car"],
        glossary="",
    )

    assert result["detections"]
    assert sum(1 for mode, _ in calls if mode == "yolo") == 2
    assert sum(1 for mode, _ in calls if mode == "rfdetr") == 2
    assert any(enabled for _, enabled in calls)
    assert len(sim_calls) == 1
    provenance = result.get("provenance") or {}
    atoms = list(provenance.get("atoms") or [])
    assert atoms
    stage_atoms = provenance.get("stage_atoms") or {}
    detector_runs = (stage_atoms.get("detector") or {})
    assert "yolo_full" in detector_runs
    assert "yolo_sahi" in detector_runs
    assert "rfdetr_full" in detector_runs
    assert "rfdetr_sahi" in detector_runs
    final_clusters = list(provenance.get("final_clusters") or [])
    assert final_clusters
    assert final_clusters[0].get("atom_ids")


def test_deep_prepass_caption_sanitizes_hints_and_token_config():
    captured = {}
    pil_img = Image.new("RGB", (64, 64), (0, 0, 0))
    payload = SimpleNamespace(
        prepass_caption=True,
        prepass_caption_profile=123,
        prepass_caption_max_tokens="not-an-int",
        prepass_caption_variant="thinking",
        model_variant="auto",
    )
    detections = [
        {"label": "car", "score": 0.9, "bbox_xyxy_px": [0, 0, 12, 12]},
        {"label": "truck", "score": 0.8, "bbox_xyxy_px": [0, 0, float("inf"), 12]},
        {"label": "person", "score": 0.7, "bbox_xyxy_px": [0, "bad", 12, 12]},
    ]

    def qwen_caption_fn(request):
        captured["request"] = request
        return SimpleNamespace(caption="A car is parked beside the road.")

    caption, windows = _agent_run_deep_prepass_caption_impl(
        payload,
        pil_img=pil_img,
        image_token="tok",
        detections=detections,
        model_id_override=None,
        glossary=None,
        grid_for_log=None,
        caption_request_cls=api.QwenCaptionRequest,
        qwen_caption_fn=qwen_caption_fn,
        sanitize_caption_fn=api._sanitize_qwen_caption_impl,
        label_counts_fn=lambda _items, limit=10: "car: 1",
        qwen_bbox_to_xyxy_fn=lambda _w, _h, bbox: bbox,
        xyxy_to_bbox_fn=lambda *_args: [0, 0, 100, 100],
        grid_cell_for_window_bbox_fn=lambda *_args: None,
        readable_format_bbox_fn=lambda _bbox: "bbox",
        unload_non_qwen_fn=lambda: None,
        caption_window_hook=ContextVar("caption_window_hook", default=None),
        http_exception_cls=HTTPException,
        http_503_code=503,
    )

    request = captured["request"]
    assert caption == "A car is parked beside the road."
    assert windows == []
    assert request.max_new_tokens == 512
    assert request.model_variant == "Thinking"
    assert [hint.label for hint in request.label_hints] == ["car"]


def test_deep_prepass_caption_clamps_large_token_config():
    captured = {}
    pil_img = Image.new("RGB", (64, 64), (0, 0, 0))
    payload = SimpleNamespace(
        prepass_caption=True,
        prepass_caption_profile="deep",
        prepass_caption_max_tokens=99999,
        prepass_caption_variant="auto",
        model_variant="auto",
    )

    def qwen_caption_fn(request):
        captured["request"] = request
        return SimpleNamespace(caption="A detailed scene caption is available.")

    _agent_run_deep_prepass_caption_impl(
        payload,
        pil_img=pil_img,
        image_token="tok",
        detections=[],
        model_id_override=None,
        glossary=None,
        grid_for_log=None,
        caption_request_cls=api.QwenCaptionRequest,
        qwen_caption_fn=qwen_caption_fn,
        sanitize_caption_fn=api._sanitize_qwen_caption_impl,
        label_counts_fn=lambda _items, limit=10: "none",
        qwen_bbox_to_xyxy_fn=lambda _w, _h, bbox: bbox,
        xyxy_to_bbox_fn=lambda *_args: [0, 0, 100, 100],
        grid_cell_for_window_bbox_fn=lambda *_args: None,
        readable_format_bbox_fn=lambda _bbox: "bbox",
        unload_non_qwen_fn=lambda: None,
        caption_window_hook=ContextVar("caption_window_hook", default=None),
        http_exception_cls=HTTPException,
        http_503_code=503,
    )

    assert captured["request"].max_new_tokens == 2000


def test_sam3_similarity_multi_prompt_seed_filter_no_type_error(monkeypatch):
    class DummyProcessor:
        def set_confidence_threshold(self, *_args, **_kwargs):
            return None

        def set_image(self, _img):
            return {}

        def add_geometric_prompt(self, _box, _label, state=None):
            return state or {}

    monkeypatch.setattr(api, "_ensure_sam3_text_runtime", lambda: (None, DummyProcessor(), None))
    monkeypatch.setattr(
        api,
        "_sam3_text_detections",
        lambda *args, **kwargs: [SimpleNamespace(bbox=[0.8, 0.8, 0.1, 0.1], score=0.91)],
    )

    pil_img = Image.new("RGB", (64, 64), (0, 0, 0))
    detections = api._run_sam3_visual_inference_multi(
        pil_img,
        bboxes_xywh=[(0.0, 0.0, 8.0, 8.0)],
        bbox_labels=None,
        threshold=0.2,
        mask_threshold=0.2,
        limit=None,
    )

    assert len(detections) == 1


def test_sam3_similarity_visual_prompt_uses_semantic_text(monkeypatch):
    class DummyProcessor:
        def __init__(self):
            self.prompts = []

        def set_confidence_threshold(self, *_args, **_kwargs):
            return None

        def set_image(self, _img):
            return {"backbone_out": {}}

        def set_text_prompt(self, prompt, state=None):
            self.prompts.append(prompt)
            state = state or {}
            state["semantic_prompt"] = prompt
            return state

        def add_geometric_prompt(self, _box, _label, state=None):
            assert state and state.get("semantic_prompt") == "utility pole"
            return state

    processor = DummyProcessor()
    monkeypatch.setattr(api, "_ensure_sam3_text_runtime", lambda: (None, processor, None))
    monkeypatch.setattr(api, "_sam3_text_detections", lambda *args, **kwargs: [])

    pil_img = Image.new("RGB", (64, 64), (0, 0, 0))
    api._run_sam3_visual_inference_multi(
        pil_img,
        bboxes_xywh=[(0.0, 0.0, 8.0, 8.0)],
        bbox_labels=None,
        threshold=0.55,
        mask_threshold=0.2,
        limit=None,
        text_prompt="utility pole",
    )

    assert processor.prompts == ["utility pole"]


def test_similarity_global_reports_type_error_warning():
    payload = SimpleNamespace(prepass_similarity_score=0.3, sam3_mask_threshold=0.2)
    pil_img = Image.new("RGB", (64, 64), (0, 0, 0))

    def _raise_type_error(**_kwargs):
        raise TypeError("bad similarity arg type")

    result = _agent_run_similarity_global(
        payload,
        pil_img=pil_img,
        image_token="tok",
        exemplars_by_label={"car": [{"bbox_2d": [0, 0, 100, 100], "handle": "h1"}]},
        sam3_similarity_fn=_raise_type_error,
    )
    warnings = list(result.get("warnings") or [])
    assert warnings
    assert "deep_prepass_similarity_type_error:car:full:" in warnings[0]
