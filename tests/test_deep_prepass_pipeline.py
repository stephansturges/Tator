from PIL import Image
from types import SimpleNamespace

import localinferenceapi as api
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

    monkeypatch.setattr(api, "_agent_tool_run_detector", fake_run_detector)
    monkeypatch.setattr(api, "_agent_tool_sam3_similarity", fake_sam3_similarity)
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
