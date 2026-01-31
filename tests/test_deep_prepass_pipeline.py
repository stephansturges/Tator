from PIL import Image

import localinferenceapi as api


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
