from __future__ import annotations

from PIL import Image

import localinferenceapi


def test_zoom_and_detect_register_default_adds_summary(monkeypatch) -> None:
    monkeypatch.setattr(localinferenceapi, "_AGENT_ACTIVE_DETECTOR_CONF", None)
    monkeypatch.setattr(
        localinferenceapi,
        "_agent_resolve_image",
        lambda _b64, _tok: (Image.new("RGB", (100, 100)), None, None),
    )
    monkeypatch.setattr(localinferenceapi, "_normalize_window_xyxy", lambda *_args, **_kwargs: (0.0, 0.0, 10.0, 10.0))
    monkeypatch.setattr(
        localinferenceapi,
        "_agent_tool_run_detector",
        lambda **_kwargs: {"detections": [{"bbox_2d": [0.0, 0.0, 10.0, 10.0], "source": "yolo"}]},
    )
    monkeypatch.setattr(
        localinferenceapi,
        "_agent_register_detections",
        lambda *_args, **_kwargs: {"clusters": 1},
    )

    out = localinferenceapi._agent_tool_zoom_and_detect(window_bbox_2d=[0.0, 0.0, 0.1, 0.1])

    assert out.get("register_summary") == {"clusters": 1}
    assert out.get("detections")[0].get("source_list") == ["yolo"]


def test_zoom_and_detect_register_false_skips_registration(monkeypatch) -> None:
    monkeypatch.setattr(localinferenceapi, "_AGENT_ACTIVE_DETECTOR_CONF", None)
    monkeypatch.setattr(
        localinferenceapi,
        "_agent_resolve_image",
        lambda _b64, _tok: (Image.new("RGB", (100, 100)), None, None),
    )
    monkeypatch.setattr(localinferenceapi, "_normalize_window_xyxy", lambda *_args, **_kwargs: (0.0, 0.0, 10.0, 10.0))
    monkeypatch.setattr(
        localinferenceapi,
        "_agent_tool_run_detector",
        lambda **_kwargs: {"detections": [{"bbox_2d": [0.0, 0.0, 10.0, 10.0], "source": "yolo"}]},
    )
    monkeypatch.setattr(
        localinferenceapi,
        "_agent_register_detections",
        lambda *_args, **_kwargs: {"clusters": 99},
    )

    out = localinferenceapi._agent_tool_zoom_and_detect(
        window_bbox_2d=[0.0, 0.0, 0.1, 0.1],
        register=False,
    )

    assert "register_summary" not in out
