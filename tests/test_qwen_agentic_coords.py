import json

from localinferenceapi import (
    AgentToolResult,
    _agent_compact_tool_response,
    _parse_tool_call_json,
    _resolve_agent_bbox_xyxy,
    _window_local_bbox_2d_to_full_xyxy,
)


def test_window_bbox_2d_remap_full_image():
    img_w = 1000
    img_h = 1000
    window_bbox_2d = [250, 250, 750, 750]
    local_bbox_2d = [0, 0, 1000, 1000]
    xyxy = _window_local_bbox_2d_to_full_xyxy(img_w, img_h, window_bbox_2d, local_bbox_2d)
    assert xyxy == (250.0, 250.0, 750.0, 750.0)


def test_resolve_agent_bbox_xyxy_window_space():
    ann = {
        "bbox_2d": [0, 0, 1000, 1000],
        "bbox_space": "window",
    }
    xyxy = _resolve_agent_bbox_xyxy(ann, 1000, 1000, window_bbox_2d=[100, 200, 300, 400])
    assert xyxy == (100.0, 200.0, 300.0, 400.0)


def test_parse_tool_call_with_thought_action():
    payload = {"name": "run_detector", "arguments": {"mode": "yolo"}}
    raw = "Thought: run detector\nAction: detect\n<tool_call>\n%s\n</tool_call>" % json.dumps(payload)
    parsed, err = _parse_tool_call_json(raw)
    assert err is None
    assert parsed == payload


def test_tool_response_includes_error():
    result = AgentToolResult(name="run_detector", result={"detections": []}, error="tool_failed:test")
    compact = _agent_compact_tool_response(result)
    assert compact["error"] == "tool_failed:test"
    assert "result" in compact
