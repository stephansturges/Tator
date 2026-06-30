from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from tools import run_openai_caption_batch_smoke as batch_smoke


def test_batch_line_uses_responses_vision_file_and_glossary_terms(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image_path = images / "scene.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(image_path)
    label_path = labels / "scene.txt"
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (dataset / "labelmap.txt").write_text("RawThing\n", encoding="utf-8")
    request_json = tmp_path / "request.json"
    request_json.write_text(
        json.dumps({"labelmap_glossary": {"RawThing": ["canonical object"]}}),
        encoding="utf-8",
    )
    case = {
        "name": "sample_001",
        "stem": "scene",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "class_counts": {"RawThing": 1},
        "caption_mode": "full",
    }
    args = SimpleNamespace(
        request_json=request_json,
        max_output_tokens=3200,
        max_boxes=50,
        qa_count=8,
        model="gpt-5.5",
        reasoning_effort="high",
        image_detail="original",
    )

    line = batch_smoke.build_batch_line(
        case=case,
        file_id="file_vision_123",
        dataset_root=dataset,
        args=args,
    )

    assert line["method"] == "POST"
    assert line["url"] == "/v1/responses"
    body = line["body"]
    assert body["model"] == "gpt-5.5"
    assert body["reasoning"] == {"effort": "high"}
    assert body["text"]["format"]["type"] == "json_schema"
    content = body["input"][0]["content"]
    assert content[1] == {"type": "input_image", "file_id": "file_vision_123", "detail": "original"}
    prompt = content[0]["text"]
    assert "canonical object" in prompt
    assert '"canonical object": 1' in prompt
    assert '"RawThing": 1' not in prompt
