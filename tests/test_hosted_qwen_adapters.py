"""Invoke the preserved Qwen pipeline with explicit CPU model doubles.

No real weights, provider calls or GPU acceptance is claimed. Prompting,
caption sequencing/recovery, detection parsing and runtime boundaries are real.
"""
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from services.hosted import qwen_adapters as adapters
from services.hosted import qwen_caption_pipeline as pipeline


class Batch(dict):
    @property
    def data(self): return self
    @data.setter
    def data(self, value):
        self.clear()
        self.update(value)
    @property
    def input_ids(self): return self["input_ids"]
    def to(self, device): return self


class ProcessorDouble:
    tokenizer = None  # Exercise the production blocking-generate path.
    image_processor = SimpleNamespace(patch_size=14)

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "\n".join(item["text"] for message in messages for item in message["content"] if item["type"] == "text")

    def __call__(self, **kwargs):
        self.inputs = kwargs
        return Batch(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones((1, 3)))

    def batch_decode(self, *args, **kwargs): return [self.output]


class ModelDouble:
    config = SimpleNamespace(max_position_embeddings=8192)
    generation_config = None

    def __init__(self, processor, calls, response):
        self.processor, self.calls, self.response = processor, calls, response

    def parameters(self): return iter([torch.zeros(1)])

    def generate(self, input_ids, **kwargs):
        images = self.processor.inputs.get("images") or []
        self.calls.append({"images": [image.size for image in images], "messages": self.processor.messages,
                           "max_new_tokens": kwargs.get("max_new_tokens"), "do_sample": kwargs.get("do_sample")})
        value = self.response(self.calls) if callable(self.response) else self.response
        if isinstance(value, BaseException):
            raise value
        self.processor.output = value
        return torch.cat([input_ids, torch.tensor([[4, 5]])], dim=1)


def runtime(monkeypatch, response="A red car is parked beside a low wall on a paved road.", *, config_overrides=None, **overrides):
    calls, models = [], []
    def load(definition, model_id, device, min_pixels, max_pixels):
        processor = ProcessorDouble()
        model = ModelDouble(processor, calls, response)
        model.model_id = model_id
        models.append(model)
        return pipeline.QwenRuntime(model, processor, model_id=model_id)
    monkeypatch.setattr(adapters, "_load_native", load)
    monkeypatch.setattr(adapters, "_vision_info", lambda messages: (
        [item["image"] for message in messages for item in message["content"] if item["type"] == "image"], None))
    config = {"actor_id": "annotator", "default_model_id": "Qwen3-VL-Instruct",
              "models": {"Qwen3-VL-Instruct": {"model_path": "/never-read-by-double", "files": {}}}}
    config.update(config_overrides or {})
    worker = adapters.QwenRuntime(config, "cpu", **overrides)
    return worker, calls, models


def test_full_caption_executes_real_pipeline_and_native_generation_seam(monkeypatch):
    worker, calls, models = runtime(monkeypatch)
    result = worker.execute("qwen_caption", Image.new("RGB", (80, 60)), {
        "user_prompt": "Describe the scene.", "restrict_to_labels": False, "include_counts": False,
        "final_caption_max_sentences": 2, "max_new_tokens": 128, "unload_others": True,
        "media_path": "/private/server/path", "media_sha256": "private-hash", "source_media_hash": "a" * 64}, {"actor_id": "annotator"})
    assert "red car" in result["caption"]
    assert len(calls) >= 1 and calls[0]["images"] == [(80, 60)]
    assert calls[0]["max_new_tokens"] == 128
    assert any(row.get("event") == "input" for row in result["hosted_evidence"]["model_io"])
    assert any(row.get("event") == "output" for row in result["hosted_evidence"]["model_io"])
    assert "/private/server/path" not in str(result) and "private-hash" not in str(result)
    assert len(models) == 1 and worker.image is None
    worker.close()
    assert not worker.loaded


@pytest.mark.parametrize("strategy", ["text_only", "visual"])
def test_windowed_caption_keeps_visual_windows_and_text_composition(monkeypatch, strategy):
    worker, calls, _ = runtime(monkeypatch)
    result = worker.execute("qwen_caption", Image.new("RGB", (800, 400)), {
        "caption_mode": "windowed", "window_size": 400, "window_overlap": 0,
        "caption_windowed_full_image_strategy": strategy, "caption_all_windows": True,
        "restrict_to_labels": False, "include_counts": False,
        "final_caption_max_sentences": 2, "max_new_tokens": 128}, {"actor_id": "annotator"})
    image_calls = [call for call in calls if call["images"]]
    text_calls = [call for call in calls if not call["images"]]
    assert len(image_calls) >= 2 and all(call["images"] == [(400, 400)] for call in image_calls[:2])
    assert text_calls, "window merge must remain a text-only reconciliation stage"
    evidence = result["hosted_evidence"]
    assert sum(row.get("event") == "caption_window" for row in evidence["model_io"]) == 2
    if strategy == "text_only":
        assert all(call["images"] == [(400, 400)] for call in image_calls)
        assert any("Text-only full-image composition" in row.get("step_detail", "") for row in evidence["progress"])
    else:
        assert any(call["images"] == [(800, 400)] for call in image_calls)
        assert any(row.get("phase") == "merge" for row in evidence["progress"])
    assert result["caption"]


def test_thinking_caption_preserves_two_stage_refinement_model(monkeypatch):
    worker, calls, models = runtime(monkeypatch, config_overrides={
        "default_model_id": "Qwen3-VL-Thinking", "refinement_model_id": "Qwen3-VL-Instruct",
        "models": {name: {"model_path": "/double", "files": {}} for name in ("Qwen3-VL-Thinking", "Qwen3-VL-Instruct")}})
    result = worker.execute("qwen_caption", Image.new("RGB", (80, 60)), {
        "two_stage_refine": True, "restrict_to_labels": False, "include_counts": False,
        "max_new_tokens": 128, "final_caption_max_sentences": 2}, {"actor_id": "annotator"})
    assert {model.model_id for model in models} == {"Qwen3-VL-Thinking", "Qwen3-VL-Instruct"}
    phases = {row.get("phase") for row in result["hosted_evidence"]["progress"]}
    assert {"draft", "refine"} <= phases
    assert any(call["images"] for call in calls) and any(not call["images"] for call in calls)


@pytest.mark.parametrize("strategy", ["text_only", "visual"])
def test_caption_prompt_preview_keeps_real_window_refinement_plan_without_loading(monkeypatch, strategy):
    worker, calls, models = runtime(monkeypatch, config_overrides={
        "default_model_id": "Qwen3-VL-Thinking", "refinement_model_id": "Qwen3-VL-Instruct",
        "models": {name: {"model_path": "/double", "files": {}} for name in ("Qwen3-VL-Thinking", "Qwen3-VL-Instruct")}})
    result = worker.execute("qwen_caption_preview", Image.new("RGB", (800, 400)), {
        "caption_mode": "windowed", "window_size": 400, "window_overlap": 0,
        "caption_windowed_full_image_strategy": strategy, "caption_all_windows": True,
        "two_stage_refine": True, "user_prompt": "Describe the road and vehicles.",
        "label_hints": [{"label": "car", "bbox": [40, 20, 100, 60]}],
        "image_width": 12345, "image_height": 12345, "max_new_tokens": 128, "source_media_hash": "a" * 64,
    }, {"actor_id": "annotator"})
    assert calls == models == [], "preview must not load or run a model"
    assert result["meta"]["planned_windows"] == 2
    assert (result["meta"]["image_width"], result["meta"]["image_height"]) == (800, 400)
    assert result["meta"]["caption_windowed_full_image_strategy"] == strategy
    assert result["meta"]["two_stage_refine"] and result["used_counts"] == {"car": 1}
    assert result["meta"]["refinement_model_id"] == "Qwen3-VL-Instruct"
    assert "Describe the road and vehicles." in result["full_text"]
    assert result["step_plan"] and result["meta"]["prompt_budget"]["explicit_max_new_tokens"]
    assert any("Conditional" in section["title"] for section in result["sections"])
    assert worker.image is None


def test_caption_loop_recovery_preserves_raw_failure_and_retries_private_model(monkeypatch):
    def response(calls):
        if len(calls) == 1:
            return pipeline.QwenCaptionLoopDetected("qwen_caption_repetition_loop", raw_output="A car repeats. " * 15,
                                                   trimmed_output="A car repeats.")
        return "A red car is parked beside a low wall on a paved road."
    worker, calls, models = runtime(monkeypatch, response)
    result = worker.execute("qwen_caption", Image.new("RGB", (80, 60)), {
        "restrict_to_labels": False, "include_counts": False, "max_new_tokens": 128,
        "caption_loop_recovery_mode": "safe_retry", "final_caption_max_sentences": 2}, {"actor_id": "annotator"})
    assert len(calls) >= 2 and len(models) >= 2
    assert result["recovery_events"]
    assert "A car repeats." in str(result["hosted_evidence"]["model_io"])
    assert "red car" in result["caption"]


@pytest.mark.parametrize("kind", ["bbox", "bbox_sam", "point"])
def test_qwen_detection_preserves_parsing_and_same_image_sam_handoff(monkeypatch, kind):
    raw = '[{"point_2d":[500,500],"label":"car"}]' if kind == "point" else '[{"bbox_2d":[100,200,400,600],"label":"car"}]'
    sam_calls = []
    def sam(image, **options):
        sam_calls.append((image.size, options))
        mask = np.zeros((1, image.height, image.width), dtype=np.uint8)
        mask[:, 20:40, 10:30] = 1
        return mask, [1], None
    worker, calls, _ = runtime(monkeypatch, raw, sam_predict=sam)
    result = worker.execute("qwen_infer", Image.new("RGB", (100, 100)),
                            {"item_list": "car", "prompt_type": kind, "source_media_hash": "a" * 64}, {"actor_id": "annotator"})
    assert len(result["boxes"]) == 1 and result["raw_response"] == raw
    assert result["boxes"][0]["qwen_label"] == "car"
    if kind == "bbox":
        assert result["boxes"][0]["bbox"] == pytest.approx([.25, .4, .3, .4])
        assert not sam_calls
    else:
        assert sam_calls[0][0] == (100, 100)
        assert result["boxes"][0]["bbox"] == pytest.approx([.2, .3, .2, .2])
        if kind == "point":
            np.testing.assert_allclose(sam_calls[0][1]["point_coords"], [[50, 50]])


def test_unregistered_models_provider_keys_and_owner_fail_before_generation(monkeypatch):
    worker, calls, _ = runtime(monkeypatch)
    image = Image.new("RGB", (10, 10))
    for payload, actor in [({"model_id": "not-registered"}, "annotator"),
                           ({"openai_api_key": "must-not-pass"}, "annotator"), ({"source_media_hash": "not-a-digest"}, "annotator"), ({}, "different-user")]:
        with pytest.raises(ValueError):
            worker.execute("qwen_caption", image, payload, {"actor_id": actor})
    assert not calls


def test_two_worker_namespaces_do_not_share_image_progress_or_models(monkeypatch):
    first, calls, models = runtime(monkeypatch)
    second = adapters.QwenRuntime({**first.config, "actor_id": "second"}, "cpu")
    payload = {"restrict_to_labels": False, "include_counts": False, "max_new_tokens": 128}
    first.execute("qwen_caption", Image.new("RGB", (40, 30)), payload, {"actor_id": "annotator"})
    saved = list(first.events)
    second.execute("qwen_caption", Image.new("RGB", (80, 60)), payload, {"actor_id": "second"})
    assert models[0] is not models[1]
    assert calls[0]["images"] == [(40, 30)] and calls[-1]["images"] == [(80, 60)]
    assert first.events == saved and first.progress is not second.progress
    assert first.pipeline["resolve_image_payload"].__self__ is first
    assert second.pipeline["resolve_image_payload"].__self__ is second
    assert first.loaded is not second.loaded and first.image is second.image is None


def test_model_snapshot_manifest_checks_real_files_and_rejects_changed_or_extra(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b'{}')
    files = {"config.json": sha256(b'{}').hexdigest()}
    assert adapters._verify_snapshot(model, files) == model.resolve()
    (model / "config.json").write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError, match="snapshot_changed"):
        adapters._verify_snapshot(model, files)
    (model / "extra").write_bytes(b'x')
    with pytest.raises(ValueError, match="inventory_changed"):
        adapters._verify_snapshot(model, files)


def test_model_snapshot_rejects_symlink(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (tmp_path / "outside").write_bytes(b'{}')
    (model / "config.json").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="link_invalid"):
        adapters._verify_snapshot(model, {"config.json": sha256(b'{}').hexdigest()})
