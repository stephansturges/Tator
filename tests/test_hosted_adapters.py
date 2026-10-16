"""Adapter boundary checks with real image/geometry helpers, no model claim."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from services.hosted import adapters
from services.hosted.config import ModelRegistry, file_digest, verify_checkpoint
from services.hosted.runtime import RequestContext
from utils.coco import _decode_binary_mask_impl


@pytest.fixture
def assets(tmp_path):
    media = tmp_path / "media"
    weights = tmp_path / "models"
    media.mkdir()
    weights.mkdir()
    checkpoint = weights / "sam.pt"
    checkpoint.write_bytes(b"test-only model identity")
    image = media / "image.png"
    Image.new("RGB", (64, 48), "red").save(image)
    registry = ModelRegistry(allowed_roots=[weights], definitions={
        "sam": {"kind": "sam1", "checkpoint": str(checkpoint), "model_type": "vit_b"},
        "detector": {"kind": "yolo", "checkpoint": str(checkpoint), "labels": ["car", "bus"]},
    })
    payload = {"media_path": str(image), "media_sha256": file_digest(image)}
    context = RequestContext("alice", "dataset", "image", "version-1", 1, "editor", session_id="session-hash", instance_id="server-1")
    return registry, media, checkpoint, payload, context


def test_registry_identity_changes_with_artifact_and_rejects_path_escape(assets, tmp_path):
    registry, media, checkpoint, _, _ = assets
    selected = registry.user_spec("alice", "sam", media_root=media, device="cpu")
    definition = registry.definition("sam")
    definition["checkpoint"] = "/not/a/real/file"
    assert registry.definition("sam")["checkpoint"] == str(checkpoint)
    assert selected.factory == "services.hosted.adapters:UserToolAdapter"
    checkpoint.write_bytes(b"changed test-only identity")
    with pytest.raises(ValueError, match="hosted_model_artifact_changed"):
        verify_checkpoint(selected.config["model"])
    registry.register("sam", {"kind": "sam1", "checkpoint": str(checkpoint), "model_type": "vit_b"})
    assert selected.identity != registry.user_spec("alice", "sam", media_root=media, device="cpu").identity
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not in approved model root")
    with pytest.raises(ValueError, match="hosted_model_path_invalid"):
        registry.register("escape", {"kind": "sam1", "checkpoint": str(outside)})
    alias = checkpoint.parent / "alias.pt"
    alias.symlink_to(checkpoint)
    with pytest.raises(ValueError, match="hosted_model_path_invalid"):
        registry.register("alias", {"kind": "sam1", "checkpoint": str(alias)})
    with pytest.raises(ValueError, match="hosted_model_settings_unknown"):
        registry.register("plugin", {"kind": "sam1", "checkpoint": str(checkpoint), "factory": "evil:Plugin"})
    config_file = checkpoint.parent / "config.json"
    config_file.write_text('{}')
    with pytest.raises(ValueError, match="hosted_model_member_invalid"):
        registry.register("qwen", {"kind": "qwen", "model_path": str(checkpoint.parent), "files": {"../outside.pt": file_digest(outside), "config.json": file_digest(config_file)}})


class PredictorDouble:
    def __init__(self):
        self.loads = 0
        self.options = None

    def set_image(self, image):
        self.loads += 1
        self.image = image

    def predict(self, **options):
        self.options = options
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        mask[10:20, 15:30] = True
        return mask[None], np.asarray([0.9]), None


def test_sam_reuses_exact_context_and_produces_existing_mask_format(assets, monkeypatch):
    registry, media, _, payload, context = assets
    predictors = []
    def build(*_):
        predictor = PredictorDouble()
        predictors.append(predictor)
        return predictor, object()
    monkeypatch.setattr(adapters, "_build_sam", build)
    spec = registry.user_spec("alice", "sam", media_root=media, device="cpu")
    adapter = adapters.UserToolAdapter(config=spec.config, device="cpu")
    kwargs = {**payload, "point_x": 20, "point_y": 15, "uuid": "prompt"}
    preloaded = adapter.execute("sam_preload", {**payload, "slot": "next"}, context.as_dict())
    state = adapter.execute("sam_state", {}, context.as_dict())
    assert state["capacity"] == 3 and state["slots"][1]["image_id"] == context.image_id
    assert state["slots"][1]["token"] == preloaded["image_token"]
    assert state["slots"][0]["token"] is None and not state["slots"][2]["loaded"]
    assert not any(row["image_id"] for row in adapter.execute("sam_state", {}, replace(context, dataset_id="other").as_dict())["slots"])
    result = adapter.execute("sam_point", kwargs, context.as_dict())
    assert len(predictors) == 2 and predictors[0].loads == 0 and predictors[1].loads == 1
    assert result["image_token"] == preloaded["image_token"]
    assert preloaded["token"] == preloaded["image_token"] and preloaded["backend"] == "sam1"
    assert result["bbox"] == pytest.approx([22.5 / 64, 15 / 48, 15 / 64, 10 / 48])
    mask = _decode_binary_mask_impl(result["mask"])
    assert mask.shape == (48, 64) and mask.sum() == 150
    moved_session = replace(context, session_id="another-session")
    adapter.execute("sam_point", kwargs, moved_session.as_dict())
    assert predictors[0].loads == 0 and predictors[1].loads == 1
    moved_edit = replace(context, annotation_version="saved-version", editor_id="next-tab", ownership_generation=2)
    adapter.execute("sam_point", kwargs, moved_edit.as_dict())
    assert predictors[1].loads == 1  # Unchanged pixels reuse features, not authority.
    other = adapters.UserToolAdapter(config=spec.config, device="cpu")
    other.execute("sam_point", kwargs, context.as_dict())
    assert predictors[2] is not predictors[0]
    with pytest.raises(ValueError, match="hosted_runtime_owner_mismatch"):
        adapter.execute("sam_point", kwargs, replace(context, actor_id="bob").as_dict())
    with pytest.raises(ValueError, match="hosted_runtime_owner_mismatch"):
        adapter.execute("sam_state", {}, replace(context, actor_id="bob").as_dict())
    with pytest.raises(ValueError, match="hosted_classifier_not_configured"):
        adapter.execute("sam_point_auto", kwargs, context.as_dict())


def test_sam_auto_classifies_actual_mask_crop_and_maps_stable_class(assets, monkeypatch):
    registry, media, checkpoint, payload, context = assets
    metadata = checkpoint.parent / "head.meta.pkl"
    metadata.write_bytes(b"fixture metadata; loading is replaced only in this test")
    registry.register("classifier", {"kind": "classifier", "checkpoint": str(checkpoint),
        "metadata_path": str(metadata), "labels": ["car", "bus"]})
    calls = []
    class ClassifierDouble:
        def __init__(self, config, device):
            assert config["metadata_sha256"] == file_digest(metadata)
        def predict(self, image, bbox, background_guard):
            calls.append((image.size, list(bbox), background_guard))
            return {"label": "car", "proba": .9, "second_label": "bus", "second_proba": .1, "margin": .8}
        def close(self):
            calls.append("closed")
    from services.hosted import assistance_adapters
    monkeypatch.setattr(assistance_adapters, "ClassifierRuntime", ClassifierDouble)
    monkeypatch.setattr(adapters, "_build_sam", lambda *_: (PredictorDouble(), object()))
    selected = registry.user_spec("alice", "sam", media_root=media, device="cpu", classifier_id="classifier")
    config = {**dict(selected.config), "classifier_bindings": {
        "car": {"id": "class-2", "name": "renamed car", "position": 2},
        "bus": {"id": "class-0", "name": "bus", "position": 0}}}
    adapter = adapters.UserToolAdapter(config=config, device="cpu")
    result = adapter.execute("sam_point_auto", {**payload, "point_x": 20, "point_y": 15}, context.as_dict())
    assert calls == [((64, 48), [15, 10, 30, 20], True)]
    assert (result["prediction"], result["class_id"], result["stable_class_id"], result["second_label"]) == ("renamed car", "2", "class-2", "bus")
    assert result["proba"] == .9 and result["mask"]
    adapter.close()
    assert calls[-1] == "closed"


def test_private_user_adapter_executes_saved_sam3_chain(assets, monkeypatch, tmp_path):
    from services.hosted.sam3_chain import Sam3ChainRuntime
    registry, media, _, payload, context = assets
    root = tmp_path / "saved-recipe"
    root.mkdir()
    recipe = root / "recipe.json"
    recipe.write_text(json.dumps({"class_name": "car", "recipe": {"schema_version": 2, "steps": [{"prompt": "car", "max_visual_seeds": 0}]}}))
    definition = {"kind": "sam3_recipe", "model_id": "car-recipe", "model_path": str(root), "recipe_file": "recipe.json",
        "files": {"recipe.json": file_digest(recipe)}, "checkpoint_sha256": file_digest(recipe),
        "component_models": {"sam3": {"kind": "sam3", "model_id": "saved-sam3"}}}
    calls = []
    class PromptModel:
        def predict(self, image, parameters, *, visual=False):
            calls.append((image.size, parameters["text_prompt"]))
            return {"detections": [{"bbox": [.5, .5, .25, .25], "score": .9, "mask": None}]}
        def close(self):
            pass
    monkeypatch.setattr(adapters, "_build_sam", lambda *_: (PredictorDouble(), object()))
    monkeypatch.setattr(Sam3ChainRuntime, "_sam", lambda *_: PromptModel())
    spec = registry.user_spec("alice", "sam", media_root=media, device="cpu")
    config = {**dict(spec.config), "sam3_recipes": {"actor_id": "alice", "recipes": {"car-recipe": definition},
        "class_bindings": {"car": {"id": "stable-car", "name": "car", "position": 2}}}}
    adapter = adapters.UserToolAdapter(config=config, device="cpu")
    result = adapter.execute("sam3_recipe_chain", {**payload, "steps": [{"recipe_id": "car-recipe"}]}, context.as_dict())
    assert calls == [((64, 48), "car")]
    assert result["detections"][0]["stable_class_id"] == "stable-car"
    assert result["detections"][0]["class_id"] == 2
    assert result["hosted_evidence"]["execution_contract"] == "hosted-sam3-step-chain-v1"
    assert result["image_token"]
    adapter.close()


def test_media_identity_and_geometry_fail_before_inference(assets, monkeypatch, tmp_path):
    registry, media, _, payload, context = assets
    predictor = PredictorDouble()
    monkeypatch.setattr(adapters, "_build_sam", lambda *_: (predictor, object()))
    spec = registry.user_spec("alice", "sam", media_root=media, device="cpu", capacity=1)
    adapter = adapters.UserToolAdapter(config=spec.config, device="cpu")
    with pytest.raises(ValueError, match="hosted_media_identity_changed"):
        adapter.execute("sam_point", {**payload, "media_sha256": "wrong"}, context.as_dict())
    outside = tmp_path / "external.png"
    Image.new("RGB", (8, 8)).save(outside)
    with pytest.raises(ValueError, match="hosted_media_path_invalid"):
        adapter.execute("sam_point", {"media_path": str(outside), "media_sha256": file_digest(outside)}, context.as_dict())
    with pytest.raises(ValueError, match="hosted_predictor_slot_disabled"):
        adapter.execute("sam_preload", {**payload, "slot": "next"}, context.as_dict())
    with pytest.raises(ValueError, match="hosted_sam_points_outside_image"):
        adapter.execute("sam_point", {**payload, "point_x": 100, "point_y": 1}, context.as_dict())
    assert predictor.options is None


def test_sam3_prompts_reuse_private_network_and_keep_mask_classification(assets, monkeypatch):
    registry, media, checkpoint, payload, context = assets
    bpe = checkpoint.parent / "bpe.gz"
    bpe.write_bytes(b"fixture tokenizer")
    registry.register("sam3", {"kind": "sam3", "checkpoint": str(checkpoint), "bpe_path": str(bpe)})
    from services.hosted import assistance_adapters
    from utils.coco import _encode_binary_mask_impl
    model = object()
    created = []
    class PromptsDouble:
        def __init__(self, network, device):
            assert network is model
            created.append(self)
        def predict(self, image, values, visual=False):
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
            mask[10:20, 15:30] = 1
            return {"detections": [{"bbox": [.5, .5, .5, .5], "qwen_label": values.get("text_prompt", "visual"), "mask": _encode_binary_mask_impl(mask), "score": .8}], "warnings": []}
        def close(self):
            created.append("closed")
    monkeypatch.setattr(assistance_adapters, "Sam3PromptRuntime", PromptsDouble)
    monkeypatch.setattr(adapters, "_build_sam", lambda *_: (PredictorDouble(), model))
    spec = registry.user_spec("alice", "sam3", media_root=media, device="cpu")
    adapter = adapters.UserToolAdapter(config=spec.config, device="cpu")
    result = adapter.execute("sam3_text", {**payload, "text_prompt": "car"}, context.as_dict())
    assert result["image_token"] and result["detections"][0]["qwen_label"] == "car"
    crops = []
    class Classifier:
        def predict(self, image, box, background_guard):
            crops.append(list(box))
            return {"label": "car", "proba": .9}
        def close(self):
            pass
    adapter.classifier = Classifier()
    adapter.config = {**dict(adapter.config), "classifier_bindings": {"car": {"id": "stable-car", "name": "vehicle", "position": 1}}}
    automatic = adapter.execute("sam3_text_auto", {**payload, "text_prompt": "car"}, context.as_dict())
    assert crops == [[15, 10, 30, 20]]  # Mask geometry, not loose grounding box.
    assert automatic["detections"][0]["prediction"] == "vehicle"
    assert len(created) == 1
    adapter.close()
    assert created[-1] == "closed"


def test_qwen_user_adapter_preserves_real_inference_pipeline_and_private_sam(assets, monkeypatch):
    registry, media, _, payload, context = assets
    from test_hosted_qwen_adapters import runtime
    from services.hosted import qwen_adapters
    worker, calls, _ = runtime(monkeypatch, '[{"point_2d":[500,500],"label":"car"}]', config_overrides={"actor_id": "alice"})
    class ExistingRuntime(qwen_adapters.QwenRuntime):
        def __new__(cls, config, device, *, sam_predict):
            worker.sam_predict = sam_predict
            return worker
    monkeypatch.setattr(qwen_adapters, "QwenRuntime", ExistingRuntime)
    predictor = PredictorDouble()
    monkeypatch.setattr(adapters, "_build_sam", lambda *_: (predictor, object()))
    spec = registry.user_spec("alice", "sam", media_root=media, device="cpu")
    adapter = adapters.UserToolAdapter(config={**dict(spec.config), "qwen": {**worker.config, "device": "cpu"}}, device="cpu")
    result = adapter.execute("qwen_infer", {**payload, "source_media_hash": "a" * 64, "item_list": "car", "prompt_type": "point"}, context.as_dict())
    assert calls and result["boxes"][0]["qwen_label"] == "car"
    assert predictor.loads == 1 and adapter.slots["current"]["key"] is None
    assert result["boxes"][0]["bbox"] == pytest.approx([22.5 / 64, 15 / 48, 15 / 64, 10 / 48])
    assert "media_path" not in str(result) and str(media) not in str(result)
    adapter.close()
    assert worker.closed


def test_caption_case_preserves_real_preview_generation_and_instruction_qa(assets, monkeypatch):
    from test_hosted_qwen_adapters import runtime
    from services.hosted.caption_case import execute_caption_case
    import json
    def respond(calls):
        text = str(calls[-1]["messages"])
        if "grounded image question/answer training rows" in text:
            return json.dumps({"qa_pairs": [{"question": "What color is the parked car?", "answer": "The parked car is red."}]})
        return "A red car is parked beside a low wall on a paved road."
    worker, calls, _ = runtime(monkeypatch, respond, config_overrides={"actor_id": "alice"})
    _, _, _, media, context = assets
    payload = {**media, "caption_request": {"restrict_to_labels": False, "include_counts": False, "max_new_tokens": 128},
        "settings": {"instruction_dataset": True, "subcaptions_per_image": 1},
        "case": {"name": "fixture", "class_counts": {"car": 1}, "label_count": 1,
            "label_hints": [{"label": "car", "bbox": [1, 2, 20, 30], "source_id": "car-1"}]}}
    with Image.open(media["media_path"]) as image:
        output = execute_caption_case(worker, image, payload, context.as_dict())
    assert output["status"] == "ok"
    assert output["preview"]["full_text"] and output["response"]["caption"]
    assert output["response"]["generated_qa_pair_count"] == 1
    assert output["response"]["generated_qa_pairs"][0]["answer"] == "The parked car is red."
    assert len(calls) >= 2 and worker.image is None and worker.operation is None
    assert any("question/answer" in str(call["messages"]) for call in calls)
    worker.close()


class TensorDouble:
    def __init__(self, values):
        self.values = np.asarray(values)
    def cpu(self):
        return self
    def numpy(self):
        return self.values


class DetectorDouble:
    def __init__(self):
        self.calls = []
    def predict(self, image, **options):
        self.calls.append((image.size, options))
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=TensorDouble([[1, 2, 9, 12], [1, 2, 9, 12]]),
            conf=TensorDouble([0.9, 0.8]), cls=TensorDouble([0, 0])))]


def test_detector_uses_existing_extraction_offsets_clamps_and_tile_merge(assets, monkeypatch):
    registry, media, _, payload, context = assets
    model = DetectorDouble()
    monkeypatch.setattr(adapters, "_build_detector", lambda *_: model)
    spec = registry.dataset_spec("dataset", "detector", media_root=media, device="cpu")
    adapter = adapters.DatasetDetectorAdapter(config={**dict(spec.config), "class_bindings": {
        "car": {"id": "car-stable", "name": "vehicle", "position": 1},
        "bus": {"id": "bus-stable", "name": "bus", "position": 0}}, "dataset_labelmap": ["bus", "vehicle"]}, device="cpu")
    region = adapter.execute("detect_region", {**payload, "bbox_left": 10, "bbox_top": 5,
        "bbox_width": 30, "bbox_height": 25, "conf": 2}, context.as_dict())
    assert region["detections"][0]["bbox"] == [11, 7, 8, 10]
    assert region["detections"][0]["class_name"] == "vehicle"
    assert region["detections"][0]["class_id"] == 1
    assert region["detections"][0]["stable_class_id"] == "car-stable"
    assert region["labelmap"] == ["bus", "vehicle"]
    assert region["warnings"] == ["conf_clamped"]
    assert model.calls[0] == ((30, 25), {"conf": 1.0, "iou": 0.45, "max_det": 300, "verbose": False, "device": "cpu"})
    # The actual shared NMS/extraction is exercised. Tiling replaces only the
    # optional SAHI dependency, which is not installed in this small test env.
    import utils.image
    monkeypatch.setattr(utils.image, "_slice_image_sahi", lambda *_: ([np.zeros((32, 32, 3), dtype=np.uint8)] * 2, [(0, 0), (0, 0)]))
    windowed = adapter.execute("detect_windowed", payload, context.as_dict())
    assert len(windowed["detections"]) == 1 and len(model.calls) == 3
    with pytest.raises(ValueError, match="hosted_runtime_owner_mismatch"):
        adapter.execute("detect_full", payload, replace(context, dataset_id="other-dataset").as_dict())


def test_rfdetr_requires_exact_native_class_identity(assets, monkeypatch):
    registry, media, checkpoint, payload, context = assets
    definition = {"kind": "rfdetr", "checkpoint": str(checkpoint), "variant": "rfdetr-base", "labels": ["car", "bus"]}
    with pytest.raises(ValueError, match="native_class_ids_required"):
        registry.register("rf", definition)
    registry.register("rf", {**definition, "class_ids": [1, 7]})
    class Detector:
        native = 7
        def predict(self, *_args, **_kwargs):
            return SimpleNamespace(xyxy=np.asarray([[1, 2, 9, 12]]), confidence=np.asarray([.9]), class_id=np.asarray([self.native]))
    model = Detector()
    monkeypatch.setattr(adapters, "_build_detector", lambda *_: model)
    spec = registry.dataset_spec("dataset", "rf", media_root=media, device="cpu")
    adapter = adapters.DatasetDetectorAdapter(config=spec.config, device="cpu")
    result = adapter.execute("detect_full", payload, context.as_dict())
    assert result["detections"][0]["class_name"] == "bus"
    assert result["warnings"] is None
    model.native = 2
    with pytest.raises(ValueError, match="hosted_detector_label_unmapped"):
        adapter.execute("detect_full", payload, context.as_dict())
