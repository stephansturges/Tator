"""Real trained artifacts/recipe math with explicit CPU model doubles.

These tests qualify the adapter mechanism, not CLIP/DINO/C-RADIO weights or GPU
accuracy. Only backbone construction is substituted in composed predictions.
"""
from hashlib import sha256
from types import SimpleNamespace

import joblib
import numpy as np
from PIL import Image
import pytest
from sklearn.linear_model import LogisticRegression
import torch

from services.hosted import assistance_adapters as adapters


def artifact(tmp_path, metadata=None, head=None):
    if head is None:
        head = LogisticRegression(C=10).fit(
            [[1, 0], [.8, .1], [0, 1], [.1, .8], [0, 0], [.05, .05]],
            ["car", "car", "person", "person", "__bg_scene", "__bg_scene"])
    recipe = {"encoder_type": "clip", "encoder_model": "fixture-encoder",
              "mlp_normalize_embeddings": False, "embedding_crop_mode": "tight",
              "preprocess_mode": "native", "embedding_view_mode": "single",
              "embedding_aggregation": "pooled", **(metadata or {})}
    checkpoint, meta = tmp_path / "content-head", tmp_path / "content-meta"
    joblib.dump(head, checkpoint)
    joblib.dump(recipe, meta)
    labels = list(head["classes"] if isinstance(head, dict) else head.classes_)
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
            "metadata_path": str(meta), "metadata_sha256": sha256(meta.read_bytes()).hexdigest(), "labels": labels}, head


class ClipDouble:
    def __init__(self):
        self.calls = []

    def encode_image(self, values):
        self.calls.append(values.clone())
        return values


def color_processor(image):
    rgb = np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) / 255
    return torch.tensor(rgb[:2])


def test_real_linear_artifact_private_models_and_exact_crop(tmp_path, monkeypatch):
    config, trained = artifact(tmp_path)
    built = []
    def build(head, device):
        assert head["encoder_model"] == "fixture-encoder" and device == "cpu"
        model = ClipDouble()
        built.append(model)
        return model, color_processor
    monkeypatch.setattr(adapters, "_build_encoder", build)
    first, second = adapters.ClassifierRuntime(config, "cpu"), adapters.ClassifierRuntime(config, "cpu")
    image = Image.new("RGB", (20, 10), "green")
    image.paste((255, 0, 0), (0, 0, 10, 10))
    result = first.predict(image, [0, 0, 10, 10], background_guard=False)
    expected = trained.predict_proba([[1, 0]])[0]
    index = int(expected.argmax())
    assert result["label"] == trained.classes_[index]
    assert result["proba"] == pytest.approx(expected[index], rel=1e-6)
    assert built[0] is not built[1] and len(built[0].calls) == 1 and not built[1].calls
    np.testing.assert_allclose(built[0].calls[0], [[1, 0]])
    first.close()
    with pytest.raises(ValueError, match="closed"):
        first.predict(image, [0, 0, 10, 10])
    assert second.predict(image, [0, 0, 10, 10])["label"] == result["label"]


def test_saved_multiview_residualizer_standardization_and_calibration(tmp_path, monkeypatch):
    head = {"classifier_type": "mlp", "embedding_dim": 4, "classes": ["first", "second"],
            "layers": [{"weight": np.array([[1, 0, 0, 0], [-1, 0, 0, 0]]), "bias": np.zeros(2), "activation": "linear"}]}
    metadata = {"embedding_view_mode": "tight_context", "mlp_normalize_embeddings": True,
                "embedding_adjustment_transform": {"keep_mask": [True, False, False, False], "mean": [0], "std": [1],
                    "beta": [[.6, 0, 0, 0], [0, 0, 0, 0]]},
                "embedding_center_values": [.1, 0, 0, 0], "embedding_std_values": [2, 1, 1, 1],
                "calibration_temperature": 2, "logit_adjustment_inference": True, "logit_adjustment": [.2, -.1]}
    config, _ = artifact(tmp_path, metadata, head)
    model = ClipDouble()
    monkeypatch.setattr(adapters, "_build_encoder", lambda head, device: (model, lambda image: torch.tensor([3., 4.])))
    runtime = adapters.ClassifierRuntime(config, "cpu")
    result = runtime.predict(Image.new("RGB", (80, 60), "red"), [20, 10, 40, 30])
    assert model.calls[0].shape == (2, 2)  # Both saved crop views reach the real encoder path.
    composed = np.array([.6, .8, .6, .8]) / np.sqrt(2)
    composed -= [.6, 0, 0, 0]
    composed /= np.linalg.norm(composed)
    first = (composed[0] - .1) / 2
    expected = torch.softmax(torch.tensor([first + .2, -first - .1]) / 2, dim=0).numpy()
    assert first < 0 and expected[0] > expected[1]  # Saved calibration reverses the raw-logit winner.
    assert result["label"] == "first"
    assert result["proba"] == pytest.approx(expected[0], rel=1e-6)


@pytest.mark.parametrize("pooling,expected", [
    ("cls", [3, 4]), ("pooler", [9, 9]), ("patch_mean", [1, 1]),
    ("cls_patch_concat", [.6, .8, 1/np.sqrt(2), 1/np.sqrt(2)]),
])
def test_dinov3_saved_pooling_excludes_register_tokens(pooling, expected):
    class DinoDouble:
        config = SimpleNamespace(num_register_tokens=2)
        def __call__(self, pixel_values):
            tokens = torch.tensor([[[3., 4.], [1000., 1000.], [2000., 2000.], [0., 2.], [2., 0.]]])
            return SimpleNamespace(last_hidden_state=tokens, pooler_output=torch.tensor([[9., 9.]]))
    values = adapters._encode_views(DinoDouble(), lambda **kwargs: {"pixel_values": torch.zeros(1)},
        [Image.new("RGB", (8, 8))], {"encoder_type": "dinov3", "dinov3_pooling": pooling}, "cpu")
    np.testing.assert_allclose(values, [expected], rtol=1e-6)


def test_real_binary_and_ovr_head_probability_contract():
    features = np.array([[.3, -.5], [2, 1]], dtype=np.float32)
    binary = {"classes": ["a", "b"], "coef": [[1, 2]], "intercept": [.5], "proba_mode": "binary",
              "temperature": 2, "logit_adjustment_inference": True, "logit_adjustment": [.1, .4]}
    logits = torch.tensor((features @ np.array([[1], [2]]) + .5 + .3) / 2)
    probability = torch.sigmoid(logits).numpy()
    np.testing.assert_allclose(adapters.predict_probabilities(features, binary), np.concatenate([1-probability, probability], axis=1), rtol=1e-6)
    ovr = {"classes": ["a", "b", "c"], "coef": [[1, 0], [0, 1], [-1, -1]], "intercept": [0, .1, -.1], "proba_mode": "ovr"}
    expected = torch.sigmoid(torch.tensor(features @ np.array(ovr["coef"]).T + ovr["intercept"]))
    expected /= expected.sum(dim=1, keepdim=True)
    np.testing.assert_allclose(adapters.predict_probabilities(features, ovr), expected.numpy(), rtol=1e-6)


def test_mlp_layer_norm_gelu_and_arcface_match_torch():
    features = np.array([[1, 2], [-1, 3]], dtype=np.float32)
    weight = np.array([[1, -1], [2, 1]], dtype=np.float32)
    first = {"weight": weight, "bias": np.array([.1, -.2]), "activation": "gelu",
             "layer_norm_weight": np.array([.5, 1.5]), "layer_norm_bias": np.array([.1, .2]), "layer_norm_eps": 1e-5}
    output = {"weight": np.array([[1, 0], [0, 2]]), "bias": np.array([100, -100]), "activation": "linear"}
    head = {"classes": ["a", "b"], "classifier_type": "mlp", "layers": [first, output], "arcface": True, "arcface_scale": 4}
    x = torch.tensor(features) @ torch.tensor(weight).T + torch.tensor(first["bias"], dtype=torch.float32)
    x = torch.nn.functional.layer_norm(x, (2,), torch.tensor(first["layer_norm_weight"], dtype=torch.float32), torch.tensor(first["layer_norm_bias"], dtype=torch.float32))
    x = torch.nn.functional.gelu(x)
    expected = torch.softmax(torch.nn.functional.normalize(x, dim=1) @ torch.eye(2) * 4, dim=1)
    np.testing.assert_allclose(adapters.predict_probabilities(features, head), expected.numpy(), rtol=1e-5)


def test_arcface_tiny_vectors_preserve_existing_epsilon():
    features = np.array([[1e-14, 0]], dtype=np.float32)
    head = {"classes": ["a", "b"], "classifier_type": "mlp", "arcface": True, "arcface_scale": 4,
            "layers": [{"weight": np.eye(2), "bias": np.zeros(2)}]}
    expected = torch.softmax(torch.nn.functional.normalize(torch.tensor(features), dim=1, eps=1e-12) * 4, dim=1)
    np.testing.assert_allclose(adapters.predict_probabilities(features, head), expected.numpy(), rtol=1e-6)


@pytest.mark.parametrize("failure", ["head_hash", "metadata_hash", "missing_metadata", "unsupported_aggregation", "labels"])
def test_recipe_and_artifact_failures_do_not_load_a_substitute(tmp_path, monkeypatch, failure):
    config, _ = artifact(tmp_path, {"embedding_aggregation": "local_salad"} if failure == "unsupported_aggregation" else None)
    if failure == "head_hash": config["checkpoint_sha256"] = "0" * 64
    if failure == "metadata_hash": config["metadata_sha256"] = "0" * 64
    if failure == "missing_metadata": config.pop("metadata_path")
    if failure == "labels": config["labels"] = ["car"]
    def forbidden(*args):
        pytest.fail("a broken recipe must not load a backbone")
    monkeypatch.setattr(adapters, "_build_encoder", forbidden)
    with pytest.raises((ValueError, KeyError)):
        adapters.ClassifierRuntime(config, "cpu")


def test_background_guard_retains_existing_first_pane_selection(tmp_path, monkeypatch):
    config, _ = artifact(tmp_path)
    config["labels"] = [label for label in config["labels"] if not label.startswith("__bg_")]
    monkeypatch.setattr(adapters, "_build_encoder", lambda head, device: (ClipDouble(), lambda image: torch.tensor([0., 0.])))
    runtime = adapters.ClassifierRuntime(config, "cpu")
    image = Image.new("RGB", (10, 10))
    ordinary = runtime.predict(image, [0, 0, 10, 10], background_guard=False)
    guarded = runtime.predict(image, [0, 0, 10, 10], background_guard=True)
    assert ordinary["label"] == "__bg_scene"
    assert guarded["label"] in {"car", "person"} and guarded["proba"] < ordinary["proba"]


class Sam3Double:
    def __init__(self, output):
        self.output = output
        self.states, self.prompts, self.boxes = [], [], []

    def set_confidence_threshold(self, threshold):
        self.threshold = threshold

    def set_image(self, image):
        state = {"ordinal": len(self.states), "size": image.size}
        self.states.append(state)
        return state

    def set_text_prompt(self, state, prompt):
        self.prompts.append((state["ordinal"], prompt))
        return {**state, **self.output(state)}

    def add_geometric_prompt(self, box, label, state):
        self.boxes.append((state["ordinal"], box, label))
        return self.output(state)


def test_sam3_text_uses_shared_network_fresh_states_and_aligned_masks(monkeypatch):
    def output(state):
        masks = np.zeros((2, 1, 10, 10), dtype=np.float32)
        masks[0, 0, 1:4, 1:4] = .8
        masks[1, 0, 5:9, 5:9] = .9
        return {"boxes": torch.tensor([[1, 1, 4, 4], [5, 5, 9, 9]]), "scores": torch.tensor([.6, .9]), "masks": masks}
    processor, network = Sam3Double(output), object()
    def build(value, device):
        assert value is network and device == "cpu"
        return processor
    monkeypatch.setattr(adapters, "_sam3_processor", build)
    runtime = adapters.Sam3PromptRuntime(network, "cpu")
    image = Image.new("RGB", (10, 10))
    first = runtime.predict(image, {"text_prompt": "person", "max_results": 1})
    second = runtime.predict(image, {"text_prompt": "vehicle", "threshold": .7})
    assert processor.prompts == [(0, "person"), (1, "vehicle")]
    assert first["detections"][0]["bbox"] == [.7, .7, .4, .4]
    assert first["detections"][0]["score"] == pytest.approx(.9)
    assert first["detections"][0]["mask"] == first["masks"][0]
    assert second["detections"][0]["qwen_label"] == "vehicle"
    runtime.close()
    with pytest.raises(ValueError, match="closed"):
        runtime.predict(image, {"text_prompt": "person"})


def test_sam3_visual_retains_positive_negative_boxes_and_drops_seeds(monkeypatch):
    processor = Sam3Double(lambda state: {"boxes": [[1, 1, 4, 4], [5, 5, 8, 8], [10, 2, 14, 6]], "scores": [.99, .9, .8]})
    monkeypatch.setattr(adapters, "_sam3_processor", lambda *args: processor)
    runtime = adapters.Sam3PromptRuntime(object(), "cpu")
    result = runtime.predict(Image.new("RGB", (20, 10)), {"text_prompt": "car", "bboxes": [[1, 1, 3, 3], [5, 5, 3, 3]], "bbox_labels": [True, False]}, visual=True)
    assert processor.prompts == [(0, "car")]
    assert processor.boxes == [(0, [.125, .25, .15, .3], True), (0, [.325, .65, .15, .3], False)]
    assert len(result["detections"]) == 1 and result["detections"][0]["bbox"] == [.6, .4, .2, .4]
    assert result["masks"] is None


def test_sam3_windowed_offsets_fusion_limit_and_masks(monkeypatch):
    from utils.coco import _decode_binary_mask_impl
    def output(state):
        # Two overlapping windows observe the same global object at x50..60;
        # the later view has the higher score and must win fusion.
        x = 50 if state["ordinal"] == 0 else 18
        mask = np.zeros((1, 64, 64), dtype=np.uint8)
        mask[0, 10:20, x:x+10] = 1
        return {"boxes": [[x, 10, x+10, 20]], "scores": [.6 if state["ordinal"] == 0 else .9], "masks": mask}
    processor = Sam3Double(output)
    monkeypatch.setattr(adapters, "_sam3_processor", lambda *args: processor)
    runtime = adapters.Sam3PromptRuntime(object(), "cpu")
    result = runtime.predict(Image.new("RGB", (96, 64)), {"text_prompt": "car", "windowed": True,
        "window_size": 64, "window_overlap": .5, "merge_iou": .5, "max_results": 1})
    assert len(processor.states) == 2 and len(result["detections"]) == 1
    assert result["detections"][0]["score"] == pytest.approx(.9)
    assert result["detections"][0]["bbox"] == pytest.approx([55/96, 15/64, 10/96, 10/64])
    mask = _decode_binary_mask_impl(result["masks"][0])
    assert mask.shape == (64, 96) and int(mask.sum()) == 100 and mask[10:20, 50:60].all()
    assert "windowed_raw_detections:2" in result["warnings"]
    assert "windowed_fused_detections:1" in result["warnings"]


def test_sam3_box_only_checkpoint_uses_real_grounding_output(monkeypatch):
    class BoxOnly(Sam3Double):
        find_stage = object()
        def set_text_prompt(self, **kwargs):
            raise KeyError("pred_masks")
    class Network:
        def _get_dummy_prompt(self): return object()
        def forward_grounding(self, **kwargs):
            return {"pred_boxes": torch.tensor([[[.5, .5, .4, .2]]]), "pred_logits": torch.tensor([[[2.]]])}
    processor = BoxOnly(None)
    processor.model = Network()
    monkeypatch.setattr(adapters, "_sam3_processor", lambda *args: processor)
    result = adapters.Sam3PromptRuntime(processor.model, "cpu").predict(Image.new("RGB", (100, 200)), {"text_prompt": "car"})
    assert result["detections"][0]["bbox"] == pytest.approx([.5, .5, .4, .2])
    assert result["masks"] is None


@pytest.mark.parametrize("payload,visual", [
    ({"text_prompt": ""}, False), ({"text_prompt": "car", "threshold": float("nan")}, False),
    ({"bboxes": [[1, 1, 3, 3]], "bbox_labels": [1]}, True),
    ({"bboxes": [[1, 1, -1, 3]]}, True),
])
def test_sam3_invalid_prompts_never_reach_model(monkeypatch, payload, visual):
    processor = Sam3Double(lambda state: pytest.fail("invalid prompt reached model"))
    monkeypatch.setattr(adapters, "_sam3_processor", lambda *args: processor)
    with pytest.raises(ValueError):
        adapters.Sam3PromptRuntime(object(), "cpu").predict(Image.new("RGB", (10, 10)), payload, visual=visual)
    assert not processor.states
