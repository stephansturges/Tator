"""Saved recipe execution with actual retained heads and CPU model doubles.

These checks establish stage composition and native visual-token plumbing, not
SAM3/CLIP weight quality. No legacy application or original datasets are used.
"""
import copy
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from services.hosted import sam3_chain as chain


def package(tmp_path, recipe, *, head=False, crops=False, label="Vehicle", identifier="recipe-a", aggregation="pooled"):
    root = tmp_path / identifier
    root.mkdir()
    if head:
        recipe = {**recipe, "clip_head": {"artifact": "clip_head/head.npz"}}
        folder = root / "clip_head"
        folder.mkdir()
        trained = {"classes": ["Vehicle", "Person", "__bg_scene"], "embedding_dim": 2,
            "coef": [[8, 0], [0, 4], [-8, 0]], "intercept": [-2, -2, 2],
            "encoder_type": "clip", "encoder_model": "fixture-clip", "embedding_crop_mode": "tight",
            "preprocess_mode": "native", "embedding_view_mode": "single", "embedding_aggregation": aggregation,
            "normalize_embeddings": False, "proba_mode": "softmax"}
        np.savez(folder / "head.npz", head=np.array([trained], dtype=object))
        (folder / "meta.json").write_text(json.dumps({"min_prob": .7, "margin": .2}))
    if crops:
        Image.new("RGB", (7, 5), "red").save(root / "positive.png")
        Image.new("RGB", (6, 4), "green").save(root / "negative.png")
    (root / "recipe.json").write_text(json.dumps({"class_name": label, "recipe": recipe}))
    files = {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    return {"kind": "sam3_recipe", "model_id": identifier, "model_path": str(root), "recipe_file": "recipe.json",
        "files": files, "checkpoint_sha256": sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "labels": ["Vehicle", "Person"], "component_models": {"sam3": {"kind": "sam3", "model_id": "sam-fixture"}}}


def runtime(definitions):
    return chain.Sam3ChainRuntime({"actor_id": "alice", "recipes": {item["model_id"]: item for item in definitions},
        "class_bindings": {"Vehicle": {"id": "stable-car", "name": "Vehicle", "position": 4},
                           "Person": {"id": "stable-person", "name": "Person", "position": 7}}}, "cpu")


def detection(box, score=.8):
    return {"bbox": list(box), "score": score, "mask": None}


class EncoderDouble:
    def encode_image(self, value):
        return value


def install_encoder(monkeypatch):
    def processor(image):
        return torch.tensor(np.asarray(image, dtype=np.float32).mean(axis=(0, 1))[:2] / 255)
    monkeypatch.setattr(chain, "_build_encoder", lambda *args: (EncoderDouble(), processor))


def test_saved_head_filters_seeds_then_real_visual_stage_then_final_gate(tmp_path, monkeypatch):
    install_encoder(monkeypatch)
    definition = package(tmp_path, {"schema_version": 2, "steps": [{"prompt": "vehicle", "seed_threshold": .15,
        "max_visual_seeds": 2, "expand_threshold": .27, "clip_seed": {"min_prob": .7, "margin": .2},
        "clip_final": {"min_prob": .7, "margin": .2}}]}, head=True)
    value = runtime([definition])
    image = Image.new("RGB", (100, 20), "green")
    image.paste((255, 0, 0), (0, 0, 60, 20))
    calls = []
    class SamDouble:
        def predict(self, pixels, settings, *, visual=False):
            calls.append((visual, copy.deepcopy(settings)))
            assert pixels is image
            rows = [detection([.1, .5, .1, .5], .7), detection([.8, .5, .1, .5], .99)] if not visual else [
                detection([.4, .5, .1, .5], .6), detection([.9, .5, .1, .5], .95)]
            return {"detections": rows}
    monkeypatch.setattr(value, "_sam", lambda _: SamDouble())
    result = value.execute(image, {"steps": [{"recipe_id": "recipe-a"}]}, {"actor_id": "alice"})
    assert [item[0] for item in calls] == [False, True]
    assert calls[1][1]["threshold"] == .27
    np.testing.assert_allclose(calls[1][1]["bboxes"], [[5., 5., 10., 10.]])
    assert len(result["detections"]) == 2
    assert {item["stable_class_id"] for item in result["detections"]} == {"stable-car"}
    stages = result["hosted_evidence"]["stages"]
    assert [event["stage"] for event in stages] == ["text_seed", "classifier_seed", "visual_expand", "classifier_final", "recipe_step_complete", "cascade_complete"]
    assert [row["accepted"] for row in stages[1]["rows"]] == [False, True]  # SAM score orders the green seed first.
    expected = torch.softmax(torch.tensor([6., -2., -6.]), dim=0).numpy()
    assert stages[1]["rows"][1]["probabilities"]["Vehicle"] == pytest.approx(expected[0], rel=1e-6)
    assert stages[4]["seed_count"] == 1 and stages[4]["expanded_count"] == 2


class GeometryNetworkDouble:
    def __init__(self):
        self.received = []

    def _encode_prompt(self, backbone_out, find_input, geometric_prompt, visual_prompt_embed=None, visual_prompt_mask=None):
        self.received.append((visual_prompt_embed.clone(), visual_prompt_mask.clone()))
        return visual_prompt_embed

    def _get_img_feats(self, backbone, ids):
        return backbone, torch.zeros(1), torch.zeros(1), [(1, 1)]

    def _get_dummy_prompt(self):
        class Prompt:
            def append_boxes(self, boxes, labels):
                assert boxes.tolist() == [[[.5, .5, 1., 1.]]]
                self.positive = labels.item()
        return Prompt()

    def geometry_encoder(self, *, geo_prompt, **kwargs):
        return torch.full((1, 1, 2), 1. if geo_prompt.positive else -1.), torch.zeros((1, 1), dtype=torch.bool)


@pytest.mark.parametrize("fail", [False, True])
def test_retained_positive_negative_crop_tokens_reach_native_seam_and_restore(tmp_path, monkeypatch, fail):
    definition = package(tmp_path, {"mode": "sam3_greedy", "text_prompts": ["vehicle", "truck"],
        "positives": [{"crop_path": "positive.png"}], "negatives": [{"crop_path": "negative.png"}]}, crops=True)
    value, network = runtime([definition]), GeometryNetworkDouble()
    sizes = []
    class Processor:
        model, device, find_stage = network, "cpu", SimpleNamespace(img_ids=[0])
        def set_image(self, pixels):
            sizes.append(pixels.size)
            return {"backbone_out": {}}
    class SamDouble:
        processor = Processor()
        def predict(self, pixels, settings, *, visual=False):
            assert pixels.size == (100, 30)
            if settings["text_prompt"] == "visual":
                network._encode_prompt({}, None, None)
                if fail:
                    raise RuntimeError("model failed")
            return {"detections": [detection([.5, .5, .2, .2])]}
    monkeypatch.setattr(value, "_sam", lambda _: SamDouble())
    if fail:
        with pytest.raises(RuntimeError, match="model failed"):
            value.execute(Image.new("RGB", (100, 30)), {"steps": [{"recipe_id": "recipe-a"}]}, {"actor_id": "alice"})
    else:
        result = value.execute(Image.new("RGB", (100, 30)), {"steps": [{"recipe_id": "recipe-a"}]}, {"actor_id": "alice"})
        bank = next(event for event in result["hosted_evidence"]["stages"] if event["stage"] == "crop_bank_tokens")
        assert bank["token_count"] == 2 and [crop["positive"] for crop in bank["crops"]] == [True, False]
        assert bank["crops"][0]["sha256"] == definition["files"]["positive.png"]
    assert sizes == [(7, 5), (6, 4)]
    assert network.received[0][0][:, 0, 0].tolist() == [1., -1.]
    assert "_encode_prompt" not in network.__dict__  # Also restored on inference failure.


def test_cascade_classifier_ranking_preserves_scores_and_group_opt_out(tmp_path, monkeypatch):
    install_encoder(monkeypatch)
    definitions = [package(tmp_path, {"schema_version": 2, "steps": [{"prompt": "vehicle", "clip_final": {"enabled": False}}]}, head=True),
        package(tmp_path, {"steps": [{"prompt": "person", "threshold": .1}]}, label="Person", identifier="recipe-b")]
    value = runtime(definitions)
    class SamDouble:
        def predict(self, pixels, settings, *, visual=False):
            return {"detections": [detection([.5, .5, .5, .5], .4 if settings["text_prompt"] == "vehicle" else .99)]}
    monkeypatch.setattr(value, "_sam", lambda _: SamDouble())
    payload = {"steps": [{"recipe_id": "recipe-a", "dedupe_group": "same"}, {"recipe_id": "recipe-b", "dedupe_group": "same"}],
               "dedupe": {"cross_class_enabled": True, "confidence": "clip_head_prob", "clip_head_recipe_id": "recipe-a"}}
    image = Image.new("RGB", (20, 20), "red")
    result = value.execute(image, payload, {"actor_id": "alice"})
    assert len(result["detections"]) == 1 and result["detections"][0]["score"] == .4
    payload["steps"][1]["participate_cross_class_dedupe"] = False
    assert len(value.execute(image, payload, {"actor_id": "alice"})["detections"]) == 2
    payload["steps"][1].update(participate_cross_class_dedupe=True, dedupe_group="other")
    assert len(value.execute(image, payload, {"actor_id": "alice"})["detections"]) == 2


def test_negative_only_crop_bank_uses_each_saved_text_target(tmp_path, monkeypatch):
    definition = package(tmp_path, {"mode": "sam3_greedy", "text_prompts": ["vehicle", "truck"],
        "negatives": [{"crop_path": "negative.png"}], "params": {"negative_strength": 1}}, crops=True)
    value, network, calls = runtime([definition]), GeometryNetworkDouble(), []
    class Processor:
        model, device, find_stage = network, "cpu", SimpleNamespace(img_ids=[0])
        def set_image(self, pixels):
            assert pixels.size == (6, 4)
            return {"backbone_out": {}}
    class SamDouble:
        processor = Processor()
        def predict(self, pixels, settings, *, visual=False):
            calls.append(settings["text_prompt"])
            network._encode_prompt({}, None, None)
            return {"detections": [detection([.5, .5, .2, .2])]}
    monkeypatch.setattr(value, "_sam", lambda _: SamDouble())
    result = value.execute(Image.new("RGB", (100, 30)), {"steps": [{"recipe_id": "recipe-a"}]}, {"actor_id": "alice"})
    assert calls == ["vehicle", "truck"] and len(result["detections"]) == 1
    assert [tokens[:, 0, 0].tolist() for tokens, _ in network.received] == [[-1.], [-1.]]
    assert "_encode_prompt" not in network.__dict__


def test_missing_required_head_identity_and_artifact_errors_are_explicit(tmp_path, monkeypatch):
    definition = package(tmp_path, {"steps": [{"prompt": "vehicle", "threshold": .1}], "clip_head": {"artifact": "clip_head/missing.npz"}})
    value = runtime([definition])
    monkeypatch.setattr(value, "_sam", lambda _: SimpleNamespace(predict=lambda *args, **kwargs: {"detections": []}))
    payload, image = {"steps": [{"recipe_id": "recipe-a"}]}, Image.new("RGB", (20, 20))
    with pytest.raises(ValueError, match="classifier_required"):
        value.execute(image, payload, {"actor_id": "alice"})
    with pytest.raises(ValueError, match="actor_changed"):
        value.execute(image, payload, {"actor_id": "bob"})
    with pytest.raises(ValueError, match="inline_not_allowed"):
        value.execute(image, {"steps": [{"recipe_id": "recipe-a", "recipe": {"steps": []}}]}, {"actor_id": "alice"})
    (tmp_path / "recipe-a" / "recipe.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshot_changed"):
        runtime([definition])


def test_optimizer_penalty_is_provenance_and_saved_native_classifier_pooling_is_preserved(tmp_path, monkeypatch):
    install_encoder(monkeypatch)
    definition = package(tmp_path, {"steps": [{"prompt": "vehicle", "threshold": .1}],
        "params": {"clip_head_background_penalty": 1.7, "negative_strength": .3, "use_negative_exemplars": False}},
        head=True, aggregation="sam3_mask_fusion_v1")
    value = runtime([definition])
    monkeypatch.setattr(value, "_sam", lambda _: SimpleNamespace(predict=lambda *args, **kwargs: {"detections": [detection([.5, .5, .2, .2], .77)]}))
    result = value.execute(Image.new("RGB", (20, 20), "red"), {"steps": [{"recipe_id": "recipe-a"}]}, {"actor_id": "alice"})
    assert result["detections"][0]["score"] == .77
    evidence = result["hosted_evidence"]["recipe_settings"]["recipe-a"]
    assert evidence["clip_head_background_penalty"] == 1.7 and evidence["background_penalty_scope"] == "recipe_optimizer"
    assert evidence["classifier_aggregation"] == "sam3_mask_fusion_v1"
