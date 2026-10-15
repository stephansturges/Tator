"""Real prepass/features/calibrated scorers; only native model calls are doubles.

These fixtures do not qualify production model weights or CUDA execution.
"""
from hashlib import sha256
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from services.hosted import adapters, assistance_adapters
from services.hosted.edr_adapters import DatasetEdrAdapter
from services.hosted.edr_features import _build_feature_names, SOURCE_RUNS
from tools.context_feature_variants import CANONICAL_BASE_VARIANT, compute_feature_schema_hash
from tools.score_ensemble_candidates import _MLP
from test_hosted_assistance_adapters import artifact, ClipDouble, color_processor, Sam3Double


def fixture(tmp_path, monkeypatch, *, recipe=None, ensemble=True, metadata=None):
    bundle, media, classifier_root = tmp_path / "bundle", tmp_path / "media", tmp_path / "classifier"
    for directory in (bundle, media, classifier_root):
        directory.mkdir()
    classifier, _ = artifact(classifier_root)
    built = []
    def encoder(*args):
        model = ClipDouble(); built.append(model)
        return model, color_processor
    monkeypatch.setattr(assistance_adapters, "_build_encoder", encoder)
    native_calls = []
    class YoloDouble:
        def predict(self, image, **options):
            native_calls.append((image.size, options))
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[2., 2., 18., 18.], [34., 2., 50., 18.]]),
                conf=torch.tensor([.8, .9]), cls=torch.tensor([0, 1])))]
    monkeypatch.setattr(adapters, "_build_detector", lambda *args: YoloDouble())
    image = Image.new("RGB", (64, 32), "red")
    image.paste((0, 255, 0), (32, 0, 64, 32)); image.save(media / "image.png")
    recipe_values = {"enable_yolo": True, "enable_rfdetr": False, "enable_sam3_text": False,
        "enable_sam3_similarity": False, "prepass_keep_all": True, "prepass_finalize": False,
        "prepass_caption": False, "sam3_text_synonym_budget": 0, **(recipe or {})}
    (bundle / "recipe.json").write_text(json.dumps(recipe_values))
    labels = ["car", "person"]
    definition = {"kind": "edr", "model_id": "recipe", "model_path": str(bundle),
        "recipe_file": "recipe.json", "labels": labels,
        "component_models": {"yolo": {"kind": "yolo", "labels": labels}, "classifier": classifier}}
    if ensemble:
        options = {"embed_proj_dim": 2, "embed_proj_seed": 42, "image_embed_proj_dim": 0,
            "image_embed_proj_seed": 4242, "embed_l2_normalize": True, "term_hash_dim": 4,
            "support_iou": .5, "context_radius": .075}
        classes = list(classifier["labels"])
        names = _build_feature_names(classes, 2, 0, labels, 4,
            ["yolo", "rfdetr", "sam3_text", "sam3_similarity"], SOURCE_RUNS)
        schema = compute_feature_schema_hash(names, classifier_classes=classes, labelmap=labels,
            context_variant_id=CANONICAL_BASE_VARIANT, variant_config=options)
        contract = {**options, "labelmap": labels, "feature_schema_hash": schema}
        (bundle / "features.json").write_text(json.dumps(contract))
        model = _MLP(len(names), [], 0)
        with torch.no_grad():
            model.net[0].weight.zero_(); model.net[0].bias.fill_(2.)
        torch.save({"input_dim": len(names), "hidden": [], "dropout": 0,
                    "state_dict": model.state_dict()}, bundle / "model.pt")
        (bundle / "meta.json").write_text(json.dumps({"calibrated_temperature": 2,
            "calibrated_threshold": .5, "calibrated_thresholds_relaxed_smoothed": {"car": .7, "person": .8},
            **(metadata or {})}))
        definition.update(ensemble_model_file="model.pt", ensemble_meta_file="meta.json", feature_contract_file="features.json")
    definition["files"] = {path.name: sha256(path.read_bytes()).hexdigest() for path in bundle.iterdir()}
    config = {"dataset_id": "dataset", "media_root": str(media), "model": definition,
              "class_bindings": {"car": {"id": "stable-car", "name": "Automobile", "position": 1},
                                 "person": {"id": "stable-person", "name": "Person", "position": 0}}}
    payload = {"recipe_id": "recipe", "media_path": str(media / "image.png"),
               "media_sha256": sha256((media / "image.png").read_bytes()).hexdigest(),
               "annotation_objects": [{"class_id": "stable-car", "bbox": [2, 2, 16, 16]}]}
    return config, payload, native_calls, built


def test_actual_prepass_features_calibrated_mlp_and_stable_classes(tmp_path, monkeypatch):
    config, payload, native_calls, built = fixture(tmp_path, monkeypatch)
    runtime = DatasetEdrAdapter(config=config, device="cpu")
    result = runtime.execute("edr_prepass", payload, {"dataset_id": "dataset"})
    assert len(native_calls) == 2  # Real full-image and SAHI calls, followed by native merge.
    assert len(result["detections"]) == 1
    detection = result["detections"][0]
    assert detection["label"] == "Automobile" and detection["stable_class_id"] == "stable-car"
    scores = next(event for event in result["hosted_evidence"]["events"] if event["type"] == "ensemble_scoring")
    assert scores["input_count"] == 2 and scores["accepted_count"] == 1
    assert [row["ensemble_prob"] for row in scores["scores"]] == pytest.approx([1/(1+np.exp(-1))]*2)
    assert {row["label"]: row["ensemble_threshold"] for row in scores["scores"]} == {"car": .7, "person": .8}
    assert built and sum(len(model.calls) for model in built) >= 2
    provenance = json.dumps(result["hosted_evidence"]["provenance"])
    assert "yolo_full" in provenance and "yolo_sahi" in provenance
    assert runtime.image is runtime.payload is runtime.context is None
    runtime.close()
    with pytest.raises(ValueError, match="context_invalid"):
        runtime.execute("edr_prepass", payload, {"dataset_id": "dataset"})


@pytest.mark.parametrize("failure", ["identity", "schema", "labels", "qwen", "policy_path"])
def test_recipe_failures_do_not_return_raw_unscored_candidates(tmp_path, monkeypatch, failure):
    config, payload, native_calls, _ = fixture(tmp_path, monkeypatch,
        recipe={"prepass_caption": failure == "qwen"},
        metadata={"selected_policy_layer": {"variant": "bad", "model_path": "../outside.pt", "meta_path": "meta.json"}}
        if failure == "policy_path" else None)
    definition = config["model"]
    if failure == "identity": definition["files"]["recipe.json"] = "0" * 64
    if failure == "labels": definition["component_models"]["yolo"]["labels"] = ["person", "car"]
    if failure == "schema":
        from pathlib import Path
        path = Path(definition["model_path"]) / "features.json"
        data = json.loads(path.read_text()); data["feature_schema_hash"] = "wrong"
        path.write_text(json.dumps(data)); definition["files"][path.name] = sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime = DatasetEdrAdapter(config=config, device="cpu")
        runtime.execute("edr_prepass", payload, {"dataset_id": "dataset"})
    if failure != "schema": assert not native_calls


def test_detector_failure_cannot_silently_score_partial_recipe(tmp_path, monkeypatch):
    config, payload, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(adapters, "_build_detector", lambda *args: (_ for _ in ()).throw(RuntimeError("model failed")))
    runtime = DatasetEdrAdapter(config=config, device="cpu")
    with pytest.raises(ValueError, match="required_stage_failed"):
        runtime.execute("edr_prepass", payload, {"dataset_id": "dataset"})
    assert runtime.image is runtime.payload is runtime.context is None


def test_actual_sam3_text_windows_and_visual_expansion(tmp_path, monkeypatch):
    config, payload, _, _ = fixture(tmp_path, monkeypatch, ensemble=False, recipe={
        "enable_sam3_text": True, "enable_sam3_similarity": True,
        "sam3_text_window_extension": True, "similarity_window_extension": True,
        "sam3_text_window_mode": "sahi", "similarity_window_mode": "sahi",
        "sam3_text_window_size": 32, "similarity_window_size": 32})
    config["model"]["component_models"]["sam3"] = {"kind": "sam3"}
    class SamDouble(Sam3Double):
        def add_geometric_prompt(self, box, label, state):
            self.boxes.append((state["ordinal"], box, label))
            return {"boxes": [[15, 21, 23, 29]], "scores": [.9]}
    processor = SamDouble(lambda state: {"boxes": [[2, 20, 10, 28]], "scores": [.95]})
    monkeypatch.setattr(adapters, "_build_sam", lambda *args: (object(), object()))
    monkeypatch.setattr(assistance_adapters, "_sam3_processor", lambda *args: processor)
    runtime = DatasetEdrAdapter(config=config, device="cpu")
    result = runtime.execute("edr_prepass", payload, {"dataset_id": "dataset"})
    assert processor.prompts and processor.boxes
    assert {state["size"] for state in processor.states} >= {(64, 32), (32, 32)}
    provenance = json.dumps(result["hosted_evidence"]["provenance"])
    assert "sam3_text" in provenance and "sam3_similarity" in provenance
    assert result["detections"]
    assert any(box[0] < 0 or box[0] + box[2]/2 > 1 for _, box, _ in processor.boxes), "partial exemplars retain full geometry"


def test_optional_qwen_caption_and_synonyms_use_exact_private_pipeline(tmp_path, monkeypatch):
    from test_hosted_qwen_adapters import runtime as qwen_fixture
    config, payload, _, _ = fixture(tmp_path, monkeypatch, ensemble=False, recipe={
        "prepass_caption": True, "enable_sam3_text": True, "sam3_text_synonym_budget": 1,
        "sam3_text_window_extension": False, "model_id": "Saved-Qwen-Instruct"})
    config["model"]["component_models"]["sam3"] = {"kind": "sam3"}
    def response(calls):
        prompt = str(calls[-1]["messages"])
        if "segmentation model" in prompt:
            return '{"car":["automobile"],"person":["pedestrian"]}'
        return "A red car is parked beside a low wall on a paved road."
    worker, calls, models = qwen_fixture(monkeypatch, response, config_overrides={"actor_id": "dataset:dataset",
        "models": {name: {"model_path": "/double", "files": {}}
                   for name in ("Qwen3-VL-Instruct", "Saved-Qwen-Instruct")}})
    config["qwen"] = worker.config
    processor = Sam3Double(lambda state: {"boxes": [[2, 20, 10, 28]], "scores": [.95]})
    monkeypatch.setattr(adapters, "_build_sam", lambda *args: (object(), object()))
    monkeypatch.setattr(assistance_adapters, "_sam3_processor", lambda *args: processor)
    runtime = DatasetEdrAdapter(config=config, device="cpu")
    runtime.qwen = worker
    result = runtime.execute("edr_prepass", payload, {"dataset_id": "dataset"})
    assert any(prompt == "automobile" for _, prompt in processor.prompts)
    assert any(call["images"] for call in calls) and any(not call["images"] for call in calls)
    events = result["hosted_evidence"]["events"]
    assert any(row["type"] == "synonym_model_io" for row in events)
    assert any(row["type"] == "caption_model_io" for row in events)
    assert any(row["type"] == "prepass_caption_window" for row in events)
    caption_io = next(row["model_io"] for row in events if row["type"] == "caption_model_io")
    assert next(row["model_id"] for row in caption_io if row.get("source") == "qwen_inference") == "Saved-Qwen-Instruct"
    assert models[0].model_id == "Saved-Qwen-Instruct"
    assert result["caption"] is None  # Existing recipe endpoint never applies a caption.
    assert result["hosted_evidence"]["recipe_caption"]["caption"]
    assert result["hosted_evidence"]["recipe_caption"]["windows"]
    assert worker.window_hook.get() is None and worker.image is None


def test_actual_final_classifier_gate_preserves_foreground_recipe(tmp_path, monkeypatch):
    config, payload, _, _ = fixture(tmp_path, monkeypatch, ensemble=False,
        recipe={"prepass_keep_all": False, "prepass_finalize": True})
    runtime = DatasetEdrAdapter(config=config, device="cpu")
    result = runtime.execute("edr_prepass", payload, {"dataset_id": "dataset"})
    assert {row["stable_class_id"] for row in result["detections"]} == {"stable-car", "stable-person"}
    assert any(row["type"] == "finalize" for row in result["hosted_evidence"]["events"])


@pytest.mark.skipif(sys.platform == "darwin" and os.environ.get("OMP_NUM_THREADS") != "1",
                   reason="macOS Torch/XGBoost OpenMP conflict: run with OMP_NUM_THREADS=1")
def test_actual_xgb_selected_policy_uses_declared_bundle_not_cwd(tmp_path, monkeypatch):
    import xgboost as xgb
    from pathlib import Path
    from tools.policy_runtime import build_policy_feature_matrix
    config, payload, _, _ = fixture(tmp_path, monkeypatch)
    definition = config["model"]
    root = Path(definition["model_path"])
    classes = definition["component_models"]["classifier"]["labels"]
    names = _build_feature_names(classes, 2, 0, ["car", "person"], 4,
        ["yolo", "rfdetr", "sam3_text", "sam3_similarity"], SOURCE_RUNS)
    values = np.zeros((2, len(names)), dtype=np.float32)
    base = xgb.train({"objective": "binary:logistic", "base_score": .1},
        xgb.DMatrix(values, label=[0, 1]), num_boost_round=0)
    base.save_model(root / "base.json")
    bundle = build_policy_feature_matrix(values, names,
        [{"label": "car", "source": "yolo"}, {"label": "person", "source": "yolo"}], [.1, .1])
    policy = xgb.train({"objective": "binary:logistic", "base_score": .9},
        xgb.DMatrix(bundle["X"], label=[0, 1]), num_boost_round=0)
    policy.save_model(root / "policy.json")
    (root / "policy-meta.json").write_text(json.dumps({"feature_schema_hash": bundle["feature_schema_hash"],
        "calibrated_threshold": .8, "calibrated_thresholds": {"car": .8, "person": .8}}))
    (root / "meta.json").write_text(json.dumps({"calibrated_threshold": .5,
        "selected_policy_layer": {"variant": "xgb", "model_path": "policy.json", "meta_path": "policy-meta.json"}}))
    # Legacy relative resolution checks cwd first. Hosted must stay in bundle.
    base.save_model(tmp_path / "policy.json")
    monkeypatch.chdir(tmp_path)
    definition["ensemble_model_file"] = "base.json"
    definition["files"] = {path.name: sha256(path.read_bytes()).hexdigest() for path in root.iterdir()}
    result = DatasetEdrAdapter(config=config, device="cpu").execute("edr_prepass", payload, {"dataset_id": "dataset"})
    scores = next(event for event in result["hosted_evidence"]["events"] if event["type"] == "ensemble_scoring")
    assert len(result["detections"]) == 2
    assert [row["ensemble_prob_raw"] for row in scores["scores"]] == pytest.approx([.1, .1])
    assert [row["ensemble_prob"] for row in scores["scores"]] == pytest.approx([.9, .9])
    assert all(row["ensemble_policy_variant"] == "xgb" for row in scores["scores"])


def test_canonical_recipe_metadata_is_inert_and_caption_is_disabled(tmp_path, monkeypatch):
    config, payload, _, _ = fixture(tmp_path, monkeypatch, recipe={"edr_package_id": "source-package",
        "edr_package_root": "/unreadable/source", "resolved_classifier_id": "/unreadable/source/head.pkl",
        "recipe_fingerprint": "original", "prepass_caption": True, "sam3_text_synonym_budget": 10})
    runtime = DatasetEdrAdapter(config=config, device="cpu")
    result = runtime.execute("edr_prepass", payload, {"dataset_id": "dataset"})
    assert runtime.qwen is None and not runtime.request.prepass_caption
    assert "edr_package_root" in result["hosted_evidence"]["non_runtime_metadata_fields"]
    assert "/unreadable" not in str(result)
