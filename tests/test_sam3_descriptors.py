"""The native bridge must preserve mask decoding and the 768D recipe."""
from types import SimpleNamespace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from services.sam3_descriptors import NativeSam3DescriptorPredictor, mask_feature_vector, native_runtime_identity


class Predictor:
    def __init__(self, *, fail=False, empty=False):
        self.model = SimpleNamespace(sam_mask_decoder=SimpleNamespace(
            conv_s0=torch.nn.Conv2d(256, 32, 1), conv_s1=torch.nn.Conv2d(256, 64, 1)))
        self.raw = [torch.arange(256 * side * side, dtype=torch.float32).reshape(1, 256, side, side) / 100 for side in (8, 4, 2)]
        self.fail, self.empty = fail, empty

    def set_image(self, image):
        self.shape = image.shape[:2]
        high = [self.model.sam_mask_decoder.conv_s0(self.raw[0]), self.model.sam_mask_decoder.conv_s1(self.raw[1])]
        if self.fail:
            raise RuntimeError("fixture_failed")
        self._features = {"high_res_feats": high, "image_embed": self.raw[2] + 7}

    def predict(self, **kwargs):
        mask = np.zeros(self.shape, dtype=bool)
        if not self.empty:
            mask[2:6, 2:6] = True
        return np.asarray([mask, ~mask]), np.asarray([.9, .1]), None

    def reset_predictor(self):
        self._features = None


def test_native_capture_keeps_decoder_projections_and_adds_memory_once():
    predictor = Predictor()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    predictor.set_image(image)
    original = [f.clone() for f in predictor._features['high_res_feats']]
    wrapper = NativeSam3DescriptorPredictor(predictor)
    wrapper.set_image(image)
    assert [f.shape[1] for f in wrapper.descriptor_features] == [256, 256, 256]
    for expected, actual in zip(original, predictor._features['high_res_feats']):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(wrapper.descriptor_features[2], predictor.raw[2] + 7)
    vector, fallback, score = mask_feature_vector(wrapper, [2, 2, 6, 6], (8, 8))
    assert vector.shape == (768,) and not fallback
    assert score == pytest.approx(.9) and np.linalg.norm(vector) == pytest.approx(1)
    assert all(len(layer._forward_pre_hooks) == 0 for layer in (predictor.model.sam_mask_decoder.conv_s0, predictor.model.sam_mask_decoder.conv_s1))


def test_failed_capture_releases_hooks_and_previous_image_features():
    predictor = Predictor()
    wrapper = NativeSam3DescriptorPredictor(predictor)
    wrapper.set_image(np.zeros((8, 8, 3), dtype=np.uint8))
    predictor.fail = True
    with pytest.raises(RuntimeError, match="fixture_failed"):
        wrapper.set_image(np.zeros((8, 8, 3), dtype=np.uint8))
    assert wrapper.descriptor_features is None
    assert not predictor.model.sam_mask_decoder.conv_s0._forward_pre_hooks
    assert not predictor.model.sam_mask_decoder.conv_s1._forward_pre_hooks


def test_native_empty_mask_uses_same_bbox_pooling_and_scale_normalization():
    predictor = Predictor(empty=True)
    wrapper = NativeSam3DescriptorPredictor(predictor)
    wrapper.set_image(np.zeros((8, 8, 3), dtype=np.uint8))
    vector, fallback, _ = mask_feature_vector(wrapper, [0, 0, 8, 8], (8, 8))
    expected = [f.mean((2, 3)).numpy().ravel() for f in wrapper.descriptor_features]
    expected = np.concatenate([f / np.linalg.norm(f) for f in expected])
    expected /= np.linalg.norm(expected)
    assert fallback
    np.testing.assert_allclose(vector, expected, atol=1e-7)
    wrapper.unload()
    assert wrapper.descriptor_features is None and predictor._features is None


@pytest.fixture
def native_source(monkeypatch, tmp_path):
    monkeypatch.setenv("TATOR_SAM3_DESCRIPTOR_BACKEND", "torch")
    for name in tuple(sys.modules):
        if name == "sam3" or name.startswith("sam3."):
            monkeypatch.delitem(sys.modules, name)
    package = tmp_path / "sam3"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("def watched():\n    return 1\n")
    (package / "late.py").write_text("class Later:\n    def value(self):\n        return 2\n")
    spec = importlib.util.spec_from_file_location("sam3", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "sam3", module)
    spec.loader.exec_module(module)
    api = SimpleNamespace(active_sam3_checkpoint=None, SAM3_BPE_PATH=None,
                          _class_analysis_file_sha256_cached=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    yield api, package, module
    sys.modules.pop("sam3.late", None)


def test_native_source_redefinition_rejects_stale_loaded_function(native_source):
    api, package, module = native_source
    native_runtime_identity(api)
    source = package / "__init__.py"
    source.write_text(source.read_text() + "\ndef watched():\n    return 2\n")
    assert module.watched() == 1
    with pytest.raises(RuntimeError, match="sam3_loaded_source_changed"):
        native_runtime_identity(api)


def test_native_source_identity_is_stable_across_lazy_imports(native_source):
    api, package, _ = native_source
    before = native_runtime_identity(api)
    assert importlib.import_module("sam3.late").Later().value() == 2
    assert native_runtime_identity(api) == before
    source = package / "late.py"
    source.write_text(source.read_text() + "\n# New producer source revision.\n")
    assert native_runtime_identity(api) != before


def test_native_vendor_revision_requires_exact_source_manifest(native_source):
    api, package, _ = native_source
    manifest = package.parent / "tator-source.json"
    manifest.write_text(json.dumps({"base_revision": "a" * 40, "python_source_sha256": {
        str(path.relative_to(package.parent)): api._class_analysis_file_sha256_cached(path)
        for path in package.glob("*.py")}}))
    identity = native_runtime_identity(api)
    assert identity["source_revision"] == "a" * 40
    assert any(row["path"] == str(manifest) for row in identity["implementation_files"])
    (package / "late.py").write_text("# Changed after this vendor snapshot was pinned.\n")
    with pytest.raises(RuntimeError, match="sam3_vendor_source_manifest_changed"):
        native_runtime_identity(api)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("caller_autocast", [False, True])
def test_native_constructor_restores_autocast_on_success_and_failure(monkeypatch, fail, caller_autocast):
    from services.sam3_descriptors import build_native_sam3_model, _autocast_nesting
    # Exercise CUDA's thread-local autocast bookkeeping without CUDA tensors.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    moved = []

    def builder(**options):
        assert options["device"] == "cpu"
        assert not torch.is_autocast_enabled("cuda")
        context = torch.autocast("cuda", dtype=torch.bfloat16)
        context.__enter__()  # The pinned native constructor owns this entry.
        if fail:
            raise RuntimeError("checkpoint_fixture_failed")
        return SimpleNamespace(inst_interactive_predictor=SimpleNamespace(
            model=SimpleNamespace(bf16_context=context)),
            to=lambda device: moved.append((device, torch.is_autocast_enabled("cuda"))))

    with torch.autocast("cuda", enabled=caller_autocast, dtype=torch.float16, cache_enabled=False):
        depth = _autocast_nesting(torch)
        if fail:
            with pytest.raises(RuntimeError, match="checkpoint_fixture_failed"):
                build_native_sam3_model(builder, device="cpu")
        else:
            build_native_sam3_model(builder, device="cpu")
            assert moved == [("cpu", False)]
        assert _autocast_nesting(torch) == depth
        assert torch.is_autocast_enabled("cuda") == caller_autocast
        assert torch.get_autocast_dtype("cuda") == torch.float16
        assert not torch.is_autocast_cache_enabled()
