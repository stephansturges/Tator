import numpy as np

from tools.context_feature_variants import copy_schema_metadata, compute_feature_schema_hash


def test_feature_schema_hash_is_stable_for_identical_inputs():
    first = compute_feature_schema_hash(
        ["a", "b", "c"],
        classifier_classes=["person"],
        labelmap=["person"],
        context_variant_id="base",
        variant_config={"drop_img_probs": False},
    )
    second = compute_feature_schema_hash(
        ["a", "b", "c"],
        classifier_classes=["person"],
        labelmap=["person"],
        context_variant_id="base",
        variant_config={"drop_img_probs": False},
    )
    assert first == second


def test_feature_schema_hash_changes_when_feature_order_changes():
    first = compute_feature_schema_hash(["a", "b", "c"])
    second = compute_feature_schema_hash(["a", "c", "b"])
    assert first != second


def test_feature_schema_hash_changes_when_variant_config_changes():
    first = compute_feature_schema_hash(["a", "b"], variant_config={"mode": "x"})
    second = compute_feature_schema_hash(["a", "b"], variant_config={"mode": "y"})
    assert first != second


def test_copy_schema_metadata_preserves_hash_and_variant_fields():
    payload = {
        "feature_names": np.asarray(["a", "b"], dtype=object),
        "classifier_classes": np.asarray(["person"], dtype=object),
        "labelmap": np.asarray(["person"], dtype=object),
        "feature_schema_hash": np.asarray("abc123"),
        "context_variant_id": np.asarray("imgraw"),
        "parent_feature_npz": np.asarray("/tmp/base.npz"),
        "parent_feature_schema_hash": np.asarray("base123"),
        "variant_config_json": np.asarray('{"drop_img_probs":true}'),
    }
    copied = copy_schema_metadata(payload)
    assert str(copied["feature_schema_hash"].item()) == "abc123"
    assert str(copied["context_variant_id"].item()) == "imgraw"
    assert str(copied["parent_feature_npz"].item()) == "/tmp/base.npz"
    assert str(copied["parent_feature_schema_hash"].item()) == "base123"
