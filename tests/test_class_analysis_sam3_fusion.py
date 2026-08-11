from __future__ import annotations

import numpy as np
from PIL import Image

import localinferenceapi as api
from utils.class_analysis_salad import (
    CLASS_ANALYSIS_SAM3_SALAD_FUSION_SCHEMA,
    class_analysis_salad_effective_train_limit,
    class_analysis_salad_settings,
    class_analysis_salad_training_fingerprint,
    class_analysis_salad_training_indices,
    compose_class_analysis_salad_features,
    fuse_class_analysis_feature_branches,
)


def test_sam3_mask_fusion_groups_images_and_preserves_record_order(
    tmp_path,
    monkeypatch,
):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (40, 30), (10, 20, 30)).save(first)
    Image.new("RGB", (50, 35), (40, 50, 60)).save(second)
    records = [
        {"_image_path": str(first), "bbox_xyxy": [1, 2, 11, 12]},
        {"_image_path": str(first), "bbox_xyxy": [3, 4, 13, 14]},
        {"_image_path": str(second), "bbox_xyxy": [5, 6, 15, 16]},
    ]

    class Predictor:
        def __init__(self):
            self.set_image_calls = 0
            self.unloaded = False

        def set_image(self, _image):
            self.set_image_calls += 1

        def unload(self):
            self.unloaded = True

    predictor = Predictor()
    monkeypatch.setattr(
        api,
        "build_mlx_sam3_predictor",
        lambda _runtime: predictor,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_sam3_mask_feature_vector",
        lambda _predictor, bbox, _size: (
            np.full(768, float(bbox[0]), dtype=np.float32),
            False,
            0.9,
        ),
    )

    matrix, runtime = api._class_analysis_extract_sam3_mask_embeddings(
        records,
        job=api.ClassAnalysisJob(job_id="sam3_fusion_test"),
        output_path=tmp_path / "sam.npy",
    )

    assert matrix.shape == (3, 768)
    assert matrix[:, 0].tolist() == [1.0, 3.0, 5.0]
    assert predictor.set_image_calls == 2
    assert predictor.unloaded is True
    assert runtime["mask_fallback_count"] == 0
    assert api._embedding_normalize_aggregation("sam3_mask_fusion") == (
        api.CLASS_ANALYSIS_SAM3_MASK_FUSION_SCHEMA
    )


def test_salad_fusion_settings_are_parametric_bounded_and_cache_identified():
    balanced = class_analysis_salad_settings({})
    large = class_analysis_salad_settings({"salad_preset": "big"})
    custom = class_analysis_salad_settings(
        {
            "salad_preset": "balanced",
            "salad_num_clusters": 23,
            "salad_cluster_dim": 41,
            "salad_token_dim": 137,
            "salad_hidden_dim": 333,
            "salad_max_train_objects": 999999,
            "salad_token_budget_mb": 64,
        }
    )

    assert balanced.descriptor_dim == 640
    assert large.descriptor_dim == 8448
    assert custom.descriptor_dim == 1080
    assert custom.max_train_objects == 16384
    assert class_analysis_salad_effective_train_limit(custom) < 16384
    assert sum(balanced.fusion_weights) == 1.0

    records = [
        {
            "_image_path": f"image_{index % 50}.jpg",
            "crop_cache_key": f"object_{index}",
            "class_name": "must_not_affect_selection",
        }
        for index in range(10000)
    ]
    limit = class_analysis_salad_effective_train_limit(balanced)
    first = class_analysis_salad_training_indices(records, limit=limit, seed=42)
    second = class_analysis_salad_training_indices(records, limit=limit, seed=42)
    assert first == second
    assert len(first) == limit
    assert len({records[index]["_image_path"] for index in first[:50]}) == 50
    fingerprint = class_analysis_salad_training_fingerprint(
        records, first, balanced, seed=42
    )
    changed = class_analysis_salad_training_fingerprint(
        records, first, large, seed=42
    )
    assert fingerprint != changed
    balanced_head = {
        "encoder_type": "dinov3",
        "encoder_model": api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL,
        "normalize_embeddings": True,
        "embedding_aggregation": CLASS_ANALYSIS_SAM3_SALAD_FUSION_SCHEMA,
        "embedding_salad_training_fingerprint": fingerprint,
        **balanced.to_request_fields(),
    }
    large_head = {
        **balanced_head,
        **large.to_request_fields(),
        "embedding_salad_training_fingerprint": changed,
    }
    assert api._class_analysis_embedding_cache_key("crop", balanced_head) != (
        api._class_analysis_embedding_cache_key("crop", large_head)
    )


def test_salad_feature_composition_keeps_dino_views_and_averages_salad_views():
    raw = np.asarray(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [1, -1, 1, -1],
        ],
        dtype=np.float32,
    )
    composed = compose_class_analysis_salad_features(
        raw,
        [2, 2],
        salad_dimension=2,
    )

    assert composed.shape == (2, 6)
    assert np.allclose(np.linalg.norm(composed[:, :4], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(composed[:, 4:], axis=1), 1.0)


def test_salad_fusion_request_normalization_preserves_scaling_parameters():
    request = api._normalize_class_analysis_request(
        {
            "encoder_type": "dinov3",
            "embedding_aggregation": "sam3_salad_fusion",
            "embedding_view_mode": "single",
            "dinov3_pooling": "patch_mean",
            "salad_preset": "large",
            "salad_weight": 0.2,
            "salad_max_train_objects": 4096,
            "salad_token_budget_mb": 512,
            "salad_epochs": 12,
        }
    )

    assert request["embedding_aggregation"] == CLASS_ANALYSIS_SAM3_SALAD_FUSION_SCHEMA
    assert request["embedding_view_mode"] == "tight_context"
    assert request["dinov3_pooling"] == "pooler"
    assert request["salad_preset"] == "large"
    assert request["salad_weight"] == 0.2
    assert request["salad_max_train_objects"] == 4096
    assert request["salad_token_budget_mb"] == 512
    assert request["salad_epochs"] == 12


def test_salad_fusion_weights_branches_in_cosine_space():
    dino = np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    sam = np.asarray([[0.0, 2.0], [0.0, 1.0]], dtype=np.float32)
    salad = np.asarray([[5.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    fused = fuse_class_analysis_feature_branches(
        [dino, sam, salad],
        [0.675, 0.225, 0.10],
    )

    assert fused.shape == (2, 6)
    assert np.allclose(np.linalg.norm(fused, axis=1), np.ones(2), atol=1e-6)
    assert np.allclose(np.sum(fused[:, :2] ** 2, axis=1), 0.675, atol=1e-6)
    assert np.allclose(np.sum(fused[:, 2:4] ** 2, axis=1), 0.225, atol=1e-6)
    assert np.allclose(np.sum(fused[:, 4:] ** 2, axis=1), 0.10, atol=1e-6)
