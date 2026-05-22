import asyncio
import json
import math
import types

import numpy as np
import pytest
import torch
from PIL import Image

import localinferenceapi as api
from services.classifier import _load_clip_head_from_classifier_impl
from tools import clip_training
from tools import run_class_split_experiments as class_split_experiments
from utils.embedding_recipe import normalize_embedding_aggregation
from utils.cradio_embedding import (
    CRADIO_DEFAULT_MODEL,
    CRadioBackendStatus,
    _unpack_cradio_outputs,
    cradio_backend_status,
    encode_cradio_images,
    normalize_cradio_pooling,
)
from utils import cradio_embedding as cradio_embedding_utils
from utils.local_salad import LocalSALADConfig, LocalSALADHead, symmetric_infonce_loss
from utils.local_salad_mlx import (
    MLXLocalSALADHead,
    encode_local_salad_mlx,
    local_salad_mlx_available,
    make_mlx_local_salad_optimizer,
    mlx_local_salad_state_dict,
    mlx_local_salad_train_step,
)


def _record(point_id: str, class_name: str) -> dict:
    return {
        "point_id": point_id,
        "class_name": class_name,
        "image_relpath": f"{point_id}.jpg",
        "split": "train",
        "bbox_xyxy": [0, 0, 10, 10],
    }


def test_class_analysis_parses_bbox_polygon_and_crop_bounds():
    bbox = api._class_analysis_parse_yolo_geometry(
        "1 0.5 0.5 0.2 0.4",
        image_width=100,
        image_height=100,
    )
    assert bbox["kind"] == "bbox"
    assert bbox["class_id"] == 1
    assert bbox["bbox_xyxy"] == [40.0, 30.0, 60.0, 70.0]

    bbox_with_confidence = api._class_analysis_parse_yolo_geometry(
        "1 0.5 0.5 0.2 0.4 0.99",
        image_width=100,
        image_height=100,
    )
    assert bbox_with_confidence["kind"] == "bbox"
    assert bbox_with_confidence["bbox_xyxy"] == [40.0, 30.0, 60.0, 70.0]


def test_local_salad_head_is_trainable_normalized_and_fixed_width():
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    patches = torch.randn(3, 12, 32, generator=gen)
    global_token = torch.randn(3, 32, generator=gen)
    head = LocalSALADHead(
        LocalSALADConfig(
            num_channels=32,
            num_clusters=4,
            cluster_dim=8,
            token_dim=16,
            hidden_dim=24,
            dropout=0.0,
        )
    )

    desc_a = head(patches, global_token=global_token)
    desc_b = head(patches, global_token=global_token)
    mismatched_global = torch.randn(3, 64, generator=gen)
    desc_mismatch = head(patches, global_token=mismatched_global)

    assert desc_a.shape == (3, 48)
    assert desc_mismatch.shape == (3, 48)
    assert torch.allclose(desc_a, desc_b, atol=1e-6)
    assert torch.isfinite(desc_a).all()
    assert torch.isfinite(desc_mismatch).all()
    assert torch.allclose(desc_a.norm(dim=1), torch.ones(3), atol=1e-5)
    cluster_blocks = desc_a[:, 16:].reshape(3, 4, 8).transpose(1, 2)
    assert torch.isfinite(cluster_blocks).all()
    loss = symmetric_infonce_loss(desc_a[:2], desc_b[:2], temperature=0.2)
    assert torch.isfinite(loss)
    assert normalize_embedding_aggregation("salad") == "local_salad"
    assert normalize_embedding_aggregation("local_salad") == "local_salad"
    assert normalize_embedding_aggregation("anything_else") == "pooled"

    polygon = api._class_analysis_parse_yolo_geometry(
        "2 0.1 0.1 0.2 0.1 0.2 0.2",
        image_width=100,
        image_height=100,
    )
    assert polygon["kind"] == "polygon"
    assert polygon["class_id"] == 2
    assert polygon["bbox_xyxy"] == [10.0, 10.0, 20.0, 20.0]

    crop_bounds = api._class_analysis_crop_bounds(
        [40, 30, 60, 70],
        image_width=100,
        image_height=100,
        crop_mode="padded_square",
        padding_ratio=0.1,
    )
    assert crop_bounds == (26, 26, 74, 74)


def test_mlx_local_salad_matches_torch_state_and_trains_one_step():
    if not local_salad_mlx_available():
        pytest.skip("MLX is not available")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(321)
    config = LocalSALADConfig(
        num_channels=8,
        num_clusters=3,
        cluster_dim=5,
        token_dim=7,
        hidden_dim=11,
        dropout=0.0,
    )
    torch_head = LocalSALADHead(config)
    torch_head.eval()
    mlx_head = MLXLocalSALADHead(config)
    mlx_head.load_torch_state_dict(torch_head.state_dict())
    patches = torch.randn(4, 9, 8, generator=gen)
    global_token = torch.randn(4, 8, generator=gen)

    with torch.no_grad():
        torch_out = torch_head(patches, global_token=global_token).detach().numpy()
    mlx_out = encode_local_salad_mlx(mlx_head, patches, global_token=global_token)

    assert mlx_out.shape == torch_out.shape == (4, 22)
    assert np.max(np.abs(torch_out - mlx_out)) < 1e-3
    assert np.allclose(np.linalg.norm(mlx_out, axis=1), np.ones(4), atol=1e-5)

    optimizer = make_mlx_local_salad_optimizer(learning_rate=1e-4, weight_decay=0.0)
    loss_value = mlx_local_salad_train_step(
        mlx_head,
        optimizer,
        patches,
        global_token,
        patches + 0.01,
        global_token + 0.01,
        temperature=0.2,
    )
    state_dict = mlx_local_salad_state_dict(mlx_head)

    assert np.isfinite(loss_value)
    assert set(torch_head.state_dict()) == set(state_dict)
    assert state_dict["token_features.0.weight"].shape == torch_head.state_dict()["token_features.0.weight"].shape


def test_class_analysis_flags_neighbor_disagreement_only_in_all_classes():
    records = [
        _record("p0", "car"),
        _record("p1", "boat"),
        _record("p2", "boat"),
        _record("p3", "boat"),
        _record("p4", "car"),
        _record("p5", "car"),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [1.0, -0.01, 0.0],
            [0.99, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    all_classes = api._class_analysis_build_result(
        records,
        embeddings,
        summary={"analysis_scope": "all_classes"},
        projection="pca",
        projection_neighbor_k=15,
        neighbor_k=3,
        seed=13,
    )
    candidate_ids = {item["point_id"] for item in all_classes["wrong_class_candidates"]}
    assert "p0" in candidate_ids
    p0 = next(point for point in all_classes["points"] if point["point_id"] == "p0")
    assert p0["suggested_neighbor_class"] == "boat"
    assert p0["is_wrong_class_candidate"] is True

    selected_class = api._class_analysis_build_result(
        records,
        embeddings,
        summary={"analysis_scope": "selected_class"},
        projection="pca",
        projection_neighbor_k=15,
        neighbor_k=3,
        seed=13,
    )
    assert selected_class["wrong_class_candidates"] == []
    assert all(point["is_wrong_class_candidate"] is False for point in selected_class["points"])


def test_class_analysis_stratified_sampling_keeps_classes_represented():
    records = [_record(f"a{i}", "alpha") for i in range(10)]
    records.extend(_record(f"b{i}", "beta") for i in range(10))

    selected = api._class_analysis_stratified_indices(records, cap=6, seed=7)
    selected_classes = [records[idx]["class_name"] for idx in selected]

    assert len(selected) == 6
    assert "alpha" in selected_classes
    assert "beta" in selected_classes


def test_class_analysis_sample_cap_defaults_to_unlimited():
    assert api._class_analysis_sample_cap(None) == 0
    assert api._class_analysis_sample_cap("") == 0
    assert api._class_analysis_sample_cap("0") == 0
    assert api._class_analysis_sample_cap("-5") == 0
    assert api._class_analysis_sample_cap("250") == 250

    records = [_record(f"p{i}", "alpha") for i in range(12)]
    assert api._class_analysis_stratified_indices(records, cap=0, seed=7) == list(range(12))


def test_class_analysis_rejects_local_salad_aggregation_before_queue():
    with pytest.raises(api.HTTPException) as disabled:
        api._normalize_class_analysis_request(
            {
                "encoder_type": "dinov3",
                "embedding_aggregation": "local_salad",
                "embedding_salad_head_id": "unit_head",
            }
        )
    assert disabled.value.status_code == 400
    assert disabled.value.detail == "local_salad_class_analysis_disabled"

    pooled = api._normalize_class_analysis_request(
        {
            "encoder_type": "dinov3",
            "embedding_aggregation": "pooled",
            "embedding_salad_head_id": "stale_head",
        }
    )
    assert pooled["embedding_aggregation"] == "pooled"
    assert pooled["embedding_salad_head_id"] == ""


def test_auto_class_training_rejects_local_salad_aggregation_before_dataset_validation():
    with pytest.raises(api.HTTPException) as disabled:
        asyncio.run(
            api.start_clip_training(
                embedding_aggregation="local_salad",
                embedding_salad_head_id="unit_head",
            )
        )
    assert disabled.value.status_code == 400
    assert disabled.value.detail == "local_salad_auto_class_disabled"


def test_auto_class_runtime_rejects_local_salad_artifacts_before_encoding():
    with pytest.raises(api.HTTPException) as disabled:
        api._encode_pil_batch_for_head(
            [Image.new("RGB", (8, 8), (10, 20, 30))],
            head={
                "encoder_type": "dinov3",
                "embedding_aggregation": "local_salad",
                "embedding_salad_head_id": "unit_head",
            },
        )
    assert disabled.value.status_code == 400
    assert disabled.value.detail == "local_salad_auto_class_disabled"


def test_cradio_embedding_contract_and_capabilities(monkeypatch):
    assert normalize_cradio_pooling("spatial") == "spatial_mean"
    assert normalize_cradio_pooling("summary+spatial") == "summary_spatial_concat"
    assert normalize_cradio_pooling("anything_else") == "summary"

    monkeypatch.setattr(
        cradio_embedding_utils,
        "_cradio_mlx_backend_status",
        lambda model_name=None, *, requested="mlx": CRadioBackendStatus(
            requested=requested,
            resolved="mlx",
            available=True,
            detail="Local MLX C-RADIOv4 backend (/tmp/model.safetensors)",
        ),
    )

    mlx = cradio_backend_status("mlx")
    assert mlx.resolved == "mlx"
    assert mlx.available is True
    assert "Local MLX C-RADIOv4 backend" in mlx.detail

    def model_specific_mlx_status(model_name=None, *, requested="mlx"):
        model = model_name or CRADIO_DEFAULT_MODEL
        return CRadioBackendStatus(
            requested=requested,
            resolved="mlx",
            available=model == CRADIO_DEFAULT_MODEL,
            detail=f"mlx status for {model}",
        )

    monkeypatch.setattr(cradio_embedding_utils, "_cradio_mlx_backend_status", model_specific_mlx_status)
    monkeypatch.setattr(cradio_embedding_utils.platform, "system", lambda: "Darwin")
    assert cradio_backend_status("auto", model_name=CRADIO_DEFAULT_MODEL).resolved == "mlx"
    assert cradio_backend_status("auto", model_name="nvidia/C-RADIOv4-H").resolved != "mlx"

    summary = torch.ones(2, 3)
    spatial = torch.zeros(2, 4, 3)
    unpacked = _unpack_cradio_outputs({"summary": summary, "spatial_features": spatial})
    assert unpacked[0] is summary
    assert unpacked[1] is spatial

    class FakeMLXEncoder:
        def encode_batch(self, images, image_size=512):
            assert len(images) == 2
            assert image_size == 512
            return types.SimpleNamespace(
                summary=np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32),
                spatial=np.asarray(
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[2.0, 0.0], [0.0, 2.0]],
                    ],
                    dtype=np.float32,
                ),
            )

    mlx_images = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]
    mlx_feats, mlx_spatial, mlx_summary = encode_cradio_images(
        FakeMLXEncoder(),
        None,
        "mlx",
        mlx_images,
        pooling="summary_spatial_concat",
        normalize=True,
        return_tokens=True,
    )
    assert mlx_feats.shape == (2, 4)
    assert mlx_spatial.shape == (2, 2, 2)
    assert mlx_summary.shape == (2, 2)
    assert np.allclose(np.linalg.norm(mlx_feats, axis=1), np.ones(2), atol=1e-6)

    caps = api._class_analysis_capabilities()
    assert "cradio" in caps["encoders"]
    assert caps["default_cradio_model"] == CRADIO_DEFAULT_MODEL
    assert "summary_spatial_concat" in caps["cradio_pooling_modes"]
    assert any(recipe["id"] == "cradio_summary" for recipe in caps["class_separation_recipes"])

    request = api._normalize_class_analysis_request(
        {
            "encoder_type": "cradio",
            "encoder_model": "",
            "cradio_pooling": "summary+spatial",
            "embedding_aggregation": "pooled",
            "embedding_salad_head_id": "stale_head",
        }
    )
    assert request["encoder_model"] == CRADIO_DEFAULT_MODEL
    assert request["cradio_pooling"] == "summary_spatial_concat"
    assert request["embedding_salad_head_id"] == ""


def test_cradio_head_encoding_uses_saved_pooling(monkeypatch):
    captured = {}

    monkeypatch.setattr(api, "resolve_cradio_torch_device", lambda _backend=None, **_kwargs: "cpu")
    monkeypatch.setattr(
        api,
        "_load_cradio_backbone_cached",
        lambda model_name, target_device, raise_on_error=False: ("model", "processor", model_name, "cpu"),
    )

    def fake_encode(model, processor, device_name, images, *, pooling, normalize=True, return_tokens=False):
        captured["pooling"] = pooling
        captured["normalize"] = normalize
        captured["return_tokens"] = return_tokens
        return np.asarray([[1.0, 2.0, 3.0] for _ in images], dtype=np.float32)

    monkeypatch.setattr(api, "encode_cradio_images", fake_encode)
    feats = api._encode_pil_batch_for_head(
        [Image.new("RGB", (8, 8), (10, 20, 30))],
        head={
            "encoder_type": "cradio",
            "encoder_model": "nvidia/C-RADIOv4-SO400M",
            "cradio_pooling": "spatial_mean",
            "normalize_embeddings": True,
        },
    )

    assert captured == {"pooling": "spatial_mean", "normalize": False, "return_tokens": False}
    assert feats.shape == (1, 3)
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0)


def test_class_analysis_capabilities_expose_only_normal_recipe_controls():
    caps = api._class_analysis_capabilities()

    assert caps["preprocess_modes"] == ["canonical"]
    assert caps["embedding_adjustments"] == ["remove_size_bias"]
    assert caps["expert_preprocess_modes"] == ["native", "canonical"]
    assert caps["expert_embedding_adjustments"] == ["none", "remove_size_bias"]
    assert caps["default_preprocess_mode"] == "canonical"
    assert caps["default_embedding_adjustment"] == "remove_size_bias"
    assert caps["default_projection_neighbor_k"] == 50
    assert caps["embedding_aggregation_modes"] == ["pooled"]
    assert "local_salad_heads" not in caps
    assert "local_salad_policy" not in caps
    assert not any(recipe["id"] == "local_salad" for recipe in caps["class_separation_recipes"])


def test_dinov3_head_encoding_uses_default_model_constant(monkeypatch):
    class DummyProcessor:
        def __call__(self, images, return_tensors="pt"):
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros(len(images), 1)}

    class DummyModel:
        def __call__(self, **inputs):
            batch = int(inputs["pixel_values"].shape[0])
            return types.SimpleNamespace(
                last_hidden_state=torch.ones(batch, 2, 2),
                pooler_output=torch.tensor([[3.0, 4.0], [0.0, 5.0]], dtype=torch.float32)[:batch],
            )

    monkeypatch.setattr(api, "dinov3_model", DummyModel())
    monkeypatch.setattr(api, "dinov3_processor", DummyProcessor())
    monkeypatch.setattr(api, "dinov3_model_name", api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL)
    monkeypatch.setattr(api, "dinov3_model_device", "cpu")
    monkeypatch.setattr(api, "_load_dinov3_backbone", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected load")))

    feats = api._encode_pil_batch_for_head(
        [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
        head={"encoder_type": "dinov3", "normalize_embeddings": True},
        device_override="cpu",
    )

    assert feats.shape == (2, 2)
    assert np.allclose(np.linalg.norm(feats, axis=1), np.ones(2), atol=1e-6)


def test_class_split_experiment_metrics_use_absolute_leakage_and_macro_purity(tmp_path):
    result = {
        "summary": {
            "object_count": 3,
            "raw_object_count": 3,
            "sample_cap": 0,
            "projection": "umap",
            "projection_neighbor_k": 50,
            "neighbor_k": 15,
            "embedding_cache": {"hits": 1, "total": 4},
            "wrong_class_candidate_count": 1,
        },
        "diagnostics": {
            "strongest_size_axis": {"metric": "bbox_area", "correlation": -0.73},
            "axis_correlations": {
                "x": {"bbox_area": -0.73, "crop_area": 0.25},
                "y": {"bbox_area": 0.11},
            },
        },
        "clusters": {"best_k": 2, "candidates": [{"silhouette": 0.31}, {"silhouette": 0.12}]},
        "points": [
            {"class_name": "car", "same_class_neighbor_ratio": 1.0},
            {"class_name": "car", "same_class_neighbor_ratio": 0.8},
            {"class_name": "boat", "same_class_neighbor_ratio": 0.2},
        ],
    }
    run = {
        "analysis_scope": "all_classes",
        "class_name": "",
        "encoder_type": "dinov3",
        "encoder_model": "test",
        "preprocess_mode": "canonical",
        "canonical_size": 336,
        "crop_mode": "padded_square",
        "padding_ratio": 0.08,
        "dinov3_pooling": "pooler",
        "embedding_aggregation": "pooled",
        "background_mode": "full_crop",
        "embedding_view_mode": "tight_context",
        "embedding_adjustment": "remove_size_bias",
        "embedding_postprocess": "none",
    }

    metrics = class_split_experiments._metrics_from_result(
        "precise_tight_context_all_classes",
        run,
        result,
        12.5,
    )

    assert metrics["variant"] == "precise_tight_context"
    assert metrics["embedding_aggregation"] == "pooled"
    assert metrics["strongest_size_axis_correlation"] == -0.73
    assert np.isclose(metrics["strongest_size_axis_abs_correlation"], 0.73)
    assert np.isclose(metrics["mean_abs_size_correlation"], (0.73 + 0.25 + 0.11) / 3)
    assert np.isclose(metrics["mean_neighbor_same_class_ratio"], (1.0 + 0.8 + 0.2) / 3)
    assert np.isclose(metrics["class_balanced_neighbor_same_class_ratio"], ((1.0 + 0.8) / 2 + 0.2) / 2)
    assert metrics["worst_class_neighbor_same_class"] == "boat"
    assert metrics["worst_class_neighbor_same_class_count"] == 1

    class_split_experiments._write_leaderboard(tmp_path, [metrics])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    leaderboard = (tmp_path / "leaderboard.csv").read_text(encoding="utf-8")
    assert "size_abs=0.730" in report
    assert "class_balanced_nn=0.550" in report
    assert "strongest_size_axis_abs_correlation" in leaderboard
    assert "class_balanced_neighbor_same_class_ratio" in leaderboard

    cradio_runs = class_split_experiments._cradio_matrix(sample_cap=11)
    assert cradio_runs
    assert all(run["encoder_type"] == "cradio" for run in cradio_runs)
    assert {run["cradio_pooling"] for run in cradio_runs} >= {"summary", "spatial_mean", "summary_spatial_concat"}
    assert all(run["sample_cap"] == 11 for run in cradio_runs)


def test_class_analysis_source_reads_active_workspace_manifest(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "images").mkdir(parents=True)
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_label": "browser snapshot",
                "labelmap": ["car", "boat"],
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "example.jpg",
                        "frontend_image_key": "train/original/example.jpg",
                        "label_lines": ["0 0.5 0.5 0.2 0.2"],
                    }
                ],
                "yolo_layout": "flat",
            }
        ),
        encoding="utf-8",
    )

    source = api._class_analysis_source(
        {
            "source_mode": "active_workspace",
            "workspace_id": "ca_test",
            "workspace_dir": str(workspace),
            "workspace_manifest_path": str(manifest_path),
        }
    )

    assert source["source_mode"] == "active_workspace"
    assert source["source_id"] == "ca_test"
    assert source["dataset_root"] == workspace.resolve()
    assert source["labelmap"] == ["car", "boat"]
    assert source["manifest"]["images"][0]["frontend_image_key"] == "train/original/example.jpg"


def test_class_analysis_encode_crops_reports_batch_progress(monkeypatch):
    calls = []

    def fake_encode(images, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        calls.append(len(images))
        return np.ones((len(images), 4), dtype=np.float32)

    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)
    job = api.ClassAnalysisJob(job_id="ca_test")
    crops = [Image.new("RGB", (8, 8), (idx, idx, idx)) for idx in range(5)]

    feats = api._class_analysis_encode_crops(
        crops,
        job=job,
        head={"encoder_type": "dinov3", "normalize_embeddings": True},
        batch_size=2,
    )

    assert feats.shape == (5, 4)
    assert calls == [2, 2, 1]
    assert job.progress == 0.70
    assert any("batch 1/3" in entry["message"] for entry in job.logs)
    assert "Encoded 5/5 crops with DINOv3" in job.message


def test_class_analysis_umap_uses_projection_neighbors(monkeypatch):
    captured = {}

    class FakeUMAP:
        def __init__(self, *, n_components, n_neighbors, min_dist, metric, random_state):
            captured.update(
                {
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "metric": metric,
                    "random_state": random_state,
                }
            )

        def fit_transform(self, embeddings):
            return np.zeros((embeddings.shape[0], 2), dtype=np.float32)

    monkeypatch.setitem(__import__("sys").modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))
    embeddings = np.eye(80, 8, dtype=np.float32)
    warnings = []

    coords, used = api._class_analysis_project_embeddings(
        embeddings,
        projection="umap",
        projection_neighbor_k=50,
        seed=99,
        warnings=warnings,
    )

    assert used == "umap"
    assert coords.shape == (80, 2)
    assert captured["n_neighbors"] == 50
    assert captured["metric"] == "cosine"
    assert warnings == []


def test_class_analysis_size_bias_adjustment_reduces_area_axis_signal():
    records = []
    raw = []
    for idx in range(30):
        side = 10 + idx * 4
        records.append(
            {
                "point_id": f"p{idx}",
                "class_name": "light_vehicle",
                "width": side,
                "height": side,
                "crop_xyxy": [0, 0, side + 4, side + 4],
            }
        )
        area_signal = np.log1p(side * side)
        semantic_signal = 1.0 if idx % 2 else -1.0
        raw.append([area_signal, semantic_signal, semantic_signal * 0.25])
    embeddings = np.asarray(raw, dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    before = api._class_analysis_projection_diagnostics(records, embeddings[:, :2])

    adjusted, info = api._class_analysis_apply_embedding_adjustment(
        embeddings,
        records,
        mode="remove_size_bias",
    )
    after = api._class_analysis_projection_diagnostics(records, adjusted[:, :2])

    assert info["applied"] is True
    assert "log_bbox_area" in info["covariates"]
    assert abs(before["strongest_size_axis"]["correlation"]) > 0.9
    assert abs(after["strongest_size_axis"]["correlation"]) < 0.25


def test_class_analysis_canonical_preprocess_and_embedding_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", tmp_path)
    calls = []

    def fake_encode(images, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        calls.append(len(images))
        return np.asarray([[idx + 1, idx + 2, idx + 3] for idx in range(len(images))], dtype=np.float32)

    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)
    crop = Image.new("RGB", (20, 10), (120, 80, 40))
    canonical = api._class_analysis_preprocess_crop(crop, mode="canonical", canonical_size=96)
    assert canonical.size == (96, 96)

    records = [
        {"point_id": "a", "crop_cache_key": "crop-a"},
        {"point_id": "b", "crop_cache_key": "crop-b"},
    ]
    head = {"encoder_type": "dinov3", "encoder_model": "test-dino", "normalize_embeddings": True}
    stats = {}
    first = api._class_analysis_encode_crops(
        [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
        job=api.ClassAnalysisJob(job_id="cache_a"),
        head=head,
        batch_size=8,
        records=records,
        cache_stats=stats,
    )
    assert first.shape == (2, 3)
    assert stats["hits"] == 0
    assert stats["misses"] == 2
    assert calls == [2]

    def fail_encode(*args, **kwargs):
        raise AssertionError("cached embeddings should avoid encoder calls")

    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fail_encode)
    stats = {}
    second = api._class_analysis_encode_crops(
        [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
        job=api.ClassAnalysisJob(job_id="cache_b"),
        head=head,
        batch_size=8,
        records=records,
        cache_stats=stats,
    )
    assert np.allclose(first, second)
    assert stats["hits"] == 2
    assert stats["misses"] == 0


def test_class_analysis_embedding_cache_rejects_invalid_arrays(tmp_path):
    bad_shape = tmp_path / "bad_shape.npy"
    bad_nan = tmp_path / "bad_nan.npy"
    good = tmp_path / "good.npy"

    np.save(bad_shape, np.zeros((1, 3), dtype=np.float32))
    np.save(bad_nan, np.asarray([1.0, np.nan], dtype=np.float32))
    np.save(good, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

    assert api._class_analysis_load_cached_embedding(bad_shape) is None
    assert api._class_analysis_load_cached_embedding(bad_nan) is None
    assert np.allclose(api._class_analysis_load_cached_embedding(good), [1.0, 2.0, 3.0])


def test_class_analysis_corrupt_cache_rematerializes_real_crop(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "sample.jpg"
    Image.new("RGB", (80, 60), (20, 40, 60)).save(image_path)
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_label": "browser snapshot",
                "labelmap": ["car"],
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "sample.jpg",
                        "frontend_image_key": "train/original/sample.jpg",
                        "label_lines": ["0 0.5 0.5 0.25 0.25"],
                    }
                ],
                "yolo_layout": "flat",
            }
        ),
        encoding="utf-8",
    )
    corrupt_embedding = tmp_path / "corrupt.npy"
    cached_thumb = tmp_path / "cached_thumb.jpg"
    np.save(corrupt_embedding, np.zeros((1, 3), dtype=np.float32))
    Image.new("RGB", (8, 8), (1, 2, 3)).save(cached_thumb)

    monkeypatch.setattr(api, "_class_analysis_embedding_cache_path", lambda _cache_key: corrupt_embedding)
    monkeypatch.setattr(api, "_class_analysis_thumbnail_cache_path", lambda _crop_cache_key: cached_thumb)

    job = api.ClassAnalysisJob(job_id="ca_corrupt_cache")
    records, crops, summary = api._class_analysis_collect_records(
        {
            "source_mode": "active_workspace",
            "workspace_id": "ca_test",
            "workspace_dir": str(workspace),
            "workspace_manifest_path": str(manifest_path),
            "analysis_scope": "selected_class",
            "class_name": "car",
            "preprocess_mode": "canonical",
            "canonical_size": 64,
            "crop_mode": "padded_square",
            "padding_ratio": 0.08,
            "background_mode": "full_crop",
            "embedding_view_mode": "single",
            "encoder_type": "dinov3",
            "encoder_model": "test-dino",
            "dinov3_pooling": "pooler",
        },
        job=job,
        out_dir=tmp_path / "out",
    )

    try:
        assert len(records) == 1
        assert len(crops) == 1
        assert summary["object_count"] == 1
        assert records[0]["crop_cache_reused"] is False
        assert records[0]["embedding_views"]
        assert crops[0].size == (64, 64)
    finally:
        for crop in crops:
            api._close_crop_item(crop)


def test_class_analysis_cache_validation_uses_cradio_recipe(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "sample.jpg"
    Image.new("RGB", (80, 60), (20, 40, 60)).save(image_path)
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_label": "browser snapshot",
                "labelmap": ["car"],
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "sample.jpg",
                        "frontend_image_key": "train/original/sample.jpg",
                        "label_lines": ["0 0.5 0.5 0.25 0.25"],
                    }
                ],
                "yolo_layout": "flat",
            }
        ),
        encoding="utf-8",
    )
    cached_thumb = tmp_path / "cached_thumb.jpg"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(cached_thumb)
    captured_heads = []

    monkeypatch.setattr(api, "_class_analysis_thumbnail_cache_path", lambda _crop_cache_key: cached_thumb)

    def fake_cached_valid(_crop_cache_key, head):
        captured_heads.append(dict(head))
        return False

    monkeypatch.setattr(api, "_class_analysis_cached_embedding_valid", fake_cached_valid)

    job = api.ClassAnalysisJob(job_id="ca_cradio_cache_recipe")
    records, crops, summary = api._class_analysis_collect_records(
        {
            "source_mode": "active_workspace",
            "workspace_id": "ca_test",
            "workspace_dir": str(workspace),
            "workspace_manifest_path": str(manifest_path),
            "analysis_scope": "selected_class",
            "class_name": "car",
            "preprocess_mode": "canonical",
            "canonical_size": 64,
            "crop_mode": "padded_square",
            "padding_ratio": 0.08,
            "background_mode": "full_crop",
            "embedding_view_mode": "single",
            "encoder_type": "cradio",
            "encoder_model": CRADIO_DEFAULT_MODEL,
            "cradio_pooling": "spatial_mean",
        },
        job=job,
        out_dir=tmp_path / "out",
    )

    try:
        assert len(records) == 1
        assert summary["object_count"] == 1
        assert captured_heads
        assert any(head["encoder_type"] == "cradio" for head in captured_heads)
        assert any(head["encoder_model"] == CRADIO_DEFAULT_MODEL for head in captured_heads)
        assert any(head["cradio_pooling"] == "spatial_mean" for head in captured_heads)
    finally:
        for crop in crops:
            api._close_crop_item(crop)


def test_class_analysis_multiview_embedding_composes_before_postprocess(monkeypatch):
    captured = {}

    def fake_encode(images, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        captured["image_count"] = len(images)
        captured["geometry_records"] = geometry_records
        return np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)
    head = {"encoder_type": "dinov3", "normalize_embeddings": True}
    feats = api._encode_embedding_items_for_head(
        [(Image.new("RGB", (8, 8)), Image.new("RGB", (12, 12)))],
        head=head,
    )

    assert captured["image_count"] == 2
    assert captured["geometry_records"] is None
    assert feats.shape == (1, 4)
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0)


def test_classifier_crop_for_head_uses_saved_embedding_recipe(monkeypatch):
    captured = {}
    head = {
        "encoder_type": "dinov3",
        "normalize_embeddings": True,
        "preprocess_mode": "canonical",
        "canonical_size": 80,
        "embedding_crop_mode": "padded_square",
        "embedding_crop_padding_ratio": 0.5,
        "background_mode": "darken_outside_box",
        "embedding_view_mode": "single",
        "embedding_adjustment_transform": {"mode": "remove_size_bias"},
    }

    def fake_encode(images, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        crop_pixels = np.asarray(images[0], dtype=np.float32)
        captured["image_size"] = images[0].size
        captured["outside_mean"] = float(crop_pixels[2, 2].mean())
        captured["inside_mean"] = float(crop_pixels[40, 40].mean())
        captured["geometry"] = geometry_records[0]
        captured["head"] = head
        return np.ones((1, 4), dtype=np.float32)

    monkeypatch.setattr(api, "_active_classifier_head_for_inference", lambda: head)
    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)

    image = Image.new("RGB", (100, 60), (20, 40, 60))
    feats = api._encode_classifier_xyxy_for_active(image, [40, 20, 60, 30])

    assert feats.shape == (1, 4)
    assert captured["image_size"] == (80, 80)
    assert captured["outside_mean"] < captured["inside_mean"]
    assert captured["geometry"]["bbox_xyxy"] == [40.0, 20.0, 60.0, 30.0]
    assert captured["geometry"]["crop_xyxy"] == [30, 5, 70, 45]
    assert captured["geometry"]["background_mode"] == "darken_outside_box"
    assert captured["geometry"]["embedding_view_mode"] == "single"
    assert captured["head"]["embedding_adjustment_transform"]["mode"] == "remove_size_bias"


def test_classifier_multiview_inference_composes_views_before_size_bias(monkeypatch):
    captured = {}
    transform = {"mode": "remove_size_bias", "sentinel": True}
    head = {
        "encoder_type": "dinov3",
        "normalize_embeddings": False,
        "preprocess_mode": "canonical",
        "canonical_size": 48,
        "embedding_crop_mode": "padded_square",
        "embedding_crop_padding_ratio": 0.08,
        "background_mode": "full_crop",
        "embedding_view_mode": "tight_context",
        "embedding_adjustment_transform": transform,
    }

    def fake_encode(images, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        captured["raw_image_sizes"] = [image.size for image in images]
        captured["raw_geometry_records"] = geometry_records
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def fake_residualizer(embeddings, covariates, residualizer, *, normalize=True):
        captured["residualizer_embedding_shape"] = embeddings.shape
        captured["residualizer_covariate_shape"] = covariates.shape
        captured["residualizer_transform"] = residualizer
        captured["residualizer_normalize"] = normalize
        return np.asarray(embeddings, dtype=np.float32) + 10.0

    monkeypatch.setattr(api, "_active_classifier_head_for_inference", lambda: head)
    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)
    monkeypatch.setattr(api, "apply_size_bias_residualizer", fake_residualizer)

    image = Image.new("RGB", (96, 72), (30, 60, 90))
    feats = api._encode_classifier_xyxy_for_active(image, [20, 18, 36, 34])

    assert feats.shape == (1, 4)
    assert captured["raw_image_sizes"] == [(64, 64), (64, 64)]
    assert captured["raw_geometry_records"] is None
    assert captured["residualizer_embedding_shape"] == (1, 4)
    assert captured["residualizer_covariate_shape"] == (1, 4)
    assert captured["residualizer_transform"] == transform
    assert captured["residualizer_normalize"] is True
    assert np.all(feats > 9.0)


def test_classifier_detection_scoring_closes_preprocessed_crops(monkeypatch):
    crop = Image.new("RGB", (16, 16), (10, 20, 30))
    original_close = crop.close
    closed = []

    def tracked_close():
        closed.append(True)
        original_close()

    crop.close = tracked_close

    def fake_crop_for_head(pil_img, xyxy, head):
        return crop, {
            "bbox_xyxy": [float(v) for v in xyxy],
            "crop_xyxy": [0, 0, 16, 16],
            "width": 10,
            "height": 10,
        }

    def fake_encode(crops, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        assert crops == [crop]
        assert geometry_records and geometry_records[0]["crop_xyxy"] == [0, 0, 16, 16]
        return np.ones((1, 2), dtype=np.float32)

    monkeypatch.setattr(api, "_classifier_crop_for_head", fake_crop_for_head)
    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)
    monkeypatch.setattr(
        api,
        "_clip_head_predict_proba",
        lambda feats, head, empty_cache_fn=None: np.asarray([[0.2, 0.8]], dtype=np.float32),
    )

    image = Image.new("RGB", (100, 60), (20, 40, 60))
    detection = {"label": "boat", "bbox_xyxy_px": [1, 2, 11, 12]}
    scores = api._score_detections_with_clip_head(
        [detection],
        pil_img=image,
        clip_head={"classes": np.asarray(["car", "boat"], dtype=object)},
        score_mode="clip_head_prob",
    )

    assert set(scores) == {id(detection)}
    assert np.isclose(scores[id(detection)], 0.8)
    assert closed == [True]


def test_classifier_loader_preserves_embedding_recipe_metadata(tmp_path):
    class DummyClassifier:
        classes_ = np.asarray(["car", "boat"])
        coef_ = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        intercept_ = np.asarray([0.0], dtype=np.float32)

    classifier_path = tmp_path / "test_classifier.pkl"
    meta_path = tmp_path / "test_classifier.meta.pkl"
    classifier_path.write_bytes(b"classifier")
    meta_path.write_bytes(b"meta")
    transform = {
        "mode": "remove_size_bias",
        "keep_mask": [True, True, False, False],
        "mean": [1.0, 2.0],
        "std": [0.5, 0.25],
        "beta": [[0.0, 0.0, 0.0, 0.0]] * 3,
    }

    def fake_joblib_load(path):
        if path.endswith(".meta.pkl"):
            return {
                "encoder_type": "dinov3",
                "encoder_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "mlp_normalize_embeddings": True,
                "preprocess_mode": "canonical",
                "canonical_size": 336,
                "embedding_crop_mode": "padded_square",
                "embedding_crop_padding_ratio": 0.08,
                "background_mode": "blur_outside_box",
                "embedding_view_mode": "tight_context",
                "embedding_adjustment": "remove_size_bias",
                "embedding_adjustment_transform": transform,
                "dinov3_pooling": "pooler",
                "cradio_pooling": "summary_spatial_concat",
                "embedding_aggregation": "local_salad",
                "embedding_salad_head_id": "unit_head",
            }
        return DummyClassifier()

    class HttpError(Exception):
        def __init__(self, *, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    head = _load_clip_head_from_classifier_impl(
        classifier_path,
        joblib_load_fn=fake_joblib_load,
        http_exception_cls=HttpError,
        clip_head_background_indices_fn=lambda classes: [],
        resolve_head_normalize_embeddings_fn=lambda clf, default: default,
        infer_clip_model_fn=lambda dim, default: default,
        active_clip_model_name=None,
        default_clip_model="ViT-B/32",
        logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
    )

    assert head["encoder_type"] == "dinov3"
    assert head["preprocess_mode"] == "canonical"
    assert head["canonical_size"] == 336
    assert head["embedding_crop_mode"] == "padded_square"
    assert head["embedding_crop_padding_ratio"] == 0.08
    assert head["background_mode"] == "blur_outside_box"
    assert head["embedding_view_mode"] == "tight_context"
    assert head["embedding_adjustment"] == "remove_size_bias"
    assert head["embedding_adjustment_transform"] == transform
    assert head["dinov3_pooling"] == "pooler"
    assert head["cradio_pooling"] == "summary_spatial_concat"
    assert head["embedding_aggregation"] == "local_salad"
    assert head["embedding_salad_head_id"] == "unit_head"


def test_classifier_loader_preserves_mlp_gelu_activation(tmp_path):
    classifier_path = tmp_path / "gelu_head.pkl"
    meta_path = tmp_path / "gelu_head.meta.pkl"
    classifier_path.write_bytes(b"classifier")
    meta_path.write_bytes(b"meta")

    clf_obj = {
        "classifier_type": "mlp",
        "classes": np.asarray(["car", "boat"], dtype=object),
        "embedding_dim": 2,
        "layers": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "gelu",
            },
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "linear",
            },
        ],
    }

    def fake_joblib_load(path):
        if path.endswith(".meta.pkl"):
            return {
                "encoder_type": "dinov3",
                "encoder_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "mlp_normalize_embeddings": True,
            }
        return clf_obj

    class HttpError(Exception):
        def __init__(self, *, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    head = _load_clip_head_from_classifier_impl(
        classifier_path,
        joblib_load_fn=fake_joblib_load,
        http_exception_cls=HttpError,
        clip_head_background_indices_fn=lambda classes: [],
        resolve_head_normalize_embeddings_fn=lambda clf, default: default,
        infer_clip_model_fn=lambda dim, default: default,
        active_clip_model_name=None,
        default_clip_model="ViT-B/32",
        logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
    )

    assert head["layers"][0]["activation"] == "gelu"
    assert head["classes"] == ["car", "boat"]


def test_clip_head_predict_proba_replays_mlp_gelu_activation():
    head = {
        "classifier_type": "mlp",
        "classes": ["car", "boat"],
        "proba_mode": "softmax",
        "layers": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "gelu",
            },
            {
                "weight": np.asarray([[1.0, -0.5], [-0.75, 0.25]], dtype=np.float32),
                "bias": np.asarray([0.1, -0.2], dtype=np.float32),
                "activation": "linear",
            },
        ],
    }
    feats = np.asarray([[-1.0, 2.0]], dtype=np.float32)

    hidden = 0.5 * feats * (1.0 + np.vectorize(math.erf)(feats / math.sqrt(2.0)))
    logits = hidden @ head["layers"][1]["weight"].T + head["layers"][1]["bias"]
    logits = logits - np.max(logits, axis=1, keepdims=True)
    expected = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    actual = api._clip_head_predict_proba(feats, head)

    assert np.allclose(actual, expected.astype(np.float32), atol=1e-6)


def test_clip_head_predict_proba_replays_mlp_arcface_output_layer():
    head = {
        "classifier_type": "mlp",
        "classes": ["car", "boat"],
        "proba_mode": "softmax",
        "arcface": True,
        "arcface_scale": 10.0,
        "layers": [
            {
                "weight": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "linear",
            },
        ],
    }
    feats = np.asarray([[3.0, 4.0]], dtype=np.float32)
    feats_norm = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    weight = head["layers"][0]["weight"]
    weight_norm = weight / np.linalg.norm(weight, axis=1, keepdims=True)
    logits = (feats_norm @ weight_norm.T) * 10.0
    logits = logits - np.max(logits, axis=1, keepdims=True)
    expected = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    actual = api._clip_head_predict_proba(feats, head)

    assert np.allclose(actual, expected.astype(np.float32), atol=1e-6)


def test_clip_head_predict_proba_normalizes_ovr_probabilities():
    feats = np.asarray([[1.0, 0.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "coef": np.asarray([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        "intercept": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "proba_mode": "ovr",
    }

    actual = api._clip_head_predict_proba(feats, head)
    raw = 1.0 / (1.0 + np.exp(-np.asarray([[1.0, 0.0, -1.0]], dtype=np.float32)))
    expected = raw / raw.sum(axis=1, keepdims=True)

    assert np.allclose(actual, expected, atol=1e-6)
    assert np.allclose(actual.sum(axis=1), [1.0])


def test_clip_head_predict_proba_accepts_numpy_array_classes():
    feats = np.asarray([[1.0, 0.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "classes": np.asarray(["car", "boat"], dtype=object),
        "coef": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "intercept": np.zeros(2, dtype=np.float32),
        "proba_mode": "softmax",
    }

    actual = api._clip_head_predict_proba(feats, head)

    assert actual is not None
    assert actual.shape == (1, 2)
    assert float(actual[0, 0]) > float(actual[0, 1])


def test_clip_auto_predict_details_accepts_numpy_array_classes(monkeypatch):
    head = {
        "classifier_type": "logreg",
        "classes": np.asarray(["car", "boat"], dtype=object),
        "coef": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "intercept": np.zeros(2, dtype=np.float32),
        "proba_mode": "softmax",
    }
    monkeypatch.setattr(api, "_active_classifier_head_for_inference", lambda: head)

    details = api._clip_auto_predict_details(
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        background_guard=False,
    )

    assert details["error"] is None
    assert details["label"] == "car"
    assert details["second_label"] == "boat"


def test_clip_head_predict_proba_fails_closed_on_embedding_width_mismatch():
    feats = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    logreg_head = {
        "classifier_type": "logreg",
        "coef": np.zeros((2, 2), dtype=np.float32),
        "intercept": np.zeros(2, dtype=np.float32),
        "proba_mode": "softmax",
    }
    mlp_head = {
        "classifier_type": "mlp",
        "layers": [
            {
                "weight": np.zeros((2, 2), dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "linear",
            }
        ],
    }

    assert api._clip_head_predict_proba(feats, logreg_head) is None
    assert api._clip_head_predict_proba(feats, mlp_head) is None


def test_clip_head_predict_proba_fails_closed_on_class_count_mismatch():
    feats = np.asarray([[1.0, 2.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "classes": ["car", "boat", "plane"],
        "coef": np.zeros((2, 2), dtype=np.float32),
        "intercept": np.zeros(2, dtype=np.float32),
        "proba_mode": "softmax",
    }

    assert api._clip_head_predict_proba(feats, head) is None


def test_clip_head_predict_proba_fails_closed_on_layer_norm_shape_mismatch():
    feats = np.asarray([[1.0, 2.0]], dtype=np.float32)
    head = {
        "classifier_type": "mlp",
        "classes": ["car", "boat"],
        "layers": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "layer_norm_weight": np.ones(3, dtype=np.float32),
                "activation": "linear",
            }
        ],
    }

    assert api._clip_head_predict_proba(feats, head) is None


def test_clip_head_predict_proba_fails_closed_on_malformed_arrays():
    feats = np.asarray([[1.0, 2.0]], dtype=np.float32)
    bad_logreg = {
        "classifier_type": "logreg",
        "coef": [[object(), 0.0]],
        "intercept": [0.0],
        "proba_mode": "softmax",
    }
    bad_mlp_weight = {
        "classifier_type": "mlp",
        "classes": ["car", "boat"],
        "layers": [
            {
                "weight": [[object(), 0.0], [0.0, 1.0]],
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "linear",
            }
        ],
    }
    bad_layer_norm = {
        "classifier_type": "mlp",
        "classes": ["car", "boat"],
        "layers": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "layer_norm_weight": [object(), 1.0],
                "activation": "linear",
            }
        ],
    }

    assert api._clip_head_predict_proba(feats, bad_logreg) is None
    assert api._clip_head_predict_proba(feats, bad_mlp_weight) is None
    assert api._clip_head_predict_proba(feats, bad_layer_norm) is None


def test_clip_head_predict_proba_ignores_malformed_logit_adjustment():
    feats = np.asarray([[1.0, 0.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "classes": ["car", "boat"],
        "coef": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "intercept": np.zeros(2, dtype=np.float32),
        "proba_mode": "softmax",
        "logit_adjustment_inference": True,
        "logit_adjustment": [object(), 0.0],
    }

    actual = api._clip_head_predict_proba(feats, head)

    assert actual is not None
    assert actual.shape == (1, 2)
    assert float(actual[0, 0]) > float(actual[0, 1])


def test_classifier_postprocess_matches_training_normalize_then_center_order():
    feats = np.asarray([[3.0, 4.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "normalize_embeddings": True,
        "embedding_center_values": [0.6, 0.8],
    }

    actual = api._postprocess_features_for_head(feats, head=head)

    assert np.allclose(actual, np.zeros((1, 2), dtype=np.float32), atol=1e-6)


def test_predict_base64_replays_classifier_crop_recipe_with_scaled_bbox(monkeypatch):
    captured = {}
    image = Image.new("RGB", (50, 100), (10, 20, 30))

    def fake_resolve(*args, **kwargs):
        return image, np.asarray(image), "token"

    def fake_encode(pil_img, xyxy):
        captured["image_size"] = pil_img.size
        captured["xyxy"] = [float(v) for v in xyxy]
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(api, "_active_encoder_ready", lambda: True)
    monkeypatch.setattr(api, "_resolve_detector_image_impl", fake_resolve)
    monkeypatch.setattr(api, "_encode_classifier_xyxy_for_active", fake_encode)
    monkeypatch.setattr(
        api,
        "_clip_auto_predict_details",
        lambda feats, background_guard=False: {
            "label": "car",
            "proba": 0.9,
            "second_label": "boat",
            "second_proba": 0.1,
            "margin": 0.8,
            "error": None,
        },
    )

    response = api.predict_base64(
        api.Base64Payload(
            image_base64="ignored",
            uuid="bbox-1",
            bbox_xyxy=[10.0, 20.0, 30.0, 60.0],
            image_width=100,
            image_height=200,
        )
    )

    assert response.prediction == "car"
    assert response.uuid == "bbox-1"
    assert captured["image_size"] == (50, 100)
    assert captured["xyxy"] == [5.0, 10.0, 15.0, 30.0]


def test_predict_base64_crop_only_uses_full_image_as_bbox(monkeypatch):
    captured = {}
    image = Image.new("RGB", (24, 16), (10, 20, 30))

    monkeypatch.setattr(api, "_active_encoder_ready", lambda: True)
    monkeypatch.setattr(
        api,
        "_resolve_detector_image_impl",
        lambda *args, **kwargs: (image, np.asarray(image), "token"),
    )
    def fake_encode(pil_img, xyxy):
        captured["xyxy"] = [float(v) for v in xyxy]
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(api, "_encode_classifier_xyxy_for_active", fake_encode)
    monkeypatch.setattr(
        api,
        "_clip_auto_predict_details",
        lambda feats, background_guard=False: {"label": "car", "error": None},
    )

    api.predict_base64(api.Base64Payload(image_base64="ignored", uuid="crop-1"))

    assert captured["xyxy"] == [0.0, 0.0, 24.0, 16.0]


def test_set_active_model_accepts_multiview_clip_embedding_width(tmp_path, monkeypatch):
    classifiers_root = tmp_path / "classifiers"
    labelmaps_root = tmp_path / "labelmaps"
    classifiers_root.mkdir()
    labelmaps_root.mkdir()
    classifier_path = classifiers_root / "clip_multiview.pkl"
    meta_path = classifiers_root / "clip_multiview.meta.pkl"
    labelmap_path = labelmaps_root / "labels.pkl"
    classifier = types.SimpleNamespace(
        classes_=np.asarray(["car", "boat"], dtype=object),
        coef_=np.zeros((2, 1536), dtype=np.float32),
        intercept_=np.zeros(2, dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    api.joblib.dump(classifier, classifier_path)
    api.joblib.dump(
        {
            "clip_model": "ViT-L/14",
            "encoder_type": "clip",
            "encoder_model": "ViT-L/14",
            "embedding_view_mode": "tight_context",
            "embedding_dim": 1536,
        },
        meta_path,
    )
    api.joblib.dump(["car", "boat"], labelmap_path)

    class FakeClipModel:
        visual = types.SimpleNamespace(output_dim=768)

    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(api, "clip_model", None)
    monkeypatch.setattr(api, "clip_preprocess", None)
    monkeypatch.setattr(api, "clip_model_name", "ViT-B/32")
    monkeypatch.setattr(api.clip, "load", lambda name, device=None: (FakeClipModel(), object()))

    payload = api.set_active_model(
        api.ActiveModelRequest(
            classifier_path=str(classifier_path),
            labelmap_path=str(labelmap_path),
        )
    )

    assert payload["encoder_type"] == "clip"
    assert payload["encoder_ready"] is True
    assert api.active_classifier_head["embedding_dim"] == 1536
    assert api.active_classifier_head["embedding_view_mode"] == "tight_context"


def test_set_active_model_accepts_multiview_dinov3_embedding_width(tmp_path, monkeypatch):
    classifiers_root = tmp_path / "classifiers"
    labelmaps_root = tmp_path / "labelmaps"
    classifiers_root.mkdir()
    labelmaps_root.mkdir()
    classifier_path = classifiers_root / "dino_multiview.pkl"
    meta_path = classifiers_root / "dino_multiview.meta.pkl"
    labelmap_path = labelmaps_root / "labels.pkl"
    classifier = types.SimpleNamespace(
        classes_=np.asarray(["car", "boat"], dtype=object),
        coef_=np.zeros((2, 2048), dtype=np.float32),
        intercept_=np.zeros(2, dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    api.joblib.dump(classifier, classifier_path)
    api.joblib.dump(
        {
            "encoder_type": "dinov3",
            "encoder_model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "embedding_view_mode": "tight_context",
            "dinov3_pooling": "pooler",
            "embedding_dim": 2048,
        },
        meta_path,
    )
    api.joblib.dump(["car", "boat"], labelmap_path)

    class FakeDinoModel:
        config = types.SimpleNamespace(hidden_size=1024)

    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(api, "dinov3_model", None)
    monkeypatch.setattr(api, "dinov3_processor", None)
    monkeypatch.setattr(api, "dinov3_initialized", False)
    monkeypatch.setattr(api, "_load_dinov3_backbone", lambda *args, **kwargs: (FakeDinoModel(), object()))

    payload = api.set_active_model(
        api.ActiveModelRequest(
            classifier_path=str(classifier_path),
            labelmap_path=str(labelmap_path),
        )
    )

    assert payload["encoder_type"] == "dinov3"
    assert payload["encoder_ready"] is True
    assert api.active_classifier_head["embedding_dim"] == 2048
    assert api.active_classifier_head["embedding_view_mode"] == "tight_context"


def test_set_active_model_accepts_cradio_mlx_without_processor(tmp_path, monkeypatch):
    classifiers_root = tmp_path / "classifiers"
    labelmaps_root = tmp_path / "labelmaps"
    classifiers_root.mkdir()
    labelmaps_root.mkdir()
    classifier_path = classifiers_root / "cradio_mlx.pkl"
    meta_path = classifiers_root / "cradio_mlx.meta.pkl"
    labelmap_path = labelmaps_root / "labels.pkl"
    classifier = types.SimpleNamespace(
        classes_=np.asarray(["car", "boat"], dtype=object),
        coef_=np.zeros((2, 16), dtype=np.float32),
        intercept_=np.zeros(2, dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    api.joblib.dump(classifier, classifier_path)
    api.joblib.dump(
        {
            "encoder_type": "cradio",
            "encoder_model": CRADIO_DEFAULT_MODEL,
            "cradio_pooling": "summary_spatial_concat",
            "embedding_dim": 16,
        },
        meta_path,
    )
    api.joblib.dump(["car", "boat"], labelmap_path)

    fake_model = types.SimpleNamespace(output_dim=8)
    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(api, "cradio_model", None)
    monkeypatch.setattr(api, "cradio_processor", None)
    monkeypatch.setattr(api, "cradio_model_name", None)
    monkeypatch.setattr(api, "cradio_model_device", None)
    monkeypatch.setattr(api, "cradio_initialized", False)
    monkeypatch.setattr(api, "resolve_cradio_torch_device", lambda **_kwargs: "mlx")
    monkeypatch.setattr(
        api,
        "_load_cradio_backbone_cached",
        lambda model_name, target_device, raise_on_error=False: (fake_model, None, model_name, "mlx"),
    )

    payload = api.set_active_model(
        api.ActiveModelRequest(
            classifier_path=str(classifier_path),
            labelmap_path=str(labelmap_path),
        )
    )

    assert payload["encoder_type"] == "cradio"
    assert payload["encoder_ready"] is True
    assert api.cradio_model is fake_model
    assert api.cradio_processor is None
    assert api.cradio_model_device == "mlx"
    assert api.active_classifier_head["embedding_dim"] == 16
    assert api.active_classifier_head["cradio_pooling"] == "summary_spatial_concat"


def test_set_active_model_rejects_cradio_embedding_width_mismatch(tmp_path, monkeypatch):
    classifiers_root = tmp_path / "classifiers"
    labelmaps_root = tmp_path / "labelmaps"
    classifiers_root.mkdir()
    labelmaps_root.mkdir()
    classifier_path = classifiers_root / "cradio_bad_width.pkl"
    meta_path = classifiers_root / "cradio_bad_width.meta.pkl"
    labelmap_path = labelmaps_root / "labels.pkl"
    classifier = types.SimpleNamespace(
        classes_=np.asarray(["car", "boat"], dtype=object),
        coef_=np.zeros((2, 15), dtype=np.float32),
        intercept_=np.zeros(2, dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    api.joblib.dump(classifier, classifier_path)
    api.joblib.dump(
        {
            "encoder_type": "cradio",
            "encoder_model": CRADIO_DEFAULT_MODEL,
            "cradio_pooling": "summary_spatial_concat",
            "embedding_dim": 15,
        },
        meta_path,
    )
    api.joblib.dump(["car", "boat"], labelmap_path)

    fake_model = types.SimpleNamespace(output_dim=8)
    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(api, "resolve_cradio_torch_device", lambda **_kwargs: "mlx")
    monkeypatch.setattr(
        api,
        "_load_cradio_backbone_cached",
        lambda model_name, target_device, raise_on_error=False: (fake_model, None, model_name, "mlx"),
    )

    with pytest.raises(api.HTTPException) as exc:
        api.set_active_model(
            api.ActiveModelRequest(
                classifier_path=str(classifier_path),
                labelmap_path=str(labelmap_path),
            )
        )

    assert exc.value.detail == "dimension_mismatch:15!=16"


def test_training_multiview_items_compose_consistent_embedding_widths():
    def fake_encode(images):
        return np.asarray(
            [[float(idx + 1), float(idx + 2)] for idx, _image in enumerate(images)],
            dtype=np.float32,
        )

    image = Image.new("RGB", (96, 72), (30, 60, 90))
    positive_views, _positive_crop_xyxy, positive_meta = clip_training._embedding_make_crop_views(
        image,
        (20, 18, 36, 34),
        crop_mode="padded_square",
        padding_ratio=0.08,
        preprocess_mode="canonical",
        canonical_size=64,
        background_mode="blur_outside_box",
        view_mode="tight_context",
    )
    background_views, _background_crop_xyxy, background_meta = clip_training._embedding_make_crop_views(
        image,
        (54, 20, 70, 36),
        crop_mode="padded_square",
        padding_ratio=0.08,
        preprocess_mode="canonical",
        canonical_size=64,
        background_mode="blur_outside_box",
        view_mode="tight_context",
    )
    try:
        positive_item = tuple(positive_views)
        background_item = tuple(background_views)
        augmented_positive = clip_training._apply_augmenter_to_item(None, positive_item)
        augmented_background = clip_training._apply_augmenter_to_item(None, background_item)

        positive_embedding = clip_training._encode_embedding_items(
            [augmented_positive],
            encode_images_fn=fake_encode,
        )
        background_embedding = clip_training._encode_embedding_items(
            [augmented_background],
            encode_images_fn=fake_encode,
        )

        assert len(positive_item) == 2
        assert len(background_item) == 2
        assert positive_embedding.shape == (1, 4)
        assert background_embedding.shape == (1, 4)
        assert positive_embedding.shape[1] == background_embedding.shape[1]
        assert [entry["view"] for entry in positive_meta] == ["tight", "context"]
        assert [entry["view"] for entry in background_meta] == ["tight", "context"]
        assert all(view.size == (64, 64) for view in positive_item)
        assert all(view.size == (64, 64) for view in background_item)
    finally:
        for name in ("augmented_positive", "augmented_background"):
            item = locals().get(name)
            if item is not None:
                clip_training._close_crop_item(item)
        clip_training._close_crop_item(positive_views)
        clip_training._close_crop_item(background_views)
        image.close()
