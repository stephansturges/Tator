import json
import types

import numpy as np
from PIL import Image

import localinferenceapi as api
from services.classifier import _load_clip_head_from_classifier_impl
from tools import clip_training


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


def test_class_analysis_capabilities_expose_only_normal_recipe_controls():
    caps = api._class_analysis_capabilities()

    assert caps["preprocess_modes"] == ["canonical"]
    assert caps["embedding_adjustments"] == ["remove_size_bias"]
    assert caps["expert_preprocess_modes"] == ["native", "canonical"]
    assert caps["expert_embedding_adjustments"] == ["none", "remove_size_bias"]
    assert caps["default_preprocess_mode"] == "canonical"
    assert caps["default_embedding_adjustment"] == "remove_size_bias"


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
        captured["image_size"] = images[0].size
        captured["geometry"] = geometry_records[0]
        captured["head"] = head
        return np.ones((1, 4), dtype=np.float32)

    monkeypatch.setattr(api, "_active_classifier_head_for_inference", lambda: head)
    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)

    image = Image.new("RGB", (100, 60), (20, 40, 60))
    feats = api._encode_classifier_xyxy_for_active(image, [40, 20, 60, 30])

    assert feats.shape == (1, 4)
    assert captured["image_size"] == (80, 80)
    assert captured["geometry"]["bbox_xyxy"] == [40.0, 20.0, 60.0, 30.0]
    assert captured["geometry"]["crop_xyxy"] == [30, 5, 70, 45]
    assert captured["geometry"]["background_mode"] == "darken_outside_box"
    assert captured["geometry"]["embedding_view_mode"] == "single"
    assert captured["head"]["embedding_adjustment_transform"]["mode"] == "remove_size_bias"


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
