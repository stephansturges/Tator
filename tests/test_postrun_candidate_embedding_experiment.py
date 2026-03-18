from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools import run_postrun_candidate_embedding_experiment as embexp


def _write_labeled(path: Path) -> None:
    feature_names = np.asarray(
        ["clf_prob::a", "clf_prob::b", "clf_emb_rp::000", "clf_emb_rp::001", "other::x"],
        dtype=object,
    )
    meta = np.asarray(
        [
            json.dumps({"image": "img1.jpg", "label": "a", "bbox_xyxy_px": [0, 0, 10, 10]}),
            json.dumps({"image": "img1.jpg", "label": "a", "bbox_xyxy_px": [10, 0, 20, 10]}),
            json.dumps({"image": "img2.jpg", "label": "b", "bbox_xyxy_px": [0, 0, 10, 10]}),
            json.dumps({"image": "img2.jpg", "label": "b", "bbox_xyxy_px": [10, 0, 20, 10]}),
        ],
        dtype=object,
    )
    np.savez(
        path,
        X=np.asarray(
            [
                [0.9, 0.1, 0.01, 0.02, 1.0],
                [0.8, 0.2, 0.03, 0.04, 2.0],
                [0.1, 0.9, 0.05, 0.06, 3.0],
                [0.2, 0.8, 0.07, 0.08, 4.0],
            ],
            dtype=np.float32,
        ),
        y=np.asarray([1, 1, 0, 0], dtype=np.int64),
        meta=meta,
        feature_names=feature_names,
        labelmap=np.asarray(["a", "b"], dtype=object),
        classifier_classes=np.asarray(["a", "b"], dtype=object),
    )


def test_replace_candidate_embedding_block_raw_plus_proto(tmp_path: Path) -> None:
    base_npz = tmp_path / "base.npz"
    _write_labeled(base_npz)
    raw_embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
        ],
        dtype=np.float32,
    )
    train_mask = np.asarray([True, True, True, False], dtype=bool)
    out_npz = tmp_path / "variant.npz"
    info = embexp._replace_candidate_embedding_block(
        base_npz=base_npz,
        output_npz=out_npz,
        raw_embeddings=raw_embeddings,
        train_mask=train_mask,
        variant="raw_l2_native_plus_prototype_margin",
        pca_fit_rows=100,
        seed=42,
    )
    assert info["variant"] == "raw_l2_native_plus_prototype_margin"
    data = np.load(out_npz, allow_pickle=True)
    names = [str(x) for x in data["feature_names"]]
    assert "clf_emb_rp::000" not in names
    assert "clf_proto_margin" in names
    assert any(name.startswith("clf_emb_raw::") for name in names)
    X = np.asarray(data["X"], dtype=np.float32)
    # keep 3 non-embedding cols + 3 raw dims + 5 prototype dims
    assert X.shape == (4, 11)


def test_replace_candidate_embedding_block_pca(tmp_path: Path) -> None:
    base_npz = tmp_path / "base.npz"
    _write_labeled(base_npz)
    raw_embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.8, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    out_npz = tmp_path / "pca.npz"
    embexp._replace_candidate_embedding_block(
        base_npz=base_npz,
        output_npz=out_npz,
        raw_embeddings=raw_embeddings,
        train_mask=np.asarray([True, True, True, False], dtype=bool),
        variant="pca_256",
        pca_fit_rows=100,
        seed=42,
    )
    data = np.load(out_npz, allow_pickle=True)
    names = [str(x) for x in data["feature_names"]]
    assert any(name.startswith("clf_emb_pca256::") for name in names)
    X = np.asarray(data["X"], dtype=np.float32)
    # keep 3 non-embedding cols + padded 256 dims
    assert X.shape == (4, 259)


def test_pilot_pass_thresholds() -> None:
    ctx = embexp.ContextSpec(
        lane="window",
        view="intersection",
        seed=42,
        variant="window",
        labeled_npz=Path("dummy"),
        prepass_jsonl=Path("dummy"),
        val_images_file=Path("dummy"),
        baseline_f1=0.80,
        baseline_precision=0.9,
        baseline_recall=0.72,
        baseline_delta_vs_union_f1=0.03,
        baseline_coverage_preservation=0.80,
    )
    assert embexp._pilot_pass(
        {"f1": 0.803, "coverage_preservation": 0.796},
        ctx,
    )
    assert not embexp._pilot_pass(
        {"f1": 0.801, "coverage_preservation": 0.80},
        ctx,
    )
