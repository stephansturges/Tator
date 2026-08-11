import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def _write_toy_xgb_model(train_path: Path, model_path: Path) -> None:
    code = """
import sys
import numpy as np
import xgboost as xgb

payload = np.load(sys.argv[1])
X = payload["X"]
y = payload["y"]
booster = xgb.train(
    {
        "objective": "binary:logistic",
        "max_depth": 2,
        "eta": 0.5,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "nthread": 1,
    },
    xgb.DMatrix(X, label=y),
    num_boost_round=10,
)
booster.save_model(sys.argv[2])
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(train_path), str(model_path)],
        env=_single_thread_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if "No module named 'xgboost'" in result.stderr:
            pytest.skip("xgboost is not installed")
        raise AssertionError(f"toy XGBoost training failed\n{result.stdout}\n{result.stderr}".strip())


def _write_image(path: Path, size=(100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _write_label(path: Path, class_id: int, cx: float, cy: float, bw: float, bh: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{class_id} {cx} {cy} {bw} {bh}\n", encoding="utf-8")


def test_eval_ensemble_xgb_dedupe_writes_analysis_sidecar(tmp_path: Path):
    dataset_id = "toy_dataset"
    yolo_root = tmp_path / "uploads" / "clip_dataset_uploads" / f"{dataset_id}_yolo"
    dataset_root = tmp_path / "uploads" / "qwen_runs" / "datasets" / dataset_id / "train"
    yolo_root.mkdir(parents=True, exist_ok=True)
    (yolo_root / "labelmap.txt").write_text("person\ntruck\n", encoding="utf-8")
    _write_image(dataset_root / "img0.jpg")
    _write_image(dataset_root / "img1.jpg")
    _write_label(yolo_root / "labels" / "train" / "img0.txt", 0, 0.25, 0.25, 0.30, 0.30)
    _write_label(yolo_root / "labels" / "train" / "img1.txt", 1, 0.70, 0.70, 0.40, 0.40)

    feature_names = np.asarray(["feat_score"], dtype=object)
    X = np.asarray([[1.0], [0.0], [1.0], [0.0]], dtype=np.float32)
    y = np.asarray([1, 0, 1, 0], dtype=np.int64)
    meta_rows = np.asarray(
        [
            json.dumps({"image": "img0.jpg", "label": "person", "bbox_xyxy_px": [10, 10, 40, 40], "score": 0.9, "score_source": "yolo", "source_list": ["yolo"]}),
            json.dumps({"image": "img0.jpg", "label": "person", "bbox_xyxy_px": [60, 10, 90, 40], "score": 0.2, "score_source": "sam3_similarity", "source_list": ["sam3_similarity"]}),
            json.dumps({"image": "img1.jpg", "label": "truck", "bbox_xyxy_px": [50, 50, 90, 90], "score": 0.85, "score_source": "rfdetr", "source_list": ["rfdetr"]}),
            json.dumps({"image": "img1.jpg", "label": "truck", "bbox_xyxy_px": [5, 55, 25, 75], "score": 0.25, "score_source": "sam3_text", "source_list": ["sam3_text"]}),
        ],
        dtype=object,
    )
    npz_path = tmp_path / "labeled.npz"
    np.savez(npz_path, X=X, y=y, meta=meta_rows, feature_names=feature_names)

    model_path = tmp_path / "model.json"
    train_path = tmp_path / "xgb_train.npz"
    np.savez(train_path, X=X, y=y)
    _write_toy_xgb_model(train_path, model_path)
    meta_path = tmp_path / "model.meta.json"
    meta_path.write_text(json.dumps({"calibrated_threshold": 0.5}, indent=2), encoding="utf-8")
    analysis_path = tmp_path / "analysis.json"

    script = Path(__file__).resolve().parents[1] / "tools" / "eval_ensemble_xgb_dedupe.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(model_path),
            "--meta",
            str(meta_path),
            "--data",
            str(npz_path),
            "--dataset",
            dataset_id,
            "--eval-iou",
            "0.5",
            "--eval-iou-grid",
            "0.5",
            "--dedupe-iou",
            "0.75",
            "--scoreless-iou",
            "0.0",
            "--analysis-json",
            str(analysis_path),
        ],
        env=_single_thread_env(),
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    assert payload["f1"] >= 0.6
    assert analysis["overall_metrics"]["f1"] == payload["reference_iou"]["xgb_ensemble"]["f1"]
    assert {row["label"] for row in analysis["per_class"]} == {"person", "truck"}
    assert {row["primary_source"] for row in analysis["per_class_per_source"]} >= {"yolo", "rfdetr"}
    assert "within_0p05" in analysis["boundary_hits"]["buckets"]
    assert analysis["calibration_diagnostics"]["candidate_count"] == 4


def test_eval_ensemble_mlp_dedupe_writes_analysis_sidecar(tmp_path: Path):
    dataset_id = "toy_dataset_mlp"
    yolo_root = tmp_path / "uploads" / "clip_dataset_uploads" / f"{dataset_id}_yolo"
    dataset_root = tmp_path / "uploads" / "qwen_runs" / "datasets" / dataset_id / "train"
    yolo_root.mkdir(parents=True, exist_ok=True)
    (yolo_root / "labelmap.txt").write_text("person\ntruck\n", encoding="utf-8")
    _write_image(dataset_root / "img0.jpg")
    _write_image(dataset_root / "img1.jpg")
    _write_label(yolo_root / "labels" / "train" / "img0.txt", 0, 0.25, 0.25, 0.30, 0.30)
    _write_label(yolo_root / "labels" / "train" / "img1.txt", 1, 0.70, 0.70, 0.40, 0.40)

    feature_names = np.asarray(["feat_score"], dtype=object)
    X = np.asarray([[2.0], [-2.0], [2.0], [-2.0]], dtype=np.float32)
    y = np.asarray([1, 0, 1, 0], dtype=np.int64)
    meta_rows = np.asarray(
        [
            json.dumps({"image": "img0.jpg", "label": "person", "bbox_xyxy_px": [10, 10, 40, 40], "score_source": "yolo", "source_list": ["yolo"]}),
            json.dumps({"image": "img0.jpg", "label": "person", "bbox_xyxy_px": [60, 10, 90, 40], "score_source": "sam3_similarity", "source_list": ["sam3_similarity"]}),
            json.dumps({"image": "img1.jpg", "label": "truck", "bbox_xyxy_px": [50, 50, 90, 90], "score_source": "rfdetr", "source_list": ["rfdetr"]}),
            json.dumps({"image": "img1.jpg", "label": "truck", "bbox_xyxy_px": [5, 55, 25, 75], "score_source": "sam3_text", "source_list": ["sam3_text"]}),
        ],
        dtype=object,
    )
    npz_path = tmp_path / "labeled_mlp.npz"
    np.savez(npz_path, X=X, y=y, meta=meta_rows, feature_names=feature_names)

    model_path = tmp_path / "model.pt"
    payload = {
        "input_dim": 1,
        "hidden": [],
        "dropout": 0.0,
        "state_dict": {"0.weight": torch.tensor([[3.0]]), "0.bias": torch.tensor([0.0])},
    }
    torch.save(payload, model_path)
    meta_path = tmp_path / "model.meta.json"
    meta_path.write_text(json.dumps({"calibrated_threshold": 0.5}, indent=2), encoding="utf-8")
    analysis_path = tmp_path / "analysis_mlp.json"

    script = Path(__file__).resolve().parents[1] / "tools" / "eval_ensemble_mlp_dedupe.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(model_path),
            "--meta",
            str(meta_path),
            "--data",
            str(npz_path),
            "--dataset",
            dataset_id,
            "--eval-iou",
            "0.5",
            "--eval-iou-grid",
            "0.5",
            "--dedupe-iou",
            "0.75",
            "--scoreless-iou",
            "0.0",
            "--analysis-json",
            str(analysis_path),
        ],
        env=_single_thread_env(),
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    assert payload["f1"] >= 0.6
    assert analysis["overall_metrics"]["f1"] == pytest.approx(payload["f1"])
    assert {row["label"] for row in analysis["per_class"]} == {"person", "truck"}
    assert {row["primary_source"] for row in analysis["per_class_per_source"]} >= {"yolo", "rfdetr"}
    assert "within_0p05" in analysis["boundary_hits"]["buckets"]
    assert analysis["calibration_diagnostics"]["candidate_count"] == 4
