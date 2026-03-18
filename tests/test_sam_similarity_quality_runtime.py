import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from tools import policy_runtime


class _FakeBooster:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, dmatrix):
        return np.full(dmatrix.num_row(), self.value, dtype=np.float32)


def _train_base_model(path: Path) -> None:
    X = np.asarray(
        [
            [0.1, 0.2],
            [0.2, 0.1],
            [0.8, 0.9],
            [0.9, 0.8],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 0, 1, 1], dtype=np.float32)
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 2,
            "eta": 0.3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "seed": 7,
        },
        xgb.DMatrix(X, label=y),
        num_boost_round=8,
        verbose_eval=False,
    )
    booster.save_model(str(path))


def test_similarity_quality_blend_only_hits_true_similarity_only_rows(tmp_path, monkeypatch):
    model_path = tmp_path / "base.json"
    _train_base_model(model_path)
    X = np.asarray(
        [
            [0.85, 0.85],
            [0.85, 0.85],
            [0.15, 0.15],
        ],
        dtype=np.float32,
    )
    meta_rows = [
        {"score_source": "sam3_similarity", "source_list": ["sam3_similarity"]},
        {"score_source": "sam3_similarity", "source_list": ["sam3_similarity", "yolo"]},
        {"score_source": "yolo", "source_list": ["yolo"]},
    ]

    baseline = policy_runtime.predict_base_probabilities(
        X,
        meta_rows=meta_rows,
        model_path=model_path,
        meta={},
    )

    def _fake_loader(path):
        if path and path.name == "sim_quality.json":
            return _FakeBooster(0.05)
        return None

    monkeypatch.setattr(policy_runtime, "_load_optional_booster", _fake_loader)
    sim_quality_path = tmp_path / "sim_quality.json"
    sim_quality_path.write_text("{}", encoding="utf-8")
    meta = {
        "sam3_similarity_quality": {
            "enabled": True,
            "model_path": str(sim_quality_path),
            "alpha": 0.8,
        }
    }
    blended = policy_runtime.predict_base_probabilities(
        X,
        meta_rows=meta_rows,
        model_path=model_path,
        meta=meta,
    )

    assert blended.shape == baseline.shape
    assert blended[0] < baseline[0]
    assert np.isclose(blended[1], baseline[1])
    assert np.isclose(blended[2], baseline[2])
