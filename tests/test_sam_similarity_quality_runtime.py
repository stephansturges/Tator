import numpy as np

from tools import policy_runtime


class _FakeDMatrix:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    def num_row(self):
        return int(self.data.shape[0])


class _FakeBaseBooster:
    def load_model(self, path):
        return None

    def predict(self, dmatrix):
        return np.asarray(dmatrix.data.mean(axis=1), dtype=np.float32)


class _FakeQualityBooster:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, dmatrix):
        return np.full(dmatrix.num_row(), self.value, dtype=np.float32)


def test_similarity_quality_blend_only_hits_true_similarity_only_rows(tmp_path, monkeypatch):
    model_path = tmp_path / "base.json"
    model_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(policy_runtime.xgb, "Booster", _FakeBaseBooster)
    monkeypatch.setattr(policy_runtime.xgb, "DMatrix", _FakeDMatrix)
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
            return _FakeQualityBooster(0.05)
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
