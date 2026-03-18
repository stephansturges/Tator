import json
import sys
from pathlib import Path

import numpy as np

from tools import build_feature_lanes_from_prepass as lane_builder
from tools.build_feature_lanes_from_prepass import _lane_config, _lane_id


def test_lane_id_is_now_just_the_variant_name():
    assert _lane_id("window") == "window"
    assert _lane_id("nonwindow") == "nonwindow"


def test_lane_config_is_now_just_the_base_key():
    assert _lane_config("cachekey") == "cachekey"


def test_lane_builder_respects_window_only_lane_selection(
    monkeypatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    prepass_jsonl = tmp_path / "window.jsonl"
    prepass_jsonl.write_text('{"image":"img1.jpg"}\n', encoding="utf-8")

    features_path = tmp_path / "window_features.npz"
    labeled_path = tmp_path / "window_labeled.npz"
    meta = np.array([json.dumps({"image": "img1.jpg"})], dtype=object)
    X = np.zeros((1, 2), dtype=np.float32)
    y = np.array([1], dtype=np.int64)
    np.savez_compressed(features_path, meta=meta, X=X, y=y)
    np.savez_compressed(labeled_path, meta=meta, X=X, y=y)

    views_dir = run_root / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    (views_dir / "window_intersection.jsonl").write_text(
        '{"image":"img1.jpg"}\n', encoding="utf-8"
    )

    monkeypatch.setattr(
        lane_builder,
        "_run",
        lambda cmd: (_ for _ in ()).throw(AssertionError(f"unexpected subprocess: {cmd}")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_feature_lanes_from_prepass.py",
            "--dataset",
            "qwen_dataset",
            "--run-root",
            str(run_root),
            "--nonwindow-key",
            "nonwindow_cache_key",
            "--window-key",
            "window_cache_key",
            "--classifier-id",
            "classifier.pkl",
            "--lane-selection",
            "window",
            "--window-prepass-jsonl",
            str(prepass_jsonl),
            "--window-features",
            str(features_path),
            "--window-labeled",
            str(labeled_path),
        ],
    )

    lane_builder.main()

    manifest = json.loads((run_root / "lane_manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["lanes"].keys()) == ["window"]
    assert manifest["views"]["full"]["nonwindow_images"] == 0
    assert manifest["views"]["full"]["window_images"] == 1
    assert manifest["intersection_prepass_jsonl"].keys() == {"window"}
