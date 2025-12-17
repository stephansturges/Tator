import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from localinferenceapi import _build_prompt_recipe, _load_clip_head_from_classifier, _sample_images_for_category


def test_sample_images_for_category_is_deterministic():
    img_ids = list(range(1, 21))
    first = _sample_images_for_category(3, img_ids, sample_size=5, seed=123)
    second = _sample_images_for_category(3, img_ids, sample_size=5, seed=123)
    assert first == second
    # Changing the seed should change the order/pick
    third = _sample_images_for_category(3, img_ids, sample_size=5, seed=124)
    assert first != third


def test_build_prompt_recipe_uses_early_stop_and_counts_fps():
    # Three GT boxes across two images, third image is negative.
    gt_index = {
        1: [("1:0", (0, 0, 1, 1)), ("1:1", (0, 0, 1, 1))],
        2: [("2:0", (0, 0, 1, 1))],
    }
    all_gt_keys = {"1:0", "1:1", "2:0"}
    per_image_gt = {1: 2, 2: 1, 3: 0}
    images = {
        1: {"file_name": "img1"},
        2: {"file_name": "img2"},
        3: {"file_name": "img3"},
    }
    image_ids = [1, 2, 3]

    # Prompt p1 covers all GTs on image 1, none on image 2, and has FPs on negative image 3.
    cand_p1 = {
        "prompt": "p1",
        "threshold": 0.2,
        "matched_gt_keys": {"1:0", "1:1"},
        "matches_by_image": {
            1: {"matched": ["1:0", "1:1"], "fps": 1},
            3: {"matched": [], "fps": 2},
        },
        "fps": 3,
        "precision": 0.66,
        "recall": 0.66,
        "det_rate": 1.0,
    }
    # Alternative threshold for p1 should be ignored (worse coverage).
    cand_p1_alt = {
        **cand_p1,
        "threshold": 0.3,
        "matched_gt_keys": {"1:0"},
        "matches_by_image": {1: {"matched": ["1:0"], "fps": 0}},
        "fps": 0,
    }
    # Prompt p2 hits remaining GT on image 2 (and duplicates one on image 1) with some FPs.
    cand_p2 = {
        "prompt": "p2",
        "threshold": 0.2,
        "matched_gt_keys": {"1:1", "2:0"},
        "matches_by_image": {
            1: {"matched": ["1:1"], "fps": 0},
            2: {"matched": ["2:0"], "fps": 1},
        },
        "fps": 1,
        "precision": 0.5,
        "recall": 0.66,
        "det_rate": 1.0,
    }
    recipe, coverage_by_image = _build_prompt_recipe(
        [cand_p1, cand_p1_alt, cand_p2],
        all_gt_keys,
        per_image_gt,
        images,
        image_ids,
        gt_index,
    )

    steps = recipe["steps"]
    summary = recipe["summary"]
    assert len(steps) == 2, "Both prompts should be kept (best threshold per prompt)."
    # Tie-break prefers fewer FPs when gain is equal, so p2 goes first, then p1.
    assert steps[0]["prompt"] == "p2" and math.isclose(steps[0]["threshold"], 0.2)
    assert steps[1]["prompt"] == "p1" and math.isclose(steps[1]["threshold"], 0.2)
    # Early stop: step 1 gains 2 GTs, step 2 only the remaining 1 (duplicate counted separately).
    assert steps[0]["gain"] == 2
    assert steps[1]["gain"] == 1
    assert steps[1]["duplicates"] == 1  # duplicate hit on image 1
    # FPs accumulate across negatives and positives.
    assert summary["fps"] == 4  # 3 from step1 (1 pos + 2 neg) + 1 from step2
    assert math.isclose(summary["coverage_rate"], 1.0)
    # Coverage map should include pos/neg labeling.
    kinds = {entry["image_id"]: entry["type"] for entry in coverage_by_image}
    assert kinds == {1: "pos", 2: "pos", 3: "neg"}


def test_load_clip_head_from_classifier_falls_back_to_multi_class_mode(tmp_path):
    # When meta.pkl is missing, we should still infer ovr vs softmax correctly.
    dummy = SimpleNamespace(
        classes_=["light_vehicle", "person", "building"],
        coef_=np.zeros((3, 4), dtype=np.float32),
        intercept_=np.zeros((3,), dtype=np.float32),
        solver="lbfgs",
        multi_class="ovr",
    )
    path = tmp_path / "head.pkl"
    joblib.dump(dummy, path)
    head = _load_clip_head_from_classifier(path)
    assert head is not None
    assert head["proba_mode"] == "ovr"
