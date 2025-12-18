import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import (  # noqa: E402
    AgentMiningRequest,
    _generate_steps_global_mutations,
    _stable_sample_ids,
)


def test_stable_sample_ids_deterministic_and_capped():
    ids = list(range(1, 101))
    a = _stable_sample_ids(ids, cap=10, seed=42, salt="x")
    b = _stable_sample_ids(ids, cap=10, seed=42, salt="x")
    c = _stable_sample_ids(ids, cap=10, seed=43, salt="x")
    assert a == b
    assert len(a) == 10
    assert a != c


def test_generate_steps_global_mutations_deterministic_and_unique_prompts():
    payload = AgentMiningRequest(
        dataset_id="ds",
        seed_threshold=0.05,
        expand_threshold=0.3,
        steps_max_steps_per_recipe=3,
        steps_max_visual_seeds_per_step=5,
        max_results=1000,
    )
    seed_stats = [
        {
            "prompt": "car",
            "matches": 10,
            "fps": 2,
            "precision": 0.83,
            "seed_threshold_recommended": 0.05,
            "seed_threshold_curve": [
                {"threshold": 0.0, "matches": 12, "fps": 10, "precision": 0.55},
                {"threshold": 0.05, "matches": 10, "fps": 2, "precision": 0.83},
                {"threshold": 0.1, "matches": 8, "fps": 1, "precision": 0.89},
            ],
        },
        {
            "prompt": "automobile",
            "matches": 8,
            "fps": 1,
            "precision": 0.89,
            "seed_threshold_recommended": 0.1,
            "seed_threshold_curve": [
                {"threshold": 0.02, "matches": 9, "fps": 4, "precision": 0.69},
                {"threshold": 0.1, "matches": 8, "fps": 1, "precision": 0.89},
            ],
        },
        {
            "prompt": "truck",
            "matches": 3,
            "fps": 0,
            "precision": 1.0,
            "seed_threshold_recommended": 0.2,
            "seed_threshold_curve": [
                {"threshold": 0.05, "matches": 3, "fps": 1, "precision": 0.75},
                {"threshold": 0.2, "matches": 3, "fps": 0, "precision": 1.0},
            ],
        },
    ]
    base = {
        "steps": [
            {
                "enabled": True,
                "prompt": "car",
                "seed_threshold": 0.05,
                "expand_threshold": 0.3,
                "max_visual_seeds": 5,
                "seed_dedupe_iou": 0.9,
                "dedupe_iou": 0.5,
                "max_results": 1000,
            },
            {
                "enabled": True,
                "prompt": "automobile",
                "seed_threshold": 0.1,
                "expand_threshold": 0.3,
                "max_visual_seeds": 5,
                "seed_dedupe_iou": 0.9,
                "dedupe_iou": 0.5,
                "max_results": 1000,
            },
        ]
    }

    muts1 = _generate_steps_global_mutations(
        base_candidate=base,
        seed_stats=seed_stats,
        payload=payload,
        max_mutations=25,
        target_precision=0.9,
        enable_max_results=True,
        enable_ordering=True,
    )
    muts2 = _generate_steps_global_mutations(
        base_candidate=base,
        seed_stats=seed_stats,
        payload=payload,
        max_mutations=25,
        target_precision=0.9,
        enable_max_results=True,
        enable_ordering=True,
    )
    assert [m.get("sig") for m in muts1] == [m.get("sig") for m in muts2]
    assert len(muts1) <= 25

    for cand in muts1:
        steps = cand.get("steps") or []
        prompts = [s.get("prompt") for s in steps]
        assert len(prompts) == len(set([str(p).lower() for p in prompts]))
        for s in steps:
            assert 0.0 <= float(s.get("seed_threshold")) <= 1.0
            assert 0.0 <= float(s.get("expand_threshold")) <= 1.0
            assert 0 <= int(s.get("max_visual_seeds")) <= 500
            assert 0.0 <= float(s.get("seed_dedupe_iou")) <= 1.0
            assert 0.0 <= float(s.get("dedupe_iou")) <= 1.0
            assert 1 <= int(s.get("max_results")) <= 5000

