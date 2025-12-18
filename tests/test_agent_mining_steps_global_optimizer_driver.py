import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import _run_steps_global_successive_halving_rounds  # noqa: E402


def test_global_optimizer_driver_improves_across_rounds():
    base = {"id": 0}

    def mutate(best, round_idx):
        # Deterministically generate candidates better than current best.
        start = int(best.get("id", 0))
        return [{"id": start + 1}, {"id": start + 2}, {"id": start + 3}]

    def evaluator(cand, budget):
        cid = int(cand.get("id", 0))
        return (cid, int(budget)), {"id": cid, "budget": int(budget)}

    best, history = _run_steps_global_successive_halving_rounds(
        base_candidate=base,
        budgets=[2, 4],
        keep_ratio=0.5,
        rounds=3,
        max_trials=4,
        mutate=mutate,
        evaluator=evaluator,
    )
    assert int(best.get("id", -1)) == 9
    assert len(history) == 3

