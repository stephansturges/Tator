from tools.train_policy_layer import _select_policy_candidate


def _candidate(*, f1, recall, fp, sam_only_delta_tp):
    return {
        "metrics": {
            "f1": f1,
            "recall": recall,
            "fp": fp,
        },
        "compare_to_baseline": {
            "subgroups": {
                "sam_only": {
                    "delta_tp": sam_only_delta_tp,
                }
            }
        },
    }


def test_select_policy_candidate_prefers_higher_f1_when_gap_is_material():
    candidates = {
        "lreg": _candidate(f1=0.812, recall=0.750, fp=100, sam_only_delta_tp=0),
        "xgb": _candidate(f1=0.815, recall=0.748, fp=105, sam_only_delta_tp=-1),
    }
    assert _select_policy_candidate(candidates) == "xgb"


def test_select_policy_candidate_prefers_lower_fp_at_near_equal_f1_and_recall():
    candidates = {
        "lreg": _candidate(f1=0.8150, recall=0.7504, fp=95, sam_only_delta_tp=0),
        "xgb": _candidate(f1=0.8158, recall=0.7501, fp=108, sam_only_delta_tp=0),
    }
    assert _select_policy_candidate(candidates) == "lreg"


def test_select_policy_candidate_prefers_smaller_sam_only_tp_loss_before_model_simplicity():
    candidates = {
        "lreg": _candidate(f1=0.8151, recall=0.7500, fp=100, sam_only_delta_tp=-3),
        "xgb": _candidate(f1=0.8152, recall=0.7500, fp=100, sam_only_delta_tp=-1),
    }
    assert _select_policy_candidate(candidates) == "xgb"


def test_select_policy_candidate_prefers_lreg_on_true_tie():
    candidates = {
        "lreg": _candidate(f1=0.8151, recall=0.7500, fp=100, sam_only_delta_tp=-1),
        "xgb": _candidate(f1=0.8152, recall=0.7500, fp=100, sam_only_delta_tp=-1),
    }
    assert _select_policy_candidate(candidates) == "lreg"
