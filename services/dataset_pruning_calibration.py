"""Recipe-bound diagnostic gates fitted only on reviewed calibration directions.

The caller supplies freshly matched evidence for each object floor. This module
never reuses a different floor's assignments, labels evaluation examples, or
turns a calibration judgment into permission to delete an image.
"""
from __future__ import annotations

import math
from collections import Counter

from services.dataset_pruning import digest

LABELS = {"removable_under_policy", "useful_variation_keep", "wrong_correspondence_keep", "uncertain"}
KEEP_LABELS = {"useful_variation_keep", "wrong_correspondence_keep"}
OBJECT_FLOORS = (.75, .80, .85, .87, .90, .93)
SCENE_FLOORS = (.65, .75, .85, .90, .92, .95)
INDEX_STEP = .001
SCHEMA = "pruning-directional-diagnostic-calibration-v1"


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid {name}; expected a finite number between 0 and 1")
    return float(value)


def normalize_labels(labels):
    if not isinstance(labels, list):
        raise ValueError("Calibration labels must be a list of reviewed directions")
    result = {}
    for item in labels:
        if not isinstance(item, dict):
            raise ValueError("Each calibration label must describe a reviewed direction")
        case_id, label = item.get("case_id"), item.get("label")
        if not isinstance(case_id, str) or not case_id or case_id in result or label not in LABELS:
            raise ValueError("Calibration labels need unique directional case IDs and a supported judgment")
        result[case_id] = {"case_id": case_id, "label": label, "reason": str(item.get("reason") or "")[:2000]}
    if not result or len(result) > 128:
        raise ValueError("Review between 1 and 128 calibration directions")
    return result


def fit_thresholds(binding, labels, measurements, *, scene_floors=SCENE_FLOORS,
                   min_removable=3, min_keep=3, review_authority="human",
                   cohort_scope="Selected calibration examples"):
    """Choose separate object/scene/index gates for this exact representation.

    A measurement is {case_id, object_threshold, scene, score, eligible}. The
    boolean eligible must come from shared matching at that exact object floor,
    with scene/index gates opened. Index candidates lie on a declared .001 grid;
    boundaries come from reviewed scores rather than numerical-ULP tuning.
    Zero observed keep errors is an empirical fitting rule for this small set,
    not a claim of population precision or a validated default.
    """
    if not isinstance(binding, dict) or not binding:
        raise ValueError("Calibration must bind an exact feature recipe and source")
    judgments = normalize_labels(labels)
    counts = Counter(item["label"] for item in judgments.values())
    supported = {key: value for key, value in judgments.items() if value["label"] != "uncertain"}
    floors = {}
    for record in measurements:
        case_id = record.get("case_id")
        if case_id not in judgments:
            raise ValueError("A calibration measurement is outside the frozen review cohort")
        floor = _number(record.get("object_threshold"), "object threshold")
        if type(record.get("eligible")) is not bool:
            raise ValueError("Calibration eligibility must come from a measured shared assignment")
        group = floors.setdefault(floor, {})
        if case_id in group:
            raise ValueError("A direction was measured twice at one object floor")
        group[case_id] = {"scene": _number(record.get("scene"), "scene score"),
                          "score": _number(record.get("score"), "pruning index"),
                          "eligible": record["eligible"]}
    if not floors or any(set(group) != set(judgments) for group in floors.values()):
        raise ValueError("Every frozen calibration direction needs fresh evidence at every object floor")
    scene_candidates = sorted({_number(value, "scene threshold") for value in scene_floors})
    if not scene_candidates:
        raise ValueError("The calibration scene grid is empty")
    common = {"schema": SCHEMA, "feature_binding": binding, "feature_recipe_digest": digest(binding),
              "review_authority": review_authority, "cohort_scope": cohort_scope,
              "reviewed_direction_count": len(judgments), "label_counts": dict(counts),
              "review_labels_digest": digest([judgments[key] for key in sorted(judgments)]),
              "uncertain_excluded_from_fit": counts["uncertain"],
              "calibration_policy": {"false_keep_allowance": 0, "min_removable": min_removable,
                                     "min_keep": min_keep, "object_floor_grid": sorted(floors),
                                     "scene_floor_grid": scene_candidates,
                                     "index_grid": "Calibration score boundaries on a fixed 0.001 grid",
                                     "index_step": INDEX_STEP,
                                     "tie_break": "Maximum removable support, then higher object, scene and index gates"},
              "qualification": "Diagnostic calibration on selected reviewed directions; population precision and detector quality are unmeasured. This is not a removal approval."}
    if counts["removable_under_policy"] < min_removable or sum(counts[label] for label in KEEP_LABELS) < min_keep:
        return {**common, "status": "insufficient_review_support", "thresholds": None,
                "reason": f"Need at least {min_removable} removable and {min_keep} keep/wrong directions; uncertain judgments do not count."}
    best = None
    for object_floor, group in floors.items():
        index_candidates = {0.0, 1.0}
        for case_id in supported:
            score = group[case_id]["score"]
            lower = math.floor(score / INDEX_STEP)
            index_candidates.add(round(lower * INDEX_STEP, 3))
            if lower < 1000:
                index_candidates.add(round((lower + 1) * INDEX_STEP, 3))
        for scene_floor in scene_candidates:
            for index_floor in sorted(index_candidates):
                flagged = [case_id for case_id in supported if group[case_id]["eligible"]
                           and group[case_id]["scene"] >= scene_floor and group[case_id]["score"] >= index_floor]
                false_keep = sum(judgments[case_id]["label"] in KEEP_LABELS for case_id in flagged)
                if false_keep:
                    continue
                found = sum(judgments[case_id]["label"] == "removable_under_policy" for case_id in flagged)
                priority = (found, object_floor, scene_floor, index_floor)
                if best is None or priority > best[0]:
                    best = (priority, flagged, group)
    if best is None or best[0][0] == 0:
        return {**common, "status": "no_supported_cutoffs", "thresholds": None,
                "reason": "No searched cutoffs find a reviewed removable direction without also flagging a reviewed keep direction."}
    (found, object_floor, scene_floor, index_floor), flagged, group = best
    uncertain_flagged = [case_id for case_id, value in judgments.items() if value["label"] == "uncertain"
                         and group[case_id]["eligible"] and group[case_id]["scene"] >= scene_floor
                         and group[case_id]["score"] >= index_floor]
    result = {**common, "status": "diagnostic_calibrated",
              "thresholds": {"object_threshold": object_floor, "scene_threshold": scene_floor,
                             "index_threshold": index_floor},
              "fit_outcome": {"removable_found": found, "removable_total": counts["removable_under_policy"],
                              "keep_flagged": 0, "keep_total": sum(counts[label] for label in KEEP_LABELS),
                              "flagged_case_ids": sorted(flagged), "uncertain_flagged_case_ids": sorted(uncertain_flagged)}}
    result["profile_id"] = "dpc_" + digest(result)[:24]
    return result


def assert_profile_binding(profile, binding, thresholds=None):
    if profile.get("schema") != SCHEMA or profile.get("status") != "diagnostic_calibrated":
        raise ValueError("This recipe has no supported calibration profile")
    if profile.get("feature_recipe_digest") != digest(binding) or profile.get("feature_binding") != binding:
        raise ValueError("Source, fitted features, correction strength or weights changed; calibrate this exact recipe again")
    if thresholds is not None and any(thresholds.get(key) != value for key, value in profile["thresholds"].items()):
        raise ValueError("The selected gates differ from this recipe's saved calibration")
    return True
