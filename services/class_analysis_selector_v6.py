"""V7 expected-review utility selector for Class Analysis.

The selector ranks the complete Stage-1 rough-candidate queue.  It does not
change candidate membership, refinement status, labels, or human-control
semantics.  Its inputs are an explicit allowlist of raw Stage-1, patch,
and same-image geometry features; no earlier selector rank, status band,
adjustment, class identity, directed-pair identity, or human disposition is a
model feature.

The frozen V6 base model uses a small pair-blind histogram-gradient booster to
learn the general visual, patch, count, density, and scale signal.  Dataset-wide
overlap frequency
remains excluded from that learned model.  V7 adds a separate statistical,
rank-only adjustment after the base model: the current class must be
affirmatively present, the alternative evidence must be localized to an exact
material nonduplicate overlap, the candidate capture/source group must be left
out of a reliable fit, and a Wilson lower bound plus support shrinkage must show
that the overlap is recurrent.  The adjustment changes only queue order.  It
does not change actionability probabilities, candidate membership, semantic
status, labels, or human control.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


LEGACY_V6_SELECTOR_PRIORITY_CONTRACT = (
    "expected-review-utility-global-boosted-gated-dataset-overlap-v6"
)
SELECTOR_PRIORITY_CONTRACT = (
    "expected-review-utility-global-boosted-statistical-overlap-rerank-v7"
)
SELECTOR_FEATURE_CONTRACT = (
    "raw-stage1-patch-same-image-and-gated-dataset-overlap-features-v3"
)
SELECTOR_MODEL_SCHEMA = "class-analysis-selector-utility-model-v2"
SELECTOR_UTILITY_POLICY_CONTRACT = (
    "actionability-times-75pct-base-plus-25pct-reviewability-v1"
)
LEGACY_V6_DATASET_OVERLAP_APPLICATION_CONTRACT = (
    "affirmative-current-localized-material-reliable-dataset-overlap-v1"
)
LEGACY_V6_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT = (
    "gated-observable-zero-scoring-effect-v1"
)
DATASET_OVERLAP_APPLICATION_CONTRACT = (
    "affirmative-current-localized-material-source-loo-v2"
)
DATASET_OVERLAP_DIAGNOSTIC_CONTRACT = (
    "capture-loo-beta-wilson-shrunk-rank-only-v1"
)
DATASET_OVERLAP_MAXIMUM_RANK_DISCOUNT_FRACTION = 0.25
GLOBAL_ACTIONABILITY_MODEL_CONTRACT = (
    "pair-blind-shallow-histogram-gradient-boosting-v1"
)
SELECTOR_MODEL_FILENAME = "class_analysis_selector_v6_default.json"

CURRENT_EVIDENCE_STATES = (
    "present",
    "absent",
    "indeterminate",
    "unavailable",
)
OVERLAP_EVIDENCE_STATES = (
    "none",
    "localized",
    "external",
    "uncertain",
    "duplicate_conflict",
)

# The learned heads deliberately exclude all dataset pair-frequency statistics.
# Those values are published only through the gated diagnostic payload.
GLOBAL_NUMERIC_FEATURES = (
    "stage1_suspicion",
    "embedding_suspicion",
    "neighbor_gap",
    "current_support_margin",
    "alternative_support_margin",
    "intrinsic_support_gap",
    "pair_probe_excess",
    "pair_probe_reliability",
    "pair_probe_lower_bound",
    "view_agreement",
    "overlap_count_log",
    "target_log_area",
    "target_log_aspect",
    "annotated_overlap",
    "nested_overlap",
    "external_alternative",
    "localized_alternative",
    "current_spatial_coherence",
    "alternative_spatial_coherence",
    "source_resolution_ok",
    "reference_reliable",
    "diagnostic_pair_reliable",
    "same_scale_outlier",
    "perspective_scale_outlier",
    "aspect_outlier",
    "current_alt_scale_contrast",
    "same_width_residual",
    "same_height_residual",
    "same_area_signed_residual",
    "same_aspect_signed_residual",
    "same_scale_available",
    "perspective_available",
    "scale_contrast_available",
    "image_object_count_log",
    "same_class_count_log",
    "same_class_fraction",
    "same_anchor_count_log",
    "alternative_anchor_count_log",
    "local_density_log",
    "local_same_class_log",
    "max_other_iou",
    "max_target_cover",
    "border_touch",
    "bbox_outside_fraction",
)
GLOBAL_CATEGORICAL_FEATURES = (
    "current_state",
    "alternative_state",
    "overlap_state",
)
GLOBAL_DENSE_FEATURES = (
    *GLOBAL_NUMERIC_FEATURES,
    *(f"current_state={state}" for state in CURRENT_EVIDENCE_STATES),
    *(f"alternative_state={state}" for state in CURRENT_EVIDENCE_STATES),
    *(f"overlap_state={state}" for state in OVERLAP_EVIDENCE_STATES),
)
DATASET_OVERLAP_DIAGNOSTIC_FEATURES = (
    "dataset_overlap_gate",
    "dataset_overlap_incidence",
    "dataset_overlap_lower_bound",
    "dataset_overlap_conservative_strength",
    "dataset_overlap_support_log",
    "dataset_overlap_positive_support_log",
    "dataset_overlap_candidate_count_log",
    "dataset_overlap_candidate_strength",
    "dataset_overlap_scale_typicality",
)
DATASET_OVERLAP_APPLICATION_REASONS = (
    "eligible_dataset_overlap_explanation",
    "current_class_not_affirmatively_present",
    "alternative_class_not_affirmatively_present",
    "duplicate_like_conflict",
    "pair_conflict_preserved",
    "material_annotated_overlap_absent",
    "alternative_not_localized_to_overlap",
    "alternative_evidence_external_to_overlap",
    "same_image_context_unavailable",
    "material_overlap_prior_unavailable",
    "dataset_screening_scope_ineligible",
    "dataset_overlap_prior_unreliable",
    "candidate_capture_group_not_left_out",
    "candidate_not_in_material_overlap_stratum",
)
FORBIDDEN_MODEL_FEATURE_PREFIXES = (
    "selector_priority_",
    "human_review_",
)
FORBIDDEN_MODEL_FEATURE_NAMES = {
    "status",
    "qualified_for_human_review",
    "frequent_overlap_applies",
    "semantic_priority_adjustment",
    "triage_frequency_adjustment",
    "priority_adjustment",
}


def _finite(value: Any, fallback: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return numeric if math.isfinite(numeric) else fallback


def _bounded_probability(value: Any) -> float:
    return max(0.0, min(1.0, _finite(value)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-min(60.0, value))
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(max(-60.0, value))
    return exponent / (1.0 + exponent)


def _logit(probability: float) -> float:
    probability = max(1e-5, min(1.0 - 1e-5, probability))
    return math.log(probability / (1.0 - probability))


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _artifact_payload_without_digest(
    artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in artifact.items()
        if key != "model_digest"
    }


def selector_model_digest(artifact: Mapping[str, Any]) -> str:
    return _canonical_digest(_artifact_payload_without_digest(artifact))


def current_evidence_state(evidence: Mapping[str, Any]) -> str:
    gates = evidence.get("decision_gates")
    gates = gates if isinstance(gates, Mapping) else {}
    if gates.get("current_present") is True:
        return "present"
    if gates.get("current_absent") is True:
        return "absent"
    if gates.get("source_resolution_sufficient") is not True:
        return "unavailable"
    return "indeterminate"


def alternative_evidence_state(evidence: Mapping[str, Any]) -> str:
    gates = evidence.get("decision_gates")
    gates = gates if isinstance(gates, Mapping) else {}
    if gates.get("alternative_present") is True:
        return "present"
    if gates.get("alternative_absent") is True:
        return "absent"
    if gates.get("source_resolution_sufficient") is not True:
        return "unavailable"
    return "indeterminate"


def overlap_evidence_state(evidence: Mapping[str, Any]) -> str:
    gates = evidence.get("decision_gates")
    gates = gates if isinstance(gates, Mapping) else {}
    if str(evidence.get("overlap_relation") or "") == "duplicate_like":
        return "duplicate_conflict"
    if gates.get("annotated_overlap") is not True:
        return "none"
    if gates.get("alternative_evidence_external_to_overlap") is True:
        return "external"
    if gates.get("alternative_evidence_localized_to_overlap") is True:
        return "localized"
    return "uncertain"


def _material_overlap_context(
    evidence: Mapping[str, Any],
    same_image_context: Mapping[str, Any],
    *,
    current_state: str,
    alternative_state: str,
    overlap_state: str,
    scale_typicality: float,
) -> Dict[str, Any]:
    """Build the candidate-level, policy-free dataset-overlap gate.

    The historical prior contains old ranking decisions in addition to raw
    cohort statistics.  V6 reads only the exact material-nonduplicate stratum
    and recomputes applicability from candidate evidence.  In particular,
    ``prior.applies`` and every historical adjustment field are ignored.
    """

    prior = evidence.get("frequent_overlap_prior")
    prior = prior if isinstance(prior, Mapping) else {}
    raw_strata = prior.get("strata")
    material = next(
        (
            item
            for item in (raw_strata or ())
            if isinstance(item, Mapping)
            and str(item.get("geometry_stratum") or "")
            == "material_nonduplicate"
        ),
        None,
    )
    gates = evidence.get("decision_gates")
    gates = gates if isinstance(gates, Mapping) else {}
    available = isinstance(material, Mapping)
    material = material if isinstance(material, Mapping) else {}

    bound_alternative_point_id = str(
        evidence.get("annotated_overlap_alternative_point_id") or ""
    ).strip()
    bound_alternative_class = str(
        evidence.get("alternative_class") or ""
    ).strip()
    exact_material_evidence = any(
        isinstance(item, Mapping)
        and str(item.get("point_id") or "").strip()
        == bound_alternative_point_id
        and str(item.get("class_name") or "").strip()
        == bound_alternative_class
        and str(item.get("geometry_stratum") or "")
        == "material_nonduplicate"
        for item in (prior.get("candidate_overlap_evidence") or ())
    )

    fit_eligible = bool(
        material.get("fit_screening_adjustment_eligible") is True
        and prior.get("fit_screening_adjustment_eligible") is True
    )
    reliable = bool(
        material.get("reliable") is True
        and material.get("source_independence_verified") is True
        and material.get("adjustment_eligible") is True
        and material.get("provisional") is not True
    )
    capture_group_left_out = bool(
        str(prior.get("candidate_capture_group_id") or "").strip()
        and prior.get("candidate_capture_group_excluded") is True
        and prior.get("candidate_source_excluded") is True
    )
    candidate_overlap = bool(
        material.get("candidate_overlap") is True
        and exact_material_evidence
    )
    if current_state != "present" or gates.get("current_absent") is True:
        reason = "current_class_not_affirmatively_present"
    elif (
        alternative_state != "present"
        or gates.get("alternative_absent") is True
    ):
        reason = "alternative_class_not_affirmatively_present"
    elif str(evidence.get("status") or "").strip() == "pair_conflict":
        reason = "pair_conflict_preserved"
    elif (
        overlap_state == "duplicate_conflict"
        or str(evidence.get("overlap_relation") or "") == "duplicate_like"
    ):
        reason = "duplicate_like_conflict"
    elif gates.get("annotated_overlap") is not True:
        reason = "material_annotated_overlap_absent"
    elif gates.get("alternative_evidence_localized_to_overlap") is not True:
        reason = "alternative_not_localized_to_overlap"
    elif gates.get("alternative_evidence_external_to_overlap") is True:
        reason = "alternative_evidence_external_to_overlap"
    elif same_image_context.get("available") is not True:
        reason = "same_image_context_unavailable"
    elif not available:
        reason = "material_overlap_prior_unavailable"
    elif not fit_eligible:
        reason = "dataset_screening_scope_ineligible"
    elif not reliable:
        reason = "dataset_overlap_prior_unreliable"
    elif not capture_group_left_out:
        reason = "candidate_capture_group_not_left_out"
    elif not candidate_overlap:
        reason = "candidate_not_in_material_overlap_stratum"
    else:
        reason = "eligible_dataset_overlap_explanation"
    applicable = reason == "eligible_dataset_overlap_explanation"

    incidence = _bounded_probability(
        material.get("smoothed_capture_group_incidence")
    )
    lower_bound = _bounded_probability(
        material.get("capture_group_incidence_wilson_lower_bound")
        if material.get("capture_group_incidence_wilson_lower_bound")
        is not None
        else material.get("source_incidence_wilson_lower_bound")
    )
    conservative = _bounded_probability(
        material.get("conservative_prior_strength")
    )
    eligible_count = max(
        0, int(_finite(material.get("eligible_capture_group_count")))
    )
    positive_count = max(
        0, int(_finite(material.get("overlap_capture_group_count")))
    )
    candidate_strength = _bounded_probability(
        material.get("candidate_overlap_strength")
    )
    candidate_count = max(
        0, int(_finite(material.get("candidate_overlap_count")))
    )
    model_values = {
        "dataset_overlap_gate": float(applicable),
        "dataset_overlap_incidence": incidence if applicable else 0.0,
        "dataset_overlap_lower_bound": lower_bound if applicable else 0.0,
        "dataset_overlap_conservative_strength": (
            conservative if applicable else 0.0
        ),
        "dataset_overlap_support_log": (
            math.log1p(eligible_count) if applicable else 0.0
        ),
        "dataset_overlap_positive_support_log": (
            math.log1p(positive_count) if applicable else 0.0
        ),
        "dataset_overlap_candidate_count_log": (
            math.log1p(candidate_count) if applicable else 0.0
        ),
        "dataset_overlap_candidate_strength": (
            candidate_strength if applicable else 0.0
        ),
        "dataset_overlap_scale_typicality": (
            _bounded_probability(scale_typicality) if applicable else 0.0
        ),
    }
    return {
        "application_contract": DATASET_OVERLAP_APPLICATION_CONTRACT,
        "available": available,
        "applicable": applicable,
        "application_reason": reason,
        "directed_pair": (
            f"{str(evidence.get('current_class') or '').strip()}->"
            f"{str(evidence.get('alternative_class') or '').strip()}"
        ),
        "geometry_stratum": "material_nonduplicate",
        "affirmative_current_evidence": current_state == "present",
        "current_evidence_state": current_state,
        "alternative_evidence_state": alternative_state,
        "overlap_evidence_state": overlap_state,
        "candidate_overlap": candidate_overlap,
        "candidate_overlap_count": candidate_count,
        "candidate_overlap_strength": candidate_strength,
        "scale_typicality": _bounded_probability(scale_typicality),
        "fit_screening_adjustment_eligible": fit_eligible,
        "reliable": reliable,
        "reliability_tier": str(material.get("reliability_tier") or "none"),
        "source_independence_verified": (
            material.get("source_independence_verified") is True
        ),
        "candidate_capture_group_id": str(
            prior.get("candidate_capture_group_id") or ""
        ).strip(),
        "candidate_capture_group_excluded": (
            prior.get("candidate_capture_group_excluded") is True
        ),
        "candidate_source_excluded": (
            prior.get("candidate_source_excluded") is True
        ),
        "smoothed_capture_group_incidence": incidence,
        "conservative_strength": conservative,
        "capture_group_incidence_lower_bound": lower_bound,
        "eligible_capture_group_count": eligible_count,
        "overlap_capture_group_count": positive_count,
        "model_values": model_values,
    }


def _dataset_overlap_rank_adjustment(
    *,
    actionability: float,
    reviewability: float,
    overlap_context: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Apply the V7 statistical overlap adjustment to ranking utility only.

    The overlap incidence is already Beta-smoothed and capture-group
    leave-one-out in the producer.  We use its Wilson lower bound rather than
    the point estimate alone, then shrink again for finite eligible/positive
    group support.  No human review outcome or directed-pair label is used.
    """

    base_utility = _bounded_probability(actionability) * (
        0.75 + 0.25 * _bounded_probability(reviewability)
    )
    overlap = dict(overlap_context)
    applicable = overlap.get("applicable") is True
    incidence = _bounded_probability(
        overlap.get("smoothed_capture_group_incidence")
    )
    lower_bound = _bounded_probability(
        overlap.get("capture_group_incidence_lower_bound")
    )
    eligible_count = max(
        0, int(_finite(overlap.get("eligible_capture_group_count")))
    )
    positive_count = max(
        0, int(_finite(overlap.get("overlap_capture_group_count")))
    )
    candidate_strength = _bounded_probability(
        overlap.get("candidate_overlap_strength")
    )
    scale_typicality = _bounded_probability(overlap.get("scale_typicality"))
    eligible_shrink = (
        eligible_count / (eligible_count + 20.0) if eligible_count else 0.0
    )
    positive_shrink = (
        positive_count / (positive_count + 5.0) if positive_count else 0.0
    )
    support_shrink = math.sqrt(eligible_shrink * positive_shrink)
    uncertainty_shrunk_frequency = math.sqrt(incidence * lower_bound)
    scale_factor = 0.5 + 0.5 * scale_typicality
    explanation_strength = (
        uncertainty_shrunk_frequency
        * support_shrink
        * candidate_strength
        * scale_factor
        if applicable
        else 0.0
    )
    explanation_strength = _bounded_probability(explanation_strength)
    discount_fraction = (
        DATASET_OVERLAP_MAXIMUM_RANK_DISCOUNT_FRACTION
        * explanation_strength
    )
    final_utility = max(0.0, base_utility * (1.0 - discount_fraction))
    utility_delta = final_utility - base_utility
    applied = applicable and utility_delta < -1e-15
    overlap.update(
        {
            "diagnostic_contract": DATASET_OVERLAP_DIAGNOSTIC_CONTRACT,
            "scoring_effect_enabled": True,
            "rank_only": True,
            "uses_human_review_labels": False,
            "applied": applied,
            "rank_adjustment_reason": (
                "statistical_common_overlap_rank_discount"
                if applied
                else str(overlap.get("application_reason") or "")
            ),
            "counterfactual_actionable_probability": float(actionability),
            "actionable_probability": float(actionability),
            "probability_delta": 0.0,
            "base_expected_review_utility": float(base_utility),
            "counterfactual_expected_review_utility": float(base_utility),
            "expected_review_utility": float(final_utility),
            "utility_delta": float(utility_delta),
            "maximum_rank_discount_fraction": (
                DATASET_OVERLAP_MAXIMUM_RANK_DISCOUNT_FRACTION
            ),
            "rank_discount_fraction": float(discount_fraction),
            "uncertainty_shrunk_frequency": float(
                uncertainty_shrunk_frequency
            ),
            "eligible_support_shrink": float(eligible_shrink),
            "positive_support_shrink": float(positive_shrink),
            "support_shrink": float(support_shrink),
            "scale_typicality_factor": float(scale_factor),
            "statistical_explanation_strength": float(explanation_strength),
        }
    )
    return final_utility, overlap


def upgrade_legacy_v6_selector_payload(
    payload: Mapping[str, Any],
    *,
    refinement_evidence: Mapping[str, Any],
    expected_model_digest: str,
) -> Dict[str, Any]:
    """Rebind one validated persisted V6 score to the V7 rank-only policy.

    This migration deliberately reuses the immutable base probabilities and
    persisted raw overlap cohort evidence.  It does not rerun crops, patches,
    embeddings, or a learned model, and it refuses unbound or effect-bearing
    V6 overlap payloads.
    """

    if (
        not isinstance(payload, Mapping)
        or payload.get("selector_contract")
        != LEGACY_V6_SELECTOR_PRIORITY_CONTRACT
        or payload.get("feature_contract") != SELECTOR_FEATURE_CONTRACT
        or payload.get("model_digest") != expected_model_digest
        or payload.get("utility_policy_contract")
        != SELECTOR_UTILITY_POLICY_CONTRACT
        or payload.get("dataset_overlap_application_contract")
        != LEGACY_V6_DATASET_OVERLAP_APPLICATION_CONTRACT
        or payload.get("dataset_overlap_diagnostic_contract")
        != LEGACY_V6_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT
        or payload.get("dataset_overlap_scoring_effect_enabled") is not False
    ):
        raise ValueError("selector_v6_migration_payload_contract_invalid")
    actionability = _bounded_probability(payload.get("actionable_probability"))
    reviewability = _bounded_probability(payload.get("reviewability_probability"))
    old_utility = actionability * (0.75 + 0.25 * reviewability)
    if not math.isclose(
        _finite(payload.get("expected_review_utility"), -1.0),
        old_utility,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("selector_v6_migration_utility_invalid")
    raw_overlap = payload.get("dataset_overlap")
    if not isinstance(raw_overlap, Mapping):
        raise ValueError("selector_v6_migration_overlap_invalid")
    if (
        raw_overlap.get("application_contract")
        != LEGACY_V6_DATASET_OVERLAP_APPLICATION_CONTRACT
        or raw_overlap.get("diagnostic_contract")
        != LEGACY_V6_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT
        or raw_overlap.get("scoring_effect_enabled") is not False
        or raw_overlap.get("applied") is not False
        or _finite(raw_overlap.get("probability_delta"), float("nan")) != 0.0
        or _finite(raw_overlap.get("utility_delta"), float("nan")) != 0.0
    ):
        raise ValueError("selector_v6_migration_overlap_effect_invalid")

    overlap = dict(raw_overlap)
    prior = refinement_evidence.get("frequent_overlap_prior")
    prior = prior if isinstance(prior, Mapping) else {}
    capture_group_left_out = bool(
        str(prior.get("candidate_capture_group_id") or "").strip()
        and prior.get("candidate_capture_group_excluded") is True
        and prior.get("candidate_source_excluded") is True
    )
    status = str(refinement_evidence.get("status") or "").strip()
    if status == "pair_conflict":
        overlap["applicable"] = False
        overlap["application_reason"] = "pair_conflict_preserved"
    elif overlap.get("applicable") is True and not capture_group_left_out:
        overlap["applicable"] = False
        overlap["application_reason"] = (
            "candidate_capture_group_not_left_out"
        )
    overlap.update(
        {
            "application_contract": DATASET_OVERLAP_APPLICATION_CONTRACT,
            "candidate_capture_group_id": str(
                prior.get("candidate_capture_group_id") or ""
            ).strip(),
            "candidate_capture_group_excluded": (
                prior.get("candidate_capture_group_excluded") is True
            ),
            "candidate_source_excluded": (
                prior.get("candidate_source_excluded") is True
            ),
        }
    )
    final_utility, overlap = _dataset_overlap_rank_adjustment(
        actionability=actionability,
        reviewability=reviewability,
        overlap_context=overlap,
    )
    upgraded = dict(payload)
    upgraded.update(
        {
            "selector_contract": SELECTOR_PRIORITY_CONTRACT,
            "base_model_selector_contract": (
                LEGACY_V6_SELECTOR_PRIORITY_CONTRACT
            ),
            "dataset_overlap_application_contract": (
                DATASET_OVERLAP_APPLICATION_CONTRACT
            ),
            "dataset_overlap_diagnostic_contract": (
                DATASET_OVERLAP_DIAGNOSTIC_CONTRACT
            ),
            "dataset_overlap_scoring_effect_enabled": True,
            "base_expected_review_utility": float(old_utility),
            "expected_review_utility": float(final_utility),
            "dataset_overlap": overlap,
        }
    )
    return upgraded


def build_selector_feature_row(
    candidate: Mapping[str, Any],
    same_image_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the immutable, allowlisted V6 model input for one candidate."""

    evidence = candidate.get("refined_outlier")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    gates = evidence.get("decision_gates")
    gates = gates if isinstance(gates, Mapping) else {}
    context = (
        same_image_context
        if isinstance(same_image_context, Mapping)
        else {}
    )

    current_class = str(
        evidence.get("current_class") or candidate.get("class_name") or ""
    ).strip()
    alternative_class = str(
        evidence.get("alternative_class")
        or candidate.get("suggested_neighbor_class")
        or ""
    ).strip()
    directed_pair = f"{current_class}->{alternative_class}"

    peer_count = max(
        0.0, _finite(context.get("trusted_same_class_anchor_count"))
    )
    alternative_peer_count = max(
        0.0,
        _finite(context.get("trusted_alternative_class_anchor_count")),
    )
    peer_shrink = peer_count / (peer_count + 5.0) if peer_count else 0.0
    contrast_peers = min(peer_count, alternative_peer_count)
    contrast_shrink = (
        contrast_peers / (contrast_peers + 5.0)
        if contrast_peers
        else 0.0
    )
    perspective_available = context.get("perspective_available") is True
    same_scale_available = (
        context.get("same_class_scale_reference_available") is True
    )
    scale_contrast_available = context.get("scale_contrast_available") is True
    image_object_count = max(0.0, _finite(context.get("image_object_count")))
    same_class_count = max(0.0, _finite(context.get("same_class_count")))
    current_threshold = _finite(evidence.get("current_support_threshold"))
    alternative_threshold = _finite(
        evidence.get("alternative_support_threshold")
    )
    pair_threshold = _finite(evidence.get("directed_pair_probe_threshold"))
    target_area = max(1e-9, _finite(evidence.get("target_bbox_area"), 1e-9))
    target_width = max(
        1e-9, _finite(evidence.get("target_bbox_width"), 1e-9)
    )
    target_height = max(
        1e-9, _finite(evidence.get("target_bbox_height"), 1e-9)
    )
    current_state = current_evidence_state(evidence)
    alternative_state = alternative_evidence_state(evidence)
    overlap_state = overlap_evidence_state(evidence)
    same_scale_outlier = (
        abs(_finite(context.get("same_class_log_area_residual")))
        * peer_shrink
        if same_scale_available
        else 0.0
    )
    perspective_scale_outlier = (
        abs(_finite(context.get("perspective_log_scale_residual")))
        * peer_shrink
        if perspective_available
        else 0.0
    )
    aspect_outlier = (
        abs(_finite(context.get("same_class_log_aspect_residual")))
        * peer_shrink
        if same_scale_available
        else 0.0
    )
    scale_typicality = math.exp(-same_scale_outlier)
    dataset_overlap = _material_overlap_context(
        evidence,
        context,
        current_state=current_state,
        alternative_state=alternative_state,
        overlap_state=overlap_state,
        scale_typicality=scale_typicality,
    )

    return {
        "feature_contract": SELECTOR_FEATURE_CONTRACT,
        "point_id": str(candidate.get("point_id") or ""),
        "stage1_suspicion": _bounded_probability(
            candidate.get("wrong_class_suspicion")
        ),
        "embedding_suspicion": _bounded_probability(
            candidate.get("embedding_wrong_class_suspicion")
        ),
        "neighbor_gap": _finite(candidate.get("top_other_neighbor_ratio"))
        - _finite(candidate.get("same_class_neighbor_ratio")),
        "current_support_margin": _finite(
            evidence.get("current_support_score")
        )
        - current_threshold,
        "alternative_support_margin": _finite(
            evidence.get("alternative_support_score")
        )
        - alternative_threshold,
        "intrinsic_support_gap": _finite(
            evidence.get("intrinsic_alternative_support")
        )
        - _finite(evidence.get("intrinsic_current_support")),
        "pair_probe_excess": _finite(
            evidence.get("directed_pair_probe_score")
        )
        - pair_threshold,
        "pair_probe_reliability": _bounded_probability(
            evidence.get("directed_pair_heldout_auroc")
        ),
        "pair_probe_lower_bound": _bounded_probability(
            evidence.get("directed_pair_eval_auroc_lower_bound")
        ),
        "view_agreement": _bounded_probability(evidence.get("view_agreement")),
        "overlap_count_log": math.log1p(
            max(0.0, _finite(evidence.get("overlap_object_count")))
        ),
        "target_log_area": math.log(target_area),
        "target_log_aspect": math.log(target_width / target_height),
        "annotated_overlap": float(gates.get("annotated_overlap") is True),
        "nested_overlap": float(gates.get("nested_overlap") is True),
        "external_alternative": float(
            gates.get("alternative_evidence_external_to_overlap") is True
        ),
        "localized_alternative": float(
            gates.get("alternative_evidence_localized_to_overlap") is True
        ),
        "current_spatial_coherence": float(
            gates.get("current_spatially_coherent") is True
        ),
        "alternative_spatial_coherence": float(
            gates.get("alternative_spatially_coherent_both_views") is True
        ),
        "source_resolution_ok": float(
            gates.get("source_resolution_sufficient") is True
        ),
        "reference_reliable": float(
            evidence.get("intrinsic_references_reliable") is True
        ),
        "diagnostic_pair_reliable": float(
            evidence.get("diagnostic_pair_reliable") is True
        ),
        "same_scale_outlier": same_scale_outlier,
        "perspective_scale_outlier": perspective_scale_outlier,
        "aspect_outlier": aspect_outlier,
        "current_alt_scale_contrast": (
            _finite(
                context.get(
                    "current_minus_alternative_abs_scale_residual"
                )
            )
            * contrast_shrink
            if scale_contrast_available
            else 0.0
        ),
        # Signed x/y/area residuals let a directed pair learn that "too
        # small" and "too large" can have different meanings. Absolute
        # residuals above remain the generic outlier-strength features.
        "same_width_residual": (
            _finite(context.get("same_class_log_width_residual"))
            * peer_shrink
            if same_scale_available
            else 0.0
        ),
        "same_height_residual": (
            _finite(context.get("same_class_log_height_residual"))
            * peer_shrink
            if same_scale_available
            else 0.0
        ),
        "same_area_signed_residual": (
            _finite(context.get("same_class_log_area_residual"))
            * peer_shrink
            if same_scale_available
            else 0.0
        ),
        "same_aspect_signed_residual": (
            _finite(context.get("same_class_log_aspect_residual"))
            * peer_shrink
            if same_scale_available
            else 0.0
        ),
        "same_scale_available": float(same_scale_available),
        "perspective_available": float(perspective_available),
        "scale_contrast_available": float(scale_contrast_available),
        "image_object_count_log": math.log1p(image_object_count),
        "same_class_count_log": math.log1p(same_class_count),
        "same_class_fraction": _bounded_probability(
            context.get("same_class_fraction")
        ),
        "same_anchor_count_log": math.log1p(peer_count),
        "alternative_anchor_count_log": math.log1p(
            alternative_peer_count
        ),
        "local_density_log": math.log1p(
            max(0.0, _finite(context.get("local_object_count_r10")))
        ),
        "local_same_class_log": math.log1p(
            max(0.0, _finite(context.get("local_same_class_count_r10")))
        ),
        "max_other_iou": _bounded_probability(
            context.get("max_other_class_iou")
        ),
        "max_target_cover": _bounded_probability(
            context.get("max_target_coverage_by_other")
        ),
        "border_touch": float(context.get("bbox_touches_border") is True),
        "bbox_outside_fraction": _bounded_probability(
            context.get("bbox_outside_fraction")
        ),
        "current_state": current_state,
        "alternative_state": alternative_state,
        "overlap_state": overlap_state,
        "current_class": current_class,
        "alternative_class": alternative_class,
        "directed_pair": directed_pair,
        "context_available": context.get("available") is True,
        "trusted_same_class_anchor_count": int(peer_count),
        "trusted_alternative_class_anchor_count": int(
            alternative_peer_count
        ),
        # Bounded observability fields are not separate model inputs; they
        # make the exact scale comparison inspectable in the VLM rail/UI.
        "image_object_count": int(image_object_count),
        "same_class_count": int(same_class_count),
        "bbox_width_norm": _finite(context.get("bbox_width_norm")),
        "bbox_height_norm": _finite(context.get("bbox_height_norm")),
        "bbox_area_fraction": _finite(context.get("bbox_area_fraction")),
        "same_class_peer_width_median_norm": _finite(
            context.get("same_class_peer_width_median_norm")
        ),
        "same_class_peer_height_median_norm": _finite(
            context.get("same_class_peer_height_median_norm")
        ),
        "same_class_peer_area_median_fraction": _finite(
            context.get("same_class_peer_area_median_fraction")
        ),
        "dataset_overlap_context": dataset_overlap,
        **dict(dataset_overlap["model_values"]),
    }


def _model_input(
    feature_row: Mapping[str, Any],
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        name: _finite(feature_row.get(name)) for name in numeric_features
    }
    for name in categorical_features:
        output[name] = str(feature_row.get(name) or "")
    return output


def _serialize_linear_pipeline(model: Any) -> Dict[str, Any]:
    vectorizer = model.named_steps["vec"]
    scaler = model.named_steps["scale"]
    classifier = model.named_steps["clf"]
    names = [str(name) for name in vectorizer.get_feature_names_out()]
    return {
        "kind": "scaled-sparse-logistic-v1",
        "feature_names": names,
        "scale": [float(value) for value in scaler.scale_.tolist()],
        "coefficients": [
            float(value) for value in classifier.coef_[0].tolist()
        ],
        "intercept": float(classifier.intercept_[0]),
    }


def _constant_model(probability: float) -> Dict[str, Any]:
    return {
        "kind": "constant-probability-v1",
        "probability": max(1e-5, min(1.0 - 1e-5, probability)),
        "feature_names": [],
        "scale": [],
        "coefficients": [],
        "intercept": 0.0,
    }


def _fit_binary_model(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    regularization_c: float,
    class_weight: Optional[str] = None,
) -> Dict[str, Any]:
    if len(rows) != len(labels) or not rows:
        raise ValueError("selector_model_training_rows_invalid")
    positives = sum(int(value) for value in labels)
    if positives == 0 or positives == len(labels):
        return _constant_model((positives + 1.0) / (len(labels) + 2.0))

    # Imported lazily: inference uses the small JSON artifact and has no
    # scikit-learn runtime dependency.
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    inputs = [
        _model_input(
            row,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        for row in rows
    ]
    pipeline = Pipeline(
        [
            ("vec", DictVectorizer(sparse=True)),
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    C=float(regularization_c),
                    max_iter=3000,
                    class_weight=class_weight,
                    random_state=0,
                ),
            ),
        ]
    )
    pipeline.fit(inputs, list(labels))
    return _serialize_linear_pipeline(pipeline)


def _dense_feature_value(
    feature_name: str, feature_row: Mapping[str, Any]
) -> float:
    if "=" in feature_name:
        key, expected = feature_name.split("=", 1)
        if key not in feature_row:
            raise ValueError(f"selector_feature_missing:{key}")
        return float(str(feature_row.get(key) or "") == expected)
    if feature_name not in feature_row:
        raise ValueError(f"selector_feature_missing:{feature_name}")
    try:
        numeric = float(feature_row.get(feature_name))
    except (TypeError, ValueError):
        return 0.0
    # Preserve explicit NaN so the portable scorer follows the missing-value
    # direction learned by sklearn. Infinities are not valid published feature
    # values and retain the established finite-zero sanitization behavior.
    return numeric if math.isfinite(numeric) or math.isnan(numeric) else 0.0


def _dense_feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> Any:
    import numpy as np

    return np.asarray(
        [
            [_dense_feature_value(name, row) for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )


def _serialize_hist_gradient_boosting(
    model: Any,
    *,
    feature_names: Sequence[str],
    training_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Export sklearn's fitted binary numeric HGB to portable JSON."""

    import numpy as np
    import sklearn

    classes = np.asarray(model.classes_)
    if classes.tolist() != [0, 1] or int(model.n_trees_per_iteration_) != 1:
        raise ValueError("selector_hgb_binary_layout_invalid")
    if len(feature_names) != int(model.n_features_in_):
        raise ValueError("selector_hgb_feature_count_mismatch")
    if getattr(model, "is_categorical_", None) is not None and bool(
        np.any(model.is_categorical_)
    ):
        raise ValueError("selector_hgb_categorical_split_unsupported")

    trees: List[List[Dict[str, Any]]] = []
    for iteration in model._predictors:
        if len(iteration) != 1:
            raise ValueError("selector_hgb_predictor_layout_invalid")
        predictor = iteration[0]
        if (
            predictor.raw_left_cat_bitsets.size
            or predictor.binned_left_cat_bitsets.size
        ):
            raise ValueError("selector_hgb_categorical_bitset_unsupported")
        nodes: List[Dict[str, Any]] = []
        for raw in predictor.nodes:
            if bool(raw["is_categorical"]):
                raise ValueError("selector_hgb_categorical_node_unsupported")
            if bool(raw["is_leaf"]):
                value = float(raw["value"])
                if not math.isfinite(value):
                    raise ValueError("selector_hgb_leaf_nonfinite")
                nodes.append({"leaf": value})
                continue
            feature = int(raw["feature_idx"])
            threshold = float(raw["num_threshold"])
            if not 0 <= feature < len(feature_names):
                raise ValueError("selector_hgb_feature_index_invalid")
            if not math.isfinite(threshold):
                raise ValueError("selector_hgb_threshold_nonfinite")
            nodes.append(
                {
                    "feature": feature,
                    "threshold": threshold,
                    "missing_left": bool(raw["missing_go_to_left"]),
                    "left": int(raw["left"]),
                    "right": int(raw["right"]),
                }
            )
        trees.append(nodes)
    baseline = float(np.asarray(model._baseline_prediction).reshape(-1)[0])
    if not math.isfinite(baseline):
        raise ValueError("selector_hgb_baseline_nonfinite")
    return {
        "kind": "binary-numeric-hist-gradient-boosting-v1",
        "classes": [0, 1],
        "feature_names": [str(name) for name in feature_names],
        "baseline_log_odds": baseline,
        "trees": trees,
        "training_config": dict(training_config),
        "training_library": {
            "name": "scikit-learn",
            "version": str(sklearn.__version__),
        },
    }


def _fit_hist_gradient_boosting(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    *,
    max_iter: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    learning_rate: float,
    l2_regularization: float,
) -> Dict[str, Any]:
    if len(rows) != len(labels) or len(rows) < 20:
        raise ValueError("selector_hgb_training_rows_invalid")
    positives = sum(int(value) for value in labels)
    if min(positives, len(labels) - positives) < 5:
        raise ValueError("selector_hgb_training_class_support_insufficient")
    from sklearn.ensemble import HistGradientBoostingClassifier

    config = {
        "loss": "log_loss",
        "learning_rate": float(learning_rate),
        "max_iter": int(max_iter),
        "max_leaf_nodes": int(max_leaf_nodes),
        "min_samples_leaf": int(min_samples_leaf),
        "l2_regularization": float(l2_regularization),
        "early_stopping": False,
        "random_state": 20260802,
    }
    model = HistGradientBoostingClassifier(**config)
    model.fit(_dense_feature_matrix(rows, GLOBAL_DENSE_FEATURES), list(labels))
    return _serialize_hist_gradient_boosting(
        model,
        feature_names=GLOBAL_DENSE_FEATURES,
        training_config=config,
    )


def _validate_hist_gradient_boosting_model(
    model: Any,
    *,
    detail: str,
) -> None:
    if (
        not isinstance(model, Mapping)
        or model.get("kind")
        != "binary-numeric-hist-gradient-boosting-v1"
        or model.get("classes") != [0, 1]
        or model.get("feature_names") != list(GLOBAL_DENSE_FEATURES)
    ):
        raise ValueError(f"selector_model_invalid:{detail}_contract")
    baseline = model.get("baseline_log_odds")
    if isinstance(baseline, bool) or not math.isfinite(
        _finite(baseline, float("nan"))
    ):
        raise ValueError(f"selector_model_invalid:{detail}_baseline")
    trees = model.get("trees")
    if not isinstance(trees, list) or not 1 <= len(trees) <= 512:
        raise ValueError(f"selector_model_invalid:{detail}_trees")
    for tree_index, nodes in enumerate(trees):
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 4096:
            raise ValueError(
                f"selector_model_invalid:{detail}_tree_shape:{tree_index}"
            )
        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(node_index: int) -> None:
            if node_index in visiting:
                raise ValueError(
                    f"selector_model_invalid:{detail}_tree_cycle:{tree_index}"
                )
            if node_index in visited:
                return
            if not 0 <= node_index < len(nodes):
                raise ValueError(
                    f"selector_model_invalid:{detail}_tree_child:{tree_index}"
                )
            visiting.add(node_index)
            node = nodes[node_index]
            if not isinstance(node, Mapping):
                raise ValueError(
                    f"selector_model_invalid:{detail}_tree_node:{tree_index}"
                )
            if set(node) == {"leaf"}:
                if not math.isfinite(_finite(node.get("leaf"), float("nan"))):
                    raise ValueError(
                        f"selector_model_invalid:{detail}_tree_leaf:{tree_index}"
                    )
            elif set(node) == {
                "feature",
                "threshold",
                "missing_left",
                "left",
                "right",
            }:
                feature = node.get("feature")
                left = node.get("left")
                right = node.get("right")
                if (
                    isinstance(feature, bool)
                    or not isinstance(feature, int)
                    or not 0 <= feature < len(GLOBAL_DENSE_FEATURES)
                    or type(node.get("missing_left")) is not bool
                    or isinstance(left, bool)
                    or not isinstance(left, int)
                    or isinstance(right, bool)
                    or not isinstance(right, int)
                    or left == node_index
                    or right == node_index
                    or not math.isfinite(
                        _finite(node.get("threshold"), float("nan"))
                    )
                ):
                    raise ValueError(
                        f"selector_model_invalid:{detail}_tree_split:{tree_index}"
                    )
                visit(left)
                visit(right)
            else:
                raise ValueError(
                    f"selector_model_invalid:{detail}_tree_keys:{tree_index}"
                )
            visiting.remove(node_index)
            visited.add(node_index)

        visit(0)
        if visited != set(range(len(nodes))):
            raise ValueError(
                f"selector_model_invalid:{detail}_tree_unreachable:{tree_index}"
            )


def _score_hist_gradient_boosting_model(
    model: Mapping[str, Any],
    feature_row: Mapping[str, Any],
) -> Tuple[float, float]:
    values = [
        _dense_feature_value(name, feature_row)
        for name in model.get("feature_names") or ()
    ]
    raw = _finite(model.get("baseline_log_odds"))
    for nodes in model.get("trees") or ():
        index = 0
        for _step in range(len(nodes)):
            node = nodes[index]
            if "leaf" in node:
                raw += _finite(node.get("leaf"))
                break
            value = values[int(node["feature"])]
            go_left = (
                bool(node["missing_left"])
                if math.isnan(value)
                else value <= float(node["threshold"])
            )
            index = int(node["left"] if go_left else node["right"])
        else:
            raise ValueError("selector_hgb_tree_traversal_invalid")
    return _sigmoid(raw), raw


def fit_selector_model_artifact(
    examples: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    global_max_iter: int = 60,
    global_max_leaf_nodes: int = 3,
    global_min_samples_leaf: int = 30,
    global_learning_rate: float = 0.05,
    global_l2_regularization: float = 10.0,
    linear_regularization_c: float = 0.03,
) -> Dict[str, Any]:
    """Fit the portable V6 base model from source-bound audit examples.

    ``skip`` and unclear labels remain abstentions. Dataset-overlap values are
    intentionally absent from every learned head; they remain observable only
    in the diagnostic payload.
    """

    feature_rows: List[Dict[str, Any]] = []
    reviewability_labels: List[int] = []
    actionable_rows: List[Dict[str, Any]] = []
    actionability_labels: List[int] = []
    strict_rows: List[Dict[str, Any]] = []
    strict_labels: List[int] = []
    source_groups: set[str] = set()
    for example in examples:
        candidate = example.get("candidate")
        prepared_feature_row = example.get("feature_row")
        if not isinstance(candidate, Mapping) and not isinstance(
            prepared_feature_row, Mapping
        ):
            continue
        source_group = str(example.get("source_group") or "").strip()
        if not source_group or type(example.get("reviewable")) is not bool:
            continue
        if isinstance(prepared_feature_row, Mapping):
            row = dict(prepared_feature_row)
            if row.get("feature_contract") != SELECTOR_FEATURE_CONTRACT:
                raise ValueError(
                    "selector_model_training_feature_contract_invalid"
                )
            for feature_name in GLOBAL_DENSE_FEATURES:
                _dense_feature_value(feature_name, row)
        else:
            row = build_selector_feature_row(
                candidate,
                example.get("same_image_context")
                if isinstance(example.get("same_image_context"), Mapping)
                else None,
            )
        source_groups.add(source_group)
        feature_rows.append(row)
        reviewability_labels.append(int(example["reviewable"]))
        if (
            example["reviewable"] is not True
            or type(example.get("actionable")) is not bool
        ):
            continue
        actionable_rows.append(row)
        actionability_labels.append(int(example["actionable"]))
        if example["actionable"] is True and type(
            example.get("strict_mislabeled")
        ) is bool:
            strict_rows.append(row)
            strict_labels.append(int(example["strict_mislabeled"]))

    if len(feature_rows) < 20 or len(actionable_rows) < 20:
        raise ValueError("selector_model_training_examples_insufficient")
    if min(
        sum(actionability_labels),
        len(actionability_labels) - sum(actionability_labels),
    ) < 5:
        raise ValueError("selector_model_training_class_support_insufficient")

    global_config = {
        "loss": "log_loss",
        "learning_rate": float(global_learning_rate),
        "max_iter": int(global_max_iter),
        "max_leaf_nodes": int(global_max_leaf_nodes),
        "min_samples_leaf": int(global_min_samples_leaf),
        "l2_regularization": float(global_l2_regularization),
        "early_stopping": False,
        "random_state": 20260802,
    }
    actionability_model = _fit_hist_gradient_boosting(
        actionable_rows,
        actionability_labels,
        max_iter=global_max_iter,
        max_leaf_nodes=global_max_leaf_nodes,
        min_samples_leaf=global_min_samples_leaf,
        learning_rate=global_learning_rate,
        l2_regularization=global_l2_regularization,
    )
    reviewability_model = _fit_hist_gradient_boosting(
        feature_rows,
        reviewability_labels,
        max_iter=global_max_iter,
        max_leaf_nodes=global_max_leaf_nodes,
        min_samples_leaf=global_min_samples_leaf,
        learning_rate=global_learning_rate,
        l2_regularization=global_l2_regularization,
    )
    strict_model = (
        _fit_binary_model(
            strict_rows,
            strict_labels,
            numeric_features=GLOBAL_NUMERIC_FEATURES,
            categorical_features=GLOBAL_CATEGORICAL_FEATURES,
            regularization_c=linear_regularization_c,
        )
        if len(strict_rows) >= 10
        else _constant_model(0.5)
    )

    artifact: Dict[str, Any] = {
        "schema": SELECTOR_MODEL_SCHEMA,
        "selector_contract": LEGACY_V6_SELECTOR_PRIORITY_CONTRACT,
        "feature_contract": SELECTOR_FEATURE_CONTRACT,
        "same_image_context_contract": (
            "same-image-object-context-features-v1"
        ),
        "utility_policy_contract": SELECTOR_UTILITY_POLICY_CONTRACT,
        "dataset_overlap_application_contract": (
            LEGACY_V6_DATASET_OVERLAP_APPLICATION_CONTRACT
        ),
        "dataset_overlap_diagnostic_contract": (
            LEGACY_V6_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT
        ),
        "dataset_overlap_scoring_effect_enabled": False,
        "global_actionability_model_contract": (
            GLOBAL_ACTIONABILITY_MODEL_CONTRACT
        ),
        "model_family": "pair-blind-global-hgb-with-overlap-diagnostics",
        "global_numeric_features": list(GLOBAL_NUMERIC_FEATURES),
        "global_categorical_features": list(GLOBAL_CATEGORICAL_FEATURES),
        "global_dense_features": list(GLOBAL_DENSE_FEATURES),
        "dataset_overlap_diagnostic_features": list(
            DATASET_OVERLAP_DIAGNOSTIC_FEATURES
        ),
        "linear_regularization_c": float(linear_regularization_c),
        "actionability_model": actionability_model,
        "reviewability_model": reviewability_model,
        "strict_mislabeled_given_actionable_model": strict_model,
        "training": {
            "example_count": len(feature_rows),
            "reviewable_count": sum(reviewability_labels),
            "actionability_example_count": len(actionable_rows),
            "actionable_count": sum(actionability_labels),
            "strict_example_count": len(strict_rows),
            "strict_mislabeled_count": sum(strict_labels),
            "source_group_count": len(source_groups),
            "global_model_config": global_config,
            "dataset_overlap_scoring_effect_enabled": False,
            "provenance": dict(provenance),
        },
        "changes_candidate_membership": False,
        "changes_semantic_status": False,
        "mutates_annotations": False,
    }
    artifact["model_digest"] = selector_model_digest(artifact)
    validate_selector_model_artifact(artifact)
    return artifact


def _validate_linear_model(model: Any, *, detail: str) -> None:
    if not isinstance(model, Mapping):
        raise ValueError(f"selector_model_invalid:{detail}")
    kind = str(model.get("kind") or "")
    if kind == "constant-probability-v1":
        probability = _finite(model.get("probability"), -1.0)
        if not 0.0 < probability < 1.0:
            raise ValueError(f"selector_model_invalid:{detail}_constant")
        return
    if kind != "scaled-sparse-logistic-v1":
        raise ValueError(f"selector_model_invalid:{detail}_kind")
    names = model.get("feature_names")
    scales = model.get("scale")
    coefficients = model.get("coefficients")
    if (
        not isinstance(names, list)
        or not isinstance(scales, list)
        or not isinstance(coefficients, list)
        or not names
        or len(names) != len(scales)
        or len(names) != len(coefficients)
        or len(set(str(name) for name in names)) != len(names)
        or any(not str(name).strip() for name in names)
        or any(
            not math.isfinite(_finite(value, float("nan")))
            for value in coefficients
        )
        or any(
            not math.isfinite(_finite(value, float("nan")))
            or _finite(value) <= 0.0
            for value in scales
        )
        or not math.isfinite(_finite(model.get("intercept"), float("nan")))
    ):
        raise ValueError(f"selector_model_invalid:{detail}_shape")


def validate_selector_model_artifact(artifact: Mapping[str, Any]) -> None:
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("schema") != SELECTOR_MODEL_SCHEMA
        or artifact.get("selector_contract")
        != LEGACY_V6_SELECTOR_PRIORITY_CONTRACT
        or artifact.get("feature_contract") != SELECTOR_FEATURE_CONTRACT
        or artifact.get("utility_policy_contract")
        != SELECTOR_UTILITY_POLICY_CONTRACT
        or artifact.get("dataset_overlap_application_contract")
        != LEGACY_V6_DATASET_OVERLAP_APPLICATION_CONTRACT
        or artifact.get("dataset_overlap_diagnostic_contract")
        != LEGACY_V6_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT
        or artifact.get("dataset_overlap_scoring_effect_enabled") is not False
        or artifact.get("global_actionability_model_contract")
        != GLOBAL_ACTIONABILITY_MODEL_CONTRACT
        or artifact.get("global_numeric_features")
        != list(GLOBAL_NUMERIC_FEATURES)
        or artifact.get("global_categorical_features")
        != list(GLOBAL_CATEGORICAL_FEATURES)
        or artifact.get("global_dense_features")
        != list(GLOBAL_DENSE_FEATURES)
        or artifact.get("dataset_overlap_diagnostic_features")
        != list(DATASET_OVERLAP_DIAGNOSTIC_FEATURES)
        or artifact.get("changes_candidate_membership") is not False
        or artifact.get("changes_semantic_status") is not False
        or artifact.get("mutates_annotations") is not False
        or str(artifact.get("model_digest") or "")
        != selector_model_digest(artifact)
    ):
        raise ValueError("selector_model_invalid:artifact_contract")
    declared_names = [
        *list(GLOBAL_NUMERIC_FEATURES),
        *list(GLOBAL_CATEGORICAL_FEATURES),
        *list(GLOBAL_DENSE_FEATURES),
        *list(DATASET_OVERLAP_DIAGNOSTIC_FEATURES),
    ]
    if any(
        name in FORBIDDEN_MODEL_FEATURE_NAMES
        or any(name.startswith(prefix) for prefix in FORBIDDEN_MODEL_FEATURE_PREFIXES)
        for name in declared_names
    ):
        raise ValueError("selector_model_invalid:forbidden_feature")
    _validate_hist_gradient_boosting_model(
        artifact.get("actionability_model"), detail="actionability"
    )
    _validate_hist_gradient_boosting_model(
        artifact.get("reviewability_model"), detail="reviewability"
    )
    _validate_linear_model(
        artifact.get("strict_mislabeled_given_actionable_model"),
        detail="strict_mislabeled",
    )
    strict_model = artifact.get("strict_mislabeled_given_actionable_model")
    if isinstance(strict_model, Mapping) and strict_model.get("kind") == (
        "scaled-sparse-logistic-v1"
    ):
        strict_feature_names = strict_model.get("feature_names")
        if not isinstance(strict_feature_names, list) or any(
            str(name) not in GLOBAL_DENSE_FEATURES
            for name in strict_feature_names
        ):
            raise ValueError(
                "selector_model_invalid:strict_mislabeled_feature_allowlist"
            )


def _expanded_feature_value(
    feature_name: str, feature_row: Mapping[str, Any]
) -> float:
    if "=" in feature_name:
        key, expected = feature_name.split("=", 1)
        return float(str(feature_row.get(key) or "") == expected)
    return _finite(feature_row.get(feature_name))


def _score_linear_model(
    model: Mapping[str, Any], feature_row: Mapping[str, Any]
) -> Tuple[float, Dict[str, float]]:
    if model.get("kind") == "constant-probability-v1":
        return _bounded_probability(model.get("probability")), {
            "constant": _logit(_bounded_probability(model.get("probability")))
        }
    names = list(model.get("feature_names") or [])
    scales = list(model.get("scale") or [])
    coefficients = list(model.get("coefficients") or [])
    contributions: Dict[str, float] = {}
    linear = _finite(model.get("intercept"))
    contributions["intercept"] = linear
    for name, scale, coefficient in zip(names, scales, coefficients):
        contribution = (
            _expanded_feature_value(str(name), feature_row)
            / _finite(scale, 1.0)
            * _finite(coefficient)
        )
        contributions[str(name)] = contribution
        linear += contribution
    return _sigmoid(linear), contributions


def _score_selector_feature_row_v6_validated(
    feature_row: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    actionability, global_raw_margin = (
        _score_hist_gradient_boosting_model(
            artifact["actionability_model"], feature_row
        )
    )
    reviewability, _reviewability_raw_margin = (
        _score_hist_gradient_boosting_model(
            artifact["reviewability_model"], feature_row
        )
    )
    strict_given_actionable, _ = _score_linear_model(
        artifact["strict_mislabeled_given_actionable_model"], feature_row
    )
    raw_overlap = feature_row.get("dataset_overlap_context")
    overlap = dict(raw_overlap) if isinstance(raw_overlap, Mapping) else {}
    base_utility = actionability * (0.75 + 0.25 * reviewability)
    utility, overlap = _dataset_overlap_rank_adjustment(
        actionability=actionability,
        reviewability=reviewability,
        overlap_context=overlap,
    )

    mislabeled_probability = actionability * strict_given_actionable
    actionable_geometry_probability = actionability * (
        1.0 - strict_given_actionable
    )
    return {
        "selector_contract": SELECTOR_PRIORITY_CONTRACT,
        "base_model_selector_contract": (
            LEGACY_V6_SELECTOR_PRIORITY_CONTRACT
        ),
        "feature_contract": SELECTOR_FEATURE_CONTRACT,
        "model_digest": str(artifact.get("model_digest") or ""),
        "utility_policy_contract": SELECTOR_UTILITY_POLICY_CONTRACT,
        "dataset_overlap_application_contract": (
            DATASET_OVERLAP_APPLICATION_CONTRACT
        ),
        "dataset_overlap_diagnostic_contract": (
            DATASET_OVERLAP_DIAGNOSTIC_CONTRACT
        ),
        "dataset_overlap_scoring_effect_enabled": True,
        "global_actionability_model_contract": (
            GLOBAL_ACTIONABILITY_MODEL_CONTRACT
        ),
        "expected_review_utility": float(utility),
        "base_expected_review_utility": float(base_utility),
        "actionable_probability": float(actionability),
        "reviewability_probability": float(reviewability),
        "conditional_annotation_state": {
            "mislabeled": float(mislabeled_probability),
            "actionable_geometry_or_composite": float(
                actionable_geometry_probability
            ),
            "valid_or_harmless": float(1.0 - actionability),
        },
        "insufficient_evidence_probability": float(1.0 - reviewability),
        "current_evidence_state": str(feature_row.get("current_state") or ""),
        "alternative_evidence_state": str(
            feature_row.get("alternative_state") or ""
        ),
        "overlap_evidence_state": str(feature_row.get("overlap_state") or ""),
        "same_image_context": {
            "available": feature_row.get("context_available") is True,
            "image_object_count": int(
                _finite(feature_row.get("image_object_count"))
            ),
            "same_class_count": int(
                _finite(feature_row.get("same_class_count"))
            ),
            "trusted_same_class_anchor_count": int(
                _finite(
                    feature_row.get("trusted_same_class_anchor_count")
                )
            ),
            "trusted_alternative_class_anchor_count": int(
                _finite(
                    feature_row.get(
                        "trusted_alternative_class_anchor_count"
                    )
                )
            ),
            "bbox_width_norm": _finite(feature_row.get("bbox_width_norm")),
            "bbox_height_norm": _finite(feature_row.get("bbox_height_norm")),
            "bbox_area_fraction": _finite(
                feature_row.get("bbox_area_fraction")
            ),
            "same_class_peer_width_median_norm": _finite(
                feature_row.get("same_class_peer_width_median_norm")
            ),
            "same_class_peer_height_median_norm": _finite(
                feature_row.get("same_class_peer_height_median_norm")
            ),
            "same_class_peer_area_median_fraction": _finite(
                feature_row.get("same_class_peer_area_median_fraction")
            ),
            "same_class_log_width_residual": _finite(
                feature_row.get("same_width_residual")
            ),
            "same_class_log_height_residual": _finite(
                feature_row.get("same_height_residual")
            ),
            "same_class_log_area_residual": _finite(
                feature_row.get("same_area_signed_residual")
            ),
            "perspective_log_scale_residual": _finite(
                feature_row.get("perspective_scale_outlier")
            ),
        },
        "dataset_overlap": overlap,
        "global_model": {
            "contract": GLOBAL_ACTIONABILITY_MODEL_CONTRACT,
            "raw_margin": float(global_raw_margin),
            "actionable_probability": float(actionability),
        },
    }


def score_selector_feature_row(
    feature_row: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_selector_model_artifact(artifact)
    return _score_selector_feature_row_v6_validated(feature_row, artifact)


_DEFAULT_MODEL_CACHE: Optional[Dict[str, Any]] = None


def default_selector_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / SELECTOR_MODEL_FILENAME


def load_default_selector_model() -> Dict[str, Any]:
    global _DEFAULT_MODEL_CACHE
    if _DEFAULT_MODEL_CACHE is not None:
        return _DEFAULT_MODEL_CACHE
    path = default_selector_model_path()
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("class_analysis_selector_v6_model_unavailable") from exc
    validate_selector_model_artifact(artifact)
    _DEFAULT_MODEL_CACHE = dict(artifact)
    return _DEFAULT_MODEL_CACHE


def score_selector_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    same_image_context_by_point_id: Mapping[str, Mapping[str, Any]],
    artifact: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Score every candidate atomically and return payloads keyed by ID."""

    model = dict(
        load_default_selector_model() if artifact is None else artifact
    )
    validate_selector_model_artifact(model)
    scored: Dict[str, Dict[str, Any]] = {}
    context_available_count = 0
    evidence_state_counts: Dict[str, int] = {}
    overlap_available_count = 0
    overlap_applicable_count = 0
    overlap_applied_count = 0
    overlap_effect_count = 0
    overlap_demoted_count = 0
    overlap_promoted_count = 0
    overlap_zero_effect_count = 0
    overlap_reason_counts: Dict[str, int] = {}
    absolute_probability_effects: List[float] = []
    absolute_utility_effects: List[float] = []
    for candidate in candidates:
        point_id = str(candidate.get("point_id") or "").strip()
        if not point_id or point_id in scored:
            raise ValueError("selector_v6_candidate_identity_invalid")
        context = same_image_context_by_point_id.get(point_id)
        feature_row = build_selector_feature_row(candidate, context)
        payload = _score_selector_feature_row_v6_validated(feature_row, model)
        if feature_row.get("context_available") is True:
            context_available_count += 1
        overlap = payload.get("dataset_overlap") or {}
        overlap_available_count += int(overlap.get("available") is True)
        overlap_applicable_count += int(overlap.get("applicable") is True)
        overlap_applied_count += int(overlap.get("applied") is True)
        probability_delta = _finite(overlap.get("probability_delta"))
        utility_delta = _finite(overlap.get("utility_delta"))
        has_effect = abs(utility_delta) > 1e-12
        overlap_effect_count += int(has_effect)
        overlap_demoted_count += int(utility_delta < -1e-12)
        overlap_promoted_count += int(utility_delta > 1e-12)
        overlap_zero_effect_count += int(not has_effect)
        absolute_probability_effects.append(abs(probability_delta))
        absolute_utility_effects.append(abs(utility_delta))
        reason = str(overlap.get("application_reason") or "")
        overlap_reason_counts[reason] = overlap_reason_counts.get(reason, 0) + 1
        state = str(payload.get("current_evidence_state") or "")
        evidence_state_counts[state] = evidence_state_counts.get(state, 0) + 1
        scored[point_id] = payload
    if len(scored) != len(candidates):
        raise ValueError("selector_v6_candidate_count_mismatch")
    return scored, {
        "contract": SELECTOR_PRIORITY_CONTRACT,
        "base_model_selector_contract": (
            LEGACY_V6_SELECTOR_PRIORITY_CONTRACT
        ),
        "feature_contract": SELECTOR_FEATURE_CONTRACT,
        "model_schema": SELECTOR_MODEL_SCHEMA,
        "model_digest": str(model.get("model_digest") or ""),
        "utility_policy_contract": SELECTOR_UTILITY_POLICY_CONTRACT,
        "dataset_overlap_application_contract": (
            DATASET_OVERLAP_APPLICATION_CONTRACT
        ),
        "dataset_overlap_diagnostic_contract": (
            DATASET_OVERLAP_DIAGNOSTIC_CONTRACT
        ),
        "dataset_overlap_scoring_effect_enabled": True,
        "global_actionability_model_contract": (
            GLOBAL_ACTIONABILITY_MODEL_CONTRACT
        ),
        "model_family": str(model.get("model_family") or ""),
        "candidate_count": len(candidates),
        "context_available_count": context_available_count,
        "current_evidence_state_counts": evidence_state_counts,
        "dataset_overlap": {
            "application_contract": DATASET_OVERLAP_APPLICATION_CONTRACT,
            "diagnostic_contract": DATASET_OVERLAP_DIAGNOSTIC_CONTRACT,
            "scoring_effect_enabled": True,
            "rank_only": True,
            "uses_human_review_labels": False,
            "maximum_rank_discount_fraction": (
                DATASET_OVERLAP_MAXIMUM_RANK_DISCOUNT_FRACTION
            ),
            "available_candidate_count": overlap_available_count,
            "applicable_candidate_count": overlap_applicable_count,
            "applied_candidate_count": overlap_applied_count,
            "effect_candidate_count": overlap_effect_count,
            "application_reason_counts": dict(
                sorted(overlap_reason_counts.items())
            ),
            "demoted_candidate_count": overlap_demoted_count,
            "promoted_candidate_count": overlap_promoted_count,
            "zero_effect_candidate_count": overlap_zero_effect_count,
            "maximum_absolute_probability_effect": max(
                absolute_probability_effects,
                default=0.0,
            ),
            "mean_absolute_probability_effect": (
                math.fsum(absolute_probability_effects)
                / len(absolute_probability_effects)
                if absolute_probability_effects
                else 0.0
            ),
            "maximum_absolute_utility_effect": max(
                absolute_utility_effects,
                default=0.0,
            ),
            "mean_absolute_utility_effect": (
                math.fsum(absolute_utility_effects)
                / len(absolute_utility_effects)
                if absolute_utility_effects
                else 0.0
            ),
        },
        "changes_candidate_membership": False,
        "changes_semantic_status": False,
        "mutates_annotations": False,
    }


__all__ = [
    "CURRENT_EVIDENCE_STATES",
    "DATASET_OVERLAP_APPLICATION_CONTRACT",
    "DATASET_OVERLAP_APPLICATION_REASONS",
    "DATASET_OVERLAP_DIAGNOSTIC_CONTRACT",
    "DATASET_OVERLAP_DIAGNOSTIC_FEATURES",
    "DATASET_OVERLAP_MAXIMUM_RANK_DISCOUNT_FRACTION",
    "GLOBAL_ACTIONABILITY_MODEL_CONTRACT",
    "GLOBAL_CATEGORICAL_FEATURES",
    "GLOBAL_DENSE_FEATURES",
    "GLOBAL_NUMERIC_FEATURES",
    "OVERLAP_EVIDENCE_STATES",
    "LEGACY_V6_DATASET_OVERLAP_APPLICATION_CONTRACT",
    "LEGACY_V6_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT",
    "LEGACY_V6_SELECTOR_PRIORITY_CONTRACT",
    "SELECTOR_FEATURE_CONTRACT",
    "SELECTOR_MODEL_SCHEMA",
    "SELECTOR_PRIORITY_CONTRACT",
    "SELECTOR_UTILITY_POLICY_CONTRACT",
    "build_selector_feature_row",
    "current_evidence_state",
    "default_selector_model_path",
    "fit_selector_model_artifact",
    "load_default_selector_model",
    "overlap_evidence_state",
    "score_selector_candidates",
    "score_selector_feature_row",
    "selector_model_digest",
    "upgrade_legacy_v6_selector_payload",
    "validate_selector_model_artifact",
]
