#!/usr/bin/env python
"""Compact feature builder for the learned second-stage policy layer."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

POLICY_LAYER_FEATURE_SCHEMA_VERSION = 1
PRIMARY_SOURCES = ["yolo", "rfdetr", "sam3_text", "sam3_similarity", "unknown"]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _logit(prob: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return float(math.log(p / (1.0 - p)))


def _normalize_source_fields(row: Dict[str, Any]) -> Tuple[str, Set[str]]:
    primary = str(row.get("score_source") or row.get("source") or "unknown").strip().lower() or "unknown"
    source_set: Set[str] = set()
    raw = row.get("source_list")
    if isinstance(raw, (list, tuple, set)):
        for src in raw:
            name = str(src or "").strip().lower()
            if name:
                source_set.add(name)
    elif isinstance(raw, str):
        name = raw.strip().lower()
        if name:
            source_set.add(name)
    score_by_source = row.get("score_by_source")
    if isinstance(score_by_source, dict):
        for src in score_by_source.keys():
            name = str(src or "").strip().lower()
            if name:
                source_set.add(name)
    source_set.add(primary)
    return primary, source_set


def _entropy(prob_vec: np.ndarray) -> float:
    if prob_vec.size == 0:
        return 0.0
    clipped = np.clip(prob_vec.astype(np.float64), 1e-9, 1.0)
    total = float(clipped.sum())
    if total <= 0.0:
        return 0.0
    probs = clipped / total
    return float(-(probs * np.log(probs)).sum())


def parse_meta_rows(meta_raw: Sequence[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in meta_raw:
        if isinstance(row, dict):
            rows.append(dict(row))
            continue
        try:
            rows.append(json.loads(str(row)))
        except Exception:
            rows.append({})
    return rows


def compute_policy_feature_schema_hash(
    feature_names: Sequence[str],
    *,
    extra_config: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "version": POLICY_LAYER_FEATURE_SCHEMA_VERSION,
        "feature_names": [str(name) for name in feature_names],
    }
    if extra_config:
        payload["extra_config"] = extra_config
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


class PolicyFeatureBundle(Dict[str, Any]):
    pass


def _normalize_anchor_similarity_cfg(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw, dict) or not bool(raw.get("enabled")):
        return {"enabled": False}
    return {
        "enabled": True,
        "min_base_prob": max(0.0, min(1.0, _safe_float(raw.get("min_base_prob"), 0.9))),
        "topk_same_label": max(1, int(_safe_float(raw.get("topk_same_label"), 4))),
        "topk_any": max(1, int(_safe_float(raw.get("topk_any"), 8))),
        "require_detector_support": bool(raw.get("require_detector_support", True)),
    }


def _compute_anchor_similarity_features(
    *,
    X: np.ndarray,
    feature_names: Sequence[str],
    name_to_idx: Dict[str, int],
    row_images: Sequence[str],
    row_labels: Sequence[str],
    row_has_detector_support: np.ndarray,
    base_probs: np.ndarray,
    anchor_cfg: Dict[str, Any],
) -> Tuple[List[str], np.ndarray]:
    if not bool(anchor_cfg.get("enabled")):
        return [], np.zeros((X.shape[0], 0), dtype=np.float32)
    emb_names = sorted([name for name in feature_names if name.startswith("clf_emb_rp::")])
    emb_idx = [name_to_idx[name] for name in emb_names]
    if not emb_idx:
        return [], np.zeros((X.shape[0], 0), dtype=np.float32)

    emb_matrix = np.asarray(X[:, emb_idx], dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb_norm = emb_matrix / norms

    feature_names_out = [
        "anchor_any_count",
        "anchor_same_label_count",
        "anchor_other_label_count",
        "anchor_any_cos_max",
        "anchor_any_cos_mean_topk",
        "anchor_same_label_cos_max",
        "anchor_same_label_cos_mean_topk",
        "anchor_other_label_cos_max",
        "anchor_margin_same_vs_other",
    ]
    out = np.zeros((X.shape[0], len(feature_names_out)), dtype=np.float32)

    image_to_indices: Dict[str, List[int]] = {}
    for idx, image in enumerate(row_images):
        image_to_indices.setdefault(image, []).append(idx)

    min_base_prob = float(anchor_cfg["min_base_prob"])
    topk_same_label = int(anchor_cfg["topk_same_label"])
    topk_any = int(anchor_cfg["topk_any"])
    require_detector_support = bool(anchor_cfg["require_detector_support"])

    def _cos_stats(vec: np.ndarray, indexes: List[int], *, topk: int) -> Tuple[float, float]:
        if not indexes:
            return 0.0, 0.0
        sims = np.asarray(emb_norm[indexes] @ vec, dtype=np.float32)
        if sims.size == 0:
            return 0.0, 0.0
        max_sim = float(np.max(sims))
        if sims.size > int(topk):
            sims = np.sort(sims)[::-1][: int(topk)]
        return max_sim, float(np.mean(sims))

    for image_indices in image_to_indices.values():
        trusted = [
            idx
            for idx in image_indices
            if float(base_probs[idx]) >= min_base_prob
            and ((not require_detector_support) or bool(row_has_detector_support[idx]))
        ]
        if not trusted:
            continue
        trusted_sorted = sorted(trusted, key=lambda idx: float(base_probs[idx]), reverse=True)
        trusted_any = trusted_sorted[:topk_any]
        trusted_by_label: Dict[str, List[int]] = {}
        for idx in trusted_sorted:
            trusted_by_label.setdefault(row_labels[idx], []).append(idx)

        for idx in image_indices:
            vec = emb_norm[idx]
            any_idx = [j for j in trusted_any if j != idx]
            same_all = [j for j in trusted_by_label.get(row_labels[idx], []) if j != idx]
            same_idx = same_all[:topk_same_label]
            other_idx = [j for j in trusted_any if j != idx and row_labels[j] != row_labels[idx]]

            any_max, any_mean = _cos_stats(vec, any_idx, topk=topk_any)
            same_max, same_mean = _cos_stats(vec, same_idx, topk=topk_same_label)
            other_max, _ = _cos_stats(vec, other_idx, topk=topk_any)
            out[idx] = np.asarray(
                [
                    float(len(any_idx)),
                    float(len(same_all)),
                    float(len(other_idx)),
                    any_max,
                    any_mean,
                    same_max,
                    same_mean,
                    other_max,
                    same_max - other_max,
                ],
                dtype=np.float32,
            )
    return feature_names_out, out


def build_policy_feature_matrix(
    X_full: np.ndarray,
    feature_names_full: Sequence[str],
    meta_rows: Sequence[Dict[str, Any]],
    base_probs: Sequence[float],
    *,
    anchor_similarity: Optional[Dict[str, Any]] = None,
) -> PolicyFeatureBundle:
    rows = list(meta_rows)
    feature_names = [str(name) for name in feature_names_full]
    X = np.asarray(X_full, dtype=np.float32)
    probs = np.asarray(base_probs, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("feature_matrix_invalid_shape")
    if X.shape[0] != len(rows) or X.shape[0] != probs.shape[0]:
        raise ValueError("row_alignment_mismatch")

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    labels = sorted(
        {
            str((row or {}).get("label") or "").strip().lower()
            for row in rows
            if str((row or {}).get("label") or "").strip()
        }
    )
    clf_prob_names = [
        name for name in feature_names if name.startswith("clf_prob::") and not name.startswith("clf_prob::__bg_")
    ]
    clf_prob_names = sorted(clf_prob_names)
    clf_prob_labels = [name.split("::", 1)[1] for name in clf_prob_names]
    clf_prob_idx = [name_to_idx[name] for name in clf_prob_names]

    common_numeric_names = [
        "cand_has_yolo",
        "cand_has_rfdetr",
        "cand_has_sam3_text",
        "cand_has_sam3_similarity",
        "cand_score_yolo",
        "cand_score_rfdetr",
        "cand_raw_score_yolo",
        "cand_raw_score_rfdetr",
        "support_count_total",
        "support_atom_count",
        "support_atom_same_label_count",
        "support_run_count",
        "support_source_count",
        "support_score_mean",
        "support_score_max",
        "support_score_std",
        "support_iou_mean",
        "support_iou_max",
        "support_source_entropy",
        "support_source_entropy_norm",
        "support_source_max_share",
        "support_detector_share",
        "support_sam_share",
        "support_detector_count",
        "support_sam_count",
        "det_iou_max_yolo_same_label",
        "det_iou_max_rfdetr_same_label",
        "det_iou_max_detector_same_label",
        "det_iou_max_detector_any_label",
        "geom_center_x",
        "geom_center_y",
        "geom_width",
        "geom_height",
        "geom_area",
        "geom_aspect_ratio",
        "geom_prior_area_z",
        "geom_prior_aspect_z",
        "geom_prior_area_tail",
        "geom_prior_aspect_tail",
        "ctx_neighbor_count_all",
        "ctx_neighbor_count_same",
        "ctx_neighbor_ratio_same",
        "ctx_neighbor_score_mean_same",
        "ctx_total_area",
        "ctx_avg_area",
        "ctx_avg_aspect_ratio",
    ]
    common_numeric_names.extend([name for name in feature_names if name.startswith("cand_run_max::")])
    common_numeric_names.extend([name for name in feature_names if name.startswith("cand_run_count::")])
    common_numeric_names = [name for name in common_numeric_names if name in name_to_idx]

    numeric_matrix = (
        X[:, [name_to_idx[name] for name in common_numeric_names]]
        if common_numeric_names
        else np.zeros((X.shape[0], 0), dtype=np.float32)
    )

    row_labels = [str((row or {}).get("label") or "").strip().lower() for row in rows]
    row_images = [str((row or {}).get("image") or "").strip() for row in rows]
    row_primary: List[str] = []
    row_has_detector_support = np.zeros(X.shape[0], dtype=bool)
    row_is_sam_text = np.zeros(X.shape[0], dtype=bool)
    row_is_sam_sim = np.zeros(X.shape[0], dtype=bool)
    row_is_sam_only = np.zeros(X.shape[0], dtype=bool)
    row_is_detector_only = np.zeros(X.shape[0], dtype=bool)
    row_is_detector_and_sam = np.zeros(X.shape[0], dtype=bool)
    for row_idx, row in enumerate(rows):
        primary_source, source_set = _normalize_source_fields(row or {})
        has_detector_support = ("yolo" in source_set) or ("rfdetr" in source_set)
        is_sam_text = primary_source == "sam3_text"
        is_sam_sim = primary_source == "sam3_similarity"
        is_sam_only = (is_sam_text or is_sam_sim) and not has_detector_support
        is_detector_only = has_detector_support and not (is_sam_text or is_sam_sim)
        is_detector_and_sam = has_detector_support and ("sam3_text" in source_set or "sam3_similarity" in source_set)
        row_primary.append(primary_source)
        row_has_detector_support[row_idx] = has_detector_support
        row_is_sam_text[row_idx] = is_sam_text
        row_is_sam_sim[row_idx] = is_sam_sim
        row_is_sam_only[row_idx] = is_sam_only
        row_is_detector_only[row_idx] = is_detector_only
        row_is_detector_and_sam[row_idx] = is_detector_and_sam

    anchor_cfg = _normalize_anchor_similarity_cfg(anchor_similarity)
    anchor_feature_names, anchor_feature_matrix = _compute_anchor_similarity_features(
        X=X,
        feature_names=feature_names,
        name_to_idx=name_to_idx,
        row_images=row_images,
        row_labels=row_labels,
        row_has_detector_support=row_has_detector_support,
        base_probs=probs,
        anchor_cfg=anchor_cfg,
    )

    policy_feature_names: List[str] = ["base_prob", "base_logit"]
    policy_feature_names.extend(common_numeric_names)
    policy_feature_names.extend([f"label_onehot::{label}" for label in labels])
    policy_feature_names.extend([f"primary_source_onehot::{src}" for src in PRIMARY_SOURCES])
    policy_feature_names.extend(
        [
            "clf_prob_label",
            "clf_prob_max",
            "clf_prob_second",
            "clf_prob_margin",
            "clf_prob_entropy",
            "clf_label_is_top1",
            "clf_label_rank",
            "has_detector_support",
            "is_sam_only",
            "is_detector_only",
            "is_detector_and_sam",
            "is_sam3_text_primary",
            "is_sam3_similarity_primary",
            "is_yolo_primary",
            "is_rfdetr_primary",
            "is_primary_source_supported_by_detector",
        ]
    )

    dynamic_label_feature_names: List[str] = [
        "label_ctx::sam3_text_max",
        "label_ctx::sam3_text_count",
        "label_ctx::sam3_sim_max",
        "label_ctx::sam3_sim_count",
        "label_ctx::src_count::yolo",
        "label_ctx::src_mean::yolo",
        "label_ctx::src_count::rfdetr",
        "label_ctx::src_mean::rfdetr",
        "label_ctx::src_count::sam3_text",
        "label_ctx::src_mean::sam3_text",
        "label_ctx::src_count::sam3_similarity",
        "label_ctx::src_mean::sam3_similarity",
    ]
    policy_feature_names.extend(dynamic_label_feature_names)
    policy_feature_names.extend(anchor_feature_names)

    interaction_names = [
        "base_logit_x_is_sam_only",
        "base_logit_x_has_detector_support",
        "base_logit_x_is_sam3_text_primary",
        "base_logit_x_is_sam3_similarity_primary",
        "base_logit_x_clf_prob_margin",
        "base_logit_x_support_detector_share",
        "base_logit_x_support_sam_share",
        "base_logit_x_det_iou_max_detector_same_label",
        "base_logit_x_support_iou_max",
        "base_logit_x_clf_label_is_top1",
        "clf_prob_margin_x_is_sam_only",
        "clf_prob_entropy_x_is_sam_only",
        "clf_prob_margin_x_has_detector_support",
        "det_iou_max_detector_same_label_x_is_sam_only",
    ]
    policy_feature_names.extend(interaction_names)

    out = np.zeros((X.shape[0], len(policy_feature_names)), dtype=np.float32)
    out[:, 0] = probs.astype(np.float32)
    base_logits = np.asarray([_logit(p) for p in probs], dtype=np.float32)
    out[:, 1] = base_logits
    cursor = 2
    if numeric_matrix.size:
        width = numeric_matrix.shape[1]
        out[:, cursor : cursor + width] = numeric_matrix
        cursor += width

    label_to_pos = {label: idx for idx, label in enumerate(labels)}
    source_to_pos = {src: idx for idx, src in enumerate(PRIMARY_SOURCES)}

    support_detector_share_idx = policy_feature_names.index("support_detector_share") if "support_detector_share" in policy_feature_names else None
    support_sam_share_idx = policy_feature_names.index("support_sam_share") if "support_sam_share" in policy_feature_names else None
    support_iou_max_idx = policy_feature_names.index("support_iou_max") if "support_iou_max" in policy_feature_names else None
    det_iou_max_detector_same_label_idx = policy_feature_names.index("det_iou_max_detector_same_label") if "det_iou_max_detector_same_label" in policy_feature_names else None

    label_offset = cursor
    source_offset = label_offset + len(labels)
    scalar_offset = source_offset + len(PRIMARY_SOURCES)
    dynamic_offset = scalar_offset + 16
    anchor_offset = dynamic_offset + len(dynamic_label_feature_names)
    interaction_offset = anchor_offset + len(anchor_feature_names)

    subgroup_flags: Dict[str, np.ndarray] = {
        "sam_only": np.zeros(X.shape[0], dtype=bool),
        "sam3_similarity_primary": np.zeros(X.shape[0], dtype=bool),
        "sam3_text_primary": np.zeros(X.shape[0], dtype=bool),
        "detector_supported": np.zeros(X.shape[0], dtype=bool),
    }

    clf_label_pos = {label: idx for idx, label in enumerate(clf_prob_labels)}

    for row_idx, row in enumerate(rows):
        label = row_labels[row_idx]
        primary_source = row_primary[row_idx]
        has_detector_support = bool(row_has_detector_support[row_idx])
        is_sam_text = bool(row_is_sam_text[row_idx])
        is_sam_sim = bool(row_is_sam_sim[row_idx])
        is_sam_only = bool(row_is_sam_only[row_idx])
        is_detector_only = bool(row_is_detector_only[row_idx])
        is_detector_and_sam = bool(row_is_detector_and_sam[row_idx])

        subgroup_flags["sam_only"][row_idx] = is_sam_only
        subgroup_flags["sam3_similarity_primary"][row_idx] = is_sam_sim
        subgroup_flags["sam3_text_primary"][row_idx] = is_sam_text
        subgroup_flags["detector_supported"][row_idx] = has_detector_support

        if label in label_to_pos:
            out[row_idx, label_offset + label_to_pos[label]] = 1.0
        out[row_idx, source_offset + source_to_pos.get(primary_source, source_to_pos["unknown"])] = 1.0

        if clf_prob_idx:
            prob_vec = X[row_idx, clf_prob_idx].astype(np.float32)
            if prob_vec.size:
                order = np.argsort(prob_vec)[::-1]
                top1 = int(order[0])
                top2 = int(order[1]) if order.size > 1 else top1
                label_idx = clf_label_pos.get(label, -1)
                label_prob = float(prob_vec[label_idx]) if label_idx >= 0 else 0.0
                max_prob = float(prob_vec[top1])
                second_prob = float(prob_vec[top2]) if order.size > 1 else max_prob
                margin = max_prob - second_prob
                ent = _entropy(prob_vec)
                label_is_top1 = 1.0 if label_idx == top1 and label_idx >= 0 else 0.0
                label_rank = (
                    float(np.where(order == label_idx)[0][0] + 1)
                    if label_idx >= 0 and np.any(order == label_idx)
                    else float(prob_vec.size + 1)
                )
            else:
                label_prob = max_prob = second_prob = margin = ent = label_is_top1 = 0.0
                label_rank = 0.0
        else:
            label_prob = max_prob = second_prob = margin = ent = label_is_top1 = label_rank = 0.0

        scalar_values = [
            label_prob,
            max_prob,
            second_prob,
            margin,
            ent,
            label_is_top1,
            label_rank,
            1.0 if has_detector_support else 0.0,
            1.0 if is_sam_only else 0.0,
            1.0 if is_detector_only else 0.0,
            1.0 if is_detector_and_sam else 0.0,
            1.0 if is_sam_text else 0.0,
            1.0 if is_sam_sim else 0.0,
            1.0 if primary_source == "yolo" else 0.0,
            1.0 if primary_source == "rfdetr" else 0.0,
            1.0 if ((is_sam_text or is_sam_sim) and has_detector_support) else 0.0,
        ]
        out[row_idx, scalar_offset : scalar_offset + len(scalar_values)] = np.asarray(scalar_values, dtype=np.float32)

        dynamic_names = [
            f"sam3_text_max::{label}",
            f"sam3_text_count::{label}",
            f"sam3_sim_max::{label}",
            f"sam3_sim_count::{label}",
            f"ctx_source_count::{label}::yolo",
            f"ctx_source_mean::{label}::yolo",
            f"ctx_source_count::{label}::rfdetr",
            f"ctx_source_mean::{label}::rfdetr",
            f"ctx_source_count::{label}::sam3_text",
            f"ctx_source_mean::{label}::sam3_text",
            f"ctx_source_count::{label}::sam3_similarity",
            f"ctx_source_mean::{label}::sam3_similarity",
        ]
        for local_idx, name in enumerate(dynamic_names):
            out[row_idx, dynamic_offset + local_idx] = float(X[row_idx, name_to_idx[name]]) if name in name_to_idx else 0.0
        if anchor_feature_names:
            out[row_idx, anchor_offset : anchor_offset + len(anchor_feature_names)] = anchor_feature_matrix[row_idx]

        support_detector_share = float(out[row_idx, support_detector_share_idx]) if support_detector_share_idx is not None else 0.0
        support_sam_share = float(out[row_idx, support_sam_share_idx]) if support_sam_share_idx is not None else 0.0
        support_iou_max = float(out[row_idx, support_iou_max_idx]) if support_iou_max_idx is not None else 0.0
        det_iou_same = float(out[row_idx, det_iou_max_detector_same_label_idx]) if det_iou_max_detector_same_label_idx is not None else 0.0
        interaction_values = [
            base_logits[row_idx] * (1.0 if is_sam_only else 0.0),
            base_logits[row_idx] * (1.0 if has_detector_support else 0.0),
            base_logits[row_idx] * (1.0 if is_sam_text else 0.0),
            base_logits[row_idx] * (1.0 if is_sam_sim else 0.0),
            base_logits[row_idx] * margin,
            base_logits[row_idx] * support_detector_share,
            base_logits[row_idx] * support_sam_share,
            base_logits[row_idx] * det_iou_same,
            base_logits[row_idx] * support_iou_max,
            base_logits[row_idx] * label_is_top1,
            margin * (1.0 if is_sam_only else 0.0),
            ent * (1.0 if is_sam_only else 0.0),
            margin * (1.0 if has_detector_support else 0.0),
            det_iou_same * (1.0 if is_sam_only else 0.0),
        ]
        out[row_idx, interaction_offset : interaction_offset + len(interaction_values)] = np.asarray(
            interaction_values, dtype=np.float32
        )

    return PolicyFeatureBundle(
        X=out,
        feature_names=policy_feature_names,
        feature_schema_hash=compute_policy_feature_schema_hash(
            policy_feature_names,
            extra_config={"anchor_similarity": anchor_cfg} if anchor_cfg["enabled"] else None,
        ),
        subgroup_flags=subgroup_flags,
        meta_rows=rows,
        anchor_similarity=anchor_cfg,
    )
