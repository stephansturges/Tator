from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

IMG_PROB_EXTRA_NAMES = {
    "img_clf_prob_label",
    "img_clf_prob_delta_label",
    "img_clf_prob_max",
    "img_clf_prob_entropy",
    "img_clf_prob_cosine",
}
CANONICAL_BASE_VARIANT = "base"
IMGRAW_VARIANT = "imgraw"
SCENE_SUMMARY_VARIANT = "scene_summary_v1"
TRUSTED_CENTROID_VARIANT = "trusted_centroid_v1"
COMBINED_VARIANT = "combined_v1"
VARIANT_FEATURE_PREFIXES = {
    CANONICAL_BASE_VARIANT: (),
    IMGRAW_VARIANT: (),
    SCENE_SUMMARY_VARIANT: ("imgctx_scene_",),
    TRUSTED_CENTROID_VARIANT: ("imgctx_trusted_",),
    COMBINED_VARIANT: ("imgctx_scene_", "imgctx_trusted_"),
}

ROW_ALIGNED_KEYS = {
    "X",
    "y",
    "y_iou",
    "best_iou_any",
    "best_label_any",
    "meta",
}


@dataclass(frozen=True)
class DerivedFeatureBlock:
    X: np.ndarray
    feature_names: list[str]
    stats: dict[str, Any]
    variant_type: str


def is_img_prob_feature(name: str) -> bool:
    text = str(name or "")
    return text.startswith("img_clf_prob::") or text in IMG_PROB_EXTRA_NAMES


def is_img_emb_feature(name: str) -> bool:
    return str(name or "").startswith("img_clf_emb_rp::")


def is_cand_emb_feature(name: str) -> bool:
    return str(name or "").startswith("clf_emb_rp::")


def parse_meta_rows(meta: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in meta:
        try:
            row = json.loads(str(raw))
        except Exception:
            row = {}
        if not isinstance(row, dict):
            row = {}
        rows.append(row)
    return rows


def compute_feature_schema_hash(
    feature_names: Sequence[str],
    *,
    classifier_classes: Sequence[str] = (),
    labelmap: Sequence[str] = (),
    context_variant_id: str = CANONICAL_BASE_VARIANT,
    variant_config: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "feature_names": [str(name) for name in feature_names],
        "classifier_classes": [str(name) for name in classifier_classes],
        "labelmap": [str(name) for name in labelmap],
        "context_variant_id": str(context_variant_id or CANONICAL_BASE_VARIANT),
        "variant_config": dict(variant_config or {}),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def infer_feature_schema_hash(payload: Mapping[str, Any]) -> str:
    existing = payload.get("feature_schema_hash")
    if existing is not None:
        value = _scalar_to_str(existing)
        if value:
            return value
    return compute_feature_schema_hash(
        [str(name) for name in payload.get("feature_names", [])],
        classifier_classes=[str(name) for name in payload.get("classifier_classes", [])],
        labelmap=[str(name) for name in payload.get("labelmap", [])],
        context_variant_id=_scalar_to_str(payload.get("context_variant_id")) or CANONICAL_BASE_VARIANT,
        variant_config=_load_variant_config(payload.get("variant_config_json")),
    )


def payload_context_variant_id(payload: Mapping[str, Any]) -> str:
    return _scalar_to_str(payload.get("context_variant_id")) or CANONICAL_BASE_VARIANT


def copy_schema_metadata(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    variant_id = payload_context_variant_id(payload)
    variant_config_json = _scalar_to_str(payload.get("variant_config_json")) or "{}"
    return {
        "feature_schema_hash": np.asarray(str(infer_feature_schema_hash(payload))),
        "context_variant_id": np.asarray(str(variant_id)),
        "parent_feature_npz": np.asarray(_scalar_to_str(payload.get("parent_feature_npz"))),
        "parent_feature_schema_hash": np.asarray(
            _scalar_to_str(payload.get("parent_feature_schema_hash"))
        ),
        "variant_config_json": np.asarray(str(variant_config_json)),
    }


def load_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_npz_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def with_schema_metadata(
    payload: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    context_variant_id: str,
    variant_config: Mapping[str, Any] | None,
    parent_feature_npz: str,
    parent_feature_schema_hash: str,
) -> dict[str, Any]:
    out = dict(payload)
    schema_hash = compute_feature_schema_hash(
        feature_names,
        classifier_classes=[str(name) for name in payload.get("classifier_classes", [])],
        labelmap=[str(name) for name in payload.get("labelmap", [])],
        context_variant_id=context_variant_id,
        variant_config=variant_config,
    )
    out["feature_names"] = np.asarray([str(name) for name in feature_names], dtype=object)
    out["feature_schema_hash"] = np.asarray(str(schema_hash))
    out["context_variant_id"] = np.asarray(str(context_variant_id))
    out["parent_feature_npz"] = np.asarray(str(parent_feature_npz))
    out["parent_feature_schema_hash"] = np.asarray(str(parent_feature_schema_hash))
    out["variant_config_json"] = np.asarray(
        json.dumps(dict(variant_config or {}), sort_keys=True, separators=(",", ":"))
    )
    return out


def subset_payload_by_images(payload: Mapping[str, Any], selected_images: Iterable[str]) -> dict[str, Any]:
    images = {str(name) for name in selected_images if str(name)}
    meta_rows = parse_meta_rows(payload.get("meta", []))
    keep_mask = np.asarray([str(row.get("image") or "") in images for row in meta_rows], dtype=bool)
    out: dict[str, Any] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        if key in ROW_ALIGNED_KEYS and arr.shape and arr.shape[0] == keep_mask.shape[0]:
            out[key] = arr[keep_mask]
        else:
            out[key] = value
    return out


def derive_imgraw_payload(
    payload: Mapping[str, Any],
    *,
    parent_feature_npz: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_names = [str(name) for name in payload.get("feature_names", [])]
    keep_idx = [idx for idx, name in enumerate(feature_names) if not is_img_prob_feature(name)]
    X = np.asarray(payload["X"], dtype=np.float32)
    X_new = X[:, keep_idx]
    feature_names_new = [feature_names[idx] for idx in keep_idx]
    stats = {
        "variant_type": "global-only",
        "dropped_img_prob_features": int(len(feature_names) - len(feature_names_new)),
        "new_feature_count": 0,
    }
    out = dict(payload)
    out["X"] = X_new
    out = with_schema_metadata(
        out,
        feature_names=feature_names_new,
        context_variant_id=IMGRAW_VARIANT,
        variant_config={"drop_img_probs": True},
        parent_feature_npz=parent_feature_npz,
        parent_feature_schema_hash=infer_feature_schema_hash(payload),
    )
    return out, stats


def derive_scene_summary_block(payload: Mapping[str, Any]) -> DerivedFeatureBlock:
    X = np.asarray(payload["X"], dtype=np.float32)
    feature_names = [str(name) for name in payload.get("feature_names", [])]
    meta_rows = parse_meta_rows(payload.get("meta", []))
    images = [str(row.get("image") or "") for row in meta_rows]
    labels = [str(row.get("label") or "").strip().lower() for row in meta_rows]

    detector_score = np.maximum(
        _feature_column(X, feature_names, "cand_score_yolo"),
        _feature_column(X, feature_names, "cand_score_rfdetr"),
    )
    detector_supported = (
        (_feature_column(X, feature_names, "cand_has_yolo") > 0.0)
        | (_feature_column(X, feature_names, "cand_has_rfdetr") > 0.0)
    )
    dual_detector = (
        (_feature_column(X, feature_names, "cand_has_yolo") > 0.0)
        & (_feature_column(X, feature_names, "cand_has_rfdetr") > 0.0)
    )
    sam_supported = (
        (_feature_column(X, feature_names, "cand_has_sam3_text") > 0.0)
        | (_feature_column(X, feature_names, "cand_has_sam3_similarity") > 0.0)
    )
    sam_only = sam_supported & (~detector_supported)
    support_total = _feature_column(X, feature_names, "support_count_total")
    candidate_score = _meta_float(meta_rows, "score")

    rows = X.shape[0]
    feats = np.zeros((rows, 16), dtype=np.float32)
    grouped: dict[str, list[int]] = {}
    for idx, image_name in enumerate(images):
        grouped.setdefault(image_name, []).append(idx)

    for idxs in grouped.values():
        by_label: dict[str, list[int]] = {}
        det_supported_total = int(np.sum(detector_supported[idxs]))
        dual_total = int(np.sum(dual_detector[idxs]))
        sam_only_total = int(np.sum(sam_only[idxs]))
        total_candidates = max(1, len(idxs))
        detector_ratio = float(det_supported_total) / float(total_candidates)
        sam_only_ratio = float(sam_only_total) / float(total_candidates)
        for idx in idxs:
            by_label.setdefault(labels[idx], []).append(idx)
        for idx in idxs:
            label = labels[idx]
            label_idxs = by_label.get(label, [])
            same_det_idxs = [j for j in label_idxs if detector_supported[j]]
            same_dual_idxs = [j for j in label_idxs if dual_detector[j]]
            same_sam_only_idxs = [j for j in label_idxs if sam_only[j]]
            det_sorted = sorted((float(detector_score[j]) for j in same_det_idxs), reverse=True)
            top1 = det_sorted[0] if det_sorted else 0.0
            top2 = det_sorted[1] if len(det_sorted) > 1 else 0.0
            gap = top1 - top2
            other_best = 0.0
            for other_label, other_idxs in by_label.items():
                if other_label == label:
                    continue
                other_best = max(other_best, float(np.max(detector_score[other_idxs])) if other_idxs else 0.0)
            cand_rank_pct = _percentile_rank(candidate_score[idx], [candidate_score[j] for j in label_idxs])
            support_rank_pct = _percentile_rank(float(support_total[idx]), [float(support_total[j]) for j in label_idxs])
            det_rank_pct = _percentile_rank(float(detector_score[idx]), [float(detector_score[j]) for j in label_idxs])
            same_label_density = float(len(label_idxs)) / float(total_candidates)
            feats[idx] = np.asarray(
                [
                    top1,
                    top2,
                    gap,
                    float(len(same_det_idxs)),
                    float(len(same_dual_idxs)),
                    float(len(same_sam_only_idxs)),
                    cand_rank_pct,
                    support_rank_pct,
                    det_rank_pct,
                    other_best,
                    top1 - other_best,
                    float(det_supported_total),
                    float(dual_total),
                    float(sam_only_total),
                    detector_ratio,
                    same_label_density + sam_only_ratio,
                ],
                dtype=np.float32,
            )
    names = [
        "imgctx_scene_same_label_detector_top1",
        "imgctx_scene_same_label_detector_top2",
        "imgctx_scene_same_label_detector_gap",
        "imgctx_scene_same_label_detector_supported_count",
        "imgctx_scene_same_label_dual_detector_count",
        "imgctx_scene_same_label_sam_only_count",
        "imgctx_scene_candidate_score_percentile",
        "imgctx_scene_support_percentile",
        "imgctx_scene_detector_percentile",
        "imgctx_scene_best_other_label_detector_score",
        "imgctx_scene_label_vs_other_margin",
        "imgctx_scene_total_detector_supported_count",
        "imgctx_scene_total_dual_detector_count",
        "imgctx_scene_total_sam_only_count",
        "imgctx_scene_detector_supported_ratio",
        "imgctx_scene_same_label_density_plus_sam_only_ratio",
    ]
    stats = summarize_new_feature_block(feats, images=images, existing_X=X)
    return DerivedFeatureBlock(X=feats, feature_names=names, stats=stats, variant_type="candidate-specific")


def derive_trusted_centroid_block(payload: Mapping[str, Any]) -> DerivedFeatureBlock:
    X = np.asarray(payload["X"], dtype=np.float32)
    feature_names = [str(name) for name in payload.get("feature_names", [])]
    meta_rows = parse_meta_rows(payload.get("meta", []))
    images = [str(row.get("image") or "") for row in meta_rows]
    labels = [str(row.get("label") or "").strip().lower() for row in meta_rows]
    cand_embed, embed_names = _feature_block(X, feature_names, is_cand_emb_feature)
    if cand_embed.shape[1] == 0:
        feats = np.zeros((X.shape[0], 10), dtype=np.float32)
        stats = summarize_new_feature_block(feats, images=images, existing_X=X)
        return DerivedFeatureBlock(
            X=feats,
            feature_names=[
                "imgctx_trusted_same_cosine",
                "imgctx_trusted_same_l2",
                "imgctx_trusted_dual_cosine",
                "imgctx_trusted_dual_l2",
                "imgctx_trusted_best_same_cosine",
                "imgctx_trusted_best_other_cosine",
                "imgctx_trusted_same_other_margin",
                "imgctx_trusted_same_pool_size",
                "imgctx_trusted_dual_pool_size",
                "imgctx_trusted_missing_pool_flag",
            ],
            stats=stats,
            variant_type="candidate-specific",
        )

    detector_supported = (
        (_feature_column(X, feature_names, "cand_has_yolo") > 0.0)
        | (_feature_column(X, feature_names, "cand_has_rfdetr") > 0.0)
    )
    dual_detector = (
        (_feature_column(X, feature_names, "cand_has_yolo") > 0.0)
        & (_feature_column(X, feature_names, "cand_has_rfdetr") > 0.0)
    )
    support_total = _feature_column(X, feature_names, "support_count_total")
    rows = X.shape[0]
    feats = np.zeros((rows, 11), dtype=np.float32)
    grouped: dict[tuple[str, str], list[int]] = {}
    grouped_image: dict[str, list[int]] = {}
    for idx, image_name in enumerate(images):
        grouped[(image_name, labels[idx])].append(idx) if (image_name, labels[idx]) in grouped else grouped.setdefault((image_name, labels[idx]), [idx])
        grouped_image.setdefault(image_name, []).append(idx)
    norms = _normalize_rows(cand_embed)
    for idx in range(rows):
        image_name = images[idx]
        label = labels[idx]
        same_idxs = grouped.get((image_name, label), [])
        trusted_idxs = [j for j in same_idxs if detector_supported[j]]
        dual_idxs = [j for j in same_idxs if dual_detector[j]]
        other_idxs = [j for j in grouped_image.get(image_name, []) if labels[j] != label and detector_supported[j]]
        same_centroid = _centroid(cand_embed, trusted_idxs)
        dual_centroid = _centroid(cand_embed, dual_idxs)
        current = cand_embed[idx]
        current_norm = norms[idx]
        best_same = _best_cosine(current, current_norm, cand_embed, norms, trusted_idxs, exclude_idx=idx)
        best_other = _best_cosine(current, current_norm, cand_embed, norms, other_idxs, exclude_idx=None)
        feats[idx] = np.asarray(
            [
                _cosine(current, same_centroid),
                _l2(current, same_centroid),
                _cosine(current, dual_centroid),
                _l2(current, dual_centroid),
                best_same,
                best_other,
                best_same - best_other,
                float(len(trusted_idxs)),
                float(len(dual_idxs)),
                float(np.mean(support_total[trusted_idxs])) if trusted_idxs else 0.0,
                1.0 if not trusted_idxs else 0.0,
            ],
            dtype=np.float32,
        )
    names = [
        "imgctx_trusted_same_cosine",
        "imgctx_trusted_same_l2",
        "imgctx_trusted_dual_cosine",
        "imgctx_trusted_dual_l2",
        "imgctx_trusted_best_same_cosine",
        "imgctx_trusted_best_other_cosine",
        "imgctx_trusted_same_other_margin",
        "imgctx_trusted_same_pool_size",
        "imgctx_trusted_dual_pool_size",
        "imgctx_trusted_same_pool_support_mean",
        "imgctx_trusted_missing_pool_flag",
    ]
    stats = summarize_new_feature_block(feats, images=images, existing_X=X)
    return DerivedFeatureBlock(X=feats, feature_names=names, stats=stats, variant_type="candidate-specific")


def derive_variant_payload(
    payload: Mapping[str, Any],
    *,
    variant: str,
    parent_feature_npz: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant_name = str(variant or "").strip().lower()
    if variant_name == IMGRAW_VARIANT:
        return derive_imgraw_payload(payload, parent_feature_npz=parent_feature_npz)

    base_payload, base_stats = derive_imgraw_payload(payload, parent_feature_npz=parent_feature_npz)
    X_base = np.asarray(base_payload["X"], dtype=np.float32)
    feature_names_base = [str(name) for name in base_payload["feature_names"]]
    blocks: list[DerivedFeatureBlock] = []
    if variant_name in {SCENE_SUMMARY_VARIANT, COMBINED_VARIANT}:
        blocks.append(derive_scene_summary_block(base_payload))
    if variant_name in {TRUSTED_CENTROID_VARIANT, COMBINED_VARIANT}:
        blocks.append(derive_trusted_centroid_block(base_payload))
    if not blocks:
        raise ValueError(f"Unsupported variant: {variant}")
    X_parts = [X_base] + [block.X for block in blocks]
    name_parts = feature_names_base + [name for block in blocks for name in block.feature_names]
    X_new = np.concatenate(X_parts, axis=1)
    combined_stats = {
        "variant_type": "candidate-specific",
        "imgraw_base": base_stats,
        "blocks": [
            {
                "name": SCENE_SUMMARY_VARIANT if block.feature_names[0].startswith("imgctx_scene_") else TRUSTED_CENTROID_VARIANT,
                "stats": block.stats,
            }
            for block in blocks
        ],
        "new_feature_count": int(sum(block.X.shape[1] for block in blocks)),
    }
    out = dict(base_payload)
    out["X"] = X_new
    out = with_schema_metadata(
        out,
        feature_names=name_parts,
        context_variant_id=variant_name,
        variant_config={
            "drop_img_probs": True,
            "blocks": [
                SCENE_SUMMARY_VARIANT if block.feature_names[0].startswith("imgctx_scene_") else TRUSTED_CENTROID_VARIANT
                for block in blocks
            ],
        },
        parent_feature_npz=parent_feature_npz,
        parent_feature_schema_hash=infer_feature_schema_hash(payload),
    )
    return out, combined_stats


def summarize_new_feature_block(
    X_new: np.ndarray,
    *,
    images: Sequence[str],
    existing_X: np.ndarray | None = None,
    eps: float = 1e-6,
    sample_rows: int = 8192,
) -> dict[str, Any]:
    X_new = np.asarray(X_new, dtype=np.float32)
    if X_new.ndim != 2:
        raise ValueError("X_new must be 2D")
    if X_new.shape[1] == 0:
        return {
            "zero_fraction": 1.0,
            "varying_fraction": 0.0,
            "duplicate_fraction": 0.0,
            "new_feature_count": 0,
            "variant_type": "global-only",
        }
    zero_fraction = float(np.mean(np.isclose(X_new, 0.0)))
    grouped: dict[str, list[int]] = {}
    for idx, image_name in enumerate(images):
        grouped.setdefault(str(image_name), []).append(idx)
    varying = 0
    if X_new.shape[1] > 0:
        for col_idx in range(X_new.shape[1]):
            col = X_new[:, col_idx]
            has_within_image_variation = False
            for idxs in grouped.values():
                if len(idxs) < 2:
                    continue
                if float(np.std(col[idxs])) > eps:
                    has_within_image_variation = True
                    break
            if has_within_image_variation:
                varying += 1
    varying_fraction = float(varying) / float(max(1, X_new.shape[1]))
    duplicate_fraction = 0.0
    if existing_X is not None and X_new.size and np.asarray(existing_X).size:
        duplicate_fraction = _approx_duplicate_fraction(
            np.asarray(existing_X, dtype=np.float32),
            X_new,
            eps=eps,
            sample_rows=sample_rows,
        )
    variant_type = "candidate-specific" if varying_fraction >= 0.2 else "global-only"
    return {
        "zero_fraction": zero_fraction,
        "varying_fraction": varying_fraction,
        "duplicate_fraction": duplicate_fraction,
        "new_feature_count": int(X_new.shape[1]),
        "variant_type": variant_type,
    }


def compute_payload_feature_block_stats(
    payload: Mapping[str, Any],
    *,
    expected_variant_id: str | None = None,
) -> dict[str, Any]:
    variant_id = payload_context_variant_id(payload)
    if expected_variant_id:
        variant_id = str(expected_variant_id or "").strip().lower() or variant_id
    prefixes = VARIANT_FEATURE_PREFIXES.get(variant_id, ())
    feature_names = [str(name) for name in payload.get("feature_names", [])]
    X = np.asarray(payload.get("X", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("payload X must be 2D")
    keep_idx = [
        idx for idx, name in enumerate(feature_names) if any(name.startswith(prefix) for prefix in prefixes)
    ]
    existing_idx = [idx for idx in range(len(feature_names)) if idx not in keep_idx]
    images = [str(row.get("image") or "") for row in parse_meta_rows(payload.get("meta", []))]
    X_new = X[:, keep_idx] if keep_idx else np.zeros((X.shape[0], 0), dtype=np.float32)
    X_existing = X[:, existing_idx] if existing_idx else np.zeros((X.shape[0], 0), dtype=np.float32)
    stats = summarize_new_feature_block(X_new, images=images, existing_X=X_existing)
    stats["context_variant_id"] = str(variant_id)
    return stats


def _approx_duplicate_fraction(
    existing_X: np.ndarray,
    new_X: np.ndarray,
    *,
    eps: float,
    sample_rows: int,
) -> float:
    rows = min(existing_X.shape[0], new_X.shape[0])
    if rows == 0 or existing_X.shape[1] == 0 or new_X.shape[1] == 0:
        return 0.0
    if rows > sample_rows:
        idx = np.linspace(0, rows - 1, num=sample_rows, dtype=np.int64)
        existing_sample = existing_X[idx]
        new_sample = new_X[idx]
    else:
        existing_sample = existing_X[:rows]
        new_sample = new_X[:rows]
    existing_norm = _normalize_cols(existing_sample)
    new_norm = _normalize_cols(new_sample)
    corr = np.abs(new_norm.T @ existing_norm) / float(max(1, new_norm.shape[0] - 1))
    max_corr = np.max(corr, axis=1) if corr.size else np.zeros((new_X.shape[1],), dtype=np.float32)
    duplicates = np.sum(max_corr >= (1.0 - max(eps, 1e-5)))
    return float(duplicates) / float(max(1, new_X.shape[1]))


def _normalize_cols(X: np.ndarray) -> np.ndarray:
    centered = X - np.mean(X, axis=0, keepdims=True)
    std = np.std(centered, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return centered / std


def _load_variant_config(value: Any) -> dict[str, Any]:
    text = _scalar_to_str(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _scalar_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


def _feature_column(X: np.ndarray, feature_names: Sequence[str], name: str) -> np.ndarray:
    try:
        idx = list(feature_names).index(name)
    except ValueError:
        return np.zeros((X.shape[0],), dtype=np.float32)
    return np.asarray(X[:, idx], dtype=np.float32)


def _feature_block(
    X: np.ndarray,
    feature_names: Sequence[str],
    predicate,
) -> tuple[np.ndarray, list[str]]:
    keep_idx = [idx for idx, name in enumerate(feature_names) if predicate(name)]
    if not keep_idx:
        return np.zeros((X.shape[0], 0), dtype=np.float32), []
    return np.asarray(X[:, keep_idx], dtype=np.float32), [str(feature_names[idx]) for idx in keep_idx]


def _meta_float(meta_rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    out = np.zeros((len(meta_rows),), dtype=np.float32)
    for idx, row in enumerate(meta_rows):
        try:
            out[idx] = float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            out[idx] = 0.0
    return out


def _percentile_rank(value: float, values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size <= 1:
        return 0.0
    return float(np.mean(arr <= float(value)))


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    return X / norms


def _centroid(X: np.ndarray, idxs: Sequence[int]) -> np.ndarray:
    if not idxs:
        return np.zeros((X.shape[1],), dtype=np.float32)
    return np.asarray(np.mean(X[np.asarray(list(idxs), dtype=np.int64)], axis=0), dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.linalg.norm(a - b))


def _best_cosine(
    current: np.ndarray,
    current_normed: np.ndarray,
    X: np.ndarray,
    X_normed: np.ndarray,
    idxs: Sequence[int],
    *,
    exclude_idx: int | None,
) -> float:
    best = 0.0
    for idx in idxs:
        if exclude_idx is not None and idx == exclude_idx:
            continue
        val = float(np.dot(current_normed, X_normed[idx]))
        if val > best:
            best = val
    return best
