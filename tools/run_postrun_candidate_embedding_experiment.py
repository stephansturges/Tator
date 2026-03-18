#!/usr/bin/env python3
"""Run winner-only candidate-embedding variants after sweep + alpha extension complete."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import localinferenceapi as api
from tools.build_ensemble_features import _activate_classifier_runtime, _resolve_image_path
from tools.context_feature_variants import compute_feature_schema_hash


@dataclass(frozen=True)
class ContextSpec:
    lane: str
    view: str
    seed: int
    variant: str
    labeled_npz: Path
    prepass_jsonl: Path
    val_images_file: Path
    baseline_f1: float
    baseline_precision: float
    baseline_recall: float
    baseline_delta_vs_union_f1: float
    baseline_coverage_preservation: float


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run(cmd: Sequence[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(ROOT),
        check=True,
        text=True,
        capture_output=capture,
    )


def _wait_for_file(path: Path, *, poll_seconds: int) -> None:
    while not path.exists():
        time.sleep(max(5, int(poll_seconds)))


def _parse_meta_rows(meta_arr: np.ndarray) -> List[Dict[str, Any]]:
    return [json.loads(str(row)) for row in meta_arr]


def _winner_baseline_contexts(
    *,
    run_root: Path,
    alpha_root: Path,
    selection_view: str,
) -> Tuple[str, float, Dict[str, Any], Dict[str, Any], List[ContextSpec]]:
    final_default = _load_json(run_root / "final_default_recipe.json")
    winner_lane = str(final_default.get("winner_lane") or "").strip()
    if not winner_lane:
        raise RuntimeError("winner_lane missing from final_default_recipe.json")
    lane_settings = final_default.get("lane_settings") if isinstance(final_default.get("lane_settings"), dict) else {}
    hp = lane_settings.get("hp") if isinstance(lane_settings.get("hp"), dict) else {}
    scenario = lane_settings.get("scenario") if isinstance(lane_settings.get("scenario"), dict) else {}
    policy = lane_settings.get("policy") if isinstance(lane_settings.get("policy"), dict) else {}
    if not hp or not scenario:
        raise RuntimeError("winner lane settings incomplete in final_default_recipe.json")

    alpha_ranked = _load_json(alpha_root / "results_ranked.json")
    selected_alpha = float(alpha_ranked.get("winner_alpha") or scenario.get("alpha") or 0.0)
    alpha_raw = _load_json(alpha_root / "results_raw.json")
    alpha_rows = alpha_raw.get("rows") if isinstance(alpha_raw.get("rows"), list) else []
    baseline_rows = [
        row
        for row in alpha_rows
        if str(row.get("lane")) == winner_lane and abs(float(row.get("alpha", -1.0)) - selected_alpha) < 1e-9
    ]
    if not baseline_rows:
        raise RuntimeError(f"No alpha-extension rows found for lane={winner_lane} alpha={selected_alpha}")

    manifest = _load_json(run_root / "lane_manifest.json")
    lane_cfg = manifest["lanes"][winner_lane]
    variant = str(lane_cfg["variant"])

    contexts: List[ContextSpec] = []
    for row in baseline_rows:
        view = str(row["view"])
        seed = int(row["seed"])
        if view == "full":
            labeled_npz = Path(str(lane_cfg["labeled"]))
            prepass_jsonl = Path(str(lane_cfg["prepass_jsonl"]))
        elif view == "intersection":
            labeled_npz = Path(str(manifest["intersection_labeled"][winner_lane]["path"]))
            prepass_jsonl = Path(str(manifest["intersection_prepass_jsonl"][variant]))
        else:
            raise ValueError(f"Unsupported view {view}")
        val_images = run_root / "splits" / view / winner_lane / f"seed_{seed}.val_images.json"
        contexts.append(
            ContextSpec(
                lane=winner_lane,
                view=view,
                seed=seed,
                variant=variant,
                labeled_npz=labeled_npz,
                prepass_jsonl=prepass_jsonl,
                val_images_file=val_images,
                baseline_f1=float(row["f1"]),
                baseline_precision=float(row["precision"]),
                baseline_recall=float(row["recall"]),
                baseline_delta_vs_union_f1=float(row["delta_vs_union_f1"]),
                baseline_coverage_preservation=float(row["coverage_preservation"]),
            )
        )
    contexts.sort(key=lambda ctx: (ctx.view, ctx.seed))
    return winner_lane, selected_alpha, hp, {"scenario": scenario, "policy": policy, "selection_view": selection_view}, contexts


def _load_classifier_head(classifier_id: str, dataset: str) -> Dict[str, Any]:
    class _ClassifierResolveError(Exception):
        def __init__(self, status_code, detail):
            super().__init__(f"{status_code}:{detail}")
            self.status_code = status_code
            self.detail = detail

    classifier_path = api._resolve_agent_clip_classifier_path_impl(
        classifier_id,
        allowed_root=(api.UPLOAD_ROOT / "classifiers").resolve(),
        allowed_exts=api.CLASSIFIER_ALLOWED_EXTS,
        path_is_within_root_fn=api._path_is_within_root_impl,
        http_exception_cls=_ClassifierResolveError,
    )
    _activate_classifier_runtime(classifier_path, dataset)
    head = api._load_clip_head_from_classifier(classifier_path)
    if not isinstance(head, dict):
        raise RuntimeError(f"Failed to load classifier head for {classifier_id}")
    return head


def _encode_raw_embeddings(
    *,
    crops: Sequence[Image.Image],
    head: Dict[str, Any],
    batch_size: int,
    min_crop_size: int,
    device: str,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for start in range(0, len(crops), max(1, int(batch_size))):
        batch: List[Image.Image] = []
        for crop in crops[start : start + batch_size]:
            w, h = crop.size
            target_w = max(int(min_crop_size), int(w))
            target_h = max(int(min_crop_size), int(h))
            if target_w != w or target_h != h:
                crop = crop.resize((target_w, target_h))
            batch.append(crop)
        feats = api._encode_pil_batch_for_head(batch, head=head, device_override=device)
        if feats is None:
            dim = int(rows[0].shape[0]) if rows else 0
            for _ in batch:
                rows.append(np.zeros((dim,), dtype=np.float32))
            continue
        feats_arr = np.asarray(feats, dtype=np.float32)
        for idx in range(feats_arr.shape[0]):
            rows.append(np.asarray(feats_arr[idx], dtype=np.float32))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    dim = max(int(row.shape[0]) for row in rows)
    out = np.zeros((len(rows), dim), dtype=np.float32)
    for idx, row in enumerate(rows):
        out[idx, : row.shape[0]] = row
    return out


def _build_raw_embedding_cache(
    *,
    labeled_npz: Path,
    dataset: str,
    classifier_head: Dict[str, Any],
    batch_size: int,
    min_crop_size: int,
    device: str,
    cache_path: Path,
) -> Dict[str, Any]:
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return {
            "raw_embeddings": np.asarray(data["raw_embeddings"], dtype=np.float32),
            "meta": np.asarray(data["meta"], dtype=object),
        }

    data = np.load(labeled_npz, allow_pickle=True)
    meta_rows = _parse_meta_rows(data["meta"])
    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, row in enumerate(meta_rows):
        image = str(row.get("image") or "").strip()
        if not image:
            raise RuntimeError(f"Missing image in row {idx} for {labeled_npz}")
        grouped.setdefault(image, []).append((idx, row))

    raw_embeddings: Optional[np.ndarray] = None
    for image_name, entries in grouped.items():
        img_path = _resolve_image_path(dataset, image_name)
        with Image.open(img_path) as img:
            pil_img = img.convert("RGB")
            crops = [pil_img.crop(tuple(entry[1]["bbox_xyxy_px"])) for entry in entries]
        emb = _encode_raw_embeddings(
            crops=crops,
            head=classifier_head,
            batch_size=batch_size,
            min_crop_size=min_crop_size,
            device=device,
        )
        if raw_embeddings is None:
            raw_embeddings = np.zeros((len(meta_rows), emb.shape[1]), dtype=np.float32)
        elif raw_embeddings.shape[1] != emb.shape[1]:
            new_dim = max(int(raw_embeddings.shape[1]), int(emb.shape[1]))
            if new_dim != raw_embeddings.shape[1]:
                resized = np.zeros((raw_embeddings.shape[0], new_dim), dtype=np.float32)
                resized[:, : raw_embeddings.shape[1]] = raw_embeddings
                raw_embeddings = resized
            if new_dim != emb.shape[1]:
                expanded = np.zeros((emb.shape[0], new_dim), dtype=np.float32)
                expanded[:, : emb.shape[1]] = emb
                emb = expanded
        for local_idx, (row_idx, _) in enumerate(entries):
            raw_embeddings[row_idx] = emb[local_idx]
    if raw_embeddings is None:
        raw_embeddings = np.zeros((0, 0), dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        raw_embeddings=raw_embeddings,
        meta=np.asarray(data["meta"], dtype=object),
    )
    return {"raw_embeddings": raw_embeddings, "meta": np.asarray(data["meta"], dtype=object)}


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return np.asarray(arr / norms, dtype=np.float32)


def _sample_for_pca(train_arr: np.ndarray, *, max_rows: int, seed: int) -> np.ndarray:
    if train_arr.shape[0] <= max_rows:
        return train_arr
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(train_arr.shape[0], size=max_rows, replace=False))
    return train_arr[idx]


def _fit_pca(
    train_arr: np.ndarray,
    *,
    n_components: int,
    max_fit_rows: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    sample = _sample_for_pca(train_arr, max_rows=max_fit_rows, seed=seed)
    effective_components = int(min(max(1, n_components), sample.shape[0], sample.shape[1]))
    pca = PCA(n_components=effective_components, svd_solver="randomized", random_state=int(seed))
    pca.fit(sample)
    return pca.components_.astype(np.float32), {
        "effective_components": effective_components,
        "fit_rows": int(sample.shape[0]),
        "input_dim": int(sample.shape[1]),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "mean": pca.mean_.astype(np.float32),
    }


def _apply_pca(
    arr: np.ndarray,
    *,
    components: np.ndarray,
    mean: np.ndarray,
    target_dim: int,
) -> np.ndarray:
    centered = arr - mean.reshape(1, -1)
    projected = np.asarray(centered @ components.T, dtype=np.float32)
    if projected.shape[1] < int(target_dim):
        padded = np.zeros((projected.shape[0], int(target_dim)), dtype=np.float32)
        padded[:, : projected.shape[1]] = projected
        projected = padded
    return projected[:, : int(target_dim)]


def _build_prototype_features(
    *,
    raw_norm: np.ndarray,
    labels: Sequence[str],
    train_mask: np.ndarray,
) -> np.ndarray:
    unique_labels = sorted({str(lbl).strip().lower() for lbl in labels if str(lbl).strip()})
    prototypes: Dict[str, np.ndarray] = {}
    for label in unique_labels:
        idx = np.asarray(
            [bool(train_mask[i]) and str(labels[i]).strip().lower() == label for i in range(len(labels))],
            dtype=bool,
        )
        if not idx.any():
            continue
        proto = np.mean(raw_norm[idx], axis=0)
        norm = float(np.linalg.norm(proto))
        if norm > 0.0:
            proto = np.asarray(proto / norm, dtype=np.float32)
            prototypes[label] = proto
    feat = np.zeros((raw_norm.shape[0], 5), dtype=np.float32)
    if not prototypes:
        return feat
    proto_labels = list(prototypes.keys())
    proto_matrix = np.stack([prototypes[label] for label in proto_labels], axis=0)
    sims = np.asarray(raw_norm @ proto_matrix.T, dtype=np.float32)
    for i, label in enumerate(labels):
        label_name = str(label).strip().lower()
        row = sims[i]
        top_idx = np.argsort(-row)
        top1 = float(row[top_idx[0]]) if top_idx.size else 0.0
        top2 = float(row[top_idx[1]]) if top_idx.size > 1 else 0.0
        own = float(row[proto_labels.index(label_name)]) if label_name in prototypes else 0.0
        other = max(
            [float(row[j]) for j, other_label in enumerate(proto_labels) if other_label != label_name] or [0.0]
        )
        feat[i] = np.asarray(
            [
                own,
                other,
                own - other,
                top1,
                top1 - top2,
            ],
            dtype=np.float32,
        )
    return feat


def _replace_candidate_embedding_block(
    *,
    base_npz: Path,
    output_npz: Path,
    raw_embeddings: np.ndarray,
    train_mask: np.ndarray,
    variant: str,
    pca_fit_rows: int,
    seed: int,
) -> Dict[str, Any]:
    data = np.load(base_npz, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    meta_arr = np.asarray(data["meta"], dtype=object)
    feature_names = [str(name) for name in data["feature_names"]]
    labels = [str(json.loads(str(row)).get("label") or "").strip().lower() for row in meta_arr]
    emb_idx = [idx for idx, name in enumerate(feature_names) if name.startswith("clf_emb_rp::")]
    if not emb_idx:
        raise RuntimeError(f"No candidate embedding block found in {base_npz}")
    keep_idx = [idx for idx in range(X.shape[1]) if idx not in emb_idx]
    keep_names = [feature_names[idx] for idx in keep_idx]
    X_keep = X[:, keep_idx]

    raw_norm = _l2_normalize(raw_embeddings)
    new_blocks: List[np.ndarray] = []
    new_names: List[str] = []
    variant_meta: Dict[str, Any] = {"variant": variant}

    if variant == "raw_l2_native":
        new_blocks.append(raw_norm)
        new_names.extend([f"clf_emb_raw::{idx:04d}" for idx in range(raw_norm.shape[1])])
    elif variant in {"pca_256", "pca_512"}:
        target_dim = int(variant.split("_", 1)[1])
        components, pca_meta = _fit_pca(
            raw_norm[train_mask],
            n_components=target_dim,
            max_fit_rows=pca_fit_rows,
            seed=seed,
        )
        transformed = _apply_pca(
            raw_norm,
            components=components,
            mean=np.asarray(pca_meta["mean"], dtype=np.float32),
            target_dim=target_dim,
        )
        new_blocks.append(transformed)
        new_names.extend([f"clf_emb_pca{target_dim}::{idx:04d}" for idx in range(transformed.shape[1])])
        variant_meta["pca"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in pca_meta.items()
            if k != "mean"
        }
    elif variant == "prototype_margin":
        pass
    elif variant == "raw_l2_native_plus_prototype_margin":
        new_blocks.append(raw_norm)
        new_names.extend([f"clf_emb_raw::{idx:04d}" for idx in range(raw_norm.shape[1])])
    elif variant in {"pca_256_plus_prototype_margin", "pca_512_plus_prototype_margin"}:
        pass
    else:
        raise RuntimeError(f"Unsupported variant {variant}")

    if variant in {
        "prototype_margin",
        "raw_l2_native_plus_prototype_margin",
        "pca_256_plus_prototype_margin",
        "pca_512_plus_prototype_margin",
    }:
        proto_feat = _build_prototype_features(raw_norm=raw_norm, labels=labels, train_mask=train_mask)
        new_blocks.append(proto_feat)
        new_names.extend(
            [
                "clf_proto_same_cosine",
                "clf_proto_other_cosine",
                "clf_proto_margin",
                "clf_proto_top1_cosine",
                "clf_proto_top1_gap",
            ]
        )

    if variant in {"pca_256_plus_prototype_margin", "pca_512_plus_prototype_margin"}:
        target_dim = int(variant.split("_", 1)[1].split("_plus", 1)[0])
        components, pca_meta = _fit_pca(
            raw_norm[train_mask],
            n_components=target_dim,
            max_fit_rows=pca_fit_rows,
            seed=seed,
        )
        transformed = _apply_pca(
            raw_norm,
            components=components,
            mean=np.asarray(pca_meta["mean"], dtype=np.float32),
            target_dim=target_dim,
        )
        new_blocks.insert(0, transformed)
        new_names = [f"clf_emb_pca{target_dim}::{idx:04d}" for idx in range(transformed.shape[1])] + new_names
        variant_meta["pca"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in pca_meta.items()
            if k != "mean"
        }

    X_new = np.concatenate([X_keep] + new_blocks, axis=1)
    feature_names_new = keep_names + new_names
    payload: Dict[str, Any] = {name: data[name] for name in data.files if name not in {"X", "feature_names"}}
    payload["X"] = X_new
    payload["feature_names"] = np.asarray(feature_names_new, dtype=object)
    payload["feature_schema_hash"] = np.asarray(
        compute_feature_schema_hash(
            feature_names_new,
            classifier_classes=[str(x) for x in data.get("classifier_classes", [])],
            labelmap=[str(x) for x in data.get("labelmap", [])],
            context_variant_id=variant,
            variant_config=variant_meta,
        )
    )
    payload["context_variant_id"] = np.asarray(variant)
    payload["parent_feature_npz"] = np.asarray(str(base_npz))
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **payload)
    return {
        "variant": variant,
        "output_npz": str(output_npz),
        "feature_count": int(X_new.shape[1]),
        "variant_meta": variant_meta,
    }


def _load_val_images(path: Path) -> set[str]:
    return {str(x).strip() for x in json.loads(path.read_text(encoding="utf-8")) if str(x).strip()}


def _train_tune_eval_variant(
    *,
    context: ContextSpec,
    labeled_npz: Path,
    dataset: str,
    hp: Dict[str, Any],
    scenario: Dict[str, Any],
    alpha: float,
    policy: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "model"
    model_json = Path(str(model_prefix) + ".json")
    model_meta = Path(str(model_prefix) + ".meta.json")
    if not model_json.exists() or not model_meta.exists():
        cmd = [
            sys.executable,
            "tools/train_ensemble_xgb.py",
            "--input",
            str(labeled_npz),
            "--output",
            str(model_prefix),
            "--seed",
            str(int(context.seed)),
            "--val-ratio",
            "0.2",
            "--max-depth",
            str(int(hp["max_depth"])),
            "--n-estimators",
            str(int(hp["n_estimators"])),
            "--learning-rate",
            str(float(hp["learning_rate"])),
            "--subsample",
            str(float(hp["subsample"])),
            "--colsample-bytree",
            str(float(hp["colsample_bytree"])),
            "--min-child-weight",
            str(float(hp["min_child_weight"])),
            "--gamma",
            str(float(hp["gamma"])),
            "--reg-lambda",
            str(float(hp["reg_lambda"])),
            "--reg-alpha",
            str(float(hp["reg_alpha"])),
            "--tree-method",
            "hist",
            "--max-bin",
            "256",
            "--early-stopping-rounds",
            "50",
            "--threshold-steps",
            "300",
            "--optimize",
            "f1",
            "--target-fp-ratio",
            "0.2",
            "--min-recall",
            "0.6",
            "--per-class",
            "--fixed-val-images",
            str(context.val_images_file),
        ]
        if bool(scenario.get("split_head")):
            cmd.append("--split-head-by-support")
        if bool(scenario.get("sam_quality")):
            cmd += ["--train-sam3-text-quality", "--sam3-text-quality-alpha", str(float(alpha))]
        _run(cmd)

    tune_done = output_dir / "tuned.done"
    if not tune_done.exists():
        _run(
            [
                sys.executable,
                "tools/tune_ensemble_thresholds_xgb.py",
                "--model",
                str(model_json),
                "--meta",
                str(model_meta),
                "--data",
                str(labeled_npz),
                "--dataset",
                str(dataset),
                "--optimize",
                "f1",
                "--target-fp-ratio",
                "0.2",
                "--relax-fp-ratio",
                "0.2",
                "--min-recall",
                "0.6",
                "--steps",
                "300",
                "--eval-iou",
                "0.5",
                "--dedupe-iou",
                "0.75",
                "--scoreless-iou",
                "0.0",
                "--use-val-split",
            ]
        )
        tune_done.write_text("ok\n", encoding="utf-8")

    policy_path = output_dir / "policy.json"
    if not policy_path.exists():
        policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    eval_json = output_dir / "eval.json"
    if not eval_json.exists():
        result = _run(
            [
                sys.executable,
                "tools/eval_ensemble_xgb_dedupe.py",
                "--model",
                str(model_json),
                "--meta",
                str(model_meta),
                "--data",
                str(labeled_npz),
                "--dataset",
                str(dataset),
                "--prepass-jsonl",
                str(context.prepass_jsonl),
                "--eval-iou",
                "0.5",
                "--eval-iou-grid",
                "0.5",
                "--dedupe-iou",
                "0.75",
                "--scoreless-iou",
                "0.0",
                "--use-val-split",
                "--policy-json",
                str(policy_path),
            ],
            capture=True,
        )
        eval_json.write_text(result.stdout.strip() + "\n", encoding="utf-8")
    payload = _load_json(eval_json)
    coverage_ub = float(
        payload.get("coverage_upper_bound", {})
        .get("candidate_all", {})
        .get("recall_upper_bound", 0.0)
    )
    union_f1 = float(
        payload.get("metric_tiers", {})
        .get("post_cluster", {})
        .get("source_attributed", {})
        .get("yolo_rfdetr_union", {})
        .get("f1", 0.0)
    )
    return {
        "precision": float(payload["precision"]),
        "recall": float(payload["recall"]),
        "f1": float(payload["f1"]),
        "delta_vs_union_f1": float(payload["f1"]) - union_f1,
        "coverage_preservation": (float(payload["recall"]) / coverage_ub) if coverage_ub > 0.0 else 0.0,
        "eval_json": str(eval_json),
        "model_json": str(model_json),
        "model_meta": str(model_meta),
    }


def _pilot_pass(row: Dict[str, Any], ctx: ContextSpec) -> bool:
    return (
        float(row["f1"]) - float(ctx.baseline_f1) >= 0.002
        and float(row["coverage_preservation"]) >= float(ctx.baseline_coverage_preservation) - 0.005
    )


def _summarize_variant_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    summary_rows: List[Dict[str, Any]] = []
    for variant, group in grouped.items():
        summary_rows.append(
            {
                "variant": variant,
                "count": len(group),
                "mean_precision": statistics.mean(float(x["precision"]) for x in group),
                "mean_recall": statistics.mean(float(x["recall"]) for x in group),
                "mean_f1": statistics.mean(float(x["f1"]) for x in group),
                "mean_delta_vs_baseline_f1": statistics.mean(float(x["f1"]) - float(x["baseline_f1"]) for x in group),
                "mean_delta_vs_union_f1": statistics.mean(float(x["delta_vs_union_f1"]) for x in group),
                "mean_coverage_preservation": statistics.mean(float(x["coverage_preservation"]) for x in group),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            -float(row["mean_f1"]),
            -float(row["mean_delta_vs_baseline_f1"]),
            -float(row["mean_coverage_preservation"]),
        )
    )
    return {"ranked_variants": summary_rows}


def _write_report(path: Path, *, pilot_summary: Dict[str, Any], full_summary: Dict[str, Any]) -> None:
    lines = [
        "# Post-Run Candidate Embedding Variant Experiment",
        "",
        f"- Generated UTC: `{_ts()}`",
        "",
        "## Pilot ranking",
        "",
        "| variant | mean F1 | delta vs baseline F1 | coverage preservation | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in pilot_summary.get("ranked_variants", []):
        lines.append(
            f"| {row['variant']} | {row['mean_f1']:.4f} | {row['mean_delta_vs_baseline_f1']:.4f} | "
            f"{row['mean_coverage_preservation']:.4f} | {row['count']} |"
        )
    if full_summary.get("ranked_variants"):
        lines += [
            "",
            "## Full matrix ranking",
            "",
            "| variant | mean F1 | delta vs baseline F1 | coverage preservation | n |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in full_summary.get("ranked_variants", []):
            lines.append(
                f"| {row['variant']} | {row['mean_f1']:.4f} | {row['mean_delta_vs_baseline_f1']:.4f} | "
                f"{row['mean_coverage_preservation']:.4f} | {row['count']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue winner-only candidate embedding variants after alpha extension.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--alpha-root", default="")
    parser.add_argument("--dataset", default="qwen_dataset")
    parser.add_argument("--classifier-id", default="uploads/classifiers/DinoV3_best_model_large.pkl")
    parser.add_argument("--selection-view", default="intersection", choices=["intersection", "full"])
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-crop-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca-fit-rows", type=int, default=100000)
    parser.add_argument("--pilot-seed", type=int, default=42)
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    run_root = (ROOT / args.run_root).resolve()
    alpha_root = (ROOT / args.alpha_root).resolve() if args.alpha_root else run_root / "postrun_alpha_extension"
    if args.wait:
        _wait_for_file(run_root / "final_default_recipe.json", poll_seconds=int(args.poll_seconds))
        _wait_for_file(alpha_root / "results_ranked.json", poll_seconds=int(args.poll_seconds))
        _wait_for_file(alpha_root / "results_raw.json", poll_seconds=int(args.poll_seconds))

    output_root = Path(args.output_root).resolve() if args.output_root else run_root / "postrun_candidate_embedding_experiment"
    output_root.mkdir(parents=True, exist_ok=True)

    winner_lane, selected_alpha, hp, lane_config, contexts = _winner_baseline_contexts(
        run_root=run_root,
        alpha_root=alpha_root,
        selection_view=str(args.selection_view),
    )
    scenario = lane_config["scenario"]
    policy = lane_config["policy"]
    classifier_head = _load_classifier_head(str(args.classifier_id), str(args.dataset))

    context_by_key = {(ctx.view, ctx.seed): ctx for ctx in contexts}
    pilot_ctx = context_by_key[(str(args.selection_view), int(args.pilot_seed))]

    variants = [
        "raw_l2_native",
        "pca_256",
        "pca_512",
        "prototype_margin",
        "raw_l2_native_plus_prototype_margin",
        "pca_256_plus_prototype_margin",
        "pca_512_plus_prototype_margin",
    ]

    raw_cache_by_view: Dict[str, Dict[str, Any]] = {}
    for view in sorted({ctx.view for ctx in contexts}):
        ctx = next(item for item in contexts if item.view == view)
        cache_path = output_root / "raw_cache" / f"{winner_lane}_{view}.raw_embeddings.npz"
        raw_cache_by_view[view] = _build_raw_embedding_cache(
            labeled_npz=ctx.labeled_npz,
            dataset=str(args.dataset),
            classifier_head=classifier_head,
            batch_size=int(args.batch_size),
            min_crop_size=int(args.min_crop_size),
            device=str(args.device),
            cache_path=cache_path,
        )

    pilot_val_images = _load_val_images(pilot_ctx.val_images_file)
    pilot_meta = _parse_meta_rows(np.asarray(raw_cache_by_view[pilot_ctx.view]["meta"], dtype=object))
    pilot_train_mask = np.asarray(
        [str(row.get("image") or "") not in pilot_val_images for row in pilot_meta],
        dtype=bool,
    )

    pilot_rows: List[Dict[str, Any]] = []
    promoted: List[str] = []
    for variant in variants:
        variant_npz = output_root / "variants" / variant / pilot_ctx.view / f"seed_{pilot_ctx.seed}" / "labeled.npz"
        _replace_candidate_embedding_block(
            base_npz=pilot_ctx.labeled_npz,
            output_npz=variant_npz,
            raw_embeddings=np.asarray(raw_cache_by_view[pilot_ctx.view]["raw_embeddings"], dtype=np.float32),
            train_mask=pilot_train_mask,
            variant=variant,
            pca_fit_rows=int(args.pca_fit_rows),
            seed=int(pilot_ctx.seed),
        )
        result = _train_tune_eval_variant(
            context=pilot_ctx,
            labeled_npz=variant_npz,
            dataset=str(args.dataset),
            hp=hp,
            scenario=scenario,
            alpha=selected_alpha,
            policy=policy,
            output_dir=output_root / "pilot" / variant,
        )
        row = {
            "variant": variant,
            "view": pilot_ctx.view,
            "seed": int(pilot_ctx.seed),
            "baseline_f1": float(pilot_ctx.baseline_f1),
            "baseline_precision": float(pilot_ctx.baseline_precision),
            "baseline_recall": float(pilot_ctx.baseline_recall),
            "baseline_delta_vs_union_f1": float(pilot_ctx.baseline_delta_vs_union_f1),
            "baseline_coverage_preservation": float(pilot_ctx.baseline_coverage_preservation),
            **result,
        }
        pilot_rows.append(row)
        if _pilot_pass(row, pilot_ctx):
            promoted.append(variant)

    pilot_summary = _summarize_variant_rows(pilot_rows)
    promoted = [row["variant"] for row in pilot_summary.get("ranked_variants", []) if row["variant"] in promoted][:2]

    full_rows: List[Dict[str, Any]] = []
    for variant in promoted:
        for ctx in contexts:
            val_images = _load_val_images(ctx.val_images_file)
            meta_rows = _parse_meta_rows(np.asarray(raw_cache_by_view[ctx.view]["meta"], dtype=object))
            train_mask = np.asarray(
                [str(row.get("image") or "") not in val_images for row in meta_rows],
                dtype=bool,
            )
            variant_npz = output_root / "variants" / variant / ctx.view / f"seed_{ctx.seed}" / "labeled.npz"
            if not variant_npz.exists():
                _replace_candidate_embedding_block(
                    base_npz=ctx.labeled_npz,
                    output_npz=variant_npz,
                    raw_embeddings=np.asarray(raw_cache_by_view[ctx.view]["raw_embeddings"], dtype=np.float32),
                    train_mask=train_mask,
                    variant=variant,
                    pca_fit_rows=int(args.pca_fit_rows),
                    seed=int(ctx.seed),
                )
            result = _train_tune_eval_variant(
                context=ctx,
                labeled_npz=variant_npz,
                dataset=str(args.dataset),
                hp=hp,
                scenario=scenario,
                alpha=selected_alpha,
                policy=policy,
                output_dir=output_root / "full_matrix" / variant / ctx.view / f"seed_{ctx.seed}",
            )
            full_rows.append(
                {
                    "variant": variant,
                    "view": ctx.view,
                    "seed": int(ctx.seed),
                    "baseline_f1": float(ctx.baseline_f1),
                    "baseline_precision": float(ctx.baseline_precision),
                    "baseline_recall": float(ctx.baseline_recall),
                    "baseline_delta_vs_union_f1": float(ctx.baseline_delta_vs_union_f1),
                    "baseline_coverage_preservation": float(ctx.baseline_coverage_preservation),
                    **result,
                }
            )

    full_summary = _summarize_variant_rows(full_rows) if full_rows else {"ranked_variants": []}
    raw_payload = {
        "generated_utc": _ts(),
        "run_root": str(run_root),
        "alpha_root": str(alpha_root),
        "winner_lane": winner_lane,
        "selected_alpha": float(selected_alpha),
        "scenario": scenario,
        "policy": policy,
        "pilot_rows": pilot_rows,
        "pilot_summary": pilot_summary,
        "promoted_variants": promoted,
        "full_rows": full_rows,
        "full_summary": full_summary,
    }
    _write_json(output_root / "results_raw.json", raw_payload)
    _write_json(output_root / "results_ranked.json", {"pilot": pilot_summary, "full": full_summary, "promoted_variants": promoted})
    _write_report(output_root / "report.md", pilot_summary=pilot_summary, full_summary=full_summary)
    print(json.dumps({"status": "completed", "winner_lane": winner_lane, "selected_alpha": selected_alpha, "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
