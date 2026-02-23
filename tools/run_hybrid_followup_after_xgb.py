#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import tune_ensemble_thresholds_xgb as tune_xgb


@dataclass
class VariantConfig:
    name: str
    dataset: str
    labeled_npz: Path
    model_json: Path
    model_meta: Path
    eval_json: Path


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def _clamp_prob(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32), 1e-6, 1.0 - 1e-6)


def _logit(arr: np.ndarray) -> np.ndarray:
    p = _clamp_prob(arr)
    return np.log(p / (1.0 - p))


def _normalize_sources(row: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    primary = str(row.get("score_source") or row.get("source") or "").strip().lower()
    if primary:
        out.add(primary)
    raw_sources = row.get("source_list")
    if isinstance(raw_sources, (list, tuple, set)):
        for src in raw_sources:
            name = str(src or "").strip().lower()
            if name:
                out.add(name)
    raw_scores = row.get("score_by_source")
    if isinstance(raw_scores, dict):
        for src in raw_scores.keys():
            name = str(src or "").strip().lower()
            if name:
                out.add(name)
    return out


def _support_flags(row: Dict[str, Any]) -> List[float]:
    sources = _normalize_sources(row)
    return [
        1.0 if "yolo" in sources else 0.0,
        1.0 if "rfdetr" in sources else 0.0,
        1.0 if "sam3_text" in sources else 0.0,
        1.0 if "sam3_similarity" in sources else 0.0,
    ]


def _load_fixed_val_images(path: Path) -> Set[str]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return set()
    if raw.lstrip().startswith("["):
        parsed = json.loads(raw)
        return {str(x) for x in parsed if str(x)}
    out: Set[str] = set()
    for line in raw.splitlines():
        item = line.strip()
        if item:
            out.add(item)
    return out


def _is_current_run_active(run_dir: Path) -> bool:
    cmd = (
        "pgrep -af 'run_emb1024_calibration.sh|label_candidates_iou90.py|"
        "train_ensemble_xgb.py|tune_ensemble_thresholds_xgb.py|eval_ensemble_xgb_dedupe.py' || true"
    )
    result = subprocess.run(["/bin/bash", "-lc", cmd], capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        if str(run_dir) in line and "pgrep -af" not in line:
            return True
    return False


def _wait_for_current_xgb(run_dir: Path, variants: Sequence[VariantConfig], timeout_sec: int, poll_sec: int) -> None:
    deadline = time.time() + max(60, int(timeout_sec))
    required = []
    for v in variants:
        required.extend([v.labeled_npz, v.model_json, v.model_meta, v.eval_json])

    while True:
        exists = all(path.exists() for path in required)
        active = _is_current_run_active(run_dir)
        if exists and not active:
            _log("Detected completed XGB control run artifacts for both variants.")
            return
        if time.time() >= deadline:
            missing = [str(path) for path in required if not path.exists()]
            raise RuntimeError(
                f"Timed out waiting for current run completion. active={active} missing={missing[:8]}"
            )
        _log(f"Waiting for current run completion... active={active} complete_files={exists}")
        time.sleep(max(5, int(poll_sec)))


def _dense_feature_indices(feature_names: Sequence[str]) -> List[int]:
    idx: List[int] = []
    for i, name in enumerate(feature_names):
        key = str(name)
        if key.startswith("clf_emb_rp::") or key.startswith("embed_proj_") or key.startswith("clf_prob::"):
            idx.append(i)
    return idx


def _struct_feature_indices(feature_names: Sequence[str], dense_idx: Sequence[int]) -> List[int]:
    dense_set = set(int(i) for i in dense_idx)
    return [i for i in range(len(feature_names)) if i not in dense_set]


def _train_dense_lr(
    X_dense: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_dense[train_mask])
    clf = LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y[train_mask])
    probs = clf.predict_proba(scaler.transform(X_dense))[:, 1].astype(np.float32)
    return probs, {"model": "logreg", "coef_dim": int(clf.coef_.shape[1])}


def _train_dense_mlp(
    X_dense: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_dense[train_mask])
    clf = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=40,
        early_stopping=True,
        n_iter_no_change=6,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        clf.fit(X_train, y[train_mask])
    probs = clf.predict_proba(scaler.transform(X_dense))[:, 1].astype(np.float32)
    return probs, {"model": "mlp", "n_iter": int(getattr(clf, "n_iter_", 0))}


def _train_struct_xgb(
    X_struct: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y_train = y[train_mask]
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    scale_pos = float(neg / max(1, pos))
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=8,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        gamma=0.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        max_bin=256,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=0,
    )
    clf.fit(X_struct[train_mask], y_train)
    probs = clf.predict_proba(X_struct)[:, 1].astype(np.float32)
    return probs, {"model": "xgb", "scale_pos_weight": scale_pos}


def _train_blender(
    dense_probs: np.ndarray,
    struct_probs: np.ndarray,
    source_flags: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    feat = np.column_stack(
        [
            _logit(dense_probs),
            _logit(struct_probs),
            source_flags.astype(np.float32),
        ]
    ).astype(np.float32)
    clf = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(feat[train_mask], y[train_mask])
    probs = clf.predict_proba(feat)[:, 1].astype(np.float32)
    return probs, {"model": "logreg_blender", "feat_dim": int(feat.shape[1])}


def _load_base_thresholds(meta_path: Path) -> Tuple[Dict[str, float], float]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _, thresholds = tune_xgb._threshold_source(meta)
    default_threshold = float(meta.get("calibrated_threshold") or 0.5)
    return thresholds, default_threshold


def _build_label_rows(
    probs: np.ndarray,
    rows: Sequence[Dict[str, Any]],
    mask: np.ndarray,
    *,
    name_to_cat: Dict[str, int],
) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, List[float]], List[str]]:
    label_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    label_probs: Dict[str, List[float]] = {}
    eval_images: Set[str] = set()
    for idx, row in enumerate(rows):
        if not bool(mask[idx]):
            continue
        image = str(row.get("image") or "")
        label = str(row.get("label") or "").strip().lower()
        bbox = row.get("bbox_xyxy_px")
        if (
            not image
            or not label
            or label not in name_to_cat
            or not isinstance(bbox, (list, tuple))
            or len(bbox) < 4
        ):
            continue
        eval_images.add(image)
        label_rows.setdefault(label, {}).setdefault(image, []).append(
            {
                "label": label,
                "bbox_xyxy_px": [float(v) for v in bbox[:4]],
                "prob": float(probs[idx]),
                "score_source": str(row.get("score_source") or row.get("source") or "unknown").strip().lower(),
            }
        )
        label_probs.setdefault(label, []).append(float(probs[idx]))
    return label_rows, label_probs, sorted(eval_images)


def _tune_thresholds_and_eval(
    *,
    probs: np.ndarray,
    rows: Sequence[Dict[str, Any]],
    val_mask: np.ndarray,
    dataset: str,
    base_thresholds: Dict[str, float],
    default_threshold: float,
    optimize: str,
    target_fp_ratio: float,
    min_recall: float,
    steps: int,
    dedupe_iou: float,
    eval_iou: float,
    scoreless_iou: float,
) -> Dict[str, Any]:
    name_to_cat_all, gt_all = tune_xgb._load_gt_boxes(dataset, sorted({str(r.get("image") or "") for r in rows}))
    label_rows, label_probs, eval_images = _build_label_rows(
        probs,
        rows,
        val_mask,
        name_to_cat=name_to_cat_all,
    )
    name_to_cat = dict(name_to_cat_all)
    gt_by_image = {image: gt_all.get(image, {}) for image in eval_images if image in gt_all}

    tuned_thresholds: Dict[str, float] = {}
    for label, cat_id in sorted(name_to_cat.items()):
        rows_by_image = label_rows.get(label, {})
        base_thr = float(base_thresholds.get(label, default_threshold))
        grid = tune_xgb._build_threshold_grid(
            label_probs.get(label, []),
            base_threshold=base_thr,
            steps=int(steps),
        )
        scored: List[Dict[str, Any]] = []
        for thr in grid:
            metrics = tune_xgb._evaluate_label_threshold(
                rows_by_image,
                gt_by_image,
                cat_id=int(cat_id),
                threshold=float(thr),
                dedupe_iou=float(dedupe_iou),
                eval_iou=float(eval_iou),
                scoreless_iou=float(scoreless_iou),
            )
            scored.append({"threshold": float(thr), "metrics": metrics})

        strict = tune_xgb._pick_best(
            scored,
            optimize=str(optimize),
            require_fp_ratio=float(target_fp_ratio),
            require_recall=float(min_recall),
        )
        fp_only = tune_xgb._pick_best(
            scored,
            optimize=str(optimize),
            require_fp_ratio=float(target_fp_ratio),
            require_recall=None,
        )
        unconstrained = tune_xgb._pick_best(
            scored,
            optimize=str(optimize),
            require_fp_ratio=None,
            require_recall=None,
        )
        chosen = strict or fp_only or unconstrained
        if chosen is None:
            chosen = {"threshold": base_thr}
        tuned_thresholds[label] = float(chosen["threshold"])

    tuned_global = tune_xgb._evaluate_global_thresholds(
        label_rows,
        gt_by_image,
        name_to_cat=name_to_cat,
        threshold_by_label=tuned_thresholds,
        default_threshold=float(default_threshold),
        dedupe_iou=float(dedupe_iou),
        eval_iou=float(eval_iou),
        scoreless_iou=float(scoreless_iou),
    )
    base_global = tune_xgb._evaluate_global_thresholds(
        label_rows,
        gt_by_image,
        name_to_cat=name_to_cat,
        threshold_by_label=base_thresholds,
        default_threshold=float(default_threshold),
        dedupe_iou=float(dedupe_iou),
        eval_iou=float(eval_iou),
        scoreless_iou=float(scoreless_iou),
    )

    final_thresholds = dict(tuned_thresholds)
    blend_alpha = 0.0
    final_global = tuned_global
    if target_fp_ratio >= 0 and tuned_global.get("tp", 0) > 0 and tuned_global.get("fp_ratio", 0.0) > target_fp_ratio:
        best_candidate = None
        for alpha in np.linspace(0.05, 1.0, 20):
            blended = {}
            for label, tuned_thr in tuned_thresholds.items():
                base_thr = float(base_thresholds.get(label, default_threshold))
                blended[label] = float(alpha * base_thr + (1.0 - alpha) * tuned_thr)
            metrics = tune_xgb._evaluate_global_thresholds(
                label_rows,
                gt_by_image,
                name_to_cat=name_to_cat,
                threshold_by_label=blended,
                default_threshold=float(default_threshold),
                dedupe_iou=float(dedupe_iou),
                eval_iou=float(eval_iou),
                scoreless_iou=float(scoreless_iou),
            )
            if metrics.get("tp", 0) <= 0:
                continue
            key = tune_xgb._score_key(metrics, str(optimize), float(alpha))
            if float(metrics.get("fp_ratio", 1e9)) <= float(target_fp_ratio):
                if best_candidate is None or key > best_candidate["key"]:
                    best_candidate = {
                        "alpha": float(alpha),
                        "thresholds": blended,
                        "metrics": metrics,
                        "key": key,
                    }
        if best_candidate is not None:
            blend_alpha = float(best_candidate["alpha"])
            final_thresholds = dict(best_candidate["thresholds"])
            final_global = dict(best_candidate["metrics"])

    return {
        "base_metrics": base_global,
        "tuned_metrics": tuned_global,
        "final_metrics": final_global,
        "blend_alpha": float(blend_alpha),
        "thresholds": final_thresholds,
        "num_eval_images": len(eval_images),
    }


def _load_rows_and_masks(
    labeled_npz: Path,
    val_images: Set[str],
) -> Dict[str, Any]:
    data = np.load(labeled_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    rows = [json.loads(str(raw)) for raw in data["meta"]]
    images = np.asarray([str(row.get("image") or "") for row in rows], dtype=object)
    val_mask = np.asarray([img in val_images for img in images], dtype=bool)
    train_mask = ~val_mask
    source_flags = np.asarray([_support_flags(row) for row in rows], dtype=np.float32)
    return {
        "X": X,
        "y": y,
        "rows": rows,
        "feature_names": feature_names,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "source_flags": source_flags,
    }


def _run_hybrid_variant(
    *,
    variant: VariantConfig,
    payload: Dict[str, Any],
    method: str,
    optimize: str,
    target_fp_ratio: float,
    min_recall: float,
    steps: int,
    dedupe_iou: float,
    eval_iou: float,
    scoreless_iou: float,
) -> Dict[str, Any]:
    X = payload["X"]
    y = payload["y"]
    rows = payload["rows"]
    feature_names = payload["feature_names"]
    train_mask = payload["train_mask"]
    val_mask = payload["val_mask"]
    source_flags = payload["source_flags"]

    dense_idx = _dense_feature_indices(feature_names)
    struct_idx = _struct_feature_indices(feature_names, dense_idx)
    if not dense_idx:
        raise RuntimeError(f"{variant.name}: no dense classifier features found.")
    if not struct_idx:
        raise RuntimeError(f"{variant.name}: no structured features found.")

    X_dense = X[:, dense_idx]
    X_struct = X[:, struct_idx]

    if method == "hybrid_lr_xgb_blend":
        dense_probs, dense_info = _train_dense_lr(X_dense, y, train_mask)
    elif method == "hybrid_mlp_xgb_blend":
        dense_probs, dense_info = _train_dense_mlp(X_dense, y, train_mask)
    else:
        raise ValueError(f"Unknown method: {method}")

    struct_probs, struct_info = _train_struct_xgb(X_struct, y, train_mask)
    blend_probs, blend_info = _train_blender(dense_probs, struct_probs, source_flags, y, train_mask)

    base_thresholds, default_threshold = _load_base_thresholds(variant.model_meta)
    tuned = _tune_thresholds_and_eval(
        probs=blend_probs,
        rows=rows,
        val_mask=val_mask,
        dataset=variant.dataset,
        base_thresholds=base_thresholds,
        default_threshold=default_threshold,
        optimize=optimize,
        target_fp_ratio=target_fp_ratio,
        min_recall=min_recall,
        steps=steps,
        dedupe_iou=dedupe_iou,
        eval_iou=eval_iou,
        scoreless_iou=scoreless_iou,
    )

    return {
        "method": method,
        "variant": variant.name,
        "dense_feature_dim": len(dense_idx),
        "struct_feature_dim": len(struct_idx),
        "train_rows": int(np.count_nonzero(train_mask)),
        "val_rows": int(np.count_nonzero(val_mask)),
        "positive_rate_train": float(np.mean(y[train_mask])) if np.count_nonzero(train_mask) else 0.0,
        "dense_info": dense_info,
        "struct_info": struct_info,
        "blend_info": blend_info,
        **tuned,
    }


def _load_xgb_control_metrics(eval_json_path: Path) -> Dict[str, Any]:
    payload = json.loads(eval_json_path.read_text(encoding="utf-8"))
    return {
        "tp": int(payload.get("tp", 0)),
        "fp": int(payload.get("fp", 0)),
        "fn": int(payload.get("fn", 0)),
        "precision": float(payload.get("precision", 0.0)),
        "recall": float(payload.get("recall", 0.0)),
        "f1": float(payload.get("f1", 0.0)),
    }


def _summary_row(method: str, variant: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": method,
        "variant": variant,
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "tp": int(metrics.get("tp", 0)),
        "fp": int(metrics.get("fp", 0)),
        "fn": int(metrics.get("fn", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for current XGB run, then run hybrid follow-up methods and compare metrics."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory of current emb1024 XGB job.")
    parser.add_argument(
        "--fixed-val-images",
        default="uploads/calibration_jobs/fixed_val_qwen_dataset_2000_images.json",
        help="Fixed validation image list.",
    )
    parser.add_argument("--timeout-sec", type=int, default=21600, help="Wait timeout for current run.")
    parser.add_argument("--poll-sec", type=int, default=30, help="Polling interval while waiting.")
    parser.add_argument("--optimize", default="f1", choices=["f1", "recall", "tp"])
    parser.add_argument("--target-fp-ratio", type=float, default=0.2)
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dedupe-iou", type=float, default=0.75)
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--scoreless-iou", type=float, default=0.0)
    parser.add_argument("--output-json", default=None, help="Optional explicit output JSON path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir_not_found:{run_dir}")

    variants = [
        VariantConfig(
            name="nonwindow_20c8",
            dataset="qwen_dataset",
            labeled_npz=run_dir / "nonwindow_20c8.labeled.npz",
            model_json=run_dir / "nonwindow_20c8.json",
            model_meta=run_dir / "nonwindow_20c8.meta.json",
            eval_json=run_dir / "nonwindow_20c8.eval.json",
        ),
        VariantConfig(
            name="window_ceab",
            dataset="qwen_dataset",
            labeled_npz=run_dir / "window_ceab.labeled.npz",
            model_json=run_dir / "window_ceab.json",
            model_meta=run_dir / "window_ceab.meta.json",
            eval_json=run_dir / "window_ceab.eval.json",
        ),
    ]

    _log("Waiting for current XGB run to complete before starting follow-up.")
    _wait_for_current_xgb(
        run_dir=run_dir,
        variants=variants,
        timeout_sec=int(args.timeout_sec),
        poll_sec=int(args.poll_sec),
    )

    val_images = _load_fixed_val_images(Path(args.fixed_val_images))
    if not val_images:
        raise SystemExit("fixed_val_images_empty")

    payload_by_variant: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        _log(f"Loading labeled data for {variant.name}")
        payload_by_variant[variant.name] = _load_rows_and_masks(variant.labeled_npz, val_images)

    results: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {
            "optimize": args.optimize,
            "target_fp_ratio": float(args.target_fp_ratio),
            "min_recall": float(args.min_recall),
            "steps": int(args.steps),
            "dedupe_iou": float(args.dedupe_iou),
            "eval_iou": float(args.eval_iou),
            "scoreless_iou": float(args.scoreless_iou),
        },
        "variants": {},
        "summary_rows": [],
    }

    for variant in variants:
        _log(f"Collecting XGB control metrics for {variant.name}")
        xgb_metrics = _load_xgb_control_metrics(variant.eval_json)
        results["variants"][variant.name] = {
            "xgb_control": xgb_metrics,
        }
        results["summary_rows"].append(_summary_row("xgb_control", variant.name, xgb_metrics))

    for method in ("hybrid_lr_xgb_blend", "hybrid_mlp_xgb_blend"):
        for variant in variants:
            _log(f"Running {method} on {variant.name}")
            out = _run_hybrid_variant(
                variant=variant,
                payload=payload_by_variant[variant.name],
                method=method,
                optimize=str(args.optimize),
                target_fp_ratio=float(args.target_fp_ratio),
                min_recall=float(args.min_recall),
                steps=int(args.steps),
                dedupe_iou=float(args.dedupe_iou),
                eval_iou=float(args.eval_iou),
                scoreless_iou=float(args.scoreless_iou),
            )
            results["variants"][variant.name][method] = out
            results["summary_rows"].append(
                _summary_row(method, variant.name, out.get("final_metrics", {}))
            )

    output_json = Path(args.output_json).resolve() if args.output_json else run_dir / "hybrid_followup_report.json"
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _log(f"Wrote follow-up report: {output_json}")
    print(json.dumps({"report": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
