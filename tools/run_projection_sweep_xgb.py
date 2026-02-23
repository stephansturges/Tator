#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class Variant:
    name: str
    dataset: str
    labeled_npz: Path
    prepass_jsonl: Path
    baseline_eval_json: Path


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def _parse_dims(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        dim = int(token)
        if dim <= 0:
            continue
        out.append(dim)
    return sorted(set(out))


def _run_cmd(cmd: Sequence[str], *, stdout_path: Path = None, stderr_to_stdout: bool = False) -> None:
    if stdout_path is None:
        subprocess.run(list(cmd), check=True)
        return
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            list(cmd),
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT if stderr_to_stdout else None,
        )


def _identify_feature_blocks(feature_names: Sequence[str]) -> Tuple[List[int], List[int], List[int]]:
    prob_idx: List[int] = []
    emb_idx: List[int] = []
    other_idx: List[int] = []
    for i, name in enumerate(feature_names):
        key = str(name)
        if key.startswith("clf_prob::"):
            prob_idx.append(i)
        elif key.startswith("clf_emb_rp::") or key.startswith("embed_proj_"):
            emb_idx.append(i)
        else:
            other_idx.append(i)
    return prob_idx, emb_idx, other_idx


def _projection_pca(
    emb: np.ndarray,
    train_mask: np.ndarray,
    *,
    dim: int,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    emb_train = scaler.fit_transform(emb[train_mask])
    emb_all = scaler.transform(emb)
    pca = PCA(n_components=int(dim), svd_solver="randomized", random_state=int(seed))
    pca.fit(emb_train)
    return pca.transform(emb_all).astype(np.float32)


def _projection_jl(
    emb: np.ndarray,
    train_mask: np.ndarray,
    *,
    dim: int,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    _ = scaler.fit_transform(emb[train_mask])
    emb_all = scaler.transform(emb).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    mat = rng.standard_normal((emb_all.shape[1], int(dim)), dtype=np.float32)
    mat = mat / np.sqrt(float(max(1, emb_all.shape[1])))
    return np.asarray(emb_all @ mat, dtype=np.float32)


def _build_reduced_npz(
    *,
    src_npz: Path,
    dst_npz: Path,
    method: str,
    dim: int,
    seed: int,
    fixed_val_images: Path,
) -> Dict[str, Any]:
    data = np.load(src_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    meta = [json.loads(str(x)) for x in data["meta"]]
    images = np.asarray([str(row.get("image") or "") for row in meta], dtype=object)
    val_set = set(json.loads(fixed_val_images.read_text(encoding="utf-8")))
    val_mask = np.asarray([img in val_set for img in images], dtype=bool)
    train_mask = ~val_mask
    if not bool(np.any(train_mask)):
        raise RuntimeError("No train rows available for projection fit.")

    prob_idx, emb_idx, other_idx = _identify_feature_blocks(feature_names)
    if not emb_idx:
        raise RuntimeError("No embedding feature block found in source NPZ.")

    emb = X[:, emb_idx]
    if int(dim) >= emb.shape[1]:
        emb_reduced = emb.astype(np.float32)
        used_dim = emb.shape[1]
    elif method == "pca":
        emb_reduced = _projection_pca(emb, train_mask, dim=int(dim), seed=int(seed))
        used_dim = int(dim)
    elif method == "jl":
        emb_reduced = _projection_jl(emb, train_mask, dim=int(dim), seed=int(seed))
        used_dim = int(dim)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    blocks = []
    names: List[str] = []
    if prob_idx:
        blocks.append(X[:, prob_idx])
        names.extend([feature_names[i] for i in prob_idx])
    blocks.append(emb_reduced)
    names.extend([f"clf_emb_{method}::{i:04d}" for i in range(int(used_dim))])
    if other_idx:
        blocks.append(X[:, other_idx])
        names.extend([feature_names[i] for i in other_idx])
    X_reduced = np.concatenate(blocks, axis=1).astype(np.float32)

    payload: Dict[str, Any] = {k: data[k] for k in data.files if k not in {"X", "feature_names"}}
    payload["X"] = X_reduced
    payload["feature_names"] = np.asarray(names, dtype=object)
    payload["projection_method"] = np.asarray(method, dtype=object)
    payload["projection_dim"] = np.asarray(int(used_dim), dtype=np.int64)
    payload["projection_seed"] = np.asarray(int(seed), dtype=np.int64)
    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_npz, **payload)

    return {
        "rows": int(X_reduced.shape[0]),
        "cols": int(X_reduced.shape[1]),
        "prob_dim": int(len(prob_idx)),
        "embed_dim": int(used_dim),
        "other_dim": int(len(other_idx)),
    }


def _read_metrics(eval_json: Path) -> Dict[str, Any]:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    return {
        "tp": int(payload.get("tp", 0)),
        "fp": int(payload.get("fp", 0)),
        "fn": int(payload.get("fn", 0)),
        "precision": float(payload.get("precision", 0.0)),
        "recall": float(payload.get("recall", 0.0)),
        "f1": float(payload.get("f1", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA/JL projection sweep on existing labeled 1024-d features.")
    parser.add_argument("--run-dir", required=True, help="Run dir containing *_*.labeled.npz control artifacts.")
    parser.add_argument(
        "--fixed-val-images",
        default="uploads/calibration_jobs/fixed_val_qwen_dataset_2000_images.json",
        help="Fixed validation image list JSON.",
    )
    parser.add_argument("--dims", default="64,128,256,512,1024", help="Comma-separated embedding dims.")
    parser.add_argument("--methods", default="pca,jl", help="Comma-separated methods: pca,jl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fp-ratio", type=float, default=0.2)
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--dedupe-iou", type=float, default=0.75)
    parser.add_argument("--scoreless-iou", type=float, default=0.0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    fixed_val_images = Path(args.fixed_val_images).resolve()
    dims = _parse_dims(args.dims)
    methods = [m.strip().lower() for m in str(args.methods or "").split(",") if m.strip()]
    methods = [m for m in methods if m in {"pca", "jl"}]
    if not dims:
        raise SystemExit("No projection dims provided.")
    if not methods:
        raise SystemExit("No valid methods provided (pca,jl).")

    variants = [
        Variant(
            name="nonwindow_20c8",
            dataset="qwen_dataset",
            labeled_npz=run_dir / "nonwindow_20c8.labeled.npz",
            prepass_jsonl=Path(
                "uploads/calibration_cache/features_backfill/20c8d44d69f51b2ffe528fb500e75672a306f67d/prepass.jsonl"
            ).resolve(),
            baseline_eval_json=run_dir / "nonwindow_20c8.eval.json",
        ),
        Variant(
            name="window_ceab",
            dataset="qwen_dataset",
            labeled_npz=run_dir / "window_ceab.labeled.npz",
            prepass_jsonl=Path(
                "uploads/calibration_cache/features_backfill/ceab65b2bff24d316ca5f858addaffed8abfdb11/prepass.jsonl"
            ).resolve(),
            baseline_eval_json=run_dir / "window_ceab.eval.json",
        ),
    ]

    for variant in variants:
        if not variant.labeled_npz.exists():
            raise SystemExit(f"Missing labeled NPZ: {variant.labeled_npz}")
        if not variant.baseline_eval_json.exists():
            raise SystemExit(f"Missing baseline eval json: {variant.baseline_eval_json}")

    out_root = run_dir / "projection_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dims": dims,
        "methods": methods,
        "params": {
            "seed": int(args.seed),
            "target_fp_ratio": float(args.target_fp_ratio),
            "min_recall": float(args.min_recall),
            "steps": int(args.steps),
            "eval_iou": float(args.eval_iou),
            "dedupe_iou": float(args.dedupe_iou),
            "scoreless_iou": float(args.scoreless_iou),
        },
        "rows": [],
    }

    for variant in variants:
        baseline_metrics = _read_metrics(variant.baseline_eval_json)
        report["rows"].append(
            {
                "variant": variant.name,
                "method": "xgb_baseline_1024",
                "dim": 1024,
                **baseline_metrics,
            }
        )
        _log(
            f"[{variant.name}] baseline 1024: "
            f"P={baseline_metrics['precision']:.4f} R={baseline_metrics['recall']:.4f} F1={baseline_metrics['f1']:.4f}"
        )

        for method in methods:
            for dim in dims:
                label = f"{variant.name}.{method}.d{dim}"
                _log(f"[{label}] start")
                reduced_npz = out_root / f"{label}.labeled.npz"
                projection_info = _build_reduced_npz(
                    src_npz=variant.labeled_npz,
                    dst_npz=reduced_npz,
                    method=method,
                    dim=int(dim),
                    seed=int(args.seed),
                    fixed_val_images=fixed_val_images,
                )

                model_prefix = out_root / f"{label}.xgb"
                train_cmd = [
                    "python",
                    "tools/train_ensemble_xgb.py",
                    "--input",
                    str(reduced_npz),
                    "--output",
                    str(model_prefix),
                    "--seed",
                    "42",
                    "--optimize",
                    "f1",
                    "--per-class",
                    "--threshold-steps",
                    str(int(args.steps)),
                    "--target-fp-ratio",
                    str(float(args.target_fp_ratio)),
                    "--min-recall",
                    str(float(args.min_recall)),
                    "--fixed-val-images",
                    str(fixed_val_images),
                ]
                _run_cmd(train_cmd)

                model_json = model_prefix.with_suffix(".json")
                model_meta = model_prefix.with_suffix(".meta.json")
                tune_cmd = [
                    "python",
                    "tools/tune_ensemble_thresholds_xgb.py",
                    "--model",
                    str(model_json),
                    "--meta",
                    str(model_meta),
                    "--data",
                    str(reduced_npz),
                    "--dataset",
                    variant.dataset,
                    "--optimize",
                    "f1",
                    "--target-fp-ratio",
                    str(float(args.target_fp_ratio)),
                    "--min-recall",
                    str(float(args.min_recall)),
                    "--steps",
                    str(int(args.steps)),
                    "--eval-iou",
                    str(float(args.eval_iou)),
                    "--dedupe-iou",
                    str(float(args.dedupe_iou)),
                    "--scoreless-iou",
                    str(float(args.scoreless_iou)),
                    "--use-val-split",
                ]
                _run_cmd(tune_cmd)

                eval_json = out_root / f"{label}.eval.json"
                eval_cmd = [
                    "python",
                    "tools/eval_ensemble_xgb_dedupe.py",
                    "--model",
                    str(model_json),
                    "--meta",
                    str(model_meta),
                    "--data",
                    str(reduced_npz),
                    "--dataset",
                    variant.dataset,
                    "--prepass-jsonl",
                    str(variant.prepass_jsonl),
                    "--eval-iou",
                    str(float(args.eval_iou)),
                    "--eval-iou-grid",
                    str(float(args.eval_iou)),
                    "--dedupe-iou",
                    str(float(args.dedupe_iou)),
                    "--scoreless-iou",
                    str(float(args.scoreless_iou)),
                    "--use-val-split",
                ]
                _run_cmd(eval_cmd, stdout_path=eval_json, stderr_to_stdout=False)
                metrics = _read_metrics(eval_json)
                report["rows"].append(
                    {
                        "variant": variant.name,
                        "method": method,
                        "dim": int(dim),
                        "projection": projection_info,
                        **metrics,
                    }
                )
                _log(
                    f"[{label}] done "
                    f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
                )

    output_json = Path(args.output_json).resolve() if args.output_json else out_root / "projection_sweep_report.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"Wrote report: {output_json}")
    print(json.dumps({"report": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
