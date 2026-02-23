#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _run(cmd: List[str], *, cwd: Path, log_path: Path) -> str:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] $ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.stdout:
            log.write(proc.stdout)
        log.flush()
        if proc.returncode != 0:
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
        return proc.stdout or ""


def _log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_utc_now()}] {message}\n")
        log.flush()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage_upper_bound(eval_payload: Dict[str, Any]) -> float:
    return float(
        eval_payload.get("metric_tiers", {})
        .get("post_prepass", {})
        .get("coverage_upper_bound", {})
        .get("candidate_all", {})
        .get("recall_upper_bound", 0.0)
    )


def _coverage_preservation(eval_payload: Dict[str, Any]) -> float:
    upper = _coverage_upper_bound(eval_payload)
    recall = float(eval_payload.get("recall", 0.0))
    return (recall / upper) if upper > 0.0 else 0.0


def _parse_dims(raw: str) -> List[int]:
    dims: List[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise SystemExit(f"invalid dim in --image-embed-dims: {token}")
        dims.append(value)
    deduped = sorted(set(dims), reverse=True)
    if not deduped:
        raise SystemExit("no --image-embed-dims provided")
    return deduped


def _parse_variants(raw: str) -> List[str]:
    allowed = {"nonwindow_4000", "window_4000"}
    values: List[str] = []
    for token in str(raw or "").split(","):
        name = token.strip()
        if not name:
            continue
        if name not in allowed:
            raise SystemExit(f"invalid variant in --variants: {name}")
        values.append(name)
    if not values:
        raise SystemExit("no --variants provided")
    # Preserve user order, remove duplicates.
    out: List[str] = []
    seen = set()
    for name in values:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _latest_pipeline_dir(base_run_dir: Path) -> Path:
    candidates = sorted(base_run_dir.glob("post_sweep_pipeline_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if (path / "pipeline.log").exists():
            return path
    raise SystemExit("no post_sweep_pipeline_* directory found")


def _is_post_pipeline_active() -> bool:
    proc = subprocess.run(
        ["pgrep", "-af", "run_post_sweep_4000_pipeline.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    me = str(os.getpid())
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if me and line.startswith(me + " "):
            continue
        return True
    return False


def _wait_for_4000_summary(base_run_dir: Path, *, poll_secs: int, log_path: Path) -> Tuple[Path, Path]:
    pipeline_dir = _latest_pipeline_dir(base_run_dir)
    summary_path = pipeline_dir / "summary.json"
    with log_path.open("a", encoding="utf-8") as log:
        while True:
            ready = summary_path.exists()
            active = _is_post_pipeline_active()
            log.write(f"[{_utc_now()}] wait_summary ready={ready} active={active} summary={summary_path}\n")
            log.flush()
            if ready:
                return pipeline_dir, summary_path
            time.sleep(max(10, int(poll_secs)))


def _materialize_if_missing(*, repo_root: Path, cache_key: str, output_jsonl: Path, log_path: Path) -> None:
    if output_jsonl.exists():
        _log(log_path, f"skip materialize (exists): {output_jsonl}")
        return
    _run(
        [
            sys.executable,
            "tools/materialize_prepass_from_cache.py",
            "--cache-key",
            cache_key,
            "--output",
            str(output_jsonl),
        ],
        cwd=repo_root,
        log_path=log_path,
    )


def _resolve_model_paths(model_prefix: Path) -> Tuple[Path, Path]:
    preferred_json = Path(f"{model_prefix}.json")
    preferred_meta = Path(f"{model_prefix}.meta.json")
    if preferred_json.exists() and preferred_meta.exists():
        return preferred_json, preferred_meta

    # `train_ensemble_xgb.py` currently uses Path.with_suffix(...) internally.
    # If `model_prefix` already has a suffix-like segment, it may emit collapsed names.
    fallback_json = model_prefix.with_suffix(".json")
    fallback_meta = model_prefix.with_suffix(".meta.json")
    if fallback_json.exists() and fallback_meta.exists():
        return fallback_json, fallback_meta

    return preferred_json, preferred_meta


def _append_report(
    *,
    report_path: Path,
    rows: List[Dict[str, Any]],
    baseline_by_variant: Dict[str, Dict[str, float]],
    image_dims: List[int],
) -> None:
    lines: List[str] = []
    lines.append("## Phase 3 — Full-Image Context Embedding Ablation (4000-image sets)")
    lines.append("")
    lines.append(
        "Candidate embedding kept at 1024-d; ablation varies additional full-image context embedding dim."
    )
    lines.append(f"Image-context dims evaluated: `{','.join(str(d) for d in image_dims)}`.")
    lines.append("")
    lines.append(
        "| Variant | Method | Precision | Recall | F1 | Coverage UB | Coverage Preservation | Delta F1 vs 4000 baseline |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    for variant in sorted(grouped):
        baseline = baseline_by_variant.get(variant, {})
        baseline_f1 = float(baseline.get("f1", 0.0))
        for row in sorted(grouped[variant], key=lambda item: int(item["image_embed_dim"]), reverse=True):
            lines.append(
                "| {variant} | img_ctx_d{dim} | {p:.4f} | {r:.4f} | {f1:.4f} | {ub:.4f} | {cov:.4f} | {df1:+.4f} |".format(
                    variant=variant,
                    dim=int(row["image_embed_dim"]),
                    p=float(row["precision"]),
                    r=float(row["recall"]),
                    f1=float(row["f1"]),
                    ub=float(row["coverage_upper_bound"]),
                    cov=float(row["coverage_preservation"]),
                    df1=float(row["f1"]) - baseline_f1,
                )
            )
    lines.append("")

    content = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if "## Phase 3 — Full-Image Context Embedding Ablation (4000-image sets)" in content:
        start = content.index("## Phase 3 — Full-Image Context Embedding Ablation (4000-image sets)")
        next_section = content.find("\n## ", start + 1)
        if next_section == -1:
            next_section = len(content)
        updated = content[:start] + "\n".join(lines) + content[next_section:]
    else:
        updated = content.rstrip() + ("\n\n" if content.strip() else "") + "\n".join(lines) + "\n"
    report_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-image context embedding ablation after 4000-image post-sweep pipeline completes."
    )
    parser.add_argument("--dataset", default="qwen_dataset")
    parser.add_argument("--base-run-dir", default="tmp/emb1024_calibration_20260219_161507")
    parser.add_argument("--prepassreport", default="prepassreport.md")
    parser.add_argument("--classifier-id", default="uploads/classifiers/DinoV3_best_model_large.pkl")
    parser.add_argument("--nonwindow-key", default="20c8d44d69f51b2ffe528fb500e75672a306f67d")
    parser.add_argument("--window-key", default="ceab65b2bff24d316ca5f858addaffed8abfdb11")
    parser.add_argument("--image-embed-dims", default="1024,128,64")
    parser.add_argument("--candidate-embed-dim", type=int, default=1024)
    parser.add_argument("--variants", default="nonwindow_4000,window_4000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-append-report",
        action="store_true",
        help="Do not write/overwrite Phase 3 section in prepassreport.md.",
    )
    parser.add_argument("--run-tag", default="")
    parser.add_argument(
        "--fixed-val-images",
        default="uploads/calibration_jobs/fixed_val_qwen_dataset_2000_images.json",
    )
    parser.add_argument("--poll-secs", type=int, default=30)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_run_dir = (repo_root / args.base_run_dir).resolve()
    base_run_dir.mkdir(parents=True, exist_ok=True)
    image_dims = _parse_dims(args.image_embed_dims)
    selected_variants = _parse_variants(args.variants)

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_tag = str(args.run_tag or "").strip()
    suffix = f"_{run_tag}" if run_tag else ""
    ablation_dir = base_run_dir / f"image_context_ablation_{ts}{suffix}"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    log_path = ablation_dir / "run.log"

    pipeline_dir, summary_path = _wait_for_4000_summary(
        base_run_dir, poll_secs=int(args.poll_secs), log_path=log_path
    )
    summary = _read_json(summary_path)
    eval_paths_4000 = summary.get("eval_paths_4000") or {}
    baseline_by_variant: Dict[str, Dict[str, float]] = {}
    for variant in ("nonwindow_4000", "window_4000"):
        eval_path_raw = eval_paths_4000.get(variant)
        if not eval_path_raw:
            continue
        eval_payload = _read_json(Path(str(eval_path_raw)))
        baseline_by_variant[variant] = {
            "precision": float(eval_payload.get("precision", 0.0)),
            "recall": float(eval_payload.get("recall", 0.0)),
            "f1": float(eval_payload.get("f1", 0.0)),
        }

    variant_to_key = {
        "nonwindow_4000": str(args.nonwindow_key).strip(),
        "window_4000": str(args.window_key).strip(),
    }

    rows: List[Dict[str, Any]] = []
    for variant in selected_variants:
        cache_key = variant_to_key.get(variant, "").strip()
        if not cache_key:
            raise SystemExit(f"missing cache key for variant: {variant}")
        backfill_root = repo_root / "uploads" / "calibration_cache" / "features_backfill" / cache_key
        backfill_root.mkdir(parents=True, exist_ok=True)
        prepass_jsonl = backfill_root / "prepass.jsonl"
        _materialize_if_missing(
            repo_root=repo_root,
            cache_key=cache_key,
            output_jsonl=prepass_jsonl,
            log_path=log_path,
        )
        for image_dim in image_dims:
            run_prefix = ablation_dir / f"{variant}_imgctx_d{int(image_dim)}"
            features_path = Path(f"{run_prefix}.features.npz")
            labeled_path = Path(f"{run_prefix}.labeled.npz")
            model_prefix = run_prefix
            eval_path = Path(f"{run_prefix}.eval.json")

            if features_path.exists():
                _log(log_path, f"skip feature build (exists): {features_path}")
            else:
                _run(
                    [
                        sys.executable,
                        "tools/build_ensemble_features.py",
                        "--input",
                        str(prepass_jsonl),
                        "--dataset",
                        args.dataset,
                        "--output",
                        str(features_path),
                        "--classifier-id",
                        args.classifier_id,
                        "--require-classifier",
                        "--support-iou",
                        "0.5",
                        "--context-radius",
                        "0.075",
                        "--embed-proj-dim",
                        str(int(args.candidate_embed_dim)),
                        "--image-embed-proj-dim",
                        str(int(image_dim)),
                        "--device",
                        str(args.device),
                    ],
                    cwd=repo_root,
                    log_path=log_path,
                )

            if labeled_path.exists():
                _log(log_path, f"skip labeling (exists): {labeled_path}")
            else:
                _run(
                    [
                        sys.executable,
                        "tools/label_candidates_iou90.py",
                        "--input",
                        str(features_path),
                        "--dataset",
                        args.dataset,
                        "--output",
                        str(labeled_path),
                        "--iou",
                        "0.5",
                    ],
                    cwd=repo_root,
                    log_path=log_path,
                )

            model_json, model_meta = _resolve_model_paths(model_prefix)
            if model_json.exists() and model_meta.exists():
                _log(log_path, f"skip train (model exists): {model_json} / {model_meta}")
            else:
                _run(
                    [
                        sys.executable,
                        "tools/train_ensemble_xgb.py",
                        "--input",
                        str(labeled_path),
                        "--output",
                        str(model_prefix),
                        "--seed",
                        "42",
                        "--optimize",
                        "f1",
                        "--per-class",
                        "--threshold-steps",
                        "300",
                        "--target-fp-ratio",
                        "0.2",
                        "--min-recall",
                        "0.6",
                        "--fixed-val-images",
                        str((repo_root / args.fixed_val_images).resolve()),
                    ],
                    cwd=repo_root,
                    log_path=log_path,
                )
                model_json, model_meta = _resolve_model_paths(model_prefix)
                if not (model_json.exists() and model_meta.exists()):
                    raise RuntimeError(
                        f"missing trained model artifacts for prefix {model_prefix}: "
                        f"checked {model_json} and {model_meta}"
                    )

            _run(
                [
                    sys.executable,
                    "tools/tune_ensemble_thresholds_xgb.py",
                    "--model",
                    str(model_json),
                    "--meta",
                    str(model_meta),
                    "--data",
                    str(labeled_path),
                    "--dataset",
                    args.dataset,
                    "--optimize",
                    "f1",
                    "--target-fp-ratio",
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
                ],
                cwd=repo_root,
                log_path=log_path,
            )
            eval_out = _run(
                [
                    sys.executable,
                    "tools/eval_ensemble_xgb_dedupe.py",
                    "--model",
                    str(model_json),
                    "--meta",
                    str(model_meta),
                    "--data",
                    str(labeled_path),
                    "--dataset",
                    args.dataset,
                    "--prepass-jsonl",
                    str(prepass_jsonl),
                    "--eval-iou",
                    "0.5",
                    "--eval-iou-grid",
                    "0.5",
                    "--dedupe-iou",
                    "0.75",
                    "--scoreless-iou",
                    "0.0",
                    "--use-val-split",
                ],
                cwd=repo_root,
                log_path=log_path,
            )
            eval_path.write_text(eval_out.strip() + "\n", encoding="utf-8")
            eval_payload = _read_json(eval_path)
            rows.append(
                {
                    "variant": variant,
                    "image_embed_dim": int(image_dim),
                    "precision": float(eval_payload.get("precision", 0.0)),
                    "recall": float(eval_payload.get("recall", 0.0)),
                    "f1": float(eval_payload.get("f1", 0.0)),
                    "coverage_upper_bound": _coverage_upper_bound(eval_payload),
                    "coverage_preservation": _coverage_preservation(eval_payload),
                    "eval_path": str(eval_path),
                }
            )

    report_path = (repo_root / args.prepassreport).resolve()
    if not bool(args.no_append_report):
        _append_report(
            report_path=report_path,
            rows=rows,
            baseline_by_variant=baseline_by_variant,
            image_dims=image_dims,
        )

    out = {
        "status": "completed",
        "pipeline_dir": str(pipeline_dir),
        "summary_path": str(summary_path),
        "ablation_dir": str(ablation_dir),
        "variants": selected_variants,
        "device": str(args.device),
        "append_report": not bool(args.no_append_report),
        "rows": rows,
        "report_path": str(report_path),
    }
    (ablation_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
