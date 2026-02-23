#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

REQUIRED_EMBED_PROJ_DIM = 1024


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


def _latest_deep_sweep_dir(base_run_dir: Path) -> Optional[Path]:
    candidates = sorted(base_run_dir.glob("deep_mlp_sweep_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if (path / "run.log").exists():
            return path
    return None


def _deep_sweep_running() -> bool:
    proc = subprocess.run(
        ["pgrep", "-af", "run_deeper_mlp_sweep_after_ensembles|train_ensemble_mlp|eval_ensemble_mlp_dedupe"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    this_pid = os.getpid()
    for line in lines:
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == this_pid:
            continue
        return True
    return False


def _wait_for_sweep_completion(sweep_dir: Path, *, poll_secs: int, log_path: Path) -> Path:
    report = sweep_dir / "deeper_mlp_sweep_report.json"
    with log_path.open("a", encoding="utf-8") as log:
        while True:
            running = _deep_sweep_running()
            done = report.exists()
            log.write(f"[{_utc_now()}] wait sweep done={done} running={running}\n")
            log.flush()
            if done and not running:
                return report
            time.sleep(max(10, poll_secs))


def _safe_div(num: float, den: float) -> float:
    if den <= 0.0:
        return 0.0
    return float(num / den)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _npz_embed_proj_dim(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        try:
            return int(data.get("embed_proj_dim", 0))
        except Exception:
            return 0


def _coverage_upper_bound(eval_payload: Dict[str, Any]) -> float:
    return float(
        eval_payload.get("metric_tiers", {})
        .get("post_prepass", {})
        .get("coverage_upper_bound", {})
        .get("candidate_all", {})
        .get("recall_upper_bound", 0.0)
    )


def _coverage_preservation(recall: float, upper_bound: float) -> float:
    if upper_bound <= 0.0:
        return 0.0
    return float(recall / upper_bound)


def _extract_method_rows_2000(base_run_dir: Path, sweep_report: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    rows_by_variant: Dict[str, Dict[str, Dict[str, float]]] = {
        "nonwindow_20c8": {},
        "window_ceab": {},
    }

    non_eval = _read_json(base_run_dir / "nonwindow_20c8.eval.json")
    win_eval = _read_json(base_run_dir / "window_ceab.eval.json")
    rows_by_variant["nonwindow_20c8"]["xgb_1024_baseline"] = {
        "precision": float(non_eval.get("precision", 0.0)),
        "recall": float(non_eval.get("recall", 0.0)),
        "f1": float(non_eval.get("f1", 0.0)),
    }
    rows_by_variant["window_ceab"]["xgb_1024_baseline"] = {
        "precision": float(win_eval.get("precision", 0.0)),
        "recall": float(win_eval.get("recall", 0.0)),
        "f1": float(win_eval.get("f1", 0.0)),
    }

    selected = _read_json(base_run_dir / "hybrid_after_sweep_jl_d512" / "selected_projection_hybrid_summary.json")
    for row in selected.get("rows", []):
        variant = str(row.get("variant") or "")
        method = str(row.get("method") or "")
        if variant not in rows_by_variant:
            continue
        if method not in {
            "xgb_jl_d512",
            "hybrid_lr_xgb_blend_jl_d512",
            "hybrid_mlp_xgb_blend_jl_d512",
        }:
            continue
        rows_by_variant[variant][method] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1", 0.0)),
        }

    # Best deep MLP row per variant from completed sweep.
    deep_best: Dict[str, Dict[str, Any]] = {}
    for row in sweep_report.get("rows", []):
        if "f1" not in row:
            continue
        variant = str(row.get("variant") or "")
        if variant not in rows_by_variant:
            continue
        current = deep_best.get(variant)
        if current is None or float(row.get("f1", 0.0)) > float(current.get("f1", 0.0)):
            deep_best[variant] = row
    for variant, row in deep_best.items():
        method = f"deep_mlp_best:{row.get('config_id')}@seed{row.get('seed')}"
        rows_by_variant[variant][method] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1", 0.0)),
        }

    return rows_by_variant


def _pick_best_setting(rows_by_variant: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    method_scores: Dict[str, List[float]] = {}
    for variant_rows in rows_by_variant.values():
        for method, metrics in variant_rows.items():
            method_scores.setdefault(method, []).append(float(metrics.get("f1", 0.0)))
    best_method = ""
    best_mean = -1.0
    for method, vals in method_scores.items():
        if not vals:
            continue
        mean_val = float(sum(vals) / len(vals))
        if mean_val > best_mean:
            best_mean = mean_val
            best_method = method
    return {"method": best_method, "mean_f1": best_mean}


def _render_2000_section(
    *,
    base_run_dir: Path,
    rows_by_variant: Dict[str, Dict[str, Dict[str, float]]],
    best_setting: Dict[str, Any],
    sweep_report_path: Path,
) -> str:
    non_eval = _read_json(base_run_dir / "nonwindow_20c8.eval.json")
    win_eval = _read_json(base_run_dir / "window_ceab.eval.json")
    ub_non = _coverage_upper_bound(non_eval)
    ub_win = _coverage_upper_bound(win_eval)

    lines: List[str] = []
    lines.append("## Phase 1 — 2000-image calibration sweep")
    lines.append("")
    lines.append("Data sources:")
    lines.append(f"- Base evals: `{base_run_dir / 'nonwindow_20c8.eval.json'}`, `{base_run_dir / 'window_ceab.eval.json'}`")
    lines.append(f"- Hybrid follow-up: `{base_run_dir / 'hybrid_after_sweep_jl_d512' / 'selected_projection_hybrid_summary.json'}`")
    lines.append(f"- Deep MLP sweep: `{sweep_report_path}`")
    lines.append("")
    lines.append("Candidate coverage upper bound (from prepass, before calibration):")
    lines.append(f"- `nonwindow_20c8`: {ub_non:.4f} recall upper bound")
    lines.append(f"- `window_ceab`: {ub_win:.4f} recall upper bound")
    lines.append("")
    lines.append(
        "Coverage preservation metric definition: `coverage_preservation = post_calibration_recall / prepass_recall_upper_bound`."
    )
    lines.append("")
    lines.append("| Variant | Method | Precision | Recall | F1 | Coverage Preservation |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for variant in ("nonwindow_20c8", "window_ceab"):
        ub = ub_non if variant == "nonwindow_20c8" else ub_win
        methods = rows_by_variant.get(variant, {})
        for method, metrics in sorted(methods.items(), key=lambda kv: kv[1].get("f1", 0.0), reverse=True):
            rec = float(metrics.get("recall", 0.0))
            cov = _coverage_preservation(rec, ub)
            lines.append(
                f"| {variant} | {method} | {metrics.get('precision', 0.0):.4f} | {rec:.4f} | {metrics.get('f1', 0.0):.4f} | {cov:.4f} |"
            )
    lines.append("")
    lines.append(
        f"Best discovered setting (mean F1 across available variants): `{best_setting.get('method')}` (mean F1={best_setting.get('mean_f1', 0.0):.4f})."
    )
    lines.append("")
    return "\n".join(lines)


def _build_calibration_payload(
    *,
    dataset_id: str,
    max_images: int,
    seed: int,
    classifier_id: str,
    prepass_config: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "max_images": int(max_images),
        "seed": int(seed),
        "classifier_id": classifier_id,
        "base_fp_ratio": 0.2,
        "relax_fp_ratio": 0.2,
        "recall_floor": 0.6,
        "per_class_thresholds": True,
        "threshold_steps": 300,
        "optimize_metric": "f1",
        "label_iou": 0.5,
        "eval_iou": 0.5,
        "eval_iou_grid": "0.5",
        "dedupe_iou": 0.75,
        "scoreless_iou": 0.0,
        "support_iou": 0.5,
        "model_seed": 42,
        "calibration_model": "xgb",
        "split_head_by_support": True,
        "train_sam3_text_quality": True,
        "sam3_text_quality_alpha": 0.35,
    }
    passthrough = [
        "enable_yolo",
        "enable_rfdetr",
        "sam3_text_synonym_budget",
        "sam3_text_window_extension",
        "sam3_text_window_mode",
        "sam3_text_window_size",
        "sam3_text_window_overlap",
        "prepass_sam3_text_thr",
        "prepass_similarity_score",
        "similarity_min_exemplar_score",
        "similarity_exemplar_count",
        "similarity_exemplar_strategy",
        "similarity_exemplar_seed",
        "similarity_exemplar_fraction",
        "similarity_exemplar_min",
        "similarity_exemplar_max",
        "similarity_exemplar_source_quota",
        "similarity_window_extension",
        "similarity_window_mode",
        "similarity_window_size",
        "similarity_window_overlap",
        "sam3_score_thr",
        "sam3_mask_threshold",
        "detector_conf",
        "sahi_window_size",
        "sahi_overlap_ratio",
        "scoreless_iou",
        "dedupe_iou",
    ]
    for key in passthrough:
        if key in prepass_config:
            payload[key] = prepass_config.get(key)
    # Keep cross-class dedupe off by default unless explicitly requested in config.
    payload["cross_class_dedupe_enabled"] = bool(prepass_config.get("cross_class_dedupe_enabled", False))
    payload["cross_class_dedupe_iou"] = float(prepass_config.get("cross_class_dedupe_iou", 0.8))
    payload["fusion_mode"] = str(prepass_config.get("fusion_mode") or "primary")
    return payload


def _submit_calibration_job(api_root: str, payload: Dict[str, Any], *, log_path: Path) -> str:
    url = f"{api_root.rstrip('/')}/calibration/jobs"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_utc_now()}] submit calibration payload={json.dumps(payload, ensure_ascii=True)}\n")
        log.flush()
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    job_id = str(body.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError(f"missing job_id in response: {body}")
    return job_id


def _wait_job(api_root: str, job_id: str, *, poll_secs: int, log_path: Path) -> Dict[str, Any]:
    url = f"{api_root.rstrip('/')}/calibration/jobs/{job_id}"
    with log_path.open("a", encoding="utf-8") as log:
        while True:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 404:
                raise RuntimeError(f"job disappeared: {job_id}")
            resp.raise_for_status()
            job = resp.json()
            status = str(job.get("status") or "").lower()
            phase = str(job.get("phase") or "")
            processed = int(job.get("processed") or 0)
            total = int(job.get("total") or 0)
            progress = float(job.get("progress") or 0.0)
            message = str(job.get("message") or "")
            log.write(
                f"[{_utc_now()}] job={job_id} status={status} phase={phase} processed={processed}/{total} progress={progress:.3f} msg={message}\n"
            )
            log.flush()
            if status in {"completed", "failed", "cancelled"}:
                return job
            time.sleep(max(10, poll_secs))


def _run_xgb_4000(
    *,
    repo_root: Path,
    out_dir: Path,
    dataset_id: str,
    fixed_val_images: Path,
    variant_inputs: Dict[str, Dict[str, Path]],
    log_path: Path,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_paths: Dict[str, Path] = {}
    for name, inputs in variant_inputs.items():
        feat_npz = Path(str(inputs.get("features") or "")).resolve()
        prepass_jsonl = Path(str(inputs.get("prepass_jsonl") or "")).resolve()
        if not feat_npz.exists():
            raise RuntimeError(f"missing feature matrix for {name}: {feat_npz}")
        if not prepass_jsonl.exists():
            raise RuntimeError(f"missing prepass jsonl for {name}: {prepass_jsonl}")
        out_prefix = out_dir / name
        labeled_npz = Path(f"{out_prefix}.labeled.npz")
        model_prefix = out_prefix
        model_json = Path(f"{model_prefix}.json")
        model_meta = Path(f"{model_prefix}.meta.json")
        eval_json = Path(f"{out_prefix}.eval.json")

        _run(
            [
                sys.executable,
                "tools/label_candidates_iou90.py",
                "--input",
                str(feat_npz),
                "--dataset",
                dataset_id,
                "--output",
                str(labeled_npz),
                "--iou",
                "0.5",
            ],
            cwd=repo_root,
            log_path=log_path,
        )

        _run(
            [
                sys.executable,
                "tools/train_ensemble_xgb.py",
                "--input",
                str(labeled_npz),
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
                str(fixed_val_images),
            ],
            cwd=repo_root,
            log_path=log_path,
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
                str(labeled_npz),
                "--dataset",
                dataset_id,
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

        out = _run(
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
                dataset_id,
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
        eval_json.write_text(out.strip() + "\n", encoding="utf-8")
        eval_paths[name] = eval_json
    return eval_paths


def _render_4000_section(
    *,
    eval_paths: Dict[str, Path],
    baseline_2000: Dict[str, Dict[str, float]],
) -> str:
    lines: List[str] = []
    lines.append("## Phase 2 — 4000-image extension (same settings, +2000 images each variant)")
    lines.append("")
    lines.append("| Variant | Split Size | Precision | Recall | F1 | Coverage UB | Coverage Preservation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    summary: Dict[str, Dict[str, float]] = {}
    for variant_key, eval_path in eval_paths.items():
        payload = _read_json(eval_path)
        ub = _coverage_upper_bound(payload)
        rec = float(payload.get("recall", 0.0))
        cov = _coverage_preservation(rec, ub)
        summary[variant_key] = {
            "precision": float(payload.get("precision", 0.0)),
            "recall": rec,
            "f1": float(payload.get("f1", 0.0)),
            "coverage_ub": ub,
            "coverage_preservation": cov,
        }
        lines.append(
            f"| {variant_key} | 4000 | {summary[variant_key]['precision']:.4f} | {summary[variant_key]['recall']:.4f} | {summary[variant_key]['f1']:.4f} | {ub:.4f} | {cov:.4f} |"
        )

    lines.append("")
    lines.append("### 2000 vs 4000 delta (same XGB-1024 pipeline)")
    lines.append("")
    lines.append("| Variant | F1@2000 | F1@4000 | Delta F1 | CovPres@2000 | CovPres@4000 | Delta CovPres |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    mapping = {
        "nonwindow_4000": "nonwindow_20c8",
        "window_4000": "window_ceab",
    }
    for variant_4000, variant_2000 in mapping.items():
        cur = summary.get(variant_4000, {})
        old = baseline_2000.get(variant_2000, {})
        lines.append(
            "| {v} | {f1_old:.4f} | {f1_new:.4f} | {df1:+.4f} | {cp_old:.4f} | {cp_new:.4f} | {dcp:+.4f} |".format(
                v=variant_4000,
                f1_old=float(old.get("f1", 0.0)),
                f1_new=float(cur.get("f1", 0.0)),
                df1=float(cur.get("f1", 0.0) - old.get("f1", 0.0)),
                cp_old=float(old.get("coverage_preservation", 0.0)),
                cp_new=float(cur.get("coverage_preservation", 0.0)),
                dcp=float(cur.get("coverage_preservation", 0.0) - old.get("coverage_preservation", 0.0)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _update_readme(
    *,
    readme_path: Path,
    baseline_2000: Dict[str, Dict[str, float]],
    eval_paths_4000: Dict[str, Path],
) -> None:
    text = readme_path.read_text(encoding="utf-8")
    marker = "### 2000 vs 4000 extension snapshot"

    payload_non = _read_json(eval_paths_4000["nonwindow_4000"])
    payload_win = _read_json(eval_paths_4000["window_4000"])

    section = []
    section.append("### 2000 vs 4000 extension snapshot")
    section.append("Using the same XGB-1024 pipeline and IoU=0.50 policy, we extended each prepass variant by +2000 images (from 2000 to 4000).")
    section.append("")
    section.append("| Variant | F1@2000 | F1@4000 | Delta F1 | CovPres@2000 | CovPres@4000 | Delta CovPres |")
    section.append("|---|---:|---:|---:|---:|---:|---:|")
    section.append(
        "| nonwindow | {f1a:.4f} | {f1b:.4f} | {df1:+.4f} | {cpa:.4f} | {cpb:.4f} | {dcp:+.4f} |".format(
            f1a=float(baseline_2000["nonwindow_20c8"]["f1"]),
            f1b=float(payload_non.get("f1", 0.0)),
            df1=float(payload_non.get("f1", 0.0) - baseline_2000["nonwindow_20c8"]["f1"]),
            cpa=float(baseline_2000["nonwindow_20c8"]["coverage_preservation"]),
            cpb=float(_coverage_preservation(float(payload_non.get("recall", 0.0)), _coverage_upper_bound(payload_non))),
            dcp=float(
                _coverage_preservation(float(payload_non.get("recall", 0.0)), _coverage_upper_bound(payload_non))
                - baseline_2000["nonwindow_20c8"]["coverage_preservation"]
            ),
        )
    )
    section.append(
        "| windowed | {f1a:.4f} | {f1b:.4f} | {df1:+.4f} | {cpa:.4f} | {cpb:.4f} | {dcp:+.4f} |".format(
            f1a=float(baseline_2000["window_ceab"]["f1"]),
            f1b=float(payload_win.get("f1", 0.0)),
            df1=float(payload_win.get("f1", 0.0) - baseline_2000["window_ceab"]["f1"]),
            cpa=float(baseline_2000["window_ceab"]["coverage_preservation"]),
            cpb=float(_coverage_preservation(float(payload_win.get("recall", 0.0)), _coverage_upper_bound(payload_win))),
            dcp=float(
                _coverage_preservation(float(payload_win.get("recall", 0.0)), _coverage_upper_bound(payload_win))
                - baseline_2000["window_ceab"]["coverage_preservation"]
            ),
        )
    )
    section.append("")
    section_text = "\n".join(section)

    if marker in text:
        start = text.index(marker)
        end = text.find("\n---\n", start)
        if end == -1:
            end = len(text)
        text = text[:start] + section_text + text[end:]
    else:
        anchor = "## Feature highlights (current status)"
        if anchor in text:
            idx = text.index(anchor)
            text = text[:idx] + section_text + "\n\n" + text[idx:]
        else:
            text = text.rstrip() + "\n\n" + section_text + "\n"
    readme_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="After deep MLP sweep, produce report and run 4000-image extension.")
    parser.add_argument("--api-root", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default="qwen_dataset")
    parser.add_argument("--base-run-dir", default="tmp/emb1024_calibration_20260219_161507")
    parser.add_argument("--prepassreport", default="prepassreport.md")
    parser.add_argument("--readme", default="readme.md")
    parser.add_argument("--classifier-id", default="uploads/classifiers/DinoV3_best_model_large.pkl")
    parser.add_argument("--nonwindow-key", default="20c8d44d69f51b2ffe528fb500e75672a306f67d")
    parser.add_argument("--window-key", default="ceab65b2bff24d316ca5f858addaffed8abfdb11")
    parser.add_argument("--poll-secs", type=int, default=30)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_run_dir = (repo_root / args.base_run_dir).resolve()
    if not base_run_dir.exists():
        raise SystemExit(f"base run dir missing: {base_run_dir}")

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = base_run_dir / f"post_sweep_pipeline_{ts}"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    log_path = pipeline_dir / "pipeline.log"

    sweep_dir = _latest_deep_sweep_dir(base_run_dir)
    if not sweep_dir:
        raise SystemExit("no deep_mlp_sweep_* directory found")
    sweep_report_path = _wait_for_sweep_completion(sweep_dir, poll_secs=int(args.poll_secs), log_path=log_path)
    sweep_report = _read_json(sweep_report_path)

    rows_by_variant = _extract_method_rows_2000(base_run_dir, sweep_report)
    best_setting = _pick_best_setting(rows_by_variant)

    non_eval_2000 = _read_json(base_run_dir / "nonwindow_20c8.eval.json")
    win_eval_2000 = _read_json(base_run_dir / "window_ceab.eval.json")
    baseline_2000 = {
        "nonwindow_20c8": {
            "precision": float(non_eval_2000.get("precision", 0.0)),
            "recall": float(non_eval_2000.get("recall", 0.0)),
            "f1": float(non_eval_2000.get("f1", 0.0)),
            "coverage_preservation": _coverage_preservation(
                float(non_eval_2000.get("recall", 0.0)),
                _coverage_upper_bound(non_eval_2000),
            ),
        },
        "window_ceab": {
            "precision": float(win_eval_2000.get("precision", 0.0)),
            "recall": float(win_eval_2000.get("recall", 0.0)),
            "f1": float(win_eval_2000.get("f1", 0.0)),
            "coverage_preservation": _coverage_preservation(
                float(win_eval_2000.get("recall", 0.0)),
                _coverage_upper_bound(win_eval_2000),
            ),
        },
    }

    report_path = (repo_root / args.prepassreport).resolve()
    phase1 = []
    phase1.append("# Prepass Calibration Report")
    phase1.append("")
    phase1.append(f"- Generated (UTC): {_utc_now()}")
    phase1.append(f"- Dataset: `{args.dataset}`")
    phase1.append(
        f"- Sweep source: `{sweep_report_path}`"
    )
    phase1.append("")
    phase1.append(
        _render_2000_section(
            base_run_dir=base_run_dir,
            rows_by_variant=rows_by_variant,
            best_setting=best_setting,
            sweep_report_path=sweep_report_path,
        )
    )
    report_path.write_text("\n".join(phase1).strip() + "\n", encoding="utf-8")

    # Expand prepass caches to 4000 for nonwindow + window variants.
    non_meta = _read_json(repo_root / "uploads" / "calibration_cache" / "prepass" / args.nonwindow_key / "prepass.meta.json")
    win_meta = _read_json(repo_root / "uploads" / "calibration_cache" / "prepass" / args.window_key / "prepass.meta.json")
    payload_non = _build_calibration_payload(
        dataset_id=args.dataset,
        max_images=4000,
        seed=42,
        classifier_id=args.classifier_id,
        prepass_config=non_meta.get("prepass_config", {}),
    )
    payload_win = _build_calibration_payload(
        dataset_id=args.dataset,
        max_images=4000,
        seed=42,
        classifier_id=args.classifier_id,
        prepass_config=win_meta.get("prepass_config", {}),
    )

    job_non = _submit_calibration_job(args.api_root, payload_non, log_path=log_path)
    status_non = _wait_job(args.api_root, job_non, poll_secs=int(args.poll_secs), log_path=log_path)
    if str(status_non.get("status") or "").lower() != "completed":
        raise RuntimeError(f"nonwindow 4000 calibration failed: {status_non}")

    job_win = _submit_calibration_job(args.api_root, payload_win, log_path=log_path)
    status_win = _wait_job(args.api_root, job_win, poll_secs=int(args.poll_secs), log_path=log_path)
    if str(status_win.get("status") or "").lower() != "completed":
        raise RuntimeError(f"window 4000 calibration failed: {status_win}")

    def _paths_from_job(status: Dict[str, Any]) -> Dict[str, Path]:
        result = status.get("result") or {}
        feature_path = Path(str(result.get("features") or "")).resolve()
        prepass_path = Path(str(result.get("prepass_jsonl") or "")).resolve()
        return {"features": feature_path, "prepass_jsonl": prepass_path}

    variant_inputs: Dict[str, Dict[str, Path]] = {
        "nonwindow_4000": _paths_from_job(status_non),
        "window_4000": _paths_from_job(status_win),
    }
    needs_legacy_backfill = False
    for variant_name, paths in variant_inputs.items():
        feature_path = paths["features"]
        prepass_path = paths["prepass_jsonl"]
        if (not feature_path.exists()) or (not prepass_path.exists()):
            needs_legacy_backfill = True
            break
        embed_dim = _npz_embed_proj_dim(feature_path)
        if embed_dim < REQUIRED_EMBED_PROJ_DIM:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"[{_utc_now()}] {variant_name} embed_proj_dim={embed_dim} (<{REQUIRED_EMBED_PROJ_DIM}); falling back to one-time backfill.\n"
                )
            needs_legacy_backfill = True
            break

    if needs_legacy_backfill:
        _run(
            [
                sys.executable,
                "tools/backfill_prepass_embeddings.py",
                "--dataset",
                args.dataset,
                "--cache-key",
                args.nonwindow_key,
                "--cache-key",
                args.window_key,
                "--classifier-id",
                args.classifier_id,
                "--embed-proj-dims",
                str(REQUIRED_EMBED_PROJ_DIM),
                "--device",
                "cuda",
                "--force",
            ],
            cwd=repo_root,
            log_path=log_path,
        )
        variant_inputs = {
            "nonwindow_4000": {
                "features": (
                    repo_root
                    / "uploads"
                    / "calibration_cache"
                    / "features_backfill"
                    / args.nonwindow_key
                    / f"ensemble_features_embed{REQUIRED_EMBED_PROJ_DIM}.npz"
                ),
                "prepass_jsonl": (
                    repo_root
                    / "uploads"
                    / "calibration_cache"
                    / "features_backfill"
                    / args.nonwindow_key
                    / "prepass.jsonl"
                ),
            },
            "window_4000": {
                "features": (
                    repo_root
                    / "uploads"
                    / "calibration_cache"
                    / "features_backfill"
                    / args.window_key
                    / f"ensemble_features_embed{REQUIRED_EMBED_PROJ_DIM}.npz"
                ),
                "prepass_jsonl": (
                    repo_root
                    / "uploads"
                    / "calibration_cache"
                    / "features_backfill"
                    / args.window_key
                    / "prepass.jsonl"
                ),
            },
        }

    # Best discovered setting currently uses XGB-1024 on these artifacts.
    xgb_4000_dir = pipeline_dir / "xgb_1024_4000"
    eval_paths = _run_xgb_4000(
        repo_root=repo_root,
        out_dir=xgb_4000_dir,
        dataset_id=args.dataset,
        fixed_val_images=repo_root / "uploads" / "calibration_jobs" / "fixed_val_qwen_dataset_2000_images.json",
        variant_inputs=variant_inputs,
        log_path=log_path,
    )

    phase2 = _render_4000_section(eval_paths=eval_paths, baseline_2000=baseline_2000)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + phase2 + "\n")
        handle.write("## Artifacts\n\n")
        handle.write(f"- Pipeline log: `{log_path}`\n")
        handle.write(f"- 4000 XGB eval nonwindow: `{eval_paths['nonwindow_4000']}`\n")
        handle.write(f"- 4000 XGB eval windowed: `{eval_paths['window_4000']}`\n")
        handle.write(
            f"- 4000 calibration job ids: nonwindow=`{job_non}`, windowed=`{job_win}`\n"
        )

    _update_readme(
        readme_path=(repo_root / args.readme).resolve(),
        baseline_2000=baseline_2000,
        eval_paths_4000=eval_paths,
    )

    summary = {
        "status": "completed",
        "pipeline_dir": str(pipeline_dir),
        "prepassreport": str(report_path),
        "jobs": {"nonwindow_4000": job_non, "window_4000": job_win},
        "best_setting": best_setting,
        "eval_paths_4000": {k: str(v) for k, v in eval_paths.items()},
    }
    (pipeline_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
