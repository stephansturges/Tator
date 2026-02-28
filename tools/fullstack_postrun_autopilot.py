#!/usr/bin/env python3
"""Wait for fullstack ablation completion, write a detailed report, then queue +2000 prepass encodings."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt(v: Any, digits: int = 4) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "n/a"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _log(msg: str, log_path: Path) -> None:
    line = f"[{_utc_now()}] {msg}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _wait_for_summary(output_root: Path, *, poll_secs: int, log_path: Path) -> Path:
    summary_path = output_root / "summary.json"
    while True:
        if summary_path.exists():
            try:
                _read_json(summary_path)
                _log(f"summary ready: {summary_path}", log_path)
                return summary_path
            except Exception as exc:  # noqa: BLE001
                _log(f"summary exists but parse failed (retry): {exc}", log_path)
        time.sleep(max(10, int(poll_secs)))


def _render_report(summary: Dict[str, Any], *, run_name: str) -> str:
    cfg = summary.get("config") or {}
    variants = summary.get("variants") or {}
    lines: List[str] = []
    lines.append("# Fullstack Prepass/Calibration Ablation Report")
    lines.append("")
    lines.append("## 1) Executive Summary")
    lines.append(
        "This report evaluates the full-stack acceptance update (source-aware policy + joint tuning + split-head "
        "support routing + SAM3-text quality head) against leave-one-out ablations on a fixed validation split."
    )
    lines.append(
        "Primary interpretation rule: compare post-calibration acceptance quality against the detector-union "
        "comparator and the constrained XGB baseline at identical IoU/eval settings."
    )
    lines.append("")
    lines.append("## 2) Experiment Context")
    lines.append(f"- Run: `{run_name}`")
    lines.append(f"- Generated (UTC): `{summary.get('generated_utc')}`")
    lines.append(f"- Dataset: `{summary.get('dataset')}`")
    lines.append(f"- Eval IoU: `{cfg.get('eval_iou', 0.5)}`")
    lines.append(f"- Dedupe IoU: `{cfg.get('dedupe_iou', 0.75)}`")
    lines.append(f"- Scoreless IoU: `{cfg.get('scoreless_iou', 0.0)}`")
    lines.append(f"- Optimize: `{cfg.get('optimize', 'f1')}`")
    lines.append(f"- Target FP ratio: `{cfg.get('target_fp_ratio', 0.2)}`")
    lines.append(f"- Recall floor: `{cfg.get('min_recall', 0.6)}`")
    lines.append(
        f"- Gate margin vs detector-union comparator: `{_safe_float(cfg.get('gate_margin', 0.02), 0.02):+.3f}` F1"
    )
    lines.append("")
    lines.append("## 3) Results by Variant")
    for variant_name, payload in variants.items():
        base = ((payload.get("baseline") or {}).get("metrics") or {})
        scenarios = payload.get("scenarios") or {}
        lines.append(f"### Variant `{variant_name}`")
        lines.append("")
        lines.append(
            "Baseline (current constrained): "
            f"P={_fmt(base.get('precision'))} "
            f"R={_fmt(base.get('recall'))} "
            f"F1={_fmt(base.get('f1'))} "
            f"Δvs union={_safe_float(base.get('delta_vs_union_f1')):+.4f}"
        )
        lines.append("")
        lines.append("| Scenario | Precision | Recall | F1 | Δ vs Union | Δ vs Baseline | Coverage Pres. | Gate |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for scenario in ("full_stack", "no_source_policy", "no_joint_tune", "no_split_head_quality"):
            row = ((scenarios.get(scenario) or {}).get("metrics") or {})
            if not row:
                lines.append(f"| {scenario} | pending | pending | pending | pending | pending | pending | pending |")
                continue
            lines.append(
                "| {scenario} | {p} | {r} | {f1} | {du} | {db} | {cp} | {gate} |".format(
                    scenario=scenario,
                    p=_fmt(row.get("precision")),
                    r=_fmt(row.get("recall")),
                    f1=_fmt(row.get("f1")),
                    du=f"{_safe_float(row.get('delta_vs_union_f1')):+.4f}",
                    db=f"{_safe_float(row.get('delta_vs_baseline_f1')):+.4f}",
                    cp=_fmt(row.get("coverage_preservation")),
                    gate="PASS" if bool(row.get("gate_pass")) else "FAIL",
                )
            )
        lines.append("")

        full = ((scenarios.get("full_stack") or {}).get("metrics") or {})
        no_src = ((scenarios.get("no_source_policy") or {}).get("metrics") or {})
        no_joint = ((scenarios.get("no_joint_tune") or {}).get("metrics") or {})
        no_split = ((scenarios.get("no_split_head_quality") or {}).get("metrics") or {})
        if full:
            lines.append("Ablation attribution (F1 impact vs `full_stack`):")
            if no_src:
                lines.append(
                    f"- Removing source policy: `{_safe_float(full.get('f1')) - _safe_float(no_src.get('f1')):+.4f}`"
                )
            if no_joint:
                lines.append(
                    f"- Removing joint threshold tune: `{_safe_float(full.get('f1')) - _safe_float(no_joint.get('f1')):+.4f}`"
                )
            if no_split:
                lines.append(
                    f"- Removing split-head + SAM3-text quality: `{_safe_float(full.get('f1')) - _safe_float(no_split.get('f1')):+.4f}`"
                )
            lines.append("")

    lines.append("## 4) How To Perceive These Updates")
    lines.append(
        "- Treat this as an acceptance-quality upgrade, not candidate-generation expansion; the observed gain should "
        "show up as better F1 at similar or improved coverage preservation."
    )
    lines.append(
        "- The strongest signal is `full_stack` vs `no_source_policy`: if this gap is material, SAM-heavy noise is "
        "being controlled correctly rather than suppressing useful detector-backed recall."
    )
    lines.append(
        "- `coverage_preservation` indicates whether calibration is throwing away too much prepass potential; high "
        "coverage with low F1 suggests thresholding/policy noise, while low coverage suggests over-pruning."
    )
    lines.append(
        "- Gate status (`Δ vs union` margin) should be treated as deployment guardrail: only promote recipe defaults "
        "when the gate passes consistently across both non-windowed and windowed variants."
    )
    lines.append("")
    lines.append("## 5) Recommended Default Recipe Adjustments")
    lines.append(
        "- Keep detector windowing on (YOLO + RF-DETR), maintain IoU=0.5 evaluation policy, and preserve current dedupe IoU=0.75."
    )
    lines.append(
        "- Keep split-head-by-support enabled and keep SAM3-text quality head enabled with alpha around `0.35` unless new ablations prove otherwise."
    )
    lines.append(
        "- Keep source-aware acceptance policy enabled with per-class overrides for SAM3 text/similarity bias, SAM-only floor, and consensus IoU."
    )
    lines.append(
        "- Keep joint threshold-shift tune enabled after policy search; it improves operating-point alignment without retraining."
    )
    lines.append(
        "- Keep cross-class dedupe disabled by default; expose only as an explicit advanced override."
    )
    lines.append("")
    lines.append("## 6) Improvements Hinted By This Research")
    lines.append(
        "- Add strict train/tune/holdout separation for policy search vs final evaluation to reduce optimism risk."
    )
    lines.append(
        "- Add confidence-gated detector support weighting (not just overlap existence) to improve split routing reliability."
    )
    lines.append(
        "- Add per-class regularization for policy search (penalize extreme class overrides) to improve transfer when dataset mix shifts."
    )
    lines.append(
        "- Add cached policy-eval memoization across scenarios to cut runtime and make larger search spaces practical."
    )
    lines.append(
        "- Add explicit variance reporting across seeds for threshold search to better quantify stability."
    )
    lines.append("")
    lines.append("## 7) Operational Follow-up")
    lines.append(
        "- In parallel with this report, the next +2000-image incremental encodings (non-windowed and windowed) are queued "
        "to push both prepass sets from 4000 -> 6000 images before next calibration cycle."
    )
    lines.append("")
    return "\n".join(lines)


def _ensure_backend(api_root: str, repo_root: Path, *, poll_secs: int, log_path: Path) -> None:
    health_url = f"{api_root.rstrip('/')}/calibration/jobs"
    try:
        resp = requests.get(health_url, timeout=8)
        if resp.status_code < 500:
            _log("backend already reachable", log_path)
            return
    except Exception:
        pass

    port = api_root.rsplit(":", 1)[-1]
    session = f"backend_auto_{int(time.time())}"
    cmd = (
        f"cd {repo_root} && source .venv/bin/activate && "
        f"python -m uvicorn app:app --host 0.0.0.0 --port {port}"
    )
    subprocess.run(["screen", "-dmS", session, "bash", "-lc", cmd], check=True)
    _log(f"started backend screen session `{session}`", log_path)
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            resp = requests.get(health_url, timeout=8)
            if resp.status_code < 500:
                _log("backend is reachable", log_path)
                return
        except Exception:
            pass
        time.sleep(max(3, poll_secs // 3))
    raise RuntimeError("backend_not_reachable_after_start")


def _submit_job(api_root: str, payload: Dict[str, Any], *, log_path: Path) -> str:
    url = f"{api_root.rstrip('/')}/calibration/jobs"
    _log(f"submit payload max_images={payload.get('max_images')} windowed={payload.get('sam3_text_window_extension')}", log_path)
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    job_id = str(body.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError(f"missing_job_id:{body}")
    _log(f"submitted job_id={job_id}", log_path)
    return job_id


def _get_job(api_root: str, job_id: str) -> Dict[str, Any]:
    url = f"{api_root.rstrip('/')}/calibration/jobs/{job_id}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _cancel_job(api_root: str, job_id: str, *, log_path: Path) -> None:
    url = f"{api_root.rstrip('/')}/calibration/jobs/{job_id}/cancel"
    try:
        resp = requests.post(url, timeout=60)
        if resp.status_code < 400:
            _log(f"cancelled job_id={job_id}", log_path)
        else:
            _log(f"cancel request returned {resp.status_code} for job_id={job_id}", log_path)
    except Exception as exc:  # noqa: BLE001
        _log(f"cancel request failed for job_id={job_id}: {exc}", log_path)


def _cache_count(repo_root: Path, cache_key: str) -> int:
    img_dir = repo_root / "uploads" / "calibration_cache" / "prepass" / cache_key / "images"
    return len(list(img_dir.glob("*.json")))


def _monitor_prepass_growth(
    *,
    api_root: str,
    repo_root: Path,
    jobs: Dict[str, str],
    target_counts: Dict[str, int],
    poll_secs: int,
    log_path: Path,
) -> None:
    done = {variant: False for variant in target_counts}
    while True:
        all_done = True
        for variant, target in target_counts.items():
            cache_key = NONWINDOW_CACHE_KEY if variant == "nonwindow" else WINDOW_CACHE_KEY
            count = _cache_count(repo_root, cache_key)
            job_id = jobs.get(variant)
            status = "n/a"
            phase = "n/a"
            if job_id:
                try:
                    job = _get_job(api_root, job_id)
                    status = str(job.get("status") or "").lower()
                    phase = str(job.get("phase") or "")
                except Exception as exc:  # noqa: BLE001
                    status = f"error:{exc}"
            _log(f"{variant}: cache_count={count}/{target} job={job_id} status={status} phase={phase}", log_path)

            if count >= target:
                done[variant] = True
                if job_id and status in {"queued", "running"}:
                    _cancel_job(api_root, job_id, log_path=log_path)
            if not done[variant]:
                all_done = False
        if all_done:
            _log("both variants reached target cache count", log_path)
            return
        time.sleep(max(15, int(poll_secs)))


NONWINDOW_CACHE_KEY = "20c8d44d69f51b2ffe528fb500e75672a306f67d"
WINDOW_CACHE_KEY = "ceab65b2bff24d316ca5f858addaffed8abfdb11"


def _payload_nonwindow(max_images: int) -> Dict[str, Any]:
    return {
        "dataset_id": "qwen_dataset",
        "max_images": int(max_images),
        "seed": 42,
        "classifier_id": "uploads/classifiers/DinoV3_best_model_large.pkl",
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
        "enable_yolo": True,
        "enable_rfdetr": True,
        "sam3_text_synonym_budget": 0,
        "sam3_text_window_extension": False,
        "sam3_text_window_mode": "grid",
        "prepass_sam3_text_thr": 0.2,
        "prepass_similarity_score": 0.3,
        "similarity_min_exemplar_score": 0.6,
        "similarity_exemplar_count": 3,
        "similarity_exemplar_strategy": "top",
        "similarity_exemplar_seed": 0,
        "similarity_window_extension": False,
        "sam3_score_thr": 0.2,
        "sam3_mask_threshold": 0.2,
        "detector_conf": 0.45,
        "sahi_window_size": 640,
        "sahi_overlap_ratio": 0.2,
        "cross_class_dedupe_enabled": False,
        "cross_class_dedupe_iou": 0.8,
        "fusion_mode": "primary",
    }


def _payload_window(max_images: int) -> Dict[str, Any]:
    return {
        "dataset_id": "qwen_dataset",
        "max_images": int(max_images),
        "seed": 42,
        "classifier_id": "uploads/classifiers/DinoV3_best_model_large.pkl",
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
        "enable_yolo": True,
        "enable_rfdetr": True,
        "sam3_text_synonym_budget": 0,
        "sam3_text_window_extension": True,
        "sam3_text_window_mode": "sahi",
        "sam3_text_window_size": 640,
        "sam3_text_window_overlap": 0.2,
        "prepass_sam3_text_thr": 0.2,
        "prepass_similarity_score": 0.3,
        "similarity_min_exemplar_score": 0.6,
        "similarity_exemplar_count": 3,
        "similarity_exemplar_strategy": "top",
        "similarity_exemplar_seed": 0,
        "similarity_window_extension": True,
        "similarity_window_mode": "sahi",
        "similarity_window_size": 640,
        "similarity_window_overlap": 0.2,
        "sam3_score_thr": 0.2,
        "sam3_mask_threshold": 0.2,
        "detector_conf": 0.45,
        "sahi_window_size": 640,
        "sahi_overlap_ratio": 0.2,
        "cross_class_dedupe_enabled": False,
        "cross_class_dedupe_iou": 0.8,
        "fusion_mode": "primary",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-root",
        default="tmp/emb1024_calibration_20260219_161507/fullstack_ablation_20260224_155306",
        help="Path to fullstack ablation output directory.",
    )
    parser.add_argument("--report-path", default="docs/fullstack_ablation_detailed_report.md")
    parser.add_argument("--api-root", default="http://127.0.0.1:8000")
    parser.add_argument("--target-max-images", type=int, default=6000)
    parser.add_argument("--poll-secs", type=int, default=30)
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    suite_root = (repo_root / args.suite_root).resolve()
    report_path = (repo_root / args.report_path).resolve()
    log_path = (
        (repo_root / args.log_path).resolve()
        if str(args.log_path).strip()
        else suite_root / "postrun_autopilot.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    _log(f"autopilot start suite_root={suite_root}", log_path)
    summary_path = _wait_for_summary(suite_root, poll_secs=args.poll_secs, log_path=log_path)
    summary = _read_json(summary_path)
    report_md = _render_report(summary, run_name=suite_root.name)
    report_path.write_text(report_md, encoding="utf-8")
    _log(f"wrote report: {report_path}", log_path)

    _ensure_backend(args.api_root, repo_root, poll_secs=args.poll_secs, log_path=log_path)

    target = int(args.target_max_images)
    current_non = _cache_count(repo_root, NONWINDOW_CACHE_KEY)
    current_win = _cache_count(repo_root, WINDOW_CACHE_KEY)
    _log(f"current cache counts nonwindow={current_non} window={current_win} target={target}", log_path)

    jobs: Dict[str, str] = {}
    if current_non < target:
        jobs["nonwindow"] = _submit_job(args.api_root, _payload_nonwindow(target), log_path=log_path)
    if current_win < target:
        jobs["window"] = _submit_job(args.api_root, _payload_window(target), log_path=log_path)
    if not jobs:
        _log("no new jobs required; cache counts already satisfy target", log_path)
        return

    _monitor_prepass_growth(
        api_root=args.api_root,
        repo_root=repo_root,
        jobs=jobs,
        target_counts={"nonwindow": target, "window": target},
        poll_secs=args.poll_secs,
        log_path=log_path,
    )
    _log("autopilot completed", log_path)


if __name__ == "__main__":
    main()
