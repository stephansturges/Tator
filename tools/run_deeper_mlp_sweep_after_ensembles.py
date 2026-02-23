#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _parse_csv_ints(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("expected at least one integer")
    return out


def _run(
    cmd: Sequence[str],
    *,
    log_handle,
    cwd: Path,
) -> str:
    log_handle.write(f"\n[{_utc_now()}] $ {' '.join(cmd)}\n")
    log_handle.flush()
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.stdout:
        log_handle.write(proc.stdout)
    log_handle.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout or ""


def _extract_json(stdout: str) -> Dict[str, Any]:
    raw = (stdout or "").strip()
    if not raw:
        raise ValueError("empty stdout")
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("stdout does not contain json object")
    return json.loads(raw[start : end + 1])


def _resolve_train_outputs(prefix: Path) -> tuple[Path, Path]:
    # train_ensemble_mlp.py uses Path(prefix).with_suffix(...), so dotted prefixes
    # drop trailing tokens. Support both naming styles defensively.
    candidates = [
        Path(str(prefix) + ".pt"),
        Path(str(prefix) + ".meta.json"),
    ]
    model_path = candidates[0]
    meta_path = candidates[1]
    if model_path.exists() and meta_path.exists():
        return model_path, meta_path
    model_path = Path(prefix).with_suffix(".pt")
    meta_path = Path(prefix).with_suffix(".meta.json")
    return model_path, meta_path


def _active_benchmark_processes() -> List[str]:
    pattern = (
        "run_projection_sweep_xgb|run_hybrid_after_projection_sweep|run_hybrid_followup_after_xgb|"
        "train_ensemble_xgb|eval_ensemble_xgb_dedupe|tune_ensemble_thresholds_xgb|"
        "run_deeper_mlp_sweep_after_ensembles"
    )
    proc = subprocess.run(
        ["pgrep", "-af", pattern],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    this_pid = os.getpid()
    lines = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == this_pid:
            continue
        if "SCREEN -dmS deep_mlp_sweep_" in line:
            continue
        lines.append(line)
    return lines


def _wait_for_other_benchmarks(*, poll_secs: int, log_handle) -> None:
    while True:
        active = _active_benchmark_processes()
        if not active:
            log_handle.write(f"[{_utc_now()}] no active benchmark processes detected\n")
            log_handle.flush()
            return
        log_handle.write(f"[{_utc_now()}] waiting for active benchmarks to finish:\n")
        for line in active:
            log_handle.write(f"  - {line}\n")
        log_handle.flush()
        time.sleep(max(5, int(poll_secs)))


def _baseline_xgb(run_dir: Path, variant: str) -> Dict[str, float]:
    payload = json.loads((run_dir / f"{variant}.eval.json").read_text())
    return {
        "precision": float(payload.get("precision") or 0.0),
        "recall": float(payload.get("recall") or 0.0),
        "f1": float(payload.get("f1") or 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deeper MLP sweep after other ensemble jobs finish.")
    parser.add_argument(
        "--run-dir",
        default="tmp/emb1024_calibration_20260219_161507",
        help="Base run directory containing *.{labeled,eval}.json artifacts.",
    )
    parser.add_argument("--dataset", default="qwen_dataset", help="Dataset id for eval.")
    parser.add_argument("--seeds", default="42,1337,2025", help="Comma-separated seeds.")
    parser.add_argument("--poll-secs", type=int, default=60, help="Wait interval while other jobs are active.")
    parser.add_argument("--device", default="cuda", help="Device for MLP training.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = (repo_root / args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = run_dir / f"deep_mlp_sweep_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    variants = {
        "nonwindow_20c8": run_dir / "nonwindow_20c8.labeled.npz",
        "window_ceab": run_dir / "window_ceab.labeled.npz",
    }
    for variant, labeled_path in variants.items():
        if not labeled_path.exists():
            raise SystemExit(f"missing labeled artifact for {variant}: {labeled_path}")
        eval_path = run_dir / f"{variant}.eval.json"
        if not eval_path.exists():
            raise SystemExit(f"missing xgb eval for {variant}: {eval_path}")

    seeds = _parse_csv_ints(args.seeds)
    configs: List[Dict[str, Any]] = [
        {
            "id": "deep_a_asym",
            "hidden": "1024,512,256,128",
            "dropout": 0.15,
            "loss": "asym_focal",
            "lr": 8e-4,
            "weight_decay": 5e-5,
            "epochs": 45,
            "early_stop": 7,
        },
        {
            "id": "deep_b_asym",
            "hidden": "1536,768,384,192",
            "dropout": 0.20,
            "loss": "asym_focal",
            "lr": 6e-4,
            "weight_decay": 1e-4,
            "epochs": 55,
            "early_stop": 8,
        },
        {
            "id": "deep_c_asym",
            "hidden": "1024,1024,512,256",
            "dropout": 0.25,
            "loss": "asym_focal",
            "lr": 6e-4,
            "weight_decay": 1e-4,
            "epochs": 55,
            "early_stop": 8,
        },
        {
            "id": "deep_a_focal_tversky",
            "hidden": "1024,512,256,128",
            "dropout": 0.15,
            "loss": "focal_tversky",
            "lr": 8e-4,
            "weight_decay": 5e-5,
            "epochs": 45,
            "early_stop": 7,
        },
        {
            "id": "deep_b_focal_tversky",
            "hidden": "1536,768,384,192",
            "dropout": 0.20,
            "loss": "focal_tversky",
            "lr": 6e-4,
            "weight_decay": 1e-4,
            "epochs": 55,
            "early_stop": 8,
        },
    ]

    rows: List[Dict[str, Any]] = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"[{_utc_now()}] starting deeper MLP sweep\n")
        log_handle.write(f"[{_utc_now()}] run_dir={run_dir}\n")
        log_handle.flush()

        _wait_for_other_benchmarks(poll_secs=int(args.poll_secs), log_handle=log_handle)
        log_handle.write(f"[{_utc_now()}] launching sweep variants={list(variants.keys())}\n")
        log_handle.flush()

        for variant, labeled_path in variants.items():
            variant_dir = out_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            xgb = _baseline_xgb(run_dir, variant)
            for cfg in configs:
                for seed in seeds:
                    run_id = f"{variant}_{cfg['id']}_s{seed}"
                    prefix = variant_dir / run_id
                    try:
                        train_cmd = [
                            sys.executable,
                            "tools/train_ensemble_mlp.py",
                            "--input",
                            str(labeled_path),
                            "--output",
                            str(prefix),
                            "--hidden",
                            str(cfg["hidden"]),
                            "--dropout",
                            str(cfg["dropout"]),
                            "--epochs",
                            str(cfg["epochs"]),
                            "--lr",
                            str(cfg["lr"]),
                            "--weight-decay",
                            str(cfg["weight_decay"]),
                            "--batch-size",
                            "2048",
                            "--grad-accum",
                            "1",
                            "--loss",
                            str(cfg["loss"]),
                            "--class-balance",
                            "per_class",
                            "--neg-weight-mode",
                            "sqrt",
                            "--sampler",
                            "weighted",
                            "--scheduler",
                            "cosine",
                            "--min-lr",
                            "1e-5",
                            "--seed",
                            str(seed),
                            "--early-stop-patience",
                            str(cfg["early_stop"]),
                            "--target-mode",
                            "hard",
                            "--device",
                            str(args.device),
                        ]
                        _run(train_cmd, log_handle=log_handle, cwd=repo_root)

                        model_path, meta_path = _resolve_train_outputs(prefix)
                        if not model_path.exists() or not meta_path.exists():
                            raise RuntimeError(f"missing model/meta for {run_id}")

                        temp_cmd = [
                            sys.executable,
                            "tools/calibrate_ensemble_temperature.py",
                            "--model",
                            str(model_path),
                            "--data",
                            str(labeled_path),
                            "--meta",
                            str(meta_path),
                            "--objective",
                            "nll",
                            "--min-temp",
                            "0.5",
                            "--max-temp",
                            "3.0",
                            "--steps",
                            "121",
                            "--use-val-split",
                        ]
                        temp_stdout = _run(temp_cmd, log_handle=log_handle, cwd=repo_root)
                        temp_metrics = _extract_json(temp_stdout)

                        thresh_cmd = [
                            sys.executable,
                            "tools/calibrate_ensemble_threshold.py",
                            "--model",
                            str(model_path),
                            "--data",
                            str(labeled_path),
                            "--meta",
                            str(meta_path),
                            "--target-fp-ratio",
                            "0.2",
                            "--min-recall",
                            "0.6",
                            "--steps",
                            "300",
                            "--per-class",
                            "--optimize",
                            "f1",
                        ]
                        _run(thresh_cmd, log_handle=log_handle, cwd=repo_root)

                        relax_cmd = [
                            sys.executable,
                            "tools/relax_ensemble_thresholds.py",
                            "--model",
                            str(model_path),
                            "--data",
                            str(labeled_path),
                            "--meta",
                            str(meta_path),
                            "--fp-ratio-cap",
                            "0.2",
                            "--global-fp-cap",
                            "0.2",
                            "--smooth-alpha",
                            "0.2",
                            "--smooth-step",
                            "0.05",
                        ]
                        _run(relax_cmd, log_handle=log_handle, cwd=repo_root)

                        eval_cmd = [
                            sys.executable,
                            "tools/eval_ensemble_mlp_dedupe.py",
                            "--model",
                            str(model_path),
                            "--meta",
                            str(meta_path),
                            "--data",
                            str(labeled_path),
                            "--dataset",
                            str(args.dataset),
                            "--eval-iou",
                            "0.5",
                            "--dedupe-iou",
                            "0.75",
                            "--scoreless-iou",
                            "0.0",
                            "--use-val-split",
                        ]
                        eval_stdout = _run(eval_cmd, log_handle=log_handle, cwd=repo_root)
                        metrics = _extract_json(eval_stdout)
                        (variant_dir / f"{run_id}.eval.json").write_text(json.dumps(metrics, indent=2))

                        row = {
                            "variant": variant,
                            "config_id": str(cfg["id"]),
                            "seed": int(seed),
                            "model_prefix": str(prefix),
                            "temperature": float(temp_metrics.get("metrics", {}).get("temperature", 1.0)),
                            "precision": float(metrics.get("precision") or 0.0),
                            "recall": float(metrics.get("recall") or 0.0),
                            "f1": float(metrics.get("f1") or 0.0),
                            "tp": int(metrics.get("tp") or 0),
                            "fp": int(metrics.get("fp") or 0),
                            "fn": int(metrics.get("fn") or 0),
                            "xgb_baseline_precision": float(xgb["precision"]),
                            "xgb_baseline_recall": float(xgb["recall"]),
                            "xgb_baseline_f1": float(xgb["f1"]),
                        }
                        rows.append(row)
                    except Exception as exc:
                        rows.append(
                            {
                                "variant": variant,
                                "config_id": str(cfg["id"]),
                                "seed": int(seed),
                                "model_prefix": str(prefix),
                                "error": str(exc),
                                "xgb_baseline_precision": float(xgb["precision"]),
                                "xgb_baseline_recall": float(xgb["recall"]),
                                "xgb_baseline_f1": float(xgb["f1"]),
                            }
                        )
                        log_handle.write(f"[{_utc_now()}] run failed {run_id}: {exc}\n")
                        log_handle.flush()

                    partial = {
                        "generated_at_utc": _utc_now(),
                        "run_dir": str(run_dir),
                        "output_dir": str(out_dir),
                        "rows": rows,
                    }
                    (out_dir / "deeper_mlp_sweep.partial.json").write_text(json.dumps(partial, indent=2))

        grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["variant"]), str(row["config_id"]))].append(row)

        summary_rows: List[Dict[str, Any]] = []
        for (variant, config_id), items in sorted(grouped.items()):
            ok_items = [it for it in items if "f1" in it]
            err_items = [it for it in items if "f1" not in it]
            n = len(ok_items)
            if ok_items:
                mean_precision = sum(float(it["precision"]) for it in ok_items) / max(1, n)
                mean_recall = sum(float(it["recall"]) for it in ok_items) / max(1, n)
                mean_f1 = sum(float(it["f1"]) for it in ok_items) / max(1, n)
                delta = float(mean_f1 - float(items[0]["xgb_baseline_f1"]))
            else:
                mean_precision = 0.0
                mean_recall = 0.0
                mean_f1 = 0.0
                delta = float("nan")
            summary_rows.append(
                {
                    "variant": variant,
                    "config_id": config_id,
                    "runs_ok": int(n),
                    "runs_error": int(len(err_items)),
                    "mean_precision": float(mean_precision),
                    "mean_recall": float(mean_recall),
                    "mean_f1": float(mean_f1),
                    "xgb_baseline_f1": float(items[0]["xgb_baseline_f1"]),
                    "delta_vs_xgb_f1": delta,
                }
            )

        report = {
            "generated_at_utc": _utc_now(),
            "run_dir": str(run_dir),
            "output_dir": str(out_dir),
            "seeds": seeds,
            "configs": configs,
            "rows": rows,
            "summary": summary_rows,
        }
        (out_dir / "deeper_mlp_sweep_report.json").write_text(json.dumps(report, indent=2))
        log_handle.write(f"[{_utc_now()}] completed deeper MLP sweep; rows={len(rows)}\n")
        log_handle.flush()

    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
