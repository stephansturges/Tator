#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _projection_active() -> bool:
    res = subprocess.run(
        ["/bin/bash", "-lc", "pgrep -af 'run_projection_sweep_xgb.py' || true"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in res.stdout.splitlines():
        if "pgrep -af" in line:
            continue
        if "run_projection_sweep_xgb.py" in line:
            return True
    return False


def _wait_for_sweep(report_path: Path, timeout_sec: int, poll_sec: int) -> None:
    deadline = time.time() + max(60, int(timeout_sec))
    while True:
        active = _projection_active()
        ready = report_path.exists()
        if ready and not active:
            _log(f"Sweep complete: {report_path}")
            return
        if time.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for sweep completion. active={active} report_exists={ready}")
        _log(f"Waiting for projection sweep... active={active} report_exists={ready}")
        time.sleep(max(5, int(poll_sec)))


def _best_projected_combo(report: Dict[str, Any]) -> Tuple[str, int, Dict[str, Dict[str, Any]]]:
    rows = list(report.get("rows") or [])
    by_combo: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method") or "").strip().lower()
        if method not in {"pca", "jl"}:
            continue
        try:
            dim = int(row.get("dim"))
        except Exception:
            continue
        variant = str(row.get("variant") or "").strip()
        if not variant:
            continue
        combo = (method, dim)
        by_combo.setdefault(combo, {})[variant] = row

    best_combo: Optional[Tuple[str, int]] = None
    best_score = -1.0
    best_rows: Dict[str, Dict[str, Any]] = {}
    for combo, variant_rows in by_combo.items():
        if "nonwindow_20c8" not in variant_rows or "window_ceab" not in variant_rows:
            continue
        f1_non = float(variant_rows["nonwindow_20c8"].get("f1") or 0.0)
        f1_win = float(variant_rows["window_ceab"].get("f1") or 0.0)
        mean_f1 = 0.5 * (f1_non + f1_win)
        if mean_f1 > best_score:
            best_score = mean_f1
            best_combo = combo
            best_rows = variant_rows

    if best_combo is None:
        raise RuntimeError("No complete projected combo found in sweep report.")
    return best_combo[0], best_combo[1], best_rows


def _symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def _prepare_staging(
    *,
    run_dir: Path,
    method: str,
    dim: int,
) -> Path:
    stage_dir = run_dir / f"hybrid_after_sweep_{method}_d{int(dim)}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    src_root = run_dir / "projection_sweep"

    for variant in ("nonwindow_20c8", "window_ceab"):
        stem = f"{variant}.{method}.d{int(dim)}"
        _symlink_force(src_root / f"{stem}.labeled.npz", stage_dir / f"{variant}.labeled.npz")
        _symlink_force(src_root / f"{stem}.json", stage_dir / f"{variant}.json")
        _symlink_force(src_root / f"{stem}.meta.json", stage_dir / f"{variant}.meta.json")
        _symlink_force(src_root / f"{stem}.eval.json", stage_dir / f"{variant}.eval.json")

    return stage_dir


def _baseline_rows(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in (report.get("rows") or []):
        if str(row.get("method") or "") == "xgb_baseline_1024":
            variant = str(row.get("variant") or "").strip()
            if variant:
                out[variant] = row
    return out


def _make_summary(
    *,
    method: str,
    dim: int,
    projected_xgb_rows: Dict[str, Dict[str, Any]],
    baseline_1024_rows: Dict[str, Dict[str, Any]],
    hybrid_report_path: Path,
    out_path: Path,
) -> None:
    hybrid_report = json.loads(hybrid_report_path.read_text(encoding="utf-8"))
    rows = list(hybrid_report.get("summary_rows") or [])

    summary_rows: List[Dict[str, Any]] = []
    for variant in ("nonwindow_20c8", "window_ceab"):
        base_1024 = baseline_1024_rows.get(variant, {})
        proj_xgb = projected_xgb_rows.get(variant, {})
        summary_rows.append(
            {
                "variant": variant,
                "method": "xgb_1024_baseline",
                "precision": float(base_1024.get("precision") or 0.0),
                "recall": float(base_1024.get("recall") or 0.0),
                "f1": float(base_1024.get("f1") or 0.0),
            }
        )
        summary_rows.append(
            {
                "variant": variant,
                "method": f"xgb_{method}_d{int(dim)}",
                "precision": float(proj_xgb.get("precision") or 0.0),
                "recall": float(proj_xgb.get("recall") or 0.0),
                "f1": float(proj_xgb.get("f1") or 0.0),
            }
        )
        for hybrid_name in ("hybrid_lr_xgb_blend", "hybrid_mlp_xgb_blend"):
            hit = next(
                (
                    r
                    for r in rows
                    if str(r.get("variant") or "") == variant and str(r.get("method") or "") == hybrid_name
                ),
                None,
            )
            if hit is None:
                continue
            summary_rows.append(
                {
                    "variant": variant,
                    "method": f"{hybrid_name}_{method}_d{int(dim)}",
                    "precision": float(hit.get("precision") or 0.0),
                    "recall": float(hit.get("recall") or 0.0),
                    "f1": float(hit.get("f1") or 0.0),
                }
            )

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_projection": {"method": method, "dim": int(dim)},
        "hybrid_report": str(hybrid_report_path),
        "rows": summary_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for projection sweep, pick best projection, run hybrid followup automatically."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--fixed-val-images", default="uploads/calibration_jobs/fixed_val_qwen_dataset_2000_images.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    sweep_report = run_dir / "projection_sweep" / "projection_sweep_report.json"

    _wait_for_sweep(sweep_report, timeout_sec=int(args.timeout_sec), poll_sec=int(args.poll_sec))
    report = json.loads(sweep_report.read_text(encoding="utf-8"))
    method, dim, projected_rows = _best_projected_combo(report)
    baseline_1024 = _baseline_rows(report)
    _log(f"Selected best projection by mean F1: {method}.d{int(dim)}")

    stage_dir = _prepare_staging(run_dir=run_dir, method=method, dim=dim)
    _log(f"Prepared hybrid staging dir: {stage_dir}")

    cmd = [
        "python",
        "tools/run_hybrid_followup_after_xgb.py",
        "--run-dir",
        str(stage_dir),
        "--fixed-val-images",
        str(Path(args.fixed_val_images).resolve()),
        "--timeout-sec",
        "600",
        "--poll-sec",
        "10",
    ]
    _log("Launching hybrid followup runner on selected projection.")
    _run(cmd)

    hybrid_report = stage_dir / "hybrid_followup_report.json"
    summary_path = stage_dir / "selected_projection_hybrid_summary.json"
    _make_summary(
        method=method,
        dim=dim,
        projected_xgb_rows=projected_rows,
        baseline_1024_rows=baseline_1024,
        hybrid_report_path=hybrid_report,
        out_path=summary_path,
    )
    _log(f"Wrote summary: {summary_path}")
    print(json.dumps({"staging_dir": str(stage_dir), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
