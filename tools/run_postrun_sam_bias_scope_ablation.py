#!/usr/bin/env python3
"""Ablate SAM source-bias scope on the post-run winning configuration."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.run_postrun_alpha_extension as alpha_ext


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run(cmd: Sequence[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        capture_output=capture,
    )


def _wait_for_file(path: Path, *, poll_seconds: int) -> None:
    while not path.exists():
        time.sleep(max(5, int(poll_seconds)))


def _scope_tag(scope: str) -> str:
    return str(scope or "").strip().lower().replace("-", "_")


def _parse_scopes(raw: str) -> List[str]:
    scopes: List[str] = []
    for token in str(raw or "").split(","):
        scope = _scope_tag(token)
        if not scope:
            continue
        if scope not in {"primary_source", "sam_only"}:
            raise ValueError(f"Unsupported sam_bias_scope: {token}")
        if scope not in scopes:
            scopes.append(scope)
    return scopes or ["sam_only"]


def _objective_args(meta: Dict[str, Any]) -> Dict[str, Any]:
    params = (
        meta.get("calibration_objective_params")
        if isinstance(meta.get("calibration_objective_params"), dict)
        else {}
    )
    return {
        "optimize": str(params.get("optimize") or meta.get("calibration_optimize") or "f1"),
        "target_fp_ratio": float(params.get("target_fp_ratio", 0.2)),
        "min_recall": float(params.get("min_recall", 0.6)),
        "steps": int(params.get("steps", 300)),
        "eval_iou": float(params.get("eval_iou", 0.5)),
        "dedupe_iou": float(params.get("dedupe_iou", 0.75)),
        "scoreless_iou": float(params.get("scoreless_iou", 0.0)),
        "use_val_split": bool(params.get("use_val_split", True)),
        "relax_fp_ratio": float(params.get("target_fp_ratio", 0.2)),
    }


def _baseline_rows(alpha_raw: Dict[str, Any], *, selected_alpha: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in alpha_raw.get("rows", []):
        if abs(float(row.get("alpha", -1.0)) - float(selected_alpha)) > 1e-9:
            continue
        copied = dict(row)
        copied["mode"] = "baseline_alpha"
        copied["scope"] = "primary_source"
        rows.append(copied)
    return rows


def _selected_contexts(
    *,
    run_root: Path,
    alpha_raw: Dict[str, Any],
    winner_lane: str,
    selected_alpha: float,
) -> List[Dict[str, Any]]:
    base_contexts = alpha_ext._winner_contexts(run_root, winner_lane)
    alpha_rows = {
        (str(row["view"]), int(row["seed"])): row
        for row in alpha_raw.get("rows", [])
        if abs(float(row.get("alpha", -1.0)) - float(selected_alpha)) <= 1e-9
    }
    contexts: List[Dict[str, Any]] = []
    for ctx in base_contexts:
        row = alpha_rows.get((ctx.view, int(ctx.seed)))
        if row is None:
            raise RuntimeError(f"Missing alpha-extension row for {ctx.view}/seed_{ctx.seed}")
        contexts.append(
            {
                "lane": ctx.lane,
                "view": ctx.view,
                "seed": int(ctx.seed),
                "variant": ctx.variant,
                "model_json": ctx.model_json,
                "labeled_npz": ctx.labeled_npz,
                "prepass_jsonl": ctx.prepass_jsonl,
                "meta_json": Path(str(row["meta_json"])),
                "policy_json": Path(str(row["policy_json"])),
                "baseline_eval_json": Path(str(row["eval_json"])),
            }
        )
    return contexts


def _run_scope_eval(
    ctx: Dict[str, Any],
    *,
    scope: str,
    output_root: Path,
) -> Dict[str, Any]:
    scope_dir = output_root / _scope_tag(scope) / str(ctx["view"]) / f"seed_{int(ctx['seed'])}"
    scope_dir.mkdir(parents=True, exist_ok=True)
    meta_copy = scope_dir / Path(str(ctx["meta_json"])).name
    policy_copy = scope_dir / Path(str(ctx["policy_json"])).name
    eval_json = scope_dir / f"{Path(str(ctx['baseline_eval_json'])).stem}.{_scope_tag(scope)}.json"
    tune_done = scope_dir / "thresholds.done"

    policy_payload = _load_json(Path(str(ctx["policy_json"])))
    policy_payload["sam_bias_scope"] = _scope_tag(scope)
    _write_json(policy_copy, policy_payload)

    meta_payload = _load_json(Path(str(ctx["meta_json"])))
    meta_payload["ensemble_policy"] = policy_payload
    _write_json(meta_copy, meta_payload)
    params = _objective_args(meta_payload)

    if not tune_done.exists():
        cmd = [
            sys.executable,
            "tools/tune_ensemble_thresholds_xgb.py",
            "--model",
            str(ctx["model_json"]),
            "--meta",
            str(meta_copy),
            "--data",
            str(ctx["labeled_npz"]),
            "--dataset",
            "qwen_dataset",
            "--optimize",
            str(params["optimize"]),
            "--target-fp-ratio",
            str(params["target_fp_ratio"]),
            "--relax-fp-ratio",
            str(params["relax_fp_ratio"]),
            "--min-recall",
            str(params["min_recall"]),
            "--steps",
            str(params["steps"]),
            "--eval-iou",
            str(params["eval_iou"]),
            "--dedupe-iou",
            str(params["dedupe_iou"]),
            "--scoreless-iou",
            str(params["scoreless_iou"]),
        ]
        if params["use_val_split"]:
            cmd.append("--use-val-split")
        _run(cmd)
        tune_done.write_text("done\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "tools/eval_ensemble_xgb_dedupe.py",
        "--model",
        str(ctx["model_json"]),
        "--meta",
        str(meta_copy),
        "--data",
        str(ctx["labeled_npz"]),
        "--dataset",
        "qwen_dataset",
        "--eval-iou",
        str(params["eval_iou"]),
        "--dedupe-iou",
        str(params["dedupe_iou"]),
        "--scoreless-iou",
        str(params["scoreless_iou"]),
        "--policy-json",
        str(policy_copy),
        "--prepass-jsonl",
        str(ctx["prepass_jsonl"]),
    ]
    result = _run(cmd, capture=True)
    eval_json.write_text(result.stdout, encoding="utf-8")
    payload = json.loads(result.stdout)
    return {
        "scope": _scope_tag(scope),
        "lane": str(ctx["lane"]),
        "view": str(ctx["view"]),
        "seed": int(ctx["seed"]),
        "precision": float(payload["precision"]),
        "recall": float(payload["recall"]),
        "f1": float(payload["f1"]),
        "delta_vs_union_f1": float(
            payload.get("metric_tiers", {})
            .get("post_xgb", {})
            .get("accepted_all", {})
            .get("f1", payload["f1"])
        )
        - float(
            payload.get("metric_tiers", {})
            .get("post_cluster", {})
            .get("source_attributed", {})
            .get("yolo_rfdetr_union", {})
            .get("f1", 0.0)
        ),
        "coverage_preservation": (
            float(payload["recall"])
            / float(
                payload.get("coverage_upper_bound", {})
                .get("candidate_all", {})
                .get("recall_upper_bound", 0.0)
            )
            if float(
                payload.get("coverage_upper_bound", {})
                .get("candidate_all", {})
                .get("recall_upper_bound", 0.0)
            )
            > 0.0
            else 0.0
        ),
        "eval_json": str(eval_json),
        "policy_json": str(policy_copy),
        "meta_json": str(meta_copy),
        "mode": "scope_ablation",
    }


def _summaries(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    views: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        views.setdefault(str(row["view"]), {}).setdefault(str(row["scope"]), []).append(row)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for view, scope_rows in views.items():
        ranked: List[Dict[str, Any]] = []
        baseline_rows = scope_rows.get("primary_source", [])
        baseline_mean = statistics.fmean(float(row["f1"]) for row in baseline_rows) if baseline_rows else 0.0
        for scope, items in scope_rows.items():
            ranked.append(
                {
                    "scope": scope,
                    "count": len(items),
                    "mean_precision": statistics.fmean(float(row["precision"]) for row in items),
                    "mean_recall": statistics.fmean(float(row["recall"]) for row in items),
                    "mean_f1": statistics.fmean(float(row["f1"]) for row in items),
                    "mean_delta_vs_baseline_f1": statistics.fmean(float(row["f1"]) for row in items)
                    - baseline_mean,
                    "mean_delta_vs_union_f1": statistics.fmean(
                        float(row["delta_vs_union_f1"]) for row in items
                    ),
                    "mean_coverage_preservation": statistics.fmean(
                        float(row["coverage_preservation"]) for row in items
                    ),
                }
            )
        ranked.sort(key=lambda item: item["mean_f1"], reverse=True)
        out[view] = ranked
    return out


def _write_report(path: Path, ranked: Dict[str, List[Dict[str, Any]]]) -> None:
    lines = [
        "# Post-Run SAM Bias Scope Ablation",
        "",
    ]
    for view in sorted(ranked):
        lines += [f"## {view}", "", "| scope | mean F1 | delta vs baseline F1 | coverage preservation | n |", "| --- | ---: | ---: | ---: | ---: |"]
        for row in ranked[view]:
            lines.append(
                f"| {row['scope']} | {row['mean_f1']:.4f} | {row['mean_delta_vs_baseline_f1']:.4f} | "
                f"{row['mean_coverage_preservation']:.4f} | {row['count']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection_view(ranked: Dict[str, List[Dict[str, Any]]]) -> str:
    if ranked.get("intersection"):
        return "intersection"
    if ranked.get("full"):
        return "full"
    return next(iter(ranked.keys()), "intersection")


def _row_by_scope(rows: Sequence[Dict[str, Any]], scope: str) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("scope") or "") == str(scope):
            return dict(row)
    return {}


def _decision_summary(
    *,
    run_root: Path,
    winner_lane: str,
    selected_alpha: float,
    ranked: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    selection_view = _selection_view(ranked)
    selection_rows = ranked.get(selection_view, [])
    winner_metrics = dict(selection_rows[0]) if selection_rows else {}
    promoted_scope = str(winner_metrics.get("scope") or "primary_source")
    baseline_metrics = _row_by_scope(selection_rows, "primary_source")
    reason_codes = ["best_mean_f1_on_selection_view"]
    if promoted_scope != "primary_source":
        reason_codes.append("scope_changed_from_primary_source")
    else:
        reason_codes.append("primary_source_retained")
    return {
        "stage": "postrun_sam_bias_scope_ablation",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(run_root),
        "winner_lane": winner_lane,
        "selected_alpha": float(selected_alpha),
        "selection_view": selection_view,
        "promoted_scope": promoted_scope,
        "baseline_metrics": baseline_metrics,
        "winner_metrics": winner_metrics,
        "promoted_config": {
            "sam_bias_scope": promoted_scope,
        },
        "reason_codes": reason_codes,
        "status": "selected",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--alpha-root", default="")
    parser.add_argument("--scopes", default="sam_only")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    alpha_root = (
        Path(args.alpha_root).resolve()
        if str(args.alpha_root).strip()
        else (run_root / "postrun_alpha_extension").resolve()
    )
    if args.wait:
        _wait_for_file(run_root / "final_default_recipe.json", poll_seconds=int(args.poll_seconds))
        _wait_for_file(alpha_root / "results_raw.json", poll_seconds=int(args.poll_seconds))
        _wait_for_file(alpha_root / "results_ranked.json", poll_seconds=int(args.poll_seconds))

    final_default = _load_json(run_root / "final_default_recipe.json")
    alpha_raw = _load_json(alpha_root / "results_raw.json")
    alpha_ranked = _load_json(alpha_root / "results_ranked.json")
    winner_lane = str(final_default.get("winner_lane") or "").strip()
    if not winner_lane:
        raise RuntimeError("winner_lane missing from final_default_recipe.json")
    selected_alpha = float(alpha_ranked["winner_alpha"])
    scopes = _parse_scopes(args.scopes)

    output_root = run_root / "postrun_sam_bias_scope_ablation"
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_rows = _baseline_rows(alpha_raw, selected_alpha=selected_alpha)
    contexts = _selected_contexts(
        run_root=run_root,
        alpha_raw=alpha_raw,
        winner_lane=winner_lane,
        selected_alpha=selected_alpha,
    )

    rows = list(baseline_rows)
    for scope in scopes:
        for ctx in contexts:
            rows.append(_run_scope_eval(ctx, scope=scope, output_root=output_root))

    ranked = _summaries(rows)
    decision_summary = _decision_summary(
        run_root=run_root,
        winner_lane=winner_lane,
        selected_alpha=selected_alpha,
        ranked=ranked,
    )
    raw_payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(run_root),
        "alpha_root": str(alpha_root),
        "winner_lane": winner_lane,
        "selected_alpha": selected_alpha,
        "rows": rows,
    }
    _write_json(output_root / "results_raw.json", raw_payload)
    _write_json(output_root / "results_ranked.json", ranked)
    _write_json(output_root / "decision_summary.json", decision_summary)
    _write_report(output_root / "report.md", ranked)
    print(
        json.dumps(
            {
                "status": "completed",
                "winner_lane": winner_lane,
                "selected_alpha": selected_alpha,
                "output_root": str(output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
