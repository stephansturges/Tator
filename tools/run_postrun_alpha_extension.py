#!/usr/bin/env python3
"""Extend the winning SAM3 text-quality alpha after the main sweep completes."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalContext:
    lane: str
    view: str
    seed: int
    variant: str
    model_json: Path
    meta_json: Path
    policy_json: Path
    labeled_npz: Path
    prepass_jsonl: Path
    baseline_eval_json: Path


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
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        capture_output=capture,
    )


def _parse_alphas(raw: str) -> List[float]:
    out: List[float] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        alpha = max(0.0, min(1.0, float(token)))
        if alpha not in out:
            out.append(alpha)
    return out


def _alpha_tag(alpha: float) -> str:
    return f"a{int(round(float(alpha) * 100)):02d}"


def _wait_for_completion(run_root: Path, *, poll_seconds: int) -> None:
    required = [
        run_root / "results_raw.json",
        run_root / "results_ranked.json",
        run_root / "final_default_recipe.json",
        run_root / "final_report.md",
    ]
    while True:
        if all(path.exists() for path in required):
            return
        time.sleep(max(5, int(poll_seconds)))


def _main_eval_candidates(seed_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in seed_dir.glob("eval_*.json")
        if not path.name.endswith(".analysis.json")
    )


def _winner_contexts(run_root: Path, winner_lane: str) -> List[EvalContext]:
    manifest = _load_json(run_root / "lane_manifest.json")
    lane_cfg = manifest["lanes"][winner_lane]
    variant = str(lane_cfg["variant"])
    contexts: List[EvalContext] = []
    final_lane_root = run_root / "final_matrix" / winner_lane
    for view_dir in sorted([p for p in final_lane_root.iterdir() if p.is_dir()]):
        view = view_dir.name
        if view == "full":
            labeled_npz = Path(str(lane_cfg["labeled"]))
            prepass_jsonl = Path(str(lane_cfg["prepass_jsonl"]))
        elif view == "intersection":
            labeled_npz = Path(str(manifest["intersection_labeled"][winner_lane]["path"]))
            prepass_jsonl = Path(str(manifest["intersection_prepass_jsonl"][variant]))
        else:
            raise ValueError(f"Unsupported view: {view}")
        for seed_dir in sorted([p for p in view_dir.iterdir() if p.is_dir()]):
            meta_candidates = sorted(seed_dir.glob("model_*.meta.json"))
            policy_candidates = sorted(seed_dir.glob("policy_*.json"))
            eval_candidates = _main_eval_candidates(seed_dir)
            if len(meta_candidates) != 1 or len(policy_candidates) != 1 or len(eval_candidates) != 1:
                raise RuntimeError(
                    f"Expected exactly one main meta/policy/eval in {seed_dir}, got "
                    f"{len(meta_candidates)}/{len(policy_candidates)}/{len(eval_candidates)}"
                )
            meta_json = meta_candidates[0]
            model_json = meta_json.with_name(meta_json.name.replace(".meta.json", ".json"))
            if not model_json.exists():
                raise FileNotFoundError(f"Missing base model for {meta_json}")
            seed = int(str(seed_dir.name).replace("seed_", ""))
            contexts.append(
                EvalContext(
                    lane=winner_lane,
                    view=view,
                    seed=seed,
                    variant=variant,
                    model_json=model_json,
                    meta_json=meta_json,
                    policy_json=policy_candidates[0],
                    labeled_npz=labeled_npz,
                    prepass_jsonl=prepass_jsonl,
                    baseline_eval_json=eval_candidates[0],
                )
            )
    return contexts


def _baseline_rows(contexts: Sequence[EvalContext], *, baseline_alpha: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ctx in contexts:
        payload = _load_json(ctx.baseline_eval_json)
        rows.append(
            {
                "alpha": float(baseline_alpha),
                "lane": ctx.lane,
                "view": ctx.view,
                "seed": int(ctx.seed),
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
                "coverage_upper_bound": float(
                    payload.get("coverage_upper_bound", {})
                    .get("candidate_all", {})
                    .get("recall_upper_bound", 0.0)
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
                "eval_json": str(ctx.baseline_eval_json),
                "meta_json": str(ctx.meta_json),
                "policy_json": str(ctx.policy_json),
                "mode": "baseline_final_matrix",
            }
        )
    return rows


def _objective_args(meta: Dict[str, Any]) -> Dict[str, Any]:
    params = meta.get("calibration_objective_params") if isinstance(meta.get("calibration_objective_params"), dict) else {}
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


def _prepare_meta_copy(src_meta: Path, dest_meta: Path, *, alpha: float) -> Dict[str, Any]:
    payload = _load_json(src_meta)
    quality_cfg = payload.get("sam3_text_quality") if isinstance(payload.get("sam3_text_quality"), dict) else {}
    if not bool(quality_cfg.get("enabled")):
        raise RuntimeError(f"Winner meta has no SAM3 text-quality head enabled: {src_meta}")
    quality_cfg = dict(quality_cfg)
    quality_cfg["alpha"] = float(alpha)
    payload["sam3_text_quality"] = quality_cfg
    dest_meta.parent.mkdir(parents=True, exist_ok=True)
    dest_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_alpha_eval(
    ctx: EvalContext,
    *,
    alpha: float,
    output_root: Path,
) -> Dict[str, Any]:
    alpha_dir = output_root / ctx.lane / ctx.view / f"seed_{ctx.seed}" / _alpha_tag(alpha)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    meta_copy = alpha_dir / ctx.meta_json.name
    policy_copy = alpha_dir / ctx.policy_json.name
    eval_json = alpha_dir / f"{ctx.baseline_eval_json.stem}.{_alpha_tag(alpha)}.json"
    tune_done = alpha_dir / f"{ctx.meta_json.stem}.{_alpha_tag(alpha)}.tuned.done"

    meta_payload = _prepare_meta_copy(ctx.meta_json, meta_copy, alpha=alpha)
    if not policy_copy.exists():
        shutil.copy2(ctx.policy_json, policy_copy)

    params = _objective_args(meta_payload)
    if not tune_done.exists():
        _run(
            [
                sys.executable,
                "tools/tune_ensemble_thresholds_xgb.py",
                "--model",
                str(ctx.model_json),
                "--meta",
                str(meta_copy),
                "--data",
                str(ctx.labeled_npz),
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
            + (["--use-val-split"] if params["use_val_split"] else [])
        )
        tune_done.write_text("ok\n", encoding="utf-8")

    if not eval_json.exists():
        result = _run(
            [
                sys.executable,
                "tools/eval_ensemble_xgb_dedupe.py",
                "--model",
                str(ctx.model_json),
                "--meta",
                str(meta_copy),
                "--data",
                str(ctx.labeled_npz),
                "--dataset",
                "qwen_dataset",
                "--prepass-jsonl",
                str(ctx.prepass_jsonl),
                "--eval-iou",
                str(params["eval_iou"]),
                "--eval-iou-grid",
                str(params["eval_iou"]),
                "--dedupe-iou",
                str(params["dedupe_iou"]),
                "--scoreless-iou",
                str(params["scoreless_iou"]),
                "--policy-json",
                str(policy_copy),
            ]
            + (["--use-val-split"] if params["use_val_split"] else []),
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
        "alpha": float(alpha),
        "lane": ctx.lane,
        "view": ctx.view,
        "seed": int(ctx.seed),
        "precision": float(payload["precision"]),
        "recall": float(payload["recall"]),
        "f1": float(payload["f1"]),
        "delta_vs_union_f1": float(payload["f1"]) - union_f1,
        "coverage_upper_bound": coverage_ub,
        "coverage_preservation": (float(payload["recall"]) / coverage_ub) if coverage_ub > 0.0 else 0.0,
        "eval_json": str(eval_json),
        "meta_json": str(meta_copy),
        "policy_json": str(policy_copy),
        "mode": "alpha_extension",
    }


def _summarize(rows: Sequence[Dict[str, Any]], *, selection_view: str) -> Dict[str, Any]:
    grouped: Dict[Tuple[float, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((float(row["alpha"]), str(row["view"])), []).append(row)

    by_view: Dict[str, List[Dict[str, Any]]] = {}
    for (alpha, view), group in grouped.items():
        by_view.setdefault(view, []).append(
            {
                "alpha": alpha,
                "count": len(group),
                "mean_precision": statistics.mean(float(x["precision"]) for x in group),
                "mean_recall": statistics.mean(float(x["recall"]) for x in group),
                "mean_f1": statistics.mean(float(x["f1"]) for x in group),
                "mean_delta_vs_union_f1": statistics.mean(float(x["delta_vs_union_f1"]) for x in group),
                "mean_coverage_preservation": statistics.mean(float(x["coverage_preservation"]) for x in group),
            }
        )

    for view_rows in by_view.values():
        view_rows.sort(
            key=lambda row: (
                -float(row["mean_f1"]),
                -float(row["mean_delta_vs_union_f1"]),
                -float(row["mean_coverage_preservation"]),
                float(row["alpha"]),
            )
        )

    selection_rows = by_view.get(selection_view, [])
    winner = selection_rows[0] if selection_rows else {}
    return {
        "selection_view": selection_view,
        "views": by_view,
        "winner_alpha": winner.get("alpha"),
        "winner_metrics": winner,
    }


def _write_report(
    path: Path,
    *,
    winner_lane: str,
    baseline_alpha: float,
    requested_alphas: Sequence[float],
    summary: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Post-Run SAM Quality Alpha Extension")
    lines.append("")
    lines.append(f"- Generated UTC: `{_ts()}`")
    lines.append(f"- Winner lane from main sweep: `{winner_lane}`")
    lines.append(f"- Baseline alpha from main sweep: `{baseline_alpha}`")
    lines.append(f"- Requested extension alphas: `{', '.join(str(a) for a in requested_alphas)}`")
    lines.append(f"- Selection view: `{summary.get('selection_view')}`")
    lines.append(f"- Selected alpha: `{summary.get('winner_alpha')}`")
    lines.append("")
    for view in ("intersection", "full"):
        rows = summary.get("views", {}).get(view, [])
        if not rows:
            continue
        lines.append(f"## {view.title()} view")
        lines.append("")
        lines.append("| alpha | mean precision | mean recall | mean F1 | delta vs union F1 | coverage preservation | n |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(
                f"| {row['alpha']:.2f} | {row['mean_precision']:.4f} | {row['mean_recall']:.4f} | "
                f"{row['mean_f1']:.4f} | {row['mean_delta_vs_union_f1']:.4f} | "
                f"{row['mean_coverage_preservation']:.4f} | {row['count']} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _selection_baseline_metrics(
    summary: Dict[str, Any],
    *,
    selection_view: str,
    baseline_alpha: float,
) -> Dict[str, Any]:
    rows = summary.get("views", {}).get(selection_view, [])
    for row in rows:
        if abs(float(row.get("alpha", -1.0)) - float(baseline_alpha)) <= 1e-9:
            return dict(row)
    return {}


def _build_decision_summary(
    *,
    run_root: Path,
    winner_lane: str,
    baseline_alpha: float,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    selection_view = str(summary.get("selection_view") or "intersection")
    winner_metrics = dict(summary.get("winner_metrics") or {})
    baseline_metrics = _selection_baseline_metrics(
        summary,
        selection_view=selection_view,
        baseline_alpha=float(baseline_alpha),
    )
    winner_alpha = winner_metrics.get("alpha")
    promoted_alpha = float(winner_alpha) if winner_alpha is not None else float(baseline_alpha)
    reason_codes = ["best_mean_f1_on_selection_view"]
    if abs(promoted_alpha - float(baseline_alpha)) > 1e-9:
        reason_codes.append("promoted_alpha_differs_from_main_sweep")
    else:
        reason_codes.append("main_sweep_alpha_retained")
    return {
        "stage": "postrun_alpha_extension",
        "generated_utc": _ts(),
        "run_root": str(run_root),
        "winner_lane": winner_lane,
        "selection_view": selection_view,
        "baseline_alpha": float(baseline_alpha),
        "winner_alpha": promoted_alpha,
        "baseline_metrics": baseline_metrics,
        "winner_metrics": winner_metrics,
        "promoted_config": {
            "train_sam3_text_quality": True,
            "sam3_text_quality_alpha": promoted_alpha,
        },
        "reason_codes": reason_codes,
        "status": "selected",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend winning SAM quality alpha after final sweep completion.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--alphas", default="0.6,0.7,0.8")
    parser.add_argument("--selection-view", default="intersection", choices=["intersection", "full"])
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    run_root = (REPO_ROOT / args.run_root).resolve()
    if args.wait:
        _wait_for_completion(run_root, poll_seconds=int(args.poll_seconds))

    final_default = _load_json(run_root / "final_default_recipe.json")
    winner_lane = str(final_default.get("winner_lane") or "").strip()
    if not winner_lane:
        raise RuntimeError("No winner lane recorded in final_default_recipe.json")

    lane_settings = final_default.get("lane_settings") if isinstance(final_default.get("lane_settings"), dict) else {}
    scenario = lane_settings.get("scenario") if isinstance(lane_settings.get("scenario"), dict) else {}
    if not bool(scenario.get("sam_quality")):
        raise RuntimeError("Winning lane does not have sam_quality enabled; alpha extension is not applicable.")
    baseline_alpha = float(scenario.get("alpha", 0.0))
    extension_alphas = _parse_alphas(args.alphas)
    all_alphas = [baseline_alpha] + [alpha for alpha in extension_alphas if abs(alpha - baseline_alpha) > 1e-12]

    output_root = Path(args.output_root).resolve() if args.output_root else run_root / "postrun_alpha_extension"
    output_root.mkdir(parents=True, exist_ok=True)

    contexts = _winner_contexts(run_root, winner_lane)
    baseline_rows = _baseline_rows(contexts, baseline_alpha=baseline_alpha)
    rows: List[Dict[str, Any]] = list(baseline_rows)
    for alpha in all_alphas:
        if abs(alpha - baseline_alpha) <= 1e-12:
            continue
        for ctx in contexts:
            rows.append(_run_alpha_eval(ctx, alpha=alpha, output_root=output_root))

    summary = _summarize(rows, selection_view=str(args.selection_view))
    decision_summary = _build_decision_summary(
        run_root=run_root,
        winner_lane=winner_lane,
        baseline_alpha=baseline_alpha,
        summary=summary,
    )
    raw_payload = {
        "generated_utc": _ts(),
        "run_root": str(run_root),
        "winner_lane": winner_lane,
        "baseline_alpha": baseline_alpha,
        "requested_alphas": extension_alphas,
        "rows": rows,
        "summary": summary,
    }
    _write_json(output_root / "results_raw.json", raw_payload)
    _write_json(output_root / "results_ranked.json", summary)
    _write_json(output_root / "decision_summary.json", decision_summary)
    _write_report(
        output_root / "report.md",
        winner_lane=winner_lane,
        baseline_alpha=baseline_alpha,
        requested_alphas=extension_alphas,
        summary=summary,
    )
    print(json.dumps({"status": "completed", "winner_lane": winner_lane, "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
