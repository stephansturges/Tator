#!/usr/bin/env python3
"""Sweep SAM bias magnitudes under sam_only scope after the post-run scope ablation."""

from __future__ import annotations

import argparse
import copy
import json
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
import tools.run_postrun_sam_bias_scope_ablation as scope_ablation


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


def _parse_bias_grid(raw: str, *, default: Sequence[float]) -> List[float]:
    if not str(raw or "").strip():
        return [float(v) for v in default]
    out: List[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value not in out:
            out.append(value)
    return out


def _bias_tag(text_bias: float, sim_bias: float) -> str:
    def _fmt(value: float) -> str:
        return f"m{abs(value):.1f}".replace(".", "p") if value < 0 else f"p{value:.1f}".replace(".", "p")

    return f"text_{_fmt(text_bias)}__sim_{_fmt(sim_bias)}"


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


def _baseline_means(scope_raw: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    by_view: Dict[str, List[Dict[str, Any]]] = {}
    for row in scope_raw.get("rows", []):
        if str(row.get("scope") or "") != "sam_only":
            continue
        by_view.setdefault(str(row["view"]), []).append(row)
    out: Dict[str, Dict[str, float]] = {}
    for view, rows in by_view.items():
        out[view] = {
            "precision": statistics.fmean(float(row["precision"]) for row in rows),
            "recall": statistics.fmean(float(row["recall"]) for row in rows),
            "f1": statistics.fmean(float(row["f1"]) for row in rows),
            "delta_vs_union_f1": statistics.fmean(float(row["delta_vs_union_f1"]) for row in rows),
            "coverage_preservation": statistics.fmean(float(row["coverage_preservation"]) for row in rows),
        }
    return out


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
            }
        )
    return contexts


def _prepare_policy(src_policy: Path, *, text_bias: float, sim_bias: float) -> Dict[str, Any]:
    payload = copy.deepcopy(_load_json(src_policy))
    payload["sam_bias_scope"] = "sam_only"
    mapping = payload.get("logit_bias_by_source_class")
    if not isinstance(mapping, dict):
        mapping = {}
    text_map = mapping.get("sam3_text")
    if not isinstance(text_map, dict):
        text_map = {}
    sim_map = mapping.get("sam3_similarity")
    if not isinstance(sim_map, dict):
        sim_map = {}
    text_map["__default__"] = float(text_bias)
    sim_map["__default__"] = float(sim_bias)
    mapping["sam3_text"] = text_map
    mapping["sam3_similarity"] = sim_map
    payload["logit_bias_by_source_class"] = mapping
    return payload


def _run_combo_eval(
    ctx: Dict[str, Any],
    *,
    text_bias: float,
    sim_bias: float,
    output_root: Path,
) -> Dict[str, Any]:
    tag = _bias_tag(text_bias, sim_bias)
    combo_dir = output_root / tag / str(ctx["view"]) / f"seed_{int(ctx['seed'])}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    meta_copy = combo_dir / Path(str(ctx["meta_json"])).name
    policy_copy = combo_dir / "policy.json"
    eval_json = combo_dir / "eval.json"
    tune_done = combo_dir / "thresholds.done"

    policy_payload = _prepare_policy(Path(str(ctx["policy_json"])), text_bias=text_bias, sim_bias=sim_bias)
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
        "tag": tag,
        "text_bias": float(text_bias),
        "sim_bias": float(sim_bias),
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
    }


def _summarize(
    rows: Sequence[Dict[str, Any]],
    *,
    baseline_by_view: Dict[str, Dict[str, float]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["view"]), {}).setdefault(str(row["tag"]), []).append(row)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for view, tag_rows in grouped.items():
        baseline = baseline_by_view[view]
        ranked: List[Dict[str, Any]] = []
        for tag, items in tag_rows.items():
            ranked.append(
                {
                    "tag": tag,
                    "text_bias": float(items[0]["text_bias"]),
                    "sim_bias": float(items[0]["sim_bias"]),
                    "count": len(items),
                    "mean_precision": statistics.fmean(float(row["precision"]) for row in items),
                    "mean_recall": statistics.fmean(float(row["recall"]) for row in items),
                    "mean_f1": statistics.fmean(float(row["f1"]) for row in items),
                    "mean_delta_vs_baseline_f1": statistics.fmean(float(row["f1"]) for row in items)
                    - float(baseline["f1"]),
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


def _promote_tags(
    pilot_ranked: List[Dict[str, Any]],
    *,
    limit: int,
    min_delta: float,
) -> List[str]:
    promoted: List[str] = []
    for row in pilot_ranked:
        if float(row["mean_delta_vs_baseline_f1"]) < float(min_delta):
            continue
        promoted.append(str(row["tag"]))
        if len(promoted) >= int(limit):
            break
    return promoted


def _write_report(
    path: Path,
    *,
    pilot_ranked: List[Dict[str, Any]],
    full_ranked: List[Dict[str, Any]],
    promoted: List[str],
) -> None:
    lines = [
        "# Post-Run SAM Bias Magnitude Sweep",
        "",
        "## Pilot (`intersection`, 3 seeds)",
        "",
        "| tag | text | sim | mean F1 | delta vs baseline F1 | coverage preservation | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pilot_ranked:
        lines.append(
            f"| {row['tag']} | {row['text_bias']:.1f} | {row['sim_bias']:.1f} | "
            f"{row['mean_f1']:.4f} | {row['mean_delta_vs_baseline_f1']:.4f} | "
            f"{row['mean_coverage_preservation']:.4f} | {row['count']} |"
        )
    lines += ["", f"Promoted tags: {', '.join(promoted) if promoted else '(none)'}", ""]
    if full_ranked:
        lines += [
            "## Promoted full-view follow-up (`full`, 3 seeds)",
            "",
            "| tag | text | sim | mean F1 | delta vs baseline F1 | coverage preservation | n |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in full_ranked:
            lines.append(
                f"| {row['tag']} | {row['text_bias']:.1f} | {row['sim_bias']:.1f} | "
                f"{row['mean_f1']:.4f} | {row['mean_delta_vs_baseline_f1']:.4f} | "
                f"{row['mean_coverage_preservation']:.4f} | {row['count']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_decision_summary(
    *,
    run_root: Path,
    winner_lane: str,
    selected_alpha: float,
    pilot_ranked: Sequence[Dict[str, Any]],
    full_ranked: Sequence[Dict[str, Any]],
    promoted: Sequence[str],
) -> Dict[str, Any]:
    full_winner = dict(full_ranked[0]) if full_ranked else {}
    pilot_winner = dict(pilot_ranked[0]) if pilot_ranked else {}
    promoted_config = None
    status = "rejected"
    reason_codes: List[str] = []
    if full_winner and float(full_winner.get("mean_delta_vs_baseline_f1", 0.0)) >= 0.0:
        promoted_config = {
            "sam_bias_scope": "sam_only",
            "sam3_text_bias_default": float(full_winner["text_bias"]),
            "sam3_similarity_bias_default": float(full_winner["sim_bias"]),
        }
        status = "promoted"
        reason_codes.append("positive_full_window_delta")
    else:
        reason_codes.append("no_full_variant_cleared_gate")
    if promoted:
        reason_codes.append("pilot_variants_promoted_to_full")
    else:
        reason_codes.append("no_variant_cleared_pilot_gate")
    return {
        "stage": "postrun_sam_bias_magnitude_sweep",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(run_root),
        "winner_lane": winner_lane,
        "selected_alpha": float(selected_alpha),
        "pilot_winner": pilot_winner,
        "full_winner": full_winner,
        "promoted_tags": list(promoted),
        "status": status,
        "promoted_config": promoted_config,
        "reason_codes": reason_codes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--alpha-root", default="")
    parser.add_argument("--scope-root", default="")
    parser.add_argument("--text-biases", default="")
    parser.add_argument("--sim-biases", default="")
    parser.add_argument("--promote-topk", type=int, default=4)
    parser.add_argument("--pilot-min-delta-f1", type=float, default=0.0)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    alpha_root = (
        Path(args.alpha_root).resolve()
        if str(args.alpha_root).strip()
        else (run_root / "postrun_alpha_extension").resolve()
    )
    scope_root = (
        Path(args.scope_root).resolve()
        if str(args.scope_root).strip()
        else (run_root / "postrun_sam_bias_scope_ablation").resolve()
    )
    if args.wait:
        _wait_for_file(run_root / "final_default_recipe.json", poll_seconds=int(args.poll_seconds))
        _wait_for_file(alpha_root / "results_raw.json", poll_seconds=int(args.poll_seconds))
        _wait_for_file(alpha_root / "results_ranked.json", poll_seconds=int(args.poll_seconds))
        _wait_for_file(scope_root / "results_raw.json", poll_seconds=int(args.poll_seconds))

    final_default = _load_json(run_root / "final_default_recipe.json")
    alpha_raw = _load_json(alpha_root / "results_raw.json")
    alpha_ranked = _load_json(alpha_root / "results_ranked.json")
    scope_raw = _load_json(scope_root / "results_raw.json")
    winner_lane = str(final_default.get("winner_lane") or "").strip()
    if not winner_lane:
        raise RuntimeError("winner_lane missing from final_default_recipe.json")
    selected_alpha = float(alpha_ranked["winner_alpha"])
    baseline_by_view = _baseline_means(scope_raw)
    contexts = _selected_contexts(
        run_root=run_root,
        alpha_raw=alpha_raw,
        winner_lane=winner_lane,
        selected_alpha=selected_alpha,
    )

    text_biases = _parse_bias_grid(args.text_biases, default=[-1.4, -1.2, -1.0, -0.8])
    sim_biases = _parse_bias_grid(args.sim_biases, default=[-1.2, -1.0, -0.8, -0.6])
    output_root = run_root / "postrun_sam_bias_magnitude_sweep"
    output_root.mkdir(parents=True, exist_ok=True)

    pilot_rows: List[Dict[str, Any]] = []
    pilot_contexts = [ctx for ctx in contexts if str(ctx["view"]) == "intersection"]
    for text_bias in text_biases:
        for sim_bias in sim_biases:
            for ctx in pilot_contexts:
                pilot_rows.append(
                    _run_combo_eval(
                        ctx,
                        text_bias=float(text_bias),
                        sim_bias=float(sim_bias),
                        output_root=output_root / "pilot",
                    )
                )

    pilot_ranked = _summarize(pilot_rows, baseline_by_view=baseline_by_view)["intersection"]
    promoted = _promote_tags(
        pilot_ranked,
        limit=int(args.promote_topk),
        min_delta=float(args.pilot_min_delta_f1),
    )

    full_rows: List[Dict[str, Any]] = []
    if promoted:
        full_contexts = [ctx for ctx in contexts if str(ctx["view"]) == "full"]
        by_tag = {(row["tag"]): (row["text_bias"], row["sim_bias"]) for row in pilot_rows}
        for tag in promoted:
            text_bias, sim_bias = by_tag[tag]
            for ctx in full_contexts:
                full_rows.append(
                    _run_combo_eval(
                        ctx,
                        text_bias=float(text_bias),
                        sim_bias=float(sim_bias),
                        output_root=output_root / "full",
                    )
                )

    ranked_payload = {
        "pilot": pilot_ranked,
        "full": _summarize(full_rows, baseline_by_view=baseline_by_view).get("full", []),
        "promoted_tags": promoted,
    }
    decision_summary = _build_decision_summary(
        run_root=run_root,
        winner_lane=winner_lane,
        selected_alpha=selected_alpha,
        pilot_ranked=pilot_ranked,
        full_ranked=ranked_payload["full"],
        promoted=promoted,
    )
    raw_payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(run_root),
        "alpha_root": str(alpha_root),
        "scope_root": str(scope_root),
        "winner_lane": winner_lane,
        "selected_alpha": selected_alpha,
        "text_biases": text_biases,
        "sim_biases": sim_biases,
        "pilot_rows": pilot_rows,
        "full_rows": full_rows,
        "promoted_tags": promoted,
    }
    _write_json(output_root / "results_ranked.json", ranked_payload)
    _write_json(output_root / "results_raw.json", raw_payload)
    _write_json(output_root / "decision_summary.json", decision_summary)
    _write_report(
        output_root / "report.md",
        pilot_ranked=pilot_ranked,
        full_ranked=ranked_payload["full"],
        promoted=promoted,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "winner_lane": winner_lane,
                "selected_alpha": selected_alpha,
                "output_root": str(output_root),
                "promoted_tags": promoted,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
