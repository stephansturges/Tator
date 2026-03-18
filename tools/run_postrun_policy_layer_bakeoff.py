#!/usr/bin/env python3
"""Run learned-policy bakeoff variants against the frozen postrun hand-policy baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_best_refined_tag(run_root: Path) -> str:
    ranked = _load_json(run_root / "postrun_sam_bias_magnitude_sweep" / "results_ranked.json")
    pilot_rows = ranked.get("pilot") if isinstance(ranked.get("pilot"), list) else []
    if not pilot_rows:
        raise SystemExit("missing_refined_policy_rankings")
    return str((pilot_rows[0] or {}).get("tag") or "").strip()


def _resolve_seed_meta_paths(run_root: Path, refined_tag: str) -> Dict[str, Path]:
    seed_meta: Dict[str, Path] = {}
    for seed_dir in sorted((run_root / "postrun_sam_bias_magnitude_sweep" / "pilot" / refined_tag / "intersection").glob("seed_*")):
        meta_candidates = sorted(seed_dir.glob("*.meta.json"))
        if not meta_candidates:
            continue
        seed_meta[seed_dir.name.replace("seed_", "")] = meta_candidates[0]
    if not seed_meta:
        raise SystemExit("missing_refined_seed_meta")
    return seed_meta


def _mean_metrics(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    if not rows:
        return {}
    fields = ["precision", "recall", "f1", "tp", "fp", "fn"]
    source = [(row.get(key) or {}) for row in rows]
    out: Dict[str, float] = {}
    for field in fields:
        values = [_safe_float(item.get(field), 0.0) for item in source]
        out[f"mean_{field}"] = sum(values) / len(values)
    return out


def _mean_subgroup_delta(rows: List[Dict[str, Any]], subgroup: str) -> Dict[str, float]:
    items = []
    for row in rows:
        compare = row.get("compare_to_baseline") or {}
        subgroups = compare.get("subgroups") if isinstance(compare.get("subgroups"), dict) else {}
        item = subgroups.get(subgroup) if isinstance(subgroups.get(subgroup), dict) else {}
        items.append(item)
    if not items:
        return {}
    out: Dict[str, float] = {}
    for field in ["delta_f1", "delta_tp", "delta_fp", "delta_fn"]:
        values = [_safe_float(item.get(field), 0.0) for item in items]
        out[f"mean_{field}"] = sum(values) / len(values)
    return out


def _render_report(summary: Dict[str, Any]) -> str:
    lines = ["# Postrun Learned Policy Bakeoff", ""]
    lines.append(f"- Run root: `{summary['run_root']}`")
    lines.append(f"- Lane: `{summary['lane']}`")
    lines.append(f"- View: `{summary['view']}`")
    lines.append(f"- Refined hand-policy tag: `{summary['refined_tag']}`")
    lines.append("")
    for variant in ["bakeoff", "xgb", "lreg"]:
        row = (summary.get("variants") or {}).get(variant)
        if not row:
            continue
        lines.append(f"## `{variant}`")
        lines.append("")
        lines.append(f"- Seeds: `{', '.join(row.get('seeds', []))}`")
        if variant == "bakeoff":
            lines.append(f"- Selected-family counts: `{row.get('selected_variant_counts', {})}`")
        metrics = row.get("selected_metrics_mean") or {}
        baseline = row.get("baseline_metrics_mean") or {}
        compare = row.get("compare_mean") or {}
        sam_only = row.get("sam_only_delta_mean") or {}
        sim_only = row.get("sam3_similarity_primary_delta_mean") or {}
        lines.append(
            f"- Selected mean: P={metrics.get('mean_precision', 0.0):.4f} "
            f"R={metrics.get('mean_recall', 0.0):.4f} F1={metrics.get('mean_f1', 0.0):.4f}"
        )
        lines.append(
            f"- Baseline mean: P={baseline.get('mean_precision', 0.0):.4f} "
            f"R={baseline.get('mean_recall', 0.0):.4f} F1={baseline.get('mean_f1', 0.0):.4f}"
        )
        lines.append(
            f"- Delta vs refined hand baseline: "
            f"dF1={compare.get('mean_delta_f1', 0.0):+.4f} "
            f"dTP={compare.get('mean_delta_tp', 0.0):+.1f} "
            f"dFP={compare.get('mean_delta_fp', 0.0):+.1f} "
            f"dFN={compare.get('mean_delta_fn', 0.0):+.1f}"
        )
        lines.append(
            f"- `sam_only` subgroup deltas: "
            f"dF1={sam_only.get('mean_delta_f1', 0.0):+.4f} "
            f"dTP={sam_only.get('mean_delta_tp', 0.0):+.1f} "
            f"dFP={sam_only.get('mean_delta_fp', 0.0):+.1f}"
        )
        lines.append(
            f"- `sam3_similarity_primary` subgroup deltas: "
            f"dF1={sim_only.get('mean_delta_f1', 0.0):+.4f} "
            f"dTP={sim_only.get('mean_delta_tp', 0.0):+.1f} "
            f"dFP={sim_only.get('mean_delta_fp', 0.0):+.1f}"
        )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learned-policy bakeoff on frozen postrun artifacts.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--lane", default="window")
    parser.add_argument("--view", default="intersection", choices=["intersection"])
    parser.add_argument("--variant", action="append", choices=["bakeoff", "xgb", "lreg"])
    parser.add_argument("--refined-tag", default=None)
    parser.add_argument("--labeled-npz", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--nested-folds", type=int, default=5)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    refined_tag = args.refined_tag or _resolve_best_refined_tag(run_root)
    labeled_npz = Path(args.labeled_npz).resolve() if args.labeled_npz else (run_root / "views" / f"{args.lane}_{args.view}.labeled.npz").resolve()
    if not labeled_npz.exists():
        raise SystemExit(f"missing_labeled_npz:{labeled_npz}")

    seed_meta_paths = _resolve_seed_meta_paths(run_root, refined_tag)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_root / "postrun_policy_layer_bakeoff").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = args.variant or ["bakeoff", "xgb", "lreg"]

    raw_rows: List[Dict[str, Any]] = []
    for variant in variants:
        for seed, meta_path in sorted(seed_meta_paths.items(), key=lambda item: int(item[0])):
            variant_dir = output_dir / variant / f"seed_{seed}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            meta_copy = variant_dir / meta_path.name
            shutil.copy2(meta_path, meta_copy)
            base_meta = _load_json(meta_copy)
            model_path = Path(str(base_meta.get("model_path") or "")).resolve()
            if not model_path.exists():
                raise SystemExit(f"missing_model_path:{model_path}")
            policy_dir = variant_dir / "policy_layer"
            cmd = [
                sys.executable,
                str(ROOT / "tools" / "train_policy_layer.py"),
                "--input",
                str(labeled_npz),
                "--base-model",
                str(model_path),
                "--base-meta",
                str(meta_copy),
                "--output-dir",
                str(policy_dir),
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--nested-folds",
                str(int(args.nested_folds)),
            ]
            subprocess.run(cmd, cwd=str(ROOT), check=True)
            selection = _load_json(policy_dir / "policy_layer_selection.json")
            raw_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "selection": selection,
                }
            )

    summary: Dict[str, Any] = {
        "run_root": str(run_root),
        "lane": args.lane,
        "view": args.view,
        "refined_tag": refined_tag,
        "labeled_npz": str(labeled_npz),
        "variants": {},
    }
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_variant[str(row["variant"])].append(row)

    for variant, rows in by_variant.items():
        selected_variant_counts = Counter(str((row.get("selection") or {}).get("selected_variant") or "") for row in rows)
        selected_rows = []
        baseline_rows = []
        compare_rows = []
        for row in rows:
            selection = row.get("selection") or {}
            selected_variant = str(selection.get("selected_variant") or "")
            candidate_row = ((selection.get("candidates") or {}).get(selected_variant) or {})
            selected_rows.append(candidate_row.get("metrics") or {})
            baseline_rows.append(selection.get("baseline") or {})
            compare_rows.append(candidate_row.get("compare_to_baseline") or {})
        summary["variants"][variant] = {
            "seeds": [str(row["seed"]) for row in rows],
            "selected_variant_counts": dict(selected_variant_counts),
            "selected_metrics_mean": _mean_metrics([{"selected": item} for item in selected_rows], "selected"),
            "baseline_metrics_mean": _mean_metrics([{"baseline": item} for item in baseline_rows], "baseline"),
            "compare_mean": {
                f"mean_{field}": sum(_safe_float(item.get(field), 0.0) for item in compare_rows) / len(compare_rows)
                for field in ["delta_f1", "delta_precision", "delta_recall", "delta_tp", "delta_fp", "delta_fn"]
            },
            "sam_only_delta_mean": _mean_subgroup_delta(
                [{"compare_to_baseline": item} for item in compare_rows],
                "sam_only",
            ),
            "sam3_similarity_primary_delta_mean": _mean_subgroup_delta(
                [{"compare_to_baseline": item} for item in compare_rows],
                "sam3_similarity_primary",
            ),
        }

    _write_json(output_dir / "results_raw.json", {"runs": raw_rows})
    _write_json(output_dir / "results_summary.json", summary)
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
