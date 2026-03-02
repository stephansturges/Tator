#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lane_table(rows: List[Dict[str, Any]]) -> List[str]:
    out = [
        "| Lane | Mean P | Mean R | Mean F1 | Mean ΔvsUnion F1 | Mean CovPres | F1 std | Guardrail Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        out.append(
            "| {lane} | {p:.4f} | {r:.4f} | {f1:.4f} | {du:+.4f} | {cp:.4f} | {f1s:.4f} | {gp} |".format(
                lane=row.get("lane"),
                p=_safe_float(row.get("mean_precision")),
                r=_safe_float(row.get("mean_recall")),
                f1=_safe_float(row.get("mean_f1")),
                du=_safe_float(row.get("mean_delta_vs_union_f1")),
                cp=_safe_float(row.get("mean_coverage_preservation")),
                f1s=_safe_float(row.get("std_f1")),
                gp="yes" if bool(row.get("guardrail_pass")) else "no",
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown decision report from final sweep JSON.")
    parser.add_argument("--ranked-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    ranked = _load(Path(args.ranked_json))
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    winner = ranked.get("winner") or {}
    intersection_rows = ranked.get("views", {}).get("intersection", {}).get("ranked_lanes", [])
    full_rows = ranked.get("views", {}).get("full", {}).get("ranked_lanes", [])
    assumptions = ranked.get("assumptions") or {}

    lines: List[str] = []
    lines.append("# Final Calibration Decision Report")
    lines.append("")
    lines.append("## Objective")
    lines.append(
        "Select the long-term default prepass+calibration lane by maximizing post-XGB F1 at IoU 0.5 "
        "with guardrails against detector-union regression and coverage collapse."
    )
    lines.append("")
    lines.append("## Winner")
    if winner:
        lines.append(f"- Lane: `{winner.get('lane')}`")
        lines.append(f"- View used for selection: `{winner.get('selection_view')}`")
        lines.append(f"- Mean F1: `{_safe_float(winner.get('mean_f1')):.4f}`")
        lines.append(
            f"- Mean Δ vs union F1: `{_safe_float(winner.get('mean_delta_vs_union_f1')):+.4f}`"
        )
        lines.append(
            f"- Mean coverage preservation: `{_safe_float(winner.get('mean_coverage_preservation')):.4f}`"
        )
        lines.append(f"- Guardrail pass: `{'yes' if bool(winner.get('guardrail_pass')) else 'no'}`")
    else:
        lines.append("- No winner selected (all lanes failed guardrails or no results).")
    lines.append("")
    lines.append("## Intersection View Ranking")
    if intersection_rows:
        lines.extend(_lane_table(intersection_rows))
    else:
        lines.append("_No intersection-view results found._")
    lines.append("")
    lines.append("## Full View Ranking")
    if full_rows:
        lines.extend(_lane_table(full_rows))
    else:
        lines.append("_No full-view results found._")
    lines.append("")
    lines.append("## Assumptions/Policy")
    lines.append(f"- Selection objective: `{assumptions.get('selection_objective', 'f1')}`")
    lines.append(f"- Guardrail delta threshold: `{_safe_float(assumptions.get('guardrail_delta_min', 0.02)):.4f}`")
    lines.append(f"- Coverage tolerance: `{_safe_float(assumptions.get('coverage_tolerance', 0.02)):.4f}`")
    lines.append(f"- Eval IoU: `{_safe_float(assumptions.get('eval_iou', 0.5)):.2f}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Ranked JSON: `{Path(args.ranked_json).resolve()}`")
    lines.append("")

    output_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_md": str(output_md.resolve())}, indent=2))


if __name__ == "__main__":
    main()

