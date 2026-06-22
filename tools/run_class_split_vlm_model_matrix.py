#!/usr/bin/env python3
"""Run a fixed Class Split vignette set across multiple reviewer VLMs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_class_split_qwen_review_benchmark.py"

MODEL_PRESETS = {
    "baseline": [
        "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit",
    ],
    "smoke-mlx": [
        "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
    ],
    "qwen36-mlx": [
        "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
    ],
    "empero": [
        "empero-ai/Qwable-9B-Claude-Fable-5",
        "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit",
    ],
    "all-mlx": [
        "vanch007/Huihui-Qwen3.6-35B-A3B-abliterated-mlx-4bit",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
    ],
}


def _slug(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = lowered.replace("/", "_")
    lowered = re.sub(r"[^a-z0-9_.-]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_.-")
    return lowered[:90] or "model"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _summarize_run(path: Path, *, model_id: str, run_id: str, returncode: int, elapsed: float) -> Dict[str, Any]:
    payload = _load_json(path)
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    decisions = Counter(str(record.get("decision") or "missing") for record in records if isinstance(record, dict))
    statuses = Counter(str(record.get("status") or "missing") for record in records if isinstance(record, dict))
    dispositions = Counter()
    signals = Counter()
    scratchpad_status = Counter()
    guarded = 0
    malformed = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        scratchpad_status[str(record.get("thinking_scratchpad_status") or "missing")] += 1
        disposition = record.get("review_disposition") if isinstance(record.get("review_disposition"), dict) else {}
        if disposition:
            dispositions[str(disposition.get("disposition") or "missing")] += 1
            signals[str(disposition.get("signal") or "missing")] += 1
        guarded_record = record.get("guarded_recommendation")
        if isinstance(guarded_record, dict) and guarded_record.get("blocked"):
            guarded += 1
        if record.get("schema_repair_attempted") or record.get("final_schema_repair_attempted"):
            malformed += 1
    completed = statuses.get("completed", 0)
    non_skip = sum(decisions.get(name, 0) for name in ("accept_suggested", "confirm_current", "change_to_other"))
    return {
        "model_id": model_id,
        "run_id": run_id,
        "path": str(path),
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 3),
        "records": len(records),
        "completed": completed,
        "completion_rate": round(completed / len(records), 4) if records else 0.0,
        "decisions": dict(decisions),
        "statuses": dict(statuses),
        "non_skip": non_skip,
        "non_skip_rate": round(non_skip / len(records), 4) if records else 0.0,
        "guarded_recommendations": guarded,
        "schema_repair_records": malformed,
        "thinking_scratchpad_statuses": dict(scratchpad_status),
        "dispositions": dict(dispositions),
        "signals": dict(signals),
    }


def _resolve_models(args: argparse.Namespace) -> List[str]:
    models: List[str] = []
    for preset in args.preset or []:
        models.extend(MODEL_PRESETS[preset])
    models.extend(args.model or [])
    seen = set()
    resolved = []
    for model in models:
        clean = str(model or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            resolved.append(clean)
    if not resolved:
        resolved = list(MODEL_PRESETS["smoke-mlx"])
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--preset", action="append", choices=sorted(MODEL_PRESETS), default=[])
    parser.add_argument("--model", action="append", default=[], help="Additional model id; may be repeated.")
    parser.add_argument("--run-prefix", default="vlm_model_matrix")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--visual-limit", type=int, default=12)
    parser.add_argument("--review-timeout-seconds", type=int, default=900)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--enable-local-consensus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-class-concept-briefs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--thinking-effort", type=float, default=None)
    parser.add_argument("--thinking-scale-factor", type=float, default=None)
    parser.add_argument("--per-review-subprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    source_run = Path(args.source_run).expanduser()
    if not source_run.is_file():
        raise SystemExit(f"Missing source run: {source_run}")
    source_run = source_run.resolve()
    root = ROOT / "uploads" / "class_analysis" / str(args.job_id) / "qwen_reviews"
    root.mkdir(parents=True, exist_ok=True)

    matrix_id = f"{_slug(args.run_prefix)}_{int(args.count)}_{int(time.time())}"
    matrix_dir = root / matrix_id
    matrix_dir.mkdir(parents=True, exist_ok=True)
    models = _resolve_models(args)
    summaries: List[Dict[str, Any]] = []
    failures = 0

    for index, model_id in enumerate(models, start=1):
        run_id = f"{matrix_id}_{index:02d}_{_slug(model_id)}"
        cmd = [
            str(args.python),
            str(RUNNER),
            "--job-id",
            str(args.job_id),
            "--source-run",
            str(source_run),
            "--count",
            str(args.count),
            "--start",
            str(args.start),
            "--run-id",
            run_id,
            "--model-id",
            model_id,
            "--max-turns",
            str(args.max_turns),
            "--visual-limit",
            str(args.visual_limit),
            "--review-timeout-seconds",
            str(args.review_timeout_seconds),
        ]
        if args.enable_local_consensus:
            cmd.append("--enable-local-consensus")
        if args.enable_class_concept_briefs:
            cmd.append("--enable-class-concept-briefs")
        if args.enable_thinking:
            cmd.append("--enable-thinking")
        if args.thinking_effort is not None:
            cmd.extend(["--thinking-effort", str(args.thinking_effort)])
        if args.thinking_scale_factor is not None:
            cmd.extend(["--thinking-scale-factor", str(args.thinking_scale_factor)])
        if args.per_review_subprocess:
            cmd.append("--per-review-subprocess")
        if args.audit:
            cmd.append("--audit")

        log_path = matrix_dir / f"{run_id}.log"
        started = time.time()
        print(f"[{index}/{len(models)}] {model_id}")
        print("  " + " ".join(cmd))
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed = time.time() - started
        output_path = root / f"{run_id}.json"
        summary = _summarize_run(
            output_path,
            model_id=model_id,
            run_id=run_id,
            returncode=completed.returncode,
            elapsed=elapsed,
        )
        summary["log_path"] = str(log_path)
        summaries.append(summary)
        print(
            f"  return={completed.returncode} records={summary['records']} "
            f"completed={summary['completed']} non_skip={summary['non_skip']} "
            f"elapsed={summary['elapsed_seconds']}s"
        )
        if completed.returncode != 0:
            failures += 1
            if args.fail_fast:
                break

    payload = {
        "matrix_id": matrix_id,
        "job_id": args.job_id,
        "source_run": str(source_run),
        "count": int(args.count),
        "start": int(args.start),
        "models": models,
        "enable_thinking": bool(args.enable_thinking),
        "thinking_effort": args.thinking_effort,
        "thinking_scale_factor": args.thinking_scale_factor,
        "summaries": summaries,
    }
    summary_path = matrix_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
