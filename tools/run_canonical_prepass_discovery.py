#!/usr/bin/env python3
"""Run the authoritative canonical EDR discovery pipeline and write the promoted EDR."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRESS_FILENAME = "canonical_discovery_progress.json"
CANONICAL_EDR_JSON_NAME = "canonical_edr.json"
CANONICAL_EDR_MD_NAME = "canonical_edr.md"
LEGACY_CANONICAL_RECIPE_JSON_NAME = "canonical_prepass_recipe.json"
LEGACY_CANONICAL_RECIPE_MD_NAME = "canonical_prepass_recipe.md"
DISCOVERY_STAGE_REUSE_VERSIONS: Dict[str, int] = {
    "main_sweep": 1,
    "alpha_extension": 1,
    "sam_bias_scope": 1,
    "sam_bias_magnitude": 1,
    "similarity_quality": 1,
    "nonwindow_confirmation": 1,
}


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_progress(
    run_root: Path,
    *,
    stage_key: str,
    stage_label: str,
    stage_index: int,
    stage_total: int,
    status: str,
    message: str,
) -> None:
    _write_json(
        run_root / PROGRESS_FILENAME,
        {
            "updated_utc": _ts(),
            "stage_key": str(stage_key),
            "stage_label": str(stage_label),
            "stage_index": int(stage_index),
            "stage_total": int(stage_total),
            "status": str(status),
            "message": str(message),
        },
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _stage_reuse_meta_path(output_path: Path) -> Path:
    return output_path.parent / f".{output_path.name}.reuse_meta.json"


def _stage_reuse_signature(stage_key: str, cmd: List[str]) -> Dict[str, Any]:
    return {
        "stage_key": str(stage_key),
        "stage_version": int(DISCOVERY_STAGE_REUSE_VERSIONS.get(stage_key, 1)),
        "command": [str(part) for part in cmd],
    }


def _is_stage_output_reusable(output_path: Path, stage_key: str, cmd: List[str]) -> bool:
    if not output_path.exists():
        return False
    meta_path = _stage_reuse_meta_path(output_path)
    if not meta_path.exists():
        return False
    try:
        payload = _load_json(meta_path)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return payload == _stage_reuse_signature(stage_key, cmd)


def _write_stage_reuse_meta(output_path: Path, stage_key: str, cmd: List[str]) -> None:
    _write_json(_stage_reuse_meta_path(output_path), _stage_reuse_signature(stage_key, cmd))


def _progress_stage_plan(
    lane_selection: str,
    *,
    discovered_winner_lane: str = "",
) -> List[Dict[str, str]]:
    stages: List[Dict[str, str]] = [
        {"key": "main_sweep", "label": "Main sweep"},
    ]
    if lane_selection == "compare_both" and not discovered_winner_lane:
        return stages

    run_window_refinements = lane_selection == "window" or (
        lane_selection == "compare_both" and discovered_winner_lane == "window"
    )
    run_nonwindow_confirmation = lane_selection == "nonwindow" or lane_selection == "compare_both"

    if run_window_refinements:
        # Learned second-stage trust analysis remains available in research-only
        # scripts, but it is intentionally excluded from the default promotion path.
        stages.extend(
            [
                {"key": "alpha_extension", "label": "Alpha extension"},
                {"key": "sam_bias_scope", "label": "SAM-bias scope ablation"},
                {"key": "sam_bias_magnitude", "label": "SAM-bias magnitude sweep"},
                {"key": "similarity_quality", "label": "Similarity-quality full-window eval"},
            ]
        )
    if run_nonwindow_confirmation:
        stages.append({"key": "nonwindow_confirmation", "label": "Non-window confirmation"})
    stages.append({"key": "write_canonical_recipe", "label": "Write canonical EDR"})
    return stages


def _run_progress_stage(
    *,
    run_root: Path,
    stage_plan: List[Dict[str, str]],
    stage_key: str,
    output_path: Path,
    cmd: List[str],
    force: bool,
) -> None:
    stage_index = next(
        idx for idx, stage in enumerate(stage_plan, start=1) if str(stage.get("key") or "") == stage_key
    )
    stage_label = next(
        str(stage.get("label") or stage_key) for stage in stage_plan if str(stage.get("key") or "") == stage_key
    )
    if _is_stage_output_reusable(output_path, stage_key, cmd) and not force:
        _write_progress(
            run_root,
            stage_key=stage_key,
            stage_label=stage_label,
            stage_index=stage_index,
            stage_total=len(stage_plan),
            status="reused",
            message=f"{stage_label}: using existing outputs.",
        )
        return
    _write_progress(
        run_root,
        stage_key=stage_key,
        stage_label=stage_label,
        stage_index=stage_index,
        stage_total=len(stage_plan),
        status="running",
        message=f"{stage_label}: running.",
    )
    _run(cmd)
    _write_stage_reuse_meta(output_path, stage_key, cmd)
    _write_progress(
        run_root,
        stage_key=stage_key,
        stage_label=stage_label,
        stage_index=stage_index,
        stage_total=len(stage_plan),
        status="completed",
        message=f"{stage_label}: completed.",
    )


def _resolve_refined_tag(run_root: Path) -> str:
    ranked = _load_json(run_root / "postrun_sam_bias_magnitude_sweep" / "results_ranked.json")
    full_rows = ranked.get("full") if isinstance(ranked.get("full"), list) else []
    pilot_rows = ranked.get("pilot") if isinstance(ranked.get("pilot"), list) else []
    rows = full_rows or pilot_rows
    if not rows:
        raise RuntimeError("sam_bias_magnitude_winner_missing")
    tag = str(rows[0].get("tag") or "").strip()
    if not tag:
        raise RuntimeError("sam_bias_magnitude_tag_missing")
    return tag


def _select_nonwindow_lane(ranked_payload: Dict[str, Any]) -> str:
    rows = ranked_payload.get("views", {}).get("intersection", {}).get("ranked_lanes", [])
    for row in rows:
        lane = str(row.get("lane") or "")
        if lane == "nonwindow":
            return lane
    raise RuntimeError("nonwindow_lane_missing_from_ranked_results")


def _load_main_sweep(run_root: Path) -> Dict[str, Any]:
    ranked = _load_json(run_root / "results_ranked.json")
    raw = _load_json(run_root / "results_raw.json")
    final_default = _load_json(run_root / "final_default_recipe.json")
    return {
        "ranked": ranked,
        "raw": raw,
        "final_default": final_default,
    }


def _best_stack_for_lane(main_sweep: Dict[str, Any], lane: str) -> Dict[str, Any]:
    best_stack = main_sweep["raw"].get("search_results", {}).get("best_stack", {})
    payload = best_stack.get(lane)
    if not isinstance(payload, dict):
        raise RuntimeError(f"best_stack_missing_for_lane:{lane}")
    return copy.deepcopy(payload)


def _set_source_bias_defaults(policy: Dict[str, Any], *, text_bias: float, sim_bias: float) -> Dict[str, Any]:
    updated = copy.deepcopy(policy)
    mapping = updated.get("logit_bias_by_source_class") if isinstance(updated.get("logit_bias_by_source_class"), dict) else {}
    text_map = mapping.get("sam3_text") if isinstance(mapping.get("sam3_text"), dict) else {}
    sim_map = mapping.get("sam3_similarity") if isinstance(mapping.get("sam3_similarity"), dict) else {}
    text_map["__default__"] = float(text_bias)
    sim_map["__default__"] = float(sim_bias)
    mapping["sam3_text"] = text_map
    mapping["sam3_similarity"] = sim_map
    updated["logit_bias_by_source_class"] = mapping
    return updated


def _window_expected_metrics(
    *,
    similarity_decision: Dict[str, Any],
    magnitude_decision: Dict[str, Any],
) -> Dict[str, Any]:
    if str(similarity_decision.get("status") or "") == "promoted":
        winner = similarity_decision.get("winner_metrics") if isinstance(similarity_decision.get("winner_metrics"), dict) else {}
        return {
            "full_mean_f1": _safe_float(winner.get("mean_f1"), 0.0),
            "full_mean_delta_f1_vs_refined_hand": _safe_float(winner.get("mean_delta_f1"), 0.0),
        }
    full_winner = magnitude_decision.get("full_winner") if isinstance(magnitude_decision.get("full_winner"), dict) else {}
    return {
        "full_mean_f1": _safe_float(full_winner.get("mean_f1"), 0.0),
        "full_mean_delta_f1_vs_refined_hand": _safe_float(full_winner.get("mean_delta_vs_baseline_f1"), 0.0),
    }


def _compose_windowed_recipe(
    *,
    base_stack: Dict[str, Any],
    alpha_decision: Dict[str, Any],
    scope_decision: Dict[str, Any],
    magnitude_decision: Dict[str, Any],
    similarity_decision: Dict[str, Any],
    decision_inputs: Dict[str, str],
) -> Dict[str, Any]:
    scenario = copy.deepcopy(base_stack.get("scenario") or {})
    policy = copy.deepcopy(base_stack.get("policy") or {})

    alpha_cfg = alpha_decision.get("promoted_config") if isinstance(alpha_decision.get("promoted_config"), dict) else {}
    scope_cfg = scope_decision.get("promoted_config") if isinstance(scope_decision.get("promoted_config"), dict) else {}
    mag_cfg = magnitude_decision.get("promoted_config") if isinstance(magnitude_decision.get("promoted_config"), dict) else {}
    similarity_cfg = similarity_decision.get("promoted_config") if isinstance(similarity_decision.get("promoted_config"), dict) else {}

    if scope_cfg.get("sam_bias_scope"):
        policy["sam_bias_scope"] = str(scope_cfg["sam_bias_scope"])
    if magnitude_decision.get("status") == "promoted" and mag_cfg:
        policy = _set_source_bias_defaults(
            policy,
            text_bias=_safe_float(mag_cfg.get("sam3_text_bias_default"), -1.4),
            sim_bias=_safe_float(mag_cfg.get("sam3_similarity_bias_default"), -1.2),
        )

    recipe = {
        "validation_status": "promoted_via_authoritative_discovery_pipeline",
        "winner_lane": "window",
        "xgb_hparams": copy.deepcopy(base_stack.get("hp") or {}),
        "scenario": {
            "split_head": bool(scenario.get("split_head")),
            "train_sam3_text_quality": bool(scenario.get("sam_quality", True)),
            "sam3_text_quality_alpha": _safe_float(alpha_cfg.get("sam3_text_quality_alpha"), _safe_float(scenario.get("alpha"), 0.5)),
            "train_sam3_similarity_quality": bool(similarity_cfg.get("train_sam3_similarity_quality", False)),
            "sam3_similarity_quality_alpha": (
                _safe_float(similarity_cfg.get("sam3_similarity_quality_alpha"), 0.0)
                if bool(similarity_cfg.get("train_sam3_similarity_quality", False))
                else None
            ),
        },
        "policy": policy,
        "expected_metrics": _window_expected_metrics(
            similarity_decision=similarity_decision,
            magnitude_decision=magnitude_decision,
        ),
        "source_decisions": {
            "main_sweep": decision_inputs["main_sweep"],
            "alpha_extension": decision_inputs["alpha_extension"],
            "sam_bias_scope": decision_inputs["sam_bias_scope"],
            "sam_bias_magnitude": decision_inputs["sam_bias_magnitude"],
            "similarity_quality": decision_inputs["similarity_quality"],
        },
    }
    return recipe


def _compose_nonwindow_recipe(
    *,
    base_stack: Dict[str, Any],
    nonwindow_decision: Dict[str, Any],
    decision_inputs: Dict[str, str],
) -> Dict[str, Any]:
    recipe = copy.deepcopy(nonwindow_decision.get("canonical_recipe") or {})
    if not recipe:
        scenario = copy.deepcopy(base_stack.get("scenario") or {})
        recipe = {
            "validation_status": "fallback_to_main_sweep_nonwindow_policy",
            "winner_lane": nonwindow_decision.get("nonwindow_lane") or "nonwindow",
            "xgb_hparams": copy.deepcopy(base_stack.get("hp") or {}),
            "scenario": {
                "split_head": bool(scenario.get("split_head")),
                "train_sam3_text_quality": bool(scenario.get("sam_quality", True)),
                "sam3_text_quality_alpha": _safe_float(scenario.get("alpha"), 0.5),
                "train_sam3_similarity_quality": False,
                "sam3_similarity_quality_alpha": None,
            },
            "policy": copy.deepcopy(base_stack.get("policy") or {}),
            "expected_metrics": {},
        }
    else:
        # Older research-only outputs may still carry the learned second-stage
        # block. The default promoted EDR path no longer publishes it.
        recipe.pop("second_stage_policy_layer", None)
    recipe["source_decisions"] = {
        "main_sweep": decision_inputs["main_sweep"],
        "nonwindow_confirmation": decision_inputs["nonwindow_confirmation"],
    }
    return recipe


def _render_md(payload: Dict[str, Any]) -> str:
    windowed = payload.get("canonical_windowed_recipe") if isinstance(payload.get("canonical_windowed_recipe"), dict) else {}
    nonwindowed = payload.get("canonical_nonwindowed_recipe") if isinstance(payload.get("canonical_nonwindowed_recipe"), dict) else {}
    lines = [
        "# Canonical EDR Discovery",
        "",
        f"- Generated UTC: `{payload.get('generated_utc')}`",
        f"- Run root: `{payload.get('run_root')}`",
        f"- Similarity-quality status: `{payload.get('promotion_status', {}).get('windowed_similarity_quality')}`",
        f"- Non-window refined-policy status: `{payload.get('promotion_status', {}).get('nonwindow_refined_policy')}`",
        "",
        "## Windowed Canonical Recipe",
        "",
        f"- Lane: `{windowed.get('winner_lane')}`",
        f"- SAM3 text quality alpha: `{(windowed.get('scenario') or {}).get('sam3_text_quality_alpha')}`",
        f"- SAM3 similarity quality alpha: `{(windowed.get('scenario') or {}).get('sam3_similarity_quality_alpha')}`",
        "",
        "## Non-Windowed Fallback",
        "",
        f"- Lane: `{nonwindowed.get('winner_lane')}`",
        f"- SAM3 text quality alpha: `{(nonwindowed.get('scenario') or {}).get('sam3_text_quality_alpha')}`",
        f"- SAM3 similarity quality alpha: `{(nonwindowed.get('scenario') or {}).get('sam3_similarity_quality_alpha')}`",
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset", default="qwen_dataset")
    parser.add_argument("--seeds", default="42,1337,2025")
    parser.add_argument(
        "--lane-selection",
        default="window",
        choices=["window", "nonwindow", "compare_both"],
    )
    parser.add_argument("--window-key", default="ceab65b2bff24d316ca5f858addaffed8abfdb11")
    parser.add_argument("--nonwindow-key", default="20c8d44d69f51b2ffe528fb500e75672a306f67d")
    parser.add_argument("--classifier-id", default="uploads/classifiers/DinoV3_best_model_large.pkl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window-prepass-jsonl", default="")
    parser.add_argument("--nonwindow-prepass-jsonl", default="")
    parser.add_argument("--window-features", default="")
    parser.add_argument("--window-labeled", default="")
    parser.add_argument("--nonwindow-features", default="")
    parser.add_argument("--nonwindow-labeled", default="")
    args = parser.parse_args()

    run_root = (REPO_ROOT / args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    stage_plan = _progress_stage_plan(str(args.lane_selection))

    _run_progress_stage(
        run_root=run_root,
        stage_plan=stage_plan,
        stage_key="main_sweep",
        output_path=run_root / "final_default_recipe.json",
        cmd=[
            sys.executable,
            "tools/run_final_calibration_sweep.py",
            "--dataset",
            str(args.dataset),
            "--run-root",
            str(run_root),
            "--window-key",
            str(args.window_key),
            "--nonwindow-key",
            str(args.nonwindow_key),
            "--classifier-id",
            str(args.classifier_id),
            "--seeds",
            str(args.seeds),
            "--lane-selection",
            str(args.lane_selection),
        ]
        + (
            ["--window-prepass-jsonl", str(args.window_prepass_jsonl)]
            if str(args.window_prepass_jsonl).strip()
            else []
        )
        + (
            ["--nonwindow-prepass-jsonl", str(args.nonwindow_prepass_jsonl)]
            if str(args.nonwindow_prepass_jsonl).strip()
            else []
        )
        + (
            ["--window-features", str(args.window_features)]
            if str(args.window_features).strip()
            else []
        )
        + (
            ["--window-labeled", str(args.window_labeled)]
            if str(args.window_labeled).strip()
            else []
        )
        + (
            ["--nonwindow-features", str(args.nonwindow_features)]
            if str(args.nonwindow_features).strip()
            else []
        )
        + (
            ["--nonwindow-labeled", str(args.nonwindow_labeled)]
            if str(args.nonwindow_labeled).strip()
            else []
        ),
        force=bool(args.force),
    )
    main_sweep = _load_main_sweep(run_root)
    discovered_winner_lane = str(main_sweep["final_default"].get("winner_lane") or "").strip() or str(args.lane_selection)
    stage_plan = _progress_stage_plan(
        str(args.lane_selection),
        discovered_winner_lane=discovered_winner_lane,
    )

    alpha_decision: Dict[str, Any] = {}
    scope_decision: Dict[str, Any] = {}
    magnitude_decision: Dict[str, Any] = {}
    similarity_decision: Dict[str, Any] = {}
    nonwindow_decision: Dict[str, Any] = {}

    run_window_refinements = args.lane_selection == "window" or (
        args.lane_selection == "compare_both" and discovered_winner_lane == "window"
    )
    run_nonwindow_confirmation = args.lane_selection in {"nonwindow", "compare_both"}

    if run_window_refinements:
        _run_progress_stage(
            run_root=run_root,
            stage_plan=stage_plan,
            stage_key="alpha_extension",
            output_path=run_root / "postrun_alpha_extension" / "decision_summary.json",
            cmd=[sys.executable, "tools/run_postrun_alpha_extension.py", "--run-root", str(run_root)],
            force=bool(args.force),
        )
        _run_progress_stage(
            run_root=run_root,
            stage_plan=stage_plan,
            stage_key="sam_bias_scope",
            output_path=run_root / "postrun_sam_bias_scope_ablation" / "decision_summary.json",
            cmd=[sys.executable, "tools/run_postrun_sam_bias_scope_ablation.py", "--run-root", str(run_root)],
            force=bool(args.force),
        )
        _run_progress_stage(
            run_root=run_root,
            stage_plan=stage_plan,
            stage_key="sam_bias_magnitude",
            output_path=run_root / "postrun_sam_bias_magnitude_sweep" / "decision_summary.json",
            cmd=[sys.executable, "tools/run_postrun_sam_bias_magnitude_sweep.py", "--run-root", str(run_root)],
            force=bool(args.force),
        )

        refined_tag = _resolve_refined_tag(run_root)
        _run_progress_stage(
            run_root=run_root,
            stage_plan=stage_plan,
            stage_key="similarity_quality",
            output_path=run_root / "postrun_similarity_quality_full_window_eval" / "decision_summary.json",
            cmd=[
                sys.executable,
                "tools/run_postrun_similarity_quality_full_window_eval.py",
                "--run-root",
                str(run_root),
                "--refined-tag",
                refined_tag,
                "--seeds",
                str(args.seeds),
            ],
            force=bool(args.force),
        )
        alpha_decision = _load_json(run_root / "postrun_alpha_extension" / "decision_summary.json")
        scope_decision = _load_json(run_root / "postrun_sam_bias_scope_ablation" / "decision_summary.json")
        magnitude_decision = _load_json(run_root / "postrun_sam_bias_magnitude_sweep" / "decision_summary.json")
        similarity_decision = _load_json(run_root / "postrun_similarity_quality_full_window_eval" / "decision_summary.json")

    if run_nonwindow_confirmation:
        _run_progress_stage(
            run_root=run_root,
            stage_plan=stage_plan,
            stage_key="nonwindow_confirmation",
            output_path=run_root / "postrun_nonwindow_policy_confirmation" / "decision_summary.json",
            cmd=[sys.executable, "tools/run_postrun_nonwindow_policy_confirmation.py", "--run-root", str(run_root)],
            force=bool(args.force),
        )
        nonwindow_decision = _load_json(run_root / "postrun_nonwindow_policy_confirmation" / "decision_summary.json")

    decision_inputs = {
        "main_sweep": str((run_root / "results_ranked.json").resolve()),
        "alpha_extension": str((run_root / "postrun_alpha_extension" / "decision_summary.json").resolve())
        if run_window_refinements
        else "",
        "sam_bias_scope": str((run_root / "postrun_sam_bias_scope_ablation" / "decision_summary.json").resolve())
        if run_window_refinements
        else "",
        "sam_bias_magnitude": str((run_root / "postrun_sam_bias_magnitude_sweep" / "decision_summary.json").resolve())
        if run_window_refinements
        else "",
        "similarity_quality": str((run_root / "postrun_similarity_quality_full_window_eval" / "decision_summary.json").resolve())
        if run_window_refinements
        else "",
        "nonwindow_confirmation": str((run_root / "postrun_nonwindow_policy_confirmation" / "decision_summary.json").resolve())
        if run_nonwindow_confirmation
        else "",
    }

    windowed_lane = discovered_winner_lane if discovered_winner_lane == "window" else "window"
    nonwindow_lane = (
        _select_nonwindow_lane(main_sweep["ranked"])
        if run_nonwindow_confirmation
        else "nonwindow"
    )

    payload = {
        "generated_utc": _ts(),
        "run_root": str(run_root),
        "dataset": str(args.dataset),
        "lane_selection": str(args.lane_selection),
        "discovered_winner_lane": discovered_winner_lane,
        "decision_inputs": decision_inputs,
        "canonical_windowed_recipe": (
            _compose_windowed_recipe(
                base_stack=_best_stack_for_lane(main_sweep, windowed_lane),
                alpha_decision=alpha_decision,
                scope_decision=scope_decision,
                magnitude_decision=magnitude_decision,
                similarity_decision=similarity_decision,
                decision_inputs=decision_inputs,
            )
            if run_window_refinements
            else {}
        ),
        "canonical_nonwindowed_recipe": (
            _compose_nonwindow_recipe(
                base_stack=_best_stack_for_lane(main_sweep, nonwindow_lane),
                nonwindow_decision=nonwindow_decision,
                decision_inputs=decision_inputs,
            )
            if run_nonwindow_confirmation
            else {}
        ),
        "promotion_status": {
            "windowed_similarity_quality": str(similarity_decision.get("status") or "rejected") if run_window_refinements else "not_run",
            "nonwindow_refined_policy": str(nonwindow_decision.get("refined_policy_status") or "rejected") if run_nonwindow_confirmation else "not_run",
            "nonwindow_similarity_quality": str(nonwindow_decision.get("similarity_quality_status") or "rejected") if run_nonwindow_confirmation else "not_run",
        },
    }

    _write_progress(
        run_root,
        stage_key="write_canonical_recipe",
        stage_label="Write canonical EDR",
        stage_index=next(
            idx for idx, stage in enumerate(stage_plan, start=1) if str(stage.get("key") or "") == "write_canonical_recipe"
        ),
        stage_total=len(stage_plan),
        status="running",
        message="Write canonical EDR: running.",
    )
    canonical_edr_json = run_root / CANONICAL_EDR_JSON_NAME
    canonical_edr_md = run_root / CANONICAL_EDR_MD_NAME
    _write_json(canonical_edr_json, payload)
    _write_text(canonical_edr_md, _render_md(payload))
    # Keep the legacy canonical_prepass_recipe.* aliases so existing registry
    # entries, tests, and in-flight runs continue to work during the rename.
    _write_json(run_root / LEGACY_CANONICAL_RECIPE_JSON_NAME, payload)
    _write_text(run_root / LEGACY_CANONICAL_RECIPE_MD_NAME, _render_md(payload))
    _write_progress(
        run_root,
        stage_key="write_canonical_recipe",
        stage_label="Write canonical EDR",
        stage_index=next(
            idx for idx, stage in enumerate(stage_plan, start=1) if str(stage.get("key") or "") == "write_canonical_recipe"
        ),
        stage_total=len(stage_plan),
        status="completed",
        message="Write canonical EDR: completed.",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "run_root": str(run_root),
                "canonical_recipe": str(canonical_edr_json.resolve()),
                "legacy_canonical_recipe": str((run_root / LEGACY_CANONICAL_RECIPE_JSON_NAME).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
