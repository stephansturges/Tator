#!/usr/bin/env python3
"""Sweep Falcon segmentation-box settings on the fixed qwen debug corpus."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.falcon_perception import unload_official_falcon_runtime  # noqa: E402
from tools.debug_falcon_direct_qwen5 import (  # noqa: E402
    REPORT_BASE,
    _sample_images,
    _to_jsonable,
)


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _split_ints(value: str) -> List[int]:
    return [int(item) for item in _split_csv(value)]


def _split_floats(value: str) -> List[float]:
    return [float(item) for item in _split_csv(value)]


def _config_id(config: Dict[str, Any]) -> str:
    return (
        f"{config['backend']}"
        f"__{config['component_mode']}"
        f"__dim{config['max_dimension']}"
        f"__hr{config['hr_upsample_ratio']}"
        f"__cd{str(config['coord_dedup_threshold']).replace('.', 'p')}"
        f"__st{str(config['segmentation_threshold']).replace('.', 'p')}"
    )


def _iter_configs(
    *,
    backends: Sequence[str],
    component_modes: Sequence[str],
    max_dimensions: Sequence[int],
    hr_upsample_ratios: Sequence[int],
    coord_dedup_thresholds: Sequence[float],
    segmentation_thresholds: Sequence[float],
) -> Iterable[Dict[str, Any]]:
    for backend in backends:
        for component_mode in component_modes:
            for max_dimension in max_dimensions:
                for hr_upsample_ratio in hr_upsample_ratios:
                    for coord_dedup_threshold in coord_dedup_thresholds:
                        for segmentation_threshold in segmentation_thresholds:
                            yield {
                                "backend": str(backend),
                                "component_mode": str(component_mode),
                                "max_dimension": int(max_dimension),
                                "hr_upsample_ratio": int(hr_upsample_ratio),
                                "coord_dedup_threshold": float(coord_dedup_threshold),
                                "segmentation_threshold": float(segmentation_threshold),
                            }


def _mean_best_iou(results: Sequence[Dict[str, Any]]) -> float:
    gt_total = sum(int(item.get("gt_count") or 0) for item in results)
    if gt_total <= 0:
        return 1.0
    weighted = 0.0
    for item in results:
        gt_count = int(item.get("gt_count") or 0)
        weighted += float(item.get("dedup_coverage", {}).get("mean_best_iou") or 0.0) * float(gt_count)
    return float(weighted) / float(gt_total)


def _server_is_healthy(server_url: str) -> tuple[bool, str]:
    base = str(server_url or "").strip().rstrip("/")
    if not base:
        return False, "falcon_server_url_required"
    try:
        response = requests.get(f"{base}/v1/health", timeout=10)
        response.raise_for_status()
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"
    try:
        payload = response.json()
    except Exception:
        payload = {}
    supported = payload.get("supported_tasks") or []
    if supported and "segmentation" not in supported:
        return False, f"segmentation_not_supported:{supported}"
    return True, "ok"


def _run_config(
    *,
    config: Dict[str, Any],
    images: Sequence[str],
    sample_seed: int,
    report_root: Path,
    falcon_model_id: str,
    falcon_device: str,
    iou_threshold: float,
    falcon_min_dimension: int,
    falcon_server_url: str,
) -> Dict[str, Any]:
    config_started = time.perf_counter()
    skip_reason = ""
    status = "completed"
    per_image: List[Dict[str, Any]] = []

    if str(config["backend"]) == "server":
        healthy, health_note = _server_is_healthy(falcon_server_url)
        if not healthy:
            status = "skipped"
            skip_reason = str(health_note)
            return {
                "config": dict(config),
                "config_id": _config_id(config),
                "status": status,
                "skip_reason": skip_reason,
                "results": [],
                "overall": {},
            }

    config_id = _config_id(config)
    subprocess_report_root = report_root / "config_reports" / config_id
    if subprocess_report_root.exists():
        for child in subprocess_report_root.iterdir():
            if child.is_file():
                child.unlink()
    subprocess_report_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(REPO_ROOT / "tools" / "debug_falcon_direct_qwen5.py"),
        "--sample-size",
        str(len(images)),
        "--sample-seed",
        str(sample_seed),
        "--falcon-model-id",
        str(falcon_model_id),
        "--falcon-device",
        str(falcon_device),
        "--falcon-backend",
        str(config["backend"]),
        "--falcon-detection-strategy",
        "segmentation_boxes",
        "--falcon-component-mode",
        str(config["component_mode"]),
        "--falcon-min-dimension",
        str(int(falcon_min_dimension)),
        "--falcon-max-dimension",
        str(int(config["max_dimension"])),
        "--falcon-coord-dedup-threshold",
        str(float(config["coord_dedup_threshold"])),
        "--falcon-hr-upsample-ratio",
        str(int(config["hr_upsample_ratio"])),
        "--falcon-segmentation-threshold",
        str(float(config["segmentation_threshold"])),
        "--iou-threshold",
        str(float(iou_threshold)),
        "--report-root",
        str(subprocess_report_root),
    ]
    if str(falcon_server_url or "").strip():
        cmd.extend(["--falcon-server-url", str(falcon_server_url).strip()])

    run_env = dict(os.environ)
    completed_proc = None
    try:
        completed_proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=run_env,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        status = "failed"
        skip_reason = f"{type(exc).__name__}:{exc}"
    finally:
        unload_official_falcon_runtime()

    report_path = subprocess_report_root / "report.json"
    if report_path.exists():
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            per_image = list(report_payload.get("results") or [])
        except Exception as exc:
            status = "failed"
            skip_reason = f"report_parse_error:{type(exc).__name__}:{exc}"
    elif status == "completed":
        status = "failed"
        stderr_snippet = ""
        if completed_proc is not None:
            stderr_snippet = str((completed_proc.stderr or completed_proc.stdout or "").strip())[-1000:]
        skip_reason = stderr_snippet or "missing_report_json"
    if completed_proc is not None and completed_proc.returncode != 0 and status == "completed":
        status = "failed"
        stderr_snippet = str((completed_proc.stderr or completed_proc.stdout or "").strip())[-1000:]
        skip_reason = stderr_snippet or f"subprocess_exit_{completed_proc.returncode}"

    gt_total = sum(int(item.get("gt_count") or 0) for item in per_image)
    raw_hits = sum(int(item.get("raw_coverage", {}).get("matched") or 0) for item in per_image)
    dedup_hits = sum(int(item.get("dedup_coverage", {}).get("matched") or 0) for item in per_image)
    avg_elapsed_sec = (
        float(sum(float(item.get("elapsed_sec") or 0.0) for item in per_image)) / float(len(per_image))
    ) if per_image else 0.0
    overall = {
        "gt_total": int(gt_total),
        "raw_hits": int(raw_hits),
        "dedup_hits": int(dedup_hits),
        "raw_coverage": (float(raw_hits) / float(gt_total)) if gt_total else 1.0,
        "dedup_coverage": (float(dedup_hits) / float(gt_total)) if gt_total else 1.0,
        "mean_best_iou": _mean_best_iou(per_image),
        "whole_window_fraction_mean": (
            float(sum(float(item.get("raw_whole_window_fraction") or 0.0) for item in per_image))
            / float(len(per_image))
        ) if per_image else 0.0,
        "degenerate_fraction_mean": (
            float(sum(float(item.get("raw_degenerate_fraction") or 0.0) for item in per_image))
            / float(len(per_image))
        ) if per_image else 0.0,
        "avg_elapsed_sec": avg_elapsed_sec,
        "total_elapsed_sec": float(time.perf_counter() - config_started),
    }
    return {
        "config": dict(config),
        "config_id": config_id,
        "status": status,
        "skip_reason": skip_reason,
        "results": per_image,
        "overall": overall,
    }


def _promotion_key(result: Dict[str, Any]) -> tuple[float, float, float, float]:
    overall = result.get("overall") or {}
    return (
        float(overall.get("dedup_hits") or 0.0),
        -float(overall.get("degenerate_fraction_mean") or 0.0),
        float(overall.get("mean_best_iou") or 0.0),
        -float(overall.get("avg_elapsed_sec") or 0.0),
    )


def _recommend_backend(completed: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    embedded = [item for item in completed if str(item.get("config", {}).get("backend")) == "embedded"]
    server = [item for item in completed if str(item.get("config", {}).get("backend")) == "server"]
    if not embedded or not server:
        return {"recommended_backend": "embedded", "reason": "server_not_compared"}
    best_embedded = max(embedded, key=_promotion_key)
    best_server = max(server, key=_promotion_key)
    emb_hits = float(best_embedded.get("overall", {}).get("dedup_hits") or 0.0)
    srv_hits = float(best_server.get("overall", {}).get("dedup_hits") or 0.0)
    emb_cov = float(best_embedded.get("overall", {}).get("dedup_coverage") or 0.0)
    srv_cov = float(best_server.get("overall", {}).get("dedup_coverage") or 0.0)
    emb_deg = float(best_embedded.get("overall", {}).get("degenerate_fraction_mean") or 0.0)
    srv_deg = float(best_server.get("overall", {}).get("degenerate_fraction_mean") or 0.0)
    if srv_hits > emb_hits and (srv_hits - emb_hits >= 2.0 or srv_cov - emb_cov >= 0.10) and (srv_deg - emb_deg <= 0.05):
        return {"recommended_backend": "server", "reason": "server_materially_better", "embedded": best_embedded["config_id"], "server": best_server["config_id"]}
    if emb_hits <= 0.0 and srv_hits > 0.0:
        return {"recommended_backend": "server", "reason": "embedded_zero_server_nonzero", "embedded": best_embedded["config_id"], "server": best_server["config_id"]}
    return {"recommended_backend": "embedded", "reason": "embedded_not_worse", "embedded": best_embedded["config_id"], "server": best_server["config_id"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=20260410)
    parser.add_argument("--report-root", default="")
    parser.add_argument("--falcon-model-id", default="tiiuae/Falcon-Perception")
    parser.add_argument("--falcon-device", default="cuda:1")
    parser.add_argument("--falcon-server-url", default=os.environ.get("FALCON_SERVER_URL", ""))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--falcon-min-dimension", type=int, default=256)
    parser.add_argument("--backends", default="embedded")
    parser.add_argument(
        "--component-modes",
        default="largest_component,component_split,component_cluster",
    )
    parser.add_argument("--max-dimensions", default="640,768,1024")
    parser.add_argument("--hr-upsample-ratios", default="8,12,16")
    parser.add_argument("--coord-dedup-thresholds", default="0.0,0.005,0.01")
    parser.add_argument("--segmentation-thresholds", default="0.2,0.3,0.4")
    args = parser.parse_args()

    images = _sample_images(int(args.sample_size), int(args.sample_seed))
    report_root = (
        Path(str(args.report_root).strip())
        if str(args.report_root or "").strip()
        else (REPORT_BASE / f"falcon_segmentation_sweep_qwen_random{args.sample_size}_seed{args.sample_seed}")
    )
    report_root.mkdir(parents=True, exist_ok=True)

    configs = list(
        _iter_configs(
            backends=_split_csv(args.backends),
            component_modes=_split_csv(args.component_modes),
            max_dimensions=_split_ints(args.max_dimensions),
            hr_upsample_ratios=_split_ints(args.hr_upsample_ratios),
            coord_dedup_thresholds=_split_floats(args.coord_dedup_thresholds),
            segmentation_thresholds=_split_floats(args.segmentation_thresholds),
        )
    )

    results: List[Dict[str, Any]] = []
    started_at = time.perf_counter()
    for index, config in enumerate(configs, start=1):
        result = _run_config(
            config=config,
            images=images,
            sample_seed=int(args.sample_seed),
            report_root=report_root,
            falcon_model_id=str(args.falcon_model_id),
            falcon_device=str(args.falcon_device),
            iou_threshold=float(args.iou_threshold),
            falcon_min_dimension=int(args.falcon_min_dimension),
            falcon_server_url=str(args.falcon_server_url or ""),
        )
        result["run_index"] = index
        result["run_total"] = len(configs)
        results.append(result)
        (report_root / f"{result['config_id']}.json").write_text(
            json.dumps(_to_jsonable(result), indent=2),
            encoding="utf-8",
        )
        partial = {
            "sample_seed": int(args.sample_seed),
            "sample_size": int(args.sample_size),
            "images": images,
            "configs_total": len(configs),
            "configs_completed": len(results),
            "results": results,
            "elapsed_sec": float(time.perf_counter() - started_at),
        }
        (report_root / "partial_summary.json").write_text(
            json.dumps(_to_jsonable(partial), indent=2),
            encoding="utf-8",
        )

    completed = [item for item in results if str(item.get("status")) == "completed"]
    promoted = max(completed, key=_promotion_key) if completed else None
    summary = {
        "sample_seed": int(args.sample_seed),
        "sample_size": int(args.sample_size),
        "images": images,
        "configs_total": len(configs),
        "elapsed_sec": float(time.perf_counter() - started_at),
        "backend_recommendation": _recommend_backend(completed),
        "promoted_config_id": None if promoted is None else promoted.get("config_id"),
        "promoted_config": None if promoted is None else promoted.get("config"),
        "promoted_overall": None if promoted is None else promoted.get("overall"),
        "results": results,
    }
    (report_root / "summary.json").write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(_to_jsonable(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
