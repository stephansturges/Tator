#!/usr/bin/env python3
"""Benchmark pooled DINOv3 versus locally trained SALAD for image diversity."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import localinferenceapi as api
from services.data_ingestion import IMAGE_EXTS


def _image_rows(image_dir: Path, sample_cap: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rows.append(
                {
                    "path": str(path.resolve()),
                    "filename": path.name,
                    "saved_name": path.name,
                    "size": path.stat().st_size,
                    "field": "benchmark",
                }
            )
    if sample_cap > 0:
        rows = rows[:sample_cap]
    return rows


def _run_selection(
    *,
    rows: List[Dict[str, Any]],
    encoder: str,
    model_name: str,
    salad_head_id: str,
    cradio_pooling: str,
    keep_fraction: float,
    out_dir: Path,
) -> Dict[str, Any]:
    job = api.DataIngestionJob(
        job_id=f"bench_{encoder}_{int(time.time())}",
        kind="analysis",
        request={},
    )
    prepared = api._data_ingestion_prepare_media(
        job,
        rows,
        out_dir=out_dir / f"prepared_{encoder}",
        frame_interval=1.0,
        max_frames_per_video=0,
        progress_start=0.0,
        progress_end=0.1,
    )
    embeddings = api._data_ingestion_encode_prepared_images(
        prepared,
        job=job,
        encoder=encoder,
        model_name=model_name,
        salad_head_id=salad_head_id,
        cradio_pooling=cradio_pooling,
        batch_size=16,
        progress_start=0.1,
        progress_end=0.9,
    )
    selected, novelty = api._data_ingestion_greedy_indices(
        embeddings,
        keep_fraction=keep_fraction,
        reference_embeddings=None,
    )
    summary = api._data_ingestion_diversity_summary(
        embeddings,
        selected_indices=selected,
        reference_embeddings=None,
    )
    return {
        "encoder": encoder,
        "salad_head_id": salad_head_id,
        "summary": summary,
        "selected_indices": selected,
        "selected_files": [prepared[idx]["filename"] for idx in selected],
        "novelty_mean": float(np.mean(novelty)) if len(novelty) else 0.0,
    }


def _train_head_if_requested(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> Optional[str]:
    if not args.train_local_salad:
        return args.salad_head_id or None
    job = api.DataIngestionJob(
        job_id=f"bench_salad_train_{int(time.time())}",
        kind="local_salad_train",
        request={
            "train_uploads": rows,
            "head_name": args.head_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_train_images": args.sample_cap,
            "max_frames_per_video": 0,
            "encoder_type": args.train_encoder,
            "encoder_model": args.cradio_model if args.train_encoder == "cradio" else api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL,
            "cradio_pooling": args.cradio_pooling,
            "num_clusters": args.num_clusters,
            "cluster_dim": args.cluster_dim,
            "token_dim": args.token_dim,
            "dropout": args.dropout,
        },
    )
    api._run_local_salad_training_job(job)
    if job.status != "completed":
        raise RuntimeError(f"local SALAD training failed: {job.error or job.message}")
    return str((job.result or {}).get("summary", {}).get("head_id") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("uploads/data_ingestion/benchmarks/salad_diversity"))
    parser.add_argument("--sample-cap", type=int, default=64)
    parser.add_argument("--keep-fraction", type=float, default=0.2)
    parser.add_argument("--salad-head-id", default="")
    parser.add_argument("--train-local-salad", action="store_true")
    parser.add_argument("--train-encoder", choices=["dinov3", "cradio"], default="dinov3")
    parser.add_argument("--include-cradio-pooled", action="store_true")
    parser.add_argument("--cradio-model", default=api.CRADIO_DEFAULT_MODEL)
    parser.add_argument(
        "--cradio-pooling",
        choices=["summary", "spatial_mean", "summary_spatial_concat"],
        default="summary",
    )
    parser.add_argument("--head-name", default="benchmark_local_salad")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-clusters", type=int, default=64)
    parser.add_argument("--cluster-dim", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--delete-trained-head", action="store_true")
    args = parser.parse_args()

    rows = _image_rows(args.image_dir, args.sample_cap)
    if len(rows) < 2:
        raise SystemExit("Need at least two images for diversity benchmarking.")
    out_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    salad_head_id = _train_head_if_requested(args, rows)
    trained_head_id = salad_head_id if args.train_local_salad else ""

    pooled = _run_selection(
        rows=rows,
        encoder="dinov3_pooled",
        model_name=api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL,
        salad_head_id="",
        cradio_pooling=args.cradio_pooling,
        keep_fraction=args.keep_fraction,
        out_dir=out_dir,
    )
    results = {"pooled": pooled}
    if args.include_cradio_pooled:
        results["cradio_pooled"] = _run_selection(
            rows=rows,
            encoder="cradio_pooled",
            model_name=args.cradio_model,
            salad_head_id="",
            cradio_pooling=args.cradio_pooling,
            keep_fraction=args.keep_fraction,
            out_dir=out_dir,
        )
    if salad_head_id:
        results["local_salad"] = _run_selection(
            rows=rows,
            encoder="local_salad",
            model_name=args.cradio_model if args.train_encoder == "cradio" else api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL,
            salad_head_id=salad_head_id,
            cradio_pooling=args.cradio_pooling,
            keep_fraction=args.keep_fraction,
            out_dir=out_dir,
        )
        pooled_set = set(pooled["selected_indices"])
        salad_set = set(results["local_salad"]["selected_indices"])
        union = pooled_set | salad_set
        results["selection_overlap_jaccard"] = float(len(pooled_set & salad_set) / len(union)) if union else 1.0
        results["recommended_recipes"] = {
            "data_ingestion": [
                {
                    "id": "local_salad_top20",
                    "encoder": "local_salad",
                    "salad_head_id": salad_head_id,
                    "keep_fraction": args.keep_fraction,
                    "reason": "Use the locally trained SALAD head for novelty ranking against candidate media or active-workspace references.",
                },
                {
                    "id": "pooled_top20",
                    "encoder": "dinov3_pooled",
                    "keep_fraction": args.keep_fraction,
                    "reason": "Baseline pooled DINOv3 recipe for comparison and fallback.",
                },
                {
                    "id": "cradio_top20",
                    "encoder": "cradio_pooled",
                    "encoder_model": args.cradio_model,
                    "cradio_pooling": args.cradio_pooling,
                    "keep_fraction": args.keep_fraction,
                    "reason": "C-RADIOv4 pooled candidate; benchmark before promoting over DINOv3.",
                },
            ],
            "class_separation": [
                {
                    "id": "precise",
                    "encoder_type": "dinov3",
                    "embedding_aggregation": "pooled",
                    "reason": "Measured pooled crop baseline. Local SALAD heads remain limited to whole-image Data Ingestion diversity scoring.",
                },
            ],
        }
    trained_head_deleted = False
    if args.delete_trained_head and trained_head_id:
        safe_head = api._class_analysis_safe_slug(trained_head_id, "")
        head_path = api.LOCAL_SALAD_HEAD_ROOT / f"{safe_head}.pt"
        try:
            head_path.unlink()
            trained_head_deleted = True
        except FileNotFoundError:
            trained_head_deleted = True
        except OSError:
            trained_head_deleted = False
    results["benchmark_metadata"] = {
        "local_salad_policy": api.LOCAL_SALAD_POLICY,
        "loaded_external_salad_checkpoint": False,
        "trained_head_id": trained_head_id,
        "trained_head_deleted": trained_head_deleted,
        "train_encoder": args.train_encoder,
        "cradio_model": args.cradio_model,
        "cradio_pooling": args.cradio_pooling,
    }
    report_path = out_dir / "benchmark.json"
    report_path.write_text(json.dumps(api._class_analysis_json_safe(results), indent=2), encoding="utf-8")
    print(json.dumps(api._class_analysis_json_safe({"output": str(report_path), **results}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
