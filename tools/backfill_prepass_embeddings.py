#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise SystemExit(f"Invalid integer in --embed-proj-dims: {token}") from exc
        if value <= 0:
            raise SystemExit(f"Embedding projection dims must be > 0, got: {value}")
        values.append(value)
    if not values:
        raise SystemExit("At least one embedding projection dim is required.")
    return sorted(set(values))


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-off backfill: build classifier-enriched feature matrices from existing prepass caches."
    )
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. qwen_dataset).")
    parser.add_argument(
        "--cache-key",
        action="append",
        required=True,
        help="Prepass cache key under uploads/calibration_cache/prepass (repeatable).",
    )
    parser.add_argument("--classifier-id", required=True, help="Classifier id/path under uploads/classifiers.")
    parser.add_argument(
        "--embed-proj-dims",
        default="1024",
        help="Comma-separated projection dims to build (default: 1024).",
    )
    parser.add_argument("--images-file", default=None, help="Optional fixed image list (JSON list or newline).")
    parser.add_argument("--support-iou", type=float, default=0.5)
    parser.add_argument("--context-radius", type=float, default=0.075)
    parser.add_argument("--term-hash-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-crop-size", type=int, default=4)
    parser.add_argument("--embed-proj-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="uploads/calibration_cache/features_backfill")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if they already exist.")
    args = parser.parse_args()

    dims = _parse_int_list(args.embed_proj_dims)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": args.dataset,
        "classifier_id": args.classifier_id,
        "cache_keys": args.cache_key,
        "embed_proj_dims": dims,
        "images_file": args.images_file,
        "device": args.device,
        "created_at": time.time(),
        "runs": [],
    }

    repo_root = Path(__file__).resolve().parents[1]
    tools_dir = repo_root / "tools"

    for cache_key in args.cache_key:
        run_root = root / cache_key
        run_root.mkdir(parents=True, exist_ok=True)
        prepass_jsonl = run_root / "prepass.jsonl"

        materialize_cmd = [
            sys.executable,
            str(tools_dir / "materialize_prepass_from_cache.py"),
            "--cache-key",
            str(cache_key),
            "--output",
            str(prepass_jsonl),
        ]
        if args.images_file:
            materialize_cmd += ["--images-file", str(args.images_file)]
        _run(materialize_cmd)

        run_entry = {
            "cache_key": cache_key,
            "prepass_jsonl": str(prepass_jsonl),
            "features": [],
        }

        for dim in dims:
            out_npz = run_root / f"ensemble_features_embed{dim}.npz"
            out_meta = run_root / f"ensemble_features_embed{dim}.meta.json"
            if out_npz.exists() and not args.force:
                run_entry["features"].append(
                    {
                        "embed_proj_dim": dim,
                        "path": str(out_npz),
                        "skipped": True,
                    }
                )
                continue

            build_cmd = [
                sys.executable,
                str(tools_dir / "build_ensemble_features.py"),
                "--input",
                str(prepass_jsonl),
                "--dataset",
                str(args.dataset),
                "--output",
                str(out_npz),
                "--classifier-id",
                str(args.classifier_id),
                "--require-classifier",
                "--support-iou",
                str(float(args.support_iou)),
                "--context-radius",
                str(float(args.context_radius)),
                "--term-hash-dim",
                str(int(args.term_hash_dim)),
                "--batch-size",
                str(int(args.batch_size)),
                "--min-crop-size",
                str(int(args.min_crop_size)),
                "--embed-proj-dim",
                str(int(dim)),
                "--embed-proj-seed",
                str(int(args.embed_proj_seed)),
                "--device",
                str(args.device),
            ]
            _run(build_cmd)

            out_meta.write_text(
                json.dumps(
                    {
                        "cache_key": cache_key,
                        "dataset": args.dataset,
                        "classifier_id": args.classifier_id,
                        "embed_proj_dim": int(dim),
                        "embed_proj_seed": int(args.embed_proj_seed),
                        "support_iou": float(args.support_iou),
                        "context_radius": float(args.context_radius),
                        "term_hash_dim": int(args.term_hash_dim),
                        "batch_size": int(args.batch_size),
                        "min_crop_size": int(args.min_crop_size),
                        "device": args.device,
                        "created_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            run_entry["features"].append(
                {
                    "embed_proj_dim": dim,
                    "path": str(out_npz),
                    "meta": str(out_meta),
                    "skipped": False,
                }
            )

        manifest["runs"].append(run_entry)

    manifest_path = root / "backfill_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "runs": len(manifest["runs"])}, indent=2))


if __name__ == "__main__":
    main()
