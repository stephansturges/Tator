#!/usr/bin/env python3
"""Provision Tator's pinned MLX SAM3 checkpoint in the Hugging Face cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.mlx_sam3 import (
    MLX_SAM3_VARIANTS,
    normalize_mlx_sam3_runtime,
)


SETUP_ROOT = REPO_ROOT / ".cache" / "tator" / "mlx-sam3"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the pinned MLX SAM3 runtime.")
    parser.add_argument("--variant", default="mlx-bf16", choices=tuple(MLX_SAM3_VARIANTS))
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if (args.model is None) != (args.revision is None):
        parser.error("--model and --revision must be supplied together")
    if args.model:
        downloads = [("custom", args.model, args.revision)]
    elif args.all_variants:
        downloads = [
            (runtime_id, spec.model_id, spec.revision)
            for runtime_id, spec in MLX_SAM3_VARIANTS.items()
        ]
    else:
        runtime_id = normalize_mlx_sam3_runtime(args.variant)
        spec = MLX_SAM3_VARIANTS[runtime_id]
        downloads = [(runtime_id, spec.model_id, spec.revision)]

    manifests = []
    for runtime_id, model_id, revision in downloads:
        model_path = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                force_download=args.force,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            )
        ).resolve()
        if not (model_path / "config.json").is_file() or not (
            (model_path / "model.safetensors").is_file()
            or (model_path / "model.safetensors.index.json").is_file()
        ):
            raise RuntimeError(f"MLX SAM3 download is incomplete: {runtime_id}")
        manifests.append(
            {
                "runtime": runtime_id,
                "model": model_id,
                "revision": revision,
                "model_path": str(model_path),
            }
        )

    SETUP_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"variants": manifests}
    (SETUP_ROOT / "setup.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"MLX SAM3 is ready: {', '.join(item['runtime'] for item in manifests)}")
    if len(manifests) == 1:
        print(f"SAM3_MLX_MODEL_PATH={manifests[0]['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
