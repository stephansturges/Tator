#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.context_feature_variants import (
    COMBINED_VARIANT,
    IMGRAW_VARIANT,
    SCENE_SUMMARY_VARIANT,
    TRUSTED_CENTROID_VARIANT,
    derive_variant_payload,
    load_npz_payload,
    save_npz_payload,
)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive image-context feature variants from a labeled .npz.")
    parser.add_argument("--input-labeled", required=True, help="Source labeled .npz artifact.")
    parser.add_argument("--output-labeled", required=True, help="Destination labeled .npz artifact.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            IMGRAW_VARIANT,
            SCENE_SUMMARY_VARIANT,
            TRUSTED_CENTROID_VARIANT,
            COMBINED_VARIANT,
        ],
        help="Derived context variant to produce.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional path for a derivation summary JSON sidecar.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_labeled).resolve()
    output_path = Path(args.output_labeled).resolve()
    payload = load_npz_payload(input_path)
    derived_payload, summary = derive_variant_payload(
        payload,
        variant=str(args.variant),
        parent_feature_npz=str(input_path),
    )
    save_npz_payload(output_path, derived_payload)

    if args.summary_json:
        _write_summary(
            Path(args.summary_json).resolve(),
            {
                "input_labeled": str(input_path),
                "output_labeled": str(output_path),
                "variant": str(args.variant),
                "summary": summary,
                "feature_schema_hash": str(derived_payload["feature_schema_hash"].item()),
            },
        )


if __name__ == "__main__":
    main()
