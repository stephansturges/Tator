#!/usr/bin/env python3
"""Metadata smoke for inference-only agent model candidates.

This intentionally avoids downloading full weights by default. Use the isolated
`.venv-qwen36-swir` environment for Transformers 5.x model types.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.agent_model_catalog import AGENT_MODEL_OPTIONS  # noqa: E402
from services.qwen_mlx import QWEN_PLATFORM_TRANSFORMERS  # noqa: E402


def _smoke_transformers_metadata(model_id: str) -> Dict[str, Any]:
    try:
        from transformers import AutoConfig, AutoProcessor
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"transformers_import_failed:{exc}"}
    result: Dict[str, Any] = {"ok": True}
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        result["config_class"] = type(config).__name__
        result["model_type"] = getattr(config, "model_type", None)
        result["architectures"] = getattr(config, "architectures", None)
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["config_error"] = str(exc)
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
        result["processor_class"] = type(processor).__name__
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["processor_error"] = str(exc)
    return result


def run_smoke() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in AGENT_MODEL_OPTIONS:
        model_id = str(entry["id"])
        row: Dict[str, Any] = {
            "id": model_id,
            "label": entry.get("label"),
            "family": entry.get("model_family"),
            "runtime_platform": entry.get("runtime_platform"),
            "vision_inference_supported": entry.get("vision_inference_supported"),
            "training_supported": entry.get("training_supported"),
        }
        if entry.get("runtime_platform") == QWEN_PLATFORM_TRANSFORMERS and entry.get(
            "vision_inference_supported", True
        ):
            row["metadata_smoke"] = _smoke_transformers_metadata(model_id)
        else:
            row["metadata_smoke"] = {
                "ok": entry.get("vision_inference_supported", True) is not False,
                "mode": "catalog_only",
                "note": entry.get("compatibility_note"),
            }
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit one JSON array instead of JSON lines.")
    args = parser.parse_args()
    rows = run_smoke()
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    return 0 if all(row.get("metadata_smoke", {}).get("ok") for row in rows if row.get("vision_inference_supported")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
