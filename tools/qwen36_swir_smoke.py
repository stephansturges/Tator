"""Small Qwen3.6 runtime smoke for the unified Transformers 5 environment.

This intentionally avoids downloading 35B weights. It verifies that the active
Transformers runtime can resolve the Qwen3.6 architecture and processor.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import transformers
from transformers import AutoConfig, AutoProcessor


DEFAULT_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


def smoke(model_id: str) -> Dict[str, Any]:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return {
        "model_id": model_id,
        "transformers": transformers.__version__,
        "config_class": type(config).__name__,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "processor_class": type(processor).__name__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()
    print(json.dumps(smoke(args.model_id), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
