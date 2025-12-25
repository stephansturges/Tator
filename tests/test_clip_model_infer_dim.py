import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import _infer_clip_model_from_embedding_dim  # noqa: E402


def test_clip_model_infer_dim_512_defaults_to_b32():
    assert _infer_clip_model_from_embedding_dim(512) == "ViT-B/32"
