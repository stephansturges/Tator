from __future__ import annotations

from pathlib import Path
from typing import Optional

from simple_tokenizer import SimpleTokenizer, default_bpe

SAM3_LITE_BPE = Path(__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"


def resolve_bpe_path(preferred: Optional[str] = None) -> Path:
    if preferred:
        preferred_path = Path(preferred)
        if preferred_path.exists():
            return preferred_path
    if SAM3_LITE_BPE.exists():
        return SAM3_LITE_BPE
    return Path(default_bpe())


def build_tokenizer(bpe_path: Optional[str] = None) -> SimpleTokenizer:
    path = resolve_bpe_path(bpe_path)
    return SimpleTokenizer(str(path))
