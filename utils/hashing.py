"""Hashing helpers."""

from __future__ import annotations

import hashlib
from typing import Sequence


def _stable_hash_impl(entries: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for item in entries:
        digest.update(item.encode("utf-8"))
    return digest.hexdigest()


def _stable_hash(entries: Sequence[str]) -> str:
    return _stable_hash_impl(entries)
