from __future__ import annotations

from pathlib import Path

import pytest

import localinferenceapi as api


def test_init_storage_root_rejects_symlinked_parent_before_mkdir(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(RuntimeError, match="storage_root_symlink"):
        api._init_storage_root(linked_parent / "nested" / "uploads")

    assert not (outside / "nested").exists()


def test_init_storage_root_creates_normal_root(tmp_path: Path) -> None:
    root = tmp_path / "uploads" / "cache"

    out = api._init_storage_root(root)

    assert out == root
    assert root.is_dir()
