from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

import localinferenceapi as api


def _fake_coco_index(ids: list[int]):
    return (
        {"images": [{"id": i} for i in ids], "annotations": [], "categories": []},
        {},
        {i: {"id": i, "path": f"img_{i}.jpg"} for i in ids},
    )


def test_agent_mining_sample_cache_hits_on_same_signature(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "agent_cache"
    monkeypatch.setattr(api, "AGENT_MINING_DET_CACHE_ROOT", cache_root)
    monkeypatch.setattr(api, "_compute_dir_signature_impl", lambda _root: "sig_a")
    monkeypatch.setattr(api, "_load_coco_index_impl", lambda _root: _fake_coco_index([1, 2, 3, 4, 5]))

    first = api._ensure_agent_mining_sample("ds1", tmp_path / "dataset", sample_size=3, seed=42)
    second = api._ensure_agent_mining_sample("ds1", tmp_path / "dataset", sample_size=3, seed=42)

    assert first["_cached"] is False
    assert second["_cached"] is True
    assert first["sample_ids"] == second["sample_ids"]
    assert first["dataset_signature"] == "sig_a"


def test_agent_mining_sample_cache_invalidates_on_signature_change(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "agent_cache"
    monkeypatch.setattr(api, "AGENT_MINING_DET_CACHE_ROOT", cache_root)
    signatures = iter(["sig_a", "sig_b"])
    monkeypatch.setattr(api, "_compute_dir_signature_impl", lambda _root: next(signatures))
    monkeypatch.setattr(api, "_load_coco_index_impl", lambda _root: _fake_coco_index([10, 11, 12, 13]))

    first = api._ensure_agent_mining_sample("ds1", tmp_path / "dataset", sample_size=2, seed=7)
    second = api._ensure_agent_mining_sample("ds1", tmp_path / "dataset", sample_size=2, seed=7)

    assert first["_cached"] is False
    assert second["_cached"] is False
    assert first["dataset_signature"] == "sig_a"
    assert second["dataset_signature"] == "sig_b"


def test_agent_mining_cache_size_skips_symlink_file_escape(
    tmp_path: Path, monkeypatch
) -> None:
    cache_root = tmp_path / "agent_cache"
    cache_root.mkdir()
    (cache_root / "safe.bin").write_bytes(b"safe")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"external")
    try:
        (cache_root / "escape.bin").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "AGENT_MINING_DET_CACHE_ROOT", cache_root)

    out = api.agent_mining_cache_size()

    assert out["bytes"] == 4
    assert out["files"] == 1


def test_agent_mining_cache_purge_unlinks_symlink_without_counting_target_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    cache_root = tmp_path / "agent_cache"
    cache_root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"external")
    try:
        (cache_root / "escape.bin").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "AGENT_MINING_DET_CACHE_ROOT", cache_root)
    with api.AGENT_MINING_JOBS_LOCK:
        api.AGENT_MINING_JOBS.clear()

    out = api.agent_mining_cache_purge()

    assert out == {"status": "ok", "deleted_bytes": 0, "deleted_files": 1}
    assert not (cache_root / "escape.bin").exists()
    assert outside.read_bytes() == b"external"


def test_agent_mining_cache_ttl_prune_unlinks_symlink_dir_without_target_delete(
    tmp_path: Path, monkeypatch
) -> None:
    cache_root = tmp_path / "agent_cache"
    cache_root.mkdir()
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "payload.bin").write_bytes(b"external")
    link = cache_root / "linked_dir"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    old = time.time() - 7200
    try:
        os.utime(link, (old, old), follow_symlinks=False)
    except OSError as exc:
        pytest.skip(f"symlink timestamp update unsupported: {exc}")
    monkeypatch.setattr(api, "AGENT_MINING_CACHE_TTL_HOURS", 1)
    monkeypatch.setattr(api, "AGENT_MINING_CACHE_MAX_BYTES", 0)
    with api.AGENT_MINING_JOBS_LOCK:
        api.AGENT_MINING_JOBS.clear()

    api._enforce_agent_mining_cache_limits(cache_root)

    assert not link.exists()
    assert (outside / "payload.bin").read_bytes() == b"external"
