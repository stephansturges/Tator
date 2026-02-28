from __future__ import annotations

from pathlib import Path

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
