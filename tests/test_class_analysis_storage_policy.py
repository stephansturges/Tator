from pathlib import Path

import pytest

from services.class_analysis_storage_policy import (
    DEFAULT_GENERATED_DATA_BUDGET_BYTES,
    StoragePolicyError,
    choose_unpinned_session_evictions,
    read_generated_data_budget,
    recoverable_session_inventory,
    write_generated_data_budget,
)


def test_policy_defaults_and_persists_atomically(tmp_path: Path) -> None:
    path = tmp_path / "storage_policy.json"
    assert read_generated_data_budget(
        path, default_bytes=DEFAULT_GENERATED_DATA_BUDGET_BYTES
    ) == DEFAULT_GENERATED_DATA_BUDGET_BYTES
    write_generated_data_budget(path, 75 * 1024**3)
    assert read_generated_data_budget(
        path, default_bytes=DEFAULT_GENERATED_DATA_BUDGET_BYTES
    ) == 75 * 1024**3
    assert not list(tmp_path.glob("*.tmp"))


def test_inventory_and_eviction_never_select_pins(tmp_path: Path) -> None:
    for name, size in (("ca_old", 30), ("ca_latest", 40), ("ca_active", 50)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "artifact.bin").write_bytes(b"x" * size)
    inventory = recoverable_session_inventory(
        tmp_path, pinned_job_ids={"ca_latest", "ca_active"}
    )
    victims = choose_unpinned_session_evictions(
        inventory["sessions"], cache_bytes=20, max_bytes=100
    )
    assert [row["job_id"] for row in victims] == ["ca_old"]
    assert all(not row["pinned"] for row in victims)


def test_policy_rejects_sub_gibibyte_budget(tmp_path: Path) -> None:
    with pytest.raises(StoragePolicyError, match="out_of_range"):
        write_generated_data_budget(tmp_path / "policy.json", 1024)
