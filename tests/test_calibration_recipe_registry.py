from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from services.calibration_recipe_registry import (
    _atomic_write_json,
    _write_text_within_parent,
    build_recipe_fingerprint,
    build_recipe_fingerprint_payload,
    discovery_lock,
    find_matching_recipe,
    load_registry,
    register_promoted_recipe,
)


def _sample_payload(**overrides):
    payload = build_recipe_fingerprint_payload(
        dataset_id="demo",
        labelmap_hash="labelhash",
        glossary_hash="glossaryhash",
        classifier_id="clf.pkl",
        lane_selection="window",
        prepass_config={"sam3_text_window_extension": True, "detector_conf": 0.45},
        selected_hash="selhash",
        selected_count=2000,
        selection_seed=42,
        requested_max_images=2000,
        support_iou=0.5,
        context_radius=0.075,
        label_iou=0.5,
        eval_iou=0.5,
        feature_version=7,
    )
    payload.update(overrides)
    return payload


def test_recipe_fingerprint_changes_when_prepass_context_changes() -> None:
    base = _sample_payload()
    changed = _sample_payload(prepass_config={"sam3_text_window_extension": False, "detector_conf": 0.45})

    assert build_recipe_fingerprint(base) != build_recipe_fingerprint(changed)


def test_recipe_fingerprint_changes_when_lane_selection_changes() -> None:
    base = _sample_payload(lane_selection="window")
    changed = _sample_payload(lane_selection="compare_both")

    assert build_recipe_fingerprint(base) != build_recipe_fingerprint(changed)


def test_recipe_fingerprint_changes_when_selection_context_changes() -> None:
    base = _sample_payload()
    changed = _sample_payload(selected_hash="other-sample", requested_max_images=300, selection_seed=7)

    assert build_recipe_fingerprint(base) != build_recipe_fingerprint(changed)


def test_register_promoted_recipe_persists_registry_entry(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    fingerprint_payload = _sample_payload()
    fingerprint = build_recipe_fingerprint(fingerprint_payload)
    discovery_root = tmp_path / "discovery"
    discovery_root.mkdir(parents=True, exist_ok=True)
    canonical_json = discovery_root / "canonical_edr.json"
    canonical_json.write_text(json.dumps({"canonical_windowed_recipe": {"winner_lane": "window"}}))
    canonical_md = discovery_root / "canonical_edr.md"
    canonical_md.write_text("# recipe\n")

    entry = register_promoted_recipe(
        cache_root,
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
        dataset_id="demo",
        canonical_recipe_json=canonical_json,
        canonical_recipe_md=canonical_md,
        report_bundle_json=None,
        discovery_run_root=discovery_root,
        edr_package={
            "id": "canonical_edr_pkg_demo_abcd1234",
            "package_root": str((tmp_path / "edr_packages" / "canonical_edr_pkg_demo_abcd1234").resolve()),
            "package_zip": str((tmp_path / "edr_packages" / "canonical_edr_pkg_demo_abcd1234" / "package.edr.zip").resolve()),
            "package_sha256": "sha256",
        },
    )

    assert entry["fingerprint"] == fingerprint
    assert Path(entry["canonical_recipe_json"]).exists()
    assert Path(entry["canonical_recipe_md"]).exists()
    assert Path(entry["canonical_edr_json"]).exists()
    assert Path(entry["legacy_canonical_recipe_json"]).exists()
    assert Path(entry["registry_entry_json"]).exists()
    assert entry["edr_package_id"] == "canonical_edr_pkg_demo_abcd1234"
    assert entry["edr_package_sha256"] == "sha256"
    assert find_matching_recipe(cache_root, fingerprint)["fingerprint"] == fingerprint
    registry = load_registry(cache_root)
    assert fingerprint in registry["entries"]


def test_recipe_registry_atomic_write_replaces_symlink_targets_without_target_write(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "registry" / "index.json"
    json_path.parent.mkdir()
    outside_tmp = tmp_path / "outside_tmp.json"
    outside_final = tmp_path / "outside_final.json"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    tmp_link = json_path.with_suffix(json_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_link.symlink_to(outside_tmp)
        json_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _atomic_write_json(json_path, {"version": 1, "entries": {}})

    assert not tmp_link.exists()
    assert not json_path.is_symlink()
    assert json.loads(json_path.read_text(encoding="utf-8"))["version"] == 1
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_recipe_registry_atomic_write_rejects_nested_symlinked_parent_before_mkdir(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="recipe_registry_json_parent_symlink"):
        _atomic_write_json(
            linked_parent / "nested" / "registry" / "index.json",
            {"version": 1, "entries": {}},
        )

    assert list(outside.iterdir()) == []


def test_recipe_registry_text_write_rejects_nested_symlinked_parent_before_mkdir(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="recipe_registry_text_parent_symlink"):
        _write_text_within_parent(
            linked_parent / "nested" / "registry" / "canonical_edr.json",
            "{}",
        )

    assert list(outside.iterdir()) == []


def test_register_promoted_recipe_replaces_symlinked_recipe_dir_without_target_write(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    fingerprint_payload = _sample_payload()
    fingerprint = build_recipe_fingerprint(fingerprint_payload)
    registry_root = cache_root / "recipe_registry"
    registry_root.mkdir(parents=True)
    outside = tmp_path / "outside_recipe"
    outside.mkdir()
    (outside / "canonical_edr.json").write_text("external", encoding="utf-8")
    try:
        (registry_root / fingerprint).symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    canonical_json = tmp_path / "canonical_edr.json"
    canonical_json.write_text(json.dumps({"canonical_windowed_recipe": {"winner_lane": "window"}}))

    entry = register_promoted_recipe(
        cache_root,
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
        dataset_id="demo",
        canonical_recipe_json=canonical_json,
        canonical_recipe_md=None,
        report_bundle_json=None,
        discovery_run_root=None,
    )

    recipe_dir = Path(entry["recipe_root"])
    assert not recipe_dir.is_symlink()
    assert (recipe_dir / "canonical_edr.json").exists()
    assert (outside / "canonical_edr.json").read_text(encoding="utf-8") == "external"


def test_load_registry_skips_symlinked_index_escape(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    registry_root = cache_root / "recipe_registry"
    registry_root.mkdir(parents=True)
    outside = tmp_path / "outside_index.json"
    outside.write_text(json.dumps({"version": 1, "entries": {"escaped": {}}}), encoding="utf-8")
    try:
        (registry_root / "index.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    registry = load_registry(cache_root)

    assert registry["entries"] == {}


def test_load_registry_skips_symlinked_cache_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside_cache"
    registry_root = outside / "recipe_registry"
    registry_root.mkdir(parents=True)
    (registry_root / "index.json").write_text(
        json.dumps({"version": 1, "entries": {"escaped": {}}}),
        encoding="utf-8",
    )
    cache_root = tmp_path / "cache"
    try:
        cache_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    registry = load_registry(cache_root)

    assert registry["entries"] == {}


def test_load_registry_skips_symlinked_cache_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside_parent"
    registry_root = outside / "cache" / "recipe_registry"
    registry_root.mkdir(parents=True)
    (registry_root / "index.json").write_text(
        json.dumps({"version": 1, "entries": {"escaped": {}}}),
        encoding="utf-8",
    )
    cache_parent = tmp_path / "linked_parent"
    try:
        cache_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    registry = load_registry(cache_parent / "cache")

    assert registry["entries"] == {}


def test_load_registry_skips_nested_symlinked_cache_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside_parent"
    registry_root = outside / "nested" / "cache" / "recipe_registry"
    registry_root.mkdir(parents=True)
    (registry_root / "index.json").write_text(
        json.dumps({"version": 1, "entries": {"escaped": {}}}),
        encoding="utf-8",
    )
    cache_parent = tmp_path / "linked_parent"
    try:
        cache_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    registry = load_registry(cache_parent / "nested" / "cache")

    assert registry["entries"] == {}


def test_register_promoted_recipe_rejects_symlinked_cache_parent_without_target_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    cache_parent = tmp_path / "linked_parent"
    try:
        cache_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    fingerprint_payload = _sample_payload()
    fingerprint = build_recipe_fingerprint(fingerprint_payload)
    canonical_json = tmp_path / "canonical_edr.json"
    canonical_json.write_text(json.dumps({"canonical_windowed_recipe": {"winner_lane": "window"}}))

    with pytest.raises(ValueError, match="recipe_registry_cache_root_symlink"):
        register_promoted_recipe(
            cache_parent / "cache",
            fingerprint=fingerprint,
            fingerprint_payload=fingerprint_payload,
            dataset_id="demo",
            canonical_recipe_json=canonical_json,
            canonical_recipe_md=None,
            report_bundle_json=None,
            discovery_run_root=None,
        )

    assert list(outside.iterdir()) == []


def test_register_promoted_recipe_rejects_nested_symlinked_cache_parent_without_target_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    cache_parent = tmp_path / "linked_parent"
    try:
        cache_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    fingerprint_payload = _sample_payload()
    fingerprint = build_recipe_fingerprint(fingerprint_payload)
    canonical_json = tmp_path / "canonical_edr.json"
    canonical_json.write_text(json.dumps({"canonical_windowed_recipe": {"winner_lane": "window"}}))

    with pytest.raises(ValueError, match="recipe_registry_cache_root_symlink"):
        register_promoted_recipe(
            cache_parent / "nested" / "cache",
            fingerprint=fingerprint,
            fingerprint_payload=fingerprint_payload,
            dataset_id="demo",
            canonical_recipe_json=canonical_json,
            canonical_recipe_md=None,
            report_bundle_json=None,
            discovery_run_root=None,
        )

    assert list(outside.iterdir()) == []


def test_discovery_lock_replaces_symlink_target_without_target_write(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    lock_root = cache_root / "discovery_runs"
    lock_root.mkdir(parents=True)
    outside = tmp_path / "outside.lock"
    outside.write_text("external", encoding="utf-8")
    lock_path = lock_root / "abc.lock"
    try:
        lock_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with discovery_lock(cache_root, "abc"):
        assert lock_path.exists()
        assert not lock_path.is_symlink()

    assert outside.read_text(encoding="utf-8") == "external"


def test_discovery_lock_rejects_symlinked_cache_parent_without_target_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    cache_parent = tmp_path / "linked_parent"
    try:
        cache_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="recipe_discovery_cache_root_symlink"):
        with discovery_lock(cache_parent / "cache", "abc"):
            pass

    assert list(outside.iterdir()) == []


def test_discovery_lock_rejects_nested_symlinked_cache_parent_without_target_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    cache_parent = tmp_path / "linked_parent"
    try:
        cache_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="recipe_discovery_cache_root_symlink"):
        with discovery_lock(cache_parent / "nested" / "cache", "abc"):
            pass

    assert list(outside.iterdir()) == []


def test_discovery_lock_rejects_pathlike_fingerprint_before_escape_creation(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"

    with pytest.raises(ValueError, match="recipe_discovery_lock_name_not_allowed"):
        with discovery_lock(cache_root, "../escape"):
            pass

    assert not (cache_root / "escape.lock").exists()


def test_register_promoted_recipe_rejects_pathlike_fingerprint_before_escape_creation(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    canonical_json = tmp_path / "canonical_edr.json"
    canonical_json.write_text(json.dumps({"canonical_windowed_recipe": {"winner_lane": "window"}}))

    with pytest.raises(ValueError, match="recipe_registry_entry_name_not_allowed"):
        register_promoted_recipe(
            cache_root,
            fingerprint="../escape",
            fingerprint_payload=_sample_payload(),
            dataset_id="demo",
            canonical_recipe_json=canonical_json,
            canonical_recipe_md=None,
            report_bundle_json=None,
            discovery_run_root=None,
        )

    assert not (cache_root / "escape").exists()


def test_register_promoted_recipe_replaces_symlinked_index_lock_without_target_write(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    registry_root = cache_root / "recipe_registry"
    registry_root.mkdir(parents=True)
    outside = tmp_path / "outside_index.lock"
    outside.write_text("external", encoding="utf-8")
    try:
        (registry_root / ".index.lock").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    fingerprint_payload = _sample_payload()
    fingerprint = build_recipe_fingerprint(fingerprint_payload)
    canonical_json = tmp_path / "canonical_edr.json"
    canonical_json.write_text(json.dumps({"canonical_windowed_recipe": {"winner_lane": "window"}}))

    register_promoted_recipe(
        cache_root,
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
        dataset_id="demo",
        canonical_recipe_json=canonical_json,
        canonical_recipe_md=None,
        report_bundle_json=None,
        discovery_run_root=None,
    )

    assert not (registry_root / ".index.lock").is_symlink()
    assert outside.read_text(encoding="utf-8") == "external"


def test_register_promoted_recipe_is_safe_under_concurrent_writes(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    discovery_root = tmp_path / "discovery"
    discovery_root.mkdir(parents=True, exist_ok=True)

    def _register(idx: int) -> str:
        fingerprint_payload = _sample_payload(selected_hash=f"sel-{idx}", requested_max_images=300 + idx)
        fingerprint = build_recipe_fingerprint(fingerprint_payload)
        canonical_json = discovery_root / f"canonical_{idx}.json"
        canonical_json.write_text(json.dumps({"canonical_windowed_recipe": {"winner_lane": "window"}}))
        canonical_md = discovery_root / f"canonical_{idx}.md"
        canonical_md.write_text("# recipe\n")
        entry = register_promoted_recipe(
            cache_root,
            fingerprint=fingerprint,
            fingerprint_payload=fingerprint_payload,
            dataset_id="demo",
            canonical_recipe_json=canonical_json,
            canonical_recipe_md=canonical_md,
            report_bundle_json=None,
            discovery_run_root=discovery_root,
        )
        return str(entry["fingerprint"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        fingerprints = sorted(pool.map(_register, [1, 2]))

    registry = load_registry(cache_root)
    assert sorted(registry["entries"].keys()) == fingerprints
