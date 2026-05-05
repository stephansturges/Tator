from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from services.calibration_recipe_registry import (
    build_recipe_fingerprint,
    build_recipe_fingerprint_payload,
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
