from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from PIL import Image
import pytest

from services.hosted.artifacts import ArtifactStore
from services.hosted.auth import AuthService
from services.hosted.errors import DomainError
from services.hosted.migration import HostedMigrator, inventory
from services.hosted.store import HostedStore
from services.hosted.types import Actor


def source(tmp_path):
    root = tmp_path / "legacy"
    for split, color in (("train", "red"), ("val", "blue")):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        Image.new("RGB", (40, 30), color).save(root / split / "images" / f"{color}.png")
        (root / split / "labels" / f"{color}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (root / "classes.txt").write_text("Vehicle\n")
    overlay = root / ".annotation_overlay"
    (overlay / "labels/train").mkdir(parents=True)
    (overlay / "labels/train/red.txt").write_text("0 0.25 0.5 0.2 0.4\n")
    (overlay / "captions").mkdir()
    (overlay / "captions/captions.jsonl").write_text("\n".join(json.dumps(record) for record in [
        {"id": "caption-one", "image_key": "train/red.png", "caption": "The current caption", "is_primary": True},
        {"id": "caption-old", "image_key": "train/red.png", "caption": "An earlier caption", "lifecycle_status": "superseded"}]) + "\n")
    (root / "groups.json").write_text('{"train/red.png":"flight-A","val/blue.png":"flight-B"}')
    (root / "recipes.json").write_text('{"components":[{"head_id":"legacy-head","encoder":"dinov3"}]}')
    weights = tmp_path / "legacy-head.pt"
    weights.write_bytes(b"immutable fixture custom head")
    mapping = {"files": {"groups.json": {"role": "groups"}, "recipes.json": {"role": "recipe"}},
               "image_metadata": {"train/red.png": {"video_group": "flight-A"}, "val/blue.png": {"video_group": "flight-B"}},
               "external_artifacts": [{"path": str(weights), "role": "salad_weights", "association": {"head_id": "legacy-head"}}]}
    return root, mapping


def authority(tmp_path):
    store = HostedStore(tmp_path / "state/hosted.sqlite3")
    auth = AuthService(store)
    row = auth.bootstrap("administrator", "a sufficiently long fixture password")
    return store, ArtifactStore(tmp_path / "state/artifacts"), Actor(row["id"], "admin", row["username"])


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}


def test_reviewed_baseline_preserves_effective_labels_metadata_and_resumes_after_commit(tmp_path):
    root, mapping = source(tmp_path)
    mapping['image_metadata']['train/red.png'].update(source_group='source-A', duplicate_group=0)
    before = hashes(root)
    plan = inventory(root, mapping=mapping)
    assert not plan["blockers"]
    store, artifacts, admin = authority(tmp_path)
    migrator = HostedMigrator(store, artifacts)
    with store.transaction() as con:
        con.execute("CREATE TRIGGER interrupt_migration BEFORE INSERT ON hosted_migration_items WHEN NEW.image_key='val/blue.png' BEGIN SELECT RAISE(ABORT,'lost migration progress after approval'); END")
    with pytest.raises(sqlite3.IntegrityError, match="lost migration progress"):
        migrator.apply(admin, plan, name="Legacy baseline", mode="approved-baseline", accept_existing_labels=True)
    with store.transaction() as con:
        con.execute("DROP TRIGGER interrupt_migration")
        assert con.execute("SELECT COUNT(*) FROM hosted_reviews WHERE action='approved'").fetchone()[0] == 2
    reopened = HostedStore(store.path)
    result = HostedMigrator(reopened, ArtifactStore(artifacts.root)).apply(admin, plan, name="Legacy baseline", mode="approved-baseline", accept_existing_labels=True)
    assert result["state"] == "complete" and result["image_count"] == 2
    assert HostedMigrator(reopened, artifacts).apply(admin, plan, name="Legacy baseline", mode="approved-baseline", accept_existing_labels=True) == result
    snapshot = reopened.snapshot(admin, result["dataset_id"])
    by_split = {row["split"]: row for row in snapshot["images"]}
    red = by_split["train"]["annotations"]
    assert red["objects"][0]["bbox"] == pytest.approx([6, 9, 8, 12])
    assert red["caption"] == "The current caption" and len(red["captions"]) == 2
    assert red["migration"]["metadata"]["video_group"] == "flight-A"
    assert by_split["val"]["annotations"]["migration"]["metadata"]["video_group"] == "flight-B"
    # Explicit migration groups must reach native analysis and native export,
    # rather than only surviving in the archived annotation JSON.
    from services.hosted.analysis import CanonicalAnalysis
    from services.class_analysis_patch_refinement import _capture_explicit_group_hint
    from services.dataset_pruning_planning import export_scene_groups
    analysis = CanonicalAnalysis(reopened, artifacts, tmp_path / 'analysis')
    job = analysis.capture(admin, result['dataset_id'], 'dqe', {}, 'migration-analysis')
    rows = analysis.dqe_request(admin, job['job_id'])['workspace_manifest']['images']
    migrated = next(row for row in rows if row['original_split'] == 'train')
    assert migrated['capture_metadata'] == {'video_group': 'flight-A', 'source_group': 'source-A', 'duplicate_group': 0}
    for field, value in migrated['capture_metadata'].items():
        assert migrated[field] == value
        assert _capture_explicit_group_hint({field: value}) == f'{field}:{value}'
        pair = [{field: value, 'image_id': str(i), 'image_sha256': str(i)} for i in range(2)]
        groups = export_scene_groups(pair, [0, 1], None)
        assert groups[0] == groups[1]
    preserved = snapshot["settings"]["migration"]["artifacts"]
    head = next(row for row in preserved if row["role"] == "salad_weights")
    assert head["association"]["head_id"] == "legacy-head"
    with artifacts.open(head["sha256"]) as stream:
        assert stream.read() == b"immutable fixture custom head"
    with reopened.read_connection() as con:
        assert con.execute("SELECT COUNT(*) FROM hosted_datasets").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM hosted_reviews WHERE action='approved'").fetchone()[0] == 2
    assert hashes(root) == before


def test_incoming_default_unknown_metadata_and_source_hash_guard(tmp_path):
    root, mapping = source(tmp_path)
    store, artifacts, admin = authority(tmp_path)
    migration = HostedMigrator(store, artifacts)
    unresolved = inventory(root)
    assert {row["path"] for row in unresolved["blockers"]} == {"groups.json", "recipes.json"}
    with pytest.raises(DomainError, match="migration_inventory_unresolved"):
        migration.apply(admin, unresolved, name="Unresolved")
    assert not store.list_datasets(admin)
    plan = inventory(root, mapping=mapping)
    with pytest.raises(DomainError, match="migration_baseline_attestation_required"):
        migration.apply(admin, plan, name="No implicit approvals", mode="approved-baseline")
    result = migration.apply(admin, plan, name="Incoming")
    assert not store.snapshot(admin, result["dataset_id"])["images"]
    assert len(store.list_images(admin, result["dataset_id"], state="incoming")) == 2
    (root / "train/labels/red.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    with pytest.raises(DomainError, match="migration_source_changed"):
        migration.apply(admin, plan, name="Incoming")


def test_inventory_rejects_symlinks_duplicate_pixels_and_runs_without_hosted_state(tmp_path):
    root, mapping = source(tmp_path)
    (root / "unknown-link").symlink_to(tmp_path / "legacy-head.pt")
    with pytest.raises(DomainError, match="migration_symlink_rejected"):
        inventory(root, mapping=mapping)
    (root / "unknown-link").unlink()
    (root / "val/images/blue.png").write_bytes((root / "train/images/red.png").read_bytes())
    plan = inventory(root, mapping=mapping)
    assert any(item["kind"] == "duplicate_media" for item in plan["blockers"])
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping))
    destination = tmp_path / "review/plan.json"
    result = subprocess.run([sys.executable, "tools/hosted_migrate.py", "inventory", "--source", str(root), "--mapping", str(mapping_path), "--output", str(destination)], check=True, text=True, capture_output=True)
    assert json.loads(result.stdout)["blockers"] == 1
    assert json.loads(destination.read_text())["plan_sha256"] == plan["plan_sha256"]
    assert destination.with_suffix(".rst").is_file()
    assert not list(tmp_path.rglob("hosted.sqlite3"))


def test_effective_caption_overlays_preserve_empty_clear_and_flat_record_keys(tmp_path):
    root = tmp_path / "flat"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "text_labels").mkdir()
    Image.new("RGB", (40, 30), "red").save(root / "images/car.png")
    (root / "labels/car.txt").write_text("0 .5 .5 .4 .4\n")
    (root / "classes.txt").write_text("Car\n")
    (root / "text_labels/car.txt").write_text("Old source caption")
    local = root / ".annotation_overlay/text_labels"
    local.mkdir(parents=True)
    (local / "car.txt").write_text("New local caption")
    external = tmp_path / "external-overlay"
    (external / "text_labels").mkdir(parents=True)
    (external / "text_labels/car.txt").write_text("")
    (external / "captions").mkdir()
    (external / "captions/captions.jsonl").write_text(json.dumps({"image_key": "train/car.png", "caption": "Alternate caption", "is_primary": True}) + "\n")
    plan = inventory(root, mapping={"overlay_root": str(external)})
    assert not plan["blockers"] and plan["images"][0]["effective_caption"] == ""
    assert {row["caption"] for row in plan["images"][0]["captions"]} == {"", "Old source caption", "New local caption", "Alternate caption"}
    store, artifacts, admin = authority(tmp_path)
    result = HostedMigrator(store, artifacts).apply(admin, plan, name="Caption preservation")
    image = store.list_images(admin, result["dataset_id"])[0]
    annotations = store.image(admin, image["id"])["annotations"]
    assert annotations["caption"] == "" and len(annotations["captions"]) == 4
    assert inventory(root)["images"][0]["effective_caption"] == "New local caption"
