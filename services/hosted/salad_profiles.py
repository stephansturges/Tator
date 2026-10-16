"""Dataset-scoped SALAD deployment over retained, immutable trained heads.

The control process does not import Torch. Head qualification uses the original
local trainer's loader in a bounded CPU child; only retained artifacts can be
selected. Associations and exact operation receipts commit in one SQLite write.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import threading

from fastapi import APIRouter, Depends, Query

from services.dataset_pruning_profiles import DatasetProfiles, validate_preferred_recipe
from .artifacts import ArtifactError, ArtifactRef
from .errors import DomainError
from .store import _json


_HEAD_ID = re.compile(r"[A-Za-z0-9_-]{1,160}\Z")
_MAX_HEAD_BYTES = 512 * 1024 * 1024


class _TransactionProfiles(DatasetProfiles):
    """Run the unchanged native mutation rules inside the caller's transaction."""
    def __init__(self, value):
        self.value = copy.deepcopy(value)

    def transaction(self):
        return nullcontext()

    def read(self):
        return copy.deepcopy(self.value)

    def write(self, value):
        self.value = copy.deepcopy(value)


def _inspect_child(path):
    # This entry point alone imports model code. Old unsafe-pickle Torch loaders
    # are rejected before invoking the existing compatibility loader.
    import inspect
    import torch
    from utils.local_salad import load_local_salad_head_file
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise ValueError("A weights-only Torch loader is required")
    torch.set_num_threads(1)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    for key in ("num_channels", "num_clusters", "cluster_dim", "token_dim", "hidden_dim"):
        if type(config.get(key)) is not int or not 1 <= config[key] <= 16384:
            raise ValueError("Unsupported head dimensions")
    estimated_parameters = config["hidden_dim"] * (3 * config["num_channels"] + config["num_clusters"] + config["cluster_dim"] + config["token_dim"])
    if estimated_parameters > 100_000_000:
        raise ValueError("Head architecture exceeds inspection limits")
    del payload
    head, metadata = load_local_salad_head_file(Path(path), device_name="cpu")
    # Keep the historical metadata in the artifact itself; the UI receives only
    # the actual trained contract, never old absolute paths or source credentials.
    fields = ("label", "policy", "trainer", "representation_scope", "encoder_type", "encoder_model",
              "cradio_pooling", "salad_backend", "train_image_count", "source_mode", "augmentation_profile",
              "train_view_max_side", "epochs", "batch_size", "learning_rate", "weight_decay", "temperature",
              "seed", "training_signature", "training_rng_receipt", "config")
    result = {key: metadata[key] for key in fields if key in metadata}
    result.setdefault("representation_scope", "whole_image")
    result.setdefault("encoder_type", "dinov3")
    if result["representation_scope"] not in {"whole_image", "bbox_crop"} or result["encoder_type"] not in {"dinov3", "cradio", "sam3"}:
        raise ValueError("Unsupported trained representation")
    if result["representation_scope"] == "whole_image" and result["encoder_type"] == "sam3":
        raise ValueError("SAM3 is a bbox representation")
    if not isinstance(result.get("encoder_model"), str) or not result["encoder_model"]:
        raise ValueError("The trained encoder model identity is required")
    result["descriptor_dim"] = head.config.token_dim + head.config.num_clusters * head.config.cluster_dim
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    if len(encoded.encode()) > 128 * 1024:
        raise ValueError("Head metadata is too large")
    print(encoded)


class HostedSaladProfiles:
    def __init__(self, store, artifacts):
        self.store, self.artifacts = store, artifacts
        self._inspections = {}
        self._inspection_lock = threading.Lock()
        with store.transaction() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS hosted_salad_heads (
                head_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                metadata_json TEXT NOT NULL, created_at REAL NOT NULL)""")
            for verb in ("UPDATE", "DELETE"):
                con.execute(f"CREATE TRIGGER IF NOT EXISTS hosted_salad_heads_immutable_{verb.lower()} "
                            f"BEFORE {verb} ON hosted_salad_heads BEGIN SELECT RAISE(ABORT,'immutable SALAD head'); END")

    @staticmethod
    def _settings(con, dataset_id):
        row = con.execute("SELECT settings_json FROM hosted_dataset_settings WHERE dataset_id=?", (dataset_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    @staticmethod
    def _heads(con):
        return [{**json.loads(row["metadata_json"]), "id": row["head_id"], "head_sha256": row["sha256"],
                 "size_bytes": row["size_bytes"], "status": "available", "updated_at": row["created_at"]}
                for row in con.execute("SELECT * FROM hosted_salad_heads ORDER BY head_id")]

    def _snapshot(self, con, dataset_id):
        dataset = self.store._dataset(con, dataset_id)
        saved = self._settings(con, dataset_id).get("salad_profiles", {})
        attached = {"head_ids": copy.deepcopy(saved.get("head_ids", [])),
                    "preferred_recipes": copy.deepcopy(saved.get("preferred_recipes", {}))}
        profiles = copy.deepcopy(saved.get("profiles", {}))
        heads = [{**head, **{key: value for key, value in profiles.get(head["id"], {}).items()
                           if key in {"display_label", "archived"}}} for head in self._heads(con)]
        return {"schema": "dataset-salad-associations-v1", "revision": dataset["revision"],
                "source_identity": "dataset:" + dataset_id, "profiles": profiles, "dataset": attached,
                "datasets": {"dataset:" + dataset_id: attached}, "heads": heads}

    def snapshot(self, actor, dataset_id):
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            return self._snapshot(con, dataset_id)

    def _save(self, con, actor, dataset_id, snapshot):
        settings = self._settings(con, dataset_id)
        settings["salad_profiles"] = {**snapshot["datasets"]["dataset:" + dataset_id], "profiles": snapshot["profiles"]}
        encoded = _json(settings)
        if len(encoded.encode()) > 1024 * 1024:
            raise DomainError("dataset_settings_too_large", 413)
        revision = snapshot["revision"]
        con.execute("INSERT INTO hosted_dataset_settings VALUES (?,?,?) ON CONFLICT(dataset_id) DO UPDATE SET version=excluded.version,settings_json=excluded.settings_json", (dataset_id, revision, encoded))
        con.execute("INSERT INTO hosted_dataset_setting_versions VALUES (?,?,?,?,?)", (dataset_id, revision, encoded, actor.user_id, self.store.clock()))
        con.execute("UPDATE hosted_datasets SET revision=? WHERE id=?", (revision, dataset_id))
        self.store._event(con, actor, dataset_id, None, "dataset_salad_profiles_changed", {"settings_version": revision})

    def mutate(self, actor, dataset_id, payload):
        payload = copy.deepcopy(payload)
        operation_id = payload.get("operation_id", payload.get("request_id", ""))
        if payload.get("source_identity", "dataset:" + dataset_id) != "dataset:" + dataset_id:
            raise DomainError("salad_dataset_mismatch", 409)
        def run(con):
            before = self._snapshot(con, dataset_id)
            if type(payload.get("expected_revision")) is not int:
                raise DomainError("salad_revision_required", 422)
            native = _TransactionProfiles({**before, "operations": {}, "head_references": {}})
            request = {**payload, "request_id": hashlib.sha256(operation_id.encode()).hexdigest()[:32],
                       "source_identity": "dataset:" + dataset_id}
            result = native.mutate(request, before["heads"])
            self._save(con, actor, dataset_id, result["profiles"])
            return {"outcome": "committed", "receipt": {**result["receipt"], "operation_id": operation_id},
                    "profiles": self._snapshot(con, dataset_id)}
        return self.store._command(actor, operation_id, "salad_profiles", {"dataset_id": dataset_id, "payload": payload}, run, admin=True)

    def _sources(self, con, dataset_id):
        self.store._dataset(con, dataset_id)
        rows = []
        if con.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_analysis_artifacts'").fetchone():
            for row in con.execute("SELECT a.*,j.consumer FROM hosted_analysis_artifacts a JOIN hosted_analysis_jobs j ON j.id=a.job_id JOIN hosted_snapshots s ON s.id=j.snapshot_id WHERE s.dataset_id=? AND j.state='complete' AND a.relative_path LIKE '%.pt' ORDER BY j.created_at DESC,a.relative_path", (dataset_id,)):
                # Only the native head namespace is a SALAD registration source;
                # arbitrary detector/checkpoint tensors remain separate assets.
                if "salad_heads" in PurePosixPath(row["relative_path"]).parts:
                    rows.append({"origin": "analysis", "owner_id": row["job_id"], "name": row["relative_path"],
                                 "sha256": row["sha256"], "size_bytes": row["size_bytes"]})
        for item in self._settings(con, dataset_id).get("migration", {}).get("artifacts", []):
            if item.get("role") == "salad_weights":
                known = con.execute("SELECT 1 FROM hosted_artifact_references WHERE kind='migration_salad_weights' AND owner_id=? AND sha256=? AND size_bytes=?", (dataset_id, item["sha256"], item["size_bytes"])).fetchone()
                if known:
                    rows.append({"origin": "migration", "owner_id": dataset_id, "name": item["source_file"],
                                 "sha256": item["sha256"], "size_bytes": item["size_bytes"]})
        for row in rows:
            row["source_id"] = hashlib.sha256(_json(row).encode()).hexdigest()
        return rows

    def _inspect(self, source):
        reference = ArtifactRef(source["sha256"], source["size_bytes"])
        if not 0 < reference.size_bytes <= _MAX_HEAD_BYTES:
            raise DomainError("salad_head_size_invalid", 422)
        # Serialize cache misses so discovery/registration cannot spawn an
        # unbounded inspection herd. Hash verification remains mandatory.
        with self._inspection_lock:
            with self.artifacts.exclusive():
                try:
                    self.artifacts.verify(reference)
                except (ArtifactError, OSError) as error:
                    raise DomainError("salad_head_unavailable", 409) from error
                cached = self._inspections.get(reference.sha256)
                if cached is not None:
                    return copy.deepcopy(cached)
            with tempfile.TemporaryDirectory(prefix="salad-inspection-", dir=self.artifacts.staging) as directory:
                with self.artifacts.exclusive():
                    self.artifacts.pin(reference, Path(directory))
                environment = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                               "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
                try:
                    result = subprocess.run([sys.executable, "-m", "services.hosted.salad_profiles", "--inspect", str(Path(directory) / reference.sha256)],
                                            env=environment, capture_output=True, timeout=90, check=True)
                    metadata = json.loads(result.stdout)
                except (subprocess.SubprocessError, ValueError) as error:
                    raise DomainError("salad_head_invalid", 422, "The saved head failed its original trained-format validation.") from error
            if len(self._inspections) >= 128:
                self._inspections.pop(next(iter(self._inspections)))
            self._inspections[reference.sha256] = metadata
            return copy.deepcopy(metadata)

    def discover(self, actor, dataset_id, *, limit=25, offset=0):
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            sources = self._sources(con, dataset_id)
        limit, offset = min(100, max(1, int(limit))), max(0, int(offset))
        choices = []
        for source in sources[offset:offset + limit]:
            try:
                metadata = self._inspect(source)
                choices.append({**source, "status": "available", "metadata": metadata,
                                "label": metadata.get("label") or PurePosixPath(source["name"]).stem,
                                "encoder_type": metadata["encoder_type"], "encoder_model": metadata["encoder_model"],
                                "representation_scope": metadata["representation_scope"]})
            except DomainError as error:
                choices.append({**source, "status": "invalid", "error": error.as_dict()})
        # Inspection can take time: logout/role changes must still win response admission.
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            self.store._dataset(con, dataset_id)
        return {"sources": choices, "offset": offset, "total": len(sources),
                "next_offset": offset + limit if offset + limit < len(sources) else None}

    def register(self, actor, dataset_id, payload):
        payload = copy.deepcopy(payload)
        operation_id = payload.get("operation_id", "")
        command = {"dataset_id": dataset_id, "payload": payload}
        # Exact successful retries do not require a still-current source listing.
        if self.store.raw_receipt(actor, operation_id):
            return self.store._command(actor, operation_id, "register_salad_head", command,
                                       lambda con: (_ for _ in ()).throw(DomainError("salad_receipt_missing", 409)), admin=True)
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            source = next((item for item in self._sources(con, dataset_id) if item["source_id"] == payload.get("source_id")), None)
        if source is None:
            raise DomainError("salad_source_unavailable", 404)
        metadata = self._inspect(source)
        head_id = payload.get("head_id") or "salad_" + source["sha256"][:24]
        label = str(payload.get("display_label") or metadata.get("label") or head_id).strip()
        if not isinstance(head_id, str) or not _HEAD_ID.fullmatch(head_id) or not label or len(label) > 160:
            raise DomainError("salad_profile_identity_invalid", 422)
        reference = ArtifactRef(source["sha256"], source["size_bytes"])
        with self.artifacts.exclusive():
            self.artifacts.verify(reference)
            def run(con):
                before = self._snapshot(con, dataset_id)
                if type(payload.get("expected_revision")) is not int or before["revision"] != payload["expected_revision"]:
                    raise DomainError("dataset_revision_changed", 409)
                if not any(item["source_id"] == source["source_id"] for item in self._sources(con, dataset_id)):
                    raise DomainError("salad_source_unavailable", 409)
                existing = con.execute("SELECT sha256,size_bytes FROM hosted_salad_heads WHERE head_id=?", (head_id,)).fetchone()
                if existing and (existing[0], existing[1]) != (reference.sha256, reference.size_bytes):
                    raise DomainError("salad_head_alias_exists", 409)
                con.execute("INSERT OR IGNORE INTO hosted_salad_heads VALUES (?,?,?,?,?)", (head_id, reference.sha256, reference.size_bytes, _json(metadata), self.store.clock()))
                con.execute("INSERT OR IGNORE INTO hosted_artifact_references VALUES ('salad_head',?,?,?,?)", (head_id, reference.sha256, reference.size_bytes, self.store.clock()))
                con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (reference.sha256,))
                before["profiles"][head_id] = {"head_sha256": reference.sha256, "display_label": label, "archived": False}
                before["dataset"]["head_ids"] = sorted(set(before["dataset"]["head_ids"]) | {head_id})
                before["revision"] += 1
                self._save(con, actor, dataset_id, before)
                return {"outcome": "committed", "receipt": {"operation_id": operation_id, "revision": before["revision"], "head_id": head_id, "action": "register_and_attach"},
                        "profiles": self._snapshot(con, dataset_id)}
            return self.store._command(actor, operation_id, "register_salad_head", command, run, admin=True)

    def capture(self, actor, snapshot_id, pin_directory):
        frozen = self.store.get_snapshot(actor, snapshot_id)["manifest"]
        pin_directory = Path(pin_directory).absolute()
        if any(path.is_symlink() for path in (pin_directory, *pin_directory.parents)):
            raise DomainError("salad_pin_directory_invalid", 422)
        pin_directory.mkdir(parents=True, exist_ok=True)
        saved = frozen.get("settings", {}).get("salad_profiles", {})
        preferences = copy.deepcopy(saved.get("preferred_recipes", {}))
        attached = {"head_ids": saved.get("head_ids", []), "preferred_recipes": preferences}
        with self.artifacts.exclusive(), self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            heads = {head["id"]: head for head in self._heads(con)}
            selected = []
            for head_id in attached["head_ids"]:
                head = heads.get(head_id)
                profile = saved.get("profiles", {}).get(head_id, {})
                if not head or profile.get("head_sha256") != head["head_sha256"]:
                    raise DomainError("salad_snapshot_head_changed", 409)
                if not profile.get("archived", False):
                    selected.append(head)
            for role, recipe in preferences.items():
                if role not in {"whole_image", "bbox_crop"}:
                    raise DomainError("salad_snapshot_recipe_invalid", 409)
                validate_preferred_recipe(recipe, role, attached, heads, saved.get("profiles", {}))
            inputs = []
            for head in selected:
                ref = self.artifacts.pin(ArtifactRef(head["head_sha256"], head["size_bytes"]), Path(pin_directory))
                inputs.append({"namespace": "salad_heads", "relative_path": head["id"] + ".pt",
                               "path": str(Path(pin_directory) / ref.sha256), **ref.as_dict()})
        defaults = {}
        if "whole_image" in preferences:
            defaults["scene_components"] = preferences["whole_image"]["components"]
        if "bbox_crop" in preferences:
            recipe = preferences["bbox_crop"]
            defaults.update(object_recipe=recipe.get("object_recipe", "inherit_dqe"), object_components=recipe["components"], object_adjustment=recipe.get("adjustment", "inherit"))
            if recipe.get("preprocessing"):
                defaults["object_preprocessing"] = recipe["preprocessing"]
        return {"snapshot_id": snapshot_id, "cache_inputs": inputs, "preferred_recipes": preferences,
                "pruning_defaults": defaults, "heads": selected}


def install_salad_profiles_api(app, store, artifacts, admin_dependency):
    service = getattr(app.state, "hosted_salad_profiles", None)
    if service is not None:
        return service
    service = HostedSaladProfiles(store, artifacts)
    app.state.hosted_salad_profiles = service
    router = APIRouter(prefix="/api/hosted/datasets/{dataset_id}/salad-profiles", tags=["hosted SALAD profiles"])

    @router.get("")
    def snapshot(dataset_id: str, identity=Depends(admin_dependency)):
        return service.snapshot(identity, dataset_id)

    @router.get("/sources")
    def sources(dataset_id: str, limit: int = Query(25, ge=1, le=100), offset: int = Query(0, ge=0), identity=Depends(admin_dependency)):
        return service.discover(identity, dataset_id, limit=limit, offset=offset)

    @router.post("")
    def mutate(dataset_id: str, body: dict, identity=Depends(admin_dependency)):
        return service.mutate(identity, dataset_id, body)

    @router.post("/register")
    def register(dataset_id: str, body: dict, identity=Depends(admin_dependency)):
        return service.register(identity, dataset_id, body)

    app.include_router(router)
    return service


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--inspect":
        raise SystemExit(2)
    _inspect_child(sys.argv[2])
