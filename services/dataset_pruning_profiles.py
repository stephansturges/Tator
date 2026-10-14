"""Dataset deployment associations for the existing immutable SALAD registry.

Head bytes and historical training metadata remain owned by the existing
ingestion registry. This record owns display names, attachments and recipe
preferences only. One file lock/CAS/idempotent receipt settles each mutation.
"""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

from fastapi import HTTPException


def reference_file_state(path):
    path = Path(path).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file():
        raise HTTPException(409, "The referenced immutable head is unavailable")
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def validate_head_reference(reference):
    """A registry reference grants access to these exact bytes, never a directory."""
    path = Path(reference.get("path") or "")
    before = reference_file_state(path)
    if before != reference.get("state"):
        raise HTTPException(409, "The referenced immutable head changed; preserve it for inspection")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if reference_file_state(path) != before or digest.hexdigest() != reference.get("head_sha256"):
        raise HTTPException(409, "The referenced immutable head bytes changed")
    return path


def recipe_head_ids(recipe):
    """Only semantic head fields are references; labels can contain any text."""
    if isinstance(recipe, dict):
        for key, value in recipe.items():
            if key == "head_id" and isinstance(value, str) and value:
                yield value
            elif isinstance(value, (dict, list)):
                yield from recipe_head_ids(value)
    elif isinstance(recipe, list):
        for value in recipe:
            yield from recipe_head_ids(value)


def validate_preferred_recipe(recipe, role, dataset, heads, profiles):
    components = recipe.get("components")
    if not isinstance(components, list) or not components or len(components) > 12:
        raise HTTPException(400, "A preferred recipe needs 1–12 explicit components")
    bases = set()
    seen = set()
    total = 0.0
    for component in components:
        if not isinstance(component, dict):
            raise HTTPException(400, "Every recipe component must declare its encoder and weight")
        base = str(component.get("encoder_type") or component.get("encoder") or component.get("backbone") or component.get("id") or "")
        aliases = {"dinov3_pooled": "dinov3", "cradio_pooled": "cradio", "sam3_mask": "sam3"}
        base = aliases.get(base, base)
        selected = component.get("head_id")
        if selected:
            if selected not in dataset["head_ids"] or selected not in heads or component.get("head_sha256") != heads[selected]["head_sha256"]:
                raise HTTPException(409, "Preferred recipe references an unattached or changed head")
            head = heads[selected]
            if head.get("representation_scope", "whole_image") != role:
                raise HTTPException(400, "The SALAD head scope does not match this recipe")
            if profiles.get(selected, {}).get("archived", head.get("archived", False)):
                raise HTTPException(409, "An archived head cannot be selected for a preferred recipe")
            head_base = head.get("encoder_type", "dinov3")
            if base in {"salad", "local_salad"}:
                base = head_base
            if base != head_base or (component.get("model") or component.get("encoder_model")) not in {None, "", head.get("encoder_model")}:
                raise HTTPException(400, "The SALAD head backbone/model does not match its component")
        elif base in {"salad", "local_salad"}:
            raise HTTPException(400, "A SALAD component requires a persisted head")
        if base not in {"dinov3", "sam3", "cradio"} or (role == "whole_image" and base == "sam3"):
            raise HTTPException(400, "This encoder is unavailable for the selected representation")
        weight = component.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight <= 0:
            raise HTTPException(400, "Component weights must be finite and positive")
        identity = (base, selected or "", component.get("pooling", ""))
        if identity in seen:
            raise HTTPException(400, "A preferred recipe cannot repeat the same component")
        seen.add(identity)
        bases.add(base)
        total += weight
    if len(bases) > 3 or not math.isfinite(total):
        raise HTTPException(400, "A recipe supports up to three backbones with finite weights")


class DatasetProfiles:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / ".dataset_profiles.json"
        self.lock = threading.RLock()

    @contextmanager
    def transaction(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".dataset_profiles.lock"
        if lock_path.is_symlink() or self.path.is_symlink():
            raise HTTPException(409, "Profile management path changed; preserve it for inspection")
        with self.lock, lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self):
        if not self.path.exists():
            return {"schema": "dataset-salad-associations-v1", "revision": 0,
                    "profiles": {}, "datasets": {}, "operations": {}, "head_references": {}}
        try:
            value = json.loads(self.path.read_text())
            if value.get("schema") != "dataset-salad-associations-v1" or type(value.get("revision")) is not int or any(not isinstance(value.get(key), dict) for key in ("profiles", "datasets", "operations")):
                raise ValueError("Invalid management record")
            if not isinstance(value.setdefault("head_references", {}), dict):
                raise ValueError("Invalid immutable head references")
            return value
        except (ValueError, OSError, AttributeError) as error:
            raise HTTPException(503, "Saved profile management cannot be read; preserve it for repair") from error

    def snapshot(self, source_identity=""):
        with self.transaction():
            value = self.read()
        value.pop("operations", None)
        if source_identity:
            value["dataset"] = copy.deepcopy(value["datasets"].get(source_identity, {"head_ids": [], "preferred_recipes": {}}))
        return value

    def write(self, value):
        temp = self.path.with_name(".dataset-profiles-" + uuid.uuid4().hex + ".tmp")
        try:
            with temp.open("w") as handle:
                json.dump(value, handle, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
            descriptor = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temp.unlink(missing_ok=True)

    def register_reference(self, payload, metadata):
        """Internal qualified-artifact bridge; deployment attachment is a separate CAS."""
        token, head_id = str(payload.get("request_id") or ""), str(payload.get("head_id") or "")
        if not re.fullmatch(r"[a-f0-9]{32}", token) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", head_id):
            raise HTTPException(400, "A stable request and immutable head alias are required")
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        reference = copy.deepcopy(payload.get("artifact_reference") or {})
        if not isinstance(reference, dict) or not re.fullmatch(r"[a-f0-9]{64}", str(reference.get("head_sha256") or "")):
            raise HTTPException(400, "An exact immutable head artifact is required")
        with self.transaction():
            value = self.read()
            receipt = value["operations"].get(token)
            if receipt:
                if receipt["identity"] != identity:
                    raise HTTPException(409, "Request identity already belongs to another profile change")
                result = copy.deepcopy(value); result.pop("operations", None)
                return {"outcome": "committed", "receipt": receipt, "profiles": result}
            if payload.get("expected_revision") != value["revision"]:
                raise HTTPException(409, "Dataset profiles changed; refresh before registering this head")
            validate_head_reference(reference)
            if (self.root / (head_id + ".pt")).exists() or head_id in value["head_references"] or head_id in value["profiles"]:
                raise HTTPException(409, "Use a new immutable alias; existing heads are preserved")
            if metadata.get("representation_scope", "whole_image") not in {"whole_image", "bbox_crop"}:
                raise HTTPException(400, "The persisted head representation scope is unsupported")
            reference["historical_metadata"] = copy.deepcopy(metadata)
            reference["registered_at"] = time.time()
            value["head_references"][head_id] = reference
            value["profiles"][head_id] = {"head_sha256": reference["head_sha256"],
                "display_label": str(payload.get("display_label") or metadata.get("label") or head_id),
                "archived": False, "updated_at": time.time()}
            value["revision"] += 1
            receipt = {"request_id": token, "identity": identity, "revision": value["revision"],
                "action": "register_immutable_reference", "head_id": head_id, "at": time.time()}
            value["operations"][token] = receipt
            self.write(value)
            result = copy.deepcopy(value); result.pop("operations", None)
            return {"outcome": "committed", "receipt": receipt, "profiles": result}

    def mutate(self, payload, available_heads):
        token = str(payload.get("request_id") or "")
        source = str(payload.get("source_identity") or "")
        if not re.fullmatch(r"[a-f0-9]{32}", token) or not re.fullmatch(r"(?:dataset|transient):[^\s]{1,256}", source):
            raise HTTPException(400, "A stable request and dataset identity are required")
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.transaction():
            value = self.read()
            receipt = value["operations"].get(token)
            if receipt:
                if receipt["identity"] != identity:
                    raise HTTPException(409, "Request identity already belongs to another profile change")
                result = copy.deepcopy(value)
                result.pop("operations", None)
                return {"outcome": "committed", "receipt": receipt, "profiles": result}
            if payload.get("expected_revision") != value["revision"]:
                raise HTTPException(409, "Dataset profiles changed; refresh before applying this change")
            dataset = value["datasets"].setdefault(source, {"head_ids": [], "preferred_recipes": {}})
            action = payload.get("action")
            head_id = str(payload.get("head_id") or "")
            heads = {head["id"]: head for head in available_heads}
            if action in {"attach", "rename", "archive", "detach"}:
                if head_id not in heads:
                    raise HTTPException(400, "The selected head is unavailable")
                head = heads[head_id]
                profile = value["profiles"].setdefault(head_id, {"head_sha256": head["head_sha256"], "display_label": head.get("label") or head_id, "archived": False})
                if profile["head_sha256"] != head["head_sha256"]:
                    raise HTTPException(409, "Head bytes changed; use a new immutable profile version")
                if action == "attach":
                    dataset["head_ids"] = sorted(set(dataset["head_ids"]) | {head_id})
                elif action == "detach":
                    if any(head_id in set(recipe_head_ids(recipe)) for recipe in dataset["preferred_recipes"].values()):
                        raise HTTPException(409, "Remove this head from preferred recipes before detaching it")
                    dataset["head_ids"] = [item for item in dataset["head_ids"] if item != head_id]
                elif action == "rename":
                    label = str(payload.get("display_label") or "").strip()
                    if not label or len(label) > 160:
                        raise HTTPException(400, "Provide a profile display name of 1–160 characters")
                    profile["display_label"] = label
                elif action == "archive":
                    archived = bool(payload.get("archived", True))
                    if archived and any(head_id in set(recipe_head_ids(recipe)) for attached in value["datasets"].values() for recipe in attached["preferred_recipes"].values()):
                        raise HTTPException(409, "Remove this head from preferred recipes before archiving it")
                    profile["archived"] = archived
                profile["updated_at"] = time.time()
            elif action == "prefer_recipe":
                role = str(payload.get("role") or "")
                recipe = payload.get("recipe")
                if role not in {"whole_image", "bbox_crop"} or not isinstance(recipe, dict):
                    raise HTTPException(400, "A whole-image or bbox recipe is required")
                validate_preferred_recipe(recipe, role, dataset, heads, value["profiles"])
                dataset["preferred_recipes"][role] = copy.deepcopy(recipe)
            else:
                raise HTTPException(400, "Unknown profile-management action")
            value["revision"] += 1
            receipt = {"request_id": token, "identity": identity, "revision": value["revision"], "source_identity": source, "action": action, "at": time.time()}
            value["operations"][token] = receipt
            self.write(value)
            result = copy.deepcopy(value)
            result.pop("operations", None)
            return {"outcome": "committed", "receipt": receipt, "profiles": result}
