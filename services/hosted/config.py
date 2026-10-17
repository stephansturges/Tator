"""Trusted model definitions for hosted workers; no application/model imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import threading
from typing import Mapping

from services.hosted.runtime import AdapterSpec

RFDETR_CLASSES = {
    "rfdetr-nano": "RFDETRNano", "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium", "rfdetr-base": "RFDETRBase",
    "rfdetr-large": "RFDETRLarge",
}


def model_uses_manifest(definition):
    return definition.get("kind") in {"qwen", "edr", "sam3_recipe"} or (
        definition.get("kind") == "sam3" and definition.get("backend") == "mlx")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path, roots) -> Path:
    candidate = Path(path).expanduser().absolute()
    resolved = candidate.resolve(strict=True)
    roots = [Path(root).resolve(strict=True) for root in roots]
    root = next((root for root in roots if resolved.is_relative_to(root)), None)
    if root is None or not resolved.is_file():
        raise ValueError("hosted_model_path_invalid")
    # Resolve platform aliases such as /var -> /private/var, but do not accept
    # a symlink anywhere inside a trusted artifact root.
    for item in (candidate, *candidate.parents):
        if item.resolve(strict=True) == root:
            break
        if item.is_symlink():
            raise ValueError("hosted_model_path_invalid")
    return resolved


def verify_checkpoint(definition):
    path = Path(definition["checkpoint"])
    if path.is_symlink() or not path.is_file() or file_digest(path) != definition["checkpoint_sha256"]:
        raise ValueError("hosted_model_artifact_changed")
    if definition.get("bpe_path") and file_digest(Path(definition["bpe_path"])) != definition.get("bpe_sha256"):
        raise ValueError("hosted_model_artifact_changed")
    if definition.get("metadata_path") and file_digest(Path(definition["metadata_path"])) != definition.get("metadata_sha256"):
        raise ValueError("hosted_model_artifact_changed")


def _native_device(device):
    if not isinstance(device, str) or not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", device):
        raise ValueError("hosted_explicit_device_required")
    return device


def validate_model_id(model_id):
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", model_id):
        raise ValueError("hosted_model_id_invalid")
    return model_id


def validate_model_member(name):
    if not isinstance(name, str) or not name or len(name) > 1024 or "\\" in name or "\x00" in name or PurePosixPath(name).is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError("hosted_model_member_invalid")
    return name


class ModelRegistry:
    """Validated in-memory view of admin-owned canonical model settings.

    The caller persists dataset/user selections in its canonical settings store.
    This registry deliberately does not create a competing model database.
    """
    def __init__(self, *, allowed_roots, definitions: Mapping | None = None):
        self.roots = tuple(Path(root).resolve(strict=True) for root in allowed_roots)
        if not self.roots or any(not root.is_dir() for root in self.roots):
            raise ValueError("hosted_model_roots_required")
        self._definitions = {}
        self._lock = threading.RLock()
        for model_id, definition in (definitions or {}).items():
            self.register(model_id, definition)

    def register(self, model_id, definition):
        validate_model_id(model_id)
        value = json.loads(json.dumps(dict(definition), allow_nan=False))
        kind = value.get("kind")
        if kind not in {"sam1", "sam3", "yolo", "rfdetr", "classifier", "qwen", "edr", "sam3_recipe"}:
            raise ValueError("hosted_model_kind_unsupported")
        allowed = {"model_id", "kind", "backend", "checkpoint", "checkpoint_sha256", "model_type", "variant", "labels", "bpe_path", "bpe_sha256", "enable_segmentation", "class_ids", "metadata_path", "metadata_sha256", "model_path", "files", "adapter_path", "adapter_files", "recipe_file", "ensemble_model_file", "ensemble_meta_file", "feature_contract_file", "components", "component_identities"}
        if set(value) - allowed:
            raise ValueError("hosted_model_settings_unknown")
        if value.get("model_id", model_id) != model_id:
            raise ValueError("hosted_model_id_invalid")
        if "backend" in value and (kind != "sam3" or value["backend"] != "mlx"):
            raise ValueError("hosted_model_backend_invalid")
        if model_uses_manifest(value):
            package_fields = {"model_id", "kind", "checkpoint_sha256", "model_path", "files"}
            if kind == "sam3":
                package_fields.add("backend")
            else:
                package_fields |= {"variant", "adapter_path", "adapter_files"} if kind == "qwen" else {"recipe_file", "ensemble_model_file", "ensemble_meta_file", "feature_contract_file", "components", "component_identities", "labels"}
            if set(value) - package_fields:
                raise ValueError("hosted_model_settings_unknown")
            for path_field, files_field in (("model_path", "files"), ("adapter_path", "adapter_files")):
                if path_field == "adapter_path" and path_field not in value and files_field not in value:
                    continue
                directory = Path(value[path_field]).expanduser().absolute()
                manifest = value[files_field]
                if not isinstance(manifest, dict) or not manifest or len(manifest) > 10000:
                    raise ValueError("hosted_model_manifest_invalid")
                for name, expected in manifest.items():
                    validate_model_member(name)
                    if any(str(parent) in manifest for parent in PurePosixPath(name).parents if str(parent) != "."):
                        raise ValueError("hosted_model_manifest_invalid")
                    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                        raise ValueError("hosted_model_manifest_invalid")
                    path = regular_file(directory / name, self.roots)
                    if not path.is_relative_to(directory.resolve(strict=True)) or file_digest(path) != expected:
                        raise ValueError("hosted_model_artifact_changed")
                value[path_field] = str(directory.resolve(strict=True))
            if kind == "sam3":
                required = {"config.json", "model.safetensors", "processor_config.json", "tokenizer.json", "tokenizer_config.json"}
                if not required.issubset(value["files"]):
                    raise ValueError("hosted_sam3_mlx_package_incomplete")
                config = json.loads((Path(value["model_path"]) / "config.json").read_text())
                # The geometric adapter uses the interactive tracker as well
                # as the vision backbone; a detector-only package cannot run it.
                if (config.get("model_type") != "sam3_video"
                        or config.get("detector_config", {}).get("model_type") != "sam3"
                        or config.get("tracker_config", {}).get("model_type") != "sam3_tracker_video"):
                    raise ValueError("hosted_sam3_mlx_configuration_invalid")
            elif kind == "qwen":
                if "config.json" not in value["files"]:
                    raise ValueError("hosted_qwen_configuration_required")
                if value.get("variant", "Instruct") not in {"Instruct", "Thinking"}:
                    raise ValueError("hosted_qwen_variant_invalid")
                value.setdefault("variant", "Instruct")
            else:
                value.setdefault("recipe_file", "recipe.json")
                for field in ("recipe_file", "ensemble_model_file", "ensemble_meta_file", "feature_contract_file"):
                    if field in value and validate_model_member(value[field]) not in value["files"]:
                        raise ValueError("hosted_edr_recipe_artifact_required")
                components = value.get("components")
                permitted = {"sam3", "extra_classifier"} if kind == "sam3_recipe" else {"yolo", "rfdetr", "sam3", "classifier", "qwen", "qwen_refinement", "qwen_fallback"}
                if not isinstance(components, dict) or not components or set(components) - permitted or kind == "sam3_recipe" and "sam3" not in components:
                    raise ValueError("hosted_edr_components_invalid")
                for component in components.values():
                    validate_model_id(component)
                identities = value.get("component_identities")
                if identities is not None and (not isinstance(identities, dict) or set(identities) != set(components) or any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in identities.values())):
                    raise ValueError("hosted_edr_component_identity_invalid")
                labels = value.setdefault("labels", []) if kind == "sam3_recipe" else value.get("labels")
                if not isinstance(labels, list) or (kind != "sam3_recipe" and not labels) or any(not isinstance(label, str) or not label.strip() for label in labels) or len({label.casefold() for label in labels}) != len(labels):
                    raise ValueError("hosted_detector_labels_required")
            value["model_id"] = model_id
            with self._lock:
                self._definitions[model_id] = value
            return
        checkpoint = regular_file(value["checkpoint"], self.roots)
        digest = file_digest(checkpoint)
        if value.get("checkpoint_sha256") and value["checkpoint_sha256"] != digest:
            raise ValueError("hosted_model_artifact_changed")
        value.update(model_id=model_id, checkpoint=str(checkpoint), checkpoint_sha256=digest)
        if kind == "sam1":
            value.setdefault("model_type", "vit_h")
            if value["model_type"] not in {"vit_h", "vit_l", "vit_b"}:
                raise ValueError("hosted_sam_model_type_invalid")
        elif kind == "sam3":
            value["bpe_path"] = str(regular_file(value["bpe_path"], self.roots))
            bpe_digest = file_digest(Path(value["bpe_path"]))
            if value.get("bpe_sha256", bpe_digest) != bpe_digest:
                raise ValueError("hosted_model_artifact_changed")
            value["bpe_sha256"] = bpe_digest
            value.setdefault("enable_segmentation", True)
            if type(value["enable_segmentation"]) is not bool:
                raise ValueError("hosted_sam_segmentation_flag_invalid")
        elif kind == "rfdetr":
            if value.get("variant") not in RFDETR_CLASSES:
                raise ValueError("hosted_rfdetr_variant_unsupported")
        elif kind == "classifier":
            metadata = regular_file(value["metadata_path"], self.roots)
            metadata_digest = file_digest(metadata)
            if value.get("metadata_sha256", metadata_digest) != metadata_digest:
                raise ValueError("hosted_model_artifact_changed")
            value.update(metadata_path=str(metadata), metadata_sha256=metadata_digest)
            mapping = value.get("class_ids")
            if mapping is not None and (not isinstance(mapping, dict) or any(not isinstance(key, str) or not isinstance(item, str) or not key or not item for key, item in mapping.items())):
                raise ValueError("hosted_classifier_class_ids_invalid")
        if kind in {"yolo", "rfdetr", "classifier"}:
            labels = value.get("labels")
            if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label.strip() for label in labels) or len({label.casefold() for label in labels}) != len(labels):
                raise ValueError("hosted_detector_labels_required")
        if kind in {"yolo", "rfdetr"}:
            class_ids = value.get("class_ids")
            if kind == "rfdetr" and class_ids is None:
                raise ValueError("hosted_rfdetr_native_class_ids_required")
            if class_ids is not None and kind != "rfdetr":
                raise ValueError("hosted_detector_class_ids_invalid")
            if class_ids is not None and (not isinstance(class_ids, list) or len(class_ids) != len(labels) or len(set(class_ids)) != len(class_ids) or any(type(item) is not int or item < 0 for item in class_ids)):
                raise ValueError("hosted_detector_class_ids_invalid")
        with self._lock:
            self._definitions[model_id] = value

    def definition(self, model_id):
        with self._lock:
            if model_id not in self._definitions:
                raise ValueError("hosted_model_unknown")
            return json.loads(json.dumps(self._definitions[model_id]))

    def model_identity(self, model_id):
        definition = self.definition(model_id)
        for field in ("checkpoint", "bpe_path", "metadata_path", "model_path", "adapter_path"):
            definition.pop(field, None)
        return hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def catalog(self):
        with self._lock:
            return json.loads(json.dumps([{key: value[key] for key in ("model_id", "kind", "backend", "checkpoint_sha256", "variant", "labels") if key in value}
                    for value in self._definitions.values()]))

    def user_spec(self, user_id, model_id, *, media_root, device, capacity=3, classifier_id=None):
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("hosted_runtime_owner_required")
        model = self.definition(model_id)
        if model["kind"] not in {"sam1", "sam3"}:
            raise ValueError("hosted_user_model_kind_invalid")
        if type(capacity) is not int or not 1 <= capacity <= 3:
            raise ValueError("hosted_predictor_capacity_invalid")
        config = {
            "actor_id": user_id, "media_root": str(Path(media_root).resolve(strict=True)),
            "model": model, "capacity": capacity,
        }
        if classifier_id is not None:
            classifier = self.definition(classifier_id)
            if classifier["kind"] != "classifier":
                raise ValueError("hosted_classifier_kind_invalid")
            config["classifier"] = classifier
        return AdapterSpec("services.hosted.adapters:UserToolAdapter", config, _native_device(device))

    def dataset_spec(self, dataset_id, model_id, *, media_root, device):
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("hosted_runtime_owner_required")
        model = self.definition(model_id)
        if model["kind"] not in {"yolo", "rfdetr"}:
            raise ValueError("hosted_dataset_model_kind_invalid")
        return AdapterSpec("services.hosted.adapters:DatasetDetectorAdapter", {
            "dataset_id": dataset_id, "media_root": str(Path(media_root).resolve(strict=True)),
            "model": model,
        }, _native_device(device))
