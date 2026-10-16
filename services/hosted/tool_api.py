"""Authenticated hosted assistance boundary and trusted model selection."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import uuid
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from services.hosted.config import ModelRegistry, file_digest, validate_model_id, validate_model_member
from services.hosted.analysis import CanonicalAnalysis
from services.hosted.job_authority import HeavyJobAuthority
from services.hosted.media import HostedMedia
from services.hosted.caption_jobs import CaptionJobRequest, CaptionJobCommand
from services.hosted.caption_batch import CaptionBatchRequest, CaptionProposalApply
from services.hosted.errors import DomainError
from services.hosted.operations import HostedOperations
from services.hosted.runtime import AdapterSpec, RequestContext, RuntimeRejected, RuntimeSupervisor, TERMINAL
from services.hosted.types import Actor, MutationContext
from services.hosted.paid_batches import PaidBatchRequest, PaidBatchCommand, PaidBatchResume, PaidBatchApply, PaidBatchArchive, RecoverBatch
from services.hosted.instruction_review import InstructionReviews, InstructionReviewApply

SAM_OPERATIONS = {"sam_state", "sam_preload", "sam_point", "sam_point_multi", "sam_bbox", "sam_point_auto", "sam_point_multi_auto", "sam_bbox_auto", "classify_bbox", "sam3_text", "sam3_text_auto", "sam3_visual"}
READ_ONLY_OPERATIONS = {"sam_state", "sam_preload"}
SAM_RECIPE_OPERATIONS = {"sam3_recipe_chain"}
CLASSIFIER_OPERATIONS = {"sam_point_auto", "sam_point_multi_auto", "sam_bbox_auto", "classify_bbox", "sam3_text_auto"}
DETECTOR_OPERATIONS = {"detect_full", "detect_region", "detect_windowed"}
EDR_OPERATIONS = {"edr_prepass"}
QWEN_OPERATIONS = {"qwen_infer", "qwen_caption", "qwen_caption_preview", "qwen_caption_case"}
NATIVE_ANALYSIS_FACTORY = "services.hosted.native_analysis:NativeAnalysisAdapter"
PARAMETERS = {"point_x", "point_y", "positive_points", "negative_points", "bbox_left", "bbox_top", "bbox_width", "bbox_height",
              "local_refine", "uuid", "slot", "conf", "iou", "max_det", "slice_size", "overlap", "merge_iou",
              "text_prompt", "prompt", "threshold", "mask_threshold", "min_size", "max_results", "windowed", "window_size", "window_overlap", "simplify_epsilon", "bboxes", "bbox_labels"}


def _parameters_for(operation):
    if operation == "sam_state":
        return set()
    if operation in SAM_RECIPE_OPERATIONS:
        return {"steps", "dedupe", "mask_threshold", "min_size", "simplify_epsilon", "max_results"}
    if operation == "qwen_caption_case":
        return {"caption_request", "settings"}
    if operation in EDR_OPERATIONS:
        return {"recipe_id"}
    if operation in {"qwen_caption", "qwen_caption_preview"}:
        from models.schemas import QwenCaptionRequest
        return set(QwenCaptionRequest.model_fields) - {"image_base64", "image_token", "image_name", "image_width", "image_height", "unload_others", "force_unload"}
    if operation == "qwen_infer":
        return {"prompt", "item_list", "image_type", "extra_context", "max_results", "prompt_type"}
    return PARAMETERS


def qwen_guidance(settings, operation, parameters):
    """Freeze dataset defaults once; explicit values (even empty) take precedence."""
    result = dict(parameters)
    if operation == "qwen_infer":
        result.setdefault("extra_context", settings.get("context"))
    elif operation in {"qwen_caption", "qwen_caption_preview", "qwen_caption_case"}:
        if operation == "qwen_caption_case":
            result["caption_request"] = qwen_guidance(settings, "qwen_caption", result.get("caption_request", {}))
        else:
            result.setdefault("user_prompt", settings.get("context"))
            result.setdefault("labelmap_glossary", settings.get("labelmap_glossary"))
    return result


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    image_id: str
    expected_version: str
    generation: int = Field(ge=0)
    editor_id: str = Field(min_length=8, max_length=128)
    operation: str = Field(min_length=1, max_length=160)
    parameters: dict = Field(default_factory=dict)
    workspace_id: str = Field(default="", max_length=128)


class ModelSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    expected_revision: int = Field(ge=0)
    operation_id: str = Field(min_length=8, max_length=128)


class DatasetModelSelection(ModelSelection):
    class_mapping: dict[str, str] | None = None


class DatasetRecipeSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_ids: list[str] = Field(max_length=32)
    expected_revision: int = Field(ge=0)
    operation_id: str = Field(min_length=8, max_length=128)


class CascadeStep(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    recipe_id: str
    enabled: bool = True
    override_class_id: int | None = None
    override_class_name: str | None = None
    dedupe_group: str = Field(default="", max_length=100)
    participate_cross_class_dedupe: bool = True
    clip_head_min_prob_override: float | None = Field(default=None, ge=0, le=1)
    clip_head_margin_override: float | None = Field(default=None, ge=0, le=1)
    extra_clip_classifier_path: str | None = None
    extra_clip_min_prob: float | None = Field(default=None, ge=0, le=1)
    extra_clip_margin: float | None = Field(default=None, ge=0, le=1)


class CascadeDedupe(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    per_class_iou: float = Field(default=.5, ge=0, le=1)
    cross_class_enabled: bool = False
    cross_class_iou: float = Field(default=.5, ge=0, le=1)
    cross_class_scope: Literal["global", "groups"] = "groups"
    confidence: Literal["sam_score", "clip_head_prob", "clip_head_margin"] = "sam_score"
    clip_head_recipe_id: str | None = None


class CascadeSave(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = None
    label: str = Field(min_length=1, max_length=200)
    steps: list[CascadeStep] = Field(min_length=1, max_length=32)
    dedupe: CascadeDedupe = Field(default_factory=CascadeDedupe)
    expected_revision: int = Field(default=0, ge=0)
    operation_id: str = Field(min_length=8, max_length=128)


class CascadeDelete(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)
    operation_id: str = Field(min_length=8, max_length=128)


class InstallModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str = Field(min_length=8, max_length=128)


class HeavyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    job: str
    operation_id: str = Field(min_length=8, max_length=128)
    parameters: dict = Field(default_factory=dict)


class ToolController:
    def __init__(self, store, auth, artifacts, root, *, configuration=None):
        self.store, self.auth, self.artifacts = store, auth, artifacts
        self.provider_gateway = None
        self.root = Path(root) / "tools"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = self.root.resolve(strict=True)
        self.media_root = self.root / "requests"
        self.model_root = self.root / "models"
        self.media_root.mkdir(exist_ok=True, mode=0o700)
        self.model_root.mkdir(exist_ok=True, mode=0o700)
        self.operations = HostedOperations(store, artifacts, root)
        self.analysis = CanonicalAnalysis(store, artifacts, Path(root) / "analysis")
        self.job_authority = HeavyJobAuthority(store)
        self.media = HostedMedia(store, artifacts)
        self._lock = threading.RLock()
        self._runtime = None
        self._pins = {}
        self._publishing = set()
        self._publication_threads = []
        self._reaper_stop = threading.Event()
        self._reaper_thread = None
        with store.transaction() as con:
            con.execute("CREATE TABLE IF NOT EXISTS hosted_tool_models (model_id TEXT PRIMARY KEY, definition_json TEXT NOT NULL, updated_at REAL NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS hosted_tool_preferences (user_id TEXT PRIMARY KEY REFERENCES hosted_users(id), model_id TEXT NOT NULL, revision INTEGER NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS hosted_tool_configuration (id INTEGER PRIMARY KEY CHECK(id=1), policy_json TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS hosted_tool_analysis (request_id TEXT PRIMARY KEY,job_id TEXT NOT NULL,actor_json TEXT NOT NULL,native_root TEXT NOT NULL,context_json TEXT NOT NULL,state TEXT NOT NULL,output_json TEXT,publication_json TEXT,error TEXT)")
            con.execute("CREATE TABLE IF NOT EXISTS hosted_tool_cascades(id TEXT PRIMARY KEY,actor_id TEXT NOT NULL,dataset_id TEXT NOT NULL,revision INTEGER NOT NULL,settings_json TEXT NOT NULL,deleted INTEGER NOT NULL DEFAULT 0)")
            saved = con.execute("SELECT policy_json FROM hosted_tool_configuration WHERE id=1").fetchone()
            installed = con.execute("SELECT model_id,definition_json FROM hosted_tool_models").fetchall()
        if configuration is None:
            configured_path = os.environ.get("TATOR_HOSTED_MODEL_CONFIG")
            configuration = json.loads(Path(configured_path).read_text()) if configured_path else json.loads(saved[0]) if saved else {}
        self.configuration = json.loads(json.dumps(configuration, allow_nan=False))
        export_defaults = {"include_caption0_in_training": True, "include_generated_qa_in_training": True,
            "include_deterministic_metadata_qa": False, "instruction_qa_restrict_speculative_language": False,
            "qa_mix": "balanced", "answer_format": "natural", "require_ready_instruction_export": True}
        self.configuration.setdefault("heavy_jobs", {}).setdefault("caption_export", {
            "label": "Caption training export", "factory": NATIVE_ANALYSIS_FACTORY, "device": "cpu",
            "operation": "caption_export", "config": {"native_root": str(Path(root) / "native-caption-export")},
            "allowed_parameters": list(export_defaults), "defaults": export_defaults, "timeout": 1800})
        self.registry = ModelRegistry(allowed_roots=[self.model_root])
        for row in installed:
            self._load_definition(row[0], json.loads(row[1]))
        from services.hosted.caption_jobs import CaptionJobs
        self.caption_jobs = CaptionJobs(self)
        from services.hosted.caption_batch import CaptionBatches
        self.caption_batches = CaptionBatches(self)
        from services.hosted.paid_batches import PaidBatches
        self.paid_batches = PaidBatches(self)

    @staticmethod
    def _actor(context):
        return Actor(context.actor_id, context.actor_role, session_id=context.session_id, instance_id=context.instance_id)

    @staticmethod
    def _context(actor, body):
        return RequestContext(actor.user_id, body.dataset_id, body.image_id, body.expected_version,
            body.generation, body.editor_id, workspace_id=body.workspace_id, actor_role=actor.role,
            session_id=actor.session_id, instance_id=actor.instance_id)

    def _load_definition(self, model_id, definition):
        validate_model_id(model_id)
        definition = dict(definition)
        directory = self.model_root / model_id
        directory.mkdir(exist_ok=True, mode=0o700)
        if directory.is_symlink():
            raise DomainError("tool_model_definition_invalid", 409)
        if definition.get("kind") in {"qwen", "edr", "sam3_recipe"}:
            # The descriptor digest names a fresh immutable directory. Updating
            # an installation never changes files beneath an existing worker.
            manifest = self.artifacts.pin(definition["checkpoint_sha256"], directory)
            with self.artifacts.open(manifest) as stream:
                descriptor = json.load(stream)
            if descriptor != {key: value for key, value in definition.items() if key != "checkpoint_sha256"}:
                raise DomainError("tool_model_definition_invalid", 409)
            revision = directory / (manifest.sha256 + ".files")
            revision.mkdir(exist_ok=True, mode=0o700)
            if revision.is_symlink():
                raise DomainError("tool_model_definition_invalid", 409)
            for field, folder in (("files", "model"), ("adapter_files", "adapter")):
                if field not in definition:
                    continue
                target_root = revision / folder
                target_root.mkdir(exist_ok=True, mode=0o700)
                if target_root.is_symlink():
                    raise DomainError("tool_model_definition_invalid", 409)
                for relative, digest in definition[field].items():
                    validate_model_member(relative)
                    target = target_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    if any(path.is_symlink() for path in (target, *target.parents)):
                        raise DomainError("tool_model_definition_invalid", 409)
                    self.artifacts.pin(digest, directory)
                    if not target.exists():
                        os.link(directory / digest, target, follow_symlinks=False)
                definition["model_path" if field == "files" else "adapter_path"] = str(target_root)
            self.registry.register(model_id, definition)
            return
        for field in ("checkpoint", "bpe", "metadata"):
            digest = definition.get(field + "_sha256")
            if not digest:
                continue
            extension = definition.pop(field + "_suffix", {"checkpoint": ".pt", "bpe": ".gz", "metadata": ".pkl"}[field])
            if extension not in {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".gz", ".pkl", ".joblib"}:
                raise DomainError("tool_model_definition_invalid", 409)
            self.artifacts.pin(digest, directory)
            target = directory / (digest + extension)
            if not target.exists():
                os.link(directory / digest, target, follow_symlinks=False)
            definition["checkpoint" if field == "checkpoint" else field + "_path"] = str(target)
        self.registry.register(model_id, definition)

    def _reload_model(self, model_id):
        with self.store.read_connection() as con:
            row = con.execute("SELECT definition_json FROM hosted_tool_models WHERE model_id=?", (model_id,)).fetchone()
        if row:
            self._load_definition(model_id, json.loads(row[0]))

    def install_model(self, actor, model_id, operation_id):
        self.auth.require_admin(actor)
        previous = self.store.raw_receipt(actor, operation_id)
        if previous:
            if previous["operation"] != "install_tool_model" or previous["result"].get("model_id") != model_id:
                raise DomainError("operation_id_reused", 409)
            with self._lock, self.artifacts.exclusive():
                self._reload_model(model_id)
            return self.store.disclose_receipt_result(actor, operation_id, previous["operation"], previous["result"])
        sources = self.configuration.get("models", {})
        if model_id not in sources:
            raise DomainError("tool_model_not_configured", 404, "An administrator must configure this model on the server.")
        source = ModelRegistry(allowed_roots=self.configuration.get("allowed_model_roots", []), definitions={model_id: sources[model_id]})
        definition = source.definition(model_id)
        with self._lock, self.artifacts.exclusive():
            previous = self.store.raw_receipt(actor, operation_id)
            if previous:
                if previous["operation"] != "install_tool_model" or previous["result"].get("model_id") != model_id:
                    raise DomainError("operation_id_reused", 409)
                self._reload_model(model_id)
                return self.store.disclose_receipt_result(actor, operation_id, previous["operation"], previous["result"])
            if definition["kind"] in {"edr", "sam3_recipe"}:
                identities = {}
                for kind, component in definition["components"].items():
                    if self._definition(component)["kind"] != ("qwen" if kind.startswith("qwen") else "classifier" if kind == "extra_classifier" else kind):
                        raise DomainError("tool_edr_component_kind_invalid", 422)
                    identities[kind] = self.registry.model_identity(component)
                if definition.get("component_identities", identities) != identities:
                    raise DomainError("tool_edr_component_changed", 409)
                definition["component_identities"] = identities
            if definition["kind"] in {"qwen", "edr", "sam3_recipe"}:
                for files_field, path_field in (("files", "model_path"), ("adapter_files", "adapter_path")):
                    if files_field not in definition:
                        continue
                    directory = Path(definition.pop(path_field))
                    for relative, digest in definition[files_field].items():
                        ref = self.artifacts.put_file(directory / relative, expected_sha256=digest)
                        ref_operation = "modelref-" + hashlib.sha256(f"{operation_id}:{files_field}:{relative}".encode()).hexdigest()
                        self.operations.register_artifact(actor, ref, "tool_model", model_id, ref_operation)
                descriptor = {key: value for key, value in definition.items() if key != "checkpoint_sha256"}
                manifest = self.artifacts.put_bytes(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
                self.operations.register_artifact(actor, manifest, "tool_model", model_id, "modelref-" + hashlib.sha256(f"{operation_id}:manifest".encode()).hexdigest())
                definition["checkpoint_sha256"] = manifest.sha256
            for field, path_field in (("checkpoint", "checkpoint"), ("bpe", "bpe_path"), ("metadata", "metadata_path")):
                if path_field not in definition:
                    continue
                path = Path(definition.pop(path_field))
                if path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".gz", ".pkl", ".joblib"}:
                    raise DomainError("tool_model_definition_invalid", 422, "The model file extension is not supported.")
                ref = self.artifacts.put_file(path, expected_sha256=definition[field + "_sha256"])
                ref_operation = "modelref-" + hashlib.sha256(f"{operation_id}:{field}".encode()).hexdigest()
                self.operations.register_artifact(actor, ref, "tool_model", model_id, ref_operation)
                definition[field + "_suffix"] = path.suffix.lower()
            # Save only portable policy. Source paths are needed for future
            # installs, not to restore installed models from their artifacts.
            policy = {key: value for key, value in self.configuration.items() if key not in {"models", "allowed_model_roots"}}
            def publish(con):
                con.execute("INSERT INTO hosted_tool_models VALUES (?,?,?) ON CONFLICT(model_id) DO UPDATE SET definition_json=excluded.definition_json,updated_at=excluded.updated_at",
                    (model_id, json.dumps(definition, sort_keys=True), self.store.clock()))
                con.execute("INSERT INTO hosted_tool_configuration VALUES (1,?) ON CONFLICT(id) DO UPDATE SET policy_json=excluded.policy_json", (json.dumps(policy),))
                return {"model_id": model_id, "checkpoint_sha256": definition["checkpoint_sha256"]}
            result = self.store._command(actor, operation_id, "install_tool_model", {"model_id": model_id, "definition": definition}, publish, admin=True)
            self._load_definition(model_id, definition)
            return result

    def _sam_selection(self, actor):
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            selected = con.execute("SELECT model_id,revision FROM hosted_tool_preferences WHERE user_id=?", (actor.user_id,)).fetchone()
        policy = self.configuration.get("user_sam", {})
        return {"model_id": selected[0] if selected else policy.get("default_model_id"), "revision": selected[1] if selected else 0}

    def _definition(self, model_id):
        try:
            return self.registry.definition(model_id)
        except ValueError as exc:
            raise DomainError("tool_model_not_installed", 503, "An administrator must install the selected model before this tool can run.") from exc

    def _dataset_model(self, dataset, kind, *, required=True):
        selected = dataset["settings"].get(kind)
        if not isinstance(selected, dict) or not selected.get("model_id"):
            if not required:
                return None, None
            raise DomainError("tool_not_configured", 503, f"An administrator must select a {kind} for this dataset.")
        definition = self._definition(selected["model_id"])
        if definition["checkpoint_sha256"] != selected.get("checkpoint_sha256") or self.registry.model_identity(selected["model_id"]) != selected.get("model_identity"):
            raise DomainError("tool_model_selection_changed", 409, f"The {kind} changed. An administrator must confirm its new version.")
        classes = {row["id"]: row for row in dataset["classes"] if not row["retired"]}
        mapping = selected.get("class_mapping", {})
        if set(mapping) != set(definition["labels"]) or any(value not in classes for value in mapping.values()):
            raise DomainError("tool_class_mapping_changed", 409, "An administrator must check this model's dataset classes.")
        return selected, {label: classes[class_id] for label, class_id in mapping.items()}

    def _qwen_config(self, actor, *, required):
        policy = self.configuration.get("user_qwen", {})
        models = {}
        for model_id in policy.get("allowed_model_ids", []):
            try:
                definition = self.registry.definition(model_id)
            except ValueError:
                continue
            if definition["kind"] != "qwen":
                raise DomainError("tool_model_kind_invalid", 422)
            models[model_id] = definition
        default = policy.get("default_model_id")
        if default not in models or any(policy.get(key) and policy[key] not in models for key in ("refinement_model_id", "fallback_model_id")):
            if required:
                raise DomainError("tool_not_configured", 503, "An administrator must install and configure a permitted local Qwen model.")
            return None
        from services.hosted.config import _native_device
        return {"actor_id": actor.user_id, "models": models, "device": _native_device(policy.get("device")),
                **{key: policy[key] for key in ("default_model_id", "refinement_model_id", "fallback_model_id", "min_pixels", "max_pixels") if key in policy}}

    def _dataset_pipeline_spec(self, actor, dataset_id, operation):
        from services.hosted.config import _native_device
        dataset = self.store.get_dataset(actor, dataset_id)
        device = _native_device(self.configuration.get("detector_device"))
        labelmap = [row["name"] for row in sorted(dataset["classes"], key=lambda row: row["position"])]
        config = {"dataset_id": dataset_id}
        selection, bindings = self._dataset_model(dataset, "detector", required=operation in DETECTOR_OPERATIONS)
        if selection:
            detector = self.registry.dataset_spec(dataset_id, selection["model_id"], media_root=self.media_root, device=device)
            config["detector"] = {**dict(detector.config), "class_bindings": bindings, "dataset_labelmap": labelmap}
        selection, bindings = self._dataset_model(dataset, "edr", required=operation in EDR_OPERATIONS)
        if selection:
            model = self._definition(selection["model_id"])
            components = {}
            for kind, model_id in model["components"].items():
                component = self._definition(model_id)
                if component["kind"] != ("qwen" if kind.startswith("qwen") else kind) or self.registry.model_identity(model_id) != model["component_identities"].get(kind):
                    raise DomainError("tool_edr_component_changed", 409, "An administrator must reinstall and confirm this recipe after its component model changes.")
                components[kind] = component
            edr = {
                "dataset_id": dataset_id, "media_root": str(self.media_root),
                "model": {**model, "component_models": components}, "class_bindings": bindings,
                "dataset_labelmap": labelmap}
            if "qwen" in components:
                edr["qwen"] = {"actor_id": "dataset:" + dataset_id, "default_model_id": model["components"]["qwen"],
                    "models": {value["model_id"]: value for key, value in components.items() if key.startswith("qwen")}, "device": device,
                    **{field: model["components"][role] for role, field in (("qwen_refinement", "refinement_model_id"), ("qwen_fallback", "fallback_model_id")) if role in components}}
            config["edr"] = edr
        return AdapterSpec("services.hosted.adapters:DatasetPipelineAdapter", config, device), "dataset"

    def _sam_recipe_config(self, dataset, *, required=False):
        saved = dataset["settings"].get("sam3_recipes", {})
        recipes = {}
        for model_id in saved.get("model_ids", []):
            definition = self._definition(model_id)
            if definition["kind"] != "sam3_recipe" or saved.get("identities", {}).get(model_id) != self.registry.model_identity(model_id):
                raise DomainError("tool_sam_recipe_changed", 409)
            components = {}
            for role, component_id in definition["components"].items():
                if definition["component_identities"][role] != self.registry.model_identity(component_id):
                    raise DomainError("tool_sam_recipe_component_changed", 409)
                components[role] = self._definition(component_id)
            recipes[model_id] = {**definition, "component_models": components}
        if required and not recipes:
            raise DomainError("tool_not_configured", 503, "An administrator must attach a saved SAM3 recipe to this dataset.")
        return {"recipes": recipes, "class_bindings": {row["name"]: row for row in dataset["classes"] if not row["retired"]}}

    def _spec(self, actor, dataset_id, operation):
        if operation in DETECTOR_OPERATIONS | EDR_OPERATIONS:
            return self._dataset_pipeline_spec(actor, dataset_id, operation)
        if operation in SAM_OPERATIONS | QWEN_OPERATIONS | SAM_RECIPE_OPERATIONS:
            dataset = self.store.get_dataset(actor, dataset_id)
            selection = self._sam_selection(actor)
            policy = self.configuration.get("user_sam", {})
            model_id = selection["model_id"]
            if model_id not in policy.get("allowed_model_ids", []):
                raise DomainError("tool_not_configured", 503, "No permitted SAM model is configured.")
            model = self._definition(model_id)
            if operation.startswith("sam3_") and operation not in SAM_RECIPE_OPERATIONS and model["kind"] != "sam3":
                raise DomainError("tool_sam3_required", 503, "Select a permitted SAM3 model to use this tool.")
            classifier, bindings = self._dataset_model(dataset, "classifier", required=operation in CLASSIFIER_OPERATIONS)
            spec = self.registry.user_spec(actor.user_id, model_id, media_root=self.media_root,
                device=policy.get("device"), capacity=policy.get("capacity", 3), classifier_id=classifier["model_id"] if classifier else None)
            if bindings:
                spec = replace(spec, config={**dict(spec.config), "classifier_bindings": bindings})
            qwen = self._qwen_config(actor, required=operation in QWEN_OPERATIONS)
            if qwen:
                spec = replace(spec, config={**dict(spec.config), "qwen": qwen})
            try:
                chain = self._sam_recipe_config(dataset, required=operation in SAM_RECIPE_OPERATIONS)
            except DomainError:
                if operation in SAM_RECIPE_OPERATIONS:
                    raise
                chain = {"recipes": {}}
            if chain["recipes"]:
                spec = replace(spec, config={**dict(spec.config), "sam3_recipes": {**chain, "actor_id": actor.user_id}})
            return spec, "user"
        raise DomainError("tool_not_implemented", 503, "This tool's hosted runtime is not connected yet.")

    def authorize(self, context, phase):
        if phase.startswith("heavy") and context.background_job_id:
            with self.store.read_connection() as con:
                self.job_authority.validate(con, context.background_job_id, context=context, completed=phase == "heavy_result")
            return True
        actor = self._actor(context)
        if phase.startswith("heavy"):
            self.auth.require_admin(actor)
            snapshot = self.store.get_snapshot(actor, context.annotation_version)
            if snapshot["manifest"]["dataset_id"] != context.dataset_id:
                raise DomainError("tool_dataset_mismatch", 409)
            if context.operation == "dqe_review":
                parent_id = context.model_config.get("parent_analysis", {}).get("job_id")
                parent = self.analysis.status(actor, parent_id)
                if parent["stale"] or parent["snapshot_id"] != context.annotation_version:
                    raise DomainError("analysis_snapshot_stale", 409)
            return True
        mutation = MutationContext(context.image_id, context.annotation_version, context.ownership_generation,
                                   context.editor_id, "inference-authority")
        image = self.store.authorize_context(actor, mutation, write=context.operation not in READ_ONLY_OPERATIONS)
        if image["dataset_id"] != context.dataset_id:
            raise DomainError("tool_dataset_mismatch", 409)
        with self._lock:
            spec, _ = self._spec(actor, context.dataset_id, context.operation)
        if context.model_config and context.as_dict()["model_config"] != replace(context, model_config=spec.config).as_dict()["model_config"]:
            raise DomainError("tool_model_selection_changed", 409)
        return True

    def _ensure_runtime(self):
        with self._lock:
            if self._runtime is None:
                self._runtime = RuntimeSupervisor(self.root / "runtime", authorize=self.authorize, event_sink=self._event)
                # Acquiring the supervisor lock establishes that the previous
                # controller is gone. These are generated hard-link pins only.
                try:
                    for directory in self.media_root.iterdir():
                        if directory.is_dir() and not directory.is_symlink() and directory.name.startswith(("request-", "heavy-")):
                            shutil.rmtree(directory)
                except OSError:
                    self._runtime.close()
                    self._runtime = None
                    raise
                self._reaper_thread = threading.Thread(target=self._reap_loop, daemon=True, name="hosted-tool-pins")
                self._reaper_thread.start()
                # A published receipt is durable even though the runtime does
                # not persist potentially large model responses. Resume only
                # retention of an already computed, captured native response.
                with self.store.transaction() as con:
                    rows = con.execute("SELECT request_id,state FROM hosted_tool_analysis WHERE state!='complete'").fetchall()
                    for row in rows:
                        if self.job_authority.recover(con, row['request_id']):
                            continue
                        grant = self.job_authority.row(con, row['request_id'])
                        if not grant:
                            # Old publication failures could leave the canonical job captured.
                            con.execute("UPDATE hosted_tool_analysis SET state='pending' WHERE request_id=? AND job_id IN (SELECT id FROM hosted_analysis_jobs WHERE state='captured')", (row['request_id'],))
                            self.job_authority.terminal(con, row['request_id'], 'interrupted', 'legacy_job_authority_unavailable')
                        elif row['state'] in {'failed', 'cancelled', 'interrupted'}:
                            self.job_authority.terminal(con, row['request_id'], row['state'], grant['error'] or 'native_analysis_interrupted')
                        else:
                            con.execute("UPDATE hosted_tool_analysis SET state='pending' WHERE request_id=?", (row['request_id'],))
                    recover = con.execute("SELECT request_id FROM hosted_tool_analysis WHERE state='pending'").fetchall()
                for row in recover:
                    self._start_publication(row[0])
            return self._runtime

    def _reap_loop(self):
        while not self._reaper_stop.wait(1):
            with self._lock:
                pins = list(self._pins.items())
            for request_id, (_, actor_id) in pins:
                try:
                    with self.store.transaction() as con:
                        grant = self.job_authority.row(con, request_id)
                        revoked = False
                        if grant and grant['state'] == 'active':
                            try:
                                self.job_authority.validate(con, request_id)
                            except DomainError as exc:
                                self.job_authority.terminal(con, request_id, 'failed', exc.code)
                                revoked = True
                    if revoked:
                        self._runtime.cancel(request_id, actor_id=actor_id)
                    status = self._runtime.request_status(request_id, actor_id=actor_id)
                    if status["state"] in TERMINAL:
                        self._terminal(request_id)
                except RuntimeRejected as exc:
                    if exc.code == "runtime_request_not_found":
                        self._terminal(request_id)
                except OSError:
                    pass  # Retain the pin entry and retry; never touch sources.

    def _event(self, event):
        if event["type"] == "request_finished":
            self._terminal(event["request_id"])

    def _analysis_binding(self, request_id):
        with self.store.read_connection() as con:
            row = con.execute("SELECT * FROM hosted_tool_analysis WHERE request_id=?", (request_id,)).fetchone()
            return dict(row) if row else None

    def _terminal(self, request_id):
        binding = self._analysis_binding(request_id)
        if binding and binding["state"] in {"pending", "publishing"}:
            self._start_publication(request_id)
        elif not binding or binding["state"] == "complete":
            if not binding and self._runtime:
                with self.store.transaction() as con:
                    grant = self.job_authority.row(con, request_id)
                    if grant and grant['state'] == 'active':
                        status = self._runtime.request_status(request_id, actor_id=grant['actor_id'])
                        if status['state'] == 'completed':
                            try:
                                self.job_authority.validate(con, request_id)
                                con.execute("UPDATE hosted_heavy_grants SET state='complete' WHERE request_id=?", (request_id,))
                            except DomainError as exc:
                                self.job_authority.terminal(con, request_id, 'failed', exc.code)
                        elif status['state'] in TERMINAL:
                            self.job_authority.terminal(con, request_id,
                                status['state'] if status['state'] in {'cancelled', 'interrupted'} else 'failed',
                                status.get('error') or 'heavy_job_interrupted')
            self._cleanup(request_id)

    def _start_publication(self, request_id):
        with self._lock:
            if request_id in self._publishing:
                return
            self._publishing.add(request_id)
            thread = threading.Thread(target=self._publish_native, args=(request_id,), daemon=True,
                                      name="hosted-analysis-publication")
            self._publication_threads = [item for item in self._publication_threads if item.is_alive()]
            self._publication_threads.append(thread)
            thread.start()

    def _publish_native(self, request_id):
        try:
            with self.store.transaction() as con:
                if self.job_authority.recover(con, request_id):
                    return
            binding = self._analysis_binding(request_id)
            if not binding or binding["state"] not in {"pending", "publishing"}:
                return
            with self.store.read_connection() as con:
                self.job_authority.validate(con, request_id, job_id=binding['job_id'])
            owner = json.loads(binding['actor_json'])['user_id']
            output = json.loads(binding["output_json"]) if binding["output_json"] else None
            if output is None:
                try:
                    status = self._runtime.request_status(request_id, actor_id=owner)
                except RuntimeRejected as exc:
                    if exc.code != 'runtime_request_not_found':
                        raise
                    status = {'state': 'interrupted', 'error': 'runtime_request_not_retained'}
                if status["state"] not in TERMINAL:
                    return
                if status["state"] == "completed" and not status.get("result_available"):
                    status = {"state": "interrupted", "error": "runtime_result_not_retained"}
                if status["state"] != "completed":
                    with self.store.transaction() as con:
                        self.job_authority.terminal(con, request_id,
                            status['state'] if status['state'] in {'cancelled', 'interrupted'} else 'failed',
                            status.get('error') or 'native_analysis_interrupted')
                    return
                output = self._runtime.result(request_id, actor_id=owner)["result"]
            with self.store.transaction() as con:
                self.job_authority.validate(con, request_id, job_id=binding['job_id'])
                con.execute("UPDATE hosted_tool_analysis SET state='publishing',output_json=? WHERE request_id=?",
                            (json.dumps(output, allow_nan=False, sort_keys=True), request_id))
            self.analysis.publish_heavy_result(request_id, binding["job_id"], output, native_root=binding["native_root"])
            try:
                self._cleanup(request_id)
            except OSError:
                pass
        except Exception as exc:
            code = exc.code if isinstance(exc, (DomainError, RuntimeRejected)) else "analysis_publication_failed"
            with self.store.transaction() as con:
                self.job_authority.terminal(con, request_id, 'failed', code)
        finally:
            with self._lock:
                self._publishing.discard(request_id)

    def _cleanup(self, request_id):
        with self._lock:
            pinned = self._pins.get(request_id)
            if pinned:
                try:
                    shutil.rmtree(pinned[0], ignore_errors=False)
                except FileNotFoundError:
                    pass  # A later supervisor may already have removed old pins.
                self._pins.pop(request_id, None)

    def submit(self, actor, body):
        if set(body.parameters) - _parameters_for(body.operation):
            raise DomainError("tool_parameter_not_allowed", 422, "Media paths, model files and factories are selected by the server.")
        if body.operation == "qwen_caption_case":
            from services.hosted.caption_case import CaptionSettings
            CaptionSettings(**body.parameters.get("settings", {}))
            if set(body.parameters.get("caption_request", {})) - _parameters_for("qwen_caption"):
                raise DomainError("tool_parameter_not_allowed", 422)
        if body.operation in QWEN_OPERATIONS:
            dataset = self.store.get_dataset(actor, body.dataset_id)
            body = body.model_copy(update={"parameters": qwen_guidance(dataset["settings"], body.operation, body.parameters)})
        context = replace(self._context(actor, body), operation=body.operation)
        self.authorize(context, "admit")
        directory = None
        try:
            with self._lock, self.artifacts.exclusive():
                runtime = self._ensure_runtime()
                image = self.store.authorize_context(actor, MutationContext(body.image_id, body.expected_version, body.generation,
                    body.editor_id, "inference-authority"), write=body.operation not in READ_ONLY_OPERATIONS)
                if image["dataset_id"] != body.dataset_id:
                    raise DomainError("tool_dataset_mismatch", 409)
                directory = Path(tempfile.mkdtemp(prefix="request-", dir=self.media_root))
                view = self.media.resolve(image['media_hash'])
                reference = self.artifacts.pin(self.media.reference(view), directory)
                payload = {**body.parameters, "media_path": str(directory / reference.sha256), "media_sha256": reference.sha256,
                           "source_media_hash": image['media_hash']}
                spec, owner = self._spec(actor, body.dataset_id, body.operation)
                if body.operation in EDR_OPERATIONS:
                    recipe_id = spec.config["edr"]["model"]["model_id"]
                    if body.parameters.get("recipe_id", recipe_id) != recipe_id:
                        raise DomainError("tool_model_not_allowed", 403)
                    payload["annotation_objects"] = image["annotations"].get("objects", [])
                if body.operation in SAM_RECIPE_OPERATIONS:
                    steps = body.parameters.get("steps")
                    if not isinstance(steps, list) or not steps or len(steps) > 32:
                        raise DomainError("tool_sam_recipe_steps_invalid", 422)
                    parsed_steps = [CascadeStep.model_validate(step) for step in steps]
                    parsed_dedupe = CascadeDedupe.model_validate(body.parameters.get("dedupe", {}))
                    self._validate_cascade(actor, body.dataset_id, parsed_steps, parsed_dedupe)
                    payload["steps"] = [step.model_dump() for step in parsed_steps]
                    payload["dedupe"] = parsed_dedupe.model_dump()
                if body.operation in QWEN_OPERATIONS:
                    available = spec.config["qwen"]["models"]
                    caption_parameters = body.parameters.get("caption_request", {}) if body.operation == "qwen_caption_case" else body.parameters
                    for key in ("model_id", "refinement_model_id", "caption_fallback_model_id"):
                        model_id = caption_parameters.get(key)
                        if model_id and model_id not in {"auto", "none", "active", "same"} and model_id not in available:
                            raise DomainError("tool_model_not_allowed", 403)
                    if body.operation == "qwen_caption_case":
                        dataset = self.store.get_dataset(actor, body.dataset_id)
                        names = {row["id"]: row["name"] for row in dataset["classes"]}
                        hints, counts = [], {}
                        for item in image["annotations"].get("objects", []):
                            if "bbox" not in item or item.get("class_id") not in names:
                                continue
                            label = names[item["class_id"]]
                            left, top, width, height = item["bbox"]
                            hints.append({"label": label, "bbox": [left, top, left + width, top + height], "source_id": item["id"]})
                            counts[label] = counts.get(label, 0) + 1
                        payload["case"] = {"name": image["filename"], "image_name": image["filename"],
                            "class_counts": counts, "label_count": len(hints), "label_hints": hints}
                if owner == "user":
                    runtime.register_user(actor.user_id, spec)
                else:
                    runtime.register_dataset(body.dataset_id, spec)
                timeout = self.configuration.get("request_timeouts", {}).get(body.operation)
                if body.operation == "qwen_caption_case":
                    timeout = CaptionSettings(**body.parameters["settings"]).per_image_timeout_seconds
                request_id = runtime.submit(context, body.operation, payload, owner=owner, timeout=timeout)
                self._pins[request_id] = (directory, actor.user_id)
                directory = None
                return runtime.request_status(request_id, actor_id=actor.user_id)
        finally:
            if directory:
                shutil.rmtree(directory)

    def select_sam(self, actor, body):
        if body.model_id not in self.configuration.get("user_sam", {}).get("allowed_model_ids", []):
            raise DomainError("tool_model_not_allowed", 403)
        if self._definition(body.model_id)["kind"] not in {"sam1", "sam3"}:
            raise DomainError("tool_model_kind_invalid", 422)
        with self._lock:
            def update(con):
                row = con.execute("SELECT revision FROM hosted_tool_preferences WHERE user_id=?", (actor.user_id,)).fetchone()
                revision = row[0] if row else 0
                if revision != body.expected_revision:
                    raise DomainError("tool_preference_changed", 409)
                con.execute("INSERT INTO hosted_tool_preferences VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET model_id=excluded.model_id,revision=excluded.revision",
                    (actor.user_id, body.model_id, revision + 1))
                return {"model_id": body.model_id, "revision": revision + 1}
            result = self.store._command(actor, body.operation_id, "select_user_sam", body.model_dump(), update)
            return result

    def select_detector(self, actor, dataset_id, body):
        return self.select_dataset_model(actor, dataset_id, body, "detector")

    def select_sam_recipes(self, actor, dataset_id, body):
        self.auth.require_admin(actor)
        request = body.model_dump(exclude={"operation_id"})
        previous = self.store.raw_receipt(actor, body.operation_id)
        if previous:
            result = previous["result"]
            if previous["operation"] != "patch_dataset_settings" or result.get("id") != dataset_id or result.get("settings", {}).get("sam3_recipes", {}).get("selection_request") != request:
                raise DomainError("operation_id_reused", 409)
            return self.store.disclose_receipt_result(actor, body.operation_id, previous["operation"], result)
        if len(body.model_ids) != len(set(body.model_ids)):
            raise DomainError("tool_recipe_selection_invalid", 422)
        identities = {}
        for model_id in body.model_ids:
            definition = self._definition(model_id)
            if definition["kind"] != "sam3_recipe":
                raise DomainError("tool_model_kind_invalid", 422)
            identities[model_id] = self.registry.model_identity(model_id)
        return self.store.patch_dataset_settings(actor, dataset_id,
            {"sam3_recipes": {"model_ids": body.model_ids, "identities": identities, "selection_request": request}}, body.expected_revision, body.operation_id)

    def _validate_cascade(self, actor, dataset_id, steps, dedupe):
        dataset = self.store.get_dataset(actor, dataset_id)
        config = self._sam_recipe_config(dataset, required=True)
        if not any(step.enabled for step in steps):
            raise DomainError("tool_recipe_selection_invalid", 422)
        for step in steps:
            recipe = config["recipes"].get(step.recipe_id)
            if recipe is None or step.extra_clip_classifier_path and step.extra_clip_classifier_path != recipe["components"].get("extra_classifier"):
                raise DomainError("tool_model_not_allowed", 403)
            if step.override_class_name:
                target = config["class_bindings"].get(step.override_class_name)
                if target is None or step.override_class_id is not None and step.override_class_id != target["position"]:
                    raise DomainError("tool_class_mapping_changed", 409)
            elif step.override_class_id is not None:
                raise DomainError("tool_class_name_required", 422)
        if dedupe.clip_head_recipe_id and dedupe.clip_head_recipe_id not in config["recipes"]:
            raise DomainError("tool_model_not_allowed", 403)
        return dataset

    def cascades(self, actor, dataset_id, cascade_id=None):
        self.store.get_dataset(actor, dataset_id)
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            rows = con.execute("SELECT * FROM hosted_tool_cascades WHERE actor_id=? AND dataset_id=? AND deleted=0 ORDER BY id", (actor.user_id, dataset_id)).fetchall()
        result = [{"id": row["id"], "revision": row["revision"], **json.loads(row["settings_json"])} for row in rows]
        if cascade_id is not None:
            found = next((row for row in result if row["id"] == cascade_id), None)
            if found is None:
                raise DomainError("tool_cascade_not_found", 404)
            return found
        return result

    def save_cascade(self, actor, dataset_id, body):
        # Immutable command replay remains valid even if an attached recipe was
        # retired after this preference was saved.
        previous = self.store.raw_receipt(actor, body.operation_id)
        dataset = None if previous else self._validate_cascade(actor, dataset_id, body.steps, body.dedupe)
        def save(con):
            cascade_id = body.id or uuid.uuid4().hex
            current = con.execute("SELECT * FROM hosted_tool_cascades WHERE id=?", (cascade_id,)).fetchone()
            if current and (current["actor_id"] != actor.user_id or current["dataset_id"] != dataset_id):
                raise DomainError("tool_cascade_not_found", 404)
            revision = current["revision"] if current else 0
            if revision != body.expected_revision or current and current["deleted"]:
                raise DomainError("tool_cascade_changed", 409)
            if not current and con.execute("SELECT COUNT(*) FROM hosted_tool_cascades WHERE actor_id=? AND deleted=0", (actor.user_id,)).fetchone()[0] >= 500:
                raise DomainError("tool_cascade_limit", 429)
            if self.store._dataset(con, dataset_id)["class_version"] != dataset["class_version"]:
                raise DomainError("tool_class_mapping_changed", 409)
            settings = body.model_dump(exclude={"id", "expected_revision", "operation_id"})
            settings["class_version"] = dataset["class_version"]
            con.execute("INSERT INTO hosted_tool_cascades VALUES(?,?,?,?,?,0) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,settings_json=excluded.settings_json", (cascade_id, actor.user_id, dataset_id, revision + 1, json.dumps(settings, allow_nan=False)))
            return {"id": cascade_id, "revision": revision + 1, **settings}
        return self.store._command(actor, body.operation_id, "save_tool_cascade", {"dataset_id": dataset_id, **body.model_dump()}, save)

    def delete_cascade(self, actor, dataset_id, cascade_id, body):
        self.store.get_dataset(actor, dataset_id)
        def remove(con):
            row = con.execute("SELECT * FROM hosted_tool_cascades WHERE id=?", (cascade_id,)).fetchone()
            if row is None or row["actor_id"] != actor.user_id or row["dataset_id"] != dataset_id:
                raise DomainError("tool_cascade_not_found", 404)
            if row["revision"] != body.expected_revision or row["deleted"]:
                raise DomainError("tool_cascade_changed", 409)
            con.execute("UPDATE hosted_tool_cascades SET deleted=1,revision=revision+1 WHERE id=?", (cascade_id,))
            return {"id": cascade_id, "revision": row["revision"] + 1, "deleted": True}
        return self.store._command(actor, body.operation_id, "delete_tool_cascade", {"dataset_id": dataset_id, "id": cascade_id, **body.model_dump()}, remove)

    def select_dataset_model(self, actor, dataset_id, body, kind):
        self.auth.require_admin(actor)
        request = body.model_dump(exclude={"operation_id"})
        previous = self.store.raw_receipt(actor, body.operation_id)
        if previous:
            result = previous["result"]
            if previous["operation"] != "patch_dataset_settings" or result.get("id") != dataset_id or result.get("settings", {}).get(kind, {}).get("selection_request") != request:
                raise DomainError("operation_id_reused", 409)
            return self.store.disclose_receipt_result(actor, body.operation_id, previous["operation"], result)
        definition = self._definition(body.model_id)
        if definition["kind"] not in ({"yolo", "rfdetr"} if kind == "detector" else {kind}):
            raise DomainError("tool_model_kind_invalid", 422)
        with self._lock:
            dataset = self.store.get_dataset(actor, dataset_id)
            active = {row["id"]: row for row in dataset["classes"] if not row["retired"]}
            names = {row["name"].casefold(): row["id"] for row in active.values()}
            mapping = body.class_mapping
            if mapping is None:
                mapping = {label: names.get(label.casefold()) for label in definition["labels"]}
            if set(mapping) != set(definition["labels"]) or any(class_id not in active for class_id in mapping.values()):
                raise DomainError("tool_class_mapping_invalid", 422, "Map each model label to an existing dataset class before selecting this model.")
            settings = {kind: {"model_id": body.model_id, "checkpoint_sha256": definition["checkpoint_sha256"], "model_identity": self.registry.model_identity(body.model_id), "class_mapping": mapping, "selection_request": request}}
            result = self.store.patch_dataset_settings(actor, dataset_id, settings, body.expected_revision, body.operation_id)
            if self._runtime and kind == "detector":
                spec, _ = self._spec(actor, dataset_id, "detect_full")
                self._runtime.register_dataset(dataset_id, spec)
            return result

    def catalog(self, actor, dataset_id=None):
        from services.hosted.caption_case import CaptionSettings
        self._sam_selection(actor)  # Verify current session, including on GET.
        allowed = set(self.configuration.get("user_sam", {}).get("allowed_model_ids", []))
        allowed.update(self.configuration.get("user_qwen", {}).get("allowed_model_ids", []))
        dataset = self.store.get_dataset(actor, dataset_id) if dataset_id else None
        if dataset:
            allowed.update(value.get("model_id") for key in ("detector", "classifier", "edr") if isinstance(value := dataset["settings"].get(key), dict))
            allowed.update(dataset["settings"].get("sam3_recipes", {}).get("model_ids", []))
            for recipe_id in dataset["settings"].get("sam3_recipes", {}).get("model_ids", []):
                allowed.update(self._definition(recipe_id).get("components", {}).values())
        installed = [row for row in self.registry.catalog() if actor.role == "admin" or row["model_id"] in allowed]
        result = {"models": installed, "sam": self._sam_selection(actor), "implemented": sorted(SAM_OPERATIONS | DETECTOR_OPERATIONS | QWEN_OPERATIONS | EDR_OPERATIONS | SAM_RECIPE_OPERATIONS),
                  "sam_allowed_model_ids": list(self.configuration.get("user_sam", {}).get("allowed_model_ids", [])),
                  "parameters": {operation: sorted(_parameters_for(operation)) for operation in QWEN_OPERATIONS | SAM_RECIPE_OPERATIONS},
                  "caption_job_parameters": {"caption_request": sorted(_parameters_for("qwen_caption")), "settings": sorted(CaptionSettings.model_fields)},
                  "caption_job_capabilities": {"current_image": True, "dataset_batch": actor.role == "admin", "providers": ["local_qwen"],
                    "instruction_qa": True, "annotation_writes": False, "automatic_restart_resume": True,
                    "pilot_certification": actor.role == "admin", "pilot_scope": "hosted_sequential_local_qwen"},
                  "qwen": {key: value for key, value in self.configuration.get("user_qwen", {}).items() if key in {"default_model_id", "allowed_model_ids", "refinement_model_id", "fallback_model_id"}}}
        from services.hosted.paid_caption import PaidCaptionOptions
        result["caption_job_parameters"]["provider_options"] = sorted(PaidCaptionOptions.model_fields)
        result["caption_job_capabilities"]["openai_batch_transport"] = False
        result["caption_job_capabilities"]["paid_local_sampling_controls"] = False
        if dataset and self.provider_gateway is not None:
            result["paid_caption"] = self.provider_gateway.catalog(actor, dataset_id)
            if result["paid_caption"]["models"]:
                result["caption_job_capabilities"]["providers"].append("openai")
            result["paid_batch"] = self.paid_batches.catalog(actor, dataset_id)
            result["caption_job_capabilities"]["openai_batch_transport"] = bool(result["paid_batch"]["models"])
        result["caption_job_capabilities"]["pilot_scopes"] = {"local_qwen": "hosted_sequential_local_qwen", "openai": "hosted_sequential_metered_responses"}
        if actor.role == "admin":
            from services.hosted.caption_batch import PilotSettings
            from models.schemas import QwenCaptionRequest
            caption_defaults = QwenCaptionRequest(image_token="catalog-defaults").model_dump()
            if dataset:
                caption_defaults.update(qwen_guidance(dataset["settings"], "qwen_caption", {}))
            result["caption_batch_defaults"] = {"caption_request": {key: caption_defaults[key] for key in _parameters_for("qwen_caption")},
                "settings": CaptionSettings().model_dump(), "pilot": PilotSettings().model_dump(), "sample_size": 20, "sample_seed": 0}
            result["available_to_install"] = sorted(self.configuration.get("models", {}))
            result["heavy_jobs"] = sorted(self.configuration.get("heavy_jobs", {}))
            result["heavy_job_options"] = {name: {"operation": value["operation"],
                "label": value.get("label", name), "allowed_parameters": list(value.get("allowed_parameters", [])),
                "defaults": {key: item for key, item in value.get("defaults", {}).items() if key in value.get("allowed_parameters", [])}}
                for name, value in self.configuration.get("heavy_jobs", {}).items()}
        if dataset_id:
            result["detector"] = dataset["settings"].get("detector")
            result["classifier"] = dataset["settings"].get("classifier")
            result["edr"] = dataset["settings"].get("edr")
            result["sam3_recipes"] = dataset["settings"].get("sam3_recipes", {})
            result["dataset_revision"] = dataset["revision"]
            result["guidance"] = {key: dataset["settings"].get(key, "") for key in ("context", "labelmap_glossary")}
        return result

    def state(self, actor, dataset_id=None):
        catalog = self.catalog(actor, dataset_id)
        snapshot = self._runtime.snapshot() if self._runtime else {"phase": "available", "owners": []}
        own = next((owner for owner in snapshot["owners"] if owner["owner"] == "user:" + actor.user_id), None)
        capabilities = {}
        if dataset_id:
            for operation in ("sam_point", "sam_point_auto", "sam3_text", "sam3_text_auto", "sam3_recipe_chain", "detect_full", "qwen_infer", "qwen_caption", "qwen_caption_preview", "edr_prepass"):
                try:
                    self._spec(actor, dataset_id, operation)
                    capabilities[operation] = {"configured": True}
                except (DomainError, ValueError, KeyError) as exc:
                    capabilities[operation] = {"configured": False, "reason": exc.code if isinstance(exc, DomainError) else "tool_configuration_invalid"}
        return {"phase": snapshot["phase"], "user": {key: own[key] for key in ("state", "queued", "active_request", "error")} if own else {"state": "unloaded", "queued": 0},
                "capabilities": capabilities, "sam": catalog["sam"], "detector": catalog.get("detector"), "classifier": catalog.get("classifier"), "edr": catalog.get("edr"), "qwen": catalog["qwen"]}

    def metadata(self, actor, dataset_id):
        """Metadata for existing controls, using IDs rather than server paths."""
        catalog = self.catalog(actor, dataset_id)
        dataset = self.store.get_dataset(actor, dataset_id)
        selected = catalog.get("classifier") or {}
        classifiers = [{"id": model["model_id"], "path": model["model_id"], "rel_path": model["model_id"],
            "filename": model["model_id"], "n_classes": len(model.get("labels", [])),
            "encoder_type": "trained", "encoder_model": "Saved training recipe", "labels": model.get("labels", []),
            "active": model["model_id"] == selected.get("model_id")}
            for model in catalog["models"] if model["kind"] == "classifier"]
        ready = False
        if selected and self._runtime:
            try:
                spec, _ = self._spec(actor, dataset_id, "classify_bbox")
                ready = any(owner["owner"] == "user:" + actor.user_id and owner["state"] == "ready" and owner["runtime_identity"] == spec.identity
                    for owner in self._runtime.snapshot()["owners"])
            except (DomainError, ValueError, KeyError):
                pass
        recipes = []
        for model_id in dataset["settings"].get("sam3_recipes", {}).get("model_ids", []):
            definition = self._definition(model_id)
            with self.artifacts.open(definition["files"][definition["recipe_file"]]) as stream:
                raw = stream.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise DomainError("tool_recipe_metadata_too_large", 413)
            wrapper = json.loads(raw)
            recipe = wrapper.get("recipe", wrapper)
            class_name = wrapper.get("class_name") or recipe.get("class_name")
            target_class = next((row for row in dataset["classes"] if not row["retired"] and row["name"] == class_name), None)
            recipes.append({"id": model_id, "label": wrapper.get("label") or model_id,
                "class_name": class_name, "class_id": target_class["position"] if target_class else None,
                "mode": recipe.get("mode"), "read_only": True,
                **({"clip_head": {"registered": True}} if recipe.get("clip_head") or wrapper.get("clip_head") else {}),
                "extra_classifier_id": definition["components"].get("extra_classifier")})
        return {"classifiers": classifiers, "sam3_recipes": recipes,
            "active_classifier": {"classifier_path": selected.get("model_id"), "labelmap_path": dataset_id,
                "encoder_type": "trained", "encoder_model": "Saved training recipe", "encoder_ready": ready,
                "configured": bool(selected), "loaded": ready},
            "sam_models": [{"id": model["model_id"], "key": model["model_id"], "path": model["model_id"],
                "variant": "sam3" if model["kind"] == "sam3" else "sam1", "promoted": True,
                "active": model["model_id"] == catalog["sam"]["model_id"]}
                for model in catalog["models"] if model["kind"] in {"sam1", "sam3"} and model["model_id"] in self.configuration.get("user_sam", {}).get("allowed_model_ids", [])],
            "labelmaps": [{"path": dataset_id, "rel_path": dataset_id, "filename": dataset["name"],
                "classes": [row["name"] for row in dataset["classes"]], "n_classes": len(dataset["classes"])}],
            "prepass_recipes": [{"id": model["model_id"], "name": model["model_id"], "read_only": True,
                "active": model["model_id"] == (catalog.get("edr") or {}).get("model_id")}
                for model in catalog["models"] if model["kind"] == "edr"],
            "sam_selection": catalog["sam"], "dataset_revision": dataset["revision"], "class_version": dataset["class_version"]}

    def status(self, actor, request_id):
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
        runtime = self._ensure_runtime()
        with self.store.read_connection() as con:
            grant = self.job_authority.row(con, request_id)
            if grant:
                self.store._actor(con, actor, admin=True)
                if grant['actor_id'] != actor.user_id:
                    raise RuntimeRejected('runtime_request_not_found')
        binding = self._analysis_binding(request_id)
        if binding and json.loads(binding["actor_json"])["user_id"] != actor.user_id:
            raise RuntimeRejected("runtime_request_not_found")
        try:
            status = runtime.request_status(request_id, actor_id=actor.user_id)
        except RuntimeRejected as exc:
            if not (binding or grant) or exc.code != "runtime_request_not_found":
                raise
            status = {"id": request_id, "state": "interrupted", "error": "runtime_request_not_retained", "result_available": False}
        if status["state"] in TERMINAL:
            self._terminal(request_id)
        if binding:
            binding = self._analysis_binding(request_id)
            status.update(analysis_job_id=binding["job_id"], publication_state=binding["state"])
            if binding["state"] == "complete":
                status.update(state="completed", error=None, result_available=True)
            elif binding["state"] in {"failed", "cancelled", "interrupted"}:
                status.update(state=binding["state"], error=binding["error"], result_available=False)
            elif status["state"] in TERMINAL:
                status.update(state="publishing", result_available=False)
        elif grant:
            with self.store.read_connection() as con:
                grant = self.job_authority.row(con, request_id)
            if grant['state'] in {'failed', 'cancelled', 'interrupted'}:
                status.update(state=grant['state'], error=grant['error'], result_available=False)
        return status

    def result(self, actor, request_id):
        status = self.status(actor, request_id)
        binding = self._analysis_binding(request_id)
        if binding:
            if binding["state"] != "complete":
                raise DomainError("analysis_result_unavailable", 409)
            context = json.loads(binding["context_json"])
            self.auth.require_admin(actor)
            self.analysis.status(actor, binding["job_id"])
            return {"request_id": request_id, "analysis_job_id": binding["job_id"],
                    "context": {key: context[key] for key in ("dataset_id", "image_id", "annotation_version", "ownership_generation", "editor_id", "operation")},
                    "result": json.loads(binding["publication_json"])}
        with self.store.read_connection() as con:
            grant = self.job_authority.row(con, request_id)
        if grant and status['state'] != 'completed':
            raise DomainError('analysis_result_unavailable', 409)
        result = self._ensure_runtime().result(request_id, actor_id=actor.user_id)
        context = result["context"]
        result["context"] = {key: context[key] for key in ("dataset_id", "image_id", "annotation_version", "ownership_generation", "editor_id", "operation")}
        return result

    def cancel(self, actor, request_id):
        status = self.status(actor, request_id)
        with self.store.transaction() as con:
            grant = self.job_authority.row(con, request_id)
            if grant:
                self.store._actor(con, actor, admin=True)
                if grant['actor_id'] != actor.user_id:
                    raise RuntimeRejected('runtime_request_not_found')
                if grant['state'] == 'complete' or self.job_authority.recover(con, request_id):
                    return status
                self.job_authority.terminal(con, request_id, 'cancelled', 'request_cancelled')
        try:
            result = self._ensure_runtime().cancel(request_id, actor_id=actor.user_id)
        except RuntimeRejected as exc:
            if not grant or exc.code != 'runtime_request_not_found':
                raise
        return self.status(actor, request_id) if grant else result

    def heavy(self, actor, body):
        self.auth.require_admin(actor)
        previous = self.store.raw_receipt(actor, body.operation_id)
        if not previous:
            definition = self.configuration.get("heavy_jobs", {}).get(body.job)
            if not isinstance(definition, dict):
                raise DomainError("heavy_tool_not_configured", 503)
            if set(body.parameters) - set(definition.get("allowed_parameters", [])):
                raise DomainError("tool_parameter_not_allowed", 422)
            native = definition["factory"] == NATIVE_ANALYSIS_FACTORY
            spec = AdapterSpec(definition["factory"], {**definition.get("config", {}),
                **({"allowed_parameters": definition.get("allowed_parameters", [])} if native else {}),
                "canonical_root": str(self.media_root)}, definition["device"])
        new_command = False
        def reserve(con):
            nonlocal new_command
            self.store._dataset(con, body.dataset_id)
            new_command = True
            return {"id": uuid.uuid4().hex, "state": "admitting"}
        receipt = self.store._command(actor, body.operation_id, "request_heavy_tool", body.model_dump(), reserve, admin=True)
        request_id = receipt["id"]
        if not new_command:
            try:
                return self.status(actor, request_id)
            except RuntimeRejected as exc:
                if exc.code != "runtime_request_not_found":
                    raise
                return {"id": request_id, "state": "unavailable", "error": "runtime_request_not_retained",
                        "detail": "This command will not run again. Submit a new command if another run is needed."}
        directory = None
        try:
            runtime = self._ensure_runtime()
            parameters = dict(body.parameters)
            capture_provenance = {}
            candidate_reference = None
            # Pinning a large dataset need not block another user's status,
            # inference completion or model preference reads on our lock.
            with self.artifacts.exclusive():
                directory = Path(tempfile.mkdtemp(prefix="heavy-", dir=self.media_root))
                native_config = dict(spec.config)
                injected_cache = []
                if native and definition["operation"] == "dqe_review":
                    parent_id = parameters.pop("parent_analysis_job_id", None)
                    if not isinstance(parent_id, str) or not parent_id:
                        raise DomainError("dqe_review_parent_required", 422)
                    parent = self.analysis.prepare_review_parent(actor, parent_id, directory)
                    if parent["dataset_id"] != body.dataset_id:
                        raise DomainError("tool_dataset_mismatch", 409)
                    snapshot_id = parent["snapshot_id"]
                    native_config["parent_analysis"] = parent["parent_analysis"]
                    injected_cache.extend(parent["cache_inputs"])
                    model_id = definition.get("review_model_id") or native_config.get("review_model_id")
                    model = self._definition(model_id)
                    if model["kind"] != "qwen" or model.get("adapter_path"):
                        raise DomainError("dqe_review_model_not_supported", 422)
                    for relative, digest in model["files"].items():
                        ref = self.artifacts.pin(digest, directory)
                        injected_cache.append({"namespace": "models", "relative_path": "review/" + relative,
                            "path": str(directory / ref.sha256), **ref.as_dict()})
                    native_config["review_model_directory"] = "models/review"
                    capture_provenance = {"parent_analysis_job_id": parent_id, "review_model_id": model_id,
                        "review_model_identity": self.registry.model_identity(model_id)}
                else:
                    snapshot = self.store.create_snapshot(actor, body.dataset_id, "heavy-snapshot-" + request_id, "Admin compute")
                    snapshot_id = snapshot["snapshot_id"]
                content = self.store.get_snapshot(actor, snapshot_id)["manifest"]
                content["snapshot_id"] = snapshot_id
                if native and definition["operation"] == "ingestion":
                    candidates = self.analysis.capture_incoming(actor, body.dataset_id, parameters.pop("candidate_image_ids", None),
                        "heavy-candidates-" + request_id)
                    candidate_manifest = candidates["manifest"]
                    for image in candidate_manifest["images"]:
                        view = self.media.resolve(image["media_hash"])
                        ref = self.artifacts.pin(self.media.reference(view), directory)
                        image["working_media"] = view
                        image["media_path"] = str(directory / ref.sha256)
                    candidate_path = directory / "incoming-candidates.json"
                    with candidate_path.open("x", encoding="utf-8") as stream:
                        json.dump(candidate_manifest, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
                        stream.flush()
                        os.fsync(stream.fileno())
                    candidate_path.chmod(0o400)
                    candidate_reference = {"candidate_set_id": candidates["candidate_set_id"], "dataset_id": body.dataset_id,
                        "manifest_path": str(candidate_path), "manifest_sha256": file_digest(candidate_path)}
                    capture_provenance.update(candidate_set_id=candidates["candidate_set_id"], candidate_manifest_sha256=candidate_reference["manifest_sha256"])
                if native and definition["operation"] in {"dqe", "pruning", "ingestion"}:
                    from services.hosted.salad_profiles import HostedSaladProfiles
                    profiles = HostedSaladProfiles(self.store, self.artifacts).capture(actor, snapshot_id, directory)
                    injected_cache.extend(profiles["cache_inputs"])
                    if definition["operation"] == "ingestion":
                        native_config["hosted_salad_heads"] = {head["id"]: head["head_sha256"] for head in profiles["heads"] if head["representation_scope"] == "whole_image"}
                    if definition["operation"] == "pruning":
                        native_config["parameters"] = {**native_config.get("parameters", {}), **profiles["pruning_defaults"]}
                if injected_cache:
                    cache = {}
                    for item in [*native_config.get("cache_inputs", []), *injected_cache]:
                        key = (item["namespace"], item["relative_path"])
                        previous_item = cache.get(key)
                        if previous_item and (previous_item["sha256"], previous_item["size_bytes"]) != (item["sha256"], item["size_bytes"]):
                            raise DomainError("native_cache_alias_conflict", 409)
                        cache[key] = item
                    native_config["cache_inputs"] = list(cache.values())
                    native_config["cache_input_roots"] = [*native_config.get("cache_input_roots", []), str(directory)]
                spec = AdapterSpec(spec.factory, native_config, spec.device)
                for image in content["images"]:
                    view = self.media.resolve(image["media_hash"])
                    ref = self.artifacts.pin(self.media.reference(view), directory)
                    image["working_media"] = view
                    image["media_path"] = str(directory / ref.sha256)
                manifest = directory / "canonical-snapshot.json"
                with manifest.open("x", encoding="utf-8") as stream:
                    json.dump(content, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
                    stream.flush()
                    os.fsync(stream.fileno())
                manifest.chmod(0o400)
                reference = {key: content[key] for key in ("snapshot_id", "dataset_id", "revision", "class_version", "settings_version")}
                reference.update(manifest_path=str(manifest), manifest_sha256=file_digest(manifest))
            with self._lock:
                context = RequestContext(actor.user_id, body.dataset_id, "", snapshot_id, 0, "", actor_role=actor.role,
                    session_id=actor.session_id, instance_id=actor.instance_id, operation=definition["operation"], model_config=spec.config,
                    background_job_id=request_id, background_adapter_identity=spec.identity)
                def bind_native(con, analysis):
                    self.job_authority.mint(con, actor, request_id, context, spec.identity, analysis['job_id'])
                    con.execute("INSERT INTO hosted_tool_analysis VALUES (?,?,?,?,?,'pending',NULL,NULL,NULL)",
                        (request_id, analysis["job_id"], json.dumps(asdict(actor)), str(spec.config["native_root"]), json.dumps(context.as_dict())))
                if native:
                    self.analysis.capture(actor, body.dataset_id, definition["operation"], {**native_config.get("parameters", {}), **parameters, **capture_provenance},
                        "heavy-capture-" + request_id, snapshot_id=snapshot_id, _bind_job=bind_native)
                else:
                    with self.store.transaction() as con:
                        self.job_authority.mint(con, actor, request_id, context, spec.identity)
                request_id = runtime.begin_heavy(context, spec, definition["operation"],
                    {**parameters, "canonical_snapshot": reference, **({"candidate_set": candidate_reference} if candidate_reference else {})}, drain_timeout=float(definition.get("drain_timeout", 30)),
                    timeout=float(definition.get("timeout", 3600)), request_id=request_id)
                self._pins[request_id], directory = (directory, actor.user_id), None
                # Admission has committed under the authenticated login. A
                # logout before this acknowledgement does not revoke the job.
                accepted = runtime.request_status(request_id, actor_id=actor.user_id)
                binding = self._analysis_binding(request_id)
                if binding:
                    accepted.update(analysis_job_id=binding['job_id'], publication_state=binding['state'])
                    if binding['state'] == 'complete':
                        accepted.update(state='completed', error=None, result_available=True)
                    elif accepted['state'] in TERMINAL:
                        accepted.update(state='publishing', result_available=False)
                return accepted
        except Exception as exc:
            code = exc.code if isinstance(exc, (DomainError, RuntimeRejected)) else "analysis_admission_failed"
            with self.store.transaction() as con:
                self.job_authority.terminal(con, request_id, "failed", code)
            raise
        finally:
            if directory:
                shutil.rmtree(directory)

    def close(self):
        self.paid_batches.close()
        self.caption_jobs.close()
        self.caption_batches.close()
        if self._runtime:
            self._runtime.close()
        self._reaper_stop.set()
        if self._reaper_thread:
            self._reaper_thread.join(timeout=2)
        # Native publication may be retaining large output files. Do not remove
        # its snapshot pins until that independent transaction has finished.
        for thread in list(self._publication_threads):
            thread.join(timeout=2)
        with self._lock:
            requests = list(self._pins)
        for request_id in requests:
            binding = self._analysis_binding(request_id)
            if not binding or binding["state"] == "complete":
                self._cleanup(request_id)


def install_tool_api(app, store, auth, artifacts, actor_dependency, admin_dependency, root):
    controller = ToolController(store, auth, artifacts, root)
    # Application construction attaches a gateway installed after this router.
    controller.provider_gateway = getattr(app.state, "provider_gateway", None)
    router = APIRouter(prefix="/api/hosted/tools")
    instruction_reviews = InstructionReviews(store)

    @router.get("/images/{image_id}/instruction-review")
    def inspect_instructions(image_id: str, actor=Depends(admin_dependency)):
        return instruction_reviews.inspect(actor, image_id)

    @router.post("/images/{image_id}/instruction-review")
    def review_instructions(image_id: str, body: InstructionReviewApply, actor=Depends(admin_dependency)):
        return instruction_reviews.apply(actor, image_id, body)

    def translate(callback, *args):
        try:
            return callback(*args)
        except RuntimeRejected as exc:
            status = 404 if exc.code == "runtime_request_not_found" else 429 if exc.code in {"runtime_queue_full", "runtime_payload_budget_exceeded", "runtime_owner_limit"} else 409
            raise DomainError(exc.code, status) from exc
        except (ValueError, KeyError) as exc:
            raise DomainError("tool_configuration_or_input_invalid", 422, "Check the tool settings and supplied parameters.") from exc

    @router.get("/paid-batches")
    def paid_batch_list(dataset_id: str | None = None, include_archived: bool = False, actor=Depends(actor_dependency)):
        return {"jobs": translate(controller.paid_batches.list, actor, dataset_id, include_archived)}

    @router.post("/paid-batches/preview")
    def paid_batch_preview(body: PaidBatchRequest, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.preview, actor, body)

    @router.post("/paid-batches", status_code=202)
    def paid_batch_start(body: PaidBatchRequest, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.start, actor, body)

    @router.get("/paid-batches/{job_id}")
    def paid_batch_status(job_id: str, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.status, actor, job_id)

    @router.get("/paid-batches/{job_id}/captions")
    def paid_batch_captions(job_id: str, offset: int = 0, limit: int = 25, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.proposals, actor, job_id, offset, limit)

    @router.post("/paid-batches/{job_id}/poll")
    def paid_batch_poll(job_id: str, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.poll, actor, job_id)

    @router.post("/paid-batches/{job_id}/collect")
    def paid_batch_collect(job_id: str, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.poll, actor, job_id)

    @router.post("/paid-batches/{job_id}/cancel")
    def paid_batch_cancel(job_id: str, body: PaidBatchCommand, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.cancel, actor, job_id, body.operation_id)

    @router.post("/paid-batches/{job_id}/archive")
    def paid_batch_archive(job_id: str, body: PaidBatchArchive, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.archive, actor, job_id, body.operation_id, body.archived)

    @router.post("/paid-batches/{job_id}/verify")
    def paid_batch_verify(job_id: str, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.verify, actor, job_id)

    @router.post("/paid-batches/{job_id}/resume", status_code=202)
    def paid_batch_resume(job_id: str, body: PaidBatchResume, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.resume, actor, job_id, body)

    @router.post("/paid-batches/{job_id}/recover", status_code=202)
    def paid_batch_recover(job_id: str, body: RecoverBatch, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.recover_remote, actor, job_id, body.remote_id, body.operation_id)

    @router.post("/paid-batches/{job_id}/apply")
    def paid_batch_apply(job_id: str, body: PaidBatchApply, actor=Depends(actor_dependency)):
        return translate(controller.paid_batches.apply, actor, job_id, body)

    @router.get("/catalog")
    def catalog(dataset_id: str | None = None, actor=Depends(actor_dependency)):
        return translate(controller.catalog, actor, dataset_id)

    @router.get("/state")
    def state(dataset_id: str | None = None, actor=Depends(actor_dependency)):
        return translate(controller.state, actor, dataset_id)

    @router.get("/metadata")
    def metadata(dataset_id: str, actor=Depends(actor_dependency)):
        return translate(controller.metadata, actor, dataset_id)

    @router.post("/models/{model_id}/install")
    def install(model_id: str, body: InstallModel, actor=Depends(admin_dependency)):
        return translate(controller.install_model, actor, model_id, body.operation_id)

    @router.post("/sam")
    def select_sam(body: ModelSelection, actor=Depends(actor_dependency)):
        return translate(controller.select_sam, actor, body)

    @router.post("/datasets/{dataset_id}/detector")
    def select_detector(dataset_id: str, body: DatasetModelSelection, actor=Depends(admin_dependency)):
        return translate(controller.select_detector, actor, dataset_id, body)

    @router.post("/datasets/{dataset_id}/classifier")
    def select_classifier(dataset_id: str, body: DatasetModelSelection, actor=Depends(admin_dependency)):
        return translate(controller.select_dataset_model, actor, dataset_id, body, "classifier")

    @router.post("/datasets/{dataset_id}/sam3-recipes")
    def select_sam_recipes(dataset_id: str, body: DatasetRecipeSelection, actor=Depends(admin_dependency)):
        return translate(controller.select_sam_recipes, actor, dataset_id, body)

    @router.get("/datasets/{dataset_id}/cascades")
    def cascades(dataset_id: str, actor=Depends(actor_dependency)):
        return translate(controller.cascades, actor, dataset_id)

    @router.get("/datasets/{dataset_id}/cascades/{cascade_id}")
    def cascade(dataset_id: str, cascade_id: str, actor=Depends(actor_dependency)):
        return translate(controller.cascades, actor, dataset_id, cascade_id)

    @router.post("/datasets/{dataset_id}/cascades")
    def save_cascade(dataset_id: str, body: CascadeSave, actor=Depends(actor_dependency)):
        return translate(controller.save_cascade, actor, dataset_id, body)

    @router.post("/datasets/{dataset_id}/cascades/{cascade_id}/delete")
    def delete_cascade(dataset_id: str, cascade_id: str, body: CascadeDelete, actor=Depends(actor_dependency)):
        return translate(controller.delete_cascade, actor, dataset_id, cascade_id, body)

    @router.post("/datasets/{dataset_id}/edr")
    def select_edr(dataset_id: str, body: DatasetModelSelection, actor=Depends(admin_dependency)):
        return translate(controller.select_dataset_model, actor, dataset_id, body, "edr")

    @router.post("/request", status_code=202)
    def submit(body: ToolRequest, actor=Depends(actor_dependency)):
        return translate(controller.submit, actor, body)

    @router.get("/requests/{request_id}")
    def status(request_id: str, actor=Depends(actor_dependency)):
        return translate(controller.status, actor, request_id)

    @router.get("/requests/{request_id}/result")
    def result(request_id: str, actor=Depends(actor_dependency)):
        return translate(controller.result, actor, request_id)

    @router.post("/requests/{request_id}/cancel")
    def cancel(request_id: str, actor=Depends(actor_dependency)):
        return translate(controller.cancel, actor, request_id)

    @router.post("/heavy", status_code=202)
    def heavy(body: HeavyRequest, actor=Depends(admin_dependency)):
        return translate(controller.heavy, actor, body)

    @router.post("/caption-jobs", status_code=202)
    def caption_start(body: CaptionJobRequest, actor=Depends(actor_dependency)):
        return translate(controller.caption_jobs.start, actor, body)

    @router.get("/caption-jobs")
    def caption_list(actor=Depends(actor_dependency)):
        return translate(controller.caption_jobs.list, actor)

    @router.get("/caption-jobs/{job_id}")
    @router.get("/caption-jobs/{job_id}/result")
    def caption_status(job_id: str, actor=Depends(actor_dependency)):
        return translate(controller.caption_jobs.status, actor, job_id)

    @router.post("/caption-jobs/{job_id}/cancel")
    def caption_cancel(job_id: str, body: CaptionJobCommand, actor=Depends(actor_dependency)):
        return translate(controller.caption_jobs.cancel, actor, job_id, body.operation_id)

    @router.post("/caption-jobs/{job_id}/resume", status_code=202)
    def caption_resume(job_id: str, body: CaptionJobRequest, actor=Depends(actor_dependency)):
        return translate(controller.caption_jobs.resume, actor, job_id, body)

    @router.post("/caption-batches", status_code=202)
    def caption_batch(body: CaptionBatchRequest, actor=Depends(admin_dependency)):
        return translate(controller.caption_batches.start, actor, body)

    @router.get("/caption-batches")
    def caption_batch_list(actor=Depends(admin_dependency)):
        return translate(controller.caption_batches.list, actor)

    @router.get("/caption-batches/{job_id}")
    @router.get("/caption-batches/{job_id}/result")
    def caption_batch_status(job_id: str, actor=Depends(admin_dependency)):
        return translate(controller.caption_batches.status, actor, job_id)

    @router.get("/caption-batches/{job_id}/captions")
    def caption_batch_captions(job_id: str, offset: int = 0, limit: int = 25, actor=Depends(admin_dependency)):
        return translate(controller.caption_batches.captions, actor, job_id, offset, limit)

    @router.post("/caption-batches/{job_id}/cancel")
    def caption_batch_cancel(job_id: str, body: CaptionJobCommand, actor=Depends(admin_dependency)):
        return translate(controller.caption_batches.cancel, actor, job_id, body.operation_id)

    @router.post("/caption-batches/{job_id}/resume", status_code=202)
    def caption_batch_resume(job_id: str, body: CaptionJobCommand, actor=Depends(admin_dependency)):
        return translate(controller.caption_batches.resume, actor, job_id, body.operation_id)

    @router.get("/caption-batches/{job_id}/images/{image_id}/proposal")
    def caption_batch_proposal(job_id: str, image_id: str, actor=Depends(admin_dependency)):
        return translate(controller.caption_batches.proposal, actor, job_id, image_id)

    @router.post("/caption-batches/{job_id}/images/{image_id}/apply")
    def caption_batch_apply(job_id: str, image_id: str, body: CaptionProposalApply, actor=Depends(admin_dependency)):
        return translate(controller.caption_batches.apply, actor, job_id, image_id, body)

    @router.get("/runtime")
    def runtime_status(actor=Depends(admin_dependency)):
        auth.require_admin(actor)
        return controller._runtime.snapshot() if controller._runtime else {"phase": "available", "owners": []}

    app.include_router(router)
    previous_lifespan = app.router.lifespan_context
    @asynccontextmanager
    async def lifespan(application):
        async with previous_lifespan(application):
            try:
                yield
            finally:
                controller.close()
    app.router.lifespan_context = lifespan
    app.state.tools = controller
    return controller
