"""Saved EDR execution inside one serial dataset worker.

Native prepass, crop/classifier gates, feature computation and calibrated
ensemble scoring are reused. No original dataset, legacy HTTP or app globals
are consulted. All files and component identities come from the registry.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import tempfile
import threading
from types import FunctionType, SimpleNamespace

import numpy as np


def _member(root, value):
    from services.hosted.config import validate_model_member
    validate_model_member(value)
    path = root / value
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("hosted_edr_artifact_missing")
    return path


def _bounded_json(path, limit=8 * 1024 * 1024):
    from services.hosted.artifacts import open_regular
    with open_regular(path) as reader:
        data = reader.read(limit + 1)
    if len(data) > limit:
        raise ValueError("hosted_edr_metadata_limit_exceeded")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("hosted_edr_metadata_invalid")
    return value


def _coerce(value, fallback, *, minimum=None, maximum=None, integer=False):
    try:
        result = int(value) if integer else float(value)
        if not math.isfinite(result):
            raise ValueError()
    except (TypeError, ValueError, OverflowError):
        result = fallback
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _run_scoring_main(module, arguments, overrides=None):
    """Invoke the existing scorer without changing process-global argv."""
    class Parser(argparse.ArgumentParser):
        def parse_args(self, *unused, **ignored):
            return SimpleNamespace(**arguments)
    namespace = dict(vars(module))
    namespace.update(overrides or {})
    namespace["argparse"] = SimpleNamespace(ArgumentParser=Parser)
    main = FunctionType(module.main.__code__, namespace, "main")
    main()


class DatasetEdrAdapter:
    def __init__(self, *, config, device):
        from models.schemas import QwenPrepassRequest
        from services.hosted.qwen_adapters import _verify_snapshot
        self.config = json.loads(json.dumps(config, allow_nan=False))
        self.device, self.definition = device, self.config["model"]
        self.root = _verify_snapshot(self.definition["model_path"], self.definition["files"])
        saved = _bounded_json(_member(self.root, self.definition["recipe_file"]))
        stored_config = dict(saved.get("config") or saved)
        # Canonical saved recipes also contain registry fingerprints, source
        # paths and report metadata. The existing request schema ignores those
        # fields; retain the bundle bytes but never resolve those source paths.
        self.recipe = {key: value for key, value in stored_config.items() if key in QwenPrepassRequest.model_fields}
        self.metadata_fields = sorted(set(stored_config) - set(self.recipe))
        if saved.get("glossary") and not self.recipe.get("labelmap_glossary"):
            self.recipe["labelmap_glossary"] = saved["glossary"]
        self.labelmap = list(self.definition["labels"])
        if not self.labelmap or len(set(self.labelmap)) != len(self.labelmap):
            raise ValueError("hosted_edr_labelmap_invalid")
        if self.recipe.get("labelmap") and self.recipe["labelmap"] != self.labelmap:
            raise ValueError("hosted_edr_labelmap_changed")
        # Canonical packages intentionally score the raw candidate pool through
        # their calibrated ensemble rather than an earlier classifier gate.
        self.ensemble = self.definition.get("ensemble_model_file") is not None
        if self.recipe.get("edr_package_id") and self.recipe.get("edr_package_apply_ensemble") is False:
            self.ensemble = False
        requires_ensemble = self.recipe.get("ensemble_enabled") or (self.recipe.get("edr_package_id") and
            self.recipe.get("edr_package_apply_ensemble") is not False)
        if requires_ensemble and not self.ensemble:
            raise ValueError("hosted_edr_ensemble_required")
        values = dict(self.recipe, prepass_only=True, labelmap=self.labelmap,
                      dataset_id=self.config["dataset_id"], image_base64=None, image_token="hosted-edr-image")
        if self.ensemble:
            values.update(prepass_keep_all=True, prepass_finalize=False)
        if self.recipe.get("edr_package_id"):
            values.update(sam3_text_synonym_budget=0, prepass_caption=False)
        self.request = QwenPrepassRequest(**values)
        self.components = self.definition.get("component_models", {})
        for name, enabled in (("yolo", self.request.enable_yolo), ("rfdetr", self.request.enable_rfdetr),
                              ("sam3", self.request.enable_sam3_text or self.request.enable_sam3_similarity),
                              ("classifier", self.ensemble or not self.request.prepass_keep_all or self.request.prepass_finalize)):
            if enabled and name not in self.components:
                raise ValueError("hosted_edr_component_required:" + name)
        for name in ("yolo", "rfdetr"):
            if name in self.components and self.components[name].get("labels") != self.labelmap:
                raise ValueError("hosted_edr_detector_labelmap_changed:" + name)
        from utils.glossary import _parse_glossary_mapping, _normalize_synonym_list
        glossary = _parse_glossary_mapping(self.request.labelmap_glossary or "", self.labelmap)
        complete_glossary = all(_normalize_synonym_list(glossary.get(label, [])) for label in self.labelmap)
        needs_qwen = self.request.prepass_caption or (self.request.enable_sam3_text and
            self.request.sam3_text_synonym_budget != 0 and not complete_glossary)
        qwen_config = self.definition.get("qwen") or self.config.get("qwen")
        if needs_qwen and not qwen_config:
            raise ValueError("hosted_edr_qwen_required")
        if needs_qwen:
            for field in ("model_id", "prepass_caption_model_id"):
                name = getattr(self.request, field)
                if name and name not in {"active", "auto", "same"} and name not in qwen_config["models"]:
                    raise ValueError("hosted_edr_qwen_association_unmapped:" + field)
        # The legacy cache key omits model identity. Keep the exact expansion
        # algorithm, but give each immutable dataset recipe its own cache.
        from services import sam3_synonyms
        namespace = dict(vars(sam3_synonyms))
        namespace.update(_SAM3_SYNONYM_CACHE={}, _SAM3_SYNONYM_CACHE_ORDER=deque())
        for name, value in vars(sam3_synonyms).items():
            if isinstance(value, FunctionType) and value.__module__ == sam3_synonyms.__name__:
                clone = FunctionType(value.__code__, namespace, name, value.__defaults__, value.__closure__)
                clone.__kwdefaults__ = value.__kwdefaults__
                namespace[name] = clone
        self.generate_synonyms = namespace["_agent_generate_sam3_synonyms"]
        self.detectors = {}
        self.classifier = self.sam = self.sam_prompts = self.qwen = None
        self.image = self.payload = self.context = None
        self.events = []
        self.evidence_size = 0
        self.execution_lock = threading.Lock()
        self.closed = False
        self.feature_contract = None
        if self.ensemble:
            self.feature_contract = _bounded_json(_member(self.root, self.definition["feature_contract_file"]))
            if not self.feature_contract.get("feature_schema_hash") or self.feature_contract.get("labelmap") != self.labelmap:
                raise ValueError("hosted_edr_feature_contract_required")
            self.ensemble_path = _member(self.root, self.definition["ensemble_model_file"])
            self.ensemble_meta_path = _member(self.root, self.definition["ensemble_meta_file"])
            self._validate_policy_paths(_bounded_json(self.ensemble_meta_path), self.ensemble_meta_path.parent)

    def _validate_policy_paths(self, meta, base):
        """Every optional XGB/learned policy dependency belongs to this bundle."""
        refs = []
        models = meta.get("split_head", {}).get("models", {}) if isinstance(meta.get("split_head"), dict) else {}
        refs.extend(models.values())
        for key in ("sam3_text_quality", "sam3_similarity_quality", "selected_policy_layer"):
            item = meta.get(key)
            if isinstance(item, dict):
                refs.extend(item.get(field) for field in ("model_path", "meta_path") if item.get(field))
        # Native policy metadata versions also store specialists directly.
        if isinstance(meta.get("specialist_models"), dict):
            refs.extend(meta["specialist_models"].values())
        for value in refs:
            if not isinstance(value, str) or Path(value).is_absolute():
                raise ValueError("hosted_edr_policy_path_invalid")
            path = (base / value).resolve(strict=True)
            if not path.is_relative_to(self.root) or path.relative_to(self.root).as_posix() not in self.definition["files"]:
                raise ValueError("hosted_edr_policy_path_invalid")
        selected = meta.get("selected_policy_layer")
        if isinstance(selected, dict):
            if not all(selected.get(key) for key in ("variant", "model_path", "meta_path")):
                raise ValueError("hosted_edr_selected_policy_invalid")
            _bounded_json((base / selected["meta_path"]).resolve(strict=True))

    def _record(self, value):
        text = json.dumps(value, allow_nan=False, default=lambda item: item.item() if isinstance(item, np.generic) else str(item))
        self.evidence_size += len(text.encode())
        if self.evidence_size > 16 * 1024 * 1024:
            raise ValueError("hosted_edr_evidence_limit_exceeded")
        self.events.append(json.loads(text))

    def _classifier(self):
        if self.classifier is None:
            from services.hosted.assistance_adapters import ClassifierRuntime
            self.classifier = ClassifierRuntime(self.components["classifier"], self.device)
        return self.classifier

    def _sam_runtime(self):
        if self.sam is None:
            from services.hosted.adapters import _build_sam
            from services.hosted.assistance_adapters import Sam3PromptRuntime
            predictor, network = _build_sam(self.components["sam3"], self.device)
            self.sam = predictor, network
            self.sam_prompts = Sam3PromptRuntime(network, self.device)
        return self.sam[1], self.sam_prompts.processor, self.device

    def _qwen_runtime(self):
        if self.qwen is None:
            from services.hosted.qwen_adapters import QwenRuntime
            configuration = self.definition.get("qwen") or self.config.get("qwen")
            if not configuration:
                raise ValueError("hosted_edr_qwen_required")
            configuration = dict(configuration, actor_id="dataset:" + self.config["dataset_id"])
            self.qwen = QwenRuntime(configuration, configuration.get("device", self.device))
        return self.qwen

    def _generate_synonyms(self, prompt, *, max_new_tokens=384, use_system_prompt=False):
        worker = self._qwen_runtime()
        with worker.execution_lock:
            worker.events.clear(); worker.io.clear(); worker.progress.clear()
            worker.io_bytes = worker.events_dropped = 0
            worker.cancelled.clear()
            try:
                model_id = self.request.model_id or worker.default_model
                if model_id in {"active", "auto", "same"}:
                    model_id = worker.default_model
                result = worker.pipeline["_run_qwen_text_inference"](prompt, model_id_override=model_id,
                    max_new_tokens=max_new_tokens, system_prompt_override=None if use_system_prompt else "")
                self._record({"type": "synonym_model_io", "events": worker.io.copy()})
                return result[0]
            except Exception as exc:
                self.stage_failures.append("synonyms:" + str(exc))
                raise
            finally:
                worker.image = None

    def _detector(self, *, mode, sahi, conf=None, iou=None, merge_iou=None, max_det=None, **unused):
        from services.hosted.adapters import DatasetDetectorAdapter
        from utils.coords import _agent_det_payload
        if mode not in self.detectors:
            self.detectors[mode] = DatasetDetectorAdapter(config={"dataset_id": self.config["dataset_id"],
                "media_root": self.config["media_root"], "model": self.components[mode]}, device=self.device)
        request = {key: self.payload[key] for key in ("media_path", "media_sha256")}
        for key, value in (("conf", conf), ("iou", iou), ("merge_iou", merge_iou), ("max_det", max_det)):
            if value is not None:
                request[key] = value
        operation = "detect_windowed" if sahi.get("enabled") else "detect_full"
        if operation == "detect_windowed":
            request.update(slice_size=sahi["slice_size"], overlap=sahi["overlap"])
        result = self.detectors[mode].execute(operation, request, self.context)
        entries = []
        for det in result["detections"]:
            x, y, width, height = det["bbox"]
            entries.append(_agent_det_payload(self.image.width, self.image.height,
                (x, y, x+width, y+height), label=det.get("class_name"),
                class_id=det.get("class_id"), score=det.get("score"), source=mode))
        return {"detections": entries, "warnings": result.get("warnings")}

    def _sam_payloads(self, *, full_img, crop_img, label, score_thr, mask_threshold,
                      max_results, window_xyxy=None, prompt=None, exemplar_boxes=None, **unused):
        from utils.coords import _agent_det_payload, _resolve_agent_bbox_xyxy, _yolo_to_xyxy
        self._sam_runtime()
        visual = exemplar_boxes is not None
        options = {"threshold": score_thr if score_thr is not None else (.55 if visual else .2),
                   "mask_threshold": mask_threshold if mask_threshold is not None else .2,
                   "max_results": max_results or 10000}
        if visual:
            boxes = []
            for entry in exemplar_boxes:
                box = _resolve_agent_bbox_xyxy(entry, full_img.width, full_img.height,
                    window_bbox_2d=entry.get("window_bbox_2d") or unused.get("window_bbox_2d"))
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                ox, oy = window_xyxy[:2] if window_xyxy else (0, 0)
                boxes.append([x1-ox, y1-oy, x2-x1, y2-y1])
            if not boxes:
                return []
            options.update(bboxes=boxes, bbox_labels=[True] * len(boxes))
        else:
            options["text_prompt"] = prompt
        result = self.sam_prompts.predict(crop_img, options, visual=visual, allow_partial_seeds=bool(window_xyxy))
        ox, oy = window_xyxy[:2] if window_xyxy else (0, 0)
        entries = []
        for det in result["detections"]:
            x1, y1, x2, y2 = _yolo_to_xyxy(crop_img.width, crop_img.height, det["bbox"])
            entries.append(_agent_det_payload(full_img.width, full_img.height, (x1+ox, y1+oy, x2+ox, y2+oy),
                label=label, class_id=self.labelmap.index(label), score=det.get("score"),
                source="sam3_similarity" if visual else "sam3_text", window=window_xyxy))
        return entries if visual else (entries, label, self.labelmap.index(label))

    def _gate(self):
        from services.hosted.edr_classifier_gate import bind_gate
        from services.hosted.assistance_adapters import predict_probabilities
        from services.classifier import _predict_proba_batched_impl
        def encode(items, *, head, geometry_records=None, **unused):
            return self._classifier().encode_items(items, head=head, geometry_records=geometry_records)
        def close(item):
            for view in item if isinstance(item, (tuple, list)) else [item]:
                view.close()
        return bind_gate({
            "PREPASS_CLASSIFIER_STRICT_MIN_PROB": .65, "PREPASS_CLASSIFIER_STRICT_MARGIN": .15,
            "PREPASS_CLASSIFIER_STRICT_BG_MARGIN": .15, "PREPASS_STRICT_SAM3_MIN_SCORE": .7,
            "_coerce_float": _coerce, "_coerce_int": lambda *a, **kw: _coerce(*a, **kw, integer=True),
            "_clip_head_predict_proba": predict_probabilities, "_close_crop_item": close,
            "_encode_embedding_items_for_head": encode, "_predict_proba_batched_impl": _predict_proba_batched_impl,
            "_resolve_classifier_batch": lambda: 32,
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        })

    def _features_and_score(self, detections, provenance):
        from services.hosted.edr_features import run_feature_builder
        from services.hosted.assistance_adapters import predict_probabilities
        from utils.coords import _qwen_bbox_to_xyxy, _xyxy_to_yolo_norm
        if not detections:
            return [], {"input_count": 0, "accepted_count": 0}
        classifier, contract = self._classifier(), self.feature_contract
        if len(detections) > 10000:
            raise ValueError("hosted_edr_candidate_limit_exceeded")
        with tempfile.TemporaryDirectory(prefix="hosted-edr-") as directory:
            root = Path(directory)
            image_path, label_path = root / "image.png", root / "image.txt"
            self.image.save(image_path)
            label_lines = []
            bindings = self.config.get("class_bindings") or {}
            ids = {entry["id"]: label for label, entry in bindings.items()}
            for item in self.payload.get("annotation_objects", []):
                label = ids.get(item.get("class_id"))
                box = item.get("bbox")
                if label not in self.labelmap or not isinstance(box, list) or len(box) != 4:
                    continue
                x, y, width, height = map(float, box)
                values = _xyxy_to_yolo_norm(self.image.width, self.image.height, x, y, x+width, y+height)
                label_lines.append(str(self.labelmap.index(label)) + " " + " ".join(map(str, values)))
            label_path.write_text("\n".join(label_lines))
            input_path, features_path, output_path = root / "input.jsonl", root / "features.npz", root / "scored.jsonl"
            input_path.write_text(json.dumps({"image": "image.png", "detections": detections, "provenance": provenance}) + "\n")
            options = dict(input=str(input_path), output=str(features_path), dataset=self.config["dataset_id"],
                classifier_id="registered", require_classifier=True, sam3_iou=.5, support_iou=contract.get("support_iou", .5),
                batch_size=32, device=str(self.device), min_crop_size=contract.get("min_crop_size", 4), max_images=1,
                term_hash_dim=contract.get("term_hash_dim", 256), context_radius=contract.get("context_radius", .075),
                embed_proj_dim=contract.get("embed_proj_dim", 0), embed_proj_seed=contract.get("embed_proj_seed", 42),
                image_embed_proj_dim=contract.get("image_embed_proj_dim", 0), image_embed_proj_seed=contract.get("image_embed_proj_seed", 4242),
                embed_l2_normalize=contract.get("embed_l2_normalize", True))
            facade = SimpleNamespace(UPLOAD_ROOT=root, CLASSIFIER_ALLOWED_EXTS={".pkl"},
                _resolve_agent_clip_classifier_path_impl=lambda *a, **kw: root / "registered.pkl",
                _path_is_within_root_impl=lambda *a: True, _load_clip_head_from_classifier=lambda path: classifier.head,
                _encode_pil_batch_for_head=lambda images, **kw: classifier.encode_crops(images, head=kw["head"]),
                _clip_head_predict_proba=predict_probabilities, _qwen_bbox_to_xyxy=_qwen_bbox_to_xyxy)
            run_feature_builder(options=options, api=facade, labelmap=self.labelmap, image_path=image_path, label_path=label_path)
            with np.load(features_path, allow_pickle=True) as features:
                schema = str(features["feature_schema_hash"].item())
                if schema != contract["feature_schema_hash"]:
                    raise ValueError("hosted_edr_feature_schema_changed")
                if features["X"].shape[0] != len(detections) or not np.isfinite(features["X"]).all():
                    raise ValueError("hosted_edr_feature_rows_invalid")
                feature_shape = list(features["X"].shape)
            arguments = dict(model=str(self.ensemble_path), meta=str(self.ensemble_meta_path), data=str(features_path),
                             output=str(output_path), threshold=None)
            if self.ensemble_path.suffix == ".pt":
                from tools import score_ensemble_candidates as scorer
                arguments["temperature"] = None
            else:
                from tools import score_ensemble_candidates_xgb as scorer
                arguments.update(policy_json=None, force_hand_policy=False)
            overrides = None
            if self.ensemble_path.suffix != ".pt":
                from tools import policy_runtime
                namespace = dict(vars(policy_runtime))
                for name, value in vars(policy_runtime).items():
                    if isinstance(value, FunctionType) and value.__module__ == policy_runtime.__name__:
                        clone = FunctionType(value.__code__, namespace, name, value.__defaults__, value.__closure__)
                        clone.__kwdefaults__ = value.__kwdefaults__
                        namespace[name] = clone
                def resolve(raw, *, base_dir):
                    if not raw:
                        return None
                    if Path(raw).is_absolute():
                        raise ValueError("hosted_edr_policy_path_invalid")
                    path = (base_dir / raw).resolve(strict=True)
                    if not path.is_relative_to(self.root) or path.relative_to(self.root).as_posix() not in self.definition["files"]:
                        raise ValueError("hosted_edr_policy_path_invalid")
                    return path
                namespace["_resolve_model_path"] = resolve
                overrides = {key: namespace[key] for key in ("predict_base_probabilities", "load_selected_policy")}
            _run_scoring_main(scorer, arguments, overrides)
            scored = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
            if len(scored) != len(detections):
                raise ValueError("hosted_edr_score_rows_changed")
            accepted = []
            for original, row in zip(detections, scored, strict=True):
                if not math.isfinite(float(row["ensemble_prob"])):
                    raise ValueError("hosted_edr_score_invalid")
                if row["ensemble_accept"]:
                    accepted.append({**original, **row})
            return accepted, {"input_count": len(scored), "accepted_count": len(accepted),
                "feature_schema_hash": schema, "feature_shape": feature_shape, "scores": scored}

    def execute(self, operation, payload, context):
        if self.closed or operation != "edr_prepass" or context["dataset_id"] != self.config["dataset_id"]:
            raise ValueError("hosted_edr_context_invalid")
        if payload.get("recipe_id") != self.definition["model_id"]:
            raise ValueError("hosted_edr_recipe_changed")
        from services.hosted.adapters import _image
        from services.prepass import (_agent_deep_prepass_cleanup_impl, _agent_run_deep_prepass_part_a_impl,
            _agent_run_deep_prepass_impl, _agent_select_similarity_exemplars)
        from services.prepass_provenance import _agent_attach_provenance, _agent_finalize_provenance
        from services.prepass_similarity import _agent_run_similarity_global, _agent_run_similarity_expansion
        from services.prepass_windows import _agent_sam3_text_windows
        from services.sam3_synonyms import _sam3_prompt_variants
        from utils.glossary import _glossary_label_key
        from utils.coords import _normalize_window_xyxy
        from services.qwen import _extract_balanced_json
        from utils.classifier_utils import _clip_head_classes, _is_background_class_name
        with self.execution_lock:
            image, _ = _image(payload, self.config)
            self.image, self.payload, self.context = image, payload, context
            self.events, self.evidence_size = [], 0
            self.stage_failures = []
            request = self.request.model_copy(deep=True)
            try:
                def part_a(req, **kwargs):
                    return _agent_run_deep_prepass_part_a_impl(req, **kwargs,
                        run_detector_fn=self._detector, attach_provenance_fn=_agent_attach_provenance,
                        generate_sam3_synonyms_fn=self.generate_synonyms, generate_text_fn=self._generate_synonyms,
                        extract_json_fn=_extract_balanced_json, default_synonyms={}, label_key_fn=_glossary_label_key,
                        sam3_text_windows_fn=_agent_sam3_text_windows, ensure_sam3_text_runtime_fn=self._sam_runtime,
                        normalize_window_xyxy_fn=_normalize_window_xyxy, sam3_prompt_variants_fn=_sam3_prompt_variants,
                        sam3_text_payloads_fn=self._sam_payloads, grid_overlap_ratio_default=.2,
                        active_sam3_score_thr=_coerce(request.sam3_score_thr, .5, minimum=0, maximum=1),
                        active_sam3_mask_thr=_coerce(request.sam3_mask_threshold, .5, minimum=0, maximum=1))
                def cleanup(req, **kwargs):
                    return _agent_deep_prepass_cleanup_impl(req, **kwargs,
                        resolve_classifier_path_fn=lambda value: "registered", load_classifier_head_fn=lambda path: self._classifier().head,
                        active_classifier_head=lambda: self._classifier().head,
                        background_from_head_fn=lambda head: [label for label in _clip_head_classes(head) if _is_background_class_name(label)] if head else [],
                        sanitize_fn=self._gate(), default_iou=.75)
                def exemplars(req, **kwargs):
                    return _agent_select_similarity_exemplars(req.similarity_min_exemplar_score if req.similarity_min_exemplar_score is not None else .6,
                        **kwargs, max_per_label=req.similarity_exemplar_count, strategy=req.similarity_exemplar_strategy,
                        seed=req.similarity_exemplar_seed, exemplar_fraction=req.similarity_exemplar_fraction,
                        exemplar_min=req.similarity_exemplar_min, exemplar_max=req.similarity_exemplar_max,
                        source_quota=req.similarity_exemplar_source_quota)
                def similarity(req, **kwargs):
                    return _agent_run_similarity_global(req, **kwargs, similarity_payloads_fn=self._sam_payloads,
                                                       ensure_sam3_text_runtime_fn=self._sam_runtime)
                def windows(req, **kwargs):
                    return _agent_run_similarity_expansion(req, **kwargs, similarity_payloads_fn=self._sam_payloads,
                        ensure_sam3_text_runtime_fn=self._sam_runtime, grid_overlap_ratio_default=.2)
                result = _agent_run_deep_prepass_impl(request, pil_img=image, image_token="hosted-edr-image",
                    labelmap=self.labelmap, glossary=request.labelmap_glossary or "", run_part_a_fn=part_a, cleanup_fn=cleanup,
                    select_exemplars_fn=exemplars, run_similarity_global_fn=similarity,
                    run_similarity_windowed_fn=windows, finalize_provenance_fn=_agent_finalize_provenance,
                    trace_writer=self._record)
                if self.stage_failures or any("failed" in warning or "type_error" in warning for warning in result["warnings"]):
                    raise ValueError("hosted_edr_required_stage_failed:" + ";".join(result["warnings"] + self.stage_failures))
                detections = result["detections"]
                if self.ensemble:
                    detections, scores = self._features_and_score(detections, result["provenance"])
                    self._record({"type": "ensemble_scoring", **scores})
                recipe_caption = None
                if request.prepass_caption:
                    caption_text, caption_windows = self._caption(request, detections)
                    recipe_caption = {"caption": caption_text, "windows": caption_windows}
                from services.hosted.edr_classifier_gate import register_clusters
                from services.prepass_grid import _agent_grid_spec_for_payload
                detections = register_clusters(detections, width=image.width, height=image.height,
                    labelmap=self.labelmap, gate=self._gate(), grid=_agent_grid_spec_for_payload(request, image.width, image.height))
                if request.prepass_finalize:
                    from services.detector_merge import _agent_merge_detections
                    head = self._classifier().head
                    detections, rejected = self._gate()(detections, pil_img=image, classifier_head=head,
                        img_w=image.width, img_h=image.height, labelmap=self.labelmap,
                        background=[label for label in _clip_head_classes(head) if _is_background_class_name(label)])
                    detections = _agent_merge_detections(detections, iou_thr=request.iou if request.iou is not None else .5,
                        max_det=None, cross_iou=request.cross_class_dedupe_iou if request.cross_class_dedupe_enabled else None)
                    self._record({"type": "finalize", "rejected": rejected, "count": len(detections)})
                bindings = self.config.get("class_bindings", {})
                for det in detections:
                    label = det.get("label")
                    if label not in bindings:
                        raise ValueError("hosted_edr_label_unmapped")
                    binding = bindings[label]
                    det.update(label=binding["name"], class_id=binding["position"], stable_class_id=binding["id"])
                return {"detections": detections, "warnings": result["warnings"] or None, "caption": None,
                        "trace": [{"step_id": 1, "phase": "merge", "summary": "deep_prepass_complete",
                                   "counts": {"detections": len(detections)}}],
                        "hosted_evidence": {"events": self.events.copy(), "provenance": result["provenance"],
                                            "recipe_id": self.definition["model_id"], "non_runtime_metadata_fields": self.metadata_fields,
                                            "recipe_caption": recipe_caption}}
            finally:
                image.close()
                self.image = self.payload = self.context = None

    def _caption(self, request, detections):
        from models.schemas import QwenCaptionRequest
        from services.prepass import _agent_run_deep_prepass_caption_impl, _agent_label_counts_summary
        from services.qwen import _sanitize_qwen_caption
        from services.prepass_grid import _agent_grid_cell_for_window_bbox
        from utils.coords import _qwen_bbox_to_xyxy, _xyxy_to_qwen_bbox
        from fastapi import HTTPException
        worker = self._qwen_runtime()
        caption_model_id = request.model_id or worker.default_model
        if caption_model_id in {"active", "auto", "same"}:
            caption_model_id = worker.default_model
        def caption(payload):
            result = worker.execute("qwen_caption", self.image, payload.model_dump(),
                                    {"actor_id": "dataset:" + self.config["dataset_id"]})
            self._record({"type": "caption_model_io", **result.pop("hosted_evidence")})
            return SimpleNamespace(**result)
        return _agent_run_deep_prepass_caption_impl(request, pil_img=self.image, image_token="hosted-edr-image",
            detections=detections, model_id_override=caption_model_id, glossary=request.labelmap_glossary,
            grid_for_log=None, caption_request_cls=QwenCaptionRequest, qwen_caption_fn=caption,
            sanitize_caption_fn=_sanitize_qwen_caption, label_counts_fn=_agent_label_counts_summary,
            qwen_bbox_to_xyxy_fn=_qwen_bbox_to_xyxy, xyxy_to_bbox_fn=_xyxy_to_qwen_bbox,
            grid_cell_for_window_bbox_fn=_agent_grid_cell_for_window_bbox, readable_format_bbox_fn=str,
            unload_non_qwen_fn=worker._forbidden_global_unload, caption_window_hook=worker.window_hook,
            http_exception_cls=HTTPException, http_503_code=503, trace_writer=self._record)

    def close(self):
        for adapter in [*self.detectors.values(), self.classifier, self.sam_prompts, self.qwen]:
            if adapter is not None:
                adapter.close()
        self.detectors.clear()
        self.sam = self.image = self.payload = self.context = None
        self.closed = True
