"""Saved SAM3 step recipes in one user-private sequential runtime.

The legacy chain's ordering, class/group deduplication and temporary classifier
ranking are retained. Its truncated per-recipe executor is repaired here with
explicit seed, trained-head, visual expansion and final-head stages. No global
app state, caller-selected files or annotation writes are used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import threading
from types import FunctionType, MethodType

import numpy as np

from models.schemas import AgentApplyImageChainRequest
from services.prepass_recipes import _classify_agent_recipe_mode, _validate_agent_recipe_structure
from utils.classifier_utils import _clip_head_background_indices, _clip_head_classes, _find_clip_head_target_index, _is_background_class_name
from utils.coords import _agent_iou_xyxy, _yolo_to_xyxy
from utils.embedding_recipe import normalize_embedding_aggregation
from .assistance_adapters import ClassifierRuntime, Sam3PromptRuntime, _build_encoder, predict_probabilities
from .edr_adapters import _bounded_json, _coerce, _member
from .edr_classifier_gate import _classifier_crop_for_head, _clip_head_keep_mask
from .qwen_adapters import _verify_snapshot


def _number(value, default, *, maximum=1.0, integer=False):
    value = default if value is None else value
    if isinstance(value, bool):
        raise ValueError("hosted_recipe_numeric_invalid")
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > maximum or integer and number != int(number):
        raise ValueError("hosted_recipe_numeric_invalid")
    return int(number) if integer else number


def _crop(image, box, head):
    # Bind the previously extracted original crop function without modifying its
    # module globals or another worker's classifier configuration.
    namespace = dict(_classifier_crop_for_head.__globals__, _coerce_float=_coerce,
                     _coerce_int=lambda *args, **kwargs: _coerce(*args, **kwargs, integer=True))
    return FunctionType(_classifier_crop_for_head.__code__, namespace)(image, box, head)


class _EmbeddedClassifier(ClassifierRuntime):
    def __init__(self, head, labels, device):
        self.head = head
        classes = _clip_head_classes(head)
        if not classes or any(label not in labels and not _is_background_class_name(label) for label in classes):
            raise ValueError("hosted_recipe_classifier_labels_changed")
        if str(head.get("encoder_type", "clip")).lower() not in {"clip", "dinov3", "cradio"}:
            raise ValueError("hosted_recipe_classifier_encoder_invalid")
        if not (head.get("encoder_model") or head.get("clip_model")):
            raise ValueError("hosted_recipe_classifier_model_required")
        aggregation = normalize_embedding_aggregation(head.get("embedding_aggregation"))
        if aggregation == "local_salad":
            raise ValueError("local_salad_auto_class_disabled")
        if aggregation == "sam3_mask_salad_fusion_v1":
            raise ValueError("hosted_recipe_trained_salad_runtime_artifact_required")
        # The existing classifier trainer/inference treats sam3_mask_fusion_v1
        # as its recorded encoder pooling here. DQE's separate image/mask branch
        # is not part of this trained classifier path. Feature dimensions below
        # still must match the retained head exactly.
        head.setdefault("encoder_type", "clip")
        self.device = device
        self.model, self.processor = _build_encoder(head, device)


class Sam3ChainRuntime:
    def __init__(self, config, device):
        self.config = copy.deepcopy(config)
        self.actor_id, self.device = self.config["actor_id"], device
        self.bindings = self.config["class_bindings"]
        self.recipes, self.sams, self.classifiers = {}, {}, {}
        self.lock = threading.RLock()
        self.closed = False
        for identifier, definition in self.config.get("recipes", {}).items():
            root = _verify_snapshot(definition["model_path"], definition["files"])
            saved = _bounded_json(_member(root, definition["recipe_file"]))
            recipe = saved.get("recipe", saved)
            _validate_agent_recipe_structure(recipe)
            components = definition.get("component_models", {})
            if components.get("sam3", {}).get("kind") != "sam3":
                raise ValueError("hosted_recipe_sam3_required")
            self.recipes[identifier] = {"definition": definition, "root": root, "saved": saved, "recipe": recipe,
                                        "params": {**(saved.get("params") or {}), **(recipe.get("params") or {})},
                                        "components": components}

    def _record(self, event):
        encoded = json.dumps(event, allow_nan=False)
        self.evidence_bytes += len(encoded.encode())
        if self.evidence_bytes > 16 * 1024 * 1024:
            raise ValueError("hosted_recipe_evidence_limit_exceeded")
        self.events.append(json.loads(encoded))

    def _sam(self, package):
        definition = package["components"]["sam3"]
        identity = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        if identity not in self.sams:
            from .adapters import _build_sam
            predictor, network = _build_sam(definition, self.device)
            self.sams[identity] = (predictor, network, Sam3PromptRuntime(network, self.device))
        return self.sams[identity][2]

    def _head(self, identifier, *, extra=False):
        key = (identifier, extra)
        if key in self.classifiers:
            return self.classifiers[key]
        package = self.recipes[identifier]
        if extra:
            definition = package["components"].get("extra_classifier")
            if not definition:
                raise ValueError("hosted_recipe_extra_classifier_required")
            runtime = ClassifierRuntime(definition, self.device)
        else:
            head_config = package["recipe"].get("clip_head") or {}
            member = head_config.get("artifact", "clip_head/head.npz")
            if member not in package["definition"]["files"]:
                return None
            # Executable NumPy/pickle head artifacts are admin-installed only,
            # hash-verified with the complete package before worker loading.
            with np.load(_member(package["root"], member), allow_pickle=True) as saved:
                head = saved["head"].item()
            if not isinstance(head, dict):
                raise ValueError("hosted_recipe_classifier_head_invalid")
            metadata = head_config
            if "clip_head/meta.json" in package["definition"]["files"]:
                metadata = _bounded_json(_member(package["root"], "clip_head/meta.json"))
            for field in ("min_prob", "margin", "background_margin", "clip_model", "encoder_type", "encoder_model", "proba_mode"):
                if metadata.get(field) is not None:
                    head[field] = metadata[field]
            head.setdefault("min_prob", .5)
            head.setdefault("margin", 0.)
            labels = package["definition"].get("labels") or list(self.bindings)
            runtime = _EmbeddedClassifier(head, labels, self.device)
        self.classifiers[key] = runtime
        return runtime

    def _probabilities(self, runtime, image, detections):
        if not detections:
            return np.empty((0, len(_clip_head_classes(runtime.head))), dtype=np.float32)
        rows = []
        # Preserve saved crop views and fitted feature transformations. Bounded
        # batches prevent all image crops being retained at once.
        for offset in range(0, len(detections), 32):
            crops, records = [], []
            try:
                for item in detections[offset:offset + 32]:
                    crop, record = _crop(image, _yolo_to_xyxy(image.width, image.height, item["bbox"]), runtime.head)
                    crops.append(crop); records.append(record)
                features = runtime.encode_items(crops, geometry_records=records)
                rows.extend(predict_probabilities(features, runtime.head))
            finally:
                for crop in crops:
                    for view in crop if isinstance(crop, (tuple, list)) else [crop]:
                        view.close()
        return np.asarray(rows)

    def _gate(self, identifier, image, detections, label, settings, *, stage, params, extra=False):
        if settings is False or isinstance(settings, dict) and settings.get("enabled") is False:
            return detections
        runtime = self._head(identifier, extra=extra)
        if runtime is None:
            if settings is None and not self.recipes[identifier]["recipe"].get("clip_head") and not params.get("use_clip_fp_guard"):
                return detections
            raise ValueError("hosted_recipe_classifier_required")
        head, settings = runtime.head, settings or {}
        classes = _clip_head_classes(head)
        target = _find_clip_head_target_index(classes, label)
        if target is None:
            raise ValueError("hosted_recipe_classifier_target_unavailable")
        min_prob = _number(settings.get("min_prob"), head.get("min_prob", .5))
        margin = _number(settings.get("margin"), head.get("margin", 0.))
        bg_apply = params.get("clip_head_background_apply") or "final"
        if bg_apply not in {"seed", "final", "both"}:
            raise ValueError("hosted_recipe_background_policy_invalid")
        background_guard = bool(params.get("clip_head_background_guard", True)) and bg_apply in {stage, "both"}
        background_margin = _number(params.get("clip_head_background_margin"), head.get("background_margin", 0.))
        probabilities = self._probabilities(runtime, image, detections)
        if not len(detections):
            return detections
        backgrounds = _clip_head_background_indices(classes)
        keep = _clip_head_keep_mask(probabilities, target_index=target, min_prob=min_prob, margin=margin,
                                   background_indices=backgrounds, background_guard=background_guard, background_margin=background_margin)
        evidence = []
        for det, row, accepted in zip(detections, probabilities, keep):
            other = float(np.max(np.delete(row, target))) if len(row) > 1 else 0.
            bg = float(np.max(row[backgrounds])) if backgrounds else 0.
            det.update(clip_head_prob=float(row[target]), clip_head_margin=float(row[target]) - other,
                       clip_head_bg_prob=bg, clip_head_bg_margin=float(row[target]) - bg)
            evidence.append({"bbox": det["bbox"], "probabilities": {name: float(value) for name, value in zip(classes, row)}, "accepted": bool(accepted)})
        self._record({"stage": ("extra_" if extra else "") + "classifier_" + stage, "recipe_id": identifier,
                      "label": label, "min_prob": min_prob, "margin": margin, "background_guard": background_guard,
                      "background_margin": background_margin, "rows": evidence})
        return [det for det, accepted in zip(detections, keep) if accepted]

    @staticmethod
    def _dedupe(detections, image, threshold, scores=None):
        # Original chain: stable score ordering, suppress IoU >= threshold.
        kept, boxes = [], []
        for det in sorted(detections, key=lambda row: scores.get(id(row), row.get("score") or 0.) if scores is not None else row.get("score") or 0., reverse=True):
            box = _yolo_to_xyxy(image.width, image.height, det["bbox"])
            if any(_agent_iou_xyxy(box, previous) >= threshold for previous in boxes):
                continue
            kept.append(det); boxes.append(box)
        return kept

    def _predict(self, package, identifier, image, settings, *, stage, visual=False):
        result = self._sam(package).predict(image, settings, visual=visual)
        detections = result["detections"]
        self._record({"stage": stage, "recipe_id": identifier, "input": settings,
                      "output": [{key: value for key, value in item.items() if key != "mask"} for item in detections],
                      "warnings": result.get("warnings", [])})
        return detections

    @staticmethod
    def _extend(target, rows, *, other_count=0):
        if len(target) + len(rows) + other_count > 10000:
            raise ValueError("hosted_recipe_candidate_limit_exceeded")
        target.extend(rows)

    def _crop_bank(self, package, identifier, image, positives, negatives, settings):
        """Encode retained crops with SAM3's own geometry encoder, then ground
        their positive/negative visual tokens against the untouched target.

        The model already has an explicit visual_prompt_embed seam. The stock
        forward_grounding does not pass it through, so the private instance's
        bound _encode_prompt receives it for this one synchronous call only.
        """
        import inspect
        import torch
        from PIL import Image
        if len(positives) + len(negatives) > 64 or not (positives or negatives):
            raise ValueError("hosted_recipe_crop_bank_invalid")
        if not positives and not str(settings.get("text_prompt") or "").strip():
            raise ValueError("hosted_recipe_negative_bank_target_required")
        runtime = self._sam(package)
        processor, network = runtime.processor, runtime.processor.model
        original = network._encode_prompt
        if not {"visual_prompt_embed", "visual_prompt_mask"}.issubset(inspect.signature(original).parameters):
            raise ValueError("hosted_recipe_visual_token_interface_unavailable")
        tokens, masks, provenance = [], [], []
        with torch.inference_mode():
            for entry, positive in [(entry, True) for entry in positives] + [(entry, False) for entry in negatives]:
                name = entry.get("crop_path") if isinstance(entry, dict) else None
                if not isinstance(name, str) or name not in package["definition"]["files"]:
                    raise ValueError("hosted_recipe_crop_artifact_required")
                path = _member(package["root"], name)
                if path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("hosted_recipe_crop_limit_exceeded")
                with Image.open(path) as source:
                    if source.width * source.height > 16_000_000:
                        raise ValueError("hosted_recipe_crop_limit_exceeded")
                    with source.convert("RGB") as crop:
                        state = processor.set_image(crop)
                backbone, features, positions, sizes = network._get_img_feats(state["backbone_out"], processor.find_stage.img_ids)
                prompt = network._get_dummy_prompt()
                prompt.append_boxes(torch.tensor([.5, .5, 1., 1.], device=processor.device).view(1, 1, 4),
                                    torch.tensor([positive], device=processor.device, dtype=torch.bool).view(1, 1))
                encoded, padding = network.geometry_encoder(geo_prompt=prompt, img_feats=features, img_sizes=sizes, img_pos_embeds=positions)
                tokens.append(encoded); masks.append(padding)
                provenance.append({"artifact": name, "sha256": package["definition"]["files"][name], "positive": positive})
            visual_tokens, visual_mask = torch.cat(tokens, dim=0), torch.cat(masks, dim=1)
            def encode_with_bank(_network, *args, **kwargs):
                kwargs.update(visual_prompt_embed=visual_tokens, visual_prompt_mask=visual_mask)
                return original(*args, **kwargs)
            had_override = "_encode_prompt" in network.__dict__
            network._encode_prompt = MethodType(encode_with_bank, network)
            try:
                result = self._predict(package, identifier, image, {**settings, "text_prompt": settings.get("text_prompt") or "visual"}, stage="external_visual_bank")
            finally:
                if had_override:
                    network._encode_prompt = original
                else:
                    delattr(network, "_encode_prompt")
        self._record({"stage": "crop_bank_tokens", "recipe_id": identifier, "crops": provenance,
                      "token_count": int(visual_tokens.shape[0]), "output_count": len(result)})
        return result

    def _recipe(self, identifier, image, step, payload):
        package = self.recipes[identifier]
        recipe, params, saved = package["recipe"], package["params"], package["saved"]
        # Background penalty belongs to recipe mining's optimizer, not runtime
        # detection scoring (the original UI labels that scope explicitly).
        _number(params.get("clip_head_background_penalty"), 0., maximum=2.)
        # No retained inference/training source defines a weighted-negative
        # equation. Binary native positive/negative geometry is executable;
        # don't invent a multiplier when an active bank requests another value.
        if recipe.get("negatives") and params.get("use_negative_exemplars") is not False and params.get("negative_strength") not in {None, 0, 1}:
            raise ValueError("hosted_recipe_weighted_negative_equation_unavailable")
        label = step.override_class_name or saved.get("class_name") or recipe.get("class_name")
        target = self.bindings.get(label)
        if not target or step.override_class_id is not None and step.override_class_id != target["position"]:
            raise ValueError("hosted_recipe_output_class_changed")
        mode = _classify_agent_recipe_mode(recipe)
        if mode == "sam3_steps":
            stages = recipe["steps"]
        elif mode == "sam3_greedy":
            stages = [{"prompt": prompt} for prompt in recipe.get("text_prompts") or []]
            if not stages and recipe.get("positives"):
                stages = [{}]
        else:
            stages = recipe.get("steps") or []
        if not stages or len(stages) > 64:
            raise ValueError("hosted_recipe_steps_invalid")
        all_detections = []
        for index, stage in enumerate(stages):
            if stage.get("enabled") is False:
                continue
            prompts = stage.get("prompts") or [stage.get("prompt")]
            positives = [stage["exemplar"]] if stage.get("exemplar") else recipe.get("positives", []) if index == 0 else []
            negatives = (recipe.get("negatives") or []) if positives or index == 0 or not recipe.get("positives") else []
            if params.get("use_negative_exemplars") is False or params.get("negative_strength") == 0:
                negatives = []
            if positives and prompts in ([None], [""]):
                prompts = []
            if not isinstance(prompts, list) or len(prompts) > 32 or not prompts and not positives or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
                raise ValueError("hosted_recipe_prompt_invalid")
            maximum = _number(stage.get("max_results"), payload.max_results, maximum=5000, integer=True)
            if maximum < 1:
                raise ValueError("hosted_recipe_result_limit_invalid")
            common = {"mask_threshold": _number(stage.get("mask_threshold"), payload.mask_threshold),
                      "min_size": payload.min_size, "simplify_epsilon": payload.simplify_epsilon, "max_results": maximum}
            seed_threshold = _number(stage.get("seed_threshold", stage.get("threshold")), params.get("seed_threshold") or 0.)
            seeds = []
            if positives:
                self._extend(seeds, self._crop_bank(package, identifier, image, positives, negatives,
                    {**common, "threshold": _number(stage.get("threshold"), params.get("similarity_score") if params.get("similarity_score") is not None else .25)}))
            for prompt in prompts:
                settings = {**common, "text_prompt": prompt, "threshold": seed_threshold}
                if negatives and not positives:
                    # SAM3 accepts negative visual tokens alongside the actual
                    # saved text target; no invented positive crop is needed.
                    self._extend(seeds, self._crop_bank(package, identifier, image, [], negatives, settings))
                else:
                    self._extend(seeds, self._predict(package, identifier, image, settings, stage="text_seed"))
            seeds = self._dedupe(seeds, image, _number(stage.get("seed_dedupe_iou"), params.get("seed_dedupe_iou") if params.get("seed_dedupe_iou") is not None else .9))
            seed_gate = stage.get("clip_seed")
            seeds = self._gate(identifier, image, seeds, label, seed_gate if seed_gate is not None else False,
                               stage="seed", params=params)
            # Explicit zero disables expansion. Missing expansion configuration
            # in legacy prompt-only recipes remains prompt-only by definition.
            count = _number(stage.get("max_visual_seeds"), params.get("max_visual_seeds") or 0, maximum=500, integer=True)
            expanded = []
            for seed in seeds[:count]:
                x1, y1, x2, y2 = _yolo_to_xyxy(image.width, image.height, seed["bbox"])
                visual = {**common, "bboxes": [[x1, y1, x2-x1, y2-y1]], "bbox_labels": [True],
                          "threshold": _number(stage.get("expand_threshold"), params.get("expand_threshold") if params.get("expand_threshold") is not None else .15),
                          "max_results": _number(stage.get("expand_max_results"), maximum, maximum=5000, integer=True)}
                self._extend(expanded, self._predict(package, identifier, image, visual, stage="visual_expand", visual=True), other_count=len(seeds))
            detections = self._dedupe([*seeds, *expanded], image, _number(stage.get("step_dedupe_iou", stage.get("dedupe_iou")), params.get("dedupe_iou") if params.get("dedupe_iou") is not None else .5))
            final_gate = copy.deepcopy(stage.get("clip_final"))
            if step.clip_head_min_prob_override is not None or step.clip_head_margin_override is not None:
                final_gate = dict(final_gate or {})
                if step.clip_head_min_prob_override is not None:
                    final_gate["min_prob"] = step.clip_head_min_prob_override
                if step.clip_head_margin_override is not None:
                    final_gate["margin"] = step.clip_head_margin_override
            detections = self._gate(identifier, image, detections, label, final_gate, stage="final", params=params)
            if step.extra_clip_classifier_path:
                extra = package["components"].get("extra_classifier", {})
                if step.extra_clip_classifier_path != extra.get("model_id"):
                    raise ValueError("hosted_recipe_extra_classifier_identity_changed")
                detections = self._gate(identifier, image, detections, label,
                    {"min_prob": step.extra_clip_min_prob, "margin": step.extra_clip_margin}, stage="final", params=params, extra=True)
            for det in detections[:maximum]:
                det.update(class_id=target["position"], stable_class_id=target["id"], class_name=target["name"])
            all_detections.extend(detections[:maximum])
            self._record({"stage": "recipe_step_complete", "recipe_id": identifier, "step": index,
                          "seed_count": len(seeds), "expanded_count": len(expanded), "accepted_count": len(detections[:maximum])})
            if len(all_detections) > 10000:
                raise ValueError("hosted_recipe_candidate_limit_exceeded")
        return self._dedupe(all_detections, image, _number(params.get("dedupe_iou"), .5))[:payload.max_results]

    def execute(self, image, payload, context):
        with self.lock:
            if self.closed or context.get("actor_id") != self.actor_id:
                raise ValueError("hosted_recipe_actor_changed")
            requested = copy.deepcopy(payload)
            for step in requested.get("steps", []):
                if step.get("recipe"):
                    raise ValueError("hosted_recipe_inline_not_allowed")
            request = AgentApplyImageChainRequest(**{**requested, "image_base64": None, "image_token": "hosted-snapshot", "sam_variant": "sam3"})
            steps = [step for step in request.steps if step.enabled]
            if len(steps) > 32 or any(step.recipe_id not in self.recipes for step in steps):
                raise ValueError("hosted_recipe_not_attached")
            self.events, self.evidence_bytes = [], 0
            detections, origins = [], {}
            used_recipes = {step.recipe_id for step in steps}
            for index, step in enumerate(steps):
                rows = self._recipe(step.recipe_id, image, step, request)
                for row in rows:
                    origins[id(row)] = {"step": index, "group": (step.dedupe_group or "").strip() or "default", "cross": step.participate_cross_class_dedupe}
                detections.extend(rows)
            ranking = None
            if request.dedupe.confidence != "sam_score":
                head_id = request.dedupe.clip_head_recipe_id
                if head_id is None:
                    head_id = next((step.recipe_id for step in steps if self._head(step.recipe_id) is not None), None)
                if head_id not in self.recipes or self._head(head_id) is None:
                    raise ValueError("hosted_recipe_dedupe_classifier_required")
                runtime = self._head(head_id)
                used_recipes.add(head_id)
                probabilities = self._probabilities(runtime, image, detections)
                classes, ranking = _clip_head_classes(runtime.head), {}
                for det, row in zip(detections, probabilities):
                    target = _find_clip_head_target_index(classes, det["class_name"])
                    if target is None:
                        raise ValueError("hosted_recipe_dedupe_class_unavailable")
                    ranking[id(det)] = float(row[target]) - (float(np.max(np.delete(row, target))) if len(row) > 1 else 0.) if request.dedupe.confidence == "clip_head_margin" else float(row[target])
                self._record({"stage": "cascade_classifier_ranking", "recipe_id": head_id, "mode": request.dedupe.confidence,
                              "rows": [{"bbox": det["bbox"], "rank_score": ranking[id(det)], "sam_score": det.get("score")} for det in detections]})
            by_class = {}
            for det in detections:
                by_class.setdefault(det["stable_class_id"], []).append(det)
            final = [det for rows in by_class.values() for det in self._dedupe(rows, image, request.dedupe.per_class_iou, ranking)]
            if request.dedupe.cross_class_enabled:
                groups = {}
                for det in final:
                    origin = origins[id(det)]
                    if origin["cross"]:
                        groups.setdefault("global" if request.dedupe.cross_class_scope == "global" else origin["group"], []).append(det)
                keep = {id(det) for rows in groups.values() for det in self._dedupe(rows, image, request.dedupe.cross_class_iou, ranking)}
                final = [det for det in final if id(det) in keep or not origins[id(det)]["cross"]]
            self._record({"stage": "cascade_complete", "input_count": len(detections), "output_count": len(final), "dedupe": request.dedupe.model_dump()})
            return {"detections": final, "warnings": [], "masks": [det.get("mask") for det in final],
                    "hosted_evidence": {"stages": self.events, "recipes": {identifier: self.recipes[identifier]["definition"]["checkpoint_sha256"] for identifier in sorted(used_recipes)},
                                        "recipe_settings": {identifier: {
                                            "negative_strength": self.recipes[identifier]["params"].get("negative_strength"),
                                            "clip_head_background_penalty": self.recipes[identifier]["params"].get("clip_head_background_penalty"),
                                            "background_penalty_scope": "recipe_optimizer",
                                            "classifier_aggregation": self.classifiers[(identifier, False)].head.get("embedding_aggregation") if (identifier, False) in self.classifiers else None,
                                        } for identifier in sorted(used_recipes)},
                                        "execution_contract": "hosted-sam3-step-chain-v1"}}

    def close(self):
        with self.lock:
            for _, _, runtime in self.sams.values():
                runtime.close()
            for runtime in self.classifiers.values():
                runtime.close()
            self.sams.clear(); self.classifiers.clear()
            self.closed = True
