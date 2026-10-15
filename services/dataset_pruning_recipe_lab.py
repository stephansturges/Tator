"""Read-only cached recipe comparisons and isolated diagnostic calibrations.

This module never changes the human removal ledger, fits geometry, encodes a
pixel, or trains a head. A completed all-component analysis supplies the raw
producer bindings and saved geometry coefficients. Calibration records belong
to their own derived namespace and cannot approve a removal.
"""
from __future__ import annotations

import copy
from contextlib import closing, nullcontext
import json
from pathlib import Path
import re
import sqlite3
import threading

import numpy as np
from fastapi import HTTPException

from services.dataset_pruning import PruningConfig, digest, pair_evidence
from services.dataset_pruning_calibration import OBJECT_FLOORS, fit_thresholds, assert_profile_binding
from services.dataset_pruning_comparison import reconstruct_cached_object_rows
from services.dataset_pruning_correction import effective_transform, validate_strength
from services.dataset_pruning_serving import states, verify, thaw

SCHEMA = "pruning-recipe-lab-v1"
PROFILE_SCHEMA = "pruning-recipe-calibration-record-v1"
LOCK = threading.RLock()
COMPONENT_IDS = {"dinov3", "sam3", "salad", "cradio"}


def _read(path, limit=8 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise HTTPException(409, "Saved recipe comparison material is unavailable or changed")
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as error:
        raise HTTPException(409, "Saved recipe comparison material cannot be read; preserve it for inspection") from error


def _strength(component, entry):
    transform = entry.get("adjustment") or {}
    recorded = validate_strength(transform.get("correction_strength"), default=1 if transform.get("applied") else 0)
    declared = validate_strength(component.get("adjustment_strength"), default=recorded)
    if declared != recorded:
        raise HTTPException(409, "The declared correction strength differs from its saved component output; recompute this exact analysis")
    return declared


def _semantic_transform(transform, alpha):
    if alpha == 0:
        return {"applied": False, "strength": 0.0}
    saved = transform.get('fitted_transform') or {}
    recorded = validate_strength(transform.get('correction_strength'), default=1 if transform.get('applied') else 0)
    if saved.get('base_coefficients'):
        base = np.asarray(saved['base_coefficients'], np.float32)
        effective_beta = np.asarray(saved.get('coefficients'), np.float32)
        if base.shape != effective_beta.shape or not np.array_equal(effective_beta, base * np.float32(recorded)):
            raise HTTPException(409, 'Saved effective geometry coefficients differ from their recorded base fit and strength; preserve this analysis for inspection')
    effective = copy.deepcopy(transform) if alpha == recorded else effective_transform(transform, alpha)
    fitted = effective.get("fitted_transform") or {}
    if not effective.get("applied"):
        raise HTTPException(409, "The saved geometry fit is unavailable; run the all-component pruning candidate before comparing recipes")
    # Bind the actual mathematical fit; cache warmth and display/status fields
    # must not make identical descriptors appear to be different recipes.
    fields = {key: fitted.get(key) for key in ["schema", "covariate_mean", "covariate_std", "normalization", "record_digest", "fit_population_count", "design_sha256"]}
    fields.update(covariates=effective.get("covariates"),
                  effective_coefficients_sha256=digest(fitted["coefficients"]),
                  base_coefficients_sha256=digest(fitted.get("base_coefficients", fitted["coefficients"])))
    return {"applied": True, "strength": alpha, "fit": fields}


def feature_binding(source, recipe, manifest, config, variant=None):
    """Exact source, producers, complete fits, weights and matching policy."""
    components = recipe.get("bbox", {}).get("components", [])
    if variant is not None:
        resolved, manifest = variant_recipe(recipe, manifest, variant)
        components = resolved["bbox"]["components"]
    objects = []
    for component in components:
        if component.get("weight", 0) <= 0:
            continue
        entry = manifest.get("components", {}).get("bbox_" + component["id"])
        if not entry or not entry.get("raw_recipe"):
            raise HTTPException(409, "Saved raw producer bindings are unavailable for this object component")
        alpha = _strength(component, entry)
        objects.append({"id": component["id"], "weight": round(float(component["weight"]), 12),
                        "dimensions": entry["dimension"], "raw_recipe_digest": digest(entry["raw_recipe"]),
                        "correction": _semantic_transform(entry.get("adjustment") or {}, alpha)})
    scenes = []
    for ordinal, component in enumerate(recipe.get("scene", {}).get("components", [])):
        entry = manifest.get("components", {}).get("scene_" + str(ordinal))
        if not entry or not entry.get("raw_recipe"):
            raise HTTPException(409, "Saved whole-image producer bindings are unavailable")
        alpha = _strength(component, entry)
        scenes.append({"weight": round(float(component["weight"]), 12), "dimensions": entry["dimension"],
                       "raw_recipe_digest": digest(entry["raw_recipe"]),
                       "correction": _semantic_transform(entry.get("adjustment") or {}, alpha)})
    matcher = {key: value for key, value in config.recipe().items()
               if key not in {"object_threshold", "scene_threshold", "index_threshold", "pair_scene_prefilter"}}
    return {"schema": "pruning-recipe-lab-feature-binding-v1",
            "source": {"identity": source["identity"], "revision": source["revision"],
                       "inventory_digest": source["inventory_digest"], "labelmap_digest": digest(source["labelmap"])},
            "object_preparation": recipe.get("bbox", {}).get("preprocessing"),
            "objects": objects, "scenes": scenes, "matcher": matcher,
            "weight_binding_decimal_places": 12, "actual_fusion_weights": "Unrounded normalized component weights",
            "fusion": "independent_unit_rows_sqrt_weight_concat_unit_rows",
            "assignment": "fresh_class_directional_appearance_v3_at_each_object_floor"}


def variant_recipe(recipe, manifest, variant):
    try:
        alpha = validate_strength(variant.get("alpha"))
        share = validate_strength(variant.get("cradio_weight"), default=.25)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if share >= 1:
        raise HTTPException(400, "Keep a positive compact-branch share; C-RADIO weight must be less than 1")
    candidate, produced = thaw(recipe), thaw(manifest)
    components = candidate["bbox"]["components"]
    if {item["id"] for item in components} != COMPONENT_IDS:
        raise HTTPException(409, "Recipe comparison needs the shared DINOv3, SAM3, crop SALAD and C-RADIO components")
    compact_weight = sum(item["weight"] for item in components if item["id"] != "cradio")
    if compact_weight <= 0:
        raise HTTPException(409, "The saved compact object recipe has no positive weight")
    for component in components:
        entry = produced["components"]["bbox_" + component["id"]]
        if component["id"] == "cradio":
            component.update(weight=share, adjustment="remove_size_bias" if alpha else "none", adjustment_strength=alpha)
            entry["weight"] = share
            try:
                entry["adjustment"] = effective_transform(entry["adjustment"], alpha)
            except ValueError as error:
                raise HTTPException(409, str(error)) from error
        else:
            component["weight"] = (1 - share) * component["weight"] / compact_weight
            entry["weight"] = component["weight"]
    candidate["bbox"]["components"] = [item for item in components if item["weight"] > 0]
    return candidate, produced


def _capture(service, job_id, payload=None, *, require_available=True):
    payload = payload or {}
    with service.lock:
        directory, result, source, images, _ = service.load_selection(
            job_id, {"removed_count": 0, **payload}, check_selection=bool(payload))
        manifest_path = directory / "components.json"
        manifest = _read(manifest_path) if manifest_path.is_file() else {}
        material = states([manifest_path, directory / "scene.npy", directory / "evidence.sqlite3"])
        if manifest and digest(manifest) != digest(result.get("recipe", {}).get("component_manifest", {})):
            raise HTTPException(409, "Saved component fits changed; preserve this analysis and recompute it")
        recipe = result["recipe"]
        config = PruningConfig.from_payload(result.get("config"))
        available, reason = True, ""
        try:
            if config.evidence_mode != "compound":
                raise HTTPException(409, "This lab compares compound scene-and-object recipes; run the compound pruning candidate first")
            if {item["id"] for item in recipe.get("bbox", {}).get("components", [])} != COMPONENT_IDS:
                raise HTTPException(409, "Run pruning_thorough_v2 to attach all four existing object components before using the recipe lab")
            for item in recipe["bbox"]["components"]:
                entry = manifest.get("components", {}).get("bbox_" + item["id"], {})
                _semantic_transform(entry.get("adjustment") or {}, 1 if item["id"] == "cradio" else _strength(item, entry))
            base = {"feature_binding": feature_binding(source, recipe, manifest, config),
                    "cradio_complete_fit": _semantic_transform(manifest["components"]["bbox_cradio"]["adjustment"], 1)}
        except (HTTPException, ValueError, KeyError) as error:
            available, reason = False, str(error.detail if isinstance(error, HTTPException) else error)
            base = {"source": {"identity": source["identity"], "revision": source["revision"], "inventory_digest": source["inventory_digest"]},
                    "recorded_recipe_digest": digest(recipe), "available": False}
        base_digest = digest(base)
        if payload and payload.get("expected_digest") != base_digest:
            raise HTTPException(409, "The source or saved recipe changed; refresh recipe comparisons before continuing")
        if require_available and not available:
            raise HTTPException(409, reason)
        return {"directory": directory, "result": result, "source": source, "images": images,
                "recipe": recipe, "manifest": manifest, "config": config, "material": material,
                "base": base, "base_digest": base_digest, "available": available, "reason": reason,
                "indices": {image["image_id"]: index for index, image in enumerate(images)}}


def _recheck(service, job_id, payload, context):
    current = _capture(service, job_id, payload)
    if current["base_digest"] != context["base_digest"]:
        raise HTTPException(409, "The source or fitted recipe changed while computing; no calibration was published")
    verify(context["material"])


def _case_id(source, candidate, witness):
    return "dpcase_" + digest([source["inventory_digest"], candidate["image_id"], witness["image_id"]])[:24]


def _public_pair(service, job_id, source, candidate, witness):
    result = {"case_id": _case_id(source, candidate, witness), "candidate_id": candidate["image_id"], "witness_id": witness["image_id"],
              "candidate": service.public_image(candidate), "witness": service.public_image(witness),
              "candidate_url": f'/dataset_pruning/jobs/{job_id}/images/{candidate["image_id"]}',
              "witness_url": f'/dataset_pruning/jobs/{job_id}/images/{witness["image_id"]}'}
    for side, image in [("candidate", candidate), ("witness", witness)]:
        result[side + "_objects"] = [{"row": obj["label_row"], "class_id": obj["class_id"],
            "url": f'/dataset_pruning/jobs/{job_id}/images/{image["image_id"]}/objects/{obj["label_row"]}'} for obj in image["objects"]]
    return result


def _pairs(context, values):
    if not isinstance(values, list) or not 1 <= len(values) <= 24:
        raise HTTPException(400, "Select between 1 and 24 directional image comparisons")
    result, seen = [], set()
    for item in values:
        if not isinstance(item, dict):
            raise HTTPException(400, "Every comparison needs two image identities")
        candidate, witness = context["indices"].get(item.get("candidate_id")), context["indices"].get(item.get("witness_id"))
        if candidate is None or witness is None or candidate == witness:
            raise HTTPException(400, "Select distinct images from this saved analysis")
        case_id = _case_id(context["source"], context["images"][candidate], context["images"][witness])
        if item.get("case_id") != case_id or case_id in seen:
            raise HTTPException(400, "Directional comparison identities changed or were repeated")
        seen.add(case_id)
        result.append({"case_id": case_id, "candidate": candidate, "witness": witness,
                       "candidate_id": item["candidate_id"], "witness_id": item["witness_id"]})
    return result


def _variants(values):
    if not isinstance(values, list) or not 1 <= len(values) <= 12:
        raise HTTPException(400, "Compare between 1 and 12 recipe variants")
    result, seen = [], set()
    for value in values:
        if not isinstance(value, dict) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(value.get("id") or "")) or value["id"] in seen:
            raise HTTPException(400, "Recipe variants need distinct simple identifiers")
        try:
            alpha = validate_strength(value.get("alpha"))
            weight = validate_strength(value.get("cradio_weight"), default=.25)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if weight == 1:
            raise HTTPException(400, "Keep a positive compact-branch share")
        seen.add(value["id"])
        result.append({"id": value["id"], "alpha": alpha, "cradio_weight": weight})
    return result


def _profile_records(directory):
    path = _database_path(directory)
    if not path.exists():
        return []
    if path.is_symlink():
        raise HTTPException(409, "Calibration profile storage changed")
    result = []
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            for (content,) in db.execute("SELECT content FROM profiles ORDER BY rowid DESC LIMIT 128"):
                value = json.loads(content)
                if value.get("schema") != PROFILE_SCHEMA or digest(value.get("profile")) != value.get("profile_digest"):
                    raise HTTPException(409, "A saved calibration profile changed; preserve it for inspection")
                result.append(value)
    except (sqlite3.DatabaseError, ValueError) as error:
        raise HTTPException(409, "Saved calibration profiles cannot be read; preserve them for inspection") from error
    return result


def _draft_config(context, variant, thresholds, reference=None):
    recipe, _ = variant_recipe(context["recipe"], context["manifest"], variant)
    config = {name: getattr(context["config"], name) for name in context["config"].__dataclass_fields__}
    config.update(thresholds)
    fields = {"id", "encoder", "encoder_type", "model", "head_id", "pooling", "weight", "adjustment", "adjustment_strength", "input_size"}
    config.update(object_recipe="thorough_quality_v1" if variant["alpha"] == 0 and variant["cradio_weight"] > 0 else "pruning_thorough_v2",
                  object_adjustment="inherit",
                  object_components=[{**{key: thaw(value) for key, value in item.items() if key in fields}, "enabled": True} for item in recipe["bbox"]["components"]],
                  scene_components=[{**{key: thaw(value) for key, value in item.items() if key in fields}, "enabled": True} for item in recipe["scene"]["components"]],
                  object_preprocessing=thaw(recipe["bbox"]["preprocessing"]),
                  threshold_qualification="diagnostic_calibrated" if reference else "unmeasured_draft")
    if reference:
        config["calibration_profile"] = reference
    return config


def _database_path(directory):
    root = directory / "calibration"
    path = root / "recipe-lab.sqlite3"
    if root.is_symlink() or path.is_symlink():
        raise HTTPException(409, "Calibration profile storage changed")
    return path


def _operation(directory, token, identity):
    path = _database_path(directory)
    if not path.exists():
        return None
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            row = db.execute("SELECT identity,response FROM operations WHERE request_id=?", (token,)).fetchone()
        if row is None:
            return None
        if row[0] != identity:
            raise HTTPException(409, "This request identity belongs to another calibration")
        return json.loads(row[1])
    except (sqlite3.DatabaseError, ValueError) as error:
        raise HTTPException(409, "Saved calibration operations cannot be read; preserve them for inspection") from error


def metadata(service, job_id):
    with service.adapter.guard():
        context = _capture(service, job_id, require_available=False)
        pairs = []
        seen = set()
        if context["available"]:
            path = context["directory"] / "evidence.sqlite3"
            with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
                for eligible in [1, 0]:
                    accepted = 0
                    for (raw,) in db.execute("SELECT payload FROM pairs WHERE valid=? ORDER BY a,b LIMIT 128", (eligible,)):
                        edge = json.loads(raw)
                        left, right = context["images"][edge["left"]], context["images"][edge["right"]]
                        if left.get("protection_errors") or right.get("protection_errors"):
                            continue
                        candidate, witness = (right, left) if not edge.get("left_valid") and edge.get("right_valid") else (left, right)
                        if not candidate['objects'] or not witness['objects']:
                            continue
                        value = _public_pair(service, job_id, context["source"], candidate, witness)
                        if value["case_id"] in seen:
                            continue
                        pairs.append(value); seen.add(value["case_id"]); accepted += 1
                        if accepted >= 12:
                            break
        profiles = []
        for record in _profile_records(context["directory"]):
            profile = copy.deepcopy(record["profile"])
            profile["reference"] = {"job_id": job_id, "profile_id": profile["profile_id"]}
            profile["available"] = profile.get("status") == "diagnostic_calibrated" and context["available"] and record["base_digest"] == context["base_digest"]
            profiles.append(profile)
        verify(context["material"])
        return {"schema": SCHEMA, "job_id": job_id, "source_revision": context["source"]["revision"],
                "selection_revision": context["result"]["selection_revision"], "base_digest": context["base_digest"],
                "available": context["available"], "reason": context["reason"],
                "baseline": {"id": "historical", "alpha": 0.0, "cradio_weight": .25},
                "candidate": {"id": "candidate", "alpha": 1.0, "cradio_weight": .25},
                "pairs": pairs, "profiles": profiles,
                "scope": "Selected labelled-object directions from the saved discovery pool; explicit negatives keep their separate negative-scene rule. This is not a random accuracy sample or a removal approval."}


def _measure(service, job_id, context, pairs, variants, floors=None):
    from services.dataset_pruning_jobs import pruning_blas_policy
    indices = sorted({value[side] for value in pairs for side in ["candidate", "witness"]})
    images = [context["images"][index] for index in indices]
    scene = np.load(context["directory"] / "scene.npy", mmap_mode="r", allow_pickle=False)
    output = []
    with service.feature_lock, pruning_blas_policy():
        for variant in variants:
            recipe, manifest = variant_recipe(context["recipe"], context["manifest"], variant)
            binding = feature_binding(context["source"], context["recipe"], context["manifest"], context["config"], variant)
            features = reconstruct_cached_object_rows(service.adapter.api, service.cache, images, recipe, manifest)
            rows = []
            profile = variant.get("calibration_profile") if floors is None else None
            variant_floors = floors if floors is not None else [profile["thresholds"]["object_threshold"] if profile else context["config"].object_threshold]
            for floor in variant_floors:
                cfg = {name: getattr(context["config"], name) for name in context["config"].__dataclass_fields__}
                cfg.update(object_threshold=floor)
                if floors is not None:
                    cfg.update(scene_threshold=0.0, index_threshold=0.0)
                elif profile:
                    cfg.update(profile["thresholds"])
                config = PruningConfig.from_payload(cfg)
                measured = {}
                for value in pairs:
                    a, b = sorted([value["candidate"], value["witness"]])
                    key = (a, b)
                    if key not in measured:
                        measured[key] = pair_evidence(context["images"][a], context["images"][b], float(scene[a] @ scene[b]), features, config)
                    edge = measured[key]
                    side = "left" if value["candidate"] == a else "right"
                    crops = []
                    for match in edge["matches"]:
                        candidate_row, witness_row = (match["left_row"], match["right_row"]) if side == "left" else (match["right_row"], match["left_row"])
                        crops.append({"candidate_row": candidate_row, "witness_row": witness_row, "class_id": match["class_id"],
                                      "appearance": match["appearance"], "scale": match["scale"],
                                      "candidate_url": f'/dataset_pruning/jobs/{job_id}/images/{value["candidate_id"]}/objects/{candidate_row}',
                                      "witness_url": f'/dataset_pruning/jobs/{job_id}/images/{value["witness_id"]}/objects/{witness_row}'})
                    rows.append({"case_id": value["case_id"], "object_threshold": float(floor),
                                 "scene": edge["scene"], "score": edge[side + "_combined"],
                                 "eligible": bool(edge[side + "_valid"]), "flagged": bool(edge[side + "_valid"]),
                                 "evidence": edge, "crops": crops,
                                 "candidate_id": value["candidate_id"], "witness_id": value["witness_id"]})
            output.append({**{key: value for key, value in variant.items() if key != "calibration_profile"},
                           "binding": binding, "feature_recipe_digest": digest(binding), "rows": rows,
                           "thresholds": profile["thresholds"] if profile else {key: getattr(context["config"], key) for key in ["object_threshold", "scene_threshold", "index_threshold"]},
                           "threshold_qualification": "diagnostic_calibrated" if profile else "active_job_gates_unqualified_for_variant",
                           "profile_reference": {"job_id": job_id, "profile_id": profile["profile_id"]} if profile else None})
            output[-1]["draft_config"] = _draft_config(context, variant, output[-1]["thresholds"], output[-1]["profile_reference"])
    return output


def compare(service, job_id, payload):
    with service.adapter.guard():
        context = _capture(service, job_id, payload)
        pairs, variants = _pairs(context, payload.get("pairs")), _variants(payload.get("variants"))
        records = _profile_records(context["directory"])
        for variant in variants:
            binding = feature_binding(context["source"], context["recipe"], context["manifest"], context["config"], variant)
            for record in records:
                profile = record["profile"]
                if profile.get("status") == "diagnostic_calibrated" and profile.get("feature_binding") == binding and profile.get("feature_recipe_digest") == digest(binding):
                    variant["calibration_profile"] = profile
                    break
        result = _measure(service, job_id, context, pairs, variants)
        _recheck(service, job_id, payload, context)
        disagreements = []
        for value in pairs:
            flags = {variant["id"]: next(row["flagged"] for row in variant["rows"] if row["case_id"] == value["case_id"]) for variant in result}
            if len(set(flags.values())) > 1:
                disagreements.append({"case_id": value["case_id"], "flags": flags})
        return {"schema": SCHEMA, "job_id": job_id, "base_digest": context["base_digest"],
                "selection_revision": context["result"]["selection_revision"], "variants": result, "disagreements": disagreements,
                "qualification": "An exactly matching saved diagnostic profile supplies its own gates. Other variants use the active job's unqualified gates for orientation. No removal approval was changed.",
                "encoder_calls": 0, "head_training_calls": 0, "geometry_refits": 0, "review_mutations": 0}


def add_pair(service, job_id, payload):
    """Bind a user-selected direction to this completed source without scores."""
    with service.adapter.guard():
        context = _capture(service, job_id, payload)
        candidate_index = context['indices'].get(payload.get('candidate_id'))
        witness_index = context['indices'].get(payload.get('witness_id'))
        if candidate_index is None or witness_index is None or candidate_index == witness_index:
            raise HTTPException(400, 'Select two distinct images from this completed analysis')
        candidate, witness = context['images'][candidate_index], context['images'][witness_index]
        if not candidate['objects'] or not witness['objects']:
            raise HTTPException(400, 'Choose labelled-object images; explicit negatives retain their separate negative-scene rule')
        pair = _public_pair(service, job_id, context['source'], candidate, witness)
        _recheck(service, job_id, payload, context)
        return {'schema': SCHEMA, 'job_id': job_id, 'pair': pair, 'base_digest': context['base_digest'],
                'selection_revision': context['result']['selection_revision'], 'review_mutations': 0}


def calibrate(service, job_id, payload):
    token = str(payload.get("request_id") or "")
    if not re.fullmatch(r"[a-f0-9]{32}", token):
        raise HTTPException(400, "A stable calibration request identity is required")
    directory = service.directory(job_id)
    identity = digest(payload)
    with LOCK, service.adapter.guard():
        previous = _operation(directory, token, identity)
        if previous is not None:
            return previous
        context = _capture(service, job_id, payload)
        pairs = _pairs(context, payload.get("pairs"))
        if any(not context['images'][item[side]]['objects'] for item in pairs for side in ['candidate', 'witness']):
            raise HTTPException(400, 'This lab calibrates labelled-object directions; explicit negatives keep their separate negative-scene rule')
        variants = _variants([payload.get("variant")])
        labels = payload.get("labels")
        if not isinstance(labels, list) or {item.get("case_id") for item in labels if isinstance(item, dict)} != {item["case_id"] for item in pairs}:
            raise HTTPException(400, "Review every selected direction before calibrating this recipe")
        submitted_protocol = payload.get('review_protocol') or {}
        if not isinstance(submitted_protocol, dict):
            raise HTTPException(400, 'Review protocol must describe the reported interaction history')
        visible_before = submitted_protocol.get('reported_scores_visible_before_all_judgments')
        if visible_before is not None and type(visible_before) is not bool:
            raise HTTPException(400, 'Score exposure must be a reported boolean when supplied')
        visibility = [item.get('scores_visible_when_judged') for item in labels if isinstance(item, dict)]
        if any(value is not None and type(value) is not bool for value in visibility):
            raise HTTPException(400, 'Each recorded score visibility must be a boolean when supplied')
        measured = _measure(service, job_id, context, pairs, variants, OBJECT_FLOORS)[0]
        try:
            profile = fit_thresholds(measured["binding"], labels, measured["rows"], review_authority="submitted_interactive_review",
                                     cohort_scope="Up to 24 selected labelled-object comparisons; explicit negatives retain their separate unchanged rule",
                                     scene_floors=(0.0,) if context['config'].scene_requirement == 'optional' else (.65, .75, .85, .90, .92, .95))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        profile.pop('profile_id', None)
        profile['review_protocol'] = {'reported_scores_visible_before_all_judgments': visible_before,
                                     'scores_visible_judgment_count': sum(value is True for value in visibility),
                                     'score_visibility_unreported_count': sum(value is None for value in visibility),
                                     'authority': 'Client-reported interaction history; no independent proof of score blindness'}
        profile['qualification'] += ' Interactive score exposure is client-reported; this profile has no independent score-blindness audit.'
        profile['variant'] = variants[0]
        profile['profile_id'] = 'dpc_' + digest(profile)[:24]
        _recheck(service, job_id, payload, context)
        reference = {"job_id": job_id, "profile_id": profile["profile_id"]}
        profile["draft_config"] = _draft_config(context, variants[0], profile["thresholds"] or
            {key: getattr(context["config"], key) for key in ["object_threshold", "scene_threshold", "index_threshold"]},
            reference if profile["status"] == "diagnostic_calibrated" else None)
        record = {"schema": PROFILE_SCHEMA, "profile": profile, "profile_digest": digest(profile),
                  "base_digest": context["base_digest"], "variant": variants[0], "labels": labels,
                  "pairs": pairs, "measurements_digest": digest(measured["rows"]), "measurements": measured["rows"]}
        result = {"outcome": "committed", "receipt": {"request_id": token, "identity": identity, "profile_reference": reference,
                    "source_revision": context["source"]["revision"], "base_digest": context["base_digest"], "review_mutations": 0},
                  "profile": {**profile, "reference": reference}, "variant": variants[0],
                  "encoder_calls": 0, "head_training_calls": 0, "geometry_refits": 0}
        path = _database_path(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        with getattr(service.adapter, 'calibration_commit_guard', nullcontext)(), closing(sqlite3.connect(path, timeout=30)) as db:
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS profiles (profile_id TEXT PRIMARY KEY,content TEXT,content_digest TEXT)")
                db.execute("CREATE TABLE IF NOT EXISTS operations (request_id TEXT PRIMARY KEY,identity TEXT,response TEXT)")
                previous = db.execute("SELECT content_digest FROM profiles WHERE profile_id=?", (profile["profile_id"],)).fetchone()
                if previous and previous[0] != record["profile_digest"]:
                    raise HTTPException(409, "A saved profile identity changed; no replacement was published")
                db.execute("INSERT OR IGNORE INTO profiles VALUES (?,?,?)", (profile["profile_id"], json.dumps(record, sort_keys=True, allow_nan=False), record["profile_digest"]))
                db.execute("INSERT INTO operations VALUES (?,?,?)", (token, identity, json.dumps(result, sort_keys=True, allow_nan=False)))
        return result


def _referenced_profile(service, reference):
    if not isinstance(reference, dict) or not re.fullmatch(r"dpc_[a-f0-9]{24}", str(reference.get("profile_id") or "")):
        raise HTTPException(400, "Select an exact saved recipe calibration reference")
    path = _database_path(service.directory(reference.get("job_id")))
    if not path.is_file():
        raise HTTPException(409, "The referenced recipe calibration is unavailable")
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            row = db.execute("SELECT content FROM profiles WHERE profile_id=?", (reference["profile_id"],)).fetchone()
        if row is None:
            raise HTTPException(409, "The referenced recipe calibration is unavailable")
        record = json.loads(row[0])
    except (sqlite3.DatabaseError, ValueError) as error:
        raise HTTPException(409, "The referenced recipe calibration cannot be read") from error
    profile = record.get("profile")
    if record.get("schema") != PROFILE_SCHEMA or not isinstance(profile, dict) or digest(profile) != record.get("profile_digest") or profile.get("profile_id") != reference["profile_id"]:
        raise HTTPException(409, "The saved calibration changed; preserve it for inspection")
    return record


def prepare_calibrated_fits(service, reference, source, recipe, config):
    """Reuse the exact calibrated fit only for an unchanged source/producer.

The profile's original completed analysis remains the fit authority. A new
analysis must not silently replace it with a new population fit merely because
the raw pixels are cached. Saved fits enter derived recipe identity, never the
raw encoder identity. Final profile validation still checks actual assembly.
    """
    record = _referenced_profile(service, reference)
    profile = record['profile']
    expected_source = {"identity": source['identity'], "revision": source['revision'],
                       "inventory_digest": source['inventory_digest'], "labelmap_digest": digest(source['labelmap'])}
    if profile.get('status') != 'diagnostic_calibrated' or profile.get('feature_binding', {}).get('source') != expected_source:
        raise HTTPException(409, 'The source differs from this calibration; review and calibrate the current dataset recipe')
    if any(getattr(config, key) != value for key, value in profile['thresholds'].items()):
        raise HTTPException(409, 'The selected gates differ from this recipe calibration')
    original = _capture(service, reference['job_id'])
    if original['base_digest'] != record['base_digest']:
        raise HTTPException(409, 'The original calibrated fit changed; preserve it for inspection')
    fits, produced = {}, {'components': {}}
    for component in recipe['bbox']['components']:
        key = 'bbox_' + component['id']
        entry = copy.deepcopy(original['manifest']['components'].get(key))
        if not entry or digest(entry['raw_recipe']) != digest(component.get('raw_recipe')) or entry['dimension'] != component['dimension']:
            raise HTTPException(409, 'An object producer or descriptor shape changed; calibrate this exact representation again')
        alpha = validate_strength(component.get('adjustment_strength'), default=0 if component['adjustment'] == 'none' else 1)
        fits[key] = copy.deepcopy(entry['adjustment'])
        entry.update(weight=component['weight'], adjustment=effective_transform(entry['adjustment'], alpha))
        produced['components'][key] = entry
    for ordinal, component in enumerate(recipe['scene']['components']):
        key = 'scene_' + str(ordinal)
        entry = copy.deepcopy(original['manifest']['components'].get(key))
        if not entry or digest(entry['raw_recipe']) != digest(component.get('raw_recipe')):
            raise HTTPException(409, 'A whole-image producer or component order changed; calibrate this exact representation again')
        alpha = validate_strength(component.get('adjustment_strength'), default=0 if component['adjustment'] == 'none' else 1)
        fits[key] = copy.deepcopy(entry['adjustment'])
        entry.update(weight=component['weight'], adjustment=effective_transform(entry['adjustment'], alpha))
        produced['components'][key] = entry
    binding = feature_binding(source, recipe, produced, config)
    try:
        assert_profile_binding(profile, binding, profile['thresholds'])
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    recipe['saved_component_fits'] = fits
    recipe['saved_fit_binding'] = {'source_inventory_digest': source['inventory_digest'],
                                  'profile_feature_recipe_digest': profile['feature_recipe_digest'],
                                  'fits': {key: digest(value) for key, value in fits.items()}}
    verify(original['material'])
    return recipe


def validate_analysis_profile(service, reference, source, recipe, config):
    """Apply only a recipe-bound diagnostic profile after real feature assembly."""
    record = _referenced_profile(service, reference)
    profile = record['profile']
    manifest = recipe.get("component_manifest") or {}
    binding = feature_binding(source, recipe, manifest, config)
    thresholds = {name: getattr(config, name) for name in ["object_threshold", "scene_threshold", "index_threshold"]}
    try:
        assert_profile_binding(profile, binding, thresholds)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {"reference": reference, "status": profile["status"], "feature_recipe_digest": profile["feature_recipe_digest"],
            "thresholds": profile["thresholds"], "qualification": profile["qualification"],
            "review_authority": profile["review_authority"], "reviewed_direction_count": profile["reviewed_direction_count"],
            "label_counts": profile["label_counts"], "review_protocol": profile.get("review_protocol")}
