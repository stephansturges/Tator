"""Detections and object descriptors for object-aware Incoming diversity.

Runs only inside the private ingestion child. The dataset's detector proposes
boxes on the Incoming candidates; those boxes and the confirmed human boxes are
described with the DINOv3 crop component of the pruning object recipe, through
the same producer and descriptor cache as dataset pruning. Detections are
evidence for one analysis result: nothing here writes an annotation, a draft or
a dataset setting.
"""
from __future__ import annotations

import math
import shutil
import time
import traceback
from pathlib import Path

from .analysis import _inside
from .errors import DomainError

MAX_BOXES_PER_IMAGE = 50
MIN_BOX_SIDE = 2.0
DETECTOR_CONFIDENCE_DEFAULT = 0.5
OBJECT_COMPONENT = "dinov3"
OBJECT_RECIPE_ID = "dinov3_crop_v1"
RECIPE_KEYS = ("object_recipe", "object_components", "object_preprocessing")
PARAMETERS = ("object_aware", "object_weight", "detector_confidence")


def detector_confidence(value, default=DETECTOR_CONFIDENCE_DEFAULT):
    """Lowest detector confidence a box needs to count, 0.05..0.99."""
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.05 <= value <= 0.99:
        raise DomainError("ingestion_detector_confidence_invalid", 422, "Detection confidence must be a number from 0.05 to 0.99.")
    return float(value)


def validate_parameters(parameters):
    """Check the object-mode request fields; return whether the mode is on."""
    from services.data_ingestion import object_aware_enabled, object_weight_value
    try:
        enabled = object_aware_enabled(parameters.get("object_aware"))
        object_weight_value(parameters.get("object_weight"))
    except ValueError as error:
        raise DomainError("ingestion_" + str(error), 422, "Check the scene and object balance: it must be a number from 0 to 1.") from error
    detector_confidence(parameters.get("detector_confidence"))
    return enabled


def _box_line(box, width, height):
    """A normalised label line for one detection (x, y, width, height in pixels)."""
    left, top = max(0.0, float(box[0])), max(0.0, float(box[1]))
    right, bottom = min(float(width), float(box[0]) + float(box[2])), min(float(height), float(box[1]) + float(box[3]))
    if right - left < MIN_BOX_SIDE or bottom - top < MIN_BOX_SIDE:
        return None
    values = [(left + right) / 2 / width, (top + bottom) / 2 / height, (right - left) / width, (bottom - top) / height]
    return [min(1.0, max(0.0, value)) for value in values]


def _detect(request, manifest, candidates, directory, threshold, should_stop, report):
    """Confident boxes per candidate image, from the dataset's detector."""
    from .adapters import DatasetDetectorAdapter
    plan = request["object_detector"]
    checkpoint = _inside(directory / plan["relative_path"], directory)
    definition = {**plan["model"], "checkpoint": str(checkpoint)}
    adapter = DatasetDetectorAdapter(config={"dataset_id": manifest["dataset_id"], "media_root": request["canonical_root"],
        "model": definition, "class_bindings": plan["class_bindings"], "dataset_labelmap": plan["dataset_labelmap"]}, device=request["device"])
    found = {}
    try:
        for index, image in enumerate(candidates):
            if should_stop():
                raise RuntimeError("cancelled")
            working = image.get("working_media") or None
            output = adapter.execute("detect_full", {"media_path": image["media_path"],
                "media_sha256": (working or {}).get("sha256", image["media_hash"]), "working_media": working,
                "source_media_hash": image["media_hash"], "conf": threshold, "max_det": 300}, {"dataset_id": manifest["dataset_id"]})
            boxes = [item for item in output["detections"] if item.get("score") is not None and float(item["score"]) >= threshold
                     and _box_line(item["bbox"], image["width"], image["height"]) is not None]
            boxes.sort(key=lambda item: -float(item["score"]))
            found[image["id"]] = (len(boxes), boxes[:MAX_BOXES_PER_IMAGE])
            if index % 16 == 0:
                report(index + 1, len(candidates))
    finally:
        adapter.close()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return found


def _describe(api, images, recipe_config, source_identity, work, cancel_event, report):
    """One unit descriptor per box, in ``images`` order; None where none could be made."""
    import numpy as np
    from services.dataset_pruning import digest
    from services.dataset_pruning_adapter import TatorPruningAdapter
    from services.dataset_pruning_cache import DescriptorCache
    from services.dataset_pruning_components import build_recipes, object_records, raw_component_recipe
    recipes = build_recipes(TatorPruningAdapter(api), {"analysis_job_id": "", "config": {
        **{key: recipe_config[key] for key in RECIPE_KEYS if key in recipe_config}, "evidence_mode": "objects_only", "object_adjustment": "raw"}},
        {"identity": source_identity})
    component = next((item for item in recipes["bbox"]["components"] if item["id"] == OBJECT_COMPONENT), None)
    if component is None:
        raise RuntimeError("object_recipe_component_missing")
    preprocessing, payload = recipes["bbox"]["preprocessing"], recipes["bbox"]["training_payload"]
    records = object_records(api, images, preprocessing)
    recipe = raw_component_recipe(api, component, preprocessing, payload)
    cache = DescriptorCache(api.CLASS_ANALYSIS_ROOT / "pruning" / "cache")
    keys = [cache.key(record["_image_sha256"], recipe, record["geometry_key"]) for record in records]
    vectors = [None] * len(records)
    for start in range(0, len(keys), 2048):
        vectors[start:start + 2048] = cache.get_many(keys[start:start + 2048])
    reused = sum(vector is not None for vector in vectors)
    missing = list({keys[index]: index for index, vector in enumerate(vectors) if vector is None}.values())
    report(f"Describing objects: {reused} reused, {len(missing)} to encode")
    if missing:
        head = {"encoder_type": component["encoder"], "encoder_model": component["model"], "normalize_embeddings": True,
                "embedding_aggregation": "pooled", "dinov3_pooling": component["pooling"], "cradio_pooling": component["pooling"],
                "cradio_output_mode": "summary_only", "qualified_producer_identity": recipe["producer"]}
        proxy = api.ClassAnalysisJob(job_id="ingestion-objects", cancel_event=cancel_event)
        work.mkdir(parents=True, exist_ok=True)
        encoded = api._class_analysis_stream_encode_records([records[index] for index in missing], {**payload, **preprocessing},
            job=proxy, head=head, batch_size=int(api._class_analysis_feature_batch_plan(payload, head)["batch_size"]), out_dir=work)
        fresh = {}
        for start in range(0, len(missing), 256):
            rows = np.asarray(encoded[start:start + 256], np.float32)
            norms = np.linalg.norm(np.where(np.isfinite(rows), rows, 0.0), axis=1)
            good = np.isfinite(rows).all(axis=1) & (norms > 1e-8)
            entries = [(keys[missing[start + offset]], rows[offset] / norms[offset]) for offset in np.flatnonzero(good)]
            cache.put_many(entries)
            fresh.update(entries)
        vectors = [fresh.get(keys[index]) if vector is None else vector for index, vector in enumerate(vectors)]
        if getattr(encoded, "_mmap", None) is not None:
            encoded._mmap.close()
        shutil.rmtree(work, ignore_errors=True)
    cache.close_maps()
    info = {"id": OBJECT_RECIPE_ID, "source_recipe": recipes["bbox"]["variant"],
            "components": [{"id": component["id"], "encoder": component["encoder"], "model": component["model"], "pooling": component["pooling"],
                            "weight": 1.0, "adjustment": "none", "dimension": int(len(next((v for v in vectors if v is not None), [])) or component.get("dimension") or 0)}],
            "preprocessing": preprocessing, "raw_recipe_digest": digest(recipe),
            "left_out": [item["id"] for item in recipes["bbox"]["components"] if item["id"] != OBJECT_COMPONENT] + ["size_correction"]}
    return records, vectors, info, {"reused": reused, "encoded": len(missing)}


def install_object_evidence(api, request, manifest, source, candidates, directory):
    """Give the ingestion job its object evidence provider; return the undo."""
    plan = request.get("object_detector")
    if not isinstance(plan, dict) or not plan.get("model") or not plan.get("relative_path"):
        raise DomainError("ingestion_object_detector_not_attached", 409)
    from services.dataset_pruning import geometry_key, parse_objects
    from services.dataset_pruning_cache import file_state
    threshold = detector_confidence(request["parameters"].get("detector_confidence"))
    incoming = {image["id"]: image for image in candidates["images"]}
    confirmed = {row["hosted_image_id"]: row for row in source["images"]}

    def provider(job, _legacy_request, candidate_rows, reference_rows):
        import numpy as np
        stop = job.cancel_event.is_set

        def identities(rows, population):
            found = [population.get(Path(str(row.get("filename") or "")).stem) for row in rows]
            if any(item is None for item in found):
                raise RuntimeError("object_evidence_identity_changed")
            return found

        candidate_images, reference_images = identities(candidate_rows, incoming), identities(reference_rows, confirmed)
        started = time.monotonic()
        try:
            detections = _detect(request, manifest, candidate_images, directory, threshold, stop,
                lambda done, total: api._data_ingestion_update(job, progress=0.88 + 0.04 * done / max(1, total), message=f"Detecting objects in Incoming image {done} of {total}"))
        except RuntimeError as error:
            if str(error) == "cancelled":
                raise
            traceback.print_exc()
            raise RuntimeError("object_detector_failed") from error
        except Exception as error:
            traceback.print_exc()
            raise RuntimeError("object_detector_failed") from error
        detect_seconds = time.monotonic() - started
        images, owners, public_boxes, skipped = [], [], [], 0
        for index, row in enumerate(reference_images):
            objects, errors = parse_objects(row["label_lines"], len(source["labelmap"]))
            kept = [item for item in objects if item["box"][2] * row["width"] >= MIN_BOX_SIDE and item["box"][3] * row["height"] >= MIN_BOX_SIDE]
            skipped += len(errors) + len(objects) - len(kept)
            if kept:
                images.append({"image_path": row["image_path"], "image_sha256": row["image_sha256"], "image_state": file_state(Path(row["image_path"])),
                    "width": row["width"], "height": row["height"], "label_lines": row["label_lines"], "objects": kept,
                    "image_id": row["hosted_image_id"], "image_relpath": row["image_relpath"], "split": row["split"]})
                owners.extend(("reference", index) for _ in kept)
        reference_box_count = len(owners)
        for index, image in enumerate(candidate_images):
            _total, boxes = detections[image["id"]]
            lines, objects = [], []
            for box in boxes:
                values = _box_line(box["bbox"], image["width"], image["height"])
                lines.append("0 " + " ".join(repr(value) for value in values))
                objects.append({"class_id": 0, "box": values, "label_row": len(objects), "geometry_key": geometry_key(values)})
                public_boxes.append({"bbox": [round(float(value), 2) for value in box["bbox"]], "class_name": box.get("class_name"),
                                     "class_id": box.get("stable_class_id"), "score": round(float(box["score"]), 4)})
            if objects:
                working = image.get("working_media") or {}
                images.append({"image_path": image["media_path"], "image_sha256": working.get("sha256", image["media_hash"]),
                    "image_state": file_state(Path(image["media_path"])), "width": image["width"], "height": image["height"],
                    "label_lines": lines, "objects": objects, "image_id": image["id"], "image_relpath": image["id"], "split": "train"})
                owners.extend(("candidate", index) for _ in objects)
        started = time.monotonic()
        try:
            records, vectors, recipe, cache = _describe(api, images, request.get("object_recipe") or {}, "hosted:" + manifest["dataset_id"],
                directory / "ingestion-object-work", job.cancel_event, lambda message: api._data_ingestion_update(job, progress=0.93, message=message))
        except RuntimeError as error:
            if str(error) in {"cancelled", "object_recipe_component_missing"}:
                raise
            traceback.print_exc()
            raise RuntimeError("object_encoder_failed") from error
        except Exception as error:
            traceback.print_exc()
            raise RuntimeError("object_encoder_failed") from error
        if len(records) != len(owners):
            raise RuntimeError("object_evidence_rows_changed")
        rows = {"reference": [], "candidate": []}
        for position, (kind, index) in enumerate(owners):
            if vectors[position] is not None:
                rows[kind].append((position, index))
        width = len(next((vector for vector in vectors if vector is not None), []))

        def matrix(kind):
            return np.asarray([vectors[position] for position, _ in rows[kind]], np.float32).reshape(len(rows[kind]), width)

        candidate_positions = [position - reference_box_count for position, _ in rows["candidate"]]
        plan_model = plan["model"]
        return {"reference_box_embeddings": matrix("reference"), "reference_box_image": [index for _, index in rows["reference"]],
                "candidate_box_embeddings": matrix("candidate"), "candidate_box_owner": [index for _, index in rows["candidate"]],
                "candidate_boxes": [public_boxes[position] for position in candidate_positions],
                "candidate_detection_totals": [detections[image["id"]][0] for image in candidate_images],
                "summary": {"detector": {"model_id": plan["model_id"], "model_identity": plan["model_identity"], "kind": plan_model["kind"],
                                "variant": plan_model.get("variant"), "checkpoint_sha256": plan_model["checkpoint_sha256"], "label_count": len(plan_model["labels"])},
                            "detection_mode": "full_image", "confidence_threshold": threshold, "max_boxes_per_image": MAX_BOXES_PER_IMAGE,
                            "min_box_side_px": MIN_BOX_SIDE, "object_recipe": recipe, "nearest_neighbour": "class_agnostic",
                            "reference": {"images_with_boxes": len({index for _, index in rows["reference"]}), "box_count": len(rows["reference"]),
                                          "boxes_skipped": skipped, "boxes_not_described": reference_box_count - len(rows["reference"])},
                            "candidates": {"box_count": len(rows["candidate"]), "boxes_not_described": len(owners) - reference_box_count - len(rows["candidate"]),
                                           "images_without_detections": sum(1 for image in candidate_images if not detections[image["id"]][1])},
                            "descriptor_cache": cache, "seconds": {"detect": round(detect_seconds, 2), "describe": round(time.monotonic() - started, 2)}}}

    previous = getattr(api, "_data_ingestion_object_evidence", None)
    api._data_ingestion_object_evidence = provider

    def restore():
        api._data_ingestion_object_evidence = previous
    return restore
