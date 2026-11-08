#!/usr/bin/env python3
"""Stage independent workflow-trial inputs from a frozen canonical snapshot.

This tool never opens a hosted database or changes source artifacts. Review the
generated inventory, then use hosted_migrate apply in Incoming mode. Original
encoded pixels are preserved; current source identities remain provenance only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.hosted.artifacts import open_regular, validate_digest
from services.hosted.class_names import validate_snapshot_classes
from services.hosted.caption_semantics import caption_lifecycle
from services.hosted.media import FORMATS, inspect_media
from services.hosted.migration import inventory, inventory_report


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _close(left, right):
    return len(left) == len(right) and all(
        math.isfinite(float(a)) and math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-7)
        for a, b in zip(left, right))


def _label(obj, classes, width, height):
    if obj.get("class_id") not in classes:
        raise ValueError("Source object refers to an absent class")
    x, y, w, h = map(float, obj["bbox"])
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("Source object has invalid geometry")
    polygon = obj.get("polygon")
    if polygon:
        if len(polygon) < 3 or any(len(point) != 2 for point in polygon):
            raise ValueError("Source polygon is invalid")
        xs, ys = zip(*polygon)
        if not _close([min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)], [x, y, w, h]):
            raise ValueError("Polygon and its independent bounding box cannot round-trip through YOLO")
        values = [value / size for px, py in polygon for value, size in ((px, width), (py, height))]
    else:
        values = [(x+w/2)/width, (y+h/2)/height, w/width, h/height]
    return str(classes[obj["class_id"]]) + " " + " ".join(format(v, ".17g") for v in values)


def _caption_rows(annotations, key):
    """Normalize legal strings/nulls and retain source addresses across collections.

    The migrator uses one captions collection. Distinct trial IDs avoid collisions
    between legal source collections; the complete original record stays in the
    image's source provenance. Primary text comes from the explicit text sidecar.
    """
    rows = []
    for field in ("captions", "alternate_captions", "text_labels"):
        records = annotations.get(field)
        records = [] if records is None else records
        if not isinstance(records, list):
            raise ValueError(f"Unsupported caption container: {key} / {field}")
        for index, original in enumerate(records):
            if not isinstance(original, (str, dict)):
                raise ValueError(f"Unsupported caption record: {key} / {field}[{index}]")
            record = {"caption": original, "source": "manual"} if isinstance(original, str) else dict(original)
            identifier = "trial-caption-" + hashlib.sha256(_json([key, field, index]).encode()).hexdigest()[:32]
            record.update(id=identifier, image_key=key, lifecycle_status=caption_lifecycle(original),
                          workflow_trial_source={"collection": field, "index": index,
                                                 "record_id": original.get("id") if isinstance(original, dict) else None})
            rows.append(record)
    return rows


def _compare(plan, selected, keys, class_indices, source_object_ids):
    if plan["blockers"] or plan["missing_labels"]:
        raise ValueError("Trial inventory has unresolved inputs: " + _json(plan["blockers"]))
    parsed = {row["key"]: row for row in plan["images"]}
    if set(parsed) != set(keys.values()):
        raise ValueError("Trial inventory selection changed")
    trial_object_ids = {obj["id"] for row in parsed.values() for obj in row["objects"]}
    if trial_object_ids & source_object_ids:
        raise ValueError("Trial object identity unexpectedly matches source dataset")
    for source in selected:
        row = parsed[keys[source["id"]]]
        original = source["annotations"]
        if row["media_hash"] != source["media_hash"]:
            raise ValueError("Trial media differs from original encoded bytes")
        if (row["width"], row["height"]) != (source["width"], source["height"]):
            raise ValueError("Trial dimensions changed")
        if row["split"] != "unsplit":  # imports never carry splits; they are decided later
            raise ValueError("Trial split changed")
        if len(row["objects"]) != len(original["objects"]):
            raise ValueError("Trial object count changed")
        for before, after in zip(original["objects"], row["objects"]):
            if class_indices[before["class_id"]] != after["class_index"] or not _close(before["bbox"], after["bbox"]):
                raise ValueError("Trial box or class changed")
            old_polygon, new_polygon = before.get("polygon") or [], after.get("polygon") or []
            if len(old_polygon) != len(new_polygon) or any(not _close(a, b) for a, b in zip(old_polygon, new_polygon)):
                raise ValueError("Trial polygon changed")
            if before.get("id") == after["id"]:
                raise ValueError("Trial object identity unexpectedly matches its source")
        primary = original.get("caption") or ""
        if not isinstance(primary, str) or row["effective_caption"] != primary.strip():
            raise ValueError("Trial current caption changed")
        expected_records = _caption_rows(original, keys[source["id"]])
        imported_records = [record for record in row["captions"] if record.get("migration_source_file")]
        if len(imported_records) != len(expected_records):
            raise ValueError("Trial caption record count changed")
        for before, after in zip(expected_records, imported_records):
            # The migrator recomputes primary flags from the explicit sidecar.
            # All text, lifecycle, QA, review and custom metadata must survive.
            canonical = lambda value: {k:v for k,v in value.items() if k not in {"is_primary", "migration_source_file"}}
            if canonical(before) != canonical(after):
                raise ValueError("Trial caption record or lifecycle changed")
        if row["metadata"]["workflow_trial"]["original_annotations"] != original:
            raise ValueError("Trial provenance is incomplete")


def stage_trial(manifest, artifacts_root, destination, plan_output, *, image_ids, namespace):
    """Stage an exact selection, returning an immutable migration plan and report.

    Extra annotation fields are retained explicitly in per-image provenance;
    labels/caption records remain active inputs. Invalid geometry is rejected.
    """
    if manifest.get("schema") != "hosted-canonical-snapshot-v1":
        raise ValueError("A frozen hosted canonical snapshot is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", namespace):
        raise ValueError("Use a short alphanumeric trial namespace")
    if not image_ids or len(image_ids) > 500 or len(set(image_ids)) != len(image_ids):
        raise ValueError("Select 1–500 distinct source images")
    validate_snapshot_classes(manifest)
    by_id = {row["id"]: row for row in manifest["images"]}
    if len(by_id) != len(manifest["images"]) or any(key not in by_id for key in image_ids):
        raise ValueError("Source image identity is ambiguous or missing")
    selected = [by_id[key] for key in image_ids]
    if len({row["media_hash"] for row in selected}) != len(selected):
        raise ValueError("Trial selection contains duplicate source pixels")
    splits = {row["split"] for row in selected}
    if not splits <= {"train", "val", "test", "unsplit"} or ("unsplit" in splits and len(splits) > 1):
        raise ValueError("Stage unsplit images separately from train/val/test images; no split is inferred")
    flat = splits == {"unsplit"}
    destination, plan_output = Path(destination).absolute(), Path(plan_output).absolute()
    report_output = plan_output.with_suffix(".rst")
    if destination.exists() or plan_output.exists() or report_output.exists():
        raise ValueError("Choose new staging and inventory paths; existing files are never replaced")
    if plan_output.resolve() == report_output.resolve():
        raise ValueError("Choose distinct plan and report paths; use a .json plan name")
    if any(path.resolve().is_relative_to(destination.resolve()) for path in (plan_output, report_output)):
        raise ValueError("Write the plan and report outside the input tree")
    artifacts_root = Path(artifacts_root).resolve(strict=True)
    if any(path.resolve().is_relative_to(artifacts_root) for path in (destination, plan_output, report_output)):
        raise ValueError("Never write trial inputs, plans or reports inside the source artifact store")
    # Read existing objects directly: constructing an ArtifactStore can create
    # directories, which is inappropriate for this read-only source operation.
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".workflow-trial-", dir=destination.parent))
    class_indices = {row["id"]: index for index, row in enumerate(manifest["classes"])}
    source_object_ids = {obj["id"] for row in manifest["images"] for obj in row["annotations"]["objects"]}
    mapping = {"label_coordinate_space": "encoded", "image_metadata": {}}
    keys, captions, warnings = {}, [], []
    try:
        (staging / "labelmap.txt").write_text("\n".join(row["name"] for row in manifest["classes"])+"\n")
        for source in selected:
            annotations = source["annotations"]
            digest = validate_digest(source["media_hash"])
            source_path = artifacts_root / "blobs" / "sha256" / digest[:2] / digest
            if not source_path.resolve(strict=True).is_relative_to(artifacts_root):
                raise ValueError("Source artifact escapes its store")
            with open_regular(source_path) as stream:
                pixels = stream.read()
            if len(pixels) != source["size_bytes"]:
                raise ValueError("Original media size does not match the snapshot")
            if hashlib.sha256(pixels).hexdigest() != source["media_hash"]:
                raise ValueError("Original media hash does not match the snapshot")
            media = inspect_media(pixels)
            if (media.width, media.height) != (source["width"], source["height"]):
                raise ValueError("Encoded source dimensions do not match snapshot")
            split = source["split"]
            suffix = FORMATS[media.format][1]
            stem = namespace + "-" + hashlib.sha256(source["id"].encode()).hexdigest()[:24]
            key = stem + suffix if flat else f"{split}/{stem}{suffix}"
            keys[source["id"]] = key
            image_root = staging if flat else staging / split
            for folder in ("images", "labels", "captions"):
                (image_root / folder).mkdir(parents=True, exist_ok=True)
            (image_root / "images" / (stem+suffix)).write_bytes(pixels)
            lines = [_label(obj, class_indices, media.width, media.height) for obj in annotations["objects"]]
            (image_root / "labels" / (stem+".txt")).write_text("\n".join(lines)+("\n" if lines else ""))
            primary = annotations.get("caption") or ""
            if not isinstance(primary, str):
                raise ValueError("Unsupported primary caption")
            # An explicit empty sidecar prevents migration from inventing a
            # primary caption from an alternate when the source has none.
            (image_root / "captions" / (stem+".txt")).write_text(primary)
            captions.extend(_caption_rows(annotations, key))
            extra = sorted(set(annotations) - {"objects", "caption", "captions", "alternate_captions", "text_labels"})
            object_extra = sorted({field for obj in annotations["objects"] for field in obj if field not in {"id", "class_id", "bbox", "polygon"}})
            if extra or object_extra:
                warnings.append({"image_id":source["id"],"annotation_fields":extra,"object_fields":object_extra,"disposition":"preserved in original_annotations provenance"})
            mapping["image_metadata"][key] = {"workflow_trial": {
                "namespace":namespace,"source_dataset_id":manifest["dataset_id"],
                "source_revision":manifest["revision"],"source_image_id":source["id"],
                "source_version_id":source["version_id"],"source_media_hash":source["media_hash"],
                "source_split":source["split"],"source_negative":bool(source["negative"]),
                "original_annotations":annotations}}
        if captions:
            (staging / "captions.jsonl").write_text("\n".join(_json(row) for row in captions)+"\n")
        _compare(inventory(staging, mapping=mapping), selected, keys, class_indices, source_object_ids)
        os.rename(staging, destination)
        plan = inventory(destination, mapping=mapping)
        _compare(plan, selected, keys, class_indices, source_object_ids)
        plan_output.parent.mkdir(parents=True, exist_ok=True)
        with plan_output.open("x") as stream:
            stream.write(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
        with report_output.open("x") as stream:
            stream.write(inventory_report(plan))
            stream.write("\nWorkflow trial\n--------------\n\nNew Incoming data; original approval is not inherited.\n")
            if warnings:
                stream.write("\nAdditional source fields retained in provenance::\n\n")
                stream.write("\n".join("    "+_json(row) for row in warnings)+"\n")
        return {"plan":str(plan_output),"report":str(report_output),"images":len(selected),
                "plan_sha256":plan["plan_sha256"],"source_to_trial_key":keys,"provenance_only_fields":warnings}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--image-ids", type=Path, required=True, help="JSON array of the exact selected source IDs")
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args(argv)
    result = stage_trial(json.loads(args.snapshot.read_text()), args.artifacts, args.destination,
        args.plan, image_ids=json.loads(args.image_ids.read_text()), namespace=args.namespace)
    print(_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
