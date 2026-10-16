"""Explicit, source-preserving migration from a reviewed YOLO inventory.

Inventory does not open a hosted database. Apply only consumes the exact
inventory, records new attribution, and can resume using durable receipts.
Unknown sidecars must be mapped for preservation before any dataset is created.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from .artifacts import ArtifactStore, open_regular
from .errors import DomainError
from .caption_semantics import caption_lifecycle, caption_text
from .class_names import parse_class_lines
from .media import HostedMedia, inspect_media, prepare_yolo_objects
from .operations import HostedOperations
from .store import HostedStore, _json
from .types import Actor, MutationContext


FORMAT = "tator-hosted-migration-v3"
LEGACY_FORMAT = "tator-hosted-migration-v1"
LEGACY_FORMATS = {LEGACY_FORMAT, "tator-hosted-migration-v2"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ROLES = {"metadata", "groups", "recipe", "salad_weights", "salad_metadata", "caption_records"}


def digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _safe_path(path: str | Path) -> Path:
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise DomainError("migration_symlink_rejected", 422, {"path": str(path)})
    return path


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise DomainError("migration_path_invalid", 422)
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise DomainError("migration_path_invalid", 422, {"path": value})
    return value


def _file(path: Path) -> dict[str, Any]:
    path = _safe_path(path)
    before = path.stat()
    hasher = hashlib.sha256()
    with open_regular(path) as stream:
        descriptor = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino) != (descriptor.st_dev, descriptor.st_ino):
            raise DomainError("migration_source_changed", 409)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise DomainError("migration_source_changed", 409, {"path": str(path)})
    return {"sha256": hasher.hexdigest(), "size_bytes": before.st_size}


def _walk(root: Path) -> dict[str, Path]:
    root = _safe_path(root)
    if not root.is_dir():
        raise DomainError("migration_root_invalid", 422)
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = _safe_path(Path(directory) / name)
            mode = path.stat().st_mode
            if name in files and not stat.S_ISREG(mode):
                raise DomainError("migration_special_file_rejected", 422)
            if name in files:
                result[path.relative_to(root).as_posix()] = path
    return result


def _read(path: Path, maximum=16 * 1024 * 1024) -> str:
    with open_regular(_safe_path(path)) as stream:
        value = stream.read(maximum + 1)
    if len(value) > maximum:
        raise DomainError("migration_metadata_too_large", 413, {"path": str(path)})
    return value.decode("utf-8-sig")


def _objects(text: str, width: int, height: int, classes: list[str], *, clip: bool, key: str) -> list[dict]:
    result = []
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        try:
            label = int(fields[0])
            values = [float(value) for value in fields[1:]]
            if not 0 <= label < len(classes) or any(not math.isfinite(value) for value in values):
                raise ValueError("class or coordinate")
            polygon = None
            if len(values) == 4:
                cx, cy, w, h = values
                if w <= 0 or h <= 0:
                    raise ValueError("nonpositive box")
                xmin, ymin, xmax, ymax = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            elif len(values) >= 6 and len(values) % 2 == 0:
                polygon = list(zip(values[::2], values[1::2]))
                xmin, ymin = min(values[::2]), min(values[1::2])
                xmax, ymax = max(values[::2]), max(values[1::2])
            else:
                raise ValueError("unsupported YOLO row")
            roundoff = 8 * math.ulp(1.0)
            if min(xmin, ymin) < -roundoff or max(xmax, ymax) > 1 + roundoff:
                if not clip:
                    raise ValueError("out-of-bounds geometry; explicitly enable clipping if intended")
            xmin, ymin, xmax, ymax = [min(1.0, max(0.0, value)) for value in (xmin, ymin, xmax, ymax)]
            if xmin >= xmax or ymin >= ymax:
                raise ValueError("empty clipped box")
            # Convert endpoints first. Multiplying a normalized span separately
            # can reconstruct an edge just outside the image (e.g. 640 + 1 ulp).
            # Keep the store's strict pixel bounds; only round this conversion
            # inward when floating-point subtraction/addition crosses the edge.
            x, y, right, bottom = xmin * width, ymin * height, xmax * width, ymax * height
            w, h = right - x, bottom - y
            if x + w > width:
                w = math.nextafter(w, 0.0)
            if y + h > height:
                h = math.nextafter(h, 0.0)
            if w <= 0 or h <= 0:
                raise ValueError("empty pixel box")
            item = {"id": "legacy-" + digest([key, line_number, fields])[:24], "class_index": label,
                    "bbox": [x, y, w, h]}
            if polygon:
                item["polygon"] = [[min(1.0, max(0.0, x)) * width, min(1.0, max(0.0, y)) * height] for x, y in polygon]
            result.append(item)
        except (ValueError, OverflowError) as error:
            raise DomainError("migration_label_invalid", 422, {"image": key, "line": line_number, "reason": str(error)}) from error
    return result


def inventory(source_root: str | Path, *, mapping: dict | None = None, clip_boxes=False) -> dict:
    """Hash all declared inputs; return a reviewable plan without creating state.

    mapping.files assigns an explicit role/association to otherwise unknown
    sidecars. mapping.image_metadata preserves per-image group IDs and other
    reviewed metadata. External model/recipe files require individual paths in
    mapping.external_artifacts. An external annotation overlay may be supplied
    explicitly as mapping.overlay_root; no global registry is scanned.
    """
    root = _safe_path(source_root)
    mapping = json.loads(_json(mapping or {}))
    if set(mapping) - {"files", "image_metadata", "external_artifacts", "overlay_root", "label_coordinate_space", "image_coordinate_spaces"}:
        raise DomainError("migration_mapping_unknown_fields", 422)
    coordinate_spaces = mapping.get("image_coordinate_spaces", {})
    if (not isinstance(coordinate_spaces, dict)
            or any(value not in ("encoded", "exif_display") for value in coordinate_spaces.values())
            or mapping.get("label_coordinate_space") not in (None, "encoded", "exif_display")):
        raise DomainError("label_coordinate_space_invalid", 422)
    paths = _walk(root)
    if mapping.get("overlay_root"):
        overlay = _safe_path(mapping["overlay_root"])
        if overlay == root or overlay.is_relative_to(root):
            raise DomainError("migration_overlay_must_be_external", 422)
        paths.update({"@overlay/" + name: path for name, path in _walk(overlay).items()})
    files = {name: {"path": str(path), **_file(path), "role": "unknown"} for name, path in sorted(paths.items())}
    explicit = mapping.get("files", {})
    if not isinstance(explicit, dict):
        raise DomainError("migration_mapping_invalid", 422)
    for name, definition in explicit.items():
        _relative(name)
        if name not in files or not isinstance(definition, dict) or definition.get("role") not in ROLES:
            raise DomainError("migration_mapping_invalid", 422, {"path": name})
        files[name].update({"role": definition["role"], "association": definition.get("association", {})})
    for number, definition in enumerate(mapping.get("external_artifacts", [])):
        if not isinstance(definition, dict) or definition.get("role") not in ROLES:
            raise DomainError("migration_mapping_invalid", 422)
        path = _safe_path(definition["path"])
        if path.is_relative_to(root):
            raise DomainError("migration_external_artifact_inside_source", 422)
        files[f"@external/{number}"] = {"path": str(path), **_file(path), "role": definition["role"], "association": definition.get("association", {})}
    labelmaps = [name for name in ("labelmap.txt", "classes.txt", "labels.txt") if name in files]
    if not labelmaps:
        raise DomainError("migration_labelmap_missing", 422)
    lists = [parse_class_lines(_read(paths[name])) for name in labelmaps]
    classes = lists[0]
    if any(value != classes for value in lists) or len({name.casefold() for name in classes}) != len(classes):
        raise DomainError("migration_labelmap_ambiguous", 422)
    for name in labelmaps:
        files[name]["role"] = "labelmap"
    for name in ("data.yaml", "data.yml"):
        if name in files:
            # Preserve arbitrary training config, but do not execute or follow
            # its paths. Actual membership is the explicitly inventoried tree.
            files[name]["role"] = "metadata"
    flat = any(name.startswith("images/") for name in files)
    split = any(name.startswith(prefix + "/images/") for name in files for prefix in ("train", "val", "test"))
    if flat == split:
        raise DomainError("migration_layout_ambiguous", 422)
    layout = "flat" if flat else "split"
    active_overlay = "@overlay/" if mapping.get("overlay_root") else ".annotation_overlay/"
    images = []
    coordinate_blockers = []
    # A basename sidecar can be selected only if its image association is
    # unique in that scope. Mirrored paths remain available for collisions.
    image_names = [name for name in files if PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES
                   and (name.startswith("images/") if flat else
                        any(name.startswith(split + "/images/") for split in ("train", "val", "test")))]
    consumed_labels = set()
    for name, entry in files.items():
        parts = PurePosixPath(name).parts
        if PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if flat and parts[0] == "images":
            split_name, relative = "unsplit", PurePosixPath(*parts[1:])
        elif not flat and len(parts) > 2 and parts[0] in {"train", "val", "test"} and parts[1] == "images":
            split_name, relative = parts[0], PurePosixPath(*parts[2:])
        else:
            continue
        entry["role"] = "image"
        media_info = inspect_media(paths[name])
        if media_info.source_sha256 != entry["sha256"]:
            raise DomainError("migration_source_changed", 409)
        width, height = media_info.width, media_info.height
        key = relative.as_posix() if flat else f"{split_name}/{relative}"
        source_label = (PurePosixPath("labels") if flat else PurePosixPath(split_name, "labels")) / relative.with_suffix(".txt")
        # Flat legacy DQE overlays use the normalized split token 'train'.
        overlay_split = "train" if flat else split_name
        candidates = [f"@overlay/labels/{overlay_split}/{relative.with_suffix('.txt')}", f".annotation_overlay/labels/{overlay_split}/{relative.with_suffix('.txt')}", source_label.as_posix()]
        found = [candidate for candidate in candidates if candidate in files]
        for candidate in found:
            if candidate in consumed_labels:
                raise DomainError("migration_label_collision", 422, {"path": candidate})
            consumed_labels.add(candidate)
            files[candidate]["role"] = "labels"
        current_label = next((candidate for candidate in found if candidate.startswith(active_overlay) or candidate == source_label.as_posix()), None)
        label_text = _read(Path(files[current_label]["path"])) if current_label else ""
        coordinate_space = coordinate_spaces.get(key, mapping.get("label_coordinate_space"))
        if label_text.strip() and media_info.orientation != 1 and coordinate_space is None:
            coordinate_blockers.append({"kind": "label_coordinate_space_required", "images": [key], "orientation": media_info.orientation})
            objects, coordinate_provenance = [], None
        else:
            objects, coordinate_provenance = prepare_yolo_objects(label_text, classes, media_info,
                key=key, label_coordinate_space=coordinate_space, clip=clip_boxes)
        captions = []
        caption_paths = []
        # Only the selected overlay is current. Keep other overlay bytes as
        # history, but do not resurrect them when the selected one has no text.
        for prefix in (active_overlay, "", *(p for p in ("@overlay/", ".annotation_overlay/") if p != active_overlay)):
            for alias in ("text_labels", "captions", "caption_labels"):
                scopes = ([prefix + f"{alias}/{overlay_split}/", prefix + f"{alias}/"] if prefix
                          else ([f"{split_name}/{alias}/", f"{alias}/{split_name}/", f"{alias}/"] if not flat else [f"{alias}/"]))
                # All qualified candidates beat a basename fallback, within
                # each alias. Source split roots use split-local uniqueness.
                for scope in scopes:
                    scoped_names = [n for n in image_names if n.startswith(split_name + "/images/")] if (not flat and (scope.startswith(split_name + "/") or f"/{split_name}/" in scope)) else image_names
                    unique_relative = sum(PurePosixPath(*PurePosixPath(n).parts[1 if flat else 2:]).with_suffix(".txt") == relative.with_suffix(".txt") for n in scoped_names) == 1
                    if unique_relative:
                        caption_paths.append(scope + relative.with_suffix(".txt").as_posix())
                for scope in scopes:
                    scoped_names = [n for n in image_names if n.startswith(split_name + "/images/")] if (not flat and (scope.startswith(split_name + "/") or f"/{split_name}/" in scope)) else image_names
                    if sum(PurePosixPath(n).stem == relative.stem for n in scoped_names) == 1:
                        caption_paths.append(scope + relative.stem + ".txt")
        effective_caption = None
        effective_caption_file = None
        for candidate in dict.fromkeys(caption_paths):
            if candidate in files:
                files[candidate]["role"] = "caption_text"
                value = _read(paths[candidate])
                selected = not candidate.startswith(("@overlay/", ".annotation_overlay/")) or candidate.startswith(active_overlay)
                current = selected and effective_caption is None
                captions.append({"caption": value, "source_file": candidate, "source": "legacy_import",
                                 "lifecycle_status": "active" if current else "superseded", "is_primary": current and bool(value.strip())})
                if current:
                    effective_caption = value.strip()  # Empty overlay deliberately clears the source caption.
                    effective_caption_file = candidate
        images.append({"key": key, "file": name, "media_hash": entry["sha256"], "size_bytes": entry["size_bytes"], "width": width, "height": height,
                       "split": split_name, "label_file": current_label, "objects": objects, "captions": captions,
                       "media": media_info.as_dict(), "media_coordinate_provenance": coordinate_provenance,
                       "effective_caption": effective_caption,
                       "effective_caption_file": effective_caption_file,
                       "metadata": mapping.get("image_metadata", {}).get(key, {})})
    if not images:
        raise DomainError("migration_no_images", 422)
    keys = {row["key"]: row for row in images}
    unmatched_metadata = (set(mapping.get("image_metadata", {})) | set(coordinate_spaces)) - set(keys)
    if unmatched_metadata:
        raise DomainError("migration_image_metadata_unmatched", 422, sorted(unmatched_metadata))
    caption_names = {"captions.jsonl", "text_labels/captions.jsonl", ".annotation_overlay/captions/captions.jsonl", "@overlay/captions/captions.jsonl"}
    caption_files = {name for name, entry in files.items() if name in caption_names or entry["role"] == "caption_records"}
    if mapping.get("overlay_root"):
        # An external overlay is a complete record store, not an incremental
        # patch. An empty/missing file or omitted ID must not revive old records.
        current_caption_files = {name for name in caption_files if name.startswith("@overlay/")}
    else:
        current_caption_files = {name for name in caption_files if name.startswith(".annotation_overlay/")}
        if not (root / ".annotation_overlay").is_dir():
            current_caption_files = set(caption_files)
    caption_ids = {}
    for name, entry in files.items():
        if name not in caption_files:
            continue
        entry["role"] = "caption_records"
        entry["current_caption_source"] = name in current_caption_files
        # Shadowed files remain hashed, copied and registered as artifacts.
        # Their records are history, not additional active caption alternatives.
        if name not in current_caption_files:
            continue
        for line in _read(Path(entry["path"])).splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise DomainError("migration_caption_invalid", 422, {"path": name})
            # Match the native _caption_lifecycle_status normalizer, including
            # null/empty/unknown values defaulting to active. An explicit
            # deletion tombstone must remain inactive rather than be revived.
            record["lifecycle_status"] = caption_lifecycle(record)
            key = record.get("image_key") or record.get("image_name") or record.get("image")
            aliases = {key} if key in keys else set()
            if flat and isinstance(key, str) and key.startswith("train/") and key[6:] in keys:
                aliases.add(key[6:])
            if len(aliases) > 1:
                raise DomainError("migration_caption_ambiguous", 422, {"path": name, "image_key": key})
            if aliases:
                key = next(iter(aliases))
            if key not in keys:
                raise DomainError("migration_caption_unmatched", 422, {"path": name, "image_key": key})
            caption_id = record.get("id")
            if caption_id:
                identity = (key, str(caption_id))
                if identity in caption_ids:
                    if caption_ids[identity] != record:
                        raise DomainError("migration_caption_id_conflict", 422, {"path": name, "image_key": key, "caption_id": caption_id})
                    continue
                caption_ids[identity] = record
            keys[key]["captions"].append({**record, "migration_source_file": name})
    for row in images:
        active = [record for record in row["captions"] if record.get("lifecycle_status", "active") == "active"]
        # Text sidecars define the primary caption independently of alternate
        # records. An explicitly empty text file must also clear primary flags.
        if row["effective_caption"] is not None:
            primary = next((record for record in active if record.get("source_file") == row["effective_caption_file"] and "migration_source_file" not in record and record.get("is_primary")), None)
        else:
            # Match the native record reader's primary/creation/ID ordering.
            primary = min(active, key=lambda record: (not bool(record.get("is_primary")), str(record.get("created_at") or ""), str(record.get("id") or "")), default=None)
        for record in row["captions"]:
            record["is_primary"] = record is primary
    duplicates = {}
    for row in images:
        duplicates.setdefault(row["media_hash"], []).append(row["key"])
    # Membership is pixel-unique in hosted datasets. Do not silently erase a
    # second split or source record while converting a legacy duplicate.
    blockers = coordinate_blockers + [{"kind": "duplicate_media", "images": group} for group in duplicates.values() if len(group) > 1]
    blockers.extend({"kind": "unmapped_file", "path": name} for name, entry in files.items() if entry["role"] == "unknown")
    plan = {"format": FORMAT, "source_root": str(root), "mapping": mapping, "clip_boxes": bool(clip_boxes), "layout": layout,
            "classes": classes, "images": images, "files": files, "blockers": blockers,
            "missing_labels": [row["key"] for row in images if row["label_file"] is None]}
    plan["plan_sha256"] = digest(plan)
    return plan


class HostedMigrator:
    def __init__(self, store: HostedStore, artifacts: ArtifactStore):
        self.store, self.artifacts = store, artifacts
        self.operations = HostedOperations(store, artifacts)
        self.media = HostedMedia(store, artifacts)
        with store.transaction() as con:
            con.execute("CREATE TABLE IF NOT EXISTS hosted_migration_runs (id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, plan_sha256 TEXT NOT NULL, mode TEXT NOT NULL, name TEXT NOT NULL, dataset_id TEXT, state TEXT NOT NULL, manifest_json TEXT NOT NULL, created_at REAL NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS hosted_migration_items (run_id TEXT NOT NULL REFERENCES hosted_migration_runs(id), image_key TEXT NOT NULL, image_id TEXT NOT NULL, version_id TEXT NOT NULL, PRIMARY KEY(run_id,image_key))")

    def apply(self, actor: Actor, plan: dict, *, name: str, mode="incoming", accept_existing_labels=False) -> dict:
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
        if mode not in {"incoming", "approved-baseline"} or (mode == "approved-baseline" and not accept_existing_labels):
            raise DomainError("migration_baseline_attestation_required", 422)
        if plan.get("format") not in {FORMAT, *LEGACY_FORMATS} or plan.get("plan_sha256") != digest({key: value for key, value in plan.items() if key != "plan_sha256"}):
            raise DomainError("migration_plan_invalid", 422)
        run_id = "migration-" + digest({"plan": plan["plan_sha256"], "actor": actor.user_id, "name": name, "mode": mode})[:40]
        payload = {"run_id": run_id, "plan_sha256": plan["plan_sha256"], "name": name, "mode": mode, "accept_existing_labels": bool(accept_existing_labels)}
        # Completed migrations are immutable receipts, including when a newer
        # caption/media interpretation would produce a different inventory.
        if self.store.raw_receipt(actor, run_id + "-finish"):
            return self.store._command(actor, run_id + "-finish", "finish_migration", payload, lambda con: None, admin=True)
        if plan["format"] in LEGACY_FORMATS:
            raise DomainError("migration_plan_requires_review", 409)
        fresh = inventory(plan["source_root"], mapping=plan["mapping"], clip_boxes=plan["clip_boxes"])
        if fresh != plan:
            raise DomainError("migration_source_changed", 409)
        if plan["blockers"] or (mode == "approved-baseline" and plan["missing_labels"]):
            raise DomainError("migration_inventory_unresolved", 409, {"blockers": plan["blockers"], "missing_labels": plan["missing_labels"]})
        def start(con):
            con.execute("INSERT INTO hosted_migration_runs VALUES (?,?,?,?,?,NULL,'running',?,?)", (run_id, actor.user_id, plan["plan_sha256"], mode, name, _json(plan), self.store.clock()))
            return {"run_id": run_id}
        self.store._command(actor, run_id + "-start", "start_migration", payload, start, admin=True)
        ds = self.store.create_dataset(actor, name, plan["classes"], run_id + "-dataset")
        with self.store.transaction() as con:
            self.store._actor(con, actor, admin=True)
            con.execute("UPDATE hosted_migration_runs SET dataset_id=? WHERE id=?", (ds["id"], run_id))
        associations = []
        for filename, source in plan["files"].items():
            if source["role"] == "image":
                continue
            with self.artifacts.exclusive():
                reference = self.artifacts.put_file(_safe_path(source["path"]), expected_sha256=source["sha256"], expected_size=source["size_bytes"])
                self.operations.register_artifact(actor, reference, "migration_" + source["role"], ds["id"], run_id + "-file-" + digest(filename)[:32])
            associations.append({"source_file": filename, "role": source["role"], "sha256": reference.sha256, "size_bytes": reference.size_bytes, "association": source.get("association", {})})
        for row in plan["images"]:
            operation = run_id + "-image-" + digest(row["key"])[:24]
            if self.store.raw_receipt(actor, operation + "-done"):
                continue
            annotations = {"objects": [{**{key: value for key, value in obj.items() if key != "class_index"}, "class_id": ds["classes"][obj["class_index"]]["id"]} for obj in row["objects"]],
                           "captions": row["captions"], "migration": {"run_id": run_id, "source_key": row["key"], "label_file": row["label_file"], "source_sha256": row["media_hash"], "geometry_clipping_authorized": plan["clip_boxes"], "metadata": row["metadata"]}}
            active_captions = [record for record in row["captions"] if record.get("lifecycle_status", "active") == "active"]
            if row.get("media_coordinate_provenance"):
                annotations["media_coordinate_provenance"] = row["media_coordinate_provenance"]
            if row.get("effective_caption") is not None:
                annotations["caption"] = row["effective_caption"]
            elif active_captions:
                primary = next((record for record in active_captions if record.get("is_primary")), active_captions[0])
                annotations["caption"] = caption_text(primary)
            source = plan["files"][row["file"]]
            with self.artifacts.exclusive():
                view = self.media.stage(_safe_path(source["path"]))
                if view["source_sha256"] != source["sha256"] or view["source_size_bytes"] != source["size_bytes"]:
                    raise DomainError("migration_source_changed", 409)
                imported = self.store.import_image(actor, ds["id"], view["source_sha256"], row["key"], annotations, operation + "-import", width=row["width"], height=row["height"], size_bytes=row["size_bytes"], split=row["split"])
            completed = imported
            if mode == "approved-baseline":
                prior_approval = self.store.raw_receipt(actor, operation + "-approve")
                if prior_approval:
                    completed = prior_approval["result"]
                else:
                    prior_save = self.store.raw_receipt(actor, operation + "-save")
                    expected_version = prior_save["result"]["current_version"] if prior_save else imported["current_version"]
                    current = self.store.image(actor, imported["id"])
                    if current["current_version"] != expected_version:
                        raise DomainError("migration_image_changed", 409, {"image_id": current["id"]})
                    lease = self.store.acquire_editor(actor, current["id"], run_id, current["current_version"], current["generation"])
                    if prior_save is None:
                        lease = self.store.save(actor, MutationContext(lease["id"], lease["current_version"], lease["generation"], run_id, operation + "-save"), annotations, complete=True, negative=not bool(row["objects"]))
                    completed = self.store.approve(actor, MutationContext(lease["id"], lease["current_version"], lease["generation"], run_id, operation + "-approve"), "Administrator accepted existing labels as the migration baseline; original review history is not asserted.")
            def done(con):
                con.execute("INSERT INTO hosted_migration_items VALUES (?,?,?,?)", (run_id, row["key"], completed["id"], completed["current_version"]))
                return {"image_id": completed["id"], "version_id": completed["current_version"]}
            self.store._command(actor, operation + "-done", "migration_image_done", {"run_id": run_id, "key": row["key"]}, done, admin=True)
        # Explicit preservation associations do not activate a legacy recipe or
        # pretend that its model has passed hosted-runtime qualification.
        settings_operation = run_id + "-settings"
        if not self.store.raw_receipt(actor, settings_operation):
            current = self.store.get_dataset(actor, ds["id"])
            self.store.patch_dataset_settings(actor, ds["id"], {"migration": {"run_id": run_id, "plan_sha256": plan["plan_sha256"], "mode": mode, "artifacts": associations}}, current["revision"], settings_operation)
        def finish(con):
            con.execute("UPDATE hosted_migration_runs SET state='complete' WHERE id=?", (run_id,))
            return {"run_id": run_id, "dataset_id": ds["id"], "mode": mode, "image_count": len(plan["images"]), "artifact_count": len(associations), "plan_sha256": plan["plan_sha256"], "state": "complete"}
        return self.store._command(actor, run_id + "-finish", "finish_migration", payload, finish, admin=True)


def inventory_report(plan: dict) -> str:
    lines = ["Hosted migration inventory", "==========================", "", f"Source: {plan['source_root']}", f"Plan SHA-256: {plan['plan_sha256']}", "",
             f"{len(plan['images'])} images; {len(plan['classes'])} classes; {len(plan['files'])} hashed source files.",
             "No hosted dataset has been created. Source files have not been changed.", "", "Unresolved items", "----------------", ""]
    lines.extend(f"* {row['kind']}: {row.get('path', row.get('images', ''))}" for row in plan["blockers"])
    if not plan["blockers"]:
        lines.append("None.")
    lines += ["", f"Images without label files: {len(plan['missing_labels'])}. These cannot enter approved-baseline mode.", "",
              "Apply defaults to Incoming. Approved-baseline mode requires explicit administrator acceptance of the existing labels.",
              "Recipe and SALAD files are preserved with their supplied associations; runtime activation is a separate step.", ""]
    return "\n".join(lines)
