"""Bridge hosted datasets into the admin workbench and apply its label edits back.

Export freezes a confirmed snapshot and publishes it under the workbench root as
a linked legacy dataset: images are hard links to the immutable working views,
labels are YOLO text, and hosted-bridge.json records each image's base version.
The legacy backend keeps every edit in its own overlay and never touches these
files. Apply reads that overlay, matches each edited label row to the base
version's objects (keeping their hosted IDs), and saves an administrator
canonical edit per image under a lease and an exact version precondition.
Images changed in the hosted workspace since their base are reported, never
overwritten.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import threading
import time
import traceback
import uuid
from pathlib import Path

import httpx

from .errors import DomainError
from .geometry import clamp_polygon, clip_box
from .store import _json
from .types import MutationContext

BRIDGE_DIR = "hosted-bridge"
GEOMETRY_TOLERANCE = 0.75  # pixels; overlay files round normalized values.
MATCH_IOU = 0.5


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix, iy = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)), max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _same_box(a, b):
    return all(abs(x - y) <= GEOMETRY_TOLERANCE for x, y in zip(a, b))


def parse_overlay(text: str, width: int, height: int, class_count: int) -> list[dict]:
    """YOLO bbox or polygon rows to pixel geometry; malformed rows are errors."""
    items = []
    for number, raw in enumerate(text.splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        try:
            index, values = int(parts[0]), [float(value) for value in parts[1:]]
        except ValueError:
            raise DomainError("workbench_label_invalid", 409, f"Unreadable label row {number}.") from None
        if not 0 <= index < class_count or len(values) < 4 or (len(values) > 4 and len(values) % 2):
            raise DomainError("workbench_label_invalid", 409, f"Unreadable label row {number}.")
        if len(values) == 4:
            cx, cy, w, h = values
            bbox = [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]
            polygon = None
        else:
            points = [[values[i] * width, values[i + 1] * height] for i in range(0, len(values), 2)]
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            bbox, polygon = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], points
        # Clipped to the image; a row with no area inside it is dropped.
        clipped = clip_box(*bbox, width, height)
        if clipped is None:
            continue
        items.append({"class_index": index, "bbox": clipped, "polygon": clamp_polygon(polygon, width, height) if polygon is not None else None})
    return items


def plan_objects(base_objects: list[dict], items: list[dict], class_ids: list[str]) -> dict:
    """Match overlay rows to base objects, keeping hosted object identity."""
    matched_base, assignment = set(), {}
    stats = {"unchanged": 0, "relabeled": 0, "moved": 0, "added": 0, "deleted": 0}
    conflict = None
    # Pass 1 and 2: identical geometry, same then different class.
    for same_class in (True, False):
        for row, item in enumerate(items):
            if row in assignment:
                continue
            for index, obj in enumerate(base_objects):
                if index in matched_base or not _same_box(obj["bbox"], item["bbox"]):
                    continue
                if same_class != (obj["class_id"] == class_ids[item["class_index"]]):
                    continue
                matched_base.add(index); assignment[row] = (index, "unchanged" if same_class else "relabeled"); break
    # Pass 3: moved or resized boxes, best overlap first.
    pairs = sorted(((_iou(obj["bbox"], item["bbox"]), row, index) for row, item in enumerate(items) if row not in assignment
                    for index, obj in enumerate(base_objects) if index not in matched_base), reverse=True)
    for overlap, row, index in pairs:
        if overlap < MATCH_IOU or row in assignment or index in matched_base:
            continue
        matched_base.add(index); assignment[row] = (index, "moved")
    objects = []
    for row, item in enumerate(items):
        class_id = class_ids[item["class_index"]]
        if row in assignment:
            index, kind = assignment[row]
            obj = copy.deepcopy(base_objects[index])
            if kind == "moved":
                if obj.get("polygon") and item["polygon"] is None:
                    conflict = "A polygon object was moved or resized in the workbench; edit polygons on the hosted canvas."
                obj["bbox"] = [round(value, 4) for value in item["bbox"]]
                if item["polygon"] is not None:
                    obj["polygon"] = [[round(x, 4), round(y, 4)] for x, y in item["polygon"]]
            if obj["class_id"] != class_id:
                obj["class_id"] = class_id
                kind = "relabeled" if kind == "unchanged" else kind
            stats[kind] += 1
        else:
            obj = {"id": "obj_" + uuid.uuid4().hex, "class_id": class_id, "bbox": [round(value, 4) for value in item["bbox"]]}
            if item["polygon"] is not None:
                obj["polygon"] = [[round(x, 4), round(y, 4)] for x, y in item["polygon"]]
            stats["added"] += 1
        objects.append(obj)
    stats["deleted"] = len(base_objects) - len(matched_base)
    return {"objects": objects, "stats": stats, "conflict": conflict}


ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
# A workbench copy is one pool of every confirmed image: the tools do not make
# or keep split decisions (the legacy layout calls that pool "train").
EXPORT_SPLITS = {"train": "train", "unsplit": "train", "val": "train", "test": "train"}
OVERLAY_LIMIT = 8 * 1024 * 1024
CHANGE_KEYS = ("relabeled", "moved", "added", "deleted")


class WorkbenchBridge:
    """Export hosted snapshots to the workbench and apply reviewed label edits back.

    Trust: the workbench folder is written by another process, so base versions,
    splits and classes always come from the hosted snapshot in SQLite, overlay
    files are opened without following symlinks, and apply saves only the exact
    overlay contents the administrator reviewed.
    """

    def __init__(self, store, artifacts, operations, root: str | Path | None, url: str):
        self.store, self.artifacts, self.operations = store, artifacts, operations
        self.root = Path(root).absolute() if root else None
        self.url = url
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._manifests: dict[str, dict] = {}
        with store.transaction() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS hosted_workbench_bridges (id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL,
                snapshot_id TEXT NOT NULL, path TEXT NOT NULL, legacy_dataset_id TEXT NOT NULL, legacy_registry_root TEXT NOT NULL,
                label TEXT NOT NULL, image_count INTEGER NOT NULL, excluded_images INTEGER NOT NULL, created_by TEXT NOT NULL,
                created_at REAL NOT NULL)""")
            con.execute("""CREATE TABLE IF NOT EXISTS hosted_workbench_applied (bridge_id TEXT NOT NULL, image_id TEXT NOT NULL,
                overlay_sha256 TEXT NOT NULL, version_id TEXT NOT NULL, applied_by TEXT NOT NULL, applied_at REAL NOT NULL,
                PRIMARY KEY(bridge_id, image_id))""")

    @property
    def available(self) -> bool:
        return bool(self.root and self.url)

    def _require(self, actor):
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
        if not self.available:
            raise DomainError("workbench_not_configured", 503, "This server has no workbench service configured.")

    def _client(self, timeout):
        from .workbench import client_options, require_private_socket
        require_private_socket(self.url)
        return httpx.Client(**client_options(self.url, asynchronous=False), timeout=timeout)

    def _inside(self, path: Path, root: Path) -> bool:
        return path.resolve().is_relative_to(root.resolve())

    # ---- jobs -----------------------------------------------------------
    def _start(self, actor, kind, target, args, *, bridge_id=None, dataset_id=None) -> dict:
        job = {"id": uuid.uuid4().hex, "kind": kind, "actor_id": actor.user_id, "state": "running", "done": 0, "total": None,
               "result": None, "error": None, "started_at": time.time(), "bridge_id": bridge_id, "dataset_id": dataset_id}
        with self._lock:
            # One copy of a dataset at a time: a second tab follows the running export.
            running = next((other for other in self._jobs.values() if dataset_id is not None and other["kind"] == kind
                            and other["state"] == "running" and other["dataset_id"] == dataset_id), None)
            if running is not None:
                if running["actor_id"] != actor.user_id:
                    raise DomainError("workbench_export_running", 409, "Another administrator is making a workbench copy of this dataset.")
                return self.job(actor, running["id"])
            if bridge_id is not None and any(other["kind"] == kind and other["state"] == "running" and other["bridge_id"] == bridge_id
                                             for other in self._jobs.values()):
                raise DomainError("workbench_apply_running", 409, "Changes from this workbench copy are already being applied.")
            # Finished jobs are only progress records; keep the registry small.
            for old_id in [key for key, value in self._jobs.items() if value["state"] != "running" and time.time() - value["started_at"] > 86400]:
                self._jobs.pop(old_id, None)
            self._jobs[job["id"]] = job
        def run():
            try:
                job["result"] = target(job, *args); job["state"] = "completed"
            except Exception as exc:
                job["state"] = "failed"
                job["error"] = exc.detail if isinstance(exc, DomainError) and isinstance(exc.detail, str) else getattr(exc, "code", "workbench_job_failed")
                if not isinstance(exc, DomainError):
                    traceback.print_exc()
        threading.Thread(target=run, daemon=True, name="hosted-workbench-" + kind).start()
        return self.job(actor, job["id"])

    def job(self, actor, job_id) -> dict:
        self._require(actor)
        job = self._jobs.get(job_id)
        if job is None or job["actor_id"] != actor.user_id:
            raise DomainError("workbench_job_not_found", 404)
        return {key: job[key] for key in ("id", "kind", "state", "done", "total", "result", "error")}

    # ---- bridges --------------------------------------------------------
    def bridges(self, actor, dataset_id: str) -> list[dict]:
        self._require(actor)
        with self.store.read_connection() as con:
            rows = con.execute("SELECT * FROM hosted_workbench_bridges WHERE dataset_id=? ORDER BY created_at DESC", (dataset_id,)).fetchall()
            applied = dict(con.execute("SELECT bridge_id, count(*) FROM hosted_workbench_applied GROUP BY bridge_id").fetchall())
        listed = [{**{key: row[key] for key in ("id", "dataset_id", "snapshot_id", "legacy_dataset_id", "label", "image_count",
                                                 "excluded_images", "created_at")},
                   "applied_images": applied.get(row["id"], 0)} for row in rows]
        if listed:
            listed[0]["hosted_changes"] = self._hosted_changes(listed[0]["id"])
        return listed

    def _hosted_changes(self, bridge_id):
        """How the hosted dataset moved on since this copy: confirmed images edited, added or gone."""
        try:
            row, bridge = self._bridge(bridge_id)
        except DomainError:
            return None
        with self.store.read_connection() as con:
            current = {image_id: (split, version) for image_id, split, version in con.execute(
                "SELECT id, split, confirmed_version FROM hosted_images WHERE dataset_id=? AND state='confirmed' "
                "AND id NOT IN (SELECT image_id FROM hosted_archived_images)", (row["dataset_id"],))
                if split in EXPORT_SPLITS}
            # Versions this copy applied itself are its own edits, not hosted changes.
            own = dict(con.execute("SELECT image_id, version_id FROM hosted_workbench_applied WHERE bridge_id=?", (bridge_id,)).fetchall())
        frozen = bridge["images"]
        return {"edited": sum(1 for image_id, (_split, version) in current.items()
                              if image_id in frozen and version not in (frozen[image_id]["version_id"], own.get(image_id))),
                "added": sum(1 for image_id in current if image_id not in frozen),
                "removed": sum(1 for image_id in frozen if image_id not in current)}

    def start_export(self, actor, dataset_id: str) -> dict:
        self._require(actor)
        if not ID_PATTERN.fullmatch(dataset_id or ""):
            raise DomainError("dataset_not_found", 404)
        return self._start(actor, "export", self._export, (actor, dataset_id), dataset_id=dataset_id)

    def _real_directory(self, path: Path) -> Path:
        if path.is_symlink():
            raise DomainError("workbench_root_unsafe", 409, "A workbench copy folder is a symlink; remove it and try again.")
        path.mkdir(exist_ok=True, mode=0o700)
        if path.is_symlink() or not path.is_dir() or not self._inside(path, self.root):
            raise DomainError("workbench_root_unsafe", 409, "A workbench copy folder is not a real directory in the workbench root.")
        return path

    def _registered(self, client, legacy_id):
        try:
            listed = client.get("/datasets").json()
        except (httpx.HTTPError, ValueError):
            return None
        return next((entry for entry in listed if isinstance(entry, dict) and entry.get("id") == legacy_id), None) if isinstance(listed, list) else None

    def _export(self, job, actor, dataset_id):
        bridge_id = "wb_" + uuid.uuid4().hex[:16]
        snapshot = self.store.create_snapshot(actor, dataset_id, "workbench-" + bridge_id, "Admin workbench copy")
        manifest = snapshot["manifest"]
        exported = [{**image, "split": EXPORT_SPLITS[image["split"]]} for image in manifest["images"] if image["split"] in EXPORT_SPLITS]
        excluded = len(manifest["images"]) - len(exported)
        job["total"] = len(exported)
        base = self._real_directory(self._real_directory(self.root / BRIDGE_DIR) / dataset_id)
        stage, final = base / (".staging-" + bridge_id), base / snapshot["snapshot_id"]
        job["staging"] = stage.name
        with self._lock:
            live = {other.get("staging") for other in self._jobs.values() if other["state"] == "running"}
        for stale in base.glob(".staging-*"):  # left by an interrupted export, never a running one
            if stale.name not in live and not stale.is_symlink() and time.time() - stale.lstat().st_mtime > 6 * 3600:
                shutil.rmtree(stale, ignore_errors=True)
        stage.mkdir(mode=0o700)
        try:
            for entry in self.operations._snapshot_entries({**manifest, "images": exported}, unsplit_as_train=True):
                target = stage / entry.path
                if not self._inside(target, stage) or target.resolve() == stage.resolve():
                    raise DomainError("workbench_export_path_invalid", 409)
                target.parent.mkdir(parents=True, exist_ok=True)
                if entry.artifact is not None:
                    try:
                        self.artifacts.link_to(entry.artifact, target)
                    except OSError as error:
                        if error.errno == 18:  # EXDEV
                            raise DomainError("workbench_root_other_disk", 409, "The workbench root must be on the same disk as hosted storage.") from None
                        raise
                    job["done"] += 1
                else:
                    with target.open("xb") as output:
                        output.write(entry.data)
            # For people only; apply reads the snapshot from the hosted database.
            (stage / "hosted-bridge.json").write_text(_json({"bridge_id": bridge_id, "dataset_id": dataset_id,
                                                            "snapshot_id": snapshot["snapshot_id"], "excluded_test_images": excluded}))
            os.rename(stage, final)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        with self.store.read_connection() as con:
            row = con.execute("SELECT name FROM hosted_datasets WHERE id=?", (dataset_id,)).fetchone()
        dataset_name = row["name"] if row else dataset_id
        label = f"{dataset_name} · hosted copy {time.strftime('%Y-%m-%d %H:%M')}"
        legacy_id = "hosted_" + bridge_id
        entry, problem = None, "The workbench service is not running."
        try:
            with self._client(1800) as client:
                try:
                    response = client.post("/datasets/register_path", json={"path": str(final), "label": label, "dataset_id": legacy_id,
                        "context": f"Hosted dataset {dataset_name}, snapshot {snapshot['snapshot_id']}. Apply edits from the admin workbench bar.",
                        "strict": False})
                    if response.status_code == 200:
                        entry = response.json()
                    else:
                        problem = f"The workbench refused the dataset: {response.text[:300]}"
                except (httpx.HTTPError, ValueError):
                    pass
                # A timeout can still finish registering it later; give the
                # workbench time before treating the copy as unregistered.
                for _ in range(24):
                    if isinstance(entry, dict) and entry.get("id") == legacy_id:
                        break
                    entry = self._registered(client, legacy_id)
                    if entry is None:
                        time.sleep(5)
        except httpx.HTTPError:
            entry = None
        registry = entry.get("registry_root") if entry else None
        if not isinstance(registry, str) or not self._inside(Path(registry), self.root):
            if entry is None:
                shutil.rmtree(final, ignore_errors=True)
                raise DomainError("workbench_register_failed", 503, problem)
            raise DomainError("workbench_register_failed", 502, "The workbench registered the copy outside its storage root.")
        with self.store.transaction() as con:
            con.execute("INSERT INTO hosted_workbench_bridges VALUES (?,?,?,?,?,?,?,?,?,?,?)", (bridge_id, dataset_id, snapshot["snapshot_id"],
                        str(final), legacy_id, registry, label, len(exported), excluded, actor.user_id, time.time()))
        return {"bridge_id": bridge_id, "legacy_dataset_id": legacy_id, "label": label, "image_count": len(exported), "excluded_images": excluded}

    def _bridge(self, bridge_id):
        """The bridge row plus its frozen snapshot, read from hosted SQLite only."""
        with self.store.read_connection() as con:
            row = con.execute("SELECT * FROM hosted_workbench_bridges WHERE id=?", (bridge_id,)).fetchone()
            if row is None:
                raise DomainError("workbench_bridge_not_found", 404)
            cached = self._manifests.get(row["snapshot_id"])
            if cached is None:
                snapshot = con.execute("SELECT dataset_id, manifest_json FROM hosted_snapshots WHERE id=?", (row["snapshot_id"],)).fetchone()
                manifest = json.loads(snapshot["manifest_json"])
                classes = sorted(manifest["classes"], key=lambda item: item["position"])
                cached = {"dataset_id": snapshot["dataset_id"], "class_ids": [item["id"] for item in classes],
                          "images": {image["id"]: {"split": EXPORT_SPLITS[image["split"]], "version_id": image["version_id"]}
                                     for image in manifest["images"] if image["split"] in EXPORT_SPLITS and ID_PATTERN.fullmatch(image["id"])}}
                with self._lock:
                    if len(self._manifests) >= 4:
                        self._manifests.pop(next(iter(self._manifests)))
                    self._manifests[row["snapshot_id"]] = cached
        if cached["dataset_id"] != row["dataset_id"]:
            raise DomainError("workbench_bridge_invalid", 409)
        if not Path(row["path"]).is_dir():
            raise DomainError("workbench_copy_missing", 409, "This workbench copy was deleted in the workbench. Make a new copy.")
        return row, cached

    # ---- changes and apply ------------------------------------------------
    def _overlay(self, labels: Path, split: str, image_id: str):
        """Read one overlay file without following links. Returns bytes, None or an error text."""
        try:
            fd = os.open(labels / split / (image_id + ".txt"), os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return None
        except OSError:
            return "The workbench label file is not a regular file."
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > OVERLAY_LIMIT:
                return "The workbench label file is not a regular file."
            return handle.read(OVERLAY_LIMIT + 1)

    def _plans(self, bridge_id):
        row, bridge = self._bridge(bridge_id)
        registry = Path(row["legacy_registry_root"])
        labels = registry / ".annotation_overlay" / "labels"
        plans = []
        if not labels.is_dir():
            return row, bridge, plans
        if not self._inside(registry, self.root) or not all(self._inside(labels / split, registry) for split in ("train", "val")):
            raise DomainError("workbench_root_unsafe", 409, "The workbench label folder points outside its storage root.")
        with self.store.read_connection() as con:
            applied = {r["image_id"]: r for r in con.execute("SELECT * FROM hosted_workbench_applied WHERE bridge_id=?", (bridge_id,))}
            active = {r["id"] for r in con.execute("SELECT id FROM hosted_classes WHERE dataset_id=? AND retired=0", (row["dataset_id"],))}
            for image_id, meta in sorted(bridge["images"].items()):
                raw = self._overlay(labels, meta["split"], image_id)
                if raw is None:
                    continue
                digest = hashlib.sha256(raw if isinstance(raw, bytes) else raw.encode()).hexdigest()
                previous = applied.get(image_id)
                if previous is not None and previous["overlay_sha256"] == digest:
                    continue
                base_version = previous["version_id"] if previous is not None else meta["version_id"]
                image = con.execute("SELECT i.*, m.width, m.height FROM hosted_images i JOIN hosted_media m ON m.sha256=i.media_hash WHERE i.id=?", (image_id,)).fetchone()
                # What the administrator reviews: these exact labels against this base version.
                review = hashlib.sha256(f"{digest}:{base_version}".encode()).hexdigest()
                plan = {"image_id": image_id, "filename": image["filename"] if image else image_id, "overlay_sha256": digest,
                        "review_sha256": review, "base_version": base_version, "conflict": None, "stats": None, "empty": False}
                if isinstance(raw, str):
                    plans.append({**plan, "conflict": raw}); continue
                if image is None or image["dataset_id"] != row["dataset_id"]:
                    plans.append({**plan, "conflict": "This image is no longer in the hosted dataset."}); continue
                version = con.execute("SELECT annotations_json, negative FROM hosted_annotation_versions WHERE id=?", (base_version,)).fetchone()
                body = json.loads(version["annotations_json"])
                try:
                    items = parse_overlay(raw.decode("utf-8"), image["width"], image["height"], len(bridge["class_ids"]))
                except (DomainError, UnicodeDecodeError) as error:
                    plans.append({**plan, "conflict": getattr(error, "detail", None) or "The workbench label file is unreadable."}); continue
                result = plan_objects(body.get("objects", []), items, bridge["class_ids"])
                if not any(result["stats"][key] for key in CHANGE_KEYS):
                    continue  # saved in the workbench without a real change
                plan.update(stats=result["stats"], conflict=result["conflict"], empty=not result["objects"], negative=bool(version["negative"]))
                plans.append(plan)
                if con.execute("SELECT 1 FROM hosted_archived_images WHERE image_id=?", (image_id,)).fetchone():
                    plan["conflict"] = "This image is no longer in the hosted dataset."
                elif image["state"] != "confirmed":
                    plan["conflict"] = "This image is not confirmed in the hosted dataset any more."
                elif image["current_version"] != base_version:
                    plan["conflict"] = "This image was edited in the hosted workspace after the workbench copy. Make a new copy to edit it."
                elif any(obj["class_id"] not in active for obj in result["objects"]):
                    plan["conflict"] = "This image uses a class that was retired. Make a new copy."
                plan["_body"], plan["_objects"], plan["_generation"] = body, result["objects"], image["generation"]
        return row, bridge, plans

    def changes(self, actor, bridge_id) -> dict:
        self._require(actor)
        row, _, plans = self._plans(bridge_id)
        visible = []
        totals = {"images": 0, "conflicts": 0, "needs_empty_confirmation": 0, "unchanged": 0, "relabeled": 0, "moved": 0, "added": 0, "deleted": 0}
        for plan in plans:
            stats = plan["stats"] or {}
            totals["images"] += 1
            totals["conflicts"] += bool(plan["conflict"])
            totals["needs_empty_confirmation"] += bool(not plan["conflict"] and plan["empty"])
            for key in ("unchanged", *CHANGE_KEYS):
                totals[key] += stats.get(key, 0)
            visible.append({key: plan[key] for key in ("image_id", "filename", "review_sha256", "base_version", "conflict", "stats", "empty")})
        return {"bridge": {key: row[key] for key in ("id", "dataset_id", "snapshot_id", "label", "image_count", "excluded_images")},
                "totals": totals, "images": visible}

    def start_apply(self, actor, bridge_id, *, expected: dict, confirm_empty: bool, operation_id: str) -> dict:
        """Apply exactly what was reviewed: {image_id: review_sha256} (labels and base version)."""
        self._require(actor)
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", operation_id or ""):
            raise DomainError("invalid_operation_id", 422)
        if (not isinstance(expected, dict) or not expected or len(expected) > 200000
                or not all(isinstance(k, str) and ID_PATTERN.fullmatch(k) and isinstance(v, str) and DIGEST_PATTERN.fullmatch(v)
                           for k, v in expected.items())):
            raise DomainError("workbench_review_required", 422, "Review the changes, then apply the reviewed images.")
        self._bridge(bridge_id)
        return self._start(actor, "apply", self._apply, (actor, bridge_id, dict(expected), bool(confirm_empty), operation_id), bridge_id=bridge_id)

    def _apply(self, job, actor, bridge_id, expected, confirm_empty, operation_id):
        _, _, plans = self._plans(bridge_id)
        pending = {plan["image_id"]: plan for plan in plans if not plan["conflict"]}
        todo = [pending[image_id] for image_id, digest in expected.items()
                if image_id in pending and pending[image_id]["review_sha256"] == digest]
        outcomes = {"applied": 0, "skipped_empty": 0, "changed_since_review": len(expected) - len(todo), "failed": []}
        job["total"] = len(todo)
        editor_id = "workbench-apply-" + operation_id[:40]
        for plan in todo:
            if plan["empty"] and not confirm_empty:
                outcomes["skipped_empty"] += 1; job["done"] += 1; continue
            image_id = plan["image_id"]
            command = "wba_" + hashlib.sha256(f"{operation_id}:{image_id}:{plan['overlay_sha256']}".encode()).hexdigest()[:40]
            lease = saved = None
            try:
                lease = self.store.acquire_editor(actor, image_id, editor_id, expected_version=plan["base_version"], generation=plan["_generation"])
                context = MutationContext(image_id, plan["base_version"], lease["generation"], editor_id, command)
                body = copy.deepcopy(plan["_body"]); body["objects"] = plan["_objects"]
                payload = {"bridge_id": bridge_id, "image_id": image_id, "overlay_sha256": plan["overlay_sha256"], "base_version": plan["base_version"]}
                def run(con, context=context, body=body, plan=plan):
                    result = self.store._save(con, actor, context, body, complete=True, negative=plan["empty"])
                    con.execute("INSERT OR REPLACE INTO hosted_workbench_applied VALUES (?,?,?,?,?,?)",
                                (bridge_id, plan["image_id"], plan["overlay_sha256"], result["current_version"], actor.user_id, time.time()))
                    return {"image_id": plan["image_id"], "current_version": result["current_version"], "generation": result["generation"]}
                saved = self.store._command(actor, command, "workbench_apply_image", payload, run, admin=True)
                outcomes["applied"] += 1
            except DomainError as error:
                outcomes["failed"].append({"image_id": image_id, "filename": plan["filename"], "error": error.code,
                                           "detail": error.detail if isinstance(error.detail, str) else None})
            finally:
                if lease is not None:  # never leave the image reserved, saved or not
                    version, generation = (saved["current_version"], saved["generation"]) if saved else (plan["base_version"], lease["generation"])
                    try:
                        self.store.release_editor(actor, MutationContext(image_id, version, generation, editor_id, command + "-release"))
                    except DomainError:
                        pass  # the lease expires on its own
            job["done"] += 1
        return outcomes
