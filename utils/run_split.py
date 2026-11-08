"""Train/validation split chosen by a training run, never written to the dataset.

Datasets do not have to carry splits. When a training run needs validation
images that the dataset does not declare, it chooses them here: images are
grouped by capture metadata (sequence, video, flight, ...), wholly unsplit
groups are ordered by a stable hash of the dataset key and group name, and
groups join validation in that order while they fit about a tenth of those
images (the smallest group when none fits), only when there is more than one
group. A group whose classes would appear only in validation
goes back to training. The caller records the summary in its run log/metadata.

Shared by the hosted native trainers (services/hosted/native_analysis.py) and
the legacy workbench trainers (YOLO, RF-DETR, SAM3, Qwen), so a dataset opened
in either place gets the same rule. This module only reads files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from utils.capture_metadata import CAPTURE_GROUP_FIELDS, portable_capture_metadata

RUN_VALIDATION_SHARE = 0.1
RUN_SPLIT_RULE = "unsplit images split for this run only; capture groups kept together; dataset splits kept"
SIDECAR_MAX_BYTES = 8 * 1024 * 1024


def capture_identities(metadata: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Capture-group identities, (field, value), from flat capture metadata."""
    portable = portable_capture_metadata(metadata if isinstance(metadata, Mapping) else {})
    return tuple((key, str(portable[key])) for key in CAPTURE_GROUP_FIELDS if key in portable)


def annotation_capture_groups(annotations: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Identities of a hosted annotation body: migration metadata, then capture metadata."""
    migration = annotations.get("migration", {})
    migration = migration.get("metadata", {}) if isinstance(migration, dict) else {}
    captured = annotations.get("capture_metadata", {})
    return capture_identities({**(migration if isinstance(migration, dict) else {}),
                               **(captured if isinstance(captured, dict) else {})})


def choose_run_split(items: Sequence[Mapping[str, Any]], dataset_key: str, *,
                     share: float = RUN_VALIDATION_SHARE) -> Tuple[List[str], Dict[str, Any]]:
    """Choose a split for every item; returns (splits aligned with items, summary).

    Each item has "id", "groups" (capture identities: (field, value) pairs or
    strings), "classes" (class ids or names) and "split" ("train", "val",
    "test" or "unsplit", default "unsplit"). Declared splits are kept and unsplit
    members of a capture group follow their group's single declared split; a
    group whose members declare different splits leaves its unsplit members
    unsplit. Wholly unsplit groups are ordered by sha256 of
    "<dataset_key>:<smallest group name>"; in that order, groups join validation
    while they fit `share` of those images (at least one image; the smallest group
    when none fits), when there is more than one such group.
    """
    parent = list(range(len(items)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def key(identity):
        return identity if isinstance(identity, str) else tuple(identity)

    def name(identity):
        return identity if isinstance(identity, str) else "=".join(str(part) for part in identity)

    identities = [list(item.get("groups") or ()) for item in items]
    first: Dict[Any, int] = {}
    for index, owned in enumerate(identities):
        for identity in owned:
            parent[find(index)] = find(first.setdefault(key(identity), index))
    names: Dict[int, List[str]] = {}
    for index, item in enumerate(items):
        names.setdefault(find(index), []).extend([name(identity) for identity in identities[index]]
                                                 or ["image=" + str(item["id"])])
    groups: Dict[int, List[int]] = {}
    for index in range(len(items)):
        groups.setdefault(find(index), []).append(index)
    declared_splits = [str(item.get("split") or "unsplit") for item in items]
    splits = list(declared_splits)
    automatic = []
    for root, members in groups.items():
        declared = {splits[index] for index in members} - {"unsplit"}
        if len(declared) == 1:
            target = declared.pop()
            for index in members:
                splits[index] = target if splits[index] == "unsplit" else splits[index]
        elif not declared:
            automatic.append((hashlib.sha256(f"{dataset_key}:{min(names[root])}".encode()).hexdigest(), members))
    automatic.sort()
    pool = sum(len(members) for _, members in automatic)
    # Groups join validation in hash order while they fit the target, so one long
    # video cannot take far more than the share; if none fits, the smallest does.
    target = max(1, round(share * pool))
    validation, chosen = [], 0
    for _, members in automatic:
        goes_to_validation = len(automatic) > 1 and chosen + len(members) <= target
        validation.append(goes_to_validation)
        chosen += len(members) if goes_to_validation else 0
    if len(automatic) > 1 and not any(validation):
        validation[min(range(len(automatic)), key=lambda position: len(automatic[position][1]))] = True
    item_classes = [set(item.get("classes") or ()) for item in items]

    def classes(members):
        return set().union(*(item_classes[index] for index in members))

    in_training = set().union(*(classes(members) for (_, members), held in zip(automatic, validation) if not held),
                              *(item_classes[index] for index, split in enumerate(splits) if split == "train"))
    for position, ((_, members), held) in enumerate(zip(automatic, validation)):
        if held and classes(members) - in_training and sum(validation) > 1:
            validation[position] = False
            in_training |= classes(members)
    for (_, members), held in zip(automatic, validation):
        for index in members:
            splits[index] = "val" if held else "train"
    return splits, {"rule": RUN_SPLIT_RULE, "validation_share": share,
                    "images_split_for_run": declared_splits.count("unsplit"),
                    "train_images": splits.count("train"), "validation_images": splits.count("val"),
                    "test_images": splits.count("test")}


def sidecar_capture_groups(annotation_dirs: Iterable[Path], image_stem: str) -> Tuple[Tuple[str, str], ...]:
    """Capture identities from a per-image annotation sidecar, if one exists.

    Hosted workbench copies keep `<split>/annotations/<image>.json` holding the
    hosted annotation body ({"annotations": {...}}). Image files there may be
    named `<original stem>__<hosted image id>`, so the part after the last "__"
    is tried as well. Symlinks and oversized files are ignored.
    """
    stems = [image_stem] + ([image_stem.rsplit("__", 1)[1]] if "__" in image_stem else [])
    for directory in annotation_dirs:
        for stem in stems:
            path = Path(directory) / f"{stem}.json"
            try:
                if path.is_symlink() or not path.is_file() or path.stat().st_size > SIDECAR_MAX_BYTES:
                    continue
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(document, dict):
                continue
            body = document.get("annotations") if isinstance(document.get("annotations"), dict) else document
            return annotation_capture_groups(body) or capture_identities(document)
    return ()


def yolo_label_classes(label_path: Path) -> set:
    """Class indices used in a YOLO label file (empty when missing)."""
    try:
        lines = Path(label_path).read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return set()
    found = set()
    for line in lines:
        parts = line.split()
        if parts:
            try:
                found.add(int(float(parts[0])))
            except ValueError:
                continue
    return found


def split_unsplit_pool(items: Sequence[Mapping[str, Any]], dataset_key: str) -> Optional[Tuple[List[bool], Dict[str, Any]]]:
    """Legacy trainers: hold out validation from a dataset that has none.

    Every item is treated as unsplit. Returns (is_validation per item, summary),
    or None when fewer than two items exist. The summary's validation_images is
    0 when the pool is a single capture group, so no honest holdout exists.
    """
    if len(items) < 2:
        return None
    splits, summary = choose_run_split([{**item, "split": "unsplit"} for item in items], dataset_key)
    held = [split == "val" for split in splits]
    summary = {**summary, "applies_when": "the dataset has no validation images"}
    if all(held):
        # The first group can be small and the next one the rest of the pool;
        # both then land in validation. Keep the run trainable and say so.
        held = [False] * len(held)
        summary.update(train_images=len(held), validation_images=0,
                       note="every capture group fell in validation, so none was held out")
    return held, summary


def describe_run_split(summary: Mapping[str, Any]) -> str:
    """One training-log line for a legacy run split."""
    if not summary.get("validation_images"):
        reason = summary.get("note") or f"its {summary.get('train_images', 0)} images form one capture group, so none could be held out"
        return f"Run split: the dataset has no validation images and {reason} for this run"
    return (f"Run split: {summary.get('train_images', 0)} train / {summary.get('validation_images', 0)} validation images "
            "chosen for this run (the dataset has no validation images; capture groups kept together; "
            f"validation share {summary.get('validation_share')}; dataset files unchanged)")
