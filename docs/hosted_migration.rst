Migrating a legacy YOLO dataset
==============================

This operator tool copies a reviewed source inventory into a new hosted dataset.
It never moves, deletes or rewrites source files. It does not discover the live
WALDO installation, stop a server or perform a production cutover.

Inventory first
---------------

Stop source annotation changes for inventory and apply. Choose the effective
dataset root and explicitly identify any external annotation overlay. Inventory
reads every file beneath that root, rejects symlinks and special files, hashes
the bytes and writes a JSON plan with a short reStructuredText report::

    python tools/hosted_migrate.py inventory \
      --source /data/legacy-dataset --output /review/migration-plan.json

Supported layouts are ``images/`` + ``labels/`` or
``train/images/`` + ``train/labels/`` with optional ``val`` and ``test`` siblings.
A ``labelmap.txt``, ``classes.txt`` or ``labels.txt`` supplies class names. Multiple
labelmaps must agree. YOLO boxes and polygons are converted to source pixels;
invalid class indices, nonfinite coordinates and invalid geometry stop the
inventory. Pixel extents are calculated from converted endpoints so a valid box
touching an image edge remains inside the strict hosted bounds despite floating
point roundoff. Out-of-bounds geometry is rejected unless ``--clip-boxes``
explicitly authorizes clipping. Source labels are retained unchanged as artifacts.

The selected overlay takes precedence over base labels and caption text. It is
the explicitly supplied external overlay, or otherwise the local
``.annotation_overlay/`` directory. Missing files fall back to the source;
selecting an external overlay never falls back to a stale local overlay. An
empty overlay file deliberately clears the corresponding labels or caption.
Within each selected scope, caption text aliases are ordered ``text_labels``,
``captions``, then ``caption_labels``. Mirrored image paths precede basename
fallbacks; a basename is used only when it identifies one image in that scope.
All recognized shadowed text remains retained as inactive history. An ambiguous
unassociated sidecar remains an inventory blocker until explicitly mapped.

Media and coordinate frames
---------------------------

Admission fully decodes each image, including JPEG scan data, before creating
membership. The input limit is 256 MiB and the decoded limit is 64 million
pixels; images are never silently resized. Canonical dimensions and annotations
use the encoded pixel grid. EXIF display orientation is ignored, including TIFF
decoder rotation. Original source bytes and their SHA-256 remain unchanged.
Browsers, exported datasets and native consumers use an immutable neutral
working representation when required; see ``hosted_operations.rst``.

Labeled images with EXIF orientation other than 1 require a declared source
coordinate frame. Add ``"label_coordinate_space": "encoded"`` or
``"label_coordinate_space": "exif_display"`` to the mapping, or declare frames
per image with ``"image_coordinate_spaces": {"train/car.jpg": "encoded"}``.
Per-image declarations take precedence. Display-frame boxes and polygons are
converted into encoded pixels and retain the source frame, orientation, source
hash and policy in annotation provenance. Unknown frames are inventory blockers;
the importer never guesses the meaning of existing labels. Empty label files
do not require a frame. Raw label files remain retained unchanged.

Historical hosted data can be inspected without rewriting it using
``HostedMedia(store, artifacts).audit()`` from an administrator's offline Python
session. It reports undecodable sources and oriented labeled versions without
coordinate provenance. Review each reported image against its original source.
An administrator can explicitly correct its annotations through the normal
editor and save a new version; the audit does not alter source bytes or history.
The report includes version IDs and examines historical versions as well as the
current one, so correcting current annotations does not erase an older warning.

Caption record history
----------------------

Caption JSONL is a complete record collection, not a patch to older files. An
explicit external overlay supplies that collection even when its JSONL file is
empty or absent. Otherwise an existing local overlay supplies the collection;
source JSONL is used only when there is no overlay directory. Shadowed JSONL
files remain immutable migration
artifacts, rather than becoming active caption alternatives. Active and inactive
records from the selected collection retain their IDs and lifecycle state.
Lifecycle spelling follows the native reader: case and surrounding whitespace
are normalized; missing, empty and unrecognized states mean active. An explicit
``deleted`` tombstone remains inactive. The original JSONL stays unchanged in
the retained artifacts.
Identical repeated IDs collapse; conflicting records with the same image and ID
stop inventory for resolution. Primary selection follows the selected text
sidecar, including an explicit clear, then the native record order (primary,
creation time, ID). Only the selected record remains primary. Older text
sidecars remain inactive history; all raw source files remain retained.
Flat-layout ``train/`` caption-record keys are
normalized only when unambiguous. Other unmatched keys stop inventory rather
than being attached by a guessed basename.

Unknown sidecars and repeated image bytes are listed as blockers. Resolve them
before apply. Repeated bytes cannot be silently collapsed across split/group
boundaries. The tool does not infer the semantics of arbitrary metadata or
follow paths inside a training YAML file.

Explicit preservation mapping
-----------------------------

Supply ``--mapping /review/mapping.json`` when there are custom files or metadata.
For example::

    {
      "overlay_root": "/data/reviewed-annotation-overlay",
      "files": {
        "groups.json": {"role": "groups"},
        "recipes.json": {"role": "recipe", "association": {"scope": "whole_image"}},
        "notes.json": {"role": "metadata"}
      },
      "image_metadata": {
        "train/yard.jpg": {"video_group": "flight-27"}
      },
      "external_artifacts": [
        {"path": "/models/yard-salad.pt", "role": "salad_weights",
         "association": {"head_id": "yard-salad", "backbone": "dinov3"}},
        {"path": "/models/yard-salad.json", "role": "salad_metadata",
         "association": {"head_id": "yard-salad"}}
      ]
    }

File roles are ``metadata``, ``groups``, ``recipe``, ``salad_weights``,
``salad_metadata`` and ``caption_records``. External overlay files use
``@overlay/`` keys in ``files``. Every unknown file needs an explicit role; even
an obsolete sidecar can be retained as metadata instead of silently dropped.
Per-image metadata keys must match inventory image keys exactly.

Explicit group fields ``capture_group_id``, ``sequence_id``, ``video_id``,
``flight_id``, ``video_group``, ``source_group`` and ``duplicate_group`` are
carried into analysis and kept together by split assignment and pruning export.
Matching values join images only within the same field; no group is inferred
from arbitrary metadata or camera identity.

Custom recipe/SALAD bytes and associations are registered for backup and retained
with the dataset settings. This preserves provenance; it does not automatically
activate a legacy recipe or assert hosted-runtime compatibility. External global
registry state is not discovered automatically and must be supplied explicitly.

Apply and resume
----------------

Apply rehashes the full inventory before creating or resuming the dataset::

    python tools/hosted_migrate.py apply \
      --plan /review/migration-plan.json --state-dir /srv/tator \
      --name "Imported dataset" --admin administrator

The administrator password is prompted, never passed as a command argument.
The default imports images into Incoming. Existing boxes and captions become
draft annotations; normal assignment and review rules then apply.

To accept an already-reviewed legacy dataset as an approved baseline, add both
``--mode approved-baseline`` and ``--accept-existing-labels``. Every image must
have a label file; an empty file becomes an explicit negative example. The
current administrator receives the new approval attribution. The importer does
not invent earlier review history.

Run records, source hashes, per-image progress and operation receipts live in the
hosted database. Repeating the same plan, administrator, name and mode resumes
that run. An interruption can leave a partially imported dataset; it does not
delete completed work. Changed source inputs or concurrently edited image
versions stop continuation rather than overwrite them. Keep the exact reviewed
plan and source tree until migration and recovery acceptance are complete.
After upgrading the importer, regenerate and review an inventory before starting
a new migration; an older inventory with different resolved annotations is
rejected instead of silently being reinterpreted.
The current inventory format is ``tator-hosted-migration-v2``. Unfinished v1
plans return ``migration_plan_requires_review``; regenerate and review their
resolved captions and coordinate decisions before starting a new run. Completed
v1 runs retain their exact finish receipt and can replay it without rereading or
changing their original source tree.

The local fixture tests exercise filesystem preservation, overlay labels,
caption alternatives, custom artifacts and restart recovery. Importing WALDO40
and verifying its actual counts, splits, recipes and continued annotation remain
separate deployment steps.
