Hosted annotation workflow
==========================

Hosted Tator stores images and annotation versions on the server. An image
becomes part of the approved dataset only after an administrator approves it.
Uploading, assigning, saving and submitting an image do not admit it to that
dataset. Existing source files are copied into immutable storage, never edited.

From Incoming to approval
-------------------------

1. An administrator imports images into Incoming and assigns selected images
   to an annotator. Only unassigned Incoming images can be assigned.
2. The annotator saves drafts while working. Before submitting, they mark the
   annotation complete, or explicitly confirm that the image contains no objects.
   An empty annotation alone is not a reviewed negative example.
3. Submission transfers editing to the administrator. The annotator can see its
   progress, but cannot continue changing or downloading the submitted image.
4. The administrator approves, corrects and approves, or returns the image with
   feedback. Returning it restores the annotator's access. The original
   submission, subsequent corrections and feedback remain in the history.

**Submitted work** shows progress across all the annotator's datasets, including
ones where they have finished their assignments. Choosing another dataset for
editing does not hide that progress or grant access to submitted image pixels.

Administrators can also annotate and approve unassigned Incoming images directly.
They cannot edit an annotator's active work without an explicit ownership
override. An override preserves saved drafts and invalidates the former editor.
After approval, only administrators can edit the image; a completed saved edit
creates a new canonical annotation version.

To recover a whole annotator list, open **Recall assigned work** in the dataset
workspace. Choose an annotator and select all their listed images, or select
individual images. Enter a reason, then recall the selection. Saved drafts and
review feedback remain intact; editing tabs and pending automated results lose
their old authority. The images return to unassigned Incoming, where an
administrator can edit or assign them again. Submitted and confirmed images are
excluded. If any selected image changes after the list was loaded, the entire
recall is rejected: refresh and check the selection again. An interrupted reply
can be retried with the same saved command without recalling newly assigned work.

One editor at a time
-------------------

Assignments identify the person responsible for an image. An editor lease
identifies their active browser tab. Opening another tab does not silently steal
the lease; an explicit takeover replaces it. Stale tabs and late automated
results cannot overwrite a newer version, ownership change or restored server.
Expired sessions and disabled accounts are checked again when a change commits.

Each command has an operation ID. Retrying the same command recovers its committed
outcome, including after a lost response. Reusing that ID for different content
is rejected. A version conflict requires reloading and reconciling changes;
retrying with a new ID does not bypass it.

Receipts disclose saved results only while the signed-in account has current
permission to read them. A committed save or submission can become inaccessible
after handoff, recall, reassignment or archive. In that case the server confirms
that the operation committed without returning annotation, dataset, model or
editor details. The client stops retrying that operation and keeps newer unsent
edits available for recovery. It does not apply the inaccessible saved version.
Administrator command results require current administrator authority. Unknown
legacy operation types receive the same restricted acknowledgment. The stored
command, fingerprint and original receipt remain unchanged.
An administrator-only endpoint can reject a demoted user's POST with an ordinary
403 before inspecting the command. Use the authenticated receipt endpoint to
check whether that earlier command committed; the restricted acknowledgment
does not grant access to its administrative result.

A known annotation-validation error can be corrected and saved under a new
operation ID only when the server certifies that this exact save rolled back.
The certification includes ``commit_outcome: rejected`` and its ``operation_id``.
Transport failures, incomplete responses and server errors leave the outcome
unknown: the browser retains the exact original command for retry. Authentication,
version and identity conflicts require reconnecting or reconciling first.
Rejected saves preserve the latest draft and the last acknowledged version;
unchanged invalid content is not repeatedly submitted by autosave.

If another tab logs out or editing authority expires, the canvas first retains
its latest unsent draft in this tab's browser storage, then disables editing.
The retained draft belongs to the original account and image. Sign in again in
the same tab and reopen the image to choose **Recover my draft**. Recovery checks
the original command's receipt and the current server version before restoring;
it cannot automatically overwrite a newer saved version. Keep this browser tab
open until recovery is complete. If browser storage cannot retain the draft, an
error says so and leaving the page still prompts about unsaved work.

The version 2 recovery journal records pending, unsent and rejected drafts
separately. Older journals remain conservatively unresolved until the original
receipt and current version can be checked. A storage failure retains the draft
in memory and keeps the unsaved-work warning active.
If you undo all changes back to the last acknowledged version, saving or leaving
the page clears an obsolete unsent or rejected draft. An unresolved server
command is still checked first, even when the canvas looks unchanged. Failure to
clear browser storage shows an error and keeps the unsaved-work warning active.
If no receipt exists after reload, recovery first retries the exact original
command. Another unknown response leaves it intact. If a replaced or expired
editing lease rejects that retry, explicit draft restoration requires an
unchanged server version and a newer editing generation. Restoring an invalid
draft still resumes retention of subsequent edits while the user corrects it.

Known language fields are validated on import, save, submit and approval. QA
``validated_against`` and ``source_fields`` values must be lists or null at the
pair, metadata and validation locations, including locations shadowed by another
value. Legacy null fields, plain caption strings and unknown extension data are
preserved. Invalid input creates no annotation version. Instruction export also
checks historical snapshots before starting native processing and reports the
image ID and precise invalid field; it does not repair or omit that content.

Assignment and dataset-archive commands require a nested exact precondition for
each image: a nonempty ``expected_version`` (or legacy ``version_id``), a strict
nonnegative integer ``generation``, and an optional nonempty ``editor_id``.
Malformed entries return 422. Supplied legacy aliases and extension keys remain
part of the fingerprint; omitted defaults are not inserted. Store validation
runs only for new commands, after checking for an exact committed receipt.

Classes and repeated imports
---------------------------

Class IDs remain stable when their names change. Annotators request additions;
administrators manage the class list. Each frozen dataset snapshot records the
class names and export indices used with its annotations.
Names cannot contain internal line separators, including Unicode separators.
This keeps one class name equal to one line in exported labelmaps. Older invalid
names remain readable and can be renamed by an administrator; new exports of an
invalid frozen map report the class that needs correction. Existing completed
export receipts still return their original files. After renaming, capture a new
snapshot for export; the old snapshot retains its original class map.

Importing the same bytes into the same dataset finds the existing image. It does
not replace saved annotations. Select images and their same-stem YOLO ``.txt``
files together in the Incoming uploader. Include one ``labelmap.txt``,
``classes.txt`` or ``labels.txt`` to map numeric class indices by name; otherwise,
the uploader asks you to confirm the dataset's current class order. Class names
must already exist. Ambiguous filenames, changed class maps and out-of-bounds
geometry are rejected rather than guessed or silently clipped. Empty sidecar
files are retained as empty imported annotations, not approved negatives.

For labeled images with EXIF rotation or reflection, use **Imported label
coordinates** to declare **Original stored pixels** or **Rotated photo viewer
pixels**. No coordinate frame is assumed for an oriented labeled import. Viewer
coordinates are converted to the original encoded grid with provenance retained;
the original image bytes remain unchanged. The upload API uses
``label_coordinate_space=encoded`` or ``exif_display`` for this declaration.
Upload and migration fully decode admitted images within the shared limits of
256 MiB input and 64 million pixels. Unsupported, corrupt or oversized media is
rejected before image membership is created.

Conflicting imported labels are retained separately. Open the image and choose
**Imported labels** to compare the current and imported boxes on the same image.
Choose **Keep current labels** or **Use imported objects**. This requires current
administrator editing authority and an unchanged saved version. Applying imported
objects preserves captions and other existing metadata, and both annotation
versions remain in history. Replacing objects with an empty list requires an
explicit negative confirmation. The reviewed migration workflow supports larger
existing labelled dataset trees, including split and metadata preservation.

The same image in another dataset has its own assignment and
annotation history, even when both memberships share immutable media bytes.

Choosing training and validation images
---------------------------------------

Browser uploads start with no split assignment. After approval, select confirmed
images and choose **Set training split**. Choose train, validation, test or unsplit,
then preview the change. The preview includes other confirmed images linked by
explicit capture, sequence, video, flight, source or duplicate groups, even when
they were not selected. Review the
added images and before/after counts before confirming. No random redistribution
takes place, and Incoming images are excluded.

The complete selection commits together. If a captured version, ownership
generation or dataset revision changed, or an image has an active editor, the
change is rejected. Existing annotation versions and older snapshots remain
intact. A lost reply can be recovered with **Check pending split change**, even
after navigating away or losing the original selection. Training
requires explicit train and validation populations and rejects unsplit images
or capture groups spread across those populations. Migration retains source
split assignments; this editor can deliberately revise them later.

Snapshots and dataset downloads
------------------------------

Human caption and generated-question review decisions belong to administrators.
Open **Review captions and QA** in the canvas to inspect and decide on individual
captions and answers, with an optional correction note.
Annotators can keep that evidence while editing other annotations. Editing text
with a hosted review stamp makes that text unreviewed again; the old decision
and model evidence remain in history. They cannot add, alter or copy a review
stamp. Imported historical decisions have no content binding, so changing their
reviewed wording requires an administrator. Removing a complete caption record
is allowed and preserves its earlier immutable versions.

Machine-validated generated answers must come from a completed server caption
job for the same user, image and annotation version, or from an explicitly
applied retained proposal. Ordinary saves cannot invent that status or change
its verifier evidence. Later wording edits keep the original evidence, but the
old validation no longer qualifies the edited text for training export.

A snapshot fixes the approved image list, annotation versions, class map and
dataset settings at one revision. Later saves, approvals and archives cannot
change that snapshot. Incoming and submitted work are excluded.

Canonical ZIP exports contain those images, YOLO bounding boxes, class names,
captions, polygon files where present, and full annotation/provenance JSON.
Existing train/validation splits are preserved; the exporter does not invent a
new split. Downloads are administrator-only. Retrying export returns the same
durable artifact rather than rebuilding it from the current dataset.

The snapshot materializer produces the same fixed input in an isolated directory
for analysis or training adapters. Every adapter must explicitly use this
snapshot; a legacy consumer that scans a live source folder is not automatically
covered by the hosted access rules.

Archive and recovery
--------------------

Archiving removes an image or dataset from current work and current canonical
snapshots. It keeps the bytes, annotations and historical snapshots. Restoring an
archive restores the captured membership and version. Dataset archive requires
the exact current member list and cannot bypass active annotator ownership.

Physical purge is separate and manual. The minimum retention period is 30 days.
An archived image still referenced by history, another dataset or a snapshot
cannot be physically purged. The current service preserves those references and
reports why purge is blocked. Truly unreferenced artifacts have a separate
preview and retention check; interrupted deletion can be reconciled by retrying
the same command. Reported removed bytes are logical file sizes, not necessarily
disk space reclaimed while backup pins exist.
Open **Export & recovery → Unused storage** to inspect unreferenced files and
preview eligible cleanup. Deletion requires confirmation, and every reference
is checked again before removal. A saved cleanup command can be retried after
an interrupted reply. Archiving alone does not reclaim disk space.

A dataset ZIP is not a server backup. Full recovery uses an age-encrypted archive
of the SQLite authority and every registered immutable artifact, including
retained drafts and model files. Supply only the public age recipient to the
server; keep the private identity separately on the Mac. Download backups
manually and verify a restore before relying on them.

The recovery CLI uses the hosted reference enumerator and restore validator by
default::

    python tools/hosted_recovery.py backup \
      --database /srv/tator/hosted.sqlite3 --artifacts /srv/tator/artifacts \
      --output /srv/recovery/tator.age --recipient AGE_PUBLIC_RECIPIENT

    python tools/hosted_recovery.py restore \
      --archive /recovery/tator.age --destination /srv/tator-restored \
      --identity /private/recovery/identity.txt

Restore into an absent or empty destination while that destination server is
stopped. Validation checks hashes, required media and registered model files
before publishing the restored root. Sessions, invitation/reset links and editor
leases are revoked; the installation identity changes. Users log in again and
resume preserved assignments and drafts. Running inference is not resumed as a
successful annotation change. Previous backup-download files are not recursively
included in new backups.

See ``hosted_operations.rst`` for storage and encryption details. Local component
and HTTP tests do not establish deployment, GPU capacity, complete integration
of legacy consumers or a successful remote recovery drill; those are separate
acceptance steps.
