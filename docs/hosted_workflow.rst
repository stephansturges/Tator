Hosted dataset and annotation guide
===================================

Use this guide to create a hosted dataset, share annotation work, review it and
export the approved result. Images and saved annotation versions live on the
server. Users sign in to the same installation; they do not exchange working
folders. Existing source files are copied into immutable storage, never edited.

.. contents:: On this page
   :local:
   :depth: 2

Where work lives
----------------

An administrator's **Dataset** view is the confirmed, canonical dataset used by
analysis, training and dataset exports. **Incoming** shows images that are not
yet confirmed, including assigned, returned and submitted work. **Review**
focuses on submitted images awaiting an administrator's decision. Uploading,
assigning, saving or submitting does not add an image to the canonical dataset.

Annotators use **My work** to open their own assigned or returned images.
**Submitted work · all datasets** tracks images they have handed over, including
datasets where they have finished every assignment. That progress list does not
grant access to the submitted pixels or annotations.

The normal route through the workspace is::

    Administrator: Add images -> Incoming (not assigned)
                                     |
                             Assign selected images
                                     |
    Annotator:                  My work -> save drafts
                                     |
                                Submit & next
                                     |
    Administrator:                 Review
                              /               \
                  Return for changes       Approve & next
                         |                       |
                    My work again          Dataset (confirmed)
                                                 |
                                          Frozen snapshot
                                                 |
                                    Analysis / training / export

An administrator can also annotate unassigned Incoming images and approve them
directly. Archiving removes current membership while retaining saved history
and earlier snapshots.

.. list-table:: Who can work on an image?
   :header-rows: 1
   :widths: 22 30 30 18

   * - Image state
     - Administrator
     - Annotator
     - In canonical snapshots?
   * - Not assigned
     - Edit, assign or approve from Incoming
     - No access
     - No
   * - Assigned or returned
     - Inspect; explicitly recover ownership before editing
     - Assigned person edits it in My work
     - No
   * - Submitted
     - Edit, approve or return it in Review
     - Track submission status; no image editing or download
     - No
   * - Confirmed
     - Edit, analyze, split, export or archive
     - No annotation access
     - Yes, while active
   * - Archived
     - Inspect archive records and restore through Export & recovery
     - No active work access
     - Excluded from new snapshots; older snapshots stay intact

Assignments choose the responsible person; an editing reservation chooses that
person's active browser tab. A second tab cannot silently take over. Stale tabs
and late tool results cannot replace newer work. The same source image in two
datasets has separate assignments and annotation histories, even when its stored
image bytes are shared.

First day: administrator
------------------------

1. Choose **New dataset**, enter a name and **Classes, one per line**, then
   **Create dataset**. Classes can be added later through **Classes**.
2. Open **People**, enter a username and role, and choose **Create invitation**.
   Send the private invitation link yourself; the person uses it to set a password.
3. Select the dataset, open **Incoming**, then **Add images**. Images alone are
   enough to start. Existing YOLO labels can accompany image uploads; use the
   reviewed migration workflow for a dataset with caption history.
4. Select unassigned images, choose an annotator in **Assign to**, and use
   **Assign selected images**. Assign a manageable set; the annotator can submit
   completed images individually without finishing the whole set.
5. Open **Tools & models** to check installed assistance and the dataset's model
   selections. Manual annotation works when no models are configured.
6. As submissions arrive, open **Review**. Inspect an image, make any corrections,
   mark **Annotations checked** or **No objects to label**, then use
   **Approve & next**. To request more work, fill **Feedback for the annotator**
   and choose **Return for changes**.

**Approve selected images** in Review approves a selection without opening each
canvas. Use it only after reviewing those images. Outcomes are recorded per
image, so inspect reported conflicts or failures before assuming the whole
selection was approved.

In **People**, **Save account** applies a role or **Disable access** change;
**Password reset link** creates a new password-setting link. Account changes
sign out existing sessions while preserving assignments and saved work. Use the
ownership recovery controls below when that work needs a different editor.

First day: annotator
--------------------

1. Open the invitation, set your password, and **Sign in**.
2. Choose a dataset and open an image in **My work**. The canvas shows the dataset,
   filename and save status. Use the image list to move through your assignments.
3. Draw or correct annotations, edit captions, and use available assistance as
   needed. Changes save automatically; **Save now** requests an immediate save.
   Check the save status before handing work over or closing the tab.
4. Inspect the complete image. Select **Annotations checked** for finished labels,
   or **No objects to label** for a reviewed negative image with no objects.
   An empty annotation alone is not a reviewed negative example.
5. Choose **Submit & next** for this image. It leaves your editable work and goes
   to the administrator. Other assigned images stay editable; no whole-dataset
   submission is required.
6. Follow **Submitted work · all datasets** in the workspace. A returned image
   reappears in **My work** with feedback. Correct it, check it and submit again.

Use **Classes & requests** in the workspace, or **Request a class** in the
canvas, when the existing list is insufficient. Choose **Send request** with a
reason. Continue with the existing classes until an administrator resolves it.

Annotation tools in the first pane
----------------------------------

The image opens in the **Label Images** canvas. Available assistance depends on
installed model packages, the dataset configuration and your current editing
permission. **Tools & models** shows what is selected. A visible tool control
does not guarantee that its required weights or recipe have been installed.

.. list-table:: Canvas assistance
   :header-rows: 1
   :widths: 27 45 28

   * - Controls
     - Use
     - Required configuration
   * - Manual boxes, polygons and classes
     - Draw and correct the image's annotations
     - Dataset classes; no model required
   * - Auto Class
     - Suggest a class for a new or selected object; optional clarity guard
     - Dataset class predictor with labels mapped to dataset classes
   * - SAM Mode; Point Mode; Multi-Point Mode
     - Interactive box or point segmentation, positive/negative points and SAM
       box refinement; Preload SAM prepares the current image
     - Your permitted SAM model; classifier for combined Auto Class operations
   * - SAM3 Text Prompt; Run SAM3; Start cascade
     - Text-driven segmentation, optional windowing, or sequential text prompts
       through Run cascade
     - Permitted SAM3 model
   * - SAM3 similarity prompt (use selected box(es))
     - Use selected positive/negative boxes as visual examples
     - Permitted SAM3 model
   * - Run detector (current image); Detector mode
     - Region, Full image or Windowed (SAHI) detection; loaded-work batch controls
       operate on the images available in this canvas
     - Administrator-selected Dataset detector, YOLO or RF-DETR
   * - Detection Recipe; Run recipe on image; Run recipe on next N
     - Apply the dataset's saved multi-stage detection recipe
     - Administrator-selected Dataset detection recipe and its installed components
   * - Qwen 3 object detection; Run detection engine
     - Propose objects from the current image and supplied context
     - Permitted local Qwen model
   * - Qwen 3 captioning; Caption image; Caption image + Q&A
     - Generate image captions and questions/answers; Preview image prompts
       shows the request before generation
     - Permitted local Qwen model, or an enabled paid provider where offered

Running a tool can add or refine draft annotations. Inspect the result before
submitting or approving the image. A completed inference does not itself approve
an image or change canonical membership. Retained caption-batch results remain
proposals until explicitly applied. Dataset exports and dataset-wide analysis
are administrator workspace actions, rather than annotator downloads.

Your SAM selection and image cache belong to your account. Local Qwen requests
and class-predictor instances are also private to each user; selecting your SAM
model does not change another annotator's selection. The administrator selects
the dataset class-predictor package and its class mapping. Direct detector and
saved Detection Recipe requests share the dataset's inference queue and run one
at a time on that instance. Multiple users can therefore annotate independently
without sharing their interactive SAM state.

Administrators install server-configured packages and select **Dataset detector**,
**Dataset class predictor** and **Dataset detection recipe** in **Tools & models**.
Each model label must map to a dataset class. **Your segmentation model** lets
each user choose an allowed installed model with **Use this SAM model**.

**Saved SAM3 recipes** manages which installed recipe packages are attached to
the dataset. These packages are distinct from the visible **Text prompt cascade**.
Saved-package execution and account/dataset-scoped preset APIs are documented in
`Hosted inference runtime <hosted_runtime.rst>`_ and
`Hosted saved SAM3 recipes <hosted_sam3_chain.rst>`_; their legacy package-cascade
controls are absent from the default annotation layout.

During administrator heavy compute, the tool status can show that assistance
is paused or reloading. Manual annotation and saving remain available. A pause
does not approve drafts, hand work to someone else or discard their saved state.

For paid captioning, an administrator configures **Paid services**, including
allowed datasets, models and spending limits. Credentials stay on the server.
An interrupted provider reply can leave a reservation unresolved; check the
existing job or bill before starting another request. **Record confirmed charge**
requires administrator confirmation of provider usage, and a linked remote Batch
must have a verified terminal status. Local verification uses provider transport
doubles; it does not establish live price, throughput or model quality.

Corrections, returns and ownership recovery
-------------------------------------------

In **Review**, an administrator can correct a submitted image before approving
it, or return it with feedback. The submission, corrections and feedback remain
in history. **Return for changes** restores the original annotator's editing
access. After approval, administrator edits saved on the confirmed image create
new canonical annotation versions; earlier snapshots keep their earlier versions.

An administrator cannot edit an annotator's active assignment without an
explicit handoff. **Recover from annotator** on the image removes that assignment
and preserves the saved draft. For several images, open **Recall assigned work**
in the dataset workspace, choose the annotator, select individual images or
**Select all in this list**, enter a reason, then confirm the recall. Images
return to unassigned Incoming for administrator editing or reassignment.
Submitted and confirmed images are excluded from recall.

Recall invalidates old editing tabs and automated results. If any selected image
changed after the list loaded, the entire recall is rejected; refresh and review
the selection again. **Check and retry saved recall** recovers an interrupted
reply without recalling work assigned after the original command.

Use **Take over my other editing tab** only to replace your own tab's reservation.
This does not take work from another person. If the editing reservation expires
while you are signed in, use **Reconnect editing**. After sign-out, sign in again;
when a retained draft is offered, recover it first as described below.

Importing labels and managing classes
-------------------------------------

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

The current working-image policy preserves encoded dimensions. Where conversion
is required, transparency is displayed on white and unsigned 16-bit values use
their high byte without contrast stretching. Original downloads and historical
inputs remain separate from these working views. See
`Encoded media and immutable working views <hosted_operations.rst#encoded-media-and-immutable-working-views>`_
for format, conversion and upgrade rules.

Conflicting imported labels are retained separately. Open the image and choose
**Imported labels** to compare the current and imported boxes on the same image.
Choose **Keep current labels** or **Use imported objects**. This requires current
administrator editing authority and an unchanged saved version. Applying imported
objects preserves captions and other existing metadata, and both annotation
versions remain in history. Replacing objects with an empty list requires an
explicit negative confirmation. The reviewed migration workflow supports larger
existing labelled dataset trees, including caption records, splits and metadata.
See `Migrating a legacy YOLO dataset <hosted_migration.rst>`_ before importing an
existing dataset tree.


Reviewing captions and questions
--------------------------------

The caption controls offer **Save as new alternate**, **Update selected**,
**Set primary** and **Delete alternate**. A primary caption and its alternatives
belong to the current image; removing a record from the current annotations does
not remove its earlier saved versions. Imported text aliases, inactive records
and extra metadata survive unrelated canvas edits.

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

Choosing training and validation images
---------------------------------------

Browser uploads start with no split assignment. After approval, select confirmed
images and choose **Set training split**. Choose **Training**, **Validation**,
**Test** or **Not assigned to a split**, then **Preview split change**. The preview
includes other confirmed images linked by explicit capture, sequence, video,
flight, source or duplicate groups, even when they were not selected. Review the
added images and before/after counts before **Apply this split change**. No random
redistribution takes place, and Incoming images are excluded.

The complete selection commits together. If a captured version, ownership
generation or dataset revision changed, or an image has an active editor, the
change is rejected. Existing annotation versions and older snapshots remain
intact. A lost reply can be recovered with **Check pending split change**, even
after navigating away or losing the original selection. Training
requires explicit train and validation populations and rejects unsplit images
or capture groups spread across those populations. Migration retains source
split assignments; this editor can deliberately revise them later.

Snapshots, analysis and dataset downloads
-----------------------------------------

Open **Export & recovery**, then **Export confirmed dataset** for a canonical
ZIP. **Quality & compute** provides configured administrator analyses and
training tasks. **Caption jobs** provides retained dataset caption work and
caption-training export. See `Canonical inputs for analysis and training
<hosted_analysis.rst>`_ for those workflows and their qualification limits.

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

When a connection or save fails
-------------------------------

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

Archive and storage cleanup
---------------------------

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

Technical contracts and server recovery
---------------------------------------

The following details are for integrations and operators. Everyday editing uses
the controls above; clients must preserve these rules when retrying commands or
restoring an installation.

Command identity and disclosure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Language validation and mutation preconditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Full server backup and restore
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A dataset ZIP is not a server backup. Full recovery uses an age-encrypted archive
of the SQLite authority and every registered immutable artifact, including
server-saved drafts and model files. Unsent edits and recovery journals in a
user's browser are not included. Supply only the public age recipient to the
server; keep the private identity on a separate trusted recovery machine.
Download backups and verify a restore before relying on them.

The `storage and recovery guide <hosted_operations.rst>`_ provides the setup,
backup and restore commands, external configuration checklist, and service-user
permissions. The recovery CLI uses the hosted reference enumerator and restore
validator by default.

Restore into an absent or empty destination while that destination server is
stopped. Validation checks hashes, required media and registered model files
before publishing the restored root. Sessions, invitation/reset links and editor
leases are revoked; the installation identity changes. Users log in again and
resume preserved assignments and drafts. Running inference is not resumed as a
successful annotation change. Previous backup-download files are not recursively
included in new backups.

Local component and HTTP tests do not establish deployment, GPU capacity,
complete integration of legacy consumers or a successful remote recovery drill;
those are separate acceptance steps.
