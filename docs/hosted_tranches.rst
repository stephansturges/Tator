Named annotation tranches
=========================

A tranche is a named group of Incoming images prepared for one annotator. It
keeps instructions, progress and review history together. Image pixels and
annotations stay in the dataset's existing storage; creating a tranche does
not make another copy of them.

Confirmed images belong to **Dataset** and cannot be assigned. **Incoming**
contains all pending images, including reserved Drafts, active assignments and
submissions awaiting review. Filter it to **Unallocated only** to prepare new work.
Tranche and annotator labels show where each image belongs; **Tranches** and
**Review** provide the corresponding work and review queues.

Prepare and assign work
-----------------------

1. Upload new images into **Incoming**. Select the images, then create a tranche;
   alternatively open **Tranches** and choose **New tranche** to start empty.
2. Give it a useful name and instructions. Choose an annotator now or later.
   **Save as Draft** reserves the images without giving an annotator access.
3. Open the Draft to add images or revise its instructions. Assign it when the
   selection is ready. **Assign now** creates the Draft and then activates that
   exact selection; access begins only after activation succeeds.
4. Open an Active tranche to see unfinished images, returned work and submitted
   images. **Review** opens only its pending submissions. Annotators can submit
   one image at a time; the rest of the tranche stays with them.

One selection change accepts at most **500 images**. A tranche can contain up
to **10,000 unfinished images**; add further selections when preparing a larger
Draft. Activation and changing annotator apply to the exact unfinished group in
one transaction. The interface does not silently split a change into several
commits. Approved and other historical outcomes do not consume this limit.

Use the filename search to find Incoming images. Select the visible results,
shift-click a range, or choose **Select all matching** in the workspace
(**Select all matching images** in the add-images dialog). The exact matching
count is shown; more than 500 matches requires a narrower selection, without
silently truncating it. The tranche list can be filtered by annotator and shows
its most recent activity.

An upload directly into a tranche accepts at most 500 image files before any
bytes are sent. Duplicate copies within the batch resolve to one image. An
uploaded duplicate of a confirmed image does not become new Incoming work.
An image already reserved elsewhere cannot be attached to a second tranche.
Available unique images attach together; excluded files list their existing
location, including copies already in the destination tranche.
If upload succeeds but attachment is interrupted, the uploaded images remain
saved. Recover the attachment or select those images from Incoming; uploading
again is unnecessary.

Annotate and review
--------------------

Annotators open **My tranches** and choose **Continue annotation** on a card to
resume immediately in the existing annotation canvas. **Open tranche** shows
its instructions and progress first. All first-pane annotation tools and
shortcuts remain available, subject to installed runtime capabilities. The administrator still
selects the dataset's detector and class predictor. Tranche instructions add
human guidance; they do not change tool permissions or model prompts.

The breadcrumb identifies the dataset and tranche. The image list, previous /
next, **Submit & next**, review navigation and neighbor prefetch stay within
that queue. More images load as needed, including when a tranche contains more
than 500 images. Completing the queue returns to the tranche instead of opening
an unrelated assignment. Explicit dataset-wide administrator analysis and
caption jobs still use their confirmed dataset scope. **Refresh work list**
updates membership and instructions without resetting your current annotations.

After submission, the annotator can see its progress but cannot open that
image's pixels or annotations. The administrator may approve it or return it
with feedback. Returned work goes back to the original author in the same
tranche. Only approval adds it to the canonical dataset. A reviewed background
image with no objects is valid: use **No objects to label** before submission.

Administrators may annotate and approve a Draft's reserved image directly.
That image counts as approved without inventing an annotator submission.
Inside a tranche, filter to **Awaiting review**, select the reviewed images and
choose **Approve selected submissions**. Each image is checked against its
captured version and ownership generation. The result identifies any rejected
images separately. A lost reply retains the exact per-image commands for retry;
it does not create new approvals against changed work. The global **Review**
view also provides tranche and annotator filters before selection and approval.

Revise an assignment
---------------------

**Remove from tranche** sends the selected unfinished images back to unallocated
Incoming and preserves their saved annotations. **Move to another tranche**
reserves them in a Draft or assigns them to an Active destination's annotator.
Both changes revoke the previous editing session, so a delayed save or inference
result cannot overwrite the handoff. If a selected image changed after the
selection was captured, refresh and review the selection before retrying.

**Change annotator** transfers only unfinished work into a new ownership
assignment. It is blocked while submissions await review. Review those images
first, or move a selected unfinished subset to another tranche. Earlier authors,
submissions and approvals remain in history. The new annotator receives the
current saved annotation as a working baseline, but not another person's private
feedback or earlier annotation history.

Disabling an account or promoting it to administrator does not discard its
work. The tranche shows **Annotator unavailable**. Recall its unfinished work;
for a pending submission, use **Withdraw to Incoming** with a reason. Withdrawal
records a terminal review, preserves the original submission and makes the
image available for a new assignment.

**Close tranche** is allowed only when no reserved, editable or submitted images
remain. Finish or explicitly remove that work first. **Reopen as Draft** lets an
administrator add new work while retaining previous outcomes; it never revives
old editing access.

Read the progress counts
-------------------------

Each image counts once per tranche, using its latest membership there:

* **Not assigned**: reserved in a Draft.
* **To finish**: assigned work.
* **Changes requested**: returned work.
* **Awaiting review**: submitted work.
* **Approved**: admitted to the canonical dataset.
* **Removed**, **Transferred**, **Archived**: terminal outcomes shown separately.

**Ever included** is the distinct total across these buckets. **Currently
retained** excludes removed, transferred and archived images. An empty group is
not shown as fully complete. **Complete** appears when a previously populated
tranche has no unassigned, editable or awaiting-review images; this does not
change its Draft / Active / Closed lifecycle. Re-adding an image does not
inflate its total, and approval after a transfer does not change the source tranche's transferred
outcome. Annotators see only their own work and outcomes; administrators see the
whole tranche and its history.

Recover an interrupted change
------------------------------

If a reply is lost, the command may already have committed. **Check and
continue** retains its original operation ID, exact image selection and
preconditions. It recovers that result before allowing another tranche change.
A successful commit remains successful even when refreshing the page fails.
A definitive rejection retires the saved command so a fresh selection can be
made. Uploaded files remain saved even if attachment was rejected.
Do not replace an uncertain command with a larger selection.

If an image is moved or recalled while its canvas is open, the server rejects
further writes from the old editing session. Reloading its old link opens a
recovery view when you still have access; it does not silently switch the work
queue. **Recover my draft** first checks any original command, then requires
current editing permission and an unchanged saved version before restoring your
local edits. Otherwise the draft remains retained for reconciliation. **Back to
tranche** returns to the original work list; **Open current work queue** is an
explicit choice when there is no retained draft.

A lost approval, return, submission or archive reply is checked before loading
the image again. A completed handoff may have removed image access; only your
own matching receipt can open the data-free recovery view. It returns to the
original work list without exposing the image. A change of account or
installation still invalidates stale requests.

Archive, restore and schema upgrades
------------------------------------

Archiving a reserved or submitted image ends its current membership. Archiving
a submission records a withdrawal; restoring it returns it to **Incoming**,
without reopening its old assignment or review. Assigned and returned images
must first be explicitly recalled. A confirmed image restores as confirmed.
These rules also apply to dataset archives. Archiving a previously approved
image does not erase its completed work history.

Store schema **2** adds tranche membership and links every ownership episode to
a tranche. Startup and recovery restore use the same transactional schema-1
migration. Existing assignment IDs, annotation versions, receipts, submissions
and reviews are preserved. Each old assignment becomes a named historical
tranche. Only its still-current nonarchived images retain live ownership.
Archived nonconfirmed images are normalized to Incoming without being
unarchived; system-attributed migration outcomes preserve the reason that their
old review is no longer pending.

Upgrade a backup copy first and verify both foreign keys and ownership
relationships. Older schema-1 applications reject schema 2. Rollback therefore
uses the same schema-2 release on the recovery host and the newest recoverable
backup, including any work submitted after cutover. Never run the old binary
against upgraded state or reopen a stale production copy beside the new one.

See `Operations and recovery <hosted_operations.rst>`_ for encrypted backups and
`Deployment and updates <hosted_deployment.rst>`_ for service configuration.
