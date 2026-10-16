Datasets and saved review sessions
==================================

This guide covers the local workbench's storage library, served by the local
backend. Hosted accounts use the dataset workspace, assignment/review flow and
administrator archive controls described in
`Hosted workflow <hosted_workflow.rst>`_ and
`Hosted operations <hosted_operations.rst>`_. The two applications have separate
storage and recovery interfaces. See the `documentation index <index.rst>`_ to
choose the appropriate guide.

Open **Data Quality Explorer → Datasets & sessions** to find datasets and
saved reviews on the connected backend. The library also shows browser
storage, reusable analysis files and deleted datasets.

Open or resume work
-------------------

Use **Datasets** to open a dataset, or expand its **Saved reviews** to resume
a particular analysis. **Review sessions** lists reviews directly. Search
matches dataset and review names; **Show more** loads the next page.

Opening a review prepares its saved graph and the corresponding annotation
source before replacing the current workspace. Current labels, captions and
pending review work are saved first. If the current images came from local
files, Tator asks you to save that workspace to the backend library before
continuing. **Keep current workspace** cancels the switch.

If preparation fails, the current workspace remains open. **Cancel opening**
can stop preparation; it is disabled during the final installation step.
An unavailable annotation source may permit **Open plot only**. That mode
lets you inspect the saved graph without attaching a dataset for editing.
When another editor holds the source lock, opening it read-only preserves
their ownership.

A saved review identifies its source and annotation lineage. Opening an old
graph does not replace newer dataset annotations with old labels. Some old
reviews need source recovery before editing can resume; the library displays
the reason instead of silently choosing another dataset.

Browser reload recovery is scoped to the backend and browser tab. A duplicated
tab gets its own recovery writer and annotation lock request. Reconnecting
checks the backend's persistent identity; reusing an address for another
server does not authorize writes to it. Dataset pruning uses the same
identity and checks it again when opened and before saving its source.
If the server identity changes, editing is blocked. Verify the connection,
then reopen the dataset from the library to establish its source and lock.

Names, retention and storage
----------------------------

**Rename** changes the library display name. **Keep** protects an entry from
cleanup; **Unkeep** removes that extra protection. These are library retention
controls, separate from pruning's decision to retain an individual image.

The summary distinguishes backend storage from this browser's storage.
Browser quota is an estimate for the site. Tator's serialized preferences
and recovery records are part of that total. Selected local files and
downloaded ZIPs are outside browser storage management.

**Forget saved plot views** clears display preferences only. It preserves
pending recovery, reviews and datasets. **Measure storage** refreshes the
backend inventory. Large inventories can publish entries before all byte
counts finish; incomplete or unknown measurements remain visible.

Shared files are counted with their consumers. Deleting one entry may free
less space than its displayed size because another entry still uses those
files. Files in trash or quarantine still consume disk space; the filesystem's
free-space reading is shown separately.

Preview cleanup before confirming
---------------------------------

Choose **Manage storage**, or select entries in **Cleanup**. Only actions
supported by every selected entry are available. **Preview selected cleanup**
shows its effects, estimated removable bytes, retained dependencies and any
blockers. Confirming applies that specific preview. Changing the selection
or action requires a new preview.

The actions have different consequences:

* **Clear reusable cache** removes derived files that can be recomputed.
  Automatic cleanup is restricted to unused reusable cache; it does not expire
  saved datasets and reviews.
* **Clear analysis files** removes rebuildable analysis output while retaining
  compact, source-bound human work. Recomputing may be necessary to display
  the graph again.
* **Permanently delete review** explicitly removes its saved review work.
* **Move dataset to trash** retains files until trash is emptied. **Restore
  dataset** brings a trashed dataset back into the library.
* **Unlink dataset** removes Tator's link and associated app-managed state;
  it does not delete the external source files.
* **Permanently delete selected trash**, **Remove abandoned upload**, and
  **Clear prepared export** act on the named stored artifacts.

Active work and shared dependencies can prevent or delay cleanup. The
operation status reports progress and failures. An interrupted request keeps
its operation identity so a retry can recover the result without executing
the same deletion twice. **Retry remaining work** resumes a failed or partly
failed operation where supported.

Implementation and compatibility
--------------------------------

The interface uses ``/storage/library`` API version 1. A persistent backend
UUID binds requests, recovery metadata, workspace reservations and cleanup
operations. Source preparation and installation are coordinated by
``dqe_workspace_switch.js``; ``dqe_library.js`` owns the library interface;
``dqe_session_store.js`` owns browser recovery records.

Older backends without the library API retain the legacy latest-session and
cache controls. They cannot supply the verified identity required for the
new pruning workflow. Update the backend and reload the page to use both
features together.

See `Dataset pruning <dataset_pruning.rst>`_ for non-destructive dataset
variants and image-level removal review. Library cleanup and pruning export
are separate operations.
