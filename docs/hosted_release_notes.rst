Release notes
=============

2026-11-09: fifty-scenario walk
-------------------------------

.. warning::

   **This release is untested on a server.** Fifty user scenarios were walked
   in a browser against disposable instances with test data on a development
   Mac. SAM1, the WALDO class predictor and the WALDO40 YOLO model ran for
   real there; native SAM3, Qwen text generation and paid providers were
   stand-ins. Nothing was run on the CUDA server or with the production
   datasets. Keep the previous release available for rollback and check the
   main workflows yourself after updating.

It follows ``5f66d2b``.

Signing in and accounts
~~~~~~~~~~~~~~~~~~~~~~~

* Signing in from a link to the workbench, pruning or the canvas returns to
  that page. Browsers get a "Page not found" page instead of raw data.
* A new password set by an administrator ends the pause after failed
  attempts. A disabled account is told it can no longer sign in.
* An annotator with nothing assigned is told so, and one who opens an
  administrator link is told why their own work is shown.

Adding images and the admin workspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Stop import** and **Resume image import**, also after a reload. A bad
  file no longer blocks the selection: the good files are added and the
  others are listed with their reason. A labelmap naming classes the dataset
  lacks is refused before anything uploads.
* An import report under Incoming lists clipped, repeated and dropped label
  rows, images already in the dataset, and new images that share a file name
  with a different image.
* **History** on every image card and in the canvas panel.
* The Review and Incoming grids and the counts refresh when the window
  regains focus, unless a selection, dialog or typing is in progress.
* Returning an image clears "Annotations checked", so the annotator confirms
  again after fixing it.
* Readable muted text, field borders and a visible focus outline in every
  theme. Pip-Boy can be switched with Shift+Enter on the theme button. On a
  phone the tabs and actions wrap.

Tranches and review
~~~~~~~~~~~~~~~~~~~

* Cards show review notes and which images are open for editing. Changing
  the annotator asks first when images are open.
* An annotator whose tranche was handed on sees "No longer assigned to you",
  is told about work in other datasets, and a finished tranche offers the
  next one.
* Recall offers to make a tranche from the recalled images. **Select all
  waiting for review** selects up to 500 images.
* Tranche history reads in words. When another administrator changed the
  tranche first, the page reloads it and says so.

Annotation canvas
~~~~~~~~~~~~~~~~~

* **Undo**, **Redo**, **Delete box** and **Help & keys** (also ``?``).
  Ctrl/⌘+S saves, Ctrl/⌘+Enter submits or approves and goes to the next
  image, Ctrl/⌘+Z and Ctrl+Y undo and redo, Ctrl/⌘+0 fits the image.
* Tab now moves between controls; the left arrow still goes to the previous
  image. Wheel zoom follows how far the wheel turns.
* Touch: one finger draws and edits, two fingers zoom and pan the image.
* When the session ends, **Sign in again** keeps the draft. Leaving always
  works after an image was handed to someone else or recalled. A view-only
  canvas notices when the other person leaves.
* An image that needs a decision says why and offers **Compare imported
  labels**. SAM messages name the file.

Workbench and server jobs
~~~~~~~~~~~~~~~~~~~~~~~~~

* The Data Quality Explorer server job no longer fails when no class is
  named. Job failures are explained in words, and results show the run split
  and the training device.
* YOLO and RF-DETR server jobs show a licence box that must be ticked.
* A completed YOLO or RF-DETR training result offers **Install as detector**;
  the dataset switches to it only when an administrator sets it in **Tools &
  models**.
* Workbench training keeps the splits set on the hosted dataset for YOLO,
  RF-DETR and SAM3 and splits only the rest; test images stay out.
* **Hosted data** can open any copy. When the workbench service is stopped,
  the bar says so and offers **Reload**, and copies are refused at once.

Models and assistance
~~~~~~~~~~~~~~~~~~~~~

* The class predictor and Qwen tools no longer need a SAM model installed.
  A damaged installed model is flagged in **Tools & models** instead of
  stopping the service.
* Refusals while the server is busy read as sentences. Caption jobs stop
  after their attempts and say why; remote caption batches show the reported
  cost.

Updating
~~~~~~~~

* Restart the hosted and workbench services together.
* A YOLO or RF-DETR server job runs only when its task allows
  ``accept_tos`` in ``allowed_parameters``.
* Workbench copies made before this release do not carry the hosted splits;
  make a fresh copy before training in the workbench.

Automated checks run
~~~~~~~~~~~~~~~~~~~~

* Hosted test suite: 429 passed, 1 skipped.
* Hosted browser tests: 50 passed.
* The scenario walks for each area, against disposable instances with test
  data, and on the combined code the workbench flows (12 of 12), arriving in
  the workbench from the admin workspace (7 of 7) and the theme check
  (12 of 12).

The legacy suites ``test_labeling_panel_layout_contract`` and
``test_dataset_linked_annotation_flows`` still fail as before.

Known limits
~~~~~~~~~~~~

* Classes cannot be retired.
* Boxes from a detector batch are saved as drafts without a mark saying they
  came from the detector; they still need the annotator to check and submit.
* Qwen runs in the workbench ignore the hosted splits. Training without a GPU
  falls back to the processor and only reports it.
* Views refresh when the window regains focus; there are no live updates.
* An image left open in a closed browser stays reserved until its
  reservation lapses.
* Exports run in one request; the time for a dataset of 18,000 images is not
  measured.

2026-11-08: workspace, canvas and workbench review
--------------------------------------------------

.. warning::

   **This release is untested.** It has not been deployed to a server, used
   by hand with real datasets, or run with real GPU training. Only the
   automated checks below have passed, on a development machine with test
   data. Keep the previous release available for rollback (see
   `Updates and rollback <hosted_deployment.rst>`_) and check the main
   workflows yourself after updating.

This release follows a review of the whole interface: the admin workspace, the
annotation canvas and the admin workbench. It follows ``b9e8be6``.

Admin workspace
~~~~~~~~~~~~~~~

* **Incoming** can be filtered to images that need a decision, images with
  labels and images without labels. **Approve selected** and **Confirm as
  background** confirm many images at once, after one confirmation that states
  the count. Each image still gets its own approval record.
* Each import keeps a report with its clipped, dropped and duplicate label
  rows, shown on the dataset.
* Each tool has one entry, the dataset name and counts stay visible, and image
  states use the same names everywhere: Incoming, Not assigned, Assigned to,
  Waiting for review, Returned for changes and Confirmed.
* Back and reload keep the open view. Returning from the canvas brings back
  the same image, filters and search.
* The Pip-Boy theme is back: double-click the theme button to switch it on or
  off, and click it to change between green and amber. The admin workspace, the
  canvas and the workbench share the choice.

Annotation canvas
~~~~~~~~~~~~~~~~~

* Submitting an image without objects no longer gets stuck, and a refused
  handoff reopens the image for editing.
* Boxes are clipped to the image while drawing, so what is shown is what is
  saved. The class of an existing box can be changed.
* The image list shows file names, and its search only filters the list.
* From 1360 pixels wide, the panel sits beside the image instead of over it.
  It can be minimised.
* Tools that have no installed model are hidden, with a one-line reason.
  Classifier and Qwen tools no longer need a SAM model to be selected.
* Back leaves the canvas. Space, Tab and the arrow keys no longer act on the
  image while a button or field has focus.

Workbench
~~~~~~~~~

* The Label Images canvas fits the screen below the bar.
* The bar follows the copy that is actually open and handles deleted copies,
  stale locks and the pending-edit count.
* Copy files are named after the original file followed by ``__`` and the
  hosted image ID.

Training
~~~~~~~~

* Datasets without split decisions can be trained, both as server jobs and in
  the workbench. Each run holds out about a tenth of the images for validation,
  keeping capture groups together, and records the split it used. The dataset
  itself is not changed. See `Analysis and dataset extension
  <hosted_analysis.rst>`_.

Import and export
~~~~~~~~~~~~~~~~~

* The importer format is now ``tator-hosted-migration-v5``. Identical boxes in
  one label file are removed. When two copies of an image have different
  labels, the removed copy's labels become an import conflict to choose from on
  the canvas. See `Migration <hosted_migration.rst>`_.
* Pruning results made on an earlier dataset version stay readable, and an
  already prepared pruned dataset can still be downloaded.
* Exports are named after the dataset, revision and date, and include
  ``filenames.csv``.

Updating
~~~~~~~~

* Restart the hosted and workbench services together.
* An inventory plan made with an earlier importer version that has not been
  applied is refused; make a new inventory. Finished imports are unaffected.
* Imports made before this release show a report derived from their stored
  run. Images they left for a decision do not show the reason, and copies they
  removed as duplicates are not offered as import conflicts.
* Workbench copies made before this release keep their earlier file names.

Automated checks run
~~~~~~~~~~~~~~~~~~~~

* Hosted test suite: 426 passed, 1 skipped.
* Hosted browser tests: 50 passed.
* Scripted browser walks against local test servers for the admin workspace,
  the canvas, the workbench and the theme.

The legacy test suites ``test_labeling_panel_layout_contract`` and
``test_dataset_linked_annotation_flows`` still fail as they did before this
release.

Known limits
~~~~~~~~~~~~

* There is no reviewer role; reviewing needs an administrator.
* Below 1360 pixels wide the canvas panel still covers part of the image.
* Changing a box's class from the keyboard needs class keys, which have no
  default bindings.
* Classifier and Qwen workers still need some SAM model installed to start.
* Opening a workbench copy can briefly report that the library is not ready;
  it retries by itself.

Related guides are listed in the `documentation index <index.rst>`_.
