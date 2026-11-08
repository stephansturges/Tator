Release notes
=============

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
