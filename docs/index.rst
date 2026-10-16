Tator documentation
===================

Tator has two operating modes. The **hosted workspace** keeps datasets and user
accounts on one server. The **local workbench** retains the original single-user,
filesystem-based workflow. Start with the guide for the mode you are using.

Use the hosted workspace
------------------------

* `Administrator and annotator guide <hosted_workflow.rst>`_: create a dataset,
  add incoming images, assign work, submit images, return feedback and approve
  the confirmed dataset.
* `Analysis and dataset extension <hosted_analysis.rst>`_: run quality analysis,
  pruning, training and incoming-data comparisons against frozen dataset inputs.
* `Dataset pruning <dataset_pruning.rst>`_: understand similarity recipes,
  image families, object weighting, unique-object loss and reviewed exports.

Operate a hosted server
-----------------------

Read these in order for a new installation:

1. `Deployment and updates <hosted_deployment.rst>`_: runtime requirements,
   administrator bootstrap, HTTPS, service configuration and acceptance checks.
2. `Migration <hosted_migration.rst>`_: inventory and copy a legacy YOLO dataset,
   retain its annotations and metadata, and review the result before cutover.
3. `Operations and recovery <hosted_operations.rst>`_: dataset exports, archive
   policy, encrypted installation backups and restore rehearsal.
4. `Models and compute <hosted_runtime.rst>`_: installed model definitions,
   dataset detector settings, private user workers, queues and provider billing.

`Operator commands <../tools/README.md>`_ provides a short command index. Run
maintenance commands from the repository root in the intended environment.

Implementation references
-------------------------

* `Annotation assistance <hosted_assistance.rst>`_: trained classifiers,
  segmentation, detection recipes and Qwen proposal adapters.
* `Saved SAM3 recipes <hosted_sam3_chain.rst>`_: retained package contents,
  execution rules and current integration limits.
* `SALAD profiles <hosted_salad_profiles.rst>`_: retained trained heads,
  dataset attachments, preferred recipes and frozen analysis settings.
* `Browser UI <../ybat-master/README.md>`_: hosted workspace, shared canvas and
  the separate local-workbench entrypoint.

The deployment guide distinguishes source/local verification from the GPU and
real-dataset checks required on the target server. A successful health response
alone does not establish that assistance models or migration are ready.

Use the local workbench
-----------------------

These guides describe the original local inference service. Its linked/managed
storage and unauthenticated local API are separate from hosted dataset ownership
and approval rules.

* `Environment setup <environment_setup.md>`_: Python profiles and installation.
* `macOS inference setup <macos_inference_setup.md>`_ and
  `runtime defaults <macos_runtime_defaults.rst>`_: Apple Silicon model setup.
* `Local dataset library <dataset_library.rst>`_: linked and managed storage,
  shared local loading, cached analysis and dataset identity.
* `Detection recipe explainer <ensemble_detection_recipe_explainer.md>`_: how
  detector and segmentation stages combine; hosted adapters are described above.
* `Local Qwen caption policies <qwen_caption_run_policies.md>`_: local run and
  resume behavior; hosted caption authority is covered in the runtime guide.

Historical context
------------------

`Pre-hosted redesign baseline <redesign_baseline.rst>`_ records the local product
before the shared workspace was introduced. It is a historical inventory, not the
current deployment or acceptance checklist.

`Back to the project overview <../readme.md>`_
