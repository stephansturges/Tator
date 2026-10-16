Hosted SALAD profiles
=====================

For the user-facing **Feature recipes** controls, see
`Hosted workflow <hosted_workflow.rst>`_. This reference describes how trained
heads and their settings are retained for
`hosted analysis <hosted_analysis.rst>`_. All guides are listed in the
`documentation index <index.rst>`_.

Administrators manage trained SALAD heads from dataset settings. The retained
artifact picker identifies the original name, trained backbone/model and whole
image or bounding-box role. Its sources are completed native analyses and
migration artifacts explicitly classified as SALAD weights. Registration never
accepts a filesystem path or a user-entered blob hash.

Registration runs the existing ``load_local_salad_head_file`` loader against the
exact retained bytes in a separate CPU process. This checks the original local
trainer policy, file format, normalization configuration and parameter shapes.
The control process does not load Torch or allocate a GPU head. This structural
qualification does not establish descriptor quality or GPU compatibility.
Inspection has a 512 MiB input limit, bounded architecture dimensions and a
90-second timeout; unsupported heads remain visible with an error.

Each immutable head alias names one content digest. Dataset attachments, display
labels, archive choices and preferred recipes are stored in versioned dataset
settings. Different datasets can deploy the same retained head independently.
Registration attaches the chosen head to that dataset. Renaming changes its
display label; it does not rename an artifact or rewrite training metadata.
Archiving and detaching cannot remove a head still used by a preferred recipe.

Profile mutations reuse ``DatasetProfiles`` validation. A preferred recipe must
name attached, available heads with exact hashes and a matching representation
scope and backbone. Positive component weights, pooling identities and supported
backbone combinations retain the existing native rules. Requests carry the
current dataset revision and an actor-bound operation ID. The settings update,
history and receipt commit together; an exact retry returns its original receipt
even after a later settings change. Every change advances the dataset revision,
so existing analysis results become historical rather than silently changing
their measured feature recipe.

The dataset API is ``/api/hosted/datasets/{dataset_id}/salad-profiles``. GET returns
the profile snapshot; POST accepts the existing attach, detach, rename, archive
and prefer_recipe actions with ``expected_revision`` and ``operation_id``.
``/sources`` lists paginated selectable retained artifacts. ``/register`` takes
``source_id``, an optional immutable ``head_id`` and ``display_label``, and the
same revision/operation fields. The pruning profile dialog uses these dataset
semantics through its authenticated hosted analysis proxy.

Frozen jobs capture their snapshot's profile settings, not whichever settings
happen to be current when the worker starts. ``HostedSaladProfiles.capture`` pins
the exact head artifacts for the existing native ``salad_heads`` cache namespace
and returns saved pruning defaults. Whole-image components map to
``scene_components``; bounding-box preferences map to ``object_components``,
``object_recipe``, ``object_adjustment`` and optional ``object_preprocessing``.
Explicit analysis choices retain precedence. This mapping is the existing
pruning UI contract; it does not invent an equivalent DQE recipe translation.

Registered heads also have central artifact references. Full recovery retains
the weights, metadata, aliases, all dataset setting versions and old snapshot
associations. Restore checks this reference closure. Pins are rebuilt from
content identities under the new server root; old absolute paths and inode
identities are not deployment authority. Migration retains raw original profile
records and recipes for inspection; it does not silently activate a recipe that
has not been selected and validated for the hosted dataset.

Focused verification uses real small saved heads, the original CPU loader,
SQLite concurrency, current and frozen profile state, restored paths and
authenticated HTTP mutations. Target GPU model execution, real WALDO migration
and the administrator browser flow require their separate acceptance checks.
