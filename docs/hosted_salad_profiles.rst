Hosted SALAD profiles
=====================

For the user-facing **Analysis presets** controls, see
`Hosted workflow <hosted_workflow.rst>`_. This reference describes how trained
heads and their settings are retained for
`hosted analysis <hosted_analysis.rst>`_. All guides are listed in the
`documentation index <index.rst>`_.

The admin **Analysis presets** entry starts with the shared Data Quality Explorer
recipes, rather than an empty per-dataset feature recipe. The catalog is resolved
from ``utils.class_analysis_quality`` and ``utils.class_analysis_salad``:

* Fast uses DINOv3.
* Compact combines DINOv3 67.5%, SAM3 22.5% and DINOv3 crop SALAD 10%.
* Thorough, the established default, combines DINOv3 50.625%, SAM3 16.875%,
  crop SALAD 7.5% and C-RADIO 25%.

These are three encoders plus a learned SALAD aggregation head, not four
backbones. Components are normalized, scaled by the square root of their
weights, concatenated and normalized again. The established size correction
is fitted on the analysed population for the first three components; C-RADIO
remains uncorrected. Experimental C-RADIO correction strengths and weights have
not replaced this shared default. Select the preset at DQE launch. The catalog
does not silently change saved dataset settings or completed analyses.

Whole-image descriptions and the final object-versus-scene pruning weight are
separate choices. A dataset can deploy its own trained whole-image heads;
their weights do not become universal global defaults. Custom pruning
overrides sit in a collapsed advanced section. No override is shown as a
single-DINO default when no such preference has been saved.

Administrators manage trained SALAD heads alongside these presets. The retained
artifact picker identifies the original name, trained backbone/model and whole
image or bounding-box role. Its sources are completed native analyses and
migration artifacts explicitly classified as SALAD weights, and trusted
administrator imports retained as ``salad_import_weights`` references. Imported
source metadata is stored separately in ``salad_imports``; the original
migration manifest is not rewritten. Registration never
accepts a filesystem path or a user-entered blob hash.

Registration runs the existing ``load_local_salad_head_file`` loader against the
exact retained bytes in a separate CPU process. This checks the original local
trainer policy, file format, normalization configuration and parameter shapes.
The control process does not load Torch or allocate a GPU head. This structural
qualification does not establish descriptor quality or GPU compatibility.
Inspection has a 512 MiB input limit, bounded architecture dimensions and a
90-second timeout; unsupported heads remain visible with an error. Large crop
training-row lists remain intact in the original head file. The catalog stores
their count and digest alongside the training signature and preprocessing
recipe, rather than returning thousands of coordinates with every request.

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

The admin-only ``/api/hosted/analysis-presets`` endpoint describes the shared
presets without requiring a dataset. The dataset API is
``/api/hosted/datasets/{dataset_id}/salad-profiles``. GET returns
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
The selected compatible DINOv3 crop head is also captured as
``crop_salad_head_id`` and ``crop_salad_head_sha256`` for DQE's Compact and
Thorough presets. Their feature weights still come from the shared preset.
The native producer checks the immutable head, encoder/model, training scope,
crop preparation and SALAD architecture before cache admission. It loads those
trained weights directly and records the original training population
separately from the current encoded population. Size-correction fits still
belong to the current analysed population. Fast ignores the crop head because
it has no SALAD branch. When no head is selected, the existing exact-training-
signature reuse and training behavior remains unchanged.

The explicit head hash enters both object-feature and whole-matrix cache
identities. An incompatible selected head fails with its mismatch rather than
silently retraining or substituting a different descriptor. Older native
whole-image ``embedding_salad_head_id`` remains separate and is not repurposed
to bypass its disabled DQE aggregation route.
Saved preference JSON is preserved exactly. At native capture, encoder-only
preferences are translated into the native producer names: pooled DINOv3,
pooled C-RADIO, or a trained ``local_salad`` whole-image component; objects use
the existing ``dinov3``, ``sam3``, ``salad`` and ``cradio`` identifiers. This
prevents a generic C-RADIO or SALAD preference from silently falling through
to the native producer's default DINOv3 branch. Crop SALAD currently requires
DINOv3, and each native object component can appear only once.

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
