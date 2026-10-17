Hosted analysis and training
============================

Administrators open **Data Quality Explorer**, **Dataset pruning** or
**Data ingestion** from the top navigation in either the dataset workspace or
annotation view. **Train models** lists the supported training tasks. Each
entry shows that tool's server-configured tasks and saved results. Leaving the
canvas saves the current image and releases its editing reservation first.
**Quality & compute** remains the advanced launcher for all job types.
Review the parameters and select **Start task** to run the selected operation.
If no compute recipes are configured, saved results remain readable; an operator
must configure the available tasks before a new run can start.

For Incoming diversity analysis, first select unassigned images in **Incoming**
and attach a whole-image SALAD descriptor in **Analysis presets**. For pruning,
start a configured pruning task here, then open its completed review page.
Changing that hosted review's producer recipe requires a new analysis.

See `Hosted workflow <hosted_workflow.rst>`_ for dataset approval and review,
`SALAD profiles <hosted_salad_profiles.rst>`_ for descriptor management, and
`Hosted inference runtime <hosted_runtime.rst>`_ for task configuration and
scheduling. All guides are listed in the `documentation index <index.rst>`_.

Canonical inputs
----------------

Hosted analysis starts from a saved, approved-only snapshot. The snapshot fixes
image membership, annotation versions, stable class IDs, split membership and
dataset settings. Incoming drafts and submitted work are never discovered by a
folder scan. Incoming diversity analysis adds a separately captured candidate
population; it does not promote those images into the confirmed snapshot.
Administrators alone can capture, inspect, export or apply results.

``CanonicalAnalysis.capture`` creates the job and, when needed, a new snapshot
in one SQLite transaction; it can also bind an existing immutable snapshot.
Its receipt identifies that exact capture. ``prepare`` materializes
the snapshot in a private directory and verifies every file before reuse.
``dqe_request`` translates this into the existing DQE active-workspace request;
client-supplied source paths and dataset IDs cannot override its binding.

The identity map connects each legacy numeric label row to its stable hosted
image ID, annotation version, object ID and class ID. It also records original
split membership. Explicit capture, sequence, video, flight, source and duplicate
group metadata is preserved through native analysis and grouped export, using
the same fields as the hosted split editor. Whole-image negative examples remain in the snapshot even
though they supply no object crops to DQE.

``resolve_yolo_input`` calls the existing dataset resolver in
``services.datasets`` with exactly one snapshot registry entry. It verifies that
the resolver returns the materialized tree. It requires an explicit train and
validation split: it does not silently reuse training images for validation.
Polygon training needs a dedicated polygon view because canonical exports retain
bounding boxes and polygon labels separately.

Before DQE or its review runs, the private native annotation sidecar is seeded
with the snapshot's existing image/object identities and exact label rows.
Legacy row-number migration must not replace these identities: object zero in
two images represents two distinct objects. The bridge verifies that every
identity remains unique and binds to the same row, without changing labels.
Unavailable held-out class metrics remain null in saved results; they are not
reported as measured zeros.

Recipes, caches and results
---------------------------

This boundary does not change DINOv3, SAM3, C-RADIO, SALAD, correction strengths or
fusion weights. Request parameters and frozen dataset model/recipe settings are
captured verbatim. Native engines remain responsible for recipe normalization,
model qualification and their established content/geometry/preprocessing cache
identities. Relabeling must not require new pixel descriptors; a changed fit
population must not silently reuse an incompatible fitted transform.

``publish_result`` retains engine output and its declared immutable artifacts
under the job. Engines must include exact resolved recipes, producer receipts,
cache provenance and fit-population identity. Register model/fit/result files as
artifacts so recovery includes them; a reference to an arbitrary external path
does not make that file recoverable.

Historical results remain readable after dataset edits. ``status`` marks them
stale. Historical image reads resolve the retained job's confirmed snapshot or
its separately captured Incoming candidates, so later archiving does not break
the saved result gallery. The administrator-only route is
``/api/hosted/analysis/jobs/{job_id}/images/{image_id}/media``; an image outside
that job's captured populations is rejected. ``apply_annotations`` permits an explicit human edit only when the entire
snapshot revision and targeted image version still match, and the administrator
holds the current editor lease. It calls the same transactional save mechanism
as the canvas. A successful exact retry recovers its receipt. A first correction
changes the population revision, so this generic application route requires a
fresh capture for another edit. The DQE review workflow described below instead
tracks its own corrections explicitly to support continued review. No legacy
annotation writer may bypass either admission boundary.

On Apple Silicon, the native child can bind ``MLX_DINOV3_MODEL_ROOT`` and
``CRADIO_MLX_ROOT`` to private model directories. Every file in either directory
must come from a hash-verified ``cache_inputs`` entry; source roots are never
inherited from the operator's home directory. SAM3's MLX package travels through
the same verified, offline Hugging Face cache mechanism. These bindings retain
the existing encoder implementations and do not substitute a different recipe.

Heavy-worker transport
----------------------

Large snapshots travel through a pinned manifest file, not through the worker's
bounded JSON IPC message. ``read_canonical_snapshot`` checks the manifest hash,
schema, snapshot/dataset/revision fields, private-directory containment and every
media hash/size. The short IPC reference contains only identity fields and the
manifest path/hash. The helper reads the manifest into memory after validation;
it is not a streaming numerical-analysis implementation.

Caption training export
-----------------------

In the annotation view, open **Caption whole dataset… → Export confirmed captions for training**, or choose
**Caption training export** in **Quality & compute**. This CPU task freezes the
confirmed dataset and calls the existing ``export_captions``, readiness checks,
instruction-bundle writer and checksum validator inside a private child. It
does not call a caption model or a paid service.

The export retains primary and alternate captions, generated questions and
answers, their validation/review states and original provider evidence. Equal
caption text does not discard distinct provenance. Known hosted model sources
are translated to the native generated-caption marker for downstream rules;
the exact original record remains in ``metadata.hosted_source``. Missing quality
or approval evidence is never filled in. Original train, validation, test and
unsplit membership is retained in the exported records and archive.

Inactive caption history remains inactive in native exports. A deleted hosted
record maps to the native ``superseded`` state; its original state remains in
``metadata.hosted_source``. Questions attached to an inactive caption cannot
become training examples independently of that caption.

An active current primary is matched only to active records. If the same text
exists solely in inactive history, review and export represent the current text
separately; accepting it adds a new active record without reviving a tombstone.
For imported dictionaries, a present ``caption`` field takes precedence, even
when null or empty. ``text`` is used only when ``caption`` is absent. Effective
text is trimmed for display, review and export while the original record remains
unchanged. The editor preserves both caption collections and writes an explicit
edit back to the record and text field it came from.

Exported image, caption and QA identities come from the server's current export
context. Extra metadata cannot replace those structural fields. The original
metadata remains available under ``hosted_source`` or ``hosted_caption_source``;
review and validation requirements still apply.

Import and annotation saves validate the known caption and question/answer
structures before recording a version. Optional null fields and legacy
string-only captions are supported; extra metadata and evidence are preserved
unchanged. Submission and approval repeat this check for older drafts. An
invalid field is reported by its location so it can be corrected; it is never
silently discarded or omitted from a canonical analysis.

Bulk approval records one actor-bound receipt per image. Retrying the same
per-image command returns its original outcome even after the image or dataset
is archived. That response describes the original approval, not current
visibility. A changed payload or a different dataset cannot reuse the receipt.

By default a failed readiness check produces a report without a training ZIP.
An administrator can explicitly turn off the readiness requirement to produce
a **diagnostic** bundle. The failure report remains included and the UI labels
that bundle as not ready. Downloads resolve only registered artifacts belonging
to the selected analysis; arbitrary file paths and annotator access are rejected.
Reports, bundles and frozen source identities are part of server recovery.

Saved analysis history is paginated. An administrator can reopen a task they
started to follow its progress or request cancellation after navigating away.
Completed results remain available when their originating snapshot becomes old.
Other administrators can inspect the retained analysis history; control of the
underlying runtime request remains with its initiating administrator.

Native execution
----------------

The following existing entrypoints implement the actual numerical work:

* DQE: ``localinferenceapi.create_class_analysis_job`` normalizes the recipe and
  queues ``_run_class_analysis_job``. Its active-workspace source accepts the
  explicit label manifest. Crop collection, cached embeddings, multi-backbone
  fusion, fitted size correction, neighbor search and selector remain there.
* Data Ingestion: ``create_data_ingestion_analysis_job`` uses the existing
  whole-image SALAD encoding, reference novelty and farthest-first selection.
  ``get_data_ingestion_distribution`` supplies the existing reference-fitted
  projection. The adapter adds stable hosted identities without changing scores.
* Whole-image SALAD: ``create_local_salad_training_job`` trains its gathering
  head on the confirmed snapshot using the existing augmentations and losses.
  The learned head is retained and can be attached in Analysis presets.
* Pruning: ``services.dataset_pruning_jobs.PruningService`` owns analysis and
  review. ``TatorPruningAdapter`` currently resolves legacy source authority;
  ``dataset_pruning_components`` reuses DQE/SAM/SALAD/C-RADIO producers and
  ``dataset_pruning_correction`` owns population-bound adjustment reuse.
* Statistics uses the existing YOLO image iterator and pruning label parser
  against the exact snapshot, including negative and unsplit images.
* YOLO, SAM3, RF-DETR and Qwen training launch through their respective
  ``localinferenceapi.create_*_training_job`` functions. YOLO's dataset resolver
  is already independently reusable; the launchers still depend on process-local
  registries and job state. CLIP/classifier training calls
  ``tools.clip_training.train_clip_from_yolo`` with the frozen train/validation
  membership. Its native feature construction, correction and fitting remain
  unchanged; the adapter replaces only random split selection.

``services.hosted.native_analysis:NativeAnalysisAdapter`` is a trusted heavy-job
factory. It launches a fresh child with private working directory, storage,
registry, HOME, temporary files and offline model cache. Only this child imports
the legacy API; it exposes no HTTP listener. The control process and ordinary
user inference workers still prohibit that import. Native mutations cannot
reach the hosted database; applying an edit uses hosted admission above.

Server configuration supplies ``native_root``, verified ``cache_inputs`` and
their allowlisted roots, optional model environment paths, and the permitted
request fields. Each cache input declares namespace, relative destination,
SHA-256 and size. Inputs are copied and verified; original files are never
hard-linked into a writable native cache. Descriptor SQLite references are
rebound only to matching attested copies. Missing or stale dependencies become
cache misses; they never fall back to the original filesystem. Raw vectors,
keys, dimensions and native recipe qualification remain unchanged. Fitted
transforms still require the original population and native validity checks.

Native pruning substitutes only source authority with the frozen manifest and
uses the existing component producers and planner. Training requires explicit
train/validation membership and refuses unsplit examples. A separate polygon
view is created for segmentation, leaving canonical box labels untouched.
Classifier training requires every validation class to be represented in
training. Supervised training rejects capture groups crossing declared splits.
Self-supervised SALAD training uses the confirmed images and does not require
an artificial train/validation split.
Qwen conversion restores explicit negative images omitted by the legacy
converter, and checks the resulting membership. SAM3's generated configuration
and child working directory use a private copy of the training source tree.

The child waits for native job finalization. After it exits, the parent seals
SQLite WAL state and hashes the final output files. A small hashed receipt
crosses IPC. ``publish_native_result`` validates it, retains all
declared result/fit/cache/model files in the artifact store and records their
relative paths. Exact publication retries survive temporary-tree removal.
``mark_terminal`` records failed, interrupted or cancelled compute attempts;
these cannot later masquerade as a successful result.

Incoming diversity analysis
---------------------------

Administrators select unassigned Incoming images before starting Data Ingestion.
Those candidates are captured separately, with exact image, annotation-version
and media identities. Assigned, submitted and confirmed images cannot enter the
candidate capture. The reference population remains the confirmed snapshot.
Both manifests and their media hashes are checked in the worker; durable media
references and the candidate record survive backup and restore.

The selected whole-image SALAD head must be attached to this dataset and match
the captured head hash. This hosted association replaces the legacy source-ID
check, which cannot recognize a migrated dataset. Original training metadata,
head scope and native numerical validation remain intact. Result rows and the
distribution graph carry stable image identities. No image is automatically
assigned, annotated, approved, archived or removed by the analysis.

Result views and pruning review
-------------------------------

``AnalysisViews`` reads retained DQE session databases through the existing
graph, queue, point and evidence queries. Object thumbnails use frozen media
and source-pixel geometry. The graph remains readable when its snapshot is
stale; applying an annotation correction requires the current hosted lease and
population revision.

``DqeReview`` reuses the annotation-entity transformer for relabel/delete and
the hosted geometry validator for box corrections. It saves complete immutable
image versions, preserving captions and unrelated objects. Deleting the last
object requires an explicit negative-image confirmation. A review tracks its
own successful edits, including several edits to the same image; an unrelated
canonical change stops further corrections against that analysis. Graph
coordinates and measured features remain the original frozen measurements.
Point responses distinguish the original version from this review's current
object class, box and version.
The correction recovery button stays disabled while the original request is in
flight. After a lost reply, it checks the same saved command before allowing a
new correction.

Bulk corrections group actions by image. Each image commits atomically with
an exact operation receipt; the batch records successful and conflicting
images separately. Processing can resume after interruption without repeating
successful writes. Cancel prevents subsequent writes; earlier successful
images remain saved. An expired lease or changed image must be reopened and
reviewed before submitting a new correction. This is not an all-or-nothing
dataset transaction.

The ``dqe_review`` heavy operation pins the retained parent session and its
exact confirmed snapshot, then runs the existing native Qwen review loop in
an isolated child. The selected registered Qwen package supplies the model;
browser paths and automatic model fallback are not used. The full native
response, model trace and generated evidence are retained as result artifacts.
The output is a proposal. Applying a correction still uses the hosted image
lease, version and population checks above. This adapter does not itself
claim browser parity or measured model effectiveness on the deployed GPU.

``HostedPruning`` mounts the existing pruning HTTP contract under one hosted
analysis ID. Its private review tree contains verified immutable results and
the exact fitted object rows retained by native analysis. Weight changes, loss
budgets, direct family comparisons, Apply, approval, Undo and grouped variant
exports use the existing pruning service. Applying a plan does not approve it.
The hosted writer fence rechecks the session and dataset revision immediately
before each native review-ledger commit. Numerical previews run outside it.

The ledger, immutable inputs and completed exports are checkpointed as immutable
artifacts referenced by the hosted database. Unchanged files are reused. Live
planning scratch directories are excluded. A
restored private tree preserves Keep choices, history and receipts; old approval
proofs are superseded because filesystem-bound accounting must be recomputed
and reviewed. Dataset source files and confirmed annotations are never deleted.
Frozen setup capabilities describe the measured recipe. Starting a different
producer recipe uses hosted heavy-job admission. Dataset SALAD-profile
management registers trained heads as retained immutable artifacts and saves
dataset-scoped labels, associations and preferred recipes. A frozen snapshot
keeps its exact head identities even after current preferences change.
The original recipe lab is available when the completed analysis contains all
four required object components and their saved correction fits. Native jobs
retain raw component rows and exact crop geometry beside the final fused rows.
Comparisons reuse the original correction, fusion, matching and calibration
functions. They do not encode images, train a head or fit new geometry. The
diagnostic calibration's final commit rechecks hosted authority; it cannot
approve a removal.

Older hosted publications can recover inline descriptor-cache rows and producer
matrices whose recorded SHA matches an artifact retained by that same job.
Recovery never follows an old absolute file path. If the required rows or fits
are absent, the lab explains that limitation while ordinary fitted-row review
remains usable. Completed comparisons and diagnostic records survive review
checkpoint recovery.

Local tests exercise a real CPU child, the actual image/label parsers, descriptor
reference reuse with unchanged numerical values and untouched original files,
durable result publication, snapshot identity, stale-edit rejection, and the
actual pruning preview/approval/export/recovery sequence. Local browser checks
exercise DQE point and bulk corrections, pruning preview/Apply, and saved-result
views against disposable fixtures. DQE tests also cover resumable corrections
and same-image continuation. Incoming capture and recovery
are tested with real SQLite and files. Ingestion/training dispatch and the VLM
adapter call use job/model doubles; these checks do not measure trained quality
or Qwen reasoning. Full
DQE/pruning GPU execution, model training, production crash recovery and the full
deployed browser workflow require separate qualification on the intended runtime.


Durable heavy-job publication
-----------------------------

Administrator-admitted heavy analyses use the scoped job authority described in
`Hosted inference runtime <hosted_runtime.rst>`_. Canonical capture, its job
grant and request binding commit
in one authenticated admission transaction. A crash before dispatch leaves a
recoverable binding; failed binding creation rolls back the capture. Logging out
after that admission commit does not cancel the job, including before the initial
request acknowledgement. All later reads still require a current login. The
native worker response is retained before publication; verified artifacts, the
canonical completed state, publication receipt and request binding commit in
one authority-checked database transaction. Account revocation or explicit
cancellation before this transaction prevents publication and records a terminal
state in both job views. A later cancellation does not undo a committed result.
Frozen dataset settings and snapshot provenance remain unchanged by subsequent
settings edits or archival; historical results report staleness separately.

After restart, only retained output may finish publication. Missing output is
interrupted and never silently recomputed. Recovery first checks an exact
already committed publication receipt, including owner, operation, payload hash,
canonical job and result identity. This repairs older failed or pending request
bindings without rerunning the numerical job. Legacy unfinished jobs with no
durable grant are interrupted; recovery does not invent authority for them.
Restoring a backup invalidates unfinished grants while retaining completed
historical analyses. A fresh authenticated login is required to inspect results.

Native materialization uses the encoded-coordinate working media recorded in
``hosted-media.json``. Its per-image map preserves the original source SHA and
size alongside the actual worker SHA, size, format and policy. Legacy engine
input manifests use the actual bytes for ``image_sha256`` and retain original
provenance in ``source_image_sha256``. Snapshot source identities never change.

Direct paid-caption preparation
-------------------------------

The direct paid-caption endpoint accepts JPEG, PNG and WebP sources; a TIFF
working PNG does not change that source-format policy. Source inspection,
working-image preparation and payload construction share the pre-dispatch
cleanup boundary. Any definite preparation refusal releases only an unsent
reservation with a zero-charge audit entry. Automation pause keeps the same
reservation for an exact retry; completed results replay without another send.
Dispatched and unknown outcomes are never refunded or resent by this cleanup.
New reservations capture the exact working-media descriptor in their audit
record and fingerprint. An exact completed older request replays before current
media preparation, subject to current session, image-version and editor
authority. An older unsent reservation without a frozen descriptor is
released only if it is still reserved, then returns ``media_policy_upgrade_required``:
start a new operation after reviewing the new pixels. If dispatch won that race,
the old request remains unresolved. These rules do not refund requests already
sent or silently change their inputs.
