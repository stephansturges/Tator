Hosted annotation assistance
============================

The hosted adapters return proposals for the existing annotation canvas. They
do not save labels or captions. The control process and supervisor authorize
the image, ownership generation, annotation version, editor lease, session and
installation before admission and again before delivering a result. Human
annotation changes still pass through the ordinary versioned save operation.

Private trained classifier
-------------------------

``services.hosted.assistance_adapters.ClassifierRuntime`` loads one trained head
and one backbone per user worker. It does not import the application monolith or
share a global CLIP/DINO/C-RADIO instance. The worker receives its selected model
and stable dataset class bindings from trusted server configuration; a tool
request cannot choose a filesystem path or change its model.

The classifier definition includes these fields:

* ``checkpoint`` and ``checkpoint_sha256`` for the saved joblib head;
* ``metadata_path`` and ``metadata_sha256`` for its saved recipe metadata;
* ``labels`` containing every foreground label present in the trained head.

The registry validates trusted paths and stores the two artifacts for recovery.
The runtime independently copies and verifies both digests before joblib loads
either object. These are executable pickle artifacts and must be registered
only by a trusted administrator. An annotator-supplied pickle is not accepted.
The private copy preserves the existing loader's adjacent ``.meta.pkl`` naming
convention even when durable storage uses digest filenames. Head and metadata
sizes are bounded at 256 MiB and 64 MiB respectively.

The saved recipe determines the actual encoder and feature composition:

* native CLIP, DINOv3 or C-RADIO backbone;
* crop shape, padding, canonical preprocessing and background treatment;
* single or multiple crop views, including their normalization and composition;
* DINOv3 pooling with register tokens excluded from patch means;
* fitted size-bias residualization, centering and standardization;
* linear binary, one-versus-rest or softmax heads, or the saved MLP layers;
* LayerNorm, ReLU/GELU, ArcFace output normalization, logit adjustment and
  temperature calibration.

The adapter reuses the existing classifier loader and crop/embedding helpers.
Its probability math follows the existing first-pane implementation, including
the background guard and second-label selection rule. ``__bg_`` labels are
recognized background classes; they need not be assigned a dataset class ID.
Foreground labels missing from the registered label list are rejected before
the backbone loads. Dataset bindings resolve model labels to stable class IDs
and current display names; the model does not invent a new dataset class.

``predict(image, xyxy, background_guard=True)`` receives immutable image pixels
and a pixel-coordinate box. It returns the label, probability, second label,
second probability and margin. SAM Auto uses the refined mask box as this input.
The worker maps the selected label to the frozen dataset class binding.
``close()`` releases that user's model references without a global GPU reset.

Only pooled recipes are currently supported by this hosted classifier adapter.
``local_salad`` Auto Class is also disabled by the existing local inference
contract. SAM3/SALAD fusion requires its own registered runtime head and is
rejected explicitly here; it must not silently become pooled inference.
Missing metadata, incompatible feature dimensions and malformed fitted recipes
also fail explicitly. The hosted path does not infer a substitute backbone.

SAM3 text and visual prompts
---------------------------

``Sam3PromptRuntime(network, device)`` wraps the user's already loaded SAM3
network in a processor. It does not load a second copy of that network. Every
request and tile starts with fresh image/prompt state. Requests remain serial
inside the user's worker.

``predict(image, payload, visual=False)`` supports text prompts, tiled text
search, and single or multiple visual boxes with positive/negative labels and
optional text. It preserves confidence and mask thresholds, minimum area,
result limits and mask simplification metadata. Visual results exclude boxes
that reproduce a seed with IoU greater than 0.9. The box-only checkpoint path
uses the network's grounding output and returns boxes without fabricated masks.

Tiled text search reuses SAHI slicing, offsets boxes and masks back to the full
image, and performs score-ordered cross-window suppression before the requested
final limit. A small final limit does not restrict each tile below 50 results.
Packed candidate masks spill to private temporary files until fusion; final
masks are reconstructed one at a time. Normal completion and exceptions remove
that temporary directory. Forced process termination can leave temporary
directories for operator cleanup.

The helper returns detections, warnings and aligned encoded masks. Boxes use
the existing normalized YOLO representation. Prompt text is descriptive output,
not a new class ID. The user adapter adds its image token and, for SAM3 Text
Auto, applies the same registered classifier to each resulting object.

Admission rejects malformed prompts and oversized work explicitly. The prompt
adapter allows at most 1,024 visual seed boxes, 256 windows, 10,000 candidates,
512 MiB of packed temporary masks, and 32 MiB of encoded masks before their
duplicated response representation. It never silently truncates masks to meet
these limits. Tiled visual prompting is not an existing first-pane operation
and is rejected.

Local Qwen detection and captions
--------------------------------

``services.hosted.qwen_adapters.QwenRuntime`` runs in the same user worker and
loads only registered local Transformers model snapshots. ``model_path`` plus
``files`` declares every relative file and SHA-256 digest. Optional
``adapter_path`` plus ``adapter_files`` registers a PEFT adapter separately.
The runtime verifies the complete inventory and hashes before loading with
``local_files_only=True`` and ``trust_remote_code=False``. Undeclared files,
changed bytes and symlinks fail. Model files remain part of the registered
portable model manifest used by hosted recovery.

The server supplies the allowed models, default, refinement and fallback
selections, actor identity, explicit device and image pixel limits. Browser
payloads cannot select an arbitrary model directory or provide credentials.
Only model IDs in that server-owned set may be selected. Registered model names
containing ``Thinking`` or ``Instruct`` retain the existing variant behavior.
This native adapter targets the hosted Transformers/CUDA deployment; it does
not silently translate an MLX checkpoint into a different model.

The static ``qwen_caption_pipeline.py`` extraction preserves the existing
caption, detection, generation and recovery functions. Its explicit
``REQUIRED_BINDINGS`` list covers model loading, image resolution, progress,
raw model I/O, cancellation, prompt settings and private state. ``bind_pipeline``
rejects missing or unknown dependencies and gives each worker a separate
function namespace. There is no runtime reading, execution or import of
``localinferenceapi.py`` or ``app``.

The caption workflow retains visual windows, optional text-only full-image
composition, Thinking draft/refinement, window reconciliation, glossary/count
handling, cleanup, safe retry and configured fallback recovery. The supplied
model generation calls retain the existing Transformers prompt, decode and
loop-guard behavior. Recovery unloads only that user's Qwen references.
``unload_others`` is disabled at the hosted boundary; attempting a global
SAM/classifier/detector unload raises an error. Heavy-job coordination remains
the supervisor's responsibility.

``execute("qwen_infer", image, payload, context)`` returns the existing parsed
detection response. The default and manual detection prompt protocol uses
0--1000 coordinates. The hosted boundary corrects the inherited mismatch in
which these coordinates were scaled using processor pixels. Point and
``bbox_sam`` proposals call the same user's SAM hook against the admitted
immutable image. They never select another image through a token or path.

``execute("qwen_caption", image, payload, context)`` returns the existing
caption response plus bounded evidence. Raw generation inputs and outputs are
retained, including loop failures before retry. Image inputs are identified by
dimensions inside this evidence; the enclosing supervisor request supplies the
authorized image identity. Internal media paths, session material and provider
credentials are not returned. Raw model I/O has an 8 MiB limit; progress retains
the latest 256 events with an explicit dropped-event count. A response exceeding
the evidence limit fails rather than silently hiding model evidence.

Progress is currently returned with the completed worker response. Live
generation-token streaming into the hosted canvas is not qualified by these
adapter tests. Cancellation and request timeouts use the supervised worker
lifecycle; model-level cancellation flags are also available inside the worker.

``execute("qwen_caption_preview", image, payload, context)`` preserves the
existing full prompt-flow preview with the same caption request schema. It
shows actual windows, selected caption/refinement models, prompt sections,
conditional guard templates and approximate token budgets without loading or
running a model. The admitted image supplies authoritative dimensions. This
preview is not a substitute for the generated-dependent prompts recorded by an
actual caption run.

Paid OpenAI dataset-caption jobs are a separate existing workflow. This local
Qwen adapter does not call a paid provider or accept an API key. Hosted provider
credentials, spending reservations and asynchronous batch reconciliation require
the separate provider gateway and canonical dataset-job integration.

Saved Detection Recipes
-----------------------

``services.hosted.edr_adapters.DatasetEdrAdapter`` executes one registered recipe
as a complete sequential dataset pipeline. It returns proposals; it never
saves annotations. The administrator-selected package contains the recipe,
labelmap, optional calibrated ensemble and feature contract. Component roles
resolve to pinned registered detector, SAM3, trained classifier and, when
required by the saved recipe, Qwen identities. Reinstalling a component requires
explicit recipe reconciliation. Browser requests supply only the selected
``recipe_id``; they cannot replace parameters or component paths.

The adapter reuses the existing deep prepass services: full-image and SAHI
detectors, SAM3 text and window extensions, synonym generation, similarity
exemplar selection, global/windowed visual expansion, provenance and cleanup.
The exact trained classifier gate, crop recipe and final cluster registration
are preserved. Server-owned windows may use intersecting exemplar boxes whose
full geometry crosses a crop boundary, matching the original recipe behavior.
Ordinary interactive visual boxes remain bounded by their image.

Canonical EDR packages disable caption generation and synonym expansion, as
the existing package runtime does. Plain saved recipes retain those optional
Qwen stages. Recipe-owned Qwen runs inside the dataset pipeline with immutable
model selections, a private synonym cache, fresh image/prompt state and cleanup
after each request. Interactive Qwen remains in the user's private worker.
Unmapped saved Qwen model associations fail before inference. No global model
unload or legacy HTTP endpoint is used.

The recipe endpoint preserves the existing ``caption=None`` public response:
running a Detection Recipe does not apply a caption. When enabled, its overall
caption and window captions are retained explicitly in
``hosted_evidence.recipe_caption``, alongside raw Qwen I/O. Cached Qwen runtime
calls retain their actual model identity in prompting and evidence even when
the worker's default selection differs.

For calibrated ensembles, the adapter retains the raw prepass candidate pool
until scoring. ``edr_features.py`` contains the existing feature calculations
with explicit I/O bindings for one immutable image and its admitted annotation
version. Existing labels supply that version's geometry priors. Classifier
probabilities, projected embeddings, source support, contextual features and
term hashing use the recorded feature options. The generated feature schema
must match the saved contract exactly. A private temporary directory carries
the original NPZ/JSONL scorer formats and is removed on ordinary completion or
failure. Forced worker termination may leave temporary files for cleanup.

The existing Torch MLP and XGBoost scorers retain calibration temperatures,
per-class thresholds, split heads, quality specialists and selected learned
policy layers. Every policy dependency must resolve to a declared file inside
the registered package; a same-named file in the process working directory
cannot override it. Final detections map recipe labels to current stable dataset
class identities. Model or required-stage failures fail the proposal request;
unscored raw candidates are not returned as a successful recipe result.

Saved registry fingerprints, report metadata and historical source paths remain
in the immutable package but are not runtime inputs. The adapter does not open
the original dataset or discover models through those paths. Evidence includes
raw stage results, provenance, feature schema, calibrated row scores and any
Qwen inputs/outputs. It is bounded at 16 MiB; exceeding the bound fails explicitly.
The static feature and classifier-gate extractions never import or execute the
application monolith.

Verification and remaining qualification
----------------------------------------

Run the focused mechanism checks with::

    .venv-macos/bin/python -m pytest -q tests/test_hosted_assistance_adapters.py
    .venv-macos/bin/python -m pytest -q tests/test_hosted_qwen_adapters.py
    OMP_NUM_THREADS=1 .venv-macos/bin/python -m pytest -q tests/test_hosted_edr_adapters.py

These checks use real sklearn/joblib artifacts, production recipe loading and
math, Torch reference calculations, production SAHI slicing, and production
mask encoding. Backbone/processor model calls use explicit CPU doubles. They
exercise calibrated multiview predictions, saved DINO pooling, private model
instances, artifact/label rejection, SAM3 prompt isolation, visual positive and
negative prompts, mask alignment, tiled fusion and box-only grounding.

They do not qualify real model accuracy, GPU allocation, memory pressure,
checkpoint compatibility or three-user performance on the target machine.
Those require actual configured weights and the user's remote GPU environment.
Backbone identifiers in saved classifier metadata currently resolve through
the existing model loaders. Fully portable offline recovery must also account
for those backbone downloads or pinned local model artifacts; retaining the
trained head and metadata alone does not prove offline recovery.

The Qwen checks invoke the actual extracted orchestrator and native generation
code with model/processor doubles. They cover both windowed strategies,
Thinking draft/refinement across two registered model IDs, loop recovery,
normalized detection coordinates, same-image SAM handoff and independent user
namespaces. They also verify real local snapshot inventories and hashes.
They do not establish quality or compatibility of actual Qwen checkpoints,
quantized weights, PEFT adapters, GPU streaming or paid-provider execution.

The recipe checks execute the actual prepass, feature calculation, classifier
gate, calibrated MLP and XGBoost selected-policy paths using real trained
classifier/ensemble artifacts and explicit native-model doubles. They verify
SAM3 text/window/similarity stages, optional Qwen stages, stable class mapping,
canonical package metadata, schema rejection and failure cleanup. On the
development Mac, importing Torch before default-thread XGBoost causes a native
OpenMP crash; ``OMP_NUM_THREADS=1`` allows the same actual XGBoost check to run.
That native test is skipped on macOS without this setting. Linux/CUDA library
compatibility, actual saved production packages, resource use and useful model
outputs still require target-machine qualification.
