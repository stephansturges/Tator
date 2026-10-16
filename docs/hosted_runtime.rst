Hosted inference runtime
========================

``services.hosted.runtime`` supervises private child processes without loading
models in the API process. A user owns one worker containing their SAM predictor
pool and class predictor. A dataset owns a separate detector worker. Each worker
executes one complete request at a time, so detector tiling, postprocessing and
other pipeline stages cannot overlap on that dataset's instance.

This is the execution boundary, not the permission or annotation store. The API
must authorize image membership and editing authority, and use the original
version, ownership generation and editor identity when applying a result. A
completed inference is not a saved annotation.

Integration interface
---------------------

Create one ``RuntimeSupervisor`` in the hosted control process, with an exclusive
state directory on local disk. Its file lock rejects a second supervisor using
the same directory. Do not run multiple API workers with independent supervisors.

``RequestContext`` captures:

* ``actor_id``, ``dataset_id`` and ``image_id``;
* ``annotation_version`` as a string and ``ownership_generation`` as an integer;
* ``editor_id`` for the editing lease, plus optional ``workspace_id``;
* ``session_id`` for an opaque or hashed session identity, never its bearer token,
  and ``instance_id`` for the authentication store's server instance;
* a frozen copy of the model configuration.

``AdapterSpec(factory, config, device)`` is registered by trusted server code.
``factory`` is a Python ``module:attribute`` path. Neither that path nor arbitrary
adapter configuration may come directly from an inference request. Its identity
hash includes factory, configuration and device. Configuration changes invalidate
queued requests and completed results from the old configuration. A running old
request can finish its computation, but its result becomes stale.

The main methods are::

    supervisor.register_user(user_id, user_adapter)
    supervisor.register_dataset(dataset_id, detector_adapter)
    request_id = supervisor.submit(context, operation, payload, owner="user")
    request_id = supervisor.submit(context, operation, payload, owner="dataset")
    supervisor.request_status(request_id, actor_id=user_id)
    supervisor.result(request_id, actor_id=user_id)
    supervisor.cancel(request_id, actor_id=user_id)
    supervisor.set_active("user", user_id, False)

Status, results and cancellation require the originating actor. The caller must
derive that actor from the authenticated session. ``snapshot()`` includes all
owners and is administrative; do not expose it unchanged to annotators.

The ``authorize(context, phase)`` callback raises or returns ``False`` to reject.
It runs at ordinary request admission (``admit``), immediately before dispatch
(``dispatch``), after computation and whenever a result is retrieved (``result``).
Check the live account, dataset membership, image version, assignment generation
and editor lease each time, including revocation of the captured session. Heavy
jobs use ``heavy`` at admission and dispatch, then ``heavy_result`` at completion
and retrieval. Admin ``/tools/heavy`` admission mints a durable internal job grant
under the current authenticated login. Its request, owner, required admin role,
account authority epoch, installation, dataset, snapshot, operation and adapter
identity (factory, device and frozen configuration) are immutable. The context
contains only an internal grant reference; clients cannot supply it.

An admitted heavy job continues after logout or session expiry. Account disable,
role changes and credential reset revoke unfinished work permanently, including
a disable/re-enable or demote/re-promote before the next worker poll. No-op account
updates and logout do not advance the authority epoch. Admission, dispatch,
loading completion, computation completion and publication validate the grant;
the reaper cancels workers whose grant was revoked. Restore changes the
installation identity and interrupts unfinished grants. Status, cancellation and
result reads still require a current login, original ownership and admin access.
Completed retained results remain readable under that current access check.

This authority applies only to the admin heavy-job endpoint. Interactive image
assistance, per-image captions and caption batches retain their captured-session,
image and editor fences, even when their coordinator uses the heavy scheduler.

Keep these callbacks short and bound their database waits. They execute outside
the scheduler lock. They cannot replace the final annotation compare-and-swap:
authority can change after a callback returns. The API must also bind any
temporary media path or artifact reference to the authorized request and keep
it alive through worker consumption. Do not allow clients to choose filesystem
paths or resolve an image through a global current-image variable.

Adapter contract
----------------

The trusted factory runs inside the child::

    adapter = factory(config=config_dict, device=device_string)
    result = adapter.execute(operation, payload_dict, context_dict)

The result must be JSON serializable. The child rejects nonfinite JSON values.
An optional ``close()`` method handles normal shutdown, but adapters must tolerate
forced process termination: cancelled, timed-out and idle workers may be killed.
Persist training artifacts atomically in an approved job directory; inference
workers must not save annotations. Adapters must whitelist operations and retain
no implicit cross-user context.

The user adapter should own instantiated SAM current/previous/next slots and a
classifier, with pixel caches keyed by dataset, image and media digest. Version,
editor and session authority are checked separately on every request. A dataset
adapter should own its loaded detector and execute the entire
sequential pipeline in ``execute``. Standalone SAM3 text/visual helpers, caption
and Qwen adapters must use the same captured context and process boundary.

Workers communicate through size-bounded binary JSON frames over private pipes.
Python and native stdout logs are redirected to a bounded private stderr buffer;
they cannot corrupt the protocol or become another user's result. The worker
actively rejects imports of ``localinferenceapi`` and ``app``. Existing model
logic runs in dependency-light adapters. The native analysis adapter is a
deliberate compatibility exception: it launches a fresh, isolated child that
imports the legacy numerical engines with a private snapshot registry and
private working directories. It does not start an HTTP service or relax the
ordinary worker's import guard. See ``hosted_analysis.rst`` for that boundary.
Plugins are trusted server code, not a sandbox for arbitrary code.

The supervisor never changes the parent's CUDA environment or invokes a global
model unload. Supply explicit devices to adapters. Workers have their own process
groups, and cancellation only signals a group created by this supervisor. On
Linux a parent-death signal terminates the worker if the control process dies;
a parent watcher also handles ordinary parent loss on macOS. A separate tiny
guardian, with no model or protocol pipes, remains in the worker's process group
and kills that group if the worker or control process dies. This covers inherited
children used by an isolated training/analysis adapter. Adapters must not launch
detached children; this module does not provide cgroup containment.

Native adapter coverage
-----------------------

``ModelRegistry`` in ``services.hosted.config`` validates trusted definitions
under approved model roots and binds checkpoints (and SAM3 BPE data) by SHA-256.
It creates fixed ``UserToolAdapter`` or ``DatasetDetectorAdapter`` specifications;
dataset/user selections still belong in canonical server settings. Updating a
checkpoint changes the runtime identity and invalidates old work. Device choices
are explicit, with no silent CPU fallback or global CUDA change.

The current native adapters implement SAM1 and SAM3 point, multipoint and bbox
prompts, local bbox refinement, and bounded current/next/previous preload slots.
They emit the existing normalized bbox and packed-mask formats. SAM image caches
key pixels by dataset, image and media digest within the private model instance.
Saving annotations or switching editor tabs does not recompute unchanged pixels.
Every cache hit still passes the complete session/version/editing authorization
at admission, dispatch and result; a cached image never authorizes access. The
current model loads at worker startup; additional slots load as used. Restoring
a worker restores its selected current model; old image features are recomputed.
The read-only ``sam_state`` operation reports actual actor-private slot names,
loaded state, image identities and tokens within the requested dataset. It does
not set an image or compute its features. The legacy ``/sam_slots`` viewer uses
this authenticated operation rather than a fabricated capability response.

YOLO and RF-DETR bounding-box models support full, region and windowed detection.
The implementation reuses the existing extraction, parameter clamps, SAHI slicing
and class-aware NMS. A complete request runs sequentially in the dataset worker.
RF-DETR definitions must supply explicit native ``class_ids`` in label order;
unknown native outputs fail instead of guessing an index offset. Custom YOLO
head-graft compatibility, segmentation detector modes and exact model versions
still require qualification with the real selected artifacts.

The API injects ``media_path`` and ``media_sha256`` after membership authorization.
Workers require a regular file under their configured media root, verify one
immutable byte snapshot, then decode it without resizing. Limits are 64 MiB
compressed media, 64 million pixels and 1,024 tiles per detector request. A
preload is read-only preparation of that exact image; the API must authorize a
neighbor image separately without borrowing the current image's editing lease.

The user worker also owns a trained classifier when one is attached to the
dataset. ``classify_bbox`` and the three ``sam_*_auto`` operations use
``ClassifierRuntime`` with the saved classifier head and mandatory recipe
metadata. Automatic SAM classification crops the predicted mask's bounding box,
preserves calibrated probabilities and applies the existing foreground guard.
See ``hosted_assistance.rst`` for supported encoder and recipe details.

SAM3 text, windowed text and visual prompts reuse the current user's existing
SAM3 network through a private ``Sam3PromptRuntime``. ``sam3_text_auto`` adds
the dataset's trained classifier to each mask's bounding box. No second SAM3
model is loaded for this helper.

``qwen_infer`` and ``qwen_caption`` lazily create the user's local Qwen runtime.
Detection keeps bbox, point and bbox-plus-SAM modes; the SAM hook uses only that
user's current predictor. Captioning retains the existing multi-stage/windowed
pipeline, configured model choices and loop recovery. Its bounded evidence
contains actual model inputs/outputs and progress at completion. Live status
reports the scheduler phase; it does not pretend to stream tokens or know a
percentage complete. Models load only from registered local file manifests.
Trusted worker payloads carry ``media_sha256`` for the verified encoded working
bytes and ``source_media_hash`` for original provenance. Qwen accepts that
server-owned provenance as an internal field, validates its digest and excludes
it from the model request. Browser parameters cannot supply either identity.
These CPU component proofs still require real-weight tool qualification.

Authenticated API and model installation
---------------------------------------

``install_tool_api`` adds the controller to the hosted application's lifespan.
The supervisor starts lazily. API requests provide image/dataset IDs, expected
annotation version, ownership generation and editing-tab ID. They cannot supply
media paths, checkpoints, worker factories or model overrides. The controller
captures the authenticated session and installation identity, validates live
authority at every runtime phase, and pins the authorized immutable media until
execution ends. A neighbor preload checks read access and the exact image
version without taking its editing lease.

An administrator supplies ``TATOR_HOSTED_MODEL_CONFIG`` with approved source
roots, model definitions, permitted user SAM choices and explicit devices::

    {
      "allowed_model_roots": ["/srv/tator-model-source"],
      "models": {
        "sam-base": {"kind": "sam1", "model_type": "vit_b",
                     "checkpoint": "/srv/tator-model-source/sam.pt"},
        "dataset-detector": {"kind": "yolo", "labels": ["car", "bus"],
                             "checkpoint": "/srv/tator-model-source/detector.pt"}
      },
      "user_sam": {"default_model_id": "sam-base",
                   "allowed_model_ids": ["sam-base"],
                   "device": "cuda:0", "capacity": 3},
      "detector_device": "cuda:1"
    }

These devices are an explicit configuration example, not a measured capacity
recommendation. Classifier definitions use ``kind: classifier``, foreground
``labels``, a checkpoint and ``metadata_path``. SAM3 additionally requires
``bpe_path``. SHA-256 fields may pin expected bytes. Installation copies these
files into immutable artifact storage, registers every required file for full
recovery, and publishes a portable model definition only after registration.
Installed models and selections survive loss of the original source directory.
Never accept executable classifier pickle/joblib files from inference payloads.

Local Qwen definitions use ``kind: qwen``, ``model_path`` and a ``files`` mapping
from relative names to SHA-256 digests. Optional ``adapter_path`` and
``adapter_files`` register a PEFT adapter. Installation builds an immutable
directory containing only declared files, registers each file plus a canonical
descriptor, and stores their digests for full recovery. Paths containing links
or escaping the model directory are rejected. ``user_qwen`` supplies
``allowed_model_ids``, ``default_model_id`` and an explicit ``device``, with
optional ``refinement_model_id``, ``fallback_model_id`` and pixel limits.
Caption model controls may choose only installed permitted IDs. Native loading
uses local files and disables remote model code. Paid providers use a separate
authorization and spending boundary; local Qwen configuration grants no provider
credentials or permission to make paid calls.

The catalog's ``parameters`` map supplies the exact allowed Qwen request fields
to the legacy canvas bridge. Image bytes, tokens, dimensions, model paths and
global unload switches are server-owned. Existing caption prompt, sampling,
windowing and refinement controls stay available. Server ``request_timeouts``
may set per-operation execution deadlines; a client cannot extend its own limit.

The API exposes:

* ``GET /api/hosted/tools/catalog?dataset_id=...`` for permitted models and current
  selections; ``POST /models/{id}/install`` is administrator-only.
* ``POST /sam`` for the user's permitted SAM choice with a preference revision.
* ``POST /datasets/{id}/detector`` and ``/classifier`` for administrator model
  attachment with the expected dataset revision and an operation ID.
* ``POST /request`` followed by ``GET /requests/{id}``, ``/result`` or
  ``POST /requests/{id}/cancel``. Only the originating actor can inspect the job.
* ``GET /state?dataset_id=...`` for the global pause phase, the current actor's
  worker state and configured capabilities. It exposes no other user's worker
  identities, queues or process IDs. ``GET /runtime`` is administrator-only.
* ``POST /heavy`` for an administrator-selected server-configured heavy adapter.

Dataset attachments freeze model-label to stable dataset-class IDs. Initial
attachment matches class names, or accepts an explicit admin ``class_mapping``.
Unknown or retired classes are rejected rather than using a model's numeric
class index as a dataset label. At execution, IDs resolve to the dataset's
current names and positions; rename/retirement makes old captured results stale.
Results include ``stable_class_id`` alongside the legacy numeric/name fields.

Heavy submissions first reserve a durable actor-bound operation receipt, then
capture and pin a confirmed-only dataset snapshot. Retrying the same command
recovers its original request ID and never launches the computation twice. If
the process dies between reservation and admission, the retry reports that the
request is unavailable and requires a new explicit command. Heavy factories and
parameter allowlists come exclusively from trusted server configuration. This
dispatch seam connects the native adapter described in ``hosted_analysis.rst``.
Admin catalogs expose each named job's operation, allowed parameters and public
defaults; factories, devices and private roots remain server-owned.

Native jobs also capture a durable ``analysis_job_id``. After computation, their
public status remains ``publishing`` until the hashed receipt and every output
artifact have been verified and retained by ``CanonicalAnalysis``. The small
worker response is persisted before publication; restarting can finish that
publication without rerunning compute. Completed results and command retries
survive removal of private working files. Publication failures remain failures
and retain private evidence. A crash before the worker response is recorded can
still lose that response; the command is never silently executed again.

Heavy IPC carries a ``canonical_snapshot`` reference containing ``snapshot_id``,
``dataset_id``, ``revision``, ``class_version``, ``settings_version``,
``manifest_path`` and ``manifest_sha256``. The immutable JSON manifest preserves
the frozen snapshot fields and adds verified pinned ``media_path`` values for
its images. It lives under the server-injected ``config.canonical_root`` until
execution and native publication finish. Adapters must verify its path, digest and identities before
reading it. Transporting a reference keeps a large dataset's annotations out of
the size-bounded IPC frame; it does not make the model's own memory use bounded.

Saved EDR pipelines
-------------------

An ``edr`` model is a portable, hashed directory package containing the saved
recipe and its ensemble, metadata and feature-contract files when applicable.
Its ``components`` map names installed YOLO, RFDETR, SAM3, classifier and optional
Qwen models. Installation freezes each component's identity. Updating a component
requires an explicit recipe reinstall and dataset selection; the runtime cannot
quietly replace it beneath a trained ensemble.

``POST /datasets/{dataset_id}/edr`` uses the same revision, operation receipt and
stable-class mapping rules as detector selection. ``edr_prepass`` accepts only
the selected ``recipe_id``. The server supplies current annotation objects after
checking the actor, editor lease, image version and ownership generation.

Direct detection and EDR share one dataset queue and a stable pipeline identity.
Their native adapters load lazily. Recipe label semantics differ from direct
dataset label mappings, so their native detector instances are currently
separate even when weights match. A recipe can own a fixed Qwen component for
synonym/caption stages; that component uses request-scoped image/prompt state in
the serialized dataset pipeline. Interactive Qwen remains user-private. These
optional instances increase GPU residency and must be measured on the target
host; no memory-fit claim follows from the CPU composition checks.

Durable caption jobs
--------------------

``POST /caption-jobs`` accepts one canonical image context (dataset, image,
version, generation, editor and optional workspace), an ``operation_id``,
``caption_request`` and ``settings``. The catalog lists the exact accepted
caption and job settings. Current-image jobs belong to their authorized editor;
administrator batches use the separate confirmed-snapshot interface below.
``provider`` defaults to ``local_qwen``. ``openai`` uses the same native caption
and QA orchestration through the metered gateway described below.

Each attempt uses the private Qwen worker's actual prompt preview, caption
pipeline and, when requested, the existing instruction-QA generation, top-up and
verification helpers. Existing caption quality checks and degraded-rate checks
decide whether a result passes. Raw output from unsuccessful attempts is
retained as well as successful output. A usable caption can still fail the
configured recovery-rate gate; the status retains that caption and explains the
gate failure instead of presenting the run as healthy.

``GET /caption-jobs`` lists the actor's own jobs. ``GET /caption-jobs/{id}`` and
``/result`` return the status and frozen context. ``POST /{id}/cancel`` takes an
operation ID; ``POST /{id}/resume`` takes the original request with a new
operation ID and a currently authorized editor/session. Resume preserves the
image/version/generation, caption settings, model identity and previous attempt
history. Repeating either command is idempotent. A controller restart records
the interrupted attempt. With ``set_and_forget`` enabled, at most
``max_auto_resumes`` (default two) automatic resumptions may occur. A valid
original session and editor lease are checked again. An expired lease puts the
job in ``waiting_for_editor``; the ordinary editor-acquisition endpoint can
resume it after granting a fresh lease for the same unchanged image version.
Only the exact generation transition checked by that acquisition is accepted;
assignment or annotation changes cannot be skipped. Recovery never acquires a
lease itself, and failure to resume does not undo a successful editor acquire.

Jobs wait while admin compute suspends new automation. Loss of image authority,
session validity, model selection or class-map identity stops a job. Result
publication repeats the guarded image check while recording artifact references.
There are no annotation or filesystem label writes: the canvas applies a chosen
caption through the ordinary human-controlled save. The default queue permits
eight pending caption jobs per user (server setting
``caption_jobs.pending_per_user``); the runtime's global request limits still
apply. Legacy MLX-specific image-size retry policies are not applied to this
registered Transformers runtime.

Confirmed-snapshot caption batches
---------------------------------

Administrators start ``POST /caption-batches`` with ``dataset_id``, an existing
``snapshot_id``, an operation ID, the same caption/settings dictionaries,
``sample_size`` (zero means all confirmed members), and ``sample_seed``.
Native stress-plus-random selection preserves dense, varied and empty examples
when sampling. Incoming records and unapproved edits are excluded. One heavy
worker executes the same caption/QA helper sequentially over pinned pixels and
frozen label hints. Annotators' manual editing and saving continue during the
automation pause. No inference result saves annotations automatically.

Every attempt writes one atomic evidence file. The controller verifies its
case identity and retains its raw output in the artifact store before recording
the attempt in SQLite. A restart reuses successful retained cases and records
an interrupted attempt; it does not regenerate already successful captions.
Retries have both a per-image attempt limit and a bounded automatic-resume
count. Expired admin authority requires an explicit resume from a valid admin
session. Cancellation and per-image timeout terminate only the owned heavy
worker group. Complete evidence survives removal of the private working files.

``GET /caption-batches`` lists the actor's jobs; ``GET /{id}`` and ``/result``
return status and the snapshot/recipe identities. ``GET /{id}/captions`` pages
through retained proposals (``offset``, ``limit`` up to 100). Cancel and resume
take a new operation ID. The admin catalog supplies exact request defaults.

Optional pilot checks reuse the existing native degraded-rate, elapsed-time,
stress-sampling and one-sided Wilson calculations. They require a generated
sample, actual prompt-budget events, bounded runtime projection and acceptable
recovery rates. Their scope is the hosted sequential local-Qwen runner; they do
not claim certification of the legacy parallel/MLX launcher. A completed batch
can have ``certification.status=not_certified``. In particular, a positive
confidence bound cannot certify an exact zero true recovery rate from a finite
sample: that combination stays visibly unmet. ``certified_pilot_id`` permits a
subsequent run only when the pilot passed for the same snapshot, recipe and
immutable model identities. No synthetic generation result is GPU qualification.

Metered provider caption recipes
-------------------------------

Current-image jobs and confirmed-snapshot batches also accept
``provider="openai"`` with ``provider_options`` containing a permitted ``model``,
``detail`` (low/high/auto, plus original only when explicitly allowed by the
operator's model metadata), ``max_output_tokens``, optional ``reasoning_effort``,
and ``service_tier`` (default, or its standard alias). An administrator must
configure the provider credential, permitted dataset/models, current prices,
token limits and spending limits first. Catalog responses expose permitted
models and option names; credentials remain inside the gateway.

The paid runner changes only the visual/text generation primitive. The existing
caption pipeline still selects windows, builds each prompt, merges descriptions,
performs requested refinement and recovery, and generates/verifies instruction
QA. Each crop call, text-only merge and QA call goes through the spending ledger.
Caption and QA stages read the same encoded-coordinate working image, including
when a source TIFF would normally rotate on opening. Ledger provenance keeps
the original source hash while each visual call records the pixels it sends.
Local sampling knobs (temperature, top-p/top-k and presence penalty) are rejected
when explicitly populated. The remote model's generation policy and the
separate provider options apply. One permitted provider model is used for every
stage; a local model ID cannot silently select a different paid model.

Paid runs use CPU orchestration without loading a local neural model. The
gateway sends requests to the fixed Responses endpoint with image inputs where
needed and ``store=false``. Exact prompts, options, derived pixels, raw results
and reported token usage are retained. The API's image-input structure is
documented at https://developers.openai.com/api/docs/guides/images-vision.

Every stage has a stable identifier derived from the job, image, attempt and
stage number. After interruption, the same attempt replays settled stages from
their retained results. An uncertain dispatched call is never automatically
sent again. A failed transport cannot become a new paid attempt through the
native caption or QA recovery loop. Cancel and the gateway's dispatch admission
serialize in SQLite: cancellation committed first prevents the next external
call. A request already sent may finish and be billed; its result cannot bypass
the current image/session or snapshot authority checks. Unfinished paid attempts
do not consume a new attempt number merely because the controller restarted.
During a GPU-heavy window, future paid stages pause too. A reservation caught
between preparation and dispatch remains undispatched until restoration; resume
uses that same stage ID. Cancellation while paused releases its reservation.
The per-image timeout excludes this waiting period. Already dispatched provider
requests may settle while the local GPU job runs.

The direct ``POST /api/hosted/providers/caption`` route uses the same dispatch
pause check. While automation is paused it returns ``409`` with
``runtime_automation_paused`` without contacting the provider. Its reservation
remains available for a retry with the exact same operation ID and request
after restoration; this does not create a second reservation. A retained,
completed result can still be replayed during the pause without a provider call.

Paid jobs under ``/caption-batches`` are sequential Responses workflows. The
separate asynchronous Batch workflow described below uses ``/paid-batches``.
Pilot results identify their scope as ``hosted_sequential_metered_responses``.
They certify only the measured recipe and frozen snapshot, never unmeasured
provider quality, live charges, throughput or GPU capacity. Local verification
uses a transport double with the real pipeline, authorization, artifact store
and spending ledger; it makes no external requests.

Asynchronous provider Batch jobs
--------------------------------

``POST /tools/paid-batches`` accepts either the current assigned image's exact
editor/version/generation context, or an administrator's confirmed snapshot
and optional image subset. The Remote Batch jobs dialog freezes and previews
that input before submission. It uses the existing combined caption-and-QA
Batch prompt builder and collector in ``run_openai_caption_batch_smoke.py``:
one structured response per image, including the configured QA requirements,
glossary, label hints, grounding rules and catch-up questions. This is a
different recipe from sequential window captions, merging and QA generation.
The preview displays the exact first-image prompt and reservation ceiling.

Only models with an explicit operator-supplied ``batch`` price and token-limit
record are offered. No discount or live price is guessed. Images are uploaded
to the fixed provider Files API using their shared encoded-coordinate working
view, followed by the frozen JSONL. New prepared rows freeze both original
source identity and working-view identity in the reservation fingerprint. Upload
audit records retain both hashes, and filenames match the actual image encoding.
An older unsent reservation keeps its original immutable request and fingerprint;
if its saved upload contains different bytes, only that definitely unsent row
replaces the image upload and invalidates its pre-create JSONL. Dispatched or
unknown submissions are never rewritten or resent. Batch creation
uses ``/v1/responses`` and the documented 24-hour completion window
(https://developers.openai.com/api/docs/guides/batch). Every image has a stable
ledger entry. One transaction checks every source version, current authority,
provider revision, cancellation flag and automation phase before marking those
entries dispatched. A heavy local job pauses future submission; an already
accepted remote Batch continues and may incur usage.
The eight-pending-job limit includes remote running jobs and unknown submissions;
archiving a running job does not release that slot. Prompt preparation spills to
disk and checks the 200 MiB JSONL bound before reserving budget or uploading.
Submission reads retained rows in bounded pages and streams the frozen JSONL
artifact. Per-image reservation checks and the final whole-batch dispatch check
avoid revalidating the entire population for each individual upload.

The controller retains the exact JSONL, remote status, raw output/error files,
parsed captions, QA and reported usage. It rejects unexpected or duplicate
output IDs before the native collector runs. Each row settles only from actual
reported usage; missing usage remains unresolved. A lost Batch-creation reply
becomes ``submission_unknown`` and is never resubmitted automatically. Recovery
requires the provider Batch ID to match the recorded input file and job metadata.
Restart recovery also finishes collection after provider completion, including
the interval before local results and usage were committed.

Outputs remain proposals. ``/captions`` pages them; ``/verify`` checks captured
image versions and classes; ``/apply`` requires the normal current editor and
proposal digest. Applying preserves objects and previous captions. Archive and
restore change list visibility without deleting data. Backup restoration checks
the frozen sources, billing links and every retained artifact reference.

Local evidence exercises the real prompt/collector, ledger, restart recovery,
backup closure and browser preview-to-explicit-save flow with the fixed-host
transport replaced by a test double. It does not qualify live provider limits,
quality, latency, charging or remote service behavior.

Applying a retained proposal
---------------------------

``GET /{id}/images/{image_id}/proposal`` returns the caption, optional generated
QA, original annotation version, evidence digest and quality findings.
An administrator explicitly accepts it with ``POST .../apply`` and the normal
editor/version/generation context plus ``proposal_sha256``. The transaction
keeps objects, preserves existing ``alternate_captions`` and records provenance;
replacing the primary caption also preserves its previous text as an alternate.
The image must still be confirmed at the captured version and class-map version.
Accepting another image's proposal may advance the dataset revision without
invalidating this image. Exact command retries recover their original receipt.

Admin compute priority
----------------------

Native DQE and pruning admissions pin SALAD heads attached to the captured
dataset snapshot. Pruning inherits its saved scene/object recipes as defaults;
explicit request fields take precedence. Cache aliases may merge only when
their content hashes and byte counts match. A conflicting same-name head is an
admission error, never an implicit replacement.

Native ``dqe_review`` uses a retained, current DQE parent and its exact snapshot.
The parent session artifacts and the administrator-configured registered Qwen
package are pinned and copied to the isolated worker. The browser supplies a
parent analysis ID and point ID, never a model directory. Capture history keeps
the parent job and registered model identity even if review fails. The original
native VLM review entrypoint runs in the confined child; no deterministic
substitute declares a review successful. Parent freshness and administrator
authority are checked at admission, dispatch and result retrieval.

``begin_heavy(context, spec, operation, payload, drain_timeout=30)`` opens one
exclusive window:

1. ``pausing`` closes new automation admission and emits a pause event.
2. ``draining`` lets active requests finish; queued requests stay suspended.
   At the deadline, remaining active requests are cancelled and their workers
   terminated. Heavy computation never starts while ordinary work remains active.
3. ``unloading`` stops ordinary workers, then ``heavy`` runs the job in a separate
   child process.
4. ``restoring`` reloads owners that were loaded or loading at the pause boundary
   and are still marked active, then returns
   to ``available``. Queued work is authorized again before dispatch.

An owner already unloaded by the idle timeout stays unloaded. Its next request
loads it normally; a heavy job does not resurrect every historical user/model.
Restoration initializes the ordinary SAM/classifier adapter. Optional Qwen
models and neighbor SAM slots remain lazy, so ``ready`` is adapter readiness,
not proof that every optional model is resident in GPU memory.

Success, model failure and cancellation all enter restoration. Cancelling a
heavy job before unloading leaves the original workers intact. An individual
reload failure is visible on that owner and retried on its next request. If a
worker cannot be stopped or an active request cannot drain, automation remains
failed instead of running heavy and ordinary work together. Manual editing and
annotation storage must remain outside this supervisor throughout.

``wait()`` waits for a request's computation, not for the entire restore phase.
Show the global phase until restoration finishes. A bounded event callback
provides best-effort notifications; missed events must be reconciled from status.
The integrating event stream must filter private actor/owner details. Notification
delivery is not a prerequisite for the pause barrier or a proof of UI acceptance.

Bounds, unloading and restart
----------------------------

Defaults are 32 queued requests per owner, 128 total pending requests, 128
registered owners, 1,024 retained request records, 128 MiB of queued payloads and
64 MiB of retained results. One protocol frame is at most 64 MiB. Old terminal
records and result bodies can be evicted; clients must handle
``runtime_result_not_retained`` by requesting fresh authorized work. Oversized
results fail explicitly. These are host-memory bounds, not a GPU admission
policy or a claim that 128 model instances fit in memory.

Idle workers unload after 15 minutes by default. The next request reloads them.
Cancelling a running request or detecting a crash drops that owner's worker;
the next request receives a fresh instance. State from a dead SAM predictor is
never assumed to survive. Pending dataset requests are interleaved between
actors while preserving each actor's order.

An atomic, fsynced journal records request identities and terminal states, without
pixels, payloads, model configuration or results. On restart, unfinished requests
become ``interrupted`` and are never replayed. Even previously completed results
must be recomputed if their bodies were lost. Register active owner configurations
again from the canonical server settings after restart.

A journal write failure closes automation admission and kills active workers;
manual storage remains independent. Repair the storage problem before recreating
the supervisor. If shutdown cannot join an outstanding callback, the exclusive
lock stays held and ``runtime_shutdown_incomplete`` is raised, preventing a
second manager from competing with it.

Evidence and remaining acceptance
---------------------------------

Installed SAM3 recipes and private presets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An administrator installs a ``sam3_recipe`` package with its recipe file,
retained crops and classifier artifacts, then attaches exact model identities
to a dataset. Its SAM3 and optional extra classifier are installed components.
The existing first-pane cascade controls now read this registry and dispatch
``sam3_recipe_chain`` to the user's private worker. The restored executor is
described in ``hosted_sam3_chain.rst``; the old per-recipe executor was incomplete,
so numerical GPU equivalence has not been established.

Cascade presets are scoped to an account and dataset, with revision checks and
idempotent save/delete receipts. Downloaded ZIPs contain settings and installed
recipe identities. Import requires those same attached versions; it cannot
install model code or weights. Existing local full-model ZIP packages must be
installed by an administrator before a hosted preset can reference them.

Qwen requests inherit dataset ``context`` and ``labelmap_glossary`` only when
the corresponding request field is absent. Explicit empty values override the
defaults. Current-image jobs freeze these fields in their durable input; batch
jobs use the confirmed snapshot's settings. Retries and reconnects reuse that
input even after guidance changes. The caption controls initialize untouched
fields with the guidance and preserve subsequent edits. Saved EDR recipes keep
their own glossary and prompts.

Native ingestion receives two separate immutable inputs: the confirmed dataset
snapshot and an explicitly selected set of unassigned Incoming images. Candidate
media and annotation versions are captured before compute and pinned for the
worker; analysis history retains the candidate-set identity. Assigned images
are rejected. Ingestion does not approve candidates or save annotations.
Its selected whole-image SALAD head must belong to the frozen dataset settings,
with matching retained bytes. Native SALAD training uses the confirmed snapshot
and does not consume Incoming candidates.

Human review of saved captions and QA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Administrators can open an image and choose ``Review captions and QA``. The
dialog shows the saved primary/alternate captions and each question and answer,
including migrated ``captions`` records. Each entry can be accepted, rejected,
marked for correction, or returned to unreviewed, with an optional note.
Proposal dialogs also disclose generated QA before it is saved.

Review uses the same editor lease, image version, generation and durable command
receipt as annotation saves. The server records the reviewer and time; the
original model output, validation, quality, provider and provenance fields remain
in the annotation history. A content hash binds the decision to the exact
question and answer. Editing that text makes the decision ineffective until it
is reviewed again. A subsequent box edit preserves both the language annotations
and their evidence.

Ordinary annotator saves cannot create or alter human review stamps. Existing
digest-bound decisions can remain as historical evidence after a text edit;
older imported decisions without that binding require administrator editing to
change the reviewed wording. The protection applies to both imported captions
and current alternates, including QA stored in their metadata.

An annotator also cannot qualify new QA by supplying ``machine_validated`` in a
save request. The stored caption's ``backend_job_id`` must identify a completed
server job for that same account, image, annotation version and editor generation.
The question, answer, verifier evidence, quality report and retained result
reference must match its durable output. Verified server-side proposal application
uses the exact retained proposal instead. Ordinary later saves preserve this
evidence. A text correction can be saved, but a content digest prevents the old
machine qualification from applying to the new wording. The native exporter then
checks it again and requires human review where appropriate.

The first-pane ``Caption image + Q&A`` action carries the original QA, quality,
job identity and evidence through its existing canvas save path. Local browser
coverage verifies a subsequent box edit and page reload preserve those fields;
the model output in that test is a fixture, not a measurement of model quality.

Create a new confirmed snapshot to export newly reviewed data. Existing snapshots
remain frozen. Human acceptance does not waive the native exporter's machine
validation or corpus checks: for example, its class-coverage requirement can
still require structured metadata QA. The local integration check exercises the
actual CPU exporter before and after explicit review, including changed-text
invalidation; it does not qualify generated language from real model weights.

``tests/test_hosted_runtime.py`` uses real child processes and the production
protocol/scheduler/journal with a small non-ML adapter. It covers user and dataset
isolation, serialization, queue limits, immutable inputs, authority changes,
admin pause/drain/restore, failure/cancellation, idle unload, configuration
changes, controller death, restart recovery and journal failure. It does not
prove model extraction, annotation saves during GPU work, tool parity, VRAM fit,
latency or throughput.

The model adapters and canonical API now have focused CPU composition checks,
including the real native statistics child and durable publication. These checks
use model doubles for neural inference and do not qualify real model weights.
Before hosted rollout, measure two annotators plus an admin using the complete
first-pane workflow on the target dual-A6000 machine. Verify image navigation, preloads, SAM prompting,
class prediction, detector pipelines, Qwen/caption paths, stale-result rejection,
manual saves during a heavy window and recovery after failure. Record actual GPU
peak memory and latency before choosing device placement or admission limits.
