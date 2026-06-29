# Qwen Caption Training Dataset Complete Partner Packet

Date: 2026-06-29

## Purpose

This document is the complete partner-facing explanation of the caption-based
VLM training-dataset work in this repository. It explains what was built, why it
was built this way, which data contracts matter, how an operator uses the UI,
how generated rows are validated and reviewed, what was hardened after failure
testing, and what still needs real-data pilot validation before a generated
corpus should be used for fine-tuning.

The implementation is intentionally dataset-neutral. It does not rely on
project-specific dataset names, class names, or private evaluation assumptions.
It is also written as a neutral external-consumer packet: do not add customer,
partner, dataset, or project code names to this document before sharing it.
Those names are not needed to review the implementation, and keeping the packet
neutral prevents product-specific assumptions from leaking into reusable
training-data contracts.

## Reader Entry Point

Use this document as the main external review packet. It is intended to answer
six questions without requiring the reviewer to reconstruct the development
history from commits:

1. What user-facing workflow exists now?
2. What training artifacts does the workflow produce?
3. Which parts of the artifact are trusted source evidence, generated language,
   deterministic code output, or selected trainer rows?
4. Which validation gates prevent stale, malformed, review-pending, or
   unsupported rows from entering a fine-tuning file?
5. Which runtime and UI hardening keeps long caption jobs from producing stale
   or unstable exports?
6. What remains to be proven with real data before using a generated corpus for
   actual training?

Recommended reading order for a technical review:

1. Read `docs/qwen_caption_training_dataset_complete_external_handoff.md` for
   the concise complete handoff: what was built, why, how to operate it, what
   is validated, and what still needs pilot proof.
2. Read **One-Page Decision Summary** to understand the current readiness claim.
3. Read **What Was Done And Why** and **Layer-By-Layer Implementation
   Narrative** to understand the design intent.
4. Read **Requirement Mapping** to verify that the product behavior requested
   by the multi-prompt training-data workflow has an implemented artifact,
   control, or guard.
5. Read **Artifact Contract** and **How To Inspect A Generated Packet** before
   reviewing sample exports.
6. Run **Reproducible Verification Commands** against the checked-out
   repository.
7. Use **Required Pilot Before Training Use** and **Open Decisions For The
   Review Team** to decide whether the implementation is ready to run on a
   real target corpus.

The companion documents are supporting references:

- `docs/qwen_caption_training_dataset_complete_external_handoff.md` is the
  concise complete external handoff for what was built, why, how it is
  validated, and what pilot proof remains.
- `docs/qwen_caption_training_dataset_reviewer_dossier.md` provides a
  self-contained external-review dossier covering what was built, why it was
  built, how to operate it, what is validated, and what remains to be proven in
  a pilot.
- `docs/qwen_caption_training_dataset_external_implementation_report.md`
  provides the external narrative of what was built, why each major design
  choice was made, and how the workflow should be reviewed.
- `docs/qwen_caption_ui_scenarios.md` describes UI behavior and operator
  scenarios.
- `docs/qwen_caption_prompt_stack.md` describes prompt construction, token
  budgets, dense-box handling, and recovery wording.
- `docs/qwen_caption_instruction_dataset_hardening_report.md` records
  implementation-hardening details.
- `docs/qwen_caption_training_dataset_external_review_handoff.md` is a shorter
  reviewer checklist.
- `docs/qwen_caption_instruction_dataset_external_partner_packet.md` remains a
  supporting implementation packet; this file is the canonical overview.

## External-Consumer Contract

This packet is meant to let an external technical reviewer independently answer
four questions:

- **Can the UI start the right workflow?** The expected workflow is a distinct
  **Create VLM training dataset** action, not a hidden reuse of ordinary
  caption export.
- **Can every trainer row be audited?** A trainable row must trace back to an
  image, a question, an answer, source evidence or generated-language
  provenance, validation state, review state, and a run report.
- **Can unsafe rows be kept for review without entering training?** Rejected,
  stale, unsupported, review-pending, or malformed candidates must remain in
  audit/review artifacts but be excluded from flat trainer JSONL.
- **Can long-running generation be operated without babysitting?** Dataset
  jobs must expose progress, failure states, attach/recover behavior, model
  availability, prompt-budget telemetry, loop recovery, and busy-state guards
  around exports and archive mutations.

The packet should not be read as a promise that any generated corpus is ready
for fine-tuning. It documents the implemented workflow and hardening. A corpus
becomes training-ready only after a real-data pilot, human review where
required, reviewed-row import, strict re-export, and trainer-loader or
fine-tuning smoke validation.

## What To Hand Over

For external review, provide this file plus the following supporting files:

- `docs/qwen_caption_training_dataset_complete_external_handoff.md`
- `docs/qwen_caption_training_dataset_external_implementation_report.md`
- `docs/qwen_caption_training_dataset_reviewer_dossier.md`
- `docs/qwen_caption_instruction_dataset_hardening_report.md`
- `docs/qwen_caption_prompt_stack.md`
- `docs/qwen_caption_ui_scenarios.md`

If sample artifacts are shared with the documentation, include the complete
artifact set from the same run:

- trainer JSONL
- instruction archive JSONL
- review JSONL
- instruction report JSON
- the run's caption/progress trace summary when available

Do not share trainer JSONL alone as proof of correctness. The trainer file is
the model-input artifact; the archive, review file, and report are the evidence
that explain whether those model-input rows should be trusted.

## One-Page Decision Summary

The current implementation is ready for external implementation review and a
small real-data pilot. It is not presented as a certified production corpus
generator until that pilot has been run, reviewed, re-exported, and loaded by
the trainer.

What exists now:

- A UI button, **Create VLM training dataset**, that launches dataset-scale
  caption-derived instruction generation.
- One broad image-caption row per image when `caption0` is enabled.
- Configurable VLM-generated visual question/answer candidates per image.
- Optional deterministic metadata QA derived only from source labels.
- Set-and-forget backend execution with progress, attach/recover behavior, loop
  recovery, model-availability feedback, and safer full-image composition
  defaults for long jobs.
- Four export artifacts: flat trainer JSONL, per-image archive JSONL,
  candidate-level review JSONL, and report JSON.
- Browser-side, backend-side, and trainer-side validation gates.
- Review JSONL import that applies review metadata only.

Why it was built this way:

- A single caption per image is too weak for VLM fine-tuning.
- Generated language is useful but cannot be allowed to become source truth.
- Training rows must be flat and model-visible, while audit artifacts must be
  richer and provenance-preserving.
- Long caption jobs can mutate caption archives while an operator is exporting;
  export actions therefore need busy-state guards and action-time checks.
- MLX/VLM generation can loop or fault; set-and-forget mode needs streaming
  loop inspection, bounded retries, fallback policy, and deterministic
  recovery only as an explicitly degraded last resort.

What still must be proven:

- Generated QA quality on the real target dataset.
- Manual review usefulness and reviewer throughput.
- Trainer import plus at least one fine-tuning or loader-plus-batch dry run on
  a reviewed pilot export.
- Acceptable rate of degraded recovery captions in unattended runs.

## Handoff Scope

The completed scope is an end-to-end application path for producing auditable
caption-derived VLM instruction datasets:

- UI controls to launch caption-derived instruction-dataset generation.
- Dataset-scale set-and-forget backend execution.
- Prompt/runtime hardening for dense labels, large prompts, thinking-capable
  models, repeated-token loops, and model availability.
- Multi-row output per image: `caption0`, generated visual QA, and optional
  deterministic label-derived QA.
- Separate trainer, archive, review, and report artifacts.
- Browser validation, server validation, and trainer-import validation.
- Review JSONL download and reviewed JSONL import.
- API/script strict export mode for trainer-ready exports.

The completed scope does not include certifying a specific generated corpus as
ready for production fine-tuning. That still requires a real-data pilot,
manual generated-QA review, reviewed-row import, re-export, trainer import, and
at least a small fine-tuning or loader-plus-batch dry run.

## Product Contract

The product contract is conservative because the output can become training
data:

- Generated language must never become source-label truth.
- Trusted label-derived facts must be traceable to source annotations or
  deterministic code.
- Trainer JSONL must contain only model-visible `image_path`, `question`, and
  `answer` rows, with optional metadata for audit.
- Archive and review artifacts must preserve enough evidence to explain why a
  row was selected or rejected.
- Human review must apply metadata decisions only; it must not silently edit
  labels, boxes, generated questions, generated answers, or final annotations.
- Browser, backend, and trainer checks must each fail closed when artifacts are
  stale, malformed, inconsistent, or not ready.

## Executive Summary

The captioning stack now has two separate product paths:

- **Captioning for annotation assistance**: generate, store, promote, and export
  caption variants for images.
- **Instruction-dataset creation for VLM training**: create one broad caption
  row plus optional generated visual question/answer rows and optional
  deterministic label-derived question/answer rows, then export flat trainer
  JSONL with audit artifacts.

The instruction-dataset path is exposed in the caption panel as
**Create VLM training dataset**. It can be run as a set-and-forget backend job
over a selected dataset. The run produces:

- `instruction_training_rows`: flat `image_path` / `question` / `answer` rows
  that the Qwen trainer can import directly.
- `instruction_archive_rows`: one per-image construction archive JSONL row
  preserving provenance, source labels, captions, generated QA candidates,
  deterministic metadata QA, selected training rows, and rejected rows.
- `instruction_review_rows`: candidate-level review JSONL rows for human
  decisions.
- `instruction_report`: run-level quality, readiness, consistency, and
  provenance metrics.

The main design choice is strict separation of trusted source data from
generated language:

- Source labels are trusted structured evidence.
- Generated captions and generated QA are language candidates.
- Deterministic metadata QA is code-generated from trusted labels.
- Flat trainer rows are selected exports, not the source of truth.

Generated language never becomes source-label truth. If generated language makes
a count, class-list, presence, absence, or simple spatial claim, the exporter
either rewrites the final training answer from trusted source annotations or
rejects the row from flattened training output. Rejected rows remain visible in
the archive and review JSONL.

## What Was Done And Why

The work was not just a new export button. It changed the caption stack into a
reviewable training-data production path while preserving the older
caption-assistance path.

| Problem | Change implemented | Why it matters |
| --- | --- | --- |
| Caption-only JSONL is too thin for VLM fine-tuning | Added instruction-dataset mode with `caption0`, generated visual QA, and optional deterministic metadata QA | Fine-tuning can see multiple grounded prompts about the same image instead of one scene description |
| Generated prose could be mistaken for trusted labels | Split `source_annotations`, `language_annotations`, deterministic QA, and flattened trainer rows | Reviewers can always tell whether a fact came from labels, code, or model language |
| Generated counts/classes can hallucinate | Structured generated claims are rewritten from source annotations when supported or rejected from trainer rows when unsupported | The final training answer for label-derived facts is reproducible from trusted evidence |
| A flat trainer file cannot explain itself | Added archive JSONL, review JSONL, report JSON, and embedded consistency proofs | Every trainable row can be traced back to its source image, source evidence, candidate, validation state, and review state |
| Review was not a closed loop | Added review JSONL download and reviewed JSONL import | Human decisions can be applied without editing boxes, labels, questions, answers, or final annotations |
| Hand-edited or stale files could be trained accidentally | Added browser, backend, report, archive, review-row, and trainer-import validation | A structurally valid but inconsistent artifact fails before use |
| Dataset-scale captioning was fragile | Made set-and-forget backend jobs the durable path with heartbeat, resume, attach/recover, and failure telemetry | Long jobs no longer depend on a browser tab staying alive |
| Dense label prompts could become huge | Kept counts authoritative but capped box lists to representative spatial subsets | The model gets useful spatial context without prompt blowups; omitted boxes are not treated as absent objects |
| Thinking-capable models need larger generation budgets | Split automatic output-token budgeting from explicit numeric overrides | Defaults can be high enough for reasoning-heavy models while user caps remain hard caps |
| Repeated-token loops could appear as hangs | Added live output-loop detection, safe retry, runtime unload, and fallback/recovery paths | Repeated punctuation or token streams are not accepted as valid captions |
| Missing model files were visually ambiguous | Styled download-needed model choices in red and local models in the normal local color | Operators can see model availability before launching a long job |
| Review import payloads were accepted inconsistently by UI, route, and backend parser | Made the API route accept any JSON body and moved body-shape enforcement into the shared backend review parser | JSON arrays, wrapper objects, and single review-row objects now follow the same fail-closed validation path instead of being rejected by framework typing before parser checks |
| Scripted trainer exports could rely too heavily on the readiness label | The server-side ready gate now independently requires a valid ready report, export-validation proof, versioned artifact-consistency proof, corpus count agreement, and returned artifact arrays that match the proofs and row-shape contract | API clients get the same critical fail-closed behavior as the browser when readiness labels, proof state, corpus metrics, or returned rows drift |

The result is a conservative pipeline: generate candidates, archive evidence,
validate aggressively, let humans review generated language, then export only
trainable rows.

## Reviewable Claims

These are the concrete claims this packet makes about the current
implementation.

| Claim | Where to verify |
| --- | --- |
| The UI exposes instruction-dataset creation as a distinct workflow, not as a hidden caption export | `ybat-master/ybat.js`, `ybat-master/ybat.css`, `tests/test_labeling_panel_layout_contract.py` |
| The operator can create one broad caption row plus configurable generated visual QA rows per image | `models/schemas.py`, `localinferenceapi.py`, UI scenario tests |
| Deterministic metadata QA is optional and generated from source labels, not from the VLM | `localinferenceapi.py`, `tests/test_qwen_caption_dataset_job.py` |
| Empty label files are treated as empty source evidence, not as a prompt to ask for nonexistent label facts | `localinferenceapi.py`, source-annotation tests and docs |
| Generated language rows are archived even when they are rejected from trainer output | archive construction tests and instruction report counts |
| Review JSONL import applies review metadata only | `localinferenceapi.py`, `api/datasets.py`, review-import tests |
| Trainer JSONL, archive JSONL, review JSONL, and report JSON are validated as one artifact set | browser validators, backend validators, artifact-consistency tests |
| API clients can require trainer-ready strict export with `require_ready_instruction_export=true` | `api/datasets.py`, strict export tests |
| The trainer can import the flat instruction rows directly and reject non-trainable rows | `tools/qwen_training.py`, `tests/test_qwen_training_backend.py` |
| Long dataset jobs use set-and-forget assumptions and expose recovery/health state | `localinferenceapi.py`, caption job and supervision tests |

These claims are structural and workflow claims. They do not claim that the
content quality of generated QA is production-ready on any particular dataset.
Content quality is intentionally deferred to a pilot and human review.

## Development Timeline In Plain Terms

The hardening work happened in four broad phases:

1. **Build the new product path**: add `caption0` plus generated QA generation,
   instruction settings, archive construction, trainer JSONL export, and
   report generation.
2. **Close the review loop**: add review JSONL export/import, stable QA ids,
   dataset identity checks, stale-text checks, duplicate-target checks, and
   metadata-only mutation.
3. **Harden runtime and prompts**: make set-and-forget the durable path, clarify
   output-token budgeting, cap dense box prompts with representative subsets,
   add stream-loop detection and recovery, and make missing model availability
   visible.
4. **Fail closed at export and trainer boundaries**: validate browser downloads,
   validate API strict exports, check proof objects and row shapes, and make the
   trainer reject malformed, stale, unknown-status, or non-trainable rows.

The reason for this order is that a training-data feature is only useful when
the generation path, audit artifacts, review loop, export gate, and trainer
import boundary all agree. Producing a JSONL file alone is not enough.

## Layer-By-Layer Implementation Narrative

This section summarizes what changed in each layer of the system and why each
change exists. It is the fastest way for an external review team to understand
the implementation before reading the lower-level contracts.

### 1. The UI Now Has A Real Instruction-Dataset Product Path

The caption panel no longer treats training-data creation as an incidental
caption export. It exposes a distinct **Create VLM training dataset** action
with controls for generated QA count, QA mix, answer format, caption0 inclusion,
generated-QA inclusion, deterministic metadata QA, read-only source-label
context, strict grounding, and ready-report enforcement.

The reason for this split is product clarity. A normal captioning run answers
"describe this image." A training-dataset run answers "construct an auditable
set of image/question/answer rows and prove which ones are safe to train on."
Those workflows share model infrastructure, but they have different artifacts,
failure modes, and review requirements.

### 2. Dataset Jobs Were Moved Toward Set-And-Forget Operation

Dataset-scale captioning and instruction generation now use persisted backend
jobs by default. The operator can start a run, monitor progress, attach or
recover from the UI, and inspect health/status messages when the backend cannot
launch or when generation recovery is triggered.

This was necessary because browser-bound captioning is too fragile for long
runs. A useful training corpus may require many images, repeated model calls,
windowed observations, full-image composition, archive construction, and export
validation. The durable path has to survive browser refreshes, model loops,
runtime unloads, process restarts, and partial progress.

### 3. Prompt Context Was Bounded Without Losing Authoritative Counts

Dense scenes can contain enough boxes to make a full prompt unstable. The new
prompt policy keeps full class counts authoritative while reducing large box
lists to representative spatial subsets. Prompt wording explains that subset
boxes are context examples, not a complete object inventory. Prompt-size
telemetry then estimates rendered prompt pressure and adapts automatic output
budgets when the prompt itself is already large.

This was added because the model needs spatial examples, but a giant serialized
box list can dominate the prompt, slow the full-image stage, and increase loop
or GPU-risk behavior. Counts remain complete because counts are structured
source evidence; detailed boxes are supporting context.

### 4. Output Budgets Now Separate Defaults From User Caps

Thinking-capable or verbose models may need much larger automatic generation
budgets than a short caption model. The UI and backend now distinguish automatic
model-aware defaults from explicit numeric user overrides. If the user sets a
number, that number is the hard cap after schema validation. If Auto is used,
the system can choose a larger default for models that need it and can reduce
that budget when prompt-size pressure is high.

This solves the confusing mismatch between UI values, rendered prompt logs, and
runtime `max_new_tokens`. The operator should be able to reconcile what they
set with what the backend actually used.

### 5. Model Output Is Inspected While It Streams

The caption runtime now treats repeated punctuation or repeated-token output as
a recoverable model failure, not as a long caption. The stream inspector can
detect loops, trim the repeated fragment for diagnostics, unload the runtime
when needed, and retry through safer decoding or fallback paths. Recovery output
is logged as recovery, not as a normal caption.

This matters because repeated output can look like a stall while the process is
still serving progress endpoints. Without live inspection, the backend may seem
healthy while the model is spending the full token budget on unusable output.

### 6. Training Artifacts Are Split By Responsibility

The instruction-dataset export produces four different artifacts rather than
one overloaded JSONL:

- trainer JSONL for fine-tuning input;
- archive JSONL for per-image construction evidence;
- review JSONL for candidate-level human decisions;
- report JSON for run-level readiness, consistency, and quality metrics.

This split is deliberate. Trainer JSONL should be small, flat, and directly
loadable. Archive JSONL should be rich enough to audit. Review JSONL should be
easy to hand to a reviewer. Report JSON should explain whether the corpus is
ready, needs review, or is blocked.

### 7. Trusted Source Data And Generated Language Stay Separate

The archive separates `source_annotations`, `language_annotations`,
`deterministic_metadata_qa_pairs`, and `flattened_training_rows`. Source
annotations come from labels and deterministic geometry. Caption0 and generated
QA are language candidates. Deterministic metadata QA is code-generated from
trusted labels. Flattened rows are selected training rows, not source truth.

This boundary is the core training-data safety rule. A model-generated sentence
may be useful, but it must not become a source count, source class list, source
box, or final annotation. When a generated answer makes a source-checkable
structured claim, the exporter either rewrites the final answer from trusted
labels or rejects the row from flattened trainer output.

### 8. Review Import Is Metadata-Only And Fail-Closed

Review JSONL import applies reviewer decisions and notes to saved caption0 and
generated-QA records. It does not edit image paths, labels, boxes, questions,
answers, deterministic QA, or final annotations. The parser accepts practical
input shapes, including JSONL-derived arrays and wrapper objects, but it rejects
wrong datasets, stale text, unsupported row origins, duplicate targets,
oversized fields, scalar bodies, and packets with no actionable decisions.

This preserves the human-control contract. A reviewer can decide whether a
generated language candidate is trainable without accidentally changing the
source data or rewriting generated content through an import side channel.

### 9. Trainer JSONL Export Is Gated In Browser And API Paths

The browser validates trainer JSONL before download and treats missing or
non-array artifact row payloads as blocked validation results rather than
runtime exceptions. The API also supports
`require_ready_instruction_export=true`, which independently checks readiness,
report shape, ready flags, corpus metrics, export-validation proof, versioned
artifact-consistency proof, and the returned artifact arrays. Trainer, archive,
and review rows must be lists, and their lengths must agree with report,
readiness, corpus-metric, and consistency counts when those counts are present.
The returned rows must also satisfy the row-shape contract: trainer rows need
required image/question/answer fields, instruction metadata, trainable
validation/review state, supported source archive, valid JSON when requested,
and unique canonical image/question identities; archive rows need source,
language, deterministic-QA, and export metadata objects; review rows need the
review schema, persisted identity fields, source summaries, review-decision
fields, selected/training-answer consistency, and unique image/QA identities.

This was added because the browser is not the only export client. Scripts and
future automation must receive the same fail-closed behavior as the UI. A
payload with a `ready` label but invalid report format, false
`ready_for_training`, hidden blocking reasons, required actions, quality
warnings, stale corpus metrics, malformed returned rows, or mismatched
proof/artifact arrays is not allowed to produce trainer JSONL under the strict
gate.

### 10. The Trainer Loader Is The Last Safety Boundary

The Qwen trainer imports flat image/question/answer rows directly, preserves
metadata, resolves image paths, and rejects non-trainable rows. It fails on
missing instruction provenance, unknown validation or review states, rejected or
needs-revision review state, invalid JSON answers for JSON-formatted rows,
duplicate canonical image/question pairs, and unresolved image aliases.

This protects actual fine-tuning from stale files, hand-edited JSONL, or
artifacts that bypassed browser/API validation. Export validation should catch
bad rows earlier, but the loader still needs to be defensive because it is the
last boundary before training.

### 11. The Documentation And Tests Track The Product Contract

The implementation is accompanied by prompt-stack docs, UI scenario docs,
hardening reports, partner-facing packets, and focused tests for the review
loop, export gates, trainer import, prompt behavior, runtime supervision, and
UI smoke behavior. The docs intentionally explain both what is implemented and
what is not yet certified.

The reason is operational: this path is meant to produce training data, so a
reviewer needs more than a passing unit test. They need to see the intended data
contract, the exact artifact shapes, the review workflow, the failure policy,
and the pilot work still required before a generated corpus is treated as
training-grade.

## Requirement Mapping

The implementation was built to satisfy the multi-prompt training-data contract
below.

| Required product | Implemented artifact or control | Status |
| --- | --- | --- |
| One broad global caption per image | `caption0` language annotation and optional flat training row | Implemented |
| Zero or more generated visual QA candidates per image | **Generated QA per image**, persisted generated-QA records, `instruction_review_rows` | Implemented, bounded `0..20` |
| Optional deterministic QA derived from real labels | **Include deterministic metadata QA**, `deterministic_metadata_qa_pairs` | Implemented, off by default |
| Per-image construction archive | `instruction_archive_rows` and versioned `instruction_archive` | Implemented |
| Flat trainer JSONL | `instruction_training_rows` with `image_path`, `question`, `answer`, optional metadata | Implemented |
| Source/generated separation | `source_annotations`, `language_annotations`, `deterministic_metadata_qa_pairs`, `flattened_training_rows` | Implemented |
| Human review loop | **Download review JSONL** and **Import reviewed JSONL** | Implemented |
| Trainer import compatibility | Qwen training loader imports flat rows directly | Implemented |
| Set-and-forget operation | persisted backend jobs, recovery controls, progress mirroring, crash-supervision checks | Implemented |
| Dense prompt safety | prompt budget accounting and representative box subsets | Implemented |
| Repetition or loop safety | streaming loop inspector, controlled retry, fallback, deterministic recovery | Implemented |
| Model-download clarity | model dropdown colors missing/download-needed models red and local models normal | Implemented |
| Safe artifact actions during long jobs | UI disabling plus action-time checks for ordinary caption exports, full dataset ZIP downloads, instruction exports, report downloads, and reviewed JSONL import | Implemented |
| Safe caption mutations during long jobs | Caption output editing, autosave, manual caption add/update/primary/delete controls, and alternate selection visibly disable or refuse during active archive mutation; text-label saves plus caption add/update/delete also refuse while the selected dataset has an active caption job | Implemented |
| Safe prompt metadata during long jobs | Prompt-stack, style, glossary, model, token, decode, set-and-forget, pilot, health-gate, save/promote, and batch-scope controls visibly lock while the selected archive is mutating; glossary reset/save and stale input events also hit archive-idle guards, and backend glossary saves refuse active caption jobs | Implemented |
| Safe dataset deletion during long jobs | Dataset deletion refuses while an active caption dataset job references the same dataset | Implemented |
| Same-dataset job concurrency | Caption dataset job start refuses while another queued, running, or cancelling caption job owns the same dataset | Implemented |
| Annotation-lock launch preflight | Caption dataset job start refuses synchronously when a write-owning job would violate an active annotation lock | Implemented |
| Script/API parity with browser export gates | Caption job start, caption export, review-import, and direct caption mutation paths block active caption jobs; caption export also supports `require_ready_instruction_export=true` strict trainer readiness | Implemented |
| Backend launch failure visibility | caption job creation failures are surfaced in caption status, backend-job status, and UI health text | Implemented |

## Latest Hardening Checkpoint

The most recent work focused on making the instruction-dataset path durable
enough for a real operator, not only a developer running one happy-path export.

### Caption And Instruction Artifact Actions Are Frozen While Captions Mutate

Ordinary caption exports, full dataset ZIP downloads, instruction artifact
downloads, instruction report downloads, and reviewed JSONL import are now
disabled until both conditions are true:

- a caption dataset is selected;
- no caption job or instruction-dataset job is actively mutating the selected
  dataset's caption archive.

The ordinary caption exports covered by this guard are caption audit JSONL,
grouped caption JSON, and caption-only VLM JSONL. The full dataset ZIP download
is covered because it can include overlay files and other dataset metadata. The
instruction artifacts covered by this guard are trainer JSONL, archive JSONL,
review JSONL, report JSON, and reviewed JSONL import.

The backend also refuses to start a second caption dataset job for a dataset
that already has a queued, running, or cancelling caption dataset job. The
check and registry insertion happen under the same job-registry lock so two
near-simultaneous API calls cannot both pass launch preflight for the same
dataset.

The HTTP caption-export route, dataset ZIP download route, reviewed JSONL
import route, text-label save path, caption add/update/delete paths, dataset
glossary save path, and dataset deletion path also use backend active-job
guards. A script or API caller that tries to export captions while an active
caption dataset job is registered for the same dataset receives a `409` with a
`caption_export_busy` detail instead of a partial archive snapshot. A script or
API caller that tries to download the full dataset ZIP during the same active
state receives `dataset_download_busy` before the backend reads the dataset or
overlay files. A script or API caller that tries to import review decisions
during the same active state receives `caption_review_import_busy` before the
backend reads the mutating caption archive. A script or API caller that tries
to save a text label or add, update, or delete caption records receives
`caption_mutation_busy` before the backend resolves the dataset. A script or UI
path that tries to save the dataset glossary while the run is active receives
`caption_metadata_busy` before the backend reads or writes metadata, preventing
mixed prompt semantics inside one job. A script, API caller, or operator that
tries to delete a dataset while an active caption dataset job references it
receives `dataset_delete_blocked_active_jobs` before the registry record or
managed dataset tree can be removed.

The dataset manager UI now downloads dataset ZIP files through an explicit
fetch-and-save flow rather than a fire-and-forget anchor click. That means
server-side busy responses and validation errors appear in the dataset status
message area instead of producing a confusing downloaded error body or browser
navigation.

Guard responses are also formatted for operators. Caption-job busy details,
metadata-busy details, active same-dataset job details, and annotation-lock
details are converted from internal `detail` strings into actionable UI text.
This prevents guarded failures from appearing as raw JSON in the dataset
manager, caption panel, direct caption/text-label save path, or review/import
surfaces.
Instruction artifact actions share one failure reporter so trainer JSONL,
archive, review JSONL, reviewed-row import, and report failures update the
caption export health row and the toast/status message consistently without
double-prefixing already formatted blocked messages.
Manual caption archive actions also report success only after the underlying
save, update, primary-selection, or delete operation returns a real mutation.
Stale clicks or scripted disabled-control bypasses therefore do not produce
false success messages for caption records that feed instruction exports. Those
same controls are visibly disabled while a caption or instruction job is
mutating the caption archive, and they only re-enable when the current image,
caption text, and selected-caption state make the action valid.
Failed backend caption jobs receive the same treatment when structured failure
reports are available. Pilot-certification, backend-supervision, and runner
preflight reports are summarized from their first failed check, so the operator
sees the failed gate and remediation detail instead of a raw runner error code.

The same busy check also runs inside each action handler. That second check
matters because UI state can go stale: a user may leave a file picker open, a
scripted click can bypass a disabled button, or a backend job can start after
the button was first rendered. In those cases the operation refuses to export
or import against a moving target and explains the reason in the caption
health/status surface.

The reason is simple: instruction JSONL, archive JSONL, review JSONL, and the
instruction report are one coherent export set. If a caption or generated-QA
record changes while one of those artifacts is being built or reviewed, the
flat trainer rows can drift away from the audit artifacts. The safe behavior is
to wait for the mutating job to finish, then export or import against a stable
caption archive.

Dataset deletion uses the same principle. A linked dataset delete removes the
registry record and overlay metadata while preserving the source image tree; a
managed dataset delete moves the dataset tree to trash. Either operation can
orphan a running caption dataset job or detach it from the archive it is
mutating. Active caption dataset jobs therefore participate in the generic
active-job deletion guard. Completed caption jobs do not block deletion.

Caption dataset job launch now reserves the dataset in the active-job registry
before the write-owning annotation-lock preflight. If a job will save text
labels or instruction records and the dataset has an active annotation lock,
the start route returns the same annotation-lock `409` that the worker would
have produced, then rolls back the reservation. A matching
`annotation_session_id` is accepted. Registering before preflight matters
because dataset deletion, export, download, and metadata-write guards can see
the queued job while preflight resolves metadata. The worker still repeats the
lock check after launch to close the race where a lock appears between
preflight and execution.

### Trainer JSONL Is Gated Twice

The browser validates trainer JSONL before writing a file. The backend also
supports the same strict policy through:

```text
require_ready_instruction_export=true
```

That server-side gate is required because the browser is not the only export
client. Scripts, API clients, or future automation can request exports without
using the UI. With the gate enabled, trainer JSONL export refuses reports whose
readiness is `blocked` or `needs_review`. Archive, review, and report artifacts
remain available because those are diagnostic and review artifacts; the flat
trainer JSONL is the artifact that can directly feed fine-tuning, so it is held
to the stricter default.

The server-side gate also checks the proof objects behind that readiness label.
A `ready` status string is not sufficient. The report must use the expected
instruction-report format, `ready_for_training` must be true, ready reports must
not carry blocking reasons, required actions, or quality warnings, and core
corpus metrics must agree with report counts. The gate also refuses export if
`instruction_export_validation` or the versioned
`instruction_artifact_consistency` proof is missing, wrong-version, not OK, has
nonzero errors, or disagrees between the API payload, report, and archive. This
prevents scripts from receiving trainer JSONL if a future bug or hand-edited
payload makes the readiness label drift away from the actual validation
evidence.

The same gate also checks the returned artifact arrays. Trainer, archive, and
review rows must be lists, and their lengths must match the selected-row,
image, review-row, manual-review, export-validation, readiness, corpus-metric,
and artifact-consistency counts when those counts are present. Selected review
rows must match selected trainer rows. The rows themselves must also preserve
the trainer metadata, archive schema, and review-row schema required by the UI
and trainer. This catches payload assembly bugs where valid proof objects are
accidentally paired with missing, malformed, stale, or wrong-length artifact
arrays.

### Backend Launch Failures Are Visible

Dataset-scale captioning can fail before a model produces text: a backend job
can fail to launch, a local runtime can be unavailable, or the process can abort
under the ML runtime. The UI now treats backend launch failure as an operator
state rather than a silent stall. Caption status, backend-job status, and health
text all surface that the job failed to start.

This is separate from model-output recovery. If the worker launches and the
model later loops, the output-loop inspector and recovery path handle it. If the
job cannot be created at all, the user needs a clear launch failure immediately.

### Review Import Remains Metadata-Only

The review loop intentionally imports decisions, not edited training content.
Accepted, rejected, and needs-revision decisions are applied to matching
caption0 and generated-QA records. The import path does not edit image paths,
source labels, boxes, generated questions, generated answers, deterministic QA,
or final annotations.

This keeps the human review loop auditable. A reviewer can decide whether a row
is trainable without silently rewriting the source of truth. Future workflows
could support reviewed replacement text, but that would need a separate explicit
contract with provenance for the replacement author and the original generated
candidate.

### Review Imports Are Bounded Data, Not Open Text Dumps

The review JSONL import path now enforces both shape and text-field limits in
the browser and backend. This is deliberately stricter than normal JSON parsing:
a syntactically valid review packet can still be unsafe if a reviewer tool
turns a text field into an object, writes extremely large notes, or produces a
row that cannot be matched back to a saved caption0 or generated-QA record.

The enforced per-field limits apply to the raw string value before trimming, so
an oversized all-whitespace field is still rejected:

| Field family | Limit |
| --- | --- |
| dataset id, QA id, row origin, split, validation status, review decision | 512 characters |
| image path, image name, image alias | 4096 characters |
| question | 4096 characters |
| candidate answer, selected training answer | 65536 characters |
| review notes | 8192 characters |

The review file still has an overall browser-side import cap and the backend
still has a row-count cap. The point is not to prevent legitimate review notes;
it is to keep a hand-edited or tool-generated review packet from persisting
unbounded text into caption metadata or causing an import path to spend time on
rows that will later fail identity checks.

Backend failures are formatted into row-specific UI messages. For example, a
long `review_notes` field reports the row number and the field limit instead of
surfacing a raw server code.

Review import accepts JSONL rows, a JSON array of rows, a wrapper object with
`rows`, `review_rows`, or `instruction_review_rows`, or a single review-row
object. Wrapper fields are resolved in that order, and an explicitly present
field must be a list. An empty `rows: []` field therefore means "no rows" and
does not fall through to another wrapper field. A wrapper object with no row
container is rejected instead of being treated as an empty review packet.

### Review Import Body Shapes Are Parser-Owned

The `/datasets/{dataset_id}/captions/instruction_review` route intentionally
accepts any JSON body and delegates validation to the instruction-review parser.
That is important because valid operator and script inputs are not all JSON
objects:

- browser-imported JSONL becomes a list of parsed review rows;
- API clients may send a JSON array of review rows;
- API clients may send a wrapper object with `rows`, `review_rows`, or
  `instruction_review_rows`;
- API clients may send a single review-row object carrying the review-row format
  marker;
- scalar values, malformed wrappers, row objects without the review-row format,
  and packets with no actionable persisted decisions fail in the backend parser.

The route no longer relies on a `dict` body type that would reject JSON arrays
before the shared parser could apply row-specific errors. Lock-owner metadata
and reviewer metadata are read only from wrapper objects; array and single-row
imports still apply the same dataset-id, stale-text, duplicate-target,
row-shape, text-limit, and metadata-only mutation rules.

## Non-Negotiable Training Shape

The trainer learns only from model-visible rows shaped like this:

```json
{"image_path":"images/example.jpg","question":"How many people are visible?","answer":"{\"object_counts\":{\"Person\":2}}"}
```

Optional `metadata` is allowed for audit, filtering, lineage, and metrics, but
it is not hidden prompt text. If a fact should teach the model something, that
fact must be written into `question` or `answer`.

This is why the system exports a flat trainer JSONL separately from the richer
archive. The archive preserves evidence. The trainer rows are the selected
training signal.

## Data Ownership Contract

The instruction archive keeps evidence types separate.

| Layer | Purpose | Can the VLM write it? | Direct trainer input? |
| --- | --- | --- | --- |
| `source_annotations` | Trusted structured facts from real labels and deterministic geometry | No | No |
| `language_annotations` | `caption0` and generated/reviewed QA candidates | Yes | No, unless flattened |
| `deterministic_metadata_qa_pairs` | Code-generated QA rows from trusted labels | Answers: no. Question wording: templated | No, unless flattened |
| `flattened_training_rows` | Final trainer rows | Exporter writes selected rows | Yes |

Allowed truth flow:

```text
labels / reviewed corrections / workflow signals
  -> source_annotations

source_annotations
  -> deterministic_metadata_qa_pairs

image + caption0 + read-only source_annotations + glossary
  -> language_annotations.generated_qa_pairs

validated language_annotations + deterministic_metadata_qa_pairs
  -> flattened_training_rows
```

Forbidden truth flow:

```text
caption0 -> source_annotations.object_counts
generated QA text -> source_annotations.visible_classes
generated scene summary -> source_annotations.spatial_facts
VLM uncertainty phrasing -> source_annotations.uncertainty
metadata -> assumed hidden model input
```

## End-To-End Data Flow

The implementation has one source-of-truth chain for each generated training
row:

1. The dataset manifest defines the image universe.
2. Existing labels and text labels are read without mutating the dataset.
3. Label files are converted into `source_annotations`, including counts,
   classes, instances, coarse geometry, spatial summaries, missing-source state,
   and empty-source state.
4. The caption job creates or reads `caption0`.
5. When instruction mode is enabled and generated QA is requested, the VLM
   generates additional question/answer candidates from the image plus read-only
   context.
6. Optional deterministic metadata QA is generated by code from
   `source_annotations`.
7. Candidate rows are validated and assigned stable identities.
8. Source-grounded claims are rewritten from source annotations when possible.
9. Unsupported or contradictory claims are rejected from flattened trainer
   output but retained in the archive and review JSONL.
10. The exporter writes:
    - flat trainer rows;
    - per-image archive rows;
    - candidate-level review rows;
    - a run-level instruction report.
11. Browser and backend consistency validators compare the artifacts against
    each other.
12. Reviewers can import decisions for caption0 and generated-QA rows.
13. The trainer imports only the final flat rows and performs its own last-line
    validation.

This flow is deliberately append-and-review oriented. It avoids treating a
generated answer as a label edit, and it avoids relying on hidden metadata as
training signal.

## User Workflow

The intended operator workflow is:

1. Select a caption dataset.
2. Keep **Set-and-forget backend run** enabled for durable jobs.
3. Configure instruction settings:
   - generated QA rows per image
   - generated QA mix
   - generated answer format
   - include or exclude `caption0`
   - include or exclude generated QA
   - include or exclude deterministic metadata QA
   - give the generator read-only label context
   - strict QA grounding
4. Click **Create VLM training dataset**.
5. Download the generated artifacts:
   - instruction JSONL for trainer import
   - instruction archive JSONL for audit
   - review JSONL for human decisions
   - instruction report JSON for readiness and metrics
6. Review generated-language rows outside the app.
7. Import reviewed JSONL decisions.
8. Re-export the final trainable instruction JSONL.
9. Import the final flat JSONL into the Qwen trainer.

Ordinary **Caption image**, **Caption next N**, and **Caption all images**
remain captioning workflows. They are not silently converted into training-data
generation.

## Operator Runbook

For a normal UI run:

1. Open a dataset in the annotation interface.
2. Confirm that the caption dataset is selected and that backend set-and-forget
   is enabled.
3. Confirm that the selected caption model is local. Download-needed models are
   shown in red.
4. Choose the instruction settings.
5. Click **Create VLM training dataset**.
6. Leave the browser open if convenient, but rely on the backend job and
   attach/recover controls for durability.
7. Download all four artifacts:
   - `caption_instruction_training.jsonl`
   - `caption_instruction_archive.jsonl`
   - `caption_instruction_review.jsonl`
   - `caption_instruction_report.json`
8. Review generated-language rows in the review JSONL.
9. Import the reviewed JSONL.
10. Re-export trainer JSONL with **Require ready report for trainer JSONL**
    enabled.
11. Run the trainer loader or a small fine-tuning dry run.

For scripted export checks, call the caption export endpoint with the same row
family settings used by the UI. The strict trainer-export gate is:

```text
require_ready_instruction_export=true
```

That server-side gate rejects exports whose report readiness is not `ready`.
Diagnostic archive, review, and report artifacts should remain available even
when trainer JSONL is not ready, because those files are what reviewers use to
repair the corpus.

## Definition Of Training-Ready

A corpus is training-ready only when all of the following are true:

- at least one row family is selected for training;
- every flattened row has non-empty `image_path`, `question`, and `answer`;
- every flattened row has required metadata and supported provenance;
- JSON-formatted answers parse as JSON;
- no duplicate canonical image/question pairs exist;
- selected rows have trainable validation and review state;
- generated-language rows that require manual review have been accepted or are
  otherwise allowed by the configured policy;
- archive, review, report, payload, and summary consistency proofs agree;
- the report's `training_readiness.status` is `ready`;
- the trainer loader accepts the file.

The ready report is a gate, not a quality guarantee. It proves structural,
provenance, and review-state safety. It does not prove that the generated prose
is semantically optimal for a final production fine-tune.

## UI Controls Implemented

The caption panel now includes a compact instruction-dataset section with:

- **Create VLM training dataset**
- **Generated QA per image**
  - default: `8`
  - range: `0..20`
  - `0` means caption0-only instruction export
- **Generated QA mix**
  - balanced
  - scene-level
  - object-focused
  - caption-variant oriented
- **Generated answer format**
  - natural text
  - JSON
- **Include caption0**
- **Include generated QA**
- **Include deterministic metadata QA**
- **Give generator read-only label context**
- **Strict QA grounding**
- **Require ready report for trainer JSONL**
- **Download instruction JSONL**
- **Download instruction archive**
- **Download review JSONL**
- **Import reviewed JSONL**
- **Download instruction report**

The UI refuses to launch an instruction job when all trainable row families are
disabled. Confirmation text distinguishes rows that will be flattened into
trainer JSONL from generated QA candidates that are archive/review-only.

The model dropdown makes availability visible: local models use the normal
local color, while models that need downloading are red. Backend jobs fail
preflight for missing models unless downloads are explicitly allowed.

## Implementation Map

The main implementation responsibilities are split as follows.

| Area | Main files | Responsibility |
| --- | --- | --- |
| Request schema | `models/schemas.py` | Normalizes instruction-dataset settings, clamps generated-QA count to `0..20`, and rejects jobs with no trainable row family |
| Dataset job launch | `localinferenceapi.py` | Starts persisted caption jobs, passes instruction settings to the runner, mirrors job progress, and applies backend job lifecycle rules |
| Source annotation and archive construction | `localinferenceapi.py` | Builds source summaries, generated-QA records, deterministic metadata QA, flattened rows, archive rows, review rows, report, export validation, artifact consistency, and readiness |
| Review import | `localinferenceapi.py` and `api/datasets.py` | Accepts JSON arrays, wrapper objects, or single review-row objects; applies metadata-only review decisions to saved caption0/generated-QA records; and rejects malformed, stale, ambiguous, duplicate, scalar, or wrong-dataset packets |
| Export API | `api/datasets.py` | Exposes caption exports and the optional `require_ready_instruction_export` server-side gate |
| UI workflow | `ybat-master/ybat.js` and `ybat-master/ybat.css` | Exposes controls, validates exports before download, formats operator errors, imports reviewed JSONL, and keeps the panel usable at narrow widths |
| Trainer import | `tools/qwen_training.py` | Imports flat image/question/answer rows, preserves metadata, resolves image paths, and rejects non-trainable rows |
| Runtime hardening | `localinferenceapi.py`, runner tooling, and caption docs | Handles prompt pressure, representative box lists, output-loop detection, recovery, set-and-forget supervision, and progress artifacts |

The implementation keeps UI checks and backend checks aligned intentionally.
Browser validation gives immediate operator feedback and must fail closed with
readable messages even when artifact payloads are missing or malformed; backend
validation is the authority for API/script use; trainer validation is the final
guard against hand-edited files.

## Artifact Contract

### Instruction Training Rows

`instruction_training_rows` is the trainer-facing JSONL shape:

```json
{
  "image_path": "train/frame.jpg",
  "question": "Describe this image in detail.",
  "answer": "A grounded visual answer.",
  "metadata": {
    "qa_id": "stable-row-id",
    "row_type": "generated_qa",
    "answer_source": "generated_qa_record",
    "answer_format": "natural",
    "validation_status": "accepted",
    "review_status": "accepted",
    "source_archive": "tator_caption_instruction_archive_v1"
  }
}
```

The Qwen trainer imports this flat shape directly and converts it into the
conversation format used by fine-tuning.

### Instruction Archive Rows

`instruction_archive_rows` are one JSONL record per image. They preserve:

- image path and split
- source annotations
- caption0 record
- generated QA candidates
- optional deterministic metadata QA
- rejection reasons
- selected flattened rows
- export metadata and consistency proofs

The archive is the audit artifact, not the direct trainer input.

### Review Rows

`instruction_review_rows` are candidate-level review records. They include:

- image path
- row origin and row type
- stable QA id
- question
- candidate answer
- selected training answer when applicable
- selected-for-training flag
- validation status
- review status
- rejection reasons
- source summary
- blank `review_decision`
- blank `review_notes`

Reviewers fill `accepted`, `rejected`, or `needs_revision` in
`review_decision` and import the reviewed JSONL back into the application.

### Instruction Report

`instruction_report` records:

- image count
- source annotation count
- caption0 count
- generated QA candidate count
- accepted and rejected generated QA counts
- deterministic metadata QA count
- QA count per image
- row-type distribution
- selected flattened row count
- split image and row counts
- rejection reasons
- source-field provenance
- corpus quality metrics
- artifact consistency
- export validation
- training readiness

## How To Inspect A Generated Packet

An external reviewer should inspect artifacts in this order:

1. **Instruction report JSON**: check `training_readiness`, row counts,
   rejection reasons, corpus quality metrics, `instruction_export_validation`,
   and `instruction_artifact_consistency`.
2. **Archive JSONL**: choose a few images and verify that source annotations,
   `caption0`, generated QA candidates, deterministic QA, rejected rows, and
   selected flattened rows are separated.
3. **Review JSONL**: confirm that generated-language rows needing review expose
   stable QA ids, image paths, row origins, candidate answers, selected training
   answers, validation state, review state, and rejection reasons.
4. **Trainer JSONL**: verify that the flattened training rows contain the
   model-visible signal only: image path, question, answer, and optional audit
   metadata.
5. **Trainer loader dry run**: import the flattened JSONL with the Qwen trainer
   loader and confirm the loader accepts the same rows the report declares
   trainable.

The archive is the evidence artifact. The review JSONL is the human-decision
artifact. The report is the run-certification artifact. The trainer JSONL is
the model-input artifact. Treating one of these as a replacement for the others
would hide either evidence, review state, or trainer simplicity.

## Example Failure Modes The Packet Should Expose

The packet is designed to make these failures visible instead of silently
training on them:

- a generated answer states a count that disagrees with trusted source labels;
- a generated QA row is useful but still review-pending;
- a reviewer marks a row rejected or needs-revision;
- a review file refers to a stale question, stale answer, wrong image, wrong
  dataset, or duplicate target;
- a trainer JSONL row is hand-edited after export;
- archive, review, report, and trainer row counts no longer agree;
- a dense label scene omits detailed boxes from prompt context while retaining
  complete authoritative counts;
- a model emits repeated punctuation or repeated tokens instead of a caption;
- a selected local model is unavailable and would need downloading.

The intended behavior is not to hide these cases. The intended behavior is to
preserve them in audit artifacts, block trainer rows when appropriate, and give
the operator a concrete reason.

## Source Annotation Behavior

`source_annotations` are derived from real label evidence and deterministic
geometry. They may include:

- object counts
- visible classes
- annotation summaries
- bounding-box instances
- coarse bounding-box geometry
- simple deterministic spatial facts
- uncertainty records
- field provenance

Important behavior:

- A missing label file is recorded as missing source evidence.
- An empty label file is treated as a real "no source objects" condition when
  the annotation workflow defines empty files that way.
- The VLM is not asked to create label-specific facts for objects absent from
  trusted labels.
- Images with no trusted labels can still receive ordinary captions and
  generated visual QA, but source-grounded count/class/spatial claims cannot be
  flattened unless supported by source annotations.

## Generated QA Behavior

Generated visual QA rows are VLM-created language candidates. They are useful
training candidates, but they are not trusted source metadata.

The generated-QA path:

1. asks for additional image-grounded question/answer candidates;
2. parses JSON output from the model;
3. validates row structure;
4. deduplicates questions within an image;
5. persists generated QA as language annotations;
6. validates again during export;
7. rewrites supported structured answers from source annotations when possible;
8. rejects unsupported or contradictory structured claims from trainer output;
9. keeps rejected rows in the archive and review JSONL.

Supported structured rewrites include:

- visible class list
- per-class counts
- object-count schema
- positive class presence
- negative class presence
- simple deterministic spatial facts

## Deterministic Metadata QA

Deterministic metadata QA is optional and off by default. It is generated by
code from trusted labels, not by the VLM.

It can create QA rows for:

- visible classes
- object counts
- per-class counts
- positive class presence
- negative class presence
- simple source-geometry spatial facts

These rows are useful when the training mix should explicitly teach
label-derived facts. They are separate from generated QA so reviewers can tell
which answers came from the model and which answers came from deterministic
source-label logic.

## Review Import Contract

The review loop is metadata-only. Importing reviewed JSONL can apply review
decisions and notes to saved caption0 and generated-QA records. It does not edit
source labels, boxes, image paths, generated questions, generated answers,
deterministic metadata QA rows, or final annotations.

Accepted decisions:

- `accepted`
- `rejected`
- `needs_revision`

Blank decisions are ignored. Deterministic metadata QA rows are skipped because
they are rebuilt from source labels during export rather than persisted as
language records.

Review import fails closed on:

- scalar or otherwise unsupported request bodies
- wrapper objects whose row container is missing or not a list
- caption0 or generated-QA review rows missing dataset identity, even before a
  reviewer fills a decision
- non-text values in text fields such as image path, QA id, question, candidate
  answer, selected training answer, review decision, or review notes
- text fields that exceed the documented import limits
- missing embedded dataset id for persisted language decisions
- dataset id mismatch
- unsupported non-blank review decisions
- missing or mismatched stable QA id
- unsupported actionable row origins
- duplicate or conflicting actionable review targets
- different row identities resolving to the same saved language record
- missing image path
- generated-QA rows missing reviewed question or answer text
- caption0 rows missing reviewed caption text
- stale generated-QA or caption0 text
- stale selected training answer text
- QA id and image-path mismatch
- ambiguous generated-QA or caption0 matches
- unresolvable caption0 targets
- forged caption0 rows that would create a new saved caption without matching
  selected-dataset, resolved-image, and current text-label provenance
- review packets that contain no accepted, rejected, or needs-revision
  caption0/generated-QA decisions to persist

Rejected and needs-revision language candidates remain in audit artifacts but
are excluded from flattened trainer rows.

## Artifact Consistency And Readiness

The browser and backend treat the instruction JSONL, archive JSONL, review
JSONL, and instruction report as one export set.

Consistency checks include:

- trainer row count matches the report selected-row count
- trainer row identities match selected review rows
- trainer row identities match archive candidates by canonical image path, QA
  id, and normalized question
- trainer answers match selected review/archive answers where present
- archive image count matches the report image count
- each archive row's selected-row count matches the flattened rows for that
  image
- archive JSONL has no duplicate canonical image paths
- review row count matches the report review-row count
- selected review row count matches the report selected-row count
- manual-review row count matches the report manual-review count
- review rows preserve the required review schema, supported review decisions,
  supported actionable origins, and selected-row training answers before export
  and before review import mutates saved metadata
- embedded consistency proofs agree across payload, archive, report, and summary

The report includes `training_readiness`:

- `ready`: structurally valid, selected language rows accepted or otherwise
  trainable, no quality warnings.
- `needs_review`: selected language rows still need human review or quality
  gates warn.
- `blocked`: no selected rows, invalid export rows, inconsistent artifacts, or
  selected rows rejected/needs-revision.

The UI refuses blocked trainer JSONL downloads. By default, it also refuses
needs-review downloads through **Require ready report for trainer JSONL**.
Scripts can request equivalent server-side refusal with
`require_ready_instruction_export=true`.

## Trainer Import Boundary

The Qwen training loader now imports flat instruction rows directly. It
preserves metadata for audit and converts each row into image/question/answer
conversation format.

The loader is a final safety boundary. It refuses instruction rows with:

- missing provenance
- missing or unknown validation status
- rejected, failed, or invalid validation status
- missing or unknown review status
- rejected or needs-revision review status
- invalid JSON answers for deterministic or JSON-formatted rows
- duplicate canonical image-path/question pairs
- unresolved or conflicting image aliases

This protects training runs from stale or hand-edited files that bypass browser
or backend export checks.

## Runtime Hardening

The instruction-dataset path depends on dataset-scale captioning, so the
runtime was hardened around observed failure modes.

### Output Token Budgets

The UI now separates Auto output-token behavior from explicit numeric override.
Thinking-capable models can receive higher automatic budgets because they may
emit far more generated tokens before the final answer. A numeric user value is
treated as the hard per-call cap after schema clamping.

Prompt preview and Qwen traces report the effective output-token budget so logs
can be reconciled with the UI setting.

### Prompt Size And Box Lists

Dense scenes no longer require dumping every box into a single prompt. Counts
remain complete and authoritative. Box lists become representative spatial
subsets when they are too large, selected for class and region coverage. UI and
prompt wording state that omitted boxes are not absent objects.

Prompt-size accounting estimates rendered prompt pressure and can reduce
automatic output budgets when the prompt is already large. Explicit user caps
are not silently raised.

### Loop Detection And Recovery

Streaming generation acts as a live repeated-output inspector. Repeated
punctuation or repeated-token loops are detected while tokens arrive, the stream
is closed, the runtime is unloaded, and recovery routes through the configured
safe path. A trimmed repeated fragment is not accepted as a successful caption.

Recovery can use:

- safer decoding retry
- fallback model
- text-only composition from completed window observations
- deterministic count/layout fallback when authoritative counts exist

### Full-Image Composition

Windowed/detailed captioning can compose a final full-image caption after crop
observations. The full-image stage can feel slower because it combines the image
or text-only crop evidence, full-frame counts, representative spatial evidence,
and user caption policy. The model is not intentionally reloaded simply because
the pipeline reaches this stage. Reloads are expected when a different runtime
or model is selected, when the operator does not keep the model resident, or
when loop/error recovery deliberately unloads the runtime before retrying.

### Set-And-Forget Mode

Set-and-forget is the durable default for long dataset jobs. It uses:

- persisted backend jobs
- isolated worker attempts
- attach/recover controls
- auto-resume metadata
- progress mirroring
- heartbeat artifacts
- loop-recovery telemetry
- failure-rate and quality gates
- supervisor/restart assumptions

Metal/GPU process aborts are not catchable Python exceptions. The operational
strategy is process isolation, persisted artifacts, supervised restart, and
resumable jobs.

## Why The Design Is Conservative

The workflow uses separate generation, archive, review, import, and
trainer-import phases. That is intentional.

This design prevents common training-data failures:

- generated prose becoming trusted source metadata;
- flat training JSONL hiding where an answer came from;
- rejected generated rows disappearing from audit;
- hand-edited trainer files bypassing validation;
- stale review rows applying to the wrong image or QA record;
- dense prompts causing oversized model inputs;
- repeated-token loops being accepted as captions;
- long backend jobs depending on a browser tab.

The flat trainer JSONL stays small and stable. The archive, review rows, and
report carry the evidence needed to decide whether the flat rows should be
trusted.

## Files To Review

Primary backend and export logic:

- `localinferenceapi.py`
- `models/schemas.py`
- `tools/run_qwen_caption_flow_benchmark.py`
- `tools/qwen_training.py`

Primary UI files:

- `ybat-master/ybat.js`
- `ybat-master/ybat.css`

Primary tests:

- `tests/test_qwen_caption_dataset_job.py`
- `tests/test_qwen_training_backend.py`
- `tests/test_dataset_linked_annotation_flows.py`
- `tests/test_labeling_panel_layout_contract.py`
- `tests/test_qwen_caption_ui_smoke_tool.py`
- `tests/test_qwen_caption_prompt.py`
- `tests/test_qwen_caption_flow_benchmark.py`
- `tests/test_qwen_progress.py`
- `tests/test_macos_backend_launcher_contract.py`
- `tests/test_qwen_caption_operation_audit.py`
- `tests/test_qwen_caption_soak_audit.py`
- `tests/test_qwen_caption_soak_certification.py`
- `tests/test_qwen_caption_soak_drill.py`
- `tests/test_qwen_caption_soak_preflight.py`
- `tests/test_qwen_caption_soak_supervisor.py`
- `tests/test_qwen_caption_soak_watchdog.py`
- `tests/test_qwen_caption_unattended_launcher.py`

Supporting docs:

- `docs/qwen_caption_training_dataset_external_review_handoff.md`
- `docs/qwen_caption_instruction_dataset_external_partner_packet.md`
- `docs/qwen_caption_instruction_dataset_partner_handoff.md`
- `docs/qwen_caption_instruction_dataset_hardening_report.md`
- `docs/qwen_caption_prompt_stack.md`
- `docs/qwen_caption_ui_scenarios.md`

## Validation Evidence

Recent validation for this checkpoint:

```bash
node --check ybat-master/ybat.js
```

```bash
./.venv-macos/bin/python -m py_compile \
  localinferenceapi.py \
  api/datasets.py \
  models/schemas.py
```

```bash
./.venv-macos/bin/python -m pytest \
  tests/test_qwen_caption_dataset_job.py \
  tests/test_qwen_training_backend.py \
  tests/test_dataset_linked_annotation_flows.py::test_caption_export_route_blocks_when_backend_caption_job_is_active \
  tests/test_dataset_linked_annotation_flows.py::test_export_captions_blocks_active_backend_caption_job_before_dataset_read \
  tests/test_dataset_linked_annotation_flows.py::test_download_dataset_entry_blocks_active_backend_caption_job_before_dataset_read \
  tests/test_dataset_linked_annotation_flows.py::test_dataset_download_route_blocks_when_backend_caption_job_is_active \
  tests/test_dataset_linked_annotation_flows.py::test_set_dataset_glossary_blocks_active_backend_caption_job_before_dataset_read \
  tests/test_dataset_linked_annotation_flows.py::test_dataset_glossary_route_blocks_when_backend_caption_job_is_active \
  tests/test_dataset_linked_annotation_flows.py::test_instruction_review_import_blocks_active_backend_caption_job_before_dataset_read \
  tests/test_dataset_linked_annotation_flows.py::test_instruction_review_route_blocks_when_backend_caption_job_is_active \
  tests/test_dataset_linked_annotation_flows.py::test_caption_mutations_block_active_backend_caption_job_before_dataset_read \
  tests/test_dataset_linked_annotation_flows.py::test_delete_linked_dataset_blocks_active_caption_dataset_job \
  tests/test_dataset_linked_annotation_flows.py::test_delete_linked_dataset_allows_completed_caption_dataset_job \
  tests/test_dataset_linked_annotation_flows.py::test_caption_instruction_strict_export_gate_requires_ready_proofs \
  tests/test_dataset_linked_annotation_flows.py::test_caption_instruction_strict_export_route_blocks_malformed_rows_when_ready_required \
  tests/test_dataset_linked_annotation_flows.py::test_caption_alternate_routes_append_update_export_and_delete \
  tests/test_labeling_panel_layout_contract.py \
  tests/test_qwen_caption_ui_smoke_tool.py \
  -q
```

Result:

```text
262 passed, 8 warnings
```

Additional focused validation recorded in the supporting hardening docs covers:

- caption dataset job start and route start rejecting same-dataset active jobs
- review-export refusal when persisted language review rows lack dataset
  identity
- zero-action review import refusal for blank-decision and deterministic-only
  API packets
- artifact consistency and same-count identity mismatches
- canonical image-path handling
- review import persistence and fail-closed cases
- stale caption0/generated-QA text rejection
- stale selected training-answer rejection
- duplicate actionable review-target rejection
- review-row text-field type and length rejection in the browser and backend
- route-level review import acceptance for JSON arrays and single review-row
  objects, plus parser-owned rejection of scalar bodies
- strict server-side trainer-export refusal when readiness is not ready,
  report format or ready flags are inconsistent, corpus metrics drift,
  export-validation proof fails, or versioned artifact-consistency proofs are
  missing, wrong-version, inconsistent, or paired with missing, malformed,
  stale, or mismatched returned artifact arrays
- ordinary caption exports and instruction artifact actions refusing to run
  while a backend caption job id is still active
- the caption export HTTP route opting into the backend active-job guard and
  returning `caption_export_busy` for API clients while a dataset job is active
- full dataset ZIP download rejecting with `dataset_download_busy` before
  dataset or overlay reads while a dataset job is active, with the UI showing
  the server-side failure instead of fire-and-forget downloading an error body
- dataset-manager download UI contract proving busy responses do not save a
  file and successful ZIP responses save through the shared blob path
- dataset glossary save rejecting with `caption_metadata_busy` before metadata
  reads or writes while a dataset job is active
- UI error formatting for caption-job busy, metadata-busy, same-dataset active
  job, and annotation-lock guard details, including dataset-manager glossary
  save failures and direct caption/text-label save failures
- manual caption archive save/update/primary/delete controls reporting success
  only after a real mutation result, with shared formatted failure messages
- shared instruction artifact failure reporting for trainer JSONL, archive,
  review JSONL, reviewed-row import, and report actions, including
  already-formatted blocked messages that must not be double-prefixed
- UI backend-job failure formatting for pilot-certification,
  backend-supervision, and runner-preflight report failures
- caption dataset job launch rejecting active annotation locks before job
  execution, rolling back failed reservations, reserving before write preflight
  to close delete/export races, and still allowing a matching
  `annotation_session_id`
- reviewed JSONL import rejecting with `caption_review_import_busy` before
  dataset/archive reads while a dataset job is active
- text-label save and caption add/update/delete rejecting with
  `caption_mutation_busy` before dataset reads while a dataset job is active
- dataset deletion rejecting with `dataset_delete_blocked_active_jobs` while a
  caption dataset job references the dataset, and allowing deletion after that
  job reaches a terminal status
- trainer import of flat rows
- trainer rejection of non-trainable rows
- rendered UI smoke for visible controls and unclipped caption actions
- restricted project-name scan across code, docs, tests, tools, UI, and backend

## Reproducible Verification Commands

From the repository root, a reviewer can reproduce the current smoke and
contract checks with:

```bash
node --check ybat-master/ybat.js
```

```bash
./.venv-macos/bin/python -m py_compile \
  localinferenceapi.py \
  api/datasets.py \
  models/schemas.py
```

```bash
./.venv-macos/bin/python -m pytest \
  tests/test_qwen_caption_dataset_job.py \
  tests/test_qwen_training_backend.py \
  tests/test_dataset_linked_annotation_flows.py::test_caption_export_route_blocks_when_backend_caption_job_is_active \
  tests/test_dataset_linked_annotation_flows.py::test_export_captions_blocks_active_backend_caption_job_before_dataset_read \
  tests/test_dataset_linked_annotation_flows.py::test_download_dataset_entry_blocks_active_backend_caption_job_before_dataset_read \
  tests/test_dataset_linked_annotation_flows.py::test_dataset_download_route_blocks_when_backend_caption_job_is_active \
  tests/test_dataset_linked_annotation_flows.py::test_set_dataset_glossary_blocks_active_backend_caption_job_before_dataset_read \
  tests/test_dataset_linked_annotation_flows.py::test_dataset_glossary_route_blocks_when_backend_caption_job_is_active \
  tests/test_dataset_linked_annotation_flows.py::test_instruction_review_import_blocks_active_backend_caption_job_before_dataset_read \
  tests/test_dataset_linked_annotation_flows.py::test_instruction_review_route_blocks_when_backend_caption_job_is_active \
  tests/test_dataset_linked_annotation_flows.py::test_caption_mutations_block_active_backend_caption_job_before_dataset_read \
  tests/test_dataset_linked_annotation_flows.py::test_delete_linked_dataset_blocks_active_caption_dataset_job \
  tests/test_dataset_linked_annotation_flows.py::test_delete_linked_dataset_allows_completed_caption_dataset_job \
  tests/test_dataset_linked_annotation_flows.py::test_caption_instruction_strict_export_gate_requires_ready_proofs \
  tests/test_dataset_linked_annotation_flows.py::test_caption_instruction_strict_export_route_blocks_malformed_rows_when_ready_required \
  tests/test_dataset_linked_annotation_flows.py::test_caption_alternate_routes_append_update_export_and_delete \
  tests/test_labeling_panel_layout_contract.py \
  tests/test_qwen_caption_ui_smoke_tool.py \
  -q
```

Run the restricted-name scan with the current internal restricted-term pattern:

```bash
rg -n -i "$RESTRICTED_PROJECT_TERMS" \
  docs ybat-master localinferenceapi.py api models tests tools \
  --glob '!**/.git/**'
```

The expected result is no matches. If this command returns any match, the docs
or code should be cleaned before external handoff.

## Acceptance Criteria For External Review

Treat the implementation as accepted for a small real-data pilot when all of
the following are true:

- The UI can launch **Create VLM training dataset** for a selected caption
  dataset without using ordinary caption export controls as a substitute.
- The job completes or records a recoverable failure with persisted status and
  enough telemetry to resume or diagnose.
- The exported report, archive, review, and trainer files agree on image and row
  counts.
- The report is `ready` only when validation proofs, artifact-consistency
  proofs, corpus metrics, review state, and returned artifact arrays agree.
- Review JSONL can be edited with accepted/rejected/needs-revision decisions and
  imported without mutating labels or generated content.
- Re-export after review changes only row selection/review state, not source
  annotations.
- The trainer loader accepts the final JSONL and rejects deliberately malformed
  or non-trainable variants.
- A manual reviewer confirms that generated QA is grounded and useful on the
  pilot images.

Treat the implementation as not yet accepted for larger training use if any of
these fail. In that case, the next work should be a targeted hardening patch,
not a larger generation run.

## What This Does Not Claim

This implementation creates and validates the training-data workflow. It does
not claim that any generated corpus is automatically training-grade.

Not claimed:

- The generated QA content is semantically perfect.
- A ready report replaces human content review.
- The caption panel launches fine-tuning.
- Generated language can update source labels.
- Python can catch process-level Metal/GPU aborts.
- Deterministic fallback is equivalent to normal model captioning.

## Required Pilot Before Training Use

Before using exported rows for fine-tuning, run a small real-data pilot:

1. Include at least:
   - one dense labeled scene
   - one image with no source objects
   - one image with missing source labels
   - one image with multiple object classes
   - one image with existing alternate captions
   - one image with small objects near crop boundaries
2. Generate instruction artifacts with set-and-forget enabled.
3. Download instruction JSONL, archive JSONL, review JSONL, and report JSON.
4. Manually review generated QA for grounding, usefulness, and answer format.
5. Import reviewed decisions.
6. Re-export trainer JSONL with the ready-report gate enabled.
7. Import the JSONL into the Qwen trainer.
8. Run at least one small fine-tuning dry run or loader-plus-batch smoke.

## External Review Checklist

Reviewers should verify:

- Generated language never populates `source_annotations`.
- The archive and trainer JSONL are separate products.
- `caption0`, generated QA, and deterministic metadata QA remain distinct.
- All model-visible training facts appear in `question` or `answer`.
- Unsupported structured generated claims are rejected or rewritten from source.
- Rejected rows remain auditable but are not flattened into trainer rows.
- Review import applies metadata only and fails closed on stale or ambiguous
  rows.
- The UI can launch instruction-dataset creation in one click after settings are
  chosen.
- The default path is set-and-forget, not manual babysitting.
- Model availability is visible before launch.
- Dense label scenes use representative box subsets while preserving complete
  authoritative counts.
- The trainer can import the exported flat rows directly.
- The trainer refuses stale or hand-edited non-trainable rows.

## Open Decisions For The Review Team

- What manual-review threshold is required before generated QA rows can be used
  for training?
- Should deterministic metadata QA remain off by default for all datasets?
- Should generated QA mixes become explicit row-type budgets instead of broad
  presets?
- Which corpus-quality metrics should be hard blockers for the first production
  fine-tuning run?
- Should review import eventually support edited replacement questions or
  answers, or should it remain metadata-only?
- Should review-pending diagnostic exports remain possible, or should trainer
  JSONL always require `ready` readiness?
- What fine-tuning dry-run size is sufficient to certify the first exported
  corpus?
