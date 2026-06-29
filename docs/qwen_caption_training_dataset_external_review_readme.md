# Qwen Caption Training Dataset External Review README

Date: 2026-06-29

## Purpose

This is the shareable external-review entry point for the caption-derived VLM
training-dataset work. It explains what was built, why it was built, how an
operator uses it, which artifacts it produces, which validation gates protect
the output, and what still must be proven before any generated corpus is used
for fine-tuning.

The document is intentionally dataset-neutral. Do not add customer, partner,
dataset, or private project names to this file. The implementation can be
reviewed from its data contracts, UI behavior, validation boundaries, and test
evidence.

## Bottom Line

The captioning stack now has a distinct workflow for creating VLM training data
from a selected image dataset. The workflow can generate one broad caption row
per image, configurable VLM-created visual question/answer candidates, optional
deterministic label-derived QA rows, and four auditable export artifacts.

The current status is:

- ready for external implementation review;
- ready for a small real-data pilot;
- not yet certified as a production corpus generator for large-scale
  fine-tuning until a reviewed pilot export is loaded by the trainer and passes
  at least a loader-plus-batch or fine-tuning dry run.

The most important design decision is evidence separation. Trusted label facts,
generated language, deterministic label-derived rows, human review decisions,
and final trainer rows are all stored as separate layers. A generated sentence
does not become a source label, and a row does not become trainable merely
because the model wrote it.

## What Changed

The previous captioning path was useful for annotation assistance, but it was
not enough for VLM training-data creation. A fine-tuning corpus needs varied
image/question/answer rows, row provenance, review state, consistency checks,
and a trainer loader that rejects bad rows.

The implemented product path now includes:

| Area | Implemented behavior | Why it matters |
| --- | --- | --- |
| UI workflow | A separate **Create VLM training dataset** action | Training-data creation is explicit and is not hidden inside ordinary caption export |
| Row families | `caption0`, generated visual QA, and optional deterministic metadata QA | A training corpus can ask multiple grounded questions about each image instead of using only one caption |
| Source truth | `source_annotations` are derived from trusted labels and deterministic geometry | Label-derived facts remain auditable and are not overwritten by generated prose |
| Generated language | Captions and generated QA are candidate language records | Useful generated text can be reviewed and selected without becoming source truth |
| Trainer rows | Flat `image_path` / `question` / `answer` JSONL | The trainer receives a simple model-visible shape |
| Audit artifacts | Archive JSONL, review JSONL, and report JSON | Reviewers can trace why each row was accepted or rejected |
| Review loop | Review JSONL export and metadata-only reviewed-row import | Human decisions can be applied without editing labels, boxes, questions, or answers |
| Validation | Browser, backend, report, archive, review-row, and trainer-loader checks | Stale, malformed, duplicated, rejected, or review-pending rows fail before training |
| Runtime | Set-and-forget jobs with progress, attach/recover, loop recovery, and active-job guards | Dataset-scale jobs can run without relying on a browser tab or a single fragile model call |
| Prompt safety | Dynamic token budgeting and representative dense-box prompts | Large or dense scenes remain bounded while authoritative counts stay complete |

## Why It Was Built This Way

### A Single Caption Is Too Thin

One caption per image is useful, but it does not teach a VLM to answer varied
questions. The new path produces a broad description row plus optional generated
and deterministic QA rows so the same image can contribute scene, object,
count, presence, spatial, and attribute prompts.

### Generated Language Is Useful But Not Trusted Source Data

Generated captions and generated answers can be plausible and wrong. The
implementation keeps generated language in `language_annotations` and source
facts in `source_annotations`. If a generated QA row makes a supported
structured claim, such as a count or class-list answer, the flattened training
answer can be rewritten from trusted source annotations. Unsupported or
contradictory structured claims stay in the archive/review files but are
excluded from trainer JSONL.

### The Trainer Needs Flat Rows, Reviewers Need Rich Evidence

The trainer should not load a complex construction archive. It should load flat
rows:

```json
{"image_path":"images/example.jpg","question":"Describe this image in detail.","answer":"A grounded answer."}
```

That row shape is correct for training, but it is too thin for audit. The
archive, review file, and report provide the missing evidence: source labels,
generated candidates, deterministic rows, selected rows, rejected rows, review
state, readiness state, and consistency proof.

### Long Caption Jobs Need Set-And-Forget Behavior

During hardening, the observed failure modes were not theoretical: oversized
prompts, confusing token-budget traces, repeated-token loops, full-image
composition stalls, missing model files, backend restarts, and MLX/Metal GPU
faults all shaped the implementation. The durable path therefore uses backend
jobs, persisted status, progress polling, attach/recover behavior, active-job
guards around archive reads and exports, stream loop inspection, bounded retry
policies, runtime unload on repeated-output loops, and degraded fallback only
when explicitly recorded.

## Operator Workflow

1. Open a dataset in the caption panel.
2. Confirm model availability. Download-needed model choices are styled as not
   locally available; local models use the normal available style.
3. Keep set-and-forget enabled for dataset-scale runs unless intentionally
   debugging a manual path.
4. Choose training-dataset settings:
   - generated QA rows per image;
   - generated QA mix;
   - generated answer format;
   - include or exclude `caption0`;
   - include or exclude generated QA;
   - include or exclude deterministic metadata QA;
   - pass read-only label context to the generator;
   - require strict QA grounding;
   - require a ready report before trainer JSONL export.
5. Click **Create VLM training dataset**.
6. Monitor progress or attach/recover after a refresh.
7. Download the complete artifact set from the same run:
   - trainer JSONL;
   - instruction archive JSONL;
   - review JSONL;
   - instruction report JSON.
8. Review generated-language rows externally.
9. Import reviewed JSONL decisions.
10. Re-export trainer JSONL with readiness gates enabled.
11. Load the final JSONL into the trainer or trainer-equivalent dry-run path.

Ordinary **Caption image**, **Caption next N**, and **Caption all images** remain
caption-assistance workflows. They are not substitutes for the instruction
dataset workflow.

## Artifact Contract

The workflow produces four coordinated artifact families from one run.

| Artifact | Shape | Purpose |
| --- | --- | --- |
| Trainer JSONL | One flat row per trainable `image_path` / `question` / `answer` | Model-visible training input |
| Instruction archive JSONL | One construction record per image | Full provenance and audit history |
| Review JSONL | One candidate-level row per reviewable item | Human acceptance/rejection workflow |
| Instruction report JSON | One run-level report | Readiness, metrics, consistency, and blocking reasons |

Do not evaluate trainer JSONL alone. The trainer file is the model input, but
the archive, review file, and report explain whether those model-input rows are
trustworthy.

## Data Ownership Model

| Layer | Written by | Contains | Can generated language write it? | Direct trainer input? |
| --- | --- | --- | --- | --- |
| `source_annotations` | Label parser and deterministic geometry | trusted counts, classes, instances, spatial facts, provenance | No | No |
| `language_annotations` | VLM caption and generated QA passes | `caption0` and generated QA candidates | Yes | Only after flattening |
| `deterministic_metadata_qa_pairs` | Code from trusted labels | label-derived QA rows | No for answers | Only when selected |
| `review_metadata` | Human review import | decisions and notes | No | Controls row eligibility |
| `flattened_training_rows` | Exporter | selected flat trainer rows | Derived only | Yes |

Allowed flow:

```text
trusted labels and deterministic geometry
  -> source_annotations
  -> deterministic metadata QA
  -> selected flat trainer rows

image and read-only context
  -> generated language candidates
  -> validation and review
  -> selected flat trainer rows
```

Forbidden flow:

```text
generated caption
  -> source_annotations.object_counts

generated answer
  -> source_annotations.visible_classes

row metadata
  -> hidden training signal
```

If a fact should train the model, it must be present in the trainer row's
`question` or `answer`, not only in metadata.

## Runtime Hardening Summary

| Failure mode observed or anticipated | Hardened behavior |
| --- | --- |
| User cap and automatic token budget were hard to reconcile | UI separates Auto from explicit numeric override; runtime logs requested and effective budgets |
| Thinking-capable models need larger defaults | Auto budgets account for thinking-capable models while numeric user caps remain hard caps |
| Dense label scenes could dump huge box lists into prompts | Counts remain authoritative; box context becomes a representative spatial subset when capped |
| Huge prompts can crowd output budget | Prompt-size estimation and runtime prompt measurement adapt non-explicit budgets |
| Repeated punctuation or token loops looked like hangs | Streaming output inspection detects repeated surfaces, trims progress display, raises a controlled loop error, unloads the runtime, and routes recovery |
| Full-image visual composition could stall after successful crop passes | Set-and-forget windowed Auto uses text-only full-image composition from completed crop evidence instead of resending the full image tensor |
| MLX/Metal GPU faults can abort the Python process | Dataset runs use isolated child attempts and recorded recovery policy; process-level aborts are handled by the parent runner, not hidden |
| Backend restarts can leave users unsure about job state | Persisted status, progress, attach/recover behavior, active-job guards, and default-root discovery mirrors for custom-output jobs make live or interrupted runs visible |
| Model availability was ambiguous | Download-needed models are visually distinct in the selector |
| Archive/export actions could race a live caption job | UI controls and backend routes check active caption dataset jobs before mutating, reading, exporting, or importing review data |

## UI And UX Contract

The external reviewer should expect these operator-facing behaviors:

- model choices communicate whether the model is available locally;
- instruction settings are visible near the caption-all controls;
- launch is blocked when no trainable row family is enabled;
- generated QA can be archive/review-only when excluded from trainer JSONL;
- caption, prompt, glossary, dataset, recipe, archive, export, and review-import
  controls lock while a caption or instruction job mutates the archive;
- stale click or input events do not silently mutate the active run settings;
- long status text wraps without squeezing action buttons;
- download failures report actionable validation messages, not raw internal
  codes;
- live-job busy responses are shown as blocked operator states, not generic
  export/import failures;
- read routes and text-label mirror routes fail closed while a same-dataset
  caption job owns the archive.

The core UX requirement is that a user can choose settings, start the training
dataset job, walk away, come back, inspect status, download artifacts, review
rows, import decisions, and re-export without manually babysitting each model
call.

## Validation Boundaries

The implementation validates the same artifacts at multiple layers:

- **Browser preflight** checks row shape, metadata, readiness, review status,
  duplicate identities, and report/archive/review/trainer consistency before
  writing downloads.
- **Backend export validation** checks flat trainer rows, report readiness,
  artifact consistency, review-row shape, image alias resolution, and strict
  export requirements for API/script callers.
- **Review import validation** rejects stale, ambiguous, duplicate,
  wrong-dataset, unsupported-origin, malformed, or content-mutating review
  files before metadata is changed.
- **Trainer loader validation** rejects unresolved images, duplicate canonical
  image/question pairs, malformed JSON answers, rejected or needs-revision
  rows, missing provenance, and unknown validation/review states.

This redundancy is deliberate. Training artifacts are files and can be copied,
edited, mixed, or shared outside the UI. The trainer loader remains the last
defense even when the UI and backend already validated the export.

## What Has Been Verified

The current implementation has focused unit and UI-contract coverage for:

- instruction request normalization and row-family settings;
- generated QA parsing, validation, deduplication, rejection, and flattening;
- deterministic metadata QA from source labels;
- artifact consistency across trainer, archive, review, and report artifacts;
- browser-side download validation and failure messages;
- backend strict export gating;
- review JSONL import shape checks and metadata-only mutation;
- trainer loader acceptance and rejection behavior;
- active-job guards around caption archive reads, text-label reads, exports,
  review import, recipe application, dataset selection, and launch actions;
- model selector availability styling;
- prompt/token-budget preview behavior;
- loop detection and recovery telemetry;
- layout behavior for wrapped caption/instruction controls.

The canonical deeper validation command list is maintained in
`docs/qwen_caption_training_dataset_complete_partner_packet.md` under
**Reproducible Verification Commands**.

## What Is Not Claimed Yet

This documentation does not claim:

- every generated QA row is semantically correct;
- a ready report replaces human content review;
- deterministic fallback captions are equivalent to normal model captions;
- Python can catch every process-level GPU fault in-process;
- the caption UI launches fine-tuning itself;
- generated language can update source labels;
- a large production corpus is certified before a real-data pilot.

## Required Pilot Before Training Use

Before using exported rows for fine-tuning, run a small reviewed pilot:

1. Include a dense labeled scene, an image with no source objects, an image with
   missing labels, a multi-class image, an image with existing alternate
   captions, and an image with small objects near crop boundaries.
2. Generate artifacts with set-and-forget enabled.
3. Download trainer JSONL, archive JSONL, review JSONL, and report JSON from
   the same run.
4. Review generated-language rows for grounding, usefulness, and answer format.
5. Import reviewed decisions.
6. Re-export trainer JSONL with ready-report gating enabled.
7. Load the JSONL through the trainer loader.
8. Run a loader-plus-batch or small fine-tuning dry run.
9. Record degraded recovery rate, rejected-row rate, manual-review throughput,
   and reviewer usefulness findings.

Only after that pilot should a reviewer decide whether to scale to a larger
training corpus.

## Review Acceptance Criteria

An external reviewer should accept the implementation for pilot use only if:

- the UI can launch **Create VLM training dataset** as a distinct workflow;
- exported artifacts agree on image identities, row identities, counts, and
  readiness;
- `source_annotations` contain only trusted label-derived or deterministic
  facts;
- generated language remains in generated-language layers unless selected into
  flat trainer rows;
- unsupported structured generated claims are rejected or rewritten from source
  facts;
- rejected and review-pending rows remain auditable but do not enter trainer
  JSONL;
- reviewed JSONL import applies metadata decisions only;
- trainer import accepts the final reviewed JSONL and rejects deliberately
  malformed variants;
- a manual reviewer finds generated QA useful enough on the pilot sample.

## Documentation Map

Use these files together:

- `docs/qwen_caption_training_dataset_complete_partner_packet.md`: canonical
  complete packet with requirement mapping, artifact contract, validation
  commands, acceptance criteria, and pilot checklist.
- `docs/qwen_caption_training_dataset_complete_external_handoff.md`: concise
  implementation handoff.
- `docs/qwen_caption_training_dataset_external_consumer_dossier.md`: detailed
  external-consumer explanation of data ownership, generated QA grounding,
  review workflow, and artifact validation.
- `docs/qwen_caption_training_dataset_external_implementation_report.md`:
  implementation narrative and hardening summary.
- `docs/qwen_caption_instruction_dataset_hardening_report.md`: detailed
  hardening log.
- `docs/qwen_caption_prompt_stack.md`: prompt-stack, token-budget, dense-box,
  loop-recovery, and set-and-forget runtime contract.
- `docs/qwen_caption_ui_scenarios.md`: UI behavior and operator scenarios.

## Handoff Packet Contents

When sharing a generated sample for review, include:

- this README and the documentation map above;
- trainer JSONL;
- instruction archive JSONL;
- review JSONL;
- instruction report JSON;
- the image files referenced by the sample;
- source labels referenced by the sample;
- any caption/progress trace summary available from the same run.

Do not share trainer JSONL alone as evidence of correctness. The supporting
artifacts are the proof that the flat rows are grounded, validated, and
reviewable.
