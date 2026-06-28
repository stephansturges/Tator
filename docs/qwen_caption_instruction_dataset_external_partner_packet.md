# Qwen Caption Instruction Dataset External Partner Packet

Date: 2026-06-29

## Purpose

This packet explains what was implemented in the caption-to-instruction-dataset
work, why those choices were made, what artifacts the workflow produces, and
what still needs real-data validation before the generated corpus is treated as
training-ready.

It is written for an external technical review. The implementation is
dataset-neutral and intentionally avoids project-specific dataset names, class
lists, or private evaluation assumptions.

## Executive Summary

The captioning workflow now supports two distinct product paths:

- Ordinary captioning for annotation assistance.
- VLM instruction-dataset creation for training-data preparation.

The instruction-dataset path can be launched from the caption panel with
**Create VLM training dataset**. A run can create one broad caption row per
image, called `caption0`, plus a configurable number of generated visual
question/answer rows. It can also include optional deterministic metadata QA
rows that are derived only from trusted source labels.

The main design decision is separation of evidence:

- Source labels remain trusted source labels.
- Generated captions and generated QA remain generated language.
- Deterministic metadata QA is code-generated from trusted labels.
- Flattened trainer rows are selected outputs, not the source of truth.

The workflow exports trainer JSONL, archive JSONL, review JSONL, and a run-level
report. A reviewer can review generated-language rows outside the app, import
accepted/rejected/needs-revision decisions back into the dataset, and re-export
only trainable rows. The Qwen trainer can import the flat exported instruction
rows directly and also fails closed on stale or hand-edited rows that contradict
validation or review state.

This is now a functional product path for creating VLM training artifacts. It
is not a claim that every generated corpus is automatically training-grade. A
real-data pilot, manual generated-QA review, review import, re-export, and a
small trainer-import or fine-tuning dry run remain required before using a
corpus for model training.

## How To Use This Packet

Use this document as the reviewer entrypoint. It is intentionally broader than
a changelog:

- The early sections explain the product behavior and the design rationale.
- The middle sections describe the artifact contract and validation boundaries.
- The runtime hardening section explains why the long-running caption path was
  changed before treating instruction-data creation as a usable workflow.
- The validation evidence section records the current tests that prove the
  wiring, guardrails, export/import boundaries, and UI contracts.
- The final checklist and remaining-work sections separate implemented
  functionality from the real-data pilot work still needed before training.

The most important review question is not only whether files are produced. It is
whether a generated training row can be traced back to source evidence,
generated language, review state, and the report metrics that certify the run.

## Why This Work Was Needed

A caption-only export is useful, but it is thin for VLM fine-tuning. Training
normally benefits from varied prompts about the same image: broad scene
descriptions, object relationships, counts, class presence, spatial facts, and
short grounded answers.

The previous caption path did not fully answer these operational questions:

- Which facts came from trusted labels?
- Which facts came from generated language?
- Which generated rows are trainable?
- Which rows should be preserved for audit but excluded from training?
- Can a human reviewer accept or reject generated rows without editing labels?
- Can the trainer import the exported rows without a manual conversion script?
- Can a long dataset job run without constant browser supervision?
- Can dense label scenes avoid oversized prompts and model loops?

The new implementation answers those questions by adding a separate
instruction-dataset mode, richer audit artifacts, explicit review metadata,
training-readiness gates, trainer import compatibility, and set-and-forget
runtime hardening.

## User-Facing Workflow

The intended operator flow is:

1. Select the caption dataset.
2. Keep set-and-forget enabled for durable backend execution.
3. Choose instruction settings:
   - generated QA rows per image
   - generated QA mix
   - generated answer format
   - include or exclude `caption0`
   - include or exclude generated QA
   - include or exclude deterministic metadata QA
   - pass read-only label context to the generator
   - strict QA grounding
4. Click **Create VLM training dataset**.
5. Download the generated artifacts:
   - instruction JSONL for trainer import
   - instruction archive JSONL for audit
   - review JSONL for human decisions
   - instruction report JSON for readiness and quality metrics
6. Review generated-language rows externally.
7. Import reviewed JSONL decisions.
8. Re-export the final trainable instruction JSONL.
9. Import the final JSONL into the Qwen trainer.

Ordinary **Caption image**, **Caption next N**, and **Caption all images** remain
available as captioning workflows. They are not silently converted into
training-data workflows.

## What Was Implemented

### Instruction Dataset Mode

The backend request model now accepts instruction-dataset settings. The runner
uses those settings to create broad caption rows and optional generated visual
QA rows. Generated QA count is bounded from `0` to `20`; `0` creates a
caption0-only instruction export.

Implemented controls include:

- **Create VLM training dataset**
- **Generated QA per image**
- **Generated QA mix**
- **Generated answer format**
- **Include caption0**
- **Include generated QA**
- **Include deterministic metadata QA**
- **Give generator read-only label context**
- **Strict QA grounding**
- **Download instruction JSONL**
- **Download instruction archive**
- **Download review JSONL**
- **Import reviewed JSONL**
- **Download instruction report**

The launch path also validates the selected row-family configuration before a
backend job starts. If `caption0`, generated QA, and deterministic metadata QA
are all disabled, the UI refuses to launch and the backend request model rejects
the payload. If generated QA candidates are requested but excluded from trainer
JSONL, the confirmation text says they are archive/review-only rather than
describing them as trainer rows.

### Source Annotation Summaries

Source annotation summaries are built from existing label evidence. They are
stored separately from generated language and include object counts, visible
classes, annotation summaries, bounding-box instances, coarse geometry, simple
spatial facts, uncertainty, and field provenance.

Important source-label behavior:

- A missing label source is recorded as missing source evidence.
- An empty label file is treated as a real "no source objects" condition.
- The system does not ask the VLM to invent label-specific facts for objects
  absent from trusted labels.
- Images with no trusted labels can still receive ordinary visual captions or
  generated visual QA.
- Source-grounded count, class-list, presence, or spatial claims are flattened
  only when supported by source annotations.

### Generated Visual QA

Generated visual QA rows are VLM-created language records. They can be useful
training candidates, but they do not become trusted label metadata.

Generated QA candidates are parsed, structurally validated, deduplicated by
image/question pair, and persisted with provenance fields. The exporter validates
them again before flattening.

If a generated QA row makes a supported structured claim, such as a count,
class-list, presence, or simple spatial claim, the final flattened answer can be
rewritten from trusted source annotations. The original generated answer remains
in metadata for audit. If the structured claim cannot be checked, the row is
kept in the archive but rejected from trainer JSONL.

### Deterministic Metadata QA

Deterministic metadata QA is optional and off by default. It is generated by
code from trusted labels, not by the VLM.

Supported deterministic QA categories include:

- visible class list
- object-count schema
- per-class count
- positive class presence
- negative class presence
- simple source-geometry spatial facts

These rows are explicit because they change the corpus mix away from purely
generated visual-language rows.

### Artifact Exports

The workflow exports four main artifact types.

`instruction_training_rows` is the trainer-facing JSONL shape:

```json
{
  "image_path": "images/example.jpg",
  "question": "Describe this image in detail.",
  "answer": "A grounded answer.",
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

`instruction_archive_rows` are one per-image audit record per JSONL line. They
preserve source annotations, `caption0`, generated QA candidates, deterministic
metadata QA, rejected rows, selected flattened rows, and export metadata.

`instruction_review_rows` are candidate-level rows for human review. They
include the image path, row origin, row type, stable QA id, candidate answer,
selected training answer when applicable, selected-for-training status,
validation status, review status, source summary, rejection reasons, and blank
review decision fields.

`instruction_report` records run-level metrics, quality warnings, blocking
reasons, source-field provenance, row counts, split counts, generated-QA metrics,
and training-readiness state.

### Artifact Consistency Gates

The browser now treats the instruction training JSONL, archive JSONL, review
JSONL, and instruction report as one coherent export set. A download can pass
row-level validation and still be blocked if its counts no longer match the
report.

The additional consistency checks are:

- Trainer JSONL row count must match the report's selected flattened-row count.
- Trainer JSONL row identities must match the selected review-row identities.
- Trainer JSONL rows must match candidates in the per-image archive by image
  path, QA id, and normalized question.
- Selected review-row training answers and archive candidate answers must match
  the flattened trainer answer where those fields are present.
- Archive JSONL row count must match the report image count and the archive
  image count.
- Each archive row's selected training-row count must match the actual
  flattened rows for that image.
- Archive JSONL must not contain duplicate `image_path` rows.
- Review JSONL row count must match the report review-row count.
- Selected review-row count must match the report selected flattened-row count.
- Manual-review row count must match the report manual-review count.
- The instruction report itself must contain valid export validation,
  review-row, manual-review, readiness, and corpus-quality fields.

This matters because reviewers should not need to infer whether a hand-edited,
stale, partial, or mixed artifact belongs to the current run. The backend emits
the same versioned `instruction_artifact_consistency` object in the archive,
report, API payload, and summary. If the object is not OK, training readiness is
forced to `blocked` with `instruction_artifacts_inconsistent`. The UI also
fails closed before writing a mismatched download, while the trainer keeps its
own final validation boundary for hand-edited paths.

Flat-layout image keys are canonicalized before merge so `foo.jpg` and
`train/foo.jpg` do not become two separate instruction objects for the same
image. This also prevents nested paths with the same basename, such as
`a/img.jpg` and `b/img.jpg`, from colliding in instruction archive rows.

### Review Import

Reviewed JSONL import closes the review loop. It accepts review decisions and
applies only review metadata back to saved caption and generated-QA records.

Accepted decisions are:

- `accepted`
- `rejected`
- `needs_revision`

Review import fails closed on:

- missing embedded dataset id on persisted caption0 or generated-QA decisions
- embedded dataset id mismatch
- missing or mismatched stable QA id on persisted caption0 or generated-QA
  decisions
- malformed actionable rows
- unsupported actionable row origins
- duplicate actionable review targets, including rows that use different row
  identities but resolve to the same saved caption or generated-QA record
- generated-QA review rows missing the reviewed question or answer text
- caption0 review rows missing the reviewed caption text
- stale generated-QA targets
- QA ids whose review-row image path does not match the stored caption or
  generated-QA record
- QA ids whose image path matches but whose reviewed question, candidate answer,
  or training answer no longer matches the stored caption or generated-QA record
- ambiguous generated-QA or caption0 matches
- unresolvable caption0 targets
- caption0 rows that would create a new saved caption without a synthetic id
  that matches the selected dataset, resolved image key, and current text-label
  caption

Review import does not edit source labels, boxes, image paths, generated
questions, generated answers, deterministic metadata QA rows, or final
annotations. Rejected and needs-revision language rows remain auditable in
archive and review artifacts but are excluded from flattened trainer rows.
Deterministic metadata decisions are treated as non-persisted because those rows
are rebuilt from trusted labels during export; the UI warns instead of sending a
deterministic-only review file to the backend.

### Training Readiness

The instruction report classifies a run as:

- `ready`
- `needs_review`
- `blocked`

`ready` means selected rows pass structural checks, selected language rows have
acceptable review state, and quality gates do not warn.

`needs_review` means selected rows still need human review or the corpus has
quality warnings.

`blocked` means the export should not be used for training, for example when
there are no selected rows, no images, selected rows are invalid, or selected
language rows were rejected or marked needs-revision.

The browser refuses blocked trainer JSONL downloads. By default it also refuses
needs-review trainer JSONL through the **Require ready report for trainer
JSONL** gate. Scripts can request the same server-side refusal with
`require_ready_instruction_export=true`.

### Browser And Server Export Validation

The browser validates instruction JSONL before writing a file. The server
performs equivalent validation as `instruction_export_validation` and includes
it in archive, report, API payloads, and instruction summaries.

Validation checks include:

- missing image path
- blank question
- blank answer
- missing row metadata
- missing `qa_id`
- missing `row_type`
- missing `answer_source`
- missing or unsupported `source_archive`
- missing `answer_format`
- missing or unknown validation status
- missing or unknown review status
- invalid JSON for JSON-formatted answers
- duplicate image/question pairs
- rejected, failed, or invalid validation status
- rejected or needs-revision review status

The Qwen trainer has its own final import boundary. It imports the flat row
shape directly, converts each row into an image/question/answer conversation,
preserves row metadata, and refuses instruction rows with missing provenance,
missing or unknown validation/review state, or non-trainable state even if a
file was edited after export.
When both `review_status` and `review_decision` are present, validators inspect
both fields and fail closed if either field is rejected, needs revision, or
unknown.

## Runtime Hardening

The implementation also hardens the caption runtime because real dataset-scale
captioning exposed several failure modes.

### Token Budgets

The UI separates `Auto` output-token behavior from explicit user overrides.
Thinking-capable models can use larger automatic generation budgets because
their hidden or explicit reasoning can consume far more generated tokens. A
numeric user value remains a hard output cap after normal schema clamping.

Prompt preview and caption traces report effective output-token budgets so logs
can be reconciled with the UI setting.

### Prompt Size And Box Lists

Dense source-label prompts no longer dump every box into a full-image prompt.
Counts remain authoritative, while prompt box lists can be capped to a
representative spatial subset that covers classes and image regions. UI and
prompt wording explain that omitted boxes are not absent objects.

Prompt-size accounting estimates prompt pressure before generation and can
reduce automatic output budgets when the rendered prompt is already large.
Explicit user caps are not silently raised.

### Loop Detection And Recovery

Streaming caption paths act as a live repeated-output inspector. Repeated
punctuation or token loops are detected while text arrives, the stream is
closed, the runtime is unloaded, and recovery is routed through the configured
safe path. A trimmed repeated fragment is not accepted as a successful caption.

Recovery options include safe retry, fallback model, text-only composition from
completed window observations, and deterministic count/layout fallback when
authoritative counts are available and model paths fail.

### Set-And-Forget Mode

Set-and-forget is the durable default for dataset jobs. It uses persisted backend
jobs, isolated attempts, health gates, attach/recover controls, progress
mirroring, resume metadata, loop-recovery telemetry, and supervised process
restart assumptions.

The implementation does not pretend that Metal/GPU process aborts are catchable
Python exceptions. The operational strategy is subprocess isolation, persisted
artifacts, supervisor restart, and resumable jobs.

### Model Availability

Model dropdown entries distinguish local models from models that require a
download. Download-needed models are styled red; local models use the normal
local color. Backend jobs fail preflight for missing models unless downloads
are explicitly allowed.

## Why The Design Is Conservative

The workflow deliberately uses separate generation, archive, review, import, and
trainer-import phases. This adds some complexity, but it protects training data
quality:

- A generated answer can be useful but not yet trainable.
- A label-derived fact should be reproducible from labels, not copied from
  generated prose.
- A human reviewer needs a safe way to reject rows without editing labels.
- A trainer needs a simple stable row shape.
- Reviewers need richer evidence than the trainer needs.
- Long caption jobs need durable recovery and auditable failure telemetry.

The flat instruction JSONL is therefore the trainer input, while the archive,
review rows, and report are the evidence layer used to decide whether the
trainer input should be trusted.

## Files To Review

Primary backend and export logic:

- `localinferenceapi.py`
- `models/schemas.py`
- `tools/run_qwen_caption_flow_benchmark.py`
- `tools/qwen_training.py`

Primary UI files:

- `ybat-master/ybat.html`
- `ybat-master/ybat.js`
- `ybat-master/ybat.css`

Primary tests:

- `tests/test_qwen_caption_dataset_job.py`
- `tests/test_qwen_training_backend.py`
- `tests/test_dataset_linked_annotation_flows.py`
- `tests/test_labeling_panel_layout_contract.py`
- `tests/test_qwen_caption_ui_smoke_tool.py`
- `tests/test_qwen_caption_flow_benchmark.py`
- `tests/test_qwen_caption_prompt.py`
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
- `docs/qwen_caption_instruction_dataset_hardening_report.md`
- `docs/qwen_caption_prompt_stack.md`
- `docs/qwen_caption_ui_scenarios.md`

## Validation Evidence

Recent validation for this checkpoint:

```bash
node --check ybat-master/ybat.js
```

```bash
./.venv-macos/bin/python -m py_compile localinferenceapi.py
```

```bash
./.venv-macos/bin/python -m pytest \
  tests/test_qwen_caption_dataset_job.py::test_caption_instruction_training_readiness_blocks_invalid_export_rows \
  tests/test_labeling_panel_layout_contract.py::test_qwen_caption_export_preserves_saved_alternates_and_primary_rows \
  tests/test_labeling_panel_layout_contract.py::test_qwen_caption_instruction_training_validator_blocks_non_trainable_rows \
  -q
```

Result:

```text
3 passed
```

```bash
./.venv-macos/bin/python -m pytest \
  tests/test_qwen_caption_dataset_job.py \
  tests/test_qwen_training_backend.py \
  tests/test_dataset_linked_annotation_flows.py::test_caption_alternate_routes_append_update_export_and_delete \
  tests/test_labeling_panel_layout_contract.py \
  tests/test_qwen_caption_ui_smoke_tool.py \
  -q
```

Result:

```text
184 passed
```

Focused artifact-consistency contract, including same-count identity mismatch
coverage:

```bash
./.venv-macos/bin/python -m pytest \
  tests/test_qwen_caption_dataset_job.py::test_caption_instruction_artifact_consistency_validator_blocks_same_count_identity_mismatches \
  tests/test_qwen_caption_dataset_job.py::test_caption_instruction_artifact_consistency_validator_blocks_mismatched_backend_counts \
  tests/test_labeling_panel_layout_contract.py::test_qwen_caption_instruction_artifact_consistency_blocks_mismatched_exports \
  -q
```

Result:

```text
3 passed
```

Additional validation recorded in the hardening report includes trainer import
boundary tests, review-import fail-closed tests, stale review-text rejection
tests, caption prompt tests, set-and-forget supervision tests, unattended soak
certification tests, rendered browser smoke, and a restricted project-name scan.

## External Review Checklist

Use this checklist to review the implementation.

1. Confirm that **Create VLM training dataset** exists as a distinct workflow
   from ordinary captioning.
2. Confirm that source labels are never overwritten by generated language.
3. Confirm that empty label files are treated as empty source evidence, not as
   missing data and not as a reason to ask the VLM for nonexistent label facts.
4. Confirm that generated QA rows are archived even when rejected from trainer
   output.
5. Confirm that deterministic metadata QA is off by default and only uses
   trusted labels.
6. Confirm that supported structured generated QA is rewritten from source
   annotations before flattening.
7. Confirm that unsupported structured generated QA is rejected from flattened
   trainer JSONL.
8. Confirm that review import applies decisions only to saved language records.
9. Confirm that rejected and needs-revision language rows remain auditable but
   do not enter flattened trainer rows.
10. Confirm that browser, server, and trainer validation all block stale,
    incomplete, unknown-status, or non-trainable instruction rows, including
    review rows whose QA id and image still match but whose question or answer
    text is no longer current.
11. Confirm that browser downloads block mismatched training/archive/review
    artifacts when their counts disagree with the instruction report.
12. Confirm that dense prompt box lists are representative while counts remain
    authoritative.
13. Confirm that loop detection, safe retry, fallback, and deterministic
    recovery are visible in audit artifacts.
14. Confirm that missing models are visually obvious and fail preflight unless
    downloads are explicitly allowed.
15. Run a small real-data pilot before treating any generated corpus as
    training-ready.

## Remaining Work Before Training Use

The implementation path is structurally wired and tested. The remaining work is
content and operational validation:

- Run a small real instruction-dataset pilot.
- Include dense scenes, empty-label images, multi-class scenes, and images with
  existing alternate captions.
- Inspect generated questions for visual answerability.
- Inspect generated answers for grounding and usefulness.
- Import review decisions.
- Re-export after review.
- Import the final JSONL into the trainer.
- Run a small fine-tuning dry run or at least a trainer-data dry run.
- Record any bad generated-QA patterns and update prompts or filters before
  larger-scale generation.

## Bottom Line

The application now has an end-to-end UI and backend path for creating
caption-derived VLM instruction datasets. It preserves source labels, keeps
generated language reviewable, exports trainer-ready rows, validates those rows
in the browser, validates them again on the server, validates them again in the
trainer, and provides runtime guardrails for long unattended caption jobs.

The design is intentionally cautious because training data needs provenance,
review state, and recoverability. The next step is not more wiring; it is a
real-data pilot and external review of generated-QA quality.
