# Qwen Caption Training Dataset Complete Partner Packet

Date: 2026-06-29

## Purpose

This document is the complete partner-facing explanation of the caption-based
VLM training-dataset work in this repository. It explains what was built, why it
was built this way, which data contracts matter, how an operator uses the UI,
how generated rows are validated and reviewed, and what still needs real-data
pilot validation before a generated corpus should be used for fine-tuning.

The implementation is intentionally dataset-neutral. It does not rely on
project-specific dataset names, class names, or private evaluation assumptions.

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

The result is a conservative pipeline: generate candidates, archive evidence,
validate aggressively, let humans review generated language, then export only
trainable rows.

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
| Review import | `localinferenceapi.py` and `api/datasets.py` | Applies metadata-only review decisions to saved caption0/generated-QA records and rejects malformed, stale, ambiguous, duplicate, or wrong-dataset packets |
| Export API | `api/datasets.py` | Exposes caption exports and the optional `require_ready_instruction_export` server-side gate |
| UI workflow | `ybat-master/ybat.js` and `ybat-master/ybat.css` | Exposes controls, validates exports before download, formats operator errors, imports reviewed JSONL, and keeps the panel usable at narrow widths |
| Trainer import | `tools/qwen_training.py` | Imports flat image/question/answer rows, preserves metadata, resolves image paths, and rejects non-trainable rows |
| Runtime hardening | `localinferenceapi.py`, runner tooling, and caption docs | Handles prompt pressure, representative box lists, output-loop detection, recovery, set-and-forget supervision, and progress artifacts |

The implementation keeps UI checks and backend checks aligned intentionally.
Browser validation gives immediate operator feedback; backend validation is the
authority for API/script use; trainer validation is the final guard against
hand-edited files.

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

- caption0 or generated-QA review rows missing dataset identity, even before a
  reviewer fills a decision
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
./.venv-macos/bin/python -m py_compile localinferenceapi.py
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
205 passed
```

Additional focused validation recorded in the supporting hardening docs covers:

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
- trainer import of flat rows
- trainer rejection of non-trainable rows
- rendered UI smoke for visible controls and unclipped caption actions
- restricted project-name scan across code, docs, tests, tools, UI, and backend

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
