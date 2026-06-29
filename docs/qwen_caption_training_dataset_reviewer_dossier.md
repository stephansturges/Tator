# Qwen Caption Training Dataset Reviewer Dossier

Date: 2026-06-29

## Purpose

This dossier explains what was built in the caption-based VLM training-dataset
workflow, why it was built this way, how an operator should use it, what
artifacts it produces, which safety boundaries are enforced, and what still
needs pilot validation before a generated corpus is used for fine-tuning.

It is written for an external technical review. The implementation is
dataset-neutral and intentionally avoids private dataset names, organization
names, and target-specific class assumptions.

## Executive Summary

The captioning panel now supports two separate workflows:

- **Caption assistance**: generate, store, promote, and export image captions.
- **VLM training-dataset creation**: generate a flat image/question/answer
  training corpus plus archive, review, and report artifacts.

The training-dataset workflow is launched from the caption panel with
**Create VLM training dataset**. For each image, the run can create:

- one broad image description row, called `caption0`;
- a configurable number of VLM-generated visual question/answer rows;
- optional deterministic metadata QA rows derived from trusted labels.

The core design principle is separation of evidence:

- trusted labels and deterministic geometry become `source_annotations`;
- model-written caption and QA text become `language_annotations`;
- deterministic label-derived QA is produced by code;
- trainer JSONL rows are selected exports, not the source of truth.

The result is an auditable training-data path rather than a simple JSONL dump.
Generated language can become a training candidate, but it does not become a
label, count, class list, geometry fact, or final annotation.

## What Was Built

### 1. A Distinct Instruction-Dataset UI Path

The caption panel now exposes a separate training-data action:
**Create VLM training dataset**.

The operator can configure:

- generated QA rows per image;
- generated QA mix;
- generated answer format;
- whether `caption0` is included in trainer JSONL;
- whether generated QA is included in trainer JSONL;
- whether deterministic metadata QA is included;
- whether source labels are passed as read-only grounding context;
- whether strict QA grounding is enforced;
- whether trainer JSONL requires a ready report before download.

This separation matters because normal captioning and training-data production
have different contracts. Captioning answers "describe this image." Training
dataset creation answers "produce trainable rows and prove how each row was
created, validated, reviewed, and selected."

### 2. Multi-Prompt Rows Per Image

The implementation no longer treats a single caption as the whole training
signal. A run can produce:

- `caption0`: a broad image-level caption row;
- generated QA: model-written visual questions and answers;
- deterministic metadata QA: optional code-generated rows from source labels.

Generated QA count is bounded from `0` to `20`. A value of `0` produces a
caption0-only instruction export. Deterministic metadata QA is off by default
because it changes the corpus mix from image-language generation to explicit
label-derived QA.

### 3. Four Export Artifacts

The workflow exports four coordinated artifacts:

- `caption_instruction_training.jsonl`: flat trainer rows with `image_path`,
  `question`, `answer`, and optional metadata.
- `caption_instruction_archive.jsonl`: one construction archive row per image,
  including source summaries, language candidates, deterministic QA, rejected
  rows, selected rows, and export metadata.
- `caption_instruction_review.jsonl`: candidate-level review rows for human
  accept/reject/needs-revision decisions.
- `caption_instruction_report.json`: run-level readiness, quality,
  consistency, row-count, provenance, and blocking-reason metrics.

These files intentionally have different jobs. The trainer file is compact and
directly loadable. The archive is evidence. The review file is a human-audit
queue. The report explains whether the corpus is ready, blocked, or still needs
review.

### 4. Review Export And Import

Generated-language rows can be reviewed outside the application. The user
downloads review JSONL, fills `accepted`, `rejected`, or `needs_revision`
decisions, and imports the reviewed file.

Review import is metadata-only:

- it can apply review decisions and notes to matching saved caption0 and
  generated-QA records;
- it cannot edit labels, boxes, image paths, generated questions, generated
  answers, deterministic QA, or final annotations.

Wrong-dataset rows, stale text, duplicate targets, unsupported row origins,
missing identities, malformed booleans, unsupported decisions, oversized text
fields, and packets with no actionable decisions fail closed before any review
metadata is written.

### 5. Trainer Import Compatibility

The Qwen training loader can import the flat exported instruction rows
directly. It validates:

- required image, question, and answer fields;
- instruction metadata and provenance;
- validation and review states;
- JSON answer shape for JSON-formatted rows;
- duplicate canonical image/question pairs;
- resolvable image aliases.

The trainer loader is intentionally a final safety boundary. Even if a file was
hand-edited after export, the trainer refuses rows that are rejected,
needs-revision, missing provenance, malformed, duplicate, or not tied to a real
image.

## Why These Choices Were Made

### A Single Caption Is Too Thin For VLM Fine-Tuning

Fine-tuning benefits from multiple grounded prompts about the same image:
scene-level description, object relationships, class presence, counts, spatial
facts, and short targeted answers. The new path keeps `caption0` but adds
configurable generated QA and optional deterministic metadata QA.

### Generated Text Must Not Become Source Truth

A VLM answer can be useful language, but it is not a label file. The archive
keeps trusted source facts and generated language in separate fields. When a
generated row asks for a count, class list, presence, absence, or simple
spatial fact that can be checked against trusted labels, the exported training
answer is rewritten from the source annotations. If the claim cannot be
checked, the row is preserved for audit but rejected from trainer output.

### A Flat Trainer File Cannot Explain Itself

The fine-tuning file should stay flat and simple. It should not carry the
entire source-label universe as hidden context. The archive and report carry
the explanation. This keeps the trainer input clear while preserving enough
evidence for reviewers to audit every row.

### Long Dataset Jobs Need Set-And-Forget Behavior

Dataset-scale captioning can involve many images, crop windows, full-image
composition, generated QA, archive construction, and validation passes. Browser
tab state is not reliable enough for that workload. The durable path is a
backend job with progress, attach/recover behavior, launch-failure visibility,
and active-job locks around exports and mutations.

### Prompt Size Has To Be Managed Explicitly

Dense scenes can contain enough boxes to create oversized prompts. Counts stay
authoritative, but detailed box lists become representative spatial subsets
when they get too large. Prompt wording makes clear that omitted boxes are not
absent objects. Prompt-size telemetry also informs automatic output budgeting.

### Thinking-Capable Models Need Separate Budget Semantics

Some models produce far more internal or verbose output than short-caption
models. The UI and backend now separate automatic model-aware defaults from
explicit numeric caps. Auto can choose a high enough default for a
thinking-capable model and reduce it under prompt pressure. A user-entered
number remains a hard override after validation.

### Runtime Loops Must Be Treated As Failures

Repeated punctuation or repeated tokens are not valid captions. Streamed output
is inspected while generation is running. When a loop is detected, the system
records recovery, trims the repeated fragment for diagnostics, retries with
safer decoding, and can unload/reload the runtime when required. Recovery is
visible instead of being mistaken for a normal caption.

## Operator Workflow

The intended UI flow is:

1. Select a caption dataset.
2. Keep **Set-and-forget backend run** enabled.
3. Confirm that selected models are local or deliberately allow downloads.
4. Configure generated QA count, mix, answer format, source-label context,
   strict grounding, and row-family inclusion.
5. Click **Create VLM training dataset**.
6. Monitor progress or attach/recover after a browser reload.
7. Download the instruction report, archive, review JSONL, and trainer JSONL.
8. Review generated-language rows in the review JSONL.
9. Import reviewed decisions.
10. Re-export trainer JSONL with the ready-report gate enabled.
11. Run the trainer loader or a small fine-tuning dry run.

Ordinary **Caption image**, **Caption next N**, and **Caption all images** remain
captioning workflows. They do not silently become training-dataset workflows.

## Artifact Contract

### Trainer JSONL Row

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

The model trains on `image_path`, `question`, and `answer`. Metadata exists for
audit, filtering, lineage, and validation; it is not hidden prompt text.

### Archive Row

Each archive row preserves:

- canonical image identity;
- source annotations and provenance;
- caption0;
- generated QA candidates;
- deterministic metadata QA when enabled;
- selected flattened rows;
- rejected rows and rejection reasons;
- export validation and artifact-consistency metadata.

### Review Row

Each review row includes:

- image path;
- row origin and row type;
- stable QA id;
- question;
- candidate answer;
- selected training answer when applicable;
- validation state;
- review state;
- selected-for-training flag;
- rejection reasons;
- source summary;
- blank review decision and notes fields.

Reviewers fill the decision field. The import path applies that decision as
metadata after identity and staleness checks.

### Report

The report records:

- image count;
- selected trainer-row count;
- generated-QA candidate count;
- accepted/rejected/generated row counts;
- deterministic QA count;
- row-type distribution;
- split counts;
- quality metrics;
- export validation;
- artifact consistency;
- training readiness;
- blocking reasons and required actions.

## Safety Boundaries

The workflow fails closed at multiple layers:

- The UI refuses instruction jobs when all trainable row families are disabled.
- The UI validates trainer, archive, review, and report artifacts before
  download.
- The backend validates strict trainer exports for API/script callers.
- Review import validates dataset identity, row identity, text freshness, row
  shape, duplicate targets, and actionable decisions.
- Active caption jobs block caption exports, instruction exports, report
  downloads, review import, caption mutations, glossary writes, dataset ZIP
  downloads, same-dataset caption jobs, and dataset deletion.
- Manual caption archive buttons are disabled while caption/instruction jobs
  are mutating the archive and also perform action-time backend checks.
- The trainer loader rejects stale or non-trainable rows before fine-tuning.

The important architectural point is that no single UI check is trusted as the
only guard. Browser, backend, and trainer validations overlap deliberately.

## UI/UX Hardening

Recent hardening focused on making the workflow reliable for operators rather
than only testable by developers:

- set-and-forget backend jobs are the default durable path;
- attach/recover controls remain available after reloads;
- missing model options are visibly red, local options remain normal;
- output-token Auto mode is model-aware and prompt-pressure-aware;
- dense box prompts preserve counts while bounding box-list size;
- repeated-token output triggers recovery instead of being accepted;
- export/import buttons disable while the caption archive is changing;
- manual caption add/update/primary/delete controls also disable while the
  archive is changing;
- action handlers repeat busy checks in case UI state went stale;
- backend launch failures surface in caption status and health text;
- operator-facing errors are formatted instead of leaking raw internal codes.

## Validation Evidence

The current implementation is covered by focused backend, trainer, UI contract,
and smoke tests. The main validation areas are:

- instruction-dataset job construction;
- generated-QA parsing and validation;
- deterministic metadata QA;
- source/generated separation;
- artifact consistency;
- strict ready-report export;
- review JSONL export/import;
- duplicate and stale-review rejection;
- caption archive mutation locks;
- dataset download and deletion busy guards;
- trainer flat-row import;
- UI layout and button-state contracts;
- unattended caption-job supervision and audit tooling.

The canonical focused validation command used for the current hardening slice
is:

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

Syntax and hygiene checks for this area are:

```bash
node --check ybat-master/ybat.js
./.venv-macos/bin/python -m py_compile tests/test_labeling_panel_layout_contract.py
uvx ruff check tests/test_labeling_panel_layout_contract.py --select E9,F63,F7,F82
git diff --check
```

## What Is Proven Now

The current implementation proves the product path and the structural safety
contract:

- the UI can launch a caption-derived instruction-dataset job;
- the backend can produce trainer, archive, review, and report artifacts;
- generated language and trusted source facts stay separated;
- exports are validated as a coherent artifact set;
- review import is metadata-only and fail-closed;
- trainer import rejects non-trainable rows;
- long-running jobs are guarded against stale exports and concurrent mutation;
- the major UI controls are present, readable, and locked during archive
  mutation.

## What Is Not Yet Proven

The current implementation does not certify a generated corpus as production
training data. That still requires:

- a real-data pilot across representative images;
- manual generated-QA review;
- reviewed decision import;
- trainer JSONL re-export with ready-report gating enabled;
- trainer loader or fine-tuning dry run;
- measurement of generated-QA usefulness, rejection rate, degraded-recovery
  rate, and reviewer throughput.

The correct conclusion is: the software path is implemented and guarded well
enough for an external technical review and a controlled pilot. Corpus quality
must still be established from pilot artifacts and human review.

## Reviewer Checklist

An external reviewer should confirm:

- the row-family controls produce the expected trainer/archive/review/report
  outputs;
- empty label files produce empty source evidence, not generated label facts;
- generated QA with unsupported structured claims is rejected from trainer
  output but kept in the archive;
- review import applies only decisions and notes;
- same-dataset active caption jobs block export/import/mutation/deletion paths;
- strict trainer export refuses non-ready reports;
- trainer import rejects hand-edited stale or malformed rows;
- a pilot packet can be traced from trainer row to review row to archive row to
  report metrics.

## Related Documents

- `docs/qwen_caption_training_dataset_complete_partner_packet.md`
- `docs/qwen_caption_instruction_dataset_hardening_report.md`
- `docs/qwen_caption_prompt_stack.md`
- `docs/qwen_caption_ui_scenarios.md`
- `docs/qwen_caption_training_dataset_external_review_handoff.md`
