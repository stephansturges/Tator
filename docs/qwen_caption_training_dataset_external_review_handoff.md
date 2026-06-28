# Qwen Caption Training Dataset External Review Handoff

Date: 2026-06-28

## Purpose

This document is a complete handoff for an external review of the Qwen captioning
and VLM-training-dataset work. It explains what was changed, why it was changed,
which invariants the implementation is meant to preserve, how the UI and backend
now behave, what has been tested, and what still needs real-data validation
before relying on the generated corpus for model training.

The work is intentionally dataset-neutral. The implementation must support
general annotation-assistance workflows and must not encode project-specific
dataset names, class assumptions, or evaluation shortcuts.

## Executive Summary

The captioning stack now has two distinct but connected paths:

- The existing caption path still creates and exports image captions.
- A new instruction-dataset path creates training-oriented rows:
  - one broad caption row, `caption0`
  - optional VLM-generated visual question/answer rows
  - optional deterministic metadata QA rows derived only from trusted labels
  - a per-image construction archive for audit
  - flattened trainer rows that the Qwen trainer can import directly

The main architectural decision is strict separation of trusted source labels
from generated language. Source annotations remain source annotations. Generated
captions and generated QA are language annotations. If a generated answer makes a
structured claim that can be checked against labels, the exporter either rewrites
the final flattened answer from trusted source data or rejects the row. Rejected
rows remain in the archive for audit but do not enter the training JSONL.

The UI now exposes a one-click **Create VLM training dataset** workflow with
controls for generated QA count, generated QA mix, answer format, deterministic
metadata QA, source-context use, and strict grounding. The browser also validates
instruction JSONL before download.

The runtime was hardened around the failure modes observed during captioning:
oversized prompts, confusing output-token settings, repeated/refinement passes,
model-output loops, long-running or stalled worker calls, Metal/GPU process
failure, and unattended batch recovery. The current implementation includes
prompt-size accounting, representative spatial box subsets, loop detection,
deterministic recovery paths, set-and-forget controls, readiness checks, and UI
model availability coloring.

## Why This Was Needed

The original caption-only workflow was not sufficient for creating a robust VLM
fine-tuning dataset. It could produce a caption, but it did not preserve enough
provenance to answer:

- Which facts came from trusted labels?
- Which facts came from the VLM?
- Which rows are safe to flatten into training data?
- Which rows should be archived but excluded?
- Can the exported rows be imported by the actual training loader without a
  manual conversion step?
- Can an operator run this over a dataset without babysitting every case?

Several concrete fragility points also had to be addressed:

- Prompt logs showed confusing token-budget behavior, including high
  `max_new_tokens` values that were appropriate for thinking-capable models but
  unclear to users.
- Detailed/windowed captioning emitted many intermediate outputs. The UI and
  logs needed to distinguish useful audit data from the final caption.
- Dense label prompts could dump too many boxes into a single prompt.
- Some model calls looped on punctuation or repeated tokens, causing apparent
  stalls.
- Metal/GPU failures could kill the backend process.
- The UI did not make it clear enough which models were local and which needed a
  download.
- The generated instruction rows were initially a flat browser-export shape, but
  the training loader consumed conversation JSONL. That would have required
  manual conversion and was not acceptable for a training path.

## Current Product Behavior

### Caption-Only Path

The existing caption path remains available:

- single-image caption generation
- batch caption generation
- alternate caption storage
- primary-caption promotion
- caption audit JSONL export
- grouped caption JSON export
- caption-only VLM JSONL export

Generated captions append as alternate caption records by default. They only
become primary if the operator explicitly enables primary promotion or if the
image does not already have a primary caption.

### Instruction-Dataset Path

The instruction-dataset workflow is a separate dataset-backed mode:

- UI button: **Create VLM training dataset**
- backend flag: `instruction_dataset=true`
- default generated QA rows per image: `8`
- allowed generated QA range: `0..20`
- default includes:
  - caption0 rows
  - generated visual QA rows
  - read-only source-label context
  - strict grounding
- default excludes:
  - deterministic metadata QA

The operator can control:

- generated QA rows per image
- generated QA mix: balanced, scene-level, object-focused, caption-variant
- generated answer format: natural text or JSON
- include caption0 in flattened training rows
- include generated QA in flattened training rows
- include deterministic metadata QA in flattened training rows
- pass source annotations to the generator as read-only context
- strict grounding for generated QA

The instruction mode exports:

- `instruction_training_rows`
- `instruction_archive_rows`
- `instruction_archive`
- `instruction_report`
- `instruction_summary`

## Data Contracts

### Source Annotations

Source annotations are derived from real label evidence only. They are not
generated by the VLM. In the instruction archive they live under:

```json
"source_annotations": {
  "format": "tator_source_annotations_v1",
  "status": "ok",
  "object_counts": {},
  "visible_classes": [],
  "annotations": [],
  "bbox_instances": [],
  "bbox_geometry": {},
  "spatial_facts": [],
  "uncertainty": [],
  "field_provenance": {}
}
```

Important behavior:

- If a label file is missing, source status records the missing label condition.
- If a caption or QA record refers to an image no longer present in the dataset
  manifest, source status becomes `source_manifest_row_missing`.
- Missing or non-manifest source data prevents flattening into trainer rows.
- The row remains visible in the archive for audit.

### Language Annotations

Generated and saved language lives separately:

```json
"language_annotations": {
  "caption0": {},
  "generated_qa_pairs": []
}
```

`caption0` is the broad description row. Generated QA rows are VLM-created
questions and answers. Neither field is allowed to become source metadata.

### Deterministic Metadata QA

Deterministic metadata QA is optional and off by default. When enabled, it is
computed from trusted source annotations only. It can generate rows for:

- visible class list
- object-count schema
- per-class count
- positive presence
- negative presence
- simple bbox-derived spatial facts

These rows are marked `machine_validated`.

### Flattened Training Rows

Flattened rows use the simple browser/export shape:

```json
{
  "image_path": "train/frame.jpg",
  "question": "Describe this image in detail.",
  "answer": "A grounded answer.",
  "metadata": {
    "qa_id": "stable-row-id",
    "row_type": "caption0",
    "answer_source": "caption_record",
    "answer_format": "natural",
    "validation_status": "accepted",
    "review_status": "unreviewed",
    "source_archive": "tator_caption_instruction_archive_v1"
  }
}
```

The Qwen trainer now imports this flat row shape directly. It normalizes each
row into the two-turn image/question/answer conversation format used by
fine-tuning:

```json
{
  "image": "train/frame.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe this image in detail."},
    {"from": "gpt", "value": "A grounded answer."}
  ]
}
```

This removes the need for a manual conversion step between browser export and
training import.

## Validation And Rejection Rules

Rows can remain in the archive while being excluded from flattened training
JSONL. This is intentional: auditability and trainability are separate.

Rows are rejected from flattened training output when:

- image path, question, or answer is missing
- image/question pair is a duplicate
- JSON answer formats contain invalid JSON
- upstream validation marked the row rejected or failed
- generated QA asks for unavailable or sensitive context
- generated QA makes a structured claim that cannot be checked or rewritten from
  trusted source annotations
- caption0 contains explicit numeric object-count claims that contradict trusted
  source labels
- caption0 contains structured claims but source annotations are unavailable
- source manifest row is missing

Supported structured generated QA can be rewritten during export:

- class-list questions are rewritten from `source_annotations.visible_classes`
- count questions are rewritten from `source_annotations.object_counts`
- presence questions are rewritten from `source_annotations.object_counts`
- simple spatial questions are rewritten from `source_annotations.spatial_facts`

When a rewrite happens, the original generated answer is preserved as candidate
metadata for audit, but the flattened training answer comes from the trusted
source annotation.

## UI/UX Work

The caption panel now includes:

- visible instruction-dataset controls
- explicit generated QA count with `0..20` bounds
- generated QA mix selector
- generated answer format selector
- toggles for caption0, generated QA, deterministic metadata QA, source-context
  use, and strict grounding
- separate downloads for:
  - instruction JSONL
  - instruction archive
  - instruction report
- readiness check button and rendered readiness results
- set-and-forget defaults and backend supervision status text
- model select styling that marks download-needed models in red and local models
  in the normal local color

The instruction help text states that generated QA is stored in a separate
instruction archive and never becomes source annotations.

The UI validates instruction JSONL before writing a file:

- missing image path
- blank question
- blank answer
- invalid JSON for JSON row types
- rejected generated QA
- duplicate image/question rows

Rendered browser smoke verifies that the instruction controls are visible,
defaults are correct, action buttons do not clip text, and no console or network
errors occur.

## Runtime Hardening Work

### Token Budgets

The stack distinguishes user-facing output-token controls from internal model
generation budgets. Thinking-capable models may need larger output budgets
because internal reasoning can consume far more tokens than concise caption
models. The implementation keeps high defaults where needed but preserves user
override paths.

### Prompt Size And Box Lists

Dense label contexts no longer imply dumping every box into the prompt. Counts
remain authoritative, while the prompt can use a representative spatial subset
of boxes for layout grounding. Prompt previews report estimated prompt budget
metadata so oversized prompts can be detected and adapted.

### Loop And Stall Handling

The caption runner detects repeated output loops, trims looped output, and can
retry with safer decoding settings. Loop/recovery events are tracked in audit
summaries so batch health can be measured instead of guessed.

### Set-And-Forget Mode

Set-and-forget mode is designed for long unattended dataset jobs. It includes:

- persisted backend jobs
- attach/recover controls in the UI
- auto-resume settings
- failure-rate and quality-rate gates
- loop-recovery gates
- watchdog and supervisor checks
- readiness/audit tooling

Set-and-forget is not treated as "manual recover only"; the goal is an operator
workflow that can run without constant attention while still preserving explicit
failure evidence.

### Metal/GPU Failure Handling

Metal/GPU faults can abort the Python process. The app-level hardening therefore
does not assume every failure is catchable inside Python. The documented
operational mode uses a backend launcher or process supervisor for restart, and
the UI reports whether crash-restart supervision is advertised.

## Files Changed Or Added

Core backend and export logic:

- `localinferenceapi.py`
  - instruction archive construction
  - source/generated separation
  - deterministic metadata QA
  - generated QA validation and rewrite
  - caption0 exact-count validation
  - instruction export summaries

Request schema:

- `models/schemas.py`
  - instruction-dataset settings normalization
  - generated QA bounds
  - set-and-forget caption controls

Runner and benchmark tooling:

- `tools/run_qwen_caption_flow_benchmark.py`
  - generated QA pass
  - prompt budget accounting
  - loop/recovery event reporting
  - set-and-forget audit support

Training loader:

- `tools/qwen_training.py`
  - direct import of flat instruction rows
  - conversion to Qwen conversation entries
  - preservation of row metadata

UI:

- `ybat-master/ybat.html`
- `ybat-master/ybat.js`
- `ybat-master/ybat.css`
  - instruction-dataset controls
  - export validation
  - readiness checks
  - model availability coloring
  - layout fixes for caption actions

Docs:

- `docs/qwen_caption_instruction_dataset_hardening_report.md`
- `docs/qwen_caption_prompt_stack.md`
- `docs/qwen_caption_ui_scenarios.md`
- this handoff document

## Validation Evidence

The latest pushed implementation is:

- `2827213 Accept flat caption instruction rows in Qwen trainer`

The previous pushed hardening checkpoint was:

- `2c7890d Validate caption instruction exports end to end`

Validation performed:

```bash
./.venv-macos/bin/python -m pytest \
  tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_imports_flat_question_answer_rows \
  tests/test_qwen_caption_dataset_job.py::test_caption_instruction_training_rows_import_into_qwen_trainer \
  -q
```

Result:

```text
2 passed
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
130 passed
```

Syntax and formatting checks:

```bash
./.venv-macos/bin/python -m py_compile \
  models/schemas.py \
  tools/run_qwen_caption_flow_benchmark.py \
  localinferenceapi.py \
  api/datasets.py \
  tools/run_qwen_caption_ui_smoke.py \
  tools/qwen_training.py

node --check ybat-master/ybat.js
git diff --check
```

Rendered UI smoke:

```bash
./.venv-macos/bin/python tools/run_qwen_caption_ui_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --out-json tmp/qwen_caption_ui_smoke_report.json \
  --screenshot tmp/qwen_caption_ui_smoke.png
```

Result:

```text
ok=true
caption readiness: 28 pass, 2 warnings, 0 fail
no console errors
no failed requests
no bad HTTP responses
no clipped caption action buttons
```

Restricted project-name scan:

```bash
rg -n "<restricted project-name pattern>" --glob '!**/.git/**' .
```

Result:

```text
no matches
```

## Suggested External Review Procedure

1. Review the data-contract separation first:
   - `source_annotations`
   - `language_annotations`
   - `deterministic_metadata_qa_pairs`
   - `training_rows`

2. Confirm that generated language never populates trusted source fields.

3. Inspect the rejection behavior:
   - missing label source
   - missing manifest row
   - duplicate image/question pair
   - invalid JSON answer
   - unsupported structured generated claim
   - caption0 count contradiction

4. Confirm that supported structured generated QA is rewritten from source
   annotations before flattening.

5. Confirm that rejected rows remain auditable in the archive but do not appear
   in flattened trainer rows.

6. Run the focused test suite listed above.

7. Run the rendered UI smoke against a live backend and inspect:
   - instruction-dataset controls
   - generated QA defaults
   - download buttons
   - readiness output
   - model availability coloring
   - button clipping

8. Create a small real instruction-dataset pilot with:
   - one dense labeled scene
   - one image with no labels
   - one image with multiple object classes
   - one image with existing alternate captions

9. Manually inspect the generated QA rows before training:
   - Are questions visually answerable?
   - Are answers grounded?
   - Are count claims correct?
   - Are row types and answer formats correct?
   - Are uncertain or unsupported claims rejected?

10. Import the exported instruction JSONL into the Qwen trainer and verify that
    image paths resolve and rows are converted to conversations.

## Remaining Work Before Treating The Corpus As Training-Ready

The implementation path is now wired and tested, but corpus quality still needs
real-data validation.

Remaining work:

- Run a small real VLM instruction-dataset pilot.
- Manually review generated QA quality for grounding and usefulness.
- Add corpus-level metrics:
  - generated QA diversity
  - rejection rate
  - duplicate question rate
  - class coverage
  - image-context coverage
  - structured rewrite rate
- Decide acceptance thresholds for generated QA rows before training.
- Run an actual small fine-tuning dry run with the exported rows, not only the
  loader import smoke.
- Add a review artifact that records manual QA decisions for the pilot corpus.

## Important Non-Goals

- This work does not automatically mutate final annotations.
- This work does not treat generated language as source label truth.
- This work does not claim the generated corpus is high quality without manual
  review and pilot metrics.
- This work does not encode dataset-specific names or assumptions.
- This work does not make the backend immune to process-level Metal/GPU aborts;
  that class of failure is handled by supervised restart and persisted job
  recovery.

## Review Questions For The External Team

- Are the flattened row fields sufficient for downstream training and auditing?
- Should deterministic metadata QA remain off by default?
- Are the generated QA mix options enough, or should they be replaced with a
  more explicit row-type budget?
- Should structured generated QA always be rewritten from source labels when
  possible, or should candidate and rewritten answers both be exported to a
  separate comparison file?
- What corpus-level acceptance thresholds should block training?
- How much manual review is required before a generated instruction dataset is
  considered training-grade?
