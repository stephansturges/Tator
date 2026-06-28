# Qwen Caption Training Dataset External Review Handoff

Date: 2026-06-28

## Purpose

This document is a complete handoff for an external review of the Qwen captioning
and VLM-training-dataset work. It explains what was changed, why it was changed,
which invariants the implementation is meant to preserve, how the UI and backend
now behave, how the exported artifacts are shaped, what has been tested, and
what still needs real-data validation before relying on the generated corpus for
model training.

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

## Reader Contract

This document is written for an external implementation and training-data
review. It is intentionally explicit about boundaries because the system now
mixes four different kinds of evidence:

- trusted source labels
- generated captions
- generated visual QA
- deterministic metadata QA derived from labels

The implementation is considered correct only if those evidence types stay
separate all the way through UI display, backend persistence, archive export,
review import, and trainer import. A reviewer should treat any path that lets
generated language overwrite trusted source labels as a product regression.

The implementation is also intentionally dataset-neutral. It should be reviewed
as a reusable captioning and instruction-dataset workflow, not as a one-off
dataset script.

## Current Checkpoint

What is implemented in this checkpoint:

- A UI-visible one-click instruction-dataset generation path.
- Dataset-backed caption jobs that can produce caption rows plus generated
  visual QA rows.
- Optional deterministic metadata QA rows derived only from trusted labels.
- Per-image instruction archives.
- Candidate-level review JSONL export.
- Reviewed JSONL import that applies audit decisions back to saved caption and
  generated-QA metadata.
- Run-level instruction reports with corpus quality metrics.
- Training-readiness classification.
- Browser-side instruction export validation.
- Direct trainer import of the flat instruction JSONL row shape.
- Runtime hardening for prompt size, output-token overrides, loop detection,
  fallback, set-and-forget supervision, and model-download state.

The latest hardening pass specifically closed the review loop:

- the frontend can import a reviewed instruction JSONL artifact;
- the backend exposes a dataset caption review-import endpoint;
- imported decisions are persisted as review metadata only;
- `accepted`, `rejected`, and `needs_revision` affect subsequent readiness and
  export decisions;
- deterministic metadata QA review rows are skipped on import because those rows
  are rebuilt from source labels during export and are not saved language
  records.

What is intentionally not claimed at this checkpoint:

- The generated corpus is not declared training-grade without a real-data pilot
  and human review of generated QA.
- Review JSONL import applies decisions and notes only. It does not edit source
  labels, questions, answers, boxes, or final annotations.
- The implementation creates training data. It does not launch fine-tuning from
  the caption panel.
- Process-level Metal/GPU aborts are not catchable Python exceptions. The
  supported strategy is subprocess isolation, persisted artifacts, supervised
  restart, and resume.

## Reviewer Starting Point

Review the system as five linked contracts:

1. User workflow contract: an operator can launch a dataset-backed instruction
   dataset run from the caption panel without manually assembling prompts or
   conversion files.
2. Trust contract: source labels, generated language, deterministic metadata QA,
   and flattened trainer rows remain separate.
3. Export contract: every flattened row can be traced back to its image,
   question, answer source, validation status, review status, and archive row.
4. Runtime contract: large prompts, model loops, missing model downloads, and
   process-level failures are observable and routed through bounded recovery.
5. Training contract: the exported flat JSONL imports directly into the Qwen
   trainer and becomes image/question/answer conversation data.

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

## Design Decisions And Rationale

### Keep Captioning And Instruction Dataset Creation Separate

Captioning remains a normal annotation-assistance feature. Instruction dataset
creation is a related but distinct training-data feature. Keeping the paths
separate avoids making every caption run more expensive, prevents unexpected
training artifacts from appearing during ordinary captioning, and lets the UI
expose training-specific controls only when the operator requests them.

### Preserve Source Labels As Source Labels

The strongest invariant is that generated language never becomes trusted label
truth. Source annotations are computed from existing labels. Generated captions
and generated QA are language annotations. Deterministic metadata QA is allowed
only because it is computed directly from source labels, not hallucinated by the
VLM.

### Export Both Trainer Rows And Audit Rows

Flattened training rows must stay simple because the trainer should not need to
know the entire annotation archive schema. The archive and review rows carry the
extra audit detail. This gives downstream training a clean input while preserving
the evidence needed to debug, reject, or regenerate bad rows.

### Prefer Rejection Over Silent Guessing

When a generated row makes a structured claim that cannot be checked, the row is
rejected from flattened training output and kept in the archive. When the claim
can be checked, the final training answer is rewritten from trusted source
annotations and the candidate answer remains in metadata. This protects training
data quality while retaining useful review evidence.

### Make Set-And-Forget A First-Class Mode

Long dataset jobs cannot depend on a browser tab, a single Python process, or
manual recovery. Set-and-forget therefore uses backend jobs, subprocess
isolation, preflight checks, strict audits, restart/resume artifacts, health
gates, and visible recovery telemetry. Direct in-browser captioning remains a
diagnostic mode, not the durability baseline.

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
- `instruction_review_rows`
- `instruction_archive`
- `instruction_report`
- `instruction_summary`

## End-To-End Data Flow

The intended flow is:

1. The operator chooses a caption dataset and clicks **Create VLM training
   dataset**.
2. The frontend sends a dataset caption job request with instruction-dataset
   settings such as generated QA count, QA mix, answer format, source-context
   use, strict grounding, and export inclusion flags.
3. The backend validates and normalizes those settings.
4. For each image, the runner creates or reuses `caption0`.
5. When enabled, the runner performs an image-grounded generated-QA pass.
6. Generated QA candidates are parsed, structurally validated, deduplicated
   within the image, and persisted as instruction records.
7. Source annotations are assembled separately from dataset label evidence.
8. Export builds a per-image instruction archive that combines source
   annotations, language annotations, optional deterministic metadata QA, row
   rejection evidence, and flattened training rows.
9. The run-level report computes corpus quality metrics and training readiness.
10. The UI can download training JSONL, archive JSONL, review JSONL, and the
    report JSON.
11. A reviewer can edit review decisions externally and import the reviewed
    JSONL back into the dataset. The import persists only `accepted`,
    `rejected`, or `needs_revision` review metadata on caption or generated-QA
    records.
12. The next export reflects those review decisions in training readiness and
    excludes rejected or needs-revision language candidates from flattened
    trainer rows.
13. The trainer imports the flat instruction JSONL and converts each row into a
    Qwen conversation record.

At no step does generated language overwrite source labels or automatically
mutate final annotations.

## Implementation Status Matrix

| Capability | Status | Why it matters |
| --- | --- | --- |
| Existing caption-only export | Implemented and preserved | The training path must not weaken ordinary annotation-assistance captioning. |
| One-click instruction dataset job | Implemented | Operators can create caption plus QA training data from the UI without assembling scripts. |
| `caption0` row generation | Implemented | Every image can retain one broad descriptive training row. |
| Generated visual QA rows | Implemented | The system can create multiple image-grounded question/answer pairs beyond a single caption. |
| Deterministic metadata QA | Implemented, off by default | Label-derived QA can be added only when the operator explicitly wants code-generated metadata rows. |
| Source/generated separation | Implemented | Trusted labels remain separate from VLM-generated language. |
| Archive JSONL | Implemented | Reviewers can inspect how each image's training candidates were built. |
| Review JSONL export | Implemented | Candidate rows can be reviewed outside the app before training. |
| Review JSONL import | Implemented | Human decisions can be applied back to saved caption and generated-QA records. |
| Training-readiness status | Implemented | Exports can be blocked, warned, or marked ready based on row and review state. |
| Flat JSONL trainer import | Implemented | The Qwen trainer can consume exported rows directly. |
| Prompt-size adaptation | Implemented | Dense label scenes no longer require dumping every box into a prompt. |
| Loop detection and retry | Implemented | Repetition stalls become observable recovery events instead of silent hangs. |
| Set-and-forget supervision | Implemented as the durable default | Long dataset jobs are designed around backend jobs, resume, and recovery rather than browser babysitting. |
| Real-data training-quality certification | Still required | Passing structural tests does not prove that generated QA is good enough for fine-tuning. |

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

## Corpus Quality Metrics

The instruction archive and instruction report include a
`corpus_quality_metrics` block. It is intended to help reviewers decide whether
an exported instruction corpus is ready for training or needs another generation
or review pass.

The metrics include:

- image-level training-row coverage
- generated-QA image coverage
- generated-QA acceptance and rejection rates
- unique generated-question count
- global duplicate-question count and rate
- per-image duplicate-question count and rate
- generated-question diversity ratio
- structured-rewrite count and rate
- duplicate image/question rejection count
- source-validated training-row count and rate
- source classes present in trusted labels
- source classes covered by flattened training rows
- source-class coverage rate
- flattened answer-format distribution

The source-grounding metrics intentionally separate two concepts:

- `source_validated_training_row_count` means the row was validated against
  source annotations or carries source fields.
- `source_classes_covered_by_training_rows` means flattened rows carry
  class-specific or class-list source fields that cover trusted source classes.

This distinction prevents broad source-validation metadata from being confused
with true class coverage.

## Training Readiness

The instruction report includes a `training_readiness` block with one of three
statuses:

- `ready`: selected rows pass structural checks, no selected language rows are
  waiting for manual acceptance, and quality gates have no warnings.
- `needs_review`: rows can be exported, but selected caption0 or generated-QA
  rows still need human review, or corpus-quality gates raise warnings such as
  duplicate questions, low generated-question diversity, high generated-QA
  rejection rate, or low source-class coverage.
- `blocked`: the export should not be used for training, for example when there
  are no images, no selected training rows, or a selected row was rejected or
  marked as needing revision by manual review.

The browser validates this readiness block before writing instruction JSONL.
Blocked exports are refused. `needs_review` exports are still downloadable, but
the UI status is a warning instead of a training-ready pass.

## Review Rows

The export payload also includes `instruction_review_rows`, a candidate-level
JSONL review artifact for human audit before training. Each row records:

- image path, split, row origin, row type, and stable QA id
- question, candidate answer, and selected training answer when applicable
- whether the candidate was selected for flattened training JSONL
- whether manual review is required
- validation status, review status, rejection reasons, and source fields
- source-label summary with counts, visible classes, annotation count, and
  uncertainty count
- blank `review_decision` and `review_notes` fields for downstream review

This artifact is deliberately separate from `instruction_training_rows`: editing
or annotating review rows must not mutate trusted source annotations or silently
change what is exported for training.

The UI also provides **Import reviewed JSONL**. The import accepts the review
JSONL artifact after a reviewer fills `review_decision` with `accepted`,
`rejected`, or `needs_revision`, then applies only review metadata and notes to
matching saved caption or generated-QA records. It intentionally skips
deterministic metadata QA rows because those rows are derived from source labels
at export time rather than persisted language records.

The review import is deliberately conservative:

- It accepts JSON arrays, JSON objects containing row arrays, and JSONL files.
- It refuses to apply rows whose embedded dataset id does not match the selected
  dataset.
- It ignores blank or unknown decisions instead of inventing a review result.
- It applies decisions only to matching caption0 or generated-QA records.
- It records reviewer, notes, source row metadata, and decision timestamps.
- It does not change questions, answers, source labels, boxes, image paths, or
  selected final annotations.
- It reports how many rows were received, applied, created as caption review
  records, or skipped.
- Rejected and needs-revision caption0 or generated-QA candidates remain in the
  archive and review rows, but they are not flattened into trainer rows.

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
  - instruction review JSONL
  - instruction report
- import for reviewed JSONL decisions
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

## Why The Implementation Is Shaped This Way

The workflow is intentionally split into generation, archive, review, import,
and training-import phases. That split adds some surface area, but it solves the
training-data problems that a single "caption all images" button cannot solve:

- A training row needs provenance, not only text.
- A generated QA answer may be useful even when it is not safe to train on yet.
- Label-derived facts should be reproducible from labels rather than copied from
  generated prose.
- A human reviewer needs to reject or accept rows without editing source
  annotations.
- The trainer needs a small stable row shape, while reviewers need a richer
  archive.
- Long caption jobs need durable recovery and audit telemetry because GPU
  failures can abort the backend process.

The result is a two-level export contract. The flat instruction JSONL is the
trainer input. The archive, review rows, and report are the evidence layer that
lets reviewers decide whether that trainer input should be trusted.

## Operator Workflows To Review

### Ordinary Captioning

Use this for annotation assistance. It appends caption variants, preserves
existing primary captions unless promotion is explicit, and exports caption-only
artifacts.

### Instruction Dataset Creation

Use this to create training rows. It can include caption0, generated QA, and
optional deterministic metadata QA. It produces training JSONL plus archive,
review, and report artifacts.

### Review Before Training

Use review JSONL and the instruction report to inspect selected and rejected
candidate rows. Import reviewed JSONL after decisions are filled in so readiness
can reflect the actual review outcome rather than only the initial generated
state.

### Fine-Tuning Dry Run

Use the Qwen training loader to import the exported flat JSONL and verify that
image paths resolve and conversations are constructed. This validates the data
shape, but it is not a substitute for content review.

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
  - review-decision metadata preserved from exported rows

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

- `docs/qwen_caption_instruction_dataset_partner_handoff.md`
- `docs/qwen_caption_instruction_dataset_hardening_report.md`
- `docs/qwen_caption_prompt_stack.md`
- `docs/qwen_caption_ui_scenarios.md`
- this handoff document

## Validation Evidence

Validation performed for the implementation described in this handoff:

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
137 passed
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
caption readiness: 30 pass, 1 warning, 0 fail
no console errors
no failed requests
no bad HTTP responses
no clipped caption action buttons
instruction import button present
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
   - review JSONL export and import buttons
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
- Import the reviewed JSONL decisions and confirm that readiness changes as
  expected.
- Decide acceptance thresholds for generated QA rows before training.
- Run an actual small fine-tuning dry run with the exported rows, not only the
  loader import smoke.

Recommended pilot shape:

- at least one dense labeled scene
- at least one image with no labels
- at least one image with multiple object classes
- at least one image with existing alternate captions
- at least one image whose labels include small objects near crop boundaries
- at least one intentionally missing or empty label file, to verify that missing
  source evidence is represented as audit state rather than guessed content

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
- Is the current review JSONL schema sufficient for external audit tools?
- Should review import remain metadata-only, or should a future guarded workflow
  also support edited replacement answers and edited questions?
- Should `needs_review` block training JSONL download in strict set-and-forget
  mode, or is the current warning sufficient?
- Which quality gates should be hard blockers for the first real fine-tuning
  run: duplicate-question rate, source-class coverage, generated-QA rejection
  rate, review coverage, or all of them?
