# Qwen Caption Instruction Dataset External Team Handoff

Date: 2026-06-29

## Start Here

For a complete external implementation review, start with
`docs/qwen_caption_instruction_dataset_external_partner_packet.md`. It explains
what changed, why it changed, which invariants matter, how artifacts are shaped,
what was tested, and what still requires a real-data pilot before training use.
This shorter handoff is a companion summary.

## Status

The captioning stack now has a functional UI and backend path for creating
caption-based VLM training datasets. The implementation is wired end to end for
dataset-backed generation, artifact export, human review import, readiness
reporting, trainer import compatibility, and trainer-side rejection of stale or
hand-edited non-trainable rows.

This is an implementation handoff, not a claim that any generated corpus is
already training-grade. The code path is structurally tested. A real-data pilot,
manual generated-QA review, and a small fine-tuning dry run are still required
before treating exported rows as final training data.

The documentation and code intentionally avoid dataset-specific names. The
workflow is meant to be reusable across annotation datasets.

## What Changed

The original caption workflow could generate captions, store alternate caption
records, and export caption-only training rows. It could not yet create a full
VLM instruction dataset made from a primary caption plus additional image-based
question/answer rows.

The new work adds a separate instruction-dataset path:

- A UI action, **Create VLM training dataset**, starts a dataset-backed
  instruction run.
- Each image can produce one broad caption row, called `caption0`.
- Each image can also produce configurable generated visual QA rows.
- Optional deterministic metadata QA rows can be generated from trusted labels
  only.
- The backend exports trainer rows, per-image archive rows, review rows, and a
  run-level report.
- The browser validates instruction JSONL before download, including required
  row metadata, instruction archive provenance, known validation/review states,
  rejected/failed/invalid validation state, non-trainable review state,
  duplicate image/question pairs, and JSON answer formats.
- The browser validates reviewed JSONL before import, including unsupported
  actionable row origins and duplicate or conflicting actionable review targets,
  and formats backend review-import failures into row-specific operator
  messages.
- A reviewer can import reviewed JSONL decisions back into the dataset.
- The Qwen trainer can import the flat exported instruction rows directly.
- The Qwen trainer refuses instruction flat rows that carry missing instruction
  provenance, missing or unknown validation/review state, rejected validation
  state, rejected or needs-revision review state, invalid deterministic JSON
  answers, or duplicate image/question pairs.

The key point is that instruction dataset creation is now a product workflow,
not a manual script chain.

## Why This Was Needed

A single caption per image is usually too thin for VLM fine-tuning. Training
often needs varied prompts about the same image: scene descriptions, object
relationships, counts, visible classes, spatial facts, and grounded short
answers. At the same time, generated language is not trusted source truth.

The old caption-only path could not answer these questions safely:

- Which facts came from real labels?
- Which facts came from the VLM?
- Which generated QA rows were accepted, rejected, or still unreviewed?
- Which rows are safe to train on?
- Which rows are useful for audit but should not be flattened into training
  JSONL?
- Can the exported JSONL be loaded directly by the training code?
- Can an operator run the job over a whole dataset without babysitting every
  image?

The new implementation solves those issues by separating source annotations,
generated language, deterministic label-derived QA, trainer rows, and review
metadata.

## Product Workflow

The intended operator flow is:

1. Select a caption dataset.
2. Leave set-and-forget enabled for durable backend execution.
3. Choose instruction settings:
   - generated QA rows per image
   - generated QA mix
   - generated answer format
   - include caption0
   - include generated QA
   - include deterministic metadata QA
   - provide read-only source-label context
   - strict QA grounding
4. Click **Create VLM training dataset**.
5. Download:
   - instruction JSONL for training
   - instruction archive JSONL for audit
   - review JSONL for human review
   - instruction report JSON for quality metrics and readiness
6. Review generated language rows externally.
7. Import reviewed JSONL decisions.
8. Re-export after review.
9. Import the final instruction JSONL into the Qwen trainer.

The UI still supports ordinary captioning separately. Captioning and instruction
dataset creation share infrastructure, but they are different product modes.

## Core Invariants

These rules define the correctness of the implementation:

- Source labels remain source labels.
- Generated captions and generated QA are language annotations.
- Generated language never overwrites trusted labels, boxes, class lists, or
  final annotations.
- Deterministic metadata QA is derived only from trusted source labels and is
  off by default.
- Rows can remain in the archive while being excluded from training JSONL.
- Human review applies review metadata only. It does not edit source labels,
  boxes, generated questions, generated answers, or final annotations.
- Human review target matching requires image context as well as QA id; a known
  QA id with a mismatched image path is rejected before any metadata is written.
- Human review target matching also requires current reviewed text; a known QA
  id with the right image is still rejected when the reviewed question,
  candidate answer, or training answer no longer matches the saved record.
- Training readiness is based on selected rows, validation state, manual review
  state, and corpus-quality gates.
- A row that is rejected or marked needs-revision by review is excluded from
  flattened training output.
- A review file with a mismatched embedded dataset id is blocked in the UI and
  rejected by the backend before any review metadata is applied.

Any change that violates these invariants should be treated as a regression.

## UI Controls

The caption panel now includes an instruction-dataset control group beneath the
ordinary batch caption controls.

Implemented controls:

- **Create VLM training dataset**
- **Generated QA per image**
  - default: `8`
  - accepted range: `0..20`
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
- **Download instruction JSONL**
- **Download instruction archive**
- **Download review JSONL**
- **Import reviewed JSONL**
- **Download instruction report**

The launch controls fail closed if all trainable row families are disabled. The
confirmation text also separates trainer JSONL rows from generated QA
candidates that will be generated only for archive/review.

The UI also makes model availability clearer. Model dropdown entries are styled
so local models use the normal local color and models needing download are red.
Backend jobs fail preflight for missing models unless downloads are explicitly
allowed.

## Exported Artifacts

### Instruction Training Rows

`instruction_training_rows` is the trainer-facing JSONL shape:

```json
{
  "image_path": "images/example.jpg",
  "question": "Describe this image in detail.",
  "answer": "A grounded visual answer.",
  "metadata": {
    "qa_id": "stable-row-id",
    "row_type": "generated_qa",
    "answer_format": "natural",
    "validation_status": "accepted",
    "review_status": "unreviewed",
    "source_archive": "tator_caption_instruction_archive_v1"
  }
}
```

The Qwen trainer imports this flat shape directly and converts each row into an
image/question/answer conversation.

The trainer import is a final safety boundary, not a blind loader. It preserves
metadata from exported rows, then rejects instruction rows with missing
provenance, missing or unknown validation/review state, explicit non-trainable
state, invalid JSON answers for deterministic or JSON-formatted types, and
duplicate image/question pairs. This protects fine-tuning runs from stale review
artifacts or manual JSONL edits that bypass the normal export readiness gate.

### Instruction Archive Rows

`instruction_archive_rows` are one JSONL record per image. They preserve the
construction evidence:

- image path and split
- caption0 record
- generated QA candidates
- optional deterministic metadata QA rows
- source annotation summary
- rejected rows and rejection reasons
- flattened rows selected for training
- export metadata

This is the audit artifact. It is intentionally richer than the trainer JSONL.

### Review Rows

`instruction_review_rows` are candidate-level review records. They include:

- image path
- row origin
- row type
- stable QA id
- question
- candidate answer
- selected training answer when applicable
- selected-for-training flag
- validation status
- review status
- rejection reasons
- source summary
- blank review decision field
- blank review notes field

Reviewers fill accepted, rejected, or needs-revision decisions in this artifact
and import it back into the application.

### Instruction Report

`instruction_report` is the run-level audit and readiness document. It includes:

- image count
- row counts by type
- selected flattened row count
- rejected row counts and rejection reasons
- split-level counts
- source-field provenance
- generated-QA metrics
- corpus-quality metrics
- training-readiness status
- blocking reasons
- required actions
- quality warnings

This report is the first artifact to inspect before deciding whether a corpus is
ready for review or training.

### Artifact Consistency

The browser validates each downloadable artifact as part of the same run-level
export set. It blocks trainer JSONL when the row count disagrees with the
report's selected flattened-row count. It blocks archive JSONL when the row
count disagrees with the report image count or archive image count, and it
rejects duplicate archive `image_path` rows. It blocks review JSONL when total
review rows, selected review rows, or manual-review rows disagree with the
instruction report.

This guard exists to catch stale, partial, mixed, or hand-edited artifacts
before a reviewer or trainer consumes them. Row validation still checks content;
artifact consistency checks whether the files agree with the run report.
The backend emits this check as `instruction_artifact_consistency` in the
archive, report, API payload, and summary. A failed backend consistency check
blocks training readiness with `instruction_artifacts_inconsistent`, and the UI
also refuses the corresponding download.

This is not only a count check. It also verifies that flattened trainer rows,
selected review rows, and archive candidates refer to the same image path, QA
id, and normalized question, and that selected review/archive answers match the
flattened training answer when those fields are present. A stale review JSONL
from another run can therefore fail even if it has the right number of rows.

Flat-layout image keys are canonicalized before this check runs. That means a
saved caption keyed as `sub/img.jpg` and a manifest row that temporarily appears
as `train/sub/img.jpg` are merged into one instruction image, not exported as a
duplicate source-manifest row.

## Source Annotation Contract

Source annotations are built from real label evidence only. They are stored in
the archive under `source_annotations`.

Expected source annotation fields include:

- `object_counts`
- `visible_classes`
- `annotations`
- `bbox_instances`
- `bbox_geometry`
- `spatial_facts`
- `uncertainty`
- `field_provenance`

If labels are missing, empty, unavailable, or the image is no longer present in
the dataset manifest, the source state records that fact. The exporter should
not invent replacement source truth.

## Generated Language Contract

Generated captions and generated QA are stored as language records. They can be
useful training candidates, but they are not trusted source metadata.

Generated QA is validated before it can be flattened:

- question must be present
- answer must be present
- image/question pair must be unique
- JSON answers must parse when JSON format is requested
- row must not be upstream-rejected
- structured claims must be supported by source annotations or be rejected
- manual review must not mark the row rejected or needs-revision

Rows that fail flattening remain visible in archive and review artifacts.

## Deterministic Metadata QA Contract

Deterministic metadata QA is optional and off by default. It is generated by code
from source labels, not by the VLM.

Supported deterministic QA types include:

- visible class list
- object-count schema
- per-class count
- positive class presence
- negative class presence
- simple spatial facts when source geometry supports them

These rows can be useful for label-grounded instruction training, but they
should remain explicit and opt-in because they change the corpus mix away from
pure visual-language generation.

## Structured Claim Handling

Generated QA sometimes asks questions whose answers can be checked against
source labels, such as counts, class lists, presence, or simple spatial facts.

The export rule is:

- If the structured claim is supported by trusted source annotations, the final
  flattened answer is rewritten from source annotations.
- The original generated answer remains in candidate metadata for audit.
- If the structured claim cannot be checked, the row is rejected from flattened
  training output.

This approach protects training data while preserving the candidate for review.

## Review Import Contract

Review import is deliberately conservative.

It accepts:

- JSON arrays
- JSON objects containing row arrays
- JSONL files

It applies only:

- `accepted`
- `rejected`
- `needs_revision`

It ignores:

- blank decisions
- unknown decisions
- deterministic metadata QA rows, because those are regenerated from labels at
  export time

It fails closed on:

- rows with an embedded dataset id that does not match the selected dataset
- generated-QA or caption0 targets whose QA id does not match the row's image
  context
- generated-QA or caption0 targets whose QA id and image context match but
  whose reviewed question or answer text is stale
- malformed review rows
- duplicate actionable review targets, including rows that use different row
  identities but resolve to the same saved caption or generated-QA record
- unsupported actionable row origins
- actionable rows without an image path
- stale generated-QA targets
- ambiguous generated-QA or caption0 matches
- synthetic caption0 review targets whose image cannot be resolved
- caption0 rows that would create a new saved caption without a synthetic id
  that matches the selected dataset, resolved image key, and current text-label
  caption

It never edits:

- source labels
- boxes
- image paths
- generated questions
- generated answers
- selected final annotations

The imported decisions affect subsequent archive output, review output,
training readiness, and flattened training rows.

## Training Readiness

The instruction report classifies the corpus as:

- `ready`
- `needs_review`
- `blocked`

`ready` means selected rows pass structural validation, selected language rows
are accepted or otherwise do not require manual acceptance, and corpus-quality
gates do not warn.

`needs_review` means the corpus still has pending manual review or quality
warnings.

`blocked` means the export should not be used for training. Examples include no
selected rows, no images, or a selected row marked rejected or needs-revision.

The browser refuses blocked instruction JSONL downloads. It also refuses
needs-review trainer JSONL by default through **Require ready report for trainer
JSONL**. Operators can disable that gate only for deliberate review-pending
diagnostics. Scripted API clients can request the same server-side behavior with
`/captions/export?require_ready_instruction_export=true`, which returns HTTP
409 unless readiness is `ready`.

## Runtime Hardening

The caption path was also hardened because real dataset-scale captioning exposed
runtime fragility.

Implemented hardening themes:

- token-budget defaults remain high enough for thinking-capable models;
- explicit user output-token overrides remain honored;
- prompt-size accounting estimates prompt load before generation;
- dense label prompts use representative spatial box subsets rather than
  dumping every box;
- authoritative counts remain authoritative even when box lists are trimmed;
- prompt wording explains when box lists are representative;
- repeated-output loops are detected and trimmed;
- loop recovery retries with safer decoding settings;
- set-and-forget backend jobs are the durable default;
- UI attach/recover is available without making manual recovery the normal
  workflow;
- backend supervision and persisted artifacts are used because process-level
  Metal/GPU faults can abort Python before normal exception handling.

The implementation does not pretend that GPU process aborts are catchable
Python errors. The correct operational pattern is supervised restart plus
resumable jobs.

## Prompt And Token Budget Behavior

The implementation separates user-facing output-token settings from internal
model generation needs.

Important behavior:

- Thinking-capable models can use larger automatic output budgets.
- A numeric user value remains a hard override.
- Prompt previews and telemetry expose prompt-size estimates.
- Full-image and windowed prompts avoid unbounded box dumping.
- Counts remain authoritative even when prompt boxes are representative.

This explains why internal logs may show a larger model-generation budget than
a user expected from a simple caption-length setting. The high budget is a
ceiling, not a request for the model to fill it.

## Files To Review

Primary implementation files:

- `localinferenceapi.py`
  - instruction archive construction
  - source annotation summaries
  - generated-QA validation
  - structured-claim rewrite/rejection
  - review import
  - corpus-quality metrics
  - training readiness
- `models/schemas.py`
  - instruction-dataset request fields
  - generated-QA bounds
  - set-and-forget caption settings
- `tools/run_qwen_caption_flow_benchmark.py`
  - caption runner instruction mode
  - generated-QA pass
  - prompt/token telemetry
  - loop/recovery audit events
- `tools/qwen_training.py`
  - flat instruction row import
  - conversion to Qwen conversation records
- `ybat-master/ybat.html`
  - caption panel controls
- `ybat-master/ybat.js`
  - instruction request construction
  - export validation
  - review import parsing
  - readiness rendering
  - model availability styling
- `ybat-master/ybat.css`
  - layout and caption panel styling

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

Related documentation:

- `docs/qwen_caption_training_dataset_external_review_handoff.md`
- `docs/qwen_caption_instruction_dataset_hardening_report.md`
- `docs/qwen_caption_ui_scenarios.md`
- `docs/qwen_caption_prompt_stack.md`

## Validation Evidence

The implementation has been checked with focused backend, trainer, UI contract,
and rendered UI smoke tests.

Current combined caption/instruction/trainer/UI contract suite:

```bash
./.venv-macos/bin/python -m pytest \
  tests/test_qwen_caption_dataset_job.py \
  tests/test_qwen_training_backend.py \
  tests/test_dataset_linked_annotation_flows.py::test_caption_alternate_routes_append_update_export_and_delete \
  tests/test_labeling_panel_layout_contract.py \
  tests/test_qwen_caption_ui_smoke_tool.py \
  -q
```

Latest recorded result:

```text
175 passed
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

Latest recorded result:

```text
3 passed
```

Focused review-import fail-closed suite:

```bash
./.venv-macos/bin/python -m pytest \
  tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_persists_review_metadata \
  tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_mismatched_dataset_id \
  tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_duplicate_actionable_targets \
  tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_duplicate_resolved_actionable_targets \
  tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_unmatchable_actionable_rows_atomically \
  tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_stale_generated_qa_text \
  tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_stale_caption0_text \
  -q
```

Latest recorded result:

```text
19 passed
```

Focused trainer-import boundary suite:

```bash
./.venv-macos/bin/python -m pytest \
  tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_imports_flat_question_answer_rows \
  tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_rejects_non_trainable_flat_rows \
  tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_rejects_duplicate_flat_questions \
  tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_ignores_blank_flat_rows_before_duplicate_check \
  -q
```

Latest recorded result:

```text
7 passed
```

Full trainer backend suite:

```bash
./.venv-macos/bin/python -m pytest tests/test_qwen_training_backend.py -q
```

Latest recorded result:

```text
25 passed
```

Focused instruction-dataset and UI contract suite:

```bash
./.venv-macos/bin/python -m pytest \
  tests/test_qwen_caption_dataset_job.py \
  tests/test_dataset_linked_annotation_flows.py::test_caption_alternate_routes_append_update_export_and_delete \
  tests/test_labeling_panel_layout_contract.py \
  tests/test_qwen_caption_ui_smoke_tool.py \
  -q
```

Latest recorded result:

```text
150 passed
```

Runtime and unattended hardening suites have also been run in prior hardening
passes and are listed in the hardening report. The important point for review is
that the current evidence is structural and workflow-oriented. It proves the
paths are wired and guarded. It does not replace human review of generated
language quality.

## Recommended External Review Procedure

1. Inspect the UI workflow and confirm that ordinary captioning and instruction
   dataset creation are separate.
2. Run a small dataset-backed instruction job.
3. Inspect the instruction report first.
4. Inspect archive rows for source/generated separation.
5. Inspect review rows for candidate answer, selected answer, validation state,
   and source summary.
6. Confirm that rejected or needs-revision language candidates are excluded from
   flattened training rows.
7. Confirm that deterministic metadata QA is absent unless explicitly enabled.
8. Import a reviewed JSONL file and verify that only review metadata changes.
9. Attempt to import a review file with a different embedded dataset id and
   confirm that it is rejected before any review metadata is applied.
10. Import the instruction JSONL into the Qwen trainer and verify conversation
    conversion.
11. Review generated QA content manually for visual grounding and usefulness.
12. Decide corpus acceptance thresholds before the first training run.

## Pilot Required Before Training Use

The next validation should be a real-data pilot that includes:

- one dense labeled scene;
- one image with no labels;
- one image with multiple source classes;
- one image with existing alternate captions;
- one image with small objects near crop or window boundaries;
- one intentionally missing or empty label file;
- one review import with accepted, rejected, and needs-revision decisions;
- one mismatched-dataset review import attempt;
- one flat JSONL trainer import dry run.

For each pilot case, reviewers should inspect:

- whether generated questions are visually answerable;
- whether generated answers are grounded in the image;
- whether counts and class claims match source evidence;
- whether uncertainty is handled honestly;
- whether source labels remain separate from generated language;
- whether archive, review, report, and trainer rows agree.

## Remaining Risks

Known remaining risks:

- Generated QA may be structurally valid but low value for training.
- Some visually plausible answers may still be unsupported by source labels.
- Dense scenes may need better question diversity than the current mix selector.
- Review-pending trainer JSONL can still be downloaded if an operator
  deliberately disables the ready-report gate.
- Process-level GPU faults require supervisor restart and resume; they cannot be
  fully handled inside the model call.
- The current tests prove contracts and wiring, not final corpus quality.

## Suggested Decisions For Reviewers

Reviewers should decide:

- whether deterministic metadata QA should remain off by default;
- whether the review-pending override should remain available, or whether
  non-ready trainer JSONL downloads should be impossible;
- what generated-QA rejection rate is acceptable;
- what duplicate-question rate is acceptable;
- what minimum source-class coverage is required;
- how many reviewed rows are needed before training;
- whether review import should eventually support edited replacement questions
  and answers, or stay metadata-only;
- whether generated QA mix should become a row-type budget instead of one
  selector.

## Bottom Line

The implementation now provides the requested full caption-to-instruction-data
path: UI launch, dataset-backed generation, caption0, generated visual QA,
optional label-derived QA, audit archive, review artifact, review import,
readiness report, export validation, and trainer import compatibility.

The reason for the design is provenance. Training rows are intentionally simple,
but the archive and report keep enough evidence to decide whether those rows are
safe to train on. The next step is not more wiring; it is a real pilot with
manual QA review and a small training import or fine-tuning dry run.
