# Qwen Caption Instruction Dataset Hardening Report

Date: 2026-06-28

## Goal

Add a production-ready VLM training dataset path to the Qwen captioning workflow
without weakening the existing caption-only path. The new path must let an
operator run captioning over a dataset, generate optional image-grounded
question/answer rows, preserve source annotations separately from generated
language, export trainer-ready rows, and keep enough provenance to audit every
row before training.

## Product Behavior Implemented

- Existing caption exports remain intact:
  - flat caption audit JSONL
  - grouped caption JSON
  - caption-only VLM JSONL with JSON caption answers
- A separate instruction dataset path now exists:
  - **Create VLM training dataset** starts a dataset-backed caption job with
    `instruction_dataset=true`.
  - `caption0` rows are included by default.
  - generated visual QA rows are included by default.
  - deterministic metadata QA is off by default and must be explicitly enabled.
  - source annotation counts can be passed to the generator as read-only context.
  - strict grounding is enabled by default.
  - generated QA per image is clamped to `0..20`; the default is `8`.
  - generated QA mix is explicit: balanced, scene-level, object-focused, or
    caption-variant oriented.
  - generated answer format is explicit: natural text or JSON.
- The backend exports:
  - `instruction_training_rows`: flattened `image_path` / `question` / `answer`
    rows for training. The Qwen trainer imports this flat shape directly and
    normalizes each row into a two-turn image/question/answer conversation.
  - `instruction_archive_rows`: one per-image construction archive record per
    image, ready to download as JSONL.
  - `instruction_review_rows`: one candidate-level review row per caption0,
    generated QA, and deterministic metadata QA item, with source summaries,
    selected-for-training flags, rejection reasons, and blank decision/note
    fields for human audit before training.
  - `instruction_archive`: a versioned per-image archive containing caption0,
    generated QA, optional deterministic metadata QA, source annotation
    provenance, rejected rows, and the flattened rows.
  - `instruction_report`: run-level counts, provenance, split expansion,
    rejection summaries, and corpus-quality metrics for audit.
  - `instruction_summary`: row, generated QA, deterministic QA, and rejection
    counts.

## Safety And Separation

- Generated question/answer rows are stored as language annotations only.
- Source annotations are derived from real label evidence and remain separate
  from generated text. The archive records object counts, visible classes, bbox
  instances, coarse bbox geometry, deterministic spatial facts, uncertainty, and
  field provenance.
- Deterministic metadata QA is only built from real source labels and is only
  exported when explicitly enabled. Its answers are typed JSON rows for class
  lists, object-count schemas, per-class counts, presence, negative presence, and
  simple bbox-derived spatial facts when supported.
- Generated QA candidates remain in the archive even when rejected, but only
  accepted generated rows are flattened into trainer rows.
- `caption0` remains visible in the archive even when rejected, but explicit
  numeric object-count claims in `caption0` must match trusted source labels
  before the row can be flattened into trainer rows.
- Caption or generated-QA records whose image no longer appears in the dataset
  manifest are kept in the instruction archive for audit, marked as
  non-flattenable, and excluded from trainer rows with an explicit rejection
  reason.
- Exact duplicate image/question rows are rejected from instruction JSONL.
- The UI blocks malformed instruction JSONL downloads before writing a file.

## UI/UX Changes

- Added a compact instruction-dataset panel under the batch caption controls.
- Added explicit controls for:
  - generated QA rows per image
  - generated QA mix
  - generated answer format
  - include caption0
  - include generated QA
  - include deterministic metadata QA
  - give generator read-only label context
  - strict QA grounding
- Added separate downloads:
  - **Download instruction JSONL**
  - **Download instruction archive**
  - **Download review JSONL**
  - **Import reviewed JSONL**
  - **Download instruction report**
- Fixed caption action layout so export and instruction buttons wrap into
  readable responsive columns instead of clipping in the sidebar.
- Fixed readiness and attach/recover rows so long status text cannot squeeze the
  action buttons.

## Backend And Runner Changes

- `QwenCaptionDatasetJobRequest` now normalizes instruction-dataset settings and
  clamps `subcaptions_per_image`.
- Dataset jobs pass instruction settings to the caption runner.
- The runner performs an extra image-grounded generated-QA pass only when
  instruction dataset mode is enabled and `subcaptions_per_image > 0`.
- Generated QA rows are parsed from JSON, structurally validated, deduplicated by
  question within the image, and persisted as instruction records.
- Generated-QA provenance fields such as answer format, source fields,
  validation targets, validation status, and review status are preserved through
  job-result parsing, persisted instruction records, archive rows, and flattened
  trainer rows.
- The export layer validates generated QA again before flattening and rejects
  unsupported structured claims when trusted source labels are missing.
- Supported generated count, class-list, presence, and simple spatial questions
  are rewritten during export so their final answers come from
  `source_annotations`, with the original generated answer preserved as
  candidate metadata.
- The browser-side instruction JSONL validator now validates row type, answer
  format, and validation status from row metadata before writing a download,
  preventing rejected or malformed rows from being saved by the UI.
- The backend now emits `instruction_export_validation` in the archive, report,
  API payload, and instruction summary. The same flattened trainer-row checks
  run server-side, and training readiness becomes blocked if malformed,
  duplicate, rejected, needs-revision, or invalid-JSON rows ever reach the
  trainer export path.
- The instruction archive now exposes both a full JSON audit object and
  per-image `instruction_archive_rows` for JSONL download. The report records
  row-type distribution, split image counts, split training-row counts,
  rejection reason counts, source-field provenance, QA count per image, and
  exclusion categories.
- The instruction export also exposes `instruction_review_rows`, a
  candidate-level audit queue that records caption0, generated-QA, and
  deterministic metadata candidates separately from flattened trainer rows.
- The caption API now accepts reviewed instruction JSONL and applies only review
  decision metadata back to saved caption and generated-QA records. This closes
  the export-review-import loop without editing source labels, questions,
  answers, boxes, or final annotations. Rows carrying a different dataset id
  are skipped by the backend and blocked by the UI import preflight.
- The backend rejects duplicate actionable review targets before applying any
  imported review metadata. Exact duplicate decisions and conflicting duplicate
  decisions both fail closed, so API or script imports cannot silently let the
  last duplicate row win.
- Caption0 or generated-QA candidates marked rejected or needs-revision by
  manual review remain in the archive and review JSONL but are excluded from
  flattened trainer rows.
- The instruction report now includes `corpus_quality_metrics` for generated-QA
  diversity, duplicate-question rate, generated-QA acceptance/rejection rates,
  structured rewrite rate, image-level training coverage, source-grounded row
  coverage, answer-format distribution, and source-class coverage.
- The instruction report now includes `training_readiness`, which classifies
  the exported corpus as `ready`, `needs_review`, or `blocked`. The browser
  blocks instruction JSONL when readiness is blocked and, by default, also
  blocks trainer JSONL when selected language rows or quality gates still need
  review. Operators must deliberately disable the ready-report gate for
  review-pending diagnostic exports. The caption export API also exposes
  `require_ready_instruction_export=true` for scripts that need server-side
  refusal of non-ready instruction exports. Reviewed-out language rows are
  removed from flattened output; any selected row that still carries a rejected
  or needs-revision decision is a hard blocker.
- Export options let callers include or exclude caption0, generated QA, and
  deterministic metadata QA without altering saved data, while preserving the
  requested generated-QA mix and answer format.
- The Qwen training dataset loader now accepts exported flat instruction rows
  directly, preserving row metadata while converting each row into the
  conversation format used by fine-tuning.

## Validation Completed

- Python syntax:
  - `./.venv-macos/bin/python -m py_compile models/schemas.py tools/run_qwen_caption_flow_benchmark.py localinferenceapi.py api/datasets.py`
  - `./.venv-macos/bin/python -m py_compile tools/run_qwen_caption_ui_smoke.py`
- JavaScript syntax:
  - `node --check ybat-master/ybat.js`
- Focused instruction-dataset, export, and UI contract tests:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py tests/test_qwen_training_backend.py tests/test_dataset_linked_annotation_flows.py::test_caption_alternate_routes_append_update_export_and_delete tests/test_labeling_panel_layout_contract.py tests/test_qwen_caption_ui_smoke_tool.py -q`
  - Result: 137 passed.
- Trainer import compatibility:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_imports_flat_question_answer_rows tests/test_qwen_caption_dataset_job.py::test_caption_instruction_training_rows_import_into_qwen_trainer -q`
  - Result: 2 passed.
- Additional instruction archive provenance and manifest-gating regression:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py -q`
  - Result: 64 passed.
- Caption0 structured-claim validation and instruction export validator
  regression:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py tests/test_labeling_panel_layout_contract.py -q`
  - Result: 112 passed.
- Prompt, runner, progress, launcher, and unattended contracts:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_flow_benchmark.py tests/test_qwen_caption_prompt.py tests/test_qwen_progress.py tests/test_macos_backend_launcher_contract.py -q`
  - Result: 191 passed.
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_operation_audit.py tests/test_qwen_caption_soak_audit.py tests/test_qwen_caption_soak_certification.py tests/test_qwen_caption_soak_drill.py -q`
  - Result: 90 passed.
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_soak_preflight.py tests/test_qwen_caption_soak_supervisor.py tests/test_qwen_caption_soak_watchdog.py tests/test_qwen_caption_unattended_launcher.py -q`
  - Result: 135 passed.
- Rendered browser smoke:
  - `./.venv-macos/bin/python tools/run_qwen_caption_ui_smoke.py --base-url http://127.0.0.1:8000 --out-json tmp/qwen_caption_ui_smoke_report.json --screenshot tmp/qwen_caption_ui_smoke.png`
  - Result: `ok=true`, caption readiness reported 39 pass, 1 warning, 0 fail;
    no console errors, no failed requests, no bad HTTP responses, no clipped
    caption action buttons. The screenshot confirms the generated-QA mix,
    answer-format, ready-report gate, archive, review import, and report
    controls are visible and readable in the caption panel.
- Restricted project-name scan:
  - Source, docs, tests, tools, UI, and backend entrypoint scan.
  - Result: no matches.

## Remaining Training-Quality Work

- Run a small real VLM instruction dataset pilot with at least one dense scene,
  one empty-label image, one image with multiple object classes, and one image
  with existing alternate captions.
- Review generated QA content manually with the review JSONL before using it
  for fine-tuning, then import reviewed decisions so readiness reflects that
  audit.
