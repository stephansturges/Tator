# Qwen Caption Instruction Dataset Hardening Report

Date: 2026-06-29

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
- Duplicate canonical image-path/question rows are rejected from instruction
  JSONL.
- Canonical image paths are used consistently across trainer rows, archive
  rows, review rows, and artifact-consistency checks. Path aliases such as
  `./frame.jpg`, `train/frame.jpg`, and split-prefixed forms cannot create
  duplicate training identities or mismatched review targets.
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
- Added instruction launch preflight so the UI refuses an instruction job with
  all trainable row families disabled and makes archive/review-only generated
  QA candidates explicit in the confirmation text.
- Fixed caption action layout so export and instruction buttons wrap into
  readable responsive columns instead of clipping in the sidebar.
- Fixed readiness and attach/recover rows so long status text cannot squeeze the
  action buttons.
- Manual caption archive controls now report success only after the underlying
  save, update, primary-selection, or delete action returns a real mutation.
  No-op paths from stale clicks or disabled-control bypasses stay quiet or show
  warnings, and backend failures use the shared operator-facing caption archive
  failure formatter. The same controls are also disabled while a caption or
  instruction job is mutating the caption archive and only re-enable when the
  current image, caption text, and selected-caption requirements are satisfied.

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
- The browser-side instruction JSONL validator now validates row id, row type,
  answer source, answer format, instruction archive provenance, validation
  status, and review status from row metadata before writing a download,
  preventing missing-provenance rows, unknown statuses, rejected, failed,
  invalid, needs-revision, or otherwise malformed rows from being saved by the
  UI.
- Browser downloads now perform artifact-level consistency checks against the
  instruction report before writing trainer JSONL, archive JSONL, or review
  JSONL. A file can pass row validation and still be blocked if its row counts
  disagree with the report's selected flattened-row count, image count,
  review-row count, or manual-review count. Archive JSONL also rejects duplicate
  `image_path` rows. The same guard now also compares trainer rows, selected
  review rows, and archive candidates by canonical image path, QA id, normalized
  question, selected per-image row counts, and matching training answers, so
  equal-count stale or mixed artifacts fail closed.
- Browser-side archive, review, and artifact-consistency validators now
  canonicalize image paths before duplicate checks, per-image selected-row
  counts, and review-target identity checks. This keeps UI preflight behavior
  aligned with the backend when exported packets contain harmless path spelling
  differences.
- The backend now emits the same versioned `instruction_artifact_consistency`
  object in the instruction archive, instruction report, API export payload, and
  instruction summary. If this object is not OK, training readiness is forced to
  `blocked` with `instruction_artifacts_inconsistent`.
- The browser-side instruction report validator now requires the embedded
  `instruction_artifact_consistency` object to be present, versioned as
  `tator_caption_instruction_artifact_consistency_v1`, boolean-OK, and
  error-free. Report downloads therefore fail closed instead of presenting a
  report as valid when the run-level artifact-consistency proof is absent or
  failed.
- The browser-side instruction report validator now checks readiness
  self-consistency. `ready_for_training` must agree with `status`, `ready`
  reports cannot carry blocking reasons, required actions, or quality warnings,
  `blocked` reports must name at least one blocking reason, and top-level image
  and selected-row counts must match the corresponding corpus-quality metrics.
- The browser-side artifact-consistency validator now validates every embedded
  consistency proof it receives in the payload, report, and archive, and blocks
  trainer, archive, or review downloads when those replicated proof objects
  disagree. A stale or hand-edited archive can no longer hide behind an OK
  top-level consistency object.
- Flat-layout instruction exports now canonicalize split-prefixed and
  non-split-prefixed image keys before merging manifest rows, saved captions,
  text-label mirrors, and generated QA. This prevents nested images such as
  `sub/img.jpg` from being duplicated as both a manifest image and a synthetic
  caption-only image, and prevents basename collisions in archive rows.
- The browser-side review JSONL import validator now rejects unsupported
  actionable row origins and duplicate or conflicting actionable review targets
  before calling the backend, giving operators immediate feedback on review
  packets that would fail the server-side transactional import.
- Browser-side instruction artifact actions now share one failure reporter for
  trainer JSONL, archive JSONL, review JSONL, reviewed-row import, and report
  downloads. Backend or validation failures update both the caption export
  health row and the toast/status message, and already formatted blocked
  messages are not double-prefixed.
- Browser and backend artifact-consistency validation now require caption0 and
  generated-QA review rows to carry dataset identity before export/import, even
  while `review_decision` is blank. Review packets therefore fail before
  download if they would become impossible to import after human review.
- Backend artifact-consistency validation now mirrors the browser review-row
  shape contract. Review JSONL rows must carry the review-row format marker,
  image path, QA id, row origin, question, candidate answer, validation status,
  boolean `selected_for_training` and `requires_manual_review` values, source
  summary, rejection-reason array, review-decision field, and review-notes
  field. Unsupported decisions, unsupported actionable row origins, duplicate
  actionable targets, duplicate image-path/QA-id pairs, and selected rows with
  no training answer fail before artifacts are treated as consistent.
- Backend review import now enforces the same review-row shape before any
  transactional matching or metadata mutation. Hand-edited review files with
  string booleans, missing source summaries, missing review columns, missing
  training answers for selected rows, or malformed rejection reasons are
  rejected with row-specific errors and UI-facing explanations.
- Review-import backend failures are formatted into row-specific operator
  messages instead of raw `review_rows_*` codes. Stale caption0/generated-QA
  text, dataset mismatch, duplicate actionable decisions, unsupported row
  origins, missing image context, and unresolved caption targets now tell the
  operator whether to export a fresh review JSONL, select the matching dataset,
  or keep one decision per target.
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
  answers, boxes, or final annotations. Persisted caption0 and generated-QA
  decisions must carry an embedded dataset id and stable QA id, and rows
  carrying a missing or different dataset id are blocked by the UI import
  preflight and rejected by the backend before any review metadata is applied.
  Rows carrying a missing QA id, a QA id that does not match the saved record,
  or a known QA id with a mismatched image path are also rejected before any
  metadata is applied. Rows carrying the same QA id and image context are still
  rejected if their reviewed question, candidate answer, or selected training
  answer no longer matches the saved caption/generated-QA record and current
  source-rewritten archive view.
- Review import now resolves image-path aliases with the same canonicalization
  policy used by the browser validator. Harmless spellings such as
  `./train//frame.jpg` can still match the saved `train/frame.jpg` target, while
  an explicit split prefix is not allowed to fall through to a different split
  that happens to share the same basename.
- The backend rejects duplicate actionable review targets before applying any
  imported review metadata. Exact duplicate decisions and conflicting duplicate
  decisions both fail closed. The backend also rejects rows that use different
  row identities but resolve to the same saved caption or generated-QA record,
  so API or script imports cannot silently let the last duplicate row win.
- The backend rejects malformed or unmatchable actionable review rows before
  applying any imported review metadata. Generated-QA rows must include the
  reviewed question and answer text, and caption0 rows must include reviewed
  caption text even when the QA id matches. Stale generated-QA targets,
  unsupported actionable row origins, missing image paths, ambiguous matches,
  and unresolvable synthetic caption0 review targets fail the whole import
  rather than partially applying earlier rows. Caption0 rows can create a saved
  caption review record only when their synthetic id matches the selected
  dataset, resolved image key, and current text-label caption; arbitrary or
  forged caption0 rows are rejected before any metadata is written.
- Browser and backend review-import validation now reject unsupported non-blank
  `review_decision` values before applying any review metadata. A typo such as
  `acceppted` blocks the import instead of being silently skipped as a missing
  decision.
- Backend review import now rejects API/script packets that contain no
  accepted, rejected, or needs-revision caption0/generated-QA decisions to
  persist. Blank-decision and deterministic-only packets can no longer return an
  `applied` status with zero persisted decisions.
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
  review-pending diagnostic exports. The trainer JSONL UI path now sends
  `require_ready_instruction_export=true` to the backend when that ready-report
  gate is enabled, so browser behavior and API/script behavior share the same
  server-side refusal boundary. Archive, review, and report downloads do not
  send that gate because they are diagnostic artifacts needed to repair a
  not-ready corpus. The caption export API also exposes
  `require_ready_instruction_export=true` for scripts that need server-side
  refusal of non-ready instruction exports. Reviewed-out language rows are
  removed from flattened output; any selected row that still carries a rejected
  or needs-revision decision is a hard blocker.
- Export options let callers include or exclude caption0, generated QA, and
  deterministic metadata QA without altering saved data, while preserving the
  requested generated-QA mix and answer format.
- The Qwen training dataset loader now accepts exported flat instruction rows
  directly, preserving row metadata while converting each row into the
  conversation format used by fine-tuning. For rows marked as instruction
  archive exports, the loader now also fails closed on stale or hand-edited rows
  with missing provenance, missing or unknown validation/review state, rejected
  validation state, rejected/needs-revision review state, invalid deterministic
  JSON answers, or duplicate canonical image-path/question pairs.

## Validation Completed

- Python syntax:
  - `./.venv-macos/bin/python -m py_compile models/schemas.py tools/run_qwen_caption_flow_benchmark.py localinferenceapi.py api/datasets.py`
  - `./.venv-macos/bin/python -m py_compile tools/run_qwen_caption_ui_smoke.py`
- JavaScript syntax:
  - `node --check ybat-master/ybat.js`
- Focused instruction-dataset, export, and UI contract tests:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py tests/test_qwen_training_backend.py tests/test_dataset_linked_annotation_flows.py::test_caption_alternate_routes_append_update_export_and_delete tests/test_labeling_panel_layout_contract.py tests/test_qwen_caption_ui_smoke_tool.py -q`
  - Current result: 205 passed.
- Current artifact-consistency UI contract tests:
  - `./.venv-macos/bin/python -m pytest tests/test_labeling_panel_layout_contract.py::test_qwen_caption_instruction_artifact_consistency_blocks_mismatched_exports tests/test_labeling_panel_layout_contract.py::test_qwen_caption_export_preserves_saved_alternates_and_primary_rows -q`
  - Result: 2 passed.
- Current backend artifact-consistency and canonical image-key tests:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py::test_caption_instruction_artifact_consistency_validator_blocks_same_count_identity_mismatches tests/test_qwen_caption_dataset_job.py::test_caption_instruction_artifact_consistency_validator_blocks_mismatched_backend_counts tests/test_qwen_caption_dataset_job.py::test_caption_instruction_artifact_consistency_validator_canonicalizes_image_paths tests/test_qwen_caption_dataset_job.py::test_caption_instruction_artifact_consistency_validator_rejects_malformed_review_rows tests/test_labeling_panel_layout_contract.py::test_qwen_caption_instruction_artifact_consistency_blocks_mismatched_exports -q`
  - Result: 5 passed.
- Current review-import fail-closed tests:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_persists_review_metadata tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_accepts_canonical_image_path_aliases tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_conflicting_split_alias tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_mismatched_dataset_id tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_missing_dataset_id tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_missing_or_mismatched_qa_id tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_duplicate_actionable_targets tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_canonical_duplicate_actionable_targets tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_rows_missing_current_text tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_unmatchable_actionable_rows_atomically tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_stale_generated_qa_text tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_stale_rewritten_training_answer tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_stale_caption0_text tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_arbitrary_caption0_creation -q`
  - Result: 32 passed.
- Current no-op review-import guard:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_no_actionable_persisted_decisions tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_unsupported_review_decision tests/test_labeling_panel_layout_contract.py::test_qwen_caption_instruction_review_import_formats_backend_failures -q`
  - Result: 4 passed.
- Current review-export dataset identity guard:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py::test_caption_instruction_artifact_consistency_validator_requires_review_dataset_identity tests/test_qwen_caption_dataset_job.py::test_caption_instruction_artifact_consistency_validator_canonicalizes_image_paths tests/test_labeling_panel_layout_contract.py::test_qwen_caption_instruction_review_validator_blocks_bad_actionable_rows tests/test_labeling_panel_layout_contract.py::test_qwen_caption_instruction_artifact_consistency_blocks_mismatched_exports tests/test_labeling_panel_layout_contract.py::test_qwen_caption_instruction_review_import_parser_accepts_reviewer_file_shapes -q`
  - Result: 5 passed.
- Current backend review-row schema guard:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py::test_caption_instruction_artifact_consistency_validator_rejects_malformed_review_rows tests/test_qwen_caption_dataset_job.py::test_caption_instruction_artifact_consistency_validator_requires_review_dataset_identity tests/test_qwen_caption_dataset_job.py::test_caption_instruction_artifact_consistency_validator_canonicalizes_image_paths tests/test_labeling_panel_layout_contract.py::test_qwen_caption_instruction_review_validator_blocks_bad_actionable_rows -q`
  - Result: 4 passed.
- Current backend review-import schema guard:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_rows_missing_current_text tests/test_qwen_caption_dataset_job.py::test_caption_instruction_review_import_rejects_malformed_review_row_shape tests/test_labeling_panel_layout_contract.py::test_qwen_caption_instruction_review_import_formats_backend_failures -q`
  - Result: 10 passed.
- Current trainer-import fail-closed boundary tests:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_imports_flat_question_answer_rows tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_rejects_non_trainable_flat_rows tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_rejects_duplicate_flat_questions tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_rejects_normalized_duplicate_flat_questions tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_rejects_resolved_duplicate_flat_image_aliases tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_ignores_blank_flat_rows_before_duplicate_check -q`
  - Result: 9 passed.
- Current full trainer backend test file:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_training_backend.py -q`
  - Result: 27 passed.
- Current caption/instruction/UI contract tests outside the trainer file:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py tests/test_dataset_linked_annotation_flows.py::test_caption_alternate_routes_append_update_export_and_delete tests/test_labeling_panel_layout_contract.py tests/test_qwen_caption_ui_smoke_tool.py -q`
  - Result: 167 passed.
- Earlier targeted trainer import compatibility:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_training_backend.py::test_qwen_conversation_dataset_imports_flat_question_answer_rows tests/test_qwen_caption_dataset_job.py::test_caption_instruction_training_rows_import_into_qwen_trainer -q`
  - Result: 2 passed.
- Additional instruction archive provenance and manifest-gating regression:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py -q`
  - Result: 113 passed.
- Caption0 structured-claim validation and instruction export validator
  regression:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py tests/test_labeling_panel_layout_contract.py -q`
  - Result: 164 passed.
- Prompt, runner, progress, launcher, and unattended contracts:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_flow_benchmark.py tests/test_qwen_caption_prompt.py tests/test_qwen_progress.py tests/test_macos_backend_launcher_contract.py -q`
  - Result: 191 passed.
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_operation_audit.py tests/test_qwen_caption_soak_audit.py tests/test_qwen_caption_soak_certification.py tests/test_qwen_caption_soak_drill.py -q`
  - Result: 90 passed.
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_soak_preflight.py tests/test_qwen_caption_soak_supervisor.py tests/test_qwen_caption_soak_watchdog.py tests/test_qwen_caption_unattended_launcher.py -q`
  - Result: 136 passed.
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
