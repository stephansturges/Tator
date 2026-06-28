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
- The backend exports:
  - `instruction_training_rows`: flattened `image_path` / `question` / `answer`
    rows for training.
  - `instruction_archive`: a versioned per-image archive containing caption0,
    generated QA, optional deterministic metadata QA, source annotation
    provenance, rejected rows, and the flattened rows.
  - `instruction_summary`: row, generated QA, deterministic QA, and rejection
    counts.

## Safety And Separation

- Generated question/answer rows are stored as language annotations only.
- Source annotations are derived from dataset manifest label lines and remain
  separate from generated text.
- Deterministic metadata QA is only built from real source labels and is only
  exported when explicitly enabled.
- Exact duplicate image/question rows are rejected from instruction JSONL.
- The UI blocks malformed instruction JSONL downloads before writing a file.

## UI/UX Changes

- Added a compact instruction-dataset panel under the batch caption controls.
- Added explicit controls for:
  - generated QA rows per image
  - include caption0
  - include generated QA
  - include deterministic metadata QA
  - give generator read-only label context
  - strict QA grounding
- Added separate downloads:
  - **Download instruction JSONL**
  - **Download instruction archive**
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
- Export options let callers include or exclude caption0, generated QA, and
  deterministic metadata QA without altering saved data.

## Validation Completed

- Python syntax:
  - `./.venv-macos/bin/python -m py_compile models/schemas.py tools/run_qwen_caption_flow_benchmark.py localinferenceapi.py api/datasets.py`
  - `./.venv-macos/bin/python -m py_compile tools/run_qwen_caption_ui_smoke.py`
- JavaScript syntax:
  - `node --check ybat-master/ybat.js`
- Backend and runner tests:
  - `./.venv-macos/bin/python -m pytest tests/test_qwen_caption_dataset_job.py tests/test_qwen_caption_flow_benchmark.py -q`
- API export test:
  - `./.venv-macos/bin/python -m pytest tests/test_dataset_linked_annotation_flows.py::test_caption_alternate_routes_append_update_export_and_delete -q`
- UI contract and smoke-tool tests:
  - `./.venv-macos/bin/python -m pytest tests/test_labeling_panel_layout_contract.py tests/test_qwen_caption_ui_smoke_tool.py -q`
- Rendered browser smoke:
  - `./.venv-macos/bin/python tools/run_qwen_caption_ui_smoke.py --base-url http://127.0.0.1:8000 --out-json tmp/qwen_caption_ui_smoke_report.json --screenshot tmp/qwen_caption_ui_smoke.png`
  - Result: `ok=true`, 29 readiness checks passing, no console errors, no failed
    requests, no bad HTTP responses, no clipped caption action buttons.
- Restricted project-name scan:
  - Full repository scan, excluding `.git`, Python bytecode, and cache folders.
  - Result: no matches.

## Remaining Training-Quality Work

- Run a small real VLM instruction dataset pilot with at least one dense scene,
  one empty-label image, one image with multiple object classes, and one image
  with existing alternate captions.
- Review generated QA content manually for grounding quality before using it for
  fine-tuning.
- Add corpus-level metrics for generated QA diversity, rejection rate, duplicate
  question rate, and class/context coverage.
- Add a trainer import smoke that reads `instruction_training_rows` and verifies
  the exact downstream loader shape.
