# Qwen Caption Training Dataset External Consumer Dossier

Date: 2026-06-29

## Purpose

This document explains the caption-derived VLM training-dataset work in terms
an external technical team can review without reconstructing the implementation
history from code and commits. It describes what was built, why it was built
that way, what artifacts are produced, what the UI exposes, what validation
gates protect the output, how long-running generation is hardened, and what
still needs real-data pilot proof before generated rows should be used for
fine-tuning.

The document is intentionally dataset-neutral. Do not add private dataset,
customer, partner, or project names to this file. The implementation should be
reviewable from its data contracts and behavior, not from organization-specific
context.

## Executive Summary

The caption panel now supports two different product workflows:

- ordinary caption assistance, where captions are generated, stored, promoted,
  edited, and exported for annotation work;
- VLM training-dataset creation, where the system builds an auditable
  image/question/answer corpus from captions, generated visual QA candidates,
  source annotations, deterministic label-derived QA, review decisions, and
  consistency reports.

The training-dataset workflow is launched separately through **Create VLM
training dataset**. It does not silently reuse **Caption all images** as a
training-data export. This separation is deliberate because a training corpus
has stricter requirements than a caption archive: every trainable row must be
traceable, validated, reviewable, and loadable by the trainer as normal
image/question/answer data.

The implementation is ready for external implementation review and a small
real-data pilot. It is not being claimed as a certified production corpus
generator until that pilot is generated, reviewed, re-exported, and loaded by
the trainer or a trainer-equivalent dry run.

## Original Product Contract

The requested workflow was not "one caption per image." It required a
multi-prompt dataset-construction layer with these products:

- one broad global caption per image, referred to in artifacts as `caption0`;
- zero or more VLM-generated visual question/answer candidates per image;
- optional deterministic question/answer rows derived from real labels;
- a per-image construction archive that preserves source evidence,
  generated-language candidates, deterministic QA, validation, rejection, and
  export decisions;
- a flattened JSONL export that a current image-question-answer trainer can
  consume.

The key trainer constraint is that the model-visible row is flat:

```json
{"image_path":"images/example.jpg","question":"How many people are visible?","answer":"{\"object_counts\":{\"Person\":2}}"}
```

Only `image_path`, `question`, and `answer` are training signal. Any optional
metadata is for audit, filtering, lineage, weighting, or metrics. It is not
hidden prompt text and the model does not train on it unless a fact is written
directly into the `question` or `answer` field.

That rule drives the whole implementation: source facts are construction
evidence first, and they become training signal only when the exporter writes
them into a normal prompt/answer row.

## What Was Built

### 1. Separate UI Path For Training-Dataset Creation

The caption panel now exposes a distinct **Create VLM training dataset** action
under the caption dataset controls. The operator can configure:

- generated QA rows per image;
- generated QA mix;
- generated answer format;
- whether `caption0` rows are included in trainer JSONL;
- whether generated QA rows are included in trainer JSONL;
- whether deterministic metadata QA is included;
- whether source labels are passed to the generator as read-only context;
- whether strict QA grounding is enforced;
- whether trainer JSONL requires a ready report before download.

The ordinary caption controls remain captioning controls. They do not mutate
the training-dataset contract unless the operator starts the explicit training
dataset workflow.

### 2. Per-Image Row Families

A training-dataset run can produce several row families for the same image:

- `caption0`: a broad image-description row;
- generated QA: model-written visual questions and answers;
- deterministic metadata QA: code-generated rows derived from trusted source
  annotations;
- rejected candidates: rows retained in the archive for audit but excluded from
  trainer JSONL.

Generated QA count is bounded. A count of `0` is valid and produces a
caption0-only instruction export when `caption0` is enabled. Deterministic
metadata QA is off by default because it changes the corpus from purely
image-language generation into explicit label-derived training.

### 3. Four Export Artifacts

The workflow produces coordinated artifacts from the same run:

- `caption_instruction_training.jsonl`: compact trainer rows with
  `image_path`, `question`, `answer`, and optional metadata.
- `caption_instruction_archive.jsonl`: one construction archive record per
  image, preserving source summaries, caption0, generated QA, deterministic QA,
  rejected candidates, selected rows, review metadata, and export metadata.
- `caption_instruction_review.jsonl`: one candidate-level row per reviewable
  item, with blank review decision and review note fields for human audit.
- `caption_instruction_report.json`: run-level readiness, provenance, quality
  metrics, consistency proof, row counts, rejection summaries, and blocking
  reasons.

The trainer JSONL is the model input. The archive, review file, and report are
the evidence that explains whether the model input should be trusted.

### 4. Trainer Import Compatibility

The Qwen training loader accepts the flat exported instruction rows directly
and turns each row into a two-turn image/question/answer conversation. The
loader validates required fields, duplicate canonical image/question pairs,
review state, validation state, row provenance, image alias resolution, and JSON
answer shape for JSON-formatted rows.

This makes the trainer loader a final safety boundary. If a trainer JSONL file
is edited after export, malformed, review-pending, rejected, duplicate, missing
provenance, or not tied to a real image, the loader rejects it rather than
silently training on it.

## Why The Implementation Is Layered

### Source Annotations Are Trusted Evidence

Source annotations are structured facts derived from real labels, human labels,
reviewed corrections, workflow signals, detector confidence already present in
the annotation workflow, or deterministic geometry over bounding boxes. They
are stored under `source_annotations` in the archive.

Generated text cannot populate these fields. A VLM may write a sentence saying
that several vehicles appear in an image, but that sentence cannot become an
object count, class list, spatial fact, or source uncertainty value.

### Generated Language Is Candidate Language

Generated captions and generated QA are stored as language annotations. They
can become candidate training rows, but only after validation and export
selection. They are useful because they provide varied natural-language
prompts, but they are not source truth.

### Deterministic Metadata QA Is Separate

Deterministic metadata QA is generated by code from trusted annotations. Its
answers are not guessed by the model. These rows are the path for teaching
class presence, counts, absence, and simple spatial facts from labels, because
those facts are written explicitly into the trainer-visible question/answer
row.

### Flattened Rows Are The Only Training Signal

The final trainer file is deliberately flat. It does not ask the trainer to
consume the archive or infer hidden facts from metadata. This keeps the training
contract simple and makes row review possible: if a fact should be learned, it
must appear in the `question` or `answer`.

## Data Ownership Contract

The archive uses clear ownership layers:

| Layer | Purpose | Can generated language write it? | Direct trainer input? |
| --- | --- | --- | --- |
| `source_annotations` | Trusted structured facts for the image | No | No |
| `language_annotations` | Caption and generated QA candidates | Yes | No, unless flattened |
| `deterministic_metadata_qa_pairs` | Code-generated QA from source annotations | No for answers | No, unless flattened |
| `flattened_training_rows` | Final selected `image_path` / `question` / `answer` rows | Exporter writes them | Yes |

Allowed truth flow:

```text
real labels / reviewed corrections / workflow signals
  -> source_annotations

source_annotations
  -> deterministic_metadata_qa_pairs

image + caption0 + read-only source_annotations + glossary
  -> language_annotations.generated_qa_pairs

validated language annotations + deterministic metadata QA
  -> flattened_training_rows
```

Forbidden truth flow:

```text
caption0
  -> source_annotations.object_counts

generated QA text
  -> source_annotations.visible_classes

generated scene summary
  -> source_annotations.spatial_facts

row metadata
  -> assumed hidden model input
```

## Generated QA Grounding

Generated QA can ask open visual questions, but rows that make structured
claims are handled conservatively.

When generated QA asks about supported structured facts such as object counts,
visible classes, class presence, negative presence, or simple spatial facts,
the final training answer is rewritten from `source_annotations`. The original
model answer remains in candidate metadata for audit.

When a structured claim cannot be validated from trusted source annotations, it
is rejected from trainer JSONL and retained in the archive/review artifacts
with a rejection reason. This avoids training on plausible but unsupported
label facts.

Strict grounding is enabled by default. It can preserve generated candidates
for review while preventing unsupported candidates from entering the flat
training file.

## Review Workflow

The workflow includes review export and import so generated-language candidates
can be audited outside the application.

The review JSONL contains candidate-level rows with:

- image path;
- dataset identity;
- QA id;
- row origin;
- question;
- candidate answer;
- selected training answer when applicable;
- validation status;
- selected-for-training flag;
- manual-review flag;
- source summary;
- rejection reasons;
- blank review decision;
- blank review notes.

Review import is metadata-only. It can attach decisions and notes to existing
caption0 and generated-QA records, but it cannot edit labels, boxes, image
paths, generated questions, generated answers, deterministic metadata QA, or
final annotations.

Review import fails closed for wrong datasets, stale text, stale selected
answers, missing row identities, unsupported row origins, duplicate targets,
malformed booleans, unsupported decisions, oversized text fields, deterministic
only packets, blank-action packets, and unmatchable actionable rows.

This design keeps human review valuable without turning review import into a
backdoor label or data mutation channel.

## UI And Operator Experience

The UI was hardened around the training-dataset workflow rather than just
adding an export button.

Important operator-facing behavior:

- The training dataset action is separate from caption actions.
- Instruction settings are shown in one compact control group below batch
  caption controls.
- The UI explains that generated QA never becomes source annotations.
- Disabling all trainable row families is blocked before launch.
- Generated QA can be archive/review-only when included in generation but not
  selected for trainer JSONL.
- Download buttons are separate for trainer JSONL, archive JSONL, review JSONL,
  and report JSON.
- Review import is a separate action.
- Caption action layouts wrap instead of clipping in the sidebar.
- Long status text cannot squeeze attach/recover buttons.
- Download-needed model options are shown in red, local model options stay in
  the normal light text color, and first-run model downloads require explicit
  operator approval.
- Auto output-token budgeting is separate from explicit user overrides.
- The UI keeps caption run settings locked while a backend job is mutating the
  caption archive so stale input events do not make a running job look like it
  captured new settings.

## Long-Running Job Hardening

Dataset-scale captioning is expected to run for long enough that browser state,
model crashes, prompt blowups, and mid-run exports have to be handled directly.

The implementation therefore includes:

- set-and-forget backend jobs as the durable default for multi-image work;
- progress, attach, recover, cancel, and completed-job handoff behavior;
- model-cache and model-download preflight;
- prompt-size measurement and prompt-budget telemetry;
- dense-box prompt reduction using representative spatial subsets while
  keeping full counts authoritative;
- prompt-aware output-budget adaptation when the user has not supplied an
  explicit numeric cap;
- model-aware output-token defaults for thinking-capable models;
- live streaming loop inspection for repeated tokens and repeated punctuation;
- safe retry after repeated-output loops;
- bounded child-process attempts for MLX/VLM failure isolation;
- deterministic count/layout fallback only as recorded degraded recovery;
- health gates for loop recovery, signal exits, deterministic fallback rate,
  failed attempts, prompt-budget adaptation, and projected duration;
- strict large-run launch gates requiring pilot certification and backend
  supervision before unattended scale-out.

This is the reason "set and forget" is now an operational mode, not just a
manual recovery button.

## Active-Job Consistency Guards

Caption datasets are mutable archives. A long-running job can be writing
caption records, generated QA, report state, and text-label mirrors while the
operator is viewing or exporting. The system now prevents stale or partial
reads from being mistaken for stable output.

Implemented guards include:

- backend refusal to start a second active caption dataset job for the same
  dataset;
- backend `caption_mutation_busy` gates for caption/text-label mutations while
  a job owns the same archive;
- backend `caption_read_busy` gates for caption archive reads and text-label
  mirror reads while a job owns the same archive;
- backend gates for caption export, review import, metadata writes, dataset ZIP
  download, and linked-dataset deletion;
- browser deferral of normal current-image archive reloads during active
  caption or instruction jobs;
- browser dropping in-flight archive responses if a backend job becomes active
  before the response applies;
- UI control locks for prompt stack, models, token limits, box limits, decode
  settings, health gates, set-and-forget settings, pilot controls, batch scope,
  glossary, caption output, dataset selection, and recipe application while the
  active archive is mutating.

Completed-job handoff can explicitly opt into reloading the archive after the
job finishes. Normal mid-run reads stay blocked or deferred.

## Artifact Validation Gates

There are several layers of validation before trainer rows are accepted.

Browser-side download validation blocks:

- blank image paths, questions, or answers;
- missing row metadata;
- unknown answer sources or answer formats;
- unknown validation or review states;
- rejected, invalid, failed, or needs-revision rows;
- invalid JSON answers for JSON-formatted rows;
- duplicate canonical image/question pairs;
- report row-count mismatches;
- archive duplicate image rows;
- review-row shape mismatches;
- missing dataset identity;
- missing or failed replicated consistency proofs.

Backend artifact consistency validation checks:

- per-image archive identity;
- canonical image paths;
- selected row counts;
- trainer rows versus archive selected candidates;
- trainer rows versus selected review rows;
- matching QA ids, normalized questions, and selected training answers;
- report counts and quality metrics;
- review-row shape and decisions;
- stale caption0 or generated-QA text during review import.

Trainer import validation checks:

- flat row shape;
- required `image_path`, `question`, and `answer`;
- image alias resolution;
- duplicate canonical image/question pairs;
- row provenance;
- accepted validation and review states;
- parseable JSON for JSON-formatted rows.

The same row can therefore be blocked in the UI, by backend export consistency,
or by the trainer loader.

## Prompt And Token-Budget Cleanup

Several caption failures came from prompt and output-budget fragility:

- the UI "max output tokens" control and backend runtime default could appear
  inconsistent;
- thinking-capable models needed much larger automatic budgets than short
  caption models;
- dense labels could produce prompts with too many boxes;
- repeated punctuation loops could consume the whole output budget;
- full-image composition carried enough context to trigger slow or unstable
  generation.

The cleaned-up contract is:

- explicit numeric user token caps remain hard overrides after validation;
- Auto mode can choose model-aware defaults and reduce them under prompt
  pressure;
- prompt size is measured and emitted as telemetry;
- dense box lists are reduced to representative spatial evidence while full
  counts stay authoritative;
- prompt text makes clear that a representative box subset is not an object
  absence claim;
- live output inspection detects loops before the budget is exhausted;
- recovery events are recorded instead of hidden.

This reconciles the need for high budgets on verbose models with the need to
avoid unbounded or misleading generation.

## Model Availability And Download Policy

The UI distinguishes local model choices from choices that require download.
Missing/download-needed model options are shown in red while local models stay
in the normal light text color, and backend jobs do not silently start
first-run model downloads unless the operator has enabled the model-download
setting.

Large unattended runs are stricter: first-run downloads are treated as a launch
risk, and strict set-and-forget readiness expects selected models to already be
available before GPU work starts.

## Empty Or Missing Label Files

An empty label file is a valid no-object annotation state, not automatically a
caption failure. The dataset-construction layer treats it as source evidence
for "no trusted labeled objects in this image" only when the file is part of
the dataset manifest and class map context. It does not ask the VLM to invent
label facts for labels that do not exist.

If an image has no labels, generated visual language can still describe the
image, but deterministic metadata QA is limited because there are no real
label-derived object counts, class lists, or spatial facts to export.

## File And Image Handling

The training packet is designed so the trainer can resolve images from
canonical image paths while archive and report artifacts preserve enough
provenance to audit image identity. Sample or handoff datasets should include
the image payloads when the goal is durability or external review, not only
paths into a local image root.

The implementation canonicalizes path aliases so `./image.jpg`,
split-prefixed paths, and nested relative paths do not create duplicate
training identities.

## What Is Ready Now

The implemented system supports:

- launching a full caption-derived VLM training-dataset job from the UI;
- generating caption0 and configurable generated QA;
- optionally generating deterministic source-label QA;
- exporting trainer, archive, review, and report artifacts;
- importing review decisions as metadata only;
- validating artifacts in browser, backend, and trainer loader layers;
- operating long-running jobs through set-and-forget backend execution;
- recovering from known repeated-output and runtime-fault classes with
  visible degraded telemetry;
- blocking stale archive reads, mutations, exports, review imports, metadata
  writes, and dataset deletion while an active job owns the same archive.

## What Is Not Claimed Yet

This work does not claim:

- that any generated corpus is training-ready before a pilot;
- that generated QA quality is sufficient on every target image distribution;
- that manual review throughput is acceptable at full scale;
- that deterministic metadata QA should be enabled for every fine-tuning run;
- that degraded recovery captions are acceptable above the configured health
  thresholds;
- that an unreviewed instruction JSONL should be treated as a final training
  corpus.

The correct claim is that the application path and artifact contracts now exist
and have focused automated coverage. Real corpus readiness still requires a
pilot and review.

## Recommended External Review Procedure

1. Read this dossier and the companion packet files.
2. Launch a small training-dataset job from the UI with set-and-forget enabled.
3. Include at least one image with labels and one image with no labels.
4. Download all four artifacts from the same run.
5. Confirm that `caption0`, generated QA, deterministic QA, rejected rows,
   selected rows, and report counts reconcile.
6. Review generated-language candidates in `caption_instruction_review.jsonl`.
7. Import review decisions.
8. Re-export trainer JSONL with the ready-report gate enabled.
9. Load the trainer JSONL through the Qwen training loader.
10. Run a small trainer dry run or batch loader smoke test.
11. Inspect health telemetry for loop recovery, signal exits, prompt-budget
    adaptation, failed attempts, deterministic fallback, and manual-review
    requirements.

## Required Pilot Before Training Use

Before using a generated corpus for fine-tuning, run a pilot that records:

- number of images;
- number of images with source labels;
- number of images with empty or missing labels;
- `caption0` selected/rejected counts;
- generated QA candidate count;
- generated QA selected count;
- generated QA rejection reasons;
- deterministic metadata QA row counts;
- manual-review requirement count;
- review accepted/rejected/needs-revision counts;
- duplicate image/question rejection count;
- JSON answer parse failure count;
- structured-claim rewrite count;
- artifact-consistency status;
- trainer-loader import result;
- loop recovery case rate;
- signal-exit case rate;
- deterministic fallback case rate;
- failed attempt row rate;
- p95 runtime projection.

Only after the pilot passes the configured health and artifact gates should the
same settings be used for a larger corpus.

## Companion Documentation

Use these files with this dossier:

- `docs/qwen_caption_training_dataset_complete_partner_packet.md`: broad
  external-review packet and reading order.
- `docs/qwen_caption_training_dataset_complete_external_handoff.md`: concise
  implementation handoff.
- `docs/qwen_caption_training_dataset_reviewer_dossier.md`: reviewer-focused
  summary.
- `docs/qwen_caption_instruction_dataset_hardening_report.md`: detailed
  hardening log.
- `docs/qwen_caption_prompt_stack.md`: prompt, token-budget, loop-recovery, and
  set-and-forget details.
- `docs/qwen_caption_ui_scenarios.md`: UI scenarios and operator behavior.

## Reproducible Verification Commands

The focused validation surface for this work is:

```bash
./.venv-macos/bin/python -m pytest \
  tests/test_qwen_caption_prompt.py \
  tests/test_qwen_caption_dataset_job.py \
  tests/test_labeling_panel_layout_contract.py \
  tests/test_dataset_linked_annotation_flows.py \
  tests/test_qwen_training_backend.py \
  tests/test_qwen_caption_flow_benchmark.py \
  tests/test_qwen_caption_soak_audit.py \
  tests/test_qwen_caption_operation_audit.py \
  tests/test_qwen_caption_unattended_launcher.py \
  tests/test_qwen_caption_soak_certification.py

node --check ybat-master/ybat.js

./.venv-macos/bin/python -m py_compile \
  localinferenceapi.py \
  api/datasets.py \
  models/schemas.py \
  ybat-master/ybat.js \
  tests/test_qwen_caption_prompt.py \
  tests/test_qwen_caption_dataset_job.py \
  tests/test_labeling_panel_layout_contract.py \
  tests/test_dataset_linked_annotation_flows.py \
  tests/test_qwen_training_backend.py

uvx ruff check \
  localinferenceapi.py \
  api/datasets.py \
  models/schemas.py \
  tests/test_qwen_caption_prompt.py \
  tests/test_qwen_caption_dataset_job.py \
  tests/test_labeling_panel_layout_contract.py \
  tests/test_dataset_linked_annotation_flows.py \
  tests/test_qwen_training_backend.py \
  --select E9,F63,F7,F82

git diff --check
```

For documentation-only changes, run at minimum:

```bash
git diff --check
rg -n -i "m[i]ril|w[a]ldo" docs ybat-master localinferenceapi.py api models tests tools --glob '!**/.git/**'
```

The restricted-name scan should return no matches in committed repository
files.
