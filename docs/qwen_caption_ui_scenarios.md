# Qwen Caption UI Scenarios

This checklist keeps the caption panel aligned with the product modes exposed in
the UI. The durable default is set-and-forget; direct single-image captioning is
only a diagnostic escape hatch.

## 1. Caption The Current Image With Defaults

Before loading a slow test dataset, the user clicks **Check caption readiness**
to verify that the critical caption controls, defaults, exports, backend origin,
set-and-forget supervision signal, uncapped alternate-caption archive status,
and VLM export validation status are present. The user then selects a caption
dataset and clicks **Caption image**. The set-and-forget checkbox is already
enabled, generated captions append as saved alternate caption records, and
**Make generated caption primary** is off unless the user deliberately wants a
new generated variant to replace the selected primary. The selected primary
caption mirrors to the legacy text-label file. The job runs as a persisted
backend job with isolated attempts, health gates, and auto-resume status. If a
caption/text-label save races an active dataset caption job or annotation lock,
the caption status and toast show the parsed guard reason and retry guidance
instead of raw backend JSON or a generic "Save failed" message. Manual caption
archive controls such as **Save as new alternate**, **Update selected**,
**Set primary**, and **Delete** only show success after the underlying action
reports a real mutation, so stale clicks or disabled-control bypasses cannot
claim that a caption record changed when nothing happened. Those controls also
visibly disable while a caption or instruction job is mutating the caption
archive, and they re-enable only when an image, caption text, and the required
selected caption state are available.
The caption output textarea itself is read-only during active caption or
instruction jobs. The worker can still stream or write generated text, but
manual input/blur events cannot overwrite the live output or autosave text-label
edits while the archive is mutating.
The caption dataset picker and dataset-refresh control are also disabled while
the archive is mutating, so the panel cannot switch to a different caption
dataset context while a backend job is still writing to the current archive.
Caption archive reloads are also deferred while the selected archive is
mutating. Image navigation or stale scheduled loads keep the current caption
view and update the backend-job status instead of fetching a half-updated
archive. If a fetch started before the backend job became active, its response
is dropped rather than repainting the caption textarea or alternate list from a
stale snapshot. The only active-job reloads allowed are explicit completion
handoffs from the job that just finished.
The caption run-configuration surface is frozen for the same reason: prompt
layers, style/opening text, glossary edit/reset/save controls, windowing,
models, token and box limits, decode controls, set-and-forget/pilot settings,
health gates, save/promote behavior, and batch scope controls are disabled
while the archive is changing. Mid-run settings edits therefore cannot look as
if they will alter a backend job that already captured its request payload.

## 2. Run A Direct Diagnostic Caption

The user deliberately unchecks **Set-and-forget backend run** to test one current
image in the active browser flow. The workflow summary must identify this as a
direct diagnostic and must not imply crash-safe unattended behavior.

## 3. Caption The Next N Images

The user enters a batch count and clicks **Caption next N**. The same
set-and-forget controls, health gates, model-download setting, and pilot
certification payload apply to the backend batch job.

## 4. Caption All Images As A Walk-Away Job

The user clicks **Caption all images** with set-and-forget enabled. The backend
requires a selected caption dataset, keeps jobs resumable, and reports backend
crash-supervision readiness before the user walks away. If the backend job
cannot be created, the caption status, backend-job status, and toast must show
the parsed launch failure instead of leaving the panel at a stale "Starting"
state or relying on console output.

## 5. Launch A 10k-Scale Certified Run

The user enables **Require certified pilot**, enters a pilot artifact directory,
keeps **Pilot min cases** at 300 or higher, keeps prompt-budget telemetry on,
uses a positive prompt-token ceiling, and keeps deterministic-recovery
confidence enabled. The backend fails closed before GPU work if any strict
large-run gate is diagnostic-only. If the pilot, crash-supervision, or preflight
gate fails, the backend-job status must show the failed gate's human-readable
detail, not only an internal runner error code.

## 6. Reuse A Saved Caption Recipe

The user saves or loads a caption recipe. Recipes carry style, prompt stack,
model, windowing, token, health-gate, set-and-forget, and pilot certification
settings, including deterministic-recovery confidence, without storing image
pixels, boxes, or generated captions. Loading or uploading a recipe is blocked
while a caption or instruction job is mutating the caption archive, because a
recipe can rewrite the same controls that describe the active training-dataset
run. Saving, deleting, or downloading browser-local recipes remains separate
from the active backend job.

## 7. Select A Missing Model Intentionally

The user opens a model dropdown. Download-needed model options are red and local
model options are white. Backend caption jobs fail preflight when a selected
model is missing unless **Allow model downloads** is explicitly enabled.

## 8. Use A Thinking Model Safely

The user chooses a Thinking model. **Max output tokens** can stay blank so Auto
uses a high Thinking-safe budget with prompt-aware reduction; a numeric value is
still a hard user override. Greedy deterministic decoding stays unavailable for
Thinking models.

## 9. Caption Dense Label Scenes

The user leaves **Max boxes** at `0 = Auto`. Counts remain authoritative, while
the prompt box list becomes representative for dense scenes. The UI and prompt
preview must make clear that omitted boxes are not absent objects.

## 10. Auto-Recover Or Attach To Existing Work

The user returns after a backend restart or browser reload. Set-and-forget jobs
auto-resume on backend startup and periodic sweeps when supervision is running;
the page also auto-attaches to relevant active work when it sees backend state.
The **Attach / recover now** button remains available for immediate manual
inspection without making manual recovery the normal path.

## 11. Export Alternate Captions For Training

The user saves multiple captions for the same image. **Download captions JSONL**
exports one audit record per caption, while **Download grouped JSON** exports
one object per image with all of that image's captions in primary-first order.
The backend `/captions/export` endpoint exposes the same versioned grouped
archive object, so browser downloads and API/script exports share one logical
multi-caption contract rather than separate shapes.
**Download VLM JSONL** exports one normal `image_path` / `question` / `answer`
training row per caption. VLM answers are JSON strings of the form
`{"caption": "..."}`, and per-image questions are varied by caption index so
downstream validation does not reject duplicate canonical image-path/question
pairs. No export imposes a per-image caption limit, and all caption exports
preserve caption identifiers, image-local caption indexes, primary flags,
sources, timestamps, and metadata.
Before writing VLM JSONL, the UI validates that each row has an image path,
question, parseable caption JSON answer, and a unique canonical image-path/question pair; bad
rows are blocked instead of downloaded. If a malformed backend or cached payload
returns no row array, the validator reports that the VLM rows must be an array
instead of throwing a browser exception.
Generated caption jobs append variants by default. Users promote one variant to
primary with **Set primary** or by enabling **Make generated caption primary**
before generation.

## 12. Create A VLM Instruction Dataset

The user clicks **Create VLM training dataset** after choosing a caption dataset.
The run creates caption0 rows and, by default, 8 generated visual QA rows per
image. **Generated QA per image** is clamped to 0-20, so `0` creates a
caption0-only instruction export and high values cannot explode the prompt or
artifact size; if a typed value or loaded recipe is outside that range, the
launch confirmation warns that the value was adjusted. **Include caption0** and
**Include generated QA** control the flattened instruction JSONL, while
**Include deterministic metadata QA** is off by default and adds code-generated
rows from real source labels only when
explicitly enabled. The UI refuses to launch an instruction run when all three
trainable row families are disabled, and its confirmation text distinguishes
trainer JSONL rows from generated QA candidates that are archive/review-only.
**Generated QA mix** controls whether the generated-QA pass
leans balanced, scene-level, object-focused, or caption-variant oriented.
**Generated answer format** controls whether generated answers are natural text
or parseable JSON. **Give generator read-only label context** may pass source
label counts into the generator as grounding context, but generated QA never
becomes source annotations. **Strict QA grounding** asks the model to answer only
from the image, caption0, or read-only source context.
While a caption or instruction job is mutating the caption archive, the
instruction row-family and generated-QA setup controls are disabled so the UI
does not imply that mid-run edits will change the active backend job. They
re-enable after the archive is stable and apply to the next launch/export
cycle. Caption image, caption batch, caption-all, and training-dataset launch
paths also repeat the same archive-idle check at action time, so stale buttons
or scripted clicks do not start a second caption job while the first job is
still mutating the archive.

**Download instruction JSONL** exports normal `image_path` / `question` /
`answer` rows. The trainer imports this flat shape directly and normalizes each
row into an image/question/answer conversation. The UI validates missing image
paths, blank questions, blank answers, required row metadata,
instruction archive provenance, missing or unknown validation/review state,
rejected/failed/invalid validation state, non-trainable review state, invalid
JSON for JSON row types, and duplicate canonical image-path/question pairs before writing the
file. If the returned trainer-row payload is missing or not an array, the
validator reports the malformed artifact and blocks export instead of throwing.
It also validates the instruction report's training-readiness block:
`blocked` readiness refuses the download, while `needs_review` readiness is
blocked by default by **Require
ready report for trainer JSONL**. Operators can disable that gate only for
deliberate review-pending diagnostics. Scripted exports can use
`require_ready_instruction_export=true` for the same server-side refusal; the
UI sends that backend gate for trainer JSONL when **Require ready report for
trainer JSONL** is checked. Archive, review, and report downloads deliberately
do not send the ready gate because those diagnostic artifacts are needed to fix
a not-ready corpus. A selected row that a reviewer marks rejected or
needs-revision moves readiness to `blocked` until the row is removed,
regenerated, or accepted.
Instruction artifact downloads and reviewed-JSONL import stay disabled until a
caption dataset is selected and no caption or instruction job is actively
mutating the caption archive, so operators do not export or import against a
missing dataset or a moving target. The same active-job check runs again inside
each export/import action, covering stale clicks, open file pickers, or scripted
UI calls that bypass the disabled button state. Any backend, validation, or
busy-state failure from caption audit JSONL, grouped caption JSON, VLM caption
JSONL, instruction JSONL, archive, review, report, or review import actions
must update the caption export health row and the status toast with the same
formatted message, without double-prefixing already formatted blocked messages.
Busy responses from a live caption dataset job remain "blocked" operator
messages rather than being relabeled as generic export/import failures.
**Download instruction archive** exports one per-image construction archive
record per JSONL line, keeping caption0, generated QA, optional deterministic
metadata QA, source annotation provenance, and per-image export metadata separate
from trainer rows. The archive download is blocked if row-level validation
passes but the archive row count no longer matches the report image count or
archive image count, or if the archive contains duplicate image paths. A missing
or non-array archive payload is reported as a validation error rather than a UI
exception. The
backend also emits `instruction_artifact_consistency`; when it is not OK, the
report readiness is blocked and the browser refuses the related download.
This guard compares artifact row identities as well as counts, so trainer rows,
selected review rows, and archive candidates must agree on image path, QA id,
normalized question, per-image selected counts, and selected answers where
available.
Flat-layout image keys are canonicalized before export so a saved caption and a
manifest row for the same nested image are not exported as duplicate instruction
objects.
**Download review JSONL** exports one candidate-level row for each caption0,
generated QA, and deterministic metadata QA item. Review rows preserve candidate
answers, selected training answers, source summaries, rejection reasons,
selected-for-training flags, and blank review decision/note fields so a human
can audit the corpus before fine-tuning. **Import reviewed JSONL** reads that
artifact after a reviewer fills accepted, rejected, or needs-revision decisions
and applies only review metadata to matching saved caption and generated-QA
records; rows from a different dataset are blocked before import. The backend
also rejects malformed, stripped, or stale actionable language review rows
before writing any review metadata, including persisted language decisions
missing an embedded dataset id, rows whose QA id is known but whose image path
no longer matches the stored record, whose reviewed question/answer or caption
text is missing, rows whose stable QA id is missing or does not match a saved
record, rows with non-boolean selection/review flags or missing review columns,
or rows whose image path still matches but whose reviewed text or selected
training answer is no longer current, so mixed valid/stale packets do not
partially apply. The browser
import preflight also catches
unsupported actionable row
origins and duplicate or conflicting actionable review targets before sending
the packet, and it rejects oversized review files before reading them into
browser memory so a bad review packet cannot lock the UI. Review row text
fields also have explicit type and length limits, so an otherwise valid packet
cannot persist huge notes or non-text fields into caption metadata. The backend
also rejects rows that use different row identities but resolve to the same
saved caption or generated-QA record. Review
exports are blocked if caption0 or generated-QA rows are missing dataset
identity, even before a reviewer fills a decision, so exported packets are
import-ready by construction. It
filters deterministic-only review files because deterministic
rows are rebuilt from source labels rather than persisted; backend API/script
imports also reject blank-decision or deterministic-only packets instead of
reporting zero persisted decisions as applied work. It does not edit source
labels, generated answers, or deterministic metadata rows.
Backend import failures are translated into row-specific operator messages,
including stale caption0/generated-QA text, dataset mismatch, duplicate
actionable decisions, resolved duplicate saved-record decisions, unsupported row
origins, missing dataset identity, missing generated-QA question/answer text,
missing stable QA ids, stale selected training answers, unresolved image
context, and caption0 rows that would create a saved caption without a
synthetic id matching the selected dataset,
resolved image key, and current text-label caption.
Rejected or needs-revision language candidates stay auditable in the archive and
review JSONL but are excluded from flattened trainer rows. Review downloads are
also blocked when the review-row count, selected review-row count, or
manual-review row count disagrees with the instruction report. Missing or
non-array review-row payloads are surfaced as validation failures with zero row
count rather than crashing the export action.
**Download instruction report** exports run-level counts, rejection reasons,
source-field provenance, split image counts, split row counts, QA count per
image, selected flattened-row counts, duplicate-question/diversity metrics,
structured-rewrite rates, answer-format distribution, source-class coverage,
and the `ready` / `needs_review` / `blocked` training-readiness status.
