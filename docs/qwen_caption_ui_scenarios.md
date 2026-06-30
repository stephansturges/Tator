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

The user clicks **Caption all images** with set-and-forget enabled. The UI sends
the selected image first, then the remaining loaded images in display order and
wraps earlier images to the end, so the first backend heartbeat matches the
operator's current frame. The backend requires a selected caption dataset, keeps
jobs resumable, and reports backend crash-supervision readiness before the user
walks away. While attached to the job, the same caption progress panel and live
output toast used by single-image captioning must stay visible for each image:
current image index/name, prompt stack step chips, bounded prompt/output trace,
token preview, retry/cooldown state, and failure text. If the backend job cannot
be created, the caption status, backend-job status, and toast must show the
parsed launch failure instead of leaving the panel at a stale "Starting" state
or relying on console output.

Set-and-forget jobs own their generated caption writes. The backend may save a
caption for the active job id while ordinary UI/user caption mutations remain
blocked against the same dataset. Resume against an existing artifact directory
must reuse the artifact manifest's original case list rather than rebuilding the
case list from current text-label state, so a newly saved caption cannot shrink a
recovered job before it reconciles the original manifest.

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
The **Attach / recover selected dataset** button remains available for immediate
manual inspection without making manual recovery the normal path. The global
**Refresh backend jobs** and **Cancel active backend jobs** controls show and
stop active backend caption work even when it does not belong to the currently
selected caption dataset.
Both **Cancel caption** and **Cancel active backend jobs** cancel the persisted
artifact lineage, not only the currently visible wrapper process. A cancelled
set-and-forget artifact is tombstoned so startup and periodic auto-recovery do
not recreate a replacement caption job unless the user launches a fresh run.
Hard launch failures such as preflight, pilot certification, and backend
supervision failures are non-resumable until the failed gate is fixed. Interrupted
wrapper jobs without runner/preflight evidence are also non-resumable. In those
cases the backend returns a conflict, and the UI must show a "not resumable"
status instead of posting another resume request or spawning another wrapper job.

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

The user can click **Preview dataset prompts** after choosing a caption dataset
to inspect the first selected case's full caption prompt flow plus the
generated-QA prompt template and generated-QA verifier/rewrite prompt template
without running the model. The preview uses the same dataset-job payload builder
as the real launch and includes a caption0 placeholder for the generated-QA
prompt because runtime inserts the actual caption after the caption pass
completes. Pilot certification is a launch-only gate, so preview still opens
before a pilot artifact directory exists and warns that the real run remains
blocked until the pilot directory is supplied; when a pilot directory is already
selected, preview warns that launch will still certify it. After the preview
looks sane, the user clicks
**Create training dataset**.
The **Check caption readiness** control is available both near the main caption
controls and inside the Training dataset action row, so the operator can run the
same readiness check immediately before previewing or launching the dataset
job.
Training-dataset launches use the same caption request-field builder as
single-image captioning. The captured request therefore includes the same model,
token, prompt-stack, windowing, label-hint, count/box, and glossary fields that
the single-image caption path would send for the selected template image; later
UI edits apply only to the next job because the backend job persists a request
snapshot at launch.
The run starts with the selected image, follows backend progress frame by frame,
and creates caption0 rows plus, by default, 8 generated visual QA rows per
image. **Generated Q&A rows per image** is clamped to 0-20, so `0` creates a
caption0-only instruction export and high values cannot explode the prompt or
artifact size; if a typed value or loaded recipe is outside that range, the
launch confirmation warns that the value was adjusted. **Q&A top-up attempts**
is clamped to 0-12 and controls how many extra visual fallback prompts may run
to replace rejected or missing generated-QA rows before the image is marked
underfilled. **Include caption0** and **Include generated Q&A** control the
flattened instruction JSONL, while
**Include deterministic metadata Q&A** is off by default and adds code-generated
rows from real source labels only when
explicitly enabled. The UI refuses to launch an instruction run when all three
trainable row families are disabled, and its confirmation text distinguishes
trainer JSONL rows from generated QA candidates that are archive/review-only.
**Generated Q&A mix** controls whether the generated-QA pass
leans balanced, scene-level, object-focused, or caption-variant oriented.
**Generated answer format** controls whether generated answers are natural text
or parseable JSON. **Imposed questions** accepts one required question per line
or a JSON array; imposed questions are answered first when possible, and answers
that are not findable in the image can say so directly instead of being invented
or silently dropped. **Give generator read-only label context** may pass source
label counts into the generator as grounding context, but generated QA never
becomes source annotations. **Strict QA grounding** asks the model to answer only
from the image, caption0, or read-only source context. **Restrict speculative
Q&A language** is off by default; leave it off when the training set should
learn grounded "unknown", "not visible", or mild inference answers, and enable it
only when generated QA must contain directly findable facts with no speculation
or unavailable-information answers.
Generated-QA prompts always remain visual when they ask for new rows: the
primary prompt and every top-up fallback prompt attach the image to the VLM
call. Only the verifier/rewrite pass is text-only, because it rewrites or
rejects already-generated candidate text rather than asking for new visual
content.
Generated QA also has a verifier/rewrite gate before export. Rows that already
pass deterministic checks are marked `machine_validated`; rows with raw label
names or malformed questions get one text-only rewrite attempt. If the optional
speculative-language restriction is enabled, speculative or unavailable-answer
wording is also rejected or rewritten. Rewritten rows keep their original
question/answer in metadata, while failed rows remain in the audit archive as
rejected candidates and are excluded from flattened trainer JSONL. Caption and
generated-QA prompts must use glossary broad terms as canonical class names; if
a class has no glossary entry, they use a natural English fallback rather than
internal labelmap spellings.
While a caption or instruction job is mutating the caption archive, the
instruction row-family and generated-QA setup controls are disabled so the UI
does not imply that mid-run edits will change the active backend job. They
re-enable after the archive is stable and apply to the next launch/export
cycle. Caption image, caption batch, caption-all, and training-dataset launch
paths also repeat the same archive-idle check at action time, so stale buttons
or scripted clicks do not start a second caption job while the first job is
still mutating the archive.
As each image succeeds, live backend progress may include the latest generated
QA pairs. The UI shows those pairs as bounded status toasts once per
job/image/question so the operator can see caption0 and generated QA quality
without waiting for the final bundle. The same latest pairs are also mirrored
into a read-only JSON textarea so the operator can inspect the full question,
answer, validation status, and rejection reasons after the toast disappears.
During the generated-QA subpass, the UI also shows a compact per-image
accumulator with the current image index, caption0 status, accepted/target QA
count, rejected count, and per-prompt-profile accepted/rejected totals for the
primary prompt plus any caption-grounded or sparse-scene top-up attempts. This
live status must make underfilled cases explicit, for example `Continuing with
5/8`, instead of forcing the operator to infer it from logs.

Generated-QA failure is not a full-image failure for the default caption0 plus
generated-QA path. If caption0 succeeds and generated QA loops, returns invalid
JSON, or verifies to fewer accepted rows than requested, the backend keeps
caption0, runs visual generated-QA top-up prompts with the image still attached,
and then records an audit warning if the accepted total remains underfilled. A
QA-only configuration remains strict and fails the case when no valid
generated-QA row exists.

The training-dataset panel is organized as **Preview prompts**, **Create
dataset**, **Review readiness**, and **Download bundle**. The status strip shows
whether a caption dataset is selected, whether the model backend is available,
whether a caption/training job is active, the latest readiness state, and the
reason actions are disabled. **Download training bundle** is the recommended
handoff artifact because it includes copied image bytes, effective labels, the
trainer JSONL, construction archive, review file, readiness report, and checksum
manifest. Completed training-dataset backend jobs also materialize that same
bundle plus `caption_instruction_report.json` into the job output directory
under `instruction_artifacts/`, so a long run does not depend on the operator
remembering to click a browser download before leaving the page.

**Download trainer JSONL** exports normal `image_path` / `question` /
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
blocked by default by **Only export training data when ready**. Operators can disable that gate only for
deliberate review-pending diagnostics. Scripted exports can use
`require_ready_instruction_export=true` for the same server-side refusal; the
UI sends that backend gate for trainer JSONL and the bundle when **Only export
training data when ready** is checked. Construction archive, review file, and
readiness report downloads deliberately do not send the ready gate because those
diagnostic artifacts are needed to fix
a not-ready corpus. A selected row that a reviewer marks rejected or
needs-revision moves readiness to `blocked` until the row is removed,
regenerated, or accepted.
Instruction artifact downloads and reviewed-JSONL import stay disabled until a
caption dataset is selected and no caption or instruction job is actively
mutating the caption archive, so operators do not export or import against a
missing dataset or a moving target. The same active-job check runs again inside
each export/import action, covering stale clicks, already-open file pickers, or
scripted UI calls that bypass the disabled button state. The reviewed-JSONL
import button also checks this guard before opening the browser file picker, so
operators are not asked to choose a file for an import that is already blocked.
Any backend, validation, or
busy-state failure from caption audit JSONL, grouped caption JSON, VLM caption
JSONL, instruction JSONL, archive, review, report, or review import actions
must update the caption export health row and the status toast with the same
formatted message, without double-prefixing already formatted blocked messages.
Busy responses from a live caption dataset job remain "blocked" operator
messages rather than being relabeled as generic export/import failures.
The diagnostic downloads live under **Advanced exports and review** so the main
path stays focused on preview, creation, readiness, and bundle export. **Download construction archive** exports one per-image construction archive
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
**Download review file** exports one candidate-level row for each caption0,
generated QA, and deterministic metadata QA item. Review rows preserve candidate
answers, selected training answers, source summaries, rejection reasons,
selected-for-training flags, and blank review decision/note fields so a human
can audit the corpus before fine-tuning. **Import review decisions** reads that
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
**Download readiness report** exports run-level counts, rejection reasons,
source-field provenance, split image counts, split row counts, QA count per
image, selected flattened-row counts, duplicate-question/diversity metrics,
structured-rewrite rates, answer-format distribution, source-class coverage,
and the `ready` / `needs_review` / `blocked` training-readiness status. The
report download also validates the sibling trainer, archive, and review row
payloads plus cross-artifact consistency before saving, so a report is not
exported as the authoritative run summary when its sibling artifacts are stale
or malformed.
