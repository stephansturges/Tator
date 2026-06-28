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
backend job with isolated attempts, health gates, and auto-resume status.

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
crash-supervision readiness before the user walks away.

## 5. Launch A 10k-Scale Certified Run

The user enables **Require certified pilot**, enters a pilot artifact directory,
keeps **Pilot min cases** at 300 or higher, keeps prompt-budget telemetry on,
uses a positive prompt-token ceiling, and keeps deterministic-recovery
confidence enabled. The backend fails closed before GPU work if any strict
large-run gate is diagnostic-only.

## 6. Reuse A Saved Caption Recipe

The user saves or loads a caption recipe. Recipes carry style, prompt stack,
model, windowing, token, health-gate, set-and-forget, and pilot certification
settings, including deterministic-recovery confidence, without storing image
pixels, boxes, or generated captions.

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
downstream validation does not reject duplicate image/question pairs. No export
imposes a per-image caption limit, and all caption exports preserve caption
identifiers, image-local caption indexes, primary flags, sources, timestamps,
and metadata.
Before writing VLM JSONL, the UI validates that each row has an image path,
question, parseable caption JSON answer, and a unique image/question pair; bad
rows are blocked instead of downloaded.
Generated caption jobs append variants by default. Users promote one variant to
primary with **Set primary** or by enabling **Make generated caption primary**
before generation.

## 12. Create A VLM Instruction Dataset

The user clicks **Create VLM training dataset** after choosing a caption dataset.
The run creates caption0 rows and, by default, 8 generated visual QA rows per
image. **Generated QA per image** is clamped to 0-20, so `0` creates a
caption0-only instruction export and high values cannot explode the prompt or
artifact size. **Include caption0** and **Include generated QA** control the
flattened instruction JSONL, while **Include deterministic metadata QA** is off
by default and adds code-generated rows from real source labels only when
explicitly enabled. **Generated QA mix** controls whether the generated-QA pass
leans balanced, scene-level, object-focused, or caption-variant oriented.
**Generated answer format** controls whether generated answers are natural text
or parseable JSON. **Give generator read-only label context** may pass source
label counts into the generator as grounding context, but generated QA never
becomes source annotations. **Strict QA grounding** asks the model to answer only
from the image, caption0, or read-only source context.

**Download instruction JSONL** exports normal `image_path` / `question` /
`answer` rows. The trainer imports this flat shape directly and normalizes each
row into an image/question/answer conversation. The UI validates missing image
paths, blank questions, blank answers, invalid JSON for JSON row types, and
duplicate image/question pairs before writing the file.
**Download instruction archive** exports one per-image construction archive
record per JSONL line, keeping caption0, generated QA, optional deterministic
metadata QA, source annotation provenance, and per-image export metadata separate
from trainer rows.
**Download instruction report** exports run-level counts, rejection reasons,
source-field provenance, split image counts, split row counts, QA count per
image, and selected flattened-row counts.
