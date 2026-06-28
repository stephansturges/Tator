# Qwen Caption Prompt Stack Contract

This note records the captioning prompt-stack invariants that prevent the same
failure modes from returning during future prompt edits.

## Contract

- Prompt preview and live caption generation must use the same evidence builder.
- All reusable caption prompt layers must be visible in the Caption prompt stack
  and portable through caption recipes. The editable layers are: combined user
  request, main system prompt, detection context prompt, window/crop prompt,
  draft/refine prompt, window merge prompt, cleanup prompt, editor system
  prompt, coverage/count guard prompt, and English rewrite prompt.
- Image-specific context is generated, not recipe-authored: image pixels,
  image tokens, boxes, counts, window bounds, matched full-frame object IDs,
  raw model outputs, and generated captions are visible in the complete prompt
  preview but must not be saved into portable recipes.
- Raw label names are backend identifiers. Model-facing prompt prose must use
  validated natural terms from the glossary or a safe naturalized fallback.
- Malformed glossary fragments such as `[`, `]`, `{`, `}`, empty text, or
  punctuation-only strings must never become class names, counts, box labels, or
  allowed-class text.
- Window captions, full-image composition, merge, cleanup, coverage, and
  language-rewrite prompts must receive the same class/count/glossary context.
- Window captions must not inherit the full-image detection-context prompt
  wholesale. They use compact crop/window context plus the editable Window
  prompt so crop observations do not fight the full-caption length policy.
- Window-local caption evidence must be reconciled back to full-frame annotation
  objects before full-image composition and merge prompts. Full-frame counts stay
  authoritative; window captions add close-up detail, not extra inventory.
- Internal object reference IDs in reconciliation text are prompt-only merge
  aids. They must never be presented as final caption content.
- Window location text must be generated from crop bounds, not crop center alone.
  Overlapping 2x2 crops should read as upper-left, upper-right, lower-left, and
  lower-right sections with percent bounds as prompt-only layout aids.
- Window caption sentence limits are explicit caption settings. They are not
  inferred from the final full-image caption length.
- Caption generation is high-cap rather than uncapped. Detailed/windowed calls
  may auto-lift output budgets, but loop detection and cleanup guardrails remain
  active.
- Only visual observation stages should send image tensors to Qwen: window
  captioning, full-image draft, and full-image composition. Caption editor
  stages such as window merge, cleanup, coverage/count guard, English
  rewrite, draft refinement, and truncated-caption repair must use the
  text-only Qwen editor path. This keeps large postprocess prompts from
  resubmitting image tensors to MLX/Metal.
- If an MLX caption generation enters a repeated-output loop, the request must
  stop that attempt as a controlled caption loop, unload the Qwen runtime, and
  use the configured loop-recovery policy. It must never continue into another
  editor/refine stage with the same unstable runtime, because the next Metal
  command can fault the backend.

## Caption Loop Recovery

Caption loop recovery is deliberately progressive and observable. It exists to
keep the captioning product path usable without hiding model/runtime failures.

1. The requested caption stage runs first with the user-selected model, prompt,
   and decode settings.
2. If loop detection fires, the backend records the stage, model, decode
   settings, and loop reason in `recovery_events`, logs the event into the
   caption trace, marks the exact stage/model/prompt/decode combination on a
   short cooldown, and unloads the Qwen runtime. MLX caption Unicode decode
   failures follow the same recoverable ladder with
   `recoverable_generation_error` events; streaming decode failures must not
   fall through to blocking MLX generation inside caption mode.
3. `Safe retry only` retries the same model once with safer decoding: non-
   Thinking models switch to deterministic generation; Thinking models keep low
   temperature sampling; both tighten repetition controls and use a bounded
   recovery token cap.
4. `Safe retry + fallback` normally performs that safe retry and, if it still
   loops, runs the same stage with the fallback model. The default
   `Auto stable fallback` selects the caption-safe Instruct/editor model for the
   active runtime. If no distinct fallback model is available, recovery continues
   instead of failing the attempt.
5. One recovery path is intentionally earlier than same-model visual retry:
   when the full-image visual composition stage in windowed mode loops after the
   window observations have already succeeded, the backend first composes a
   text-only caption from those completed window observations, full-frame counts,
   and the user request. That avoids immediately resending the same large
   full-image tensor/prompt combination into MLX after it has just emitted a
   repeated-output loop.
6. Non-windowed full-image visual composition also has an unattended recovery
   path. After the primary visual pass and safe visual retry loop, the backend
   composes a text-only caption from authoritative counts, class inventory,
   representative spatial layout, and the user request. This is marked in
   `recovery_events`.
7. If every model-based recovery path loops, fails, or is unavailable, `Safe
   retry + fallback` can still finish with a deterministic count/layout caption.
   This is a degraded set-and-forget fallback, not the normal captioning path,
   and it is recorded as `deterministic_recovery_succeeded`.
8. The unattended parent runner has one final set-and-forget fallback outside
   the child process. If all child attempts are exhausted and the case has
   authoritative object counts, the parent writes a deterministic count/layout
   caption row with `parent_deterministic_recovery=true`, a
   `deterministic_recovery_succeeded` event, and a normal `captions.jsonl`
   record. If no counts are available, the case remains a terminal failed case
   for manual review because the parent has no reliable visual content to
   summarize.
9. `Off` reports the controlled loop failure after unloading the runtime.
10. When cooldown is enabled, an exact stage/model/prompt/decode combination that
   already looped skips the risky primary attempt during the cooldown window and
   goes straight to the configured recovery path.

The UI exposes these controls in **Generation and guards**:

- `Loop recovery`: `Safe retry + fallback`, `Safe retry only`, or `Off`.
- `Fallback model`: `Auto stable fallback`, `No fallback model`, or a concrete
  caption-capable Qwen model.
- `Cooldown repeated loops`: protects against immediately retrying the exact
  same known-bad combination.

Recovery must remain visible. API responses include `recovery_events`, saved
caption records keep those events, and the caption toast/trace receives the
recovery event text so the user can tell whether a final caption came from the
original model, text-only window-evidence recovery, a safe retry, or a fallback
model, prompt-context recovery, or deterministic count/layout recovery. Caption
progress has a shorter stale timeout than general Qwen jobs so a blocked MLX
caption generation releases the UI state in minutes rather than remaining active
for the general backend timeout.

## Caption Recipes

Caption recipes are browser-local until downloaded as JSON. A portable recipe is
allowed to contain reusable setup only:

- caption style text and opening phrase guidance
- every editable prompt-stack layer
- caption scope, windowing, model, and generation controls
- optional editable glossary text

Recipes must not contain image payloads, per-image boxes, label hints, used
counts, generated captions, prompt-preview placeholders, or backend image tokens.
Uploaded recipe JSON is applied to the current controls and saved locally only
after it passes this reusable-setup shape check.

The complete prompt-flow preview remains the source of truth for a specific
image. It combines recipe-controlled prompt layers with generated per-image
evidence and shows conditional templates for cleanup, coverage/count guards,
and English rewrite guards.

## Token Budgets And Trace Visibility

For UI-level walkthroughs, see
[`docs/qwen_caption_ui_scenarios.md`](qwen_caption_ui_scenarios.md).

Caption token control uses an explicit Auto/override split:

- A blank `Max output tokens` control means Auto. Auto keeps ordinary
  Instruct/full-image calls compact. Thinking models get the high caption budget
  because Thinking traces can consume far more generated tokens before the final
  caption.
- Instruct detailed/windowed calls use a modest automatic visual budget and the
  text-only merge pass uses a modest editor budget. The prompt-aware budget check
  then measures rendered prompt size and reduces non-explicit output budgets when
  the prompt itself is large.
- Dense full-image Instruct prompts do not get a larger output budget merely
  because many boxes are present; those scenes should be summarized from counts
  and representative layout evidence rather than encouraged to generate thousands
  of extra tokens.
- A numeric `Max output tokens` value is a hard per-call output cap after the
  normal schema clamp. Windowed mode, merge, cleanup, and guard passes must not
  silently raise an explicit user value such as `2000` to a larger budget.
- Runtime progress should report the effective output-token budget actually used
  by generation so UI traces reconcile with the user setting. Caption IO traces
  log both `requested_max_new_tokens` and `effective_max_new_tokens` in a
  `prompt_budget` event after the rendered prompt has been measured.
- Streaming generation acts as the live output inspector: repeated caption
  surfaces are detected while tokens arrive, the stream is closed, the runtime is
  unloaded, and recovery is routed through the configured safe path. This
  inspector reduces loop damage, but it does not replace the hard max-token cap:
  blocking generation paths, non-repeating runaway prose, and dependency
  fallback paths still need a finite fuse.
- The live inspector is an incremental guard shared by streaming caption paths.
  When it fires, the partial repeated text may be trimmed for logging/progress,
  but the caption call still raises a controlled loop error so a cleaned fragment
  such as a single punctuation mark is not accepted as a successful caption.
- A `Max boxes` value of `0` means Auto. Auto preserves authoritative class
  counts but caps very dense prompt box lists to a representative spatial subset
  selected for class coverage and image-region spread, so MLX full-image vision
  prompts do not receive hundreds of coordinate entries in one generation call.
  The UI and generated default prompt text must make clear that this box list is
  representative rather than exhaustive; omitted boxes are not absent objects.
  Numeric values remain explicit user caps and use the same representative
  selection when they truncate a dense box list.
- Detailed/windowed captioning has an explicit full-image composition strategy.
  `visual` sends the full image again after crop observations. `text_only`
  composes from completed window observations, authoritative counts/classes,
  prompt context, and representative box evidence without another full-image
  tensor. Browser Auto resolves to `text_only` for set-and-forget windowed runs
  and `visual` for direct/manual diagnostics; CLI set-and-forget wrappers apply
  the same default unless `--windowed-full-image-strategy visual` is explicit.
  Pilot certification and resume fingerprints include this strategy because the
  two flows have different MLX crash surfaces and cannot be treated as the same
  caption configuration.
- Prompt preview reports a prompt-size and effective output-budget estimate for
  each caption stage. The estimate is for operator visibility and test coverage;
  runtime still remeasures the rendered chat prompt with the active
  processor/tokenizer when available, then applies prompt-aware output budgeting
  before generation.
- The benchmark/soak runner records `preview_prompt_budget` per image so large
  unattended batches can be audited for prompt growth without opening every
  prompt artifact by hand.
- The benchmark/soak runner also summarizes copied per-run `qwen_caption_io`
  JSONL traces into each attempt row. `degraded_rates` now exposes
  `stream_loop_detected_*`, `loop_trim_*`, and `loop_guard_*` counters, so a
  set-and-forget run records when the live output inspector stopped a
  repeated-token stream instead of leaving that evidence only in transient UI
  progress or raw logs. Unbound
  `qwen_caption_io_latest.*` files are live/manual diagnostics only; unattended
  evidence must be tied to the worker's mirrored Qwen `run_id`, and missing
  per-run traces are recorded as missing instead of being filled from a stale
  global latest file.
- Each worker attempt mirrors compact Qwen progress into `worker_progress.json`
  and append-only `worker_progress.jsonl`. The parent runner copies changed
  worker progress into `heartbeat.json`, including the active Qwen step, message,
  and token counters. This makes long stages such as **Compose full-image
  caption** visible to the backend and audit artifacts without treating chattery
  stdout as progress.
- MLX caption image tensors are downscaled on the generation path by default
  (`QWEN_CAPTION_MLX_MAX_IMAGE_SIDE`, default `512`). This does not alter source
  image dimensions, box coordinates, window geometry, or saved annotations; it
  only reduces caption-generation GPU pressure.
- Backend set-and-forget dataset jobs use a safer first-attempt image side by
  default (`QWEN_CAPTION_SET_AND_FORGET_MLX_MAX_IMAGE_SIDE`, default `224`)
  while retaining the normal `512` default for direct/manual diagnostics. A job
  request can override this with `runner_mlx_max_image_side` in
  `caption_request` when intentionally trading stability for more visual detail.
  The matching set-and-forget retry floor defaults lower
  (`QWEN_CAPTION_SET_AND_FORGET_MIN_RETRY_IMAGE_SIDE`, default `192`), so a
  native signal exit on the first child attempt can still retry with a smaller
  tensor. The direct supervisor and the 10k unattended wrapper apply the same
  `224`/`192` defaults whenever `--set-and-forget` is active and those flags
  were not supplied explicitly. The same set-and-forget profile inserts a
  short successful-case pacing delay
  (`QWEN_CAPTION_SET_AND_FORGET_COOLDOWN_AFTER_SUCCESS_SECONDS`, default `5`)
  before starting the next child process; CLI users can override it with
  `--cooldown-after-success`, and backend/API jobs can override it with
  `cooldown_after_success_seconds` or `runner_cooldown_after_success_seconds`.
  Set-and-forget also defaults to three isolated VLM child attempts before
  deterministic count/layout fallback, while manual diagnostics remain at two
  attempts unless the operator enters an explicit override. This extra attempt
  keeps rare MLX child aborts on the model-captioning path instead of spending
  the deterministic fallback budget too early.
  Backend, CLI, pilot-generation, and launchd handoffs therefore share one
  walk-away runtime profile unless the operator explicitly changes it.
- In unattended benchmark/soak runs, later attempts for the same image use an
  adaptive retry profile that can reduce only the MLX caption image side. The
  first attempt uses the requested `--mlx-max-image-side`; attempt 2 and later
  normally use `--retry-image-side-scale` with `--min-retry-image-side` as a
  floor. If the previous child attempt died from a native signal such as
  `SIGABRT`, the next isolated VLM attempt jumps directly to
  `--min-retry-image-side` instead of stepping down gradually. This keeps
  set-and-forget recovery on the Qwen/VLM path after Metal signal exits instead
  of immediately relying on deterministic fallback, and each attempt records
  its effective profile and retry reason in `results.jsonl` and heartbeat
  artifacts.

The live caption UI should stay compact by default. Full prompt IO remains in
`logs/qwen_caption_io_latest.log`, while prompt/rendered-prompt events are
summarized in the live view instead of being dumped inline with every stage
output.

## Regression Mechanisms To Avoid

- A malformed glossary term became the preferred broad term, producing prompt
  text like `broad term "["`, `COUNTS: [: 8`, and `{"label":"["}`.
- Center-based window wording made overlapping corner crops appear as
  `middle-center`, which misled the merge pass.
- Loose window observation text made repeated crop-edge detections look like
  separate objects. The reconciler must match window boxes to full-frame boxes
  whenever possible and mark unmatched evidence as context only.
- Merge and cleanup prompts previously received partial context and could lose
  glossary or count semantics after the first caption pass.
- Prompt preview assembled some sections separately from runtime, allowing the
  visible prompt and actual prompt to drift.
- Caption trace input events previously printed the same prompt as messages,
  system prompt, user prompt, and prompt text. The readable trace should show
  the chat messages plus rendered prompt without duplicating the same text in
  multiple audit blocks.
- Overlap deduplication guidance appeared twice in the preview merge template.
- Window coverage guards and later cleanup/rewrite stages previously used
  image-backed inference. After a repeated window output, that could send the
  same MLX runtime into another vision generation and crash with a Metal page
  fault. These stages are now text-only editor passes, and repeated MLX caption
  loops are stopped, unloaded, and routed through the explicit recovery policy
  before any follow-up generation.
- Window observation visual loops must also have a text-only recovery path based
  on crop counts, class inventory, representative box layout, and user request.
  One loop-prone crop must not kill the full windowed caption request.
- Automatic coverage/count guard passes are separate from the optional
  Draft + refine flow. UI labels and trace section names must call them guards,
  not generic `refine output`, unless the user explicitly enabled the
  Draft + refine stage.
- Authoritative count repair must treat qualified wording such as `at least 42`
  or `approximately 42` as non-exact. When any exact count is missing, repair
  should prepend one compact count sentence covering all positive authoritative
  counts so the final sentence cap cannot drop separate count sentences.
- Windowed token auto-lifts previously happened in both the frontend and backend,
  making a user-entered cap like `2000` appear as `3000` at runtime. Auto may use
  high defaults, but explicit user caps must remain explicit.
- Loop recovery must not become silent model substitution. Recovery events must
  tell the user when safe retry or fallback changed the effective model or decode
  settings for a stage.

## Verification

Focused caption tests should assert that generated previews contain no malformed
bracket labels, preserve natural counts, expose cleanup and merge templates, use
bounds-aware window descriptions, and show reconciled full-frame/window object
evidence without turning window-only details into extra counts. They should also
assert that merge and cleanup prefer text-only editor runners, that an MLX
caption loop raises a controlled error rather than falling back to blocking
generation, and that loop-recovery controls are preserved by the request schema,
browser recipe layer, and UI contract.

Dense/sparse live coverage is exercised by
`tools/run_qwen_caption_flow_benchmark.py`. The harness runs each selected case
in a child process so uncaught Metal aborts are isolated during testing, captures
prompt previews, final captions, stdout/stderr, and readable/JSONL caption IO,
and fails on structural quality flags such as missing counts, raw count
inventory leakage, prompt leaks, repeated-token loops, and selected unsupported
specific terms.

For unattended durability testing, the same harness can run as a resumable
dataset soak:

```bash
NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python tools/supervise_qwen_caption_soak.py \
  --dataset-root uploads/datasets/data_ingestion_reference_current_label_images_dataset_9526 \
  --output-dir tmp/qwen_caption_benchmark/dataset_soak \
  --all-images \
  --attempts 2 \
  --cooldown-after-crash 5 \
  --cooldown-backoff-multiplier 2 \
  --max-cooldown-after-crash 60 \
  --max-runner-restarts 25 \
  --max-heartbeat-age 900 \
  --heartbeat-startup-grace 120 \
  --max-artifact-log-bytes 1048576 \
  --model-id mlx-community/Qwen3-VL-2B-Instruct-4bit
```

The supervisor command is the preferred CLI set-and-forget path. Before every
launch or relaunch it runs the local preflight described below, then starts the
resumable runner with `--resume`. If preflight finds a live runner lock, the
supervisor waits and retries instead of launching a competing writer or exiting
immediately; this covers launchd/backend restarts while an older runner is still
alive. If existing artifacts already pass the strict completion audit, the
supervisor exits 0 without launching a no-op runner. If the runner exits
nonzero, a strict audit does not pass, or a current runner heartbeat becomes
older than `--max-heartbeat-age`, the supervisor logs the event to
`supervisor.jsonl`, terminates the runner if needed, waits for the configured
restart delay, and starts the runner again until `--max-runner-restarts` is
exhausted. A runner that has not yet written any current `heartbeat.json` gets
the separate `--heartbeat-startup-grace` window before it is classified as
missing-heartbeat, so slow launch or model warmup does not get confused with a
stale active runner. It only exits 0 after existing or newly generated artifacts
pass a strict artifact audit. Use this wrapper from Terminal, launchd, tmux, or
another machine-level supervisor when the backend UI is not the owner of the job.
Runner stdout is captured for audit/debugging, but it does not satisfy heartbeat
health: a chatty runner with a stale or missing `heartbeat.json` is still
terminated and restarted.
Supervised runners are launched in a separate process group; stale or missing
heartbeat termination signals the whole group and escalates to `SIGKILL` after
`--kill-timeout`, so an active per-image MLX worker does not survive as an
orphan while the supervisor starts a replacement runner.
On restart, the supervisor ignores stale heartbeat files whose filesystem
timestamp predates the replacement child process. That prevents a killed runner
from poisoning every later restart with its old `heartbeat.json`, while still
detecting a heartbeat newly written by the current child and already stale.

The supervisor restart path is directly testable without Qwen or GPU access:

```bash
./.venv-macos/bin/python tools/run_qwen_caption_soak_drill.py --pretty
```

The drill uses the real supervisor against a synthetic runner that exits
nonzero, stalls with a stale heartbeat, stalls without any current heartbeat,
then completes. It writes `drill_report.json`, `supervisor.jsonl`, and captured
supervisor stdout under a timestamped `tmp/qwen_caption_benchmark/soak_drill`
directory. A status of `ok` means the set-and-forget restart machinery handled
all scripted interruptions, the final strict audit passed, and the global
`qwen_caption_io` trace cache was retention-checked so a long unattended run
does not rely on manual log cleanup.

The retention check can be run by itself when debugging disk-growth or prompt
trace issues:

```bash
./.venv-macos/bin/python tools/run_qwen_caption_soak_drill.py \
  --caption-io-retention-only \
  --output-dir tmp/qwen_caption_benchmark/soak_drill/caption_io_retention \
  --force \
  --pretty
```

That standalone mode creates synthetic per-run trace files, calls the same
backend reset/prune path used by caption generation, verifies active-run files
survive, verifies old inactive traces are pruned by count and byte caps, and
verifies symlink targets are not followed.

In set-and-forget mode, the watchdog asks for a cooperative restart before it
uses hard launchd remediation. When a nonterminal run stays unhealthy past the
configured threshold, it writes `restart_requested.json` in the run directory.
Current runners advertise `graceful_restart_request`,
`parent_deterministic_recovery`, and `caption_io_event_summary` in
`runner_capabilities` on `heartbeat.json` and `manifest.json`; the watchdog only
uses the cooperative path when the restart capability is present. The runner
checks for the request file only between cases, writes
`restart_acknowledged.json`, exits with a `restart_requested` heartbeat, and lets
the supervisor relaunch with fresh code and `--resume`. If the active runner
does not advertise the capability, or if the request remains unacknowledged past
`--graceful-restart-timeout`, the watchdog may escalate to launchd kickstart or
rebootstrap when those remediation options are configured.
This avoids orphaning an active MLX worker while still letting audit and
recovery improvements take effect before the whole dataset finishes.

Before trusting a checkout for a large dataset, also run the no-GPU endurance
variant:

```bash
./.venv-macos/bin/python tools/run_qwen_caption_soak_drill.py \
  --endurance-cases 10000 \
  --endurance-chunk-size 1000 \
  --output-dir tmp/qwen_caption_benchmark/soak_drill/endurance_10000 \
  --force \
  --pretty
```

The endurance drill still uses the real supervisor and strict audit, but the
synthetic runner appends many case rows over repeated child invocations. It
first leaves a resumable partial run behind after a nonzero exit, then kills a
child process with `SIGABRT` to mimic a Metal abort trap, then exercises
stale-heartbeat and missing-heartbeat restarts, then finishes the remaining
cases in chunks. The report status is `ok` only if every synthetic case has a
latest `ok` row, the final audit is clean, the signal exit was observed,
multiple restarts were observed, prompt-budget telemetry is present, and the
compact `summary.json` snapshot stays bounded while `results.jsonl` carries the
full ledger. This does not replace a real GPU pilot, but it proves the
set-and-forget wrapper can
preserve progress and audit coverage across hard child crashes and restart churn
at dataset scale. By default it also runs the same `qwen_caption_io` retention
drill; pass `--skip-caption-io-retention-drill` only for a narrow supervisor
debugging run where log-retention coverage is being tested elsewhere.

For a manual launch sequence, run the same preflight and runner directly:

```bash
NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python tools/preflight_qwen_caption_soak.py \
  --dataset-root uploads/datasets/data_ingestion_reference_current_label_images_dataset_9526 \
  --output-dir tmp/qwen_caption_benchmark/dataset_soak \
  --all-images \
  --resume \
  --attempts 2 \
  --max-artifact-log-bytes 1048576 \
  --model-id mlx-community/Qwen3-VL-2B-Instruct-4bit \
  --refinement-model-id same \
  --fallback-model-id auto

NO_ALBUMENTATIONS_UPDATE=1 ./.venv-macos/bin/python tools/run_qwen_caption_flow_benchmark.py \
  --all-images \
  --attempts 2 \
  --resume \
  --output-dir tmp/qwen_caption_benchmark/dataset_soak \
  --model-id mlx-community/Qwen3-VL-2B-Instruct-4bit \
  --timeout 900
```

The preflight command is a cheap local gate; it does not load Qwen or run image
generation. It uses the same case-selection rules as the soak runner, checks
that an existing resume manifest matches the requested case list, and compares
the manifest's run-settings fingerprint against the requested caption setup
before allowing resume. The fingerprint covers caption-affecting settings such
as model choices, prompt text, output-token cap, sampling settings, windowing,
box limits, preview mode, and the sanitized `--request-json` template contents,
not the transient backend metadata paths used to pass generated `cases.json` or
`request_fields.json` files to the runner.
This prevents a set-and-forget run from appending captions generated under a
different prompt, model, or decode policy into the same artifact directory.
Legacy manifests that lack a fingerprint are warnings; a mismatched fingerprint
is an error. Preflight also rejects live runner locks in the output directory,
audits existing run artifacts when they exist, and estimates disk needs from the
remaining case count, attempt count, and per-attempt log cap. Recoverable
interrupted-run artifacts, such as incomplete coverage or a stale running
heartbeat from a dead runner, are warnings when `--resume` is active so the
supervisor can continue. Unsafe evidence-corruption errors such as malformed
`results.jsonl`, malformed `captions.jsonl`, invalid manifests, case-set
mismatches, run-settings mismatches, and live runner locks remain hard errors.
It creates and
removes small probe files in the artifact directory before launch, and when
`--save-dataset-text-labels` is requested it probes the target dataset
`text_labels` directories as well, so permission or mount problems fail before
GPU time is spent. If `--request-json` is supplied, preflight loads that same
caption request template, strips image-specific fields just like the runner, and
applies model-related overrides before checking cache state. Missing or invalid
request-template files are hard preflight errors so an unattended run does not
silently fall back to different CLI defaults. It also resolves the
effective caption, refinement, and safe-retry fallback models and verifies that
each concrete model has local weight files in the Hugging Face cache or at a
local path. Missing or partial model caches are `error` by default so a
set-and-forget run does not silently spend its first hours downloading or fail
only after the runner starts. Add `--allow-model-download` only when that
download is intentional for a manual diagnostic; the same state then becomes a
`warn`. Strict `--tenk-set-and-forget` rejects `--allow-model-download` before
pilot generation, certification, drills, or launchd handoff, because a two-week
handoff must start with all selected models already local. A status of `error`
should block a set-and-forget run. A status of `warn` is allowed only when the
operator deliberately accepts the condition, such as intentional downloads or
`--max-artifact-log-bytes 0` for full raw logs. Add
`--fail-on-warn` when using it from launchd or another supervisor that should
require a completely clean start.
Backend dataset caption jobs run the same preflight using generated
`cases.json` and `request_fields.json` files stored in the backend job metadata
directory, not in a separately requested runner artifact directory. This keeps
backend preparation from touching a shared artifact directory before preflight
and the runner lock establish ownership. Hard preflight failures leave the job
in `failed` with `caption_runner_preflight_failed` and preserve the preflight
report in the job result, so UI-launched set-and-forget jobs also fail before
spending GPU time or mutating captions. When preflight sees that a live runner
owns the artifact directory, it reports the lock error and skips the artifact
write probe instead of creating even a temporary probe file there. When the
UI/backend request asks to save text labels, that save intent is passed into
both preflight and the child runner: preflight checks the target `text_labels`
directories, and the
runner writes the durable dataset text-label file as soon as an image succeeds.
The backend still records the caption in its job result and saves through the
UI-facing caption-record path while processing runner rows, so a backend restart
can resume from already-written runner artifacts without losing successful
captions. Saved generated captions append as alternates, and the chosen primary
caption mirrors to the legacy text-label file for compatibility with older
exports and skip-existing checks. When `save_text_labels` is enabled, strict and
live artifact audits also require each generated or resumed-completed success
row to reference existing non-empty saved-caption evidence. This keeps
set-and-forget jobs from reporting a clean caption run when the dataset mutation
evidence is missing.
The CLI set-and-forget path follows the same rule: `--save-dataset-text-labels`
implies saved-label coverage in supervisor terminal audits, and generated
watchdog, live-status, and final-audit commands include
`--require-saved-text-labels`.
Backend callers can set `allow_model_download=true` for deliberate first-run
downloads, but the default keeps red/missing model selections from starting
unattended caption jobs.
At terminal state, backend dataset jobs also run the same strict artifact audit
as the CLI supervisor. The audit report is stored in the job result under
`strict_audit`. Normal manual-review jobs may still complete with reviewable
failed cases, but set-and-forget jobs require the strict audit status to be
`ok`; stale, incomplete, corrupt, or warning-level terminal artifacts fail the
job with `caption_runner_strict_audit_failed`. Incomplete, latest-failed,
pending-retry, or quality-failed artifacts remain eligible for the normal
auto-resume path. Completed artifacts that fail only non-recoverable health
gates, such as historical signal-exit or loop-recovery rates, stop with a
terminal strict-audit failure instead of burning restarts that cannot change the
existing ledger.
The caption panel exposes this as **Allow model downloads** next to the
set-and-forget controls. Backend jobs materialize an effective
`request_fields.json` for the runner: implicit model choices such as empty,
`active`, `auto`, or `default` are resolved to the concrete caption model and
variant before preflight and before launching the subprocess. The same concrete
model fields are passed on the runner command line, so the model cache that
preflight checks is the model the worker uses. The persisted job request is also
updated to that effective caption request before the runner starts, so manual
and automatic resumes keep using the same concrete model even if environment
defaults change later.

In soak mode, every image attempt runs in a fresh child process, so a Metal
process abort is recorded as one failed attempt instead of destroying the parent
run. The supervisor writes `runner_exit` events with `status=signal_exit`,
`return_signal`, and `return_signal_name` when the child dies from a signal such
as `SIGABRT`. `results.jsonl` is append-only and remains the authoritative full
ledger, `summary.json` is rewritten atomically as a compact status snapshot, and
`captions.jsonl` receives successful captions. `summary.json` keeps complete
totals plus a bounded recent row sample by default (`--summary-row-limit`, use
`-1` only for manual diagnostics that need every row copied into the summary), so
a 10k-image run does not rewrite an ever-growing full-row JSON document after
each case. Re-running with `--resume` skips completed cases only when the
selected case set and run-settings fingerprint are compatible with the existing
manifest. Resume skips reuse the durable prior `ok` rows by default instead of
appending synthetic `skipped_completed` rows, so repeated unattended restarts do
not make `results.jsonl` grow by the number of already-finished images. Pass
`--record-resume-skips` only for a manual diagnostic run that needs explicit
legacy skip breadcrumbs in the ledger.
Each attempt row records `attempt_failure_kind`, `return_signal`, and
`return_signal_name` when applicable, so a real Metal abort can be tied back to
the image, attempt directory, stdout/stderr tail, model settings, and prompt
preview. `--max-failures`, `--attempts`, `--cooldown-after-crash`,
`--cooldown-backoff-multiplier`, `--max-cooldown-after-crash`, and
`--continue-on-quality-failures` control whether a long run keeps going after
bad outputs or child crashes. Failed attempts emit an `attempt_cooldown`
heartbeat before sleeping, and hard failures such as signal exits, timeouts, and
nonzero exits use capped exponential backoff so repeated GPU crashes do not
tight-loop the machine. `--save-dataset-text-labels`
is intentionally opt-in so durability testing does not mutate dataset captions
unless the operator explicitly asks for that.

The parent runner owns an artifact-level `.runner.lock` for the full duration
of a run. This prevents a restarted backend, a second UI request, or a manual
CLI resume from appending to the same `results.jsonl` while an older runner is
still alive. Dead-owner locks and malformed or unreadable lock files are
recoverable: preflight reports them as warnings, and the runner removes them
before resume. A live-owner lock is never overtaken automatically, even if its
heartbeat is stale; that state is reported by preflight and audit because taking
over would risk two writers if the old process wakes back up. The CLI supervisor
waits on that live-lock preflight state and retries until the owner clears or
the optional live-lock wait timeout is reached. The backend set-and-forget
sweeper adopts eligible live-lock owners into monitor jobs rather than waiting
silently, so the UI can still show progress after a backend restart. A live
lock is waited on and emits periodic non-JSON wait lines so the backend
supervisor remains aware that the runner is not silent. The lock heartbeat is
refreshed alongside the parent heartbeat during long child attempts. The lock is
released on normal completion, and the soak audit warns if a terminal run leaves
it behind.

Raw per-attempt diagnostic blobs are bounded by default for unattended runs.
The runner keeps the structured artifacts that drive resume and audit
(`manifest.json`, `results.jsonl`, `captions.jsonl`, `summary.json`,
`heartbeat.json`, and each `result.json`) intact, but caps child stdout/stderr
and copied `qwen_caption_io` trace files to `--max-artifact-log-bytes`
(default 1 MiB per file). Truncated files include an explicit marker and keep
the tail, which is usually where a crash or repeated-output failure appears.
The worker mirrors the active Qwen `run_id` into `worker_progress.jsonl`, then
copies the matching per-run trace files from `logs/qwen_caption_io/` into the
attempt directory. This avoids the older failure mode where a late guard pass
overwrote `qwen_caption_io_latest.jsonl` and hid the earlier windowed or
full-image events. If the worker crashes before a run id is mirrored, or if a
mirrored run id has no matching per-run trace files, the attempt receives an
explicit missing-trace summary; it does not copy `qwen_caption_io_latest.*` into
the attempt. The worker computes `qwen_caption_io_summary.json` from the
complete per-run JSONL before applying the artifact size cap; the parent stores
that summary under `qwen_caption_io` in `results.jsonl`. Raw trace files remain
bounded for forensics, but prompt-budget, loop-guard, and recovery counts do not
depend on which part of a capped JSONL file survived. The repo-level
`logs/qwen_caption_io/<run-id>.*` cache is also bounded independently so a
multi-day run does not fill disk with transient global traces after the attempt
artifacts have been copied. `QWEN_CAPTION_IO_RUN_LOG_MAX_FILES` defaults to
400 direct run-log files and `QWEN_CAPTION_IO_RUN_LOG_MAX_BYTES` defaults to
512 MiB; the active run's files are always protected, and `0` disables the
corresponding retention cap. Soak audit and certification use the larger of
preview-estimated and runtime-measured prompt tokens, because the runtime
tokenizer measurement is the launch gate whenever it is available.
Set the CLI flag or backend request field to `0` only for active debugging when
full raw prompt traces are worth the disk cost. The caption UI exposes this as
**Attempt log cap (MB)** for dataset-backed backend jobs.
Strict audit treats malformed append-only JSONL rows as evidence corruption.
Invalid `results.jsonl` or present `captions.jsonl` lines are reported with
their line number and a short preview, and the run fails audit instead of
silently dropping those rows from coverage, degraded-rate, resume, or caption
artifact decisions.
Preflight and the runner apply the same fail-closed rule before direct
`--resume`: if `results.jsonl` or `captions.jsonl` contains malformed or
non-object rows, preflight returns a blocking `resume_rows` or `caption_rows`
error and the parent process exits with `resume_results_jsonl_invalid` or
`resume_captions_jsonl_invalid` before rewriting the manifest, invoking a child
worker, or appending any new rows. A resume that finds every selected case
already complete still refreshes `summary.json` and `heartbeat.json`, but it
does not append duplicate rows unless `--record-resume-skips` is set.

The production UI path for large caption runs is `/qwen/caption/jobs`. The
backend job builds an explicit case list from the current annotation manifest,
including overlay label edits and existing text-label state, then launches the
soak runner as a subprocess. This keeps MLX/Metal failures outside the FastAPI
process while preserving backend-owned progress, cancellation, result files, and
optional overlay text-label writes. The browser uses this backend job for
dataset-backed `Caption image`, `Caption next N`, and `Caption all images`
requests. Browser captioning is fail-closed for set-and-forget work: if no
caption dataset is selected or registered, the UI refuses to start a multi-image
batch, and the default set-and-forget single-image path refuses to fall back to
the direct in-process `/qwen/caption` endpoint. That keeps batch and
set-and-forget runs behind subprocess isolation, where an MLX/Metal abort can
kill a worker or runner process without taking down the FastAPI backend. Users
can still disable set-and-forget for an ad-hoc direct single-image diagnostic.
Dataset-backed single-image, next-N, and all-image paths let the backend job
write overlay text labels when the save checkbox is enabled, send
`set_and_forget=true` by default, and explicitly run with `max_failures=0`, so a
single bad image is recorded and resumable instead of
stopping or failing an unattended dataset job after the remaining images
finish. Exhausted failed images are appended to `results.jsonl` as terminal
`failed` rows, not only summarized in `summary.json`, so live and final audits
can distinguish real terminal failures from retryable failed attempts. Completion
with failed images is reported as a completed job with nonzero failure counts in
`summary.json`, `results.jsonl`, and the UI status. In strict set-and-forget
audits, a terminal failed case violates the zero-failure cap immediately; only a
`failed_attempt` row that still has a scheduled next attempt is treated as a live
pending retry.
If no dataset id is available, only a user who has deliberately disabled
set-and-forget can run a direct current-image request for local interactive
diagnostics; multi-image captioning remains backend-job only.

Backend jobs supervise the runner subprocess separately from the per-image
worker timeout. `per_image_timeout_seconds` bounds each image attempt inside
the runner, while `runner_no_output_timeout_seconds` bounds how long the
FastAPI job supervisor will wait without heartbeat or result-row progress from
the runner. The parent runner also writes `heartbeat.json` in the artifact
directory before and after cases and refreshes it during child attempts
according to `runner_heartbeat_interval_seconds` (default 30 seconds). The
backend persists the latest heartbeat into the job result and uses heartbeat
phase/attempt changes and parsed JSON result rows as progress signals, but raw
runner stdout is treated as audit logging only. Repeated heartbeat refreshes for
the same attempt and chatty non-JSON stdout do not indefinitely reset the quiet
watchdog. When the child worker writes changed `worker_progress.json`, the
parent increments heartbeat sequence and surfaces the Qwen stage in the backend
job message; if that worker progress also stops changing, the quiet watchdog can
still fire. If that no-progress watchdog fires, the job terminates the runner,
persists `caption_runner_no_output_timeout`, and leaves the result files already
written so the operator can resume into the same output directory rather than
trusting a forever-running status.
`POST /qwen/caption/jobs/{job_id}/resume` starts a fresh backend job record from
the persisted request, points the runner at the old artifact directory with
`resume=true`, and keeps the new job metadata in its own job directory. The
runner then skips completed cases from `results.jsonl` and continues the
remaining images without overwriting the old job record. `GET /qwen/caption/jobs`
includes persisted job records after a backend restart, marking formerly
`queued` or `running` jobs as `interrupted` when no live in-memory job owns them.
The caption UI's **Attach / recover now** button uses that list to attach to a
still-running job or resume an interrupted, failed, or completed-with-failed-
images job for the selected dataset. Cancelled jobs remain terminal and are not
resumed by automatic or manual recovery. Live running jobs reject resume so two
runners do not write into the same artifact directory concurrently. The page
also performs an immediate quiet set-and-forget auto-attach check when a caption
dataset is available and repeats that check while the panel is open, so the
button is an immediate operator action rather than the normal unattended
recovery path.

Set-and-forget caption jobs are the unattended path, not just a manual recovery
shortcut. Dataset-backed **Caption image**, next-N, and Caption all jobs opt in
with `set_and_forget=true`; the backend then auto-resumes persisted `queued`,
`running`, `interrupted`, `failed`, or completed-with-failed-images jobs after a
backend restart. After the startup pass, a periodic reconciliation sweeper
repeats the same bounded scan for orphaned persisted set-and-forget records.
This covers cases where a job record remains `running` or `failed` on disk
without a live in-memory backend owner. The sweeper is conservative: it will not
overtake any artifact directory currently owned by an active live backend job
(`queued`, `running`, or `cancelling`). Terminal in-memory records such as
`failed` jobs do not block the periodic sweep, so a missed immediate auto-resume
can still be repaired without manual intervention.
Dataset-backed single-image caption jobs also save caption records from the
backend when **Save generated captions** is enabled, so a browser disconnect or
backend restart does not leave the completed caption only in transient UI
state. Caption export supports flat audit JSONL, with one caption record per
saved caption; grouped JSON, with one image object containing all primary-first
alternate captions for review and archive workflows; and VLM JSONL, with one
normal `image_path` / `question` / `answer` row per caption for caption-only
training. VLM JSONL answers are explicit JSON caption strings, and alternate
caption rows receive stable question variants so downstream validators do not
see duplicate image/question pairs. The separate instruction-dataset path keeps
caption0, VLM-generated visual question/answer rows, optional deterministic
metadata QA, source annotations, provenance, rejected rows, and flattened trainer
rows in a full versioned `tator_caption_instruction_archive_v1` audit object.
It also exposes `instruction_archive_rows`, one construction archive record per
image for JSONL download; `instruction_review_rows`, one candidate-level review
row per caption0, generated-QA, and deterministic metadata item; and
`instruction_report`, a run-level count, provenance, split, and rejection
summary. Generated question/answer rows are language annotations only; they are
never written back as source annotations.
Source annotations are built from real label evidence into `object_counts`,
`visible_classes`, `bbox_instances`, `bbox_geometry`, `spatial_facts`,
`uncertainty`, and field provenance. Deterministic metadata QA is off by default
and appears only when explicitly enabled; when enabled, its answers are typed
JSON rows computed from source annotations, including class-list, object-count,
presence, absence, and simple bbox-derived spatial rows when supported.
The backend `/captions/export` response carries the same logical grouped
archive as a versioned `tator_caption_grouped_v1` object, in addition to the
flat records, compatibility grouped map, caption-only training rows, instruction
training rows, per-image instruction archive rows, instruction review rows, the
instruction report, and the full instruction audit object. This keeps browser
downloads and scriptable exports aligned around stable multi-caption and
instruction-dataset contracts.
The Qwen trainer accepts the flat `image_path` / `question` / `answer` training
row shape directly and normalizes each row into the conversation format used for
fine-tuning, so the download remains easy to inspect without requiring a manual
conversion step.
The instruction report includes corpus-quality metrics for generated-QA
diversity, duplicate-question rates, generated-QA acceptance/rejection,
structured rewrites, source-grounded row coverage, answer-format distribution,
and source-class coverage. It also includes `training_readiness`, a
`ready`/`needs_review`/`blocked` status computed from selected training rows,
manual-review decisions, and quality gate thresholds. Browser instruction JSONL
export blocks `blocked` readiness and warns on `needs_review` instead of
presenting a structurally valid but unreviewed corpus as training-ready.
Generated captions append as alternate records by default. **Make generated
caption primary** is a separate opt-in promotion control; when it is off, a
generated caption can still become primary only if the image has no existing
primary caption. This keeps repeated caption runs useful for alternate-caption
training without silently replacing the selected text-label mirror.
The caption panel exposes archive and export status before launch: the archive
status states the current image caption count and that no per-image caption cap
is enforced, while the VLM export health line reports client-side validation
before a training JSONL file is written. The caption-only VLM export validator
blocks rows with missing image paths, blank questions, non-JSON answers, answers
without exactly one `caption` key, or duplicate `image_path`/`question` pairs.
  The instruction JSONL validator blocks missing `image_path`, blank question,
  blank answer, missing instruction row metadata, missing or unsupported
  instruction archive provenance, missing or unknown validation/review state,
  invalid JSON for JSON row types, generated QA rejected by archive validation,
  non-trainable review state, and duplicate `image_path`/`question` rows before
  download. The instruction panel also exposes generated QA mix and generated
  answer format so job launch and browser exports use the same row policy.
For detailed/windowed jobs, set-and-forget also changes the Auto value of
**Windowed full-image compose** to `text_only`. The crop observations still come
from visual Qwen calls; only the later full-image composition avoids resending a
large full-image tensor after the window passes. Operators can explicitly choose
`visual` for manual diagnostics or to compare behavior, but that choice is part
of the run fingerprint and a certified pilot must use the same strategy as the
large run.
`QWEN_CAPTION_SET_AND_FORGET_SWEEP_INTERVAL_SECONDS`
controls the periodic interval (default 300 seconds); set it to `0` to keep only
startup/manual/live-failure recovery.
`QWEN_CAPTION_SET_AND_FORGET_ADOPTION_POLL_SECONDS` controls how often an
adopted live-runner monitor refreshes progress from artifact state (default 5
seconds). Cancelled jobs remain cancelled. Completed set-and-forget jobs are
audited against the same degraded-output policy as the CLI audit; if the
terminal result exceeds a configured health gate, the backend marks the job
`failed` with `caption_runner_degraded_rates`, records the exact violated
rates, and uses the same bounded auto-resume path. When the violated gate
involves recovery events, loop recovery, or deterministic recovery, the resumed
runner is told to reprocess those recovered rows instead of skipping them as
already complete.
Auto-resume groups persisted records by artifact directory, treats the newest
persisted owner as authoritative, and resumes only when that newest owner is
eligible. Before starting a resume, the backend runs a strict artifact audit on
the selected artifact directory; if that directory already passes, the persisted
job record is marked completed from the existing artifacts instead of launching
a no-op runner. A clean newer completion, cancellation, exhausted retry budget,
or pilot-certification failure suppresses older failed/interrupted records for
the same artifact directory, so an original job record and later resume record
do not both restart against the same `results.jsonl`. When startup or periodic
auto-resume finds a live `.runner.lock` owned by a runner outside the current
backend process, it does not start a competing writer. Instead, for eligible
set-and-forget records, the backend adopts the live runner into an in-memory
monitor job with the same job ID, updates progress from `heartbeat.json` and
the live artifact audit, and reconciles the final artifact when the runner
finishes or exits. If the adopted runner disappears before a strict terminal
artifact exists, the normal bounded auto-resume path takes over after the lock
is gone. Each automatic resume increments `auto_resume_count`; adoption itself
does not consume that retry budget because it observes the existing writer
instead of launching another one. The request may set `max_auto_resumes`,
otherwise `QWEN_CAPTION_SET_AND_FORGET_MAX_AUTO_RESUMES` defaults to 25. Set
`QWEN_CAPTION_SET_AND_FORGET_AUTO_RESUME=0` to disable startup auto-resume while
keeping manual recovery available. While the backend remains alive, the same
bounded auto-resume path also restarts opted-in jobs after runner-process
failures such as `caption_runner_no_output_timeout` or nonzero runner exits.
The failed supervisor records `auto_resumed_job_id`, and the caption UI monitor
follows that replacement job automatically instead of reporting the old
supervisor as the final failure. The manual attach/resume button remains useful
when an operator wants to inspect or attach to a job, but a normal set-and-forget
run should not require pressing it. Backend preparation/configuration failures
are not retried automatically, and strict-audit failures caused by malformed
`results.jsonl` or `captions.jsonl` are held for operator repair instead of
being auto-resumed into the same corrupt append-only artifacts. The caption
panel exposes the health gates as set-and-forget controls; `-1` disables an
individual rate gate, and terminal runs are always checked even when the
live-rate sample size is below `min_rate_cases`. Set-and-forget mode allows a
small recovered-loop rate by default (`max_loop_recovery_case_rate=0.05`),
because a safely recovered repeated-output loop is degraded telemetry rather
than a terminal failure. On live, nonterminal runs, that low-percentage loop
gate also waits for enough evidence to observe at least three loop-recovery
events at the configured rate before it can make the run unhealthy; for the
default 5% gate, the floor is 60 processed cases even if `min_rate_cases` is
lower. The same default 5% set-and-forget rate now applies to
`max_loop_guard_case_rate`, which is the union of stream-loop detection and
loop-trim cases whether or not a full retry was needed. Broader
recovery-event spikes still use `max_recovery_event_case_rate` and can fail
earlier, while completed runs always enforce the configured loop-recovery and
loop-guard thresholds. Deterministic count/layout recovery is a separate,
tighter degraded-output gate: set-and-forget defaults
`max_deterministic_recovery_case_rate` to `0.01`, waits for enough live cases to
observe at least two deterministic-recovery events at that rate, and enforces
the threshold immediately on terminal audits. Recovered signal exits are also
bounded instead of zero-tolerance in set-and-forget mode:
`max_signal_exit_attempt_row_rate` defaults to `0.05` so rare isolated MLX/Metal
child aborts can be retried by the VLM path without dooming a long run after all
latest cases succeeded. This is intentionally looser than deterministic
count/layout recovery, because a recovered signal exit still produced a model
caption on a later child attempt while deterministic recovery is degraded text
written only after model attempts were exhausted. Backend set-and-forget jobs
pass the same signal-exit threshold into pilot certification, live status
summaries, and final strict audits. Enter `0` for any recovery or signal-exit
gate when validating a configuration that must have zero recovered captions or
process aborts, and use `-1` only when intentionally disabling that gate.
For 10k-scale UI/backend launches, **Require certified pilot** is part of the
set-and-forget contract, not a manual recovery option. A non-preview
set-and-forget backend job with at least
`QWEN_CAPTION_SET_AND_FORGET_REQUIRE_PILOT_CASES` cases, default `10000`, fails
closed with `caption_runner_pilot_required` before preflight or runner launch
unless pilot certification is enabled for that job. Point **Certified pilot
artifact dir** at a completed pilot soak under the workspace or caption job
root. The UI/backend default **Pilot min cases** is
`QWEN_CAPTION_DEFAULT_PILOT_MIN_CASES` and defaults to 300 timed pilot cases for
the confidence-backed set-and-forget floor. A 100-case clean pilot is useful as
a smoke test, but it does not bound a 1% deterministic count/layout fallback
budget tightly enough for a two-week unattended run. Users can still lower the
pilot size or set
**Pilot deterministic recovery confidence** to `0` for controlled diagnostics;
strict 10k set-and-forget treats those diagnostic settings as launch blockers.
For large set-and-forget launches, the backend treats diagnostic settings as
launch blockers: pilot prompt-budget telemetry must stay required, the pilot
prompt-size ceiling must be positive, the prompt-budget adaptation gate must be
enabled, the p95 projected-duration gate must be positive, deterministic
recovery confidence certification must remain enabled, and **Pilot min cases**
must remain at least the confidence-backed floor. The backend runs the same
certification gate before launching the runner subprocess, writes
`required_pilot_certification.json` in the target artifact directory, and
requires the pilot manifest's `run_settings.fingerprint` to match the exact
caption settings about to be launched. Certification also requires current
runner capability markers in the pilot `manifest.json` or `heartbeat.json`;
old pilots that do not prove cooperative restart, deterministic parent recovery,
copied `qwen_caption_io` stream-loop event summaries, and worker-progress
heartbeats fail before the large run starts. Backend set-and-forget jobs also
reject pilot reports that have prompt-budget rows but lack run-bound
`qwen_caption_io_per_run` source evidence, so an old or hand-edited `ok`
certification report cannot start the runner. The job fails with
`caption_runner_pilot_certification_failed` if the pilot is stale, incomplete,
too small, too slow on average, too slow in p95 case-time projection, degraded,
or was produced by different model, prompt, windowing, token-budget, or decode
settings, or came from an older runner that lacks the current unattended
capability markers. This failure is deliberately not startup-auto-resumed; the
operator must fix the pilot or choose a new pilot directory before trying the
large set-and-forget run again.

For CLI launches, `tools/run_qwen_caption_unattended.py --tenk-set-and-forget`
is the set-and-forget launch gate, not a manual recovery checklist. In that
mode, readiness fails closed unless the run has clean wrapper preflight, the
supervisor and watchdog drills, pilot certification or clean live-run adoption
evidence, current runner capability markers, prompt-budget telemetry, a positive
projected-duration gate, a positive live disk-reserve gate, supervisor and
watchdog LaunchAgent plists, launchd installation for both roles, watchdog
remediation, and launchd sleep prevention. Missing pieces are recorded as
readiness errors in `readiness.json` with `ready_for_10k_set_and_forget=false`.
Outside `--tenk-set-and-forget`, the same runbook can still be used for manual
diagnostics and recovery planning, where omitted launchd or pilot pieces remain
warnings unless `--require-readiness-ok` is used.

The macOS backend launcher (`tools/run_macos_backend.sh`) is part of that
contract. It restarts the backend after unexpected nonzero exits such as Metal
abort traps by default, while still exiting on normal shutdown, Ctrl-C, or
termination signals. Set `TATOR_BACKEND_RESTART_ON_CRASH=0` to disable crash
restart, `TATOR_BACKEND_RESTART_MAX` to bound restart attempts (`0` means
unlimited), and `TATOR_BACKEND_RESTART_DELAY` /
`TATOR_BACKEND_RESTART_MAX_DELAY` to tune the backoff. For the 10k handoff, keep
`TATOR_BACKEND_RESTART_MAX=0` for unlimited crash restarts or set it to at least
`QWEN_CAPTION_SET_AND_FORGET_MIN_BACKEND_RESTARTS` (default `25`), and keep the
maximum backoff at or below
`QWEN_CAPTION_SET_AND_FORGET_MAX_BACKEND_RESTART_DELAY_SECONDS` (default `300`).
The launcher stamps the backend process with `TATOR_BACKEND_LAUNCHER*`
environment metadata, and `/qwen/status` plus `/qwen/progress` expose
`supervision.restart_capable`, `supervision.restart_policy`, and
`supervision.set_and_forget_ready`. The caption UI uses those fields to warn when
set-and-forget state will be persisted but the current backend process is not
advertising crash-restart supervision or its restart policy is not large-run
ready. For large non-preview backend set-and-forget jobs at or above
`QWEN_CAPTION_SET_AND_FORGET_REQUIRE_PILOT_CASES`, crash-restart supervision and
its restart policy are backend launch gates by default: the job fails closed with
`caption_runner_backend_supervision_required` before preflight, pilot
certification, or runner launch if the backend is not advertising restart
capability or the advertised restart budget/backoff is too weak. Set
`QWEN_CAPTION_SET_AND_FORGET_REQUIRE_BACKEND_SUPERVISION=0` only for controlled
diagnostics. If the backend is started without this launcher or another process
supervisor, smaller persisted set-and-forget jobs can still be resumed later,
but they cannot auto-resume while the backend process is down.

For unattended runs, the artifact directory is also directly auditable with
`tools/audit_qwen_caption_soak.py <output_dir>`. The audit reads
`manifest.json`, append-only `results.jsonl`, atomic `summary.json`, and
`heartbeat.json`, plus `captions.jsonl` and the `.runner.lock` when present; it exits 0 only when
the latest result rows cover the expected case list, the summary matches those
rows, every generated successful case has a non-empty caption row, no latest
cases failed, no quality warnings remain, degraded-output rates stay inside
their configured thresholds, any running heartbeat is still fresh, any active
attempt is still within its own timeout plus grace, and the runner-lock state
matches the run state. Resume skips, existing-caption skips, and preview-only
rows do not require a new caption row because they do not generate a new caption
artifact. Stale running heartbeats, live locks whose owner process is gone,
mismatched summaries, incomplete completed runs, generated `ok` rows without
caption records, failed latest rows, excessive failed attempt rows, signal-exit
attempt rows, excessive caption recovery, loop recovery, active attempts past
timeout plus `--max-attempt-overrun`, or leftover terminal locks produce a
nonzero status while still reporting whether the run is resumable from the
existing artifacts. This is the intended health probe for set-and-forget
automation; manual recovery remains available for operator-directed
intervention.
Use `--allow-running-incomplete` for live monitoring during a long run: a fresh
running heartbeat with a live runner lock is then considered healthy even before
all cases have result rows, while failed rows, stale heartbeat, overdue active
attempts, missing live lock, or terminal-run leftovers still produce nonzero
status. Omit that flag for the final completion audit.
The degraded-output gate reports failed latest cases, quality-warning cases,
failed attempt rows, signal-exit attempt rows and signal names, caption recovery
events, loop-recovery cases, loop-guard cases, deterministic-recovery cases,
prompt-budget adaptation, and the largest observed prompt size. Terminal runs
are always checked against those rates; running jobs only activate the rate gate after
`--min-rate-cases` so the first few images do not cause premature churn. Defaults
are strict for failed cases, quality failures, loop recoveries, and loop guards,
allow up to 25% of cases to need generic recovery events, and allow up to 25%
failed attempt rows before the audit errors. Tune `--max-failed-case-rate`,
`--max-quality-failure-rate`, `--max-recovery-event-case-rate`,
`--max-loop-recovery-case-rate`, `--max-loop-guard-case-rate`,
`--max-deterministic-recovery-case-rate`, `--max-failed-attempt-row-rate`,
`--max-signal-exit-attempt-row-rate`, and `--max-attempt-overrun` when a benchmark
intentionally explores a noisy or slow configuration; use `-1` to disable an
individual gate. The default
`--max-attempt-overrun 60` catches the specific case where the heartbeat remains
fresh while a child caption attempt has already exceeded its configured timeout.
The same audit records per-rate headroom. For case-based recovery gates on a
live incomplete run, headroom includes the full-dataset denominator, the count
budget available at that terminal size, and the terminal rate that would result
if no more recovery events happened. That makes near-threshold rates auditable
without requiring an operator to manually convert current percentages into 10k
case budgets.
Prompt-budget telemetry and copied stream-loop event summaries are also recorded
per case. For current-runner pilots, certification requires those telemetry
contracts by default so a large set-and-forget run does not launch from stale
artifacts that cannot prove prompt size or repeated-token-loop observability. Use
`--max-prompt-tokens` to turn the largest observed prompt estimate into a launch
gate, and `--max-prompt-budget-adapted-case-rate` to fail pilots where too many
cases needed prompt-budget adaptation. A value of `0` disables the prompt-size
gate only for manual diagnostics; explicit 10k set-and-forget launch readiness
requires a positive prompt-size ceiling. A rate of `-1` disables the
adaptation-rate gate. The compact live audit also prints max prompt tokens,
prompt-budget telemetry coverage, and adaptation rate so operators can see
prompt growth without opening the JSON report.
For long running jobs, the audit can also enforce a wall-clock throughput budget
with `--max-projected-duration-hours`. This is separate from
`--max-no-progress`: no-progress catches a stalled job, while projected duration
catches a job that is still alive and completing cases but is no longer on track
to finish within the unattended window. The projection uses the earliest
supervisor event time when available, falling back to runner start telemetry, and
waits for `--min-rate-cases` before activating on live runs. The generic manual
audit default is `0` (disabled); `--tenk-set-and-forget` generated runbooks set
it to 336 hours unless an explicit supervisor argument overrides it.
Long unattended runs also carry a live disk-reserve gate. Preflight estimates
the planned artifact budget before launch, while `--min-free-gb` is checked by
the watchdog, live-status audit, supervisor strict completion audit, and final
audit while the run is in progress. The supervisor default is `5.0` GiB; set it
explicitly when a dataset needs a larger safety margin, and use `0` only for a
manual diagnostic run where disk monitoring is handled elsewhere.

For a long unattended run, start an independent watchdog in another terminal or
launchd job:

```bash
./.venv-macos/bin/python tools/watch_qwen_caption_soak.py \
  tmp/qwen_caption_benchmark/dataset_soak \
  --interval 60 \
  --max-heartbeat-age 900 \
  --max-no-progress 3600 \
  --max-consecutive-unhealthy 3
```

When the supervised run is owned by launchd, the watchdog can also perform
bounded remediation instead of only reporting failure. Add
`--remediate-launchd-label`, `--remediate-launchd-domain`,
`--remediate-launchd-plist`, `--max-remediations`, and
`--remediation-cooldown` to let it repair the supervisor LaunchAgent after the
configured number of unhealthy checks. Strict unattended runbooks also record a
positive `--graceful-restart-timeout`, so a capable runner gets a between-case
`restart_requested.json` handoff before launchd escalation. It first tries
`launchctl kickstart -k`.
If that fails or times out and a supervisor plist was supplied, it then runs a
bounded `launchctl bootout` plus `launchctl bootstrap` fallback from the recorded
plist. The remediation result, commands, return codes, and output tails are
written into the watchdog JSON event; a failed repair still exits unhealthy.
The watchdog persists no-progress and remediation counters in
`watchdog_state.json`, but it also mirrors that state into the append-only
`watchdog.jsonl` events. If `watchdog_state.json` is missing or corrupt after a
crash or interrupted write, the next watchdog process recovers the latest state
from the event log before deciding whether another launchd remediation is still
within budget.

For a run that should survive terminal closure or user logout, generate a
runbook plus supervisor and watchdog LaunchAgent plists with the unattended
wrapper. The wrapper performs the same preflight as the supervisor, runs a
10000-case no-GPU supervisor endurance drill by default, runs a no-GPU watchdog
launchd-remediation drill by default, writes `<output-dir>/unattended_run.json`
with the exact supervisor/watchdog/live-status/audit/drill commands, and can emit
plists whose `KeepAlive` policies restart each role only after unsuccessful
exits. The supervisor drill proves the resumable runner survives nonzero exits,
signal exits, stale heartbeats, missing heartbeats, and repeated restarts. The
watchdog drill proves a stale artifact-health state causes the watchdog to run
bounded launchd remediation and then observe restored health from the supervisor
LaunchAgent path. The supervisor plist keeps the resumable caption runner alive;
the watchdog plist keeps independent artifact-health audits running so terminal
closure does not remove the health probe. For strict 10k set-and-forget plans,
generated LaunchAgent commands are wrapped with `/usr/bin/caffeinate -dimsu` by
default so macOS idle sleep does not pause the GPU run; use
`--no-launchd-caffeinate` only for a manually supervised diagnostic plan. When a
supervisor LaunchAgent is part of the generated plan, the watchdog command also
includes bounded supervisor remediation by default,
including the supervisor plist path for rebootstrap fallback:
`--watchdog-max-remediations 25`, `--watchdog-remediation-cooldown 300`, and
`--watchdog-remediation-timeout 30`. Set
`--watchdog-max-remediations 0` to disable that behavior for a manual diagnostic
plan; explicit 10k set-and-forget readiness treats disabled watchdog remediation
as an error.
Use `--install-launchd-plists` for the actual set-and-forget handoff. With that
flag, the wrapper writes both plists, runs the launch gates, records the
`launchctl` commands and their results in the runbook, bootstraps both
LaunchAgents, relies on each plist's `RunAtLoad` start policy, verifies each
loaded service with `launchctl print`, and exits without also starting a
foreground supervisor. Each launchctl step has a timeout fuse so a stuck
launchd command fails closed in the runbook instead of hanging the wrapper
forever. When strict readiness is active through `--require-readiness-ok` or
`--tenk-set-and-forget`, a successful `launchctl` install is still not enough:
the wrapper then polls the live operation audit until the supervisor,
watchdog, watchdog status/state artifacts, sleep-prevention assertion, runbook,
readiness file, and live caption artifacts are all healthy. If that post-install
operation audit does not reach `ok` before
`--post-install-operation-audit-timeout`, the wrapper writes
`post_install_operation_audit` into the runbook, updates `readiness.json`, and
returns a precheck failure instead of a false successful handoff. Use
`--skip-post-install-operation-audit` only for a manual diagnostic handoff;
explicit 10k set-and-forget launch readiness rejects that skip. In `--dry-run`
mode the wrapper still writes the plists and records the launchctl plan, but it
does not call `launchctl` or the post-install operation audit.
When wrapper-level preflight sees a live runner lock, it records that state in
the runbook as deferred to the supervisor instead of failing the launch plan; the
supervisor then waits on that owner using the live-lock wait policy above.
For a run that is already actively processing in an artifact directory, use
`--adopt-live-run` with `--install-launchd-plists` instead of starting another
foreground supervisor. This is an explicit takeover mode: wrapper preflight must
find a live runner lock, the live set-and-forget artifact audit must be clean,
the active runner must advertise `graceful_restart_request` in its
`runner_capabilities`, the active run must have enough completed cases for
throughput evidence, prompt budget telemetry coverage must meet the adoption
gate, and the generated LaunchAgents must be installed. When those checks pass,
the LaunchAgent supervisor waits on the existing live lock rather than launching
a competing writer; after the old owner clears, launchd owns the resume/restart
path. If the active runner does not advertise cooperative restart support,
adoption fails closed because launchd can restart the supervisor but cannot
prove it can safely interrupt an older live runner. Without `--adopt-live-run`,
a live runner lock remains a warning or blocking condition for strict readiness
rather than being silently treated as a handoff.
The drill reports are written to `<output-dir>/supervisor_drill/drill_report.json`
and `<output-dir>/watchdog_drill/watchdog_drill_report.json`, then embedded in
the runbook. A failed required drill blocks launch before Qwen is loaded.
Use `--supervisor-drill-cases` and `--supervisor-drill-chunk-size` to tune the
synthetic launch gate; use `--supervisor-drill-cases 1` for the minimal
single-case restart drill. Use `--skip-supervisor-drill` only when a separate
recent drill report already proves the same checkout and Python environment.
The wrapper also writes `<output-dir>/readiness.json` and embeds the same
report under `readiness` in `unattended_run.json`. This report is the
single-file set-and-forget checklist: it reduces preflight, model-cache state,
disk budget, live disk-reserve command coverage, supervisor drill evidence,
pilot certification, pilot runner-capability and stream-loop telemetry proof, prompt-budget gates,
recovery commands, live-status command coverage, supervisor LaunchAgent output,
watchdog LaunchAgent output, operational-audit command coverage, and watchdog
latest-status/state artifact coverage into one status. It also reports whether
the watchdog can automatically kickstart or rebootstrap the supervisor
LaunchAgent after repeated unhealthy checks. A certification report whose
top-level status is `ok` but lacks current `runner_capabilities` or complete
`prompt_budget` telemetry is treated as stale evidence and fails readiness when
pilot certification is required. The current capability set includes the
`caption_io_event_summary` marker, so a pilot produced before copied
`qwen_caption_io` event summaries is not accepted for set-and-forget launch.
Pilot certification also validates the runtime trace source: prompt-budget or
loop-guard evidence copied from unbound `qwen_caption_io_latest.*` is rejected,
and strict set-and-forget pilots require run-bound
`qwen_caption_io_per_run` runtime prompt-budget evidence for every generated
latest case. Strict readiness repeats that source check, so a hand-edited or
older saved certification report with prompt-budget rows but no run-bound
`qwen_caption_io_sources` proof still fails before launch.
Readiness also rechecks the prompt-size and
prompt-budget adaptation gates recorded in the certification report, so a saved
report cannot bypass the current launch limits. Normal launch mode still blocks
fatal errors and records warnings.
Use `--tenk-set-and-forget` when the intent is a true 10k
unattended run; it enables the same strict readiness gate as
`--require-readiness-ok`, records the explicit 10k mode in the runbook, and
blocks gaps such as missing pilot certification, skipped drill, pending model
downloads, explicit `--allow-model-download`, fewer than three VLM attempts,
disabled prompt-size ceiling, no supervisor LaunchAgent plist, no watchdog
LaunchAgent plist, missing watchdog latest-status/state artifacts, disabled
watchdog remediation, disabled launchd sleep prevention, a missing/disabled
projected-duration gate, or a missing/disabled live disk-reserve gate. In strict
10k mode, generated plists alone are still a manual plan; include
`--install-launchd-plists` so readiness reflects a real launchd-backed
set-and-forget start. The three-attempt minimum is intentional: a signal exit or
Metal abort must get multiple isolated VLM attempts. After a signal exit, the
next attempt uses the configured minimum retry image side by default, before
the parent runner is allowed to preserve progress with deterministic
count/layout fallback.
If no `--require-pilot-certification`, `--create-pilot-output-dir`, or
`--adopt-live-run` option is supplied, `--tenk-set-and-forget` automatically
plans a generated pilot in `<output-dir>/pilot`, certifies that pilot, and only
then installs or hands off the large run. The auto pilot uses a target-scaled
sample floor: for the default 10k target it runs and requires 300 timed pilot
cases, keeping the stress-plus-random sample-selection metadata that
certification expects. It also defaults the pilot prompt-size ceiling to `9000`
tokens. Explicit `--pilot-sample-size`, `--pilot-min-cases`, or
`--pilot-max-prompt-tokens` values remain respected for manual diagnostics;
strict 10k readiness still rejects an explicit `--pilot-max-prompt-tokens 0`,
an undersized explicit pilot, or disabled deterministic-recovery confidence.
Pilot artifacts must be isolated from the large-run artifact directory: the
default `<output-dir>/pilot` is valid, but pointing `--create-pilot-output-dir`
or `--require-pilot-certification` at the same directory as `--output-dir`
fails the static 10k launch gate before any drill, pilot, certification, or GPU
work starts.
Before it runs the no-GPU drills, generated pilot, live-run adoption audit,
pilot certification, or large supervisor, the wrapper performs a static 10k
launch gate and writes it to `tenk_static_launch_gate` in the runbook. That gate
fails fast on configuration defects that can be known without touching Qwen or a
live runner: disabled preflight, skipped drills, missing pilot/adoption
evidence, a pilot directory that is the same as the target output directory,
disabled prompt-size or prompt-budget gates, missing LaunchAgents, disabled
launchd installation, disabled post-install operation-audit proof, disabled
sleep prevention, disabled watchdog remediation, disabled cooperative restart
requests, and disabled duration or disk-reserve gates. This keeps unattended
setup mistakes from spending GPU time or producing misleading late-stage
adoption/certification artifacts.

```bash
./.venv-macos/bin/python tools/run_qwen_caption_unattended.py \
  --dry-run \
  --tenk-set-and-forget \
  --install-launchd-plists \
  --write-launchd-plist ~/Library/LaunchAgents/com.tator.qwen-caption-soak.plist \
  --launchd-label com.tator.qwen-caption-soak \
  --write-watchdog-launchd-plist ~/Library/LaunchAgents/com.tator.qwen-caption-soak.watchdog.plist \
  --watchdog-launchd-label com.tator.qwen-caption-soak.watchdog \
  -- \
  --dataset-root /path/to/dataset \
  --output-dir /path/to/caption_soak \
  --all-images \
  --save-dataset-text-labels \
  --allow-model-download \
  --max-runner-restarts 25
```

After reviewing the dry-run runbook and preflight status, remove `--dry-run` and
rerun the same command to install and start both LaunchAgents. If you prefer a
manual handoff, omit `--install-launchd-plists` and load both plists yourself:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tator.qwen-caption-soak.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tator.qwen-caption-soak.watchdog.plist
```

The supervisor still runs the strict completion audit before returning success.
In set-and-forget mode, that audit keeps failed cases, quality warnings, signal
exit attempts, stale heartbeats, and attempt overruns bounded and explicit,
while allowing recovered loop events up to the configured loop-recovery rate.
The live operation audit also treats launchd restart gaps as bounded transient
states rather than immediate hard failures: by default it waits up to 10 seconds
for states such as `spawn scheduled` to become running, records the state
history and retry count in the `*_launchd` check, and still fails if the service
does not settle. Use `--launchd-settle-seconds` and
`--launchd-settle-interval` when an environment needs a shorter or longer
restart-settle window.
For live-run adoption, strict operation audit also requires a successful
`live_runner_restart_capability` certification, and it fails if the watchdog
command disables cooperative restart requests with
`--graceful-restart-timeout 0`.
The default loop-recovery and loop-guard rates are 5%, and the default recovered
signal-exit attempt-row rate is 1%; passing
`--max-loop-recovery-case-rate 0`, `--max-loop-guard-case-rate 0`, or
`--max-signal-exit-attempt-row-rate 0` restores manual-strict behavior where any
matching recovery blocks completion.
For explicit 10k set-and-forget launches, the generated watchdog and final audit
commands also include `--max-projected-duration-hours 336` unless the supervisor
arguments supply another value. The generated watchdog, live-status, and final
audit commands also include the supervisor `--min-free-gb` value so a disk
reserve failure trips the same unattended health path as stale heartbeats,
attempt overruns, and excessive recovery rates. If the supervisor saves dataset
text labels, strict operation audit also requires the generated watchdog,
live-status, and final-audit commands to carry `--require-saved-text-labels`.
Strict operation audit also verifies that watchdog, live-status, final-audit,
and any pilot-certification command carry the degraded-rate gates, including
`--max-loop-guard-case-rate`, so a runbook cannot silently drop the loop-inspector
health policy.
Use the runbook's `final_audit_shell` command before trusting a completed 10k
run. During a live run, use the runbook's `operational_audit_shell` command to
verify the current handoff envelope: loaded supervisor and watchdog
LaunchAgents, loaded LaunchAgent arguments matching the on-disk plists and
runbook commands, fresh watchdog latest/status files, restart-stable watchdog
state, active caffeinate sleep assertions, and the live caption artifact audit.
In strict mode, the watchdog latest snapshot must also name the runbook output
directory and the watchdog state must name the runbook state path. This prevents
an old or cross-run `watchdog_latest.json` / `watchdog_state.json` from making a
different run look healthy during a walk-away handoff.
For a walk-away handoff, the command should include `--strict-set-and-forget`;
new runbooks generated with `--require-readiness-ok` or `--tenk-set-and-forget`
include that flag automatically, and the launcher runs this audit automatically
after installing LaunchAgents before it reports handoff success. Strict audit
also requires the runbook's recorded `operational_audit` command to include
`--allow-running-incomplete`, `--compact`, `--write-json`, and
`--strict-set-and-forget`, so a copied or hand-edited runbook cannot drop its
own live self-audit recipe. Strict operational audit is the proof that the
set-and-forget machinery is still installed, self-monitoring, allowed to
remediate the supervisor, protected against sleep, and backed by
supervisor/watchdog plist paths under a
`Library/LaunchAgents` directory. It also verifies restart-safe plist policy
(`RunAtLoad`, unsuccessful exit `KeepAlive`, throttle interval, repository
working directory, and stdout / stderr capture paths) and checks that the
supervisor command itself carries the projected-duration and disk-reserve gates,
so terminal success cannot bypass the same durability limits used by the
watchdog and final audit. The plain artifact audit only proves that the latest
caption files are currently healthy:

```bash
./.venv-macos/bin/python tools/audit_qwen_caption_operation.py \
  /path/to/unattended_run.json \
  --allow-running-incomplete \
  --strict-set-and-forget \
  --compact
```

For only the caption artifacts, use the live status audit with `--compact` for a
short operator-facing status page:

```bash
./.venv-macos/bin/python tools/audit_qwen_caption_soak.py \
  /path/to/caption_soak \
  --set-and-forget \
  --allow-running-incomplete \
  --max-projected-duration-hours 336 \
  --min-free-gb 5 \
  --compact
```

The compact view prints progress, heartbeat state, failure counts, recovery
rates, signal-exit rates, projected duration, disk-reserve check status, any
active attempt's case, attempt number, runtime, and timeout, and any
near-threshold caution bands from the same audit report that powers the JSON
output. Caption coverage includes durable successful rows from prior resumes and
the optional `skipped_completed` resume rows from `--record-resume-skips`, so a
restarted supervisor still has to prove that every completed case has a
non-empty row in `captions.jsonl` without requiring duplicate ledger rows. It is
a readability layer only; the exit code and health gates are unchanged.

Before scaling a new caption configuration to a 10k-image set-and-forget run,
certify a smaller pilot artifact directory. The certification command runs the
strict artifact audit, checks the pilot sample size, and projects the observed
mean per-image runtime against the target duration with a safety factor. It also
projects the target duration as if every target image ran at the pilot p95 case
time. That p95 gate defaults to the same value as `--max-duration-hours`; set
`--max-p95-duration-hours -1` only for manual diagnostics where tail runtime is
being inspected but should not block launch.

When a pilot uses `--sample-size`, selected cases are stress-biased rather than
purely random: the runner forces representative dense, class-diverse,
dominant-class, sparse, empty, and per-mode dense cases into the sample first,
then fills the remainder with deterministic random cases. The manifest and
preflight report record `sample_selection` so the operator can verify which hard
cases were included. Pilot certification fails sampled pilot artifacts that lack
this metadata, because an old purely random pilot is not strong evidence for a
10k set-and-forget launch.

```bash
./.venv-macos/bin/python tools/certify_qwen_caption_soak.py \
  /path/to/pilot_caption_soak \
  --target-cases 10000 \
  --max-duration-hours 336 \
  --max-p95-duration-hours 336 \
  --min-pilot-cases 300 \
  --duration-safety-factor 1.25 \
  --max-prompt-tokens 9000 \
  --max-prompt-budget-adapted-case-rate 1 \
  --set-and-forget
```

Exit code `0` means the pilot artifacts are clean enough to scale. A non-zero
exit means the long run should not be launched unattended yet; inspect
`certification.json`, fix the prompt/model/runtime setting that failed, and run
another pilot. This keeps recovery automatic during normal operation while
still blocking fragile configurations before they spend days on a large dataset.
Pilot certification distinguishes the latest/resumed case state from generated
model evidence. `skipped_completed` and `skipped_existing_caption` rows can
prove that an unattended restart preserved progress, but they do not count as
generated pilot cases unless the same `results.jsonl` ledger already contains an
earlier non-skipped successful generated row for that case. The report exposes
both `pilot_cases` and `generated_pilot_cases`; strict set-and-forget launches
require enough generated cases, current runner capability markers, and run-bound
`qwen_caption_io_per_run` prompt-budget telemetry for those generated rows. A
manual recovery run may still record explicit skip breadcrumbs, but a 10k
set-and-forget launch cannot be certified by a skipped-only or hand-assembled
artifact. Backend dataset jobs and unattended launch readiness re-check
`generated_pilot_cases` against the configured pilot minimum after certification
returns, so an old, mocked, or hand-edited `ok` report without generated model
evidence still blocks the runner before preflight or launchd handoff.
Runtime-tail skew is still reported as an advisory, but p95 projected target
duration is now a launch gate for set-and-forget certification. This catches
caption settings where the average case would finish inside the budget but a
tail-heavy distribution would make a two-week run miss its deadline.
When a 10k launch requires pilot certification, the backend and unattended
wrapper pass the expected runner `run_settings` fingerprint into this command,
so a pilot generated with a different prompt stack, model set, output-token
budget, windowing setup, or decode configuration cannot certify the large run.
For the actual long LaunchAgent/supervisor command, wire that pilot gate into
the unattended wrapper. If a pilot already exists, point
`--require-pilot-certification` at it:

```bash
./.venv-macos/bin/python tools/run_qwen_caption_unattended.py \
  --tenk-set-and-forget \
  --require-pilot-certification /path/to/pilot_caption_soak \
  --pilot-target-cases 10000 \
  --pilot-max-duration-hours 336 \
  --pilot-max-p95-duration-hours 336 \
  --pilot-min-cases 300 \
  --pilot-max-prompt-tokens 9000 \
  --pilot-max-prompt-budget-adapted-case-rate 1 \
  --install-launchd-plists \
  --write-launchd-plist ~/Library/LaunchAgents/com.tator.qwen-caption-soak.plist \
  --write-watchdog-launchd-plist ~/Library/LaunchAgents/com.tator.qwen-caption-soak.watchdog.plist \
  -- \
  --dataset-root /path/to/dataset \
  --output-dir /path/to/caption_soak \
  --all-images \
  --save-dataset-text-labels \
  --max-runner-restarts 25
```

For a one-command set-and-forget launch, let the wrapper create the pilot first.
`--create-pilot-output-dir` runs a supervised sample from the same target
population as the large run, using the same model, prompt, decode, timeout, and
health-gate settings, then certifies that pilot before launching the full
supervisor. For example, an `--all-images` launch creates an `--all-images`
sample pilot instead of falling back to the default labeled-case subset. Dry-runs
write the exact pilot-generation command into `unattended_run.json` without
loading Qwen. Generated pilots are read-only with respect to dataset text labels:
even when the later large run uses `--save-dataset-text-labels`, the pilot writes
only its own artifacts under the pilot output directory and does not mutate the
dataset. Do not reuse the production `--output-dir` as the pilot directory; use
the automatic `<output-dir>/pilot` location or another separate artifact
directory so pilot evidence cannot be mixed with production resume artifacts.

```bash
./.venv-macos/bin/python tools/run_qwen_caption_unattended.py \
  --tenk-set-and-forget \
  --create-pilot-output-dir /path/to/pilot_caption_soak \
  --pilot-sample-size 300 \
  --pilot-target-cases 10000 \
  --pilot-max-duration-hours 336 \
  --pilot-max-p95-duration-hours 336 \
  --pilot-min-cases 300 \
  --pilot-max-prompt-tokens 9000 \
  --pilot-max-prompt-budget-adapted-case-rate 1 \
  --install-launchd-plists \
  --write-launchd-plist ~/Library/LaunchAgents/com.tator.qwen-caption-soak.plist \
  --write-watchdog-launchd-plist ~/Library/LaunchAgents/com.tator.qwen-caption-soak.watchdog.plist \
  -- \
  --dataset-root /path/to/dataset \
  --output-dir /path/to/caption_soak \
  --all-images \
  --save-dataset-text-labels \
  --max-runner-restarts 25
```

With `--require-pilot-certification` or `--create-pilot-output-dir`, the wrapper
writes `<output-dir>/required_pilot_certification.json`, embeds the report in
`unattended_run.json`, and exits before starting the supervisor when the pilot is
too small, too slow, stale, incomplete, missing prompt-budget telemetry, above
the configured prompt-size gate, missing current runner capability markers,
degraded, or carrying recovered signal-exit attempts above the configured
threshold. It also rejects stale runtime prompt-budget evidence whose source is
`qwen_caption_io_latest` instead of the worker's per-run Qwen trace. The
generated certification command includes `--set-and-forget` when
the supervisor run is in set-and-forget mode, so the pilot uses the same bounded
recovery defaults as the large unattended run. The strict default threshold for
signal exits is `0` for manual audits; with `--set-and-forget`, the default
threshold is `0.05`. Add `--max-signal-exit-attempt-row-rate 0` when the pilot
must prove zero recovered `SIGABRT` or other signal-exit attempts before
scaling. The deterministic count/layout fallback budget is tighter:
`--set-and-forget` defaults `--max-deterministic-recovery-case-rate` to `0.01`,
and certification also applies a one-sided confidence bound, default `0.95`,
to generated pilot rows. Use `--deterministic-recovery-confidence 0` only for a
diagnostic pilot that should not certify a 10k handoff. That is the intended
set-and-forget launch path for 10k-scale runs. Use
`--no-pilot-prompt-budget-required` or `--no-require-runner-capabilities` only
for legacy diagnostic review, not for a certified set-and-forget handoff. For
strict set-and-forget launches, do not rely on first-run model downloads:
pre-download the selected caption/refinement models until readiness reports
`model_downloads` as `ok`; strict 10k mode also requires the static
`model_downloads_disabled` gate to pass.

The watchdog appends compact snapshots to `watchdog.jsonl` in the artifact
directory, writes the full current snapshot to `watchdog_latest.json`, persists
restart-stable progress and remediation counters in `watchdog_state.json`, and
prints compact JSON events to stdout. The latest-status file is the quick,
detailed check for a long unattended run. The JSONL file remains compact
forensic history and intentionally omits heavyweight nested audit detail such as
per-threshold `rate_headroom` records so a two-week run does not grow noisy logs
just from routine polling. It still retains aggregate degraded-rate telemetry,
including loop-guard counts/rates and prompt-budget coverage/adaptation, so the
compact history remains useful for unattended-run triage. Each watchdog event
carries both `checked_at` ISO-8601 time and numeric `checked_epoch`/`time`
fields, so long-run forensic tools can sort and age status snapshots without
reparsing display text. The state file lets a restarted watchdog continue the
same no-progress timer, consecutive unhealthy count, remediation count, and
remediation cooldown instead of granting a fresh grace period after every
LaunchAgent restart. Each event includes the audit's structured
`active_attempt` snapshot when a caption attempt is running, so a set-and-forget
monitor can report which case is active and how close it is to the attempt
timeout without scraping compact text. When `worker_progress` is present inside
that active attempt, `--max-no-progress` resets on active Qwen worker movement
such as changed worker sequence, stage, generated-token count, live-output size,
or prompt trace count; otherwise it falls back to completed-case progress. A
long image can therefore keep running while tokens or stages advance, but a
frozen caption worker still trips the same remediation path. When the
projected-duration gate is
enabled, each event also includes a top-level `runtime_projection` summary with
processed cases, cases/hour, projected total hours, remaining hours, and the
active duration budget. The watchdog uses
live-health audit while the run is still `running`, then switches to strict
completion audit when the runner heartbeat becomes terminal. Exit code `0` means
a clean terminal run or healthy bounded watch, `1` means a terminal run failed
strict audit, and `2` means the run stayed unhealthy for the configured
consecutive-check threshold.
In addition to the artifact audit, the watchdog keeps its own completed-case
progress timer. `--max-no-progress` defaults to `3600` seconds and marks a
nonterminal watch unhealthy when `processed_cases` has not increased for that
long; use `0` only when debugging a deliberately paused run.
