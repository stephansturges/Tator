# Qwen Caption Run Policies

This document describes how caption dataset runs decide which images to process
and which generated rows become active training data.

## User Story

Dataset creation runs can take a long time and may be run repeatedly. Users need
to run small tests, resume larger runs, fill missing images, replace generated
rows, append variants, or add only more generated Q&A without losing existing
work.

The core mechanism remains visual model generation. Run policies only decide
which cases should run and how generated caption or Q&A rows are persisted.

## Durable Caption Artifact Store

Caption and generated-QA outputs are persisted in two layers:

- The legacy dataset overlay remains the compatibility layer used by current
  caption bundles, text labels, instruction records, and training exports.
- The backend caption artifact store under `uploads/caption_artifact_store/`
  is the canonical append-only output ledger for new work.

The artifact store identifies images only by `image_sha256`. Paths, filenames,
dataset IDs, split names, and dataset image keys are aliases/provenance, not
identity. A user may load the same image through another dataset later; newly
generated or imported artifacts should still attach to the same image hash.

Each stored output links:

- the image hash,
- a hashed prompt context,
- a hashed generation spec,
- a generation attempt record,
- the artifact payload (`base_caption` or `qa_pair`),
- and optional caption-set membership events.

Batch shards, local runner attempts, OpenAI Standard calls, and manual edits are
transport or provenance details. UI coverage should be reported per image from
the artifact store when available, while legacy overlay records remain the
compatibility projection until downstream export paths fully consume caption
sets directly.

## Modes

- `run_kind=production`: normal durable run.
- `run_kind=test`: same generation path, marked as a test run.
- `test_outputs_count_toward_completion=true`: test outputs are durable rows and
  count in later coverage checks.
- `test_outputs_count_toward_completion=false`: test outputs remain diagnostic
  runner artifacts and are not written into caption stores.

Small smoke runs use the same durable planner as production. Set **Run kind** to
`test`, set **Max images** to the desired sample size, and leave
`test_outputs_count_toward_completion=true` when the generated rows should be
kept as complete work. This avoids a separate disposable path: prompts, visual
model calls, QA top-ups, verifier passes, logs, artifacts, and resume behavior
remain identical to the full run.

## Caption Providers

- `caption_provider=local_qwen`: use the local Qwen runtime and local model
  cache/Metal health gates.
- `caption_provider=openai`: use the OpenAI Responses API for the model-call
  primitives while keeping the same dataset planner, caption prompt builder,
  artifact logging, run settings fingerprint, and resume gates. OpenAI Standard
  runs use the normal synchronous dataset job path, including generated-QA
  accumulator and verifier/rewrite stages. OpenAI Batch runs use a durable
  throughput path: one visual request per image asks for the base caption and
  generated-QA rows together, then held incomplete rows are handled by explicit
  visual `Catch up QA` Batch jobs.

The OpenAI provider is intentionally dataset-backed. Single-image remote
captioning still goes through a persisted backend job when set-and-forget and a
caption dataset are selected; it does not fall through to the browser-only local
diagnostic endpoint. The backend reads `OPENAI_API_KEY` first, then the
configured backend-local key file path. The key value is never inserted into
prompts, exports, job logs, run settings, or browser-visible job payloads. The
server-side durable job record may retain the backend-local key path so a Batch
job can be resumed, polled, collected, or cancelled after a browser reload.

Set-and-forget discovery uses a compact `job.discovery.json` sidecar beside each
durable `job.json`. The sidecar is invalidated by the source file's size and
nanosecond mtime and contains only the fields needed to select a possible
adoption or auto-resume candidate. Periodic sweeps stream these summaries,
precompute cancelled artifact lineages, and load the full job payload only for a
selected live or resumable candidate. Existing jobs are migrated lazily. This
keeps thousands of historical multi-megabyte result/log payloads out of backend
memory without weakening restart recovery.

OpenAI visual caption and generated-QA/top-up calls send both text and image
content. Text-only editor, guard, verifier, and rewrite stages remain text-only.
The default image detail is `original` for full-resolution annotation work when
the selected model supports it; `auto`, `high`, and `low` are available with UI
warnings because they may change visual fidelity or cost. The backend publishes
provider metadata for pricing, supported image-detail modes, and the default
reasoning effort, and launch/preview endpoints validate the model/detail pair
server-side. The remote UI defaults to Batch durable async because paid dataset
creation must be recoverable across browser reloads and backend restarts.
Standard launches remain available for synchronous diagnostics and use
Responses API calls in the normal persisted dataset job path. Batch launches use
durable OpenAI Batch jobs that are submitted, polled, collected, and explicitly
imported through the backend.

Batch runs are sharded for large datasets. The UI default is `100` images per
OpenAI Batch shard. A logical dataset run may therefore own many child Batch
jobs, while the backend keeps one parent run for status, recovery, collection,
and import. Completed shards can be collected and imported without waiting for
every shard in the parent run to finish.

Batch prompt construction uses the same durable case planner as local and
standard OpenAI runs. Each case carries its own generated-QA request count, so a
fill-missing run that only lacks one imposed question on an otherwise complete
image asks the remote model for that remaining QA work instead of regenerating
the full global QA target.

Collected Batch rows are split by training readiness. Rows that contain the
requested caption plus full generated-QA count are accepted into
`captions.jsonl`; paid rows with a usable caption but an underfilled generated-QA
array are held in `incomplete_captions.jsonl`. Held rows stay visible in the
manager attention filter and can be topped up with `Catch up QA`, which resends
the image and asks only for the missing non-duplicate QA pairs.

Remote Batch import is bound to the submitted target image fingerprint and the
loaded dataset. Import requires an actively selected caption dataset; the
backend rejects imports without an explicit `dataset_id` rather than falling
back to an older stored snapshot. Exact dataset-ID matches import normally when
image hashes still match. If the dataset ID changed but the target image
fingerprint still matches, the manager requires explicit confirmation before
import. Label snapshots are stored as prompt provenance and produce warnings
when they changed after submission. They are not a hard import block unless the
backend request explicitly enables strict label-hash matching. Already imported
Batch shards are terminal for import purposes and are not collected or imported
again by parent collection jobs. Partially imported jobs also write durable
per-row import markers so repeated imports do not duplicate accepted rows while
incomplete rows remain held for catch-up.

The remote cost estimate in the UI is a planning aid. It combines the selected
provider model, image detail, reasoning effort, pricing tier, current image
count or Max images cap, caption/QA target settings, top-up attempts, verifier
calls, box-policy prompt estimate, and output-token cap. In OpenAI Standard
mode, QA-only and zero-base-increment runs still perform a visual
caption/grounding call before generated-QA creation; that base caption is used
for QA context but is not persisted as a base-caption row. Actual billing can
differ because the API tokenizes images and prompts exactly at request time and
because recovery paths are conditional.

For remote Batch caption-plus-QA requests, leave Max output tokens on Auto
unless there is a specific reason to cap it. Auto currently resolves to a
larger remote cap than local captioning so JSON QA payloads are not cut off.
Billing is usage-based; the cap prevents runaway generation but is not itself a
promise that all capped tokens will be billed. High caps may still affect
admission, latency, or rate-limit planning, so they should be kept bounded.

When caption0 is disabled as a training row family, the effective base-caption
coverage target is zero. This is separate from visual grounding: generated-QA
runs may still generate or use a caption internally for QA context, but excluded
caption0 rows do not make images incomplete. If caption0 and generated QA are
both disabled while deterministic metadata QA is enabled, no VLM work is
required. Local runs and OpenAI Batch runs materialize the scoped instruction
bundle/report directly from source annotations. The export respects the run
scope, including selected image names, split, and Max images, so a test run does
not silently produce a full-dataset deterministic bundle.

Remote Batch jobs also persist per-job cost summaries. At submission the backend
records an estimated token/cost summary in the job JSON. Batch estimates read
the planned per-image generated-QA counts from `cases.json`, so catch-up and
fill-missing jobs are estimated from the actual remaining work rather than the
default target for every image. After outputs are collected, the backend records
a usage-based actual summary from Batch response usage or collected response-row
usage. Standard OpenAI caption dataset jobs retain Responses usage in their
backend job result and expose a usage-based `openai_cost_summary`. Sharded
parent jobs aggregate the child summaries. The `Remote spend` control is
separate: it uses the OpenAI organization Costs API with an admin key to show
account-level spend for the last N days, grouped by API key/model/Batch where
the API provides those fields.

Batch recovery is deliberately tolerant of restored or manually recovered
artifacts. Raw OpenAI statuses such as `completed` are normalized into the local
manager state model before collection/import/catch-up decisions, and a stale
`batch_status.json` snapshot is not allowed to override a newer completed Batch
response stored in the job record.

## Write Policies

- `fill_missing`: process images that have not reached the configured per-image
  active caption and generated-QA targets.
- `replace_generated`: process selected images and soft-archive prior generated
  caption and generated-QA rows before saving new generated rows. Manual rows are
  preserved.
- `append_variants`: process selected images and append new generated variants.
- `qa_only_extend`: preserve current base captions and add only enough generated
  Q&A rows to reach the target.
- `qa_only_replace`: preserve current base captions, soft-archive generated Q&A,
  and save replacement generated Q&A rows.

Soft-archived rows stay in the on-disk JSONL with `lifecycle_status` set to
`superseded`. Extra generated Q&A beyond the configured per-image total is saved
as `overflow`. Training export uses active rows only.

## Completion Targets

`completion_mode=per_image_totals` treats the configured target as the desired
final active count per image. For example, target `1` base caption and `8`
generated Q&A rows means an image with one caption and five active generated Q&A
rows will run and save three active Q&A rows; any additional generated rows are
kept as overflow audit rows.

`completion_mode=incremental` does not skip complete images. It is used for runs
whose purpose is to add another set of generated outputs. In this mode,
`increment_generated_qa_per_image` controls how many newly generated Q&A rows are
requested and kept active for each image. Imposed questions still raise the
request count when needed. `increment_base_captions_per_image=0` means the run
may still generate a caption internally for Q&A grounding, but that generated
caption is not stored as an active base-caption row.

## Coverage

The coverage endpoint reports active base-caption and generated-QA counts for
the selected dataset against the same targets used by the run planner. A primary
text label counts as one base caption. Active caption records count as stored
caption variants. Active generated-QA instruction records count toward generated
Q&A coverage. When imposed questions are configured, coverage also compares the
normalized active generated-QA question texts against those imposed questions.
An image is not complete until both the numeric generated-QA target and all
imposed-question requirements are satisfied. The UI sends the current imposed
question list to the coverage endpoint, displays missing imposed-question totals,
and keeps those images eligible for fill-missing, QA-only, or Batch catch-up
runs. For QA-only write policies, the effective base-caption target is zero,
matching the run planner and avoiding a misleading incomplete-readiness state
when the requested work is only Q&A. When caption0 rows are disabled for
training, the effective base-caption target is also zero. When generated-QA
rows are disabled for training, the effective generated-QA coverage target is
zero, so readiness is not blocked by a row family the user intentionally
excluded. Imposed questions only affect coverage when generated QA is enabled.

## Safety Rails

- Manual caption records are not soft-archived by generated-row replace modes.
- Existing generated rows are soft-archived rather than deleted.
- Resume recovery rows marked `skipped_completed` are allowed to re-save their
  recovered caption artifact even if per-image targets appear complete.
- Prompt preview includes the run policy metadata so prompt content and
  persistence semantics can be reviewed together. For OpenAI runs, preview also
  applies the same model/detail compatibility validation as launch.
- Provider settings are part of the run settings fingerprint. A resumed run must
  use the same local/remote provider, OpenAI model, image detail, reasoning
  effort, key-file path, service tier, timeout, prompts, and generation settings
  before new rows are appended to an existing manifest.
