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
  generated-QA accumulator, verifier/rewrite passes, artifact logging, run
  settings fingerprint, and resume gates.

The OpenAI provider is intentionally dataset-backed. Single-image remote
captioning still goes through a persisted backend job when set-and-forget and a
caption dataset are selected; it does not fall through to the browser-only local
diagnostic endpoint. The backend reads `OPENAI_API_KEY` first, then the
configured backend-local key file path. The key value is never inserted into
prompts, exports, job logs, or run settings.

OpenAI visual caption and generated-QA/top-up calls send both text and image
content. Text-only editor, guard, verifier, and rewrite stages remain text-only.
The default image detail is `original` for full-resolution annotation work;
`auto`, `high`, and `low` are available with UI warnings because they may change
visual fidelity or cost. Current direct launches use synchronous Responses API
calls. Batch pricing is exposed only as a planning estimate until a separate
offline Batch submission/collection path exists.

The remote cost estimate in the UI is a planning aid. It combines the selected
provider model, image detail, pricing tier, current image count or Max images
cap, caption/QA target settings, top-up attempts, verifier calls, box-policy
prompt estimate, and output-token cap. Actual billing can differ because the
API tokenizes images and prompts exactly at request time and because recovery
paths are conditional.

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
Q&A coverage.

## Safety Rails

- Manual caption records are not soft-archived by generated-row replace modes.
- Existing generated rows are soft-archived rather than deleted.
- Resume recovery rows marked `skipped_completed` are allowed to re-save their
  recovered caption artifact even if per-image targets appear complete.
- Prompt preview includes the run policy metadata so prompt content and
  persistence semantics can be reviewed together.
- Provider settings are part of the run settings fingerprint. A resumed run must
  use the same local/remote provider, OpenAI model, image detail, key-file path,
  service tier, timeout, prompts, and generation settings before new rows are
  appended to an existing manifest.
