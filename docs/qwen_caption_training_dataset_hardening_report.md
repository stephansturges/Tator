# Qwen Caption Training Dataset Hardening Report

Date: 2026-06-29

## Scope

This pass hardened the Qwen caption path used by **Create VLM training dataset**.
The product goal remains unchanged: the UI starts a dataset-backed VLM training
data job that can create caption0 plus generated image/question/answer rows,
preserve provenance, write review/export artifacts, and keep generated content
separate from source annotations.

## Failure Mode Addressed

The default set-and-forget path could look like it was stuck in
`caption runner attempt...` and cooldown because the UI did not send an explicit
auto-resume limit. The backend fallback was high, so a persistent first-image
failure or GPU runtime fault could trigger many restart/resume cycles before the
job stopped.

There were also observability gaps:

- Live progress could display synthetic case IDs instead of the real image name.
- The backend case builder filtered requested image names with a set and then
  processed the dataset manifest order, so a UI-sent batch could start on a
  different frame than the current/requested one.
- The all-image UI actions sent the raw loaded-image order, so even after the
  backend preserved request order, a selected image in the middle of the list
  would not be the first backend case.
- The active image in the viewer did not follow backend progress.
- The cancel button could miss an auto-resumed replacement job if the UI only
  knew the original job ID.
- Active backend progress could show a stale previous caption when the current
  worker had not emitted live text yet.
- Generated-QA stages were not clearly separated in progress output.

## Changes

- Added an **Auto-resume limit** control to the caption generation settings.
  The UI default is 2.
- Lowered the backend fallback
  `QWEN_CAPTION_SET_AND_FORGET_MAX_AUTO_RESUMES` default to 2 for non-UI callers.
- Sent `max_auto_resumes` explicitly from both single-image and batch/training
  dataset launches.
- Preserved the setting in caption recipes.
- Added a **Follow backend image** toggle, enabled by default, so long backend
  jobs can switch the viewer to the image currently being processed.
- Propagated `image_name` through runner heartbeats, result rows, backend live
  progress, and UI progress summaries.
- Preserved the UI's requested image order when building backend cases, including
  sequential case naming after ordering and `max_images` trimming.
- Changed **Caption all images** and **Create VLM training dataset** to send the
  selected image first, then continue through the loaded list and wrap earlier
  images to the end.
- Changed cancel handling to refresh active backend jobs for the selected
  dataset and cancel every active matching job, including auto-resumed
  replacements.
- Prevented active progress from falling back to stale latest-caption text.
- Added generated-QA progress markers so instruction-dataset work is visible as
  a distinct stage.
- Confirmed the model dropdown availability styling remains present: missing
  models are shown in red and locally available models in white.

## Expected Operator Behavior

With defaults, a set-and-forget training-dataset job still gets isolated child
attempts and backend crash recovery, but persistent failures stop after a small
bounded number of automatic resume cycles. Operators can raise the limit for a
supervised long run, set it to 0 to disable automatic resumes, or use the cancel
controls to stop active jobs.

The UI should now show the real image name, current attempt, progress stage, and
live output for the currently active backend worker. All-image launches begin
with the selected frame so the first visible backend case matches the operator's
current image. If an auto-resumed job is created, the monitor follows it
automatically and the cancel action targets it.

## Verification

Focused checks run in this pass:

- `python -m py_compile localinferenceapi.py models/schemas.py tools/run_qwen_caption_flow_benchmark.py`
- `python -m py_compile tools/run_qwen_caption_ui_smoke.py`
- `node --check ybat-master/ybat.js`
- `git diff --check`
- `pytest tests/test_qwen_caption_dataset_job.py -q`
- `pytest tests/test_qwen_caption_flow_benchmark.py -q`
- `pytest tests/test_labeling_panel_layout_contract.py tests/test_qwen_caption_ui_smoke_tool.py -q`

The live backend was not left running by this report.
