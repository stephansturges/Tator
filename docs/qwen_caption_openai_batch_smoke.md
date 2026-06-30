# OpenAI Batch Caption Smoke Runner

This document records the remote-provider smoke path for caption plus generated
question-answer rows. It is intentionally separate from the local Qwen runner:
local runs are still useful for interactive development, while OpenAI Batch is
the durable high-throughput path for larger paid runs.

## API Shape

The batch runner uses the OpenAI Batch API with the Responses API as the
underlying endpoint:

- Each image is uploaded to `/v1/files` with `purpose=vision`.
- The runner writes one JSONL line per image with `method=POST`,
  `url=/v1/responses`, and a Responses `body`.
- Each Responses body uses `input_text` plus `input_image` by `file_id`, with
  `detail=original` by default.
- The JSONL file is uploaded to `/v1/files` with `purpose=batch`.
- The batch is created at `/v1/batches` with `endpoint=/v1/responses` and
  `completion_window=24h`.
- Results are joined back by `custom_id`; batch output order is not treated as
  stable.

Official references:

- Batch API guide: `https://developers.openai.com/api/docs/guides/batch`
- Vision input guide: `https://developers.openai.com/api/docs/guides/images-vision`
- Structured outputs guide:
  `https://developers.openai.com/api/docs/guides/structured-outputs`

Using `file_id` image inputs keeps the batch JSONL small even when source
images are full resolution. The runner still preserves the image upload ledger,
so rerunning the same output directory can resume without re-uploading already
registered vision files.

## Prompt And Output Contract

The batch prompt asks for one JSON object:

```json
{
  "caption": "final caption text",
  "qa_pairs": [
    {"question": "question text?", "answer": "answer text"}
  ]
}
```

The request also includes a Responses structured-output JSON schema for the
object shape. The exact QA target remains in the prompt and in post-processing,
instead of relying on optional schema keywords. This keeps the API request shape
conservative while still making failed or short QA rows observable.

The prompt uses the same dataset glossary path as the local runner. Counts and
representative box labels are canonicalized through the glossary before they
enter the prompt. If no glossary term exists for a class, the runner falls back
to the natural English form of the class name.

## Artifacts

For an output directory such as `$RUN_ROOT/run_100_batch`, the runner writes:

- `manifest.json`: model, detail, QA target, and API endpoint shape.
- `image_files.jsonl`: one row per uploaded image, including `file_id` and
  upload metadata.
- `batch_input.jsonl`: exact request lines submitted to Batch.
- `batch_input_file.json`: uploaded batch-file response.
- `batch.json`: created batch response.
- `batch_status.jsonl`: polling history.
- `batch_status.json`: latest batch status.
- `batch_output.jsonl`: completed batch output file, when available.
- `batch_error.jsonl`: batch error file, when available.
- `captions.jsonl`: parsed caption plus generated QA rows.
- `results.jsonl`: per-image status, usage, and QA-count diagnostics.
- `summary.json`: aggregate counts.

These files are safe to keep under ignored run-artifact directories. They do
not include the API key.

## Command

```bash
.venv-macos/bin/python tools/run_openai_caption_batch_smoke.py \
  --cases-json "$RUN_ROOT/cases_100.json" \
  --dataset-root "$RUN_ROOT/input_dataset" \
  --request-json "$RUN_ROOT/request_template.json" \
  --output-dir "$RUN_ROOT/run_100_batch" \
  --api-key-file openAI_API_KEY_DoNotCommit \
  --model gpt-5.5 \
  --reasoning-effort high \
  --image-detail original \
  --qa-count 8 \
  --max-boxes 120 \
  --max-output-tokens 3200 \
  --upload-workers 8 \
  --poll-seconds 30 \
  --wait-seconds 1800
```

If the command exits before the batch is complete, rerun the same command. The
runner will reuse uploaded vision files and the existing batch id, then continue
polling and collecting outputs when available.
