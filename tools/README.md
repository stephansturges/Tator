# Operator tools

## Start the macOS backend

```bash
tools/run_macos_backend.sh
```

Set `PORT` to use a different port:

```bash
PORT=8080 tools/run_macos_backend.sh
```

The launcher exits if the requested port is already occupied.

## Reorder a label map

```bash
python tools/reorder_labelmap.py --help
```

## Inspect class mismatches

```bash
python tools/detect_missclassifications.py \
  --images_path <images> \
  --labels_path <labels> \
  --interactive
```

The interactive inspector supports suggested remaps, undo, and partial
skip-log saves.

## Watch a calibration job

```bash
tools/watch_calibration_job.sh --base-url http://127.0.0.1:8000 <job_id>
```

Set `INTERVAL` to change the polling interval.
