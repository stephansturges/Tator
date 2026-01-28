Tools

- Labelmap reorder via Hungarian assignment
  - Run: `python tools/reorder_labelmap.py --help`

- Interactive mismatch inspector + class remapper (PyQt5)
  - Run: `python tools/detect_missclassifications.py --images_path ... --labels_path ... [--interactive]`
  - Supports auto-fix threshold, class-wide remap suggestions, undo, and partial skip-log save.

- Qwen prepass smoke test (10-image baseline)
  - Run: `bash tools/run_qwen_prepass_smoke.sh --count 10 --seed 42 --dataset qwen_dataset`

Planned consolidation
- Dataset utilities (mismatch detection, dataset checks) live in `tools/`.
- Optional: unify CLIs with Typer and package entry points.
