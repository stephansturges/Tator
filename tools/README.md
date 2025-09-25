Tools

- Labelmap reorder via Hungarian assignment
  - Run: `python tools/reorder_labelmap.py --help`

- Interactive mismatch inspector + class remapper (PyQt5)
  - Run: `python tools/detect_missclassifications.py --images_path ... --labels_path ... [--interactive]`
  - Supports auto-fix threshold, class-wide remap suggestions, undo, and partial skip-log save.

Planned consolidation
- Dataset utilities (mismatch detection, dataset checks) live in `tools/`.
- Optional: unify CLIs with Typer and package entry points.
