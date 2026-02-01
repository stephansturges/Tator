Tools

- Labelmap reorder via Hungarian assignment
  - Run: `python tools/reorder_labelmap.py --help`

- Interactive mismatch inspector + class remapper (PyQt5)
  - Run: `python tools/detect_missclassifications.py --images_path ... --labels_path ... [--interactive]`
  - Supports auto-fix threshold, class-wide remap suggestions, undo, and partial skip-log save.

- Qwen prepass smoke test (10-image baseline)
  - Run: `bash tools/run_qwen_prepass_smoke.sh --count 10 --seed 42 --dataset qwen_dataset`

- Refactor validation (py_compile + Tier-0/Tier-1 fuzz)
  - Run: `BASE_URL=http://127.0.0.1:8000 SKIP_GPU=1 tools/run_refactor_validation.sh`
  - Add `RUN_UNUSED_SCAN=1` to include the unused-def scan.

- Fuzz smoke + lite (Tier-0/Tier-1)
  - Run: `BASE_URL=http://127.0.0.1:8000 SKIP_GPU=1 tools/run_fuzz_fast.sh`

- Unused-def scanner (heuristic, module-level only)
  - Run: `python tools/scan_unused_defs.py` (add `--include-decorated` to include route handlers, `--include-tests` to scan tests)

Planned consolidation
- Dataset utilities (mismatch detection, dataset checks) live in `tools/`.
- Optional: unify CLIs with Typer and package entry points.
