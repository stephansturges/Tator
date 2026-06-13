Tools

- macOS backend launcher
  - Run: `tools/run_macos_backend.sh`
  - Optional port override: `PORT=8080 tools/run_macos_backend.sh`

- Labelmap reorder via Hungarian assignment
  - Run: `python tools/reorder_labelmap.py --help`

- Interactive mismatch inspector + class remapper (PyQt5)
  - Run: `python tools/detect_missclassifications.py --images_path ... --labels_path ... [--interactive]`
  - Supports auto-fix threshold, class-wide remap suggestions, undo, and partial skip-log save.

- Qwen prepass smoke test (10-image baseline)
  - Run: `bash tools/run_qwen_prepass_smoke.sh --count 10 --seed 42 --dataset qwen_dataset`

- Class Split Qwen review benchmark
  - Run: `python tools/run_class_split_qwen_review_benchmark.py --job-id ... --source-run ... --count 100 --run-label ... --audit`
  - Use `--source-backend-tier`, `--source-decision`, `--source-disposition`, `--source-disposition-signal`, `--source-guarded-only`, and `--source-reviewable-only` to filter a prior source run before `--start/--count` slicing.
  - Analyze saved runs with `python tools/analyze_class_split_qwen_review_benchmark.py <run.json> --fail-on-unsafe`.

- Refactor validation (py_compile + Tier-0/Tier-1 fuzz)
  - Run: `BASE_URL=http://127.0.0.1:8000 SKIP_GPU=1 tools/run_refactor_validation.sh`
  - Add `RUN_UNUSED_SCAN=1` to include the unused-def scan.

- Fuzz smoke + lite (Tier-0/Tier-1)
  - Run: `BASE_URL=http://127.0.0.1:8000 SKIP_GPU=1 tools/run_fuzz_fast.sh`

- UI endpoint and contract checks
  - Endpoint method map: `python tools/run_ui_endpoint_method_check.py http://127.0.0.1:8000`
  - UI contract checks: `python tools/run_ui_contract_tests.py http://127.0.0.1:8000`
  - Playwright control coverage: `python tools/check_playwright_control_coverage.py`

- Unused-def scanner (heuristic, module-level only)
  - Run: `python tools/scan_unused_defs.py`
  - Source definitions are reported only from first-party runtime packages;
    references from tests and tools are still counted so maintained
    compatibility wrappers do not appear as dead code.
  - Add `--include-underscore` to include private helpers.
  - Increase sensitivity with `--max-uses 1` to include definitions referenced only once.

Planned consolidation
- Dataset utilities (mismatch detection, dataset checks) live in `tools/`.
- Optional: unify CLIs with Typer and package entry points.
