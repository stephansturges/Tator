# Deep Review Report (CPU-First)

Date: 2026-02-27
Scope: Exhaustive review of recent backend/UI/tooling hardening wave, with GPU-dependent runtime checks explicitly deferred.

## Executive Summary

- CPU-side review completed across API routes, archive import/export paths, dataset lifecycle, calibration/prepass wiring, and UI/API contract surface.
- No new P0/P1 correctness or security defects were found in this pass after fixes already landed.
- One low-risk quality fix was applied in this pass: Ruff config migration in `pyproject.toml` to remove deprecated top-level lint settings warnings.
- Remaining unvalidated items are GPU-runtime behaviors tracked in `docs/gpu_deferred_validation_queue.md`.

Merge-readiness (CPU-only): **Ready with deferments**.

## Review Method

1. Route and contract integrity checks.
2. End-to-end path inspection for high-risk flows:
   - dataset upload/download/delete
   - agent recipe/cascade import/export
   - clip classifier/labelmap download/delete/rename
   - prepass/calibration config hashing and canonicalization
3. Static correctness gates.
4. Full unit/regression test run.

## Validation Gates Executed

All commands were run successfully in `.venv`:

- `python tools/run_ui_endpoint_map_check.py`
  - result: `ui_paths=128`, `missing=[]`
- `python tools/run_ui_endpoint_method_check.py`
  - result: `ui_fetches=222`, `failures=[]`
- `python tools/run_ui_contract_tests.py`
  - result: `tested=86`, `failures={}`
- `ruff check . --select F821,B904,B905,B023,B007,E722`
  - result: pass
- `python -m compileall -q app api services tools utils models localinferenceapi.py`
  - result: pass
- `pytest -q tests`
  - result: `155 passed`

## Findings

### P0 (Blocking)

- None in this review pass.

### P1 (High)

- None in this review pass.

### P2 (Medium / Technical Debt)

1. `models/schemas.py` still uses many Pydantic V1 `@root_validator` declarations.
   - Current impact: non-blocking deprecation warnings only.
   - Risk: upgrade friction for future Pydantic major version moves.
   - Recommendation: migrate to `@model_validator` in a dedicated compatibility PR with focused schema tests.

2. GPU-runtime validations are still deferred while compute is occupied.
   - Tracked items: detector/SAM/prepass/calibration/training/agent/segmentation GPU smoke checks.
   - Source of truth: `docs/gpu_deferred_validation_queue.md`.

## Changes Applied During This Review Pass

- `pyproject.toml`
  - migrated Ruff settings from deprecated top-level keys to:
    - `[tool.ruff.lint]`
    - `[tool.ruff.lint.isort]`
  - outcome: lint warning removed; behavior unchanged.

## Flow Coverage Snapshot

Current matrix is in `docs/flow_audit_matrix.md`:

- CPU-clean: system/runtime, datasets, clip registry downloads, prepass/calibration CPU paths, agent import/export CPU paths, tooling checks, API route integrity, UI contract/method map.
- Deferred GPU validation: inference, GPU-heavy prepass/calibration lifecycle, training lifecycle smoke, GPU-backed export/build paths.

## Residual Risks

- GPU-only runtime regressions cannot be excluded until deferred queue is executed.
- Schema deprecation warnings should be addressed before major dependency upgrades.

## Next Recommended Actions

1. Execute the deferred GPU queue when GPUs are available and update both:
   - `docs/gpu_deferred_validation_queue.md`
   - `docs/flow_audit_matrix.md`
2. Run a dedicated schema modernization pass (`root_validator` -> `model_validator`) with regression coverage.
3. After GPU closure, publish final "all flows validated" audit summary.
