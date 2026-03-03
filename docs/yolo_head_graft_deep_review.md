# YOLO Head-Grafting Deep Review

Date (UTC): 2026-03-02
Scope: static + non-GPU runtime review of the YOLO head-graft flow across UI, API, worker orchestration, patching, and artifact export.

## Reviewed Code Paths
- UI controls and job lifecycle:
  - `ybat-master/ybat.html:1990`
  - `ybat-master/ybat.js:5654`
  - `ybat-master/ybat.js:5665`
  - `ybat-master/ybat.js:5727`
  - `ybat-master/ybat.js:5775`
  - `ybat-master/ybat.js:12570`
- API/router wiring:
  - `api/yolo_training.py:42`
  - `api/yolo.py:66`
  - `localinferenceapi.py:20311`
  - `localinferenceapi.py:20350`
  - `localinferenceapi.py:20426`
  - `localinferenceapi.py:21295`
- Head-graft runtime and model patching:
  - `localinferenceapi.py:11694`
  - `localinferenceapi.py:11733`
  - `localinferenceapi.py:17785`
  - `services/detectors.py:1096`
  - `services/detector_jobs.py:266`

## What Is Working
- End-to-end route wiring is complete for create/dry-run/list/get/cancel and bundle download.
- Worker preconditions are strong for the real run path (base run exists, detect-only, disjoint labelmaps, YOLO-ready dataset).
- Audit logging is structured (`head_graft_audit.jsonl`) and includes status transitions and major milestones.
- Merge stage has a sanity inference pass and records warnings without crashing the run.

## Findings (Ordered by Severity)

### 1) High — Forced cancellation uses async thread exception injection (unsafe)
Evidence:
- Force-stop primitive: `services/detector_jobs.py:266`
- Cancel route invoking force-stop: `localinferenceapi.py:20440`
- Worker entry lacks one outer `try/finally` guard around entire function: `localinferenceapi.py:17786`

Why this matters:
- `PyThreadState_SetAsyncExc` can interrupt arbitrary Python frames.
- Cleanup/finalization and metadata writes can be skipped depending on interruption point.
- This can leave partially written artifacts and inconsistent runtime state.

Recommendation:
- Remove hard-kill cancellation for head-graft jobs.
- Use cooperative cancellation only (`cancel_event` + callback checks) and ensure one top-level `try/finally` around the entire worker body for deterministic cleanup.

### 2) Medium — Dry-run parity gap vs actual start checks
Evidence:
- Dry-run `ok` predicate: `localinferenceapi.py:20389`
- Real-run checks for `best.pt`: `localinferenceapi.py:17839`
- Real-run checks for base variant presence: `localinferenceapi.py:17872`

Why this matters:
- Dry-run can return `ok=true` while start still fails on missing `best.pt` or missing base `variant`.
- This creates false confidence and avoidable retries.

Recommendation:
- Make dry-run include the same preconditions as the worker preflight (at minimum: base `best.pt` exists, base `variant` present).
- Keep dry-run failure payload explicit (`error_code`, `detail`) for UI display.

### 3) Medium — Head-graft patch is process-global and reused by regular YOLO inference
Evidence:
- Global patch function: `localinferenceapi.py:11733`
- Patched symbols include `parse_model`, `DetectionModel.__init__`, and `BaseModel._apply`: `localinferenceapi.py:12038`
- Inference runtime path always calls patch hook before model load: `services/detectors.py:83`
- Hook is wired into standard YOLO inference runtime: `localinferenceapi.py:12047`

Why this matters:
- Experimental head-graft behavior is not isolated from normal YOLO runtime behavior.
- Any patch incompatibility can affect inference and training paths unrelated to grafting.

Recommendation:
- Gate patch activation to head-graft-specific load contexts only, or introduce a guarded compatibility mode with explicit runtime flag and telemetry.

### 4) Medium — Bundle endpoint can return incomplete artifact zips silently
Evidence:
- Required set declared: `localinferenceapi.py:21310`
- Missing required files are skipped rather than rejected: `localinferenceapi.py:21313`

Why this matters:
- Users can download “successful” bundles lacking required files.
- Breaks reproducibility and import workflows downstream.

Recommendation:
- Validate required files exist before streaming bundle; return 412 with missing file list when incomplete.

### 5) Low — UI payload parsing drops explicit zero values
Evidence:
- `parseInt(...) || null` for head-graft numeric fields: `ybat-master/ybat.js:5670`

Why this matters:
- Explicit `0` (e.g., workers/seed) is converted to `null` and silently replaced by backend defaults.

Recommendation:
- Use explicit finite-number parsing helper (same pattern used elsewhere) instead of truthy-coalescing.

### 6) Low — No dedicated automated tests for head-graft flow
Evidence:
- No `tests/` matches for `head_graft`/`ConcatHead`.

Why this matters:
- Regression risk remains high in a complex, patch-heavy path.

Recommendation:
- Add non-GPU unit tests for:
  - dry-run parity checks,
  - yaml graft generation (`Detect + Detect + ConcatHead` shape),
  - cancel semantics (cooperative only),
  - bundle completeness validation.

## Non-GPU Validation Performed
- Syntax checks passed:
  - `python -m py_compile localinferenceapi.py api/yolo.py api/yolo_training.py services/detectors.py services/detector_jobs.py models/schemas.py`
  - `node --check ybat-master/ybat.js`
- Functional helper probe passed:
  - `_yolo_write_head_graft_yaml_impl` emits expected `Detect/Detect/ConcatHead` tail and updated `nc`.

## Priority Fix Order
1. Remove async thread-kill cancellation and add top-level worker cleanup guard.
2. Align dry-run with worker preflight checks.
3. Enforce required bundle completeness.
4. Isolate or gate global Ultralytics patch scope.
5. Add targeted head-graft unit tests.
6. Clean UI numeric parsing for explicit zero values.

## Fix Status (2026-03-02)
- Implemented: removed async force-stop cancellation path; cancellation is cooperative (`cancelling` -> worker observes `cancel_event`).
- Implemented: head-graft worker now has outer exception/finalization handling and always persists terminal run metadata.
- Implemented: dry-run now checks base `best.pt` and base `variant` parity before reporting `ok`.
- Implemented: bundle download now fails with HTTP 412 + missing-file list when required artifacts are absent.
- Implemented: runtime patching is gated to head-grafted runs only in normal YOLO inference loading.
- Implemented: UI payload parsing for head-graft numerics now preserves explicit `0` values.
- Implemented: regression coverage added in `tests/test_yolo_head_graft_flow.py`.
