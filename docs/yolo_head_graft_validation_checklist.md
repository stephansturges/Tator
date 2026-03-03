# YOLO Head-Graft Validation Checklist

Date (UTC): 2026-03-02
Goal: deterministic, repeatable validation of head-graft behavior after fixes.

## A. Static/Contract Checks (No GPU)
1. API route presence
- Verify:
  - `POST /yolo/head_graft/jobs`
  - `POST /yolo/head_graft/dry_run`
  - `GET /yolo/head_graft/jobs`
  - `GET /yolo/head_graft/jobs/{job_id}`
  - `POST /yolo/head_graft/jobs/{job_id}/cancel`
  - `GET /yolo/head_graft/jobs/{job_id}/bundle`

2. Syntax integrity
```bash
source .venv/bin/activate
python -m py_compile localinferenceapi.py api/yolo.py api/yolo_training.py services/detectors.py services/detector_jobs.py models/schemas.py
node --check ybat-master/ybat.js
```

3. YAML graft helper shape
- Assert generated YAML has two `Detect` heads plus terminal `ConcatHead` and `nc == base_nc + new_nc`.

## B. Dry-Run Parity Checks (No GPU)
1. Missing `best.pt` base run should fail dry-run.
2. Base run without variant should fail dry-run.
3. Overlapping base/new class names should fail dry-run with overlap list.
4. Detect-only and disjoint class setup should pass dry-run.

Expected outcome:
- Dry-run should reject exactly what start would reject in preflight.

## C. Cancellation Safety Checks
1. Start head-graft job and send cancel during head training stage.
2. Confirm cancel path is cooperative only (no async thread exception injection).
3. Confirm deterministic cleanup occurs:
- final metadata written
- runtime finalization path executed
- status transitions remain coherent (`running -> cancelling/cancelled`)

## D. Bundle Integrity Checks
1. With complete artifacts present, bundle endpoint returns zip with required files:
- `best.pt`
- `labelmap.txt`
- `head_graft_audit.jsonl`
- `run.json` (or configured meta name)
- emitted YAML files

2. If any required file is missing, endpoint should fail with precondition error and missing-file list.

## E. GPU Functional Checks (Defer/Run When GPUs Free)
1. End-to-end head-graft with disjoint class dataset.
2. Sanity inference across several images.
3. Optional ONNX export check and warning behavior.
4. Activate merged run and run standard YOLO inference endpoints.

## F. Regression Tests to Add
- `tests/test_yolo_head_graft_dry_run_parity.py`
- `tests/test_yolo_head_graft_yaml_builder.py`
- `tests/test_yolo_head_graft_bundle_integrity.py`
- `tests/test_yolo_head_graft_cancel_semantics.py`

## Acceptance Gate
- All static checks pass.
- Dry-run/start preconditions are parity-aligned.
- Cancel path no longer relies on thread exception injection.
- Bundle endpoint is strict about required artifacts.
- End-to-end GPU run completes and merged inference is valid.
