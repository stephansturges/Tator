# Flow Audit Matrix (CPU + GPU Validation)

This matrix tracks the exhaustive flow audit progress.

Status fields:
- `cpu_status`: `pending`, `in_progress`, `cpu_clean`
- `gpu_status`: `not_required`, `deferred`, `validated`

| Flow ID | Domain | Trigger | Key Path | cpu_status | gpu_status | Findings | Notes |
|---|---|---|---|---|---|---|---|
| SYS-001 | System health summary | `GET /system/health_summary` | `api/system.py -> health_summary_fn` | cpu_clean | not_required | none | Contract path audited; CPU-safe. |
| SYS-002 | System GPU/status endpoints | `GET /system/gpu`, `GET /system/storage_check` | `api/system.py -> payload providers` | cpu_clean | not_required | none | Endpoint wiring audited; storage path CPU-safe. |
| RT-001 | Runtime unload | `POST /runtime/unload` | `api/runtime.py -> unload_all_fn` | cpu_clean | validated | none | Covered in GPU closure run `gpu_validation_20260228_130032`; see `docs/gpu_validation_closure_report.md`. |
| DS-001 | Dataset upload | `POST /datasets/upload` | `api/datasets.py -> upload_dataset_zip` | cpu_clean | not_required | none | Zip safety checks already covered by regression tests. |
| DS-002 | Dataset delete | `DELETE /datasets/{dataset_id}` | `api/datasets.py -> delete_dataset_entry` | cpu_clean | not_required | none | Root-constraint guard audited. |
| DS-003 | Dataset download export | `GET /datasets/{dataset_id}/download` | `api/datasets.py -> download_dataset_entry` | cpu_clean | not_required | fixed | Fixed transient export temp-dir leak; added regression tests. |
| DS-004 | Dataset checks + glossary/text labels | `/datasets/{id}/check`, glossary/text label endpoints | `api/datasets.py -> localinference handlers` | cpu_clean | not_required | none | Covered by dataset and upload/check regression suites in current pass. |
| CLP-001 | CLIP classifier download | `GET /clip/classifiers/download` | `api/clip_registry.py -> download_clip_classifier` | cpu_clean | not_required | fixed | Switched to `FileResponse` to avoid open-stream descriptor leak risk. |
| CLP-002 | CLIP labelmap download | `GET /clip/labelmaps/download` | `api/clip_registry.py -> download_clip_labelmap` | cpu_clean | not_required | fixed | Switched to `FileResponse` to avoid open-stream descriptor leak risk. |
| INF-001 | Detector inference | YOLO/RF-DETR infer endpoints | `api/yolo.py`, `api/rfdetr.py` -> detector services | cpu_clean | validated | none | Validated in closure run `gpu_validation_20260228_130032` (`INF-002..INF-007`); see `docs/gpu_validation_closure_report.md`. |
| INF-002 | SAM/SAM3 prompt flows | `/sam*` endpoints | `api/sam3_prompts.py` + runtime/services | cpu_clean | validated | none | Validated in closure run `gpu_validation_20260228_130032` (`INF-008..INF-016`). |
| PRE-001 | Prepass recipes + run orchestration | `/prepass/*`, `/qwen/prepass` | API -> prepass services -> cache | cpu_clean | validated | none | `POST /qwen/prepass` validated (`JOB-001`) with run-scoped dataset in closure run. |
| CAL-001 | Calibration jobs | `/calibration/jobs*` | API -> `services/calibration.py` | cpu_clean | validated | none | Lifecycle validated (`JOB-002`) with deterministic tiny dataset and IoU=0.5 defaults in payload. |
| TRN-001 | Training jobs | YOLO/RF-DETR/SAM3/Qwen/CLIP train APIs | API -> training services/tools | cpu_clean | validated | fixed | Start/cancel lifecycle validated (`JOB-010..JOB-015A`). Resolved agent-mining classifier helper signature bug affecting GPU validation path. |
| AGT-001 | Agent mining + cascade import/export | `/agent_mining/*` | API -> `services/agent_cascades.py` etc | cpu_clean | validated | fixed | `POST /agent_mining/jobs`, `apply_image`, `apply_image_chain` validated (`JOB-003..JOB-005`) after fixing classifier helper signature mismatch (`services/classifier.py`). |
| EXP-001 | Crop ZIP + segmentation build | `/crop_zip_*`, `/segmentation/build/*` | API -> build/export services | cpu_clean | validated | fixed | Segmentation GPU path validated (`JOB-009`). Crop ZIP remained CPU-clean from prior pass; no new GPU dependency introduced. |
| UI-001 | UI endpoint contract and method map | UI JS fetches vs OpenAPI | `tools/run_ui_endpoint_*` | cpu_clean | not_required | none | Current checks pass (`missing=[]`, `failures=[]`). |
| UI-002 | UI interaction/state flows | user clicks -> result | `ybat-master/ybat.js` + API routes | cpu_clean | deferred | fixed | UI contract runner tolerates slow health probes; endpoint contract/method checks and contract suite pass in CPU mode. Final cleanup pass removed stale hard DOM-id lookups for legacy SAM3 controls (`getElementById` missing set now 0). Interactive GPU-backed browser smoke remains deferred. |
| API-001 | Route registration integrity | app startup route table | `localinferenceapi.app.routes` | cpu_clean | not_required | none | Added route uniqueness guard test; verifies no duplicate method/path and route count floor. |
| TOL-001 | Tooling and evaluation scripts | CLI tools in `tools/` | script -> artifacts/results | cpu_clean | not_required | fixed | Repo-wide targeted bugbear/F821 subset cleaned; full `tests/` suite and tool-contract checks pass in CPU mode. Final cleanup smoke checks confirm support scripts compile/help and execute attribution/eval smoke runs against existing artifacts. |
