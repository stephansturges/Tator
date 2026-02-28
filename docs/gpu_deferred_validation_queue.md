# GPU Deferred Validation Queue

All previously deferred GPU validations were executed with strict endpoint-level coverage.

- Final closure run: `gpu_validation_20260228_130032`
- Results artifact: `tmp/gpu_validation_20260228_130032/results.json`
- Totals: **49 passed / 49 checks**

Execution states:
- `passed`: validated in closure run
- `failed`: executed but failed (none in final run)
- `rerun-passed`: failed once then passed on rerun

| Check ID | Phase | Method | Endpoint | State | Evidence |
|---|---|---|---|---|---|
| SETUP-DATASET-UPLOAD | setup | POST | `/datasets/upload` | passed | `gpu_validation_20260228_130032` |
| SETUP-CLIP-ACTIVE_MODEL | setup | GET | `/clip/active_model` | passed | `gpu_validation_20260228_130032` |
| SETUP-YOLO-ACTIVE | setup | GET | `/yolo/active` | passed | `gpu_validation_20260228_130032` |
| SETUP-RFDETR-ACTIVE | setup | GET | `/rfdetr/active` | passed | `gpu_validation_20260228_130032` |
| OBS-001 | observability | GET | `/system/gpu` | passed | `gpu_validation_20260228_130032` |
| OBS-002 | observability | GET | `/predictor_settings` | passed | `gpu_validation_20260228_130032` |
| OBS-003 | observability | GET | `/system/health_summary` | passed | `gpu_validation_20260228_130032` |
| CTRL-001 | control | POST | `/runtime/unload` | passed | `gpu_validation_20260228_130032` |
| CTRL-002 | control | POST | `/qwen/unload` | passed | `gpu_validation_20260228_130032` |
| CTRL-003 | control | POST | `/sam3/models/activate` | passed | `gpu_validation_20260228_130032` |
| CTRL-004 | control | POST | `/qwen/models/activate` | passed | `gpu_validation_20260228_130032` |
| CTRL-005 | control | POST | `/clip/active_model` | passed | `gpu_validation_20260228_130032` |
| CTRL-006 | control | POST | `/sam_preload` | passed | `gpu_validation_20260228_130032` |
| CTRL-006A | control | GET | `/sam_slots` | passed | `gpu_validation_20260228_130032` |
| CTRL-007 | control | POST | `/sam_activate_slot` | passed | `gpu_validation_20260228_130032` |
| INF-001 | inference | POST | `/predict_base64` | passed | `gpu_validation_20260228_130032` |
| INF-002 | inference | POST | `/yolo/predict_full` | passed | `gpu_validation_20260228_130032` |
| INF-003 | inference | POST | `/yolo/predict_windowed` | passed | `gpu_validation_20260228_130032` |
| INF-004 | inference | POST | `/yolo/predict_region` | passed | `gpu_validation_20260228_130032` |
| INF-005 | inference | POST | `/rfdetr/predict_full` | passed | `gpu_validation_20260228_130032` |
| INF-006 | inference | POST | `/rfdetr/predict_windowed` | passed | `gpu_validation_20260228_130032` |
| INF-007 | inference | POST | `/rfdetr/predict_region` | passed | `gpu_validation_20260228_130032` |
| INF-008 | inference | POST | `/sam3/text_prompt` | passed | `gpu_validation_20260228_130032` |
| INF-009 | inference | POST | `/sam3/text_prompt_auto` | passed | `gpu_validation_20260228_130032` |
| INF-010 | inference | POST | `/sam3/visual_prompt` | passed | `gpu_validation_20260228_130032` |
| INF-011 | inference | POST | `/sam_point` | passed | `gpu_validation_20260228_130032` |
| INF-012 | inference | POST | `/sam_point_auto` | passed | `gpu_validation_20260228_130032` |
| INF-013 | inference | POST | `/sam_bbox` | passed | `gpu_validation_20260228_130032` |
| INF-014 | inference | POST | `/sam_bbox_auto` | passed | `gpu_validation_20260228_130032` |
| INF-015 | inference | POST | `/sam_point_multi` | passed | `gpu_validation_20260228_130032` |
| INF-016 | inference | POST | `/sam_point_multi_auto` | passed | `gpu_validation_20260228_130032` |
| INF-017 | inference | POST | `/qwen/infer` | passed | `gpu_validation_20260228_130032` |
| INF-018 | inference | POST | `/qwen/caption` | passed | `gpu_validation_20260228_130032` |
| JOB-001 | jobs | POST | `/qwen/prepass` | passed | `gpu_validation_20260228_130032` |
| JOB-002 | jobs | POST | `/calibration/jobs` | passed | `gpu_validation_20260228_130032` |
| JOB-003 | jobs | POST | `/agent_mining/jobs` | passed | `gpu_validation_20260228_130032` |
| JOB-004 | jobs | POST | `/agent_mining/apply_image` | passed | `gpu_validation_20260228_130032` |
| JOB-005 | jobs | POST | `/agent_mining/apply_image_chain` | passed | `gpu_validation_20260228_130032` |
| JOB-006 | jobs | POST | `/sam3/prompt_helper/jobs` | passed | `gpu_validation_20260228_130032` |
| JOB-007 | jobs | POST | `/sam3/prompt_helper/search` | passed | `gpu_validation_20260228_130032` |
| JOB-008 | jobs | POST | `/sam3/prompt_helper/recipe` | passed | `gpu_validation_20260228_130032` |
| JOB-009 | jobs | POST | `/segmentation/build/jobs` | passed | `gpu_validation_20260228_130032` |
| JOB-010 | jobs | POST | `/yolo/train/jobs` | passed | `gpu_validation_20260228_130032` |
| JOB-011 | jobs | POST | `/yolo/head_graft/jobs` | passed | `gpu_validation_20260228_130032` |
| JOB-012 | jobs | POST | `/rfdetr/train/jobs` | passed | `gpu_validation_20260228_130032` |
| JOB-013 | jobs | POST | `/sam3/train/jobs` | passed | `gpu_validation_20260228_130032` |
| JOB-014 | jobs | POST | `/qwen/train/jobs` | passed | `gpu_validation_20260228_130032` |
| JOB-015 | jobs | POST | `/clip/train` | passed | `gpu_validation_20260228_130032` |
| JOB-015A | jobs | POST | `/clip/train/{job_id}/cancel` | passed | `gpu_validation_20260228_130032` |

## Notes
- Cleanup was run with run-scoped namespace deletion only; original datasets/models were not removed.
- Earlier validation run exposed a real bug in agent-mining CLIP-head inference wiring (fixed before final closure).
