# Archived Tator Validation Matrix (February 2026)

This file preserves the February post-refactor GPU validation scratchpad and
run log. It is not the current product status or open-issue list. Current
coverage and closure status live in:

- [Flow Audit Matrix](flow_audit_matrix.md)
- [GPU Validation Closure Report](gpu_validation_closure_report.md)
- [Backend Storage Hardening Log](backend_storage_hardening_log.md)
- the README `Update Tracking` section

Historical legend: ✅ completed | 🟡 partial | ❌ not run or not validated in
that February pass.

## System / Runtime
- ✅ `/system/gpu`
- ✅ `/system/storage_check`
- ✅ `/system/health_summary`
- ✅ `/runtime/unload` (heavy GPU load exercised with YOLO + Qwen; GPU freed)

## Dataset & Glossary Management
- ✅ `/datasets` list/upload/delete/download
- ✅ `/datasets/{id}/build/qwen`
- ✅ `/datasets/{id}/check`
- ✅ `/datasets/{id}/glossary` get/post
- ✅ `/datasets/{id}/text_labels/{image}` get/post
- ✅ `/glossaries` get/post/delete
- 🟡 End‑to‑end: upload → glossary edit → recipe uses glossary → prepass uses same glossary (prepass ran but detector labelmap mismatch; SAM3-only prepass produced 0 dets)

## Prepass Recipes
- ✅ `/prepass/recipes` list/get/create/delete/import/export
- ✅ End‑to‑end: recipe → prepass → calibration → export/import recipe (prepass + export/import ok; calibration path verified, XGB zero-metric report later traced to evaluator bug)

## Calibration
- ✅ `/calibration/jobs` start/list/get/cancel (API contract)
- 🟡 GPU calibration on cached prepass (MLP + XGB) (runs completed; XGB eval bug fixed and re-eval now non-zero; full fresh rerun still pending)
- ✅ Calibration results recorded in README evaluation table

## YOLO Inference
- ✅ `/yolo/variants`, `/yolo/runs`, `/yolo/active`
- 🟡 `/yolo/predict_region|full|windowed` (smoke only)
- ✅ Inference with trained run in prepass pipeline (detector-only prepass with SAHI succeeded; warnings None)

## YOLO Training
- ✅ `/yolo/train/jobs` start/list/get/cancel (API + serialization)
- 🟡 Completion + auto‑activate + inference with new model (job failed in prune step; manual activation required)

## YOLO Head‑Graft
- ✅ `/yolo/head_graft/*` (API contract)
- ❌ Actual graft job + sanity inference + bundle export/import (failed: yolo_labelmap_overlap)

## RF‑DETR Inference
- ✅ `/rfdetr/variants`, `/rfdetr/runs`, `/rfdetr/active`
- 🟡 `/rfdetr/predict_region|full|windowed` (smoke only)
- ✅ Full inference on real images post‑refactor

## RF‑DETR Training
- ✅ `/rfdetr/train/jobs` start/list/get/cancel (API contract)
- ❌ Full training run end‑to‑end (dataset_not_ready)

## SAM3 (Inference & Registry)
- ✅ `/sam3/text_prompt`, `/sam3/visual_prompt`, `/sam_point`, `/sam_bbox`
- ✅ `/sam3/text_prompt_auto`
- ✅ `/sam3/models` + `/sam3/models/activate`
- 🟡 Heavy‑use SAM3 in prepass (windowed + full) on GPU (ran; 0 detections)

## SAM3 Training
- ✅ `/sam3/train/jobs` + cache purge (API)
- ❌ Actual training run

## SAM3 Prompt Helper
- ✅ `/sam3/prompt_helper/*` endpoints (API)
- 🟡 Full helper job + preset save/export + integration with prepass glossary expansion (job failed; preset saved; expand 500)

## CLIP / DINO Classifier
- ✅ `/clip/backbones`, `/clip/classifiers`, `/clip/labelmaps`
- ✅ download / rename / delete endpoints
- ✅ `/clip/train` start/list/get/cancel (API)
- ❌ Actual training runs (DINOv3 + any CLIP variants)
- ❌ Verify registry after training + use in prepass pipeline

## Qwen (Caption/Infer/Prepass)
- ✅ `/qwen/models`, `/qwen/models/activate`, `/qwen/status`, `/qwen/settings`, `/qwen/unload`
- 🟡 `/qwen/caption`, `/qwen/infer`, `/qwen/prepass` (caption ok; detector-only prepass succeeded; SAM3+caption not exercised)
- 🟡 GPU caption + prepass (windowed + non‑windowed) on real images

## Qwen Training
- ✅ `/qwen/train/jobs` + cache purge (API)
- ❌ Full training run end‑to‑end

## Agent Mining
- ✅ `/agent_mining/*` endpoints (API contract)
- 🟡 Real mining job + recipe export/import + cache purge + apply_image_chain on GPU (job start failed; export/import ok; apply_image_chain 500)

## Segmentation Build
- ✅ `/segmentation/build/jobs` (API contract)
- ❌ Actual segmentation build (qwen_dataset path failed: _load_qwen_dataset_metadata_impl args missing)

## Crop Zip / File Upload
- ✅ `/crop_zip_*` endpoints (API contract)
- ✅ `/fs/upload_classifier`, `/fs/upload_labelmap`
- 🟡 End‑to‑end: upload → registry → use in prepass (upload ok; prepass failed with active_clip_model_name NameError)

## Notes
- GPU validation order should follow model availability + cluster load.
- Archived ❌/🟡 rows below are historical evidence, not current known failures.
  Revalidate against the current backend before treating any row as actionable.

**Run Log (2026-02-04)**
Test: `/runtime/unload` under heavy GPU load. Result: PASS. Details: GPU free MB dev0 before=47105.19, after load=38387.19, after unload=47105.19; YOLO detections=28; Qwen caption OK; runtime unload returned `{"status":"unloaded"}`.

Test: Dataset upload → glossary edit → recipe uses glossary → prepass uses same glossary. Result: PARTIAL. Details: Uploaded `test_upload_1770240761` (2 images). Prepass with detectors enabled failed with warnings `detector_labelmap_mismatch:yolo`; SAM3-only prepass ran but 0 detections.

Test: Prepass recipe → prepass → calibration → export/import recipe. Result: PARTIAL. Details: Prepass failed with `qwen_prepass_failed:name 'active_clip_model_name' is not defined`; export/import succeeded (`prepass_e2e_1770240926` → `4d1c1af8129f4f8592404aedb94fe004`).

Test: Calibration (MLP + XGB). Result: PARTIAL. Details: Initial runs failed with `calibration_classifier_required`. Retried with `classifier_id=DinoV3_best_model_large.pkl`: `cal_68996356` (xgb) completed with all-zero metrics; `cal_5cc4733e` (mlp) completed with precision=0.8333, recall=0.1538, f1=0.2597 (max_images=50).

Test: YOLO inference in prepass pipeline. Result: FAIL. Details: `/qwen/prepass` on `qwen_dataset` fails with NameError `active_clip_model_name` before YOLO stage.

Test: YOLO training completion + auto-activate + inference. Result: PARTIAL. Details: Job `6502285e5dd...` status=failed due to `_yolo_prune_run_dir_impl() missing keep_files`; run artifacts complete; inference works after manual activation.

Test: YOLO head-graft. Result: FAIL. Details: `yolo_labelmap_overlap` (base/new classes overlap) for job `head_graft_1770242862`.

Test: RF-DETR inference. Result: PASS. Details: `/rfdetr/predict_full` detections=34, warnings=None.

Test: RF-DETR training. Result: FAIL. Details: `rfdetr_tos_required` then `dataset_not_ready` when using prior run dataset root.

Test: SAM3 heavy-use prepass. Result: PARTIAL. Details: Windowed SAM3 text+similarity ran on `test_upload_1770240761` with 0 detections.

Test: SAM3 training. Result: FAIL. Details: `/sam3/datasets/{id}/convert` returned `sam3_dataset_type_unsupported`.

Test: SAM3 prompt helper. Result: PARTIAL. Details: Job `ph_35bc18b8` failed (`_load_sam3_dataset_metadata_impl()` missing keyword args); preset saved (`phset_bf9d77ad`); expand endpoint 500.

Test: CLIP/DINO training. Result: FAIL. Details: CLIP job `8684b3ba...` and DINO job `2b1e043a...` failed with `name '_unload_runtime' is not defined`.

Test: Qwen caption + prepass. Result: PARTIAL. Details: Caption full/windowed succeeded (caption lengths 345/313). Prepass (full+windowed) on `test_upload_1770240761` ran with 0 detections.

Test: Qwen training. Result: FAIL. Details: `/qwen/train/jobs` returned 500 Internal Server Error (no detail).

Test: Agent Mining. Result: PARTIAL. Details: Job start failed `agent_mining_clip_head_required`; recipe export/import succeeded (`ar_84796056` → `ar_a629131b`); cache purge ok; apply_image_chain returned 500.

Test: Segmentation build. Result: FAIL. Details: `test_upload_1770240761` failed `segmentation_source_metadata_missing`; `qwen_dataset` failed `_load_qwen_dataset_metadata_impl() missing keyword args`.

Test: Crop zip + file upload. Result: PARTIAL. Details: crop_zip flow OK (zip size 2448). Uploaded `test_classifier.pkl` and `test_labelmap.txt` (registry entries present). Prepass with `classifier_id=test_classifier.pkl` failed with NameError `active_clip_model_name`.

**Run Log (2026-02-05)**
Test: Full prepass pipeline (detector-only) on trained YOLO+RF-DETR with SAHI. Result: PASS. Details: `qwen_prepass` on `qwen_dataset` image `f8473fa6-13df-4685-973d-abf7dac7b825.jpg` with `enable_yolo=true`, `enable_rfdetr=true`, `enable_sam3_text=false`, `enable_sam3_similarity=false`, `prepass_caption=false`, `sahi_window_size=2048`. Detections=22, warnings=None. Trace: `uploads/qwen_prepass_traces/prepass_1770290730_e2650f56.jsonl`, full trace: `uploads/qwen_prepass_traces_full/prepass_full_1770290730_9822e55b.jsonl`.

Test: Prepass recipe → prepass → calibration → export/import recipe. Result: PASS. Details: Saved recipe `f558e93d13774fd9b7d553fc26e7b614`; prepass on `qwen_dataset` image `f8473fa6-13df-4685-973d-abf7dac7b825.jpg` succeeded (detections=22, warnings=None; trace `uploads/qwen_prepass_traces/prepass_1770300284_d46086b8.jsonl`). Exported to `uploads/prepass_recipe_exports/prepass_recipe_f558e93d13774fd9b7d553fc26e7b614_afqrif37.zip`, imported as `453c5b1c7b0549c3bf70d8cf606cdc62`. Calibration rerun separately (see below).

Test: Calibration XGB verification. Result: PASS (superseded). Details: Job `cal_b32925bc` on `qwen_dataset`, `max_images=10`, `classifier_id=DinoV3_best_model_large.pkl`, `sam3_text_synonym_budget=0`, `sahi_window_size=2048`. Completed with metrics `tp=0 fp=0 fn=0 precision=0 recall=0 f1=0` (all IoU grid entries zero). Later investigation (2026-02-08) traced this to an XGB evaluator GT-loading bug in `tools/eval_ensemble_xgb_dedupe.py`.

Test: SAM3 inference (all modes). Result: PARTIAL. Details: Using `qwen_dataset` image `f8473fa6-13df-4685-973d-abf7dac7b825.jpg` with `HF_HUB_OFFLINE=1`. `sam3_text_prompt` and `sam3_text_prompt_auto` returned 0 detections with `no_results`. `sam3_visual_prompt` returned 1 detection. `sam_point`, `sam_bbox`, `sam_point_multi` returned masks and bboxes. Auto variants (`sam_point_auto`, `sam_bbox_auto`, `sam_point_multi_auto`) returned `prediction=unknown` with `classifier_head_unavailable` (no active CLIP head).

**Run Log (2026-02-08)**
Test: Calibration XGB evaluation audit/fix (windowed prepass artifacts). Result: PASS. Details: Root cause for all-zero XGB metrics was in `tools/eval_ensemble_xgb_dedupe.py`, which expected `gt_bbox_xyxy_px` fields that are not present in ensemble meta rows, so no GT was loaded and metrics stayed zero. Updated evaluator to load YOLO labels from `uploads/clip_dataset_uploads/{dataset}_yolo` (same GT strategy as MLP eval), use dataset image sizes for box conversion, and evaluate across the validation split correctly. Re-eval results: `cal_1be1a9c0` best point `tp=2276 fp=89 fn=9783 precision=0.9624 recall=0.1887 f1=0.3156`; `cal_617f7a8b` best point `tp=2278 fp=90 fn=9781 precision=0.9620 recall=0.1889 f1=0.3158`.
