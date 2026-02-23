# Hybrid Calibrator Follow-up Plan (Post-Current 1024 Run)

## Trigger
Start this plan only after the active run in `tmp/emb1024_calibration_20260219_161507/` finishes for both non-window and window variants.

## Goal
Test two alternatives to pure XGB that can better exploit dense classifier embeddings while preserving strong structured-source logic.

## Option A (Primary)
`LR_dense + XGB_struct + meta_blender`

- Dense expert model:
  - Input: `clf_emb_rp::*` + `clf_prob::*`
  - Model: regularized logistic regression
  - Output: `p_dense`
- Structured expert model:
  - Input: all non-embedding/non-prob features (source support, counts, geometry, agreement, context)
  - Model: XGB
  - Output: `p_struct`
- Blender model:
  - Input: `logit(p_dense)`, `logit(p_struct)`, small support flags (`has_yolo`, `has_rfdetr`, `has_sam3_text`, `has_sam3_similarity`)
  - Model: logistic regression
  - Output: `p_final`

## Option B (Secondary)
`MLP_dense + XGB_struct + meta_blender`

- Same as Option A, except dense expert is a small MLP.
- Keep architecture intentionally small to reduce overfit risk.

## Experimental Protocol (Strict)

- Dataset/split:
  - Same fixed validation set already used in current run: `uploads/calibration_jobs/fixed_val_qwen_dataset_2000_images.json`
- IoU and eval policy:
  - Label IoU: `0.5`
  - Eval IoU: `0.5`
  - Dedupe IoU: `0.75`
- Features:
  - Use completed 1024-d backfilled artifacts from:
    - `uploads/calibration_cache/features_backfill/20c8d44d69f51b2ffe528fb500e75672a306f67d/ensemble_features_embed1024.npz`
    - `uploads/calibration_cache/features_backfill/ceab65b2bff24d316ca5f858addaffed8abfdb11/ensemble_features_embed1024.npz`
- Thresholding:
  - Same threshold optimization settings as current XGB control.

## Metrics to Compare

- Main: Precision / Recall / F1 on fixed val split.
- Diagnostics:
  - TP, FP, FN
  - SAM-only accepted share
  - Detector-supported accepted share
  - Source-attributed acceptance mix

## Acceptance Criteria

- Option A is adopted if it improves F1 over current XGB control without unacceptable precision collapse.
- Option B is adopted only if it beats Option A or offers materially better recall at acceptable precision.

## Execution Order

1. Complete current 1024 XGB control run.
2. Run Option A on non-window and window.
3. Run Option B on non-window and window.
4. Produce one side-by-side comparison table and choose default calibrator path.

## Notes

- Keep candidate-level embedding flow unchanged (dedupe first, then crop/encode per candidate).
- Do not switch to atom-level embeddings for this experiment set.
