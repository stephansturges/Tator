# Prepass Calibration Report

- Generated (UTC): 2026-02-20T21:29:33Z
- Dataset: `qwen_dataset`
- Sweep source: `/home/steph/Tator/tmp/emb1024_calibration_20260219_161507/deep_mlp_sweep_20260220_194647/deeper_mlp_sweep_report.json`

## Phase 1 — 2000-image calibration sweep

Data sources:
- Base evals: `/home/steph/Tator/tmp/emb1024_calibration_20260219_161507/nonwindow_20c8.eval.json`, `/home/steph/Tator/tmp/emb1024_calibration_20260219_161507/window_ceab.eval.json`
- Hybrid follow-up: `/home/steph/Tator/tmp/emb1024_calibration_20260219_161507/hybrid_after_sweep_jl_d512/selected_projection_hybrid_summary.json`
- Deep MLP sweep: `/home/steph/Tator/tmp/emb1024_calibration_20260219_161507/deep_mlp_sweep_20260220_194647/deeper_mlp_sweep_report.json`

Candidate coverage upper bound (from prepass, before calibration):
- `nonwindow_20c8`: 0.9393 recall upper bound
- `window_ceab`: 0.9506 recall upper bound

Coverage preservation metric definition: `coverage_preservation = post_calibration_recall / prepass_recall_upper_bound`.

| Variant | Method | Precision | Recall | F1 | Coverage Preservation |
|---|---|---:|---:|---:|---:|
| nonwindow_20c8 | xgb_1024_baseline | 0.9278 | 0.7048 | 0.8011 | 0.7503 |
| nonwindow_20c8 | xgb_jl_d512 | 0.9335 | 0.6984 | 0.7990 | 0.7435 |
| nonwindow_20c8 | hybrid_lr_xgb_blend_jl_d512 | 0.7864 | 0.7791 | 0.7827 | 0.8294 |
| nonwindow_20c8 | hybrid_mlp_xgb_blend_jl_d512 | 0.7823 | 0.7830 | 0.7827 | 0.8336 |
| nonwindow_20c8 | deep_mlp_best:deep_b_asym@seed2025 | 0.6807 | 0.8710 | 0.7642 | 0.9273 |
| window_ceab | xgb_1024_baseline | 0.9094 | 0.6908 | 0.7852 | 0.7267 |
| window_ceab | xgb_jl_d512 | 0.9149 | 0.6877 | 0.7852 | 0.7234 |
| window_ceab | hybrid_mlp_xgb_blend_jl_d512 | 0.8835 | 0.6768 | 0.7665 | 0.7120 |
| window_ceab | hybrid_lr_xgb_blend_jl_d512 | 0.8824 | 0.6734 | 0.7639 | 0.7084 |
| window_ceab | deep_mlp_best:deep_a_asym@seed1337 | 0.6622 | 0.8320 | 0.7375 | 0.8752 |

Best discovered setting (mean F1 across available variants): `xgb_1024_baseline` (mean F1=0.7931).

## Phase 2 — 4000-image extension (same settings, +2000 images each variant)

| Variant | Split Size | Precision | Recall | F1 | Coverage UB | Coverage Preservation |
|---|---:|---:|---:|---:|---:|---:|
| nonwindow_4000 | 4000 | 0.9101 | 0.6900 | 0.7849 | 0.9393 | 0.7345 |
| window_4000 | 4000 | 0.9083 | 0.6668 | 0.7690 | 0.9506 | 0.7014 |

### 2000 vs 4000 delta (same XGB-1024 pipeline)

| Variant | F1@2000 | F1@4000 | Delta F1 | CovPres@2000 | CovPres@4000 | Delta CovPres |
|---|---:|---:|---:|---:|---:|---:|
| nonwindow_4000 | 0.8011 | 0.7849 | -0.0162 | 0.7503 | 0.7345 | -0.0158 |
| window_4000 | 0.7852 | 0.7690 | -0.0162 | 0.7267 | 0.7014 | -0.0253 |

## Artifacts

- Pipeline log: `/home/steph/Tator/tmp/emb1024_calibration_20260219_161507/post_sweep_pipeline_20260220_201926/pipeline.log`
- 4000 XGB eval nonwindow: `/home/steph/Tator/tmp/emb1024_calibration_20260219_161507/post_sweep_pipeline_20260220_201926/xgb_1024_4000/nonwindow_4000.eval.json`
- 4000 XGB eval windowed: `/home/steph/Tator/tmp/emb1024_calibration_20260219_161507/post_sweep_pipeline_20260220_201926/xgb_1024_4000/window_4000.eval.json`
- 4000 calibration job ids: nonwindow=`cal_eb01dfb9`, windowed=`cal_46b14a8f`
