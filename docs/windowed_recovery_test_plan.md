# Windowed Recovery Test Plan (No Re-encoding)

## Goal
Recover the expected advantage of the windowed SAM3 prepass path while preserving all candidates (no hard source caps), using only existing cached prepass artifacts.

## Frozen Inputs
- Fixed validation split: `uploads/calibration_jobs/fixed_val_qwen_dataset_2000_images.json`
- Non-window prepass cache: `uploads/calibration_cache/prepass/20c8d44d69f51b2ffe528fb500e75672a306f67d`
- Windowed prepass cache: `uploads/calibration_cache/prepass/ceab65b2bff24d316ca5f858addaffed8abfdb11`
- Baseline references:
  - `uploads/calibration_jobs/cal_recal_nonwindow_fixed/eval_output.json`
  - `uploads/calibration_jobs/cal_recal_window_fixed/eval_output.json`

## Baseline Snapshot (IoU 0.5)
- Non-windowed XGB: `P=0.9148 R=0.8803 F1=0.8972`
- Windowed XGB: `P=0.9055 R=0.8762 F1=0.8906`
- Windowed has higher candidate recall upper bound (`0.9506` vs `0.9393`) but worse final F1.

## New Levers To Test
1. Per-class SAM-only acceptance floors (soft gating, no candidate deletion).
2. Per-class detector consensus IoU for SAM corroboration.
3. Per-source, per-class logit correction after XGB score.
4. Per-class geometric prior penalty (area/aspect outlier penalty).
5. Source-agreement/entropy feature in ranking.
6. Per-class final decision thresholds.
7. Source-weighted dedupe representative scoring (detector-favoring when evidence conflicts).

## Implementation Plan

### Phase 1: Policy Layer (eval-time, no retrain)
Add a composable policy layer in `tools/eval_ensemble_xgb_dedupe.py` that applies after model probability and before final accept:
- `--policy-json <path>` for structured config.
- Policy operators:
  - `sam_only_floor_by_class`
  - `consensus_iou_by_class`
  - `logit_bias_by_source_class`
  - `geom_penalty_by_class`
  - `threshold_by_class_override`
- Emit policy diagnostics:
  - accept/reject counts by reason,
  - per-source and per-class score shift histograms.

Deliverable:
- Reproducible policy-only sweeps on fixed split for both non-window and window.

### Phase 2: Feature/Training Integration
Integrate durable signals into training so gains are learned, not only post-hoc rules:
- `tools/build_ensemble_features.py`:
  - add agreement entropy feature per cluster (`support entropy`, `detector-vs-sam agreement`),
  - add class-conditional geometry z-score features (area/aspect priors).
- `services/prepass.py`:
  - add optional source-weighted representative scoring for cluster primary selection.
- Bump calibration feature schema version.

Deliverable:
- Retrained XGB on existing prepass caches (no re-encoding) with new features.

### Phase 3: Controlled Sweep
Run the exact same split and objective for all variants:
1. Baseline (current).
2. + SAM-only floors.
3. + consensus IoU by class.
4. + source/class logit bias.
5. + geometry penalty.
6. + per-class thresholds.
7. + all policy levers combined.
8. + retrained features (agreement + geometry) without policy.
9. + retrained features + combined policy.

Also run windowed ablations with same policy stack:
- SAM text only,
- SAM similarity only,
- both.

## Success Criteria
- Primary: windowed final F1 > non-windowed baseline F1 (`0.8972`) at IoU `0.5`.
- Secondary:
  - windowed recall >= non-windowed recall (`0.8803`),
  - precision drop limited to <= `0.01`,
  - SAM-only accepted share decreases without suppressing detector-supported recall.

## Reporting Format
Single comparison table per run:
- `P/R/F1/TP/FP/FN`
- candidate_total, accepted_total
- accepted source mix (primary + attributed)
- SAM-only accepted rate
- detector-supported accepted rate
- policy reason counters

## Order of Execution
1. Implement Phase 1 policy hooks + diagnostics.
2. Run Phase 3 policy-only sweep.
3. Implement Phase 2 feature changes.
4. Run Phase 3 retrain + combined sweep.
5. Publish summary to `readme.md` and append detailed log in `docs/test_matrix.md`.
