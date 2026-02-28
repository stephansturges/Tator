# Fullstack Prepass/Calibration Ablation Report

## 1) Executive Summary
This report evaluates the full-stack acceptance update (source-aware policy + joint tuning + split-head support routing + SAM3-text quality head) against leave-one-out ablations on a fixed validation split.
Primary interpretation rule: compare post-calibration acceptance quality against the detector-union comparator and the constrained XGB baseline at identical IoU/eval settings.

## 2) Experiment Context
- Run: `fullstack_ablation_20260224_155306`
- Generated (UTC): `2026-02-24T15:53:06Z`
- Dataset: `qwen_dataset`
- Eval IoU: `0.5`
- Dedupe IoU: `0.75`
- Scoreless IoU: `0.0`
- Optimize: `f1`
- Target FP ratio: `0.2`
- Recall floor: `0.6`
- Gate margin vs detector-union comparator: `+0.020` F1

## 3) Results by Variant
### Variant `nonwindow`

Baseline (current constrained): P=0.8980 R=0.6969 F1=0.7848 Δvs union=+0.0145

| Scenario | Precision | Recall | F1 | Δ vs Union | Δ vs Baseline | Coverage Pres. | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_stack | 0.9011 | 0.7465 | 0.8166 | +0.0463 | +0.0318 | 0.7948 | PASS |
| no_source_policy | 0.8853 | 0.7203 | 0.7943 | +0.0241 | +0.0096 | 0.7668 | PASS |
| no_joint_tune | 0.9533 | 0.6996 | 0.8069 | +0.0367 | +0.0222 | 0.7448 | PASS |
| no_split_head_quality | 0.8933 | 0.7432 | 0.8114 | +0.0412 | +0.0266 | 0.7913 | PASS |

Ablation attribution (F1 impact vs `full_stack`):
- Removing source policy: `+0.0222`
- Removing joint threshold tune: `+0.0096`
- Removing split-head + SAM3-text quality: `+0.0052`

### Variant `window`

Baseline (current constrained): P=0.8980 R=0.6685 F1=0.7665 Δvs union=+0.0073

| Scenario | Precision | Recall | F1 | Δ vs Union | Δ vs Baseline | Coverage Pres. | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_stack | 0.9120 | 0.7217 | 0.8057 | +0.0466 | +0.0393 | 0.7592 | PASS |
| no_source_policy | 0.8563 | 0.7165 | 0.7802 | +0.0210 | +0.0137 | 0.7537 | PASS |
| no_joint_tune | 0.9556 | 0.6688 | 0.7869 | +0.0277 | +0.0204 | 0.7036 | PASS |
| no_split_head_quality | 0.9108 | 0.7103 | 0.7981 | +0.0390 | +0.0317 | 0.7472 | PASS |

Ablation attribution (F1 impact vs `full_stack`):
- Removing source policy: `+0.0256`
- Removing joint threshold tune: `+0.0189`
- Removing split-head + SAM3-text quality: `+0.0076`

## 4) How To Perceive These Updates
- Treat this as an acceptance-quality upgrade, not candidate-generation expansion; the observed gain should show up as better F1 at similar or improved coverage preservation.
- The strongest signal is `full_stack` vs `no_source_policy`: if this gap is material, SAM-heavy noise is being controlled correctly rather than suppressing useful detector-backed recall.
- `coverage_preservation` indicates whether calibration is throwing away too much prepass potential; high coverage with low F1 suggests thresholding/policy noise, while low coverage suggests over-pruning.
- Gate status (`Δ vs union` margin) should be treated as deployment guardrail: only promote recipe defaults when the gate passes consistently across both non-windowed and windowed variants.

## 5) Recommended Default Recipe Adjustments
- Keep detector windowing on (YOLO + RF-DETR), maintain IoU=0.5 evaluation policy, and preserve current dedupe IoU=0.75.
- Keep split-head-by-support enabled and keep SAM3-text quality head enabled with alpha around `0.35` unless new ablations prove otherwise.
- Keep source-aware acceptance policy enabled with per-class overrides for SAM3 text/similarity bias, SAM-only floor, and consensus IoU.
- Keep joint threshold-shift tune enabled after policy search; it improves operating-point alignment without retraining.
- Keep cross-class dedupe disabled by default; expose only as an explicit advanced override.

## 6) Improvements Hinted By This Research
- Add strict train/tune/holdout separation for policy search vs final evaluation to reduce optimism risk.
- Add confidence-gated detector support weighting (not just overlap existence) to improve split routing reliability.
- Add per-class regularization for policy search (penalize extreme class overrides) to improve transfer when dataset mix shifts.
- Add cached policy-eval memoization across scenarios to cut runtime and make larger search spaces practical.
- Add explicit variance reporting across seeds for threshold search to better quantify stability.

## 7) Operational Follow-up
- In parallel with this report, the next +2000-image incremental encodings (non-windowed and windowed) are queued to push both prepass sets from 4000 -> 6000 images before next calibration cycle.
