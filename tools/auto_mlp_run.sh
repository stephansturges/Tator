#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/steph/Tator"
LOG_PATH="$ROOT_DIR/logs/auto_mlp_run.log"
PREPASS_JSONL="${PREPASS_JSONL:-$ROOT_DIR/qwen_prepass_benchmark_2000img_prepass_full_v3.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/qwen_prepass_benchmark_2000img_prepass_full_v3.summary.json}"
FEATURES="${FEATURES:-$ROOT_DIR/uploads/ensemble_features_2000img_v4.npz}"
LABELED="${LABELED:-$ROOT_DIR/uploads/ensemble_features_2000img_v4_iou05.npz}"
MODEL_DIR="$ROOT_DIR/uploads/ensemble_mlp_runs"
TARGET_FP_RATIO="${TARGET_FP_RATIO:-0.1}"
RELAX_FP_RATIO="${RELAX_FP_RATIO:-0.2}"
CALIBRATE_OPTIMIZE="${CALIBRATE_OPTIMIZE:-f1}"
LABEL_IOU="${LABEL_IOU:-0.5}"
EVAL_IOU="${EVAL_IOU:-0.5}"
DEDUPE_IOU="${DEDUPE_IOU:-0.1}"
SCORELESS_IOU="${SCORELESS_IOU:-0.0}"
SUPPORT_IOU="${SUPPORT_IOU:-0.5}"
SKIP_FEATURES="${SKIP_FEATURES:-0}"
SKIP_LABELING="${SKIP_LABELING:-0}"
SELECT_METRIC="${SELECT_METRIC:-f1}"

mkdir -p "$MODEL_DIR"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_PATH"
}

SLEEP_SECS="${SLEEP_SECS:-10800}"
if [[ "$SLEEP_SECS" -gt 0 ]]; then
  log "auto_mlp_run: sleep ${SLEEP_SECS}s before checking prepass status"
  sleep "$SLEEP_SECS"
else
  log "auto_mlp_run: sleep skipped"
fi

log "auto_mlp_run: checking for prepass completion"
while true; do
  if [[ -f "$SUMMARY_JSON" ]]; then
    log "auto_mlp_run: summary found at $SUMMARY_JSON"
    break
  fi
  if ! pgrep -af "run_qwen_prepass_benchmark.sh --count 2000" >/dev/null; then
    log "auto_mlp_run: benchmark process not found; waiting for summary file"
  else
    log "auto_mlp_run: benchmark still running"
  fi
  sleep 600
done

source "$ROOT_DIR/.venv/bin/activate"
if [[ "$SKIP_FEATURES" -eq 1 && -f "$FEATURES" ]]; then
  log "auto_mlp_run: skipping feature build (existing $FEATURES)"
else
  log "auto_mlp_run: building ensemble features"
  python "$ROOT_DIR/tools/build_ensemble_features.py" \
    --input "$PREPASS_JSONL" \
    --dataset qwen_dataset \
    --output "$FEATURES" \
    --classifier-id "$ROOT_DIR/uploads/classifiers/DinoV3_best_model_large.pkl" \
    --min-crop-size 4 \
    --support-iou "$SUPPORT_IOU" \
    --device cuda | tee -a "$LOG_PATH"
fi

if [[ "$SKIP_LABELING" -eq 1 && -f "$LABELED" ]]; then
  log "auto_mlp_run: skipping labeling (existing $LABELED)"
else
  log "auto_mlp_run: labeling candidates (IoU >= ${LABEL_IOU})"
  python "$ROOT_DIR/tools/label_candidates_iou90.py" \
    --input "$FEATURES" \
    --dataset qwen_dataset \
    --output "$LABELED" \
    --iou "$LABEL_IOU" | tee -a "$LOG_PATH"
fi

log "auto_mlp_run: training MLP sweep"

HIDDEN_LIST=(
  "256,128"
  "512,256"
  "512,256,128"
  "1024,512"
)
DROPOUT_LIST=(0.0 0.1)
LR_LIST=(0.002 0.001 0.0005)
WEIGHT_DECAY_LIST=(0.0001 0.00001)
EPOCHS_LIST=(20 40)
SEED_LIST=(42 1337 2025)
SCHEDULER_LIST=("none" "cosine")
MIN_LR_LIST=(0.00001)
STEP_SIZE_LIST=(10)
GAMMA_LIST=(0.5)
LOSS_LIST=("bce" "focal" "asym_focal" "tversky" "focal_tversky")
FOCAL_GAMMA_LIST=(2.0)
FOCAL_ALPHA_LIST=(-1.0)
ASYM_GAMMA_POS_LIST=(1.0)
ASYM_GAMMA_NEG_LIST=(4.0)
TVERSKY_ALPHA_LIST=(0.7)
TVERSKY_BETA_LIST=(0.3)
FOCAL_TVERSKY_GAMMA_LIST=(1.0)
CLASS_BALANCE_LIST=("global" "per_class")
NEG_WEIGHT_MODE_LIST=("sqrt")
SAMPLER_LIST=("none" "weighted")
BATCH_BALANCE_LIST=("none")
POS_FRACTION_LIST=(0.5)
POS_THRESHOLD_LIST=(0.5)
BATCH_SIZE_LIST=(1024)
GRAD_ACCUM_LIST=(1)
EARLY_STOP_LIST=(5)
TARGET_MODE_LIST=("hard")

if [[ -n "${HIDDEN_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a HIDDEN_LIST <<< "$HIDDEN_LIST_OVERRIDE"; fi
if [[ -n "${DROPOUT_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a DROPOUT_LIST <<< "$DROPOUT_LIST_OVERRIDE"; fi
if [[ -n "${LR_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a LR_LIST <<< "$LR_LIST_OVERRIDE"; fi
if [[ -n "${WEIGHT_DECAY_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a WEIGHT_DECAY_LIST <<< "$WEIGHT_DECAY_LIST_OVERRIDE"; fi
if [[ -n "${EPOCHS_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a EPOCHS_LIST <<< "$EPOCHS_LIST_OVERRIDE"; fi
if [[ -n "${SEED_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a SEED_LIST <<< "$SEED_LIST_OVERRIDE"; fi
if [[ -n "${SCHEDULER_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a SCHEDULER_LIST <<< "$SCHEDULER_LIST_OVERRIDE"; fi
if [[ -n "${MIN_LR_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a MIN_LR_LIST <<< "$MIN_LR_LIST_OVERRIDE"; fi
if [[ -n "${STEP_SIZE_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a STEP_SIZE_LIST <<< "$STEP_SIZE_LIST_OVERRIDE"; fi
if [[ -n "${GAMMA_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a GAMMA_LIST <<< "$GAMMA_LIST_OVERRIDE"; fi
if [[ -n "${LOSS_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a LOSS_LIST <<< "$LOSS_LIST_OVERRIDE"; fi
if [[ -n "${FOCAL_GAMMA_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a FOCAL_GAMMA_LIST <<< "$FOCAL_GAMMA_LIST_OVERRIDE"; fi
if [[ -n "${FOCAL_ALPHA_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a FOCAL_ALPHA_LIST <<< "$FOCAL_ALPHA_LIST_OVERRIDE"; fi
if [[ -n "${ASYM_GAMMA_POS_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a ASYM_GAMMA_POS_LIST <<< "$ASYM_GAMMA_POS_LIST_OVERRIDE"; fi
if [[ -n "${ASYM_GAMMA_NEG_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a ASYM_GAMMA_NEG_LIST <<< "$ASYM_GAMMA_NEG_LIST_OVERRIDE"; fi
if [[ -n "${TVERSKY_ALPHA_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a TVERSKY_ALPHA_LIST <<< "$TVERSKY_ALPHA_LIST_OVERRIDE"; fi
if [[ -n "${TVERSKY_BETA_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a TVERSKY_BETA_LIST <<< "$TVERSKY_BETA_LIST_OVERRIDE"; fi
if [[ -n "${FOCAL_TVERSKY_GAMMA_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a FOCAL_TVERSKY_GAMMA_LIST <<< "$FOCAL_TVERSKY_GAMMA_LIST_OVERRIDE"; fi
if [[ -n "${CLASS_BALANCE_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a CLASS_BALANCE_LIST <<< "$CLASS_BALANCE_LIST_OVERRIDE"; fi
if [[ -n "${NEG_WEIGHT_MODE_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a NEG_WEIGHT_MODE_LIST <<< "$NEG_WEIGHT_MODE_LIST_OVERRIDE"; fi
if [[ -n "${SAMPLER_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a SAMPLER_LIST <<< "$SAMPLER_LIST_OVERRIDE"; fi
if [[ -n "${BATCH_BALANCE_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a BATCH_BALANCE_LIST <<< "$BATCH_BALANCE_LIST_OVERRIDE"; fi
if [[ -n "${POS_FRACTION_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a POS_FRACTION_LIST <<< "$POS_FRACTION_LIST_OVERRIDE"; fi
if [[ -n "${POS_THRESHOLD_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a POS_THRESHOLD_LIST <<< "$POS_THRESHOLD_LIST_OVERRIDE"; fi
if [[ -n "${BATCH_SIZE_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a BATCH_SIZE_LIST <<< "$BATCH_SIZE_LIST_OVERRIDE"; fi
if [[ -n "${GRAD_ACCUM_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a GRAD_ACCUM_LIST <<< "$GRAD_ACCUM_LIST_OVERRIDE"; fi
if [[ -n "${EARLY_STOP_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a EARLY_STOP_LIST <<< "$EARLY_STOP_LIST_OVERRIDE"; fi
if [[ -n "${TARGET_MODE_LIST_OVERRIDE:-}" ]]; then IFS=';' read -r -a TARGET_MODE_LIST <<< "$TARGET_MODE_LIST_OVERRIDE"; fi

needs_iou=0
for mode in "${TARGET_MODE_LIST[@]}"; do
  if [[ "$mode" == "iou" ]]; then
    needs_iou=1
  fi
done
if [[ "$needs_iou" -eq 1 && -f "$LABELED" ]]; then
  has_iou=$(python - <<PY
import numpy as np
data = np.load("$LABELED", allow_pickle=True)
print(1 if "y_iou" in data else 0)
PY
)
  if [[ "$has_iou" -eq 0 ]]; then
    log "auto_mlp_run: labeled data missing y_iou; forcing re-label"
    SKIP_LABELING=0
  fi
fi

BEST_SCORE=0
BEST_MODEL=""

for hidden in "${HIDDEN_LIST[@]}"; do
  for dropout in "${DROPOUT_LIST[@]}"; do
    for lr in "${LR_LIST[@]}"; do
      for weight_decay in "${WEIGHT_DECAY_LIST[@]}"; do
        for epochs in "${EPOCHS_LIST[@]}"; do
          for seed in "${SEED_LIST[@]}"; do
            for scheduler in "${SCHEDULER_LIST[@]}"; do
              for min_lr in "${MIN_LR_LIST[@]}"; do
                for step_size in "${STEP_SIZE_LIST[@]}"; do
                  for gamma in "${GAMMA_LIST[@]}"; do
                    for loss in "${LOSS_LIST[@]}"; do
                      for focal_gamma in "${FOCAL_GAMMA_LIST[@]}"; do
                        for focal_alpha in "${FOCAL_ALPHA_LIST[@]}"; do
                          for asym_gamma_pos in "${ASYM_GAMMA_POS_LIST[@]}"; do
                            for asym_gamma_neg in "${ASYM_GAMMA_NEG_LIST[@]}"; do
                              for tversky_alpha in "${TVERSKY_ALPHA_LIST[@]}"; do
                                for tversky_beta in "${TVERSKY_BETA_LIST[@]}"; do
                                  for focal_tversky_gamma in "${FOCAL_TVERSKY_GAMMA_LIST[@]}"; do
                                    for class_balance in "${CLASS_BALANCE_LIST[@]}"; do
                                      for neg_weight_mode in "${NEG_WEIGHT_MODE_LIST[@]}"; do
                                        for sampler in "${SAMPLER_LIST[@]}"; do
                                          for batch_balance in "${BATCH_BALANCE_LIST[@]}"; do
                                            for pos_fraction in "${POS_FRACTION_LIST[@]}"; do
                                              for pos_threshold in "${POS_THRESHOLD_LIST[@]}"; do
                                                for batch_size in "${BATCH_SIZE_LIST[@]}"; do
                                                  for grad_accum in "${GRAD_ACCUM_LIST[@]}"; do
                                                    for early_stop in "${EARLY_STOP_LIST[@]}"; do
                                                      for target_mode in "${TARGET_MODE_LIST[@]}"; do
          stamp=$(date '+%Y%m%d_%H%M%S')
          dropout_tag="${dropout//./p}"
          lr_tag="${lr//./p}"
          wd_tag="${weight_decay//./p}"
          sched_tag="${scheduler}"
          minlr_tag="${min_lr//./p}"
          loss_tag="${loss}"
          ta_tag="${tversky_alpha//./p}"
          tb_tag="${tversky_beta//./p}"
          ftg_tag="${focal_tversky_gamma//./p}"
          cb_tag="${class_balance}"
          sampler_tag="${sampler}"
          bb_tag="${batch_balance}"
          pf_tag="${pos_fraction//./p}"
          pt_tag="${pos_threshold//./p}"
          tm_tag="${target_mode}"
          prefix="$MODEL_DIR/ensemble_mlp_${stamp}_h${hidden//,/-}_d${dropout_tag}_lr${lr_tag}_wd${wd_tag}_e${epochs}_s${seed}_${sched_tag}_minlr${minlr_tag}_${loss_tag}_ta${ta_tag}_tb${tb_tag}_ftg${ftg_tag}_cb${cb_tag}_smpl${sampler_tag}_bb${bb_tag}_pf${pf_tag}_pt${pt_tag}_tm${tm_tag}_bs${batch_size}_ga${grad_accum}_es${early_stop}"
          log "auto_mlp_run: train hidden=$hidden dropout=$dropout lr=$lr wd=$weight_decay epochs=$epochs seed=$seed scheduler=$scheduler loss=$loss class_balance=$class_balance sampler=$sampler batch_balance=$batch_balance pos_fraction=$pos_fraction pos_threshold=$pos_threshold target_mode=$target_mode batch=$batch_size output=$prefix"
          python "$ROOT_DIR/tools/train_ensemble_mlp.py" \
            --input "$LABELED" \
            --output "$prefix" \
            --hidden "$hidden" \
            --dropout "$dropout" \
            --epochs "$epochs" \
            --lr "$lr" \
            --weight-decay "$weight_decay" \
            --seed "$seed" \
            --scheduler "$scheduler" \
            --min-lr "$min_lr" \
            --step-size "$step_size" \
            --gamma "$gamma" \
            --loss "$loss" \
            --focal-gamma "$focal_gamma" \
            --focal-alpha "$focal_alpha" \
            --asym-gamma-pos "$asym_gamma_pos" \
            --asym-gamma-neg "$asym_gamma_neg" \
            --tversky-alpha "$tversky_alpha" \
            --tversky-beta "$tversky_beta" \
            --focal-tversky-gamma "$focal_tversky_gamma" \
            --class-balance "$class_balance" \
            --neg-weight-mode "$neg_weight_mode" \
            --sampler "$sampler" \
            --batch-balance "$batch_balance" \
            --pos-fraction "$pos_fraction" \
            --pos-threshold "$pos_threshold" \
            --batch-size "$batch_size" \
            --grad-accum "$grad_accum" \
            --early-stop-patience "$early_stop" \
            --target-mode "$target_mode" \
            --device cuda | tee -a "$LOG_PATH"

          model_path="${prefix}.pt"
          meta_path="${prefix}.meta.json"
          if [[ ! -f "$model_path" || ! -f "$meta_path" ]]; then
            log "auto_mlp_run: missing model/meta for $prefix"
            continue
          fi

          log "auto_mlp_run: calibrating per-class thresholds for $prefix"
          python "$ROOT_DIR/tools/calibrate_ensemble_threshold.py" \
            --model "$model_path" \
            --data "$LABELED" \
            --meta "$meta_path" \
            --target-fp-ratio "$TARGET_FP_RATIO" \
            --steps 200 \
            --per-class \
            --optimize "$CALIBRATE_OPTIMIZE" | tee -a "$LOG_PATH"
          log "auto_mlp_run: relaxing thresholds (fp_ratio_cap=$RELAX_FP_RATIO) for $prefix"
          python "$ROOT_DIR/tools/relax_ensemble_thresholds.py" \
            --model "$model_path" \
            --data "$LABELED" \
            --meta "$meta_path" \
            --fp-ratio-cap "$RELAX_FP_RATIO" | tee -a "$LOG_PATH"

          eval_out="${prefix}.eval.json"
          python "$ROOT_DIR/tools/eval_ensemble_mlp_dedupe.py" \
            --model "$model_path" \
            --meta "$meta_path" \
            --data "$LABELED" \
            --dataset qwen_dataset \
            --eval-iou "$EVAL_IOU" \
            --dedupe-iou "$DEDUPE_IOU" \
            --scoreless-iou "$SCORELESS_IOU" \
            > "$eval_out"
          metrics=$(cat "$eval_out")
          echo "$metrics" | tee -a "$LOG_PATH"

          f1=$(python - <<PY
import json
with open("$eval_out", "r", encoding="utf-8") as handle:
    m=json.load(handle)
print(m.get("f1", 0.0))
PY
)
          recall=$(python - <<PY
import json
with open("$eval_out", "r", encoding="utf-8") as handle:
    m=json.load(handle)
print(m.get("recall", 0.0))
PY
)
          fp=$(python - <<PY
import json
with open("$eval_out", "r", encoding="utf-8") as handle:
    m=json.load(handle)
print(m.get("fp", 0.0))
PY
)
          tp=$(python - <<PY
import json
with open("$eval_out", "r", encoding="utf-8") as handle:
    m=json.load(handle)
print(m.get("tp", 0.0))
PY
)
          fp_ratio=$(python - "$fp" "$tp" <<PY
import sys
fp=float(sys.argv[1])
tp=float(sys.argv[2])
print(fp / tp if tp else 999.0)
PY
)

          best_ok=$(python - "$fp_ratio" "$TARGET_FP_RATIO" <<PY
import sys
fp_ratio=float(sys.argv[1])
target=float(sys.argv[2])
print(1 if fp_ratio <= target else 0)
PY
)

          candidate_score=$(python - "$SELECT_METRIC" "$f1" "$recall" <<PY
import sys
metric=sys.argv[1]
f1=float(sys.argv[2])
recall=float(sys.argv[3])
if metric == "recall":
    print(recall)
else:
    print(f1)
PY
)
          best_score="$BEST_SCORE"

          if [[ "$best_ok" -eq 1 ]]; then
            better=$(python - "$candidate_score" "$best_score" <<PY
import sys
print(1 if float(sys.argv[1]) > float(sys.argv[2]) else 0)
PY
)
            if [[ "$better" -eq 1 ]]; then
              BEST_SCORE=$candidate_score
              BEST_MODEL="$prefix"
              log "auto_mlp_run: new best (fp_ratio=${fp_ratio}) ${SELECT_METRIC}=$candidate_score f1=$f1 recall=$recall model=$prefix"
            fi
          fi
                                                      done
                                                    done
                                                  done
                                                done
                                              done
                                            done
                                          done
                                        done
                                      done
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

log "auto_mlp_run: best model = $BEST_MODEL (f1=$BEST_SCORE)"
log "auto_mlp_run: completed"
