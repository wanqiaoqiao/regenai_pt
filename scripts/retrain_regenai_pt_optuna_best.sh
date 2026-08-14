#!/usr/bin/env bash
set -Eeuo pipefail

# Retrain the combined round-1/round-2 RegenAI-PT model with the best
# hyperparameters selected by the Optuna study.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REGISTRY_DIR="${REGISTRY_DIR:-state}"
DATASET_ID="${DATASET_ID:-ds_6c6ecc56d6b6}"
OUTPUT_DIR="${1:-${OUTPUT_DIR:-outputs/regenai_pt_optuna_best_retrain}}"
MAX_EPOCHS="${MAX_EPOCHS:-120}"
DEVICE="${DEVICE:-auto}"
RANDOM_SEED="${RANDOM_SEED:-0}"
CONTROL_TREATMENT="${CONTROL_TREATMENT:-control}"
DE_LOSS_WEIGHT="${DE_LOSS_WEIGHT:-0.0}"
DELTA_LOSS_WEIGHT="${DELTA_LOSS_WEIGHT:-0.0}"
DELTA_CONTEXT_KEYS="${DELTA_CONTEXT_KEYS:-iPSC_line,round1_treatment}"
DURATION_REGULARIZATION_WEIGHT="${DURATION_REGULARIZATION_WEIGHT:-0.1}"

IPSC_TWIN="$ROOT_DIR/.venv/bin/ipsc-twin"
if [[ ! -x "$IPSC_TWIN" ]]; then
  echo "Error: $IPSC_TWIN was not found or is not executable." >&2
  echo "Create/install the Production environment before running this script." >&2
  exit 1
fi

if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  if [[ "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "Error: output directory is not empty: $OUTPUT_DIR" >&2
    echo "Choose a new directory as the first argument, or set ALLOW_OVERWRITE=1." >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/training.log"

cat <<SETTINGS
Starting RegenAI-PT combined-round retraining
  dataset_id:              $DATASET_ID
  registry_dir:            $REGISTRY_DIR
  output_dir:              $OUTPUT_DIR
  max_epochs:              $MAX_EPOCHS
  device:                  $DEVICE
  random_seed:             $RANDOM_SEED
  batch_size:              128
  n_latent:                128
  n_hidden:                512
  n_layers:                2
  dropout:                 0.12303616213380454
  learning_rate:           8.409303267239007e-05
  weight_decay:            2.3345864076016238e-06
  warmup_epochs:           7
  ramp_epochs:             35
  max_adversarial_weight:  0.002225770634981146
  de_loss_weight:          $DE_LOSS_WEIGHT
  delta_loss_weight:       $DELTA_LOSS_WEIGHT
  delta_context_keys:      $DELTA_CONTEXT_KEYS
  duration_reg_weight:     $DURATION_REGULARIZATION_WEIGHT
SETTINGS

export PYTHONUNBUFFERED=1

"$IPSC_TWIN" train \
  --registry-dir "$REGISTRY_DIR" \
  --dataset-id "$DATASET_ID" \
  --model-type regenai_pt \
  --treatment-mode combined_rounds \
  --output-dir "$OUTPUT_DIR" \
  --input-layer log_normalized \
  --covariate-keys iPSC_line,batch,round,time_point \
  --max-epochs "$MAX_EPOCHS" \
  --batch-size 128 \
  --n-latent 128 \
  --n-hidden 512 \
  --n-layers 2 \
  --dropout 0.12303616213380454 \
  --learning-rate 8.409303267239007e-05 \
  --weight-decay 2.3345864076016238e-06 \
  --warmup-epochs 7 \
  --ramp-epochs 35 \
  --max-adversarial-weight 0.002225770634981146 \
  --lr-scheduler-factor 0.5 \
  --lr-scheduler-patience 5 \
  --early-stopping-patience 15 \
  --gradient-clip-norm 5.0 \
  --n-de-genes 100 \
  --de-loss-weight "$DE_LOSS_WEIGHT" \
  --delta-loss-weight "$DELTA_LOSS_WEIGHT" \
  --delta-context-keys "$DELTA_CONTEXT_KEYS" \
  --duration-regularization-weight "$DURATION_REGULARIZATION_WEIGHT" \
  --cell-type-key time_point \
  --control-treatment "$CONTROL_TREATMENT" \
  --random-seed "$RANDOM_SEED" \
  --device "$DEVICE" \
  2>&1 | tee "$LOG_FILE"

printf '\nTraining completed. Artifacts:\n'
find "$OUTPUT_DIR" -maxdepth 1 -type f -print | sort
