#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

: "${CONFIG_PATH:=configs/genrec_hybrid_diffusion_amazon_stage2_min.yaml}"
: "${BASELINE_OUTPUT_DIR:=checkpoints/genrec_hybrid_diffusion_amazon_50k}"
: "${RESUME_CHECKPOINT:=}"
: "${OUTPUT_DIR:=checkpoints/genrec_hybrid_diffusion_amazon_stage2_min_from_old50k}"
: "${NUM_PROCESSES:=2}"
: "${MIXED_PRECISION:=bf16}"
: "${TRAIN_BATCH_SIZE:=8}"
: "${EVAL_BATCH_SIZE:=16}"
: "${MAX_TRAIN_STEPS:=60000}"
: "${SAVE_STEPS:=5000}"
: "${EVAL_STEPS:=5000}"
: "${LOGGING_STEPS:=50}"
: "${LOG_PATH:=logs/genrec_stage2_min_train.log}"

mkdir -p "$(dirname "${LOG_PATH}")"

if [[ -z "${RESUME_CHECKPOINT}" ]]; then
  if [[ -f "${BASELINE_OUTPUT_DIR}/checkpoint-50000/pytorch_model.bin" ]]; then
    RESUME_CHECKPOINT="${BASELINE_OUTPUT_DIR}/checkpoint-50000"
  elif [[ -f "${BASELINE_OUTPUT_DIR}/final/pytorch_model.bin" ]]; then
    RESUME_CHECKPOINT="${BASELINE_OUTPUT_DIR}/final"
  else
    echo "Could not find baseline checkpoint under ${BASELINE_OUTPUT_DIR}."
    exit 1
  fi
fi

echo "========== Stage2 Minimal Training =========="
echo "config_path       : ${CONFIG_PATH}"
echo "resume_checkpoint : ${RESUME_CHECKPOINT}"
echo "output_dir        : ${OUTPUT_DIR}"
echo "max_train_steps   : ${MAX_TRAIN_STEPS}"
echo "save_steps        : ${SAVE_STEPS}"
echo "eval_steps        : ${EVAL_STEPS}"
echo "log_path          : ${LOG_PATH}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  scripts/train_genrec_hybrid_diffusion.py \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume_from_checkpoint "${RESUME_CHECKPOINT}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  2>&1 | tee "${LOG_PATH}"
