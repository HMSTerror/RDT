#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

: "${OUTPUT_DIR:=checkpoints/genrec_hybrid_diffusion_amazon_50k}"
: "${CONFIG_PATH:=configs/genrec_hybrid_diffusion_amazon.yaml}"
: "${NUM_PROCESSES:=2}"
: "${MIXED_PRECISION:=bf16}"
: "${TRAIN_BATCH_SIZE:=8}"
: "${EVAL_BATCH_SIZE:=16}"
: "${MAX_TRAIN_STEPS:=50000}"
: "${SAVE_STEPS:=10000}"
: "${EVAL_STEPS:=1000}"
: "${LOGGING_STEPS:=50}"
: "${CLEAN_OUTPUT_DIR:=1}"
: "${LOG_PATH:=logs/genrec_stage2_train_50k.log}"

mkdir -p "$(dirname "${LOG_PATH}")"

repo_abs="$(realpath -m "${ROOT_DIR}")"
output_abs="$(realpath -m "${OUTPUT_DIR}")"

if [[ "${output_abs}" != "${repo_abs}"/* ]]; then
  echo "Refuse to delete outside repo root: ${output_abs}"
  exit 1
fi

mkdir -p "${output_abs}"

if [[ "${CLEAN_OUTPUT_DIR}" == "1" ]]; then
  echo "========== Clean old checkpoints in ${output_abs} =========="
  find "${output_abs}" -maxdepth 1 -type d -name "checkpoint-*" -print -exec rm -rf {} +
  rm -rf "${output_abs}/final"
  rm -f "${output_abs}/train_run_manifest.json"
fi

echo "========== Start Hybrid GenRec training (50k / save@10k) =========="
accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  scripts/train_genrec_hybrid_diffusion.py \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  2>&1 | tee "${LOG_PATH}"

