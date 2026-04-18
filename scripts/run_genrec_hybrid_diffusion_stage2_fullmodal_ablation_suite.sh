#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

: "${CONFIG_PATH:=configs/genrec_hybrid_diffusion_amazon_stage2_fullmodal_30ep.yaml}"
: "${OUTPUT_DIR:=checkpoints/genrec_hybrid_diffusion_amazon_stage2_fullmodal_30ep}"
: "${CHECKPOINT:=}"
: "${SPLIT:=test}"
: "${NUM_PROCESSES:=1}"
: "${MIXED_PRECISION:=no}"
: "${BATCH_SIZE:=16}"
: "${NUM_INFERENCE_STEPS:=50}"
: "${TOPK:=5,10,20}"
: "${GROUP_STRATEGY:=equal_items}"
: "${FREQUENCY_SOURCE_SPLIT:=train}"
: "${EXCLUDE_HISTORY_ITEMS:=1}"
: "${MAX_EVAL_BATCHES:=0}"
: "${PRINT_EVERY:=20}"
: "${EVAL_SEED:=42}"
: "${ABLATION_LOG_ROOT:=logs/genrec_stage2_fullmodal_ablation}"

if [[ -z "${CHECKPOINT}" ]]; then
  if [[ -f "${OUTPUT_DIR}/final/pytorch_model.bin" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/final"
  else
    latest_ckpt="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)"
    if [[ -n "${latest_ckpt}" && -f "${latest_ckpt}/pytorch_model.bin" ]]; then
      CHECKPOINT="${latest_ckpt}"
    else
      echo "No checkpoint found under ${OUTPUT_DIR}. Set CHECKPOINT manually."
      exit 1
    fi
  fi
fi

CONFIG_PATH="${CONFIG_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
MIXED_PRECISION="${MIXED_PRECISION}" \
SPLIT="${SPLIT}" \
BATCH_SIZE="${BATCH_SIZE}" \
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
TOPK="${TOPK}" \
GROUP_STRATEGY="${GROUP_STRATEGY}" \
FREQUENCY_SOURCE_SPLIT="${FREQUENCY_SOURCE_SPLIT}" \
EXCLUDE_HISTORY_ITEMS="${EXCLUDE_HISTORY_ITEMS}" \
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES}" \
PRINT_EVERY="${PRINT_EVERY}" \
EVAL_SEED="${EVAL_SEED}" \
ABLATION_LOG_ROOT="${ABLATION_LOG_ROOT}" \
bash scripts/run_genrec_hybrid_diffusion_ablation_suite.sh
