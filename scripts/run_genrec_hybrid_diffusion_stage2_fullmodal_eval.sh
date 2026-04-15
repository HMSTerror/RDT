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
: "${PRINT_EVERY:=20}"
: "${EVAL_SEED:=42}"
: "${LOG_DIR:=logs/genrec_stage2_fullmodal_eval}"

CHECKPOINT="${CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CONFIG_PATH="${CONFIG_PATH}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
MIXED_PRECISION="${MIXED_PRECISION}" \
SPLIT="${SPLIT}" \
BATCH_SIZE="${BATCH_SIZE}" \
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
TOPK="${TOPK}" \
GROUP_STRATEGY="${GROUP_STRATEGY}" \
FREQUENCY_SOURCE_SPLIT="${FREQUENCY_SOURCE_SPLIT}" \
EXCLUDE_HISTORY_ITEMS="${EXCLUDE_HISTORY_ITEMS}" \
PRINT_EVERY="${PRINT_EVERY}" \
EVAL_SEED="${EVAL_SEED}" \
LOG_DIR="${LOG_DIR}" \
bash scripts/run_genrec_hybrid_diffusion_eval.sh
